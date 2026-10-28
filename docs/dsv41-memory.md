# DeepSeek-V4.1 serving memory

How `deepseek_v41` spends memory under CED prefill: the ledger, the per-layer cache split, the 1M-token state, the
prefill that borrows decoder pools and the fully resident load. Index: [CLAUDE.md](../CLAUDE.md#docs-index); related:
[dsv41-arch](dsv41-arch.md), [engine-expert-streaming](engine-expert-streaming.md).

## The ledger (an STQ pack, `--ssd-budget-gb 96` = 103.08 GB, no vision tower; the [tower](#vision-requant) adds 0.50 GB)

| term | GB |
|---|---|
| trunk (resident STQ banks, head, the 384-expert routers as stored in bf16; the 0.37 GB embedding is read from SSD) | 4.25 |
| DSpark stages (resident) | 3.92 |
| union workspace (one layer's 384 experts at the widest layer) | 3.70 |
| bounce + selected | 0.60 |
| MLX buffer pool, decode cap (`dsv41_pool_bytes`, 1 GiB) | 1.07 |
| expert cache | 89.6 |

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
  - **Prefill** (a wider forward): the RAM-wide cap, with no cap of its own. The transient cap
    ([below](#transient-caps)) bounds the pool together with the forward's allocations, so MLX trims the pool on demand
    at it and the parked pool never passes what was billed. The decode cap is a standing term of the live headroom
    (`server.dsv41Headroom` takes it off ceiling less active, since the parked pool sits in MLX's cache, not its active
    bytes); a lower cap clears the pool. Admission (`prefillNeededAtChunk`) bills the decode cap beside every request;
    at load `expertStreamingServingBytes` leaves it out, the ledger's staging term already holds it.
  - Unloading a DeepSeek model gives the next model the process-wide cap `applyMlxCacheLimit` installed, forgets the
    raised cap and lifts the transient cap (`dsv41_borrow.unloadModel`, in the scheduler's unload and eviction paths).
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
  points) while the request runs.

<a id="transient-caps"></a>
## Transient caps

MLX allocates through a patched allocator that can refuse a buffer ([engine-mlx-gotchas](engine-mlx-gotchas.md#transient-cap)).
Every DeepSeek phase runs under a cap on the bytes MLX allocates itself beyond what it holds when the phase starts: the
cap is `limit = active - imported + (the phase's bytes + the decode pool)`, set on the inference thread by
`dsv41_borrow.enterForward` before each forward (prefill chunk, decode, DSpark verify; the draft runs under the cap of
the forward before it) and by `enterVision` around an image encode. The pool counts in it, so the allocator trims the
pool on demand and the phase's bytes plus the pool never pass the limit. A request that reaches it fails as
`TransientCapExceeded` (503) and the server keeps serving. A request's end lifts the cap. There is no flag.

The phase's bytes are the bill: `Config.forwardCap(rows, seq)` for a forward, `server.visionEncodeBill` for an image.

    prefill-width forward (> 17 rows):  (rows x dim x 480 + seq x (state/16 + worst layer)) x 5/4 + 256 MiB
    decode or verify forward (<= 17):   (seq x 16 x 4 x 3 + seq x (state/16 + worst layer)) x 5/4 + 256 MiB

The per-row 480 B a channel is the 2.3 MiB a row measured from 256 to 2048 rows, rounded up (one 363-row chunk read 2.8
MiB with its state rows); the per-context term is a decode block's scores (16 rows, 3 copies, f32) and the row stores'
growth (a sixteenth of the state, plus one layer's old rows beside the new buffer: 55 + 356 B a token); the 256 MiB is
twice the 0.12 GiB of small tensors a decode forward holds. A last chunk absorbs a tail, so a chunk bills
`chunk + 511` rows (`chunkCap`).

Measured peaks of MLX's own bytes, imported weights left out and the state's rows in (`SUSHI_DSV41_TRANSIENT_PEAK=1`,
binary 26a28040, STQ-1.75bpw, `--ssd-budget-gb 108`, DSpark, one 225k-token cold prefill, raw
`p2b-1028`), and the cap beside them (the pool allowance, 1 GiB, is added on top):

| phase | rows | context | measured peak, GiB | cap, GB |
|---|---|---|---|---|
| decode | 1 | 9k / 225k | 0.116 / 0.336 | 0.28 / 0.44 |
| DSpark draft, verify | 5, <= 17 | 225k | 0.304, 0.327 | 0.44 |
| prefill chunk | 2048 | 2k / 225k | 4.57 / 4.77 | 6.6 / 6.7 |
| prefill chunk | 256 / 1000 / 1183 | short | 0.57 / 2.25 / 2.48 | 1.0 / 3.4 / 3.9 |
| vision encode | 8463 patches (1031 tokens) | | 1.11 | 1.47 (`visionEncodeBill`) |

A cap is over the level its phase starts from, which holds the state: each cap covers a quarter over the measured peak
less the state of the context (`dsv41 forward cap` test). Extrapolated to 1M tokens: a decode or verify forward 1.06 GB,
a 2048-row chunk 7.1 GB and a tail-merged one (2559 rows) 8.7 GB. The decode slope the measurements give beyond the
state's 890 B a token is 204 B; the cap carries 603 B (x 5/4).

**What the bill is now** (`prefillNeededAtChunk`, the borrow's `prefillNeed`; the old bill was
`transientBytes x 5/4 + (8 GiB - 1 GiB)` pool growth for a prefill and `decodeTransientBytes(17) x 5/4` for a decode):

| term | context | old, GB | new, GB |
|---|---|---|---|
| borrow need, 2048 rows | 14.8k / 225k / 1M | 15.1 / 15.3 / 15.8 | 6.6 / 6.7 / 7.1 |
| chunk bill, 2048 + 511 rows | 14.8k / 225k / 1M | 16.4 / 16.6 / 17.1 | 8.1 / 8.2 / 8.7 |
| decode or verify block | 9k / 225k / 1M | 0.55 / 0.71 / 1.33 | 0.28 / 0.44 / 1.06 |
| image encode | any | unchanged: the bill already covered the measured peak by a quarter | |

The pool growth (7.5 GB) was most of the old bill and the cap replaces it: a 2048-row chunk's need falls 8.3-8.7 GB,
the rest being the measured per-row rate against the old formula. At wired 124000 with DSpark and the SSD tier on (the
`with its decoder pools lendable` test), budget 117 and no decoder pool to lend, the advertised context is 390,144
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
the load's own rule (`dsv41_vision.quantizesAtLoad`) to the shard headers. The codes are MLX's own allocations, not
sushi-owned mappings: `[dsv41-mem]` does not count them and `active_bytes` does.

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

<a id="owned-weights"></a>
## Weight memory sushi owns

The weight load builds every long-lived weight of 1 MiB or more in a mapping of its own that MLX imports without a copy
(`model.ownedRegion` and `wrapOwned`); `/props` `memory.owned_bytes` and the `[dsv41-mem]` load line report it.
- A shard tensor is read into its mapping (`readTensorArray`); the weight load refuses one MLX copied instead
  (`OwnedImportCopied`), since a copy would sit outside the ledger.
- The DSpark stage's trellis experts are nine regions (trellis, `suh`, `svh` per projection), the groups' bytes back to
  back, copied on the host; each group's source is dropped after its copy, so a stage's load peak is its storage plus
  one group (`dsv41_owned.raggedOwned`, into `Ragged.initPools`).
- f32 widenings (bf16 to f32 is the bits shifted up, as MLX's cast) and the fused-stack bank of `fuseShared` are built
  in regions; the slices into the stack stay views.
- Left in MLX: tensors under 1 MiB, the meta tables, the rope tables, Engram's q.k product and the vision tower's
  requantized codes ([above](#vision-requant)).
- The bill (`dsv41_source.sumBytes`) rounds each tensor from 1 MiB, and each DSpark (stage, projection, part) as one
  region, up to the page. MLX refuses no page-aligned mapping; it also accepts a start inside a page, so the
  `OwnedImportCopied` check is the data-pointer test, not an alignment rule.

## Plans that lost

Replayed on recorded routes, two plans were dropped. Pinning the encoder's experts resident (layers 0-19) saves prefill
fill but costs decode misses on the decoder layers, and decode reads all 40 layers: it loses once a turn decodes more
than about a fifth of its new prompt tokens. Swapping the encoder and decoder halves between phases moves ~29.5 GB a
flip (about 3.3 s at 9 GB/s) and loses at every decode length, with the swap in the foreground or the background.

## The 1M KV reserve

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

**The borrow** (`src/dsv41_borrow.zig`, `Engine.borrowLayer`/`restoreLayer`):
- Before every forward wider than 17 rows, its need, `Config.forwardCap(rows, seq)` (6.6 GB at 2048 rows),
  against the live headroom (ceiling less MLX's active bytes). Short of it, whole decoder pools go, latest layer first
  (39 down to the CED split, 20): the plan is made from the pools' bytes before anything is released, a forward the
  pools cannot cover is refused (`PrefillDoesNotFit`) with nothing released, and the live headroom is re-read after each
  release (a short count takes the next pool too).
- A release drops the layer's bank, waits for MLX to let go of each slab and unmaps it: MLX's active count falls by
  exactly the pool's bytes (tested), and the forward waits up to 1 s for 90% of them to leave the wired set.
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
