# Engine: expert streaming (`--ssd-budget-gb` / per-model `ssd_budget_gb`)

How a checkpoint whose routed experts do not fit in memory is served from SSD: the budget ledger, the per-layer LRU,
the zero-copy slab I/O and the correctness bars. Read this before touching `src/expert_stream.zig`,
`src/expert_io.zig` or `src/glm5_stream.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-exl3-experts](engine-exl3-experts.md),
[engine-memory-admission](engine-memory-admission.md), [qwen4-arch](qwen4-arch.md),
[mimo2-arch](mimo2-arch.md), [glm5-arch](glm5-arch.md), [server-lifecycle](server-lifecycle.md#settings).

## Code map

| File | Role |
|---|---|
| `src/expert_stream.zig` | `ExpertStore` spans, per-layer group-exact LRU + union bridge, zero-copy slabs, `GroupedLayer` (a rate-grouped layer's pools), `BudgetLedger` (`--ssd-budget-gb`), MTP refusal |
| `src/expert_io.zig` | SSD→Metal I/O: F_NOCACHE positioned-read `FillPool`, `PageSlab` epoch leases, verified zero-copy `importSlab` |
| `src/expert_bf16_kernels.zig` | bf16 selected-expert kernels over a slab |
| `src/glm5_stream.zig` | GLM's routed experts over the engine (BF16, FP8 or EXL3 slabs), the native teacher's capture budget |
| `src/imatrix.zig` | imatrix capture on the streamed forward |
| `src/hidden_capture.zig` | block-boundary residual capture under `kld capture` |

## What streams

One engine (`expert_stream.Engine`) streams every served model, keyed by the checkpoint's routed-expert layout, and
one plan (`scheduler.planExpertStreaming`) sizes it for `serve`/`run`, the registry's cold-load gate and `sushi kld`.
Each forward runs its own resident expert kernels on slab-local ids, so a streamed forward is bit-identical to the
resident one; the ledger bills from the stored headers, a slot sized to the widest layer rate. The trunk (and MiMo's
coarse lm_head) stays resident; with no budget a pack loads resident.

| Model | Checkpoint | Layout | `serve`, `run`, `kld compare` | `kld capture` |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | Sushi EXL3 pack | `exl3_k4` | streams | streams |
| Qwen3.8-Flash-Next | BF16 source (335 GB) | `bf16_fused` | streams; required | streams (the teacher) |
| MiMo-V2.6-Flash | Sushi EXL3 pack | `exl3_k4` beside the FP8 trunk | streams | streams |
| MiMo-V2.6-Flash | original checkpoint | `mxfp4_individual` | streams; required | streams (the teacher) |
| GLM-5.3-Flash | Sushi EXL3 pack | `exl3_k4` | streams | streams (a BF16-latent student reference) |
| GLM-5.3-Flash | BF16 source (643 GB) | `bf16_individual` | streams; required | native streamed teacher |
| GLM-5.3-Flash | FP8 release | `fp8_individual` beside its FP8 trunk | streams (hermetic proof only) | refused |

- Rate-group (`.gN`) EXL3 packs stream on Flash-Next ([below](#rate-groups)); on MiMo, and for pruned layers, they are
  refused by name (`Exl3RateGroupsStreamingUnsupported`, `Exl3RaggedStreamingUnsupported`) and serve resident. GLM
  serves no rate groups at all (`Exl3RateGroupsUnsupported`).
- Vision is off under streaming unless asked for, except DeepSeek-V4.1, which loads it by default ([below](#vision)); a streamed model without the tower serves text only.
- GLM's routed experts go through `glm5_stream.Stream`: BF16 source experts through the clamped `gather_mm`
  composite, EXL3 banks through the resident `routedExl3` dispatch; router ids become slot ids, scores, clamps and
  the FP32 reduction are untouched, and each layer completes before its slabs are reused.
- The FP8 release's experts stream as stored: e4m3 codes plus f32 block-128 scales. A layer's routed slots
  dequantize to bf16(code x scale) through `fp8_block` and run the BF16 composite; the copies and gathered codes (at
  most every expert of one layer, about 21.7 GB) are billed beside the trunk (`scheduler.fp8ExpertScratchBytes`).
  Its capture is refused: the release is a block quantization of the BF16 source, which stays the only teacher.
- The FP8 release is verified by hermetic tests only, with no live boot, KLD or speed number: the checkpoint is no
  longer on the box.
- A streamed GLM keeps no speculative state: its DFlash2 assistant stays unloaded and an explicit `--drafter` is
  refused (`GlmStreamingSpecUnsupported`).
- A streamed GLM serves the kv8 or the BF16 latent; its request bill adds the fill peak, and its planned KV counts the
  latent at the KV width plus the pooled index.
- The native BF16 teacher capture (`glm5_kld_capture`) keeps its own budget (`glm5_stream.captureBudget`): trunk,
  request reserve (at least 8 GiB), the full union slab, bounce buffers and at least one slot per MoE layer, the lazy
  trunk bounded before any tensor is evaluated, one request, chunks of at most 512.
- Its layer-major mode ([quality-kld](quality-kld.md#layer-major)) pins one layer at a time: `Stream.pin` reads all
  of a layer's experts into the union workspace once (`prepareHost` over every id) and maps router ids through it
  for every window of the batch, so a batch reads each MoE layer once (GLM BF16: 14.50 GB per layer, 608.8 GB per
  batch). Slot position does not change the gather's result, so output stays byte-identical.

<a id="vision"></a>
## Vision (`--vision`)

- **A streamed load leaves the tower off by default (DeepSeek-V4.1 excepted)** and logs what it would cost (`[vision] off under expert
  streaming; pass --vision to load the tower (+X GB, -N slots/layer)`); an image is then a 400 naming `--vision`.
  `--vision` or a per-model `vision: true` loads it (`--no-vision` > `--vision`/`vision` > default; both flags at
  once are refused). A resident load keeps its tower on by default and takes `--vision` as a no-op. DeepSeek-V4.1 is the
  exception: its streamed default is ON (`visionChoiceFor`), billed in the same `vision` term, and a budget too small for
  it is refused (`SsdBudgetBelowVision`) until raised or `--no-vision` is passed.
- **The tower is resident and billed in the ledger** (`vision` term, `resolveStreamedVision`): the expert cache
  shrinks by its weights. A budget that cannot hold it beside two slots per layer is `SsdBudgetBelowVision` (503).
- **The encode transient is not reserved.** The load proves the largest single image the tower's processor admits
  (`server.largestImageEncodeBytes`) beside the budget: `budget + planned KV + that encode <= wired limit`, else
  `SsdBudgetExceedsWiredLimit`. Every request is then billed live like a resident one (`towerFitFault`); its vision
  tokens ride the prompt's admission bill.
- The forward splices the rows the same way streamed or resident (the embedding step is shared): greedy bytes match.
- `/props settings.vision` reports `loaded`, `source` and the streamed tower and encode bytes.

| Model | Tower (stored) | Largest image (patches) | Encode bill |
|---|---:|---|---:|
| Qwen3.8-Flash-Next packs, BF16 source | 897,862,112 B (597,327,776 B billed, [requantized](qwen4-arch.md#vision-tower)) | 1,003,520 px default bound (3,920) | 1.77 GB |
| MiMo-V2.6-Flash 2.3bpw | 1,457,188,864 B | 1536² engine cap (9,216) | 2.26 GB |
| GLM-5.3-Flash 2.3/2.5bpw (affine tower) | 493,389,824 B | 8,000 merged tokens (32,000) | ~5.8 GB |
| GLM-5.3-Flash BF16 source | 1,127,254,016 B | 8,000 merged tokens (32,000) | ~5.8 GB |

## Budget

- `expert_stream.budgetLedger`, one `[expert-stream] ssd budget` boot line. `--ssd-budget-gb N` is a TOTAL resident
  target of N GiB = trunk + MTP + vision tower (under `--vision`) + the all-experts union workspace + selected slab +
  bounce + the MLX buffer pool a model caps for itself (`StoreShape.pool_bytes`, DeepSeek only; others keep the
  RAM/16 cap outside the ledger); the remainder is a uniform per-layer LRU. The ledger refuses a budget under two slots per layer, and under
  one expert of every rate group of a grouped layer (`SsdBudgetBelowResident`, `StoreShape.layer_floor_bytes`).
- **Resident**: a budget that holds every routed expert (`StoreShape.all_bytes`, set when every routed layer is
  rate-grouped) beside the trunk, MTP and tower bills no union workspace, selected slab or bounce. The engine sizes
  every pool to its whole layer, reads all of them at load (`[expert-stream] resident:` line; no 80% warm) and maps
  each expert to its slot (`LayerState.resident_map`), so a DeepSeek STQ route takes its slots on the GPU with no
  host read of the picks (`Engine.residentRoute`). Same bytes out as the streamed path (tested on the tiny STQ pack).
- Precedence: `--ssd-budget-gb` > setting > `ExpertStreamingRequired` 503 naming both.
  An explicit launch flag always beats `model-settings.json` ([server-lifecycle](server-lifecycle.md#settings)).
- The load's fit check prices serving at the per-request ladder's floor rung (512), since a request picks its own
  rung against free memory; a pinned chunk is priced as given; an explicit
  `--prefill-chunk` only lowers the floor the load proves.
- **A streamed Flash-Next bills no expert fill peak** (`scheduler.billedFillPeakBytes`); MiMo and GLM keep theirs. The
  fill peak (`fillPeakBytes`: two union workspaces, a cache layer, a batch) rode on top of a budget that already holds
  the union workspace and the bounce. M1 Max, `--ssd-budget-gb 23`, kv8, lean: a cold 8k prompt peaked 1,224 MiB over
  the loaded weights and a 45k 1,683, the resident pack's own peak, against bills of 3,270 and 3,821 that carried it.
- Admission `budget + planned KV (+ the largest image encode under --vision) <= wired limit`; the refusal names the
  `iogpu.wired_limit_mb` that would admit.
  Planned KV is the session bill per token (KV at its width plus Qwen's QSA history or GLM's pooled index) times the
  planned context. Under `--no-mtp` the head is not loaded at all.
- An imatrix capture's accumulators live in GPU headroom that admission reads: budgets for capture runs drop
  (MiMo 100 → 94 GB; Flash-Next 96 → 80 GiB no longer admits higher under current bills).

## Cache policy

- `GroupCache`: plain per-layer LRU, prefill misses at MRU, every HIT of a route touched before any admit, surplus
  misses fall to the union workspace. Batched decode rides the union path.
- **MTP is on by default on a Sushi EXL3 Qwen pack** (`streamedMtpHeadSupported`): the head and its own routed experts
  load resident (`ModelConfig.stream_mtp_head`, billed in the ledger and its KV in the session bill), verify rows take
  the streamed decode path, and the PLE window rolls back like a resident verify. `--no-mtp` or `"mtp": false` loads
  without it (`[mtp] off`). Any other streamed model or layout refuses an explicit `--mtp`
  (`ExpertStreamingMtpUnsupported`), drops a settings `mtp: true` with a warning and resolves the default off;
  grouped multi-request verify stays resident-only; `enable_mtp:true` without a loaded head is a named 400.
- **DFlash2 on a streamed Flash-Next**: a drafter named with `--drafter` loads resident and its bytes join the trunk in the ledger, so the expert
  cache is cut from the budget that remains (`planExpertStreamingWith`; Sushi-2.6bpw at 32 GiB: 342 -> 326 slots/layer). Its 2..8-row
  verify rides the same streamed decode path as an MTP verify (greedy bank byte-equal to serial, stream == non-stream).
- **Per-layer split from a profile**: a pack's [`expert_cache_profile.json`](pack-format.md#expert-cache-profile) lets
  `profiledAllot` move the uniform cache bytes between rate-grouped layers by decode hits per byte (same total, same
  bill). Without one, every layer gets the same bytes ([dsv41-memory](dsv41-memory.md)).
- **Dropped experts** (`--drop-expert n`, [dsv41-arch](dsv41-arch.md#drop-expert)): the plan, the ledger, the profile split, the per-group
  pools and the union bank count the kept experts only; a route that names a dropped id is `ExpertDropped`, and the warm skips them.
- **Lending pools** (`Engine.borrowLayer`, `restoreLayer`, from `borrow_from` on): a whole rate-grouped layer's pool can be
  released (its bank dropped, its slabs unmapped once MLX lets go; MLX's active bytes fall by the pool) and later
  rebuilt with the experts it held. A released layer's routes read every distinct pick into the union workspace and
  run on the union bank, the same bytes out. DeepSeek's STQ prefill borrows its decoder layers' pools for its working
  set ([dsv41-memory](dsv41-memory.md#borrow)); the engine publishes the bytes still lendable to
  `ModelConfig.expert_borrow_bytes`. A resident pack has no union workspace: its first release makes one (sized for
  the lendable layers, paid out of the pools the plan releases) and the last restore drops it.
- **Load-time cache warm**: preload the lowest expert IDs into `floor(0.8 * slots_per_layer)` slots per MoE layer
  before kernel warmup and readiness, within the existing budget. These are ordinary LRU entries, not predicted
  routes; dense prefix layers are skipped. A rate-grouped pack with an `expert_usage.json` (`--drop-expert`'s list)
  instead fills every pool of every layer with its most used experts, the most used last so they leave the LRU last
  (`rankedWarm`). DeepSeek STQ 1.75bpw, `--no-mtp` serial, reqC (5.4k prompt) twice after the warm request: the first
  600 tokens rebuilt 2062 layers before and 1106 after, 43.4 -> 46.5 tok/s (80% of each pool by usage: 1315 rebuilds,
  45.6); the second pass, with no misses, is 50.2 either way. Greedy text byte-identical.

<a id="rate-groups"></a>
## Rate groups under streaming (Flash-Next)

A layer whose experts carry several rates ([pack-format](pack-format.md#rate-groups-per-expert-rates-uneven-expert-counts))
has no single slot size, so it keeps one slot pool per rate group instead of one uniform LRU.

- **Pools.** Each group owns a `GroupCache` over its own experts and a slab region sized in its own rate; the layer's
  nine slabs hold the pools back to back (trellis flat per projection, one scale row per slot). The cache plan and the
  ledger are shared: a layer is billed `slots_per_layer` average experts (`QuantStore.expertBytes`, the pack's mean
  expert), and `GroupedLayer.sizePools` spends exactly that bill, never more, caching the same fraction of every
  group. A group too small to afford one expert of its share is pinned at one slot, paid first; a budget below one
  slot per group is refused by the ledger before the engine opens. The union workspace is one set sized for the widest
  grouped layer, billed as every expert at the mean; a pack that also has uniform streamed layers keeps a second set
  for them and bills both (`StoreShape.union_sets`). The ledger bills each slot's `Ragged` meta row and each grouped
  layer's union meta table (32 bytes per expert).
- **Compute.** The resident `Ragged` bank is rebuilt over the slabs (`Ragged.initPools`), so a streamed forward runs
  `moeRagged` on global slot ids: pool g's slots follow pool g-1's, and its meta table depends only on the pool
  structure, never on what is cached. The kernels, the zero-copy import and the epoch leases are the uniform
  path's; a streamed forward is bit-identical to the resident `Ragged` forward.
- **Resolve.** A route resolves against each group's cache separately (hits touched before admits, as in a uniform
  layer). One group overflowing its pool sends the whole layer through the union workspace: every group's experts are
  read into it at their group position, misses that fit are admitted into their pool, and the bank over the
  workspace serves the forward.
- **A wide route splits instead** (`prepareHostWide`, a route only the per-group prefill GEMMs read): every group
  keeps its hits and admitted misses in its pool, only the declined misses are read into the workspace, and
  `Ragged.initSplit` serves each overflowing group as two pools, its cache and its spill. No host copies, the same
  SSD reads, the same bytes out. The decode kernels index one trellis, so a decode-width route keeps the copying union.
- The breakdown line's `copy_ms` / `unions` time the host copies of a union route: cached hits copied into the
  workspace and admitted misses copied back. A read-ahead (the next router on this layer's rows) was measured and cut:
  it guessed about a quarter of a 1024-row chunk's picks and did not move cold prefill.
- **Grouped layers queue from their own map**: `specRoute` serves a uniform cache map only; a rate-grouped layer's map
  is `Engine.specGrouped` (the global slot of each ready expert of its pools, one GPU copy per layer, refreshed with
  its host copy), used by DeepSeek-V4.1 STQ ([dsv41-arch](dsv41-arch.md#stq)). The lossy pick reaches DeepSeek's grouped
  layers through `Engine.swapGroupedMisses` (below) and keeps the waiting schedule. The load-time warm of a pack without a usage file preloads the lowest expert ids, which fill group 0 first.
- Rate groups on MiMo stay resident-only; GLM serves none.

## Decode schedule

- A streamed layer is latency-bound: one GPU→host read of the router ids per layer. The layer's expert compute is
  submitted with `mlx_async_eval` as soon as it is built, and at decode widths (<= 16 rows) the shared expert is
  submitted between the ids and the blocking read, so the GPU runs it during the host round trip. Same ops, same
  order; output is bit-identical.
- DeepSeek-V4.1 STQ (`dsv41_stream.StqStream`) at decode widths queues each layer's routed chain from its GPU map
  (`Engine.specGrouped`) and the host settles it one layer late, after the next layer is submitted: it reads the picks,
  resolves them in the pools (`prepareHost`: fills, LRU and stats move here only) and keeps the queued chain when
  `specMatches` holds, else builds the exact chain and the successor is rebuilt from it. At most one layer is
  unresolved; a union route, a wide chunk, `sync_layers`, tracing/profiling, a lossy pick and a pass that has already
  rebuilt 3 layers plus a third of the layers built take the waiting schedule (picks submitted, shared expert queued behind them, one
  host read per layer). The breakdown line's `idswait_ms` is the settle's wait
  for a layer's picks, `compute_ms` the exact chain's build (miss only), `fill_ms` the SSD reads.
  `[dsv41] route-spec: N layers kept, M rebuilt` is the engagement line ([dsv41-arch](dsv41-arch.md#stq)).
  `SUSHI_DSV41_READ_DIAG=1` counts the blocking host reads of a decode forward by site (`settle`, `settle_last`,
  `picks_sync`, `engram`, `state`, `head`) and logs their mean per forward at debug level every 64 forwards.
- DeepSeek-V4.1 STQ also reads experts ahead (`Engine.prefetch`, `Prefetcher`): a layer's router input times the gate
  matrix of the layer `d` ahead (d = 3, for forwards of 2 or more rows only: at one row the GEMV and the waits cost more than the misses they save; `Model.prefetch_lead_rows`) ranks that layer's experts, the
  host passes each row's first uncached one of its top 12 to a thread that reads them into the pool before the layer's
  chain is queued, so the miss becomes a hit. Rules: a job owns its layer's cache and slabs until `prefetchWait(layer)`
  returns (`specGrouped`, `prepareHost*`, `swapGroupedMisses`, `borrowLayer`, `restoreLayer` call it, `warmCache` and
  `deinit` drain the thread); only layers `d >= 2` ahead are targeted, so no queued chain reads a slot the job writes;
  the prefetch pool has its own fill workers (inside `BOUNCE_BYTES`, so no new bill); it changes what is cached, never
  the arithmetic. `[expert-stream] prefetch: ...` is the engagement line. Measurements: [dsv41-arch](dsv41-arch.md#stq).
- At decode widths each layer's experts are queued from a GPU copy of the cache map (`Engine.specRoute`: expert →
  slot, -1 where not ready) before the host reads the router ids. The host keeps that result only when every routed
  id resolved to the slot the GPU gathered (`specMatches`), else rebuilds it; a union workspace always rebuilds. The
  host map and its GPU copy change together (`LayerState.refreshSpec`), or a stale GPU map could pass the check.
- Qwen4 single-token decode keeps at most one unresolved layer while submitting the next GDN MoE layer's router
  and cache-map expert compute. A cache miss discards that successor, restores its recurrent handles, and rebuilds
  from the preceding MLP's exact output; PLE and full-attention successors verify first. Cache resolution and its
  accounting occur only after the predecessor passes, so discarded routes never fill or touch LRU state.
- Deferred HC writes belong to the speculative stream: rollback restores the preceding MLP's stream and injection
  gate, then replaces the pending write with its exact result. Rollback transfers the saved HC handle, so a later
  MLX error leaves one valid cleanup owner. Profiling, dtype tracing, layer captures, stand-ins,
  imatrix collection and batched/wide forwards keep synchronous verification; a lossy pick defers only when the
  GPU made it (below).
- `SUSHI_EXPERT_DEFER_SYNC=1` selects synchronous verification for an exact schedule comparison: deferred
  verification overlaps host work on hits but spends an extra GDN build and speculative compute on misses.

## Lossy expert pick (`--expert-pick-tolerance <n>`, 0..0.6, default 0 = exact)

- On a cache miss at decode widths, a routed expert may be replaced by the best cached expert outside the row's top-k
  when its router probability is at least `(1-n)` times the missed one's (`ln(1-n)` on the logit difference). The substitute keeps the
  missed expert's routing weight. `expert_stream.substituteMisses`; skipped under imatrix capture.
- A sigmoid router (MiMo) feeds the pick log sigmoid(logit), unbiased (its correction bias only selects), so the
  same gap test reads sigma_sub >= (1-n) sigma_miss (`routerSwapLogits`; -logaddexp(0, -x), finite where sigmoid
  underflows). A raw sigmoid logit gap is not a probability ratio.
- A row's expert is swapped at most `PICK_STARVE_LIMIT` (3) times in a row, then fetched, so a hot expert cannot stay
  out of the cache. Misses are taken in descending logit order, and a missed expert another row is already fetching
  counts as loading, not as a swap target.
- At one-row decode the pick runs on the GPU (`expertPickGpu`, the same algorithm as `substituteMisses`, fed the
  layer's map and starvation counts) and the experts are queued from its slots; the picked ids ride the ids' command
  buffer. The host still makes its own pick and keeps the GPU result only when both agree.
- On a rate-grouped layer (DeepSeek-V4.1 STQ, `dsv41_stream.StqStream`, decode widths): a missed expert may take an
  expert held in any of the layer's pools, which its own group's pool serves; the scores compared are the router's
  sqrt-softplus scores, the weights' own scale. Same starvation limit; the `[expert-swap]` lines count it
  (`Engine.swapped_ids`). A resident layer never misses, so it never swaps: a load whose budget holds every expert
  (the Resident bullet above) logs `[expert-swap] --expert-pick-tolerance has no effect` at boot, and an identical
  KLD with no `[expert-swap] engaged` line means that. A DSpark verify block (2..16 rows) picks like a decode step;
  a span over `DECODE_ROWS_MAX` (16) rows, a prefill chunk, stays exact. `kld compare` takes the pick on the
  one-row steps that score the generated tokens; its prompt prefill is exact.
- A streamed GLM has no pick: `glm5_stream` always fetches a miss, so the flag does nothing there.
- Output depends on cache state, so it is not reproducible across runs or prompt histories. `kld capture` refuses a
  non-zero tolerance: the teacher is exact routing. Numbers: [quality-kld](quality-kld.md#lossy-expert-pick).

## I/O

- `FillPool` = F_NOCACHE + F_RDAHEAD 0 positioned preads, fd cache validated by (dev, ino, size, mtime), spans sorted
  and coalesced to 64 MiB, page-aligned bounce otherwise.
- **A tile span** (`FillSpan.tile`, DeepSeek-V4.1 STQ pools) is read through the worker's bounce buffer and permuted
  into the slab by `layout.tileMajor`, in place of the direct read or the bounce copy. It is never coalesced with a
  neighbour, so a job is one projection of one expert (at most ~6 MB) and the bounce the ledger already bills
  (`BOUNCE_BYTES`) is the only scratch. A pool-to-union or union-to-pool copy stays a memcpy: both are tile-major.
- `FillPool` workers park on their condition without broadcasting: `submit` signals after every push, and a
  broadcast before the wait makes idle workers wake each other forever.
- `PageSlab` epoch leases (`free → filling → ready → leased → readers_complete → reclaimable`, CPU writes only in
  `filling`).
- `importSlab` = `mlx_array_new_data_managed_payload` verified by pointer identity (`ExpertSlabImportCopied`
  refuses). Bench: `tests/ssd_fill_bench.sh`.
- **MLX releases an IMPORTED host buffer asynchronously**: `mlx_array_free` returns BEFORE the payload deleter runs;
  wait for the deleter (`SlabOperand.destroy`), leak (counted, logged on the breakdown line) rather than unmap what
  MLX still holds.

## Compute

- bf16 checkpoint → `expert_bf16_kernels` (`downKernelPreferred(rows) = rows >= 2`: the in-dispatch k-reduction tail
  loses at one row; ticks past the kernels' row/top-k limits take the `gather_mm` composite).
- Quantized packs → the RESIDENT fused kernels over the slab with remapped ids, bit-identical to the resident load
  (bytes and top-20 logprobs on greedy prompts). A warm quantized forward pays the per-layer barrier, not the fills.

## Correctness bars

- Store-level same-expert byte identity (`real qwen expert store spans and source bytes are exact`); teacher replay
  via `kld compare` (the affine pack is the control); greedy determinism.
- Streamed logits equal resident logits bit for bit through eviction and the union (`GLM serving streams EXL3
  experts ...`, `GLM serving streams FP8 source experts ...`); `every real streamed pack and source on this box
  plans a streamed load` plans each pack on the box.
- Live (main `1e484c10` plus this change): GLM-5.3-Flash-Sushi-2.3bpw `kld compare --limit 1` on the 4x512 BF16
  teacher scores the same every field streamed at `--ssd-budget-gb 32` (81 slots/layer) and resident, with the BF16
  latent (KLD 0.044647, top-1 489/512) and with `--kv-quant 8` (KLD 0.042316, top-1 492/512).
- Cross-day comparisons must match forwards on `hits` + `fill_bytes_per_row` (the SSD's delivered rate drifts).
- `SUSHI_NGRAM_BF16_DIR=<hf checkpoint>` serves any pack with the ORIGINAL bf16 n-gram table so `kld compare`
  isolates the PLE table's cost.

<a id="imatrix"></a>
## Imatrix capture

Imatrix capture rides the streamed bf16 forward (`SUSHI_IMATRIX_OUT=<abs>.safetensors`, `src/imatrix.zig`):
per-layer per-expert sum(x²) and routed counts accumulate ON the GPU keyed by GLOBAL expert ids (slab slots are
remapped), in the collector's contract the converter reads; the flush runs on the INFERENCE thread (loop exit or
`/v1/unload-model`), never on `Scheduler.deinit`'s caller thread. MiMo's o_proj and lm_head inputs ride the same
file as per-channel mean squares under their source weight names ([mimo2-arch](mimo2-arch.md)). The drivers that feed it a corpus live in the private
converter repo. Routed counts reconcile to
tokens x top-k exactly on every layer; the two load-time warmup forwards add a few tokens.

- **GLM**: the native BF16 teacher capture (`sushi kld capture --layer-major`, also window-major) accumulates the same file
  from `Stream.apply`: the MLP input and the clamped SwiGLU activation rows, keyed by the router's global expert ids,
  under the source names `model.language_model.layers.{L}.mlp.experts.*`. `--imatrix-windows N` counts only the first N
  windows (a batch never straddles the cut), so held-out windows can ride the same pass for the tune's boundaries.
  A layer-major run resumes with its sums: each committed batch leaves `imatrix-<windows>.safetensors` (raw
  accumulators) in `<out>.partial`, and a resume without the matching file is refused (`GlmLayerMajorImatrixStateMissing`).
  Only BF16 experts are observed; any other layout is refused (`GlmImatrixNeedsBf16Experts`).
- **Hidden capture** (`SUSHI_HIDDEN_OUT=<abs dir>`): `sushi kld capture` (no prefix cache, no warmup) appends every
  prompt token's residual at each block boundary (`boundary-XX.bin`, raw bf16 [tokens, hidden]; 00 = layer 0's input,
  b = layer b-1's output), then its ids (`tokens.bin`, u32); `forwardMoeWith` and the native GLM teacher (all four
  HC streams, `Request.boundaries`) only; logits bit-identical.
- Its output files are private: each is created exclusively without following a link, and an existing one is appended
  to only when it is a regular file with one link (a hard-linked or symlinked output is refused, its target untouched).

## Discovery

A dense qwen4_exp checkpoint with a complete streaming index registers as a streaming stub
(`streamingStubMarker`); `/v1/models` carries `streaming`, `streaming_required`, `ssd_budget_gb` at top level and
`input_modalities: ["text"]` when it must stream. Guards: `tests/test_bf16_streaming.sh`,
`tests/test_model_settings.sh` [5].
