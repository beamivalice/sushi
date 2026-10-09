# Engine: memory, admission and context sizing

How the engine decides what fits on a unified-memory Mac: the GPU ceiling, load-time preflight, auto-context,
prefill chunk width, the admission line for a long prompt, and why under-billing is fatal. Read this before touching
any `*Bytes` bill, `Scheduler.init`, the preflight, or the admission path.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-kv-cache](engine-kv-cache.md),
[engine-qsa-long-context](engine-qsa-long-context.md#admission-and-load-time-bills),
[engine-prefix-cache](engine-prefix-cache.md#budget), [engine-expert-streaming](engine-expert-streaming.md#budget),
[arch-mimo-v2](arch-mimo-v2.md#bills-the-bill-follows-the-storage-in-the-same-commit).

## Metal OOM

- **Metal OOM is UNCATCHABLE and Metal at the working-set edge returns ZEROS before it aborts**: all-zero logits from
  healthy inputs = MEMORY symptom.
- `currentGpuMemoryCeiling` must see EXTERNAL pressure; under-billing is a Metal OOM, so a bill goes down only where
  the bytes are gone.
- `--wired-margin-gib` (default 1, integers 1..32) is how far under a raised `iogpu.wired_limit_mb` a plan may reach.
- The box: M5 Max 128 GB; the default wired limit admits about 120 GB; a resident MiMo EXL3 pack is over 90 GB, so two
  heavy GPU jobs at once risk an OOM for both (and concurrent conversions have died together in a GPU reset).

## Load-time preflight

- The preflight's available figure is free RAM CAPPED at Metal's working-set limit (`effectiveAvailableBytes`): a
  lowered `iogpu.wired_limit_mb` binds below free RAM, and a load past it failed warmup, then every request.
- Preflight refusals → `InsufficientMemory` → 503 + entry reset to `.unloaded`. A refusal quotes the number it
  COMPARED (`loadRequirementBytes`) and the flag that would admit (`--wired-margin-gib`, `--skip-mem-preflight`,
  `iogpu.wired_limit_mb`).
- **One bill, `loadRequirementBytes`**: the weights the loader bills (enabled tensors only: MTP head, vision tower,
  DFlash2 assistant and its runtime cache) + the larger of the arch's measured warmup transient and the requested
  context's admission bill (the warmup ends before any request allocates) + a fixed 1 GiB net. No proportional term,
  no flat headroom. The cold-load registry gates (`residentColdLoadBillBytes`, `glmColdLoadBillBytes`) call the same
  function, so gate and preflight read one number.
- **The warmup transient is measured per arch** (`loadWarmupBytes`, [Load warmup](#load-warmup)): Flash-Next 0.125 GiB
  plus 1.125 GiB while MTP is on, MiMo 0.25 GiB, GLM its fixed KDA state, pool rings and first 1024 latent rows plus
  0.375 GiB. A new arch or layout adds its own measured figure here, never a percentage of its weights.
- Qwen drops disabled native MTP tensors before the transformer binds them. An auto context bills no context term: it
  is pinned after the load from what the weights leave. Their default KV remains affine 8-bit.
- An explicit launch or model-settings context is checked before loading resident Sushi Qwen and MiMo EXL3 packs.
  `loadServingBill` uses `prefillNeededAtChunk`, including KV, rings, recurrent/history state, MTP and prefill scratch.
  It prices an explicit prefill width at that width after the architecture's cap; otherwise it proves the per-request
  floor. A refusal reports the largest context that fits the same bill (net included) and available memory, capped at
  the model's context limit. Existing active MLX buffers are subtracted from the GPU working-set limit before this
  check. Cold-load HTTP 503 responses retain the numeric maximum through the transient unloaded reset; retry and
  successful load clear it.
- Request-time memory refusals for these packs also report a maximum context using the request's actual KV width,
  effective chunk, MTP choice and warm-prefix credits. The connection-thread 400 and a later scheduler refusal
  carry the same numeric diagnostic; generation error mapping consumes the slot's diagnostic on its connection
  thread so it cannot leak into another request.
- **A streamed load bills what it holds** (`streamedLoadRequirementBytes`): resident trunk, MTP and vision + the expert
  cache + the whole-layer fill union + the bounce buffers + the net; `expertCacheFitForLoad` separately proves the
  planned KV against the wired limit. GLM allocates KV row by row per request, so its load bill carries no context
  term; request admission bills it.
- GLM evicts its hot cache to admit (`admissionEvictsHotCache`), and its context sizer reserves no cache. With the
  prefix cache on, the inference thread's bill adds the KDA checkpoints a prefill holds (up to 9 x 147,619,840 bytes,
  from the capture schedule the generator runs), one assistant window and the RAM tier's row copy (the SSD writer's
  1 GiB permit is the shared headroom term below, for every arch), and keeps fewer checkpoints where they do not fit, never refusing for them
  ([engine-prefix-cache](engine-prefix-cache.md#glm)).
- `modelDiskBytes` bills the shards the INDEX names; an index that names NO shard on disk is STALE (every shard
  loads, one warning). Every size sum stats THROUGH symlinks (HF-cache models).
- Load-time bills run INSIDE `Scheduler.init` ([engine-qsa-long-context](engine-qsa-long-context.md)).
- **The kernel unwires a freed Metal buffer asynchronously** (~0.5 s for 50 GB): an unload and an eviction-before-load
  wait until most of the freed bytes left the wired set (`waitForUnwire`, bounded at 3 s), or the next preflight reads
  them as taken (45 GB free where 95 GB was a moment later).
- **A separately loaded MTP sidecar is billed beside the shards** (`mtpSidecarBytes`, at its file size, only when MTP is on and
  `loadMtp` will read it): in the plain bill and in the Sushi bill, where its coarse rerank copy follows. A head in the checkpoint
  is already in the shard bill and adds nothing; MiMo and GLM never call `loadMtp`.
- A ready entry's `bytes_resident` (the registry's resident-memory gate, `/v1/models`) is the weights the preflight
  billed (`residentWeightBytes`): a boot `--model` entry has no discovery `bytes_on_disk`, so it measures the shards.
- **A resident Sushi Qwen or MiMo cold load reserves its load preflight's own requirement** (`residentColdLoadBillBytes`:
  exact enabled weights and the same full-context admission callback), rather than the 1.1x disk-size guess. The auto resident cap bounds co-residence only ([server-lifecycle](server-lifecycle.md)).

<a id="load-warmup"></a>
### Load warmup

The warmup term of `loadRequirementBytes` is the pre-request `/props` `peak_bytes` less the bytes the load bills, at the
worst setting measured for each arch. Binary: `165c2d59` plus the margin and auto-context changes (before this bill),
ReleaseFast, M5 Max 128 GB, `taskpolicy -a`, one `wired-margin` GPU lock per boot, one boot per row, 2026-10-08.

| arch, pack, settings | billed weights (GiB) | peak (GiB) | peak - billed (GiB) |
|---|---:|---:|---:|
| Flash-Next 2.6bpw, `--mtp` (vision billed, tower not yet resident) | 44.20 | 44.53 | 0.32 |
| Flash-Next 2.6bpw, `--mtp --no-vision` | 43.37 | 44.53 | 1.16 |
| Flash-Next 2.6bpw, `--no-mtp` | 43.08 | 42.36 | -0.72 |
| Flash-Next 4bpw, `--mtp` | 63.94 | 64.26 | 0.32 |
| MiMo 2.3bpw, `--mtp` | 89.65 | 89.88 | 0.23 |
| MiMo 2.3bpw, `--mtp --no-vision` | 88.30 | 88.52 | 0.22 |
| GLM 2.4bpw, DFlash2 + vision | 89.76 | 90.16 | 0.40 |
| GLM 2.4bpw, `--no-drafter --no-vision` | 88.58 | 88.98 | 0.40 |

The peak is the same with `--ctx-size 1248` and with an auto context (the KV allocates at the first request).
Flash-Next's transient is its MTP head load: peak less resident bytes is 1.11 GiB with MTP and 0.08 without. GLM's 0.40
is its KDA state, rings and latent rows (0.15) plus 0.25 of activations.

The bill against the same peaks (this change; `--mtp` where the arch has it; the context row is the request-time
admission bill at 1248 tokens, which replaces the warmup once it is larger):

| pack, settings | weights | warmup or context | net | bill | peak | slack |
|---|---:|---:|---:|---:|---:|---:|
| Flash-Next 2.6bpw, auto or `--ctx-size 1248` | 44.20 | 1.25 | 1.00 | 46.45 | 44.53 | 1.93 |
| Flash-Next 2.6bpw, `--no-vision` | 43.37 | 1.25 | 1.00 | 45.62 | 44.53 | 1.09 |
| MiMo 2.3bpw, auto | 89.65 | 0.25 | 1.00 | 90.90 | 89.88 | 1.02 |
| MiMo 2.3bpw, `--ctx-size 1248` | 89.65 | 1.79 | 1.00 | 92.44 | 89.88 | 2.56 |
| GLM 2.4bpw, auto or `--ctx-size 1248` | 89.76 | 0.52 | 1.00 | 91.28 | 90.16 | 1.12 |
| Flash-Next 2bpw streamed, `--ssd-budget-gb 18 --ctx-size 66000` | 5.57 | 13.69 | 1.00 | 20.27 | 19.07 | 1.20 |

Flash-Next's extra slack is the vision tower: it is billed at load for the first image and not resident until then.
MiMo at 1248 tokens bills a request's prefill scratch (1.79) above its warmup. The streamed warmup is its cache, fill
union and bounce buffers. Every bill is above its peak; the 1 GiB net is what separates them where nothing else does.
Previous bills for the same boots: 46.20, 91.65, 91.91 and 7.27 GiB (the streamed one was 12 GiB under its peak).

Other expert layouts of these archs bill the same terms; their warmup is not measured on this box.

## Context and chunk

- **Auto-context is PINNED at load** (`pinAutoContext`, one 93% margin on the memory ceiling for every model, then
  capped at the checkpoint's own maximum, un-margined); ask `getEffectiveContextLength`. It bills KV at the CONFIGURED
  width and activations ONCE.
- The prefill CHUNK is a machine decision (`resolvePrefillChunk`, ladder 8192→512 at ≤ a quarter of the serving
  budget). `--prefill-chunk` pins it off the per-request ladder; on the ladder it is the widest rung. `prefillMemoryNeeded` takes STORED and SCORED widths as two parameters.
- GLM's load-time pin, the width its advertised context is billed at, is the widest rung up to 2048 that advertises
  as much context as 512: the `max_safe_context` bill picks it, not a quarter of free memory
  ([arch-glm5-next](arch-glm5-next.md#memory)).
- RAM retention is off by default and the SSD tier on, so every model bills as SSD-first: a ringed arch with a disk tier
  bills its restore's coexistence (`oldBuffersInEvalWindow`) ([engine-prefix-cache](engine-prefix-cache.md#defaults)). Context sizing reserves nothing
  for an idle cache (`ctxSizingCacheReserve` is 0 with RAM off). A model whose tier did not come up bills as without one
  (`server.diskTierOn`).
- **Admission bills the SSD writers' host bytes** (`prefillAdmissionBill` -> `admissionAvailable`): the writer holds its
  backlog, up to its ~1 GiB permit, outside `mlx_get_active_memory`, so every resident model's `writerHostBytes` (the
  larger of the backlog and the permit) is taken off the headroom BEFORE the width and the cache reservation are chosen,
  an explicit `--prefill-chunk` included. The ONE reading is `scheduler.diskWriterHostBytes`, for Qwen, MiMo and GLM
  alike; GLM's `glmPrefixStateBytes`/`glmCommitStateBytes` carry no writer term, so the permit is billed exactly once.
- The ungated hot-cache ask is zero when RAM retention is off, `--prefix-cache-entries 0`, or the model's hot cache
  never loads (`HotPrefixCache.shouldUse`); otherwise it is `--prefix-cache-mem`.
- A per-request arch (`perRequestPrefillChunk`: qwen4_exp, the ringed mimo_v2, glm5_next) prefills every request
  from its start width (`prefillStartWidth`: 4096, 2048, 2048; an explicit `--prefill-chunk` sets it instead). Each
  chunk boundary steps down a rung, never up again, only where the next chunk's cost no longer fits the live
  headroom beside the KV (`adaptivePrefillWidth`). Admission bills the request's top at the widest rung whose whole
  bill fits (`chooseRequestPrefillChunk`), so a long prompt is never under-billed; the load-time pin is only the
  fallback and the width the advertised context is billed at.
- **The load line names a per-request arch's pin as the fallback** (`prefillChunkLoadLine`: "per request, up to N at
  a short prompt; load-time fallback M"; Flash-Next's bound narrows as the context grows). MiMo's pin swings 512-2048 between boots with the memory active at load (the ungated cap,
  (ceiling - active - hot-cache ask) / 4, is ~4 GiB beside a 3.6 GiB 2048 reserve), while every request up to 256k
  prefills at 2048 (bill 4.7 GiB at 64k, 7.5 GiB at 256k, against ~17.9 GiB available).
- An explicit `--ctx-size` outranks auto-context and `model-settings.json` `ctx_size`.
- Disconnect cancellation takes effect at the next prefill chunk boundary; a wall-time cancellation test must bound
  its chunk size rather than assume the auto-sized chunk fits a fixed deadline.
- Admission-bill tests pin synthetic contexts and use request shapes that fit them, never the test host's auto-context.

<a id="recipe-64gb"></a>
### The 48 GB and 64 GB recipe contexts

The README's 64 GB `--ctx-size` values are checked with the engine's own full-context admission bill
(`prefillNeededAtChunk` through the per-request ladder) at a 59,000 MB ceiling: a prompt that fills the context, MTP
on, `--mtp-head-kv-quant`, the 1 GiB hot cache pinned (not evictable). No live boot: the wired limit has only test
seams (`wired_limit_mb_override`, `static_ceiling_override`), so the bill was computed in a scratch test at `ad5e6be8`.

| pack, KV | `--ctx-size` | bill (MiB) | width | available (MiB) | weights + bill (GiB) | largest admitted |
|---|---:|---:|---:|---:|---:|---:|
| Sushi-2.6bpw, kv8 | 250000 | 11866 | 4096 | 12975 | 55.5 | 470000 |
| Sushi-2.6bpw, kv4 | 450000 | 12719 | 4096 | 12975 | 56.4 | 786000 |
| Sushi-3bpw, kv8 | 128000 | 7329 | 2048 | 7463 | 56.5 | 200000 |
| Sushi-3bpw, kv4 | 248000 | 6987 | 1024 | 7463 | 56.2 | 322000 |

The 48 GB recipe is the same check at a 43,000 MB ceiling (probe at `eab60060`, weights 37,552,413,730 bytes):

| pack, KV | `--ctx-size` | bill (MiB) | width | available (MiB) | weights + bill (GiB) | largest admitted |
|---|---:|---:|---:|---:|---:|---:|
| Sushi-2bpw, kv8 | 131072 | 5905 | 512 | 6163 | 40.7 | 141072 |

The README memory table's "needed" row is the weights plus this full-context bill at the 512 rung plus a 1 GiB hot
cache. At `eab60060` the bill is the same for every Flash-Next pack and for `--max-tokens` 32000 or 64000: 5905 /
8825 / 14025 / 24425 MiB at 128k / 256k / 512k / 1M tokens (KV 2080 / 4160 / 8320 / 16640 of it).
Its max-context column is the largest multiple of 8192 tokens whose weights + 512-rung bill + 1 GiB hot cache + 256
MiB spare fits each wired limit, capped at 1M (same probe, kv8 and kv4).

The ladder widens the chunk until the bill nearly fills what is available, so the spare in a row is small by
construction; "largest admitted" (2000-token steps) is where even the 512 rung stops fitting. The ceiling assumes
the full limit is reachable: on a real 64 GB Mac the free-RAM term can bind lower (`currentGpuMemoryCeiling`).

## Admission

- One `[admission] needed=… available=… reclaimable=… width=… verdict=…` line per decision.
- An explicit `--prefill-chunk N` caps the per-request ladder, never pins it: N if it fits, else the widest rung
  below N that fits, down to 512; refused only when that floor does not fit. `requestPrefillPick` is the one rule for
  the bill and the scheduler, `generate.requestPrefillChunk` the width both run; the hot-cache clamp and a streamed
  load prove that floor (`perRequestFloorWidth`). `SUSHI_PREFILL_CHUNK_PER_REQUEST=0` restores the pin.
- A long prefill evicts the hot cache on the INFERENCE thread to be admitted (`evictLruToAdmit`) on every arch where
  `admissionEvictsHotCache` holds (qwen4_exp and the ringed mimo_v2; the connection thread's `creditedAdmissionBill`
  and the scheduler's `admissionPassArmed` read that one predicate), crediting only
  provably reclaimable bytes; `PrefillDoesNotFit` → 400 by name. A warm share that does not fit is first taken
  over (`checkoutRestored`: its append donates, so the restored rows are not billed twice).
- qwen4_exp bills a warm request AFTER its restore, so a disk-restored buffer is live memory at the bill. The SSD
  tier fills buffers the slot owns (`LookupResult.slot_owned`), so its rows are credited like a checkout's: the
  first append grows each layer and its old rows are freed at that layer's eval window, which is all the bill keeps
  (`grow_coexist_bytes`). MiMo's ring restore is not credited. The restore itself runs unbilled, so it holds the
  restored KV plus one chunk ([engine-prefix-cache](engine-prefix-cache.md#basics)).
- Measured (Sushi-3bpw, kv8, `--ctx-size 262144 --prefix-cache-disk 20GB --prefix-cache-entries 1`, MTP on, a
  140,565-token SSD restore, one boot, `taskpolicy -a`, lock held, 2026-10-01): the peak over live memory at the bill
  is 1,534 MiB with a 2,328-token tail and 3,072 MiB with 9,144, against a credited bill of 9,425 / 9,641 MiB (11,430
  / 11,647 MiB uncredited). The grown buffers alone exceed the restored rows' 1,750 MiB, so those rows were freed
  before the peak.
- A warm restore whose buffers hold the prompt but not the reservation (seq <= C < R) is grown to R before the
  prefill's first chunk (`KVCache.growToReservation`), one KV layer per eval, so it bills one window of old rows
  (`oldBuffersInEvalWindow`) for a donated restore and nothing for a share. At a 128k entry, kv8: +170 MiB on
  qwen4_exp and +212.5 MiB on mimo_v2 over the bill without the grow; growing at the first decode step instead
  held every layer's old rows at once.
- The eviction pass drains the GPU stream before it reads live memory: a command buffer in flight holds its inputs'
  buffers, so an eviction read early frees nothing and trips the shared-entry stop.
- **Concurrent arrivals are each billed against the SAME free memory** on their connection threads. The gated arch
  (qwen4_exp) re-bills live memory before each prefill in `runPrefill`; an ungated one (mimo_v2) is re-billed at the
  pending drain (`admitsWithinMemory`: live requests plus this tick's earlier admits). One that does not fit beside
  company waits in `pending` (`[admission] held`); alone it proceeds.
- **An admitted request keeps its unallocated cache growth promised across ticks** (`Slot.growth_commit`, the bill's
  `commit`, released as the slot's resident cache reaches it): a later admission subtracts every live slot's
  outstanding share from its headroom. Native GLM MLA grows row by row, so two short-prompt requests with large output
  limits otherwise both admit and together outgrow the box; a DFlash2 request reserves its capacity after prefill and
  owes nothing. Qwen and MiMo take their reserved KV at the first grow, resident before the next admission, so their
  `commit` is zero.
- The hot-cache budget is clamped at load and follows residency ([engine-prefix-cache](engine-prefix-cache.md#budget)).
- Context-overflow 400s name BOTH counts.
- **A freed reserved-KV slot goes back to the OS, not MLX's pool** (`deinitSlotsReturningPool`, and the prefill-end
  clear, both on `reservesKvCapacity`): the request-end clear runs before the slot is freed, so its KV parked there
  (MiMo: 1.6 GiB after a 128k request, 3.2 after 256k) and the next admission read it as spent.
- MiMo MTP adds a constant per-request and load-time reserve (`mimo_mtp.State.billedBytes`) for all three sliding head KVs, retained hiddens, and catch-up/concatenation buffers; it is zero with MTP off and never scales with context.
- **A vision encode is billed before it runs** (`towerFitFault`, `server.visionEncodeBill`): the largest block's tower
  scratch (`qwen_vision.encodeScratchBytes`, fitted >= 25% over the measured peak) plus every block's float32 pixels
  and three bf16 copies of its soft-token rows (group outputs, video concatenation, request concatenation); past what
  the GPU has left it is a named 400. The tower evaluates per block, so the peak is one block's f32 score sheet
  (heads x N^2) and rows; table in [arch-qwen4exp](arch-qwen4exp.md#vision-tower).
- **A queued vision encode is billed again on the inference thread** (`runVisionEncode`, `vision_encode_available`),
  right before the tower runs: a prefill's cache growth or an earlier encode's resident output may have taken the
  headroom since the connection thread's check, and the refusal is that same named 400.
- **A streamed `--vision` load bills its tower in the ssd budget and proves its largest image beside it**: the load
  admits `budget + planned KV + largestImageEncodeBytes <= wired limit`; requests are still billed live as above
  ([engine-expert-streaming](engine-expert-streaming.md#vision)).
- **A video's block is ONE temporal group**, never the whole video: `forwardVideo` encodes and evaluates each group
  alone, so the bill grows linearly with the group count (the old N^2 over all groups billed 79 GB at 8x46x82).
  Measured peak (pixel upload to evaluated output) = one group's scratch + all pixels + the earlier groups' rows:

  | video (t x h x w patches) | peak | old bill | bill | bill / peak |
  |---|---|---|---|---|
  | 1 x 46x82 | 1249 MB | 1626 MB | 1658 MB | 1.33x |
  | 2 x 46x82 | 1277 MB | 5.6 GB | 1696 MB | 1.33x |
  | 4 x 46x82 | 1333 MB | 20.6 GB | 1771 MB | 1.33x |
  | 8 x 46x82 | 1445 MB | 79.1 GB | 1922 MB | 1.33x |
  | 2 x 24x42 | 172 MB | 590 MB | 246 MB | 1.43x |
  | 8 x 24x42 | 217 MB | 6.3 GB | 306 MB | 1.41x |
  | 2 x 96x96 (1536² cap) | 6237 MB | 30.3 GB | 8270 MB | 1.33x |
  | 8 x 96x96 (1536² cap) | 6648 MB | 460 GB | 8822 MB | 1.33x |

  `qwen vision ubench` on `242a5545` plus this change, Sushi-3bpw tower, random pixels, 3 passes (the image rows'
  peaks reproduced to the MB), `taskpolicy -a`, GPU lock `video-bill`, 2026-09-25; pinned by `visionEncodeBill covers
  each measured video peak`.

GLM source-FP8 packs retain one byte per E4M3FN weight and four bytes per 128×128 tile
scale. The selected tensor payload bill includes those grids. They share MiMo's `fp8_block`
GEMV/dequantize path. Beyond its 16-row GEMV limit, `glmFp8DequantScratchBytes` charges
all FP8 projections in the largest pending layer, times the native evaluation window.
For GLM-5.3's default two pending layers this is 576 MiB; decode adds no dense weight copy.
Resident GLM also honors an explicitly raised `iogpu.wired_limit_mb` ceiling, retaining the
same configurable 8 GiB reserve as Qwen and MiMo. An unchanged system limit retains the
physical free-memory ceiling.

GLM uploads only indexed payloads from one shard at a time, preserving their stored
dtypes. It closes each shard before opening the next. Each tensor is read through a page-aligned
window on an `F_NOCACHE` descriptor (`model.readTensorArray`; macOS ignores `F_NOCACHE` unless the file offset, length
and buffer are all page-aligned). Four workers each read the next 16 MiB chunk into their own scratch and copy it
into place (`expert_io.ParallelReader`); a tensor of 1 MiB or more lands at offset 0 of its own mapping, imported by MLX
without a copy (a view at an offset inside a buffer gave NaN logits). Same settings, H N H N per pack, sequential aligned reads → the
overlapped reader: GLM upload 13.6/13.9 → 9.0/8.8 s (listening 18.4/15.9 → 12.1/10.7 s), Qwen 4bpw 12.0/10.5 → 9.5/7.7 s
(14.8/12.3 → 11.9/9.3 s); no compression or swap in any arm.
Four workers instead of one read-ahead thread, file cache evicted before each arm: GLM upload 9.0/9.3 → 8.0/8.0 s
(listening 11.7/10.9 → 10.4/9.5 s); Qwen 4bpw against `mlx_load_safetensors` 6.7/6.7 vs 7.0/7.2 s (listening 8.3/8.2
vs 8.7/8.6 s), compressor +4.4/+5.5 GB vs 0. Read
through the file cache, the ~90 GB upload filled it and macOS compressed the weights already uploaded (not yet wired)
rather than drop it, then decompressed them when the wired limit applied: on a busier box, that compression is swap.
Sushi-2.4bpw, `serve` defaults, `taskpolicy -a`, lock held, 2026-10-08, ~90 GB free at start: compressor +59.0 GB →
+0.6 GB, file cache peak 64.2 → 21.7 GB, listening 22 → 15 s, warmup 5.6 → 0.16 s (the loader at `3f3ff7bc` → the aligned read). MiMo's
source trunk (`mimo_source.readTensor`) and every `model.loadSafetensorsFile` shard (Qwen, streamed trunks) read the
same way; the latter no longer goes through `mlx_load_safetensors`, whose descriptor cannot take `F_NOCACHE`. Qwen3.8-Flash-Next-Sushi-4bpw
(64 GB resident), same settings, A F A F: compressor +5.7/+4.3 GB → +0.0/+0.0 GB, no swap either way, listening 12/8 s →
13/10 s (`mlx_load_safetensors` → sushi's own aligned reader). MLX lazy safetensor Load nodes
kept one descriptor per shard alive until evaluation, so the 566-shard affine pack
exceeded macOS's default 256-handle terminal limit despite a successful memory
preflight. The bounded reader also reports descriptor exhaustion separately from
missing files. A 300-shard regression runs with a 128-descriptor soft limit.

## Observing memory

`/props` reports `active_bytes`, `memory.cache_bytes`, `batching`; RSS is blind to Metal.
