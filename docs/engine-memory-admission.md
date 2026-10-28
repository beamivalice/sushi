# Engine: memory, admission and context sizing

How the engine decides what fits on a unified-memory Mac: the GPU ceiling, load-time preflight, auto-context,
prefill chunk width, the admission line for a long prompt, and why under-billing is fatal. Read this before touching
any `*Bytes` bill, `Scheduler.init`, the preflight, or the admission path.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-kv-cache](engine-kv-cache.md),
[qwen4-qsa-long-context](qwen4-qsa-long-context.md#admission-and-load-time-bills),
[engine-prefix-cache](engine-prefix-cache.md#budget), [engine-expert-streaming](engine-expert-streaming.md#budget),
[mimo2-arch](mimo2-arch.md#bills-the-bill-follows-the-storage-in-the-same-commit).

## Metal OOM

- **Metal OOM is UNCATCHABLE and Metal at the working-set edge returns ZEROS before it aborts**: all-zero logits from
  healthy inputs = MEMORY symptom.
- `currentGpuMemoryCeiling` must see EXTERNAL pressure; under-billing is a Metal OOM, so a bill goes down only where
  the bytes are gone.
- `--wired-margin-gib` (default 1, integers 1..32) is how far under a raised `iogpu.wired_limit_mb` a plan may reach.
- **A command buffer fails (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) once MLX's peak passes the wired limit less
  the GPU memory MLX does not own** (#61). The limit counts AGX "In use system memory", which is system-wide: other
  processes' Metal allocations (WindowServer, every app with a layer) count against it. Measured on an M1 Max 32 GB,
  `iogpu.wired_limit_mb=30000` (31.46 GB), Qwen3.8-Flash-Next Ternary, kv8, `--no-mtp`, sushi 1.2.4, 8k-35k prompts,
  30+ cold prefills: a 0.62 GB floor idle, MLX peaks of 30.57 GB and below passed, 30.85 GB and above failed.
  - **Non-MLX GPU floor**: under a declared limit the ceiling is the limit less the larger of `--wired-margin-gib` and
    floor + 512 MiB slack (`wiredUnplannedBytes`). The floor is read once, at startup, before MLX holds a model
    (`measureGpuFloor`): right after a load it read 0 and 407 MB (weights still being wired), and after an unload it
    lags high while freed pages unwire.
  - Above the floor the counter is elastic: 1-op/1-MB command buffers drove it to the limit itself, with 1.85 GB of
    non-MLX share, and the prefill passed.
  - Other processes' memory is not the trigger here. A ballast of 1-4 GB touched incompressible RAM was swapped out
    and the prefill passed at the same speed. With the model resident, `mlock` of even 0.5 GB was refused (EAGAIN).
- **The bill counts MLX's command-buffer size** (`cadenceInflightBytes`, in `prefillNeededAtChunk` and
  `prefillTransientReserveAtKv`). MLX keeps each op's inputs alive until its command buffer completes, up to 10 in
  flight, and commits a buffer once its distinct inputs pass `MLX_MAX_MB_PER_BUFFER`. The prefill terms were fitted
  at 50 MB, MLX's Max/Ultra default, so the term is (MB per buffer - 50) MiB x 9/8 and 0 at the default.
  - Measured, same box: 1000 MB lifted MLX's prefill peak by 1.11 / 1.12 / 1.03 GB at 8k / 19k / 35k. 1000 ops at
    50 MB moved nothing.
  - At 1000/1000, 19k and the 35k edge failed inside admission, 3/3 and 2/2; the default passed at every length.
  - Startup logs the cadence MLX resolved from its arch table or the env: `[mem] MLX command buffers: 50 ops / 50 MB
    (default, applegpu_g13s); prefill bill +0 MB in flight`.
- The box: M5 Max 128 GB; the default wired limit admits about 120 GB; a resident MiMo EXL3 pack is over 90 GB, so two
  heavy GPU jobs at once risk an OOM for both (and concurrent conversions have died together in a GPU reset).

## Load-time preflight

- The preflight's available figure is free RAM CAPPED at Metal's working-set limit (`effectiveAvailableBytes`): a
  lowered `iogpu.wired_limit_mb` binds below free RAM, and a load past it failed warmup, then every request.
- **`status.getAvailableMemBytes` is the reclaimable page-class sum, never total minus used**: free (less speculative) plus file-backed pages, capped by the RAM outside wired and compressor. Mach's classes overlap (a resident model's wired pages are also in the anonymous set), so subtracting `wired + compressor + internal` charged them twice and read 4.5 GiB where 19 GiB was grantable (`computeAvailableBytes` tests carry the live counters).
- **The file-backed share is grantable slowly, not at once.**
  - **Slow climb, clean:** MiMo's BF16 KV filled to 550K with no swap while the file cache drained to 3 GiB.
  - **Fast large load, swap:** a 24 GB streamed load beside an idle resident GLM read 32 GiB available, then swapped
    17 GB as macOS compressed the idle model.
  - Both runs: `e8c9c47e`, 2026-10-11.
  - Kept as is: under a declared `iogpu.wired_limit_mb` the wired-limit floor, not this figure, sets the request ceiling.
- **Under a declared `iogpu.wired_limit_mb` the ceiling IS the limit less `--wired-margin-gib`** (`gpuCeilingWithWiredFloor`),
  however much RAM is free. It was the larger of that and the free-RAM term, whose working set is the limit itself: a
  streamed load on a quiet 32 GB box saw 27,936 MiB of ceiling where its requests saw 26,976, and proved an 86k context
  each request then refused.
- **A declared `iogpu.wired_limit_mb` lifts the resident Sushi preflight's available figure to the same floor request admission reads** (`effectiveSushiAvailableBytes`: wired limit less `--wired-margin-gib`, less resident buffers, still under the working set). Free RAM alone read file cache and other apps' pages as gone and refused a 32 GB box whose declared limit admitted the load.
- Preflight refusals → `InsufficientMemory` → 503 + entry reset to `.unloaded`. A refusal quotes the number it
  COMPARED (`loadRequirementBytes`) and the flag that would admit (`--wired-margin-gib`, `--skip-mem-preflight`,
  `iogpu.wired_limit_mb`).
- **One bill, `loadRequirementBytes`**: the weights the loader bills (enabled tensors only: MTP head, vision tower,
  DFlash2 assistant and its runtime cache) + the larger of the arch's measured warmup transient plus a fixed 1 GiB net
  and the requested context's admission bill (the warmup ends before any request allocates). No proportional term,
  no flat headroom. The cold-load registry gates (`residentColdLoadBillBytes`, `glmColdLoadBillBytes`) call the same
  function, so gate and preflight read one number.
- **The warmup transient is measured per arch** (`loadWarmupBytes`, [Load warmup](#load-warmup)): Flash-Next 0.125 GiB
  plus 1.125 GiB while MTP is on, MiMo 0.25 GiB, GLM its fixed KDA state, pool rings and first 1024 latent rows plus
  0.375 GiB. A new arch or layout adds its own measured figure here, never a percentage of its weights.
- **Flash-Next with a DFlash2 drafter bills the drafter as it is resident and as it loads**: `assistantLoadBytes` reads the
  safetensors headers and bills what the 8-bit gs64 quantize step leaves (0.71 GiB for the shipped drafter, not its 1.12
  GiB of BF16 file bytes: the codebooks and norms stay dense) plus a load transient, the dense linears the weight map keeps
  until the load ends (0.89 GiB), which `loadWarmupBytes` adds through `dflash_load_transient_bytes` (stamped with the
  other drafter terms). Measured on the real drafter: resident 760.8 MB, peak over before 1712.4 MB; the formula is
  never under either, within 0.04%.
- **A request bills the MTP head only when it runs MTP** (`requestBillsMtpHead`, the `requestSpecModes` verdict): a
  DFlash2 request never builds the head's cache, state or QSA ring. `enable_drafter:false` hands the request back to MTP
  and bills the head. The head weights and coarse lm_head stay billed at load for that hand-back. The load-time and
  auto-context bills (`mtpHeadDefaultOn`, `sessionBytesPerToken`) drop the head terms when Qwen's default round is DFlash2.
- **The cold-load gate stamps the drafter's request terms before reading the serving bill** (`coldServingBillBytes`), the
  same config the load preflight reads after `stampDflashRequestTerms`; unstamped, the gate missed the DFlash2 per-request
  term (0.48 GB at chunk 4096).
- Qwen drops disabled native MTP tensors before the transformer binds them. An auto context bills no context term: it
  is pinned after the load from what the weights leave. Their default KV remains affine 8-bit.
- An explicit launch or model-settings context is checked before loading resident Sushi Qwen and MiMo EXL3 packs.
  `loadServingBill` uses `prefillNeededAtChunk`, including KV, rings, recurrent/history state, MTP and prefill scratch.
  It prices an explicit prefill width at that width after the architecture's cap; otherwise it proves the per-request
  floor. A refusal reports the largest context that fits the same bill and available memory, capped at
  the model's context limit. Existing active MLX buffers are subtracted from the GPU working-set limit before this
  check. Cold-load HTTP 503 responses retain the numeric maximum through the transient unloaded reset; retry and
  successful load clear it.
- Request-time memory refusals for these packs also report a maximum context using the request's actual KV width,
  effective chunk, MTP choice and warm-prefix credits. The connection-thread 400 and a later scheduler refusal
  carry the same numeric diagnostic; generation error mapping consumes the slot's diagnostic on its connection
  thread so it cannot leak into another request.
- **A streamed load bills what it holds** (`streamedLoadRequirementBytes`): resident trunk, MTP and vision + the expert
  cache + the whole-layer fill union + the bounce buffers + the MLX pool's decode cap (DeepSeek, `ModelConfig.expertPoolBytes`, 1 GiB) + the net; `expertCacheFitForLoad` separately proves the
  planned KV against the wired limit. GLM allocates KV row by row per request, so its load bill carries no context
  term; request admission bills it.
- GLM evicts its hot cache to admit (`admissionEvictsHotCache`), and its context sizer reserves no cache. With the
  prefix cache on, the inference thread's bill adds the KDA checkpoints a prefill holds (up to 9 x 147,619,840 bytes,
  from the capture schedule the generator runs; 2 where SSD-only storage writes them through and frees their RAM
  copy, `WarmPrefix.released`), one assistant window and the RAM tier's row copy (the SSD writer's
  host bytes are not billed, below), and keeps fewer checkpoints where they do not fit, never refusing for them
  ([engine-prefix-cache](glm5-prefix-cache.md#glm)).
- `modelDiskBytes` bills the shards the INDEX names; an index that names NO shard on disk is STALE (every shard
  loads, one warning). Every size sum stats THROUGH symlinks (HF-cache models).
- Load-time bills run INSIDE `Scheduler.init` ([qwen4-qsa-long-context](qwen4-qsa-long-context.md)).
- **The kernel unwires a freed Metal buffer asynchronously** (~0.5 s for 50 GB): an unload and an eviction-before-load
  wait until most of the freed bytes left the wired set (`waitForUnwire`, bounded at 3 s), or the next preflight reads
  them as taken (45 GB free where 95 GB was a moment later).
- **An MTP head rides the checkpoint's shards**: the bill adds nothing for it beyond the coarse rerank copy.
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
admission bill at 1248 tokens, which replaces warmup and net once it is larger):

| pack, settings | weights | warmup or context | net | bill | peak | slack |
|---|---:|---:|---:|---:|---:|---:|
| Flash-Next 2.6bpw, auto or `--ctx-size 1248` | 44.20 | 1.25 | 1.00 | 46.45 | 44.53 | 1.93 |
| Flash-Next 2.6bpw, `--no-vision` | 43.37 | 1.25 | 1.00 | 45.62 | 44.53 | 1.09 |
| MiMo 2.3bpw, auto | 89.65 | 0.25 | 1.00 | 90.90 | 89.88 | 1.02 |
| MiMo 2.3bpw, `--ctx-size 1248` | 89.65 | 1.79 | — | 91.44 | 89.88 | 1.56 |
| GLM 2.4bpw, auto or `--ctx-size 1248` | 89.76 | 0.52 | 1.00 | 91.28 | 90.16 | 1.12 |
| Flash-Next 2bpw streamed, `--ssd-budget-gb 18 --ctx-size 66000 --qwen-gdn 16` | 5.57 | 13.69 | 1.00 | 20.27 | 19.07 | 1.20 |

Flash-Next's extra slack is the vision tower: it is billed at load for the first image and not resident until then.
MiMo at 1248 tokens bills a request's prefill scratch (1.79) above its warmup. The streamed warmup is its cache, fill
union and bounce buffers. Every bill is above its peak; the 1 GiB net is what separates them where nothing else does.
The net does not stack on a context bill: no request allocates during the warmup. The 64 GB recipe (2.6bpw, kv8, 250000,
59,000 MB wired) runs `--qwen-gdn 16` and has 5.1 GiB of slack at load and 1.83 GiB for a cold 250k request at the 4096
width admission picks; the f32 default leaves 4.0 and 0.77 GiB (bill arithmetic on the shipped config fixture,
2927817e against 3adb8002). With the SSD tier up the request holds 3 checkpoints, not 16
([Checkpoints in the bill](#bill-checkpoints)): its 4096-width bill is 10,954 MiB at bf16 and 11,157 at f32, which
leaves 2.7 and 2.5 GiB.
Previous bills for the same boots: 46.20, 91.65, 91.91 and 7.27 GiB (the streamed one was 12 GiB under its peak).

Other expert layouts of these archs bill the same terms; their warmup is not measured on this box.

## Context and chunk

- **Auto-context is PINNED at load** (`pinAutoContext`, one 93% margin on the memory ceiling for every model, then
  capped at the checkpoint's own maximum, un-margined); ask `getEffectiveContextLength`. It bills KV at the CONFIGURED
  width and activations ONCE.
- **Flash-Next's auto-context is the largest context its request bill admits at the floor rung** (`memoryContextAtChunk`
  inverts `prefillNeededAtChunk` over the headroom admission reads: ceiling less resident bytes; `billedAdvertisedContext`). Within `MEASURED_NET_MAX_TOKENS` that is the advertised
  context, no 93% margin on top: the bill is the one reserve. Past it the 93% margin stays. Sizer and admission were
  two formulas: a 32 GB M1 Max advertised 71,680 tokens and refused past 20,549. The other archs keep `contextForBudget`.
- The prefill CHUNK is a machine decision (`resolvePrefillChunk`, ladder 8192→512 at ≤ a quarter of the serving
  budget). `--prefill-chunk` pins it off the per-request ladder; on the ladder it is the widest rung. `prefillMemoryNeeded` takes STORED and SCORED widths as two parameters.
- GLM's load-time pin, the width its advertised context is billed at, is the widest rung up to 2048 that advertises
  as much context as 512: the `max_safe_context` bill picks it, not a quarter of free memory
  ([glm5-arch](glm5-arch.md#memory)).
- RAM retention is off by default and the SSD tier on, so every model bills as SSD-first: a ringed arch with a disk tier
  bills its restore's coexistence (`oldBuffersInEvalWindow`) ([engine-prefix-cache](engine-prefix-cache.md#defaults)). Context sizing reserves nothing
  for an idle cache (`ctxSizingCacheReserve` is 0 with RAM off). A model whose tier did not come up bills as without one
  (`server.diskTierOn`).
- A ringed (MiMo) hot entry bills its ring checkpoints at their real bytes in `kv_bytes`: up to eight until its flush, the
  newest two after it when an SSD tier holds the rest ([engine-prefix-cache](engine-prefix-cache.md#basics)). The
  eight-checkpoint term in `oneSessionEntryBytes` sizes only a RAM tier with no SSD tier (a ringed arch with one sizes
  through `ssdFirstSessionKvBytes`: the session plus `slotRingBytes`, the slot's own six). SSD-only retains none, and
  the writer's ring files ride its permit, outside the wired limit (below).
- **Admission does not bill the SSD writers' host bytes** (`admissionAvailable` = ceiling less MLX's bytes): they are
  pageable host memory, outside the wired limit a Metal OOM is measured against, and the ceiling's 1 GiB margin under
  that limit is the one reserve. Billing them too took 1 GiB of context off every request for no crash it prevented.
- **The writer's permit is a 128th of host RAM, 128 MiB to 1 GiB** (`kv_disk_writer.permitForRam`, set once at startup
  into `server.disk_writer_permit_bytes`, which arms every writer). A prefill stages one blob per chunk and the SSD
  drains it before the next, so a larger permit buys no speed and on a small Mac crowds macOS's share of RAM.
- The ungated hot-cache ask is zero when RAM retention is off, `--prefix-cache-entries 0`, or the model's hot cache
  never loads (`HotPrefixCache.shouldUse`); otherwise it is `--prefix-cache-mem`.
- A per-request arch (`perRequestPrefillChunk`: qwen4_exp, the ringed mimo_v2, glm5_next, deepseek_v41) prefills every
  request from its start width (`prefillStartWidth`: 4096, 2048, 2048, 2048; an explicit `--prefill-chunk` sets it instead). Each
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

<a id="bill-net"></a>
### The request bill's net

- `prefillMemoryNeeded` sums its terms and adds a net: a quarter (`uplift_16ths` 20) for every arch, none (16) for
  Flash-Next up to `MEASURED_NET_MAX_TOKENS` (86,016), where its summed terms alone measured at least 1.1x the live
  peak (the bracket test's floor). Past it the terms were fitted at 300k+ against measured
  steps (`the deployed pack's 8192 step...` and the 364k and 64 GB recipe tests), so the quarter stays; the README
  tables and recipes at 128k and above move only by the checkpoint share below.
- Within the range the summed terms ran 4% to 13% over the live peak, which is why a sixteenth is enough. **The SSD
  tier's checkpoints were real memory**: 34k tokens at the 512 rung peaked 2,284 MiB with the tier on and 1,318 off,
  the billed 897 MiB of checkpoints. A landed checkpoint now leaves RAM ([below](#bill-checkpoints)); the table's live
  peaks are from before that. Measure with the default tier; off is a dev-only reading.
- The forward is not at the gate's width: it starts at 2048, steps down by headroom, and its peak was flat at
  1,287 MiB from 2.8k to 16.9k tokens, while the engine priced that first chunk at 2,631 MiB.

A lean Flash-Next pack (27.4 GB resident), M1 Max 32 GB, `iogpu.wired_limit_mb=30000`, 4a1f1f1b plus the
preflight commit with the diagnostic counters, ReleaseFast, `taskpolicy -a`, GPU lock, no baseline rerun. `--qwen-gdn 16
--no-vision --no-mtp --prefill-chunk 512`, SSD tier on, one fresh prompt per boot, 16 new tokens; peak = `peak_bytes` less
`active_bytes`, MiB:

| tokens | live peak | summed terms | quarter net (before) | sixteenth net (now) |
|---:|---:|---:|---:|---:|
| 2,857 | 1,117 | 1,265 | 1,581 | 1,347 |
| 8,505 | 1,774 | 1,946 | 2,433 | 2,075 |
| 16,874 | 2,043 | 2,125 | 2,656 | 2,272 |
| 33,958 | 2,228 | 2,496 | 3,120 | 2,682 |
| 59,314 | 2,822 | 3,036 | 3,795 | 3,278 |

Final binary, no flags but `--no-vision --no-mtp --qwen-gdn 16` (1,842 MiB available): advertised 26,624 tokens, a
26,426-token prompt admitted at a bill of 1,781 and peaking 1,542 over the weights; `--kv-quant 4` advertised 32,768, a
32,556-token prompt billed 1,723 and peaking 1,430. A prompt past the advertised context is refused by name. The
tests `Flash-Next's request bill brackets the peaks measured...` and `the advertised Flash-Next context is the
largest one the request bill admits` hold both.

On 5cd9f946 (checkpoints leave RAM) the same five lengths peaked 935, 984, 1,196, 1,381 and 1,976 MiB against bills
of 1,168, 1,300, 1,497, 1,907 and 2,503 (the sixteenth net; the terms alone are 1.18x to 1.30x): the bracket test's
pins. On 5cd9f946 plus the unbilled writer, no net to 86,016 and no margin on top (2,866 MiB available), the same
recipe with no `--prefill-chunk` advertises 80,896 tokens; a cold 80,638-token prompt was admitted at a bill of 2,855
and peaked 2,189 over the weights (1.30x), the writer never waiting. A cold 74,734 peaked 2,112 against 2,722. Warm
turns extending a cached prefix to 70k, 84k and 90k added 1,270 to 1,300 MiB over their start against bills of 2,322
to 2,542. The bill's slack is in its transient terms (the QSA score sheet, 439 MiB at 75k, the KV dequant, the MLP
envelope, the 512 MiB floor), and grows with the prompt: 330 MiB at 59k, 600 at 75k.

<a id="bill-checkpoints"></a>
### Checkpoints in the bill

A Qwen request's GDN checkpoints are stored on the SSD tier once their file lands, so the bill counts the states the
request holds, not the ones it captures (`server.ssmResidentCap`, `transformer.SSM_RESIDENT_MAX`):

| storage | held per request | why |
|---|---:|---|
| SSD tier up, with or without a RAM tier (the default) | 3 | a write-through prefill holds 2 (the previous newest until its file is written, and the one just taken); a request it cannot serve is thinned to 2 and holds a third while one is captured |
| RAM only, or the tier declined | up to `--ssm-checkpoint-max` | nowhere else to restore from |

The write-through frees the older state the way GLM's does (`DiskTier.settleCheckpoints`: wait out its file, then
`release`), so the bound is hard, not "whatever has landed". The scheduler thins a request the write-through cannot
serve (a span under one disk chunk) to two listed checkpoints (`qwenCheckpointCap`) rather than
under-bill it. A RAM tier's hot entry keeps the newest two of those held states, so it adds nothing to the request's.

An image-bearing request arms the write-through like any other since media-aware persistence
([engine-prefix-cache](engine-prefix-cache.md#checkpoints-on-ssd)). Qwen's bill does not move -- it was already 3 --
but such a request now really holds two states instead of three. GLM's bill DROPS: `glmCheckpointsRelease` rides the
same gate, so a media turn bills `glm5_prefix.checkpoint_ram_held` (2) KDA checkpoints where it billed the prefill's
peak (up to 9), about 1.2 GiB less at and above ~18,432 tokens once the branch's 5/4 is applied. No model's advertised
or auto context moves: context sizing bills checkpoints from an EMPTY `WarmPrefix`, where Qwen's term is the resident
cap already and GLM's is zero.

Bill at the 512 rung, kv8, 16 new tokens, no MTP, the lean Flash-Next geometry (`qwen4MeasuredTestConfig`), MiB, before
this change (RAM only is that bill: the main before it ignored the tier) against after (one scratch test, ReleaseFast). Stride is the default chunk, so a
prompt captures a checkpoint every 8192 tokens up to 16:

| tokens | `--qwen-gdn` | RAM only | RAM + SSD, SSD only |
|---:|---|---:|---:|
| 8,192 | 16 / 32 | 1,234 / 1,348 | 1,234 / 1,348 |
| 32,768 | 16 / 32 | 1,998 / 2,285 | 1,879 / 2,051 |
| 65,536 | 16 / 32 | 3,007 / 3,523 | 2,649 / 2,821 |
| 131,072 | 16 / 32 | 5,704 / 6,784 | 4,792 / 4,995 |

The advertised auto-context of the 32 GB recipe (wired 30,000 MiB, `--no-vision --no-mtp --qwen-gdn 16`, kv8, that lean
pack's 27,378,205,274 resident bytes) went from a raw 28,995 tokens to 31,532 (advertised after the 93% margin and the
1,024 step: 26,624 to 28,672). With the writer unbilled and no net or margin within the measured range it advertises
80,896 ([above](#bill-net)).

Measured on the M5 Max (Sushi-2.6bpw, `--qwen-gdn 16 --no-mtp --no-vision --prefill-chunk 512`, SSD tier on, one fresh
prompt per boot, 16 new tokens, `taskpolicy -a`, GPU lock; peak = the request's `peak_bytes` less idle `active_bytes`,
MB; 8k reads the sampled `active_bytes`, the load peak hiding the request's; the before column is one run of the main
before this change, not repeated): 8,227 tokens 1,629 before and 799 after, 36,152 tokens 2,259 and 1,371, 66,902
tokens 2,770 and 1,882. That is 830 to 888 MB, 14 to 15 checkpoints, more than the 13 the bill drops at those
lengths, so the bill's lower bracket against the live peaks holds after it (the bracket test subtracts only the billed
share from the M1 peaks). The upper bracket moves from a quarter to 30%: the freed share was billed exactly, so the
bill's fixed slack is a larger part of what is left. At the step-0 shape (`--prefill-chunk 256`, 11,402 tokens, 16
checkpoints captured) the request peaked 105 MB above a run with the SSD tier off, 988 MB before.

<a id="recipe-64gb"></a>
### The 48 GB and 64 GB recipe contexts

The README's 64 GB `--ctx-size` values are checked with the engine's own full-context admission bill
(`prefillNeededAtChunk` through the per-request ladder) at a 59,000 MB ceiling: a prompt that fills the context, MTP
on, `--mtp-head-kv-quant`, the 1 GiB hot cache pinned (not evictable). No live boot: the wired limit has only test
seams (`wired_limit_mb_override`, `static_ceiling_override`), so the bill is computed in a scratch test: here at `ed8fe174`,
`--qwen-gdn 16`, the SSD tier on, the largest admitted in 2000-token steps with 32000 new tokens, and
available = 59,000 MB less the pack's 49,784,531,962 resident bytes (trunk, vision tower, MTP head and its rerank copy)
less the hot cache.

| pack, KV | `--ctx-size` | bill (MiB) | width | available (MiB) | weights + bill (GiB) | largest admitted |
|---|---:|---:|---:|---:|---:|---:|
| Sushi-STQ-2.75bpw, kv8 | 250000 | 9135 | 2048 | 10497 | 55.3 | 392000 |
| Sushi-STQ-2.75bpw, kv4 | 450000 | 10004 | 2048 | 10497 | 56.1 | 654000 |

The 48 GB recipe is the same check at a 43,000 MB ceiling (probe at `eab60060`, weights 37,552,413,730 bytes):

| pack, KV | `--ctx-size` | bill (MiB) | width | available (MiB) | weights + bill (GiB) | largest admitted |
|---|---:|---:|---:|---:|---:|---:|
| Sushi-2bpw, kv8 | 131072 | 4993 | 512 | 6163 | 39.8 | 182000 |

The README memory table is the billed resident weights (trunk, vision tower, MTP head and its rerank copy; GLM with its
assistant) plus this full-context bill at the 512 rung (`--max-tokens 0`, `--ctx-size` the context); the prompt cache is
not counted. The bill is the same for every Flash-Next pack: 4993 / 7913 / 13113 / 23513 MiB at 128k / 256k / 512k /
1M tokens at `--qwen-gdn 16` (KV 2080 / 4160 / 8320 / 16640 of it), and 203 MiB more at each size at the f32 default;
with the tier up the request holds 3 checkpoints ([above](#bill-checkpoints)). Before that it was 5905 / 8825 / 14025 /
24425 and 1080 more (the probe at `eab60060`, reproduced at 4b919abb). Its max-context column is the largest multiple of 8192 tokens whose
weights + 512-rung bill + 256 MiB spare fits each wired limit, capped at 1M (kv8 and kv4); the 48 and 64 GB rows run at
`--qwen-gdn 16`, the 96 and 128 GB rows at the f32 default.

The ladder widens the chunk until the bill nearly fills what is available, so the spare in a row is small by
construction; "largest admitted" (2000-token steps) is where even the 512 rung stops fitting. The ceiling assumes
the full limit is reachable: on a real 64 GB Mac the free-RAM term can bind lower (`currentGpuMemoryCeiling`).

## Admission

- One `[admission] needed=… available=… reclaimable=… width=… verdict=…` line per decision.
- An explicit `--prefill-chunk N` caps the per-request ladder, never pins it: N if it fits, else the widest rung
  below N that fits, down to 512; refused only when that floor does not fit. `requestPrefillPick` is the one rule for
  the bill and the scheduler, `generate.requestPrefillChunk` the width both run; the hot-cache clamp and a streamed
  load prove that floor (`perRequestFloorWidth`).
- A long prefill evicts the hot cache on the INFERENCE thread to be admitted (`evictLruToAdmit`) on every arch where
  `admissionEvictsHotCache` holds (qwen4_exp and the ringed mimo_v2; the connection thread's `creditedAdmissionBill`
  and the scheduler's `admissionPassArmed` read that one predicate), crediting only
  provably reclaimable bytes; `PrefillDoesNotFit` → 400 by name. A warm share that does not fit is first taken
  over (`checkoutRestored`: its append donates, so the restored rows are not billed twice).
- **DeepSeek STQ bills a request's state, not its prefill working set**: the working set comes out of decoder-layer
  expert pools each prefill forward releases when its need passes the live headroom, refused before anything is released
  where the pools cannot cover it; its transient cap (`Config.chunkCap`, the pool inside it) is that working set; admission bills that term only past the lendable bytes, plus a decode forward's
  working set and the decode pool parked in MLX's cache (1 GiB, outside the active bytes admission reads; counted once, in the bill, and taken off the live headroom), and SSD-first its restore points' in-flight bytes ([dsv41-memory](dsv41-memory.md#borrow)).
- DeepSeek's resident bill (`dsv41_source.sumBytes`) is page-rounded per weight mapping: a tensor from 1 MiB, and a DSpark stage's nine ragged
  regions each ([dsv41-memory](dsv41-memory.md#owned-weights)).
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
- **A vision encode is billed before it runs** (`towerFitFault`, `server.visionEncodeBill`), for the blocks the prefix
  restore leaves to encode, never the whole history: the largest block's tower
  scratch (`qwen_vision.encodeScratchBytes`, fitted >= 25% over the measured peak) plus every block's float32 pixels
  and three bf16 copies of its soft-token rows (group outputs, video concatenation, request concatenation); past what
  the GPU has left it is a named 400. The tower evaluates per block, so the peak is one block's f32 score sheet
  (heads x N^2) and rows; table in [qwen4-arch](qwen4-arch.md#vision-tower).
- **A vision encode is checked on the inference thread** (`runVisionEncode`, `vision_encode_available`), right before
  the tower runs and after the restore: a prefill's cache growth or an earlier encode's resident output may have taken
  the headroom, and the refusal is a named 400 (`MediaRefused`).
- **A streamed `--vision` load bills its tower in the ssd budget and proves its largest image beside it**: the load
  admits `budget + planned KV + largestImageEncodeBytes <= wired limit`; requests are still billed live as above
  ([engine-expert-streaming](engine-expert-streaming.md#vision)).
- **DeepSeek-V4.1's tower is billed in that term**: `dsv41_source.visionBytes` sums the pack's `vision.*`, `aligner.*`,
  `image_*` and each layer's `bias_vl` as stored (bf16 tower 970,536,960 bytes plus 40 f32 biases of 1,536 bytes on an STQ pack), so the
  `[expert-stream] ssd budget` line reads `vision 0.97 GB` by default (`0.00` under `--no-vision`, or with `vision: false`);
  a load without the tower bills 0 and leaves those tensors unread. A request's encode is `dsv41_vision.encodeScratchBytes`
  (activations of the widest block, the 512-query score sheet and the aligner, a bound for the 9,198 patches a 1024-token
  span can hold) plus the f32 patches and three copies of the 5120-wide span rows.
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
13/10 s (`mlx_load_safetensors` → sushi's own aligned reader).

Landing A/B of the merged loader (`39e546a4` vs its base `d28a87ee`, first run then four reruns; `serve` defaults, `taskpolicy -a`,
one GPU lock per boot, M5 Max 128 GB, 2026-10-10). To `[kv-cache]` is the weight-upload phase, Listening the full
boot, Warmup the first-prompt weight warmup, Compressor the memory-compressor growth during the load:

| Pack        | Arm  | To `[kv-cache]` | Listening | Warmup  | Compressor |
| ----------- | ---- | --------------: | --------: | ------: | ---------: |
| Qwen 4bpw   | main | 5.36 s          | 8.30 s    | —       | +4.3 GB    |
| Qwen 4bpw   | PR   | 7.27 s          | 11.65 s   | 949 ms  | +0.2 GB    |
| GLM 2.4bpw  | main | 14.25 s         | 22.18 s   | 6219 ms | +57.3 GB   |
| GLM 2.4bpw  | PR   | 9.47 s          | 12.23 s   | 1427 ms | +6.7 GB    |
| MiMo 2.3bpw | main | 14.95 s         | 25.67 s   | 843 ms  | +56.1 GB   |
| MiMo 2.3bpw | PR   | 9.56 s          | 15.06 s   | 859 ms  | +4.6 GB    |

Qwen 4bpw, the four reruns: main 5.26/5.25/5.22/5.27 s to `[kv-cache]`, listening 10.00/9.28/8.99/9.14 s, warmup
543 then 80/81/80 ms, spec-warmup 3.04/2.73/2.67/2.66 s, compressor +3.5 to +4.0 GB; PR 6.66/6.74/6.58/6.69 s,
listening 10.67/9.96/9.72/9.89 s, warmup 683 then 88/85/86 ms, spec-warmup 2.82/2.63/2.55/2.61 s, compressor
+0.2 then 0. Qwen pays ~1.4 s of upload (per-tensor page-mapped reads instead of one bulk MLX read) for a
compressor at ~zero; GLM and MiMo win on time and compression alike, ~57 GB of load-time compression down to <7 GB.

MLX lazy safetensor Load nodes
kept one descriptor per shard alive until evaluation, so the 566-shard affine pack
exceeded macOS's default 256-handle terminal limit despite a successful memory
preflight. The bounded reader also reports descriptor exhaustion separately from
missing files. A 300-shard regression runs with a 128-descriptor soft limit.

## Transient cap (billed for DeepSeek-V4.1)

MLX can be capped on the bytes it allocates itself, imported weights left out
([engine-mlx-gotchas](engine-mlx-gotchas.md#transient-cap)): no flag, off until a DeepSeek phase installs it. Each
phase installs `active - imported + (its bill + the decode pool)` and the bill IS that cap: `Config.forwardCap` for a
prefill chunk or a decode and verify forward, `visionEncodeBill` for an image encode
([dsv41-memory](dsv41-memory.md#transient-caps)). A refused allocation fails the request as `TransientCapExceeded`,
a named 503. The pool is inside the cap, so a prefill's pool is no separate bill. A request's end lifts it.
`SUSHI_DSV41_TRANSIENT_PEAK=1` logs the peak per phase, which the caps were sized from.

## Observing memory

`/props` reports `active_bytes`, `memory.cache_bytes`, `batching`; RSS is blind to Metal.
