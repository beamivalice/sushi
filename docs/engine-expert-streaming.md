# Engine: expert streaming (`--ssd-budget-gb` / `--expert-cache-gb` / per-model `ssd_budget_gb`)

How a checkpoint whose routed experts do not fit in memory is served from SSD: the budget ledger, the per-layer LRU,
the zero-copy slab I/O and the correctness bars. Read this before touching `src/expert_stream.zig`,
`src/expert_io.zig` or `src/glm5_stream.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-exl3-experts](engine-exl3-experts.md),
[engine-memory-admission](engine-memory-admission.md), [arch-qwen4exp](arch-qwen4exp.md),
[arch-mimo-v2](arch-mimo-v2.md), [arch-glm5-next](arch-glm5-next.md), [server-lifecycle](server-lifecycle.md#settings).

## Code map

| File | Role |
|---|---|
| `src/expert_stream.zig` | `ExpertStore` spans, per-layer group-exact LRU + union bridge, zero-copy slabs, `BudgetLedger` (`--ssd-budget-gb`), MTP refusal |
| `src/expert_io.zig` | SSD→Metal I/O: F_NOCACHE positioned-read `FillPool`, `PageSlab` epoch leases, verified zero-copy `importSlab` |
| `src/expert_bf16_kernels.zig` | bf16 selected-expert kernels over a slab |
| `src/glm5_stream.zig` | GLM's routed experts over the engine (BF16 or EXL3 slabs), the native teacher's capture budget |
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

- Rate-group (`.gN`) and pruned EXL3 packs are refused by name (`Exl3RateGroupsStreamingUnsupported`,
  `Exl3RaggedStreamingUnsupported`) and serve resident: an open gap, not the design.
- Vision is not loaded under streaming: every streamed model serves text only.
- GLM's routed experts go through `glm5_stream.Stream`: BF16 source experts through the clamped `gather_mm`
  composite, EXL3 banks through the resident `routedExl3` dispatch; router ids become slot ids, scores, clamps and
  the FP32 reduction are untouched, and each layer completes before its slabs are reused.
- A streamed GLM keeps no speculative state: its DFlash2 assistant stays unloaded and an explicit `--drafter` is
  refused (`GlmStreamingSpecUnsupported`).
- A streamed GLM serves the kv8 or the BF16 latent; its request bill adds the fill peak, and its planned KV counts the
  latent at the KV width plus the pooled index.
- The native BF16 teacher capture (`glm5_kld_capture`) keeps its own budget (`glm5_stream.captureBudget`): trunk,
  request reserve (at least 8 GiB), the full union slab, bounce buffers and at least one slot per MoE layer, the lazy
  trunk bounded before any tensor is evaluated, one request, chunks of at most 512.

## Budget

- `expert_stream.budgetLedger`, one `[expert-stream] ssd budget` boot line. `--ssd-budget-gb N` is a TOTAL resident
  target of N GiB = trunk + MTP + the all-experts union workspace + selected slab + bounce; the remainder is a uniform
  per-layer LRU. `--expert-cache-gb` overrides (decimal GB of expert cache).
- Precedence: `--expert-cache-gb` > `--ssd-budget-gb` > setting > `ExpertStreamingRequired` 503 naming all three.
  An explicit launch flag always beats `model-settings.json` ([server-lifecycle](server-lifecycle.md#settings)).
- The load's fit check prices serving at the per-request ladder's floor rung (512), since a request picks its own
  rung against free memory; a pinned chunk is priced as given; an explicit
  `--prefill-chunk` only lowers the floor the load proves.
- Admission `budget + planned KV <= wired limit`; the refusal names the `iogpu.wired_limit_mb` that would admit.
  Planned KV is the session bill per token (KV at its width plus Qwen's QSA history or GLM's pooled index) times the
  planned context. Under `--no-mtp` the head is not loaded at all.
- An imatrix capture's accumulators live in GPU headroom that admission reads: budgets for capture runs drop
  (MiMo 100 → 94 GB; Flash-Next 96 → 80 GiB no longer admits higher under current bills).

## Cache policy

- `GroupCache`: plain per-layer LRU, prefill misses at MRU, every HIT of a route touched before any admit, surplus
  misses fall to the union workspace. Batched decode rides the union path.
- **MTP needs an explicit `--mtp`** (qwen4_exp): the head and its own routed experts load resident
  (`ModelConfig.stream_mtp_head`, billed in the ledger), verify rows take the streamed decode path, and the PLE
  window rolls back like a resident verify. The engine default resolves off (`[mtp] off (streaming; default)`), a
  `model-settings.json` `mtp: true` is dropped with a warning, and `enable_mtp:true` without a loaded head is a named 400.
- **Load-time cache warm**: preload the lowest expert IDs into `floor(0.8 * slots_per_layer)` slots per MoE layer
  before kernel warmup and readiness, within the existing budget. These are ordinary LRU entries, not predicted
  routes; dense prefix layers are skipped.

## Decode schedule

- A streamed layer is latency-bound: one GPU→host read of the router ids per layer. The layer's expert compute is
  submitted with `mlx_async_eval` as soon as it is built, and at decode widths (<= 16 rows) the shared expert is
  submitted between the ids and the blocking read, so the GPU runs it during the host round trip. Same ops, same
  order; output is bit-identical.
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
- Output depends on cache state, so it is not reproducible across runs or prompt histories. `kld capture` refuses a
  non-zero tolerance: the teacher is exact routing. Numbers: [quality-kld](quality-kld.md#lossy-expert-pick).

## I/O

- `FillPool` = F_NOCACHE + F_RDAHEAD 0 positioned preads, fd cache validated by (dev, ino, size, mtime), spans sorted
  and coalesced to 64 MiB, page-aligned bounce otherwise.
- `PageSlab` epoch leases (`free → filling → ready → leased → readers_complete → reclaimable`, CPU writes only in
  `filling`).
- `importSlab` = `mlx_array_new_data_managed_payload` verified by pointer identity (`ExpertSlabImportCopied`
  refuses). Bench: `tests/ssd_fill_bench.sh`.
- **MLX releases an IMPORTED host buffer asynchronously**: `mlx_array_free` returns BEFORE the payload deleter runs;
  wait for the deleter (`SlabOperand.destroy`), leak (counted, logged on the breakdown line) rather than unmap what
  MLX still holds.

## Compute

- bf16 checkpoint → `expert_bf16_kernels` (`downKernelPreferred(rows) = rows >= 2`: the in-dispatch k-reduction tail
  loses at one row; `SUSHI_EXPERT_BF16_KERNELS=0` restores the `gather_mm` composite).
- Quantized packs → the RESIDENT fused kernels over the slab with remapped ids, bit-identical to the resident load
  (bytes and top-20 logprobs on greedy prompts). A warm quantized forward pays the per-layer barrier, not the fills.

## Correctness bars

- Store-level same-expert byte identity (`real qwen expert store spans and source bytes are exact`); teacher replay
  via `kld compare` (the affine pack is the control); greedy determinism.
- Streamed logits equal resident logits bit for bit through eviction and the union (`GLM serving streams EXL3
  experts ...`); `every real streamed pack and source on this box plans a streamed load` plans each pack on the box.
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
file as per-channel mean squares under their source weight names ([arch-mimo-v2](arch-mimo-v2.md)). The drivers that feed it a corpus live in the private
converter repo. Routed counts reconcile to
tokens x top-k exactly on every layer; the two load-time warmup forwards add a few tokens.

- **Hidden capture** (`SUSHI_HIDDEN_OUT=<abs dir>`): `sushi kld capture` (no prefix cache, no warmup) appends every
  prompt token's residual at each block boundary (`boundary-XX.bin`, raw bf16 [tokens, hidden]; 00 = layer 0's input,
  b = layer b-1's output), then its ids (`tokens.bin`, u32); `forwardMoeWith` only; logits bit-identical.
- Its output files are private: each is created exclusively without following a link, and an existing one is appended
  to only when it is a regular file with one link (a hard-linked or symlinked output is refused, its target untouched).

## Discovery

A dense qwen4_exp checkpoint with a complete streaming index registers as a streaming stub
(`streamingStubMarker`); `/v1/models` carries `streaming`, `streaming_required`, `ssd_budget_gb` at top level and
`input_modalities: ["text"]` when it must stream. Guards: `tests/test_bf16_streaming.sh`,
`tests/test_model_settings.sh` [5].
