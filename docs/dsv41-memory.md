# DeepSeek-V4.1 serving memory

How `deepseek_v41` spends memory under CED prefill: the ledger, the per-layer cache split, the 1M-token state, the
prefill that borrows decoder pools and the fully resident load. Index: [CLAUDE.md](../CLAUDE.md#docs-index); related:
[dsv41-arch](dsv41-arch.md), [engine-expert-streaming](engine-expert-streaming.md).

## The ledger (an STQ pack, `--ssd-budget-gb 96` = 103.08 GB)

| term | GB |
|---|---|
| trunk (resident STQ banks, embedding, head) | 4.78 |
| DSpark stages (resident) | 3.92 |
| union workspace (one layer's 384 experts at the widest layer) | 3.70 |
| bounce + selected | 0.60 |
| expert cache | 90.09 |

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
- The per-request state (890 B a token, 0.93 GB at 1M) and the prefill transient are not in the SSD budget. They come
  from the serving budget (the GPU ceiling less everything resident), billed per request.

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
hold RAM ([below](#borrow)), so a 1M request bills 3.69 GB.

<a id="borrow"></a>
## Admission bills the state: the prefill borrows the decoder pools, restore points go to SSD

Rules: a prompt's prefill working set must not limit the admitted context, and the prefix cache is SSD with RAM 0.

**What admission bills** (`server.prefillNeededAtChunk`, x 5/4): the request's state (`(seq + max_tokens) x 890 B` and
the rings), the restore points in flight to SSD (below), and a decode or verify forward's working set
(`Config.decodeTransientBytes(17, seq)`, the prefill formula at 17 rows; 1.06 GB at 1M). The prefill working set is billed
only where the decoder pools a prefill may still release (`ModelConfig.expert_borrow_bytes`, published live by the
engine) do not cover it. At 1M: 3.69 GB in all.

**The borrow** (`src/dsv41_borrow.zig`, `Engine.borrowLayer`/`restoreLayer`):
- Before every forward wider than 17 rows, its need, `(transientBytes(rows - 511, seq) + rows x 890) x 5/4`, is checked
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

The 1M bill with the prefill borrowed is 3.69 GB (state 0.93, writer in flight 0.93, rings, decode floor 1.09, x5/4).
`--drop-expert` ([dsv41-arch](dsv41-arch.md#drop-expert)) frees bytes from the experts to make this fit: on an STQ pack
whose experts total 119.6 GB, dropping 8 per layer frees 2.1-2.5 GB and dropping 16 frees 4.3-5.0 GB.

- The near-resident arm with DSpark is the streamed path at the largest budget whose 1M bill fits (94% of the expert
  bytes cached). It keeps the 3.70 GB union and reads its misses from SSD; the lossy pick (`--expert-pick-tolerance`)
  swaps those misses on grouped layers.
- Every arm wires more than `vm.global_user_wire_limit` (116.82 GB); a decode measured at half speed past it once.
