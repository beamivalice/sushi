# DeepSeek-V4.1 serving memory plan (proposal)

A phase-aware memory plan for `deepseek_v41` STQ packs under CED prefill. Owner outline:
1. A KV reserve for a 1M context at all times.
2. The encoder's routed experts fully resident for a prefill turn.
3. The decoder's experts swapped in for decode turns.

This doc does the arithmetic, models the hit rates from real routing, and ranks the pieces. Nothing here is built
yet. Index: [CLAUDE.md](../CLAUDE.md#docs-index); related: [dsv41-arch](dsv41-arch.md),
[engine-expert-streaming](engine-expert-streaming.md). What is built and its current numbers: [dsv41-memory](dsv41-memory.md).

## The pack and the ledger today (`dsv41-stq-d4`, `--ssd-budget-gb 96` = 103.08 GB)

| term | GB |
|---|---|
| trunk (resident STQ banks, embedding, head) | 4.78 |
| DSpark stages (resident) | 3.92 |
| union workspace (one layer's 384 experts at the widest layer) | 3.70 |
| bounce + selected | 0.60 |
| expert cache | 90.09 |

Routed experts by layer, rate-sized (shard headers): the encoder (layers 0-19) holds 57.13 GB (6.8-9.6 MB an
expert), the decoder (20-39) holds 62.46 GB (7.1-9.2 MB), 119.6 GB in all. The cache holds 75% of it.

- The ledger allots each layer the same bytes: 234 slots of 9.62 MB, the largest layer average (layer 15), which
  makes 2.25 GB a layer.
- Inside a layer the rate-group pools split that allotment by bytes. A small-expert layer therefore caches more
  experts (layer 0: 331 of 384), and layer 15 caches 234.
- Slots are already rate-sized within a layer. What the plan can change is the split between layers.
- The per-request state (890 B a token, 0.93 GB at 1M) and the prefill transient are not in the SSD budget. They come
  from the serving budget (the GPU ceiling less everything resident), billed per request.

## Routing locality (measured, not assumed)

Source: the layer-major capture's `route-LL.u16`, the native model's top-6 for 1,015,808 tokens in 496 windows of
2048 (prose, chat, code, design, reasoning). The script is `agent-runs/opus-dspre-01/sim/routes.py`; 32 windows
were sampled.

| per-layer, mean over layers | 128 rows | 512 rows | 1024 rows | 2048 rows |
|---|---|---|---|---|
| distinct experts a chunk touches (of 384) | ~130 (100-186) | ~220 (180-289) | ~270 (228-329) | ~315 (276-357) |

| decode: per-layer LRU hit rate at N slots | 96 | 128 | 160 | 192 | 234 | 288 | 384 |
|---|---|---|---|---|---|---|---|
| typical layer | 0.87-0.91 | 0.91-0.94 | 0.93-0.96 | 0.95-0.97 | 0.97-0.98 | 0.98-0.99 | 1.0 |
| worst layers (0, 1, 19) | 0.71-0.78 | 0.79-0.85 | 0.85-0.89 | 0.90-0.93 | 0.94-0.96 | 0.97-0.98 | 1.0 |

What the tables say:
- A 1024-row encoder chunk touches about 70% of a layer's experts.
- A 128-row decoder tail touches about a third of them.
- Decode is forgiving: halving a layer's slots from 234 to ~128 moves its hit rate from ~97% to ~92%.

## Policies, replayed on the routes

Script: `sim/policy.py`. 32 prompts in a row, each a 2048-token CED prefill: chunk 1024, encoder over every row,
decoder over each chunk's last 128. After each prompt, a 256-token decode on the next window's routes. Every layer
runs LRU, and fill is counted at the layer's average expert bytes. Each prompt is a different document, so these
numbers are a pessimistic, cold-ish working set.

| policy | prefill SSD fill per 2048-token prompt | decode misses per token |
|---|---|---|
| A. today: uniform bytes per layer (331..234 slots) | 16.9 GB | 6.2 |
| B. encoder pinned: layers 0-19 all 384 (57.1 GB), decoder the rest (202 slots a layer) | 12.9 GB (-24%) | 7.4 (+1.2) |
| C. phase swap: B for prefill, decoder all 384 + encoder 185 for decode, LRU-driven swaps | 32.4 GB | 14.9 |

- **C loses.** At ~9 GB/s each flip moves ~29.5 GB, about 3.3 s. A 256-token decode never earns that back, and the
  next prefill pays it again. It wins only when a decode runs for thousands of tokens, with the swap done in the
  background (below).
- **B wins only at short decodes (256 tokens); the long-decode table below drops it.** No fill at all in the encoder half of every chunk. The decoder tail misses about 30% of
  its ~130 experts, against a 202-slot LRU.
  - At 9 GB/s that is -0.45 s per 2048 prompt tokens (~0.2 ms a token), about 10% of prefill at the 500 tok/s target.
  - The cost is +1.2 decode misses a token, ~1 ms of a ~65 ms token (-1.5% decode).

## Long decodes: 2k and 20k tokens a turn (the owner's real turns)

Script: `sim/policy2.py`, 12 turns. Each turn is a 2048-token CED prefill (chunk 1024, decoder tail 128), then a 2k or 20k
token decode that reads the capture's next windows in order. Fill is counted at each layer's average expert bytes.
SSD seconds assume 9 GB/s on the critical path, with every miss stalling the token.
- C-bulk loads the whole incoming half at each phase change, as a stall.
- C-bg loads the decoder half in the SSD's spare time during decode (65 ms a token, less that token's misses). The
  encoder half comes back by demand fill on the next prefill.
- O is a static per-layer allotment for decode. Slots are given greedily by decode hits per MB, read from LRU stack
  distances on 48 held-out windows. Layers 0, 1, 19 and 23 get 340-374 slots; layers 16 and 25 get 212-221.

| policy | decode | prefill fill / 2048-token prompt | flip stall / turn | decode misses / token (steady, turn) | SSD s / turn vs B |
|---|---|---|---|---|---|
| A uniform | 2k | 23.2 GB | 0 | 2.64, 3.48 | -2.2 |
| B encoder pinned | 2k | 19.1 GB | 0 | 4.06, 4.92 | 0 |
| C-bulk | 2k | 15.1 GB | 61.7 GB | 5.57, 6.50 | +8.2 |
| C-bg (25 GB in background) | 2k | 40.0 GB | 0 | 5.57, 6.78 | +4.7 |
| O decode-optimal | 2k | 22.3 GB | 0 | 2.32, 3.13 | -3.1 |
| A uniform | 20k | 23.6 GB | 0 | 3.05, 3.12 | -23.4 |
| B encoder pinned | 20k | 19.4 GB | 0 | 4.33, 4.43 | 0 |
| C-bulk | 20k | 15.5 GB | 61.7 GB | 5.95, 6.04 | +25.9 |
| C-bg (25 GB in background) | 20k | 40.4 GB | 0 | 5.95, 6.06 | +22.4 |
| O decode-optimal | 20k | 22.6 GB | 0 | 2.70, 2.76 | -31.5 |

What the table says:
- **Decode runs all 40 layers.** The swap's premise, that decode needs the decoder half, is wrong. C gives the
  decoder layers all 384 slots and the encoder layers 185, so the decode misses move into the encoder half.
  - Layers 0, 1 and 19 are the hardest to cache, so C misses more than B in steady state (5.6-6.0 against 4.1-4.3).
  - Both C variants lose at every decode length. The background loader removes the stall, but the next prefill
    refills the encoder by demand, and that is slower than B.
- **B loses once decodes are long.** It saves 4 GB of fill per 2048 prompt tokens, about 2 MB a prompt token. It
  costs 1.3 decode misses a token, about 10.5 MB a decode token.
  - B wins only when a turn's new prompt tokens outnumber its decode tokens about 5 to 1.
  - At 2k-20k decode tokens a turn, that means 10k-100k uncached prompt tokens a turn.
- **O wins against all of them, but by little.** Decode misses drop 10-12% from A, and prefill fill matches A.
  - That is -0.9 s on a 2k turn and -8 s on a 20k turn: about 0.6% of a turn at ~65 ms a token.
  - It needs per-layer slot counts in the ledger (today one `slots_per_layer`), plus a source for the curves: either
    the pack (calibration routes) or an online ghost-list profile.

### O against A at other cache sizes

Same 12 turns. The table gives steady decode misses per token, A then O, and O's SSD seconds per turn less A's.

| expert cache | 2k: misses A, O | 2k: O - A | 20k: misses A, O | 20k: O - A |
|---|---|---|---|---|
| 60 GB | 9.72, 9.00 | -1.6 s (-7% misses) | 10.31, 9.52 | -15.4 s (-8%) |
| 75 GB | 5.33, 4.83 | -1.2 s (-8%) | 5.93, 5.38 | -11.2 s (-9%) |
| 90 GB | 2.65, 2.33 | -0.9 s (-10%) | 3.06, 2.72 | -8.1 s (-11%) |
| 100 GB | 1.45, 1.19 | -0.8 s (-14%) | 1.72, 1.48 | -6.2 s (-14%) |
| 110 GB | 0.68, 0.43 | -0.7 s (-27%) | 0.80, 0.57 | -5.6 s (-28%) |
| 119.6 GB (every expert) | 0.31, 0.06 | -0.9 s (-63%) | 0.34, 0.00 | -7.0 s (-94%) |

- O's relative gain grows with the cache.
- Under A, a full-size cache still misses. Every layer gets the same bytes, so the 9.6 MB-expert layers can't hold
  all 384 while small layers strand bytes.
- The absolute gain is largest on small caches, where each miss costs more.

### O live (54d44bde, `--ssd-budget-gb 96`, DSpark on, 1024 tokens with `ignore_eos`, 180 s idle before each arm)

| arm | decode at 4k | decode at 16k | verify ms a round (4k, 16k) |
|---|---|---|---|
| uniform, first | 36.4 tok/s | 40.7 tok/s | 157, 135 |
| O (profile, 215..368 slots) | 68.9 tok/s | 63.8 tok/s | 76, 84 |
| uniform, repeat | 68.1 tok/s | 65.0 tok/s | 77, 83 |

- All three arms produce byte-identical text with identical DSpark acceptance.
- **O has no measurable decode effect on these prompts.** Hits are ~99% in every arm, and fill is under 10 ms a forward.
- The first uniform arm lost its time in the end-of-forward GPU wait (77 ms against 17 ms), not in misses. The cause
  is unexplained. A test build ran during that arm's idle; the repeat, with no build, matches O.
- A 4k-16k prompt and a 1024-token decode warm too few experts for O's split to matter at 90 GB. The sim's gain
  comes from long decodes over changing documents.

## Budget per phase at 96 GiB (B, static; kept for the arithmetic, B is dropped)

| term | GB | note |
|---|---|---|
| fixed (trunk, DSpark, workspace, bounce) | 13.0 | unchanged |
| encoder experts, all resident | 57.13 | pinned after the warm load; never evicted |
| decoder experts, LRU | 32.96 | 53% of 62.46 GB, ~202 slots a layer, rate-sized pools as today |
| KV reserve (outside the SSD budget) | 1.0 | state for 1M (0.93) plus rings: billed in admission as today, always admitted (see below) |
| prefill working memory at 1024 (outside) | 7.45 billed today, ~3-4 expected | per-row term 5.5 MB is ~75% attention gathers; the per-span scopes (2052a76a) should cut it. Measure, then refit |

The union workspace (3.70 GB) is needed only when a layer's chunk union exceeds its slots. Under B with CED, an
encoder layer holds all 384, and a decoder tail's union (≤ 190) stays below its 202 slots. Only a non-CED wide chunk
still needs it: all-row logits (`.all`), a calibration capture, a long verify block. It can shrink later to what those
paths need. Not in the first step.

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
## Admission bills the state: the prefill borrows the decoder pools, restore points go to SSD (built)

Owner rules: a prompt's prefill working set must not limit the admitted context; the prefix cache is SSD with RAM 0.

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
## Fully resident, and the wired 125000 ledger (built)

A budget that holds every routed expert beside the trunk and DSpark loads the pack resident
([engine-expert-streaming](engine-expert-streaming.md#budget)): no union workspace (3.70 GB), selected slab or bounce,
every expert read at load, and each layer's picks mapped to slots on the GPU, so a forward never waits on a host read
of the picks. The borrow stays: a prefill past the headroom releases decoder pools, its first release paying for a
union workspace (3.54 GB, the widest decoder layer) that the last restore drops.

MLX bytes at idle: trunk 4.78 GB, DSpark 3.92 GB (the fit124 boots' active bytes less their pools and union), experts as
the pack's headers sum them (119.59 GB at 1.75 bpw). The 1M bill with the prefill borrowed is 3.69 GB (state 0.93, writer
in flight 0.93, rings, decode floor 1.09, x5/4). Ceiling at 125000: 129.15 GB with the 1,317 MB GPU floor the
budget-100 boot saw, 130.00 GB on an idle GPU. Budget = the smallest GiB count that holds trunk bill (5.35, its
host Engram rows included) + DSpark + experts.

| experts | DSpark | budget | held | 1M slack, floor 1317 / idle | memory context, floor 1317 / idle |
|---|---|---|---|---|---|
| 1.75 bpw, 119.59 GB | on | 121 | 128.29 | -2.82 / -1.98 | 101,239 / 384,614 |
| 1.75 bpw | off | 117 | 124.37 | +1.10 / +1.94 | 1M / 1M |
| 1.70 bpw (x1.70/1.75), 116.18 GB | on | 117 | 124.87 | +0.60 / +1.44 | 1M / 1M |
| 1.75, `--drop-expert 8` (2.1-2.5 GB freed, mid 2.3) | on | 118 | 125.99 | -0.52 / +0.32 | 873,375 / 1M |
| 1.75, `--drop-expert 16` (4.3-5.0 GB freed, mid 4.65) | on | 116 | 123.64 | +1.83 / +2.67 | 1M / 1M |

- At 1.75 bpw with DSpark the resident load passes its gate only on an idle GPU (+0.37 GB; -0.47 at the 1,317 MB floor).
- The near-resident arm with DSpark is the streamed path at the largest budget whose 1M bill fits: 118 (93.8% of
  bytes) at the 1,317 MB floor, 119 (94.7%) idle. It keeps the 3.70 GB union and reads its misses from SSD; the lossy
  pick (`--expert-pick-tolerance`) now swaps those misses on grouped layers.
- Every arm wires more than `vm.global_user_wire_limit` (116.82 GB); a decode measured at half speed past it once.

## Phase swap mechanics (if C is ever built)

- **When it flips.** Prefill to decode at the first decode tick; decode to prefill at the first wide chunk of the
  next request.
- **What moves.** ~29.5 GB each way: the decoder's missing 47% in, the encoder's 52% out.
- **Overlap.** At 9 GB/s, 3.3 s. During decode the SSD idles between misses, so a background loader thread can
  bring in ~0.5 GB a token (65 ms) without stalling. That is a full swap in ~60 tokens, while LRU misses serve the
  hot experts first.
- **Mid-swap safety.** A DSpark verify (5 rows) or a request arriving mid-swap needs the slot table to stay
  consistent: the loader fills free slots only, and the inference thread publishes them. This is the same
  single-writer rule as today's fill (the inference thread is the sole MLX caller).
- **Dropped.** Decode reads all 40 layers, so the encoder layers at 185 slots miss more than the decoder layers gain (long-decode table above), and the swap back
  costs a full flip on every prompt.

## Ranked, with what decides each

| rank | piece | expected gain | cost | decided by |
|---|---|---|---|---|
| 1 (done) | Per-row transient refit after the per-span scopes: 1075 to 400 B a row per channel, 12 to 3 score copies; the 1024 reserve at load is 3.9 GB | 1024 at budget 96 (a 2x wider chunk: experts read once per 1024 rows) | one real run per chunk width, then the `row_bytes_per_dim` refit | `[prefill-trace] mem` peaks at 1024 |
| 2 (done) | Fused gather-attention for wide spans (no `[span, 512, 512]` materialization) | measured: a 256-row span 2.7-3.3 ms against 13-16 ms; the core is ~0.23 s of a warm 1024-token chunk (~2.1 s, profiled) | a Metal kernel and its f32-truth proof | ubench vs `attnCore`, then the stage profile |
| 3 (done) | O: decode-optimal per-layer slots | measured: no decode difference against uniform at 4k/16k (68.9/63.8 against 68.1/65.0 tok/s); prefill unchanged | per-layer slots in the ledger and a curve source | decode tok/s at 4k/16k, A then O |
| 4 (done) | Streamed index top-k over key tiles, packed-key decode scores, growth by a sixteenth | the transient bill at 1M fell from 7.6 to 4.3 GB at a 512 chunk (16.3 to 12.1 GB the request); budget 96 now advertises 974848 | `indexStream`, `indexScoresPacked`; byte-identical picks | `SUSHI_DSV41_INDEX_UBENCH` peaks; the live per-chunk peak at long context is still to measure |
| 5 (built) | Restore points straight to SSD | the last 4.5 GB of the 1M bill | shared-row capture, DeepSeek write-through | the live write-through stall at long context |
| 6 (built) | Prefill working set borrowed from the decoder pools | budget up to 117 at 1M (93.7% resident) | engine release/restore, per-forward plan | the live long-prefill peak, decode tok/s after a long prefill against a short one |
| - | B: encoder pinned | +11% prefill at 2048-token prompts, -1.8% decode | ledger split | dropped: loses at 2k-20k decode tokens a turn |
| - | C: phase swap, bulk or background | none | loader, slot publication | dropped: decode reads all 40 layers, so C loses at every decode length |

Build order: 1 now, then the routed-expert prefill and 2. The workspace shrink waits until its non-CED paths are sized.
