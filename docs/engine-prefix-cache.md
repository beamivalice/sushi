# Engine: prefix cache and SSD-first

How prompt-prefix KV reuse works: the hot RAM cache, hybrid (GDN/QSA) restore points, the SSD tier and SSD-first
mode, checkouts and donations, spec state riding the cache, and GLM's native state. Read this before touching `src/prefix_cache.zig`,
`src/kv_disk_cache.zig` or `src/kv_disk_writer.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-kv-cache](engine-kv-cache.md),
[engine-memory-admission](engine-memory-admission.md), [engine-mtp](engine-mtp.md),
[mimo2-arch](mimo2-arch.md#sliding-layers-the-ring).

## Code map

| File | Role |
|---|---|
| `src/prefix_cache.zig` | Hot prefix cache (`--prefix-cache-entries`, `--prefix-cache-mem`) |
| `src/kv_disk_cache.zig` | SSD tier (`--prefix-cache-disk`), its default budget (`resolveDiskBudget`) and startup/usage lines |
| `src/kv_disk_writer.zig` | SSD-first mode's background writer thread |
| `src/restore_dump.zig` | Prefix-cache restore diagnostics (`tests/diff_restore_dump.py`) |
| `src/glm5_prefix.zig` | GLM restore points: KDA checkpoints, MLA rows, their SSD layout ([GLM](glm5-prefix-cache.md#glm)) |

## Basics

- KV reuse via prompt-prefix matching; invalidated after tool calls + pad-only gens (`commitDeclinesPadOnly`: only an
  ALL-pad generation declines); hot cache spills to SSD; RAM invalidation propagates to disk.
- Restore ALWAYS clamps (`truncate(final_len)`); a failed restore hands back an EMPTY cache; every eviction loop has
  a no-progress exit (checked-out entries are unevictable). A restore leaves a token to forward: a one-token prompt
  prefills cold (its full hit restored everything and the empty prefill crashed upstream).
- **Media keys are a CHAIN** (`MediaSpan`: per block, a hash of its pixels, its position and every block before it;
  the entry key is the last). An entry keyed by a request's block k restores up to block k+1, so a turn that appends
  a screenshot reuses everything before it; any other key mismatch shares only the text before the first media row
  (`crossKeyBoundary`). The splice resumes at the placeholder count inside the restored prefix.
- **A restore is not bit-identical on a HYBRID** (≤ 0.047 nats; the chunking class ~0.3 nats top-5 for QSA state) ⇒
  byte-stable greedy needs `--prefix-cache-entries 0`. A hybrid cache hit moves the top logprob ~0.2 nats, so any
  scorer boots with the cache off.
- The always-on SSM snapshot sits 30 tokens BEFORE prompt end; a restored tail inside that window forwards as ONE
  span (`ssmSnapshotBackoff`). Guard: `tests/test_hybrid_reuse_equivalence.sh`.
- **A ringed (sliding-window) entry restores at its end or at one of its ring checkpoints**
  (`KVCache.ringCheckpoint`, `Entry.ring_cps`, up to `RING_CHECKPOINT_MAX` = 8): each ringed layer's
  window + 30 rows at a position (down to the window when the ring holds no more, as one restored off a checkpoint
  does); the slot's own are its restore point, its prompt end and its message marks (`SlotRingCps`). A reply longer than the
  ring's slack compacts it past where the next turn diverges (the previous reply re-renders); the checkpoint's rows
  go under the ringed layers (`restoreRing`) and the usual clamp follows.
  A checkpoint restore of fewer than `RING_RESTORE_MIN_TOKENS` (64) cold-prefills: below it a restore cost more than
  the cold prefill it replaced.
  Below both, `SlidingRingRewindPastWindow` → cold prefill, and the declined entry keeps its recency: promoted, a
  header-only match made the entry a later turn needed the next count-cap victim.
- **A ringed prefill marks the message starts it forwards** (`ringMarkPositions`: `<|im_start|>` past the restored
  prefix and short of the prompt end's reach, at most `RING_MARKS_MAX` = 4, thinned keeping the first and the last):
  each ringed KV write fills a mark it reaches before its compaction drops the rows (`KVCache.ring_marks`), so no
  chunk is split and the forward is unchanged. A new session sharing only another's system prompt and tools
  diverges inside its first user message, below every fork and prompt end, and restores at that mark.
  Measured (MiMo 2.3bpw, kv8, MTP and prefix cache at their defaults, a 12,042-token tools + system prefix, a
  ~300-token first task, 12,346-token prompts, `taskpolicy -a`, lock per boot, busy box, 2026-10-01). One boot of
  63476cd1: the first session cold-prefills in 9,876 ms; the second and third restore 12,037 / 12,042 tokens and
  prefill in 434 / 420 ms. One boot of main 819b4751: the first session cold in 15,682 ms, and the second and third
  cold again (`cached_n` 0) in 14,970 / 14,582 ms. Splitting a chunk at the boundary instead would cost ~0.4 s per
  split on MiMo (the fixed per-chunk cost the 2025- and 4096-row prefill meter rows imply).
- **Ring checkpoints thin span-preserving** (`thinRingCps`, on merge, inheritance and shed): the lowest (a shared
  preamble's mark) and the newest stay longest; kept highest-first, a conversation's later turns pushed the preamble's
  mark out after two turns.
- **The SSD tier restores a ringed entry only at a ring file** (`bestRingMatch`, `restoreIntoRinged`): chunks hold the
  global layers, `r{pos}.safetensors` each restore point's ringed rows (the RAM entry's checkpoints plus its end,
  `RING_DISK_MAX_PER_ENTRY` = 8 kept, thinned as in RAM, salvaged per file at scan; manifest v9, which an older reader
  drops) ([mimo2-arch](mimo2-arch.md#sliding-layers-the-ring)).
- **A disk restore fills its buffers chunk by chunk** (`restoreKvInto`): each chunk is evaluated into buffers
  allocated at the restored length before the next file opens. A lazy `mlx_load_safetensors` holds its file open until
  eval (one eval at the end failed past the soft limit of 256 files), and a concatenation at the end held every chunk
  beside the result, twice the restored KV before any bill saw it. Each chunk's eval is drained before the next chunk
  writes: undrained, a write that beat the command buffer's release copied the whole buffers instead of donating, up
  to three copies of the restored KV at once on CI's M1 VM. The restore entry points drop the MLX latch they
  raised, or the cold fallback's prefill fails on it. Measured on a 150k-token Sushi-3bpw entry (147 chunks; b9dbbf53
  plus this change, `--ctx-size 262144 --prefix-cache-disk 20GB --prefix-cache-entries 1`, arms O P M M P O, 4
  restores per boot, `taskpolicy -a`, fans max, a lock per boot, 2026-09-27): a warm restore takes 170-177 ms with the
  fill, 185-188 ms with a per-chunk eval and the final concatenation, 183-187 ms on b9dbbf53; the first restore of a
  process takes 611-618 ms with the fill, 642-917 ms without it, 307-326 ms on b9dbbf53 (one eval per chunk).
- **A commit that forked off another entry inherits that entry's ring checkpoints below the fork** (`bestRingDonor`,
  refcount-shared and billed per entry like SSM checkpoints): a request appending to the conversation (a client's
  side request: the chat + a reminder) otherwise holds only its own prompt end, and once the count cap evicts the
  main entry the next main turn, diverging where the reminder was appended, cold-prefilled every turn. The slot's
  own checkpoint at its restore covers a donor that another slot's commit evicts before this one commits.

## Candidate ranking and trimming

- **Hybrid candidates rank by RESTORABLE checkpoint position, not raw match** (`findBestRestorableMatch` RAM,
  `bestHybridMatch` disk). Ringed candidates rank by `ringRestore`; an un-restorable one stays eligible at 0, so a
  lookup with nothing better still declines by name.
- **A lookup that restores 0 rows is not a use**, SSD-first or not (a hybrid with no usable checkpoint, the QSA
  history decline): the entry keeps its recency and its admission protection drops, else the count cap's next
  victim is an entry that can serve.
- Checkpoint retention thins the INTERIOR with a dense newest quarter (`spanPreservingDropIndex`, `ThinPolicy`).
- An oversized candidate is TRIMMED to the longest restorable prefix that fits (`trimLenForBudget`,
  `KVCacheSnapshot.trimmedCopy` is a REAL copy); a QSA trim bills the bank on the final retained checkpoint.
- **A ringed candidate trims only where it restores** (`ringTrimLen`): its end, dropping the reservation's spare
  capacity, or a ring checkpoint whose rows become its ring (`trimmedCopy`'s `ring_cp`). Its ringed layers are a
  constant, not a per-token price: priced per token, every MiMo target fell below the ring and every entry declined.
- A decline carries its `TrimDecline` reason; a RAM-budget decline spills to SSD (`spillDeclinedToDisk`).

## Budget

- An in-place SSD commit keeps the bill for an owned QSA history file when no new QSA checkpoint arrives; sidecar-only commits bill the change in all retained non-chunk files, including rings.

- A commit declines and frees its incoming snapshot when checked-out residents prevent satisfying either the entry-count or byte cap; the request continues and one `[hot-cache]` line names the limiting cap.
  The byte cap is judged AFTER the new entry sheds checkpoints (`retainNewEntry`): a qwen4_exp trim is priced against
  its shed survivors, and judging it unshed declined every session past the budget, so each turn cold-prefilled.

- **The hot-cache budget is CLAMPED at load** to what the weights leave under the GPU ceiling and is a HARD cap; it
  FOLLOWS residency (`reviseHotCacheBudgets` after every load/unload, repeated for 10 s because the OS returns pages
  lazily).
- **A RAM tier with no named size holds one session at the working context** (`oneSessionFor`, >= 2 GB, both arms; only
  reachable below the CLI now that RAM retention is opt-in)
  where the ceiling holds it beside the weights, the n-gram page cache (`page_cache_claim`) and a cold full-context
  prompt's bill (MiMo refuses rather than evicts); else that room, at most half the bill, so an outgrown session's
  trim copy fits beside it. A flag stands, `2GB` too; context sizing and the chunk pin still read the raw ask.
- A replacement over the budget sheds ring checkpoints (thinned as above) before the entry goes (`shedRingCheckpoints`).
- Measured (b9dbbf53 plus this change, Sushi-3bpw, auto context 1M, kv8, MTP on; a ~200k-token three-turn session; `taskpolicy -a`,
  fans max, GPU lock per boot; 2026-09-27): unset, the budget is 11516 MB and turns 2-3 prefill in 0.35 s (199.7k
  reused); `--prefix-cache-mem 2GB` keeps a 139k-147k prefix and prefills in 36.0 / 31.3 s (turn 1: 114-115 s cold). The
  n-gram table stayed 100% resident (mincore) in both arms.
- Eviction is WORKLOAD-fair (`cache_key`: `prompt_cache_key` > `metadata.user_id` > system-prompt hash;
  `lruIndexExcluding`).

<a id="defaults"></a>
## Defaults: SSD tier on, RAM tier off

- **RAM retention is opt-in.** Unnamed, `server.prefix_cache_ram_enabled` is false and the RAM budget is 0 bytes for
  every served model; `--prefix-cache-mem <n>` turns it on (`0` or `off` keeps it off, like `--no-prefix-cache-ram`) and `--no-prefix-cache-ram`
  wins over it. Launch flag > default; `model-settings.json` has no prefix-cache key. With RAM off every model runs
  SSD-first (`prefix_cache.ssdFirstActive`: a disk tier and RAM off).
- **The SSD tier is on and sized per model at load** (`server.prefixCacheDiskForLoad` through
  `LoadParams.prefix_cache_disk_resolver`, the pure `kv_disk_cache.resolveDiskBudget`):
  `min(entries x context x bytes_per_token + 2 GB, 20 GB, free disk - 4 GB)`. **Sizes are binary: 1 GB = 1 GiB and 1 KB = 1 KiB**, as
  `--prefix-cache-disk 20GB` is read; every `GB` and `KB` in the lines, the logs and the page is that unit.
  - entries = `--prefix-cache-entries`; context = the model's PINNED served context (`diskContextForLoad` runs the
    idempotent `pinAutoContext`), so the line, `/props` and `/v1/models` name one number; free disk =
    `kv_disk_cache.volumeSpaceNear` of the cache dir (`SUSHI_PREFIX_CACHE_DIR` or `~/.sushi/kv-cache`).
  - bytes_per_token = `server.diskBytesPerToken`, from the SERIALIZED geometry: KV rows at the served width, the one
    pooled QSA history an entry writes, and the MTP head's KV and history where the head is on. The QSA score bank and a
    second history copy are execution state and never count, so no QSA switch moves it. GLM's KDA checkpoints and
    DFlash2 window, Qwen's GDN checkpoints and MiMo's ring files are fixed per entry, not per token; the 2 GB slack is
    all the formula leaves them.
  - **The tier's own bytes on disk count as free** (`kv_disk_cache.tierBytes` is added to the volume's free space at
    resolution, as `refreshDiskBudget` adds `total_bytes` live), and `DiskTier.operator_cap` is only the flag or the
    formula / 20 GB cap, never a free-space number; else a restart on a full volume evicts what the last boot stored.
  - A result of zero or less turns the tier off and one line says why. `--prefix-cache-disk <n>` is the budget as
    given (an operator's cap: the tier still stores no more than the volume leaves) and `0`/`off` disables;
    `--prefix-cache-entries 0` disables both tiers.
  - It is a budget, not a preallocation, and ONE reserve governs it: free space less 4 GB (`kv_disk_cache.freeDiskRoom`,
    `DISK_FREE_RESERVE`) at resolution, before every store (`refreshDiskBudget`) and in the report. A budget under 1 GB
    still stores.
  - **The tier comes up before anything is sized for it.** `doLoadOnInferenceThread` resolves and attaches it first; a
    tier that was wanted and did not come up (no room, a fingerprint or init failure) sets
    `ModelConfig.prefix_cache_disk_declined`, and `server.diskTierOn(config)` then answers as `--prefix-cache-disk off`
    for the RAM sizing (`ssdFirstBudgetForLoad`, `reviseHotCacheBudgets`), GLM's writer and checkpoint bills, MiMo's
    restore coexistence and checkpoint capture. With no tier and no RAM the model gets no cache object at all.
- **Startup prints one line per tier in use**, each starting `Allocating`: `Allocating 20.0 GB SSD for the prefix cache
  (32 entries x 1048576 tokens x 15.5 KB + 2 GB; bound: 20 GB cap) at <dir>` (bound: the formula, `20 GB cap`,
  `free disk - 4 GB` or `the flag`), and `Allocating 2.0 GB RAM for the prefix cache (--prefix-cache-mem)` only when RAM
  retention is on.
- **Each finished turn logs `[disk-cache] usage <used> / <budget> GB, <n> entries`** from the FINISHING model's own
  tier (`scheduler.logDiskUsage`), and `/props settings.prefix_cache` carries that model's `disk_bytes` (the budget),
  `disk_used_bytes` and `disk_entries` ([server-http-apis](server-http-apis.md)). `publishDiskStats` writes them onto each
  `LoadedModel.disk_stats` (a sequence-locked snapshot, so the four numbers come from one publish) after every commit,
  load and unload; `/props?model=A` reads A's, whatever model loaded last.
- **Measured at the defaults** (GLM-5.3-Flash-Sushi-2.5bpw, a 146,795-token prompt of source files, greedy, 16 tokens,
  streamed TTFT, tokens reused; binary 80b97a28, the code of the SSD-default commit, a private cache dir per boot, a lock
  per boot, `taskpolicy -a`, 2026-10-05). The `Allocating` line reads `32 entries x 1048576 tokens x 6.5 KB + 2 GB;
  bound: 20 GB cap`; the disk holds 2.05 GB after turn 1.

  | setup | cold | repeat | append | append 2 | after a restart (append) |
  |---|---|---|---|---|---|
  | defaults (RAM off, SSD on) | 315 s | 0.46 s, 146,764 | 0.72 s, 146,764 | 0.56 s, 146,768 | 0.92 s, 146,772 (restore 409 ms) |
  | `--prefix-cache-mem 1GB` + SSD | 233 s | 0.46 s, 146,764 | 0.73 s, 146,764 | 0.55 s, 146,768 | 0.91 s, 146,772 |
  | `--prefix-cache-mem 2GB` + SSD | 224 s | 0.34 s, 146,764 | 0.64 s, 146,764 | 0.44 s, 146,768 | 0.93 s, 146,772 |

  The 2 GB setup serves repeats from RAM; the others restore from SSD. The cold times are the box's load, not the
  cache. MiMo-V2.6-Flash-Sushi-2.3bpw at the defaults (148,898 tokens; line `32 entries x 946176 tokens x 12.0 KB
  + 2 GB; bound: 20 GB cap`): 0.45 / 0.47 / 0.46 s from SSD, 0.67 s after a restart (restore 492 ms), 1.73 GB on disk.
  Qwen3.8-Flash-Next-Sushi-2.6bpw: `tests/test_prefix_cache_ssd_default.sh`.
- Bytes per token at kv8 (`diskBytesPerToken`): Qwen3.8-Flash-Next 13,824 for the trunk rows and the pooled history, plus
  the MTP head's KV and history when the head is on (about 15.5 KB deployed); MiMo-V2.6-Flash 12,240 (global layers
  only; the head is a fixed window); GLM-5.3 6,688 (BF16: 11,968). With 32 entries the formula is past the cap at any
  context over about 40K tokens, so all three get 20 GB: one 1M-token Qwen entry (16.7 GB at 15.5 KB) fits, or four of
  260K tokens, not 32. Fixed state is on top (a Qwen GDN checkpoint is 58.8 MB, a GLM KDA checkpoint 147.6 MB, a MiMo
  ring about 128 MB): a 20K-token Qwen turn measured 0.46 GB on disk against 0.32 GB of per-token bytes.

## SSD-only storage

`--no-prefix-cache-ram --prefix-cache-disk 10GB` (the default arrangement, with a sized budget) keeps reusable text
prefixes on SSD without retaining idle KV snapshots in the RAM cache. The live request still needs KV memory, and queued disk writes can hold
buffers temporarily. The entry count must remain positive: `--prefix-cache-entries 0` disables both tiers.
With RAM and disk disabled, SSM checkpoint capture is disabled too. `/props` reports
`settings.prefix_cache.ram_enabled=false` and `mem_bytes=0` when RAM retention is off.
Context and prefill-chunk sizing also reserve zero idle-cache bytes in this mode, regardless
of `--prefix-cache-mem`; live KV and temporary SSD write buffers still consume memory.

Qwen prefill chunks write through continuously in SSD-only mode. Hybrid SSM checkpoints, MiMo ring
restore points and GLM state ([GLM](glm5-prefix-cache.md#glm)) survive restart. Image-bearing entries remain ineligible for disk persistence. RAM+SSD is the opt-in
`--prefix-cache-mem` arrangement. `SUSHI_PREFIX_CACHE_DIR` can select an absolute cache directory; unset, the root stays
`~/.sushi/kv-cache`. Live tests use a separate root without changing home settings.

Ported from [mlx-serve #680](https://github.com/ddalcu/mlx-serve/pull/680), with Sushi's ring checkpoint handling.

<a id="one-tier-per-root"></a>
## One live tier per root

- **A root has one live tier** (`DiskTier.root_lock`: an flock on `<root>/.lock` from `init` to `deinit`; the file is
  never unlinked). Two processes on one pack (`sushi run X` beside `sushi serve X`) used to share the root, both
  numbered entries from `e1`, and one restored the other's KV for its own prompt with an HTTP 200.
- **A second tier keeps a private root** `<root>/p<pid>-<n>` and logs one `[disk-cache] ... private SSD tier` line. It
  reuses its own turns, never the shared root's entries, has its own budget, and is removed at `deinit`. A crashed
  one's is reaped (`reapPrivateRoots`) by the next tier or sweep on that fingerprint once its lock is free and its
  files are 10 minutes old.
- A new entry claims its id by creating `e<id>` (an existing directory is never adopted); staging files are
  `<file>.<pid>.tmp`.
- **A restore re-reads `tokens.bin` up to the restored length**: a record that differs from the index poisons the
  entry and the request prefills cold (`DiskCacheTokenMismatch`).
- `scan` drops an index-less entry with no age bar: the lock keeps every other writer out of the root. `sweepBase`
  holds each sibling root's lock while it sweeps it and skips one a live tier holds (`rootIsLive`); `tierBytes`
  counts nothing of a live tier's root. The skip is flock-based, so a second tier in the same process is skipped too.
- **A byte is counted once per inode** in `sweepBase` and `tierBytes` (`dirBytes` + `InodeSet`): chunk sharing
  hard-links one chunk into several entries. The tier's own `total_bytes` already bills it once.
- **A restore that finds a chunk file missing or short poisons the entry** (`restoreKvInto`), so the request prefills
  cold and the next commit of that prompt stores fresh instead of seeing it as superseded.
- A binary older than the lock takes none, so beside a newer one it still shares the root; only the token check
  covers that.

<a id="ssd-flush"></a>
## The SSD flush

- **A commit lands whole on disk after its response, whatever the RAM tier keeps.** Every disk tier arms the
  background writer. A commit records the live state before the RAM budget trims it (`pending_disk`: the rows, every
  restore point, the drafter window, the MTP history), and the flush after the response writes it in
  `FLUSH_PIECE_BYTES` (2 GiB) pieces until it is whole (`DiskTier.appendCommitWhole`). A piece's checkpoints and
  ring files ride outside its byte bound. Nothing is written on the response path: a RAM decline, a checked-out
  resident's included, rides the same record.
  - Before, a RAM-on tier that was not SSD-first (GLM, MiMo) flushed the entry its RAM budget had trimmed,
    synchronously, one 512 MB piece per turn: 58K of a 147K GLM prompt per turn, from a RAM entry trimmed to 117K.
  - An SSD-first flush stopped at 2 GiB with its checkpoints counted, so turn 1 of the same prompt in SSD-only mode
    ended 1.4K tokens short of its prompt-end checkpoint.
  - A tier whose writer failed to start writes the record whole synchronously when it is SSD-first (its source, a
    finished request's buffers, is gone after the flush), and otherwise keeps the legacy path: the RAM entry, one
    piece per turn, a declined candidate spilled synchronously.
- **The record carries the restore points the RAM entry inherits**: a donor's SSM or ring checkpoints below the
  shared prefix (`bestCheckpointDonor`, `bestRingDonor`), taken before the RAM tier sheds any. Chunk sharing links a
  fork's chunks, never its donor's checkpoint or ring files, so without them a fork restored only above the fork
  once the donor left the SSD tier.
- **A second commit before the flush flushes the first** (`capturePendingDisk`): the cull pass commits every
  cancelled GLM slot before any flush, and the second capture used to discard the first.
- **A sole entry never outgrows the tier** (`DiskTier.budgetTarget`): a record whose chunks, checkpoints, ring files
  and sidecar would pass the byte budget alone is cut to the highest checkpoint or ring restore point that fits (any
  length on plain attention), without its spec state; nothing fits, nothing is written. GLM picks its SSD rows the
  same way (`glmDiskLen` prices every checkpoint at or below them): a 1 GiB tier and 131K rows with eight checkpoints
  is 1.9 GiB, which used to land whole.
- **The writer's permit bounds the staged host bytes** (`Writer.waitForRoom` runs before a blob's buffer exists, and
  a file larger than half the permit goes in parts that append to one `tmp`, the last renaming it). A file's header
  and the manifest (a few KB) are built before their room is reserved. At most 1 GiB beside the GPU, billed on every
  disk tier. A flush that outruns the writer waits on it after the response, which delays the next request and any
  concurrent decode, not this one.
- **The spec sidecar is written once per commit**, by the piece that completes the rows, and staged like any file;
  an earlier piece keeps what the entry had. Each piece used to rewrite it synchronously.
- **Nothing is fsynced.** A file lands as `tmp` + `rename` and the manifest last, so a crash of the process leaves
  whole entries; a power loss can leave a manifest naming bytes the disk never kept. A failed restore cold-prefills.
- **GLM hands the SSD tier the request's own rows** (`MlaRows.shareLive`, `HotPrefixCache.glmDiskLen`): every row
  through the newest checkpoint that fits the tier's budget, never copied. The share holds the request's buffers
  from its commit to the flush, when nothing writes them. The RAM tier copies only what its budget keeps.
- **A writer-armed tier hard-links a diverging turn's whole shared chunks** (`chunkShareDonor`), as SSD-first did, so
  each turn of a RAM+SSD conversation writes its new rows, not the conversation again.
- **The longer tier restores.** A lookup takes the SSD entry when its restorable position beats the RAM entry's by
  `MIN_DISK_ADVANTAGE_TOKENS` (256), else the RAM one.
- **A flush costs the inference thread its readback alone**: 2,054 MB in 207-225 ms with 0 ms waiting on the
  writer, whose `write` lands in the page cache faster than the readback fills it (`[disk-cache] persisted ...
  ms waiting on the writer`).
- **Measured** (GLM-5.3-Flash-Sushi-2.5bpw, kv8, a 146,795-token prompt of source files, `/v1/chat/completions`,
  greedy, 16 tokens, `reasoning_effort` low; streamed TTFT with wall in brackets, cached tokens after; `taskpolicy -a`,
  a lock per boot, busy box, 2026-10-05). Before is de867e94, non-streamed wall; after is this change on 42504add
  before its review fixes (binary stamp 625fe7e8); those add inherited checkpoints to a fork's record and write
  the spec sidecar once, and leave the lookup and restore path these TTFTs time unchanged.
  The cold turn (245-297 s) runs the same code in every arm.

  | arm | turn 1 on disk | repeat | append | append 2 | after a restart |
  |---|---|---|---|---|---|
  | SSD-only, before | 145,408 | [3.80 s] 145,408 | [1.23 s] 146,764 | [1.15 s] 146,768 | |
  | SSD-only, after | 146,764 | 0.51 s [0.84] 146,764 | 0.77 s [1.16] 146,764 | 0.61 s [1.12] 146,768 | 2.36 s [2.87] 146,772 |
  | 1 GiB RAM + SSD, before | 58,368 | [61.4 s] 116,736 | [30.1 s] 131,072 | [30.2 s] 131,072 | |
  | 1 GiB RAM + SSD, after | 146,764 | 0.49 s [0.83] 146,764 | 0.77 s [1.17] 146,764 | 0.59 s [1.05] 146,768 | 3.11 s [3.62] 146,772 |
  | 2 GB RAM, no SSD, before | | [0.68 s] 146,764 | [1.06 s] 146,764 | [0.93 s] 146,768 | |
  | 2 GB RAM + SSD, after | 146,764 | 0.34 s [0.66] 146,764 | 0.61 s [1.02] 146,764 | 0.43 s [0.93] 146,768 | |

  - With 1 GiB of RAM the RAM tier keeps 116,736 tokens and one checkpoint, so every warm turn restores from SSD in
    147-165 ms; at 2 GB it keeps the whole prompt and restores from RAM.
  - The first restore after a restart took 382 ms and 1,215 ms against 121-165 ms warm. It allocates every restored
    buffer anew beside a busy box: the files themselves read at 7.8 GB/s with the page cache dropped, and read-ahead
    advice made no difference. The first forward after a boot adds about 1.5 s.
- **Qwen3.8-Flash-Next-Sushi-2.6bpw**, a 157,678-token prompt, same driver and conditions:
  - SSD-only on 42504add: cold 82.9 s; repeat, append and append 2 0.43 / 0.58 / 0.52 s TTFT, 157,647-157,652 reused
    from SSD; after a restart 2.40 s, 157,656 restored in 666 ms. Its prefill write-through already landed every turn
    whole.
  - The default RAM tier plus SSD, this change: cold 105 s; 0.19 / 0.40 / 0.28 s from RAM. Turn 1's write-through
    banked one chunk per prefill chunk (39,936 tokens), and the flush after the response landed the rest in one piece
    (1,523 MB, 202 ms, 19 ms waiting on the writer). After a restart 2.46 s, 157,656 restored from SSD in 707 ms.
- **MiMo-V2.6-Flash-Sushi-2.3bpw**, a 148,898-token prompt, same driver and conditions:
  - SSD-only on 42504add: cold 260 s; turn 1 whole on disk in one 1,771 MB flush; repeat, append and append 2 0.43 /
    0.46 / 0.45 s TTFT from SSD (ring restore points); after a restart 0.85 s, 148,902 restored in 506 ms.
  - The default RAM tier (6.3 GB) plus SSD, this change: turn 1 whole on disk after the response (1,771 MB, 139 ms,
    0 ms waiting), where it used to persist one synchronous 512 MB piece per turn; 0.28 / 0.32 / 0.29 s from RAM;
    each later turn hard-links 145 chunks and writes one. After a restart 0.87 s, 148,902 restored from SSD in 524 ms.

## SSD-first

- Disk fingerprints include the model path, config size/mtime and overrides, plus sorted indexed weight-shard (or unindexed safetensors) names and size/mtime and `ngram_table.bin` size/mtime; payloads are statted through symlinks, never content-hashed.

- `prefix_cache.ssdFirstActive` = a disk tier AND (capable arch OR RAM retention disabled), mirrored onto `HotPrefixCache.ssd_first` +
  `DiskTier.ssd_first`: with RAM enabled it floors at ONE session, `--prefix-cache-mem` = the IDLE allowance.
  SSD-only storage retains no idle RAM entry.
- Spill and EVICT are two decisions (`PersistOutcome`: only `.persisted` + an agreeing index + landed files license
  discarding RAM); writes ride `kv_disk_writer.zig` (FIFO, `meta.json` last, epoch fence at the ONE removal site);
  per-chunk write-through; a diverging turn hard-links the donor's LANDED chunks; a full-prefix hit CHECKS the entry
  OUT so the first append donates.
- **The free-space probe runs only before an actual store** (after the superseded check): the idle spill commits every
  idle entry at each request finish, so a copy already on disk must cost no probe; below the store floor it still
  counts as persisted. The probe's budget GC swap-removes entries, so the extend/SSM-only candidate is selected
  AGAIN after it (`selectStoreTarget`); an index held across a probe names another entry.
- **A checkout is a PROMISE until the append DONATES** (`donateCheckout` right before `Generator.initWithOptions`,
  below every refusal; `releaseCheckout` hands an undonated entry back intact).
- **Off SSD-first, a warm share that does not fit is taken over, not refused** (`checkoutRestored`, qwen4_exp's
  admission pass): a full-entry hit is checked out on demand and billed as donated. The tradeoff: a request that
  fails after donating loses the entry. Disk checkpoints come off the TOP of
  the flush budget; the disk tier serves the pre-media text prefix only.
- **A media commit persists its text to the SSD tier, never a media row** (`diskTextLen`): the record stops at the
  first item (none when the boundary is unknown), a hybrid at its last checkpoint at or below it (`hybrid`, set at
  load; the QSA bank is sliced onto that checkpoint), a ringed cache only where a ring checkpoint sits at or below
  it. Spec snapshots (DFlash window, MTP history) are not persisted with a cut record. Idle spill still skips media
  entries: the commit already wrote their text.
- **"Free disk" is what the OS will GRANT** (`sushi_volume_free_for_use`, statfs fallback): purgeable space is released
  on demand. The `volumeSpace` test must not race the OS's purgeable answer.

<a id="spec-state"></a>
## Spec state rides the cache

`Entry.mtp` + `restoreSpecSnap`, adopt only on `base + step == matched`; MTP trims to `mtpCommittedLen`; survives the
SSD tier (`spec.safetensors`). An adopted spec cache has ONE owner at a time (`runPrefill` clears its locals BEFORE
`initWithOptions`).

## GLM-5.3 (`glm5_next`)

Moved to [glm5-prefix-cache](glm5-prefix-cache.md).

## Guards

`tests/test_prefix_cache_*.sh` (budget revisit, disk, hot, mem, workloads), `tests/test_hybrid_reuse_equivalence.sh`,
`tests/test_mimo_ring_reuse.sh`, `tests/test_mimo_ring_fork_ssd.sh`, `tests/test_qwen4_mtp_head_persist.sh`,
`tests/test_glm_prefix_reuse.sh`, `tests/test_prefix_cache_tiers.sh`. Grep the log for `[cache]`, `[hot-cache]`,
`[disk-cache]`.
