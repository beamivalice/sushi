# GLM-5.3: prefix cache, restore and spec state

GLM-5.3 prefix cache, restore and spec state, split out of [engine-prefix-cache](engine-prefix-cache.md); everything shared by the served architectures stays there.


<a id="glm"></a>
## GLM-5.3 (`glm5_next`)

GLM's state lives in its slot's `glm5_forward.Request`, not a `KVCache`: 34 FP32 KDA states, 11 MLA latents and a
pooled index (`src/glm5_prefix.zig`; [glm5-arch](glm5-arch.md)).

- **A restore point is a pool boundary** (a multiple of 4), where the IndexPool tail is empty. The state there is:
  - the KDA conv and recurrence of every linear layer, an `SSMCheckpoint` of 147,619,840 bytes;
  - latent rows [0,P) and pooled rows [0,P/4), a prefix of the entry's `MlaRows`.
  One `MlaRows` (every row through the entry's newest checkpoint) serves all of the entry's checkpoints.
- **Checkpoints sit on a 2048-token grid and at the prompt end.**
  - The grid is absolute multiples of GLM's widest chunk (`glm_checkpoint_stride`), never a request's width.
  - A narrower chunk divides 2048, and `nextChunkEnd` ends a chunk on every grid point, so every grid point is a
    real chunk boundary whatever widths a request steps through.
  - The prompt-end backoff grows to 30-33 so its position is a pool boundary (`glmSnapshotBackoff`).
  - At most 8 per entry (`glm5_prefix.checkpoint_cap`), thinned span-preserving with a dense newest quarter.
- **A restore on the grid is bit-identical to cold when both run the same chunk widths from that point on.** That
  holds at the 2048 default. The suffix runs the same absolute chunks, tail merge, backoff and final span.
  - A request stepped down to narrower tail chunks near 1M matches only a cold run that steps down the same way.
  - Guards: the generator fixture tests (including a mid-request step-down), the hot-cache fixture tests, and
    `tests/test_glm_prefix_reuse.sh`.
- **A restore takes the nearest checkpoint at or below the match** (owner decision), usually the previous prompt's
  end. Off the grid, only the chunk around the checkpoint runs in a different shape, and the suffix rejoins the cold
  grid at the next boundary.
- **Measured bound** (Sushi-2.3bpw, kv8, `/v1/completions`, greedy, 256 tokens, top-5 logprobs, against the same
  prompt cold; this change on b8267038, `taskpolicy -a`, lock per boot, 2026-10-05):
  - Appended prompt restored at the previous prompt-end checkpoint:
    - First-token |Δ logprob| was 0.06, 0.07 and 0.02 nats at 8.6K, 32K and 60K tokens; a second 8.6K run gave 0.25.
    - While greedy agrees, the chosen token moves at most 0.56 nats.
    - Greedy flips at near-ties: at token 0 at 32K and token 6 at 60K. At 8.6K it held for all 256 tokens.
  - Restore on the grid, at 4,096, 28,672 and 55,296 tokens: every token and every top-5 logprob equal to cold.
- **TTFT, same runs:**
  - Appended prompt: 0.57, 0.41 and 0.43 s warm against 11.1, 42.4 and 79.6 s cold.
  - Grid restore with up to 2K tokens to prefill: 2.1, 1.6 and 3.3 s against 6.2, 37.5 and 75.8 s.
  - Decode is unchanged, 29.9 to 30.2 tok/s.
  - SSD-only (`--no-prefix-cache-ram --prefix-cache-disk 12GB`), turns growing from 32K to 42K tokens: 6.5 to 7.6 s
    warm against 33.9 s cold at 28K. A restart restored 41,720 tokens from disk in 115 ms.
- **Sushi-2.5bpw + vision + A4, cache on at its defaults** (this change on fbdedc01, kv8, `--prefix-cache-entries 1`
  so the cold arms follow an eviction, `taskpolicy -a`, lock held, 2026-10-05):
  - It advertises 1,048,576 and admits every request at 2048-row chunks.
  - An 8.6K appended prompt reuses 8,552 tokens and prefills in 0.60 s against 13.0 s cold. First-token |Δ logprob|
    is 0.10 nats; greedy flips at token 1.
  - A grid restore at 4,096 of a 5.5K prompt prefills in 2.4 s against 7.5 s, every token and top-5 logprob equal
    to cold.
- **Checkpoint state is copied bit for bit** (`bitsOwnedCopy`: an integer view plus an integer zero).
  `materializedOwnedCopy` adds a float zero, which turns -0.0 into +0.0, and the restored KDA state carried that
  into the next chunk.
- **The RAM tier is opt-in; named without a size it is 1 GiB** (`GLM_PREFIX_CACHE_MEM_DEFAULT`, reachable only below
  the CLI, which always names a size). The advertised context reserves nothing for it: admission evicts it to admit a
  long prefill.
  - At kv8 it keeps a 30K session at its prompt end with 5 of 8 checkpoints, a 60K one with 4, and trims a 140K one
    to its checkpoint near 121K with 1. Each case keeps the assistant window.
  - For long reuse add `--prefix-cache-disk`, or run SSD-only (`--no-prefix-cache-ram --prefix-cache-disk 12GB`).
- **The RAM tier's rows are a real copy at commit**, through the newest checkpoint it keeps
  (`HotPrefixCache.glmCommitLen`, chosen before the copy; a restore resumes from a checkpoint, so later rows are never
  read). It keeps rows, checkpoints and window within its budget, and the checkpoints above the chosen row are freed
  first, so there is no full copy followed by a trim. A share would keep the request's reservation (up to the whole
  context) alive while billing only the rows. The SSD tier takes a share instead, only until its flush
  ([SSD flush](engine-prefix-cache.md#ssd-flush)).
  - Billed in `kv_bytes` beside the checkpoints: 6,688 bytes per row at kv8, 11,968 at BF16.
  - A budget trim lands on a checkpoint (`MlaRows.trimmedCopy`), sheds interior checkpoints and keeps the window.
- **A restore shares the rows; the first append copies them.** A checkout releases them, so that append donates.
- **The assistant window rides the entry as prefill left it** (`Generator.glm_prefill_window`). Every GLM restore
  point is at or below the prompt end, and a window cropped at the reply's end misses it once the reply passes
  2,016 tokens.
- **SSD tier**:
  - Rows go in the usual chunk files as a dense pseudo-cache of two entries per layer (`glm5_prefix.diskEntries`).
    kv8 is keyed `{off, 8, 64}`, which the manifest keeps.
  - KDA checkpoints go in `s{pos}` files, at most 8 per entry, the window in the spec sidecar.
  - `spec.safetensors` is replaced in place before the manifest commits and equal windows share a size, so it carries a
    `d.pos`/`m.pos` = `base:step` stamp; a load that finds it absent or different declines the spec (trunk restores).
  - A restore reads only its own checkpoint file (`DiskTier.restoreIntoKda`); the QSA check that rereads the
    newest one is Qwen's.
  - GLM is never SSD-first while RAM retention is on. Under SSD-only storage it is, and keeps no idle RAM entry.
    Either way the commit captures every row through the newest checkpoint, the checkpoints and the window into the
    pending flush, which lands whole after the response ([SSD flush](engine-prefix-cache.md#ssd-flush)). There is no prefill write-through.
- **A decode-phase cancel commits in `cullDecoding`**, before `releaseNativeState` resets the request that the
  cleanup drain's commit would otherwise read. The drop decision is taken once per slot under `queue_mu`; the commit
  and the release run outside it on the dropped slots, so a late cancel waits for the next tick.
- **`commitImpl` owns the transferred checkpoints on every outcome**, including a failure of the retention snapshot.
- **One schedule drives the capture and its bill** (`generate.glmCaptureSchedule`: the grid points in the tail, the
  prompt-end checkpoint, pool alignment, cold/warm backoff, the cap). The configured stride never enters, and
  `glmChunkEnd` keeps the tail merge from absorbing a grid point, so the billed count is the captured count.
- **Bills.** A GLM request holds up to 9 checkpoints during prefill (the cap plus the copy taken before each thin)
  and one assistant window. The commit moment is billed beside the live cache (`glmCommitStateBytes`): the RAM
  tier's row copy (at most its budget; the SSD tier copies none), the checkpoints and the window, whichever of that
  and the prefill's transient is larger. At 1M tokens it is the smaller, so it costs no checkpoints. The writer's
  1 GiB permit (the previous request's staged flush) is not in this bill: admission takes it off the headroom for every
  arch (`scheduler.diskWriterHostBytes`, [engine-memory-admission](engine-memory-admission.md)); SSD-only reserves no
  idle cache.
  - Only the inference thread's admission pass bills the checkpoints (`WarmPrefix.checkpoints`). The connection
    thread, the context sizer and the cache clamp bill none, so the advertised context is the cache-off one.
  - The pass evicts RAM entries LRU first, sparing the one it restored from. If the request still does not fit, it
    keeps fewer checkpoints, down to the prompt end and then none, rather than be refused
    (`scheduler.fewerCheckpointsToAdmit`). With none it commits nothing.
  - The checkpoints yield to the width: the prefill width is chosen as if the request kept none.
