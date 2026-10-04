# Engine: MTP speculative decoding (native heads, on by default)

How the native MTP heads draft and verify: Qwen3.8-Flash-Next's one head and MiMo-V2.6's three, their inputs, the
spec-verify invariant, draft re-scoring, the measured round-cost table, head KV and norms. Read this before touching
`src/mtp.zig`, `src/mtp_*.zig`, `src/mimo_mtp.zig`, `src/round_cost.zig` or the MTP orchestration in
`src/generate.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [arch-qwen4exp](arch-qwen4exp.md),
[engine-qsa-long-context](engine-qsa-long-context.md), [engine-prefix-cache](engine-prefix-cache.md#spec-state),
[perf-baselines](perf-baselines.md#mtp).

## Code map

| File | Role |
|---|---|
| `src/mtp.zig` | MTP head (in-checkpoint `mtp.*` or sidecar via `resolveMtpSource`) |
| `src/mtp_acceptance.zig` | acceptance modes `exact|typical|tokenv3` |
| `src/mtp_group_planner.zig` / `src/mtp_group_cost.zig` | grouped verify planner |
| `src/mtp_qmv.zig` | M=1-exact qmv rows |
| `src/mimo_mtp.zig` | MiMo's three trained heads (`model.mtp.layers.{0,1,2}`), per-request row state |
| `src/round_cost.zig` | Measured per-model/width/KV-bucket spec round-cost table (`Transformer.round_cost`) |
| `src/mtp_lookup.zig` | Prompt lookup inside the MTP round: the committed-context index and its gate |
| `src/generate.zig` | MTP orchestration, `commitForcedTokens` |

## The head

- The head = the checkpoint's own QSA+MoE layer over the PRE-mixer stream (`Qwen4Mtp`, `MtpHeadRef.qwen4`):
  `capture_hidden(_all)` = `[B,L,hc*hidden]`, never the mixed 2560; head row r = (stream at r, token r+1) at query
  position r+1, so QSA takes a `pos_base`.
- Per-request state is a `Qwen4MtpState` swapped onto the module (`qwen4MtpActivate` before EVERY head touch);
  nothing is module-owned, so MTP slots are not exclusive.
- `--no-mtp` gates the IN-CHECKPOINT head too (`entry.mtp` reads `mtpChoiceFor`, logged `[mtp] on|off (<source>)`);
  an explicit `--mtp`/`--no-mtp` beats `model-settings.json` `mtp`. An explicit `--mtp` is refused while
  streaming; the engine default resolves off ([engine-expert-streaming](engine-expert-streaming.md)).

<a id="mimo"></a>
## MiMo's three heads

- **What is generic and what is qwen4's.** The controller is head-agnostic (`MtpHeadRef` switches five
  operations): the round phases, verify invariant, draft rerank, acceptance modes, EV planner, round-cost table,
  depth caps and EV seed serve any head. qwen4-only: the pre-mixer hyper-connection stream as the head input, the
  mixer output as the lm_head input, QSA `pos_base`, the deferred PLE leaf, `forwardQwen4VerifyRows`, head
  persistence, the G17 cost profile and merged multi-slot verify.
- **Semantics (SGLang's multi-layer EAGLE for MiMo; vLLM runs layer 0 only).** Head k's row p =
  `eh_proj(cat[enorm(embed(x_{p+k+1})), hnorm(h_p)])` at rope position p, `h_p` the trunk's FINAL-NORMED hidden
  (`capture_hidden_all`), predicting x_{p+k+2}. Every head reads the target's hidden, never the previous head's
  output, so a round drafts d1..d3 by running head i at the round's last committed position q; head i's rows past
  q-i carry drafts and are truncated at the next round (`mimo_mtp.State.truncate`).
- Each head is a sliding (128) layer with sinks: FP8 qkv (rank-local, tp 4 solved from the 116 scale rows) + bf16
  o_proj, FP8 dense SwiGLU 16384, own `final_layernorm`, the trunk's embedding and lm_head. Its K/V live in a
  per-request `RowCache` holding the window, never a `KVCache`; the prompt appends only its last window per head.
  The target hiddens ride a 256-row ring (`State.record`), copied out of a prefill chunk's hiddens and evaluated at
  once: a slice of them held the whole chunk (35 MB at 4096 rows) until the first round.
- The `.mimo` arm maps the generic stash + merged first step onto head 0 and each later step onto head i
  (`draftStep`); the step index rides `hidden_next` (a scalar), host token ids ride `host_ids`. Depth and the free
  EV cap clamp to the head count and to the verify row budget; rounds stay solo (`mtpRoundsStaySolo`); no
  prefix-cache persistence (the head rebuilds from the forwarded tail's last window).
- **The heads need no state across a prefix-cache restore**: a warm full reuse, whose heads start from one row,
  decodes as fast as the same greedy request prefilled cold (56.5 vs 56.1 tok/s, 1.71 vs 1.51 accepted per round;
  12 pairs, 2k-8k context, code and prose, one boot, 63476cd1, kv8, `taskpolicy -a`, busy box, 2026-10-01). The heads
  draft from the trunk's hidden at the current position; their own window adds nothing measurable.
- **The load warms every verify row count and head** (`warmupMimoVerify`, `Head.warmup`, `[spec-warmup] MiMo …`): each
  row count JITs its own pipelines, and a new binary's first round at each width stalled 450-630 ms.
- **Verify rows keep decode arithmetic** (`ForwardCtx.verify_rows`, up to `MIMO_VERIFY_ROWS_MAX` = 8 rows, the FP8
  GEMV's direct-row limit; the three heads still draft at most 3, so only a lookup or PLD verify runs wider): every
  row's attention runs through `mimoDecodeAttn` on the keys its own decode tick saw (`mimoVerifyRowsAttn`; a
  sliding layer's rows in one dispatch, `mimoSlidingRowsAttn`), the rest of the forward is row-identical already
  (FP8 GEMV <= 8 rows, `mtp_qmv` affine-8, the router rows' one f32 gemv `mtp_qmv.f32GemvRows`, the EXL3 decode
  chain). A partial accept truncates the cache (attention-only trunk).
- **A global layer's verify rows share the packed-cache walk** (`mimoGlobalRowsMpp`, `sushi_qkv_mpp_rows`): from 4096
  keys on matrix units, rows go in pairs, with a last three in one pass. Each K/V page is staged once per group, and
  each row runs its decode tick's own matmuls, softmax and rescale on it.
  - A group engages only where every row shares its last row's split partition. A shorter row then reads at most one
    more page, fully masked, which adds exact zeros. Otherwise the rows go one by one.
  - Byte-identical to decode ticks; -5% to -7% per verify forward at 128k keys
    ([perf-baselines](perf-baselines.md#mimo-verify-global-rows)).
- Oracle: `tests/dump_mimo_v2_mtp_fixtures.py` renders the heads from the HF reference's own modules on the tiny
  fixture model; `mimo mtp heads track the torch rendering…` replays history, rounds, wrong drafts and rollbacks.
- **A MiMo verify row reads its own 8 routed experts**, so it costs a large share of a forward and depth pays only
  on predictable text (code, lists, JSON; prose loses). Greedy MTP is byte-identical to serial (18/18 pairs at 256
  tokens, forced and auto).
- **Real-text verify rows share ~30% of their expert slots**, which the grouped decode GEMVs exploit (one weight
  decode per pair of slots, [engine-exl3-experts](engine-exl3-experts.md#kernels)). The global layers' split-K follows
  each row's own key count, so batching their rows is not bit-identical.
- **MTP adds to each prefill chunk only the heads' catch-up** (three heads x the 128-row window). Compare prefill arms
  interleaved in one boot, never one reading per arm.
- **A MiMo verify's hidden captures stay lazy** (`capturePrefillHidden(.., settle = !ctx.verify_rows)`): the trunk,
  lm_head and accept read go out as one dispatch, where a settled capture made the host wait for the trunk before it
  built the lm_head ([perf-baselines](perf-baselines.md#mimo-mtp-round)). A prefill chunk's capture still settles.
- **A MiMo draft step is ~0.95 ms of head forward plus the coarse readout**, and the readout is 2-bit on MiMo
  (`mimo_mtp.rerankBits`; `SUSHI_MTP_DRAFT_HEAD_BITS` overrides it). At forced depth 3 the 2-bit readout accepted
  what the 3-bit one did (5 prompts, within 1 accept over ~380 rounds, same bytes out), and a three-draft chain
  dropped from 4.22 to 3.80 ms (the 2-bit readout on 2f15cc97, kv8, 2.3bpw).
- The EV planner prices a MiMo EXL3 round with its own surface (`.mimo_exl3`, `MTP_EV_MIMO_EXL3_COSTS`: draft
  .04, verify row .44 of a forward, flat to depth 3); the generic surface prices a row at .20 and over-drafts prose.
- **t1 streams at the prefill handover** (`scheduler.publishHandoverToken`): an MTP request's first token is on the
  host when prefill ends, so it goes out before round 1; the round still commits it, and its echo is swallowed once
  (`Slot.takeHandoverEcho`). An EOS t1 stays with the round. MiMo's last prompt chunk leaves the heads' catch-up to the
  first draft chain. A first token the stream can show arrives one round earlier. On MiMo with thinking on, t1 is the
  `<think>` opener, so the first visible token does not move. llmprobe measures decode first frame to last, so where t1
  shows, its decode rate reads about a round lower for the same token times.

## Spec verify invariant

- `cache.step = prompt_len + emitted`, t1 NOT in cache on entry, verify input `[t1, draft…]`, partial-accept
  correction from ORIGINAL `verify_logits[accepted]`.
- A block decoder checks its ENTRY token before drafting (`generate.tokenStops`); the token budget is a PRE-COMMIT
  invariant; a committed argmax is a `CommittedArgmax` (only `verifyArgmax` builds one, masking reserved ids).
- A think penalty shifts every verify block right after its forward (`thinkShiftRows`, solo, grouped and batched-head
  alike): row j is gated on the verify ids through j, so it reads the logits its serial tick would; drafts stay raw.
- Scoped file/request logit biases share these prefix-dependent shifts for MTP, PLD and prompt lookup; active
  rewards and penalties bypass the target head shortlist. The think-penalty preset retains its original arithmetic.
- logprobs>0, grammar and a repeat/presence penalty disable spec.

## Cost and acceptance

- **A qwen4 verify row is BYTES, not dispatches**: a second row's own experts are read, so a depth-2 round ≈ 2
  serial forwards and prose accepts ~1.0. On the MCG K3 pack this no longer holds cleanly: verify rows cost
  ~4.6 ms each (28.2 / ~33.5 / ~37.5 ms at 2/3/4 rows) because routed experts are a minority of the bytes; measure
  before relying on either reading.
- **Off NAX (M1-M4) a qwen4 verify row is ~47% of a forward** (~16 ms of ~34 on an M2 Max; the M5 Max ~26%), spread
  over experts, GDN, HC and attention, so MTP nets ~1.1-1.35x there ([perf-baselines](perf-baselines.md#m2max-decode)).
- Acceptance is a PROMPT-TYPE property (code ≫ prose; `SUSHI_MTP_FORCE_DEPTH=n` + `acc_idx=` on `[mtp-trace]`).
- **MTP is ON by default for both served models** (owner policy, `server.defaultEnableMtp` `served`): a request
  that omits `enable_mtp` runs the loaded head. `--no-mtp`, `"mtp": false` in `model-settings.json` or
  `enable_mtp:false` turn it off; an SSD-streamed pack loads with the head off (`[mtp] off (streaming; default)`,
  `scheduler.mtpDefaultOffUnderStreaming`) unless an explicit `--mtp` keeps it resident. The load-time bill prices the head's
  KV whenever it runs by default (`server.mtpHeadDefaultOn`).
- **Concurrent qwen4 MTP streams can share a verify**: the group planner (on by default, `SUSHI_MTP_GROUP_PLANNER`;
  a request opts out with `enable_batch_mtp:false`) runs a grouped round (`[mtp-planner] rows=N widths=…`, row-axis
  verify) when it prices one cheaper, and `mtpRoundsStaySolo` does not gate it. Only when the planner declines the
  tick do rounds stay solo, two interleave and three or more go plain (`mtpRoundsStaySolo`;
  `SUSHI_MTP_BATCHED_QWEN4` opts in; `mergedVerifyDeclineReason` names the decline). Four MTP streams on MCG K3
  aggregate ~85 tok/s; a linear model of the measured verify-row cost predicts ~95-125 with merged verify at depth
  2-3. Measure before any code.
- <a id="greedy-shortlist"></a>**A greedy MiMo forward reads the trunk head through the drafts' shortlist**
  (`lmHeadShortlistFor`, up to 16 argmax-only rows, serial ticks and verify rows alike): the coarse top-32 per row,
  re-scored through the full head (the same qmv, so each value is the full readout's), every other id -inf. The copy
  exists with MTP or without, so MTP == serial holds by construction. Gate `generate.argmaxOnlyRequest` (greedy or
  top-1; no penalty, logprobs, grammar, forced call or PLD); `kld` reads the full head. Not certified: a full argmax
  outside the top-32 changes the token; `SUSHI_LMHEAD_SHORTLIST_AUDIT=1` counts such rows, `SUSHI_LMHEAD_FULL=1`
  reads the full head ([perf-baselines](perf-baselines.md#mimo-lmhead-shortlist)).
- **Drafts shortlist on a coarse lm_head copy and re-score exactly** from the MIXER output
  (`buildRerankCoarse`/`rerankShortlist`/`fullReadoutArgmax`, `StepWant.mixed`; `SUSHI_MTP_DRAFT_RERANK=0`
  restores the full readout). A greedy target drafts the argmax (byte-identity contract); a sampled target draws
  from the re-scored top-32 (`mtpDraftStepPath`); draft temperature is per family.

<a id="lookup"></a>
## Prompt lookup inside the round

- **A lookup stands in for the chain when the output copies its context** (`mtpLookupChain`, ported from mlx-serve
  #523/#533): the last 3 committed tokens plus t1 matched earlier in the prompt or output, agreeing back 8+ tokens,
  make the drafts with no head forward. On by default for the qwen4 and MiMo heads; `SUSHI_MTP_LOOKUP=0` turns it off.
- **An ordinary match (suffix under 32) must agree past the start of a line**: a unified diff echoes the file's
  lines behind a `-`/`+`/space prefix, so its matches agree to the end of one line and fail at the next (-4.8% on
  the diff before the rule, -1.6% after). A line's own last token (`):\n`) agrees whatever the next line starts
  with, so a break counts only with agreement after it ([perf-baselines](perf-baselines.md#mtp-lookup)).
- **At most 8 drafts (9 verify rows)**: `MtpHistStash.host_ids` holds `MAX_DEPTH + 1` committed ids, and 9 rows is
  the widest verify whose rows match the decode tick (`sdpaTickIdenticalGroups`). Upstream drafts 14.
- **The gate prices both rounds from the planner's own `MtpCostSource`** (measured width row, else the EV surface):
  a lookup of k drafts reads the table's runtime lookup row, else the MTP round at k without its k head steps. The
  lookup row is never stored, and the first round at each draft count is dropped as its compile.
- **The MTP side is priced at the width its rounds draft** (`mtpLookupPriceWidth`: the request's drafted-per-round
  EMA inside [m_lo, m_hi]), the width its acceptance EMA came from. Priced at a two-chunk plan's m_lo, a copy after a
  prose request (m_lo 1 extending to 3, ~2.7 accepted) read MTP as a 3.7-token one-draft round and ran half its lookup
  rounds (MiMo 2.3bpw: 33 vs 66 per 550-token copy; 62 with the fix, same bytes).
- **A lookup round feeds no EV, depth, planner, round-cost, regime or adaptive-serial price state**, and the MTP
  round after one stays untimed; an untimed round ends the regime interval (`mtpRoundUntimed`), or the next MTP
  round is billed for it.
- **A lookup round applies the pending history stash** (`mtpApplyStash`): left pending, it grows across lookup
  rounds into one oversized head forward.
- **On MiMo a lookup drafts no wider than its verify keeps decode rows** (`mtpLookupDraftCap` =
  `mtpVerifyDraftsMax(true)`); its stash append is the head-generic `mtpApplyStash`, and a lookup as round 1 leaves the
  prompt's lazy head catch-up to the next head forward, as an MTP round 1 does.
- **Declined** under `SUSHI_MTP_FORCE_DEPTH` (the byte bar's measurement mode), for a batched head and in
  planner-owned rounds; a serial block (adaptive serial past 32k) runs none either.
- **Output**: greedy is serial byte for byte (every row is a decode tick's); sampled under `exact` keeps the target
  distribution, but seeded text differs from lookup-off because the draws land differently. A lookup round always
  verifies with `exact` (`acceptGraphFor`/`acceptPrefixFor`, keep the copy with probability p; MTP rounds keep the
  installed mode): a draft is a point mass, so `typical`'s floor makes any plausible copy certain, the output echoes
  its context and the loop guard cuts it (mlx-serve #614).
- **Measured on Sushi-3bpw**: copies and edits of a file +16-21%, a write_file tool call +9-11%, long-context edits
  +18-24%; diff, new code and prose inside noise ([perf-baselines](perf-baselines.md#mtp-lookup)).
- **Measured on MiMo 2.3bpw** at seven drafts (8-row verify): a verbatim copy +23%, a rename +19%, prose unchanged;
  at three drafts it was neutral, since the heads already land ~3.9 tokens per round on a copy
  ([perf-baselines](perf-baselines.md#mimo-verify-8)).
- Engagement: `[mtp] prompt-lookup drafts engaged: k=… suffix=…` once, `[spec-stats] … lookup=rounds/drafted/landed`
  and `lookup_table=`. A/B driver: `tests/bench_mtp_lookup.sh`.

## Round cost table

- **Round cost is MEASURED** per model/width/KV bucket from live single-chunk rounds (`round_cost.zig`;
  `SUSHI_MTP_COST_TABLE=0` = prior only); width trials m_lo then m_lo+1 never m_lo−1; the silicon depth row is a
  COLD-START cap.
- **The KV grid is per arch** (`round_cost.layoutFor`): qwen4 edges to 256k; MiMo on to 512k and 768k, since its global
  layers read every key. On one `32k+` cell MiMo dropped nearly every 64k-256k round as implausible against 32k cells.
- **`[spec-stats] … stalls=N/max_ms`** counts this request's rounds slower than twice the median round at their width
  (`round_cost.RoundLog`), so an info-level log shows a hiccup the table dropped quietly; smooth reads `stalls=0/0`.
- **The regime gate** compares the two round SHAPES at one base depth: two-chunk (draft m_lo, sync on the chain's
  confidence, maybe extend to m_hi) against single-chunk at m_lo, each as round wall over tokens. A round emits 1..m+1
  tokens, so each shape is judged on the running mean of `MTP_REGIME_MIN_SAMPLES` rounds or more; a verdict on one
  round per shape judged acceptance luck (the 40-70 ms/tok first verdicts on Flash-Next, then 128 rounds throttled).
- Persistence is OPT-IN (`SUSHI_ROUND_COST_PERSIST=1`); an A/B with the table live measures the TABLE, so set
  `=0` on BOTH arms. A round's wall is between round ENDS, so an interleaved prefill chunk drops the round clock too.
- **The EV seed lives on `Qwen4Mtp`** (`ev_seed_accept`/`ev_seed_m_lo`), per loaded model; publish AND consume
  decline under `SUSHI_MTP_FORCE_DEPTH`. `MtpCostProfile` comes from the runtime fingerprint
  (`g17_nax_qwen4_q4_gs64`; `SUSHI_MTP_QWEN4_PROFILE=0` revokes it); unmeasured = generic/cap-6.
- EXL3 packs take the chip's generic depth row (6 on M5 Max). A deeper round pays only on predictable text: at the
  MCG K3 round costs (31.2 / 36.0 / 42.4 ms at depth 1 / 2 / 3, ~5 ms per row after) and a ~20 ms serial token,
  depth 2 breaks even at ~0.53 per-draft acceptance, depth 3 at ~0.60, depth 4 at ~0.63; depth 4-6 wins only above
  ~0.85, where the model says cap 2 leaves 25-35% (computed from the recorded costs, not yet measured live).
- <a id="adaptive-serial"></a>**Adaptive serial** (qwen4 at kv >= 32k, MiMo at kv >= 64k,
  `MTP_ADAPTIVE_MIN_KV_MIMO`: below it MiMo's rounds beat serial even on prose,
  [perf-baselines](perf-baselines.md#mimo-mtp-vs-serial)): the plan's base width is voted against the bucket's measured serial token
  (table AND this request's 16-round window must both lose by 5%, three rounds running). A serial request re-enters
  MTP when its OWN KV bucket changes: the read bucket maps a never-measured bucket back onto the switch's, so a
  request that went serial at 33k stayed serial to 91.8k (main 36ae6d0, Sushi-3bpw, kv8, temp 1 thinking).
  At 211k keys MiMo prose reaches serial speed while code keeps MTP; all eight replayed responses preserve bytes
  and cache counts ([measurement](perf-baselines.md#mimo-adaptive-serial)).

## Reproducibility

- **Greedy MTP output is serial's byte for byte, auto mode included**: every verify row computes its position with
  the decode tick's arithmetic at any width the plan can pick (`mtpVerifyDraftsMax`: `MAX_DEPTH` drafts on qwen4,
  `MIMO_VERIFY_ROWS_MAX` rows on MiMo), so the width a round takes cannot flip a greedy token.
- `test_mtp_equivalence.sh` compares full output bytes with `--no-mtp` and acquits nothing
  (`test_mtp_equivalence_strict.py`); its servers boot `--prefix-cache-entries 0`.
- **Sampled auto-mode output follows the round times**: the plan reads measured round costs, and the draft counts
  decide which draws land where. A seeded byte comparison pins the plan (`SUSHI_MTP_ADAPTIVE=0
  SUSHI_MTP_COST_TABLE=0`, or `SUSHI_MTP_FORCE_DEPTH`).
- Forced-depth outputs are byte-equal to the pack's own no-MTP greedy (48/48 on two K3 packs).
- `SUSHI_MTP_DENSE_ROWS=1` stays off by default: one `test_mtp_equivalence.sh` run with it on failed (top-2 gap
  1.125 nats, a slow loaded run) and seven reruns passed ([perf-baselines](perf-baselines.md#m2max-decode)).

## Head KV and norms

- **Head KV**: dense by default; `--mtp-head-kv-quant` opts qwen4's into `--kv-quant` (billed at its effective width
  either way, `mtpHeadKvBytesPerToken`; MiMo's heads keep their dense window and the boot log says the flag does
  nothing there, `server.mtpHeadKvLine`); a spec sidecar under another scheme is declined at restore and rewritten
  on the next commit. Head persistence with its QSA half: `tests/test_qwen4_mtp_head_persist.sh`.
- Acceptance modes `exact|typical|tokenv3` (`mtp_acceptance.zig`, per-model `mtp_acceptance`).
  Only sampled decoding reads the mode: temperature < 0.01 (`isGreedyTemperature`, shared with serial sampling) always takes the argmax check (`mtpAcceptRowGreedy`), so
  greedy output is identical under every mode. Sushi-4bpw at T=1.0 / top_k 20 / top_p 0.95, 16 fixture prompts x 512,
  2 seeds, binary b64c5a0e: typical 0.2 decoded 80.0 / 76.6 tok/s vs exact 69.2 / 63.3 (2.4-2.6 vs 2.1-2.2 tokens per
  round); target NLL of the emitted text 0.7819 vs 0.7805 nats, a paired difference inside the seed noise (~0.1 nats).
- **Greedy tail** (`--mtp-greedy-tail`, per-model `mtp_greedy_tail`, off by default): a sampled request draws only
  depth 0 from the draft sampler and drafts every later depth by argmax (`mtpDraftSampling(step)`, the one place a
  depth's proposal is resolved). Those verify rows carry a one-hot q, so `exact` stays distribution-exact and
  `typical` judges the argmax against the target's floor; a greedy request is untouched. A chain that carries q
  drafts every depth as a `[1]` id: the accept graph concatenates them.
- `--fast` turns on MTP with `typical` and the greedy tail (plus kv8) in one flag
  ([server-lifecycle](server-lifecycle.md#settings)).
- Sampled output under `typical` acceptance, and so under `--fast`, has been seen looping (2026-09-28): the README and
  pack cards recommend neither until the cause is known.
- **The greedy tail pays only beside `typical`, and pulls sampled text toward the argmax.** Sushi-2.6bpw at T=1.0 /
  top_k 20 / top_p 0.95, 16 fixture prompts x 512, 2 seeds, binary 506efc5b: typical 0.2 + tail decoded 95.9 / 92.6
  tok/s vs typical 84.0 / 89.0 and exact 74.3 / 74.6 (2.7-2.9 vs 2.5 vs 2.1 tokens per round). Its emitted-text NLL
  fell below exact (0.603 vs 0.680 nats, paired interval [-0.23, +0.01]; non-argmax tokens 19.2% vs 21.6%): less
  diverse text, not a likelihood loss. Exact + tail stayed inside the NLL noise but decoded 72.8 / 72.2 tok/s.
- **Norms**: delta-encoded head norms AUTO-FOLD at load (raw-HF heads get the `+1` repair, `mtpNormNeedsRepair` reads
  the norm's OWN negative fraction, whole-head 5% bar); publish packs FOLDED (the converter folds them).
  Quant re-solved PER WEIGHT; a sidecar's mode is solved from GEOMETRY (`quantParamsFromGeometry`); dense bf16 head
  trunks requantize at load (`SUSHI_MTP_HEAD_QUANT_BITS` 4/g64).

<a id="ple-defer"></a>
## The deferred PLE leaf

The deferred PLE leaf is filled before anything evaluates the build (`ForwardCtx.ple_defer` + `flushDeferredPle`,
set + flushed by BOTH `lazyForward` and the MTP verify build; `pleClaimSpecCapture` claims the spec slot at BUILD
time): a host token read inside the graph build serialized the build with the GPU, and a capture evaluated before
the fill saw a zero PLE.
