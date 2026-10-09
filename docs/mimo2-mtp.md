# MiMo-V2.6-Flash: the three MTP heads

MiMo's three MTP heads, split out of [engine-mtp](engine-mtp.md); everything shared by the served architectures stays there.


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
    ([perf-baselines](mimo2-perf.md#mimo-verify-global-rows)).
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
  built the lm_head ([perf-baselines](mimo2-perf.md#mimo-mtp-round)). A prefill chunk's capture still settles.
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
