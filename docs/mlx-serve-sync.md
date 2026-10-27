# mlx-serve sync: what upstream changed and what we took

The ledger of reviews of upstream mlx-serve (`ddalcu/mlx-serve`, the engine this repo forked) against Sushi. Each pass
reads the upstream commits since the cursor, decides per commit whether it speeds up or fixes a served architecture
(`qwen4_exp`, `mimo_v2`, `glm5_next`, `deepseek_v41`), and records the verdict here. The other direction, mlx-serve
consuming Sushi, is [mlx-serve-integration](mlx-serve-integration.md).

Index: [CLAUDE.md](../CLAUDE.md#docs-index).

## Cursor

| | upstream `origin/main` | Sushi `main` | date |
|---|---|---|---|
| fork point | `ef5e667d` | | 2026-09-17 |
| earlier pass | `25e94c33` | | 2026-10-04 |
| previous pass | `a21784b0` | `92446711` | 2026-10-09 |
| **latest pass (reviewed through)** | `7b83ef30` | `eacf1f00` | 2026-10-11 |

Next pass starts at `7b83ef30`.

## How to run a pass

```bash
git -C <mlx-serve checkout> fetch origin
git -C <mlx-serve checkout> log --format='%h %ad %an %s' --date=short <cursor>..origin/main
```

1. Drop what Sushi does not serve: image, audio, music, embeddings, llama.cpp, ds4, nemotron, Linux/CUDA, UI and app.
2. Speed candidates: `git show --stat` each remaining `perf`/`feat` commit and compare with the matching
   `*-perf` and kernel doc's ruled-out list before proposing a port. A lever already ruled out here stays out.
3. Bug fixes: read the upstream diff, then trace OUR code. A verdict is `HAS-BUG` only when the failing input reaches our
   code. Our parsers, grammar and sampler were rewritten after the fork, so a commit message proves nothing.
4. Record every reviewed commit below, ported or not, then move the cursor.
5. Upstream numbers come from other boxes (an M5 Ultra for GLM) and other baselines. They size an idea; they never
   count as a Sushi measurement.
6. A GLM kernel candidate is looked up in the [GLM kernel map](#glm-kernel-map) first; a stage that has a verdict
   there needs new evidence, not a new reading of the same upstream code.

## Speed

| upstream | what | verdict |
|---|---|---|
| `846977ac` | GLM DFlash2: confidence-driven row count over a measured 1..16-row ladder, plain rounds, verbatim copy chains up to 15 drafts, row paths to 16 | **skip**: owner decision 2026-10-11 |
| `c84a5a63` | GLM: fuse the KDA verify step, batch decode across slots | covered: grouped GLM forwards and the KDA tree prework exist; a fused T3 KDA core measured -0.7% ([glm5-kernels](glm5-kernels.md)) |
| `03cc3ca2` | Bonsai ternary M4 verify lane | not served |
| `340016fc` | 16 lanes per output row in the affine MoE down+reduce, G17 | not applicable: our experts are EXL3; the change also alters accumulation order, so it would need the KLD gate |
| `431ac5d7`, `ad65559b` | hyper-connection one-row up launch, one read for 1 to 16 rows | Qwen only, and our HC path is already row-grouped ([qwen4-perf](qwen4-perf.md)) |
| `22f068f6` | KDA decode step probes its threadgroup size | robustness only: `glm5_kda_fused.zig` launches a fixed 1024-thread group with no probe, which only a virtualized GPU with a lower cap would hit |
| `178c43da` | upstream serves Sushi's GLM and MiMo packs in-process | upstream runs our engine; nothing to port |
| `c1fd12e8`, `846977ac` | GLM `mtp.0` head (depth 3), its own M5 Ultra rows: MTP / DFlash2 / serial 61 tok/s, code 86 / 99, edit 91 / 111, copy 157 / 182, JSON 92 / 95, thinking 75 / 70, story 65 / 61, llmprobe 78.0 / 81.6 | **skip**: owner decision 2026-10-11 |
| `89283f81` | simd_qmm for affine projections of 8 to 16 rows on M1-M3 | skip: M1-M3 without a tile only; our GLM groups stop at four rows and run their own affine row tiles |
| `d9b3243a` | `prefillDqGemm` off by default on CUDA | not served; the same dequantize-plus-dense-GEMM lever is ours for A6 at exactly 2048 rows, with its ruled-out range ([glm5-kernels](glm5-kernels.md)) |

MiMo: no upstream performance commit since the fork that Sushi lacks.

**Skipped by the owner (2026-10-11): longer verbatim-copy chains on GLM** (`846977ac`). Ours is fixed at 3 drafts (`Generator.glm_lookup_drafts`,
a four-row verify, `glmLookupProposal`); upstream copies up to 15 and reports copy at 182 tok/s against 61 serial (3.0x,
M5 Ultra, affine experts) where we record 1.6x ([glm5-arch](glm5-arch.md)). The box decides it. Upstream's verify ladder at 1K
context is 17.7 ms for one row, 25.5 for two, 33.6 for four, 47.9 for eight, 69.1 for sixteen (`dflash_policy.zig`), about 3
ms per row past the fourth. Ours is EXL3, ALU-bound in the trellis decode: 29 ms for one row, 39 for two, 42 for three, 51 for four
([glm5-kernels](glm5-kernels.md#decode-one-evaluation-per-token-submitted-every-four-layers), [glm5-perf](glm5-perf.md#glm-w14-lanes)),
about 7 ms per added row inside four rows and unmeasured beyond. At 95% acceptance per draft, a three-draft chain emits 3.7
tokens (73 tok/s) and a seven-draft chain 6.7: 85 tok/s at 7 ms per row past the fourth (+17%), 74 at 10 ms (+2%). The
chain pays only if rows five to eight cost under about 10 ms each; estimates, not measurements.
- **Step one needs no kernel.** `verifyGroups` already takes 16 rows and the generic paths serve them, so timing one
  chain group of 5 to 8 rows beside the four-row chain on a copy prompt gives the ladder; the rows ubench
  (`SUSHI_GLM_ROWS_UBENCH`) times B one-row requests, so it needs a chain-group arm.
- **If the marginal row is under about 10 ms**, the work is the paths capped at four rows: the lane and group2 experts
  (`glm_group2.max_rows` = the overlay's `max_tail`), MLA B3/B4 and the overlay tail, the affine row tiles, the SIMD32 HC
  collapse (one to four rows). Each falls back to a slower exact arm beyond four, so each is worth its own gate.
- **Then** upstream's `CopyStats` (hit rate per match length, forgetting) is a small way to pick the chain length.
Its other policy parts are already ruled out: the confidence-driven row count and the drafting backoff are the measured
cost planner that matched the fixed plan within noise, and upstream's gains for them (JSON +19%, thinking +19%, edit +44%) are
against its own fixed eight-row block with a sticky gate, not against a two-node tree ([glm5-kernels](glm5-kernels.md)).

**Skipped by the owner (2026-10-11): GLM MTP head** (`glm_mtp.zig`). The checkpoint's `mtp.0` layer (one plain DSA plus routed-MoE block behind `eh_proj`, no
HC) drafts at depth 3 upstream; the shipped pack carries it (`layers.45.*`) and nothing here reads it. Upstream's own
rows size it: 1.28x serial on llmprobe against DFlash2's 1.34x, ahead of DFlash2 on thinking and story text where a tree
of two drafts barely pays. For us it matters where DFlash2 is absent: its assistant is a separate download the pack does
not ship, and without it a request decodes serial (about 32 tok/s here). Cost is L: head forward on EXL3 experts, its own
latent and pool cache (1,536 B per token in BF16), prefix and SSD snapshots, admission bill, rollback, the final-normed
hidden capture, and the layer-45 name mapping. Output stays serial-identical while the verify rows keep their contract.
The unknown is acceptance on our packs, which can be measured without engine code: run mlx-vlm's `glm5_next_mtp` drafter
over the 4x512 teacher prompts and read depth-1 to depth-3 acceptance on code and prose. Build only if depth-2 emits about
1.9 tokens per round or more on prose, the break-even of a two-draft round here.

## GLM kernel map

Upstream (`7b83ef30`) built GLM independently after the fork: the trunk is composed MLX ops with oMLX-derived Metal kernels
(`glm5_next.zig`, `kda_recurrence.zig`, `dflash_policy.zig`, `glm_mtp.zig`), and its EXL3 experts are our `sushi_exl3` module
at the v1.2.1 pin, so they are not an upstream optimization. One line per stage; a verdict here outranks a reading of the code.

| stage | upstream | Sushi | verdict |
|---|---|---|---|
| MLA decode attention | `glmDsaAttn` over `latentAttendFolded` (64 heads folded into the GEMM M, activation-dtype scores); `headQmv`, oMLX's per-head qmv, with the attention scale folded into `embed_q`'s rounding | B1 FP32 GEMMs, block-masked on NAX, precise softmax (`glm5_attention_decode_batch.zig`); absorb and unembed through MLX's batched `quantized_matmul` | skip: `headQmv` built bit-identical without the scale fold and measured 2026-10-05 on `c3f29b8e`: 11 MLA layers 1.20 to 1.09 ms per token at one row, about 0.3% of a 38 ms token, under the model gate; the scale fold changes rounding |
| MLA verify, 2 to 16 rows | folded causal attention under the indexer budget, a one-query select per row past it | B3/B4 batched FP32 attention with a four-row ancestry overlay; wider groups settle per node through B1 | open: part of the chain candidate above |
| MLA prefill and indexer | `selectTokens`: NAX scores for any pool count (FP32 relu and head weight), argpartition, chronological sort; `sparseLatentNax` reads keys through the indices, P as fp16 hi and lo | head-packed gather plus SDPA (B16/B32); NAX scores only for 9-16 rows over 3584-8192 pools with the BF16 epilogue; ranked top-512 | skip: indexed K/V loads measured 85% slower here, NAX scoring past 8192 pools fails the drift screen, and FP32 scores rank differently from the reference ([glm5-kernels](glm5-kernels.md)) |
| decode index select | `decodeSelect`: one dispatch scores, one threadgroup radix-selects | tree scorer, ArgPartition | skip: ranked decode selection is ruled out |
| KDA prefill | `kda_recurrence.recur`, oMLX's per-core kernel: one threadgroup per GPU core, 12-token staging, gate evaluated in staging; `planFor(64, 128, 40)` is null | R4 value rows, one-dispatch prework, exact | skip: needs 64 or more GPU cores (an M5 Ultra, not our 40), another reduction order, and our staged schedules lost |
| KDA decode and verify | `decodeStep`: a threadgroup per head, up to 8 rows a launch (two launches to 16), every row's state captured in-kernel, the second-stage gate projections (`f_b`, `g_b`, quantized) inside the kernel, QKV, FA, GA and beta as one joined projection | the same oMLX body for one token; QKV in one dispatch over three banks; FA, GA and beta as separate BF16 GEMVs, FB and GB inside the body for dense BF16 weights (separate GEMVs for affine, FP8 and the covariance tap); verify is the tree prework tape plus a replay of the accepted path | landed by owner decision: the in-kernel BF16 `f_b`/`g_b` is bit-identical and measured 28.24 -> 28.18 ms/token on GLM Sushi-2.4bpw ctx 1K (-0.2%, inside noise) |
| router and top-k | register top-k shared with Qwen | FP32 GEMV, stable top-8, two dispatches | skip: recorded dead end |
| routed experts | EXL3 through `moeClamped` (ours, `3f3ff7bc`); affine `moeDecode` (`glm5_moe_*.metal`) for oQ packs | EXL3 lanes, group2, the T2048 grid | nothing to take: affine kernels are not served |
| dense and shared FFN | shared expert inside the two-dispatch `moeDecode`; generic `prefillDqGemm` | `moeWithShared`, shared expert inside the routed reduce at three to four rows, A6 dense-once at 2048 rows | Sushi ahead: the same levers, measured |
| HC | `hcPre`: one dispatch for RMS, mixes, gates, collapse and norm (cross-threadgroup counter), the expand deferred into the next `hcPre`, rows to 16; prefill mixes by simdgroup matrices from 512 rows with the RMS applied after the dot | SIMD32 collapse on one to four rows, RMS-fused 24-output projection (exact), expand plus norm fused at three rows | skip: whole-prep fusion measured 0 to +2% on top of the SIMD32 collapse (`de867e94`); the deferred expand uses fast exp, fast divide and an FMA order our exact chain does not, and factored RMS fails the drift bound |
| DFlash2 draft | composed MLX ops plus a fused dynamic conv (`dflash.zig`) | fused A4 pipeline, about 4.1 ms ([glm5-perf](glm5-perf.md#glm-round-levers)) | Sushi ahead |
| DFlash2 verify and policy | `dflash_policy.zig`: row count by calibrated confidence, plain rounds, drafting backoff, copy chains to 15 | fixed N2 tree plus a three-draft lookup chain | policy, backoff and longer chains ruled out (owner) |
| MTP | `glm_mtp.zig`, depth 3 | none | skip (owner) |
| concurrency | batched decode, 8 rows a forward | grouped forwards, 4 rows | cap stays: 14.5 ms per extra row on EXL3 |
| prefill schedule | the generic path, eval every 4 layers, no per-token stream bill on NAX | 2048-row chunks, two layers pending; the chunk is a numerics decision | nothing to take |
| prefix cache | shared hybrid cache plus SSD tier, MTP history in snapshots | KDA checkpoints, MLA rows, assistant window | Sushi ahead; upstream's poison on a failed restore is a gap (below) |
| KV | BF16 or generic kv8/kv4 including the index history | kv8 latent, lossless pool | Sushi ahead |

## Bug fixes

| upstream | what | verdict | status |
|---|---|---|---|
| `9c0ca7fe`, `3129a772` | json_schema: `$ref` resolution, typeless nodes accept objects, cyclic `$ref` relaxes to any, `~0`/`~1` | **HAS-BUG**: no `$ref` handling; a typeless node forbids every key, so `items: {"$ref": ...}` completes only as `{}` (`response_format` without tools) | fixed: `c7817d3b` (typeless nodes), `d73273a5` (`$ref`, cyclic relax, `~0`/`~1`) |
| `f3da8d69` | `min_p` from the request and `generation_config.json` | **HAS-BUG**: the field is dropped silently | open; L (five sampling sites, spec/MTP densities must stay identical); only if llama.cpp-style clients matter |
| `6ddac95f` | tool name escaping in the stream delta; name ends at `>`, newline or `<` | partial: delta already escapes; a missing `>` drops the call and leaks the tool markup as content | fixed: `88c7773f` |
| `348fd414` | `/v1/completions` token-id prompts, `echo` refused | **HAS-BUG (minor)**: ids get a misleading 400; `echo:true` is ignored | fixed: `5bbf5f5d` |
| `2baca521` | forced tool opener must not end on a lone `=` | partial: no double open, no wrong name; the opener ends on `=` and names are encoded without it, so a multi-tool `required` choice starts off-distribution | open; needs a live multi-tool `required` test before changing |
| `8f7f79f3` | coerce `oneOf`/`anyOf` object params from JSON text | not affected: `declaredJsonType` already resolves unions | closed |
| `41896655` | a grouped MTP verify's GDN rows take the solo verify kernel | **HAS-BUG**: grouped Qwen verify (`verifyRowsJoined` into `gatedDeltaNetProjected`) runs the composed chain while solo verify and serial use the fused kernel, so a greedy answer can depend on who shares the round | open; S (lift the guard in `transformer.zig`), then run the row-axis test with `QWEN4_TEST_MODEL` |
| `5596adba` | hybrid tool round resumes where decode ended, not ~31 tokens before the prompt end | **HAS-BUG (speed)**: the newest SSM checkpoint sits 30 tokens before the prompt end, so a Qwen agent round re-forwards 30 tokens plus the whole previous reply | open; M (snapshot the live SSM state at the committed length; the MTP tail hidden and the QSA history need care); the largest speed win in this pass |
| `2137bd00` | release every handle on MTP head forward error paths | partial: leaks exist only under test fault injection, not reachable in serving | open; S errdefers, low value |
| `a3804125` | auto `--max-resident-mem` is Metal's working-set limit, not 80% | partial: we cap at 80% but a lone model loads past the cap; only co-residence is capped | optional; S, loosens eviction headroom |
| `431b3537` | omp sends `reasoning_effort` | not affected: `launch.zig` already does | closed |
| `a2419eba` | tokenizer: `\s*[\r\n]+` ends at the run's last newline, `\s` is Unicode White_Space (U+3000, NBSP) | not affected: GLM, Qwen and MiMo take the exact word rules (`matchRuleNewline`, `unicode_props.whitespace`); only the `.legacy` path and the cased Llama-3 grammar share the flaw, and neither serves a model | closed |
| `46a59ae8` | a loop convicted inside an open think block closes the thought instead of ending the turn | **HAS-BUG**: `loopGuardTick` cuts a thought-only reply with `finish_reason` stop and no tool call, which an agent reads as a finished turn | open; M; do NOT arm an unreachable `think_bound` at request start as upstream does: `glmDflashEligible` refuses DFlash2 for any request carrying one, so build the closing line lazily when the loop is convicted |
| `d9b3243a` | request reader: a stalled or dripping client is dropped after 30 s, a wrapping `Content-Length` is a 413, `Expect: 100-continue` is answered | **HAS-BUG (hostile input)**: `handleConnection` reads with no deadline, so a silent connection holds its thread, and `header_end_pos + cl` wraps past the 413 check at `Content-Length` 2^64-1 | open; S |
| `d9b3243a` | an SSD entry whose restore fails is poisoned, not retried | partial: `restoreGlmFromDisk` and the other restore arms fall back to cold but leave a same-size rotted chunk to fail again on every lookup | open; S (`poisonId` on `error.MlxError`) |
| `d9b3243a` | a load's requester takes its reference at the claim; a failed load drops its latched MLX error | **HAS-BUG (rare)**: `ensureLoaded` refcounts after `.ready`, so a concurrent load can evict the model in that window, and `runLoadRequest` leaves a failed load's error latched for the next request | open; S |
| `94670d96` | OpenAI `frequency_penalty` read as `repeat_penalty` 1 + x (#912), plus purgeable pages in the memory bill (#932) | **HAS-BUG**: `parseRepeatPenalty` and the Responses reader turned `frequency_penalty` 1.0 into a 2.0 multiplier on every seen logit; negatives clamped to 0. The #932 half is app/status code we do not carry | fixed: its own `SamplingParams.frequency_penalty`, subtracted per occurrence in `seen_penalty` (serial, MTP verify rows, lazy sampler); `presence_penalty` and `frequency_penalty` read -2..2 |
| `fdb7d36a` | Claude-style `<function_calls><invoke name=...>` calls on qwen4_exp (#928); the rest of the commit is console login, registry eviction, embeddings and the app | **HAS-BUG**: no parser knew a bare `<invoke`, and neither the stream gate nor the scrubber held or cut it, so the call shipped as content with `stop`. The other parts are not served or not ours | fixed: `BARE_INVOKE` through the ATEM parser (`parseInvokeToolCalls`), gate and scrubber openers, corpus family `qwen4_exp` |
| `e6a35b14` | the embedded n-gram table's width read off the shard geometry instead of 4-bit only (#936) | not applicable: sushi reads no table embedded in the weight shards (no `qwen4_ple` reader); `NgramTable` takes its width from the `ngram_table.bin` header (`bits`, any of 2/3/4/5/6/8, checked against `dim * bits == cols * 32`) | closed |
| `d9b3243a` | empty-prompt 400, settled availability after an unload, bounded SSD flush with a parked snapshot (#797), JSON nesting cap | covered: `EmptyPrompt`, `settleAvailableBytes`, `flushPendingRecord` lands an earlier pending record first, `jsonNestingExceeds` | closed |
| `00337561` | available memory is the reclaimable page-class sum, not total minus wired, compressor and anonymous | partial: `computeAvailableBytes` still double-charges a resident model's wired pages (21.6 GiB live upstream); the wired-limit floor in `gpuCeilingWithWiredFloor` hides it only when `iogpu.wired_limit_mb` is raised | optional; S, on a loaded box first |
| `2e22c489` | a `PrefillDoesNotFit` refusal quotes the bytes it compared | message only: ours is a fixed string | optional; S |
| `b0145204` | `/v1/completions` `echo` with prompt logprobs | feature: ours refuses `echo` by name (`echoRejectReason`); GLM prefill reads only the last row's logits, so prompt logprobs would be a new head pass | skip; L |

Items of the previous pass, checked as done: GLM tokenizer `ignore_merges` (`7c7bdcb1`), predraft after budget/EOS
(`9a6a605c`), K/V 16-byte alignment guard (`aec87e66`), request-outcome metrics counted once (`518d887b`, `0f0b1adb`),
disk free-space probe only before a store (`fecf6f35`), MTP sidecar admission (`00ea1b23`). Router top-k registers and
counting-sort routing are recorded dead ends.

Ranked next ports: (1) typeless/`$ref` schema nodes, a valid request returns wrong content; (2) `41896655`, one line that
restores grouped = solo verify; (3) `5596adba`, the speed win for every Qwen agent tool round; (4) `46a59ae8`, a thought-only
`stop` ends an agent's turn; (5) the request reader's deadline and wrapping length.

GLM speed work from the map: none. The in-kernel BF16 gate projections measured inside noise; longer copy chains and the
MTP head are owner skips; per-head MLA GEMV, fused
or deferred HC, per-core KDA, the NAX indexer and sparse kernel, and the confidence-driven row policy stay out.

## Not reviewed

Everything the filter in step 1 drops, and upstream commits dated after `7b83ef30`. This pass dropped: the Linux/CUDA
work (`34ca46f2`, `20cec717`, `1b00c65a`), the app and web console (`7203ac0f`, `ed11df73`, `4d1eb3ba`, `ada6187f`), media
and launcher features (`30768579`, `3563ee8e`), the release commit's Kokoro, MiniMax and LoRA parts (`d9b3243a`), DSpark
opt-outs on the embedded engines (`97f10c36`) and the Linux CI cache change (`7b83ef30`).
