# DeepSeek-V4.1 against Qwen3.8-Flash-Next: serving parity

What `qwen4_exp` serves that `deepseek_v41` does not, or serves differently, with the evidence for each row. Geometry,
bills and the DSpark drafter are in [dsv41-arch](dsv41-arch.md); memory is in [dsv41-memory](dsv41-memory.md); this file
is the feature ledger.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Status words: **works** (proven by the named test or live log), **gap**
(named, with a design below), **N/A** (by design, said why), **main-wide** (the same defect on every arch; not a DeepSeek
item).

## Matrix

| Feature (Qwen) | DeepSeek | Evidence |
|---|---|---|
| Native speculative decode, greedy | works; text is the same at every round-width mix (`SUSHI_DSV41_DSPARK_EXACT=0` trades that for the faster arms) | `[spec-stats] mode=dspark` in the server log; `dsv41 generator: with exact DSpark rounds`. Adaptive DSpark picks plain or drafted rounds from measured timings, and each linear has one arm at every verify width, so greedy text does not depend on the mix; it can still part from `--no-mtp` at a near-tie, and a stream and a non-stream reply carry the same bytes ([dsv41-arch](dsv41-arch.md#dspark)) |
| Spec decode for sampled requests | works | point-mass draft, exact accept rule (`Generator.dsparkSampledVerdict`); sampled determinism test |
| Spec decode under repeat / presence / think penalty | works | verify rows shifted by `penaltyShiftRows`, `thinkShiftRows`; distinct-id test |
| Spec decode under a thinking bound (Anthropic `budget_tokens`, `reasoning_budget_tokens`) | works | bound checked between rounds, then serial; `a thinking bound fires between DSpark rounds` |
| Spec decode with logprobs, grammar, forced call | N/A | MTP refuses the same three (`mtpRefusal`); they decode serially |
| Batched decode, several slots | gap | `ownsModuleDecodeState`: one request at a time; `/props` `batching.reason = exclusive` |
| Hot / SSD prefix cache | works | restore points of the whole state, RAM and SSD tiers, bit-identical to cold ([dsv41-arch](dsv41-arch.md#prefix-cache)) |
| Per-request prefill chunk ladder | works | `perRequestPrefillChunk` is true for the arch; each request takes the widest rung its bill admits ([dsv41-arch](dsv41-arch.md#ced)) |
| `--mtp-min-depth` / `--mtp-max-depth` | works | bound the DSpark verify width to depth + 1 rows; the boot log prints `[spec] DSpark verifies a..b rows a round` when either is given |
| `--kv-quant`, `--qwen-gdn`, `--mtp-head-kv-quant` | N/A, refused | no KV cache or GDN; a launch flag refuses the load before the weights (`archFlagRefusal`, `LaunchFlagUnsupported`), a model-settings `kv_quant` is ignored with one log line (`settingsKvQuantIgnoredLine`) |
| Image input | works | the release's processor and tower, on by default (`--no-vision` skips them) ([dsv41-arch](dsv41-arch.md#image-input)) |
| Video / audio input | N/A | refused by name (`mediaRejectReason`) |
| Image token spelled in text-only input | works | `deepseek_v41: the image token spelled in a text-only prompt is text`: no media, no placeholder check |
| Chat template | works | 14 byte-exact cases against the release's `encode_messages`; V4.1 is in the format-corpus family matrices (late system, forced tool_choice, continuation, tool traffic, non-object arguments, forced call parse-back, string value round trip) |
| Tool calls (DSML, both spellings) stream and non-stream | works | corpus `dsv41-dsml` entries incl. invoke markup inside a value and prose naming `<invoke>` |
| Forced `tool_choice` | works | `forcedCallText` reads the V4.1 template |
| Reasoning split, `reasoning_tokens` | works | reasoning is split from content as on Qwen |
| Effort words, default thinking | works | off, low, high, max; medium to high, xhigh to max with `effort_notice`. A request naming no thinking or effort thinks at `high` (`/v1/models` says so); an explicit field wins. No hard thinking budget for any word (Qwen: low 2048, medium 8192) |
| Stop sequences, EOS inside a speculative block | works | generic stop gate; `publishSpeculativeBlock` cuts at the first EOS |
| `ignore_eos`, `min_tokens` | main-wide | chat 400s `ignore_eos` by design on every arch, `/v1/completions` honours it; `min_tokens` is not read on any arch |
| `/v1/messages/count_tokens` | main-wide gap | not served by any arch |
| Anthropic `/v1/messages`, Responses, completions | works by construction | same handlers; the smoke matrix boots the arch |
| `/props` | works, three fields mislead | `mtp.depth` (6) and `mtp.adaptive` describe MTP, not the DSpark block; `model_info.quantization_bits` 0 |
| `/metrics` | works | no acceptance counters on any arch |
| `sushi launch <agent>` | works by construction | rows come from `/v1/models`; the efforts list is the arch's four words |
| `model-settings.json` | works | `ssd_budget_gb`, `mtp`, `ctx_size`; `kv_quant` is ignored |
| `sushi kld` | works | capture and compare through the generic path; `--drop-expert` scores a pruned student ([quality-kld](quality-kld.md)) |
| Smoke and format matrices | works | `tests/test_smoke_matrix.sh`, `tests/test_format_matrix.sh` carry a `deepseek_v41` row |
| Engram read-ahead | works | every pass reads its rows on a thread from `beginPass` ([dsv41-arch](dsv41-arch.md#what-loads)) |

## Larger gaps

Ranked by what an agent client feels.

1. **Prompt-lookup rounds.** MTP replaces its draft with a verbatim copy of the context when the suffix matches
   (`mtpLookupChain`); DSpark never does, and agent edits repeat their context. Design: `mtp_lookup.Index` over prompt and
   generated ids, and in `nextDspark` a match with a strong suffix becomes the verify block (up to `ring_slack` = 16 rows,
   then `dsv41Finish(accepted)` as now); a simple gate (suffix length, expected rows against the 5-row block, since each
   extra row widens the expert union read from SSD). Needs a live measurement of verify ms against rows.
2. **Prefill throughput.** The routed experts' fill and grouped GEMM are about 55% of a warm chunk
   ([dsv41-arch](dsv41-arch.md#ced)). A layer-ahead read of the next layer's guessed experts was tried and cut: it guessed
   about a quarter of the picks and bought nothing.
3. **Batched decode.** The model holds one request's rings, compressed rows, Engram history and DSpark windows
   (`Model.state`). Batching needs that state per slot (a `State` already exists as one struct), a batched
   `extend` over rows of different positions, and the expert union across slots. Large.
4. **Hard thinking budgets per effort word.** Today none; Qwen caps low and medium.
5. **`/props` DSpark fields**, **`count_tokens`**: small, generic, outside this arch.

## Live checks (a real pack)

All with `--ssd-budget-gb` sized to the box, a binary built from the head under test (stamp commit and mtime beside every
number), under the GPU lock.

1. `tests/test_smoke_matrix.sh` with `SMOKE_ARCHES=deepseek_v41 SMOKE_CONFIGS=default,nospec`: every check ok, and
   `dspark: a sampled request drafts`. One boot per cell.
2. `tests/test_format_matrix.sh` with `FORMAT_MODELS=deepseek_v41`: tool-call fidelity, stream and non-stream.
3. Sampled speculation: one request at `temperature 1, top_p 0.95`, 14k to 28k context, thinking off. Expect
   `[spec-stats] mode=dspark` with `per_draft_pct` between the greedy and zero, and decode above the serial figure.
4. Thinking bound: Anthropic `/v1/messages` with `thinking: {type: enabled, budget_tokens: 256}`. Expect
   `[think-bound] reasoning budget 256 reached ...; thought closed` and `[spec-stats] mode=dspark` for the rounds before it.
5. Penalties: `repeat_penalty 1.2` request; expect `mode=dspark` in its stats line and no repeated-token run.
6. Boot log: `[kv-cache] off (default)`, no `[mtp] depth range` line; `--kv-quant 4` (and `--qwen-gdn 16`, `--mtp-head-kv-quant`) each exit at start with the named error before any weights load; a settings `kv_quant` boots with
   one ignored line. A request with no thinking field returns `reasoning_content`.
7. Forced tool call: `tool_choice` naming a declared function; expect the `[tool-choice] forced call` line and a parsed call.
8. `tests/test_dsv41_media.sh`: image input live.
