# Server: HTTP APIs, streaming, sampling and constrained output

What each HTTP surface promises its clients: the OpenAI chat/completions/Responses and Anthropic Messages contracts,
streaming and usage chunks, logprobs, seeds and sampling, reasoning budgets and constrained JSON, plus the agent
launcher. Read this before touching `src/server.zig`, `src/http_conn.zig`, `src/anthropic.zig`, `src/responses_api.zig`, `src/completions.zig`, `src/generation_stream.zig`, `src/request_policy.zig`, `src/responses.zig`, `src/launch.zig` or the sampler.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [server-tool-calling](server-tool-calling.md),
[server-lifecycle](server-lifecycle.md), [engine-mtp](engine-mtp.md).

## Surfaces

- OpenAI completion and Anthropic message IDs are opaque and process-unique even when the wall clock repeats or
  moves backwards; all OpenAI SSE chunks of one response keep the same ID.
- **OpenAI chat/completions + Responses**: usage ALWAYS carries `prompt_tokens_details.cached_tokens`; thinking
  opt-ins = `reasoning_effort` OR `enable_thinking` (top-level, else vLLM's `chat_template_kwargs.enable_thinking`;
  `reasoning_budget_tokens` outranks); a request naming neither takes the arch default (`defaultEnableThinking`);
  on Responses a `reasoning` object decides alone, and without one the same rule applies; `n>1` 400s.
- **Effort vocabulary** `off on minimal low medium high xhigh max` (`none` = off; `minimal` keeps the legacy 1024
  budget on inherited arches). Qwen, DeepSeek and GLM each serve a **thinking-effort table** (`effort.Table`): one row
  per word (`off low medium high xhigh max`) with `offered`, `sends` (what the template's `reasoning_effort`
  receives) and `cap` (thinking tokens; null = no cap), plus a default word. The built-in tables
  (`effort.builtin`) are the engine's own; `model-settings.json` holds only the fields an operator changed and the
  merge is field by field (`effort.merge`; [Effort tables](#effort-tables)). Every request resolves its word against
  the served table once (`effort.servedArms` → `effort.resolve`; a request reads every family's table in one
  `effort.snapshot` and resolves word, default, cap and refusal text against it) on chat, Responses, Anthropic, `--think` and the
  REPL's `/think` alike; the arm's `sends` is what the render passes (`ReasoningEffort.sends`,
  `model.defaultTemplateEffort`), its cap the decode-time bound. A word the table does not offer is refused with the
  accepted list, and so is an effort field that holds no word (a number such as 40, on `reasoning_effort`,
  `reasoning.effort` or `output_config.effort`; `request_policy.nonWordEffort`); null reads as absent. An unoffered `off` refuses every way a request turns thinking off (`enable_thinking: false`,
  `chat_template_kwargs`, Anthropic `thinking: disabled`) with the same 400, as GLM's off always is
  (`request_policy.thinkingOffRefused`); a model that thinks off by its own default still serves a silent request. **Folds** are a row state of the built-in tables only: Qwen `high` and `max` → `xhigh`, GLM `medium`
  → `high` and `xhigh` → `max` (each word the family has no level for uses the nearest one). A fold is served at its
  target, logs `[effort]` and rides back as `X-Sushi-Effort-Notice` plus `effort_notice` (not in the answer text); a
  fold whose target is not offered is refused. A row the file sets `offered` on is no longer a fold: true serves the
  word as itself (its own `sends` and `cap`, no notice: a mapping, not a re-route), false refuses it. No other word
  folds. MiMo's silent on-fold is unchanged (it has no table).
- qwen4_exp (built-in): off, low (2048), medium (8192), xhigh (uncapped); the template reads the word (`sends` takes
  low|medium|xhigh, which the template accepts); `high` and `max` fold to `xhigh`; `minimal` still 400.
- deepseek_v41 (built-in): off, low 50, medium 62, high 75, xhigh 87, max 100 on the encoder's 1-100 scale (`sends`
  takes any whole number 1-100; 50, 75 and 100 reach the encoder as its own words, `chat.dsv41EffortFor`); no caps.
- glm5_next (built-in): low, high, max (uncapped, a prompt instruction; `sends` takes low|high|max); `medium` folds to
  `high` and `xhigh` to `max`; off can never be offered (the template has no thinking-off switch).
- mimo_v2 advertises off, on: its template has only on/off, so every thinking word (`minimal` through `max`) is on,
  uncapped, and off/none is off.
- `/v1/models` rows, loaded or not, are read from the served table on every call: `reasoning_efforts` (the offered
  words; folds are not listed), `default_reasoning_effort`, the word a request naming none runs at
  (`model.defaultReasoningEffort`: `--think`, else the table's default word, else off/on per `defaultEnableThinking`),
  `reasoning_effort_budgets`, the thinking-token cap of each capped word (built-in: Qwen `low` 2048, `medium` 8192;
  `{}` for MiMo, GLM and DeepSeek), and `reasoning_effort_values`, the template value of each offered thinking word
  (a word for Qwen and GLM, a number for DeepSeek: `{"low":50,"medium":62,...}`; `{}` for MiMo).
- CLI `--think X` requires an explicit effort word and is checked against the `--model` model only; a model loaded on demand takes X by the same rule (MiMo:
  any X but off is on) or, lacking it, keeps its own default. It never fails a load.
- **Qwen thinks at medium by default** (owner decision): a request naming no thinking field (no `enable_thinking`,
  `reasoning_effort`, Anthropic `thinking` or Responses `reasoning`, as `sushi run` sends none) runs with thinking on
  at the table's default word, built-in `medium` and its 8192-token cap (`model.tableDefaultArm`, read by
  `defaultEnableThinking`, `defaultEffortWord`, `defaultTemplateEffort` and `defaultThinkBudget`). Precedence: the
  request's own fields, then `--think`, then the
  table's default word (the file's, else the built-in one); a checkpoint whose `generation_config.json` declares
  `enable_thinking` false stays off unless the file names a default word (the operator outranks the checkpoint, as a
  flag does). A file default of `off` makes a silent request think off. Qwen under an `off` default (`--think off` or
  the table's) stays off for a request naming no effort word, its `enable_thinking: true` or Anthropic `enabled`
  included; an effort word, or a budget (`reasoning_budget_tokens`, `budget_tokens`), still turns it on, a budget alone
  at low (`request_policy.holdOffDefault`, on chat, Anthropic and Responses alike). Explicit thinking off stays off. MiMo and GLM keep their defaults. A thinking
  request on another arch that names NO effort gets no server budget: Qwen3.8's template renders an absent word as
  low (`chat.qwen38EffortFor`), whose preamble shortens the thought without truncating it. `/v1/responses`:
  `sequence_number` on every event, stateful via
  `ResponseStore`, WS via Upgrade. A replayed `reasoning` input item (`content[].reasoning_text`, else
  `summary[].summary_text`) joins the following message and calls as ONE assistant turn's `reasoning_content`, and the
  store keeps it in the same field, so an input replay and a `previous_response_id` chain render the same prompt;
  an encrypted, empty or malformed item is dropped, never a 400. Continuing a partial reply: `continue_final_message` explicit on chat, INFERRED on
  `/v1/messages`.
- **Anthropic `/v1/messages`** (Claude Code): typed blocks, `input_schema`→`parameters`, stop-reason map incl.
  `stop_sequence` echo, full SSE block lifecycle (a block opens only once the open one is stopped, so each owns its
  index: `closeAnthropicBlock`); `usage.input_tokens` excludes `cache_read_input_tokens`, which clients add back, and
  `message_delta` restates both since the cache hit is known only after prefill; a `system`-role message past index 0
  (Claude Code's hook output, Codex's mid-input `developer` turn on Responses) renders where it was sent, folded only
  for a template that cannot place it ([server-tool-calling](server-tool-calling.md#templates)); `developer` reads as
  `system` (`canonicalRole`).
- **`/v1/messages/count_tokens`** takes a `/v1/messages` body (`max_tokens` optional, ignored) and answers
  `{"input_tokens": N}`: the prompt `/v1/messages` prefills for it, through the same `renderAnthropicPrompt`.
  Image and video rows count as `placeMedia` expands them, without the tower (`placeRequestMedia`); a body
  `/v1/messages` refuses for its shape is refused with the same Anthropic error, and the model routes as there.
  Nothing generates or queues, the prefix cache and the tokenize LRU are untouched, and a prompt past the context
  is still counted (only `/v1/messages` 400s it). The effort word counts as the word it is served at (the template
  renders it), and a rerouted one carries the effort notice, header and field, as on `/v1/messages`. The count equals
  `usage.input_tokens + cache_read_input_tokens`.
- **`preserve_thinking`** (templates that read it: Qwen3.8), resolved per request as `chat_template_kwargs.preserve_thinking`
  (bool, all three surfaces) > `--preserve-thinking on|off` > `preserve_thinking` in model-settings.json > the template
  default (undefined: every turn's thinking kept). Off renders only the latest user turn's thinking.
- `tool_choice` reads every surface's wire shape; `required`/`any` and a named function are enforced at decode, a
  named function the request does not declare is a 400 ([server-tool-calling](server-tool-calling.md#tool_choice)).
- `/v1/models` `bytes_on_disk` is the bytes the pack reads, symlinked shards included; `bytes_linked` (0 when none) is the part
  that comes through symlinks to files outside the pack folder (a symlink inside it counts as the pack's own), so disk use is
  `bytes_on_disk - bytes_linked`. The Models tab details show it as "linked from elsewhere".
- `/v1/models` rows carry `context_length` + `max_model_len` at TOP level. Context-overflow 400s name BOTH counts.
- A prompt that tokenizes to zero tokens is a 400 `invalid_request_error` and never reaches generate: completions
  refuse it after tokenizing, `Scheduler.submit` refuses it for every surface (`error.EmptyPrompt`), and
  `Generator.initWithOptions` returns that error rather than index the prompt's last token.
- `/v1/models` `meta.default_max_tokens`, `default_temperature`, `default_top_p` and `default_top_k` on a loaded model's row are what a request that omits the field gets (`--max-tokens` / the `max_tokens` setting / the architecture's own, 0 = the rest of the window; then a launch flag > `generation_config.json` > the hardcoded value, `request_policy.sampling*`); the chat page's placeholders read them.
- `/v1/models` `meta.pack_update` (`{"current":"1.0","latest":"2.0","command":"sushi pull deepseek-v4.1-flash","repo":"beamster/..."}`) is present on a loaded model's row only while Hugging Face publishes a higher `pack_revision` of its pack ([usage](usage.md#sushi-pull-and-sushi-list)); absent otherwise, offline, or under `--no-update-check`. The check (`pack_update.zig`) runs on its own thread after the load and then daily, never on the inference thread; `sushibar` and the Models tab show it. The tab never offers `sushi pull` for an update (it writes a second copy): its callout opens the Models tab's Available list at the row of `repo`, which updates in place.
- `/v1/models` `meta.quantization` reports EXL3’s configured expert rate and dense width (e.g. `EXL3 3bpw experts, 8-bit dense`) for loaded and unloaded packs; affine labels remain `{bits}-bit`, and `/props` numeric quantization fields retain their dense-trunk meaning.
- Endpoint EXISTENCE never depends on model state and the 404 is answered BEFORE the model resolves (`ROUTE_PATHS`);
  a status route never reaches `ensureLoaded` (`handlePropsNoModel`). Removed upstream routes answer named 404s.
- **Only an unknown NAME falls back to the default model** (SDK names like `gpt-4`); a PATH (`/…`, `~/…`, never
  `org/repo`) names its own entry as `/v1/load-model` resolves it (`routeRequestModel` → `registry.peekPath`), and an
  unregistered path is a 404 `model_not_found` (Anthropic `not_found_error`), never another model's answer.
- A content array's text parts JOIN in order (`joinedTextParts`); its media parts render at the offset they sat at.
- A body field the server does not read is ignored, never rejected: a client still sending the removed `kv_attn_mode` is served as if it were absent.
- **Media is read from EVERY message** on all three surfaces: chat `image_url`/`video_url` parts in any role
  (`tool` included), Anthropic `image` blocks beside the text or inside a `tool_result`, Responses `input_image` in
  a message or a `function_call_output` array. Only base64 data URLs decode (remote URLs are refused, not fetched);
  a failure is a 400 naming the message index and the reason, an `input_audio` part a 400 on a model without an
  audio encoder, more than 1024 images a 400 with both counts, an encode failure a 500 (Anthropic: `api_error`).
  Stored Responses history keeps text only.

<a id="during-the-load"></a>
## During the launch model's load

A `--model` boot listens once the load has passed its refusals ([server-lifecycle](server-lifecycle.md#startup)), and
answers while the weights load:
- **`/health` reports the boot** (`boot_gate.healthFor`): 503 `{"status":"loading"}` during the load, 200
  `{"status":"ok"}` once the model is ready and its boot is done, 503 `{"status":"error"}` while a failed boot exits.
  A readiness poll (the `run` REPL, `tests/lib/boot.sh`, scripts, load balancers) thus waits for the model as it did
  when the port opened only after the load; a liveness probe reads any answer. Headless boots answer 200 at once.
- The page (`/`, `/chat`, `/dashboard`, `/models`, `/settings`), `/v1/models` (the launch model's row has
  `"state":"loading"`), `/props` (the no-model body with the live memory counters), `/metrics`, `/metrics.json`,
  `GET`/`POST /v1/settings`, `GET /v1/prefix-cache`, the stored Responses routes and `/v1/models/rescan` answer at once.
- A generation request (and `/tokenize`, `/detokenize`, `count_tokens`, `POST /v1/load-model` for the launch model)
  waits in `ensureLoaded` for the load, as for a cold load, then at the boot gate until the context is pinned; a
  failed load answers it with the load's named error (`sendLoadFailedResponse`, `sendNamedLoadRefusal`). A
  prefix-cache clear waits for the inference thread, so for the load; `/v1/update` is refused as busy.
- Guard: `tests/test_early_listener.sh`; the `boot gate:` and `ModelRegistry: a launch model claimed …` unit tests.

## Streaming

- Chat usage reports `completion_tokens_details.reasoning_tokens` by tokenizing concatenated delivered reasoning; streaming includes it only with `stream_options.include_usage`.

- **A stream and a non-stream answer are the SAME BYTES**; whitespace is the one thing a stream may withhold: a
  leading run (`streamContentLead`), and whole whitespace tokens until text follows (`ContentStream`). A spent reasoning budget WITHHOLDS the rest of the thought; a non-stream tool-call reply
  carries the pre-markup text (`visibleToolPreamble`); a non-stream disconnect reports `client_disconnect`, never
  `length`, and is noticed after every decoded token as well as during prefill, so a client's timed-out retry never
  leaves a ghost decoding to `max_tokens`; request ints clamp (`parseRequestSeed`,
  `clampJsonI32`).
- **A stop sequence cuts at the EARLIEST occurrence in the text** (shortest stop at the same start, never the first one
  listed): `stop_sequences.earliest` on every non-stream surface, `stop_sequences.Gate` on every stream, ahead of the
  reasoning/content/tool paths. The gate HOLDS back a tail that could still complete a stop (or an earlier match) and
  flushes it at the end; the `format corpus` stop test pins stream == non-stream for any token split and stop order.
  The gate also owns the UTF-8 carry (a token ending mid-character), and the end of the generation releases carry and
  held tail together, so a stream cut off inside a character ends with the same bytes the escaper sanitizes in a
  non-stream answer.
  A non-stream surface also feeds the gate per token (`EarlyStop`) and cancels the slot when a stop completes, only if
  the whole decoded text holds it too, so the answer bytes stay `earliest`'s and generation ends within one token.
  The cancelled slot is quiesced (`Scheduler.quiesce`, waits out `in_pass`) BEFORE its statistics are read, and usage
  counts the tokens returned, not the ones a speculative block decoded past the stop.
- **`stream_options.include_usage` chunk ships `"choices": []`** (`sendSSEUsageChunk`); the ending appears on exactly
  ONE chunk; a client cannot time our stream — use the final chunk's server `timings`.
- Liveness is a property of the SOCKET: `beatStreamKeepalive` at the bottom of every streaming loop, emit on 5 s
  byte-silence. SSE comments keep the transport alive but do not count as model progress for every client.
  While a stream waits in the queue behind another request, the comment reads `: queued behind N
  request(s); running G of M tokens (~S s at R tok/s)` (G/M: the running request's generated tokens and max); an
  Anthropic stream gets it before each `ping`.
  `sushi launch omp` sets the Sushi provider's `compat.streamIdleTimeoutMs: 0` so buffered calls can finish;
  explicit omp timeout settings, environment overrides and per-call options still take precedence (verified with
  omp 18.3.0). Non-loopback `--url` targets also have a separate first-event deadline: omp's
  `providers.streamFirstEventTimeoutSeconds` controls it, with zero allowing unlimited initial buffering.
  `--timeout` remains a token-progress STALL timeout (`degenerate.StallClock`).
- **NO string built from model bytes is guaranteed UTF-8**: sanitizing lives INSIDE the escaper (`chat.utf8Next`
  under every `jsonEscape`/`appendJsonString`); logprobs `bytes` keeps the exact bytes. Hand-written error text is
  escaped at the SINK.
- **A name the server did not write enters a JSON body through `json_text`** (`Str` in a `{f}` slot, `innerAlloc` for a slot inside quotes): a model id or error name in `/v1/models`, `/props` and the load/unload answers, the echoed request `model` of the chat, completion and Anthropic answers, and the flag text of a `/props` setting. A folder named with a quote or a backslash lists as valid JSON.

### Closing a streamed response

`Conn.close` half-closes the write side of a close-delimited response, then holds the socket until the peer
hangs up, server shutdown starts, or five minutes pass. This prevents macOS's orphaned FIN_WAIT_2 timeout from
resetting a client that is still reading after the final SSE event. The request has already released its model
slot; only the connection thread waits. Responses with `Content-Length` close immediately.

Ported from [mlx-serve #673](https://github.com/ddalcu/mlx-serve/pull/673). Socket-pair tests cover peer closure,
length-framed responses and shutdown; `tests/test_responses_streaming.sh` also waits past the TCP FIN timeout
before reading a completed stream and requires clean EOF.

## Seeds, logprobs, sampling

- **A `seed` binds EVERY sampler with a fresh key PER DRAW** (`sampler.seedKey`).
- Logprobs are the MODEL's distribution (pre-temperature), ids travel WITH values, entry belongs to the RETURNED
  token (one-token delay); `logprobs.content` describes `message.content` (`contentTokenRange`); streaming logprobs
  are a SIBLING of `delta` shipped EXACTLY once against a high-water mark. logprobs>0 + grammar disable spec.
- Logprobs are `logits - logsumexp` in f32 (`sampler.computeLogprobs`): `log(softmax)` in bf16 lands on bf16's grid, 0.125
  apart between -16 and -32.
- Sampling defaults for omitted fields: body > launch flags > model `generation_config.json` > hardcoded.
- Top-k and top-p are ONE pass (`sampler.filterTopKTopP`); a filter cuts by RANK, never by value (bf16 ties at the top
  constantly; `sampler.ranksDescending` ties by lowest id); the nucleus is the mass STRICTLY above each rank, cumsum in f32;
  `top_p` 0 is GREEDY (`sampler.applyTopP` floors at `floatMin(f32)`).
- A sampler never draws a RESERVED special or PADDING row (`sampler.installSuppressMask`; logprobs stay RAW).
- **A non-streaming `/v1/completions` reply carries the chat reply's `timings`** (`prompt_n`, `cached_n`, `prompt_ms`, `predicted_ms`; `prompt_ms` includes the prefix restore); `sushi bench` reads it.
- **A text `/v1/completions` prompt is encoded raw, plus the model's own opening id** (`ModelConfig.textPromptBosId`: DeepSeek's BOS, which its chat template writes literally; its base model answers EOS at once without it). Other families add none, and Qwen's config "BOS" is `<|endoftext|>`.
- **`ignore_eos: true`** (vLLM's field) decodes a `/v1/completions` request past EOS to `max_tokens`
  (`requestEosSlice`); stop sequences and the loop stops still end it, and its text skips `special: true` tokens
  (`completionShowsToken`, vLLM's default `skip_special_tokens`). Chat refuses it with a named 400
  (`chatIgnoreEosRejectReason`): past its end of turn the model writes another turn, and that turn's think block is
  merged into the reasoning by the non-stream reply (`normalizeEmbeddedThinkBlocks`) but not by the live stream.
- **A repeat/presence/frequency penalty samples on the synchronous serial path** (`samplesSync`, like logprobs): no DFlash or
  drafter path (`mtp_plan.draftsRefused`), no batched tick (`.penalty`), never the lazy pipeline, which never applied it.
  MTP does run: each verify row carries the penalty ([engine-mtp](engine-mtp.md#spec-verify-invariant)).
  A penalty-mask allocation failure fails the request instead of sampling without its penalty. Chat and
  completions parse it alike (`parseRepeatPenalty`); a repeat penalty of 0 or below is off. OpenAI's `presence_penalty` and
  `frequency_penalty` (-2 to 2, `parseAdditivePenalty`) are subtracted from the logit once per seen id and once per
  occurrence; `frequency_penalty` is never read as `repeat_penalty` 1 + x, which halved every seen logit at 1.0.
- **Think penalty** (`think_penalty` request > `--think-penalty` > model-settings.json > off, 0-20; arXiv 2606.00206):
  while the thought is open every single-token spelling of the paper's markers (bare or space-led, lower or capital)
  loses λ; a split spelling is skipped (its first piece starts other words). The paper penalises everywhere; we stop
  at the closer. Every sampled position, serial or verify row, is shifted by its own prefix (`thinkShifted`,
  `thinkShiftRows`), so greedy MTP keeps serial's bytes; logprobs stay RAW.


## Experimental logit bias

The per-model `logit_bias_file` setting in `model-settings.json` loads a JSON or CSV file at model load. The default is off. Each entry names exactly one `id`, exact vocabulary `token` string,
or `word`; words expand to single-token lower/capital spellings with and without a leading space. Split spellings
are skipped and counted in the load log.

```json
{"entries":[{"word":"Wait","delta":-1,"scope":"reasoning"},{"id":1234,"delta":2}]}
```

The equivalent CSV columns are `kind,target,delta,scope`:

```csv
kind,target,delta,scope
word,Wait,-1,reasoning
id,1234,2,all
```

`delta` is finite and between -100 and 100: negative penalizes, positive rewards. `scope` is `reasoning`, `answer`,
or `all` (default). Answer scope applies outside reasoning, including before an opener; the closer ends reasoning.
Unknown targets/scopes, out-of-vocabulary ids and malformed files fail model load with a named `LogitBias*` error.
The load line reports the file, entry count, expanded ids and skipped spellings.

`/v1/chat/completions` and `/v1/completions` accept OpenAI `logit_bias`, a map such as `{"1234":-2}`. These deltas
apply to all positions and add to file entries and the optional think-penalty preset. Invalid ids, nonnumeric or
out-of-range biases return 400; `null` and `[]` mean no bias. `think_penalty: 0` disables the preset only; file and request biases still apply.
Overlapping entries add. Sampling uses shifted logits; returned logprobs remain raw.

Scoped vectors are prepared once per request on the inference thread. Active biases use the full target vocabulary
head, including positive rewards. Serial, MTP, prompt-lookup and GLM DFlash2 rows apply the same prefix-dependent
shifts; streaming and non-streaming preserve the same generated tokens. Existing non-stream-only repetition-tail trimming
can still shorten displayed loop-stop replies; `SUSHI_LOOP_TRIM=0` disables that presentation step for strict byte
comparisons while retaining loop detection. The unchanged `--think-penalty` preset remains off by default.

Chat and legacy completions accept `repetition_penalty` as an alias for `repeat_penalty`. A positive
`repeat_penalty` takes precedence when both are supplied; invalid or nonpositive values retain the existing fallback.

## Reasoning budget

Enforced at DECODE (`server.armThinkBound` → `SamplingParams.think_bound`, `scheduler.thinkBoundTick`): at the budget
the early-stop line + the atomic closer commit as ONE multi-token forward (`commitForcedTokens`); the whole closed
thought is delivered. Every decode tick checks it with the loop stop (`loopGuardTick`), the plain batched tick too
(`batchedTickRows`): skipped there, a budget under `--max-concurrent` overran by up to ~2.5k tokens while its slot
batched plain. Guard: `tests/test_reasoning_budget_stream.sh`. Effort budgets = pi's ladder
(the served effort table for served arches, `responses.effortBudget` for the rest).
Every surface arms it with one precedence: explicit budget (`reasoning_budget_tokens`, Anthropic `budget_tokens`) > the
request effort word's cap in the served table (`request_policy.requestThinkBudget`). When the request omits effort,
the default word's cap applies (`--think`'s, else the table's). A word with no cap has none; the `--reasoning-budget`
launch flag is gone ([flags](flags.md#removed)). Built-in GLM and DeepSeek words impose no caps.

<a id="effort-tables"></a>
## Effort tables

- **Storage**: one section per family in `model-settings.json`, `"effort:qwen4_exp"`, `"effort:deepseek_v41"`,
  `"effort:glm5_next"` (a model's key is an absolute path, so the two never meet; `effort.sectionName`). Keys: `default`
  (a word) and each row word, whose object holds only the changed fields: `offered` (bool), `sends` (Qwen
  low|medium|xhigh, GLM low|high|max, DeepSeek an integer 1-100), `cap` (integer ≥ 1, or null = no cap). A row's
  `sends` or `cap` on `off` is refused (off sends nothing and has nothing to cap); `offered: true` on GLM's off is
  refused. The default word must be offered as itself. Under `--think`, a save that would unoffer the word `--think`
  serves on that family is refused: the flag would stop binding and the next start with it would exit.
  A save reads the file, checks, writes every family it names in one rewrite and publishes them, all under the
  settings write lock (`model_settings.lockWrites`): two saves at once never drop each other's rows, and a failed
  write publishes nothing.
- **Validation** (`effort.parseRow`, `checkSends`, `checkCap`, `checkTable`): a save is refused with a 400 naming the
  field (`effort qwen4_exp.high.sends: Qwen takes low, medium or xhigh`) and writes nothing; a start reads the file
  leniently (`model_settings.publishEffortTables`): each bad field is skipped with one `[effort] model-settings.json
  effort:<family>.<word>.<field>: ...; ignored` line, and a default left unoffered drops the file's default first, then
  its `offered: false` on the default row. A hand-edit never breaks a render.
- **Takes effect on the next request, no reload**: a save writes the file (only fields that differ from built-in,
  `effort.normalize`; an emptied section goes) and publishes the family's edits to a spinlocked store
  (`effort.publish`); request handlers on connection threads copy the merged table once per request
  (`effort.servedArms`), so the inference thread and MLX never see it. Hand-edits reach a running server at its
  next start.
- `GET /v1/settings` `effort`: `applies` ("applies at once") and `families` (MiMo has none: only on/off), each with `sends` (`{"kind":"word",
  "choices":[...]}` or `{"kind":"scale","min":1,"max":100,"presets":{"low":50,"high":75,"max":100}}`),
  `off_offerable`, `default` (`builtin`, `file`, `value`, `flag`, `locked`, `source`; `--think <word>` pins the default
  of each family it binds, reported as `flag: "--think <word>"`) and six `rows` with `builtin`, `file` (the stored
  fields, or null) and `value` (the served row), each `{offered, sends, cap, fold_to}`.
- `POST /v1/settings` `{"effort":{"<family>": null | {"default": word|null, "<word>": row|null}}}`: null resets the
  family (or the row, or the default) to built-in; a named row replaces the stored row whole, rows not named keep
  theirs. The answer's `applied.effort` is "applies at once, from the next request" (with a note while `--think` pins
  the default). `mimo_v2` or any other family is a 400 `unknown_setting`.
- Guards: the `effort table` tests (effort.zig, request_policy.zig incl. the characterization test that built-in =
  before the table, model.zig, settings_api.zig, chat.zig), `node tests/test_webui_effort_table.cjs`, live
  `tests/test_effort_table.sh`.

## Constrained JSON

- Constrained generation reports each returned token against its raw logits before the grammar mask, including forced choices; those logprobs have no one-token delay.

- The payload offset is AUTHORITATIVE (`reasoning_protocol.Delivery`, all surfaces, stream + non-stream).
- The grammar mask never walks the whole vocabulary (`token_mask.buildMask`); every grammar state has a legal byte;
  no whitespace OUTSIDE the root value, the model's OWN layout inside (`MAX_FREE_WS` 16).
- A root number ends only at EOS, so it is complete once terminable; the empty-mask disable in `nextConstrainedToken`
  is the logged last resort (a new dead end gets a schema in the `no reachable grammar state is a dead end` test).
- DeepSeek's DSpark verifies each row under that row's mask (`row_grammar.zig`), before and after the reasoning close; logprobs and forced calls still disable spec.
- Every schema-mask surface uses ONE thinking policy (`schemaMasksThinking`); tools present = no mask. Per-model
  grammar table lives on `LoadedModel`.
- Code: `src/json_schema.zig` / `src/json_grammar.zig` / `src/token_mask.zig` / `src/regex.zig` (schema IR →
  streaming grammar → per-token mask), `src/reasoning_protocol.zig`.

- `/props.settings.prefix_cache` = `{ram_enabled, mem_bytes, disk_bytes, disk_used_bytes, disk_entries, volume}`: `disk_bytes` is
  the SSD budget the model was given at load (0 = tier off), `disk_used_bytes` and `disk_entries` what the tier holds,
  published by the inference thread after each commit (the web page's meter reads them; the same numbers log as
  `[disk-cache] usage <used> / <budget> GB, <n> entries` once per finished turn). They are the REQUESTED model's own
  tier (`/props?model=X`), the four numbers from one publish. `GB` is binary (1 GB = 1 GiB), as the flags read it.
  `volume` = `{name, path, free_bytes, total_bytes}` of the cache root, read per request; `name`, `free_bytes` and
  `total_bytes` are null when it is unavailable ([engine-prefix-cache](engine-prefix-cache.md#listing-and-clearing-from-the-admin-api)).
- `/props.settings.prefill_decode_share` reports the effective process-wide share, including zero under the interleave kill switch.

## Security and observability

- `--api-key`: loopback exempt, `/health` + OPTIONS + `GET` of the chat page open, `constTimeEql`.
- A JSON request nests at most `chat.max_json_nesting` (256) levels, checked on the raw body before any route parses it (HTTP and each WebSocket message); an assistant history `tool_calls[].arguments` nested deeper is sent to the template as a string. A schema nests at most `json_schema.max_schema_depth` (64).
- A request reader waits at most `REQUEST_READ_LIMIT_MS` (30 s) between bytes and for its whole head, then drops the connection without a reply, so a silent or dripping client never holds its thread; a `Content-Length` sums into the size check with saturating arithmetic, so a wrapping value is a 413 instead of a small allocation.
- A schema `pattern` repeat count is at most 1000 and its NFA at most 32768 states (`regex.zig`); past either, `compile` fails with `InvalidPattern`.
- Metrics are ON by default (`--no-metrics` is the opt-out; `--metrics` is accepted and does nothing); TTFT at prefill completion; live tok/s via ONE atomic per tick; `/metrics(.json)`. Cost with metrics on: per request a handful of relaxed atomic adds at `finishSlot`; per decode tick one atomic store (`generation_tokens_live`, set by the inference thread, never the sampler, so a finished slot is not in both the total and the in-flight sum; the cleanup drain sets it again after it counts a stop-sequence reply; the tokens of a cancelled or failed request go to `generation_tokens_unfinished_total` (Prometheus `sushi:generation_tokens_unfinished_total`) and stay in the gauge, so it never falls and at rest live equals `generation_tokens_total` plus that counter) and the live-session rows (`publishLiveKvResidency`: one clock read and a ~200 B row copy per decoding slot, under the `queue_mu` hold the cull already takes); per prefill chunk one atomic store and the same row publish; every 2 s the sampler thread (two MLX allocator counter reads, two syscalls, one brief `queue_mu` hold, one ring push). Nothing runs per token. With `--no-metrics` none of it runs.
- **History ring**: the sampler also pushes one sample per 2 s tick (the token gauges only move once per tick) into `metrics_history.process`, a fixed ring of 1800 samples (one hour, 64 B each, about 100 KB static, no allocation). `GET /metrics.json?since=<unix ms>` adds `"history":{interval_ms,capacity,started_ms,fields,samples:[[t_ms,gen_tokens,prefill_tokens,running,waiting,prefilling,batched,gpu_pct,memory_mb,mlx_active,mlx_cache,gpu_mhz],...]}` with the samples newer than `since` (`since=0` is everything kept); `started_ms` is the PROCESS start (stamped at `main` entry, so it stays right after a `POST /metrics/enable` on a `--no-metrics` boot; a host that never stamps it falls back to the first sample's time), which the page reads as the server's start for its uptime; without `since` the feed is unchanged. A client reads the columns by `fields` name, new ones are only ever appended. The dashboard backfills from it when it was in the background and plots by `t_ms`, so a hidden tab leaves no gap and a real gap (server down) is drawn as one.
- <a id="gpu-clock"></a>**GPU clock** (`gpu_clock_live.zig`, no sudo): the gauge sampler reads IOReport's `GPU Stats / GPU Performance States` residency and weights each P-state's frequency by it (OFF excluded, so a fully idle window is 0). The frequencies are the `voltage-states9` pairs of the `pmgr` registry entry (Hz or kHz, normalised to MHz; P14 and up clamp to the top row). `/metrics.json` gauges `gpu_clock_mhz` (the window since the previous sample) and `gpu_clock_max_mhz` (the table's top, constant per boot; 0 when the clock cannot be read), Prometheus `sushi:gpu_clock_mhz` and `sushi:gpu_clock_max_mhz`. `sushi bench` opens its own reader of the same source for the throttle verdict.
- **Finished-request log** (`request_log.zig`): `finishSlot` and the cleanup drain push one row per request into `Metrics.finished`, a fixed ring of 128 rows (about 30 KB static, no allocation) behind one mutex: a few field copies per finished request, nothing per token, and nothing at all on a `--no-metrics` boot (the ring lives in the metrics core, so a runtime enable starts it empty). It rides the same `?since=` as the history: `GET /metrics.json?since=<unix ms>` ends with `"requests":[...]` after `"history"`, the rows finished after `since`, oldest first; without `since` the key is absent. A full array (128 rows) may have dropped older rows. Each row: `{"request_id":42,"model":"<id>","t_ms":1762100000123,"prompt_tokens":70,"cached_tokens":30,"prefill_seconds":0.25,"generated_tokens":7,"decode_seconds":1.5,"outcome":"ok"}`.
  - `request_id` is the `sessions` row id. `t_ms` is the server wall clock of `history` samples and strictly increases across rows (a tie is bumped by 1 ms), so polling with `since=<newest t_ms>` neither skips nor repeats a request.
  - `prompt_tokens` are the tokens the model computed (prompt minus restored); `cached_tokens` are the ones restored from the prefix cache. The prompt is their sum. A request that ended inside its prefill (cancelled, or an error) logs the tokens its forward had computed so far (`Slot.prefill_computed`, per chunk) and `cached_tokens` 0: its restore was not kept and counts as no tier hit.
  - `prefill_seconds` is `Slot.prefill_ns`: the lookup and restore of the prefix cache (the SSD tier included) plus the forward over the uncached tail, minus decode ticks interleaved into it and without the queue wait; 0 when the request ended before its prefill finished. A restore-heavy row therefore reads a lower prompt speed than the forward ran at. `decode_seconds` is the slot's decode wall time (a batched tick counts whole for every slot in it), so `generated_tokens / decode_seconds` is the speed the client saw.
  - `outcome`: `ok`, `cancelled` (disconnect or cancelled in prefill), `error` (a generation or latched MLX failure), or `rejected` (refused before its first forward: `PrefillDoesNotFit`, `MediaRefused`; zero tokens). A refusal before a slot exists (context overflow, memory preflight) has no row; `sushi:request_rejected_total` counts it. Rows are recorded where the counter outcome is (`recordSlotEnd`/`recordSlotCleanup`), so each request appears once.
- **Metrics at runtime**: `POST /metrics/enable` turns the same core on in a server booted with `--no-metrics` (200
  `{"metrics":true,"already_enabled":false}`, and `true` on every later call: idempotent; any other method is a 405). There is
  no off switch. It sits behind the `--api-key` gate like the other mutating routes and refuses a foreign `Origin` (403, same
  rule as `/v1/load-model`); `curl -X POST http://127.0.0.1:12345/metrics/enable` needs no Origin. The state is readable from
  the feeds themselves: `GET /metrics` and `/metrics.json` answer 503 exactly while metrics are off, and the 503 body names
  this endpoint (no `/props` field). Counters, histograms and the session rows count from the moment of the enable; a
  request already in flight is counted when it ends.
- Thread safety of the runtime enable: `metrics.process` (a `Switch`) holds the core in static storage behind one atomic
  state (off, building, on). The first `enable` builds the core and publishes `on` with release; a racing caller waits for
  `on`; `get()` is one acquire load that returns null while off, so every reader keeps the no-op path until then. Connection
  threads read it through `metrics.current()`. The inference thread owns `Scheduler.metrics` and adopts the published core
  right after it wakes for work (`adoptPublished`), never mid-tick. The gauge sampler thread starts once, from `--metrics` at
  boot or from the first enable (`startGaugeSampler`, a state machine joined by `stopGaugeSampler` before the scheduler is
  torn down). The core calls no MLX; the sampler's `mlx_get_*_memory` reads are the existing `--metrics` behaviour.
- **A request outcome is counted exactly once** (`Slot.metrics_recorded`, inference thread): `finishSlot` or the cleanup drain,
  whichever sees the slot first (`recordSlotEnd`/`recordSlotCleanup`). The outcome comes from the slot's finish state, never
  from `Slot.cancelled` (`complete` sets it on every completion): success feeds the histograms; `sushi:request_cancelled_total`
  (a disconnect mid-decode; a request ended by its own stop sequence sets `Slot.stop_hit` first and counts as success), `sushi:request_failed_total` and a refusal before a slot
  (`sushi:request_rejected_total`: context overflow, memory preflight, `PrefillDoesNotFit`) move only their counter.
- **SSD prefix-cache writes** ([engine-prefix-cache](engine-prefix-cache.md#write-counters)): `/metrics` carries
  `sushi:prefix_cache_write_bytes_total{kind="kv|ssm|qsa|spec|ring|meta"}` and `sushi:prefix_cache_write_failed_bytes_total`;
  `/metrics.json` carries `"prefix_cache_writes":{"written_bytes":{<kind>:n},"failed_bytes":n}` after `decode_serial`.
  Unlike the request counters they run from process start whatever turns metrics on, and the routes keep the 503-while-off
  rule. `/props settings.prefix_cache.disk_written_bytes` is that model's own total since its tier opened (`disk_stats`,
  the same sequence-locked publish as `disk_used_bytes`), so an unload and reload restarts it while the process total keeps it.
- <a id="speculative-and-tier-counters"></a>**Speculative and tier counters** (`spec_stats.zig`, `cache_stats.zig`; like the write counters they run from process
  start, and the routes keep the 503-while-off rule). `/metrics.json` carries, after `prefix_cache_writes`:
  - `"prefix_cache_tiers":{"hits":{"hot":n,"ssd":n},"evictions":{...},"evicted_bytes":{...},"hot_bytes":n}`: requests
    whose restored prefix came from memory or the SSD tier, entries each tier dropped to stay inside its budget (and their
    bytes), and the RAM tier's resident bytes. `/metrics`: `sushi:prefix_cache_tier_hits_total{tier}`,
    `sushi:prefix_cache_evictions_total{tier}`, `sushi:prefix_cache_evicted_bytes_total{tier}`, `sushi:prefix_cache_hot_bytes`
    ([engine-prefix-cache](engine-prefix-cache.md#tier-counters)).
  - `"speculative":{"mtp":{...},"dflash":{...},"dspark":{...}}`, one object per drafting mode (MTP: Qwen and MiMo heads;
    DFlash2: GLM; DSpark: DeepSeek) with `requests` (that ran a round), `rounds` (one verify forward each), `drafted`,
    `accepted` (drafted tokens the verifier kept, not the always-committed first token of a round) and `plain_rounds`
    (DSpark: rounds that carried one verifier row, already inside `rounds`; MTP: planner ticks that ran without a draft,
    outside `rounds`; DFlash2: the same, GLM's ticks that ran without a draft, outside `rounds`). DSpark adds `rows`, `{<verifier rows>:
    rounds}`. They are the sums of the figures `[spec-stats]` logs, added once when a request ends
    ([engine-mtp](engine-mtp.md)), also for a request that errors mid-decode (`finishSpecAtCleanup`, once: `Slot.spec_recorded`). Acceptance % is `accepted / drafted`, mean accepted per round `accepted / rounds`.
    MTP's figures include its prompt-lookup rounds (head + lookup), as DFlash2's do; the `[spec-stats]` `per_draft_pct`
    stays head-only. `/metrics`: `sushi:spec_requests_total{mode}`, `sushi:spec_rounds_total{mode}`, `sushi:spec_drafted_tokens_total{mode}`,
    `sushi:spec_accepted_tokens_total{mode}`, `sushi:spec_plain_rounds_total{mode}`, `sushi:spec_rows_total{mode,rows}`.
  - A `requests` row of a request that speculated carries `"spec":{"mode","rounds","drafted","accepted"}`; a plain request
    has no `spec` member.
- `/metrics.json` ends with `"sessions"`, one row per live request (phases `prefill` and `decode`, cap 32; published
  by the inference thread under `queue_mu`, copied by the reader under the same lock — there is no separate
  `/requests` route): `model`, `request_id` (submit sequence; stable across polls of one request, never reused),
  `phase`, `context_tokens` (prompt + decoded; the FULL prompt on a prefill row), `context_length` (the model's
  effective limit, 0 = not ready/unknown), `cached_tokens`, `generated_tokens`, `max_tokens` (the request's own
  output cap), `elapsed_seconds` (age since arrival, refreshed every publish) and `state_bytes` (GPU bytes of the
  slot's own KV + SSM buffers; a restored share stays billed to the hot-cache entry, never to the row; a DeepSeek
  slot's is the model-held state, rows and rings, plus the live restore points).
- A live row's `state_bytes` is at capacity: a ringed layer's ring, each QSA bank by its capacity buffer (never also
  its view), and the ring restore points the slot holds.
- After the live rows come `cached` rows, one per hot-cache entry of every ready model (cap 32, published under
  `digest_mu`): `context_tokens` = `cached_tokens` = the entry's tokens, `state_bytes` its `kv_bytes`. A cached row has
  no request, so `request_id`, `max_tokens`, `generated_tokens` and `elapsed_seconds` are 0.
- An entry a live row restored from is listed once, as that row: the dedupe keys on an internal entry id that
  `/metrics.json` does not emit.
- `/props` `template_fallbacks` counts renders where a `<|im_end|>` model's own chat template raised and ChatML
  answered (other models refuse with 500 `chat template unusable`, uncounted) ([server-tool-calling](server-tool-calling.md#templates)); the count is process-wide, and it is zero when
  every prompt is the checkpoint's own.
- `/props` `memory.total_bytes` = the machine's physical RAM (`hw.memsize`), the ceiling the Compact view's Memory tile and chart measure against.
- `/props` `memory.kv_cache_bytes` = the current model's hot-cache residency + every live slot's state, published
  each tick with or without `--metrics`; a donated checkout's buffers, which the entry bills until release, count once.

## SSD prefix cache and settings (`/v1/prefix-cache`, `/v1/settings`)

All four routes sit behind `--api-key` like the other admin routes, and a request carrying a foreign `Origin` is a 403
(`crossOriginRefused`). Code: `src/prefix_cache_admin.zig`, `src/settings_api.zig`.

- `GET /v1/prefix-cache`: `{root, volume:{name,path,free_bytes,total_bytes}|null, total_bytes, namespaces:[{id, bytes, entries,
  newest_mtime (unix s), model|null, stale, held_by_other_process, tier:{used_bytes,cap_bytes,entries,written_bytes}|null}]}`.
  A namespace is a `<16 hex>` directory of the cache root (`SUSHI_PREFIX_CACHE_DIR`, else `~/.sushi/kv-cache`) holding
  `e<id>` entry directories (a private `p<pid>-<n>` root inside it counts too). `bytes` is the apparent size (the sum of the
  files' lengths, not disk blocks) with a hard-linked file counted once; `entries` counts entry directories with a `meta.json`. `model` is the loaded model whose
  tier lives there (and `tier` its live numbers, published by the inference thread); `stale` means no loaded model uses it
  and no other server holds its lock, or the lock of a private `p<pid>-<n>` root inside it (`held_by_other_process`). A symlink or a folder that is not a namespace name is never listed.
  `tier.written_bytes` is cumulative since the tier opened; a client takes the write rate between two polls.
- `POST /v1/prefix-cache/clear` `{"scope":"model","model":id}` | `{"scope":"namespace","id":"<16 hex>"}` | `{"scope":"stale"}` | `{"scope":"all"}` answers `{scope,
  freed_bytes, model_freed_bytes, model_entries_removed, stale_freed_bytes, stale_namespaces_removed,
  skipped_live_namespaces, refused_symlinks, busy_models}`. `model` empties that model's tier on the inference thread (its
  sole owner): staged writes are fenced, the index, `total_bytes` and the files go together, and the tier takes new writes
  at once. It is refused with 409 `model_busy` while a request holds the model (retry; it is not queued), and 409
  `model_not_loaded` for a model without a tier. `stale` deletes the namespace directories no loaded model uses, each under
  the root's lock (a root another server holds is skipped), and never follows a symlink out of the cache root. `all` is both,
  with busy models listed and left alone. A `model` scope needs a string `model` (400 without one: the empty id would
  mean the default model). `namespace` deletes exactly that namespace directory under the same locks as `stale`, so a
  namespace another server holds is skipped (`skipped_live_namespaces: 1`, nothing removed) and a symlink is refused;
  it answers the `stale_*` fields of the namespace it removed, 400 for an id that is not a namespace name, 404 for
  one not in the cache and 409 `namespace_in_use` for one a loaded model owns (use `model`).
- `GET /v1/settings` reports `~/.sushi/model-settings.json` as the page edits it: `server.models_dir`,
  `server.update_check` and `server.pack_update_check` (booleans, `default: true`, `file`, `flag`, `source`, `locked`;
  `--no-update-check` pins both and reports `flag: "--no-update-check"`; a change reaches the check loops at once, no
  restart), `server.skip_mem_preflight` (the same fields, `default: false`; `--skip-mem-preflight` pins it and reports
  `flag: "--skip-mem-preflight"`; the next load reads it, no restart), `server.port` (an integer 1-65535: `file`, `flag` as typed when `--port` pins it, `default: 12345`, `source`,
  `locked`, `applies` "the next start; a running server keeps its port"; the port a server binds without `--port`, and what
  sushibar and `sushi launch` target) and
  `server.prefix_cache_disk_gb` (`file`, `flag` when a launch flag pins it, as typed: `models_dir` whole, the size in the
  flag's own syntax such as `512MB`, `source` `flag|file|default`, `locked`; the size also carries `effective_bytes`, the
  largest running tier's cap or, with none running, the ask; `ask_bytes`, the number the next load starts from; `default_bytes`, the size an unset key asks for (a load may
  take less to fit its context or the free space); and
  `running_tiers`, `[{model, cap_bytes, used_bytes}]`), the
  `model_keys` metadata (type, range, choices) and, per registered model, each key's `file` value, `flag`, `source` and
  `locked`. A pinned `mtp_acceptance` reports the flag as typed with its value (`--mtp-tokenv3 0.5`, `--mtp-typical 0.3`) and `mtp_greedy_tail` reports `--mtp-greedy-tail`; the other model keys report their value alone; and `effort`, the thinking-effort tables ([Effort tables](#effort-tables)). `POST /v1/settings` takes `{"server":{key:value|null}}`, `{"model":id,"set":{key:value|null}}` and/or `{"effort":{...}}`; null removes
  a key. Each model row also carries `keys`, the keys that apply to that model (`model_settings.applies`, the same gates
  the loader's ignored-key log lines use: `ssd_budget_gb` needs expert streaming, `ssd_prefetch` is DeepSeek-V4.1's,
  `kv_quant`, `mtp_acceptance` and `mtp_greedy_tail` are not DeepSeek-V4.1's, `vision` needs a tower), and `loaded_with`,
  null while the model is not loaded and otherwise `{key: value|null}` for every key: the file value the load stamped
  (null: the file had none then), in the shape of `settings.<key>.file`, so a page tells "saved differs from loaded".
  A ready model's row also carries `defaults`, `{key: value}` for the keys whose unset value the server can state (null while the model
  is not loaded): `settings_api.resolvedDefaults` reads it from the loader's own resolvers on the config with the file's values and the
  launch flags cleared, so it is never a flag's value. `ctx_size` and `ssd_budget_gb` are absent (measured at load); `max_tokens` is
  0 when the model has no output budget of its own and an omitted `max_tokens` takes the rest of the window. A page writes "Default (value)"
  from it and a bare "Default" when a key is absent.
  Each row also carries `limits`, an object holding an entry only for a key the model constrains (`model_settings.limit`, read
  from the same `Fit` the loader's gates use): `disabled: true` (the loader ignores the key; every key missing from `keys` has
  one), `choices` (the subset a choice key takes: GLM's `kv_quant` is `off|6|8`, the widths its MLA latent stores), `max`
  (`ctx_size`: the model's context window, YaRN-aware), `requires: {"mtp": true}` (`mtp_acceptance`, `mtp_greedy_tail`: read
  only while `mtp` is not false, unset counting as on) and `unset: ["ssd_budget_gb"]` (a key the pack ignores while an SSD
  budget makes it stream, because its MTP head cannot run then), each with a one-sentence `why`. A POST for a model refuses
  with a 400 `invalid_setting` naming the key and the reason a non-null value of a disabled key, a choice or integer outside
  its limit, and a set key whose `requires`/`unset` fails against the file plus the request; removing a key and any value
  already in the file pass, and the loader still never refuses a load over the file. Not in `limits`: bounds that need the
  machine's memory or the pack's weight bill (the smallest `ssd_budget_gb`, a tower beside a small budget, the memory-safe
  `ctx_size`); the load refuses those by name.
  `kv_quant` is a choice name (`off`, `4`, `6`, `8`) wherever it is read or written; a request may say `0`, `16`, `"16"` or `"off"`
  and the file stores `"off"`. Unknown keys are a 400 `unknown_setting`, a bad value a 400 `invalid_setting`, a `models_dir` that is not a readable
  folder a 400, an unknown model a 404, a settings file that is not a JSON object a 409; every key is checked before anything
  is written, and the write is a temp file plus rename (through a symlinked settings file, to its target; a model section is found
  without regard to a trailing slash in its key). A key a launch flag pins is still written (it applies at a start
  without the flag) and reported with the flag's source. Per-model keys are stamped at load, so they apply at the model's next
  load (the page offers the reload). `prefix_cache_disk_gb` reaches the running tiers at once (the tier re-derives its budget
  and evicts down to it) and every later load, cold loads included; the answer says "applied to the running tiers" only
  when a tier was resized, else "applies at the next model load"; 0 and removal apply at the next load. `models_dir`
  triggers a rescan that adds the folder's models, also on a `--model`-only server, and says how many (add-only; the boot
  discovery follows it at the next start).
- The page shows both in its **Models** (`/models`: a list with a detail form per model: size, load/unload, settings, a
  model's cache) and **Settings** (`/settings`: SSD cache size and use, per-namespace rows with their own Clear, the clear
  buttons, models directory, version) views. The page is served by the same build as the API, so it reads these fields
  without fallbacks: per model `keys` (which settings the model has), `loaded_with` (what the loaded model started with,
  for the reload prompt) and `limits` (per key: `disabled`, `choices`, `max`, `requires`, `unset`, each with a `why`
  the page shows on the row; a key outside `keys` with a `disabled` limit shows greyed in Advanced), `kv_quant` as
  `off|4|6|8`, `server.*.flag` as the flag's own text, `server.prefix_cache_disk_gb.ask_bytes`, and `POST
  /v1/prefix-cache/clear {"scope":"namespace","id":...}`. A `flag` is printed as the server sends it: a text starting with `--`
  (`--mtp-tokenv3 0.5`) is the flag's whole spelling, anything else is the value the page puts after the flag's name; a locked
  choice shows the choice the spelling names, else "Set at launch". The Settings page's "Automatic update checks" card is a Default (On) / On / Off group per check.
  Each server setting's badge is `Now`, `Next load` or `Next start`, read from its `applies` sentence; a setting row has no
  Default badge. A Settings box shows Save only while it differs from the file, and an emptied box says "Use default" and sends
  `null`. A refusal shows on the setting's row (labels, never keys), not as a toast; while the server cannot be reached the dimmed
  Models and Settings areas are `inert`. A load toast names the models the server unloaded to fit it. The Settings write rate is the difference of two `tier.written_bytes` polls per namespace (no
  `/metrics.json` read, which is 503 while metrics are off). Guards: `tests/test_prefix_cache_admin.sh`,
  `node tests/test_webui_admin.cjs` (pure functions), `node tests/test_webui_admin_ui.cjs` (the tabs against a stub).

## Downloads (`/v1/downloads`)

Behind `--api-key` like the other admin routes, a foreign `Origin` is a 403 (`crossOriginRefused`), and every method is
answered before any model resolves, so a headless server with no model serves them. `POST /v1/downloads` from another
machine is a 403 unless the server has `--api-key` (`download_progress.startRefused`). Code: `src/downloads.zig` (routes,
catalog, thread) and `src/download_progress.zig` (host tier: names, space, byte accounting, speed).

- `GET /v1/downloads[?refresh=1|?background=1]`: `{ram_bytes, ram_gb, update_check, dest:{path, source:
  flag|file|default, locked, volume, free_bytes, total_bytes}, catalog: loading|ready|offline, refreshing,
  packs:[{repo, name, tag, size_bytes|null, tier_gb|null, fits|null, streams_experts, recommended, installed,
  path|null, partial_bytes|null, update:{current,latest}|null, launch|null}], jobs:[{repo, dest, phase, stopping, files,
  file_index, total_bytes, done_bytes, file:{path,size,done}|null, speed_bps|null, eta_s|null, message|null}]}`.
- The catalog is the `cli.aliases` table, then the org's public repos named `-Sushi-` from `/api/models?author=beamster`
  whose name is plain `[A-Za-z0-9._-]` (a pack published after the build: no tier, so `fits` null). Sizes are `sushi
  list`'s (`?blobs=true`); a repo the API answers 401/403/404 for is left out; with no answer at all the known packs stand
  with null sizes and `catalog` `offline`. It is built on a background thread, at most every 10 minutes (1 minute while
  offline, at once with `refresh=1`), so the answer never waits on Hugging Face; a finished download clears its own
  row's `update` without asking again. `fits`, `tier_gb` and `recommended` are `sushi list`'s rules (`verdictFor`,
  `recommendedAlias`) on `hw.memsize`; `streams_experts` marks a pack whose tier is for streamed experts (DeepSeek).
  `update` is `pack_update.checkPack` for a downloaded pack. With the automatic checks off (`--no-update-check` or
  `server.pack_update_check: false`; `update_check` false) a `background=1` GET (the page on another tab) starts no
  catalog refresh, the open list fetches the catalog only on its first look and on `refresh=1` (Check), and only
  `refresh=1` looks up an installed pack's update.
- `installed` looks in the destination and the boot roots (`catalogPackPath`, both layouts); `partial_bytes` is a
  destination folder that is there but not complete, null while that pack downloads. `launch` is `sushi serve --model
  '<path>'` plus the flags of the pack's recommended launch in [usage](usage.md#recommended-launch)
  (`Alias.serve_flags`), only for a downloaded pack in the table; the page does not show it.
- `dest` is the Settings tab's models folder: the first `--model-dir` (locked), else `server.models_dir`, else
  `~/.sushi/models`; a new pack lands in `<dest>/beamster/<repo>`, an update in the folder it is installed in.
- `POST /v1/downloads {"repo":"beamster/<repo>"}` takes only a pack the catalog lists, under the catalog's spelling
  (404 otherwise, 400 for another org or a name outside `[A-Za-z0-9._-]`, `.` or `..`), and answers 202 `{ok, queued,
  repo, dest, warning}` (`queued` false when that pack is already queued or running). An update of a loaded pack is a
  409 `model_loaded` naming the model to unload; the other way round, a load of a pack folder a queued or running
  download or update fills (`POST /v1/load-model`, or a request naming the model) is a 409 `pack_downloading` naming the
  pack, since the pull replaces its files one by one; a folder a stopped or failed pull left (its `.pull-incomplete`
  marker) is a 409 `pack_downloading` too, saying to resume it on Downloads. Both checks run under the registry's mutex (`load_gate`), so
  neither slips past the other. A new pack whose size, less what its folder holds, would leave less
  than a twentieth of the volume (at least 4 GiB, the SSD cache's reserve) is a 507 `insufficient_storage`; under a
  tenth free it carries a `warning`. The download checks again before it writes, against the listing: the listed bytes
  less the files already at their size and the partials of the others (`hf_pull.bytesPresent`).
- One download runs at a time on a thread `downloads.zig` owns (never the inference thread or a connection thread),
  through `cli.pullRepo`: same manifest, content-id skip, `curl -C -` partials and verify-before-rename as `sushi pull`.
  `phase` is `queued|listing|checking|downloading|verifying|done|failed|cancelled`; `done_bytes` counts finished files and
  the partial in flight; `speed_bps` is the average over the last 5 s of the samples each GET takes (a resumed partial is
  not speed); `message` is the pull's error line (a checksum mismatch names the file). A finished pack rescans the registry and is listed by its own folder when no root
  the server walks holds it (a `--model`-only server lists the pack it downloaded, and no other folder).
- `POST /v1/downloads/cancel {"repo":...}` drops a queued pack or stops a running one: `stopping` is true until it
  reports. The curl in flight gets SIGTERM (its pid is held unreaped until the job clears it, so the signal never reaches
  another process), the hash of the files on disk stops between files, and a file already being verified finishes its
  verify. The partials and `.pull-incomplete` stay, so a later start resumes, after a restart too (the queue is in
  memory; the folder is the state). Server shutdown stops the curl the same way and refuses new starts (503).
- The page's **Models** tab draws it as its **Available** list (an Installed | Available switch in the tab's head;
  `/downloads` is the list's own URL): per pack the size, Recommended for this Mac, Needs N GB (with
  `sushi list`'s words: "may stream (--ssd-budget-gb)", or the streamed tier for DeepSeek; a click asks first), Downloaded
  with Open (the pack selected in Installed, where it loads), Update, Resume for a partial folder, and for a running
  download a bar, bytes, speed, time left and the file in flight; Stop shows "Stopping…" until the job reports. It polls
  every second while a download runs or the catalog loads, every 5 s otherwise, and writes no node on an unchanged
  answer; on Installed a running download polls every 2 s and shows its percent on the switch. From another tab it asks
  (`background=1`) 3 s after the page opens and then every 30 minutes (every 5 s while the catalog loads); a pack (or a
  pack's update) missing from `localStorage` `sushi-seen-packs` puts a dot on the Models tab (its name reads "Models,
  new packs") and on Available until the list is shown. Only a ready catalog seeds or marks the list, the first one
  silently; with the automatic checks off there is no dot and no background ask after the first. Removing a pack is not
  offered. Guards: `tests/test_downloads.sh`, `node
  tests/test_webui_downloads.cjs`, the `downloads:` tests (download_progress.zig, downloads.zig, http_conn.zig).

## Chat page (`GET /`, `GET /chat`, `GET /dashboard`, `GET /models`, `GET /settings`, `GET /downloads`)

- Design rules (tokens, salmon accent, motion, logo, tooltips): [server-webui](server-webui.md).
- One self-contained file, `src/webui/index.html` (CSS, JS and the logo inline, no external fetch), embedded with
  `@embedFile` and served as `text/html; charset=utf-8`. Any other method on those three paths is a 405 answered BEFORE
  model resolution, so it can never cold-load a model. `GET /dashboard` serves the same page and is open under
  `--api-key` like the other two. Guards: `tests/test_webui.sh`, the `chat page:` and `dashboard page:` tests.
- The root `/` lands on the Dashboard (`viewOf` maps it there), `/chat` is the chat. A **Dashboard | Chat | Models | Settings** navigator in the top bar, each tab a small icon beside its label (the icon
  alone on a phone, 640 px and under), switches the views of that one page in place (`history.pushState`,
  `popstate` for Back/Forward). The chat panel is hidden with `inert` and `visibility`, never detached, so the open
  conversation, a streaming answer, the composer, the scroll position and the API key overlay survive a switch.
  Cmd+1..4 (Ctrl+1..4 off a Mac) click the four tabs in that order, so unsaved Models/Settings edits still ask; they do
  nothing behind the API key overlay or an open dialog. A browser that reserves those keys for its own tabs keeps them.
- **One reader of `/v1/models`**: `modelStore` (main script) polls it (30 s on Chat, 5 s on Dashboard and Models, 4 s
  while the server is unreachable, 2 s while a load is pending) and notifies subscribers (top-bar status, chat model list, dashboard strip, Models tab) only
  when the answer changed; no timer runs while the tab is hidden, and loads and unloads go through its `act`. There is
  no separate `/health` poll. `confirmAction` is the one confirmation dialog (destructive buttons name the bytes they free).
- The dashboard (ported from @cosmic-quokka's contribution, #55) polls this server's own `/metrics.json`,
  `/v1/models` (through `modelStore`) and `/props` with the chat's stored API key as `Authorization: Bearer`, at 2 s or slower. The live
  charts need metrics: without them the page says so, shows a **Turn on live statistics** button (only while `/metrics.json` answers
  503; it `POST`s `/metrics/enable` with the chat's key, then polls at once, so the charts start without a reload and the
  button goes away; a refusal leaves the button and says why), and the page says metrics are off instead of drawing empty charts. While hidden it polls only
  `/metrics.json`, and only when no chat request is in flight and the browser tab is visible. Switches and tab
  returns keep the 2 s minimum between polls; a transient error backs off (2 s doubling, up to 30 s), while a 503 or
  401 stops the hidden poll until the dashboard is shown again. It follows the chat's
  theme. The GPU chart (Compact and Advanced) draws the utilization and the clock as a share of `gpu_clock_max_mhz` on one 0 to 100% axis (idle windows are gaps in the clock; its tooltip gives MHz) from the history's `gpu_pct` and `gpu_mhz`; with `gpu_clock_max_mhz` 0 it draws utilization alone, drops the clock from its legend and says the clock is unavailable. Guard: `node tests/test_webui_dashboard.cjs` (headless Chromium through Playwright; skips if absent).
- **Compact and Advanced**: a header toggle (remembered in `localStorage` `sushi-ui-mode`, Compact the default) switches
  the page's detail level through `body[data-mode]`. Compact shows what is loaded, decode and prefill tok/s, memory used and
  available, the SSD cache's use and the request count as big tiles; Advanced adds the sessions table and the tile strip
  (models live on the Models tab). The charts are one grid for both views (next bullet). Elements only
  Advanced shows carry the `adv` class, Compact-only ones `cmp`; the Models and Settings views hide their expert rows
  and columns the same way, and a model's settings form lists every key it has in Advanced but the common five in
  Compact. The Advanced switch builds no Advanced DOM while Compact is shown.
- **The dashboard's charts come from the server's history**, not from the page's own polls: each poll asks
  `/metrics.json?since=<last sample time>`, appends the new samples with decode and prefill rates taken between neighbouring
  samples (`historyRates`), and plots at the server's `t_ms`. A tab in the background does not poll, and on return the next
  poll fills the whole interval in; two samples more than 7 s apart (the server was down, or the ring was cut) are drawn as a
  gap, never joined. A wider window than the one shown asks for everything again (`since=0`), also while paused. A change
  made during a poll in flight is polled for as soon as that poll ends. `history.started_ms` is the process start: a
  different value is a restart, which draws a marker at that server time, drops the old server's request dots and last-request
  tiles and shows "Server restarted at ..." for a minute. Uptime reads from it. All times on the charts are the server's clock.
- **Per-request log**: the same call returns `requests`, one entry per finished request after `since` (`request_id`, `t_ms`,
  `prompt_tokens` computed, `cached_tokens` restored, `prefill_seconds`, `generated_tokens`, `decode_seconds`, `outcome`).
  A request is sent again until a history sample passes it, so the page keeps the ones it has seen by `request_id@t_ms`.
- The **Compact view** shows a status strip (model, Idle / Reading prompt / Generating / Waiting (requests only queued), or Ready while metrics are off, uptime from the process start),
  five tiles and three charts. Prefill and decode are separate charts with their own scales. A request's prefill speed is its
  computed prompt tokens over its `prefill_seconds`, one dot per finished `ok` request at its own `t_ms`; the tile shows the
  latest, or the mean since start until one is seen. A request that restored more than it computed is a hollow dot labelled
  "N computed, M restored" (the time includes the restore, so it is no prefill speed) and never feeds the tile; a partly
  restored one reads as a lower bound. Decode is the history's rate with idle samples drawn as gaps; the tile reads live while
  generating, else the last request's `generated_tokens / decode_seconds` (any length). Offline, the tiles read "-" with
  "server unreachable" and the charts dim. The SSD tile reads "-" while no model is loaded. A gridline's label prints its
  value exactly (1.875k, 0.75), and a duration rounds before it picks its units (2m 0s, never 1m 60s). The charts answer
  a tap (the nearest point; a tap elsewhere hides it) and the arrow keys (the canvases take focus; Escape hides).
  The Decode and Prefill tiles of both views are the same function (`speedTiles`).
- **Dashboard charts** (`#dashCharts`, one `.chart` card per `data-chart` id, every card half the row from 900 px and one per
  row below), fourteen of them: canvas time series (prefill, decode, requests stacked as generating, reading the prompt and
  waiting with batched as a line over them, GPU, memory stacked as MLX active, MLX cache and other under the machine
  ceiling, draft acceptance) and cards of bars, columns and histograms built from the `/metrics.json` counters,
  histograms, `decode_serial`, `speculative`, `prefix_cache_tiers`, `sessions` and the request log: prefix cache reuse
  (the share of prompt tokens restored, cached vs computed tokens per request, requests that reused, hits by tier),
  storage (hot and SSD use, evictions per tier), writes by kind, request outcomes, response time (the average request as
  one bar of waiting, prefill and decode, then TTFT and end-to-end per bucket), prompt and output length side by side per
  bucket, serial-decode reasons and DSpark rows per round. The draft acceptance card is one dot per request whose log row
  carries `spec`, with a row per drafting mode under it (acceptance and mean accepted per round, DSpark plain-round
  share); the dots alone say when nothing drafted or the server does not report it. Each bar row prints its exact value and has a tip (`data-tip` text). A card's bar
  rows (and the cached-vs-computed columns) are one Tab stop the arrow keys, Home and End walk; a poll that changes them
  updates the same nodes when the content keeps its shape, else hands the focus to the same place (`setHtml`, `rove`). A histogram takes the focus and the arrow keys step its buckets.
  Each card is a `role="group"` named by its picker title. A card under 440 px wide (container query) puts each bar
  under its label and value and labels every other bucket. A member the server does not export
  (`speculative`, a histogram, `prefix_cache_writes`) reads "Not reported by this server", never as zero; with no model
  loaded the hot tier reads "-", not "off"; a rejected request is no column of the cached-vs-computed chart; an empty
  histogram says so. The open picker's ticks follow the view when Compact/Advanced changes the default. A count
  chart (Requests) puts gridlines on whole numbers only: a step of 1, 2, 5, 10, 20 ... with at most four. A card that is
  hidden is not filled. A **Charts** button opens the picker, an inline panel above the page body that pushes the page
  down (config mode): tick charts on or off, Reset, Done. While it is open each chart shows a strip: drag it onto another
  chart (pointer events, so a finger works too; `touch-action: none` on the strip) or use its arrow buttons, which keep
  the focus ([server-webui](server-webui.md#dashboard-config-mode)). The choice is `localStorage` `sushi-dash:charts` (`{order, hidden}`), a browser preference and no server
  setting; until one exists Compact shows prefill, decode, GPU and memory and Advanced shows all but DSpark rows per round.
  A stored choice that names a card since merged (`memsplit`, `specmodes`, `cachetok`, `ttft`, `e2e`, `ptok`, `otok`)
  reads as the card that replaced it, at the first place of any of its parts, hidden only when all of them were; the
  migrated choice is stored back (`migrateLayout`). The hot tier's bytes are
  the server's `prefix_cache_tiers.hot_bytes`. The prefix cache writes chart shows the `prefix_cache_writes` totals:
  bytes per kind, the rate between the last two polls (absent on the first poll and across a restart), bytes written per
  byte now on disk (the sum of every ready model's `/props` `disk_used_bytes`, refreshed every 15 s) and the failed or
  dropped bytes. A server without the counters shows "Not reported by this server". Logic: `prefixWrites`,
  `renderCards`, `acceptChart` and the layout block in the dashboard script, guarded by `tests/test_webui_prefix_cache.cjs`
  and `tests/test_webui_dashboard.cjs` (`chartPicker`, `chartExtras`).
- The right sidebar's "Prompt cache (SSD)" section is a bar meter of `/props` `settings.prefix_cache`
  (`disk_used_bytes / disk_bytes`, "<used> / <budget> GB" and the entry count beside it). It is read after each reply
  and again 2.5 s later (the server stores a turn's entry after answering) and stays hidden while `disk_bytes` is 0.
- It speaks only the public API: `/v1/models` for the picker (`vision` or an `image` input modality shows the attach
  button), `/v1/chat/completions` streamed with `include_usage` (the
  readout is the final chunk's `usage` + `timings`), `reasoning_content` shown collapsed. Stop aborts the fetch; the
  server cancels on disconnect.
- `POST /v1/slots/cancel {"id": <request_id>}` stops an in-flight slot (pending, prefilling or decoding) by its request id:
  the slot is marked cancelled, its prefill or decode loop breaks at the next chunk/token boundary, and its allocated memory
  is returned to the pool. Answers `{"ok":true}` (200), `400` on invalid payload, or `404` when no in-flight slot matches the id.
  The dashboard's **Sessions** table provides a **Cancel** button on active prefill and decode rows, invoking this endpoint.
- The effort menu lists exactly the row's `reasoning_efforts`, with no default entry; it shows the stored choice where
  the model takes it, else the row's `default_reasoning_effort`, and every request sends the shown word. A saved
  effort table reaches it through `/v1/models` (the Settings card refreshes the model list after a save).
- Settings › Inference › **Thinking effort** edits the effort tables ([Effort tables](#effort-tables)); each model's
  detail on the Models page carries one read-only line from its `/v1/models` row (`Thinking: medium by default · low
  2048 / medium 8192 tokens · Edit`, Edit opening that family's tab; a model without an effort table shows no line). Logic:
  `effortPreview`, `effortRowProblem`, `effortPostBody`, `effortSummary`; guard `tests/test_webui_effort_table.cjs`.
- The thinking-budget box beside it takes the word's `reasoning_effort_budgets` value whenever the model or the word
  changes (empty for an uncapped word, disabled at `off`); a typed value holds until then and is sent as
  `reasoning_budget_tokens`.
- Under `--api-key` the page is served without the key (it holds no data), asks for it on the first 401 and sends it as
  a Bearer token. Its fetches use `credentials: "omit"`: the 401's Basic challenge would otherwise open the browser's
  own login dialog. Every page fetch but `/health` goes through `api()`, which adds both. The key overlay is modal
  (`#appShell` is `inert` behind it, the key box has the focus); Forget the API key is in the phone `⋯` menu and in
  Settings › About.
- Conversations live in the browser's `localStorage` (every access guarded); attached images stay in memory only, as a
  few photos would fill the storage quota. Saved history is sanitised on load (a damaged chat or message is dropped or
  coerced, never fatal). Tabs share one list: a save merges the stored list by id (this tab's deletions stay deleted)
  and the `storage` event reloads it. A full store warns once and keeps the newest conversations. A new conversation is
  saved and listed when its first message is sent. A reply that breaks off (cut socket, an SSE `error`, a close
  without `finish_reason` or `[DONE]`) keeps the words already shown as an assistant message marked `interrupted` and
  shows Retry; Stop before the first token shows Retry too. Retry or Regenerate on a turn whose images did not survive
  a reload is refused with a toast, and such a message goes on the wire with a note in place of the images, never empty.
- The chat shows a model by its `state`: `loading` and `evicting` get a badge and no button, `error` gets Retry load.
  Error toasts leave after 8 s and a message already on screen is not shown twice.
- `/props` gives the version and the update banner: shown while `update.available`, its button POSTs `/v1/update`,
  polls `/health` until the server has gone down and come back, reloads, and reports the new version or `update.error`.
- Startup prints `chat in your browser: <url>/chat` once (`chatPageUrl`: a `0.0.0.0` bind shows as `127.0.0.1`); `sushi run`
  prints it under its banner, since its log is quieted to warn.

## Self-update (`POST /v1/update`, `/props.update`)

- `/props` carries `update: {current, latest, available, checked_at, url, error, command}` with or without a model
  (`update.propsJson`): `latest`/`checked_at`/`url` stay null until a daily check has answered; `error` is why the
  last update failed, cleared by the next success; `command` is `brew upgrade sushi` for a Homebrew install (the page
  shows it in place of its button), else null.
- `update.guard`, first refusal wins: a non-loopback bind 403 (update on the server with `sushi update`), a
  non-loopback peer 403, `--api-key` set and not presented 401 FROM LOOPBACK TOO, no Origin or one other than the bind's
  own `http://<host>:<port>` (127.0.0.1 and localhost interchangeable; DNS rebinding carries its own name) 403,
  `--parent-pid` 403 (the host updates its engine), a request decoding or queued or a model loading 409 `update_busy`
  (refused, never queued), an install that cannot replace itself (Homebrew, source build, app bundle, unwritable) 409
  by name, no newer release known 409.
- A running `sushibar` (the menu bar app) never blocks an update: it restarts itself when the `sushi` beside it changes version.
- Accepted: 202 `{"status":"updating","from","to"}`, then the SIGTERM shutdown path and the in-place updater
  ([server-lifecycle](server-lifecycle.md#self-update)).

## Agent launcher (`sushi launch <agent>`)

- `src/launch.zig` (claude/pi/omp/opencode/dsh/codex/hermes/aider/zcode/grok): reads `/v1/models`, writes agent configs into
  `~/.sushi/<agent>/`. Launcher env: `ANTHROPIC_BASE_URL` + dummy keys + `ANTHROPIC_DEFAULT_*_MODEL=sushi`.
- Every model id and URL in the zsh script is one single-quoted word (`appendQuoted`) and every config writer escapes
  them for its format (JSON/TOML/YAML `Esc`): ids come from folder names or a `--url` server, so never interpolate bare.
- codex gets `model_instructions_file` (`~/.sushi/codex/model-instructions.md`, written each launch from `src/assets/codex_model_instructions.md`): without it codex sends fallback instructions naming an `apply_patch` tool it no longer declares; keep the asset's tool list equal to codex's declared tools.
- opencode runs the newest version among every `opencode` on PATH, not the first (an old npm or `~/.opencode` copy
  would otherwise shadow Homebrew's 2.x); `--standalone` is probed on that binary.
- Claude Code's stream watchdogs and 10-min request timeout are raised and its non-stream fallback is off: a long
  prefill plus a long think tripped them, and each fallback re-sent the whole prompt, then timed out and retried.
- Agent budgets (`launch.budgetForContext` + `compactionReserve`): output share ctx/2, compaction reserve ctx/4
  capped at 20000, carried into pi's `settings.json` and opencode's `compaction` + `limit.output`. A launch below the
  agent's context floor WARNS (claude 64k, opencode 32k, others 16k).
- pi sends its thinking level as `reasoning_effort` through a per-model `thinkingLevelMap` built from the row's
  `reasoning_efforts` (`launch.piEffortFor`: exact, else the next accepted word up, else down, `on` being the lowest;
  Qwen3.8 high → xhigh, MiMo every level → on). pi's `thinkingFormat: qwen` sent only `enable_thinking`, so low/medium
  never reached the server. The launch tests build their vocabularies from `effort.effortArms`.
- omp (a pi fork) has no off entry in its maps: off rides the qwen dialect (`enable_thinking: false`), `whenThinking`
  switches thinking requests to `reasoning_effort`, and a per-model `thinking` block remaps each level with the same
  rule; `requiresEffort: false` stops omp clamping off to the lowest effort.
- opencode 2.x talks to a background service that never sees `OPENCODE_CONFIG_CONTENT` and refuses `--model` on its
  default command: the launcher passes `--standalone` only when CLI help advertises it (after a subcommand), carries the model as
  `model`, and marks a row with efforts `reasoning` + `interleaved: reasoning_content` + one `variants` entry per graded
  word (GLM: low/high/max; on/off make none).
- OpenCode `--think WORD` validates the selected model's advertised efforts and sets only its `options.reasoningEffort`; omission preserves existing defaults.
- OpenCode `--persist` backs up and merges the global JSON/JSONC config under `XDG_CONFIG_HOME` or `~/.config`, preserves unrelated settings, and normalizes comments into the backup; successful replacements print the backup's full path to stderr; invalid or ambiguous files are refused.
- opencode sends no `max_tokens`, and GLM reserves a request's whole window without one (1M rows): serve GLM with
  `--max-tokens N` for opencode (live: a 12k-token agent prompt hit `GlmReserveMemoryLimit` with 16 GB free). grok
  sends its configured `max_completion_tokens`.
- `sushi launch grok` writes `~/.sushi/grok/config.toml` and sets `GROK_HOME` there (the owner's `~/.grok` stays
  untouched): one `[model."<id>"]` per chat row on `api_backend = "chat_completions"`, dummy `api_key`, advertised
  `context_window`, `max_completion_tokens` from `budgetForContext`, the row's efforts as `reasoning_efforts`, and
  `[session] auto_compact_threshold_percent` = share of the window that leaves `compactionReserve` free (min 50).
- `sushi launch dsh` (DeepSeek Harness) writes ONE patch layer, `~/.sushi/dsh/patch.yml`, and starts `dsh --profile web
  --patch "$HOME/.sushi/dsh/patch.yml"` (`dsh` on PATH, else `npx --yes @deepseek-ai/dsh`; `web` is the README's
  interactive profile). The user's `~/.dsh/profiles/<profile>/cordis.patch.yml` is never read or written: dsh applies
  `--patch` after the profile layer, so the generated entries (`agent-default-model`, `llm-pi-ai` with `providers.sushi`)
  win only for their own ids and the user's other entries stay. That is the opencode/omp contract (the agent's own config
  untouched, sushi's in a sushi-owned place). The provider is `openai-completions` at `<base>/v1` with `compat`
  (`supportsDeveloperRole: false`, `supportsReasoningEffort: true`, `maxTokensField: max_tokens`), a `Bearer sushi`
  header (pi-ai refuses to send without a key; like every launch target the key is the dummy, the server exempts
  loopback), every chat row as a model with its own `contextWindow`, `input` and `reasoningEfforts` (each advertised
  word to itself; `on` as `high`), the default model's `default_reasoning_effort` as `reasoning`, and the loaded row (else
  the first) as the default model. `maxTokens` is `launch.dshMaxTokens`: half the window up to 256000, DeepSeek's
  own omitted-`max_tokens` default (`/v1/models` carries no output limit). Test: `python3 tests/test_dsh_launch.py --bin
  zig-out/bin/sushi`.
- `sushi launch zcode [--url U] --model ID [--print] [-- zcode args]` writes schema-1
  `~/.sushi/zcode/provider_config.json` and points `ZCODE_PERSONAL_PROVIDER_CONFIG_FILE`, `ZCODE_DATA_BASE_DIR` and
  `ZCODE_STORAGE_DIR` into `~/.sushi/zcode`; ZCode's own source and project config stay untouched. `--model` must be
  an advertised `/v1/models` chat ID (default: first loaded chat row; media/embedding rows excluded).
- ZCode speaks `/v1/chat/completions` (SSE reasoning, function calls, tool-result replay); each model gets
  `clamp(context / 2, 1024, 65536)` output (older rows: 32768 context, 8192 output) and exactly its advertised
  `reasoning_efforts` (prefers medium). CPU test: `python3 tests/test_zcode_launch.py --bin zig-out/bin/sushi`
  (`--zcode <zcode.cjs>` adds the real-client fixture).

## Web UI research tools

- The composer's **Tools on/off** button enables the same research pack as `sushi run`, off by default.
  The preference persists in this browser. It is fixed for a turn; the button is disabled while a reply runs.
- The **Edit chip** after Tools shows only while Tools is on and turns `write_file` and `edit_file` on for that
  chat only. A new chat starts from the server's `--edit` (off unless launched with `--edit on`); the state
  travels with the saved chat and returns to that default when the chat's folder changes.
- The browser sends definitions, assembles streamed tool calls, executes them through `POST /v1/tools`, and
  sends results back to the model. Eight tool rounds maximum, then the REPL's nudge (never saved to the chat) and a final request without tools.
  Results are collapsible in the transcript. Tool rows show the query, URL or file argument on one line,
  ellipsized to fit with the full label on hover (including saved conversations).
  Stop cancels browser requests and records cancelled results for
  remaining calls so the conversation stays valid. An already-running server tool may finish its bounded work.
- The **Folder** button opens a folder picker: browse subfolders, move to the parent, or enter an absolute
  path, then choose **Use this folder**. The selection is saved per chat; new chats start at the server's
  working folder. Selection is disabled during a turn. It never changes the server process's working directory.
- `POST /v1/tools` with `{ "vision": false }` lists definitions and the file root. With `name`, JSON-string
  `arguments`, and `vision`, it executes one call and returns `text` plus optional `image` data URL.
  `directory` optionally selects an absolute folder, resolved and validated with the REPL's `/cd` checks.
  `write` (boolean) is the chat's Edit chip and alone decides whether the two write tools are offered and run;
  the listing answers with `edit_default` (the server's `--edit`, what a new chat starts with).
  With `browse: true`, the endpoint instead returns `root`, `parent`, `directories`, and `truncated` for the
  picker (up to 1000 visible, non-secret subfolders). The browser passes the selected canonical directory
  separately from model arguments on every call. Vision models get `view_image`; returned images remain in memory only.
- This bridge requires a loopback bind and peer, the chat page's Origin, and the normal API-key policy.
  It works from `localhost` or `127.0.0.1`, not a remote browser or wildcard bind. File tools are confined to
  the chat's selected folder, with the existing hidden/secret-file and symlink checks; network tools keep
  the REPL's public-address restrictions. No MCP configuration is added.
- Writing rides the same bridge and gate (loopback peer, loopback bind, the page's Origin); every chat starts off.
  The model never supplies `directory`: the page does, from its own per-chat state, so no tool call moves the root.
- Same-origin rule: `POST /v1/load-model`, `/v1/unload-model`, `/v1/models/rescan` and the `/v1/responses` WebSocket
  upgrade answer 403 when the request carries an Origin other than this server's own page (`crossOriginRefused`);
  an explicit bind compares against its host, a wildcard bind against the request's `Host` when that is an IP literal
  (a LAN page is its own page at the address it reached; a DNS name is never trusted, since one rebound to this Mac
  sends itself as both Origin and Host) and against loopback otherwise; requests without an Origin (curl, SDKs) pass,
  and the inference routes stay CORS-open.
- Image workflow: `web_search` finds pages, `fetch_url` exposes up to 20 resolved image URLs from `img src`
  or `data-src`, and vision models use `view_image` to inspect them. This is page-based discovery, not a
  dedicated image-search index. Tool image results are visible when expanded; the answer can show a direct
  image URL using Markdown `![description](https://...)`. Displaying images does not require a vision model.
- Markdown image previews fetch through the existing public-address-checked `view_image` bridge (including
  redirect checks and the 2 MB limit), never directly from a model-selected browser URL. A source link remains
  when previewing fails. The page caches up to 32 image requests; pixels are not persisted in chat storage.
- Markdown emphasis follows CommonMark flanking: `*` and `**` open before a non-space and close after one, so `2 * 3 * 4`
  stays text; `***x***` is `<em><strong>`. A list nests by indentation to any depth (`readList`), each level keeping its
  numbered or bulleted kind and a numbered one its first number.
- A reader who scrolled up stays put through a tool round and the reply; `renderTranscript` follows the bottom only for a
  view already there, or when the user sends a message or opens a chat.
- No control sits inside another: a conversation row is a `listitem` holding a `button` (open) beside the Rename and Delete
  buttons, a Models row a `listitem` holding a link (`/models#<id>`, arrows move between them) beside Load. An Enter that
  rebuilds the list must `preventDefault`, or it also presses the control that took the focus.
- The dashboard's read of a newly loaded model's `/props` skips the 2 s minimum gap once, so the SSD cache tile fills at
  once; every other poll keeps the gap.
- SVG code blocks offer **Render SVG** / **Hide SVG**. Only clicking renders: the bundled DOMPurify 3.4.16
  SVG profile removes active content; styles, embedded HTML, images, animation and external references are
  disallowed. The result loads in an isolated image, never as live SVG in the chat DOM. The original code stays
  copyable. The upstream minified bundle is inline to keep the page self-contained; its license is in
  `src/webui/DOMPurify.LICENSE`. Updates should use a reviewed upstream release and rerun the renderer checks.
- The page stays under 1 MB (`tests/test_webui.sh` checks it). Shared design tokens (`light-dark()` colours, one
  spacing, type and radius scale) and components (`.btn`, `.badge`, `.card`, `.tile`, `.tbl`, `.set`, `.seg`, `confirm`
  dialog) serve all four tabs; `node tests/test_webui_shell.cjs` guards the top bar, the 390 px phone fit of every tab, the
  shared left edge and the font-size scale.
- Checks: `node tests/test_webui_tools.cjs`, `node tests/test_webui_effort.cjs`, `node tests/test_webui_prefix_cache.cjs`, `node tests/test_webui_dashboard.cjs`, `tests/test_webui.sh`, and the
  `web tools:` unit test. Generate the
  browser renderer suite with `node tests/test_webui_render.cjs /tmp/sushi-render-tests/index.html`, serve that
  directory locally (or open the HTML), and require **All renderer checks passed**.

## `sushi run` research tools (client-side)

- **The REPL orchestrates and runs its tools locally** (`src/repl_tools.zig`, loop `cli.runToolTurn`): it sends `tools`,
  runs the returned calls, appends `tool` messages and asks again. OFF by default: `--tool on|off`, `/tool on|off`,
  bare `/tool` shows the state and list. One dim trace line per call (`search:`, `fetch:`, `read:` …).
- Tools: `web_search` (GET html.duckduckgo.com, top 8 title/url/snippet, `uddg=` unwrapped, ads dropped),
  `fetch_url` (GET, ≤5 redirects, 10 s wall clock, 2 MB, HTML → text ≤20k chars), `read_file` (≤256 KB),
  `list_dir`, `search_files` (substring or regex, ≤100 hits), `view_image` (only when `/v1/models` lists `vision`),
  and with `/edit on` also `write_file` and `edit_file` (see the writing rules below).
- **8 tool rounds per user turn**, then a user nudge and one request WITHOUT tools for the final answer.
- **Only the latest USER turn's images are decoded** (`server.activeWireMediaIndex`): a tool image rides a synthetic
  user turn after the tool results. `/image <path>` attaches to the next message; a pasted path is never attached.
- **File tools are confined to one folder by REAL path**, the start folder until `/cd <folder>` moves it
  (`changeRoot`: absolute, `~` or relative to the current folder; must be a directory, symlinks resolved, a path with
  a secret name refused; bare `/cd` shows it). `..`, outside absolutes and escaping symlinks are refused, as are dot
  entries and secret names (`.env*`, `*.pem`, `*.key`, `id_*`, `*.p12`, `credentials*`, `*.keychain*`, `.ssh`,
  `.aws`, `.gnupg`), checked both as typed and after resolution (`confinePath`).
- **Editing is a second switch on top of the file tools**: `/edit on|off` in the REPL (needs `/tool on` too) and the
  per-chat Edit chip on the page; `--edit on` only sets the starting state of both. `write_file`/`edit_file` are
  offered only while it is on; a write call with it off is refused with how to turn it on.
- **A new file is confined through its parent.** `realpath(3)` fails on a missing last component, so
  `confineWriteTarget` falls back to `confineNewPath` (real parent re-checked with `within` + `componentRefusal`);
  dot and secret names stay refused, so a chat at `$HOME` cannot touch `.zshrc`.
- **A replacement lands atomically.** `replaceFileBytes` writes through `createFileAtomic` (a random exclusive temp
  name in the target's folder, then one rename) and copies the old file's permissions onto it, so a failed write
  leaves the original whole and an edited script stays executable.
- **`write_file`** holds at most `max_write_bytes` (= the 256 KiB read cap, so every write reads back whole) and
  refuses a folder or a binary target. **`edit_file`** reads to EOF, refuses a file over the cap rather than rewrite
  a partial read, needs a non-empty `old_string`, and one match unless `replace_all`. Both arm the `fetch_url`
  approval like the reads.
- An outside path's refusal tells the model the folder is fixed and the user can type `/cd <folder>`, so it asks for
  that instead of guessing other paths.
- The user's `/image <path>` (`loadUserImage`): a RELATIVE path resolves in the `/cd` folder under the same
  confinement; an ABSOLUTE or `~` path (typed or dragged in) may leave it, but a secret name anywhere on it or a hidden
  file name is refused, as typed and resolved (`userPathRefusal`). The model's `view_image` stays confined.
- Every prompt carries that folder and the tools state, dim: `~/project · tools on >>> ` (`formatPromptStatus`: `~` for
  `$HOME`, `…` and the tail past 32 characters); it is rebuilt before each input, so `/cd` and `/tool` show at once.
  The ready banner prints the same pair.
- **Web tools reach public hosts only**: http/https, no userinfo, local names refused, EVERY resolved address and the
  connected peer (`getpeername`, defeats DNS rebinding) must classify public (`classifyIp4/6`; mapped, NAT64 and 6to4
  judged by their IPv4); each redirect hop re-checked; no cookies, auth headers or POST.
- **Once a file tool has run in the session, a `fetch_url` whose URL has a query string or a path over 80 characters asks
  the user y/N** with the sanitized full URL (`fetchNeedingApproval`; a non-tty stdin answers no): a page-steered model
  must not carry file content out in a URL.
- **Everything the model chose reaches the terminal through `TermFilter`** (trace lines, thought, answer, server
  errors): ESC/CSI/OSC sequences, C0 other than `\n` `\t`, DEL, C1 and invalid UTF-8 are dropped, with state across
  stream deltas, so a tool argument or a fetched page cannot rewrite the trace line.
- **The prompt reads a tty line with ICANON off** (`repl_input.zig`): a canonical line stops at 1024 bytes on macOS, so a
  pasted stack trace never submitted. Echo, backspace and Ctrl-U are ours; raw mode is held only while a line is read,
  and an `atexit` hook restores the terminal when Ctrl-C ends the process mid-read.
- Every failure is a short tool-result string; results are data, never executed. A DuckDuckGo bot check (HTTP 202,
  `anomaly-modal`) reads as "search unavailable", never as zero results.
