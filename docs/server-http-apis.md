# Server: HTTP APIs, streaming, sampling and constrained output

What each HTTP surface promises its clients: the OpenAI chat/completions/Responses and Anthropic Messages contracts,
streaming and usage chunks, logprobs, seeds and sampling, reasoning budgets and constrained JSON, plus the agent
launcher. Read this before touching `src/server.zig`, `src/responses.zig`, `src/launch.zig` or the sampler.

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
  budget on inherited arches): each served arch accepts a subset (`model.effortArms`), read through
  `model.selectEffort` on chat, Responses, Anthropic, `--think` and the REPL's `/think` alike; a word outside it is
  refused with the accepted list, never rounded. Uncapped = `--reasoning-budget`.
- qwen4_exp: off, low (2048), medium (8192), xhigh (uncapped); the template reads the word; `minimal`, high, max 400.
- glm5_next: low, high, max (uncapped, a prompt instruction); off and every other word 400.
- mimo_v2 advertises off, on: its template has only on/off, so every thinking word (`minimal` through `max`) is on,
  uncapped, and off/none is off.
- `/v1/models` rows, loaded or not, list `reasoning_efforts` and `default_reasoning_effort`, the word a request naming
  none runs at (`model.defaultReasoningEffort`: `--think`, else GLM high, else off/on per `defaultEnableThinking`,
  where a thinking Qwen silence renders low).
- `--think X` is checked against the `--model` model only; a model loaded on demand takes X by the same rule (MiMo:
  any X but off is on) or, lacking it, keeps its own default. It never fails a load.
- A thinking request that names NO effort gets no server budget (decided): Qwen3.8 renders it as low
  (`chat.qwen38EffortFor`), whose preamble shortens the thought without truncating it. `/v1/responses`:
  `sequence_number` on every event, stateful via
  `ResponseStore`, WS via Upgrade. Continuing a partial reply: `continue_final_message` explicit on chat, INFERRED on
  `/v1/messages`.
- **Anthropic `/v1/messages`** (Claude Code): typed blocks, `input_schema`→`parameters`, stop-reason map incl.
  `stop_sequence` echo, full SSE block lifecycle; a `system`-role message past index 0 (Claude Code's hook output,
  Codex's mid-input `developer` turn on Responses) renders where it was sent, folded only for a template that cannot
  place it ([server-tool-calling](server-tool-calling.md#templates)); `developer` reads as `system` (`canonicalRole`).
- **`preserve_thinking`** (templates that read it: Qwen3.8), resolved per request as `chat_template_kwargs.preserve_thinking`
  (bool, all three surfaces) > `--preserve-thinking on|off` > `preserve_thinking` in model-settings.json > the template
  default (undefined: every turn's thinking kept). Off renders only the latest user turn's thinking.
- `tool_choice` reads every surface's wire shape; `required`/`any` and a named function are enforced at decode, a
  named function the request does not declare is a 400 ([server-tool-calling](server-tool-calling.md#tool_choice)).
- `/v1/models` rows carry `context_length` + `max_model_len` at TOP level. Context-overflow 400s name BOTH counts.
- A prompt that tokenizes to zero tokens is a 400 `invalid_request_error` and never reaches generate: completions
  refuse it after tokenizing, `Scheduler.submit` refuses it for every surface (`error.EmptyPrompt`), and
  `Generator.initWithOptions` returns that error rather than index the prompt's last token.
- `/v1/models` `meta.quantization` reports EXL3’s configured expert rate and dense width (e.g. `EXL3 3bpw experts, 8-bit dense`) for loaded and unloaded packs; affine labels remain `{bits}-bit`, and `/props` numeric quantization fields retain their dense-trunk meaning.
- Endpoint EXISTENCE never depends on model state and the 404 is answered BEFORE the model resolves (`ROUTE_PATHS`);
  a status route never reaches `ensureLoaded` (`handlePropsNoModel`). Removed upstream routes answer named 404s.
- **Only an unknown NAME falls back to the default model** (SDK names like `gpt-4`); a PATH (`/…`, `~/…`, never
  `org/repo`) names its own entry as `/v1/load-model` resolves it (`routeRequestModel` → `registry.peekPath`), and an
  unregistered path is a 404 `model_not_found` (Anthropic `not_found_error`), never another model's answer.
- A content array's text parts JOIN in order (`joinedTextParts`); its media parts render at the offset they sat at.
- **Media is read from EVERY message** on all three surfaces: chat `image_url`/`video_url` parts in any role
  (`tool` included), Anthropic `image` blocks beside the text or inside a `tool_result`, Responses `input_image` in
  a message or a `function_call_output` array. Only base64 data URLs decode (remote URLs are refused, not fetched);
  a failure is a 400 naming the message index and the reason, an `input_audio` part a 400 on a model without an
  audio encoder, more than 64 images a 400 with both counts, an encode failure a 500 (Anthropic: `api_error`).
  Stored Responses history keeps text only.

## Streaming

- **A stream and a non-stream answer are the SAME BYTES**; leading whitespace is the one thing a stream may withhold
  (`streamContentLead`). A spent reasoning budget WITHHOLDS the rest of the thought; a non-stream tool-call reply
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
  `sushi launch omp` sets the Sushi provider's `compat.streamIdleTimeoutMs: 0` so buffered calls can finish;
  explicit omp timeout settings, environment overrides and per-call options still take precedence (verified with
  omp 18.3.0). Non-loopback `--url` targets also have a separate first-event deadline: omp's
  `providers.streamFirstEventTimeoutSeconds` controls it, with zero allowing unlimited initial buffering.
  `--timeout` remains a token-progress STALL timeout (`StallClock`).
- **NO string built from model bytes is guaranteed UTF-8**: sanitizing lives INSIDE the escaper (`chat.utf8Next`
  under every `jsonEscape`/`appendJsonString`); logprobs `bytes` keeps the exact bytes. Hand-written error text is
  escaped at the SINK.

### Closing a streamed response

`Conn.close` half-closes the write side of a close-delimited response, then holds the socket until the peer
hangs up, server shutdown starts, or five minutes pass. This prevents macOS's orphaned FIN_WAIT_2 timeout from
resetting a client that is still reading after the final SSE event. The request has already released its model
slot; only the connection thread waits. Responses with `Content-Length` close immediately.

Ported from [mlx-serve #673](https://github.com/ddalcu/mlx-serve/pull/673). Socket-pair tests cover peer closure,
length-framed responses and shutdown; `tests/test_responses_streaming.sh` also waits past the TCP FIN timeout
before reading a completed stream and requires clean EOF.

## Seeds, logprobs, sampling

- **A `seed` binds EVERY sampler with a fresh key PER DRAW** (`generate.seedKey`).
- Logprobs are the MODEL's distribution (pre-temperature), ids travel WITH values, entry belongs to the RETURNED
  token (one-token delay); `logprobs.content` describes `message.content` (`contentTokenRange`); streaming logprobs
  are a SIBLING of `delta` shipped EXACTLY once against a high-water mark. logprobs>0 + grammar disable spec.
- Logprobs are `logits - logsumexp` in f32 (`computeLogprobs`): `log(softmax)` in bf16 lands on bf16's grid, 0.125
  apart between -16 and -32.
- Sampling defaults for omitted fields: body > launch flags > model `generation_config.json` > hardcoded.
- Top-k and top-p are ONE pass (`filterTopKTopP`); a filter cuts by RANK, never by value (bf16 ties at the top
  constantly; `ranksDescending` ties by lowest id); the nucleus is the mass STRICTLY above each rank, cumsum in f32;
  `top_p` 0 is GREEDY (`applyTopP` floors at `floatMin(f32)`).
- A sampler never draws a RESERVED special or PADDING row (`installSuppressMask`; logprobs stay RAW).
- **`ignore_eos: true`** (vLLM's field) decodes a `/v1/completions` request past EOS to `max_tokens`
  (`requestEosSlice`); stop sequences and the loop stops still end it, and its text skips `special: true` tokens
  (`completionShowsToken`, vLLM's default `skip_special_tokens`). Chat refuses it with a named 400
  (`chatIgnoreEosRejectReason`): past its end of turn the model writes another turn, and that turn's think block is
  merged into the reasoning by the non-stream reply (`normalizeEmbeddedThinkBlocks`) but not by the live stream.
- **A repeat/presence penalty samples on the synchronous serial path** (`samplesSync`, like logprobs): no draft
  path (`draftsRefused`), no batched tick (`.penalty`), never the lazy pipeline, which never applied it.
  A penalty-mask allocation failure fails the request instead of sampling without its penalty. Chat and
  completions parse it alike (`parseRepeatPenalty`); a repeat penalty of 0 or below is off.
- **Think penalty** (`think_penalty` request > `--think-penalty` > model-settings.json > off, 0-20; arXiv 2606.00206):
  while the thought is open every single-token spelling of the paper's markers (bare or space-led, lower or capital)
  loses λ; a split spelling is skipped (its first piece starts other words). The paper penalises everywhere; we stop
  at the closer. Every sampled position, serial or verify row, is shifted by its own prefix (`thinkShifted`,
  `thinkShiftRows`), so greedy MTP keeps serial's bytes; logprobs stay RAW.


## Experimental logit bias

`--logit-bias-file <path>` loads a JSON or CSV file at model load; the launch flag overrides the per-model
`logit_bias_file` setting. The default is off. Each entry names exactly one `id`, exact vocabulary `token` string,
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
head, including positive rewards. Serial, MTP, PLD and prompt-lookup rows apply the same prefix-dependent shifts;
streaming and non-streaming preserve the same generated tokens. Existing non-stream-only repetition-tail trimming
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
(`model.effortArms` for served arches, `responses.effortBudget` for the rest).
Every surface arms it with one precedence: explicit budget (`reasoning_budget_tokens`, Anthropic `budget_tokens`) > the
request effort word's budget. When the request omits effort, the architecture's `--think` budget supplies the default;
uncapped effort words fall back to `--reasoning-budget` (unlimited by default). GLM effort words do not impose token caps.

## Constrained JSON

- Constrained generation reports each returned token against its raw logits before the grammar mask, including forced choices; those logprobs have no one-token delay.

- The payload offset is AUTHORITATIVE (`reasoning_protocol.Delivery`, all surfaces, stream + non-stream).
- The grammar mask never walks the whole vocabulary (`token_mask.buildMask`); every grammar state has a legal byte;
  no whitespace OUTSIDE the root value, the model's OWN layout inside (`MAX_FREE_WS` 16).
- A root number ends only at EOS, so it is complete once terminable; the empty-mask disable in `nextConstrainedToken`
  is the logged last resort (a new dead end gets a schema in the `no reachable grammar state is a dead end` test).
- Every schema-mask surface uses ONE thinking policy (`schemaMasksThinking`); tools present = no mask. Per-model
  grammar table lives on `LoadedModel`.
- Code: `src/json_schema.zig` / `src/json_grammar.zig` / `src/token_mask.zig` / `src/regex.zig` (schema IR →
  streaming grammar → per-token mask), `src/reasoning_protocol.zig`.

- `/props.settings.prefix_cache` = `{ram_enabled, mem_bytes, disk_bytes, disk_used_bytes, disk_entries}`: `disk_bytes` is
  the SSD budget the model was given at load (0 = tier off), `disk_used_bytes` and `disk_entries` what the tier holds,
  published by the inference thread after each commit (the web page's meter reads them; the same numbers log as
  `[disk-cache] usage <used> / <budget> GB, <n> entries` once per finished turn). They are the REQUESTED model's own
  tier (`/props?model=X`), the four numbers from one publish. `GB` is binary (1 GB = 1 GiB), as the flags read it.
- `/props.settings.prefill_decode_share` reports the effective process-wide share, including zero under the interleave kill switch.

## Security and observability

- `--api-key`: loopback exempt, `/health` + OPTIONS + `GET` of the chat page open, `constTimeEql`.
- A JSON request nests at most `chat.max_json_nesting` (256) levels, checked on the raw body before any route parses it (HTTP and each WebSocket message); an assistant history `tool_calls[].arguments` nested deeper is sent to the template as a string. A schema nests at most `json_schema.max_schema_depth` (64).
- A schema `pattern` repeat count is at most 1000 and its NFA at most 32768 states (`regex.zig`); past either, `compile` fails with `InvalidPattern`.
- `--metrics`: zero cost off; TTFT at prefill completion; live tok/s via ONE atomic per tick; `/metrics(.json)`.
- **A request outcome is counted exactly once** (`Slot.metrics_recorded`, inference thread): `finishSlot` or the cleanup drain,
  whichever sees the slot first (`recordSlotEnd`/`recordSlotCleanup`). The outcome comes from the slot's finish state, never
  from `Slot.cancelled` (`complete` sets it on every completion): success feeds the histograms; `sushi:request_cancelled_total`
  (a disconnect mid-decode; a request ended by its own stop sequence sets `Slot.stop_hit` first and counts as success), `sushi:request_failed_total` and a refusal before a slot
  (`sushi:request_rejected_total`: context overflow, memory preflight, `PrefillDoesNotFit`) move only their counter.
- `/metrics.json` ends with `"sessions"`, one row per live request (phases `prefill` and `decode`, cap 32; published
  by the inference thread under `queue_mu`, copied by the reader under the same lock — there is no separate
  `/requests` route): `model`, `request_id` (submit sequence; stable across polls of one request, never reused),
  `phase`, `context_tokens` (prompt + decoded; the FULL prompt on a prefill row), `context_length` (the model's
  effective limit, 0 = not ready/unknown), `cached_tokens`, `generated_tokens`, `max_tokens` (the request's own
  output cap), `elapsed_seconds` (age since arrival, refreshed every publish) and `state_bytes` (GPU bytes of the
  slot's own KV + SSM buffers; a restored share stays billed to the hot-cache entry, never to the row).
- A live row's `state_bytes` is at capacity: a ringed layer's ring, each QSA bank by its capacity buffer (never also
  its view), and the ring restore points the slot holds.
- After the live rows come `cached` rows, one per hot-cache entry of every ready model (cap 32, published under
  `digest_mu`): `context_tokens` = `cached_tokens` = the entry's tokens, `state_bytes` its `kv_bytes`. A cached row has
  no request, so `request_id`, `max_tokens`, `generated_tokens` and `elapsed_seconds` are 0.
- An entry a live row restored from is listed once, as that row: the dedupe keys on an internal entry id that
  `/metrics.json` does not emit.
- `/props` `template_fallbacks` counts renders where the model's own chat template raised and the generic format
  answered ([server-tool-calling](server-tool-calling.md#templates)); the count is process-wide, and it is zero when
  every prompt is the checkpoint's own.
- `/props` `memory.kv_cache_bytes` = the current model's hot-cache residency + every live slot's state, published
  each tick with or without `--metrics`; a donated checkout's buffers, which the entry bills until release, count once.

## Chat page (`GET /`, `GET /chat`)

- One self-contained file, `src/webui/index.html` (CSS, JS and the logo inline, no external fetch), embedded with
  `@embedFile` and served as `text/html; charset=utf-8`. Any other method on those two paths is a 405 answered BEFORE
  model resolution, so it can never cold-load a model. Guards: `tests/test_webui.sh`, the `chat page:` tests.
- The right sidebar's "Prefix cache (SSD)" section is a bar meter of `/props` `settings.prefix_cache`
  (`disk_used_bytes / disk_bytes`, "<used> / <budget> GB" and the entry count beside it). It is read after each reply
  and again 2.5 s later (the server stores a turn's entry after answering) and stays hidden while `disk_bytes` is 0.
- It speaks only the public API: `/v1/models` for the picker (`vision` or an `image` input modality shows the attach
  button), `/v1/chat/completions` streamed with `include_usage` (the
  readout is the final chunk's `usage` + `timings`), `reasoning_content` shown collapsed. Stop aborts the fetch; the
  server cancels on disconnect.
- The effort menu lists exactly the row's `reasoning_efforts`, with no default entry; it shows the stored choice where
  the model takes it, else the row's `default_reasoning_effort`, and every request sends the shown word.
- Under `--api-key` the page is served without the key (it holds no data), asks for it on the first 401 and sends it as
  a Bearer token. Its fetches use `credentials: "omit"`: the 401's Basic challenge would otherwise open the browser's
  own login dialog.
- Conversations live in the browser's `localStorage` (every access guarded); attached images stay in memory only, as a
  few photos would fill the storage quota.
- `/props` gives the version and the update banner: shown while `update.available`, its button POSTs `/v1/update`,
  polls `/health` until the server has gone down and come back, reloads, and reports the new version or `update.error`.
- Startup prints `chat in your browser: <url>` once (`chatPageUrl`: a `0.0.0.0` bind shows as `127.0.0.1`); `sushi run`
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
- Accepted: 202 `{"status":"updating","from","to"}`, then the SIGTERM shutdown path and the in-place updater
  ([server-lifecycle](server-lifecycle.md#self-update)).

## Agent launcher (`sushi launch <agent>`)

- `src/launch.zig` (claude/pi/omp/opencode/codex/hermes/aider/zcode/grok): reads `/v1/models`, writes agent configs into
  `~/.sushi/<agent>/`. Launcher env: `ANTHROPIC_BASE_URL` + dummy keys + `ANTHROPIC_DEFAULT_*_MODEL=sushi`.
- Every model id and URL in the zsh script is one single-quoted word (`appendQuoted`) and every config writer escapes
  them for its format (JSON/TOML/YAML `Esc`): ids come from folder names or a `--url` server, so never interpolate bare.
- Claude Code's stream watchdogs and 10-min request timeout are raised and its non-stream fallback is off: a long
  prefill plus a long think tripped them, and each fallback re-sent the whole prompt, then timed out and retried.
- Agent budgets (`launch.budgetForContext` + `compactionReserve`): output share ctx/2, compaction reserve ctx/4
  capped at 20000, carried into pi's `settings.json` and opencode's `compaction` + `limit.output`. A launch below the
  agent's context floor WARNS (claude 64k, opencode 32k, others 16k).
- pi sends its thinking level as `reasoning_effort` through a per-model `thinkingLevelMap` built from the row's
  `reasoning_efforts` (`launch.piEffortFor`: exact, else the next accepted word up, else down, `on` being the lowest;
  Qwen3.8 high → xhigh, MiMo every level → on). pi's `thinkingFormat: qwen` sent only `enable_thinking`, so low/medium
  never reached the server. The launch tests build their vocabularies from `model.effortArms`.
- omp (a pi fork) has no off entry in its maps: off rides the qwen dialect (`enable_thinking: false`), `whenThinking`
  switches thinking requests to `reasoning_effort`, and a per-model `thinking` block remaps each level with the same
  rule; `requiresEffort: false` stops omp clamping off to the lowest effort.
- opencode 2.x talks to a background service that never sees `OPENCODE_CONFIG_CONTENT` and refuses `--model` on its
  default command: the launcher passes `--standalone` (after a subcommand, flags bind to it), carries the model as
  `model`, and marks a row with efforts `reasoning` + `interleaved: reasoning_content` + one `variants` entry per graded
  word (GLM: low/high/max; on/off make none, and a default effort option would send words GLM refuses).
- opencode sends no `max_tokens`, and GLM reserves a request's whole window without one (1M rows): serve GLM with
  `--max-tokens N` for opencode (live: a 12k-token agent prompt hit `GlmReserveMemoryLimit` with 16 GB free). grok
  sends its configured `max_completion_tokens`.
- `sushi launch grok` writes `~/.sushi/grok/config.toml` and sets `GROK_HOME` there (the owner's `~/.grok` stays
  untouched): one `[model."<id>"]` per chat row on `api_backend = "chat_completions"`, dummy `api_key`, advertised
  `context_window`, `max_completion_tokens` from `budgetForContext`, the row's efforts as `reasoning_efforts`, and
  `[session] auto_compact_threshold_percent` = share of the window that leaves `compactionReserve` free (min 50).
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
- The **pencil chip** next to Folder turns `write_file` and `edit_file` on for that chat only. It starts
  off for every new chat, travels with the saved chat, and goes back off the moment the chat's folder
  changes. It is disabled, with the reason in its tooltip, when the Tools chip is off or when the server
  was started without `--edit on` — the page can ask for less than the server allows, never more.
- The browser sends definitions, assembles streamed tool calls, executes them through `POST /v1/tools`, and
  sends results back to the model. Eight tool rounds maximum, followed by a final request without tools.
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
  `write` (boolean) is the chat's pencil chip; the listing answers with `edit_allowed` (the server's `--edit`
  ceiling) and offers the two write tools only when ceiling and request are both on.
  With `browse: true`, the endpoint instead returns `root`, `parent`, `directories`, and `truncated` for the
  picker (up to 1000 visible, non-secret subfolders). The browser passes the selected canonical directory
  separately from model arguments on every call. Vision models get `view_image`; returned images remain in memory only.
- This bridge requires a loopback bind and peer, the chat page's Origin, and the normal API-key policy.
  It works from `localhost` or `127.0.0.1`, not a remote browser or wildcard bind. File tools are confined to
  the chat's selected folder, with the existing hidden/secret-file and symlink checks; network tools keep
  the REPL's public-address restrictions. No MCP configuration is added.
- Writing rides the same bridge and gate (loopback peer, loopback bind, the page's Origin); `--edit` defaults off.
  The model never supplies `directory`: the page does, from its own per-chat state, so no tool call moves the root.
- Same-origin rule: `POST /v1/load-model`, `/v1/unload-model`, `/v1/models/rescan` and the `/v1/responses` WebSocket
  upgrade answer 403 when the request carries an Origin other than this server's own page (`crossOriginRefused`);
  requests without an Origin (curl, SDKs) pass, and the inference routes stay CORS-open.
- Image workflow: `web_search` finds pages, `fetch_url` exposes up to 20 resolved image URLs from `img src`
  or `data-src`, and vision models use `view_image` to inspect them. This is page-based discovery, not a
  dedicated image-search index. Tool image results are visible when expanded; the answer can show a direct
  image URL using Markdown `![description](https://...)`. Displaying images does not require a vision model.
- Markdown image previews fetch through the existing public-address-checked `view_image` bridge (including
  redirect checks and the 2 MB limit), never directly from a model-selected browser URL. A source link remains
  when previewing fails. The page caches up to 32 image requests; pixels are not persisted in chat storage.
- SVG code blocks offer **Render SVG** / **Hide SVG**. Only clicking renders: the bundled DOMPurify 3.4.16
  SVG profile removes active content; styles, embedded HTML, images, animation and external references are
  disallowed. The result loads in an isolated image, never as live SVG in the chat DOM. The original code stays
  copyable. The upstream minified bundle is inline to keep the page self-contained; its license is in
  `src/webui/DOMPurify.LICENSE`. Updates should use a reviewed upstream release and rerun the renderer checks.
- Checks: `node tests/test_webui_tools.cjs`, `node tests/test_webui_effort.cjs`, `node tests/test_webui_prefix_cache.cjs`, `tests/test_webui.sh`, and the
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
- **Editing is a second switch on top of the file tools**: `--edit on|off` and `/edit on|off` (needs `/tool on` too),
  the per-chat pencil chip on the page (capped by `--edit`). `write_file`/`edit_file` are offered only while it is on;
  a write call with it off is refused with how to turn it on.
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
