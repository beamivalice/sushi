# Server: chat templates, thinking and tool calling

How prompts are rendered and how model output is split into reasoning, content and tool calls: the Jinja render and
its silent fallback, the tool-call parse chain and its one chokepoint, think-tag handling, loop stops, and the
replay-pinned invariants. Read this before touching `src/chat.zig`, `src/tool_calls.zig`, `src/tool_schema.zig`, `src/think_tags.zig`, the tool paths in `src/server.zig`, or
`src/format_corpus_test.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [server-http-apis](server-http-apis.md),
[tests/CLAUDE.md](../tests/CLAUDE.md).

## The pipeline

- With `tools`, tokens buffer for detection; thinking buffers separately. Parse chain strict → tolerant repairs →
  truncation salvage, then the ONE chokepoint `server.parseToolCallsForRequest` = parse → inferred-name filter →
  parallel clamp → buried-param hoist → schema coercion (last two gated by `--no-tool-autocorrect`; emitted
  `arguments` ALWAYS valid JSON).
- Template input `chat.templateMessages`: role "tool" native, a call's `arguments` the client's JSON object (`{}` for
  anything else), every string valid UTF-8 (U+FFFD for a bad sequence, as `appendJsonString` writes). Streaming: full args in ONE SSE delta, thinking → `reasoning_content`.
- Buffered calls may exceed a client's inter-event progress deadline even while SSE keepalives arrive. The omp launcher
  disables that deadline only for its Sushi provider; argument truncation and loop-stop rules remain unchanged.
- **Hard invariants (replay-pinned)**: emitted args ALWAYS valid JSON; every converter escapes + dedups; coercion
  never worsens conformance; a parsed NAME never contains `<|`; no tag leaks. Harness:
  `src/tool_traffic_replay_test.zig` over `src/fixtures/tool_traffic.jsonl`.
- **Control-token text a client sends is text**: every byte of a render knows who wrote it (`Template.renderSpans`):
  the template (and the server's own text around it: tool fallbacks, image atoms, tails), an assistant message, or
  other request text (user, system and tool messages, tools, the effort word, the tool_choice instruction and a forced
  call's function name, which name the client's function). `Tokenizer.encodeSpans` reads a
  special's text as its token only where every byte of it is the template's, or an assistant's and the model can
  write that special inside a turn; in other input, or put together from two origins, it is text, and text is never
  split where its origin changes. A file read into Claude Code quoting `<think>`/`<|im_end|>` otherwise read as a real
  turn or thought boundary. The text itself is unchanged: a template reads what the client sent, U+FDD0 included.
- **Assistant text**: the model's earlier output, so a special it can write stays its token and the next prompt holds
  the ids it generated. A stop (eos, an arch's turn opener) and a special the sampler masks are text there: the model
  never writes either inside a turn. `Tokenizer.bindTemplate` derives both sets once per load from the flagged
  specials, the template's facts and the stops; with no template nothing is masked and assistant text keeps every
  special. Origins follow bytes through every builtin (`template.zig`'s builtin sweep pins it); a transform with no
  byte-for-byte map (`tojson`, `indent`, `reverse`, a container's repr) takes the strictest origin of its input.
- **Token reuse across turns** (`segment_cache.zig`): a segment, the text between two specials the encoder
  recognised, encodes to the same ids wherever it sits, so each loaded model keeps the ids of the segments it encoded
  and an agent turn encodes only its new ones. Origin decides where a segment ends, never what it encodes to. The cache
  is the tokenizer's, dropped with it at unload: 64 MiB per model, everything it allocates billed (each entry pays
  its node and eight map slots; the map is rebuilt when removals leave tombstones or it outgrows its entries), least
  recently used out first, one entry at most a sixteenth of that, segments under 32 bytes never kept, one lock for
  every request thread. Pageable host memory, so admission does not bill it (as the SSD writers' bytes); `/metrics`
  reports it (`token_cache`). The corpus pin holds cached ids equal to uncached ones on the three pack fixtures, eight
  threads at once, under eviction too. Measured (`taskpolicy -a` on the M1 Max, Qwen 3.8's real tokenizer, the
  tool-calling conversation of "the largest served conversation", best of three, one process): 400 turns (0.87 MB,
  218k ids) encode in 2,235 ms without it, 2,305 ms cold, 41 ms warm, and the next turn in 43 ms; 4,000 turns (8.7 MB,
  2.2M ids) in 22.3 s, 23.4 s cold, 0.41 s warm and 0.39 s for the next turn, with 21.6 MB held. Rendering takes 4.5
  and 46 ms. The request cache (`tokenize_cache.zig`) stays: an exact repeat through it costs 0.15 and 1.8 ms.
- A live failure revealing a CLASS ships the instance test plus a corpus entry or invariant in
  `src/format_corpus_test.zig` plus a rule here. Capture traffic: `SUSHI_RAW_DUMP_FILE=<abs>` →
  `tests/harvest_tool_traffic.py`. Reproduce tool bugs `stream:false` first.

## Templates

- One-shot CLI prompts carry the same thinking fields through the chat request policy; no CLI render may hardcode thinking off.
- **Dialects are named by format and asked of the compiled template, never of the model type**: `templateSettings`
  maps the client's effort word per template dialect (scale: `thinking_mode`, a 1-100 ladder; ladder: the literal
  `'xhigh'`, xhigh|medium|low; word: `effective_reasoning_effort` with `[gMASK]<sop>`, the word as sent; any other gets
  high/no_think). An arch brings only what a template cannot say: a template it pins when the checkpoint's cannot be
  parsed by the engine (`pin` in its `<arch>_chat.zig`, read by `loadChatConfig` from the model type in `config.json`;
  DeepSeek-V4.1's, with its image atom), and a pre-tokenizer spelling (`pretokenizer`, read by the tokenizer). See
  [engine-arch-seam](engine-arch-seam.md).

- **The engine is `src/template/`, a Jinja2 renderer in Zig held to Python Jinja2**: `formatChat` calls it once per
  render through `chat.renderEngine` (`Template.render`: the compiled tree and a `Context` the caller builds in the
  render's own arena, once per render: the messages as `chat.templateMessages` shapes them, the request's tools read
  once by the engine's `fromJson` (exact digits) with the optional keys `fillToolDefKeys` adds, and the settings of
  `chat.templateSettings`; no JSON text between server and engine, strings referenced, not copied; the prompt, or `TemplateSyntax`, `TemplateRaised`, `TemplateRuntime` with line, column and message
  out). A template compiles once, at load (`template.Template`, `ChatConfig.compiled`, the late-system form beside
  it): one that does not compile fails the load with `[load] chat template: line L column C: message`; a render only
  reads the tree, on any thread. Each probe render (tool content, late system, forced call) is memoized on its
  compiled template by probe and settings (`ProbeMemo`: 256 entries, started over when full). It reads what
  jinja.cpp read (the filters, tests, methods, globals and loop fields of `src/template/builtins.zig`; the names only
  jinja.cpp served give its result, pinned in `jinja_cpp_only`); any other filter, test or statement (`include`,
  `extends`, `raw`, `recursive`, `**`, `//`) is a `TemplateSyntax` naming it, and a template leaning on a
  jinja.cpp leniency (a list mutator but `append`, `|length` of a `map` result) fails at render, both with the failed
  render's handling below. Limits, each a `TemplateRuntime` by name: 200 levels of nesting (a link of `a + b + c` or
  `x|f|g` counts), 100 macro calls deep, 20M steps, a 256 MiB string, 1 GiB of render memory, a `range` of 100000.
  Every fn there opens with `@setRuntimeSafety(true)` (lint `template-runtime-safety`): template text comes from packs.
  Work is metered by what it touches (a comparison or a search by its bytes), and searches and sorts are linear or
  n log n whatever the text: `value.Finder` (KMP), `value.stableSort`, which a NaN key fails by name (lint
  `template-superlinear-std`). The continuation search in `chat.zig` uses the same finder. A loop over `range(...)` is
  lazy and a loop pays per item walked, as Python's: GLM-5.3 scans `range(i, messages|length)` per tool block and
  breaks at once. The limits hold a 4,000-turn, 7.7 MB conversation on every served template under a quarter of each
  (test "the largest served conversation"); a raise longer than the 1 KiB Diagnostic is cut at a character and never
  matches a known raise (`Diagnostic.messageIs`). The oracle keys each engine call by the JSON text its context once
  was: `chat_json_test.zig` writes it, byte for byte the old serializers' text, in test binaries only (it fails to
  compile into any other build).
- **The server asks a template through its facts, never its text** (`template/facts.zig`, `chat.asks`; lint
  `template-text-search`): the names it reads, its literals, the literals it compares against, and the literals it MAY
  print. The may-print set errs toward output: only comparison and test operands, subscript keys, slice bounds and
  raise arguments are left out; a call's argument counts wherever the call sits (`l.append(x)` stores it), and
  `raise_exception` is no raise once anything in the template binds that name (a macro, a set, a loop target, a
  macro's or a caller's parameter). A word in a comment counts for nothing. What rides on it:
  rendering the tool list, the tool-role probe, `preserve_thinking`, the think span, each effort dialect, `/v1/models`
  reasoning support, the generated think opener and channel protocol, and the reserved-token mask
  (`Tokenizer.bindTemplate` keeps a special the template may print). Limit: a literal only pieced together
  (`'<|im_' ~ 'end|>'`) is seen by neither the facts nor a text search. A config built by hand calls `compile`
  (`arch_fixtures_test.qwenChatConfig` does): `ChatConfig.facts` panics on a template never compiled or swapped in
  after its compile (`compiledFacts` returns `TemplateNotCompiled`), so no question answers from empty facts.
  One render of a tool-calling conversation, the same bytes from both engines (the engine as landed on `54e1aef3`,
  ReleaseFast test binary, M5 Max, best of three, the four served templates): 400 turns (0.8 MB) 2.2 to 3.8 ms where
  jinja.cpp took 90 to 597 ms; 4,000 turns (8.1 MB) 21 to 36 ms where jinja.cpp took 6.4 to 51 s (GLM-5.3 the slowest).
  Known approximations, each pinned in `src/template_oracle_test.zig`: case mapping is ASCII-only, `wordcount`'s `\w`
  and `repr`'s isprintable past ASCII, `sameas` as identity of the engine's values, integers past 64 bits printed
  exactly but never computed with (`TemplateRuntime`).
- **Numbers print as Python prints them**: `tojson` and `{{ }}` write an integer exactly at any size and a float as
  Python's shortest repr (`1.0`, `1e-05`, `3.141592653589793`); jinja.cpp wrote `%g` (`1`, `3.14159`, 2^64 as
  `1.84467e+19`), so a replayed call's arguments carried other bytes than the client sent.
- **Control bytes**: ONE raw byte <0x20 in history once killed jinja.cpp's strict render → `fallbackFormatChat` on a
  `<|im_end|>` model (ChatML, the stop token kept), a named 500 `chat template unusable: <reason>` on any other.
  Everything through `appendJsonString`; wrong-family tags out ⇒ suspect silent fallback first. A NUL byte once
  truncated the rendered prompt (tell: the same `prompt=` count on consecutive turns); the engine returns a slice.
- **A parse or runtime gap in the engine is the same silent downgrade**: GLM-5.3's `m.content.0.output` (a number after
  a dot) failed to parse in jinja.cpp on EVERY tool message, so each agent turn after a tool result ran in the
  untrained generic format and broke prefix reuse (`x.0` is `x[0]`; `x.1.0` is two subscripts, never a float); the
  pack template is a committed fixture. Only a model whose turns end in `<|im_end|>` (Qwen, MiMo) has a fallback
  format, ChatML; GLM and DeepSeek answer a raise or a missing template with 500 `chat template unusable: <reason>` on
  every generative surface, streamed or not; the reason names the template's line and column. A fallback logs
  `jinja render failed (…)` at warn on every render and
  counts into `/props.template_fallbacks`: nonzero on a model that ships a template means a broken prompt, so grep
  that line first. A new template ships its fixture and a no-fallback case in the `format corpus` tool-traffic test.
  Cross-check suspect templates against Python Jinja2 on the same JSON before blaming the model.
- **The template oracle holds the engine to Python Jinja2**: `src/template_oracle_test.zig` captures every engine call
  `formatChat` makes (probes too) over a request corpus through the four served templates, as each pack serves them,
  and requires the Zig engine to give `src/fixtures/template_oracle.json` on every case (transformers' chat-template
  environment, written by `tests/dump_template_oracle.py`). A change to what formatChat hands the engine fails the corpus test until the fixture is
  regenerated (`SUSHI_TEMPLATE_ORACLE_DUMP`, then the dumper's `render`). The builtin table
  (`src/fixtures/template_builtins.json`, the dumper's `builtins`) holds one case or more per name the engine serves;
  a name without one fails by name.
- **A loop iteration is its own scope, as in Jinja2**: a body `set` (GLM-5.3's per-message `reasoning_content`)
  neither reaches the next iteration nor leaves the loop (nor does one in the loop's `else`); only a `namespace`
  carries state across. One shared scope
  rendered an earlier turn's thought into a reasoning-less assistant turn instead of `<think></think>`.
- **A `chat_template` value can be a POINTER** (`{% include 'chat_template.jinja' %}`): `chat.isIncludeStub` reads it
  as "no inline template" so the sidecar loads. Grep the log for `jinja` first.
- A template can raise on OUR extra-context values: `templateSettings` sniffs the family; tool-call `arguments`
  stay OBJECTS (a history call whose arguments are empty, null, an array, a scalar or malformed text embeds `{}`:
  `arguments|items` on anything else raises into the silent fallback; an object that repeats a key passes as sent, and
  the template reads its last value at the first key's place, as `json.loads` does); history tool_calls carry `"id"`; `noThinkTailSuffix` closes the block a template opens unconditionally (GLM-5.3's
  `<|assistant|><think>`) when thinking is off.
- **A system turn past index 0 renders where the template allows it**: a template that raises on it or drops it gets
  it folded into the leading system (`templateProbeRendersLateSystem`, every surface); MiMo's role loop keeps it in
  place, byte for byte. The fold rewrites every earlier byte of the prompt, so a Qwen3.8 template whose parsed tree
  holds the "System message must be at the beginning" branch (`{% if message.role == "system" %}` opening with an
  `{% if not loop.first %}` around only a `raise_exception`, at any spacing: `lateSystemTemplate` of the qwen4 chat
  face, fixture `qwen38_flash_next_chat_template.jinja`) gets ONLY that inner body replaced and renders the note in
  place; a template without it keeps the fold, and a `qwen4_exp` load says so in one `[load]` line. Only the face of
  the config's own `model_type` is asked (`ChatConfig.model_type`, from config.json): another arch's template, or one
  with no model type, keeps the fold whatever its shape. The adapter runs only when a late system turn exists, so a prompt without one is byte-identical; a second
  LEADING system turn is now a turn of its own too. `QWEN_MID_SYSTEM_MODEL_DIR=<pack> zig build test
  -Dtest-filter="real Qwen pack"` proves a loaded pack engages.
- **A surface never folds before the render**: a pre-fold moved Codex's mid-input `developer` turn and Claude Code's
  hook output to the front, so on MiMo turn N's prompt stopped being a prefix of turn N+1's. Responses' fresh
  `instructions` replace only the stored history's LEADING system turn.
- **A generic ChatML role header preserves tool roles**: absence of a literal `'tool'` branch does not license
  rewriting tool results as user text (`asks.toolRole`: a comparison against `'tool'`, else the probe).
- **`preserve_thinking` reaches only a template that reads it** (`templateSettings`), resolved per request:
  `chat_template_kwargs` > `--preserve-thinking` > model-settings.json > undefined (Qwen3.8 keeps every turn's
  thinking). Each value renders its own prompt, so the tokenize-cache key carries it.
- **A continuation is the template's own rendering of the partial turn**, cut after its text (transformers'
  `continue_final_message` rule): `<think>\n\n</think>\n\n` on Qwen3.8, `<think></think>` on MiMo, thinking on or off.
  The thinking-on prompt plus a bare closer gave `<think>\n</think>` and no block at all; it is only the fallback now.
- **Assistant-history reasoning round-trips** (`Message.reasoning_content`, OMITTED when absent). A contract COMMENT
  is read as a spec — pin it with a test.

## Parsing tool calls

- Nameless calls are discarded; XML function names reject empty placeholders and whitespace or tag fragments from prose.

- **A `<tool_call>` body carrying `<function=` is the XML dialect and is read FIRST** (qwen 3.5+ template mandates
  it); a parameter VALUE never decides the call. A `<parameter>` VALUE may spell the dialect's own close tags
  (`hermesValueEnd` = LAST `</parameter>` before the next opener). A Hermes value keeps its own whitespace
  (`stripHermesValueFraming`).
- **Framing is removed only where the family writes it**: a newline-framed `<function=>` body (Qwen3.8) loses one
  newline per value side, an unframed one (MiMo) keeps the value verbatim, so a written file keeps its last newline.
- **A JSON call cut INSIDE the object still names its tool** (`truncatedJsonCallName`): recover NAME + `{}`, NEVER ship
  partial values, never ship raw markup as content. Model-mangled arg JSON → `looseRepairToolCallJson`, never drop
  the whole call. A tag parser never bails on ONE missing delimiter.
- **Claude's `<function_calls><invoke name=…><parameter name=…>` is a call** when no other family matched
  (`BARE_INVOKE`; qwen4_exp writes it after Claude transcripts). It is
  `inferred`, so an undeclared name stays prose. The stream gate holds `<function_calls>` and `<invoke ` with every
  growing prefix (`invokeOpenerHoldsForTools`); the scrubber cuts it only beside a `<parameter ` or `</invoke>`
  (`bareInvokeMarkupCut`), and the parser reads it only beside one of them too (`parseInvokeToolCalls`), so prose
  naming the tag survives even when the name it shows is a declared tool. Corpus families `qwen4_exp`, `dsv41-dsml`.
- **Inferred `<invoke>` is the LAST tag family tried**: a native `<tool_call>` can carry invoke-shaped text in a value. An
  inferred opener with no `<parameter ` or close before the next opener is prose and never borrows the next call's body.
- **A lone `</｜DSML｜` closer is leaked markup**: the scrubber cuts at it and the stream gate holds `</` and every
  prefix of `</｜DSML｜`, while a completed `</div>` flushes. Corpus family `dsv4-dsml`.
- **An argument key never carries markup or a line break** (corpus invariant): a doubled `<arg_key><arg_key>K` keys on
  the LAST opener; a special token spelled where `<` belongs before `function=` right after `<tool_call>` is read as
  `<` (`hermesSpecialOpenerRepair`).
- **A `</think>` inside a tool ARGUMENT is payload** (`thinkCloseIsToolCallPayload`): decline a close whose nearest
  preceding tool opener is still OPEN AND whose block closes afterwards.
- **Types come from the SCHEMA, never the value's spelling** (`coerceToolArgsToSchema`; undecidable → untouched).
  A property's types resolve through `type` arrays, `anyOf`/`oneOf` (every branch), `allOf` (first typed branch) and
  a local `$ref` into `$defs`/`definitions` (depth-bounded); a value already fitting one candidate stays, else the
  candidates are tried in declaration order (`collectJsonTypes`).
  Buried required params hoist only on all-schema-read unanimity. A container string with a key repeated at the SAME
  value still coerces (`parseContainerAllowingRepeats`). Heuristic raw-JSON inference must name a DECLARED tool
  (`filterInferredBySchema`).

## tool_choice

- **One parser, every wire shape** (`tool_calls.parseToolChoice`): chat nests the name under `function`, Responses and
  Anthropic carry it flat, Anthropic spells required `any`; anything else is `auto`, which is unchanged.
- **`none` withholds the tools, skips the parser, and on chat completions bans the call opener**: a model that sees
  earlier calls in the history writes one anyway, as raw text in the answer. `server.noneCallBan` adds a -100 bias on
  the template's opener tag (`chat.callOpener`, `<tool_call>` on all three families) when it is one token, and leaves
  it unbanned when the tokenizer splits it (its first piece would be a plain `<`). Drafts stay on: it is a request bias.
  The chat page and `sushi run` send `tool_choice: "none"` on every request that offers no tools.
- **`required`/`any` and a named function are enforced at decode** (`generate.CallForce`, armed by
  `server.armCallForce`): the markup the template itself writes after a closed thought (`chat.forcedCallText`; Qwen3.8
  `\n\n<tool_call>\n<function=`, MiMo `<tool_call><function=`) is committed as tokens, through the name and its
  delimiter when one is named. For `required` the name is then one of the DECLARED names (each with its delimiter, as
  ids: a lone continuation is committed, a branch is a sample among its ids), since the model wrote `getTime` and
  `call: get_time` there. The arguments stay the model's; the parse chain and its valid-JSON invariant run unchanged.
- **They are also prompt-level**: the instruction closes the last user turn (or opens one after tool results), so every
  byte before the last message stays and the prefix cache still serves it; the generic fallback inlines it with the
  tools. A continuation keeps the client's prefill: no instruction, no force.
- **A forced call does not skip the thought**: it runs free (under any reasoning budget) and the call takes the
  position after the closer; a closed-thought prompt takes it at token 0. Where the template leaves the opener to the
  model (MiMo), the opener is committed at token 0, as a prompt-opened block would be. A thought that ends the turn
  (EOS, `length`, loop stop) ends it without a call.
- While the call is pending each successor is decided after its predecessor is known (no pipelined fast path), draft
  paths stay off for the request (`mtp_plan.draftsRefused`), and the batched tick leaves the slot serial (`forced_call`).
- A named choice for a function the request does not declare is a 400 on every surface. Guard:
  `tests/test_tool_choice_forced.sh`.

## Thinking

- Strip pos-0 unclosed openers; `trimTrailingThinkClosers`; unparsed tool markup never rides out as reasoning OR
  content (`trimLeakedToolMarkup`, ONE cut).
- Whether a prompt ends inside a think block is a property of the RENDERED BYTES (`promptOpensThink`), never ANDed
  with `enable_thinking`; `in_think_block` seeds from `prompt_opened_think` ALONE at every stream site; a model can
  open its OWN block (`modelThinkOpener`).
- **Streaming + tools + thinking**: buffer until pattern resolution; reasoning streams INCREMENTALLY on the tools
  path (`.hold_thinking` + `unstreamedReasoning`, never a resend); the think gate scans with a CURSOR (`ThinkScan`).
  `/v1/messages` streams a held thought only while its start cannot move (`heldThoughtSoFar`: prompt-opened or opener
  first); a thought reopened after visible text arrives whole at its close.
- **The tools-path buffer keeps whole text; a split reads only the undelivered tail** (`text_delivered`, both
  stream paths): the tool gate needs the turn from its start, so the buffer is never cleared on a text flush. A
  thought the model opens after visible text then split over the WHOLE buffer and shipped that text again as
  reasoning (the block read `Done\n<think>plan` after `Done` had streamed). The end-of-stream path already had this
  shape, since it rebuilds from the pending tokens only.
- **A per-token stream check scans only new bytes** (`ThinkScan`, `ToolHoldScan`, `ThoughtScan`: a marker overlap, not the
  reply); each has a whole-buffer twin kept as its oracle, and reset wherever its buffer starts anew (split, flush).
  Guards: the `incremental` tests in `think_tags.zig`, the format corpus and the tool-traffic replay.
- Thinking-off is enforced in the PROMPT; generated reasoning is ALWAYS delivered (every site splits via
  `splitThinkBlock(text, true, …)`).
- **A thought is decided at its first byte when no opener can start it** (`think_tags.thinkOpenerPossible`: every opener
  starts with `<`); otherwise the three stream sites wait for 7 bytes. On the
  surface budget path (`armThinkBound` declined) the budget then counts from that earlier consumption.
- **An open thought streams only what its closed split delivers** (`trim(thought, "\n ")`): a trailing `"\n "` run and
  a close tag still arriving wait (`think_tags.settledReasoning` on the tools path, `think_tags.openThoughtFlush` /
  `closedThoughtDelta` on raw flushes). Streaming the newline before `</think>` made stream and non-stream differ.
- **A think tag inside an inline code span is text** (`openCodeSpanStart`, read by `indexOfThinkCloseTag` and
  `indexOfThinkOpenTag`, the ONE scan every surface and `normalizeEmbeddedThinkBlocks` use): a model explaining
  its format writes `` `<think>x</think>` `` into a thought or an answer. The verdict reads only the bytes BEFORE the
  tag, so a stream never revises it; `openThoughtFlush` therefore keeps an open span in the buffer. A span ends with
  its line and three backticks are a fence. Tags quoted any other way (plain or double quotes) are still structure.
- **A thought the length limit cuts ends on what its split delivers** (`think_tags.cutThoughtDelta`): a lone opener is
  structure, and a thinking block or reasoning item opens at its first delta, so an empty thought streams none.
- **A token that opens an assistant turn ends the reply** (`ModelConfig.turn_opener`, merged into the eos ids at load
  by `mergeTokenizerStops`): the GLM pack answers, then writes `<|assistant|>` and a second short thought and answer,
  which rode out as content with an orphan `</think>`. The cause is the checkpoint, on serial and DFlash2 alike; the stop
  is the engine's boundary. A KLD run keeps the config's own eos set.
- **A closer in content is dropped wherever it lands, on every surface**: a stream drops the marker token
  (`isChannelMarkerToken`), so `normalizeEmbeddedThinkBlocks` cuts closers out of the content between and after blocks
  (`contentWithoutClosers`; a closer inside a code span is text) and `trimTrailingThinkClosers` keeps the tail.
  `<think>t</think>391</think>391` is content `391391` stream and non-stream.
- **Every stream's content arm is `think_tags.ContentStream`**: a think tag token inside an open code span is text
  (`CodeSpan`, the `openCodeSpanStart` reading kept as bytes arrive); a `<think>` outside one re-opens a thought,
  which streams as reasoning joined by `\n\n` (`ThoughtJoin`) while the content parts around it are trimmed and
  joined by `\n`, as `normalizeEmbeddedThinkBlocks` merges them. Whitespace waits only as whole tokens.
  A re-opened thought cut by the end of the reply is reasoning on both sides. Anthropic gives a thought after
  text its own thinking block (a stopped block takes no deltas); Responses keeps one reasoning item, listed after
  the message when the stream opened it there (`responsesReasoningFirst`).
- **A live row never asserts that a model thinks**: a thought the model closes empty (`</think>42`) has no reasoning
  and no thinking block, on every surface. Rows 4 and 20 of `tests/test_format_matrix.sh` branch on an
  independent non-stream turn: a stream's billed reasoning tokens come from what it delivered, so they prove nothing.

## Loop stops

A short exact cycle convicts on SPAN (`degenerate.degenerate_loop_min_span` 128; a 24-wide map row is legit), near-repeat needs
THREE low ratios incl. PROGRESS (1024-token window, `degenerate.near_repeat_min_span` 4096), long-period tier 9..64 at 10 reps.
Cuts are intentional stops: `finish_reason "stop"`, `finish_details:{"type":"repetition_loop"}`, `[loop-stop]`
logged, non-streaming trimmed to the span start. Guard: `tests/test_loop_stop_signal.sh`.
The decode loop asks every tick through `degenerate.TailGuard`, which counts the window's distinct tokens as tokens arrive and runs the
full near-repeat judgement only when that vocabulary ratio could convict; `degenerateTail` is its oracle and must give the same verdict.
- **A loop convicted inside an open thought closes the thought instead of ending the turn** (`closeThoughtOnLoop`): a `stop` with no answer or call reads as a finished agent turn. The reasoning budget's early-stop line and closer are committed once per request (`[loop-recover]`) and the model answers; an answer-side loop, a second loop or no room before `max_tokens` keeps the cut. The line is built when the loop is convicted (`server.loopCloseBound`), never armed at request start: a request carrying a `think_bound` is refused DFlash2 (`glmDflashEligible`). The forced tokens ride `commitThoughtClose` like the budget's, so stream and non-stream see the same bytes.
