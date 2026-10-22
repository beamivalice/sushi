# Changelog

sushi began as a fork of [mlx-serve](https://github.com/ddalcu/mlx-serve) and was detached from it on 2026-09-17 at
mlx-serve commit `ef5e667` (two commits after mlx-serve v26.9.4). This file covers sushi's own changes since then;
earlier history is mlx-serve's, in that project's changelog.

## Unreleased

- **After a Metal out-of-memory failure the next request no longer fails at once**: the failed request is reported as the out-of-memory it was, and the error no longer carries over to the request after it. Thanks to [@holistech](https://github.com/holistech) for the report (#61).
- **The memory bill counts GPU memory other apps hold and the size of MLX's command buffers**: on a Mac with a raised GPU memory limit, admission plans under the limit less what other apps hold on the GPU, and a larger `MLX_MAX_MB_PER_BUFFER` is billed, so a prompt that would hit Metal's "Insufficient Memory" mid-prefill is refused up front instead. Startup logs both. Thanks to [@holistech](https://github.com/holistech) for the report (#61).
- **Qwen3.8-Flash-Next verifies 2 to 8 rows at an even price per row**: MTP and DFlash2 rounds of five rows and wider no longer pay a step at the fifth and seventh row, an eight-row verify is about 16% faster, and the output stays byte-identical to serial decoding. DFlash2 prices its rows from the measured cost, so it verifies more rows where drafts pay.
- **Qwen with DFlash2 reserves the memory it uses**: the MTP head's cache is no longer reserved for requests DFlash2 serves and the drafter counts at its loaded size, so about 2.8 GB more fits at 1M context, while a model swap now reserves the drafter's per-request memory it used to miss.
- **DFlash2 speeds up image, video and sampled requests on Qwen3.8-Flash-Next**: a request with images now decodes with the drafter (about 1.7-2x plain decoding, output unchanged) instead of falling back to plain decoding, and the speed gate judges a request on its steady rounds against today's verify cost, so it no longer stops DFlash2 where it is winning.
- **Adaptive DFlash2 for GLM-5.3-Flash**: GLM verifies 2 to 8 draft rows per round as the drafter's confidence pays, and a request whose drafts stop paying falls back to plain decoding and retries later (`--drafter-min-depth` / `--drafter-max-depth`).

## v1.3.0 — Qwen EXL3 Quantizer, Adaptive DFlash2, KV6, Dashboard

- **Qwen3.8-Flash-Next EXL3 quantizer**: `sushi convert exl3` writes a Sushi pack from the BF16 checkpoint at the rate you pick, with the trunk in the format the packs serve; runs resume where they stopped, keep finished layers and packs, and fit their Metal threadgroups to any Apple GPU. Check the result with `sushi kld`. Thanks to [@cosmic-quokka](https://github.com/cosmic-quokka) for the M2 Pro report (#57).
- **Adaptive DFlash2 for Qwen3.8-Flash-Next**: point `--drafter` at the DFlash2 drafter and Qwen verifies 2 to 8 draft rows per round as the drafter's confidence pays, up to 20% faster than MTP on Sushi-2.6bpw and level with it on Sushi-4bpw (`--drafter-min-depth` / `--drafter-max-depth`); MTP stays the default. Thanks to [@1RAFAL](https://github.com/1RAFAL) for the request (#59).
- **`--kv-quant 6` on all three models**: a 6-bit KV cache between kv4 and kv8 for Qwen, MiMo and GLM, also as `kv_quant` 6 in `model-settings.json` and per request; GLM stores its MLA latent at 6 bits and keeps DFlash2 and prefix reuse.
- **Web UI Dashboard**: a Chat | Dashboard switch in the top bar with live throughput, requests, GPU and memory charts plus sessions and models tables (needs `--metrics`); switching never interrupts a chat. Thanks to [@cosmic-quokka](https://github.com/cosmic-quokka) for the dashboard (#55).
- **Requests that used to fall back now run as asked**: MTP stays on with `presence_penalty` and `repeat_penalty`; Qwen serves `high` and `max` effort as `xhigh`, GLM serves `medium` as `high` and `xhigh` as `max`, each with a notice in the log and the response.

## v1.2.4 — Faster Qwen3.8-Flash-Next, more context admission, image cache

- **Qwen3.8-Flash-Next Sushi-2.6bpw decodes faster with the same output**: its expert kernels run about 10% faster
  per layer, about 2-4% per forward when MTP verifies drafts.
- **Twice the context on a 32 GB Mac**: Qwen3.8-Flash-Next streamed from the SSD stops billing memory it never
  uses, so Sushi-2bpw on a 32 GB Mac advertises 261k tokens instead of 140k and serves a 250k-token prompt, and a
  context that loads is one its own requests admit.
- **An interrupted `sushi pull` resumes instead of reporting the pack downloaded**: `sushi pull` now checks every file
  against the pack's listing, which also finishes a folder an earlier version left short, and `sushi list` no longer
  shows a pack as downloaded while a pull of it is unfinished.
- **Qwen3.8-Flash-Next decodes long prompts faster with the same output**: decode without MTP runs about 5% faster
  from 4k to 32k tokens of context.
- **Image conversations keep their prompt cache across a restart**: an SSD prefix-cache record now carries the images
  it was computed under, so a text -> image -> text conversation restores past its images instead of re-reading
  everything after the first one, and a changed image still invalidates only what follows it.
- **A thinking budget in the web UI**: a budget box beside the effort menu shows the chosen effort's thinking-token
  cap and takes a typed value for the request.
- **Quoted control tokens in a message no longer end the turn**: `<think>`, `</think>` or `<|im_end|>` written as text
  in a user, system or tool message (a source file Claude Code read) stays text instead of reading as a real boundary.

## v1.2.3 — Faster GLM, 32-bit Qwen GDN state

- **GLM-5.3-Flash is faster with the same output**: decode without DFlash2 runs about 9% faster and prompt processing
  about 5% faster, token for token identical to before.
- **Qwen3.8-Flash-Next forgets at its true rate**: the linear-attention decay gate and recurrent state are now f32,
  which a bf16 gate rounded to "never forget" for the slowest heads. A cached checkpoint grows by 54 MB and four-stream
  throughput drops about 2%, so `--qwen-gdn 16` keeps the bf16 state (the gate stays f32) to save memory on smaller
  Macs, and the 32 GB, 48 GB and 64 GB launch lines use it. `--qwen-gdn` now takes `32` or `16`: drop a v1.2.2
  `--qwen-gdn fp32`, which is the default now. The Qwen SSD prefix cache starts empty on the first boot after this
  change, since older entries hold a bf16 gate.
- **64 GB Macs boot the recommended context again**: the load check no longer asks for 1 GB more than the requested
  context needs, which could refuse the documented 64 GB launch (`--ctx-size 250000`) in v1.2.2.
- **GLM keeps its speed in `sushi run`**: DFlash2 now stays on for requests that carry `logit_bias` or
  `tool_choice: "none"`, which every `sushi run` turn sends, so thinking answers there decode about 40% faster.
- **Less memory per request with the SSD prefix cache**: Qwen and GLM write each prefill checkpoint to the SSD as they
  go and keep only the newest two in memory, where they held up to 16 and 9. A long Qwen request bills about 1.7 GiB
  less (0.9 GiB at `--qwen-gdn 16`) and a GLM one about 1.3 GB less, so a 48 GB Mac now reaches 200k tokens on
  Sushi-2bpw (was 160k) and a 64 GB Mac 528k on Sushi-2.6bpw (was 480k). With a RAM prefix cache beside the SSD,
  Qwen, GLM and MiMo keep only an entry's newest two restore points in RAM. Restores from the SSD give the same output,
  and a long GLM prompt sent again restores from its very end instead of re-reading its last 2k tokens.
- **Small Macs get the context they advertise**: Qwen's auto context is now the largest one its request admission
  accepts, so a prompt within the advertised context is no longer refused on a tight Mac, and the load check honors a
  raised `iogpu.wired_limit_mb` instead of refusing a model that limit fits.

---

## v1.2.2 — GLM on 96 GB, faster loads, built-in file editing

- **GLM-5.3-Flash on 96 GB Macs**: the new Sushi-2bpw pack serves GLM-5.3-Flash with DFlash2 and images on a 96 GB
  Mac, with a 512K context at 8-bit KV (300K at BF16). `sushi list` shows every Sushi pack with its size, whether this
  Mac fits it and which one to pick; `sushi pull` warns before pulling a pack that will need `--ssd-budget-gb`.
- **Faster loads**: weights load without filling the macOS file cache, so a large load no longer compresses or swaps
  the weights already in memory. GLM-5.3-Flash Sushi-2.4bpw is serving in 12 s instead of 22 s and MiMo-V2.6-Flash
  in 15 s instead of 26 s, with about 57 GB less memory compression each. Thanks @ViRb3.
- **Built-in file editing**: the chat page and `sushi run` can let the model create and edit files inside the folder
  you choose, switched per conversation with the Edit chip or `/edit on|off`; every write stays inside that folder.
  Thanks @lukadisanto for asking for it.
- **Agents and APIs**: Claude Code no longer counts cached context twice and compacts at half its real size;
  `/v1/messages` streams thinking live when tools are on; `tool_choice: "none"` stops stray tool calls; structured
  output follows `$ref`, so nested Pydantic and Zod models are enforced; a repetition loop inside a thought now ends
  the thought and lets the model answer; chat responses report their reasoning tokens; `sushi launch opencode` works
  with OpenCode builds without `--standalone`, and its `--think <effort>` is checked against what the model advertises,
  with `--persist` merging it into your OpenCode config behind a backup. Thanks @felk-dev.
- **Speed**: GLM-5.3-Flash drafts about 6% faster per DFlash2 round, EXL3 prefill no longer waits on the GPU to build
  host-side expert tables, and expert rates from 1.5 bits per weight take the fast kernels, all with identical output.
  On a slower GPU, a GLM DFlash2 commit no longer copies its whole reserved cache when it should append in place.
  Thanks @davidtai.
- **Memory and reliability**: the load check bills what each model really allocates plus 1 GiB instead of a flat
  7 GiB, and the available-memory figure no longer counts a resident model twice, so loads that fit are no longer
  refused. A stalled client is dropped after 30 seconds, model-load races are fixed, and a broken SSD cache entry is
  dropped instead of failing every later match.
- **Leaner engine**: 12% less source code, a 12% faster release build and a test build in half the time. `--help` lists every flag, and a bad value or stray argument stops before
  anything loads. `--fast`, prompt lookup decoding (`--pld*`), `--expert-cache-gb` (use `--ssd-budget-gb`),
  `--logit-bias-file`, `--embedding-max-length`, `--kv-attn-mode`, `--decode-attn-quant` and other flags with no real
  choice are gone; `--no-prefix-cache-ram` is now `--prefix-cache-mem off`. `sushi pull` downloads only from the
  `beamster` org, and over a hundred internal environment switches are removed. New: `--qwen-gdn fp32` keeps Qwen's
  recurrent state in f32.

---

## v1.2.1 — Safer prompt cache and server hardening

- **Prompt cache**: two sushi processes on the same model no longer share one SSD cache folder, which could return
  another session's answer; a restart on a nearly full disk keeps the cache, and loading a second model no longer
  deletes the first one's entries.
- **Crash fixes**: an empty `/v1/completions` prompt, deeply nested JSON and oversized WebSocket messages are refused
  with a 400 instead of stopping the server.
- **GLM-5.3-Flash**: after a client disconnects, the next request on a streamed GLM no longer fails; `sushi pull`
  now fetches the DFlash2 assistant, and deleting its BF16 source keeps DFlash2 on.
- **Tools and structured output**: earlier tool calls with empty or non-object arguments render in the model's own
  format, `anyOf`/`oneOf`/`$ref` parameters get their real types, and JSON-schema output stops cleanly on numbers and
  bounded arrays; `logit_bias: null` is accepted again.
- **CLI**: `sushi launch` quotes model names, `sushi serve` without `--model` honours the sampling flags, `update`
  no longer logs the API key, and `sushi run` accepts long pasted lines and filters terminal escapes from model output.
- **mlx-serve**: the guest manifest now lists GLM-5.3-Flash.

---

## v1.2.0 — GLM-5.3-Flash, 32 GB streaming, zero-RAM prompt cache

- **GLM-5.3-Flash**: the new Sushi-2.4bpw pack runs on 128 GB Macs (KLD 0.074 against the BF16 model) with image and
  video input, the full 1M context and DFlash2 drafting: 42–55 tok/s decode on an M5 Max. Up to four requests decode
  together.
- **Streaming with MTP**: `--ssd-budget-gb` streams any model's experts from the SSD, so Qwen3.8 Sushi-2bpw runs on a
  32 GB Mac at about 20 tok/s.
- **Prompt cache with 0 GB of RAM**: prompts are reused across turns from an SSD cache that sizes itself (up to 20 GB);
  keeping them in RAM is now opt-in with `--prefix-cache-mem`.
- **Faster**: MiMo decodes about 12% and prefills about 15% faster, Qwen3.8-Flash-Next prefills faster, and concurrent
  requests decode together on every model.
- **Agents and API**: `sushi launch grok` is new and opencode 2.x works again; streamed and non-streamed answers match
  byte for byte; penalties, `repetition_penalty` and `ignore_eos` work; GLM history renders exactly as its template.
- **Flags**: `--mtp-min-depth`/`--mtp-max-depth` replace `--mtp-depth`; new `--no-mtp-lookup` and `--gpu-warm-secs`;
  `--wired-margin-gib` defaults to 4 GiB.
- **Reliability**: long GLM sessions, memory pressure and restarts are handled cleanly, and an idle streamed server no
  longer uses CPU.

Thanks @cnsiva (request-budget defaults, `repetition_penalty`), @jasontitus (unique response IDs, the M1–M4 GLM fix,
portable GLM tests), @ShoichiTect (idle SSD-read workers) and @gomezvd (MTP on streamed Qwen).

---

## v1.1.1 — Long MiMo prompts and agent sessions

- **MiMo long prompts are admitted again**: a resident MiMo server now sizes requests against the GPU limit you set
  (`iogpu.wired_limit_mb`), not against what other apps happen to leave free, so a long agent session no longer gets
  "requires ~N MB GPU memory" on a 768k server; the prompt cache gives its memory back to a request that needs it.
- **`--prefill-chunk` is a maximum**: a request that does not fit at your chunk steps down to a narrower one instead
  of being refused. 2048 is the recommended value; wider chunks cost memory without prefilling faster.
- **Warm agent turns stop spiking memory**: a turn that reuses the cached conversation grows its KV buffers during the
  prefill, one layer at a time, instead of all at once on the first reply token, so long agent sessions stay admitted.
- **Claude Code on a local model**: `sushi launch claude` keeps each turn on one streamed request, and a request whose
  client disconnects now stops generating instead of running on for nobody.
- **Homebrew gets each release right away**: `brew upgrade sushi` sees a new version as soon as it is published.

---

## v1.1.0 — MiMo-V2.6-Flash and SSD streaming

- **MiMo-V2.6-Flash**: sushi's second model. `MiMo-V2.6-Flash-Sushi-2.3bpw` serves text and image input from one
  resident pack with native MTP, up to its full 1M-token context on a 128 GB Mac.
- **Sushi packs stream from SSD**: `--ssd-budget-gb N` keeps the trunk resident and streams the routed experts from
  SSD, so a Mac with less memory than the pack can serve it; replies are identical to a resident load. Every Sushi
  Qwen pack and MiMo-V2.6-Flash-Sushi-2.3bpw stream, and Sushi-2bpw serves on a 32 GB M1 Max at a 20 GB budget.
- **Faster Flash-Next**: on an M5 Max, Sushi-4bpw decodes 8% faster (83 -> 89 tok/s) and prefills a 10k-token prompt
  19% faster (1,796 -> 2,139 tok/s). oMLX's tensor-unit sparse attention now serves prefill from the first sparse
  chunk, GDN prefill runs a software-pipelined recurrence, batched decode overlaps GPU work with graph building, and
  `--prefill-decode-share` keeps decoders moving while a long prompt prefills. Thanks @STRML and @cowboycoderhq.
- **Better and smaller packs**: new expert weights for Sushi-4bpw (KLD 0.0632 -> 0.0592) and Sushi-3bpw
  (0.1047 -> 0.1036), and Sushi-2bpw for 48 GB Macs.
- **Live sessions on `/metrics.json`**: every in-flight request and every cached conversation, with its phase,
  context against the model's limit and the GPU memory its KV holds. Thanks @yoyo930021 and @ddalcu.
- **Updates itself**: `sushi update` installs the newest release after checking its SHA-256 and signature and keeps
  the old one for `--rollback`; a daily check, one-click update from the chat page, or `brew install
  beamivalice/tap/sushi`.
- **Fixes**: MiMo long-context memory returns to the OS, SSD prompt-cache restores keep one copy, JSON-constrained
  logprobs pair with their tokens, and a failed model load can be retried after a rescan. Thanks @brandondyal and
  @jasontitus; the EXL3 engine is now a module mlx-serve builds against, thanks @ddalcu.

---

## v1.0.5 — Sushi-2.6bpw at full speed

- **Sushi-2.6bpw for 64 GB Macs, as fast as Sushi-3bpw**: the new Qwen3.8-Flash-Next pack carries 43.95 GiB of
  weights, 5.4 GiB less than Sushi-3bpw, so a 64 GB Mac serves 250k tokens of context at 8-bit KV. Its experts now run
  on the same fast kernels as Sushi-3bpw's: 18% faster decode and 12% faster prompts on an M5 Max, output unchanged.
- **Faster Flash-Next decoding**: when a reply copies text already in the conversation (a file returned with an edit,
  a tool call carrying a file), speculative decoding drafts from that text, 16-21% faster file edits and 9-11% faster
  file writes; linear attention, hyper-connections and the sparse-attention indexer also take fewer GPU dispatches.
  Output is unchanged; several of these are ported from mlx-serve, thanks @STRML.
- **Long conversations stay cached**: by default the RAM prompt cache holds a whole conversation where memory allows,
  a session longer than the cache keeps the longest prefix that fits, and sessions past about 250k tokens restore
  from the SSD cache with half the memory. Ported in part from mlx-serve, thanks @STRML, @brandondyal and
  @celestial-rose.
- **Forced tool calls that work**: `tool_choice` `required` (Anthropic `any`) or a named function now makes
  Qwen3.8-Flash-Next call one of the declared tools on every API, after its thinking; naming an undeclared function
  is a 400. Streamed and non-streamed thinking are now the same text, and a continued reply resumes after its
  closed think block.
- **`--preserve-thinking on|off`**: keep every turn's thinking in the prompt (the default) or only the latest turn's,
  per model in model-settings.json or per request with `chat_template_kwargs.preserve_thinking`.
- **`sushi run qwen3.8-flash-next`**: `sushi run` and `sushi pull` name the Sushi packs (`:2.6bpw`, `:4bpw`, 3bpw by
  default) instead of models sushi does not serve.
- **Thanks, @jasontitus**: for this release's faster MTP verification on Flash-Next (hyper-connection weights read
  once per group of draft tokens), the fix that stops long prompts being refused while the GPU finishes earlier work,
  and the build-from-source docs, and, belatedly, for v1.0.4's 4x faster prompts on M1–M4 Macs and parallel n-gram
  reads on 64 GB Macs.

---

## v1.0.4 — Faster prompts and browser chat

- **Faster prompts on M1–M4**: Macs without the M5's neural accelerators read prompts about 4x faster on the EXL3
  packs (M2 Max, 3–4k-token prompts, default settings).
- **Faster Flash-Next prompts on 64 GB Macs**: when the n-gram table cannot stay in memory beside the model, prompt
  processing reads it in parallel by default instead of one row at a time.
- **Smaller contexts need less free memory to load**: resident Flash-Next EXL3 packs without separate sidecars or
  ANE now size load headroom from the chosen context instead of always asking for 7 GB above the weights.
- **Chat in your browser**: `sushi serve` and `sushi run` serve a chat page at `http://127.0.0.1:12345/` that streams
  replies, shows the model's thinking, takes images for vision models and keeps your conversations in the browser.
- **`/cd <folder>` in `sushi run`**: moves the folder the file tools and relative `/image` paths read from, the prompt
  always shows that folder and whether tools are on, and a model that asks for a file outside it now suggests `/cd`.

---

## v1.0.3 — Hotfixes

- **Prompt cache**: re-packing a model in place no longer restores stale SSD cache entries, and the RAM cache stays
  within its cap when every entry is in use.
- **Video and labels**: multi-part videos that fit are no longer refused, and `/v1/models` names an EXL3 pack's
  expert rate beside its dense width.

---

## v1.0.2 — Hotfixes

- **Stability**: two cached conversations can no longer share a prompt-cache key, a failed long-context cache copy no
  longer frees memory twice, and very low temperatures behave the same with and without MTP.
- **Memory and loading**: two resident models no longer over-commit memory at admission, and packs with invalid EXL3
  rate stamps are refused by name.

---

## v1.0.1 — Hotfixes

- **Tool calls and streaming**: streamed tool calls are always valid JSON, the reasoning budget applies to tool replies
  and Anthropic streams, and a disconnected Responses request is no longer stored as completed.
- **Prefix cache and loading**: fixes for SSD cache restores and failed cache writes, and malformed pack configs are
  refused by name.

---

## v1.0.0 — Qwen3.8-Flash-Next, sushi-packed

![Sushi-3bpw decode and prefill from 4k to 1M tokens on an M5 Max](https://raw.githubusercontent.com/beamivalice/sushi/main/docs/assets/perf-sushi3bpw-1m.png)

- **Two Qwen3.8-Flash-Next packs, EXL3 experts**: Sushi-3bpw (49.3 GiB, for 64 GB Macs) and Sushi-4bpw (63.7 GiB, for
  96 GB and up). At about 50 GiB, Sushi-3bpw has half the KLD of mlx-serve's iQ-MLX 3.3bpw; Sushi-4bpw matches oMLX
  oQ5e's quality in 20 GiB less memory.
- **Up to 1M tokens of context on one Mac**: 94 tok/s decode and about 1,900 tok/s prefill on an M5 Max, still 57 tok/s
  at 1M, with the 8-bit KV cache and the model's own MTP draft head on by default. `--mtp-typical 0.2` makes sampled
  decoding 15-20% faster.
- **Images in every API**: OpenAI chat and Responses, Anthropic messages and tool results all carry images to the
  vision tower, each where it was sent, and a large image needs about half the memory it did.
- **A drop-in local server**: OpenAI- and Anthropic-compatible HTTP on `127.0.0.1:12345`, clear of mlx-serve's 11234.
  `sushi run` chats in the terminal with read-only web and file tools, `sushi launch` sets up Claude Code, pi, omp,
  opencode and codex, and one thinking-effort vocabulary (off to max) works across every API.
- **Memory you can plan**: a load that would not fit is refused by name, concurrent long prompts wait instead of
  crashing, an unload answers once its memory is free, and `/v1/models` reports the real resident size.
- **Install with curl**: one ad-hoc signed binary for Apple Silicon on macOS 26.2 or later, with no Python at serve time.

---
