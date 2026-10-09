# Changelog

sushi began as a fork of [mlx-serve](https://github.com/ddalcu/mlx-serve) and was detached from it on 2026-09-17 at
mlx-serve commit `ef5e667` (two commits after mlx-serve v26.9.4). This file covers sushi's own changes since then;
earlier history is mlx-serve's, in that project's changelog.

## Unreleased

- **Cleanup**: code for architectures sushi does not serve is removed, including the dormant ANE prefill path and its
  `--ane-prefill` flag, the `sushi:ane_*` metrics and the `ane` object in `/props`; unsupported models are still
  refused by name.
- **Memory**: the load check bills what each model allocates plus a 1 GiB safety net instead of a flat 7 GiB, the
  auto context keeps 93% of the memory ceiling on every model, and `--wired-margin-gib` defaults to 1 GiB.
- **GLM-5.3-Flash**: DFlash2 drafting is about 6% faster per round and verification about 1% faster, with identical
  output.
- **EXL3 packs**: expert rates from 1.5 bits per weight take the fast decode and prefill kernels on every served
  architecture, including GLM-5.3-Flash's, with output identical to the reference readers.
- **Structured output**: a `response_format` schema property that names no type (a bare `description`, `true` or `{}`) now accepts any JSON object instead of only `{}`.
- **Structured output**: `response_format` schemas may use `$ref` to `#`, `#/$defs/Name` or `#/definitions/Name`, recursive refs included, so nested Pydantic or Zod models are enforced instead of collapsing to `{}`.
- **Tool calls**: a `<function=name` whose closing `>` the model dropped is recovered with its parameters instead of the call being lost as plain text.

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
