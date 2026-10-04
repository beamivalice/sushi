# Changelog

sushi began as a fork of [mlx-serve](https://github.com/ddalcu/mlx-serve) and was detached from it on 2026-09-17 at
mlx-serve commit `ef5e667` (two commits after mlx-serve v26.9.4). This file covers sushi's own changes since then;
earlier history is mlx-serve's, in that project's changelog.

## Unreleased

- **`--mtp` works with SSD-streamed Qwen3.8-Flash-Next packs**: the head and its own routed experts load resident and
  are billed in the budget, verify rows run on the streamed decode path, and the n-gram window rolls back after a
  partially accepted round, so a greedy reply equals the resident and the serial one. The default stays off.
- **`scripts/build-mlx.sh` refuses MLX submodules left at an older pin** and prints the `git submodule update` command,
  instead of failing to compile mlx-c after a plain `git pull`.
- **GLM-5.3 now stores its MLA latent at 8 bits by default**, like every model: 6,688 instead of 11,968 bytes per
  token. `--kv-quant 16` (or a request's `kv_quant: 16`) keeps it BF16; `16` is accepted for every model.
- **GLM-5.3-Flash streams its experts from SSD** under `--ssd-budget-gb`/`--expert-cache-gb`, for Sushi packs and the
  BF16 source checkpoint alike, in `serve`, `run` and `kld compare`; streamed output is identical to resident, text
  only, without the DFlash2 assistant.
- **`sushi kld` plans a streamed load exactly as `serve` does**: rate-group packs are refused by name and a streamed
  MiMo bills its coarse lm_head.
- **A streamed Qwen's load admission counts its QSA history** beside the KV cache, as the context sizer already did.
- **GLM's first load prepares its DFlash2 assistant as A4/group-64** (was A6/group-128), 0.24 GB smaller with identical
  output; an existing A6 cache is rebuilt once.
- **`--wired-margin-gib` defaults to 4 GiB (was 8)**, so a raised `iogpu.wired_limit_mb` admits 4 GiB more weights
  and context.
- **MiMo takes every thinking effort word** (`minimal` through `max`) as thinking on instead of answering 400, on
  chat, Responses and Anthropic requests, `--think` and the REPL's `/think` alike; `off` and `none` still turn
  thinking off.
- **`--think` binds only the `--model` model**: another model loaded on demand that lacks the word now keeps its own
  default instead of failing to load.
- **`sushi launch pi`, `omp` and `zcode` give MiMo thinking on** for every thinking level they offer.
- **The chat page's effort menu lists exactly the selected model's words and starts on its real default**, which
  `/v1/models` now reports as `default_reasoning_effort`; every message sends the shown effort.

- **`sushi launch omp` lets local buffered tool calls finish without the default five-minute retry**; the generated
  Sushi provider disables its model-progress deadline while preserving explicit user timeout overrides.

- **Long-context MiMo requests switch to serial decode when their MTP rounds lose on measured cost**, while
  workloads that benefit from speculation keep it.
- **Experimental token logit biases**: load scoped penalties and rewards from JSON/CSV with `--logit-bias-file`,
  or send an OpenAI `logit_bias` map per request; the optional think-penalty preset remains off by default.

- **Faster MiMo 2.3bpw**: the prefill's expert GEMMs, the MTP verify rows (whose global layers share one walk of a
  long cache), the draft heads and long-prompt attention do less work per token; output is byte-identical.
- **Faster Qwen3.8-Flash-Next prefill**: the expert routing table is built on the GPU and the expert outputs are
  reduced in place, with no host round trip or un-sort copy per layer; output is byte-identical.
- **MiMo echoes and edits of a file in context decode faster**: prompt lookup now runs inside MiMo's MTP rounds, and a
  lookup or PLD round verifies up to seven drafts; output is byte-identical.
- **`sushi run mimo-v2.6-flash`** (and `sushi pull`) fetches and serves the MiMo 2.3bpw pack by its short name.
- **Greedy MiMo decode reads the vocabulary head through a coarse top-32 shortlist** re-scored on the full head,
  with or without MTP, and on a streamed MiMo too; sampled, logprob, penalty and grammar requests keep the full head.
- **`--gpu-warm-secs <n>`** (default 60, 0 = off): the server keeps the GPU awake for this long after a request, so
  the next request no longer starts with a GPU wake-up delay.
- **With MTP, the first token streams when prefill ends** instead of after the first speculative round (when that
  token is visible: a template-opened thought or thinking off).
- **Faster SSD streaming**: a streamed decode layer queues its experts from a GPU copy of the expert cache map
  before the host reads the router ids, and verifies them one layer later; Sushi-2bpw at a 20 GB budget on an M1 Max
  decodes 10.3 -> 19.6 tok/s (llmprobe, 512 context) with byte-identical output.
- **`--expert-pick-tolerance <n>`** (0 to 0.6, default off): a streamed pack may serve a cached expert in place of a
  missed one when the router rates it at least `1 - n` as likely; lossy, trading a little accuracy for fewer SSD reads (KLD in `docs/quality-kld.md`; 22.3 tok/s greedy at 0.2 on the same
  M1 Max); on a streamed MiMo it compares the sigmoid router's probabilities (5.8 -> 10.5 tok/s at 0.2 with a 60 GB
  budget on an M5 Max).
- **A reply cut short while thinking streams the reasoning the non-streamed reply returns**: no lone `<think>` as
  reasoning (MiMo at `max_tokens: 1`), no empty Anthropic thinking block, and no dropped thought of a few characters.
- **The MiMo pack loads on demand on a 128 GB Mac at default flags** (the app's path): the automatic resident-memory
  cap now limits only models sharing memory, and a model loading alone is judged by the load's own memory check; an
  explicit `--max-resident-mem` still applies.
- **`presence_penalty`, `frequency_penalty` and `repeat_penalty` take effect**: the default decode path and MTP
  ignored them; a penalised request now decodes serially, so it is slower than an unpenalised one.
- **A request that names a pack by its path is answered by that pack or refused**: an unregistered path is a 404
  instead of an answer from the default model; unknown model names still fall back to the default.
- **A new MiMo session that shares only another's system prompt and tools reuses them from the prefix cache**
  instead of prefilling them again (a subagent, a second chat).
- **MiMo prefills long prompts in 2048-token chunks by default** (`--prefill-chunk 4096` raises it back), and the
  load line now says each request picks its own chunk width, with the load-time width only a fallback.
- **`ignore_eos: true`** on a `/v1/completions` request decodes past the end-of-sequence token up to `max_tokens`,
  as in vLLM; a chat request that sets it gets a 400 naming the field.
- **A reasoning budget closes the thought on time when several requests decode at once**; batched, a thought could
  run thousands of tokens past it.
- **A Flash-Next turn restored from the SSD prompt cache is no longer billed as if its restored prefix were new**,
  so a long session after a restart is admitted where that bill refused it.
- **A malformed `config.json` is refused by name** (a wrong field type, a negative or oversized number) instead of
  loading undefined values; model discovery skips one whose top level is not an object.
- **Concurrent MiMo requests decode together** in one forward of up to four streams; crowded MTP requests retain
  their head state and resume solo rounds, with the same output as each alone.

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
