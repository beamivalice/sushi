# Changelog

sushi began as a fork of [mlx-serve](https://github.com/ddalcu/mlx-serve) and was detached from it on 2026-09-17 at
mlx-serve commit `ef5e667` (two commits after mlx-serve v26.9.4). This file covers sushi's own changes since then;
earlier history is mlx-serve's, in that project's changelog.

## Unreleased

- **Metrics can be turned on in a running server**: `POST /metrics/enable` (curl, or the dashboard's new Enable metrics button, shown only while metrics are off) starts the same `/metrics` and `/metrics.json` feeds as `--metrics` without a restart; counters start at zero when it is pressed.
- **A prompt that does not fit unloads idle resident models before it is refused**: with several models loaded, a long prompt used to get a 400 while an idle model held the memory; it now unloads the least recently used idle models (never its own or one serving a request) until it fits.
- **DeepSeek-V4.1 reads its routers as stored**: the BF16 router weights are no longer widened to f32 at load, so the logits are bit-identical, each decode step reads half the router bytes, and 0.16 GB returns to the expert cache.
- **Qwen and GLM agent turns no longer re-prefill the previous reply's thought**: a thinking request with tools now keeps a restore point right behind its generated `</think>` (on GLM the pool position at or below it), so the next turn restores through the thought, MTP and DFlash2 rounds cut to land on it with byte-identical output.
- **A running stream keeps a quarter of the time while another request prefills**: `--prefill-decode-share` now defaults to 0.25 (was 0), so a concurrent decode no longer advances one token per prefill chunk; the share only adds decode ticks between chunks and never narrows a prefill chunk, on any model.
- **Every model keeps SSD restore points every 8192 tokens, to 1M tokens**: Qwen and GLM no longer thin a long conversation to 16 and 8 points, so an edit deep in a long prompt restores from the point just below it; raise `--prefix-cache-disk` for 1M-token conversations, since one that outgrows the budget drops its oldest points first. `--ssm-checkpoint-max` now caps only the states held in memory.
- **`/v1/responses` keeps the model's own reasoning in a tool loop**: a replayed plaintext `reasoning` item now reaches the prompt on its assistant turn (it was dropped), so a thinking model keeps its chain of thought between tool calls and the prefix cache hits; encrypted or malformed reasoning items are still ignored.
- **Tool-call parsing handles five more malformed shapes**: stray DeepSeek `</｜DSML｜` closers no longer leak as the answer, a `<tool_call>` whose value holds an `<invoke>` stays that call, prose `<invoke>` openers no longer take the next call's arguments, a doubled GLM `<arg_key>` keeps a clean key, and a special token written in place of `<` before `function=` still parses.
- **A small-memory Mac gets a usable default context**: the prefix cache's reserve no longer takes more than half of the free GPU memory when sizing the context, and a context that would fit under 1024 tokens is served as 1024.
- **Image input accepts GIF (first frame) and BMP** beside JPEG, PNG and WebP.
- **DeepSeek-V4.1-Flash no longer aborts on M1 and M2 Macs**: the indexer's top-k kernel probes the largest threadgroup the GPU runs (832 threads there, not 1024) at load, with identical picks.
- **`sushi launch codex` gives local models codex's current tool contract**: the launch writes `~/.sushi/codex/model-instructions.md` and points `model_instructions_file` at it, so models stop receiving fallback instructions that name an `apply_patch` tool codex no longer declares and end edit turns with no change.
- **A reasoning budget no longer turns MTP off for the answer**: Qwen and MiMo re-seed their MTP head after the forced thought close and keep speculating, and the close lands at the same token as without MTP, alone or beside concurrent requests, with byte-identical output.
- **GLM-5.3-Flash decode uses fewer GPU launches per MoE layer**: the shared expert now runs inside the routed experts' pair and down launches, for a serial token (two fewer), for batched requests and for DFlash2 verify rows (four fewer), with byte-identical output.
- **`--vision-bf16` keeps a bf16 image tower as stored**: DeepSeek-V4.1 and Qwen skip the load-time 8-bit requantization and bill the full tower (about 0.47 GB and 0.30 GB more memory) for the stored tower's exact image quality.
- **Qwen3.8-Flash-Next decode uses fewer GPU launches per MoE layer**: the shared expert, its gate logit and the gated add now run inside the routed experts' pair, down and finish launches, for a serial token (four fewer) and for MTP and DFlash2 verify rows (seven or more fewer), on both the one-bank and the rate-grouped packs, with byte-identical output.
- **A load that wires more than macOS allows warns at startup**: past `vm.global_user_wire_limit` (116.82 GB by default on a 128 GB Mac) a prefill runs slower, so the log now names the limit and the `sudo sysctl` that raises it.
- **DeepSeek-V4.1-Flash decodes with 200 fewer GPU launches per token**: the shared expert now runs inside the routed experts' decode launches on every speculated layer, serial decode and DSpark verify alike, with byte-identical output.
- **DeepSeek-V4.1-Flash reads its token embedding from SSD**: the 4-bit table's 0.37 GB is no longer resident and goes to the expert cache; output is unchanged.
- **Qwen's image tower takes a third less memory**: the packs' bf16 vision tower is requantized to 8 bits at load (0.90 GB to 0.60 GB, about 30 ms), so a resident load holds 0.30 GB less and a streamed `--vision` load gives the expert cache the difference; the tower's output stays within cosine 0.999 of the bf16 tower.
- **DeepSeek-V4.1's image tower takes half the memory**: the pack's bf16 vision tower is requantized to 8 bits at load (0.97 GB to 0.50 GB, under 0.1 s), so images leave the expert cache about 0.5 GB more; the tower's output stays within cosine 0.999 of the bf16 tower.
- **DeepSeek-V4.1-Flash prefills in about 8 GB less memory**: sushi now owns the pack's weight memory and every forward runs under a hard per-phase cap, so a 2048-row prefill chunk reserves 6.6–7.1 GB instead of 15–16 GB and no longer borrows expert layers. Output and prefill speed are unchanged. A request that would pass its cap fails with a 503 and the server keeps serving.
- **DeepSeek speculates under JSON schemas**: a `response_format` JSON schema keeps DSpark on, with each draft row checked against the grammar.
- **A queued request says so**: a streamed request waiting behind another gets an SSE comment naming the queue, and the log shows `[queue]` lines; a shutdown ends queued and running requests with a 503 instead of a truncated `length` finish.
- **`sushi launch opencode` runs the newest opencode**: with an older opencode ahead of a newer one on PATH, the launcher started the older; it now picks the highest version installed.
- **The server log names the token that ended each turn**: a reply that finishes `stop` logs `[stop]` with the EOS token id and text, the token before it, the completion count and the decode path, so a turn that ends mid-sentence can be traced to its token.
- **`--model` and `sushi run` accept a relative path**: `sushi serve --model ./pack` used to fail with `ExpertLayoutUnsupported` (or lose expert streaming detection) unless the path was absolute; it is now resolved against the current directory at launch.
- **Claude Code can count tokens**: `POST /v1/messages/count_tokens` answers `{"input_tokens": N}`, the prompt a
  `/v1/messages` request with the same body would prefill, images included, without generating anything. Thanks to
  [@cosmic-quokka](https://github.com/cosmic-quokka) for the request and the patch.

## v1.4.0 — DeepSeek-V4.1-Flash, Qwen STQ 2.75bpw, sushi bench

- **DeepSeek-V4.1-Flash on a 128 GB Mac**: the new DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw pack (332 GB with its Engram tables) streams its routed experts from the SSD within `--ssd-budget-gb`, with DSpark speculative decoding, image input on by default, up to 1M tokens of context, DSML tool calls, thinking at high effort by default and `--drop-expert` to trade a few re-routed experts for memory. Greedy DSpark output is the same from run to run on the same Mac (`SUSHI_DSV41_DSPARK_EXACT=0` trades that for a plain step about 0.3 ms faster). On an M5 Max it decodes 76.6 tok/s at 4k on code and 49.3 on prose, and prefills 1,024 tok/s (sushi bench, `--ssd-budget-gb 108`).
- **Long agent sessions on DeepSeek resume where they left off**: the SSD prefix cache keeps a restore point every 8192 tokens, a follow-up turn that diverges inside the last answer rolls the live state back in place with no SSD read, a thinking turn restores from its own thought, and a request may carry up to 1024 images, of which only those the restored prefix does not hold are decoded and encoded again.
- **Qwen3.8-Flash-Next Sushi-STQ-2.75bpw** replaces Sushi-2.6bpw and Sushi-3bpw as the 64 GB recommendation (`sushi pull qwen3.8-flash-next`, KLD 0.0692 at 16x512). Qwen now thinks at medium effort by default, runs Claude-style `<function_calls>` markup as tool calls, and skips the MTP head's 1.2 GB at load under `--no-mtp`. `sushi pull` and `sushi list` use the token `hf auth login` saved, so a private pack you have access to lists with its size and pulls without `HF_TOKEN`. Thanks to [@bobosola](https://github.com/bobosola) for the report (#63).
- **Faster speculative decoding**: Qwen verifies 2 to 8 rows at an even price per row (an eight-row verify about 16% faster), DFlash2 now serves Qwen's image, video and sampled requests (about 1.7-2x plain decoding), GLM-5.3-Flash gets adaptive DFlash2 over 2 to 8 rows, and Macs without matrix units verify 3 to 8 rows up to 2x faster, all with output byte-identical to before.
- **Memory you can trust**: admission plans under the GPU memory other apps hold and counts MLX's command buffers, a Metal out-of-memory failure no longer fails the request after it, DFlash2 reserves exactly what it uses (about 2.8 GB more fits at 1M context), and DeepSeek keeps MLX's buffer pool inside its budget. Thanks to [@holistech](https://github.com/holistech) for the report (#61).
- **`sushi bench`** drives a running server through the release bench: one cold prefill in 8k restore windows up to 128k and three decode samples a rung, on a code and a prose corpus, with the GPU clock read beside every request. `frequency_penalty` now follows OpenAI's definition.
- **`sushi convert exl3 --codebook` is now `--expert-codebook`** (breaking): `--codebook mcg` becomes `--expert-codebook exl3-mcg` and `--codebook mul1` becomes `--expert-codebook exl3-mul1`; packs and resumable runs are unchanged.

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
