# Launch flags

A flag exists for a setting a user picks: a real tradeoff (`--kv-quant 4` for context at some quality cost, `--expert-pick-tolerance` for speed against exactness) or a fit to their own machine (`--mtp-min-depth` / `--mtp-max-depth` per Apple chip, `--qwen-gdn 16` for memory, memory and context limits). Operator settings qualify too: ports, API key, paths, logging. A flag that offers no real choice is bloat: a developer A/B switch, a kill switch for an adopted default, a rejected or never-measured experiment, a flag that does nothing, two flags that set the same thing, a flag whose every useful value is the default. A test that needs to force an arm gets a test seam or a temporary env knob ([envvar](envvar.md)), not a flag.

Precedence: an explicit flag outranks `model-settings.json`, which outranks the default ([server-lifecycle](server-lifecycle.md#settings)). A flag that mirrors a settings key adds the precedence and nothing else; that is not bloat by itself.

The map is the code's: every argument the parsers accept, at `main` 393228b9 less the flags listed under [Removed](#removed) (the parse-site line numbers below are 393228b9's and drift). `main.zig` is the one parser for `run`, `serve` and the bare `sushi [options]` form; `launch`, `update`, `kld` and `convert` parse their own. `pull` and `list` take no flags and reject extra arguments.

**Reading the columns.** *default → effect* ends with the parse site. *who* is `user` (a tradeoff), `machine` (fits a Mac), `operator` (port, auth, paths, logging), `dev` (A/B, debug, bench), `test` (only a test needs it), `dead` (no effect left). *documented* names `help` (`sushi --help` or the subcommand's), `usage` (`docs/usage.md`), `README`, the topic docs, and how many `tests/` scripts pass it (`scripts/` counted apart); *hidden* means parsed but absent from `--help`. *recommendation* starts with `keep`, `cut`, `merge` (names the target) or `owner call` (the evidence that decides it is under [Owner calls](#owner-calls)).

**Summary.** 104 rows: 57 shared by `run`, `serve` and the bare form, 47 in `run`, `launch`, `update`, `kld` and `convert`. By who: user 32, operator 28, machine 17, dev 26, dead 1. By recommendation: keep 96, cut 4, owner call 4. Among the 57 shared rows: keep 51, cut 4, owner call 2.

## Serving, network and auth

| flag | default → effect | who | documented | recommendation |
|---|---|---|---|---|
| `--serve` | off → HTTP server mode; the headless `--serve --model-dir` form is what the app launches (main.zig:620) | operator | help, server-lifecycle; 123 tests | keep: the app depends on it; overlaps the `serve` subcommand only in spelling |
| `--host` | `127.0.0.1` → bind address, `0.0.0.0` opens the LAN (main.zig:617) | operator | help, server-lifecycle; 52 tests | keep |
| `--port` | `12345` → bind port (main.zig:614) | operator | help, server-lifecycle; 139 tests | keep |
| `--api-key` | none → required on every request, loopback exempt (main.zig:728) | operator | help, server-http-apis; 2 tests | keep |
| `--api-key-strict` | off → also require the key from loopback (main.zig:732) | operator | help; 1 test | keep: embedders on shared machines |
| `--api-key-env` | none → read the key from a named env var so argv does not carry it (main.zig:734) | operator | help, envvar; 0 tests | keep: the only way to keep the key out of the process table |
| `--metrics` | off → Prometheus `/metrics` and `/metrics.json` (main.zig:726) | operator | help, server-http-apis; 10 tests, 2 scripts | keep |
| `--max-concurrent` | `1` → only sizes the submit queue (`n + 32`) and floors the prefix-cache capacity; slots batch at any value (main.zig:913) | dead | help; server-lifecycle; 21 tests, 2 scripts | cut: the queue depth is fixed, batching is the model's answer, not this number |
| `--parent-pid` | none → exit when that pid dies (main.zig:811) | operator | help, server-http-apis; 1 test | keep: for a host that embeds the engine |
| `--timeout` | `300` → stall timeout in seconds, 0 = none (main.zig:690) | operator | help, server-http-apis; 1 test | keep |
| `--no-prevent-sleep` | off → allow idle sleep during inference (main.zig:697) | operator | help; 1 test | keep |
| `--gpu-warm-secs` | `60` → keep the GPU awake this long after a request (main.zig:953) | machine | help, server-lifecycle; 0 tests | keep: a power-against-latency choice |
| `--no-tool-autocorrect` | off → stop coercing tool-call arguments to the schema's types (main.zig:706) | dev | help, server-tool-calling; 0 tests | cut: kill switch of an adopted default, no test, no measurement |
| `--no-update-check` | off → never ask GitHub for a newer release (main.zig:818) | operator | help, server-lifecycle; 2 tests | keep |

## Model, memory and context

| flag | default → effect | who | documented | recommendation |
|---|---|---|---|---|
| `--model` | `~/.sushi/models` default → model directory (main.zig:611) | operator | help, usage; 135 tests | keep |
| `--model-dir` | none → repeatable discovery root, up to 8 (main.zig:916) | operator | help, server-lifecycle; 27 tests | keep |
| `--max-resident-models` | `3` → cap on loaded models, bad value falls back to 3 (main.zig:933) | machine | help; 5 tests | keep |
| `--max-resident-mem` | `auto` = 80% of the wired limit → cap on summed resident bytes (main.zig:939) | machine | help, server-lifecycle; 0 tests | keep |
| `--idle-evict-secs` | off → unload a model idle this long (main.zig:959) | machine | help; 1 test | keep |
| `--ctx-size` | model max → context length; also `ctx_size` in settings (main.zig:663) | machine | help, usage, README; 40 tests | keep |
| `--config-overrides` | none → JSON deep-merged into every model's `config.json`, e.g. YaRN extension (main.zig:666) | user | help; 0 tests | keep: the context-extension lever; add a test |
| `--vision` / `--no-vision` | tower loads on a resident model, not on a streamed one → force it on or save its memory; also `vision` in settings (main.zig:693) | machine | help, README, topic docs; 17 tests | keep |
| `--skip-mem-preflight` | off → bypass the free-RAM check before a load (main.zig:699) | machine | help, engine-memory-admission, qwen4-perf; 0 tests | keep: the refusal text names it; the M2 Max runs need it |
| `--wired-margin-gib` | `1` → headroom under `iogpu.wired_limit_mb`, 1..32 (main.zig:898) | machine | help, README, engine-memory-admission; 0 tests | keep |
| `--ssd-budget-gb` | none → stream experts from SSD within a TOTAL resident target; also `ssd_budget_gb` in settings (main.zig:985) | machine | help, README, engine-expert-streaming; 9 tests | keep |
| `--expert-pick-tolerance` | `0` exact → swap a missed streamed expert for a cached one within n, 0..0.6 (main.zig:904) | user | help, README, engine-expert-streaming; 1 test | keep: a measured speed-for-exactness tradeoff |
| `--prefill-chunk` | `8192` → ceiling on tokens per prefill chunk, auto-capped per model (main.zig:824) | machine | help, usage, engine-memory-admission; 11 tests, 2 scripts | keep |
| `--prefill-decode-share` | `0` → target decode share of wall time during another request's prefill, 0..0.9 (main.zig:910) | user | help, server-lifecycle; 1 test | keep: latency of live streams against prefill speed |

## KV and attention

| flag | default → effect | who | documented | recommendation |
|---|---|---|---|---|
| `--kv-quant` | `8` → KV scheme 8, 6, 4 or 16 (`off` = BF16; GLM takes 8, 6 and 16); also `kv_quant` in settings and per request (main.zig:972) | user | help, usage, engine-kv-cache; 21 tests | keep |
| `--qwen-gdn` | `32` → GatedDeltaNet state width in bits, `32` (f32) or `16` (bf16), Qwen only; the decay gate is f32 either way (main.zig:830) | machine | help, usage, engine-kv-cache, quality-kld; 3 tests (parse, config, boot line) | keep: the 32, 48 and 64 GB recipes pass `16` to halve the state per checkpoint (about 54 MB each); against the f32 default the bf16 state (arm b) is about 2% faster on 4-stream aggregate and its KLD is inside the noise ([qwen4-kld](qwen4-kld.md#gdn-f32-teacher), [qwen4-perf](qwen4-perf.md#gdn-f32-cost)) |
| `--mtp-head-kv-quant` | off → Qwen's MTP head KV follows `--kv-quant`; no effect on MiMo (main.zig:762) | machine | help, usage, README, engine-mtp; 0 tests | keep: the long-context memory bills use it |

## Speculation (MTP, prompt lookup, drafter)

| flag | default → effect | who | documented | recommendation |
|---|---|---|---|---|
| `--mtp` / `--no-mtp` | on for Qwen and MiMo, off for GLM → force the head on (required for a streamed pack) or off; also `mtp` in settings (main.zig:754) | user | help, usage, engine-mtp; 22 tests | keep |
| `--mtp-min-depth` | `1` → fewest tokens a round drafts, 1..8 (main.zig:771) | machine | help, engine-mtp; 5 tests | keep: depth range is a property of the chip; launch-only by design |
| `--mtp-max-depth` | adaptive, up to 8 on eligible M5 → most tokens a round drafts, 1..8 (main.zig:771) | machine | help, engine-mtp; 5 tests | keep |
| `--mtp-typical` | off, exact → typical acceptance, d > 0, lossy for sampled requests; also `mtp_acceptance` in settings (main.zig:780) | user | help, engine-mtp; 1 test | keep: measured 15-20% on sampled decode; docs warn it has looped |
| `--mtp-tokenv3` | off → TokenV3 cascade acceptance, 0..1, lossy (main.zig:783) | user | help; 2 tests | owner call |
| `--mtp-greedy-tail` | off → sampled requests draft only the first token by sampling; also in settings (main.zig:786) | user | help, engine-mtp, mimo2-perf; 1 test | keep: measured to pay only beside `--mtp-typical` |
| `--max-mtp-ctx` | `0` no ceiling → MTP off past n context tokens (main.zig:790) | machine | help, `/props`; 0 tests | owner call |
| `--no-mtp-lookup` | off → drop prompt-lookup drafts inside MTP rounds and GLM DFlash2 chains (cli.zig:708) | dev | help, engine-mtp; 2 tests | cut: kill switch of an adopted default; one arm per boot is why a bench wants it |
| `--drafter` | none → path of a DFlash2 assistant stored elsewhere (GLM, Qwen3.8-Flash-Next); on Qwen3.8-Flash-Next it is the only way to run DFlash2, a drafter beside the pack is ignored and MTP stays the default (main.zig:719) | operator | help, usage, glm5-arch, qwen4-arch; 7 tests | keep |
| `--drafter-min-depth` / `--drafter-max-depth` | Qwen3.8-Flash-Next: `2` / `8` → the verifier rows (t1 plus drafts) a DFlash2 round may carry, the round choosing within them from the drafter's confidence; equal values pin one count. GLM-5.3-Flash: the same, `2` / `8`, chosen from the assistant's tree. Integers 2..16 (8 on Qwen and GLM); a bad value or a min above the max exits before any load, a model without a drafter logs that they are ignored (main.zig) | machine | help, qwen4-arch, glm5-arch; 3 tests | keep: the best row count is a property of the chip's verify cost; mirrors `--mtp-min-depth`/`--mtp-max-depth` |
| `--no-drafter` | on when the assistant is found → never load it, saves its memory (main.zig:752) | machine | help; usage, glm5-arch; 9 tests | keep |

## Prefix cache

| flag | default → effect | who | documented | recommendation |
|---|---|---|---|---|
| `--prefix-cache-entries` | `32` → hot-cache capacity, `0` disables all reuse; a bad value silently becomes 1 (main.zig:840) | user | help, usage, engine-prefix-cache; 41 tests, 2 scripts | keep |
| `--prefix-cache-mem` | off → RAM retention with this byte budget, `0` or `off` keeps it off (main.zig:843) | machine | help, usage, engine-prefix-cache; 19 tests, 2 scripts | keep |
| `--prefix-cache-disk` | min(sized + 2 GiB, 20 GiB, free - 4 GiB) → SSD tier budget, `0` or `off` disables (main.zig:854) | machine | help, usage, engine-prefix-cache; 16 tests, 2 scripts | keep |
| `--ssm-checkpoint-max` | `16` → checkpoints kept per entry, `0` unlimited (main.zig:895) | dev | help; 2 tests, 2 scripts | cut: the scripts pass the default and the cache byte budget already bounds it |

## Sampling and reasoning defaults

| flag | default → effect | who | documented | recommendation |
|---|---|---|---|---|
| `--max-tokens` | `100` offline; serve: default for requests that omit it (main.zig:649) | operator | help, usage; 9 tests | keep |
| `--temp` | `0` offline; serve: default temperature before `generation_config.json` (main.zig:653) | operator | help, usage; 2 tests | keep |
| `--top-p` | model config → serve default for requests that omit it (main.zig:657) | operator | help; 1 test | keep |
| `--top-k` | model config → serve default for requests that omit it (main.zig:660) | operator | help; 1 test | keep |
| `--reasoning-budget` | unlimited → max thinking tokens per request (main.zig:799) | user | help, server-http-apis; 2 tests | keep |
| `--preserve-thinking` | template default → keep every turn's thinking, `on` or `off`; also in settings (main.zig:876) | user | help, usage; 0 tests (settings test) | keep |
| `--think-penalty` | `0` off → lower overthinking-marker logits inside the reasoning span; also in settings (main.zig:868) | user | help, server-http-apis; 2 tests | keep |

## Logging, diagnostics and info

| flag | default → effect | who | documented | recommendation |
|---|---|---|---|---|
| `--log-level` | `info` → error, warn, info or debug; a bad value is silently ignored (main.zig:802) | operator | help; 81 tests | keep |
| `--log-file` | `~/.sushi/logs/sushi-<port>.log` → path, or `off` (main.zig:808) | operator | help; 16 tests | keep |
| `--version` | prints app, MLX, mlx-c and NAX status and exits (main.zig:573) | operator | help; 2 tests, 1 script | keep: the app and `update` parse it |
| `--guest-manifest` | prints the host-check JSON and exits (main.zig:592) | operator | help; 2 tests | keep: `update` and the release gate run it |
| `--help` / `-h` | prints usage (main.zig:608) | operator | help; 5 tests | keep |

## Subcommands

`pull <model>` and `list` take no flags and reject any extra argument (`HF_TOKEN` is read from the environment); `pull` accepts only the `beamster` org. `serve` and `run <model>` take every flag above.

| flag | default → effect | who | documented | recommendation |
|---|---|---|---|---|
| `run`: `--think` | model default → reasoning effort, per model (main.zig:623) | user | help, server-http-apis; 2 tests | keep |
| `run`: `--tool` | `off` → REPL tools: web, fetch, read files in the folder (main.zig:631) | user | help, server-http-apis; 1 test | keep |
| `run`: `--edit` | `off` → add `write_file` and `edit_file` inside that folder (main.zig:637) | user | help, server-http-apis; 2 tests | keep |
| `run`: `--prompt` / `-p` | none → one prompt and exit (cli.zig:2354) | user | help; 4 tests | keep |
| `run`: `--stream` | off → stream tokens with `--prompt` (main.zig:644) | user | help; 1 test | keep |
| `launch`: `--model` | the server's default → model id (launch.zig:867) | operator | help, usage; 0 tests | keep |
| `launch`: `--url` | `http://127.0.0.1:<port>` → server base URL (launch.zig:871) | operator | help; test_launch_cmd, test_zcode_launch | keep |
| `launch`: `--port` | `12345` → server port for the default URL (launch.zig:875) | operator | help; 0 tests | keep |
| `launch`: `--print` | off → write configs and print the script instead of running (launch.zig:879) | user | help, usage; test_launch_cmd | keep |
| `launch`: `--` | none → everything after goes to the agent (launch.zig:864) | user | help, usage; test_launch_cmd | keep |
| `update`: `--check` | off → report only (update.zig:709) | user | help; 1 test | keep |
| `update`: `--pre` | off → consider prereleases (update.zig:711) | user | help; 0 tests | keep |
| `update`: `--force` | off → update while a sushi of this install runs (update.zig:713) | user | help; 0 tests | keep |
| `update`: `--rollback` | off → swap `<install>.previous` back (update.zig:715) | user | help; 1 test | keep |
| `update`: `--relaunch --` | internal: the updater restarts the server with its argv (update.zig:717) | dev | not in help (internal); 1 test | keep: plumbing, not a user flag |
| `kld`: `--model` | required → pack to score or capture (kld.zig:99) | dev | help, quality-kld | keep |
| `kld`: `--prompts` | required for capture → `standard4`, a fixture dir, `*.txt` dir or `.jsonl` (kld.zig:100) | dev | help, quality-kld | keep |
| `kld`: `--out` | required for capture → fixture output dir (kld.zig:101) | dev | help, quality-kld | keep |
| `kld`: `--fixture` | required for compare → teacher fixture dir (kld.zig:102) | dev | help, quality-kld | keep |
| `kld`: `--json` | none → compare writes the numbers here (kld.zig:103) | dev | help, quality-kld | keep |
| `kld`: `--label` | `sushi` → run label in the output (kld.zig:104) | dev | help, quality-kld | keep |
| `kld`: `--tokens` | `64` → greedy tokens captured per prompt (kld.zig:105) | dev | help, quality-kld | keep |
| `kld`: `--top-k` | `10` → top_k recorded in `baseline.json` (kld.zig:106) | dev | help, quality-kld | keep |
| `kld`: `--limit` | `0` all → first n prompts only (kld.zig:107) | dev | help, quality-kld | keep |
| `kld`: `--no-template` | off → feed raw prompt text (kld.zig:143) | dev | help, quality-kld | keep |
| `kld`: `--ctx-size` | model max → context override (kld.zig:108) | dev | help | keep |
| `kld`: `--kv-quant` | dense BF16 → KV scheme; the teacher adds no quantization of its own (kld.zig:109) | dev | help, quality-kld | keep |
| `kld`: `--qwen-gdn` | `32` → `16` scores a bf16-state student against the f32 teacher; a GLM capture refuses it (kld.zig:107) | dev | help, quality-kld; 2 tests | keep |
| `kld`: `--ssd-budget-gb` | none → expert streaming budget (kld.zig:111) | dev | help, quality-kld | keep |
| `kld`: `--expert-pick-tolerance` | `0` → compare only, lossy swap of a missed expert (kld.zig:113) | dev | help, quality-kld; 1 test | keep |
| `kld`: `--wired-margin-gib` | `1` → headroom under the wired limit (kld.zig:114) | dev | help | keep |
| `kld`: `--mtp` / `--no-mtp` | head not resident → keep the head loaded (kld.zig:145) | dev | help; quality-kld | keep |
| `kld`: `--layer-major` | off → native GLM BF16 teacher in batches, layer by layer, resumable (kld.zig:151) | dev | help, quality-kld, glm5-arch | keep |
| `kld`: `--batch-windows` | from the budget → windows per layer-major batch (kld.zig:115) | dev | help, quality-kld | keep |
| `kld`: `--imatrix-windows` | all → layer-major imatrix covers the first n windows (kld.zig:116) | dev | help, engine-expert-streaming | keep |
| `kld`: `--pause-file` | none → layer-major idles while the file exists (kld.zig:117) | dev | help, quality-kld | keep |
| `convert exl3`: `--model` | required → source checkpoint directory (encoder/cli.zig) | user | help, encoder | keep |
| `convert exl3`: `--out` | required → new pack directory, refused when non-empty unless `--resume` (encoder/cli.zig) | user | help, encoder | keep |
| `convert exl3`: `--experts-k` | `2.5` → routed-expert rate, 1.5 to 8 in steps of 0.125 (encoder/cli.zig) | user | help, encoder | keep |
| `convert exl3`: `--codebook` | `mcg` → `mcg` or `mul1` (encoder/cli.zig) | user | help, encoder | keep |
| `convert exl3`: `--window` | `15` → trellis search window, 12 to 16 (encoder/cli.zig) | dev | help, encoder | keep: the window is part of the pack's stamp |
| `convert exl3`: `--calib` | built-in 64k mix → calibration text file (encoder/cli.zig) | user | help, encoder | keep |
| `convert exl3`: `--calib-tokens` | `65536` → calibration tokens, at least 2048 (encoder/cli.zig) | user | help, encoder | keep |
| `convert exl3`: `--resume` | off → continue an interrupted conversion in `--out` (encoder/cli.zig) | operator | help, encoder | keep |
| `convert exl3`: `--batch` | `16` → routed experts quantized together per bank, 1 to 64; the pack bytes do not depend on it (encoder/cli.zig) | operator | help, encoder | keep |
| `convert exl3`: `--factor-precision` | `64` → Hessian factor on the CPU in float64 (`64`) or on the GPU in float32 (`32`, changes the pack bytes; encoder/cli.zig) | operator | help, encoder | owner call |
| `convert exl3`: `--trunk-bits` | the model's served width (Qwen and MiMo 8, GLM 6) → trunk affine bits 2, 3, 4, 5, 6 or 8 (encoder/cli.zig) | user | help, encoder | keep |
| `convert exl3`: `--trunk-group` | the model's served group (Qwen and MiMo 64, GLM 128) → trunk affine group 32, 64 or 128 (encoder/cli.zig) | user | help, encoder | keep |

## Cut list

Delete the flag, its parse arm, its help line and the code only its off arm reaches. Where a test needs the arm, that test gets a temporary env knob or a test seam first ([envvar](envvar.md)).

- **Serving**
  - `--max-concurrent`: only sizes a submit queue (`n + 32`) and a prefix-cache floor; decode batching is decided by the model. 21 tests and 2 scripts pass it, so drop the argument there in the same landing; `/props max_concurrent` and the boot line go with it.
  - `--no-tool-autocorrect`: kill switch of the schema-coercion default, no test, no measurement.
- **Speculation**
  - `--no-mtp-lookup`: kill switch of an adopted default; a bench that wants one arm per boot needs an env knob, not a flag.
- **Prefix cache**
  - `--ssm-checkpoint-max`: the byte budget already bounds the cache; the two scripts pass the default `16`.

<a id="removed"></a>
## Removed

Landed; their rows are gone from the tables above. A removed flag now fails as `unrecognized argument` (`tests/test_serving_deps.sh`).

| flag | what replaced it |
|---|---|
| `--no-safety` | nothing; it was accepted and ignored |
| `--warmup-eager`, `--no-warmup-eager` | the boot warmup always runs; test boots no longer shorten it |
| `--kv-attn-mode` | nothing: no arm read it. The per-request `kv_attn_mode` field and the `SUSHI_KV_ATTN_*` env vars are gone with it; a request still sending the field is served as if it were absent |
| `--decode-attn-quant`, `--no-decode-attn-quant` | nothing: the whole int8 requantization of dense attention at decode is gone; it only engaged on the BF16 source checkpoint, which made the KLD teacher lossy |
| `--mtp-depth` | `--mtp-min-depth` / `--mtp-max-depth`; the tombstone message is gone |
| `--mtp-history-window` | full head history (the old default 0); the windowing code is gone |
| `--draft-block-size` | the machine's automatic block cap (`dflash.blockCapForMachine`) alone |
| `--tokenize-cache-entries` | a fixed 4 (`server.tokenize_cache_entries`) |
| `--prefill-trace` | `--log-level debug` writes the `[prefill-trace]` lines to the log |
| `launch --no-start` | nothing; it was accepted and ignored |
| `--hf-overrides` | `--config-overrides` (the same arm) |
| `--mtp-cascade` | `--mtp-tokenv3` (the same arm) |
| `--no-prefix-cache-ram` | `--prefix-cache-mem off` (or `0`); the last `--prefix-cache-mem` on the line wins, where the old flag won in either order |
| `--fast` | the flags it set: `--mtp`, `--mtp-typical 0.2`, `--mtp-greedy-tail`, `--kv-quant 8`; the `fast` source label and preset are gone |
| `--pld`, `--no-pld`, `--pld-draft-len`, `--pld-key-len` | nothing: standalone prompt lookup decoding, its request field `enable_pld` and `/props settings.pld` are gone; MTP's and GLM DFlash2's own lookups stay (`--no-mtp-lookup`) |
| `--ssm-checkpoint-stride` | a fixed 256 (`server.SSM_CHECKPOINT_STRIDE_DEFAULT`); only tests move it |
| `--expert-cache-gb` (also `kld`) | `--ssd-budget-gb`; the BF16-source streaming test passes it |
| `--embedding-max-length` | the model's window |
| `--logit-bias-file` | request `logit_bias`; the `logit_bias_file` key in `model-settings.json` stays (a settings twin, left alone) |

## Owner calls

What would decide each.

- `--mtp-tokenv3`: a same-boot 16x512 sampled A/B, two seeds, exact against typical against TokenV3 (tok/s and emitted-text NLL, as `engine-mtp` records for typical). None is recorded; it was ported with the upstream verifiers. Cut if it does not beat typical.
- `--max-mtp-ctx`: whether the adaptive speculate-or-serial vote (on by default) already switches MTP off at the contexts a ceiling would, on the M1/M2/M5 long-context ladders. Run `--max-mtp-ctx n` against the default with the vote on; cut if the vote covers every chip.

## Help and parse mismatches

Every flag the parsers accept is in `--help` (the subcommand's for `kld`, `update`, `launch`), except `update --relaunch`, which the updater passes to itself to restart the server and which is internal on purpose (a comment in `update.zig` says so). `kld` keeps its value flags in one table (`kld.value_flags`) that a unit test checks against its usage text; `main.zig` parses with an if-chain, so its help is checked by reading, not by a test.

A bad value for a numeric launch flag, an unknown `--log-level`, and `--tool` outside `sushi run` or `--stream` without `--prompt` exit with an error naming the flag and the value, before any model loads. `sushi pull` and `sushi list` reject extra arguments.
