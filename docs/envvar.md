# Environment variables

A temporary A/B knob is fine while a kernel, optimization or fix is being developed, tuned or bisected; once the answer is known, the knob goes together with its losing arm. What stays is a diagnostic (including a knob a correctness test, an equivalence, parity or byte test, needs) or a lever still in active development, read once through `diagEnvOn` (absent, empty or `0` = off). A setting a user picks for a tradeoff or to fit their machine (GDN state width for long-context precision, MTP depth bounds per chip) is a launch flag, never an env twin ([CLAUDE.md](../CLAUDE.md)). The map is the code's: every `getenv`, `diagEnvOn`, `diagEnvOnCached`, `diagEnvValueOn` and `readEnv*` read in `src/` (the three served architectures and their shared code). `HOME`, `TMPDIR`, the variable `--api-key-env` names, the `QWEN4_PROFILE_` prefix scan of the round-cost store and the agent-launch scripts `sushi launch` writes are not levers and are left out. Tooling that only `scripts/` and `tests/` read is in the last table.

**Reading the columns.** *where* is file:line of the read site (served path first; a group lists files). *read* is `once` (cached on first use on the served path), `boot` or `load` (once at startup or model load), `per request`, `per call` / `per forward` / `per round` (an uncached `getenv` on a repeated path), or `test` (never on the served path). Every diagnostic read treats empty and a leading `0` as off. *documented* is a hit in `docs/`, `README.md`, `tests/CLAUDE.md` or `--help`.

**Summary.** 137 vars. By kind: tradeoff levers 14, diagnostics 51, microbenches 19, test-only seams 43, paths and config 8, flag twins that are not equivalents 2.

## Tradeoff levers (14)

Two arms that trade speed, memory or exactness, or a policy a user may want to move.

| var | where | default → effect | read | documented |
|---|---|---|---|---|
| `MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER` | read by MLX (`device.cpp`), not sushi; mirrored by `server.resolveMlxCadence` | MLX's arch default (50 / 50 on Max and Ultra, 40 / 40 on base and Pro). A larger MB per buffer keeps more prefill inputs alive and the bill adds (MB - 50) MiB x 9/8 for it; ops per buffer is not billed (measured no effect) | once, at the first Metal device | yes: engine-memory-admission |
| `MLX_ENABLE_TF32` | `qwen4_qsa_select.zig:174, glm5_kld_capture.zig:196` | default on: the QSA score GEMM may use TF32 on NAX; `0` keeps f32. GLM teacher capture requires `0` | once | yes: glm5-kernels, glm5-kld, quality-kld, ... |
| `QWEN4_PLE_PREFETCH_PREFILL` | `qwen4_exp.zig:951` | wide n-gram gather arm: unset = kv gate (pool from 256k keys, serial walk below), `0` serial, `1` pool; both arms measured, the best flips with table residency | once | yes: qwen4-arch, qwen4-perf |
| `SUSHI_DECODE_ASYNC_LADDER` | `transformer.zig:1473` | unset: batched Qwen4 decode ladders every 4th layer, other paths serial; `0\|off` off, `auto` every 8th, `<n>` every n-th | boot | yes: engine-kernels, qwen4-perf |
| `SUSHI_DFLASH_QUANT_BITS` | `dflash.zig:461` | affine width of the DFlash sidecar requant (default 8; `0\|off` or an unsupported width keeps dense bf16) | load | no |
| `SUSHI_DSV41_DSPARK_EXACT` | `dsv41_serving.zig` (`initDsv41`) | default on: a DSpark round's trunk linears give a row the same bytes at every verify width (the NAX arm at one row too, the lm_head on the SIMD rows arm), so greedy text is the same at every round-width mix; `0` takes the faster arms (SIMD one-row step, NAX head), whose text may differ with the mix | load | yes: dsv41-arch |
| `SUSHI_LMHEAD_FULL` | `transformer.zig:5722` | `1` reads the full lm_head instead of the greedy top-32 shortlist (MiMo; the shortlist is not certified exact) | once | yes: engine-mtp |
| `SUSHI_LOOP_TRIM` | `slot_outcome.zig:68` | default on: trims displayed loop-stop replies; `0` keeps the whole degenerate tail for strict bytes | once | yes: server-http-apis |
| `SUSHI_MTP_ADAPTIVE_SERIAL` | `mtp_policy.zig:811` | default on; `0` disables the adaptive speculate-vs-serial vote at long KV | once | yes: mimo2-perf |
| `SUSHI_MTP_DRAFT_HEAD_BITS` | `mtp.zig:374, mimo_mtp.zig:33` | coarse rerank readout width (default 3, MiMo 2; `0\|off` disables); verification is untouched | load | yes: mimo2-mtp |
| `SUSHI_NGRAM_WARM` | `qwen4_exp.zig:148` | default on: pread the Flash-Next n-gram table at load; `0` skips it | once | yes: qwen4-arch, qwen4-perf |
| `SUSHI_QSA_SCORE_SHEET_MB` | `qwen4_qsa_attn.zig:1214` | QSA block-score sheet budget in MB (default 256); the memory bill reads it too, and a moved value is logged | once | no |
| `SUSHI_ROUND_COST_PERSIST` | `round_cost.zig:759` | `1` persists the measured MTP round-cost table across boots (default off) | once | yes: engine-mtp, mimo2-perf, qwen4-perf, ... |

## Diagnostics (51)

Log, trace or dump output, off by default, and the knobs a correctness test needs (the test is named in the row).

| var | where | default → effect | read | documented |
|---|---|---|---|---|
| `QWEN4_DEBUG_SCORES` | `qwen4_qsa.zig:1334, 1383` | prints QSA pooled shapes and score rows | once | no |
| `QWEN4_DUMP_QSA_BLOCKS` | `qwen4_qsa.zig:627` | `<path>` dumps the selected QSA block indices | once | no |
| `QWEN4_FIXTURE_ROWS` | `qwen4_fixture_test.zig` | prints per-row cos/margin in the Qwen4 fixture test | test | no |
| `SUSHI_TEST_NO_GPU` | `mlx.zig` (`testNoGpuEnv`) | a test build sees no Metal backend, so `requireGpu` tests skip | test | yes: testsuite |
| `QWEN4_GDN_CHECK` | `qwen4_gdn.zig:2703, 2849` | runs the composed GDN prework beside the fused kernel and compares | once | no |
| `QWEN4_NO_POOLED` | `qwen4_qsa.zig:1242` | forces `QsaPooledRequired` to exercise the pooled-key fallback | once | no |
| `QWEN4_OPCOUNT` | `qwen4_forward.zig:2940` | FFI op counts per block of the Qwen4 forward | once | no |
| `QWEN4_PROFILE_FWD` | `qwen4_forward.zig:1454, 2936; qwen4_exp.zig:783; qwen4_hc.zig:1350; qwen4_fixture_test.zig:1600, 3481` | `1` times S 2..16 forwards per block with a sync each, `all` every S; also logs PLE reads and keeps verify off the grouped-rows path | once | no |
| `QWEN4_PROFILE_QSA` | `qwen4_qsa.zig:1359, qwen4_forward.zig:2521` | logs synced QSA indexer/gather laps | once | no |
| `QWEN4_STANDIN` | `qwen4_forward.zig:2898` | comma list (`gdn,attn,mlp,hc,moe_*,...`) replaces blocks with free stand-ins for ablation under the decode meter | once | yes: engine-kernels, qwen4-perf |
| `QWEN4_STREAM_F32` | `qwen4_forward.zig:1518, 1780, 1968` | runs the Qwen4 residual streams in f32 (math oracle) | once | yes: qwen4-arch |
| `SUSHI_DSV41_PREFILL_PROFILE` | `dsv41_serving.zig:41` | logs a synced per-stage `[dsv41-prof]` line per wide DeepSeek chunk | load | yes: dsv41-arch |
| `SUSHI_DECODE_FWD_GRAPH` | `model_load.zig:1872` | `<path>` writes one decode forward's primitive list at the forward meter; inert without `SUSHI_DECODE_FWD_UBENCH` | load | yes: mimo2-arch |
| `SUSHI_DECODE_PROFILE` | `forward_diag.zig:119, 127` | arms the decode sub-block profiler; `=<rows>` also times verify widths | once | no |
| `SUSHI_DSV41_ENGRAM_REUSE` | `dsv41_engram.zig` (test) | `<u32 tokens file>`: the Engram row cache's hit rate by capacity over a token stream | test | yes: dsv41-arch |
| `SUSHI_DSV41_NO_DECODE_ENGRAM_AHEAD` | `dsv41_serving.zig` | TEMPORARY A/B: decode-width passes read their Engram rows at the layer; deleted after the live measurement | load | no |
| `SUSHI_DSV41_READ_DIAG` | `dsv41_serving.zig`, `dsv41_model.zig` | `1` counts the blocking host reads of decode-width DeepSeek forwards by site (route-spec settle, settle before a successor, waiting-schedule picks, Engram rows, DSpark state, logits) and logs `[dsv41] reads/forward <site>` per 64 forwards at debug level | load | yes: engine-expert-streaming |
| `SUSHI_DSV41_COMPOSED` | `dsv41_serving.zig`, `dsv41_model.zig` | `1` runs deepseek_v41 on the op chains instead of the fused kernels (the KLD A/B's teacher arm; the load logs `[dsv41] fused kernels on\|off`) | load | yes: dsv41-arch |
| `SUSHI_DSV41_DSPARK_ROWS` | `dsv41_dspark.zig` | deepseek_v41 DSpark diagnostic: `sweep` cycles the verify width 1..8 per round, `<n>` forces it, `log` keeps the policy; each round logs `[dspark-round]` (ctx, rows, draft and verify ms, confidence logits) | request | yes: dsv41-arch |
| `SUSHI_DSV41_VERIFY_PROFILE` | `dsv41_serving.zig`, `dsv41_model.zig` | `1` times every stage of decode-width chunks (a verify) as `[dsv41-prof]` lines, each stage evaluated before the next (the absolute ms inflate about 2x) | load | yes: dsv41-arch |
| `SUSHI_DSV41_CAPTURE_LEAN` | `kld_capture.zig` | `1` (only with `SUSHI_DSV41_HEAD_ROWS`) the layer-major capture writes only tokens, windows, fixture, layout and the head rows | once | yes: quality-kld |
| `SUSHI_DSV41_HEAD_ROWS` | `kld_capture.zig` | `1` the layer-major capture also writes `calib/head-in.bf16`, the head's input for every row | once | yes: quality-kld |
| `SUSHI_DFLASH_TRACE` | `generate.zig:4016` | `1` per-phase DFlash round wall-time trace | once | no |
| `SUSHI_DISPATCH_PROBE` | `transformer.zig:5999` | `<n>` injects n small dependent dispatches per MoE layer to price a dispatch | once | yes: mimo2-perf, tests/CLAUDE |
| `SUSHI_DUMP_RESTORE` | `restore_dump.zig:31` | `<dir>` dumps prefix-restore tensors | once | no |
| `SUSHI_EXL3_STAGE_PROFILE` | `exl3/expert_exl3_kernels.zig:108` | syncs and times each stage of a routed MoE layer (pair GEMV, down, reduce, sort, token prepare, GEMMs), logged as `[exl3-stage]` | once | yes: engine-exl3-experts |
| `SUSHI_EXL3_SWIGLU_MAXABS` | `exl3/expert_exl3_kernels.zig:130` | dumps SwiGLU activation maxima (f16 overflow hunt) | once | yes: engine-exl3-experts |
| `SUSHI_EXL3_UNION_HIST` | `exl3/expert_exl3_kernels.zig:190` | logs per-layer expert-count histograms (`[exl3-counts]`) on prefill | once | yes: engine-exl3-experts, qwen4-perf |
| `SUSHI_EXPERT_DEFER_SYNC` | `qwen4_forward.zig:2007` | `1` forces synchronous expert verification for an exact schedule comparison | once | yes: engine-expert-streaming |
| `SUSHI_FORCE_GPU_FAMILY_FALLBACK` | `nax_gate.zig:136, exl3/expert_exl3_kernels.zig:1112` | `1` pretends the matrix units are absent to rehearse the M1-M4 path on an M5; the EXL3 and verify-qmm parity tests set it | once | yes: glm5-arch, glm5-kernels, glm5-perf, ... |
| `SUSHI_KERNEL_BUILD_LOG` | `mlx_kernel_keys.zig` (the `mlx_fast_metal_kernel_*` wrappers of `mlx.zig`) | `1` logs every custom-kernel library key MLX has not compiled yet as `[kernel-build] #N name src=... template args -> outputs in: inputs`; each is a Metal compile on the inference thread | once | yes: engine-mlx-gotchas |
| `SUSHI_KVQ_PARITY_TRACE`, `SUSHI_MIMO_VISION_DIAG`, `SUSHI_EXL3_PARITY_TRACE` | `test_support.zig`, `mimo_forward.zig`, `mimo_vision.zig`, `mimo_fixture_test.zig` | prints extra parity / vision diagnostics inside a test (the EXL3 one: the MiMo EXL3 tests' relative RMS) | test | yes: CLAUDE |
| `SUSHI_LAYER_CAP` | `forward_diag.zig:63` | `<n>` runs only the first n layers so a ms-vs-layers slope is measurable (`tests/fwd_ubench.sh`) | once | yes: tests/CLAUDE |
| `SUSHI_LMHEAD_SHORTLIST_AUDIT` | `transformer.zig:1821` | `1` re-reads the full head on every shortlist row and counts disagreements | once | yes: engine-mtp, mimo2-perf |
| `SUSHI_MOE_DUMP` | `forward_diag.zig:168` | `<dir>` writes one forward's MoE tensors as raw f32 | once | no |
| `SUSHI_VQMM_PARITY_DEBUG`, `REPLAY_DUMP_SOFT` | `verify_qmm.zig`, `tool_traffic_replay_test.zig` | extra prints inside a test | test | no |
| `SUSHI_MTP_TRACE` | `mtp_policy.zig:488, round_cost.zig:785` | `1` per-round MTP phase trace; also disarms round-cost persistence | once | yes: mimo2-perf, qwen4-perf, tests/CLAUDE |
| `SUSHI_MTP_TRACE_SYNC` | `mtp_round.zig:1718` | drains the GPU before the verify build so the `sync` lap is separable; read only while tracing | once | no |
| `SUSHI_PROFILE_ATTN` | `qwen4_forward.zig:3024` | `1` logs synced indexer/projection/QSA/tail laps per Qwen4 forward | once | yes: qwen4-qsa-long-context |
| `SUSHI_PREFILL_CHUNK_PEAK` | `generate.zig` | at `--log-level debug`, resets the MLX peak after each `[prefill-trace] mem` line, so each line's `peak_bytes` is its own chunk's | once | yes: dsv41-arch |
| `SUSHI_RAW_DUMP_FILE` | `anthropic.zig:1718`, `generation_stream.zig:1678` | `<abs path>` appends the full pre-parse buffer of every tools request (corpus harvest) | per tool request | yes: server-tool-calling |
| `SUSHI_EXL3_GEMM_WIN`, `SUSHI_EXL3_WIN_ALIGN` | `exl3/expert_exl3_kernels.zig` | prefill GEMM window rows (default 32) and `0` unaligns it; the window-geometry parity tests drive both | once | yes: engine-exl3-experts |
| `SUSHI_MTP_COST_TABLE` | `mtp_policy.zig:223` | `0` uses the prior only, no measured round costs; `tests/test_mtp_equivalence.sh` needs it for a seeded sampled byte comparison | once | yes: engine-mtp, tests/CLAUDE |
| `SUSHI_MTP_HEAD_PERSIST` | `generate.zig:413` | default on: Qwen4 MTP head history rides the prefix cache; `0` turns it off (kill switch; no script boots it) | once | no |
| `SUSHI_PREFILL_INTERLEAVE` | `scheduler.zig:2991` | default on: a cold prefill yields to decode streams; `0` disables it (kill switch; no script boots it) | once | yes: server-lifecycle |
| `SUSHI_QSA_NAX` | `qwen4_qsa_gather.zig:356, 432` | `0` runs the stock QSA arm instead of the NAX path (dev A/B lever; `tests/qwen4/test_qsa_nax_prefill.py` runs the default NAX arm) | once | no |
| `SUSHI_VISION_CHUNKED` | `prefill_plan.zig:329` | `0` restores the whole-prompt vision prefill and its full-width bill; `tests/test_vision_chunked_prefill.sh` boots both arms | once | yes: tests/CLAUDE |

## Microbenches (19)

In-process meters. Load-time meters run on the served binary; the rest are tests that skip when unset.

| var | where | default → effect | read | documented |
|---|---|---|---|---|
| `SUSHI_DECODE_FWD_UBENCH` | `model_load.zig:1762` | `<n>` times n decode-width forwards at load (no sampling, no cache bookkeeping), then resets the KV | load | yes: engine-kernels, mimo2-arch, mimo2-perf, ... |
| `SUSHI_DECODE_FWD_UBENCH_S`, `SUSHI_DECODE_FWD_UBENCH_KV` | `model_load.zig` | `<rows>[,...]` verify widths to time; `<tokens>` prefill before timing | load | yes: perf-baselines |
| `SUSHI_DECODE_FWD_UBENCH_PROFILE` | `model_load.zig` | one extra profiled pass per width | load | no |
| `SUSHI_DECODE_TICK_UBENCH` | `generate.zig:87, exl3/expert_exl3_kernels.zig:1080` | `<n>` times n decode ticks end to end | once | no |
| `SUSHI_GLM_ROUND_UBENCH`, `SUSHI_GLM_ROUND_UBENCH_CTX`, `SUSHI_GLM_ROUND_UBENCH_TEXT` | `glm5_rows_tick.zig`, `model_load.zig` | `<n>` greedy DFlash2 rounds per arm at load; context list in tokens; `<file>` text prompt | load | yes: glm5-perf |
| `SUSHI_GLM_ROWS_UBENCH`, `SUSHI_GLM_ROWS_UBENCH_CTX`, `SUSHI_GLM_ROWS_UBENCH_TEXT` | `glm5_rows_tick.zig`, `model_load.zig` | `<n>` rounds per width B in 1..4 of plain decode at load, then one request's 1..8-row chain verify (ms, and bit-equality of every row against serial decode) and a routed layer's feed-forward at 1..8 rows; context list; `<file>` text prompt | load | yes: glm5-kernels, mlx-serve-sync, perf-baselines |
| `SUSHI_VISION_Q8_REPORT`, `SUSHI_QWEN_VISION_UBENCH` | `dsv41_vision.zig`, `dsv41_thought_point.zig`, `qwen_vision.zig` | requantized-tower report (cosine, time) on a real tower, Qwen vision encoder meter (`<reps>`) | test | yes: mimo2-perf, tests/CLAUDE |
| `SUSHI_PREFILL_UBENCH`, `SUSHI_PREFILL_UBENCH_ARMS`, `SUSHI_PREFILL_UBENCH_ROWS`, `SUSHI_PREFILL_UBENCH_TEXT` | `model_load.zig` | `<n>` cold one-chunk prefills per arm at load; arm list; chunk rows (capped at the admitted chunk, default 2025); `<file>` real text | load | yes: engine-kernels, mimo2-perf, qwen4-perf |
| `SUSHI_QSA_PREFILL_UBENCH`, `SUSHI_QSA_WIDE_UBENCH` | `qwen4_qsa_gather.zig` | QSA prefill/wide meters | test | no |

## Test-only seams (43)

Inputs and hooks that real-pack tests and scripts use.

| var | where | default → effect | read | documented |
|---|---|---|---|---|
| `DFLASH_FIXTURES`, `DFLASH_ASSISTANT_DIR`, `DFLASH2_FIXTURES`, `DFLASH2_ASSISTANT_DIR` | `dflash.zig` | DFlash oracle fixtures and sidecar dirs (set by the `tests/dump_dflash*_fixtures.py` dumpers) | test | no |
| `MIMO_V2_MODEL`, `MIMO_V2_SOURCE`, `MIMO_V2_FIXTURE`, `MIMO_V2_MTP_FIXTURE`, `MIMO_VISION_ORACLE`, `MIMO_STREAM_FIXTURE`, `MIMO_SLICED_REAL_FIXTURE`, `MIMO_SLICED_OUTPUT`, `MIMO_DIFFERENTIAL_FIXTURE`, `MIMO_DIFFERENTIAL_LAYER`, `MIMO_DIFFERENTIAL_PACK`, `MIMO_PACK_REFERENCE` | `mimo_fixture_test.zig`, `mimo_config.zig`, `mimo_generate_test.zig`, `mimo_mtp.zig`, `mimo_vision.zig` | MiMo pack, source checkpoint and oracle-fixture inputs for real-pack tests | test | yes: tests/CLAUDE, mimo2-arch |
| `MTP_TEST_MODEL`, `MTP_LMHEAD_TEST_FILE`, `SINGLE_MTP_TEST_CONTEXT` | `verify_moe.zig`, `qwen4_fixture_test.zig`, `qwen4_forward.zig`, `qwen4_generate_test.zig` | MTP real-pack tests (pack, lm_head blob, context length) | test | yes: tests/CLAUDE |
| `QWEN4_FIXTURE`, `QWEN4_FIXTURE_BF16`, `QWEN4_FIXTURE_VISION`, `QWEN4_ROWS_OLD`, `QWEN4_TEST_HF_MODEL`, `QWEN4_TEST_PREFIX_TOKENS`, `QWEN4_VERIFY_BENCH`, `QWEN4_YARN_FIXTURES`, `QWEN4_BF16_STREAM_MODEL` | `qwen4_fixture_test.zig`, `mrope.zig`, `expert_io.zig` | oracle fixtures, HF source dir and probe switches for the Qwen4 oracle tests | test | yes: tests/CLAUDE |
| `QWEN4_TEST_MODEL` | `qwen4_fixture_test.zig:474, 618, 641, +24`, `qwen4_generate_test.zig:387, 531, 640` | `<pack dir>` for the real-pack Qwen4 tests; they skip without it | test | yes: mlx-serve-sync, server-lifecycle, tests/CLAUDE |
| `QWEN_VISION_TEST_MODEL`, `QWEN_VISION_FIXTURE`, `QWEN_PREPROCESS_FIXTURE`, `QV_GH`, `QV_GW`, `QWEN_MID_SYSTEM_MODEL_DIR` | `qwen_vision.zig`, `chat.zig` | Qwen vision/preprocess oracle inputs and a chat-template pack | test | yes: server-tool-calling, tests/CLAUDE |
| `SUSHI_FORCE_BATCHED` | `scheduler.zig:1209` | `1` routes single-slot ticks through the batched kernel (`tests/test_batched_equivalence.sh`) | boot | yes: tests/CLAUDE |
| `SUSHI_MEMORY_AUDIT_MODEL`, `SUSHI_TOKENIZER_PARITY_FIXTURE` | `server.zig`, `model_load.zig`, `tokenizer.zig` | memory-bill audit pack and tokenizer parity fixture | test | no |
| `SUSHI_MLX_FAULT_CHUNK`, `SUSHI_MLX_FAULT_STEP` | `mlx.zig` | `<n>` latches a synthetic Metal OOM at the n-th prefill-chunk check / decode step (`tests/test_mlx_error_recovery.sh`) | once | yes: tests/CLAUDE |
| `SUSHI_MODELS_DIR` | `test_models.zig:7` | root for real-pack tests (`${SUSHI_MODELS_DIR:-~/.sushi/models}`); the engine itself does not read it | test | yes: tests/CLAUDE |
| `SUSHI_NO_SUCH_DIAG_SWITCH_PROBE` | `transformer.zig` | unit probe that `diagEnvOnCached` honors `0`/empty/set | test | no |
| `SUSHI_UPDATE_API` | `update.zig:28` | replaces `https://api.github.com` for the update check (`tests/test_self_update.sh`) | per update check | yes: server-lifecycle, tests/CLAUDE |

## Paths and config (8)

A path, a credential or an output location.

| var | where | default → effect | read | documented |
|---|---|---|---|---|
| `HF_TOKEN`, `HF_TOKEN_PATH`, `HF_HOME` | `cli.zig` `HfEnv` | Bearer token for `sushi pull` and `sushi list`: `HF_TOKEN`, else the file `HF_TOKEN_PATH`, else `$HF_HOME/token`, else `~/.cache/huggingface/token` (the standard Hugging Face names); curl reads it from a 0600 temp header file, never argv | per call | yes: usage |
| `SUSHI_HIDDEN_OUT`, `SUSHI_HIDDEN_SPOOL_BYTES`, `SUSHI_HIDDEN_SPOOL_FIRST` | `hidden_capture.zig` | `<abs dir>` enables hidden-state block-boundary capture; spool budget and first window index of its spool mode | boot | yes: engine-expert-streaming, glm5-arch, quality-kld |
| `SUSHI_IMATRIX_OUT` | `imatrix.zig:34` | `<abs>.safetensors` enables expert-imatrix capture (also the file a DeepSeek-V4.1 layer-major capture writes it to) | boot | yes: engine-expert-streaming, quality-kld |
| `SUSHI_NGRAM_BF16_DIR` | `model.zig:1086` | serves a Flash-Next pack with the original bf16 n-gram table (KLD isolation) | load | yes: engine-expert-streaming, quality-kld, qwen4-arch |

## Flag twins that are not equivalents (2)

Kept because the launch flag cannot express what the env does.

| var | where | default → effect | read | documented |
|---|---|---|---|---|
| `SUSHI_MTP_ADAPTIVE` | `mtp_policy.zig:467` | `0` selects the windowed depth controller, a different algorithm that can also disable MTP; `--mtp-min-depth n --mtp-max-depth n` pins a depth but cannot express it | once | yes: engine-mtp, tests/CLAUDE, --help |
| `SUSHI_PREFIX_CACHE_DIR` | `disk_placement.zig:484` | absolute directory for the SSD prefix cache (default under `~/.sushi`); `--prefix-cache-disk` sets a size, not a directory | boot | yes: engine-prefix-cache, tests/CLAUDE, --help |

## Reads to fix

- *Stale scripts*: benches that export `SUSHI_ROUND_COST_PERSIST=0` set the default (persistence is opt-in); `SUSHI_QSA_NAX_PRECISION` is script-side only.

## Removed 2026-10-11

Cut from the code with the arm only they reached; the defaults are unchanged.

- *Speculation* (MTP planner, drafting, DFlash): `SUSHI_MTP_BATCH_CORR`, `_EARLY_DISPATCH`, `_PREDRAFT`, `_EV_SEED`, `_EXT_DRY`, `_LIVECOST`, `_REGIME`, `_BATCHED`, `_GROUP_PLANNER`, `_DRAFT_RERANK`, `_KV_ONLY`, `_QWEN4_PROFILE`, `_ADAPTIVE_CONFIRM`, `_ADAPTIVE_MARGIN`, `_ADAPTIVE_MIN_KV`, `_DRAFT_GREEDY`, `_DRAFT_TEMP`, `_EV_COSTS`; `SPEC_GATE_WARMUP`, `SPEC_YIELD_WARMUP`, `DFLASH_GATE_WARMUP`, `SUSHI_DFLASH_SELECTOR`, `SUSHI_DFLASH_MARKOV` (a sidecar's Markov chain always runs).
- *Attention, QSA and KV*: `SUSHI_QSA_FUSED`, `_GATHER`, `_DECODE_GATHER`, `_BATCHED_GATHER`, `_VERIFY_GATHER`, `_SELECT_KERNEL`, `_SELECT_SPLIT`, `_SCORE_FUSED`, `_ATTN_KERNEL`, `_HISTORY_SHARE`, `_ATTN_BK`, `_ATTN_NSPLIT`, `_ATTN_MIN_S`, `_GATHER_BK`, `_GATHER_MIN_KV`, `_VERIFY_GATHER_MIN_KV`, `_SELECT_TG`; `SUSHI_KV_GROW`, `SUSHI_KV_RESERVE`, `SUSHI_ATTN_PD_NAX`, `SUSHI_NAX_SDPA`, `SUSHI_QK_NORM_ROPE_FUSED`, `SUSHI_SDPA_SPLIT`, `SUSHI_SLIDING_BLOCK_TRIM`, `SUSHI_FUSED_256_BUDGET`, `SUSHI_FUSED_256_CAUSAL`.
- *Qwen4 GDN, hyper-connections and PLE*: `SUSHI_GDN_DECODE_FUSED`, `_PREFILL_FUSED`, `_PREWORK`, `_BLOCKED`, `_BLOCK_T`; `SUSHI_HC_FUSED`, `SUSHI_HC_PREFILL`; `QWEN4_NO_HCFUSE`, `QWEN4_PLE_PREFETCH`, `QWEN4_PLE_PREFETCH_PREFILL_MIN_KV`.
- *MoE, verify-qmm and expert kernels*: `SUSHI_EXPERT_BF16_KERNELS`, `SUSHI_MOE_DOWN_REDUCE_FUSED`, `_GATEUP_FUSED`, `_ROUTER_FUSED`, `_ROWS_FUSED`, `_GATHER_DECODE`; `SUSHI_SWIGLU_FUSED`, `SUSHI_PREFILL_DQ_GEMM`; `SUSHI_VERIFY_QMM`, `_MIXED_PLAIN`, `_NAX_MIXED`, `_NAX_MIN_M`; `SUSHI_VQMM_BN`, `SUSHI_VQMM_KPARTS`.
- *Prefix cache, prefill and load policy*: `SUSHI_PREFIX_SSD_FIRST`, `SUSHI_RESTORE_MOVE`, `SUSHI_SSD_CHUNK_SHARE`, `SUSHI_PREFILL_CHUNK_CTX_BAR`, `_PER_REQUEST`, `_ADAPTIVE`, `SUSHI_F16_NARROW_1D`, `SUSHI_MLX_ERROR_LATCH`, `SUSHI_SUPPRESS_RESERVED`.
- *Fixed policy, no knob*: `SUSHI_CACHE_LIMIT`, `SUSHI_WIRED`, `SUSHI_WIRED_SLACK_MB` (the policy is always the one the code names).
- *A launch flag is the control*: `SUSHI_DFLASH` (`--no-drafter`), `SUSHI_MTP_TOKENV3` and `SUSHI_MTP_TYPICAL` (`--mtp-tokenv3`, `--mtp-typical`), `SUSHI_PREFILL_CHUNK` (`--prefill-chunk`; the env also bypassed the safety cap), `SUSHI_PREFILL_DECODE_SHARE`, `SUSHI_PREFILL_TRACE`, `SUSHI_NO_UPDATE_CHECK`, `SUSHI_DECODE_ATTN_QUANT` (the env outranked the flag).
- *Duplicates of another env*: `SUSHI_KVQ_FORCE_SPLITK`, `SUSHI_MOE_GATHER_QMV_OFF`, `SUSHI_VERIFY_QMM_NAX`.
- *Rejected or never-adopted experiments*: `SUSHI_DFLASH_CHOOSER`, `_DRAFT_HEAD_BITS`, `_SAMPLED_DRAFTS`, `SUSHI_MTP_ADAPTIVE_REENTRY_TOKENS`, `SUSHI_MTP_BATCHED_QWEN4`, `SUSHI_FUSED_QKV`, `SUSHI_LMHEAD_PRUNE`, `_PRUNE_TRACE`, `SUSHI_LMHEAD_UBENCH` (its meter), `SUSHI_MOE_BATCHED_DECODE`, `SUSHI_MOE_VERIFY_PAIRED_GU`, `SUSHI_VERIFY_QMM_CROSSROW`, `SUSHI_ADD_RMSNORM_FUSED` (the gated entry only; the kernel stays, MiMo wires it), `SUSHI_COMPILE_FORWARD`.
- *Decode-attn-quant*: `SUSHI_DECODE_ATTN_QUANT_NVFP4_FROM` with the nvfp4 tail, then the whole int8 requantization of dense attention at decode (it engaged only on the BF16 source checkpoint).
- *Kill switches of adopted arms*: `SUSHI_FUSED_256` (the hd-256 `sushi_attn_p256` kernel; the composed path still serves declined shapes and is the tests' parity oracle) and `SUSHI_SSD_WRITE_THROUGH` (the prefill write-through to the SSD tier, always on once its gates pass).
- *Measured, then adopted or dropped*: `SUSHI_GDN_PIPELINED` (the software-pipelined GDN prefill kernel is the NAX Qwen route, +3.8% prefill at 32K) and `SUSHI_QSA_ATTN_BALANCED` (the balanced split-K arm: no gain, not bit-identical; [qwen4-perf](qwen4-perf.md#gdn-pipelined)).
- *Dead or leftover*: `CODEX_SANDBOX`, `HY3_MODEL_DIR`, `MINICPM5_MODEL_DIR`, `PERF_REPORT`, `REAL_BLOB`, `SUSHI_DSV4_DSPARK`, and the `--dspark` flag that only set it.

## Removed 2026-10-26

- *The Qwen3.5/3.6 MTP sidecar loader*: `SUSHI_MTP_HEAD_QUANT_BITS` (requant of a dense sidecar head's trunk) went with `MtpModel`, which no served architecture reaches.

## Removed 2026-10-13

- *Kill switches of a dead arm*: `SUSHI_KV_ATTN_KERNEL` and `SUSHI_KV_ATTN_VERIFY` (the SIMD packed decode / verify kernels; Flash-Next's gqa 12 x dk 256 never fit their threadgroup budget, and no other served model reached them).
- *Packed-read switches MiMo never moves at its default*: `SUSHI_KV_ATTN_FUSED` (`0` sent every MiMo global decode to the whole-cache rebuild and turned batched decode off) and `SUSHI_KV_ATTN_MIN_TK` (replaced the 4096-key floors; no test set it). The floors are constants now.

## Tooling-only variables

Read by `scripts/` and `tests/` (shell and Python), not by the engine. Engine vars the scripts export are in the tables above.

| var | used by | meaning |
|---|---|---|
| `SUSHI_BINARY`, `SUSHI_BIN` | ~50 `tests/*.sh`, bench scripts | the binary under test; a control binary must sit in `zig-out/bin/` for its rpath |
| `SUSHI_MODELS_DIR` | ~100 scripts, `src/test_models.zig` | pack root, `${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/<pack>`; the engine does not read it |
| `SUSHI_TEST_EXTRA_ARGS` | `test_mtp_equivalence.sh`, `test_soak_24h.sh`, 2 more | extra server flags appended to every boot |
| `QWEN4_MODEL`, `QWEN_MODEL`, `QWEN_VISION_MODEL`, `MIMO_MEDIA_MODEL`, `MIMO_STREAM_MODEL`, `MIMO_SSD_BUDGET_GB`, `GLM_MODEL`, `MTP_ADAPTIVE_MODEL`, `MTP_BATCHED_MODEL`, `MLX_HYBRID_MODEL`, `DFLASH_TEST_MODEL`, `DFLASH_TEST_DRAFTER` | per-script live tests | a per-script pack override that wins over `SUSHI_MODELS_DIR` |
| `DFLASH_TEST_PORT`, `DFLASH_ON_REQS`, `QWEN4_AB_PORT`, `QWEN4_AB_OUT`, `BENCH_MODEL`, `BENCH_OUT`, `BENCH_LABEL` | `test_dflash.sh`, `qwen4_ab.sh`, bench scripts | ports, labels and output roots |
| `GPU_LOCK_DIR`, `GPU_LOCK_OWNER`, `GPU_LOCK_POLL_S` | `scripts/gpu-lock.sh`, scripts that take the lock | lock directory, holder name, poll interval |
| `SUSHI_RUNS_DIR`, `RUNS_DIR` | bench scripts | run output roots |
| `MTP_FORCE_ENABLE`, `MTP_EXPECT_AUTO_DEPTH`, `MTP_EXPECT_AUTO_PROFILE`, `MTP_TEST_OUTPUT_DIR`, `MIMO_BATCH_OUTPUT_DIR`, `SUSHI_PROMPT_MODEL`, `SUSHI_PROMPT_OUTPUT_DIR`, `MIMO_CLI_TEST_HOME`, `MIMO_CLI_TEST_TIMEOUT` | MTP, batched and CLI tests | expected profile/depth pins, output dirs, private HOME |
| `SUSHI_QSA_NAX_PRECISION` | `qsa_nax_precision.py` | verbose mode of the Python precision harness; not read by the engine |
| `QWEN4_REF_DTYPE`, `QWEN4_TEACHER_FIXTURE`, `MIMO_V2_REF`, `MIMO_V2_FIXTURE_DEBUG`, `HF_HUB_OFFLINE`, `PYTHONPATH`, `KLD_JSON`, `QWEN_PREPROCESS_FIXTURE`-family (`QWEN_PREPROCESS_MIN_PIXELS`, `_MAX_PIXELS`, `_DOWNSCALE_*`) | `tests/dump_*_fixtures.py`, parity scripts | oracle dumpers and preprocessing parity inputs |
| `MLX_SRC`, `MLX_SHA`, `MLXC_SRC`, `MLXC_SHA`, `MLXC_PATCH`, `MLX_DEPLOYMENT_TARGET`, `MLX_STAGE`, `ZIG`, `ZIG_VERSION`, `GH_REPO` | `scripts/build-mlx.sh`, `fetch-zig.sh`, `release.sh` | pinned build inputs and release plumbing |
