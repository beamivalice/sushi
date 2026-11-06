# Source map (served path)

Which file under `src/` holds what, and the doc to read before changing it. A new served-path file adds its row here.

| File | Role | Doc |
|---|---|---|
| `main.zig` / `cli.zig` | entry, flags, subcommands (`run/pull/list/serve/launch/kld/convert`), REPL | server-lifecycle |
| `repl_tools.zig` | `run` REPL client tools (web, confined files, images; write needs `/edit`) | server-http-apis |
| `server.zig` / `responses.zig` / `ws.zig` / `stop_sequences.zig` / `http_conn` / `request_policy` / `request_media` / `models_api` / `logprobs` / `generation_stream` / `anthropic` / `completions` / `responses_api` | all HTTP: `/v1/*`, `/metrics`, WS, `--api-key`, Responses store | server-http-apis |
| `chat.zig` / `think_tags` / `tool_calls` / `tool_schema` | chat templates, thinking tags, tool-call parse/repair/coercion | server-tool-calling |
| `reasoning_protocol.zig` / `json_schema.zig` / `json_grammar.zig` / `token_mask.zig` / `regex.zig` | constrained decoding | server-http-apis |
| `launch.zig` | `sushi launch <agent>` | server-http-apis |
| `scheduler.zig` / `generate.zig` / `degenerate.zig` / `mtp_plan.zig` / `prefill_plan.zig` / `sampler.zig` / `model_load` / `slot_admission` / `mtp_tick` / `slot_outcome` / `vision_queue` / `mtp_policy` / `mtp_round` | slots, inference thread, batching, admission, load bills; generation; MTP policy/rounds/plan; stop loop, prefill plan, sampler | server-lifecycle |
| `model.zig` / `effort` / `model_settings.zig` / `model_discovery.zig` / `model_registry.zig` | config + weights, effort words and the per-family effort tables, per-model settings, discovery, registry | server-lifecycle, server-http-apis |
| `transformer.zig` / `kv_cache.zig` / `ssm_cache.zig` / `layer_weights` / `forward_diag` / `test_support` | shared `Transformer`, `ForwardCtx`, arch dispatch, quant resolution, KV/SSM caches, layer weights, `diagEnvOn` | arch-*, engine-* |
| `nax_gate` / `attn_kernels` / `verify_qmm` / `verify_moe` / `moe_router` / `moe_gqmv` / `moe_stream_compute` | NAX/OS gates; shared attention; verify QMM and MoE; router; gather-QMV; streamed-expert compute | engine-kernels |
| `qwen4/qwen4_forward.zig` / `qwen4_hc.zig` / `qwen4_qsa.zig` | Flash-Next forward, GDN, MTP, verify / hyper-connections, PLE / QSA | qwen4-arch, engine-mtp |
| `mimo2/mimo_forward.zig` | MiMo forward, attention arms, MoE dispatch, batched decode | mimo2-arch |
| `glm5/glm5_*.zig` | GLM forward, MLA/KDA attention, router, DFlash2 verify, kernels, prefix state, vision | glm5-* |
| `dsv41/dsv41_*.zig` / `dspark_rows.zig` / `expert_usage.zig` / `tokenizer_deepseek.zig` | DeepSeek-V4.1 loader, forward, QAT, Engram, stream hooks, prefill borrow, prefix state, vision; DSpark width; `--drop-expert` list, pre-tokenizer | dsv41-arch, dsv41-memory |
| `qwen4/qwen4_exp.zig` / `hc_prefill.zig` | Flash-Next n-gram host side, fused HC prefill | qwen4-arch |
| `qwen4/gdn_decode.zig` | fused GDN decode/verify step (one dispatch) | engine-kernels |
| `mimo2/mimo_source.zig` / `fp8_block.zig` | MiMo source headers, FP8 trunk + GEMV, rank-local QKV, stamp check | mimo2-arch |
| `expert_quant.zig` / `exl3/` (`sushi_exl3` module) / `trunk_stq.zig` | expert layout, EXL3/STQ decoders and kernels, rate groups, dense linears; trellis trunk banks | engine-exl3-experts |
| `encoder/` | `sushi convert exl3`: trellis search, source readers, trunk/shard writers, calibration; imports `sushi_exl3` | encoder, engine-exl3-encoder |
| `expert_stream.zig` / `prefetch_plan.zig` / `expert_io.zig` / `expert_bf16_kernels.zig` / `imatrix.zig` | SSD expert streaming | engine-expert-streaming |
| `mtp*.zig` / `round_cost.zig` | MTP head, acceptance, planner, cost table | engine-mtp |
| `kv_quant.zig` | quantized KV contract | engine-kv-cache |
| `prefix_cache.zig` / `kv_disk_cache.zig` / `disk_placement.zig` / `kv_disk_writer.zig` / `restore_dump.zig` | prefix cache, SSD tier, budget, roots | engine-prefix-cache |
| `mlx_kernel_keys.zig` | custom-kernel library keys for `mlx.zig` wrappers | engine-mlx-gotchas |
| `tokenizer.zig` / `tokenize_cache.zig` | BPE, special tokens, `digit_group`; prompt LRU | engine-mlx-gotchas |
| `vision.zig` / `vision_common.zig` / `qwen4/qwen_vision.zig` / `mimo_vision.zig` / `glm5_vision.zig` / `mrope.zig` | media INPUT: preprocessing, Qwen3-VL, M-RoPE, MiMo-ViT, GLM towers | server-lifecycle, mimo2-arch |
| `kld.zig` | `sushi kld capture|compare` | quality-kld |
| `bench.zig` / `bench_box.zig` | `sushi bench`: release bench client (corpus, ladder, report); fans, cooldown, power | process-measurement |
| `ubench_main.zig` / `ubench_report.zig` / `exl3/ubench.zig` | `sushi-ubench` (`zig build ubench`) | process-ubench |
| `sushibar_main.zig` / `sushibar/` | `sushibar` menu bar app | server-lifecycle |
| `metrics.zig` / `status.zig` / `log.zig` | metrics, status bar, log | server-http-apis |
| `format_corpus_test.zig` / `tool_traffic_replay_test.zig` | format corpus, traffic replay | server-tool-calling |
