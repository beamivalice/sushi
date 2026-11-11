# Source map (served path)

Which file under `src/` holds what, and the doc to read before changing it. A new served-path file adds its row here.

| File | Role | Doc |
|---|---|---|
| `main.zig` / `cli.zig` | entry, flags, subcommands (`run/pull/list/serve/launch/kld/convert`), REPL | server-lifecycle |
| `repl_tools.zig` | `run` REPL client tools (web, confined files, images; write needs `/edit`) | server-http-apis |
| `server_state.zig` / `http.zig` / `request_parse.zig` | the server's process globals; the HTTP transport (bind, response writers, body log, error body, JSON escape); request reading (MLX-free, host tier) | server-lifecycle |
| `help.zig` / `arch/flags.zig` / `arch/packs.zig` / `arch/serve.zig` / `<arch>/<arch>_flags.zig` | `--help` assembled from the enabled archs; their flags, subcommands, launch refusals and serve hooks; the pack alias table | engine-arch-seam |
| `server.zig` / `responses.zig` / `ws.zig` / `stop_sequences.zig` / `http_conn` / `request_policy` / `request_media` / `models_api` / `logprobs` / `generation_stream` / `anthropic` / `completions` / `responses_api` | all HTTP: `/v1/*`, `/metrics`, WS, `--api-key`, Responses store | server-http-apis |
| `chat.zig` / `think_tags` / `tool_calls` / `tool_schema` | chat templates, thinking tags, tool-call parse/repair/coercion | server-tool-calling |
| `arch/chat.zig` / `<arch>/<arch>_chat.zig` | each arch's MLX-free chat face: effort table, pinned template and image atom, pre-tokenizer; the host tier reads them through `registry.Chat` | engine-arch-seam |
| `reasoning_protocol.zig` / `json_schema.zig` / `json_grammar.zig` / `token_mask.zig` / `regex.zig` | constrained decoding | server-http-apis |
| `launch.zig` | `sushi launch <agent>` | server-http-apis |
| `scheduler.zig` / `generate.zig` / `degenerate.zig` / `mtp_plan.zig` / `prefill_plan.zig` / `sampler.zig` / `model_load` / `load_clock.zig` / `slot_admission` / `mtp_tick` / `slot_outcome` / `vision_queue` / `mtp_policy` / `mtp_round` | slots, inference thread, batching, admission, load bills, the `[load]` phase clock; generation; MTP policy/rounds/plan; stop loop, prefill plan, sampler | server-lifecycle |
| `settings_resolve.zig` | the one resolver, flag > `model-settings.json` > default (`Setting`, `resolve`), the launch flags no load params carry, and the table test that pins every key (MLX-free, host tier) | server-lifecycle |
| `model.zig` / `effort` / `model_settings.zig` / `model_discovery.zig` / `model_registry.zig` | config + weights, effort words and the per-family effort tables, per-model settings, discovery, registry | server-lifecycle, server-http-apis |
| `bills.zig` / `bills_fixtures_test.zig` / `gpu_mem.zig` / `<arch>/<arch>_bill.zig` | every bill, admission, context and chunk function (MLX-free, host tier) / the served shapes as test configs / the live GPU readers, the only place a bill's inputs touch the device / each arch's own bill terms behind its `bill` decl | engine-memory-admission, engine-arch-seam |
| `arch/sched.zig` / `<arch>/<arch>_sched.zig` | the scheduler's dispatch to each arch: batching, request state, commit, restore, write-through, thought points; `glm5_rows_tick` and `dsv41_thought_point` hold the code those faces call | server-lifecycle, engine-arch-seam |
| `arch/registry.zig` / `arch/*.zig` / `arch/handle.zig` / `arch_host.zig` / `<arch>/arch.zig` | the only arch names outside a folder, `inline else` dispatch per area, the loaded model's handle and every engine read of it, the engine services an arch may import, each arch's public face | engine-arch-seam |
| `pack_index.zig` | a pack's safetensors index from its shard headers, and the EXL3 routed-bank checks every arch's pack takes at load (`validateExl3Pack`) | engine-exl3-experts |
| `trunk.zig` | `Trunk`: embeddings, lm head, rope/M-RoPE/YaRN tables, compiled closures, the MoE family, eval cadence, the warmup scaffold | arch-*, engine-* |
| `moe_shared_fold.zig` | the gated shared expert folded into the routed EXL3 launches, for any trunk MoE layer that gates one | engine-exl3-experts, qwen4-perf |
| `model_handle.zig` | `ForwardCtx`, `CaptureLayers`, the loaded-model operations that dispatch through the registry (`deinit`, `resetCache`, `warmup`, `forwardWith`, ...) | arch-* |
| `metal.zig` / `metal_mlx.zig` | Metal from Zig (device, pipelines, buffers, passes, fences); the Metal buffer behind an MLX array | engine-own-forward |
| `tensor/core.zig` / `route.zig` / `ops.zig` / `kernel.zig` | the engine of our own behind `mlx.zig`: tensors, stores and the pool, passes, lending to and importing from MLX; the mlx-c names answered by the engine and the tests against MLX; views, elementwise and gather ops; a model's custom kernels | engine-own-forward |
| `pipeline_manifest.zig` / `tensor/pipeline_registry.zig` / `tensor/pipeline_cache.zig` / `dsv41/dsv41_pipelines.zig` | the engine's pipeline manifest: file format, key and directory (host tier); the recipe of every pipeline made and those made ahead; making recipes on worker threads; DeepSeek's key, the load's cases and the `[pipelines]` lines | engine-own-forward |
| `tensor/stock.zig` / `matmul.zig` / `reduce.zig` / `sort.zig` / `scan.zig` / `random.zig` | MLX's compiled kernels launched by the engine with MLX's dispatch: RMS norm, softmax, dequantize; matmul; sum, max, argmax; sort and argsort; cumulative sum; random bits, uniform and categorical as MLX composes them | engine-own-forward |
| `mlx_util.zig` / `trunk_helpers_test.zig` | MLX helpers with no model knowledge (scalars, owned copy, masks); test-only fixtures shared by the trunk, KV, attention and MoE tests | engine-mlx-gotchas |
| `<arch>/<arch>_state.zig` | what an arch owns beyond the trunk: load, `deinit`, cache reset | arch-* |
| `kv_cache.zig` / `ssm_cache.zig` / `layer_weights` / `forward_diag` / `test_support` | KV/SSM caches, quant resolution, layer weights, `diagEnvOn` | arch-*, engine-* |
| `nax_gate` / `attn_kernels` / `verify_qmm` / `verify_moe` / `moe_router` / `moe_gqmv` / `moe_stream_compute` | NAX/OS gates; shared attention; verify QMM and MoE; router; gather-QMV; streamed-expert compute | engine-kernels |
| `qwen4/qwen4_forward.zig` / `qwen4_hc.zig` / `qwen4_qsa.zig` | Flash-Next forward, GDN, MTP, verify / hyper-connections, PLE / QSA | qwen4-arch, engine-mtp |
| `mimo2/mimo_forward.zig` / `mimo2_shortlist.zig` | MiMo forward, attention arms, MoE dispatch, batched decode; the greedy lm-head shortlist | mimo2-arch, engine-mtp |
| `glm5/glm5_*.zig` | GLM forward, MLA/KDA attention, router, DFlash2 verify, kernels, prefix state, vision | glm5-* |
| `dsv41/dsv41_*.zig` / `row_grammar.zig` / `tokenizer_deepseek.zig` | DeepSeek-V4.1 loader, forward, QAT, Engram, stream hooks, prefill borrow, prefix state, vision, DSpark rounds, per-row grammar masks, pre-tokenizer; `dsv41/fixtures/` its Engram token map | dsv41-arch, dsv41-memory |
| `dspark_rows.zig` / `expert_usage.zig` / `weights_where.zig` | DSpark verify width and the generator's DSpark types (a drafting method like DFlash); the pack's `expert_usage.json` and the `--drop-expert` set it ranks; the keep-predicate shard loader | dsv41-arch, pack-format |
| `qwen4/qwen4_exp.zig` / `hc_prefill.zig` | Flash-Next n-gram host side, fused HC prefill | qwen4-arch |
| `qwen4/gdn_decode.zig` | fused GDN decode/verify step (one dispatch) | engine-kernels |
| `mimo2/mimo_source.zig` / `mimo2/mimo2_load.zig` / `fp8_block.zig` | MiMo source headers, FP8 trunk + GEMV, rank-local QKV; its own load (weights, resident bytes, heads, spec warmup) | mimo2-arch |
| `expert_store.zig` / `expert_drop.zig` / `exl3/pack.zig` | expert layouts and `QuantStore` (byte spans of every routed bank), the `--drop-expert` set, EXL3/STQ pack reading (std only); `expert_quant.zig` re-exports the store | engine-expert-streaming |
| `exl3/` (`sushi_exl3` module) | EXL3/STQ decoders and kernels, rate groups, dense linears; `exl3/trunk.zig` the trellis trunk math (matmul, matmulGeo, fuse, sharedProject) and `linear.Geo`, a trunk bank's shapes and launch configs read once; no file or pub name there names an architecture; `exl3/host_min.zig` is the host `exl3-host-check` compiles it against; the benches are the separate module `exl3_ubench/` | engine-exl3-experts |
| `linear.zig` | `Linear`: a trunk projection as dense, affine, FP8 block or trellis, bound once from the weight triple (`layer_weights.bindLayer`) and applied with a tag switch by `Trunk.qmatmul` | engine-kernels |
| `trunk_stq.zig` | the weight-map binding of the trellis trunk banks (`load`, `drop`); re-exports `exl3/trunk.zig` | engine-exl3-experts |
| `encoder/` | `sushi convert exl3`: trellis search, source readers, trunk/shard writers, calibration; imports `sushi_exl3` | encoder, engine-exl3-encoder |
| `expert_stream.zig` / `prefetch_plan.zig` / `expert_io.zig` / `expert_bf16_kernels.zig` / `imatrix.zig` | SSD expert streaming | engine-expert-streaming |
| `mtp*.zig` / `round_cost.zig` | MTP head, acceptance, planner, cost table | engine-mtp |
| `kv_quant.zig` | quantized KV contract | engine-kv-cache |
| `prefix_cache.zig` / `kv_disk_cache.zig` / `disk_placement.zig` / `kv_disk_writer.zig` / `restore_dump.zig` / `arch/prefix.zig` | prefix cache, SSD tier, budget, roots; the dispatch to each arch's `<arch>_prefix.zig` | engine-prefix-cache |
| `kv_disk_fs.zig` | the SSD tier's file operations (`DiskTier.fs`) on its two backends, plain files or the content-addressed layer (`SUSHI_PREFIX_FS`), tensor files with each tensor's row map, and the tests' file helpers | engine-prefix-cache |
| `prefix_vfs.zig` | content-addressed virtual files under the SSD tier: block packs, journal, snapshot, POSIX-shaped `stat`/`link`/`rename`; host tier | engine-prefix-cache |
| `prefix_vfs_tensors.zig` | tensor files on the VFS: arrays staged as row-group and fixed-part blocks, loaded back as `mlx_load_safetensors` maps, the safetensors bytes regenerated | engine-prefix-cache |
| `sched_policy.zig` | the scheduler's decisions (tick buckets and caps, admission order, slot end states, interleave and idle arithmetic), host tier | server-lifecycle |
| `prefix_policy.zig` / `disk_format.zig` | the prefix cache's rules (thinning, lookup winner, eviction victim, spec adoption) and the SSD record's wire format (safetensors header, `meta.json`), both host tier | engine-prefix-cache |
| `mlx_kernel_keys.zig` | custom-kernel library keys for `mlx.zig` wrappers | engine-mlx-gotchas |
| `mlx_ops.zig` / `tool_root.zig` | the eager-graph `Ops` (a bounded list of MLX arrays freed together) every arch building a short graph shares / the one root of the offline tools an arch folder builds (`-D<option>`), which finds the tool in the arch's `arch.zig` `tools` | engine-mlx-gotchas, engine-arch-seam |
| `tokenizer.zig` / `tokenize_cache.zig` | BPE, special tokens, `digit_group`; prompt LRU | engine-mlx-gotchas |
| `vision.zig` / `vision_common.zig` / `arch/media.zig` / `<arch>/<arch>_media.zig` / `qwen4/qwen_vision.zig` / `mimo_vision.zig` / `glm5_vision.zig` / `mrope.zig` | media INPUT: the tower holder, preprocessing, the media dispatch and each arch's processor, Qwen3-VL, M-RoPE, MiMo-ViT, GLM towers | server-lifecycle, mimo2-arch |
| `kld.zig` / `kld_fixture.zig` / `kld_capture_util.zig` / `arch/offline.zig` | `sushi kld capture|compare`; the MLX-free fixture reader and scoring; the per-arch hooks | quality-kld |
| `bench.zig` / `bench_box.zig` | `sushi bench`: release bench client (corpus, ladder, report); fans, cooldown, power | process-measurement |
| `ubench_main.zig` / `ubench_report.zig` / `exl3_ubench/ubench.zig` / `glm5/ubench_glm5.zig` / `qwen4/ubench_qwen4.zig` | `sushi-ubench` (`zig build ubench`); each built arch's `ubench` decl is listed | process-ubench |
| `sushibar_main.zig` / `sushibar/` | `sushibar` menu bar app | server-lifecycle |
| `metrics.zig` / `status.zig` / `log.zig` | metrics, status bar, log | server-http-apis |
| `format_corpus_test.zig` / `tool_traffic_replay_test.zig` | format corpus, traffic replay | server-tool-calling |
| `arch_fixtures_test.zig` / `launch_fixtures_test.zig` / `mrope_fixtures_test.zig` / `request_media_fixtures_test.zig` / `ubench_report_fixtures_test.zig` | the arch data (templates, tower geometries, effort words, the rope golden, a fake bench registry) the shared tests of those areas read, so no shared file declares it | engine-arch-seam, testsuite |
