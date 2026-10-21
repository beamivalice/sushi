# Architecture: Qwen3.8-Flash-Next (`qwen4_exp`)

How the engine serves Qwen3.8-Flash-Next: the three blocks it adds around the qwen3_5 GDN+MoE trunk, which
module state is shared, where the oracle lives, and the facts a change to this arch must respect. Read this before
touching `forwardQwen4With`, `src/qwen4_exp.zig` or `src/hc_prefill.zig`.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-exl3-experts](engine-exl3-experts.md),
[engine-expert-streaming](engine-expert-streaming.md), [engine-mtp](engine-mtp.md),
[qwen4-qsa-long-context](qwen4-qsa-long-context.md), [perf-baselines](perf-baselines.md),
[quality-kld](quality-kld.md).

## The model

125B-A6B MoE + 51B n-gram table + 4B MTP head. Dispatch is on `config.json` `model_type`; a qwen4_exp checkpoint
is NOT a qwen3_5 pack: three blocks sit around the qwen3_5 GDN+MoE trunk. 48 layers, 512 routed experts, top-k 10,
hidden 2560, expert intermediate 640.

## Code map

| File | Role |
|---|---|
| `src/qwen4_forward.zig` / `src/qwen4_hc.zig` / `src/qwen4_qsa.zig` | Flash Next trunk = `forwardQwen4With` (attention, GDN, native MTP, decode/verify rows) / hyper-connections + PLE / QSA mask and block selection |
| `src/transformer.zig` | shared `Transformer`, `KVCache`, `ForwardCtx`, arch dispatch; EXL3 MoE = `moeExl3` |
| `src/qwen4_exp.zig` | Host side: n-gram hash (splitmix multipliers, per-head primes, eos-segment shifts) + the mmapped `ngram_table.bin` row gather (2/3/4/5/6/8-bit or raw bf16) + `PrefetchPool`/`startWarm` |
| `src/hc_prefill.zig` | Fused hyper-connection norm/mix prefill kernels (chunk width as a scalar INPUT, never a template) |
| `src/mtp.zig` | The native MTP head (`Qwen4Mtp`), see [engine-mtp](engine-mtp.md) |

## Trunk rules

- **Trunk**: 4 hyper-connection streams (`hcRead`/`hcWrite`, norms folded by the converter), n-gram PLE at layer 1
  (host gather from `ngram_table.bin`, never resident), QSA sparse attention past 2048 tokens (`qsaMask`),
  `hyper_connection_mixer` replaces `model.norm`. The residual stream is bf16 like the checkpoint; the f32 fixture
  is the MATH oracle (`QWEN4_STREAM_F32=1`).
- **Module state is READ-ONLY**: text slots batch-decode, prefix cache ON. A resident pack runs the slots as rows of one
  forward (`forwardQwen4DecodeRows`), each byte-identical to its solo tick (hermetic: `qwen4 per-row batched decode == solo
  ticks bit for bit`); a streamed pack keeps the padded `forwardMoeBatchedDecode`. Vision
  (Qwen3-VL tower, `model.visual.` prefix, added at conversion) decodes serially; a streamed load carries it
  only under `--vision` ([engine-expert-streaming](engine-expert-streaming.md#vision)).
- **The deferred PLE leaf is filled before anything evaluates the build**: see [engine-mtp](engine-mtp.md#ple-defer).
  A host token read inside the graph build serialized the build with the GPU.
- **Decode kernels**: the fused hc read is LATENCY-bound (`hcWrite` DEFERS into the next
  read); HC + GDN prefill fusions take the chunk WIDTH as a scalar input. Each of the 36 GDN layers decodes and
  verifies (S <= 8) in two dispatches: `gdn_decode.step` (`src/gdn_decode.zig`), then the norm-gate. Details in
  [engine-kernels](engine-kernels.md).
- **A GDN trunk's `KVCache.step` is 0 forever**: see [engine-kv-cache](engine-kv-cache.md#gdn).

<a id="vision-tower"></a>
## Vision tower

- **The tower evaluates after every block and at its output** (`QwenVision.eval_per_block`): a lazy 27-block graph
  also held every block's transposed weights (~1 GB at any size) and more than one block's score sheets. The encode
  bill (`encodeScratchBytes`: 32 MB + 1.3 x one f32 score sheet + 107 KB per patch) covers the per-block peak by
  1.26x or more at every measured size; `qwen vision ubench` asserts bill >= peak.
- Measured on main `68001a57` plus this change, with `qwen vision ubench`: Sushi-3bpw tower only, random pixels, 5 reps with the lazy and
  per-block arms interleaved, `taskpolicy -a`, GPU lock held. Times are best-of-5 and CONTENDED: a live Qwen server
  was decoding on the same GPU. Peaks were identical across two runs.

  | grid (patches) | image | lazy time / peak | per-block time / peak | old N² bill | bill |
  |---|---|---|---|---|---|
  | 14x14 (196) | small | 18.3 ms / 1013 MB | 23.8 ms / 21 MB | 14 MB | 58 MB |
  | 46x82 (3772) | 1920x1080 at the packs' 1 Mpx bound | 569 ms / 3280 MB | 557 ms / 1225 MB | 2.8 GB | 1621 MB |
  | 68x120 (8160) | 1920x1080 at the 1536² cap | 2169 ms / 9155 MB | 2122 ms / 4923 MB | 13.1 GB | 6446 MB |
  | 96x96 (9216) | the 1536² cap | 2630 ms / 6112 MB | 2640 ms / 6112 MB | 16.6 GB | 8086 MB |

- The served packs ship no processor config, so their bound is the 1,003,520-pixel default: a 1920x1080 screenshot is
  46x82 patches (943 tokens). Only a checkpoint that declares a larger bound reaches the 1536² cap.

## The n-gram table (PLE)

- `mx.quantize` packs DENSELY (element i at bit offset `i*bits`, straddling words at 3/5/6 bits; `dequantRow` tested
  at every width) or raw bf16 (bits-16 arm).
- A random read into a cold 32 GB mmap is a serial SSD fault, so `gather` rides a `PrefetchPool`
  and `startWarm` preads the table at load (`SUSHI_NGRAM_WARM=0`).
- A wide prefill gather walks a resident table serially below `PREFILL_PREFETCH_MIN_KV` (256k) and pools past it;
  `QWEN4_PLE_PREFETCH_PREFILL=0|1` forces either arm, and the one-shot `PLE prefill gather:` line names the reason.
- Which arm a wide gather takes is MEASURED on the table, not predicted from RAM: `calibrateArm` reads 128 random rows
  both ways, on DISJOINT sets so neither arm warms the other's pages, before the warm thread faults the table in, and
  takes the pool only when it wins by 20%; SSD and page-cache state determine the choice on each load.
- `SUSHI_NGRAM_WARM` preads the whole table, and the residency cap (`ngramCacheLimit` = half of RAM) declines any
  table that cannot be held, bf16 or 4-bit (a 32 GB Mac reads the 4-bit table by row). Calibration and residency measurements: [perf-baselines](qwen4-perf.md#ngram-arm).
- A table under the cap is a page-cache claim from `startWarm` to `close` (`page_cache_claim`), billed as GPU memory
  by the hot cache's unnamed budget ([engine-prefix-cache](engine-prefix-cache.md#budget)): wired KV evicts it.
- The n-gram hash's eos is the TEXT config's (`ngram_eos`).
- `SUSHI_NGRAM_BF16_DIR=<hf checkpoint>` serves any pack with the ORIGINAL bf16 n-gram table, so `kld compare`
  isolates the PLE table's cost.

<a id="dflash2"></a>
## DFlash2 drafter (opt-in)

`incoai/Qwen3.8-Flash-Next-DFlash2` (Apache-2.0, 5 layers, block 8) drafts a block per round from five trunk taps; the
trunk verifies it in one forward. MTP is the default. The drafter runs only when `--drafter <dir>` names it and the
request does not opt out (`--no-drafter`, `enable_drafter:false`); a drafter beside the pack is never loaded and costs no
memory. The boot log says which decodes (`[spec] Qwen3.8-Flash-Next decodes with MTP (DFlash2: --drafter <path>)` or
`... with DFlash2 (2..8 rows per round; ...)`). Fetch it with
`hf download incoai/Qwen3.8-Flash-Next-DFlash2 --local-dir <dir>`. It is the generic `Generator.nextDflash` path, not GLM's
native tree.

- **Taps**: the COMPLETED outputs of layers 5, 15, 25, 35 and 45 (zero-based), each the mean over the four hyper-connection
  streams (`CaptureLayers.hc_mean`), 5 x 2560 features. The raw four-stream tensor is the wrong tap (4x the width).
- **Memory**: the drafter loads A8 (8-bit affine) at load; there is no on-disk cache (A4 saved 1.3% of a round). On a
  streamed load its bytes join the trunk in the ledger, so the expert cache is cut from what remains.
- **Serving**: all-sliding drafter (`sliding_serving`: window trim, block tail, cached mask), the path GLM serves. The
  drafter's context grows by the committed taps only.
- **Verify**: ONE batched forward of `[t1, drafts]` with per-position GDN capture; a partial accept rolls GDN, PLE and
  QSA state back through `ssmRollbackFromCapture`, shared with the MTP verify (rows are decode-identical up to 9).
  Greedy DFlash2 output equals greedy serial output byte for byte (`tests/test_dflash2_qwen.sh`: six prompts, stream ==
  non-stream, opt-out, and that a drafter beside the pack is ignored; run on Sushi-2bpw, 2.6bpw, 4bpw and an STQ SPEC1
  pack, and streamed at `--ssd-budget-gb 32` on 2.6bpw).
- **Rows**: a verify row is bytes, not dispatches (about 3.3 ms per row on Sushi-2.6bpw). The drafter drafts a block of
  `--drafter-max-depth` - 1 (default 8 rows) and the round verifies only as many rows as `dflash_policy.zig` finds worth
  their cost: acceptance of each draft is learned online from the selector's confidence, and the rows are the argmax of
  expected tokens per ladder cost, never below `--drafter-min-depth` (default 2). Prose and thinking settle near 2-3 rows,
  code, JSON and copy take 8. Equal min and max pin one row count.
- **Yield gate**: a request whose DFlash rounds cost more per token than its serial decode (5% margin, after 8 rounds)
  falls back to plain ticks for the rest of the request. The serial price is the round-cost table's width-0 cell of the
  request's context bucket; without a sample the gate asks for the ladder's break-even yield at the request's MEAN
  verified rows, never the block's.
- **Bill**: `dflashRequestBytes` (window x 4 + tap and encoder rows of one prefill chunk + 64 MiB) joins the request
  bill in the commit that loads the drafter; the weights join the load bill.
- **Lessons**: an accepted-per-round floor that is right for a dense 27B target disabled rounds that beat serial here
  (1.8 drafts against 1.37 needed at 4 rows): gates compare ms per token. A fixed 4-row block lost to MTP at 4k and 128k;
  adaptive rows won 6-20% on Sushi-2.6bpw at every rung and tie on Sushi-4bpw, where novel prose runs 11% under MTP
  (numbers and settings: [qwen4-perf](qwen4-perf.md#dflash2)). That is why MTP stays the default.

## Oracle and fixtures

- `tests/dump_qwen4_exp_fixtures.py` is the oracle dumper. HF `hidden_states[i]` is the INPUT of layer i: compare
  layer-i output with stream_{i+1}.
- The converters, allocators and imatrix drivers live in the private converter repo; the format they owe us is
  [pack-format](pack-format.md).
- **Ties**: a tiny MoE oracle ties everywhere and the tie RATE is scale-invariant (relu leaves exact-zero block
  scores). The fixture dumps the reference's OWN margins; `Qwen4Ties` acquits under 3% by those, never by our
  output. Selection coverage for a k < E fixture is the MTP head's one MoE layer (`--topk 2`, `route_gap`).

## Packs on this box

| pack | directory under the models root | notes |
|---|---|---|
| MCG K3 w15 (served target) | `Qwen3.8-Flash-Next-Sushi-3bpw` | our MCG experts (window 15) |
| affine 4/8 control | `Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit` | the control column in every table |
| bf16 source | `Qwen/Qwen3.8-Flash-Next` | the KLD teacher; streams (335 GB) |

Bytes read per decoded token on the MCG K3 pack: trunk 4.01 GB (affine-8), lm_head 0.66, routed 0.91 = 5.58 GB,
a 107 tok/s ceiling at 600 GB/s against 61.8 measured (58%). Numbers and their sources: [perf-baselines](perf-baselines.md).
