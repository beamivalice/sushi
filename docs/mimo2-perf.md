# Performance baselines: MiMo-V2.6-Flash (`mimo_v2`)

The recorded speed numbers for MiMo-V2.6-Flash (`mimo_v2`) on this box. Method, roofline, shared EXL3 kernel timings and
release ladders: [perf-baselines](perf-baselines.md); the rules there (inherit the recorded baseline, name
commit, binary stamp, QoS and lock beside every number) apply to every section here. Architecture: [mimo2-arch](mimo2-arch.md).

<a id="mimo-decode"></a>
## MiMo

Its n36 experts reached the fast decode arms only with the rate-generic readers:
[exl3-rate-generic](perf-baselines.md#exl3-rate-generic) (live decode 35 -> 61 tok/s with MTP).

<a id="mimo-longctx"></a>
### MiMo Sushi-2.25bpw long-context ladder, MTP (3d11b0f7)

One boot, one request per rung (the source-tree prompt ladder, "explain the code"), 256 tokens, T=0,
`--ctx-size 1048576 --kv-quant 8 --mtp`, info log, `taskpolicy -a`, GPU lock, fans max, 2026-09-28. Decode / prefill in
tok/s; tok/step = 1 + accepted drafts per round; stalls = rounds over twice the median at their width.

| rung | decode | prefill | TTFT | tok/step | stalls |
|---|---|---|---|---|---|
| 4k | 55.7 | 839 | 4.9 s | 2.44 | 5 (214 ms) |
| 8k | 58.9 | 1119 | 7.4 s | 2.61 | 1 |
| 16k | 56.4 | 1086 | 15.2 s | 2.42 | 0 |
| 32k | 57.0 | 949 | 34.6 s | 2.39 | 2 |
| 64k | 44.4 | 816 | 80.4 s | 2.17 | 1 |
| 128k | 43.2 | 661 | 198.5 s | 1.92 | 0 |
| 256k | 31.8 | 449 | 584.6 s | 2.17 | 0 |

- Baseline, main 1ea9492a on the same prompts: 57.2 / 55.5 / 54.8 / 57.4 decode at 4k-32k; its 64k and 256k
  cells were contended (18.0 and 22.6). The slope past 32k is the global layers' attention per verify row.
- The stalls came with other agents' builds and fan changes on the box: a quiet box read 0 on every rung, both
  before the pool fix (typical-sampling boots below) and after it (64d9c341: 63.8 / 60.8 / 55.8 / 48.9 / 39.9 at
  4k / 8k / 32k / 64k / 128k, 0 stalls, 07:38).
- Typical 0.2 at T=1.0, top_p 0.95, seed 7, two boots back to back after 3 min idle: `--mtp-greedy-tail` (90f8bf7f)
  vs without (3d11b0f7) decodes +15.1 / +12.6 / +13.2% at 16k / 32k / 64k (tok/step 3.51 / 3.51 / 3.12 vs
  2.78 / 2.72 / 2.69), with GPU clocks within 1-5% and prefill within 1-8%; at 4k / 8k the clocks differed 14-19%.

<a id="mimo-ladder"></a>
### MiMo-V2.6-Flash-Sushi-2.3bpw, 4k-128k context ladder (1a7f92d1)

Pack `MiMo-V2.6-Flash-Sushi-2.3bpw` (MCG K2.25 w12, last layer K4), binary built from `1a7f92d1` with MLX `d73eb752`
(SHA-256 `a4dd6fe7`), `--ctx-size 1048576 --kv-quant 8 --mtp`, llmprobe 0.6.12 `--bench-only --rungs
4k,8k,16k,32k,64k,128k --timeout 3600`, `taskpolicy -a`, lock `bench-mimo23-ladder`, quiet box, fans at max from a
49 °C start, 2026-09-30. Headline cells: decode 70.0 tok/s, prefill 1132 tok/s at 2k, first token 327 ms, MTP 3.69
tokens per step (predictable 80.5, novel 50.1). llmprobe saw a 12.1% sustained-load slide (70.0 -> 61.5) over the 12 min
run. Each rung is one run.

| context | decode tok/s | prefill tok/s | first token | tokens per step |
|---|---|---|---|---|
| 4k | 65.2 | 1091 | 3.8 s | 2.78 |
| 8k | 60.1 | 1134 | 7.3 s | 2.09 |
| 16k | 60.7 | 1100 | 14.8 s | 2.78 |
| 33k | 61.8 | 1025 | 31.9 s | 2.56 |
| 66k | 57.3 | 894 | 73.4 s | 2.46 |
| 131k | 52.0 | 715 | 183.3 s | 3.05 |

At `--ctx-size 1048576` the pack loads (preflight 96.7 of 102.9 GB) but admission then has 12.9 GB left, under the
16.76 GiB bill of a 1M session at kv8. That boot ran at the default GPU limit; at `iogpu.wired_limit_mb=120000` the
admission bill of a full 1M prompt (weights, 512-rung bill, 1 GiB hot cache) is 101.4 GiB and fits, 94.3 GiB at kv4
(probe on `83dc9b6c`, the method of [engine-memory-admission](engine-memory-admission.md)).

<a id="mimo-verify-2p3"></a>
### MiMo Sushi-2.3bpw decode forward: attribution and verify-row levers (2f15cc97)

Forward meter (`SUSHI_DECODE_FWD_UBENCH`), kv8, `taskpolicy -a`, GPU lock, 2026-09-30, box busy with other builds
(absolute numbers read high; compare arms only). Base 2f15cc97: 20.80 ms at 1 row, verify rows 2 / 3 / 4 = 28.54 /
34.85 / 41.47 ms at 1024 keys.

Where a 1-row forward goes (in-process stand-ins: each family replaced by a view of its input, same boot, 1024 keys):
routed experts with their router ~8.3 ms, FP8 QKV ~5.3, affine-8 o_proj ~3.3, lm_head ~1.2-1.5, attention ~1.2, the
rest (norms, layer-0 MLP, embed) ~1.5. The routed experts are ALU-bound (2.7 GB in ~8 ms) and grow ~6.7 ms per extra
verify row; the trunk GEMVs are flat in rows in isolation (FP8 direct GEMV ~128 us per sliding layer at 1-4 rows;
the shipped NR 1 / SGS 2 was the best of NR 1-4 x SGS 1-8) and run at ~470-480 GB/s.

Three row levers, each row still its decode tick's bytes: a sliding layer's verify rows in ONE dispatch
(`mimoSlidingRowsAttn`, a port of MLX's `sdpa_vector` with one threadgroup per head and row), the router's rows in
one f32 gemv (`mtp_qmv.f32GemvRows`, MLX's M=1 gemv per row), and the affine-8 row kernel at two output rows per
simdgroup on the unrolled path. Per-forward alternating A/B in one boot (arms interleaved forward by forward, n=60
per arm, 160 keys = the llmprobe decode cell's context), median ms:

| rows | no lever | all three | sliding rows | router rows | affine-8 two rows |
|---|---|---|---|---|---|
| 2 | 29.29 | 28.47 | -0.38 | -0.11 | -0.45 |
| 3 | 38.86 | 37.35 | -0.72 | -0.23 | -0.61 |
| 4 | 49.17 | 47.09 | -0.91 | -0.06 | -0.50 |

Per-lever columns: paired mean of all three on minus that lever off (negative = the lever saves). Primitives per
4-row forward 5704 -> 4978.

The pair GEMV on a once-prepared input (`pairGemvPrepared`, lane-ordered half4 reads), same meter and settings, arms
prepared / self-preparing: 2 rows 27.72 / 28.14, 3 rows 33.66 / 34.55, 4 rows 40.70 / 41.85 ms (paired -0.43 /
-0.86 / -1.21). One row keeps the fused prepare: there the extra dispatch lost 0.08-0.10 ms. In the chained kernel
microbench (47 layers, E=256, ~19 of 32 slots unique at 4 rows) the pair step went 315 -> 296 us at 4 rows without the
lane order; decode ablations at 1 row: the MCG decode stand-in -12%, no weight loads -16%, constant inputs (no prepare)
-11%, decode and loads both removed -43%; groups of three or four members spill (1.7-2x slower) and unroll_count(2)
on the pair GEMV's k loop is 5-7% slower.
The affine-8 rows kernel at four rows per simdgroup and four simdgroups beat two rows / two simdgroups by 18% on
lm_head in an isolated 12-copy microbench but lost 0.5-1.1 ms per forward; judge it by the meter.
Greedy chat, 4 prompts x 320 tokens, kv8: base, the three levers and the prepared pair byte-identical, serial and MTP,
and MTP == serial on each.

Measured and not taken (same meters): a dependent dispatch costs ~1.7 us in the live graph (`SUSHI_DISPATCH_PROBE`
0 8 8 0: +376 dispatches, +0.6-0.7 ms at 1 and 4 rows; a forward has ~1150), so a one-dispatch fusion is worth
~k x 48 x 1.7 us; MLX command-buffer commits cost nothing (`MLX_MAX_MB_PER_BUFFER=1000000`, ops 400: within boot
noise); a global layer's verify rows in one dispatch below 1024 keys (paired -0.05 / -0.15 / +0.01 ms at 2 / 3 / 4
rows); the FP8 direct GEMV loading two chunks ahead (+0.2 ms at 4 rows); the pair at one output tile per threadgroup
(faster only with heavy row sharing, 1% slower at 24 of 32 unique slots); pair K splits 1 / 4 / 8 (2 is best);
half2 FMA on the decoded weights (lossy; +5% at 1 row, -4% at 4 rows).

Rope + kv8 quantize in one dispatch (`mimoDecodeQkvPrep`, on top of the levers above, same meter, 30 forwards per pass,
`_QKV_PREP_ARMS` off/on/on/off twice in one boot): 1 row 21.23 -> 20.95 and 21.51 -> 21.26 ms (-0.27 ms, -1.3%);
4 rows 39.57 -> 39.82 and 42.21 -> 41.35 ms under a 5 ms upward drift across the passes (-0.3 ms mean, not
resolved). Primitives per forward 3420 -> 3040 at 1 row, 4766 -> 4386 at 4. Two more boots at 16 passes each
(four off/on/on/off sets): 2 rows -0.45 / -0.20 / -0.38 / -0.40 ms per set (-0.36 ms, -1.3%); 4 rows +0.30 /
-0.34 / -0.72 / -0.65 ms, the first set inside a 3 ms warm-up rise (-0.35 ms mean, -0.57 without it).

<a id="mimo-prefill-nax-body"></a>
### MiMo 2.3bpw prefill: where a 2k chunk goes, and the branch-free NAX GEMM body (2f15cc97)

One cold 2025-row chunk (the llmprobe 2k cell minus its cached prefix and final row), in-process prefill meter
(`SUSHI_PREFILL_UBENCH`), `--kv-quant 8 --mtp --ctx-size 1048576`, `taskpolicy -a`, GPU lock, busy box (other
workers building), 2026-09-30. Before this change: 1590-1624 ms (1247-1263 tok/s). A Metal System Trace put the GPU
busy 98.6% of the forward, so host graph build and syncs cost nothing. A per-stage synced pass (+18% sync inflation)
splits it:

| stage | ms per forward | share |
|---|---|---|
| EXL3 GEMMs (gate 409, up 408, down 405) | 1222 | 65% |
| FP8 QKV (dequant + MLX bf16 GEMMs) | 264 | 14% |
| affine-8 o_proj (NAX qmm; the dq+GEMM route starts at 2048 rows) | 159 | 8% |
| router + top-k + argsort, token prepare, SwiGLU mid, sorted finish | 110 | 6% |
| rope, transposes, kv8 write | 46 | 2% |
| attention (sliding band 21, global 20) | 41 | 2% |

Real text routes unevenly: per layer the busiest expert takes 583-1400 of 2025 rows and 9-60 experts none, so a
layer runs ~640 live 32-row windows (25 rows each) and 1136 16-row MMA blocks (12% padding).

The NAX body with clamped x rows, the unswitched k loop and unroll 2 (bytes equal the branch-guarded body's), kernel
ubench `SUSHI_EXL3_GEMM_ARMS` (E256, n36 MCG w12, 16200 slots on five real layers' routing counts, arms interleaved,
median of 12), us per GEMM:

| projection | branch-guarded body | this body |
|---|---|---|
| gate/up 4096->2048 | 7539-7872 | 5566-5850 (x0.73-0.75) |
| down 2048->4096 | 7457-8102 | 5542-5983 (x0.73-0.75) |

In-process meter, arms alternated in one boot (0 = branch-guarded body), ms per 2025-row chunk: 2081 / 1726,
1989 / 1777, 2225 / 2070 (x0.83 / x0.89 / x0.93; the box drifted slower through the run). A 4096-row chunk:
3512 / 3016, 3561 / 3043 (x0.86 / x0.85). A quieter moment read 1292 ms per 2025-row chunk on this body (1567
tok/s, the `SUSHI_PREFILL_UBENCH` meter on 2f15cc97 + this change). On the real pack and real text the two bodies'
final hidden states of a 2025-row chunk are byte-identical (0 of 8,294,400 bf16 differ, four alternated arms).

- On this body a GEMM is MMA-bound (ablations, 5.6 ms: no weight decode 5.04, no x loads 5.39, no MMA 2.70), and 12%
  of its MMA rows are 16-row padding.
- The synced profile overstated the trunk: unsynced microbenches put a sliding layer's FP8 QKV at 4.40 ms (MLX's bf16
  GEMM alone 4.04 ms, 60 TFLOPS) and the affine-8 o_proj at 2.74 ms (MLX `qmm_t_nax`, ~50 TFLOPS).

<a id="mimo-batched-decode"></a>
### MiMo 2.3bpw: concurrent streams, batched plain rows against interleaved MTP (d7a20bf9, the landed change's code)

One boot, `--kv-quant 8 --max-concurrent 4`, greedy, thinking off, 256 tokens per stream, short distinct prompts
(a 300-word story or a Python module with tests), N requests fired together; MTP on (default) against
`enable_mtp:false` per request, which now decodes as rows of one forward (`forwardMimoBatchedDecode`).
`taskpolicy -a`, GPU lock, fans max, die 78 C at start, 2026-10-01. Aggregate tok/s (tokens over wall):

| streams | prose MTP | prose batched | code MTP | code batched |
|---|---|---|---|---|
| 1 | 57.5 | 50.4 | 62.9 | 47.9 |
| 2 | 53.8 | 64.2 | 65.9 | 62.7 |
| 3 | 55.2 | 74.3 | 65.7 | 71.9 |
| 4 | 54.9 | 79.9 | 65.3 | 76.1 |

Interleaved MTP streams share the GPU round by round, so their aggregate stays where one stream is (~55 prose, ~66
code). Batched rows read most of a forward's weights once for the group: past 2 streams on prose and 3 on code
they beat it.

<a id="mimo-crowded-mtp"></a>
**Crowded MTP (7bc8e280, 2026-10-01).** Two consecutive boots of the same ReleaseFast binary, with
`SUSHI_MTP_BATCHED=0` then `1`, `SUSHI_ROUND_COST_PERSIST=0`, `--mtp --no-pld --kv-quant 8 --prefill-chunk 2048
--ctx-size 8192 --prefix-cache-entries 0 --max-concurrent 4 --metrics`. Greedy, thinking off, 256 output tokens,
identical prompts and warmup, `taskpolicy -a`, exclusive GPU lock per arm, fans max. The arm-boundary maximum sensor
reading was 75.6 C. Warm serial rates were 46.8 and 46.7 tok/s. All 14 response pairs were byte-identical; metrics
reported batch width zero with the policy off and widths three/four with it on.

| workload | streams | interleaved MTP | crowded batching |
|---|---|---|---|
| prose | 3 | 48.1 | 65.0 |
| prose | 4 | 48.7 | 70.9 |
| code | 3 | 57.4 | 64.3 |
| code | 4 | 59.0 | 69.3 |

Aggregate output tokens / request-group wall time, tok/s. Separate boots include run variation. The earlier
cache-enabled calibration had a different priming sequence (nine restored tokens versus three), so its byte strings
were not used as this comparison's oracle. The strict live gate also matched solo, crowded and streamed output.

<a id="mimo-mtp-vs-serial"></a>
### MiMo 2.3bpw: MTP against serial per KV bucket (63476cd1; the 256k boot on 0f5e7a95)

`--kv-quant 8`, MTP on (auto depth) against `enable_mtp:false` per request, greedy, thinking off, 256 tokens; the
context is the repo's Zig source in the system message, the task a 300-word story (prose) or a Python LRU cache
with tests (code). Per bucket and kind two pairs, each MTP request the first with its nonce (it restores at the
user-message mark, so its heads see a full window) and its serial twin after it; MTP == serial bytes on all 24
pairs. One boot for 4k-128k and one for 256k, `taskpolicy -a`, GPU lock, fans max, busy box, 2026-10-01. Decode
tok/s, MTP / serial (accepted drafts per round):

| context (tokens) | prose MTP | prose serial | prose ratio | code MTP | code serial | code ratio |
|---|---|---|---|---|---|---|
| 4,540 | 45.7 / 44.6 (0.71) | 43.1 / 42.5 | 1.05 | 64.1 / 66.6 (2.31) | 43.6 / 43.3 | 1.51 |
| 15,874 | 42.4 / 44.3 (0.78) | 41.1 / 40.2 | 1.07 | 59.6 / 58.0 (1.95) | 41.4 / 39.9 | 1.45 |
| 31,742 | 40.8 / 43.9 (0.76) | 38.6 / 39.5 | 1.08 | 55.2 / 54.4 (1.55) | 37.9 / 38.4 | 1.44 |
| 57,586 | 38.9 / 39.2 (0.66) | 37.2 / 36.5 | 1.06 | 50.4 / 50.6 (1.72) | 38.9 / 39.4 | 1.29 |
| 109,468 | 33.0 / 34.6 (0.68) | 35.6 / 36.2 | 0.94 | 48.8 / 46.1 (1.53) | 34.0 / 36.4 | 1.35 |
| 211,087 | 26.7 / 26.8 (0.89) | 32.7 / 32.4 | 0.82 | 39.7 / 37.1 (1.90) | 30.5 / 30.8 | 1.25 |

Prose stops paying past ~100k keys (each verify row reads every global key); code pays at every context. The
adaptive serial switch takes MiMo from 64k ([engine-mtp](engine-mtp.md#adaptive-serial)).

<a id="mimo-adaptive-serial"></a>
**Adaptive serial at 211k keys (0cfb70f2, 2026-10-01).** The frozen 256k-bucket requests above were replayed with
`SUSHI_MTP_ADAPTIVE_SERIAL=1 SUSHI_ROUND_COST_PERSIST=0`, `--mtp --kv-quant 8 --prefill-chunk 4096 --ctx-size 581632
--prefix-cache-entries 32 --prefix-cache-mem 5840MB --max-concurrent 1`; PLD on, greedy, thinking off, 256 output
tokens. Actual admission width was 4096. One boot, `taskpolicy -a`, exclusive GPU lock, fans max. All eight responses
matched the recorded baseline's bytes, prompt lengths, restored-prefix lengths and output-token counts.

| workload | recorded MTP, switch unavailable | adaptive MTP request | serial in this boot | switches |
|---|---|---|---|---|
| prose, 211,087 tokens | 26.7 / 26.8 | 32.1 / 31.4 | 30.9 / 30.8 | one per request |
| code, 211,088 tokens | 39.7 / 37.1 | 45.7 / 41.1 | 29.5 / 30.4 | none |

Decode tok/s, two requests per cell. Prose reached serial speed; code retained speculation. The historical arm is
the recorded 0f5e7a95 boot above, so its speed differences include intervening engine changes and run variation.

<a id="mimo-mtp-round"></a>
### MiMo-V2.6-Flash-Sushi-2.3bpw: where an MTP round goes (2f15cc97)

Setup: `--kv-quant 8 --mtp --ctx-size 1048576`, llmprobe's decode-cell prompt, greedy, `SUSHI_MTP_TRACE=1`, the
forward meter at 4k keys and a Metal System Trace, `taskpolicy -a`, lock held per boot, contended box, 2026-09-30.
- The round is GPU work. The GPU is busy 96.3% of a 5 s decode window. Rounds run 43-46 ms at ~3.0 tokens (1.9-2.0
  accepts of 2.5 drafts). The headline's 3.69 tok/step is llmprobe's predictable cell, not the decode cell.
- The verify trunk takes 29.2 / 36.6 / 44.1 ms per forward at 2 / 3 / 4 rows, lm_head included (1.0 ms). Its graph
  builds in 1.3-1.6 ms of CPU, hidden behind the draft chain.
- The draft chain takes ~1.45 ms of GPU per step: a head forward of 0.94-0.99 ms at 1-4 catch-up rows, plus the
  coarse readout. The head's GEMVs sum to ~0.61 ms at 550-700 GB/s; the rest is ~35 small dependent dispatches.
  Fusing two add+norms, the SwiGLU and the value scale saved nothing measurable.
- Host gaps per single-chunk round were ~1 ms: 0.27 + 0.25 ms around the verify's capture sync and ~0.4 ms after the
  argmax readback. Keeping the verify captures lazy removes the first two: the GPU goes from 96.3% to 98.3% busy, with
  one ~0.45 ms gap per round (the lazy capture and the 2-bit readout on 2f15cc97, same flags).
- Forced depth 3 vs the auto planner on the decode cell: 3.25 vs 2.74-3.10 tokens per round, within ~0-4% on tok/s.
  Forced 3 loses 11% on prose. The planner is not a lever here.
- Ruled out: dropping routed experts under 2% of a row's weight removes 2.8% of slots for +0.0029 KLD (16x512 to EOS,
  kv8); 4% removes 7.4% for about +0.007, over the gate.
- First token on llmprobe's 2k prefill cell, outside the 2025-row chunk: the final 1-token forward 24-25 ms, the ring
  checkpoint's copies 2.8 ms of host encode (prefix cache on), request plumbing ~3 ms, and round 1 (30-45 ms), which
  produces the first visible token when thinking is on (t1 is `<think>`). The final forward stays separate: it keeps
  a cold request's t1 decode-shaped, as its warm full-prefix replay's is.
- Not taken: a serial first step instead of round 1 when t1 is invisible. Round 1 runs at depth 1 there, a 2-row
  verify of ~29 ms against ~21-22 ms serial, so it saves ~7-8 ms of first token (~0.45%) and costs llmprobe's decode
  window ~0.3-0.5% for the token round 1 no longer commits. The ~0.45 ms gap after each round's argmax is readback
  wake-up, the commit, a 0.08 ms chain build and the first command buffer's encode; only a pre-dispatched next chain
  would remove it, at ~4 ms of wasted GPU per partial accept.

<a id="mimo-2p3-quiet-ab"></a>
### MiMo-V2.6-Flash-Sushi-2.3bpw: the verify-row, prefill-GEMM and round changes together (2f15cc97 base)

llmprobe 0.6.12 `--bench-only --rungs 4k`, `--kv-quant 8 --mtp --ctx-size 1048576`, `taskpolicy -a`, GPU lock per boot,
quiet box (no other job), fans at max from 44 °C, 2026-10-01, one boot per arm in the order A B C C B A. A = 2f15cc97;
B = the three sections above plus the one-dispatch rope + kv8 quantize (this change); C = B without the lazy verify
capture.

| arm | decode tok/s | prefill tok/s (2037-2038 tokens) | tokens per step |
|---|---|---|---|
| A | 64.0 / 64.2 | 1136.7 / 1119.2 | 2.82 / 3.69 |
| B | 72.5 / 71.1 (+12.0%) | 1308.3 / 1290.1 (+15.2%) | 3.62 / 3.10 |
| C | 71.1 / 68.6 | 1298.3 / 1297.3 | 3.15 / 3.62 |

- The 192-token decode requests in the server logs read 62-73 tok/s on A and 67-77 on B across both boots, so the
  gain is round time, not acceptance. B over C (+2.8%) is inside boot noise; the lazy capture's effect is the removed
  idle gaps a Metal trace shows ([#mimo-mtp-round](#mimo-mtp-round)).
- 16x512 KLD to first EOS on B, kv8: 0.086034761, the v1.1.0 value to the digit. B passes `test_mtp_equivalence.sh`
  on this pack (19/19) and the full suite.
- The decode cell's thinking is on, so the prefill handover does not move either cell here.

<a id="mimo-lmhead-shortlist"></a>
### MiMo 2.3bpw: greedy lm_head through the coarse top-32 (on d1408a57, 2026-10-01)

Decode meter (`SUSHI_DECODE_FWD_UBENCH=40`, 4096 keys, `_LMHEAD_ARMS` alternating the full head and the shortlist in
one boot, `--kv-quant 8 --mtp`, `taskpolicy -a`, GPU lock), ms per forward: 1 row 20.89 / 20.10 / 20.07 / 21.27
(full / shortlist / shortlist / full, about -1.0 ms); 4 verify rows 43.65 / 42.83 / 42.97 / 48.28 (at least -0.8 ms).
The full affine-8 head reads 0.64 GB; the 2-bit copy ~0.19 GB plus the 32-row re-score.

Greedy identity, 9 prompts x up to 512 tokens (story, code, prose, JSON, math, a tool call, code and prose with
thinking off, an explanation with thinking on), MTP and serial each, full-head boot vs shortlist boot: every answer
byte-identical (tool-call ids carry a timestamp), and MTP == serial in both boots. A `--no-mtp` boot on the trunk's
own copy returned the same bytes. The audit (`SUSHI_LMHEAD_SHORTLIST_AUDIT=1`) also covered the 16 KLD wikitext
prompts, raw, at 512 tokens. Over ~32,500 audited rows (serial ticks and verify rows) the full argmax was never
outside the coarse top-32, and the served argmax never differed. In the final boot (17,664 rows), every shortlist logit
was also bit-equal to the full head's. `tests/test_mtp_equivalence.sh` on this pack: 19 passed, 0 failed, with the
`--no-mtp` base on the shortlist.

llmprobe 0.6.12 `--bench-only --rungs 4k`, `--kv-quant 8 --mtp --ctx-size 1048576`, quiet box, fans at max, lock per
boot, A B B A (A = d1408a57, B = this change): decode 69.8 / 70.0 -> 73.8 / 70.5 tok/s (per-request medians of the
192-token decodes 69.9 -> 72.6); prefill 1307 / 1287 -> 1289 / 1305 tok/s (unchanged). 16x512 KLD to first EOS on B:
0.086034761, unchanged (the KLD tool reads the full head).

<a id="mimo-verify-8"></a>
### MiMo-V2.6-Flash-Sushi-2.3bpw: verify to 8 rows and prompt lookup in MTP rounds (this change vs main 27e81cfc)

FP8 trunk GEMV, `SUSHI_FP8_UBENCH=1` (`_SWEEP=direct` for the geometry), bf16 x, six weight copies, median of 30-60
laps, `taskpolicy -a`, GPU lock, busy box, 2026-10-01; us per call at 8 rows (5-7 rows rank the same):

| shape | staged (NR 4, SGS 8) | direct, one-row geometry (NR 1, SGS 2) | direct, NR 2, SGS 8 (shipped past 4 rows) |
|---|---|---|---|
| qkv global | 208 | 331 | 155 |
| qkv sliding | 220 | 368 | 169 |
| L0 gate/up | 256 | 439 | 163 |
| L0 down | 288 | 465 | 201 |

Decode forward meter on this change (`SUSHI_DECODE_FWD_UBENCH=40`, `_S=1..8`, 1024 keys, kv8, quiet box, lock): 21.6 /
30.1 / 38.0 / 46.4 / 54.6 / 62.1 / 72.0 / 80.0 ms at 1-8 rows, ~8.2 ms per extra row; a fully accepted 8-row round is
10.0 ms per token against 11.6 at 4 rows.

`tests/bench_mtp_lookup.sh`, greedy, thinking off, `--kv-quant 8 --prefix-cache-entries 0`, 2 reps per boot, quiet box
(other workers frozen), fans at max, 3 min idle first, lock per boot, A B B A (A = main 27e81cfc, B = this change), mean
tok/s over the four runs per arm:

| task | main | lookup + 8-row verify | ratio | lookup rounds/drafted/landed (rep0, rep1) |
|---|---|---|---|---|
| copy_verbatim | 85.0 | 104.8 | 1.23 | 66/462/439, 33/231/216 |
| rename | 84.2 | 99.9 | 1.19 | 54/378/361, 35/245/233 |
| prose | 65.5 | 67.7 | 1.03 (no lookup round; noise) | 0/0/0 |

- Greedy bytes identical across all eight runs of each task.
- Each boot's second rep runs fewer lookups (33 vs 66) at +10-18%: the rep0 prose request trains the model's round
  table to narrow MTP widths, and the gate prices the MTP chain at the plan's base width with the request's
  MTP-round acceptance.
- Lookup alone at three drafts (ae92c897, `SUSHI_MTP_LOOKUP=0|1` (now `--no-mtp-lookup`), A B B A, busy box) was neutral: the three heads
  already land ~3.9 tokens per round on a verbatim copy, and a three-draft lookup round (47-51 ms) costs what an MTP
  round does.

<a id="mimo-fp8-tile"></a>
### MiMo-V2.6-Flash-Sushi-2.3bpw: the FP8 trunk audit and the 9-128-row tile (this change, measured at 01319267)

Audit, `SUSHI_FP8_UBENCH=1` at MiMo's trunk shapes, six weight copies, arms interleaved in one process, medians of 5-30
laps, `taskpolicy -a`, GPU lock, fans at max, 180 s idle, 2026-10-04:

- 1-4 rows (direct GEMV): 443-477 GB/s against ~538 delivered, 1.7x MLX's bf16 GEMV and 4-12% ahead of its affine-8
  qmv; the geometry was already swept. 5-8 rows (NR 2, SGS 8): 147-202 us, as recorded at [8 rows](#mimo-verify-8).
- 2048 rows: the dequant pass costs 6-8% over the bare MLX GEMM per call, ~1% of a 2k chunk.
- The weak band was 9-256 rows: the staged arm ran 2-3x slower than MLX's affine-8 kernel at 16 rows, and the dequant
  pass costs a flat 300-400 us per call (182 MB at ~575 GB/s) on top of the GEMM.

The tile against the arm it replaces (staged at 9-16, dequant + MLX GEMM from 17), us per call:

| shape | 9 | 16 | 32 | 64 | 128 | 256 (not served) |
|---|---|---|---|---|---|---|
| qkv global | 146 / 233 | 158 / 494 | 196 / 720 | 329 / 752 | 647 / 824 | 1310 / 966 |
| qkv sliding | 150 / 243 | 177 / 524 | 217 / 754 | 370 / 785 | 798 / 863 | 1533 / 1038 |
| L0 gate/up | 160 / 265 | 189 / 565 | 242 / 800 | 417 / 807 | 887 / 954 | 1789 / 1062 |
| L0 down | 208 / 328 | 196 / 653 | 212 / 804 | 317 / 807 | 904 / 868 | 1309 / 1164 |

In-process prefill meter (`SUSHI_PREFILL_UBENCH=6`, real text, one cold forward, `--kv-quant 8`, arms off / on / on /
off in one boot, `taskpolicy -a`, lock, 2026-10-05), ms per forward:

| rows | tile off | tile on | prefill tok/s |
|---|---|---|---|
| 32 | 155.1 / 155.8 | 120.8 / 120.8 | 206 -> 265 (x1.29) |
| 64 | 185.1 / 195.4 | 156.9 / 165.1 | 337 -> 398 (x1.18) |
| 128 | 226.5 / 232.0 | 208.0 / 208.9 | 559 -> 614 (x1.10) |

Serving, A B B A boots (A = tile off), 17-25-token chat prompts, greedy, thinking off, `--kv-quant 8
--max-concurrent 12`:

- MTP on, 32 prompts, 256 tokens each: 2.414 / 2.406 / 2.461 / 2.480 tokens per round (A 2.447, B 2.434 mean: B
  sits inside A's boot-to-boot spread). Each arm's greedy text repeats across its boots; the arms answer 27 of 32
  prompts differently (prompt-prefill rounding). On the 5 answered identically (277 tokens): 115 / 120 / 118 / 109
  rounds, B more on 3 and equal on 2; a prompt moves up to 4 rounds between boots of one arm.
- 12 concurrent plain requests: 64.0 / 64.3 / 55.6 / 55.6 tok/s aggregate, unchanged. MiMo's batched decode groups
  at most 4 rows (`batchGroupCap`), so the tile never runs in decode; it runs each request's 11-row prefill after a
  prefix hit.

Ruled out:
- The staged arm's geometry at 16 rows (13 geometries): the shipped NR 4 / SGS 8 / S 1 is best or tied on QKV; NR 2
  wins only on the layer-0 MLP (513 vs 557, 543 vs 682 us), and the tile now takes 9-16 rows.
- The tile past 128 rows: issue-bound at ~22 TFLOPS against MLX GEMM's 58 (table above).

<a id="mimo-ttft-idle"></a>
### MiMo-V2.6-Flash-Sushi-2.3bpw: where the time to first token goes, and the GPU wake after idle (e2d5be76 base)

llmprobe's prefill-cell prompt (2038-2041 tokens, streamed, thinking on, `max_tokens` 8), `--kv-quant 8 --mtp
--ctx-size 1048576`, prefix cache on, `taskpolicy -a`, GPU lock per boot, fans at max, 2026-10-01.

- Outside the 2k chunk forward: HTTP, template, tokenize and slot ~3 ms; hot-cache lookup 0.03 ms; the one-row
  final forward 19-25 ms; generator setup and the ring checkpoint 4-5 ms; round 1 28-33 ms (t1 is `<think>`, so the
  first visible token needs it).
- The live chunk equals the in-process meter's in the same GPU state. Meter passes alternated in one boot (ms, median
  of 4): plain 1589, MLX pool cleared before each forward 1572 / 1600, every row's hidden captured 1573 / 1586, both
  1587 / 1590. Neither the pool clear nor the MTP capture costs anything measurable.
- The GPU state moves the chunk by up to 1.8x. Back to back, the first ~4 s run at 1255-1276 ms, then 1460-1537.
  After 20 s idle a forward takes 1976-2285 ms. A one-element op right before it takes 598-999 ms itself, and the
  forward then runs at 1261. A tick every second keeps it at 1264-1268; ticks every 2, 4 or 5 s do not (2200-2270).
- `--gpu-warm-secs` (this change on e2d5be76), one boot per arm, 6 requests each after 10 s idle, TTFT ms: off 2284,
  2305, 1956, 2307, 2344, 1967 (mean 2194); on 1355, 1356, 1355, 1354, 1354, 1354 (mean 1355, -38%).
- llmprobe 0.6.12 `--bench-only --rungs 4k` A B B A against 9297b93b (quiet box, fans max): decode 71.6 / 69.6 -> 69.1
  / 69.1, prefill 1313 / 1306 -> 1299 / 1301 tok/s: unchanged, because llmprobe sends its cells back to back and the
  GPU never idles long enough to pay the wake.

<a id="mimo-attn-kernels"></a>
### MiMo attention kernels (attention only: no expert pack in these timings)

Prefill attention on the matrix units (`sushi_attn_pd_nax`, 2026-09-24, binary b33ec32 built 01:30, taskpolicy -a,
lock attnpd-nax; baselines on 7ed9795).
One global layer, H 64 / Hk 4, qL 2048, kv8, ms: kL 2048 8.04 -> 2.90, 4096 22.1 -> 7.18, 16384 113.0 -> 33.0,
65536 448 -> 159, 262144 2012 -> 601 (34-39 TFLOPS; the SIMD kernel 11-12). 39 sliding layers' band call, per call:
qL 512 1.96 -> 0.78, 2048 1.66 -> 0.95-0.97, 4096 2.98 -> 0.89. The SIMD kernel's K^T staging fix on M5: 2048x16384
106.9 -> 99.4 ms, 2048x65536 448-458 -> 415 ms. Per call on real prefills (4 chunks to 6k keys, carries, kv8 slices,
band + sinks) the two arms' error against an f32 reference agrees to 1e-4 relative RMS.

`sushi_attn_pd_nax` speed work (2026-09-24). Harness: a python replica of the dispatch chain, qL 4096, H 64 / Hk 4,
kv8 slices, arms interleaved in one process, `taskpolicy -a`, lock `lever2-attn`. One global layer, ms (this run read
~30% slower in absolute terms than the same arms an hour earlier, on a contended box; the interleaved ratios hold):

| kL | 08cec69 (before f16 P) | 3ba7272 (f16 P) | f16 P + lockstep causal, clamped loads (250M budget) | this kernel (+ 1e9 NAX budget) |
|---|---|---|---|---|
| 4096 | 11.02 | 10.38 | 8.85 | 7.71 (-30%) |
| 16384 | 74.4 | 72.0 | 61.5 | 59.4 (-20%) |
| 65536 | 457 | 421 | 364 | 357 (-22%) |
| 262144 | 2004 | 1930 | 1658 | 1581 (-21%) |

- At qL 2048 (the same harness), the lockstep kernel with f16 P is -20% to -23% at every kL. The 1e9 budget adds
  nothing there beyond 4k keys.
- f16 P alone buys little: -3% to -8%. The gain comes when the causal simdgroups also walk in lockstep and loads are
  branch-free. Without f16 P, those two changes gave only -2% to -8%.
- Sliding band call, per call, ms (39 layers, window 128, sinks; band keeps per-simdgroup walks): qL 512
  0.316 -> 0.299, 2048 0.576 -> 0.532, 4096 0.947 -> 0.852.
- Ruled out in the same harness:
  - a strict float P (1.4x slower);
  - an int8 correction term (costs what a bf16 one does);
  - 16x32x32 tiles (<= 2%);
  - `max_total_threads_per_threadgroup` (0);
  - fast exp2 (0);
  - 8 simdgroups (slower);
  - a larger budget on the non-lockstep kernel: slower at long kL (1e9: +3% at 64k, +13% at 256k), because K/V fall
    out of cache.

Long-context decode, global-layer attention (2026-09-24, kv8, `taskpolicy -a`, lock `lever3-kv`): `sushi_qkv_mpp` on 4
simdgroups with packed words prefetched in registers (e4dc88e, landed as a338ca2, bit-identical) against the
8-simdgroup kernel of 79a4cb4.

Attention-only µbench (9 dependent layers, us per layer, arms interleaved in one process, two runs): 16k
133-135 -> 136-139, 64k 402 -> 321-323, 256k 1697-1782 -> 1185-1252, 512k 3591-3979 -> 2398-2670. Earlier
same-session runs had main's kernel at 1468-1481 us at 256k (~243 GB/s of a ~540 GB/s read peak).
Split-K on M5 at the same shape: 155 / 528 / 1923 / 3767 us at 16k / 64k / 256k / 512k, so it never beats matmul2d
at 8k keys or more.
Ablations at 256k show where the old kernel's time went:
- dropping both matmuls left the barrier and softmax loop at 545 us with 8 simdgroups, 217 us with 4;
- the rest was the two 16-row matmul2d calls plus loads that the per-page barriers exposed.
What did not help: 64-key pages, separate K/V tiles with two barriers, vector tile stores, transposed QK, V one page
ahead, 256 splits (-3% at 256k, worse at 16k), and one merge kernel (-10 us/layer at 2-4k only, not bit-identical).
From 4k to 16k keys the new kernel costs 3-10 us more per layer, under 0.1 ms per token.
Byte identity, no PLD: greedy serial new == main on 3 prompts (4.9k / 9.8k / 18.6k tokens, 256 generated each).
Forced-depth-3 MTP == serial on the same 3 prompts, on the new kernel rebased onto 36ae6d0.

<a id="mimo-longctx-prefill-attn"></a>
### MiMo long-context prefill: what the global layers' attention costs (kernels of 819b4751)

In-process microbench `SUSHI_ATTN_PD_UBENCH=1` (test filter "MiMo prefill attention per chunk"): one chunk over a kv8
cache built by `KVCache.update`, the served `fusedSdpaPrefillKv` chain, arms interleaved, median of 5. Built on
5834210c, whose attention code is unchanged in 819b4751. `taskpolicy -a`, locks `attn-ub1` / `attn-ub2`, fans at max,
2026-10-01. The table gives ms per global layer per chunk (Hq 64 / Hk 4, qk 192 / v 128, causal); where two runs
differ, both are shown:

| keys | qL 1024 | qL 2048 | qL 4096 |
|---|---|---|---|
| = qL | 0.76 | 1.90 | |
| 16k | 15.7 | 26.8 | |
| 32k | 34.1 | 53.4 | |
| 64k | 69.4 | 108.4-108.7 | 263.4 |
| 128k | 139.8 | 239.6-261.2 | 510.0 |
| 256k | 282.8 | 540.9-554.9 | 1004.6 |

- Throughput counts only the useful causal FLOPs. It is 39-45 TFLOPS at long context, and up to 50 at 32k-64k with
  qL 2048. qL 2048 is the fastest width per row at 128k keys or fewer.
- The dequant and the fp32 carries cost little:
  - The per-dispatch dequant alone takes 0.4 / 1.2 / 2.1 / 3.9-8.2 ms at 16k / 64k / 128k / 256k keys.
  - The same dispatches over pre-dequantized bf16 K/V run 0-10% faster than the served chain.
  - One carry-free dispatch is 6-13% slower than the chained ones from 64k keys up, because K/V fall out of cache.
  - The dispatch budget is not a lever: 5e8, 1e9 and 2e9 land within ±3% of each other.
- One sliding layer's band call, ring dequant included, takes 0.42 / 0.57 / 0.90 ms at qL 1024 / 2048 / 4096. Over 39
  layers that is 16 / 22 / 35 ms per chunk.
- Kernel ablations at 128k keys, qL 2048, on the dense chain (timing-only source edits, `AttnPdNaxAblation`), ms:

  | variant | ms |
  |---|---|
  | served kernel | 240.2 |
  | K/V rows pinned to one block (L1 hits) | 238.1 |
  | K/V fragments built in registers | 200.1 |
  | Q fragments built in registers | 223.2 |
  | no loads at all (55.6 TFLOPS) | 196.3 |
  | no loads and no matmuls | 29.9 |

  So about 69% of the time is matmul issue, about 18% is load instructions (Q is reloaded every key block, and every
  simdgroup loads its own K/V fragments), and about 12% is the softmax, masks and rescale. K/V memory traffic is about
  1%.
- The load diet, measured the same way with lock `attn-fin1`, 9 samples per arm. Build: ba87fd2b's tree before a
  comment-only edit to the header, test binary SHA-256 `f6264dab`. Each fragment row is now one 8-byte
  vector load (`SushiNax::load2`) instead of four element reads. Old kernel vs new, ms per
  global layer: 26.27 -> 25.14 at 16k keys, 108.96 -> 103.40 at 64k, 244.87 -> 226.48 at 128k, 543.56 -> 520.06 at
  256k (-4% to -8%). The output is byte-identical to the old kernel on the kv8 chain at all four lengths and on the
  band + sinks call. Two earlier runs with the loads inlined read -3% to -6%.
- Load-diet attempts that were byte-identical but slower: Q held in registers +39%, which needs the d loop fully
  unrolled, and that unroll alone costs +43%. K/V staged once per threadgroup in threadgroup memory: +107%. Two key
  blocks per Q load: +10%. Unroll 2 or 6 instead of 4: +3-4%. Dropping the mid-PV barrier: +20%. The vector loads
  with unroll 3 read the same as with unroll 4.
- Prefill meter (`SUSHI_PREFILL_UBENCH=6`, rows 2048, real text) on 819b4751, lock `attn-meter2048`: median 1669 ms
  per chunk, minimum 1342. Chunks run back to back slow down ([mimo-ttft-idle](#mimo-ttft-idle)). Attention at
  2048 keys is 39 ms of that time.
- Predicted TTFT, summed over full 2048-row chunks: each chunk costs the rest of the chunk + 9 x global + 39 x band.
  The rest is 1468 ms, fit to the ladder's 16k rung, so that rung matches by construction. The ladder's measured TTFT
  is 12.9 / 28.0 / 67.0 / 172.6 s at 16.3k / 32.7k / 65.6k / 131.0k tokens.

| prompt | rest s | global s | band s | TTFT s | attention share | predicted tok/s | ladder on 819b4751 |
|---|---|---|---|---|---|---|---|
| 16k | 11.7 | 1.0 | 0.18 | 13.0 | 9% | 1265 | 1268 |
| 32k | 23.5 | 4.0 | 0.36 | 27.9 | 16% | 1175 | 1167 |
| 64k | 47.0 | 15.9 | 0.71 | 63.6 | 26% | 1030 | 980 |
| 128k | 94.0 | 69.8 | 1.43 | 165.2 | 43% | 793 | 759 |
| 256k | 187.9 | 306.2 | 2.85 | 497.0 | 62% | 527 | |

- The ladder column ran 4096-row chunks: MiMo picks its width per request, and before 2048 became its default the
  64k-256k bills admitted 4096. The 2048 on the load line was only the load-time fallback.
- 4096 vs 2048 in one boot. Build: a scratch build of 27e81cfc that caps each long request's width in turn. Lock
  `lp-abba1`, 2026-10-01, the same 63,946-token code prompt each time, prefix cache off, thinking off, kv8, MTP on,
  `taskpolicy -a`, fans at max. Prefill: 64.1 s at 4096, 70.4 and 72.6 s at 2048, then 75.5 s at 4096. Each request
  ran slower than the one before (the sustained-load clock drop), and the ABBA means (69.8 vs 71.5 s) are within
  that drift.
  - A per-chunk fit of the live trace puts global attention at 25-27% of the prefill at 2048 and 26-32% at 4096.
    Attention costs ~8% more per row-key at qL 4096, and the rest of the chunk is cheaper per row.
  - The two widths are not byte-identical: the first-token logprob is -1.0685 at 4096 and -1.0807 at 2048. Each
    width repeated its own bytes exactly.
  - 2048 is now the default ([engine-memory-admission](engine-memory-admission.md#context-and-chunk)), so the
    table's 2048 model is the served width.

<a id="mimo-verify-global-rows"></a>
### MiMo verify rows on the global layers: one page walk per row group (A6)

Arms: each global layer's verify rows one `sushi_qkv_mpp` decode dispatch at a time, against `sushi_qkv_mpp_rows`,
which runs each group's rows on one page walk (pairs, a last three in one pass). Both arms are byte-identical to the
decode ticks: unit test at 4k / 64k / 256k keys, 16/1 and 64/4 heads, page and split edges, widths 2-8 and 15, kv8 and kv4.

Attention microbench (`SUSHI_MIMO_ROWS_UBENCH=1`, 9 dependent layers, 64/4 heads, kv8, median of 5, arms interleaved,
`taskpolicy -a`, lock `attn-rows7`, branch perf/mimo-a6-verify-rows at 829187c5), us per layer, per row -> grouped.
829187c5 runs the landed commit's kernel and its groups for 2-4 rows; the landed commit only adds the groups for
wider verifies, the warmup order and test cases. Neither run pinned the fans; the arms are interleaved in one process
or boot.

| keys | 2 rows | 3 rows | 4 rows |
|---|---|---|---|
| 64k | 585 -> 482 (-17.6%) | 854 -> 744 (-12.9%) | 1112 -> 886 (-20.3%) |
| 128k | 1024 -> 857 (-16.3%) | 1519 -> 1304 (-14.1%) | 2021 -> 1683 (-16.7%) |

- All rows in one pass through `sushi_qkv_mpp` at TQ = rows is also byte-identical, but it is 17-31% slower at 3-4
  rows: its matmuls grow to 16 x rows, and each simdgroup's softmax loop walks 4 x rows rows in turn.
- The rows kernel with all four rows in one pass saves only 3-4%: four running outputs cost registers. Two pairs save
  17-20%.

Decode-forward meter (`SUSHI_DECODE_FWD_UBENCH=12`, `_S=2,3,4`, `_KV=131072`, `_GLOBAL_ROWS_ARMS=1` off / on / on /
off), MiMo-V2.6-Flash-Sushi-2.3bpw, `--kv-quant 8 --mtp --ctx-size 1048576`, binary from 829187c5 (SHA-256
`e44e8894`), `taskpolicy -a`, lock `attn-meter128`, 2026-10-01. Results are ms per verify forward at 131k keys:

| rows | off | on | change |
|---|---|---|---|
| 2 | 37.52 / 38.70 | 34.34 / 36.39 | -7.2% |
| 3 | 51.21 / 50.74 | 46.31 / 48.29 | -7.2% |
| 4 | 63.65 / 63.25 | 60.57 / 59.89 | -5.1% |

<a id="mimo-stream-pick"></a>
## Streamed MiMo: the sigmoid-probability lossy pick

MiMo-V2.6-Flash-MOPD (MXFP4 experts) streamed, `--ssd-budget-gb 60 --no-mtp --kv-quant 8 --ctx-size 65536`, M5 Max,
llmprobe 0.6.12 `--bench-only --rungs 4k`, one boot per arm on the same binary (built at cf23043d, the landed change's
pick code), `taskpolicy -a`, GPU lock per boot, fans max + 10 s, 2026-10-01:

| `--expert-pick-tolerance` | decode tok/s (min-max) | prefill tok/s @2k | ids swapped | speculated layers kept | mean fill / wall per forward |
|---|---|---|---|---|---|
| 0 (exact) | 5.8 (5.5-5.8) | 228.6 | - | 37% | 1.10 GB / 188 ms |
| 0.2 | 10.5 (10.3-11.2) | 228.3 | ~9% | 56% | 0.67 GB / 114 ms |

The exact arm matches the recorded 819b4751 streamed cell (5.5, 5.3-5.9). Per token the exact arm spends ~85 ms waiting
on the router ids and ~100 ms filling misses at ~11 GB/s; the pick turns ~9% of routed ids into cached substitutes and
cuts the fill by 40% (means over the logged decode forwards). KLD: [quality-kld](quality-kld.md#lossy-expert-pick-mimo).
