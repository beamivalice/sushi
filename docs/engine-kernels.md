# Engine: custom Metal kernels

The rules every custom kernel in this engine follows: which decode, MoE, prefill and verify kernels exist and when
each engages, how a kernel is proven correct, and the Metal/NAX pitfalls that cost days. Read this before writing or
changing any `mlx_fast_metal_kernel` source or an eligibility predicate.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-exl3-experts](engine-exl3-experts.md),
[engine-mlx-gotchas](engine-mlx-gotchas.md), [qwen4-qsa-long-context](qwen4-qsa-long-context.md),
[engine-kv-cache](engine-kv-cache.md), [perf-baselines](perf-baselines.md).

## Decode kernels

- A custom kernel can be LATENCY-bound rather than op-bound (qwen4 fused hc read, `SUSHI_HC_FUSED=0`; `hcWrite`
  DEFERS into the next read).
- A kernel keyed on `batch*seq == 1` declines every verify row AND batched slot, so the grid carries the rows
  (`HC_FUSED_MAX_ROWS`/`GDN_FUSED_MAX_ROWS` 16).
- GDN decode = three fused dispatches (`SUSHI_GDN_DECODE_FUSED=0`; S 1..9 bit-identity is SAMPLING). At B=1,
  decode (S=1) and capturing verify (S 2..8) run two: `gdn_decode.step` (prework + recurrence, mlx-serve #517)
  then the norm-gate, bit-identical to the chain, ~0.4 ms per forward
  ([perf-baselines](qwen4-perf.md#gdn-decode-recur)); either chain switch off keeps the chain. A/B seam:
  `SUSHI_DECODE_FWD_UBENCH_GDN_ARMS=1`.
- Capturing GDN verify (B=1, S=2, separate projections) folds the norm-gate and rollback convolution history
  into the recurrence. A per-width pipeline probe uses independent inputs so a deferred PLE leaf stays lazy;
  an unsupported threadgroup limit falls back to the existing recurrence and epilogue (an M1 declines S 3..8 and
  folds S=2), so the parity test skips a declined width above two. Parity includes every
  captured state with bf16 carry rounding. Wider captures keep the existing path after the S=5 regression
  in [the width sweep](qwen4-perf.md#gdn-verify-fold). Same-process A/B: `SUSHI_DECODE_FWD_UBENCH_GDN_FOLD_ARMS=1`.
- A fused kernel that replaces a capture chain carries the chain's per-step STORE rounding:
  `gated_delta_step_seq` carries the stored bf16 state to the next token, so serial decode and rollback agree; a
  kernel that carries f32 (upstream's verbatim verify kernel) differs from verify row 1 on.
- The GDN prework and norm-gate serve bf16 activations only; the chain's recurrence outputs take the stored state's
  width, so an f32 state (`--qwen-gdn fp32`) runs the chain while `gdn_decode` declines it.
- The qwen4 fused HC read groups verify rows (`HC_ROW_GROUP` 8 per D/U dispatch group), so each weight word is read
  once per group. Configs are cached per (row count, inject, pending write): MTP alternates widths every round, and
  the first read, reads after a flush and the mixer differ in the other two. The row count stays a template argument:
  as a scalar input it cost 1-4% per verify forward ([perf-baselines](qwen4-perf.md#hc-row-group)).
- A GEMV that beats MLX's qmv in a chained in-graph ubench can still lose inside the forward: a vectorized affine-8
  reader 10-57% faster in-graph was 2-4% slower per decode forward on an M2 Max
  ([perf-baselines](qwen4-perf.md#m2max-decode)). Judge a decode kernel by the decode meter. The affine-8 verify
  rows kernel takes two output rows per simdgroup: four rows and four simdgroups won an isolated microbench and lost
  in the forward ([perf-baselines](mimo2-perf.md#mimo-verify-2p3)).
- A dependent-kernel cut that REDISTRIBUTES a reduction into every threadgroup loses; a routing-independent chain
  the GPU already OVERLAPS is not a dispatch to fuse. Meter: `SUSHI_DECODE_FWD_UBENCH`.
- A matmul2d decode tile of 16 query rows is latency-bound: its barriers and small matmuls cost more than its
  reads. `sushi_qkv_mpp` runs 4 simdgroups, not 8, and holds packed words in registers one phase ahead.
  Tried with no gain: more splits, 64-key pages, split K/V tiles, vector tile stores, transposed QK, a fused merge.
  Numbers: [perf-baselines](mimo2-perf.md#mimo-attn-kernels).
- Decode on this box is dispatch-gap bound: ~860 kernels per Flash-Next token, kernel time ~9.8 of ~18 ms, ~7 us
  per boundary. `MLX_MAX_OPS_PER_BUFFER` and `MLX_METAL_FAST_SYNCH` gave nothing; decode wins come from fewer,
  denser kernels ([perf-baselines](qwen4-perf.md#exl3)).
- At one decode row the shared-expert gate's dot runs on the router kernel's idle simdgroups in MLX's own
  `dot_product` order (bit-equal to the matmul; declines at 32 or fewer experts, K % 8 != 0 or K > 16384).

## Batched Qwen4 decode overlap

- Qwen4 batched S=1 forwards submit the live residual and pending MLP output every fourth layer. An explicit
  `SUSHI_DECODE_ASYNC_LADDER` setting overrides the stride; serial decode remains off when it is unset.
- A deferred host PLE leaf is filled before early evaluation only when its token IDs are already available;
  otherwise that forward skips the ladder and retains its normal terminal evaluation.
- The `[qwen4] batched decode ladder engaged` line distinguishes the batched path from a serial opt-in ladder.
  Model-free parity tests cover eager/lazy IDs, N=2 logits, per-slot PLE history and an unfilled-leaf negative control.

## MoE kernels (affine)

- Fused gate+up made `gatherQmv` the decode default (eligibility = the kernel's OWN conditions,
  `useGatherQmvDecode`, never a model_type list).
- Down+reduce is ONE dispatch splitting each row over 8 lanes with packs hoisted (bar = fp32-truth RMS no worse than
  the composed chain). A 16-lane variant with f32 scores passes the bar but is slower at every width on MiMo; parked.
- 3-bit is a BYTE TRIPLE (`sushi_qpack`). MoE PREFILL uses `_gather_sort`.
- The grouped-expert NAX tile at verify widths is a measured LOSS (parked upstream).
- EXL3 expert kernels: [engine-exl3-experts](engine-exl3-experts.md).

## Prefill kernels

- `sushi_attn_pd` at (qk,v) 256/256 and 192/128 — the widths MLX's steel kernel lacks (band always fused; q_len < 16
  declined); a width `prefillHeadDimFused` lists owes a dispatch at EVERY prefill site scoring at it.
- On NAX the 192/128 widths (MiMo global + sliding layers) run `sushi_attn_pd_nax` (`src/kernels/attn_pd_nax.metal`):
  16 query rows per simdgroup, 16x32x16 matmul2d, K/V fragments read from device.
  - Causal: the four simdgroups walk the threadgroup's key range in lockstep, with MLX's barriers, so each K/V block
    serves all four while it is in cache.
  - Band: each simdgroup walks its own rows' band, with no barriers.
  - Loads past the last row clamp to it, so the loads carry no branches.
  - Same inputs, fp32 carries and slices as the SIMD kernel. A 4x dispatch budget (`ATTN_PD_NAX_DISPATCH_BUDGET`,
    ~20 ms per dispatch at qL 4096) cuts the carries 4x.
  - A chunked chain is bit-identical to one dispatch on either arm.
  - Gate `attnPdNaxServes`: NAX + macOS 26.3 + a one-tile probe of both instantiations (causal, band + sinks)
    against an f32 reference; a failed probe declines by name. `SUSHI_ATTN_PD_NAX=0` = SIMD.
- At 128k keys its time is ~69% matmul issue, ~18% load instructions (Q reloaded each key block, K/V fragments per
  simdgroup) and ~12% softmax; K/V memory traffic is ~1% (`SUSHI_ATTN_PD_UBENCH_ABL=1`,
  [perf-baselines](mimo2-perf.md#mimo-longctx-prefill-attn)).
- Each fragment row is ONE 8-byte vector load (`SushiNax::load2`); element-wise reads cost ~5% of the kernel
  ([perf-baselines](mimo2-perf.md#mimo-longctx-prefill-attn) has the ruled-out load layouts).
- Contract: every q/k/v row the engine hands it starts 8-byte aligned (views slice only the token axis). A misaligned
  row reads correctly on M5 (unit test), but a misaligned vector load is undefined in MSL.
- Its PV feeds P as ONE f16 term (P is in [0, 1]; f16 keeps 11 bits): 16x512 KLD -0.19%, inside the rounding-flip
  floor ([quality-kld](quality-kld.md#the-standard-reading)). Parity bar: per element vs fp64 no
  worse than the SIMD kernel beyond a store rounding flip plus 2^-11 of max|V|. A float P operand into the relaxed
  matmul is truncated (~1e-3).
- Its ceiling is the matrix units' issue rate for 16x32x16 ops (~55 TFLOPS issued on the M5 Max, the rate MLX's hd-128
  NAX sdpa also reaches), so a P that costs a second PV pass costs ~20% of the kernel. The f16 P pays only
  together with the lockstep walk and the branch-free loads ([perf-baselines](mimo2-perf.md#mimo-attn-kernels)).
- The SIMD kernel stages K^T with consecutive lanes on consecutive KEY rows: lanes spread over head-dim chunks
  stride 8*LDK halves, one bank. Bit-identical output.
- hd 256 stays off the NAX attn_pd arm: the same kernel at 256/256 (O in 128 registers per lane) ran 3.2x slower
  than MLX's stock NAX sdpa (2048x16384: 150.9 vs 45.9 ms), which already serves hd-256 causal prefill.
- On NAX the stock sdpa is the hd-256 kernel (`naxSdpaPreferred`, `SUSHI_NAX_SDPA=0|1`).
- MLX sdpa has a WIDTH WALL at hd 256 (dense causal q 6..9 ride `splitCausalSdpa`); `use_fallback` has NO fused arm
  for an hd-256 ARRAY mask (`splitMaskedSdpa256`).
- MiMo's FP8 trunk at 9-128 rows (`fp8_block` tile, serving forwards only, NAX only): 8 simdgroups split K over
  32 stored rows; each step a simdgroup writes the dequant route's own `bf16(code x block scale)` into a 16x32
  threadgroup tile and runs 16x32x16 matmul2d against up to 8 sixteen-row tiles of x (wider inputs split rows over
  grid z). Same operands as the dequant route, f32 sums in another order: KLD-gated, never byte-identical
  ([quality-kld](mimo2-kld.md#mimo)). Past ~128 rows it is issue-bound (~22 TFLOPS against MLX GEMM's 58) and
  the dequant + MLX GEMM route serves. `kld capture` keeps that route at every width (`fp8_block.reference_route`),
  so teacher fixtures reproduce; GLM's raw-FP8 trunk keeps `fp8_block.linear`.
  [perf-baselines](mimo2-perf.md#mimo-fp8-tile) has the audit.
- Qwen4 HC + GDN prefill fusions take the chunk WIDTH as a scalar INPUT (`SUSHI_HC_PREFILL=0` /
  `SUSHI_GDN_PREFILL_FUSED=0`).
- GDN prefill (S >= 64) takes one of three recurrences (`GdnRoute`): stock, blocked-seq, or oMLX's software-pipelined
  kernel (mlx-serve #641; 8 lanes per value row, 12-token blocks prefetched), the default for qwen4_exp on NAX GPUs.
  `SUSHI_GDN_PIPELINED=0` keeps the blocked kernel, `=1` forces it; not bit-identical (dot order).

## Verify lanes

- `vqmmLaneFor`: split-K M 2–7 / wide tile N≥100K / NAX m16 M 8–16; parity = fp32-dequant per width, never vs
  stock's worst element (`VerifyQmmParity`); a verify lane is never byte-identical to stock. Sub-4-bit weights fall
  outside it (a 4/5/6-bit specialisation).
- `--decode-attn-quant` (default ON, LOSSY) requants dense attention at decode AND verify.

## NAX and Metal pitfalls

- **A cooperative-tensor template arg is `metal::remove_addrspace_t<decltype(t)>`**, never `decltype(t)` (the
  macOS 27 MPP header rejects the `thread` qualifier).
- Metal JIT-compiles at first EVAL, not at apply, so an optional NAX arm is PROBED on a one-tile problem before it
  is trusted (`buildNaxGemmKernel`); a failed probe declines by name and the sorted arm serves.
- Every matmul2d tile in the engine is f16/bf16 (EXL3 expert GEMM, QSA gather, attention); int8 NAX would need int8
  activations (W8A8) and is not planned. An int8 x bf16 op costs the same as bf16 x bf16.
- A cooperative tensor's layout depends on the element types and the precision flag. Read it with
  `get_multidimensional_index`, never assume it. A STRICT (unrelaxed) float operand changes all three operands'
  layouts. It also runs ~1.4x slower than two bf16 ops, because it is emulated.
- **A `simdgroup_matrix` array under a data-dependent loop bound spills**: loop to a compile-time count and predicate
  the stores ([engine-exl3-experts](engine-exl3-experts.md#kernels), 2.6x on the non-NAX GEMM).
- Cooperative-only matmul2d takes M, N, K in {16, 32}, with at least one of them 32. Larger tiles are the 16x16
  fragments concatenated. A K=32 op runs no faster than two K=16 ops.
- **Metal caps each compiled kernel's threads per threadgroup by its register use**; MLX throws at a dispatch above it.
  M3 and later grant every kernel 1024, so an M5 never sees it; M1/M2 grant 1024 up to 52 GPRs, down to 384 at 128.
  A group above 384 threads probes and declines (`sushi_gdn_verify_fold`), or its GPRs come from a `metal-tt` G13 build.
- **A custom kernel whose source fails to compile kills the process** at its first eval (mlx-c `array.cpp:352`, exit
  255). The JIT probe declines only kernels that build. Compile a new source offline first: wrap it in MLX's
  custom-kernel template (inputs, their `_strides`, the attribute arguments, one instantiation per template set) and
  run `xcrun -sdk macosx metal -std=metal4.0 -c -I lib/mlx/include`. That is CPU only and needs no GPU lock.
- **A `uint4` K/V staging load needs a 16-byte aligned BASE, not only aligned strides**: a last-axis slice of a padded
  cache (`[3,259)` of 264) passes the stride gates and reads wrong values. `attn256Base16Aligned` declines it before the
  p256 and QSA gather kernels; a lazy array has no address and passes.
- MSL takes no arrays of cooperative tensors ("cannot declare array of non-constant size type"). Name one per row:
  `sushi_qkv_mpp_rows` expands a macro per row.
- `matmul2d::run` and a cooperative tensor's `store` take lvalue tensors: bind a `slice<...>(...)` to a name before
  passing it.

## Proving a kernel

- GPU parity = no-worse-than fp32 ground truth, never kernel-vs-kernel; a parity loop asserts FINITENESS before it
  diffs; every shape an eligibility predicate adopts gets its own A/B.
- Drive a parity case at the model's REAL activation magnitudes, not synthetic unit-scale inputs, and score it
  against a TRUE f32 oracle (an oracle that mirrors the kernel's own f16 stores cannot see a saturation).
- A `metal_kernel` config is cached by FULL SHAPE (`ShapeKey`); a per-token-varying TEMPLATE value is a fresh JIT
  per value — ramping values ride INPUTS.
- Threadgroup memory is an OCCUPANCY decision (≤ ~10 KiB).
- JIT vs metallib transcendentals disagree (a 16-bit domain is swept ENTIRELY, `swigluSigTable`).
- A custom kernel's signature comes from each input's ACTUAL dtype; <8-element arrays land in `constant`; every new
  kernel ships a one-shot "engaged" log + parity on the LIVE dtype.

## Reproducing MLX

- Reproducing an MLX op means reproducing its REDUCTION TREE and ACCUMULATOR; `mlx_compile` on the same math is NOT
  output-preserving; a weight-layout fusion changes which KERNEL runs; a fusion pays only if it shortens the
  DEPENDENCY CHAIN.
- A lever that pays in another harness may pay for a constraint we don't have — a DEFAULT belongs to the engine that
  MEASURED it.
- MLX's mean over a NON-last axis sums a column in row order only while that axis is <= 8 (`col_reduce_small`,
  threadgroup_y = min(8, R)); longer columns pre-sum at stride 8 or in trees, so a serial mirror declines past 8.
- `mlx_fast_rope` is not the composed rope: it keeps cos/sin in f32 and rounds once; the chain rounds the bf16 table,
  each product and the sum.

## Timing a kernel

- Interleave A/B kernels in ONE process (separate runs drift 15%); one-shot per-kernel ubench timings are not
  evidence (codebook-free kernels swung 20-40% between arms from clock ramp).
- Size every probe and config cache for all the arms of an A/B. A full cache re-probes on every call, which added
  ~170 us per call to the dev arms. An ablation that skips a tile's only reader lets the compiler drop the tile's
  stores and their loads.
- Attribute DECODE by `QWEN4_STANDIN` ablations under the decode meter, never by shader samples: the profiler
  under-samples short kernels and mis-shares the rest ([perf-baselines](qwen4-perf.md#m2max-decode)).
- A Metal System Trace: `xcrun xctrace record --template 'Metal System Trace' --instrument 'Metal GPU Counters'
  --attach <pid>`, then export `metal-shader-profiler-intervals` (the profiler under-samples short kernels).
- Time a prefill chunk with the load-time meter `SUSHI_PREFILL_UBENCH=N` (`_ROWS`, capped at the admitted chunk;
  `_TEXT=<file>` for real routing; `_ARMS=0,1,0,1` alternates the EXL3 NAX reference and served bodies in one boot).
- Time MiMo's prefill attention for one chunk with `SUSHI_ATTN_PD_UBENCH=1` (test filter "MiMo prefill attention per
  chunk"). Knobs: `_QL`, `_KL` (empty skips the main table), `_BUDGETS`, `_LEAN`, `_REPS`; kernel variants on the
  dense chain at `_ABL_KL` keys: `_ABL=1` (ablations) and `_ALT=<a.metal>,...` (replacement sources, byte-checked
  against the served kernel).
- Every timing run takes the GPU lock and restores QoS ([CLAUDE.md, Team process](../CLAUDE.md#team-process)).
