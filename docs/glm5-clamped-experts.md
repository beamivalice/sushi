# GLM-5.3: clamped EXL3 experts

GLM-5.3 clamped experts, split out of [engine-exl3-experts](engine-exl3-experts.md); everything shared by the served architectures stays there.


<a id="glm"></a>
## GLM clamped experts

GLM-5.3 routes 288 experts top-8 (hidden 4096, expert width 2048; the shipped Sushi-2.4bpw is MCG W14, K2.25 (n36) in most layers and K2.5 (n40) in
layers 37–44; every path serves any window) through
`moeClamped`: the gate upper clamp and symmetric up clamp (limit 10) apply in FP32 before SwiGLU, at every packed rate
the format admits (n16 to n128 in eighth-bit steps). Bank geometry (H128 alignment, matching gate/up/down shapes and expert counts, U16
trellises, F16 scale grids, routed input/score shapes) is checked before dispatch; router IDs inside the expert range
are the router's precondition, never synced to the CPU. Every path below is bit-identical to the staged chain at every
admitted rate and is always on for eligible shapes ([glm5-arch](glm5-arch.md)).

- **GLM's cooperative GEMV is not MiMo's.** `INDEXED_COOP_SOURCE` writes all lane partials and adds them r=0..15
  outside simdgroup g=0..3 before the F16 store; MiMo's grouped epilogue (XOR shuffles, K-split planes) rounds
  differently, so a MiMo kernel is never a GLM oracle. Gate and up share one dispatch (grid Z picks the projection).
- **Decode** keeps slots in top-k order and prepares both gate/up input planes from token rows in one kernel, stored
  in GEMV lane order (tile rows 2q, 2q+1, 2q+8, 2q+9) so each lane loads one `half4` (rows 1–16, equal-shaped MCG
  banks at any window; component −17% at 1 row, −29% at 16). The middle is prepared separately in lane order and the down reads it
  by `half4` (`downLanePrepare` + `downLaneCoop`: every admitted n, MCG at any window, BF16 out; −10–15% against the fused
  middle/down, which now serves only what the lane path declines).
- **Verification rows share weight reads** (`src/exl3/glm_group2.zig`, 3–4 BF16 rows, 4096/2048, top-8, clamp 10,
  MCG at any window, every admitted n, gate/up equal and down free): a ballot pairs equal-expert slots in original slot order, the leader decodes each weight once and
  feeds two independent FP32 accumulator sets, and a serial 4 KiB member reduction keeps the r-then-simdgroup order.
  Singleton leaders run the unchanged body. Routed-chain replay −20% on layers with expert overlap; DFlash2 N2 512/64
  decode 42.43 → 45.45 tok/s at n36 (`ba106e5e`); at n40 (Sushi-2.5bpw, kv8, A4 DFlash2, ABBA in one boot, AC power, `taskpolicy -a`, lock `glm-n40`) +3.2% at 512/64 (4/4 pairs) and +5.0% at 8K/128, same bytes. Real 8K verify rounds are singleton-heavy (70% of assignments). The gate is `glm_group2.servesRate`; a guard test enumerates every admitted n.
- **Prefill** prepares gate/up straight from token rows, shares one window table across the three projections,
  builds the inverse routing on the GPU (at most 512 experts) and finishes from the sorted down plane; the stride
  fallback scatters. WIN32 already skips its second 16-row MMA for runs of at most 16 rows (512-token prompts touch a
  median 230 of 288 experts).
- **Full T2048 chunks transpose the grid** (`src/exl3/glm_prefill_grid.zig`, B1, H4096/I2048, E288, top-8, every admitted
  n including mixed per-projection rates, MCG at every window (one NAX kernel per window, like the GEMM), clamp 10): physical X walks routing windows and Y the 128-column output stripes; logical IDs, dot body and
  stores are unchanged. Actual L20 chain 19.64 → 17.96 ms at n36 (−8.6%, 11/11); at n40 (Sushi-2.5bpw, kv8, ABBA in one boot, AC power, `taskpolicy -a`, lock `glm-n40`) prefill +4.5% at 8K and +2.7% at 32K, same bytes. A test enumerates every admitted n against the sorted chain. The T2048 routed chain is GEMM-bound
  (gate/up ≈60%, down ≈30%; sort/prepare/middle/finish ≈1.75 of 17.35 ms).
- **MCG/W12 decode is pure ALU** (mask, multiply/mask/xor, half adds): there is no codebook table or expanded weight
  plane to cache, and cross-round expert reuse at 8K is only ~40%.

Ruled out for GLM experts (each exact unless noted; "component" = an isolated replay on real banks):

- Window height 16 or one 16-row accumulator: 7–19% slower per GEMM; WIN32 keeps decode reuse and two-tile scheduling.
- 64/32-thread output groups: group 64 only 0.7–1.8% faster, group 32 flat or slower; 128 kept.
- n36 SIMD word sharing (load 36 words once, shuffle): 83–86% slower; fewer source reads are not fewer transactions.
- Three verify rows through the sorted prefill NAX body: 11–32% slower (7–19% with 25-window capacity), not exact.
- Grid transpose at T1536–2047: −9.6% on a sliced component, +1.24% on the 2037-token model gate.
- Three-member read sharing: −9% on one high-overlap case, +2–8% elsewhere; parallel reduction lost every case.
- Group2 partner prepass (one dispatch emitting partner slots): 0.16% median, 5/11 wins.
- Singleton-only/paired-only gate/up kernels: +1.61%, 0/11 on the 42-layer T3 replay.
- Grouped middle/down fusion for verify rows (12 KiB staged members): +11.52%, 0/11.
- One-row fused middle/down: model decode 26.18 vs 26.54 tok/s; superseded by the lane down.
- Route6 (keep the 6 heaviest of 8 routes at T2048): −20% on L20 but code mean KL 0.334 (bound 0.01) on the 16K screen.
- Steady 4096-row prefill chunks (grid −8.4% at 4096): moved IndexPool NAX engagement and failed the 16K screen
  (code mean KL 0.034, prose max 0.18).
- Expert-pair prefill (join two routed layers' gate/up): exact L20 −2.47% component, model/assistant gate never
  closed; code deleted.
