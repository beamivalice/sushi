# Hy4-Flash against Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash: serving parity

What a Hy4-Flash (`hy_v4_flash`) load serves of what Qwen and DeepSeek serve, on branch `exp/hy4` (1.8.0, not on main
yet). "Tiny" evidence runs on the beamster fixtures (`test-fixtures/hy4-bf16-tiny`, `hy4-bf16-tiny-stq`,
`hy4-stq-tiny`, `hy4-stq-tiny-mixed`); no real STQ pack exists yet, so nothing below is a live check on one. The
forward and its parts are in [hy4-flash-arch](hy4-flash-arch.md); DeepSeek's own matrix is
[dsv41-parity](dsv41-parity.md).

## Matrix

| Feature | Qwen | DeepSeek | Hy4-Flash | Evidence or gap |
|---|---|---|---|---|
| Runs on Sushi Cooler | yes (MLX opt-out) | yes (MLX opt-out) | Cooler only | no MLX arm (`hy4_contract.mlx_arm = false`); MLX goes at 2.0 |
| BF16 release streamed | yes | yes | yes | 0.37 tok/s decode on the 32 GB M1; fits 32 GB |
| STQ pack (trellis trunk and experts) | yes | yes | loader and kernels done | trunk K4/K6, PAIR1 experts in rate groups at mixed K, K6 embedding gather; a synthetic release-geometry pack serves (decode 4.2 tok/s at `--ssd-budget-gb 18`, M1); the real pack is not encoded |
| Expert streaming from SSD | yes | yes | yes | rate groups stream; no union workspace (Hy4 routes within its cache) |
| KV cache widths | kv8 default, 4/6/8/off | none (own state) | kv8 default, off | 4 and 6 are not offered and a launch flag naming one is refused (`hy4_flags.kv_quant_choices`) |
| Hot (RAM) prefix cache | yes | yes | yes | whole-state restore points; a next turn continues the live state in place |
| SSD prefix tier | yes | yes | yes | points stored as whole records; restored equals cold, byte for byte |
| Restore point behind a reply, stored | decode-end points | yes | no | a next turn reuses the reply only by continuing the live state in RAM; after another conversation or a restart it re-prefills the reply |
| Image input | yes | yes | yes (tiny) | the release's processor (parity to one ulp), the tower by query-row chunks with a billed encode, the splice across prefill chunks, all on the tiny; no live check on a real pack |
| Video / audio | Qwen video | refused | refused | no video processor, as DeepSeek |
| Native speculative decode | MTP | DSpark | not served | DFly2's drafter forward runs on the tiny (1x); its rounds and verify are not wired to the generator |
| Batched decode (several slots) | yes | resident loads | no | one request at a time; a block row is ~2.4e-6 off its single row, so rows are not width-exact |
| Chat template, effort words | yes | yes | yes | renders as transformers' `apply_chat_template` on the reference cases; effort words as the owner ruled |
| Tool calls, forced `tool_choice` | yes | yes | parse | GLM's `arg_key` family under the `:opensource` spelling; forced calls not checked |
| `sushi kld` | yes | yes | yes | capture at f32 products (lossless teacher), compare as served; KLD 0 on readback, kv8 small on the tiny |
| Warm and pipeline manifest | yes | yes | yes | a load writes its manifest; the next load makes its kernels from it |
| Memory bill | yes | yes | yes | forward, state a token and the load billed as held; one open contract cell (below) |
| Engram | n/a | read-ahead on a thread | rows read in the pass | tables are never held; no read-ahead yet |
| Smoke and format matrices | yes | yes | no row | needs a real pack to boot |

## Larger gaps

- **Image encode cost.** An encode at the 1536² cap bills about twice Qwen's (each bf16 projection's four buffers and
  the tanh GELU's seven temporaries live to the end of their block); sharing Qwen's tower core is row 2b.
- **A stored point behind the reply.** DeepSeek keeps one per turn and restores the reply from SSD; Hy4 keeps only
  prefill points.
- **Speculative decode.** DFly2 is the release's own drafter; its rounds are the remaining work (1x, 1z).
- **Decode pool.** After a 4,097-token prefill a decode's in-use peak fits its cap (64.6 of 65.1 MB on the tiny), but
  the pool holds 89.5 MB: freed buffers of sizes a step does not reuse.

## Live checks (a real pack)

None yet: the real STQ pack is not encoded. The synthetic release-geometry pack (random codes) loaded, warmed and
answered on the M1; its text is meaningless by construction.
