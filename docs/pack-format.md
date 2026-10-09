# Pack format — the consumer contract

What this engine reads out of an EXL3 pack. The conversion side lives in the
private converter repo; this document is the interface between them, and a
converter change that changes any line here is a format change.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related:
[engine-exl3-experts](engine-exl3-experts.md), [quality-kld](quality-kld.md),
[qwen4-arch](qwen4-arch.md), [mimo2-arch](mimo2-arch.md).

Readers in this repo: `src/expert_quant.zig` (layout), `src/exl3/root.zig`
(`expert_quant` parse), `src/exl3/expert_exl3.zig` (decode), `src/mimo_source.zig` (`validateShardStamps`),
`src/model.zig` (weight loading).

## Routed-expert tensors

Routed experts are stacked per layer and projection, expert `e` on axis 0, so a
gather kernel indexes it directly. Per MoE layer `L` and projection `P` in
`{gate_proj, up_proj, down_proj}`:

| tensor | dtype | shape |
|---|---|---|
| `<prefix>.layers.{L}.mlp.switch_mlp.{P}.trellis` | `U16` | `[E, in/16, out/16, n]` |
| `<prefix>.layers.{L}.mlp.switch_mlp.{P}.suh` | `F16` | `[E, in]` |
| `<prefix>.layers.{L}.mlp.switch_mlp.{P}.svh` | `F16` | `[E, out]` |

`<prefix>` is the arch's own nesting: `language_model.model` for `qwen4_exp`,
`model` for `mimo_v2`. Every other module keeps the trunk's own names and
layout; only the routed banks are EXL3.

`n` is the packed halfwords per 256-weight tile and it, not an integer K, is
what every reader keys on: weight `t`'s codeword is the 16-bit window ending at
`((t+1)*n)>>4`. Even `n` in `[16, 128]` is admitted (K1 to K8) — 24 = K1.5, 36 = K2.25, 48 = K3,
64 = K4. `k` printed anywhere reads 2.25, never 36. The per-tensor rate is read
from the trellis shape, so a shard at or below the config's rate is over-billed
rather than refused; a wider one refuses.

`in` and `out` are the tile grid times 16, and the Hadamard block is 128
(`H128`), so both must be multiples of it.

### MTP bank

The MTP head's own MoE layer is the same three tensors under the head's prefix
(`language_model.mtp.layers.0.mlp.switch_mlp.{P}.{trellis,suh,svh}`) and is read
by the same decoder. It shares the trunk's geometry, which is why a converter's
resume names its shards `mtp-…`: geometry alone cannot tell the two apart.

Its rows ride the decode chain and refuse wider
(`Exl3MtpRowsExceedDecode`).

### suh / svh, and the g-scale

`suh` scales the input before the Hadamard and `svh` scales the output after it
(`expert_exl3.prepareInput` / `finishOutput`), both through f16 rounding. The
converter's per-expert global codebook scale is **folded into `suh`** — it is
divided out there and never stored as a separate field, so the engine applies
one scale vector per side and nothing else.

### Rate groups (per-expert rates, uneven expert counts)

A layer's routed experts may be split into **rate groups**: contiguous runs of experts, each group an ordinary
stacked bank per projection with its own rate, read from its own trellis shape.
- Names: `<prefix>.<gate|up|down>_proj.gN.<trellis|suh|svh>`, `N` from 0 with no gaps, at most 32 groups;
  `<prefix>` is `language_model.model.layers.L.mlp.switch_mlp` (Qwen), `language_model.mtp.layers.0.mlp.switch_mlp`
  (Qwen MTP) or `model.layers.L.mlp.switch_mlp` (MiMo). A layer without `.gN` names is one group, as before; a layer
  never mixes the two.
- Each group holds all nine tensors: U16 trellis `[E_g, in/16, out/16, n]`, F16 `suh` `[E_g, in]`, F16 `svh`
  `[E_g, out]`. Gate, up and down may each have their own n.
- The layer's expert count is the sum of its group sizes and must equal the router's rows; router rows (and MiMo's
  correction bias) follow the concatenated group order, so router index e is (group, local) by cumulative sizes.
  Pruned experts are simply absent. Top-k must fit the layer.
- No new config field: `expert_quant.k` stays the ceiling every group's rate must fit under.
- Refused by name: `Exl3GroupNameInvalid`, `Exl3GroupMissing`, `Exl3GroupGeometry`, `Exl3GroupDtype`,
  `Exl3MixedGroupLayout`, `Exl3TrellisGeometry` (a rate this build cannot decode), `Exl3RouterWidthMismatch`,
  `Exl3TopKExceedsExperts`, and `Exl3RaggedStreamingUnsupported` (SSD streaming serves uniform banks only).
- Each group is one decode pass per layer (3 kernel calls, 4 when gate and up differ); keep a layer to 1-2 groups.

## `config.json`

```json
"expert_quant": { "format": "exl3", "k": 3, "codebook": "mcg", "window": 15 }
```

- `format` — must be the string `exl3`; anything else is `ExpertLayoutUnsupported`.
- `k` — the rate, a JSON number and possibly fractional. It names the WIDEST
  rate a layer packs and is what the engine bills.
- `codebook` — `mul1` or `mcg`. MCG is the codebook for new packs; MUL1 serves
  turboderp's packs; any other name is `ExpertLayoutUnsupported` (in a shard
  stamp, `Exl3ShardStampMismatch`). The codebook and window follow the MODEL: `exl3.moe` sets them
  (`expert_exl3_kernels.setDecodeParams`) before every dispatch, so packs with
  different codebooks can be resident together, and every weight kernel
  inlines its `exl3_pairh`.
- `window` — the codeword width the search hashed, 8..16. **Absent means 16.**
  The same bitstream decodes to different weights at each width, so a window
  this build cannot decode is `Exl3WindowUnsupported`, never a fallback to 16.

`num_experts_per_tok` must be ≤ 32 (the decode reduce bank,
`Exl3TopKExceedsReduceBank`).

The parser trusts no field: a wrong JSON type or a value outside its field's range is `InvalidConfigField`
(`cfgField`/`cfgInt`/`cfgF32`), a JSON `null` leaves the default (`sliding_window: null` still disables), and a
qwen4_exp geometry the forward divides by or indexes with is `InvalidQwen4Geometry`. The served packs' configs, cut
to what the engine reads, are committed in `src/fixtures/model-configs/` and pinned by a parse test.

### Stored-affine trunk linears (`mimo_v2`)

A MiMo pack may STORE `o_proj` (every layer's `model.layers.{L}.self_attn.o_proj`),
`lm_head` and `model.embed_tokens` packed, each as MLX's affine triple under the
source name:

| tensor | dtype | shape |
|---|---|---|
| `<base>.weight` | `U32` | `[out, in * bits / 32]` |
| `<base>.scales` | `BF16` | `[out, in / group_size]` |
| `<base>.biases` | `BF16` | `[out, in / group_size]` |

The index points all three names at the shard holding them; a source shard that
still carries the bf16 `<base>.weight` is simply not indexed for it. The engine
serves the bytes as stored (o_proj and lm_head through `quantized_matmul`, the
embedding through the quantized row gather) and bills them as stored; there is
no load-time quantization. (bits, group_size) is solved from the shapes like
every affine weight (bits 2, 3, 4, 5, 6, 8; group 32, 64, 128); a triple that is
incomplete, or grids beside a bf16 weight, is `AffineTrunkIncomplete`, shapes
that solve to no admitted width are `MimoTensorShapeMismatch`. Any subset of
the three linears may be stored; the rest stay bf16. The original checkpoint
`kld capture` reads stores all three bf16, so the teacher is unchanged. The
engine quantizes nothing at load: a `config.json` `trunk_quant` block is
`UnsupportedMimoV2Config`.

## The shard stamp

Each written shard carries a safetensors `__metadata__` map — every value a
string, because that is all safetensors stores. The engine reads these keys:

| key | value |
|---|---|
| `format` | `exl3` |
| `k` | the rate as written, e.g. `2.25` or `4` |
| `codebook` | `mul1` \| `mcg` |
| `window` | the codeword width, e.g. `12` |

A converter adds its own provenance keys beside them; the engine reads none of
those, and they are documented in the private converter repo.

A stored-affine trunk shard stamps `format: affine` with its `bits` and
`group_size`, and carries no `k`, `codebook` or `window`, so the load check
below has nothing to compare on it.

**Load (`mimo_source.validateShardStamps`, the MiMo loader only — Flash-Next packs are not stamp-checked today).**
Before any bytes are uploaded, a stamped shard's `codebook`, `window` and `k` are checked against `expert_quant`.
A disagreement is `Exl3ShardStampMismatch` — a named refusal, never garbage
weights. An **unstamped** shard is legacy and is admitted. `k` is compared as
halfwords: at or below the config's rate passes, wider refuses.

## Component packs

The flat component layout keeps embeddings, the output head, two layer-aligned
trunk shards, one routed-expert shard per layer (gate/up/down together), MTP and
vision in separate files. `model.safetensors.index.json` maps the **unchanged**
tensor names onto them; `ngram_table.bin` stays outside safetensors. Legacy
shard names still load. Payload bytes are identical either way — repacking never
decodes or requantizes. See README.md.

## Fixtures

`src/exl3/fixtures/exl3_*_linear.safetensors` are committed and `@embedFile`d by
`src/exl3/expert_exl3.zig` (`fixtures`, which `src/transformer.zig` reads). They are produced by the
private converter; regenerate one only to change the format, and keep the one
searched and decoded at window 12 (`exl3_k2p5_mcg_w12_linear.safetensors`),
which is what certifies a narrowed window against the converter's own decode
rather than against our own masking of a w16 bitstream.

## Quality bar

A pack is judged by KLD against the bf16 teacher (`sushi kld capture` /
`kld compare`, see [quality-kld](quality-kld.md)), never by bytes against an
affine pack: the EXL3 kernel arms round once and are not byte-identical to any
composite.

## Loader rules the engine applies to every pack

- **Expert layout is solved from PACKED shapes** (`expert_quant.zig`): affine
  (bits, group_size) from `w_cols*32 / in_dim`; EXL3 K from the trellis shape;
  `expert_layout` decides `moeExl3` vs the affine kernels and the streaming byte
  plan (`exl3ExpertBytes`). No literal quant width at any
  `mlx_quantized_matmul`/`mlx_dequantize` site — `affineParamsFromGeometry`.
  Affine bits outside {2,3,4,5,6,8} reject at PARSE.
- A trellis whose packed shape does not match the config's rate or expert
  geometry is refused by name at load; a config `k` narrower than a shard would
  under-bill, which on this engine is a Metal OOM rather than an error.
- MiMo EXL3 gate and up trellises in each layer must have equal packed rates (`n`); a mismatch refuses before upload as `Exl3GateUpRateMismatch`; down may use a different admitted rate.
- **Quant modes resolve PER WEIGHT** (`computeQuantParams`; scales dtype decides
  fp8 vs affine; `.biases` mandatory under affine, optional in `loadLinear`;
  `qLinearFwd` passes `mode.cstr()`). A layer-init path that DEMANDS `.scales`
  can't load a DENSE checkpoint (`getLayerScaleOpt`; every dense contracted
  weight owes `maybeTransposeForBf16`).
- **A gather-read table may be quantized only where the READER has a
  quantized-gather path**: LM `embed_tokens` via `gatherQuantizedRows` is a
  SIZE decision (our packs quantize it).
- A component pack's shared files are immutable: a converter replaces a
  hard-linked file, never modifies it in place.
- **A tensor two shards carry loads from the shard the index names** (`ShardOwners`): the MiMo 2.3bpw pack's
  source shard still holds bf16 `o_proj`, `embed_tokens` and `lm_head` beside the affine ones, and the generic loader
  kept whichever file the directory listed last and leaked the other's name. An owner shard that is not on disk
  claims nothing (a partly stale index); without an owner the later file stands and the earlier is released. The
  served MiMo path reads through `mimo_source`, which refuses a duplicate.
