# mlx-serve integration

mlx-serve serves Sushi packs' EXL3 routed experts through a vendored copy of Sushi's `sushi_exl3` module
(`src/exl3`). This file is the contract for a re-vendor: the names mlx-serve calls and what the host owes the
module. The module itself is described in [engine-exl3-experts](engine-exl3-experts.md).

## How mlx-serve consumes the module

mlx-serve vendors Sushi 1.3.0's `src/exl3` as `src/sdk/quant/exl3/sushi/` (ddalcu/mlx-serve#978). It has no `lib/sushi`
submodule and pins no Sushi commit, so a Sushi change reaches it only by a re-vendor and cannot break its build. The
copy builds as module `sushi_exl3` with one import, `mlx_host`, which re-exports `mlx`, `log` and `io_util` from
mlx-serve's own host root. It has diverged from `src/exl3` (Metal sources in `.metal` files, its own host env), and
Sushi's later `kernel_source.zig`, `project_rows.zig` and `linear.zig` are not in it. A re-vendor takes
the whole `src/exl3` and keeps the names below at the same paths.

## The contract

The module reaches `mlx`, `log` and `io_util` only through `mlx_host`, whose root must expose them
as `pub const`. It never imports a Sushi file by path, so it builds inside any host that provides
those three.

mlx-serve calls exactly these names. Renaming one, or changing its signature or meaning, breaks
mlx-serve's next re-vendor; add new names instead, and say so in the CHANGELOG.

| name | used for |
|---|---|
| `Spec`, `parseExpertQuant` | reading `expert_quant` from `config.json` |
| `admitTopK`, `trellisAdmitted` | refusing a pack the kernels cannot serve, by name, at load |
| `Bank` (`Proj`) | one layer's `trellis`/`suh`/`svh` banks |
| `moe` | the routed SwiGLU dispatch (decode chain or prefill GEMM) |
| `format.Decode`, `format.Codebook.mcg` | the decode spec passed to `moe` |
| `kernels.DECODE_ROWS_MAX`, `kernels.usesPrefillArm`, `kernels.rowsOfShape` | the host's own row planning around `moe` |

Added since mlx-serve's old pin (44315136) and not called by it yet; each is additive, so the names above keep their
meaning:

| name | what it does | since |
|---|---|---|
| `GroupLayout` (`group_layout.Layout`) | validates a layer's rate-group (`<proj>.gN.*`) names, shapes and dtypes, bills their stored bytes, maps a router index to (group, local expert) | 4ca5ece4 |
| `moeGroups` | the routed SwiGLU over a layer stored as rate groups: one `moe` pass per group, summed in f32 | 4ca5ece4 |
| `moeWithShared` | `moe` plus an already gated shared expert; at one row the add folds into the decode reduction, rounded as the separate add | 698c4ae8 |

STQ support is additive too: `Spec` gains `rates` and `table`, `format.Decode` gains `table` (null for MCG and MUL1, so
existing struct literals keep working), `format.Codebook` gains `std1`, `spec1` and `pair1`, and these names are new:

| name | what it does |
|---|---|
| `codebook_tables` | the named STQ tables, their digests and rate ranges |
| `Ragged`, `moeRagged` | a rate-grouped layer as one ragged bank and its routed SwiGLU dispatch, one decode dispatch per stage |

`parseExpertQuant` now admits `format: stq`, so a host must either refuse an STQ `Spec` or pass `Spec.table` in
`Decode.table` to every `moe` call and check each bank with `trellisAdmitted` against `Spec.rates` as well (`Rate.in`).
A `Decode` that names std1 or spec1 without its table is refused by name (`Exl3CodebookTableUnknown`) at the dispatch, never
decoded. A host that only ever sees MCG and MUL1 packs needs no change.

The module's tests run as their own artifacts (`exl3-test-N`) on `zig build test`; a change that
passes them and keeps the names above is safe to re-vendor.

## What the host owes the module

- **`DECODE_ROWS_MAX` rows is the host's eager boundary.** A forward of more than `kernels.DECODE_ROWS_MAX` rows that
  is not a verify forward takes the prefill GEMM (`kernels.usesPrefillArm`), which may read the routing back on the
  host to build its window table (MiMo and Flash-Next geometries build it on the GPU; every other geometry on the
  host). That read evaluates the stream, so an input the host fills after building the graph (a deferred leaf, such
  as Flash-Next's n-gram embedding) is read empty: gather such inputs eagerly at those widths. Up to
  `DECODE_ROWS_MAX` rows, and any `verify_rows` call, never read back.
- **qwen4_exp packs are uniform-rate only on mlx-serve.** Its loader binds one `Bank` per layer
  (`switch_mlp.{gate,up,down}_proj.{trellis,suh,svh}`) and checks each projection with `trellisAdmitted` against the
  config's `num_experts`: every expert of a projection sits in one bank at one rate (gate, up and down may differ
  from each other from 4ca5ece4 on). A rate-group or pruned pack ([pack-format](pack-format.md#rate-groups-per-expert-rates-uneven-expert-counts))
  does not load there; serving one takes `GroupLayout` at load and `moeGroups` at dispatch, as this repo does.
- **MiMo and GLM packs serve through the same names.** MiMo binds its banks to `moe`; GLM to `moeClamped` (limit =
  `swiglu_limit`, 10) with the host adding its own ungated shared expert, and `trellisAdmitted` bills the config's k, so
  GLM's K2.25 layers sit under k2.5. The packs' source-layout trunk (block-FP8 QKV, `kv_b_proj`, hyper-connection names)
  and the shards the index owns for a duplicated tensor are the host's to read; none of it is module API.

## MLX pins

Each repo builds its own MLX and mlx-c; the module adds no pin of its own. Both repos pin the
same commits: MLX 64ea011 (v0.32.3) and mlx-c 56b2d39. Move both pins together.
