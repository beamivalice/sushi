# mlx-serve integration

mlx-serve serves Sushi packs' EXL3 routed experts by importing Sushi's `sushi_exl3` module
(`src/exl3`). This file is the contract between the two repos: what mlx-serve calls, how it pins
Sushi, and how a new engine reaches it. The module itself is described in
[engine-exl3-experts](engine-exl3-experts.md).

## How mlx-serve consumes the module

- **Submodule.** mlx-serve carries Sushi as `lib/sushi` (`.gitmodules`: url
  `https://github.com/ddalcu/sushi`, a fork of this repo, branch `exl3-module`). The committed
  gitlink is the pin: a full commit, nothing floats.
- **Build.** mlx-serve's `build.zig` (`addExl3Module`) builds `lib/sushi/src/exl3/root.zig` as the
  module `sushi_exl3`, with one import, `mlx_host`, pointing at mlx-serve's own host root.
  `-Dsushi-dir=/abs/path` builds against a Sushi checkout instead of the submodule.
- **Use.** `ModelConfig.exl3: ?sushi_exl3.Spec` marks a pack whose routed experts are EXL3
  (`expert_quant` in `config.json`); every other module stays the host's own affine path.

## The contract

The module reaches `mlx`, `log` and `io_util` only through `mlx_host`, whose root must expose them
as `pub const`. It never imports a Sushi file by path, so it builds inside any host that provides
those three.

mlx-serve calls exactly these names. Renaming one, or changing its signature or meaning, breaks
mlx-serve's build on its next bump; add new names instead, and say so in the CHANGELOG.

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

The module's tests run as their own artifact (`exl3-test`) on `zig build test`; a change that
passes them and keeps the names above is safe to hand over.

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
same commits: MLX d73eb752 (v0.32.2 plus the sorted `gather_qmm` NAX 32K-row fix,
ml-explore/mlx#3922) and mlx-c 56b2d39. Move both pins together. MLX v0.32.3 needs a newer
mlx-c: its `gather_qmm` gained a `global_scale` argument that mlx-c 56b2d39 does not pass.

## Recommended pin

sushi v1.2.1 (3f3ff7bc). mlx-serve main pins it since ddalcu/mlx-serve#768, which also serves the GLM and MiMo packs
in-process. Reasons it matters, oldest first:
- Before 4ca5ece4 (in v1.1.1), a pack whose gate and up trellises differ in rate passes `trellisAdmitted` and then
  fails at dispatch with `BadExl3Shape`; from 4ca5ece4 `moe` serves it.
- d1408a57 (after v1.1.1) adds the branch-free NAX prefill GEMM body: each expert GEMM runs in x0.73-0.75 of its time,
  output bytes unchanged ([perf-baselines](mimo2-perf.md#mimo-prefill-nax-body)).

## Handing a new engine to mlx-serve

1. Land the change on Sushi `main` with `zig build test` green, `exl3-test` included.
2. Make the commit reachable from the submodule's url (`ddalcu/sushi`): the fork must fetch it
   from this repo, or the change must go through it.
3. Give mlx-serve the full commit hash. mlx-serve bumps the gitlink
   (`git -C lib/sushi fetch && git -C lib/sushi checkout <sha>`, then commits `lib/sushi`) and runs
   its own tests.

The gitlink is the whole handoff: no release asset or separate manifest is needed.
