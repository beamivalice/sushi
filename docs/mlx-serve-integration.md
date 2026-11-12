# mlx-serve integration: the `sushi_exl3` contract

`src/exl3` is a vendorable unit: a Zig module, `sushi_exl3`, that another MLX host can root without any Sushi file.
mlx-serve vendors it. This file is the contract for a re-vendor: what the unit holds, the names a host may call, the
host surface it owes the unit, the state the unit keeps behind the calls, and what mlx-serve has today. The code is
described in [engine-exl3-experts](engine-exl3-experts.md); the pack format it reads in [pack-format](pack-format.md).

Renaming a name below, or changing its signature or meaning, breaks a host's next re-vendor: add new names instead, and
say so in the CHANGELOG. A change that keeps the names, passes `zig build exl3-host-check` and the module's tests
(`exl3-test-N` on `zig build test`) is safe to re-vendor.

## What the unit holds

- **Experts:** EXL3 MCG and MUL1 packs and STQ packs (STD1, SPEC1 and PAIR1 tables): banks, rate groups (`Ragged`), the
  decode chain, the prefill GEMM, the shared-expert folds and riders, the prefill grid.
- **Trunk linears:** the STQ trellis projections of a dense trunk: `trunk` (`exl3/trunk.zig`), `linear`, `trunk_gemm`,
  `project_rows`.
- **Pack reading:** `pack` (`exl3/pack.zig`, std only): tensor keys, bank completeness, tile order, rate-group spans.
- **`version`:** the Sushi version the copy was cut at, a string literal in `root.zig`. A Sushi test checks it against
  `build.zig.zon`, so a version bump fails until the literal follows.

The unit names no architecture; Sushi's per-arch code sits outside it. The microbenches are not in it: they are the
module `sushi_exl3_ubench` (`src/exl3_ubench/`), which only `sushi-ubench` builds.

## The host surface

The unit imports `std`, `builtin`, its own files and one module, and nothing else: `mlx_host`, a root file with
exactly three `pub const`s: `mlx`, `log`, `io_util` (below). The build owes it nothing else.

`zig build exl3-host-check` compiles the unit and its tests against `src/exl3/host_min.zig`, a host root that exports
exactly `mlx`, `log` and `io_util` (Sushi's own files behind them). It also walks the unit's public declarations, down
to the members of each type in a file (`linear.Geo.get`, `project_rows.launchRows`), so code no test calls is covered.
A fourth `mlx_host` name fails that build, not a host's. The check compiles only: nothing links or runs.

What the three names must provide (the names the unit calls, which the host check enforces):

| name | members the unit uses |
|---|---|
| `mlx` | the mlx-c bindings: the `mlx_*` types and the 97 `mlx_*` names (types and functions) the unit calls (`grep -oh 'mlx\.mlx_[a-z_0-9]*' src/exl3/*.zig`); the helpers `check`, `getShape`, `gpuStream`, `streamIsGpu`, `requireGpu`, `errorPending`, `dropLatchedErrorUnless`, `installErrorHandler`; and `kernel_keys` (`count`, `trackForTest`) |
| `log` | `info` and `warn` |
| `io_util` | `Stopwatch` (`init(io)`, `read`) |

The unit never imports a Sushi file by path.

## Public names

All names are on the module root (`@import("sushi_exl3")`); a nested name is `file.name`. A host that serves only the
experts calls the first four groups; a host with its own weight loader that also binds a dense STQ trunk adds the
fifth.

**Reading a pack**

| name | used for |
|---|---|
| `Spec`, `parseExpertQuant`, `parseQuantBlock` | `expert_quant` from `config.json`: `rate`, `codebook`, `window`, `rates`, `table`, `tile_order` |
| `shardStampAgrees`, `shardTileOrderAgrees` | a shard's metadata stamp against the `Spec` |
| `admitTopK`, `trellisAdmitted` | refusing a pack the kernels cannot serve, by name, at load. Against `Spec.rates` too for STQ |
| `pack` | `Component`, `Projection`, `Part`, `exl3TensorKey`, `exl3TensorKeyWithPrefix`, `EXL3_PREFIXES`, `exl3BankComplete`, `hasAnyExl3Key`, `exl3LayerGrouped`, `hasGroupedExl3Index`, `tileOrderOfMap`, `tileOrderOfDir`, `LayerGroups`, `GroupPieces`, `LayerDims`, `GroupedSpans`, `groupedLayerSpans` |
| `group_layout` (`GroupLayout`) | a layer's rate-group (`<proj>.gN.*`) names, shapes and dtypes, their stored bytes, a router index to (group, local expert); `Name.parse`, `trellisRate`, `tileDims` read `trellis` and `trellis_pm` |
| `codebook_tables` | the named STQ tables, their digests and rate ranges |

**Decode spec**

| name | used for |
|---|---|
| `format.Decode`, `format.Codebook`, `format.Window`, `format.Rate`, `format.TileOrder` | the decode passed to every dispatch. `Decode.table` is null for MCG and MUL1; a std1 or spec1 `Decode` without its table is refused by name (`Exl3CodebookTableUnknown`) |

**Routed experts**

| name | used for |
|---|---|
| `Proj`, `Bank` | one projection's `trellis`/`suh`/`svh`; a layer's `gate`/`up`/`down` |
| `moe`, `moeWithShared`, `moeWithGatedShared` | the routed SwiGLU: decode chain up to `DECODE_ROWS_MAX` rows or any verify call, else the prefill GEMM. The shared variants fold an already computed shared expert into the decode reduce |
| `moeClamped`, `moeClampedWithShared`, `validateClampedProjection` | the clamped SwiGLU (GLM, DeepSeek) |
| `rate_groups.SharedExpert.lane`, `rate_groups.SharedLane` | a shared bank riding a rate-grouped decode chain (DeepSeek's): `SharedLane.read(bank, rb)` reads its rate, widths and one-row cooperative arms once at load (`SharedLane.of(bank)` reads them before the `Ragged` is known; `fits(rb)` checks them against it), and `SharedExpert.lane` (a `?*const SharedLane`, default null) saves the launch the bank's three shapes |
| `fold_gated.moeGatedSharedFoldedGeo`, `rate_groups.moeGatedSharedFoldedGeo`, `riders_gated.geometry`, `riders_gated.geometryFits` | the gated shared-expert fold (`moeGatedSharedFolded`, one-bank and rate-grouped) with the shared expert's launch geometry given: `geometry(sh, hidden, rows)` read once at load, for rows 1 to `project_rows.NAX_ROWS_MAX` (the fold serves up to `riders_gated.max_rows`). The `Geo` entry takes a `*const riders_gated.Geometry` and returns null, having made nothing, for one that does not fit `x` or `sh` (`geometryFits`); `moeGatedSharedFolded` is `geometry` then the `Geo` entry, with the same bytes |
| `fold_gated.BankShape`, `fold_gated.bankShape`, `fold_gated.moeGatedSharedFoldedBound` | the one-bank gated fold with the routed bank also read at load: `bankShape(bank, hidden)` validates the bank and returns its shapes (null: it does not fit the fold), and the `Bound` entry takes that shape beside the `Geo` entry's geometry, with the same bytes |
| `Ragged`, `moeRagged` | a rate-grouped layer as one ragged bank, one decode dispatch per stage (STQ) |
| `moeGroups` | a rate-grouped layer as one `moe` pass per group, summed in f32 |
| `kernels.DECODE_ROWS_MAX`, `kernels.usesPrefillArm`, `kernels.rowsOfShape`, `kernels.gemmNaxOn`, `kernels.moePrefill` | the host's row planning around the dispatch; `moePrefill` takes a last `window_meta: ?bool` (null: the geometry decides) |
| `coop_rows2`, `fold_clamped`, `fold_gated`, `riders_gated`, `prefill_grid`, `shared_trellis`, `rate_groups` | the verify-row kernels, the shared-expert folds and riders, the prefill grid transpose, the rate-group kernels: Sushi's arch folders call them, `moe*` reach them for a host. A host binding only the `moe*` entry points never names them |

**Trunk linears** (an STQ projection of a dense trunk; one bank, trellis `[1, in/16, out/16, n]`)

| name | used for |
|---|---|
| `trunk.Triple`, `trunk.isStq`, `trunk.outDim` | a bank carried in an affine triple's slots (`w`, `s` = suh `[1, in]`, `b` = svh `[1, out]`); tells it from an affine or dense weight (U16) and reads its output width |
| `trunk.matmul(s, x, w, suh, svh, dec, wide)` | `x` through one bank in x's dtype, each row rounding as it does alone (exact rows); `wide` marks a spec verify, which keeps that arm up to 16 rows |
| `trunk.fuse`, `trunk.sharedProject`, `trunk.Shared`, `trunk.SHARED_ROWS_MAX` | projections that read the same rows, stacked into one bank and served by one launch. `fuse` also fills the stack's `linear.Shared.geo`, so it can fail through `project_rows.pinCfg`; it computes the geo before it mutates anything |
| `trunk.matmulGeo(s, x, w, suh, svh, geo, dec, wide)` | `matmul` for a bank whose launches `geo` holds: the weight's shapes and the launch's configs are not read again. A call `geo` does not serve (another dtype, rows past the multi-row arm) is `matmul` |
| `linear.Geo` (`Geo.init`, `Geo.get`, `row`, `launch`, `Geo.Row`) | what a one-bank (or stacked) trunk linear's decode-width launches need from its weight, read once at bind: shape, rate, and per row count 1 to 16 the launch geometry and pinned configs, in the `project_rows.Mode` it is bound for. `Geo.get(trellis, bounds, mode)` returns a `*const Geo` shared by banks of one shape and `mode.exact`; `init` makes an unshared one. Only `mode.exact` shapes the geometry: `{ .exact = true }` is the trunk's (NAX at every width), `{}` a plain decode's (the SIMD arm at one row). A weight the multi-row kernels do not take gets an empty geo |
| `linear.Shared.geo` (`?*const Geo = null`), `linear.linearGeo`, `linear.linearSharedGeo` | the launch from a geo, for bf16 `x`. `linearGeo` returns null and `linearSharedGeo` returns false, having made nothing, when the geo does not serve the call (wider rows, `force_arm`, an empty geo, a launch of the other `exact` than the geo's, a NAX launch after the latch set); the caller then runs `linearWith` / `linearSharedWith`, which gives the same bytes |
| `project_rows.launchRows`, `RowsLaunch`, `PinnedCfg`, `pinCfg` | `projectRowsWith` for operands whose shapes the caller holds (`RowsLaunch`) and configs it may have pinned (`pinCfg`, for a load to call, not a forward). `projectRowsWith` is `launchRows` after it reads the shapes. `buildRowsCfg` and `buildFinishCfg` behind `pinCfg` are private |
| `linear.linear`, `linearWith`, `linearGrouped`, `linearGroupedWith`, `linearShared`, `linearSharedWith` | a one-bank projection at any row count: `linear.Opts{ .rows, .decoded_billed }`; the plain names take the default options |
| `linear.Shared`, `linear.SHARED_MAX`, `linear.gemmInputs`, `linear.windowTable`, `linear.decodeDense` | stacked projections; the GEMM arm's bank ids and window table (built once per `(groups, rows, win)` in a small cache per side that `dropHostArrays` frees); the f32 host decode a test multiplies beside it |
| `project_rows.projectRows`, `projectRowsSegmented`, `projectRowsWith`, `Mode`, `PROJECT_ROWS_MAX`, `NAX_ROWS_MAX`, `rowsMax`, `rowsGeometry`, `RowsGeom` | the dense multi-row projection behind `linear`; `Mode{ .exact, .wide }` is a parameter of every call |
| `trunk_gemm.decodedServes(billed, ...)`, `DECODED_MIN_ROWS`, `DECODED_MAX_WEIGHTS`, `DECODED_SCRATCH_BYTES`, `denseOrderMatches`, `innerDense` | the decoded-weight arm of a wide linear: a caller that bills the scratch passes `decoded_billed` |

Sushi's weight-map binding of the trunk (`load`, `drop`, `view`) is `src/trunk_stq.zig`, outside the unit.

**Not API.** `f32_truth` and `synth` are test-only exports (below). `fuse.zig`, `kernel_source.zig` and
`sorted_windows.zig` are not on the root: they ship with the unit but a host reaches them only through the names above.
Every `pub` inside a kernel file that the tables do not list (kernel sources, fixtures, `*_for_test`) is an internal
the arch folders and tests use; it may change in any release.

## Renamed names

The unit names no architecture: files and public names say what the code does or the geometry it serves. The old
names are those of 1.5.1 and earlier; 1.6.0 keeps no alias for them, so a host renames them when it re-vendors.

| 1.5.1 name | name now |
|---|---|
| `glm_group2` | `coop_rows2` |
| `glm_fold` | `fold_clamped` |
| `qwen_riders` | `riders_gated` |
| `qwen_fold` | `fold_gated` |
| `glm_prefill_grid` | `prefill_grid` |
| `rate_groups.qwenFoldedCount`, `qwenLaunchCount` | `rate_groups.foldedCount`, `foldLaunchCount` |
| `format.Codebook.servesGlmFastPaths` | `format.Codebook.servesLaneFastPaths` |

**Kernel names.** The Metal kernels the unit builds name no architecture either. A kernel name is an in-process MLX
JIT key and nothing persists it (no cache file, no pipeline archive, no log line a doc or test quotes), so a host that
re-vendors changes nothing but the strings. 1.6.0 names:

| 1.5.1 kernel name | name now |
|---|---|
| `sushi_glm_lane_prepare`, `sushi_glm_lane_unsorted`, `sushi_glm_lane_pair`, `sushi_glm_lane_pair_one` | `sushi_lane_prepare`, `sushi_lane_unsorted`, `sushi_lane_pair`, `sushi_lane_pair_one` |
| `sushi_glm_down_lane_coop`, `sushi_glm_down_lane_prepare`, `sushi_glm_down_one` | `sushi_down_lane_coop`, `sushi_down_lane_prepare`, `sushi_down_one` |
| `sushi_glm_prefill_grid_transpose` (+ codebook and window suffix) | `sushi_prefill_grid_transpose` |
| `sushi_glm_exl3_group2`, `..._group2_tile2_lane`, `..._group3_tile2_lane`, `..._pair_group2`, `..._pair_group2_tile2_lane`, `..._pair_group3_tile2_lane` | `sushi_exl3_group2`, `sushi_exl3_group2_tile2_lane`, `sushi_exl3_group3_tile2_lane`, `sushi_exl3_pair_group2`, `sushi_exl3_pair_group2_tile2_lane`, `sushi_exl3_pair_group3_tile2_lane` |
| `sushi_glm_fold_*` (`fold_clamped`) | `sushi_clamped_fold_*` |
| `sushi_qwen_fold_*` (`fold_gated`) | `sushi_gated_fold_*` |
| `sushi_qwen_rg_fold_*` (`rate_groups`) | `sushi_rg_fold_*` |
| MSL function `qwen_qmv_rows` (inside `riders_gated`'s source) | `qmv_rows` |

The kernel sources and dispatch are byte for byte the same; only the strings differ.

Test fixtures renamed without an alias (not host API): `kernels.MimoMoeCase`, `MimoMoeFixture`, `mimoMoeFixture`,
`mimoPrefillArm`, `MIMO_BANKS` are `MoeCase`, `MoeFixture`, `moeFixture`, `moePrefillArm`, `SERVED_W12_BANKS`.

**Signatures that changed with no alias**, so a host calling any of them updates on a re-vendor: `projectRows` and
`projectRowsSegmented` (a last `mode: Mode`), `rowsGeometry` (a last `mode: Mode`), `rowsMax` (a `mode: Mode`),
`trunk_gemm.decodedServes` (a first `billed: bool`), `kernels.moePrefill` (a last `window_meta: ?bool`) and `linear.Geo.init` / `Geo.get` (a last `mode: project_rows.Mode`; the geo entry points below were added after the last vendored release). The process
globals they replaced (`project_rows.rows_exact`, `rows_wide`, `trunk_gemm.decoded_billed`,
`sorted_windows.gpu_window_prefill_force`) are gone. Per-call modes are parameters, never globals.

The geo entry points (`matmulGeo`, `linearGeo`, `linearSharedGeo`, `launchRows`, `moeGatedSharedFoldedGeo`) are additive: every older entry point
keeps its signature and its bytes, and is the fallback each geo call takes when it does not serve a call. A host that
calls only the older ones keeps working.

## State the unit keeps

A call is not a pure function of its arguments in these ways. A host serving several models or threads must know them.

**Production**

| state | what it does | what the host owes |
|---|---|---|
| `kernels.setDecodeParams(dec)` (`kernels.active_decode`, read by `activeDecode`) | the codebook and window the next dispatch decodes with; weight kernels are built per (codebook, window) | every `moe*`, `linear*` and `moeRagged` entry point sets it from its `dec` argument, so a host that calls them needs nothing. A host that calls a kernel entry point directly must set it first, per call, because several packs can be resident |
| kernel caches (`kernel_source.CfgCache`, per-kernel slots) | kernels JIT-built once per shape and decode, kept for the process | nothing is ever freed, and the caches are unguarded: call the unit from the one thread that calls MLX |
| host arrays per side (`linear.gemmInputs` ring, `rate_groups` unit score) | kept per side (`host_side.of`) until `dropHostArrays(side)`, rebuilt at the next ask | call `dropHostArrays(side)` on the thread that calls MLX once no model of that side is left; Sushi does at the last unload of each side |
| `linear.Geo.get` table, `project_rows.pinCfg` configs | process-lifetime, unlocked, keyed by shape (the geo also by dtype, bounds, `mode.exact` and the NAX state at bind), immutable once built | bind (`Geo.get`, `trunk.fuse`) and launch on the thread that owns MLX |
| NAX latch in a geo | `trunk.fuse` and `Geo.get` read `gemmNaxOn()` when they run, so a geo's geometry follows the latch as it stood at bind; each launch rechecks the latch and falls back to the shape-reading path once it has set | nothing |
| `kernels.gemm_nax_failed` | latches NAX GEMM off for the process when its kernel fails to build, then `gemmNaxOn()` is false | nothing; a host may read `gemmNaxOn()` |
| env, read once per process | `SUSHI_EXL3_GEMM_WIN` and `SUSHI_EXL3_WIN_ALIGN` (prefill window tunables), `SUSHI_FORCE_GPU_FAMILY_FALLBACK=1` (forces the non-NAX arms) | set before the first dispatch, or leave unset |
| env, diagnostics | `SUSHI_EXL3_STAGE_PROFILE`, `SUSHI_EXL3_SWIGLU_MAXABS`, `SUSHI_EXL3_UNION_HIST`, `SUSHI_DECODE_TICK_UBENCH` | none; each only adds a log line (`[exl3-union]`, `[exl3-counts]`) or a timing |
| log lines | `log.info` `[exl3-groups] ... engaged` once per arm | a `log.info` that can write |

**Test-only** (never set them in a served process)

| name | what it does |
|---|---|
| `linear.force_arm` | forces `linear` onto one `Arm` (`rows`, `gemm`, `gemv`) |
| `trunk_gemm.enabled`, `trunk_gemm.decoded_calls` | `enabled` false keeps every linear on the sorted GEMM; `decoded_calls` counts decoded-path planes |
| `kernels.nax_reference_override`, `gemm_simdmat_force`, `unfolded_for_test`, `grouped_widest`, `one_row_coop_off`, `rows_baseline_kernel`, `prefill_chains` | A/B arms of the GEMM, grouped and one-row kernels, and test counters |
| `rate_groups.prepared_force`, `group_force`, `fused_shared_first`, `fold_clamped.shared_first`, `fold_gated.shared_first` | forced arms of the grouped and fused chains |
| `f32_truth` | f32 reference helpers the geometry and real-pack parity tests share |
| `synth` | random banks (`makeBank`, `freeBank`, `randomWords`) for tests and the benches |

A test-only global is a process global on purpose and is not part of a call's inputs: a host that sets one leaks it
into every dispatch.

## What the host owes the unit

- **`DECODE_ROWS_MAX` rows is the host's eager boundary.** A forward of more than `kernels.DECODE_ROWS_MAX` rows that
  is not a verify forward takes the prefill GEMM (`kernels.usesPrefillArm`), which may read the routing back on the
  host to build its window table (MiMo and Flash-Next geometries build it on the GPU; every other geometry on the
  host). That read evaluates the stream, so an input the host fills after building the graph (a deferred leaf, such
  as Flash-Next's n-gram embedding) is read empty: gather such inputs eagerly at those widths. Up to
  `DECODE_ROWS_MAX` rows, and any `verify_rows` call, never read back.
- **Refuse what the host cannot serve.** `parseExpertQuant` admits `format: stq`, so a host either refuses an STQ
  `Spec` or passes `Spec.table` in `Decode.table` to every dispatch and checks each bank with `trellisAdmitted`
  against `Spec.rates`. It refuses `Spec.tile_order == .pair_major` by name if it binds only `…trellis`: those packs
  store `trellis_pm` ([pack-format](pack-format.md#tile-order)).
- **Rate-group and pruned packs** ([pack-format](pack-format.md#rate-groups-per-expert-rates-uneven-expert-counts)) take
  `GroupLayout` at load and `moeGroups` or `Ragged`/`moeRagged` at dispatch.
- **MiMo and GLM packs serve through the same names.** MiMo binds its banks to `moe`; GLM to `moeClamped` (limit =
  `swiglu_limit`, 10) with the host adding its own ungated shared expert, and `trellisAdmitted` bills the config's k, so
  GLM's K2.25 layers sit under k2.5. The packs' source-layout trunk (block-FP8 QKV, `kv_b_proj`, hyper-connection
  names) and the shards the index owns for a duplicated tensor are the host's to read; none of it is module API.

## What mlx-serve has today

mlx-serve vendors Sushi 1.3.0's `src/exl3` as `src/sdk/quant/exl3/sushi/` (ddalcu/mlx-serve#978). It has no `lib/sushi`
submodule and pins no Sushi commit, so a Sushi change reaches it only by a re-vendor and cannot break its build. The
copy builds as module `sushi_exl3` with one import, `mlx_host`, which re-exports `mlx`, `log` and `io_util` from
mlx-serve's own host root. It has diverged from `src/exl3` (Metal sources in `.metal` files, its own host env).

Sushi 1.3.0 already had the STQ experts: `codebook_tables`, `rate_groups`, `layout` (`group_layout`), `glm_group2` and
`glm_prefill_grid`. Those two root names, and `Codebook.servesGlmFastPaths` inside their files, are the only 1.5.1
Zig names the copy uses (mlx-serve at v26.11.2, nothing outside the vendored folder names them); a re-vendor renames
them to the table above. The copy also builds fourteen kernels under 1.5.1 names (`sushi_glm_lane_prepare`,
`_lane_unsorted`, `_lane_pair`, `_lane_pair_one`, `_down_lane_coop`, `_down_lane_prepare`, `_down_one`,
`_prefill_grid_transpose` and six `sushi_glm_exl3_*`, two of them 1.3.0 spellings, `_group2_lane` and
`_pair_group2_lane`); the kernel table above has the new names, and a 1.3.0 spelling takes the same prefix change
(`sushi_exl3_group2_lane`). What the copy lacks is everything after 1.3.0:

- the trunk contract: `linear`, `trunk_gemm`, `project_rows`, `trunk`, `fuse`, `pack`;
- `kernel_source`, `sorted_windows`, `shared_trellis`, the folds and riders;
- the renames and the per-call modes above, and `root.version`.

It therefore cannot serve a `trunk_quant` pack (a dense trunk in STQ), nor a rate-grouped Qwen pack: its `qwen4_exp`
loader binds one uniform `Bank` per layer (`switch_mlp.{gate,up,down}_proj.{trellis,suh,svh}`), checks each projection
with `trellisAdmitted` against the config's `num_experts` (every expert of a projection in one bank at one rate; gate,
up and down may differ from each other), and refuses the rest.

A re-vendor takes the whole `src/exl3` (without `host_min.zig`) and keeps the names above at the same paths.
mlx-serve's extracted Metal files named after a Zig file (`glm_group2_*.metal`) are renamed to the new file names
(`coop_rows2_*.metal`, ...), and the kernel names inside them take the table above. mlx-serve's own
`sdk/quant/exl3/contract.zig` labels its one-row launch `sushi_glm_down_one`: that string only names the contract's
own test kernel, so it needs no change to build, but it should read `sushi_down_one`.

## MLX pins

Each repo builds its own MLX and mlx-c; the unit adds no pin of its own. Both repos pin the same commits: MLX 64ea011
(v0.32.3) and mlx-c 56b2d39. Move both pins together.
