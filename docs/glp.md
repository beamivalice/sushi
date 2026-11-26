# GLP steering

A **GLP vector** (GGUF Layer Projection) is a few hundred KB of per-layer directions that change a
model's behaviour at inference time without touching its weights. The file is a GGUF v3 container
with a `glp.*` metadata namespace and `direction.N` F32 tensors; `direction.N` applies at layer `N`,
with no offset. The operation is **projective**:

```
h  <-  h - alpha * (h . d_hat) d_hat
```

— remove the component along the unit direction `d_hat` — not llama.cpp's additive
`h <- h + v`. The two are not interchangeable, and a projective direction loaded into an additive
consumer raises no error and produces wrong output, so the operation travels with the data in
`glp.mode` and an unrecognised mode is fatal. The format and its reader contract are specified in
[GLP.md](https://github.com/msuiche/weightless/blob/main/docs/spec/GLP.md) in the
[weightless](https://github.com/msuiche/weightless) repo.

## What this build implements

One contract, and nothing else:

| key | value |
|---|---|
| `glp.spec_version` | `1` |
| `glp.mode` | `project` |
| `glp.hook_point` | `residual_stream_post_layer` (absent means the same) |
| `glp.rank` | `1` (absent means 1) |

`glp.alpha_default` is the dose; `--glp-alpha` or `WEIGHTLESS_STEER_ALPHA` overrides it at runtime.
The dose is never folded into the direction: projection is quadratic in the direction's norm, so a
strength baked into the data would not mean what a caller expects. Directions are normalised to
unit norm on load, and `glp.content_sha256`, when present, is verified against the tensor bytes.

## The apply site

Every served arch is hyper-connection (multi-copy): the residual stream is `[T, C, n_embd]`, and
there is no plain `[T, n_embd]` stream between layers. `residual_stream_post_layer` therefore maps
to the **post-fold stream**, applied on the model's canonical reduction — the mean over copies:

```
h          <-  mean_c(stream)                       [T, n_embd]
stream_c   <-  stream_c - alpha * (h . d_hat) d_hat  for every copy c
```

Subtracting the rank-1 term from every copy shifts the mean by exactly `alpha*(h.d_hat)d_hat`, so
this is the spec's hc formula, not an approximation of it. The apply is in
`dsv41_model.zig`'s `buildLayer`, immediately after the layer's block returns the post-fold stream.
DeepSeek-V4.1 is the first arch; another arch refuses `--glp` at load by name.

## Fail-closed rules

The reader (`src/glp.zig`) refuses, with a named and actionable message, before any weight is read:

1. a file with no `glp.*` metadata at all — a legacy llama.cpp additive control vector (or a
   pre-GLP `dspark.*` file) — because this build has no additive path to fall back to;
2. `glp.mode` absent (means `add`), `add`, or any value other than `project`;
3. `glp.spec_version` absent or not `1`;
4. a `glp.hook_point` other than `residual_stream_post_layer` (`hc_fold_pre_engram` is a
   spec_version 3 site that also needs the companion attention-writer arm);
5. `glp.rank` other than `1`, and any `direction.N.j` tensor;
6. `glp.dir_scales` / `glp.layer_scales` under spec_version 1 (they require 2);
7. `direction.0` (this build implements no layer-0 apply), a direction past the model's layer
   count, a duplicate layer, a non-`direction.N` tensor, a non-F32 tensor, a shape that is not
   `[n_embd]`, a non-finite or zero-norm direction, and a `glp.content_sha256` mismatch;
8. a direction whose width differs from the loaded model's hidden size;
9. a second `--glp` on the same launch: two projections do not compose into a projection along the
   sum of their directions.

## Usage

```
sushi --model <deepseek-v4.1 pack> --glp glp.inkling2mini-GLP-48-L1-48.gguf
sushi --model <pack> --glp vector.gguf --glp-alpha 2.0
WEIGHTLESS_STEER_ALPHA=2.0 sushi --model <pack> --glp vector.gguf
```

The load logs `[glp] <path>: <n> layers steered, alpha <a>` before the weights load. `--glp-alpha`
outranks `WEIGHTLESS_STEER_ALPHA`, which outranks the file's `glp.alpha_default`.

## Tests

`src/glp.zig` embeds synthetic GGUF v3 fixtures written by `tests/glp/make_fixtures.py` (stdlib
only): one valid v1 project vector and one file per refusal path. They run in the host tier
(`zig build test-host -Dtest-filter=glp`), so the reader contract is checked without a model or a
GPU.
