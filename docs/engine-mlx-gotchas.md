# Engine: MLX and Zig gotchas

The ways MLX, mlx-c and Zig have bitten this engine: dtype promotion, lazy evaluation and barriers, view vs copy
semantics, error handling, the allocator pool, and a few Zig and tokenizer traps. Read this before writing graph-
building code or anything that crosses the mlx-c boundary.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [engine-kernels](engine-kernels.md),
[engine-memory-admission](engine-memory-admission.md), [server-lifecycle](server-lifecycle.md).

## Runtime revision and C wrapper

Sushi follows mlx-serve's MLX0.32.3 pin `64ea011c` and mlx-c `56b2d39`.
The C wrapper patch supplies the new optional global-scale argument in its
C++ `gather_qmm` call; the C ABI stays unchanged. The build stamp includes
both source revisions and the hash of every patch. `tests/test_mlx_staged_nax.sh`
checks those fields against the source, along with NAX symbols and min-OS.
An older staged library can load successfully while retaining a sorted-row
overflow bug above32K; library presence alone does not qualify the runtime.
Sorted `gather_qmm` requires ascending expert IDs in that bank's own ID space.
After slab remapping, sort each reference arm separately and invert its output
permutation before comparing bytes; grouping by the other arm's IDs is invalid.

<a id="transient-cap"></a>
## The transient cap patch

`patches/mlx-transient-cap.patch` (mlx) and `patches/mlxc-transient-cap.patch` (mlx-c C ABI) are applied by
`build-mlx.sh`; the stamp's `patch=` hashes all three patches. Stock `MetalAllocator` counts an imported (no-copy)
buffer in `active_memory_`, `malloc` never refuses and `set_memory_limit` only throttles eval, so nothing could bound
what MLX allocates itself (graph intermediates, custom-kernel outputs, the freed-buffer pool).
- `external_memory_` is the imported bytes (`make_buffer` adds, `release` subtracts); transient = active - external.
- `mlx_set_transient_limit` (0 = off, the default): a new buffer (not a pool hit) whose transient + pool + size passes
  it first trims the pool by the excess, then throws `std::runtime_error` starting `[mlx-transient-cap]`.
- `mlx_get_transient_peak` / `mlx_reset_transient_peak`: high-water of transient; a reset restarts it at the current level.
- The throw is an ordinary MLX exception: mlx-c turns it into the latch, `mlx.checkError` raises
  `error.TransientCapExceeded` (a 503 naming the cap, not the memory 503) and `settledErrorName` consumes it. After a
  refusal drain the stream (`mlx_synchronize`) and clear the latch before the next op; the failed array stays valid and
  evaluates once there is room.
- A primitive allocates its outputs in `eval_gpu`/`eval_cpu` on the evaluating thread inside `eval_impl`, whose catch
  flushes the open streams before it rethrows, so the throw unwinds.
- The limit is absolute and the peak is too, so a phase installs `active - external + its bill` at its start
  (`dsv41_borrow`, [dsv41-memory](dsv41-memory.md#transient-caps)); a cap is judged against a peak less the state it started from.
- `SUSHI_DSV41_TRANSIENT_PEAK=1` logs `[transient-peak]` per DeepSeek phase (decode, DSpark draft/verify, prefill chunk,
  vision encode). Guards: the `transient cap` tests in `mlx.zig`.

## The one MLX thread

The inference thread is the SOLE mlx caller (even frees). `Slot.deinit` runs on conn threads: it stores marks, the
inference thread frees. A pointer-keyed cache is invalidated by an ATOMIC MARK, never an off-thread free.

## Errors

- **An MLX failure is CATCHABLE; mlx-c's DEFAULT handler `exit(-1)` is what killed us** (`mlx.installErrorHandler`
  once in `main()`): `checkError` per prefill chunk (before snapshot/persist) + `checkErrorDecode` per tick; a
  latched error never 200s; streaming shares `mapGenerationError`.
- A swallowed failure must DROP the latch it raised (`dropLatchedErrorUnless(had_error)`, passing the
  `errorPending()` read BEFORE the op). Guard: `tests/test_mlx_error_recovery.sh`.
- **A failure caught as `error.MlxError` leaves the latch SET**: the next request's first `checkError` consumed it and
  failed at once as `OutOfMemory`. Name it with `mlx.settledErrorName(err)` where it is caught: that consumes the latch
  and reports a Metal out-of-memory behind it as one.
- A client-supplied PATH is proven on OUR side of the mlx boundary (stat → 400; an MLX error there latches).
- In the unit-test binary a failing test's MLX error stays latched for the next test: a decode tick consumes it and
  errors its slot. A test that runs a tick asserts `!mlx.errorPending()` first.

## Dtypes

- **An f32 SCALAR array promotes every bf16 operand it touches**: scalars go through `scalarOf(v, dtype)`; a chain
  that returns f32 BY DESIGN makes the CALLER own the dtype; `[dtype-trace] residual widened` is the tell.
- A load-time constant table in the WRONG DTYPE silently widens every read (`constTableAs`).
- Dtype-gate any `mlx_array_data_float32` load-time read.

## Custom kernels

- **MLX binds a `metal_kernel` input of fewer than 8 elements in the `constant` address space**: a kernel source never
  declares `const device T*` from an input (`auto p = in + off`); the FP8 tile kernel's scales did, and a tiny linear
  failed to build at 9..128 rows. Guard: the fp8_block small-scales test.
- **Every template arg value is its own Metal library, compiled on the inference thread at first eval (tens of ms)**:
  MLX keys a `metal_kernel` library by name, source, template args, each input's dtype and binding (scalar, `constant`
  under 8 elements, device) and output dtypes. A row count, pick count, split count or context width never rides a
  template arg: read it from `threads_per_grid`, an input's `_shape`, or a small input buffer. DeepSeek's index top-k
  width and packed-KV row counts did, and decode below 512 tokens of context compiled a library every token (the
  "slow state"). `SUSHI_KERNEL_BUILD_LOG=1` logs each new key; guard: the `dsv41 kernels:` zero-new-library tests.

## Laziness and barriers

- **A host read inside a layer loop is a GPU BARRIER**: defer non-consumed reads into ONE batched eval.
- A rollback that re-forwards is a SECOND forward: capture only what verify overwrites, truncate the rest by offset.
- A multi-token forward is not a prefill (`prefillEvalCadenceApplies`, seq ≥ 32).
- A weight outside every warmup forward is still LAZY at serve time (`appendWeightArrays`).
- Helpers that could take a block use the `_axis` op.

## Views, copies, ownership

- Slice-born weights into gather_qmm/quantized_matmul are `mlx_contiguous`-materialized at load; mlx
  `Copy`/`contiguous` are VIEW ops (a slice OUTLIVING its parent goes through `materializedOwnedCopy`).
- A raw data-pointer read must PROVE row-major contiguity; a helper that materializes a VIEW owns it (`takeContig`).
- Lazy transpose/broadcast descriptors can expose placeholder contiguous strides; evaluate the view before a
  pointer/stride oracle, and do not infer physical copies or broadcast storage from unevaluated metadata.
- A weights MAP outliving the model pins every buffer; mlx-c `iterator_next` hands a +1; `mlx_array_new_data`
  COPIES shape-worth of bytes.
- **A refcount-shared snapshot makes every later write copy the whole buffer** (MLX donates only a sole owner's
  buffer): a spec rollback that can truncate by offset takes no `KVCache.snapshot`. Speculative branches that share
  a reserved buffer keep their new rows in a small array beside it (GLM `State.pool_tail`).
- **`mlx_eval` returns before its command buffer lets go of the outputs**, so a write right after one can find its
  buffer shared and copy it: a chain of in-place writes drains the stream (`mlx_synchronize`) between evals.
- MLX releases an IMPORTED host buffer asynchronously ([engine-expert-streaming](engine-expert-streaming.md#io)).

## The allocator pool

- `mlx_clear_cache()` once per CHUNK and per emitted block, INTERVAL-based (`step -| last_clear >= 256`),
  un-skippable; `Generator.advanceStep` is the one step mover.
- The prefill loop's per-chunk clear stays: handing the pool to the next chunk measured 5-7% slower prefill and a
  3 GB larger footprint ([glm5-perf](glm5-perf.md#glm-prefill-pool-clear)).
- `active` flat while phys climbs = the POOL (`memory.cache_bytes`). RSS is blind to Metal.

## Zig

- A struct inside a generic fn that captures NO comptime param is memoized to ONE type.
- `openDirAbsolute` on an empty/relative path is ReleaseFast UB — guard every site.
- `.string` on unchecked `std.json.Value` panics.
- A test that aliases embedded bytes through an alignment cast is a coin flip per binary: copy fixtures to aligned
  storage (Debug builds catch what ReleaseFast hides).
- A failing `std.debug.assert` is UB in ReleaseFast: the full suite can pass while the same test, filtered alone,
  dies with SIGTRAP (inlining decides whether the trap is emitted). Rerun it under ReleaseSafe for the panic.
- Compile time: Sema and LLVM run on one core per binary. Comptime string work is interpreted: never `++` a byte at
  a time or `comptimePrint` large kernel templates (build them at first use instead); `zig build --time-report`
  names the slow decls.
- `std.testing.expectError` and `expectEqual` on a pointer print the value with `{any}`: one printer per payload
  type, tens of KiB of code for an engine struct. Use `test_expect.expectError` or `expect(a == b)` there.
- The unit tests compile as one binary per `test_buckets` entry in `build.zig`; a new file joins the last bucket.

## Tokenizer

- Special-token splitting only in `Tokenizer.encode` (first-byte buckets); any per-position loop over a
  vocab-derived collection needs an index.
- A hand-rolled pretokenizer is calibrated to ONE tokenizer.json — digit GROUPING is per-model (`digit_group`);
  cross-check `/tokenize` vs HF at bring-up.
- Unicode segmentation is model-local: the exact supported `Split` regex and `Isolated`/`invert:false` +
  `ByteLevel(use_regex:false, add_prefix_space:false)` pipeline select the rule. MiMo's word branch is `\p{L}+`;
  Qwen's is `[\p{L}\p{M}]+`. Marks therefore belong in different branches. Never select this by language,
  directory name or architecture, and never fix Thai by extending a shared block-range heuristic.
- These two grammars use Unicode 16.0 general-category tables (regenerate with
  `scripts/gen-tokenizer-unicode.py` on Python 3.14). Other pre-tokenizer pipelines retain their existing path.
  Exact token IDs matter: decoding back to the same text does **not** prove correct segmentation.
  Guards: hermetic cross-grammar BPE fixtures in `tokenizer.zig` + the format corpus;
  `tests/test_tokenizer_reference.sh` compares the three real tokenizer files to HF on CPU without loading weights.
  Its synthetic inputs are normalized first to isolate Split/BPE: the older missing NFC-normalizer behavior on
  decomposed text is a separate, still-open issue. `TOKENIZER_CASES_JSON` can add an already-normalized prompt pack.
- The cased `llama3` grammar needs `\p{Lu}` in the Split regex: contractions plus `\p{N}{1,3}` is also GLM's plain
  Llama-3 pattern, which is the gpt2 path with `.letters` word rules and 3-digit groups. The BPE `ignore_merges` flag
  (GLM sets it) is honoured in `bpeMerge`. The parity script covers GLM, Qwen and MiMo on code and tool-call text.
- DeepSeek's `Scope.out` handle belongs to the caller: a `block` result chained through a loop in a scope must be
  adopted with `sc.keep` or it leaks every iteration (the DSpark draft leaked its stage outputs per round). Guard: the
  `DSpark requests leave the MLX active memory flat` test in `generate.zig`.
