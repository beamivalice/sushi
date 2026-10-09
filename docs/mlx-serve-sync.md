# mlx-serve sync: what upstream changed and what we took

The ledger of reviews of upstream mlx-serve (`ddalcu/mlx-serve`, the engine this repo forked) against Sushi. Each pass
reads the upstream commits since the cursor, decides per commit whether it speeds up or fixes a served architecture
(`qwen4_exp`, `mimo_v2`, `glm5_next`), and records the verdict here. The other direction, mlx-serve consuming Sushi, is
[mlx-serve-integration](mlx-serve-integration.md).

Index: [CLAUDE.md](../CLAUDE.md#docs-index).

## Cursor

| | upstream `origin/main` | Sushi `main` | date |
|---|---|---|---|
| fork point | `ef5e667d` | | 2026-09-17 |
| previous pass | `25e94c33` | | 2026-10-04 |
| **latest pass (reviewed through)** | `a21784b0` | `92446711` | 2026-10-09 |

Next pass starts at `a21784b0`.

## How to run a pass

```bash
git -C <mlx-serve checkout> fetch origin
git -C <mlx-serve checkout> log --format='%h %ad %an %s' --date=short <cursor>..origin/main
```

1. Drop what Sushi does not serve: image, audio, music, embeddings, llama.cpp, ds4, nemotron, Linux/CUDA, UI and app.
2. Speed candidates: `git show --stat` each remaining `perf`/`feat` commit and compare with the matching
   `*-perf` and kernel doc's ruled-out list before proposing a port. A lever already ruled out here stays out.
3. Bug fixes: read the upstream diff, then trace OUR code. A verdict is `HAS-BUG` only when the failing input reaches our
   code. Our parsers, grammar and sampler were rewritten after the fork, so a commit message proves nothing.
4. Record every reviewed commit below, ported or not, then move the cursor.
5. Upstream numbers come from other boxes (an M5 Ultra for GLM) and other baselines. They size an idea; they never
   count as a Sushi measurement.

## Speed

| upstream | what | verdict |
|---|---|---|
| `846977ac` | GLM DFlash2: confidence-driven row count over a measured 1..16-row ladder, plain rounds, verbatim copy chains up to 15 drafts, row paths to 16 | **candidate**, see below |
| `c84a5a63` | GLM: fuse the KDA verify step, batch decode across slots | covered: grouped GLM forwards and the KDA tree prework exist; a fused T3 KDA core measured -0.7% ([glm5-kernels](glm5-kernels.md)) |
| `03cc3ca2` | Bonsai ternary M4 verify lane | not served |
| `340016fc` | 16 lanes per output row in the affine MoE down+reduce, G17 | not applicable: our experts are EXL3; the change also alters accumulation order, so it would need the KLD gate |
| `431ac5d7`, `ad65559b` | hyper-connection one-row up launch, one read for 1 to 16 rows | Qwen only, and our HC path is already row-grouped ([qwen4-perf](qwen4-perf.md)) |
| `22f068f6` | KDA decode step probes its threadgroup size | robustness only: `glm5_kda_fused.zig` launches a fixed 1024-thread group with no probe, which only a virtualized GPU with a lower cap would hit |
| `178c43da` | upstream serves Sushi's GLM and MiMo packs in-process | upstream runs our engine; nothing to port |

MiMo: no upstream performance commit since the fork that Sushi lacks.

**Open candidate: longer verbatim-copy chains on GLM.** Ours is fixed at 3 drafts (`Generator.glm_lookup_drafts`, a
four-row verify); upstream reports copy at about 3x serial against our recorded 1.6x ([glm5-arch](glm5-arch.md)). It needs
verify rows past 4 in the routed experts, MLA attention and the KDA step, and our grouped 2-row forward already costs
about 33% over a serial token. First measure a 6 to 8 row verify on one real copy prompt; port only if the round cost
per row stays flat enough. Wider trees and a measured-cost planner are already ruled out.

## Bug fixes

| upstream | what | verdict | status |
|---|---|---|---|
| `9c0ca7fe`, `3129a772` | json_schema: `$ref` resolution, typeless nodes accept objects, cyclic `$ref` relaxes to any, `~0`/`~1` | **HAS-BUG**: no `$ref` handling; a typeless node forbids every key, so `items: {"$ref": ...}` completes only as `{}` (`response_format` without tools) | open; S for the typeless nodes, M for `$ref` |
| `f3da8d69` | `min_p` from the request and `generation_config.json` | **HAS-BUG**: the field is dropped silently | open; L (five sampling sites, spec/MTP densities must stay identical); only if llama.cpp-style clients matter |
| `6ddac95f` | tool name escaping in the stream delta; name ends at `>`, newline or `<` | partial: delta already escapes; a missing `>` drops the call and leaks the tool markup as content | open; S, rare on Qwen3.8 and MiMo |
| `348fd414` | `/v1/completions` token-id prompts, `echo` refused | **HAS-BUG (minor)**: ids get a misleading 400; `echo:true` is ignored | open; S, port `parseCompletionPrompt` and the `echo` 400 |
| `2baca521` | forced tool opener must not end on a lone `=` | partial: no double open, no wrong name; the opener ends on `=` and names are encoded without it, so a multi-tool `required` choice starts off-distribution | open; needs a live multi-tool `required` test before changing |
| `8f7f79f3` | coerce `oneOf`/`anyOf` object params from JSON text | not affected: `declaredJsonType` already resolves unions | closed |

| `41896655` | a grouped MTP verify's GDN rows take the solo verify kernel | **HAS-BUG**: grouped Qwen verify (`verifyRowsJoined` into `gatedDeltaNetProjected`) runs the composed chain while solo verify and serial use the fused kernel, so a greedy answer can depend on who shares the round | open; S (lift the guard in `transformer.zig`), then run the row-axis test with `QWEN4_TEST_MODEL` |
| `5596adba` | hybrid tool round resumes where decode ended, not ~31 tokens before the prompt end | **HAS-BUG (speed)**: the newest SSM checkpoint sits 30 tokens before the prompt end, so a Qwen agent round re-forwards 30 tokens plus the whole previous reply | open; M (snapshot the live SSM state at the committed length; the MTP tail hidden and the QSA history need care); the largest speed win in this pass |
| `2137bd00` | release every handle on MTP head forward error paths | partial: leaks exist only under test fault injection, not reachable in serving | open; S errdefers, low value |
| `a3804125` | auto `--max-resident-mem` is Metal's working-set limit, not 80% | partial: we cap at 80% but a lone model loads past the cap; only co-residence is capped | optional; S, loosens eviction headroom |
| `431b3537` | omp sends `reasoning_effort` | not affected: `launch.zig` already does | closed |

Items of the previous pass, checked as done: GLM tokenizer `ignore_merges` (`7c7bdcb1`), predraft after budget/EOS
(`9a6a605c`), K/V 16-byte alignment guard (`aec87e66`), request-outcome metrics counted once (`518d887b`, `0f0b1adb`),
disk free-space probe only before a store (`fecf6f35`), MTP sidecar admission (`00ea1b23`). Router top-k registers and
counting-sort routing are recorded dead ends.

Ranked next ports: (1) typeless/`$ref` schema nodes, a valid request returns wrong content; (2) `41896655`, one line that
restores grouped = solo verify; (3) `5596adba`, the speed win for every Qwen agent tool round.

## Not reviewed

Everything the filter in step 1 drops, and upstream commits dated after `a21784b0`.
