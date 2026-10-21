---
name: release
description: sushi pre-release validation checklist, SemVer versioning, release steps, and CHANGELOG style. Use when preparing or cutting a release, running pre-release validation, or writing CHANGELOG entries.
---

## Pre-release validation

On the Apple M5 Max 128 GB, on the FINAL release tree, with a fresh `zig build -Doptimize=ReleaseFast`. Run the steps
directly, one at a time: no driver script, no pause/resume bookkeeping. A failed step goes to the owner before anything
else. Model loads take the GPU lock; step 2 waits for a quiet box (no builds, no AirDrop or other transfers).

| # | Step | Command | Pass |
|---|---|---|---|
| 1 | Suite + binary | `zig build test -Doptimize=ReleaseFast`; `zig build -Doptimize=ReleaseFast`; `sushi --version` | 0 fail; names `build.zig.zon`'s version |
| 2 | **Perf gate** | the `docs/bench/<ver>/` run (below) on Qwen3.8-Flash-Next-Sushi-2.6bpw, Sushi-4bpw, MiMo-V2.6-Flash-Sushi-2.3bpw and GLM-5.3-Flash-Sushi-2.4bpw | each model within noise of its previous `docs/bench/` report, speculative mode engaged; commit `docs/bench/v<ver>/`, append this release's `benchmarks.md` column |
| 3 | **KLD gate** | `sushi kld compare` 16x512 to first EOS on one Sushi pack and MiMo | within ~1% of its row in `docs/quality-kld.md` |
| 4 | Live | `test_format_matrix.sh`, `test_smoke_matrix.sh`, plus the live test of each area the release changed (no `llmprobe --quick`: step 2's full bench covers it) | all pass |
| 5 | CI | none to dispatch: `release.yml` runs the full suite on the macOS 26.2 runner before it packages, and a failed test leaves no draft. To check a tree without cutting: `gh workflow run release.yml -f dry_run=true` | green (the macOS 26.2 build gate) |
| 6 | Packs | each HF pack repo holds the shards, `ngram_table.bin` and its model card | card numbers match `docs/quality-kld.md` |
| 7 | Cross-engine (only before a public claim) | start each engine yourself, `./tests/bench.sh --url <host:port> -m <id> --full` | recorded in `~/.sushi/runs/bench-<tag>/`, engine named beside every win |

**Rules:**
- **Steps 2 and 7 are different questions.** 2 = "did our code regress", sushi only, every release. 7 = the public
  comparison; re-run it only when another engine's version bumps.
- **The perf gate times on AC power only.** Check `pmset -g batt` right before each model's run and log the power
  source beside it. A run that touched battery is discarded and rerun on AC: battery cost GLM 8–18% decode on 2026-10-11.
  - All other GPU work keeps the battery-above-30% rule.
- **The perf gate is four models: Qwen 2.6bpw, Qwen 4bpw, MiMo 2.3bpw, GLM 2.4bpw.** It is ONE run per model whose
  output IS the published `docs/bench/v<ver>/` report, in the format of `docs/bench/v1.2.0-dev/` (`summary.md` plus
  one `<model>.md` per run): pinned `npx llmprobe@<ver> --bench-only --rungs 2k,4k,8k,16k,32k,64k,128k --runs 2`, one
  model per server, GPU lock per model, fans at max and 3 min idle before the first server and after each one (60 s let heat
  carry into the next model), commit, binary SHA-256 and mtime in
  the header. A model whose speculative mode (MTP, DFlash2) did not engage is chased before shipping; a low cell gets
  one rerun on a quiet box before anyone bisects.
- **Each model's server starts on an EMPTY SSD prefix cache**: a fresh `HOME` per server (or `~/.sushi/kv-cache`
  emptied right before it). A shared cache over its cap made every boot evict other models' entries during the probe:
  MiMo prefill read 8–17% low on 2026-10-15, and an empty-cache A/B put v1.2.3 and 1.2.4 within 1%.
- **`--full`** takes median-of-3 per rung and climbs to 32k/64k; the default is one run per rung to 16k.
- **Never quote a win without naming the engine it is over.**
- **`benchmarks.md` gets one new column per release**, from step 2's report. Obey its header rules: tables
  only, M5 Max only.

## Release artifacts

The release record is `docs/bench/v<ver>/` and `benchmarks.md`, plus the raw llmprobe reports (JSON and HTML) under
`~/.sushi/runs/bench-<tag>/`; no CSVs or charts land in `docs/`. The working baselines agents inherit between releases live in
`docs/perf-baselines.md` (the release column is also added there as a cited row). A number taken mid-cycle is stale the
moment another perf round lands: run the gate on the final tree.

## Versioning & Releases

SemVer `MAJOR.MINOR.PATCH`, tagged `v1.0.0`. MAJOR breaks a public contract (HTTP API, flags, the pack format); MINOR
adds a model, a feature or a flag; PATCH fixes without adding.

**The version source is `build.zig.zon`'s `.version`**: a plain `zig build` stamps it into `sushi --version`, and
`-Dversion` (CI) must be SemVer or the build stops. `release.sh` dispatches only when the FIRST `## ` heading of
`CHANGELOG.md` names that same version and no GitHub release or tag carries it yet; the workflow's "Extract version"
step sources `release.sh` and applies the same checks (a pushed tag must be `v<zon>` or `v<zon>-pre-release.<n>`).
Nothing is computed from the date.

**Signing**: there is no Apple Developer ID, so the release binary ships ad-hoc signed and not notarized. The
workflow signs with a Developer ID and notarizes only when the `APPLE_*` repo secrets exist
(`tests/test_release_workflow_gates.sh`).

**Release**:
1. Set `build.zig.zon`'s `.version` to the next version and rename the top `## Unreleased` entry to
   `## v<version> — Headline` (check `gh release list --limit 1` first — never reuse an existing tag)
2. Dont commit or push
3. After the owner (or `./release.sh`) cuts it, the Release workflow leaves a DRAFT. Check its tarball against the
   `.sha256` and run its `sushi --version`, then publish it. Right after publishing, run `scripts/bump-tap.sh`: it
   triggers `beamivalice/homebrew-tap`'s bump with this machine's `gh` login (no repo-secret token) and fails unless
   `Formula/sushi.rb` names the new tag. The tap's hourly schedule is the fallback.
4. Once the GitHub release is published, bump mlx-serve's Sushi pin to the release commit
   ([mlx-serve-integration](../../../docs/mlx-serve-integration.md#handing-a-new-engine-to-mlx-serve)): make the
   commit reachable from the submodule url (`ddalcu/sushi`, branch `exl3-module`), then in the mlx-serve checkout
   `git -C lib/sushi fetch && git -C lib/sushi checkout v<version>`, commit `lib/sushi`, run mlx-serve's tests and open
   the PR from `beamivalice/mlx-serve`. Update the doc's "Recommended pin" to the release commit.

### CHANGELOG style

**One entry per shipped release. No new entries for unshipped work — fold it into the next pending entry.** Always run
`gh release list --limit 1` first; if the topmost CHANGELOG entry is newer than the latest GitHub release, that entry is
unshipped and any new bullets get merged into it. Unshipped work lives under `## Unreleased`; the version heading is
written in the release step. A model that is not public yet (MiMo-V2.6-Flash until v1.1) stays out of the entry.

**Contributor credit (every release):** list outside contributors' PRs since the last tag (`gh pr list --state merged
--search "merged:>=<last tag date>"`, plus PRs landed as a squash or cherry-pick whose commit subject carries `(#N)`).
The entry thanks each by @handle, naming what they shipped in user terms, in the bullet for that change or in one
closing "Thanks" bullet. A credit a shipped release missed goes into the next entry, marked as belated.

Tone: high-level executive bullets, marketing-style. The audience is users/integrators, not contributors reading the diff.

- Lead each bullet with **what changed for the user** (capability, speed, model support), not the implementation.
- Quantify where impressive — concrete tok/s percentages, model names, the workload it applies to.
- Avoid: file paths, function names, internal symbol renames, line-count diffs, "we discovered that…", PR/issue numbers.
- 4–7 bullets per release. If you need more, the release is too big and should ship sooner.

Template:

```markdown

## vMAJOR.MINOR.PATCH — Two-to-five-word headline

- **<User-visible thing>**: one or two sentences on the impact. Numbers if you have them.
- **<New model / API / behavior>**: what unlocks, when it kicks in, what stays the same.
- **<Speed or reliability win>**: workload + measured gain.
- **<Removed / deprecated thing, if any>**: why, and what users should do instead.

---
```

When in doubt, look at the existing entries — keep the same density and tone.
