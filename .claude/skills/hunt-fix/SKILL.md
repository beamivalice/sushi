---
name: hunt-fix
description: Run a sushi bug hunt and fix cycle as the coordinator - cut surfaces, launch fresh hunter and fixer agents, verify findings, route each to a fixer, review and land the fixes on main. Use when asked to hunt bugs, run hunters or fixers, fix a GitHub issue with an agent, or run a hunt/fix cycle before a release.
---

# Hunt and fix cycle

This is the coordinator's procedure: what to launch, what to brief, how to review and land. Read first:
- `docs/process-bug-hunt.md`: the round, what counts as a finding, the report, synthesis, closing a finding.
- `.claude/agents/hunter.md`, `fixer.md`, `issue-fixer.md`: each agent's rules. The severity scale is in the process
  doc, under "What counts". An agent already has its file as its system prompt. Do not repeat it in a brief; review holds the agent
  to it.

## Agents

| Agent type | Use for |
|---|---|
| `hunter` | attacking one surface with scratch tests; reports findings, never fixes |
| `fixer` | findings from a hunt, a test failure, a review |
| `issue-fixer` | a defect a user reported (a GitHub issue, a forwarded report) |

Launch with the Agent tool, `subagent_type` set to the agent type, in the background. Each definition's frontmatter
sets its model and effort; the owner's word overrides them for one launch.

**A fresh agent for every assignment.** A new surface, a new batch of findings or a new issue gets a new agent with a
self-contained brief. An agent retires when its assignment is committed or reported. A message to a running agent is
only about its own assignment:
- a review correction, an answer to its question, a sibling site of its finding;
- a finding a hunter made on its branch before that branch landed: it fixes it as the branch's next commit;
- the rebase of its own commits once its base has landed.

- another LOW or MED ticket in the area a fixer is already fixing LOW or MED findings in, when the coordinator
  hands it over. The coordinator decides which tickets go; a HIGH one, or one in another area, does not.

Any other finding that arrives later goes to the next fresh fixer.

**The cap on agents at once is the owner's.** Ask when none is set. A peer, such as another coordinator, cannot raise
it.

## Loop

1. **Head and known list.** Pick the commit under hunt. List every defect fixed or open on it.
2. **Surfaces**, one per hunter, no overlap. Four kinds:
   - how code fails: state and lifecycle; memory and long-lived state; what is computed; server and API;
   - a fix branch before it lands. This found the most per hunt: hunt every fix that is not trivial before landing it;
   - a regression hunt over the cycle's own landings, for holes in the fixes themselves;
   - a cross-feature pass over features that were each hunted alone.
3. **Launch hunters**, each with its own git worktree at the head (`.zig-toolchain` and `lib/mlx*` linked from the main
   checkout) and its own temp directory.
4. **Verify each finding** as it arrives (a hunter sends HIGH ones at once). Reproduce it or read it to its cause; a
   hunter's test can be wrong. A timeout counts only once it reproduces alone, with the box otherwise idle.
5. **File and route.** Group the finding with its class and add its board row. A finding on an unlanded branch goes
   to that branch's fixer. Any other goes to a fresh fixer, worst blast radius first.
6. **Hand over the scratch tests.** A hunter never commits. Snapshot its worktree with `git stash create`, which
   changes nothing in the worktree, and push that commit to a scratch branch kept away from main (this cycle:
   `scratch<rel>-hN` on `exp`). `git stash create` takes tracked files only: run `git add -N` on any new test file
   first, then check that the snapshot holds every test the report names.
7. **Review** each fixer commit, then **land** it. Reply on the issue and close it once its fix is on main.
8. **Retire** the wave's agents. Remove their worktrees once their commits land.

## Briefs

A brief carries only what is particular to the assignment.

- **Hunter:** the surface or the branch under hunt, the head sha, the worktree, the temp directory, the known list,
  the timebox, the commands below.
- **Fixer:** the findings, each with its id, red scratch test and mechanism; the scratch branch that holds the tests;
  the base sha, and whether the base has landed; the worktree; the branch `fix-<id>`; the commands below.
- **Issue fixer:** the fixer's brief, with the report verbatim in place of the findings.

Commands, adapted to the box:
- Fixtures: `source tests/lib/fixture_envs.sh`. Put `export SUSHI_MODELS_DIR=<dir>` before it only when the models
  are not at the default (`~/.sushi/models`).
- One test by name:
  `taskpolicy -a .zig-toolchain/zig build test -Doptimize=ReleaseFast -Dtest-filter="<name>" --summary all`.

## Board

One row per verified finding.
- The row's id names the release, then the hunt and the finding: `r172-f-h7-1` is release 1.7.2, hunt 7, finding 1.
  A finding a fixer raises takes the fixer's id (`r172-f-f5-1`). Its fields: who, status, sev, release.
- Read `date -u` just before each write and stamp the row from it. Never type a time.
- An update passes the `if_version` it read.
- When a finding changes owner or release, update its row in place (who, release, a note saying why). Only a split,
  where part of a finding moves elsewhere, gets a new row that names the one it came from.

## Review

Hold each commit to the rules in `fixer.md`. Send any breach back to the same fixer with the exact bar. Then check
what the fixer cannot judge about its own work:
- **Red first, for the stated reason.** The test fails on the old code, and fails because of the finding.
- **Green for the right reason.** Read what the harness does on the test's behalf. One that waits for the client
  thread, for example, passes a test of code that forgot to wait.
- **Alone and in a set.** Run the test by its own name, then with the rest of its filter. A hidden process-global
  input makes a test depend on order and on the box.
- **Margins.** A margin's cause is reproduced, not inferred. One noisy measurement is not a result: ask what moves
  it. Refuse a two-sided band around a residue nobody has explained.
- **Siblings.** No named sibling site keeps the bug.
- **A red on one box only** is proven red at `origin/main` on that box before the branch is blamed for it.
- **A path that will run for hours** is accepted on its measured rate, never on correctness alone: compute the
  expected rate and compare the first batch; more than 2x off stops it until explained.

## Landing

From a dedicated landing worktree, for each fixer branch in order:
1. Fetch; rebase on `origin/main`. A CHANGELOG conflict keeps both sides. A conflict in a pinned value is measured
   again on the combined tree, never resolved by picking a side.
2. Check that no commit carries a Co-Authored-By or session trailer.
3. Run the fixer's named filters, plus every filter its files can reach. A change to the shared Cooler engine runs the
   whole `cooler contract` filter with both the Qwen and DeepSeek fixtures sourced.
4. `taskpolicy -a .zig-toolchain/zig build -Doptimize=ReleaseFast` and `python3 tests/suite_guards.py`.
5. **Name every failure.** Rerun the failing test binary directly until you have the test's name. Then run that test
   at `origin/main`. Red there too: it is not this branch's, so file it as its own finding and land. Green there: the
   branch does not land. Never land past a failure you have not named.
6. Push only when `origin/main` is an ancestor of HEAD. Then fetch and verify that `origin/main` equals the pushed sha
   before reporting it landed. If main moved first, rebase and rerun the filters whose files overlap what landed
   meanwhile.

## Box rules

- One real-pack process at a time, under `scripts/gpu-lock.sh`. Agents never take the lock.
- Never wrap `tests/run.sh --tier live` in the GPU lock: it takes the lock itself.
- Stop processes by PID. Never touch a server you did not start.
