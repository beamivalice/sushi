---
name: hunt-fix
description: Run a sushi bug hunt and fix cycle as the coordinator - cut surfaces, launch fresh hunter and fixer agents (hunter Opus 5.5 high, fixer Sonnet 5.5 high, a user-reported issue's fixer Opus 5.4 high), verify findings, route each to a fixer, review and land the fixes on main. Use when asked to hunt bugs, run hunters or fixers, fix a GitHub issue with an agent, or run a hunt/fix cycle before a release.
---

# Hunt and fix cycle

The process, what counts as a finding and how a finding is closed are in `docs/process-bug-hunt.md`; read it first.
This skill is the coordinator's procedure: who to launch, what to brief, how to land.

## Agents

| Role | Agent type | Model | Use for |
|---|---|---|---|
| Hunter | `hunter` | Opus 5.5, high effort | attacking a surface with scratch tests; reports findings, never fixes |
| Fixer | `fixer` | Sonnet 5.5, high effort | findings from a hunt, a test failure, a review |
| Issue fixer | `issue-fixer` | Opus 5.4, high effort | a defect a user reported (a GitHub issue, a forwarded report) |

The definitions are in `.claude/agents/`. Launch with the Agent tool, `subagent_type` set to the role, run in the
background. The owner's word overrides the default model for one launch.

**A fresh agent for every assignment.** A new surface, a new batch of findings or a new issue gets a new agent with a
self-contained brief. Messages to a running agent are only about its own assignment: a review correction, an answer
to its question, a sibling site of the same finding. A finding that arrives later goes to the next fresh fixer. An agent
retires when its assignment is committed or reported.

Keep the number of agents at once within the owner's cap; ask when none is set.

## Coordinator loop

1. **Head and known list.** Pick the commit under hunt, and list every defect fixed or open on it.
2. **Surfaces.** Cut by how code fails (state and lifecycle; memory and long-lived state; what is computed; server
   and API), one per hunter, no overlap.
3. **Launch hunters**, each with its own git worktree at the head (`.zig-toolchain` and `lib/mlx*` linked from the
   main checkout) and its own temp directory. Brief: the template below.
4. **On each finding** (a hunter sends HIGH ones as they reproduce): reproduce it or read it to its cause, group it with
   its class, add one board row for it, and route it to a fresh fixer.
5. **Review each fixer commit** before landing: the red test fails for the stated reason, the bar would catch the bug,
   the class's sibling sites are covered, nothing is weakened (see Review).
6. **Land** (below), then reply on the issue and close it when its fix is on main.
7. **Retire** the wave's agents; remove their worktrees once their commits land.

## Hunter brief (template)

- The surface, the head sha, the worktree and temp directory, the known list.
- Read `docs/process-bug-hunt.md`, `CLAUDE.md`, `docs/testsuite.md` and the surface's docs.
- Fixtures: `export SUSHI_MODELS_DIR=<models dir>; source tests/lib/fixture_envs.sh`. Tests by name only:
  `taskpolicy -a .zig-toolchain/zig build test -Doptimize=ReleaseFast -Dtest-filter="<name>" --summary all`.
- Scratch tests only, on the tiny fixtures; never commit, fix product code, load a real pack, boot a server or take
  the GPU lock. To hand scratch tests to fixers, push them to a scratch branch only if the coordinator names one.
- Send each HIGH finding (crash, corruption, wrong output, hang, a bill that under-covers storage) as soon as it
  reproduces: test file and name, command, trimmed red output, mechanism at file:line, bug class with sibling sites.
- A timebox. The final report in `docs/process-bug-hunt.md`'s format.

## Fixer brief (template)

- The findings (each with its red scratch test and mechanism), the base sha, the worktree, branch `fix-<id>`.
- TDD: the failing test first, run it, quote the red output, then the fix. A bar a real bug would still pass is no bar.
- Fix the class, not the instance: cover every sibling site named, or say why one differs. Prefer a structure that
  makes the class impossible (a required argument, one owner of the state, one parser) over a guard at each site.
- Explicit behaviour only: no silent fallback, default or behaviour change the owner did not ask for. A refusal names
  its reason.
- One commit per finding, by pathspec, in the repo's message style, with the owner's git identity and no
  Co-Authored-By or AI trailer. Never push, merge, rebase main, open a PR or comment on an issue: the coordinator lands.
- Comments state intent the code cannot; no history. The doc for the area gets the rule the fix teaches (one line);
  a user-visible fix gets one `## Unreleased` CHANGELOG sentence.
- Run the tests the diff can break, by name, with counts. Never the full suite, a real pack, a server or the GPU lock.
- After each commit, message the coordinator: sha, red output, the fix in two lines, siblings covered, runs with counts.

An issue fixer's brief adds the issue's report verbatim, and asks it to measure before changing any bill, limit or
threshold, and to report the measured table first.

## Review

Refuse, and send back to the same fixer with the exact bar, any of:
- a widened tolerance, a shortened test, a removed assertion, or a two-sided band where the bar should be exact;
- a test that passes on the old code (red-first evidence missing or wrong);
- a fix at one site while a named sibling keeps the bug;
- a fallback or default nobody asked for;
- a bar with no named cause (a margin must say what it covers).

A single noisy measurement is not a result: ask what moves it before accepting a margin.

## Landing

From a dedicated landing worktree, for each fixer branch in order:
1. Fetch; rebase on `origin/main` (CHANGELOG conflicts merge both sides).
2. Check no commit carries a Co-Authored-By or session trailer.
3. Run the fixer's named filters, plus every filter its files can reach. A change to the shared Cooler engine runs the
   whole `cooler contract` filter with both the Qwen and DeepSeek fixtures sourced.
4. `taskpolicy -a .zig-toolchain/zig build -Doptimize=ReleaseFast` and `python3 tests/suite_guards.py`.
5. Push only when all are green and `origin/main` is an ancestor of HEAD; then fetch and **verify** that
   `origin/main` equals the pushed sha before reporting it landed. If main moved, rebase and rerun the filters whose
   files overlap the commits that landed meanwhile.

## Box rules

- One real-pack process at a time, under `scripts/gpu-lock.sh`; agents never take the lock.
- Never wrap `tests/run.sh --tier live` in the GPU lock (it takes the lock itself).
- Stop processes by PID. Never touch a server you did not start.
- Before quoting a rate from a long GPU job, compute the expected rate and compare the first batch; more than 2x off
  stops the run until it is explained. A long job runs at nice 0, not as a child of an agent shell.
