# Bug hunt

A bug hunt is how a body of new code is closed: hunters attack it with tests they write, and a synthesizer decides
what the findings say about the design. It ends when every finding is fixed, or refused with a test that proves it is
not a defect. The engine is ours to maintain: what a hunt leaves open comes back as debt.

Rules: [CLAUDE.md](../CLAUDE.md#team-process). Tests and tiers: [testsuite](testsuite.md).

## When

- Before a phase of work is called done (a port, a new kernel family, a new serving path), and before a release
  that carries one.
- After a live failure that shows a class: one hunt on the component it came from.

## The round

1. **Head.** The code under hunt is committed on one branch; every hunter starts from that commit.
2. **Surfaces.** The coordinator cuts the code into surfaces that do not overlap, one per hunter, by how it fails:
   state and lifecycle (orders of events, failed and repeated loads, models side by side), memory and long-lived
   state (pool, caps, bills against storage, caches, leaks), what is computed (kernels against a reference at edge
   shapes, determinism, rollback, sampling).
3. **Hunters.** Three reviewers at the strongest model and effort the owner allows, launched together. Each gets its
   own git worktree at the head (toolchain and `lib/mlx` linked from the main checkout) and its own temp directory.
4. **Known list.** The brief names every defect already fixed or open, so a hunter spends its runs on new ones and
   on deeper siblings.
5. **Report**, then **synthesis**, then the fixes, then the worktrees go.

## What a hunter may do

- Write and run **scratch tests**: at the bottom of a test file the bucket already runs, on the tiny fixtures, with
  the bucket's filter. A scratch test may print to stderr (never stdout); it is not held to the quiet-test rule.
- Leave a one-line local change that proves a diagnosis, marked `// SCRATCH PROOF`.
- Use the fault injector to reach error paths: `mlx.fault.arm(k)` fails the k-th checked MLX call.

A hunter never commits, pushes, stashes or fixes product code, never loads a real pack, boots a server, binds a port
or takes the GPU lock, and checks `QUIET` before every build. It works only in its own worktree.

## What counts

A wrong result, a crash, a hang, a leak, storage the bill does not cover, a broken engine invariant, a silent
fallback, state left dirty after an error or a cancel, the same request giving two answers. A red scratch test is a
finding only when its expectation is the contract: the report cites the doc line, comment or sibling test that says
so. A suspicion that did not reproduce is listed as such in one line.

## The report

Per finding, most severe first: severity and whether it was reproduced; where and by what mechanism; the scratch test,
its command and its trimmed output; a suggested fix; and the **bug class**: the property of the component that let it
happen, with every sibling site that has the same shape, misbehaving today or not. The report ends with **component
health**: for each component, whether it keeps producing the same kind of bug, and the redesign that would make the
class impossible (the type, the owner of the state, the shape of the API). Then the scratch tests worth keeping.

## Synthesis

The coordinator is the synthesizer and does it alone, with every report in hand.

1. **Verify.** Each finding is reproduced or read to its cause before it is believed; a hunter's test can be wrong.
2. **Group by class,** across hunters and across earlier rounds. One class seen by two hunters is one finding.
3. **Verdict per component: refactor or normal fix.** The long-term codebase decides, not the size of today's diff.
   - **Refactor** when the class has produced a bug before, when sibling sites exist, or when the structure makes
     the bug possible (state without an owner, a mode passed by a side channel, a result nobody must check). The
     refactor removes the possibility; fixing the instances one by one is not an answer to a class.
   - **Normal fix** when the defect is one of a kind in a component that held under attack.
   - A refactor that would change another architecture's behaviour, a flag or a public contract goes to the owner
     with its reason; everything else the coordinator decides and starts.
4. **Tell the owner** the table (component, what keeps going wrong, evidence, verdict), the healthy components, and
   anything in an earlier report that the hunt showed to be wrong.
5. **Fix in order of blast radius:** crashes and corruption, wrong results, leaks and bills, the rest.

## Closing a finding

- A finding is closed by a fix with its regression test, or by a test that proves the behaviour is the contract.
  Never by argument, and never by moving it to a later phase.
- A scratch test worth keeping becomes a regression test: quiet, asserting, placed by [testsuite](testsuite.md).
- A class that a rule can name becomes a lint rule (`scripts/lint/`), so the next instance fails the build.
- The matching doc gets the rule the class taught, one line per rule.
- The hunt's worktrees and temp files are removed when its fixes land.
