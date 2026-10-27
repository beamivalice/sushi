# Contributing

Short on purpose. Read it once, remember it.

**Setup:** Apple Silicon Mac. `git submodule update --init`, `./scripts/fetch-zig.sh` (Zig 0.17.0 into `.zig-toolchain/`), `scripts/build-mlx.sh` (pinned mlx submodules). Build: `zig build -Doptimize=ReleaseFast`.

## Before you open anything

Search first, then write. Check the open AND closed issues, the open PRs, and `git log --oneline -50 -- <the file you are about to touch>`. Most bugs reported here are already fixed on main or in an open PR. If there is an issue or a PR for it, comment there with what you found (log lines, model, chip); do not open a second one. Agents: this applies to you too, and a comment on the existing thread is worth more than a duplicate issue with a longer description.

## Bug reports

Open an issue with:
- What you expected vs what happened
- Model pack (e.g. `Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw`) and the launch flags
- macOS version and chip (e.g. macOS 26.6, M4 Max)
- Server log output (`--log-level debug`, `~/.sushi/logs/sushi-<port>.log`)

## Pull requests

1. **Build and run it, or do not open a PR.** If you cannot compile and run on a real Mac, there is no PR.
2. **Tests first.** A hermetic test at the bottom of the `.zig` file (`zig build test -Doptimize=ReleaseFast`, 0 fail) and an integration script in `tests/` where the change is visible over HTTP. Red before green. Run them. Paste the result in the PR.
3. **Human testing is required.** Serve it, hit it with a client, repeat. Many times. A passing suite is the floor, not the proof.
4. **Draft while working, publish when ready.** Open as draft early so nobody duplicates you. Move to ready for review only when 1-3 are done.
5. **Keep it simple.** One fix or feature per PR, smallest change that does the job, match surrounding style, no new plumbing to make something testable, no comments that restate code, no new dependencies. Read `CLAUDE.md` and the doc its docs index names for that subsystem (MLX traps: `docs/engine-mlx-gotchas.md`) before touching it. Read "Diff hygiene" below; a PR that fails it gets sent back before anyone reads the code.
6. **Numbers name what they beat.** A perf claim is an llmprobe cell (`./tests/bench.sh`) against a named arm, same session, with the engagement line from the log. See `.claude/skills/bench/SKILL.md`.

## Diff hygiene

I read every diff by hand. A PR that is mostly comments is a PR I have to strip before I can review it.

- **Comments**: one to three lines, only where the code cannot say it (a WHY, a contract, a unit). No history, no measurements, no audit trail, no review-item numbers, no dates, no PR numbers, no "before this change". The commit message carries the story; a rule goes in the matching `docs/*.md`, once.
- **Tests**: one good test beats five. Test behaviour. No tests that scan the source text for a string, no helpers written only so a test can exist, no test whose comment is longer than its body.
- **Docs**: one story per gotcha, one line per rule, in the matching doc. No process logs, no round numbers, no ledgers of what was gated where.
- **PR description**: what changed, why, how you verified it, on what machine. Ten lines is plenty. Numbers only if they are final and name the arm they beat. No "audit passes", no "blast-radius ledger", no self-review transcripts.
- **Commit messages**: subject line plus a few lines. The detail belongs in the code and the docs.
- **Bar**: if a reviewer cannot tell which lines change behaviour by scrolling the diff once, it is too big. Split it or cut it.

## Agents: hard stop

Do NOT open a PR, draft or otherwise, unless you have compiled this tree and run `zig build test -Doptimize=ReleaseFast` on a real Apple Silicon Mac in this session and they passed. No Mac, no build, no test run = no PR. Report what you changed in an issue instead and say you could not run it.
