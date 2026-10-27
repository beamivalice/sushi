# Process: running a GPU job and recording a number

The step-by-step procedure behind CLAUDE.md's Team process rules: how to take the GPU lock, build and stamp the
binary under test, restore QoS, wait for a job without hanging, find the baseline to inherit, and record the result.
Every brief that measures on the GPU points here.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [perf-baselines](perf-baselines.md),
[quality-kld](quality-kld.md), the `/bench` skill.

## 1. Find the baseline first

- Look up the matching cell in [perf-baselines](perf-baselines.md) or [quality-kld](quality-kld.md) (and
  `benchmarks.md` for release columns). Same pack, same flags, same methodology.
- Found → do NOT rerun it. Run only the new arm and cite the recorded commit, settings and date beside the new number.
- Within one session, inherit the previous number. An old-binary baseline is rerun only in a clean new session (fresh
  box state) or when none exists for that exact setting; say which beside the number.
- Existing pack shards are the byte-identity baseline for a converter change at the same settings.

## 2. Build and stamp the binary under test

```sh
git diff --quiet HEAD || echo "tree is dirty: commit or stash your change first"
./.zig-toolchain/zig build -Doptimize=ReleaseFast
echo "$(git rev-parse --short HEAD) $(stat -f %Sm zig-out/bin/sushi)" > <run>.binary.txt
```

- Rebuild right before the run, inside the queue script if the run is queued. `zig build test` and cherry-picks do
  not refresh `zig-out/bin/sushi`.
- Reject any number whose binary stamp is older than the change it claims to measure.

## 3. Take the lock for exactly one run

```sh
scripts/gpu-lock.sh acquire <owner>
# one boot+probe, one KLD, one microbench, one pilot or one trace
scripts/gpu-lock.sh release <owner>
```

- Lock = a full model load (server boot, `kld capture|compare`, live test) or a quiet-box bench or timing whose
  absolute number is recorded. Conversions, pilots, builds and tests run in parallel without it when memory fits:
  a conversion peaks near 20 GB, so it may run beside a Qwen-size load (~50-65 GB) but never beside a MiMo-size one
  (~105 GB), and a MiMo load or a quiet bench waits until the parallel work ends. A timing taken beside parallel work
  is stated as contended.
- Acquire immediately before each run, release as soon as it ends: never across a queue or batch, never while
  analysing, editing, building or waiting. An A B B A re-acquires per arm. All agents on the box share one lock
  directory (`GPU_LOCK_DIR`, default `/tmp/sushi-gpu.lock.d`).
- Waiters are served first-come-first-served: `acquire` takes the next ticket in `${GPU_LOCK_DIR}.queue/` (a symlink
  whose text is `<pid> <owner>`) and only the lowest live ticket may take the lock, so a later arrival that polls
  faster cannot jump the queue. A ticket whose waiter process is gone is skipped and pruned. `scripts/gpu-lock.sh
  status` prints the holder (or `free`), then the queue in order.
- In a script: `trap "scripts/gpu-lock.sh release <owner>" EXIT` right after the acquire.
- A dead holder (its run is gone but `status` still names it) blocks every waiter: `acquire` exits once it holds the
  lock, so no live PID is recorded and no script can tell a dead holder from a slow one. A worker that suspects one
  reports it to the coordinator and keeps waiting. The coordinator checks that the holder's run is gone (its PID, END
  marker, log), then runs `scripts/gpu-lock.sh break <holder>`, which frees the lock only if it names the current
  holder, so it cannot break a newer one. The next ticket then takes the lock. Only the coordinator breaks a lock.

## 4. Restore QoS for agent-launched jobs

Processes spawned from an agent harness inherit background QoS (priority 4 vs 31) and run up to ~3x slower. Launch
timed jobs with `taskpolicy -a <cmd>` (or restore the running PID), and state the QoS used beside the number. A number
2-4x worse than a terminal run points at QoS before anything else.

The same holds for builds and tests: on the M5 Max an agent-launched process with inherited QoS stays on the 12
Performance cores and never reaches the 6 Super cores (CPUs 12-17); under `taskpolicy -a` it does. Run every CPU-heavy
agent command (`zig build`, `zig build test`, test scripts, conversions) under `taskpolicy -a`.

`taskpolicy` is SIP-protected, so `DYLD_*` in its own environment never reaches the job it launches: write
`taskpolicy -a env DYLD_LIBRARY_PATH=<stage> <bin>`, and prove the stage with `DYLD_PRINT_LIBRARIES=1` in the log.

## 4b. Cool the box before a bench

The M5 Max throttles hard: a 1M ladder read 1358 tok/s prefill at 2k right after hours of GPU work, and 1678 after a cooled
start (fans at max, 4 min idle). Even with fans at max the die reached 97 °C inside one minute of 4k load. So before any
bench:
- heavy GPU workload AND a die sensor over 90 °C: fans to max, then 3 min with nothing running, then start;
- otherwise: fans to max, wait 10 s, start.

Under this protocol one A arm then one B arm is enough. Run A B B A only when the expected difference is within a few
percent, where a thermal or drift step between two single arms would read as the effect.

Set the fans back to auto when the bench ends. The fan control and temperature readout are the box's own tooling (on this
box, the fan-control MCP: `max_fans`, `set_fan_auto`, `get_thermal_status`). A bench script waits on a marker file that
the driver touches once the fans are at max. Nothing else may compute during the idle and the bench (quiet box).

<a id="release-bench"></a>
## 4c. The release bench

`tests/bench.sh --version <ver> <pack> [-- <serve flags>]` is the perf gate: one model, one server boot, one run. It
takes the box rules itself (clean tree, fresh ReleaseFast build, QUIET marker, AC power, GPU lock, fans max and an idle
under 50 °C for at most 3 min, `taskpolicy -a`, a private `SUSHI_PREFIX_CACHE_DIR` under `~/.sushi/runs/bench-<tag>/`,
refusal while any `sushi`, `sushi-repack` or `sushi-grid` runs), boots the server by PID, runs `sushi bench` against it
and stops it. The raw `bench.json` (every request), the server log and `bench.log` stay in the run directory; the report
`docs/bench/v<ver>/<pack>.md` carries the commit, binary SHA-256 and mtime, flags and hardware.

The protocol, fixed for every model and release:
- **Two corpora, two arms.** The code arm is a seeded synthetic TypeScript codebase (`bench.Corpus`, seed fixed). The prose
  arm is Pride and Prejudice (`src/fixtures/prose_pride_and_prejudice.txt`, Project Gutenberg #1342, public domain, header
  and footer removed): speculative decoding accepts fewer tokens on prose than on code (1.95 against 2.9 a round on a 209k
  DSpark trace), so code decode alone flatters it. Each is tokenized through the server's `/tokenize`, sent as token ids
  on `/v1/completions`: greedy, `ignore_eos`, 256 decoded tokens on the code arm and 128 on the prose arm, the server's default speculative mode (the
  `[spec-stats] mode=` lines of each request name it in the report). One `sushi bench` run does the code arm, then the
  prose arm, each with the whole schedule below, on one server and one private prefix-cache directory: the two prompts
  share no prefix, so the prose arm's 4k request is cold. It is not expert-cache cold: the code arm has run before it.
- **One cold prefill.** A cold 4k request, then `corpus[:8192k]` for k = 1..16. Each request restores the previous
  prompt-end point and prefills only the new tokens: the report's window table is the continuous prefill curve, 0 to 128k.
  Windows hold about 8,223 new tokens, not 8,192: the prompt-end checkpoint trails the prompt end by 30 tokens (GLM 31 to
  34, a pool boundary), so every window re-forwards that tail. The tables name the actual restored and new counts.
- **Prefill rate** = new tokens over the server's `prompt_ms` minus the restore time. `prompt_ms` includes the SSD
  restore (it covers the whole prefix lookup), and each request logs `[prefix] reused N of M ... lookup and restore X ms`
  (`[dsv41-prefix]` on DeepSeek), which the bench subtracts. TTFT = request wall time minus `predicted_ms`.
- **Decode rungs 4k, 8k, 16k, 32k, 64k, 128k.** The ladder request decodes (sample 1, the first request, never
  discarded); samples 2 and 3 are the same prompt sent right after it, restoring at the prompt end (about 31 tokens to
  forward). Order rule, the one for every rung and release: ladder step, its two repeats, next ladder step, ascending.
  Repeating a rung straight away keeps its restore point from being thinned by the per-entry checkpoint cap (Qwen,
  GLM) that a 128k entry would apply to it later. Expert-cache warmth is whatever the preceding requests left.
- **The report** puts code and prose decode side by side per rung (with the prose/code ratio), one window table per
  arm, and a summary row set: the code rows keep the old shape and stay comparable across releases, and the prose rows
  are added beneath them as `<pack> (prose)`.
- **The GPU clock is read, and a throttled run stops.** `sushi bench` runs `sudo -n powermetrics --samplers gpu_power,thermal`
  (500 ms) beside the run and keeps, for each request, the median active MHz of the samples with at least 50% GPU
  residency and the worst thermal pressure level; both are columns beside the tok/s and the header gives the run's
  minimum and median. A request whose busy clock is at 1100 MHz or under aborts the bench with an error naming it
  (the top clock is 1620 MHz; a throttled run sat at 1025 to 1120). The pressure level is recorded only: a DeepSeek run
  reached `Heavy` 25 minutes in with the GPU still at 1539 MHz. The partial report is `bench-aborted.md` in the run
  directory, never `docs/bench`. `bench.sh` logs the pressure level after the cooldown. Without a passwordless `sudo -n powermetrics` both say the clock is not monitored and
  go on; a decode number from a run that was not monitored needs a `powermetrics` look beside it before it is recorded.
- **The prose arm starts cool.** Between the arms `sushi bench` sets the fans to max and idles until the hottest
  `fanctl status` die is under 50 C, at most 180 s (a fixed 180 s without `fanctl`); the wait is logged and named in the report's protocol line.
- **A request that did not restore** within 64 tokens of the point it should, one without a restore log line, or a
  short decode is listed under Warnings and makes the run exit non-zero: it is not a measurement. A decode the
  server's loop stop ended (`finish_details.type` = `repetition_loop`) is valid: it counts and is listed under Notes.

Cross-engine comparison is not this bench: start the other engine, run llmprobe by hand
(`npx -y llmprobe@latest <host:port> -m <id> --bench-only`) and name the engine beside every number.

## 5. Wait without hanging

Never wait on `pgrep -f "<string>"`: every agent shell's own `zsh -c "<command>"` contains the string and matches
forever. Wait on an END marker the job writes after it exits (`echo "END <job> rc=$?" >> <phases file>`), on a PID
captured at launch (`kill -0 $pid`), or on `pgrep -x <binary>` / `ps -axo pid,comm` filtered on the executable name.

## 6. Check what actually ran

- Launch flags outrank `model-settings.json`; confirm from the load lines (`[kv-cache] … (source)`,
  `[mtp] on|off (source)`) that the arm ran the settings you meant. A per-model `mtp: true` once turned an MTP-off
  control into an MTP run.
- An A/B arm is proven by ENGAGEMENT lines in its own log, never by its launch env.
- MTP off is verified by ~1.01 tokens per step.

## 7. Record

Write the number into the matching doc (and `benchmarks.md` for a release column) in the same landing, with:
commit, binary stamp, pack name, flags, QoS, lock owner, date, and the baseline it is compared with. The raw-file
path goes to the gitignored `docs/private/measurement-raw.md`, keyed by doc and section; a committed doc names no
local path.

## Build environment notes

- A Zig bump changes `ZIG_VERSION` and the sha256 table in `scripts/fetch-zig.sh` together, and lands only on a tree
  that builds and passes the full suite.
- A git worktree has neither `.zig-toolchain/` nor the built `lib/mlx/`: symlink both from the main checkout.
- `lib/mlx-src` and `lib/mlxc-src` are the only submodules; they are needed only to rebuild MLX.
