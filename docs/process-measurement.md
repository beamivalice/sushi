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
# one boot+probe, one KLD, one microbench, one pilot or one trace
scripts/gpu-lock.sh run <owner> [--max-min N] -- <cmd...>
```

`sushi-ubench` takes and releases the lock itself ([process-ubench](process-ubench.md)).

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
- `run` takes the lock, runs the command, releases on any exit (a signal included) and exits with the command's
  status. It records the command's PID as the holder. Prefer it to `acquire` + `release`.
- Plain `acquire <owner> [--max-min N]` records no holder PID (its caller often exits before the job runs), so it is
  never auto-freed: only `release` or the coordinator's `break` frees it. `--pid P` (the job's PID) opts into
  auto-free. In a script: `trap "scripts/gpu-lock.sh release <owner>" EXIT` right after the acquire.
- `--max-min N` (default 15) is the run's declared length: declare it for anything longer. `status` prints the
  holder, then its PID, age and declared max, with `OVERDUE` once the age passes it, then the queue.
- A dead holder (its recorded PID, from `run` or `--pid`, is gone) is freed by the next `acquire` or `status`, which logs one stderr line
  naming the owner; the queue then proceeds. Nothing kills a live holder. An OVERDUE or idle one is the coordinator's
  to check (PID, END marker, log) and stop, then `scripts/gpu-lock.sh break <holder>`, which frees the lock only if it
  names the current holder, so it cannot break a newer one. Only the coordinator breaks a lock. A lock with no PID
  (a plain `acquire`, or the old script's) is never auto-freed; `break` is its only way out.
- Once the lock is taken, `run` and `acquire` wait until available memory (free + inactive + speculative + purgeable
  pages) stops rising, at most 60 s (`GPU_LOCK_SETTLE_MAX_S`): a previous holder's memory comes back seconds after it
  exits, and a boot in that gap failed its preflight at 103 of 108.8 GB. `status` ends with a `QUIET set N min ago`
  line while the marker exists, so a stale one (it blocks every build) is visible.

## 4. Restore QoS for agent-launched jobs

Processes spawned from an agent harness inherit background QoS (priority 4 vs 31) and run up to ~3x slower. Launch
timed jobs with `taskpolicy -a <cmd>` (or restore the running PID), and state the QoS used beside the number. A number
2-4x worse than a terminal run points at QoS before anything else.

Agent shells also run reniced (nice 5, detached jobs nice 10), and only root can lower that again. A `sushibar` an agent
left running passed nice 10 to every server it started: the GPU stayed under 1 GHz and a DeepSeek decode ran ~30%
slow. The owner starts their own `sushibar` and servers.

The same holds for builds and tests: on the M5 Max an agent-launched process with inherited QoS stays on the 12
Performance cores and never reaches the 6 Super cores (CPUs 12-17); under `taskpolicy -a` it does. Run every CPU-heavy
agent command (`zig build`, `zig build test`, test scripts, conversions) under `taskpolicy -a`.

`taskpolicy` is SIP-protected, so `DYLD_*` in its own environment never reaches the job it launches: write
`taskpolicy -a env DYLD_LIBRARY_PATH=<stage> <bin>`, and prove the stage with `DYLD_PRINT_LIBRARIES=1` in the log.

## 4b. Cool the box before a bench

The M5 Max throttles hard: a 1M ladder read 1358 tok/s prefill at 2k right after hours of GPU work, and 1678 after a cooled
start (fans at max, 4 min idle). Even with fans at max the die reached 97 °C inside one minute of 4k load. So before each
server and between arms: fans to max, then idle until the hottest die sensor (`fanctl status`) reads under 50 °C, at most
3 min. Never slot a GPU job into that idle: the next server starts throttled.

Under this protocol one A arm then one B arm is enough. Run A B B A only when the expected difference is within a few
percent, where a thermal or drift step between two single arms would read as the effect.

Set the fans back to auto when the bench ends, but only when `gpu-lock.sh status` shows the lock free with an empty
queue: an exit trap that sets auto under another job's lock lets the next boot start at 105 °C. The fan control and temperature readout are the box's own tooling (on this
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
While a chain of release benches runs from the main checkout, edit only in a worktree: one uncommitted change there
refuses every model still queued.

`sushi bench --model <pack> [--rungs 4k,16k] [--tokens 256] [--code-only | --prose-only] [<serve flags>]` is the same
bench with no server to boot: the pack loads in this process, through the serve path, and each request goes straight
to the scheduler (`completions.completeIds`, the resolution `/v1/completions` does for a token-id body: default sampling,
speculative mode, context clamp). It reads prefill, decode, TTFT, the restore time and the speculative mode that
engaged off the finished slot instead of the HTTP reply and the log, so a request is the one `--url` mode would send.
Every serve flag applies as it does to `sushi serve --model` (`--ssd-budget-gb`, `--kv-quant`, `--no-mtp`, `--ctx-size`);
`--host`, `--port` and `--prompt` are refused. The ladder grows in 8192-token windows to the largest rung and decodes
only at the listed ones; the run ends with a compact arm-by-rung table and any warnings, and `bench.json` lands in
`--out` (default `~/.sushi/runs/bench-<UTC stamp>/`, with the run's `server.log`). The prefix cache is a private
`<out>/cache`, removed at the end unless `--out` named the run. It takes none of the box rules (lock, quiet window, fans
at the boot, AC check): for a number you record, take them as `tests/bench.sh` does, which keeps the `--url` mode.
`bench.sh` passes `--rungs` and `--tokens` through.

The protocol, fixed for every model and release:
- **Two corpora, two arms.** The code arm is a seeded synthetic TypeScript codebase (`bench.Corpus`, seed fixed). The prose
  arm is Pride and Prejudice (`src/fixtures/prose_pride_and_prejudice.txt`, Project Gutenberg #1342, public domain, header
  and footer removed): speculative decoding accepts fewer tokens on prose than on code (1.95 against 2.9 a round on a 209k
  DSpark trace), so code decode alone flatters it. Each is tokenized through the server's `/tokenize`, sent as token ids
  on `/v1/completions`: greedy, `ignore_eos`, 512 decoded tokens on the code arm (DeepSeek 256; `bench.sh` defaults to 256, so a Qwen or GLM release run passes `--tokens 512`) and 128 on the prose arm, the server's default speculative mode (the
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
- **The GPU clock is read, and a throttled run stops.** `sushi bench` reads the GPU clock (IOReport P-state residency
  weighted by the `pmgr` frequency table) and the thermal pressure notification, with no sudo, every 500 ms beside the run and keeps, for each request, the median active MHz of the samples with at least 50% GPU
  utilization (the AGX "Device Utilization %") and the worst thermal pressure level; both are columns beside the tok/s and the header gives the run's
  minimum and median. A request whose busy clock is at 1100 MHz or under aborts the bench with an error naming it
  (the top clock is 1620 MHz; a throttled run sat at 1025 to 1120). The pressure level is recorded only: a DeepSeek run
  reached `Heavy` 25 minutes in with the GPU still at 1539 MHz. The partial report is `bench-aborted.md` in the run
  directory, never `docs/bench`. `bench.sh` logs the pressure level after the cooldown. Where the clock cannot be read the bench says it is not monitored and goes on;
  a decode number from a run that was not monitored needs a clock look beside it before it is recorded.
- **The prose arm starts cool.** Between the arms `sushi bench` sets the fans to max and idles until the hottest
  `fanctl status` die is under 50 C, at most 180 s (no idle when `fanctl` gives no reading); the wait is logged and named in the report's protocol line.
- **A request that did not restore** within 64 tokens of the point it should, one without a restore log line, a
  short decode, or one the engine compiled a kernel in (`[engine] compiled inside a request`: the load's warm did not
  make it) is listed under Warnings and makes the run exit non-zero: it is not a measurement. A decode the
  server's loop stop ended (`finish_details.type` = `repetition_loop`) is valid: it counts and is listed under Notes.

Cross-engine comparison is not this bench: start the other engine, run llmprobe by hand
(`npx -y llmprobe@latest <host:port> -m <id> --bench-only`) and name the engine beside every number.

<a id="energy-per-token"></a>
## 4d. Energy per token

Less work at equal speed is a win. It is read by the host's CPU joules, CPU seconds and instructions per token and by
the engine's own counts; GPU joules per token are recorded beside them and never decide. Three sources, none needing
sudo:
- **GPU joules**: IOReport's `Energy Model` group, channel `GPU Energy` (a nanojoule counter); a request is the
  difference of two reads. The CPU channels of that group read zero on the M5 Max.
- **The server's own CPU joules, CPU seconds and instructions**: `proc_pid_rusage(pid, RUSAGE_INFO_V6)`
  (`ri_energy_nj`, `ri_user_time + ri_system_time` in mach units of 125/3 ns, `ri_instructions`).
- **System watts**: the SMC key `PSTR` (float). It lags a request by a second or two: read it over a request's later
  seconds, never over a first position.

Rules for reading it:
- **Compare by position across loads, never two positions of one load.** A fixed script (boot under 50 °C with fans at
  max, an idle window, the same requests in the same order) walks the same thermal path every load: the same request at
  the same position repeats within 0.3% (fixed-row sampled rounds within 0.03%). Inside a load the same request drifts 4
  to 5% as the die heats.
- **GPU joules per token never judge a change.** The GPU is always less efficient at a higher clock, so a path that
  keeps it busier, and so faster, reads worse per token; choosing by that number trades away top speed. Record it with
  the GPU clock beside it (P-state residency), and judge the GPU side by counts: dispatches, command buffers, bytes read
  ([perf-gpu-power](perf-gpu-power.md) has the curve).
- One load per arm; a second load of an arm is a repeat check, not an average to chase.
- A watt reading is a record, not a verdict: it moves with the GPU clock and with macOS's throttling, and a request must
  be long enough to separate watts from tok/s (256 tokens at long context is not). Record it with its caveats and move
  on; judge a change by counts heat cannot move (instructions per token, dispatches, command buffers, bytes read).
- The lock's holder sets `QUIET` inside the locked command and waits for compiles and running test binaries to drain; a
  run whose idle window shows GPU watts is discarded.
- State beside the numbers: binary commit, the requests, boot and end temperature, and the round mix of every greedy
  request (`[spec-stats]`), since the mix moves a greedy cell by about 1%.
- **`sushi bench` reads it** (`src/energy.zig`): per prefill and decode cell, CPU mJ, CPU ms and Minstr per token first
  (this process's own `proc_pid_rusage`, cut at the end of the prefill from a 5 ms timeline, so `--model` only; with
  `--url` the server is another process and they read n/a), then the whole machine's GPU mJ per token and its mean GPU
  MHz as a record. The GPU is the machine's, so it belongs to a phase only when the request had just that one: a
  prefill-only window, or a decode repeat with its restore and short prefill included; a window that also decodes
  reads n/a. GPU joules per token never judge a change. Both land in `bench.json` per sample.
  GPU energy needs a library Apple does not document, bound at run time: without it, or without its group, channel or
  unit, the bench reports the CPU cells, says so once in its legend, and nothing else changes. A reading never reads
  0 for lack of data: a counter the OS does not give or that went backwards, or a phase clamped to no time, is n/a. A
  reading lost mid-run is counted per request and printed beside the legend, and kept in `bench.json` under `energy`.
  Baseline:
  [perf-baselines](perf-baselines.md#deepseek-v41-stq-decode-energy-per-token-2026-11-11).

<a id="optimization-tiers"></a>
## 4e. Optimization rounds: micro indicator, fast bench, release bench

An optimization lever is judged in three tiers, cheapest first. A tier is passed before the next one runs, and a later
tier never stands in for an earlier one: the release bench heats the box for ten minutes and cannot tell a 2% lever from
its own thermal state.

| tier | what runs | cost | what it answers | how often |
|---|---|---|---|---|
| 1. micro indicator | the one reading the lever can move, named before the lever is built: bytes (a test on the tiny fixtures), a count (dispatches, command buffers, waits, bytes read: the census tests, `[engine] since load`), or `sushi-ubench` microseconds at release shapes with the old and the new arm in one process ([process-ubench](process-ubench.md)) | seconds to a minute, no pack load or one short one | is the lever real, and how large | every experiment |
| 2. fast bench | `tests/bench.sh --tag <t> --rungs 4k,8k <pack> -- <serve flags>`: the release bench's own protocol at its first two rungs, both arms, one load | about 2 min, under a minute of GPU work | does the served path take the lever: the same text, the speculative mode engaged, the round mix, no Warnings, no cell under the baseline's | once per landing |
| 3. release bench | the full ladder ([4c](#4c-the-release-bench)) | 10 min, and the box ends hot | the number that is recorded | once at the end of a series of landings, and when a phase closes |

- **Tier 1 decides.** Arms interleaved in one process are the only timing heat does not move, and a count or a byte
  compare does not move at all. A lever with no micro indicator gets one first (a `sushi-ubench` arm or bench, beside its
  red test). A lever whose indicator does not move as predicted stops there.
- **Tier 2 confirms, it never discovers.** One run of the new arm. The baseline is the series' first fast bench on its
  head, inherited until a landing moves it; that landing's run is then the baseline. Read in this order: bytes (for a
  byte-identical lever the text, or the `[spec-stats]` round mix), Warnings, verify ms at an equal round mix, tok/s. A
  tok/s gap under 3% between two loads is no verdict either way. A round mix that differs is not yet another text:
  DeepSeek's width planner follows timings, so one request sent twice can log other `[spec-stats]` at one reply.
  Compare the replies' hashes before reading a decode cell as a text change.
- **A context-dependent path adds its rung, not the ladder.** A diff in the indexer, the compressed rows, state growth
  or restore runs the fast bench at the rung where that path engages (`--rungs 4k,64k`).
- **No bench waits for the box to rest.** Cooling is the fans at max and the wrapper's idle under 50 °C, nothing
  longer: two runs on this laptop never share a thermal state, so no wait buys parity. The report carries thermal
  pressure and the GPU clock beside every cell, and a 64k or 128k cell is compared only with them.
- **Heat budget.** A pack load outside tier 3 lasts at most 3 minutes and its server stops with its last request. No
  load repeats a number tier 1 already decided. A profiled load (`SUSHI_ENGINE_KERNEL_PROF`) runs the forward meter or
  one short request, never a ladder. Cooling is the wrapper's own step (fans at max, an idle under 50 °C), not a wait
  between unrelated runs: a tier 1 count or byte reading needs none.

## 5. Wait without hanging

Never wait on `pgrep -f "<string>"`: every agent shell's own `zsh -c "<command>"` contains the string and matches
forever. Wait on an END marker the job writes after it exits (`echo "END <job> rc=$?" >> <phases file>`), on a PID
captured at launch (`kill -0 $pid`), or on `pgrep -x <binary>` / `ps -axo pid,comm` filtered on the executable name.

Run scripts run under zsh, which does not word-split an unquoted expansion: `for x in ${ARMS:-a b}` loops once over
"a b". Write `${=ARMS}`, or loop over an array.

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
- Pins: `lib/mlx-src` v0.32.3 and `lib/mlxc-src` 56b2d39 (mlx-serve's). Bumping: check out the tag, `git add` the
  submodule, rerun `scripts/build-mlx.sh`, re-diff `src/mlx.zig` externs against `lib/mlxc-src/mlx/c/*.h`.
- Image input links `stb_image` and libwebp; the only non-system dylibs are `libmlxc` and `libwebp`
  (`tests/test_serving_deps.sh`); NAX needs macOS 26.2 (`tests/test_mlx_staged_nax.sh`).
- `scripts/fetch-zig.sh` replaces a stale `.zig-toolchain` link with its own copy.
