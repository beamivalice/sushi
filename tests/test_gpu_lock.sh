#!/bin/bash
# sushi-test: tier=hermetic
# test_gpu_lock.sh — the one-heavy-GPU-job lock (scripts/gpu-lock.sh).
#
# Checks, on a private lock dir:
#   1. status reports "free" when nobody holds it
#   2. acquire takes it and status names the owner
#   3. a second acquire waits while the lock is held
#   4. release by another owner (queued or not) is refused and keeps the lock
#   5. release by the holder frees it, and the waiter then gets it
#   6. waiters are served first-come-first-served, even when a later one polls faster
#   7. a waiter that died leaves the queue: status drops it and the next waiter is served
#   8. status lists the holder, then the queue in ticket order
#   9. break (coordinator only, for a dead holder) frees the lock only when it names the holder
#  10. a waiter whose environment breaks `ls` still takes its ticket behind the queue
#  11. a plain acquire records no PID and is never auto-freed; a holder whose recorded PID (acquire --pid) died is
#      freed by the next acquire and by status; a lock with no PID file stays until break clears it
#  12. run releases on success, failure and SIGTERM, passes the exit code and stdin, records the command's PID
#  13. status flags OVERDUE past --max-min, never stops the holder; bad --max-min / --pid usage exits 2
#  14. waiters behind a dead holder never hold the lock at the same time
#
# No GPU, no model, seconds.
#
# Usage: ./tests/test_gpu_lock.sh

set -u
cd "$(dirname "$0")/.." || exit 1

LOCK=scripts/gpu-lock.sh
PASS=0
FAIL=0
check() { if [ "$1" = 0 ]; then PASS=$((PASS + 1)); echo "PASS $2"; else FAIL=$((FAIL + 1)); echo "FAIL $2"; fi; }

[ -x "$LOCK" ] || { echo "FAIL $LOCK missing or not executable"; exit 1; }

TMP=$(mktemp -d)
BG=
trap 'for p in $BG; do kill $p 2>/dev/null; done; rm -rf "$TMP"' EXIT
export GPU_LOCK_DIR="$TMP/gpu.lock.d"
export GPU_LOCK_POLL_S=1

queued() { $LOCK status | sed -nE 's/^ +[0-9]+\. ([^ ]+).*/\1/p' | tr '\n' ' '; }
# Wait (max 5 s) until <owner> is in the queue, so tickets are taken in a known order.
wait_queued() {
  local i
  for i in $(seq 50); do case " $(queued)" in *" $1 "*) return 0 ;; esac; sleep 0.1; done
  return 1
}
# Wait (max <s> s) for a background pid to exit; returns its status, or 124 on timeout.
wait_exit() {
  local i
  for i in $(seq $(($2 * 10))); do kill -0 "$1" 2>/dev/null || { wait "$1"; return $?; }; sleep 0.1; done
  return 124
}

[ "$($LOCK status)" = free ]; check $? "status is free before any acquire"

$LOCK acquire alpha; check $? "acquire succeeds on a free lock"
$LOCK status | grep -q '^alpha '; check $? "status names the holder"

$LOCK acquire beta & WAITER=$!; BG="$BG $WAITER"
sleep 2
kill -0 $WAITER 2>/dev/null; check $? "a second acquire waits while the lock is held"

$LOCK release beta 2>/dev/null; [ $? != 0 ]; check $? "release by a queued non-holder is refused"
$LOCK release gamma 2>/dev/null; [ $? != 0 ]; check $? "release by a stranger is refused"
$LOCK status | grep -q '^alpha '; check $? "a refused release keeps the holder"
[ "$(queued)" = "beta " ]; check $? "a refused release keeps the queue"

$LOCK release alpha; check $? "release by the holder succeeds"
wait_exit $WAITER 5; check $? "the waiter acquires once the holder releases"
$LOCK status | grep -q '^beta '; check $? "status names the new holder"
$LOCK release beta
[ "$($LOCK status)" = free ]; check $? "status is free after the last release"

# FIFO: w1 arrives first but polls slowest; a poll race would hand the lock to w3.
: > "$TMP/order"
$LOCK acquire alpha
for w in w1:1 w2:0.5 w3:0.1; do
  name=${w%%:*}
  ( GPU_LOCK_POLL_S=${w#*:} $LOCK acquire "$name" && echo "$name" >> "$TMP/order" && sleep 0.3 \
      && $LOCK release "$name" ) &
  BG="$BG $!"
  wait_queued "$name"; check $? "$name takes a ticket"
done
$LOCK status | head -1 | grep -q '^alpha '; check $? "status shows the holder first"
[ "$(queued)" = "w1 w2 w3 " ]; check $? "status lists the queue in ticket order (got: $(queued))"
$LOCK release alpha
for i in $(seq 100); do [ "$(wc -l < "$TMP/order")" -ge 3 ] && break; sleep 0.1; done
[ "$(tr '\n' ' ' < "$TMP/order")" = "w1 w2 w3 " ]; check $? "waiters are served in arrival order (got: $(tr '\n' ' ' < "$TMP/order"))"
for i in $(seq 20); do [ "$($LOCK status)" = free ] && break; sleep 0.1; done
[ "$($LOCK status)" = free ]; check $? "status is free once the queue drains"

# A dead waiter must not block the queue.
$LOCK acquire alpha
GPU_LOCK_POLL_S=0.1 $LOCK acquire ghost & GHOST=$!; BG="$BG $GHOST"
wait_queued ghost; check $? "ghost takes a ticket"
GPU_LOCK_POLL_S=0.1 $LOCK acquire next & NEXT=$!; BG="$BG $NEXT"
wait_queued next; check $? "next queues behind ghost"
kill -9 $GHOST; wait $GHOST 2>/dev/null
[ "$(queued)" = "next " ]; check $? "status drops a dead waiter (got: $(queued))"
$LOCK release alpha
wait_exit $NEXT 5; check $? "the waiter behind a dead ticket is served"
$LOCK status | grep -q '^next '; check $? "status names the waiter served past the dead ticket"
$LOCK release next

# A dead holder is cleared by the coordinator naming it; the queue then proceeds.
$LOCK acquire dead
GPU_LOCK_POLL_S=0.1 $LOCK acquire after & AFTER=$!; BG="$BG $AFTER"
wait_queued after; check $? "a waiter queues behind the dead holder"
$LOCK break after 2>/dev/null; [ $? != 0 ]; check $? "break naming a non-holder is refused"
$LOCK status | grep -q '^dead '; check $? "a refused break keeps the holder"
$LOCK break dead; check $? "break naming the holder succeeds"
wait_exit $AFTER 5; check $? "the waiter behind a broken lock is served"
$LOCK status | grep -q '^after '; check $? "status names the waiter served after the break"
$LOCK release after

# A waiter whose `ls` lists nothing (an odd PATH or environment) still takes a ticket behind the queue.
mkdir -p "$TMP/nols"; printf '#!/bin/sh\nexit 0\n' > "$TMP/nols/ls"; chmod +x "$TMP/nols/ls"
$LOCK acquire front
GPU_LOCK_POLL_S=0.1 $LOCK acquire first & FIRST=$!; BG="$BG $FIRST"
wait_queued first; check $? "first queues behind the holder"
PATH="$TMP/nols:$PATH" GPU_LOCK_POLL_S=0.1 $LOCK acquire blind & BLIND=$!; BG="$BG $BLIND"
wait_queued blind; check $? "a waiter with a blind ls still queues"
[ "$(queued)" = "first blind " ]; check $? "the blind waiter queues behind first (got: $(queued))"
$LOCK release front
wait_exit $FIRST 5; check $? "first is served before the blind waiter"
$LOCK release first
wait_exit $BLIND 5; check $? "the blind waiter is served next"
$LOCK release blind

# Make a lock dir by hand: <owner> <pid|-> [age seconds] (owner file in the old "<owner> <time>" format).
forge() {
  mkdir "$GPU_LOCK_DIR" || return 1
  echo "$1 2026-01-01 00:00:00" > "$GPU_LOCK_DIR/owner"
  [ "$2" = - ] && return 0
  echo $(($(date +%s) - ${3:-0})) > "$GPU_LOCK_DIR/start"
  echo 15 > "$GPU_LOCK_DIR/max_min"
  echo "$2" > "$GPU_LOCK_DIR/pid"
}
dead_pid() { sleep 0 & local p=$!; wait $p; echo $p; }

# A plain acquire records no PID: its caller may exit before the job runs, so it is never auto-freed.
( $LOCK acquire plain; sleep 60 ) & HOLDER=$!; BG="$BG $HOLDER"
for i in $(seq 50); do [ "$($LOCK status | head -1 | cut -d' ' -f1)" = plain ] && break; sleep 0.1; done
$LOCK status | grep -qx '  no pid recorded'; check $? "plain acquire records no pid"
pkill -P $HOLDER sleep; kill -9 $HOLDER; wait $HOLDER 2>/dev/null
$LOCK status 2> /dev/null | grep -q '^plain '; check $? "a plain acquire is not auto-freed when its caller dies"
$LOCK break plain 2> /dev/null; check $? "break frees a plain acquire"

# acquire --pid records the job's PID and is freed when it dies.
sleep 60 & JOB=$!; BG="$BG $JOB"
$LOCK acquire pidded --pid $JOB
$LOCK status | grep -q "^  pid $JOB alive, held 0 min of 15 max\$"; check $? "acquire --pid records the given PID"
kill $JOB; wait $JOB 2>/dev/null
$LOCK status 2> "$TMP/err" | grep -qx free; check $? "status frees a lock whose --pid holder died"
grep -q 'freed the lock of pidded' "$TMP/err"; check $? "the auto-free logs the owner to stderr"

forge deadlock "$(dead_pid)"
$LOCK acquire after-dead; check $? "acquire frees a dead holder and takes the lock"
$LOCK status | grep -q '^after-dead '; check $? "status names the waiter that replaced the dead holder"
$LOCK release after-dead

forge oldfmt -
$LOCK status | head -2 | tr '\n' '|' | grep -q '^oldfmt .*|  no pid recorded|$'; check $? "status shows an old-format lock without a PID"
GPU_LOCK_POLL_S=0.1 $LOCK acquire behind-old & OLD=$!; BG="$BG $OLD"
sleep 1
kill -0 $OLD 2>/dev/null; check $? "a lock without a PID file is not auto-freed"
$LOCK status | grep -q '^oldfmt '; check $? "the old-format holder is kept"
$LOCK break oldfmt; check $? "break clears an old-format lock"
wait_exit $OLD 5; check $? "the waiter behind the broken old-format lock is served"
$LOCK release behind-old

forge live-break $$
$LOCK break live-break; check $? "break still frees a live holder that names itself"
[ "$($LOCK status)" = free ]; check $? "status is free after break"

# run
$LOCK run ok -- true; check $? "run passes exit status 0"
$LOCK run bad -- sh -c 'exit 7'; [ $? = 7 ]; check $? "run passes the command's exit code"
[ "$($LOCK status)" = free ]; check $? "run released after success and failure"
[ "$(echo hello | $LOCK run piped -- cat)" = hello ]; check $? "run passes stdin to the command"
$LOCK run nocmd -- 2>/dev/null; [ $? = 2 ]; check $? "run without a command is a usage error"
$LOCK run x --max-min 0 -- true 2>/dev/null; [ $? = 2 ]; check $? "--max-min 0 is a usage error"
$LOCK acquire x --max-min abc 2>/dev/null; [ $? = 2 ]; check $? "a non-numeric --max-min is a usage error"
$LOCK run x --pid 1 -- true 2>/dev/null; [ $? = 2 ]; check $? "run refuses --pid"

$LOCK run term -- sleep 60 & RUN=$!; BG="$BG $RUN"
for i in $(seq 50); do [ "$($LOCK status | head -1 | cut -d' ' -f1)" = term ] && break; sleep 0.1; done
for i in $(seq 50); do CMD=$(pgrep -P $RUN); [ -n "$CMD" ] && break; sleep 0.1; done
for i in $(seq 50); do $LOCK status | grep -q "^  pid $CMD alive" && break; sleep 0.1; done
$LOCK status | grep -q "^  pid $CMD alive"; check $? "run records the command's PID as the holder"
kill -TERM $RUN
wait_exit $RUN 5; [ $? = 143 ]; check $? "run exits 143 on SIGTERM"
[ "$($LOCK status)" = free ]; check $? "run released after SIGTERM"
kill -0 "$CMD" 2>/dev/null; [ $? != 0 ]; check $? "SIGTERM reached the command"

$LOCK run second-in-line -- true & BG="$BG $!"
$LOCK run first-in-line -- sleep 1 & BG="$BG $!"
wait; [ "$($LOCK status)" = free ]; check $? "run waiters drain the queue"

# OVERDUE
$LOCK acquire slow --max-min 1 --pid $$
$LOCK status | grep -q '^  pid .* of 1 max$'; check $? "status shows the declared max and no OVERDUE when fresh"
echo $(($(date +%s) - 3600)) > "$GPU_LOCK_DIR/start"
$LOCK status | grep -q 'held 60 min of 1 max OVERDUE'; check $? "status flags OVERDUE past --max-min"
$LOCK status | grep -q '^slow '; check $? "an OVERDUE live holder is not freed"
$LOCK release slow

# Waiters behind a dead holder never overlap.
forge ghosts "$(dead_pid)"
: > "$TMP/overlap"
for w in c1 c2 c3 c4; do
  ( GPU_LOCK_POLL_S=0.05 $LOCK acquire $w && { mkdir "$TMP/in" 2>/dev/null || echo $w >> "$TMP/overlap"; sleep 0.2; rmdir "$TMP/in" 2>/dev/null; $LOCK release $w; } ) &
  BG="$BG $!"
done
wait
[ ! -s "$TMP/overlap" ]; check $? "concurrent waiters behind a dead holder never overlap (got: $(cat "$TMP/overlap"))"
[ "$($LOCK status)" = free ]; check $? "status is free after the concurrent waiters"

echo "gpu-lock: $PASS passed, $FAIL failed"
[ $FAIL = 0 ]
