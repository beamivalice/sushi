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

echo "gpu-lock: $PASS passed, $FAIL failed"
[ $FAIL = 0 ]
