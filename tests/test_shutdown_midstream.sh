#!/bin/bash
# sushi-test: tier=live boot=own pack=default boots=2
# Regression: SIGTERM (graceful shutdown) while streaming requests are in flight must NOT crash the server.
# serve() cancels in-flight slots and drains connection threads before scheduler.deinit tears the slot queues down;
# without that a detached connection thread inside Scheduler.complete() hit a use-after-free.
#
# Each iteration boots a server, fires three long streaming generations, sends SIGTERM at a varied phase (prefill,
# then mid-decode) and asserts the process exits within the drain bound and not by SIGSEGV or SIGABRT.
# SHUTDOWN_ITERS overrides the count.
set -u
. "$(dirname "$0")/lib/boot.sh"
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; NC=$'\033[0m'
ITERS="${SHUTDOWN_ITERS:-2}"
PACK=$(sushi_pack qwen4) || exit $?

PASS=0; FAIL=0
for i in $(seq 1 "$ITERS"); do
    sushi_boot "shutdown-$i" "$PACK" || exit $?
    SVPID=$SUSHI_PID
    # More concurrent connection threads, a better chance one is inside complete() when deinit runs.
    CURLPIDS=()
    for _ in 1 2 3; do
        curl -s "$SUSHI_URL/v1/chat/completions" -H 'Content-Type: application/json' \
            -d '{"model":"sushi","stream":true,"max_tokens":400,"temperature":0.7,"messages":[{"role":"user","content":"Write a long detailed essay about the history of computing, at least 300 words."}]}' \
            > /dev/null 2>&1 &
        CURLPIDS+=($!)
    done
    case $((i % 4)) in 0) d=0.4 ;; 1) d=0.8 ;; 2) d=1.4 ;; 3) d=2.2 ;; esac
    sleep "$d"
    kill -TERM "$SVPID" 2> /dev/null

    exited=0
    for _ in $(seq 1 70); do # 35s, the drain bound plus margin
        kill -0 "$SVPID" 2> /dev/null || { exited=1; break; }
        sleep 0.5
    done
    for cp in "${CURLPIDS[@]}"; do kill -9 "$cp" 2> /dev/null; done
    if [ "$exited" != 1 ]; then
        echo -e "  ${RED}FAIL${NC} iter $i (delay ${d}s): server HUNG after SIGTERM"
        FAIL=$((FAIL + 1))
        sushi_stop 1
        continue
    fi
    wait "$SVPID" 2> /dev/null
    rc=$?
    # 139 = 128+11 SIGSEGV, 134 = SIGABRT; a graceful exit is 0.
    SUSHI_EXPECT_GONE=1 sushi_stop 0
    if [ "$rc" = 139 ] || [ "$rc" = 134 ]; then
        echo -e "  ${RED}FAIL${NC} iter $i (delay ${d}s): crashed rc=$rc"
        FAIL=$((FAIL + 1))
    else
        echo -e "  ${GREEN}PASS${NC} iter $i (delay ${d}s): clean exit rc=$rc"
        PASS=$((PASS + 1))
    fi
done

echo "=== shutdown-midstream: $PASS passed, $FAIL failed ($ITERS iters) ==="
[ "$FAIL" -eq 0 ]
