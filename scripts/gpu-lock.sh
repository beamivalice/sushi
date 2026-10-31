#!/bin/bash
# gpu-lock.sh — the box-exclusive lock (CLAUDE.md, Team process).
#
# Take it for a full model load (server boot, `kld capture|compare`, a live test) or a quiet-box bench or
# timing run whose absolute number is recorded. Conversions, pilots, builds and tests run without it, in
# parallel, as long as memory fits (never beside a MiMo-size load).
# Acquire right before EACH run and release as soon as it ends; never hold the
# lock across a queue, or while analysing, editing, building or waiting.
#
#   gpu-lock.sh run <owner> [--max-min N] -- <cmd...>
#                                 take the lock, run <cmd>, release on any exit (signals included), exit with its
#                                 status; the recorded holder PID is the command's
#   gpu-lock.sh acquire <owner> [--pid P] [--max-min N]
#                                 take a ticket, wait for it to come up, then hold the lock as <owner>. No holder PID
#                                 is recorded unless --pid is given, so a plain acquire is never auto-freed (a caller
#                                 may exit before its job runs): only `release` or the coordinator's `break` frees it.
#   gpu-lock.sh release <owner>   release a lock <owner> holds (refused otherwise)
#   gpu-lock.sh break <holder>    COORDINATOR ONLY: free a live overdue/idle holder, or one with no recorded PID
#                                 (refused unless it names the holder)
#   gpu-lock.sh status            holder, its PID and age, OVERDUE past its declared max, then the live queue in order
#
# A holder whose recorded PID (`run`, or `acquire --pid`) is gone is freed by the next `acquire` or `status`. A live holder past its
# --max-min (default 15) is only flagged OVERDUE; stopping it is the coordinator's call.
# Waiters are served first-come-first-served: only the lowest live ticket may take
# the lock. A ticket whose waiter process is gone is skipped and pruned.
# GPU_LOCK_DIR overrides the lock directory; every agent on the box must share it.
set -u

L="${GPU_LOCK_DIR:-/tmp/sushi-gpu.lock.d}"
Q="$L.queue"
POLL="${GPU_LOCK_POLL_S:-5}"

# A ticket is a symlink $Q/<n> whose text is "<pid> <owner>": created atomically, contents
# included. A served ticket reads "done" and stays until a higher ticket prunes it, so the
# highest number is never deleted and ticket numbers only grow.
# Listed with a glob, not ls: a waiter whose ls lists nothing would number itself 1 and jump the queue.
tickets() {
  local f
  for f in "$Q"/*; do
    f=${f##*/}
    case "$f" in '' | *[!0-9]*) ;; *) echo "$f" ;; esac
  done | sort -n
}
live() {
  local v pid
  v=$(readlink "$Q/$1" 2>/dev/null) && [ "$v" != done ] || return 1
  pid=${v%% *}
  kill -0 "$pid" 2>/dev/null || ps -p "$pid" >/dev/null 2>&1
}

alive() { kill -0 "$1" 2>/dev/null || ps -p "$1" >/dev/null 2>&1; }

# The lock dir holds `owner` (old format: "<owner> <time>") and, additively, `start` (epoch), `max_min` and `pid`.
# `pid` is written last, so a lock without one is old-format or still being filled and is never auto-freed.
# Renaming `pid` away is the claim: of several reapers only one wins.
reap() {
  local p o
  [ -d "$L" ] && p=$(cat "$L/pid" 2>/dev/null) && [ -n "$p" ] && ! alive "$p" || return 1
  o=$(cat "$L/owner" 2>/dev/null)
  mv "$L/pid" "$L/pid.reap.$$" 2>/dev/null || return 1
  if [ "$(cat "$L/pid.reap.$$")" != "$p" ]; then mv "$L/pid.reap.$$" "$L/pid" 2>/dev/null; return 1; fi
  rm -rf "$L"
  echo "gpu-lock: freed the lock of ${o%% *}: holder pid $p is gone" >&2
}

# take <owner> <pid|empty> <max-min>
take() {
  local owner=$1 n last t ahead
  mkdir -p "$Q"
  last=$(tickets | tail -1)
  n=$((${last:-0} + 1))
  until ln -s "$$ $owner" "$Q/$n" 2>/dev/null; do n=$((n + 1)); done
  while :; do
    reap
    ahead=
    for t in $(tickets); do
      [ "$t" -lt "$n" ] || break
      if live "$t"; then ahead=$t; break; fi
      rm -f "$Q/$t"
    done
    [ -z "$ahead" ] && mkdir "$L" 2>/dev/null && break
    sleep "$POLL"
  done
  echo "$owner $(date '+%Y-%m-%d %H:%M:%S')" > "$L/owner"
  date +%s > "$L/start"
  echo "$3" > "$L/max_min"
  [ -z "$2" ] || echo "$2" > "$L/pid"
  ln -s done "$Q/.done.$$" && mv -f "$Q/.done.$$" "$Q/$n"
}

# release_as <owner> [break]: fails unless <owner> holds the lock
release_as() {
  local holder
  holder=$(cut -d' ' -f1 "$L/owner" 2>/dev/null)
  if [ "$holder" != "$1" ]; then
    echo "gpu-lock: $1 does not hold the lock (holder: ${holder:-none})" >&2
    return 1
  fi
  [ "${2:-}" = break ] && echo "gpu-lock: broke the lock held by $(cat "$L/owner")" >&2
  rm -rf "$L"
}

usage() {
  echo "usage: gpu-lock.sh run <owner> [--max-min N] -- <cmd...> | acquire <owner> [--pid P] [--max-min N] | release|break <owner> | status" >&2
  exit 2
}

# parse <acquire|run> args...: sets $pid and $max, and REST (the command, for run).
parse() {
  local mode=$1
  shift
  while [ $# -gt 0 ]; do
    case "$1" in
      --pid) [ "$mode" = acquire ] || usage; pid=${2:-}; shift 2 || usage ;;
      --max-min) max=${2:-}; shift 2 || usage ;;
      --) [ "$mode" = run ] || usage; shift; break ;;
      *) usage ;;
    esac
  done
  case "$pid$max" in *[!0-9]*) usage ;; esac
  [ "$max" -ge 1 ] || usage
  REST=("$@")
  [ "$mode" = acquire ] || [ ${#REST[@]} -gt 0 ] || usage
}

case "${1:-}" in
  acquire)
    owner="${2:-}"; [ -n "$owner" ] || usage
    shift 2
    pid= max=15
    parse acquire "$@"
    take "$owner" "$pid" "$max"
    ;;
  run)
    owner="${2:-}"; [ -n "$owner" ] || usage
    shift 2
    pid= max=15
    parse run "$@"
    take "$owner" "$$" "$max"
    child=
    trap 'release_as "$owner"' EXIT
    trap 'kill -TERM "$child" 2>/dev/null' INT TERM HUP
    "${REST[@]}" <&0 &
    child=$!
    echo "$child" > "$L/pid"
    while :; do
      wait "$child"
      rc=$?
      kill -0 "$child" 2>/dev/null || break
    done
    exit "$rc"
    ;;
  release | break)
    owner="${2:-}"; [ -n "$owner" ] || usage
    release_as "$owner" "$1" || exit 1
    ;;
  status)
    reap
    if [ -d "$L" ]; then
      cat "$L/owner" 2>/dev/null || echo "unknown (lock dir without owner)"
      p=$(cat "$L/pid" 2>/dev/null)
      if [ -n "$p" ]; then
        max=$(cat "$L/max_min" 2>/dev/null || echo 15)
        now=$(date +%s)
        mins=$(((now - $(cat "$L/start" 2>/dev/null || echo "$now")) / 60))
        flag=
        [ "$mins" -gt "$max" ] && flag=" OVERDUE"
        echo "  pid $p alive, held $mins min of $max max$flag"
      else
        echo "  no pid recorded"
      fi
    else
      echo free
    fi
    i=0
    for t in $(tickets); do
      live "$t" || continue
      v=$(readlink "$Q/$t")
      i=$((i + 1))
      echo "  $i. ${v#* } (pid ${v%% *}, ticket $t)"
    done
    ;;
  *) usage ;;
esac
