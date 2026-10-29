#!/bin/bash
# sushi-test: tier=lib
# Sourced by tests/run.sh and by `boot=own` scripts. Contract: docs/testsuite.md, "Integration scripts".
# Call these directly, never inside $(...): the boot state lives in this shell. Written for bash 3.2.

SUSHI_ROOT="${SUSHI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SUSHI_BIN="${SUSHI_BIN:-$SUSHI_ROOT/zig-out/bin/sushi}"
. "$(dirname "${BASH_SOURCE[0]}")/profiles.sh"

# sushi_pack <model|alias|path>: the pack dir, or SKIP (3) when it is missing
sushi_pack() {
    local key="${1:?sushi_pack <model|alias>}" name var dir
    case "$key" in
        /*) dir="$key" ;;
        *)
            case "$key" in
                qwen4) name=Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw ;;
                glm5) name=GLM-5.3-Flash-Sushi-2.4bpw ;;
                dsv41) name=DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw ;;
                mimo2) name=MiMo-V2.6-Flash-Sushi-2.3bpw ;;
                qwen4-bf16) name=Qwen/Qwen3.8-Flash-Next ;;
                mimo2-src) name=MiMo-V2.6-Flash ;;
                *) name="$key" ;;
            esac
            var="SUSHI_TEST_PACK_$(echo "$key" | tr 'a-z-' 'A-Z_')"
            dir="${!var:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/$name}"
            ;;
    esac
    if [ ! -d "$dir" ]; then
        echo "SKIP: pack '$key' missing at $dir" >&2
        return 3
    fi
    echo "$dir"
}

# sushi_arch <pack>: the config.json model_type, empty when unreadable
sushi_arch() {
    python3 -I -c 'import json,sys
try: print(json.load(open(sys.argv[1] + "/config.json")).get("model_type", ""))
except Exception: print("")' "$1"
}

_sushi_free_port() {
    python3 -I -c 'import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

_sushi_size() { if [ -f "$1" ]; then wc -c < "$1" | tr -d ' '; else echo 0; fi; }

_sushi_tail_from() { tail -c +"$(($2 + 1))" "$1"; }

# sushi_boot <label> <pack> [flags...]: take the GPU lock and boot a private server. Fails (1) with the log
# tail when it dies or never becomes ready; SKIP (3) when the pack is missing. The label names the log,
# the END marker and the private HOME and cache root, so booting the same label twice is a restart.
sushi_boot() {
    local label="${1:?sushi_boot <label> <pack> [flags...]}" pack="${2:?sushi_boot <label> <pack> [flags...]}"
    shift 2
    [ -x "$SUSHI_BIN" ] || { echo "[boot] $SUSHI_BIN missing: zig build -Doptimize=ReleaseFast" >&2; return 1; }
    [ -d "$pack" ] || { echo "SKIP: pack $pack missing" >&2; return 3; }
    [ -z "${SUSHI_PID:-}" ] || { echo "[boot] $label: server '$SUSHI_LABEL' is still running" >&2; return 1; }

    : "${SUSHI_RUN_DIR:=$HOME/.sushi/runs/tests-$(date +%Y%m%d-%H%M%S)-$$}"
    local home="$SUSHI_RUN_DIR/$label.home" cache="$SUSHI_RUN_DIR/$label.kv-cache" port off limit t0 tp=""
    mkdir -p "$home/.sushi" "$cache"
    if [ -n "${SUSHI_BOOT_SETTINGS:-}" ]; then cp "$SUSHI_BOOT_SETTINGS" "$home/.sushi/model-settings.json"; fi
    port=$(_sushi_free_port) || return 1
    command -v taskpolicy > /dev/null && tp="taskpolicy -a"

    SUSHI_LABEL="$label"
    SUSHI_LOCK_OWNER="${GPU_LOCK_OWNER:-tests}-$label"
    "$SUSHI_ROOT/scripts/gpu-lock.sh" acquire "$SUSHI_LOCK_OWNER" || return 1

    SUSHI_LOG="$SUSHI_RUN_DIR/$label.log"
    off=$(_sushi_size "$SUSHI_LOG")
    HOME="$home" SUSHI_PREFIX_CACHE_DIR="$cache" $tp "$SUSHI_BIN" --serve --model "$pack" --host 127.0.0.1 \
        --port "$port" --log-file off "$@" >> "$SUSHI_LOG" 2>&1 &
    SUSHI_PID=$!
    SUSHI_PORT="$port"
    SUSHI_URL="http://127.0.0.1:$port"
    SUSHI_PACK="$pack"
    SUSHI_ARCH=$(sushi_arch "$pack")
    SUSHI_PROFILE="${SUSHI_PROFILE:-own}"
    export SUSHI_URL SUSHI_PORT SUSHI_PID SUSHI_LOG SUSHI_PACK SUSHI_ARCH SUSHI_PROFILE SUSHI_RUN_DIR
    _sushi_arm_trap

    limit=${SUSHI_BOOT_TIMEOUT_S:-$((300 + $(du -smL "$pack" | cut -f1) / 100))}
    t0=$(date +%s)
    while :; do
        if ! kill -0 "$SUSHI_PID" 2> /dev/null; then
            echo "[boot] $label: server died during boot; log tail:" >&2
            _sushi_tail_from "$SUSHI_LOG" "$off" | tail -20 >&2
            sushi_stop 1 > /dev/null 2>&1
            return 1
        fi
        if curl -sf "$SUSHI_URL/health" > /dev/null 2>&1 && _sushi_tail_from "$SUSHI_LOG" "$off" | grep -q 'Model ready'; then
            return 0
        fi
        if [ $(($(date +%s) - t0)) -ge "$limit" ]; then
            echo "[boot] $label: not ready after ${limit}s; log tail:" >&2
            _sushi_tail_from "$SUSHI_LOG" "$off" | tail -20 >&2
            sushi_stop 1 > /dev/null 2>&1
            return 1
        fi
        sleep 0.25
    done
}

# sushi_stop [status]: TERM the recorded PID, KILL after the wait, release the lock and write
# <run>/<label>.END. A server that was already gone is a crash: reported, and it returns 1.
sushi_stop() {
    local rc="${1:-0}" state=clean ws waited=0
    [ -n "${SUSHI_PID:-}" ] || return 0
    if kill -0 "$SUSHI_PID" 2> /dev/null; then
        # The braces keep the shell's own "Terminated" job notice off stderr.
        {
            kill -TERM "$SUSHI_PID"
            while kill -0 "$SUSHI_PID"; do
                if [ "$waited" -ge $((${SUSHI_STOP_WAIT_S:-60} * 5)) ]; then
                    kill -KILL "$SUSHI_PID"
                    state=killed
                    break
                fi
                sleep 0.2
                waited=$((waited + 1))
            done
            wait "$SUSHI_PID"
        } 2> /dev/null
    else
        { wait "$SUSHI_PID"; } 2> /dev/null
        ws=$?
        state="crash"
        echo "[boot] CRASH: server '$SUSHI_LABEL' (pid $SUSHI_PID) was already gone, exit status $ws; log: $SUSHI_LOG" >&2
    fi
    printf 'exit=%s server=%s\n' "$rc" "$state" > "$SUSHI_RUN_DIR/$SUSHI_LABEL.END"
    "$SUSHI_ROOT/scripts/gpu-lock.sh" release "$SUSHI_LOCK_OWNER" 2> /dev/null
    unset SUSHI_PID SUSHI_URL SUSHI_PORT SUSHI_PROFILE SUSHI_LOCK_OWNER
    [ "$state" != crash ]
}

# Chains the script's own EXIT trap; run.sh sets SUSHI_TRAP_ARMED and stops the server in its own.
_sushi_arm_trap() {
    [ -z "${SUSHI_TRAP_ARMED:-}" ] || return 0
    SUSHI_TRAP_ARMED=1
    local prev
    prev=$(trap -p EXIT)
    SUSHI_PREV_TRAP=""
    if [ -n "$prev" ]; then
        eval "set -- $prev"
        SUSHI_PREV_TRAP="$3"
    fi
    trap '_sushi_exit $?' EXIT
}

_sushi_exit() {
    local rc="$1"
    trap - EXIT
    sushi_stop "$rc" || { [ "$rc" -ne 0 ] || rc=1; }
    [ -z "$SUSHI_PREV_TRAP" ] || eval "$SUSHI_PREV_TRAP"
    exit "$rc"
}

# sushi_alive: the booted server still runs
sushi_alive() { [ -n "${SUSHI_PID:-}" ] && kill -0 "$SUSHI_PID" 2> /dev/null; }

# sushi_attach: top of every attach script. The nonce starts every prompt, so no script restores another's prefix.
sushi_attach() {
    if [ -z "${SUSHI_URL:-}" ] || [ -z "${SUSHI_LOG:-}" ]; then
        echo "$(basename "$0"): attach script: run it with tests/run.sh --only <file> --model <m>, or set SUSHI_URL and SUSHI_LOG" >&2
        exit 1
    fi
    SUSHI_LOG_OFFSET=$(_sushi_size "$SUSHI_LOG")
    SUSHI_NONCE="n$(date +%s)p$$r$RANDOM"
    export SUSHI_LOG_OFFSET SUSHI_NONCE
}

# sushi_log_since: the shared log from this script's attach point
sushi_log_since() { _sushi_tail_from "$SUSHI_LOG" "${SUSHI_LOG_OFFSET:-0}"; }

# _sushi_probe <path> <python expr over the JSON d>: prints 1 or 0
_sushi_probe() {
    curl -sf "$SUSHI_URL$1" | python3 -I -c 'import sys, json
try:
    d = json.load(sys.stdin)
    print(1 if eval(sys.argv[1]) else 0)
except Exception:
    print(0)' "$2"
}

# sushi_require <mtp|vision|video|batching|stq|thinking|spec>: exit 3 (SKIP) when the boot lacks it
sushi_require() {
    local cap="${1:?sushi_require <cap>}" ok=0 arch="${SUSHI_ARCH:-}"
    case "$cap" in
        mtp) ok=$(_sushi_probe /props 'd["settings"]["mtp"]["loaded"]') ;;
        vision) ok=$(_sushi_probe /props 'd["settings"]["vision"]["loaded"]') ;;
        video)
            case "$arch" in qwen4_exp | glm5_next) ok=$(_sushi_probe /props 'd["settings"]["vision"]["loaded"]') ;; esac
            ;;
        batching) ok=$(_sushi_probe /props 'd["batching"]["supported"]') ;;
        spec) ok=$(_sushi_probe /props 'd["settings"]["mtp"]["loaded"] or d["settings"]["drafter"] not in ("none", "")') ;;
        thinking) ok=$(_sushi_probe /v1/models 'any("reasoning" in m.get("capabilities", []) for m in d["data"])') ;;
        stq)
            ok=$(python3 -I -c 'import json, sys
try: print(1 if json.load(open(sys.argv[1] + "/config.json"))["expert_quant"]["format"] == "stq" else 0)
except Exception: print(0)' "${SUSHI_PACK:-}")
            ;;
        *) echo "sushi_require: unknown capability '$cap'" >&2; exit 1 ;;
    esac
    if [ "$ok" != 1 ]; then
        echo "SKIP $(basename "$0"): the boot lacks $cap${arch:+ ($arch)}"
        exit 3
    fi
}
