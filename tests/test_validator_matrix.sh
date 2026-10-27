#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# test_validator_matrix.sh — API-compliance + agentic matrix across both
# served architectures (qwen4_exp, mimo_v2).
#
# Per model, two layers:
#   1. llmprobe (~/projects/agents/llmprobe) in auto
#      mode — compliance suites for /v1/responses, /v1/chat/completions and
#      /v1/messages against a server this script boots and tears down.
#   2. The pi 2-turn agentic html case (multi-turn tool calls: create
#      mlx.html, then add JS) via tests/pi_integration_run.sh, which manages
#      its own server lifecycle and appends audit_format leak markers.
#
# One model loaded at a time; missing weights skip cleanly. Both models are
# too large to coexist — this script refuses to start while any sushi runs,
# and stops only the servers it started.
#
# Usage:
#   ./tests/test_validator_matrix.sh                      # full matrix
#   VALIDATOR_MODELS=qwen4_exp ./tests/...                # csv filter
#   SKIP_PI=1 ./tests/test_validator_matrix.sh            # llmprobe only
#   SKIP_PROBE=1 ./tests/test_validator_matrix.sh         # pi only
#   BINARY=...  PORT=...  LLMPROBE_DIR=...                # overrides
#
# Output: per-model logs in tests/validator-results/, summary table on
# stdout + tests/validator-results/summary.tsv. Exit 1 if anything failed.

set -uo pipefail
cd "$(dirname "$0")/.."

REPO="$(pwd)"
BINARY="${BINARY:-$REPO/zig-out/bin/sushi}"
PORT="${PORT:-11298}"
LLMPROBE_DIR="${LLMPROBE_DIR:-$HOME/projects/agents/llmprobe}"
LLMPROBE_MJS="$LLMPROBE_DIR/bin/dist/llmprobe.mjs"
RESULTS="$REPO/tests/validator-results"
SUMMARY="$RESULTS/summary.tsv"
mkdir -p "$RESULTS"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

# logical|display|path|pi_case|probe_timeout_s|extra server flags
# pi_case must exist in pi_integration_run.sh's html matrix; empty = no pi layer.
MODELS=(
    "qwen4_exp|Qwen3.8 Flash-Next (qwen4_exp)|${QWEN4_EXP_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw}|html-qwen4|240|"
    "mimo_v2|MiMo-V2.6-Flash EXL3 (mimo_v2)|${MIMO_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/MiMo-V2.6-Flash-Sushi-2.3bpw}|html-mimo|240|--no-vision"
)

if [[ -n "${VALIDATOR_MODELS:-}" ]]; then
    IFS=',' read -r -a WANTED <<< "$VALIDATOR_MODELS"
    FILTERED=()
    for entry in "${MODELS[@]}"; do
        name="${entry%%|*}"
        for w in "${WANTED[@]}"; do
            [[ "$name" == "$w" ]] && FILTERED+=("$entry")
        done
    done
    MODELS=("${FILTERED[@]:-}")
fi

if [[ ! -x "$BINARY" ]]; then
    echo "[fatal] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi
if [[ -z "${SKIP_PROBE:-}" && ! -f "$LLMPROBE_MJS" ]]; then
    echo "[fatal] llmprobe bundle missing: $LLMPROBE_MJS (cd $LLMPROBE_DIR && npm run build:cli)"
    exit 1
fi

SERVER_PID=""
stop_server() {
    [[ -n "$SERVER_PID" ]] || return 0
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
}

start_server() { # path logfile extra-flags -> 0 on healthy
    # A previous row's server still on the port would be scored as this row's.
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -q LISTEN; then
        echo "port $PORT is still in use; not booting this row" >&2
        return 1
    fi
    # shellcheck disable=SC2086 # $3 is a flag list
    "$BINARY" --model "$1" --serve --port "$PORT" --log-level info \
        --ctx-size 32768 $3 > "$2" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 300); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    return 1
}

strip_ansi() { sed -E $'s/\x1b\\[[0-9;]*m//g'; }

trap 'stop_server' EXIT
running=$(pgrep -x sushi | paste -sd' ' -)
if [[ -n "$running" ]]; then
    echo "sushi is already running (pid $running); stop it first, the models cannot coexist with it" >&2
    exit 1
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -q LISTEN; then
    echo "port $PORT is already in use; stop that server or set PORT" >&2
    exit 1
fi

[[ -f "$SUMMARY" ]] || printf "timestamp\tmodel\tprobe\tpi\tnotes\n" > "$SUMMARY"

overall_fail=0
declare -a ROWS

for entry in "${MODELS[@]}"; do
    IFS='|' read -r logical display path pi_case probe_timeout extra <<< "$entry"
    echo -e "\n${YELLOW}===== $logical — $display =====${NC}"
    if [[ ! -e "$path" ]]; then
        echo -e "${YELLOW}SKIP${NC}: model path missing: $path"
        ROWS+=("$logical|SKIP|SKIP|model missing")
        continue
    fi

    probe_cell="skipped"; pi_cell="skipped"; notes=""

    # ---- Layer 1: llmprobe across all three API surfaces -------------------
    if [[ -z "${SKIP_PROBE:-}" ]]; then
        server_log="$RESULTS/$logical.server.log"
        probe_log="$RESULTS/$logical.probe.log"
        if start_server "$path" "$server_log" "$extra"; then
            node "$LLMPROBE_MJS" "127.0.0.1:$PORT" --timeout "$probe_timeout" \
                2>&1 | strip_ansi > "$probe_log"
            probe_rc=${PIPESTATUS[0]}
            # Per-suite lines: "Results: X passed, Y failed, Z skipped, N total"
            probe_cell=$(awk '/^Results:/ {gsub(/[^0-9 ]/,""); p+=$1; f+=$2; s+=$3; t+=$4}
                END { if (t=="") print "no-results"; else printf "%d/%d", p, t-s }' "$probe_log")
            failed_ids=$(grep '^Failed:' "$probe_log" | sed 's/^Failed: //' | paste -sd';' - | head -c 300)
            if [[ "$probe_rc" -ne 0 || "$probe_cell" == "no-results" ]]; then
                overall_fail=1
                notes="$notes probe-fail:[${failed_ids:-see $probe_log}]"
                echo -e "${RED}probe: $probe_cell${NC} ($probe_log)"
            else
                echo -e "${GREEN}probe: $probe_cell${NC}"
            fi
        else
            overall_fail=1
            probe_cell="server-start-fail"
            notes="$notes server-start-fail"
            echo -e "${RED}server failed to boot — tail of $server_log:${NC}"
            tail -n 20 "$server_log"
        fi
        stop_server
    fi

    # ---- Layer 2: pi agentic 2-turn html case (multi-turn tool calls) ------
    if [[ -z "${SKIP_PI:-}" && -n "$pi_case" ]]; then
        pi_log="$RESULTS/$logical.pi.log"
        MLX_BIN="$BINARY" PI_CASES="$pi_case" "$REPO/tests/pi_integration_run.sh" html \
            2>&1 | strip_ansi > "$pi_log"
        score_line=$(grep -E '^SCORE:' "$pi_log" | tail -n1)
        if [[ -n "$score_line" ]]; then
            pi_cell=$(echo "$score_line" | awk '{print $2}')
            pi_notes=$(echo "$score_line" | cut -d' ' -f3-)
            notes="$notes $pi_notes"
            if echo "$score_line" | grep -qE ' -[a-z]' || [[ "${pi_cell%%/*}" != "${pi_cell##*/}" && "${pi_cell%%/*}" -lt "${pi_cell##*/}" ]]; then
                overall_fail=1
                echo -e "${RED}pi: $score_line${NC} ($pi_log)"
            else
                echo -e "${GREEN}pi: $score_line${NC}"
            fi
        else
            overall_fail=1
            pi_cell="no-score"
            notes="$notes pi-no-score"
            echo -e "${RED}pi produced no SCORE line — tail of $pi_log:${NC}"
            tail -n 20 "$pi_log"
        fi
    fi

    ROWS+=("$logical|$probe_cell|$pi_cell|${notes# }")
    printf "%s\t%s\t%s\t%s\t%s\n" "$(date +%Y-%m-%dT%H:%M:%S)" "$logical" \
        "$probe_cell" "$pi_cell" "${notes# }" >> "$SUMMARY"
done

echo ""
echo "===== validator matrix summary ====="
printf "%-14s %-18s %-10s %s\n" "model" "llmprobe" "pi" "notes"
for row in "${ROWS[@]}"; do
    IFS='|' read -r m p a n <<< "$row"
    printf "%-14s %-18s %-10s %s\n" "$m" "$p" "$a" "$n"
done
echo ""
[[ "$overall_fail" -eq 0 ]] && echo -e "${GREEN}ALL GREEN${NC}" || echo -e "${RED}FAILURES — see $RESULTS${NC}"
exit "$overall_fail"
