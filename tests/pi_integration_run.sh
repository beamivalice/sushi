#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# pi ↔ sushi integration test driver.
#
# Tests every model × streaming × thinking combo by pointing the `pi`
# (https://github.com/badlogic/pi-mono) coding agent at a running sushi
# instance and having it build + test a tiny express todo app.
#
# Requires: pi (npm -g @mariozechner/pi-coding-agent), node, python3.
#
# Usage: tests/pi_integration_run.sh [matrix]
#   matrix=all (default): qwen4_exp thinking off + on, express-todo scenario
#   matrix=quick       : qwen4_exp thinking on only, express-todo scenario
#   matrix=html        : qwen4_exp and mimo_v2, 2-turn html scenario —
#                        turn 1 creates mlx.html, turn 2 adds JS; scored on
#                        file existence/structure/content/JS plus the
#                        audit_format markers (junk filenames, tag leaks,
#                        thinking separation in the pi session)
#   matrix=html-quick  : same as html
#   PI_CASES=csv       : filter cases by label (e.g. PI_CASES=html-qwen4)
#   QWEN4_EXP_MODEL    : pack path override
#   MIMO_MODEL         : MiMo pack path override (served resident)
#   MLX_BIN=path       : server binary override (default: zig-out/bin/sushi)
#
# Writes per-run logs into tests/pi-results/ and appends a
# summary line to tests/pi_integration_run.summary.tsv.

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$REPO/tests/pi-results"
SUMMARY="$REPO/tests/pi_integration_run.summary.tsv"
MLX_BIN="${MLX_BIN:-$REPO/zig-out/bin/sushi}"
PI_MODELS_JSON="$HOME/.pi/agent/models.json"
PORT="${PORT:-8080}"
SERVED_MODEL="${SERVED_MODEL:-}"
WORKSPACE_ROOT="/tmp/pi_mlx_workspaces"

MATRIX="${1:-all}"
case "$MATRIX" in
    html*) SCENARIO="html" ;;
    *)     SCENARIO="todo" ;;
esac

mkdir -p "$RESULTS" "$WORKSPACE_ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'

QWEN4="${QWEN4_EXP_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw}"
MIMO="${MIMO_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/MiMo-V2.6-Flash-Sushi-2.3bpw}"

SUSHI_PID=""
kill_sushi() {
    [ -n "$SUSHI_PID" ] || return 0
    kill "$SUSHI_PID" 2>/dev/null
    # A large model unloads slowly; past a minute it is forced, never left on the port.
    for _ in $(seq 1 120); do
        if ! kill -0 "$SUSHI_PID" 2>/dev/null; then
            SUSHI_PID=""
            return 0
        fi
        sleep 0.5
    done
    kill -9 "$SUSHI_PID" 2>/dev/null
    SUSHI_PID=""
}

# A server still on the port would answer the next case with the wrong model.
require_free_port() {
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -q LISTEN; then
        echo "port $PORT is already in use; stop that server or set PORT" >&2
        return 1
    fi
}

start_sushi() {
    local path="$1"
    local logfile="$2"
    require_free_port || return 1
    "$MLX_BIN" --model "$path" --serve --port "$PORT" \
        --log-level info --ctx-size 32768 > "$logfile" 2>&1 &
    local pid=$!
    echo "$pid"
    # Wait up to 4 min for large models (35B has 27GB weights)
    for i in $(seq 1 240); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "ready-after=${i}s" >&2
            return 0
        fi
        sleep 1
    done
    echo "FAILED to start sushi" >&2
    return 1
}

write_pi_models_config() {
    local model_id="$1"
    local thinking_format="$2"   # "qwen" for Qwen, empty otherwise
    local reasoning="$3"          # true/false
    mkdir -p "$(dirname "$PI_MODELS_JSON")"
    if [ -n "$thinking_format" ]; then
        cat > "$PI_MODELS_JSON" <<EOF
{
  "providers": {
    "mlx": {
      "baseUrl": "http://127.0.0.1:$PORT/v1",
      "api": "openai-completions",
      "apiKey": "mlx",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "maxTokensField": "max_tokens",
        "thinkingFormat": "$thinking_format"
      },
      "models": [
        {"id": "$model_id", "name": "mlx-$model_id", "input": ["text"],
         "contextWindow": 32768, "maxTokens": 8192, "reasoning": $reasoning}
      ]
    }
  }
}
EOF
    else
        cat > "$PI_MODELS_JSON" <<EOF
{
  "providers": {
    "mlx": {
      "baseUrl": "http://127.0.0.1:$PORT/v1",
      "api": "openai-completions",
      "apiKey": "mlx",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {"id": "$model_id", "name": "mlx-$model_id", "input": ["text"],
         "contextWindow": 32768, "maxTokens": 8192, "reasoning": $reasoning}
      ]
    }
  }
}
EOF
    fi
}

# -----------------------------------------------------------------------------
# The actual agentic test. Two turns in a single pi session:
#   1. "build me a todo app with express"
#   2. "now add jest tests for it"
# Success criteria:
#   * package.json exists with express + jest deps
#   * at least one .js source file under project root that imports express
#   * at least one *.test.js file
#   * running `node -e "require('./...')"` doesn't crash on syntax
# -----------------------------------------------------------------------------
run_agent_turn() {
    # $1=model_id, $2=thinking_flag, $3=session_path, $4=workspace, $5=prompt, $6=logfile
    local model="$1" thinking="$2" session="$3" workspace="$4" prompt="$5" logfile="$6"
    local session_arg=""
    if [ -f "$session" ]; then session_arg="--session $session"; fi
    # Absolute timeout 15 min so pathological runs can't hang CI
    local t0=$(date +%s)
    (
        cd "$workspace"
        # PI_OFFLINE avoids network checks at startup
        PI_OFFLINE=1 pi --provider mlx --model "$model" \
            $thinking $session_arg --session-dir "$workspace/.pi-session" \
            --tools read,bash,edit,write,grep,find,ls \
            -p "$prompt" 2>&1 | tee -a "$logfile"
    )
    local rc=$?
    local t1=$(date +%s)
    echo "[elapsed=$((t1-t0))s exit=$rc]" | tee -a "$logfile"
    return $rc
}

score_workspace() {
    local ws="$1"
    local score=0
    local notes=""
    if [ -f "$ws/package.json" ]; then
        if grep -q '"express"' "$ws/package.json"; then
            score=$((score+1)); notes="$notes +package.json(express)"
        else
            notes="$notes -package.json(no-express)"
        fi
        if grep -qE '"jest"|"vitest"|"mocha"|"node:test"' "$ws/package.json"; then
            score=$((score+1)); notes="$notes +test-runner"
        fi
    else
        notes="$notes -no-package.json"
    fi
    local src
    src=$(find "$ws" -maxdepth 3 -name "*.js" -not -path "*/node_modules/*" -not -path "*/.pi-session/*" 2>/dev/null | grep -v '\.test\.js$' | head -1)
    if [ -n "$src" ] && grep -q "express" "$src" 2>/dev/null; then
        score=$((score+1)); notes="$notes +src($(basename $src))"
    else
        notes="$notes -no-express-src"
    fi
    local test
    test=$(find "$ws" -maxdepth 3 -name "*.test.js" -not -path "*/node_modules/*" -not -path "*/.pi-session/*" 2>/dev/null | head -1)
    if [ -n "$test" ]; then
        score=$((score+1)); notes="$notes +test($(basename $test))"
    else
        notes="$notes -no-test-file"
    fi
    # Bonus: does `npm test` actually pass? (skip if deps not installed)
    if [ -d "$ws/node_modules" ] && [ -n "$test" ]; then
        if (cd "$ws" && npx --no-install jest --silent >/dev/null 2>&1); then
            score=$((score+1)); notes="$notes +jest-green"
        else
            notes="$notes -jest-failed-or-not-run"
        fi
    else
        notes="$notes -no-node-modules"
    fi
    echo "$score|$notes"
}

# HTML scenario scorer — 2-turn create→extend flow:
#   turn 1 must produce mlx.html with real HTML structure and MLX content,
#   turn 2 must add the requested inline JavaScript to the SAME file.
score_workspace_html() {
    local ws="$1"
    local score=0
    local notes=""
    if [ -f "$ws/mlx.html" ]; then
        score=$((score+1)); notes="$notes +mlx.html"
        if grep -qi "<html" "$ws/mlx.html"; then
            score=$((score+1)); notes="$notes +html-structure"
        else
            notes="$notes -no-html-tag"
        fi
        if grep -qi "mlx" "$ws/mlx.html"; then
            score=$((score+1)); notes="$notes +mlx-content"
        else
            notes="$notes -no-mlx-content"
        fi
        # Inline JS counts as either a <script> block or an inline handler
        # (onclick=…) — both satisfy "inline JavaScript in the same file".
        if grep -qi "alert" "$ws/mlx.html" && grep -qiE "<script|onclick" "$ws/mlx.html"; then
            score=$((score+1)); notes="$notes +js-added"
        else
            notes="$notes -no-js"
        fi
    else
        notes="$notes -no-mlx.html"
    fi
    echo "$score|$notes"
}

# Format-correctness audit (Layer 3 of the cross-model format suite; the other
# layers are src/format_corpus_test.zig and tests/test_format_matrix.sh).
# Appends markers to the TSV notes column — score is untouched, but any `-`
# marker here is a hard-fail signal:
#   -junk-filename(<name>)  a created file whose basename carries tag/quote/
#                           brace garbage (the brace-swallow bug literally
#                           wrote "mlx_pi1.html`}" to disk)
#   +exact-filenames        the task's required files exist under their exact
#                           names (package.json, app.js, app.test.js)
#   -tag-leak(<match>)      a raw control tag (think/channel/tool-call/string
#                           delimiter) reached the agent transcript
audit_format() {
    local ws="$1" agent_log="$2"
    local notes=""
    local junk
    junk=$(find "$ws" -maxdepth 3 -not -path "*/node_modules/*" -not -path "*/.pi-session/*" -print0 2>/dev/null \
        | python3 -c '
import sys, os, re
bad = []
for p in sys.stdin.buffer.read().split(b"\0"):
    if not p:
        continue
    name = os.path.basename(p.decode("utf-8", "replace"))
    if re.search(r"[`<>|}{\"\x27]", name) or "<|" in name:
        bad.append(name)
print(",".join(bad[:3]))
')
    if [ -n "$junk" ]; then
        notes="$notes -junk-filename($junk)"
    elif { [ -f "$ws/package.json" ] && [ -f "$ws/app.js" ] && [ -f "$ws/app.test.js" ]; } \
        || [ -f "$ws/mlx.html" ]; then
        notes="$notes +exact-filenames"
    fi
    local leak
    leak=$(grep -aoE '</?think>|<\|channel>|<channel\|>|<\|tool_call|<tool_call>|<\|"\|>' "$agent_log" 2>/dev/null | head -1)
    if [ -n "$leak" ]; then
        notes="$notes -tag-leak($leak)"
    else
        notes="$notes +no-tag-leak"
    fi
    # Session-level audit: in the pi session jsonl, thinking must live in
    # thinking blocks and text blocks must never carry raw control tags.
    #   +thinking-blocks         at least one non-empty thinking block
    #   -thinking-in-text(<tag>) a raw control tag inside a TEXT block
    local sess
    sess=$(ls -t "$ws/.pi-session/"*.jsonl 2>/dev/null | head -1)
    if [ -n "$sess" ]; then
        local sres
        sres=$(python3 - "$sess" <<'PY'
import json, sys
tags = ["<think>", "</think>", "<|channel>", "<channel|>", "<|tool_call", "<tool_call>", '<|"|>']
think = 0
leak = ""
for line in open(sys.argv[1]):
    try:
        o = json.loads(line)
    except Exception:
        continue
    m = o.get("message") or {}
    if m.get("role") != "assistant":
        continue
    c = m.get("content")
    if not isinstance(c, list):
        continue
    for b in c:
        if b.get("type") == "thinking" and (b.get("thinking") or "").strip():
            think = 1
        if b.get("type") == "text":
            t = b.get("text", "")
            for tag in tags:
                if tag in t:
                    leak = tag
                    break
print(f"{think}|{leak}")
PY
)
        local sthink="${sres%%|*}" sleak="${sres#*|}"
        [ -n "$sleak" ] && notes="$notes -thinking-in-text($sleak)"
        [ "$sthink" = "1" ] && notes="$notes +thinking-blocks"
    fi
    echo "$notes"
}

run_one_case() {
    local label="$1"        # e.g. "qwen-think-medium"
    local path="$2"
    local served_name="$3"
    local thinking_flag="$4"
    local thinking_format="$5"
    local reasoning="$6"

    echo -e "${YELLOW}===== $label =====${NC}"
    local ws="$WORKSPACE_ROOT/$label"
    rm -rf "$ws" && mkdir -p "$ws"
    local server_log="$RESULTS/$label.server.log"
    local agent_log="$RESULTS/$label.agent.log"
    : > "$server_log"; : > "$agent_log"

    local pid load_start load_end
    load_start=$(date +%s)
    if [ -z "${SKIP_SERVER_START:-}" ]; then
        pid=$(start_sushi "$path" "$server_log" 2> >(tee -a "$agent_log"))
        if [ -z "$pid" ]; then
            echo "FAIL: server failed" | tee -a "$agent_log"
            printf "%s\t%s\t%s\t%s\t%s\n" "$(date +%Y-%m-%dT%H:%M:%S)" "$label" "server-start-fail" "0" "" >> "$SUMMARY"
            return 1
        fi
        SUSHI_PID=$pid
        load_end=$(date +%s)
        echo "sushi PID=$pid (loaded in $((load_end-load_start))s)" | tee -a "$agent_log"
    else
        pid="external"
        load_end=$load_start
        echo "[using already-running server at :$PORT]" | tee -a "$agent_log"
    fi

    # Allow env override of the model id used in pi's models.json.
    local effective_name="${SERVED_MODEL:-$served_name}"
    write_pi_models_config "$effective_name" "$thinking_format" "$reasoning"
    served_name="$effective_name"

    # Turn 1
    local turn1_prompt turn2_prompt_html
    if [ "$SCENARIO" = "html" ]; then
        turn1_prompt="Make me an html page about the MLX framework on Mac. Keep it minimal: a heading and a few bullet points. Save it as mlx.html in this directory. When done, say 'page ready'."
    else
        turn1_prompt="Create a minimal Express.js todo app in this directory. Requirements: package.json with express as dep, a file app.js exporting the Express app, in-memory todo storage, REST endpoints GET /todos, POST /todos (json body {text}), DELETE /todos/:id. Keep it in one file. Do NOT start the server yourself. When done, say 'app ready'."
    fi
    local t0=$(date +%s)
    run_agent_turn "$served_name" "$thinking_flag" "$ws/.pi-session/session.jsonl" \
        "$ws" "$turn1_prompt" "$agent_log"
    local t1=$(date +%s)
    echo "[turn1 elapsed=$((t1-t0))s]" | tee -a "$agent_log"

    # Turn 2 — continue same session
    local turn2_prompt
    if [ "$SCENARIO" = "html" ]; then
        turn2_prompt="Now add a button to mlx.html that shows an alert saying 'Hello from MLX' when clicked, using inline JavaScript in the same file. When done, say 'js added'."
    else
        turn2_prompt="Now add jest as a dev dependency in package.json and write a jest test file app.test.js that uses supertest to test all three endpoints. Also install dependencies with npm install. When done, run the tests and show me the output."
    fi
    local session_path="$ws/.pi-session/session.jsonl"
    # pi keeps the first session we started — use --continue
    t0=$(date +%s)
    (
        cd "$ws"
        PI_OFFLINE=1 pi --provider mlx --model "$served_name" \
            $thinking_flag --session-dir "$ws/.pi-session" --continue \
            --tools read,bash,edit,write,grep,find,ls \
            -p "$turn2_prompt" 2>&1 | tee -a "$agent_log"
    )
    t1=$(date +%s)
    echo "[turn2 elapsed=$((t1-t0))s]" | tee -a "$agent_log"

    # Grade
    local result max_score
    if [ "$SCENARIO" = "html" ]; then
        result=$(score_workspace_html "$ws")
        max_score=4
    else
        result=$(score_workspace "$ws")
        max_score=5
    fi
    local score=$(echo "$result" | cut -d'|' -f1)
    local notes=$(echo "$result" | cut -d'|' -f2)
    notes="$notes$(audit_format "$ws" "$agent_log")"
    local total_elapsed=$(( $(date +%s) - load_start ))
    echo "SCORE: $score/$max_score $notes [total=${total_elapsed}s]" | tee -a "$agent_log"
    printf "%s\t%s\t%s\t%s\t%s\n" "$(date +%Y-%m-%dT%H:%M:%S)" "$label" "$score/$max_score" "$total_elapsed" "$notes" >> "$SUMMARY"

    if [ -z "${SKIP_SERVER_START:-}" ]; then
        kill_sushi
    fi
    return 0
}

# -----------------------------------------------------------------------------
# Cases
# -----------------------------------------------------------------------------
declare -a CASES

# Case format: label|path|served_name|thinking_flag|thinking_format|reasoning  (6 fields, 5 pipes)
# qwen4_exp — streaming, thinking MEDIUM (pi sends enable_thinking=true)
CASES+=("qwen4-think|$QWEN4|qwen4_exp|--thinking medium|qwen|true")

if [ "$MATRIX" = "all" ]; then
    # qwen4_exp — streaming, thinking OFF (pi sends enable_thinking=false)
    CASES+=("qwen4-no-think|$QWEN4|qwen4_exp|--thinking off|qwen|true")
fi

# MiMo serves resident; its template reads enable_thinking as Qwen's does.
if [ "$SCENARIO" = "html" ]; then
    CASES=()
    CASES+=("html-qwen4|$QWEN4|qwen4_exp|--thinking medium|qwen|true")
    CASES+=("html-mimo|$MIMO|mimo_v2|--thinking medium|qwen|true")
fi

if [ ! -f "$SUMMARY" ]; then
    printf "timestamp\tlabel\tscore\telapsed_s\tnotes\n" > "$SUMMARY"
fi

if [ -z "${SKIP_SERVER_START:-}" ]; then
    require_free_port || exit 1
fi

for case in "${CASES[@]}"; do
    IFS='|' read -r label path served_name thinking_flag thinking_format reasoning <<< "$case"
    if [ -n "${PI_CASES:-}" ]; then
        case ",$PI_CASES," in
            *",$label,"*) ;;
            *) continue ;;
        esac
    fi
    if [ ! -e "$path" ]; then
        echo -e "${YELLOW}SKIP${NC}: $label (model path missing: $path)"
        continue
    fi
    run_one_case "$label" "$path" "$served_name" "$thinking_flag" "$thinking_format" "$reasoning"
done

kill_sushi
rm -f "$PI_MODELS_JSON"
echo -e "${GREEN}Done. Summary: $SUMMARY${NC}"
cat "$SUMMARY"
