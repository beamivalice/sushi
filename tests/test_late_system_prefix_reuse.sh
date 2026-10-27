#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# test_late_system_prefix_reuse.sh — a mid-conversation system turn keeps the prefix cache.
#
# Codex sends a new `developer` turn mid-input and Claude Code sends hook output as
# a `system` message between turns. /v1/responses and /v1/messages used to fold every
# such turn into the leading system message, so on MiMo (whose template places a late
# system turn) turn N's prompt stopped being a prefix of turn N+1's and every turn
# prefilled from scratch. Per surface, greedy, thinking off:
#
#   1. turn 1 answers; turn 2 = turn 1 + its reply + a NEW late system/developer turn
#      + a user turn
#   2. MiMo and Qwen3.8: turn 2 reads at least 90% of turn 1's prompt from the cache
#      (`input_tokens_details.cached_tokens`, `cache_read_input_tokens`)
#   3. Qwen3.8 (the stock template refuses a late system turn; the late-system adapter renders
#      it in place): the same 90% bar, which assumes a pack that ships the stock template
#
# Usage: ./tests/test_late_system_prefix_reuse.sh [pack_dir ...]
# Env: SUSHI_MODELS_DIR (default $HOME/.sushi/models), QWEN_MODEL, MIMO_MODEL,
#      PORT (default 19143), BINARY, GPU_LOCK_OWNER. One GPU-lock run per pack under `taskpolicy -a`;
#      logs under ~/.sushi/runs/late-system-prefix-reuse/.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MODELS_DIR="${SUSHI_MODELS_DIR:-$HOME/.sushi/models}"
if [ $# -gt 0 ]; then
    PACKS=("$@")
else
    PACKS=("${MIMO_MODEL:-$MODELS_DIR/MiMo-V2.6-Flash-Sushi-2.3bpw}" "${QWEN_MODEL:-$MODELS_DIR/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw}")
fi
PORT="${PORT:-19143}"
BIN="${BINARY:-./zig-out/bin/sushi}"
BASE="http://127.0.0.1:$PORT"
LOCK_OWNER="${GPU_LOCK_OWNER:-late-system-prefix-reuse}"
RUNS="$HOME/.sushi/runs/late-system-prefix-reuse"

[ -x "$BIN" ] || { echo "fail: build sushi first (zig build -Doptimize=ReleaseFast)"; exit 1; }
curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1 && { echo "fail: port $PORT is busy"; exit 1; }

SERVER_PID=""
LOCKED=0
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
    [ "$LOCKED" = 1 ] && scripts/gpu-lock.sh release "$LOCK_OWNER" >/dev/null
    LOCKED=0
}
trap cleanup EXIT

TOTAL_FAIL=0
for PACK in "${PACKS[@]}"; do
    if [ ! -d "$PACK" ]; then echo "SKIP: pack not found: $PACK"; continue; fi
    NAME="$(basename "$PACK")"
    OUT="$RUNS/$NAME-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$OUT"
    MODEL_TYPE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]+"/config.json")).get("model_type",""))' "$PACK")"
    echo "[late-system] === $NAME ($MODEL_TYPE, commit $(git rev-parse --short HEAD), binary $(stat -f %Sm "$BIN")) ==="

    scripts/gpu-lock.sh acquire "$LOCK_OWNER" >/dev/null || { echo "fail: GPU lock"; exit 1; }
    LOCKED=1
    taskpolicy -a "$BIN" --model "$PACK" --serve --host 127.0.0.1 --port "$PORT" --ctx-size 32768 \
        --prefix-cache-entries 2 --log-level info > "$OUT/server.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 900); do
        curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && break
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -20 "$OUT/server.log"; exit 1; }
        sleep 1
    done

    python3 - "$BASE" "$MODEL_TYPE" <<'PY'
import json, sys, urllib.request

BASE, MODEL_TYPE = sys.argv[1], sys.argv[2]
places_late_system = MODEL_TYPE in ("mimo_v2", "qwen4_exp")
passed = failed = 0

def check(label, ok, detail=""):
    global passed, failed
    if ok: passed += 1; print(f"  PASS {label}")
    else: failed += 1; print(f"  FAIL {label} {detail}")

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())

def long_system(topic):
    return (f"You are a careful assistant for the {topic} project. Answer in one word when asked. ") * 40

# /v1/responses, Codex-style: the whole input every turn, a new developer turn mid-conversation.
instructions = long_system("responses")
turn1 = [{"role": "developer", "content": "Environment: repository at /repo."},
         {"role": "user", "content": "Reply with the single word: ready."}]
r1 = post("/v1/responses", {"model": "x", "instructions": instructions, "input": turn1, "temperature": 0,
                            "max_output_tokens": 16, "reasoning": {"effort": "none"}})
reply = "".join(p.get("text", "") for it in r1.get("output", []) if it.get("type") == "message" for p in it.get("content") or [])
turn2 = turn1 + [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": reply}]},
                 {"role": "developer", "content": "Approval mode: never ask."},
                 {"role": "user", "content": "Reply with the single word: done."}]
r2 = post("/v1/responses", {"model": "x", "instructions": instructions, "input": turn2, "temperature": 0,
                            "max_output_tokens": 16, "reasoning": {"effort": "none"}})
p1, c2 = r1["usage"]["input_tokens"], r2["usage"]["input_tokens_details"]["cached_tokens"]
print(f"    responses: turn 1 prompt {p1}, turn 2 prompt {r2['usage']['input_tokens']} cached {c2}")
check("responses: both turns answer", r1.get("status") == "completed" and r2.get("status") == "completed")
if places_late_system:
    check(f"responses: turn 2 reuses turn 1's prompt ({c2} cached of {p1})", c2 >= 0.9 * p1)

# /v1/messages, Claude Code-style: hook output arrives as a system message between turns.
system = long_system("messages")
m_turn1 = [{"role": "user", "content": "Reply with the single word: ready."}]
m1 = post("/v1/messages", {"model": "x", "system": system, "messages": m_turn1, "max_tokens": 16, "temperature": 0,
                           "thinking": {"type": "disabled"}})
m_reply = "".join(b.get("text", "") for b in m1.get("content", []) if b.get("type") == "text")
m_turn2 = m_turn1 + [{"role": "assistant", "content": m_reply},
                     {"role": "system", "content": "Hook: session started."},
                     {"role": "user", "content": "Reply with the single word: done."}]
m2 = post("/v1/messages", {"model": "x", "system": system, "messages": m_turn2, "max_tokens": 16, "temperature": 0,
                           "thinking": {"type": "disabled"}})
mp1, mc2 = m1["usage"]["input_tokens"], m2["usage"]["cache_read_input_tokens"]
print(f"    messages: turn 1 prompt {mp1}, turn 2 prompt {m2['usage']['input_tokens']} cached {mc2}")
check("messages: both turns answer", m1.get("type") == "message" and m2.get("type") == "message")
if places_late_system:
    check(f"messages: turn 2 reuses turn 1's prompt ({mc2} cached of {mp1})", mc2 >= 0.9 * mp1)

print(f"[late-system] {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
PY
    RC=$?
    cleanup
    [ "$RC" = 0 ] || TOTAL_FAIL=$((TOTAL_FAIL + 1))
    echo "[late-system] log: $OUT/server.log"
done
exit "$TOTAL_FAIL"
