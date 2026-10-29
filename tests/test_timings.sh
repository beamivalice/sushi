#!/bin/bash
# sushi-test: tier=live boot=api
# Integration test: chat completions emit a llama.cpp-style `timings` block
# alongside `usage`, on both the non-stream JSON body and the final SSE chunk
# of a streaming response (when stream_options.include_usage is set).
#
# Required fields (llama.cpp shape):
#   prompt_n, prompt_ms, prompt_per_second,
#   predicted_n, predicted_ms, predicted_per_second
#
# `timings.tokenize_ms` rides every API surface (chat completions, Anthropic messages, responses) as a finite,
# non-negative number. Its size is not asserted: a wall-clock bound flips with the box.
#
# Attaches to the shared api boot: tests/run.sh --tier live --model <m> --only test_timings.sh

. "$(dirname "$0")/lib/boot.sh"
sushi_attach
BASE=$SUSHI_URL
FAIL=0

echo "=== Chat-completions timings field ==="
echo "Server: $BASE ($SUSHI_ARCH)"

command -v jq >/dev/null || { echo "SKIP test_timings.sh: jq is required"; exit 3; }

# ── Test 1: non-streaming chat completion ──────────────────────────────────
echo
echo "--- Test 1: non-stream chat.completion carries a top-level 'timings' object ---"
RESULT=$(curl -sf -m 600 "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sushi",
    "messages": [{"role":"user","content":"'"$SUSHI_NONCE"' Say hi in five words."}],
    "max_tokens": 16,
    "temperature": 0.0,
    "stream": false
  }')
if [ -z "$RESULT" ]; then
    echo "  FAIL: empty response from server"
    exit 1
fi

python3 -I - "$RESULT" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
t = r.get("timings")
if not isinstance(t, dict):
    print(f"  FAIL: no 'timings' object in response. keys={list(r.keys())}")
    sys.exit(1)
required = ["prompt_n","prompt_ms","prompt_per_second",
           "predicted_n","predicted_ms","predicted_per_second"]
missing = [k for k in required if k not in t]
if missing:
    print(f"  FAIL: timings missing fields: {missing}. got={list(t.keys())}")
    sys.exit(1)
for k in ["prompt_n","predicted_n"]:
    if not isinstance(t[k], int) or t[k] < 0:
        print(f"  FAIL: timings.{k} not a non-negative int: {t[k]!r}")
        sys.exit(1)
for k in ["prompt_ms","predicted_ms","prompt_per_second","predicted_per_second"]:
    if not isinstance(t[k], (int,float)) or t[k] < 0:
        print(f"  FAIL: timings.{k} not a non-negative number: {t[k]!r}")
        sys.exit(1)
# usage must still contain the standard OpenAI counts and agree with timings.
u = r.get("usage", {})
if u.get("prompt_tokens") != t["prompt_n"]:
    print(f"  FAIL: usage.prompt_tokens={u.get('prompt_tokens')} != timings.prompt_n={t['prompt_n']}")
    sys.exit(1)
if u.get("completion_tokens") != t["predicted_n"]:
    print(f"  FAIL: usage.completion_tokens={u.get('completion_tokens')} != timings.predicted_n={t['predicted_n']}")
    sys.exit(1)
# When tokens > 0 the per-second figures must be > 0 (we generated something).
if t["predicted_n"] > 0 and t["predicted_per_second"] <= 0:
    print(f"  FAIL: predicted_n={t['predicted_n']} but predicted_per_second={t['predicted_per_second']}")
    sys.exit(1)
print(f"  PASS  timings={json.dumps(t)}")
PY
RC=$?
[ $RC -ne 0 ] && FAIL=1

# ── Test 2: streaming chat completion ──────────────────────────────────────
echo
echo "--- Test 2: streaming final SSE chunk carries 'timings' alongside 'usage' ---"
STREAM=$(curl -sN -m 600 "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sushi",
    "messages": [{"role":"user","content":"'"$SUSHI_NONCE"' Say hi in five words."}],
    "max_tokens": 16,
    "temperature": 0.0,
    "stream": true,
    "stream_options": {"include_usage": true}
  }')
if [ -z "$STREAM" ]; then
    echo "  FAIL: empty stream"
    exit 1
fi

python3 -I - "$STREAM" <<'PY'
import json, sys
raw = sys.argv[1]
chunks = []
for line in raw.splitlines():
    if not line.startswith("data: "):
        continue
    payload = line[len("data: "):].strip()
    if payload == "[DONE]" or not payload:
        continue
    try:
        chunks.append(json.loads(payload))
    except json.JSONDecodeError:
        pass

# Find the chunk that carries usage — that's where timings must live too.
usage_chunks = [c for c in chunks if isinstance(c.get("usage"), dict)
                                  and "prompt_tokens" in c["usage"]]
if not usage_chunks:
    print("  FAIL: no SSE chunk with usage object found")
    print("  chunks=", [list(c.keys()) for c in chunks[-3:]])
    sys.exit(1)
last = usage_chunks[-1]
t = last.get("timings")
if not isinstance(t, dict):
    print(f"  FAIL: usage chunk has no 'timings'. keys={list(last.keys())}")
    sys.exit(1)
required = ["prompt_n","prompt_ms","prompt_per_second",
           "predicted_n","predicted_ms","predicted_per_second"]
missing = [k for k in required if k not in t]
if missing:
    print(f"  FAIL: stream timings missing fields: {missing}. got={list(t.keys())}")
    sys.exit(1)
u = last["usage"]
if u.get("prompt_tokens") != t["prompt_n"] or u.get("completion_tokens") != t["predicted_n"]:
    print(f"  FAIL: stream usage/timings token counts disagree: usage={u} timings={t}")
    sys.exit(1)
if t["predicted_n"] > 0 and t["predicted_per_second"] <= 0:
    print(f"  FAIL: predicted_n={t['predicted_n']} but predicted_per_second={t['predicted_per_second']}")
    sys.exit(1)
print(f"  PASS  timings={json.dumps(t)}")
PY
RC=$?
[ $RC -ne 0 ] && FAIL=1

# ── Tests 3-5: timings.tokenize_ms on every API surface ────────────────────
# Pull `.timings.tokenize_ms` from a JSON response and verify it is a finite non-negative number.
assert_tokenize_ms_in() {
    local label="$1" body="$2" val
    val="$(echo "$body" | jq -r '.timings.tokenize_ms // empty')"
    if [ -z "$val" ]; then
        echo "  FAIL: $label: timings.tokenize_ms missing"
        FAIL=1
        return
    fi
    if ! python3 -I -c "import sys, math; v=float(sys.argv[1]); assert v >= 0 and math.isfinite(v)" "$val" 2>/dev/null; then
        echo "  FAIL: $label: tokenize_ms=$val (not a finite non-negative number)"
        FAIL=1
        return
    fi
    echo "  PASS  $label: tokenize_ms=${val}ms"
}

echo
echo "--- Test 3: /v1/chat/completions carries timings.tokenize_ms ---"
RESP="$(curl -sf -m 600 -X POST "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"'"$SUSHI_NONCE"' Say hi in one word."}],"max_tokens":5,"temperature":0,"stream":false}')"
assert_tokenize_ms_in "chat-completions" "$RESP"

echo
echo "--- Test 4: /v1/messages carries timings.tokenize_ms ---"
RESP="$(curl -sf -m 600 -X POST "$BASE/v1/messages" \
    -H 'Content-Type: application/json' \
    -d '{"model":"sushi","messages":[{"role":"user","content":"'"$SUSHI_NONCE"' Say hi in one word."}],"max_tokens":5,"temperature":0,"stream":false}')"
assert_tokenize_ms_in "anthropic-messages" "$RESP"

echo
echo "--- Test 5: /v1/responses carries timings.tokenize_ms ---"
RESP="$(curl -sf -m 600 -X POST "$BASE/v1/responses" \
    -H 'Content-Type: application/json' \
    -d '{"model":"sushi","input":"'"$SUSHI_NONCE"' Say hi in one word.","max_output_tokens":16,"temperature":0,"stream":false}')"
assert_tokenize_ms_in "responses" "$RESP"

echo
if [ $FAIL -ne 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL TESTS PASSED"
