#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# repeat_penalty / presence_penalty reach the default decode path (one boot, MTP on by default).
#
#   1. APPLIED: a penalised request answers the same bytes as with `logprobs` on, whose synchronous
#      path always applied the penalty (before the fix the pipelined sampler and every MTP verify
#      dropped it); and the penalty changes the answer on at least one of the two penalties.
#   2. MTP == SERIAL: the penalised request gives the same bytes with enable_mtp true and false, and
#      it ENGAGES MTP (a new `[spec-stats] mode=mtp` line with attempts > 0), so the equality is not
#      two serial runs.
#   3. STREAM == NON-STREAM for the penalised request, and for a json_schema + penalty request,
#      whose content must still parse against its schema.
#   4. A penalised request beside a concurrent unpenalised neighbour: both keep their solo bytes.
#
# Usage: PENALTY_TEST_MODEL=<pack> ./tests/test_penalty_paths.sh [port]
# Default model: ${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-3bpw. SUSHI_TEST_EXTRA_ARGS adds server flags
# (a streamed pack: `--ssd-budget-gb 18`).
set -u
MODEL="${PENALTY_TEST_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-3bpw}"
PORT="${1:-11319}"
BIN="${SUSHI_BINARY:-./zig-out/bin/sushi}"
EXTRA_ARGS="${SUSHI_TEST_EXTRA_ARGS:-}"
if [ ! -d "$MODEL" ]; then
    echo "SKIP: model not found at $MODEL"
    exit 0
fi
if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/health"; then
    echo "FAIL: port $PORT is already serving; pass a free port"
    exit 1
fi
ARTIFACTS="${PENALTY_TEST_OUTPUT_DIR:-$(mktemp -d)}"
mkdir -p "$ARTIFACTS"
LOG="$ARTIFACTS/server.log"
# shellcheck disable=SC2086
"$BIN" --model "$MODEL" --serve --port "$PORT" --no-drafter --prefix-cache-entries 0 \
    --ctx-size 8192 --max-concurrent 4 --log-level info $EXTRA_ARGS >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null' EXIT
for _ in $(seq 1 400); do
    grep -q "Model ready (loaded on inference thread)" "$LOG" && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "FAIL: server exited"; tail -20 "$LOG"; exit 1; }
    sleep 3
done

ENGAGE_PRE=$(grep -c "\[spec-stats\] mode=mtp" "$LOG")
curl -s --max-time 600 -o /dev/null -H 'Content-Type: application/json' "http://127.0.0.1:$PORT/v1/chat/completions" \
    -d '{"model":"default","temperature":0,"max_tokens":160,"enable_thinking":false,"presence_penalty":2.0,"messages":[{"role":"user","content":"Count from 1 to 40 separated by commas, then write the word apple ten times."}]}'
sleep 1
if [ "$(grep -c "\[spec-stats\] mode=mtp" "$LOG")" -le "$ENGAGE_PRE" ]; then
    echo "FAIL: a penalised request logged no '[spec-stats] mode=mtp' line (it ran without MTP)"
    exit 1
fi
ATTEMPTS=$(grep -o '\[spec-stats\] mode=mtp.*' "$LOG" | tail -1 | grep -o 'attempts=[0-9]*' | grep -o '[0-9]*')
if [ "${ATTEMPTS:-0}" -le 0 ]; then
    echo "FAIL: the penalised request's MTP stats show attempts=${ATTEMPTS:-none}"
    exit 1
fi

python3 - "$PORT" <<'PY'
import json, sys, threading, urllib.request

port = sys.argv[1]
BASE = {"model": "default", "temperature": 0, "max_tokens": 160, "enable_thinking": False,
        "messages": [{"role": "user", "content": "Count from 1 to 40 separated by commas, then write the word apple ten times."}]}
OTHER = dict(BASE, max_tokens=200, messages=[{"role": "user", "content": "Explain in a paragraph why the sky is blue."}])
SCHEMA = {"type": "json_schema", "json_schema": {"name": "fruits", "schema": {
    "type": "object", "properties": {"fruits": {"type": "array", "items": {"type": "string"}}}, "required": ["fruits"]}}}

def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read().decode()

def plain(base=BASE, **kw):
    return json.loads(post(dict(base, stream=False, **kw)))["choices"][0]["message"]["content"]

def streamed(base=BASE, **kw):
    out = []
    for line in post(dict(base, stream=True, **kw)).splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            for ch in json.loads(line[6:]).get("choices", []):
                out.append(ch.get("delta", {}).get("content") or "")
    return "".join(out)

fails = []
free = plain()
pen_mtp = plain(presence_penalty=2.0)
pen_serial = plain(presence_penalty=2.0, enable_mtp=False)
pen_stream = streamed(presence_penalty=2.0)
rep_mtp = plain(repeat_penalty=1.5)
rep_serial = plain(repeat_penalty=1.5, enable_mtp=False)
if pen_mtp != plain(presence_penalty=2.0, logprobs=True, top_logprobs=1):
    fails.append("presence_penalty: default path != the logprobs path, which applies it")
if rep_mtp != plain(repeat_penalty=1.5, logprobs=True, top_logprobs=1):
    fails.append("repeat_penalty: default path != the logprobs path, which applies it")
if pen_mtp == free and rep_mtp == free:
    fails.append("neither penalty changed the answer: the prompt no longer exercises them")
if pen_mtp != pen_serial:
    fails.append("presence_penalty: enable_mtp true != false")
if rep_mtp != rep_serial:
    fails.append("repeat_penalty: enable_mtp true != false")
if pen_mtp != pen_stream:
    fails.append("presence_penalty: stream != non-stream")
BOTH = dict(presence_penalty=1.0, repeat_penalty=1.2)
if plain(OTHER, **BOTH) != plain(OTHER, enable_mtp=False, **BOTH):
    fails.append("both penalties, second prompt: enable_mtp true != false")

schema_plain = plain(presence_penalty=1.0, response_format=SCHEMA)
schema_stream = streamed(presence_penalty=1.0, response_format=SCHEMA)
try:
    if not isinstance(json.loads(schema_plain).get("fruits"), list):
        fails.append("json_schema + penalty: no fruits array")
except ValueError:
    fails.append(f"json_schema + penalty: content is not JSON: {schema_plain[:80]!r}")
if schema_plain != schema_stream:
    fails.append("json_schema + penalty: stream != non-stream")

other_solo = plain(OTHER)
got = {}
threads = [threading.Thread(target=lambda: got.__setitem__("pen", plain(presence_penalty=2.0))),
           threading.Thread(target=lambda: got.__setitem__("other", plain(OTHER)))]
for t in threads:
    t.start()
for t in threads:
    t.join()
if got.get("pen") != pen_mtp:
    fails.append("penalised request beside a neighbour != its solo bytes")
if got.get("other") != other_solo:
    fails.append("unpenalised neighbour != its solo bytes")
for f in fails:
    print("FAIL:", f)
print("PASS" if not fails else f"{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
PY
