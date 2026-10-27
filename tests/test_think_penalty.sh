#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# Think penalty (`--think-penalty`) live contract on a qwen4_exp pack.
#
#   1. ENGAGEMENT: the load logs `[think-penalty] lambda L (--think-penalty)`, every thinking
#      request with lambda > 0 logs `think penalty L:`, and a request's `think_penalty: 0` arms
#      nothing.
#   2. MTP == SERIAL: greedy reasoning + answer bytes with MTP equal --no-mtp's, on a thought the
#      cap cuts open (xhigh) and one that closes before its answer (low).
#   3. STREAM == NON-STREAM on every request.
#
# Usage: THINK_TEST_MODEL=<qwen4_exp pack> ./tests/test_think_penalty.sh [port]
# Default model: ${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw.
set -u
MODEL="${THINK_TEST_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw}"
PORT="${1:-11317}"
BIN="${SUSHI_BINARY:-./zig-out/bin/sushi}"
LAMBDA="${THINK_TEST_LAMBDA:-2}"
if [ ! -d "$MODEL" ]; then
    echo "SKIP: model not found at $MODEL"
    exit 0
fi
ARTIFACTS="${THINK_TEST_OUTPUT_DIR:-$(mktemp -d)}"
mkdir -p "$ARTIFACTS"
SERVER_PID=""
trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; wait 2>/dev/null' EXIT

start_server() { # $1 = arm name, $2 = extra flags
    LOG="$ARTIFACTS/server-$1.log"
    # shellcheck disable=SC2086
    "$BIN" --model "$MODEL" --serve --port "$PORT" --no-drafter --prefix-cache-entries 0 \
        --ctx-size 16384 --think-penalty "$LAMBDA" --log-level info $2 >"$LOG" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 400); do
        grep -q "Model ready (loaded on inference thread)" "$LOG" && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 3
    done
    echo "FAIL: server did not become ready ($1)"; tail -20 "$LOG"; exit 1
}

stop_server() {
    kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""
}

run_requests() { # $1 = arm name
    python3 - "$PORT" "$ARTIFACTS/$1.json" <<'PY'
import json, sys, urllib.request

port, out = sys.argv[1], sys.argv[2]
REQS = [
    ("closes", {"reasoning_effort": "low", "max_tokens": 900, "messages": [{"role": "user", "content":
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"}]}),
    ("open", {"reasoning_effort": "xhigh", "max_tokens": 500, "messages": [{"role": "user", "content":
        "How many positive integers n below 100 make n^2 + n + 41 composite? Check your work."}]}),
    ("off", {"reasoning_effort": "low", "max_tokens": 300, "think_penalty": 0, "messages": [{"role": "user", "content":
        "Name three prime numbers above 50."}]}),
]

def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.read().decode()

res = {}
for name, body in REQS:
    body = dict(body, model="default", temperature=0)
    m = json.loads(post(dict(body, stream=False)))["choices"][0]["message"]
    plain = (m.get("reasoning_content") or "") + "\x01" + (m.get("content") or "")
    think, text = [], []
    for line in post(dict(body, stream=True)).splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        for ch in json.loads(line[6:]).get("choices", []):
            d = ch.get("delta", {})
            think.append(d.get("reasoning_content") or "")
            text.append(d.get("content") or "")
    res[name] = {"plain": plain, "stream": "".join(think) + "\x01" + "".join(text)}
json.dump(res, open(out, "w"))
PY
}

start_server mtp "--mtp"
run_requests mtp || { echo "FAIL: requests (mtp)"; exit 1; }
stop_server
start_server serial "--no-mtp"
run_requests serial || { echo "FAIL: requests (serial)"; exit 1; }
stop_server

python3 - "$ARTIFACTS" "$LAMBDA" <<'PY'
import json, re, sys

art, lam = sys.argv[1], sys.argv[2]
fails = []
mtp = json.load(open(f"{art}/mtp.json"))
serial = json.load(open(f"{art}/serial.json"))
for arm, res in (("mtp", mtp), ("serial", serial)):
    log = open(f"{art}/server-{arm}.log").read()
    if not re.search(r"\[think-penalty\] lambda %s \(--think-penalty\)" % re.escape(lam), log):
        fails.append(f"{arm}: no load line for lambda {lam}")
    armed = len(re.findall(r"think penalty [0-9.]+: markers lowered", log))
    if armed != 4:
        fails.append(f"{arm}: {armed} armed requests, want 4 (think_penalty 0 arms none)")
    for name, r in res.items():
        if r["plain"] != r["stream"]:
            fails.append(f"{arm}/{name}: stream != non-stream")
for name in mtp:
    if mtp[name]["plain"] != serial[name]["plain"]:
        a, b = mtp[name]["plain"], serial[name]["plain"]
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        fails.append(f"{name}: MTP != --no-mtp at char {i}: {a[max(0, i - 40):i + 40]!r} vs {b[max(0, i - 40):i + 40]!r}")
if "\x01" not in mtp["closes"]["plain"] or not mtp["closes"]["plain"].split("\x01", 1)[1].strip():
    print("note: the low-effort thought did not close inside its cap; the after-span gate went unexercised")
if not re.search(r"\[spec-stats\] mode=mtp", open(f"{art}/server-mtp.log").read()):
    fails.append("mtp: no [spec-stats] mode=mtp line")
for f in fails:
    print("FAIL:", f)
print("PASS" if not fails else f"{len(fails)} failure(s); artifacts in {art}")
sys.exit(1 if fails else 0)
PY
