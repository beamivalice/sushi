#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
set -euo pipefail
MODEL="${LOGIT_BIAS_TEST_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw}"
PORT="${1:-11318}"
MODE="${2:-both}"
BIN="${SUSHI_BINARY:-./zig-out/bin/sushi}"
ARTIFACTS="${LOGIT_BIAS_TEST_OUTPUT_DIR:-$(mktemp -d)}"
FORMAT="${LOGIT_BIAS_FORMAT:-json}"
mkdir -p "$ARTIFACTS"
SERVER_PID=""
trap 'if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi' EXIT
if [ "$MODE" != compare ]; then
    test -d "$MODEL" || { echo "SKIP: model not found: $MODEL"; exit 0; }
    python3 - "$MODEL" "$ARTIFACTS" <<'PY'
import csv, json, pathlib, sys
model, out = map(pathlib.Path, sys.argv[1:])
vocab = json.loads((model / "tokenizer.json").read_text())["model"]["vocab"]
target = vocab["A"]
entries = [{"word":"Wait","delta":-1,"scope":"reasoning"},
           {"word":"actually","delta":-0.5,"scope":"answer"},
           {"id":target,"delta":0.25,"scope":"all"}]
(out / "bias.json").write_text(json.dumps({"entries":entries}))
(out / "empty.json").write_text(json.dumps({"entries":[]}))
(out / "target.json").write_text(json.dumps({"id":target,"text":"A"}))
with (out / "bias.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kind","target","delta","scope"])
    for e in entries:
        kind = next(k for k in ("word","token","id") if k in e)
        w.writerow([kind,e[kind],e["delta"],e["scope"]])
PY
fi
start_server() {
    local arm="$1"
    local flag=--mtp
    [ "$arm" = serial ] && flag=--no-mtp
    local bias_flags=(--think-penalty 1) bias_file="$ARTIFACTS/bias.$FORMAT"
    [ "${LOGIT_BIAS_DISABLED:-0}" = 1 ] && bias_flags=(--think-penalty 0) && bias_file="$ARTIFACTS/empty.json"
    # The bias file is the model's `logit_bias_file` setting, read from a private HOME.
    mkdir -p "$ARTIFACTS/home/.sushi"
    python3 -c 'import json, sys; print(json.dumps({sys.argv[1]: {"logit_bias_file": sys.argv[2]}}))' "$MODEL" "$bias_file" \
        >"$ARTIFACTS/home/.sushi/model-settings.json"
    HOME="$ARTIFACTS/home" "$BIN" --model "$MODEL" --serve --port "$PORT" --no-drafter \
        --prefix-cache-entries 0 --ctx-size 16384 --log-level info "$flag" "${bias_flags[@]}" \
        >"$ARTIFACTS/server-$arm.log" 2>&1 &
    SERVER_PID=$!
    for ((i=0; i<400; i++)); do
        if grep -q 'Model ready (loaded on inference thread)' "$ARTIFACTS/server-$arm.log"; then return; fi
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 3
    done
    tail -30 "$ARTIFACTS/server-$arm.log"
    echo "FAIL: server did not become ready ($arm)"
    exit 1
}
run_requests() {
    python3 - "$PORT" "$ARTIFACTS" "$1" <<'PY'
import json, pathlib, sys, urllib.error, urllib.request
port, folder, arm = sys.argv[1:]
out = pathlib.Path(folder)
target = json.loads((out / "target.json").read_text())
results = {}
def post(endpoint, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/{endpoint}", json.dumps(body).encode(), {"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.read().decode()
def text(raw, endpoint, streaming):
    if not streaming:
        c = json.loads(raw)["choices"][0]
        if endpoint == "completions": return c["text"]
        m = c["message"]
        return (m.get("reasoning_content") or "") + "\x01" + (m.get("content") or "")
    reason, answer = [], []
    for line in raw.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        event = json.loads(line[6:])
        assert "error" not in event, event
        for c in event.get("choices", []):
            if endpoint == "completions": answer.append(c.get("text") or "")
            else:
                d = c.get("delta", {})
                reason.append(d.get("reasoning_content") or "")
                answer.append(d.get("content") or "")
    return ("".join(reason) + "\x01" if endpoint != "completions" else "") + "".join(answer)
cases = [
    ("reasoning", "chat/completions", {"messages":[{"role":"user","content":"How many positive integers n below 100 make n*n+n+41 composite? Check your work."}], "reasoning_effort":"xhigh", "max_tokens":256}),
    ("answer", "chat/completions", {"messages":[{"role":"user","content":"A bat and ball cost $1.10 total. The bat costs $1 more. What is the ball's price?"}], "reasoning_effort":"low", "max_tokens":512}),
    ("preset-off", "chat/completions", {"messages":[{"role":"user","content":"Name three prime numbers above 50."}], "reasoning_effort":"low", "think_penalty":0,"logit_bias":{},"max_tokens":256}),
    ("chat-reward", "chat/completions", {"messages":[{"role":"user","content":"Answer with one letter."}],"chat_template_kwargs":{"enable_thinking":False},"logit_bias":{str(target["id"]):100},"max_tokens":1}),
    ("reward", "completions", {"prompt":"Answer with one letter: ","logit_bias":{str(target["id"]):100},"max_tokens":1}),
    ("penalty", "completions", {"prompt":"Answer with one letter: ","logit_bias":{str(target["id"]):-100},"max_tokens":1}),
]
for name, endpoint, params in cases:
    body = dict(params, model="default", temperature=0)
    pair = {}
    for streaming in (False, True):
        raw = post(endpoint, dict(body, stream=streaming))
        suffix = "stream" if streaming else "plain"
        (out / f"{arm}-{name}-{suffix}.txt").write_text(raw)
        pair[suffix] = text(raw, endpoint, streaming)
    assert pair["plain"] == pair["stream"], f"{arm}/{name}: stream != non-stream"
    if name == "reward": assert pair["plain"] == target["text"], pair
    if name == "chat-reward": assert pair["plain"] == "\x01" + target["text"], pair
    if name == "penalty": assert pair["plain"] != target["text"], pair
    results[name] = pair
for endpoint in ("chat/completions","completions"):
    base = {"messages":[{"role":"user","content":"Hi"}]} if endpoint.startswith("chat") else {"prompt":"Hi"}
    for bias in ({"-1":1}, {str(target["id"]):101}, {"not-an-id":1}):
        try:
            post(endpoint, dict(base, model="default",max_tokens=1,logit_bias=bias))
        except urllib.error.HTTPError as e:
            assert e.code == 400, (endpoint,e.code,e.read())
        else:
            raise AssertionError(f"{endpoint}: invalid map accepted: {bias}")
(out / f"{arm}.json").write_text(json.dumps(results))
PY
}
if [ "$MODE" = both ]; then arms=(mtp serial); elif [ "$MODE" = mtp ] || [ "$MODE" = serial ]; then arms=("$MODE"); elif [ "$MODE" = compare ]; then arms=(); else echo 'mode must be mtp, serial, both, or compare'; exit 2; fi
if [ "$MODE" != compare ]; then
for arm in "${arms[@]}"; do
    start_server "$arm"
    run_requests "$arm"
    kill "$SERVER_PID"
    wait "$SERVER_PID" || true
    SERVER_PID=""
    if [ "${LOGIT_BIAS_DISABLED:-0}" != 1 ]; then grep -q '\[logit-bias\].*entries.*expanded ids' "$ARTIFACTS/server-$arm.log"; fi
    if [ "$arm" = mtp ]; then grep -q '\[spec-stats\] mode=mtp' "$ARTIFACTS/server-$arm.log"; fi
    echo "PASS: $arm; artifacts $ARTIFACTS"
done
fi
if [ "$MODE" = both ] || [ "$MODE" = compare ]; then
    python3 - "$ARTIFACTS" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
a, b = (json.loads((out / f"{arm}.json").read_text()) for arm in ("mtp", "serial"))
assert a == b, "MTP != serial bytes"
print("PASS: strict MTP == serial and stream == non-stream")
PY
fi
