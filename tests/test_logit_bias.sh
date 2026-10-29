#!/bin/bash
# sushi-test: tier=live boot=knobs
# Request biases and the model-settings `logit_bias_file` (tests/lib/knobs-bias.json, named by the knobs boot's
# settings file): rewards, penalties, named 400 validation, stream == non-stream, and strict MTP == serial bytes
# (the two arms are `enable_mtp` true and false on the one boot).
set -euo pipefail
. "$(dirname "$0")/lib/boot.sh"
sushi_attach
sushi_require mtp
ARTIFACTS=$(mktemp -d)
trap 'rm -rf "$ARTIFACTS"' EXIT

run_requests() { # $1 = arm name, $2 = enable_mtp (True|False)
    python3 - "$SUSHI_URL" "$ARTIFACTS" "$1" "$2" "$SUSHI_NONCE" <<'PY'
import json, pathlib, sys, urllib.error, urllib.request
base, folder, arm, mtp, nonce = sys.argv[1:]
mtp = mtp == "True"
out = pathlib.Path(folder)
results = {}
def post(endpoint, body):
    req = urllib.request.Request(f"{base}/v1/{endpoint}", json.dumps(body).encode(), {"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.read().decode()
# The id of the token "A": the bias file rewards the token spelled A, the request maps below name its id.
tok = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/tokenize", json.dumps({"content": "A"}).encode(), {"Content-Type": "application/json"}), timeout=60).read())["tokens"]
assert len(tok) == 1, tok
target = {"id": tok[0], "text": "A"}
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
    ("reasoning", "chat/completions", {"messages":[{"role":"user","content":nonce + " How many positive integers n below 100 make n*n+n+41 composite? Check your work."}], "reasoning_effort":"xhigh", "max_tokens":256}),
    ("answer", "chat/completions", {"messages":[{"role":"user","content":nonce + " A bat and ball cost $1.10 total. The bat costs $1 more. What is the ball's price?"}], "reasoning_effort":"low", "max_tokens":512}),
    ("preset-off", "chat/completions", {"messages":[{"role":"user","content":nonce + " Name three prime numbers above 50."}], "reasoning_effort":"low", "think_penalty":0,"logit_bias":{},"max_tokens":256}),
    ("chat-reward", "chat/completions", {"messages":[{"role":"user","content":nonce + " Answer with one letter."}],"chat_template_kwargs":{"enable_thinking":False},"logit_bias":{str(target["id"]):100},"max_tokens":1}),
    ("reward", "completions", {"prompt":nonce + " Answer with one letter: ","logit_bias":{str(target["id"]):100},"max_tokens":1}),
    ("penalty", "completions", {"prompt":nonce + " Answer with one letter: ","logit_bias":{str(target["id"]):-100},"max_tokens":1}),
]
for name, endpoint, params in cases:
    body = dict(params, model="default", temperature=0, enable_mtp=mtp)
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
    base_body = {"messages":[{"role":"user","content":nonce + " Hi"}]} if endpoint.startswith("chat") else {"prompt":nonce + " Hi"}
    for bias in ({"-1":1}, {str(target["id"]):101}, {"not-an-id":1}):
        try:
            post(endpoint, dict(base_body, model="default",max_tokens=1,logit_bias=bias))
        except urllib.error.HTTPError as e:
            assert e.code == 400, (endpoint,e.code,e.read())
        else:
            raise AssertionError(f"{endpoint}: invalid map accepted: {bias}")
(out / f"{arm}.json").write_text(json.dumps(results))
PY
}

for arm in mtp serial; do
    before=$(sushi_log_since | wc -c)
    if [ "$arm" = mtp ]; then enable=True; else enable=False; fi
    run_requests "$arm" "$enable"
    sushi_log_since | tail -c +"$((before + 1))" > "$ARTIFACTS/server-$arm.log"
    if [ "$arm" = mtp ]; then grep -q '\[spec-stats\] mode=mtp' "$ARTIFACTS/server-$arm.log"; fi
    echo "PASS: $arm"
done
# The bias file loads once at boot, before this script attached.
grep -q '\[logit-bias\].*entries.*expanded ids' "$SUSHI_LOG"
python3 - "$ARTIFACTS" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
a, b = (json.loads((out / f"{arm}.json").read_text()) for arm in ("mtp", "serial"))
assert a == b, "MTP != serial bytes"
print("PASS: strict MTP == serial and stream == non-stream")
PY
