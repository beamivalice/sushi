#!/usr/bin/env bash
# sushi-test: tier=live boot=exact
# The exact boot is the disabled arm: the prefix cache reports zero RAM, the server still answers, an identical
# chat request three times never reports a cached token, and the log carries no `[hot-cache]` line.
set -euo pipefail
. "$(dirname "$0")/lib/boot.sh"
sushi_attach
python3 - "$SUSHI_URL" "$SUSHI_NONCE" <<'PY'
import json
import sys
import urllib.request

base, nonce = sys.argv[1:]


def post(path, body):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as r:
        return json.load(r)


for _ in range(3):
    with urllib.request.urlopen(base + "/props", timeout=5) as r:
        props = json.load(r)
    assert props["settings"]["prefix_cache"]["mem_bytes"] == 0, props
    response = post("/v1/completions", {"model": "default", "prompt": nonce + " Hello", "max_tokens": 2})
    assert response.get("choices"), response
    chat = post("/v1/chat/completions", {"model": "default", "temperature": 0, "max_tokens": 1,
                                         "messages": [{"role": "user", "content": nonce + " Name a colour."}]})
    cached = chat["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached == 0, chat["usage"]
print("PASS: disabled prefix cache reports zero RAM, serves three requests and reports no cached token")
PY
LOGS=$(sushi_log_since)
if grep -q '\[hot-cache\]' <<< "$LOGS"; then
    echo "FAIL: the disabled cache logged a [hot-cache] line"
    exit 1
fi
echo "PASS: no [hot-cache] line"
