#!/usr/bin/env bash
# sushi-test: tier=live boot=exact
# The exact boot is the disabled arm: the prefix cache reports zero RAM and the server still answers.
set -euo pipefail
. "$(dirname "$0")/lib/boot.sh"
sushi_attach
python3 - "$SUSHI_URL" "$SUSHI_NONCE" <<'PY'
import json
import sys
import urllib.request

base, nonce = sys.argv[1:]
for _ in range(3):
    with urllib.request.urlopen(base + "/props", timeout=5) as r:
        props = json.load(r)
    assert props["settings"]["prefix_cache"]["mem_bytes"] == 0, props
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps({"model": "default", "prompt": nonce + " Hello", "max_tokens": 2}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as r:
        response = json.load(r)
    assert response.get("choices"), response
print("PASS: disabled prefix cache reports zero RAM and serves three requests")
PY
