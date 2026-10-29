#!/bin/bash
# sushi-test: tier=live boot=own pack=default boots=1
# An MLX error costs ONE REQUEST, never the server. SUSHI_MLX_FAULT_CHUNK / _STEP latch a synthetic Metal OOM at the
# n-th prefill / decode checkpoint of the process and disarm, so one boot sees both failures in order:
#   [1] a long prompt takes the prefill fault: 503 with a body, `[mlx]` in the log
#   [2] a streaming request takes the decode fault: the mapped SSE error event, never a swallowed 200
#   [3] the next request, and the one after it, answer 200 (the faults are one-shot)
#   [4] a long prompt with no max_tokens is admitted
set -u
. "$(dirname "$0")/lib/boot.sh"
PASS=0
FAIL=0
ok() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

PACK=$(sushi_pack qwen4) || exit $?
BODY=$(mktemp)
trap 'rm -f "$BODY"' EXIT
# Chunk 2, not 1, so real forward work precedes the fault.
SUSHI_MLX_FAULT_CHUNK=2 SUSHI_MLX_FAULT_STEP=2 sushi_boot mlxerr "$PACK" --log-level info || exit $?

req() { # prompt -> http code, body in $BODY
    curl -s -o "$BODY" -w '%{http_code}' -m 300 -H 'content-type: application/json' \
        -d "{\"model\":\"sushi\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":8,\"temperature\":0,\"stream\":false}" \
        "$SUSHI_URL/v1/chat/completions"
}

echo "[1] the prompt that hits the injected prefill error is refused, by name"
# More than one prefill chunk (the default chunk is 512 at its narrowest), so the fault lands after real forward work.
CODE=$(req "$(python3 -c "print('Explain the following list. ' + ' '.join(str(i) for i in range(4000)))")")
# 503 ONLY: the memory classification is part of what is under test, and a body is as load-bearing as the code.
case "$CODE" in
    503) ok "injected MLX OOM answered 503 ($(head -c 120 "$BODY"))" ;;
    000 | "") bad "no HTTP response: the server died" ;;
    500) bad "answered 500: the memory class was lost between the latch and the surface" ;;
    *) bad "unexpected status $CODE: $(head -c 200 "$BODY")" ;;
esac
grep -q "FAULT INJECTION armed" "$SUSHI_LOG" || bad "injector never armed: the env hook moved"
grep -q "\[mlx\] " "$SUSHI_LOG" || bad "no [mlx] line logged for the latched error"

echo "[2] an MLX error during DECODE of a streaming request sends the mapped SSE error"
curl -s -m 300 -N -H 'content-type: application/json' \
    -d '{"model":"sushi","messages":[{"role":"user","content":"Write one sentence about the sea."}],"max_tokens":32,"temperature":0,"stream":true}' \
    "$SUSHI_URL/v1/chat/completions" > "$BODY" 2>&1
if grep -q '"finish_reason":"error"' "$BODY" && grep -q 'ran out of GPU memory' "$BODY" && grep -q '"code":503' "$BODY"; then
    ok "streaming decode fault sent the mapped 503 error event"
else
    bad "streaming decode fault frame: $(tail -c 300 "$BODY")"
fi
grep -q 'Internal server error: GenerationOutOfMemory' "$BODY" && bad "the raw error NAME is what the streaming client is told"

echo "[3] the server is still serving, and the faults were one-shot"
sushi_alive || bad "server process is gone"
for p in "Say hello." "Say goodbye."; do
    CODE=$(req "$p")
    if [ "$CODE" = 200 ] && grep -q '"content"' "$BODY"; then ok "request '$p' answered 200 with content"; else bad "request '$p' status $CODE: $(head -c 200 "$BODY")"; fi
done

echo "[4] a long prompt with NO max_tokens field is admitted"
# The omitted sentinel (maxInt/4) must be clamped before the guard bills it. The prompt must pass
# KVCache.RESERVE_MIN_TOKENS (32k) and still fit the context.
CTX=$(curl -s -m 10 "$SUSHI_URL/v1/models" | python3 -c "import json,sys; d=json.load(sys.stdin); print(max([m.get('context_length',0) for m in d.get('data',[])] or [0]))" 2> /dev/null || echo 0)
if [ "${CTX:-0}" -lt 65536 ]; then
    echo "  SKIP: model context $CTX < 65536, no room for a 45k-token prompt"
else
    CODE=$(python3 - "$SUSHI_URL" "$BODY" <<'PY'
import json, sys, urllib.error, urllib.request
base, out = sys.argv[1:3]
long_text = ("The quick brown fox jumps over the lazy dog. " * 9000)[:180000]
body = {"model": "sushi", "messages": [{"role": "user", "content": long_text + "\n\nReply with one word."}],
        "temperature": 0, "stream": False}   # NO max_tokens field on purpose
req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        json.load(r); print("200")
except urllib.error.HTTPError as e:
    open(out, "w").write(e.read().decode("utf-8", "replace")); print(e.code)
except Exception as e:
    open(out, "w").write(f"{type(e).__name__}: {e}"); print("000")
PY
)
    if [ "$CODE" = 200 ]; then ok "omitted max_tokens on a long prompt answered 200"; else bad "omitted max_tokens refused with $CODE: $(head -c 240 "$BODY")"; fi
fi

echo "---- $PASS passed, $FAIL failed ----"
[ "$FAIL" -eq 0 ]
