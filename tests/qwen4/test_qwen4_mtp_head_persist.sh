#!/usr/bin/env bash
# sushi-test: tier=live boot=api
# qwen4_exp: the in-checkpoint MTP head's committed history rides the prefix cache. The head is not KV-only
# (it owns a QSA key history), so a cache hit used to draft from an empty head. Asserts the invariant: the
# identical second turn restores from the cache AND the head is restored, the restore is never declined, and
# the needle is still answered.
set -u
. "$(dirname "$0")/../lib/boot.sh"
sushi_attach
sushi_require mtp
U="$SUSHI_URL"
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

# One long prompt with a needle, well past the QSA budget so the head's
# history is the thing under test and not an incidental short window.
body() { python3 -c "
import json,sys
filler=('The archivist catalogued the shelves in the long hall. ')*700
print(json.dumps({'messages':[{'role':'user','content':sys.argv[1]+' '+filler+' The secret code is PELICAN-42. '+filler+' What is the secret code? Answer with the code only.'}],'max_tokens':24,'temperature':0,'enable_thinking':False,'enable_mtp':True}))" "$SUSHI_NONCE"; }

b=$(body)
# Turn 1 cold-prefills and COMMITS the head; turn 2 is the cache hit.
curl -s -m 1800 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$b" >/dev/null
sleep 3 # the api boot keeps no RAM tier: the entry reaches the SSD tier in the background
curl -s -m 1800 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$b" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" > "$T/answer"
LOGS=$(sushi_log_since)

echo "[1] the second turn restores the head"
check "turn 2 restored from the cache" "$(grep -cE '\[hot-cache\] (reused|full reuse)|\[disk-cache\] restored' <<< "$LOGS" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "MTP head restored" "$(grep -c '\[qwen4\] MTP head restored' <<< "$LOGS" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "head restore never declined" "$(grep -c '\[qwen4\] MTP head restore declined' <<< "$LOGS")" "0"
check "needle recovered on the restored turn" "$(grep -c 'PELICAN-42' "$T/answer")" "1"

echo "pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
