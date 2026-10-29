#!/bin/bash
# sushi-test: tier=live boot=own pack=default boots=1
# Issue #390: a 1-token-tail commit that inherits checkpoints from a shared
# prefix must not poison later restores with QsaHistoryGap.
#
# Two conversations share one ~15k-token prefix. Conversation A grows past
# the prefix (donor bank sits above the shared position). Conversation B
# restores, prefills a 1-token tail at an odd position, commits (cps=null
# shape), then restores again. Every turn must 200; the log must not contain
# `prefill failed for slot: QsaHistoryGap`. The self-heal line is allowed
# and counted as WARN. After inherit, one request is posted TWICE verbatim so
# the 1-token tail / full-reuse shape actually occurs; the script fails if it
# never did.
#
# The repeated verbatim request restores from the entry the first one committed: its answer must equal the
# first one's, so a poisoned restore shows as a changed answer as well as the log line. (The old cold-boot
# replay needed its own boot; every restore shape is exercised here.)

set -e
. "$(dirname "$0")/../lib/boot.sh"
MODEL=$(sushi_pack qwen4) || exit $?
# The 1-token tail needs a hot cache of eight entries; the ram boot's four evict the donor first.
sushi_boot inherit-qsa "$MODEL" --prefix-cache-entries 8 --prefix-cache-mem 2048MB --prefix-cache-disk off --log-level debug || exit $?
BASE="$SUSHI_URL"
export SUSHI_NONCE=inh
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

SYSTEM=$(python3 -c "import sys; print(sys.argv[1] + ' You are a careful assistant for the Orion project. Rule %d: answer briefly. ' * 900 % tuple(range(900)))" "$SUSHI_NONCE")

chat() {
    local hist_json="$1"
    local user="$2"
    python3 -c "
import json,sys,urllib.request
hist=json.loads(sys.argv[1]); hist.append({'role':'user','content':sys.argv[2]})
body=json.dumps({'messages':[{'role':'system','content':sys.argv[3]}]+hist,'max_tokens':24,'temperature':0})
req=urllib.request.Request('$BASE/v1/chat/completions', data=body.encode(), headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=600) as r:
    raw=r.read(); code=r.status
obj=json.loads(raw)
reply=obj['choices'][0]['message']['content']
hist.append({'role':'assistant','content':reply})
print(json.dumps({'code':code,'hist':hist}))
" "$hist_json" "$user" "$SYSTEM"
}

fail=0
A='[]'
B='[]'
turn() {
    local which="$1" user="$2"
    local out
    if [ "$which" = A ]; then
        out=$(chat "$A" "$user") || { echo -e "${RED}FAIL${NC} A request failed"; fail=1; return; }
        A=$(echo "$out" | python3 -c "import json,sys; o=json.load(sys.stdin); print(json.dumps(o['hist'])); raise SystemExit(0 if o['code']==200 else 1)") || { echo -e "${RED}FAIL${NC} A not 200"; fail=1; }
    else
        out=$(chat "$B" "$user") || { echo -e "${RED}FAIL${NC} B request failed"; fail=1; return; }
        B=$(echo "$out" | python3 -c "import json,sys; o=json.load(sys.stdin); print(json.dumps(o['hist'])); raise SystemExit(0 if o['code']==200 else 1)") || { echo -e "${RED}FAIL${NC} B not 200"; fail=1; }
    fi
}

turn A "Hello"
turn A "Say one word."
turn B "Hello"
turn B "Hi"

BODY=$(python3 -c "
import json,sys
hist=json.loads(sys.argv[1])
hist.append({'role':'user','content':sys.argv[2]})
print(json.dumps({'messages':[{'role':'system','content':sys.argv[3]}]+hist,'max_tokens':24,'temperature':0}))
" "$B" "Ping." "$SYSTEM")

post_body() {
    python3 -c "
import json,sys,urllib.request
body=sys.argv[1].encode()
base=sys.argv[2]
req=urllib.request.Request(base+'/v1/chat/completions', data=body, headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=600) as r:
    raw=r.read(); code=r.status
obj=json.loads(raw)
print(json.dumps({'code':code,'text':obj['choices'][0]['message']['content']}))
" "$1" "$2"
}

WARM1=$(post_body "$BODY" "$BASE") || { echo -e "${RED}FAIL${NC} verbatim request 1 failed"; fail=1; WARM1='{"code":0,"text":""}'; }
WARM2=$(post_body "$BODY" "$BASE") || { echo -e "${RED}FAIL${NC} verbatim request 2 failed"; fail=1; WARM2='{"code":0,"text":""}'; }
echo "$WARM1" | python3 -c "import json,sys; o=json.load(sys.stdin); raise SystemExit(0 if o['code']==200 else 1)" || fail=1
echo "$WARM2" | python3 -c "import json,sys; o=json.load(sys.stdin); raise SystemExit(0 if o['code']==200 else 1)" || fail=1

LOGS=$(sushi_log_since)
# The restore lands on a checkpoint below the prompt end, so the old one-token tail no longer occurs; what the
# class needs is every turn after the first restoring an entry (B's two turns and the verbatim pair).
RESTORES=$(grep -c "reused [0-9]*/[0-9]* tokens" <<< "$LOGS" || true)
if [ "$RESTORES" -ge 4 ]; then
    echo -e "${GREEN}PASS${NC} $RESTORES restores from shared-prefix entries"
else
    echo -e "${RED}FAIL${NC} expected at least 4 hot-cache restores, saw $RESTORES"
    grep -n "hot-cache" <<< "$LOGS" | head -20
    fail=1
fi

if grep -q "prefill failed for slot: QsaHistoryGap" <<< "$LOGS"; then
    echo -e "${RED}FAIL${NC} log contains prefill failed for slot: QsaHistoryGap"
    grep -n "QsaHistoryGap" <<< "$LOGS" | head -10
    fail=1
else
    echo -e "${GREEN}PASS${NC} no QsaHistoryGap prefill failure"
fi

heal=$(grep -c "restored entry failed the QSA history check — dropped, cold prefill" <<< "$LOGS" || true)
if [ "$heal" -gt 0 ]; then
    echo -e "${YELLOW}WARN${NC} self-heal events: $heal"
else
    echo "  self-heal events: 0"
fi

if [ "$(echo "$WARM1" | python3 -c "import json,sys; print(json.load(sys.stdin)['text'])")" = "$(echo "$WARM2" | python3 -c "import json,sys; print(json.load(sys.stdin)['text'])")" ]; then
    echo -e "${GREEN}PASS${NC} the full-reuse answer equals the first restored answer"
else
    echo -e "${RED}FAIL${NC} the verbatim repeat answered differently from its first restore"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo -e "${GREEN}PASS${NC} test_prefix_inherit_qsa"
    exit 0
fi
exit 1
