#!/bin/bash
# sushi-test: tier=live boot=ram
# Workload-fair hot-cache eviction (issue #378).
#
# One model, two workloads sharing a 4-entry hot prefix cache: a conversation
# under a system prompt, and a sweep of six distinct documents. Plain LRU let
# the sweep evict the conversation every time. Eviction now picks the LRU of
# the workload key holding the most entries (`server.requestCacheKey`:
# prompt_cache_key > metadata.user_id > system-prompt hash > anonymous), so the
# sweep only evicts its own documents.
#
#   [1] conversation turn 2 is warm (cached_tokens > 1000: the system prompt, not a 2-token BOS match)
#   [2] six docs with prompt_cache_key:"batch"; turn 3 is still warm
#   [3] no eviction during that sweep names the anonymous key (one or two keys: the batch, and the conversation's stale entry)
#   [4] control: six unkeyed, system-less docs (anonymous group); turn 4 still warm
#   [5] the control sweep evicts key-0 entries (it pays for itself)
# A turn may evict the conversation's own stale earlier entry, and a tie between
# groups falls back to plain LRU: both are the policy, so only the NEWEST
# conversation entry ([2], [4]) is the bar.
#   [6] hot cache: two conversations alternate (A1 B1 A2 B2, multi-turn) on the ram boot's four entries; the
#       follow-ups reuse (`[hot-cache] reused|full reuse`), at least one reuse sees two or more entries at once,
#       and every response carries `usage.prompt_tokens_details.cached_tokens` (<= prompt, >= 100 when warm)
#
# The ram boot's cache holds other scripts' entries, so a keyed pre-sweep fills it first; the group-fair
# policy then evicts those leftovers before any of this script's entries.

set -e
. "$(dirname "$0")/lib/boot.sh"
sushi_attach
BASE="$SUSHI_URL"
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# The slice since $1 bytes into this script's log.
log_after() { sushi_log_since | tail -c +"$(($1 + 1))"; }
log_size() { sushi_log_since | wc -c; }

# ~2k tokens of system prompt so the conversation is a real entry.
SYSTEM=$(python3 -c "print('You are a careful assistant for the Orion project. Rule %d: answer briefly. ' * 220 % tuple(range(220)))")
SYSTEM="$SUSHI_NONCE $SYSTEM"
HISTORY="[]"

# $1 = user text; appends the turn to HISTORY, prints cached_tokens.
conv_turn() {
    local body resp reply
    body=$(python3 -c "
import json,sys
hist=json.loads(sys.argv[1]); hist.append({'role':'user','content':sys.argv[2]})
print(json.dumps({'messages':[{'role':'system','content':sys.argv[3]}]+hist,'max_tokens':12,'temperature':0}))
" "$HISTORY" "$1" "$SYSTEM")
    resp=$(curl -s -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$body")
    reply=$(echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
    HISTORY=$(python3 -c "
import json,sys
hist=json.loads(sys.argv[1]); hist.append({'role':'user','content':sys.argv[2]}); hist.append({'role':'assistant','content':sys.argv[3]})
print(json.dumps(hist))" "$HISTORY" "$1" "$reply")
    echo "$resp" | python3 -c "import json,sys; print(json.load(sys.stdin)['usage']['prompt_tokens_details']['cached_tokens'])"
}

# $1 = doc index, $2 = extra JSON fields (e.g. '"prompt_cache_key":"batch",')
sweep_doc() {
    local doc
    doc=$(python3 -c "print(('Document $1 paragraph %d about topic $1. ' * 8) % tuple(range(8)) * 20)")
    python3 -c "
import json,sys
print(json.dumps({'messages':[{'role':'user','content':'$SUSHI_NONCE Summarize in three words: '+sys.argv[1]}],'max_tokens':8,'temperature':0}))" "$doc" \
      | sed "s/^{/{$2/" \
      | curl -s -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @- > /dev/null
}

# $1 = log size at the start of the phase.
evicted_keys_since() {
    log_after "$1" | grep -o 'evicted LRU entry ([^;]*; key=[0-9a-f]*' | sed 's/.*key=//' | sort -u
}

FAIL=0
# Leave only keyed entries of our own behind: six keyed docs push every other script's entries out.
for i in 101 102 103 104 105 106; do sweep_doc "$i" '"prompt_cache_key":"batch",'; done
c1=$(conv_turn "Say hello.")
c2=$(conv_turn "What is rule 3?")
if [ "$c2" -gt 1000 ]; then echo -e "${GREEN}PASS${NC} [1] turn 2 warm (cached_tokens=$c2)"; else echo -e "${RED}FAIL${NC} [1] turn 2 cold (cached_tokens=$c2)"; FAIL=1; fi

mark=$(log_size)
for i in 1 2 3 4 5 6; do sweep_doc "$i" '"prompt_cache_key":"batch",'; done
keys=$(evicted_keys_since "$mark")
c3=$(conv_turn "And rule 7?")
if [ "$c3" -gt 1000 ]; then echo -e "${GREEN}PASS${NC} [2] turn 3 warm after a keyed sweep (cached_tokens=$c3)"; else echo -e "${RED}FAIL${NC} [2] keyed sweep evicted the conversation (cached_tokens=$c3)"; FAIL=1; fi

nkeys=$(echo "$keys" | grep -c . || true)
# The conversation's own stale entry may also go (header note), so one or two keys, never the anonymous one.
if [ "$nkeys" -ge 1 ] && [ "$nkeys" -le 2 ] && ! echo "$keys" | grep -qx 0; then
    echo -e "${GREEN}PASS${NC} [3] no eviction names the anonymous key ($(echo $keys))"
else
    echo -e "${RED}FAIL${NC} [3] expected one non-anonymous evicted key, got: $(echo $keys)"; log_after "$mark" | grep 'evicted LRU'; FAIL=1
fi

mark=$(log_size)
for i in 7 8 9 10 11 12; do sweep_doc "$i" ''; done
anon_evicted=$(evicted_keys_since "$mark" | grep -c -x 0 || true)
c4=$(conv_turn "Rule 11?")
if [ "$c4" -gt 1000 ]; then echo -e "${GREEN}PASS${NC} [4] turn 4 warm after an anonymous sweep (cached_tokens=$c4)"; else echo -e "${RED}FAIL${NC} [4] anonymous sweep evicted the conversation (cached_tokens=$c4)"; FAIL=1; fi

if [ "$anon_evicted" = "1" ]; then
    echo -e "${GREEN}PASS${NC} [5] the anonymous sweep evicts its own documents"
else
    echo -e "${RED}FAIL${NC} [5] no key=0 eviction during the anonymous sweep"; log_after "$mark" | grep 'evicted LRU'; FAIL=1
fi

# [6] two alternating conversations on the hot cache.
mark=$(log_size)
SYS6="$SUSHI_NONCE You are an expert software engineer assistant. You provide concise, technically correct answers. You explain trade-offs when relevant. You always cite specific function names, file paths, or line numbers when discussing code. You prefer concrete examples over abstract advice. You do not pad your answers with hedges or apologies. You assume the user is also a software engineer. You write in Markdown when formatting helps. You keep code blocks small and self-contained. You ask clarifying questions only when truly necessary. You favor depth over breadth in your explanations. You show your reasoning when it would help the reader."
A_ASSISTANT="A mutex protects shared resources from race conditions, while a semaphore limits concurrent access to a fixed-size pool. Both use atomic operations under the hood."
B_ASSISTANT="TCP is connection-oriented and reliable, with handshakes, retransmits, and ordered delivery. UDP is connectionless and fire-and-forget, relying on the application for any reliability guarantees."
# $1 = user1, $2 = assistant1 (empty = single turn), $3 = user2; prints the usage object.
hot_turn() {
    jq -nc --arg s "$SYS6" --arg u1 "$1" --arg a1 "$2" --arg u2 "$3" \
        '{model:"x",messages:([{role:"system",content:$s},{role:"user",content:$u1}] + (if $a1 == "" then [] else [{role:"assistant",content:$a1},{role:"user",content:$u2}] end)),max_tokens:1,temperature:0.0,stream:false,enable_thinking:false}' \
        | curl -sf -m 600 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @- | jq -c '.usage'
}
A1="What is the difference between a mutex and a semaphore?"
B1="What is the difference between TCP and UDP at the protocol level?"
u_a1=$(hot_turn "$A1" "" "")
u_b1=$(hot_turn "$B1" "" "")
u_a2=$(hot_turn "$A1" "$A_ASSISTANT" "Now explain when to use each of those in practice.")
u_b2=$(hot_turn "$B1" "$B_ASSISTANT" "Now explain when each is preferred in practice.")
hot_log=$(log_after "$mark")
hits=$(echo "$hot_log" | grep -cE "\[hot-cache\] (reused|full reuse)" || true)
if [ "$hits" -ge 2 ]; then echo -e "${GREEN}PASS${NC} [6] the follow-ups reuse the hot cache ($hits reuse events)"; else echo -e "${RED}FAIL${NC} [6] expected >= 2 hot-cache reuse events, got $hits"; echo "$hot_log" | grep '\[hot-cache\]'; FAIL=1; fi
if echo "$hot_log" | grep -qE "entry [0-9]+/[2-9][0-9]*\)"; then echo -e "${GREEN}PASS${NC} [6] the cache held two or more entries at once"; else echo -e "${RED}FAIL${NC} [6] no reuse saw two entries at once"; echo "$hot_log" | grep '\[hot-cache\]'; FAIL=1; fi
ok6=1
for u in "$u_a1" "$u_b1" "$u_a2" "$u_b2"; do
    c=$(echo "$u" | jq '.prompt_tokens_details.cached_tokens'); n=$(echo "$u" | jq '.prompt_tokens')
    { [ "$c" != null ] && [ "$c" -le "$n" ]; } || ok6=0
done
for u in "$u_a2" "$u_b2"; do [ "$(echo "$u" | jq '.prompt_tokens_details.cached_tokens')" -ge 100 ] || ok6=0; done
if [ "$ok6" = 1 ]; then echo -e "${GREEN}PASS${NC} [6] cached_tokens is reported on every response and warm follow-ups report >= 100 ($u_a2 / $u_b2)"; else echo -e "${RED}FAIL${NC} [6] cached_tokens missing, above the prompt, or cold on a warm follow-up (a1=$u_a1 b1=$u_b1 a2=$u_a2 b2=$u_b2)"; FAIL=1; fi

[ "$FAIL" = "0" ] && echo -e "${GREEN}ALL PASS${NC} test_prefix_cache_workloads" || { sushi_log_since | tail -40; exit 1; }
