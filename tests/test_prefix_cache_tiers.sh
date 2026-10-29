#!/bin/bash
# sushi-test: tier=live boot=own pack=glm5 boots=2
# The SSD prefix tier gets the whole entry on the turn that commits it, whatever the RAM tier keeps
# (docs/engine-prefix-cache.md#ssd-flush). Two boots on the GLM pack, each on a private cache root:
#   hybrid: a RAM tier smaller than the prompt (`--prefix-cache-mem $TIERS_MEM`) plus the SSD tier
#   ssd:    `--prefix-cache-mem off` plus the SSD tier
# Per arm:
#  1. Every tier writes through the background writer.
#  2. Turn 1 is whole on disk before the next request: `e<id> complete on disk: N tokens`, N within
#     the prompt-end backoff of the prompt.
#  3. Its identical re-issue reuses that whole prefix. On GLM the RAM tier cannot hold it, so it
#     comes from SSD, and the answer is turn 1's byte for byte (a prompt-end restore is exact).
#  4. An appended turn reuses at least turn 1's prompt.
# The restart round trip lives in test_prefix_cache_restart.sh.
#
# Env: TIERS_MEM (default 200MB), TIERS_WORDS (default 9000), TIERS_ARMS (default "hybrid ssd").

set -uo pipefail
. "$(dirname "$0")/lib/boot.sh"
command -v jq >/dev/null || { echo "needs jq"; exit 1; }
MODEL=$(sushi_pack glm5) || exit $?
MEM="${TIERS_MEM:-200MB}"
WORDS="${TIERS_WORDS:-9000}"
# The prompt-end checkpoint backs off up to 33 tokens (GLM) from the prompt's end.
SLACK=64

boot() { # arm, flags...
    local arm="$1"; shift
    sushi_boot "tiers-$arm" "$MODEL" --log-level info "$@" || exit $?
    BASE="$SUSHI_URL"
    LOG="$SUSHI_LOG"
}

words() { python3 -c "import random,sys; r=random.Random($1); w='agent tool file cache ring window layer prompt reply token session restore commit budget chunk kernel latent pool state'.split(); print(' '.join(r.choice(w) for _ in range($2)))"; }
NOTES="You are a careful engineer. Notes: $(words 7 "$WORDS")"
Q1="Summarise the notes in one sentence."
REPLY="The notes repeat a handful of systems words in random order."
Q2="Now name the word they repeat most."

chat() { # q, reply, q2
    jq -nc --arg n "$NOTES" --arg q "$1" --arg r "$2" --arg q2 "$3" '
        {messages:([{role:"system",content:$n},{role:"user",content:$q}]
            + (if $r == "" then [] else [{role:"assistant",content:$r},{role:"user",content:$q2}] end)),
         max_tokens:32,temperature:0,stream:false,reasoning_effort:"low"}'
}
ask() { curl -sf --max-time 1800 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$1"; }
text() { echo "$1" | jq -r '(.choices[0].message.reasoning_content // "") + "\u0001" + (.choices[0].message.content // "")'; }
field() { echo "$1" | jq -r "$2"; }
need() { [ -n "$1" ] || { echo "fail: $2"; tail -30 "$LOG"; exit 1; }; }

EC=0
fail() { echo "FAIL [$ARM]: $*"; EC=1; }

# The flush after turn 1's response: the largest `complete on disk` count within 60 s.
whole_on_disk() {
    local best=0 n
    for _ in $(seq 1 60); do
        n=$(grep -oE 'complete on disk: [0-9]+ tokens' "$LOG" | grep -oE '[0-9]+' | sort -n | tail -1)
        [ -n "$n" ] && best=$n
        [ "$best" -ge "$1" ] && break
        sleep 1
    done
    echo "$best"
}

for ARM in ${TIERS_ARMS:-hybrid ssd}; do
    if [ $ARM = hybrid ]; then FLAGS=(--prefix-cache-mem "$MEM" --prefix-cache-disk 20GB); else FLAGS=(--prefix-cache-mem off --prefix-cache-disk 20GB); fi
    boot "$ARM" "${FLAGS[@]}"
    GLM=0; grep -q '\[glm\] prefix cache' "$LOG" && GLM=1
    grep -q 'background writer armed' "$LOG" || fail "no background writer"

    T1=$(ask "$(chat "$Q1" "" "")"); need "$T1" "turn 1"
    P1=$(field "$T1" .timings.prompt_n)
    [ "$(field "$T1" .timings.cached_n)" = 0 ] || fail "turn 1 restored a prefix on an empty cache"
    ON_DISK=$(whole_on_disk $((P1 - SLACK)))
    echo "[$ARM] turn 1: prompt $P1, whole on disk $ON_DISK tokens"
    [ "$ON_DISK" -ge $((P1 - SLACK)) ] || fail "turn 1 is not whole on disk ($ON_DISK of $P1)"

    T2=$(ask "$(chat "$Q1" "" "")"); need "$T2" "turn 1 again"
    C2=$(field "$T2" .timings.cached_n)
    echo "[$ARM] identical re-issue: cached $C2 of $(field "$T2" .timings.prompt_n)"
    [ "$C2" -ge $((P1 - SLACK)) ] || fail "the re-issue reused $C2 of $P1"
    if [ $GLM = 1 ]; then
        grep -q '\[disk-cache\] restored' "$LOG" || fail "the re-issue did not restore from SSD"
        [ "$(text "$T1")" = "$(text "$T2")" ] || fail "the SSD restore at the prompt end answered differently from cold"
    fi

    T3=$(ask "$(chat "$Q1" "$REPLY" "$Q2")"); need "$T3" "appended turn"
    P3=$(field "$T3" .timings.prompt_n)
    echo "[$ARM] appended turn: cached $(field "$T3" .timings.cached_n) of $P3"
    [ "$(field "$T3" .timings.cached_n)" -ge $((P1 - SLACK)) ] || fail "the appended turn did not reuse turn 1"
    whole_on_disk $((P3 - SLACK)) > /dev/null
    sushi_stop 0
done

[ $EC = 0 ] && echo "PASS"
exit $EC
