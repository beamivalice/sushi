#!/bin/bash
# sushi-test: tier=live boot=ram
# test_cancel_commit.sh — cancelled requests preserve their committed KV.
#
# Two contracts, one per cancellation phase:
#   [1] decode-phase cancel: a streaming request killed mid-generation
#       (client disconnect) must COMMIT prompt + emitted tokens to the hot
#       prefix cache — `complete()` pulls the slot straight into the cleanup
#       queue, so without the drain-hook commit its KV dies with the slot.
#       Proven by an identical re-issue reporting `cached_tokens` ≈ the full
#       prompt (full-match reuse) where it previously reported ~0.
# [2] prefill-phase cancel: a long prompt killed while still prefilling must
#     commit the FORWARDED PREFIX (chunk-aligned). Hybrids ride the
#     Cancelled-checkpoint salvage: `initWithOptions` hands its captured
#     SSM stride checkpoints to the slot on abort, so the commit carries
#     them and the re-issue restores at the last checkpoint ≤ matched.
#     The killed stream's captured body tells us which phase we were in:
#     no content/reasoning chunks ⇒ still prefilling (the strict
#     "committed N/M prompt tokens from a cancelled prefill" log line is
#     then REQUIRED); content chunks ⇒ prefill had finished and the
#     decode-cancel contract of [1] covers it (section skipped, not failed).
#
# Attaches to the ram boot; the prefix cache is on, so each section's prompt carries the nonce and its own text.

set -u
. "$(dirname "$0")/lib/boot.sh"
sushi_attach
BASE="$SUSHI_URL"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
PASS=0
FAIL=0
# kv4 rides the request on the archs that serve it (GLM refuses kv4, DeepSeek has no KV cache).
case "$SUSHI_ARCH" in qwen4_exp | mimo_v2) KVQ=4 ;; *) KVQ=null ;; esac
# GLM's template has no thinking-off mode.
case "$SUSHI_ARCH" in glm5_next) THINK='{"reasoning_effort":"low"}' ;; *) THINK='{"enable_thinking":false}' ;; esac

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS+1)); echo -e "${GREEN}PASS${NC}: $desc"
    else
        FAIL=$((FAIL+1)); echo -e "${RED}FAIL${NC}: $desc"
    fi
}

# Long system prompt (well past the ~40-token template prologue and the
# 256-token cancelled-prefill floor) so a cache hit is unambiguous.
SYSTEM_PROMPT="$SUSHI_NONCE You are an expert software engineer assistant. You provide concise, technically correct answers. You explain trade-offs when relevant. You always cite specific function names, file paths, or line numbers when discussing code. You prefer concrete examples over abstract advice. You do not pad your answers with hedges or apologies. You assume the user is also a software engineer. You write in Markdown when formatting helps. You keep code blocks small and self-contained. You ask clarifying questions only when truly necessary. You favor depth over breadth in your explanations. You show your reasoning when it would help the reader. You never invent APIs. You state uncertainty explicitly when you are unsure."

# A much longer one (~70k tokens) so prefill takes long enough for a
# mid-prefill kill to land inside it even on a fast box.
LONG_SYSTEM=""
for i in $(seq 1 450); do
    LONG_SYSTEM="$LONG_SYSTEM Chapter $i. $SYSTEM_PROMPT"
done

# The shared server is already loaded, but a first request may still pay one-time warmup.
warmup_model() {
    curl -sf --max-time 600 -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$SUSHI_NONCE hi\"}],\"max_tokens\":1,\"temperature\":0.0}" \
        >/dev/null 2>&1
}

# The log since $1 bytes into this script's slice.
log_after() { sushi_log_since | tail -c +"$(($1 + 1))"; }

body_chat() {
    local sys="$1" stream="$2" maxtok="$3"
    jq -nc --arg s "$sys" \
        '{model:"x",messages:[{role:"system",content:$s},{role:"user",content:"Tell me about cache invalidation."}],max_tokens:$mt,temperature:0.0,stream:$st,kv_quant:$kvq} + $think' \
        --argjson think "$THINK" --argjson mt "$maxtok" --argjson st "$stream" --argjson kvq "$KVQ"
}

# ─────────────────────────────────────────────────────────────────────────
# [1] decode-phase cancel
# ─────────────────────────────────────────────────────────────────────────
decode_cancel_section() {
    echo "== [1] decode-phase cancel =="
    warmup_model
    local mark
    mark=$(sushi_log_since | wc -c)

    # Fire a streaming generation and kill the client mid-decode. The short
    # prompt prefills in well under a second on any model, and a thinking-off
    # answer can hit EOS in a few seconds on a small one, so the kill lands
    # early.
    local body
    body="$(body_chat "$SYSTEM_PROMPT" true 2000)"
    curl -sN --max-time 3 -o "$T/partial.sse" \
        -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body" 2>/dev/null
    local curl_rc=$?
    # 28 = timeout kill (what we want); 18/56/55 also fine (partial/close).
    if [ "$curl_rc" -eq 0 ]; then
        check "killed stream actually got cut early (rc=$curl_rc, needed >0)" 0
        return 1
    fi
    # Some decode tokens must have flown before the kill, else this is not a
    # decode-phase cancel. grep for content chunks (choices with delta).
    if ! grep -qE '"(content|reasoning_content)":"[^"]' "$T/partial.sse" 2>/dev/null; then
        check "decode tokens observed before the kill (mid-decode cancel)" 0
        return 1
    fi

    # Give the server a moment to notice the dead socket, cancel the slot
    # and drain the cleanup queue (the commit happens there).
    sleep 4

    # Identical re-issue: full prompt should hit the committed entry.
    local resp cached
    resp="$(curl -sf --max-time 120 -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(body_chat "$SYSTEM_PROMPT" false 1)")"
    cached="$(echo "$resp" | jq -r '.usage.prompt_tokens_details.cached_tokens // 0' 2>/dev/null)"
    check "identical re-issue after decode-cancel reuses cached KV (cached_tokens=$cached)" \
        "$([[ "${cached:-0}" -ge 100 ]] && echo 1 || echo 0)"

    log_after "$mark" | grep -q "\[hot-cache\] reused" && ok=1 || ok=0
    check "server log shows [hot-cache] reused" "$ok"

}

# ─────────────────────────────────────────────────────────────────────────
# [2] prefill-phase cancel (hybrid + plain-attention)
# ─────────────────────────────────────────────────────────────────────────
prefill_cancel_section() {
    echo "== [2] prefill-phase cancel =="

    warmup_model
    local mark
    mark=$(sushi_log_since | wc -c)

    # ~8k-token prompt. Kill at 8s: the conn thread only notices the dead
    # socket at its next keepalive probe (5s of stream silence), so an 8s
    # kill is detected at the ~10s probe — by then ~2 chunks (≥1024 tokens
    # at even a slow 115 tok/s) are forwarded and checkpointed, comfortably
    # past the 256-token floor and the first 512-stride boundary. A 2s kill
    # races the first chunk on slow models (step=0, salvage empty); the
    # chunk loop only polls the cancel flag at chunk boundaries, so the
    # chunk must also be small enough to complete inside the window
    # (the shared boot keeps the auto chunk).
    local body
    body="$(body_chat "$LONG_SYSTEM" true 2000)"
    curl -sN --max-time 8 -o "$T/prefill.sse" \
        -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$body" 2>/dev/null
    local curl_rc=$?

    sleep 8  # keepalive-detection window + chunk-loop abort + cleanup drain

    local cached
    cached="$(curl -sf --max-time 180 -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(body_chat "$LONG_SYSTEM" false 1)" \
        | jq -r '.usage.prompt_tokens_details.cached_tokens // 0' 2>/dev/null)"

    # A prefill that finishes inside the 5 s keepalive silence window is
    # only seen as disconnected at the first token write: no content in the
    # stream, yet the cache holds a FULL entry (e4b prefills 8k in ~3 s).
    if grep -qE '"(content|reasoning_content)":"[^"]' "$T/prefill.sse" 2>/dev/null \
       || log_after "$mark" | grep -q "\[hot-cache\] full reuse"; then
        # Prefill completed before the kill landed — the cancel hit decode
        # phase; not this section's contract (and [1] covers it).
        echo "NOTE: kill landed in decode phase (prompt prefilled too fast) — cached_tokens=$cached reported for information"
        check "cancelled request KV still preserved (cached_tokens=$cached)" \
            "$([[ "${cached:-0}" -ge 100 ]] && echo 1 || echo 0)"
    else
        # Genuinely mid-prefill: the forwarded-prefix commit is REQUIRED.
        check "mid-prefill kill reuses forwarded prefix (cached_tokens=$cached)" \
            "$([[ "${cached:-0}" -ge 256 ]] && echo 1 || echo 0)"
        log_after "$mark" | grep -q "from a cancelled prefill" && ok=1 || ok=0
        check "server log shows the cancelled-prefill commit" "$ok"
    fi

}

decode_cancel_section
prefill_cancel_section

echo
echo "RESULT: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]] || exit 1
exit 0
