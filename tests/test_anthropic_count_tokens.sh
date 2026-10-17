#!/bin/bash
# Integration test for POST /v1/messages/count_tokens.
# The count must equal the prompt /v1/messages prefills for the same body: usage.input_tokens
# plus usage.cache_read_input_tokens (input_tokens excludes the cached part).
# Usage: ./tests/test_anthropic_count_tokens.sh [port]
# Requires a running sushi server with a loaded model.

PORT=${1:-8080}
BASE="http://127.0.0.1:$PORT"
FIXTURE="$(cd "$(dirname "$0")" && pwd)/../src/fixtures/count_tokens_sample.json"
PASS=0
FAIL=0
TOTAL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

assert_eq() {
    TOTAL=$((TOTAL + 1))
    local desc="$1" expected="$2" actual="$3"
    if [ -n "$expected" ] && [ "$expected" = "$actual" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $desc ($actual)"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $desc"
        echo -e "    expected: $expected"
        echo -e "    actual:   $actual"
    fi
}

# assert_same <desc> <a> <b>: equal, empty included.
assert_same() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = "$3" ]; then
        PASS=$((PASS + 1))
        echo -e "  ${GREEN}PASS${NC} $1 (${2:-none})"
    else
        FAIL=$((FAIL + 1))
        echo -e "  ${RED}FAIL${NC} $1"
        echo -e "    count_tokens: $2"
        echo -e "    messages:     $3"
    fi
}

# json_get <json> <python expression over `d`>
json_get() {
    echo "$1" | python3 -c "import sys,json; d=json.load(sys.stdin); print($2)" 2>/dev/null
}

# with_fields <json> <python dict literal>: the body with those top-level fields set.
with_fields() {
    echo "$1" | python3 -c "import sys,json; d=json.load(sys.stdin); d.update($2); print(json.dumps(d))"
}

count_tokens() {
    curl -s "$BASE/v1/messages/count_tokens?beta=true" -H "Content-Type: application/json" \
        -H "anthropic-version: 2023-06-01" -d "$1"
}

# The prompt /v1/messages prefilled for <body>, with one generated token.
prefilled() {
    local reply
    reply=$(curl -s "$BASE/v1/messages" -H "Content-Type: application/json" \
        -H "anthropic-version: 2023-06-01" -d "$(with_fields "$1" '{"max_tokens": 1}')")
    json_get "$reply" "d['usage']['input_tokens'] + d['usage'].get('cache_read_input_tokens', 0)"
}

echo "=== Anthropic count_tokens Integration Tests ==="
echo "Server: $BASE"

if ! curl -sf "$BASE/health" > /dev/null 2>&1; then
    echo "SKIP: Server not running on port $PORT"
    exit 0
fi

SAMPLE=$(cat "$FIXTURE")
TOOLS=$(with_fields "$SAMPLE" '{"tools": [{"name": "get_time", "description": "Get the time in a city", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]}')
THINKING=$(with_fields "$SAMPLE" '{"thinking": {"type": "enabled", "budget_tokens": 1024}}')

echo ""
echo "--- 1: count_tokens equals the prompt /v1/messages prefills ---"
for case in SAMPLE TOOLS THINKING; do
    body="${!case}"
    reply=$(count_tokens "$body")
    counted=$(json_get "$reply" "d['input_tokens']")
    keys=$(json_get "$reply" "','.join(sorted(d))")
    assert_eq "$case: the answer is {input_tokens}" "input_tokens" "$keys"
    assert_eq "$case: input_tokens == usage.input_tokens + cache_read_input_tokens" "$(prefilled "$body")" "$counted"
done

echo ""
echo "--- 2: an unknown model name counts with the default model ---"
DEFAULT_ID=$(json_get "$(curl -s "$BASE/v1/models")" "d['data'][0]['id']")
by_default=$(json_get "$(count_tokens "$(with_fields "$SAMPLE" "{\"model\": \"$DEFAULT_ID\"}")")" "d['input_tokens']")
by_unknown=$(json_get "$(count_tokens "$(with_fields "$SAMPLE" '{"model": "gpt-4"}')")" "d['input_tokens']")
assert_eq "gpt-4 counts as $DEFAULT_ID" "$by_default" "$by_unknown"

echo ""
echo "--- 3: refusals take Anthropic's error shape ---"
status_and_type() {
    local out
    out=$(curl -s -w '\n%{http_code}' "$BASE/v1/messages/count_tokens" -H "Content-Type: application/json" -d "$1")
    echo "$(echo "$out" | tail -1) $(json_get "$(echo "$out" | sed '$d')" "d['type'] + ':' + d['error']['type']")"
}
assert_eq "invalid JSON" "400 error:invalid_request_error" "$(status_and_type '{not json')"
assert_eq "missing messages" "400 error:invalid_request_error" "$(status_and_type '{"model":"sushi"}')"
assert_eq "non-array messages" "400 error:invalid_request_error" "$(status_and_type '{"messages":"hi"}')"
assert_eq "unregistered model path" "404 error:not_found_error" "$(status_and_type '{"model":"/no/such/pack","messages":[{"role":"user","content":"hi"}]}')"

echo ""
echo "--- 4: an effort word counts and reports as on /v1/messages ---"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# answer <path> <body> <python expression for the token count>:
# sets STATUS, TOKENS, HEADER (X-Sushi-Effort-Notice) and FIELD (effort_notice).
answer() {
    STATUS=$(curl -s -D "$TMP/head" -o "$TMP/body" -w '%{http_code}' "$BASE$1" \
        -H "Content-Type: application/json" -H "anthropic-version: 2023-06-01" -d "$2")
    TOKENS=$(json_get "$(cat "$TMP/body")" "$3")
    HEADER=$(grep -i '^X-Sushi-Effort-Notice:' "$TMP/head" | sed 's/^[^:]*: *//' | tr -d '\r')
    FIELD=$(json_get "$(cat "$TMP/body")" "d.get('effort_notice', '')")
}
EFFORT_HIGH=$(with_fields "$SAMPLE" '{"output_config": {"effort": "high"}}')
EFFORT_LOW=$(with_fields "$SAMPLE" '{"output_config": {"effort": "low"}}')
EFFORT_THINKING=$(with_fields "$THINKING" '{"output_config": {"effort": "low"}}')
for case in EFFORT_HIGH EFFORT_LOW EFFORT_THINKING; do
    body="${!case}"
    answer /v1/messages/count_tokens "$body" "d['input_tokens']"
    c_status=$STATUS c_tokens=$TOKENS c_header=$HEADER c_field=$FIELD
    answer /v1/messages "$(with_fields "$body" '{"max_tokens": 1}')" \
        "d['usage']['input_tokens'] + d['usage'].get('cache_read_input_tokens', 0)"
    assert_eq "$case: status" "$STATUS" "$c_status"
    assert_eq "$case: input_tokens == usage.input_tokens + cache_read_input_tokens" "$TOKENS" "$c_tokens"
    assert_same "$case: X-Sushi-Effort-Notice header" "$c_header" "$HEADER"
    assert_same "$case: effort_notice field" "$c_field" "$FIELD"
done

echo ""
echo "=== Summary ==="
echo -e "Total: $TOTAL  ${GREEN}Passed: $PASS${NC}  ${RED}Failed: $FAIL${NC}"
if [ $FAIL -gt 0 ]; then
    exit 1
fi
echo "All count_tokens tests passed."
