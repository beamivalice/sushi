#!/bin/bash
# sushi-test: tier=live boot=api
# test_api_edges.sh — request-validation edges across the text surfaces, attached to the shared api boot.
# Every check is a server CONTRACT (status + shape), never a checkpoint expectation: what a request that is
# malformed, out of range, or unsupported gets back, and that the server is still alive.
#
# Every arch thinks by default, so a request whose text matters turns thinking off (`$NT_*`). GLM's template
# cannot (400), so it runs under a 16-token thought; "answers" checks read content plus reasoning_content.
. "$(dirname "$0")/lib/boot.sh"; sushi_attach
set -uo pipefail

BASE=$SUSHI_URL
N=$SUSHI_NONCE
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
if [ "${SUSHI_ARCH:-}" = glm5_next ]; then
    NT_CHAT='"reasoning_effort":"low","reasoning_budget_tokens":16'
    NT_MSG='"thinking":{"type":"enabled","budget_tokens":16}'
    NT_RESP='"reasoning":{"effort":"low"},"reasoning_budget_tokens":16'
else
    NT_CHAT='"enable_thinking":false'
    NT_MSG='"thinking":{"type":"disabled"}'
    NT_RESP='"reasoning":{"effort":"none"}'
fi

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); echo "  PASS $1"; }
bad() { FAIL=$((FAIL+1)); echo "  FAIL $1 — ${2:-}"; }
# req <method> <path> [body]  → status in $R, body in $BODY
R=""
req() {
    local method="$1" path="$2" body="${3:-}"
    if [[ -n "$body" ]]; then
        R=$(curl -s -m 300 -o "$WORK/body" -w '%{http_code}' -X "$method" "$BASE$path" -H 'Content-Type: application/json' --data-binary "$body")
    else
        R=$(curl -s -m 300 -o "$WORK/body" -w '%{http_code}' -X "$method" "$BASE$path")
    fi
    BODY=$(cat "$WORK/body")
}
J() { python3 -c "import sys,json; d=json.load(sys.stdin); print($1)" 2>/dev/null; }
expect_status() { # <want> <name>
    if [[ "$R" == "$1" ]]; then ok "$2: $1"; else bad "$2: want $1 got $R" "$(echo "$BODY" | head -c 160)"; fi
}
CONTENT='d["choices"][0]["message"]["content"]'
GENERATED='(d["choices"][0]["message"].get("reasoning_content") or "")+(d["choices"][0]["message"]["content"] or "")'
PROMPT_TOKENS='d["usage"]["prompt_tokens"]'
U="{\"role\":\"user\",\"content\":\"$N hi\"}"
# prompt_tokens of a chat body, one generated token
prompt_tokens() {
    req POST /v1/chat/completions "$1"
    echo "$BODY" | J "$PROMPT_TOKENS"
}

echo "=== request validation ==="
req POST /v1/chat/completions '';                                   expect_status 400 "empty body"
req POST /v1/chat/completions '{not json';                          expect_status 400 "invalid JSON"
req POST /v1/chat/completions '[1,2]';                              expect_status 400 "array body"
req POST /v1/chat/completions '{"model":"m"}';                      expect_status 400 "no messages"
req POST /v1/chat/completions '{"model":"m","messages":[]}';        expect_status 400 "empty messages"
req POST /v1/chat/completions '{"model":"m","messages":"x"}';       expect_status 400 "messages not array"
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"n\":2}"; expect_status 400 "n>1"
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"max_tokens\":5,$NT_CHAT,\"x\":$(python3 -c 'print("["*3000+"]"*3000)')}"
expect_status 200 "3000-deep nesting in an ignored field"
req GET  /v1/chat/completions;                                      expect_status 404 "GET on a POST route"
req POST /v1/bogus '{"model":"m"}';                                 expect_status 404 "unknown route"
req OPTIONS /v1/chat/completions;                                   expect_status 204 "CORS preflight"

echo "=== sampling edges ==="
# enable_mtp:false keeps the three greedy arms on one decode path
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"top_p\":0,\"temperature\":1.0,\"max_tokens\":8,\"enable_mtp\":false,$NT_CHAT}"
GREEDY=$(echo "$BODY" | J "$GENERATED")
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"temperature\":0,\"max_tokens\":8,\"enable_mtp\":false,$NT_CHAT}"
T0=$(echo "$BODY" | J "$GENERATED")
[[ -n "$GREEDY" && "$GREEDY" == "$T0" ]] && ok "top_p 0 is greedy (== temperature 0)" || bad "top_p 0 is greedy" "top_p0='$GREEDY' temp0='$T0'"
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"top_k\":1,\"temperature\":1.0,\"max_tokens\":8,\"enable_mtp\":false,$NT_CHAT}"
[[ -n "$T0" && "$(echo "$BODY" | J "$GENERATED")" == "$T0" ]] && ok "top_k 1 is greedy" || bad "top_k 1 is greedy"
for f in '"temperature":-1' '"temperature":100' '"top_p":2' '"top_k":-3' '"seed":9223372036854775808' '"seed":"abc"' '"max_tokens":0' '"max_tokens":-5' '"logit_bias":{"1":-100}' '"presence_penalty":2,"frequency_penalty":2'; do
    [[ "$f" == '"max_tokens"'* ]] && cap="" || cap=',"max_tokens":5'
    req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],$f$cap,$NT_CHAT}"
    [[ "$R" == 200 && -n "$(echo "$BODY" | J "$GENERATED")" ]] && ok "out-of-range $f answers" || bad "out-of-range $f" "$R $(echo "$BODY" | head -c 120)"
done

echo "=== stop sequences ==="
CNT="{\"role\":\"user\",\"content\":\"$N Count from one to ten in words, comma separated.\"}"
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$CNT],\"stop\":\"four\",\"max_tokens\":90,\"temperature\":0,$NT_CHAT}"
C=$(echo "$BODY" | J "$CONTENT"); FR=$(echo "$BODY" | J 'd["choices"][0]["finish_reason"]')
[[ "$C" != *four* && "$FR" == stop ]] && ok "stop string cuts before the match, finish stop" || bad "stop string" "fr=$FR c=${C:0:80}"
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"stop\":\"\",\"max_tokens\":5,$NT_CHAT}"
[[ -n "$(echo "$BODY" | J "$GENERATED")" ]] && ok "empty stop string is ignored" || bad "empty stop string is ignored" "$(echo "$BODY" | head -c 160)"
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"stop\":[\"\",\"zzz\"],\"max_tokens\":5,$NT_CHAT}"
[[ -n "$(echo "$BODY" | J "$GENERATED")" ]] && ok "empty stop entry in an array is ignored" || bad "empty stop entry in an array" "$(echo "$BODY" | head -c 160)"
req POST /v1/messages "{\"model\":\"m\",\"max_tokens\":90,\"stop_sequences\":[\"four\"],\"messages\":[$CNT],\"temperature\":0,$NT_MSG}"
[[ "$(echo "$BODY" | J 'd["stop_reason"]')" == stop_sequence && "$(echo "$BODY" | J 'd["stop_sequence"]')" == four ]] && ok "/v1/messages echoes the matched stop_sequence" || bad "/v1/messages stop_sequence echo" "$(echo "$BODY" | head -c 160)"

echo "=== structured output ==="
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"x\",\"schema\":\"notaschema\"}},\"max_tokens\":20}"
expect_status 400 "json_schema with a non-object schema"
req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[$U],\"response_format\":{\"type\":\"json_schema\"},\"max_tokens\":20}"
expect_status 400 "json_schema without a schema"
req POST /v1/messages "{\"model\":\"m\",\"max_tokens\":20,\"output_config\":{\"format\":{\"type\":\"json_schema\"}},\"messages\":[$U]}"
expect_status 400 "messages: output_config json_schema without a schema"
req POST /v1/responses '{"model":"m","max_output_tokens":20,"text":{"format":{"type":"json_schema","name":"x","schema":"nope"}},"input":"hi"}'
expect_status 400 "responses: text.format json_schema with a non-object schema"
req POST /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"'"$N"' Describe a cat."}],"response_format":{"type":"json_schema","json_schema":{"name":"x","strict":true,"schema":{"type":"object","properties":{"kind":{"type":"string","enum":["cat","dog"]},"legs":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}},"meta":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}},"required":["kind","legs","tags","meta"],"additionalProperties":false}}},"max_tokens":160,"temperature":0,'"$NT_CHAT"'}'
C=$(echo "$BODY" | J "$CONTENT")
echo "$C" | python3 -c 'import sys,json; d=json.load(sys.stdin); assert d["kind"] in ("cat","dog") and isinstance(d["legs"],int) and isinstance(d["tags"],list) and isinstance(d["meta"]["ok"],bool) and set(d)=={"kind","legs","tags","meta"}' 2>/dev/null \
    && ok "enum + nested object + array schema is enforced" || bad "nested schema enforced" "${C:0:160}"
req POST /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"'"$N"' Give a JSON object with keys a and b."}],"response_format":{"type":"json_object"},"max_tokens":100,"temperature":0,'"$NT_CHAT"'}'
echo "$BODY" | J "$CONTENT" | python3 -c 'import sys,json; json.load(sys.stdin)' 2>/dev/null && ok "json_object yields parseable JSON" || bad "json_object parseable"

echo "=== media on the wire ==="
# a pack without a vision tower refuses images with its own message, so the contract below needs the tower
if (sushi_require vision) > /dev/null 2>&1; then
    IMG_BAD='{"type":"image_url","image_url":{"url":"data:image/png;base64,!!!notbase64"}}'
    IMG_HTTP='{"type":"image_url","image_url":{"url":"http://127.0.0.1:1/x.png"}}'
    req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"$N what\"},$IMG_BAD]}],\"max_tokens\":5}"
    expect_status 400 "chat: undecodable image_url refused"
    req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":[$IMG_HTTP]}],\"max_tokens\":5}"
    expect_status 400 "chat: remote image URL refused by name"
    [[ "$(echo "$BODY" | J 'd["error"]["message"]')" == *"remote URLs are not fetched"* ]] && ok "chat: refusal names the cause" || bad "chat: refusal names the cause" "$(echo "$BODY" | head -c 160)"
    req POST /v1/messages '{"model":"m","max_tokens":5,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}]}]}'
    expect_status 400 "messages: undecodable image block refused"
    req POST /v1/responses '{"model":"m","input":[{"role":"user","content":[{"type":"input_image","image_url":"http://127.0.0.1:1/x.png"}]}],"max_output_tokens":5}'
    expect_status 400 "responses: remote input_image refused"
    # a historical (non-active) image is never decoded, so it is never refused
    req POST /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":[$IMG_HTTP,{\"type\":\"text\",\"text\":\"$N old\"}]},{\"role\":\"assistant\",\"content\":\"ok\"},$U],\"max_tokens\":5,$NT_CHAT}"
    expect_status 200 "chat: a historical bad image is not the active turn's problem"
else
    echo "  (the pack has no vision tower: media checks skipped)"
fi

echo "=== content shapes ==="
# Prompt-token counts stand in for "the model saw it": equal renderings tokenize to equal counts.
PT_PARTS=$(prompt_tokens '{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"'"$N"' Repeat exactly: ALPHA"},{"type":"text","text":" BRAVO"}]}],"max_tokens":1,"temperature":0}')
PT_STRING=$(prompt_tokens '{"model":"m","messages":[{"role":"user","content":"'"$N"' Repeat exactly: ALPHA BRAVO"}],"max_tokens":1,"temperature":0}')
[[ -n "$PT_PARTS" && "$PT_PARTS" == "$PT_STRING" ]] && ok "text parts join like one string ($PT_PARTS tokens)" || bad "text parts join like one string" "parts=$PT_PARTS string=$PT_STRING"
# \u0000 is how a JSON body carries a NUL byte; the prompt must run on past it
PT_HEAD=$(prompt_tokens '{"model":"m","messages":[{"role":"user","content":"'"$N"' Line1"}],"max_tokens":1,"temperature":0}')
PT_NUL=$(prompt_tokens '{"model":"m","messages":[{"role":"user","content":"'"$N"' Line1\u0000Line2. Reply with the word DONE."}],"max_tokens":1,"temperature":0}')
[[ -n "$PT_HEAD" && "${PT_NUL:-0}" -gt $((PT_HEAD + 5)) ]] && ok "NUL byte does not truncate the prompt" || bad "NUL byte prompt" "head=$PT_HEAD nul=$PT_NUL"
# raw UTF-8 and the same text as \u escapes (a surrogate pair for the emoji) must tokenize alike
PT_RAW=$(prompt_tokens "$(printf '{"model":"m","messages":[{"role":"user","content":"%s Say: \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e \xf0\x9f\x8d\xa3"}],"max_tokens":1,"temperature":0}' "$N")")
PT_ESC=$(prompt_tokens "$(printf '{"model":"m","messages":[{"role":"user","content":"%s Say: \\u65e5\\u672c\\u8a9e \\ud83c\\udf63"}],"max_tokens":1,"temperature":0}' "$N")")
PT_ASCII=$(prompt_tokens '{"model":"m","messages":[{"role":"user","content":"'"$N"' Say:"}],"max_tokens":1,"temperature":0}')
[[ -n "$PT_RAW" && "$PT_RAW" == "$PT_ESC" && "$PT_RAW" -gt "$PT_ASCII" ]] && ok "CJK + emoji: raw UTF-8 == \\u escapes through the escaper" || bad "CJK + emoji" "raw=$PT_RAW escaped=$PT_ESC ascii=$PT_ASCII"
req POST /v1/chat/completions "$(printf '{"model":"m","messages":[{"role":"user","content":"%s bad \xff\xfe bytes"}],"max_tokens":5}' "$N")"
[[ "$R" == 400 || "$R" == 200 ]] && ok "invalid UTF-8 body is answered ($R), not dropped" || bad "invalid UTF-8 body" "$R"
req POST /v1/chat/completions '{"model":"m","messages":[{"role":"developer","content":"Always answer in uppercase."},{"role":"user","content":"'"$N"' say hello"}],"max_tokens":10,"temperature":0,'"$NT_CHAT"'}'
expect_status 200 "developer role accepted"
req POST /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"'"$N"' Write the alphabet."},{"role":"assistant","content":"A, B, C, D,"}],"continue_final_message":true,"max_tokens":24,"temperature":0,'"$NT_CHAT"'}'
C=$(echo "$BODY" | J "$CONTENT")
[[ "$R" == 200 && "$C" != *"A, B"* ]] && ok "continue_final_message returns only the continuation" || bad "continuation echoes the prefix" "$R $C"

echo "=== tokenize ==="
req POST /tokenize '{"content":"hello world"}'
[[ "$(echo "$BODY" | J 'len(d["tokens"])')" -ge 2 ]] && ok "/tokenize" || bad "/tokenize" "$BODY"

echo "=== Responses API ==="
req POST /v1/responses '{"model":"m"}';                                expect_status 400 "responses: no input"
req POST /v1/responses '{"model":"m","input":"hi","background":true}';  expect_status 400 "responses: background"
req POST /v1/responses '{"model":"m","input":"hi","previous_response_id":"resp_nope"}'; expect_status 404 "responses: unknown previous_response_id"
req POST /v1/responses '{"model":"m","input":"'"$N"' Give a person with name and age.","text":{"format":{"type":"json_schema","name":"p","schema":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}}},"max_output_tokens":120,"temperature":0,'"$NT_RESP"'}'
echo "$BODY" | python3 -c 'import sys,json; d=json.load(sys.stdin); t=[c["text"] for o in d["output"] if o["type"]=="message" for c in o["content"]][0]; j=json.loads(t); assert isinstance(j["age"],int)' 2>/dev/null && ok "responses: text.format json_schema enforced" || bad "responses json_schema" "$(echo "$BODY" | head -c 160)"
req POST /v1/messages "{\"model\":\"m\",\"messages\":[$U]}";           expect_status 400 "messages: max_tokens required"

echo "=== alive ==="
req GET /health; expect_status 200 "server alive after every edge"
sushi_log_since | grep -qE "\[mlx\] error|panic|Segmentation" && bad "no MLX error / crash line in the log" || ok "no MLX error / crash line in the log"

echo
echo "api-edges: $PASS passed, $FAIL failed  (log: $SUSHI_LOG)"
[[ "$FAIL" -eq 0 ]]
