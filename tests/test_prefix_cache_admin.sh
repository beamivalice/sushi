#!/bin/bash
# sushi-test: tier=host
# The SSD prefix cache admin API and the settings API, on a headless server with a private HOME and cache root:
#
#  1. GET /v1/prefix-cache lists the namespace directories on disk with their bytes (a hard link counted once), entry
#     counts and `stale: true` (no model is loaded); a directory that is not a 16-hex namespace is not listed.
#  2. POST /v1/prefix-cache/clear: a bad scope is 400, an unknown model 404, a foreign Origin 403; scope `stale` deletes
#     the stale namespaces, reports the freed bytes, keeps the non-cache directory, refuses a namespace-named symlink
#     and never touches what it points at.
#  3. GET/POST /v1/settings: unknown keys and bad values are named 400s; a server key and a model key are written to
#     the settings file and read back; null removes; a models_dir that is not a folder is refused, a real one rescans.
#  4. Under --api-key --api-key-strict none of the three endpoints answers without the key.
#  5. A model-scope clear without a model id is a 400 (never the default model's tier); scope `namespace` deletes one
#     namespace by id and, like `stale`, skips one whose private `p<pid>-<n>` root another process holds.
#  6. GET /v1/settings: a long --model-dir and a sub-GB --prefix-cache-disk read back whole as the flag text; a hand-edited
#     trailing-slash model key is the one an edit changes; kv_quant is stored under its choice name; a symlinked settings
#     file is written through; each model lists the keys that apply to it and loaded_with (null while unloaded).
#  6b. Each model lists `limits` (disabled keys with a reason, a narrowed kv_quant, a context bound) and POST refuses
#     a value the model cannot honour with a 400 naming the key; a removal and a value already in the file pass.
#  7. With the tiny DeepSeek fixture (GPU lock): a cache size changed after boot reaches the next load, effective_bytes
#     reports the tier in force, and the file's models_dir on a --model-only server rescans.
#
# Fully hermetic: stub model folders, nothing loads. Usage: ./tests/test_prefix_cache_admin.sh [port]

set -u

PORT="${1:-18861}"
BINARY="${BINARY:-./zig-out/bin/sushi}"
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0

check() {
    if [ "$2" = "1" ] || [ "$2" = "true" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi
}
is() { [ "$1" = "$2" ] && echo 1 || echo 0; }

[ -x "$BINARY" ] || { echo "[fail] $BINARY not found; build first: zig build -Doptimize=ReleaseFast"; exit 1; }
command -v jq >/dev/null || { echo "SKIP: jq is required"; exit 3; }
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -q LISTEN; then
    echo "port $PORT is already in use; pass another port" >&2
    exit 1
fi

WORK="$(mktemp -d)"
MODELS="$WORK/home/.sushi/models"
OTHER="$WORK/other-models"
CACHE="$WORK/cache"
LOG="$WORK/server.log"
SPID=""
stop_server() {
    [ -n "$SPID" ] && kill "$SPID" 2>/dev/null && wait "$SPID" 2>/dev/null
    SPID=""
}
trap 'stop_server; rm -rf "$WORK"' EXIT

mk_model() { mkdir -p "$1/$2"; printf '{"model_type":"%s"}' "$3" > "$1/$2/config.json"; head -c 4096 /dev/zero > "$1/$2/model.safetensors"; }
mk_model "$MODELS" alpha qwen4_exp
mk_model "$OTHER" beta qwen4_exp

# Two stale namespaces: A holds a hard-linked file twice (counted once), B one entry. `notes` is not a namespace.
NS_A=00000000000000a1
NS_B=00000000000000b2
mkdir -p "$CACHE/$NS_A/e1" "$CACHE/$NS_A/e2" "$CACHE/$NS_B/e1" "$CACHE/notes" "$WORK/outside"
head -c 100000 /dev/zero > "$CACHE/$NS_A/e1/c000000.safetensors"
ln "$CACHE/$NS_A/e1/c000000.safetensors" "$CACHE/$NS_A/e2/c000000.safetensors"
echo '{}' > "$CACHE/$NS_A/e1/meta.json"
echo '{}' > "$CACHE/$NS_A/e2/meta.json"
head -c 50000 /dev/zero > "$CACHE/$NS_B/e1/c000000.safetensors"
echo '{}' > "$CACHE/$NS_B/e1/meta.json"
echo keep > "$CACHE/notes/keep.txt"
echo precious > "$WORK/outside/precious.txt"
ln -s "$WORK/outside" "$CACHE/00000000000000d4"


boot() {
    stop_server
    : > "$LOG"
    HOME="$WORK/home" SUSHI_PREFIX_CACHE_DIR="$CACHE" "$BINARY" --serve --port "$PORT" --log-file off "$@" > "$LOG" 2>&1 &
    SPID=$!
    for _ in $(seq 1 60); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
        kill -0 "$SPID" 2>/dev/null || break
        sleep 0.5
    done
    echo "  (server did not come up; log follows)"; cat "$LOG"
    return 1
}
get() { curl -s "$BASE$1" "${@:2}"; }
post() { curl -s -X POST "$BASE$1" -H 'Content-Type: application/json' -d "$2" "${@:3}"; }
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

boot || { echo "[fail] boot"; exit 1; }

echo "[1/4] GET /v1/prefix-cache"
L="$(get /v1/prefix-cache)"
check "the cache root is the private one" "$(is "$(echo "$L" | jq -r .root)" "$CACHE")"
check "the volume reports free and total bytes" "$(echo "$L" | jq '(.volume.free_bytes > 0) and (.volume.total_bytes > 0)')"
check "two namespaces are listed, the non-hex folder and the symlink are not" "$(is "$(echo "$L" | jq '.namespaces | length')" 2)"
check "namespace A: 100000 bytes (hard link counted once) plus two 3-byte metas, 2 entries, stale" \
    "$(echo "$L" | jq --arg id "$NS_A" '.namespaces[] | select(.id == $id) | (.bytes == 100006) and (.entries == 2) and .stale and (.model == null)')"
check "namespace B: 50003 bytes, 1 entry" \
    "$(echo "$L" | jq --arg id "$NS_B" '.namespaces[] | select(.id == $id) | (.bytes == 50003) and (.entries == 1)')"
check "the total is the sum" "$(is "$(echo "$L" | jq .total_bytes)" 150009)"

echo "[2/4] POST /v1/prefix-cache/clear"
check "a bad scope is a 400" "$(is "$(code -X POST "$BASE/v1/prefix-cache/clear" -d '{"scope":"everything"}')" 400)"
check "an unknown model is a 404" "$(is "$(code -X POST "$BASE/v1/prefix-cache/clear" -d '{"scope":"model","model":"nope"}')" 404)"
check "an unloaded model has no tier to clear (409)" "$(is "$(code -X POST "$BASE/v1/prefix-cache/clear" -d '{"scope":"model","model":"alpha"}')" 409)"
check "a foreign Origin is refused (403)" "$(is "$(code -X POST "$BASE/v1/prefix-cache/clear" -H 'Origin: http://evil.test' -d '{"scope":"stale"}')" 403)"
R="$(post /v1/prefix-cache/clear '{"scope":"stale"}')"
check "stale: both namespaces removed, 150009 bytes freed" \
    "$(echo "$R" | jq '(.stale_namespaces_removed == 2) and (.freed_bytes == 150009) and (.refused_symlinks == 1)')"
check "the namespaces are gone" "$([ ! -e "$CACHE/$NS_A" ] && [ ! -e "$CACHE/$NS_B" ] && echo 1 || echo 0)"
check "the non-cache folder is untouched" "$([ -f "$CACHE/notes/keep.txt" ] && echo 1 || echo 0)"
check "the symlink and what it points at are untouched" "$([ -L "$CACHE/00000000000000d4" ] && [ -f "$WORK/outside/precious.txt" ] && echo 1 || echo 0)"
check "the listing is empty afterwards" "$(is "$(get /v1/prefix-cache | jq '.namespaces | length')" 0)"

echo "[2b/4] clear scopes: model id required, one namespace by id, a private root held elsewhere"
check "a model scope without a model is a 400" "$(is "$(code -X POST "$BASE/v1/prefix-cache/clear" -d '{"scope":"model"}')" 400)"
check "a model scope with a number is a 400" "$(is "$(code -X POST "$BASE/v1/prefix-cache/clear" -d '{"scope":"model","model":5}')" 400)"
check "a namespace scope with a bad id is a 400" "$(is "$(code -X POST "$BASE/v1/prefix-cache/clear" -d '{"scope":"namespace","id":"../x"}')" 400)"
check "a namespace scope with an absent id is a 404" "$(is "$(code -X POST "$BASE/v1/prefix-cache/clear" -d '{"scope":"namespace","id":"00000000000000ee"}')" 404)"
NS_C=00000000000000c3
NS_E=00000000000000e5
mkdir -p "$CACHE/$NS_C/e1" "$CACHE/$NS_C/p999-0/e1" "$CACHE/$NS_E/e1"
echo '{}' > "$CACHE/$NS_C/e1/meta.json"; echo '{}' > "$CACHE/$NS_C/p999-0/e1/meta.json"; echo '{}' > "$CACHE/$NS_E/e1/meta.json"
head -c 1000 /dev/zero > "$CACHE/$NS_C/p999-0/e1/c000000.safetensors"
python3 -I -c '
import fcntl, sys, time
f = open(sys.argv[1], "w"); fcntl.flock(f, fcntl.LOCK_EX); print("held", flush=True); time.sleep(120)' "$CACHE/$NS_C/p999-0/.lock" > "$WORK/holder.out" &
HOLDER=$!
for _ in $(seq 1 20); do grep -q held "$WORK/holder.out" && break; sleep 0.2; done
L="$(get /v1/prefix-cache)"
check "a namespace whose private root is held is held_by_other_process, not stale"     "$(echo "$L" | jq --arg id "$NS_C" '.namespaces[] | select(.id == $id) | (.stale == false) and .held_by_other_process')"
R="$(post /v1/prefix-cache/clear '{"scope":"stale"}')"
check "stale skips it and deletes the other"     "$(echo "$R" | jq '(.skipped_live_namespaces == 1) and (.stale_namespaces_removed == 1)')"
check "the held namespace is intact" "$([ -f "$CACHE/$NS_C/p999-0/e1/c000000.safetensors" ] && echo 1 || echo 0)"
mkdir -p "$CACHE/$NS_E/e1"; echo '{}' > "$CACHE/$NS_E/e1/meta.json"
R="$(post /v1/prefix-cache/clear "{\"scope\":\"namespace\",\"id\":\"$NS_C\"}")"
check "the namespace scope skips a held namespace too" "$(echo "$R" | jq '(.skipped_live_namespaces == 1) and (.stale_namespaces_removed == 0)')"
R="$(post /v1/prefix-cache/clear "{\"scope\":\"namespace\",\"id\":\"$NS_E\"}")"
check "the namespace scope deletes the one named" "$(echo "$R" | jq '(.stale_namespaces_removed == 1)')"
kill "$HOLDER" 2>/dev/null; wait "$HOLDER" 2>/dev/null
R="$(post /v1/prefix-cache/clear "{\"scope\":\"namespace\",\"id\":\"$NS_C\"}")"
check "once the holder is gone it is deleted" "$(echo "$R" | jq '(.stale_namespaces_removed == 1)')"

echo "[3/4] GET and POST /v1/settings"
S="$(get /v1/settings)"
check "the file is under the private HOME" "$(is "$(echo "$S" | jq -r .file)" "$WORK/home/.sushi/model-settings.json")"
check "the model is listed with every key" "$(echo "$S" | jq '((.models | length) == 1) and ((.models[0].settings | length) == (.model_keys | length))')"
check "an unloaded model lists the keys that apply and no loaded values" \
    "$(echo "$S" | jq '.models[0] | (.loaded_with == null) and (.keys | index("kv_quant") != null) and (.keys | index("ssd_prefetch") == null)')"
check "the SSD cache size reports the ask beside what is in force" "$(echo "$S" | jq '.server.prefix_cache_disk_gb | (.ask_bytes > 0) and (.running_tiers == []) and (.effective_bytes == .ask_bytes)')"
check "kv_quant 5 is not a choice" "$(is "$(post /v1/settings '{"model":"alpha","set":{"kv_quant":5}}' | jq -r .error.type)" invalid_setting)"
post /v1/settings '{"model":"alpha","set":{"kv_quant":"16"}}' >/dev/null
check "kv_quant 16 is accepted and stored as off" "$(jq -e --arg p "$MODELS/alpha" '.[$p].kv_quant == "off"' "$WORK/home/.sushi/model-settings.json" >/dev/null && echo 1 || echo 0)"
post /v1/settings '{"model":"alpha","set":{"kv_quant":null}}' >/dev/null
check "nothing is set: every key is default" "$(echo "$S" | jq '[.models[0].settings[] | .source] | all(. == "default")')"
check "an unknown key is refused" "$(is "$(post /v1/settings '{"model":"alpha","set":{"temperature":1}}' | jq -r '.error.type')" unknown_setting)"
check "a bad value is refused" "$(is "$(post /v1/settings '{"model":"alpha","set":{"ssd_prefetch":101}}' | jq -r '.error.type')" invalid_setting)"
check "an unknown model is a 404" "$(is "$(code -X POST "$BASE/v1/settings" -d '{"model":"nope","set":{"mtp":true}}')" 404)"
check "saving a model key is ok" "$(post /v1/settings '{"model":"alpha","set":{"kv_quant":"4","think_penalty":2}}' | jq '.ok')"
S="$(get /v1/settings)"
check "the model key reads back with source file" \
    "$(echo "$S" | jq '.models[0].settings | (.kv_quant.file == "4") and (.kv_quant.source == "file") and (.think_penalty.file == 2)')"
check "the file holds the model's path as the key" "$(jq -e --arg p "$MODELS/alpha" '.[$p].kv_quant == "4"' "$WORK/home/.sushi/model-settings.json" >/dev/null && echo 1 || echo 0)"
post /v1/settings '{"model":"alpha","set":{"kv_quant":null,"think_penalty":null}}' >/dev/null
check "null removes the keys" "$(get /v1/settings | jq '.models[0].settings | (.kv_quant.file == null) and (.think_penalty.source == "default")')"
check "max_tokens is a listed key" "$(get /v1/settings | jq '(.model_keys | map(.name) | index("max_tokens")) != null')"
check "max_tokens 0 is refused" "$(is "$(post /v1/settings '{"model":"alpha","set":{"max_tokens":0}}' | jq -r .error.type)" invalid_setting)"
check "max_tokens is saved" "$(post /v1/settings '{"model":"alpha","set":{"max_tokens":32000}}' | jq '.ok')"
check "and reads back from the file" "$(get /v1/settings | jq '.models[0].settings.max_tokens | (.file == 32000) and (.source == "file")')"
check "a model that is not loaded reports no defaults" "$(get /v1/settings | jq '.models[0].defaults == null')"
post /v1/settings '{"model":"alpha","set":{"max_tokens":null}}' >/dev/null
check "null removes max_tokens" "$(get /v1/settings | jq '.models[0].settings.max_tokens | (.file == null) and (.source == "default")')"
check "the SSD cache size is saved" "$(post /v1/settings '{"server":{"prefix_cache_disk_gb":7}}' | jq '.ok')"
check "and read back" "$(get /v1/settings | jq '.server.prefix_cache_disk_gb | (.file == 7) and (.source == "file")')"
check "a models_dir that is not a folder is refused" "$(is "$(post /v1/settings "{\"server\":{\"models_dir\":\"$WORK/missing\"}}" | jq -r .error.type)" invalid_setting)"
check "a real models_dir rescans and finds its models" "$(post /v1/settings "{\"server\":{\"models_dir\":\"$OTHER\"}}" | jq '.ok')"
check "the other folder's model is listed" "$(get /v1/settings | jq '[.models[].id] | index("beta") != null')"
check "a settings write keeps the other keys" "$(jq -e '.server.prefix_cache_disk_gb == 7' "$WORK/home/.sushi/model-settings.json" >/dev/null && echo 1 || echo 0)"

echo "[3b/4] trailing-slash key, symlinked settings file, flag text"
SETTINGS="$WORK/home/.sushi/model-settings.json"
printf '{"%s/":{"ctx_size":64}}' "$MODELS/alpha" > "$SETTINGS"
check "a slashed key is read" "$(get /v1/settings | jq '.models[] | select(.id == "alpha") | .settings.ctx_size.file == 64')"
post /v1/settings '{"model":"alpha","set":{"ctx_size":128}}' >/dev/null
check "an edit changes that key and adds no second section" \
    "$(is "$(get /v1/settings | jq '.models[] | select(.id == "alpha") | .settings.ctx_size.file == 128')$(jq 'length' "$SETTINGS")" "true1")"
mv "$SETTINGS" "$WORK/real-settings.json"; ln -s "$WORK/real-settings.json" "$SETTINGS"
post /v1/settings '{"model":"alpha","set":{"ctx_size":256}}' >/dev/null
check "a symlinked settings file stays a link and its target takes the write" \
    "$([ -L "$SETTINGS" ] && jq -e '[.[]] | .[0].ctx_size == 256' "$WORK/real-settings.json" >/dev/null && echo 1 || echo 0)"
rm -f "$SETTINGS"
LONG="$WORK/a-models-directory-with-a-rather-long-name-so-that-the-path-does-not-fit-a-small-buffer-0123456789/inner"
mk_model "$LONG" gamma qwen4_exp
boot --model-dir "$LONG" --prefix-cache-disk 512MB || { echo "[fail] boot"; exit 1; }
S="$(get /v1/settings)"
check "a long --model-dir reads back whole" "$(is "$(echo "$S" | jq -r .server.models_dir.flag)" "$LONG")"
check "a sub-GB --prefix-cache-disk reads back with its unit and the effective bytes" \
    "$(echo "$S" | jq '.server.prefix_cache_disk_gb | (.flag == "512MB") and (.locked) and (.effective_bytes == 536870912)')"

echo "[3c/4] per-model limits: what a model cannot honour is not offered and is refused with its reason"
LIM="$WORK/limit-models"
mk_model "$LIM" glm glm5_next
mk_model "$LIM" ds deepseek_v41
mk_model "$LIM" qw qwen4_exp
boot --model-dir "$LIM" || { echo "[fail] boot"; exit 1; }
S="$(get /v1/settings)"
lim() { echo "$S" | jq -r --arg id "$1" ".models[] | select(.id == \$id) | .limits$2"; }
check "GLM offers kv_quant off, 6 and 8 only" "$(is "$(lim glm '.kv_quant.choices | join(",")')" "off,6,8")"
check "GLM has no MTP: mtp and its tail keys are disabled with a reason" \
    "$(echo "$S" | jq '.models[] | select(.id == "glm") | [.limits.mtp, .limits.mtp_acceptance, .limits.mtp_greedy_tail] | all(.disabled and (.why | length > 0))')"
check "DeepSeek keeps no KV cache: kv_quant is disabled and absent from keys" \
    "$(echo "$S" | jq '.models[] | select(.id == "ds") | .limits.kv_quant.disabled and (.keys | index("kv_quant") == null)')"
check "a pack without STQ experts has no expert prefetcher" "$(is "$(lim ds '.ssd_prefetch.disabled')" true)"
check "a pack with no MTP head disables mtp" "$(is "$(lim qw '.mtp.disabled')" true)"
check "GLM refuses kv_quant 4 with a 400 naming it" "$(post /v1/settings '{"model":"glm","set":{"kv_quant":"4"}}' | jq '(.error.type == "invalid_setting") and (.error.message | contains("kv_quant"))')"
check "GLM takes kv_quant 6" "$(post /v1/settings '{"model":"glm","set":{"kv_quant":"6"}}' | jq '.ok')"
check "GLM refuses mtp true" "$(is "$(post /v1/settings '{"model":"glm","set":{"mtp":true}}' | jq -r .error.type)" invalid_setting)"
check "DeepSeek refuses any kv_quant" "$(is "$(post /v1/settings '{"model":"ds","set":{"kv_quant":"8"}}' | jq -r .error.type)" invalid_setting)"
check "a removal is always accepted" "$(post /v1/settings '{"model":"glm","set":{"mtp":null,"kv_quant":null}}' | jq '.ok')"
printf '{"%s":{"kv_quant":"4","mtp":true}}' "$LIM/glm" > "$WORK/home/.sushi/model-settings.json"
check "a value already in the file is still reported, not refused" \
    "$(get /v1/settings | jq '.models[] | select(.id == "glm") | .settings.kv_quant.file == "4"')"

echo "[4/4] --api-key --api-key-strict"
boot --api-key sekret --api-key-strict || { echo "[fail] boot"; exit 1; }
for ep in "GET /v1/prefix-cache" "POST /v1/prefix-cache/clear" "GET /v1/settings" "POST /v1/settings"; do
    m="${ep%% *}"; p="${ep#* }"
    check "$ep without the key is 401" "$(is "$(code -X "$m" "$BASE$p" -d '{}')" 401)"
done
check "GET /v1/prefix-cache with the key is 200" "$(is "$(code "$BASE/v1/prefix-cache" -H 'Authorization: Bearer sekret')" 200)"
check "GET /v1/settings with the key is 200" "$(is "$(code "$BASE/v1/settings" -H 'Authorization: Bearer sekret')" 200)"

stop_server

echo "[5/5] a loaded model's tier: clear through the inference thread, then a new write (tiny DeepSeek fixture, GPU lock)"
FIXTURE="${PREFIX_ADMIN_TINY_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/test-fixtures/dsv41-dspark-tiny}"
if [ ! -f "$FIXTURE/config.json" ]; then
    echo "  SKIP no tiny fixture at $FIXTURE"
else
    TINY="$WORK/tiny-models/tiny"
    mkdir -p "$TINY"
    for f in "$FIXTURE"/model-*.safetensors "$FIXTURE"/model.safetensors.index.json "$FIXTURE"/engram_token_map.json; do
        [ -e "$f" ] && ln -s "$f" "$TINY/$(basename "$f")"
    done
    python3 -I - "$FIXTURE/config.json" "$TINY" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
text = cfg["text_config"]
text["max_position_embeddings"] = 16384
text["rope_scaling"]["original_max_position_embeddings"] = 4096
text["rope_scaling"]["factor"] = 4
json.dump(cfg, open(sys.argv[2] + "/config.json", "w"))
symbols = [chr(b) for b in range(0x21, 0x7F)] + ["Ġ", "Ċ"]
tok = {"model": {"type": "BPE", "vocab": {s: i for i, s in enumerate(symbols)}, "merges": []},
       "pre_tokenizer": {"type": "ByteLevel"}, "added_tokens": []}
json.dump(tok, open(sys.argv[2] + "/tokenizer.json", "w"))
json.dump({}, open(sys.argv[2] + "/tokenizer_config.json", "w"))
PY
    LOCK="$(cd "$(dirname "$0")/.." && pwd)/scripts/gpu-lock.sh"
    owner="prefix-cache-admin-$$"
    "$LOCK" acquire "$owner" >/dev/null || { echo "FAIL: could not take the GPU lock"; exit 1; }
    trap '"$LOCK" release "$owner" >/dev/null 2>&1; stop_server; rm -rf "$WORK"' EXIT
    rm -rf "$CACHE"; mkdir -p "$CACHE"
    # The file turns the tier off at boot; a later size has to reach the next load.
    mkdir -p "$WORK/home/.sushi"
    echo '{"server":{"prefix_cache_disk_gb":0}}' > "$WORK/home/.sushi/model-settings.json"
    HOME="$WORK/home" SUSHI_PREFIX_CACHE_DIR="$CACHE" "$BINARY" --serve --model-dir "$WORK/tiny-models" --ssd-budget-gb 6 --port "$PORT" --log-file off > "$LOG" 2>&1 &
    SPID=$!
    for _ in $(seq 1 120); do curl -sf "$BASE/health" >/dev/null 2>&1 && break; sleep 1; done
    PROMPT="$(python3 -c 'print("abcdefghij " * 150)')"
    ask() { curl -s -m 120 "$BASE/v1/completions" -H 'Content-Type: application/json' -d "{\"model\":\"tiny\",\"prompt\":\"$PROMPT\",\"max_tokens\":4,\"temperature\":0}" | jq -r '.choices[0].text'; }
    ask >/dev/null
    check "a tier switched off by the file writes nothing" "$(is "$(get /v1/prefix-cache | jq '.namespaces | length')" 0)"
    check "the boot log says the 0 was a request, not a shortage" "$(grep -q 'SSD prefix cache OFF: set to 0 by model-settings.json' "$LOG" && echo 1 || echo 0)"
    R="$(post /v1/settings '{"server":{"prefix_cache_disk_gb":2}}')"
    check "with no running tier the answer says it applies at the next load" "$(echo "$R" | jq '.applied.prefix_cache_disk_gb == "saved; applies at the next model load"')"
    check "the ask is reported while no tier runs" "$(get /v1/settings | jq '.server.prefix_cache_disk_gb | (.ask_bytes == 2147483648) and (.effective_bytes == 2147483648) and (.running_tiers == [])')"
    code -X POST "$BASE/v1/unload-model" -d '{"model":"tiny"}' >/dev/null
    first="$(ask)"
    check "after a reload the new size is in force: the tier exists" "$(get /v1/prefix-cache | jq '.namespaces | length >= 1')"
    check "the boot log names the file as the source of the size" "$(grep -q 'Allocating 2.0 GB SSD for the prefix cache (model-settings.json)' "$LOG" && echo 1 || echo 0)"
    check "effective_bytes is the running tier's cap and running_tiers names it" "$(get /v1/settings | jq '.server.prefix_cache_disk_gb | (.effective_bytes == 2147483648) and (.running_tiers[0].model == "tiny") and (.running_tiers[0].cap_bytes == 2147483648)')"
    check "the loaded model reports the keys that apply and what it loaded with" \
        "$(get /v1/settings | jq '.models[] | select(.id == "tiny") | (.loaded_with != null) and (.keys | index("kv_quant") == null) and (.keys | index("ssd_prefetch") == null) and (.limits.ssd_prefetch.disabled) and (.limits.kv_quant.disabled)')"
    S="$(get /v1/props 2>/dev/null)"
    P="$(get /props)"
    check "/props does not report the file's acceptance as applying on DeepSeek" "$(echo "$P" | jq '.settings.mtp.acceptance_applies == false')"
    # The entry lands after the response; wait for the listing to show it.
    wait_entries() { for _ in $(seq 1 20); do [ "$(get /v1/prefix-cache | jq '[.namespaces[].entries] | add // 0')" -ge 1 ] && return 0; sleep 0.5; done; return 1; }
    first="$(ask)"
    check "the first request answers" "$([ -n "$first" ] && [ "$first" != null ] && echo 1 || echo 0)"
    wait_entries
    L="$(get /v1/prefix-cache)"
    check "the loaded model owns a namespace with a live tier" "$(echo "$L" | jq '.namespaces[0] | (.model == "tiny") and (.stale == false) and (.tier.used_bytes > 0) and (.entries >= 1)')"
    R="$(post /v1/prefix-cache/clear '{"scope":"model","model":"tiny"}')"
    check "clearing the model frees its bytes and entries" "$(echo "$R" | jq '(.freed_bytes > 0) and (.model_entries_removed >= 1)')"
    L="$(get /v1/prefix-cache)"
    check "the index, the budget and the files agree on empty" "$(echo "$L" | jq '.namespaces[0] | (.bytes == 0) and (.entries == 0) and (.tier.used_bytes == 0) and (.tier.entries == 0)')"
    check "the model still answers after the clear" "$([ "$(ask)" = "$first" ] && echo 1 || echo 0)"
    wait_entries
    check "and its cache is written again" "$(get /v1/prefix-cache | jq '.namespaces[0] | (.bytes > 0) and (.tier.entries >= 1) and (.tier.used_bytes > 0)')"
    check "a new size reaches the running tier at once" "$(post /v1/settings '{"server":{"prefix_cache_disk_gb":1}}' | jq '.applied.prefix_cache_disk_gb == "applied to the running tiers"')"
    check "the tier cap follows it" "$(get /v1/prefix-cache | jq '.namespaces[0].tier.cap_bytes <= 1073741824')"
    check "and effective_bytes reports the cap in force" "$(get /v1/settings | jq '.server.prefix_cache_disk_gb.effective_bytes <= 1073741824')"
    R="$(post /v1/prefix-cache/clear "{\"scope\":\"namespace\",\"id\":\"$(get /v1/prefix-cache | jq -r '.namespaces[0].id')\"}")"
    check "a loaded model's namespace is not deletable by id" "$(is "$(echo "$R" | jq -r .error.type)" namespace_in_use)"
    stop_server
fi

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
