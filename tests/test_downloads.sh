#!/bin/bash
# sushi-test: tier=host
. "$(dirname "$0")/private_cache.sh"
# The downloads API on a headless server (private HOME, an empty --model-dir, no model): offline it lists the
# known packs without sizes; through tests/hf_stub.py (SUSHI_HF_ENDPOINT) it lists the org's public Sushi repo with its
# size and hides the known packs the stub lacks and a private one; a download reports its file, bytes and speed, a
# cancel keeps the partials, a restarted server shows the pack as partly on disk and finishes it with a ranged request
# (resume), the finished pack is in the models list (rescan), a corrupt file fails naming that file, a load of a pack
# while its update replaces files is refused naming the update, and so is a load after that update was stopped.
#
# Usage: ./tests/test_downloads.sh [port]
set -u

PORT="${1:-18872}"
STUB_PORT=$((PORT + 1))
BIN="${SUSHI_BIN:-./zig-out/bin/sushi}"
[ -x "$BIN" ] || { echo "FAIL: $BIN not found — build first: zig build -Doptimize=ReleaseFast"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="http://127.0.0.1:$PORT"

WORK="$(cd "$(mktemp -d)" && pwd -P)"
SPID=""
STUB=""
stop_server() { [ -n "$SPID" ] && { kill "$SPID"; wait "$SPID"; } 2>/dev/null; SPID=""; }
trap 'stop_server; [ -n "$STUB" ] && { kill "$STUB"; wait "$STUB"; } 2>/dev/null; rm -rf "$WORK"' EXIT
PASS=0
FAIL=0
check() {
    if [ "$2" = "1" ]; then echo "  PASS $1"; PASS=$((PASS + 1)); else echo "  FAIL $1"; FAIL=$((FAIL + 1)); fi
}
ok() { if "$@" > /dev/null 2>&1; then echo 1; else echo 0; fi; }

export HOME="$WORK/home"
mkdir -p "$HOME"
unset HF_TOKEN HF_TOKEN_PATH HF_HOME
export SUSHI_HF_ENDPOINT="http://127.0.0.1:$STUB_PORT"
MODELS="$WORK/models"
mkdir -p "$MODELS"
HF="$WORK/hf"
REPO_ID=beamster/Stub-Sushi-1bpw
REPO="$HF/$REPO_ID"
DEST="$MODELS/$REPO_ID"
mkdir -p "$REPO" "$HF/beamster/Hidden-Sushi-1bpw"
printf '{"model_type":"qwen4_exp"}\n' > "$REPO/config.json"
printf 'tokenizer\n' > "$REPO/tokenizer.json"
head -c 2000000 /dev/urandom > "$REPO/model-00001.safetensors"
head -c 2000000 /dev/urandom > "$REPO/model-00002.safetensors"
printf 'x\n' > "$HF/beamster/Hidden-Sushi-1bpw/config.json"
echo beamster/Hidden-Sushi-1bpw > "$HF/.private"
SIZE=$(cat "$REPO"/* | wc -c | tr -d ' ')

boot() {
    "$BIN" --serve --model-dir "$MODELS" --port "$PORT" --log-file off > "$WORK/server.log" 2>&1 &
    SPID=$!
    for _ in $(seq 1 60); do curl -sf "$BASE/health" > /dev/null && return 0; sleep 0.5; done
    echo "  (server did not come up)"; cat "$WORK/server.log"; return 1
}
get() { curl -s "$BASE/v1/downloads$1" > "$WORK/d.json"; }
# Python over the last answer: `q 'expr'` prints the value, `t 'expr'` is 1/0.
q() { python3 -c "import json,sys; d=json.load(open('$WORK/d.json')); print($1)"; }
t() { python3 -c "import json,sys; d=json.load(open('$WORK/d.json')); sys.exit(0 if ($1) else 1)" && echo 1 || echo 0; }
until_d() { # poll until expr holds (max ~$2 s)
    for _ in $(seq 1 $(( ${2:-20} * 4 ))); do get "${3:-}"; [ "$(t "$1")" = 1 ] && return 0; sleep 0.25; done
    return 1
}
post() { curl -s -o "$WORK/p.json" -w '%{http_code}' -X POST "$BASE/v1/downloads$1" -H 'Content-Type: application/json' -d "$2"; }
job() { echo "[j for j in d['jobs'] if j['repo']=='$REPO_ID'][-1]"; }
pk() { echo "[p for p in d['packs'] if p['repo']=='$REPO_ID'][0]"; }

echo "[1/7] offline: the known packs, without sizes"
boot || exit 1
until_d "d['catalog'] != 'loading'" 30
check "with Hugging Face unreachable the catalog says offline" "$(t "d['catalog'] == 'offline'")"
check "the known packs are listed with their tiers, no size" "$(t "len(d['packs']) > 0 and all(p['size_bytes'] is None and p['tier_gb'] for p in d['packs'])")"
check "this Mac's RAM and the destination are reported" "$(t "d['ram_bytes'] > 0 and d['dest']['path'] == '$MODELS' and d['dest']['locked'] and d['dest']['free_bytes'] > 0")"

echo "[2/7] the stub: the org's public pack with its size, nothing private or unpublished"
python3 -I "$HERE/hf_stub.py" "$HF" "$STUB_PORT" > "$WORK/stub.log" 2>&1 &
STUB=$!
for _ in $(seq 1 40); do curl -sf "$SUSHI_HF_ENDPOINT/api/models/$REPO_ID" > /dev/null && break; sleep 0.25; done
get "?refresh=1"
until_d "d['catalog'] == 'ready' and not d['refreshing']" 30
check "the catalog is the one public Sushi repo" "$(t "[p['repo'] for p in d['packs']] == ['$REPO_ID']")"
check "its size is the repo's" "$(t "$(pk)['size_bytes'] == $SIZE and $(pk)['tier_gb'] is None and not $(pk)['installed']")"
check "a foreign page cannot start a download" "$(ok test "$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/downloads" -H 'Origin: http://evil.test' -d "{\"repo\":\"$REPO_ID\"}")" = 403)"
check "another org is refused" "$(ok test "$(post "" '{"repo":"other/Stub-Sushi-1bpw"}')" = 400)"
check "a name that leaves the org folder is refused" "$(ok test "$(post "" '{"repo":"beamster/.."}')" = 400)"
check "a repo the catalog does not list is refused" "$(ok test "$(post "" '{"repo":"beamster/Hidden-Sushi-1bpw"}')" = 404)"
get ""
check "a refused start queues nothing" "$(t "d['jobs'] == []")"

echo "[3/7] a download in progress, then Stop"
echo 400000 > "$HF/.rate"
check "POST /v1/downloads queues it, under the catalog's spelling" "$(ok test "$(post "" '{"repo":"beamster/stub-sushi-1bpw"}')" = 202)"
check "the job and folder carry the repo's own spelling" "$(ok grep -q "\"repo\":\"$REPO_ID\"" "$WORK/p.json")"
until_d "$(job)['phase'] == 'downloading' and $(job)['file'] and $(job)['file']['done'] > 0" 20
check "the job names the file in flight and its bytes" "$(t "$(job)['file']['path'].startswith('model-0000') and $(job)['files'] == 4 and $(job)['total_bytes'] == $SIZE")"
until_d "$(job)['speed_bps'] is not None" 10
check "a speed and a time left appear" "$(t "100000 < $(job)['speed_bps'] < 2000000 and $(job)['eta_s'] is not None")"
post "" "{\"repo\":\"$REPO_ID\"}" > /dev/null
check "a second start of the same pack is not queued twice" "$(ok grep -q '"queued":false' "$WORK/p.json")"
check "Stop answers" "$(ok test "$(post /cancel "{\"repo\":\"$REPO_ID\"}")" = 200)"
until_d "$(job)['phase'] == 'cancelled'" 10
check "the job is stopped" "$(t "$(job)['phase'] == 'cancelled'")"
check "the partial and the incomplete marker stay" "$(ok ls "$DEST"/*.partial "$DEST/.pull-incomplete")"
check "no curl outlives the stop" "$(ok test "$(ps -axo command | grep -c "^curl .*127.0.0.1:$STUB_PORT")" = 0)"

echo "[4/7] a restart, then resume"
stop_server
# Without its config.json the boot's discovery skips the folder, so only the rescan can list the finished pack.
rm -f "$DEST/config.json"
boot || exit 1
check "before the resume the models list lacks it" "$(ok test "$(curl -s "$BASE/v1/models" | grep -c Stub-Sushi-1bpw)" = 0)"
until_d "d['catalog'] == 'ready' and not d['refreshing']" 30
check "after a restart the pack shows as partly on disk, no job" "$(t "$(pk)['partial_bytes'] > 0 and not $(pk)['installed'] and d['jobs'] == []")"
rm "$HF/.rate"
: > "$HF/.ranges"
post "" "{\"repo\":\"$REPO_ID\"}" > /dev/null
until_d "$(job)['phase'] in ('done', 'failed')" 30
check "the download finishes" "$(t "$(job)['phase'] == 'done' and $(job)['done_bytes'] == $SIZE")"
check "the interrupted file resumed with a ranged request" "$(ok grep -E 'model-0000[12].safetensors [1-9]' "$HF/.ranges")"
check "the files are the repo's" "$(ok cmp "$REPO/model-00001.safetensors" "$DEST/model-00001.safetensors")" 
check "the second file is the repo's" "$(ok cmp "$REPO/model-00002.safetensors" "$DEST/model-00002.safetensors")"
check "the pull manifest is written and the marker gone" "$(ok test -f "$DEST/.sushi-pull.json" -a ! -e "$DEST/.pull-incomplete")"
check "the pack reads as downloaded, with no launch line (no recipe for a pack the table lacks)" "$(t "$(pk)['installed'] and $(pk)['path'] == '$DEST' and $(pk)['launch'] is None")"
check "the models list has it (rescan)" "$(ok grep -q "Stub-Sushi-1bpw" <(curl -s "$BASE/v1/models"))"

echo "[5/7] a corrupt file fails by name"
head -c 1000 /dev/urandom > "$REPO/model-00003.safetensors"
echo model-00003.safetensors > "$HF/.corrupt"
post "" "{\"repo\":\"$REPO_ID\"}" > /dev/null
until_d "$(job)['phase'] in ('done', 'failed')" 30
check "the job fails naming the file" "$(t "$(job)['phase'] == 'failed' and 'model-00003.safetensors does not match' in $(job)['message']")"

echo "[6/7] a load while an update replaces the pack's files"
rm -f "$HF/.corrupt"
post "" "{\"repo\":\"$REPO_ID\"}" > /dev/null
until_d "$(job)['phase'] in ('done', 'failed')" 30
check "with the file served whole again the pack completes" "$(t "$(job)['phase'] == 'done'")"
printf '{"model_type":"qwen4_exp","revision":2}\n' > "$REPO/config.json"
head -c 2000000 /dev/urandom > "$REPO/model-00002.safetensors"
echo 400000 > "$HF/.rate"
post "" "{\"repo\":\"$REPO_ID\"}" > /dev/null
until_d "$(job)['phase'] == 'downloading' and $(job)['file'] and $(job)['file']['path'] == 'model-00002.safetensors'" 30
check "mid-update the folder mixes revisions: the new config.json beside the old model-00002" \
    "$(cmp -s "$REPO/config.json" "$DEST/config.json" && ! cmp -s "$REPO/model-00002.safetensors" "$DEST/model-00002.safetensors" && echo 1 || echo 0)"
load() { curl -s -o "$WORK/l.json" -w '%{http_code}' -X POST "$BASE/v1/load-model" -H 'Content-Type: application/json' -d "{\"model\":\"$DEST\"}"; }
CODE=$(load)
check "a load of the pack while it updates is refused with 409" "$(ok test "$CODE" = 409)"
check "... naming the update" "$(ok grep -q "$REPO_ID is being updated" "$WORK/l.json")"
[ "$CODE" = 409 ] || echo "    (the load answered $CODE: $(head -c 300 "$WORK/l.json"))"
rm "$HF/.rate"
until_d "$(job)['phase'] in ('done', 'failed')" 30
check "the update finishes" "$(t "$(job)['phase'] == 'done'")"
check "once it is done the pack loads again (no 409)" "$(ok test "$(load)" != 409)"

echo "[7/7] a load after an update was stopped half way"
printf '{"model_type":"qwen4_exp","revision":3}\n' > "$REPO/config.json"
head -c 2000000 /dev/urandom > "$REPO/model-00002.safetensors"
echo 400000 > "$HF/.rate"
post "" "{\"repo\":\"$REPO_ID\"}" > /dev/null
until_d "$(job)['phase'] == 'downloading' and $(job)['file'] and $(job)['file']['path'] == 'model-00002.safetensors'" 30
post /cancel "{\"repo\":\"$REPO_ID\"}" > /dev/null
until_d "$(job)['phase'] == 'cancelled'" 10
check "the stopped update left a mixed folder with its incomplete marker" \
    "$(cmp -s "$REPO/config.json" "$DEST/config.json" && ! cmp -s "$REPO/model-00002.safetensors" "$DEST/model-00002.safetensors" && [ -e "$DEST/.pull-incomplete" ] && echo 1 || echo 0)"
# The stub weights never load, so step 6's load left the entry failed; a rescan makes it loadable again.
curl -s -X POST "$BASE/v1/models/rescan" > /dev/null
CODE=$(load)
check "a load of the half-updated pack is refused with 409" "$(ok test "$CODE" = 409)"
check "... saying the update did not finish and to resume it on Downloads" "$(ok grep -q "did not finish.*Downloads" "$WORK/l.json")"
[ "$CODE" = 409 ] || echo "    (the load answered $CODE: $(head -c 300 "$WORK/l.json"))"
rm "$HF/.rate"
post "" "{\"repo\":\"$REPO_ID\"}" > /dev/null
until_d "$(job)['phase'] in ('done', 'failed')" 30
check "resumed, the update finishes" "$(t "$(job)['phase'] == 'done'")"
check "and the pack loads again (no 409)" "$(ok test "$(load)" != 409)"

stop_server
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
