#!/bin/bash
# sushi-test: tier=live boot=own pack=default boots=3
# Loading a model after boot: /v1/load-model, launch flags, `model-settings.json` and idle eviction, on symlink clones
# of the default pack under one `--model-dir` (the clones cost nothing; a second real pack cannot sit beside the first
# in 128 GB, so every switch unloads first: precedence, not coexistence). Three boots:
#
#  Boot 1, `--no-mtp --no-drafter` and a settings file for clone A:
#   [1] the file is honoured at boot where no flag was given (ctx, kv, source names) and `--no-mtp` outranks its mtp
#   [2] /v1/load-model: the response shape, 404 for an unknown id, a chat on the loaded id answers
#   [3] a COLD load of clone B carries the launch flags (ensureLoaded builds a second LoadRequest; three rounds of
#       this bug shipped): `--no-drafter` skips the in-dir drafter probe, the settings file does not leak to B
#   [4] `"default": true` re-points the default; a side-load without it does not; an omitted `model` follows it
#   [5] an edited file applies on unload + load with no restart; a malformed file never stops a load;
#       `ssd_budget_gb` in the file streams the load at that budget
#   [6] with no `--idle-evict-secs` the model stays resident after 8 s idle (the flag is the gate)
#  Boot 2, explicit launch flags beside a file that disagrees: every flag outranks its key, on boot and on a cold load;
#   an explicit `--mtp` refuses a BF16-source streamed load (skipped without that checkpoint)
#  Boot 3, `--idle-evict-secs 2`: a model nobody uses leaves residency, a status poll is not a load or a use, reloads
#   under concurrent readers of the retained CPU state work, and a reload frees what it replaces (RSS)
set -u
. "$(dirname "$0")/lib/boot.sh"

PACK=$(sushi_pack qwen4) || exit $?
STREAM_MODEL=$(sushi_pack qwen4-bf16 2> /dev/null) || STREAM_MODEL=""
RSS_GROWTH_LIMIT_MB="${IDLE_EVICT_RSS_LIMIT_MB:-40}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
PASS=0; FAIL=0
check() { if [ "$2" = 1 ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi; }
is() { [ "$1" = "$2" ] && echo 1 || echo 0; }

# Two clones, each with an in-dir drafter the probe would accept: config.json only, so the arm that probes fails
# its load loudly and the arm that does not is silent.
ROOT="$WORK/models"
A="$ROOT/scratch-org/cold-a"
B="$ROOT/scratch-org/cold-b"
for c in "$A" "$B"; do
    mkdir -p "$c/drafter"
    for f in "$PACK"/*; do ln -s "$f" "$c/$(basename "$f")"; done
    echo '{"model_type":"probe_assistant","block_size":4,"mask_token_id":1,"target_layer_ids":[0,1]}' > "$c/drafter/config.json"
done
unset f c

settings_file() { echo "$SUSHI_RUN_DIR/$SUSHI_LABEL.home/.sushi/model-settings.json"; }
post() { # route json -> http code
    curl -s -o "$WORK/body.json" -w '%{http_code}' --max-time 1800 -X POST "$SUSHI_URL/v1/$1" -H 'Content-Type: application/json' -d "$2"
}
load() { # path-or-id [json extras] -> http code; an unload's pages return lazily, so the preflight may 503 for a few seconds
    local code i
    for i in $(seq 1 12); do
        code=$(post load-model "{\"model\":\"$1\"${2:+,$2}}")
        [ "$code" = 503 ] || break
        sleep 5
    done
    echo "$code"
}
unload() { post unload-model "{\"model\":\"$1\"}"; }
models() { curl -s "$SUSHI_URL/v1/models"; }
row() { # <clone path> <ctx|kv|src>: context_length, meta.kv_quant or meta.kv_cache.source of that READY row
    models | python3 -c "
import sys, json
want = sys.argv[1].rstrip('/')
for m in json.load(sys.stdin)['data']:
    if want.endswith('/' + m['id']):
        f = sys.argv[2]
        print(m['context_length'] if f == 'ctx' else m['meta']['kv_cache']['source'] if f == 'src' else m['meta'].get('kv_quant'))
        break
" "$1" "$2"
}
model_id() { models | python3 -c "
import sys, json
want = sys.argv[1].rstrip('/')
print(next((m['id'] for m in json.load(sys.stdin)['data'] if want.endswith('/' + m['id'])), ''))" "$1"; }
default_id() { models | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])"; }
loaded_ids() { models | python3 -c "import sys,json; print(' '.join(sorted(m['id'] for m in json.load(sys.stdin)['data'] if m.get('loaded'))))"; }
resident_count() { models | grep -o '"loaded":true' | wc -l | tr -d ' '; }
props_mtp() { # <clone path> <source|default_on>
    curl -s "$SUSHI_URL/props?model=$(model_id "$1")" | python3 -c "import sys, json; print(json.load(sys.stdin)['settings']['mtp'][sys.argv[1]])" "$2"
}
chat() { # model id -> http code
    curl -s -o /dev/null -w '%{http_code}' -m 900 -X POST "$SUSHI_URL/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":4}"
}
server_rss_mb() { ps -o rss= -p "$SUSHI_PID" | awk '{printf "%d", $1/1024}'; }
logline() { grep -q -- "$1" "$SUSHI_LOG" && echo 1 || echo 0; }

# ---------------------------------------------------------------- boot 1
cat > "$WORK/settings.json" <<JSON
{ "$A/": { "ctx_size": 4096, "kv_quant": "4", "mtp": true, "mtp_acceptance": "typical", "mtp_greedy_tail": true }, "not-a-model": 1 }
JSON
echo "== boot 1: --no-mtp --no-drafter, settings file for clone A"
SUSHI_BOOT_SETTINGS="$WORK/settings.json" sushi_boot cold-1 "$A" --model-dir "$ROOT" --no-mtp --no-drafter --max-resident-models 4 || exit $?
SETTINGS=$(settings_file)

echo "[1] boot load honours the file where no flag was given"
check "context_length 4096 from the file (got $(row "$A" ctx))" "$(is "$(row "$A" ctx)" 4096)"
check "meta.kv_quant 4 from the file (got $(row "$A" kv))" "$(is "$(row "$A" kv)" 4)"
check "kv_cache source model-settings.json (got $(row "$A" src))" "$(is "$(row "$A" src)" model-settings.json)"
check "load log names the KV and ctx choices" "$(logline "\[kv-cache\] kv4 (model-settings.json); ctx 4096 (model-settings.json)")"
check "load log: --no-mtp outranks mtp:true, acceptance from the file" "$(logline "\[mtp\] off (--no-mtp); acceptance typical (model-settings.json)")"
check "load log: greedy tail from the file" "$(logline "acceptance typical (model-settings.json); greedy tail on (model-settings.json)")"
check "/props settings.mtp.source --no-mtp (got $(props_mtp "$A" source))" "$(is "$(props_mtp "$A" source)" --no-mtp)"
check "the loaded row carries its architecture" "$(models | python3 -c "
import sys, json
print(int(any(m.get('loaded') and m.get('meta', {}).get('architecture') for m in json.load(sys.stdin)['data'])))")"
A_ID=$(model_id "$A")
B_ID=$(model_id "$B")

echo "[2] /v1/load-model shape"
check "an unknown id is a 404" "$(is "$(post load-model '{"model":"no-such-model-xyz"}')" 404)"
check "the loaded model answers a chat (200)" "$(is "$(chat "$A_ID")" 200)"

echo "[3] a cold load carries the launch flags"
check "unload clone A -> 200" "$(is "$(unload "$A_ID")" 200)"
CODE=$(load "$B_ID")
check "cold load of clone B -> 200 (got $CODE)" "$(is "$CODE" 200)"
check "the response says loaded + ready" "$(python3 -c "import json; m=json.load(open('$WORK/body.json'))['model']; print(int(m['loaded'] and m['state'] == 'ready'))" 2> /dev/null || echo 0)"
check "clone B reached ready (the silent arm did the work)" "$(logline "model id=$B_ID ready")"
check "--no-drafter survives a cold load: no DFlash probe line" "$(grep -q DFlash "$SUSHI_LOG" && echo 0 || echo 1)"
check "B keeps the kv8 default (got $(row "$B" kv))" "$(is "$(row "$B" kv)" 8)"
check "B kv_cache source default (got $(row "$B" src))" "$(is "$(row "$B" src)" default)"
check "cold-load log: kv8 default, ctx auto" "$(logline "\[kv-cache\] kv8 (default); ctx auto (default)")"
check "cold-load log: --no-mtp, exact acceptance, greedy tail off" "$(logline "\[mtp\] off (--no-mtp); acceptance exact (default); greedy tail off (default)")"
check "/props settings.mtp.source --no-mtp for B (got $(props_mtp "$B" source))" "$(is "$(props_mtp "$B" source)" --no-mtp)"
check "a chat on the cold-loaded id answers (200)" "$(is "$(chat "$B_ID")" 200)"

echo "[4] \"default\": true re-points the default"
BOOT_DEFAULT=$(default_id)
check "boot default is clone A ($BOOT_DEFAULT)" "$(is "$BOOT_DEFAULT" "$A_ID")"
check "a side-load without the flag left the default alone ($(default_id))" "$(is "$(default_id)" "$A_ID")"
check "unload B, load B with default:true -> 200" "$(is "$(unload "$B_ID")$(load "$B_ID" '"default":true')" 200200)"
check "default re-pointed ($(default_id))" "$(is "$(default_id)" "$B_ID")"
CODE=$(curl -s -o /dev/null -w '%{http_code}' -m 900 -X POST "$SUSHI_URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}')
check "a request omitting model is answered by the new default, with nothing else loaded (code $CODE, loaded: $(loaded_ids))" "$([ "$CODE" = 200 ] && [ "$(loaded_ids)" = "$B_ID" ] && echo 1 || echo 0)"

echo "[5] settings edits apply on reload; a malformed file never stops a load; ssd_budget_gb streams"
unload "$B_ID" > /dev/null
cat > "$SETTINGS" <<JSON
{ "$A/": { "ctx_size": 8192, "kv_quant": "8" } }
JSON
CODE=$(load "$A_ID" '"default":true')
check "reload clone A after the edit -> 200 (got $CODE)" "$(is "$CODE" 200)"
check "A now ctx 8192 (got $(row "$A" ctx)), kv 8 (got $(row "$A" kv)) from the file" "$([ "$(row "$A" ctx)" = 8192 ] && [ "$(row "$A" kv)" = 8 ] && [ "$(row "$A" src)" = model-settings.json ] && echo 1 || echo 0)"
check "the server never restarted" "$(sushi_alive && echo 1 || echo 0)"
echo '{nope' > "$SETTINGS"
unload "$A_ID" > /dev/null
CODE=$(load "$A_ID")
check "malformed file: load -> 200 (got $CODE), defaults apply (kv source $(row "$A" src))" "$([ "$CODE" = 200 ] && [ "$(row "$A" src)" = default ] && echo 1 || echo 0)"
check "malformed file logged" "$(logline "\[model-settings\] .*malformed")"
cat > "$SETTINGS" <<JSON
{ "$A/": { "ssd_budget_gb": 60 } }
JSON
unload "$A_ID" > /dev/null
CODE=$(load "$A_ID")
check "ssd_budget_gb on a Sushi pack: load -> 200 (got $CODE)" "$(is "$CODE" 200)"
check "the load streams at the file's budget, never ignoring it" "$(logline "\[expert-stream\] ssd budget 60 GiB from model-settings.json")"
check "ssd_budget_gb is not ignored" "$(grep -q "ssd_budget_gb ignored" "$SUSHI_LOG" && echo 0 || echo 1)"

echo "[6] no --idle-evict-secs: the model stays resident"
sleep 8
check "still resident after 8 s idle" "$(is "$(resident_count)" 1)"
sushi_stop 0

# ---------------------------------------------------------------- boot 2
cat > "$WORK/settings.json" <<JSON
{ "$A/": { "ctx_size": 4096, "kv_quant": "8", "mtp": false, "mtp_acceptance": "typical", "mtp_greedy_tail": false } }
JSON
echo "== boot 2: explicit launch flags beside a file that disagrees"
SUSHI_BOOT_SETTINGS="$WORK/settings.json" sushi_boot cold-2 "$A" --model-dir "$ROOT" --no-drafter --ctx-size 16384 --kv-quant 4 \
    --mtp --mtp-tokenv3 0.9 --mtp-greedy-tail || exit $?
check "--ctx-size 16384 outranks ctx_size 4096 (got $(row "$A" ctx))" "$(is "$(row "$A" ctx)" 16384)"
check "--kv-quant 4 outranks kv_quant 8 (got $(row "$A" kv), source $(row "$A" src))" "$([ "$(row "$A" kv)" = 4 ] && [ "$(row "$A" src)" = --kv-quant ] && echo 1 || echo 0)"
check "load log names the flags" "$(logline "\[kv-cache\] kv4 (--kv-quant); ctx 16384 (--ctx-size)")"
check "--mtp outranks mtp:false, --mtp-tokenv3 outranks typical" "$(logline "\[mtp\] on (--mtp); acceptance tokenv3 (--mtp-tokenv3)")"
check "--mtp-greedy-tail outranks mtp_greedy_tail:false" "$(logline "acceptance tokenv3 (--mtp-tokenv3); greedy tail on (--mtp-greedy-tail)")"
check "/props settings.mtp.source --mtp (got $(props_mtp "$A" source))" "$(is "$(props_mtp "$A" source)" --mtp)"
A_ID=$(model_id "$A")
B_ID=$(model_id "$B")
unload "$A_ID" > /dev/null
CODE=$(load "$B_ID")
check "cold load of clone B -> 200 (got $CODE)" "$(is "$CODE" 200)"
check "the flags apply on the cold load too (ctx $(row "$B" ctx), kv $(row "$B" kv), source $(row "$B" src))" "$([ "$(row "$B" ctx)" = 16384 ] && [ "$(row "$B" kv)" = 4 ] && [ "$(row "$B" src)" = --kv-quant ] && echo 1 || echo 0)"
if [ -n "$STREAM_MODEL" ]; then
    unload "$B_ID" > /dev/null
    printf '{ "%s/": { "ssd_budget_gb": 60 } }' "$STREAM_MODEL" > "$(settings_file)"
    post load-model "{\"model\":\"$STREAM_MODEL\"}" > /dev/null
    check "an explicit --mtp refuses the BF16-source streamed load (got $(head -c 160 "$WORK/body.json"))" "$(grep -q expert_streaming_mtp_unsupported "$WORK/body.json" && echo 1 || echo 0)"
else
    echo "  note: no BF16-source checkpoint; the streamed --mtp refusal row is skipped"
fi
sushi_stop 0

# ---------------------------------------------------------------- boot 3
echo "== boot 3: --idle-evict-secs 2"
# 2 s window: the sweep floors at 1 s, so eviction lands ~3 s after the last request.
sushi_boot cold-3 "$A" --model-dir "$ROOT" --no-drafter --no-mtp --idle-evict-secs 2 || exit $?
A_ID=$(model_id "$A")
wait_unloaded() { local i; for i in $(seq 1 40); do [ "$(resident_count)" = 0 ] && return 0; sleep 0.5; done; return 1; }
echo "  cycle 1: load, go idle, evict"
check "a chat answers (200)" "$(is "$(chat "$A_ID")" 200)"
check "the model is resident after a request" "$(is "$(resident_count)" 1)"
if wait_unloaded; then
    check "the sweep logged the eviction" "$(logline "idle-evicting model id=")"
else
    check "model unloaded within 20 s of a 2 s window" 0
fi
echo "    resident bytes released; server RSS $(server_rss_mb) MB"
# /props is the only status route that reaches ensureLoaded: a status poll must not be a load, or the tray's 3 s poll
# retakes the memory the sweep handed back.
curl -fs "$SUSHI_URL/props" > /dev/null || check "/props answers with nothing loaded" 0
curl -fs "$SUSHI_URL/metrics" > /dev/null 2>&1
models > /dev/null
sleep 2
check "status polls left it unloaded" "$(is "$(resident_count)" 0)"
# ...and a status poll must not be USE: a release that restamps the idle clock lets a 1 s watcher pin every model.
check "a chat reloads it for the polling arm" "$(is "$(chat "$A_ID")" 200)"
POLLED_OUT=1
for _ in $(seq 1 40); do
    curl -fs "$SUSHI_URL/props" > /dev/null 2>&1
    sleep 0.5
    [ "$(resident_count)" = 0 ] && { POLLED_OUT=0; break; }
done
check "polling /props did not pin the model for 20 s against a 2 s window" "$(is "$POLLED_OUT" 0)"
# Each reload runs beside unrefcounted readers of the retained CPU state (/v1/models, the pre-load text-gen gate).
BASE_RSS=""
for i in 2 3; do
    PIDS=()
    for _ in 1 2 3; do ( chat "$A_ID" > /dev/null ) & PIDS+=($!); done
    for _ in 1 2 3 4 5 6; do ( curl -fs "$SUSHI_URL/v1/models" > /dev/null ) & PIDS+=($!); done
    OK=1
    for p in "${PIDS[@]}"; do wait "$p" || OK=0; done
    check "cycle $i: every concurrent request succeeded" "$OK"
    check "cycle $i: the server is alive" "$(curl -fs "$SUSHI_URL/health" > /dev/null && echo 1 || echo 0)"
    wait_unloaded || { check "cycle $i: went idle again" 0; break; }
    # Cycle 2 is the warm baseline: the first concurrent cycle pays a one-time warm-up worth hundreds of MB.
    [ -n "$BASE_RSS" ] || BASE_RSS=$(server_rss_mb)
    echo "    cycle $i ok, RSS $(server_rss_mb) MB"
done
if [ -n "$BASE_RSS" ]; then
    GROWTH=$(($(server_rss_mb) - BASE_RSS))
    check "RSS with nothing resident grew +${GROWTH} MB across the last cycle (limit ${RSS_GROWTH_LIMIT_MB}): a reload frees what it replaces" "$([ "$GROWTH" -le "$RSS_GROWTH_LIMIT_MB" ] && echo 1 || echo 0)"
fi
sushi_stop 0

echo "pass=$PASS fail=$FAIL"
[ "$FAIL" = 0 ]
