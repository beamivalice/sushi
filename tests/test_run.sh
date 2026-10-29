#!/bin/bash
# sushi-test: tier=hermetic
# tests/run.sh and tests/lib/boot.sh against a fixture tree and a Python stub `sushi` (serves /health, /props,
# /v1/models and prints `Model ready`), on a private GPU_LOCK_DIR:
#   [1] --check: a valid tree passes; a headerless file, a bad header, an unknown folder, a missing matrix row,
#       a row whose file is gone and an attach script that boots are each refused
#   [2] --list: the plan per pass, tier filters and --only
#   [3] the helper: pack lookup, profiles, boot/stop (lock, END marker, flags, private HOME and cache root), a boot
#       that dies or never gets ready, a crash, a server that ignores TERM, the EXIT trap, attach, log slice, require
#   [4] a run: PASS/SKIP/FAIL/CRASH per script, the reboot after a crash, the halt after a failure, --keep-going
# No model, no GPU, no real sushi.
set -u
cd "$(dirname "$0")/.." || exit 1
TESTS="$PWD/tests"
PASS=0
FAIL=0
check() { if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi; }
eq() { check "$1" "$([ "$2" = "$3" ] && echo 1 || { echo "    got:      $2" >&2; echo "    expected: $3" >&2; echo 0; })"; }
has() { check "$1" "$(printf '%s' "$2" | grep -qF -- "$3" && echo 1 || { echo "    missing: $3" >&2; echo 0; })"; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export GPU_LOCK_DIR="$WORK/lock.d" GPU_LOCK_POLL_S=0.2
LOCK="$PWD/scripts/gpu-lock.sh"
RUN="$TESTS/run.sh"

# ---- fixture tree ----

# mk <root> <rel> <header...>: a script with the header on the line after the shebang
mk() {
    local root="$1" rel="$2" cm="#"
    shift 2
    mkdir -p "$root/$(dirname "$rel")"
    case "$rel" in *.cjs | *.mjs) cm="//" ;; esac
    case "$rel" in
        *.py) printf '#!/usr/bin/env python3\n# sushi-test: %s\n' "$*" > "$root/$rel" ;;
        *.cjs | *.mjs) printf '// sushi-test: %s\n' "$*" > "$root/$rel" ;;
        *) printf '#!/bin/bash\n# sushi-test: %s\n' "$*" > "$root/$rel" ;;
    esac
}

# matrix <root>: a tests/CLAUDE.md row for every runnable script
matrix() {
    local root="$1" rel
    printf '# fixture matrix\n\n| Command | Purpose |\n|---|---|\n' > "$root/CLAUDE.md"
    (cd "$root" && grep -rlE 'sushi-test: tier=(hermetic|host|pack|live)' . | sed 's#^\./##' | sort) | while read -r rel; do
        printf '| `./tests/%s` | pins something |\n' "$rel" >> "$root/CLAUDE.md"
    done
}

fx_list() {
    local r="$1"
    mk "$r" test_hermetic.sh tier=hermetic
    mk "$r" test_node.cjs tier=hermetic needs=node
    mk "$r" test_host.sh tier=host
    mk "$r" test_pack.sh tier=pack
    mk "$r" test_api_a.sh tier=live boot=api
    mk "$r" test_api_b.py tier=live boot=api
    mk "$r" test_sleep_inhibit.sh tier=live boot=api
    mk "$r" test_exact.sh tier=live boot=exact
    mk "$r" test_ram.sh tier=live boot=ram
    mk "$r" test_knobs.sh tier=live boot=knobs
    mk "$r" test_own_each.sh tier=live boot=own pack=each boots=1
    mk "$r" test_own_glm.sh tier=live boot=own pack=glm5 boots=2
    mk "$r" test_own_def.sh tier=live boot=own
    mk "$r" bench_x.sh tier=bench
    mk "$r" tool_y.py tier=tool
    mk "$r" lib/helper.sh tier=lib
    mk "$r" qwen4/test_q_api.sh tier=live boot=api
    mk "$r" qwen4/test_q_own.sh tier=live boot=own boots=2
    mk "$r" qwen4/test_q_host.sh tier=host
    mk "$r" glm5/test_g_host.sh tier=host
    matrix "$r"
}

echo "[1/4] --check"
FX="$WORK/fx"
fx_list "$FX"
out=$(SUSHI_TESTS_DIR="$FX" "$RUN" --check 2>&1)
rc=$?
eq "a valid tree passes --check" "$rc" "0"
has "the pass line counts the scripts" "$out" "20 scripts with headers, no problems"

refused() { # name, expected text, then the mutation as a function name
    local d="$WORK/mut-$1"
    rm -rf "$d"
    cp -R "$FX" "$d"
    "$3" "$d"
    out=$(SUSHI_TESTS_DIR="$d" "$RUN" --check ${4:-} 2>&1)
    rc=$?
    check "--check refuses $1 (rc 1)" "$([ "$rc" = 1 ] && echo 1 || echo 0)"
    has "  and names it" "$out" "$2"
}
m_headerless() { printf '#!/bin/bash\necho hi\n' > "$1/test_noheader.sh"; }
m_nolivebo() { mk "$1" test_x.sh tier=live; }
m_badkey() { mk "$1" test_x.sh tier=host flavour=spicy; }
m_badtier() { mk "$1" test_x.sh tier=slow; }
m_bootnotlive() { mk "$1" bench_z.sh tier=bench boot=api; }
m_packonapi() { mk "$1" test_x.sh tier=live boot=api pack=glm5; }
m_second() { mk "$1" test_x.sh tier=host; printf '# sushi-test: boots=2\n' >> "$1/test_x.sh"; }
m_folder() { mk "$1" other/test_x.sh tier=host; }
m_norow() { mk "$1" test_new.sh tier=host; }
m_rowgone() { printf '| `./tests/test_gone.sh` | gone |\n' >> "$1/CLAUDE.md"; }
m_attachboots() { mk "$1" test_boots.sh tier=live boot=api; printf '"$SUSHI_BIN" --serve --model x\n' >> "$1/test_boots.sh"; matrix "$1"; }
refused "a headerless file" "test_noheader.sh: no '# sushi-test: ...' header" m_headerless
refused "a live script without boot=" "a live script needs boot=" m_nolivebo
refused "an unknown header key" "unknown header key 'flavour'" m_badkey
refused "an unknown tier" "tier 'slow' is not" m_badtier
refused "boot= on a bench" "boot= is for tier=live only" m_bootnotlive
refused "pack= on an api script" "pack= is for boot=own and tier=pack only" m_packonapi
refused "a second line that is not needs=" "a second header line may only hold needs=" m_second
refused "an unknown folder" "folder 'other' is not" m_folder
refused "a script with no matrix row" "test_new.sh: no row in tests/CLAUDE.md" m_norow
refused "a row whose file is gone" "test_gone.sh: a tests/CLAUDE.md row names a file that is gone" m_rowgone
refused "an attach script that launches sushi" "test_boots.sh: boot=api is an attach script but launches sushi" m_attachboots
d="$WORK/mut-nomatrix"
cp -R "$FX" "$d"
m_norow "$d"
SUSHI_TESTS_DIR="$d" "$RUN" --check --no-matrix > /dev/null 2>&1
eq "--no-matrix waives the matrix rules" "$?" "0"

echo "[2/4] --list"
P_QWEN4='pass qwen4
  host qwen4/test_q_host.sh
  boot api: --metrics --log-level debug --max-concurrent 4
    qwen4/test_q_api.sh
    test_api_a.sh
    test_api_b.py
    test_sleep_inhibit.sh
  boot exact: --metrics --log-level debug --max-concurrent 4 --prefix-cache-entries 0 --prefix-cache-mem off --prefix-cache-disk off --mtp-min-depth 3 --mtp-max-depth 3
    test_exact.sh
  boot ram: --metrics --log-level debug --max-concurrent 4 --prefix-cache-entries 4 --prefix-cache-mem 2GB --prefix-cache-disk off
    test_ram.sh
  own qwen4/test_q_own.sh pack=default boots=2
  own test_own_each.sh pack=each boots=1
  boots: 3 shared + 3 own'
P_SHARED='pass shared
  hermetic test_hermetic.sh
  hermetic test_node.cjs needs=node
  host test_host.sh
  pack test_pack.sh
  boot knobs: --metrics --log-level debug --think medium --reasoning-budget 1 --think-penalty 2 --prefix-cache-entries 0 --prefix-cache-mem off --prefix-cache-disk off
    test_knobs.sh
  own test_own_def.sh pack=default boots=1
  own test_own_glm.sh pack=glm5 boots=2
  boots: 1 shared + 3 own'
got=$(SUSHI_TESTS_DIR="$FX" "$RUN" --list --tier all --model qwen4)
eq "--tier all --model qwen4: folder first, root alphabetical, sleep_inhibit last, boots in order" "$got" "$P_QWEN4"
got=$(SUSHI_TESTS_DIR="$FX" "$RUN" --list --tier all --model shared)
eq "--model shared: non-live root scripts, knobs, own scripts of another pack" "$got" "$P_SHARED"
got=$(SUSHI_TESTS_DIR="$FX" "$RUN" --list)
eq "the default is hermetic,host over every pass" "$got" "pass qwen4
  host qwen4/test_q_host.sh
  boots: 0 shared + 0 own
pass glm5
  host glm5/test_g_host.sh
  boots: 0 shared + 0 own
pass shared
  hermetic test_hermetic.sh
  hermetic test_node.cjs needs=node
  host test_host.sh
  boots: 0 shared + 0 own"
got=$(SUSHI_TESTS_DIR="$FX" "$RUN" --list --tier live --model glm5)
eq "a model with no folder scripts still gets the root attach and each scripts" "$got" "pass glm5
  boot api: --metrics --log-level debug --max-concurrent 4
    test_api_a.sh
    test_api_b.py
    test_sleep_inhibit.sh
  boot exact: --metrics --log-level debug --max-concurrent 4 --prefix-cache-entries 0 --prefix-cache-mem off --prefix-cache-disk off
    test_exact.sh
  boot ram: --metrics --log-level debug --max-concurrent 4 --prefix-cache-entries 4 --prefix-cache-mem 2GB --prefix-cache-disk off
    test_ram.sh
  own test_own_each.sh pack=each boots=1
  boots: 3 shared + 1 own"
got=$(SUSHI_TESTS_DIR="$FX" "$RUN" --list --tier live --model qwen4 --only 'test_a*')
eq "--only filters by glob" "$got" "pass qwen4
  boot api: --metrics --log-level debug --max-concurrent 4
    test_api_a.sh
    test_api_b.py
  boots: 1 shared + 0 own"
got=$(SUSHI_TESTS_DIR="$FX" "$RUN" --list --tier live --model dsv41 | sed -n 2p)
eq "a streamed DeepSeek serves one slot on a --ssd-budget-gb" "$got" "  boot api: --metrics --log-level debug --ssd-budget-gb 96 --max-concurrent 1"
out=$(SUSHI_TESTS_DIR="$FX" "$RUN" --list --tier warm 2>&1)
has "a bad --tier is refused" "$out" "--tier 'warm' is not"

echo "[3/4] the boot helper"
STUB="$WORK/stub-sushi"
cat > "$STUB" <<'PY'
#!/usr/bin/env python3
import json, os, signal, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
mode = os.environ.get("STUB_MODE", "")
print("argv: " + " ".join(args), flush=True)
print("home: " + os.environ.get("HOME", ""), flush=True)
print("cache: " + os.environ.get("SUSHI_PREFIX_CACHE_DIR", ""), flush=True)
if mode == "die":
    print("fatal: the stub refuses to boot", flush=True)
    sys.exit(1)
if mode == "ignoreterm":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        bodies = {
            "/health": b"ok",
            "/props": json.dumps({"batching": {"supported": True}, "settings": {
                "mtp": {"loaded": True}, "vision": {"loaded": False}, "drafter": "none"}}).encode(),
            "/v1/models": json.dumps({"data": [{"capabilities": ["chat", "reasoning"]}]}).encode(),
        }
        body = bodies.get(self.path)
        self.send_response(200 if body is not None else 404)
        self.end_headers()
        self.wfile.write(body or b"")

    def log_message(self, *a):
        pass


srv = HTTPServer(("127.0.0.1", int(args[args.index("--port") + 1])), H)
if mode != "noready":
    print("Model ready.", flush=True)
srv.serve_forever()
PY
chmod +x "$STUB"
mkdir -p "$WORK/models/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw" "$WORK/models/stubpack"
echo '{"model_type":"qwen4_exp","expert_quant":{"format":"stq"}}' > "$WORK/models/stubpack/config.json"
echo '{"model_type":"qwen4_exp","expert_quant":{"format":"stq"}}' > "$WORK/models/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw/config.json"

SCEN="$WORK/scenario.sh"
cat > "$SCEN" <<'SH'
#!/bin/bash
. "$TESTS/lib/boot.sh"
pack="$WORK/models/stubpack"
lockhead() { "$SUSHI_ROOT/scripts/gpu-lock.sh" status | head -1 | cut -d' ' -f1; }
case "$1" in
boot_stop)
    sushi_boot l1 "$pack" --foo bar
    echo "boot_rc=$?"
    echo "lock_during=$(lockhead)"
    echo "arch=$SUSHI_ARCH"
    echo "health=$(curl -sf "$SUSHI_URL/health")"
    alive=0
    sushi_alive && alive=1
    echo "alive=$alive"
    echo "profile=$SUSHI_PROFILE"
    pid=$SUSHI_PID
    log=$SUSHI_LOG
    sushi_stop 0
    echo "stop_rc=$?"
    echo "lock_after=$(lockhead)"
    echo "end=$(cat "$SUSHI_RUN_DIR/l1.END")"
    gone=1
    kill -0 "$pid" 2> /dev/null && gone=0
    echo "pid_gone=$gone"
    echo "argv=$(grep -m1 '^argv:' "$log")"
    echo "home=$(grep -m1 '^home:' "$log")"
    echo "cache=$(grep -m1 '^cache:' "$log")"
    echo "pid_cleared=${SUSHI_PID:-none}"
    ;;
crash)
    sushi_boot l2 "$pack" > /dev/null
    kill -KILL "$SUSHI_PID"
    sleep 0.3
    sushi_alive && echo "alive=1" || echo "alive=0"
    sushi_stop 0
    echo "stop_rc=$?"
    echo "lock_after=$(lockhead)"
    echo "end=$(cat "$SUSHI_RUN_DIR/l2.END")"
    ;;
die)
    STUB_MODE=die sushi_boot l3 "$pack"
    echo "boot_rc=$?"
    echo "lock_after=$(lockhead)"
    echo "end=$(cat "$SUSHI_RUN_DIR/l3.END")"
    ;;
noready)
    STUB_MODE=noready SUSHI_BOOT_TIMEOUT_S=2 sushi_boot l4 "$pack"
    echo "boot_rc=$?"
    echo "lock_after=$(lockhead)"
    ;;
ignoreterm)
    STUB_MODE=ignoreterm sushi_boot l5 "$pack" > /dev/null
    pid=$SUSHI_PID
    SUSHI_STOP_WAIT_S=1 sushi_stop 0
    echo "end=$(cat "$SUSHI_RUN_DIR/l5.END")"
    gone=1
    kill -0 "$pid" 2> /dev/null && gone=0
    echo "pid_gone=$gone"
    ;;
trap_exit)
    trap 'echo prev_trap_ran' EXIT
    sushi_boot l6 "$pack" > /dev/null || exit 9
    echo "pid=$SUSHI_PID"
    exit 7
    ;;
attach)
    sushi_boot l7 "$pack" > /dev/null || exit 9
    sushi_attach
    echo "nonce_head=${SUSHI_NONCE:0:1}"
    echo "offset_is_size=$([ "$SUSHI_LOG_OFFSET" = "$(wc -c < "$SUSHI_LOG" | tr -d ' ')" ] && echo 1 || echo 0)"
    echo "line-after-attach" >> "$SUSHI_LOG"
    echo "since=$(sushi_log_since | tr '\n' '|')"
    for cap in mtp batching spec thinking stq; do
        ( sushi_require $cap > /dev/null; ) ; echo "require_$cap=$?"
    done
    for cap in vision video; do
        out=$( ( sushi_require $cap ) ); echo "require_$cap=$?"
        echo "skip_line_$cap=$out"
    done
    ( sushi_require bogus ) > /dev/null 2>&1; echo "require_bogus=$?"
    ( unset SUSHI_URL; sushi_attach ) > /dev/null 2>&1; echo "attach_unset=$?"
    python3 - "$TESTS/lib" <<'PY'
import os, sys
sys.path.insert(0, sys.argv[1])
import sushi_test as t
n = t.attach()
with open(os.environ["SUSHI_LOG"], "a") as f:
    f.write("py-line\n")
print("py_nonce_head=" + n[0])
print("py_since=" + t.log_since().strip())
t.require("mtp")
print("py_mtp=ok")
try:
    t.require("vision")
except SystemExit as e:
    print("py_vision_exit=%s" % e.code)
PY
    sushi_stop 0
    ;;
gone)
    sushi_boot l8 "$pack" > /dev/null || exit 9
    kill -TERM "$SUSHI_PID"
    while kill -0 "$SUSHI_PID" 2> /dev/null; do sleep 0.1; done
    SUSHI_EXPECT_GONE=1 sushi_stop 0
    echo "stop_rc=$?"
    echo "end=$(cat "$SUSHI_RUN_DIR/l8.END")"
    echo "lock_after=$(lockhead)"
    ;;
pyboot)
    python3 - "$TESTS/lib" "$pack" <<'PY'
import os, subprocess, sys
sys.path.insert(0, sys.argv[1])
import sushi_boot as b
with b.Server("p1", model=sys.argv[2], flags=["--foo", "bar"]) as s:
    s.wait_ready()
    st, _ = b.request(s.base, "/props")
    print("props=%s" % st)
    print("lock_during=%s" % subprocess.check_output([str(b.LOCK), "status"], text=True).split()[0])
    print("argv=%s" % [l for l in s.log_text().splitlines() if l.startswith("argv:")][0])
    print("home=%s" % [l for l in s.log_text().splitlines() if l.startswith("home:")][0])
print("end=%s" % (b.run_dir() / "p1.END").read_text().strip())
print("lock_after=%s" % subprocess.check_output([str(b.LOCK), "status"], text=True).split()[0])
print("pack_skip=%s" % subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import sushi_boot as b; b.pack('glm5')", sys.argv[1]],
                                      capture_output=True, text=True, env={**os.environ, "SUSHI_MODELS_DIR": os.environ["WORK"] + "/models"}).returncode)
PY
    ;;
pack)
    export SUSHI_MODELS_DIR="$WORK/models"
    echo "qwen4=$(sushi_pack qwen4)"
    sushi_pack glm5 > /dev/null 2> "$WORK/pack.err"
    echo "glm5_rc=$?"
    echo "glm5_err=$(cat "$WORK/pack.err")"
    mkdir -p "$WORK/elsewhere"
    echo "override=$(SUSHI_TEST_PACK_GLM5="$WORK/elsewhere" sushi_pack glm5)"
    echo "alias=$(SUSHI_TEST_PACK_MIMO2_SRC="$WORK/elsewhere" sushi_pack mimo2-src)"
    echo "abs=$(sushi_pack "$WORK/elsewhere")"
    ;;
esac
SH
scen() { env TESTS="$TESTS" WORK="$WORK" SUSHI_BIN="$STUB" SUSHI_RUN_DIR="$WORK/run" /bin/bash "$SCEN" "$@"; }
val() { printf '%s\n' "$1" | grep -m1 "^$2=" | cut -d= -f2-; }
lock_free() { [ "$("$LOCK" status | head -1)" = free ] && echo 1 || echo 0; }

out=$(scen boot_stop 2> "$WORK/boot_stop.err")
eq "boot: returns 0" "$(val "$out" boot_rc)" "0"
eq "boot: the lock is held as tests-<label> while the server runs" "$(val "$out" lock_during)" "tests-l1"
eq "boot: the arch is read from the pack's config.json" "$(val "$out" arch)" "qwen4_exp"
eq "boot: /health answers" "$(val "$out" health)" "ok"
eq "boot: sushi_alive is true while it runs" "$(val "$out" alive)" "1"
eq "boot: the profile defaults to own" "$(val "$out" profile)" "own"
eq "stop: returns 0 on a clean stop" "$(val "$out" stop_rc)" "0"
eq "stop: the lock is released" "$(val "$out" lock_after)" "free"
eq "stop: the END marker carries the status" "$(val "$out" end)" "exit=0 server=clean"
eq "stop: the server process is gone" "$(val "$out" pid_gone)" "1"
eq "stop: the recorded PID is cleared" "$(val "$out" pid_cleared)" "none"
got=$(val "$out" argv)
got=$(printf '%s' "$got" | sed -E 's/--port [0-9]+/--port P/')
eq "boot: flags are --serve --model <pack> --host 127.0.0.1 --port <p> --log-file off then the caller's" "$got" \
    "argv: --serve --model $WORK/models/stubpack --host 127.0.0.1 --port P --log-file off --foo bar"
eq "boot: a private HOME under the run dir" "$(val "$out" home)" "home: $WORK/run/l1.home"
eq "boot: a private prefix-cache root under the run dir" "$(val "$out" cache)" "cache: $WORK/run/l1.kv-cache"
check "boot: the log lands at <run>/<label>.log" "$([ -s "$WORK/run/l1.log" ] && echo 1 || echo 0)"

out=$(scen crash 2> "$WORK/crash.err")
eq "crash: sushi_alive is false" "$(val "$out" alive)" "0"
eq "crash: sushi_stop returns 1" "$(val "$out" stop_rc)" "1"
eq "crash: the END marker says crash" "$(val "$out" end)" "exit=0 server=crash"
eq "crash: the lock is released" "$(val "$out" lock_after)" "free"
has "crash: reported by name on stderr" "$(cat "$WORK/crash.err")" "CRASH: server 'l2'"

out=$(scen die 2> "$WORK/die.err")
eq "a boot that dies returns 1" "$(val "$out" boot_rc)" "1"
eq "  and releases the lock" "$(val "$out" lock_after)" "free"
has "  and shows the log tail" "$(cat "$WORK/die.err")" "the stub refuses to boot"
has "  and writes the END marker" "$(val "$out" end)" "exit=1"

out=$(scen noready 2> "$WORK/noready.err")
eq "a boot never ready within the bound returns 1" "$(val "$out" boot_rc)" "1"
eq "  and releases the lock" "$(val "$out" lock_after)" "free"
has "  and names the bound" "$(cat "$WORK/noready.err")" "not ready after 2s"

out=$(scen ignoreterm 2> /dev/null)
eq "a server that ignores TERM is killed after the wait" "$(val "$out" end)" "exit=0 server=killed"
eq "  and is gone" "$(val "$out" pid_gone)" "1"

out=$(scen gone 2> /dev/null)
eq "a server the script stopped on purpose is not a crash when SUSHI_EXPECT_GONE is set" "$(val "$out" stop_rc)" "0"
eq "  its END marker says gone" "$(val "$out" end)" "exit=0 server=gone"
eq "  and the lock is released" "$(val "$out" lock_after)" "free"

out=$(scen pyboot 2> /dev/null)
eq "python Server: ready and /props answers" "$(val "$out" props)" "200"
eq "  the lock is held as tests-<label>" "$(val "$out" lock_during)" "tests-p1"
has "  the flags reach the server" "$(val "$out" argv)" "--model $WORK/models/stubpack"
has "  and the caller's flags come last" "$(val "$out" argv)" "--log-file off --foo bar"
eq "  a private HOME under the run dir" "$(val "$out" home)" "home: $WORK/run/p1.home"
eq "  stop writes the END marker and releases the lock" "$(val "$out" end)/$(val "$out" lock_after)" "exit=0 server=clean/free"
eq "  pack() exits 3 (SKIP) on a missing pack" "$(val "$out" pack_skip)" "3"

out=$(scen trap_exit 2> /dev/null)
rc=$?
pid=$(val "$out" pid)
eq "the EXIT trap keeps the script's own status" "$rc" "7"
has "  runs the script's earlier EXIT trap" "$out" "prev_trap_ran"
eq "  releases the lock" "$(lock_free)" "1"
eq "  writes the END marker with that status" "$(cat "$WORK/run/l6.END")" "exit=7 server=clean"
eq "  stops the server" "$(kill -0 "$pid" 2> /dev/null && echo 0 || echo 1)" "1"

out=$(scen attach 2> /dev/null)
eq "attach: the nonce is set" "$(val "$out" nonce_head)" "n"
eq "attach: the log offset is the log size" "$(val "$out" offset_is_size)" "1"
eq "attach: sushi_log_since is only what came after" "$(val "$out" since)" "line-after-attach|"
for cap in mtp batching spec thinking stq; do eq "require $cap passes on the stub" "$(val "$out" require_$cap)" "0"; done
for cap in vision video; do
    eq "require $cap exits 3 (SKIP)" "$(val "$out" require_$cap)" "3"
    has "  with one line naming it" "$(val "$out" skip_line_$cap)" "the boot lacks $cap (qwen4_exp)"
done
eq "require of an unknown capability is an error (1)" "$(val "$out" require_bogus)" "1"
eq "attach without SUSHI_URL exits 1" "$(val "$out" attach_unset)" "1"
eq "python twin: the nonce" "$(val "$out" py_nonce_head)" "n"
eq "python twin: log_since slices at attach" "$(val "$out" py_since)" "py-line"
eq "python twin: require passes and SKIPs with 3" "$(val "$out" py_mtp)/$(val "$out" py_vision_exit)" "ok/3"
eq "the lock is free after the attach scenario" "$(lock_free)" "1"

out=$(scen pack 2> /dev/null)
eq "sushi_pack: the default pack under SUSHI_MODELS_DIR" "$(val "$out" qwen4)" "$WORK/models/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw"
eq "sushi_pack: a missing pack is SKIP (3)" "$(val "$out" glm5_rc)" "3"
has "  with one line naming it" "$(val "$out" glm5_err)" "SKIP: pack 'glm5' missing"
eq "sushi_pack: SUSHI_TEST_PACK_<MODEL> overrides" "$(val "$out" override)" "$WORK/elsewhere"
eq "sushi_pack: an alias maps to its own variable" "$(val "$out" alias)" "$WORK/elsewhere"
eq "sushi_pack: an absolute path is taken as it is" "$(val "$out" abs)" "$WORK/elsewhere"

PROF="$WORK/profiles.sh"
printf '. "%s/lib/profiles.sh"\nsushi_profile_flags "$@"\n' "$TESTS" > "$PROF"
eq "profile exact on glm5 carries no MTP depth" "$(/bin/bash "$PROF" exact glm5)" "--metrics --log-level debug --max-concurrent 4 --prefix-cache-entries 0 --prefix-cache-mem off --prefix-cache-disk off"
eq "profile exact accepts an arch name" "$(/bin/bash "$PROF" exact mimo_v2)" "$(/bin/bash "$PROF" exact mimo2)"
eq "SUSHI_TEST_DSV41_BUDGET sets the streamed budget" "$(SUSHI_TEST_DSV41_BUDGET=64 /bin/bash "$PROF" ram dsv41)" "--metrics --log-level debug --ssd-budget-gb 64 --max-concurrent 1 --prefix-cache-entries 4 --prefix-cache-mem 2GB --prefix-cache-disk off"
/bin/bash "$PROF" api nope > /dev/null 2>&1
eq "an unknown model is an error" "$?" "1"
printf '. "%s/lib/profiles.sh"\nsushi_profile_settings "$@"\n' "$TESTS" > "$WORK/settings.sh"
eq "profile knobs writes a settings file keyed by the pack and naming the bias file" \
    "$(/bin/bash "$WORK/settings.sh" knobs /pack/q "$WORK/knobs.json" > /dev/null; python3 -I -c 'import json, sys; d = json.load(open(sys.argv[1])); print(list(d), d["/pack/q"]["logit_bias_file"].endswith("/lib/knobs-bias.json"))' "$WORK/knobs.json")" \
    "['/pack/q'] True"
eq "a profile without settings writes nothing" "$(/bin/bash "$WORK/settings.sh" api /pack/q "$WORK/api.json"; ls "$WORK/api.json" 2> /dev/null)" ""

echo "[4/4] a run"
RFX="$WORK/rfx"
mkdir -p "$RFX"
L='. "$SUSHI_TEST_LIB/boot.sh"'
put() { # rel, header, body lines...
    local rel="$1" hdr="$2"
    shift 2
    mk "$RFX" "$rel" "$hdr"
    printf '%s\n' "$L" "$@" >> "$RFX/$rel"
}
put qwen4/test_a_ok.sh "tier=live boot=api" 'sushi_attach' 'curl -sf "$SUSHI_URL/health" > /dev/null || exit 1' 'sushi_require mtp' '[ -n "$SUSHI_NONCE" ]'
put qwen4/test_b_skip.sh "tier=live boot=api" 'sushi_attach' 'sushi_require vision'
put qwen4/test_c_kill.sh "tier=live boot=api" 'sushi_attach' 'kill -KILL "$SUSHI_PID"' 'sleep 0.3'
put qwen4/test_d_after.sh "tier=live boot=api" 'sushi_attach' 'curl -sf "$SUSHI_URL/health" > /dev/null || exit 1'
put qwen4/test_e_fail.sh "tier=live boot=exact" 'sushi_attach' 'echo "the exact arm is red"' 'exit 1'
put qwen4/test_f_ram.sh "tier=live boot=ram" 'sushi_attach'
put qwen4/test_g_own.sh "tier=live boot=own" 'pack=$(sushi_pack "$SUSHI_PASS_MODEL") || exit $?' 'sushi_boot g-own "$pack" --x || exit 1' 'curl -sf "$SUSHI_URL/health" > /dev/null || exit 1'
put qwen4/test_h_host.sh "tier=host" 'exit 0'
put test_hermetic.sh "tier=hermetic" 'exit 0'
matrix "$RFX"
runrun() { # out dir, args...
    local o="$1"
    shift
    env SUSHI_TESTS_DIR="$RFX" SUSHI_BIN="$STUB" SUSHI_TEST_LIB="$TESTS/lib" SUSHI_MODELS_DIR="${RUN_MODELS:-$WORK/models}" \
        HOME="$WORK/home" "$RUN" --no-build --out "$o" "$@" > "$o.stdout" 2>&1
}
eq "the fixture tree passes --check" "$(SUSHI_TESTS_DIR="$RFX" "$RUN" --check > /dev/null 2>&1; echo $?)" "0"

runrun "$WORK/out1" --tier host,live --model qwen4 --keep-going
rc=$?
S="$WORK/out1/summary.txt"
eq "a failure and a crash make the run exit 1" "$rc" "1"
has "PASS on an attach script" "$(cat "$S")" "PASS	qwen4	qwen4/test_a_ok.sh"
has "SKIP (exit 3) on a missing capability" "$(cat "$S")" "SKIP	qwen4	qwen4/test_b_skip.sh"
has "CRASH when the server died under a script" "$(cat "$S")" "CRASH	qwen4	qwen4/test_c_kill.sh"
has "the boot is redone once and the next script PASSes on it" "$(cat "$S")" "PASS	qwen4	qwen4/test_d_after.sh"
has "FAIL on exit 1" "$(cat "$S")" "FAIL	qwen4	qwen4/test_e_fail.sh"
has "a failure shows the script's log tail" "$(cat "$WORK/out1.stdout")" "the exact arm is red"
has "host tier runs before the boots" "$(cat "$S")" "PASS	qwen4	qwen4/test_h_host.sh"
has "--keep-going runs the ram boot after the failure" "$(cat "$S")" "PASS	qwen4	qwen4/test_f_ram.sh"
has "  and the own script, which boots through the helper" "$(cat "$S")" "PASS	qwen4	qwen4/test_g_own.sh"
eq "the own script's boot wrote its END marker" "$(cat "$WORK/out1/g-own.END")" "exit=0 server=clean"
has "the summary stamps the commit and the binary" "$(cat "$S")" "commit: "
has "the summary lists the boots" "$(cat "$S")" "qwen4	api"
eq "the END marker holds the exit status" "$(cat "$WORK/out1/END")" "exit=1"
eq "the lock is free after the run" "$(lock_free)" "1"
check "no server is left running" "$(pgrep -f "$STUB" > /dev/null && echo 0 || echo 1)"
eq "every boot wrote its END marker" "$(ls "$WORK/out1" | grep -c '\.END$')" "4"

runrun "$WORK/out2" --tier live --model qwen4
rc=$?
S="$WORK/out2/summary.txt"
eq "without --keep-going a crash exits 1" "$rc" "1"
has "  the boot's scripts all ran" "$(cat "$S")" "PASS	qwen4	qwen4/test_d_after.sh"
check "  and the pass stopped at the end of that boot: exact, ram and own did not run" "$(grep -qE 'test_e_fail|test_f_ram|test_g_own' "$S" && echo 0 || echo 1)"
has "  the summary says so" "$(cat "$S")" "stopped after a failure"
eq "  the lock is free" "$(lock_free)" "1"

runrun "$WORK/out6" --tier live --model qwen4 --only 'test_[ef]*'
rc=$?
S="$WORK/out6/summary.txt"
eq "a FAIL exits 1" "$rc" "1"
has "FAIL on exit 1" "$(cat "$S")" "FAIL	qwen4	qwen4/test_e_fail.sh"
has "a failure shows the script's log tail" "$(cat "$WORK/out6.stdout")" "the exact arm is red"
check "the ram boot after the failing exact boot did not run" "$(grep -q 'test_f_ram' "$S" && echo 0 || echo 1)"
has "the END marker holds the exit status" "$(cat "$WORK/out6/END")" "exit=1"

runrun "$WORK/out3" --tier live,hermetic --model qwen4 --only 'test_a*'
eq "a green selection exits 0" "$?" "0"
has "  and reports its total" "$(cat "$WORK/out3/summary.txt")" "PASS=1 FAIL=0 SKIP=0 CRASH=0"

runrun "$WORK/out4" --tier live --model qwen4 --only 'test_b*'
eq "a selection that only SKIPs exits 0" "$?" "0"

RUN_MODELS="$WORK/nopacks" runrun "$WORK/out5" --tier live --model qwen4
eq "a pass whose pack is missing SKIPs every boot script and exits 0" "$?" "0"
has "  naming the pack" "$(cat "$WORK/out5.stdout")" "SKIP: pack 'qwen4' missing"
check "  without booting anything" "$([ ! -e "$WORK/out5/qwen4-api.END" ] && echo 1 || echo 0)"

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
