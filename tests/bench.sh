#!/bin/bash
# bench.sh — the release bench for ONE model: boots a sushi server, drives it with `sushi bench`, stops it.
#
#   ./tests/bench.sh --version 1.4.0 <pack> [-- <sushi serve flags>]
#   ./tests/bench.sh --tag try1 <pack> -- --ssd-budget-gb 108 --ctx-size 1000000 --max-tokens 128000
#
# <pack> is a directory, or a name under ${SUSHI_MODELS_DIR:-$HOME/.sushi/models}. Everything after `--` goes to
# the server; the shipping defaults (speculative mode, kv8) stay on unless a flag changes them. The protocol,
# the order rule and the report format are in docs/process-measurement.md (release bench).
#
# What it takes on for the box rules: a clean tree and a fresh ReleaseFast build, the QUIET marker, AC power, the GPU
# lock for the whole model, fans at max with an idle under 50 C (at most 3 min) `taskpolicy -a`,
# and a private SSD prefix cache. It refuses to start while another sushi, sushi-repack or sushi-grid runs.
#
# Another engine: this script does not drive one. Start it yourself and run llmprobe by hand,
#   npx -y llmprobe@latest <host:port> -m <id> --bench-only
# and name the engine beside every number.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$ROOT/zig-out/bin/sushi"
ZIG="$ROOT/.zig-toolchain/zig"
LOCK="$ROOT/scripts/gpu-lock.sh"
TAG="$(date +%Y%m%d-%H%M%S)"
VERSION=""
PORT=11250
DECODE_TOKENS=256
PACK=""
SERVER_FLAGS=()

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
die() { echo "bench.sh: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)       VERSION="${2:?--version needs a value}"; shift 2 ;;
        --tag)           TAG="${2:?--tag needs a value}"; shift 2 ;;
        --port)          PORT="${2:?--port needs a value}"; shift 2 ;;
        --decode-tokens) DECODE_TOKENS="${2:?--decode-tokens needs a value}"; shift 2 ;;
        -h|--help)       usage ;;
        --)              shift; SERVER_FLAGS=("$@"); break ;;
        -*)              die "unknown flag: $1 (try --help)" ;;
        *)               [[ -z "$PACK" ]] || die "one pack per run, got a second: $1"; PACK="$1"; shift ;;
    esac
done
[[ -n "$PACK" ]] || die "name the pack (try --help)"
[[ "$PACK" == /* || "$PACK" == ./* ]] || PACK="${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/$PACK"
[[ -d "$PACK" ]] || die "no pack at $PACK"
NAME="$(basename "$PACK")"

OUT="$HOME/.sushi/runs/bench-$TAG"
mkdir -p "$OUT"
QUIET="$HOME/.sushi/runs/QUIET"
pid=""
fans=""
have_lock=0
owner="bench-$TAG"

cleanup() {
    local rc=$?
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null
    [[ -n "$fans" ]] && kill "$fans" 2>/dev/null
    fanctl auto >/dev/null 2>&1
    [[ "$have_lock" -eq 1 ]] && "$LOCK" release "$owner" >/dev/null 2>&1
    rm -f "$QUIET"
    echo "END bench $NAME rc=$rc" >> "$OUT/END"
}
trap cleanup EXIT

# ── Refusals ──
for p in sushi sushi-repack sushi-grid; do
    pgrep -x "$p" >/dev/null && die "a $p process is running; the bench needs the box to itself"
done
pmset -g batt | grep -q "AC Power" || die "not on AC power: the release bench times on AC only"
git -C "$ROOT" diff --quiet HEAD || die "the tree is dirty: commit first, the stamp names the commit"

# ── Build, then the quiet window ──
until [ ! -e "$QUIET" ]; do sleep 10; done
(cd "$ROOT" && taskpolicy -a "$ZIG" build -Doptimize=ReleaseFast) || die "build failed"
touch "$QUIET"
while pgrep -x zig >/dev/null || pgrep -x clang >/dev/null; do sleep 5; done

COMMIT="$(git -C "$ROOT" rev-parse --short=8 HEAD)"
SHA="$(shasum -a 256 "$BINARY" | cut -c1-16)"
MTIME="$(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' "$BINARY")"
CHIP="$(sysctl -n machdep.cpu.brand_string) $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
STAMP="Commit \`$COMMIT\`, binary SHA-256 \`$SHA\` (built $MTIME). Hardware: $CHIP, AC power, ReleaseFast, \`taskpolicy -a\`, GPU lock \`$owner\`, $(date '+%Y-%m-%d')."
STAMP="$STAMP\\nServer flags: ${SERVER_FLAGS[*]:-defaults}. Private SSD prefix cache, one model per server, fans at max with an idle under 50 C before the boot."

# ── Lock, then cool ──
"$LOCK" acquire "$owner" || die "could not take the GPU lock"
have_lock=1
fanctl max --ttl 900 >/dev/null 2>&1 || echo "bench.sh: fanctl max failed; the run is not thermally controlled" >&2
# The fan daemon grants at most 15 min of max whatever the ttl asked, then goes back to auto: renew it for the whole run.
( while sleep 300; do fanctl max --ttl 900 >/dev/null 2>&1; done ) &
fans=$!
hottest() {
    fanctl status 2>/dev/null | python3 -c 'import json,sys; print(max(t["celsius"] for t in json.load(sys.stdin)["topTemps"]))' 2>/dev/null || echo 99
}
for _ in $(seq 1 36); do
    t="$(hottest)"
    [[ "${t%.*}" -lt 50 ]] && break
    sleep 5
done
echo "bench.sh: die at ${t%.*} C at the boot" | tee -a "$OUT/bench.log"
# The GPU clock is read by `sushi bench` itself, which aborts a throttled request; the pressure level is only logged.
pressure="$(sudo -n powermetrics --samplers thermal -i 1000 -n 1 2>/dev/null | sed -n 's/^Current pressure level: *//p')"
if [[ -z "$pressure" ]]; then
    echo "bench.sh: thermal pressure not read (no passwordless sudo -n powermetrics); the GPU clock is not monitored" | tee -a "$OUT/bench.log"
else
    echo "bench.sh: thermal pressure $pressure at the boot" | tee -a "$OUT/bench.log"
fi

# ── Boot, run, stop ──
export SUSHI_PREFIX_CACHE_DIR="$OUT/cache"
rm -rf "$SUSHI_PREFIX_CACHE_DIR"
mkdir -p "$SUSHI_PREFIX_CACHE_DIR"
taskpolicy -a "$BINARY" --serve --model "$PACK" --port "$PORT" ${SERVER_FLAGS[@]+"${SERVER_FLAGS[@]}"} >"$OUT/server.log" 2>&1 &
pid=$!
up=0
for _ in $(seq 1 1800); do
    kill -0 "$pid" 2>/dev/null || break
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
    sleep 1
done
[[ "$up" -eq 1 ]] || die "the server never came up; see $OUT/server.log"

REPORT=()
if [[ -n "$VERSION" ]]; then
    mkdir -p "$ROOT/docs/bench/v$VERSION"
    REPORT=(--report "$ROOT/docs/bench/v$VERSION/$NAME.md" --version "$VERSION")
fi
taskpolicy -a "$BINARY" bench --url "127.0.0.1:$PORT" --out "$OUT" --log "$OUT/server.log" \
    --decode-tokens "$DECODE_TOKENS" --stamp "$STAMP" ${REPORT[@]+"${REPORT[@]}"} 2>&1 | tee -a "$OUT/bench.log"
rc=${PIPESTATUS[0]}
echo "bench.sh: sushi bench rc=$rc; raw samples and the server log are in $OUT"
exit "$rc"
