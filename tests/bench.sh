#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# bench.sh — the performance bench. llmprobe measures; this drives sushi.
#
# One `llmprobe --bench-only` run per model gives the decode/prefill/TTFT
# medians AND the context ladder. The numbers go into benchmarks.md by hand —
# there is no CSV, no chart pipeline and no engine matrix here any more.
#
#   ./tests/bench.sh                                # the perf gate (Sushi-4bpw), then MiMo
#   ./tests/bench.sh --only sushi-4bpw              # one row
#   ./tests/bench.sh --url 127.0.0.1:1234 -m <id>   # a server someone else started
#   ./tests/bench.sh --full                         # median of 3 per rung, to 64k
#   ./tests/bench.sh --sampled                      # also a sampled cell (llmprobe's creative preset: T1.0, top_p 0.95)
#
# Each cell is sushi at its FASTEST: MTP is forced on where the checkpoint
# carries a head, so a model-settings.json cannot turn it off. The mode
# that actually engaged is printed beside the number, from the server's own
# log — a mode that silently stops engaging shows up as a bare cell.
#
# Comparing against another engine: start it yourself (LM Studio, oMLX, MTPLX,
# llama-server, whatever), then point --url at it. Same protocol, same probe,
# one less thing in this script to keep in sync.
#
# Requirements: node (npx), curl, sushi built ReleaseFast (Debug is 2-4x
# slower = a fake regression).
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ONLY=""
FULL=0
SAMPLED=0
URL=""
URL_MODEL=""
TAG="$(date +%Y%m%d-%H%M%S)"
SETTLE="${SETTLE:-20}"

BINARY="${BINARY:-$ROOT/zig-out/bin/sushi}"
LLMPROBE="${LLMPROBE:-npx -y llmprobe@latest}"
PORT=11250

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)    ONLY="$2"; shift 2 ;;
        --url)     URL="$2"; shift 2 ;;
        -m|--model) URL_MODEL="$2"; shift 2 ;;
        --full)    FULL=1; shift ;;
        --sampled) SAMPLED=1; shift ;;
        --tag)     TAG="$2"; shift 2 ;;
        --settle)  SETTLE="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown flag: $1 (try --help)" >&2; exit 1 ;;
    esac
done

# Reports live outside the repo, somewhere that survives reboots (/tmp does not).
OUT="$HOME/.sushi/runs/bench-$TAG"
mkdir -p "$OUT"

# ── Model matrix: logical|path ──
# A missing path skips the row silently — a bench you can't run on this box
# isn't an error on the box that can.
MD="${SUSHI_MODELS_DIR:-$HOME/.sushi/models}"
TARGETS=(
    "sushi-4bpw|$MD/Qwen3.8-Flash-Next-Sushi-4bpw"
    "mimo-2.3bpw|$MD/MiMo-V2.6-Flash-Sushi-2.3bpw"
)

# Only ever called on the path that STARTED a server: --url may be pointed at
# a local sushi someone else is using, and a bench must not kill it.
stop_server() {
    [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null
    for _ in $(seq 1 30); do
        lsof -ti tcp:"$PORT" >/dev/null 2>&1 || return 0
        sleep 1
    done
}

probe() { # logical host model_id
    local depth=(--bench-only)
    [[ "$FULL" -eq 1 ]] && depth+=(--full)
    echo "── $1 ($2, $3) ──"
    # shellcheck disable=SC2086
    $LLMPROBE "$2" -m "$3" "${depth[@]}" --save "$OUT/$1.json" \
        || echo "  llmprobe failed for $1" >&2
    # Greedy bench cells never see a draft rejected at temperature; agent clients send T1.0, top_p 0.95, top_k 20
    # (the pack's generation_config supplies top_k).
    if [[ "$SAMPLED" -eq 1 ]]; then
        $LLMPROBE "$2" -m "$3" "${depth[@]}" --sampling creative --save "$OUT/$1-sampled.json" \
            || echo "  llmprobe (sampled) failed for $1" >&2
    fi
}

# --mtp restates the default wherever the checkpoint ships a head, so the
# cell never depends on a model-settings.json.
spec_flags() { # model_path
    local f=""
    if ls "$1"/*mtp*.safetensors >/dev/null 2>&1 || [ -d "$1/mtp" ] \
       || grep -qi '"mtp' "$1/config.json" 2>/dev/null; then
        f=" --mtp"
    fi
    echo "$f"
}

# ── Run ──
if [[ -n "$URL" ]]; then
    [[ -n "$URL_MODEL" ]] || { echo "--url needs -m <model id>" >&2; exit 1; }
    echo "=== bench: $URL ($URL_MODEL) ==="
    probe "$(echo "$URL_MODEL" | tr '/ ' '__')" "$URL" "$URL_MODEL"
else
    [[ -x "$BINARY" ]] || { echo "no $BINARY — build ReleaseFast first" >&2; exit 1; }
    echo "=== bench: sushi, tag=$TAG, reports → $OUT ==="
    trap 'stop_server' EXIT
    stop_server
    for row in "${TARGETS[@]}"; do
        IFS='|' read -r logical path <<< "$row"
        [[ -n "$ONLY" && "$logical" != *"$ONLY"* ]] && continue
        [[ -e "$path" ]] || { echo "SKIP $logical (no checkpoint at $path)" >&2; continue; }
        flags="$(spec_flags "$path")"
        echo; echo ">> $logical$flags"
        # shellcheck disable=SC2086
        "$BINARY" --serve --model "$path" --port "$PORT" $flags >"$OUT/$logical.log" 2>&1 &
        pid=$!
        for _ in $(seq 1 300); do
            curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
            sleep 1
        done
        if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            probe "$logical" "localhost:$PORT" "$(basename "$path")"
        else
            echo "  sushi never came up for $logical" >&2
        fi
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
        stop_server
        sleep "$SETTLE"
    done
    stop_server
fi

# ── The only artifact: rows to paste into benchmarks.md ──
echo
python3 - "$OUT" <<'PY'
import json, re, sys
from pathlib import Path

for path in sorted(Path(sys.argv[1]).glob("*.json")):
    bench = (json.loads(path.read_text()) or {}).get("bench") or {}
    decode = (bench.get("decodeTokPerSec") or {}).get("median")
    prefill = (bench.get("prefillTokPerSec") or {}).get("median")
    tps = (bench.get("speculative") or {}).get("tokensPerStep") or 1.0
    # WHICH speculative mode ran is only knowable from the server's own log
    # (llmprobe reports that one engaged, not which one). Name it in the cell
    # only when it actually paid: armed-but-not-accepting is not "mtp".
    log = path.with_name(path.stem.removesuffix("-sampled") + ".log")
    modes = re.findall(r"\[spec-stats\] mode=(\w+)",
                       log.read_text(errors="replace")) if log.exists() else []
    mode = f" {max(set(modes), key=modes.count)}" if modes and tps > 1.05 else ""
    if decode is None:
        print(f"| {path.stem} | · |  (no bench block)")
        continue
    print(f"| {path.stem} | {decode:.0f}{mode} |"
          f"  (prefill {prefill:.0f}, {tps:.2f} tok/step)")
PY
echo
echo "=== reports $OUT"
