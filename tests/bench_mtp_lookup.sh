#!/usr/bin/env bash
# sushi-test: tier=bench
. "$(dirname "$0")/private_cache.sh"
# bench_mtp_lookup.sh — the MTP prompt-lookup workload: agent-style copies and edits of a file in
# context, plus tasks that copy nothing. A benchmark driver, NOT a test.
#
#   tests/bench_mtp_lookup.sh <label> <model-dir> [extra sushi flags...]   one boot, one arm
#   tests/bench_mtp_lookup.sh --compare <run-dir-A> <run-dir-B> [...]      paired table of runs
#
# One arm per boot: `--no-mtp-lookup` is read once per process. An A/B is A B B A, e.g.
#   scripts/gpu-lock.sh acquire lookup-ab && taskpolicy -a tests/bench_mtp_lookup.sh A <pack> --no-mtp-lookup; scripts/gpu-lock.sh release lookup-ab
#   scripts/gpu-lock.sh acquire lookup-ab && taskpolicy -a tests/bench_mtp_lookup.sh B <pack>; scripts/gpu-lock.sh release lookup-ab
# A third boot with `--no-mtp` (label S) turns the compare's greedy byte line into the serial bar.
# Speeds are the server's own `timings.predicted_per_second`; each request's `[spec-stats]` lines
# are cut from the server log. The arm is proven by `lookup=R/D/L` and the one-shot
# `[mtp] prompt-lookup drafts engaged` line, never by the launch flags.
#
# Env: TASKS (comma list, default all), PORT (11377), BIN (./zig-out/bin/sushi), REPS (2), MAX_TOKENS (2048), THINK (0: thinking
# off, so the answer is the copy), SAMPLED (1: also a seeded sampled pass), LONG (0; 1 adds the
# file buried behind ~32k and ~64k tokens of repo text), RUNS_DIR (${SUSHI_RUNS_DIR:-~/.sushi/runs}).
set -uo pipefail

if [ "${1:-}" = "--compare" ]; then
    shift
    exec python3 - "$@" <<'PYCMP'
import json, statistics, sys
# Runs sharing a label pool (an A B B A reads as two arms); the ratio column is the second
# label over the first.
arms = {}
for d in sys.argv[1:]:
    label = json.load(open(f"{d}/meta.json"))["label"]
    arms.setdefault(label, []).extend(json.loads(l) for l in open(f"{d}/results.jsonl"))
labels = list(arms)
keys = []
for rows in arms.values():
    for r in rows:
        if (r["task"], r["mode"]) not in keys:
            keys.append((r["task"], r["mode"]))
def pick(rows, k):
    return [r for r in rows if (r["task"], r["mode"]) == k]
def med(rows, field):
    v = [r[field] for r in rows if r.get(field) is not None]
    return statistics.median(v) if v else None
print("task/mode".ljust(24) + "".join(f"{l:>10}" for l in labels) + ("     ratio" if len(labels) == 2 else "") + "   tok/round   lookup R/D/L (last)")
for k in keys:
    tps = [med(pick(arms[l], k), "tps") for l in labels]
    line = f"{k[0]}/{k[1]}".ljust(24) + "".join(f"{t:10.1f}" if t else f"{'-':>10}" for t in tps)
    if len(labels) == 2:
        line += f"{tps[1] / tps[0]:10.3f}" if tps[0] and tps[1] else f"{'-':>10}"
    line += "   " + " ".join(f"{t:.2f}" if t else "-" for t in (med(pick(arms[l], k), "tok_per_round") for l in labels))
    line += "   " + " ".join((pick(arms[l], k) or [{"lookup": "-"}])[-1]["lookup"] for l in labels)
    print(line)
greedy = [k for k in keys if k[1] == "greedy"]
differ = [k[0] for k in greedy if len({r["sha"] for l in labels for r in pick(arms[l], k)}) != 1]
print(f"\ngreedy outputs byte-identical across every run: {len(greedy) - len(differ)}/{len(greedy)}" + (f"  DIFFER: {', '.join(differ)}" if differ else ""))
PYCMP
fi

LABEL="${1:?usage: bench_mtp_lookup.sh <label> <model-dir> [flags...] | --compare <run-dirs...>}"
MODEL="${2:?usage: bench_mtp_lookup.sh <label> <model-dir> [flags...]}"
shift 2
PORT="${PORT:-11377}"
BIN="${BIN:-./zig-out/bin/sushi}"
RUNS_DIR="${RUNS_DIR:-${SUSHI_RUNS_DIR:-$HOME/.sushi/runs}}"
OUT="$RUNS_DIR/bench-mtp-lookup-$LABEL-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/server.log"

{
    echo "{\"label\":\"$LABEL\",\"model\":\"$(basename "$MODEL")\",\"commit\":\"$(git rev-parse --short HEAD 2>/dev/null)\","
    echo "\"binary_mtime\":\"$(stat -f %Sm "$BIN" 2>/dev/null)\","
    echo "\"reps\":${REPS:-2},\"max_tokens\":${MAX_TOKENS:-2048},\"think\":${THINK:-0},\"long\":${LONG:-0},\"flags\":\"$*\"}"
} >"$OUT/meta.json"

# Both arms: no stored round-cost table, so each boot measures from the same cold start.
SUSHI_ROUND_COST_PERSIST=0 "$BIN" --model "$MODEL" --serve --port "$PORT" --ctx-size 131072 --kv-quant 8 \
    --prefix-cache-entries 0 --no-drafter --log-level info "$@" >"$LOG" 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null' EXIT
for _ in $(seq 1 600); do
    grep -q "Model ready (loaded on inference thread)" "$LOG" && break
    kill -0 "$SRV" 2>/dev/null || { echo "server exited during load" >&2; tail -20 "$LOG" >&2; exit 1; }
    sleep 1
done
grep -E "\[mtp\] (on|off)|\[kv-cache\]" "$LOG" | head -3

python3 - "$PORT" "$LOG" "$OUT" "${REPS:-2}" "${MAX_TOKENS:-2048}" "${THINK:-0}" "${SAMPLED:-1}" "${LONG:-0}" <<'PYRUN'
import glob, hashlib, json, os, re, sys, time, urllib.request
port, log_path, out, reps, max_tokens, think, sampled, long_ctx = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6] == "1", sys.argv[7] == "1", sys.argv[8] == "1"

FILE = '''import csv
import json
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Item:
    sku: str
    name: str
    qty: int
    price_cents: int

    @property
    def value_cents(self):
        return self.qty * self.price_cents


def load_rows(path):
    with open(path, newline="") as fh:
        return [row for row in csv.DictReader(fh)]


def to_items(rows):
    items = []
    for row in rows:
        items.append(Item(row["sku"], row["name"], int(row["qty"]), int(row["price_cents"])))
    return items


def total_by_sku(items):
    totals = defaultdict(int)
    for item in items:
        totals[item.sku] += item.qty
    return dict(totals)


def value_by_sku(items):
    values = defaultdict(int)
    for item in items:
        values[item.sku] += item.value_cents
    return dict(values)


def low_stock(items, threshold=5):
    return sorted(sku for sku, qty in total_by_sku(items).items() if qty < threshold)


def top_value(items, n=3):
    ranked = sorted(value_by_sku(items).items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:n]


def summarize(path, threshold=5):
    items = to_items(load_rows(path))
    lines = []
    for i in range(1, len(items)):
        item = items[i]
        lines.append(f"{item.sku} {item.name}: {item.qty} @ {item.price_cents / 100:.2f}")
    lines.append(f"low stock: {', '.join(low_stock(items, threshold)) or 'none'}")
    for sku, cents in top_value(items):
        lines.append(f"top value: {sku} {cents / 100:.2f}")
    return "\\n".join(lines)


def merge(paths):
    items = []
    for path in paths:
        items.extend(to_items(load_rows(path)))
    return total_by_sku(items)


def export_json(path, out_path):
    items = to_items(load_rows(path))
    payload = {
        "totals": total_by_sku(items),
        "values": value_by_sku(items),
        "low_stock": low_stock(items),
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return out_path


if __name__ == "__main__":
    import sys

    print(summarize(sys.argv[1]))
'''
CODE = f"```python\n{FILE}```"
RENAME = "Rename the function `load_rows` to `read_rows` everywhere in inventory.py below (its definition and every call)."
WRITE_FILE = [{"type": "function", "function": {"name": "write_file", "description": "Write a file to disk, replacing it.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}]

def filler(chars):
    parts, n = [], 0
    for path in sorted(glob.glob("docs/*.md")) + ["src/scheduler.zig", "src/server.zig"]:
        text = open(path).read()
        parts.append(f"--- {path} ---\n{text}")
        n += len(text)
        if n >= chars:
            break
    return "".join(parts)[:chars]

tasks = [
    ("copy_verbatim", f"Repeat inventory.py below exactly as written. Output only the file, no commentary.\n\n{CODE}", None),
    ("rename", f"{RENAME} Output the complete updated file only, no commentary.\n\n{CODE}", None),
    ("fix_bug", f"`summarize` in inventory.py below skips the first item. Fix the bug and output the complete corrected file only, no commentary.\n\n{CODE}", None),
    ("unified_diff", f"{RENAME} Output only a unified diff (git style) of the change, nothing else.\n\n{CODE}", None),
    ("tool_json", f"{RENAME} Save the complete updated file with the write_file tool (path inventory.py).\n\n{CODE}", WRITE_FILE),
    ("new_code", "Write a Python module that parses an Apache access log, counts requests per status code and per hour, and prints a report. Output only the code.", None),
    ("prose", f"Explain what inventory.py below does, function by function, in plain prose. No code.\n\n{CODE}", None),
]
if long_ctx:
    for name, chars in (("long_rename_32k", 120_000), ("long_rename_64k", 240_000)):
        tasks.append((name, f"Here is background material from the repository:\n\n{filler(chars)}\n\nNow the task. {RENAME} Output the complete updated file only, no commentary.\n\n{CODE}", None))

only = [t for t in os.environ.get("TASKS", "").split(",") if t]
if only:
    tasks = [t for t in tasks if t[0] in only]
os.makedirs(f"{out}/outputs", exist_ok=True)

modes = [("greedy", {"temperature": 0})]
if sampled:
    modes.append(("sampled", {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 7}))

def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)

def log_size():
    return os.path.getsize(log_path)

def spec_stats(since):
    with open(log_path, "rb") as fh:
        fh.seek(since)
        text = fh.read().decode(errors="replace")
    lines = [l for l in text.splitlines() if "[spec-stats] mode=mtp" in l]
    if not lines:
        return {}
    l = lines[-1]
    get = lambda k: (re.search(rf" {k}=([0-9./]+)", l) or [None, None])[1]
    return {"attempts": get("attempts"), "accepts": get("accepts"), "lookup": get("lookup") or "0/0/0", "avg_per_round": get("avg_per_round")}

post({"model": "default", "max_tokens": 16, "temperature": 0, "enable_thinking": False,
      "messages": [{"role": "user", "content": "Say hi."}]})

with open(f"{out}/results.jsonl", "w") as res:
    for rep in range(reps):
        for name, prompt, tools in tasks:
            for mode, sampling in modes:
                body = {"model": "default", "stream": False, "max_tokens": max_tokens, "enable_thinking": think,
                        "messages": [{"role": "user", "content": prompt}], **sampling}
                if tools:
                    body["tools"] = tools
                since = log_size()
                t0 = time.time()
                r = post(body)
                wall = time.time() - t0
                time.sleep(0.3)
                msg = r["choices"][0]["message"]
                # A tool call's id is minted per request; the output is its name and arguments.
                calls = [c.get("function") for c in (msg.get("tool_calls") or [])]
                text = (msg.get("content") or "") + json.dumps(calls, sort_keys=True)
                with open(f"{out}/outputs/{name}-{mode}-rep{rep}.txt", "w") as fh:
                    fh.write(text)
                tm = r.get("timings", {})
                st = spec_stats(since)
                rounds = int(st.get("attempts") or 0) + int((st.get("lookup") or "0/0/0").split("/")[0])
                n = tm.get("predicted_n") or r["usage"]["completion_tokens"]
                row = {"rep": rep, "task": name, "mode": mode, "tps": tm.get("predicted_per_second"), "predicted_n": n,
                       "predicted_ms": tm.get("predicted_ms"), "wall_s": round(wall, 3), "finish": r["choices"][0].get("finish_reason"),
                       "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "lookup": st.get("lookup", "-"),
                       "attempts": st.get("attempts"), "tok_per_round": (n / rounds) if rounds else None}
                res.write(json.dumps(row) + "\n")
                res.flush()
                print(f"rep{rep} {name:18} {mode:8} {row['tps'] or 0:7.1f} tok/s  n={n:5}  lookup={row['lookup']:>10}  attempts={row['attempts']}  {row['finish']}", flush=True)
PYRUN
RC=$?
if grep -q "prompt-lookup drafts engaged" "$LOG"; then echo "ENGAGED: $(grep -m1 'prompt-lookup drafts engaged' "$LOG")"; else echo "ENGAGED: none (no lookup round in this boot)"; fi
echo "run: $OUT"
exit $RC
