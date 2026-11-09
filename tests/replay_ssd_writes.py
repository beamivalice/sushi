#!/usr/bin/env python3
# sushi-test: tier=tool
"""SSD prefix-cache replay: scripted flows against private servers, greedy with a fixed seed and one request in
flight. Each flow writes a restore tape (`SUSHI_RESTORE_TAPE`, compared with tests/diff_restore_tape.py) and a
per-request table of the bytes the SSD tier wrote by kind, the restore source and the timings.

    tests/replay_ssd_writes.py FLOW --out DIR [--bin sushi] [--pack qwen-stq|glm|dsv41|mimo] [--twice]
                               [--turns N] [--doc-tokens N] [--max-later-turn-mb MB]
    tests/replay_ssd_writes.py --list

DIR gets `<flow>.tape`, `<flow>.turns.jsonl`, `<flow>.summary.json`, one log per boot and a private cache root.
--twice runs the flow into DIR/run1 and DIR/run2 and fails unless the two tapes are equal: a flow that is not
deterministic cannot serve as a proof. --max-later-turn-mb fails the run when the mean bytes written by a later turn
(any request marked later) passes it. Every boot takes the GPU lock around its server (tests/lib/sushi_boot.py).
"""
import argparse
import base64
import hashlib
import json
import os
import random
import re
import statistics
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))

MODELS = Path(os.environ.get("SUSHI_MODELS_DIR", Path.home() / ".sushi/models"))

PACKS = {
    "qwen-stq": dict(dir="Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw-v2.0", flags=["--ctx-size", "262144"],
                     body={"enable_thinking": False}, tools=True, images=True),
    "glm": dict(dir="GLM-5.3-Flash-Sushi-2.4bpw", flags=["--ctx-size", "65536"],
                body={"reasoning_effort": "low"}, tools=False, images=False),
    "dsv41": dict(dir="DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw",
                  flags=["--ctx-size", "65536", "--ssd-budget-gb", "48", "--max-concurrent", "1"],
                  body={}, tools=False, images=False),
    "mimo": dict(dir="MiMo-V2.6-Flash-Sushi-2.3bpw", flags=["--ctx-size", "65536", "--kv-quant", "8"],
                 body={"enable_thinking": False}, tools=False, images=False, env_pack="SUSHI_TEST_PACK_MIMO"),
}
SSD_ONLY = ["--prefix-cache-mem", "off", "--prefix-cache-disk", "20GB"]

WORDS = ("harbour quay stone merchant herring barrel timber rope tar cooper oak lamp ladder ship baltic iron coal glass "
         "ale railway ice market school gasworks terrace mast storm tide river moor wool salt lifeboat volunteer gull "
         "museum smack model photograph street winter summer north south ledger account invoice cargo manifest clerk "
         "warehouse crane dock pier lighthouse keeper signal flag compass chart anchor hull keel sail canvas deck").split()

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file from the project and return its text.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "file path"}},
                   "required": ["path"]}}}]
SYSTEM = ("You are a code agent. Work only through tools. After each tool result, immediately call read_file on "
          "the next file the result names. Never answer in prose.\n\nProject handbook:\n")
SYSTEM_PLAIN = "You are a careful assistant. Answer in two short sentences.\n\nProject handbook:\n"
KINDS = ("kv", "ssm", "qsa", "spec", "ring", "meta")


def prose(rng, n_words):
    out, line = [], []
    for i in range(n_words):
        line.append(rng.choice(WORDS))
        if len(line) >= rng.randint(9, 16):
            s = " ".join(line)
            out.append(s[0].upper() + s[1:] + ".")
            line = []
        if i % 400 == 399:
            out.append(f"\n\n[section {i // 400}: ref {rng.randint(10**5, 10**6)}]\n")
    return " ".join(out)


def png(w, h, seed):
    rng = random.Random(seed)
    rows = b""
    c = [rng.randint(0, 255) for _ in range(3)]
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            q = (x * 4 // w) + 4 * (y * 4 // h)
            row += bytes(((c[0] + 37 * q) % 256, (c[1] + 71 * q) % 256, (c[2] + 13 * q) % 256))
        rows += bytes(row)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def written_delta(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


def later_mean(rows, key="written_mb"):
    """Mean MB under `key` of the requests marked `later`; None when no request is."""
    later = [r.get(key, 0.0) for r in rows if r.get("later")]
    return round(sum(later) / len(later), 1) if later else None


def ceiling_key(rows):
    """On the content-addressed layer the SSD takes what it stored, not the files' bytes."""
    return "stored_mb" if any(r.get("stored_mb") for r in rows) else "written_mb"


def ceiling_failure(rows, max_mb):
    key = ceiling_key(rows)
    mean = later_mean(rows, key)
    if max_mb is not None and mean is not None and mean > max_mb:
        return f"later-turn mean {mean} MB {key.removesuffix('_mb')} passes the ceiling {max_mb} MB"
    return None


class Conv:
    """One conversation as the client holds it: the message list, its rng and the next tool result."""

    def __init__(self, ident, cfg, doc_tokens, shared_tokens=4000, images=0):
        self.id = ident
        self.cfg = cfg
        self.rng = random.Random(100 + ident)
        shared = prose(random.Random(7), int(shared_tokens * 0.75))
        doc = prose(self.rng, int(doc_tokens * 0.75))
        system = (SYSTEM if cfg["tools"] else SYSTEM_PLAIN) + shared
        self.msgs = [{"role": "system", "content": system},
                     {"role": "user", "content": f"Task {ident}: audit the project. Notes:\n{doc}\n\nStart by reading part_1.txt."}]
        self.images = images

    def absorb(self, reply, t, result_tokens=1500):
        """Append the assistant reply and the next tool result (or user note), and a screenshot when due."""
        calls = reply.get("tool_calls") or []
        am = {"role": "assistant", "content": reply.get("content") or ""}
        if calls:
            am["tool_calls"] = calls
        self.msgs.append(am)
        text = f"part_{t + 1}.txt:\n" + prose(self.rng, int(result_tokens * 0.75)) + f"\n\nNext file: part_{t + 2}.txt"
        if calls:
            self.msgs.append({"role": "tool", "tool_call_id": calls[0].get("id", f"call{len(self.msgs)}"), "content": text})
        else:
            self.msgs.append({"role": "user", "content": text + "\nSummarize it in two sentences."})
        if self.images and t % self.images == self.images - 1:
            b64 = base64.b64encode(png(448, 448, 1000 * self.id + t)).decode()
            self.msgs.append({"role": "user", "content": [
                {"type": "text", "text": "Attached image(s) from tool result:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]})

    def fork(self):
        return [dict(m) for m in self.msgs]


def get(base, route, timeout=60):
    with urllib.request.urlopen(base + route, timeout=timeout) as r:
        return json.load(r)


def post(base, route, body, timeout=3600):
    req = urllib.request.Request(base + route, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class Boot:
    """One server under the GPU lock on the flow's private cache root, writing the flow's tape."""

    def __init__(self, run, label, flags, env=None):
        import sushi_boot
        sushi_boot.BIN = Path(run.bin)
        os.environ["SUSHI_RUN_DIR"] = str(run.dir)
        self.run = run
        cfg = run.cfg
        home = run.dir / f"{label}.home"
        (home / ".sushi").mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "HOME": str(home), "SUSHI_PREFIX_CACHE_DIR": str(run.cache),
               "SUSHI_RESTORE_TAPE": str(run.tape), **(env or {})}
        argv = cfg["flags"] + ["--log-level", "debug", "--metrics"] + flags
        self.server = sushi_boot.Server(label, model=run.pack_dir, flags=argv, env=env)
        self.label = label
        self.off = 0

    def __enter__(self):
        self.server.start()
        self.server.wait_ready()
        self.base = self.server.base
        self.off = self.server.log.stat().st_size
        self.before = self.written()
        return self

    def __exit__(self, exc_type, *_):
        self.stop(clean=not exc_type)

    def stop(self, clean=True, kill=False):
        p = self.server.proc
        if p is None:
            return
        if kill:
            p.kill()
            p.wait()
            self.server.stop(gone_ok=True)
        else:
            self.server.stop(0 if clean else 1, gone_ok=True)

    def alive(self):
        return self.server.proc is not None and self.server.proc.poll() is None

    def written(self):
        """Per kind, the files' bytes written and (`stored:` keys) what the content-addressed layer stored."""
        pw = get(self.base, "/metrics.json").get("prefix_cache_writes", {})
        w = {k: int(v) for k, v in pw.get("written_bytes", {}).items()}
        w.update({f"stored:{k}": int(v) for k, v in pw.get("stored_bytes", {}).items()})
        return w

    def settle(self):
        """Until the counters hold still: the writer has landed everything this request staged."""
        prev, same = None, 0
        while same < 4:
            time.sleep(0.5)
            cur = self.written()
            same = same + 1 if cur == prev else 0
            prev = cur
        return prev

    def log_since(self):
        with self.server.log.open("rb") as f:
            f.seek(self.off)
            seg = f.read().decode("utf-8", "replace")
        self.off += len(seg.encode())
        return seg

    def ask(self, msgs, tag, later=False, max_tokens=160, tools=None, settle=True):
        body = {"model": "x", "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "seed": 7, "stream": False,
                **self.run.cfg["body"]}
        if tools:
            body["tools"] = tools
        ts = time.time()
        r = post(self.base, "/v1/chat/completions", body)
        wall = time.time() - ts
        after = self.settle() if settle else self.written()
        seg = self.log_since()
        both = written_delta(after, self.before)
        self.before = after
        delta = {k: v for k, v in both.items() if not k.startswith("stored:")}
        stored = {k.removeprefix("stored:"): v for k, v in both.items() if k.startswith("stored:")}
        msg = r["choices"][0]["message"]
        m = re.search(r"reused (\d+) of (\d+) prompt tokens from (\w+)(?:, prefilling \d+)?(?:; lookup and restore (\d+) ms)?", seg)
        props = get(self.base, "/props")
        pc = props.get("settings", {}).get("prefix_cache", {})
        row = {"flow": self.run.flow, "boot": self.label, "tag": tag, "later": later,
               "prompt": r["usage"]["prompt_tokens"], "completion": r["usage"]["completion_tokens"],
               "finish": r["choices"][0]["finish_reason"],
               "reused": int(m.group(1)) if m else None, "from": m.group(3) if m else None,
               "restore_ms": int(m.group(4)) if m and m.group(4) else None,
               "prompt_ms": r.get("timings", {}).get("prompt_ms"), "wall_s": round(wall, 2),
               "written": delta, "written_mb": round(sum(delta.values()) / 2**20, 1),
               "stored": stored, "stored_mb": round(sum(stored.values()) / 2**20, 1),
               "disk_used_mb": round(pc.get("disk_used_bytes", 0) / 2**20, 1), "disk_entries": pc.get("disk_entries"),
               "persist": re.findall(r"\[disk-cache\] (?:persisted|chunk share|evicted|segment share)[^\n]*", seg),
               "reply_sha": hashlib.sha256(json.dumps([msg.get("content"), msg.get("tool_calls")], sort_keys=True).encode()).hexdigest()[:16]}
        self.run.rows.append(row)
        with (self.run.dir / f"{self.run.flow}.turns.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{self.run.flow}] {tag} prompt={row['prompt']} reused={row['reused']} {row['from']} "
              f"restore={row['restore_ms']}ms prompt_ms={row['prompt_ms']} wrote={row['written_mb']}MB "
              f"stored={row['stored_mb']}MB", flush=True)
        return msg


class Run:
    def __init__(self, a, flow, directory):
        self.flow = flow
        self.dir = Path(directory).resolve()
        self.bin = a.bin
        self.a = a
        self.cfg = PACKS[a.pack]
        self.pack_dir = Path(os.environ.get(self.cfg.get("env_pack", ""), MODELS / self.cfg["dir"]))
        self.cache = self.dir / "kv-cache"
        self.tape = self.dir / f"{flow}.tape"
        self.rows = []
        self.boots = 0

    def reset(self):
        subprocess.run(["rm", "-rf", str(self.cache)], check=True)
        self.cache.mkdir(parents=True, exist_ok=True)

    def boot(self, flags, env=None):
        self.boots += 1
        return Boot(self, f"{self.flow}-b{self.boots}", flags, env)

    def tools(self):
        return TOOLS if self.cfg["tools"] else None


# ------------------------------------------------------------------ flows

def agent_turns(run, b, cv, first, last, images_note=False):
    for t in range(first, last):
        msg = b.ask(cv.msgs, f"c{cv.id}.t{t}", later=t > 0, tools=run.tools())
        cv.absorb(msg, t)


def flow_agent(run):
    a = run.a
    cv = Conv(0, run.cfg, a.doc_tokens, images=2 if run.cfg["images"] else 0)
    with run.boot(SSD_ONLY) as b:
        agent_turns(run, b, cv, 0, a.turns)


def flow_retry(run):
    a = run.a
    cv = Conv(0, run.cfg, a.doc_tokens)
    with run.boot(SSD_ONLY) as b:
        b.ask(cv.msgs, "first")
        b.ask(cv.msgs, "retry", later=True)
        edited = cv.fork()
        edited[-1] = dict(edited[-1], content=edited[-1]["content"] + "\nPlease be brief.")
        b.ask(edited, "edited-last", later=True)


def flow_fork(run):
    a = run.a
    cv = Conv(0, run.cfg, a.doc_tokens)
    with run.boot(SSD_ONLY) as b:
        msg = b.ask(cv.msgs, "base")
        cv.absorb(msg, 0)
        base = cv.fork()
        side_a = base + [{"role": "user", "content": "Branch A: list the first three section references."}]
        side_b = base + [{"role": "user", "content": "Branch B: name the most common word in the notes."}]
        reply_a = b.ask(side_a, "fork-a", later=True)
        b.ask(side_b, "fork-b", later=True)
        side_a2 = side_a + [{"role": "assistant", "content": reply_a.get("content") or ""},
                            {"role": "user", "content": "Branch A, continued: now the last three."}]
        b.ask(side_a2, "fork-a-next", later=True)


def flow_far_divergence(run):
    a = run.a
    cv = Conv(0, run.cfg, a.doc_tokens)
    with run.boot(SSD_ONLY) as b:
        agent_turns(run, b, cv, 0, 3)
        early = cv.fork()
        early[0] = dict(early[0], content="You are a meticulous code agent.\n" + early[0]["content"])
        b.ask(early, "early-edit", later=True)
        agent_turns(run, b, cv, 3, 4)


def flow_images(run):
    a = run.a
    cv = Conv(0, run.cfg, a.doc_tokens, images=1)
    with run.boot(SSD_ONLY) as b:
        agent_turns(run, b, cv, 0, a.turns)


def flow_restart(run):
    a = run.a
    cv = Conv(0, run.cfg, a.doc_tokens)
    half = max(2, a.turns // 2)
    with run.boot(SSD_ONLY) as b:
        agent_turns(run, b, cv, 0, half)
    with run.boot(SSD_ONLY) as b:
        agent_turns(run, b, cv, half, a.turns)


def flow_crash(run):
    """kill -9 once the writer is idle, then after the k-th landing; each time a restart serves the same turn again."""
    a = run.a
    cv = Conv(0, run.cfg, a.doc_tokens)
    with run.boot(SSD_ONLY) as b:
        agent_turns(run, b, cv, 0, 3)
        b.stop(kill=True)
    with run.boot(SSD_ONLY) as b:
        agent_turns(run, b, cv, 2, 4)
    for k in a.crash_at:
        run.reset()
        cv = Conv(1, run.cfg, a.doc_tokens)
        died = False
        b = run.boot(SSD_ONLY, {"SUSHI_TEST_CRASH_AFTER_LANDINGS": str(k)})
        with b:
            for t in range(2):
                try:
                    msg = b.ask(cv.msgs, f"k{k}.t{t}", later=t > 0, tools=run.tools())
                except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                    died = True
                    break
                cv.absorb(msg, t)
                time.sleep(2)
                if not b.alive():
                    died = True
                    break
            if not died:
                b.stop(kill=True)
        with run.boot(SSD_ONLY) as b2:
            t = len(cv.msgs) // 2 - 1 if died else 1
            b2.ask(cv.msgs, f"k{k}.after-restart", later=True, tools=run.tools())


def flow_budget(run):
    a = run.a
    convs = [Conv(i, run.cfg, a.doc_tokens) for i in range(2)]
    with run.boot(["--prefix-cache-mem", "off", "--prefix-cache-disk", "2GB"]) as b:
        for t in range(a.turns):
            for cv in convs:
                msg = b.ask(cv.msgs, f"c{cv.id}.t{t}", later=t > 0, tools=run.tools())
                cv.absorb(msg, t)


def flow_ram(run):
    a = run.a
    cv = Conv(0, run.cfg, a.doc_tokens)
    with run.boot(["--prefix-cache-mem", "2GB", "--prefix-cache-disk", "20GB"]) as b:
        agent_turns(run, b, cv, 0, a.turns)


def flow_glm_4conv(run):
    a = run.a
    convs = [Conv(i, run.cfg, a.doc_tokens) for i in range(4)]
    with run.boot(SSD_ONLY) as b:
        for t in range(a.turns):
            for cv in convs:
                msg = b.ask(cv.msgs, f"c{cv.id}.t{t}", later=t > 0, tools=run.tools(), max_tokens=128)
                cv.absorb(msg, t)


def flow_dsv41_live(run):
    """Live-state rollback (an edited last message), hand-over to another conversation, and an idle flush."""
    a = run.a
    cv = Conv(0, run.cfg, a.doc_tokens)
    other = Conv(1, run.cfg, a.doc_tokens)
    with run.boot(["--prefix-cache-mem", "4GB", "--prefix-cache-disk", "20GB"]) as b:
        msg = b.ask(cv.msgs, "a.t0", max_tokens=128)
        before = cv.fork()
        cv.absorb(msg, 0)
        msg = b.ask(cv.msgs, "a.t1", later=True, max_tokens=128)
        rolled = before + [{"role": "user", "content": "A different follow-up: name three ships."}]
        b.ask(rolled, "a.rollback", later=True, max_tokens=128)
        b.ask(other.msgs, "b.t0", max_tokens=128)
        time.sleep(a.idle_s)
        cv.absorb(msg, 1)
        b.ask(cv.msgs, "a.t2-after-idle", later=True, max_tokens=128)


def flow_mimo_ring(run):
    """Message-start marks across a conversation whose replies pass the window, then a fork off its first mark."""
    a = run.a
    cv = Conv(0, run.cfg, a.doc_tokens)
    with run.boot(["--prefix-cache-mem", "4GB", "--prefix-cache-disk", "20GB"]) as b:
        for t in range(3):
            msg = b.ask(cv.msgs, f"main.t{t}", later=t > 0, max_tokens=400)
            cv.absorb(msg, t)
        side = cv.msgs[:2] + [{"role": "user", "content": "<system-reminder>Side note: be terse.</system-reminder>\nWhat is part 1 about?"}]
        b.ask(side, "fork-first-mark", later=True, max_tokens=200)
        b.ask(cv.msgs, "main.t3-after-fork", later=True, max_tokens=400)


FLOWS = {
    "agent": (flow_agent, "qwen-stq", dict(turns=20, doc_tokens=100000)),
    "retry": (flow_retry, "qwen-stq", dict(turns=3, doc_tokens=30000)),
    "fork": (flow_fork, "qwen-stq", dict(turns=4, doc_tokens=30000)),
    "far-divergence": (flow_far_divergence, "qwen-stq", dict(turns=4, doc_tokens=30000)),
    "images": (flow_images, "qwen-stq", dict(turns=24, doc_tokens=10000)),
    "restart": (flow_restart, "qwen-stq", dict(turns=6, doc_tokens=20000)),
    "crash": (flow_crash, "qwen-stq", dict(turns=4, doc_tokens=12000)),
    "budget": (flow_budget, "qwen-stq", dict(turns=8, doc_tokens=20000)),
    "ram": (flow_ram, "qwen-stq", dict(turns=8, doc_tokens=20000)),
    "glm-4conv": (flow_glm_4conv, "glm", dict(turns=3, doc_tokens=12000)),
    "dsv41-live": (flow_dsv41_live, "dsv41", dict(turns=3, doc_tokens=12000)),
    "mimo-ring": (flow_mimo_ring, "mimo", dict(turns=4, doc_tokens=8000)),
}


def run_flow(a, directory):
    run = Run(a, a.flow, directory)
    run.dir.mkdir(parents=True, exist_ok=True)
    for f in (run.tape, run.dir / f"{a.flow}.turns.jsonl"):
        f.unlink(missing_ok=True)
    run.reset()
    FLOWS[a.flow][0](run)
    mean = later_mean(run.rows)
    summary = {"flow": a.flow, "pack": a.pack, "requests": len(run.rows), "later_mean_mb": mean,
               "later_mean_stored_mb": later_mean(run.rows, "stored_mb"),
               "total_stored_mb": round(sum(r["stored_mb"] for r in run.rows), 1),
               "first_mb": run.rows[0]["written_mb"] if run.rows else None,
               "total_mb": round(sum(r["written_mb"] for r in run.rows), 1),
               "by_kind_mb": {k: round(sum(r["written"].get(k, 0) for r in run.rows) / 2**20, 1) for k in KINDS}}
    (run.dir / f"{a.flow}.summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)
    return run, ceiling_failure(run.rows, a.max_later_turn_mb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flow", nargs="?", choices=sorted(FLOWS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--bin", default=os.environ.get("SUSHI_BIN", str(HERE.parent / "zig-out/bin/sushi")))
    ap.add_argument("--pack", choices=sorted(PACKS))
    ap.add_argument("--twice", action="store_true")
    ap.add_argument("--turns", type=int)
    ap.add_argument("--doc-tokens", type=int)
    ap.add_argument("--max-later-turn-mb", type=float)
    ap.add_argument("--crash-at", default="2,12,30", help="landing counts for the crash flow")
    ap.add_argument("--idle-s", type=float, default=20.0, help="idle wait of the dsv41-live flow")
    a = ap.parse_args()
    if a.list:
        for name, (_, pack, d) in sorted(FLOWS.items()):
            print(f"{name}\t{pack}\tturns={d['turns']} doc_tokens={d['doc_tokens']}")
        return 0
    if not a.flow or not a.out:
        ap.error("FLOW and --out are required")
    _, pack, defaults = FLOWS[a.flow]
    a.pack = a.pack or pack
    a.turns = a.turns or defaults["turns"]
    a.doc_tokens = a.doc_tokens or defaults["doc_tokens"]
    a.crash_at = [int(x) for x in a.crash_at.split(",")]
    out = Path(a.out)
    if not a.twice:
        _, bad = run_flow(a, out)
        if bad:
            print(bad, file=sys.stderr)
        return 1 if bad else 0
    import diff_restore_tape as drt
    runs = []
    for n in (1, 2):
        run, bad = run_flow(a, out / f"run{n}")
        runs.append(run)
        if bad:
            print(bad, file=sys.stderr)
            return 1
    why = drt.compare(drt.load(runs[0].tape), drt.load(runs[1].tape))
    if why:
        print(f"{a.flow} is not deterministic: {why}", file=sys.stderr)
        return 1
    print(f"{a.flow}: both runs wrote the same tape", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
