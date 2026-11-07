#!/usr/bin/env python3
# sushi-test: tier=tool
"""Characterization tapes: what a served pack produces today, recorded once and re-checked after a refactor stage.

    tests/tape.sh record <pack> <base-dir> [--runs 3] [--blocks capture,serial,spec,hot,ssd,stream]
    tests/tape.sh check  <pack> <base-dir> [--blocks ...]
    tests/tape.sh diff   <tape.json> <tape.json>

<pack> is qwen-stq (revision 2.0), qwen-2bpw (external drive: byte identity only, never a timed number), qwen-4bpw
(the streamed-experts case), glm, dsv41 or mimo. A tape has two halves ([the spec](../docs/testsuite.md)):

  logit tape    `sushi kld capture` of the pack against itself (4 prompts, 256 tokens); check captures again and `cmp`s
                every prompts/*/logits.f32 and the token lists.
  serving tape  HTTP against `--serve` under a throwaway HOME: serial, batched, speculative (pinned depth), hot-prefix
                restore, SSD restore across a restart, image, long prompt, streamed experts, thinking multi-turn.
                Each case stores the response (text, finish_reason, usage, a digest of every logprob row) and the
                engine log tags that were identical across the recording runs. A case is in the tape only if all
                --runs recordings agree byte for byte; otherwise it is stored as `unstable` and checked by an
                invariant. `check` runs each case once and names the first differing case and position.

Every model load takes the GPU lock right before it (GPU_LOCK_OWNER names the owner, SUSHI_BOOT_LOCK_MAX_MIN the
declared wait) and releases it when that server or capture exits. SUSHI_BIN picks the binary under test, SUSHI_TEST_PACK_<PACK> (QWEN_2BPW, MIMO) a pack outside SUSHI_MODELS_DIR.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/lib"))

MODELS = Path(os.environ.get("SUSHI_MODELS_DIR", Path.home() / ".sushi/models"))
LOCK = ROOT / "scripts/gpu-lock.sh"
PROMPTS_SRC = MODELS / "kld-teacher"
IMAGE = ROOT / "tests/fixtures/house.jpeg"

EXACT = ["--prefix-cache-entries", "0", "--prefix-cache-mem", "off", "--prefix-cache-disk", "off"]
HOT = ["--prefix-cache-entries", "8", "--prefix-cache-mem", "4GB", "--prefix-cache-disk", "off"]
SSD = ["--prefix-cache-mem", "off", "--prefix-cache-disk", "4GB"]
SLOTS = ["--max-concurrent", "4"]
MTP33 = ["--mtp-min-depth", "3", "--mtp-max-depth", "3"]

# cases per boot; a boot's flags are the pack's slots plus the block's own
PACKS = {
    "qwen-stq": dict(dir="Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw-v2.0", ids="qwen", n_ids=4, slots=SLOTS,
                     captures={"kv8": ["--kv-quant", "8"], "kvbf16-gdn32": ["--kv-quant", "off", "--qwen-gdn", "32"]},
                     blocks={"serial": (EXACT + ["--no-mtp"], ["serial", "raw", "batch2", "long"]),
                             "spec": (EXACT + MTP33, ["spec"]),
                             "hot": (HOT, ["hot", "image"]),
                             "ssd": (SSD, ["ssd"])}),
    "qwen-2bpw": dict(dir="Qwen3.8-Flash-Next-Sushi-2bpw", ids="qwen", n_ids=4, slots=SLOTS,
                      captures={"kv8": ["--kv-quant", "8"]},
                      blocks={"serial": (EXACT + ["--no-mtp"], ["serial", "raw", "batch2", "long"]),
                              "spec": (EXACT + MTP33, ["spec"])}),
    "qwen-4bpw": dict(dir="Qwen3.8-Flash-Next-Sushi-4bpw", ids="qwen", n_ids=4, slots=SLOTS, captures={},
                      blocks={"stream": (EXACT + ["--no-mtp", "--ssd-budget-gb", "24"], ["serial", "raw"])}),
    "glm": dict(dir="GLM-5.3-Flash-Sushi-2.4bpw", ids="text", n_ids=4, slots=SLOTS,
                captures={"generic": []},
                blocks={"serial": (EXACT + ["--no-drafter"], ["serial", "raw", "batch2", "batch4"]),
                        "spec": (EXACT + ["--drafter-min-depth", "4", "--drafter-max-depth", "4"], ["spec"]),
                        "hot": (HOT, ["hot", "image"]),
                        "ssd": (SSD, ["ssd"])}),
    "mimo": dict(dir="MiMo-V2.6-Flash-Sushi-2.3bpw", ids="text", n_ids=4, slots=SLOTS,
                 captures={"kv8": ["--kv-quant", "8"]},
                 blocks={"serial": (EXACT + ["--no-mtp"], ["serial", "raw", "batch2"]),
                         "spec": (EXACT + MTP33, ["spec"]),
                         "hot": (HOT, ["hot", "image"]),
                         "ssd": (SSD, ["ssd"])}),
    "dsv41": dict(dir="DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw", ids="dsv41", n_ids=2,
                  slots=["--ssd-budget-gb", "110", "--max-concurrent", "1"],
                  captures={"streamed": ["--ssd-budget-gb", "110"]},
                  blocks={"serial": (EXACT + ["--no-mtp"], ["serial", "raw"]),
                          "spec": (EXACT + ["--mtp-min-depth", "2", "--mtp-max-depth", "2"], ["spec"]),
                          "hot": (HOT + ["--vision"], ["hot", "image", "think"]),
                          "ssd": (SSD, ["ssd"])}),
}
CAPTURE_IDS = ["code-python-intervalset", "math-compute-integral", "multilingual-translate-th-zh-ja-ar", "story-night-train"]
IDS_FILE = {"qwen": "qwen38-flash-next-bf16-gdnf32-8cat-v1.prompts.jsonl",
            "text": "qwen38-flash-next-bf16-gdnf32-8cat-v1.prompts.jsonl",
            "dsv41": "deepseek-v41-flash-1017-native-8cat-v1.prompts.jsonl"}

PROMPTS = ["Write a Python function that merges overlapping intervals, then explain its time complexity in two sentences.",
           "Describe the water cycle for a ten year old, in a short flowing paragraph without lists.",
           "Write a Rust function that parses a comma separated list of integers and returns their sum, with error handling.",
           "Tell a short story about a lighthouse keeper who finds a message in a bottle."]
RAW = "The three laws of thermodynamics state that"
PARA = [
    "The harbour town of Merrowick sits where the river Aln meets the grey sea, and its history is a record of tides. "
    "In the thirteenth century a wooden quay was built there by salt merchants who traded barrels of cured herring for "
    "wool from the northern moors. The quay was rebuilt in stone two hundred years later after a winter storm carried "
    "most of the first one out to sea, and the stone still carries the merchants' marks cut deep into its capstones.",
    "By the eighteenth century the town had grown around three streets that ran parallel to the water. Chandlers sold "
    "rope and tar, coopers shaped staves from oak felled upriver, and a lamp-lighter walked the length of the quay each "
    "dusk with a ladder over his shoulder. Ships from the Baltic unloaded timber and iron, and left carrying coal, glass "
    "and the pale ale for which the local brewery had become known as far as Hamburg.",
    "The coming of the railway in 1861 changed the town more than any storm. Fish that had once taken two days to reach "
    "the inland cities now arrived in six hours, packed in ice cut from the mill pond each January. The population "
    "doubled in a generation. A school, a gasworks and a covered market were built, and the wealthier families moved to "
    "terraces on the hill, from which the masts could be counted at breakfast.",
    "The twentieth century was harder. The herring shoals thinned and then vanished, the brewery closed in 1934, and two "
    "wars took a share of the young men that the memorial on the green records name by name. The railway branch was cut "
    "in 1966. What remained was a quiet place of grey stone, gulls and a lifeboat station whose volunteers still answer "
    "the maroon in any weather, as they have since 1824.",
    "Today the quay is a promenade. The old cooperage holds a museum with a reconstructed fishing smack, a collection of "
    "ship models and a wall of photographs in which the same three streets can be seen growing and shrinking over a "
    "century and a half. In August the town holds a regatta, and the brewery has been reopened by two cousins who found "
    "the original recipe in a ledger behind a bricked-up fireplace.",
    "The town archive, kept in the upper room of the market hall, holds the harbour ledgers from 1702 onward. They record "
    "every vessel that entered, its master, its cargo and the dues paid, in a series of hands that change with the "
    "decades. A volunteer has begun to transcribe them, and the first volume is already online, searchable by ship, by "
    "captain and by the goods carried, which range from slate and sugar to live eels.",
]


def document(sections):
    """About 100 tokens a section; the heading keeps repeats from collapsing into a loop."""
    return "\n\n".join(f"Section {i + 1}. {PARA[i % len(PARA)]}" for i in range(sections))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- records and comparison

def response_record(resp):
    ch = resp["choices"][0]
    msg = ch.get("message") or {}
    u = resp["usage"]
    rec = {"text": ch["text"] if "text" in ch else msg.get("content") or "", "finish": ch.get("finish_reason"),
           "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"]}
    if msg.get("reasoning_content"):
        rec["reasoning"] = msg["reasoning_content"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is not None:
        rec["cached_tokens"] = cached
    lp = ch.get("logprobs")
    if lp and "content" in lp:
        rec["positions"] = [{"t": e["token"], "h": digest([e["token"], e["logprob"], e.get("top_logprobs")])}
                            for e in lp["content"]]
    elif lp and lp.get("tokens"):
        rec["positions"] = [{"t": t, "h": digest([t, a, list(m.items())])}
                            for t, a, m in zip(lp["tokens"], lp["token_logprobs"], lp["top_logprobs"])]
    return rec


NOISE = [(re.compile(r"\d+(\.\d+)?\s*(ms|us|ns|s|MB/s|GB/s|tok/s|t/s)\b"), "#"), (re.compile(r"\b\w*_ms=\S+"), "#"),
         (re.compile(r"\d+\.\d+"), "#"), (re.compile(r"\b[ep]\d+-\d+\b"), "#"), (re.compile(r"\be\d+\b"), "e#")]
TAGS = re.compile(r"\[(cache|disk-cache|hot-cache|spec-stats|mtp-planner|kv-cache|mtp|dflash|dsv41-prefix|prefix|batched)\]")


def log_lines(seg):
    out = []
    for line in seg.splitlines():
        if TAGS.search(line):
            for rx, sub in NOISE:
                line = rx.sub(sub, line)
            out.append(line.strip())
    return out


def first_diff(a, b, path):
    """The first difference between two records as one line naming where, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        pa, pb = a.get("positions"), b.get("positions")
        if pa is not None or pb is not None:
            pa, pb = pa or [], pb or []
            for i, (x, y) in enumerate(zip(pa, pb)):
                if x != y:
                    return f"{path} position {i}: token {x['t']!r} vs {y['t']!r}" + ("" if x["t"] != y["t"] else " (same token, logprob row differs)")
            if len(pa) != len(pb):
                return f"{path} position {min(len(pa), len(pb))}: {len(pa)} positions vs {len(pb)}"
        for k in sorted(set(a) | set(b)):
            if k == "positions":
                continue
            if k not in a or k not in b:
                return f"{path}.{k}: present in one tape only"
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None if len(a) == len(b) else f"{path}: {len(a)} entries vs {len(b)}"
    if a == b:
        return None
    if isinstance(a, str) and isinstance(b, str):
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        return f"{path} char {i}: {a[max(0, i - 12):i + 12]!r} vs {b[max(0, i - 12):i + 12]!r}"
    return f"{path}: {a!r} vs {b!r}"


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def compare_case(name, want, got, serial_of=None):
    """None when a case agrees with its tape entry, else one line naming the first difference."""
    if want.get("unstable"):
        # A non-deterministic case is held to an invariant: each batched row starts like the serial answer.
        for i, row in enumerate(got.get("rows") or []):
            ref = (serial_of or {}).get(got["prompts"][i])
            if ref is not None and common_prefix(ref["text"], row["text"]) < min(40, len(ref["text"])):
                return f"{name} row {i}: unstable case, and its text no longer starts like the serial answer"
        return None
    d = first_diff(want["response"], got["response"], f"{name}")
    if d:
        return d
    if "log" in want and want["log"] != got.get("log"):
        a, b = want["log"], got.get("log") or []
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        return f"{name} log line {i}: {a[i] if i < len(a) else None!r} vs {b[i] if i < len(b) else None!r}"
    return None


def compare_boots(want, got):
    """The first differing case, in tape order, or None."""
    for boot, cases in want["boots"].items():
        if boot not in got["boots"]:
            continue
        for name, entry in cases.items():
            g = got["boots"][boot].get(name)
            if g is None:
                return f"{boot}/{name}: case missing from the replay"
            d = compare_case(f"{boot}/{name}", entry, g, serial_of=None)
            if d:
                return d
    return None


# ---------------------------------------------------------------- HTTP

def post(base, route, body, timeout=3600):
    req = urllib.request.Request(base + route, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(base, route):
    with urllib.request.urlopen(base + route, timeout=30) as r:
        return json.load(r)


class Replay:
    """The cases of one boot against a running server, whose log is read for the engine's own evidence."""

    def __init__(self, srv):
        self.s, self.mark = srv, len(srv.log_text())
        efforts = get(srv.base, "/v1/models")["data"][0].get("reasoning_efforts") or []
        self.think = {"enable_thinking": False} if "off" in efforts else {"reasoning_effort": "low"}
        self.props = get(srv.base, "/props")

    def slice(self):
        text = self.s.log_text()
        seg, self.mark = text[self.mark:], len(text)
        return seg

    def chat(self, messages, n, lp=True, think=None):
        body = {"model": "sushi", "messages": messages, "max_tokens": n, "temperature": 0, "seed": 1, "stream": False,
                **(self.think if think is None else think)}
        if lp:
            body.update(logprobs=True, top_logprobs=5)
        return post(self.s.base, "/v1/chat/completions", body)

    def raw(self, prompt, n, lp=True):
        body = {"model": "sushi", "prompt": prompt, "max_tokens": n, "temperature": 0, "seed": 1, "ignore_eos": True}
        if lp:
            body["logprobs"] = 5
        return post(self.s.base, "/v1/completions", body)

    def barrier(self, bodies):
        """Every request leaves behind one barrier, so the batch forms from the first tick."""
        res, bar = [None] * len(bodies), threading.Barrier(len(bodies))

        def one(i):
            bar.wait()
            res[i] = post(self.s.base, "/v1/chat/completions", bodies[i])
        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(bodies))]
        [t.start() for t in ts]
        [t.join() for t in ts]
        return res

    def wait_log(self, pattern, timeout=15.0):
        end, seg = time.monotonic() + timeout, ""
        while True:
            seg += self.slice()
            if re.search(pattern, seg) or time.monotonic() > end:
                return seg
            time.sleep(0.25)

    def persisted(self, since):
        """An SSD entry lands in the background after the response; a restore before it lands would run cold."""
        end = time.monotonic() + 120
        while time.monotonic() < end:
            if len(re.findall(r"complete on disk|restore points to SSD", self.s.log_text())) > since:
                return
            time.sleep(0.5)
        raise AssertionError(f"no SSD entry persisted within 120 s; log {self.s.log}")

    def persisted_count(self):
        return len(re.findall(r"complete on disk|restore points to SSD", self.s.log_text()))

    def result(self, resp_or_list, seg, **extra):
        r = {"response": [response_record(x) for x in resp_or_list] if isinstance(resp_or_list, list) else response_record(resp_or_list),
             "log": log_lines(seg)}
        r.update(extra)
        return r

    def user(self, text):
        return [{"role": "user", "content": text}]

    # --- the cases; each returns {"response": ..., "log": [...]} for pass k
    def case_serial(self, k):
        self.slice()
        out = [self.chat(self.user(p), 256) for p in PROMPTS]
        return self.result(out, self.slice(), prompts=PROMPTS)

    def case_raw(self, k):
        self.slice()
        return self.result(self.raw(RAW, 256), self.slice())

    def _batch(self, n):
        self.slice()
        bodies = [{"model": "sushi", "messages": self.user(p), "max_tokens": 256, "temperature": 0, "seed": 1, "stream": False,
                   **self.think} for p in PROMPTS[:n]]
        res = self.barrier(bodies)
        return self.result(res, self.slice(), prompts=PROMPTS[:n])

    def case_batch2(self, k):
        return self._batch(2)

    def case_batch4(self, k):
        return self._batch(4)

    def case_long(self, k):
        self.slice()
        return self.result(self.chat(self.user(document(60) + "\n\nSummarize the history above in three sentences."), 128),
                           self.slice())

    def case_spec(self, k):
        self.slice()
        out = [self.chat(self.user(p), 256, lp=False) for p in PROMPTS[:2]]
        seg = self.wait_log(r"\[spec-stats\] mode=")
        if "[spec-stats] mode=" not in seg:
            raise AssertionError(f"speculation never engaged; log {self.s.log}")
        return self.result(out, seg, prompts=PROMPTS[:2])

    def case_hot(self, k):
        """~2k-token prompt, then the same plus a follow-up turn; the cache starts empty."""
        q1 = document(20) + "\n\nIn one sentence, what is the oldest structure in the town?"
        self.slice()
        a = self.chat(self.user(q1), 64)
        seg_a = self.slice()
        follow = [*self.user(q1), {"role": "assistant", "content": "The stone quay."},
                  {"role": "user", "content": "And what was the quay built from before it was rebuilt in stone?"}]
        b = self.chat(follow, 64)
        seg_b = self.slice()
        return {"response": [response_record(a), response_record(b)], "log": log_lines(seg_a) + ["--"] + log_lines(seg_b)}

    def case_image(self, k):
        url = "data:image/jpeg;base64," + base64.b64encode(IMAGE.read_bytes()).decode()
        msg = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                            {"type": "text", "text": "What is in this image? Answer in one sentence."}]}]
        self.slice()
        return self.result(self.chat(msg, 64), self.slice())

    def case_think(self, k):
        """Two thinking turns; the second carries the first's reasoning back, so the thought point is exercised."""
        think = {"reasoning_effort": "low"}
        m1 = self.user("A bat and a ball cost 1.10 together and the bat costs one more than the ball. What does the ball cost?")
        self.slice()
        a = self.chat(m1, 400, think=think)
        seg_a = self.slice()
        msg = a["choices"][0]["message"]
        m2 = [*m1, {"role": "assistant", "content": msg.get("content") or "", "reasoning_content": msg.get("reasoning_content") or ""},
              {"role": "user", "content": "Now answer the same question for a bat that costs two more."}]
        b = self.chat(m2, 400, think=think)
        return {"response": [response_record(a), response_record(b)], "log": log_lines(seg_a) + ["--"] + log_lines(self.slice())}


def run_boot(label, pack, flags, cases, runs):
    """One server under the lock; every case `runs` times. {case: [result per pass]}, plus the boot's load-time facts."""
    import sushi_boot
    with sushi_boot.Server(label, model=pack, flags=["--metrics", "--log-level", "debug"] + flags) as srv:
        srv.wait_ready()
        load = log_lines(srv.log_text())
        rp = Replay(srv)
        out = {c: [] for c in cases}
        for k in range(runs):
            for c in cases:
                out[c].append(getattr(rp, f"case_{c}")(k))
        return out, load


def run_fresh(label, pack, flags, cases, runs):
    """Cases that leave state in the prefix cache record each run on a boot of its own: no key isolates a run."""
    out, load = {c: [] for c in cases}, None
    for k in range(runs):
        one, load = run_boot(f"{label}-{k}", pack, flags, cases, 1)
        for c in cases:
            out[c] += one[c]
    return out, load


def run_ssd(label, pack, flags, runs):
    """Case 5: one boot persists prompt A, then `runs` boots each start from a copy of that cache and restore it."""
    import sushi_boot
    doc = document(20)
    qa = doc + "\n\nIn one sentence, what is the oldest structure in the town?"
    follow = [{"role": "user", "content": qa}, {"role": "assistant", "content": "The stone quay."},
              {"role": "user", "content": "And what was the quay built from before it was rebuilt in stone?"}]
    env = sushi_boot.private_env(label + "-persist")
    with sushi_boot.Server(label + "-persist", model=pack, flags=["--metrics", "--log-level", "debug"] + flags, env=env) as srv:
        srv.wait_ready()
        rp = Replay(srv)
        n = rp.persisted_count()
        rp.slice()
        first = rp.chat([{"role": "user", "content": qa}], 64)
        seg = rp.slice()
        rp.persisted(n)
        persisted = rp.result(first, seg)
    snap = env["SUSHI_PREFIX_CACHE_DIR"] + ".snapshot"
    shutil.rmtree(snap, ignore_errors=True)
    shutil.copytree(env["SUSHI_PREFIX_CACHE_DIR"], snap, symlinks=True)
    restored = []
    for k in range(runs):
        e = sushi_boot.private_env(f"{label}-restore{k}")
        shutil.rmtree(e["SUSHI_PREFIX_CACHE_DIR"], ignore_errors=True)
        shutil.copytree(snap, e["SUSHI_PREFIX_CACHE_DIR"], symlinks=True)
        with sushi_boot.Server(f"{label}-restore{k}", model=pack, flags=["--metrics", "--log-level", "debug"] + flags, env=e) as srv:
            srv.wait_ready()
            rp = Replay(srv)
            rp.slice()
            r = rp.chat(follow, 64)
            seg = rp.slice()
            if not re.search(r"reused [1-9]\d* of \d+ prompt tokens from SSD", seg):
                raise AssertionError(f"restore boot {k} did not restore from SSD; log {srv.log}")
            restored.append(rp.result(r, seg))
    return {"persist": [persisted], "restore": restored}


def capture(pack, name, flags, out, prompts, binary):
    import sushi_boot
    home = Path(out).parent / f"{name}.home"
    (home / ".sushi").mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    owner = f"{os.environ.get('GPU_LOCK_OWNER', 'tape')}-capture-{name}"
    max_min = os.environ.get("SUSHI_BOOT_LOCK_MAX_MIN", "40")
    cmd = [str(LOCK), "run", owner, "--max-min", max_min, "--", "taskpolicy", "-a", str(binary), "kld", "capture",
           "--model", str(pack), "--prompts", str(prompts), "--out", str(out), "--tokens", "256", "--top-k", "10"] + flags
    log = Path(out).parent / f"{name}.log"
    with log.open("wb") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "HOME": str(home)})
    if r.returncode != 0:
        raise AssertionError(f"capture {name} failed ({r.returncode}); log {log}")


def cmp_captures(a, b):
    """The first differing file of two captures, in name order, or None. Timings in baseline.json are not compared."""
    for p in sorted(Path(a).glob("prompts/*/*")):
        if p.name in ("logits.f32", "generated_tokens.txt", "prompt_tokens.txt", "rendered_prompt.txt"):
            q = Path(b) / p.relative_to(a)
            if not q.is_file():
                return f"{p.relative_to(a)}: missing from the new capture"
            if p.read_bytes() != q.read_bytes():
                size = p.stat().st_size
                x, y = p.read_bytes(), q.read_bytes()
                i = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
                unit = f"f32 {i // 4}" if p.suffix == ".f32" else f"byte {i}"
                return f"{p.relative_to(a)} differs at {unit} of {size // 4 if p.suffix == '.f32' else size}"
    return None


def write_prompts(spec, dest):
    src = PROMPTS_SRC / IDS_FILE[spec["ids"]]
    rows = {d["id"]: d for d in map(json.loads, src.read_text().splitlines())}
    ids = CAPTURE_IDS[:spec["n_ids"]]
    with dest.open("w") as f:
        for i in ids:
            d = rows[i]
            line = {"id": i, "prompt_ids": d["prompt_ids"]} if spec["ids"] != "text" else {"id": i, "prompt": d["text"]}
            f.write(json.dumps(line) + "\n")


def binary_stamp(binary):
    st = Path(binary).stat()
    sha = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
    commit = os.environ.get("SUSHI_TAPE_COMMIT") or subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip()
    return f"commit {commit}\nmtime {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}\nsha256 {sha}\npath {binary}\n"


def stable(passes):
    return all(p == passes[0] for p in passes[1:])


def settle(passes):
    """The tape entry for the recorded passes of one case."""
    if not stable([p["response"] for p in passes]):
        return {"unstable": True, "runs": [p["response"] for p in passes], "rows": passes[0]["response"] if isinstance(passes[0]["response"], list) else None,
                "prompts": passes[0].get("prompts")}
    e = {"response": passes[0]["response"]}
    if stable([p["log"] for p in passes]):
        e["log"] = passes[0]["log"]
    if "prompts" in passes[0]:
        e["prompts"] = passes[0]["prompts"]
    return e


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["record", "check", "diff"])
    ap.add_argument("pack", help="qwen-stq|qwen-2bpw|qwen-4bpw|glm|dsv41|mimo, or a tape.json for diff")
    ap.add_argument("base", help="the tape directory (record writes it, check reads it); the second tape for diff")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--blocks", default="")
    a = ap.parse_args(argv)

    if a.mode == "diff":
        d = compare_boots(json.loads(Path(a.pack).read_text()), json.loads(Path(a.base).read_text()))
        print(d or "IDENTICAL")
        return 1 if d else 0

    import sushi_boot
    spec = PACKS[a.pack]
    pack = Path(os.environ.get(f"SUSHI_TEST_PACK_{a.pack.upper().replace('-', '_')}") or MODELS / spec["dir"])
    if not pack.is_dir():
        sushi_boot.skip(f"pack {pack} is missing")
    binary = sushi_boot.BIN
    base = Path(a.base).expanduser()
    recording = a.mode == "record"
    work = base if recording else Path.home() / ".sushi/runs" / f"tape-check-{time.strftime('%Y%m%d-%H%M%S')}" / a.pack
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SUSHI_RUN_DIR", str(work / "run"))
    runs = a.runs if recording else 1
    want_blocks = [b for b in a.blocks.split(",") if b] or ["capture"] + list(spec["blocks"])
    if recording:
        (work / "BINARY").write_text(binary_stamp(binary))
        (work / "FLAGS").write_text(json.dumps({"slots": spec["slots"], "captures": spec["captures"],
                                                "blocks": {k: v for k, v in spec["blocks"].items()}}, indent=1) + "\n")
        write_prompts(spec, work / "prompts.jsonl")
    tape_file = base / "tape.json"
    tape = json.loads(tape_file.read_text()) if tape_file.exists() else {"pack": spec["dir"], "boots": {}}
    got = {"pack": spec["dir"], "boots": {}}
    failures = []

    if "capture" in want_blocks:
        for name, flags in spec["captures"].items():
            out = work / f"capture-{name}"
            capture(pack, name, flags, out, base / "prompts.jsonl", binary)
            if not recording:
                d = cmp_captures(base / f"capture-{name}", out)
                print(f"capture/{name}: " + (f"FAIL {d}" if d else "IDENTICAL"))
                if d:
                    failures.append(f"capture/{name}: {d}")
            else:
                print(f"capture/{name}: recorded")

    for block, (flags, cases) in spec["blocks"].items():
        if block not in want_blocks:
            continue
        if block == "ssd":
            res = run_ssd(f"tape-{a.pack}-ssd", pack, spec["slots"] + flags, runs)
            entries = {"persist": settle(res["persist"]), "restore": settle(res["restore"])}
        else:
            res, load = (run_fresh if block == "hot" else run_boot)(f"tape-{a.pack}-{block}", pack, spec["slots"] + flags, cases, runs)
            entries = {c: settle(res[c]) for c in cases}
            entries["load"] = {"response": None, "log": load, "note": True}
        got["boots"][block] = entries
        if recording:
            tape["boots"][block] = entries
            for c, e in entries.items():
                if e.get("unstable"):
                    print(f"{block}/{c}: UNSTABLE across {runs} runs (invariant only)")
            tape_file.write_text(json.dumps(tape, indent=1, ensure_ascii=False) + "\n")
        else:
            for c, e in entries.items():
                if c == "load":
                    continue
                w = (tape["boots"].get(block) or {}).get(c)
                if w is None:
                    print(f"{block}/{c}: not in the tape, skipped")
                    continue
                if w.get("unstable"):
                    ser = entries.get("serial") or {}
                    ref = dict(zip(ser.get("prompts") or [], ser.get("response") or []))
                    d = compare_case(f"{block}/{c}", w, {"rows": e["response"], "prompts": e.get("prompts")}, serial_of=ref)
                else:
                    d = compare_case(f"{block}/{c}", w, e)
                print(f"{block}/{c}: " + (f"FAIL {d}" if d else "IDENTICAL"))
                if d:
                    failures.append(d)
    if recording:
        print(f"recorded {a.pack} to {base}")
        return 0
    (work / "check.json").write_text(json.dumps(got, indent=1, ensure_ascii=False) + "\n")
    print("DIFF " + failures[0] if failures else f"IDENTICAL to {base}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
