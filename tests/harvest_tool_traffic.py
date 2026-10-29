#!/usr/bin/env python3
# sushi-test: tier=tool
"""Harvest real (declared tool schema, raw model output) pairs from sushi
into a replay fixture.

Every pair is a REAL thing a model emitted with a REAL client's tools attached,
so `src/tool_traffic_replay_test.zig` can replay it through the actual
parse+coerce path and assert the universal invariants forever. This turns an
hour of agentic soak into permanent regression coverage.

Source of truth is `SUSHI_RAW_DUMP_FILE`, which the server writes at the one
site where the schema and the raw text are both in scope:

    \\n===MLX_RAW_DUMP tools=<T> raw=<N>===\\n<T bytes of tools JSON><N bytes of raw>

Do NOT reconstruct these from the debug log: it truncates lines at 16 KB, so large
request bodies never re-parse and a scraper silently pairs a model output with an
earlier request's schema (which is exactly the bug this file was rewritten to fix).

Usage:
    SUSHI_RAW_DUMP_FILE=/tmp/rawdump.txt \\
      sushi --model <m> --serve --log-level debug
    # drive agents at it, then:
    tests/harvest_tool_traffic.py --dump /tmp/rawdump.txt \\
        --out src/fixtures/tool_traffic.jsonl
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys, pathlib

HDR = re.compile(rb"\n===MLX_RAW_DUMP tools=(\d+) raw=(\d+)===\n")


CALLED_RE = [
    re.compile(r"<function=([A-Za-z0-9_]+)"),
    re.compile(r"call:([A-Za-z0-9_]+)"),
    re.compile(r'"name"\s*:\s*"([A-Za-z0-9_]+)"'),
]


def _called_names(raw):
    names = set()
    for rx in CALLED_RE:
        names.update(rx.findall(raw))
    for m in re.finditer(r"<tool_([a-z]+)>", raw):
        if m.group(1) not in ("name", "call", "calls"):
            names.add(m.group(1))
    return names


def _trim_tools(tools, raw):
    """Keep only the schema(s) of the tool(s) actually called. A full Claude Code
    tools array is ~97 KB — storing it per record bloats the fixture 100x. The
    replay only needs the called tool's schema (lookup is by name). Keeps at least
    one tool so schema-lookup is still exercised."""
    names = _called_names(raw)
    kept = [t for t in tools if (t.get("function") or {}).get("name") in names]
    return kept or tools[:1]


def records(dump_path):
    blob = pathlib.Path(dump_path).read_bytes()
    for m in HDR.finditer(blob):
        t, n = int(m.group(1)), int(m.group(2))
        s = m.end()
        tools_b, raw_b = blob[s:s + t], blob[s + t:s + t + n]
        if len(tools_b) != t or len(raw_b) != n:
            continue  # torn tail (server still writing)
        try:
            tools = json.loads(tools_b.decode("utf-8"))
        except Exception:
            continue
        if not tools:
            continue
        raw = raw_b.decode("utf-8", "replace")
        yield {"tools": _trim_tools(tools, raw), "raw": raw}


def tools_id(tools):
    key = json.dumps(tools, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def read_fixture(path):
    """Records with their tools arrays inlined. A line carrying `tools_id` and `tools` defines an array
    once; a record names it by `tools_id`. A record with inline `tools` is read as it is."""
    table, out = {}, []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "raw" not in r:
            table[r["tools_id"]] = r["tools"]
            continue
        if "tools_id" in r:
            r = {"tools": table[r.pop("tools_id")], **r}
        out.append(r)
    return out


def write_fixture(path, recs):
    """Each distinct tools array once, ahead of the records that name it."""
    ids = [tools_id(r["tools"]) for r in recs]
    table = {}
    for i, r in zip(ids, recs):
        table.setdefault(i, r["tools"])
    with path.open("w") as fh:
        for i, tools in table.items():
            fh.write(json.dumps({"tools_id": i, "tools": tools}, ensure_ascii=False) + "\n")
        for i, r in zip(ids, recs):
            rec = {"tools_id": i, **{k: v for k, v in r.items() if k != "tools"}}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def key_of(r):
    names = ",".join(sorted(
        (t.get("function") or {}).get("name") or "" for t in r["tools"]))
    return hashlib.sha1((r["raw"] + "|" + names).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, action="append",
                    help="raw-dump file (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-bytes", type=int, default=60000,
                    help="skip absurdly large records so the fixture stays reviewable")
    ap.add_argument("--model", default=None, help="tag records with the model that produced them")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = read_fixture(out) if out.exists() else []
    seen = {key_of(r) for r in existing}

    added, scanned = [], 0
    for d in a.dump:
        if not pathlib.Path(d).exists():
            continue
        for r in records(d):
            scanned += 1
            if len(r["raw"]) > a.max_bytes or not r["raw"].strip():
                continue
            k = key_of(r)
            if k in seen:
                continue
            seen.add(k)
            if a.model:
                r["model"] = a.model
            added.append(r)

    write_fixture(out, existing + added)
    print(f"scanned {scanned} records; {len(added)} new; "
          f"fixture now {len(existing) + len(added)} -> {out}")


if __name__ == "__main__":
    sys.exit(main())
