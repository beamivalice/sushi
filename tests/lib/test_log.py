#!/usr/bin/env python3
# sushi-test: tier=lib
"""The per-test log of `zig build test -Dtest-log=<path>` (scripts/test_runner.zig), read for tests/gate.sh.

Each line is `<UTC>\t<step>\t<PASS|SKIP|FAIL>\t<ms>\t<test name>`, and a test process that ran a test ends on
`<UTC>\t<step>\tDONE\t<ms>\t<p> pass, <s> skip, <f> fail`.

  test_log.py follow <log> <done-marker>
      prints each line as it lands, until the marker exists and every line written before it is printed
  test_log.py summary <log> <build-log> [--step-failed]
      the per-step table, the 10 slowest tests and, last, every failed test with the error block zig printed for it
      in <build-log>; with --step-failed and no failure to name, the build log's last 40 lines
"""
import datetime
import os
import re
import sys
import time

STATUSES = ("PASS", "SKIP", "FAIL")


def parse(line):
    """(time, step, status, ms, text) of a log line, or None for one that is not."""
    parts = line.rstrip("\n").split("\t", 4)
    if len(parts) != 5 or not parts[3].isdigit():
        return None
    try:
        t = datetime.datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        return None
    return t, parts[1], parts[2], int(parts[3]), parts[4]


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return [r for r in (parse(line) for line in f) if r]


def secs(ms):
    return f"{ms / 1000:.1f} s" if ms < 60_000 else f"{ms / 60_000:.1f} min"


def live(r):
    t, step, status, ms, text = r
    mark = "  <-- FAIL" if status == "FAIL" else ""
    return f"{t.astimezone().strftime('%H:%M:%S')}  {step:<44} {status:<4} {secs(ms):>9}  {text}{mark}"


def follow(path, marker):
    pos, pending = 0, ""
    while True:
        done = os.path.exists(marker)
        data = ""
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                data = f.read()
                pos = f.tell()
        except FileNotFoundError:
            pass
        *lines, pending = (pending + data).split("\n")
        for line in lines:
            r = parse(line)
            print(live(r) if r else line, flush=True)
        if done and not data:
            return
        time.sleep(0.5)


# The build runner's per-test error forms (Maker/Step/Run.zig): its message, then continuation lines under 7 spaces.
ERROR = re.compile(r"^error: '(.*)' ((?:failed|errored|leaked|logged|timed out|terminated|exited)[^']*)$")


def error_blocks(build_log):
    """{test name: the lines zig printed for its error}, in the order they appear."""
    blocks, name = {}, None
    try:
        with open(build_log, encoding="utf-8", errors="replace") as f:
            lines = f.read().split("\n")
    except FileNotFoundError:
        return blocks, []
    for line in lines:
        m = ERROR.match(line)
        if m:
            name = m.group(1)
            head = m.group(2).removesuffix(":")
            blocks.setdefault(name, []).extend([] if head == "failed" else [head])
        elif name and line.startswith("       "):
            blocks[name].append(line[7:])
        else:
            name = None
    return blocks, lines


def summary(path, build_log, step_failed):
    rows = read(path) if os.path.exists(path) else []
    tests = [r for r in rows if r[2] in STATUSES]
    steps = {}
    for t, step, status, ms, _ in rows:
        s = steps.setdefault(step, {"PASS": 0, "SKIP": 0, "FAIL": 0, "start": None, "end": None, "done": False})
        start = t - datetime.timedelta(milliseconds=ms)
        s["start"] = start if s["start"] is None else min(s["start"], start)
        s["end"] = t if s["end"] is None else max(s["end"], t)
        if status in STATUSES:
            s[status] += 1
        elif status == "DONE":
            s["done"] = True
    print("gate: tests by step")
    print(f"  {'step':<44} {'tests':>6} {'pass':>6} {'skip':>5} {'fail':>5} {'wall':>9}")
    for step, s in sorted(steps.items(), key=lambda kv: kv[1]["start"]):
        n = s["PASS"] + s["SKIP"] + s["FAIL"]
        wall = int((s["end"] - s["start"]).total_seconds() * 1000)
        open_note = "" if s["done"] else "  (no DONE: its process ended early)"
        print(f"  {step:<44} {n:>6} {s['PASS']:>6} {s['SKIP']:>5} {s['FAIL']:>5} {secs(wall):>9}{open_note}")
    total = {k: sum(s[k] for s in steps.values()) for k in STATUSES}
    print(f"  {'all':<44} {len(tests):>6} {total['PASS']:>6} {total['SKIP']:>5} {total['FAIL']:>5}")
    if tests:
        print("gate: the 10 slowest tests")
        for _, step, _, ms, name in sorted(tests, key=lambda r: -r[3])[:10]:
            print(f"  {secs(ms):>9}  {step:<44} {name}")

    blocks, lines = error_blocks(build_log)
    failed = [r for r in tests if r[2] == "FAIL"]
    named = {r[4] for r in failed}
    others = [n for n in blocks if n not in named]
    if failed or others:
        print(f"gate: {len(failed) + len(others)} failed")
        for _, step, _, ms, name in failed:
            print(f"  FAIL [{step}] {name} ({secs(ms)})")
            for line in blocks.get(name, ["(no error block in the build log)"]):
                print(f"      {line}")
        for name in others:
            print(f"  FAIL [no log line: its process ended inside it] {name}")
            for line in blocks[name]:
                print(f"      {line}")
    elif step_failed:
        print("gate: the test step failed outside any test; its last 40 lines:")
        for line in lines[-41:]:
            print(f"  {line}")


def main(argv):
    if len(argv) == 3 and argv[0] == "follow":
        follow(argv[1], argv[2])
    elif len(argv) in (3, 4) and argv[0] == "summary" and argv[3:] in ([], ["--step-failed"]):
        summary(argv[1], argv[2], argv[3:] == ["--step-failed"])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
