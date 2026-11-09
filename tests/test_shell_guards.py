#!/usr/bin/env python3
# sushi-test: tier=hermetic
"""Guard for tests/shell_guards.py: each pipefail-early-exit form fails on a planted script in a scratch tree, the near
misses pass, the allowlist excuses one hit and rejects a stale or reasonless entry, and the repo passes. Stdlib only;
a passing run prints nothing.

    python3 tests/test_shell_guards.py
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "shell_guards.py")
PF = "#!/bin/bash\nset -euo pipefail\n"

RED = {
    "-q": 'blog | grep -q READY\n',
    "-qE": 'blog | grep -qE "a|b"\n',
    "-Eq": 'blog | grep -Eq "a|b"\n',
    "-qi": 'blog | grep -qi ready\n',
    "--quiet": 'blog | grep --quiet READY\n',
    "-m 1": 'blog | grep -m 1 READY\n',
    "-m1 in a cluster": 'blog | grep -im1 READY\n',
    "egrep -q": 'blog | egrep -q "a|b"\n',
    "a later stage of a longer pipe": 'blog | tail -n 50 | grep -q READY\n',
    "the pipe at the end of the line": 'blog |\n    grep -q READY\n',
    "a backslash continuation": 'blog \\\n    | grep -q READY\n',
    "inside $(...) in a string": 'check "x" "$(curl -s "$URL" | grep -q ok && echo 1 || echo 0)"\n',
    "after && in an if": 'if [ -f x ] && blog | grep -q READY; then :; fi\n',
    "a flag after the pattern": 'blog | grep READY -q\n',
    "a pattern arg before the flag": 'blog | grep -e READY -q\n',
    "head -1": 'x=$(blog | head -1)\n',
    "head -n 5": 'blog | head -n 5\n',
    "head -c 200": 'blog | head -c 200\n',
    "head with no count": 'blog | head\n',
    "head after a grep": 'blog | grep READY | head -3\n',
    "sed 1q": "blog | sed 1q\n",
    "sed q": "blog | sed -n '/READY/{p;q}'\n",
    "sed /re/q": "blog | sed '/READY/q'\n",
    "sed -e 5q": "blog | sed -e '5q'\n",
    "sed with a prefix command": "blog | sed -n '1p;2q'\n",
    "awk exit": "blog | awk '/READY/{print; exit}'\n",
    "awk with exit in a condition": "blog | awk 'NR==2 {exit} {print}'\n",
}
GREEN = {
    "grep -c to /dev/null": 'blog | grep -c READY >/dev/null\n',
    "grep -cE to /dev/null": 'blog | grep -cE "a|b" >/dev/null\n',
    "grep on a file": 'grep -q READY "$LOG"\n',
    "a herestring": 'grep -q READY <<<"$OUT"\n',
    "a plain pipe into grep": 'blog | grep READY\n',
    "a pattern that looks like a flag, after --": 'blog | grep -- -q\n',
    "-q as the argument of -e": 'blog | grep -e -q\n',
    "grep -A1 and a pipe on": 'blog | grep -A1 READY | tail -1\n',
    "a comment": '# blog | grep -q READY\n',
    "a trailing comment": 'blog | grep READY # not | grep -q\n',
    "a single-quoted string": "echo 'blog | grep -q READY'\n",
    "a double-quoted string": 'echo "blog | grep -q READY"\n',
    "a heredoc body": "cat <<'EOF'\nblog | grep -q READY\nEOF\n",
    "grep -q after ||": 'blog || grep -q READY file\n',
    "a case pattern": 'case "$x" in a) blog | grep -c a >/dev/null ;; esac\n',
    "sed -n 1p": "blog | sed -n 1p\n",
    "sed -n 1,3p": "blog | sed -n 1,3p\n",
    "a sed substitution holding a q": "blog | sed 's/q/x/;s/a;q/b/'\n",
    "a sed quit-looking word in a pattern": "blog | sed -n '/quit/p'\n",
    "awk that reads everything": "blog | awk 'NR==1{print}'\n",
    "awk with exit as a field name": "blog | awk '{print $exit_code}'\n",
    "tail": "blog | tail -n 1\n",
    "head fed from a file": 'head -n 1 "$LOG"\n',
    "head on the left of a pipe": 'head -c 10 "$LOG" | tr a b\n',
}


def run(root, *args):
    p = subprocess.run([sys.executable, "-I", TOOL, "--root", root, *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def put(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def expect(cond, what, out=""):
    if not cond:
        sys.exit(f"FAIL: {what}\n{out}")


def forms(root):
    for what, body in RED.items():
        put(root, "tests/test_probe.sh", PF + body)
        rc, out = run(root)
        expect(rc == 1 and "tests/test_probe.sh:" in out and "pipefail-early-exit" in out, f"red: {what} fails", out)
    for what, body in GREEN.items():
        put(root, "tests/test_probe.sh", PF + body)
        rc, out = run(root)
        expect(rc == 0 and out == "", f"green: {what} passes silently", out)


def reported_line(root):
    put(root, "tests/test_probe.sh", PF + "a\n\nblog |\n  grep -q READY\n")
    rc, out = run(root)
    expect("tests/test_probe.sh:6:" in out, "the hit names the line of the grep", out)


def scope(root):
    put(root, "tests/test_probe.sh", "#!/bin/bash\nset -u\nblog | grep -q READY\n")
    rc, out = run(root)
    expect(rc == 0, "a script without pipefail is left alone", out)
    rc, out = run(root, "--all")
    expect(rc == 1, "--all lists it anyway", out)
    put(root, "tests/test_probe.sh", "#!/bin/bash\nset -u\nset -o pipefail\nblog | grep -q READY\n")
    expect(run(root)[0] == 1, "set -o pipefail on its own line counts")
    put(root, "tests/test_probe.sh", "#!/bin/bash\n# set -o pipefail\nblog | grep -q READY\n")
    expect(run(root)[0] == 0, "a commented set does not count")
    os.remove(os.path.join(root, "tests/test_probe.sh"))
    put(root, "tests/lib/helper.sh", "wait_ready() { blog | grep -q READY; }\n")
    rc, out = run(root)
    expect(rc == 1 and "tests/lib/helper.sh:1:" in out, "a sourced lib counts even without its own pipefail", out)
    os.remove(os.path.join(root, "tests/lib/helper.sh"))
    put(root, "scripts/tool.sh", PF + "blog | grep -q READY\n")
    rc, out = run(root)
    expect(rc == 1 and "scripts/tool.sh:3:" in out, "scripts/ is scanned too", out)
    os.remove(os.path.join(root, "scripts/tool.sh"))


def allowlist(root):
    allow = "tests/guards/pipefail-early-exit.allow"
    put(root, "tests/test_probe.sh", PF + "tiny | grep -q READY\n")
    put(root, allow, "# header\ntests/test_probe.sh:3 the writer emits one short line and has exited before grep reads\n")
    rc, out = run(root)
    expect(rc == 0 and out == "", "an allowlisted hit with a reason passes", out)
    put(root, allow, "tests/test_probe.sh:3\n")
    rc, out = run(root)
    expect(rc == 1 and "no reason" in out, "an entry without a reason fails", out)
    put(root, allow, "tests/test_probe.sh:9 stale\n")
    rc, out = run(root)
    expect(rc == 1 and "matches nothing" in out and "test_probe.sh:3" in out, "a stale entry fails and the hit is still reported", out)
    os.remove(os.path.join(root, allow))
    os.remove(os.path.join(root, "tests/test_probe.sh"))


def main():
    with tempfile.TemporaryDirectory() as root:
        forms(root)
        reported_line(root)
        scope(root)
        allowlist(root)
    rc, out = run(os.path.dirname(HERE))
    expect(rc == 0, "the repo passes the pipefail-early-exit guard (use grep -c ... >/dev/null)", out)


main()
