#!/usr/bin/env python3
# sushi-test: tier=tool
"""Pattern guards for the repo's shell scripts. Default mode checks and prints file:line per finding.

    python3 tests/shell_guards.py [--root DIR] [--all]

pipefail-early-exit: under `set -o pipefail` a pipeline `cmd | reader` fails with 141 when the reader exits before
`cmd` has finished writing (SIGPIPE), although the reader succeeded. The early-exit readers are `grep -q`, `--quiet`,
`--silent` and `-m N`; `head` in any form; `sed` with a quit command (`q`, `Nq`, `/re/q`); `awk` with `exit`. A pipe
into one is refused in every shell script under tests/ and scripts/ that sets pipefail, and in tests/lib/ (sourced by
scripts that may). Read everything instead: `grep -c PAT >/dev/null`, `sed -n 1p`, `awk 'NR==1'`, or give the reader
the file. tests/guards/pipefail-early-exit.allow holds `<file>:<line> <reason>` for one deliberate hit (a writer
that is finite and small); an entry that matches nothing, or has no reason, fails. --all ignores the pipefail
condition and lists every hit.
"""
import argparse
import os
import re
import sys

ALLOW = "tests/guards/pipefail-early-exit.allow"
GREPS = {"grep", "egrep", "fgrep", "rg", "ggrep"}
ARG_TAKING = set("efAmBCdD")
SED_QUIT = re.compile(r"(^|[;{}\s])(\$|\d+|\d+,\d+|/[^/]*/)?!?\s*[qQ]\d*\s*($|[;}])")
AWK_EXIT = re.compile(r"\bexit\b")
SCAN_DIRS = ("tests", "scripts")
HEREDOC = re.compile(r"<<(-?)\s*(?:'([^']*)'|\"([^\"]*)\"|(\\?[\w.-]+))")


def tokens(text):
    """[(kind, value, line)] with kind 'word' or 'op' ('|', '||', '&&', ';', '&', '(', ')', 'nl').

    Quote- and $(...)-aware; comments and heredoc bodies are skipped. Not a full shell parser: enough to see
    which command each unquoted pipe feeds."""
    i, n, line = 0, len(text), 1
    stack = []  # 'dq' or 'cmd' per open quote or ( / $(; empty at the top level
    word, wline, quoted = [], 0, False
    heredocs = []
    out = []

    def flush():
        nonlocal word, quoted
        if word or quoted:
            out.append(("word", "".join(word), wline))
        word, quoted = [], False

    def op(v):
        flush()
        out.append(("op", v, line))

    def add(s):
        nonlocal wline
        if not word and not quoted:
            wline = line
        word.extend(s)

    while i < n:
        c = text[i]
        if stack and stack[-1] == "dq":
            if c == "\\" and i + 1 < n:
                add(text[i + 1])
                line += text[i + 1] == "\n"
                i += 2
            elif c == '"':
                stack.pop()
                i += 1
            elif c == "$" and text.startswith("$(", i):
                op("(")
                stack.append("cmd")
                i += 2
            else:
                add(c)
                line += c == "\n"
                i += 1
            continue
        if c == "\n":
            op("nl")
            line += 1
            i += 1
            for term, strip in heredocs:
                while i < n:
                    j = text.find("\n", i)
                    j = n if j < 0 else j
                    body = text[i:j]
                    i = min(j + 1, n)
                    line += 1
                    if (body.strip() if strip else body) == term:
                        break
            heredocs = []
        elif c == "\\" and i + 1 < n:
            if text[i + 1] == "\n":
                line += 1
            else:
                add(text[i + 1])
            i += 2
        elif c == "'":
            j = text.find("'", i + 1)
            j = n if j < 0 else j
            if not word:
                wline = line
            quoted = True
            word.extend(text[i + 1 : j])
            line += text.count("\n", i + 1, j)
            i = j + 1
        elif c == '"':
            if not word:
                wline = line
            quoted = True
            stack.append("dq")
            i += 1
        elif c == "#" and not word and not quoted:
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif c in " \t":
            flush()
            i += 1
        elif text.startswith("<<<", i):
            i += 3
        elif text.startswith("<<", i) and HEREDOC.match(text, i):
            m = HEREDOC.match(text, i)
            heredocs.append(((m.group(2) or m.group(3) or m.group(4)).lstrip("\\"), bool(m.group(1))))
            i = m.end()
        elif c == "|":
            width = 2 if text[i + 1 : i + 2] in ("|", "&") else 1
            op("||" if text.startswith("||", i) else "|")
            i += width
        elif c in "&;":
            if text[i : i + 2] in ("&&", ";;"):
                op(text[i : i + 2])
                i += 2
            elif word and word[-1] in "<>" or text[i + 1 : i + 2] == ">":
                add(c)
                i += 1
            else:
                op(c)
                i += 1
        elif c == "(":
            op("(")
            stack.append("cmd")
            i += 1
        elif c == ")":
            op(")")
            if stack:
                stack.pop()
            i += 1
        else:
            add(c)
            i += 1
    flush()
    return out


def grep_early(args):
    """Whether these grep arguments make grep exit at its first match."""
    args, i = list(args), 0
    while i < len(args):
        a = args[i]
        i += 1
        if a == "--":
            return False
        if a in ("--quiet", "--silent") or a.startswith("--max-count"):
            return True
        if a.startswith("--") or not a.startswith("-") or len(a) < 2:
            continue
        for ch in a[1:]:
            if ch in "qm":
                return True
            if ch in ARG_TAKING:
                i += 1
                break
    return False


def early_reader(name, args):
    """The reason this command stops reading before its input ends, or None."""
    args = list(args)
    if name in GREPS:
        return "grep that quits at its first match" if grep_early(args) else None
    if name == "head":
        return "head, which stops after its count"
    if name == "sed":
        return "sed with a quit command" if any(SED_QUIT.search(a) for a in args if not a.startswith("-") or " " in a) else None
    if name in ("awk", "gawk", "mawk"):
        return "awk with exit" if any(AWK_EXIT.search(a) for a in args) else None
    return None


def hits(text):
    """(line, reason) of every pipe into a command that can exit before its input ends."""
    toks = tokens(text)
    found = []
    for k, (kind, val, _) in enumerate(toks):
        if kind != "op" or val != "|":
            continue
        j = k + 1
        while j < len(toks) and toks[j][:2] == ("op", "nl"):
            j += 1
        words = []
        while j < len(toks) and toks[j][0] == "word":
            words.append(toks[j])
            j += 1
        while words and words[0][1] in ("command", "builtin", "exec", "!"):
            words.pop(0)
        if words:
            why = early_reader(os.path.basename(words[0][1]), (w[1] for w in words[1:]))
            if why:
                found.append((words[0][2], why))
    return found


def sets_pipefail(text):
    toks = tokens(text)
    for k, (kind, val, _) in enumerate(toks):
        if kind == "word" and val in ("set", "shopt") and (k == 0 or toks[k - 1][0] == "op"):
            j = k + 1
            while j < len(toks) and toks[j][0] == "word":
                if toks[j][1] == "pipefail":
                    return True
                j += 1
    return False


def shell_files(root):
    for d in SCAN_DIRS:
        for dp, dn, fn in os.walk(os.path.join(root, d)):
            dn[:] = sorted(x for x in dn if x not in ("fixtures", "node_modules", "lint") and not x.startswith("."))
            for f in sorted(fn):
                if f.endswith(".sh"):
                    yield os.path.relpath(os.path.join(dp, f), root)


def read_allow(root):
    path = os.path.join(root, ALLOW)
    allow = {}
    if os.path.exists(path):
        with open(path) as f:
            for l in f:
                if l.strip() and not l.startswith("#"):
                    key, _, reason = l.strip().partition(" ")
                    allow[key] = reason.strip()
    return allow


def check(root, everything):
    allow, used, fail = read_allow(root), set(), 0
    for rel in shell_files(root):
        with open(os.path.join(root, rel), errors="replace") as f:
            text = f.read()
        if not everything and not (rel.startswith("tests/lib/") or sets_pipefail(text)):
            continue
        for line, why in hits(text):
            key = f"{rel}:{line}"
            if key in allow:
                used.add(key)
                continue
            fail += 1
            print(f"pipefail-early-exit: {key}: a pipe into {why}; under pipefail the writer's SIGPIPE (141) fails the "
                  f"pipeline. Read it all (grep -c ... >/dev/null, sed -n 1p), or give the reader the file", file=sys.stderr)
    for key, reason in sorted(allow.items()):
        if key not in used:
            fail += 1
            print(f"pipefail-early-exit: {ALLOW}: entry {key} matches nothing; delete it", file=sys.stderr)
        elif not reason:
            fail += 1
            print(f"pipefail-early-exit: {ALLOW}: entry {key} has no reason", file=sys.stderr)
    return 1 if fail else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    sys.exit(check(a.root, a.all))


if __name__ == "__main__":
    main()
