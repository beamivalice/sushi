#!/usr/bin/env python3
# sushi-test: tier=tool
"""Scope guards for the Zig sources. Default mode checks; --write regenerates the allowlists in tests/guards/.

    python3 tests/suite_guards.py [--root DIR] [--write | --map]

--map prints the folder map of docs/testsuite.md: test counts per folder and tier.

Arch-test guard: a test whose name mentions a model must live in that model's folder (src/<model>/) unless its
name is in tests/guards/arch-tests.allow (shared at arch geometry). The list keys the name, not the file, so
moving a test between shared files never breaks it.

Model footprint guard: shared sources (src/ root, src/exl3, src/encoder) may name a model only at dispatch points.
tests/guards/model-footprint.allow holds, per model, how many mentions the shared sources may keep. A count above
its entry fails; below it prints the number to lower. The entry keys the model, not the file, so code moving
between shared files keeps its count. Comments, test blocks, whole test files and src/<model>/ do not count;
engine features (SWA, ring cache, FP8) are not model names.
"""
import argparse
import os
import re
import sys

MODELS = {
    "qwen4": re.compile(r"(?<![a-z])qwen", re.I),
    "mimo2": re.compile(r"(?<![a-z])mimo", re.I),
    "glm5": re.compile(r"(?<![a-z])glm", re.I),
    "dsv41": re.compile(r"(?<![a-z])(dsv41|deepseek)", re.I),
}
ALLOW_DIR = "tests/guards"
ARCH_ALLOW = f"{ALLOW_DIR}/arch-tests.allow"
FOOTPRINT_ALLOW = f"{ALLOW_DIR}/model-footprint.allow"
SHARED_DIRS = [("src", False), ("src/exl3", True), ("src/encoder", True)]


def blank(text, keep_strings):
    """The text with comments (and, unless keep_strings, string and char contents) replaced by spaces, newlines kept."""
    out = list(text)
    i, n = 0, len(text)

    def wipe(a, b):
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = text[i]
        if c == "/" and text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            wipe(i, j)
            i = j
        elif c == "\\" and text.startswith("\\\\", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            if not keep_strings:
                wipe(i, j)
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            if not keep_strings:
                wipe(i + 1, min(j, n))
            i = j + 1
        else:
            i += 1
    return "".join(out)


TEST_HEAD = re.compile(r'(?<![\w.@])test\s*(?:"(\s*)"|[A-Za-z_]\w*)?\s*\{')


def tests_of(text):
    """(name or None, start, end) of every test block, found on a view whose strings are blanked."""
    shape = blank(text, False)
    named = blank(text, True)
    found = []
    for m in TEST_HEAD.finditer(shape):
        depth, k = 1, m.end()
        while k < len(shape) and depth:
            depth += {"{": 1, "}": -1}.get(shape[k], 0)
            k += 1
        name = None
        if m.group(1) is not None:
            q = shape.index('"', m.start(), m.end())
            name = named[q + 1 : q + 1 + len(m.group(1))]
        found.append((name, m.start(), k))
    return found, named


def zig_files(root, dirs, recurse):
    for d in dirs:
        base = os.path.join(root, d)
        if not os.path.isdir(base):
            continue
        for dp, dn, fn in os.walk(base):
            if not recurse:
                dn[:] = []
            dn.sort()
            for f in sorted(fn):
                if f.endswith(".zig"):
                    yield os.path.relpath(os.path.join(dp, f), root)


def model_dirs(root):
    return [m for m in ("qwen4", "mimo2", "glm5", "dsv41") if os.path.isdir(os.path.join(root, "src", m))]


def all_tests(root):
    """(rel path, line, name) of every named test under src/."""
    for rel in zig_files(root, ["src"], True):
        with open(os.path.join(root, rel)) as f:
            text = f.read()
        found, _ = tests_of(text)
        for name, start, _end in found:
            if name is not None:
                yield rel, text.count("\n", 0, start) + 1, name


def folder_model(rel):
    parts = rel.split("/")
    return parts[1] if len(parts) > 2 and parts[1] in MODELS else None


def arch_violations(root):
    """Tests named for a model that live outside every model folder they name: {name: [file:line]}."""
    bad = {}
    for rel, line, name in all_tests(root):
        named = {m for m, rx in MODELS.items() if rx.search(name)}
        if named and folder_model(rel) not in named:
            bad.setdefault(name, []).append(f"{rel}:{line}")
    return bad


def read_allow(root, rel):
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]


def write_allow(root, rel, header, lines):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(f"# {h}\n" for h in header) + "".join(l + "\n" for l in lines))


def arch_check(root, write):
    bad = arch_violations(root)
    if write:
        write_allow(root, ARCH_ALLOW, [
            "Tests named for a model that stay outside src/<model>/ because they test shared code at that model's geometry.",
            "Generated: python3 tests/suite_guards.py --write. One test name per line; it keys the name, not the file.",
        ], sorted(bad))
        return 0
    allowed = set(read_allow(root, ARCH_ALLOW))
    fail = 0
    for name in sorted(set(bad) - allowed):
        fail += 1
        print(f"arch-test guard: {bad[name][0]}: test \"{name}\" names a model but is outside its folder; "
              f"move it into src/<model>/ or list it with --write", file=sys.stderr)
    for name in sorted(allowed - set(bad)):
        print(f"arch-test guard: \"{name}\" is no longer outside its folder; run --write to drop it", file=sys.stderr)
    return fail


def is_test_file(rel):
    stem = os.path.basename(rel)[: -len(".zig")]
    return stem.endswith("_test") or stem.startswith(("test_", "tests_"))


def footprint(root):
    """({model: mentions}, {model: {file: mentions}}) over the shared sources' non-test code."""
    total = {m: 0 for m in MODELS}
    by_file = {m: {} for m in MODELS}
    for d, recurse in SHARED_DIRS:
        for rel in zig_files(root, [d], recurse):
            if is_test_file(rel):
                continue
            with open(os.path.join(root, rel)) as f:
                found, text = tests_of(f.read())
            for _, a, b in found:
                text = text[:a] + " " * (b - a) + text[b:]
            for m, rx in MODELS.items():
                n = len(rx.findall(text))
                if n:
                    total[m] += n
                    by_file[m][rel] = n
    return total, by_file


def footprint_check(root, write):
    total, by_file = footprint(root)
    if write:
        write_allow(root, FOOTPRINT_ALLOW, [
            "Mentions of each model in the shared sources (src/ root, src/exl3, src/encoder), outside comments and tests.",
            "The count may only shrink. Generated: python3 tests/suite_guards.py --write. Keyed by model, not file.",
        ], [f"{m} {total[m]}" for m in MODELS])
        return 0
    allowed = {l.split()[0]: int(l.split()[1]) for l in read_allow(root, FOOTPRINT_ALLOW)}
    fail = 0
    for m in MODELS:
        have, may = total[m], allowed.get(m, 0)
        if have > may:
            fail += 1
            worst = ", ".join(f"{f} {n}" for f, n in sorted(by_file[m].items(), key=lambda kv: -kv[1])[:3])
            print(f"model footprint guard: {m}: {have} mention{'s' * (have != 1)} in shared sources, allows {may}; "
                  f"name a model only in its src folder or at a dispatch point (largest: {worst})", file=sys.stderr)
        elif have < may:
            print(f"model footprint guard: lower {m} to {have} (was {may}): run --write", file=sys.stderr)
    return fail


FOLDERS = [
    ("src", "shared (every model)"), ("src/exl3", "shared (the sushi_exl3 module)"), ("src/encoder", "shared (the encoder, `-Dmodel=enc`)"),
    ("src/qwen4", "qwen4"), ("src/mimo2", "mimo2"), ("src/glm5", "glm5"), ("src/dsv41", "dsv41"),
]
PACK_MARKS = ("SkipZigTest", "getenv", "packPath")


def host_files(root):
    path = os.path.join(root, "src/tests_host.zig")
    with open(path) as f:
        text = f.read()
    return {"src/" + p for p in re.findall(r'_ = @import\("([^"]+\.zig)"\);', text)}


def tier_of(rel, body, host):
    """host: a src/tests_host.zig file. pack: the body reads an env var or skips. gpu: the body needs Metal. cpu: the rest.
    A test that skips only inside a helper counts as gpu or cpu."""
    if rel in host:
        return "host"
    if any(k in body for k in PACK_MARKS):
        return "pack"
    return "gpu" if "requireGpu" in body else "cpu"


def folder_map(root):
    host = host_files(root)
    rows = []
    for d, model in FOLDERS:
        counts = {"host": 0, "gpu": 0, "pack": 0, "cpu": 0}
        files = 0
        for rel in zig_files(root, [d], d != "src"):
            with open(os.path.join(root, rel)) as f:
                text = f.read()
            found, named = tests_of(text)
            files += any(n is not None for n, _, _ in found)
            for name, a, b in found:
                if name is not None:
                    counts[tier_of(rel, named[a:b], host)] += 1
        rows.append((d, model, files, counts))
    out = ["| folder | model | test files | host | gpu | pack | cpu | tests |", "|---|---|---|---|---|---|---|---|"]
    for d, model, files, c in rows:
        out.append(f"| `{d}/` | {model} | {files} | {c['host']} | {c['gpu']} | {c['pack']} | {c['cpu']} | {sum(c.values())} |")
    tot = {k: sum(r[3][k] for r in rows) for k in rows[0][3]}
    out.append(f"| total | | {sum(r[2] for r in rows)} | {tot['host']} | {tot['gpu']} | {tot['pack']} | {tot['cpu']} | {sum(tot.values())} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--write", action="store_true", help="regenerate the allowlists instead of checking")
    ap.add_argument("--map", action="store_true", help="print the folder map table")
    args = ap.parse_args()
    if args.map:
        print(folder_map(args.root))
        return
    fail = arch_check(args.root, args.write) + footprint_check(args.root, args.write)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
