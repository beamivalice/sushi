#!/usr/bin/env python3
# sushi-test: tier=tool
"""Compare two restore tapes (`SUSHI_RESTORE_TAPE`): exit 0 when every row is equal, 1 naming the first row and field
that differ. A tape is JSONL; rows are `lookup`, `tier` and `gen`, in the order the server wrote them.

    python3 tests/diff_restore_tape.py A.tape B.tape [--ignore FIELD ...]

--ignore drops a field by name from every row, for the fields a change is meant to move (`bytes` in a `tier` row when
a stage changes how the tier bills).
"""
import argparse
import json
import sys


def load(path):
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    sys.exit(f"{path}:{n}: not JSON: {e}")
    return rows


def first_diff(a, b, where=""):
    """The path of the first difference between two JSON values, with both sides; None when equal."""
    if type(a) is not type(b):
        return where, a, b
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{where}.{k}" if where else k, a.get(k, "<absent>"), b.get(k, "<absent>")
            d = first_diff(a[k], b[k], f"{where}.{k}" if where else k)
            if d:
                return d
        return None
    if isinstance(a, list):
        for i in range(min(len(a), len(b))):
            d = first_diff(a[i], b[i], f"{where}[{i}]")
            if d:
                return d
        if len(a) != len(b):
            return f"{where}.length", len(a), len(b)
        return None
    return None if a == b else (where, a, b)


def strip(row, ignore):
    return {k: v for k, v in row.items() if k not in ignore}


def label(row):
    if row is None:
        return "<none>"
    return f"{row.get('row')} request={row.get('request')} seq={row.get('seq', '-')}"


def compare(a_rows, b_rows, ignore=()):
    """None when equal, else a one-line description of the first difference."""
    for i in range(max(len(a_rows), len(b_rows))):
        a = a_rows[i] if i < len(a_rows) else None
        b = b_rows[i] if i < len(b_rows) else None
        if a is None or b is None:
            return f"row {i}: only one tape has it ({label(a)} vs {label(b)})"
        d = first_diff(strip(a, ignore), strip(b, ignore))
        if d:
            field, x, y = d
            return f"row {i} ({label(a)}): field {field}: {x!r} != {y!r}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--ignore", action="append", default=[])
    args = ap.parse_args()
    why = compare(load(args.a), load(args.b), set(args.ignore))
    if why:
        print(why)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
