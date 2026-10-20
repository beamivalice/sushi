#!/usr/bin/env python3
"""Per-category tables of `sushi kld compare --json` output.

A prompt's category is its id up to the first '-' (`code-rust-...` -> code). Each cell is position-weighted to the
teacher's first EOS, like the compare's own to-first-EOS line, so the `all` column equals it.

    python3 tests/kld_by_category.py [--csv-dir DIR] <compare.json> [<compare.json> ...]

Prints one table per file (category, KLD to first EOS, top-1, positions), then a KLD matrix with one row per file's
label and one column per category. --csv-dir also writes that matrix as DIR/matrix.csv and the top-1 matrix as
DIR/matrix-top1.csv (columns: model, the categories, all).
"""
import csv
import json
import os
import sys

ORDER = ["code", "math", "reasoning", "design", "wikitext", "multilingual", "story", "stem"]


def by_category(report):
    rows = {}
    for p in report["prompts"]:
        r = rows.setdefault(p["id"].split("-", 1)[0], [0.0, 0, 0])
        r[0] += p["kld_to_eos"] * p["positions_to_eos"]
        r[1] += p["top1_to_eos"]
        r[2] += p["positions_to_eos"]
    rows["all"] = [sum(r[i] for r in rows.values()) for i in range(3)]
    return {c: (kld / n, top1 / n, n) for c, (kld, top1, n) in rows.items() if n}


def columns(tables):
    seen = list(dict.fromkeys(c for _, t in tables for c in t if c != "all"))
    return [c for c in ORDER if c in seen] + [c for c in seen if c not in ORDER] + ["all"]


def render(tables):
    out = []
    for label, t in tables:
        out += [f"### {label}", "", "| category | KLD to first EOS | top-1 | positions |", "|---|---|---|---|"]
        out += [f"| {c} | {t[c][0]:.5f} | {t[c][1] * 100:.2f}% | {t[c][2]} |" for c in columns([(label, t)])]
        out.append("")
    cols = columns(tables)
    out += ["### KLD to first EOS", "", "| model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    out += [f"| {label} | " + " | ".join(f"{t[c][0]:.5f}" if c in t else "" for c in cols) + " |"
            for label, t in tables]
    out.append("")
    return "\n".join(out)


def write_csv(path, tables, index, fmt):
    cols = columns(tables)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + cols)
        for label, t in tables:
            w.writerow([label] + [fmt(t[c][index]) if c in t else "" for c in cols])


def main(argv):
    csv_dir = None
    if argv[:1] == ["--csv-dir"]:
        csv_dir, argv = argv[1], argv[2:]
    if not argv:
        sys.exit(__doc__)
    tables = []
    for path in argv:
        with open(path) as f:
            report = json.load(f)
        tables.append((report["label"], by_category(report)))
    sys.stdout.write(render(tables))
    if csv_dir:
        write_csv(os.path.join(csv_dir, "matrix.csv"), tables, 0, lambda v: f"{v:.6f}")
        write_csv(os.path.join(csv_dir, "matrix-top1.csv"), tables, 1, lambda v: f"{v:.6f}")


if __name__ == "__main__":
    main(sys.argv[1:])
