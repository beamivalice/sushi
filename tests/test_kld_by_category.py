#!/usr/bin/env python3
"""Guard for tests/kld_by_category.py. Stdlib only; a passing run prints nothing.

    python3 tests/test_kld_by_category.py
"""
import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kld_by_category", HERE / "kld_by_category.py")
kbc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kbc)


def prompt(pid, kld, top1, n):
    return {"id": pid, "kld_to_eos": kld, "top1_to_eos": top1, "positions_to_eos": n}


REPORT = {
    "label": "pack-a",
    "mean_kld_to_eos": (0.1 * 100 + 0.3 * 300 + 0.05 * 512) / 912,
    "prompts": [prompt("wikitext", 0.05, 500, 512), prompt("code-python-x", 0.1, 90, 100),
                prompt("code-rust-y", 0.3, 240, 300)],
}
OTHER = dict(REPORT, label="pack-b", prompts=REPORT["prompts"][1:])


class KldByCategory(unittest.TestCase):
    def test_cells_are_position_weighted_and_all_matches_the_compare(self):
        t = kbc.by_category(REPORT)
        self.assertAlmostEqual(t["code"][0], (0.1 * 100 + 0.3 * 300) / 400)
        self.assertAlmostEqual(t["code"][1], 330 / 400)
        self.assertEqual(t["code"][2], 400)
        self.assertAlmostEqual(t["all"][0], REPORT["mean_kld_to_eos"])
        self.assertEqual(t["all"][2], 912)

    def test_columns_follow_the_fixed_order_and_end_with_all(self):
        tables = [("x", kbc.by_category(dict(REPORT, prompts=REPORT["prompts"] + [prompt("misc", 0.1, 1, 2)])))]
        self.assertEqual(kbc.columns(tables), ["code", "wikitext", "misc", "all"])

    def test_matrix_has_one_row_per_model(self):
        text = kbc.render([("pack-a", kbc.by_category(REPORT)), ("pack-b", kbc.by_category(OTHER))])
        self.assertIn("| model | code | wikitext | all |", text)
        self.assertIn("| pack-b | 0.25000 |  | 0.25000 |", text)

    def test_cli_writes_the_kld_and_top1_csv(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for r in (REPORT, OTHER):
                paths.append(Path(d) / f"{r['label']}.json")
                paths[-1].write_text(json.dumps(r))
            out = subprocess.run([sys.executable, str(HERE / "kld_by_category.py"), "--csv-dir", d, *map(str, paths)],
                                 capture_output=True, text=True, check=True).stdout
            kld = list(csv.reader(open(Path(d) / "matrix.csv")))
            top1 = list(csv.reader(open(Path(d) / "matrix-top1.csv")))
        self.assertIn("| code | 0.25000 | 82.50% | 400 |", out)
        self.assertEqual(kld[0], ["model", "code", "wikitext", "all"])
        self.assertEqual(kld[2], ["pack-b", "0.250000", "", "0.250000"])
        self.assertEqual(top1[1][:3], ["pack-a", "0.825000", f"{500 / 512:.6f}"])


if __name__ == "__main__":
    result = unittest.main(argv=[sys.argv[0]], exit=False, testRunner=unittest.TextTestRunner(stream=io.StringIO()))
    if not result.result.wasSuccessful():
        for _, tb in result.result.failures + result.result.errors:
            sys.stderr.write(tb)
        sys.exit(1)
