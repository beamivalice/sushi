#!/usr/bin/env python3
# sushi-test: tier=hermetic
"""Guard for tests/replay_ssd_writes.py: the ceiling reads only the requests marked later, a delta is per kind, a
conversation is the same text on every run (a flow that is not byte-stable cannot be a proof), every flow of the
restore-tape stage exists and names a known pack. Stdlib only; no server; silent on pass.

    python3 tests/test_replay_ssd_writes.py
"""
import importlib.util
import io
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("replay", HERE / "replay_ssd_writes.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

FLOWS = ["agent", "retry", "fork", "far-divergence", "images", "restart", "crash", "budget", "ram",
         "glm-4conv", "dsv41-live", "mimo-ring"]


class ReplayTests(unittest.TestCase):
    def test_every_flow_exists_with_a_known_pack(self):
        self.assertEqual(sorted(FLOWS), sorted(replay.FLOWS))
        for name, (fn, pack, defaults) in replay.FLOWS.items():
            self.assertIn(pack, replay.PACKS, name)
            self.assertTrue(callable(fn))
            self.assertGreater(defaults["turns"], 0)

    def test_written_delta_is_per_kind(self):
        self.assertEqual(replay.written_delta({"kv": 10, "ssm": 5}, {"kv": 4}), {"kv": 6, "ssm": 5})

    def test_ceiling_reads_only_later_requests(self):
        rows = [{"written_mb": 3000.0, "later": False}, {"written_mb": 100.0, "later": True},
                {"written_mb": 300.0, "later": True}]
        self.assertEqual(replay.later_mean(rows), 200.0)
        self.assertIsNone(replay.ceiling_failure(rows, 200.0))
        self.assertIn("passes the ceiling", replay.ceiling_failure(rows, 150.0))
        self.assertIsNone(replay.ceiling_failure(rows, None))
        self.assertIsNone(replay.ceiling_failure([{"written_mb": 9.0, "later": False}], 1.0))

    def test_ceiling_reads_what_the_layer_stored_when_it_reports_it(self):
        rows = [{"written_mb": 400.0, "stored_mb": 0.0, "later": True}]
        self.assertIn("passes the ceiling", replay.ceiling_failure(rows, 300.0))
        rows = [{"written_mb": 400.0, "stored_mb": 90.0, "later": True}, {"written_mb": 400.0, "stored_mb": 110.0, "later": True}]
        self.assertEqual(replay.later_mean(rows, "stored_mb"), 100.0)
        self.assertIsNone(replay.ceiling_failure(rows, 300.0))
        self.assertIn("stored", replay.ceiling_failure(rows, 50.0))
        self.assertIn("passes the ceiling", replay.ceiling_failure([{"written_mb": 400.0, "later": True}], 300.0))

    def test_a_conversation_is_the_same_text_on_every_build(self):
        cfg = replay.PACKS["qwen-stq"]
        a = replay.Conv(0, cfg, 600)
        b = replay.Conv(0, cfg, 600)
        self.assertEqual(a.msgs, b.msgs)
        a.absorb({"content": "x"}, 0)
        b.absorb({"content": "x"}, 0)
        self.assertEqual(a.msgs, b.msgs)
        self.assertNotEqual(a.msgs[1], replay.Conv(1, cfg, 600).msgs[1])

    def test_absorb_adds_a_screenshot_on_its_turn(self):
        cv = replay.Conv(0, replay.PACKS["qwen-stq"], 300, images=2)
        cv.absorb({"content": ""}, 0)
        self.assertEqual(len(cv.msgs), 4)
        cv.absorb({"content": ""}, 1)
        self.assertIsInstance(cv.msgs[-1]["content"], list)


if __name__ == "__main__":
    out = io.StringIO()
    result = unittest.TextTestRunner(stream=out, verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReplayTests))
    if not result.wasSuccessful():
        sys.stderr.write(out.getvalue())
        sys.exit(1)
