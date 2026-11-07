#!/usr/bin/env python3
# sushi-test: tier=hermetic
"""Guard for tests/tape.py: the comparison names the first differing case and position, an unstable case is held to its
invariant, log timings never fail a check, and a capture differs by the byte. Stdlib only; silent on pass.

    python3 tests/test_tape.py
"""
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("tape", Path(__file__).resolve().parent / "tape.py")
tape = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tape)


def chat(tokens, text=None):
    return {"choices": [{"message": {"content": text or "".join(tokens)}, "finish_reason": "length",
                         "logprobs": {"content": [{"token": t, "logprob": -0.5, "top_logprobs": [{"token": t, "logprob": -0.5}]}
                                                  for t in tokens]}}],
            "usage": {"completion_tokens": len(tokens), "prompt_tokens": 7, "prompt_tokens_details": {"cached_tokens": 0}}}


def entry(tokens=("a", "b", "c"), log=("[prefix] reused 5 of 9 prompt tokens from SSD",)):
    return {"response": tape.response_record(chat(list(tokens))), "log": list(log)}


class Tape(unittest.TestCase):
    def test_equal_entries_agree(self):
        self.assertIsNone(tape.compare_case("serial", entry(), entry()))

    def test_changed_logprob_row_names_the_position(self):
        got = entry()
        got["response"]["positions"][1]["h"] = "0" * 16
        self.assertIn("serial position 1", tape.compare_case("serial", entry(), got))

    def test_changed_token_names_both_tokens(self):
        d = tape.compare_case("serial", entry(), entry(("a", "z", "c")))
        self.assertIn("position 1", d)
        self.assertIn("'b' vs 'z'", d)

    def test_log_timing_is_not_a_difference_but_a_restore_is(self):
        a = entry(log=["[hot-cache] restored in 12 ms, 3.5 GB/s", "[prefix] reused 5 of 9 prompt tokens from SSD"])
        b = entry(log=[tape.log_lines("[hot-cache] restored in 99 ms, 8.1 GB/s")[0], a["log"][1]])
        a["log"][0] = tape.log_lines(a["log"][0])[0]
        self.assertIsNone(tape.compare_case("hot", a, b))
        b["log"][1] = "[prefix] reused 0 of 9 prompt tokens from SSD"
        self.assertIn("log line 1", tape.compare_case("hot", a, b))

    def test_unstable_case_needs_only_the_serial_prefix(self):
        want = {"unstable": True}
        serial = {"p0": {"text": "The quick brown fox jumps over the lazy dog again and again"}}
        ok = {"rows": [{"text": "The quick brown fox jumps over the lazy dog, then slept"}], "prompts": ["p0"]}
        bad = {"rows": [{"text": "Completely different words entirely, none shared"}], "prompts": ["p0"]}
        self.assertIsNone(tape.compare_case("batch2", want, ok, serial_of=serial))
        self.assertIn("batch2 row 0", tape.compare_case("batch2", want, bad, serial_of=serial))

    def test_settle_keeps_only_what_every_run_agrees_on(self):
        p1 = {"response": entry()["response"], "log": ["x"]}
        p2 = {"response": entry()["response"], "log": ["y"]}
        e = tape.settle([p1, p2])
        self.assertIn("response", e)
        self.assertNotIn("log", e)
        p3 = {"response": entry(("a", "q", "c"))["response"], "log": ["x"]}
        self.assertTrue(tape.settle([p1, p3])["unstable"])

    def test_capture_difference_names_file_and_index(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d, "a"), Path(d, "b")
            for r in (a, b):
                (r / "prompts/00_x").mkdir(parents=True)
                (r / "prompts/00_x/logits.f32").write_bytes(bytes(32))
            self.assertIsNone(tape.cmp_captures(a, b))
            (b / "prompts/00_x/logits.f32").write_bytes(bytes(8) + b"\x01" + bytes(23))
            self.assertIn("prompts/00_x/logits.f32 differs at f32 2 of 8", tape.cmp_captures(a, b))


if __name__ == "__main__":
    out = io.StringIO()
    result = unittest.TextTestRunner(stream=out, verbosity=0).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tape))
    if not result.wasSuccessful():
        sys.stderr.write(out.getvalue())
        sys.exit(1)
