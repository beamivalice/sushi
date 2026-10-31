#!/usr/bin/env python3
# sushi-test: tier=host
"""A pair-major pack is refused by name by a build that predates it; this build's load is the Zig loader test's.

SUSHI_OLD_BIN is a `sushi` built before `trellis_pm` existed (a release binary); SUSHI_PM_PACK is a pair-major
DeepSeek pack (default `${DSV41_STQ_TINY}-pm`, the tiny fixture the repacker writes). The load stops at the config and
index, before any weight, so neither run boots a model.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    old = os.environ.get("SUSHI_OLD_BIN")
    pack = os.environ.get("SUSHI_PM_PACK")
    if not pack and os.environ.get("DSV41_STQ_TINY"):
        pack = os.environ["DSV41_STQ_TINY"].rstrip("/") + "-pm"
    if not old or not pack or not Path(pack).is_dir():
        print("SKIP: set SUSHI_OLD_BIN and SUSHI_PM_PACK (or DSV41_STQ_TINY with its -pm sibling)", file=sys.stderr)
        return 3
    args = ["--model", pack, "--no-mtp", "--no-vision", "--ctx-size", "4096", "--serve", "--host", "127.0.0.1", "--port", "0"]
    with tempfile.TemporaryDirectory(prefix="pm-refusal-") as home:
        env = dict(os.environ, HOME=home)
        # The old build finds no trellis tensors of any routed group, so it refuses the pack at the index: it never
        # reads a tile in the wrong order.
        result = subprocess.run([old] + args, env=env, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        assert result.returncode != 0, output
        assert "ExpertLayoutUnsupported" in output or "does not list every routed expert" in output, output
        assert "Loading weights..." not in output, output
    return 0


if __name__ == "__main__":
    sys.exit(main())
