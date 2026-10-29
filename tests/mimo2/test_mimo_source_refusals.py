#!/usr/bin/env python3
"""Check raw-source refusals without loading model weights."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    source = os.environ.get("MIMO_V2_SOURCE")
    if not source:
        print("SKIP: set MIMO_V2_SOURCE to the original HF checkpoint", file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parents[2]
    binary = root / "zig-out/bin/sushi"
    with tempfile.TemporaryDirectory(prefix="mimo-refusals-", dir=root / ".zig-cache") as home:
        env = dict(os.environ, HOME=home)
        args = [
            str(binary), "--model", source, "--no-mtp",
            "--no-vision", "--ctx-size", "4096",
        ]
        cases = [
            (["--ssd-budget-gb", "100"], "requires expert streaming"),
            (["--serve", "--host", "127.0.0.1", "--port", "0"], "ExpertStreamingRequired"),
        ]
        for flags, message in cases:
            result = subprocess.run(
                args + flags, env=env, capture_output=True, text=True, timeout=60,
            )
            output = result.stdout + result.stderr
            assert result.returncode != 0, output
            assert message in output, output
            assert "Loading weights..." not in output, output
            assert "[mimo-source] loading original shards" not in output, output
    return 0


if __name__ == "__main__":
    sys.exit(main())
