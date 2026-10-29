#!/usr/bin/env python3
# sushi-test: tier=lib
"""The Python twin of tests/lib/boot.sh's attach helpers: SUSHI_URL, SUSHI_LOG, the nonce, the log slice, require."""
import json
import os
import random
import sys
import time
import urllib.request

EXIT_SKIP = 3


def skip(reason):
    print(f"SKIP {os.path.basename(sys.argv[0])}: {reason}")
    sys.exit(EXIT_SKIP)


def url():
    return os.environ["SUSHI_URL"]


def attach():
    """Top of every attach script: records the log offset and returns the nonce every prompt starts with."""
    if not os.environ.get("SUSHI_URL") or not os.environ.get("SUSHI_LOG"):
        sys.exit(f"{os.path.basename(sys.argv[0])}: attach script: run it with tests/run.sh --only <file> "
                 "--model <m>, or set SUSHI_URL and SUSHI_LOG")
    size = os.path.getsize(os.environ["SUSHI_LOG"]) if os.path.exists(os.environ["SUSHI_LOG"]) else 0
    os.environ["SUSHI_LOG_OFFSET"] = str(size)
    os.environ["SUSHI_NONCE"] = f"n{int(time.time())}p{os.getpid()}r{random.randrange(32768)}"
    return os.environ["SUSHI_NONCE"]


def log_since():
    with open(os.environ["SUSHI_LOG"], "rb") as f:
        f.seek(int(os.environ.get("SUSHI_LOG_OFFSET", "0")))
        return f.read().decode("utf-8", "replace")


def get_json(path):
    with urllib.request.urlopen(url() + path, timeout=30) as r:
        return json.load(r)


def _has(cap):
    arch = os.environ.get("SUSHI_ARCH", "")
    if cap in ("mtp", "vision", "spec", "batching"):
        d = get_json("/props")
        if cap == "batching":
            return bool(d["batching"]["supported"])
        if cap == "vision":
            return bool(d["settings"]["vision"]["loaded"])
        return bool(d["settings"]["mtp"]["loaded"] or (cap == "spec" and d["settings"]["drafter"] not in ("none", "")))
    if cap == "video":
        return arch in ("qwen4_exp", "glm5_next") and _has("vision")
    if cap == "thinking":
        return any("reasoning" in m.get("capabilities", []) for m in get_json("/v1/models")["data"])
    if cap == "stq":
        try:
            with open(os.path.join(os.environ["SUSHI_PACK"], "config.json")) as f:
                return json.load(f)["expert_quant"]["format"] == "stq"
        except (KeyError, OSError, ValueError):
            return False
    raise ValueError(f"unknown capability '{cap}'")


def require(cap):
    """Exit 3 (SKIP) when the boot lacks the capability: mtp|vision|video|batching|stq|thinking|spec."""
    try:
        has = _has(cap)
    except (KeyError, OSError, ValueError) as e:
        if isinstance(e, ValueError) and str(e).startswith("unknown"):
            sys.exit(f"require: {e}")
        has = False
    if not has:
        arch = os.environ.get("SUSHI_ARCH", "")
        skip(f"the boot lacks {cap}" + (f" ({arch})" if arch else ""))
