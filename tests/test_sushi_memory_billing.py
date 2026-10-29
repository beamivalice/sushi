#!/usr/bin/env python3
# sushi-test: tier=live boot=own pack=qwen4 boots=4
"""Live Sushi memory billing, KV override and context refusal, on the default Qwen pack and, when present, MiMo.

Per pack, two resident boots at a 1248-token context (MTP + vision on, and `--no-mtp --no-vision`):
  - the enabled boot holds more resident bytes than the disabled one, advertises images, and reports the MTP head
    in /v1/models and /props; KV8 is the default and a per-request KV4 override is honoured and logged;
  - the load bills `[preflight] weights ~W GB, needs ~N GB`: N is at least W plus the 1 GiB net plus the measured
    warmup, `/props` peak memory fits N and sits within 3 GB of it, and a short chat completes.
Then, without loading weights: a startup refusal at an absurd context names the maximum context before any shard
loads, and a headless `--model-dir` server answers `/v1/load-model` with a 503 `out_of_memory` naming it, twice.
`--refusals-only` runs just the last two.
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import sushi_boot as b  # noqa: E402

CTX = 1248
ABSURD_CTX = 4_000_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refusals-only", action="store_true")
    args = parser.parse_args()
    models = Path(os.environ.get("SUSHI_MODELS_DIR", str(Path.home() / ".sushi/models")))
    packs = [p for p in (b.pack("qwen4"), b.pack("mimo2", required=False)) if p is not None]
    out = b.run_dir()
    common = ["--prefill-chunk", "512", "--max-tokens", "8", "--prefix-cache-entries", "0", "--no-drafter"]

    def settle():
        time.sleep(2)  # Metal unwires buffers asynchronously after a process exits

    def exercise(server, name, enabled, base_bytes):
        base = server.base
        status, listing = b.request(base, "/v1/models")
        assert status == 200, listing
        loaded = next(m for m in listing["data"] if m.get("loaded"))
        assert loaded["meta"]["mtp_loaded"] == enabled, loaded["meta"]
        assert ("image" in loaded["input_modalities"]) == enabled, loaded["input_modalities"]
        if enabled:
            assert loaded["bytes_resident"] > base_bytes, (loaded["bytes_resident"], base_bytes)
        status, props = b.request(base, "/props")
        assert status == 200, props
        assert props["default_generation_settings"]["n_ctx"] == CTX, props["default_generation_settings"]
        assert props["settings"]["kv_cache"]["scheme"] == "kv8", props["settings"]
        assert props["settings"]["kv_quant"] == "8", props["settings"]
        assert props["settings"]["mtp"]["loaded"] == enabled, props["settings"]["mtp"]
        assert props["settings"]["mtp"]["default_on"] == enabled, props["settings"]["mtp"]
        found = re.findall(r"\[preflight\] weights ~([0-9.]+) GB, needs ~([0-9.]+) GB", server.log_text())
        assert len(found) == 1, found
        weights, needed = map(float, found[0])
        # The billed weights plus the larger of the measured warmup plus the 1 GiB net and the context bill.
        warmup = 0.25 if name.startswith("MiMo") else (1.25 if enabled else 0.125)
        assert needed - weights >= 1.0 + warmup - 0.01, (weights, needed, warmup)
        assert 0 < props["memory"]["active_bytes"] <= props["memory"]["peak_bytes"], props["memory"]
        peak = props["memory"]["peak_bytes"] / 2**30
        assert peak <= needed + 0.005, props["memory"]
        assert needed - peak <= 3.0, (needed, peak)
        (out / f"{server.label}.props.json").write_text(json.dumps(props, indent=2))
        status, reply = b.request(base, "/v1/chat/completions", {
            "model": loaded["id"], "messages": [{"role": "user", "content": "Reply with one greeting."}],
            "kv_quant": 4, "max_tokens": 8, "temperature": 0, "stream": False, "enable_thinking": False,
        })
        assert status == 200 and reply.get("choices"), reply
        assert reply["usage"]["completion_tokens"] > 0, reply
        assert "kv-quant override: affine 4-bit (per-request)" in server.log_text()
        print(f"PASS {server.label}: {loaded['bytes_resident']} bytes, KV8 default, KV4 request, weights {weights} GB, "
              f"needs {needed} GB, peak {peak:.2f} GB", flush=True)
        return loaded["bytes_resident"]

    if not args.refusals_only:
        for pack in packs:
            disabled_bytes = 0
            for enabled in (False, True):
                label = f"billing-{pack.name}-{'enabled' if enabled else 'disabled'}"
                flags = common + ["--ctx-size", str(CTX), "--kv-quant", "8"] + (["--mtp"] if enabled else ["--no-mtp", "--no-vision"])
                with b.Server(label, model=pack, flags=flags) as server:
                    server.wait_ready()
                    got = exercise(server, pack.name, enabled, disabled_bytes)
                    if not enabled:
                        disabled_bytes = got
                settle()

    for pack in packs:
        label = f"billing-{pack.name}-startup-refusal"
        server = b.Server(label, model=pack, flags=common + ["--ctx-size", str(ABSURD_CTX), "--mtp"]).start()
        try:
            code = server.wait_exit(120)
        finally:
            server.stop(gone_ok=True)
        text = server.log.read_text()
        assert code > 0, f"expected a clean refusal, got exit {code}: {label}"
        maximum = re.search(r"Maximum context with this memory budget: (\d+) tokens", text)
        assert maximum, f"numeric startup refusal missing; see {server.log}"
        assert "Loading model-" not in text, "refusal happened after shard loading"
        print(f"PASS {label}: maximum context {maximum.group(1)}", flush=True)
        settle()

    with b.Server("billing-cold-refusal", model_dir=models, flags=common + ["--ctx-size", str(ABSURD_CTX), "--mtp"]) as server:
        server.wait_ready()
        for pack in packs:
            for attempt in range(2):
                status, body = b.request(server.base, "/v1/load-model", {"model": pack.name})
                assert status == 503 and body["error"]["type"] == "out_of_memory", (status, body)
                maximum = re.search(r"Maximum context: (\d+) tokens", body["error"]["message"])
                assert maximum, body
                print(f"PASS {pack.name} cold refusal {attempt}: maximum context {maximum.group(1)}", flush=True)
        _, props = b.request(server.base, "/props")
        assert props["memory"]["active_bytes"] < 512 * 1024 * 1024, props["memory"]
        assert "Loading model-" not in server.log_text()
    return 0


if __name__ == "__main__":
    sys.exit(main())
