#!/usr/bin/env python3
# sushi-test: tier=tool
"""Checks a pack made by `sushi convert exl3 --trunk stq` against the pack it was made from.

usage: verify_qwen_stq_trunk.py <base pack> <new pack> [trunk n]

A trunk bank is a 3-D `.trellis`; the routed experts of an STQ base pack are 4-D and are not checked here. The trunk banks
share one rate (n 64 for --trunk-k 4, 96 for --trunk-k 6), read from the pack; `trunk n` asserts it. `lm_head` is always n 96.

Reads headers and compares bytes only; no model is loaded. Exits non-zero on the first failed check.
"""
import filecmp
import json
import os
import struct
import sys


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensors(path):
    h, _ = header(path)
    meta = h.pop("__metadata__", {})
    return h, meta


def need(ok, what):
    if not ok:
        print("FAIL:", what)
        sys.exit(1)
    print("ok:", what)


def main(base, new, trunk_n=None):
    cfg = json.load(open(os.path.join(new, "config.json")))
    bcfg = json.load(open(os.path.join(base, "config.json")))
    tq = cfg["trunk_quant"]
    need(tq["format"] == "stq" and tq["codebook"] == "spec1" and tq["window"] == 15 and tq["codebook_table"] == "spec1-w15", "trunk_quant names SPEC1 w15")
    need(float(tq["k"]) == 6.0, "trunk_quant.k is the widest bank (6)")
    need(cfg["expert_quant"] == bcfg["expert_quant"], "expert_quant is the base's")
    need(tq.get("calib_sha256") == bcfg["expert_quant"].get("calib_sha256"), "trunk calibration is the experts' calibration")

    index = json.load(open(os.path.join(new, "model.safetensors.index.json")))["weight_map"]
    files = sorted(f for f in os.listdir(new) if f.endswith(".safetensors"))
    need(set(index.values()) == set(files), "the index names every shard and only them")

    seen = {}
    banks = {}
    kept_affine = 0
    for f in files:
        h, meta = tensors(os.path.join(new, f))
        for name, t in h.items():
            need(name not in seen, f"{name} is in one shard only") if name in seen else None
            seen[name] = f
            need(index[name] == f, f"index maps {name} to {f}") if index.get(name) != f else None
            if name.endswith(".trellis") and len(t["shape"]) == 3:
                base_name = name[: -len(".trellis")]
                banks[base_name] = (f, t, meta)
    need(len(banks) > 0, f"{len(banks)} banks")
    rates = {t["shape"][2] for b, (_, t, _) in banks.items() if not b.endswith("lm_head")}
    need(len(rates) == 1 and rates <= {64, 96}, f"the trunk banks share one rate (n {sorted(rates)})")
    if trunk_n is not None:
        need(rates == {trunk_n}, f"the trunk banks are n={trunk_n}")

    bases = {}
    for b, (f, t, meta) in banks.items():
        hh, _ = tensors(os.path.join(new, f))
        su, sv = hh[b + ".suh"], hh[b + ".svh"]
        a, o, n = t["shape"]
        ok = (t["dtype"] == "U16" and su["dtype"] == "F16" and sv["dtype"] == "F16" and su["shape"] == [a * 16] and sv["shape"] == [o * 16])
        need(ok, f"{b}: trellis {t['shape']} with suh/svh") if not ok else None
        want_n = 96 if b.endswith("lm_head") else next(iter(rates))
        need(n == want_n, f"{b}: rate n={n}") if n != want_n else None
        need(meta.get("format") == "stq" and meta.get("codebook") == "spec1", f"{f} stamps stq spec1") if meta.get("format") != "stq" else None
        for leaf in ("weight", "scales", "biases"):
            need((b + "." + leaf) not in seen, f"{b}.{leaf} is gone") if (b + "." + leaf) in seen else None
        bases[b] = (a * 16, o * 16, n)
    print("ok: all banks well formed")

    # what the base held affine and the new pack no longer does is exactly the banks
    bseen = {}
    for f in sorted(os.listdir(base)):
        if f.endswith(".safetensors"):
            for name in tensors(os.path.join(base, f))[0]:
                bseen[name] = f
    lost = {n for n in bseen if n not in seen}
    gained = {n for n in seen if n not in bseen}
    need(all(n.rsplit(".", 1)[0] in bases for n in lost), "every tensor that left the pack belongs to a bank")
    need(all(n.rsplit(".", 1)[0] in bases for n in gained), "every tensor that joined the pack belongs to a bank")
    need(all(seen[n] == bseen[n] for n in seen if n in bseen and not n.startswith("language_model.model.layers.") or n in bseen and ".switch_mlp." in n), "non-trunk tensors stay in their shards")

    # the pack's kept affine linears are the base's bytes
    changed = 0
    for f in files:
        if f.startswith("trunk-stq-"):
            continue
        same = os.path.exists(os.path.join(base, f)) and filecmp.cmp(os.path.join(base, f), os.path.join(new, f), shallow=False)
        if not same:
            need(f.startswith("trunk-"), f"{f} differs from the base but is a trunk shard")
            changed += 1
    need(changed > 0, f"{changed} trunk shards were rewritten")
    for f in files:
        if f.startswith("layer-") or f.startswith("mtp-"):
            need(filecmp.cmp(os.path.join(base, f), os.path.join(new, f), shallow=False), f"{f} is the base's bytes")
    need(filecmp.cmp(os.path.join(base, "ngram_table.bin"), os.path.join(new, "ngram_table.bin"), shallow=False), "ngram_table.bin is the base's bytes")

    size = sum(os.path.getsize(os.path.join(new, f)) for f in files)
    bsize = sum(os.path.getsize(os.path.join(base, f)) for f in os.listdir(base) if f.endswith(".safetensors"))
    trunk = sum(os.path.getsize(os.path.join(new, f)) for f in files if f.startswith("trunk-"))
    btrunk = sum(os.path.getsize(os.path.join(base, f)) for f in os.listdir(base) if f.startswith("trunk-") and f.endswith(".safetensors"))
    print(f"safetensors: base {bsize / 1e9:.3f} GB -> {size / 1e9:.3f} GB; trunk-* shards {btrunk / 1e9:.3f} GB -> {trunk / 1e9:.3f} GB; {len(banks)} banks")
    print("VERIFIED")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
