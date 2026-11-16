#!/usr/bin/env python3
"""A served Qwen3.8-Flash-Next pack cut to its first N decoder layers, the test fixture that reaches every kernel a
served load runs (each kernel is gated on the served widths, which no toy built at small widths has).

    python3 -I tests/qwen4/cut_qwen4_pack.py --pack <served STQ pack> --out <dir> [--layers 4] [--ngram-base 4096]

Every tensor of a kept layer, the MTP head, the embedding, the head and the vision tower are kept byte for byte (a
shard holding only kept tensors is an APFS clone). The n-gram table is replaced by a small random one at the served
row width over `--ngram-base` (the served table is 32 GB): the PLE's per-head vocabulary sizes and offsets are
written as the engine's `NgramHash` derives them. `--layers` keeps a full-attention layer only when it covers the
config's `full_attention_interval`.
"""
import argparse
import array
import json
import os
import re
import shutil
import struct
import subprocess

LAYER = re.compile(r"\.layers\.(\d+)\.")


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def write_shard(path, tensors, metadata):
    """`tensors` is [(name, dtype, shape, bytes)] in file order."""
    header = {"__metadata__": metadata} if metadata is not None else {}
    off = 0
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, _, data in tensors:
            f.write(data)


def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def ngram_sizes(base, heads, divisor):
    """`NgramHash.init` for the one PLE: head i takes the (i+1)-th prime from `base`, rows round up to `divisor`."""
    sizes, p = [], base
    while len(sizes) < heads:
        if is_prime(p):
            sizes.append(p)
        p += 1
    offsets = [sum(sizes[:i]) for i in range(heads)]
    total = sum(sizes)
    return sizes, offsets, (total + divisor - 1) // divisor * divisor


def random_ngram_table(path, rows, dim, bits, group):
    meta = {"format": "mlx-serve-ngram", "bits": str(bits), "group_size": str(group)}
    if bits == 16:
        # BF16 rows of magnitude 2^-8 to 2^-7, either sign.
        raw = array.array("H", os.urandom(rows * dim * 2))
        for i in range(len(raw)):
            raw[i] = (raw[i] & 0x807F) | 0x3C00
        return write_shard(path, [("weight", "BF16", [rows, dim], raw.tobytes())], meta)
    cols = dim * bits // 32
    groups = dim // group
    w = os.urandom(rows * cols * 4)
    scale = struct.pack("<H", 0x3C23) * (rows * groups)  # bf16 0.00995
    bias = struct.pack("<H", 0xBD23) * (rows * groups)  # bf16 -0.0398
    tensors = [("weight", "U32", [rows, cols], w), ("scales", "BF16", [rows, groups], scale), ("biases", "BF16", [rows, groups], bias)]
    write_shard(path, tensors, meta)


def kept(name, layers):
    m = LAYER.search(name)
    return m is None or name.startswith("language_model.mtp.") or int(m.group(1)) < layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ngram-base", type=int, default=4096)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    os.makedirs(a.out)

    cfg = json.load(open(os.path.join(a.pack, "config.json")))
    tc = cfg["text_config"]
    if a.layers % tc["full_attention_interval"] != 0:
        raise SystemExit(f"--layers {a.layers} keeps no whole full-attention period of {tc['full_attention_interval']}")
    heads = (tc["ngram_size"] - 1) * tc["heads_per_ngram"]
    sizes, offsets, rows = ngram_sizes(a.ngram_base, heads, tc["make_ngram_vocab_size_divisible_by"])

    index = json.load(open(os.path.join(a.pack, "model.safetensors.index.json")))
    by_shard = {}
    for name, shard in index["weight_map"].items():
        by_shard.setdefault(shard, []).append(name)
    new_map = {}
    for shard, names in sorted(by_shard.items()):
        keep = [n for n in names if kept(n, a.layers)]
        if not keep:
            continue
        for n in keep:
            new_map[n] = shard
        header, base = read_header(os.path.join(a.pack, shard))
        rewrite = {n for n in keep if n.endswith("ngram_heads_offsets") or n.endswith("ngram_heads_vocab_sizes")}
        whole = [n for n in header if n != "__metadata__"]
        if set(keep) == set(whole) and not rewrite:
            subprocess.run(["cp", "-c", os.path.join(a.pack, shard), os.path.join(a.out, shard)], check=True)
            continue
        out = []
        with open(os.path.join(a.pack, shard), "rb") as f:
            for n in sorted(keep, key=lambda n: header[n]["data_offsets"][0]):
                e = header[n]
                if n.endswith("ngram_heads_offsets") or n.endswith("ngram_heads_vocab_sizes"):
                    vals = offsets if n.endswith("offsets") else sizes
                    assert e["dtype"] == "I64" and e["shape"] == [heads], (n, e)
                    out.append((n, "I64", e["shape"], struct.pack(f"<{heads}q", *vals)))
                    continue
                f.seek(base + e["data_offsets"][0])
                out.append((n, e["dtype"], e["shape"], f.read(e["data_offsets"][1] - e["data_offsets"][0])))
        write_shard(os.path.join(a.out, shard), out, header.get("__metadata__"))
    index["weight_map"] = new_map
    json.dump(index, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)

    tc["num_hidden_layers"] = a.layers
    tc["layer_types"] = tc["layer_types"][: a.layers]
    tc["ple_layer_ids"] = [i for i in tc["ple_layer_ids"] if i < a.layers]
    tc["ngram_vocab_size_base"] = a.ngram_base
    for key in ("quantization", "quantization_config"):
        cfg[key] = {k: v for k, v in cfg[key].items() if kept(k + ".", a.layers)}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)

    nt = cfg["ngram_table"]
    random_ngram_table(os.path.join(a.out, nt["file"]), rows, tc["ple_embed_dim"] // heads, nt["bits"], nt["group_size"])
    for name in os.listdir(a.pack):
        src = os.path.join(a.pack, name)
        if name.endswith(".safetensors") or name in ("config.json", "model.safetensors.index.json", nt["file"], "README.md"):
            continue
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(a.out, name))
    print(f"{a.out}: {a.layers} layers, n-gram table {rows} rows ({len(new_map)} tensors)")


if __name__ == "__main__":
    main()
