#!/usr/bin/env python3
"""DeepSeek-V4.1's Engram token map, built from the tokenizer exactly as DeepSeek's inference/engram.py
(`build_compressed_token_map`) builds it, packed into src/fixtures/dsv41_engram_token_map.bin.

Format: one bit per token id (LSB first) set where the token opens a new compressed id (ids are handed out in
token order), then every other token's compressed id as a little-endian u32.

  python -I tests/dump_dsv41_token_map.py <dir with tokenizer.json> [--check]

`--check` compares the committed map instead of writing it.
"""
import os
import struct
import sys

from tokenizers import Regex, Tokenizer, normalizers


def build_compressed_token_map(tok):
    # a private-use char, so a token that is exactly one space survives Strip() instead of collapsing to ""
    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    key_to_new = {}
    lookup = []
    for token_id in range(tok.get_vocab_size(with_added_tokens=True)):
        text = tok.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        lookup.append(key_to_new.setdefault(key, len(key_to_new)))
    return lookup, len(key_to_new)


def pack(lookup):
    bits = bytearray((len(lookup) + 7) // 8)
    repeats = []
    nxt = 0
    for t, v in enumerate(lookup):
        if v == nxt:
            bits[t // 8] |= 1 << (t % 8)
            nxt += 1
        else:
            repeats.append(v)
    return bytes(bits) + struct.pack(f"<{len(repeats)}I", *repeats)


lookup, n = build_compressed_token_map(Tokenizer.from_file(os.path.join(sys.argv[1], "tokenizer.json")))
print(f"{len(lookup)} tokens -> {n} compressed ids")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "fixtures", "dsv41_engram_token_map.bin")
if "--check" in sys.argv:
    same = open(out, "rb").read() == pack(lookup)
    print("committed map " + ("matches" if same else "DIFFERS"))
    sys.exit(0 if same else 1)
open(out, "wb").write(pack(lookup))
print(f"wrote {out} ({os.path.getsize(out)} bytes)")
