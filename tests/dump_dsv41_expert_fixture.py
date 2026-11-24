#!/usr/bin/env python3
# sushi-test: tier=tool
"""Oracle for the encoder's DeepSeek-V4.1 routed-expert reader (`src/encoder/source.zig`, `Arch.deepseek_v41`).

One tiny layer-3 expert 5 in the release's own layout: `I8` bytes holding two e2m1 nibbles (low first) per
weight row and one `F8_E8M0` scale per 32 columns, for w1 (gate) and w3 (up) of shape [32, 64] and w2 (down) of
shape [64, 32]. Beside each tensor sits its numpy dequantization in the reader's public [in, out] f32 layout.

  python -I tests/dump_dsv41_expert_fixture.py --out src/encoder/fixtures/dsv41_expert.safetensors
"""

import argparse
import json
import struct

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
DIM, INTER = 64, 32


def dequant(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    rows, half = weight.shape
    lo, hi = weight & 0x0F, weight >> 4
    nib = np.stack([lo, hi], axis=-1).reshape(rows, half * 2)
    mag = E2M1[nib & 7] * np.where(nib & 8, -1.0, 1.0).astype(np.float32)
    exp = np.exp2(scale.astype(np.float32) - 127.0).repeat(32, axis=1)
    return (mag * exp).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(41)
    tensors: list[tuple[str, str, tuple[int, ...], bytes]] = []
    for name, rows, cols in (("w1", INTER, DIM), ("w3", INTER, DIM), ("w2", DIM, INTER)):
        weight = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
        scale = rng.integers(120, 134, size=(rows, cols // 32), dtype=np.uint8)
        base = f"layers.3.ffn.experts.5.{name}"
        public = np.ascontiguousarray(dequant(weight, scale).T)
        tensors.append((f"{base}.weight", "I8", weight.shape, weight.tobytes()))
        tensors.append((f"{base}.scale", "F8_E8M0", scale.shape, scale.tobytes()))
        tensors.append((f"{base}.public", "F32", public.shape, public.tobytes()))
    header, offset = {}, 0
    for key, dtype, shape, raw in tensors:
        header[key] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for _, _, _, raw in tensors:
            f.write(raw)


if __name__ == "__main__":
    main()
