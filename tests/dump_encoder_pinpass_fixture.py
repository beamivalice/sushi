#!/usr/bin/env python3
# sushi-test: tier=tool
"""Write src/encoder/fixtures/exl3_encode_pinpass.safetensors: one tile set searched by sashimi's Metal search with a shortened
pinning pass (`pin_pass` A < 128), the codewords of every tile in kernel order, for MCG and the STD1 and SPEC1 tables at K1.5, K2
and K3 (window 15) and for the other search plans at one A. Needs the Metal GPU.

    PYTHONPATH=<sashimi checkout> python tests/dump_encoder_pinpass_fixture.py --sashimi <sashimi checkout>
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

CONFIGS = [(cb, n, 15, a) for cb in ('mcg', 'std1', 'spec1') for n in (24, 32, 48) for a in (32, 64)] + [
    ('mcg', 32, 15, 16), ('mcg', 32, 12, 64), ('mcg', 40, 16, 64)]


def tiles() -> np.ndarray:
    """Typical, huge, all-zero, tied and spiked tiles: where half sums tie or clamp, the choice must still match."""
    rng = np.random.default_rng(0xA11CE)
    return np.concatenate([
        rng.normal(0, 1.2, (8, 256)),
        rng.normal(0, 300, (1, 256)),
        np.zeros((1, 256)),
        rng.choice([-1e4, -1, 0, 1, 1e4], (1, 256)),
        np.concatenate([80 * np.sign(rng.normal(size=(1, 16))), rng.normal(0, 1.2, (1, 240))], axis=1),
    ]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sashimi', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('src/encoder/fixtures/exl3_encode_pinpass.safetensors'))
    args = ap.parse_args()
    sys.path.insert(0, str(args.sashimi.resolve()))
    from sashimi.convert.metal_search import quantize_tiles_mlx_np
    from sashimi.ref.codebook import CodebookMode, TableCodebook

    books = {'mcg': CodebookMode.MCG, 'std1': TableCodebook('std1'), 'spec1': TableCodebook('spec1-w15')}
    x = tiles()
    out = {'x': x}
    for cb, n, window, a in CONFIGS:
        _, idx = quantize_tiles_mlx_np(x, n / 16, books[cb], window=window, pin_pass=a)
        out[f'{cb}_n{n}_w{window}_a{a}'] = idx.astype(np.uint16)
    sha = subprocess.run(['git', '-C', str(args.sashimi), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    header, blobs, off = {'__metadata__': {'sashimi': sha}}, [], 0
    for name, arr in out.items():
        raw = np.ascontiguousarray(arr).tobytes()
        header[name] = {'dtype': 'F32' if arr.dtype == np.float32 else 'U16', 'shape': list(arr.shape), 'data_offsets': [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    head = json.dumps(header, separators=(',', ':')).encode()
    head += b' ' * (-len(head) % 8)
    args.output.write_bytes(len(head).to_bytes(8, 'little') + head + b''.join(blobs))


if __name__ == '__main__':
    main()
