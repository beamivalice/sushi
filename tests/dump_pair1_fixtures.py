"""Write src/exl3/fixtures/exl3_pair1_tiles.safetensors: random packed PAIR1 tiles at every even rate from
K1.5 to K4, each decoded row-major by sashimi's numpy reference (`decode_packed_tile`). CPU only.

    python tests/dump_pair1_fixtures.py --sashimi <sashimi checkout>
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sashimi', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('src/exl3/fixtures/exl3_pair1_tiles.safetensors'))
    ap.add_argument('--tiles', type=int, default=3)
    args = ap.parse_args()
    sys.path.insert(0, str(args.sashimi.resolve()))
    from sashimi.ref.codebook import CodebookMode
    from sashimi.ref.decode import decode_packed_tile

    rng = np.random.default_rng(0xCA27)
    tensors = {}
    for n in range(24, 65, 2):
        packed = rng.integers(0, 1 << 16, (args.tiles, n)).astype(np.uint16)
        decoded = np.stack([decode_packed_tile(t, n / 16, CodebookMode.PAIR1) for t in packed])
        tensors[f'trellis_n{n}'] = packed
        tensors[f'decoded_n{n}'] = np.ascontiguousarray(decoded.reshape(args.tiles, 256)).view(np.uint16)
    sha = subprocess.run(['git', '-C', str(args.sashimi), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    header, blobs, off = {'__metadata__': {'sashimi': sha}}, [], 0
    for name, arr in tensors.items():
        raw = arr.tobytes()
        header[name] = {'dtype': 'U16', 'shape': list(arr.shape), 'data_offsets': [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    head = json.dumps(header, separators=(',', ':')).encode()
    head += b' ' * (-len(head) % 8)
    args.output.write_bytes(len(head).to_bytes(8, 'little') + head + b''.join(blobs))


if __name__ == '__main__':
    main()
