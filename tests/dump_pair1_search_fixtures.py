#!/usr/bin/env python3
# sushi-test: tier=tool
"""Write src/encoder/fixtures/pair1_search_pin{64,32}.bin: sashimi's PAIR1 pair search (pin 64 and 32 pairs) of 4 N(0,1)
tiles per even n from 24 to 64, the first tile x8 (past the half-cost clamp). Each file holds, per rate in order, tiles f32
[4,256], pair codewords u16 [4,128] and decoded f32 [4,256]. Needs the Metal GPU.

    <sashimi>/.venv/bin/python tests/dump_pair1_search_fixtures.py src/encoder/fixtures
"""
import sys, os
import numpy as np
from sashimi.tinyquant.pair_search import pair_search_mlx
from sashimi.ref.codebook import pair1_constants
out = sys.argv[1]
T = 4
for pin in (64, 32):
    with open(os.path.join(out, f"pair1_search_pin{pin}.bin"), "wb") as f:
        for n in range(24, 65, 2):
            k = n / 16
            rng = np.random.default_rng(7000 + 100 * pin + n)
            tiles = rng.standard_normal((T, 256)).astype(np.float32)
            tiles[0] *= 8.0  # past the half-cost clamp
            m, r, a, b = pair1_constants(k)
            q, idx = pair_search_mlx(tiles, k, mult=m, rot=r, mask0=a, mask1=b, pin_pass=pin)
            q2, idx2 = pair_search_mlx(tiles, k, mult=m, rot=r, mask0=a, mask1=b, pin_pass=pin)
            assert np.array_equal(np.array(idx), np.array(idx2))
            f.write(tiles.tobytes()); f.write(np.array(idx).astype(np.uint16).tobytes()); f.write(np.array(q, dtype=np.float32).tobytes())
    print("pin", pin, "done")
