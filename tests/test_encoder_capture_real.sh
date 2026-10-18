#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${SUSHI_MODELS_DIR:-$HOME/.sushi/models}
SOURCE=${ENCODER_REAL_SOURCE:-$MODELS/Qwen/Qwen3.8-Flash-Next}

if [[ ! -f "$SOURCE/config.json" ]]; then
    echo "SKIP encoder real-model capture: Qwen BF16 source absent"
    exit 0
fi

cd "$ROOT"
./.zig-toolchain/zig build test-build -Doptimize=ReleaseFast -Dtest-filter="capture (real Qwen"
SUSHI_ENCODER_REAL_MODEL=1 SUSHI_MODELS_DIR="$(dirname "$(dirname "$SOURCE")")" ./zig-out/tests/encoder-test
echo "PASS encoder capture: a real four-layer Qwen slice converts layer by layer and its hidden boundaries, MLP input rows, routing and Hessians equal the full forward's"
