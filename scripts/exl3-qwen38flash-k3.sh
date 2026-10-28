#!/usr/bin/env bash
# Serve the resident Sushi-STQ-2.75bpw pack.
# Extra arguments are appended, e.g. ./exl3-qwen38flash-k3.sh --port 11235.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
binary="$root/zig-out/bin/sushi"
if [[ ! -x "$binary" ]]; then
  printf '%s\n' "Missing executable: $binary" \
    "Build $root first: zig build -Doptimize=ReleaseFast" >&2
  exit 1
fi

exec "$binary" serve \
  --model "${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw" \
  --host 127.0.0.1 \
  --port 12345 \
  --ctx-size 1048576 \
  --prefill-chunk 8192 \
  --max-concurrent 1 \
  --kv-quant 8 \
  --max-tokens 64000 \
  --mtp \
  --prefix-cache-mem 12GB \
  --prefix-cache-entries 1 \
  --prefix-cache-disk 100GB \
  --ssm-checkpoint-max 16 \
  --metrics \
  "$@"
