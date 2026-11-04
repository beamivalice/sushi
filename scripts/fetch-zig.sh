#!/usr/bin/env bash
# Fetch the pinned Zig release, check its sha256, and stage it at .zig-toolchain/
# (a stable path, whatever the tarball's own top-level dir is called). This is the
# single source of truth for the Zig version: CI and local builds refetch on a bump.
set -euo pipefail

ZIG_VERSION="0.17.1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/.zig-toolchain"
STAMP="$DEST/.version"

# Idempotent: skip when the staged copy already matches the pinned version.
if [ -f "$STAMP" ] && [ -x "$DEST/zig" ]; then
  if [ "$(cat "$STAMP")" = "$ZIG_VERSION" ]; then
    echo "[fetch-zig] .zig-toolchain already at $ZIG_VERSION — nothing to do"
    exit 0
  fi
  echo "[fetch-zig] staged version '$(cat "$STAMP")' != '$ZIG_VERSION' — refetching"
  # The build cache holds configure-time paths from the old toolchain.
  rm -rf "$REPO_ROOT/.zig-cache"
fi

case "$(uname -m)" in
  arm64) ARCH="aarch64" ;;
  x86_64) ARCH="x86_64" ;;
  *) echo "[fetch-zig] ERROR: unsupported arch $(uname -m)" >&2; exit 1 ;;
esac
case "$(uname -s)" in
  Darwin) OS="macos" ;;
  Linux) OS="linux" ;;
  *) echo "[fetch-zig] ERROR: unsupported OS $(uname -s)" >&2; exit 1 ;;
esac

ASSET="zig-${ARCH}-${OS}-${ZIG_VERSION}.tar.xz"
URL="https://ziglang.org/download/${ZIG_VERSION}/${ASSET}"
# sha256 of each release tarball (minisign-verified); update them with ZIG_VERSION.
case "$ARCH-$OS" in
  aarch64-macos) SHA256="75f1bc7a1e9852f17ec9a0c1895f767c4ec56d46bc8a834920eece1d0359dd8d" ;;
  x86_64-macos) SHA256="9af07b20319fbd0c92ee46ad23e81d6a782b1aaea2cfb8dcbff35f1d2c7ddbb7" ;;
  aarch64-linux) SHA256="a1546d5646eb184437918d6635ca6b762a0e889eac09cef720ee7a1ed3fe7795" ;;
  x86_64-linux) SHA256="3581863e3cbcfd0fce0efc682ce7e9e93da96668a977791d356a26d2a7e9b17a" ;;
esac

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[fetch-zig] downloading $URL"
curl -fSL --retry 3 -o "$TMP/zig.tar.xz" "$URL"
GOT="$(shasum -a 256 "$TMP/zig.tar.xz" | cut -d' ' -f1)"
if [ "$GOT" != "$SHA256" ]; then
  echo "[fetch-zig] ERROR: $ASSET sha256 $GOT, expected $SHA256" >&2
  exit 1
fi

echo "[fetch-zig] extracting"
tar xf "$TMP/zig.tar.xz" -C "$TMP"

EXTRACTED="$TMP/zig-${ARCH}-${OS}-${ZIG_VERSION}"
if [ ! -x "$EXTRACTED/zig" ]; then
  echo "[fetch-zig] ERROR: no zig executable in $ASSET" >&2
  exit 1
fi

# A worktree may symlink .zig-toolchain to the main checkout's copy: replace
# the link with its own copy, never write through it.
if [ -L "$DEST" ]; then rm "$DEST"; else rm -rf "$DEST"; fi
mkdir -p "$DEST"
cp -R "$EXTRACTED"/. "$DEST"/

echo "$ZIG_VERSION" > "$STAMP"

echo "[fetch-zig] staged Zig ($ZIG_VERSION):"
echo "  $DEST/zig ($("$DEST/zig" version))"
echo ""
echo "  Add it to PATH for this shell:"
echo "    export PATH=\"$DEST:\$PATH\""
