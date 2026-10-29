# sushi-test: tier=lib
# Source at the top of a script that boots a server: the prefix cache's SSD tier is on by default, so a
# boot without its own cache root would read and write (and sweep) the real ~/.sushi/kv-cache.
# A root the caller already set is left alone; the one made here is wiped only because this script made it.
if [ -z "${SUSHI_PREFIX_CACHE_DIR:-}" ]; then
    SUSHI_PREFIX_CACHE_DIR="$HOME/.sushi/runs/test-kv-cache/$(basename "$0" .sh)"
    rm -rf "$SUSHI_PREFIX_CACHE_DIR"
    mkdir -p "$SUSHI_PREFIX_CACHE_DIR"
    export SUSHI_PREFIX_CACHE_DIR
fi
