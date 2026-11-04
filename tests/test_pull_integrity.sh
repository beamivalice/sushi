#!/bin/bash
# sushi-test: tier=host
. "$(dirname "$0")/private_cache.sh"
# `sushi pull` keeps a file on disk only when its CONTENT matches the listing, never by size alone: a pack revision
# re-uploaded at the same byte size must be fetched again. Hermetic: tests/hf_stub.py stands in for huggingface.co
# (SUSHI_HF_ENDPOINT, the test-only hook) and serves a repo folder re-read on every request.
#
# Usage: ./tests/test_pull_integrity.sh [stub-port]
set -u

PORT="${1:-18853}"
BIN="${SUSHI_BIN:-./zig-out/bin/sushi}"
if [ ! -x "$BIN" ]; then
    echo "SKIP: $BIN not found — build first: zig build -Doptimize=ReleaseFast"
    exit 0
fi
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"
HERE="$(cd "$(dirname "$0")" && pwd)"

SCRATCH=$(mktemp -d)
STUB=""
trap '[ -n "$STUB" ] && { kill "$STUB"; wait "$STUB"; } 2>/dev/null; rm -rf "$SCRATCH"' EXIT
PASS=0
FAIL=0
check() {
    if [ "$2" -eq 0 ]; then echo "PASS: $1"; PASS=$((PASS + 1)); else echo "FAIL: $1"; FAIL=$((FAIL + 1)); fi
}

HF="$SCRATCH/hf"
REPO="$HF/beamster/StubPack"
DEST="$SCRATCH/home/.sushi/models/beamster/StubPack"
mkdir -p "$REPO"
printf '{"model_type":"qwen4_exp"}\n' > "$REPO/config.json"
printf 'tokenizer-v1\n' > "$REPO/tokenizer.json"
head -c 4096 /dev/zero | tr '\0' 'a' > "$REPO/model-00001.safetensors"
head -c 4096 /dev/zero | tr '\0' 'b' > "$REPO/model-00002.safetensors"

python3 -I "$HERE/hf_stub.py" "$HF" "$PORT" > "$SCRATCH/stub.log" 2>&1 &
STUB=$!
for _ in $(seq 1 40); do
    curl -sf "http://127.0.0.1:$PORT/api/models/beamster/StubPack" > /dev/null && break
    sleep 0.25
done

OUT="$SCRATCH/pull.txt"
pull() {
    : > "$HF/.requests"
    HOME="$SCRATCH/home" SUSHI_HF_ENDPOINT="http://127.0.0.1:$PORT" "$BIN" pull beamster/StubPack > "$OUT" 2>&1
}
requested() { sort "$HF/.requests" | tr '\n' ' ' | sed 's/ $//'; }
has() { grep -q "$1" "$2"; echo $?; }

# ── 1. a fresh pull writes the manifest ──
pull
check "fresh pull succeeds" "$?"
check "fresh pull fetched every file" "$([ "$(requested)" = "config.json model-00001.safetensors model-00002.safetensors tokenizer.json" ]; echo $?)"
M="$DEST/.sushi-pull.json"
check "manifest written" "$([ -f "$M" ]; echo $?)"
check "manifest names the repo and commit" "$(grep -q '"repo":"beamster/StubPack"' "$M" && grep -q stubcommit0001 "$M"; echo $?)"
SHA1=$(shasum -a 256 "$REPO/model-00001.safetensors" | cut -d' ' -f1)
check "manifest holds the sha256 of an LFS file" "$(has "$SHA1" "$M")"
check "manifest holds the git blob id of a small file" "$(has '"git_oid"' "$M")"
check "pull marker removed" "$([ ! -e "$DEST/.pull-incomplete" ]; echo $?)"

# ── 2. unchanged files are skipped ──
pull
check "re-pull of an unchanged repo succeeds" "$?"
check "unchanged files are not requested" "$([ -z "$(requested)" ]; echo $?)"

# ── 3. the bug: same size, other content ──
head -c 4096 /dev/zero | tr '\0' 'c' > "$REPO/model-00001.safetensors"
pull
check "re-pull after a same-size upstream change succeeds" "$?"
check "only the changed file is fetched again" "$([ "$(requested)" = "model-00001.safetensors" ]; echo $?)"
check "the changed file now has the new bytes" "$(cmp -s "$REPO/model-00001.safetensors" "$DEST/model-00001.safetensors"; echo $?)"

# ── 4. a file the repo dropped goes, an unrelated file stays ──
rm "$REPO/tokenizer.json"
mkdir -p "$DEST/.dspark"
printf 'cache\n' > "$DEST/.dspark/state"
printf 'mine\n' > "$DEST/notes.txt"
pull
check "re-pull after a dropped file succeeds" "$?"
check "the dropped file is removed" "$([ ! -e "$DEST/tokenizer.json" ]; echo $?)"
check "an unrelated file survives" "$([ -f "$DEST/notes.txt" ] && [ -f "$DEST/.dspark/state" ]; echo $?)"
check "nothing was fetched" "$([ -z "$(requested)" ]; echo $?)"

# ── 5. a folder without a manifest is hashed once, only differing files download ──
rm "$M"
head -c 4096 /dev/zero | tr '\0' 'd' > "$REPO/model-00002.safetensors"
pull
check "pull of a folder without a manifest succeeds" "$?"
check "it says it is checking the files on disk" "$(has 'checking 3 files already on disk' "$OUT")"
check "only the differing file is fetched" "$([ "$(requested)" = "model-00002.safetensors" ]; echo $?)"
check "the manifest is back" "$([ -f "$M" ]; echo $?)"

# ── 6. a corrupt download fails by name and renames nothing ──
head -c 4096 /dev/zero | tr '\0' 'e' > "$REPO/model-00003.safetensors"
echo model-00003.safetensors > "$HF/.corrupt"
pull
RC=$?
check "a download that does not match its checksum fails" "$([ "$RC" -ne 0 ]; echo $?)"
check "the failure names the file" "$(has 'model-00003.safetensors does not match' "$OUT")"
check "no renamed file and no partial is left" "$([ ! -e "$DEST/model-00003.safetensors" ] && [ ! -e "$DEST/model-00003.safetensors.partial" ]; echo $?)"
check "the folder is still marked incomplete" "$([ -e "$DEST/.pull-incomplete" ]; echo $?)"
rm "$HF/.corrupt"
pull
check "the next pull completes" "$?"
check "the marker is gone and the file is there" "$([ ! -e "$DEST/.pull-incomplete" ] && cmp -s "$REPO/model-00003.safetensors" "$DEST/model-00003.safetensors"; echo $?)"

# ── 7. a partial is resumed only for the remote file it was started for ──
# The sidecar `<file>.partial.oid` holds "<content id> <size>" of that remote file.
P="$DEST/model-00004.safetensors.partial"
NEW4="$REPO/model-00004.safetensors"
head -c 4096 /dev/zero | tr '\0' 'f' > "$NEW4"
OID4=$(shasum -a 256 "$NEW4" | cut -d' ' -f1)
ranged() { [ -s "$HF/.ranges" ] && cut -d' ' -f1 "$HF/.ranges" | sort -u | tr '\n' ' ' | sed 's/ $//'; }
fetch4() { rm -f "$HF/.ranges"; pull; RC=$?; }

# 7a. an unkeyed partial (a pull before keys existed) restarts from zero even though its bytes are a valid prefix
head -c 2048 "$NEW4" > "$P"
rm -f "$P.oid"
fetch4
check "an unkeyed partial: the pull succeeds" "$RC"
check "an unkeyed partial is not resumed" "$([ -z "$(ranged)" ]; echo $?)"
check "an unkeyed partial: the file is right and no partial is left" "$(cmp -s "$NEW4" "$DEST/model-00004.safetensors" && [ ! -e "$P" ] && [ ! -e "$P.oid" ]; echo $?)"

# 7b. a partial keyed to another revision (same size, other bytes) is discarded
rm "$DEST/model-00004.safetensors"
head -c 2048 /dev/zero | tr '\0' 'x' > "$P"
printf '%s 4096\n' "$(printf old-revision | shasum -a 256 | cut -d' ' -f1)" > "$P.oid"
fetch4
check "a stale partial: the pull succeeds" "$RC"
check "a stale partial is not resumed" "$([ -z "$(ranged)" ]; echo $?)"
check "a stale partial: the file has the new bytes" "$(cmp -s "$NEW4" "$DEST/model-00004.safetensors"; echo $?)"

# 7c. a stale partial longer than the new file is no dead end either
rm "$DEST/model-00004.safetensors"
head -c 6000 /dev/zero | tr '\0' 'x' > "$P"
rm -f "$P.oid"
fetch4
check "an oversized stale partial: the pull succeeds" "$RC"
check "an oversized stale partial: the file has the new bytes" "$(cmp -s "$NEW4" "$DEST/model-00004.safetensors"; echo $?)"

# 7d. a partial keyed to this very file resumes with a ranged request
rm "$DEST/model-00004.safetensors"
head -c 2048 "$NEW4" > "$P"
printf '%s 4096\n' "$OID4" > "$P.oid"
fetch4
check "a matching partial: the pull succeeds" "$RC"
check "a matching partial is resumed with a ranged request" "$([ "$(ranged)" = "model-00004.safetensors" ] && grep -q '^model-00004.safetensors 2048$' "$HF/.ranges"; echo $?)"
check "a matching partial: the file is right and no partial is left" "$(cmp -s "$NEW4" "$DEST/model-00004.safetensors" && [ ! -e "$P" ] && [ ! -e "$P.oid" ]; echo $?)"

# 7e. a download that fails its checksum leaves neither partial nor key
rm "$DEST/model-00004.safetensors"
echo model-00004.safetensors > "$HF/.corrupt"
fetch4
check "a corrupt download fails" "$([ "$RC" -ne 0 ]; echo $?)"
check "a corrupt download leaves no partial and no key" "$([ ! -e "$P" ] && [ ! -e "$P.oid" ]; echo $?)"
rm "$HF/.corrupt"
fetch4
check "the next pull completes after a corrupt one" "$RC"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
