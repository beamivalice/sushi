#!/bin/bash
# sushi-test: tier=host
# `sushi repack --pack-revision` on a tiny pack generated in a temp dir (no weights, no GPU): stamp every shard, stamp a
# selection and then the rest, a symlinked shard is refused unless listed, a pattern that matches nothing is refused
# before a write, a different id is refused, `--check` follows the loader's rule, and a rerun changes nothing.
# Usage: ./tests/test_repack_stamp.sh
set -u
. "$(dirname "$0")/private_cache.sh"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BINARY:-$ROOT/zig-out/bin/sushi}"
PASS=0
FAIL=0

check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi
}
yes_if() { if eval "$1"; then echo 1; else echo 0; fi; }

[ -x "$BIN" ] || { echo "SKIP: no binary at $BIN (zig build -Doptimize=ReleaseFast)"; exit 3; }
command -v python3 >/dev/null || { echo "SKIP: python3 not found"; exit 3; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

make_pack() {
    python3 -I - "$1" <<'PY'
import json, os, struct, sys
d = sys.argv[1]
os.makedirs(d)
open(d + "/config.json", "w").write('{\n  "model_type": "tiny",\n  "rms_norm_eps": 1e-20\n}\n')
def shard(name, meta, seed, align):
    head = {}
    if meta is not None:
        head["__metadata__"] = meta
    head["a.weight"] = {"dtype": "U8", "shape": [3000], "data_offsets": [0, 3000]}
    head["b.weight"] = {"dtype": "F16", "shape": [5], "data_offsets": [3000, 3010]}
    raw = json.dumps(head, separators=(",", ":")).encode()
    while (8 + len(raw)) % align:
        raw += b" "
    data = bytes((i * 31 + seed) & 255 for i in range(3010))
    open(d + "/" + name, "wb").write(struct.pack("<Q", len(raw)) + raw + data)
shard("model-00001-of-00003.safetensors", None, 1, 8)
shard("model-00002-of-00003.safetensors", {"tile_order": "pair_major", "k": "4"}, 2, 16384)
shard("model-00003-of-00003.safetensors", {"k": "4"}, 3, 8)
PY
}

# The stamp of a shard (empty when none) and a hash of its data section.
stamp_of() {
    python3 -I - "$1" <<'PY'
import json, struct, sys
b = open(sys.argv[1], "rb").read()
n = struct.unpack("<Q", b[:8])[0]
print(json.loads(b[8:8 + n]).get("__metadata__", {}).get("pack_revision", ""))
PY
}
data_hash() {
    python3 -I - "$1" <<'PY'
import hashlib, struct, sys
b = open(sys.argv[1], "rb").read()
n = struct.unpack("<Q", b[:8])[0]
print(hashlib.sha256(b[8 + n:]).hexdigest())
PY
}
# config.json: pack_revision is the last member, with this id and unstamped list; keys before it keep their order.
config_is() {
    python3 -I - "$@" <<'PY'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
ok = list(c)[:2] == ["model_type", "rms_norm_eps"] and list(c)[-1] == "pack_revision"
ok = ok and c["pack_revision"] == {"id": sys.argv[2], "unstamped": sys.argv[3:]}
sys.exit(0 if ok else 1)
PY
}
tree_hash() { (cd "$1" && find . -type f | sort | xargs shasum | shasum | cut -c1-40); }

S1=model-00001-of-00003.safetensors
S2=model-00002-of-00003.safetensors
S3=model-00003-of-00003.safetensors

P="$WORK/all"
make_pack "$P"
H1="$(data_hash "$P/$S1")"; H2="$(data_hash "$P/$S2")"; H3="$(data_hash "$P/$S3")"

echo "== stamp every shard =="
out="$("$BIN" repack "$P" --pack-revision 2.0 2>&1)"; rc=$?
check "exit 0" "$(yes_if "[ $rc -eq 0 ]")"
check "summary names the count" "$(yes_if "echo '$out' | grep -q '3 shards stamped'")"
for s in $S1 $S2 $S3; do check "$s carries 2.0" "$(yes_if "[ \"\$(stamp_of $P/$s)\" = 2.0 ]")"; done
check "data sections identical" "$(yes_if "[ \"\$(data_hash $P/$S1)\" = $H1 ] && [ \"\$(data_hash $P/$S2)\" = $H2 ] && [ \"\$(data_hash $P/$S3)\" = $H3 ]")"
check "config gains the member, other keys keep their order" "$(yes_if "config_is $P 2.0")"
check "page-aligned shard keeps a page-aligned data start" "$(yes_if "[ \$(python3 -I -c \"import struct;print((8+struct.unpack('<Q',open('$P/$S2','rb').read(8))[0])%16384)\") -eq 0 ]")"
check "no temp or lock file left" "$(yes_if "[ -z \"\$(ls -A $P | grep -E 'stamp-tmp|lock')\" ]")"
"$BIN" repack "$P" --check >/dev/null 2>&1; check "--check passes" "$(yes_if "[ $? -eq 0 ]")"
"$BIN" repack "$P" --check >"$WORK/out" 2>&1; check "--check never sets up MLX" "$(yes_if "! grep -q '\[mem\]' $WORK/out")"
"$BIN" --version >"$WORK/out" 2>&1; check "--version never sets up MLX" "$(yes_if "[ -s $WORK/out ] && ! grep -q '\[mem\]' $WORK/out")"
before="$(tree_hash "$P")"
"$BIN" repack "$P" --pack-revision 2.0 >/dev/null 2>&1; rc=$?
check "a rerun exits 0 and changes nothing" "$(yes_if "[ $rc -eq 0 ] && [ \"\$(tree_hash $P)\" = \"$before\" ]")"
"$BIN" repack "$P" --pack-revision 3.0 >/dev/null 2>"$WORK/err"; rc=$?
check "a different id is refused, naming the old one" "$(yes_if "[ $rc -ne 0 ] && grep -q '2.0' $WORK/err && [ \"\$(tree_hash $P)\" = \"$before\" ]")"

echo "== a selection, then the rest =="
P="$WORK/sel"
make_pack "$P"
"$BIN" repack "$P" --pack-revision 2.0 --shards "$S1" >/dev/null 2>&1; rc=$?
check "selection exit 0" "$(yes_if "[ $rc -eq 0 ]")"
check "selected shard stamped, others not" "$(yes_if "[ \"\$(stamp_of $P/$S1)\" = 2.0 ] && [ -z \"\$(stamp_of $P/$S2)\" ] && [ -z \"\$(stamp_of $P/$S3)\" ]")"
"$BIN" repack "$P" --check >/dev/null 2>"$WORK/err"; rc=$?
check "--check fails and names the unstamped shard" "$(yes_if "[ $rc -ne 0 ] && grep -q '$S2' $WORK/err")"
"$BIN" repack "$P" --pack-revision 2.0 --shards '*00002*' --shards "$S3" >/dev/null 2>&1; rc=$?
check "the rest stamped" "$(yes_if "[ $rc -eq 0 ] && [ \"\$(stamp_of $P/$S2)\" = 2.0 ] && [ \"\$(stamp_of $P/$S3)\" = 2.0 ]")"
"$BIN" repack "$P" --check --pack-revision 2.0 >/dev/null 2>&1; check "--check passes on the full pack" "$(yes_if "[ $? -eq 0 ]")"

echo "== a pattern that matches nothing =="
P="$WORK/nomatch"
make_pack "$P"
before="$(tree_hash "$P")"
"$BIN" repack "$P" --pack-revision 2.0 --shards "$S1" --shards 'nope-*' >/dev/null 2>"$WORK/err"; rc=$?
check "refused naming the pattern, nothing written" "$(yes_if "[ $rc -ne 0 ] && grep -q 'nope-\\*' $WORK/err && [ \"\$(tree_hash $P)\" = \"$before\" ]")"

echo "== a symlinked shard =="
P="$WORK/link"
make_pack "$P"
mv "$P/$S3" "$WORK/shared.safetensors"
ln -s "$WORK/shared.safetensors" "$P/$S3"
shared="$(shasum "$WORK/shared.safetensors")"
before="$(tree_hash "$P")"
"$BIN" repack "$P" --pack-revision 2.0 >/dev/null 2>"$WORK/err"; rc=$?
check "refused naming the link, nothing written" "$(yes_if "[ $rc -ne 0 ] && grep -q '$S3' $WORK/err && [ \"\$(tree_hash $P)\" = \"$before\" ]")"
"$BIN" repack "$P" --pack-revision 2.0 --unstamped "$S3" >/dev/null 2>&1; rc=$?
check "listed, the others are stamped" "$(yes_if "[ $rc -eq 0 ] && [ \"\$(stamp_of $P/$S1)\" = 2.0 ] && [ \"\$(stamp_of $P/$S2)\" = 2.0 ]")"
check "the link and its target are untouched" "$(yes_if "[ -L $P/$S3 ] && [ \"\$(shasum $WORK/shared.safetensors)\" = \"$shared\" ]")"
check "config lists it as unstamped" "$(yes_if "config_is $P 2.0 $S3")"
"$BIN" repack "$P" --check >/dev/null 2>&1; check "--check passes with the link listed" "$(yes_if "[ $? -eq 0 ]")"

echo "== bad arguments =="
for bad in "--bogus" "--shards $S1" "--unstamped $S1" "--pack-revision" "--pack-revision bad/id"; do
    # shellcheck disable=SC2086
    "$BIN" repack "$WORK/all" $bad >/dev/null 2>"$WORK/err"; rc=$?
    check "'$bad' exits non-zero" "$(yes_if "[ $rc -ne 0 ] && [ -s $WORK/err ]")"
done

echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
