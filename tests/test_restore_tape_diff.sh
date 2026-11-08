#!/bin/bash
# sushi-test: tier=hermetic
# test_restore_tape_diff.sh — tests/diff_restore_tape.py on synthetic tapes: equal tapes exit 0, a changed digest exits 1
# naming the row and the field, a missing row and --ignore are told apart. No server, no GPU; silent on pass.
#
# Usage: ./tests/test_restore_tape_diff.sh

set -u
cd "$(dirname "$0")/.." || exit 1
DIFF="python3 -I tests/diff_restore_tape.py"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

cat > "$TMP/a.tape" <<'EOF'
{"row":"lookup","seq":1,"request":1,"matched":0,"tier":"none","kv":[],"ssm":[],"arch":null}
{"row":"gen","request":1,"count":3,"sha":"aa","finish":"stop"}
{"row":"lookup","seq":2,"request":2,"matched":4096,"tier":"ssd","kv":["k0","k1"],"ssm":["s0","s1"],"arch":null}
{"row":"tier","request":2,"total_bytes":10,"entries":[{"id":1,"bytes":10,"ssm_positions":[4096]}],"lru":[1]}
EOF
cp "$TMP/a.tape" "$TMP/same.tape"

$DIFF "$TMP/a.tape" "$TMP/same.tape" > "$TMP/out" || fail "equal tapes must exit 0"
[ ! -s "$TMP/out" ] || fail "equal tapes must print nothing"

sed 's/"s1"/"sX"/' "$TMP/a.tape" > "$TMP/ssm.tape"
$DIFF "$TMP/a.tape" "$TMP/ssm.tape" > "$TMP/out" && fail "a changed digest must exit 1"
grep -q 'row 2' "$TMP/out" || fail "must name the row: $(cat "$TMP/out")"
grep -q 'field ssm\[1\]' "$TMP/out" || fail "must name the field: $(cat "$TMP/out")"

sed 's/"ssm_positions":\[4096\]/"ssm_positions":[4096,8192]/' "$TMP/a.tape" > "$TMP/tier.tape"
$DIFF "$TMP/a.tape" "$TMP/tier.tape" > "$TMP/out" && fail "a changed tier row must exit 1"
grep -q 'field entries\[0\].ssm_positions.length' "$TMP/out" || fail "must name the tier field: $(cat "$TMP/out")"

head -3 "$TMP/a.tape" > "$TMP/short.tape"
$DIFF "$TMP/a.tape" "$TMP/short.tape" > "$TMP/out" && fail "a missing row must exit 1"
grep -q 'row 3: only one tape has it' "$TMP/out" || fail "must name the missing row: $(cat "$TMP/out")"

sed 's/"total_bytes":10/"total_bytes":11/' "$TMP/a.tape" > "$TMP/bytes.tape"
$DIFF "$TMP/a.tape" "$TMP/bytes.tape" > /dev/null && fail "a changed total must exit 1"
$DIFF "$TMP/a.tape" "$TMP/bytes.tape" --ignore total_bytes > "$TMP/out" || fail "--ignore must drop the field"
exit 0
