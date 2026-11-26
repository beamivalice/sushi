#!/usr/bin/env bash
# sushi-test: tier=hermetic
# Hermetic tests for release.sh — the SemVer parse, the "only dispatch when
# build.zig.zon and the CHANGELOG agree on an unreleased version" gate, and the
# local gate record (tests/gate.sh) it requires for the commit it cuts. No
# network and no real release: release.sh is SOURCED (its main() doesn't
# auto-run), and the tag lookup, origin's main head and `gh` are stubs.
#
# Run: bash tests/test_release.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

PASS=0; FAIL=0
ok() { # name  actual  expected
  if [ "$2" = "$3" ]; then PASS=$((PASS + 1)); echo "  PASS $1"
  else FAIL=$((FAIL + 1)); echo "  FAIL $1 — expected [$3], got [$2]"; fi
}

# shellcheck source=/dev/null
source "$ROOT/release.sh"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
DISPATCH_LOG="$TMP/dispatched.log"

# Stub the externals so the gate can be exercised offline:
#  - tag_exists → true only for the tags listed in EXISTING_TAGS
#  - remote_head_sha → REMOTE_SHA, the commit origin's main points at
#  - gh → record the dispatch instead of calling GitHub
EXISTING_TAGS=""
tag_exists() { case " $EXISTING_TAGS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
SHA_A=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
SHA_B=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
REMOTE_SHA=$SHA_A
remote_head_sha() { echo "$REMOTE_SHA"; }
gh() { echo "DISPATCHED $*" >> "$DISPATCH_LOG"; }

# record <file sha> <sha inside>: a gate record as tests/gate.sh writes it
export SUSHI_GATE_DIR="$TMP/gate"
record() {
  mkdir -p "$SUSHI_GATE_DIR"
  printf '{"sha": "%s", "summary": "%s green: 939/1001 tests passed"}\n' "$2" "${2:0:12}" > "$SUSHI_GATE_DIR/$1.json"
}
reset_records() { rm -rf "$SUSHI_GATE_DIR"; }

has() { # name  haystack  needle
  if printf '%s' "$2" | grep -qF -- "$3"; then PASS=$((PASS + 1)); echo "  PASS $1"
  else FAIL=$((FAIL + 1)); echo "  FAIL $1 — [$3] missing from: $2"; fi
}
dispatched() { [ -f "$DISPATCH_LOG" ] && echo yes || echo no; }
reset_dispatch() { rm -f "$DISPATCH_LOG"; }

printf '.{\n    .name = .sushi,\n    .version = "1.0.0",\n}\n' > "$TMP/build.zig.zon"
export ZON="$TMP/build.zig.zon"

echo "── parsing ──"
ok "reads build.zig.zon's version"          "$(zon_version "$ZON")" "1.0.0"
printf '# Changelog\n\n## v1.0.0 — Headline\n\n- a bullet\n\n## v0.9.0 — Older\n' > "$TMP/match.md"
ok "parses the top entry's version"         "$(changelog_top_version "$TMP/match.md")" "1.0.0"
printf '# Changelog\n\n## Unreleased\n\n- a bullet\n\n## v0.9.0 — Older\n'      > "$TMP/unreleased.md"
ok "empty when the top entry is Unreleased" "$(changelog_top_version "$TMP/unreleased.md")" ""
is_semver 1.0.0;   ok "1.0.0 is SemVer"            "$?" "0"
is_semver 26.9;    ok "26.9 is not SemVer"         "$?" "1"
is_semver 01.0.0;  ok "a leading zero is not SemVer" "$?" "1"

echo "── dispatch gate ──"
printf '## v26.9.1 — CalVer\n' > "$TMP/calver.md"
record "$SHA_A" "$SHA_A"

reset_dispatch
( CHANGELOG="$TMP/match.md"; main -y ) >/dev/null 2>&1; rc=$?
ok "matching versions dispatch"     "$(dispatched)" "yes"
ok "matching versions exit 0"       "$rc"           "0"

reset_dispatch
( CHANGELOG="$TMP/calver.md"; main -y ) >/dev/null 2>&1; rc=$?
ok "a 26.9.x heading doesn't dispatch" "$(dispatched)" "no"
ok "a 26.9.x heading exit 1"           "$rc"           "1"

reset_dispatch
( CHANGELOG="$TMP/unreleased.md"; main -y ) >/dev/null 2>&1; rc=$?
ok "an Unreleased top entry doesn't dispatch" "$(dispatched)" "no"
ok "an Unreleased top entry exit 1"           "$rc"           "1"

reset_dispatch
( CHANGELOG="$TMP/match.md"; EXISTING_TAGS="v1.0.0"; main -y ) >/dev/null 2>&1; rc=$?
ok "an existing tag doesn't dispatch" "$(dispatched)" "no"
ok "an existing tag exit 1"           "$rc"           "1"

reset_dispatch
( CHANGELOG="$TMP/match.md"; main --dry-run ) >/dev/null 2>&1; rc=$?
ok "dry-run never dispatches"       "$(dispatched)" "no"
ok "dry-run exit 0"                 "$rc"           "0"

echo "── the local gate record ──"
reset_dispatch; reset_records
out="$( ( CHANGELOG="$TMP/match.md"; main -y ) 2>&1 )"; rc=$?
ok "no gate record doesn't dispatch"      "$(dispatched)" "no"
ok "no gate record exit 1"                "$rc"           "1"
has "  naming tests/gate.sh"              "$out"          "tests/gate.sh $SHA_A"
has "  and the sha it would cut"          "$out"          "no gate record for $SHA_A"

reset_dispatch; reset_records; record "$SHA_B" "$SHA_B"
out="$( ( CHANGELOG="$TMP/match.md"; main -y ) 2>&1 )"; rc=$?
ok "a record for another sha doesn't dispatch" "$(dispatched)" "no"
ok "a record for another sha exit 1"           "$rc"           "1"
has "  naming the sha it is for"               "$out"          "the newest gate record is for $SHA_B"
has "  and tests/gate.sh"                      "$out"          "tests/gate.sh $SHA_A"

reset_dispatch; reset_records; record "$SHA_A" "$SHA_B"
out="$( ( CHANGELOG="$TMP/match.md"; main -y ) 2>&1 )"; rc=$?
ok "a record whose content names another sha doesn't dispatch" "$(dispatched)" "no"
has "  naming both"                                            "$out"          "is for $SHA_B, not $SHA_A"

reset_dispatch; reset_records; record "$SHA_A" "$SHA_A"
out="$( ( CHANGELOG="$TMP/match.md"; main -y ) 2>&1 )"; rc=$?
ok "a record for the cut sha dispatches"  "$(dispatched)" "yes"
ok "  exit 0"                             "$rc"           "0"
has "  with the gate sha as the workflow input" "$(cat "$DISPATCH_LOG" 2> /dev/null)" "-f gate_sha=$SHA_A"
has "  printing the record's summary"     "$out"          "${SHA_A:0:12} green: 939/1001 tests passed"

reset_dispatch
out="$( ( CHANGELOG="$TMP/match.md"; main --dry-run ) 2>&1 )"; rc=$?
ok "dry-run with a record never dispatches" "$(dispatched)" "no"
ok "  exit 0"                               "$rc"           "0"
has "  reporting the record"                "$out"          "${SHA_A:0:12} green: 939/1001 tests passed"
has "  and the dispatch it would make"      "$out"          "gate_sha=$SHA_A"

reset_dispatch; reset_records
out="$( ( CHANGELOG="$TMP/match.md"; main --dry-run ) 2>&1 )"; rc=$?
ok "dry-run without a record exit 1"      "$rc"           "1"
has "  reporting it found none"           "$out"          "no gate record for $SHA_A"

echo "── no terminal and no --yes ──"
reset_dispatch; reset_records; record "$SHA_A" "$SHA_A"
out="$( ( CHANGELOG="$TMP/match.md"; main ) 2>&1 < /dev/null )"; rc=$?
ok "no terminal without --yes doesn't dispatch" "$(dispatched)" "no"
ok "  exit 1"                                   "$rc"           "1"
has "  naming --yes"                            "$out"          "needs --yes"

echo ""
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then echo "PASS $TOTAL/$TOTAL"; exit 0
else echo "FAIL $FAIL/$TOTAL"; exit 1; fi
