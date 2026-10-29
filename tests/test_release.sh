#!/usr/bin/env bash
# sushi-test: tier=hermetic
# Hermetic tests for release.sh — the SemVer parse and the "only dispatch when
# build.zig.zon and the CHANGELOG agree on an unreleased version" gate. No
# network and no real release: release.sh is SOURCED (its main() doesn't
# auto-run), and the tag lookup + `gh` dispatch are overridden with stubs.
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

# Stub the two externals so the gate can be exercised offline:
#  - tag_exists → true only for the tags listed in EXISTING_TAGS
#  - gh → record the dispatch instead of calling GitHub
EXISTING_TAGS=""
tag_exists() { case " $EXISTING_TAGS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
gh() { echo "DISPATCHED $*" >> "$DISPATCH_LOG"; }

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

echo ""
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then echo "PASS $TOTAL/$TOTAL"; exit 0
else echo "FAIL $FAIL/$TOTAL"; exit 1; fi
