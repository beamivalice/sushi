#!/usr/bin/env bash
# release.sh — trigger the GitHub "Release" workflow for the version in
# build.zig.zon, but ONLY if the top CHANGELOG.md entry documents that same
# version, no GitHub release or tag already carries it, and the local gate
# (tests/gate.sh) passed on the exact commit origin's main points at.
#
# Versions are SemVer MAJOR.MINOR.PATCH (tag v1.0.0). build.zig.zon's
# `.version` is the one source: `zig build` stamps it into `sushi --version`,
# and the workflow's "Extract version" step sources this file and applies the
# same checks, so a dispatch can only cut the version both files name. The
# dispatch carries the gated sha (`gate_sha`); the workflow refuses any other.
#
# Usage:
#   ./release.sh            # verify, confirm, then dispatch
#   ./release.sh -y         # skip the confirmation prompt
#   ./release.sh --dry-run  # print what it found and would do, never dispatch
#
# Env overrides: CHANGELOG, ZON, WORKFLOW (default release.yml), REF (default main), GH_REPO (default: origin),
# SUSHI_GATE_DIR (default ~/.sushi/runs/gate: tests/gate.sh's records).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHANGELOG="${CHANGELOG:-$REPO_ROOT/CHANGELOG.md}"
ZON="${ZON:-$REPO_ROOT/build.zig.zon}"
WORKFLOW="${WORKFLOW:-release.yml}"
REF="${REF:-main}"
# gh must act on this repo's origin, never an `upstream` remote it would otherwise pick (a fork has both).
GH_REPO="${GH_REPO:-$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null | sed -E 's#^(https?://|git@)##; s#:#/#; s#\.git$##')}"
export GH_REPO

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

is_semver() {
  [[ "$1" =~ ^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$ ]]
}

# The `.version = "X"` field of a build.zig.zon.
zon_version() {
  awk '
    /^[[:space:]]*\.version[[:space:]]*=/ {
      s = $0
      sub(/^[^"]*"/, "", s)
      sub(/".*$/, "", s)
      print s
      exit
    }
  ' "$1"
}

# The version the FIRST "## " heading of CHANGELOG.md names, sans leading 'v'
# ("## v1.0.0 — Headline" → "1.0.0"). Empty when that heading names none, e.g.
# "## Unreleased". Single awk pass (no pipe, so a `set -o pipefail` caller
# can't trip on SIGPIPE).
changelog_top_version() {
  awk '
    /^##[[:space:]]/ {
      if (match($0, /^##[[:space:]]*v[0-9][0-9A-Za-z.+-]*/)) {
        s = substr($0, RSTART, RLENGTH)
        sub(/^##[[:space:]]*v/, "", s)
        print s
      }
      exit
    }
  ' "$1"
}

# Prints the release version when build.zig.zon and the top CHANGELOG entry
# agree on one MAJOR.MINOR.PATCH; otherwise says why on stderr and fails.
release_version() {
  local changelog="$1" zon="$2" v cl
  v="$(zon_version "$zon")"
  if ! is_semver "$v"; then
    echo "release.sh: $zon version '$v' is not MAJOR.MINOR.PATCH" >&2
    return 1
  fi
  cl="$(changelog_top_version "$changelog")"
  if [ -z "$cl" ]; then
    echo "release.sh: the top entry of $changelog is not a '## v$v' heading" >&2
    return 1
  fi
  if ! is_semver "$cl"; then
    echo "release.sh: CHANGELOG heading v$cl is not MAJOR.MINOR.PATCH" >&2
    return 1
  fi
  if [ "$cl" != "$v" ]; then
    echo "release.sh: CHANGELOG top entry v$cl != build.zig.zon version v$v" >&2
    return 1
  fi
  echo "$v"
}

tag_exists() {
  gh release view "$1" >/dev/null 2>&1 \
    || git -C "$REPO_ROOT" ls-remote --exit-code --tags origin "refs/tags/$1" >/dev/null 2>&1
}

# The commit origin's branch <ref> points at: the one a dispatch on it builds.
remote_head_sha() {
  git -C "$REPO_ROOT" ls-remote --exit-code origin "refs/heads/$1" | cut -f1
}

# Prints the summary of tests/gate.sh's record for <sha>; with none for it, says why on stderr and fails.
gate_record_summary() {
  local sha="$1" dir="${SUSHI_GATE_DIR:-$HOME/.sushi/runs/gate}" rec f newest="" inside
  rec="$dir/$sha.json"
  if [ ! -f "$rec" ]; then
    echo "release.sh: no gate record for $sha (origin's $REF): run tests/gate.sh $sha after a git fetch" >&2
    for f in "$dir"/*.json; do
      [ -f "$f" ] && { [ -z "$newest" ] || [ "$f" -nt "$newest" ]; } && newest="$f"
    done
    [ -z "$newest" ] || echo "release.sh: the newest gate record is for $(plutil -extract sha raw -o - "$newest" 2> /dev/null)" >&2
    return 1
  fi
  inside="$(plutil -extract sha raw -o - "$rec" 2> /dev/null)" || inside=""
  if [ "$inside" != "$sha" ]; then
    echo "release.sh: the gate record $rec is for ${inside:-no sha}, not $sha: run tests/gate.sh $sha" >&2
    return 1
  fi
  plutil -extract summary raw -o - "$rec"
}

main() {
  set -euo pipefail

  local dry_run=0 assume_yes=0
  for arg in "$@"; do
    case "$arg" in
      --dry-run)   dry_run=1 ;;
      -y|--yes)    assume_yes=1 ;;
      -h|--help)   usage; return 0 ;;
      *) echo "release.sh: unknown argument '$arg'" >&2; usage >&2; return 2 ;;
    esac
  done

  local version
  version="$(release_version "$CHANGELOG" "$ZON")" || return 1
  echo "Release version     : v$version (build.zig.zon = CHANGELOG top entry)"

  if tag_exists "v$version"; then
    echo "release.sh: v$version already exists as a GitHub release or tag — bump build.zig.zon and the CHANGELOG heading" >&2
    return 1
  fi

  local sha summary
  sha="$(remote_head_sha "$REF")" || { echo "release.sh: cannot read the head of origin's $REF" >&2; return 1; }
  summary="$(gate_record_summary "$sha")" || return 1
  echo "Commit              : $sha (origin's $REF)"
  echo "Local gate          : $summary"

  if [ "$dry_run" -eq 1 ]; then
    echo "[dry-run] would run: gh workflow run $WORKFLOW --ref $REF -f gate_sha=$sha"
    return 0
  fi

  if [ "$assume_yes" -ne 1 ]; then
    read -r -p "Trigger the Release workflow for v$version on '$REF' ($sha)? [y/N] " reply
    case "$reply" in
      [yY] | [yY][eE][sS]) ;;
      *) echo "aborted."; return 0 ;;
    esac
  fi

  gh workflow run "$WORKFLOW" --ref "$REF" -f gate_sha="$sha"
  echo "✓ dispatched v$version on '$REF' ($sha)."
  echo "  Watch:  gh run watch \$(gh run list --workflow=$WORKFLOW --limit 1 --json databaseId --jq '.[0].databaseId')"
}

# Only run main when executed directly — sourcing (tests, the release
# workflow) just loads the functions.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
