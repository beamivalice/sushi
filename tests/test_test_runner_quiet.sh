#!/bin/bash
# sushi-test: tier=host
# test_test_runner_quiet.sh — a passing test prints NOTHING, on either stream.
#
# CLAUDE.md already bans stdout from a test: under `zig build test` fd 1 is the
# build runner's protocol pipe and one stray byte hangs the runner forever.
# stderr is not free either. Zig 0.17 reports ANY stderr a test
# step produced through its failure renderer, so a suite that passed 69/69
# prints
#
#     test
#     +- run test w
#     <the test's own output>
#     failed command: .../test --cache-dir=./.zig-cache --seed=0x... --listen=-
#
# and no Build Summary, while exiting 0. That is indistinguishable from a real
# failure at a glance and has been read as one. So: benchmark and diagnostic
# lines in a test ride an env switch (SUSHI_EXL3_PARITY_TRACE for the EXL3
# ones); only a test that is about to FAIL may print.
#
# Usage: ./tests/test_test_runner_quiet.sh [test-filter]   (default: the whole suite)

set -u
cd "$(dirname "$0")/.." || exit 1

ZIG=${ZIG:-$PWD/.zig-toolchain/zig}
FILTER=${1:-}
STEP=${TEST_STEP:-test}

if [ ! -x "$ZIG" ]; then
  echo "SKIP: no $ZIG (run ./scripts/fetch-zig.sh)"
  exit 0
fi

# The run step is cached on success, so a second invocation would prove nothing.
# Drop the manifests that name a test binary; every other step stays cached.
# Zig 0.17 manifests are binary files with NUL-terminated paths and no extension.
LC_ALL=C grep -l -a -E 'o/[0-9a-f]+/(exl3-)?test' .zig-cache/h/* 2>/dev/null | xargs rm -f 2>/dev/null

if [ -n "$FILTER" ]; then
  WHAT="zig build $STEP -Dtest-filter=$FILTER"
  OUT=$("$ZIG" build "$STEP" -Doptimize=ReleaseFast -Dtest-filter="$FILTER" 2>&1)
else
  WHAT="zig build $STEP"
  OUT=$("$ZIG" build "$STEP" -Doptimize=ReleaseFast 2>&1)
fi
RC=$?

FAIL=0
if [ $RC -ne 0 ]; then
  echo "  FAIL: $WHAT exited $RC"
  FAIL=1
fi
if [ -n "$OUT" ]; then
  echo "  FAIL: the test step wrote $(printf '%s' "$OUT" | wc -l | tr -d ' ') lines; a passing suite is silent"
  printf '%s\n' "$OUT" | head -20 | sed 's/^/    | /'
  FAIL=1
fi

# -Dtest-log=<abs path>: each test binary's runner appends `<UTC>\t<step>\t<PASS|SKIP|FAIL>\t<ms>\t<name>` per test
# and one `DONE` line with its totals, and still prints nothing. Every binary shares the runner, so the host step's
# format corpus stands for all of them.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
LINE=$'^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{3}Z\t[^\t]+\t(PASS|SKIP|FAIL|DONE)\t[0-9]+\t.+$'
# counts <log> <step>: the step's PASS, SKIP and FAIL lines as its DONE line words them.
counts() {
  awk -F'\t' -v s="$2" '$2 == s && $3 == "PASS" {p++} $2 == s && $3 == "SKIP" {k++} $2 == s && $3 == "FAIL" {f++}
    END {printf "%d pass, %d skip, %d fail", p, k, f}' "$1"
}
LC_ALL=C grep -l -a -E 'o/[0-9a-f]+/(exl3-)?test' .zig-cache/h/* 2>/dev/null | xargs rm -f 2>/dev/null
LOG_OUT=$("$ZIG" build test-host -Doptimize=ReleaseFast -Dtest-filter="format corpus" -Dtest-log="$TMP/host.tsv" 2>&1)
LOG_RC=$?
if [ $LOG_RC -ne 0 ] || [ -n "$LOG_OUT" ]; then
  echo "  FAIL: the -Dtest-log run exited $LOG_RC with $(printf '%s' "$LOG_OUT" | wc -l | tr -d ' ') lines of output"
  printf '%s\n' "$LOG_OUT" | head -20 | sed 's/^/    | /'
  FAIL=1
elif [ ! -s "$TMP/host.tsv" ] || LC_ALL=C grep -v -E "$LINE" "$TMP/host.tsv" >/dev/null; then
  echo "  FAIL: -Dtest-log wrote no log, or a line that is not '<UTC>\\t<step>\\t<status>\\t<ms>\\t<name>'"
  head -5 "$TMP/host.tsv" 2>/dev/null | sed 's/^/    | /'
  FAIL=1
elif ! LC_ALL=C grep -E $'\ttest-host\tPASS\t[0-9]+\t.*format corpus' "$TMP/host.tsv" >/dev/null; then
  echo "  FAIL: no PASS line of the test-host step names a format corpus test"
  FAIL=1
elif [ "$(awk -F'\t' '$3 == "DONE"' "$TMP/host.tsv" | cut -f2,5)" != "test-host"$'\t'"$(counts "$TMP/host.tsv" test-host)" ]; then
  echo "  FAIL: the step's one DONE line does not carry its own totals ($(counts "$TMP/host.tsv" test-host))"
  awk -F'\t' '$3 == "DONE"' "$TMP/host.tsv" | sed 's/^/    | /'
  FAIL=1
fi

if "$ZIG" build test-host -Dtest-log=relative.tsv >"$TMP/out" 2>&1 || ! grep -q -- "-Dtest-log=relative.tsv is not an absolute path" "$TMP/out"; then
  echo "  FAIL: a relative -Dtest-log should be refused by name"
  FAIL=1
fi

# The runner on scratch binaries, over the build protocol and run by hand: a skip, a failure, a leak, a logged error
# and a crash each say what they were, the build runner's restart after the crash goes on logging, and each process
# that ran a test ends on DONE with its own totals.
mkdir -p "$TMP/scratch"
cp scripts/test_runner.zig "$TMP/scratch/"
cat >"$TMP/scratch/scratch.zig" <<'EOF'
const std = @import("std");
test "passes" {}
test "skips" {
    return error.SkipZigTest;
}
test "fails" {
    try std.testing.expect(false);
}
test "leaks" {
    _ = try std.testing.allocator.alloc(u8, 1);
}
test "logs an error" {
    std.log.err("logged", .{});
}
test "crashes" {
    @panic("boom");
}
test "runs after the crash" {}
EOF
cat >"$TMP/scratch/build.zig" <<'EOF'
const std = @import("std");
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const t = b.addTest(.{
        .root_module = b.createModule(.{ .root_source_file = b.path("scratch.zig"), .target = target }),
        .test_runner = .{ .path = b.path("test_runner.zig"), .mode = .server },
    });
    const run = b.addRunArtifact(t);
    run.setEnvironmentVariable("SUSHI_TEST_LOG", b.option([]const u8, "log", "").?);
    run.setEnvironmentVariable("SUSHI_TEST_STEP", "scratch step");
    b.default_step.dependOn(&run.step);
    b.installArtifact(b.addTest(.{
        .name = "by-hand",
        .root_module = b.createModule(.{ .root_source_file = b.path("by_hand.zig"), .target = target }),
        .test_runner = .{ .path = b.path("test_runner.zig"), .mode = .server },
    }));
}
EOF
cat >"$TMP/scratch/by_hand.zig" <<'EOF'
const std = @import("std");
test "passes" {}
test "skips" {
    return error.SkipZigTest;
}
test "fails" {
    try std.testing.expect(false);
}
EOF
(cd "$TMP/scratch" && "$ZIG" build "-Dlog=$TMP/scratch.tsv" >"$TMP/out" 2>&1)
got=$(cut -f2,3,5 "$TMP/scratch.tsv" 2>/dev/null | tr '\t' '|' | tr '\n' ';')
want="scratch step|PASS|scratch.test.passes;scratch step|SKIP|scratch.test.skips;scratch step|FAIL|scratch.test.fails;"
want+="scratch step|FAIL|scratch.test.leaks;scratch step|FAIL|scratch.test.logs an error;"
want+="scratch step|PASS|scratch.test.runs after the crash;scratch step|DONE|1 pass, 0 skip, 0 fail;"
if [ "$got" != "$want" ] || LC_ALL=C grep -v -E "$LINE" "$TMP/scratch.tsv" >/dev/null; then
  echo "  FAIL: the scratch run's log should read [$want], got [$got]"
  sed 's/^/    | /' "$TMP/out" | head -20
  FAIL=1
fi
SUSHI_TEST_LOG="$TMP/by-hand.tsv" SUSHI_TEST_STEP="by hand" "$TMP/scratch/zig-out/bin/by-hand" >/dev/null 2>&1
got=$(cut -f2,3,5 "$TMP/by-hand.tsv" 2>/dev/null | tr '\t' '|' | tr '\n' ';')
want="by hand|PASS|by_hand.test.passes;by hand|SKIP|by_hand.test.skips;by hand|FAIL|by_hand.test.fails;"
want+="by hand|DONE|1 pass, 1 skip, 1 fail;"
if [ "$got" != "$want" ] || LC_ALL=C grep -v -E "$LINE" "$TMP/by-hand.tsv" >/dev/null; then
  echo "  FAIL: a test binary run by hand should log [$want], got [$got]"
  FAIL=1
fi

if [ $FAIL -eq 0 ]; then
  echo "  PASS: $WHAT is silent and green, and -Dtest-log records each test, its status and its step silently"
  echo "PASS"
  exit 0
fi
echo "FAIL"
exit 1
