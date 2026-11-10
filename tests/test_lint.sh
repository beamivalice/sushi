#!/bin/bash
# sushi-test: tier=hermetic
# test_lint.sh — scripts/lint/, which `zig build lint` (and so `zig build test`) runs over src/: per rule a red probe
# in a temporary tree, the allowlist, a stale entry, and the real tree with the real allowlist. Compile only, no GPU;
# it obeys the QUIET marker like any build. A new rule adds its probe cases here.
#
# Usage: ./tests/test_lint.sh

set -u
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
ZIG="${ZIG:-$REPO/.zig-toolchain/zig}"
QUIET=$HOME/.sushi/runs/QUIET
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAIL=0
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; sed 's/^/    /' "$TMP/out" | head -12; }

while [ -e "$QUIET" ]; do sleep 5; done
taskpolicy -a "$ZIG" build-exe scripts/lint/main.zig "-femit-bin=$TMP/lint" >"$TMP/out" 2>&1 || { fail "the lint builds"; echo "FAIL"; exit 1; }

# A tree is src/ plus an allowlist; the lint runs from its root so reports name src/<file>.
tree() { rm -rf "$TMP/t"; mkdir -p "$TMP/t/src"; : > "$TMP/t/allow"; }
lint() { (cd "$TMP/t" && "$TMP/lint" src allow) >"$TMP/out" 2>&1; }
expect_red() { # <what> <text the report holds>
  if lint; then fail "$1 should fail"; elif ! grep -q -- "$2" "$TMP/out"; then fail "$1 should report '$2'"; fi
}
expect_green() { lint || fail "$1 should pass"; [ ! -s "$TMP/out" ] || fail "$1 should print nothing"; }

# rule const-write
tree
printf '%s\n' 'const S = struct { x: u32 };' 'fn f(self: *const S) void {' '    @constCast(self).x = 1;' '}' > "$TMP/t/src/p.zig"
expect_red "a *const param cast" "src/p.zig:3:5 \[const-write\] f: @constCast of const param 'self'"

tree
printf '%s\n' 'fn g(buf: []const u8) void {' '    @constCast(buf.ptr)[0] = 1;' '}' > "$TMP/t/src/p.zig"
expect_red "a []const param cast" "\[const-write\] g: @constCast of const param 'buf'"

tree
printf '%s\n' 'const Job = struct { n: u32 };' 'fn h() void {' '    const job = Job{ .n = 0 };' '    @constCast(&job).n = 1;' '}' > "$TMP/t/src/p.zig"
expect_red "a const local cast" "\[const-write\] h: @constCast of const local 'job'"

tree
printf '%s\n' 'fn o(p: ?*const u8) void {' '    @constCast(p.?).* = 1;' '}' > "$TMP/t/src/p.zig"
expect_red "a ?*const param cast" "const param 'p'"

tree
printf '%s\n' 'const S = struct { x: u32 };' 'fn m(self: *S, other: []const u8) void {' '    var s: S = .{ .x = 0 };' \
  '    @constCast(self).x = 1;' '    @constCast(&s).x = 1;' '    { const job = S{ .x = 0 }; _ = job; }' \
  '    var job: S = .{ .x = 0 };' '    @constCast(&job).x = 2;' '    _ = other;' '}' > "$TMP/t/src/p.zig"
expect_green "a cast of a mutable param, a var local and a name reused after its const scope"

# rule unused-import
tree
printf '%s\n' 'const std = @import("std");' 'const dead = @import("dead.zig");' 'pub fn f() void { _ = std; }' > "$TMP/t/src/p.zig"
expect_red "an unused import" "src/p.zig:2:7 \[unused-import\] dead: unused import 'dead'"

tree
printf '%s\n' 'const std = @import("std");' 'const Foo = std.mem.Allocator;' 'const Bar = std;' 'pub fn f() void { _ = std; }' > "$TMP/t/src/p.zig"
expect_red "an unused alias" "\[unused-import\] Foo: unused alias 'Foo'"

tree
printf '%s\n' 'const std = @import("std");' 'const note = @import("note.zig");' 'pub fn f() void {' '    _ = std; // note' '    _ = "note";' \
  '    _ = std.note;' '}' > "$TMP/t/src/p.zig"
expect_red "a name only in a comment, a string and a field" "\[unused-import\] note:"

tree
printf '%s\n' 'const std = @import("std");' 'const used = @import("used.zig");' 'const Alias = std.mem.Allocator;' \
  'const only_test = @import("t.zig");' 'const @"odd name" = @import("odd.zig");' 'pub const exported = @import("e.zig");' \
  'pub const Exported = std.mem.Allocator;' 'comptime { _ = @"odd name"; }' 'const _ = @import("side.zig");' \
  'pub fn f(a: Alias) void { _ = used; _ = a; }' 'test "t" { _ = only_test; }' > "$TMP/t/src/p.zig"
expect_green "used, pub, test-only, quoted and discard imports"

# rule std-xxhash3
tree
printf '%s\n' 'const std = @import("std");' 'pub fn f(b: []const u8) u64 {' '    return std.hash.XxHash3.hash(0, b);' '}' > "$TMP/t/src/p.zig"
expect_red "the standard library's XxHash3" "src/p.zig:3:21 \[std-xxhash3\] -: std.hash.XxHash3"

tree
printf '%s\n' 'const std = @import("std");' 'const XxHash3 = struct {};' 'pub fn f(b: []const u8) u64 {' '    _ = XxHash3;' '    return std.hash.Wyhash.hash(0, b);' '}' > "$TMP/t/src/p.zig"
expect_green "another hash and a local type of the same name"

# rule undefined-nested-write
tree
printf '%s\n' 'const Cfg = struct { a: u32 = 0, b: u32 = 0 };' 'const T = struct { config: Cfg, n: u32 };' \
  'test "nested write" {' '    var t: T = undefined;' '    t.n = 1;' '    t.config.a = 2;' '    t.config.b = 3;' '}' > "$TMP/t/src/p.zig"
expect_red "a field written into the nested struct of an undefined local" "src/p.zig:6:5 \[undefined-nested-write\] nested_write: 't.config' is written field by field into an undefined 't'"

tree
printf '%s\n' 'const Cfg = struct { a: u32 = 0 };' 'const T = struct { config: Cfg, n: u32 };' 'fn init(t: *T) void { t.* = undefined; }' \
  'test "near misses" {' '    var whole: T = undefined;' '    whole.config = .{};' '    whole.config.a = 1;' \
  '    var all: T = undefined;' '    all = .{ .config = .{}, .n = 0 };' '    all.config.a = 1;' \
  '    var flat: T = undefined;' '    flat.n = 1;' '    var typed = T{ .config = .{}, .n = 0 };' '    typed.config.a = 1;' \
  '    var method: T = undefined;' '    method.reset();' '    method.config.a = 1;' \
  '    var in_fn: T = undefined;' '    _ = &in_fn;' '}' \
  'fn notATest() void {' '    var t: T = undefined;' '    t.config.a = 1;' '}' > "$TMP/t/src/p.zig"
expect_green "a whole assignment first, a flat write, an initialized local, a method call and a write outside a test"

# the allowlist
tree
printf '%s\n' 'const S = struct { x: u32 };' 'fn handleOf(s: *S) *S { return s; }' 'fn f(self: *const S) *S {' \
  '    return handleOf(@constCast(self));' '}' > "$TMP/t/src/p.zig"
echo 'const-write src/p.zig:f dispatch only, never written' > "$TMP/t/allow"
expect_green "an allowlisted dispatch"
echo 'const-write src/p.zig:other no such fn' >> "$TMP/t/allow"
expect_red "a stale allowlist entry" "stale entry 'const-write src/p.zig:other'"
echo 'const-write src/q.zig:f wrong file' > "$TMP/t/allow"
expect_red "an entry for another file" "\[const-write\] f:"
echo 'no-such-rule src/p.zig:f reason' > "$TMP/t/allow"
expect_red "an unknown rule" "unknown rule 'no-such-rule'"

# the real tree
"$TMP/lint" src tests/guards/lint.allow >"$TMP/out" 2>&1 || fail "the real src/ passes with tests/guards/lint.allow"

[ "$FAIL" -eq 0 ] || { echo "FAIL"; exit 1; }
echo "PASS"
