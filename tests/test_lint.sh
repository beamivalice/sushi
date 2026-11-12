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

# rule mlx-extern
tree
printf '%s\n' 'fn f() void {' '    const c = struct {' '        extern "c" fn mlx_matmul(a: u32) c_int;' '    };' '    _ = c;' '}' > "$TMP/t/src/p.zig"
expect_red "an extern fn of mlx-c beside its user" "src/p.zig:3:[0-9]* \[mlx-extern\] mlx_matmul: extern fn 'mlx_matmul': declare it in src/mlx.zig"

tree
printf '%s\n' 'extern "c" fn _mlx_array_is_available(res: *bool) c_int;' > "$TMP/t/src/p.zig"
expect_red "an extern fn of mlx-c with the private prefix" "\[mlx-extern\] _mlx_array_is_available:"

tree
printf '%s\n' 'extern "c" var mlx_flag: c_int;' > "$TMP/t/src/p.zig"
expect_red "an extern var of mlx-c" "\[mlx-extern\] mlx_flag: extern var"

tree
printf '%s\n' 'const F = fn () callconv(.c) c_int;' 'fn f() *const F {' '    return @extern(*const F, .{ .name = "mlx_eval" });' '}' > "$TMP/t/src/p.zig"
expect_red "an @extern of an mlx-c name" "src/p.zig:3:[0-9]* \[mlx-extern\] mlx_eval: @extern of 'mlx_eval'"

tree
printf '%s\n' 'extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;' 'fn mlx_helper() void {}' \
  'const Fn = fn () callconv(.c) c_int;' 'fn g() *const Fn { return @extern(*const Fn, .{ .name = "sysctlbyname" }); }' \
  'const label = "mlx_eval";' 'pub fn h() void { _ = label; mlx_helper(); }' > "$TMP/t/src/p.zig"
printf '%s\n' 'extern "c" fn mlx_array_new() u32;' 'extern "c" var mlx_flag: c_int;' > "$TMP/t/src/mlx.zig"
expect_green "another extern, a Zig fn named mlx_, a string and the seam itself"

tree
printf '%s\n' 'extern "c" fn mlx_array_new() u32;' > "$TMP/t/src/p.zig"
echo 'mlx-extern src/p.zig:mlx_array_new reaches a symbol mlx.zig cannot name' > "$TMP/t/allow"
expect_green "an allowlisted mlx-c extern"

# rule double-free
tree
printf '%s\n' 'fn free(x: u32) void { _ = x; }' 'fn work(x: u32) !u32 { return x; }' 'fn f(fold: bool) !u32 {' '    const y: u32 = 1;' \
  '    errdefer free(y);' '    if (fold) return y;' '    defer free(y);' '    return work(y);' '}' > "$TMP/t/src/p.zig"
expect_red "a handle freed by an errdefer and a later defer" "src/p.zig:7:5 \[double-free\] f: 'y' is freed by the errdefer on line 5 and by this defer"

tree
printf '%s\n' 'const m = struct { fn free(x: u32) u32 { return x; } };' 'test "two frees" {' '    const y: u32 = 1;' \
  '    errdefer _ = m.free(y);' '    {' '        defer _ = m.free(y);' '    }' '}' > "$TMP/t/src/p.zig"
expect_red "the same call through a namespace, discarded, in a nested block" "\[double-free\] two_frees: 'y' is freed"

tree
printf '%s\n' 'fn free(x: u32) void { _ = x; }' 'fn other(x: u32) void { _ = x; }' 'fn g() !void {' '    const a: u32 = 1;' '    const b: u32 = 2;' \
  '    errdefer free(a);' '    defer free(b);' '    defer other(a);' '    {' '        const c: u32 = 3;' '        errdefer free(c);' '    }' \
  '    const c: u32 = 4;' '    defer free(c);' '}' 'fn h() void {' '    const a: u32 = 1;' '    defer free(a);' '}' > "$TMP/t/src/p.zig"
expect_green "another name, another call, a defer after the errdefer's block closed, and another fn"

# rule scope-state
tree
printf '%s\n' 'const mlx = struct { const engine = struct { var active: bool = false; var enabled: bool = false; }; };' \
  'fn forward() void {' '    const was = mlx.engine.active;' '    mlx.engine.active = true;' '    mlx.engine.active = was;' '}' > "$TMP/t/src/p.zig"
expect_red "the engine's side written by hand around a forward" "src/p.zig:4:9 \[scope-state\] forward: 'engine.active' is written by hand"

tree
printf '%s\n' 'const mlx = struct { const engine = struct { var active: bool = false; var enabled: bool = false; }; };' \
  'fn load() void {' '    mlx.engine.enabled = true;' '}' 'test "leaves the side on" {' '    mlx.engine.active = false;' '}' > "$TMP/t/src/p.zig"
expect_red "the engine turned on by hand outside a test" "\[scope-state\] load: 'engine.enabled' is written by hand"
expect_red "the engine's side written by hand in a test" "\[scope-state\] leaves_the_side_on: 'engine.active' is written by hand"

tree
printf '%s\n' 'const mlx = struct { const engine = struct { var active: bool = false; var enabled: bool = false; }; };' \
  'fn reads() bool {' '    return mlx.engine.active == mlx.engine.enabled;' '}' 'test "turns the engine on for its span" {' \
  '    const was = mlx.engine.enabled;' '    mlx.engine.enabled = true;' '    defer mlx.engine.enabled = was;' '}' > "$TMP/t/src/p.zig"
mkdir -p "$TMP/t/src/tensor"
printf '%s\n' 'const engine = struct { var active: bool = false; };' 'pub fn makers(own: bool) void {' '    engine.active = own;' '}' > "$TMP/t/src/tensor/route.zig"
expect_green "a read, a test that turns the engine on, and the owner's own file"

# rule template-runtime-safety
tree
mkdir -p "$TMP/t/src/template"
printf '%s\n' 'fn lex(s: []const u8) usize {' '    return s.len;' '}' > "$TMP/t/src/template/p.zig"
expect_red "a template fn without runtime safety" "src/template/p.zig:1:4 \[template-runtime-safety\] lex: fn 'lex' does not start with @setRuntimeSafety(true)"

tree
mkdir -p "$TMP/t/src/template"
printf '%s\n' 'fn f(s: []const u8) usize {' '    const n = s.len;' '    @setRuntimeSafety(true);' '    return n;' '}' \
  'fn g() void {' '    @setRuntimeSafety(false);' '}' 'const S = struct {' '    fn h() void {' '        @setRuntimeSafety(true);' \
  '        const inner = struct {' '            fn k() void {}' '        };' '        _ = inner;' '    }' '};' > "$TMP/t/src/template/p.zig"
expect_red "safety set late" "\[template-runtime-safety\] f:"
expect_red "safety turned off" "\[template-runtime-safety\] g:"
expect_red "a fn nested in a struct inside a fn" "\[template-runtime-safety\] k:"

tree
mkdir -p "$TMP/t/src/template"
printf '%s\n' 'fn f(s: []const u8) usize {' '    @setRuntimeSafety(true);' '    {' '        @setRuntimeSafety(false);' '        return s.len;' '    }' '}' \
  'comptime {' '    @setRuntimeSafety(false);' '}' > "$TMP/t/src/template/p.zig"
expect_red "safety turned off in a nested block" "src/template/p.zig:4:9 \[template-runtime-safety\] f: @setRuntimeSafety(false) under src/template/"
expect_red "safety turned off outside any fn" "\[template-runtime-safety\] -: @setRuntimeSafety(false)"

tree
mkdir -p "$TMP/t/src/template"
printf '%s\n' 'fn f(s: []const u8) usize {' '    @setRuntimeSafety(true);' '    return s.len;' '}' \
  'extern "c" fn time(t: ?*c_long) c_long;' 'test "t" { _ = f("x"); }' > "$TMP/t/src/template/p.zig"
printf '%s\n' 'fn outside() void {}' > "$TMP/t/src/p.zig"
expect_green "a safe fn, an extern, a test and a fn outside src/template/"

# rule template-superlinear-std
tree
mkdir -p "$TMP/t/src/template"
printf '%s\n' 'const std = @import("std");' 'const mem = std.mem;' 'fn f(s: []const u8, xs: []u8) ?usize {' '    @setRuntimeSafety(true);' \
  '    std.sort.block(u8, xs, {}, std.sort.asc(u8));' '    _ = mem.count(u8, s, "ab");' '    return std.mem.lastIndexOf(u8, s, "ab");' '}' > "$TMP/t/src/template/p.zig"
expect_red "a std sort under src/template/" "src/template/p.zig:5:9 \[template-superlinear-std\] f: std.sort under src/template/"
expect_red "a substring count through an alias" "\[template-superlinear-std\] f: mem.count"
expect_red "a substring search" "\[template-superlinear-std\] f: mem.lastIndexOf"

tree
mkdir -p "$TMP/t/src/template"
printf '%s\n' 'const std = @import("std");' 'fn f(s: []const u8) ?usize {' '    @setRuntimeSafety(true);' '    return std.mem.indexOfScalar(u8, s, 0x2c);' '}' \
  'test "an oracle" { _ = std.mem.indexOf(u8, "ab", "b"); }' > "$TMP/t/src/template/p.zig"
printf '%s\n' 'const std = @import("std");' 'fn outside(s: []const u8) ?usize {' '    return std.mem.indexOf(u8, s, "ab");' '}' > "$TMP/t/src/p.zig"
expect_green "a scalar search, a test oracle and a search outside src/template/"

# rule arch-test-unlisted
tree
mkdir -p "$TMP/t/src/arch1"
printf '%s\n' 'pub const tests = .{' '    @import("listed.zig"),' '};' > "$TMP/t/src/arch1/arch_tests.zig"
printf '%s\n' 'pub fn f() void {}' 'test "listed" { f(); }' > "$TMP/t/src/arch1/listed.zig"
printf '%s\n' 'pub fn g() void {}' 'test "never run" { g(); }' > "$TMP/t/src/arch1/un.zig"
expect_red "a file with a test its arch_tests.zig does not list" "src/arch1/un.zig:2:1 \[arch-test-unlisted\] -: un.zig declares a test and src/arch1/arch_tests.zig does not list it"

tree
mkdir -p "$TMP/t/src/arch1" "$TMP/t/src/shared"
printf '%s\n' 'pub const tests = .{' '    @import("listed.zig"),' '};' '// @import("comment.zig")' > "$TMP/t/src/arch1/arch_tests.zig"
printf '%s\n' 'pub fn f() void {}' 'test "listed" { f(); }' > "$TMP/t/src/arch1/listed.zig"
printf '%s\n' 'pub fn h() void {}' > "$TMP/t/src/arch1/no_tests.zig"
printf '%s\n' 'pub fn k() void {}' 'test "a folder with no list" { k(); }' > "$TMP/t/src/shared/k.zig"
expect_green "a listed file, a file with no test and a folder with no arch_tests.zig"
printf '%s\n' 'pub fn c() void {}' 'test "named only in a comment" { c(); }' > "$TMP/t/src/arch1/comment.zig"
expect_red "a file the list names only in a comment" "\[arch-test-unlisted\] -: comment.zig"

# rule template-text-search
tree
printf '%s\n' 'const std = @import("std");' 'const mem = std.mem;' 'const C = struct { chat_template: []const u8 };' \
  'fn f(cc: *const C, tpl: []const u8) bool {' '    if (std.mem.indexOf(u8, cc.chat_template, "<think>") != null) return true;' \
  '    return mem.count(u8, tpl, "tools") > 0;' '}' > "$TMP/t/src/p.zig"
expect_red "a search of a config's template text" "src/p.zig:5:17 \[template-text-search\] f: mem.indexOf over the template text"
expect_red "a count of a template parameter through an alias" "\[template-text-search\] f: mem.count over the template text"

tree
mkdir -p "$TMP/t/src/template"
printf '%s\n' 'const std = @import("std");' 'fn f(rendered: []const u8, tpl: []const u8) bool {' \
  '    return std.mem.indexOf(u8, rendered, "<think>") != null and std.mem.eql(u8, tpl, "x");' '}' \
  'test "a test reads its template" { _ = std.mem.indexOf(u8, "{{ x }}", "x"); const tpl = "a"; _ = std.mem.count(u8, tpl, "a"); }' > "$TMP/t/src/p.zig"
printf '%s\n' 'const std = @import("std");' 'fn g(template: []const u8) ?usize {' '    @setRuntimeSafety(true);' \
  '    return std.mem.indexOfScalar(u8, template, 0x7b);' '}' > "$TMP/t/src/template/q.zig"
expect_green "a search of rendered text, an equality, a test, and the engine itself"

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
