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

# rule shape-offset-write
tree
printf '%s\n' 'const max_dims = 8;' 'fn f(dims: []const c_int) void {' '    var shape: [max_dims]c_int = undefined;' '    @memcpy(shape[1..][0..dims.len], dims);' '}' > "$TMP/t/src/p.zig"
expect_red "an offset write into a shape buffer" "src/p.zig:4:5 \[shape-offset-write\] f: @memcpy into shape buffer 'shape' at an offset"

tree
printf '%s\n' 'const core = struct { const max_dims = 8; };' 'fn g(dims: []const c_int, at: usize) void {' '    var s: [core.max_dims]c_int = undefined;' '    @memcpy(s[at..][0..dims.len], dims);' '}' > "$TMP/t/src/p.zig"
expect_red "an offset write into a core.max_dims buffer" "\[shape-offset-write\] g: @memcpy into shape buffer 's'"

tree
printf '%s\n' 'const max_dims = 8;' 'fn h(dims: []const c_int) void {' '    var shape: [max_dims]c_int = undefined;' '    @memcpy(shape[0..dims.len], dims);' '    _ = shape;' '}' > "$TMP/t/src/p.zig"
expect_green "a write from axis 0"

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

# rule reply-bypass
tree
printf '%s\n' 'const tc = @import("tool_calls.zig");' 'pub fn f(a: u8) void {' '    _ = tc.parseToolCalls(a, "x");' '}' > "$TMP/t/src/anthropic.zig"
expect_red "a surface calling the raw parser" "src/anthropic.zig:3:12 \[reply-bypass\] f: parseToolCalls: read calls"

tree
printf '%s\n' 'pub fn f() void {' '    _ = .{ .text = "", .prefill_tps = 0.0 };' '}' > "$TMP/t/src/responses_api.zig"
expect_red "a surface building a GenerationResult" "src/responses_api.zig:2:25 \[reply-bypass\] f: a GenerationResult literal"

tree
printf '%s\n' 'const tc = @import("tool_calls.zig");' 'pub fn f(a: u8) void {' '    _ = tc.parseToolCalls(a, "x");' '}' 'test "t" { _ = .{ .prefill_tps = 0.0 }; }' > "$TMP/t/src/other.zig"
printf '%s\n' 'test "t" { _ = .{ .prefill_tps = 0.0 }; }' > "$TMP/t/src/generation_stream.zig"
expect_green "the raw parser outside a surface, and a result literal in a test or in generation_stream.zig"

# rule request-bool-read
tree
printf '%s\n' 'fn f(root: Obj) bool {' '    return if (root.get("stream")) |v| v == .bool and v.bool else false;' '}' > "$TMP/t/src/server.zig"
expect_red "a request bool read that takes its default for another type" "src/server.zig:2:42 \[request-bool-read\] f: a bool compared with .bool"

tree
printf '%s\n' 'fn f(root: Obj) bool {' '    return if (root.get("stream")) |v| v == .bool and v.bool else false;' '}' > "$TMP/t/src/other.zig"
printf '%s\n' 'fn g(v: Value) bool {' '    return switch (v) { .bool => |b| b, else => false };' '}' 'test "t" { _ = v == .bool; }' > "$TMP/t/src/server.zig"
expect_green "a bool compared in a file that reads no request, a switch prong and a test"

# rule error-body-literal
tree
printf '%s\n' 'fn f(s: *S) !void {' '    try send(s, "{\"error\":\"metrics off\"}");' '}' > "$TMP/t/src/p.zig"
expect_red "a hand-built OpenAI error body" "src/p.zig:2:17 \[error-body-literal\] f: a hand-built error body"

tree
printf '%s\n' 'fn g(s: *S) !void {' '    try send(s,' '        \\{{"type":"error","error":{{"type":"x"}}}}' '    );' '}' > "$TMP/t/src/p.zig"
expect_red "a hand-built Anthropic error body in a multiline literal" "src/p.zig:3:9 \[error-body-literal\] g: a hand-built error body"

# Each spelling of the envelope's key: spaced, a format argument, split, "type": "error", a JSON writer's field,
# a multiline literal, a newline after the brace.
spellings=(
  'fn a1(m: []const u8) !void {|    _ = try std.fmt.allocPrint(alloc, "{{ \"error\": {{\"message\":\"{s}\"}}}}", .{m});|}|2:39'
  'fn a2(m: []const u8) !void {|    _ = try std.fmt.allocPrint(alloc, "{{\"{s}\":{{\"message\":\"{s}\"}}}}", .{ "error", m });|}|2:81'
  'fn a3(s: *S) !void {|    try send(s, "{" ++ "\"error\":{\"message\":\"x\"}}");|}|2:24'
  'fn a4(s: *S) !void {|    try send(s, "{\"type\": \"error\",\"error\":{\"type\":\"x\"}}");|}|2:17'
  'fn a5(w: *W, m: []const u8) !void {|    try w.write(.{ .@"error" = .{ .message = m } });|}|2:21'
  'fn a6(s: *S) !void {|    try send(s,|        \\{ "error": { "message": "x" } }|    );|}|3:9'
  'fn a7(s: *S) !void {|    try send(s, "{\n\"error\":{\"message\":\"x\"}}");|}|2:17'
)
for case in "${spellings[@]}"; do
  tree
  IFS='|' read -r -a lines <<< "$case"
  at="${lines[${#lines[@]}-1]}"
  unset 'lines[${#lines[@]}-1]'
  printf '%s\n' "${lines[@]}" > "$TMP/t/src/p.zig"
  expect_red "an error envelope spelled as in ${lines[0]%%(*}" "src/p.zig:$at \[error-body-literal\]"
done

tree
printf '%s\n' 'fn g(m: []const u8, n: u32) !void {' '    _ = try std.fmt.allocPrint(alloc, ",\"{s}\":{d}", .{ "error", n });' \
  '    _ = .{ .ok = true, .@"error" = m };' '    _ = parsed.value.@"error".message;' '    _ = "{\"errors\": [], \"type\": \"error_rate\"}";' \
  '    _ = "a" ++ ",\"error\":null";' '    _ = std.mem.eql(u8, kind, "error");' '}' > "$TMP/t/src/p.zig"
expect_green "an error field after another, a parsed error read, other keys and values, and a plain \"error\" string"

tree
printf '%s\n' 'fn f(s: *S) !void {' '    try send(s, "{\"error\":\"x\"}");' '}' > "$TMP/t/src/http.zig"
printf '%s\n' 'fn g() []const u8 {' '    return ",\"error\":null";' '}' 'test "t" { try expect(body, "{\"error\":{\"message\":\"x\"}}"); }' > "$TMP/t/src/p.zig"
printf '%s\n' 'fn h() []const u8 {' '    return "{\"error\":{\"message\":\"x\"}}";' '}' > "$TMP/t/src/p_test.zig"
expect_green "the sender's own file, an error field, a test's expected body and a test file"

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
printf '%s\n' 'fn free(x: u32) void { _ = x; }' 'fn join(p: []u32) !void { _ = p; }' 'fn v(parts: []u32) !void {' '    var built: usize = 0;' \
  '    errdefer for (parts[0..built]) |p| free(p);' '    built = parts.len;' '    defer for (parts) |p| free(p);' '    try join(parts);' '}' > "$TMP/t/src/p.zig"
expect_red "a slice freed by an errdefer loop and a later defer loop" "src/p.zig:7:5 \[double-free\] v: 'parts' is freed by the errdefer on line 5 and by this defer"

tree
printf '%s\n' 'const m = struct { fn free(x: u32) u32 { return x; } };' 'test "two frees" {' '    const y: u32 = 1;' \
  '    errdefer _ = m.free(y);' '    {' '        defer _ = m.free(y);' '    }' '}' > "$TMP/t/src/p.zig"
expect_red "the same call through a namespace, discarded, in a nested block" "\[double-free\] two_frees: 'y' is freed"

tree
printf '%s\n' 'fn free(x: u32) void { _ = x; }' 'fn other(x: u32) void { _ = x; }' 'fn g() !void {' '    const a: u32 = 1;' '    const b: u32 = 2;' \
  '    errdefer free(a);' '    defer free(b);' '    defer other(a);' '    {' '        const c: u32 = 3;' '        errdefer free(c);' '    }' \
  '    const c: u32 = 4;' '    defer free(c);' '}' 'fn h() void {' '    const a: u32 = 1;' '    defer free(a);' '}' > "$TMP/t/src/p.zig"
expect_green "another name, another call, a defer after the errdefer's block closed, and another fn"

# rule engine-hold-end
tree
mkdir -p "$TMP/t/src/arch" "$TMP/t/src/tensor"
printf '%s\n' 'const mlx = struct { const engine = struct { fn hold() void {} fn unhold() void {} }; };' \
  'fn load() void {' '    mlx.engine.hold();' '    errdefer mlx.engine.unhold();' '}' > "$TMP/t/src/p.zig"
expect_red "a hold ended by hand" "src/p.zig:4:18 \[engine-hold-end\] load: 'engine.unhold' ends a hold by hand"

tree
mkdir -p "$TMP/t/src/arch" "$TMP/t/src/tensor"
printf '%s\n' 'const mlx = struct { const engine = struct { fn unhold() void {} }; };' 'pub fn engineHoldEnd() void {' '    mlx.engine.unhold();' '}' > "$TMP/t/src/arch/handle.zig"
printf '%s\n' 'const engine = struct { pub fn unhold() void {} };' 'fn turnEnd() void {' '    engine.unhold();' '}' > "$TMP/t/src/tensor/route.zig"
printf '%s\n' 'const mlx = struct { const engine = struct { fn unhold() void {} }; };' 'pub fn engineHoldEnd() void {' '    mlx.engine.unhold();' '}' > "$TMP/t/src/p.zig"
expect_red "the owner's name in another file" "src/p.zig:3:9 \[engine-hold-end\] engineHoldEnd"
rm "$TMP/t/src/p.zig"
expect_green "the owner and the engine's own files"

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

# rule template-unchecked-utf8
tree
mkdir -p "$TMP/t/src/template"
printf '%s\n' 'const std = @import("std");' 'fn f(s: []const u8) usize {' '    var it = std.unicode.Utf8View.initUnchecked(s).iterator();' '    var n: usize = 0;' '    while (it.nextCodepointSlice()) |_| n += 1;' '    return n;' '}' 'test "t" { _ = std.unicode.Utf8View.initUnchecked("x"); }' > "$TMP/t/src/template/p.zig"
expect_red "an unchecked UTF-8 view under src/template/" "src/template/p.zig:3:35 \[template-unchecked-utf8\] f: Utf8View.initUnchecked under src/template/"

tree
printf '%s\n' 'const std = @import("std");' 'fn f(s: []const u8) void { _ = std.unicode.Utf8View.initUnchecked(s); }' > "$TMP/t/src/p.zig"
expect_green "an unchecked UTF-8 view outside src/template/"

# rule utf8-decode-raw
tree
printf '%s\n' 'const std = @import("std");' 'fn f(s: []const u8) u21 {' '    return std.unicode.utf8Decode(s) catch 0;' '}' > "$TMP/t/src/p.zig"
expect_red "a raw utf8Decode on a slice" "src/p.zig:3:24 \[utf8-decode-raw\] f: std.unicode.utf8Decode: decode with utf8_seq.decode"

tree
printf '%s\n' 'const std = @import("std");' 'pub fn decode(s: []const u8) ?u21 {' '    return std.unicode.utf8Decode(s) catch null;' '}' > "$TMP/t/src/utf8_seq.zig"
printf '%s\n' 'const std = @import("std");' 'fn f(s: []const u8) usize {' '    return std.unicode.utf8ByteSequenceLength(s[0]) catch 1;' '}' > "$TMP/t/src/p.zig"
expect_green "utf8_seq.zig itself, and utf8ByteSequenceLength elsewhere"

# rule refusing-eager
tree
printf '%s\n' 'const core = @import("core.zig");' 'fn f(g: ?usize) core.Error!usize {' '    const negative = core.refusing("a negative width", error.TensorUnsupported);' \
  '    return g orelse negative;' '}' > "$TMP/t/src/p.zig"
expect_red "a refusal reason bound before any refusal" "src/p.zig:3:27 \[refusing-eager\] f: core.refusing bound to 'negative' before it refuses"

tree
printf '%s\n' 'const core = @import("core.zig");' 'fn f(g: ?usize) core.Error!usize {' '    return g orelse core.refusing("a negative width", error.TensorUnsupported);' \
  '}' 'pub fn refusing(comptime why: []const u8, e: core.Error) core.Error {' '    _ = why;' '    return e;' '}' > "$TMP/t/src/p.zig"
expect_green "a refusal reason given where it refuses, and refusing's own definition"

# rule refusal-unnamed
tree
mkdir -p "$TMP/t/src/tensor"
printf '%s\n' 'const core = @import("core.zig");' 'fn f(n: usize) core.Error!usize {' '    if (n == 0) return error.TensorUnsupported;' '    return n;' '}' > "$TMP/t/src/tensor/p.zig"
expect_red "an engine refusal with no reason" "src/tensor/p.zig:3:24 \[refusal-unnamed\] f: error.TensorUnsupported refused with no reason: give one through core.refusing"

tree
mkdir -p "$TMP/t/src/tensor"
printf '%s\n' 'const core = @import("core.zig");' 'fn f(n: usize) core.Error!usize {' '    if (n == 0) return error.TensorDtype;' '    return n;' '}' > "$TMP/t/src/tensor/p.zig"
expect_red "a dtype refusal with no reason" "src/tensor/p.zig:3:24 \[refusal-unnamed\] f: error.TensorDtype refused with no reason: give one through core.refusing"

tree
mkdir -p "$TMP/t/src/tensor"
printf '%s\n' 'const core = @import("core.zig");' 'fn f(n: usize) core.Error!usize {' '    if (n == 0) return core.refusing("an empty tensor", error.TensorUnsupported);' \
  '    return n;' '}' 'fn g(e: core.Error) bool {' '    return switch (e) {' '        error.TensorUnsupported => true,' '        else => e == error.TensorUnsupported,' '    };' '}' > "$TMP/t/src/tensor/p.zig"
printf '%s\n' 'fn h(n: usize) !usize {' '    if (n == 0) return error.TensorUnsupported;' '    return n;' '}' > "$TMP/t/src/q.zig"
expect_green "a refusal with its reason, a prong or comparison, and a file outside src/tensor/"

# rule disk-read-stream
tree
printf '%s\n' 'fn load() void {' '    const cpu = mlx.mlx_default_cpu_stream_new();' '    _ = cpu;' '}' > "$TMP/t/src/kv_disk_cache.zig"
expect_red "a disk-tier read on its own CPU stream" "src/kv_disk_cache.zig:2:21 \[disk-read-stream\] load: a CPU stream of its own in the disk tier"

tree
printf '%s\n' 'fn reading() void {' '    _ = mlx.mlx_default_cpu_stream_new();' '}' 'test "t" {' '    _ = mlx.mlx_default_cpu_stream_new();' '}' > "$TMP/t/src/kv_disk_cache.zig"
printf '%s\n' 'fn other() void {' '    _ = mlx.mlx_default_cpu_stream_new();' '}' > "$TMP/t/src/model.zig"
expect_green "reading itself, a test, and a file outside the disk tier"

# rule cache-gpu-stream
tree
printf '%s\n' 'fn restore(e: *E) !void {' '    try patch(mlx.gpuStream(), e);' '}' > "$TMP/t/src/ssm_cache.zig"
expect_red "MLX's stream in the SSM cache" "src/ssm_cache.zig:2:19 \[cache-gpu-stream\] restore: MLX's stream in the cache"

tree
printf '%s\n' 'fn arraysStream(self: *const C) mlx.mlx_stream {' '    return self.arrays_on orelse mlx.gpuStream();' '}' \
  'fn seed() void {' '    _ = mlx.gpuStream();' '}' 'test "t" {' '    seed();' '    _ = mlx.gpuStream();' '}' > "$TMP/t/src/prefix_cache.zig"
printf '%s\n' 'fn other() void {' '    _ = mlx.gpuStream();' '}' > "$TMP/t/src/model.zig"
expect_green "arraysStream's fallback, a test and a fn only tests reach, and a file outside the caches"

# rule in-place-buffer
tree
printf '%s\n' 'fn grow(buf: *A, s: S) !void {' '    var z = mlx.mlx_array_new();' '    try mlx.check(mlx.mlx_zeros(&z, &shape, 2, dt, s));' '    try mlx.check(mlx.mlx_slice_update(&up, z, buf.*, &a, 2, &b, 2, &c, 2, s));' '}' > "$TMP/t/src/p.zig"
expect_red "a buffer made by mlx_zeros and written by slice_update" "src/p.zig:3:23 \[in-place-buffer\] grow: a buffer made by mlx_zeros and written by slice_update"

tree
printf '%s\n' 'fn inPlaceBuffer(s: S) !A {' '    try mlx.check(mlx.mlx_zeros(&z, &shape, 2, dt, s));' '}' 'fn fill(z: A, s: S) !void {' '    try mlx.check(mlx.mlx_slice_update(&up, z, x, &a, 2, &b, 2, &c, 2, s));' '}' \
  'fn seed(s: S) !void {' '    try mlx.check(mlx.mlx_zeros(&z, &shape, 2, dt, s));' '    try mlx.check(mlx.mlx_slice_update(&up, z, x, &a, 2, &b, 2, &c, 2, s));' '}' 'test "t" {' '    try seed(s);' '    try mlx.check(mlx.mlx_zeros(&z, &shape, 2, dt, s));' '    try mlx.check(mlx.mlx_slice_update(&up, z, x, &a, 2, &b, 2, &c, 2, s));' '}' > "$TMP/t/src/p.zig"
printf '%s\n' 'fn grow(s: S) !void {' '    try mlx.check(mlx.mlx_zeros(&z, &shape, 2, dt, s));' '    try mlx.check(mlx.mlx_slice_update(&up, z, x, &a, 2, &b, 2, &c, 2, s));' '}' > "$TMP/t/src/mlx_pages.zig"
expect_green "zeros and a write in different fns, a test and a fn only tests reach, and mlx_pages itself"

# rule cache-room
tree
printf '%s\n' 'fn grow(s: S) !void {' '    mlx.engine.roomFor(&shape, dt);' '    buf = try mlx_pages.inPlaceBuffer(&shape, dt, s);' '}' > "$TMP/t/src/kv_cache.zig"
expect_red "the engine's room taken beside a constructor" "src/kv_cache.zig:2:16 \[cache-room\] grow: the engine's room taken outside mlx_pages"

tree
printf '%s\n' 'fn zeros(s: S) !A {' '    mlx.engine.roomFor(&shape, dt);' '}' > "$TMP/t/src/mlx_pages.zig"
mkdir -p "$TMP/t/src/tensor"
printf '%s\n' 'pub fn roomFor(dims: []const c_int) void {' '    room(dims);' '}' 'fn other() void {' '    roomFor(&shape);' '}' > "$TMP/t/src/tensor/route.zig"
printf '%s\n' 'test "t" {' '    mlx.engine.roomFor(&shape, dt);' '}' > "$TMP/t/src/p.zig"
expect_green "mlx_pages, the engine itself and a test"

# rule entry-checkout
tree
printf '%s\n' 'fn refuse(e: *Entry) void {' '    e.checkout = null;' '}' > "$TMP/t/src/scheduler.zig"
expect_red "an entry's checkout cleared outside the hot cache" "src/scheduler.zig:2:7 \[entry-checkout\] refuse: an entry's checkout written by hand"

tree
printf '%s\n' 'fn retake(e: *Entry) void {' '    if (e.checkout) |*c| c.slot = 1;' '}' 'fn redonate(e: *Entry) void {' '    e.checkout.?.donated = 4;' '}' > "$TMP/t/src/prefix_cache.zig"
expect_red "a checkout written through a capture" "src/prefix_cache.zig:2:11 \[entry-checkout\] retake"
expect_red "a checkout written through .?" "src/prefix_cache.zig:5:7 \[entry-checkout\] redonate"

tree
printf '%s\n' 'fn beginCheckout(e: *Entry) void {' '    e.checkout = .{ .slot = 1, .kind = .whole };' '}' 'fn donateCheckout(e: *Entry) void {' '    if (e.checkout) |*c| c.donated = 4;' '}' \
  'fn endCheckout(e: *Entry) void {' '    e.checkout = null;' '}' 'fn held(e: *const Entry) bool {' '    return e.checkout != null and e.checkout.?.slot == 1;' '}' \
  'fn stub(e: *Entry) void {' '    e.checkout = null;' '}' 'test "t" {' '    stub(undefined);' '}' > "$TMP/t/src/prefix_cache.zig"
printf '%s\n' 'test "t" {' '    e.checkout = null;' '}' > "$TMP/t/src/scheduler.zig"
expect_green "the hot cache's checkout calls, a read, a test and a fn only tests reach"

# rule cooler-side-query
tree
printf '%s\n' 'fn side(c: *const C) bool {' '    return c.cooler_ask == .none;' '}' > "$TMP/t/src/p.zig"
expect_red "a side input read outside cooler_side" "src/p.zig:2:14 \[cooler-side-query\] side: cooler_ask outside src/cooler_side.zig"

tree
printf '%s\n' 'fn side(c: *const C) bool {' '    return c.cooler_ask == .none;' '}' > "$TMP/t/src/cooler_side.zig"
printf '%s\n' 'test "t" {' '    c.cooler_for_test = true;' '}' > "$TMP/t/src/p.zig"
expect_green "cooler_side itself, and a test"

# rule promise-active
tree
printf '%s\n' 'fn contextNow(c: *const C) u32 {' '    const active = gpu_mem.activeBytes();' '    return contextAt(c, active);' '}' > "$TMP/t/src/p.zig"
expect_red "a pinned context read off the raw active bytes" "src/p.zig:2:28 \[promise-active\] contextNow: the active bytes read raw"

tree
printf '%s\n' 'fn contextNow(c: *const C) u32 {' '    const active = gpu_mem.activeBytes();' '    return contextAt(ceilingOf(active), promisedActive(active));' '}' \
  'test "t" {' '    _ = gpu_mem.activeBytes();' '}' > "$TMP/t/src/p.zig"
printf '%s\n' 'pub fn activeBytes() usize {' '    return 0;' '}' 'fn floor() usize {' '    return activeBytes();' '}' > "$TMP/t/src/gpu_mem.zig"
expect_green "a reading through promisedActive, a test, and gpu_mem's own"

# rule response-store-field
tree
printf '%s\n' 'fn take(store: *S, id: []const u8) void {' '    _ = store.by_id.fetchRemove(id);' '}' > "$TMP/t/src/p.zig"
expect_red "the Responses store's map outside its file" "src/p.zig:2:15 \[response-store-field\] take: by_id outside src/responses.zig: go through ResponseStore's methods"

tree
printf '%s\n' 'fn take(self: *S, id: []const u8) void {' '    _ = self.by_id.fetchRemove(id);' '    self.recency.remove(id);' '}' > "$TMP/t/src/responses.zig"
expect_green "the store's own file"

# rule engine-lifecycle
tree
printf '%s\n' 'fn load() void {' '    mlx.engine.hold();' '    mlx.engine.own_weights = true;' '}' > "$TMP/t/src/p.zig"
expect_red "a load's engine hold outside the shared seam" "src/p.zig:2:16 \[engine-lifecycle\] load: engine.hold outside the shared Cooler seam"

tree
mkdir -p "$TMP/t/src/arch"
printf '%s\n' 'fn load() void {' '    mlx.engine.hold();' '}' > "$TMP/t/src/arch/cooler.zig"
printf '%s\n' 'fn wrap() bool {' '    return mlx.engine.own_weights;' '}' 'test "t" {' '    mlx.engine.seal(true);' '}' > "$TMP/t/src/p.zig"
expect_green "the seam itself, a read of own_weights, and a test"

# rule random-draw-seam
tree
printf '%s\n' 'fn draw(o: *A, l: A, k: A, s: S) void {' '    _ = mlx.mlx_random_categorical(o, l, -1, k, s);' '}' > "$TMP/t/src/p.zig"
expect_red "a draw keyed outside the sampler" "src/p.zig:2:13 \[random-draw-seam\] draw: a categorical draw outside src/sampler.zig"

tree
printf '%s\n' 'fn draw(o: *A, l: A, k: A, s: S) void {' '    _ = mlx.mlx_random_categorical(o, l, -1, k, s);' '}' > "$TMP/t/src/sampler.zig"
printf '%s\n' 'test "t" {' '    _ = mlx.mlx_random_categorical(o, l, -1, k, s);' '}' > "$TMP/t/src/p.zig"
expect_green "the sampler itself, and a test"

# rule host-maker-unchecked
tree
printf '%s\n' 'fn ids(v: *const i32) A {' '    return mlx.mlx_array_new_data(v, &.{1}, 1, .int32);' '}' > "$TMP/t/src/p.zig"
expect_red "a raw maker on the shared path" "src/p.zig:2:16 \[host-maker-unchecked\] ids: mlx_array_new_data unchecked"

tree
mkdir -p "$TMP/t/src/glm5"
printf '%s\n' 'fn ids(v: *const i32) !A {' '    const a = try host_side.made(mlx.mlx_array_new_data(v, &.{1}, 1, .int32));' '    _ = try cache.put(0, 1, mlx.mlx_array_new_int(1));' '    return a;' '}' > "$TMP/t/src/p.zig"
printf '%s\n' 'fn one() A {' '    return mlx.mlx_array_new_int(1);' '}' > "$TMP/t/src/glm5/g.zig"
printf '%s\n' 'test "t" {' '    _ = mlx.mlx_array_new_float(1);' '}' > "$TMP/t/src/q.zig"
expect_green "a maker made or put, an MLX-only arch, and a test"

# rule closure-compile-seam
tree
printf '%s\n' 'const arch_state = @import("arch_state.zig");' 'fn load(x: *T) void {' '    x.compileMoeRouting();' '    arch_state.compile(x, .gates);' '}' > "$TMP/t/src/p.zig"
expect_red "a loader compiling its own closures" "src/p.zig:3:7 \[closure-compile-seam\] load: 'compileMoeRouting' compiled outside model_load.compileClosures"

tree
printf '%s\n' 'const arch_state = @import("arch_state.zig");' 'pub fn compileClosures(x: *T) void {' '    x.compileMoeRouting();' '    arch_state.compile(x, .all);' '}' > "$TMP/t/src/model_load.zig"
printf '%s\n' 'fn f(x: *T) void {' '    x.compile();' '}' 'test "t" {' '    x.compileMoeRouting();' '}' > "$TMP/t/src/p.zig"
printf '%s\n' 'fn g(x: *T) void {' '    x.compileGelu();' '}' > "$TMP/t/src/p_test.zig"
expect_green "the seam itself, another compile, a test, and a test file"

# rule error-fallback
tree
printf '%s\n' 'fn fused() !?u8 { return null; }' 'fn f() u8 {' '    blk: {' '        const v = (fused() catch null) orelse break :blk;' '        return v;' '    }' '    return 0;' '}' > "$TMP/t/src/p.zig"
expect_red "an error caught into the null fallback" "src/p.zig:4:28 \[error-fallback\] f: an error caught into the null fallback"

tree
printf '%s\n' 'fn fused() !?u8 { return null; }' 'fn f() !u8 {' '    blk: {' '        const v = (try fused()) orelse break :blk;' '        return v;' '    }' '    return 0;' '}' \
  'fn g() ?u8 {' '    return (fused() catch null) orelse return null;' '}' > "$TMP/t/src/p.zig"
expect_green "a tried fused arm, and a catch null that is not a fallback break"

# rule caught-error-unsettled
tree
printf '%s\n' 'fn f(slot: anytype) void {' '    g() catch |e| {' '        slot.markError(@errorName(e));' '    };' '}' 'fn g() !void {}' > "$TMP/t/src/scheduler.zig"
expect_red "an inference-thread failure named by @errorName" "src/scheduler.zig:3:24 \[caught-error-unsettled\] f: @errorName(e) of a caught failure"

tree
printf '%s\n' 'const log = @import("log.zig");' 'fn f() ?u8 {' '    return g() catch |err| blk: {' '        log.warn("{s}\\n", .{@errorName(err)});' '        break :blk null;' '    };' '}' 'fn g() !u8 { return 0; }' > "$TMP/t/src/mtp_tick.zig"
expect_red "a labelled handler" "src/mtp_tick.zig:4:30 \[caught-error-unsettled\] f: @errorName(err)"

tree
printf '%s\n' 'const mlx = @import("mlx.zig");' 'fn f(slot: anytype) void {' '    g() catch |e| slot.markError(mlx.settledErrorName(e));' '    mlx.checkErrorDecode() catch |e| slot.markError(@errorName(e));' '    g() catch |e| slot.markError(@errorName(error.Other));' '    _ = e2;' '}' 'const e2 = 0;' 'fn g() !void {}' > "$TMP/t/src/scheduler.zig"
printf '%s\n' 'fn f(slot: anytype) void {' '    g() catch |e| slot.markError(@errorName(e));' '}' 'fn g() !void {}' > "$TMP/t/src/p.zig"
expect_green "a settled name, a checkpoint's own error, another error's name, and a file off the inference thread"

# rule config-free-stale
tree
printf '%s\n' 'const mlx = @import("mlx.zig");' 'var cfg: ?u8 = null;' 'fn f() !void {' '    if (cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);' '    try build();' '}' 'fn build() !void {}' > "$TMP/t/src/p.zig"
expect_red "a freed config left in its slot" "src/p.zig:4:5 \[config-free-stale\] f: a kernel config freed and left in its slot"

tree
printf '%s\n' 'const mlx = @import("mlx.zig");' 'var cfgs: [2]?u8 = .{ null, null };' 'fn f(i: usize) void {' '    if (cfgs[i]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);' '    cfgs[i] = null;' '}' \
  'fn g(i: usize, n: u8) void {' '    if (cfgs[i]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);' '    cfgs[i] = n;' '}' > "$TMP/t/src/p.zig"
expect_green "a freed slot nulled or given the new config next"

# rule returned-handle-unguarded
tree
printf '%s\n' 'const mlx = @import("mlx.zig");' 'fn f(x: mlx.mlx_array) !mlx.mlx_array {' '    var out = mlx.mlx_array_new();' '    try mlx.check(mlx.mlx_sigmoid(&out, x, null));' '    return out;' '}' > "$TMP/t/src/p.zig"
expect_red "a returned handle with no errdefer" "src/p.zig:5:5 \[returned-handle-unguarded\] f: 'out' is returned, and an error before it leaks it"

tree
printf '%s\n' 'const mlx = @import("mlx.zig");' 'fn f(x: mlx.mlx_array) !mlx.mlx_array {' '    var out = mlx.mlx_array_new();' '    errdefer _ = mlx.mlx_array_free(out);' '    try mlx.check(mlx.mlx_sigmoid(&out, x, null));' '    return out;' '}' \
  'fn g(x: mlx.mlx_array) !void {' '    var t = mlx.mlx_array_new();' '    defer _ = mlx.mlx_array_free(t);' '    try mlx.check(mlx.mlx_sigmoid(&t, x, null));' '}' 'fn h() mlx.mlx_array {' '    var e = mlx.mlx_array_new();' '    return e;' '}' > "$TMP/t/src/p.zig" \
  && printf '%s\n' 'const mlx = @import("mlx.zig");' 'fn k(x: mlx.mlx_array) !mlx.mlx_array {' '    var out = mlx.mlx_array_new();' '    try mlx.check(mlx.mlx_sigmoid(&out, x, null));' '    return out;' '}' > "$TMP/t/src/p_test.zig"
expect_green "a guarded returned handle, a deferred local, a return with no try, and a _test file"

# rule array-unowned-across-try
tree
printf '%s\n' 'const mlx = @import("mlx.zig");' 'fn f(x: mlx.mlx_array, out: *mlx.mlx_array) !void {' '    var m = mlx.mlx_array_new();' '    try mlx.check(mlx.mlx_sigmoid(&m, x, null));' '    try mlx.check(mlx.mlx_array_eval(m));' '    out.* = m;' '}' \
  'fn g(x: mlx.mlx_array) !mlx.mlx_array {' '    const parts = [_]mlx.mlx_array{' '        try h(x),' '        try h(x),' '    };' '    return parts[0];' '}' 'fn h(x: mlx.mlx_array) !mlx.mlx_array {' '    return x;' '}' > "$TMP/t/src/p.zig"
expect_red "a filled local across a later try" "src/p.zig:5:5 \[array-unowned-across-try\] f: 'm' holds a filled array across this try"
expect_red "an array literal of several trys" "src/p.zig:11:9 \[array-unowned-across-try\] g: an array literal of mlx arrays made by several trys"

tree
printf '%s\n' 'const mlx = @import("mlx.zig");' 'const Sc = struct {' '    fn keep(_: Sc, x: mlx.mlx_array) !mlx.mlx_array {' '        return x;' '    }' '    fn res(_: Sc, rc: c_int, o: *const mlx.mlx_array) !mlx.mlx_array {' '        try mlx.check(rc);' '        return o.*;' '    }' '};' \
  'fn f(x: mlx.mlx_array, out: *mlx.mlx_array) !void {' '    var m = mlx.mlx_array_new();' '    errdefer _ = mlx.mlx_array_free(m);' '    try mlx.check(mlx.mlx_sigmoid(&m, x, null));' '    try mlx.check(mlx.mlx_array_eval(m));' '    out.* = m;' '}' \
  'fn g(x: mlx.mlx_array, out: *mlx.mlx_array, sc: Sc) !void {' '    var a = mlx.mlx_array_new();' '    try mlx.check(mlx.mlx_sigmoid(&a, x, null));' '    out.* = a;' '    try mlx.check(mlx.mlx_array_eval(x));' \
  '    var b = mlx.mlx_array_new();' '    try mlx.check(mlx.mlx_sigmoid(&b, x, null));' '    _ = try sc.keep(b);' \
  '    var c = mlx.mlx_array_new();' '    const r = try sc.res(mlx.mlx_sigmoid(&c, x, null), &c);' '    try mlx.check(mlx.mlx_array_eval(r));' '}' \
  'fn h(x: mlx.mlx_array, out: *mlx.mlx_array, wide: bool) !void {' '    var d = mlx.mlx_array_new();' '    if (wide) try mlx.check(mlx.mlx_sigmoid(&d, x, null)) else try mlx.check(mlx.mlx_exp(&d, x, null));' '    out.* = d;' '}' > "$TMP/t/src/p.zig" \
  && printf '%s\n' 'const mlx = @import("mlx.zig");' 'fn k(x: mlx.mlx_array) !void {' '    var m = mlx.mlx_array_new();' '    try mlx.check(mlx.mlx_sigmoid(&m, x, null));' '    try mlx.check(mlx.mlx_array_eval(m));' '}' > "$TMP/t/src/p_test.zig"
expect_green "a guarded local, a hand-off before the try, a scope's keep and res, fills on two branches, and a _test file"

tree
printf '%s\n' 'const mlx = @import("mlx.zig");' 'fn take(e: *E, s: mlx.mlx_stream) !void {' '    const own = e.aux;' '    e.aux = .{ .ctx = null };' '    try work(s);' '    e.aux = own;' '}' > "$TMP/t/src/p.zig"
expect_red "a field's array taken out across a try" "src/p.zig:5:5 \[array-unowned-across-try\] take: 'own' holds an array it took from its owner across this try"

tree
printf '%s\n' 'const mlx = @import("mlx.zig");' 'fn take(e: *E, s: mlx.mlx_stream) !void {' '    const own = e.aux;' '    e.aux = .{ .ctx = null };' '    errdefer e.aux = own;' '    try work(s);' '    e.aux = own;' '}' \
  'fn give(e: *E, r: *R) !void {' '    const pre = e.state;' '    e.state = .{ .ctx = null };' '    try r.hold(pre);' '}' > "$TMP/t/src/p.zig"
expect_green "a taken array returned by an errdefer, and one a call takes"

# rule json-unchecked-shape
tree
printf '%s\n' 'const std = @import("std");' 'pub fn f(parsed: std.json.Parsed(std.json.Value)) ?std.json.Value {' '    const root = parsed.value.object;' \
  '    return root.get("k").?.object.get("j");' '}' > "$TMP/t/src/p.zig"
expect_red "a parsed document read as an object unchecked" "src/p.zig:3:31 \[json-unchecked-shape\] f: unchecked JSON shape 'parsed.value.object'"
expect_red "an unwrapped field read as an object unchecked" "src/p.zig:4:28 \[json-unchecked-shape\] f: unchecked JSON shape 'root.get(\"k\").?.object'"

tree
printf '%s\n' 'const std = @import("std");' 'pub fn g(parsed: std.json.Parsed(std.json.Value)) !?std.json.Value {' '    if (parsed.value != .object) return error.Shape;' \
  '    const v = parsed.value.object.get("k") orelse return null;' '    return if (v == .object) v.object.get("j") else null;' '}' \
  'fn helper(parsed: std.json.Parsed(std.json.Value)) std.json.ObjectMap { return parsed.value.object; }' \
  'fn h(x: anytype) void { _ = x.hidden.?.array(0); }' 'test "t" { _ = helper; }' > "$TMP/t/src/p.zig"
printf '%s\n' 'const std = @import("std");' 'pub fn k(parsed: std.json.Parsed(std.json.Value)) std.json.ObjectMap { return parsed.value.object; }' > "$TMP/t/src/p_test.zig"
expect_green "a checked tag, a test-only helper, a method named like a tag and a test file"

tree
printf '%s\n' 'const std = @import("std");' 'pub fn f(m: std.json.ObjectMap, v: std.json.Value) !u32 {' '    var it = m.iterator();' \
  '    while (it.next()) |e| _ = e.value_ptr.string;' '    return @intCast(v.integer);' '}' > "$TMP/t/src/p.zig"
expect_red "a map entry read as a string unchecked" "src/p.zig:4:43 \[json-unchecked-shape\] f: unchecked JSON shape 'e.value_ptr.string'"
expect_red "an @intCast of a JSON integer" "src/p.zig:5:12 \[json-unchecked-shape\] f: @intCast of a JSON integer"

tree
printf '%s\n' 'const std = @import("std");' 'pub fn g(m: std.json.ObjectMap, v: std.json.Value) !u32 {' '    var it = m.iterator();' \
  '    while (it.next()) |e| if (e.value_ptr.* == .string) {' '        _ = e.value_ptr.string;' '    };' \
  '    if (v != .integer or v.integer < 0 or v.integer > 9) return error.Range;' '    return @intCast(v.integer);' '}' > "$TMP/t/src/p.zig"
expect_green "a map entry checked through .*, and an integer bounded before its cast"

tree
printf '%s\n' 'const std = @import("std");' 'pub fn f(v: std.json.Value) ?u32 {' '    return switch (v) {' \
  '        .integer => |i| if (i < 0) null else @intCast(i),' '        .float => |x| @intFromFloat(x),' '        else => null,' '    };' '}' > "$TMP/t/src/p.zig"
expect_red "an @intCast of an .integer prong's capture" "src/p.zig:4:46 \[json-unchecked-shape\] f: @intCast of a JSON integer"

tree
printf '%s\n' 'const std = @import("std");' 'pub fn g(v: std.json.Value, w: u64) ?u32 {' '    return switch (v) {' \
  '        .integer => |i| std.math.cast(u32, i),' '        .string => |i| @intCast(i.len + w),' '        else => @intCast(w),' '    };' '}' > "$TMP/t/src/p.zig"
expect_green "an .integer capture narrowed by std.math.cast, and a cast in another prong"

# rule owned-insert-arg
tree
printf '%s\n' 'const std = @import("std");' 'pub fn f(a: std.mem.Allocator, xs: *std.ArrayList([]u8), s: []const u8) !void {' \
  '    try xs.append(a, try a.dupe(u8, s));' '}' > "$TMP/t/src/p.zig"
expect_red "a copy made inside an append" "src/p.zig:3:22 \[owned-insert-arg\] f: 'dupe' makes an owned value inside 'append'"

tree
printf '%s\n' 'const std = @import("std");' 'pub fn g(a: std.mem.Allocator, xs: *std.ArrayList([]u8), s: []const u8) !void {' \
  '    try xs.ensureUnusedCapacity(a, 1);' '    xs.appendAssumeCapacity(try a.dupe(u8, s));' '    try appendRow(a, try a.dupe(u8, s));' '}' \
  'fn appendRow(a: std.mem.Allocator, s: []u8) !void { a.free(s); }' > "$TMP/t/src/p.zig"
expect_green "a reserved insert, and a free function named like an insert"

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

# rule test-unrooted
tree
mkdir -p "$TMP/t/src/sub" "$TMP/t/src/exl3"
printf '%s\n' 'test {' '    _ = @import("listed.zig");' '    _ = @import("sub/nested.zig");' '}' '// _ = @import("comment.zig");' > "$TMP/t/src/tests.zig"
printf '%s\n' 'test {' '    _ = @import("own.zig");' '    _ = @import("../up.zig");' '}' > "$TMP/t/src/exl3/root.zig"
printf '%s\n' 'pub fn f() void {}' 'test "listed" { f(); }' > "$TMP/t/src/listed.zig"
printf '%s\n' 'pub fn f() void {}' 'test "nested" { f(); }' > "$TMP/t/src/sub/nested.zig"
printf '%s\n' 'pub fn f() void {}' 'test "own" { f(); }' > "$TMP/t/src/exl3/own.zig"
printf '%s\n' 'pub fn f() void {}' 'test "up" { f(); }' > "$TMP/t/src/up.zig"
printf '%s\n' 'pub fn h() void {}' > "$TMP/t/src/sub/no_tests.zig"
expect_green "files a test root names, by a path from the root's folder, and a file with no test"
printf '%s\n' 'pub fn g() void {}' 'test "never run" { g(); }' > "$TMP/t/src/sub/un.zig"
expect_red "a file with a test no test root names" "src/sub/un.zig:2:1 \[test-unrooted\] -: sub/un.zig declares a test and no test root imports it"
rm "$TMP/t/src/sub/un.zig"
printf '%s\n' 'pub fn c() void {}' 'test "named only in a comment" { c(); }' > "$TMP/t/src/comment.zig"
expect_red "a file a root names only in a comment" "\[test-unrooted\] -: comment.zig"
rm "$TMP/t/src/comment.zig"
mkdir -p "$TMP/t/src/arch1"
printf '%s\n' 'pub const tests = .{};' > "$TMP/t/src/arch1/arch_tests.zig"
printf '%s\n' 'pub fn g() void {}' 'test "an arch file" { g(); }' > "$TMP/t/src/arch1/un.zig"
lint; grep -q "test-unrooted" "$TMP/out" && fail "a file beside an arch_tests.zig is arch-test-unlisted's, reported once"

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

# rule device-probe
tree
printf '%s\n' 'const std = @import("std");' 'pub fn withheld() bool {' '    return std.c.getenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK") != null;' '}' > "$TMP/t/src/p.zig"
expect_red "the QA switch read outside the device profile" "src/p.zig:3:25 \[device-probe\] withheld: \"SUSHI_FORCE_GPU_FAMILY_FALLBACK\" is read here"

tree
printf '%s\n' 'extern fn sysctlbyname(name: [*:0]const u8, a: ?*anyopaque, b: ?*usize, c: ?*anyopaque, d: usize) c_int;' \
  'pub fn os(buf: []u8) c_int {' '    var n = buf.len;' '    return sysctlbyname("kern.osproductversion", buf.ptr, &n, null, 0);' '}' \
  'fn info(key: [*:0]const u8) void { _ = key; }' 'pub fn arch() void {' '    info("architecture");' '}' > "$TMP/t/src/p.zig"
expect_red "the OS version read outside the device profile" "\[device-probe\] os: \"kern.osproductversion\" is read here"
expect_red "the GPU's name read outside the device profile" "\[device-probe\] arch: \"architecture\" is read here"

tree
mkdir -p "$TMP/t/src/exl3"
printf '%s\n' 'const std = @import("std");' 'pub fn withheld() bool {' '    return std.c.getenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK") != null;' '}' > "$TMP/t/src/exl3/device.zig"
printf '%s\n' 'const names = [_][]const u8{ "SUSHI_FORCE_GPU_FAMILY_FALLBACK", "MLX_METAL_GPU_ARCH" };' \
  'pub fn count() usize {' '    const text = "architecture";' '    return names.len + text.len;' '}' > "$TMP/t/src/p.zig"
expect_green "the device profile's own reads, and the names in a list or a constant"

# rule log-line-end
tree
printf '%s\n' 'const log = @import("log.zig");' 'pub fn skip(name: []const u8) void {' '    log.info("[discovery] skip {s}: no config", .{name});' '}' > "$TMP/t/src/p.zig"
expect_red "a log line with no line end" "src/p.zig:3:14 \[log-line-end\] skip: log.info format has no line end"

tree
printf '%s\n' 'const std = @import("std");' 'const log = @import("log.zig");' 'pub fn f(name: []const u8, line: []const u8) void {' \
  '    log.warn("[load] {s}\n", .{name});' '    log.debug("{s}", .{line});' '    log.err("[load] " ++ "{s}\n", .{name});' \
  '    std.log.info("no end", .{});' '}' > "$TMP/t/src/p.zig"
expect_green "a line end, a whole line passed through, a joined format and another logger"

# rule scheduler-hand-built
tree
printf '%s\n' 'const scheduler_mod = @import("scheduler.zig");' 'test "t" {' '    var sch: scheduler_mod.Scheduler = undefined;' \
  '    sch.queue_cap = 2;' '    _ = try sch.submit(.{});' '}' > "$TMP/t/src/p.zig"
expect_red "a field-by-field Scheduler given a submit" "src/p.zig:3:9 \[scheduler-hand-built\] t: Scheduler 'sch' built field by field reaches submit"

tree
printf '%s\n' 'const scheduler_mod = @import("scheduler.zig");' 'test "t" {' '    var sch = scheduler_mod.submitTestScheduler(a, 2);' \
  '    _ = try sch.submit(.{});' '    var bare: scheduler_mod.Scheduler = undefined;' '    _ = bare.cancelSlotById(1);' '}' > "$TMP/t/src/p.zig"
expect_green "a Scheduler from submitTestScheduler, and a bare one that never submits"

# rule conn-read-unbounded
tree
printf '%s\n' 'const Conn = struct {' '    fn reader(c: *Conn) *R { return &c.read_state.interface; }' \
  '    fn takeReadable(c: *Conn, buf: []u8) !usize { return c.reader().readVec(buf); }' '};' \
  'fn body(c: *Conn, buf: []u8) !void {' '    try c.reader().readSliceAll(buf);' '}' > "$TMP/t/src/http_conn.zig"
expect_red "a read of the connection's reader outside Conn.read" "src/http_conn.zig:6:11 \[conn-read-unbounded\] body: the connection's reader outside Conn.read"

tree
printf '%s\n' 'fn chunk(c: *Conn, buf: []u8) !usize {' '    return c.takeReadable(buf);' '}' > "$TMP/t/src/http_conn.zig"
expect_red "a take that skips Conn.read's wait" "src/http_conn.zig:2:14 \[conn-read-unbounded\] chunk:"

tree
printf '%s\n' 'fn peek(conn: *Conn) usize {' '    return conn.read_state.interface.bufferedLen();' '}' > "$TMP/t/src/server.zig"
expect_red "read_state.interface outside the connection file" "src/server.zig:2:17 \[conn-read-unbounded\] peek:"

tree
printf '%s\n' 'const Conn = struct {' '    fn reader(c: *Conn) *R { return &c.read_state.interface; }' \
  '    fn awaitReadable(c: *Conn) bool { return c.reader().bufferedLen() > 0; }' \
  '    fn read(c: *Conn, buf: []u8) !usize { return c.takeReadable(buf); }' '};' \
  'fn open(f: F, io: Io, buf: []u8) R { return f.reader(io, buf); }' 'fn err(c: *Conn) ?E { return c.read_state.err; }' \
  'test "t" {' '    _ = conn.reader().bufferedLen();' '}' > "$TMP/t/src/http_conn.zig"
printf '%s\n' 'fn lines(r: *R) void {' '    _ = r.reader().takeDelimiter(10);' '}' > "$TMP/t/src/cli.zig"
expect_green "Conn's own reads, a reader with arguments, read_state.err, a test, and another file's reader()"

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
