//! Flash-Next's chat face: the effort words its template takes (`arch/chat.zig` reads it).
const std = @import("std");
const eff = @import("../effort.zig");

pub const model_types = [_][]const u8{"qwen4_exp"};

pub const fixtures = @import("qwen4_fixtures.zig");

/// high and max are folds: the template has no such word, so they are served at xhigh, with a notice.
pub const effort: eff.Offer = .{ .table = eff.makeSpec("qwen4_exp", "Qwen", &.{ .low, .medium, .xhigh }, &.{}, true, .{ .default = .medium, .rows = .{
    .{ .offered = true },
    .{ .offered = true, .sends = .{ .word = .low }, .cap = 2048 },
    .{ .offered = true, .sends = .{ .word = .medium }, .cap = 8192 },
    .{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh },
    .{ .offered = true, .sends = .{ .word = .xhigh } },
    .{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh },
} }) };

/// The stock Flash-Next template raises on a system turn that is not first. Its edit replaces the body of that one
/// branch, found in the parsed tree at any spacing, so the turn renders in place; every other path keeps its bytes.
/// A template without the branch (or with two) gets null: it keeps its native contract and the fold.
pub fn lateSystemTemplate(allocator: std.mem.Allocator, tpl: []const u8) !?[]const u8 {
    var diag: engine.Diagnostic = .{};
    var t = engine.Template.compile(allocator, tpl, &diag) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    defer t.deinit();
    if (!definesMacro(t.body, "render_content")) return null;
    var found: ?ast.Branch = null;
    var count: usize = 0;
    findRaise(t.body, &found, &count);
    if (count != 1) return null;
    // The offsets are into the normalized text (CRLF read as LF), which renders as the raw text does.
    const src = t.src;
    const b = found.?;
    const body_start = b.open_end + @as(u32, if (src[b.open_end] == '%') 2 else 3);
    return try std.mem.concat(allocator, u8, &.{
        src[0..body_start],
        "{{- '<|im_start|>system\\n' + (render_content(message.content, false, true)|trim) + '<|im_end|>\\n' }}",
        src[b.close..],
    });
}

const engine = @import("../template/template.zig");
const ast = engine.ast;

fn definesMacro(nodes: []const ast.Node, name: []const u8) bool {
    for (nodes) |n| if (n.kind == .macro and std.mem.eql(u8, n.kind.macro.name, name)) return true;
    return false;
}

/// Every `{% if message.role == "system" %}` whose body opens with `{% if not loop.first %}` holding only a
/// `raise_exception` call: `found` keeps that inner branch, `count` how many there are.
fn findRaise(nodes: []const ast.Node, found: *?ast.Branch, count: *usize) void {
    for (nodes) |n| switch (n.kind) {
        .if_ => |f| {
            for (f.branches) |br| {
                if (isRoleSystem(br.test_)) if (firstSignificant(br.body)) |first| if (raiseUnlessFirst(first)) |inner| {
                    found.* = inner;
                    count.* += 1;
                };
                findRaise(br.body, found, count);
            }
            findRaise(f.else_, found, count);
        },
        .for_ => |f| {
            findRaise(f.body, found, count);
            findRaise(f.else_, found, count);
        },
        .set_block => |sb| findRaise(sb.body, found, count),
        .macro => |m| findRaise(m.body, found, count),
        .call_block => |cb| findRaise(cb.caller.body, found, count),
        .filter_block => |fb| findRaise(fb.body, found, count),
        .generation => |g| findRaise(g, found, count),
        else => {},
    };
}

fn isName(e: *const ast.Expr, name: []const u8) bool {
    return e.kind == .name and std.mem.eql(u8, e.kind.name, name);
}

fn isAttr(e: *const ast.Expr, obj: []const u8, name: []const u8) bool {
    return e.kind == .getattr and isName(e.kind.getattr.obj, obj) and std.mem.eql(u8, e.kind.getattr.name, name);
}

fn isRoleSystem(e: *const ast.Expr) bool {
    if (e.kind != .compare) return false;
    const c = e.kind.compare;
    if (c.ops.len != 1 or c.ops[0].op != .eq or !isAttr(c.first, "message", "role")) return false;
    const rhs = c.ops[0].expr;
    return rhs.kind == .string and std.mem.eql(u8, rhs.kind.string, "system");
}

fn isBlank(n: ast.Node) bool {
    return n.kind == .data and std.mem.trim(u8, n.kind.data, " \t\r\n").len == 0;
}

fn firstSignificant(nodes: []const ast.Node) ?ast.Node {
    for (nodes) |n| if (!isBlank(n)) return n;
    return null;
}

/// The branch of `{% if not loop.first %}{{ raise_exception(...) }}{% endif %}`, else null.
fn raiseUnlessFirst(n: ast.Node) ?ast.Branch {
    if (n.kind != .if_) return null;
    const f = n.kind.if_;
    if (f.branches.len != 1 or f.else_.len != 0) return null;
    const br = f.branches[0];
    if (br.test_.kind != .not_ or !isAttr(br.test_.kind.not_, "loop", "first")) return null;
    var outputs: usize = 0;
    for (br.body) |b| {
        if (isBlank(b)) continue;
        if (b.kind != .output) return null;
        const call = b.kind.output;
        if (call.kind != .call or !isName(call.kind.call.callee, "raise_exception")) return null;
        outputs += 1;
    }
    return if (outputs == 1) br else null;
}

test "Flash-Next's built-in effort table: medium by default, low and medium capped, high and max fold to xhigh" {
    const testing = @import("std").testing;
    const q = effort.table.table;
    try testing.expectEqual(eff.Effort.medium, q.default);
    try testing.expectEqualDeep(eff.Row{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh }, q.row(.high));
    try testing.expectEqualDeep(eff.Row{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh }, q.row(.max));
    try testing.expectEqualDeep(eff.Row{ .offered = true, .sends = .{ .word = .low }, .cap = 2048 }, q.row(.low));
}
