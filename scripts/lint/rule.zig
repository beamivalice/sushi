//! What a lint rule sees and says. A rule file exports `name`, `reason` and `check`; main.zig lists the rules.
const std = @import("std");
const Ast = std.zig.Ast;

pub const Finding = struct {
    file: []const u8,
    rule: []const u8,
    fn_name: []const u8,
    line: usize,
    col: usize,
    message: []const u8,
};

pub const Report = struct {
    gpa: std.mem.Allocator,
    file: []const u8,
    rule: []const u8,
    findings: *std.ArrayList(Finding),

    /// Records a finding at `token`, inside the fn `fn_name` (a construct outside any fn passes "-").
    pub fn add(r: Report, tree: *const Ast, token: u32, fn_name: []const u8, comptime fmt: []const u8, args: anytype) !void {
        const loc = tree.tokenLocation(0, token);
        try r.findings.append(r.gpa, .{
            .file = r.file,
            .rule = r.rule,
            .fn_name = try r.gpa.dupe(u8, fn_name),
            .line = loc.line + 1,
            .col = loc.column + 1,
            .message = try std.fmt.allocPrint(r.gpa, fmt, args),
        });
    }
};

/// A fn or a test of the file: the allowlist key a finding inside it takes.
pub const Owner = struct { key: []const u8, first: u32, last: u32, is_test: bool };

/// Every fn and test of `tree`. `arena` outlives the call.
pub fn owners(arena: std.mem.Allocator, tree: *const Ast) !std.ArrayList(Owner) {
    var list: std.ArrayList(Owner) = .empty;
    for (0..tree.nodes.len) |i| {
        const node: Ast.Node.Index = @fromBackingInt(@intCast(i));
        switch (tree.nodeTag(node)) {
            .fn_decl => {
                var buf: [1]Ast.Node.Index = undefined;
                const proto = tree.fullFnProto(&buf, node) orelse continue;
                try list.append(arena, .{
                    .key = if (proto.name_token) |t| tree.tokenSlice(t) else "(anonymous)",
                    .first = tree.firstToken(node),
                    .last = tree.lastToken(node),
                    .is_test = false,
                });
            },
            .test_decl => {
                const data = tree.nodeData(node).opt_token_and_node;
                try list.append(arena, .{ .key = try testKey(arena, tree, data[0].unwrap()), .first = tree.firstToken(data[1]), .last = tree.lastToken(data[1]), .is_test = true });
            },
            else => {},
        }
    }
    return list;
}

/// The innermost fn or test that holds `token`; null outside any.
pub fn ownerOf(list: []const Owner, token: u32) ?Owner {
    var found: ?Owner = null;
    for (list) |o| if (token >= o.first and token <= o.last and (found == null or o.last - o.first < found.?.last - found.?.first)) {
        found = o;
    };
    return found;
}

/// Whether `token` is inside a test, directly or in a fn a test declares.
pub fn inTest(list: []const Owner, token: u32) bool {
    for (list) |o| if (o.is_test and token >= o.first and token <= o.last) return true;
    return false;
}

/// The allowlist key of a test: its name with everything but letters and digits as `_`.
pub fn testKey(arena: std.mem.Allocator, tree: *const Ast, name_token: ?u32) ![]const u8 {
    const t = name_token orelse return "(unnamed)";
    const slice = tree.tokenSlice(t);
    const inner = if (slice.len >= 2 and slice[0] == '"') slice[1 .. slice.len - 1] else slice;
    const out = try arena.dupe(u8, inner);
    for (out) |*c| if (!std.ascii.isAlphanumeric(c.*)) {
        c.* = '_';
    };
    return out;
}
