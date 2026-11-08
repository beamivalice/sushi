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
