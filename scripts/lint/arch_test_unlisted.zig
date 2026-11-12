//! Rule `arch-test-unlisted`: a file that declares a test, sits beside an `arch_tests.zig` and is not imported
//! there. The test buckets select tests by the files that list names, so the file's tests never run and nothing
//! fails. Reads the list from disk and scans its string literal tokens; no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "arch-test-unlisted";
pub const reason = "the test buckets select tests by the files arch_tests.zig lists: an unlisted file's tests never run";

const list_name = "arch_tests.zig";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    const slash = std.mem.lastIndexOfScalar(u8, report.file, '/') orelse return;
    const base = report.file[slash + 1 ..];
    if (std.mem.eql(u8, base, list_name)) return;
    const at = for (tree.rootDecls()) |decl| {
        if (tree.nodeTag(decl) == .test_decl) break tree.firstToken(decl);
    } else return;
    const list_path = try std.fmt.allocPrint(arena, "{s}/{s}", .{ report.file[0..slash], list_name });
    const io = std.Io.Threaded.global_single_threaded.io();
    const text = std.Io.Dir.cwd().readFileAllocOptions(io, list_path, arena, .limited(1 << 20), .of(u8), 0) catch |e| switch (e) {
        error.FileNotFound => return,
        else => return e,
    };
    const list = try Ast.parse(arena, text, .{});
    var t: u32 = 0;
    while (t < list.tokens.len) : (t += 1) {
        if (list.tokenTag(t) != .string_literal) continue;
        const quoted = list.tokenSlice(t);
        if (quoted.len == base.len + 2 and std.mem.eql(u8, quoted[1 .. quoted.len - 1], base)) return;
    }
    try report.add(tree, at, "-", "{s} declares a test and {s} does not list it: its tests never run", .{ base, list_path });
}
