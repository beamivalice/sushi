//! Rule `test-unrooted`: a file under the source dir that declares a test and that no test root imports. Zig runs a
//! file's tests only when a test block reaches the file by `_ = @import(...)`; a file the code merely imports keeps
//! its tests out of every binary, and nothing fails. The roots are the files build.zig compiles tests from; a file
//! beside an `arch_tests.zig` is `arch-test-unlisted`'s. Active when the source dir has `tests.zig`. Reads the roots
//! from disk and scans their string literal tokens; no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "test-unrooted";
pub const reason = "Zig runs a file's tests only when a test root imports the file: an unrooted file's tests never run";

/// The test roots build.zig compiles, relative to the source dir: the MLX and host tiers, the bucket list the MLX
/// root pulls in, and the exl3, encoder and sushibar binaries.
const roots = [_][]const u8{ "tests.zig", "tests_host.zig", "test_rest.zig", "exl3/root.zig", "exl3/host_min.zig", "encoder/root.zig", "sushibar_main.zig" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    const at = for (tree.rootDecls()) |decl| {
        if (tree.nodeTag(decl) == .test_decl) break tree.firstToken(decl);
    } else return;
    const cut = std.mem.indexOfScalar(u8, report.file, '/') orelse return;
    const src = report.file[0..cut];
    const rel = report.file[cut + 1 ..];
    for (roots) |r| if (std.mem.eql(u8, r, rel)) return;
    const dir = if (std.mem.lastIndexOfScalar(u8, report.file, '/')) |slash| report.file[0..slash] else return;
    const io = std.Io.Threaded.global_single_threaded.io();
    const cwd = std.Io.Dir.cwd();
    const beside = try std.fmt.allocPrint(arena, "{s}/arch_tests.zig", .{dir});
    if (cwd.access(io, beside, .{})) |_| return else |_| {}
    var active = false;
    for (roots) |r| {
        const path = try std.fmt.allocPrint(arena, "{s}/{s}", .{ src, r });
        const text = cwd.readFileAllocOptions(io, path, arena, .limited(1 << 20), .of(u8), 0) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => return e,
        };
        if (std.mem.eql(u8, r, "tests.zig")) active = true;
        const root_dir = std.fs.path.dirname(path) orelse src;
        const list = try Ast.parse(arena, text, .{});
        var t: u32 = 0;
        while (t < list.tokens.len) : (t += 1) {
            if (list.tokenTag(t) != .string_literal) continue;
            const quoted = list.tokenSlice(t);
            const named = quoted[1 .. quoted.len - 1];
            if (!std.mem.endsWith(u8, named, ".zig")) continue;
            const resolved = try std.fs.path.resolvePosix(arena, &.{ root_dir, named });
            if (std.mem.eql(u8, resolved, report.file)) return;
        }
    }
    if (!active) return;
    try report.add(tree, at, "-", "{s} declares a test and no test root imports it: its tests never run", .{rel});
}
