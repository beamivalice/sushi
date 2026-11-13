//! `lint <src dir> <allowlist>`: parses every .zig file under the source dir once and runs each registered rule on its
//! tree. A finding prints as `file:line:col [rule] fn: message` and fails the run unless the allowlist names it.
//!
//! The allowlist holds `<rule> <file>:<fn> <reason>` lines. An entry that matches no finding is stale and fails, so
//! the list can only shrink. Adding a rule: a file beside this one, one line in `rules`, a probe in tests/test_lint.sh.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

const rules = .{
    @import("const_write.zig"),
    @import("std_xxhash3.zig"),
    @import("unused_import.zig"),
    @import("undefined_nested_write.zig"),
    @import("mlx_extern.zig"),
    @import("double_free.zig"),
    @import("scope_state.zig"),
    @import("template_runtime_safety.zig"),
    @import("template_superlinear_std.zig"),
    @import("template_unchecked_utf8.zig"),
    @import("json_unchecked_shape.zig"),
    @import("owned_insert_arg.zig"),
    @import("arch_test_unlisted.zig"),
    @import("template_text_search.zig"),
    @import("device_probe.zig"),
};

const skipped_dirs = [_][]const u8{ "zig-cache", ".zig-cache", "zig-out", "lib" };

fn skipped(path: []const u8) bool {
    var parts = std.mem.splitScalar(u8, path, '/');
    while (parts.next()) |part| for (skipped_dirs) |s| if (std.mem.eql(u8, part, s)) return true;
    return false;
}

const Entry = struct { rule: []const u8, key: []const u8, used: bool = false };

fn parseAllowlist(gpa: std.mem.Allocator, text: []const u8) !std.ArrayList(Entry) {
    var entries: std.ArrayList(Entry) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        var words = std.mem.tokenizeScalar(u8, line, ' ');
        const rule_name = words.next().?;
        const key = words.next() orelse {
            std.debug.print("allowlist: '{s}' needs <rule> <file>:<fn> <reason>\n", .{line});
            return error.BadAllowlist;
        };
        try entries.append(gpa, .{ .rule = rule_name, .key = key });
    }
    return entries;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var run_arena: std.heap.ArenaAllocator = .init(init.gpa);
    defer run_arena.deinit();
    const gpa = run_arena.allocator();
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    _ = args.next();
    const src_dir = args.next() orelse return error.MissingSourceDir;
    const allow_path = args.next() orelse return error.MissingAllowlist;

    const cwd = std.Io.Dir.cwd();
    const entries = try parseAllowlist(gpa, try cwd.readFileAlloc(io, allow_path, gpa, .limited(1 << 20)));
    for (entries.items) |e| {
        var known = false;
        inline for (rules) |r| known = known or std.mem.eql(u8, r.name, e.rule);
        if (!known) {
            std.debug.print("{s}: unknown rule '{s}'\n", .{ allow_path, e.rule });
            std.process.exit(1);
        }
    }

    var dir = try cwd.openDir(io, src_dir, .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(gpa);
    defer walker.deinit();
    var findings: std.ArrayList(rule.Finding) = .empty;
    var failed = false;
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, ".zig") or skipped(entry.path)) continue;
        const source = try dir.readFileAllocOptions(io, entry.path, gpa, .limited(64 << 20), .of(u8), 0);
        defer gpa.free(source);
        const file = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ std.mem.trimEnd(u8, src_dir, "/"), entry.path });

        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const tree = try Ast.parse(arena, source, .{});
        if (tree.errors.len != 0) {
            std.debug.print("{s}: does not parse\n", .{file});
            failed = true;
            continue;
        }
        inline for (rules) |r| try r.check(arena, &tree, .{ .gpa = gpa, .file = file, .rule = r.name, .findings = &findings });
    }

    for (findings.items) |f| {
        var allowed = false;
        for (entries.items) |*e| {
            const sep = std.mem.lastIndexOfScalar(u8, e.key, ':') orelse continue;
            if (!std.mem.eql(u8, e.rule, f.rule) or !std.mem.eql(u8, e.key[0..sep], f.file) or !std.mem.eql(u8, e.key[sep + 1 ..], f.fn_name)) continue;
            e.used = true;
            allowed = true;
        }
        if (allowed) continue;
        failed = true;
        std.debug.print("{s}:{d}:{d} [{s}] {s}: {s}\n", .{ f.file, f.line, f.col, f.rule, f.fn_name, f.message });
    }
    for (entries.items) |e| if (!e.used) {
        failed = true;
        std.debug.print("{s}: stale entry '{s} {s}' matches no finding\n", .{ allow_path, e.rule, e.key });
    };
    if (failed) std.process.exit(1);
}
