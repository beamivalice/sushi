//! Rule `detached-global-read`: a file that detaches a thread (`.detach()`) reads a server global that shutdown frees
//! (`server_state.global_registry`, `server_state.global_scheduler`). A detached thread cannot be joined, so one that
//! reaches those globals reads what `main` frees as the server stops, unless `stopServing` waits for it: the
//! connection threads (`server.zig`, counted by `active_conn_threads`) and `downloads.zig`'s `RegistryUse` (counted by
//! `registryBusy`) are the waited-for ones. A fn of such a file that reads a global is an allowlist entry naming the
//! thread that runs it and what makes the wait cover it. Tests, and the fixtures after a file's first test (tests sit
//! last), are not scanned. Token scan, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "detached-global-read";
pub const reason = "a detached thread that reads a freed-at-shutdown global is joined by nothing: it needs a gate the shutdown waits for";

const globals = [_][]const u8{ "global_registry", "global_scheduler" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (rule.testFile(report.file) or std.mem.eql(u8, report.file, "src/server.zig")) return;
    const n = tree.tokens.len;
    var detaches = false;
    // A range whose start is past its end is undefined, so both scans count up: a file may hold one token, or open
    // with its test.
    var t: u32 = 1;
    while (t + 1 < n) : (t += 1) {
        if (tree.tokenTag(t) == .identifier and tree.tokenTag(t - 1) == .period and tree.tokenTag(t + 1) == .l_paren and std.mem.eql(u8, tree.tokenSlice(t), "detach")) detaches = true;
    }
    if (!detaches) return;
    var owners: ?std.ArrayList(rule.Owner) = null;
    var first_test: u32 = @intCast(n);
    for (0..n) |i| if (tree.tokenTag(@intCast(i)) == .keyword_test) {
        first_test = @intCast(i);
        break;
    };
    t = 2;
    while (t < first_test) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or tree.tokenTag(t - 1) != .period or tree.tokenTag(t - 2) != .identifier) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(t - 2), "server_state")) continue;
        const field = tree.tokenSlice(t);
        const hit = for (globals) |g| {
            if (std.mem.eql(u8, field, g)) break true;
        } else false;
        if (!hit) continue;
        if (owners == null) owners = try rule.owners(arena, tree);
        if (rule.inTest(owners.?.items, t)) continue;
        const key = if (rule.ownerOf(owners.?.items, t)) |o| o.key else "-";
        try report.add(tree, t, key, "server_state.{s} read in a file that detaches a thread: say which thread runs this and what shutdown waits for", .{field});
    }
}
