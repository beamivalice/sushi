//! Rule `conn-read-unbounded`: a connection's reader reached outside `Conn.read` and `Conn.awaitReadable`. A read of
//! the reader that wants more than it holds blocks in the socket with no limit and no shutdown poll; a chunked upload
//! sent whole waited there forever. `Conn.read` is the one read. Tests are exempt. Token scan.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "conn-read-unbounded";
pub const reason = "a read of the connection's reader can block past the read limit and the shutdown; read through Conn.read";

const allowed = [_][]const u8{ "reader", "read", "awaitReadable" };

fn allowedOwner(key: []const u8) bool {
    for (allowed) |a| if (std.mem.eql(u8, key, a) or (std.mem.endsWith(u8, key, a) and key.len > a.len and key[key.len - a.len - 1] == '.')) return true;
    return false;
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    const base = std.fs.path.basename(report.file);
    const conn_file = std.mem.eql(u8, base, "http_conn.zig") or std.mem.eql(u8, base, "ws.zig");
    const owners = try rule.owners(arena, tree);
    var t: u32 = 1;
    while (t + 2 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or tree.tokenTag(t - 1) != .period) continue;
        const word = tree.tokenSlice(t);
        // `read_state.interface` anywhere; `.reader()` with no arguments in the connection files.
        const hit = if (std.mem.eql(u8, word, "read_state"))
            tree.tokenTag(t + 1) == .period and std.mem.eql(u8, tree.tokenSlice(t + 2), "interface")
        else
            conn_file and std.mem.eql(u8, word, "reader") and tree.tokenTag(t + 1) == .l_paren and tree.tokenTag(t + 2) == .r_paren;
        if (!hit) continue;
        const owner = rule.ownerOf(owners.items, t) orelse continue;
        if (owner.is_test or rule.inTest(owners.items, t) or allowedOwner(owner.key)) continue;
        try report.add(tree, t, owner.key, "the connection's reader outside Conn.read: read through Conn.read", .{});
    }
}
