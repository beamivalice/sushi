//! Rule `entry-checkout`: a write of a hot-cache entry's `checkout` (an assignment, a struct field set, a pointer
//! capture, a write through `.?`) outside `prefix_cache_evict.zig`'s `beginCheckout`, `donateCheckout` and `endCheckout`.
//! A checkout is one slot's take of the entry's buffers; a caller that sets or clears a piece of it by hand leaves the
//! rest describing another take. Tests, and fns only tests reach, are exempt. Token scan.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "entry-checkout";
pub const reason = "an entry's checkout is begun, donated and ended by the hot cache's checkout calls alone";

const owners_allowed = [_][]const u8{ "beginCheckout", "donateCheckout", "endCheckout" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (rule.testFile(report.file)) return;
    const owner_file = std.mem.eql(u8, std.fs.path.basename(report.file), "prefix_cache_evict.zig");
    var owners: ?std.ArrayList(rule.Owner) = null;
    var test_only: []const bool = &.{};
    const n = tree.tokens.len;
    var t: u32 = 1;
    while (t + 1 < n) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or tree.tokenTag(t - 1) != .period) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(t), "checkout")) continue;
        if (!writes(tree, t)) continue;
        if (owners == null) {
            owners = try rule.owners(arena, tree);
            test_only = try rule.testOnly(arena, tree, owners.?.items);
        }
        const list = owners.?.items;
        if (rule.inTest(list, t)) continue;
        if (for (list, test_only) |o, m| {
            if (m and t >= o.first and t <= o.last) break true;
        } else false) continue;
        const key = if (rule.ownerOf(list, t)) |o| o.key else "-";
        if (owner_file and for (owners_allowed) |a| {
            if (std.mem.eql(u8, key, a)) break true;
        } else false) continue;
        try report.add(tree, t, key, "an entry's checkout written by hand: begin, donate or end it through the hot cache", .{});
    }
}

/// `.checkout =`, `.checkout) |*`, or `.checkout.?.<field> =`.
fn writes(tree: *const Ast, t: u32) bool {
    if (tagAt(tree, t + 1, .equal)) return true;
    if (tagAt(tree, t + 1, .r_paren) and tagAt(tree, t + 2, .pipe) and tagAt(tree, t + 3, .asterisk)) return true;
    return tagAt(tree, t + 1, .period) and tagAt(tree, t + 2, .question_mark) and tagAt(tree, t + 3, .period) and
        tagAt(tree, t + 4, .identifier) and tagAt(tree, t + 5, .equal);
}

fn tagAt(tree: *const Ast, i: usize, want: std.zig.Token.Tag) bool {
    return i < tree.tokens.len and tree.tokenTag(@intCast(i)) == want;
}
