//! Rule `utf8-decode-raw`: a call of `std.unicode.utf8Decode` (or its `2`/`3`/`4`/`AllowSurrogateHalf` forms) outside
//! `src/utf8_seq.zig`. They are `unreachable` on a slice shorter than its lead byte announces, which a character cut
//! by a segment end gives; `utf8_seq.decode` answers null. Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "utf8-decode-raw";
pub const reason = "a raw utf8Decode on a cut character is unreachable (UB in ReleaseFast); use utf8_seq.decode";

const owner_file = "src/utf8_seq.zig";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.eql(u8, report.file, owner_file)) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 2;
    while (t < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or !std.mem.startsWith(u8, tree.tokenSlice(t), "utf8Decode")) continue;
        if (tree.tokenTag(t - 1) != .period or tree.tokenTag(t - 2) != .identifier) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(t - 2), "unicode")) continue;
        const owner = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
        try report.add(tree, t, owner, "std.unicode.{s}: decode with utf8_seq.decode, which a cut sequence cannot crash", .{tree.tokenSlice(t)});
    }
}
