//! Rule `tag-lead-named`: `tag_lead` named outside `think_tags.zig`. A reply's think tag tokens carry that lead in its
//! text; only `think_tags` writes it and takes it off (`withoutLead`, called by the JSON escapers and the call reader), so
//! a file that handled the lead itself is a second exit that can miss a byte. Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "tag-lead-named";
pub const reason = "the tag lead is written and dropped only in think_tags.zig; every client-bound string leaves through think_tags.withoutLead";

pub fn check(_: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.eql(u8, std.fs.path.basename(report.file), "think_tags.zig")) return;
    var t: u32 = 0;
    while (t < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or !std.mem.eql(u8, tree.tokenSlice(t), "tag_lead")) continue;
        try report.add(tree, t, "-", "tag_lead: drop a lead with think_tags.withoutLead, never name it here", .{});
    }
}
