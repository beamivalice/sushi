//! Rule `std-xxhash3`: `std.hash.XxHash3` named anywhere. Its long-input loop runs at about 1.3 GB/s on Apple Silicon;
//! `src/xxh3.zig` gives the same values at about 30. Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "std-xxhash3";
pub const reason = "std.hash.XxHash3 hashes bulk bytes about 25 times slower than xxh3.zig, which gives the same values";

pub fn check(_: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    var t: u32 = 4;
    while (t < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or !std.mem.eql(u8, tree.tokenSlice(t), "XxHash3")) continue;
        if (tree.tokenTag(t - 1) != .period or tree.tokenTag(t - 3) != .period) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(t - 2), "hash") or !std.mem.eql(u8, tree.tokenSlice(t - 4), "std")) continue;
        try report.add(tree, t, "-", "std.hash.XxHash3: hash with xxh3.zig", .{});
    }
}
