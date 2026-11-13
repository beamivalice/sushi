//! Rule `device-probe`: a call that takes the GPU's architecture name, the OS version or the QA switch by name, outside
//! `src/exl3/device.zig`. The device profile reads each once; a second reader is a second parse, cache and test seam
//! of the same fact, and they have disagreed before. Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "device-probe";
pub const reason = "the GPU's name, the OS version and the QA switch are read by the device profile alone: ask device.current()";

const owner_file = "/exl3/device.zig";
const probes = [_][]const u8{ "\"SUSHI_FORCE_GPU_FAMILY_FALLBACK\"", "\"MLX_METAL_GPU_ARCH\"", "\"kern.osproductversion\"", "\"architecture\"" };

/// The token is an argument of a call: the nearest opener that encloses it is a parenthesis, not a brace or a bracket.
fn inCall(tree: *const Ast, token: u32) bool {
    var depth: u32 = 0;
    var t = token;
    while (t > 0) {
        t -= 1;
        switch (tree.tokenTag(t)) {
            .r_paren, .r_brace, .r_bracket => depth += 1,
            .l_paren, .l_brace, .l_bracket => |tag| {
                if (depth == 0) return tag == .l_paren;
                depth -= 1;
            },
            else => {},
        }
    }
    return false;
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.endsWith(u8, report.file, owner_file)) return;
    var owners: ?std.ArrayList(rule.Owner) = null;
    for (0..tree.tokens.len) |i| {
        const t: u32 = @intCast(i);
        if (tree.tokenTag(t) != .string_literal) continue;
        const text = tree.tokenSlice(t);
        for (probes) |probe| {
            if (!std.mem.eql(u8, text, probe) or !inCall(tree, t)) continue;
            if (owners == null) owners = try rule.owners(arena, tree);
            const key = if (rule.ownerOf(owners.?.items, t)) |o| o.key else "-";
            try report.add(tree, t, key, "{s} is read here: the device profile reads it once, ask device.current()", .{probe});
        }
    }
}
