//! Rule `host-maker-unchecked`: a raw host maker (`mlx_array_new_data`, `_int`, `_float`, `_bool`) on code a Cooler
//! load runs, whose result is not handed straight to `made(..)` or a side cache's `put(..)`. MLX's maker never fails;
//! the engine's hands back an empty handle when its pool refuses the buffer, and an unchecked one ends in a crash or in
//! a call to MLX at rest. Use `mlx.newData`/`newInt`/`newFloat`/`newBool` (`host_side.made` in the EXL3 module). The
//! archs that never run on the engine, the seam itself, tests and fns only tests reach are exempt. Token scan, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "host-maker-unchecked";
pub const reason = "an engine's host maker can be refused; an unchecked empty handle crashes or reaches MLX";

const makers = [_][]const u8{ "mlx_array_new_data", "mlx_array_new_int", "mlx_array_new_float", "mlx_array_new_bool" };
const checked_by = [_][]const u8{ "made", "put" };
/// Code a Cooler load runs: the archs on the engine, the EXL3 module and the shared files at `src/` and `src/arch/`.
const cooler_dirs = [_][]const u8{ "src/qwen4/", "src/dsv41/", "src/exl3/", "src/arch/" };
const exempt = [_][]const u8{ "src/mlx.zig", "src/test_support.zig", "src/ubench_shared.zig" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (!onCoolerPath(report.file) or rule.testFile(report.file)) return;
    for (exempt) |f| if (std.mem.eql(u8, report.file, f)) return;
    const owners = try rule.owners(arena, tree);
    const test_only = try rule.testOnly(arena, tree, owners.items);
    var t: u32 = 1;
    while (t + 1 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or tree.tokenTag(t - 1) != .period or tree.tokenTag(t + 1) != .l_paren) continue;
        const what = tree.tokenSlice(t);
        if (!isMaker(what)) continue;
        const owner = rule.ownerOf(owners.items, t) orelse continue;
        if (owner.is_test or inTestOnly(owners.items, test_only, t)) continue;
        if (enclosingCall(tree, t - 1)) |outer| if (isChecker(outer)) continue;
        try report.add(tree, t, owner.key, "{s} unchecked: an engine's pool may refuse it, use the mlx.new* maker", .{what});
    }
}

/// Whether token `t` sits in a fn only tests reach (`rule.testOnly`).
fn inTestOnly(list: []const rule.Owner, test_only: []const bool, t: u32) bool {
    for (list, test_only) |o, m| if (m and t >= o.first and t <= o.last) return true;
    return false;
}

fn onCoolerPath(file: []const u8) bool {
    for (cooler_dirs) |d| if (std.mem.startsWith(u8, file, d)) return true;
    // A file directly under `src/`: the shared machinery.
    return std.mem.startsWith(u8, file, "src/") and std.mem.indexOfScalar(u8, file["src/".len..], '/') == null;
}

fn isMaker(what: []const u8) bool {
    for (makers) |m| if (std.mem.eql(u8, what, m)) return true;
    return false;
}

fn isChecker(what: []const u8) bool {
    for (checked_by) |c| if (std.mem.eql(u8, what, c)) return true;
    return false;
}

/// The name of the call whose argument list holds token `at`, if it is in the same statement. A braced argument (`{}`,
/// an anonymous literal) is stepped over whole.
fn enclosingCall(tree: *const Ast, at: u32) ?[]const u8 {
    var depth: u32 = 0;
    var braces: u32 = 0;
    var t = at;
    while (t > 0) {
        t -= 1;
        switch (tree.tokenTag(t)) {
            .r_brace => braces += 1,
            .l_brace => {
                if (braces == 0) return null;
                braces -= 1;
            },
            .r_paren => depth += 1,
            .l_paren => {
                if (depth == 0) return if (t > 0 and tree.tokenTag(t - 1) == .identifier) tree.tokenSlice(t - 1) else null;
                depth -= 1;
            },
            .semicolon => return null,
            else => {},
        }
    }
    return null;
}
