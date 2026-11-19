//! Rule `caught-error-unsettled`: in the files that run a request's work on the inference thread, a failure caught with
//! a capture (`catch |e|`) and named `@errorName(e)`. The failure may have left the MLX latch set (an engine maker's
//! `HostArrayRefused` leaves the pool's refusal latched, `check`'s `MlxError` the mlx-c message), and a latch left set
//! fails the next request's first checkpoint. `mlx.settledErrorName(e)` takes it and names the refusal behind it. The
//! error of a `checkError*` or `checkLatch` call is the latch it consumed and passes. Token scan, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "caught-error-unsettled";
pub const reason = "a caught failure may have left the MLX latch; @errorName leaves it set to fail the next request";

/// The files whose code runs a request's prefill, ticks, commits and encodes on the inference thread.
const files = [_][]const u8{ "src/scheduler.zig", "src/mtp_tick.zig", "src/slot_outcome.zig", "src/vision_queue.zig" };
const consuming = [_][]const u8{ "checkError", "checkErrorDecode", "checkLatch" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (!onFile(report.file)) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 0;
    while (t + 4 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .keyword_catch or tree.tokenTag(t + 1) != .pipe) continue;
        if (tree.tokenTag(t + 2) != .identifier or tree.tokenTag(t + 3) != .pipe) continue;
        if (consumesLatch(tree, t)) continue;
        const capture = tree.tokenSlice(t + 2);
        const end = handlerEnd(tree, t + 4);
        var k = t + 4;
        while (k + 3 <= end) : (k += 1) {
            if (tree.tokenTag(k) != .builtin or !std.mem.eql(u8, tree.tokenSlice(k), "@errorName")) continue;
            if (tree.tokenTag(k + 1) != .l_paren or tree.tokenTag(k + 2) != .identifier or tree.tokenTag(k + 3) != .r_paren) continue;
            if (!std.mem.eql(u8, tree.tokenSlice(k + 2), capture)) continue;
            if (rule.inTest(owners.items, k)) continue;
            const owner = if (rule.ownerOf(owners.items, k)) |o| o.key else "-";
            try report.add(tree, k, owner, "@errorName({s}) of a caught failure: name it with mlx.settledErrorName, which takes the latch it left", .{capture});
        }
    }
}

fn onFile(file: []const u8) bool {
    for (files) |f| if (std.mem.eql(u8, file, f)) return true;
    return false;
}

/// Whether the caught expression is a call that consumed the latch into its error (`mlx.checkErrorDecode() catch`).
fn consumesLatch(tree: *const Ast, catch_token: u32) bool {
    if (catch_token < 3 or tree.tokenTag(catch_token - 1) != .r_paren) return false;
    var depth: u32 = 0;
    var t = catch_token - 1;
    while (t > 0) : (t -= 1) {
        switch (tree.tokenTag(t)) {
            .r_paren => depth += 1,
            .l_paren => {
                depth -= 1;
                if (depth == 0) {
                    if (tree.tokenTag(t - 1) != .identifier) return false;
                    const callee = tree.tokenSlice(t - 1);
                    for (consuming) |c| if (std.mem.eql(u8, callee, c)) return true;
                    return false;
                }
            },
            else => {},
        }
    }
    return false;
}

/// The last token of the handler that starts at `start`: a block (after an optional label) to its closing brace, else
/// an expression to the end of its statement or argument.
fn handlerEnd(tree: *const Ast, start: u32) u32 {
    var t = start;
    if (tree.tokenTag(t) == .identifier and t + 1 < tree.tokens.len and tree.tokenTag(t + 1) == .colon) t += 2;
    var depth: u32 = 0;
    while (t < tree.tokens.len) : (t += 1) {
        switch (tree.tokenTag(t)) {
            .l_paren, .l_bracket, .l_brace => depth += 1,
            .r_paren, .r_bracket, .r_brace => {
                if (depth == 0) return t;
                depth -= 1;
            },
            .semicolon, .comma => if (depth == 0) return t,
            else => {},
        }
    }
    return t - 1;
}
