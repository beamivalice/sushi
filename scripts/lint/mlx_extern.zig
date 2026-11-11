//! Rule `mlx-extern`: an `extern` fn or var named `mlx_*` or `_mlx_*`, or an `@extern` of such a name, anywhere but
//! `src/mlx.zig`. Every call into mlx-c passes the counter there (`mlx.calls`); a declaration elsewhere would reach
//! mlx-c unseen. Parse only, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const Node = Ast.Node;
const rule = @import("rule.zig");

pub const name = "mlx-extern";
pub const reason = "a declaration of an mlx-c symbol outside src/mlx.zig reaches mlx-c without passing the call counter";

const seam = "src/mlx.zig";

fn isMlxSymbol(symbol: []const u8) bool {
    return std.mem.startsWith(u8, symbol, "mlx_") or std.mem.startsWith(u8, symbol, "_mlx_");
}

/// The quoted contents of a string-literal token, or null.
fn stringBody(tree: *const Ast, token: u32) ?[]const u8 {
    if (tree.tokenTag(token) != .string_literal) return null;
    const slice = tree.tokenSlice(token);
    return slice[1 .. slice.len - 1];
}

pub fn check(_: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.eql(u8, report.file, seam)) return;
    var i: u32 = 0;
    while (i < tree.nodes.len) : (i += 1) {
        const node: Node.Index = @enumFromInt(i);
        switch (tree.nodeTag(node)) {
            .fn_proto, .fn_proto_one, .fn_proto_multi, .fn_proto_simple => {
                var buf: [1]Node.Index = undefined;
                const proto = tree.fullFnProto(&buf, node) orelse continue;
                const extern_token = proto.extern_export_inline_token orelse continue;
                if (tree.tokenTag(extern_token) != .keyword_extern) continue;
                const name_token = proto.name_token orelse continue;
                const symbol = tree.tokenSlice(name_token);
                if (isMlxSymbol(symbol)) try report.add(tree, name_token, symbol, "extern fn '{s}': declare it in src/mlx.zig", .{symbol});
            },
            .global_var_decl, .local_var_decl, .simple_var_decl, .aligned_var_decl => {
                const decl = tree.fullVarDecl(node) orelse continue;
                const extern_token = decl.extern_export_token orelse continue;
                if (tree.tokenTag(extern_token) != .keyword_extern) continue;
                const symbol = tree.tokenSlice(decl.ast.mut_token + 1);
                if (isMlxSymbol(symbol)) try report.add(tree, decl.ast.mut_token + 1, symbol, "extern var '{s}': declare it in src/mlx.zig", .{symbol});
            },
            .builtin_call_two, .builtin_call_two_comma, .builtin_call, .builtin_call_comma => {
                const first = tree.nodeMainToken(node);
                if (!std.mem.eql(u8, tree.tokenSlice(first), "@extern")) continue;
                const last = tree.lastToken(node);
                var t = first;
                while (t <= last) : (t += 1) {
                    const body = stringBody(tree, t) orelse continue;
                    if (isMlxSymbol(body)) try report.add(tree, t, body, "@extern of '{s}': declare it in src/mlx.zig", .{body});
                }
            },
            else => {},
        }
    }
}
