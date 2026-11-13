//! The template tree. Every node keeps its source offset, for the line and column of an error.
const std = @import("std");
const builtins = @import("builtins.zig");

pub const Expr = struct {
    pos: u32,
    kind: Kind,

    pub const Kind = union(enum) {
        none,
        boolean: bool,
        int: i64,
        bigint: []const u8,
        float: f64,
        string: []const u8,
        name: []const u8,
        list: []const *const Expr,
        tuple: []const *const Expr,
        dict: []const Pair,
        getattr: struct { obj: *const Expr, name: []const u8 },
        getitem: struct { obj: *const Expr, key: *const Expr },
        slice: struct { obj: *const Expr, start: ?*const Expr, stop: ?*const Expr, step: ?*const Expr },
        call: struct { callee: *const Expr, args: Args },
        filter: struct { operand: ?*const Expr, filter: builtins.Filter, args: Args },
        @"test": struct { operand: *const Expr, test_: builtins.Test, args: Args },
        not_: *const Expr,
        neg: *const Expr,
        pos: *const Expr,
        binary: struct { op: BinOp, left: *const Expr, right: *const Expr },
        and_: struct { left: *const Expr, right: *const Expr },
        or_: struct { left: *const Expr, right: *const Expr },
        concat: []const *const Expr,
        compare: struct { first: *const Expr, ops: []const CompareOp },
        cond: struct { test_: *const Expr, yes: *const Expr, no: ?*const Expr },
    };
};

pub const Pair = struct { key: *const Expr, value: *const Expr };
pub const Keyword = struct { name: []const u8, value: *const Expr };
pub const Args = struct { positional: []const *const Expr = &.{}, keywords: []const Keyword = &.{} };

pub const BinOp = enum { add, sub, mul, div, mod };
pub const CmpOp = enum { eq, ne, lt, lteq, gt, gteq, in, notin };
pub const CompareOp = struct { op: CmpOp, expr: *const Expr };

pub const Param = struct { name: []const u8, default: ?*const Expr };

pub const Macro = struct {
    name: []const u8,
    params: []const Param,
    body: []const Node,
    /// The special names its body reads, as Jinja2 finds them: they decide what a call may pass.
    reads: Specials = .{},
};

/// `caller`, `varargs` and `kwargs` read by a macro body (a macro nested in it included).
pub const Specials = struct { caller: bool = false, varargs: bool = false, kwargs: bool = false };

/// A name, a tuple of targets (`a, (b, c)`), or a namespace attribute (`ns.x`, in `set` only).
pub const Target = union(enum) {
    name: []const u8,
    tuple: []const Target,
    attr: struct { namespace: []const u8, name: []const u8 },

    /// Whether the target stores into `name`.
    pub fn binds(self: Target, name: []const u8) bool {
        @setRuntimeSafety(true);
        return switch (self) {
            .name => |n| std.mem.eql(u8, n, name),
            .tuple => |ts| for (ts) |t| {
                if (t.binds(name)) break true;
            } else false,
            .attr => false,
        };
    }
};

pub const Node = struct {
    pos: u32,
    kind: Kind,

    pub const Kind = union(enum) {
        data: []const u8,
        output: *const Expr,
        if_: struct { branches: []const Branch, else_: []const Node },
        for_: struct { target: Target, iter: *const Expr, filter: ?*const Expr, body: []const Node, else_: []const Node },
        set: struct { target: Target, value: *const Expr },
        set_block: struct { target: Target, filter: ?*const Expr, body: []const Node },
        macro: *const Macro,
        call_block: struct { call: *const Expr, caller: *const Macro },
        filter_block: struct { filter: *const Expr, body: []const Node },
        /// transformers' `{% generation %}`: renders its body.
        generation: []const Node,
        break_,
        continue_,
    };
};

/// How a block's body stands to the frame around it, as Jinja2 compiles it: `inline_` runs in that frame (an `if`);
/// `scope` gets a frame of its own, so its `set`s stay in it, while a `break` still leaves the enclosing loop (a set
/// block, a filter block, a for loop's body); `call` is a macro body as well, which no `break` crosses (a macro, a
/// call block's caller, transformers' `generation`, itself a call block).
pub const Body = enum { inline_, scope, call };

pub fn bodyOf(kind: std.meta.Tag(Node.Kind)) Body {
    @setRuntimeSafety(true);
    return switch (kind) {
        .set_block, .filter_block, .for_ => .scope,
        .macro, .call_block, .generation => .call,
        else => .inline_,
    };
}

pub const Branch = struct {
    test_: *const Expr,
    body: []const Node,
    /// Source offsets: the `%}` that closes the branch's own tag, and the `{%` of the tag that ends its body.
    open_end: u32 = 0,
    close: u32 = 0,
};
