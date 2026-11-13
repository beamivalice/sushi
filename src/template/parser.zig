//! Jinja2's grammar for the statements and expressions jinja.cpp supports; anything else is a syntax error that
//! names it. Precedence as Jinja2: `if`-else < or < and < not < comparisons and `in` < `+ -` < `~` < `* / %` <
//! unary `- +` < postfix (`.x`, `[i]`, calls) < filters and tests.
const std = @import("std");
const ast = @import("ast.zig");
const lexer = @import("lexer.zig");
const builtins = @import("builtins.zig");
const value = @import("value.zig");

const Allocator = std.mem.Allocator;
const Token = lexer.Token;
const Expr = ast.Expr;
const Node = ast.Node;

pub const Error = lexer.Error;

/// Deepest nesting of expressions and statements a template may use; each link of a left-associative chain
/// (`a + b + c`, `x.a.b`, `x|f|g`) nests one deeper, as the tree it builds does.
pub const max_nesting = 200;

/// The work a compile may do, beside the render's `Limits.max_steps`: a token taken or a pair of names compared is a
/// step. A template of `template.max_template_len` takes far fewer; past it the compile fails by name.
pub const max_compile_steps = 64_000_000;

const unsupported_statements = [_][]const u8{ "include", "import", "from", "extends", "block", "with", "do", "autoescape", "trans", "print", "raw", "endraw" };

pub const Parser = struct {
    arena: Allocator,
    tokens: []const Token,
    i: usize = 0,
    err: *lexer.SyntaxError,
    depth: usize = 0,
    /// Inside a for loop body, not across a macro: `break` and `continue` are allowed.
    in_loop: bool = false,
    /// Inside a for loop's body or else, across macros too: a target may not be `loop`.
    in_for: bool = false,
    /// The special names read since the innermost macro body began.
    specials: ast.Specials = .{},
    /// The compile's steps (`max_compile_steps`), checked wherever the parser nests.
    meter: value.Meter = .{ .max = max_compile_steps },

    fn fail(self: *Parser, pos: u32, comptime fmt: []const u8, args: anytype) Error {
        @setRuntimeSafety(true);
        self.err.* = .{ .pos = pos, .message = std.fmt.allocPrint(self.arena, fmt, args) catch "out of memory" };
        return error.TemplateSyntax;
    }

    fn cur(self: *const Parser) Token {
        @setRuntimeSafety(true);
        return self.tokens[self.i];
    }

    fn peekAt(self: *const Parser, n: usize) Token {
        @setRuntimeSafety(true);
        return self.tokens[@min(self.i + n, self.tokens.len - 1)];
    }

    fn advance(self: *Parser) Token {
        @setRuntimeSafety(true);
        self.meter.used +|= 1;
        const t = self.tokens[self.i];
        if (t.tag != .eof) self.i += 1;
        return t;
    }

    fn isOp(self: *const Parser, op: []const u8) bool {
        @setRuntimeSafety(true);
        const t = self.cur();
        return t.tag == .op and std.mem.eql(u8, t.text, op);
    }

    fn isName(self: *const Parser, name: []const u8) bool {
        @setRuntimeSafety(true);
        const t = self.cur();
        return t.tag == .name and std.mem.eql(u8, t.text, name);
    }

    fn skipOp(self: *Parser, op: []const u8) bool {
        @setRuntimeSafety(true);
        if (!self.isOp(op)) return false;
        _ = self.advance();
        return true;
    }

    fn skipName(self: *Parser, name: []const u8) bool {
        @setRuntimeSafety(true);
        if (!self.isName(name)) return false;
        _ = self.advance();
        return true;
    }

    fn describe(t: Token) []const u8 {
        @setRuntimeSafety(true);
        return switch (t.tag) {
            .eof => "end of template",
            .block_end => "end of statement block",
            .var_end => "end of print statement",
            .data => "template data",
            else => t.text,
        };
    }

    fn expectOp(self: *Parser, op: []const u8) Error!void {
        @setRuntimeSafety(true);
        if (!self.skipOp(op)) return self.fail(self.cur().pos, "expected '{s}', got '{s}'", .{ op, describe(self.cur()) });
    }

    fn expectTag(self: *Parser, tag: lexer.Tag, what: []const u8) Error!void {
        @setRuntimeSafety(true);
        if (self.cur().tag != tag) return self.fail(self.cur().pos, "expected {s}, got '{s}'", .{ what, describe(self.cur()) });
        _ = self.advance();
    }

    fn expectName(self: *Parser) Error![]const u8 {
        @setRuntimeSafety(true);
        if (self.cur().tag != .name) return self.fail(self.cur().pos, "expected a name, got '{s}'", .{describe(self.cur())});
        return self.advance().text;
    }

    fn enter(self: *Parser) Error!void {
        @setRuntimeSafety(true);
        self.depth += 1;
        if (self.depth > max_nesting) return self.fail(self.cur().pos, "the template nests deeper than {d}", .{max_nesting});
        try self.checkSteps();
    }

    fn checkSteps(self: *Parser) Error!void {
        @setRuntimeSafety(true);
        if (self.meter.spent()) return self.fail(self.cur().pos, "the template takes more than {d} steps to compile", .{self.meter.max});
    }

    fn mk(self: *Parser, pos: u32, kind: Expr.Kind) Error!*const Expr {
        @setRuntimeSafety(true);
        const e = try self.arena.create(Expr);
        e.* = .{ .pos = pos, .kind = kind };
        return e;
    }

    // ── Statements ──

    /// Nodes until a `{%` whose statement name is one of `ends` (that name is the current token), or the end.
    pub fn subparse(self: *Parser, ends: []const []const u8) Error![]const Node {
        @setRuntimeSafety(true);
        try self.enter();
        defer self.depth -= 1;
        var body = std.ArrayList(Node).empty;
        while (true) {
            const t = self.cur();
            switch (t.tag) {
                .eof => {
                    if (ends.len > 0) return self.fail(t.pos, "unexpected end of template, expected '{s}'", .{ends[ends.len - 1]});
                    return body.items;
                },
                .data => {
                    _ = self.advance();
                    try body.append(self.arena, .{ .pos = t.pos, .kind = .{ .data = t.text } });
                },
                .var_begin => {
                    _ = self.advance();
                    const e = try self.tuple(.{});
                    try self.expectTag(.var_end, "'}}'");
                    try body.append(self.arena, .{ .pos = t.pos, .kind = .{ .output = e } });
                },
                .block_begin => {
                    _ = self.advance();
                    for (ends) |end| if (self.isName(end)) return body.items;
                    try body.append(self.arena, try self.statement(t.pos));
                    try self.expectTag(.block_end, "'%}'");
                },
                else => return self.fail(t.pos, "unexpected '{s}'", .{describe(t)}),
            }
        }
    }

    /// The body after a statement's own `%}`, through the end name (consumed) that closes it.
    fn block(self: *Parser, ends: []const []const u8) Error![]const Node {
        @setRuntimeSafety(true);
        try self.expectTag(.block_end, "'%}'");
        return self.subparse(ends);
    }

    fn statement(self: *Parser, pos: u32) Error!Node {
        @setRuntimeSafety(true);
        const name_tok = self.cur();
        if (name_tok.tag != .name) return self.fail(name_tok.pos, "tag name expected", .{});
        const name = name_tok.text;
        _ = self.advance();
        if (std.mem.eql(u8, name, "if")) return self.ifStatement(pos);
        if (std.mem.eql(u8, name, "for")) return self.forStatement(pos);
        if (std.mem.eql(u8, name, "set")) return self.setStatement(pos);
        if (std.mem.eql(u8, name, "macro")) return self.macroStatement(pos);
        if (std.mem.eql(u8, name, "call")) return self.callBlock(pos);
        if (std.mem.eql(u8, name, "filter")) return self.filterBlock(pos);
        if (std.mem.eql(u8, name, "generation")) {
            const outer = self.in_loop;
            self.in_loop = outer and ast.bodyOf(.generation) != .call;
            const body = try self.block(&.{"endgeneration"});
            self.in_loop = outer;
            _ = self.advance();
            return .{ .pos = pos, .kind = .{ .generation = body } };
        }
        if (std.mem.eql(u8, name, "break") or std.mem.eql(u8, name, "continue")) {
            if (!self.in_loop) return self.fail(name_tok.pos, "'{s}' outside a for loop", .{name});
            return .{ .pos = pos, .kind = if (name[0] == 'b') .break_ else .continue_ };
        }
        for (unsupported_statements) |u| if (std.mem.eql(u8, name, u))
            return self.fail(name_tok.pos, "unsupported statement '{s}'", .{name});
        return self.fail(name_tok.pos, "unknown statement '{s}'", .{name});
    }

    fn ifStatement(self: *Parser, pos: u32) Error!Node {
        @setRuntimeSafety(true);
        var branches = std.ArrayList(ast.Branch).empty;
        var else_: []const Node = &.{};
        while (true) {
            const test_ = try self.tuple(.{ .condexpr = false });
            const open_end = self.cur().pos;
            const body = try self.block(&.{ "elif", "else", "endif" });
            try branches.append(self.arena, .{ .test_ = test_, .body = body, .open_end = open_end, .close = self.tokens[self.i - 1].pos });
            const end = self.advance().text;
            if (std.mem.eql(u8, end, "elif")) continue;
            if (std.mem.eql(u8, end, "else")) {
                else_ = try self.block(&.{"endif"});
                _ = self.advance();
            }
            break;
        }
        return .{ .pos = pos, .kind = .{ .if_ = .{ .branches = branches.items, .else_ = else_ } } };
    }

    fn forStatement(self: *Parser, pos: u32) Error!Node {
        @setRuntimeSafety(true);
        const target = try self.assignTarget(false);
        if (target.binds("loop")) return self.fail(pos, "Can't assign to special loop variable in for-loop target", .{});
        if (!self.skipName("in")) return self.fail(self.cur().pos, "expected 'in', got '{s}'", .{describe(self.cur())});
        const iter = try self.tuple(.{ .condexpr = false, .end_name = "recursive" });
        const filter = if (self.skipName("if")) try self.expression(true) else null;
        if (self.isName("recursive")) return self.fail(self.cur().pos, "unsupported 'recursive' loop", .{});
        const outer = self.in_loop;
        const outer_for = self.in_for;
        defer self.in_for = outer_for;
        self.in_for = true;
        self.in_loop = true;
        const body = try self.block(&.{ "endfor", "else" });
        self.in_loop = outer;
        var else_: []const Node = &.{};
        if (std.mem.eql(u8, self.advance().text, "else")) {
            else_ = try self.block(&.{"endfor"});
            _ = self.advance();
        }
        return .{ .pos = pos, .kind = .{ .for_ = .{ .target = target, .iter = iter, .filter = filter, .body = body, .else_ = else_ } } };
    }

    fn setStatement(self: *Parser, pos: u32) Error!Node {
        @setRuntimeSafety(true);
        const target = try self.assignTarget(true);
        if (self.in_for and target.binds("loop")) return self.fail(pos, "Can't assign to special loop variable in for-loop target", .{});
        if (self.skipOp("=")) return .{ .pos = pos, .kind = .{ .set = .{ .target = target, .value = try self.tuple(.{}) } } };
        const filter = if (self.isOp("|")) try self.filterChain(null, false) else null;
        const body = try self.block(&.{"endset"});
        _ = self.advance();
        return .{ .pos = pos, .kind = .{ .set_block = .{ .target = target, .filter = filter, .body = body } } };
    }

    fn assignTarget(self: *Parser, with_namespace: bool) Error!ast.Target {
        @setRuntimeSafety(true);
        const start = self.cur();
        if (with_namespace and start.tag == .name and self.peekAt(1).tag == .op and std.mem.eql(u8, self.peekAt(1).text, ".")) {
            _ = self.advance();
            _ = self.advance();
            return .{ .attr = .{ .namespace = start.text, .name = try self.expectName() } };
        }
        return self.targetTuple(false);
    }

    /// Jinja2's simplified tuple: targets split by commas. A trailing comma or an empty tuple needs parentheses, and one
    /// target in parentheses is that target (`(a)` is `a`).
    fn targetTuple(self: *Parser, parens: bool) Error!ast.Target {
        @setRuntimeSafety(true);
        try self.enter();
        defer self.depth -= 1;
        var items = std.ArrayList(ast.Target).empty;
        var is_tuple = false;
        while (true) {
            if (items.items.len > 0) try self.expectOp(",");
            if (parens and self.isOp(")")) break;
            try items.append(self.arena, try self.targetPrimary());
            if (!self.isOp(",")) break;
            is_tuple = true;
        }
        if (!is_tuple and items.items.len == 1) return items.items[0];
        return .{ .tuple = items.items };
    }

    fn targetPrimary(self: *Parser) Error!ast.Target {
        @setRuntimeSafety(true);
        if (self.skipOp("(")) {
            const t = try self.targetTuple(true);
            try self.expectOp(")");
            return t;
        }
        const t = self.cur();
        if (t.tag != .name or isKeyword(t.text)) return self.fail(t.pos, "can't assign to '{s}'", .{describe(t)});
        _ = self.advance();
        return .{ .name = t.text };
    }

    fn isKeyword(name: []const u8) bool {
        @setRuntimeSafety(true);
        for ([_][]const u8{ "true", "false", "True", "False", "none", "None" }) |k| if (std.mem.eql(u8, name, k)) return true;
        return false;
    }

    fn signature(self: *Parser) Error![]const ast.Param {
        @setRuntimeSafety(true);
        var params = std.ArrayList(ast.Param).empty;
        var names = std.ArrayList(Named).empty;
        try self.expectOp("(");
        var defaults = false;
        while (!self.isOp(")")) {
            if (params.items.len > 0) try self.expectOp(",");
            if (self.isOp(")")) break;
            const at = self.cur().pos;
            const name = try self.expectName();
            var default: ?*const Expr = null;
            if (self.skipOp("=")) {
                default = try self.expression(true);
                defaults = true;
            } else if (defaults) return self.fail(self.cur().pos, "non-default argument follows default argument", .{});
            try params.append(self.arena, .{ .name = name, .default = default });
            try names.append(self.arena, .{ .name = name, .pos = at });
        }
        try self.expectOp(")");
        if (try self.firstRepeat(names.items)) |dup| return self.fail(dup.pos, "duplicate argument '{s}' in function definition", .{dup.name});
        return params.items;
    }

    const Named = struct { name: []const u8, pos: u32 };

    /// The first name of `names` that an earlier one already has, found by a sort: a compile reads a pack's own
    /// template, whose names a scan per name would compare in their count squared.
    fn firstRepeat(self: *Parser, names: []const Named) Error!?Named {
        @setRuntimeSafety(true);
        if (names.len < 2) return null;
        const sorted = try self.arena.dupe(Named, names);
        const Ctx = struct {
            meter: *value.Meter,
            fn less(c: *@This(), a: Named, b: Named) value.Failure!bool {
                @setRuntimeSafety(true);
                try c.meter.tick(1 + @min(a.name.len, b.name.len) / 64);
                return std.mem.order(u8, a.name, b.name) == .lt;
            }
        };
        var ctx = Ctx{ .meter = &self.meter };
        value.stableSort(Named, sorted, try self.arena.alloc(Named, sorted.len), &ctx, Ctx.less) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return self.fail(self.cur().pos, "the template takes more than {d} steps to compile", .{self.meter.max}),
        };
        // Equal names keep their order, so each run's second is that name's first repeat.
        var first: ?Named = null;
        for (sorted[1..], sorted[0 .. sorted.len - 1]) |b, a| {
            if (!std.mem.eql(u8, a.name, b.name)) continue;
            if (first == null or b.pos < first.?.pos) first = b;
        }
        return first;
    }

    fn macroBody(self: *Parser, pos: u32, name: []const u8, params: []const ast.Param, end: []const u8) Error!*const ast.Macro {
        @setRuntimeSafety(true);
        const outer = self.in_loop;
        const outer_specials = self.specials;
        self.in_loop = outer and ast.bodyOf(.macro) != .call;
        self.specials = .{};
        const body = try self.block(&.{end});
        const reads = self.specials;
        // A `caller` parameter shadows the caller the body reads, so it must have a default to be called without one.
        if (reads.caller) for (params) |p| if (p.default == null and std.mem.eql(u8, p.name, "caller"))
            return self.fail(pos, "When defining macros or call blocks the special \"caller\" argument must be omitted or be given a default.", .{});
        self.in_loop = outer;
        self.specials = .{
            .caller = outer_specials.caller or reads.caller,
            .varargs = outer_specials.varargs or reads.varargs,
            .kwargs = outer_specials.kwargs or reads.kwargs,
        };
        _ = self.advance();
        const m = try self.arena.create(ast.Macro);
        m.* = .{ .name = name, .params = params, .body = body, .reads = reads };
        return m;
    }

    fn macroStatement(self: *Parser, pos: u32) Error!Node {
        @setRuntimeSafety(true);
        const name = try self.expectName();
        const params = try self.signature();
        return .{ .pos = pos, .kind = .{ .macro = try self.macroBody(pos, name, params, "endmacro") } };
    }

    fn callBlock(self: *Parser, pos: u32) Error!Node {
        @setRuntimeSafety(true);
        const params: []const ast.Param = if (self.isOp("(")) try self.signature() else &.{};
        const call = try self.expression(true);
        if (call.kind != .call) return self.fail(call.pos, "expected a call", .{});
        return .{ .pos = pos, .kind = .{ .call_block = .{ .call = call, .caller = try self.macroBody(pos, "caller", params, "endcall") } } };
    }

    fn filterBlock(self: *Parser, pos: u32) Error!Node {
        @setRuntimeSafety(true);
        const filter = try self.filterChain(null, true);
        const body = try self.block(&.{"endfilter"});
        _ = self.advance();
        return .{ .pos = pos, .kind = .{ .filter_block = .{ .filter = filter, .body = body } } };
    }

    // ── Expressions ──

    const TupleOpts = struct { condexpr: bool = true, explicit_parens: bool = false, end_name: ?[]const u8 = null };

    fn tupleEnd(self: *const Parser, opts: TupleOpts) bool {
        @setRuntimeSafety(true);
        const t = self.cur();
        if (t.tag == .var_end or t.tag == .block_end or t.tag == .eof) return true;
        if (t.tag == .op and std.mem.eql(u8, t.text, ")")) return true;
        if (opts.end_name) |n| if (self.isName(n)) return true;
        return false;
    }

    fn tuple(self: *Parser, opts: TupleOpts) Error!*const Expr {
        @setRuntimeSafety(true);
        const pos = self.cur().pos;
        var items = std.ArrayList(*const Expr).empty;
        var is_tuple = false;
        while (true) {
            if (items.items.len > 0) try self.expectOp(",");
            if (self.tupleEnd(opts)) break;
            try items.append(self.arena, try self.expression(opts.condexpr));
            if (self.isOp(",")) is_tuple = true else break;
        }
        if (!is_tuple) {
            if (items.items.len > 0) return items.items[0];
            if (!opts.explicit_parens) return self.fail(self.cur().pos, "expected an expression, got '{s}'", .{describe(self.cur())});
        }
        return self.mk(pos, .{ .tuple = items.items });
    }

    pub fn expression(self: *Parser, condexpr: bool) Error!*const Expr {
        @setRuntimeSafety(true);
        try self.enter();
        defer self.depth -= 1;
        if (!condexpr) return self.orExpr();
        var e = try self.orExpr();
        const base = self.depth;
        defer self.depth = base;
        while (self.isName("if")) {
            try self.enter();
            const pos = self.advance().pos;
            const test_ = try self.orExpr();
            const no = if (self.skipName("else")) try self.expression(true) else null;
            e = try self.mk(pos, .{ .cond = .{ .test_ = test_, .yes = e, .no = no } });
        }
        return e;
    }

    fn orExpr(self: *Parser) Error!*const Expr {
        @setRuntimeSafety(true);
        var left = try self.andExpr();
        const base = self.depth;
        defer self.depth = base;
        while (self.isName("or")) {
            try self.enter();
            const pos = self.advance().pos;
            left = try self.mk(pos, .{ .or_ = .{ .left = left, .right = try self.andExpr() } });
        }
        return left;
    }

    fn andExpr(self: *Parser) Error!*const Expr {
        @setRuntimeSafety(true);
        var left = try self.notExpr();
        const base = self.depth;
        defer self.depth = base;
        while (self.isName("and")) {
            try self.enter();
            const pos = self.advance().pos;
            left = try self.mk(pos, .{ .and_ = .{ .left = left, .right = try self.notExpr() } });
        }
        return left;
    }

    fn notExpr(self: *Parser) Error!*const Expr {
        @setRuntimeSafety(true);
        if (self.isName("not")) {
            const pos = self.advance().pos;
            try self.enter();
            defer self.depth -= 1;
            return self.mk(pos, .{ .not_ = try self.notExpr() });
        }
        return self.compare();
    }

    fn compare(self: *Parser) Error!*const Expr {
        @setRuntimeSafety(true);
        const first = try self.math1();
        var ops = std.ArrayList(ast.CompareOp).empty;
        while (true) {
            const t = self.cur();
            const op: ?ast.CmpOp = if (t.tag == .op) blk: {
                for ([_]struct { s: []const u8, op: ast.CmpOp }{
                    .{ .s = "==", .op = .eq }, .{ .s = "!=", .op = .ne }, .{ .s = "<", .op = .lt },
                    .{ .s = "<=", .op = .lteq }, .{ .s = ">", .op = .gt }, .{ .s = ">=", .op = .gteq },
                }) |m| if (std.mem.eql(u8, t.text, m.s)) break :blk m.op;
                break :blk null;
            } else if (self.isName("in")) .in else if (self.isName("not") and self.peekAt(1).tag == .name and std.mem.eql(u8, self.peekAt(1).text, "in")) .notin else null;
            const o = op orelse break;
            _ = self.advance();
            if (o == .notin) _ = self.advance();
            try ops.append(self.arena, .{ .op = o, .expr = try self.math1() });
        }
        if (ops.items.len == 0) return first;
        return self.mk(first.pos, .{ .compare = .{ .first = first, .ops = ops.items } });
    }

    fn math1(self: *Parser) Error!*const Expr {
        @setRuntimeSafety(true);
        var left = try self.concat();
        const base = self.depth;
        defer self.depth = base;
        while (self.isOp("+") or self.isOp("-")) {
            try self.enter();
            const t = self.advance();
            left = try self.mk(t.pos, .{ .binary = .{ .op = if (t.text[0] == '+') .add else .sub, .left = left, .right = try self.concat() } });
        }
        return left;
    }

    fn concat(self: *Parser) Error!*const Expr {
        @setRuntimeSafety(true);
        const first = try self.math2();
        if (!self.isOp("~")) return first;
        var parts = std.ArrayList(*const Expr).empty;
        try parts.append(self.arena, first);
        while (self.skipOp("~")) try parts.append(self.arena, try self.math2());
        return self.mk(first.pos, .{ .concat = parts.items });
    }

    fn math2(self: *Parser) Error!*const Expr {
        @setRuntimeSafety(true);
        var left = try self.pow();
        const base = self.depth;
        defer self.depth = base;
        while (true) {
            const t = self.cur();
            if (t.tag != .op) break;
            const op: ast.BinOp = if (std.mem.eql(u8, t.text, "*")) .mul else if (std.mem.eql(u8, t.text, "/")) .div else if (std.mem.eql(u8, t.text, "%")) .mod else if (std.mem.eql(u8, t.text, "//"))
                return self.fail(t.pos, "unsupported operator '//'", .{})
            else
                break;
            try self.enter();
            _ = self.advance();
            left = try self.mk(t.pos, .{ .binary = .{ .op = op, .left = left, .right = try self.pow() } });
        }
        return left;
    }

    fn pow(self: *Parser) Error!*const Expr {
        @setRuntimeSafety(true);
        const left = try self.unary(true);
        if (self.isOp("**")) return self.fail(self.cur().pos, "unsupported operator '**'", .{});
        return left;
    }

    fn unary(self: *Parser, with_filter: bool) Error!*const Expr {
        @setRuntimeSafety(true);
        try self.enter();
        defer self.depth -= 1;
        const t = self.cur();
        var node = if (self.isOp("-")) blk: {
            _ = self.advance();
            break :blk try self.mk(t.pos, .{ .neg = try self.unary(false) });
        } else if (self.isOp("+")) blk: {
            _ = self.advance();
            break :blk try self.mk(t.pos, .{ .pos = try self.unary(false) });
        } else try self.primary();
        node = try self.postfix(node);
        if (with_filter) node = try self.filterExpr(node);
        return node;
    }

    fn primary(self: *Parser) Error!*const Expr {
        @setRuntimeSafety(true);
        const t = self.cur();
        switch (t.tag) {
            .name => {
                _ = self.advance();
                if (std.mem.eql(u8, t.text, "true") or std.mem.eql(u8, t.text, "True")) return self.mk(t.pos, .{ .boolean = true });
                if (std.mem.eql(u8, t.text, "false") or std.mem.eql(u8, t.text, "False")) return self.mk(t.pos, .{ .boolean = false });
                if (std.mem.eql(u8, t.text, "none") or std.mem.eql(u8, t.text, "None")) return self.mk(t.pos, .none);
                if (std.mem.eql(u8, t.text, "caller")) self.specials.caller = true;
                if (std.mem.eql(u8, t.text, "varargs")) self.specials.varargs = true;
                if (std.mem.eql(u8, t.text, "kwargs")) self.specials.kwargs = true;
                return self.mk(t.pos, .{ .name = t.text });
            },
            .string => {
                _ = self.advance();
                if (self.cur().tag != .string) return self.mk(t.pos, .{ .string = t.text });
                var buf = std.ArrayList(u8).empty;
                try buf.appendSlice(self.arena, t.text);
                while (self.cur().tag == .string) try buf.appendSlice(self.arena, self.advance().text);
                return self.mk(t.pos, .{ .string = buf.items });
            },
            .integer => {
                _ = self.advance();
                return self.mk(t.pos, try self.integer(t));
            },
            .float => {
                _ = self.advance();
                return self.mk(t.pos, .{ .float = std.fmt.parseFloat(f64, t.text) catch return self.fail(t.pos, "invalid float literal", .{}) });
            },
            .op => {
                if (self.skipOp("(")) {
                    const e = try self.tuple(.{ .explicit_parens = true });
                    try self.expectOp(")");
                    return e;
                }
                if (self.skipOp("[")) {
                    var items = std.ArrayList(*const Expr).empty;
                    while (!self.isOp("]")) {
                        if (items.items.len > 0) try self.expectOp(",");
                        if (self.isOp("]")) break;
                        try items.append(self.arena, try self.expression(true));
                    }
                    try self.expectOp("]");
                    return self.mk(t.pos, .{ .list = items.items });
                }
                if (self.skipOp("{")) {
                    var pairs = std.ArrayList(ast.Pair).empty;
                    while (!self.isOp("}")) {
                        if (pairs.items.len > 0) try self.expectOp(",");
                        if (self.isOp("}")) break;
                        const key = try self.expression(true);
                        try self.expectOp(":");
                        try pairs.append(self.arena, .{ .key = key, .value = try self.expression(true) });
                    }
                    try self.expectOp("}");
                    return self.mk(t.pos, .{ .dict = pairs.items });
                }
            },
            else => {},
        }
        return self.fail(t.pos, "unexpected '{s}'", .{describe(t)});
    }

    fn literal(v: value.Value) Expr.Kind {
        @setRuntimeSafety(true);
        return switch (v) {
            .int => |i| .{ .int = i },
            .bigint => |t| .{ .bigint = t },
            else => unreachable,
        };
    }

    fn integer(self: *Parser, t: Token) Error!Expr.Kind {
        @setRuntimeSafety(true);
        var digits = std.ArrayList(u8).empty;
        for (t.text) |c| if (c != '_') try digits.append(self.arena, c);
        const text = digits.items;
        if (text.len > 2 and text[0] == '0' and std.ascii.isAlphabetic(text[1])) {
            const base: u8 = switch (std.ascii.toLower(text[1])) {
                'b' => 2,
                'o' => 8,
                else => 16,
            };
            return literal(value.integer(try self.decimal(t, text[2..], base)));
        }
        var start: usize = 0;
        while (start + 1 < text.len and text[start] == '0') start += 1;
        return literal(value.integer(text[start..]));
    }

    /// The decimal text of `digits` in `base`, past 64 bits; past 1024 digits the literal is refused by name.
    fn decimal(self: *Parser, t: Token, digits: []const u8, base: u8) Error![]const u8 {
        @setRuntimeSafety(true);
        if (digits.len > 1024) return self.fail(t.pos, "integer literal '{s}' is too long", .{t.text[0..16]});
        // Little-endian base-10 digits, multiplied by `base` and added to per input digit.
        var out = std.ArrayList(u8).empty;
        for (digits) |c| {
            var carry: u32 = std.fmt.charToDigit(c, base) catch return self.fail(t.pos, "invalid integer literal '{s}'", .{t.text});
            for (out.items) |*d| {
                const v = @as(u32, d.*) * base + carry;
                d.* = @intCast(v % 10);
                carry = v / 10;
            }
            while (carry > 0) : (carry /= 10) try out.append(self.arena, @intCast(carry % 10));
        }
        if (out.items.len == 0) try out.append(self.arena, 0);
        std.mem.reverse(u8, out.items);
        for (out.items) |*d| d.* += '0';
        return out.items;
    }

    fn postfix(self: *Parser, start: *const Expr) Error!*const Expr {
        @setRuntimeSafety(true);
        var node = start;
        const base = self.depth;
        defer self.depth = base;
        while (true) {
            if (self.isOp(".") or self.isOp("[")) {
                try self.enter();
                node = try self.subscript(node);
            } else if (self.isOp("(")) {
                try self.enter();
                node = try self.callExpr(node);
            } else break;
        }
        return node;
    }

    fn filterExpr(self: *Parser, start: *const Expr) Error!*const Expr {
        @setRuntimeSafety(true);
        var node = start;
        const base = self.depth;
        defer self.depth = base;
        while (true) {
            if (self.isOp("|")) {
                node = try self.filterChain(node, false);
            } else if (self.isName("is")) {
                try self.enter();
                node = try self.testExpr(node);
            } else if (self.isOp("(")) {
                try self.enter();
                node = try self.callExpr(node);
            } else break;
        }
        return node;
    }

    fn subscript(self: *Parser, obj: *const Expr) Error!*const Expr {
        @setRuntimeSafety(true);
        const t = self.advance();
        if (t.text[0] == '.') {
            const attr = self.advance();
            if (attr.tag == .name) return self.mk(t.pos, .{ .getattr = .{ .obj = obj, .name = attr.text } });
            if (attr.tag != .integer) return self.fail(attr.pos, "expected name or number", .{});
            return self.mk(t.pos, .{ .getitem = .{ .obj = obj, .key = try self.mk(attr.pos, try self.integer(attr)) } });
        }
        var args = std.ArrayList(*const Expr).empty;
        var slice: ?Expr.Kind = null;
        while (!self.isOp("]")) {
            if (args.items.len > 0 or slice != null) try self.expectOp(",");
            if (slice != null) return self.fail(self.cur().pos, "unsupported subscript", .{});
            const first: ?*const Expr = if (self.isOp(":")) null else try self.expression(true);
            if (!self.isOp(":")) {
                try args.append(self.arena, first.?);
                continue;
            }
            _ = self.advance();
            const stop: ?*const Expr = if (self.isOp(":") or self.isOp("]") or self.isOp(",")) null else try self.expression(true);
            var step: ?*const Expr = null;
            if (self.skipOp(":")) step = if (self.isOp("]") or self.isOp(",")) null else try self.expression(true);
            if (args.items.len > 0) return self.fail(t.pos, "unsupported subscript", .{});
            slice = .{ .slice = .{ .obj = obj, .start = first, .stop = stop, .step = step } };
        }
        try self.expectOp("]");
        if (slice) |s| return self.mk(t.pos, s);
        if (args.items.len == 0) return self.fail(t.pos, "expected a subscript", .{});
        const key = if (args.items.len == 1) args.items[0] else try self.mk(t.pos, .{ .tuple = args.items });
        return self.mk(t.pos, .{ .getitem = .{ .obj = obj, .key = key } });
    }

    fn callArgs(self: *Parser) Error!ast.Args {
        @setRuntimeSafety(true);
        try self.expectOp("(");
        var positional = std.ArrayList(*const Expr).empty;
        var keywords = std.ArrayList(ast.Keyword).empty;
        var names = std.ArrayList(Named).empty;
        var first = true;
        while (!self.isOp(")")) {
            if (!first) {
                try self.expectOp(",");
                if (self.isOp(")")) break;
            }
            first = false;
            if (self.isOp("*") or self.isOp("**")) return self.fail(self.cur().pos, "unsupported argument unpacking '{s}'", .{self.cur().text});
            if (self.cur().tag == .name and self.peekAt(1).tag == .op and std.mem.eql(u8, self.peekAt(1).text, "=")) {
                const at = self.cur().pos;
                const name = self.advance().text;
                _ = self.advance();
                try keywords.append(self.arena, .{ .name = name, .value = try self.expression(true) });
                try names.append(self.arena, .{ .name = name, .pos = at });
            } else {
                if (keywords.items.len > 0) return self.fail(self.cur().pos, "positional argument follows keyword argument", .{});
                try positional.append(self.arena, try self.expression(true));
            }
        }
        try self.expectOp(")");
        // Python's compile of the code Jinja2 makes refuses a keyword given twice.
        if (try self.firstRepeat(names.items)) |dup| return self.fail(dup.pos, "keyword argument repeated: {s}", .{dup.name});
        return .{ .positional = positional.items, .keywords = keywords.items };
    }

    fn callExpr(self: *Parser, callee: *const Expr) Error!*const Expr {
        @setRuntimeSafety(true);
        return self.mk(callee.pos, .{ .call = .{ .callee = callee, .args = try self.callArgs() } });
    }

    /// `| name(args)` filters, as many as follow; `inline` starts at the name (a filter block).
    fn filterChain(self: *Parser, operand: ?*const Expr, inline_start: bool) Error!*const Expr {
        @setRuntimeSafety(true);
        var node = operand;
        var first = inline_start;
        const base = self.depth;
        defer self.depth = base;
        while (first or self.isOp("|")) {
            try self.enter();
            if (!first) _ = self.advance();
            first = false;
            const t = self.cur();
            const name = try self.expectName();
            if (self.isOp(".")) return self.fail(t.pos, "unsupported filter '{s}.'", .{name});
            const filter = std.meta.stringToEnum(builtins.Filter, name) orelse return self.fail(t.pos, "unsupported filter '{s}'", .{name});
            const args: ast.Args = if (self.isOp("(")) try self.callArgs() else .{};
            node = try self.mk(t.pos, .{ .filter = .{ .operand = node, .filter = filter, .args = args } });
        }
        return node.?;
    }

    fn testExpr(self: *Parser, operand: *const Expr) Error!*const Expr {
        @setRuntimeSafety(true);
        const pos = self.advance().pos;
        const negated = self.skipName("not");
        const t = self.cur();
        const name = try self.expectName();
        if (self.isOp(".")) return self.fail(t.pos, "unsupported test '{s}.'", .{name});
        const test_ = std.meta.stringToEnum(builtins.Test, name) orelse return self.fail(t.pos, "unsupported test '{s}'", .{name});
        var args: ast.Args = .{};
        if (self.isOp("(")) {
            args = try self.callArgs();
        } else {
            const c = self.cur();
            const starts_arg = switch (c.tag) {
                .name => !(std.mem.eql(u8, c.text, "else") or std.mem.eql(u8, c.text, "or") or std.mem.eql(u8, c.text, "and")),
                .string, .integer, .float => true,
                .op => std.mem.eql(u8, c.text, "(") or std.mem.eql(u8, c.text, "[") or std.mem.eql(u8, c.text, "{"),
                else => false,
            };
            if (starts_arg) {
                if (self.isName("is")) return self.fail(c.pos, "cannot chain multiple tests with 'is'", .{});
                const arg = try self.postfix(try self.primary());
                args = .{ .positional = try self.arena.dupe(*const Expr, &.{arg}) };
            }
        }
        const node = try self.mk(pos, .{ .@"test" = .{ .operand = operand, .test_ = test_, .args = args } });
        return if (negated) self.mk(pos, .{ .not_ = node }) else node;
    }
};

pub fn parse(arena: Allocator, tokens: []const Token, err: *lexer.SyntaxError) Error![]const Node {
    @setRuntimeSafety(true);
    var p = Parser{ .arena = arena, .tokens = tokens, .err = err };
    return p.subparse(&.{});
}
