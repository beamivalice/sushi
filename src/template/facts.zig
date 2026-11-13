//! What a compiled template says about itself, read from its tree once at compile: the names it reads, the string
//! literals it holds and compares, the text it may emit and the texts it raises. These replace searches over the
//! template source, which count a word inside a comment or an unrelated literal as use.
//!
//! The emitted set is sound in the may-reach-output direction, because a special it misses is masked from sampling:
//! template data always counts, and a string literal counts unless its place provably never reaches output (an
//! operand of a comparison or `in`, a test's operand or argument, a subscript key, a `raise_exception` argument).
//! When in doubt it counts. A special assembled from pieces (`'<|im_' ~ 'end|>'`) is in neither the emitted set nor
//! the template source, and is not seen.
const std = @import("std");
const ast = @import("ast.zig");
const value = @import("value.zig");

const Allocator = std.mem.Allocator;
const Expr = ast.Expr;
const Node = ast.Node;
const Set = std.StringHashMapUnmanaged(void);

pub const Facts = struct {
    /// Names the template reads (`x` of `x`, `x.y`, `x|f`), never only assigns.
    names: Set = .empty,
    /// Attribute names it reads (`.y` of `x.y`, a string key of `x['y']`).
    attributes: Set = .empty,
    /// Every string literal in the tree.
    literals: Set = .empty,
    /// String literals compared (`==`, `!=`, `in` and the order operators, or a test's argument).
    compared: Set = .empty,
    /// Template data and the string literals that may reach output, NUL between pieces.
    emitted: []const u8 = "",
    /// The literal texts `raise_exception` is called with.
    raises: []const []const u8 = &.{},

    pub fn reads(self: *const Facts, name: []const u8) bool {
        @setRuntimeSafety(true);
        return self.names.contains(name);
    }

    pub fn readsAttribute(self: *const Facts, name: []const u8) bool {
        @setRuntimeSafety(true);
        return self.attributes.contains(name);
    }

    pub fn hasLiteral(self: *const Facts, text: []const u8) bool {
        @setRuntimeSafety(true);
        return self.literals.contains(text);
    }

    pub fn compares(self: *const Facts, text: []const u8) bool {
        @setRuntimeSafety(true);
        return self.compared.contains(text);
    }

    /// Whether `text` lies inside one emitted piece (data or a literal): a mention is a whole piece or within one.
    pub fn emits(self: *const Facts, text: []const u8) bool {
        @setRuntimeSafety(true);
        if (text.len == 0) return true;
        var buf: [4096]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buf);
        const gpa = if (text.len * @sizeOf(usize) <= buf.len) fba.allocator() else std.heap.page_allocator;
        const finder = value.Finder.init(gpa, text, false) catch return true;
        defer finder.deinit(gpa);
        var scanned: usize = 0;
        return finder.find(self.emitted, &scanned) != null;
    }

    /// The emitted pieces, data and literals, in tree order.
    pub fn pieces(self: *const Facts) std.mem.SplitIterator(u8, .scalar) {
        @setRuntimeSafety(true);
        return std.mem.splitScalar(u8, self.emitted, 0);
    }
};

/// The facts of `body`, allocated in `arena` (the template's own).
pub fn of(arena: Allocator, body: []const Node) error{OutOfMemory}!Facts {
    @setRuntimeSafety(true);
    var w = Walker{ .arena = arena, .raise_shadowed = shadowsRaise(body) };
    try w.nodes(body);
    w.facts.emitted = w.emitted.items;
    w.facts.raises = w.raises.items;
    return w.facts;
}

/// Whether anything in the template binds the name `raise_exception` (a macro, a set, a loop target, a parameter, a
/// caller's parameter): then a call of it may print its argument, so no call of it counts as the raise.
fn shadowsRaise(body: []const Node) bool {
    @setRuntimeSafety(true);
    for (body) |n| switch (n.kind) {
        .macro => |m| if (std.mem.eql(u8, m.name, "raise_exception") or bindsInMacro(m)) return true,
        .set => |st| if (targetBinds(st.target)) return true,
        .set_block => |st| if (targetBinds(st.target) or shadowsRaise(st.body)) return true,
        .if_ => |f| {
            for (f.branches) |b| if (shadowsRaise(b.body)) return true;
            if (shadowsRaise(f.else_)) return true;
        },
        .for_ => |f| if (targetBinds(f.target) or shadowsRaise(f.body) or shadowsRaise(f.else_)) return true,
        .call_block => |c| if (bindsInMacro(c.caller)) return true,
        .filter_block => |f| if (shadowsRaise(f.body)) return true,
        .generation => |g| if (shadowsRaise(g)) return true,
        else => {},
    };
    return false;
}

fn bindsInMacro(m: *const ast.Macro) bool {
    @setRuntimeSafety(true);
    for (m.params) |p| if (std.mem.eql(u8, p.name, "raise_exception")) return true;
    return shadowsRaise(m.body);
}

fn targetBinds(t: ast.Target) bool {
    @setRuntimeSafety(true);
    return t.binds("raise_exception");
}

const Walker = struct {
    arena: Allocator,
    raise_shadowed: bool,
    facts: Facts = .{},
    emitted: std.ArrayList(u8) = .empty,
    raises: std.ArrayList([]const u8) = .empty,

    fn emit(self: *Walker, text: []const u8) error{OutOfMemory}!void {
        @setRuntimeSafety(true);
        // A NUL inside a piece would split it; the piece still counts, in parts.
        if (self.emitted.items.len > 0) try self.emitted.append(self.arena, 0);
        try self.emitted.appendSlice(self.arena, text);
    }

    fn nodes(self: *Walker, body: []const Node) error{OutOfMemory}!void {
        @setRuntimeSafety(true);
        for (body) |*n| try self.node(n);
    }

    fn node(self: *Walker, n: *const Node) error{OutOfMemory}!void {
        @setRuntimeSafety(true);
        switch (n.kind) {
            .data => |d| try self.emit(d),
            .output => |e| try self.expr(e, true),
            .if_ => |s| {
                for (s.branches) |b| {
                    try self.expr(b.test_, true);
                    try self.nodes(b.body);
                }
                try self.nodes(s.else_);
            },
            .for_ => |s| {
                try self.expr(s.iter, true);
                if (s.filter) |f| try self.expr(f, true);
                try self.nodes(s.body);
                try self.nodes(s.else_);
            },
            .set => |s| try self.expr(s.value, true),
            .set_block => |s| {
                if (s.filter) |f| try self.expr(f, true);
                try self.nodes(s.body);
            },
            .macro => |m| try self.macro(m),
            .call_block => |s| {
                try self.expr(s.call, true);
                try self.macro(s.caller);
            },
            .filter_block => |s| {
                try self.expr(s.filter, true);
                try self.nodes(s.body);
            },
            .generation => |body| try self.nodes(body),
            .break_, .continue_ => {},
        }
    }

    fn macro(self: *Walker, m: *const ast.Macro) error{OutOfMemory}!void {
        @setRuntimeSafety(true);
        for (m.params) |p| if (p.default) |d| try self.expr(d, true);
        try self.nodes(m.body);
    }

    fn args(self: *Walker, a: ast.Args, out: bool) error{OutOfMemory}!void {
        @setRuntimeSafety(true);
        for (a.positional) |e| try self.expr(e, out);
        for (a.keywords) |k| try self.expr(k.value, out);
    }

    /// `out`: the value may reach output, so a string literal here counts as emitted.
    fn expr(self: *Walker, e: *const Expr, out: bool) error{OutOfMemory}!void {
        @setRuntimeSafety(true);
        switch (e.kind) {
            .none, .boolean, .int, .bigint, .float => {},
            .string => |s| {
                try self.facts.literals.put(self.arena, s, {});
                if (out) try self.emit(s);
            },
            .name => |n| try self.facts.names.put(self.arena, n, {}),
            .list, .tuple, .concat => |xs| for (xs) |x| try self.expr(x, out),
            .dict => |pairs| for (pairs) |p| {
                try self.expr(p.key, out);
                try self.expr(p.value, out);
            },
            .getattr => |g| {
                try self.facts.attributes.put(self.arena, g.name, {});
                try self.expr(g.obj, out);
            },
            .getitem => |g| {
                if (g.key.kind == .string) try self.facts.attributes.put(self.arena, g.key.kind.string, {});
                try self.expr(g.obj, out);
                try self.expr(g.key, false);
            },
            .slice => |s| {
                try self.expr(s.obj, out);
                for ([_]?*const Expr{ s.start, s.stop, s.step }) |b| if (b) |x| try self.expr(x, false);
            },
            .call => |c| {
                // A call's arguments count wherever it sits: a method may store them (`l.append(x)`), a macro print them.
                const raise = !self.raise_shadowed and c.callee.kind == .name and std.mem.eql(u8, c.callee.kind.name, "raise_exception");
                if (raise) for (c.args.positional) |a| if (a.kind == .string) try self.raises.append(self.arena, a.kind.string);
                try self.expr(c.callee, out);
                try self.args(c.args, !raise);
            },
            .filter => |f| {
                if (f.operand) |o| try self.expr(o, out);
                try self.args(f.args, out);
            },
            .@"test" => |t| {
                try self.expr(t.operand, false);
                try self.compared_args(t.args);
            },
            .not_, .neg, .pos => |x| try self.expr(x, out),
            .binary => |b| {
                try self.expr(b.left, out);
                try self.expr(b.right, out);
            },
            .and_ => |b| {
                try self.expr(b.left, out);
                try self.expr(b.right, out);
            },
            .or_ => |b| {
                try self.expr(b.left, out);
                try self.expr(b.right, out);
            },
            .compare => |c| {
                try self.comparedOperand(c.first);
                for (c.ops) |op| try self.comparedOperand(op.expr);
            },
            .cond => |c| {
                try self.expr(c.test_, out);
                try self.expr(c.yes, out);
                if (c.no) |n| try self.expr(n, out);
            },
        }
    }

    fn compared_args(self: *Walker, a: ast.Args) error{OutOfMemory}!void {
        @setRuntimeSafety(true);
        for (a.positional) |e| try self.comparedOperand(e);
        for (a.keywords) |k| try self.comparedOperand(k.value);
    }

    /// An operand of a comparison: a literal here, or one in a list or tuple here, is compared and never output.
    fn comparedOperand(self: *Walker, e: *const Expr) error{OutOfMemory}!void {
        @setRuntimeSafety(true);
        switch (e.kind) {
            .string => |s| try self.facts.compared.put(self.arena, s, {}),
            .list, .tuple => |xs| for (xs) |x| if (x.kind == .string) try self.facts.compared.put(self.arena, x.kind.string, {}),
            else => {},
        }
        try self.expr(e, false);
    }
};

const testing = std.testing;
const template = @import("template.zig");

fn factsOf(arena: Allocator, source: []const u8) !Facts {
    @setRuntimeSafety(true);
    var diag = template.Diagnostic{};
    const t = try template.Template.compile(arena, source, &diag);
    return t.facts;
}

test "template facts: a literal that may reach output is emitted, by every way it can get there" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const flows = [_][]const u8{
        "{% set t = '</think>' %}{{ t }}",
        "{% macro m(x) %}{{ x }}{% endmacro %}{{ m('</think>') }}",
        "{% macro m(x='</think>') %}{{ x }}{% endmacro %}{{ m() }}",
        "{{ 'a' if y else '</think>' }}",
        "{{ y ~ '</think>' }}",
        "{{ ['<think>', '</think>']|join('') }}",
        "{{ y|default('</think>') }}",
        "{% for t in ['</think>'] %}{{ t }}{% endfor %}",
        "{% set ns = namespace(t='</think>') %}{{ ns.t }}",
        "{{ {'k': '</think>'}|tojson }}",
        "{% if y %}</think>{% endif %}",
        "{% set l = [] %}{% if l.append('</think>') is none %}{% endif %}{{ l[0] }}",
        "{% set l = [] %}{% if l.append('</think>') == y %}{% endif %}{{ l|join }}",
        "{% macro raise_exception(m) %}{{ m }}{% endmacro %}{{ raise_exception('</think>') }}",
        "{% set raise_exception = y %}{{ raise_exception('</think>') }}",
        "{% for raise_exception in [y] %}{{ raise_exception('</think>') }}{% endfor %}",
        "{% set a, raise_exception = y, z %}{{ raise_exception('</think>') }}",
        "{% macro m(raise_exception) %}{{ raise_exception('</think>') }}{% endmacro %}{{ m(y) }}",
        "{% macro m() %}{{ caller('</think>') }}{% endmacro %}{% call(raise_exception) m() %}{{ raise_exception }}{% endcall %}",
        "{% macro m() %}{{ caller(y) }}{% endmacro %}{% call(raise_exception) m() %}{{ raise_exception('</think>') }}{% endcall %}",
        "{% filter upper %}{{ '</think>' }}{% endfilter %}",
        "{% call m('</think>') %}{% endcall %}",
    };
    for (flows) |source| {
        const f = try factsOf(arena, source);
        if (!f.emits("</think>")) {
            std.debug.print("not emitted: {s}\n", .{source});
            return error.TestUnexpectedResult;
        }
    }
}

test "template facts: a literal only compared, tested, used as a key or raised is not emitted" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const kept = [_][]const u8{
        "{# </think> #}ok",
        "{% if y == '</think>' %}ok{% endif %}",
        "{% if '</think>' in y %}ok{% endif %}",
        "{% if y in ['</think>', 'x'] %}ok{% endif %}",
        "{% if y is equalto '</think>' %}ok{% endif %}",
        "{{ y['</think>'] }}",
        "{{ raise_exception('</think>') }}",
    };
    for (kept) |source| {
        const f = try factsOf(arena, source);
        if (f.emits("</think>")) {
            std.debug.print("emitted: {s}\n", .{source});
            return error.TestUnexpectedResult;
        }
    }
    const raised = try factsOf(arena, "{{ raise_exception('No user query found in messages.') }}");
    try testing.expectEqualStrings("No user query found in messages.", raised.raises[0]);
}

test "template facts: names and attributes read, literals held and compared" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const f = try factsOf(arena, "{# tools enable_thinking #}{% set x = 1 %}{% for m in messages %}{% if m.role == 'tool' %}{{ m['reasoning_content'] }}{% endif %}{% endfor %}{{ reasoning_effort|default('xhigh') }}");
    try testing.expect(f.reads("messages"));
    try testing.expect(f.reads("reasoning_effort"));
    try testing.expect(!f.reads("tools"));
    try testing.expect(!f.reads("enable_thinking"));
    try testing.expect(!f.reads("x"));
    try testing.expect(f.readsAttribute("role"));
    try testing.expect(f.readsAttribute("reasoning_content"));
    try testing.expect(f.compares("tool"));
    try testing.expect(!f.compares("xhigh"));
    try testing.expect(f.hasLiteral("xhigh"));
    try testing.expect(f.emits("xhigh"));
    try testing.expect(!f.emits("tool"));
}
