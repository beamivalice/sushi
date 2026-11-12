//! Evaluation with Jinja2's meaning: a for-loop iteration and a macro call are scopes of their own (only a
//! `namespace` carries state out), an `if` is not; attribute and item lookup as the sandboxed environment does it;
//! undefined values print nothing and fail when used.
const std = @import("std");
const ast = @import("ast.zig");
const value = @import("value.zig");
const builtins = @import("builtins.zig");

const Allocator = std.mem.Allocator;
const Value = value.Value;
const Str = value.Str;
const Expr = ast.Expr;
const Node = ast.Node;
pub const Failure = value.Failure;

pub const Frame = struct {
    vars: std.StringHashMapUnmanaged(Value) = .empty,
    parent: ?*Frame,
};

/// What one render may spend; past any of them it fails with `TemplateRuntime`.
pub const Limits = struct {
    /// Statements, loop iterations and macro calls.
    max_steps: u64 = 20_000_000,
    /// The largest string a render may build, the output included.
    max_string: usize = 256 << 20,
    /// Every byte the render allocates (parse, values, strings, scopes), added up.
    max_memory: usize = 1 << 30,
    max_call_depth: usize = 100,
    /// Native stack a render may use below its entry. Nesting and call depth multiply, so only a budget on the stack
    /// itself bounds them jointly; 4 MiB leaves the 8 MiB main thread (CLI, kld) its headroom.
    max_stack: usize = 4 << 20,
};

/// An allocator that refuses to hand out more than `limit` bytes in all, and records that it refused. It never
/// frees and never moves: a value may hold a slice of a list another value grows (a for loop over a list its body
/// appends to), so everything a render allocated stays where it is until the caller frees `child` whole.
pub const Budget = struct {
    child: Allocator,
    limit: usize,
    used: usize = 0,
    exceeded: bool = false,

    pub fn allocator(self: *Budget) Allocator {
        @setRuntimeSafety(true);
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable: Allocator.VTable = .{ .alloc = alloc, .resize = resize, .remap = remap, .free = free };

    fn fits(self: *Budget, n: usize) bool {
        @setRuntimeSafety(true);
        if (n <= self.limit - self.used) return true;
        self.exceeded = true;
        return false;
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        @setRuntimeSafety(true);
        const self: *Budget = @ptrCast(@alignCast(ctx));
        if (!self.fits(len)) return null;
        const p = self.child.rawAlloc(len, alignment, ret_addr) orelse return null;
        self.used += len;
        return p;
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        @setRuntimeSafety(true);
        const self: *Budget = @ptrCast(@alignCast(ctx));
        const grow = new_len -| memory.len;
        if (!self.fits(grow)) return false;
        if (!self.child.rawResize(memory, alignment, new_len, ret_addr)) return false;
        self.used += grow;
        return true;
    }

    /// In place only: a remap that moves would free the old place.
    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        @setRuntimeSafety(true);
        return if (resize(ctx, memory, alignment, new_len, ret_addr)) memory.ptr else null;
    }

    fn free(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize) void {
        @setRuntimeSafety(true);
    }
};

pub const Kw = struct { name: []const u8, value: Value };

const Flow = enum { normal, break_, continue_ };

pub const Renderer = struct {
    arena: Allocator,
    limits: Limits,
    out: *value.Builder,
    /// The step budget: `limits.max_steps`, shared with the value walks.
    meter: value.Meter,
    /// The frame address where the render began; the stack grows down from it.
    stack_base: usize,
    call_depth: usize = 0,
    /// The offset of the node being evaluated, and the first failure's message.
    pos: u32 = 0,
    message: ?[]const u8 = null,
    fail_pos: u32 = 0,
    /// The largest string charged so far, for `Usage`.
    largest_string: usize = 0,

    pub fn fail(self: *Renderer, comptime fmt: []const u8, args: anytype) Failure {
        @setRuntimeSafety(true);
        if (self.message == null) {
            self.message = try std.fmt.allocPrint(self.arena, fmt, args);
            self.fail_pos = self.pos;
        }
        return error.TemplateRuntime;
    }

    pub fn raise(self: *Renderer, text: []const u8) Failure {
        @setRuntimeSafety(true);
        if (self.message == null) {
            self.message = text;
            self.fail_pos = self.pos;
        }
        return error.TemplateRaised;
    }

    /// Refuses a string of `n` bytes about to be built past `max_string`, before it is allocated.
    pub fn charge(self: *Renderer, n: usize) Failure!void {
        @setRuntimeSafety(true);
        self.largest_string = @max(self.largest_string, n);
        if (n > self.limits.max_string) return self.fail("the render builds more text than its limit", .{});
    }

    /// One level deeper of a macro call or a filter a filter applies; the caller undoes it with
    /// `call_depth -= 1`.
    pub fn enterCall(self: *Renderer) Failure!void {
        @setRuntimeSafety(true);
        if (self.call_depth >= self.limits.max_call_depth) return self.fail("calls nest deeper than {d}", .{self.limits.max_call_depth});
        self.call_depth += 1;
    }

    /// Every recursion of the render passes `exec` or `eval`; past `max_stack` below the entry it fails by name.
    fn stackCheck(self: *Renderer) Failure!void {
        @setRuntimeSafety(true);
        const used = self.stack_base -| @frameAddress();
        if (used > self.limits.max_stack) return self.fail("the render nests deeper than its stack allows", .{});
    }

    fn step(self: *Renderer) Failure!void {
        @setRuntimeSafety(true);
        return self.tick(1);
    }

    /// `n` steps of work.
    pub fn tick(self: *Renderer, n: u64) Failure!void {
        @setRuntimeSafety(true);
        self.meter.tick(n) catch return self.fail("the render takes more than {d} steps", .{self.limits.max_steps});
    }

    /// The work of a pass over `bytes` bytes: one step per 64.
    pub fn scan(self: *Renderer, bytes: usize) Failure!void {
        @setRuntimeSafety(true);
        return self.tick(bytes / 64 + 1);
    }

    pub fn write(self: *Renderer, text: Str) Failure!void {
        @setRuntimeSafety(true);
        try self.charge(text.bytes.len);
        if (self.out.bytes.items.len + text.bytes.len > self.limits.max_string) return self.fail("the render builds more text than its limit", .{});
        try self.out.append(self.arena, text);
    }

    pub fn newList(self: *Renderer, items: []const Value, kind: value.List.Kind) Failure!Value {
        @setRuntimeSafety(true);
        const l = try self.arena.create(value.List);
        l.* = .{ .kind = kind };
        try l.items.appendSlice(self.arena, items);
        return .{ .list = l };
    }

    /// A list over `items`, which it takes without a copy.
    pub fn ownList(self: *Renderer, items: []Value, kind: value.List.Kind) Failure!Value {
        @setRuntimeSafety(true);
        const l = try self.arena.create(value.List);
        l.* = .{ .kind = kind, .items = .fromOwnedSlice(items) };
        return .{ .list = l };
    }

    pub fn newString(self: *Renderer, text: Str) Failure!Value {
        @setRuntimeSafety(true);
        try self.charge(text.bytes.len);
        return .{ .string = text };
    }

    /// Python `str(v)`, its bytes taking the strictest origin of the strings in `v`.
    pub fn toText(self: *Renderer, v: Value) Failure!Str {
        @setRuntimeSafety(true);
        if (v.strOf()) |s| return s;
        return .{ .bytes = try self.toStr(v), .origin = value.originIn(v, 0) };
    }

    /// Python `str(v)` as a new string.
    pub fn toStr(self: *Renderer, v: Value) Failure![]const u8 {
        @setRuntimeSafety(true);
        if (v.str()) |s| return s;
        var buf = std.ArrayList(u8).empty;
        value.writeStr(&self.meter, &buf, self.arena, v, 0) catch |e| {
            if (e == error.TemplateRuntime and !self.meter.spent()) if (value.unprintable(v, 0)) |kind|
                return self.fail("a {s} cannot be printed: Python prints it with its address", .{kind});
            return self.wrap(e, "the value nests too deep to print");
        };
        try self.charge(buf.items.len);
        return buf.items;
    }

    /// Python's TypeError for a key it cannot hash.
    pub fn hashable(self: *Renderer, key: Value) Failure!void {
        @setRuntimeSafety(true);
        if (value.unhashable(key, 0)) |t| return self.fail("unhashable type: '{s}'", .{t});
    }

    /// Gives a bare failure from `value.zig` its message.
    pub fn wrap(self: *Renderer, e: Failure, comptime what: []const u8) Failure {
        @setRuntimeSafety(true);
        return switch (e) {
            error.TemplateRuntime => if (self.meter.spent())
                self.fail("the render takes more than {d} steps", .{self.limits.max_steps})
            else
                self.fail(what, .{}),
            else => e,
        };
    }

    // ── Statements ──

    pub fn renderBody(self: *Renderer, nodes: []const Node, frame: *Frame) Failure!Flow {
        @setRuntimeSafety(true);
        for (nodes) |*node| {
            const flow = try self.exec(node, frame);
            if (flow != .normal) return flow;
        }
        return .normal;
    }

    /// The body rendered into its own string.
    /// What `nodes` write, and how they ended: a `break` or `continue` leaves the enclosing loop's step, as in
    /// Python, where the captured text is then never assigned or written.
    fn capture(self: *Renderer, nodes: []const Node, frame: *Frame) Failure!struct { Str, Flow } {
        @setRuntimeSafety(true);
        var buf: value.Builder = .{};
        const saved = self.out;
        self.out = &buf;
        defer self.out = saved;
        const flow = try self.renderBody(nodes, frame);
        return .{ buf.str(), flow };
    }

    fn exec(self: *Renderer, node: *const Node, frame: *Frame) Failure!Flow {
        @setRuntimeSafety(true);
        value.noteStack();
        try self.step();
        self.pos = node.pos;
        try self.stackCheck();
        switch (node.kind) {
            .data => |text| try self.write(.{ .bytes = text }),
            .output => |e| try self.write(try self.toText(try self.eval(e, frame))),
            .if_ => |s| {
                for (s.branches) |b| if (value.truthy(try self.eval(b.test_, frame))) return self.renderBody(b.body, frame);
                return self.renderBody(s.else_, frame);
            },
            .for_ => |s| return self.forLoop(s, frame),
            .set => |s| try self.assign(s.target, try self.eval(s.value, frame), frame),
            .set_block => |s| {
                const text, const flow = try self.capture(s.body, frame);
                if (flow != .normal) return flow;
                var v: Value = .{ .string = text };
                if (s.filter) |f| v = try self.evalFilter(f, v, frame);
                try self.assign(s.target, v, frame);
            },
            .macro => |m| try frame.vars.put(self.arena, m.name, try self.macroValue(m, frame, false)),
            .call_block => |s| {
                const caller = try self.macroValue(s.caller, frame, true);
                const v = try self.evalCall(s.call, frame, caller);
                try self.write(try self.toText(v));
            },
            .filter_block => |s| {
                const text, const flow = try self.capture(s.body, frame);
                if (flow != .normal) return flow;
                const body: Value = .{ .string = text };
                try self.write(try self.toText(try self.evalFilter(s.filter, body, frame)));
            },
            .generation => |body| return self.renderBody(body, frame),
            .break_ => return .break_,
            .continue_ => return .continue_,
        }
        return .normal;
    }

    fn macroValue(self: *Renderer, m: *const ast.Macro, frame: *Frame, is_caller: bool) Failure!Value {
        @setRuntimeSafety(true);
        const mv = try self.arena.create(value.Macro);
        mv.* = .{ .node = m, .closure = frame, .is_caller = is_caller };
        return .{ .macro = mv };
    }

    fn forLoop(self: *Renderer, s: anytype, frame: *Frame) Failure!Flow {
        @setRuntimeSafety(true);
        if (s.filter == null) if (try self.lazyRange(s.iter, frame)) |range| return self.walk(s, frame, &.{}, range);
        const iterable = try self.eval(s.iter, frame);
        const all = try self.walkable(iterable);
        var items = all;
        if (s.filter) |f| {
            const scope = try self.arena.create(Frame);
            scope.* = .{ .parent = frame };
            var kept = std.ArrayList(Value).empty;
            for (all) |item| {
                try self.step();
                scope.vars.clearRetainingCapacity();
                try self.assign(s.target, item, scope);
                if (value.truthy(try self.eval(f, scope))) try kept.append(self.arena, item);
            }
            items = kept.items;
        }
        return self.walk(s, frame, items, null);
    }

    /// The bounds of a loop's iterable when it is a call of the global `range`, null for any other.
    fn lazyRange(self: *Renderer, iter: *const Expr, frame: *Frame) Failure!?value.Loop.Range {
        @setRuntimeSafety(true);
        if (iter.kind != .call) return null;
        const c = iter.kind.call;
        if (c.callee.kind != .name or !std.mem.eql(u8, c.callee.kind.name, "range")) return null;
        const callee = try self.eval(c.callee, frame);
        if (callee != .function or callee.function.* != .global or callee.function.global != .range) return null;
        const args = try self.evalArgs(c.args, frame);
        self.pos = iter.pos;
        const b = try builtins.rangeOf(self, args.positional, args.keywords);
        return .{ .start = b.start, .step = b.stride, .len = b.len };
    }

    /// The iterations of a for loop over `items`, or over the lazy `range`, and its `else`.
    fn walk(self: *Renderer, s: anytype, frame: *Frame, items: []const Value, range: ?value.Loop.Range) Failure!Flow {
        @setRuntimeSafety(true);
        // One scope and one `loop` serve every iteration, cleared between them: a loop costs no memory per
        // iteration, and a macro the body defines keeps a live frame (arena, not stack) after the loop.
        const scope = try self.arena.create(Frame);
        scope.* = .{ .parent = frame };
        const loop = try self.arena.create(value.Loop);
        loop.* = .{ .index0 = 0, .items = items, .range = range };
        // Jinja2 marks a loop iterated at the end of a body: `else` runs when no iteration got there, every one
        // left by `break` or `continue`, or there were none.
        var iterated = false;
        for (0..loop.len()) |i| {
            try self.step();
            scope.vars.clearRetainingCapacity();
            loop.index0 = i;
            try scope.vars.put(self.arena, "loop", .{ .loop = loop });
            try self.assign(s.target, loop.at(i), scope);
            switch (try self.renderBody(s.body, scope)) {
                .normal => iterated = true,
                .continue_ => {},
                .break_ => break,
            }
        }
        if (iterated) return .normal;
        // The else body is a scope of its own, as an iteration is.
        scope.vars.clearRetainingCapacity();
        return self.renderBody(s.else_, scope);
    }

    /// The items a for loop or a filter walks: a list's items, a dict's keys, a string's characters.
    /// Each item handed out is a step: whatever walks them is metered by the walk.
    pub fn iterate(self: *Renderer, v: Value) Failure![]const Value {
        @setRuntimeSafety(true);
        if (v == .list) try self.tick(v.list.items.items.len);
        return self.walkable(v);
    }

    /// The items of `v` as `iterate` gives them, a list's charged by whoever walks them (a for loop, a step per
    /// iteration, so one that breaks early pays only for what it walked); dict keys and characters, which are
    /// copied out here, are charged here.
    fn walkable(self: *Renderer, v: Value) Failure![]const Value {
        @setRuntimeSafety(true);
        switch (v) {
            .dict => |d| try self.tick(d.entries.items.len),
            .string, .markup => |s| try self.scan(s.bytes.len),
            else => {},
        }
        return switch (v) {
            .undefined => &.{},
            .list => |l| blk: {
                const items = l.items.items;
                if (l.isIterator()) l.items = .empty;
                break :blk items;
            },
            .dict => |d| blk: {
                const keys = try self.arena.alloc(Value, d.entries.items.len);
                for (d.entries.items, keys) |e, *k| k.* = e.key;
                break :blk keys;
            },
            .string, .markup => |s| blk: {
                var chars = std.ArrayList(Value).empty;
                var it = std.unicode.Utf8View.initUnchecked(s.bytes).iterator();
                var at: usize = 0;
                while (it.nextCodepointSlice()) |c| : (at += c.len) try chars.append(self.arena, .{ .string = try s.slice(self.arena, at, at + c.len) });
                break :blk chars.items;
            },
            else => self.fail("'{s}' object is not iterable", .{value.typeName(v)}),
        };
    }

    fn assign(self: *Renderer, target: ast.Target, v: Value, frame: *Frame) Failure!void {
        @setRuntimeSafety(true);
        switch (target) {
            .name => |n| try frame.vars.put(self.arena, n, v),
            .names => |names| {
                const items = try self.iterate(v);
                if (items.len != names.len) return self.fail("cannot unpack {d} values into {d} names", .{ items.len, names.len });
                for (names, items) |n, item| try frame.vars.put(self.arena, n, item);
            },
            .attr => |a| {
                const ns = lookup(frame, a.namespace) orelse return self.fail("'{s}' is undefined", .{a.namespace});
                if (ns != .namespace) return self.fail("cannot assign attribute on non-namespace object", .{});
                try ns.namespace.put(&self.meter, self.arena, .plain(a.name), v);
            },
        }
    }

    fn lookup(frame: *Frame, name: []const u8) ?Value {
        @setRuntimeSafety(true);
        var f: ?*Frame = frame;
        while (f) |cur| : (f = cur.parent) if (cur.vars.get(name)) |v| return v;
        return null;
    }

    // ── Expressions ──

    pub fn eval(self: *Renderer, e: *const Expr, frame: *Frame) Failure!Value {
        @setRuntimeSafety(true);
        value.noteStack();
        self.pos = e.pos;
        try self.stackCheck();
        return switch (e.kind) {
            .none => .none,
            .boolean => |b| .{ .boolean = b },
            .int => |i| .{ .int = i },
            .bigint => |t| .{ .bigint = t },
            .float => |f| .{ .float = f },
            .string => |s| .plain(s),
            .name => |n| lookup(frame, n) orelse if (std.meta.stringToEnum(value.Global, n)) |g|
                try self.function(.{ .global = g })
            else
                .{ .undefined = n },
            .list, .tuple => |items| blk: {
                const vals = try self.arena.alloc(Value, items.len);
                for (items, vals) |item, *v| v.* = try self.eval(item, frame);
                break :blk try self.newList(vals, if (e.kind == .tuple) .tuple else .list);
            },
            .dict => |pairs| blk: {
                const d = try self.arena.create(value.Dict);
                d.* = .{};
                for (pairs) |p| {
                    const k = try self.eval(p.key, frame);
                    try self.hashable(k);
                    try d.put(&self.meter, self.arena, k, try self.eval(p.value, frame));
                }
                break :blk .{ .dict = d };
            },
            .getattr => |g| try self.getAttr(try self.eval(g.obj, frame), g.name),
            .getitem => |g| blk: {
                const obj = try self.eval(g.obj, frame);
                break :blk try self.getItem(obj, try self.eval(g.key, frame));
            },
            .slice => |s| try self.evalSlice(s, frame),
            .call => try self.evalCall(e, frame, null),
            .filter => try self.evalFilter(e, null, frame),
            .@"test" => |t| blk: {
                const operand = try self.eval(t.operand, frame);
                const args = try self.evalArgs(t.args, frame);
                self.pos = e.pos;
                break :blk .{ .boolean = try builtins.applyTest(self, t.test_, operand, args.positional, args.keywords) };
            },
            .not_ => |x| .{ .boolean = !value.truthy(try self.eval(x, frame)) },
            .neg, .pos => |x| try self.unaryOp(try self.eval(x, frame), e.kind == .neg),
            .binary => |b| blk: {
                const l = try self.eval(b.left, frame);
                const r = try self.eval(b.right, frame);
                self.pos = e.pos;
                break :blk try builtins.binary(self, b.op, l, r);
            },
            .and_ => |b| blk: {
                const l = try self.eval(b.left, frame);
                break :blk if (!value.truthy(l)) l else try self.eval(b.right, frame);
            },
            .or_ => |b| blk: {
                const l = try self.eval(b.left, frame);
                break :blk if (value.truthy(l)) l else try self.eval(b.right, frame);
            },
            .concat => |parts| blk: {
                var buf: value.Builder = .{};
                for (parts) |p| {
                    const v = try self.eval(p, frame);
                    const s = try self.toText(v);
                    try self.scan(s.bytes.len);
                    try self.charge(buf.bytes.items.len + s.bytes.len);
                    try buf.append(self.arena, s);
                }
                break :blk .{ .string = buf.str() };
            },
            .compare => |c| try self.evalCompare(c, frame),
            .cond => |c| if (value.truthy(try self.eval(c.test_, frame)))
                try self.eval(c.yes, frame)
            else if (c.no) |no|
                try self.eval(no, frame)
            else
                .{ .undefined = "" },
        };
    }

    fn unaryOp(self: *Renderer, v: Value, negate: bool) Failure!Value {
        @setRuntimeSafety(true);
        return switch (v) {
            .boolean => |b| .{ .int = if (negate) -@as(i64, @intFromBool(b)) else @intFromBool(b) },
            // The one negation past 64 bits is exact: 2^63, a bigint as Python prints it.
            .int => |i| if (negate) (if (std.math.negate(i)) |n| .{ .int = n } else |_| value.integer("9223372036854775808")) else v,
            .float => |f| .{ .float = if (negate) -f else f },
            .bigint => |t| if (!negate) v else value.integer(if (t[0] == '-') t[1..] else try std.fmt.allocPrint(self.arena, "-{s}", .{t})),
            .undefined => |n| self.fail("'{s}' is undefined", .{n}),
            else => self.fail("bad operand type for unary {s}: '{s}'", .{ if (negate) "-" else "+", value.typeName(v) }),
        };
    }

    fn evalCompare(self: *Renderer, c: anytype, frame: *Frame) Failure!Value {
        @setRuntimeSafety(true);
        var left = try self.eval(c.first, frame);
        for (c.ops) |op| {
            const right = try self.eval(op.expr, frame);
            const ok = switch (op.op) {
                .eq => value.eql(&self.meter, left, right, 0) catch |e| return self.wrap(e, "the values nest too deep to compare"),
                .ne => !(value.eql(&self.meter, left, right, 0) catch |e| return self.wrap(e, "the values nest too deep to compare")),
                .in => try builtins.contains(self, right, left),
                .notin => !try builtins.contains(self, right, left),
                else => blk: {
                    const o = (value.order(&self.meter, left, right, 0) catch |e| return self.wrap(e, "the values cannot be ordered")) orelse break :blk false;
                    break :blk switch (op.op) {
                        .lt => o == .lt,
                        .lteq => o != .gt,
                        .gt => o == .gt,
                        .gteq => o != .lt,
                        else => unreachable,
                    };
                },
            };
            if (!ok) return .{ .boolean = false };
            left = right;
        }
        return .{ .boolean = true };
    }

    pub const EvaluatedArgs = struct { positional: []const Value, keywords: []const Kw };

    fn evalArgs(self: *Renderer, args: ast.Args, frame: *Frame) Failure!EvaluatedArgs {
        @setRuntimeSafety(true);
        const pos = try self.arena.alloc(Value, args.positional.len);
        for (args.positional, pos) |a, *v| v.* = try self.eval(a, frame);
        const kws = try self.arena.alloc(Kw, args.keywords.len);
        for (args.keywords, kws) |k, *v| v.* = .{ .name = k.name, .value = try self.eval(k.value, frame) };
        return .{ .positional = pos, .keywords = kws };
    }

    /// A filter node, its innermost operand replaced by `operand` when the node has none (a filter block or a
    /// filtered `set` block).
    fn evalFilter(self: *Renderer, e: *const Expr, operand: ?Value, frame: *Frame) Failure!Value {
        @setRuntimeSafety(true);
        const f = e.kind.filter;
        const input = if (f.operand) |o| (if (o.kind == .filter) try self.evalFilter(o, operand, frame) else try self.eval(o, frame)) else operand.?;
        const args = try self.evalArgs(f.args, frame);
        self.pos = e.pos;
        return builtins.applyFilter(self, f.filter, input, args.positional, args.keywords);
    }

    fn evalCall(self: *Renderer, e: *const Expr, frame: *Frame, caller: ?Value) Failure!Value {
        @setRuntimeSafety(true);
        const c = e.kind.call;
        const callee = try self.eval(c.callee, frame);
        var args = try self.evalArgs(c.args, frame);
        self.pos = e.pos;
        if (caller) |cv| {
            const kws = try self.arena.alloc(Kw, args.keywords.len + 1);
            @memcpy(kws[0..args.keywords.len], args.keywords);
            kws[args.keywords.len] = .{ .name = "caller", .value = cv };
            args.keywords = kws;
        }
        return self.call(callee, args.positional, args.keywords);
    }

    pub fn call(self: *Renderer, callee: Value, args: []const Value, kws: []const Kw) Failure!Value {
        @setRuntimeSafety(true);
        return switch (callee) {
            .macro => |m| self.callMacro(m, args, kws),
            .function => |f| builtins.callFunction(self, f.*, args, kws),
            .undefined => |n| self.fail("'{s}' is undefined", .{n}),
            else => self.fail("'{s}' object is not callable", .{value.typeName(callee)}),
        };
    }

    fn callMacro(self: *Renderer, m: *const value.Macro, args: []const Value, kws: []const Kw) Failure!Value {
        @setRuntimeSafety(true);
        try self.step();
        try self.enterCall();
        defer self.call_depth -= 1;
        const node = m.node;
        const reads = node.reads;
        if (args.len > node.params.len and !reads.varargs) return self.fail("macro '{s}' takes not more than {d} argument(s)", .{ node.name, node.params.len });
        // In the arena: a macro defined in this body may outlive the call.
        const scope = try self.arena.create(Frame);
        scope.* = .{ .parent = @ptrCast(@alignCast(m.closure)) };
        for (node.params, 0..) |p, i| if (i < args.len) try scope.vars.put(self.arena, p.name, args[i]);
        const extra = try self.arena.create(value.Dict);
        extra.* = .{};
        for (kws) |kw| {
            const index = for (node.params, 0..) |p, i| {
                if (std.mem.eql(u8, p.name, kw.name)) break i;
            } else null;
            if (index) |i| {
                if (i < args.len) return self.fail("macro '{s}' got multiple values for argument '{s}'", .{ node.name, kw.name });
                try scope.vars.put(self.arena, kw.name, kw.value);
            } else if (std.mem.eql(u8, kw.name, "caller") and !m.is_caller and reads.caller) {
                try scope.vars.put(self.arena, "caller", kw.value);
            } else if (reads.kwargs) {
                try extra.put(&self.meter, self.arena, .plain(kw.name), kw.value);
            } else if (std.mem.eql(u8, kw.name, "caller") and !m.is_caller) {
                return self.fail("macro '{s}' was invoked with two values for the special caller argument", .{node.name});
            } else return self.fail("macro '{s}' takes no keyword argument '{s}'", .{ node.name, kw.name });
        }
        if (reads.varargs) try scope.vars.put(self.arena, "varargs", try self.newList(args[@min(args.len, node.params.len)..], .tuple));
        if (reads.kwargs) try scope.vars.put(self.arena, "kwargs", .{ .dict = extra });
        // A default is evaluated in the macro's own scope, as Jinja2 does: a parameter it names is the call's
        // value, or undefined when the call gave none and no earlier default set it.
        const given = try self.arena.alloc(bool, node.params.len);
        for (node.params, given) |p, *g| {
            g.* = scope.vars.contains(p.name);
            if (!g.*) try scope.vars.put(self.arena, p.name, .{ .undefined = p.name });
        }
        for (node.params, given) |p, g| if (!g) {
            if (p.default) |d| try scope.vars.put(self.arena, p.name, try self.eval(d, scope));
        };
        const text, _ = try self.capture(node.body, scope);
        return .{ .string = text };
    }

    pub fn function(self: *Renderer, f: value.Function) Failure!Value {
        @setRuntimeSafety(true);
        const p = try self.arena.create(value.Function);
        p.* = f;
        return .{ .function = p };
    }

    // ── Lookup, as Jinja2's sandboxed `getattr` and `getitem` ──

    pub fn getAttr(self: *Renderer, obj: Value, name: []const u8) Failure!Value {
        @setRuntimeSafety(true);
        switch (obj) {
            .undefined => |n| return self.fail("'{s}' is undefined", .{if (n.len > 0) n else "value"}),
            .loop => |l| return self.loopField(l, name),
            else => {},
        }
        if (try builtins.attribute(self, obj, name)) |v| return v;
        return switch (obj) {
            .dict, .namespace => |d| try d.getStr(&self.meter, name) orelse .{ .undefined = name },
            else => .{ .undefined = name },
        };
    }

    pub fn getItem(self: *Renderer, obj: Value, key: Value) Failure!Value {
        @setRuntimeSafety(true);
        switch (obj) {
            .undefined => |n| return self.fail("'{s}' is undefined", .{if (n.len > 0) n else "value"}),
            .list => |l| if (l.isSequence()) if (indexOf(key)) |i| {
                return if (wrapIndex(i, l.items.items.len)) |at| l.items.items[at] else .{ .undefined = "" };
            },
            .string, .markup => |s| if (indexOf(key)) |i| {
                try self.scan(s.bytes.len);
                const n = cpCount(s.bytes);
                const at = wrapIndex(i, n) orelse return .{ .undefined = "" };
                const c = cpSlice(s.bytes, at, at + 1);
                const from = @intFromPtr(c.ptr) - @intFromPtr(s.bytes.ptr);
                return .{ .string = try s.slice(self.arena, from, from + c.len) };
            },
            .dict => |d| if (try d.get(&self.meter, key)) |v| return v,
            else => {},
        }
        if (key.str()) |name| return self.getAttr(obj, name);
        return .{ .undefined = "" };
    }

    fn loopField(self: *Renderer, l: *const value.Loop, name: []const u8) Failure!Value {
        @setRuntimeSafety(true);
        const n = l.len();
        const i = l.index0;
        const Field = enum { index, index0, revindex, revindex0, first, last, length, previtem, nextitem, depth, depth0, cycle, changed };
        const field = std.meta.stringToEnum(Field, name) orelse return .{ .undefined = name };
        return switch (field) {
            .index => .{ .int = @intCast(i + 1) },
            .index0 => .{ .int = @intCast(i) },
            .revindex => .{ .int = @intCast(n - i) },
            .revindex0 => .{ .int = @intCast(n - i - 1) },
            .first => .{ .boolean = i == 0 },
            .last => .{ .boolean = i + 1 == n },
            .length => .{ .int = @intCast(n) },
            .previtem => if (i > 0) l.at(i - 1) else .{ .undefined = "previtem" },
            .nextitem => if (i + 1 < n) l.at(i + 1) else .{ .undefined = "nextitem" },
            .depth, .depth0, .cycle, .changed => self.fail("unsupported loop attribute 'loop.{s}'", .{name}),
        };
    }

    fn evalSlice(self: *Renderer, s: anytype, frame: *Frame) Failure!Value {
        @setRuntimeSafety(true);
        const obj = try self.eval(s.obj, frame);
        var bounds_ok = true;
        const start = if (s.start) |x| sliceBound(try self.eval(x, frame), &bounds_ok) else null;
        const stop = if (s.stop) |x| sliceBound(try self.eval(x, frame), &bounds_ok) else null;
        const step_v = if (s.step) |x| sliceBound(try self.eval(x, frame), &bounds_ok) else null;
        if (obj == .undefined) return self.fail("'{s}' is undefined", .{if (obj.undefined.len > 0) obj.undefined else "value"});
        // Jinja2 compiles a slice as Python's own subscript, not through the sandbox's getitem.
        const sliceable = switch (obj) {
            .list => |l| l.isSequence(),
            .string, .markup => true,
            else => false,
        };
        if (obj == .dict) return self.fail("a dict has no slice as a key", .{});
        if (!sliceable) return self.fail("'{s}' object is not subscriptable", .{value.typeName(obj)});
        if (!bounds_ok) return self.fail("slice indices must be integers or None or have an __index__ method", .{});
        const stride = step_v orelse 1;
        if (stride == 0) return self.fail("slice step cannot be zero", .{});
        switch (obj) {
            .list => |l| {
                const picks = try sliceIndices(self.arena, l.items.items.len, start, stop, stride);
                const out = try self.arena.alloc(Value, picks.len);
                for (picks, out) |i, *v| v.* = l.items.items[i];
                const sliced = try self.newList(out, l.kind);
                // Python's compute_slice: the new range starts and stops at the old one's i-th and j-th steps.
                if (l.bounds) |b| {
                    const ij = sliceEnds(l.items.items.len, start, stop, stride);
                    if (rangeSlice(b.*, ij[0], ij[1], stride)) |nb| {
                        const p = try self.arena.create([3]i128);
                        p.* = nb;
                        sliced.list.bounds = p;
                    }
                }
                return sliced;
            },
            .string, .markup => |str| {
                try self.scan(str.bytes.len);
                var bounds = std.ArrayList(usize).empty;
                var it = std.unicode.Utf8View.initUnchecked(str.bytes).iterator();
                var at: usize = 0;
                while (it.nextCodepointSlice()) |c| : (at += c.len) try bounds.append(self.arena, at);
                try bounds.append(self.arena, str.bytes.len);
                const picks = try sliceIndices(self.arena, bounds.items.len - 1, start, stop, stride);
                var buf: value.Builder = .{};
                for (picks) |i| try buf.appendPart(self.arena, str, bounds.items[i], bounds.items[i + 1]);
                return if (obj == .markup) .{ .markup = buf.str() } else .{ .string = buf.str() };
            },
            else => unreachable,
        }
    }
};

fn sliceBound(v: Value, ok: *bool) ?i64 {
    @setRuntimeSafety(true);
    return switch (v) {
        .none => null,
        .int => |i| i,
        .boolean => |b| @intFromBool(b),
        // Past 64 bits a bound is past either end of anything sliceable, as the i64 extreme is.
        .bigint => |t| if (t[0] == '-') std.math.minInt(i64) else std.math.maxInt(i64),
        else => {
            ok.* = false;
            return null;
        },
    };
}

fn indexOf(key: Value) ?i64 {
    @setRuntimeSafety(true);
    return switch (key) {
        .int => |i| i,
        .boolean => |b| @intFromBool(b),
        else => null,
    };
}

fn wrapIndex(i: i64, len: usize) ?usize {
    @setRuntimeSafety(true);
    const n: i64 = @intCast(len);
    const at = if (i < 0) i + n else i;
    return if (at >= 0 and at < n) @intCast(at) else null;
}

/// The indices Python's `seq[start:stop:step]` picks from a sequence of `len`.
/// Python's `slice(start, stop, step).indices(len)`: the first index and the end the walk stops at.
fn sliceEnds(len: usize, start: ?i64, stop: ?i64, step: i64) [2]i64 {
    @setRuntimeSafety(true);
    const n: i64 = @intCast(len);
    const clamp = struct {
        fn f(v: ?i64, default: i64, n_: i64, lo: i64, hi: i64) i64 {
            @setRuntimeSafety(true);
            // i128: a bound near the i64 edge plus the length must not overflow.
            var x: i128 = v orelse return default;
            if (x < 0) x += n_;
            return @intCast(@max(lo, @min(hi, x)));
        }
    }.f;
    if (step > 0) return .{ clamp(start, 0, n, 0, n), clamp(stop, n, n, 0, n) };
    return .{ clamp(start, n - 1, n, -1, n - 1), clamp(stop, -1, n, -1, n - 1) };
}

fn sliceIndices(arena: Allocator, len: usize, start: ?i64, stop: ?i64, step: i64) Failure![]const usize {
    @setRuntimeSafety(true);
    const ij = sliceEnds(len, start, stop, step);
    var out = std.ArrayList(usize).empty;
    // i128: a step near the i64 edge must not overflow the index past the end.
    var i: i128 = ij[0];
    if (step > 0) {
        while (i < ij[1]) : (i += step) try out.append(arena, @intCast(i));
    } else {
        while (i > ij[1]) : (i += step) try out.append(arena, @intCast(i));
    }
    return out.items;
}

/// The bounds of `range(b)[i:j:k]`, or null when one passes what i128 holds.
fn rangeSlice(b: [3]i128, i: i64, j: i64, k: i64) ?[3]i128 {
    @setRuntimeSafety(true);
    const start = std.math.add(i128, b[0], std.math.mul(i128, i, b[2]) catch return null) catch return null;
    const stop = std.math.add(i128, b[0], std.math.mul(i128, j, b[2]) catch return null) catch return null;
    return .{ start, stop, std.math.mul(i128, b[2], k) catch return null };
}

pub fn cpCount(s: []const u8) usize {
    @setRuntimeSafety(true);
    var n: usize = 0;
    for (s) |c| {
        if (c & 0xC0 != 0x80) n += 1;
    }
    return n;
}

/// Code points `from..to` of `s`.
pub fn cpSlice(s: []const u8, from: usize, to: usize) []const u8 {
    @setRuntimeSafety(true);
    var start: usize = s.len;
    var end: usize = s.len;
    var n: usize = 0;
    for (s, 0..) |c, i| {
        if (c & 0xC0 == 0x80) continue;
        if (n == from) start = i;
        if (n == to) {
            end = i;
            break;
        }
        n += 1;
    }
    return s[@min(start, end)..end];
}
