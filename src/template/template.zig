//! The chat-template engine: a Jinja2 template rendered as transformers renders a chat template (Python Jinja2,
//! sandboxed, `trim_blocks` and `lstrip_blocks` on), over the JSON a chat request carries.
const std = @import("std");
const lexer = @import("lexer.zig");
const parser = @import("parser.zig");
const value = @import("value.zig");
const render = @import("render.zig");
pub const ast = @import("ast.zig");
const facts_mod = @import("facts.zig");

pub const Facts = facts_mod.Facts;

pub const Limits = render.Limits;

/// The last index of `needle` in `hay`, as `std.mem.lastIndexOf` gives it, in time linear in `hay`.
pub fn lastIndexOf(gpa: std.mem.Allocator, hay: []const u8, needle: []const u8) error{OutOfMemory}!?usize {
    @setRuntimeSafety(true);
    const finder = try value.Finder.init(gpa, needle, true);
    defer finder.deinit(gpa);
    var scanned: usize = 0;
    return finder.find(hay, &scanned);
}

pub const Error = error{ TemplateSyntax, TemplateRaised, TemplateRuntime, OutOfMemory };

/// Where and why a render failed. For `TemplateRaised` the message is the template's own `raise_exception` text.
pub const Diagnostic = struct {
    /// 1-based; 0 when the failure has no place in the template (the JSON inputs).
    line: u32 = 0,
    column: u32 = 0,
    buf: [1024]u8 = undefined,
    len: usize = 0,
    /// The message was longer than `buf` and is cut at a character boundary.
    truncated: bool = false,

    pub fn message(self: *const Diagnostic) []const u8 {
        @setRuntimeSafety(true);
        return self.buf[0..self.len];
    }

    /// Whether the message is `text` whole: a cut message equals no text, so a caller matching a known raise
    /// never takes a long one for it.
    pub fn messageIs(self: *const Diagnostic, text: []const u8) bool {
        @setRuntimeSafety(true);
        return !self.truncated and std.mem.eql(u8, self.message(), text);
    }

    fn set(self: *Diagnostic, src: []const u8, pos: ?u32, text: []const u8) void {
        @setRuntimeSafety(true);
        self.truncated = text.len > self.buf.len;
        var n = @min(text.len, self.buf.len);
        // Back off a cut inside a UTF-8 sequence: the message stays valid text.
        if (self.truncated) while (n > 0 and text[n] & 0xC0 == 0x80) : (n -= 1) {};
        self.len = n;
        @memcpy(self.buf[0..n], text[0..n]);
        self.line = 0;
        self.column = 0;
        const at = pos orelse return;
        const end = @min(@as(usize, at), src.len);
        self.line = 1;
        var line_start: usize = 0;
        for (src[0..end], 0..) |c, i| if (c == '\n') {
            self.line += 1;
            line_start = i + 1;
        };
        self.column = @intCast(1 + render.cpCount(src[line_start..end]));
    }
};

/// A template parsed once, rendered any number of times from any thread: a render reads the tree and never writes
/// it. The tree lives in the template's own arena until `deinit`.
pub const Template = struct {
    arena: std.heap.ArenaAllocator,
    /// The normalized source, for the line and column of a render's failure.
    src: []const u8,
    body: []const ast.Node,
    /// What the tree says about itself, read once here (`facts.zig`).
    facts: Facts,

    pub fn compile(gpa: std.mem.Allocator, source: []const u8, diag: *Diagnostic) Error!Template {
        @setRuntimeSafety(true);
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const tree = try parseTemplate(arena.allocator(), source, diag);
        const facts = try facts_mod.of(arena.allocator(), tree.body);
        return .{ .arena = arena, .src = tree.src, .body = tree.body, .facts = facts };
    }

    pub fn deinit(self: *Template) void {
        @setRuntimeSafety(true);
        self.arena.deinit();
    }

    /// The rendered prompt, allocated in `arena` as `renderJson` allocates it.
    pub fn render(self: *const Template, arena: std.mem.Allocator, in: Inputs, diag: *Diagnostic) Error![]const u8 {
        @setRuntimeSafety(true);
        return renderTree(arena, .{ .src = self.src, .body = self.body }, in, diag, null);
    }
};

/// What a render reads besides the template.
pub const Inputs = struct {
    messages_json: []const u8,
    /// Null or empty: the context has no `tools`.
    tools_json: ?[]const u8,
    /// A JSON object whose keys join the context last (bos and eos tokens, thinking switches).
    extra_json: []const u8,
    add_generation_prompt: bool,
    limits: Limits = .{},
    /// When set, gets the work the render took against its limits, whether it succeeded or not.
    usage: ?*Usage = null,
};

pub const Input = struct {
    template: []const u8,
    messages_json: []const u8,
    /// Null or empty: the context has no `tools`.
    tools_json: ?[]const u8,
    /// A JSON object whose keys join the context last (bos and eos tokens, thinking switches).
    extra_json: []const u8,
    add_generation_prompt: bool,
    limits: Limits = .{},
    /// When set, gets the work the render took against its limits, whether it succeeded or not.
    usage: ?*Usage = null,
};

/// The work a render took, in the units of its limits.
pub const Usage = struct {
    /// Steps of the meter (`max_steps`).
    steps: u64 = 0,
    /// Bytes the render allocated (`max_memory`); nothing is freed before the render ends.
    memory: usize = 0,
    /// The largest string the render built or was about to build (`max_string`), the output included.
    largest_string: usize = 0,
};

/// The largest template text the engine reads.
pub const max_template_len = 4 << 20;

/// The rendered prompt, allocated in `arena` with everything the render built: the template is parsed for this
/// render alone, inside its memory budget. Nothing is freed or moved before the caller frees `arena` (an arena: the
/// render leaks into any other allocator).
pub fn renderJson(arena: std.mem.Allocator, in: Input, diag: *Diagnostic) Error![]const u8 {
    @setRuntimeSafety(true);
    return renderTree(arena, null, .{
        .messages_json = in.messages_json,
        .tools_json = in.tools_json,
        .extra_json = in.extra_json,
        .add_generation_prompt = in.add_generation_prompt,
        .limits = in.limits,
        .usage = in.usage,
    }, diag, in.template);
}

const Tree = struct { src: []const u8, body: []const ast.Node };

fn parseTemplate(arena: std.mem.Allocator, source: []const u8, diag: *Diagnostic) Error!Tree {
    @setRuntimeSafety(true);
    if (source.len > max_template_len) {
        diag.set("", null, "the template is larger than the engine reads");
        return error.TemplateSyntax;
    }
    if (!std.unicode.utf8ValidateSlice(source)) {
        diag.set("", null, "the template is not valid UTF-8");
        return error.TemplateSyntax;
    }
    const src = try lexer.Lexer.normalize(arena, source);
    var syntax: lexer.SyntaxError = .{ .pos = 0, .message = "" };
    const tokens = lexer.tokenize(arena, src, &syntax) catch |e| {
        if (e == error.TemplateSyntax) diag.set(src, syntax.pos, syntax.message);
        return e;
    };
    const body = parser.parse(arena, tokens, &syntax) catch |e| {
        if (e == error.TemplateSyntax) diag.set(src, syntax.pos, syntax.message);
        return e;
    };
    return .{ .src = src, .body = body };
}

/// Renders `tree`, or with `source` set, parses it first inside the same memory budget.
fn renderTree(arena: std.mem.Allocator, tree: ?Tree, in: Inputs, diag: *Diagnostic, source: ?[]const u8) Error![]const u8 {
    @setRuntimeSafety(true);
    var budget = render.Budget{ .child = arena, .limit = in.limits.max_memory };
    defer if (in.usage) |u| {
        u.memory = budget.used;
    };
    return renderBudgeted(budget.allocator(), tree, in, diag, source) catch |e| {
        if (e != error.OutOfMemory or !budget.exceeded) return e;
        diag.set("", null, "the render uses more memory than its limit");
        return error.TemplateRuntime;
    };
}

fn renderBudgeted(arena: std.mem.Allocator, given: ?Tree, in: Inputs, diag: *Diagnostic, source: ?[]const u8) Error![]const u8 {
    @setRuntimeSafety(true);
    const tree = given orelse try parseTemplate(arena, source.?, diag);
    const src = tree.src;
    const body = tree.body;

    var globals = render.Frame{ .parent = null };
    try globals.vars.put(arena, "messages", try jsonInput(arena, in.messages_json, "messages", diag));
    try globals.vars.put(arena, "add_generation_prompt", .{ .boolean = in.add_generation_prompt });
    if (in.tools_json) |t| if (t.len > 0) try globals.vars.put(arena, "tools", try jsonInput(arena, t, "tools", diag));
    const extra = try jsonInput(arena, in.extra_json, "extra context", diag);
    if (extra != .dict) {
        diag.set("", null, "the extra context is not a JSON object");
        return error.TemplateRuntime;
    }
    for (extra.dict.entries.items) |e| try globals.vars.put(arena, e.key.string, e.value);

    var out = std.ArrayList(u8).empty;
    var r = render.Renderer{ .arena = arena, .limits = in.limits, .out = &out, .meter = .{ .max = in.limits.max_steps }, .stack_base = @frameAddress() };
    defer if (in.usage) |u| {
        u.steps = r.meter.used;
        u.largest_string = @max(r.largest_string, out.items.len);
    };
    var frame = render.Frame{ .parent = &globals };
    _ = r.renderBody(body, &frame) catch |e| {
        if (e != error.OutOfMemory) {
            // A bare failure from the value walkers that ran out of steps still names the limit.
            const spent = if (r.meter.spent()) std.fmt.allocPrint(arena, "the render takes more than {d} steps", .{in.limits.max_steps}) catch "the render takes more steps than its limit" else "the render failed";
            diag.set(src, r.fail_pos, r.message orelse spent);
        }
        return e;
    };
    return out.items;
}

fn jsonInput(arena: std.mem.Allocator, text: []const u8, comptime what: []const u8, diag: *Diagnostic) Error!value.Value {
    @setRuntimeSafety(true);
    return value.fromJson(arena, text) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidJson => {
            diag.set("", null, "the " ++ what ++ " are not valid JSON");
            return error.TemplateRuntime;
        },
        error.TooDeep => {
            diag.set("", null, std.fmt.comptimePrint("the " ++ what ++ " nest deeper than {d}", .{value.max_depth}));
            return error.TemplateRuntime;
        },
    };
}

const testing = std.testing;

/// `template` rendered over `messages`; the error and its diagnostic when it fails.
pub fn renderForTest(arena: std.mem.Allocator, template: []const u8, messages: []const u8, extra: []const u8, limits: Limits, diag: *Diagnostic) Error![]const u8 {
    @setRuntimeSafety(true);
    return renderJson(arena, .{ .template = template, .messages_json = messages, .tools_json = null, .extra_json = extra, .add_generation_prompt = false, .limits = limits }, diag);
}

test "template: a template compiled once renders on many threads at once, each as a fresh compile renders" {
    // Macros, a namespace, loops and literals: everything a render reads from the tree, shared read-only.
    const source =
        \\{%- macro tag(m) -%}<{{ m.role }}>{%- endmacro -%}
        \\{%- set ns = namespace(n=0) -%}
        \\{%- for m in messages -%}{{ tag(m) }}{{ m.content|upper }}{% set ns.n = ns.n + 1 %}{%- endfor -%}
        \\[{{ ns.n }}{{ ' tools' if tools is defined }}]{% if add_generation_prompt %}<assistant>{% endif %}
    ;
    var diag = Diagnostic{};
    var compiled = try Template.compile(testing.allocator, source, &diag);
    defer compiled.deinit();
    const inputs = [_]Inputs{
        .{ .messages_json = "[{\"role\": \"user\", \"content\": \"hi\"}]", .tools_json = null, .extra_json = "{}", .add_generation_prompt = true },
        .{ .messages_json = "[{\"role\": \"system\", \"content\": \"s\"}, {\"role\": \"user\", \"content\": \"q\"}]", .tools_json = "[{}]", .extra_json = "{}", .add_generation_prompt = false },
    };
    const Worker = struct {
        fn run(t: *const Template, in: Inputs, ok: *bool) void {
            @setRuntimeSafety(true);
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            var d = Diagnostic{};
            var fresh = Diagnostic{};
            for (0..200) |_| {
                const got = t.render(arena.allocator(), in, &d) catch return;
                const want = renderJson(arena.allocator(), .{ .template = source, .messages_json = in.messages_json, .tools_json = in.tools_json, .extra_json = in.extra_json, .add_generation_prompt = in.add_generation_prompt }, &fresh) catch return;
                if (!std.mem.eql(u8, got, want)) return;
            }
            ok.* = true;
        }
    };
    var oks: [8]bool = @splat(false);
    var threads: [8]std.Thread = undefined;
    for (&threads, &oks, 0..) |*t, *ok, i| t.* = try std.Thread.spawn(.{}, Worker.run, .{ &compiled, inputs[i % inputs.len], ok });
    for (threads) |t| t.join();
    for (oks) |ok| try testing.expect(ok);
}

test "template: a template that does not compile fails at compile, with its place" {
    var diag = Diagnostic{};
    try testing.expectError(error.TemplateSyntax, Template.compile(testing.allocator, "ok\n{% for x in %}", &diag));
    try testing.expectEqual(@as(u32, 2), diag.line);
    try testing.expect(diag.message().len > 0);
}

test "template: a raise reports its text and where it was raised" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var diag = Diagnostic{};
    try testing.expectError(error.TemplateRaised, renderForTest(arena.allocator(), "ok\n  {{ raise_exception('No user query found in messages.') }}", "[]", "{}", .{}, &diag));
    try testing.expectEqualStrings("No user query found in messages.", diag.message());
    try testing.expectEqual(@as(u32, 2), diag.line);
    try testing.expectEqual(@as(u32, 6), diag.column);
}

test "template: a raise longer than the diagnostic is cut at a character and never matches a text whole" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var diag = Diagnostic{};
    // 'é' is two bytes after one 'x': a cut at the buffer's even length would split one.
    try testing.expectError(error.TemplateRaised, renderForTest(arena.allocator(), "{{ raise_exception('x' ~ 'é' * 1500) }}", "[]", "{}", .{}, &diag));
    try testing.expect(diag.truncated);
    try testing.expect(diag.message().len <= diag.buf.len);
    try testing.expect(std.unicode.utf8ValidateSlice(diag.message()));
    try testing.expect(!diag.messageIs(diag.message()));

    var short = Diagnostic{};
    try testing.expectError(error.TemplateRaised, renderForTest(arena.allocator(), "{{ raise_exception('No user query found in messages.') }}", "[]", "{}", .{}, &short));
    try testing.expect(!short.truncated);
    try testing.expect(short.messageIs("No user query found in messages."));
}

/// `head` + `unit` × `n` + `tail`.
fn repeated(arena: std.mem.Allocator, head: []const u8, unit: []const u8, n: usize, tail: []const u8) ![]const u8 {
    @setRuntimeSafety(true);
    var buf = std.ArrayList(u8).empty;
    try buf.appendSlice(arena, head);
    for (0..n) |_| try buf.appendSlice(arena, unit);
    try buf.appendSlice(arena, tail);
    return buf.items;
}

test "template: hostile template text is an error, never a crash or a hang" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const deep = 100_000;
    const cases = [_]struct { text: []const u8, want: Error }{
        .{ .text = "{{", .want = error.TemplateSyntax },
        .{ .text = "{%", .want = error.TemplateSyntax },
        .{ .text = "{#", .want = error.TemplateSyntax },
        .{ .text = "{{ 'abc }}", .want = error.TemplateSyntax },
        .{ .text = "{{ '\\x4 }}'", .want = error.TemplateSyntax },
        .{ .text = "{{ '\\", .want = error.TemplateSyntax },
        .{ .text = "{% if x %}open", .want = error.TemplateSyntax },
        .{ .text = "{% for %}{% endfor %}", .want = error.TemplateSyntax },
        .{ .text = "{% endif %}", .want = error.TemplateSyntax },
        .{ .text = "{{ ) }}", .want = error.TemplateSyntax },
        .{ .text = "{{ x[ }}", .want = error.TemplateSyntax },
        .{ .text = "{% raw %}x{% endraw %}", .want = error.TemplateSyntax },
        .{ .text = "{% include 'x' %}", .want = error.TemplateSyntax },
        .{ .text = "{{ x|nosuchfilter }}", .want = error.TemplateSyntax },
        .{ .text = "\xff\xfe{{ 1 }}", .want = error.TemplateSyntax },
        .{ .text = "{{ '\xc3' }}", .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "{{ ", "(", deep, "1 }}"), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "{{ ", "[", deep, " }}"), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "{{ ", "not ", deep, "1 }}"), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "{{ ", "-", deep, "1 }}"), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "{{ 1", " + 1", deep, " }}"), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "{{ 1", " and 1", deep, " }}"), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "{{ 'a'", "|upper", deep, " }}"), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "{{ x", ".a", deep, " }}"), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "{{ x", "[0]", deep, " }}"), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "{{ f", "()", deep, " }}"), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "{{ 1", " is defined", deep, " }}"), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "{{ 1", " if 1 else 1", deep, " }}"), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "", "{% if 1 %}", deep, ""), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "", "{% for x in y %}", deep, ""), .want = error.TemplateSyntax },
        .{ .text = try repeated(arena, "{{ '", "a", 1 << 20, "' }}"), .want = error.TemplateSyntax },
    };
    for (cases, 0..) |c, i| {
        var case_arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer case_arena.deinit();
        var diag = Diagnostic{};
        const got = renderForTest(case_arena.allocator(), c.text, "[]", "{}", .{}, &diag);
        if (got) |out| {
            // The huge literal is not an error: it renders, whole.
            if (i == cases.len - 1 and out.len == 1 << 20) continue;
            std.debug.print("hostile case {d} rendered\n", .{i});
            return error.TestUnexpectedResult;
        } else |e| if (e != c.want) {
            std.debug.print("hostile case {d}: {s} ({s})\n", .{ i, @errorName(e), diag.message() });
            return error.TestUnexpectedResult;
        }
    }
}

test "template: hostile input values are an error, never a crash" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cases = [_][]const u8{
        "[\"\xff\xfe\"]",
        "[\"\xc3\"]",
        "[{\"a\": 1",
        try repeated(arena, "", "[", 100_000, ""),
        try repeated(arena, "", "[", 100_000, "1"),
    };
    for (cases, 0..) |messages, i| {
        var case_arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer case_arena.deinit();
        var diag = Diagnostic{};
        const got = renderForTest(case_arena.allocator(), "{% for m in messages %}{{ m }}{% for c in m %}{{ c }}{% endfor %}{% endfor %}{{ messages|tojson }}", messages, "{}", .{}, &diag);
        if (got) |_| {
            std.debug.print("hostile value {d} rendered\n", .{i});
            return error.TestUnexpectedResult;
        } else |e| try testing.expectEqual(error.TemplateRuntime, e);
    }
}

fn expectLimit(text: []const u8, limits: Limits, message_part: []const u8) !void {
    @setRuntimeSafety(true);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var diag = Diagnostic{};
    try testing.expectError(error.TemplateRuntime, renderForTest(arena.allocator(), text, "[]", "{}", limits, &diag));
    if (std.mem.indexOf(u8, diag.message(), message_part) == null) {
        std.debug.print("limit message: {s}\n", .{diag.message()});
        return error.TestUnexpectedResult;
    }
}

test "template: a render past its recursion, output, step or memory limit fails by name" {
    try expectLimit("{% macro r(n) %}{{ r(n + 1) }}{% endmacro %}{{ r(0) }}", .{}, "nest deeper than 100");
    try expectLimit("{% for i in range(2000) %}x{% endfor %}", .{ .max_string = 1000 }, "more text than its limit");
    try expectLimit("{{ 'ab' * 600000000 }}", .{}, "more text than its limit");
    try expectLimit("{% for x in range(1000000000) %}{% endfor %}", .{}, "Range too big");
    // `range` stops at 100000 items, so only nested loops reach the step budget.
    try expectLimit("{% for i in range(100000) %}{% for j in range(100000) %}{% endfor %}{% endfor %}", .{ .max_steps = 1_000_000 }, "more than 1000000 steps");
    try expectLimit("{% for i in range(100000) %}{% set xs = range(100000) %}{% endfor %}", .{ .max_memory = 64 << 20 }, "more memory than its limit");
}

test "template: a loop's iterations reuse their scope, so memory follows what the render keeps" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var diag = Diagnostic{};
    const out = try renderForTest(arena.allocator(), "{% for i in range(10000) %}{% for j in range(100) %}{% set k = j %}{% endfor %}{% endfor %}ok", "[]", "{}", .{ .max_memory = 64 << 20 }, &diag);
    try testing.expectEqualStrings("ok", out);
}

test "template: a macro kept past the scope that defined it renders, never reads a dead frame" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var diag = Diagnostic{};
    const in_macro = try renderForTest(arena.allocator(), "{% set ns = namespace() %}{% macro outer() %}{% macro inner(a) %}<{{ a }}>{% endmacro %}{% set ns.f = inner %}{% endmacro %}{{ outer() }}{{ ns.f(1) }}", "[]", "{}", .{}, &diag);
    try testing.expectEqualStrings("<1>", in_macro);
    // Python prints its own sentinel here; the engine only has to stay sound.
    _ = try renderForTest(arena.allocator(), "{% set ns = namespace() %}{% for i in [1, 2] %}{% macro m() %}{{ i }}{% endmacro %}{% set ns.f = m %}{% endfor %}{{ ns.f() }}{% for i in [3] %}{% set x = [i] * 100 %}{% endfor %}{{ ns.f() }}", "[]", "{}", .{}, &diag);
}

test "template: arithmetic past 64 bits is a runtime error, never a wrong number" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var diag = Diagnostic{};
    try testing.expectEqualStrings("18446744073709551616", try renderForTest(arena.allocator(), "{{ x }}", "[]", "{\"x\": 18446744073709551616}", .{}, &diag));
    try testing.expectError(error.TemplateRuntime, renderForTest(arena.allocator(), "{{ x + 1 }}", "[]", "{\"x\": 18446744073709551616}", .{}, &diag));
    try testing.expectError(error.TemplateRuntime, renderForTest(arena.allocator(), "{{ 9223372036854775807 + 1 }}", "[]", "{}", .{}, &diag));
    try testing.expectError(error.TemplateRuntime, renderForTest(arena.allocator(), "{{ '99999999999999999999'|int }}", "[]", "{}", .{}, &diag));
}

test "template: integers at the i64 edges in slices, ranges, counts and lengths never abort a render" {
    const texts = [_][]const u8{
        "{{ 'abc'[1::9223372036854775807] }}",
        "{{ 'abc'[::-9223372036854775807 - 1] }}",
        "{{ [1, 2, 3][-9223372036854775807 - 1:9223372036854775807:9223372036854775807] }}",
        "{{ 'abc'|truncate(9223372036854775807) }}",
        "{{ 'abcdefghij'|truncate(5, leeway=9223372036854775807) }}",
        "{{ range(-9223372036854775807 - 1, 9223372036854775807, 9223372036854775807)|list }}",
        "{{ range(9223372036854775807, -9223372036854775807 - 1, -9223372036854775807 - 1)|list }}",
        "{{ 'a,b,c'.split(',', -9223372036854775807 - 1) }}{{ 'a b c'.split(None, -9223372036854775807 - 1) }}",
        "{{ 'a,b,c'.rsplit(',', -9223372036854775807 - 1) }}{{ ' a b '.rsplit(None, -9223372036854775807 - 1) }}",
        "{{ 'aaa'.replace('a', 'b', -9223372036854775807 - 1) }}{{ 'ab'.replace('', '-', -9223372036854775807 - 1) }}",
        "{{ 'aaa'|replace('a', 'b', -9223372036854775807 - 1) }}{{ 'ab'|replace('', '-', -9223372036854775807 - 1) }}",
        "{{ 'abc'.startswith('a', -9223372036854775807 - 1, 9223372036854775807) }}{{ 'abc'[-9223372036854775807 - 1] }}",
        "{{ -(-9223372036854775807 - 1) }}",
        "{{ (-9223372036854775807 - 1)|abs }}",
        "{{ (-9223372036854775807 - 1) % -1 }}{{ (-9223372036854775807 - 1) * -1 }}",
        "{{ 'x'|indent(9223372036854775807) }}",
        "{{ 'x' * 9223372036854775807 }}",
        "{{ [1] * 9223372036854775807 }}",
        "{{ 0 / 9007199254740993 }}{{ 0 / -9007199254740993 }}{{ (-9223372036854775807 - 1) / 9007199254740993 }}{{ 1 / (-9223372036854775807 - 1) }}",
        "{{ (-9223372036854775807 - 1) / (-9223372036854775807 - 1) }}{{ 9223372036854775807 / 1 }}{{ 1 / 9223372036854775807 }}",
        "{{ 9223372036854775808 < 9223372036854775808.0 }}{{ -9223372036854775809 < -9223372036854775808.0 }}{{ 9223372036854775808 == 5e-324 }}",
        "{{ 99999999999999999999999 > 1.7976931348623157e308 }}{{ -99999999999999999999999 < -1.7976931348623157e308 }}{{ 9223372036854775808 > -0.0 }}",
        "{{ (-9223372036854775807 - 1) < -9.223372036854775808e18 }}{{ 9223372036854775807 < 9.223372036854775807e18 }}{{ 0 == -0.0 }}{{ 0 < 5e-324 }}",
        "{{ range(-9223372036854775807 - 1, 9223372036854775807, 9223372036854775807)[::-9223372036854775807 - 1] }}",
        "{{ range(-9223372036854775807 - 1, 9223372036854775807, 9223372036854775807)[1:][1:][::9223372036854775807] }}{{ range(5)[9223372036854775807:] }}",
        "{% for i in range(9223372036854775807, -9223372036854775807 - 1, -9223372036854775807 - 1) %}{{ loop.nextitem }}{{ loop.previtem }}{% endfor %}",
        "{{ 0x0 }}{{ 0x00000000000000000000000001 }}{{ 0b0 }}{{ 0xFFFFFFFFFFFFFFFF }}{{ -0x8000000000000001 }}",
        "{{ '9223372036854775807'|int }}{{ (-9223372036854775807 - 1)|int }}{{ 9.3e18|int }}{{ -9.3e18|int(1) }}",
    };
    for (texts) |text| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        var diag = Diagnostic{};
        _ = renderForTest(arena.allocator(), text, "[]", "{}", .{}, &diag) catch |e| try testing.expectEqual(error.TemplateRuntime, e);
    }
}

/// `text` over `extra` renders `want` in at most `max_steps` steps of the meter: work the engine charges as it
/// does it, so a bound linear in the text is a bound on the time.
fn expectSteps(text: []const u8, extra: []const u8, want: []const u8, max_steps: u64) !void {
    @setRuntimeSafety(true);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var diag = Diagnostic{};
    var usage = Usage{};
    const got = try renderJson(arena.allocator(), .{ .template = text, .messages_json = "[]", .tools_json = null, .extra_json = extra, .add_generation_prompt = false, .usage = &usage }, &diag);
    try testing.expectEqualStrings(want, got);
    if (usage.steps > max_steps) {
        std.debug.print("{d} steps, past {d}: {s}\n", .{ usage.steps, max_steps, text[0..@min(text.len, 120)] });
        return error.TestUnexpectedResult;
    }
}

const step_limit = "the render takes more than 20000000 steps";

test "template: repeating an empty string or list any number of times is immediate" {
    try expectSteps("{{ ('' * 20000000000)|length }} {{ ([] * 20000000000)|length }}", "{}", "0 0", 16);
    try expectSteps("{{ ('' * 9223372036854775807)|length }} {{ ([] * 9223372036854775807)|length }} {{ (() * 9223372036854775807)|length }}", "{}", "0 0 0", 16);
}

test "template: the finder and lastIndexOf give std.mem's first and last index on ordinary and adversarial text" {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rand = prng.random();
    var hay_buf: [64]u8 = undefined;
    var needle_buf: [12]u8 = undefined;
    const fixed = [_][2][]const u8{
        .{ "", "" },           .{ "abc", "" },     .{ "", "a" },          .{ "a", "ab" },
        .{ "aaaaab", "aab" },  .{ "abababa", "aba" }, .{ "abcabcabd", "abcabd" }, .{ "xyz", "xyz" },
        .{ "aaaa", "aa" },     .{ "<think>z</think>x", "</think>" },
        .{ "<|user|>hi<|assistant|><think></think>Hello, wörld<|user|>again<|assistant|><think></think>Hello, wörld", "Hello, wörld" },
        .{ "<|im_start|>assistant\n<think>\n\n</think>\n\nPartial ans", "Partial ans" },
        .{ "<|im_start|>assistant\nPartial", "Partial answer" },
    };
    for (fixed) |c| try expectFinderAgrees(c[0], c[1]);
    for (0..20000) |_| {
        // A two-letter alphabet makes the partial matches a naive search is slow on.
        const hay = hay_buf[0..rand.uintAtMost(usize, hay_buf.len)];
        const needle = needle_buf[0..rand.uintAtMost(usize, needle_buf.len)];
        for (hay) |*c| c.* = 'a' + rand.uintAtMost(u8, 1);
        for (needle) |*c| c.* = 'a' + rand.uintAtMost(u8, 1);
        try expectFinderAgrees(hay, needle);
    }
}

fn expectFinderAgrees(hay: []const u8, needle: []const u8) !void {
    @setRuntimeSafety(true);
    var scanned: usize = 0;
    const first = try value.Finder.init(testing.allocator, needle, false);
    defer first.deinit(testing.allocator);
    try testing.expectEqual(std.mem.indexOf(u8, hay, needle), first.find(hay, &scanned));
    try testing.expect(scanned <= hay.len);
    const last = try value.Finder.init(testing.allocator, needle, true);
    defer last.deinit(testing.allocator);
    _ = last.find(hay, &scanned);
    try testing.expect(scanned <= hay.len);
    try testing.expectEqual(std.mem.lastIndexOf(u8, hay, needle), try lastIndexOf(testing.allocator, hay, needle));
}

test "template: a loop that breaks early pays for the items it walked, never the whole list" {
    // As Python's lazy iteration: a scan per message that stops at its neighbour stays linear in the messages.
    try expectSteps("{% set xs = range(10000)|list %}{% for i in range(3000) %}{% for j in range(i, 10000) %}{% break %}{% endfor %}" ++
        "{% for x in xs %}{% break %}{% endfor %}{% endfor %}ok", "{}", "ok", 100_000);
}

test "template: substring search, strip(chars) and long comparisons are metered by the bytes they read" {
    // 2.4 MB of text is 37500 steps a pass: each case stays within a few passes, where a search that rereads
    // the needle per position would need billions.
    const setup = "{% set h = 'a' * 1600000 %}{% set n = 'a' * 800000 ~ 'b' %}";
    try expectSteps(setup ++ "{{ n in h }}", "{}", "False", 100_000);
    try expectSteps(setup ++ "{{ h.split(n)|length }} {{ h.rsplit(n)|length }}", "{}", "1 1", 200_000);
    try expectSteps(setup ++ "{{ h.replace(n, '')|length }} {{ (h|replace(n, ''))|length }}", "{}", "1600000 1600000", 300_000);
    try expectSteps("{{ ('a' * 1000000).strip('b' * 100000 ~ 'a')|length }} {{ (('a' * 1000000)|trim('b' * 100000 ~ 'a'))|length }}", "{}", "0 0", 150_000);
    const twins = "{% set x = 'a' * 50000000 %}{% set y = 'a' * 49999999 ~ 'a' %}";
    try expectLimit(twins ++ "{% for i in range(5000) %}{% if x < y %}{% endif %}{% endfor %}", .{}, step_limit);
    try expectLimit(twins ++ "{% for i in range(5000) %}{% if x == y %}{% endif %}{% endfor %}", .{}, step_limit);
    try expectLimit(twins ++ "{% set d = {x: 1} %}{% for i in range(5000) %}{% set t = d[y] %}{% endfor %}", .{}, step_limit);
}

test "template: printing a generator fails by naming it, since Python prints its address" {
    try expectLimit("{{ [1, 2]|map('string') }}", .{}, "a generator cannot be printed");
    try expectLimit("{{ [[1]|reverse] }}", .{}, "a list_reverseiterator cannot be printed");
}

test "template: a filter that applies a filter nests against the call depth" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    try expectLimit(try repeated(arena_state.allocator(), "{{ 'a'|map(", "'map', ", 100_000, "'upper')|list }}"), .{}, "nest deeper than 100");
}

test "template: work inside comparisons, lookups, sorts and string scans is metered by the step budget" {
    const doubling = "{% set ns = namespace(x=1, y=1) %}{% for i in range(60) %}{% set ns.x = [ns.x, ns.x] %}{% set ns.y = [ns.y, ns.y] %}{% endfor %}";
    try expectLimit(doubling ++ "{{ ns.x == ns.y }}", .{}, step_limit);
    try expectLimit(doubling ++ "{{ ns.x < ns.y }}", .{}, step_limit);
    try expectLimit(doubling ++ "{{ ns.x }}", .{}, step_limit);
    try expectLimit(doubling ++ "{{ ns.x|tojson }}", .{}, step_limit);
    try expectLimit(doubling ++ "{{ ns.x in [ns.y] }}", .{}, step_limit);
    try expectLimit("{% set xs = range(100000)|list %}{% for i in range(100000) %}{% if -1 in xs %}{% endif %}{% endfor %}", .{}, step_limit);
    try expectLimit("{{ range(100000)|unique|list|length }}", .{}, step_limit);
    try expectLimit("{% set s = 'a' * 10000000 %}{% for i in range(100000) %}{% if 'b' in s %}{% endif %}{% endfor %}", .{}, step_limit);
    try expectLimit("{% set s = 'a' * 10000000 %}{% for i in range(100000) %}{% set t = s|wordcount %}{% endfor %}", .{}, step_limit);
    try expectLimit("{% set s = 'a' * 10000000 %}{% for i in range(100000) %}{% set t = s[-1] %}{% endfor %}", .{}, step_limit);
    // An O(n log n) sort: a hundred thousand items sort well inside the budget.
    // Two sorts of 100000 items: about n log2 n = 1.7M comparisons each.
    try expectSteps("{{ range(100000)|reverse|sort|first }} {{ range(100000)|reverse|list|sort(reverse=true)|first }}", "{}", "0 99999", 8_000_000);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var keys = std.ArrayList(u8).empty;
    try keys.appendSlice(arena.allocator(), "{\"d\": {");
    var i: usize = 100_000;
    while (i > 0) : (i -= 1) try keys.print(arena.allocator(), "\"k{d:0>6}\": 0{s}", .{ i - 1, if (i > 1) "," else "" });
    try keys.appendSlice(arena.allocator(), "}}");
    try expectSteps("{{ (d|tojson(sort_keys=true))[:13] }}", keys.items, "{\"k000000\": 0", 2_000_000);
}

const StackProbe = struct {
    text: []const u8,
    extra: []const u8,
    result: Error![]const u8 = "",
    peak: usize = 0,

    fn run(self: *StackProbe, arena: std.mem.Allocator) void {
        @setRuntimeSafety(true);
        const top = @frameAddress();
        value.stack_low = std.math.maxInt(usize);
        var diag = Diagnostic{};
        self.result = renderForTest(arena, self.text, "[]", self.extra, .{}, &diag);
        self.peak = top - value.stack_low;
    }
};

test "template: a render at every depth limit at once returns on a request thread's stack" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // 100 macro calls (the whole call depth), and at the bottom an expression nested to the parser's limit printing
    // a value nested near the value limit. Measured peak: about 0.9 MiB of the 16 MiB a request thread has.
    const deep_value = try repeated(arena, "{\"d\": ", "[", 500, try repeated(arena, "1", "]", 500, "}"));
    var nesting: usize = parser.max_nesting;
    const probe_text = while (nesting > 0) : (nesting -= 1) {
        const text = try repeated(arena, "{% macro m(n) %}{% if n > 0 %}{{ m(n - 1) }}{% else %}{{ ", "not ", nesting, "(d|tojson|length) }}{{ d|string|length }}{% endif %}{% endmacro %}{{ m(99) }}");
        var diag = Diagnostic{};
        _ = renderForTest(arena, text, "[]", "{\"d\": 1}", .{}, &diag) catch |e| if (e == error.TemplateSyntax) continue;
        break text;
    } else unreachable;
    try testing.expect(nesting > parser.max_nesting - 10);

    var probe = StackProbe{ .text = probe_text, .extra = deep_value };
    const thread = try std.Thread.spawn(.{ .stack_size = std.Thread.SpawnConfig.default_stack_size }, StackProbe.run, .{ &probe, arena });
    thread.join();
    _ = try probe.result;
    try testing.expect(probe.peak < std.Thread.SpawnConfig.default_stack_size / 4);
}

test "template: an input nested past the value limit is named as that, not as invalid JSON" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var diag = Diagnostic{};
    const deep = try repeated(arena.allocator(), "", "[", 600, try repeated(arena.allocator(), "", "]", 600, ""));
    try testing.expectError(error.TemplateRuntime, renderForTest(arena.allocator(), "x", deep, "{}", .{}, &diag));
    try testing.expectEqualStrings("the messages nest deeper than 512", diag.message());
    try testing.expectError(error.TemplateRuntime, renderForTest(arena.allocator(), "x", "[1,", "{}", .{}, &diag));
    try testing.expectEqualStrings("the messages are not valid JSON", diag.message());
}

/// An allocator that poisons what is freed and keeps it, so a slice read after its free reads garbage.
const PoisonOnFree = struct {
    child: std.mem.Allocator,

    fn allocator(self: *PoisonOnFree) std.mem.Allocator {
        @setRuntimeSafety(true);
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        @setRuntimeSafety(true);
        const self: *PoisonOnFree = @ptrCast(@alignCast(ctx));
        return self.child.rawAlloc(len, a, ra);
    }
    fn resize(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) bool {
        @setRuntimeSafety(true);
        const self: *PoisonOnFree = @ptrCast(@alignCast(ctx));
        return self.child.rawResize(m, a, n, ra);
    }
    fn remap(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        @setRuntimeSafety(true);
        // A moving allocator: every growth moves, and the old place is freed (poisoned).
        const self: *PoisonOnFree = @ptrCast(@alignCast(ctx));
        if (n <= m.len) return self.child.rawRemap(m, a, n, ra);
        const p = self.child.rawAlloc(n, a, ra) orelse return null;
        @memcpy(p[0..m.len], m);
        free(ctx, m, a, ra);
        return p;
    }
    fn free(_: *anyopaque, m: []u8, _: std.mem.Alignment, _: usize) void {
        @setRuntimeSafety(true);
        @memset(m, 0);
    }
};

test "template: a render never frees or moves what it allocated, so a list appended while iterated stays sound" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var poison = PoisonOnFree{ .child = arena.allocator() };
    var diag = Diagnostic{};
    const text = "{% set xs = [1, 2] %}{% for x in xs %}{{ x }}{% for i in range(40) %}{% set _ = xs.append(i) %}{% endfor %}{% endfor %}{{ xs|length }}";
    try testing.expectEqualStrings("1282", try renderForTest(poison.allocator(), text, "[]", "{}", .{}, &diag));
}

test "template: nesting inside every level of the call depth fails by name, never overflows a thread's stack" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const shapes = [_][]const u8{
        try repeated(arena, "{% macro m(n) %}{% if n > 0 %}{{ ", "not ", 100, "m(n - 1) }}{% endif %}{% endmacro %}{{ m(99) }}"),
        try repeated(arena, "{% macro m(n) %}{% if n > 0 %}", "{% if true %}", 150, try repeated(arena, "{{ m(n - 1) }}", "{% endif %}", 150, "{% endif %}{% endmacro %}{{ m(99) }}")),
        try repeated(arena, "{% macro m(n) %}{% if n > 0 %}{{ ", "'a'|replace('a', ", 60, try repeated(arena, "m(n - 1)", ")", 60, " }}{% endif %}{% endmacro %}{{ m(99) }}")),
        try repeated(arena, "{% macro m(n) %}{% if n > 0 %}{{ m(n - 1)", "|string", 150, " }}{% endif %}{% endmacro %}{{ m(99) }}"),
    };
    // 16 MiB: what a connection thread gets; 8 MiB: the main thread the CLI and kld render on.
    for ([_]usize{ std.Thread.SpawnConfig.default_stack_size, 8 << 20 }) |stack| for (shapes) |text| {
        var probe = StackProbe{ .text = text, .extra = "{}" };
        const thread = try std.Thread.spawn(.{ .stack_size = stack }, StackProbe.run, .{ &probe, arena });
        thread.join();
        _ = probe.result catch |e| try testing.expectEqual(error.TemplateRuntime, e);
    };
}

test "template: a sort whose keys hold a NaN fails by name, whatever its length" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var xs = std.ArrayList(u8).empty;
    try xs.appendSlice(arena.allocator(), "{\"xs\": [");
    for (0..1200) |i| try xs.print(arena.allocator(), "{s}\"{s}\"", .{ if (i > 0) "," else "", if (i % 3 == 0) "nan" else try std.fmt.allocPrint(arena.allocator(), "{d}", .{i * 7919 % 101}) });
    try xs.appendSlice(arena.allocator(), "]}");
    var diag = Diagnostic{};
    try testing.expectError(error.TemplateRuntime, renderForTest(arena.allocator(), "{{ xs|map('float')|sort|length }}", "[]", xs.items, .{}, &diag));
    try testing.expectEqualStrings("sort: a NaN cannot be ordered", diag.message());
    try testing.expectError(error.TemplateRuntime, renderForTest(arena.allocator(), "{{ {1.5: 1, (1e308 * 10 - 1e308 * 10): 2}|tojson(sort_keys=true) }}", "[]", "{}", .{}, &diag));
}
