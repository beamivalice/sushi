//! The chat-template engine: a Jinja2 template rendered as transformers renders a chat template (Python Jinja2,
//! sandboxed, `trim_blocks` and `lstrip_blocks` on), over the JSON a chat request carries.
const std = @import("std");
const lexer = @import("lexer.zig");
const parser = @import("parser.zig");
const value = @import("value.zig");
const render = @import("render.zig");
pub const ast = @import("ast.zig");
pub const Value = value.Value;
pub const Origin = value.Origin;
pub const Run = value.Run;
pub const Str = value.Str;
pub const Builder = value.Builder;
pub const fromStruct = value.fromStruct;
pub const fromJson = value.fromJson;
pub const JsonError = value.JsonError;
pub const Dict = value.Dict;
const facts_mod = @import("facts.zig");

pub const Facts = facts_mod.Facts;

pub const Limits = render.Limits;

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
        return (try self.renderSpans(arena, in, diag)).bytes;
    }

    /// The rendered prompt with the origin of each byte: the template's own text, or the context's string it came from.
    pub fn renderSpans(self: *const Template, arena: std.mem.Allocator, in: Inputs, diag: *Diagnostic) Error!Str {
        @setRuntimeSafety(true);
        return renderTree(arena, .{ .src = self.src, .body = self.body }, in, diag, null);
    }
};

/// What a render reads besides the template. Its context's values may reference the caller's strings without
/// copying them: what they point at (the request's messages and tools) outlives the render.
pub const Inputs = struct {
    context: Context,
    add_generation_prompt: bool,
    limits: Limits = .{},
    /// When set, gets the work the render took against its limits, whether it succeeded or not.
    usage: ?*Usage = null,
};

/// The variables a render reads: `messages`, `tools` when there are any, and a dict whose keys join last (bos and
/// eos tokens, thinking switches).
pub const Vars = struct { messages: Value, tools: ?Value = null, extra: Value };

/// Builds a render's `Vars` in the render's own arena, inside its memory budget, once per render, so a template that
/// changes a list changes only its own copy. A failure sets `diag`.
pub const Context = struct {
    ptr: *const anyopaque,
    build: *const fn (ptr: *const anyopaque, arena: std.mem.Allocator, diag: *Diagnostic) Error!Vars,
};

/// A context read from JSON text, as Python's `json.loads` reads it (the oracle's form): the messages and tools are
/// input, the extra context the template's.
pub const JsonContext = struct {
    messages: []const u8,
    /// Null or empty: the context has no `tools`.
    tools: ?[]const u8 = null,
    /// A JSON object.
    extra: []const u8,

    pub fn context(self: *const JsonContext) Context {
        @setRuntimeSafety(true);
        return .{ .ptr = self, .build = build };
    }

    fn build(ptr: *const anyopaque, arena: std.mem.Allocator, diag: *Diagnostic) Error!Vars {
        @setRuntimeSafety(true);
        const self: *const JsonContext = @ptrCast(@alignCast(ptr));
        const tools: ?Value = if (self.tools) |t| if (t.len > 0) try readJson(arena, t, "tools", .input, diag) else null else null;
        return .{ .messages = try readJson(arena, self.messages, "messages", .input, diag), .tools = tools, .extra = try readJson(arena, self.extra, "extra context", .template, diag) };
    }
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
    const json = JsonContext{ .messages = in.messages_json, .tools = in.tools_json, .extra = in.extra_json };
    return (try renderTree(arena, null, .{
        .context = json.context(),
        .add_generation_prompt = in.add_generation_prompt,
        .limits = in.limits,
        .usage = in.usage,
    }, diag, in.template)).bytes;
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
fn renderTree(arena: std.mem.Allocator, tree: ?Tree, in: Inputs, diag: *Diagnostic, source: ?[]const u8) Error!Str {
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

fn renderBudgeted(arena: std.mem.Allocator, given: ?Tree, in: Inputs, diag: *Diagnostic, source: ?[]const u8) Error!Str {
    @setRuntimeSafety(true);
    const tree = given orelse try parseTemplate(arena, source.?, diag);
    const src = tree.src;
    const body = tree.body;

    const vars = try in.context.build(in.context.ptr, arena, diag);
    var globals = render.Frame{ .parent = null };
    try globals.vars.put(arena, "messages", vars.messages);
    try globals.vars.put(arena, "add_generation_prompt", .{ .boolean = in.add_generation_prompt });
    if (vars.tools) |t| try globals.vars.put(arena, "tools", t);
    if (vars.extra != .dict) {
        diag.set("", null, "the extra context is not a JSON object");
        return error.TemplateRuntime;
    }
    for (vars.extra.dict.entries.items) |e| try globals.vars.put(arena, e.key.string.bytes, e.value);

    var out: value.Builder = .{};
    var r = render.Renderer{ .arena = arena, .limits = in.limits, .out = &out, .meter = .{ .max = in.limits.max_steps }, .stack_base = @frameAddress() };
    defer if (in.usage) |u| {
        u.steps = r.meter.used;
        u.largest_string = @max(r.largest_string, out.bytes.items.len);
    };
    var frame = render.Frame{ .parent = &globals };
    _ = r.renderBody(body, &frame) catch |e| {
        if (e != error.OutOfMemory) {
            // A bare failure a walk passed up through a plain `try` is named here, at the node that ran it.
            if (r.message == null) r.named(e) catch {};
            diag.set(src, r.fail_pos, r.message orelse "the render failed");
        }
        return e;
    };
    return out.str();
}

/// `v` as the template reads it (`value.fromStruct`), or a failure naming `what` in `diag` when its text is not UTF-8.
pub fn fromInput(arena: std.mem.Allocator, v: anytype, origin: Origin, comptime what: []const u8, diag: *Diagnostic) Error!value.Value {
    @setRuntimeSafety(true);
    return value.fromStruct(arena, v, origin) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidUtf8 => {
            diag.set("", null, "the " ++ what ++ " are not valid UTF-8");
            return error.TemplateRuntime;
        },
    };
}

/// `text` read as `json.loads` reads it, or a failure naming `what` in `diag` (not JSON, or nested too deep).
pub fn readJson(arena: std.mem.Allocator, text: []const u8, comptime what: []const u8, origin: Origin, diag: *Diagnostic) Error!value.Value {
    @setRuntimeSafety(true);
    return value.fromJson(arena, text, origin) catch |e| switch (e) {
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
    const jsons = [_]JsonContext{
        .{ .messages = "[{\"role\": \"user\", \"content\": \"hi\"}]", .extra = "{}" },
        .{ .messages = "[{\"role\": \"system\", \"content\": \"s\"}, {\"role\": \"user\", \"content\": \"q\"}]", .tools = "[{}]", .extra = "{}" },
    };
    const Worker = struct {
        fn run(t: *const Template, json: *const JsonContext, ok: *bool) void {
            @setRuntimeSafety(true);
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            var d = Diagnostic{};
            var fresh = Diagnostic{};
            const agp = json.tools == null;
            for (0..200) |_| {
                const got = t.render(arena.allocator(), .{ .context = json.context(), .add_generation_prompt = agp }, &d) catch return;
                const want = renderJson(arena.allocator(), .{ .template = source, .messages_json = json.messages, .tools_json = json.tools, .extra_json = json.extra, .add_generation_prompt = agp }, &fresh) catch return;
                if (!std.mem.eql(u8, got, want)) return;
            }
            ok.* = true;
        }
    };
    var oks: [8]bool = @splat(false);
    var threads: [8]std.Thread = undefined;
    for (&threads, &oks, 0..) |*t, *ok, i| t.* = try std.Thread.spawn(.{}, Worker.run, .{ &compiled, &jsons[i % jsons.len], ok });
    for (threads) |t| t.join();
    for (oks) |ok| try testing.expect(ok);
}

test "template: a template that does not compile fails at compile, with its place" {
    var diag = Diagnostic{};
    try testing.expectError(error.TemplateSyntax, Template.compile(testing.allocator, "ok\n{% for x in %}", &diag));
    try testing.expectEqual(@as(u32, 2), diag.line);
    try testing.expect(diag.message().len > 0);
}

test "template: a macro or call block that names one argument twice does not compile" {
    for ([_][]const u8{ "{% macro m(a, a) %}{% endmacro %}", "{% macro m(a, b, a=1) %}{% endmacro %}", "{% call(a, a) m() %}{% endcall %}" }) |src| {
        var diag = Diagnostic{};
        try testing.expectError(error.TemplateSyntax, Template.compile(testing.allocator, src, &diag));
        try testing.expectEqualStrings("duplicate argument 'a' in function definition", diag.message());
    }
}

test "template: a function, which Python prints with its address, is refused by name when printed" {
    for ([_][]const u8{ "{{ 'abc'.upper }}", "{{ range }}", "{{ namespace }}", "{{ [raise_exception] }}" }) |src| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        var diag = Diagnostic{};
        try testing.expectError(error.TemplateRuntime, renderForTest(arena.allocator(), src, "[]", "{}", .{}, &diag));
        try testing.expectEqualStrings("a function cannot be printed: Python prints it with its address", diag.message());
    }
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
    // Python builds the indent first, so one past the string limit is refused whether or not it is written.
    try expectLimit("{{ 5|tojson(indent=2000) }}", .{ .max_string = 1000 }, "tojson: indent 2000 is longer than the string limit");
    try expectLimit("{{ [[[[[[1]]]]]]|tojson(indent=900) }}", .{ .max_string = 1000 }, "more text than its limit");
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

test "template: the finder gives std.mem's first and last index on ordinary and adversarial text" {
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
    scanned = 0;
    try testing.expectEqual(std.mem.lastIndexOf(u8, hay, needle), last.find(hay, &scanned));
    try testing.expect(scanned <= hay.len);
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
    // a value nested near the value limit. Measured peak: about 0.9 MiB of the 16 MiB a request thread has, and 4.7 MiB
    // in a Debug build, whose frames are several times larger.
    const stack_share: usize = if (@import("builtin").mode == .debug) 2 else 4;
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
    try testing.expect(probe.peak < std.Thread.SpawnConfig.default_stack_size / stack_share);
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

/// Each byte of `s` as 't', 'a' or 'i': the template's, an assistant message's, or other input's.
fn originMask(arena: std.mem.Allocator, s: Str) ![]u8 {
    @setRuntimeSafety(true);
    const mask = try arena.alloc(u8, s.bytes.len);
    const letter = struct {
        fn of(o: Origin) u8 {
            @setRuntimeSafety(true);
            return switch (o) {
                .template => 't',
                .assistant => 'a',
                .input => 'i',
            };
        }
    }.of;
    if (s.runs.len == 0) {
        @memset(mask, letter(s.origin));
        return mask;
    }
    var start: usize = 0;
    for (s.runs) |run| {
        @memset(mask[start..run.end], letter(run.origin));
        start = run.end;
    }
    try testing.expectEqual(s.bytes.len, start);
    return mask;
}

test "template: every string builtin keeps each byte's origin, the template's and the input's, where it lands" {
    // Expected text with the input's bytes between « and »; the message's content is `c`, its strings input.
    const Case = struct { tpl: []const u8, c: []const u8 = "xy", want: []const u8, covers: []const []const u8 = &.{} };
    const cases = [_]Case{
        .{ .tpl = "{{ c }}", .want = "«xy»", .covers = &.{"output"} },
        .{ .tpl = "<{{ c }}>", .want = "<«xy»>" },
        .{ .tpl = "{{ '<' ~ c ~ '>' }}", .want = "<«xy»>", .covers = &.{"~"} },
        .{ .tpl = "{{ '<' + c + '>' }}", .want = "<«xy»>", .covers = &.{"+"} },
        .{ .tpl = "{{ c * 2 }}", .want = "«xyxy»", .covers = &.{"*"} },
        .{ .tpl = "{{ '<%s>' % c }}", .want = "<«xy»>", .covers = &.{"%"} },
        .{ .tpl = "{{ '<%s>'|format(c) }}", .want = "<«xy»>", .covers = &.{"format"} },
        .{ .tpl = "{% macro m(x) %}<{{ x }}>{% endmacro %}{{ m(c) }}", .want = "<«xy»>", .covers = &.{"macro"} },
        .{ .tpl = "{% set s %}<{{ c }}>{% endset %}{{ s }}", .want = "<«xy»>", .covers = &.{"set block"} },
        .{ .tpl = "{% filter upper %}<{{ c }}>{% endfilter %}", .want = "<«XY»>", .covers = &.{ "filter block", "upper" } },
        .{ .tpl = "{% macro w() %}<{{ caller() }}>{% endmacro %}{% call w() %}{{ c }}{% endcall %}", .want = "<«xy»>", .covers = &.{"call block"} },
        .{ .tpl = "{{ c[1:3] }}-{{ c[0] }}", .c = "wxyz", .want = "«xy»-«w»", .covers = &.{ "slice", "index" } },
        .{ .tpl = "{% for ch in c %}{{ ch }}.{% endfor %}", .want = "«x».«y».", .covers = &.{"iteration"} },
        .{ .tpl = "{{ c|trim }}-{{ c|strip }}-{{ c.strip() }}", .c = " xy ", .want = "«xy»-«xy»-«xy»", .covers = &.{ "trim", "strip", "str.strip" } },
        .{ .tpl = "-{{ c.lstrip() }}-{{ c.rstrip() }}-", .c = " xy ", .want = "-«xy »-« xy»-", .covers = &.{ "str.lstrip", "str.rstrip" } },
        .{ .tpl = "{{ c.split(',')|join('-') }}", .c = "x,y", .want = "«x»-«y»", .covers = &.{ "str.split", "join" } },
        .{ .tpl = "{{ c.rsplit(',')[0] }}-{{ c.split()[1] }}", .c = "x y,z", .want = "«x y»-«y,z»", .covers = &.{"str.rsplit"} },
        .{ .tpl = "{{ c|upper }}-{{ c.upper() }}", .want = "«XY»-«XY»", .covers = &.{"str.upper"} },
        .{ .tpl = "{{ c|lower }}-{{ c.lower() }}", .c = "Xy", .want = "«xy»-«xy»", .covers = &.{ "lower", "str.lower" } },
        .{ .tpl = "{{ c|title }}-{{ c.title() }}", .c = "xy z", .want = "«Xy Z»-«Xy Z»", .covers = &.{ "title", "str.title" } },
        .{ .tpl = "{{ c|capitalize }}-{{ c.capitalize() }}", .c = "xy z", .want = "«Xy z»-«Xy z»", .covers = &.{ "capitalize", "str.capitalize" } },
        .{ .tpl = "{{ c|replace('y', '-') }}+{{ c.replace('y', '-') }}", .c = "xyx", .want = "«x»-«x»+«x»-«x»", .covers = &.{ "replace", "str.replace" } },
        .{ .tpl = "{{ '1y1'|replace('y', c) }}-{{ '1y1'.replace('y', c) }}", .want = "1«xy»1-1«xy»1" },
        .{ .tpl = "{{ c|replace('', '-') }}", .want = "-«x»-«y»-" },
        .{ .tpl = "{{ [c, c]|join('-') }}+{{ ['1', '2']|join(c) }}", .want = "«xy»-«xy»+1«xy»2" },
        .{ .tpl = "{{ c|truncate(5, true, '..', 0) }}", .c = "abcdefgh", .want = "«abc»..", .covers = &.{"truncate"} },
        .{ .tpl = "{{ 'abcdefgh'|truncate(5, true, c, 0) }}", .want = "abc«xy»" },
        .{ .tpl = "{{ c|first }}{{ c|last }}", .want = "«xy»", .covers = &.{ "first", "last" } },
        .{ .tpl = "{{ c|reverse }}", .want = "«yx»", .covers = &.{"reverse"} },
        .{ .tpl = "{{ c|string }}{{ c|safe }}{{ c|default('-') }}{{ d|default('-') }}", .want = "«xyxyxy»-", .covers = &.{ "string", "safe", "default" } },
        .{ .tpl = "{{ c|indent(2) }}+{{ c.indent(2) }}", .c = "x\ny", .want = "«x\n  y»+«x\n  y»", .covers = &.{ "indent", "str.indent" } },
        .{ .tpl = "{{ c|tojson }}", .want = "«\"xy\"»", .covers = &.{"tojson"} },
        .{ .tpl = "{{ messages[0] }}", .want = "«{'role': 'user', 'content': 'xy'}»", .covers = &.{"repr"} },
        .{ .tpl = "{{ [c]|map('upper')|first }}", .want = "«XY»", .covers = &.{"map"} },
        .{ .tpl = "{% for k in messages[0] %}{{ k }}.{% endfor %}", .want = "«role».«content».", .covers = &.{"dict keys"} },
    };
    // Builtins that make no new string: they return an item they were given, a number, a bool, or a list.
    const no_string = [_][]const u8{
        "abs",          "float",        "int",        "items",      "length",     "list",      "sum",       "wordcount",
        "select",       "reject",       "selectattr", "rejectattr", "sort",       "unique",    "max",       "min",
        "str.startswith", "str.endswith", "str.get", "str.keys",  "str.values", "str.items", "str.append",
    };
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    for (cases) |c| {
        var text = std.ArrayList(u8).empty;
        var mask = std.ArrayList(u8).empty;
        var in_input = false;
        var i: usize = 0;
        while (i < c.want.len) {
            if (std.mem.startsWith(u8, c.want[i..], "«") or std.mem.startsWith(u8, c.want[i..], "»")) {
                in_input = std.mem.startsWith(u8, c.want[i..], "«");
                i += "«".len;
                continue;
            }
            try text.append(arena, c.want[i]);
            try mask.append(arena, if (in_input) 'i' else 't');
            i += 1;
        }
        const source = try std.mem.concat(arena, u8, &.{ "{%- set c = messages[0].content -%}", c.tpl });
        var diag = Diagnostic{};
        var compiled = try Template.compile(testing.allocator, source, &diag);
        defer compiled.deinit();
        const messages = try std.fmt.allocPrint(arena, "[{{\"role\": \"user\", \"content\": {f}}}]", .{std.json.fmt(c.c, .{})});
        const json = JsonContext{ .messages = messages, .extra = "{}" };
        const got = try compiled.renderSpans(arena, .{ .context = json.context(), .add_generation_prompt = false }, &diag);
        testing.expectEqualStrings(text.items, got.bytes) catch |e| {
            std.debug.print("case {s}\n", .{c.tpl});
            return e;
        };
        testing.expectEqualStrings(mask.items, try originMask(arena, got)) catch |e| {
            std.debug.print("case {s}\n", .{c.tpl});
            return e;
        };
    }
    // A string builtin added later gets a case here.
    const covered = struct {
        fn has(list: []const Case, extra: []const []const u8, name: []const u8) bool {
            @setRuntimeSafety(true);
            for (extra) |n| if (std.mem.eql(u8, n, name)) return true;
            for (list) |c| for (c.covers) |n| if (std.mem.eql(u8, n, name)) return true;
            return false;
        }
    }.has;
    inline for (@typeInfo(@import("builtins.zig").Filter).@"enum".field_names) |name| try testing.expect(covered(&cases, &no_string, name));
    inline for (@typeInfo(value.Method).@"enum".field_names) |name| try testing.expect(covered(&cases, &no_string, "str." ++ name));
}

test "template: strftime_now keeps the origin of the format it was handed" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag = Diagnostic{};
    var compiled = try Template.compile(testing.allocator, "{{ strftime_now(messages[0].content) }}", &diag);
    defer compiled.deinit();
    const json = JsonContext{ .messages = "[{\"role\": \"user\", \"content\": \"<x>\"}]", .extra = "{}" };
    const got = try compiled.renderSpans(arena, .{ .context = json.context(), .add_generation_prompt = false }, &diag);
    try testing.expectEqualStrings("<x>", got.bytes);
    try testing.expectEqualStrings("iii", try originMask(arena, got));
}

test "template: a context built from Zig values with text that is not UTF-8 fails by name before the render reads it" {
    const Ctx = struct {
        text: []const u8,
        fn build(ptr: *const anyopaque, arena: std.mem.Allocator, diag: *Diagnostic) Error!Vars {
            @setRuntimeSafety(true);
            const self: *const @This() = @ptrCast(@alignCast(ptr));
            const msgs = try fromInput(arena, @as([]const []const u8, &.{self.text}), .input, "messages", diag);
            return .{ .messages = msgs, .extra = try fromInput(arena, .{}, .template, "extra context", diag) };
        }
    };
    for ([_][]const u8{ "ab\xC3", "\xFF", "\x80b", "\xED\xA0\x80", "\xC0\xAF" }) |text| {
        const ctx = Ctx{ .text = text };
        var diag = Diagnostic{};
        var t = try Template.compile(testing.allocator, "{% for c in messages[0] %}{{ c }}{% endfor %}{{ messages[0][::-1] }}", &diag);
        defer t.deinit();
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        try testing.expectError(error.TemplateRuntime, t.render(arena.allocator(), .{ .context = .{ .ptr = &ctx, .build = Ctx.build }, .add_generation_prompt = false }, &diag));
        try testing.expectEqualStrings("the messages are not valid UTF-8", diag.message());
    }
}

test "template: a set block, a filter block and a generation body keep their sets to themselves, as Jinja2's inner frames" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][2][]const u8{
        .{ "{% set x %}{% set inner = 1 %}{% endset %}[{{ inner }}]", "[]" },
        .{ "{% filter upper %}{% set q = 1 %}a{% endfilter %}[{{ q }}]", "A[]" },
        .{ "{% generation %}{% set g = 1 %}G{% endgeneration %}[{{ g }}]", "G[]" },
        // The body still reads the enclosing scope, and a loop's break still leaves through a set or filter block.
        .{ "{% set o = 'o' %}{% set x %}{{ o }}{% endset %}{{ x }}", "o" },
        .{ "{% for i in [1, 2, 3] %}{% filter upper %}{{ i }}{% if i == 2 %}{% break %}{% endif %}{% endfilter %}{% endfor %}", "1" },
        .{ "{% for i in [1, 2] %}{% set x %}{% if i == 1 %}{% continue %}{% endif %}{{ i }}{% endset %}{{ x }}{% endfor %}", "2" },
    };
    for (cases) |c| {
        var diag = Diagnostic{};
        const got = renderForTest(a, c[0], "[]", "{}", .{}, &diag) catch |e| {
            std.debug.print("{s}: {s} {s}\n", .{ c[0], @errorName(e), diag.message() });
            return e;
        };
        try testing.expectEqualStrings(c[1], got);
    }
    // A generation body is a call block's: a break in it does not reach an enclosing loop.
    var diag = Diagnostic{};
    try testing.expectError(error.TemplateSyntax, renderForTest(a, "{% for i in [1] %}{% generation %}{% break %}{% endgeneration %}{% endfor %}", "[]", "{}", .{}, &diag));
    try testing.expectEqualStrings("'break' outside a for loop", diag.message());
}

test "template: a value walk that stops is named for what stopped it: the steps, the nesting, or the types it met" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Two tuples nested 600 deep, equal down to their innermost items (1 and 2).
    const deep = "{% set ns = namespace(t=1, u=2) %}{% for i in range(600) %}{% set ns.t = (ns.t,) %}{% set ns.u = (ns.u,) %}{% endfor %}{% set t = ns.t %}{% set u = ns.u %}";
    const nested = std.fmt.comptimePrint("the values nest deeper than {d}", .{value.max_depth});
    const cases = [_][2][]const u8{
        .{ deep ++ "{{ {t: 1} }}", nested },
        .{ deep ++ "{{ {}.get(t) }}", nested },
        .{ deep ++ "{{ t in {} }}", nested },
        .{ deep ++ "{{ [u, t]|sort }}", nested },
        .{ deep ++ "{{ t < u }}", nested },
        .{ deep ++ "{{ [t, u]|max }}", nested },
        .{ "{{ [1, 'a']|sort }}", "'<' not supported between instances of 'str' and 'int'" },
        .{ "{{ 18446744073709551616 is divisibleby 2 }}", "arithmetic on an integer past 64 bits is not supported" },
    };
    for (cases) |c| {
        var diag = Diagnostic{};
        if (renderForTest(a, c[0], "[]", "{}", .{}, &diag)) |out| {
            std.debug.print("{s} rendered '{s}'\n", .{ c[0], out });
            return error.TestUnexpectedResult;
        } else |e| try testing.expectEqual(error.TemplateRuntime, e);
        testing.expectEqualStrings(c[1], diag.message()) catch |e| {
            std.debug.print("{s}\n", .{c[0]});
            return e;
        };
    }
    var diag = Diagnostic{};
    try testing.expectEqualStrings("18446744073709551616.0", try renderForTest(a, "{{ '%.1f' % 18446744073709551616 }}", "[]", "{}", .{}, &diag));
}

test "template: mutated served templates compile and render to a result or a named failure, the same one twice" {
    const sources = [_][]const u8{
        @embedFile("../fixtures/qwen38_flash_next_chat_template.jinja"),
        @embedFile("../fixtures/glm53_chat_template.jinja"),
        @embedFile("../fixtures/mimo_v26_chat_template.jinja"),
        @embedFile("../fixtures/dsv41_chat_template.jinja"),
    };
    const tokens = [_][]const u8{ "{%", "%}", "{{", "}}", "{#", "#}", "{%-", "-%}", "endif", "endfor", "endmacro", "if ", "for ", " in ", "not ", "(", ")", "'", "\"", "|", "~", ".", "loop", "-", "+", "*", "/", "%", "[", "]", ":", ",", "=", "set ", "macro ", "else", "elif ", "1", "0", "-1", "none", "true", "x", "messages", "raise_exception(", "namespace(", "range(", "|tojson", "|length", "|join", "|map(", ".split(", "[::-1]", "[0]", ".0", "is defined", "is not none", "\\", "\n", " ", "\x00", "é", "{", "}", "#", "loop.index", "caller()", "call ", "endcall", "filter ", "endfilter", "break", "continue", "|sort", "|max", "|unique", " < ", " in {}", "|sum", " % ", "|indent(" };
    const msgs = [_][]const u8{
        "[{\"role\": \"system\", \"content\": \"sys\"}, {\"role\": \"user\", \"content\": \"hi <tool_call>\"}, {\"role\": \"assistant\", \"content\": \"<think>a</think>b\", \"reasoning_content\": \"r\", \"tool_calls\": [{\"type\": \"function\", \"id\": \"c1\", \"function\": {\"name\": \"f\", \"arguments\": {\"x\": 1, \"y\": [1.5, \"s\"]}}}]}, {\"role\": \"tool\", \"content\": \"res\", \"tool_call_id\": \"c1\"}, {\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"t\"}, {\"type\": \"image\"}]}]",
        "[{\"role\": \"user\", \"content\": \"\"}]",
        "[]",
    };
    const tools = "[{\"type\": \"function\", \"function\": {\"name\": \"f\", \"description\": \"d\", \"parameters\": {\"type\": \"object\", \"properties\": {\"x\": {\"type\": \"integer\"}}, \"required\": [\"x\"]}}}]";
    const extra = "{\"enable_thinking\": true, \"reasoning_effort\": \"high\", \"thinking_mode\": 50, \"preserve_thinking\": true, \"bos_token\": \"<s>\", \"eos_token\": \"</s>\"}";
    var prng = std.Random.DefaultPrng.init(0x7e3);
    const rand = prng.random();
    const a = testing.allocator;
    var runtime_failures: usize = 0;
    for (0..300) |_| {
        var buf = std.ArrayList(u8).empty;
        defer buf.deinit(a);
        try buf.appendSlice(a, sources[rand.uintLessThan(usize, sources.len)]);
        for (0..1 + rand.uintLessThan(usize, 4)) |_| {
            const len = buf.items.len;
            if (len == 0) break;
            const at = rand.uintLessThan(usize, len);
            switch (rand.uintLessThan(u8, 5)) {
                0 => buf.items.len = at,
                1 => buf.items[at] = "{}%#-+'\"()[]|.,:~ \n\\_0"[rand.uintLessThan(usize, 22)],
                2 => try buf.replaceRange(a, at, @min(len, at + 1 + rand.uintLessThan(usize, 64)) - at, ""),
                3 => {
                    const piece = try a.dupe(u8, buf.items[at..@min(len, at + 1 + rand.uintLessThan(usize, 200))]);
                    defer a.free(piece);
                    try buf.insertSlice(a, rand.uintLessThan(usize, len), piece);
                },
                else => try buf.insertSlice(a, at, tokens[rand.uintLessThan(usize, tokens.len)]),
            }
        }
        var diag = Diagnostic{};
        var t = Template.compile(a, buf.items, &diag) catch |e| {
            try testing.expectEqual(error.TemplateSyntax, e);
            try testing.expect(diag.message().len > 0);
            continue;
        };
        defer t.deinit();
        for (msgs, 0..) |m, mi| {
            var arena = std.heap.ArenaAllocator.init(a);
            defer arena.deinit();
            const json = JsonContext{ .messages = m, .tools = if (mi == 0) tools else null, .extra = extra };
            const limits = Limits{ .max_steps = 2_000_000, .max_memory = 64 << 20, .max_string = 16 << 20 };
            var d1 = Diagnostic{};
            var usage = Usage{};
            value.work = 0;
            value.work_bytes = 0;
            const r1 = t.render(arena.allocator(), .{ .context = json.context(), .add_generation_prompt = mi != 1, .limits = limits, .usage = &usage }, &d1);
            // What the meter charged covers the work done, within a constant.
            try testing.expect(value.workDone() <= 32 * usage.steps + 1024);
            var d2 = Diagnostic{};
            const r2 = t.render(arena.allocator(), .{ .context = json.context(), .add_generation_prompt = mi != 1, .limits = limits }, &d2);
            if (r1) |x| {
                try testing.expectEqualStrings(x, try r2);
                try testing.expect(std.unicode.utf8ValidateSlice(x));
            } else |e| {
                try testing.expectError(e, r2);
                try testing.expectEqualStrings(d1.message(), d2.message());
                try testing.expect(e == error.TemplateRaised or e == error.TemplateRuntime);
                if (e == error.TemplateRuntime) {
                    runtime_failures += 1;
                    testing.expect(d1.message().len > 0 and !std.mem.eql(u8, d1.message(), "the render failed")) catch |err| {
                        std.debug.print("an unnamed failure of:\n{s}\n", .{buf.items});
                        return err;
                    };
                }
            }
        }
    }
    // The corpus reaches the renderer's failures, or it proves nothing about their names.
    try testing.expect(runtime_failures > 0);
}

/// The work of rendering `template` over `messages`, against the steps its meter charged: a walk the meter does not
/// see makes the first outgrow the second.
fn workWithinSteps(a: std.mem.Allocator, template: []const u8, messages: []const u8) bool {
    @setRuntimeSafety(true);
    var diag = Diagnostic{};
    var usage = Usage{};
    value.work = 0;
    value.work_bytes = 0;
    _ = renderJson(a, .{ .template = template, .messages_json = messages, .tools_json = null, .extra_json = "{}", .add_generation_prompt = false, .usage = &usage }, &diag) catch {};
    if (value.workDone() <= 32 * usage.steps + 1024) return true;
    std.debug.print("work {d} against {d} steps: {s}\n", .{ value.workDone(), usage.steps, template });
    return false;
}

test "template: the work of a render stays within a constant of the steps its meter charges" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const msgs = "[{\"role\": \"user\", \"content\": \"x\"}]";
    const cases = [_][]const u8{
        // A case test walks the whole string each time.
        "{% set s = 'a' * 100000 %}{% for i in range(300) %}{% if s is lower %}{% endif %}{% endfor %}",
        // Each probe of a startswith tuple is work.
        "{% set t = ('b',) * 40000 %}{% for i in range(100) %}{{ 'a'.startswith(t) }}{% endfor %}",
        // The hash check walks a whole tuple on every `in` against a dict.
        "{% set t = (1,) * 20000 %}{% for i in range(200) %}{{ t in {} }}{% endfor %}",
        // Thousands of origin runs, cut a character at a time: a for loop, a slice, split, replace and % format.
        "{% set s = ([messages[0].content, '|'] * 2000)|join %}{% for c in s %}{% endfor %}",
        "{% set s = ([messages[0].content, '|'] * 2000)|join %}{{ s[::1]|length }}",
        "{% set s = ([messages[0].content, ' '] * 2000)|join %}{{ s.split()|length }}",
        "{% set s = ([messages[0].content, '|'] * 2000)|join %}{{ s.replace('', '-')|length }}",
        "{% set s = ([messages[0].content, '|'] * 2000)|join %}{{ (s ~ '%%' * 2000) % ()|length }}",
    };
    var over: usize = 0;
    for (cases) |c| over += @intFromBool(!workWithinSteps(a, c, msgs));
    try testing.expectEqual(@as(usize, 0), over);
}

/// One case of the differential harness: the template, the messages, extra context and tools as JSON, and
/// `add_generation_prompt`.
const DiffCase = struct { t: []const u8, m: []const u8, x: []const u8 = "{}", tools: ?[]const u8 = null, agp: bool = false };

// A dev tool, off unless `SUSHI_TEMPLATE_DIFF` names a JSONL file of cases: one result line per case goes to
// `<file>.out`, from case `SUSHI_TEMPLATE_DIFF_START` on, so a driver can resume after a case that crashed.
test "template differential: each case of SUSHI_TEMPLATE_DIFF rendered to one result line" {
    const in_path = std.mem.span(std.c.getenv("SUSHI_TEMPLATE_DIFF") orelse return);
    const start: usize = if (std.c.getenv("SUSHI_TEMPLATE_DIFF_START")) |s| try std.fmt.parseInt(usize, std.mem.span(s), 10) else 0;
    const gpa = std.heap.page_allocator;
    const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, in_path, gpa, .limited(1 << 30));
    defer gpa.free(text);
    const out_path = try std.fmt.allocPrint(gpa, "{s}.out", .{in_path});
    defer gpa.free(out_path);
    const file = try std.Io.Dir.createFileAbsolute(testing.io, out_path, .{});
    defer file.close(testing.io);
    var lines = std.mem.splitScalar(u8, text, '\n');
    var i: usize = 0;
    while (lines.next()) |line| : (i += 1) {
        if (line.len == 0 or i < start) continue;
        var parsed = try std.json.parseFromSlice(DiffCase, gpa, line, .{});
        defer parsed.deinit();
        const c = parsed.value;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var diag = Diagnostic{};
        var usage = Usage{};
        const rendered = renderJson(arena.allocator(), .{ .usage = &usage, .template = c.t, .messages_json = c.m, .tools_json = c.tools, .extra_json = c.x, .add_generation_prompt = c.agp }, &diag);
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        if (rendered) |out| {
            try buf.print(gpa, "{f}\n", .{std.json.fmt(.{ .i = i, .ok = true, .out = out, .steps = usage.steps, .mem = usage.memory }, .{})});
        } else |e| {
            try buf.print(gpa, "{f}\n", .{std.json.fmt(.{ .i = i, .ok = false, .kind = @errorName(e), .msg = diag.message(), .line = diag.line, .steps = usage.steps, .mem = usage.memory }, .{})});
        }
        try file.writeStreamingAll(testing.io, buf.items);
    }
}
