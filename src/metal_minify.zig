//! Metal source minifier. MLX rebuilds and hashes a custom kernel's whole source and header on every launch, so the
//! bytes handed to `mlx_fast_metal_kernel_new` are a per-token host cost. Minifying changes no token the compiler
//! sees: it drops comments, indentation, blank lines and the spaces around `{ } ; , ( ) [ ]`, and keeps every
//! preprocessor line on a line of its own with its spacing collapsed to single spaces (a macro's `F(x)` and `F (x)`
//! differ). String and character literals are copied verbatim.
const std = @import("std");

/// The minified text of `src` as a NUL-terminated copy; ends in a newline unless empty. The caller frees it.
pub fn minify(alloc: std.mem.Allocator, src: []const u8) ![:0]u8 {
    var spliced: std.ArrayList(u8) = .empty;
    defer spliced.deinit(alloc);
    try spliced.ensureTotalCapacity(alloc, src.len);
    var i: usize = 0;
    while (i < src.len) : (i += 1) {
        if (src[i] == '\\') {
            if (i + 1 < src.len and src[i + 1] == '\n') {
                i += 1;
                continue;
            }
            if (i + 2 < src.len and src[i + 1] == '\r' and src[i + 2] == '\n') {
                i += 2;
                continue;
            }
        }
        spliced.appendAssumeCapacity(src[i]);
    }
    const s = spliced.items;

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    try out.ensureTotalCapacity(alloc, s.len + 1);
    var at_line_start = true;
    var directive = false;
    var space = false;
    i = 0;
    while (i < s.len) {
        const c = s[i];
        switch (c) {
            ' ', '\t', '\r', 0x0b, 0x0c => {
                space = true;
                i += 1;
            },
            '\n' => {
                if (directive) {
                    out.appendAssumeCapacity('\n');
                    directive = false;
                }
                at_line_start = true;
                space = true;
                i += 1;
            },
            '/' => if (i + 1 < s.len and s[i + 1] == '/') {
                while (i < s.len and s[i] != '\n') i += 1;
                space = true;
            } else if (i + 1 < s.len and s[i + 1] == '*') {
                i = if (std.mem.indexOfPos(u8, s, i + 2, "*/")) |e| e + 2 else s.len;
                space = true;
            } else {
                try token(alloc, &out, c, &space, directive);
                at_line_start = false;
                i += 1;
            },
            '"', '\'' => {
                const prev = if (out.items.len > 0) out.items[out.items.len - 1] else 0;
                const separator = c == '\'' and (std.ascii.isAlphanumeric(prev) or prev == '_');
                try token(alloc, &out, c, &space, directive);
                at_line_start = false;
                i += 1;
                if (separator) continue;
                while (i < s.len and s[i] != c and s[i] != '\n') {
                    const step: usize = if (s[i] == '\\' and i + 1 < s.len and s[i + 1] != '\n') 2 else 1;
                    try out.appendSlice(alloc, s[i..@min(i + step, s.len)]);
                    i += step;
                }
                if (i < s.len and s[i] == c) {
                    try out.append(alloc, c);
                    i += 1;
                }
            },
            '#' => {
                if (at_line_start) {
                    if (out.items.len > 0 and out.items[out.items.len - 1] != '\n') try out.append(alloc, '\n');
                    directive = true;
                    space = false;
                    try out.append(alloc, '#');
                } else try token(alloc, &out, c, &space, directive);
                at_line_start = false;
                i += 1;
            },
            else => {
                try token(alloc, &out, c, &space, directive);
                at_line_start = false;
                i += 1;
            },
        }
    }
    if (out.items.len > 0 and out.items[out.items.len - 1] != '\n') try out.append(alloc, '\n');
    return out.toOwnedSliceSentinel(alloc, 0);
}

fn isPunct(c: u8) bool {
    return switch (c) {
        '{', '}', ';', ',', '(', ')', '[', ']' => true,
        else => false,
    };
}

fn isOp(c: u8) bool {
    return switch (c) {
        '+', '-', '*', '/', '%', '&', '|', '^', '~', '!', '<', '>', '=', '?', ':' => true,
        else => false,
    };
}

/// Whether the space between `last` and `c` can go without joining two tokens: a bracket or an operator on exactly
/// one side, and not into a number's exponent (`0x1E - 1` must not become the one token `0x1E-1`).
fn spaceIsOptional(items: []const u8, c: u8) bool {
    const last = items[items.len - 1];
    if (isPunct(c) or isPunct(last)) return true;
    if (isOp(c) == isOp(last)) return false;
    if ((c == '+' or c == '-') and std.mem.indexOfScalar(u8, "eEpP", last) != null) {
        var start = items.len;
        while (start > 0 and (isIdentChar(items[start - 1]) or items[start - 1] == '.')) start -= 1;
        if (start < items.len and std.ascii.isDigit(items[start])) return false;
    }
    return true;
}

fn token(alloc: std.mem.Allocator, out: *std.ArrayList(u8), c: u8, space: *bool, directive: bool) !void {
    const items = out.items;
    if (space.* and items.len > 0 and items[items.len - 1] != '\n' and (directive or !spaceIsOptional(items, c))) {
        try out.append(alloc, ' ');
    }
    space.* = false;
    try out.append(alloc, c);
}

fn expectMinified(src: []const u8, want: []const u8) !void {
    const got = try minify(std.testing.allocator, src);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings(want, got);
}

test "metal minify drops comments and indentation and joins statements" {
    try expectMinified(
        \\// header
        \\static inline float f(float a, float b) {
        \\    /* sum */
        \\    float c = a + b;   // trailing
        \\
        \\    return c * 2.0f;
        \\}
    , "static inline float f(float a,float b){float c=a+b;return c*2.0f;}\n");
}

test "metal minify keeps preprocessor lines on their own lines with collapsed spacing" {
    try expectMinified(
        \\int a;
        \\  #define  F(x)   ((x) + 1)
        \\#if  defined(FOO)  &&  1
        \\int b;
        \\#endif
        \\int c;
    , "int a;\n#define F(x) ((x) + 1)\n#if defined(FOO) && 1\nint b;\n#endif\nint c;\n");
}

test "metal minify tells a function-like macro from an object-like one" {
    try expectMinified("#define F(x) x\n#define G (x)\n", "#define F(x) x\n#define G (x)\n");
}

test "metal minify honours // inside a block comment and /* inside a line comment" {
    try expectMinified("int a; /* x // y */ int b; // p /* q\nint c; /* z */", "int a;int b;int c;\n");
}

test "metal minify splices a continuation before it reads comments and directives" {
    try expectMinified("#define M(a) \\\n    do { \\\n        a; \\\n    } while (0)\nint x;\n", "#define M(a) do { a; } while (0)\nint x;\n");
    try expectMinified("#define M 1 // note \\\n still the comment\nint y;\n", "#define M 1\nint y;\n");
    try expectMinified("int a\\\r\nb;", "int ab;\n");
}

test "metal minify copies string and character literals verbatim" {
    try expectMinified("const char* s = \"a  //  /* b  \";\nchar c = '\"'; char d = '\\'';", "const char*s=\"a  //  /* b  \";char c='\"';char d='\\'';\n");
}

test "metal minify keeps a digit separator out of the character-literal path" {
    try expectMinified("uint n = 1'000;  // k\nuint m = 2;", "uint n=1'000;uint m=2;\n");
}

test "metal minify keeps spaces that separate tokens" {
    try expectMinified("a = b - -c; d = e < < f; x = y > > z; if (a) else (b);", "a=b- -c;d=e< <f;x=y> >z;if(a)else(b);\n");
    try expectMinified("template <typename T, int N>\nstruct S { T v[N]; };", "template<typename T,int N>struct S{T v[N];};\n");
    try expectMinified("[[kernel]] void k(\n  device float* o [[buffer(0)]],\n  uint i [[thread_position_in_grid]]) {\n}", "[[kernel]]void k(device float*o[[buffer(0)]],uint i[[thread_position_in_grid]]){}\n");
}

test "metal minify joins operators to their operands but not into one token" {
    try expectMinified("x = a * -b / (c % d) ? e : f;", "x=a* -b/(c%d)?e:f;\n");
    try expectMinified("uint a = 0x1E - 1u;  float b = 1e-3f + 2.0f;  uint c = N - 1u;", "uint a=0x1E -1u;float b=1e-3f+2.0f;uint c=N-1u;\n");
    try expectMinified("a = b / *p; c = d / /* x */ e;", "a=b/ *p;c=d/e;\n");
}

test "metal minify keeps a directive after a block comment that spans lines" {
    try expectMinified("/* a\n b */ #define X 1\nint z;", "#define X 1\nint z;\n");
}

test "metal minify of an empty or blank input is empty" {
    try expectMinified("", "");
    try expectMinified("  \n\t\n// only a comment\n", "");
}

test "metal minify is idempotent" {
    const text =
        \\#define M(a) \
        \\  ((a) * 2)
        \\// c
        \\constant float t[3] = { 1.0f, 2.0f, 3.0f };
        \\inline float g(float x) { return M(x) + t[1]; }
    ;
    const once = try minify(std.testing.allocator, text);
    defer std.testing.allocator.free(once);
    try expectMinified(once, once);
}

/// `header` without the top-level helpers that neither `source` nor anything they keep refers to; `header` and
/// `source` are minified text. A definition is dropped only when it is a struct, a `using` alias or a function whose
/// name appears nowhere in the live text, and all definitions of one name (overloads, specializations) stand or fall
/// together. Anything it cannot read as such stays, and so does the whole header when its braces do not balance or a
/// macro pastes tokens.
pub fn shake(alloc: std.mem.Allocator, header: []const u8, source: []const u8) ![:0]u8 {
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const items = (try splitItems(arena, header)) orelse return alloc.dupeSentinel(u8, header, 0);

    var live: std.StringHashMapUnmanaged(void) = .empty;
    try addIdents(arena, &live, source);
    const kept = try arena.alloc(bool, items.len);
    for (items, kept) |item, *k| {
        k.* = item.name == null;
        if (k.*) try addIdents(arena, &live, item.text);
    }
    var again = true;
    while (again) {
        again = false;
        for (items, kept) |item, *k| {
            if (k.* or !live.contains(item.name.?)) continue;
            k.* = true;
            again = true;
            try addIdents(arena, &live, item.text);
        }
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    for (items, kept) |item, k| {
        if (!k) continue;
        if (item.directive and out.items.len > 0 and out.items[out.items.len - 1] != '\n') try out.append(alloc, '\n');
        try out.appendSlice(alloc, item.text);
    }
    if (out.items.len > 0 and out.items[out.items.len - 1] != '\n') try out.append(alloc, '\n');
    return out.toOwnedSliceSentinel(alloc, 0);
}

const Item = struct {
    text: []const u8,
    directive: bool,
    /// Set only for a definition that may be dropped.
    name: ?[]const u8,
};

fn isIdentStart(c: u8) bool {
    return std.ascii.isAlphabetic(c) or c == '_';
}

fn isIdentChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

/// The next identifier of `s` at or after `i.*`; numbers are skipped whole so `0x1E` or `1u` yield none.
fn nextIdent(s: []const u8, i: *usize) ?[]const u8 {
    while (i.* < s.len) {
        const c = s[i.*];
        if (std.ascii.isDigit(c)) {
            while (i.* < s.len and (isIdentChar(s[i.*]) or s[i.*] == '.')) i.* += 1;
        } else if (isIdentStart(c)) {
            const start = i.*;
            while (i.* < s.len and isIdentChar(s[i.*])) i.* += 1;
            return s[start..i.*];
        } else i.* += 1;
    }
    return null;
}

fn addIdents(arena: std.mem.Allocator, live: *std.StringHashMapUnmanaged(void), text: []const u8) !void {
    var i: usize = 0;
    while (nextIdent(text, &i)) |id| try live.put(arena, id, {});
}

/// `text` as top-level items, or null when it cannot be read as such: unbalanced braces or parentheses, or a macro
/// that pastes tokens.
fn splitItems(arena: std.mem.Allocator, text: []const u8) !?[]Item {
    var items: std.ArrayList(Item) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        if (text[i] == '\n' or text[i] == ' ') {
            i += 1;
            continue;
        }
        if (text[i] == '#') {
            const end = std.mem.indexOfScalarPos(u8, text, i, '\n') orelse text.len;
            const line = text[i..end];
            if (std.mem.indexOf(u8, line, "##") != null) return null;
            try items.append(arena, .{ .text = text[i..@min(end + 1, text.len)], .directive = true, .name = null });
            i = end + 1;
            continue;
        }
        var depth: usize = 0;
        var parens: usize = 0;
        var head_end: ?usize = null;
        var j = i;
        const end: usize = scan: while (j < text.len) : (j += 1) {
            switch (text[j]) {
                '"', '\'' => {
                    const q = text[j];
                    j += 1;
                    while (j < text.len and text[j] != q and text[j] != '\n') j += if (text[j] == '\\') 2 else 1;
                },
                '(' => parens += 1,
                ')' => {
                    if (parens == 0) return null;
                    parens -= 1;
                },
                '{' => {
                    if (depth == 0 and head_end == null) head_end = j;
                    depth += 1;
                },
                '}' => {
                    if (depth == 0) return null;
                    depth -= 1;
                    if (depth == 0 and parens == 0) {
                        const head = text[i..head_end.?];
                        const after = std.mem.indexOfNone(u8, text[j + 1 ..], " ") orelse text.len - j - 1;
                        const semi = j + 1 + after < text.len and text[j + 1 + after] == ';';
                        if (semi) break :scan j + 1 + after;
                        if (std.mem.indexOfScalar(u8, head, '=') != null or hasWord(head, "struct") or hasWord(head, "class") or hasWord(head, "enum") or hasWord(head, "union")) {
                            j = std.mem.indexOfScalarPos(u8, text, j, ';') orelse return null;
                            break :scan j;
                        }
                        break :scan j;
                    }
                },
                ';' => if (depth == 0 and parens == 0) break :scan j,
                else => {},
            }
        } else return null;
        const body = text[i .. end + 1];
        try items.append(arena, .{ .text = body, .directive = false, .name = definedName(body) });
        i = end + 1;
    }
    return items.items;
}

fn hasWord(s: []const u8, word: []const u8) bool {
    var i: usize = 0;
    while (nextIdent(s, &i)) |id| if (std.mem.eql(u8, id, word)) return true;
    return false;
}

/// The name of a struct, a `using` alias or a function definition in `item`; null for anything else, which stays.
fn definedName(item: []const u8) ?[]const u8 {
    if (std.mem.indexOfScalar(u8, item, '#') != null) return null;
    var r = item;
    if (std.mem.startsWith(u8, r, "template")) {
        var k: usize = "template".len;
        while (k < r.len and r[k] == ' ') k += 1;
        if (k >= r.len or r[k] != '<') return null;
        var angle: usize = 0;
        var parens: usize = 0;
        while (k < r.len) : (k += 1) {
            switch (r[k]) {
                '(' => parens += 1,
                ')' => parens -|= 1,
                '<' => if (parens == 0) {
                    angle += 1;
                },
                '>' => if (parens == 0) {
                    angle -= 1;
                    if (angle == 0) break;
                },
                else => {},
            }
        } else return null;
        r = r[k + 1 ..];
    }
    var i: usize = 0;
    const first = nextIdent(r, &i) orelse return null;
    if (std.mem.eql(u8, first, "struct") or std.mem.eql(u8, first, "class")) return nextIdent(r, &i);
    if (std.mem.eql(u8, first, "using")) {
        const alias = nextIdent(r, &i) orelse return null;
        if (std.mem.eql(u8, alias, "namespace")) return null;
        return alias;
    }
    if (r[r.len - 1] != '}') return null;
    const open = std.mem.indexOfScalar(u8, r, '(') orelse return null;
    if (std.mem.indexOfScalar(u8, r[0..open], '=') != null) return null;
    var name: ?[]const u8 = null;
    i = 0;
    while (i < open) {
        const id = nextIdent(r[0..open], &i) orelse break;
        if (std.mem.eql(u8, id, "operator")) return null;
        name = id;
    }
    var depth: usize = 0;
    var close = open;
    while (close < r.len) : (close += 1) {
        if (r[close] == '(') depth += 1;
        if (r[close] == ')') {
            depth -= 1;
            if (depth == 0) break;
        }
    }
    if (std.mem.indexOfScalarPos(u8, r, close, '{') == null) return null;
    const tail = std.mem.trim(u8, r[close + 1 .. std.mem.indexOfScalarPos(u8, r, close, '{').?], " ");
    if (tail.len != 0 and !std.mem.eql(u8, tail, "const")) return null;
    return name;
}

fn expectShaken(header: []const u8, source: []const u8, want: []const u8) !void {
    const got = try shake(std.testing.allocator, header, source);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings(want, got);
}

test "metal shake drops a helper nothing refers to and keeps one the source calls" {
    try expectShaken("static inline uint a(uint x){return x;}static inline uint b(uint x){return x;}", "out[0]=a(1u);", "static inline uint a(uint x){return x;}\n");
}

test "metal shake follows a kept helper to the helpers it calls, in either order" {
    try expectShaken("static inline uint c(uint x){return x;}static inline uint b(uint x){return c(x);}static inline uint a(uint x){return 0u;}", "y=b(1u);", "static inline uint c(uint x){return x;}static inline uint b(uint x){return c(x);}\n");
}

test "metal shake keeps every definition of a name: overloads, specializations and the primary template" {
    const header = "template<bool W> struct T{using type = uint;};template<> struct T<true>{using type = ulong;};static inline uint f(uint v){return v;}static inline uint f(ulong v){return uint(v);}struct Dead{uint v;};";
    try expectShaken(header, "T<true>::type v = f(1u);", "template<bool W> struct T{using type = uint;};template<> struct T<true>{using type = ulong;};static inline uint f(uint v){return v;}static inline uint f(ulong v){return uint(v);}\n");
}

test "metal shake keeps directives, everything it cannot read and what a macro names" {
    const header = "#define CALL(x) helper(x)\nstatic_assert(1 == 1, \"ok\");using namespace metal;constant uint table[2]={1u,2u};static inline uint helper(uint x){return x;}static inline uint unused(uint x){return x;}";
    try expectShaken(header, "y=CALL(1u);", "#define CALL(x) helper(x)\nstatic_assert(1 == 1, \"ok\");using namespace metal;constant uint table[2]={1u,2u};static inline uint helper(uint x){return x;}\n");
}

test "metal shake keeps an alias and a default argument's names" {
    try expectShaken("template<uint N> using bits = typename lane<N>::type;template<uint N> struct lane{using type = uint;};struct dead{uint v;};", "bits<2u> v;", "template<uint N> using bits = typename lane<N>::type;template<uint N> struct lane{using type = uint;};\n");
}

test "metal shake leaves a header whose braces do not balance, or that pastes tokens, as it is" {
    try expectShaken("#if A\nstatic inline uint a(uint x){\n#else\nstatic inline uint a(uint x){{\n#endif\nreturn x;}", "y=1;", "#if A\nstatic inline uint a(uint x){\n#else\nstatic inline uint a(uint x){{\n#endif\nreturn x;}");
    try expectShaken("#define P(a) f_##a\nstatic inline uint f_1(uint x){return x;}", "y=P(1);", "#define P(a) f_##a\nstatic inline uint f_1(uint x){return x;}");
}

test "metal shake drops nothing from a header with nothing to drop and returns an empty one empty" {
    try expectShaken("", "y=1;", "");
}
