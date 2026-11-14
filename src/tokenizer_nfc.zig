//! Unicode NFC (UAX #15): canonical decomposition, canonical ordering, canonical composition. The tables are
//! `tokenizer_nfc_tables.zig`, generated with the Unicode version of the category tables.
const std = @import("std");
const tables = @import("tokenizer_nfc_tables.zig");

const s_base: u21 = 0xAC00;
const l_base: u21 = 0x1100;
const v_base: u21 = 0x1161;
const t_base: u21 = 0x11A7;
const l_count = 19;
const v_count = 21;
const t_count = 28;
const n_count = v_count * t_count;
const s_count = l_count * n_count;

/// `text` in NFC, or null when it already is. Invalid UTF-8 is left as it is: only a valid text is normalized.
pub fn normalize(allocator: std.mem.Allocator, text: []const u8) !?[]u8 {
    for (text) |b| {
        if (b >= 0x80) break;
    } else return null;
    if (!std.unicode.utf8ValidateSlice(text)) return null;

    var cps = std.ArrayList(u21).empty;
    defer cps.deinit(allocator);
    var it = std.unicode.Utf8View.initUnchecked(text).iterator();
    while (it.nextCodepoint()) |cp| try decompose(allocator, &cps, cp);
    reorder(cps.items);
    const n = compose(cps.items);

    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    for (cps.items[0..n]) |cp| {
        var buf: [4]u8 = undefined;
        const len = std.unicode.utf8Encode(cp, &buf) catch unreachable;
        try out.appendSlice(allocator, buf[0..len]);
    }
    if (std.mem.eql(u8, out.items, text)) {
        out.deinit(allocator);
        return null;
    }
    return try out.toOwnedSlice(allocator);
}

fn decompose(allocator: std.mem.Allocator, out: *std.ArrayList(u21), cp: u21) !void {
    if (cp >= s_base and cp < s_base + s_count) {
        const s = cp - s_base;
        try out.append(allocator, l_base + s / n_count);
        try out.append(allocator, v_base + (s % n_count) / t_count);
        if (s % t_count != 0) try out.append(allocator, t_base + s % t_count);
        return;
    }
    const i = std.sort.binarySearch(tables.Decomp, &tables.decomp, cp, struct {
        fn order(c: u21, d: tables.Decomp) std.math.Order {
            return std.math.order(c, d.cp);
        }
    }.order) orelse return out.append(allocator, cp);
    const d = tables.decomp[i];
    try out.appendSlice(allocator, tables.decomp_cps[d.at..][0..d.len]);
}

fn combiningClass(cp: u21) u8 {
    if (cp < 0x300) return 0;
    const i = std.sort.binarySearch(tables.Ccc, &tables.ccc, cp, struct {
        fn order(c: u21, e: tables.Ccc) std.math.Order {
            return std.math.order(c, e.cp);
        }
    }.order) orelse return 0;
    return tables.ccc[i].class;
}

/// Canonical ordering: a stable sort by class of each run of non-starters.
fn reorder(cps: []u21) void {
    var i: usize = 1;
    while (i < cps.len) : (i += 1) {
        const c = combiningClass(cps[i]);
        if (c == 0) continue;
        var j = i;
        while (j > 0) : (j -= 1) {
            const p = combiningClass(cps[j - 1]);
            if (p == 0 or p <= c) break;
            std.mem.swap(u21, &cps[j - 1], &cps[j]);
        }
    }
}

fn composite(first: u21, second: u21) ?u21 {
    if (first >= l_base and first < l_base + l_count and second >= v_base and second < v_base + v_count)
        return s_base + ((first - l_base) * v_count + (second - v_base)) * t_count;
    if (first >= s_base and first < s_base + s_count and (first - s_base) % t_count == 0 and second > t_base and second < t_base + t_count)
        return first + (second - t_base);
    const i = std.sort.binarySearch(tables.Pair, &tables.pairs, [2]u21{ first, second }, struct {
        fn order(k: [2]u21, p: tables.Pair) std.math.Order {
            const o = std.math.order(k[0], p.first);
            return if (o != .eq) o else std.math.order(k[1], p.second);
        }
    }.order) orelse return null;
    return tables.pairs[i].composite;
}

/// Canonical composition in place; returns the new length.
fn compose(cps: []u21) usize {
    if (cps.len == 0) return 0;
    var starter: ?usize = if (combiningClass(cps[0]) == 0) 0 else null;
    var last_class: u16 = if (starter == null) 256 else 0;
    var n: usize = 1;
    for (cps[1..]) |cp| {
        const class = combiningClass(cp);
        if (starter) |s| if (last_class < class or last_class == 0) if (composite(cps[s], cp)) |c| {
            cps[s] = c;
            continue;
        };
        if (class == 0) starter = n;
        last_class = class;
        cps[n] = cp;
        n += 1;
    }
    return n;
}

test "nfc: every oracle case composes as Python's unicodedata does" {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice([]const [2][]const u8, a, @embedFile("fixtures/nfc_cases.json"), .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.len > 1000);
    for (parsed.value) |c| {
        const got = (try normalize(a, c[0])) orelse return error.NotNormalized;
        defer a.free(got);
        try std.testing.expectEqualStrings(c[1], got);
        try std.testing.expect((try normalize(a, c[1])) == null);
    }
}

test "nfc: ASCII, composed text and invalid UTF-8 are left as they are" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "", "plain ascii", "caf\u{e9} \u{d55c}\u{ad6d}\u{c5b4}", "bad \xff byte e\u{301}" }) |t| try std.testing.expect((try normalize(a, t)) == null);
}
