//! Restore tape: a JSONL record of what each prefix-cache lookup restored, what the SSD tier held after
//! each flush, and what each request generated. Two runs of the same flow on equal code write equal
//! tapes, so a diff between a baseline tape and a changed tree names the first thing that moved.
//! `SUSHI_RESTORE_TAPE=<abs file>` turns it on; off, nothing here runs. Host tier: no MLX array.

const std = @import("std");
const disk_format = @import("disk_format.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Hex = [Sha256.digest_length * 2]u8;

pub fn hex(digest: [Sha256.digest_length]u8) Hex {
    return std.fmt.bytesToHex(digest, .lower);
}

/// SHA-256 of token ids as little-endian u32.
pub fn idsHex(ids: []const u32) Hex {
    var h = Sha256.init(.{});
    for (ids) |t| h.update(&std.mem.toBytes(std.mem.nativeToLittle(u32, t)));
    return hex(h.finalResult());
}

pub var path_override: ?[]const u8 = null;
var path_cached: ?[]const u8 = null;
var path_read: bool = false;
var seq: std.atomic.Value(u64) = .init(0);
var append_mu: std.Io.Mutex = .init;

pub fn path() ?[]const u8 {
    if (path_override) |p| return if (p.len == 0) null else p;
    if (!path_read) {
        path_cached = pathFromEnv(std.c.getenv("SUSHI_RESTORE_TAPE"));
        path_read = true;
    }
    return path_cached;
}

pub fn pathFromEnv(raw: ?[*:0]const u8) ?[]const u8 {
    const v = raw orelse return null;
    if (v[0] == 0 or v[0] == '0') return null;
    return std.mem.span(v);
}

pub fn on() bool {
    return path() != null;
}

pub fn nextSeq() u64 {
    return seq.fetchAdd(1, .monotonic) + 1;
}

/// Appends one finished line (with its newline) to the tape file.
pub fn append(line: []const u8) void {
    const p = path() orelse return;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const z = std.fmt.bufPrintSentinel(&buf, "{s}", .{p}, 0) catch return;
    const io = std.Io.Threaded.global_single_threaded.io();
    append_mu.lockUncancelable(io);
    defer append_mu.unlock(io);
    const f = std.c.fopen(z.ptr, "a") orelse return;
    defer _ = std.c.fclose(f);
    _ = std.c.fwrite(line.ptr, 1, line.len, f);
}

/// Speculative-side state of a lookup: the base the tier offered and whether the request adopted it.
pub const Spec = struct { base: ?usize = null, adopted: bool = false };

pub const Lookup = struct {
    seq: u64,
    request: u64,
    prompt_len: usize,
    prompt_sha: Hex,
    matched: usize,
    /// "none", "ram", "ssd" or "live".
    tier: []const u8,
    source_id: u64 = 0,
    source_tokens: ?usize = null,
    dflash: Spec = .{},
    mtp: Spec = .{},
    kv: []const Hex = &.{},
    ssm: []const Hex = &.{},
    dflash_ctx: ?Hex = null,
    mtp_kv: ?Hex = null,
    mtp_head: ?Hex = null,
    arch: ?Hex = null,
};

fn optHex(w: *std.Io.Writer, key: []const u8, v: ?Hex) !void {
    if (v) |h| try w.print(",\"{s}\":\"{s}\"", .{ key, h }) else try w.print(",\"{s}\":null", .{key});
}

fn optNum(w: *std.Io.Writer, v: ?usize) !void {
    if (v) |n| try w.print("{d}", .{n}) else try w.writeAll("null");
}

fn nums(w: *std.Io.Writer, list: anytype) !void {
    try w.writeAll("[");
    for (list, 0..) |n, i| try w.print("{s}{d}", .{ if (i == 0) "" else ",", n });
    try w.writeAll("]");
}

fn hexList(w: *std.Io.Writer, key: []const u8, list: []const Hex) !void {
    try w.print(",\"{s}\":[", .{key});
    for (list, 0..) |h, i| try w.print("{s}\"{s}\"", .{ if (i == 0) "" else ",", h });
    try w.writeAll("]");
}

pub fn writeLookup(w: *std.Io.Writer, l: Lookup) !void {
    try w.print("{{\"row\":\"lookup\",\"seq\":{d},\"request\":{d},\"prompt_len\":{d},\"prompt_sha\":\"{s}\",\"matched\":{d},\"tier\":\"{s}\"", .{ l.seq, l.request, l.prompt_len, l.prompt_sha, l.matched, l.tier });
    try w.print(",\"source_id\":{d},\"source_tokens\":", .{l.source_id});
    try optNum(w, l.source_tokens);
    try w.print(",\"restored_pos\":{d}", .{l.matched});
    inline for (.{ .{ "dflash", l.dflash }, .{ "mtp", l.mtp } }) |p| {
        try w.print(",\"{s}\":{{\"base\":", .{p[0]});
        try optNum(w, p[1].base);
        try w.print(",\"adopted\":{}}}", .{p[1].adopted});
    }
    try hexList(w, "kv", l.kv);
    try hexList(w, "ssm", l.ssm);
    try optHex(w, "dflash_ctx", l.dflash_ctx);
    try optHex(w, "mtp_kv", l.mtp_kv);
    try optHex(w, "mtp_head", l.mtp_head);
    try optHex(w, "arch", l.arch);
    try w.writeAll("}\n");
}

pub fn writeGen(w: *std.Io.Writer, request: u64, ids: []const u32, finish: []const u8) !void {
    try w.print("{{\"row\":\"gen\",\"request\":{d},\"count\":{d},\"sha\":\"{s}\",\"finish\":\"{s}\"}}\n", .{ request, ids.len, idsHex(ids), finish });
}

fn writeSpecMeta(w: *std.Io.Writer, key: []const u8, m: ?disk_format.SpecMeta) !void {
    const s = m orelse return w.print(",\"{s}\":null", .{key});
    try w.print(",\"{s}\":{{\"base\":{d},\"step\":{d},\"layers\":{d},\"head\":", .{ key, s.base, s.step, s.layers });
    if (s.head) |h| {
        try w.print("{{\"pos_base\":{d},\"ratio\":{d},\"pooled\":{},\"rows\":{d},\"marks\":", .{ h.pos_base, h.ratio, h.pooled, h.rows });
        try nums(w, h.marks[0..h.mark_count]);
        try w.writeAll("}");
    } else try w.writeAll("null");
    try w.writeAll("}");
}

fn lessLastUsed(entries: []const disk_format.IndexEntry, a: usize, b: usize) bool {
    return entries[a].last_used < entries[b].last_used;
}

/// The tier after a flush: every record, then the ids oldest-used first (the eviction order).
pub fn writeTier(w: *std.Io.Writer, allocator: std.mem.Allocator, request: u64, entries: []const disk_format.IndexEntry, total_bytes: u64) !void {
    try w.print("{{\"row\":\"tier\",\"request\":{d},\"total_bytes\":{d},\"entries\":[", .{ request, total_bytes });
    for (entries, 0..) |e, i| {
        try w.print("{s}{{\"id\":{d},\"tokens\":{d},\"tokens_sha\":\"{s}\",\"kv_len\":{d},\"ssm_positions\":", .{ if (i == 0) "" else ",", e.id, e.tokens.len, idsHex(e.tokens), e.kv_len });
        try nums(w, e.ssm_positions);
        try w.writeAll(",\"ssm_bytes\":");
        try nums(w, e.ssm_bytes);
        try w.writeAll(",\"rings\":[");
        for (e.rings, 0..) |r, j| try w.print("{s}[{d},{d},{d}]", .{ if (j == 0) "" else ",", r.pos, r.rows, r.bytes });
        try w.print("],\"chunks\":{d},\"inherited_chunks\":{d},\"qsa_rows\":{d},\"qsa_bytes\":{d},\"qsa_inherited\":{},\"grown\":{d},\"inherited_grown\":{d},\"vision_key\":{d},\"media_start\":", .{ e.chunk_bytes.len, e.inherited_chunks, e.qsa_history_rows, e.qsa_history_bytes, e.inherited_qsa, e.grown.len, e.inherited_grown, e.vision_key });
        try optNum(w, if (e.media_start) |m| m else null);
        try w.print(",\"has_tools\":{},\"poisoned\":{},\"spec_bytes\":{d}", .{ e.has_tools, e.poisoned, e.spec_bytes });
        try writeSpecMeta(w, "spec_dflash", e.spec_dflash);
        try writeSpecMeta(w, "spec_mtp", e.spec_mtp);
        try w.print(",\"bytes\":{d}}}", .{e.bytes});
    }
    const order = try allocator.alloc(usize, entries.len);
    defer allocator.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, entries, lessLastUsed);
    try w.writeAll("],\"lru\":[");
    for (order, 0..) |o, i| try w.print("{s}{d}", .{ if (i == 0) "" else ",", entries[o].id });
    try w.writeAll("]}\n");
}

test "pathFromEnv: absent, empty or 0 is off" {
    try std.testing.expect(pathFromEnv(null) == null);
    try std.testing.expect(pathFromEnv("") == null);
    try std.testing.expect(pathFromEnv("0") == null);
    try std.testing.expectEqualStrings("/tmp/t.jsonl", pathFromEnv("/tmp/t.jsonl").?);
}

test "a lookup row carries its fields and every digest family" {
    const a = std.testing.allocator;
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    const kv = [_]Hex{ idsHex(&.{1}), idsHex(&.{2}) };
    try writeLookup(&out.writer, .{
        .seq = 3,
        .request = 7,
        .prompt_len = 5,
        .prompt_sha = idsHex(&.{ 1, 2, 3, 4, 5 }),
        .matched = 4,
        .tier = "ssd",
        .source_id = 12,
        .source_tokens = 40,
        .mtp = .{ .base = 0, .adopted = true },
        .kv = &kv,
        .arch = idsHex(&.{9}),
    });
    const parsed = try std.json.parseFromSlice(std.json.Value, a, out.written(), .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqualStrings("lookup", o.get("row").?.string);
    try std.testing.expectEqual(@as(i64, 4), o.get("matched").?.integer);
    try std.testing.expectEqual(@as(i64, 4), o.get("restored_pos").?.integer);
    try std.testing.expectEqualStrings("ssd", o.get("tier").?.string);
    try std.testing.expectEqual(@as(i64, 40), o.get("source_tokens").?.integer);
    try std.testing.expect(o.get("mtp").?.object.get("adopted").?.bool);
    try std.testing.expect(o.get("dflash").?.object.get("base").? == .null);
    try std.testing.expectEqual(@as(usize, 2), o.get("kv").?.array.items.len);
    try std.testing.expectEqualStrings(&idsHex(&.{9}), o.get("arch").?.string);
    try std.testing.expect(o.get("mtp_head").? == .null);
    try std.testing.expectEqual(@as(u8, '\n'), out.written()[out.written().len - 1]);
}

test "a gen row hashes the generated ids" {
    const a = std.testing.allocator;
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    try writeGen(&out.writer, 2, &.{ 5, 6, 7 }, "stop");
    const parsed = try std.json.parseFromSlice(std.json.Value, a, out.written(), .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqualStrings("gen", o.get("row").?.string);
    try std.testing.expectEqual(@as(i64, 3), o.get("count").?.integer);
    try std.testing.expectEqualStrings(&idsHex(&.{ 5, 6, 7 }), o.get("sha").?.string);
    try std.testing.expect(!std.mem.eql(u8, &idsHex(&.{ 5, 6, 7 }), &idsHex(&.{ 5, 7, 6 })));
}

test "a tier row lists each record and the eviction order" {
    const a = std.testing.allocator;
    var toks = [_]u32{ 1, 2, 3 };
    var cb = [_]u64{10};
    var pos = [_]u32{2};
    var sb = [_]u64{99};
    const base: disk_format.IndexEntry = .{
        .id = 0,
        .tokens = &toks,
        .kv_len = 3,
        .has_tools = false,
        .quant = .dense,
        .bytes = 123,
        .chunk_bytes = &cb,
        .ssm_positions = &pos,
        .ssm_bytes = &sb,
        .last_used = 0,
    };
    var entries = [_]disk_format.IndexEntry{ base, base };
    entries[0].id = 4;
    entries[0].last_used = 9;
    entries[1].id = 6;
    entries[1].last_used = 2;
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    try writeTier(&out.writer, a, 1, &entries, 246);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, out.written(), .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqualStrings("tier", o.get("row").?.string);
    try std.testing.expectEqual(@as(i64, 246), o.get("total_bytes").?.integer);
    const list = o.get("entries").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), list.len);
    try std.testing.expectEqual(@as(i64, 3), list[0].object.get("tokens").?.integer);
    try std.testing.expectEqual(@as(i64, 2), list[0].object.get("ssm_positions").?.array.items[0].integer);
    try std.testing.expect(list[0].object.get("spec_mtp").? == .null);
    const lru = o.get("lru").?.array.items;
    try std.testing.expectEqual(@as(i64, 6), lru[0].integer);
    try std.testing.expectEqual(@as(i64, 4), lru[1].integer);
}

test "append adds one line per call to the tape file and nothing when off" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    const file = try std.fmt.allocPrint(std.testing.allocator, "{s}/t.jsonl", .{dir});
    defer std.testing.allocator.free(file);
    defer path_override = null;
    path_override = "";
    append("ignored\n");
    path_override = file;
    append("one\n");
    append("two\n");
    const got = try tmp.dir.readFileAlloc(std.testing.io, "t.jsonl", std.testing.allocator, .limited(64));
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("one\ntwo\n", got);
}
