//! `sushi repack --pack-revision` against small generated packs.
const std = @import("std");
const testing = std.testing;
const stamp = @import("repack_stamp.zig");

extern "c" fn symlink(target: [*:0]const u8, linkpath: [*:0]const u8) c_int;

fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

const Tmp = struct {
    tmp: testing.TmpDir,
    path: []u8,

    fn make() !Tmp {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const n = try tmp.dir.realPath(io(), &buf);
        return .{ .tmp = tmp, .path = try testing.allocator.dupe(u8, buf[0..n]) };
    }

    fn deinit(self: *Tmp) void {
        testing.allocator.free(self.path);
        self.tmp.cleanup();
    }
};

fn writeFile(dir: []const u8, name: []const u8, data: []const u8) !void {
    var d = try std.Io.Dir.openDirAbsolute(io(), dir, .{});
    defer d.close(io());
    try d.writeFile(io(), .{ .sub_path = name, .data = data });
}

fn readFile(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    var d = try std.Io.Dir.openDirAbsolute(io(), dir, .{});
    defer d.close(io());
    return d.readFileAlloc(io(), name, a, .limited(64 << 20));
}

/// A shard of two tensors whose header is padded so the data starts on a multiple of `align_to`.
fn shardBytes(a: std.mem.Allocator, meta_json: ?[]const u8, seed: u8, align_to: usize) ![]u8 {
    var header: std.ArrayList(u8) = .empty;
    try header.append(a, '{');
    if (meta_json) |m| try header.print(a, "\"__metadata__\":{s},", .{m});
    try header.appendSlice(a, "\"a.weight\":{\"dtype\":\"U8\",\"shape\":[3000],\"data_offsets\":[0,3000]},\"b.weight\":{\"dtype\":\"F16\",\"shape\":[5],\"data_offsets\":[3000,3010]}}");
    var head_len = header.items.len;
    while ((8 + head_len) % align_to != 0) head_len += 1;
    try header.appendNTimes(a, ' ', head_len - header.items.len);
    var file: std.ArrayList(u8) = .empty;
    try file.appendNTimes(a, 0, 8);
    std.mem.writeInt(u64, file.items[0..8], header.items.len, .little);
    try file.appendSlice(a, header.items);
    for (0..3010) |i| try file.append(a, @truncate(i *% 31 +% seed));
    return file.items;
}

const config_text =
    \\{
    \\  "model_type": "tiny",
    \\  "rms_norm_eps": 1e-20,
    \\  "nested": {"pack_revision_not_me": 1}
    \\}
    \\
;

const s1 = "model-00001-of-00003.safetensors";
const s2 = "model-00002-of-00003.safetensors";
const s3 = "model-00003-of-00003.safetensors";
const long_id: []const u8 = &@as([65]u8, @splat('x'));
const linked_name = "model-00004-of-00003.safetensors";

fn makePack(a: std.mem.Allocator, dir: []const u8) !void {
    try writeFile(dir, "config.json", config_text);
    try writeFile(dir, s1, try shardBytes(a, null, 1, 8));
    try writeFile(dir, s2, try shardBytes(a, "{\"tile_order\":\"pair_major\",\"k\":\"4\"}", 2, 16384));
    try writeFile(dir, s3, try shardBytes(a, "{\"k\":\"4\"}", 3, 8));
}

fn dataOf(file: []const u8) []const u8 {
    const len = std.mem.readInt(u64, file[0..8], .little);
    return file[8 + len ..];
}

fn stampOf(a: std.mem.Allocator, file: []const u8) !?[]const u8 {
    const len = std.mem.readInt(u64, file[0..8], .little);
    const h = try std.json.parseFromSliceLeaky(std.json.Value, a, file[8..][0..len], .{});
    const meta = h.object.get("__metadata__") orelse return null;
    const v = meta.object.get("pack_revision") orelse return null;
    return v.string;
}

fn run(dir: []const u8, o: stamp.Options) !stamp.Report {
    var report: stamp.Report = .{};
    try stamp.stampPack(testing.allocator, io(), dir, o, &report);
    return report;
}

test "repack stamp: a header gains the key with or without metadata, keeps every other field, and pads to the alignment" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]?[]const u8{ null, "{\"k\":\"4\",\"tile_order\":\"pair_major\"}" }) |meta| {
        for ([_]usize{ 8, 16384 }) |al| {
            const file = try shardBytes(a, meta, 7, al);
            const old_len = std.mem.readInt(u64, file[0..8], .little);
            const head = try stamp.stampedHead(a, file[8..][0..old_len], "2.0", 8 + old_len);
            const new_len = std.mem.readInt(u64, head[0..8], .little);
            try testing.expectEqual(@as(u64, 0), (8 + new_len) % al);
            try testing.expectEqual(head.len, 8 + new_len);
            const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, head[8..], .{});
            const m = parsed.object.get("__metadata__").?.object;
            try testing.expectEqualStrings("2.0", m.get("pack_revision").?.string);
            try testing.expectEqual(@as(usize, if (meta == null) 1 else 3), m.count());
            try testing.expectEqualStrings("__metadata__", parsed.object.keys()[0]);
            const old = try std.json.parseFromSliceLeaky(std.json.Value, a, file[8..][0..old_len], .{});
            for (old.object.keys(), old.object.values()) |k, v| {
                if (std.mem.eql(u8, k, "__metadata__")) continue;
                const got = parsed.object.get(k).?;
                try testing.expectEqual(v.object.get("data_offsets").?.array.items[1].integer, got.object.get("data_offsets").?.array.items[1].integer);
                try testing.expectEqualStrings(v.object.get("dtype").?.string, got.object.get("dtype").?.string);
            }
        }
    }
}

test "repack stamp: the key lands after the existing metadata and a stamp already there is replaced" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const file = try shardBytes(a, "{\"k\":\"4\",\"pack_revision\":\"1.0\"}", 7, 8);
    const old_len = std.mem.readInt(u64, file[0..8], .little);
    const head = try stamp.stampedHead(a, file[8..][0..old_len], "2.0", 8 + old_len);
    const m = (try std.json.parseFromSliceLeaky(std.json.Value, a, head[8..], .{})).object.get("__metadata__").?.object;
    try testing.expectEqual(@as(usize, 2), m.count());
    try testing.expectEqualStrings("k", m.keys()[0]);
    try testing.expectEqualStrings("2.0", m.get("pack_revision").?.string);
}

test "repack stamp: the shard selector matches names, globs with * and ?, and nothing else" {
    try testing.expect(stamp.globMatch(s1, s1));
    try testing.expect(stamp.globMatch("model-*-of-00003.safetensors", s2));
    try testing.expect(stamp.globMatch("model-0000?-of-*", s3));
    try testing.expect(stamp.globMatch("*", s1));
    try testing.expect(stamp.globMatch("*00002*", s2));
    try testing.expect(!stamp.globMatch("*00002*", s3));
    try testing.expect(!stamp.globMatch("model-00001", s1));
    try testing.expect(!stamp.globMatch("model-0000?-of-00002.safetensors", s1));
    try testing.expect(!stamp.globMatch("", s1));
}

test "repack stamp: the whole pack is stamped, config gains the member last, and nothing else moves" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try Tmp.make();
    defer tmp.deinit();
    try makePack(a, tmp.path);
    const before = [_][]u8{ try readFile(a, tmp.path, s1), try readFile(a, tmp.path, s2), try readFile(a, tmp.path, s3) };
    const report = try run(tmp.path, .{ .id = "2.0" });
    try testing.expectEqual(@as(usize, 3), report.stamped);
    for ([_][]const u8{ s1, s2, s3 }, before) |name, old| {
        const now = try readFile(a, tmp.path, name);
        try testing.expectEqualStrings("2.0", (try stampOf(a, now)).?);
        try testing.expect(std.mem.eql(u8, dataOf(old), dataOf(now)));
        const al: usize = if (std.mem.eql(u8, name, s2)) 16384 else 8;
        try testing.expectEqual(@as(usize, 0), (now.len - dataOf(now).len) % al);
    }
    const cfg = try readFile(a, tmp.path, "config.json");
    try testing.expect(std.mem.startsWith(u8, cfg, "{\n  \"model_type\": \"tiny\",\n  \"rms_norm_eps\": 1e-20,\n  \"nested\": {\"pack_revision_not_me\": 1}"));
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, cfg, .{});
    const pr = parsed.object.get("pack_revision").?.object;
    try testing.expectEqualStrings("2.0", pr.get("id").?.string);
    try testing.expectEqual(@as(usize, 0), pr.get("unstamped").?.array.items.len);
    try testing.expectEqualStrings("pack_revision", parsed.object.keys()[parsed.object.count() - 1]);
    var d = try std.Io.Dir.openDirAbsolute(io(), tmp.path, .{ .iterate = true });
    defer d.close(io());
    var it = d.iterate();
    while (try it.next(io())) |e| {
        try testing.expect(!std.mem.endsWith(u8, e.name, ".stamp-tmp"));
        try testing.expect(!std.mem.endsWith(u8, e.name, ".lock"));
    }
}

test "repack stamp: a selection stamps only its shards, the rest follows, and a rerun changes nothing" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try Tmp.make();
    defer tmp.deinit();
    try makePack(a, tmp.path);
    const third = try readFile(a, tmp.path, s3);
    var r = try run(tmp.path, .{ .id = "2.0", .shards = &.{s1} });
    try testing.expectEqual(@as(usize, 1), r.stamped);
    try testing.expect(std.mem.eql(u8, third, try readFile(a, tmp.path, s3)));
    try testing.expectEqualStrings("2.0", (try stampOf(a, try readFile(a, tmp.path, s1))).?);
    try testing.expect(null == try stampOf(a, try readFile(a, tmp.path, s2)));
    try testing.expectError(error.RepackStampMissing, stamp.checkPack(testing.allocator, io(), tmp.path, "2.0", &.{}, &r));

    r = try run(tmp.path, .{ .id = "2.0", .shards = &.{ "*00002*", s3 } });
    try testing.expectEqual(@as(usize, 2), r.stamped);
    try stamp.checkPack(testing.allocator, io(), tmp.path, "2.0", &.{}, &r);

    const snap = [_][]u8{ try readFile(a, tmp.path, s1), try readFile(a, tmp.path, s2), try readFile(a, tmp.path, s3), try readFile(a, tmp.path, "config.json") };
    r = try run(tmp.path, .{ .id = "2.0" });
    try testing.expectEqual(@as(usize, 0), r.stamped);
    try testing.expectEqual(@as(usize, 3), r.skipped);
    for ([_][]const u8{ s1, s2, s3, "config.json" }, snap) |name, was| try testing.expect(std.mem.eql(u8, was, try readFile(a, tmp.path, name)));
}

test "repack stamp: a pattern that matches nothing is refused by name before a byte is written" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try Tmp.make();
    defer tmp.deinit();
    try makePack(a, tmp.path);
    const before = try readFile(a, tmp.path, s1);
    var report: stamp.Report = .{};
    try testing.expectError(error.RepackNoShardMatch, stamp.stampPack(testing.allocator, io(), tmp.path, .{ .id = "2.0", .shards = &.{ s1, "nope-*" } }, &report));
    try testing.expect(std.mem.indexOf(u8, report.text(), "nope-*") != null);
    try testing.expect(std.mem.eql(u8, before, try readFile(a, tmp.path, s1)));
    try testing.expectEqualStrings(config_text, try readFile(a, tmp.path, "config.json"));
}

test "repack stamp: a symlinked shard is refused by name unless listed, and listed it is neither read through nor written" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try Tmp.make();
    defer tmp.deinit();
    try makePack(a, tmp.path);
    try writeFile(tmp.path, "shared.bin", try shardBytes(a, null, 9, 8));
    const target = try std.fmt.allocPrintSentinel(a, "{s}/shared.bin", .{tmp.path}, 0);
    const link = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ tmp.path, linked_name }, 0);
    try testing.expectEqual(@as(c_int, 0), symlink(target.ptr, link.ptr));
    const linked = try readFile(a, tmp.path, "shared.bin");

    var report: stamp.Report = .{};
    try testing.expectError(error.RepackShardIsLink, stamp.stampPack(testing.allocator, io(), tmp.path, .{ .id = "2.0" }, &report));
    try testing.expect(std.mem.indexOf(u8, report.text(), linked_name) != null);
    try testing.expect(null == try stampOf(a, try readFile(a, tmp.path, s1)));

    try testing.expectError(error.RepackUnknownShard, stamp.stampPack(testing.allocator, io(), tmp.path, .{ .id = "2.0", .unstamped = &.{"model-00009.safetensors"} }, &report));
    try testing.expect(std.mem.indexOf(u8, report.text(), "model-00009.safetensors") != null);

    report = try run(tmp.path, .{ .id = "2.0", .unstamped = &.{linked_name}, .shards = &.{"*"} });
    try testing.expectEqual(@as(usize, 3), report.stamped);
    try testing.expect(std.mem.eql(u8, linked, try readFile(a, tmp.path, "shared.bin")));
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, try readFile(a, tmp.path, "config.json"), .{});
    const un = parsed.object.get("pack_revision").?.object.get("unstamped").?.array.items;
    try testing.expectEqual(@as(usize, 1), un.len);
    try testing.expectEqualStrings(linked_name, un[0].string);
    try stamp.checkPack(testing.allocator, io(), tmp.path, "2.0", &.{}, &report);
}

test "repack stamp: a different existing id is refused, in a shard or in the config, and a bad id is refused" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try Tmp.make();
    defer tmp.deinit();
    try makePack(a, tmp.path);
    _ = try run(tmp.path, .{ .id = "2.0", .shards = &.{s1} });
    const snap = [_][]u8{ try readFile(a, tmp.path, s1), try readFile(a, tmp.path, s2), try readFile(a, tmp.path, "config.json") };
    var report: stamp.Report = .{};
    try testing.expectError(error.RepackStampConflict, stamp.stampPack(testing.allocator, io(), tmp.path, .{ .id = "3.0" }, &report));
    try testing.expect(std.mem.indexOf(u8, report.text(), "2.0") != null);
    for ([_][]const u8{ s1, s2, "config.json" }, snap) |name, was| try testing.expect(std.mem.eql(u8, was, try readFile(a, tmp.path, name)));
    for ([_][]const u8{ "", "a b", "1\"2", long_id }) |bad| {
        try testing.expectError(error.RepackStampBadId, stamp.stampPack(testing.allocator, io(), tmp.path, .{ .id = bad }, &report));
    }
}

test "repack stamp: the verify step catches a corrupted rewrite and the original stays untouched" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try Tmp.make();
    defer tmp.deinit();
    try makePack(a, tmp.path);
    const before = [_][]u8{ try readFile(a, tmp.path, s1), try readFile(a, tmp.path, "config.json") };
    var report: stamp.Report = .{};
    try testing.expectError(error.RepackVerifyFailed, stamp.stampPack(testing.allocator, io(), tmp.path, .{ .id = "2.0", .corrupt_tmp = true }, &report));
    for ([_][]const u8{ s1, "config.json" }, before) |name, was| try testing.expect(std.mem.eql(u8, was, try readFile(a, tmp.path, name)));
    var d = try std.Io.Dir.openDirAbsolute(io(), tmp.path, .{ .iterate = true });
    defer d.close(io());
    var it = d.iterate();
    while (try it.next(io())) |e| try testing.expect(!std.mem.endsWith(u8, e.name, ".stamp-tmp"));
}

test "repack stamp: an interrupted run resumes: stamped shards are skipped and the config lands last" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try Tmp.make();
    defer tmp.deinit();
    try makePack(a, tmp.path);
    var report: stamp.Report = .{};
    try testing.expectError(error.RepackInterrupted, stamp.stampPack(testing.allocator, io(), tmp.path, .{ .id = "2.0", .fail_after_shards = 1 }, &report));
    try testing.expectEqualStrings(config_text, try readFile(a, tmp.path, "config.json"));
    report = try run(tmp.path, .{ .id = "2.0" });
    try testing.expectEqual(@as(usize, 2), report.stamped);
    try testing.expectEqual(@as(usize, 1), report.skipped);
    try stamp.checkPack(testing.allocator, io(), tmp.path, null, &.{}, &report);
}

test "repack stamp: a full disk is refused before a write, and check names the shard that lacks the id" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try Tmp.make();
    defer tmp.deinit();
    try makePack(a, tmp.path);
    var report: stamp.Report = .{};
    try testing.expectError(error.RepackNoSpace, stamp.stampPack(testing.allocator, io(), tmp.path, .{ .id = "2.0", .free_bytes = 1000 }, &report));
    try testing.expectEqualStrings(config_text, try readFile(a, tmp.path, "config.json"));
    try testing.expectError(error.RepackStampMissing, stamp.checkPack(testing.allocator, io(), tmp.path, null, &.{}, &report));
    _ = try run(tmp.path, .{ .id = "2.0" });
    try testing.expectError(error.RepackStampConflict, stamp.checkPack(testing.allocator, io(), tmp.path, "9.9", &.{}, &report));
    try writeFile(tmp.path, s2, try shardBytes(a, "{\"k\":\"4\"}", 2, 8));
    try testing.expectError(error.RepackStampMissing, stamp.checkPack(testing.allocator, io(), tmp.path, null, &.{}, &report));
    try testing.expect(std.mem.indexOf(u8, report.text(), s2) != null);
}

test "repack stamp: the config editor replaces an existing member in place and keeps every other byte" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const raw = "{\"a\": 1, \"pack_revision\": {\"id\": \"1.0\", \"unstamped\": []}, \"z\": {\"pack_revision\": 5}}";
    try testing.expectEqualStrings(
        "{\"a\": 1, \"pack_revision\": {\"id\": \"2.0\", \"unstamped\": [\"x.safetensors\"]}, \"z\": {\"pack_revision\": 5}}",
        try stamp.stampedConfig(a, raw, "2.0", &.{"x.safetensors"}),
    );
    try testing.expectError(error.InvalidConfig, stamp.stampedConfig(a, "[1]", "2.0", &.{}));
    try testing.expectEqualStrings("{\"a\":1,\"pack_revision\": {\"id\": \"2.0\", \"unstamped\": []}}", try stamp.stampedConfig(a, "{\"a\":1}", "2.0", &.{}));
}

test "repack stamp: the command line takes the documented options and refuses the rest" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const plain = try stamp.parseArgs(a, &.{"pack"});
    try testing.expect(!plain.stamping());
    const full = try stamp.parseArgs(a, &.{ "pack", "--pack-revision", "2.0", "--shards", "a*", "--shards", "b", "--unstamped", "c" });
    try testing.expectEqualStrings("2.0", full.id.?);
    try testing.expectEqual(@as(usize, 2), full.shards.len);
    try testing.expectEqual(@as(usize, 1), full.unstamped.len);
    const chk = try stamp.parseArgs(a, &.{ "--check", "pack" });
    try testing.expect(chk.check and chk.id == null);
    for ([_][]const []const u8{
        &.{},
        &.{ "pack", "--bogus" },
        &.{ "pack", "--pack-revision" },
        &.{ "pack", "--shards", "a" },
        &.{ "pack", "--unstamped", "a" },
        &.{ "pack", "--pack-revision", "2.0", "--pack-revision", "3.0" },
        &.{ "pack", "other" },
        &.{ "pack", "--check", "--unstamped", "a" },
        &.{ "pack", "--pack-revision", "bad id" },
    }) |args| try testing.expectError(error.RepackBadArgs, stamp.parseArgs(a, args));
}
