//! A pack's revision stamp (docs/pack-format.md#pack-revision): config.json names the revision and every shard
//! carries it in its safetensors `__metadata__`, so a folder that mixes two revisions is refused before any
//! weight loads. Reads headers only; links no MLX.
const std = @import("std");
const log = @import("log.zig");

pub const Revision = struct {
    id: []u8,
    /// Shard file names or pack-relative paths admitted whatever their header says (byte-identical across revisions).
    unstamped: [][]u8,

    pub fn deinit(self: *Revision, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        for (self.unstamped) |n| allocator.free(n);
        allocator.free(self.unstamped);
        self.* = .{ .id = &.{}, .unstamped = &.{} };
    }

    fn isUnstamped(self: Revision, name: []const u8) bool {
        const base = std.fs.path.basename(name);
        for (self.unstamped) |n| if (std.mem.eql(u8, n, name) or std.mem.eql(u8, n, base)) return true;
        return false;
    }
};

/// The `pack_revision` object of a parsed config.json root; null for a legacy pack.
pub fn fromConfig(allocator: std.mem.Allocator, root: std.json.ObjectMap) !?Revision {
    const v = root.get("pack_revision") orelse return null;
    if (v != .object) return error.InvalidPackRevision;
    const id = v.object.get("id") orelse return error.InvalidPackRevision;
    if (id != .string or id.string.len == 0) return error.InvalidPackRevision;
    var names: std.ArrayList([]u8) = .empty;
    errdefer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }
    if (v.object.get("unstamped")) |list| {
        if (list != .array) return error.InvalidPackRevision;
        for (list.array.items) |item| {
            if (item != .string) return error.InvalidPackRevision;
            const copy = try allocator.dupe(u8, item.string);
            errdefer allocator.free(copy);
            try names.append(allocator, copy);
        }
    }
    const id_copy = try allocator.dupe(u8, id.string);
    errdefer allocator.free(id_copy);
    return .{ .id = id_copy, .unstamped = try names.toOwnedSlice(allocator) };
}

/// The text a refusal logs for one shard.
pub fn describe(buf: []u8, name: []const u8, have: ?[]const u8, pack_id: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s} is pack revision {s}{s}, the pack is {s}: the folder mixes two revisions; run `sushi pull <name>` again or re-download it", .{
        name,
        have orelse "1.0",
        if (have == null) " (unstamped)" else "",
        pack_id,
    }) catch name;
}

/// Refuses with `PackRevisionMixed` when a shard the loader will read is not stamped with `rev.id`.
pub fn verify(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, rev: Revision) !void {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }
    try shardNames(io, allocator, dir, &names);
    var mixed = false;
    for (names.items) |name| {
        if (rev.isUnstamped(name)) continue;
        const stamp = try readStamp(io, allocator, dir, name) orelse continue;
        defer if (stamp.value) |s| allocator.free(s);
        if (stamp.value) |s| if (std.mem.eql(u8, s, rev.id)) continue;
        var buf: [768]u8 = undefined;
        log.err("PackRevisionMixed: {s}\n", .{describe(&buf, name, stamp.value, rev.id)});
        mixed = true;
    }
    if (mixed) return error.PackRevisionMixed;
}

/// The index's shards (MTP included) when it names any on disk, else every `.safetensors` one folder deep.
fn shardNames(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, out: *std.ArrayList([]u8)) !void {
    if (dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(16 * 1024 * 1024))) |raw| {
        defer allocator.free(raw);
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch null;
        if (parsed) |p| {
            defer p.deinit();
            if (p.value == .object) if (p.value.object.get("weight_map")) |wm| if (wm == .object) {
                for (wm.object.values()) |v| {
                    if (v != .string or hasName(out.items, v.string)) continue;
                    _ = dir.statFile(io, v.string, .{}) catch continue;
                    try out.append(allocator, try allocator.dupe(u8, v.string));
                }
            };
        }
    } else |_| {}
    if (out.items.len > 0) return;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        try out.append(allocator, try allocator.dupe(u8, entry.name));
    }
    it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .directory) continue;
        var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
        defer sub.close(io);
        var sit = sub.iterate();
        while (try sit.next(io)) |se| {
            if (se.kind != .file and se.kind != .sym_link) continue;
            if (!std.mem.endsWith(u8, se.name, ".safetensors")) continue;
            try out.append(allocator, try std.fmt.allocPrint(allocator, "{s}/{s}", .{ entry.name, se.name }));
        }
    }
}

fn hasName(names: []const []u8, name: []const u8) bool {
    for (names) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

const Stamp = struct { value: ?[]u8 };

/// The shard's `__metadata__.pack_revision` (null value: unstamped); null when the shard cannot be opened,
/// which the loader reports under its own name.
fn readStamp(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, name: []const u8) !?Stamp {
    const file = dir.openFile(io, name, .{}) catch return null;
    defer file.close(io);
    const size = (try file.stat(io)).size;
    var buf: [8192]u8 = undefined;
    var reader = file.reader(io, &buf);
    const header_len = try reader.interface.takeInt(u64, .little);
    if (header_len == 0 or header_len > 128 * 1024 * 1024 or header_len > size -| 8) return error.InvalidSafetensorsHeader;
    const header = try allocator.alloc(u8, @intCast(header_len));
    defer allocator.free(header);
    try reader.interface.readSliceAll(header);
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, header, .{}) catch return error.InvalidSafetensorsHeader;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSafetensorsHeader;
    const meta = parsed.value.object.get("__metadata__") orelse return .{ .value = null };
    if (meta != .object) return .{ .value = null };
    const v = meta.object.get("pack_revision") orelse return .{ .value = null };
    if (v != .string) return .{ .value = null };
    return .{ .value = try allocator.dupe(u8, v.string) };
}

const testing = std.testing;

fn writeShard(dir: std.Io.Dir, name: []const u8, stamp: ?[]const u8) !void {
    var hdr: [256]u8 = undefined;
    const tensor = "\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}";
    const json = if (stamp) |s|
        try std.fmt.bufPrint(&hdr, "{{\"__metadata__\":{{\"pack_revision\":\"{s}\"}},{s}}}", .{ s, tensor })
    else
        try std.fmt.bufPrint(&hdr, "{{{s}}}", .{tensor});
    var bytes: [300]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], json.len, .little);
    @memcpy(bytes[8..][0..json.len], json);
    @memset(bytes[8 + json.len ..][0..4], 0);
    try dir.writeFile(testing.io, .{ .sub_path = name, .data = bytes[0 .. 8 + json.len + 4] });
}

fn tmpPath(tmp: *testing.TmpDir, buf: []u8) ![]const u8 {
    const n = try tmp.dir.realPath(testing.io, buf);
    return buf[0..n];
}

const rev_2 = "{\"pack_revision\":{\"id\":\"2.0\",\"unstamped\":[\"engram.safetensors\"]}}";

fn revisionOf(json: []const u8) !Revision {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    return (try fromConfig(testing.allocator, parsed.value.object)).?;
}

test "pack revision: a config without the field is a legacy pack" {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"model_type\":\"x\"}", .{});
    defer parsed.deinit();
    try testing.expect((try fromConfig(testing.allocator, parsed.value.object)) == null);
}

test "pack revision: a malformed field is refused" {
    for ([_][]const u8{ "{\"pack_revision\":\"2.0\"}", "{\"pack_revision\":{}}", "{\"pack_revision\":{\"id\":2}}", "{\"pack_revision\":{\"id\":\"2\",\"unstamped\":[1]}}" }) |j| {
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, j, .{});
        defer parsed.deinit();
        try testing.expectError(error.InvalidPackRevision, fromConfig(testing.allocator, parsed.value.object));
    }
}

test "pack revision: a consistent pack passes, with the unstamped name admitted" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try writeShard(tmp.dir, "model-1.safetensors", "2.0");
    try writeShard(tmp.dir, "engram.safetensors", null);
    var path: [std.fs.max_path_bytes]u8 = undefined;
    var rev = try revisionOf(rev_2);
    defer rev.deinit(testing.allocator);
    try verify(testing.io, testing.allocator, try tmpPath(&tmp, &path), rev);
}

test "pack revision: an unstamped entry matches a pack-relative path as well as a name, whatever the header says" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "mtp");
    try writeShard(tmp.dir, "model-1.safetensors", "2.0");
    var mtp = try tmp.dir.openDir(testing.io, "mtp", .{});
    defer mtp.close(testing.io);
    try writeShard(mtp, "x.safetensors", "1.0");
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const dir_path = try tmpPath(&tmp, &path);
    var by_path = try revisionOf("{\"pack_revision\":{\"id\":\"2.0\",\"unstamped\":[\"mtp/x.safetensors\"]}}");
    defer by_path.deinit(testing.allocator);
    try verify(testing.io, testing.allocator, dir_path, by_path);
    var by_name = try revisionOf("{\"pack_revision\":{\"id\":\"2.0\",\"unstamped\":[\"x.safetensors\"]}}");
    defer by_name.deinit(testing.allocator);
    try verify(testing.io, testing.allocator, dir_path, by_name);
    var none = try revisionOf("{\"pack_revision\":{\"id\":\"2.0\"}}");
    defer none.deinit(testing.allocator);
    try testing.expectError(error.PackRevisionMixed, verify(testing.io, testing.allocator, dir_path, none));
}

test "pack revision: an unstamped shard is refused" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try writeShard(tmp.dir, "model-1.safetensors", "2.0");
    try writeShard(tmp.dir, "model-2.safetensors", null);
    var path: [std.fs.max_path_bytes]u8 = undefined;
    var rev = try revisionOf(rev_2);
    defer rev.deinit(testing.allocator);
    try testing.expectError(error.PackRevisionMixed, verify(testing.io, testing.allocator, try tmpPath(&tmp, &path), rev));
}

test "pack revision: a shard stamped with another revision is refused" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try writeShard(tmp.dir, "model-1.safetensors", "1.0");
    var path: [std.fs.max_path_bytes]u8 = undefined;
    var rev = try revisionOf(rev_2);
    defer rev.deinit(testing.allocator);
    try testing.expectError(error.PackRevisionMixed, verify(testing.io, testing.allocator, try tmpPath(&tmp, &path), rev));
}

test "pack revision: the index's MTP shard is checked, a shard it does not name is not" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "mtp");
    try writeShard(tmp.dir, "model-1.safetensors", "2.0");
    try writeShard(tmp.dir, "stray.safetensors", "1.0");
    var mtp = try tmp.dir.openDir(testing.io, "mtp", .{});
    defer mtp.close(testing.io);
    try writeShard(mtp, "weights.safetensors", "1.0");
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"a\":\"model-1.safetensors\",\"b\":\"mtp/weights.safetensors\"}}" });
    var path: [std.fs.max_path_bytes]u8 = undefined;
    var rev = try revisionOf(rev_2);
    defer rev.deinit(testing.allocator);
    const dir_path = try tmpPath(&tmp, &path);
    try testing.expectError(error.PackRevisionMixed, verify(testing.io, testing.allocator, dir_path, rev));
    try writeShard(mtp, "weights.safetensors", "2.0");
    try verify(testing.io, testing.allocator, dir_path, rev);
}

test "pack revision: the refusal names the file, both revisions and the fix" {
    var buf: [768]u8 = undefined;
    try testing.expectEqualStrings(
        "model-trunk-03.safetensors is pack revision 1.0 (unstamped), the pack is 2.0: the folder mixes two revisions; run `sushi pull <name>` again or re-download it",
        describe(&buf, "model-trunk-03.safetensors", null, "2.0"),
    );
    try testing.expect(std.mem.indexOf(u8, describe(&buf, "a.safetensors", "1.5", "2.0"), "pack revision 1.5, the pack is 2.0") != null);
}
