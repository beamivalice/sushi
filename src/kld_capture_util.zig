//! Helpers the native streamed capture paths share: atomic JSON files, content hashes, the layer-major pause file and
//! the baseline completion record.
const std = @import("std");
const kld = @import("kld.zig");
const fixture = @import("kld_fixture.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub fn json(a: std.mem.Allocator, io: std.Io, directory: []const u8, name: []const u8, value: anytype) !void {
    const raw = try std.json.Stringify.valueAlloc(a, value, .{ .whitespace = .indent_2 });
    defer a.free(raw);
    var dir = try std.Io.Dir.cwd().openDir(io, directory, .{});
    defer dir.close(io);
    const tmp = try std.fmt.allocPrint(a, "{s}.tmp", .{name});
    defer a.free(tmp);
    try dir.writeFile(io, .{ .sub_path = tmp, .data = raw });
    try dir.rename(tmp, dir, name, io);
}

pub fn greedy(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = i;
    };
    return @intCast(best);
}

pub fn shaHex(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

pub fn inputsSha(prompts: []const fixture.Prompt, inputs: []const []u32) [64]u8 {
    var h = Sha256.init(.{});
    for (prompts, inputs) |p, ids| {
        h.update(std.mem.asBytes(&@as(u64, p.id.len)));
        h.update(p.id);
        h.update(std.mem.asBytes(&@as(u64, ids.len)));
        h.update(std.mem.sliceAsBytes(ids));
    }
    return std.fmt.bytesToHex(h.finalResult(), .lower);
}

pub fn engineSha(a: std.mem.Allocator, io: std.Io) ![64]u8 {
    const path = try std.process.executablePathAlloc(io, a);
    defer a.free(path);
    return @import("update.zig").fileSha256Hex(io, path);
}

pub fn fsyncPath(path: []const u8) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const fd = std.c.open(try std.fmt.bufPrintSentinel(&buf, "{s}", .{path}, 0), .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.LayerMajorSyncFailed;
    defer _ = std.c.close(fd);
    if (std.c.fsync(fd) != 0) return error.LayerMajorSyncFailed;
}

/// While `pause` exists the capture idles between batches, `<pause>.ack` saying so.
pub fn waitWhilePaused(a: std.mem.Allocator, io: std.Io, pause: []const u8, committed: usize, out: *kld.Out) !void {
    if (pause.len == 0) return;
    const cwd = std.Io.Dir.cwd();
    cwd.access(io, pause, .{}) catch return;
    const ack = try std.fmt.allocPrint(a, "{s}.ack", .{pause});
    defer a.free(ack);
    const body = try std.fmt.allocPrint(a, "{{\"committed_windows\":{d}}}\n", .{committed});
    defer a.free(body);
    try cwd.writeFile(io, .{ .sub_path = ack, .data = body });
    out.print("[kld] layer-major paused with {d} windows committed; remove {s} to continue\n", .{ committed, pause });
    while (true) {
        cwd.access(io, pause, .{}) catch break;
        var tick = std.c.timespec{ .sec = 0, .nsec = 250_000_000 };
        _ = std.c.nanosleep(&tick, null);
    }
    cwd.deleteFile(io, ack) catch {};
}

pub fn appendLine(fd: std.c.fd_t, line: []const u8) !void {
    for ([_][]const u8{ line, "\n" }) |part| {
        var done: usize = 0;
        while (done < part.len) {
            const got = std.c.write(fd, part[done..].ptr, part.len - done);
            if (got <= 0) return error.LayerMajorLogWriteFailed;
            done += @intCast(got);
        }
    }
}

pub fn metadataHash(a: std.mem.Allocator, io: std.Io, directory: []const u8, name: []const u8) ![64]u8 {
    var dir = try std.Io.Dir.openDirAbsolute(io, directory, .{});
    defer dir.close(io);
    const bytes = try dir.readFileAlloc(io, name, a, .limited(128 * 1024 * 1024));
    defer a.free(bytes);
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    return std.fmt.bytesToHex(hash, .lower);
}

pub fn completeBaseline(a: std.mem.Allocator, io: std.Io, directory: []const u8, count: usize, tokens: u32) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, directory, .{});
    defer dir.close(io);
    const raw = try dir.readFileAlloc(io, "baseline.json", a, .limited(16 * 1024 * 1024));
    defer a.free(raw);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadBaselineJson;
    const arena = parsed.arena.allocator();
    try parsed.value.object.put(arena, "complete", .{ .bool = true });
    try parsed.value.object.put(arena, "requested_prompt_count", .{ .integer = @intCast(count) });
    try parsed.value.object.put(arena, "completed_prompt_count", .{ .integer = @intCast(count) });
    try parsed.value.object.put(arena, "requested_tokens_per_prompt", .{ .integer = tokens });
    try parsed.value.object.put(arena, "actual_positions", .{ .integer = @intCast(count * tokens) });
    try parsed.value.object.put(arena, "human_truncated", .{ .bool = false });
    try parsed.value.object.put(arena, "quality_scope", .{ .string = "Requested prompt study; full fixed-length capture. The standard release verdict requires sixteen prompts." });
    try json(a, io, directory, "baseline.json", parsed.value);
}

pub fn jsonNumber(value: std.json.Value) !f64 {
    return switch (value) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.NotANumber,
    };
}
