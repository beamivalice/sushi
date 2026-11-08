//! A pair-major copy of a rate-grouped EXL3/STQ pack, built from the legacy one by permuting its routed trellis tensors
//! in memory, and the faults a bad copy can carry. The tiny pack tests start from it.
const std = @import("std");
const exl3 = @import("sushi_exl3");

const tile_layout = exl3.group_layout;

pub const Fault = enum {
    none,
    /// `expert_quant.tile_order` names an order this build does not know.
    bad_order_value,
    /// One routed projection keeps its legacy name and bytes beside the renamed rest.
    mixed_keys,
    /// One routed tensor declares its 5-D shape with the input and output grids swapped.
    shape_mismatch,
    /// The shard that holds the routed tensors carries no `tile_order` stamp.
    stamp_missing,
    /// A legacy pack whose routed shard stamps `tile_order: pair_major`.
    stamp_on_legacy,
    /// The config names `pair_major` over legacy tensor names.
    config_without_keys,
};

const exempt_key = "model.layers.0.mlp.switch_mlp.up_proj.g0.trellis";
const swapped_key = "model.layers.0.mlp.switch_mlp.gate_proj.g0.trellis";

fn isRoutedName(key: []const u8) bool {
    return std.mem.indexOf(u8, key, ".mlp.switch_mlp.") != null and std.mem.endsWith(u8, key, ".trellis");
}

fn renamed(key: []const u8, fault: Fault) bool {
    if (!isRoutedName(key)) return false;
    return switch (fault) {
        .stamp_on_legacy, .config_without_keys => false,
        .mixed_keys => !std.mem.eql(u8, key, exempt_key),
        else => true,
    };
}

fn readAll(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    return d.readFileAlloc(io, name, allocator, .limited(512 << 20));
}

fn writeAll(io: std.Io, dir: []const u8, name: []const u8, data: []const u8) !void {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = name, .data = data });
}

fn rewriteConfig(allocator: std.mem.Allocator, raw: []const u8, fault: Fault) ![]u8 {
    if (fault == .stamp_on_legacy) return allocator.dupe(u8, raw);
    const at = std.mem.indexOf(u8, raw, "\"expert_quant\"") orelse return error.FixtureConfig;
    const open = std.mem.indexOfScalarPos(u8, raw, at, '{') orelse return error.FixtureConfig;
    const value = if (fault == .bad_order_value) "rowmajor" else "pair_major";
    return std.fmt.allocPrint(allocator, "{s}\"tile_order\": \"{s}\",{s}", .{ raw[0 .. open + 1], value, raw[open + 1 ..] });
}

fn rewriteIndex(allocator: std.mem.Allocator, raw: []const u8, fault: Fault) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, raw, from, ".trellis\"")) |hit| {
        const key_start = std.mem.lastIndexOfScalar(u8, raw[0..hit], '"').? + 1;
        const end = hit + ".trellis".len;
        try out.appendSlice(allocator, raw[from..end]);
        if (renamed(raw[key_start..end], fault)) try out.appendSlice(allocator, "_pm");
        from = end;
    }
    try out.appendSlice(allocator, raw[from..]);
    return out.toOwnedSlice(allocator);
}

fn rewriteShard(allocator: std.mem.Allocator, raw: []const u8, fault: Fault, page: ?usize, off_page: []const u8) ![]u8 {
    const header_len: usize = @intCast(std.mem.readInt(u64, raw[0..8], .little));
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw[8..][0..header_len], .{});
    defer parsed.deinit();
    const data_in = raw[8 + header_len ..];

    // The new data section: a routed tensor starts on a `page` boundary when asked, the rest follow as they were.
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(allocator);
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    try header.append(allocator, '{');
    var first = true;
    var pads: usize = 0;
    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (!first) try header.append(allocator, ',');
        first = false;
        if (std.mem.eql(u8, key, "__metadata__")) {
            try header.appendSlice(allocator, "\"__metadata__\":{");
            var mit = entry.value_ptr.object.iterator();
            while (mit.next()) |m| try header.print(allocator, "\"{s}\":\"{s}\",", .{ m.key_ptr.*, m.value_ptr.string });
            if (fault != .stamp_missing and fault != .config_without_keys) {
                try header.appendSlice(allocator, "\"tile_order\":\"pair_major\"");
            } else if (header.items[header.items.len - 1] == ',') header.items.len -= 1;
            try header.append(allocator, '}');
            continue;
        }
        const obj = entry.value_ptr.object;
        const dtype = obj.get("dtype").?.string;
        const shape_in = obj.get("shape").?.array.items;
        const offs = obj.get("data_offsets").?.array.items;
        const start: usize = @intCast(offs[0].integer);
        const end: usize = @intCast(offs[1].integer);
        var shape: [5]u64 = undefined;
        for (shape_in, 0..) |d, i| shape[i] = @intCast(d.integer);
        const major = renamed(key, fault);
        const aligned = major and page != null and (off_page.len == 0 or std.mem.indexOf(u8, key, off_page) == null);
        if (aligned) {
            // A real zero tensor fills the gap, as the repacker writes it: safetensors readers reject gaps.
            const gap = std.mem.alignForward(usize, data.items.len, page.?) - data.items.len;
            if (gap > 0) {
                try header.print(allocator, "\"__pad__.test.{d}\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}},", .{ pads, gap, data.items.len, data.items.len + gap });
                try data.appendNTimes(allocator, 0, gap);
                pads += 1;
            }
        }
        const at = data.items.len;
        try data.appendSlice(allocator, data_in[start..end]);
        if (!major) {
            try header.print(allocator, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ key, dtype });
            for (shape_in, 0..) |d, i| try header.print(allocator, "{s}{d}", .{ if (i == 0) "" else ",", d.integer });
            try header.print(allocator, "],\"data_offsets\":[{d},{d}]}}", .{ at, at + end - start });
            continue;
        }
        if (shape_in.len != 4 or !std.mem.eql(u8, dtype, "U16")) return error.FixtureShape;
        const experts = shape[0];
        const it_tiles: usize = @intCast(shape[1]);
        const ot_tiles: usize = @intCast(shape[2]);
        const tile_bytes: usize = @intCast(shape[3] * 2);
        const per_expert = it_tiles * ot_tiles * tile_bytes;
        for (0..@intCast(experts)) |e| {
            tile_layout.tileMajor(u8, data.items[at + e * per_expert ..][0..per_expert], data_in[start + e * per_expert ..][0..per_expert], it_tiles, ot_tiles, tile_bytes);
        }
        const swapped = fault == .shape_mismatch and std.mem.eql(u8, key, swapped_key);
        try header.print(allocator, "\"{s}_pm\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d},{d},2,{d}],\"data_offsets\":[{d},{d}]}}", .{
            key,
            dtype,
            experts,
            if (swapped) it_tiles else ot_tiles / 2,
            if (swapped) ot_tiles / 2 else it_tiles,
            shape[3],
            at,
            at + end - start,
        });
    }
    try header.append(allocator, '}');
    while ((8 + header.items.len) % (page orelse 8) != 0) try header.append(allocator, ' ');

    const out = try allocator.alloc(u8, 8 + header.items.len + data.items.len);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8..][0..header.items.len], header.items);
    @memcpy(out[8 + header.items.len ..], data.items);
    return out;
}

/// Writes the pack at `src` into the empty directory `dst`, its routed tensors pair-major and carrying `fault`;
/// everything the pack's tensors do not touch is linked to `src`.
pub fn write(allocator: std.mem.Allocator, src: []const u8, dst: []const u8, fault: Fault) !void {
    return writeWith(allocator, src, dst, fault, null, "");
}

/// `write`, each routed tensor starting on a `page` boundary of the file, as the repacker lays them.
pub fn writeAligned(allocator: std.mem.Allocator, src: []const u8, dst: []const u8, page: usize) !void {
    return writeWith(allocator, src, dst, .none, page, "");
}

/// `writeAligned`, except the routed tensors whose key contains `off_page` follow their neighbour with no pad.
pub fn writeAlignedExcept(allocator: std.mem.Allocator, src: []const u8, dst: []const u8, page: usize, off_page: []const u8) !void {
    return writeWith(allocator, src, dst, .none, page, off_page);
}

fn writeWith(allocator: std.mem.Allocator, src: []const u8, dst: []const u8, fault: Fault, page: ?usize, off_page: []const u8) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = try std.Io.Dir.openDirAbsolute(io, src, .{ .iterate = true });
    defer dir.close(io);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }
    var walk = dir.iterate();
    while (try walk.next(io)) |entry| try names.append(allocator, try allocator.dupe(u8, entry.name));

    for (names.items) |name| {
        if (std.mem.eql(u8, name, "config.json")) {
            const raw = try readAll(io, allocator, src, name);
            defer allocator.free(raw);
            const out = try rewriteConfig(allocator, raw, fault);
            defer allocator.free(out);
            try writeAll(io, dst, name, out);
        } else if (std.mem.eql(u8, name, "model.safetensors.index.json")) {
            const raw = try readAll(io, allocator, src, name);
            defer allocator.free(raw);
            const out = try rewriteIndex(allocator, raw, fault);
            defer allocator.free(out);
            try writeAll(io, dst, name, out);
        } else if (std.mem.startsWith(u8, name, "model") and std.mem.endsWith(u8, name, ".safetensors")) {
            const raw = try readAll(io, allocator, src, name);
            defer allocator.free(raw);
            const out = try rewriteShard(allocator, raw, fault, page, off_page);
            defer allocator.free(out);
            try writeAll(io, dst, name, out);
        } else {
            var from_buf: [std.fs.max_path_bytes]u8 = undefined;
            var to_buf: [std.fs.max_path_bytes]u8 = undefined;
            const from = try std.fmt.bufPrint(&from_buf, "{s}/{s}", .{ src, name });
            const to = try std.fmt.bufPrint(&to_buf, "{s}/{s}", .{ dst, name });
            try std.Io.Dir.symLinkAbsolute(io, from, to, .{});
        }
    }
}

/// `DSV41_STQ_TINY`, or null when the pack tier is not set up.
pub fn tinyPack() ?[]const u8 {
    const raw = std.c.getenv("DSV41_STQ_TINY") orelse return null;
    return std.mem.span(raw);
}
