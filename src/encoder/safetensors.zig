const std = @import("std");

pub const tiles = aligned(@embedFile("fixtures/exl3_encode_tiles.safetensors"));
pub const tensor = aligned(@embedFile("fixtures/exl3_encode_tensor.safetensors"));
pub const sources = aligned(@embedFile("fixtures/exl3_encode_sources.safetensors"));

fn aligned(comptime raw: []const u8) *align(8) const [raw.len]u8 {
    const holder = struct {
        const value: [raw.len]u8 align(8) = raw[0..raw.len].*;
    };
    return &holder.value;
}

pub const TensorView = struct {
    dtype: []const u8,
    shape: []const usize,
    bytes: []const u8,
};

pub const File = struct {
    tensors: std.StringHashMap(TensorView),
    metadata: std.StringHashMap([]const u8),

    pub fn deinit(self: *File, allocator: std.mem.Allocator) void {
        var it = self.tensors.iterator();
        while (it.next()) |e| {
            allocator.free(e.key_ptr.*);
            allocator.free(e.value_ptr.dtype);
            allocator.free(e.value_ptr.shape);
        }
        self.tensors.deinit();
        var meta = self.metadata.iterator();
        while (meta.next()) |e| {
            allocator.free(e.key_ptr.*);
            allocator.free(e.value_ptr.*);
        }
        self.metadata.deinit();
    }
};

pub fn parse(allocator: std.mem.Allocator, raw: []const u8) !File {
    if (raw.len < 8) return error.TruncatedSafetensors;
    const header_len = std.mem.readInt(u64, raw[0..8], .little);
    if (8 + header_len > raw.len) return error.TruncatedSafetensors;
    const header = raw[8 .. 8 + header_len];
    const data = raw[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, header, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadSafetensorsHeader;
    var tensors = std.StringHashMap(TensorView).init(allocator);
    errdefer tensors.deinit();
    var metadata = std.StringHashMap([]const u8).init(allocator);
    errdefer metadata.deinit();
    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) {
            if (entry.value_ptr.* != .object) return error.BadSafetensorsHeader;
            var meta = entry.value_ptr.*.object.iterator();
            while (meta.next()) |m| {
                if (m.value_ptr.* != .string) return error.BadSafetensorsHeader;
                try metadata.put(try allocator.dupe(u8, m.key_ptr.*), try allocator.dupe(u8, m.value_ptr.*.string));
            }
            continue;
        }
        if (entry.value_ptr.* != .object) continue;
        const obj = entry.value_ptr.*.object;
        const dtype = obj.get("dtype") orelse continue;
        const shape_v = obj.get("shape") orelse continue;
        const offsets = obj.get("data_offsets") orelse continue;
        if (dtype != .string or shape_v != .array or offsets != .array or offsets.array.items.len != 2) continue;
        const start: usize = @intCast(offsets.array.items[0].integer);
        const end: usize = @intCast(offsets.array.items[1].integer);
        if (end > data.len or start > end) return error.TruncatedSafetensors;
        var shape = try allocator.alloc(usize, shape_v.array.items.len);
        for (shape_v.array.items, 0..) |d, i| shape[i] = @intCast(d.integer);
        try tensors.put(try allocator.dupe(u8, entry.key_ptr.*), .{
            .dtype = try allocator.dupe(u8, dtype.string),
            .shape = shape,
            .bytes = data[start..end],
        });
    }
    return .{ .tensors = tensors, .metadata = metadata };
}

pub fn slice(comptime T: type, view: TensorView) []const T {
    return @alignCast(std.mem.bytesAsSlice(T, view.bytes));
}
