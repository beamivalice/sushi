//! A GGUF v3 container reader: the header, every metadata value type, the tensor infos and the
//! aligned data section. GLP control vectors (`src/glp.zig`) are GGUF v3 files; nothing else in
//! sushi reads GGUF, so this is the one parser.
//!
//! The reader borrows the bytes it is given: metadata strings and tensor data are slices of
//! `raw`, which the caller keeps alive for the `File`'s life. Keys, array items and tensor infos
//! are owned and freed by `deinit`.
const std = @import("std");

pub const Error = error{
    NotGguf,
    UnsupportedVersion,
    Truncated,
    BadValueType,
    BadTensorType,
    BadAlignment,
    TensorOutOfBounds,
    DuplicateKey,
    DuplicateTensor,
    TooManyDimensions,
};

/// GGUF's metadata value types (`gguf_metadata_value_type`).
pub const ValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool_ = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
    _,
};

pub const Array = struct {
    elem_type: ValueType,
    items: []Value,
};

pub const Value = union(enum) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool_: bool,
    string: []const u8,
    array: Array,
    uint64: u64,
    int64: i64,
    float64: f64,
};

pub const TensorInfo = struct {
    name: []const u8,
    dims: []u64,
    /// The ggml tensor type (`ggml_type`); 0 is F32.
    type_: u32,
    /// Bytes from the start of the data section.
    offset: u64,
};

pub const File = struct {
    allocator: std.mem.Allocator,
    version: u32,
    metadata: std.StringHashMapUnmanaged(Value) = .empty,
    tensors: []TensorInfo = &.{},
    /// The data section, from its first aligned byte.
    data: []const u8 = &.{},
    alignment: usize = 32,

    pub fn deinit(self: *File) void {
        var it = self.metadata.iterator();
        while (it.next()) |e| {
            self.allocator.free(e.key_ptr.*);
            freeValue(self.allocator, e.value_ptr.*);
        }
        self.metadata.deinit(self.allocator);
        for (self.tensors) |t| {
            self.allocator.free(t.name);
            self.allocator.free(t.dims);
        }
        self.allocator.free(self.tensors);
        self.* = undefined;
    }

    pub fn get(self: *const File, key: []const u8) ?Value {
        return self.metadata.get(key);
    }

    pub fn tensor(self: *const File, name: []const u8) ?*const TensorInfo {
        for (self.tensors) |*t| if (std.mem.eql(u8, t.name, name)) return t;
        return null;
    }

    /// The tensor's bytes in the data section. Only the types GLP needs are sized; a type this
    /// reader cannot size is refused rather than guessed.
    pub fn tensorBytes(self: *const File, t: *const TensorInfo) Error![]const u8 {
        const elem = typeSize(t.type_) orelse return error.BadTensorType;
        var count: u64 = 1;
        for (t.dims) |d| count = std.math.mul(u64, count, d) catch return error.TensorOutOfBounds;
        const bytes = std.math.mul(u64, count, elem) catch return error.TensorOutOfBounds;
        const start = std.math.cast(usize, t.offset) orelse return error.TensorOutOfBounds;
        const len = std.math.cast(usize, bytes) orelse return error.TensorOutOfBounds;
        if (start > self.data.len or len > self.data.len - start) return error.TensorOutOfBounds;
        return self.data[start..][0..len];
    }
};

/// The byte width of one element of a ggml tensor type; null for a type this reader does not size.
pub fn typeSize(t: u32) ?u64 {
    return switch (t) {
        0 => 4, // F32
        1 => 2, // F16
        30 => 2, // BF16
        else => null,
    };
}

fn freeValue(a: std.mem.Allocator, v: Value) void {
    switch (v) {
        .array => |arr| {
            for (arr.items) |item| freeValue(a, item);
            a.free(arr.items);
        },
        else => {},
    }
}

const Cursor = struct {
    buf: []const u8,
    pos: usize = 0,

    fn take(self: *Cursor, n: usize) Error![]const u8 {
        if (n > self.buf.len - self.pos) return error.Truncated;
        const s = self.buf[self.pos..][0..n];
        self.pos += n;
        return s;
    }

    fn int(self: *Cursor, comptime T: type) Error!T {
        const s = try self.take(@sizeOf(T));
        return std.mem.readInt(T, s[0..@sizeOf(T)], .little);
    }

    fn readU32(self: *Cursor) Error!u32 {
        return self.int(u32);
    }

    fn readU64(self: *Cursor) Error!u64 {
        return self.int(u64);
    }

    fn string(self: *Cursor) Error![]const u8 {
        const n = try self.readU64();
        const len = std.math.cast(usize, n) orelse return error.Truncated;
        return self.take(len);
    }

    fn value(self: *Cursor, a: std.mem.Allocator, vt: ValueType) !Value {
        return switch (vt) {
            .uint8 => .{ .uint8 = (try self.take(1))[0] },
            .int8 => .{ .int8 = @bitCast((try self.take(1))[0]) },
            .uint16 => .{ .uint16 = try self.int(u16) },
            .int16 => .{ .int16 = try self.int(i16) },
            .uint32 => .{ .uint32 = try self.int(u32) },
            .int32 => .{ .int32 = try self.int(i32) },
            .float32 => .{ .float32 = @bitCast(try self.int(u32)) },
            .bool_ => .{ .bool_ = (try self.take(1))[0] != 0 },
            .string => .{ .string = try self.string() },
            .array => blk: {
                const et_raw = try self.readU32();
                const et = std.enums.fromInt(ValueType, et_raw) orelse return error.BadValueType;
                const n = try self.readU64();
                const len = std.math.cast(usize, n) orelse return error.Truncated;
                const items = try a.alloc(Value, len);
                var j: usize = 0;
                errdefer {
                    for (items[0..j]) |item| freeValue(a, item);
                    a.free(items);
                }
                while (j < len) : (j += 1) items[j] = try self.value(a, et);
                break :blk .{ .array = .{ .elem_type = et, .items = items } };
            },
            .uint64 => .{ .uint64 = try self.int(u64) },
            .int64 => .{ .int64 = try self.int(i64) },
            .float64 => .{ .float64 = @bitCast(try self.int(u64)) },
            else => error.BadValueType,
        };
    }
};

pub fn parse(allocator: std.mem.Allocator, raw: []const u8) !File {
    var c = Cursor{ .buf = raw };
    if (try c.readU32() != 0x46554747) return error.NotGguf; // "GGUF"
    const version = try c.readU32();
    if (version != 3) return error.UnsupportedVersion;
    const tensor_count = try c.readU64();
    const kv_count = try c.readU64();

    var f = File{ .allocator = allocator, .version = version };
    errdefer f.deinit();

    var i: u64 = 0;
    while (i < kv_count) : (i += 1) {
        const key = try c.string();
        const vt_raw = try c.readU32();
        const vt = std.enums.fromInt(ValueType, vt_raw) orelse return error.BadValueType;
        const val = try c.value(allocator, vt);
        errdefer freeValue(allocator, val);
        const owned_key = try allocator.dupe(u8, key);
        errdefer allocator.free(owned_key);
        const gop = try f.metadata.getOrPut(allocator, owned_key);
        if (gop.found_existing) return error.DuplicateKey;
        gop.value_ptr.* = val;
    }

    if (f.metadata.get("general.alignment")) |v| switch (v) {
        .uint32 => |n| f.alignment = n,
        .uint64 => |n| f.alignment = std.math.cast(usize, n) orelse return error.BadAlignment,
        else => return error.BadAlignment,
    };
    if (f.alignment == 0 or !std.math.isPowerOfTwo(f.alignment)) return error.BadAlignment;

    const n_tensors = std.math.cast(usize, tensor_count) orelse return error.Truncated;
    var tensors: std.ArrayList(TensorInfo) = .empty;
    errdefer {
        for (tensors.items) |t| {
            allocator.free(t.name);
            allocator.free(t.dims);
        }
        tensors.deinit(allocator);
    }
    var built: usize = 0;
    while (built < n_tensors) : (built += 1) {
        const name = try c.string();
        const n_dims = try c.readU32();
        const dims = try allocator.alloc(u64, @intCast(n_dims));
        errdefer allocator.free(dims);
        for (dims) |*d| d.* = try c.readU64();
        const type_ = try c.readU32();
        const offset = try c.readU64();
        const owned_name = try allocator.dupe(u8, name);
        errdefer allocator.free(owned_name);
        try tensors.append(allocator, .{
            .name = owned_name,
            .dims = dims,
            .type_ = type_,
            .offset = offset,
        });
    }
    f.tensors = try tensors.toOwnedSlice(allocator);

    const data_start = std.mem.alignForward(usize, c.pos, f.alignment);
    if (data_start > raw.len) return error.Truncated;
    f.data = raw[data_start..];
    return f;
}

test "a GGUF v3 header parses every metadata value type and the aligned data section" {
    const a = std.testing.allocator;
    var buf: std.Io.Writer.Allocating = .init(a);
    defer buf.deinit();
    const w = &buf.writer;

    try w.writeInt(u32, 0x46554747, .little);
    try w.writeInt(u32, 3, .little);
    try w.writeInt(u64, 1, .little); // tensors
    try w.writeInt(u64, 14, .little); // kv pairs

    try writeString(w, "general.alignment");
    try w.writeInt(u32, @backingInt(ValueType.uint32), .little);
    try w.writeInt(u32, 32, .little);

    try writeString(w, "u8");
    try w.writeInt(u32, @backingInt(ValueType.uint8), .little);
    try w.writeByte(7);
    try writeString(w, "i8");
    try w.writeInt(u32, @backingInt(ValueType.int8), .little);
    try w.writeByte(@bitCast(@as(i8, -3)));
    try writeString(w, "u16");
    try w.writeInt(u32, @backingInt(ValueType.uint16), .little);
    try w.writeInt(u16, 513, .little);
    try writeString(w, "i16");
    try w.writeInt(u32, @backingInt(ValueType.int16), .little);
    try w.writeInt(i16, -513, .little);
    try writeString(w, "u32");
    try w.writeInt(u32, @backingInt(ValueType.uint32), .little);
    try w.writeInt(u32, 70000, .little);
    try writeString(w, "i32");
    try w.writeInt(u32, @backingInt(ValueType.int32), .little);
    try w.writeInt(i32, -70000, .little);
    try writeString(w, "f32");
    try w.writeInt(u32, @backingInt(ValueType.float32), .little);
    try w.writeInt(u32, @bitCast(@as(f32, 1.5)), .little);
    try writeString(w, "bool");
    try w.writeInt(u32, @backingInt(ValueType.bool_), .little);
    try w.writeByte(1);
    try writeString(w, "str");
    try w.writeInt(u32, @backingInt(ValueType.string), .little);
    try writeString(w, "hello");
    try writeString(w, "u64");
    try w.writeInt(u32, @backingInt(ValueType.uint64), .little);
    try w.writeInt(u64, 1 << 40, .little);
    try writeString(w, "i64");
    try w.writeInt(u32, @backingInt(ValueType.int64), .little);
    try w.writeInt(i64, -(1 << 40), .little);
    try writeString(w, "f64");
    try w.writeInt(u32, @backingInt(ValueType.float64), .little);
    try w.writeInt(u64, @bitCast(@as(f64, 2.5)), .little);
    try writeString(w, "arr");
    try w.writeInt(u32, @backingInt(ValueType.array), .little);
    try w.writeInt(u32, @backingInt(ValueType.uint32), .little);
    try w.writeInt(u64, 3, .little);
    for ([_]u32{ 4, 5, 6 }) |v| try w.writeInt(u32, v, .little);

    // One F32 tensor, 2 x 3, at offset 0.
    try writeString(w, "direction.1");
    try w.writeInt(u32, 2, .little);
    try w.writeInt(u64, 2, .little);
    try w.writeInt(u64, 3, .little);
    try w.writeInt(u32, 0, .little);
    try w.writeInt(u64, 0, .little);

    // Pad to the 32-byte data alignment, then the tensor's six floats.
    while (buf.written().len % 32 != 0) try w.writeByte(0);
    for ([_]f32{ 1, 2, 3, 4, 5, 6 }) |v| try w.writeInt(u32, @bitCast(v), .little);

    var f = try parse(a, buf.written());
    defer f.deinit();

    try std.testing.expectEqual(@as(u32, 3), f.version);
    try std.testing.expectEqual(@as(usize, 32), f.alignment);
    try std.testing.expectEqual(@as(u8, 7), f.get("u8").?.uint8);
    try std.testing.expectEqual(@as(i8, -3), f.get("i8").?.int8);
    try std.testing.expectEqual(@as(u16, 513), f.get("u16").?.uint16);
    try std.testing.expectEqual(@as(i16, -513), f.get("i16").?.int16);
    try std.testing.expectEqual(@as(u32, 70000), f.get("u32").?.uint32);
    try std.testing.expectEqual(@as(i32, -70000), f.get("i32").?.int32);
    try std.testing.expectEqual(@as(f32, 1.5), f.get("f32").?.float32);
    try std.testing.expect(f.get("bool").?.bool_);
    try std.testing.expectEqualStrings("hello", f.get("str").?.string);
    try std.testing.expectEqual(@as(u64, 1 << 40), f.get("u64").?.uint64);
    try std.testing.expectEqual(@as(i64, -(1 << 40)), f.get("i64").?.int64);
    try std.testing.expectEqual(@as(f64, 2.5), f.get("f64").?.float64);
    try std.testing.expectEqual(@as(usize, 3), f.get("arr").?.array.items.len);
    try std.testing.expectEqual(@as(u32, 5), f.get("arr").?.array.items[1].uint32);

    const t = f.tensor("direction.1").?;
    try std.testing.expectEqual(@as(usize, 2), t.dims.len);
    try std.testing.expectEqual(@as(u64, 3), t.dims[1]);
    const bytes = try f.tensorBytes(t);
    try std.testing.expectEqual(@as(usize, 24), bytes.len);
    const floats = std.mem.bytesAsSlice(f32, bytes);
    try std.testing.expectEqual(@as(f32, 6), floats[5]);
}

test "a GGUF that is not v3, or is truncated, is refused" {
    const a = std.testing.allocator;
    var buf: std.Io.Writer.Allocating = .init(a);
    defer buf.deinit();
    const w = &buf.writer;
    try w.writeInt(u32, 0x46554747, .little);
    try w.writeInt(u32, 2, .little);
    try w.writeInt(u64, 0, .little);
    try w.writeInt(u64, 0, .little);
    try std.testing.expectError(error.UnsupportedVersion, parse(a, buf.written()));
    try std.testing.expectError(error.Truncated, parse(a, buf.written()[0..6]));
    try std.testing.expectError(error.NotGguf, parse(a, "nope"));
}

fn writeString(w: anytype, s: []const u8) !void {
    try w.writeInt(u64, s.len, .little);
    try w.writeAll(s);
}
