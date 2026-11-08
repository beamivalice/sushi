//! A pack's safetensors index read from its shard headers (every tensor's dtype, shape and byte range, and each
//! shard's converter stamp), and the checks an EXL3 pack's routed banks take at load, before anything is read.
const std = @import("std");
const model = @import("model.zig");
const arch_config = @import("arch/config.zig");

const Allocator = std.mem.Allocator;

pub const MAX_HEADER_BYTES: u64 = 512 * 1024 * 1024;

pub const MAX_JSON_BYTES: usize = 512 * 1024 * 1024;

pub const DType = enum {
    bf16,
    f16,
    f32,
    u16,
    u32,
    fp8_e4m3,
    unknown,
};

pub const TensorMeta = struct {
    dtype: DType,
    shape: []const u64,
    data_start: u64,
    data_end: u64,
    data_base: u64,
    file: []const u8,
};

pub const SourceIndex = struct {
    weight_map: std.StringHashMap([]const u8),
    files: std.StringHashMap(void),
    tensors: std.StringHashMap(TensorMeta),
    /// Per shard, what the converter stamped into `__metadata__`.
    stamps: std.StringHashMap(ShardStamp),
};

/// The decoder a shard was written for, as sashimi stamps it
/// (docs/pack-format.md). Every value is a string there. A shard
/// naming none predates the stamp and is admitted; one that names a decoder
/// the config does not is refused, because the same bytes decode to different
/// weights under each.
pub const ShardStamp = struct {
    k: ?[]const u8 = null,
    codebook: ?[]const u8 = null,
    window: ?[]const u8 = null,
};

pub fn readFileAlloc(
    io: std.Io,
    allocator: Allocator,
    path: []const u8,
    limit: usize,
) ![]u8 {
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch
        return error.PackSourceFileMissing;
    defer file.close(io);
    var read_buf: [8192]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    return reader.interface.allocRemaining(allocator, .limited(limit)) catch
        return error.PackSourceRead;
}

pub fn preadExact(fd: std.c.fd_t, dst: []u8, offset: u64) !void {
    var done: usize = 0;
    while (done < dst.len) {
        const got = std.c.pread(
            fd,
            dst[done..].ptr,
            dst.len - done,
            @intCast(offset + done),
        );
        if (got < 0) {
            if (std.c._errno().* == @backingInt(std.c.E.INTR)) continue;
            return error.PackSourceRead;
        }
        if (got == 0) return error.SafetensorsUnexpectedEof;
        done += @intCast(got);
    }
}

pub fn shardPath(allocator: Allocator, model_dir: []const u8, file: []const u8) ![:0]u8 {
    return std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file }, 0);
}

pub const HeaderBytes = struct {
    bytes: []u8,
    data_base: u64,
};

pub fn readShardHeader(
    allocator: Allocator,
    model_dir: []const u8,
    file: []const u8,
) !HeaderBytes {
    const path = try shardPath(allocator, model_dir, file);
    defer allocator.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.MissingPackShard;
    defer _ = std.c.close(fd);

    var header_len_bytes: [8]u8 = undefined;
    try preadExact(fd, &header_len_bytes, 0);
    const header_len = std.mem.readInt(u64, &header_len_bytes, .little);
    if (header_len > MAX_HEADER_BYTES) return error.SafetensorsHeaderTooLarge;
    const header_len_usize = std.math.cast(usize, header_len) orelse
        return error.SafetensorsHeaderTooLarge;
    const header = try allocator.alloc(u8, header_len_usize);
    errdefer allocator.free(header);
    try preadExact(fd, header, 8);
    const data_base = std.math.add(u64, 8, header_len) catch
        return error.InvalidSafetensorsHeader;
    return .{ .bytes = header, .data_base = data_base };
}

pub fn parseDType(name: []const u8) DType {
    if (std.mem.eql(u8, name, "BF16")) return .bf16;
    if (std.mem.eql(u8, name, "F16")) return .f16;
    if (std.mem.eql(u8, name, "F32")) return .f32;
    if (std.mem.eql(u8, name, "U16")) return .u16;
    if (std.mem.eql(u8, name, "U32")) return .u32;
    if (std.mem.eql(u8, name, "F8_E4M3") or std.mem.eql(u8, name, "F8_E4M3FN"))
        return .fp8_e4m3;
    return .unknown;
}

pub fn parseShape(allocator: Allocator, value: std.json.Value) ![]u64 {
    if (value != .array) return error.InvalidSafetensorsHeader;
    const shape = try allocator.alloc(u64, value.array.items.len);
    errdefer allocator.free(shape);
    for (value.array.items, 0..) |dim, i| {
        if (dim != .integer or dim.integer < 0) return error.InvalidSafetensorsHeader;
        shape[i] = std.math.cast(u64, dim.integer) orelse
            return error.InvalidSafetensorsHeader;
    }
    return shape;
}

pub fn parseOffsets(value: std.json.Value) ![2]u64 {
    if (value != .array or value.array.items.len != 2) return error.InvalidSafetensorsHeader;
    var offsets: [2]u64 = undefined;
    for (value.array.items, 0..) |v, i| {
        if (v != .integer or v.integer < 0) return error.InvalidSafetensorsHeader;
        offsets[i] = std.math.cast(u64, v.integer) orelse
            return error.InvalidSafetensorsHeader;
    }
    if (offsets[1] < offsets[0]) return error.InvalidSafetensorsHeader;
    return offsets;
}

pub fn parseTensorMeta(
    allocator: Allocator,
    value: std.json.Value,
    file: []const u8,
    data_base: u64,
) !TensorMeta {
    if (value != .object) return error.InvalidSafetensorsHeader;
    const dtype_value = value.object.get("dtype") orelse return error.InvalidSafetensorsHeader;
    if (dtype_value != .string) return error.InvalidSafetensorsHeader;
    const shape_value = value.object.get("shape") orelse return error.InvalidSafetensorsHeader;
    const offsets_value = value.object.get("data_offsets") orelse
        return error.InvalidSafetensorsHeader;
    const shape = try parseShape(allocator, shape_value);
    errdefer allocator.free(shape);
    const offsets = try parseOffsets(offsets_value);
    _ = std.math.add(u64, data_base, offsets[1]) catch
        return error.InvalidSafetensorsHeader;
    return .{
        .dtype = parseDType(dtype_value.string),
        .shape = shape,
        .data_start = offsets[0],
        .data_end = offsets[1],
        .data_base = data_base,
        .file = file,
    };
}

pub fn loadSourceIndex(io: std.Io, allocator: Allocator, model_dir: []const u8) !SourceIndex {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir))
        return error.InvalidPackModelPath;

    const index_path = try std.fmt.allocPrint(
        allocator,
        "{s}/model.safetensors.index.json",
        .{model_dir},
    );
    const index_bytes = try readFileAlloc(io, allocator, index_path, MAX_JSON_BYTES);
    const parsed = std.json.parseFromSliceLeaky(
        std.json.Value,
        allocator,
        index_bytes,
        .{},
    ) catch return error.InvalidSafetensorsIndex;
    if (parsed != .object) return error.InvalidSafetensorsIndex;
    const weight_map_value = parsed.object.get("weight_map") orelse
        return error.InvalidSafetensorsIndex;
    if (weight_map_value != .object or weight_map_value.object.count() == 0)
        return error.InvalidSafetensorsIndex;

    var source = SourceIndex{
        .weight_map = std.StringHashMap([]const u8).init(allocator),
        .files = std.StringHashMap(void).init(allocator),
        .tensors = std.StringHashMap(TensorMeta).init(allocator),
        .stamps = std.StringHashMap(ShardStamp).init(allocator),
    };
    var it = weight_map_value.object.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.* != .string or entry.value_ptr.string.len == 0)
            return error.InvalidSafetensorsIndex;
        if (source.weight_map.contains(entry.key_ptr.*))
            return error.AmbiguousSafetensorsIndex;
        try source.weight_map.put(entry.key_ptr.*, entry.value_ptr.string);
        try source.files.put(entry.value_ptr.string, {});
    }

    var files = source.files.iterator();
    while (files.next()) |file_entry| {
        const filename = file_entry.key_ptr.*;
        var header_arena = std.heap.ArenaAllocator.init(allocator);
        defer header_arena.deinit();
        const header_scratch = header_arena.allocator();
        const header = try readShardHeader(header_scratch, model_dir, filename);
        const header_root = std.json.parseFromSliceLeaky(
            std.json.Value,
            header_scratch,
            header.bytes,
            .{},
        ) catch return error.InvalidSafetensorsHeader;
        if (header_root != .object) return error.InvalidSafetensorsHeader;
        try source.stamps.put(filename, try readShardStamp(allocator, header_root.object));

        // Index iteration is deliberate. It leaves all header-name slices in
        // the short-lived arena and copies only shapes for indexed tensors.
        var indexed = source.weight_map.iterator();
        while (indexed.next()) |indexed_entry| {
            if (!std.mem.eql(u8, indexed_entry.value_ptr.*, filename)) continue;
            const tensor_value = header_root.object.get(indexed_entry.key_ptr.*) orelse
                continue;
            if (source.tensors.contains(indexed_entry.key_ptr.*))
                return error.AmbiguousSafetensorsTensor;
            const meta = try parseTensorMeta(
                header_scratch,
                tensor_value,
                filename,
                header.data_base,
            );
            const shape = try allocator.dupe(u64, meta.shape);
            try source.tensors.put(indexed_entry.key_ptr.*, .{
                .dtype = meta.dtype,
                .shape = shape,
                .data_start = meta.data_start,
                .data_end = meta.data_end,
                .data_base = meta.data_base,
                .file = filename,
            });
        }
    }

    var indexed = source.weight_map.iterator();
    while (indexed.next()) |entry| {
        if (!source.tensors.contains(entry.key_ptr.*))
            return error.MissingIndexedSafetensorsTensor;
    }
    return source;
}

pub fn readShardStamp(allocator: Allocator, header: std.json.ObjectMap) !ShardStamp {
    const meta = header.get("__metadata__") orelse return .{};
    if (meta != .object) return .{};
    var out: ShardStamp = .{};
    inline for (.{ "k", "codebook", "window" }) |field| {
        if (meta.object.get(field)) |v| {
            if (v == .string) @field(out, field) = try allocator.dupe(u8, v.string);
        }
    }
    return out;
}

pub fn validateShardStamps(source: *const SourceIndex, config: *const model.ModelConfig) !void {
    if (config.expert_layout != .exl3_k4) return;
    const spec = @import("sushi_exl3").Spec{
        .rate = config.expert_quant_rate,
        .codebook = config.expert_quant_codebook,
        .window = config.expert_quant_window,
    };
    var it = source.stamps.valueIterator();
    while (it.next()) |stamp| {
        try @import("sushi_exl3").shardStampAgrees(spec, stamp.k, stamp.codebook, stamp.window);
    }
}

pub fn shapeProduct(shape: []const u64) !u64 {
    var product: u64 = 1;
    for (shape) |dim| {
        product = std.math.mul(u64, product, dim) catch
            return error.SafetensorsShapeOverflow;
    }
    return product;
}

pub fn payloadBytes(meta: TensorMeta, expected_dtype: ?DType) !u64 {
    if (expected_dtype) |want| if (meta.dtype != want) return error.PackTensorDtypeMismatch;
    const elem_size: u64 = switch (meta.dtype) {
        .bf16, .f16, .u16 => 2,
        .f32, .u32 => 4,
        .fp8_e4m3 => 1,
        .unknown => return error.UnsupportedPackTensorDtype,
    };
    const want = std.math.mul(u64, try shapeProduct(meta.shape), elem_size) catch
        return error.SafetensorsShapeOverflow;
    if (meta.data_end - meta.data_start != want) return error.SafetensorsPayloadMismatch;
    return want;
}

pub fn validateExl3Layer(source: *const SourceIndex, allocator: Allocator, config: *const model.ModelConfig, layer_prefix: []const u8) !@import("sushi_exl3").GroupLayout {
    var plan = @import("sushi_exl3").GroupLayout.init(config.hidden_size, config.moe_intermediate_size, config.expert_quant_rate);
    plan.rates = config.expert_quant_rates;
    const layout = arch_config.packLayout(config);
    plan.streams_groups = layout.streams_groups;
    const prefix = try std.fmt.allocPrint(allocator, "{s}.mlp.switch_mlp.", .{layer_prefix});
    const router = source.tensors.get(try std.fmt.allocPrint(allocator, "{s}.mlp.gate.weight", .{layer_prefix})) orelse return error.MissingPackRequiredTensor;
    if (router.shape.len != 2 or router.shape[0] > std.math.maxInt(u32)) return error.Exl3RouterWidthMismatch;
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        if (!std.mem.startsWith(u8, entry.key_ptr.*, prefix)) continue;
        const meta = entry.value_ptr.*;
        _ = try plan.add(entry.key_ptr.*[prefix.len..], meta.shape, switch (meta.dtype) {
            .u16 => .u16,
            .f16 => .f16,
            else => .other,
        });
        _ = try payloadBytes(meta, null);
    }
    try plan.finish(@intCast(router.shape[0]), config.num_experts_per_tok, config.expert_streaming, config.num_experts);
    if (config.moe_n_group > 1 and (plan.grouped.? or router.shape[0] != config.num_experts)) return error.Exl3RouterGroupsUnsupported;
    if (plan.grouped.? and layout.refuses_rate_groups) return error.Exl3RateGroupsUnsupported;
    return plan;
}

pub const TestTensor = struct {
    key: []const u8,
    dtype: []const u8,
    shape: []const u64,
    bytes: []const u8,
};

pub const TestIndexEntry = struct {
    key: []const u8,
    file: []const u8,
};

pub fn appendTestFormat(
    allocator: Allocator,
    list: *std.ArrayList(u8),
    comptime format: []const u8,
    args: anytype,
) !void {
    const text = try std.fmt.allocPrint(allocator, format, args);
    defer allocator.free(text);
    try list.appendSlice(allocator, text);
}

pub fn writeTestShard(
    io: std.Io,
    allocator: Allocator,
    dir: std.Io.Dir,
    filename: []const u8,
    tensors: []const TestTensor,
) !void {
    return writeTestShardStamped(io, allocator, dir, filename, tensors, null);
}

pub fn writeTestShardStamped(
    io: std.Io,
    allocator: Allocator,
    dir: std.Io.Dir,
    filename: []const u8,
    tensors: []const TestTensor,
    stamp: ?[]const u8,
) !void {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    try header.append(allocator, '{');
    if (stamp) |body| try appendTestFormat(allocator, &header, "\"__metadata__\":{{{s}}},", .{body});
    var data_size: usize = 0;
    for (tensors, 0..) |tensor, i| {
        if (i != 0) try header.append(allocator, ',');
        try appendTestFormat(
            allocator,
            &header,
            "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[",
            .{ tensor.key, tensor.dtype },
        );
        for (tensor.shape, 0..) |dim, j| {
            if (j != 0) try header.append(allocator, ',');
            try appendTestFormat(allocator, &header, "{d}", .{dim});
        }
        const end = std.math.add(usize, data_size, tensor.bytes.len) catch
            return error.TestFixtureTooLarge;
        try appendTestFormat(
            allocator,
            &header,
            "],\"data_offsets\":[{d},{d}]}}",
            .{ data_size, end },
        );
        data_size = end;
    }
    try header.append(allocator, '}');

    const total = std.math.add(usize, 8 + header.items.len, data_size) catch
        return error.TestFixtureTooLarge;
    const file_bytes = try allocator.alloc(u8, total);
    defer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.items.len, .little);
    @memcpy(file_bytes[8 .. 8 + header.items.len], header.items);
    var at = 8 + header.items.len;
    for (tensors) |tensor| {
        @memcpy(file_bytes[at..][0..tensor.bytes.len], tensor.bytes);
        at += tensor.bytes.len;
    }
    try dir.writeFile(io, .{ .sub_path = filename, .data = file_bytes });
}

pub fn writeTestIndex(
    io: std.Io,
    allocator: Allocator,
    dir: std.Io.Dir,
    entries: []const TestIndexEntry,
) !void {
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(allocator);
    try json.appendSlice(allocator, "{\"weight_map\":{");
    for (entries, 0..) |entry, i| {
        if (i != 0) try json.append(allocator, ',');
        try appendTestFormat(allocator, &json, "\"{s}\":\"{s}\"", .{ entry.key, entry.file });
    }
    try json.appendSlice(allocator, "}}");
    try dir.writeFile(io, .{
        .sub_path = "model.safetensors.index.json",
        .data = json.items,
    });
}

pub fn testBf16Bytes(allocator: Allocator, count: usize, bits: u16) ![]u8 {
    const bytes = try allocator.alloc(u8, count * 2);
    for (0..count) |i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], bits, .little);
    return bytes;
}

test "EXL3 shard stamp rejects invalid and nonfinite rates" {
    var source: SourceIndex = undefined;
    source.stamps = std.StringHashMap(ShardStamp).init(std.testing.allocator);
    defer source.stamps.deinit();
    var config: model.ModelConfig = undefined;
    config.expert_layout = .exl3_k4;
    config.expert_quant_rate = .{ .n = 64 };
    for ([_][]const u8{ "0", "0.5", "2.0625", "4.5", "nan", "inf", "-inf" }) |k| {
        try source.stamps.put("expert.safetensors", .{ .k = k });
        try std.testing.expectError(error.Exl3ShardStampMismatch, validateShardStamps(&source, &config));
    }
    for ([_][]const u8{ "1", "1.5", "2", "2.125", "2.5", "3", "4" }) |k| {
        try source.stamps.put("expert.safetensors", .{ .k = k });
        try validateShardStamps(&source, &config);
    }
}

pub fn validateExl3Pack(io: std.Io, allocator: Allocator, model_dir: []const u8, config: *const model.ModelConfig) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const source = try loadSourceIndex(io, alloc, model_dir);
    try validateExl3Source(&source, alloc, config);
}

pub fn validateExl3Source(source: *const SourceIndex, alloc: Allocator, config: *const model.ModelConfig) !void {
    const layout = arch_config.packLayout(config);
    if (layout.stamped) try validateShardStamps(source, config);
    const prefix = layout.prefix;
    for (config.first_k_dense_replace..config.num_hidden_layers) |layer| {
        const base = try std.fmt.allocPrint(alloc, "{s}.layers.{d}", .{ prefix, layer });
        _ = try validateExl3Layer(source, alloc, config, base);
    }
    switch (layout.head) {
        .none => {},
        .trailing => {
            const layer_end = @as(u64, config.num_hidden_layers) + arch_config.trailingPackLayers(config);
            var keys = source.tensors.keyIterator();
            const start = try std.fmt.allocPrint(alloc, "{s}.layers.", .{prefix});
            while (keys.next()) |key| {
                if (!std.mem.startsWith(u8, key.*, start) or std.mem.indexOf(u8, key.*, ".mlp.switch_mlp.") == null) continue;
                const rest = key.*[start.len..];
                const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.PackLayerOutOfRange;
                const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return error.PackLayerOutOfRange;
                if (layer < config.first_k_dense_replace or @as(u64, layer) >= layer_end) return error.PackLayerOutOfRange;
            }
            for (config.num_hidden_layers..@intCast(layer_end)) |layer| {
                const base = try std.fmt.allocPrint(alloc, "{s}.layers.{d}", .{ prefix, layer });
                const probe = try std.fmt.allocPrint(alloc, "{s}.mlp.switch_mlp.", .{base});
                var parts = source.tensors.keyIterator();
                while (parts.next()) |key| {
                    if (std.mem.startsWith(u8, key.*, probe)) {
                        _ = try validateExl3Layer(source, alloc, config, base);
                        break;
                    }
                }
            }
        },
        .layer => |head| {
            const probe = try std.fmt.allocPrint(alloc, "{s}.", .{head});
            var it = source.tensors.keyIterator();
            while (it.next()) |key| {
                if (std.mem.startsWith(u8, key.*, probe)) {
                    _ = try validateExl3Layer(source, alloc, config, head);
                    break;
                }
            }
        },
    }
}
