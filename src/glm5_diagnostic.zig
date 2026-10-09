//! Native GLM weight selection, loading and resident bills.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const Arr = mlx.mlx_array;

fn keepTextKey(name: []const u8, layers: usize) bool {
    const prefix = "model.language_model.layers.";
    if (std.mem.startsWith(u8, name, prefix)) {
        const tail = name[prefix.len..];
        const end = std.mem.indexOfScalar(u8, tail, '.') orelse return false;
        const layer = std.fmt.parseInt(usize, tail[0..end], 10) catch return false;
        if (layer >= layers) return false;
    }
    if (std.mem.indexOf(u8, name, ".mtp.") != null or std.mem.startsWith(u8, name, "mtp.")) return false;
    return std.mem.startsWith(u8, name, "model.language_model.") or std.mem.startsWith(u8, name, "lm_head.");
}

pub fn storedBytes(weights: *const model.Weights) u64 {
    var total: u64 = 0;
    var it = weights.map.valueIterator();
    while (it.next()) |v| total += mlx.mlx_array_size(v.*) * mlx.mlx_array_itemsize(v.*);
    return total;
}

/// Complete assistant safetensors payload, following its index when present.
/// DFlash2's native loader retains the stored tensor precision.
pub fn assistantResidentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !u64 {
    _ = allocator;
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    var referenced = @import("model_discovery.zig").indexShardSet(io, dir);
    defer if (referenced) |*r| @import("model_discovery.zig").freeShardSet(r);
    var files = dir.iterate();
    var bytes: u64 = 0;
    while (try files.next(io)) |entry| {
        if ((entry.kind != .file and entry.kind != .sym_link) or !std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) continue;
        const file = try dir.openFile(io, entry.name, .{});
        defer file.close(io);
        const stat = try file.stat(io);
        var buffer: [8]u8 = undefined;
        var reader = file.reader(io, &buffer);
        const header = try reader.interface.takeInt(u64, .little);
        if (header == 0 or header > 128 * 1024 * 1024 or header > stat.size -| 8) return error.InvalidSafetensorsHeader;
        bytes = try std.math.add(u64, bytes, stat.size - header - 8);
    }
    if (bytes == 0) return error.MissingIndexedGlmWeight;
    return bytes;
}

pub fn loadWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: mlx.mlx_stream) !model.Weights {
    return loadWeightsBounded(io, allocator, model_dir, s, false, std.math.maxInt(u64));
}

fn keepLoadKey(name: []const u8, layers: usize, trunk_only: bool, vision: bool) bool {
    if (vision and std.mem.startsWith(u8, name, "model.visual.")) return true;
    return keepTextKey(name, layers) and (!trunk_only or (std.mem.indexOf(u8, name, ".mlp.experts.") == null and std.mem.indexOf(u8, name, ".mlp.switch_mlp.") == null));
}

/// The same indexed text tensors the native loader retains, before allocating MLX arrays.
/// Counts payloads, excluding vision, extra prediction layers and unindexed shard contents.
pub fn residentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, layers: usize) !u64 {
    return residentBytesWithVision(io, allocator, model_dir, layers, false);
}

/// Counts exactly the enabled trunk and vision tensors retained by the loader.
pub fn residentBytesWithVision(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, layers: usize, vision: bool) !u64 {
    return selectedBytes(io, allocator, model_dir, layers, false, vision);
}

/// What a streamed load keeps resident: the text trunk without its routed experts, and the tower it may add.
pub fn streamedSplit(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, layers: usize) !model.ResidentSplit {
    const trunk = try selectedBytes(io, allocator, model_dir, layers, true, false);
    const with_tower = try selectedBytes(io, allocator, model_dir, layers, true, true);
    return .{ .trunk = trunk, .mtp = 0, .vision = with_tower - trunk };
}

fn selectedBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, layers: usize, trunk_only: bool, vision: bool) !u64 {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
    defer dir.close(io);
    const raw = try dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(raw);
    const index = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer index.deinit();
    if (index.value != .object) return error.InvalidGlmWeightIndex;
    const wm = index.value.object.get("weight_map") orelse return error.InvalidGlmWeightIndex;
    if (wm != .object) return error.InvalidGlmWeightIndex;
    var files = std.StringHashMap(void).init(allocator);
    defer files.deinit();
    var it = wm.object.iterator();
    var expected: usize = 0;
    while (it.next()) |entry| {
        if (!keepLoadKey(entry.key_ptr.*, layers, trunk_only, vision)) continue;
        const value = entry.value_ptr.*;
        if (value != .string or value.string.len == 0 or std.mem.indexOfAny(u8, value.string, "/\\") != null or std.mem.eql(u8, value.string, "..")) return error.InvalidGlmShardName;
        try files.put(value.string, {});
        expected += 1;
    }
    if (expected == 0) return error.MissingIndexedGlmWeight;
    var found: usize = 0;
    var total: u64 = 0;
    var file_it = files.keyIterator();
    while (file_it.next()) |file| {
        const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file.* }, 0);
        defer allocator.free(path);
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.MissingIndexedGlmWeight;
        defer _ = std.c.close(fd);
        var size: [8]u8 = undefined;
        try @import("expert_io.zig").readExact(fd, &size, 0);
        const len = std.mem.readInt(u64, &size, .little);
        if (len == 0 or len > 128 * 1024 * 1024) return error.InvalidSafetensorsHeader;
        const raw_header = try allocator.alloc(u8, @intCast(len));
        defer allocator.free(raw_header);
        try @import("expert_io.zig").readExact(fd, raw_header, 8);
        const header = try std.json.parseFromSlice(std.json.Value, allocator, raw_header, .{});
        defer header.deinit();
        if (header.value != .object) return error.InvalidSafetensorsHeader;
        var tensors = header.value.object.iterator();
        while (tensors.next()) |entry| {
            const name = entry.key_ptr.*;
            if (!keepLoadKey(name, layers, trunk_only, vision)) continue;
            const owner = wm.object.get(name) orelse continue;
            if (owner != .string or !std.mem.eql(u8, owner.string, file.*)) continue;
            if (entry.value_ptr.* != .object) return error.InvalidSafetensorsTensor;
            const offsets = entry.value_ptr.object.get("data_offsets") orelse return error.InvalidSafetensorsTensor;
            if (offsets != .array or offsets.array.items.len != 2) return error.InvalidSafetensorsTensor;
            const lo = offsets.array.items[0];
            const hi = offsets.array.items[1];
            if (lo != .integer or hi != .integer or lo.integer < 0 or hi.integer < lo.integer) return error.InvalidSafetensorsTensor;
            total = try std.math.add(u64, total, @intCast(hi.integer - lo.integer));
            found += 1;
        }
    }
    if (found != expected) return error.MissingIndexedGlmWeight;
    return total;
}

/// Upload selected payloads directly. MLX's lazy Load nodes retain a descriptor
/// per shard, which exceeds a terminal's default limit on finely sharded packs.
/// One descriptor and one temporary payload suffice here. Source E4M3 codes use
/// U8 storage as in mimo_source; no tensor is converted or requantized.
fn loadStoredShard(allocator: std.mem.Allocator, reader: *@import("expert_io.zig").ParallelReader, path: [:0]const u8, file: []const u8, owners: std.json.ObjectMap, layers: usize, trunk_only: bool, vision: bool, result: *model.Weights, max_bytes: u64) !void {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) {
        const code = std.c._errno().*;
        @import("log.zig").err("[glm-loader] cannot open {s}: errno {d}\n", .{ path, code });
        return switch (code) {
            @backingInt(std.c.E.MFILE) => error.ProcessFdQuotaExceeded,
            @backingInt(std.c.E.NFILE) => error.SystemFdQuotaExceeded,
            @backingInt(std.c.E.NOENT) => error.MissingIndexedGlmWeight,
            else => error.GlmShardOpenFailed,
        };
    }
    defer _ = std.c.close(fd);
    const expert_io = @import("expert_io.zig");
    expert_io.applyReadHints(fd, .{ .readahead_off = false });
    var size: [8]u8 = undefined;
    const readExact = expert_io.readExact;
    try readExact(fd, &size, 0);
    const len = std.mem.readInt(u64, &size, .little);
    if (len == 0 or len > 128 * 1024 * 1024) return error.InvalidSafetensorsHeader;
    const raw_header = try allocator.alloc(u8, @intCast(len));
    defer allocator.free(raw_header);
    try readExact(fd, raw_header, 8);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw_header, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSafetensorsHeader;
    var tensors = parsed.value.object.iterator();
    var bytes = storedBytes(result);
    while (tensors.next()) |entry| {
        const name = entry.key_ptr.*;
        const owner = owners.get(name) orelse continue;
        if (!keepLoadKey(name, layers, trunk_only, vision) or owner != .string or !std.mem.eql(u8, owner.string, file)) continue;
        if (entry.value_ptr.* != .object) return error.InvalidSafetensorsTensor;
        const meta = entry.value_ptr.object;
        const dtype = meta.get("dtype") orelse return error.InvalidSafetensorsTensor;
        const dims = meta.get("shape") orelse return error.InvalidSafetensorsTensor;
        const offsets = meta.get("data_offsets") orelse return error.InvalidSafetensorsTensor;
        if (dtype != .string or dims != .array or dims.array.items.len > 8 or offsets != .array or offsets.array.items.len != 2) return error.InvalidSafetensorsTensor;
        const dt: mlx.mlx_dtype = if (isFp8Dtype(dtype.string)) .uint8 else if (std.mem.eql(u8, dtype.string, "BF16")) .bfloat16 else if (std.mem.eql(u8, dtype.string, "F32")) .float32 else if (std.mem.eql(u8, dtype.string, "F16")) .float16 else if (std.mem.eql(u8, dtype.string, "U16")) .uint16 else if (std.mem.eql(u8, dtype.string, "U32")) .uint32 else return error.UnsupportedGlmStorage;
        const itemsize: u64 = switch (dt) {
            .uint8 => 1,
            .uint16, .bfloat16, .float16 => 2,
            else => 4,
        };
        var shape: [8]c_int = undefined;
        var expected: u64 = itemsize;
        for (dims.array.items, 0..) |dim, i| {
            if (dim != .integer or dim.integer <= 0 or dim.integer > std.math.maxInt(c_int)) return error.InvalidSafetensorsTensor;
            shape[i] = @intCast(dim.integer);
            expected = try std.math.mul(u64, expected, @intCast(dim.integer));
        }
        const lo = offsets.array.items[0];
        const hi = offsets.array.items[1];
        if (lo != .integer or hi != .integer or lo.integer < 0 or hi.integer < lo.integer or @as(u64, @intCast(hi.integer - lo.integer)) != expected) return error.InvalidSafetensorsTensor;
        bytes = try std.math.add(u64, bytes, expected);
        if (bytes > max_bytes) return error.GlmResidentBudgetExceeded;
        const read = try model.readTensorArray(reader, fd, try std.math.add(u64, len + 8, @intCast(lo.integer)), @intCast(expected), shape[0..dims.array.items.len], dt);
        const value = read.array;
        errdefer _ = mlx.mlx_array_free(value);
        const raw = read.bytes;
        if (isFp8Dtype(dtype.string)) {
            for (raw) |code| if (code & 0x7f == 0x7f) return error.InvalidFp8Value;
        } else if (std.mem.endsWith(u8, name, ".weight_scale_inv")) {
            if (dt != .float32) return error.InvalidFp8Scale;
            const bf16_max: f32 = @bitCast(@as(u32, 0x7f7f0000));
            for (0..raw.len / 4) |i| {
                const scale: f32 = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
                if (!std.math.isFinite(scale) or @abs(scale) * 448.0 > bf16_max) return error.InvalidFp8Scale;
            }
        }
        if (result.get(name) != null) return error.DuplicateGlmWeight;
        const key = try allocator.dupe(u8, name);
        errdefer allocator.free(key);
        try result.map.put(key, value);
    }
}

fn isFp8Dtype(dtype: []const u8) bool {
    return std.mem.eql(u8, dtype, "F8_E4M3") or std.mem.eql(u8, dtype, "F8_E4M3FN");
}

/// The budget is checked before each selected payload is read or uploaded.
/// Expert-only shards are never opened; mixed shards materialize only trunk tensors.
pub fn loadWeightsWithVision(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: mlx.mlx_stream, vision: bool) !model.Weights {
    return loadWeightsBoundedWithVision(io, allocator, model_dir, s, false, std.math.maxInt(u64), vision);
}

pub fn loadWeightsBounded(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: mlx.mlx_stream, trunk_only: bool, max_bytes: u64) !model.Weights {
    return loadWeightsBoundedWithVision(io, allocator, model_dir, s, trunk_only, max_bytes, false);
}

pub fn loadWeightsBoundedWithVision(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: mlx.mlx_stream, trunk_only: bool, max_bytes: u64, vision: bool) !model.Weights {
    _ = s; // Upload leaves only; unified storage is consumed by GPU operations.
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
    defer dir.close(io);
    const config_raw = try dir.readFileAlloc(io, "config.json", allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(config_raw);
    const config = try std.json.parseFromSlice(std.json.Value, allocator, config_raw, .{});
    defer config.deinit();
    if (config.value != .object) return error.InvalidGlmConfig;
    const text_config = config.value.object.get("text_config") orelse config.value;
    if (text_config != .object) return error.InvalidGlmConfig;
    const layer_value = text_config.object.get("num_hidden_layers") orelse return error.InvalidGlmConfig;
    if (layer_value != .integer or layer_value.integer < 1) return error.InvalidGlmConfig;
    const layers: usize = @intCast(layer_value.integer);
    const raw = try dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(16 * 1024 * 1024));
    defer allocator.free(raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidGlmWeightIndex;
    const wm = parsed.value.object.get("weight_map") orelse return error.InvalidGlmWeightIndex;
    if (wm != .object) return error.InvalidGlmWeightIndex;
    var files = std.StringHashMap(void).init(allocator);
    defer files.deinit();
    var owners = wm.object.iterator();
    var expected: usize = 0;
    while (owners.next()) |entry| {
        if (!keepLoadKey(entry.key_ptr.*, layers, trunk_only, vision)) continue;
        const value = entry.value_ptr.*;
        if (value != .string or value.string.len == 0 or std.mem.indexOfAny(u8, value.string, "/\\") != null or std.mem.eql(u8, value.string, "..")) return error.InvalidGlmShardName;
        try files.put(value.string, {});
        expected += 1;
    }
    if (expected == 0) return error.MissingIndexedGlmWeight;
    var result = model.Weights.init(allocator);
    errdefer result.deinit();
    var reader = @import("expert_io.zig").ParallelReader.init(allocator, @import("expert_io.zig").ParallelReader.default_chunk, @import("expert_io.zig").ParallelReader.default_workers);
    defer reader.deinit();
    var file_it = files.keyIterator();
    while (file_it.next()) |file| {
        const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file.* }, 0);
        defer allocator.free(path);
        try loadStoredShard(allocator, &reader, path, file.*, wm.object, layers, trunk_only, vision, &result, max_bytes);
    }
    if (result.count() != expected) {
        @import("log.zig").err("[glm-loader] loaded {d}/{d} indexed tensors\n", .{ result.count(), expected });
        var missing = wm.object.iterator();
        var reported: usize = 0;
        while (missing.next()) |entry| {
            if (!keepLoadKey(entry.key_ptr.*, layers, trunk_only, vision) or result.get(entry.key_ptr.*) != null) continue;
            if (reported < 20) @import("log.zig").err("[glm-loader] missing {s} from {s}\n", .{ entry.key_ptr.*, entry.value_ptr.string });
            reported += 1;
        }
        return error.MissingIndexedGlmWeight;
    }
    if (storedBytes(&result) > max_bytes) return error.GlmResidentBudgetExceeded;
    // Filter ownership/vision/MTP before materializing any device allocation.
    var values = result.map.valueIterator();
    while (values.next()) |v| try mlx.check(mlx.mlx_array_eval(v.*));
    return result;
}

fn fixture(dir: std.Io.Dir, name: []const u8, header: []const u8, data: []const u8) !void {
    const a = std.testing.allocator;
    const bytes = try a.alloc(u8, 8 + header.len + data.len);
    defer a.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memcpy(bytes[8 + header.len ..], data);
    try dir.writeFile(std.testing.io, .{ .sub_path = name, .data = bytes });
}

fn tmpPath(tmp: anytype) ![]u8 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd = std.c.getcwd(&buf, buf.len) orelse return error.NoCwd;
    return std.fmt.allocPrint(std.testing.allocator, "{s}/.zig-cache/tmp/{s}", .{ std.mem.span(@as([*:0]const u8, @ptrCast(cwd))), tmp.sub_path });
}

test "GLM diagnostic loader preserves stored dtypes and strict index ownership" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const header = "{\"lm_head.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"model.language_model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[4,6]},\"model.language_model.layers.0.x.suh\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[6,8]},\"model.language_model.layers.0.x.trellis\":{\"dtype\":\"U16\",\"shape\":[1],\"data_offsets\":[8,10]},\"unindexed\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[10,14]}}";
    try fixture(tmp.dir, "owned.safetensors", header, &.{ 0, 0, 128, 63, 128, 63, 1, 60, 17, 0, 0, 0, 128, 63 });
    try fixture(tmp.dir, "old.safetensors", "{\"lm_head.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", &.{ 0, 0, 0, 64 });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "vision.safetensors", .data = "must not be opened" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"lm_head.weight\":\"owned.safetensors\",\"model.language_model.norm.weight\":\"owned.safetensors\",\"model.language_model.layers.0.x.suh\":\"owned.safetensors\",\"model.language_model.layers.0.x.trellis\":\"owned.safetensors\",\"model.visual.weight\":\"vision.safetensors\",\"model.language_model.mtp.weight\":\"vision.safetensors\",\"model.language_model.layers.1.mlp.gate.weight\":\"vision.safetensors\"}}" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"text_config\":{\"num_hidden_layers\":1}}" });
    const path = try tmpPath(tmp);
    defer a.free(path);
    var weights = try loadWeights(std.testing.io, a, path, mlx.gpuStream());
    defer weights.deinit();
    try std.testing.expectEqual(@as(u32, 4), weights.count());
    try std.testing.expectEqual(@as(u64, 10), storedBytes(&weights));
    try std.testing.expectEqual(storedBytes(&weights), try residentBytes(std.testing.io, a, path, 1));
    try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(weights.get("lm_head.weight").?));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.language_model.norm.weight").?));
    const half = weights.get("model.language_model.layers.0.x.suh").?;
    try std.testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(half));
    try mlx.check(mlx.mlx_array_eval(half));
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_astype(&wide, half, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(wide));
    try std.testing.expectEqual(@as(f32, 1.0009765625), mlx.mlx_array_data_float32(wide).?[0]);
    try std.testing.expectEqual(mlx.mlx_dtype.uint16, mlx.mlx_array_dtype(weights.get("model.language_model.layers.0.x.trellis").?));
}

test "GLM diagnostic loader refuses a missing indexed text tensor" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try fixture(tmp.dir, "a.safetensors", "{\"other\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", &.{ 0, 0, 0, 0 });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"lm_head.weight\":\"a.safetensors\"}}" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"text_config\":{\"num_hidden_layers\":1}}" });
    const path = try tmpPath(tmp);
    defer a.free(path);
    try std.testing.expectError(error.MissingIndexedGlmWeight, loadWeights(std.testing.io, a, path, mlx.gpuStream()));
}

test "GLM stream CPU trunk loader never opens expert shards and refuses a resident overbudget" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture(tmp.dir, "trunk.safetensors", "{\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}", &.{ 128, 63 });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"num_hidden_layers\":4}" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"lm_head.weight\":\"trunk.safetensors\",\"model.language_model.layers.3.mlp.experts.0.gate_proj.weight\":\"nonexistent.safetensors\",\"model.language_model.layers.4.mlp.gate.weight\":\"mtp-unopened.safetensors\"}}" });
    const path = try tmpPath(tmp);
    defer a.free(path);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try std.testing.expectError(error.GlmResidentBudgetExceeded, loadWeightsBounded(std.testing.io, a, path, cpu, true, 1));
    var weights = try loadWeightsBounded(std.testing.io, a, path, cpu, true, 2);
    defer weights.deinit();
    try std.testing.expectEqual(@as(u32, 1), weights.count());
    try std.testing.expectEqual(@as(u16, 0x3f80), mlx.mlx_array_data_bfloat16(weights.get("lm_head.weight").?).?[0]);
}

test "GLM streamed load and bill keep the text trunk and, only when asked, the tower: no routed experts or MTP layer" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const one = "{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}";
    try fixture(tmp.dir, "trunk.safetensors", "{\"lm_head.weight\":" ++ one ++ "}", &.{ 128, 63 });
    try fixture(tmp.dir, "experts.safetensors", "{\"model.language_model.layers.3.mlp.experts.0.gate_proj.weight\":" ++ one ++ ",\"model.language_model.layers.3.mlp.switch_mlp.gate_proj.trellis\":{\"dtype\":\"U16\",\"shape\":[1],\"data_offsets\":[2,4]}}", &.{ 128, 63, 1, 0 });
    try fixture(tmp.dir, "mtp.safetensors", "{\"model.language_model.layers.4.mlp.gate.weight\":" ++ one ++ "}", &.{ 128, 63 });
    try fixture(tmp.dir, "vision.safetensors", "{\"model.visual.patch_embed.weight\":" ++ one ++ "}", &.{ 128, 63 });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"num_hidden_layers\":4}" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"lm_head.weight\":\"trunk.safetensors\",\"model.language_model.layers.3.mlp.experts.0.gate_proj.weight\":\"experts.safetensors\",\"model.language_model.layers.3.mlp.switch_mlp.gate_proj.trellis\":\"experts.safetensors\",\"model.language_model.layers.4.mlp.gate.weight\":\"mtp.safetensors\",\"model.visual.patch_embed.weight\":\"vision.safetensors\"}}" });
    const path = try tmpPath(tmp);
    defer a.free(path);
    for ([_]@import("expert_quant.zig").Layout{ .bf16_individual, .exl3_k4 }) |layout| {
        const cfg = model.ModelConfig{ .model_type = "glm5_next", .num_hidden_layers = 4, .expert_streaming = true, .expert_layout = layout, .glm5_vision = true };
        try std.testing.expectEqual(model.ResidentSplit{ .trunk = 2, .mtp = 0, .vision = 2 }, try model.streamingResidentSplit(std.testing.io, a, path, &cfg));
    }
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .num_hidden_layers = 4, .expert_streaming = true, .expert_layout = .bf16_individual, .glm5_vision = true };
    for ([_]bool{ false, true }) |vision| {
        var weights = try model.loadWeightsForConfig(std.testing.io, a, path, &cfg, vision);
        defer weights.deinit();
        try std.testing.expectEqual(@as(u32, if (vision) 2 else 1), weights.count());
        try std.testing.expect(weights.get("lm_head.weight") != null);
        try std.testing.expectEqual(vision, weights.get("model.visual.patch_embed.weight") != null);
    }
}

test "GLM vision enabled payload bill and CPU loader retain exactly the same indexed tensors" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture(tmp.dir, "text.safetensors", "{\"lm_head.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", &.{ 0, 0, 128, 63 });
    try fixture(tmp.dir, "vision.safetensors", "{\"model.visual.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]},\"unindexed\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", &.{ 128, 63, 0, 64, 0, 0, 0, 64 });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"lm_head.weight\":\"text.safetensors\",\"model.visual.weight\":\"vision.safetensors\",\"model.language_model.mtp.weight\":\"missing.safetensors\",\"model.language_model.layers.1.x\":\"missing.safetensors\"}}" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"text_config\":{\"num_hidden_layers\":1}}" });
    const path = try tmpPath(tmp);
    defer a.free(path);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    for ([_]bool{ false, true }) |enabled| {
        var w = try loadWeightsWithVision(std.testing.io, a, path, cpu, enabled);
        defer w.deinit();
        try std.testing.expectEqual(enabled, w.get("model.visual.weight") != null);
        try std.testing.expect(w.get("unindexed") == null);
        try std.testing.expectEqual(@as(u64, if (enabled) 8 else 4), storedBytes(&w));
        try std.testing.expectEqual(storedBytes(&w), try residentBytesWithVision(std.testing.io, a, path, 1, enabled));
    }
}

test "GLM raw FP8 CPU loader preserves codes scales owners and exact resident bill" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const base = "model.language_model.layers.0.p";
    const header = "{\"" ++ base ++ ".weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[129,128],\"data_offsets\":[0,16512]},\"" ++ base ++ ".weight_scale_inv\":{\"dtype\":\"F32\",\"shape\":[2,1],\"data_offsets\":[16512,16520]},\"model.language_model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[16520,16522]},\"unindexed\":{\"dtype\":\"F8_E4M3\",\"shape\":[1],\"data_offsets\":[16522,16523]}}";
    const raw = try a.alloc(u8, 16523);
    defer a.free(raw);
    @memset(raw[0..16512], 0x38);
    @memcpy(raw[16512..], &[_]u8{ 0, 0, 128, 63, 0, 0, 0, 64, 128, 63, 0x7f });
    try fixture(tmp.dir, "raw.safetensors", header, raw);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"num_hidden_layers\":1}" });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"" ++ base ++ ".weight\":\"raw.safetensors\",\"" ++ base ++ ".weight_scale_inv\":\"raw.safetensors\",\"model.language_model.norm.weight\":\"raw.safetensors\"}}" });
    const path = try tmpPath(tmp);
    defer a.free(path);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var weights = try loadWeights(std.testing.io, a, path, cpu);
    defer weights.deinit();
    try std.testing.expectEqual(@as(u64, 16522), storedBytes(&weights));
    try std.testing.expectEqual(storedBytes(&weights), try residentBytes(std.testing.io, a, path, 1));
    try std.testing.expectEqual(@as(usize, 3), weights.count());
    const linear = try @import("glm5_model.zig").Linear.load(&weights, base, 128);
    try std.testing.expect(linear.isFp8());
    try std.testing.expectEqual(mlx.mlx_dtype.uint8, mlx.mlx_array_dtype(linear.w));
    try std.testing.expectEqualSlices(u8, raw[0..16512], mlx.mlx_array_data_uint8(linear.w).?[0..16512]);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, mlx.mlx_array_data_float32(linear.scales).?[0..2]);
    try std.testing.expectError(error.GlmResidentBudgetExceeded, loadWeightsBounded(std.testing.io, a, path, cpu, false, 16521));
    try std.testing.expectError(error.InvalidGlmLinear, @import("glm5_model.zig").Linear.load(&weights, base, 256));
    raw[0] = 0x7f;
    try fixture(tmp.dir, "raw.safetensors", header, raw);
    try std.testing.expectError(error.InvalidFp8Value, loadWeights(std.testing.io, a, path, cpu));
    raw[0] = 0x38;
    @memcpy(raw[16512..16516], &[_]u8{ 0, 0, 128, 127 });
    try fixture(tmp.dir, "raw.safetensors", header, raw);
    try std.testing.expectError(error.InvalidFp8Scale, loadWeights(std.testing.io, a, path, cpu));
}

test "GLM sharded loader succeeds with more shards than its descriptor limit" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(a);
    try index.appendSlice(a, "{\"weight_map\":{");
    for (0..300) |i| {
        const name = try std.fmt.allocPrint(a, "model.language_model.layers.0.p{d}.weight", .{i});
        defer a.free(name);
        const file = try std.fmt.allocPrint(a, "p{d}.safetensors", .{i});
        defer a.free(file);
        const header = try std.fmt.allocPrint(a, "{{\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}}}", .{name});
        defer a.free(header);
        try fixture(tmp.dir, file, header, &.{ 128, 63 });
        const entry = try std.fmt.allocPrint(a, "{s}\"{s}\":\"{s}\"", .{ if (i == 0) "" else ",", name, file });
        defer a.free(entry);
        try index.appendSlice(a, entry);
    }
    try index.appendSlice(a, "}}");
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"num_hidden_layers\":1}" });
    const path = try tmpPath(tmp);
    defer a.free(path);
    var prior: std.c.rlimit = undefined;
    if (std.c.getrlimit(.NOFILE, &prior) != 0) return error.SkipZigTest;
    const limited = std.c.rlimit{ .cur = @min(prior.cur, 128), .max = prior.max };
    if (std.c.setrlimit(.NOFILE, &limited) != 0) return error.SkipZigTest;
    defer _ = std.c.setrlimit(.NOFILE, &prior);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var weights = try loadWeights(std.testing.io, a, path, cpu);
    defer weights.deinit();
    try std.testing.expectEqual(@as(usize, 300), weights.count());
    try std.testing.expectEqual(@as(u64, 600), storedBytes(&weights));
    try std.testing.expectEqual(@as(u64, 600), try residentBytes(std.testing.io, a, path, 1));
    var values = weights.map.valueIterator();
    while (values.next()) |v| try std.testing.expectEqual(@as(u16, 0x3f80), mlx.mlx_array_data_bfloat16(v.*).?[0]);
}
