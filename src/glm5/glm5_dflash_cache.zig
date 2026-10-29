//! Local runtime A4/group64 cache. The shipped BF16 checkpoint is read-only.
const std = @import("std");
const mlx = @import("../mlx.zig");
const draft = @import("../dflash.zig");
const model = @import("../model.zig");
const log = @import("../log.zig");
const schema = "sushi-glm-dflash-runtime-a4g64-v1";
const cache_bits = 4;
const cache_group = 64;
var space_for_test: ?u64 = null;
extern "c" fn sushi_available_disk_bytes(path: [*:0]const u8, out: *u64) c_int;

pub fn isShippedSource(directory: []const u8, model_dir: []const u8) bool {
    return std.mem.eql(u8, std.fs.path.basename(directory), draft.SHIPPED_GLM_SUBDIR) and
        std.mem.eql(u8, std.fs.path.dirname(directory) orelse "", std.mem.trimEnd(u8, model_dir, "/"));
}

fn quantizeKey(name: []const u8, shape: []const c_int) bool {
    if (shape.len != 2 or shape[1] <= 0 or @rem(shape[1], cache_group) != 0) return false;
    if (std.mem.eql(u8, name, "fc.weight") or std.mem.eql(u8, name, "encoder.fc.weight")) return true;
    if (!std.mem.startsWith(u8, name, "layers.")) return false;
    for ([_][]const u8{ ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight", ".self_attn.o_proj.weight", ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight", ".attention_conv.kernel_projection.weight", ".mlp_conv.kernel_projection.weight" }) |suffix| {
        if (std.mem.endsWith(u8, name, suffix)) return true;
    }
    return false;
}

pub fn plannedResidentBytes(io: std.Io, a: std.mem.Allocator, source: []const u8) !u64 {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{source}, 0);
    defer a.free(path);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.DflashSourceMissing;
    defer _ = std.c.close(fd);
    var size: [8]u8 = undefined;
    try @import("../expert_io.zig").readExact(fd, &size, 0);
    const len = std.mem.readInt(u64, &size, .little);
    if (len == 0 or len > 16 * 1024 * 1024) return error.InvalidSafetensorsHeader;
    const raw = try a.alloc(u8, @intCast(len));
    defer a.free(raw);
    try @import("../expert_io.zig").readExact(fd, raw, 8);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSafetensorsHeader;
    var total: u64 = 0;
    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
        const meta = entry.value_ptr.*;
        if (meta != .object) return error.InvalidSafetensorsTensor;
        const dt = meta.object.get("dtype") orelse return error.InvalidSafetensorsTensor;
        const dims = meta.object.get("shape") orelse return error.InvalidSafetensorsTensor;
        if (dt != .string or !std.mem.eql(u8, dt.string, "BF16") or dims != .array or dims.array.items.len > 8) return error.DflashSourceMustBeBf16;
        var shape: [8]c_int = undefined;
        var count: u64 = 1;
        for (dims.array.items, 0..) |d, i| {
            if (d != .integer or d.integer <= 0 or d.integer > std.math.maxInt(c_int)) return error.InvalidSafetensorsTensor;
            shape[i] = @intCast(d.integer);
            count = try std.math.mul(u64, count, @intCast(d.integer));
        }
        const offsets = meta.object.get("data_offsets") orelse return error.InvalidSafetensorsTensor;
        if (offsets != .array or offsets.array.items.len != 2) return error.InvalidSafetensorsTensor;
        const lo = offsets.array.items[0];
        const hi = offsets.array.items[1];
        const original_bytes = try std.math.mul(u64, count, 2);
        if (lo != .integer or hi != .integer or lo.integer < 0 or hi.integer < lo.integer or @as(u64, @intCast(hi.integer - lo.integer)) != original_bytes) return error.InvalidSafetensorsTensor;
        const bytes = if (quantizeKey(entry.key_ptr.*, shape[0..dims.array.items.len])) count * cache_bits / 8 + count / cache_group * 4 else count * 2;
        total = try std.math.add(u64, total, bytes);
    }
    _ = io;
    return total;
}

fn sourceIdentity(io: std.Io, a: std.mem.Allocator, source: []const u8) ![64]u8 {
    var dir = try std.Io.Dir.openDirAbsolute(io, source, .{});
    defer dir.close(io);
    const raw = try dir.readFileAlloc(io, "config.json", a, .limited(2 * 1024 * 1024));
    defer a.free(raw);
    const st = try dir.statFile(io, "model.safetensors", .{});
    const stamp = try std.fmt.allocPrint(a, "{s}:{d}:{d}", .{ source, st.size, st.mtime.nanoseconds });
    defer a.free(stamp);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(raw);
    hash.update(stamp);
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

pub fn cacheValid(io: std.Io, a: std.mem.Allocator, model_dir: []const u8, directory: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, directory, .{}) catch return false;
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, "sushi-runtime-cache.json", a, .limited(64 * 1024)) catch |err| return err == error.FileNotFound; // Existing user-supplied assistants remain usable.
    defer a.free(raw);
    const p = std.json.parseFromSlice(std.json.Value, a, raw, .{}) catch return false;
    defer p.deinit();
    if (p.value != .object) return false;
    const format = p.value.object.get("schema") orelse return false;
    const saved = p.value.object.get("source_id") orelse return false;
    const bytes = p.value.object.get("file_size") orelse return false;
    const mtime = p.value.object.get("file_mtime") orelse return false;
    if (format != .string or !std.mem.eql(u8, format.string, schema) or saved != .string or bytes != .integer or mtime != .integer) return false;
    const source = std.fs.path.join(a, &.{ model_dir, draft.SHIPPED_GLM_SUBDIR }) catch return false;
    defer a.free(source);
    const stat = dir.statFile(io, "model.safetensors", .{}) catch return false;
    if (bytes.integer != stat.size or mtime.integer != stat.mtime.nanoseconds) return false;
    // A deleted source cannot be re-checked: the intact cache is the assistant.
    var source_dir = std.Io.Dir.openDirAbsolute(io, source, .{}) catch |err| return err == error.FileNotFound;
    source_dir.close(io);
    const identity = sourceIdentity(io, a, source) catch return false;
    return std.mem.eql(u8, saved.string, &identity);
}

pub const Prepared = struct { path: []u8, generated: bool = false, fallback: bool = false };
fn fallback(a: std.mem.Allocator, source: []const u8, reason: []const u8) !Prepared {
    log.warn("[glm-dflash] {s}; using shipped BF16 DFlash2 unchanged (full BF16 weights billed)\n", .{reason});
    return .{ .path = try a.dupe(u8, source), .fallback = true };
}

pub fn prepare(io: std.Io, a: std.mem.Allocator, source: []const u8, s: mlx.mlx_stream) !Prepared {
    const parent = std.fs.path.dirname(source) orelse return error.InvalidDflashSourcePath;
    const destination = try std.fs.path.join(a, &.{ parent, draft.DFLASH2_IN_DIR_SUBDIR });
    errdefer a.free(destination);
    if (draft.probeIsDflash(io, a, destination) and cacheValid(io, a, parent, destination)) return .{ .path = destination };
    const lock_path = try std.fmt.allocPrintSentinel(a, "{s}/.dflash2-cache.lock", .{parent}, 0);
    defer a.free(lock_path);
    const fd = std.c.open(lock_path, .{ .ACCMODE = .RDWR, .CREAT = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) {
        const result = try fallback(a, source, "cannot write the local cache");
        a.free(destination);
        return result;
    }
    defer _ = std.c.close(fd);
    if (std.c.flock(fd, std.c.LOCK.EX | std.c.LOCK.NB) != 0) {
        log.info("[glm-dflash] another Sushi process is preparing DFlash2; please wait\n", .{});
        if (std.c.flock(fd, std.c.LOCK.EX) != 0) return error.DflashCacheLockFailed;
    }
    defer _ = std.c.flock(fd, std.c.LOCK.UN);
    if (draft.probeIsDflash(io, a, destination) and cacheValid(io, a, parent, destination)) return .{ .path = destination };
    const parent_z = try a.dupeSentinel(u8, parent, 0);
    defer a.free(parent_z);
    var free: u64 = 0;
    const payload = try plannedResidentBytes(io, a, source);
    const space_ok = sushi_available_disk_bytes(parent_z, &free) == 0;
    if (comptime @import("builtin").is_test) if (space_for_test) |forced| {
        free = forced;
    };
    if (!space_ok or free < payload + 16 * 1024 * 1024) {
        const result = try fallback(a, source, "not enough disk space for the local A4/group64 cache");
        a.free(destination);
        return result;
    }
    const stage = try std.fmt.allocPrint(a, "{s}/.dflash2-building-{d}", .{ parent, std.c.getpid() });
    defer a.free(stage);
    var parent_dir = try std.Io.Dir.openDirAbsolute(io, parent, .{});
    defer parent_dir.close(io);
    parent_dir.deleteTree(io, std.fs.path.basename(stage)) catch {};
    parent_dir.createDirPath(io, std.fs.path.basename(stage)) catch {
        const result = try fallback(a, source, "cannot write the local cache");
        a.free(destination);
        return result;
    };
    defer parent_dir.deleteTree(io, std.fs.path.basename(stage)) catch {};
    log.info("Preparing GLM 5.3 Flash DFlash2 for Sushi ... please wait for a few minutes.\n", .{});
    log.info("[glm-dflash] quantizing the checkpoint to a local A4/group64 cache; shipped BF16 files stay unchanged\n", .{});
    const timer = @import("../io_util.zig").Stopwatch.init(io);
    build(io, a, source, stage, s, payload) catch |err| {
        if (err == error.DflashCacheStorageUnavailable or err == error.AccessDenied or err == error.ReadOnlyFileSystem or err == error.NoSpaceLeft or err == error.DiskQuota) {
            const result = try fallback(a, source, "cache storage unavailable");
            a.free(destination);
            return result;
        }
        return err;
    };
    // Only replace a cache this runtime owns; never delete a supplied assistant.
    if (parent_dir.statFile(io, draft.DFLASH2_IN_DIR_SUBDIR, .{})) |_| {
        var old = try parent_dir.openDir(io, draft.DFLASH2_IN_DIR_SUBDIR, .{});
        defer old.close(io);
        _ = old.statFile(io, "sushi-runtime-cache.json", .{}) catch return error.DflashCacheDestinationExists;
        const backup = try std.fmt.allocPrint(a, "{s}/.dflash2-old-{d}", .{ parent, std.c.getpid() });
        defer a.free(backup);
        parent_dir.deleteTree(io, std.fs.path.basename(backup)) catch {};
        try std.Io.Dir.renameAbsolute(destination, backup, io);
        std.Io.Dir.renameAbsolute(stage, destination, io) catch |err| {
            std.Io.Dir.renameAbsolute(backup, destination, io) catch {};
            return err;
        };
        try parent_dir.deleteTree(io, std.fs.path.basename(backup));
    } else |err| {
        if (err != error.FileNotFound) return err;
        try std.Io.Dir.renameAbsolute(stage, destination, io);
    }
    log.info("[glm-dflash] A4/group64 cache ready in {d:.2} seconds ({d:.3} GiB); future loads reuse {s}\n", .{ @as(f64, @floatFromInt(timer.read())) / 1e9, @as(f64, @floatFromInt(payload)) / (1 << 30), destination });
    return .{ .path = destination, .generated = true };
}

fn insert(a: std.mem.Allocator, map: mlx.mlx_map_string_to_array, name: []const u8, value: mlx.mlx_array) !void {
    const z = try a.dupeSentinel(u8, name, 0);
    defer a.free(z);
    try mlx.check(mlx.mlx_map_string_to_array_insert(map, z, value));
}

fn build(io: std.Io, a: std.mem.Allocator, source: []const u8, stage: []const u8, s: mlx.mlx_stream, payload: u64) !void {
    const src = try std.fmt.allocPrint(a, "{s}/model.safetensors", .{source});
    defer a.free(src);
    var weights = model.Weights.init(a);
    defer weights.deinit();
    const src_z = try a.dupeSentinel(u8, src, 0);
    defer a.free(src_z);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try model.loadSafetensorsFile(a, &weights, src_z, cpu, false);
    const map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(map);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    var it = weights.map.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const raw = entry.value_ptr.*;
        if (mlx.mlx_array_dtype(raw) != .bfloat16) return error.DflashSourceMustBeBf16;
        if (quantizeKey(name, mlx.getShape(raw))) {
            var q = try draft.quantizeDense(raw, cache_bits, cache_group, s);
            defer q.deinit();
            for ([_]mlx.mlx_array{ q.w, q.scales, q.biases }) |v| try mlx.check(mlx.mlx_array_eval(v));
            try insert(a, map, name, q.w);
            const base = name[0 .. name.len - ".weight".len];
            const scales = try std.fmt.allocPrint(a, "{s}.scales", .{base});
            defer a.free(scales);
            const biases = try std.fmt.allocPrint(a, "{s}.biases", .{base});
            defer a.free(biases);
            try insert(a, map, scales, q.scales);
            try insert(a, map, biases, q.biases);
        } else {
            try mlx.check(mlx.mlx_array_eval(raw));
            try insert(a, map, name, raw);
        }
    }
    const path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{stage}, 0);
    defer a.free(path);
    if (mlx.mlx_save_safetensors(path, map, meta) != 0) {
        var message: [512]u8 = undefined;
        const reason = mlx.takeError(&message) orelse "";
        if (std.c._errno().* == @backingInt(std.c.E.NOSPC) or std.c._errno().* == @backingInt(std.c.E.ACCES) or std.mem.indexOf(u8, reason, "space") != null or std.mem.indexOf(u8, reason, "Permission") != null) return error.DflashCacheStorageUnavailable;
        return error.MlxError;
    }
    var from = try std.Io.Dir.openDirAbsolute(io, source, .{});
    defer from.close(io);
    var into = try std.Io.Dir.openDirAbsolute(io, stage, .{});
    defer into.close(io);
    const raw_config = try from.readFileAlloc(io, "config.json", a, .limited(2 * 1024 * 1024));
    defer a.free(raw_config);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, raw_config, .{});
    defer parsed.deinit();
    const arena = parsed.arena.allocator();
    var quant: std.json.ObjectMap = .empty;
    try quant.put(arena, "bits", .{ .integer = cache_bits });
    try quant.put(arena, "group_size", .{ .integer = cache_group });
    try quant.put(arena, "mode", .{ .string = "affine" });
    try quant.put(arena, "candidate_selector.hidden_projection", .{ .bool = false });
    var root = parsed.value;
    try root.object.put(arena, "quantization", .{ .object = quant });
    const config = try std.json.Stringify.valueAlloc(a, root, .{ .whitespace = .indent_2 });
    defer a.free(config);
    try into.writeFile(io, .{ .sub_path = "config.json", .data = config });
    from.copyFile("README.md", into, "README.md", io, .{}) catch |err| if (err != error.FileNotFound) return err;
    from.copyFile("LICENSE", into, "LICENSE", io, .{}) catch |err| if (err != error.FileNotFound) return err;
    const id = try sourceIdentity(io, a, source);
    const st = try into.statFile(io, "model.safetensors", .{});
    // Header excluded; count the actual written tensor bytes before publishing.
    const written_payload = @import("glm5_diagnostic.zig").assistantResidentBytes(io, a, stage) catch return error.DflashCacheStorageUnavailable;
    if (written_payload != payload) return error.DflashCacheStorageUnavailable;
    const manifest = try std.json.Stringify.valueAlloc(a, .{ .schema = schema, .source_id = id[0..], .file_size = st.size, .file_mtime = st.mtime.nanoseconds, .payload_bytes = payload, .local_only = true, .do_not_redistribute = true, .license = "cc-by-nc-nd-4.0", .license_url = "https://creativecommons.org/licenses/by-nc-nd/4.0/", .source_repo = "incoai/GLM-5.3-Flash-DFlash2" }, .{ .whitespace = .indent_2 });
    defer a.free(manifest);
    try into.writeFile(io, .{ .sub_path = "sushi-runtime-cache.json", .data = manifest });
    try into.writeFile(io, .{ .sub_path = ".gitignore", .data = "*\n" });
    for ([_][]const u8{ "model.safetensors", "config.json", "sushi-runtime-cache.json" }) |name| {
        const file = try into.openFile(io, name, .{});
        defer file.close(io);
        if (std.c.fsync(file.handle) != 0) return error.DflashCacheStorageUnavailable;
    }
    if (std.c.fsync(into.handle) != 0) return error.DflashCacheStorageUnavailable;
}

fn tinySource(io: std.Io, a: std.mem.Allocator, parent: std.Io.Dir, path: []const u8, s: mlx.mlx_stream) ![]u8 {
    try parent.createDirPath(io, draft.SHIPPED_GLM_SUBDIR);
    var dir = try parent.openDir(io, draft.SHIPPED_GLM_SUBDIR, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"architectures\":[\"DFlash2DraftModel\"],\"hidden_size\":128,\"dflash_config\":{\"block_size\":8,\"mask_token_id\":31,\"target_layer_ids\":[1,3]}}" });
    try dir.writeFile(io, .{ .sub_path = "README.md", .data = "license: cc-by-nc-nd-4.0\nOriginal untouched checkpoint\n" });
    const source = try std.fs.path.join(a, &.{ path, draft.SHIPPED_GLM_SUBDIR });
    errdefer a.free(source);
    const file = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{source}, 0);
    defer a.free(file);
    const map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(map);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    for ([_][]const u8{ "fc.weight", "candidate_selector.hidden_projection.weight", "candidate_selector.predecessor_codebook" }, [_][2]c_int{ .{ 128, 256 }, .{ 16, 128 }, .{ 32, 16 } }) |name, shape| {
        var array = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(array);
        try mlx.check(mlx.mlx_ones(&array, &shape, 2, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(array));
        try insert(a, map, name, array);
    }
    try mlx.check(mlx.mlx_save_safetensors(file, map, meta));
    return source;
}

test "GLM runtime cache creates A4 group64 once and preserves the shipped BF16 source" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const parent = path_buf[0..n];
    const s = mlx.gpuStream();
    const source = try tinySource(io, a, tmp.dir, parent, s);
    defer a.free(source);
    var from = try std.Io.Dir.openDirAbsolute(io, source, .{});
    defer from.close(io);
    const original = try from.readFileAlloc(io, "model.safetensors", a, .limited(1 << 20));
    defer a.free(original);
    const planned = try plannedResidentBytes(io, a, source);
    try std.testing.expectEqual(@as(u64, 128 * 256 / 2 + 128 * 256 / 64 * 4 + 16 * 128 * 2 + 32 * 16 * 2), planned);
    const made = try prepare(io, a, source, s);
    defer a.free(made.path);
    try std.testing.expect(made.generated and !made.fallback);
    try std.testing.expect(cacheValid(io, a, parent, made.path));
    try std.testing.expectEqual(planned, try @import("glm5_diagnostic.zig").assistantResidentBytes(io, a, made.path));
    var weights = try model.loadWeights(io, a, made.path);
    defer weights.deinit();
    try std.testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(weights.get("fc.weight").?));
    try std.testing.expectEqualSlices(c_int, &.{ 128, 4 }, mlx.getShape(weights.get("fc.scales").?));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("candidate_selector.hidden_projection.weight").?));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("candidate_selector.predecessor_codebook").?));
    const again = try prepare(io, a, source, s);
    defer a.free(again.path);
    try std.testing.expect(!again.generated and !again.fallback);
    const after = try from.readFileAlloc(io, "model.safetensors", a, .limited(1 << 20));
    defer a.free(after);
    try std.testing.expectEqualSlices(u8, original, after);
    try from.writeFile(io, .{ .sub_path = "config.json", .data = "{\"architectures\":[\"DFlash2DraftModel\"],\"hidden_size\":128,\"changed\":true,\"dflash_config\":{\"block_size\":8,\"mask_token_id\":31,\"target_layer_ids\":[1,3]}}" });
    try std.testing.expect(!cacheValid(io, a, parent, made.path));
    const refreshed = try prepare(io, a, source, s);
    defer a.free(refreshed.path);
    try std.testing.expect(refreshed.generated);
}

test "GLM runtime cache regenerates a cache written under the earlier A6 policy" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const parent = path_buf[0..n];
    const s = mlx.gpuStream();
    const source = try tinySource(io, a, tmp.dir, parent, s);
    defer a.free(source);
    const made = try prepare(io, a, source, s);
    defer a.free(made.path);
    var cache = try std.Io.Dir.openDirAbsolute(io, made.path, .{});
    defer cache.close(io);
    const manifest = try cache.readFileAlloc(io, "sushi-runtime-cache.json", a, .limited(64 * 1024));
    defer a.free(manifest);
    const old = try std.mem.replaceOwned(u8, a, manifest, schema, "sushi-glm-dflash-runtime-a6g128-v1");
    defer a.free(old);
    try cache.writeFile(io, .{ .sub_path = "sushi-runtime-cache.json", .data = old });
    try std.testing.expect(!cacheValid(io, a, parent, made.path));
    const again = try prepare(io, a, source, s);
    defer a.free(again.path);
    try std.testing.expect(again.generated);
}

test "GLM runtime cache uses unchanged BF16 when the cache parent is read only" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const parent = path_buf[0..n];
    const source = try tinySource(io, a, tmp.dir, parent, mlx.gpuStream());
    defer a.free(source);
    const path_z = try a.dupeSentinel(u8, parent, 0);
    defer a.free(path_z);
    try std.testing.expectEqual(@as(c_int, 0), std.c.chmod(path_z, 0o500));
    defer _ = std.c.chmod(path_z, 0o700);
    const result = try prepare(io, a, source, mlx.gpuStream());
    defer a.free(result.path);
    try std.testing.expect(result.fallback and !result.generated);
    try std.testing.expectEqualStrings(source, result.path);
}

test "GLM runtime cache uses unchanged BF16 when disk space cannot fit the cache" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const source = try tinySource(io, a, tmp.dir, path_buf[0..n], mlx.gpuStream());
    defer a.free(source);
    space_for_test = 0;
    defer space_for_test = null;
    const result = try prepare(io, a, source, mlx.gpuStream());
    defer a.free(result.path);
    try std.testing.expect(result.fallback and !result.generated);
    try std.testing.expectEqualStrings(source, result.path);
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "dflash2", .{}));
}

test "GLM runtime cache stays valid after the source folder is deleted" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const parent = path_buf[0..n];
    const s = mlx.gpuStream();
    const source = try tinySource(io, a, tmp.dir, parent, s);
    defer a.free(source);
    const made = try prepare(io, a, source, s);
    defer a.free(made.path);
    try tmp.dir.deleteTree(io, draft.SHIPPED_GLM_SUBDIR);
    try std.testing.expect(cacheValid(io, a, parent, made.path));
    const found = draft.resolveInDirDrafter(io, a, parent) orelse return error.TestExpectedInDirDrafter;
    defer a.free(found);
    try std.testing.expectEqualStrings(made.path, found);
}

test "GLM runtime cache is invalid when a present source changed" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const parent = path_buf[0..n];
    const s = mlx.gpuStream();
    const source = try tinySource(io, a, tmp.dir, parent, s);
    defer a.free(source);
    const made = try prepare(io, a, source, s);
    defer a.free(made.path);
    var from = try std.Io.Dir.openDirAbsolute(io, source, .{});
    defer from.close(io);
    try from.writeFile(io, .{ .sub_path = "config.json", .data = "{\"architectures\":[\"DFlash2DraftModel\"],\"hidden_size\":128,\"changed\":true,\"dflash_config\":{\"block_size\":8,\"mask_token_id\":31,\"target_layer_ids\":[1,3]}}" });
    try std.testing.expect(!cacheValid(io, a, parent, made.path));
}

test "GLM runtime cache with damaged weights is invalid even when the source is gone" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    const parent = path_buf[0..n];
    const s = mlx.gpuStream();
    const source = try tinySource(io, a, tmp.dir, parent, s);
    defer a.free(source);
    const made = try prepare(io, a, source, s);
    defer a.free(made.path);
    try tmp.dir.deleteTree(io, draft.SHIPPED_GLM_SUBDIR);
    var cache = try std.Io.Dir.openDirAbsolute(io, made.path, .{});
    defer cache.close(io);
    try cache.writeFile(io, .{ .sub_path = "model.safetensors", .data = "truncated" });
    try std.testing.expect(!cacheValid(io, a, parent, made.path));
}
