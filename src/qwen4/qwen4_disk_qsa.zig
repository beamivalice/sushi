const kd = @import("../kv_disk_cache.zig");
const disk_placement = @import("../disk_placement.zig");
const write_stats = @import("../disk_write_stats.zig");
const log = @import("../log.zig");
const std = @import("std");
const mlx = @import("../mlx.zig");
const kv_quant = @import("../kv_quant.zig");
const transformer_mod = @import("../transformer.zig");
const model = @import("../model.zig");
const KVCache = transformer_mod.KVCache;
const testing = std.testing;
const SSMCacheEntry = transformer_mod.SSMCacheEntry;
const DiskTier = kd.DiskTier;
const ChunkDonor = DiskTier.ChunkDonor;
const MetaPair = DiskTier.MetaPair;
const NamedTensor = DiskTier.NamedTensor;
const QsaHistoryResult = DiskTier.QsaHistoryResult;
const IndexEntry = kd.IndexEntry;
const PersistOutcome = kd.PersistOutcome;
const buildHybridEntries = kd.buildHybridEntries;
const cacheValueAt = kd.cacheValueAt;
const chunkShareTokens = kd.chunkShareTokens;
const fileSize = kd.fileSize;
const fillCache = kd.fillCache;
const freeHybridEntries = kd.freeHybridEntries;
const makeArange = kd.makeArange;
const readFileAlloc = kd.readFileAlloc;
const ssmArrVal = kd.ssmArrVal;
const statFile = kd.statFile;
const tmpRoot = kd.tmpRoot;


/// A test tensor of `shape` filled with `v` (f32). Owned by the caller.
fn filledArray(shape: []const c_int, v: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const scalar = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(scalar);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_full(&out, shape.ptr, shape.len, scalar, .float32, s));
    return out;
}

fn stAuxPooledBytes(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !struct { aux: u64, pooled: u64 } {
    const raw = readFileAlloc(allocator, io, path, 4 * 1024 * 1024) orelse return error.TestUnexpectedResult;
    defer allocator.free(raw);
    if (raw.len < 8) return error.TestUnexpectedResult;
    const hdr_len = std.mem.readInt(u64, raw[0..8], .little);
    if (8 + hdr_len > raw.len) return error.TestUnexpectedResult;
    const hdr = raw[8 .. 8 + hdr_len];
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, hdr, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.TestUnexpectedResult;
    var aux: u64 = 0;
    var pooled: u64 = 0;
    var it = parsed.value.object.iterator();
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
        if (e.value_ptr.* != .object) continue;
        const o = e.value_ptr.*.object;
        const dtype = if (o.get("dtype")) |d| d.string else continue;
        const shape_v = o.get("shape") orelse continue;
        if (shape_v != .array) continue;
        var el: u64 = 1;
        for (shape_v.array.items) |x| {
            if (x != .integer) continue;
            el *= @intCast(x.integer);
        }
        const b: u64 = el * switch (dtype[0]) {
            'F' => if (dtype.len > 1 and dtype[1] == '3') @as(u64, 4) else 2,
            'B' => 2,
            'U', 'I' => if (dtype.len > 1 and dtype[1] == '8') @as(u64, 1) else 4,
            else => 1,
        };
        if (std.mem.indexOf(u8, e.key_ptr.*, ".aux") != null) aux += b;
        if (std.mem.indexOf(u8, e.key_ptr.*, ".pooled") != null) pooled += b;
    }
    return .{ .aux = aux, .pooled = pooled };
}

fn testFingerprintPayloadChange(indexed: bool, mtime_only: bool) !void {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    const files: []const []const u8 = if (indexed) &.{ "trunk.safetensors", "experts.safetensors", "ngram_table.bin" } else &.{ "model.safetensors", "ngram_table.bin" };
    if (indexed) try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"a\":\"trunk.safetensors\",\"b\":\"experts.safetensors\",\"c\":\"trunk.safetensors\"}}" });
    for (files) |name| try tmp.dir.writeFile(io, .{ .sub_path = name, .data = "old" });
    for (files) |name| {
        const before = try disk_placement.modelFingerprint(testing.allocator, io, base);
        defer testing.allocator.free(before);
        const unchanged = try disk_placement.modelFingerprint(testing.allocator, io, base);
        defer testing.allocator.free(unchanged);
        try testing.expectEqualStrings(before, unchanged);
        const f = try tmp.dir.openFile(io, name, .{ .mode = .read_write });
        defer f.close(io);
        const st = try f.stat(io);
        if (mtime_only) {
            try f.setTimestamps(io, .{ .modify_timestamp = .{ .new = .{ .nanoseconds = st.mtime.nanoseconds + 2_000_000_000 } } });
        } else {
            try tmp.dir.writeFile(io, .{ .sub_path = name, .data = "replacement" });
            try f.setTimestamps(io, .{ .modify_timestamp = .{ .new = st.mtime } });
        }
        const changed = try disk_placement.modelFingerprint(testing.allocator, io, base);
        defer testing.allocator.free(changed);
        try testing.expect(!std.mem.eql(u8, before, changed));
    }
}

pub fn qsaHistoryRowsOf(cp: *const transformer_mod.SSMCheckpoint) c_int {
    var r: c_int = 0;
    for (cp.layers) |*l| {
        if (l.qsa_rows > r) r = l.qsa_rows;
        if (!transformer_mod.snapshotHasQsaHistory(l)) continue;
        if (l.aux_state.ctx != null) {
            const sh = mlx.getShape(l.aux_state);
            if (sh.len >= 2 and sh[1] > r) r = sh[1];
        }
    }
    return r;
}

pub fn newestQsaCheckpoint(cps: []const transformer_mod.SSMCheckpoint) ?*const transformer_mod.SSMCheckpoint {
    var i = cps.len;
    while (i > 0) {
        i -= 1;
        if (transformer_mod.checkpointHasQsaPooled(&cps[i])) return &cps[i];
    }
    return null;
}

pub fn writeQsaHistoryFile(
    self: *DiskTier,
    dir_rel: []const u8,
    cp: *const transformer_mod.SSMCheckpoint,
    s: mlx.mlx_stream,
) !u64 {
    var list = std.ArrayList(NamedTensor).empty;
    defer self.freeNamed(&list);
    var meta = std.ArrayList(MetaPair).empty;
    defer meta.deinit(self.allocator);
    var ratio_buf: [16]u8 = undefined;
    var rows_buf: [16]u8 = undefined;
    var ratio_written = false;
    const rows = DiskTier.qsaHistoryRowsOf(cp);
    if (rows > 0) {
        const rs = try std.fmt.bufPrint(&rows_buf, "{d}\x00", .{rows});
        try meta.append(self.allocator, .{ .key = "qsa_rows", .value = rs[0 .. rs.len - 1] });
    }
    for (cp.layers, 0..) |l, li| {
        const names = .{ "aux", "pooled" };
        const arrs = .{ l.aux_state, l.qsa_pooled };
        inline for (names, arrs) |name, arr| {
            if (arr.ctx != null) {
                const key = try std.fmt.allocPrint(self.allocator, "l{d}." ++ name, .{li});
                errdefer self.allocator.free(key);
                var owned = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(owned);
                try mlx.check(mlx.mlx_array_set(&owned, arr));
                try list.append(self.allocator, .{ .key = key, .arr = owned });
            }
        }
        if (!ratio_written and (l.aux_state.ctx != null or l.qsa_pooled.ctx != null)) {
            const rs = try std.fmt.bufPrint(&ratio_buf, "{d}\x00", .{l.qsa_ratio});
            try meta.append(self.allocator, .{ .key = "qsa_ratio", .value = rs[0 .. rs.len - 1] });
            ratio_written = true;
        }
    }
    if (list.items.len == 0) return 0;
    const path = try std.fmt.allocPrint(self.allocator, "{s}/qsa.safetensors", .{dir_rel});
    return self.saveNamed(path, list.items, meta.items, .qsa, s);
}

pub fn loadQsaHistoryFile(self: *DiskTier, id: u64, n_layers: usize) !transformer_mod.SSMCheckpoint {
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/qsa.safetensors\x00", .{ self.root, id });
    defer self.allocator.free(path);
    if (fileSize(self.io, path[0 .. path.len - 1]) == null) return error.DiskCacheNoCheckpoint;
    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, @ptrCast(path.ptr), cpu));

    const layers = try self.allocator.alloc(transformer_mod.SSMCacheEntrySnapshot, n_layers);
    for (layers) |*l| l.* = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
    };
    var cp: transformer_mod.SSMCheckpoint = .{ .pos = 0, .layers = layers };
    errdefer cp.deinit(self.allocator);
    for (layers, 0..) |*l, li| {
        const akey = try std.fmt.allocPrint(self.allocator, "l{d}.aux\x00", .{li});
        defer self.allocator.free(akey);
        var aux = mlx.mlx_array_new();
        if (mlx.mlx_map_string_to_array_get(&aux, tensor_map, @ptrCast(akey.ptr)) == 0) {
            l.aux_state = aux;
        } else {
            _ = mlx.mlx_array_free(aux);
        }
        const pkey = try std.fmt.allocPrint(self.allocator, "l{d}.pooled\x00", .{li});
        defer self.allocator.free(pkey);
        var pooled = mlx.mlx_array_new();
        if (mlx.mlx_map_string_to_array_get(&pooled, tensor_map, @ptrCast(pkey.ptr)) == 0) {
            l.qsa_pooled = pooled;
        } else {
            _ = mlx.mlx_array_free(pooled);
        }
    }
    var ratio_c: [*:0]const u8 = undefined;
    if (mlx.mlx_map_string_to_string_get(&ratio_c, meta_map, "qsa_ratio") == 0) {
        const ratio = std.fmt.parseInt(c_int, std.mem.span(ratio_c), 10) catch return error.DiskCacheCorruptSsm;
        for (layers) |*l| l.qsa_ratio = ratio;
    }
    var rows_c: [*:0]const u8 = undefined;
    if (mlx.mlx_map_string_to_string_get(&rows_c, meta_map, "qsa_rows") == 0) {
        const rows = std.fmt.parseInt(c_int, std.mem.span(rows_c), 10) catch return error.DiskCacheCorruptSsm;
        for (layers) |*l| l.qsa_rows = rows;
        cp.pos = @intCast(rows);
    } else {
        cp.pos = @intCast(DiskTier.qsaHistoryRowsOf(&cp));
    }
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var count: usize = 0;
        for (layers) |*l| {
            inline for (.{ l.aux_state, l.qsa_pooled }) |arr| {
                if (arr.ctx != null) {
                    _ = mlx.mlx_vector_array_append_value(vec, arr);
                    count += 1;
                }
            }
        }
        if (count > 0) try mlx.check(mlx.mlx_eval(vec));
    }
    if (cp.pos == 0) cp.pos = @intCast(DiskTier.qsaHistoryRowsOf(&cp));
    // A v7 file holds the WHOLE raw history, and a mid-block checkpoint's leftover is
    // interior to it: the ring truncation belongs to the live entry the restore seeds,
    // not to the file it slices that ring out of.
    for (layers) |*l| {
        if (l.qsa_rows == 0) l.qsa_rows = @intCast(cp.pos);
    }
    if (!transformer_mod.checkpointHasQsaHistory(&cp)) return error.DiskCacheNoCheckpoint;
    return cp;
}

pub fn persistQsaHistory(
    self: *DiskTier,
    dir_rel: []const u8,
    cps_opt: ?[]const transformer_mod.SSMCheckpoint,
    inherited: bool,
    inherited_rows: u32,
    held_bytes: u64,
    prefix_rows: u32,
    s: mlx.mlx_stream,
) !QsaHistoryResult {
    // No new checkpoint leaves the owned QSA file, and its bill, intact.
    const kept: QsaHistoryResult = .{ .inherited = inherited, .rows = inherited_rows, .bytes = if (inherited) 0 else held_bytes };
    const cps = cps_opt orelse return kept;
    const src = DiskTier.newestQsaCheckpoint(cps) orelse return kept;
    const rows: u32 = @intCast(DiskTier.qsaHistoryRowsOf(src));
    // An owned bank already covering the newcomer stays: retained checkpoints may need its extra rows.
    if (!inherited and held_bytes > 0 and rows <= inherited_rows) return kept;
    if (inherited and rows > 0 and rows <= inherited_rows and rows <= prefix_rows) {
        return .{ .inherited = true, .rows = @min(inherited_rows, prefix_rows), .bytes = 0 };
    }
    if (inherited) {
        const qpath = try std.fmt.allocPrint(self.allocator, "{s}/qsa.safetensors", .{dir_rel});
        defer self.allocator.free(qpath);
        if (self.writer) |w| w.fence(qpath);
        std.Io.Dir.deleteFileAbsolute(self.io, qpath) catch {};
    }
    const bytes = try self.writeQsaHistoryFile(dir_rel, src, s);
    return .{ .bytes = bytes, .rows = rows, .inherited = false };
}

pub fn linkInheritedQsa(self: *DiskTier, d: ChunkDonor, id: u64) bool {
    const donor = &self.entries.items[d.idx];
    if (donor.qsa_history_bytes == 0 or donor.qsa_history_rows == 0) return false;
    const old_abs = std.fmt.allocPrint(self.allocator, "{s}/e{d}/qsa.safetensors", .{ self.root, d.id }) catch return false;
    defer self.allocator.free(old_abs);
    if (!self.chunkLanded(old_abs, donor.qsa_history_bytes)) return false;
    var root_dir = std.Io.Dir.openDirAbsolute(self.io, self.root, .{}) catch return false;
    defer root_dir.close(self.io);
    const old_sub = std.fmt.allocPrint(self.allocator, "e{d}/qsa.safetensors", .{d.id}) catch return false;
    defer self.allocator.free(old_sub);
    const new_sub = std.fmt.allocPrint(self.allocator, "e{d}/qsa.safetensors", .{id}) catch return false;
    defer self.allocator.free(new_sub);
    std.Io.Dir.hardLink(root_dir, old_sub, root_dir, new_sub, self.io, .{}) catch |err| {
        log.warn("  [disk-cache] chunk share: link e{d}/qsa -> e{d} failed: {s} — writing the history instead\n", .{ d.id, id, @errorName(err) });
        return false;
    };
    log.info("  [disk-cache] chunk share: e{d} inherits qsa history from e{d} by hard link\n", .{ id, d.id });
    return true;
}

test "DiskTier: hybrid entry round-trips SSM checkpoints (Phase 3)" {
    // GatedDeltaNet + a conv-only layer (null ssm_state) + plain
    // attention (uninitialized) in one entry: the null-state layer shapes
    // (same SSMCacheEntrySnapshot contract).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-hybrid", 0, 128);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32); // >= MIN_PERSIST_TOKENS
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Two checkpoints at 128 / 256 with distinguishable state (base 100/500
    // vs 200/600), captured through the real production capture path.
    var src128 = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src128);
    var src256 = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src256);
    // Layer 2 at 256 also carries qwen4_exp aux state: a QSA key history +
    // pooled block keys, and layer 1 the PLE token history.
    const aux_shape = [_]c_int{ 1, 256, 4 };
    const pooled_shape = [_]c_int{ 1, 64, 4 };
    src256[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src256[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src256[2].qsa_ratio = 4;
    src256[1].ple_prev = .{ 42, 43, 0, 0, 0, 0, 0, 0 };
    src256[1].ple_prev_valid = true;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src256, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src256, s);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());

    // Fresh tier (restart): both checkpoint positions survive the scan.
    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-hybrid", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    try testing.expectEqual(@as(?u32, 256), tier2.highestSsmPosAtOrBelow(0, 300));
    try testing.expectEqual(@as(?u32, 128), tier2.highestSsmPosAtOrBelow(0, 200));
    try testing.expectEqual(@as(?u32, null), tier2.highestSsmPosAtOrBelow(0, 100));

    // Restore at 256 into a fresh KVCache + ssm_entries.
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    const restored = try tier2.restoreIntoHybrid(&cache2, &dst, 0, 256, s);
    try testing.expectEqual(@as(u32, 256), restored);
    try testing.expectEqual(@as(usize, 256), cache2.step);

    // Layer 0: conv (base 200) + ssm (base 600) — a KEY SWAP would flip these.
    try testing.expect(dst[0].initialized);
    try testing.expectEqual(@as(f32, 200.0), ssmArrVal(dst[0].conv_state, 0, s));
    try testing.expectEqual(@as(f32, 200.0 + 23.0), ssmArrVal(dst[0].conv_state, 23, s));
    try testing.expectEqual(@as(f32, 600.0), ssmArrVal(dst[0].ssm_state, 0, s));
    try testing.expectEqual(@as(f32, 600.0 + 31.0), ssmArrVal(dst[0].ssm_state, 31, s));
    // Layer 1: conv only — conv present (base 10200), ssm stays null.
    try testing.expect(dst[1].initialized);
    try testing.expectEqual(@as(f32, 10_200.0), ssmArrVal(dst[1].conv_state, 0, s));
    try testing.expect(dst[1].ssm_state.ctx == null);
    // Layer 2: uninitialized plain-attention layer — both null, but the
    // qwen4_exp aux state round-trips (key history 700.., pooled 800..).
    try testing.expect(!dst[2].initialized);
    try testing.expect(dst[2].conv_state.ctx == null);
    try testing.expect(dst[2].ssm_state.ctx == null);
    try testing.expect(dst[2].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 256), dst[2].qsa_hist_rows);
    try testing.expectEqual(@as(f32, 800.0 + 11.0), ssmArrVal(dst[2].qsa_pooled, 11, s));
    try testing.expectEqual(@as(c_int, 4), dst[2].qsa_ratio);
    try testing.expect(dst[1].ple_prev_valid and dst[1].ple_prev[0] == 42 and dst[1].ple_prev[1] == 43);
    try testing.expect(!dst[0].ple_prev_valid and dst[0].aux_state.ctx == null);

    // KV rewound to 256 in lockstep, values byte-exact against the original.
    for (cache2.entries) |*ce| {
        try testing.expect(ce.initialized);
        try testing.expectEqual(@as(usize, 256), ce.offset);
    }
    try testing.expectEqual(
        try cacheValueAt(&cache, 1, 200, 3, s),
        try cacheValueAt(&cache2, 1, 200, 3, s),
    );

    // Restore at the lower checkpoint installs THAT position's state.
    var cache3 = try KVCache.init(testing.allocator, 3);
    defer cache3.deinit();
    var dst2: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst2);
    const restored128 = try tier2.restoreIntoHybrid(&cache3, &dst2, 0, 128, s);
    try testing.expectEqual(@as(u32, 128), restored128);
    try testing.expectEqual(@as(usize, 128), cache3.step);
    try testing.expectEqual(@as(f32, 100.0), ssmArrVal(dst2[0].conv_state, 0, s));
    try testing.expectEqual(@as(f32, 500.0), ssmArrVal(dst2[0].ssm_state, 0, s));

    // A position that was never checkpointed is rejected, not silently served.
    try testing.expectError(error.DiskCacheNoCheckpoint, tier2.restoreIntoHybrid(&cache3, &dst2, 0, 200, s));
}

test "DiskTier: QSA history bytes are O(rows), not O(checkpoints x rows)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-once", 0, 128);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 128, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    const s128 = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/s0000128.safetensors", .{ tier.root, tier.entries.items[0].id });
    defer testing.allocator.free(s128);
    const s256 = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/s0000256.safetensors", .{ tier.root, tier.entries.items[0].id });
    defer testing.allocator.free(s256);
    const qsa_path = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/qsa.safetensors", .{ tier.root, tier.entries.items[0].id });
    defer testing.allocator.free(qsa_path);
    const a128 = try stAuxPooledBytes(testing.allocator, io, s128);
    const a256 = try stAuxPooledBytes(testing.allocator, io, s256);
    try testing.expectEqual(@as(u64, 0), a128.aux + a128.pooled);
    try testing.expectEqual(@as(u64, 0), a256.aux + a256.pooled);
    const hq = try stAuxPooledBytes(testing.allocator, io, qsa_path);
    const one: u64 = 64 * 8 * 4;
    try testing.expectEqual(one, hq.aux + hq.pooled);

    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-qsa-once", 0, 128);
    defer tier2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 128), try tier2.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
    try testing.expectEqual(@as(c_int, 128), dst[2].qsa_hist_rows);
    try testing.expectEqual(@as(c_int, 32), mlx.getShape(dst[2].qsa_pooled)[1]);
}

test "DiskTier: a shorter pooled bank never replaces the longer one an entry's checkpoints need" {
    // An image turn persists only the text before its media: a short prefix of an existing entry.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-keep", 0, 128);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 2048, 8, 0.0, .float32);
    var tokens: [2048]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const aux_shape = [_]c_int{ 1, 256, 8 };
    var long = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&long);
    long[2].aux_state = makeArange(s, &aux_shape, 700.0);
    long[2].qsa_pooled = makeArange(s, &[_]c_int{ 1, 500, 8 }, 800.0);
    long[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &long, 1024, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &long, 2000, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &long, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
    try testing.expectEqual(@as(u32, 2000), tier.entries.items[0].qsa_history_rows);

    // The image turn: 600 text tokens, a checkpoint at 512, a bank of 512 rows.
    var short = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&short);
    short[2].aux_state = makeArange(s, &aux_shape, 700.0);
    short[2].qsa_pooled = makeArange(s, &[_]c_int{ 1, 128, 8 }, 800.0);
    short[2].qsa_ratio = 4;
    var cps2 = [_]transformer_mod.SSMCheckpoint{try transformer_mod.captureSsmCheckpoint(testing.allocator, &short, 512, s)};
    defer for (&cps2) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps2, &short, s);
    _ = try tier.appendCommit(cache.entries, 600, cache.config, tokens[0..600], false, &cps2, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(usize, 3), tier.entries.items[0].ssm_positions.len);
    try testing.expectEqual(@as(u32, 2000), tier.entries.items[0].qsa_history_rows);

    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-qsa-keep", 0, 128);
    defer tier2.deinit();
    for ([_]u32{ 1024, 2000 }) |pos| {
        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectEqual(pos, try tier2.restoreIntoHybrid(&cache2, &dst, 0, pos, s));
        try testing.expectEqual(@as(c_int, @intCast(pos)), dst[2].qsa_hist_rows);
        try testing.expectEqual(@as(f32, 800.0), ssmArrVal(dst[2].qsa_pooled, 0, s));
    }
}

test "DiskTier: a mid-block checkpoint restore overlays the pooled bank onto its own leftover" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-mid", 0, 128);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    // 46 is not a multiple of the ratio, so this checkpoint carries a 2-row leftover — the
    // shape `SSM_SNAPSHOT_BACKOFF` puts the always-on snapshot at.
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 46, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-qsa-mid", 0, 128);
    defer tier2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 46), try tier2.restoreIntoHybrid(&cache2, &dst, 0, 46, s));
    try testing.expect(transformer_mod.entriesHaveQsaHistory(&dst));
    try testing.expectEqual(@as(c_int, 46), dst[2].qsa_hist_rows);
    // Pooled blocks 0..46/4 come from the history file; the leftover is the checkpoint's own.
    try testing.expect(dst[2].qsa_pooled.ctx != null);
    try testing.expectEqual(@as(c_int, 11), mlx.getShape(dst[2].qsa_pooled)[1]);
    try testing.expectEqual(@as(f32, 800.0), ssmArrVal(dst[2].qsa_pooled, 0, s));
    try testing.expectEqual(@as(f32, 880.0), ssmArrVal(dst[2].qsa_pooled, 80, s));
    try testing.expectEqual(@as(c_int, 2), mlx.getShape(dst[2].aux_state)[1]);
    try testing.expectEqual(@as(f32, 1052.0), ssmArrVal(dst[2].aux_state, 0, s));
}

test "DiskTier: a GDN-layer aux window is persisted in the checkpoint file" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ple-aux", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var src = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src);
    const ple_shape = [_]c_int{ 1, 3, 8 };
    src[0].aux_state = makeArange(s, &ple_shape, 900.0);
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 128, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try testing.expect(cps[0].layers[0].aux_state.ctx != null);
    try testing.expect(!transformer_mod.checkpointHasQsaHistory(&cps[0]));

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
    const sp = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/s0000128.safetensors", .{ tier.root, tier.entries.items[0].id });
    defer testing.allocator.free(sp);
    const fam = try stAuxPooledBytes(testing.allocator, io, sp);
    try testing.expectEqual(@as(u64, 3 * 8 * 4), fam.aux);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 128), try tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
    try testing.expectEqual(@as(f32, 900.0), ssmArrVal(dst[0].aux_state, 0, s));
    try testing.expectEqual(@as(f32, 900.0 + 23.0), ssmArrVal(dst[0].aux_state, 23, s));
}

test "DiskTier: a pre-v7 interior restore overlays QSA history from the latest s* file" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-prev7", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var src128 = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src128);
    var src256 = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src256);
    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    src256[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src256[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src256[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src256, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src256, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    const id = tier.entries.items[0].id;
    var latest = try tier.loadSsmFile(id, 256, 3);
    defer latest.deinit(testing.allocator);
    var qsa = try tier.loadQsaHistoryFile(id, 3);
    defer qsa.deinit(testing.allocator);
    for (latest.layers, qsa.layers) |*l, q| {
        if (q.aux_state.ctx != null) {
            if (l.aux_state.ctx != null) _ = mlx.mlx_array_free(l.aux_state);
            l.aux_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&l.aux_state, q.aux_state));
        }
        if (q.qsa_pooled.ctx != null) {
            if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
            l.qsa_pooled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&l.qsa_pooled, q.qsa_pooled));
        }
        l.qsa_ratio = q.qsa_ratio;
    }
    const dir_rel = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}", .{ tier.root, id });
    defer testing.allocator.free(dir_rel);
    var interior = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s);
    defer interior.deinit(testing.allocator);
    _ = try tier.writeSsmFile(dir_rel, &interior, s);
    disk_placement.test_ssm_write_qsa_aux = true;
    defer disk_placement.test_ssm_write_qsa_aux = false;
    _ = try tier.writeSsmFile(dir_rel, &latest, s);
    const qsa_path = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/qsa.safetensors", .{ tier.root, id });
    defer testing.allocator.free(qsa_path);
    try std.Io.Dir.deleteFileAbsolute(io, qsa_path);
    tier.entries.items[0].qsa_history_bytes = 0;
    tier.entries.items[0].qsa_history_rows = 0;
    tier.entries.items[0].inherited_qsa = false;
    try tier.writeMeta(tier.entries.items[0]);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 128), try tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
    try testing.expect(transformer_mod.entriesHaveQsaHistory(&dst));
    try testing.expectEqual(@as(c_int, 128), dst[2].qsa_hist_rows);
}

test "DiskTier: a pre-v7 interior restore misses when the latest s* file is gone" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-prev7-miss", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var src128 = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src128);
    var src256 = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src256);
    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    src256[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src256[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src256[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src256, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src256, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    const id = tier.entries.items[0].id;
    var latest = try tier.loadSsmFile(id, 256, 3);
    defer latest.deinit(testing.allocator);
    var qsa = try tier.loadQsaHistoryFile(id, 3);
    defer qsa.deinit(testing.allocator);
    for (latest.layers, qsa.layers) |*l, q| {
        if (q.aux_state.ctx != null) {
            if (l.aux_state.ctx != null) _ = mlx.mlx_array_free(l.aux_state);
            l.aux_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&l.aux_state, q.aux_state));
        }
        if (q.qsa_pooled.ctx != null) {
            if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
            l.qsa_pooled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&l.qsa_pooled, q.qsa_pooled));
        }
        l.qsa_ratio = q.qsa_ratio;
    }
    const dir_rel = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}", .{ tier.root, id });
    defer testing.allocator.free(dir_rel);
    var interior = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s);
    defer interior.deinit(testing.allocator);
    _ = try tier.writeSsmFile(dir_rel, &interior, s);
    disk_placement.test_ssm_write_qsa_aux = true;
    defer disk_placement.test_ssm_write_qsa_aux = false;
    _ = try tier.writeSsmFile(dir_rel, &latest, s);
    const qsa_path = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/qsa.safetensors", .{ tier.root, id });
    defer testing.allocator.free(qsa_path);
    try std.Io.Dir.deleteFileAbsolute(io, qsa_path);
    const latest_path = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/s0000256.safetensors", .{ tier.root, id });
    defer testing.allocator.free(latest_path);
    try std.Io.Dir.deleteFileAbsolute(io, latest_path);
    tier.entries.items[0].qsa_history_bytes = 0;
    tier.entries.items[0].qsa_history_rows = 0;
    tier.entries.items[0].inherited_qsa = false;
    try tier.writeMeta(tier.entries.items[0]);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectError(error.DiskCacheQsaHistoryGap, tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
    try testing.expectEqual(@as(usize, 0), cache2.step);
    try testing.expect(dst[2].aux_state.ctx == null);
}

test "DiskTier: a v7 entry with missing or short qsa.safetensors misses before KV/SSM adopt" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 128, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-gap", 0, 128);
        defer tier.deinit();
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        const id = tier.entries.items[0].id;
        const qsa_path = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/qsa.safetensors", .{ tier.root, id });
        defer testing.allocator.free(qsa_path);
        try std.Io.Dir.deleteFileAbsolute(io, qsa_path);

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectError(error.DiskCacheQsaHistoryGap, tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
        try testing.expectEqual(@as(usize, 0), cache2.step);
        try testing.expect(dst[2].aux_state.ctx == null);
    }

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-short", 0, 128);
        defer tier.deinit();
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        const id = tier.entries.items[0].id;
        var src64 = buildHybridEntries(s, 200.0, 600.0);
        defer freeHybridEntries(&src64);
        const aux64 = [_]c_int{ 1, 64, 8 };
        const pooled64 = [_]c_int{ 1, 16, 8 };
        src64[2].aux_state = makeArange(s, &aux64, 700.0);
        src64[2].qsa_pooled = makeArange(s, &pooled64, 800.0);
        src64[2].qsa_ratio = 4;
        var cps64 = [_]transformer_mod.SSMCheckpoint{
            try transformer_mod.captureSsmCheckpoint(testing.allocator, &src64, 64, s),
        };
        defer for (&cps64) |*cp| cp.deinit(testing.allocator);
        try transformer_mod.attachQsaHistoryToLatest(&cps64, &src64, s);
        const dir_rel = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}", .{ tier.root, id });
        defer testing.allocator.free(dir_rel);
        _ = try tier.writeQsaHistoryFile(dir_rel, &cps64[0], s);

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectError(error.DiskCacheQsaHistoryGap, tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
        try testing.expectEqual(@as(usize, 0), cache2.step);
        try testing.expect(dst[2].aux_state.ctx == null);
    }

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-wt", 0, 128);
        defer tier.deinit();
        var wt_cps = [_]transformer_mod.SSMCheckpoint{
            try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 128, s),
            try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
        };
        defer for (&wt_cps) |*cp| cp.deinit(testing.allocator);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &wt_cps, s);
        try testing.expectEqual(@as(u32, 0), tier.entries.items[0].qsa_history_rows);

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectError(error.DiskCacheQsaHistoryGap, tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
        try testing.expectEqual(@as(usize, 0), cache2.step);
        try testing.expect(dst[2].aux_state.ctx == null);
    }
}

test "DiskTier: a history tensor shorter than cp_pos is a miss, not a short hit" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-short-tensor", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const aux_shape = [_]c_int{ 1, 8, 8 };
    const pooled_shape = [_]c_int{ 1, 2, 8 };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 16, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 64, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try testing.expectEqual(@as(c_int, 16), cps[0].layers[2].qsa_rows);
    try testing.expectEqual(@as(c_int, 64), cps[1].layers[2].qsa_rows);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);
    try testing.expect(cps[1].layers[2].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 64), cps[1].layers[2].qsa_rows);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectError(error.DiskCacheQsaHistoryGap, tier.restoreIntoHybrid(&cache2, &dst, 0, 16, s));
    try testing.expectEqual(@as(usize, 0), cache2.step);
    try testing.expect(dst[2].aux_state.ctx == null);
}

test "DiskTier: a QSA overlay error resets the half-built restore" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-ovl", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    disk_placement.test_qsa_overlay_mismatch = true;
    defer disk_placement.test_qsa_overlay_mismatch = false;
    try testing.expectError(error.SsmCheckpointLayerMismatch, tier.restoreIntoHybrid(&cache2, &dst, 0, 256, s));
    try testing.expectEqual(@as(usize, 0), cache2.step);
}

test "DiskTier chunk share: a prefix-diverging entry hard-links qsa.safetensors" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-share", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);

    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, &cps, s);
    const donor_id = tier.entries.items[0].id;
    try testing.expect(tier.entries.items[0].qsa_history_bytes > 0);
    try testing.expectEqual(@as(u32, 256), tier.entries.items[0].qsa_history_rows);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, &cps, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    const heir = &tier.entries.items[1];
    try testing.expect(heir.inherited_qsa);
    try testing.expectEqual(@as(u8, 8), DiskTier.metaVersionFor(heir.*));

    var pbuf: [1024]u8 = undefined;
    const dp = try std.fmt.bufPrint(&pbuf, "{s}/fp-qsa-share/e{d}/qsa.safetensors", .{ base, donor_id });
    const dstat = statFile(io, dp).?;
    var hbuf: [1024]u8 = undefined;
    const hp = try std.fmt.bufPrint(&hbuf, "{s}/fp-qsa-share/e{d}/qsa.safetensors", .{ base, heir.id });
    const hstat = statFile(io, hp).?;
    try testing.expectEqual(dstat.inode, hstat.inode);
    try testing.expectEqual(@as(u64, 2), @as(u64, @intCast(hstat.nlink)));

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    const m = tier.bestMatch(&toks.b, false, kv_quant.KVQuantConfig.dense, .{}).?;
    try testing.expectEqual(@as(u32, 256), try tier.restoreIntoHybrid(&cache2, &dst, m.idx, 256, s));
    try testing.expect(dst[2].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 256), dst[2].qsa_hist_rows);
}

test "DiskTier: an inherited QSA history is dropped on extend past the common prefix" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-ext", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);

    var donor_src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&donor_src);
    const donor_aux = [_]c_int{ 1, 600, 8 };
    const donor_pooled = [_]c_int{ 1, 150, 8 };
    donor_src[2].aux_state = makeArange(s, &donor_aux, 700.0);
    donor_src[2].qsa_pooled = makeArange(s, &donor_pooled, 800.0);
    donor_src[2].qsa_ratio = 4;
    var donor_cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &donor_src, 600, s),
    };
    defer for (&donor_cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&donor_cps, &donor_src, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, &donor_cps, s);

    var heir_src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&heir_src);
    const heir_aux1 = [_]c_int{ 1, 256, 8 };
    const heir_pooled1 = [_]c_int{ 1, 64, 8 };
    heir_src[2].aux_state = makeArange(s, &heir_aux1, 700.0);
    heir_src[2].qsa_pooled = makeArange(s, &heir_pooled1, 800.0);
    heir_src[2].qsa_ratio = 4;
    var heir_cps1 = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_src, 256, s),
    };
    defer for (&heir_cps1) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&heir_cps1, &heir_src, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, &heir_cps1, s);
    try testing.expect(tier.entries.items[1].inherited_qsa);

    var heir_src2 = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&heir_src2);
    const heir_aux2 = [_]c_int{ 1, 560, 8 };
    const heir_pooled2 = [_]c_int{ 1, 140, 8 };
    heir_src2[2].aux_state = makeArange(s, &heir_aux2, 900.0);
    heir_src2[2].qsa_pooled = makeArange(s, &heir_pooled2, 1000.0);
    heir_src2[2].qsa_ratio = 4;
    var heir_cps2 = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_src2, 256, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_src2, 560, s),
    };
    defer for (&heir_cps2) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&heir_cps2, &heir_src2, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, &heir_cps2, s);

    const heir = &tier.entries.items[1];
    try testing.expect(!heir.inherited_qsa);
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    const m = tier.bestMatch(&toks.b, false, kv_quant.KVQuantConfig.dense, .{}).?;
    try testing.expectEqual(@as(u32, 560), try tier.restoreIntoHybrid(&cache2, &dst, m.idx, 560, s));
    try testing.expect(dst[2].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 560), dst[2].qsa_hist_rows);
}

test "modelFingerprint: the f32 and the --qwen-gdn 16 GDN layouts each get their own SSD namespace" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    try tmp.dir.createDirPath(io, "m");
    try tmp.dir.writeFile(io, .{ .sub_path = "m/config.json", .data = "{}" });
    const dir = try std.fmt.allocPrint(testing.allocator, "{s}/m", .{base});
    defer testing.allocator.free(dir);

    const wide = model.ModelConfig{ .linear_num_value_heads = 48 };
    var narrow = wide;
    narrow.gdn_state_f32 = false;
    var glm = wide;
    glm.model_type = "glm5_next";
    const mimo = model.ModelConfig{ .model_type = "mimo_v2" };

    const today = try disk_placement.modelFingerprint(testing.allocator, io, dir);
    defer testing.allocator.free(today);
    // Entries written before the gate became f32 sit under `today`'s value and are never restored.
    const fp_narrow = try disk_placement.modelFingerprintWithLayout(testing.allocator, io, dir, narrow.cacheLayoutNamespace());
    defer testing.allocator.free(fp_narrow);
    try testing.expect(!std.mem.eql(u8, today, fp_narrow));
    const fp_f32 = try disk_placement.modelFingerprintWithLayout(testing.allocator, io, dir, wide.cacheLayoutNamespace());
    defer testing.allocator.free(fp_f32);
    try testing.expect(!std.mem.eql(u8, today, fp_f32));
    try testing.expect(!std.mem.eql(u8, fp_narrow, fp_f32));
    // GLM (KDA state, f32 always) and MiMo (no GDN layers) keep the historical namespace.
    for ([_]*const model.ModelConfig{ &glm, &mimo }) |c| {
        const fp = try disk_placement.modelFingerprintWithLayout(testing.allocator, io, dir, c.cacheLayoutNamespace());
        defer testing.allocator.free(fp);
        try testing.expectEqualStrings(today, fp);
    }
}

test "DiskTier: the qwen4 MTP head's QSA half round-trips exactly; a head-less sidecar declines the head only" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-head", 0, 128);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 600, 8, 9.5, .float32);

    var aux_src: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer {
        _ = mlx.mlx_array_free(aux_src.conv_state);
        _ = mlx.mlx_array_free(aux_src.ssm_state);
        transformer_mod.ssmFreeQsaState(&aux_src);
    }
    aux_src.aux_state = try filledArray(&[_]c_int{ 1, 600, 8 }, 4.25, s);
    aux_src.qsa_pooled = try filledArray(&[_]c_int{ 1, 150, 8 }, -1.75, s);
    aux_src.qsa_ratio = 4;
    var head_snap = transformer_mod.ssmSnapshot(&aux_src);
    defer transformer_mod.ssmSnapshotDeinit(&head_snap);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 11);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{
            .entries = mtp.entries,
            .step = mtp.step,
            .config = mtp.config,
            .base_pos = 0,
            .head_aux = &head_snap,
            .head_pos_base = 1,
        },
        s,
    );

    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-head", 0, 128);
    defer tier2.deinit();
    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
    var loaded = tier2.loadSpecSnap(m.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    defer loaded.snap.deinit();
    try testing.expectEqual(@as(usize, 600), loaded.snap.step);
    var back = loaded.head_aux orelse return error.TestExpectedHeadSnap;
    defer transformer_mod.ssmSnapshotDeinit(&back);
    try testing.expectEqual(@as(c_int, 1), loaded.head_pos_base);
    try testing.expectEqual(@as(c_int, 4), back.qsa_ratio);
    try testing.expectEqual(@as(c_int, 600), mlx.getShape(back.aux_state)[1]);
    try testing.expectEqual(@as(c_int, 150), mlx.getShape(back.qsa_pooled)[1]);
    try testing.expectEqual(@as(f32, 4.25), ssmArrVal(back.aux_state, 0, s));
    try testing.expectEqual(@as(f32, -1.75), ssmArrVal(back.qsa_pooled, 0, s));

    // Second entry, MTP history but no head: KV half loads, head half absent (a pre-v5 sidecar).
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 700_000);
    _ = try tier2.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens_b,
        false,
        null,
        null,
        .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 },
        s,
    );
    const mb = tier2.bestMatch(&tokens_b, false, kv_quant.KVQuantConfig.dense, .{}).?;
    var kv_only = tier2.loadSpecSnap(mb.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    defer kv_only.snap.deinit();
    try testing.expectEqual(@as(usize, 600), kv_only.snap.step);
    try testing.expect(kv_only.head_aux == null);
}

test "DiskTier: a head snap with no raw-history tensor drops the head half and arms no MLX latch" {
    try mlx.requireGpu();
    // Bar: a pooled-only head (qsa_rows > 0, empty `aux_state`) writes no `h.aux` and leaves the latch clear.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-noraw", 0, 128);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 600, 8, 9.5, .float32);

    var aux_src: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer {
        _ = mlx.mlx_array_free(aux_src.conv_state);
        _ = mlx.mlx_array_free(aux_src.ssm_state);
        transformer_mod.ssmFreeQsaState(&aux_src);
    }
    aux_src.qsa_pooled = try filledArray(&[_]c_int{ 1, 150, 8 }, -1.75, s);
    aux_src.qsa_ratio = 4;
    aux_src.qsa_hist_rows = 600;
    var head_snap = transformer_mod.ssmSnapshot(&aux_src);
    defer transformer_mod.ssmSnapshotDeinit(&head_snap);
    try testing.expect(head_snap.aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 600), head_snap.qsa_rows);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 31);
    try testing.expect(!mlx.errorPending());
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{
            .entries = mtp.entries,
            .step = mtp.step,
            .config = mtp.config,
            .base_pos = 0,
            .head_aux = &head_snap,
            .head_pos_base = 1,
        },
        s,
    );
    try testing.expect(!mlx.errorPending());

    // A pooled bank without its raw history is not restorable: KV half only, head declined.
    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-noraw", 0, 128);
    defer tier2.deinit();
    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
    var loaded = tier2.loadSpecSnap(m.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    defer loaded.snap.deinit();
    try testing.expectEqual(@as(usize, 600), loaded.snap.step);
    try testing.expect(loaded.head_aux == null);
}

test "DiskTier: a raise anywhere in appendCommit frees its owned results exactly once" {
    // Bar: the fault sweep reaches the QSA history write; a double free aborts under the testing allocator.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-exits", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const aux_shape = [_]c_int{ 1, 256, 8 };
    const pooled_shape = [_]c_int{ 1, 64, 8 };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 128, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);

    var k: u64 = 1;
    while (true) : (k += 1) {
        mlx.fault.arm(k);
        const r = tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        mlx.fault.disarm();
        if (!mlx.fault.didFire()) {
            _ = try r;
            break;
        }
        try testing.expectError(error.MlxError, r);
        tier.invalidateAll();
    }
    try testing.expect(k > 1);
}

test "DiskTier: MTP head QSA half round-trips a ring plus logical rows" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-head-ring", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 64, 8, 9.5, .float32);

    var aux_src: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer {
        _ = mlx.mlx_array_free(aux_src.conv_state);
        _ = mlx.mlx_array_free(aux_src.ssm_state);
        transformer_mod.ssmFreeQsaState(&aux_src);
    }
    aux_src.aux_state = try filledArray(&[_]c_int{ 1, transformer_mod.QSA_RING_ROWS, 8 }, 4.25, s);
    aux_src.qsa_pooled = try filledArray(&[_]c_int{ 1, 16, 8 }, -1.75, s);
    aux_src.qsa_ratio = 4;
    aux_src.qsa_hist_rows = 64;
    var head_snap = transformer_mod.ssmSnapshot(&aux_src);
    defer transformer_mod.ssmSnapshotDeinit(&head_snap);
    try testing.expectEqual(@as(c_int, 64), head_snap.qsa_rows);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 11);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{
            .entries = mtp.entries,
            .step = mtp.step,
            .config = mtp.config,
            .base_pos = 0,
            .head_aux = &head_snap,
            .head_pos_base = 1,
        },
        s,
    );

    const m = tier.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
    var loaded = tier.loadSpecSnap(m.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    defer loaded.snap.deinit();
    var back = loaded.head_aux orelse return error.TestExpectedHeadSnap;
    defer transformer_mod.ssmSnapshotDeinit(&back);
    try testing.expectEqual(@as(c_int, 64), back.qsa_rows);
    try testing.expectEqual(transformer_mod.QSA_RING_ROWS, mlx.getShape(back.aux_state)[1]);
    try testing.expectEqual(@as(c_int, 16), mlx.getShape(back.qsa_pooled)[1]);
}

test "DiskTier: v8 persists the head's checkpoint leftovers; a sidecar without them loads blind" {
    // The ring on disk is 32 rows and a warm clamp lands a whole generated tail below it, so
    // the leftovers the trunk's checkpoints marked have to survive the round trip — including
    // the meta.json parse a restart reads them back through.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-head-marks", 0, 128);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 64, 8, 9.5, .float32);

    var aux_src: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer {
        _ = mlx.mlx_array_free(aux_src.conv_state);
        _ = mlx.mlx_array_free(aux_src.ssm_state);
        transformer_mod.ssmFreeQsaState(&aux_src);
    }
    aux_src.aux_state = try filledArray(&[_]c_int{ 1, transformer_mod.QSA_RING_ROWS, 8 }, 4.25, s);
    aux_src.qsa_pooled = try filledArray(&[_]c_int{ 1, 16, 8 }, -1.75, s);
    aux_src.qsa_ratio = 4;
    aux_src.qsa_hist_rows = 64;
    var head_snap = transformer_mod.ssmSnapshot(&aux_src);
    defer transformer_mod.ssmSnapshotDeinit(&head_snap);

    // Two trunk checkpoint positions, both mid-block.
    var marks: transformer_mod.QsaHeadMarkSet = .{};
    defer marks.deinit();
    marks.put(26, try filledArray(&[_]c_int{ 1, 2, 8 }, 7.5, s));
    marks.put(46, try filledArray(&[_]c_int{ 1, 2, 8 }, -3.25, s));

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 11);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{
            .entries = mtp.entries,
            .step = mtp.step,
            .config = mtp.config,
            .base_pos = 0,
            .head_aux = &head_snap,
            .head_pos_base = 1,
            .head_marks = marks.slice(),
        },
        s,
    );

    // A restart: the positions come back through meta.json, the rows through the sidecar.
    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-head-marks", 0, 128);
    defer tier2.deinit();
    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
    try testing.expectEqual(@as(u8, 8), DiskTier.metaVersionFor(tier2.entries.items[m.idx]));
    var loaded = tier2.loadSpecSnap(m.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    defer loaded.snap.deinit();
    var back = loaded.head_aux orelse return error.TestExpectedHeadSnap;
    defer transformer_mod.ssmSnapshotDeinit(&back);
    var back_marks = loaded.head_marks;
    defer back_marks.deinit();
    try testing.expectEqual(@as(usize, 2), back_marks.len);
    const lv26 = back_marks.find(26) orelse return error.TestExpectedHeadMark;
    const lv46 = back_marks.find(46) orelse return error.TestExpectedHeadMark;
    try testing.expectEqual(@as(c_int, 2), mlx.getShape(lv26)[1]);
    try testing.expectEqual(@as(f32, 7.5), ssmArrVal(lv26, 0, s));
    try testing.expectEqual(@as(f32, -3.25), ssmArrVal(lv46, 0, s));

    // A v5..v7 sidecar carries none: the head still restores, and clamps as it does today.
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 700_000);
    _ = try tier2.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens_b,
        false,
        null,
        null,
        .{
            .entries = mtp.entries,
            .step = mtp.step,
            .config = mtp.config,
            .base_pos = 0,
            .head_aux = &head_snap,
            .head_pos_base = 1,
        },
        s,
    );
    const mb = tier2.bestMatch(&tokens_b, false, kv_quant.KVQuantConfig.dense, .{}).?;
    try testing.expectEqual(@as(u8, 5), DiskTier.metaVersionFor(tier2.entries.items[mb.idx]));
    var no_marks = tier2.loadSpecSnap(mb.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    defer no_marks.snap.deinit();
    var nb = no_marks.head_aux orelse return error.TestExpectedHeadSnap;
    defer transformer_mod.ssmSnapshotDeinit(&nb);
    try testing.expectEqual(@as(usize, 0), no_marks.head_marks.len);
}

test "DiskTier: an SSM checkpoint STAGES through the writer — no filesystem write on the inference thread" {
    // The checkpoint file is staged through the writer, before the meta.json that indexes it,
    // and the staged bytes are a real safetensors image (a fresh tier reads them back with
    // their `__metadata__`).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32); // >= MIN_PERSIST_TOKENS
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // A checkpoint with every optional part qwen4_exp carries.
    var src128 = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src128);
    const aux_shape = [_]c_int{ 1, 128, 4 };
    const pooled_shape = [_]c_int{ 1, 32, 4 };
    src128[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src128[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src128[2].qsa_ratio = 4;
    src128[1].ple_prev = .{ 42, 43, 0, 0, 0, 0, 0, 0 };
    src128[1].ple_prev_valid = true;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src128, s);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ssd-cpstage", 0, 128);
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);
    tier.writer.?.setPaused(true);
    defer if (tier.writer) |w| w.setPaused(false);

    try testing.expectEqual(
        PersistOutcome.persisted,
        try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s),
    );
    try testing.expectEqual(@as(usize, 1), tier.entries.items[0].ssm_positions.len);
    try testing.expectEqual(@as(u32, 128), tier.entries.items[0].ssm_positions[0]);

    // Bar 1: the checkpoint file is not on disk; the commit returned without writing it.
    try testing.expectError(
        error.FileNotFound,
        tmp.dir.statFile(io, "fp-ssd-cpstage/e1/s0000128.safetensors", .{}),
    );
    try testing.expectEqual(@as(u64, 0), tier.writer.?.filesWritten());
    try testing.expect(tier.entries.items[0].ssm_bytes[0] > 0);

    // Bar 2: staged before meta.json.
    var paths = std.ArrayList([]const u8).empty;
    defer {
        for (paths.items) |pp| testing.allocator.free(pp);
        paths.deinit(testing.allocator);
    }
    try tier.writer.?.stagedPaths(&paths, testing.allocator);
    const cp_at = for (paths.items, 0..) |pp, i| {
        if (std.mem.endsWith(u8, pp, "/s0000128.safetensors")) break i;
    } else return error.CheckpointNotStaged;
    const meta_at = for (paths.items, 0..) |pp, i| {
        if (std.mem.endsWith(u8, pp, "/meta.json")) break i;
    } else return error.MetaNotStaged;
    try testing.expect(cp_at < meta_at);

    tier.writer.?.setPaused(false);
    tier.drainWriter();
    const st = try tmp.dir.statFile(io, "fp-ssd-cpstage/e1/s0000128.safetensors", .{});
    try testing.expectEqual(tier.entries.items[0].ssm_bytes[0], st.size);

    // Bar 3: a fresh tier reads the hand-rolled image back, `__metadata__` included.
    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-ssd-cpstage", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 128), try tier2.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
    try testing.expectEqual(@as(f32, 100.0), ssmArrVal(dst[0].conv_state, 0, s));
    try testing.expectEqual(@as(f32, 500.0), ssmArrVal(dst[0].ssm_state, 0, s));
    try testing.expect(dst[2].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 128), dst[2].qsa_hist_rows);
    try testing.expectEqual(@as(f32, 800.0 + 11.0), ssmArrVal(dst[2].qsa_pooled, 11, s));
    try testing.expectEqual(@as(c_int, 4), dst[2].qsa_ratio);
    try testing.expect(dst[1].ple_prev_valid and dst[1].ple_prev[0] == 42 and dst[1].ple_prev[1] == 43);
    try testing.expectEqual(
        try cacheValueAt(&cache, 0, 127, 3, s),
        try cacheValueAt(&cache2, 0, 127, 3, s),
    );
}

test "DiskTier: a v7 full-aux QSA file serves a mid-block leftover inside a RING_ROWS ring; v>8 is refused" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-v8", 0, 128);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const hd: c_int = 8;
    const aux_shape = [_]c_int{ 1, 256, hd };
    const pooled_shape = [_]c_int{ 1, 64, hd };
    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 46, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    // A v7 entry: its snaps carry recurrent state and `qsa_rows` only, and the whole raw
    // indexer history — every checkpoint's leftover included — is the entry-level file.
    for (&cps) |*cp| for (cp.layers) |*l| {
        if (l.aux_state.ctx == null) continue;
        _ = mlx.mlx_array_free(l.aux_state);
        l.aux_state = .{ .ctx = null };
    };
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
    const id = tier.entries.items[0].id;
    const dir_rel = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}", .{ tier.root, id });
    defer testing.allocator.free(dir_rel);
    const full_layers = try testing.allocator.alloc(transformer_mod.SSMCacheEntrySnapshot, 3);
    for (full_layers) |*l| l.* = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
    };
    full_layers[2].aux_state = makeArange(s, &aux_shape, 700.0);
    full_layers[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    full_layers[2].qsa_ratio = 4;
    full_layers[2].qsa_rows = 256;
    var full_cp: transformer_mod.SSMCheckpoint = .{ .pos = 256, .layers = full_layers };
    defer full_cp.deinit(testing.allocator);
    _ = try tier.writeQsaHistoryFile(dir_rel, &full_cp, s);

    // The file keeps every row it was written with; the ring truncation is the live entry's.
    var loaded = try tier.loadQsaHistoryFile(id, 3);
    defer loaded.deinit(testing.allocator);
    try testing.expectEqual(@as(c_int, 256), loaded.layers[2].qsa_rows);
    try testing.expectEqual(@as(c_int, 256), mlx.getShape(loaded.layers[2].aux_state)[1]);

    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-qsa-v8", 0, 128);
    defer tier2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 46), try tier2.restoreIntoHybrid(&cache2, &dst, 0, 46, s));
    try testing.expectEqual(@as(c_int, 46), dst[2].qsa_hist_rows);
    try testing.expectEqual(@as(c_int, 11), mlx.getShape(dst[2].qsa_pooled)[1]);
    try testing.expectEqual(@as(f32, 800.0), ssmArrVal(dst[2].qsa_pooled, 0, s));
    // A restore at 46 (ratio 4) seeds exactly the 2 leftover rows 44 and 45, which are the checkpoint's
    // own leftover — source rows 44 and 45.
    const leftover = mlx.getShape(dst[2].aux_state)[1];
    try testing.expectEqual(@as(c_int, 2), leftover);
    try testing.expectEqual(@as(f32, 1052.0), ssmArrVal(dst[2].aux_state, 0, s));
    try testing.expectEqual(@as(f32, 1060.0), ssmArrVal(dst[2].aux_state, hd, s));

    try tmp.dir.createDirPath(io, "fp-qsa-v8/e9");
    try tmp.dir.writeFile(io, .{
        .sub_path = "fp-qsa-v8/e9/meta.json",
        .data = "{\"v\":9,\"kv_len\":1,\"tokens\":0,\"has_tools\":false,\"scheme\":\"off\",\"bits\":0,\"group_size\":0,\"chunk_tokens\":128,\"inherited_chunks\":0,\"bytes\":0,\"chunk_bytes\":[],\"ssm\":[]}",
    });
    try testing.expect(tier2.loadEntry(9) == null);
}

test "fingerprint indexed shards and ngram size" {
    try testFingerprintPayloadChange(true, false);
}

test "fingerprint indexed shards and ngram mtime" {
    try testFingerprintPayloadChange(true, true);
}

test "fingerprint single shard and ngram size" {
    try testFingerprintPayloadChange(false, false);
}

test "fingerprint single shard and ngram mtime" {
    try testFingerprintPayloadChange(false, true);
}

test "upstream bugfix: DiskTier: in-place commits keep an entry's bytes equal to the files it owns" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-qsa-bill", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 900, 8, 0.0, .float32);
    var tokens: [900]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var src = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&src);
    const aux_shape = [_]c_int{ 1, 600, 8 };
    const pooled_shape = [_]c_int{ 1, 150, 8 };
    src[2].aux_state = makeArange(s, &aux_shape, 700.0);
    src[2].qsa_pooled = makeArange(s, &pooled_shape, 800.0);
    src[2].qsa_ratio = 4;
    var cps = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 128, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 384, s),
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 600, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &src, s);
    // A fifth checkpoint with no QSA history: the append adds a file and leaves qsa.safetensors alone.
    var more = [_]transformer_mod.SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 64, s),
    };
    defer for (&more) |*cp| cp.deinit(testing.allocator);

    const Owned = struct {
        // Every file in a non-sharing entry's dir is its own; meta.json is the manifest, never billed.
        fn bytes(t: *DiskTier, e: *const IndexEntry) !u64 {
            const dir = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}", .{ t.root, e.id });
            defer testing.allocator.free(dir);
            var d = try std.Io.Dir.openDirAbsolute(t.io, dir, .{});
            defer d.close(t.io);
            const meta = try d.statFile(t.io, "meta.json", .{});
            var seen = disk_placement.InodeSet.init(testing.allocator);
            defer seen.deinit();
            return disk_placement.dirBytes(t.io, dir, &seen) - meta.size;
        }
    };

    // Turn 1: 600 tokens land with four checkpoints and a QSA history file.
    _ = try tier.appendCommit(cache.entries, 600, cache.config, tokens[0..600], false, &cps, s);
    try testing.expect(tier.entries.items[0].qsa_history_bytes > 0);
    try testing.expectEqual(try Owned.bytes(&tier, &tier.entries.items[0]), tier.entries.items[0].bytes);

    // Turn 2: same tokens, an MTP snap and one more checkpoint -> the ssm/spec-only append.
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);
    _ = try tier.appendCommitWithSpec(cache.entries, 600, cache.config, tokens[0..600], false, &more, null, .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 }, s);
    try testing.expect(tier.entries.items[0].spec_bytes > 0);
    try testing.expectEqual(@as(usize, 5), tier.entries.items[0].ssm_positions.len);
    try testing.expectEqual(try Owned.bytes(&tier, &tier.entries.items[0]), tier.entries.items[0].bytes);
    try testing.expectEqual(tier.entries.items[0].bytes, tier.total_bytes);

    // Turn 3: the conversation grows to 900 tokens, again with no checkpoints -> extend in place.
    _ = try tier.appendCommit(cache.entries, 900, cache.config, &tokens, false, null, s);
    const e = &tier.entries.items[0];
    try testing.expectEqual(@as(u32, 900), e.kv_len);
    try testing.expect(e.qsa_history_bytes > 0);
    try testing.expectEqual(try Owned.bytes(&tier, e), e.bytes);
    try testing.expectEqual(e.bytes, tier.total_bytes);
}

test "DiskTier write counters: a QSA history rewritten for a longer checkpoint counts its full bytes again, beside the checkpoint files" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-wqsa", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 2048, 8, 0.0, .float32);
    var tokens: [2048]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const aux_shape = [_]c_int{ 1, 256, 8 };
    var short = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&short);
    short[2].aux_state = makeArange(s, &aux_shape, 700.0);
    short[2].qsa_pooled = makeArange(s, &[_]c_int{ 1, 256, 8 }, 800.0);
    short[2].qsa_ratio = 4;
    var first = [_]transformer_mod.SSMCheckpoint{try transformer_mod.captureSsmCheckpoint(testing.allocator, &short, 1024, s)};
    defer for (&first) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&first, &short, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &first, s);
    const qsa_one = tier.entries.items[0].qsa_history_bytes;
    try testing.expect(qsa_one > 0);
    try testing.expectEqual(qsa_one, tier.stats.snapshot().written[@intFromEnum(write_stats.Kind.qsa)]);

    var long = buildHybridEntries(s, 200.0, 600.0);
    defer freeHybridEntries(&long);
    long[2].aux_state = makeArange(s, &aux_shape, 700.0);
    long[2].qsa_pooled = makeArange(s, &[_]c_int{ 1, 500, 8 }, 800.0);
    long[2].qsa_ratio = 4;
    var second = [_]transformer_mod.SSMCheckpoint{try transformer_mod.captureSsmCheckpoint(testing.allocator, &long, 2000, s)};
    defer for (&second) |*cp| cp.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&second, &long, s);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &second, s);
    const e = &tier.entries.items[0];
    try testing.expectEqual(@as(u32, 2000), e.qsa_history_rows);
    try testing.expect(e.qsa_history_bytes > qsa_one);
    const snap = tier.stats.snapshot();
    try testing.expectEqual(qsa_one + e.qsa_history_bytes, snap.written[@intFromEnum(write_stats.Kind.qsa)]);
    var points: u64 = 0;
    for (e.ssm_bytes) |b| points += b;
    try testing.expectEqual(points, snap.written[@intFromEnum(write_stats.Kind.ssm)]);
    try testing.expectEqual(@as(u64, 0), snap.failed);
}
