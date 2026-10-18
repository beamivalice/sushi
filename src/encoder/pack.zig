const std = @import("std");
const source = @import("source.zig");
const st = @import("safetensors.zig");

pub const Error = error{ UnknownSourceTensor, TrunkShapeUnsupported } || std.mem.Allocator.Error;

pub const flush_bytes: usize = 2 << 30;
const write_chunk: usize = 1 << 30;

pub fn writeAllChunked(file: std.Io.File, io: std.Io, bytes: []const u8) !void {
    var at: usize = 0;
    while (at < bytes.len) : (at += write_chunk) try file.writeStreamingAll(io, bytes[at..][0..@min(write_chunk, bytes.len - at)]);
}

pub const ShardBuilder = struct {
    allocator: std.mem.Allocator,
    header: std.ArrayList(u8) = .empty,
    payload: std.ArrayList(u8) = .empty,
    count: usize = 0,

    pub fn init(allocator: std.mem.Allocator) ShardBuilder {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ShardBuilder) void {
        self.header.deinit(self.allocator);
        self.payload.deinit(self.allocator);
    }

    pub fn add(self: *ShardBuilder, name: []const u8, dtype: []const u8, shape: []const usize, bytes: []const u8) !void {
        while (self.payload.items.len % 8 != 0) try self.payload.append(self.allocator, 0);
        const start = self.payload.items.len;
        try self.payload.appendSlice(self.allocator, bytes);
        try self.header.print(self.allocator, ",{f}:{{\"dtype\":\"{s}\",\"shape\":[", .{ std.json.fmt(name, .{}), dtype });
        for (shape, 0..) |d, i| try self.header.print(self.allocator, "{s}{d}", .{ if (i == 0) "" else ",", d });
        try self.header.print(self.allocator, "],\"data_offsets\":[{d},{d}]}}", .{ start, start + bytes.len });
        self.count += 1;
    }

    pub fn size(self: *const ShardBuilder) usize {
        return self.payload.items.len;
    }

    pub fn flush(self: *ShardBuilder, io: std.Io, dir: std.Io.Dir, file_name: []const u8, metadata: []const [2][]const u8) !void {
        var head = std.ArrayList(u8).empty;
        defer head.deinit(self.allocator);
        try head.appendSlice(self.allocator, "{\"__metadata__\":{");
        for (metadata, 0..) |pair, i| {
            try head.print(self.allocator, "{s}{f}:{f}", .{ if (i == 0) "" else ",", std.json.fmt(pair[0], .{}), std.json.fmt(pair[1], .{}) });
        }
        try head.append(self.allocator, '}');
        try head.appendSlice(self.allocator, self.header.items);
        try head.append(self.allocator, '}');
        while (head.items.len % 8 != 0) try head.append(self.allocator, ' ');
        var len_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &len_buf, head.items.len, .little);

        var tmp_buf: [160]u8 = undefined;
        const tmp_name = try std.fmt.bufPrint(&tmp_buf, ".{s}.tmp", .{file_name});
        var file = try dir.createFile(io, tmp_name, .{});
        {
            defer file.close(io);
            try file.writeStreamingAll(io, &len_buf);
            try file.writeStreamingAll(io, head.items);
            try writeAllChunked(file, io, self.payload.items);
        }
        try dir.rename(tmp_name, dir, file_name, io);
        self.header.clearRetainingCapacity();
        self.payload.clearRetainingCapacity();
        self.count = 0;
    }
};

pub const IndexEntry = struct { tensor: []u8, file: []u8 };
pub const Override = struct { base: []u8, bits: u32, group: u32 };

pub const Trunk = struct {
    allocator: std.mem.Allocator,
    index: std.ArrayList(IndexEntry) = .empty,
    overrides: std.ArrayList(Override) = .empty,
    shard_files: usize = 0,

    pub fn deinit(self: *Trunk) void {
        for (self.index.items) |e| {
            self.allocator.free(e.tensor);
            self.allocator.free(e.file);
        }
        self.index.deinit(self.allocator);
        for (self.overrides.items) |o| self.allocator.free(o.base);
        self.overrides.deinit(self.allocator);
    }
};

const ngram_marker = ".ple.ple_embedding.ngram_embedding.shard_";

pub fn qwenIsExpertOrNgram(key: []const u8) bool {
    return std.mem.indexOf(u8, key, ".mlp.experts.") != null or std.mem.indexOf(u8, key, ngram_marker) != null;
}

pub fn qwenServedName(buf: []u8, key: []const u8) ![]const u8 {
    const map = [_][2][]const u8{
        .{ "model.language_model.", "language_model.model." },
        .{ "mtp.", "language_model.mtp." },
        .{ "lm_head.", "language_model.lm_head." },
        .{ "model.visual.", "vision_tower." },
    };
    for (map) |m| {
        if (std.mem.startsWith(u8, key, m[0])) return std.fmt.bufPrint(buf, "{s}{s}", .{ m[1], key[m[0].len..] });
    }
    return error.UnknownSourceTensor;
}

fn sortedKeys(allocator: std.mem.Allocator, src: *const source.SafetensorsSource) ![][]const u8 {
    var keys = std.ArrayList([]const u8).empty;
    errdefer keys.deinit(allocator);
    var it = src.tensors.keyIterator();
    while (it.next()) |k| try keys.append(allocator, k.*);
    std.mem.sortUnstable([]const u8, keys.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    return keys.toOwnedSlice(allocator);
}

fn trunkShardName(buf: []u8, n: usize) ![]const u8 {
    return std.fmt.bufPrint(buf, "model-trunk-{d:0>5}.safetensors", .{n});
}

fn record(trunk: *Trunk, tensor: []const u8, file: []const u8) !void {
    const t = try trunk.allocator.dupe(u8, tensor);
    errdefer trunk.allocator.free(t);
    const f = try trunk.allocator.dupe(u8, file);
    errdefer trunk.allocator.free(f);
    try trunk.index.append(trunk.allocator, .{ .tensor = t, .file = f });
}

fn bf16FromF32(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

pub fn writeQwenTrunk(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    src: *const source.SafetensorsSource,
    trunk_bits: u32,
    trunk_group: u32,
) !Trunk {
    var trunk = Trunk{ .allocator = allocator };
    errdefer trunk.deinit();
    const keys = try sortedKeys(allocator, src);
    defer allocator.free(keys);

    var builder = ShardBuilder.init(allocator);
    defer builder.deinit();
    var pending = std.ArrayList([]u8).empty;
    defer {
        for (pending.items) |p| allocator.free(p);
        pending.deinit(allocator);
    }
    const meta = [_][2][]const u8{.{ "format", "mlx" }};
    var shard_no: usize = 0;
    var file_buf: [64]u8 = undefined;

    for (keys) |key| {
        if (qwenIsExpertOrNgram(key)) continue;
        var name_buf: [256]u8 = undefined;
        const served = try qwenServedName(&name_buf, key);
        const loc = src.tensors.get(key).?;
        const action = source.trunkSpecForTensor(.qwen4_exp, key, loc.dtype, loc.shape, trunk_bits, trunk_group);
        const nbytes = loc.end - loc.start;
        const raw = try allocator.alloc(u8, nbytes);
        defer allocator.free(raw);
        try src.readRange(loc, 0, raw);

        switch (action) {
            .retain_source => {
                try builder.add(served, loc.dtype, loc.shape, raw);
                try pending.append(allocator, try allocator.dupe(u8, served));
            },
            .retain_plus_one => {
                for (0..raw.len / 2) |i| {
                    const word = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
                    const plus = bf16FromF32(@as(f32, @bitCast(@as(u32, word) << 16)) + 1.0);
                    std.mem.writeInt(u16, raw[i * 2 ..][0..2], plus, .little);
                }
                try builder.add(served, loc.dtype, loc.shape, raw);
                try pending.append(allocator, try allocator.dupe(u8, served));
            },
            .retain_conv_last => {
                try builder.add(served, loc.dtype, &.{ loc.shape[0], loc.shape[2], loc.shape[1] }, raw);
                try pending.append(allocator, try allocator.dupe(u8, served));
            },
            .retain_channels_last => {
                const moved = try allocator.alloc(u8, raw.len);
                defer allocator.free(moved);
                const o = loc.shape[0];
                const i = loc.shape[1];
                const inner = loc.shape[2] * loc.shape[3] * loc.shape[4];
                for (0..o) |oi| for (0..i) |ii| for (0..inner) |k| {
                    @memcpy(moved[((oi * inner + k) * i + ii) * 2 ..][0..2], raw[((oi * i + ii) * inner + k) * 2 ..][0..2]);
                };
                try builder.add(served, loc.dtype, &.{ loc.shape[0], loc.shape[2], loc.shape[3], loc.shape[4], loc.shape[1] }, moved);
                try pending.append(allocator, try allocator.dupe(u8, served));
            },
            .affine => |a| {
                const rows = loc.shape[0];
                const cols = loc.shape[1];
                const words: []align(1) const u16 = std.mem.bytesAsSlice(u16, raw);
                const bf = try allocator.alloc(u16, words.len);
                defer allocator.free(bf);
                @memcpy(bf, words);
                var triple = try source.quantizeAffineBf16(allocator, bf, rows, cols, a.bits, a.group_size);
                defer triple.deinit(allocator);
                const base = served[0 .. served.len - ".weight".len];
                var nm: [272]u8 = undefined;
                const parts = [_]struct { suffix: []const u8, dtype: []const u8, bytes: []const u8, cols: usize }{
                    .{ .suffix = ".weight", .dtype = "U32", .bytes = std.mem.sliceAsBytes(triple.weight), .cols = triple.w_cols },
                    .{ .suffix = ".scales", .dtype = "BF16", .bytes = std.mem.sliceAsBytes(triple.scales), .cols = triple.s_cols },
                    .{ .suffix = ".biases", .dtype = "BF16", .bytes = std.mem.sliceAsBytes(triple.biases), .cols = triple.s_cols },
                };
                for (parts) |part| {
                    const full = try std.fmt.bufPrint(&nm, "{s}{s}", .{ base, part.suffix });
                    try builder.add(full, part.dtype, &.{ rows, part.cols }, part.bytes);
                    try pending.append(allocator, try allocator.dupe(u8, full));
                }
                if (a.override_in_config) {
                    const owned = try allocator.dupe(u8, base);
                    errdefer allocator.free(owned);
                    try trunk.overrides.append(allocator, .{ .base = owned, .bits = a.bits, .group = a.group_size });
                }
            },
        }
        if (builder.size() >= flush_bytes) {
            const file = try trunkShardName(&file_buf, shard_no);
            try builder.flush(io, dir, file, &meta);
            for (pending.items) |p| try record(&trunk, p, file);
            for (pending.items) |p| allocator.free(p);
            pending.clearRetainingCapacity();
            shard_no += 1;
        }
    }
    if (builder.count != 0) {
        const file = try trunkShardName(&file_buf, shard_no);
        try builder.flush(io, dir, file, &meta);
        for (pending.items) |p| try record(&trunk, p, file);
        shard_no += 1;
    }
    trunk.shard_files = shard_no;
    return trunk;
}

pub const NgramSpec = struct { bits: u32 = 4, group: u32 = 32 };
pub const NgramResult = struct { rows: u64, dim: usize, shards: usize };

const NgramShard = struct { key: []const u8, index: usize };

fn ngramShardIndex(key: []const u8) !?struct { layer: usize, shard: usize } {
    const at = std.mem.indexOf(u8, key, ngram_marker) orelse return null;
    const layer_text = key["model.language_model.layers.".len..at];
    const rest = key[at + ngram_marker.len ..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.UnknownSourceTensor;
    const layer = std.fmt.parseInt(usize, layer_text, 10) catch return error.UnknownSourceTensor;
    const shard = std.fmt.parseInt(usize, rest[0..dot], 10) catch return error.UnknownSourceTensor;
    return .{ .layer = layer, .shard = shard };
}

pub fn writeNgramTable(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    src: *const source.SafetensorsSource,
    spec: NgramSpec,
) !?NgramResult {
    var shards = std.ArrayList(NgramShard).empty;
    defer shards.deinit(allocator);
    var layer: ?usize = null;
    var it = src.tensors.keyIterator();
    while (it.next()) |k| {
        const parsed = (try ngramShardIndex(k.*)) orelse continue;
        if (layer) |l| {
            if (l != parsed.layer) return error.UnknownSourceTensor;
        } else layer = parsed.layer;
        try shards.append(allocator, .{ .key = k.*, .index = parsed.shard });
    }
    if (shards.items.len == 0) return null;
    std.mem.sortUnstable(NgramShard, shards.items, {}, struct {
        fn lt(_: void, a: NgramShard, b: NgramShard) bool {
            return a.index < b.index;
        }
    }.lt);
    for (shards.items, 0..) |sh, i| if (sh.index != i) return error.UnknownSourceTensor;

    const first = src.tensors.get(shards.items[0].key).?;
    if (!std.mem.eql(u8, first.dtype, "BF16") or first.shape.len != 2) return error.TrunkShapeUnsupported;
    const rps = first.shape[0];
    const dim = first.shape[1];
    if (spec.group == 0 or dim % spec.group != 0 or (dim * spec.bits) % 32 != 0) return error.TrunkShapeUnsupported;
    for (shards.items) |sh| {
        const loc = src.tensors.get(sh.key).?;
        if (!std.mem.eql(u8, loc.dtype, "BF16") or loc.shape.len != 2 or loc.shape[0] != rps or loc.shape[1] != dim) return error.TrunkShapeUnsupported;
    }
    const total_rows: usize = rps * shards.items.len;
    const wc = dim * spec.bits / 32;
    const sc = dim / spec.group;
    const w_bytes = total_rows * wc * 4;
    const s_bytes = total_rows * sc * 2;

    var head = std.ArrayList(u8).empty;
    defer head.deinit(allocator);
    try head.print(allocator, "{{\"__metadata__\":{{\"format\":\"mlx-serve-ngram\",\"bits\":\"{d}\",\"group_size\":\"{d}\"}}", .{ spec.bits, spec.group });
    try head.print(allocator, ",\"weight\":{{\"dtype\":\"U32\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}}", .{ total_rows, wc, w_bytes });
    try head.print(allocator, ",\"scales\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ total_rows, sc, w_bytes, w_bytes + s_bytes });
    try head.print(allocator, ",\"biases\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}}}", .{ total_rows, sc, w_bytes + s_bytes, w_bytes + 2 * s_bytes });
    while (head.items.len % 8 != 0) try head.append(allocator, ' ');
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, head.items.len, .little);
    const data_at: u64 = 8 + head.items.len;

    const tmp_name = ".ngram_table.bin.tmp";
    var file = try dir.createFile(io, tmp_name, .{});
    {
        defer file.close(io);
        try file.writeStreamingAll(io, &len_buf);
        try file.writeStreamingAll(io, head.items);
        const chunk_rows: usize = 8192;
        const raw = try allocator.alloc(u8, chunk_rows * dim * 2);
        defer allocator.free(raw);
        const bf = try allocator.alloc(u16, chunk_rows * dim);
        defer allocator.free(bf);
        for (shards.items, 0..) |sh, si| {
            const loc = src.tensors.get(sh.key).?;
            var r0: usize = 0;
            while (r0 < rps) {
                const n = @min(chunk_rows, rps - r0);
                const bytes = raw[0 .. n * dim * 2];
                try src.readRange(loc, @intCast(r0 * dim * 2), bytes);
                for (bf[0 .. n * dim], 0..) |*v, i| v.* = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
                var triple = try source.quantizeAffineBf16(allocator, bf[0 .. n * dim], n, dim, spec.bits, spec.group);
                defer triple.deinit(allocator);
                const row0 = si * rps + r0;
                try file.writePositionalAll(io, std.mem.sliceAsBytes(triple.weight), data_at + row0 * wc * 4);
                try file.writePositionalAll(io, std.mem.sliceAsBytes(triple.scales), data_at + w_bytes + row0 * sc * 2);
                try file.writePositionalAll(io, std.mem.sliceAsBytes(triple.biases), data_at + w_bytes + s_bytes + row0 * sc * 2);
                r0 += n;
            }
        }
    }
    try dir.rename(tmp_name, dir, "ngram_table.bin", io);
    return .{ .rows = total_rows, .dim = dim, .shards = shards.items.len };
}

pub const ExpertQuantSpec = struct {
    k_text: []const u8,
    codebook: []const u8,
    window: u32,
    provenance: []const [2][]const u8 = &.{},
};

pub fn buildConfig(
    allocator: std.mem.Allocator,
    source_config: []const u8,
    trunk_bits: u32,
    trunk_group: u32,
    overrides: []const Override,
    ngram: ?NgramSpec,
    eq: ExpertQuantSpec,
) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, source_config, .{}) catch return error.UnknownSourceFormat;
    if (parsed != .object) return error.UnknownSourceFormat;
    var root = parsed.object;

    var quant = std.json.ObjectMap.empty;
    try quant.put(a, "group_size", .{ .integer = trunk_group });
    try quant.put(a, "bits", .{ .integer = trunk_bits });
    try quant.put(a, "mode", .{ .string = "affine" });
    for (overrides) |o| {
        var entry = std.json.ObjectMap.empty;
        try entry.put(a, "bits", .{ .integer = o.bits });
        try entry.put(a, "group_size", .{ .integer = o.group });
        try entry.put(a, "mode", .{ .string = "affine" });
        try quant.put(a, try a.dupe(u8, o.base), .{ .object = entry });
    }
    try root.put(a, "quantization", .{ .object = quant });
    try root.put(a, "quantization_config", .{ .object = quant });

    if (ngram) |n| {
        var table = std.json.ObjectMap.empty;
        try table.put(a, "file", .{ .string = "ngram_table.bin" });
        try table.put(a, "bits", .{ .integer = n.bits });
        try table.put(a, "group_size", .{ .integer = n.group });
        try root.put(a, "ngram_table", .{ .object = table });
    }

    var expert = std.json.ObjectMap.empty;
    try expert.put(a, "format", .{ .string = "exl3" });
    try expert.put(a, "k", .{ .number_string = eq.k_text });
    try expert.put(a, "codebook", .{ .string = eq.codebook });
    try expert.put(a, "window", .{ .integer = eq.window });
    try expert.put(a, "out_scales", .{ .string = "svh" });
    for (eq.provenance) |pair| try expert.put(a, pair[0], .{ .string = pair[1] });
    try root.put(a, "expert_quant", .{ .object = expert });

    return std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = root }, .{ .whitespace = .indent_2 });
}

pub fn buildIndex(allocator: std.mem.Allocator, entries: []const IndexEntry, total_size: u64) ![]u8 {
    const order = try allocator.alloc(usize, entries.len);
    defer allocator.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sortUnstable(usize, order, entries, struct {
        fn lt(e: []const IndexEntry, a: usize, b: usize) bool {
            return std.mem.lessThan(u8, e[a].tensor, e[b].tensor);
        }
    }.lt);
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator, "{{\"metadata\":{{\"total_size\":{d}}},\"weight_map\":{{", .{total_size});
    for (order, 0..) |idx, i| {
        try out.print(allocator, "{s}{f}:{f}", .{ if (i == 0) "" else ",", std.json.fmt(entries[idx].tensor, .{}), std.json.fmt(entries[idx].file, .{}) });
    }
    try out.appendSlice(allocator, "}}");
    return out.toOwnedSlice(allocator);
}

fn sidecarWanted(name: []const u8, kind: std.Io.File.Kind) bool {
    if (kind != .file or name.len == 0 or name[0] == '.') return false;
    const skip_suffix = [_][]const u8{ ".safetensors", ".bin", ".pt", ".pth", ".gguf", ".orig-262k" };
    for (skip_suffix) |suf| if (std.mem.endsWith(u8, name, suf)) return false;
    if (std.mem.eql(u8, name, "config.json") or std.mem.eql(u8, name, "model.safetensors.index.json")) return false;
    return true;
}

pub fn copySidecars(allocator: std.mem.Allocator, io: std.Io, src_dir: std.Io.Dir, out_dir: std.Io.Dir) !usize {
    var names = std.ArrayList([]u8).empty;
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }
    var it = src_dir.iterate();
    while (try it.next(io)) |entry| {
        if (sidecarWanted(entry.name, entry.kind)) try names.append(allocator, try allocator.dupe(u8, entry.name));
    }
    for (names.items) |n| try src_dir.copyFile(n, out_dir, n, io, .{});
    return names.items.len;
}

pub const fixture = struct {
    const hidden = 128;

    fn bf16Bytes(allocator: std.mem.Allocator, rows: usize, cols: usize, seed: u64) ![]u8 {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const out = try allocator.alloc(u8, rows * cols * 2);
        for (0..rows * cols) |i| {
            const v: u16 = @truncate(@as(u32, @bitCast(r.floatNorm(f32) * 0.05)) >> 16);
            std.mem.writeInt(u16, out[i * 2 ..][0..2], v, .little);
        }
        return out;
    }

    pub const Spec = struct { name: []const u8, dtype: []const u8, shape: []const usize };

    pub fn write(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, specs: []const Spec) !void {
        var b = ShardBuilder.init(allocator);
        defer b.deinit();
        for (specs, 0..) |spec, i| {
            var n: usize = 1;
            for (spec.shape) |d| n *= d;
            const bytes = if (std.mem.eql(u8, spec.dtype, "BF16")) try bf16Bytes(allocator, n, 1, 100 + i) else blk: {
                const buf = try allocator.alloc(u8, n * 4);
                @memset(buf, 0);
                break :blk buf;
            };
            defer allocator.free(bytes);
            try b.add(spec.name, spec.dtype, spec.shape, bytes);
        }
        try b.flush(io, dir, "model.safetensors", &.{});
    }
};

test "pack: the Qwen trunk is renamed, affine-quantized by rule, and the experts and n-gram shards are left out" {
    const t = std.testing;
    const io = t.io;
    const specs = [_]fixture.Spec{
        .{ .name = "model.language_model.embed_tokens.weight", .dtype = "BF16", .shape = &.{ 256, 128 } },
        .{ .name = "model.language_model.layers.0.self_attn.q_proj.weight", .dtype = "BF16", .shape = &.{ 128, 128 } },
        .{ .name = "model.language_model.layers.0.linear_attn.in_proj_z.weight", .dtype = "BF16", .shape = &.{ 128, 256 } },
        .{ .name = "model.language_model.layers.0.linear_attn.A_log", .dtype = "F32", .shape = &.{16} },
        .{ .name = "model.language_model.layers.0.mlp.gate.weight", .dtype = "BF16", .shape = &.{ 8, 128 } },
        .{ .name = "model.language_model.layers.0.mlp.experts.gate_up_proj", .dtype = "BF16", .shape = &.{ 2, 256, 128 } },
        .{ .name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", .dtype = "BF16", .shape = &.{ 16, 32 } },
        .{ .name = "model.visual.blocks.0.attn.qkv.weight", .dtype = "BF16", .shape = &.{ 64, 128 } },
        .{ .name = "mtp.fc_hidden.weight", .dtype = "BF16", .shape = &.{ 128, 256 } },
        .{ .name = "lm_head.weight", .dtype = "BF16", .shape = &.{ 256, 256 } },
    };
    var src_dir = t.tmpDir(.{});
    defer src_dir.cleanup();
    try fixture.write(t.allocator, io, src_dir.dir, &specs);
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir.dir);
    defer src.deinit();

    var out_dir = t.tmpDir(.{});
    defer out_dir.cleanup();
    var trunk = try writeQwenTrunk(t.allocator, io, out_dir.dir, &src, 8, 64);
    defer trunk.deinit();
    try t.expectEqual(@as(usize, 1), trunk.shard_files);

    const raw = try out_dir.dir.readFileAlloc(io, "model-trunk-00000.safetensors", t.allocator, .limited(1 << 24));
    defer t.allocator.free(raw);
    const aligned = try t.allocator.alignedAlloc(u8, .of(u16), raw.len);
    defer t.allocator.free(aligned);
    @memcpy(aligned, raw);
    var file = try st.parse(t.allocator, aligned);
    defer file.deinit(t.allocator);

    const must_exist = [_][]const u8{
        "language_model.model.embed_tokens.weight",         "language_model.model.embed_tokens.scales",
        "language_model.model.layers.0.self_attn.q_proj.biases", "language_model.model.layers.0.linear_attn.in_proj_z.scales",
        "language_model.model.layers.0.linear_attn.A_log",  "language_model.model.layers.0.mlp.gate.weight",
        "vision_tower.blocks.0.attn.qkv.weight",            "language_model.mtp.fc_hidden.weight",
        "language_model.lm_head.biases",
    };
    for (must_exist) |name| try t.expect(file.tensors.get(name) != null);
    try t.expect(file.tensors.get("language_model.model.layers.0.mlp.gate.scales") == null);
    try t.expect(file.tensors.get("vision_tower.blocks.0.attn.qkv.scales") == null);
    var it = file.tensors.keyIterator();
    while (it.next()) |k| {
        try t.expect(std.mem.indexOf(u8, k.*, "experts") == null);
        try t.expect(std.mem.indexOf(u8, k.*, "ngram") == null);
    }
    try t.expectEqual(@as(usize, trunk.index.items.len), file.tensors.count());

    const z = file.tensors.get("language_model.model.layers.0.linear_attn.in_proj_z.scales").?;
    try t.expectEqual(@as(usize, 4), z.shape[1]);
    const lm = file.tensors.get("language_model.lm_head.scales").?;
    try t.expectEqual(@as(usize, 4), lm.shape[1]);
    const q = file.tensors.get("language_model.model.layers.0.self_attn.q_proj.scales").?;
    try t.expectEqual(@as(usize, 2), q.shape[1]);

    try t.expectEqual(@as(usize, 0), trunk.overrides.items.len);
    var nb: [32]u8 = undefined;
    try t.expectError(error.UnknownSourceTensor, qwenServedName(&nb, "unknown.tensor"));
}

test "pack: the Qwen trunk stores norm weights as one plus the source, conv kernels as [C,K,1] and the patch embedding channels-last" {
    const t = std.testing;
    const io = t.io;
    const specs = [_]fixture.Spec{
        .{ .name = "model.language_model.layers.0.self_attn.q_norm.weight", .dtype = "BF16", .shape = &.{64} },
        .{ .name = "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight", .dtype = "BF16", .shape = &.{64} },
        .{ .name = "mtp.pre_fc_norm_hidden.weight", .dtype = "BF16", .shape = &.{64} },
        .{ .name = "model.language_model.layers.0.linear_attn.norm.weight", .dtype = "BF16", .shape = &.{64} },
        .{ .name = "model.visual.blocks.0.norm1.weight", .dtype = "BF16", .shape = &.{64} },
        .{ .name = "model.language_model.layers.0.linear_attn.conv1d.weight", .dtype = "BF16", .shape = &.{ 96, 1, 4 } },
        .{ .name = "model.visual.patch_embed.proj.weight", .dtype = "BF16", .shape = &.{ 6, 3, 2, 4, 4 } },
    };
    var src_dir = t.tmpDir(.{});
    defer src_dir.cleanup();
    try fixture.write(t.allocator, io, src_dir.dir, &specs);
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir.dir);
    defer src.deinit();
    var out_dir = t.tmpDir(.{});
    defer out_dir.cleanup();
    var trunk = try writeQwenTrunk(t.allocator, io, out_dir.dir, &src, 8, 64);
    defer trunk.deinit();

    const raw = try out_dir.dir.readFileAlloc(io, "model-trunk-00000.safetensors", t.allocator, .limited(1 << 24));
    defer t.allocator.free(raw);
    const aligned = try t.allocator.alignedAlloc(u8, .of(u16), raw.len);
    defer t.allocator.free(aligned);
    @memcpy(aligned, raw);
    var file = try st.parse(t.allocator, aligned);
    defer file.deinit(t.allocator);

    const source_words = struct {
        fn read(allocator: std.mem.Allocator, s: *const source.SafetensorsSource, key: []const u8) ![]u16 {
            const loc = s.tensors.get(key).?;
            const bytes = try allocator.alloc(u8, loc.end - loc.start);
            defer allocator.free(bytes);
            try s.readRange(loc, 0, bytes);
            const words = try allocator.alloc(u16, bytes.len / 2);
            for (words, 0..) |*w, i| w.* = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
            return words;
        }
    }.read;

    const plus_one = [_][2][]const u8{
        .{ "model.language_model.layers.0.self_attn.q_norm.weight", "language_model.model.layers.0.self_attn.q_norm.weight" },
        .{ "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight", "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight" },
        .{ "mtp.pre_fc_norm_hidden.weight", "language_model.mtp.pre_fc_norm_hidden.weight" },
    };
    for (plus_one) |pair| {
        const words = try source_words(t.allocator, &src, pair[0]);
        defer t.allocator.free(words);
        const got = st.slice(u16, file.tensors.get(pair[1]).?);
        try t.expectEqual(words.len, got.len);
        for (words, got) |w, g| try t.expectEqual(bf16FromF32(@as(f32, @bitCast(@as(u32, w) << 16)) + 1.0), g);
    }
    for ([_][2][]const u8{
        .{ "model.language_model.layers.0.linear_attn.norm.weight", "language_model.model.layers.0.linear_attn.norm.weight" },
        .{ "model.visual.blocks.0.norm1.weight", "vision_tower.blocks.0.norm1.weight" },
    }) |pair| {
        const words = try source_words(t.allocator, &src, pair[0]);
        defer t.allocator.free(words);
        try t.expectEqualSlices(u16, words, st.slice(u16, file.tensors.get(pair[1]).?));
    }

    const conv = file.tensors.get("language_model.model.layers.0.linear_attn.conv1d.weight").?;
    try t.expectEqualSlices(usize, &.{ 96, 4, 1 }, conv.shape);
    const conv_src = try source_words(t.allocator, &src, "model.language_model.layers.0.linear_attn.conv1d.weight");
    defer t.allocator.free(conv_src);
    try t.expectEqualSlices(u16, conv_src, st.slice(u16, conv));

    const patch = file.tensors.get("vision_tower.patch_embed.proj.weight").?;
    try t.expectEqualSlices(usize, &.{ 6, 2, 4, 4, 3 }, patch.shape);
    const patch_src = try source_words(t.allocator, &src, "model.visual.patch_embed.proj.weight");
    defer t.allocator.free(patch_src);
    const patch_got = st.slice(u16, patch);
    for (0..6) |o| for (0..3) |i| for (0..2) |a| for (0..4) |b| for (0..4) |c| {
        try t.expectEqual(patch_src[(((o * 3 + i) * 2 + a) * 4 + b) * 4 + c], patch_got[(((o * 2 + a) * 4 + b) * 4 + c) * 3 + i]);
    };
}

test "pack: the n-gram shards merge into one 4-bit group-32 table the engine's reader describes" {
    const t = std.testing;
    const io = t.io;
    const specs = [_]fixture.Spec{
        .{ .name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight", .dtype = "BF16", .shape = &.{ 24, 64 } },
        .{ .name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", .dtype = "BF16", .shape = &.{ 24, 64 } },
        .{ .name = "model.language_model.embed_tokens.weight", .dtype = "BF16", .shape = &.{ 8, 64 } },
    };
    var src_dir = t.tmpDir(.{});
    defer src_dir.cleanup();
    try fixture.write(t.allocator, io, src_dir.dir, &specs);
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir.dir);
    defer src.deinit();

    var out_dir = t.tmpDir(.{});
    defer out_dir.cleanup();
    const result = (try writeNgramTable(t.allocator, io, out_dir.dir, &src, .{})).?;
    try t.expectEqual(@as(u64, 48), result.rows);
    try t.expectEqual(@as(usize, 64), result.dim);
    try t.expectEqual(@as(usize, 2), result.shards);

    const raw = try out_dir.dir.readFileAlloc(io, "ngram_table.bin", t.allocator, .limited(1 << 24));
    defer t.allocator.free(raw);
    const aligned = try t.allocator.alignedAlloc(u8, .of(u16), raw.len);
    defer t.allocator.free(aligned);
    @memcpy(aligned, raw);
    var file = try st.parse(t.allocator, aligned);
    defer file.deinit(t.allocator);
    try t.expectEqualStrings("mlx-serve-ngram", file.metadata.get("format").?);
    try t.expectEqualStrings("4", file.metadata.get("bits").?);
    try t.expectEqualStrings("32", file.metadata.get("group_size").?);
    const w = file.tensors.get("weight").?;
    try t.expectEqualSlices(usize, &.{ 48, 8 }, w.shape);
    try t.expectEqualSlices(usize, &.{ 48, 2 }, file.tensors.get("scales").?.shape);

    const want = try t.allocator.alloc(u16, 48 * 64);
    defer t.allocator.free(want);
    for ([_][]const u8{ "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_1.weight" }, 0..) |key, i| {
        const loc = src.tensors.get(key).?;
        const bytes = try t.allocator.alloc(u8, 24 * 64 * 2);
        defer t.allocator.free(bytes);
        try src.readRange(loc, 0, bytes);
        for (want[i * 24 * 64 ..][0 .. 24 * 64], 0..) |*v, j| v.* = std.mem.readInt(u16, bytes[j * 2 ..][0..2], .little);
    }
    var triple = try source.quantizeAffineBf16(t.allocator, want, 48, 64, 4, 32);
    defer triple.deinit(t.allocator);
    try t.expectEqualSlices(u32, triple.weight, st.slice(u32, w));
    try t.expectEqualSlices(u16, triple.scales, st.slice(u16, file.tensors.get("scales").?));
    try t.expectEqualSlices(u16, triple.biases, st.slice(u16, file.tensors.get("biases").?));

    var none_dir = t.tmpDir(.{});
    defer none_dir.cleanup();
    try t.expect((try writeNgramTable(t.allocator, io, none_dir.dir, &src, .{ .bits = 4, .group = 32 })) != null);
}

test "pack: config gains the quantization block, the n-gram table and the expert_quant stamp; the index sorts" {
    const t = std.testing;
    const io = t.io;
    const src_cfg =
        \\{"model_type":"qwen4_exp","text_config":{"hidden_size":2560,"num_experts":512},"tie_word_embeddings":false}
    ;
    var overrides = [_]Override{.{ .base = @constCast("language_model.lm_head"), .bits = 8, .group = 128 }};
    const prov = [_][2][]const u8{ .{ "quantizer", "ldlq-rotated" }, .{ "calib_tokens", "65536" } };
    const out = try buildConfig(t.allocator, src_cfg, 8, 64, &overrides, .{}, .{ .k_text = "2.5", .codebook = "mcg", .window = 15, .provenance = &prov });
    defer t.allocator.free(out);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, out, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try t.expectEqualStrings("qwen4_exp", root.get("model_type").?.string);
    const q = root.get("quantization").?.object;
    try t.expectEqual(@as(i64, 8), q.get("bits").?.integer);
    try t.expectEqual(@as(i64, 64), q.get("group_size").?.integer);
    try t.expectEqual(@as(i64, 128), q.get("language_model.lm_head").?.object.get("group_size").?.integer);
    try t.expect(root.get("quantization_config") != null);
    try t.expectEqualStrings("ngram_table.bin", root.get("ngram_table").?.object.get("file").?.string);
    const eq = root.get("expert_quant").?.object;
    try t.expectEqualStrings("exl3", eq.get("format").?.string);
    try t.expectEqual(@as(f64, 2.5), eq.get("k").?.float);
    try t.expectEqual(@as(i64, 15), eq.get("window").?.integer);
    try t.expectEqualStrings("65536", eq.get("calib_tokens").?.string);

    const entries = [_]IndexEntry{
        .{ .tensor = @constCast("b.weight"), .file = @constCast("model-trunk-00000.safetensors") },
        .{ .tensor = @constCast("a.weight"), .file = @constCast("layer-0.safetensors") },
    };
    const index = try buildIndex(t.allocator, &entries, 1234);
    defer t.allocator.free(index);
    try t.expect(std.mem.indexOf(u8, index, "\"a.weight\"").? < std.mem.indexOf(u8, index, "\"b.weight\"").?);
    const ip = try std.json.parseFromSlice(std.json.Value, t.allocator, index, .{});
    defer ip.deinit();
    try t.expectEqual(@as(i64, 1234), ip.value.object.get("metadata").?.object.get("total_size").?.integer);

    var src_dir = t.tmpDir(.{});
    defer src_dir.cleanup();
    for ([_][]const u8{ "tokenizer.json", "chat_template.jinja", "config.json", "model-00001.safetensors", "pytorch_model.bin", ".gitattributes" }) |n| {
        try src_dir.dir.writeFile(io, .{ .sub_path = n, .data = "x" });
    }
    var out_dir = t.tmpDir(.{});
    defer out_dir.cleanup();
    try t.expectEqual(@as(usize, 2), try copySidecars(t.allocator, io, src_dir.dir, out_dir.dir));
    try out_dir.dir.access(io, "tokenizer.json", .{});
    try t.expectError(error.FileNotFound, out_dir.dir.access(io, "config.json", .{}));
}

fn modelsRoot(buf: []u8, name: []const u8) error{SkipZigTest}![]const u8 {
    const root = std.c.getenv("SUSHI_MODELS_DIR") orelse return error.SkipZigTest;
    const text = std.mem.span(root);
    if (text.len == 0 or !std.fs.path.isAbsolute(text)) return error.SkipZigTest;
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ text, name }) catch error.SkipZigTest;
}

test "pack: the trunk plan names every tensor of a served Qwen pack and quantizes the same ones" {
    const t = std.testing;
    const io = t.io;
    var src_buf: [std.fs.max_path_bytes]u8 = undefined;
    var served_buf: [std.fs.max_path_bytes]u8 = undefined;
    const src_path = try modelsRoot(&src_buf, "Qwen/Qwen3.8-Flash-Next");
    const served_path = try modelsRoot(&served_buf, "Qwen3.8-Flash-Next-Sushi-2.6bpw");
    var src_dir = std.Io.Dir.cwd().openDir(io, src_path, .{ .iterate = true }) catch return error.SkipZigTest;
    defer src_dir.close(io);
    var served_dir = std.Io.Dir.cwd().openDir(io, served_path, .{}) catch return error.SkipZigTest;
    defer served_dir.close(io);
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir);
    defer src.deinit();
    const raw = try served_dir.readFileAlloc(io, "model.safetensors.index.json", t.allocator, .limited(64 << 20));
    defer t.allocator.free(raw);
    const index = try std.json.parseFromSlice(std.json.Value, t.allocator, raw, .{});
    defer index.deinit();
    const wm = index.value.object.get("weight_map").?.object;

    var planned: usize = 0;
    var it = src.tensors.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        if (qwenIsExpertOrNgram(key)) continue;
        var nb: [256]u8 = undefined;
        const served = try qwenServedName(&nb, key);
        const action = source.trunkSpecForTensor(.qwen4_exp, key, e.value_ptr.dtype, e.value_ptr.shape, 8, 64);
        planned += 1;
        switch (action) {
            .retain_source, .retain_plus_one, .retain_conv_last, .retain_channels_last => try t.expect(wm.contains(served)),
            .affine => {
                const base = served[0 .. served.len - ".weight".len];
                var sb: [272]u8 = undefined;
                try t.expect(wm.contains(try std.fmt.bufPrint(&sb, "{s}.scales", .{base})));
                try t.expect(wm.contains(try std.fmt.bufPrint(&sb, "{s}.biases", .{base})));
                try t.expect(wm.contains(served));
            },
        }
    }
    try t.expect(planned > 1000);
    var extra: usize = 0;
    var wit = wm.iterator();
    while (wit.next()) |e| {
        const name = e.key_ptr.*;
        if (std.mem.indexOf(u8, name, ".switch_mlp.") != null or std.mem.endsWith(u8, name, ".weight_scale")) continue;
        if (std.mem.endsWith(u8, name, ".scales") or std.mem.endsWith(u8, name, ".biases")) continue;
        extra += 1;
    }
    try t.expectEqual(planned, extra);
}

test "pack: a shard of more than 2 GiB is written whole" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var b = ShardBuilder.init(t.allocator);
    defer b.deinit();
    const big = try t.allocator.alloc(u8, (2 << 30) + 4096);
    defer t.allocator.free(big);
    @memset(big, 7);
    try b.add("big", "U8", &.{big.len}, big);
    const payload = b.size();
    try b.flush(io, tmp.dir, "big.safetensors", &.{});
    var f = try tmp.dir.openFile(io, "big.safetensors", .{});
    defer f.close(io);
    const stat = try f.stat(io);
    try t.expect(stat.size > payload);
    var tail: [1]u8 = undefined;
    try t.expectEqual(@as(usize, 1), try f.readPositionalAll(io, &tail, stat.size - 1));
    try t.expectEqual(@as(u8, 7), tail[0]);
}
