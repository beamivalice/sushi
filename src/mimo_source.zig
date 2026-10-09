//! Direct loader for the raw MiMo-V2.6-Flash HF text trunk.
//!
//! The source checkpoint is not an MLX checkpoint: routed experts remain
//! individual native tensors for the expert-store adapter, while the FP8 trunk
//! projections stay their source bytes (e4m3 codes + f32 tile scales) for
//! `fp8_block`. This loader is the KLD teacher path, so the trunk carries no
//! quantization step of its own.

const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const expert_exl3 = @import("sushi_exl3").format;
const expert_quant = @import("expert_quant.zig");
const expert_io = @import("expert_io.zig");
const fp8_block = @import("fp8_block.zig");

const Allocator = std.mem.Allocator;

const MAX_HEADER_BYTES: u64 = 512 * 1024 * 1024;
const MAX_JSON_BYTES: usize = 512 * 1024 * 1024;
const FP8_BLOCK: u64 = 128;

const DType = enum {
    bf16,
    f16,
    f32,
    u16,
    u32,
    fp8_e4m3,
    unknown,
};

const TensorMeta = struct {
    dtype: DType,
    shape: []const u64,
    data_start: u64,
    data_end: u64,
    data_base: u64,
    file: []const u8,
};

const SourceIndex = struct {
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
const ShardStamp = struct {
    k: ?[]const u8 = null,
    codebook: ?[]const u8 = null,
    window: ?[]const u8 = null,
};

const TensorKind = enum {
    resident,
    /// A stacked EXL3 routed-expert bank: resident bytes the kernels read as
    /// they are.
    routed_expert,
    fp8_weight,
    fp8_scale,
    skipped,
};

const QkvGeometry = struct {
    q_rows: u64,
    k_rows: u64,
    v_rows: u64,

    fn total(self: QkvGeometry) u64 {
        return self.q_rows + self.k_rows + self.v_rows;
    }
};

pub fn loadWeights(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    config: *const model.ModelConfig,
) !model.Weights {
    try validateConfig(config);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();
    var source = try loadSourceIndex(io, scratch, model_dir);
    try validatePlan(&source, scratch, config);

    var weights = model.Weights.init(allocator);
    errdefer weights.deinit();

    var reader = expert_io.ParallelReader.init(allocator, expert_io.ParallelReader.default_chunk, expert_io.ParallelReader.default_workers);
    defer reader.deinit();
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const meta = entry.value_ptr.*;
        switch (try classifyKey(key, config)) {
            .skipped, .fp8_scale => {},
            .resident, .routed_expert => |kind| {
                if (kind == .routed_expert and config.expert_streaming) continue;
                var arr = (try readDense(&reader, model_dir, meta, null)).array;
                errdefer _ = mlx.mlx_array_free(arr);
                try putWeight(&weights, allocator, key, arr);
                arr = .{};
            },
            .fp8_weight => try loadFp8Weight(&reader, &weights, allocator, model_dir, key, meta, &source),
        }
    }

    if (weights.count() == 0) return error.NoResidentMimoWeights;
    return weights;
}

/// The checkpoint's MTP heads (`model.mtp.layers.*`) as stored: FP8 projections
/// keep their codes under `.weight` and tile scales under `.scales`, the rest is
/// uploaded as is. Empty when the checkpoint carries none.
pub fn loadMtpWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !model.Weights {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var source = try loadSourceIndex(io, arena.allocator(), model_dir);
    var weights = model.Weights.init(allocator);
    errdefer weights.deinit();
    var reader = expert_io.ParallelReader.init(allocator, expert_io.ParallelReader.default_chunk, expert_io.ParallelReader.default_workers);
    defer reader.deinit();
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const meta = entry.value_ptr.*;
        if (!isMtpKey(key) or std.mem.endsWith(u8, key, ".weight_scale_inv")) continue;
        if (meta.dtype == .fp8_e4m3) {
            try loadFp8Weight(&reader, &weights, allocator, model_dir, key, meta, &source);
            continue;
        }
        const arr = (try readDense(&reader, model_dir, meta, null)).array;
        errdefer _ = mlx.mlx_array_free(arr);
        try putWeight(&weights, allocator, key, arr);
    }
    return weights;
}

/// Resident bytes `loadMtpWeights` uploads.
pub fn mtpResidentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !u64 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const source = try loadSourceIndex(io, arena.allocator(), model_dir);
    var total: u64 = 0;
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        if (isMtpKey(entry.key_ptr.*)) total += try payloadBytes(entry.value_ptr.*, null);
    }
    return total;
}

/// Uploads the checkpoint's vision tower (`visual.*`) as stored into `weights`.
/// The rank-5 Conv3d patch weight lands as the [out, C*T*P*P] Linear it is over
/// a flattened patch.
pub fn loadVisionWeightsInto(weights: *model.Weights, io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var source = try loadSourceIndex(io, arena.allocator(), model_dir);
    var reader = expert_io.ParallelReader.init(allocator, expert_io.ParallelReader.default_chunk, expert_io.ParallelReader.default_workers);
    defer reader.deinit();
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (!isVisionKey(key)) continue;
        var meta = entry.value_ptr.*;
        var flat: [2]u64 = undefined;
        if (meta.shape.len > 4) {
            flat = .{ meta.shape[0], try shapeProduct(meta.shape[1..]) };
            meta.shape = &flat;
        }
        const arr = (try readDense(&reader, model_dir, meta, null)).array;
        errdefer _ = mlx.mlx_array_free(arr);
        try putWeight(weights, allocator, key, arr);
    }
}

/// Resident bytes `loadVisionWeightsInto` uploads.
pub fn visionResidentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !u64 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const source = try loadSourceIndex(io, arena.allocator(), model_dir);
    var total: u64 = 0;
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        if (isVisionKey(entry.key_ptr.*)) total += try payloadBytes(entry.value_ptr.*, null);
    }
    return total;
}

fn isVisionKey(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "visual.");
}

fn isMtpKey(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "model.mtp.layers.");
}

/// Exact resident byte count of the raw source trunk as served.
pub fn residentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !u64 {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir))
        return error.InvalidMimoModelPath;
    var config = try model.parseConfig(io, allocator, model_dir);
    defer config.deinit(allocator);
    return residentBytesWithConfig(io, allocator, model_dir, &config);
}

/// Variant for callers that already parsed the source config. The returned
/// value counts only arrays emitted into `model.Weights`; expert banks, MTP,
/// and media are not resident trunk bytes; FP8 scale grids are.
pub fn residentBytesWithConfig(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    config: *const model.ModelConfig,
) !u64 {
    try validateConfig(config);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();
    var source = try loadSourceIndex(io, scratch, model_dir);
    try validatePlan(&source, scratch, config);
    return countResidentBytes(&source, scratch, config);
}

fn validateConfig(config: *const model.ModelConfig) !void {
    if (config.num_hidden_layers == 0 or config.num_hidden_layers > 128 or
        config.hidden_size == 0 or config.vocab_size == 0 or
        config.num_attention_heads == 0 or config.num_key_value_heads == 0 or
        config.head_dim == 0 or config.v_head_dim == 0 or
        config.intermediate_size == 0)
        return error.InvalidMimoGeometry;
    if (config.num_hidden_layers > 1 and config.num_experts == 0)
        return error.InvalidMimoGeometry;

    for (0..config.num_hidden_layers) |i| {
        const layer: u32 = @intCast(i);
        const heads = config.layerNumHeads(layer);
        const kv = config.layerKVHeads(layer);
        const hd = config.layerHeadDim(layer);
        const vd = config.layerVHeadDim(layer);
        if (heads == 0 or kv == 0 or hd == 0 or vd == 0 or heads % kv != 0)
            return error.InvalidMimoGeometry;
        if (@as(u64, heads) * hd > std.math.maxInt(u32) or
            @as(u64, kv) * hd > std.math.maxInt(u32) or
            @as(u64, kv) * vd > std.math.maxInt(u32))
            return error.InvalidMimoGeometry;
    }
}

fn readFileAlloc(
    io: std.Io,
    allocator: Allocator,
    path: []const u8,
    limit: usize,
) ![]u8 {
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch
        return error.MimoSourceFileMissing;
    defer file.close(io);
    var read_buf: [8192]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    return reader.interface.allocRemaining(allocator, .limited(limit)) catch
        return error.MimoSourceRead;
}

fn preadExact(fd: std.c.fd_t, dst: []u8, offset: u64) !void {
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
            return error.MimoSourceRead;
        }
        if (got == 0) return error.SafetensorsUnexpectedEof;
        done += @intCast(got);
    }
}

fn shardPath(allocator: Allocator, model_dir: []const u8, file: []const u8) ![:0]u8 {
    return std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file }, 0);
}

const HeaderBytes = struct {
    bytes: []u8,
    data_base: u64,
};

fn readShardHeader(
    allocator: Allocator,
    model_dir: []const u8,
    file: []const u8,
) !HeaderBytes {
    const path = try shardPath(allocator, model_dir, file);
    defer allocator.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.MissingMimoShard;
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

fn parseDType(name: []const u8) DType {
    if (std.mem.eql(u8, name, "BF16")) return .bf16;
    if (std.mem.eql(u8, name, "F16")) return .f16;
    if (std.mem.eql(u8, name, "F32")) return .f32;
    if (std.mem.eql(u8, name, "U16")) return .u16;
    if (std.mem.eql(u8, name, "U32")) return .u32;
    if (std.mem.eql(u8, name, "F8_E4M3") or std.mem.eql(u8, name, "F8_E4M3FN"))
        return .fp8_e4m3;
    return .unknown;
}

fn parseShape(allocator: Allocator, value: std.json.Value) ![]u64 {
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

fn parseOffsets(value: std.json.Value) ![2]u64 {
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

fn parseTensorMeta(
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

fn loadSourceIndex(io: std.Io, allocator: Allocator, model_dir: []const u8) !SourceIndex {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir))
        return error.InvalidMimoModelPath;

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

fn readShardStamp(allocator: Allocator, header: std.json.ObjectMap) !ShardStamp {
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

fn validateShardStamps(source: *const SourceIndex, config: *const model.ModelConfig) !void {
    if (config.expert_layout != .exl3_k4) return;
    var it = source.stamps.valueIterator();
    while (it.next()) |stamp| {
        if (stamp.codebook) |name| {
            const cb = expert_exl3.Codebook.fromName(name) orelse return error.Exl3ShardStampMismatch;
            if (cb != config.expert_quant_codebook) return error.Exl3ShardStampMismatch;
        }
        if (stamp.window) |bits| {
            const parsed = std.fmt.parseInt(i64, bits, 10) catch return error.Exl3ShardStampMismatch;
            const w = expert_exl3.Window.fromBits(parsed) orelse return error.Exl3ShardStampMismatch;
            if (w != config.expert_quant_window) return error.Exl3ShardStampMismatch;
        }
        if (stamp.k) |text| {
            // The stamp spells a rate ("2.5", "4"); the engine keys on the
            // halfwords per tile it implies. The relation is the trellis
            // shape's (`exl3TrellisAdmitted`): the config names the widest rate
            // a layer packs and bills it, so a shard at or below it is
            // over-billed rather than wrong, and only a wider one refuses.
            const k = std.fmt.parseFloat(f64, text) catch return error.Exl3ShardStampMismatch;
            if (!std.math.isFinite(k)) return error.Exl3ShardStampMismatch;
            const scaled = @round(k * 16.0);
            if (@abs(k * 16.0 - scaled) > 1e-6) return error.Exl3ShardStampMismatch;
            if (scaled < expert_exl3.Rate.min_n or scaled > expert_exl3.Rate.max_n) return error.Exl3ShardStampMismatch;
            const rate = expert_exl3.kFromPackedDim(@intFromFloat(scaled)) orelse return error.Exl3ShardStampMismatch;
            if (rate.n > config.expert_quant_rate.n)
                return error.Exl3ShardStampMismatch;
        }
    }
}

fn layerKey(key: []const u8) ?struct { layer: u32, rest: []const u8 } {
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, key, prefix)) return null;
    const after = key[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, after, '.') orelse return null;
    if (dot == 0) return null;
    const layer = std.fmt.parseInt(u32, after[0..dot], 10) catch return null;
    return .{ .layer = layer, .rest = after[dot + 1 ..] };
}

fn classifyKey(key: []const u8, config: *const model.ModelConfig) !TensorKind {
    if (std.mem.startsWith(u8, key, "visual.") or
        std.mem.startsWith(u8, key, "audio_encoder.") or
        std.mem.startsWith(u8, key, "speech_embeddings.") or
        std.mem.startsWith(u8, key, "model.mtp.") or
        std.mem.startsWith(u8, key, "mtp."))
        return .skipped;
    if (std.mem.startsWith(u8, key, "model.layers.") and
        std.mem.indexOf(u8, key, ".mlp.experts.") != null)
        return .skipped;
    if (std.mem.startsWith(u8, key, "model.layers.") and
        std.mem.indexOf(u8, key, ".mlp.switch_mlp.") != null)
        return if (config.expert_layout == .exl3_k4)
            .routed_expert
        else
            error.UnclassifiedMimoTensor;

    if (std.mem.eql(u8, key, "model.embed_tokens.weight") or
        std.mem.eql(u8, key, "lm_head.weight") or
        std.mem.eql(u8, key, "model.norm.weight") or
        affineGridOf(key, "model.embed_tokens") or affineGridOf(key, "lm_head"))
        return .resident;

    const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
    if (ref.layer >= config.num_hidden_layers) return error.MimoLayerOutOfRange;
    // The dense prefix is `first_k_dense_replace` layers, as the parser and the
    // transformer read it.
    if (std.mem.eql(u8, ref.rest, "self_attn.qkv_proj.weight") or
        (ref.layer < config.first_k_dense_replace and
            (std.mem.eql(u8, ref.rest, "mlp.gate_proj.weight") or
                std.mem.eql(u8, ref.rest, "mlp.up_proj.weight") or
                std.mem.eql(u8, ref.rest, "mlp.down_proj.weight"))))
        return .fp8_weight;
    if (std.mem.eql(u8, ref.rest, "self_attn.qkv_proj.weight_scale_inv") or
        (ref.layer < config.first_k_dense_replace and
            (std.mem.eql(u8, ref.rest, "mlp.gate_proj.weight_scale_inv") or
                std.mem.eql(u8, ref.rest, "mlp.up_proj.weight_scale_inv") or
                std.mem.eql(u8, ref.rest, "mlp.down_proj.weight_scale_inv"))))
        return .fp8_scale;

    if (std.mem.eql(u8, ref.rest, "input_layernorm.weight") or
        std.mem.eql(u8, ref.rest, "post_attention_layernorm.weight") or
        std.mem.eql(u8, ref.rest, "self_attn.o_proj.weight") or
        affineGridOf(ref.rest, "self_attn.o_proj") or
        std.mem.eql(u8, ref.rest, "self_attn.attention_sink_bias") or
        std.mem.eql(u8, ref.rest, "mlp.gate.weight") or
        std.mem.eql(u8, ref.rest, "mlp.gate.e_score_correction_bias"))
        return .resident;

    return error.UnclassifiedMimoTensor;
}

fn qkvGeometry(config: *const model.ModelConfig, layer: u32) QkvGeometry {
    const heads = config.layerNumHeads(layer);
    const kv = config.layerKVHeads(layer);
    const hd = config.layerHeadDim(layer);
    const vd = config.layerVHeadDim(layer);
    return .{
        .q_rows = @as(u64, heads) * hd,
        .k_rows = @as(u64, kv) * hd,
        .v_rows = @as(u64, kv) * vd,
    };
}

fn shapeProduct(shape: []const u64) !u64 {
    var product: u64 = 1;
    for (shape) |dim| {
        product = std.math.mul(u64, product, dim) catch
            return error.SafetensorsShapeOverflow;
    }
    return product;
}

fn payloadBytes(meta: TensorMeta, expected_dtype: ?DType) !u64 {
    if (expected_dtype) |want| if (meta.dtype != want) return error.MimoTensorDtypeMismatch;
    const elem_size: u64 = switch (meta.dtype) {
        .bf16, .f16, .u16 => 2,
        .f32, .u32 => 4,
        .fp8_e4m3 => 1,
        .unknown => return error.UnsupportedMimoTensorDtype,
    };
    const want = std.math.mul(u64, try shapeProduct(meta.shape), elem_size) catch
        return error.SafetensorsShapeOverflow;
    if (meta.data_end - meta.data_start != want) return error.SafetensorsPayloadMismatch;
    return want;
}

fn expectShape(meta: TensorMeta, expected: []const u64) !void {
    if (meta.shape.len != expected.len) return error.MimoTensorShapeMismatch;
    for (meta.shape, expected) |got, want| {
        if (got != want) return error.MimoTensorShapeMismatch;
    }
}

fn qkvSplit(geometry: QkvGeometry, scale_rows: u64) !fp8_block.RowSplit {
    return fp8_block.RowSplit.qkv(geometry.q_rows, geometry.k_rows, geometry.v_rows, scale_rows);
}

fn isQkvWeightKey(key: []const u8) bool {
    return std.mem.endsWith(u8, key, ".self_attn.qkv_proj.weight");
}

fn fp8Base(key: []const u8) []const u8 {
    return key[0 .. key.len - ".weight".len];
}

fn scaleKey(allocator: Allocator, key: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}.weight_scale_inv", .{fp8Base(key)});
}

fn validateFp8Pair(
    source: *const SourceIndex,
    allocator: Allocator,
    config: *const model.ModelConfig,
    key: []const u8,
    meta: TensorMeta,
) !void {
    if (meta.dtype != .fp8_e4m3) return error.MimoTensorDtypeMismatch;
    if (meta.shape.len != 2) return error.MimoTensorShapeMismatch;
    const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
    const s_key = try scaleKey(allocator, key);
    const scale_meta = source.tensors.get(s_key) orelse return error.MissingFp8Scale;
    if (scale_meta.dtype != .f32 or scale_meta.shape.len != 2)
        return error.InvalidFp8ScaleShape;

    const rows = meta.shape[0];
    const cols = meta.shape[1];
    if (cols == 0 or cols % FP8_BLOCK != 0)
        return error.InvalidFp8Shape;
    if (isQkvWeightKey(key)) {
        if (cols != config.hidden_size) return error.InvalidFp8Shape;
        const geometry = qkvGeometry(config, ref.layer);
        if (rows != geometry.total()) return error.InvalidQkvGeometry;
        if (scale_meta.shape[1] != cols / FP8_BLOCK)
            return error.InvalidFp8ScaleShape;
        _ = try qkvSplit(geometry, scale_meta.shape[0]);
    } else {
        var expected_rows = rows;
        var expected_cols = cols;
        if (ref.layer >= config.first_k_dense_replace) return error.UnclassifiedMimoTensor;
        if (std.mem.eql(u8, ref.rest, "mlp.gate_proj.weight") or
            std.mem.eql(u8, ref.rest, "mlp.up_proj.weight"))
        {
            expected_rows = config.intermediate_size;
            expected_cols = config.hidden_size;
        } else if (std.mem.eql(u8, ref.rest, "mlp.down_proj.weight")) {
            expected_rows = config.hidden_size;
            expected_cols = config.intermediate_size;
        } else {
            return error.UnclassifiedMimoTensor;
        }
        if (rows != expected_rows or cols != expected_cols)
            return error.InvalidFp8Shape;
        const required_scale_rows = (rows + FP8_BLOCK - 1) / FP8_BLOCK;
        if (scale_meta.shape[0] < required_scale_rows or
            scale_meta.shape[1] != cols / FP8_BLOCK)
            return error.InvalidFp8ScaleShape;
    }
    _ = try payloadBytes(meta, .fp8_e4m3);
    _ = try payloadBytes(scale_meta, .f32);
}

fn denseExpectedShape(
    key: []const u8,
    config: *const model.ModelConfig,
) !struct { shape: [2]u64, len: usize, dtype: DType } {
    if (std.mem.eql(u8, key, "model.embed_tokens.weight") or
        std.mem.eql(u8, key, "lm_head.weight"))
        return .{ .shape = .{ config.vocab_size, config.hidden_size }, .len = 2, .dtype = .bf16 };
    if (std.mem.eql(u8, key, "model.norm.weight"))
        return .{ .shape = .{ config.hidden_size, 0 }, .len = 1, .dtype = .bf16 };

    const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
    const layer = ref.layer;
    if (std.mem.eql(u8, ref.rest, "input_layernorm.weight") or
        std.mem.eql(u8, ref.rest, "post_attention_layernorm.weight"))
        return .{ .shape = .{ config.hidden_size, 0 }, .len = 1, .dtype = .bf16 };
    if (std.mem.eql(u8, ref.rest, "self_attn.o_proj.weight")) {
        return .{
            .shape = .{
                config.hidden_size,
                @as(u64, config.layerNumHeads(layer)) * config.layerVHeadDim(layer),
            },
            .len = 2,
            .dtype = .bf16,
        };
    }
    if (std.mem.eql(u8, ref.rest, "self_attn.attention_sink_bias"))
        return .{ .shape = .{ config.layerNumHeads(layer), 0 }, .len = 1, .dtype = .bf16 };
    if (std.mem.eql(u8, ref.rest, "mlp.gate.weight"))
        return .{ .shape = .{ config.num_experts, config.hidden_size }, .len = 2, .dtype = .bf16 };
    if (std.mem.eql(u8, ref.rest, "mlp.gate.e_score_correction_bias"))
        return .{ .shape = .{ config.num_experts, 0 }, .len = 1, .dtype = .f32 };
    return error.UnclassifiedMimoTensor;
}

/// Stacked bank geometry: `[E, in/16, out/16, n]` trellis with `suh`/`svh` the
/// two axis scales. gate and up project hidden->inter, down the other way.
fn validateExl3Expert(key: []const u8, meta: TensorMeta, config: *const model.ModelConfig) !void {
    const ref = layerKey(key) orelse return error.UnclassifiedMimoTensor;
    if (ref.layer >= config.num_hidden_layers or ref.layer < config.first_k_dense_replace)
        return error.MimoLayerOutOfRange;
    const bank_prefix = "mlp.switch_mlp.";
    if (!std.mem.startsWith(u8, ref.rest, bank_prefix)) return error.UnclassifiedMimoTensor;
    const rest = ref.rest[bank_prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.UnclassifiedMimoTensor;
    const proj = rest[0..dot];
    const named = try @import("sushi_exl3").group_layout.Name.parse(rest);
    const part = ([_][]const u8{ "trellis", "suh", "svh" })[named.part];
    const down = std.mem.eql(u8, proj, "down_proj");
    if (!down and !std.mem.eql(u8, proj, "gate_proj") and !std.mem.eql(u8, proj, "up_proj"))
        return error.UnclassifiedMimoTensor;
    const hidden: u64 = config.hidden_size;
    const inter: u64 = config.moe_intermediate_size;
    const in_dim: u64 = if (down) inter else hidden;
    const out_dim: u64 = if (down) hidden else inter;
    if (meta.shape.len == 0 or meta.shape[0] == 0) return error.MimoTensorShapeMismatch;
    const experts = meta.shape[0];
    if (std.mem.eql(u8, part, "trellis")) {
        if (meta.dtype != .u16) return error.MimoTensorDtypeMismatch;
        if (meta.shape.len != 4) return error.MimoTensorShapeMismatch;
        if (in_dim % 128 != 0 or out_dim % 128 != 0) return error.MimoTensorShapeMismatch;
        if (expert_exl3.kFromPackedDim(meta.shape[3]) == null) return error.Exl3TrellisGeometry;
        try expectShape(meta, &[_]u64{ experts, in_dim / 16, out_dim / 16, meta.shape[3] });
    } else if (std.mem.eql(u8, part, "suh")) {
        if (meta.dtype != .f16) return error.MimoTensorDtypeMismatch;
        try expectShape(meta, &[_]u64{ experts, in_dim });
    } else if (std.mem.eql(u8, part, "svh")) {
        if (meta.dtype != .f16) return error.MimoTensorDtypeMismatch;
        try expectShape(meta, &[_]u64{ experts, out_dim });
    } else return error.UnclassifiedMimoTensor;
    _ = try payloadBytes(meta, meta.dtype);
}

/// o_proj, lm_head and embed_tokens may be STORED affine (docs/pack-format.md):
/// U32 `<base>.weight` codes beside bf16 `<base>.scales` and `<base>.biases`.
fn affineGridOf(key: []const u8, base: []const u8) bool {
    if (!std.mem.startsWith(u8, key, base)) return false;
    const part = key[base.len..];
    return std.mem.eql(u8, part, ".scales") or std.mem.eql(u8, part, ".biases");
}

/// The linear `key` is a part of when that linear may be stored affine.
fn affineTrunkBase(key: []const u8) ?[]const u8 {
    const dot = std.mem.lastIndexOfScalar(u8, key, '.') orelse return null;
    const base = key[0..dot];
    if (!std.mem.eql(u8, key[dot..], ".weight") and !affineGridOf(key, base)) return null;
    if (std.mem.eql(u8, base, "lm_head") or std.mem.eql(u8, base, "model.embed_tokens")) return base;
    const ref = layerKey(base) orelse return null;
    return if (std.mem.eql(u8, ref.rest, "self_attn.o_proj")) base else null;
}

fn validateResident(source: *const SourceIndex, allocator: Allocator, key: []const u8, meta: TensorMeta, config: *const model.ModelConfig) !void {
    if (config.expert_layout == .exl3_k4) {
        if (layerKey(key)) |ref| {
            if (std.mem.eql(u8, ref.rest, "mlp.gate.weight") or std.mem.eql(u8, ref.rest, "mlp.gate.e_score_correction_bias")) {
                const router_key = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp.gate.weight", .{ref.layer});
                const router = source.tensors.get(router_key) orelse return error.MissingMimoRequiredTensor;
                if (router.shape.len != 2 or router.shape[0] == 0 or router.shape[0] > std.math.maxInt(u32)) return error.Exl3RouterWidthMismatch;
                var local = config.*;
                local.num_experts = @intCast(router.shape[0]);
                return validateDense(key, meta, &local);
            }
        }
    }
    const base = affineTrunkBase(key) orelse return validateDense(key, meta, config);
    if (std.mem.endsWith(u8, key, ".weight") and meta.dtype != .u32) return validateDense(key, meta, config);
    const w_key = try std.fmt.allocPrint(allocator, "{s}.weight", .{base});
    const w = source.tensors.get(w_key) orelse return error.AffineTrunkIncomplete;
    const s = source.tensors.get(try std.fmt.allocPrint(allocator, "{s}.scales", .{base})) orelse return error.AffineTrunkIncomplete;
    const b = source.tensors.get(try std.fmt.allocPrint(allocator, "{s}.biases", .{base})) orelse return error.AffineTrunkIncomplete;
    if (w.dtype != .u32) return error.AffineTrunkIncomplete;
    const dense = try denseExpectedShape(w_key, config);
    if (w.shape.len != 2 or s.shape.len != 2 or w.shape[0] != dense.shape[0]) return error.MimoTensorShapeMismatch;
    try expectShape(s, &[_]u64{ dense.shape[0], s.shape[1] });
    try expectShape(b, s.shape);
    if (expert_quant.affineGeomFromShapes(w.shape[1], s.shape[1], dense.shape[1]) == null)
        return error.MimoTensorShapeMismatch;
    _ = try payloadBytes(w, .u32);
    _ = try payloadBytes(s, .bf16);
    _ = try payloadBytes(b, .bf16);
}

fn validateDense(key: []const u8, meta: TensorMeta, config: *const model.ModelConfig) !void {
    const expected = try denseExpectedShape(key, config);
    if (meta.dtype != expected.dtype) return error.MimoTensorDtypeMismatch;
    const expected_slice = expected.shape[0..expected.len];
    try expectShape(meta, expected_slice);
    _ = try payloadBytes(meta, expected.dtype);
}

fn requireKind(
    source: *const SourceIndex,
    allocator: Allocator,
    config: *const model.ModelConfig,
    key: []const u8,
    expected: TensorKind,
) !void {
    const meta = source.tensors.get(key) orelse return error.MissingMimoRequiredTensor;
    if (try classifyKey(key, config) != expected)
        return error.MimoRequiredTensorKindMismatch;
    switch (expected) {
        .resident => try validateResident(source, allocator, key, meta, config),
        .routed_expert => try validateExl3Expert(key, meta, config),
        .fp8_weight => try validateFp8Pair(source, allocator, config, key, meta),
        .fp8_scale, .skipped => {},
    }
}

fn validateRequired(
    source: *const SourceIndex,
    allocator: Allocator,
    config: *const model.ModelConfig,
) !void {
    try requireKind(source, allocator, config, "model.embed_tokens.weight", .resident);
    try requireKind(source, allocator, config, "lm_head.weight", .resident);
    try requireKind(source, allocator, config, "model.norm.weight", .resident);

    for (0..config.num_hidden_layers) |i| {
        const layer: u32 = @intCast(i);
        const layer_prefix = try std.fmt.allocPrint(allocator, "model.layers.{d}", .{layer});
        try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.input_layernorm.weight", .{layer_prefix}), .resident);
        try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.post_attention_layernorm.weight", .{layer_prefix}), .resident);
        try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.self_attn.qkv_proj.weight", .{layer_prefix}), .fp8_weight);
        try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.self_attn.o_proj.weight", .{layer_prefix}), .resident);
        if (config.layerHasAttnSinks(layer)) {
            try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.self_attn.attention_sink_bias", .{layer_prefix}), .resident);
        }
        if (layer < config.first_k_dense_replace) {
            for ([_][]const u8{ "gate", "up", "down" }) |projection| {
                const k = try std.fmt.allocPrint(
                    allocator,
                    "{s}.mlp.{s}_proj.weight",
                    .{ layer_prefix, projection },
                );
                try requireKind(source, allocator, config, k, .fp8_weight);
            }
        } else {
            try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.mlp.gate.weight", .{layer_prefix}), .resident);
            try requireKind(source, allocator, config, try std.fmt.allocPrint(allocator, "{s}.mlp.gate.e_score_correction_bias", .{layer_prefix}), .resident);
            if (config.expert_layout == .exl3_k4) {
                _ = try validateExl3Layer(source, allocator, config, layer_prefix);
            }
        }
    }
}

fn validateExl3Layer(source: *const SourceIndex, allocator: Allocator, config: *const model.ModelConfig, layer_prefix: []const u8) !@import("sushi_exl3").GroupLayout {
    var plan = @import("sushi_exl3").GroupLayout.init(config.hidden_size, config.moe_intermediate_size, config.expert_quant_rate);
    const prefix = try std.fmt.allocPrint(allocator, "{s}.mlp.switch_mlp.", .{layer_prefix});
    const router = source.tensors.get(try std.fmt.allocPrint(allocator, "{s}.mlp.gate.weight", .{layer_prefix})) orelse return error.MissingMimoRequiredTensor;
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
    return plan;
}

fn validatePlan(source: *const SourceIndex, allocator: Allocator, config: *const model.ModelConfig) !void {
    try validateShardStamps(source, config);
    try validateRequired(source, allocator, config);
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const meta = entry.value_ptr.*;
        switch (try classifyKey(key, config)) {
            .skipped => {},
            .resident => try validateResident(source, allocator, key, meta, config),
            .routed_expert => try validateExl3Expert(key, meta, config),
            .fp8_weight => try validateFp8Pair(source, allocator, config, key, meta),
            .fp8_scale => {
                const suffix = ".weight_scale_inv";
                if (!std.mem.endsWith(u8, key, suffix))
                    return error.UnclassifiedMimoTensor;
                const base = key[0 .. key.len - suffix.len];
                const weight_key = try std.fmt.allocPrint(allocator, "{s}.weight", .{base});
                if (!source.tensors.contains(weight_key))
                    return error.MissingFp8Weight;
            },
        }
    }
}

fn countResidentBytes(
    source: *const SourceIndex,
    allocator: Allocator,
    config: *const model.ModelConfig,
) !u64 {
    var total: u64 = 0;
    var it = source.tensors.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const meta = entry.value_ptr.*;
        switch (try classifyKey(key, config)) {
            .skipped, .fp8_scale => {},
            .resident, .routed_expert => |kind| {
                if (kind == .routed_expert and config.expert_streaming) continue;
                var bytes = try payloadBytes(meta, null);
                // The transformer loader keeps an f32 copy of each router for f32 routing.
                if (layerKey(key)) |ref| if (std.mem.eql(u8, ref.rest, "mlp.gate.weight")) {
                    bytes += try shapeProduct(meta.shape) * 4;
                };
                total = std.math.add(u64, total, bytes) catch
                    return error.ResidentBytesOverflow;
            },
            .fp8_weight => {
                const scale_key = try scaleKey(allocator, key);
                const scale_meta = source.tensors.get(scale_key) orelse
                    return error.MissingFp8Scale;
                const bytes = try payloadBytes(meta, .fp8_e4m3) + try payloadBytes(scale_meta, .f32);
                total = std.math.add(u64, total, bytes) catch
                    return error.ResidentBytesOverflow;
            },
        }
    }
    return total;
}

/// The tensor as an MLX array in its stored dtype (or `as`), read around the file cache
/// (`model.readTensorArray`). Caller frees `.array`; `.bytes` lives as long.
fn readDense(reader: *expert_io.ParallelReader, model_dir: []const u8, meta: TensorMeta, as: ?mlx.mlx_dtype) !model.TensorArray {
    const len_u64 = meta.data_end - meta.data_start;
    const len = std.math.cast(usize, len_u64) orelse return error.SafetensorsShapeOverflow;
    var shape: [4]c_int = undefined;
    const shape_slice = try shapeForUpload(meta, &shape);
    const dtype: mlx.mlx_dtype = as orelse switch (meta.dtype) {
        .bf16 => .bfloat16,
        .f16 => .float16,
        .f32 => .float32,
        .u16 => .uint16,
        .u32 => .uint32,
        else => return error.MimoTensorDtypeMismatch,
    };
    const path = try shardPath(reader.allocator, model_dir, meta.file);
    defer reader.allocator.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.MissingMimoShard;
    defer _ = std.c.close(fd);
    expert_io.applyReadHints(fd, .{ .readahead_off = false });
    const absolute = std.math.add(u64, meta.data_base, meta.data_start) catch
        return error.InvalidSafetensorsHeader;
    return model.readTensorArray(reader, fd, absolute, len, shape_slice, dtype);
}

fn shapeForUpload(meta: TensorMeta, shape: *[4]c_int) ![]const c_int {
    if (meta.shape.len == 0 or meta.shape.len > shape.len)
        return error.MimoTensorShapeMismatch;
    for (meta.shape, 0..) |dim, i| {
        shape[i] = std.math.cast(c_int, dim) orelse return error.MimoTensorShapeMismatch;
    }
    return shape[0..meta.shape.len];
}

/// A NaN code, or a scale whose largest product leaves bf16 (the prefill
/// scratch `fp8_block.dequantize` writes), is a broken checkpoint.
fn validateFp8Payload(codes: []const u8, scales: []const u8) !void {
    for (codes) |code| {
        if (code & 0x7f == 0x7f) return error.InvalidFp8Value;
    }
    if (scales.len % 4 != 0) return error.SafetensorsPayloadMismatch;
    const bf16_max: f32 = @bitCast(@as(u32, 0x7f7f0000));
    for (0..scales.len / 4) |i| {
        const scale: f32 = @bitCast(std.mem.readInt(u32, scales[i * 4 ..][0..4], .little));
        if (!std.math.isFinite(scale) or @abs(scale) * 448.0 > bf16_max) return error.InvalidFp8Scale;
    }
}

fn putWeight(weights: *model.Weights, allocator: Allocator, key: []const u8, arr: mlx.mlx_array) !void {
    if (weights.map.contains(key)) return error.DuplicateOutputWeight;
    const owned_key = try allocator.dupe(u8, key);
    errdefer allocator.free(owned_key);
    try weights.map.put(owned_key, arr);
}

/// The codes and their tile scales as stored, under `{base}.weight` and
/// `{base}.scales`; a QKV keeps its rank-local rows (the forward splits them).
fn loadFp8Weight(
    reader: *expert_io.ParallelReader,
    weights: *model.Weights,
    allocator: Allocator,
    model_dir: []const u8,
    key: []const u8,
    meta: TensorMeta,
    source: *const SourceIndex,
) !void {
    const scale_name = try scaleKey(allocator, key);
    defer allocator.free(scale_name);
    const scale_meta = source.tensors.get(scale_name) orelse return error.MissingFp8Scale;
    const codes = try readDense(reader, model_dir, meta, .uint8);
    var w = codes.array;
    errdefer if (w.ctx != null) {
        _ = mlx.mlx_array_free(w);
    };
    const scales = try readDense(reader, model_dir, scale_meta, null);
    var sc = scales.array;
    errdefer if (sc.ctx != null) {
        _ = mlx.mlx_array_free(sc);
    };
    try validateFp8Payload(codes.bytes, scales.bytes);

    try putWeight(weights, allocator, key, w);
    w = .{};
    const sc_key = try std.fmt.allocPrint(allocator, "{s}.scales", .{fp8Base(key)});
    defer allocator.free(sc_key);
    try putWeight(weights, allocator, sc_key, sc);
    sc = .{};
}

const TestTensor = struct {
    key: []const u8,
    dtype: []const u8,
    shape: []const u64,
    bytes: []const u8,
};

const TestIndexEntry = struct {
    key: []const u8,
    file: []const u8,
};

fn appendTestFormat(
    allocator: Allocator,
    list: *std.ArrayList(u8),
    comptime format: []const u8,
    args: anytype,
) !void {
    const text = try std.fmt.allocPrint(allocator, format, args);
    defer allocator.free(text);
    try list.appendSlice(allocator, text);
}

fn writeTestShard(
    io: std.Io,
    allocator: Allocator,
    dir: std.Io.Dir,
    filename: []const u8,
    tensors: []const TestTensor,
) !void {
    return writeTestShardStamped(io, allocator, dir, filename, tensors, null);
}

fn writeTestShardStamped(
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

fn writeTestIndex(
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

fn testBf16Bytes(allocator: Allocator, count: usize, bits: u16) ![]u8 {
    const bytes = try allocator.alloc(u8, count * 2);
    for (0..count) |i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], bits, .little);
    return bytes;
}

fn testFp8Bytes(allocator: Allocator, count: usize, code: u8) ![]u8 {
    const bytes = try allocator.alloc(u8, count);
    @memset(bytes, code);
    return bytes;
}

fn testF32Bytes(allocator: Allocator, values: []const f32) ![]u8 {
    const bytes = try allocator.alloc(u8, values.len * 4);
    for (values, 0..) |value, i| {
        std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
    }
    return bytes;
}

pub const TinySourceFixture = struct {
    allocator: Allocator,
    path: []u8,
    config: model.ModelConfig,
    qkv_weight: []u8,
    qkv_scales: []u8,
    mlp_weight: []u8,
    mlp_scales: []u8,
    embed: []u8,

    pub fn deinit(self: *TinySourceFixture) void {
        self.allocator.free(self.path);
        self.allocator.free(self.qkv_weight);
        self.allocator.free(self.qkv_scales);
        self.allocator.free(self.mlp_weight);
        self.allocator.free(self.mlp_scales);
        self.allocator.free(self.embed);
    }
};

const TINY_SOURCE_INDEX = [_]TestIndexEntry{
    .{ .key = "model.embed_tokens.weight", .file = "model-00001.safetensors" },
    .{ .key = "model.layers.0.self_attn.qkv_proj.weight", .file = "model-00001.safetensors" },
    .{ .key = "lm_head.weight", .file = "model-00002.safetensors" },
    .{ .key = "model.norm.weight", .file = "model-00002.safetensors" },
    .{ .key = "model.layers.0.input_layernorm.weight", .file = "model-00002.safetensors" },
    .{ .key = "model.layers.0.post_attention_layernorm.weight", .file = "model-00002.safetensors" },
    .{ .key = "model.layers.0.self_attn.o_proj.weight", .file = "model-00002.safetensors" },
    .{ .key = "model.layers.0.mlp.gate_proj.weight", .file = "model-00002.safetensors" },
    .{ .key = "model.layers.0.mlp.gate_proj.weight_scale_inv", .file = "model-00002.safetensors" },
    .{ .key = "model.layers.0.mlp.up_proj.weight", .file = "model-00002.safetensors" },
    .{ .key = "model.layers.0.mlp.up_proj.weight_scale_inv", .file = "model-00002.safetensors" },
    .{ .key = "model.layers.0.mlp.down_proj.weight", .file = "model-00002.safetensors" },
    .{ .key = "model.layers.0.mlp.down_proj.weight_scale_inv", .file = "model-00002.safetensors" },
    .{ .key = "model.layers.0.mlp.experts.0.gate_proj.weight", .file = "model-00002.safetensors" },
    .{ .key = "model.mtp.layers.0.fake.weight", .file = "model-00002.safetensors" },
    .{ .key = "visual.fake", .file = "model-00002.safetensors" },
    .{ .key = "model.layers.0.self_attn.qkv_proj.weight_scale_inv", .file = "model-00003.safetensors" },
};

/// A pack's surgery on the tiny source: `tensors` land in a NEW shard and the
/// index points their names at it; the old bf16 bytes stay, unindexed.
fn redirectToNewShard(io: std.Io, allocator: Allocator, tmp: *std.testing.TmpDir, tensors: []const TestTensor) !void {
    var dir = try tmp.dir.openDir(io, "mimo-source", .{});
    defer dir.close(io);
    try writeTestShard(io, allocator, dir, "model-affine.safetensors", tensors);
    var entries: std.ArrayList(TestIndexEntry) = .empty;
    defer entries.deinit(allocator);
    for (TINY_SOURCE_INDEX) |e| {
        const moved = for (tensors) |t| {
            if (std.mem.eql(u8, t.key, e.key)) break true;
        } else false;
        if (!moved) try entries.append(allocator, e);
    }
    for (tensors) |t| try entries.append(allocator, .{ .key = t.key, .file = "model-affine.safetensors" });
    try writeTestIndex(io, allocator, dir, entries.items);
}

pub fn makeTinySourceFixture(
    io: std.Io,
    allocator: Allocator,
    tmp: *std.testing.TmpDir,
) !TinySourceFixture {
    try tmp.dir.createDirPath(io, "mimo-source");
    var dir = try tmp.dir.openDir(io, "mimo-source", .{});
    defer dir.close(io);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try dir.realPath(io, &path_buf);
    const path = try allocator.dupe(u8, path_buf[0..path_len]);
    errdefer allocator.free(path);

    try dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"mimo_v2","vocab_size":2,"hidden_size":128,
        \\ "num_hidden_layers":1,"intermediate_size":128,
        \\ "moe_intermediate_size":128,"n_routed_experts":1,
        \\ "num_experts_per_tok":1,"n_group":1,"topk_group":1,
        \\ "num_attention_heads":8,"num_key_value_heads":8,"head_dim":16,
        \\ "v_head_dim":16,"swa_num_attention_heads":8,
        \\ "swa_num_key_value_heads":8,"swa_head_dim":16,"swa_v_head_dim":16,
        \\ "hybrid_layer_pattern":[1],"moe_layer_freq":[0],
        \\ "attention_projection_layout":"fused_qkv",
        \\ "add_swa_attention_sink_bias":false,
        \\ "add_full_attention_sink_bias":false}
    });

    const embed_shape = [_]u64{ 2, 128 };
    const vector_shape = [_]u64{128};
    const matrix_shape = [_]u64{ 128, 128 };
    const qkv_shape = [_]u64{ 384, 128 };
    const qkv_scale_shape = [_]u64{ 4, 1 };
    const mlp_scale_shape = [_]u64{ 1, 1 };
    const embed = try testBf16Bytes(allocator, 2 * 128, 0x3f80);
    errdefer allocator.free(embed);
    const qkv_weight = try testFp8Bytes(allocator, 384 * 128, 0x38);
    errdefer allocator.free(qkv_weight);
    const qkv_scales = try testF32Bytes(allocator, &.{ 1.0, 2.0, 3.0, 4.0 });
    errdefer allocator.free(qkv_scales);
    const mlp_weight = try testFp8Bytes(allocator, 128 * 128, 0x38);
    errdefer allocator.free(mlp_weight);
    const mlp_scales = try testF32Bytes(allocator, &.{1.0});
    errdefer allocator.free(mlp_scales);
    const norm = try testBf16Bytes(allocator, 128, 0x3f80);
    defer allocator.free(norm);
    const o_proj = try testBf16Bytes(allocator, 128 * 128, 0x3f80);
    defer allocator.free(o_proj);
    const ignored_expert = [_]u8{0};
    const ignored_mtp = [_]u8{ 0, 0 };
    const ignored_media = [_]u8{ 0, 0, 0, 0 };

    const shard_a = [_]TestTensor{
        .{ .key = "model.embed_tokens.weight", .dtype = "BF16", .shape = &embed_shape, .bytes = embed },
        .{ .key = "model.layers.0.self_attn.qkv_proj.weight", .dtype = "F8_E4M3", .shape = &qkv_shape, .bytes = qkv_weight },
    };
    const shard_b = [_]TestTensor{
        .{ .key = "lm_head.weight", .dtype = "BF16", .shape = &embed_shape, .bytes = embed },
        .{ .key = "model.norm.weight", .dtype = "BF16", .shape = &vector_shape, .bytes = norm },
        .{ .key = "model.layers.0.input_layernorm.weight", .dtype = "BF16", .shape = &vector_shape, .bytes = norm },
        .{ .key = "model.layers.0.post_attention_layernorm.weight", .dtype = "BF16", .shape = &vector_shape, .bytes = norm },
        .{ .key = "model.layers.0.self_attn.o_proj.weight", .dtype = "BF16", .shape = &matrix_shape, .bytes = o_proj },
        .{ .key = "model.layers.0.mlp.gate_proj.weight", .dtype = "F8_E4M3", .shape = &matrix_shape, .bytes = mlp_weight },
        .{ .key = "model.layers.0.mlp.gate_proj.weight_scale_inv", .dtype = "F32", .shape = &mlp_scale_shape, .bytes = mlp_scales },
        .{ .key = "model.layers.0.mlp.up_proj.weight", .dtype = "F8_E4M3", .shape = &matrix_shape, .bytes = mlp_weight },
        .{ .key = "model.layers.0.mlp.up_proj.weight_scale_inv", .dtype = "F32", .shape = &mlp_scale_shape, .bytes = mlp_scales },
        .{ .key = "model.layers.0.mlp.down_proj.weight", .dtype = "F8_E4M3", .shape = &matrix_shape, .bytes = mlp_weight },
        .{ .key = "model.layers.0.mlp.down_proj.weight_scale_inv", .dtype = "F32", .shape = &mlp_scale_shape, .bytes = mlp_scales },
        .{ .key = "model.layers.0.mlp.experts.0.gate_proj.weight", .dtype = "U8", .shape = &[_]u64{1}, .bytes = &ignored_expert },
        .{ .key = "model.mtp.layers.0.fake.weight", .dtype = "BF16", .shape = &[_]u64{1}, .bytes = &ignored_mtp },
        .{ .key = "visual.fake", .dtype = "F32", .shape = &[_]u64{1}, .bytes = &ignored_media },
    };
    const shard_c = [_]TestTensor{
        .{ .key = "model.layers.0.self_attn.qkv_proj.weight_scale_inv", .dtype = "F32", .shape = &qkv_scale_shape, .bytes = qkv_scales },
    };
    try writeTestShard(io, allocator, dir, "model-00001.safetensors", &shard_a);
    try writeTestShard(io, allocator, dir, "model-00002.safetensors", &shard_b);
    try writeTestShard(io, allocator, dir, "model-00003.safetensors", &shard_c);

    try writeTestIndex(io, allocator, dir, &TINY_SOURCE_INDEX);

    return .{
        .allocator = allocator,
        .path = path,
        .config = .{
            .model_type = "mimo_v2",
            .vocab_size = 2,
            .hidden_size = 128,
            .intermediate_size = 128,
            .num_hidden_layers = 1,
            .num_attention_heads = 8,
            .num_key_value_heads = 8,
            .head_dim = 16,
            .v_head_dim = 16,
            .num_experts = 1,
            .first_k_dense_replace = 1,
            .has_sliding_window = false,
        },
        .qkv_weight = qkv_weight,
        .qkv_scales = qkv_scales,
        .mlp_weight = mlp_weight,
        .mlp_scales = mlp_scales,
        .embed = embed,
    };
}

/// A two-layer MiMo checkpoint whose layer 1 carries stacked EXL3 banks at the
/// given packed halfword count. `n = 0` writes no banks at all.
/// `stamp` is the shard's `__metadata__` body, as the converter writes it.
const TinyExl3 = struct { n: u64, stamp: ?[]const u8 = null };

const TinyBank = union(enum) {
    none,
    exl3: TinyExl3,
};

fn writeTinyExl3Source(io: std.Io, allocator: Allocator, dir: std.Io.Dir, n: u64) !void {
    return writeTinySource(io, allocator, dir, if (n == 0) .none else .{ .exl3 = .{ .n = n } }, 1);
}

/// `dense` is the size of the dense prefix (`first_k_dense_replace`); exactly
/// one MoE layer follows it, carrying `bank`.
fn writeTinySource(
    io: std.Io,
    allocator: Allocator,
    dir: std.Io.Dir,
    bank: TinyBank,
    dense: u32,
) !void {
    const dim: u64 = 128;
    const tiles = dim / 16;
    const n: u64 = switch (bank) {
        .exl3 => |v| v.n,
        else => 0,
    };
    const stamp: ?[]const u8 = switch (bank) {
        .exl3 => |v| v.stamp,
        else => null,
    };
    const layers: usize = @as(usize, dense) + 1;
    var pattern: [64]u8 = undefined;
    var freq: [64]u8 = undefined;
    var pattern_len: usize = 0;
    var freq_len: usize = 0;
    for (0..layers) |l| {
        pattern_len += (try std.fmt.bufPrint(pattern[pattern_len..], "{s}1", .{if (l == 0) "" else ","})).len;
        freq_len += (try std.fmt.bufPrint(freq[freq_len..], "{s}{d}", .{ if (l == 0) "" else ",", @intFromBool(l >= dense) })).len;
    }
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = try std.fmt.allocPrint(allocator,
        \\{{"model_type":"mimo_v2","vocab_size":2,"hidden_size":128,
        \\ "num_hidden_layers":{d},"intermediate_size":128,
        \\ "moe_intermediate_size":128,"n_routed_experts":2,
        \\ "num_experts_per_tok":1,"n_group":1,"topk_group":1,
        \\ "num_attention_heads":8,"num_key_value_heads":8,"head_dim":16,
        \\ "v_head_dim":16,"swa_num_attention_heads":8,
        \\ "swa_num_key_value_heads":8,"swa_head_dim":16,"swa_v_head_dim":16,
        \\ "hybrid_layer_pattern":[{s}],"moe_layer_freq":[{s}],
        \\ "attention_projection_layout":"fused_qkv",
        \\ "add_swa_attention_sink_bias":false,
        \\ "add_full_attention_sink_bias":false,
        \\ "expert_quant":{{"format":"exl3","k":2.5,"codebook":"mcg"}}}}
    , .{ layers, pattern[0..pattern_len], freq[0..freq_len] }) });
    const embed = try testBf16Bytes(allocator, 2 * 128, 0x3f80);
    defer allocator.free(embed);
    const norm = try testBf16Bytes(allocator, 128, 0x3f80);
    defer allocator.free(norm);
    const matrix = try testBf16Bytes(allocator, 128 * 128, 0x3f80);
    defer allocator.free(matrix);
    const router = try testBf16Bytes(allocator, 2 * 128, 0x3f80);
    defer allocator.free(router);
    const corr = try testF32Bytes(allocator, &.{ 0.0, 0.0 });
    defer allocator.free(corr);
    const qkv_weight = try testFp8Bytes(allocator, 384 * 128, 0x38);
    defer allocator.free(qkv_weight);
    const qkv_scales = try testF32Bytes(allocator, &.{ 1.0, 2.0, 3.0, 4.0 });
    defer allocator.free(qkv_scales);
    const mlp_weight = try testFp8Bytes(allocator, 128 * 128, 0x38);
    defer allocator.free(mlp_weight);
    const mlp_scales = try testF32Bytes(allocator, &.{1.0});
    defer allocator.free(mlp_scales);
    const trellis = try allocator.alloc(u8, @intCast(2 * tiles * tiles * @max(n, 1) * 2));
    defer allocator.free(trellis);
    @memset(trellis, 0x5a);
    const axis = try testBf16Bytes(allocator, 2 * 128, 0x3c00);
    defer allocator.free(axis);

    var tensors: std.ArrayList(TestTensor) = .empty;
    defer tensors.deinit(allocator);
    var entries: std.ArrayList(TestIndexEntry) = .empty;
    defer entries.deinit(allocator);
    const embed_shape = [_]u64{ 2, dim };
    const vector_shape = [_]u64{dim};
    const matrix_shape = [_]u64{ dim, dim };
    const qkv_shape = [_]u64{ 384, dim };
    const qkv_scale_shape = [_]u64{ 4, 1 };
    const mlp_scale_shape = [_]u64{ 1, 1 };
    const corr_shape = [_]u64{2};
    const trellis_shape = [_]u64{ 2, tiles, tiles, n };
    const axis_shape = [_]u64{ 2, dim };
    try tensors.append(allocator, .{ .key = "model.embed_tokens.weight", .dtype = "BF16", .shape = &embed_shape, .bytes = embed });
    try tensors.append(allocator, .{ .key = "lm_head.weight", .dtype = "BF16", .shape = &embed_shape, .bytes = embed });
    try tensors.append(allocator, .{ .key = "model.norm.weight", .dtype = "BF16", .shape = &vector_shape, .bytes = norm });
    for (0..layers) |layer| {
        inline for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight" }) |leaf| {
            try tensors.append(allocator, .{
                .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.{s}", .{ layer, leaf }),
                .dtype = "BF16",
                .shape = &vector_shape,
                .bytes = norm,
            });
        }
        try tensors.append(allocator, .{
            .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn.o_proj.weight", .{layer}),
            .dtype = "BF16",
            .shape = &matrix_shape,
            .bytes = matrix,
        });
        try tensors.append(allocator, .{
            .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn.qkv_proj.weight", .{layer}),
            .dtype = "F8_E4M3",
            .shape = &qkv_shape,
            .bytes = qkv_weight,
        });
        try tensors.append(allocator, .{
            .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn.qkv_proj.weight_scale_inv", .{layer}),
            .dtype = "F32",
            .shape = &qkv_scale_shape,
            .bytes = qkv_scales,
        });
        if (layer < dense) {
            inline for ([_][]const u8{ "gate", "up", "down" }) |proj| {
                try tensors.append(allocator, .{
                    .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp.{s}_proj.weight", .{ layer, proj }),
                    .dtype = "F8_E4M3",
                    .shape = &matrix_shape,
                    .bytes = mlp_weight,
                });
                try tensors.append(allocator, .{
                    .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp.{s}_proj.weight_scale_inv", .{ layer, proj }),
                    .dtype = "F32",
                    .shape = &mlp_scale_shape,
                    .bytes = mlp_scales,
                });
            }
        } else {
            try tensors.append(allocator, .{ .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp.gate.weight", .{layer}), .dtype = "BF16", .shape = &embed_shape, .bytes = router });
            try tensors.append(allocator, .{ .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp.gate.e_score_correction_bias", .{layer}), .dtype = "F32", .shape = &corr_shape, .bytes = corr });
            if (n > 0) {
                inline for ([_][]const u8{ "gate", "up", "down" }) |proj| {
                    try tensors.append(allocator, .{
                        .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp.switch_mlp.{s}_proj.trellis", .{ layer, proj }),
                        .dtype = "U16",
                        .shape = &trellis_shape,
                        .bytes = trellis[0..@intCast(2 * tiles * tiles * n * 2)],
                    });
                    try tensors.append(allocator, .{
                        .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp.switch_mlp.{s}_proj.suh", .{ layer, proj }),
                        .dtype = "F16",
                        .shape = &axis_shape,
                        .bytes = axis,
                    });
                    try tensors.append(allocator, .{
                        .key = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp.switch_mlp.{s}_proj.svh", .{ layer, proj }),
                        .dtype = "F16",
                        .shape = &axis_shape,
                        .bytes = axis,
                    });
                }
            }
        }
    }
    for (tensors.items) |tensor| try entries.append(allocator, .{ .key = tensor.key, .file = "model-00001.safetensors" });
    try writeTestShardStamped(io, allocator, dir, "model-00001.safetensors", tensors.items, stamp);
    try writeTestIndex(io, allocator, dir, entries.items);
}

const expectError = @import("test_expect.zig").expectError;

test "mimo source loads and bills EXL3 routed banks beside the prepared trunk" {
    const t = std.testing;
    const io = t.io;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "mimo-exl3");
    var dir = try tmp.dir.openDir(io, "mimo-exl3", .{});
    defer dir.close(io);
    try writeTinyExl3Source(io, alloc, dir, 40);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try dir.realPath(io, &path_buf);
    const path = path_buf[0..path_len];

    var config = try model.parseConfig(io, t.allocator, path);
    defer config.deinit(t.allocator);
    try t.expectEqual(expert_quant.Layout.exl3_k4, config.expert_layout);
    try t.expectEqual(@as(u32, 40), config.expert_quant_rate.n);
    try t.expect(!config.expertStreamingRequired());
    try t.expect(config.usesMimoSourceTrunk());

    // Nine banks: three projections of [2,8,8,40] u16 plus two [2,128] f16 axes.
    const bank_bytes: u64 = 3 * (2 * 8 * 8 * 40 * 2 + 2 * 2 * 128 * 2);
    const with_banks = try residentBytesWithConfig(io, t.allocator, path, &config);
    try tmp.dir.createDirPath(io, "mimo-trunk-only");
    var bare = try tmp.dir.openDir(io, "mimo-trunk-only", .{});
    defer bare.close(io);
    try writeTinyExl3Source(io, alloc, bare, 0);
    var bare_buf: [std.fs.max_path_bytes]u8 = undefined;
    const bare_len = try bare.realPath(io, &bare_buf);
    var bare_config = try model.parseConfig(io, t.allocator, bare_buf[0..bare_len]);
    defer bare_config.deinit(t.allocator);
    const without = try residentBytesWithConfig(io, t.allocator, bare_buf[0..bare_len], &bare_config);
    try t.expectEqual(bank_bytes, with_banks - without);

    var weights = try loadWeights(io, t.allocator, path, &config);
    defer weights.deinit();
    // The bill is what the loader emits plus the f32 copy of the [2,128] router
    // the transformer keeps for f32 routing.
    var emitted: u64 = 0;
    var it = weights.map.iterator();
    while (it.next()) |e| emitted += mlx.mlx_array_size(e.value_ptr.*) * mlx.mlx_array_itemsize(e.value_ptr.*);
    try t.expectEqual(emitted + 2 * 128 * 4, with_banks);
    const trellis = weights.get("model.layers.1.mlp.switch_mlp.gate_proj.trellis") orelse
        return error.MissingWeight;
    try t.expectEqualSlices(c_int, &[_]c_int{ 2, 8, 8, 40 }, mlx.getShape(trellis));
    try t.expectEqual(mlx.mlx_dtype.uint16, mlx.mlx_array_dtype(trellis));
    const suh = weights.get("model.layers.1.mlp.switch_mlp.down_proj.suh") orelse
        return error.MissingWeight;
    try t.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(suh));
}

test "mimo source refuses an EXL3 trellis the kernels cannot decode" {
    const t = std.testing;
    const io = t.io;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "mimo-exl3-bad");
    var dir = try tmp.dir.openDir(io, "mimo-exl3-bad", .{});
    defer dir.close(io);
    try writeTinyExl3Source(io, alloc, dir, 41);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try dir.realPath(io, &path_buf);
    const path = path_buf[0..path_len];
    var config = try model.parseConfig(io, t.allocator, path);
    defer config.deinit(t.allocator);
    config.expert_layout = .exl3_k4;
    try t.expectError(error.Exl3TrellisGeometry, residentBytesWithConfig(io, t.allocator, path, &config));
}

test "mimo source refuses an EXL3 shard whose stamp disagrees with the config" {
    const t = std.testing;
    const io = t.io;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();

    // The config every arm parses names k 2.5 / mcg / w16.
    const Case = struct { dir: []const u8, stamp: ?[]const u8, window: expert_exl3.Window, rate: ?expert_exl3.Rate = null, want: ?anyerror };
    const cases = [_]Case{
        .{ .dir = "stamp-matches", .stamp = "\"k\":\"2.5\",\"codebook\":\"mcg\",\"window\":\"16\"", .window = .w16, .want = null },
        .{ .dir = "stamp-window", .stamp = "\"k\":\"2.5\",\"codebook\":\"mcg\",\"window\":\"16\"", .window = .w12, .want = error.Exl3ShardStampMismatch },
        .{ .dir = "stamp-codebook", .stamp = "\"k\":\"2.5\",\"codebook\":\"mul1\",\"window\":\"16\"", .window = .w16, .want = error.Exl3ShardStampMismatch },
        .{ .dir = "stamp-unknown", .stamp = "\"k\":\"2.5\",\"codebook\":\"mul2\",\"window\":\"16\"", .window = .w16, .want = error.Exl3ShardStampMismatch },
        // The bill prices the config's rate, so a shard WIDER than it refuses.
        .{ .dir = "stamp-k", .stamp = "\"k\":\"3\",\"codebook\":\"mcg\",\"window\":\"16\"", .window = .w16, .want = error.Exl3ShardStampMismatch },
        // A tail-bumped pack: the config names the widest rate and the body
        // layers stamp below it. Over-billed, not wrong — it loads.
        .{ .dir = "stamp-k-below", .stamp = "\"k\":\"2.5\",\"codebook\":\"mcg\",\"window\":\"16\"", .window = .w16, .rate = .{ .n = 64 }, .want = null },
        // A pack written before the stamp existed is admitted as legacy.
        .{ .dir = "stamp-absent", .stamp = null, .window = .w12, .want = null },
    };
    for (cases) |case| {
        try tmp.dir.createDirPath(io, case.dir);
        var dir = try tmp.dir.openDir(io, case.dir, .{});
        defer dir.close(io);
        try writeTinySource(io, alloc, dir, .{ .exl3 = .{ .n = 40, .stamp = case.stamp } }, 1);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try dir.realPath(io, &path_buf);
        const path = path_buf[0..path_len];
        var config = try model.parseConfig(io, t.allocator, path);
        defer config.deinit(t.allocator);
        config.expert_quant_window = case.window;
        if (case.rate) |r| config.expert_quant_rate = r;
        const got = residentBytesWithConfig(io, t.allocator, path, &config);
        if (case.want) |want| try t.expectError(want, got) else _ = try got;
    }
}

test "mimo source rejects QKV input width that differs from hidden size" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var fixture = try makeTinySourceFixture(t.io, t.allocator, &tmp);
    defer fixture.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var source = try loadSourceIndex(t.io, arena.allocator(), fixture.path);
    const key = "model.layers.0.self_attn.qkv_proj.weight";
    var meta = source.tensors.get(key).?;
    meta.shape = &.{ 384, 256 };
    meta.data_start = 0;
    meta.data_end = 384 * 256;
    const scales = source.tensors.getPtr("model.layers.0.self_attn.qkv_proj.weight_scale_inv").?;
    scales.shape = &.{ 4, 2 };
    scales.data_start = 0;
    scales.data_end = 4 * 2 * 4;
    try t.expectError(error.InvalidFp8Shape, validateFp8Pair(&source, arena.allocator(), &fixture.config, key, meta));
}

test "mimo source rejects malformed FP8 scale grids" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var fixture = try makeTinySourceFixture(t.io, t.allocator, &tmp);
    defer fixture.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var source = try loadSourceIndex(t.io, arena.allocator(), fixture.path);
    const key = "model.layers.0.self_attn.qkv_proj.weight";
    const scales = source.tensors.getPtr("model.layers.0.self_attn.qkv_proj.weight_scale_inv").?;
    // Five tiles explain no rank count of [128 | 128 | 128] rows.
    scales.shape = &.{ 5, 1 };
    scales.data_end = scales.data_start + 5 * 4;
    try t.expectError(error.InvalidQkvGeometry, validateFp8Pair(&source, arena.allocator(), &fixture.config, key, source.tensors.get(key).?));
}

test "mimo source refuses NaN codes and scales whose products leave bf16" {
    const t = std.testing;
    const bf16_max: f32 = @bitCast(@as(u32, 0x7f7f0000));
    const cases = [_]struct { code: u8, scale: f32, want: ?anyerror }{
        .{ .code = 0x7f, .scale = 1, .want = error.InvalidFp8Value },
        .{ .code = 0xff, .scale = 1, .want = error.InvalidFp8Value },
        .{ .code = 0x38, .scale = std.math.inf(f32), .want = error.InvalidFp8Scale },
        .{ .code = 0x38, .scale = std.math.nan(f32), .want = error.InvalidFp8Scale },
        .{ .code = 0x38, .scale = bf16_max / 256.0, .want = error.InvalidFp8Scale },
        .{ .code = 0x7e, .scale = bf16_max / 512.0, .want = null },
    };
    for (cases) |case| {
        const codes: [128]u8 = @splat(case.code);
        var scales: [4]u8 = undefined;
        std.mem.writeInt(u32, &scales, @bitCast(case.scale), .little);
        if (case.want) |want| {
            try t.expectError(want, validateFp8Payload(&codes, &scales));
        } else try validateFp8Payload(&codes, &scales);
    }
}

/// An 8-bit g64 stored-affine triple for `base` over a [rows, 128] linear.
fn affineTriple(allocator: Allocator, base: []const u8, rows: u64, out: *std.ArrayList(TestTensor)) !void {
    const shapes = try allocator.alloc(u64, 4);
    shapes[0..4].* = .{ rows, 32, rows, 2 };
    try out.append(allocator, .{ .key = try std.fmt.allocPrint(allocator, "{s}.weight", .{base}), .dtype = "U32", .shape = shapes[0..2], .bytes = try testBf16Bytes(allocator, rows * 64, 0x1234 + @as(u16, @intCast(rows))) });
    try out.append(allocator, .{ .key = try std.fmt.allocPrint(allocator, "{s}.scales", .{base}), .dtype = "BF16", .shape = shapes[2..4], .bytes = try testBf16Bytes(allocator, rows * 2, 0x3c00) });
    try out.append(allocator, .{ .key = try std.fmt.allocPrint(allocator, "{s}.biases", .{base}), .dtype = "BF16", .shape = shapes[2..4], .bytes = try testBf16Bytes(allocator, rows * 2, 0xbf00) });
}

test "mimo source serves a pack's stored affine o_proj, lm_head and embed_tokens as stored and bills those bytes" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var fixture = try makeTinySourceFixture(io, t.allocator, &tmp);
    defer fixture.deinit();
    const plain_bytes = try residentBytesWithConfig(io, t.allocator, fixture.path, &fixture.config);

    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tensors: std.ArrayList(TestTensor) = .empty;
    try affineTriple(a, "model.layers.0.self_attn.o_proj", 128, &tensors);
    try affineTriple(a, "lm_head", 2, &tensors);
    try affineTriple(a, "model.embed_tokens", 2, &tensors);
    try redirectToNewShard(io, a, &tmp, tensors.items);

    var weights = try loadWeights(io, t.allocator, fixture.path, &fixture.config);
    defer weights.deinit();
    for (tensors.items) |want| {
        const got = weights.get(want.key) orelse return error.TestMissingWeight;
        const is_codes = std.mem.endsWith(u8, want.key, ".weight");
        try t.expectEqual(if (is_codes) mlx.mlx_dtype.uint32 else mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(got));
        try t.expectEqual(want.shape[0], @as(u64, @intCast(mlx.getShape(got)[0])));
        try t.expectEqual(want.shape[1], @as(u64, @intCast(mlx.getShape(got)[1])));
        try mlx.check(mlx.mlx_array_eval(got));
        const bytes: [*]const u8 = if (is_codes)
            @ptrCast(mlx.mlx_array_data_uint32(got) orelse return error.TestUnexpectedNullData)
        else
            @ptrCast(mlx.mlx_array_data_bfloat16(got) orelse return error.TestUnexpectedNullData);
        try t.expectEqualSlices(u8, want.bytes, bytes[0..want.bytes.len]);
    }
    // Billed as stored: the three bf16 tensors leave, the packed triples arrive.
    const bf16_bytes: u64 = 128 * 128 * 2 + 2 * (2 * 128 * 2);
    const packed_bytes: u64 = (128 * 32 * 4 + 2 * 128 * 2 * 2) + 2 * (2 * 32 * 4 + 2 * 2 * 2 * 2);
    try t.expectEqual(plain_bytes - bf16_bytes + packed_bytes, try residentBytesWithConfig(io, t.allocator, fixture.path, &fixture.config));
}

test "mimo source refuses a stored affine trunk triple that is incomplete or does not solve" {
    const t = std.testing;
    const io = t.io;
    const o = "model.layers.0.self_attn.o_proj";
    const Case = struct { shapes: [3][]const u64, parts: []const u8, want: anyerror };
    const cases = [_]Case{
        // codes without their grids; grids beside the bf16 weight
        .{ .shapes = .{ &.{ 128, 32 }, &.{ 128, 2 }, &.{ 128, 2 } }, .parts = "w", .want = error.AffineTrunkIncomplete },
        .{ .shapes = .{ &.{ 128, 32 }, &.{ 128, 2 }, &.{ 128, 2 } }, .parts = "sb", .want = error.AffineTrunkIncomplete },
        // 7 words cover 128 inputs at no admitted width; 3 groups split no row; rows short
        .{ .shapes = .{ &.{ 128, 7 }, &.{ 128, 2 }, &.{ 128, 2 } }, .parts = "wsb", .want = error.MimoTensorShapeMismatch },
        .{ .shapes = .{ &.{ 128, 32 }, &.{ 128, 3 }, &.{ 128, 3 } }, .parts = "wsb", .want = error.MimoTensorShapeMismatch },
        .{ .shapes = .{ &.{ 128, 32 }, &.{ 128, 2 }, &.{ 128, 4 } }, .parts = "wsb", .want = error.MimoTensorShapeMismatch },
        .{ .shapes = .{ &.{ 64, 32 }, &.{ 64, 2 }, &.{ 64, 2 } }, .parts = "wsb", .want = error.MimoTensorShapeMismatch },
    };
    for (cases) |case| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        var fixture = try makeTinySourceFixture(io, t.allocator, &tmp);
        defer fixture.deinit();
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var tensors: std.ArrayList(TestTensor) = .empty;
        for ([_][]const u8{ "w", "s", "b" }, [_][]const u8{ "weight", "scales", "biases" }, [_][]const u8{ "U32", "BF16", "BF16" }, case.shapes) |tag, part, dtype, shape| {
            if (std.mem.indexOf(u8, case.parts, tag) == null) continue;
            const elem: u64 = if (tag[0] == 'w') 4 else 2;
            try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.{s}", .{ o, part }), .dtype = dtype, .shape = shape, .bytes = try a.alloc(u8, shape[0] * shape[1] * elem) });
        }
        try redirectToNewShard(io, a, &tmp, tensors.items);
        try t.expectError(case.want, residentBytesWithConfig(io, t.allocator, fixture.path, &fixture.config));
        try t.expectError(case.want, loadWeights(io, t.allocator, fixture.path, &fixture.config));
    }
}

test "mimo source keeps the FP8 trunk in its source bytes and bills them" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var fixture = try makeTinySourceFixture(io, std.testing.allocator, &tmp);
    defer fixture.deinit();

    // 34560 bytes of already-bf16 trunk plus the FP8 codes and f32 scale
    // grids exactly as stored: QKV [384,128] + [4,1], three MLP [128,128] + [1,1].
    const expected_bytes: u64 = 34560 + (384 * 128 + 4 * 4) + 3 * (128 * 128 + 4);
    try std.testing.expectEqual(
        expected_bytes,
        try residentBytesWithConfig(io, std.testing.allocator, fixture.path, &fixture.config),
    );
    try std.testing.expectEqual(
        expected_bytes,
        try residentBytes(io, std.testing.allocator, fixture.path),
    );

    var weights = try loadWeights(io, std.testing.allocator, fixture.path, &fixture.config);
    defer weights.deinit();
    try std.testing.expectEqual(@as(u32, 14), weights.count());
    try std.testing.expect(weights.get("model.layers.0.self_attn.q_proj.weight") == null);
    try std.testing.expect(weights.get("model.layers.0.mlp.gate_proj.biases") == null);
    try std.testing.expect(weights.get("model.layers.0.mlp.experts.0.gate_proj.weight") == null);
    try std.testing.expect(weights.get("model.mtp.layers.0.fake.weight") == null);
    try std.testing.expect(weights.get("visual.fake") == null);

    const embed = weights.get("model.embed_tokens.weight").?;
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(embed));
    const embed_data = mlx.mlx_array_data_bfloat16(embed) orelse return error.TestUnexpectedNullData;
    for (0..fixture.embed.len / 2) |i| {
        const want = std.mem.readInt(u16, fixture.embed[i * 2 ..][0..2], .little);
        try std.testing.expectEqual(want, embed_data[i]);
    }

    const cases = [_]struct { base: []const u8, codes: []const u8, scales: []const u8, shape: [2]c_int, grid: [2]c_int }{
        .{ .base = "model.layers.0.self_attn.qkv_proj", .codes = fixture.qkv_weight, .scales = fixture.qkv_scales, .shape = .{ 384, 128 }, .grid = .{ 4, 1 } },
        .{ .base = "model.layers.0.mlp.gate_proj", .codes = fixture.mlp_weight, .scales = fixture.mlp_scales, .shape = .{ 128, 128 }, .grid = .{ 1, 1 } },
        .{ .base = "model.layers.0.mlp.down_proj", .codes = fixture.mlp_weight, .scales = fixture.mlp_scales, .shape = .{ 128, 128 }, .grid = .{ 1, 1 } },
    };
    var name_buf: [96]u8 = undefined;
    for (cases) |case| {
        const w = weights.get(try std.fmt.bufPrint(&name_buf, "{s}.weight", .{case.base})).?;
        try std.testing.expectEqual(mlx.mlx_dtype.uint8, mlx.mlx_array_dtype(w));
        try std.testing.expectEqualSlices(c_int, &case.shape, mlx.getShape(w));
        const codes = mlx.mlx_array_data_uint8(w) orelse return error.TestUnexpectedNullData;
        try std.testing.expectEqualSlices(u8, case.codes, codes[0..case.codes.len]);
        const sc = weights.get(try std.fmt.bufPrint(&name_buf, "{s}.scales", .{case.base})).?;
        try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(sc));
        try std.testing.expectEqualSlices(c_int, &case.grid, mlx.getShape(sc));
        const grid = mlx.mlx_array_data_float32(sc) orelse return error.TestUnexpectedNullData;
        try std.testing.expectEqualSlices(u8, case.scales, std.mem.sliceAsBytes(grid[0 .. case.scales.len / 4]));
    }
}

test "mimo source uploads the vision tower on request, the Conv3d patch weight as a Linear, and bills what it uploads" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var fixture = try makeTinySourceFixture(io, t.allocator, &tmp);
    defer fixture.deinit();
    const trunk_bytes = try residentBytesWithConfig(io, t.allocator, fixture.path, &fixture.config);

    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const patch = try testBf16Bytes(a, 4 * 3 * 2 * 2 * 2, 0x3f80);
    const norm = try testBf16Bytes(a, 4, 0x4000);
    const audio = try testBf16Bytes(a, 2, 0x3f80);
    const tensors = [_]TestTensor{
        .{ .key = "visual.patch_embed.proj.weight", .dtype = "BF16", .shape = &[_]u64{ 4, 3, 2, 2, 2 }, .bytes = patch },
        .{ .key = "visual.blocks.0.norm1.weight", .dtype = "BF16", .shape = &[_]u64{4}, .bytes = norm },
        .{ .key = "audio_encoder.fake", .dtype = "BF16", .shape = &[_]u64{2}, .bytes = audio },
        .{ .key = "speech_embeddings.0.weight", .dtype = "BF16", .shape = &[_]u64{2}, .bytes = audio },
    };
    try redirectToNewShard(io, a, &tmp, &tensors);

    var weights = model.Weights.init(t.allocator);
    defer weights.deinit();
    try loadVisionWeightsInto(&weights, io, t.allocator, fixture.path);
    // `visual.fake` (F32 [1]) plus the two tower tensors; audio stays on disk.
    try t.expectEqual(@as(u32, 3), weights.count());
    try t.expect(weights.get("audio_encoder.fake") == null and weights.get("speech_embeddings.0.weight") == null);
    const pw = weights.get("visual.patch_embed.proj.weight") orelse return error.TestMissingWeight;
    try t.expectEqualSlices(c_int, &.{ 4, 24 }, mlx.getShape(pw));
    try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(pw));
    try t.expectEqual(@as(u64, 4 + patch.len + norm.len), try visionResidentBytes(io, t.allocator, fixture.path));
    // The trunk bill is unchanged: the tower is billed apart, like the MTP heads.
    try t.expectEqual(trunk_bytes, try residentBytesWithConfig(io, t.allocator, fixture.path, &fixture.config));
}

test "mimo source loads the MTP heads apart from the trunk and bills what it uploads" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var fixture = try makeTinySourceFixture(io, std.testing.allocator, &tmp);
    defer fixture.deinit();
    var heads = try loadMtpWeights(io, std.testing.allocator, fixture.path);
    defer heads.deinit();
    try std.testing.expectEqual(@as(u32, 1), heads.count());
    try std.testing.expect(heads.get("model.mtp.layers.0.fake.weight") != null);
    try std.testing.expectEqual(@as(u64, 2), try mtpResidentBytes(io, std.testing.allocator, fixture.path));
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

test "mimo EXL3 preflight rejects dimensions outside H128" {
    var config = model.ModelConfig{};
    config.num_hidden_layers = 2;
    config.first_k_dense_replace = 1;
    config.num_experts = 2;
    config.hidden_size = 128;
    config.moe_intermediate_size = 144;
    const shape = [_]u64{ 2, 8, 9, 40 };
    const meta = TensorMeta{
        .dtype = .u16,
        .shape = &shape,
        .data_start = 0,
        .data_end = 2 * 8 * 9 * 40 * 2,
        .data_base = 0,
        .file = "experts.safetensors",
    };
    try std.testing.expectError(error.MimoTensorShapeMismatch, validateExl3Expert("model.layers.1.mlp.switch_mlp.gate_proj.trellis", meta, &config));
    var down_meta = meta;
    down_meta.shape = &.{ 2, 9, 8, 40 };
    try std.testing.expectError(error.MimoTensorShapeMismatch, validateExl3Expert("model.layers.1.mlp.switch_mlp.down_proj.trellis", down_meta, &config));
    config.moe_intermediate_size = 256;
    var valid_meta = meta;
    valid_meta.shape = &.{ 2, 8, 16, 40 };
    valid_meta.data_end = 2 * 8 * 16 * 40 * 2;
    try validateExl3Expert("model.layers.1.mlp.switch_mlp.gate_proj.trellis", valid_meta, &config);
}

test "sushi coder MiMo preflight admits independent gate and up rates" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try writeTinyExl3Source(t.io, alloc, tmp.dir, 40);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const path = path_buf[0..path_len];
    var config = try model.parseConfig(t.io, alloc, path);
    defer config.deinit(alloc);
    config.expert_layout = .exl3_k4;
    config.expert_quant_rate = .{ .n = 64 };
    var source = try loadSourceIndex(t.io, alloc, path);
    try validatePlan(&source, alloc, &config);
    const up = source.tensors.getPtr("model.layers.1.mlp.switch_mlp.up_proj.trellis").?;
    const original = up.*;
    up.shape = &.{ 2, 8, 8, 48 };
    up.data_end = up.data_start + 2 * 8 * 8 * 48 * 2;
    try validatePlan(&source, alloc, &config);
    up.* = original;
    const down = source.tensors.getPtr("model.layers.1.mlp.switch_mlp.down_proj.trellis").?;
    down.shape = &.{ 2, 8, 8, 48 };
    down.data_end = down.data_start + 2 * 8 * 8 * 48 * 2;
    try validatePlan(&source, alloc, &config);
}

test "mimo source classifies a multi-layer dense prefix's FP8 MLP as the trunk it is" {
    // The dense prefix is config-driven: a checkpoint whose `first_k_dense_replace`
    // is 2 puts an FP8 MLP on layer 1 too, and the parser and the transformer both
    // read it from the config. The loader must not decide otherwise.
    const t = std.testing;
    const io = t.io;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try writeTinySource(io, alloc, tmp.dir, .{ .exl3 = .{ .n = 40 } }, 2);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    const path = path_buf[0..path_len];
    var config = try model.parseConfig(io, alloc, path);
    defer config.deinit(alloc);
    try t.expectEqual(@as(u32, 2), config.first_k_dense_replace);
    config.expert_layout = .exl3_k4;
    config.expert_quant_rate = .{ .n = 64 };

    var source = try loadSourceIndex(io, alloc, path);
    try t.expectEqual(TensorKind.fp8_weight, try classifyKey("model.layers.1.mlp.gate_proj.weight", &config));
    try t.expectEqual(TensorKind.fp8_scale, try classifyKey("model.layers.1.mlp.gate_proj.weight_scale_inv", &config));
    try t.expectEqual(
        TensorKind.routed_expert,
        try classifyKey("model.layers.2.mlp.switch_mlp.gate_proj.trellis", &config),
    );
    // The whole plan, which requires each dense layer's own FP8 MLP pair.
    try validatePlan(&source, alloc, &config);
    _ = try residentBytesWithConfig(io, t.allocator, path, &config);
    var weights = try loadWeights(io, t.allocator, path, &config);
    defer weights.deinit();
    for ([_][]const u8{ "gate", "up", "down" }) |proj| {
        const k = try std.fmt.allocPrint(alloc, "model.layers.1.mlp.{s}_proj.weight", .{proj});
        const w = weights.get(k) orelse return error.TestMissingWeight;
        try t.expectEqual(mlx.mlx_dtype.uint8, mlx.mlx_array_dtype(w));
    }
}

test "sushi coder MiMo accepts grouped tensors and pruned experts" {
    var config = model.ModelConfig{};
    config.expert_layout = .exl3_k4;
    config.expert_quant_rate = .{ .n = 64 };
    config.num_hidden_layers = 3;
    config.first_k_dense_replace = 1;
    config.num_experts = 256;
    config.hidden_size = 128;
    config.moe_intermediate_size = 128;
    const meta = TensorMeta{ .dtype = .u16, .shape = &.{ 2, 8, 8, 32 }, .data_start = 0, .data_end = 2 * 8 * 8 * 32 * 2, .data_base = 0, .file = "experts.safetensors" };
    try validateExl3Expert("model.layers.1.mlp.switch_mlp.gate_proj.g0.trellis", meta, &config);
    try validateExl3Expert("model.layers.2.mlp.switch_mlp.gate_proj.trellis", meta, &config);
}

test "sushi coder MiMo ragged bill equals all stored tensors" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try writeTinyExl3Source(t.io, alloc, tmp.dir, 40);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const len = try tmp.dir.realPath(t.io, &path_buf);
    const path = path_buf[0..len];
    var config = try model.parseConfig(t.io, alloc, path);
    config.expert_quant_rate = .{ .n = 64 };
    var source = try loadSourceIndex(t.io, alloc, path);
    const before = try countResidentBytes(&source, alloc, &config);
    for ([_][]const u8{ "gate", "up", "down" }, 0..) |proj, p| {
        for ([_][]const u8{ "trellis", "suh", "svh" }, 0..) |part, i| {
            const old = try std.fmt.allocPrint(alloc, "model.layers.1.mlp.switch_mlp.{s}_proj.{s}", .{ proj, part });
            const original = source.tensors.fetchRemove(old).?.value;
            for (0..2) |group| {
                var meta = original;
                const e: u64 = if (group == 0) 1 else 2;
                const n: u64 = @intCast(32 + 16 * ((group + p) % 3));
                meta.shape = if (i == 0) try alloc.dupe(u64, &.{ e, 8, 8, n }) else try alloc.dupe(u64, &.{ e, 128 });
                meta.data_end = meta.data_start + (if (i == 0) e * 8 * 8 * n * 2 else e * 128 * 2);
                const key = try std.fmt.allocPrint(alloc, "model.layers.1.mlp.switch_mlp.{s}_proj.g{d}.{s}", .{ proj, group, part });
                try source.tensors.put(key, meta);
            }
        }
    }
    const router = source.tensors.getPtr("model.layers.1.mlp.gate.weight").?;
    router.shape = &.{ 3, 128 };
    router.data_end = router.data_start + 3 * 128 * 2;
    const bias = source.tensors.getPtr("model.layers.1.mlp.gate.e_score_correction_bias").?;
    bias.shape = &.{3};
    bias.data_end = bias.data_start + 3 * 4;
    try validatePlan(&source, alloc, &config);
    const old_experts = 2 * 3 * (8 * 8 * 40 * 2 + 256 * 2);
    const new_experts = 3 * (8 * 8 * (32 + 48 + 64) * 2 + 3 * 256 * 2);
    try t.expectEqual(before - old_experts + new_experts + 128 * 6 + 4, try countResidentBytes(&source, alloc, &config));
    config.num_experts_per_tok = 4;
    try t.expectError(error.Exl3TopKExceedsExperts, validatePlan(&source, alloc, &config));
    config.num_experts_per_tok = 1;
    config.moe_n_group = 2;
    try t.expectError(error.Exl3RouterGroupsUnsupported, validatePlan(&source, alloc, &config));
    config.moe_n_group = 1;
    config.expert_streaming = true;
    try t.expectError(error.Exl3RaggedStreamingUnsupported, validatePlan(&source, alloc, &config));
}

pub fn validateExl3Pack(io: std.Io, allocator: Allocator, model_dir: []const u8, config: *const model.ModelConfig) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const source = try loadSourceIndex(io, alloc, model_dir);
    try validateExl3Source(&source, alloc, config);
}

fn validateExl3Source(source: *const SourceIndex, alloc: Allocator, config: *const model.ModelConfig) !void {
    const mimo = std.mem.eql(u8, config.model_type, "mimo_v2");
    const glm = config.isGlm5();
    if (mimo or glm) try validateShardStamps(source, config);
    const prefix = if (glm) "model.language_model" else if (mimo) "model" else "language_model.model";
    for (config.first_k_dense_replace..config.num_hidden_layers) |layer| {
        const base = try std.fmt.allocPrint(alloc, "{s}.layers.{d}", .{ prefix, layer });
        _ = try validateExl3Layer(source, alloc, config, base);
    }
    if (glm) {
        var keys = source.tensors.keyIterator();
        const start = "model.language_model.layers.";
        while (keys.next()) |key| {
            if (!std.mem.startsWith(u8, key.*, start) or std.mem.indexOf(u8, key.*, ".mlp.switch_mlp.") == null) continue;
            const rest = key.*[start.len..];
            const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.MimoLayerOutOfRange;
            const layer = std.fmt.parseInt(u32, rest[0..dot], 10) catch return error.MimoLayerOutOfRange;
            if (layer < config.first_k_dense_replace or @as(u64, layer) >= @as(u64, config.num_hidden_layers) + config.glm_mtp_layers) return error.MimoLayerOutOfRange;
        }
        for (0..config.glm_mtp_layers) |extra| {
            const base = try std.fmt.allocPrint(alloc, "{s}.layers.{d}", .{ prefix, config.num_hidden_layers + extra });
            const probe = try std.fmt.allocPrint(alloc, "{s}.mlp.switch_mlp.", .{base});
            var parts = source.tensors.keyIterator();
            while (parts.next()) |key| {
                if (std.mem.startsWith(u8, key.*, probe)) {
                    _ = try validateExl3Layer(source, alloc, config, base);
                    break;
                }
            }
        }
    } else if (!mimo) {
        var it = source.tensors.keyIterator();
        while (it.next()) |key| {
            if (std.mem.startsWith(u8, key.*, "language_model.mtp.layers.0.")) {
                _ = try validateExl3Layer(source, alloc, config, "language_model.mtp.layers.0");
                break;
            }
        }
    }
}

test "sushi coder incomplete grouped EXL3 index refuses instead of guessing a layout" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try writeTinyExl3Source(t.io, alloc, tmp.dir, 40);
    const raw = try tmp.dir.readFileAlloc(t.io, "model.safetensors.index.json", alloc, .limited(1024 * 1024));
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, raw, .{});
    const map = &parsed.value.object.getPtr("weight_map").?.object;
    var grouped: std.json.ObjectMap = .empty;
    var it = map.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (std.mem.eql(u8, key, "model.layers.1.mlp.switch_mlp.up_proj.trellis")) continue;
        const renamed = if (std.mem.indexOf(u8, key, ".mlp.switch_mlp.") != null) blk: {
            const dot = std.mem.lastIndexOfScalar(u8, key, '.').?;
            break :blk try std.fmt.allocPrint(alloc, "{s}.g0{s}", .{ key[0..dot], key[dot..] });
        } else key;
        try grouped.put(alloc, renamed, entry.value_ptr.*);
    }
    map.* = grouped;
    const broken = try std.json.Stringify.valueAlloc(alloc, parsed.value, .{});
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors.index.json", .data = broken });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const len = try tmp.dir.realPath(t.io, &buf);
    try expectError(error.ExpertLayoutUnsupported, model.parseConfig(t.io, alloc, buf[0..len]));
}

test "sushi coder qwen trunk and MTP preflight use each router width" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var tensors: std.ArrayList(TestTensor) = .empty;
    var entries: std.ArrayList(TestIndexEntry) = .empty;
    for ([_][]const u8{ "language_model.model.layers.0", "language_model.model.layers.1", "language_model.mtp.layers.0" }, 0..) |base, layer| {
        const e: u64 = if (layer == 1) 2 else 3;
        try tensors.append(alloc, .{
            .key = try std.fmt.allocPrint(alloc, "{s}.mlp.gate.weight", .{base}),
            .dtype = "BF16",
            .shape = try alloc.dupe(u64, &.{ e, 128 }),
            .bytes = try testBf16Bytes(alloc, @intCast(e * 128), 0x3f80),
        });
        for (0..2) |group| {
            const count: u64 = if (group == 0) 1 else e - 1;
            for ([_][]const u8{ "gate", "up", "down" }, 0..) |projection, p| {
                for ([_][]const u8{ "trellis", "suh", "svh" }, 0..) |part, i| {
                    const n: u64 = @intCast(32 + 16 * ((group + p) % 3));
                    try tensors.append(alloc, .{
                        .key = try std.fmt.allocPrint(alloc, "{s}.mlp.switch_mlp.{s}_proj.g{d}.{s}", .{ base, projection, group, part }),
                        .dtype = if (i == 0) "U16" else "F16",
                        .shape = if (i == 0) try alloc.dupe(u64, &.{ count, 8, 8, n }) else try alloc.dupe(u64, &.{ count, 128 }),
                        .bytes = try testBf16Bytes(alloc, @intCast(if (i == 0) count * 8 * 8 * n else count * 128), 0),
                    });
                }
            }
        }
    }
    for (tensors.items) |tensor| try entries.append(alloc, .{ .key = tensor.key, .file = "ragged.safetensors" });
    try writeTestShard(t.io, alloc, tmp.dir, "ragged.safetensors", tensors.items);
    try writeTestIndex(t.io, alloc, tmp.dir, entries.items);
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const len = try tmp.dir.realPath(t.io, &buf);
    var cfg = model.ModelConfig{ .model_type = "qwen4_exp", .expert_layout = .exl3_k4, .num_hidden_layers = 2, .num_experts = 512, .num_experts_per_tok = 2, .hidden_size = 128, .moe_intermediate_size = 128 };
    try validateExl3Pack(t.io, alloc, buf[0..len], &cfg);
    cfg.num_experts_per_tok = 3;
    try t.expectError(error.Exl3TopKExceedsExperts, validateExl3Pack(t.io, alloc, buf[0..len], &cfg));
    cfg.num_experts_per_tok = 2;
    cfg.expert_streaming = true;
    try t.expectError(error.Exl3RaggedStreamingUnsupported, validateExl3Pack(t.io, alloc, buf[0..len], &cfg));
}

test "sushi coder legacy uniform billing preserves trunk-only config" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var bills: [2]u64 = undefined;
    for ([_]u64{ 0, 40 }, 0..) |n, i| {
        const name = if (n == 0) "trunk" else "uniform";
        try tmp.dir.createDirPath(t.io, name);
        var dir = try tmp.dir.openDir(t.io, name, .{});
        defer dir.close(t.io);
        try writeTinyExl3Source(t.io, alloc, dir, n);
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const len = try dir.realPath(t.io, &buf);
        var config = try model.parseConfig(t.io, alloc, buf[0..len]);
        defer config.deinit(alloc);
        try t.expectEqual(if (n == 0) expert_quant.Layout.bf16_fused else expert_quant.Layout.exl3_k4, config.expert_layout);
        bills[i] = try residentBytesWithConfig(t.io, alloc, buf[0..len], &config);
    }
    try t.expectEqual(@as(u64, 3 * (2 * 8 * 8 * 40 * 2 + 2 * 2 * 128 * 2)), bills[1] - bills[0]);
}

fn checkMimoExl3StreamTrunk(runtime: bool) !void {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try writeTinyExl3Source(t.io, a, tmp.dir, 36);
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const len = try tmp.dir.realPath(t.io, &buf);
    const path = buf[0..len];
    var cfg = try model.parseConfig(t.io, a, path);
    defer cfg.deinit(a);
    const resident_bill = try residentBytesWithConfig(t.io, a, path, &cfg);
    cfg.expert_streaming = true;
    const streamed_bill = try residentBytesWithConfig(t.io, a, path, &cfg);
    try t.expectEqual(@as(u64, 3 * (2 * 8 * 8 * 36 * 2 + 2 * 2 * 128 * 2)), resident_bill - streamed_bill);
    try t.expectEqual(streamed_bill, (try model.streamingResidentSplit(t.io, a, path, &cfg)).trunk);
    if (!runtime) return;
    cfg.expert_streaming = false;
    var resident = try model.loadWeightsForConfig(t.io, a, path, &cfg, false);
    defer resident.deinit();
    cfg.expert_streaming = true;
    var streamed = try model.loadWeightsForConfig(t.io, a, path, &cfg, false);
    defer streamed.deinit();
    var it = resident.map.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (std.mem.indexOf(u8, key, ".switch_mlp.") != null) {
            try t.expect(streamed.get(key) == null);
            continue;
        }
        const actual = streamed.get(key) orelse return error.MissingStreamedTrunk;
        try t.expectEqual(mlx.mlx_array_dtype(entry.value_ptr.*), mlx.mlx_array_dtype(actual));
        try t.expectEqualSlices(c_int, mlx.getShape(entry.value_ptr.*), mlx.getShape(actual));
        try mlx.check(mlx.mlx_array_eval(entry.value_ptr.*));
        try mlx.check(mlx.mlx_array_eval(actual));
        const data = struct {
            fn ptr(arr: mlx.mlx_array) [*]const u8 {
                return switch (mlx.mlx_array_dtype(arr)) {
                    .uint8 => mlx.mlx_array_data_uint8(arr).?,
                    .bfloat16 => @ptrCast(mlx.mlx_array_data_bfloat16(arr).?),
                    .float32 => @ptrCast(mlx.mlx_array_data_float32(arr).?),
                    else => unreachable,
                };
            }
        }.ptr;
        const expected_bytes = data(entry.value_ptr.*);
        const actual_bytes = data(actual);
        const bytes = mlx.mlx_array_size(actual) * @as(usize, switch (mlx.mlx_array_dtype(actual)) {
            .uint8 => 1,
            .bfloat16 => 2,
            .float32 => 4,
            else => unreachable,
        });
        try t.expectEqualSlices(u8, expected_bytes[0..bytes], actual_bytes[0..bytes]);
    }
}

test "MiMo EXL3 streaming CPU bills the FP8 trunk without routed banks" {
    try checkMimoExl3StreamTrunk(false);
}

test "MiMo EXL3 streaming loads the identical FP8 trunk without routed banks" {
    try checkMimoExl3StreamTrunk(true);
}

test "GLM EXL3 preflight refuses orphan MTP and out of range packed tensors" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var source = SourceIndex{ .weight_map = std.StringHashMap([]const u8).init(a), .files = std.StringHashMap(void).init(a), .tensors = std.StringHashMap(TensorMeta).init(a), .stamps = std.StringHashMap(ShardStamp).init(a) };
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .expert_layout = .exl3_k4, .expert_quant_rate = .{ .n = 36 }, .num_hidden_layers = 2, .first_k_dense_replace = 1, .num_experts = 2, .num_experts_per_tok = 1, .hidden_size = 128, .moe_intermediate_size = 128, .glm_mtp_layers = 1 };
    const code = TensorMeta{ .dtype = .u16, .shape = &.{ 2, 8, 8, 36 }, .data_start = 0, .data_end = 2 * 8 * 8 * 36 * 2, .data_base = 0, .file = "test.safetensors" };
    const scale = TensorMeta{ .dtype = .f16, .shape = &.{ 2, 128 }, .data_start = 0, .data_end = 2 * 128 * 2, .data_base = 0, .file = "test.safetensors" };
    for ([_][]const u8{ "gate", "up", "down" }) |p| for ([_][]const u8{ "trellis", "suh", "svh" }, 0..) |part, i| {
        try source.tensors.put(try std.fmt.allocPrint(a, "model.language_model.layers.1.mlp.switch_mlp.{s}_proj.{s}", .{ p, part }), if (i == 0) code else scale);
    };
    try source.tensors.put("model.language_model.layers.1.mlp.gate.weight", .{ .dtype = .bf16, .shape = &.{ 2, 128 }, .data_start = 0, .data_end = 512, .data_base = 0, .file = "test.safetensors" });
    try validateExl3Source(&source, a, &cfg);
    for ([_]u32{ 2, 0, 3 }) |li| {
        const key = try std.fmt.allocPrint(a, "model.language_model.layers.{d}.mlp.switch_mlp.up_proj.trellis", .{li});
        try source.tensors.put(key, code);
        defer _ = source.tensors.remove(key);
        if (validateExl3Source(&source, a, &cfg)) |_| return error.ExpectedGlmPackRejection else |_| {}
    }
}
