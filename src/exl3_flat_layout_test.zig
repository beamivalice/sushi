//! Characterization coverage for the flat EXL3 component layout. The fixture
//! uses real safetensors headers and data so discovery and resident byte
//! accounting exercise the same index/file paths as a downloaded pack.

const std = @import("std");
const expert_quant = @import("expert_quant.zig");
const model = @import("model.zig");
const model_discovery = @import("model_discovery.zig");

const TensorSpec = struct {
    key: []const u8,
    dtype: []const u8,
    shape: []const u64,
    bytes: usize,
};

const IndexSpec = struct {
    key: []const u8,
    file: []const u8,
};

const TRELLIS_SHAPE = [_]u64{ 2, 1, 1, 64 };
const SIDE_SHAPE = [_]u64{ 2, 16 };
const VECTOR_SHAPE = [_]u64{ 1, 16 };
const MATRIX_SHAPE = [_]u64{ 16, 16 };
const ONE_SHAPE = [_]u64{1};

const EXL3_SPECS = [_]TensorSpec{
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis", .dtype = "U16", .shape = TRELLIS_SHAPE[0..], .bytes = 256 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.suh", .dtype = "F16", .shape = SIDE_SHAPE[0..], .bytes = 64 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.svh", .dtype = "F16", .shape = SIDE_SHAPE[0..], .bytes = 64 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.up_proj.trellis", .dtype = "U16", .shape = TRELLIS_SHAPE[0..], .bytes = 256 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.up_proj.suh", .dtype = "F16", .shape = SIDE_SHAPE[0..], .bytes = 64 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.up_proj.svh", .dtype = "F16", .shape = SIDE_SHAPE[0..], .bytes = 64 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.down_proj.trellis", .dtype = "U16", .shape = TRELLIS_SHAPE[0..], .bytes = 256 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.down_proj.suh", .dtype = "F16", .shape = SIDE_SHAPE[0..], .bytes = 64 },
    .{ .key = "language_model.model.layers.0.mlp.switch_mlp.down_proj.svh", .dtype = "F16", .shape = SIDE_SHAPE[0..], .bytes = 64 },
};

const EMBED_SPEC = TensorSpec{
    .key = "language_model.model.embed_tokens.weight",
    .dtype = "F32",
    .shape = VECTOR_SHAPE[0..],
    .bytes = 64,
};

const LM_HEAD_SPEC = TensorSpec{
    .key = "language_model.lm_head.weight",
    .dtype = "F32",
    .shape = MATRIX_SHAPE[0..],
    .bytes = 1024,
};

const TRUNK_SPEC = TensorSpec{
    .key = "language_model.model.layers.0.self_attn.q_proj.weight",
    .dtype = "F32",
    .shape = MATRIX_SHAPE[0..],
    .bytes = 1024,
};

const TRUNK_AUX_SPEC = TensorSpec{
    .key = "language_model.model.layers.0.self_attn.k_proj.weight",
    .dtype = "F32",
    .shape = MATRIX_SHAPE[0..],
    .bytes = 1024,
};

const MTP_SPECS = [_]TensorSpec{
    .{
        .key = "language_model.mtp.fc_hidden.weight",
        .dtype = "F32",
        .shape = MATRIX_SHAPE[0..],
        .bytes = 1024,
    },
    .{
        .key = "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.trellis",
        .dtype = "U16",
        .shape = TRELLIS_SHAPE[0..],
        .bytes = 256,
    },
    .{
        .key = "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.suh",
        .dtype = "F16",
        .shape = SIDE_SHAPE[0..],
        .bytes = 64,
    },
    .{
        .key = "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.svh",
        .dtype = "F16",
        .shape = SIDE_SHAPE[0..],
        .bytes = 64,
    },
};

const VISION_SPEC = TensorSpec{
    .key = "model.visual.fake",
    .dtype = "F32",
    .shape = ONE_SHAPE[0..],
    .bytes = 4,
};

fn appendFormat(
    allocator: std.mem.Allocator,
    list: *std.ArrayList(u8),
    comptime format: []const u8,
    args: anytype,
) !void {
    const text = try std.fmt.allocPrint(allocator, format, args);
    defer allocator.free(text);
    try list.appendSlice(allocator, text);
}

fn writeSafetensors(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    filename: []const u8,
    specs: []const TensorSpec,
) !void {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    try header.append(allocator, '{');

    var offset: u64 = 0;
    for (specs, 0..) |spec, i| {
        if (i != 0) try header.append(allocator, ',');
        try appendFormat(allocator, &header, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ spec.key, spec.dtype });
        for (spec.shape, 0..) |dim, shape_i| {
            if (shape_i != 0) try header.append(allocator, ',');
            try appendFormat(allocator, &header, "{d}", .{dim});
        }
        const end = offset + @as(u64, @intCast(spec.bytes));
        try appendFormat(allocator, &header, "],\"data_offsets\":[{d},{d}]}}", .{ offset, end });
        offset = end;
    }
    try header.append(allocator, '}');

    const data_len: usize = @intCast(offset);
    const file_bytes = try allocator.alloc(u8, 8 + header.items.len + data_len);
    defer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.items.len, .little);
    @memcpy(file_bytes[8 .. 8 + header.items.len], header.items);
    for (file_bytes[8 + header.items.len ..], 0..) |*byte, i| {
        byte.* = @intCast((i + 17) % 251);
    }
    try dir.writeFile(io, .{ .sub_path = filename, .data = file_bytes });
}

fn writeIndex(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    entries: []const IndexSpec,
) !void {
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(allocator);
    try index.appendSlice(allocator, "{\"weight_map\":{");
    for (entries, 0..) |entry, i| {
        if (i != 0) try index.append(allocator, ',');
        try appendFormat(allocator, &index, "\"{s}\":\"{s}\"", .{ entry.key, entry.file });
    }
    try index.appendSlice(allocator, "}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}

fn writeFlatFixture(
    io: std.Io,
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    donor_dir: std.Io.Dir,
) !void {
    try dir.writeFile(io, .{
        .sub_path = "config.json",
        .data = "{\"model_type\":\"qwen4_exp\",\"hidden_size\":16,\"num_hidden_layers\":1,\"num_experts\":2,\"num_experts_per_tok\":1,\"moe_intermediate_size\":16,\"expert_quant\":{\"format\":\"exl3\",\"k\":4,\"codebook\":\"mul1\"}}",
    });
    try dir.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = "tiny ngram placeholder" });

    try writeSafetensors(io, allocator, dir, "model-experts-L00.safetensors", EXL3_SPECS[0..]);
    try writeSafetensors(io, allocator, dir, "model-embed.safetensors", &.{EMBED_SPEC});
    try writeSafetensors(io, allocator, dir, "model-lm-head.safetensors", &.{LM_HEAD_SPEC});
    try writeSafetensors(io, allocator, dir, "model-trunk-00001-of-00002.safetensors", &.{TRUNK_SPEC});
    try writeSafetensors(io, allocator, dir, "model-mtp.safetensors", MTP_SPECS[0..]);
    try writeSafetensors(io, allocator, dir, "model-vision.safetensors", &.{VISION_SPEC});

    // This donor lives outside the model pack. The canonical flat shard is a
    // hardlink to it, so the indexed loader reads one tensor exactly once.
    try writeSafetensors(io, allocator, donor_dir, "trunk-donor.safetensors", &.{TRUNK_AUX_SPEC});
    try std.Io.Dir.hardLink(
        donor_dir,
        "trunk-donor.safetensors",
        dir,
        "model-trunk-00002-of-00002.safetensors",
        io,
        .{},
    );

    var entries: [EXL3_SPECS.len + MTP_SPECS.len + 5]IndexSpec = undefined;
    var at: usize = 0;
    for (EXL3_SPECS) |spec| {
        entries[at] = .{ .key = spec.key, .file = "model-experts-L00.safetensors" };
        at += 1;
    }
    for (MTP_SPECS) |spec| {
        entries[at] = .{ .key = spec.key, .file = "model-mtp.safetensors" };
        at += 1;
    }
    entries[at] = .{ .key = EMBED_SPEC.key, .file = "model-embed.safetensors" };
    at += 1;
    entries[at] = .{ .key = LM_HEAD_SPEC.key, .file = "model-lm-head.safetensors" };
    at += 1;
    entries[at] = .{ .key = TRUNK_SPEC.key, .file = "model-trunk-00001-of-00002.safetensors" };
    at += 1;
    entries[at] = .{ .key = TRUNK_AUX_SPEC.key, .file = "model-trunk-00002-of-00002.safetensors" };
    at += 1;
    entries[at] = .{ .key = VISION_SPEC.key, .file = "model-vision.safetensors" };
    try writeIndex(io, allocator, dir, entries[0 .. at + 1]);
}

fn createFixture(
    io: std.Io,
    allocator: std.mem.Allocator,
    tmp: *std.testing.TmpDir,
) ![]u8 {
    try tmp.dir.createDirPath(io, "flat-exl3");
    try tmp.dir.createDirPath(io, "external-donor");
    var donor_dir = try tmp.dir.openDir(io, "external-donor", .{});
    defer donor_dir.close(io);
    var dir = try tmp.dir.openDir(io, "flat-exl3", .{});
    defer dir.close(io);
    try writeFlatFixture(io, allocator, dir, donor_dir);

    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    return std.fmt.allocPrint(allocator, "{s}/flat-exl3", .{root_buf[0..root_len]});
}

test "flat EXL3 semantic shards remain a complete discovery candidate" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    const model_path = try createFixture(io, allocator, &tmp);
    defer allocator.free(model_path);

    try std.testing.expectEqual(
        expert_quant.Layout.exl3_k4,
        model_discovery.qwen4StreamingIndexComplete(io, allocator, model_path).?,
    );

    var model_dir = try std.Io.Dir.openDirAbsolute(io, model_path, .{});
    defer model_dir.close(io);
    var shards = model_discovery.indexShardSet(io, model_dir).?;
    defer model_discovery.freeShardSet(&shards);
    try std.testing.expect(shards.contains("model-experts-L00.safetensors"));
    try std.testing.expect(shards.contains("model-embed.safetensors"));
    try std.testing.expect(shards.contains("model-trunk-00001-of-00002.safetensors"));
    try std.testing.expect(shards.contains("model-trunk-00002-of-00002.safetensors"));
    try std.testing.expect(shards.contains("model-lm-head.safetensors"));
    try std.testing.expect(shards.contains("model-mtp.safetensors"));
    try std.testing.expect(shards.contains("model-vision.safetensors"));
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &root_buf);
    const donor_path = try std.fmt.allocPrint(allocator, "{s}/external-donor", .{root_buf[0..root_len]});
    defer allocator.free(donor_path);
    const donor_stat = blk: {
        var donor_dir = try std.Io.Dir.openDirAbsolute(io, donor_path, .{});
        defer donor_dir.close(io);
        break :blk try donor_dir.statFile(io, "trunk-donor.safetensors", .{});
    };
    const canonical_stat = try model_dir.statFile(io, "model-trunk-00002-of-00002.safetensors", .{});
    try std.testing.expectEqual(donor_stat.inode, canonical_stat.inode);

    var result = try model_discovery.discoverModels(io, allocator, root_buf[0..root_len]);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.models.len);
    try std.testing.expectEqualStrings("flat-exl3", result.models[0].id);
    try std.testing.expectEqualStrings("qwen4_exp", result.models[0].model_type);
    try std.testing.expect(result.models[0].streaming_index_complete);
}

test "flat EXL3 resident split drops co-located routed banks and isolates MTP" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    const model_path = try createFixture(io, allocator, &tmp);
    defer allocator.free(model_path);

    const split = try model.streamingResidentSplit(io, allocator, model_path, &.{ .expert_layout = .exl3_k4 });
    // embed (64) + lm_head (1024) + two trunk tensors (2048); the nine routed EXL3
    // tensors and model.visual are not trunk bytes. A kept head bills its own routed tensor.
    try std.testing.expectEqual(@as(u64, 3136), split.trunk);
    try std.testing.expectEqual(@as(u64, 1408), split.mtp);
    try std.testing.expectEqual(@as(u64, 4), split.vision);
    for ([_]bool{ false, true }) |vision| {
        var weights = try model.loadWeightsStreaming(io, allocator, model_path, .exl3_k4, vision, false);
        defer weights.deinit();
        try std.testing.expectEqual(vision, weights.get("model.visual.fake") != null);
    }
    // The split bills a kept head's routed bank, so the loader must keep exactly it.
    const head_experts = "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.trellis";
    for ([_]bool{ false, true }) |keep| {
        var weights = try model.loadWeightsStreaming(io, allocator, model_path, .exl3_k4, false, keep);
        defer weights.deinit();
        try std.testing.expectEqual(keep, weights.get(head_experts) != null);
    }
}

test "Sushi quant memory bills resident routed banks and enabled vision and MTP exactly" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try createFixture(io, a, &tmp);
    defer a.free(path);
    // Text trunk (3136) plus all three resident expert projections (1152).
    const text = try model.qwenResidentWeightBytes(io, a, path, false, false);
    try std.testing.expectEqual(@as(u64, 4288), text.trunk);
    try std.testing.expectEqual(@as(u64, 0), text.mtp);
    const all = try model.qwenResidentWeightBytes(io, a, path, true, true);
    try std.testing.expectEqual(text.trunk + 4, all.trunk);
    try std.testing.expectEqual(@as(u64, 1408), all.mtp);
    // A duplicate of an indexed tensor in another indexed shard must not double its bill.
    var dir = try std.Io.Dir.openDirAbsolute(io, path, .{});
    defer dir.close(io);
    try writeSafetensors(io, a, dir, "model-vision.safetensors", &.{ VISION_SPEC, TRUNK_SPEC });
    const duplicate = try model.qwenResidentWeightBytes(io, a, path, true, true);
    try std.testing.expectEqual(all, duplicate);
}

const BankGroup = struct { experts: u64, n: u64, group: ?u32 };

/// The nine tensors of one layer's routed banks; a group is named `.g<N>`, a null group is the uniform bank.
fn appendBankSpecs(aa: std.mem.Allocator, specs: *std.ArrayList(TensorSpec), entries: *std.ArrayList(IndexSpec), layer: u32, groups: []const BankGroup, hidden: u64, inter: u64) !void {
    const proj = [3][]const u8{ "gate", "up", "down" };
    const part = [3][]const u8{ "trellis", "suh", "svh" };
    for (groups) |g| for (0..9) |ci| {
        const pi = ci / 3;
        const input: u64 = if (pi == 2) inter else hidden;
        const output: u64 = if (pi == 2) hidden else inter;
        const weight = ci % 3 == 0;
        const shape = try aa.alloc(u64, if (weight) 4 else 2);
        if (weight) @memcpy(shape, &[_]u64{ g.experts, input / 16, output / 16, g.n }) else @memcpy(shape, &[_]u64{ g.experts, if (ci % 3 == 1) input else output });
        var elements: u64 = 1;
        for (shape) |d| elements *= d;
        const key = if (g.group) |gi|
            try std.fmt.allocPrint(aa, "language_model.model.layers.{d}.mlp.switch_mlp.{s}_proj.g{d}.{s}", .{ layer, proj[pi], gi, part[ci % 3] })
        else
            try std.fmt.allocPrint(aa, "language_model.model.layers.{d}.mlp.switch_mlp.{s}_proj.{s}", .{ layer, proj[pi], part[ci % 3] });
        try specs.append(aa, .{ .key = key, .dtype = if (weight) "U16" else "F16", .shape = shape, .bytes = @intCast(elements * 2) });
        try entries.append(aa, .{ .key = key, .file = "experts.safetensors" });
    };
}

test "Sushi quant memory bills a rate-grouped layer's per-expert table" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var specs: std.ArrayList(TensorSpec) = .empty;
    var entries: std.ArrayList(IndexSpec) = .empty;
    try appendBankSpecs(arena.allocator(), &specs, &entries, 0, &.{ .{ .experts = 2, .n = 32, .group = 0 }, .{ .experts = 4, .n = 48, .group = 1 } }, 128, 256);
    try writeSafetensors(io, a, tmp.dir, "experts.safetensors", specs.items);
    try writeIndex(io, a, tmp.dir, entries.items);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    var stored: u64 = 0;
    for (specs.items) |spec| stored += spec.bytes;
    const bill = try model.qwenResidentWeightBytes(io, a, path, false, false);
    try std.testing.expectEqual(stored + 6 * @import("sushi_exl3").rate_groups.META_WORDS * 4, bill.trunk);
}

test "flat EXL3 routed-key filtering is independent of shard filename" {
    var key_buf: [256]u8 = undefined;
    for (EXL3_SPECS) |spec| {
        try std.testing.expect(
            model.qwen4StreamingWeightKey(.exl3_k4, &key_buf, spec.key) == null,
        );
    }
    try std.testing.expectEqualStrings(
        EMBED_SPEC.key,
        model.qwen4StreamingWeightKey(.exl3_k4, &key_buf, EMBED_SPEC.key).?,
    );
    try std.testing.expectEqualStrings(
        "language_model.mtp.fc_hidden.weight",
        model.qwen4StreamingWeightKey(.exl3_k4, &key_buf, "language_model.mtp.fc_hidden.weight").?,
    );
}

test "EXL3 streaming CPU config engages only with a budget and refuses MTP" {
    const stream = @import("expert_stream.zig");
    const cfg = model.ModelConfig{
        .model_type = "qwen4_exp",
        .num_hidden_layers = 48,
        .num_experts = 512,
        .num_experts_per_tok = 10,
        .hidden_size = 2560,
        .moe_intermediate_size = 640,
        .expert_layout = .exl3_k4,
        .quant_bits = 8,
    };
    try std.testing.expect(cfg.supportsExpertStreaming());
    try std.testing.expect(!cfg.expertStreamingRequired());
    try std.testing.expect(!stream.expertStreamingEngaged(cfg.supportsExpertStreaming(), cfg.expertStreamingRequired(), 0));
    try std.testing.expect(stream.expertStreamingEngaged(cfg.supportsExpertStreaming(), cfg.expertStreamingRequired(), 20 << 30));
    try std.testing.expect(stream.mtpRefusal(true, true, false) != null);
    try std.testing.expectEqual(stream.MtpUnderStreaming.keep, stream.mtpUnderStreaming(true, false, false, true));
}

test "EXL3 streaming CPU store spans all nine tensors with exact source bytes" {
    const stream = @import("expert_stream.zig");
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try createFixture(io, a, &tmp);
    defer a.free(path);
    var store = try stream.ExpertStore.openLayout(a, path, .{ .layers = 1, .experts = 2, .hidden = 16, .intermediate = 16 }, .exl3_k4);
    defer store.deinit();
    try std.testing.expectEqual(@as(usize, 9), store.componentCount());
    try std.testing.expectEqual(@as(u64, 576), store.perExpertBytes());
    var offset: usize = 0;
    for (EXL3_SPECS, 0..) |spec, ci| {
        const slab = store.slabSpec(ci);
        try std.testing.expectEqual(@as(u8, 2), slab.elem_bytes);
        try std.testing.expectEqual(spec.bytes / 2, slab.slot_bytes);
        for (0..2) |expert| {
            const span = store.spanAt(0, @intCast(expert), ci);
            const bytes = try a.alloc(u8, @intCast(span.len));
            defer a.free(bytes);
            try stream.readExact(store.fdAt(span.file), bytes, span.offset);
            for (bytes, 0..) |byte, j| try std.testing.expectEqual(@as(u8, @intCast((offset + expert * bytes.len + j + 17) % 251)), byte);
        }
        offset += spec.bytes;
    }
}

test "EXL3 streaming CPU admits a table bank only at a rate its codebook table has a row for" {
    const stream = @import("expert_stream.zig");
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const path = try createFixture(io, a, &tmp);
    defer a.free(path);
    const geometry: stream.Geometry = .{ .layers = 1, .experts = 2, .hidden = 16, .intermediate = 16, .exl3_rates = @as(u128, 1) << 16 };
    try std.testing.expectError(error.Exl3CodebookTableRateMissing, stream.ExpertStore.openLayout(a, path, geometry, .exl3_k4));
    var named = geometry;
    named.exl3_rates.? |= @as(u128, 1) << 32;
    var store = try stream.ExpertStore.openLayout(a, path, named, .exl3_k4);
    store.deinit();
}

test "EXL3 streaming CPU Sushi geometry ledger at 20 GiB bills every bank" {
    const stream = @import("expert_stream.zig");
    for ([_]u32{ 32, 42, 48, 64 }) |n| {
        const per = try stream.exl3ExpertBytes(.{ .layers = 48, .experts = 512, .hidden = 2560, .intermediate = 640, .exl3_n = n });
        try std.testing.expectEqual(@as(u64, 38400) * n + 19200, per);
        const ledger = try stream.budgetLedger(20 << 30, 6 << 30, 0, 0, 48, 512, 10, per, stream.BOUNCE_BYTES, .{});
        const fixed = (6 << 30) + 512 * per + 10 * per + stream.BOUNCE_BYTES;
        try std.testing.expectEqual(@min(512, ((20 << 30) - fixed) / (48 * per)), ledger.slots_per_layer);
        try std.testing.expect(fixed + ledger.cache_bytes <= 20 << 30);
        try std.testing.expect(fixed + ledger.cache_bytes + 48 * per > 20 << 30);
        var slab_pages: u64 = 0;
        for ([_]u64{ 2560 * 640 / 256 * n * 2, 2560 * 2, 640 * 2 }) |component| {
            slab_pages += 3 * (48 * std.mem.alignForward(u64, component * ledger.slots_per_layer, 16384) + std.mem.alignForward(u64, component * 512, 16384));
        }
        const spans = @sizeOf(stream.SourceSpan) * 48 * 512 * 9;
        const cache_metadata = 48 * (512 * @sizeOf(i32) + @as(u64, ledger.slots_per_layer) * (@sizeOf(u16) + @sizeOf(u64) + @sizeOf(bool)));
        const actual_bounce = stream.FILL_WORKERS * 64 * 1024 * 1024;
        try std.testing.expect((6 << 30) + slab_pages + spans + cache_metadata + actual_bounce + ledger.selected_bytes <= 20 << 30);
    }
}

test "EXL3 streaming CPU refuses invalid trellis geometry, and grouped banks outside Flash-Next" {
    const stream = @import("expert_stream.zig");
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(std.testing.io, &path_buf);
    const path = path_buf[0..path_len];
    const geometry: stream.Geometry = .{ .layers = 1, .experts = 2, .hidden = 16, .intermediate = 16, .exl3_n = 32 };
    const grouped = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.g0.trellis";
    try writeIndex(std.testing.io, a, tmp.dir, &.{.{ .key = grouped, .file = "experts.safetensors" }});
    try std.testing.expectError(error.Exl3RateGroupsStreamingUnsupported, expert_quant.streamingLayoutOfDir(a, std.testing.io, "mimo_v2", path, 1, 0));
    var entries: [9]IndexSpec = undefined;
    for (EXL3_SPECS, 0..) |spec, ci| entries[ci] = .{ .key = spec.key, .file = "experts.safetensors" };
    try writeIndex(std.testing.io, a, tmp.dir, &entries);
    try writeSafetensors(std.testing.io, a, tmp.dir, "experts.safetensors", &EXL3_SPECS);
    try std.testing.expectError(error.Exl3TrellisGeometry, stream.ExpertStore.openLayout(a, path, geometry, .exl3_k4));
}

const GroupSpec = struct { experts: u32, n: [3]u32 };

const GroupedPack = struct {
    allocator: std.mem.Allocator,
    /// Every tensor's words, group-major then component (trellis, suh, svh per projection).
    words: [][]u16,

    fn deinit(self: *GroupedPack) void {
        for (self.words) |w| self.allocator.free(w);
        self.allocator.free(self.words);
    }
};

/// Layer 0 of a rate-grouped pack with deterministic, distinct words per tensor.
fn writeGroupedPack(dir: std.Io.Dir, a: std.mem.Allocator, hidden: u32, inter: u32, groups: []const GroupSpec) !GroupedPack {
    const io = std.testing.io;
    const proj = [3][]const u8{ "gate", "up", "down" };
    const part = [3][]const u8{ "trellis", "suh", "svh" };
    var pack: GroupedPack = .{ .allocator = a, .words = try a.alloc([]u16, groups.len * 9) };
    var made: usize = 0;
    errdefer {
        for (pack.words[0..made]) |w| a.free(w);
        a.free(pack.words);
    }
    const specs = try a.alloc(TensorSpec, groups.len * 9);
    defer a.free(specs);
    const entries = try a.alloc(IndexSpec, groups.len * 9);
    defer a.free(entries);
    const shapes = try a.alloc([4]u64, groups.len * 9);
    defer a.free(shapes);
    const keys = try a.alloc([128]u8, groups.len * 9);
    defer a.free(keys);
    var total: usize = 0;
    for (groups, 0..) |group, g| for (0..9) |ci| {
        const pi = ci / 3;
        const input: u64 = if (pi == 2) inter else hidden;
        const output: u64 = if (pi == 2) hidden else inter;
        const at = g * 9 + ci;
        const weight = ci % 3 == 0;
        shapes[at] = if (weight) .{ group.experts, input / 16, output / 16, group.n[pi] } else .{ group.experts, if (ci % 3 == 1) input else output, 0, 0 };
        const rank: usize = if (weight) 4 else 2;
        var elements: usize = 1;
        for (shapes[at][0..rank]) |d| elements *= @intCast(d);
        const key = try std.fmt.bufPrint(&keys[at], "language_model.model.layers.0.mlp.switch_mlp.{s}_proj.g{d}.{s}", .{ proj[pi], g, part[ci % 3] });
        specs[at] = .{ .key = key, .dtype = if (weight) "U16" else "F16", .shape = shapes[at][0..rank], .bytes = elements * 2 };
        entries[at] = .{ .key = key, .file = "experts.safetensors" };
        pack.words[at] = try a.alloc(u16, elements);
        made += 1;
        for (pack.words[at], 0..) |*word, j| {
            word.* = if (weight) @truncate(j *% 3571 +% ci *% 997 +% g *% 131 +% group.n[pi]) else @import("sushi_exl3").format.f32ToF16Bits(if ((j + ci + g) % 3 == 0) -0.25 else 0.375);
        }
        total += elements * 2;
    };
    try writeSafetensors(io, a, dir, "experts.safetensors", specs);
    try writeIndex(io, a, dir, entries);
    const file = try dir.readFileAlloc(io, "experts.safetensors", a, .limited(1 << 26));
    defer a.free(file);
    var at: usize = 8 + @as(usize, @intCast(std.mem.readInt(u64, file[0..8], .little)));
    for (pack.words) |words| {
        for (words, 0..) |word, j| std.mem.writeInt(u16, file[at + j * 2 ..][0..2], word, .little);
        at += words.len * 2;
    }
    try dir.writeFile(io, .{ .sub_path = "experts.safetensors", .data = file });
    return pack;
}

test "EXL3 streaming CPU reads a rate-grouped layer as per-expert spans and bills its average expert" {
    const stream = @import("expert_stream.zig");
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const hidden = 128;
    const inter = 256;
    const groups = [_]GroupSpec{ .{ .experts = 2, .n = .{ 32, 32, 32 } }, .{ .experts = 4, .n = .{ 48, 48, 64 } } };
    var pack = try writeGroupedPack(tmp.dir, a, hidden, inter, &groups);
    defer pack.deinit();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(std.testing.io, &path_buf)];
    try std.testing.expectEqual(expert_quant.Layout.exl3_k4, try expert_quant.streamingLayoutOfDir(a, std.testing.io, "qwen4_exp", path, 1, 0));
    try std.testing.expectError(error.Exl3RateGroupsStreamingUnsupported, expert_quant.streamingLayoutOfDir(a, std.testing.io, "glm5_next", path, 1, 0));
    const geometry: stream.Geometry = .{ .layers = 1, .experts = 6, .hidden = hidden, .intermediate = inter, .exl3_n = 64 };
    var store = try stream.ExpertStore.openLayout(a, path, geometry, .exl3_k4);
    defer store.deinit();

    const layer_groups = store.quantized.?.groupsOf(0) orelse return error.TestExpectedGroups;
    try std.testing.expectEqual(@as(u8, 2), layer_groups.count);
    try std.testing.expectEqualSlices(u16, &.{ 2, 4 }, layer_groups.experts[0..2]);
    try std.testing.expectEqualSlices(u16, &.{ 0, 2 }, layer_groups.base[0..2]);
    try std.testing.expectEqualSlices(u32, &.{ 48, 48, 64 }, &layer_groups.n[1]);
    try std.testing.expectEqual(@as(u8, 1), layer_groups.groupOf(2));

    for (0..6) |expert| {
        const g: usize = if (expert < 2) 0 else 1;
        const local = expert - layer_groups.base[g];
        for (0..9) |ci| {
            const words = pack.words[g * 9 + ci];
            const per = words.len / groups[g].experts;
            const span = store.spanAt(0, @intCast(expert), ci);
            try std.testing.expectEqual(@as(u64, per * 2), span.len);
            const bytes = try a.alloc(u8, @intCast(span.len));
            defer a.free(bytes);
            try stream.readExact(store.fdAt(span.file), bytes, span.offset);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(words[local * per ..][0..per]), bytes);
        }
    }

    // Plus the slot's `Ragged` meta row.
    var expected: u64 = @import("sushi_exl3").rate_groups.META_WORDS * 4;
    for (0..3) |pi| {
        const tiles: u64 = (hidden / 16) * (inter / 16);
        var total: u64 = 0;
        for (groups) |group| total += group.experts * tiles * group.n[pi] * 2;
        expected += (total + 5) / 6 + 2 * (hidden + inter);
    }
    try std.testing.expectEqual(expected, store.perExpertBytes());
}

fn gpuExl3Bank(arrays: [9]@import("mlx.zig").mlx_array) @import("sushi_exl3").Bank {
    return .{
        .gate = .{ .trellis = arrays[0], .suh = arrays[1], .svh = arrays[2] },
        .up = .{ .trellis = arrays[3], .suh = arrays[4], .svh = arrays[5] },
        .down = .{ .trellis = arrays[6], .suh = arrays[7], .svh = arrays[8] },
    };
}

test "EXL3 streaming GPU slab matches resident decode prefill hits misses and union" {
    const mlx = @import("mlx.zig");
    const exl3 = @import("sushi_exl3");
    const stream = @import("expert_stream.zig");
    const a = std.testing.allocator;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const experts = 6;
    const hidden = 128;
    const intermediate = 256;
    for ([_]u32{ 32, 40, 42, 48, 64 }) |n| {
        for ([_]mlx.mlx_dtype{ .float16, .bfloat16 }) |dtype| {
            var tmp = std.testing.tmpDir(.{});
            defer tmp.cleanup();
            var keys: [9][192]u8 = undefined;
            var shapes: [9][4]u64 = undefined;
            var specs: [9]TensorSpec = undefined;
            var entries: [9]IndexSpec = undefined;
            var resident: [9]mlx.mlx_array = @splat(.{});
            defer for (resident) |arr| {
                if (arr.ctx != null) _ = mlx.mlx_array_free(arr);
            };
            for (0..9) |ci| {
                const c: expert_quant.Component = @fromBackingInt(@intCast(ci));
                const input: u32 = if (expert_quant.projectionOf(c) == .down) intermediate else hidden;
                const output: u32 = if (expert_quant.projectionOf(c) == .down) hidden else intermediate;
                const weight = expert_quant.partOf(c) == .weight;
                const rank: usize = if (weight) 4 else 2;
                shapes[ci] = if (weight) .{ experts, input / 16, output / 16, n } else .{ experts, if (expert_quant.partOf(c) == .scales) input else output, 0, 0 };
                var elements: usize = 1;
                for (shapes[ci][0..rank]) |d| elements *= @intCast(d);
                const key = try expert_quant.exl3TensorKey(&keys[ci], 0, c);
                specs[ci] = .{ .key = key, .dtype = if (weight) "U16" else "F16", .shape = shapes[ci][0..rank], .bytes = elements * 2 };
                entries[ci] = .{ .key = key, .file = "experts.safetensors" };
            }
            try writeSafetensors(io, a, tmp.dir, "experts.safetensors", &specs);
            try writeIndex(io, a, tmp.dir, &entries);
            const file = try tmp.dir.readFileAlloc(io, "experts.safetensors", a, .limited(1 << 24));
            defer a.free(file);
            var at: usize = 8 + @as(usize, @intCast(std.mem.readInt(u64, file[0..8], .little)));
            for (specs, 0..) |spec, ci| {
                const words = try a.alloc(u16, spec.bytes / 2);
                defer a.free(words);
                for (words, 0..) |*word, j| {
                    word.* = if (ci % 3 == 0) @truncate(j *% 3571 +% ci *% 997 +% n) else exl3.format.f32ToF16Bits(if ((j + ci) % 3 == 0) -0.25 else 0.375);
                    std.mem.writeInt(u16, file[at + j * 2 ..][0..2], word.*, .little);
                }
                var shape: [4]c_int = undefined;
                for (spec.shape, 0..) |d, j| shape[j] = @intCast(d);
                resident[ci] = mlx.mlx_array_new_data(words.ptr, &shape, @intCast(spec.shape.len), if (ci % 3 == 0) .uint16 else .float16);
                at += spec.bytes;
            }
            try tmp.dir.writeFile(io, .{ .sub_path = "experts.safetensors", .data = file });
            var path_buf: [std.fs.max_path_bytes]u8 = undefined;
            const len = try tmp.dir.realPath(io, &path_buf);
            const geom: stream.Geometry = .{ .layers = 1, .experts = experts, .hidden = hidden, .intermediate = intermediate, .exl3_n = n };
            const per = try stream.exl3ExpertBytes(geom);
            var engine = try stream.Engine.initWithOptions(a, path_buf[0..len], geom, 3 * per, s, .{ .layout = .exl3_k4, .bounce_size = 1 << 20 });
            defer engine.deinit();
            try engine.warmCache();
            try std.testing.expectEqual(@as(u64, 0), engine.fallback_imports);
            for ([_]usize{ 1, 1, 2, 33, 65, 1 }) |rows| {
                const topk: usize = if (rows == 1 and engine.layers[0].stats.groups > 4) 4 else 2;
                const routes = try a.alloc(u16, rows * topk);
                defer a.free(routes);
                for (routes, 0..) |*id, j| id.* = if (rows <= 2 and topk == 2) ([_]u16{ 5, 1, 4, 1 })[j % 4] else @intCast((j * 5 + 2) % experts);
                var seen: [experts]bool = @splat(false);
                var expected_misses: u64 = 0;
                for (routes) |id| {
                    if (seen[id]) continue;
                    seen[id] = true;
                    const slot = engine.layers[0].cache.expert_to_slot[id];
                    if (slot < 0 or !engine.slotReady(0, @intCast(slot))) expected_misses += 1;
                }
                const before = engine.fill_experts_total;
                var prepared = try engine.prepareHost(0, routes);
                defer prepared.deinit();
                try std.testing.expectEqual(expected_misses, engine.fill_experts_total - before);
                try std.testing.expectEqual(rows > 2 or topk > 3, prepared.workspace);
                var slab: [9]mlx.mlx_array = undefined;
                for (&slab, 0..) |*arr, ci| {
                    arr.* = prepared.quantOperand(@fromBackingInt(@intCast(ci)));
                    const actual = mlx.getShape(arr.*);
                    try std.testing.expectEqual(specs[ci].shape.len, actual.len);
                    try std.testing.expectEqual(if (ci % 3 == 0) mlx.mlx_dtype.uint16 else mlx.mlx_dtype.float16, mlx.mlx_array_dtype(arr.*));
                }
                const xh = try a.alloc(u16, rows * hidden);
                defer a.free(xh);
                for (xh, 0..) |*v, j| {
                    const value = @as(f32, @floatFromInt(@as(i32, @intCast(j % 31)) - 15)) / 32;
                    v.* = if (dtype == .float16) exl3.format.f32ToF16Bits(value) else @truncate(@as(u32, @bitCast(value)) >> 16);
                }
                const scores = try a.alloc(f32, routes.len);
                defer a.free(scores);
                for (scores, 0..) |*v, j| v.* = @as(f32, @floatFromInt(j % topk + 1)) / 16;
                const route_shape = [_]c_int{ 1, @intCast(rows), @intCast(topk) };
                const x = mlx.mlx_array_new_data(xh.ptr, &.{ 1, @intCast(rows), hidden }, 3, dtype);
                defer _ = mlx.mlx_array_free(x);
                const weights = mlx.mlx_array_new_data(scores.ptr, &route_shape, 3, .float32);
                defer _ = mlx.mlx_array_free(weights);
                const ids = mlx.mlx_array_new_data(routes.ptr, &route_shape, 3, .uint16);
                defer _ = mlx.mlx_array_free(ids);
                const local = mlx.mlx_array_new_data(prepared.remapped.ptr, &route_shape, 3, .uint16);
                defer _ = mlx.mlx_array_free(local);
                const dec: exl3.format.Decode = .{ .codebook = .mcg, .window = .w12 };
                const reference = try exl3.moe(s, x, gpuExl3Bank(resident), ids, weights, dec, false);
                defer _ = mlx.mlx_array_free(reference);
                const actual = try exl3.moe(s, x, gpuExl3Bank(slab), local, weights, dec, false);
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_array_eval(reference));
                try mlx.check(mlx.mlx_array_eval(actual));
                try std.testing.expectEqual(dtype, mlx.mlx_array_dtype(actual));
                const ref_words: [*]const u16 = if (dtype == .float16) @ptrCast(mlx.mlx_array_data_float16(reference) orelse return error.UnreadableExl3Output) else mlx.mlx_array_data_bfloat16(reference) orelse return error.UnreadableExl3Output;
                const actual_words: [*]const u16 = if (dtype == .float16) @ptrCast(mlx.mlx_array_data_float16(actual) orelse return error.UnreadableExl3Output) else mlx.mlx_array_data_bfloat16(actual) orelse return error.UnreadableExl3Output;
                for (ref_words[0 .. rows * hidden]) |word| try std.testing.expect(std.math.isFinite(if (dtype == .float16) exl3.format.f16BitsToF32(word) else @as(f32, @bitCast(@as(u32, word) << 16))));
                try std.testing.expectEqualSlices(u16, ref_words[0 .. rows * hidden], actual_words[0 .. rows * hidden]);
            }
        }
    }
}

test "EXL3 streaming GPU rate-grouped layer matches the resident ragged bank across hits misses and union" {
    const mlx = @import("mlx.zig");
    const exl3 = @import("sushi_exl3");
    const stream = @import("expert_stream.zig");
    const a = std.testing.allocator;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const hidden = 128;
    const inter = 256;
    const experts = 6;
    const groups = [_]GroupSpec{ .{ .experts = 2, .n = .{ 32, 32, 32 } }, .{ .experts = 4, .n = .{ 48, 40, 64 } } };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var pack = try writeGroupedPack(tmp.dir, a, hidden, inter, &groups);
    defer pack.deinit();

    var arrays: [groups.len * 9]mlx.mlx_array = @splat(.{});
    defer for (arrays) |arr| {
        if (arr.ctx != null) _ = mlx.mlx_array_free(arr);
    };
    var banks: [groups.len]exl3.Bank = undefined;
    for (groups, 0..) |group, g| {
        for (0..9) |ci| {
            const pi = ci / 3;
            const input: c_int = if (pi == 2) inter else hidden;
            const output: c_int = if (pi == 2) hidden else inter;
            const e: c_int = @intCast(group.experts);
            const shape4 = [_]c_int{ e, @divExact(input, 16), @divExact(output, 16), @intCast(group.n[pi]) };
            const shape2 = [_]c_int{ e, if (ci % 3 == 1) input else output };
            const words = pack.words[g * 9 + ci];
            arrays[g * 9 + ci] = if (ci % 3 == 0)
                mlx.mlx_array_new_data(words.ptr, &shape4, 4, .uint16)
            else
                mlx.mlx_array_new_data(words.ptr, &shape2, 2, .float16);
        }
        banks[g] = gpuExl3Bank(arrays[g * 9 ..][0..9].*);
    }
    var resident = try exl3.Ragged.init(s, &banks);
    defer resident.deinit();

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const geom: stream.Geometry = .{ .layers = 1, .experts = experts, .hidden = hidden, .intermediate = inter, .exl3_n = 64 };
    const per = try stream.expertBytesFor(a, path, geom, .exl3_k4);
    for ([_]u64{ 3, 6 }) |layer_slots| {
        var engine = try stream.Engine.initWithOptions(a, path, geom, layer_slots * per, s, .{ .layout = .exl3_k4, .bounce_size = 1 << 20 });
        defer engine.deinit();
        try engine.warmCache();
        try std.testing.expectEqual(@as(u64, 0), engine.fallback_imports);
        const grouped = engine.layers[0].grouped orelse return error.TestExpectedGroupedLayer;
        var saw_union = false;
        var saw_cache = false;
        var saw_wide_union = false;
        for ([_]usize{ 1, 1, 2, 33, 65, 1, 1, 65 }) |rows| {
            const topk: usize = 2;
            const routes = try a.alloc(u16, rows * topk);
            defer a.free(routes);
            for (routes, 0..) |*id, j| id.* = if (rows <= 2) ([_]u16{ 5, 1, 4, 1, 0, 2 })[(j + rows) % 6] else @intCast((j * 5 + 2) % experts);

            var seen: [experts]bool = @splat(false);
            var distinct: [groups.len]u32 = @splat(0);
            var expected_misses: u64 = 0;
            for (routes) |id| {
                if (seen[id]) continue;
                seen[id] = true;
                const g = grouped.groups.groupOf(id);
                distinct[g] += 1;
                const slot = grouped.caches[g].expert_to_slot[id - grouped.groups.base[g]];
                if (slot < 0 or !grouped.caches[g].ready[@intCast(slot)]) expected_misses += 1;
            }
            var expect_union = false;
            for (distinct, grouped.slots[0..groups.len]) |count, slots| expect_union = expect_union or count > slots;

            // A route the prefill GEMMs serve reads an overflowing group's cached experts in place: no host copies.
            const wide = rows > exl3.kernels.DECODE_ROWS_MAX;
            const before = engine.fill_experts_total;
            const copied = engine.copy_bytes_total;
            var prepared = try if (wide) engine.prepareHostWide(0, routes) else engine.prepareHost(0, routes);
            defer prepared.deinit();
            try std.testing.expectEqual(expected_misses, engine.fill_experts_total - before);
            try std.testing.expectEqual(expect_union, prepared.workspace);
            if (wide) try std.testing.expectEqual(copied, engine.copy_bytes_total);
            saw_wide_union = saw_wide_union or (wide and expect_union);
            saw_union = saw_union or expect_union;
            saw_cache = saw_cache or !expect_union;
            const bank = prepared.ragged orelse return error.TestExpectedRaggedBank;

            const xh = try a.alloc(u16, rows * hidden);
            defer a.free(xh);
            for (xh, 0..) |*v, j| v.* = exl3.format.f32ToF16Bits(@as(f32, @floatFromInt(@as(i32, @intCast(j % 31)) - 15)) / 32);
            const scores = try a.alloc(f32, routes.len);
            defer a.free(scores);
            for (scores, 0..) |*v, j| v.* = @as(f32, @floatFromInt(j % topk + 1)) / 16;
            const route_shape = [_]c_int{ 1, @intCast(rows), @intCast(topk) };
            const x = mlx.mlx_array_new_data(xh.ptr, &.{ 1, @intCast(rows), hidden }, 3, .float16);
            defer _ = mlx.mlx_array_free(x);
            const weights = mlx.mlx_array_new_data(scores.ptr, &route_shape, 3, .float32);
            defer _ = mlx.mlx_array_free(weights);
            const ids = mlx.mlx_array_new_data(routes.ptr, &route_shape, 3, .uint16);
            defer _ = mlx.mlx_array_free(ids);
            const local = mlx.mlx_array_new_data(prepared.remapped.ptr, &route_shape, 3, .uint16);
            defer _ = mlx.mlx_array_free(local);
            const dec: exl3.format.Decode = .{ .codebook = .mcg, .window = .w12 };
            const reference = try exl3.moeRagged(s, x, &resident, ids, weights, dec, false, null, null);
            defer _ = mlx.mlx_array_free(reference);
            const actual = try exl3.moeRagged(s, x, bank, local, weights, dec, false, null, null);
            defer _ = mlx.mlx_array_free(actual);
            try mlx.check(mlx.mlx_array_eval(reference));
            try mlx.check(mlx.mlx_array_eval(actual));
            const ref_words: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(reference) orelse return error.UnreadableExl3Output);
            const actual_words: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(actual) orelse return error.UnreadableExl3Output);
            for (ref_words[0 .. rows * hidden]) |word| try std.testing.expect(std.math.isFinite(exl3.format.f16BitsToF32(word)));
            try std.testing.expectEqualSlices(u16, ref_words[0 .. rows * hidden], actual_words[0 .. rows * hidden]);
        }
        try std.testing.expect(saw_cache);
        try std.testing.expectEqual(layer_slots < experts, saw_union);
        try std.testing.expectEqual(layer_slots < experts, saw_wide_union);
    }
}

test "EXL3 streaming CPU refusals survive the registry and HTTP boundary" {
    const registry = @import("model_registry.zig").ModelRegistry;
    const server = @import("server.zig");
    inline for (.{ error.Exl3RateGroupsStreamingUnsupported, error.Exl3GateUpRateMismatch }) |err| {
        try std.testing.expectEqual(err, registry.loadErrorFromName(@errorName(err)));
        const refusal = server.loadRefusalFor(err) orelse return error.MissingNamedRefusal;
        try std.testing.expect(refusal.type.len > 0 and refusal.message.len > 0);
    }
}

test "EXL3 streaming CPU rejects malformed banks and per-layer gate up rate mismatches" {
    const stream = @import("expert_stream.zig");
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const len = try tmp.dir.realPath(io, &path_buf);
    const path = path_buf[0..len];
    const geometry: stream.Geometry = .{ .layers = 1, .experts = 2, .hidden = 16, .intermediate = 16 };
    var entries: [18]IndexSpec = undefined;
    var specs: [18]TensorSpec = undefined;
    var keys: [9][192]u8 = undefined;
    for (0..9) |ci| {
        specs[ci] = EXL3_SPECS[ci];
        specs[ci + 9] = EXL3_SPECS[ci];
        specs[ci + 9].key = try expert_quant.exl3TensorKey(&keys[ci], 1, @fromBackingInt(@intCast(ci)));
        entries[ci] = .{ .key = specs[ci].key, .file = "experts.safetensors" };
        entries[ci + 9] = .{ .key = specs[ci + 9].key, .file = "experts.safetensors" };
    }
    try writeIndex(io, a, tmp.dir, entries[0..9]);
    specs[1].dtype = "BF16";
    try writeSafetensors(io, a, tmp.dir, "experts.safetensors", specs[0..9]);
    try std.testing.expectError(error.Exl3TrellisGeometry, stream.ExpertStore.openLayout(a, path, geometry, .exl3_k4));
    specs[1] = EXL3_SPECS[1];
    const narrow = [_]u64{ 2, 1, 1, 32 };
    specs[3].shape = &narrow;
    specs[3].bytes = 128;
    try writeSafetensors(io, a, tmp.dir, "experts.safetensors", specs[0..9]);
    try std.testing.expectError(error.Exl3GateUpRateMismatch, stream.ExpertStore.openLayout(a, path, geometry, .exl3_k4));
    specs[3] = EXL3_SPECS[3];
    try writeSafetensors(io, a, tmp.dir, "experts.safetensors", specs[0..8]);
    try std.testing.expectError(error.MissingExpertTensor, stream.ExpertStore.openLayout(a, path, geometry, .exl3_k4));
    try writeIndex(io, a, tmp.dir, &entries);
    var two = geometry;
    two.layers = 2;
    specs[9].shape = &narrow;
    specs[9].bytes = 128;
    try writeSafetensors(io, a, tmp.dir, "experts.safetensors", &specs);
    try std.testing.expectError(error.Exl3GateUpRateMismatch, stream.ExpertStore.openLayout(a, path, two, .exl3_k4));
}

test "EXL3 streaming GPU mixed grouped and uniform layers stay within the workspaces the ledger bills" {
    const stream = @import("expert_stream.zig");
    const a = std.testing.allocator;
    const io = std.testing.io;
    const mlx = @import("mlx.zig");
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var specs: std.ArrayList(TensorSpec) = .empty;
    var entries: std.ArrayList(IndexSpec) = .empty;
    try appendBankSpecs(arena.allocator(), &specs, &entries, 0, &.{ .{ .experts = 2, .n = 32, .group = 0 }, .{ .experts = 4, .n = 48, .group = 1 } }, 128, 256);
    try appendBankSpecs(arena.allocator(), &specs, &entries, 1, &.{.{ .experts = 6, .n = 48, .group = null }}, 128, 256);
    try writeSafetensors(io, a, tmp.dir, "experts.safetensors", specs.items);
    try writeIndex(io, a, tmp.dir, entries.items);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const geom: stream.Geometry = .{ .layers = 2, .experts = 6, .hidden = 128, .intermediate = 256, .exl3_n = 64 };
    const bill = try stream.expertBillFor(a, path, geom, .exl3_k4);
    try std.testing.expectEqual(@as(u8, 2), bill.shape.union_sets);
    const ledger = try stream.budgetLedger(1 << 30, 0, 0, 0, 2, 6, 2, bill.expert_bytes, stream.BOUNCE_BYTES, bill.shape);
    var engine = try stream.Engine.initWithOptions(a, path, geom, 2 * 3 * bill.expert_bytes, mlx.gpuStream(), .{ .layout = .exl3_k4, .bounce_size = 1 << 20 });
    defer engine.deinit();
    var held: u64 = 0;
    for (engine.union_slabs) |op| held += @as(u64, op.stride) * op.count;
    for (engine.grouped_union) |op| held += @as(u64, op.stride) * op.count;
    try std.testing.expect(held <= ledger.workspace_bytes);
    try std.testing.expectEqual(ledger.workspace_bytes, engine.plan.workspace_bytes);
}

test "EXL3 streaming GPU a cache profile moves slots to the layer that reuses its experts, within the uniform bytes" {
    const stream = @import("expert_stream.zig");
    const a = std.testing.allocator;
    const io = std.testing.io;
    const mlx = @import("mlx.zig");
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var specs: std.ArrayList(TensorSpec) = .empty;
    var entries: std.ArrayList(IndexSpec) = .empty;
    const banks = [_]BankGroup{ .{ .experts = 2, .n = 32, .group = 0 }, .{ .experts = 4, .n = 48, .group = 1 } };
    for (0..2) |layer| try appendBankSpecs(arena.allocator(), &specs, &entries, @intCast(layer), &banks, 128, 256);
    try writeSafetensors(io, a, tmp.dir, "experts.safetensors", specs.items);
    try writeIndex(io, a, tmp.dir, entries.items);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const geom: stream.Geometry = .{ .layers = 2, .experts = 6, .hidden = 128, .intermediate = 256, .exl3_n = 64 };
    const bill = try stream.expertBillFor(a, path, geom, .exl3_k4);
    const uniform_bytes = 2 * 4 * bill.expert_bytes;

    var slots: [2][2]u32 = undefined;
    for (0..2) |arm| {
        // Layer 1 keeps hitting deep in its stack; layer 0 only on its hottest expert.
        if (arm == 1) try tmp.dir.writeFile(io, .{ .sub_path = stream.cache_profile_file, .data = "{\"version\":1,\"experts\":6,\"tokens\":100,\"source\":\"test\",\"layers\":[[90,0,0,0,0,0,10],[20,20,20,20,20,20,10]]}" });
        var engine = try stream.Engine.initWithOptions(a, path, geom, uniform_bytes, mlx.gpuStream(), .{ .layout = .exl3_k4, .bounce_size = 1 << 20 });
        defer engine.deinit();
        var held: u64 = 0;
        for (engine.layers, 0..) |layer, li| {
            const g = layer.grouped orelse return error.TestExpectedGroupedLayer;
            slots[arm][li] = g.cache_slots;
            for (0..g.groups.count) |gi| held += @as(u64, g.slots[gi]) * g.groups.expertBytes(gi, 128, 256);
        }
        try std.testing.expect(held <= uniform_bytes);
    }
    try std.testing.expectEqual(slots[0][0], slots[0][1]);
    try std.testing.expect(slots[1][1] > slots[0][1] and slots[1][0] < slots[0][0]);
}

test "EXL3 streaming CPU bills a grouped store's floor, one expert of every rate group, and one workspace set per layer kind" {
    const stream = @import("expert_stream.zig");
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var specs: std.ArrayList(TensorSpec) = .empty;
    var entries: std.ArrayList(IndexSpec) = .empty;
    try appendBankSpecs(arena.allocator(), &specs, &entries, 0, &.{ .{ .experts = 2, .n = 32, .group = 0 }, .{ .experts = 4, .n = 48, .group = 1 } }, 128, 256);
    try writeSafetensors(io, a, tmp.dir, "experts.safetensors", specs.items);
    try writeIndex(io, a, tmp.dir, entries.items);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const bill = try stream.expertBillFor(a, path, .{ .layers = 1, .experts = 6, .hidden = 128, .intermediate = 256, .exl3_n = 64 }, .exl3_k4);
    // Per group: three trellises of 128 tiles at the group's n, and three pairs of scale rows.
    try std.testing.expectEqual(@as(u64, 3 * 128 * 2 * (32 + 48) + 2 * 3 * (128 + 256) * 2), bill.shape.layer_floor_bytes);
    try std.testing.expectEqual(@as(u8, 1), bill.shape.union_sets);
    try std.testing.expectEqual(@as(u64, 6 * 32), bill.shape.union_meta_bytes);
}
