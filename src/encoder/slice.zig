const std = @import("std");
const source = @import("source.zig");
const pack = @import("pack.zig");

pub const Spec = struct {
    layers: usize,
    experts: usize,
    ngram_vocab_base: u64 = 4000,
    ngram_heads: usize = 16,
    ngram_divisor: u64 = 128,
};

fn isPrime(v: u64) bool {
    if (v < 2) return false;
    var d: u64 = 2;
    while (d * d <= v) : (d += 1) if (v % d == 0) return false;
    return true;
}

fn nextPrime(after: u64) u64 {
    var p = after + 1;
    while (!isPrime(p)) p += 1;
    return p;
}

pub const NgramPlan = struct {
    vocab: [16]i64,
    offsets: [16]i64,
    total_rows: u64,
};

pub fn ngramPlan(spec: Spec) NgramPlan {
    var plan = NgramPlan{ .vocab = @splat(0), .offsets = @splat(0), .total_rows = 0 };
    var total: u64 = 0;
    var p = spec.ngram_vocab_base - 1;
    for (0..spec.ngram_heads) |i| {
        p = nextPrime(p);
        plan.vocab[i] = @intCast(p);
        plan.offsets[i] = @intCast(total);
        total += p;
    }
    plan.total_rows = (total + spec.ngram_divisor - 1) / spec.ngram_divisor * spec.ngram_divisor;
    return plan;
}

fn keepKey(key: []const u8, layers: usize) bool {
    if (std.mem.startsWith(u8, key, "mtp.") or std.mem.startsWith(u8, key, "model.visual.")) return false;
    const marker = "model.language_model.layers.";
    if (!std.mem.startsWith(u8, key, marker)) return true;
    const rest = key[marker.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return false;
    const idx = std.fmt.parseInt(usize, rest[0..dot], 10) catch return false;
    return idx < layers;
}

pub fn sliceQwen(allocator: std.mem.Allocator, io: std.Io, src_dir: std.Io.Dir, src: *const source.SafetensorsSource, out: std.Io.Dir, spec: Spec) !void {
    var keys = std.ArrayList([]const u8).empty;
    defer keys.deinit(allocator);
    var it = src.tensors.keyIterator();
    while (it.next()) |k| if (keepKey(k.*, spec.layers)) try keys.append(allocator, k.*);
    std.mem.sortUnstable([]const u8, keys.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);

    const plan = ngramPlan(spec);
    var builder = pack.ShardBuilder.init(allocator);
    defer builder.deinit();
    var names = std.ArrayList([]u8).empty;
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }
    var entries = std.ArrayList(pack.IndexEntry).empty;
    defer {
        for (entries.items) |e| {
            allocator.free(e.tensor);
            allocator.free(e.file);
        }
        entries.deinit(allocator);
    }
    var shard_no: usize = 0;
    var file_buf: [64]u8 = undefined;

    for (keys.items) |key| {
        const loc = src.tensors.get(key).?;
        var shape = try allocator.dupe(usize, loc.shape);
        defer allocator.free(shape);
        var nbytes: usize = @intCast(loc.end - loc.start);
        var override: ?[]u8 = null;
        defer if (override) |o| allocator.free(o);
        if (std.mem.indexOf(u8, key, ".mlp.experts.") != null or std.mem.endsWith(u8, key, ".mlp.gate.weight")) {
            const per = nbytes / shape[0];
            shape[0] = spec.experts;
            nbytes = per * spec.experts;
        } else if (std.mem.indexOf(u8, key, "ngram_embedding.shard_") != null) {
            if (!std.mem.endsWith(u8, key, "shard_0.weight")) continue;
            const row_bytes = shape[1] * 2;
            shape[0] = @intCast(plan.total_rows);
            nbytes = row_bytes * @as(usize, @intCast(plan.total_rows));
        } else if (std.mem.endsWith(u8, key, "ngram_heads_offsets") or std.mem.endsWith(u8, key, "ngram_heads_vocab_sizes")) {
            const values = if (std.mem.endsWith(u8, key, "offsets")) plan.offsets else plan.vocab;
            override = try allocator.dupe(u8, std.mem.sliceAsBytes(&values));
            nbytes = override.?.len;
        }
        const raw = try allocator.alloc(u8, nbytes);
        defer allocator.free(raw);
        if (override) |o| @memcpy(raw, o) else try src.readRange(loc, 0, raw);
        try builder.add(key, loc.dtype, shape, raw);
        try names.append(allocator, try allocator.dupe(u8, key));
        if (builder.size() >= (1 << 30)) {
            const file = try std.fmt.bufPrint(&file_buf, "model-{d:0>5}.safetensors", .{shard_no});
            try builder.flush(io, out, file, &.{.{ "format", "pt" }});
            for (names.items) |n| try entries.append(allocator, .{ .tensor = n, .file = try allocator.dupe(u8, file) });
            names.clearRetainingCapacity();
            shard_no += 1;
        }
    }
    if (builder.count != 0) {
        const file = try std.fmt.bufPrint(&file_buf, "model-{d:0>5}.safetensors", .{shard_no});
        try builder.flush(io, out, file, &.{.{ "format", "pt" }});
        for (names.items) |n| try entries.append(allocator, .{ .tensor = n, .file = try allocator.dupe(u8, file) });
        names.clearRetainingCapacity();
    }
    var owned = std.ArrayList(pack.IndexEntry).empty;
    defer owned.deinit(allocator);
    try owned.appendSlice(allocator, entries.items);
    const index = try pack.buildIndex(allocator, owned.items, 0);
    defer allocator.free(index);
    try out.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });

    const cfg_raw = try src_dir.readFileAlloc(io, "config.json", allocator, .limited(16 << 20));
    defer allocator.free(cfg_raw);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, cfg_raw, .{});
    var root = parsed.object;
    var text = root.get("text_config").?.object;
    try text.put(a, "num_hidden_layers", .{ .integer = @intCast(spec.layers) });
    try text.put(a, "num_experts", .{ .integer = @intCast(spec.experts) });
    try text.put(a, "mtp_num_hidden_layers", .{ .integer = 0 });
    try text.put(a, "ngram_vocab_size_base", .{ .integer = @intCast(spec.ngram_vocab_base) });
    var kinds = std.json.Array.init(a);
    for (text.get("layer_types").?.array.items[0..spec.layers]) |k| try kinds.append(k);
    try text.put(a, "layer_types", .{ .array = kinds });
    _ = text.orderedRemove("mtp");
    try root.put(a, "text_config", .{ .object = text });
    try root.put(a, "language_model_only", .{ .bool = true });
    const cfg = try std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = root }, .{ .whitespace = .indent_2 });
    defer allocator.free(cfg);
    try out.writeFile(io, .{ .sub_path = "config.json", .data = cfg });
    _ = try pack.copySidecars(allocator, io, src_dir, out);
}

fn envText(name: [*:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(name) orelse return null;
    const text = std.mem.span(raw);
    return if (text.len == 0) null else text;
}

fn envCount(name: [*:0]const u8, default: usize) usize {
    const text = envText(name) orelse return default;
    return std.fmt.parseInt(usize, text, 10) catch default;
}

test "slice (real Qwen checkpoint, SUSHI_ENCODER_REAL_MODEL=1, SUSHI_MODELS_DIR and ENCODER_SLICE_OUT): writes a layers-by-experts slice for conversion runs" {
    const t = std.testing;
    const io = t.io;
    if (envText("SUSHI_ENCODER_REAL_MODEL") == null) return error.SkipZigTest;
    const out_path = envText("ENCODER_SLICE_OUT") orelse return error.SkipZigTest;
    const models = envText("SUSHI_MODELS_DIR") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(models) or !std.fs.path.isAbsolute(out_path)) return error.SkipZigTest;
    const layers = envCount("ENCODER_SLICE_LAYERS", 3);
    const experts = envCount("ENCODER_SLICE_EXPERTS", 64);

    const src_path = try std.fmt.allocPrint(t.allocator, "{s}/Qwen/Qwen3.8-Flash-Next", .{models});
    defer t.allocator.free(src_path);
    var real = std.Io.Dir.cwd().openDir(io, src_path, .{ .iterate = true }) catch return error.SkipZigTest;
    defer real.close(io);
    var real_src = try source.SafetensorsSource.open(t.allocator, io, real);
    defer real_src.deinit();
    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, out_path);
    var out = try cwd.openDir(io, out_path, .{ .iterate = true });
    defer out.close(io);
    try sliceQwen(t.allocator, io, real, &real_src, out, .{ .layers = layers, .experts = experts });

    var made = try source.SafetensorsSource.open(t.allocator, io, out);
    defer made.deinit();
    var key: [128]u8 = undefined;
    const gate_up = made.tensors.get(try std.fmt.bufPrint(&key, "model.language_model.layers.{d}.mlp.experts.gate_up_proj", .{layers - 1})).?;
    try t.expectEqual(experts, gate_up.shape[0]);
    try t.expect(!made.tensors.contains(try std.fmt.bufPrint(&key, "model.language_model.layers.{d}.mlp.experts.gate_up_proj", .{layers})));
}
