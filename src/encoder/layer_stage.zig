const std = @import("std");
const api = @import("sushi_exl3");
const format = api.format;
const pack = @import("pack.zig");
const shard = @import("shard.zig");

const trunk_prefix = "language_model.model.layers.";

pub const PackTensor = struct { file: usize, at: u64, len: u64, dtype: []u8, shape: []usize };

pub const PackMap = struct {
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    files: [][]u8,
    map: std.StringHashMap(PackTensor),

    pub fn scan(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !PackMap {
        var files = std.ArrayList([]u8).empty;
        errdefer {
            for (files.items) |f| allocator.free(f);
            files.deinit(allocator);
        }
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (entry.kind != .file or entry.name[0] == '.' or !std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
            try files.append(allocator, try allocator.dupe(u8, entry.name));
        }
        var self = PackMap{ .allocator = allocator, .dir = dir, .files = try files.toOwnedSlice(allocator), .map = std.StringHashMap(PackTensor).init(allocator) };
        errdefer self.deinit();
        for (self.files, 0..) |name, fi| {
            var f = try dir.openFile(io, name, .{});
            defer f.close(io);
            var len_buf: [8]u8 = undefined;
            if (try f.readPositionalAll(io, &len_buf, 0) != 8) return error.TruncatedSafetensors;
            const header_len = std.mem.readInt(u64, &len_buf, .little);
            const header = try allocator.alloc(u8, header_len);
            defer allocator.free(header);
            if (try f.readPositionalAll(io, header, 8) != header_len) return error.TruncatedSafetensors;
            const parsed = try std.json.parseFromSlice(std.json.Value, allocator, header, .{});
            defer parsed.deinit();
            var jt = parsed.value.object.iterator();
            while (jt.next()) |e| {
                if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
                const obj = e.value_ptr.object;
                const offs = obj.get("data_offsets").?.array.items;
                const shape_v = obj.get("shape").?.array.items;
                const shape = try allocator.alloc(usize, shape_v.len);
                errdefer allocator.free(shape);
                for (shape_v, shape) |d, *o| o.* = @intCast(d.integer);
                const dtype = try allocator.dupe(u8, obj.get("dtype").?.string);
                errdefer allocator.free(dtype);
                const start: u64 = @intCast(offs[0].integer);
                const end: u64 = @intCast(offs[1].integer);
                try self.map.put(try allocator.dupe(u8, e.key_ptr.*), .{ .file = fi, .at = 8 + header_len + start, .len = end - start, .dtype = dtype, .shape = shape });
            }
        }
        return self;
    }

    pub fn deinit(self: *PackMap) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            self.allocator.free(e.key_ptr.*);
            self.allocator.free(e.value_ptr.dtype);
            self.allocator.free(e.value_ptr.shape);
        }
        self.map.deinit();
        for (self.files) |f| self.allocator.free(f);
        self.allocator.free(self.files);
    }

    fn read(self: *const PackMap, io: std.Io, t: PackTensor, dst: []u8) !void {
        var f = try self.dir.openFile(io, self.files[t.file], .{});
        defer f.close(io);
        if (try f.readPositionalAll(io, dst, t.at) != dst.len) return error.TruncatedSafetensors;
    }

    fn sortedNames(self: *const PackMap, allocator: std.mem.Allocator) ![][]const u8 {
        var names = std.ArrayList([]const u8).empty;
        errdefer names.deinit(allocator);
        var it = self.map.keyIterator();
        while (it.next()) |k| try names.append(allocator, k.*);
        std.mem.sortUnstable([]const u8, names.items, {}, struct {
            fn lt(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);
        return names.toOwnedSlice(allocator);
    }
};

fn isLayerTensor(name: []const u8) bool {
    return std.mem.startsWith(u8, name, trunk_prefix);
}

fn layerOf(name: []const u8) ?u32 {
    if (!isLayerTensor(name)) return null;
    const rest = name[trunk_prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    return std.fmt.parseInt(u32, rest[0..dot], 10) catch null;
}

pub fn writeGlobals(allocator: std.mem.Allocator, io: std.Io, map: *const PackMap, root: std.Io.Dir) !void {
    const names = try map.sortedNames(allocator);
    defer allocator.free(names);
    var b = pack.ShardBuilder.init(allocator);
    defer b.deinit();
    for (names) |name| {
        if (isLayerTensor(name) or std.mem.startsWith(u8, name, "language_model.mtp.") or std.mem.startsWith(u8, name, "vision_tower.")) continue;
        const t = map.map.get(name).?;
        const raw = try allocator.alloc(u8, @intCast(t.len));
        defer allocator.free(raw);
        try map.read(io, t, raw);
        try b.add(name, t.dtype, t.shape, raw);
    }
    try b.flush(io, root, "globals.safetensors", &.{.{ "format", "mlx" }});
}

pub fn writePlaceholderExperts(allocator: std.mem.Allocator, io: std.Io, root: std.Io.Dir, geom: anytype, rate: format.Rate, dec: format.Decode) !void {
    const projs = [3][2]usize{ .{ geom.hidden, geom.inter }, .{ geom.hidden, geom.inter }, .{ geom.inter, geom.hidden } };
    var banks: [3]shard.Bank = undefined;
    var made: usize = 0;
    defer for (banks[0..made]) |bk| {
        allocator.free(bk.trellis);
        allocator.free(bk.suh);
        allocator.free(bk.svh);
    };
    while (made < 3) : (made += 1) {
        const in = projs[made][0];
        const out = projs[made][1];
        const trellis = try allocator.alloc(u16, geom.experts * (in / 16) * (out / 16) * rate.n);
        errdefer allocator.free(trellis);
        @memset(trellis, 0);
        const suh = try allocator.alloc(u16, geom.experts * in);
        errdefer allocator.free(suh);
        @memset(suh, format.f32ToF16Bits(1.0));
        const svh = try allocator.alloc(u16, geom.experts * out);
        @memset(svh, format.f32ToF16Bits(1.0));
        banks[made] = .{ .experts = geom.experts, .in_features = in, .out_features = out, .trellis = trellis, .suh = suh, .svh = svh };
    }
    var quant_buf: [160]u8 = undefined;
    root.deleteFile(io, "layer-0.safetensors") catch {};
    _ = try shard.writeLayer(allocator, io, root, shard.trunk_prefix, 0, .{ .gate = banks[0], .up = banks[1], .down = banks[2], .rate = rate, .dec = dec }, &quant_buf);
}

fn renamePrefix(allocator: std.mem.Allocator, name: []const u8, from: u32, to_ple_of: ?u32) ![]u8 {
    _ = to_ple_of;
    var from_buf: [96]u8 = undefined;
    const from_text = try std.fmt.bufPrint(&from_buf, "{s}{d}.", .{ trunk_prefix, from });
    return std.fmt.allocPrint(allocator, "{s}0.{s}", .{ trunk_prefix, name[from_text.len..] });
}

pub const Kind = enum { linear, full };

pub const Stage = struct {
    layer: u32,
    kind: Kind,
    ple_layer: u32,
    experts: Experts,
    source_config: []const u8,
    source_dir_abs: []const u8,
    root_abs: []const u8,
    pack_abs: []const u8,
    trunk_bits: u32,
    trunk_group: u32,
    overrides: []const pack.Override,
    has_ngram: bool,
    eq: pack.ExpertQuantSpec,

    pub const Experts = union(enum) { shard: []const u8, placeholder };
};

fn patchedSourceConfig(allocator: std.mem.Allocator, stage: Stage) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, stage.source_config, .{});
    var root = parsed.object;
    var text = root.get("text_config").?.object;
    try text.put(a, "num_hidden_layers", .{ .integer = 1 });
    var kinds = std.json.Array.init(a);
    try kinds.append(.{ .string = if (stage.kind == .full) "full_attention" else "linear_attention" });
    try text.put(a, "layer_types", .{ .array = kinds });
    try text.put(a, "full_attention_interval", .{ .integer = if (stage.kind == .full) 1 else 1000 });
    var ple = std.json.Array.init(a);
    try ple.append(.{ .integer = 1 });
    try text.put(a, "ple_layer_ids", .{ .array = ple });
    try text.put(a, "mtp_num_hidden_layers", .{ .integer = 0 });
    _ = text.orderedRemove("mtp");
    try root.put(a, "text_config", .{ .object = text });
    try root.put(a, "language_model_only", .{ .bool = true });
    return std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = root }, .{});
}

fn copyRenamedShard(allocator: std.mem.Allocator, io: std.Io, src_dir: std.Io.Dir, src_name: []const u8, dst_dir: std.Io.Dir, dst_name: []const u8, layer: u32, names_out: *std.ArrayList([]u8)) !void {
    var f = try src_dir.openFile(io, src_name, .{});
    defer f.close(io);
    var len_buf: [8]u8 = undefined;
    if (try f.readPositionalAll(io, &len_buf, 0) != 8) return error.TruncatedSafetensors;
    const header_len = std.mem.readInt(u64, &len_buf, .little);
    const header = try allocator.alloc(u8, header_len);
    defer allocator.free(header);
    if (try f.readPositionalAll(io, header, 8) != header_len) return error.TruncatedSafetensors;
    var from_buf: [96]u8 = undefined;
    const from_text = try std.fmt.bufPrint(&from_buf, ".layers.{d}.", .{layer});
    const replaced = try std.mem.replaceOwned(u8, allocator, header, from_text, ".layers.0.");
    defer allocator.free(replaced);
    var padded = std.ArrayList(u8).empty;
    defer padded.deinit(allocator);
    try padded.appendSlice(allocator, std.mem.trimEnd(u8, replaced, " "));
    while (padded.items.len % 8 != 0) try padded.append(allocator, ' ');
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, padded.items, .{});
    defer parsed.deinit();
    var it = parsed.value.object.iterator();
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
        try names_out.append(allocator, try allocator.dupe(u8, e.key_ptr.*));
    }
    const stat = try f.stat(io);
    var tmp_buf: [96]u8 = undefined;
    const tmp = try std.fmt.bufPrint(&tmp_buf, ".{s}.tmp", .{dst_name});
    var out = try dst_dir.createFile(io, tmp, .{});
    {
        defer out.close(io);
        std.mem.writeInt(u64, &len_buf, padded.items.len, .little);
        try out.writeStreamingAll(io, &len_buf);
        try out.writeStreamingAll(io, padded.items);
        const chunk = try allocator.alloc(u8, 16 << 20);
        defer allocator.free(chunk);
        var at: u64 = 8 + header_len;
        while (at < stat.size) {
            const want: usize = @intCast(@min(chunk.len, stat.size - at));
            if (try f.readPositionalAll(io, chunk[0..want], at) != want) return error.TruncatedSafetensors;
            try out.writeStreamingAll(io, chunk[0..want]);
            at += want;
        }
    }
    try dst_dir.rename(tmp, dst_dir, dst_name, io);
}

fn link(io: std.Io, dir: std.Io.Dir, target_abs: []const u8, name: []const u8) !void {
    dir.deleteFile(io, name) catch {};
    try dir.symLink(io, target_abs, name, .{});
}

pub fn stageLayer(allocator: std.mem.Allocator, io: std.Io, map: *const PackMap, root: std.Io.Dir, stage_name: []const u8, stage: Stage) !void {
    const cwd = std.Io.Dir.cwd();
    root.deleteTree(io, stage_name) catch {};
    try root.createDirPath(io, stage_name);
    var dir = try root.openDir(io, stage_name, .{ .iterate = true });
    defer dir.close(io);

    var index = std.ArrayList(pack.IndexEntry).empty;
    defer {
        for (index.items) |e| {
            allocator.free(e.tensor);
            allocator.free(e.file);
        }
        index.deinit(allocator);
    }
    const names = try map.sortedNames(allocator);
    defer allocator.free(names);

    var globals = std.ArrayList([]u8).empty;
    defer {
        for (globals.items) |n| allocator.free(n);
        globals.deinit(allocator);
    }
    for (names) |name| {
        if (isLayerTensor(name) or std.mem.startsWith(u8, name, "language_model.mtp.") or std.mem.startsWith(u8, name, "vision_tower.")) continue;
        try index.append(allocator, .{ .tensor = try allocator.dupe(u8, name), .file = try allocator.dupe(u8, "globals.safetensors") });
    }
    var globals_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const globals_path = try std.fmt.bufPrint(&globals_path_buf, "{s}/globals.safetensors", .{stage.root_abs});
    try link(io, dir, globals_path, "globals.safetensors");

    var b = pack.ShardBuilder.init(allocator);
    defer b.deinit();
    var renamed = std.ArrayList(pack.Override).empty;
    defer {
        for (renamed.items) |o| allocator.free(o.base);
        renamed.deinit(allocator);
    }
    var layer_names = std.ArrayList([]u8).empty;
    defer {
        for (layer_names.items) |n| allocator.free(n);
        layer_names.deinit(allocator);
    }
    for (names) |name| {
        const li = layerOf(name) orelse continue;
        if (std.mem.indexOf(u8, name, ".mlp.switch_mlp.") != null) continue;
        const is_own = li == stage.layer;
        const is_ple = li == stage.ple_layer and std.mem.indexOf(u8, name, ".ple.") != null and stage.layer != stage.ple_layer;
        if (!is_own and !is_ple) continue;
        const t = map.map.get(name).?;
        const raw = try allocator.alloc(u8, @intCast(t.len));
        defer allocator.free(raw);
        try map.read(io, t, raw);
        const new_name = try renamePrefix(allocator, name, li, null);
        defer allocator.free(new_name);
        try b.add(new_name, t.dtype, t.shape, raw);
        try layer_names.append(allocator, try allocator.dupe(u8, new_name));
    }
    try b.flush(io, dir, "trunk.safetensors", &.{.{ "format", "mlx" }});
    for (layer_names.items) |n| try index.append(allocator, .{ .tensor = try allocator.dupe(u8, n), .file = try allocator.dupe(u8, "trunk.safetensors") });
    for (stage.overrides) |o| {
        if (std.mem.startsWith(u8, o.base, trunk_prefix)) {
            const rest = o.base[trunk_prefix.len..];
            const dot = std.mem.indexOfScalar(u8, rest, '.') orelse continue;
            const li = std.fmt.parseInt(u32, rest[0..dot], 10) catch continue;
            const keep = li == stage.layer or (li == stage.ple_layer and std.mem.startsWith(u8, rest[dot + 1 ..], "ple.") and stage.layer != stage.ple_layer);
            if (!keep) continue;
            const base = try std.fmt.allocPrint(allocator, "{s}0.{s}", .{ trunk_prefix, rest[dot + 1 ..] });
            errdefer allocator.free(base);
            try renamed.append(allocator, .{ .base = base, .bits = o.bits, .group = o.group });
        } else {
            const base = try allocator.dupe(u8, o.base);
            errdefer allocator.free(base);
            try renamed.append(allocator, .{ .base = base, .bits = o.bits, .group = o.group });
        }
    }

    var expert_names = std.ArrayList([]u8).empty;
    defer {
        for (expert_names.items) |n| allocator.free(n);
        expert_names.deinit(allocator);
    }
    switch (stage.experts) {
        .shard => |file| try copyRenamedShard(allocator, io, map.dir, file, dir, "experts.safetensors", stage.layer, &expert_names),
        .placeholder => {
            var root_map_dir = try cwd.openDir(io, stage.root_abs, .{});
            defer root_map_dir.close(io);
            try copyRenamedShard(allocator, io, root_map_dir, "layer-0.safetensors", dir, "experts.safetensors", 0, &expert_names);
        },
    }
    for (expert_names.items) |n| try index.append(allocator, .{ .tensor = try allocator.dupe(u8, n), .file = try allocator.dupe(u8, "experts.safetensors") });

    const index_json = try pack.buildIndex(allocator, index.items, 0);
    defer allocator.free(index_json);
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index_json });

    const patched = try patchedSourceConfig(allocator, stage);
    defer allocator.free(patched);
    const config = try pack.buildConfig(allocator, patched, stage.trunk_bits, stage.trunk_group, renamed.items, if (stage.has_ngram) pack.NgramSpec{} else null, stage.eq);
    defer allocator.free(config);
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = config });

    if (stage.has_ngram) {
        var ng_buf: [std.fs.max_path_bytes]u8 = undefined;
        try link(io, dir, try std.fmt.bufPrint(&ng_buf, "{s}/ngram_table.bin", .{stage.pack_abs}), "ngram_table.bin");
    }
    var src_dir = try cwd.openDir(io, stage.source_dir_abs, .{ .iterate = true });
    defer src_dir.close(io);
    var sit = src_dir.iterate();
    while (try sit.next(io)) |entry| {
        if (entry.kind != .file or entry.name[0] == '.' or std.mem.endsWith(u8, entry.name, ".safetensors") or std.mem.endsWith(u8, entry.name, ".json") and (std.mem.eql(u8, entry.name, "config.json") or std.mem.eql(u8, entry.name, "model.safetensors.index.json"))) continue;
        if (!(std.mem.endsWith(u8, entry.name, ".json") or std.mem.endsWith(u8, entry.name, ".txt") or std.mem.endsWith(u8, entry.name, ".jinja"))) continue;
        var tb: [std.fs.max_path_bytes]u8 = undefined;
        try link(io, dir, try std.fmt.bufPrint(&tb, "{s}/{s}", .{ stage.source_dir_abs, entry.name }), entry.name);
    }
}

const driver = @import("driver.zig");
const st = @import("safetensors.zig");
const host = @import("mlx_host");

const StageFixture = struct {
    work: std.testing.TmpDir,
    model: []u8,
    out: []u8,
    cfg: []u8,

    fn init(allocator: std.mem.Allocator) !StageFixture {
        const io = std.testing.io;
        var work = std.testing.tmpDir(.{});
        errdefer work.cleanup();
        try work.dir.createDirPath(io, "src");
        var src_dir = try work.dir.openDir(io, "src", .{});
        defer src_dir.close(io);
        try driver.writeTinySource(allocator, io, src_dir);
        const model = try driver.tmpPath(allocator, &work, "src");
        errdefer allocator.free(model);
        const out = try driver.tmpPath(allocator, &work, "pack");
        errdefer allocator.free(out);
        _ = try driver.run(allocator, io, host.mlx.gpuStream(), driver.tinyOptions(model, out, false), null, null);
        const cfg = try src_dir.readFileAlloc(io, "config.json", allocator, .limited(1 << 20));
        return .{ .work = work, .model = model, .out = out, .cfg = cfg };
    }

    fn deinit(self: *StageFixture, allocator: std.mem.Allocator) void {
        allocator.free(self.model);
        allocator.free(self.out);
        allocator.free(self.cfg);
        self.work.cleanup();
    }
};

fn fixtureStage(fx: *const StageFixture, layer: u32, kind: Kind, experts: Stage.Experts, root_abs: []const u8) Stage {
    return .{
        .layer = layer,
        .kind = kind,
        .ple_layer = 1,
        .experts = experts,
        .source_config = fx.cfg,
        .source_dir_abs = fx.model,
        .root_abs = root_abs,
        .pack_abs = fx.out,
        .trunk_bits = 8,
        .trunk_group = 64,
        .overrides = &.{},
        .has_ngram = true,
        .eq = .{ .k_text = "2", .codebook = "mcg", .window = 12 },
    };
}

test "layer_stage: one layer of a pack becomes a one-layer model whose names, index and config say so" {
    const t = std.testing;
    const io = t.io;
    var fx = try StageFixture.init(t.allocator);
    defer fx.deinit(t.allocator);
    var pack_dir = try std.Io.Dir.cwd().openDir(io, fx.out, .{ .iterate = true });
    defer pack_dir.close(io);
    var map = try PackMap.scan(t.allocator, io, pack_dir);
    defer map.deinit();

    try pack_dir.createDirPath(io, ".stage-root");
    var root = try pack_dir.openDir(io, ".stage-root", .{ .iterate = true });
    defer root.close(io);
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_abs = root_buf[0..try root.realPath(io, &root_buf)];
    try writeGlobals(t.allocator, io, &map, root);
    try writePlaceholderExperts(t.allocator, io, root, .{ .experts = @as(usize, 2), .hidden = @as(usize, 128), .inter = @as(usize, 128) }, .{ .n = 32 }, .{ .codebook = .mcg, .window = .w12 });

    try stageLayer(t.allocator, io, &map, root, "real", fixtureStage(&fx, 1, .full, .{ .shard = "layer-1.safetensors" }, root_abs));
    var stage = try root.openDir(io, "real", .{ .iterate = true });
    defer stage.close(io);
    const cfg_raw = try stage.readFileAlloc(io, "config.json", t.allocator, .limited(1 << 20));
    defer t.allocator.free(cfg_raw);
    const cfg = try std.json.parseFromSlice(std.json.Value, t.allocator, cfg_raw, .{});
    defer cfg.deinit();
    const text = cfg.value.object.get("text_config").?.object;
    try t.expectEqual(@as(i64, 1), text.get("num_hidden_layers").?.integer);
    try t.expectEqual(@as(i64, 1), text.get("full_attention_interval").?.integer);
    try t.expectEqualStrings("full_attention", text.get("layer_types").?.array.items[0].string);
    try t.expectEqual(@as(i64, 1), text.get("ple_layer_ids").?.array.items[0].integer);
    try t.expectEqual(@as(i64, 0), text.get("mtp_num_hidden_layers").?.integer);
    try t.expect(cfg.value.object.get("expert_quant") != null);
    try t.expect(cfg.value.object.get("ngram_table") != null);

    const idx_raw = try stage.readFileAlloc(io, "model.safetensors.index.json", t.allocator, .limited(1 << 20));
    defer t.allocator.free(idx_raw);
    const idx = try std.json.parseFromSlice(std.json.Value, t.allocator, idx_raw, .{});
    defer idx.deinit();
    const wm = idx.value.object.get("weight_map").?.object;
    try t.expectEqualStrings("experts.safetensors", wm.get("language_model.model.layers.0.mlp.switch_mlp.up_proj.trellis").?.string);
    try t.expectEqualStrings("globals.safetensors", wm.get("language_model.lm_head.weight").?.string);
    try t.expect(wm.get("language_model.model.layers.1.mlp.switch_mlp.up_proj.trellis") == null);

    const real_raw = try pack_dir.readFileAlloc(io, "layer-1.safetensors", t.allocator, .limited(1 << 24));
    defer t.allocator.free(real_raw);
    const real_al = try t.allocator.alignedAlloc(u8, .of(u16), real_raw.len);
    defer t.allocator.free(real_al);
    @memcpy(real_al, real_raw);
    var original = try st.parse(t.allocator, real_al);
    defer original.deinit(t.allocator);
    const moved_raw = try stage.readFileAlloc(io, "experts.safetensors", t.allocator, .limited(1 << 24));
    defer t.allocator.free(moved_raw);
    const moved_al = try t.allocator.alignedAlloc(u8, .of(u16), moved_raw.len);
    defer t.allocator.free(moved_al);
    @memcpy(moved_al, moved_raw);
    var moved = try st.parse(t.allocator, moved_al);
    defer moved.deinit(t.allocator);
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
        for ([_][]const u8{ "trellis", "suh", "svh" }) |part| {
            var a: [160]u8 = undefined;
            var b: [160]u8 = undefined;
            const was = original.tensors.get(try std.fmt.bufPrint(&a, "language_model.model.layers.1.mlp.switch_mlp.{s}.{s}", .{ proj, part })).?;
            const now = moved.tensors.get(try std.fmt.bufPrint(&b, "language_model.model.layers.0.mlp.switch_mlp.{s}.{s}", .{ proj, part })).?;
            try t.expectEqualSlices(u8, was.bytes, now.bytes);
        }
    }
    try t.expectEqualStrings("exl3", moved.metadata.get("format").?);

    var link_buf: [std.fs.max_path_bytes]u8 = undefined;
    for ([_][]const u8{ "globals.safetensors", "ngram_table.bin", "tokenizer.json" }) |name| {
        const n = try stage.readLink(io, name, &link_buf);
        try t.expect(std.fs.path.isAbsolute(link_buf[0..n]));
    }

    try stageLayer(t.allocator, io, &map, root, "capture", fixtureStage(&fx, 0, .linear, .placeholder, root_abs));
    var cap = try root.openDir(io, "capture", .{ .iterate = true });
    defer cap.close(io);
    const cap_cfg = try cap.readFileAlloc(io, "config.json", t.allocator, .limited(1 << 20));
    defer t.allocator.free(cap_cfg);
    const cc = try std.json.parseFromSlice(std.json.Value, t.allocator, cap_cfg, .{});
    defer cc.deinit();
    try t.expectEqual(@as(i64, 1000), cc.value.object.get("text_config").?.object.get("full_attention_interval").?.integer);
    try cap.access(io, "trunk.safetensors", .{});
}
