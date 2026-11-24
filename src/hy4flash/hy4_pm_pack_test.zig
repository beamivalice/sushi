//! A pair-major copy of a Hy4-Flash STQ pack, built from its pack-order one by permuting the routed trellis tensors in
//! memory (`layout.tileMajor`, as `sushi repack` lays them), each starting on a page of its shard when asked: the fixture
//! of the Hy4 cells that want a pack laid another way than the tiny's. A pack already pair-major keeps its bytes.
const std = @import("std");
const exl3 = @import("sushi_exl3");

const tile_layout = exl3.group_layout;

fn isRouted(key: []const u8) bool {
    return std.mem.indexOf(u8, key, ".mlp.switch_mlp.") != null and (std.mem.endsWith(u8, key, ".trellis") or std.mem.endsWith(u8, key, ".trellis_pm"));
}

fn readAll(io: std.Io, allocator: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    return d.readFileAlloc(io, name, allocator, .limited(512 << 20));
}

fn writeAll(io: std.Io, dir: []const u8, name: []const u8, data: []const u8) !void {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    try d.writeFile(io, .{ .sub_path = name, .data = data });
}

/// `config.json` with `"tile_order": "pair_major"` as the first member of `expert_quant`.
fn rewriteConfig(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    if (std.mem.indexOf(u8, raw, "\"tile_order\"") != null) return allocator.dupe(u8, raw);
    const at = std.mem.indexOf(u8, raw, "\"expert_quant\"") orelse return error.FixtureConfig;
    const open = std.mem.indexOfScalarPos(u8, raw, at, '{') orelse return error.FixtureConfig;
    return std.fmt.allocPrint(allocator, "{s}\"tile_order\": \"pair_major\",{s}", .{ raw[0 .. open + 1], raw[open + 1 ..] });
}

/// The index with every routed `.trellis` key renamed `.trellis_pm`, and each pad `pads` names listed under its shard.
fn rewriteIndex(allocator: std.mem.Allocator, raw: []const u8, pads: []const Pad) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, raw, from, ".trellis\"")) |hit| {
        const key_start = std.mem.lastIndexOfScalar(u8, raw[0..hit], '"').? + 1;
        const end = hit + ".trellis".len;
        try out.appendSlice(allocator, raw[from..end]);
        if (isRouted(raw[key_start..end])) try out.appendSlice(allocator, "_pm");
        from = end;
    }
    try out.appendSlice(allocator, raw[from..]);
    if (pads.len > 0) {
        const close = std.mem.lastIndexOf(u8, out.items, "}}") orelse return error.FixtureIndex;
        out.items.len = close;
        for (pads) |pad| try out.print(allocator, ",\"{s}\":\"{s}\"", .{ pad.key, pad.shard });
        try out.appendSlice(allocator, "}}");
    }
    return out.toOwnedSlice(allocator);
}

/// A zero filler tensor a page-aligned shard carries, and the shard that holds it.
const Pad = struct { key: []u8, shard: []u8 };

/// Whether the shard `raw` holds a routed trellis tensor.
fn holdsRouted(allocator: std.mem.Allocator, raw: []const u8) !bool {
    const header_len: usize = @intCast(std.mem.readInt(u64, raw[0..8], .little));
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw[8..][0..header_len], .{});
    defer parsed.deinit();
    for (parsed.value.object.keys()) |key| if (isRouted(key)) return true;
    return false;
}

/// The shard with its routed trellis tensors pair-major under `<key>_pm`, each starting on a `page` boundary of the file
/// when asked (a zero `__pad__` tensor fills the gap), and `tile_order` stamped in its metadata.
fn rewriteShard(allocator: std.mem.Allocator, raw: []const u8, page: ?usize, shard: []const u8, pad_list: *std.ArrayList(Pad)) ![]u8 {
    const header_len: usize = @intCast(std.mem.readInt(u64, raw[0..8], .little));
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw[8..][0..header_len], .{});
    defer parsed.deinit();
    const data_in = raw[8 + header_len ..];
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(allocator);
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    try header.append(allocator, '{');
    var first = true;
    var pads: usize = 0;
    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (!first) try header.append(allocator, ',');
        first = false;
        if (std.mem.eql(u8, key, "__metadata__")) {
            try header.appendSlice(allocator, "\"__metadata__\":{");
            var mit = entry.value_ptr.object.iterator();
            while (mit.next()) |m| if (!std.mem.eql(u8, m.key_ptr.*, "tile_order")) try header.print(allocator, "\"{s}\":\"{s}\",", .{ m.key_ptr.*, m.value_ptr.string });
            try header.appendSlice(allocator, "\"tile_order\":\"pair_major\"}");
            continue;
        }
        const obj = entry.value_ptr.object;
        const dtype = obj.get("dtype").?.string;
        const shape_in = obj.get("shape").?.array.items;
        const offs = obj.get("data_offsets").?.array.items;
        const start: usize = @intCast(offs[0].integer);
        const end: usize = @intCast(offs[1].integer);
        const major = isRouted(key);
        if (major and page != null) {
            const gap = std.mem.alignForward(usize, data.items.len, page.?) - data.items.len;
            if (gap > 0) {
                const stem = shard[0 .. shard.len - ".safetensors".len];
                try pad_list.append(allocator, .{ .key = try std.fmt.allocPrint(allocator, "__pad__.{s}.{d}", .{ stem, pads }), .shard = try allocator.dupe(u8, shard) });
                try header.print(allocator, "\"__pad__.{s}.{d}\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}},", .{ stem, pads, gap, data.items.len, data.items.len + gap });
                try data.appendNTimes(allocator, 0, gap);
                pads += 1;
            }
        }
        const at = data.items.len;
        try data.appendSlice(allocator, data_in[start..end]);
        if (!major) {
            try header.print(allocator, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ key, dtype });
            for (shape_in, 0..) |d, i| try header.print(allocator, "{s}{d}", .{ if (i == 0) "" else ",", d.integer });
            try header.print(allocator, "],\"data_offsets\":[{d},{d}]}}", .{ at, at + end - start });
            continue;
        }
        if (std.mem.endsWith(u8, key, "_pm")) {
            try header.print(allocator, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ key, dtype });
            for (shape_in, 0..) |d, i| try header.print(allocator, "{s}{d}", .{ if (i == 0) "" else ",", d.integer });
            try header.print(allocator, "],\"data_offsets\":[{d},{d}]}}", .{ at, at + end - start });
            continue;
        }
        if (shape_in.len != 4 or !std.mem.eql(u8, dtype, "U16")) return error.FixtureShape;
        const experts: usize = @intCast(shape_in[0].integer);
        const it_tiles: usize = @intCast(shape_in[1].integer);
        const ot_tiles: usize = @intCast(shape_in[2].integer);
        const tile_bytes: usize = @intCast(shape_in[3].integer * 2);
        const per_expert = it_tiles * ot_tiles * tile_bytes;
        for (0..experts) |e| tile_layout.tileMajor(u8, data.items[at + e * per_expert ..][0..per_expert], data_in[start + e * per_expert ..][0..per_expert], it_tiles, ot_tiles, tile_bytes);
        try header.print(allocator, "\"{s}_pm\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d},{d},2,{d}],\"data_offsets\":[{d},{d}]}}", .{ key, dtype, experts, ot_tiles / 2, it_tiles, shape_in[3].integer, at, at + end - start });
    }
    try header.append(allocator, '}');
    while ((8 + header.items.len) % (page orelse 8) != 0) try header.append(allocator, ' ');
    const out = try allocator.alloc(u8, 8 + header.items.len + data.items.len);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8..][0..header.items.len], header.items);
    @memcpy(out[8 + header.items.len ..], data.items);
    return out;
}

/// Writes the pack at `src` into the empty directory `dst` with its routed tensors pair-major (starting on `page`
/// boundaries when given); every file the routed tensors do not touch is linked to `src`.
pub fn write(allocator: std.mem.Allocator, src: []const u8, dst: []const u8, page: ?usize) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = try std.Io.Dir.openDirAbsolute(io, src, .{ .iterate = true });
    defer dir.close(io);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }
    var walk = dir.iterate();
    while (try walk.next(io)) |entry| try names.append(allocator, try allocator.dupe(u8, entry.name));
    var pads: std.ArrayList(Pad) = .empty;
    defer {
        for (pads.items) |p| {
            allocator.free(p.key);
            allocator.free(p.shard);
        }
        pads.deinit(allocator);
    }
    // The index lists the pads the shards made, so it is written last.
    const index_name = "model.safetensors.index.json";
    for (names.items) |name| {
        if (std.mem.eql(u8, name, index_name)) continue;
        const rewritten: ?[]u8 = blk: {
            if (std.mem.eql(u8, name, "config.json")) {
                const raw = try readAll(io, allocator, src, name);
                defer allocator.free(raw);
                break :blk try rewriteConfig(allocator, raw);
            }
            if (std.mem.startsWith(u8, name, "model") and std.mem.endsWith(u8, name, ".safetensors")) {
                const raw = try readAll(io, allocator, src, name);
                defer allocator.free(raw);
                if (!try holdsRouted(allocator, raw)) break :blk null;
                break :blk try rewriteShard(allocator, raw, page, name, &pads);
            }
            break :blk null;
        };
        if (rewritten) |out| {
            defer allocator.free(out);
            try writeAll(io, dst, name, out);
            continue;
        }
        var from_buf: [std.fs.max_path_bytes]u8 = undefined;
        var to_buf: [std.fs.max_path_bytes]u8 = undefined;
        const from = try std.fmt.bufPrint(&from_buf, "{s}/{s}", .{ src, name });
        const to = try std.fmt.bufPrint(&to_buf, "{s}/{s}", .{ dst, name });
        try std.Io.Dir.symLinkAbsolute(io, from, to, .{});
    }
    const raw = try readAll(io, allocator, src, index_name);
    defer allocator.free(raw);
    const index = try rewriteIndex(allocator, raw, pads.items);
    defer allocator.free(index);
    try writeAll(io, dst, index_name, index);
}

// ---- cells ----

const testing = std.testing;
const mlx = @import("../mlx.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const model_mod = @import("../model.zig");
const hy4_config = @import("hy4_config.zig");
const hy4_stq = @import("hy4_stq.zig");
const owned_weights = @import("../owned_weights.zig");
const Arr = mlx.mlx_array;

fn tinyDir(env: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(env)) |p| std.mem.span(p) else null;
}

fn tmpPath(tmp: *testing.TmpDir, buf: *[std.fs.max_path_bytes]u8) ![]const u8 {
    const io = std.Io.Threaded.global_single_threaded.io();
    return buf[0..try tmp.dir.realPath(io, buf)];
}

/// A routed-layer weight map of the tiny at `dir`: each `model-layer-NN` shard's tensors.
fn loadLayers(dir: []const u8, s: mlx.mlx_stream) !model_mod.Weights {
    var w = model_mod.Weights.init(testing.allocator);
    errdefer w.deinit();
    for (1..6) |l| {
        const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model-layer-{d:0>2}.safetensors", .{ dir, l }, 0);
        defer testing.allocator.free(path);
        try model_mod.loadSafetensorsFile(testing.allocator, &w, path.ptr, s, true);
    }
    return w;
}

test "hy4 pair-major: a pack of the pack order is refused by name at parse, naming `sushi repack`, and a pair-major one parses (HY4_STQ_TINY)" {
    const dir = tinyDir("HY4_STQ_TINY") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const raw = try readAll(io, a, dir, "config.json");
    defer a.free(raw);
    const stamp = "\"tile_order\": \"pair_major\",";
    // The tiny's config in either order, whichever it is stored in.
    const plain = if (std.mem.indexOf(u8, raw, stamp)) |at| try std.mem.concat(a, u8, &.{ raw[0..at], raw[at + stamp.len ..] }) else try a.dupe(u8, raw);
    defer a.free(plain);
    const major = try rewriteConfig(a, plain);
    defer a.free(major);
    const cases = [_]struct { config: []const u8, leaf: []const u8, want: ?anyerror }{
        .{ .config = plain, .leaf = "trellis", .want = error.Exl3TileOrderUnsupported },
        .{ .config = major, .leaf = "trellis", .want = error.Exl3TileOrderMismatch },
        .{ .config = plain, .leaf = "trellis_pm", .want = error.Exl3TileOrderMismatch },
        .{ .config = major, .leaf = "trellis_pm", .want = null },
    };
    for (cases) |c| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        var index: std.ArrayList(u8) = .empty;
        defer index.deinit(a);
        try index.appendSlice(a, "{\"weight_map\":{");
        var first = true;
        for (1..6) |layer| for ([_][]const u8{ "gate", "up", "down" }) |proj| for ([_][]const u8{ c.leaf, "suh", "svh" }) |leaf| {
            try index.print(a, "{s}\"model.layers.{d}.mlp.switch_mlp.{s}_proj.g0.{s}\":\"model.safetensors\"", .{ if (first) "" else ",", layer, proj, leaf });
            first = false;
        };
        try index.appendSlice(a, "}}");
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = c.config });
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try tmpPath(&tmp, &buf);
        if (c.want) |err| {
            try testing.expectError(err, model_mod.parseConfig(io, a, path));
        } else {
            var config = try model_mod.parseConfig(io, a, path);
            defer config.deinit(a);
            try testing.expectEqual(exl3.format.TileOrder.pair_major, config.expert_tile_order);
        }
    }
}

/// `hy4_stq.routed`'s bytes over `rb` for `t` rows of `lat`.
fn routedBytes(o: *Ops, es: mlx.mlx_stream, dec: exl3.format.Decode, limit: f32, rb: *const exl3.Ragged, lat: Arr, pk: Arr, wa: Arr) ![]const u8 {
    const y = try o.own(try hy4_stq.routed(es, dec, limit, rb, lat, pk, wa));
    const c = try o.contiguous(y);
    try mlx.check(mlx.mlx_array_eval(c));
    return (mlx.mlx_array_data_uint8(c) orelse return error.HostRead)[0 .. mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c)];
}

fn bankIdentity(env: [:0]const u8) !void {
    const dir = tinyDir(env) orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const text = try readAll(io, a, dir, "config.json");
    defer a.free(text);
    const c = try hy4_config.parseText(text);
    const dec = (try hy4_stq.parseQuantText(a, text)).expert.?;
    var w = try loadLayers(dir, es);
    defer w.deinit();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    var rng = std.Random.DefaultPrng.init(0x9a1);
    const k: usize = c.top_k;
    for (c.dense_layers..c.layers) |l| {
        // The pack-order copy of the layer's banks: each expert's tiles back through `tileMajorInverse`.
        var banks: [exl3.group_layout.max_groups]exl3.Bank = undefined;
        var held: std.ArrayList(Arr) = .empty;
        defer {
            for (held.items) |h| _ = mlx.mlx_array_free(h);
            held.deinit(a);
        }
        var groups: usize = 0;
        while (groups < banks.len) : (groups += 1) {
            var nb: [128]u8 = undefined;
            if (w.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.switch_mlp.gate_proj.g{d}.trellis_pm", .{ l, groups })) == null) break;
            inline for (.{ "gate", "up", "down" }) |proj| {
                const tm = w.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.switch_mlp." ++ proj ++ "_proj.g{d}.trellis_pm", .{ l, groups })).?;
                const sh = mlx.getShape(tm);
                const experts: usize = @intCast(sh[0]);
                const ot: usize = @intCast(sh[1] * 2);
                const it: usize = @intCast(sh[2]);
                const n: usize = @intCast(sh[4]);
                try mlx.check(mlx.mlx_array_eval(tm));
                const src = (mlx.mlx_array_data_uint16(tm) orelse return error.HostRead)[0..mlx.mlx_array_size(tm)];
                const dst = try a.alloc(u16, src.len);
                defer a.free(dst);
                const per = it * ot * n;
                for (0..experts) |e| tile_layout.tileMajorInverse(u16, dst[e * per ..][0..per], src[e * per ..][0..per], it, ot, n);
                const legacy = try mlx.newData(dst.ptr, &.{ sh[0], sh[2], @intCast(ot), sh[4] }, .uint16);
                try held.append(a, legacy);
                @field(banks[groups], proj) = .{
                    .trellis = legacy,
                    .suh = w.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.switch_mlp." ++ proj ++ "_proj.g{d}.suh", .{ l, groups })).?,
                    .svh = w.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.switch_mlp." ++ proj ++ "_proj.g{d}.svh", .{ l, groups })).?,
                };
            }
        }
        var pack_rb = try owned_weights.raggedOwned(es, banks[0..groups], .pack, null);
        defer pack_rb.deinit();
        var major_rb = (try hy4_stq.loadExperts(&w, l, es, a)).?;
        defer major_rb.deinit();
        try testing.expect(major_rb.tile_major and !pack_rb.tile_major);
        // 1..8 rows run the decode chain (a verify block among them), 40 the prefill GEMMs.
        for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 40 }) |t| {
            const picks = try a.alloc(u32, t * k);
            defer a.free(picks);
            const wts = try a.alloc(f32, t * k);
            defer a.free(wts);
            for (0..t) |row| {
                var chosen: usize = 0;
                while (chosen < k) {
                    const e = rng.random().uintLessThan(u32, c.experts);
                    if (std.mem.indexOfScalar(u32, picks[row * k ..][0..chosen], e) != null) continue;
                    picks[row * k + chosen] = e;
                    wts[row * k + chosen] = 0.25 + rng.random().float(f32);
                    chosen += 1;
                }
            }
            const lat_v = try a.alloc(f32, t * c.moe_inter);
            defer a.free(lat_v);
            for (lat_v) |*d| d.* = (rng.random().float(f32) * 2 - 1) * 4;
            const lat = try o.own(try mlx.newData(lat_v.ptr, &.{ @intCast(t), @intCast(c.moe_inter) }, .float32));
            const pk = try o.own(try mlx.newData(picks.ptr, &.{ @intCast(t), @intCast(k) }, .uint32));
            const wa = try o.own(try mlx.newData(wts.ptr, &.{ @intCast(t), @intCast(k) }, .float32));
            const want = try a.dupe(u8, try routedBytes(&o, es, dec, c.swiglu_limit, &pack_rb, lat, pk, wa));
            defer a.free(want);
            const got = try routedBytes(&o, es, dec, c.swiglu_limit, &major_rb, lat, pk, wa);
            testing.expectEqualSlices(u8, want, got) catch |err| {
                std.debug.print("layer {d}, {d} rows: the pair-major bank's routed sum differs from the pack-order bank's\n", .{ l, t });
                return err;
            };
        }
    }
}

test "hy4 pair-major: the clamped chain over a pair-major bank gives the pack-order bank's bytes at 1 to 8 rows (decode and verify) and a prefill, each routed layer (HY4_STQ_TINY)" {
    try bankIdentity("HY4_STQ_TINY");
}

test "hy4 pair-major: the clamped chain over a pair-major bank gives the pack-order bank's bytes at 1 to 8 rows and a prefill, two rate groups a layer (HY4_STQ_TINY_MIXED)" {
    try bankIdentity("HY4_STQ_TINY_MIXED");
}

fn hostU16(a: Arr) ![]const u16 {
    try mlx.check(mlx.mlx_array_eval(a));
    return (mlx.mlx_array_data_uint16(a) orelse return error.HostRead)[0..mlx.mlx_array_size(a)];
}

/// One routed expert's stored bytes (layer 1, group 0: the tiny's widest).
fn expertBytes(w: *const model_mod.Weights) !u64 {
    var total: u64 = 0;
    for ([_][]const u8{ "gate", "up", "down" }) |proj| for ([_][]const u8{ "trellis_pm", "suh", "svh" }) |leaf| {
        var nb: [96]u8 = undefined;
        const t = w.get(try std.fmt.bufPrint(&nb, "model.layers.1.mlp.switch_mlp.{s}_proj.g0.{s}", .{ proj, leaf })).?;
        total += mlx.mlx_array_size(t) * mlx.mlx_array_itemsize(t) / @as(u64, @intCast(mlx.getShape(t)[0]));
    };
    return total;
}

/// The streaming engine's pool bytes of `slot` in projection `pi` of `rb` equal expert `e`'s tiles in the pack's tensors.
fn expectSlotIsFile(w: *const model_mod.Weights, l: usize, e: usize, rb: *const exl3.Ragged, slot: usize) !void {
    try mlx.check(mlx.mlx_array_eval(rb.meta));
    const meta = (mlx.mlx_array_data_uint32(rb.meta) orelse return error.HostRead);
    const pools = [3][]const u16{ try hostU16(rb.trellis[0]), try hostU16(rb.trellis[1]), try hostU16(rb.trellis[2]) };
    var off: usize = 0;
    var group: usize = 0;
    while (true) : (group += 1) {
        var nb: [128]u8 = undefined;
        const g = w.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.switch_mlp.gate_proj.g{d}.trellis_pm", .{ l, group })).?;
        const count: usize = @intCast(mlx.getShape(g)[0]);
        if (e < off + count) break;
        off += count;
    }
    inline for (.{ "gate", "up", "down" }, 0..) |proj, pi| {
        var nb: [128]u8 = undefined;
        const file = w.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.switch_mlp." ++ proj ++ "_proj.g{d}.trellis_pm", .{ l, group })).?;
        const sh = mlx.getShape(file);
        const per: usize = @intCast(sh[1] * sh[2] * sh[3] * sh[4]);
        try testing.expectEqual(@as(u32, @intCast(sh[4])), meta[slot * 8 + pi]);
        const want = (try hostU16(file))[(e - off) * per ..][0..per];
        const at = @as(usize, meta[slot * 8 + 4 + pi]) * 2;
        try testing.expectEqualSlices(u16, want, pools[pi][at..][0..per]);
    }
}

/// A streamed load of the pack at `dir` at `slots` experts a layer, after a forward: every expert its pools hold is the
/// pack's tiles unchanged, and no fill moved a tile on the CPU.
fn poolsHoldFile(dir: []const u8, slots: u64) !void {
    const Trunk = @import("../trunk.zig").Trunk;
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    var file = try loadLayers(dir, mlx.engine.stream());
    defer file.deinit();
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    try testing.expectEqual(exl3.format.TileOrder.pair_major, config.expert_tile_order);
    config.expert_streaming = true;
    config.expert_source_dir = try a.dupe(u8, dir);
    config.expert_cache_bytes = slots * config.num_hidden_layers * try expertBytes(&file);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const engine = xfm.expert_stream.?;
    var o: Ops = .{ .s = xfm.s };
    defer o.deinit();
    var ids: [12]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast(1 + (i * 7919) % 1000);
    const arr = try o.own(mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, ids.len }, 2, .uint32));
    try mlx.check(mlx.mlx_array_eval(try o.own(try xfm.forward(arr))));
    const experts: usize = 8;
    for (1..config.num_hidden_layers) |l| {
        if (slots >= experts) {
            const rr = engine.residentRoute(@intCast(l)).?;
            try mlx.check(mlx.mlx_array_eval(rr.map));
            const map = (mlx.mlx_array_data_int32(rr.map) orelse return error.HostRead)[0..experts];
            for (0..experts) |e| try expectSlotIsFile(&file, l, e, rr.bank, @intCast(map[e]));
        } else {
            const picks = [_]u16{ 5, 2, 7, 0 };
            engine.beginForward();
            var prepared = try engine.prepareHost(@intCast(l), &picks);
            defer prepared.deinit();
            for (picks, prepared.remapped) |e, slot| try expectSlotIsFile(&file, l, e, prepared.ragged.?, slot);
            engine.finishForward(1);
        }
    }
    const stats = engine.fillStats();
    try testing.expect(stats.bytes_read > 0);
    try testing.expectEqual(@as(u64, 0), stats.bytes_reshuffled);

}

test "hy4 pair-major: a streamed load's pools hold the pack's tiles unchanged, every expert resident or through a cache of 4 slots a layer, and its fills reshuffle none (HY4_STQ_TINY)" {
    const dir = tinyDir("HY4_STQ_TINY") orelse return error.SkipZigTest;
    try poolsHoldFile(dir, 8);
    try poolsHoldFile(dir, 4);
}

test "hy4 pair-major: a page-aligned pack, its zero pads listed in the index and skipped by the load, bills what the unpadded pack does and fills the pools with its tiles unchanged (HY4_STQ_TINY)" {
    const dir = tinyDir("HY4_STQ_TINY") orelse return error.SkipZigTest;
    if (@import("../expert_io.zig").pageSize() != 16384) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try tmpPath(&tmp, &buf);
    try write(testing.allocator, dir, path, 16384);
    var plain = try model_mod.parseConfig(io, testing.allocator, dir);
    defer plain.deinit(testing.allocator);
    var padded = try model_mod.parseConfig(io, testing.allocator, path);
    defer padded.deinit(testing.allocator);
    const load = @import("hy4_load.zig");
    for ([_]bool{ true, false }) |streamed| {
        plain.expert_streaming = streamed;
        padded.expert_streaming = streamed;
        try testing.expectEqual(try load.residentLoadBytes(io, testing.allocator, dir, &plain, false, false), try load.residentLoadBytes(io, testing.allocator, path, &padded, false, false));
    }
    try poolsHoldFile(path, 8);
    try poolsHoldFile(path, 4);
}

test "hy4 load: resident load refuses a routed pair-major shard whose tile_order stamp is missing (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const src = tinyDir("HY4_STQ_TINY") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dst = try tmpPath(&tmp, &path_buf);
    var sdir = try std.Io.Dir.openDirAbsolute(io, src, .{ .iterate = true });
    defer sdir.close(io);
    var it = sdir.iterate();
    while (try it.next(io)) |entry| {
        if (std.mem.eql(u8, entry.name, "model-layer-01.safetensors")) {
            const raw = try sdir.readFileAlloc(io, entry.name, a, .limited(64 << 20));
            defer a.free(raw);
            const needle = "\"tile_order\":\"pair_major\"";
            const at = std.mem.indexOf(u8, raw, needle) orelse return error.TestUnexpectedResult;
            const copy = try a.dupe(u8, raw);
            defer a.free(copy);
            @memcpy(copy[at..][0..needle.len], "\"xile_order\":\"pair_major\"");
            try tmp.dir.writeFile(io, .{ .sub_path = entry.name, .data = copy });
        } else {
            var from_buf: [std.fs.max_path_bytes]u8 = undefined;
            var to_buf: [std.fs.max_path_bytes]u8 = undefined;
            const from = try std.fmt.bufPrint(&from_buf, "{s}/{s}", .{ src, entry.name });
            const to = try std.fmt.bufPrint(&to_buf, "{s}/{s}", .{ dst, entry.name });
            try std.Io.Dir.symLinkAbsolute(io, from, to, .{});
        }
    }
    var cfg = try model_mod.parseConfig(io, a, dst);
    defer cfg.deinit(a);
    cfg.expert_source_dir = try a.dupe(u8, dst);
    const res = model_mod.loadWeightsForConfig(io, a, dst, &cfg, false);
    if (res) |*w| {
        var ww = w.*;
        ww.deinit();
        return error.TestUnexpectedResult;
    } else |err| {
        try testing.expectEqual(error.Exl3ShardStampMismatch, err);
    }
}
