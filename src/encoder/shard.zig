const std = @import("std");
const api = @import("sushi_exl3");
const format = api.format;
const encode = @import("encode.zig");
const st = @import("safetensors.zig");
const quantize = @import("quantize.zig");

pub const trunk_prefix = "language_model.model";
pub const mtp_prefix = "language_model.mtp";

pub const Bank = struct {
    experts: usize,
    in_features: usize,
    out_features: usize,
    trellis: []const u16,
    suh: []const u16,
    svh: []const u16,
};

pub const Layer = struct {
    gate: Bank,
    up: Bank,
    down: Bank,
    rate: format.Rate,
    dec: format.Decode,
    provenance: []const [2][]const u8 = &.{},
};

pub fn shardFileName(buf: []u8, layer: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "layer-{d}.safetensors", .{layer});
}

pub fn mtpShardFileName(buf: []u8, layer: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "mtp-{d}.safetensors", .{layer});
}

/// Writes `layer-{d}.safetensors` (`mtp-{d}.safetensors` for the MTP head) for a qwen4_exp bank; `prefix` is
/// `language_model.model` or `language_model.mtp`.
/// Does not touch `config.json`. Returns the `expert_quant` object, copied into `quant_buf`, for
/// the pack assembler to merge. A crash leaves the temp name, never a truncated shard.
pub fn writeLayer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, prefix: []const u8, layer: u32, banks: Layer, quant_buf: []u8) ![]const u8 {
    if (!prefixOk(prefix)) return error.PrefixMismatch;
    if (banks.rate.n % 2 != 0 or banks.rate.n < format.Rate.min_n or banks.rate.n > format.Rate.max_n) return error.RateUnsupported;
    if (banks.dec.window.bits() < 12 or banks.dec.window.bits() > 16) return error.WindowUnsupported;
    if (!bankOk(banks.gate, banks.rate) or !bankOk(banks.up, banks.rate) or !bankOk(banks.down, banks.rate)) return error.BankShape;
    var name_buf: [64]u8 = undefined;
    const final_name = if (std.mem.eql(u8, prefix, mtp_prefix)) try mtpShardFileName(&name_buf, layer) else try shardFileName(&name_buf, layer);
    if (dir.statFile(io, final_name, .{})) |_| return error.ShardExists else |_| {}

    var k_buf: [16]u8 = undefined;
    const k_text = banks.rate.kText(&k_buf);
    const quant = try std.fmt.bufPrint(quant_buf, "{{\"format\":\"exl3\",\"k\":{s},\"codebook\":\"{s}\",\"window\":{d}}}", .{ k_text, @tagName(banks.dec.codebook), banks.dec.window.bits() });

    const file_bytes = try packShard(allocator, prefix, layer, banks, k_text);
    defer allocator.free(file_bytes);
    var tmp_name_buf: [80]u8 = undefined;
    const tmp_name = try std.fmt.bufPrint(&tmp_name_buf, ".{s}.tmp", .{final_name});
    try dir.writeFile(io, .{ .sub_path = tmp_name, .data = file_bytes });
    publish(io, dir, tmp_name, final_name) catch |err| {
        if (!keepTemp(err)) dir.deleteFile(io, tmp_name) catch {};
        return err;
    };
    return quant;
}

fn prefixOk(prefix: []const u8) bool {
    if (!std.mem.eql(u8, prefix, trunk_prefix) and !std.mem.eql(u8, prefix, mtp_prefix)) return false;
    for (prefix) |c| {
        if (!((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or c == '_' or c == '.')) return false;
    }
    return true;
}

fn bankOk(bank: Bank, rate: format.Rate) bool {
    if (bank.experts == 0 or bank.in_features == 0 or bank.out_features == 0) return false;
    if (bank.in_features % 128 != 0 or bank.out_features % 128 != 0) return false;
    const tiles = (bank.in_features / 16) * (bank.out_features / 16);
    if (tiles == 0) return false;
    return bank.trellis.len == bank.experts * tiles * rate.n and
        bank.suh.len == bank.experts * bank.in_features and
        bank.svh.len == bank.experts * bank.out_features;
}

fn publish(io: std.Io, dir: std.Io.Dir, tmp_name: []const u8, final_name: []const u8) !void {
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_len = try dir.realPath(io, &dir_buf);
    const dir_path = dir_buf[0..dir_len];
    var tmp_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    var final_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try std.fmt.bufPrint(&tmp_path_buf, "{s}/{s}", .{ dir_path, tmp_name });
    const final_path = try std.fmt.bufPrint(&final_path_buf, "{s}/{s}", .{ dir_path, final_name });
    var tmp_z: [std.fs.max_path_bytes + 1]u8 = undefined;
    var final_z: [std.fs.max_path_bytes + 1]u8 = undefined;
    @memcpy(tmp_z[0..tmp_path.len], tmp_path);
    tmp_z[tmp_path.len] = 0;
    @memcpy(final_z[0..final_path.len], final_path);
    final_z[final_path.len] = 0;
    if (std.c.link(tmp_z[0..tmp_path.len :0].ptr, final_z[0..final_path.len :0].ptr) != 0) return linkFailure(std.c._errno().*);
    dir.deleteFile(io, tmp_name) catch {};
}

fn linkFailure(errno: c_int) error{ ShardExists, PublishFailed }!void {
    return if (errno == @intFromEnum(std.c.E.EXIST)) error.ShardExists else error.PublishFailed;
}

fn keepTemp(err: anyerror) bool {
    return err == error.PublishFailed;
}

fn packShard(allocator: std.mem.Allocator, prefix: []const u8, layer: u32, banks: Layer, k_text: []const u8) ![]u8 {
    const projs = [_]struct { name: []const u8, bank: Bank }{
        .{ .name = "gate_proj", .bank = banks.gate },
        .{ .name = "up_proj", .bank = banks.up },
        .{ .name = "down_proj", .bank = banks.down },
    };
    var header = std.ArrayList(u8).empty;
    defer header.deinit(allocator);
    try header.print(allocator, "{{\"__metadata__\":{{\"format\":\"exl3\",\"k\":\"{s}\",\"codebook\":\"{s}\",\"window\":\"{d}\"", .{ k_text, @tagName(banks.dec.codebook), banks.dec.window.bits() });
    for (banks.provenance) |pair| {
        try header.print(allocator, ",{f}:{f}", .{ std.json.fmt(pair[0], .{}), std.json.fmt(pair[1], .{}) });
    }
    try header.appendSlice(allocator, "}");
    var payload = std.ArrayList(u8).empty;
    defer payload.deinit(allocator);
    var offset: usize = 0;
    for (projs) |proj| {
        const bank = proj.bank;
        const parts = [_]struct { part: []const u8, dtype: []const u8, rank: usize, bytes: []const u8, d1: usize }{
            .{ .part = "trellis", .dtype = "U16", .rank = 4, .bytes = std.mem.sliceAsBytes(bank.trellis), .d1 = bank.in_features / 16 },
            .{ .part = "suh", .dtype = "F16", .rank = 2, .bytes = std.mem.sliceAsBytes(bank.suh), .d1 = bank.in_features },
            .{ .part = "svh", .dtype = "F16", .rank = 2, .bytes = std.mem.sliceAsBytes(bank.svh), .d1 = bank.out_features },
        };
        for (parts) |part| {
            try header.appendSlice(allocator, ",\"");
            try header.appendSlice(allocator, prefix);
            try header.print(allocator, ".layers.{d}.mlp.switch_mlp.", .{layer});
            try header.appendSlice(allocator, proj.name);
            try header.appendSlice(allocator, ".");
            try header.appendSlice(allocator, part.part);
            try header.appendSlice(allocator, "\":{\"dtype\":\"");
            try header.appendSlice(allocator, part.dtype);
            try header.print(allocator, "\",\"shape\":[{d},{d}", .{ bank.experts, part.d1 });
            if (part.rank == 4) try header.print(allocator, ",{d},{d}", .{ bank.out_features / 16, banks.rate.n });
            const end = offset + part.bytes.len;
            try header.print(allocator, "],\"data_offsets\":[{d},{d}]}}", .{ offset, end });
            try payload.appendSlice(allocator, part.bytes);
            offset = end;
        }
    }
    try header.appendSlice(allocator, "}");
    while (header.items.len % 8 != 0) try header.append(allocator, ' ');
    var file = std.ArrayList(u8).empty;
    errdefer file.deinit(allocator);
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.items.len, .little);
    try file.appendSlice(allocator, &len_buf);
    try file.appendSlice(allocator, header.items);
    try file.appendSlice(allocator, payload.items);
    return file.toOwnedSlice(allocator);
}

fn fillBank(allocator: std.mem.Allocator, prng: *std.Random.DefaultPrng, experts: usize, in_features: usize, out_features: usize, rate: format.Rate, dec: format.Decode) !Bank {
    var enc = try encode.Encoder.init(allocator, rate, dec);
    defer enc.deinit();
    const tiles_in = in_features / 16;
    const tiles_out = out_features / 16;
    const n = rate.n;
    const trellis = try allocator.alloc(u16, experts * tiles_in * tiles_out * n);
    const suh = try allocator.alloc(u16, experts * in_features);
    const svh = try allocator.alloc(u16, experts * out_features);
    for (suh, 0..) |*v, i| v.* = format.f32ToF16Bits(0.25 + @as(f32, @floatFromInt(i)) * 0.01);
    for (svh, 0..) |*v, i| v.* = format.f32ToF16Bits(-0.5 - @as(f32, @floatFromInt(i)) * 0.01);
    const r = prng.random();
    var tile: [encode.tile_len]f32 = undefined;
    for (0..experts) |e| {
        for (0..tiles_in) |ti| for (0..tiles_out) |to| {
            for (&tile) |*v| v.* = r.floatNorm(f32);
            const at = ((e * tiles_in + ti) * tiles_out + to) * n;
            _ = try enc.encodeTile(&tile, trellis[at..][0..n], null);
        };
    }
    return .{ .experts = experts, .in_features = in_features, .out_features = out_features, .trellis = trellis, .suh = suh, .svh = svh };
}

fn reconstructExpert(allocator: std.mem.Allocator, bank: Bank, expert: usize, rate: format.Rate, dec: format.Decode) ![]u16 {
    const tiles = (bank.in_features / 16) * (bank.out_features / 16);
    const trellis = bank.trellis[expert * tiles * rate.n ..][0 .. tiles * rate.n];
    const suh = bank.suh[expert * bank.in_features ..][0..bank.in_features];
    const svh = bank.svh[expert * bank.out_features ..][0..bank.out_features];
    const out = try allocator.alloc(u16, bank.in_features * bank.out_features);
    try format.reconstructPublic(allocator, trellis, suh, svh, bank.in_features, bank.out_features, rate, dec, out);
    return out;
}

test "shard: serialization and offsets round-trip through the loader and the engine decoder" {
    const t = std.testing;
    const io = t.io;
    const rate = format.Rate{ .n = 32 };
    const dec = format.Decode{ .codebook = .mcg, .window = .w12 };
    var prng = std.Random.DefaultPrng.init(19);
    const gate = try fillBank(t.allocator, &prng, 2, 128, 128, rate, dec);
    defer t.allocator.free(gate.trellis);
    defer t.allocator.free(gate.suh);
    defer t.allocator.free(gate.svh);
    const up = try fillBank(t.allocator, &prng, 2, 128, 128, rate, dec);
    defer t.allocator.free(up.trellis);
    defer t.allocator.free(up.suh);
    defer t.allocator.free(up.svh);
    const down = try fillBank(t.allocator, &prng, 2, 128, 128, rate, dec);
    defer t.allocator.free(down.trellis);
    defer t.allocator.free(down.suh);
    defer t.allocator.free(down.svh);
    var want: [3][2][]u16 = undefined;
    for ([_]Bank{ gate, up, down }, 0..) |bank, p| {
        for (0..2) |e| want[p][e] = try reconstructExpert(t.allocator, bank, e, rate, dec);
    }
    defer for (want) |proj| for (proj) |row| t.allocator.free(row);

    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var quant_buf: [160]u8 = undefined;
    const quant = try writeLayer(t.allocator, io, tmp.dir, "language_model.model", 0, .{
        .gate = gate,
        .up = up,
        .down = down,
        .rate = rate,
        .dec = dec,
    }, &quant_buf);
    const cfg_bytes = try std.fmt.allocPrint(t.allocator, "{{\"expert_quant\":{s}}}", .{quant});
    defer t.allocator.free(cfg_bytes);
    const cfg = try std.json.parseFromSlice(std.json.Value, t.allocator, cfg_bytes, .{});
    defer cfg.deinit();
    const spec = try api.parseExpertQuant(cfg.value.object);
    try t.expectEqual(rate.n, spec.rate.n);
    try t.expectEqual(dec.codebook, spec.codebook);
    try t.expectEqual(dec.window, spec.window);
    try t.expectError(error.FileNotFound, tmp.dir.access(io, "config.json", .{}));

    const raw = try tmp.dir.readFileAlloc(io, "layer-0.safetensors", t.allocator, .limited(1 << 22));
    defer t.allocator.free(raw);
    const aligned = try t.allocator.alignedAlloc(u8, .of(u16), raw.len);
    defer t.allocator.free(aligned);
    @memcpy(aligned, raw);
    var file = try st.parse(t.allocator, aligned);
    defer file.deinit(t.allocator);
    var k_buf: [16]u8 = undefined;
    try t.expectEqualStrings(rate.kText(&k_buf), file.metadata.get("k").?);
    try t.expectEqualStrings(@tagName(dec.codebook), file.metadata.get("codebook").?);
    var win_buf: [8]u8 = undefined;
    try t.expectEqualStrings(try std.fmt.bufPrint(&win_buf, "{d}", .{dec.window.bits()}), file.metadata.get("window").?);
    try t.expectEqualStrings("exl3", file.metadata.get("format").?);

    var plan = api.GroupLayout.init(128, 128, rate);
    const projs = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
    const banks = [_]Bank{ gate, up, down };
    for (projs, banks) |proj, bank| {
        for ([_][]const u8{ "trellis", "suh", "svh" }) |part| {
            var key: [160]u8 = undefined;
            const name = try std.fmt.bufPrint(&key, "language_model.model.layers.0.mlp.switch_mlp.{s}.{s}", .{ proj, part });
            const view = file.tensors.get(name) orelse return error.MissingTensor;
            const suffix = name["language_model.model.layers.0.mlp.switch_mlp.".len..];
            var shape: [4]u64 = undefined;
            for (view.shape, 0..) |d, i| shape[i] = d;
            const dtype: api.group_layout.Dtype = if (std.mem.eql(u8, part, "trellis")) .u16 else .f16;
            _ = try plan.add(suffix, shape[0..view.shape.len], dtype);
            _ = bank;
        }
    }
    try plan.finish(2, 1, false, 2);

    for (projs, banks, 0..) |proj, bank, p| {
        var key: [160]u8 = undefined;
        const trellis = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&key, "language_model.model.layers.0.mlp.switch_mlp.{s}.trellis", .{proj})).?);
        const suh = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&key, "language_model.model.layers.0.mlp.switch_mlp.{s}.suh", .{proj})).?);
        const svh = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&key, "language_model.model.layers.0.mlp.switch_mlp.{s}.svh", .{proj})).?);
        for (0..2) |e| {
            const tiles = 8 * 8;
            const got = try reconstructExpert(t.allocator, .{
                .experts = 2,
                .in_features = bank.in_features,
                .out_features = bank.out_features,
                .trellis = trellis,
                .suh = suh,
                .svh = svh,
            }, e, rate, dec);
            defer t.allocator.free(got);
            try t.expectEqualSlices(u16, want[p][e], got);
            try t.expectEqualSlices(u16, bank.trellis, trellis);
            try t.expectEqualSlices(u16, bank.suh, suh);
            try t.expectEqualSlices(u16, bank.svh, svh);
            _ = tiles;
        }
    }
}

test "shard: the writer refuses a bad rate, a window outside 12..16, and a bank whose bytes do not match" {
    const t = std.testing;
    const io = t.io;
    const empty: []const u16 = &.{};
    const gate = Bank{ .experts = 1, .in_features = 128, .out_features = 128, .trellis = empty, .suh = empty, .svh = empty };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var quant: [160]u8 = undefined;
    try t.expectError(error.BankShape, writeLayer(t.allocator, io, tmp.dir, "language_model.model", 0, .{
        .gate = gate,
        .up = gate,
        .down = gate,
        .rate = .{ .n = 32 },
        .dec = .{ .codebook = .mcg, .window = .w12 },
    }, &quant));
    try t.expectError(error.RateUnsupported, writeLayer(t.allocator, io, tmp.dir, "language_model.model", 0, .{
        .gate = gate,
        .up = gate,
        .down = gate,
        .rate = .{ .n = 15 },
        .dec = .{ .codebook = .mcg, .window = .w12 },
    }, &quant));
    try t.expectError(error.WindowUnsupported, writeLayer(t.allocator, io, tmp.dir, "language_model.model", 0, .{
        .gate = gate,
        .up = gate,
        .down = gate,
        .rate = .{ .n = 32 },
        .dec = .{ .codebook = .mcg, .window = .w11 },
    }, &quant));
}

test "shard: a mismatched bank, a foreign prefix, or an existing shard is refused" {
    const t = std.testing;
    const io = t.io;
    const rate = format.Rate{ .n = 32 };
    const dec = format.Decode{ .codebook = .mcg, .window = .w12 };
    const empty: []const u16 = &.{};
    const gate = Bank{ .experts = 1, .in_features = 128, .out_features = 128, .trellis = empty, .suh = empty, .svh = empty };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var quant: [160]u8 = undefined;
    const layer = Layer{ .gate = gate, .up = gate, .down = gate, .rate = rate, .dec = dec };
    try t.expectError(error.BankShape, writeLayer(t.allocator, io, tmp.dir, "language_model.model", 0, layer, &quant));
    try t.expectError(error.PrefixMismatch, writeLayer(t.allocator, io, tmp.dir, "model.layers", 0, layer, &quant));
    try t.expectError(error.PrefixMismatch, writeLayer(t.allocator, io, tmp.dir, "language_model.model\"", 0, layer, &quant));
}

test "shard: quantizeNamed written and loaded matches its reconstruction and stays near the weight" {
    const t = std.testing;
    const io = t.io;
    const in: usize = 128;
    const out: usize = 128;
    var prng = std.Random.DefaultPrng.init(29);
    const r = prng.random();
    const weight = try t.allocator.alloc(f32, in * out);
    defer t.allocator.free(weight);
    const hessian = try t.allocator.alloc(f32, in * in);
    defer t.allocator.free(hessian);
    for (weight) |*v| v.* = r.floatNorm(f32) * 0.05;
    for (0..in) |i| for (0..in) |j| {
        var s: f32 = if (i == j) 0.05 else 0;
        for (0..16) |m| s += weight[i * out + m] * weight[j * out + m];
        hessian[i * in + j] = s;
    };
    const dec = format.Decode{ .codebook = .mcg, .window = .w12 };
    const rate = format.Rate{ .n = 32 };
    var q = try quantize.quantizeNamed(t.allocator, weight, hessian, in, out, 3, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.e0", rate, dec, .never);
    defer q.deinit(t.allocator);
    const memory = try t.allocator.alloc(u16, in * out);
    defer t.allocator.free(memory);
    try format.reconstructPublic(t.allocator, q.packed_out, q.suh, q.svh, in, out, rate, dec, memory);
    const bank = Bank{ .experts = 1, .in_features = in, .out_features = out, .trellis = q.packed_out, .suh = q.suh, .svh = q.svh };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var quant_buf: [160]u8 = undefined;
    _ = try writeLayer(t.allocator, io, tmp.dir, "language_model.model", 0, .{ .gate = bank, .up = bank, .down = bank, .rate = rate, .dec = dec }, &quant_buf);
    const raw = try tmp.dir.readFileAlloc(io, "layer-0.safetensors", t.allocator, .limited(1 << 22));
    defer t.allocator.free(raw);
    const aligned = try t.allocator.alignedAlloc(u8, .of(u16), raw.len);
    defer t.allocator.free(aligned);
    @memcpy(aligned, raw);
    var file = try st.parse(t.allocator, aligned);
    defer file.deinit(t.allocator);
    const loaded = try t.allocator.alloc(u16, in * out);
    defer t.allocator.free(loaded);
    try format.reconstructPublic(t.allocator, st.slice(u16, file.tensors.get("language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis").?), st.slice(u16, file.tensors.get("language_model.model.layers.0.mlp.switch_mlp.gate_proj.suh").?), st.slice(u16, file.tensors.get("language_model.model.layers.0.mlp.switch_mlp.gate_proj.svh").?), in, out, rate, dec, loaded);
    try t.expectEqualSlices(u16, memory, loaded);
    var num: f64 = 0;
    var den: f64 = 0;
    for (weight, loaded) |src, bits| {
        const d = @as(f64, format.f16BitsToF32(bits)) - src;
        num += d * d;
        den += @as(f64, src) * src;
    }
    const rel = @sqrt(num / den);
    // K2 on these weights: the tile fixtures at n=32 spend about 20 squared error over 256
    // unit-scale weights (RMS ~0.28). LDLQ does not multiply that by more than about 2, so
    // a public relative RMS of 0.75 is the failure line, not a tuned fit.
    if (!(rel < 0.75)) return error.QuantizedErrorTooLarge;
}
test "shard: writing a layer twice refuses instead of clobbering the shard" {
    const t = std.testing;
    const io = t.io;
    const n: u32 = 32;
    const trellis = try t.allocator.alloc(u16, 8 * 8 * n);
    defer t.allocator.free(trellis);
    const suh = try t.allocator.alloc(u16, 128);
    defer t.allocator.free(suh);
    const svh = try t.allocator.alloc(u16, 128);
    defer t.allocator.free(svh);
    @memset(trellis, 1);
    @memset(suh, 1);
    @memset(svh, 1);
    const bank = Bank{ .experts = 1, .in_features = 128, .out_features = 128, .trellis = trellis, .suh = suh, .svh = svh };
    const layer = Layer{ .gate = bank, .up = bank, .down = bank, .rate = .{ .n = n }, .dec = .{ .codebook = .mcg, .window = .w12 } };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var quant: [160]u8 = undefined;
    _ = try writeLayer(t.allocator, io, tmp.dir, "language_model.model", 0, layer, &quant);
    try t.expectError(error.ShardExists, writeLayer(t.allocator, io, tmp.dir, "language_model.model", 0, layer, &quant));
}

test "shard: only an existing destination is ShardExists; any other link failure keeps the temp" {
    const t = std.testing;
    try t.expectError(error.ShardExists, linkFailure(@intFromEnum(std.c.E.EXIST)));
    try t.expectError(error.PublishFailed, linkFailure(@intFromEnum(std.c.E.NOSPC)));
    try t.expectError(error.PublishFailed, linkFailure(@intFromEnum(std.c.E.ACCES)));
    try t.expect(keepTemp(error.PublishFailed));
    try t.expect(!keepTemp(error.ShardExists));
}

test "shard: provenance pairs ride beside the engine's stamp keys" {
    const t = std.testing;
    const io = t.io;
    const rate = format.Rate{ .n = 32 };
    const dec = format.Decode{ .codebook = .mcg, .window = .w12 };
    var prng = std.Random.DefaultPrng.init(41);
    const bank = try fillBank(t.allocator, &prng, 1, 128, 128, rate, dec);
    defer t.allocator.free(bank.trellis);
    defer t.allocator.free(bank.suh);
    defer t.allocator.free(bank.svh);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var quant_buf: [160]u8 = undefined;
    const pairs = [_][2][]const u8{ .{ "encoder", "sushi-encoder-1" }, .{ "calib_tokens", "65536" } };
    _ = try writeLayer(t.allocator, io, tmp.dir, "language_model.model", 3, .{ .gate = bank, .up = bank, .down = bank, .rate = rate, .dec = dec, .provenance = &pairs }, &quant_buf);
    const raw = try tmp.dir.readFileAlloc(io, "layer-3.safetensors", t.allocator, .limited(1 << 22));
    defer t.allocator.free(raw);
    const aligned = try t.allocator.alignedAlloc(u8, .of(u16), raw.len);
    defer t.allocator.free(aligned);
    @memcpy(aligned, raw);
    var file = try st.parse(t.allocator, aligned);
    defer file.deinit(t.allocator);
    try t.expectEqualStrings("sushi-encoder-1", file.metadata.get("encoder").?);
    try t.expectEqualStrings("65536", file.metadata.get("calib_tokens").?);
    try t.expectEqualStrings("exl3", file.metadata.get("format").?);
}

test "shard: the MTP head's bank takes its own prefix and file name" {
    const t = std.testing;
    const io = t.io;
    const rate = format.Rate{ .n = 32 };
    const dec = format.Decode{ .codebook = .mcg, .window = .w12 };
    var prng = std.Random.DefaultPrng.init(43);
    const bank = try fillBank(t.allocator, &prng, 1, 128, 128, rate, dec);
    defer t.allocator.free(bank.trellis);
    defer t.allocator.free(bank.suh);
    defer t.allocator.free(bank.svh);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var quant_buf: [160]u8 = undefined;
    _ = try writeLayer(t.allocator, io, tmp.dir, "language_model.mtp", 0, .{ .gate = bank, .up = bank, .down = bank, .rate = rate, .dec = dec }, &quant_buf);
    const raw = try tmp.dir.readFileAlloc(io, "mtp-0.safetensors", t.allocator, .limited(1 << 22));
    defer t.allocator.free(raw);
    const aligned = try t.allocator.alignedAlloc(u8, .of(u16), raw.len);
    defer t.allocator.free(aligned);
    @memcpy(aligned, raw);
    var file = try st.parse(t.allocator, aligned);
    defer file.deinit(t.allocator);
    try t.expect(file.tensors.get("language_model.mtp.layers.0.mlp.switch_mlp.down_proj.suh") != null);
    try t.expectError(error.PrefixMismatch, writeLayer(t.allocator, io, tmp.dir, "language_model.other", 0, .{ .gate = bank, .up = bank, .down = bank, .rate = rate, .dec = dec }, &quant_buf));
}
