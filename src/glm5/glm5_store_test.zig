const std = @import("std");
const io_mod = @import("../expert_io.zig");
const store_mod = @import("../expert_store.zig");
const fixture = @import("glm_stream_fixture.zig");
const mlx = @import("../mlx.zig");
const cli = @import("../cli.zig");
const expert_stream = @import("../expert_stream.zig");
const expectError = @import("../test_expect.zig").expectError;

const Geometry = store_mod.Geometry;
const QuantStore = store_mod.QuantStore;
const Layout = store_mod.Layout;
const Projection = store_mod.Projection;
const layoutOfDirWithFirstMoe = store_mod.layoutOfDirWithFirstMoe;
const weightOf = store_mod.weightOf;
const scalesOf = store_mod.scalesOf;
const biasesOf = store_mod.biasesOf;

test "GLM BF16 individual source preserves shard bytes and leaves dense prefix and MTP unopened" {
    const t = std.testing;
        var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.write(t.allocator, tmp.dir, .none);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const geometry = Geometry{ .layers = 4, .experts = 4, .hidden = 32, .intermediate = 16, .first_moe_layer = 3 };
    var store = try QuantStore.openForLayout(t.allocator, path_buf[0..path_len], geometry, .bf16_individual);
    defer store.deinit();
    try t.expectEqual(@as(usize, 3), store.files.len);
    try t.expectEqual(@as(u64, 3072), store.expertBytes());
    for (0..3) |layer| try t.expect(!store.hasExpertLayer(@intCast(layer)));
    try t.expect(store.hasExpertLayer(3));
    try t.expect(!store.hasExpertLayer(4));
    var got: [3072]u8 = undefined;
    for (0..4) |expert| {
        try store.readExpert(3, @intCast(expert), &got);
        for ([_]Projection{ .gate, .up, .down }, 0..) |projection, pi| {
            const c = weightOf(projection);
            try t.expectEqual(io_mod.Dtype.bf16, store.dtypeOf(c));
            try t.expectEqual(@as(u32, if (pi == 2) 32 else 16), store.rowsOf(c));
            try t.expectEqual(@as(u32, if (pi == 2) 16 else 32), store.colsOf(c));
            try t.expect(!store.componentPresent(scalesOf(projection)));
            try t.expect(!store.componentPresent(biasesOf(projection)));
            for (0..512) |i| try t.expectEqual(fixture.value(expert, pi), std.mem.readInt(u16, got[pi * 1024 + i * 2 ..][0..2], .little));
        }
    }
    try t.expectError(error.ExpertLayerAbsent, store.readExpert(0, 0, &got));
    try t.expectError(error.ExpertOutOfRange, store.readExpert(3, 4, &got));
}

test "GLM FP8 individual source resolves its layout and keeps e4m3 codes and block scales as stored" {
    const t = std.testing;
        var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.writeStorage(t.allocator, tmp.dir, .none, 128, 128, .fp8);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(t.io, &path_buf)];
    try t.expectEqual(Layout.fp8_individual, layoutOfDirWithFirstMoe(t.allocator, t.io, "glm5_next", path, 4, 3).?);
    const geometry = Geometry{ .layers = 4, .experts = 4, .hidden = 128, .intermediate = 128, .first_moe_layer = 3 };
    var store = try QuantStore.openForLayout(t.allocator, path, geometry, .fp8_individual);
    defer store.deinit();
    try t.expectEqual(@as(u64, 3 * (128 * 128 + 4)), store.expertBytes());
    var got: [3 * (128 * 128 + 4)]u8 = undefined;
    for (0..4) |expert| {
        try store.readExpert(3, @intCast(expert), &got);
        var at: usize = 0;
        for ([_]Projection{ .gate, .up, .down }, 0..) |projection, pi| {
            try t.expectEqual(io_mod.Dtype.f8e4m3, store.dtypeOf(weightOf(projection)));
            try t.expectEqual(io_mod.Dtype.f32, store.dtypeOf(scalesOf(projection)));
            try t.expect(!store.componentPresent(biasesOf(projection)));
            for (0..128 * 128) |i| try t.expectEqual(fixture.fp8Code(expert, pi, i), got[at + i]);
            at += 128 * 128;
            try t.expectEqual(fixture.fp8Scale(expert, pi, 0), @as(f32, @bitCast(std.mem.readInt(u32, got[at..][0..4], .little))));
            at += 4;
        }
    }
    var missing = t.tmpDir(.{});
    defer missing.cleanup();
    try fixture.writeStorage(t.allocator, missing.dir, .fp8_scale, 128, 128, .fp8);
    const missing_path = path_buf[0..try missing.dir.realPath(t.io, &path_buf)];
    try t.expect(layoutOfDirWithFirstMoe(t.allocator, t.io, "glm5_next", missing_path, 4, 3) == null);
    try t.expectError(error.MissingExpertTensor, QuantStore.openForLayout(t.allocator, missing_path, geometry, .fp8_individual));
}

test "GLM BF16 individual source rejects incomplete quantized or malformed banks" {
    const t = std.testing;
        for ([_]fixture.Fault{ .missing, .fp16, .wrong_shape, .truncated, .dense_prefix, .mixed, .fp8_scale }) |fault| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        try fixture.write(t.allocator, tmp.dir, fault);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try tmp.dir.realPath(t.io, &path_buf);
        const geometry = Geometry{ .layers = 4, .experts = 4, .hidden = 32, .intermediate = 16, .first_moe_layer = 3 };
        const result = QuantStore.openForLayout(t.allocator, path_buf[0..path_len], geometry, .bf16_individual);
        const expected: anyerror = switch (fault) {
            .missing => error.MissingExpertTensor,
            .fp16, .wrong_shape, .fp8_scale => error.InvalidExpertTensor,
            .truncated => error.ExpertTensorOutOfBounds,
            .dense_prefix => error.ExpertLayerOutOfRange,
            .mixed => error.MixedExpertBankGeometry,
            .none => unreachable,
        };
        try t.expectError(expected, result);
    }
}

test "GLM BF16 stream imports split expert slabs and preserves bytes across cache and union" {
    try mlx.requireGpu();
    const t = std.testing;
        var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.write(t.allocator, tmp.dir, .none);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const geometry = Geometry{ .layers = 4, .experts = 4, .hidden = 32, .intermediate = 16, .first_moe_layer = 3 };
    const s = mlx.gpuStream();
    var engine = try expert_stream.Engine.initWithOptions(t.allocator, path_buf[0..path_len], geometry, 2 * 3072, s, .{ .layout = .bf16_individual, .bounce_size = 1 << 20 });
    defer engine.deinit();
    try t.expectEqual(@as(u64, 6), engine.slab_imports);
    try t.expectEqual(@as(u64, 0), engine.fallback_imports);
    try t.expectEqual(@as(u64, 3072), engine.store.perExpertBytes());
    try expectError(error.ExpertLayerAbsent, engine.prepareHost(0, &.{0}));
    try t.expect((try engine.specRoute(3)) == null);
    const groups = [_][]const u16{ &.{ 3, 1 }, &.{ 1, 3, 1 }, &.{ 0, 1, 2, 3 }, &.{ 2, 0 } };
    for (groups, 0..) |ids, group| {
        var prepared = try engine.prepareHost(3, ids);
        defer prepared.deinit();
        try t.expect(!prepared.quantized);
        try t.expectEqual(group == 2, prepared.workspace);
        for ([_]mlx.mlx_array{ prepared.gate, prepared.up, prepared.down }, 0..) |array, pi| {
            try t.expect(array.ctx != null);
            try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(array));
            const shape = mlx.getShape(array);
            try t.expectEqual(@as(usize, 3), shape.len);
            try t.expectEqual(@as(c_int, if (pi == 2) 16 else 32), shape[1]);
            try t.expectEqual(@as(c_int, if (pi == 2) 32 else 16), shape[2]);
            const input_dim: c_int = if (pi == 2) 16 else 32;
            const input_data: [32]u16 = @splat(0x3f80);
            const input = mlx.mlx_array_new_data(&input_data, &[_]c_int{ 1, 1, input_dim }, 3, .bfloat16);
            defer _ = mlx.mlx_array_free(input);
            const first_slot = [_]u32{prepared.remapped[0]};
            const selected = mlx.mlx_array_new_data(&first_slot, &[_]c_int{1}, 1, .uint32);
            defer _ = mlx.mlx_array_free(selected);
            var result = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(result);
            try mlx.check(mlx.mlx_gather_mm(&result, input, array, .{ .ctx = null }, selected, false, s));
            try mlx.check(mlx.mlx_array_eval(result));
            const result_data = mlx.mlx_array_data_bfloat16(result).?;
            // Summing 16 or 32 identical powers-of-two-scaled BF16 values is exact.
            const expected = fixture.value(ids[0], pi) + @as(u16, if (pi == 2) 4 else 5) * 128;
            for (result_data[0..mlx.mlx_array_size(result)]) |v| try t.expectEqual(expected, v);
            const ci = pi * 3;
            for (ids, prepared.remapped) |expert, slot| {
                const bytes = if (prepared.workspace) engine.union_slabs[ci].slotBytes(slot) else engine.cacheSlotBytesAt(3, slot, ci);
                for (0..512) |i| try t.expectEqual(fixture.value(expert, pi), std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little));
            }
        }
        if (group == 1) try t.expectEqual(@as(u64, 2), engine.fill_experts_total);
    }
}

test "a relative pack directory resolves to the absolute one the layout probe can read" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDir(io, "pack", .default_dir);
    var pack = try tmp.dir.openDir(io, "pack", .{});
    defer pack.close(io);
    try fixture.writeStorage(a, pack, .none, 128, 128, .fp8);

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_len = try std.Io.Dir.cwd().realPath(io, &cwd_buf);
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_len = try tmp.dir.realPath(io, &tmp_buf);
    const rel = try std.fs.path.relative(a, cwd_buf[0..cwd_len], null, cwd_buf[0..cwd_len], tmp_buf[0..tmp_len]);
    defer a.free(rel);
    const rel_pack = try std.fs.path.join(a, &.{ rel, "pack" });
    defer a.free(rel_pack);
    try std.testing.expect(!std.fs.path.isAbsolute(rel_pack));

    const abs = cli.resolveRelativeDir(a, io, rel_pack) orelse return error.TestUnexpectedResult;
    defer a.free(abs);
    const want = try std.fs.path.join(a, &.{ tmp_buf[0..tmp_len], "pack" });
    defer a.free(want);
    try std.testing.expectEqualStrings(want, abs);
    const expert_quant = @import("../expert_quant.zig");
    try std.testing.expect(expert_quant.layoutOfDirWithFirstMoe(a, io, "glm5_next", rel_pack, 4, 3) == null);
    try std.testing.expectEqual(expert_quant.Layout.fp8_individual, expert_quant.layoutOfDirWithFirstMoe(a, io, "glm5_next", abs, 4, 3).?);
}
