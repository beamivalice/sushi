const std = @import("std");
const io_mod = @import("../expert_io.zig");
const store_mod = @import("../expert_store.zig");
const fixture = @import("glm_stream_fixture.zig");

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
