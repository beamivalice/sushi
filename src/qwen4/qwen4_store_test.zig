const std = @import("std");
const io_mod = @import("../expert_io.zig");
const store_mod = @import("../expert_store.zig");

const QuantGeom = store_mod.QuantGeom;
const QuantStore = store_mod.QuantStore;
const Component = store_mod.Component;
const component_count = store_mod.component_count;
const tensorKey = store_mod.tensorKey;
const affineGeomFromShapes = store_mod.affineGeomFromShapes;

test "real quantized pack resolves nine regions per layer and the per expert bill" {
    const t = std.testing;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try @import("../test_models.zig").packPath(&path_buf, "Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit");
    var dir = std.Io.Dir.openDirAbsolute(t.io, path, .{}) catch return error.SkipZigTest;
    defer dir.close(t.io);
    var store = try QuantStore.open(t.allocator, path, .{ .layers = 48, .experts = 512, .hidden = 2560, .intermediate = 640 });
    defer store.deinit();

    try t.expectEqual(@as(u64, 819_200), store.slotBytes(.gate_w));
    try t.expectEqual(@as(u64, 51_200), store.slotBytes(.gate_s));
    try t.expectEqual(@as(u64, 51_200), store.slotBytes(.gate_b));
    try t.expectEqual(@as(u64, 819_200), store.slotBytes(.down_w));
    try t.expectEqual(@as(u64, 51_200), store.slotBytes(.down_s));
    try t.expectEqual(@as(u64, 2_764_800), store.expertBytes());
    try t.expectEqual(QuantGeom{ .bits = 4, .group_size = 64 }, store.geomOf(.gate_w));
    try t.expectEqual(QuantGeom{ .bits = 4, .group_size = 64 }, store.geomOf(.down_w));
    try t.expectEqual(@as(u32, 640), store.rowsOf(.gate_w));
    try t.expectEqual(@as(u32, 320), store.colsOf(.gate_w));
    try t.expectEqual(@as(u32, 2560), store.rowsOf(.down_w));
    try t.expectEqual(@as(u32, 80), store.colsOf(.down_w));

    // The direct arm re-derives every offset from the shard header, so a wrong
    // stride or a swapped component fails here instead of agreeing with itself.
    const layer: u16 = 3;
    const expert: u16 = 129;
    const total: usize = @intCast(store.expertBytes());
    const through = try t.allocator.alloc(u8, total);
    defer t.allocator.free(through);
    const direct = try t.allocator.alloc(u8, total);
    defer t.allocator.free(direct);
    try store.readExpert(layer, expert, through);
    const index_raw = try dir.readFileAlloc(t.io, "model.safetensors.index.json", t.allocator, .limited(64 * 1024 * 1024));
    defer t.allocator.free(index_raw);
    const index_parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, index_raw, .{});
    defer index_parsed.deinit();
    const weight_map = index_parsed.value.object.get("weight_map").?.object;
    var at: usize = 0;
    for (0..component_count) |ci| {
        const c: Component = @fromBackingInt(@intCast(ci));
        var key_buf: [192]u8 = undefined;
        const key = try tensorKey(&key_buf, layer, c);
        const shard = try std.fmt.allocPrintSentinel(t.allocator, "{s}/{s}", .{ path, weight_map.get(key).?.string }, 0);
        defer t.allocator.free(shard);
        const fd = try io_mod.openHinted(shard, .{});
        defer _ = std.c.close(fd);
        const region = try io_mod.tensorRegion(t.allocator, fd, key);
        const per = region.tensor_bytes / region.shape[0];
        const offset = region.data_offset + region.tensor_offset + @as(u64, expert) * per;
        try t.expectEqual(per, store.slotBytes(c));
        try t.expectEqual(offset, store.span(layer, expert, c).offset);
        const len: usize = @intCast(per);
        try io_mod.readExact(fd, direct[at..][0..len], offset);
        at += len;
    }
    try t.expectEqual(total, at);
    try t.expectEqualSlices(u8, direct, through);

    // A collapsed stride or a duplicated component would still agree above.
    const neighbour = try t.allocator.alloc(u8, total);
    defer t.allocator.free(neighbour);
    try store.readExpert(layer, expert + 1, neighbour);
    try t.expect(!std.mem.eql(u8, through, neighbour));
    const gate_len: usize = @intCast(store.slotBytes(.gate_w));
    const up_at: usize = @intCast(store.slotBytes(.gate_w) + store.slotBytes(.gate_s) + store.slotBytes(.gate_b));
    try t.expect(!std.mem.eql(u8, through[0..gate_len], through[up_at..][0..gate_len]));
}

test "an eight bit tensor beside four bit ones solves to its own width" {
    const t = std.testing;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try @import("../test_models.zig").packPath(&path_buf, "Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit");
    var dir = std.Io.Dir.openDirAbsolute(t.io, path, .{}) catch return error.SkipZigTest;
    dir.close(t.io);
    const shard = try std.fmt.allocPrintSentinel(t.allocator, "{s}/model-00051.safetensors", .{path}, 0);
    defer t.allocator.free(shard);
    const fd = try io_mod.openHinted(shard, .{});
    defer _ = std.c.close(fd);
    const w = try io_mod.tensorRegion(t.allocator, fd, "language_model.model.layers.3.mlp.shared_expert.gate_proj.weight");
    const sc = try io_mod.tensorRegion(t.allocator, fd, "language_model.model.layers.3.mlp.shared_expert.gate_proj.scales");
    try t.expectEqual(io_mod.Dtype.u32, w.dtype);
    try t.expectEqual(io_mod.Dtype.bf16, sc.dtype);
    try t.expectEqual(QuantGeom{ .bits = 8, .group_size = 64 }, affineGeomFromShapes(w.shape[1], sc.shape[1], 2560).?);
}
