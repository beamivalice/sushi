//! Flash-Next's pack preflight: a rate-grouped pack's router widths and the streaming refusal.
const std = @import("std");
const model = @import("../model.zig");
const pack_index = @import("../pack_index.zig");
const Allocator = std.mem.Allocator;
const TestTensor = pack_index.TestTensor;
const TestIndexEntry = pack_index.TestIndexEntry;
const testBf16Bytes = pack_index.testBf16Bytes;
const writeTestShard = pack_index.writeTestShard;
const writeTestIndex = pack_index.writeTestIndex;
const validateExl3Pack = pack_index.validateExl3Pack;

/// A two-group Flash-Next pack: trunk layers 0 and 1, then the MTP head, with a router `layer_experts[i]` wide each.
fn writeRaggedQwenFixture(io: std.Io, alloc: Allocator, dir: std.Io.Dir, layer_experts: [3]u64) !void {
    var tensors: std.ArrayList(TestTensor) = .empty;
    var entries: std.ArrayList(TestIndexEntry) = .empty;
    for ([_][]const u8{ "language_model.model.layers.0", "language_model.model.layers.1", "language_model.mtp.layers.0" }, 0..) |base, layer| {
        const e = layer_experts[layer];
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
    try writeTestShard(io, alloc, dir, "ragged.safetensors", tensors.items);
    try writeTestIndex(io, alloc, dir, entries.items);
}

test "sushi coder qwen trunk and MTP preflight use each router width" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try writeRaggedQwenFixture(t.io, alloc, tmp.dir, .{ 3, 2, 3 });
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

test "sushi coder a rate-grouped Flash-Next pack passes the streaming preflight, a pruned one does not" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try writeRaggedQwenFixture(t.io, alloc, tmp.dir, .{ 3, 3, 3 });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const len = try tmp.dir.realPath(t.io, &buf);
    var cfg = model.ModelConfig{ .model_type = "qwen4_exp", .expert_layout = .exl3_k4, .num_hidden_layers = 2, .num_experts = 3, .num_experts_per_tok = 2, .hidden_size = 128, .moe_intermediate_size = 128, .expert_streaming = true };
    try validateExl3Pack(t.io, alloc, buf[0..len], &cfg);
    cfg.num_experts = 512;
    try t.expectError(error.Exl3RaggedStreamingUnsupported, validateExl3Pack(t.io, alloc, buf[0..len], &cfg));
}
