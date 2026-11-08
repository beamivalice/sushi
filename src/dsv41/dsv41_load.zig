//! DeepSeek-V4.1's own load (the `load` slot of `arch.zig`): its resident trunk from the release's shards, read by name.
const std = @import("std");
const model = @import("../model.zig");
const log = @import("../log.zig");
const expert_quant = @import("../expert_quant.zig");
const source = @import("dsv41_source.zig");
const arch = @import("arch.zig");
const ModelConfig = model.ModelConfig;

/// The resident trunk, and the vision tower when asked for; the routed experts stream.
pub fn weights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool) !?model.Weights {
    const stq = config.dsv41.?.expert_dec != null;
    if (config.expert_layout != (if (stq) expert_quant.Layout.exl3_k4 else .mxfp4_w123)) {
        log.err("deepseek_v41: model.safetensors.index.json does not list every routed expert (layout {s}); the download is incomplete\n", .{@tagName(config.expert_layout)});
        return error.ExpertLayoutUnsupported;
    }
    try source.validateTileOrder(io, allocator, model_dir, config.expert_tile_order, config.hidden_size, config.moe_intermediate_size);
    return try source.loadResident(io, allocator, model_dir, .{ .dspark = arch.dsparkOn(config), .vision = load_vision and config.has_vision });
}

/// Whether the pack ships the DSpark stages its drafter runs on.
pub fn shipsHead(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    return source.shipsDspark(io, allocator, model_dir);
}
