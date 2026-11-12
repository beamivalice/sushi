//! DeepSeek-V4.1's own load (the `load` slot of `arch.zig`): its resident trunk from the release's shards, read by name.
const std = @import("std");
const model = @import("../model.zig");
const log = @import("../log.zig");
const expert_quant = @import("../expert_quant.zig");
const source = @import("dsv41_source.zig");
const arch = @import("arch.zig");
const mlx = @import("../mlx.zig");
const dsv41_borrow = @import("dsv41_borrow.zig");
const ModelConfig = model.ModelConfig;

/// The resident trunk, and the vision tower when asked for; the routed experts stream.
pub fn weights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool) !?model.Weights {
    const stq = config.dsv41.?.expert_dec != null;
    if (config.expert_layout != (if (stq) expert_quant.Layout.exl3_k4 else .mxfp4_w123)) {
        log.err("deepseek_v41: model.safetensors.index.json does not list every routed expert (layout {s}); the download is incomplete\n", .{@tagName(config.expert_layout)});
        return error.ExpertLayoutUnsupported;
    }
    try source.validateTileOrder(io, allocator, model_dir, config.expert_tile_order, config.hidden_size, config.moe_intermediate_size);
    // The engine is on before its first weight is wrapped; `initDsv41` binds them under the same answer.
    const own_weights = dsv41_borrow.ownDecode(&config.dsv41.?);
    if (own_weights) mlx.engine.enable();
    pipelines_before = mlx.engine.pipelinesMade().count;
    mlx.engine.own_weights = own_weights;
    defer mlx.engine.own_weights = false;
    return try source.loadResident(io, allocator, model_dir, .{ .dspark = arch.dsparkOn(config), .vision = load_vision and config.has_vision });
}

/// The engine's pipelines before this load wrapped its first weight: the load's own ops on its weights make some, and
/// the load line counts them with the warm's.
var pipelines_before: ?u64 = null;

/// `pipelines_before`, once: the model init that follows a weight load takes it.
pub fn takePipelinesBefore() ?u64 {
    defer pipelines_before = null;
    return pipelines_before;
}

/// Whether the pack ships the DSpark stages its drafter runs on.
pub fn shipsHead(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    return source.shipsDspark(io, allocator, model_dir);
}

pub const refusals = struct {
    pub const SsdBudgetBelowVision = "--ssd-budget-gb cannot hold the vision tower (--vision, or the default on DeepSeek-V4.1) beside the minimum expert cache. Raise the budget or pass --no-vision.";
};
