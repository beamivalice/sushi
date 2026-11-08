//! GLM's part of a model load (`arch/load.zig`): the resident bill, the DFlash2 assistant's bytes, runtime cache and
//! binding, and the load lines.
const std = @import("std");
const mlx = @import("../mlx.zig");
const bills = @import("../bills.zig");
const log = @import("../log.zig");
const status = @import("../status.zig");
const scheduler_mod = @import("../scheduler.zig");
const trunk_mod = @import("../trunk.zig");
const dflash_mod = @import("../dflash.zig");
const model_mod = @import("../model.zig");
const model_load = @import("../model_load.zig");
const arch_load = @import("../arch/load.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const diagnostic = @import("glm5_diagnostic.zig");
const dflash_cache = @import("glm5_dflash_cache.zig");
const dflash_native = @import("glm5_dflash.zig");
const glm5_model = @import("glm5_model.zig");
const glm5_prefix = @import("glm5_prefix.zig");
const glm5_state = @import("glm5_state.zig");
const rows_tick = @import("glm5_rows_tick.zig");

const ModelConfig = model_mod.ModelConfig;
const Trunk = trunk_mod.Trunk;

pub fn weights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool) !model_mod.Weights {
    return diagnostic.loadWeightsBoundedWithVision(io, allocator, model_dir, mlx.gpuStream(), config.expert_streaming, std.math.maxInt(u64), load_vision and config.glm5_vision);
}

pub fn residentLoadBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool, mtp_on: bool) !u64 {
    _ = mtp_on;
    return diagnostic.residentBytesWithVision(io, allocator, model_dir, config.num_hidden_layers, load_vision and config.glm5_vision);
}

/// A streamed GLM forward keeps no speculative state: a found assistant stays unloaded,
/// an asked-for one is refused by name.
pub fn drafterOff(config: *const ModelConfig, drafter_dir: []const u8) !bool {
    if (!config.expert_streaming) return false;
    if (drafter_dir.len > 0) return error.GlmStreamingSpecUnsupported;
    log.info("[glm-dflash] off: unsupported under expert streaming\n", .{});
    return true;
}

fn dflashLoadBytes(io: std.Io, allocator: std.mem.Allocator, config: *ModelConfig, model_dir: []const u8, directory: []const u8) !u64 {
    if (directory.len == 0) return 0;
    model_load.stampDflashRequestTerms(io, allocator, config, directory) catch |err| return switch (err) {
        error.DflashAssistantRequired => error.GlmDflashAssistantRequired,
        error.DflashTargetMismatch => error.GlmDraftTargetMismatch,
        else => err,
    };
    return if (dflash_cache.isShippedSource(directory, model_dir))
        try dflash_cache.plannedResidentBytes(io, allocator, directory)
    else
        try diagnostic.assistantResidentBytes(io, allocator, directory);
}

pub fn stageDrafter(io: std.Io, allocator: std.mem.Allocator, config: *ModelConfig, model_dir: []const u8, drafter_dir: []const u8, no_drafter: bool) !u64 {
    const assistant_bytes = try dflashLoadBytes(io, allocator, config, model_dir, drafter_dir);
    if (assistant_bytes > 0) log.info("[glm-dflash] resident assistant {d:.3} GiB; window {d} MiB; BF16 context; verification scratch billed per request\n", .{ @as(f64, @floatFromInt(assistant_bytes)) / (1024 * 1024 * 1024), config.dflash_window_bytes >> 20 });
    if (drafter_dir.len == 0 and !no_drafter and !config.expert_streaming)
        log.warn("[glm-dflash] off: no assistant found (expected {s}/{s}, or an intact {s}/{s} cache); re-run `sushi pull` to fetch it\n", .{ model_dir, dflash_mod.SHIPPED_GLM_SUBDIR, model_dir, dflash_mod.DFLASH2_IN_DIR_SUBDIR });
    return assistant_bytes;
}

/// What a resident cold load reserves in the registry: the requirement its load preflight compares with free memory.
pub fn coldLoadBillBytes(io: std.Io, allocator: std.mem.Allocator, config: *ModelConfig, model_dir: []const u8, load_vision: bool, no_drafter: bool, drafter_dir: []const u8) !u64 {
    const drafter = model_load.LoadDrafterDir.resolve(io, allocator, config, no_drafter, drafter_dir, model_dir);
    defer drafter.deinit(allocator);
    const resident = try residentLoadBytes(io, allocator, model_dir, config, load_vision, false);
    return model_load.loadRequirementBytes(config, resident +| try dflashLoadBytes(io, allocator, config, model_dir, drafter.dir), false, 0);
}

/// The shipped assistant becomes the runtime cache built beside the pack; a build that falls back bills the BF16 bytes.
pub fn prepareDrafter(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, drafter_dir: []const u8, resident_bytes: ?u64) !?arch_load.Prepared {
    if (!dflash_cache.isShippedSource(drafter_dir, model_dir)) return null;
    const prepared = try dflash_cache.prepare(io, allocator, drafter_dir, mlx.gpuStream());
    errdefer allocator.free(prepared.path);
    var bytes = resident_bytes;
    if (prepared.fallback) {
        const cached_bytes = try dflash_cache.plannedResidentBytes(io, allocator, drafter_dir);
        const bf16_bytes = try diagnostic.assistantResidentBytes(io, allocator, prepared.path);
        bytes = resident_bytes.? -| cached_bytes +| bf16_bytes;
        if (!scheduler_mod.skipMemPreflight()) {
            const available = bills.effectiveAvailableBytes(status.getAvailableMemBytes(), status.getProcAvailableMemBytes(), mlx.maxRecommendedWorkingSet());
            const needed = model_load.loadRequirementBytes(config, bytes.?, false, 0);
            log.info("[preflight] BF16 DFlash2 fallback: weights ~{d:.2} GiB, needs ~{d:.2} GiB, available {d:.2} GiB\n", .{ @as(f64, @floatFromInt(bytes.?)) / (1 << 30), @as(f64, @floatFromInt(needed)) / (1 << 30), @as(f64, @floatFromInt(available)) / (1 << 30) });
            if (available > 0 and needed > available) return error.InsufficientMemory;
        }
    }
    _ = mlx.mlx_clear_cache();
    return .{ .path = prepared.path, .resident_bytes = bytes };
}

pub fn loadDflash(io: std.Io, allocator: std.mem.Allocator, xfm: *Trunk, dir: []const u8) !dflash_mod.DflashModel {
    return dflash_native.loadAssistantStored(io, allocator, dir, glm5_state.modelOf(xfm).?);
}

pub fn bindDflash(allocator: std.mem.Allocator, xfm: *Trunk, d: *dflash_mod.DflashModel, kv: trunk_mod.KVQuantConfig, tok: *tokenizer_mod.Tokenizer) bool {
    d.sliding_serving = true;
    log.info("[glm-dflash] native DFlash2 loaded; greedy and sampled target decisions use layerwise tree verification\n", .{});
    if (trunk_mod.diagEnvOn("SUSHI_GLM_ROUND_UBENCH")) rows_tick.glmRoundUbench(allocator, glm5_state.modelOf(xfm).?, d, kv, tok);
    return true;
}

pub fn dflashRowBounds(min: ?u32, max: ?u32) !?u32 {
    const bounds = dflash_native.rowBounds(min, max) catch |err| {
        log.err("--drafter-min-depth/--drafter-max-depth: GLM-5.3-Flash verifies {d}..{d} rows per round\n", .{ dflash_native.MIN_ROWS, dflash_native.MAX_ROWS });
        return err;
    };
    return bounds.max;
}

pub fn logDflashReady(block_size: u32) void {
    log.info("[spec] GLM-5.3-Flash decodes with DFlash2 ({d}..{d} rows per round)\n", .{ dflash_mod.drafter_min_depth orelse dflash_native.MIN_ROWS, block_size });
}

pub fn logLoad(config: *const ModelConfig, kv: trunk_mod.KVQuantConfig, mtp_on: bool, params: anytype) void {
    const latent_bits: u64 = if (kv.isQuant()) kv.bits else 16;
    log.info("[glm] native {s} MLA: {d} latent + {d} pooled-index bytes/token; serial decode\n", .{ if (!kv.isQuant()) "BF16" else if (kv.bits == 6) "kv6" else "kv8", bills.kvBytesPerTokenAtBits(config.kvBytesPerToken(), latent_bits), config.qsaHistoryBytesPerToken() });
    log.info("[glm] NAX arms {s}\n", .{if (glm5_model.naxArms()) "on: packed sparse and B1/B3 attention, NAX index scores, KDA cluster, A6 dense-once, MLA head batches" else "off: FP32 composite sparse and B1/B3 attention, scalar index scores, staged KDA cluster, affine QMM trunk"});
    if (mtp_on) log.warn("[glm] MTP head is not integrated with serving; MTP off\n", .{});
    if (params.prefix_cache_capacity > 0 and params.ssm_checkpoint_stride > 0) log.info("[glm] prefix cache: KDA checkpoints every 8192 tokens and at the prompt end (<= {d} held in memory, {d} MiB each)\n", .{ glm5_prefix.checkpointMax(params.ssm_checkpoint_max), config.ssmCheckpointBytes() >> 20 });
}

pub fn warmUbench(allocator: std.mem.Allocator, xfm: *Trunk, kv: trunk_mod.KVQuantConfig, tok: *tokenizer_mod.Tokenizer) void {
    if (trunk_mod.diagEnvOn("SUSHI_GLM_ROWS_UBENCH")) if (glm5_state.modelOf(xfm)) |target| rows_tick.glmRowsUbench(allocator, target, kv, tok);
}
