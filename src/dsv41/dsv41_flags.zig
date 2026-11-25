//! DeepSeek-V4.1's launch face (the `flags` slot of `arch.zig`): the launch flags it cannot act on and the borrow
//! hooks the server installs.
const std = @import("std");
const model = @import("../model.zig");
const flags = @import("../arch/flags.zig");
const borrow = @import("dsv41_borrow.zig");
const dsv41_bill = @import("dsv41_bill.zig");
const drafter_rows = @import("../drafter_rows.zig");
const source = @import("dsv41_source.zig");
const arch_serve = @import("../arch/serve.zig");
const depth_bounds = @import("../mtp_depth_bounds.zig");
const log = @import("../log.zig");

pub const help = [_]flags.HelpChunk{
    .{ .before = "  --kv-quant", .text =
        \\  --ds41-mlx-engine   DeepSeek-V4.1 only: decode on MLX instead of Sushi Cooler,
        \\                      the engine of our own (the default on an STQ pack
        \\                      with a matrix-unit GPU). An opt-out if Sushi Cooler
        \\                      misbehaves; report it at
        \\                      https://github.com/beamivalice/sushi/issues/new.
        \\                      Outranks the model's ds41_cooler setting.
        \\
    },
};

pub const settings_kv_quant_ignored = "model-settings.json kv_quant is ignored on deepseek_v41 (no KV cache to quantize)";

pub const serve = struct {
    /// The borrow pool sizes itself against the engine's headroom and the MLX pool cap.
    pub fn install(pool_ram_cap: u64) void {
        borrow.headroom_fn = &dsv41_bill.dsv41Headroom;
        borrow.pool_ram_cap = pool_ram_cap;
    }
    pub fn uninstall() void {
        borrow.headroom_fn = null;
        borrow.pool_ram_cap = 0;
    }
    pub fn poolCapped(prev: u64) void {
        borrow.poolCapped(prev);
    }
    pub fn unloadModel(config: *const model.ModelConfig) void {
        borrow.unloadModelMlx(config);
    }
    pub fn enterVision(bill: u64, config: *const model.ModelConfig) void {
        borrow.enterVisionMlx(bill, config);
    }
    pub fn leaveVision(config: *const model.ModelConfig) void {
        borrow.leaveVisionMlx(config);
    }
    pub fn plannedBorrowBytes(config: *const model.ModelConfig, exl3: bool, cache_bytes: u64) u64 {
        const c = config.dsv41 orelse return 0;
        return borrow.plannedBytes(&c, exl3, cache_bytes);
    }
    pub fn loadLines(config: *const model.ModelConfig, ctx: arch_serve.LoadLog) void {
        const c = config.dsv41 orelse return;
        log.info("[dsv41] {d} bytes/token of compressed state\n", .{config.kvBytesPerToken()});
        if (config.drafterOn()) {
            log.info("[spec] DeepSeek-V4.1-Flash decodes with DSpark ({d} stages, block {d}, resident; {s}{s})\n", .{ c.n_mtp, c.dspark.block, ctx.mtp_source, if (c.widthExactLayout()) "" else "; fixed width: this layout is not width-exact" });
            if (depth_bounds.active.explicit()) {
                const range = drafter_rows.Range.of(depth_bounds.active, c.dspark.block);
                log.info("[spec] DSpark verifies {d}..{d} rows a round (--mtp-min-depth/--mtp-max-depth)\n", .{ range.min_rows, range.max_rows });
            }
        } else {
            const why: []const u8 = if (c.dspark.block == 0 or !source.shipsDspark(ctx.io, ctx.allocator, ctx.model_dir)) "no DSpark stages in this checkpoint" else "--no-mtp or model-settings mtp:false";
            log.info("[spec] DeepSeek-V4.1-Flash decodes with plain ticks ({s})\n", .{why});
        }
    }
};

/// A launch flag with nothing to act on here; null when every flag given applies.
pub fn refusal(config: *const model.ModelConfig, use: flags.LaunchFlagUse) ?[]const u8 {
    if (use.kv_quant) return "--kv-quant has no effect on deepseek_v41: its compressed state is stored as the QAT round trips left it, so there is no KV cache to quantize; drop the flag";
    if (use.qwen_gdn_16) return "--qwen-gdn has no effect on deepseek_v41: it has no GatedDeltaNet layers; drop the flag";
    if (use.mtp_head_kv_quant) return "--mtp-head-kv-quant has no effect on deepseek_v41: its DSpark stages keep their own windows; drop the flag";
    const block = if (config.dsv41) |c| c.dspark.block else 0;
    if (block > 0) {
        if (use.mtp_max_depth > block) return depthAboveBlock("--mtp-max-depth", use.mtp_max_depth, block);
        if (use.mtp_min_depth > block) return depthAboveBlock("--mtp-min-depth", use.mtp_min_depth, block);
    }
    return null;
}

var depth_refusal_buf: [160]u8 = undefined;

fn depthAboveBlock(flag: []const u8, given: u32, block: u32) []const u8 {
    return std.fmt.bufPrint(&depth_refusal_buf, "{s} {d} is above deepseek_v41's DSpark block of {d} drafts; use {d} or lower", .{ flag, given, block, block }) catch "an MTP depth flag is above deepseek_v41's DSpark block";
}
