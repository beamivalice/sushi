//! DeepSeek-V4.1's launch face (the `flags` slot of `arch.zig`): `--drop-expert`, `repack`, the launch flags it
//! cannot act on, and the borrow hooks the server installs.
const std = @import("std");
const model = @import("../model.zig");
const flags = @import("../arch/flags.zig");
const model_settings = @import("../model_settings.zig");
const expert_usage = @import("../expert_usage.zig");
const repack = @import("repack.zig");
const borrow = @import("dsv41_borrow.zig");
const dsv41_bill = @import("dsv41_bill.zig");
const dspark_rows = @import("../dspark_rows.zig");
const source = @import("dsv41_source.zig");
const arch_serve = @import("../arch/serve.zig");
const depth_bounds = @import("../mtp_depth_bounds.zig");
const log = @import("../log.zig");

pub const options = [_]flags.Option{
    .{ .name = "--drop-expert", .takes_value = true, .apply = setDropExpert, .expected = "expected a multiple of 8 routed experts per layer (0 = off)" },
    .{ .name = "--ds41-mlx-engine", .takes_value = false, .apply = setMlxEngine },
};

pub const help = [_]flags.HelpChunk{
    .{ .before = "  update ", .text =
        \\  repack <pack>       Rewrite an STQ pack in place into the
        \\                      pair-major layout. A shard is replaced only after
        \\                      its rewrite reads back equal; a cut-off run
        \\                      resumes when repeated. Stop any server using the
        \\                      pack first. Never downloads. Options:
        \\                      --pack-revision <id>  stamp the pack with a
        \\                        revision (config.json and every shard header)
        \\                      --shards <name-or-glob>  stamp only these shards
        \\                        (repeatable; a pattern matching nothing is
        \\                        refused)
        \\                      --unstamped <shard>  leave this shard without a
        \\                        stamp (repeatable; a symlinked shard needs it)
        \\                      --check  read only: every shard not unstamped
        \\                        carries the config's revision
        \\
    },
    .{ .before = "  --kv-quant", .text =
        \\  --ds41-mlx-engine   DeepSeek-V4.1 only: decode on MLX instead of Sushi Cooler,
        \\                      the engine of our own (the default on an STQ pack
        \\                      with a matrix-unit GPU). An opt-out if Sushi Cooler
        \\                      misbehaves; report it at
        \\                      https://github.com/beamivalice/sushi/issues/new.
        \\                      Outranks the model's ds41_cooler setting.
        \\  --drop-expert <n>   DeepSeek-V4.1 only: never load or pick the n least-used
        \\                      routed experts of every layer, ranked by the pack's
        \\                      expert_usage.json (n a multiple of 8; default 0 =
        \\                      off). A token's pick moves to its next-best expert;
        \\                      the memory bill shrinks by the dropped bytes. Answers
        \\                      change, so score it with `sushi kld compare`.
        \\
    },
};

pub const command = struct {
    pub const name = "repack";
    pub const runs_gpu = false;
    pub fn run(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
        return repack.cmdRepack(allocator, io, args);
    }
};

pub const drop_expert_elsewhere = "--drop-expert has no effect on this model: only deepseek_v41 packs carry an expert usage list; drop the flag";

pub const cooler_setting_elsewhere = "model-settings.json ds41_cooler is ignored: only deepseek_v41 runs on Sushi Cooler";

pub const mlx_engine_elsewhere = "--ds41-mlx-engine has no effect on this model: only deepseek_v41 runs on Sushi Cooler; drop the flag";

/// Why decode cannot run on Sushi Cooler for this pack; null when its experts and trunk are STQ. Any GPU serves them.
pub fn coolerUnavailable(config: *const model.ModelConfig) ?[]const u8 {
    const c = config.dsv41 orelse return null;
    return if (c.expert_dec != null and c.trunk_dec != null) null else model_settings.cooler_why_not_stq;
}

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
        if (config.dsparkOn()) {
            log.info("[spec] DeepSeek-V4.1-Flash decodes with DSpark ({d} stages, block {d}, resident; {s})\n", .{ c.n_mtp, c.dspark.block, ctx.mtp_source });
            if (depth_bounds.active.explicit()) {
                const range = dspark_rows.Range.of(depth_bounds.active, c.dspark.block);
                log.info("[spec] DSpark verifies {d}..{d} rows a round (--mtp-min-depth/--mtp-max-depth)\n", .{ range.min_rows, range.max_rows });
            }
        } else {
            const why: []const u8 = if (c.dspark.block == 0 or !source.shipsDspark(ctx.io, ctx.allocator, ctx.model_dir)) "no DSpark stages in this checkpoint" else "--no-mtp or model-settings mtp:false";
            log.info("[spec] DeepSeek-V4.1-Flash decodes with plain ticks ({s})\n", .{why});
        }
    }
};

fn setMlxEngine(_: []const u8) bool {
    model.cooler_off_flag = true;
    return true;
}

fn setDropExpert(value: []const u8) bool {
    model.drop_expert = expert_usage.parseCount(value) orelse return false;
    return true;
}

/// A launch flag with nothing to act on here; null when every flag given applies.
pub fn refusal(config: *const model.ModelConfig, use: flags.LaunchFlagUse) ?[]const u8 {
    if (use.kv_quant) return "--kv-quant has no effect on deepseek_v41: its compressed state is stored as the QAT round trips left it, so there is no KV cache to quantize; drop the flag";
    if (use.qwen_gdn_16) return "--qwen-gdn has no effect on deepseek_v41: it has no GatedDeltaNet layers; drop the flag";
    if (use.mtp_head_kv_quant) return "--mtp-head-kv-quant has no effect on deepseek_v41: its DSpark stages keep their own windows; drop the flag";
    if (use.mtp_typical) return "--mtp-typical has no effect on deepseek_v41: DSpark accepts drafts by the exact rule only; drop the flag";
    if (use.mtp_tokenv3) return "--mtp-tokenv3 has no effect on deepseek_v41: DSpark accepts drafts by the exact rule only; drop the flag";
    if (use.mtp_greedy_tail) return "--mtp-greedy-tail has no effect on deepseek_v41: DSpark drafts one block from the trunk's own state, with no MTP tail; drop the flag";
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

/// A model-settings.json `mtp_acceptance` or `mtp_greedy_tail` is ignored here with one line naming the keys set.
pub fn settingsMtpIgnoredLine(config: *const model.ModelConfig) ?[]const u8 {
    const acceptance = config.mtp_acceptance_override != null;
    const tail = config.mtp_greedy_tail_override != null;
    if (acceptance and tail) return "model-settings.json mtp_acceptance and mtp_greedy_tail are ignored on deepseek_v41 (DSpark accepts drafts by the exact rule only, with no MTP tail)";
    if (acceptance) return "model-settings.json mtp_acceptance is ignored on deepseek_v41 (DSpark accepts drafts by the exact rule only)";
    if (tail) return "model-settings.json mtp_greedy_tail is ignored on deepseek_v41 (DSpark has no MTP tail)";
    return null;
}
