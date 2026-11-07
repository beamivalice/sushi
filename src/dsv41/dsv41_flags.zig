//! DeepSeek-V4.1's launch face (the `flags` slot of `arch.zig`): `--drop-expert`, `repack`, the launch flags it
//! cannot act on, and the borrow hooks the server installs.
const std = @import("std");
const model = @import("../model.zig");
const flags = @import("../arch/flags.zig");
const expert_usage = @import("expert_usage.zig");
const repack = @import("repack.zig");
const borrow = @import("dsv41_borrow.zig");
const bills = @import("../bills.zig");

pub const options = [_]flags.Option{
    .{ .name = "--drop-expert", .takes_value = true, .apply = setDropExpert, .expected = "expected a multiple of 8 routed experts per layer (0 = off)" },
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

pub const settings_kv_quant_ignored = "model-settings.json kv_quant is ignored on deepseek_v41 (no KV cache to quantize)";

pub const serve = struct {
    /// The borrow pool sizes itself against the engine's headroom and the MLX pool cap.
    pub fn install(pool_ram_cap: u64) void {
        borrow.headroom_fn = &bills.dsv41Headroom;
        borrow.pool_ram_cap = pool_ram_cap;
    }
    pub fn uninstall() void {
        borrow.headroom_fn = null;
        borrow.pool_ram_cap = 0;
    }
};

fn setDropExpert(value: []const u8) bool {
    model.drop_expert = expert_usage.parseCount(value) orelse return false;
    return true;
}

/// A launch flag with nothing to act on here; null when every flag given applies.
pub fn refusal(use: flags.LaunchFlagUse) ?[]const u8 {
    if (use.kv_quant) return "--kv-quant has no effect on deepseek_v41: its compressed state is stored as the QAT round trips left it, so there is no KV cache to quantize; drop the flag";
    if (use.qwen_gdn_16) return "--qwen-gdn has no effect on deepseek_v41: it has no GatedDeltaNet layers; drop the flag";
    if (use.mtp_head_kv_quant) return "--mtp-head-kv-quant has no effect on deepseek_v41: its DSpark stages keep their own windows; drop the flag";
    if (use.mtp_typical) return "--mtp-typical has no effect on deepseek_v41: DSpark accepts drafts by the exact rule only; drop the flag";
    if (use.mtp_tokenv3) return "--mtp-tokenv3 has no effect on deepseek_v41: DSpark accepts drafts by the exact rule only; drop the flag";
    if (use.mtp_greedy_tail) return "--mtp-greedy-tail has no effect on deepseek_v41: DSpark drafts one block from the trunk's own state, with no MTP tail; drop the flag";
    return null;
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
