//! Flash-Next's launch face (the `flags` slot of `arch.zig`): the flags only it takes, with their help and boot lines.
const model = @import("../model.zig");
const flags = @import("../arch/flags.zig");
const trunk = @import("../trunk.zig");
const qwen4_exp = @import("qwen4_exp.zig");

pub const options = [_]flags.Option{
    .{ .name = "--mtp-head-kv-quant", .takes_value = false, .apply = setMtpHeadKvQuant, .boot_line = mtpHeadKvLine },
    .{ .name = "--qwen-gdn", .takes_value = true, .apply = setQwenGdn, .expected = "expected 32 or 16", .boot_line = qwenGdnLine },
};

pub const help = [_]flags.HelpChunk{
    .{ .before = "  --mtp-min-depth", .text =
        \\  --mtp-head-kv-quant Quantize the qwen4 MTP head's own KV with
        \\                        --kv-quant (default OFF: the head keeps
        \\                        dense bf16 KV). No effect on MiMo, whose
        \\                        heads keep a dense sliding window.
        \\
    },
    .{ .before = "  --kv-quant", .text =
        \\  --qwen-gdn <32|16>  Qwen only: bits of the GatedDeltaNet recurrent state
        \\                      between tokens (default 32 = f32; the decay gate
        \\                      is f32 either way). 16 (bf16) halves the state in
        \\                      every cache checkpoint (about 54 MB each on
        \\                      Flash-Next) to save memory on smaller machines.
        \\                      No effect on GLM or MiMo, which have no GDN arm.
        \\
    },
};

pub const think_rank = 1;
pub const think_help = "Qwen off|low|medium|xhigh; ";
pub const think_refusal = "Qwen off|low|medium|xhigh";

pub const warm = struct {
    pub fn bytes(comptime order: @import("std").builtin.AtomicOrder) u64 {
        return qwen4_exp.live_warm_bytes.load(order);
    }
    pub fn total(comptime order: @import("std").builtin.AtomicOrder) u64 {
        return qwen4_exp.live_warm_total.load(order);
    }
};

fn setMtpHeadKvQuant(_: []const u8) bool {
    trunk.Trunk.mtp_head_kv_quant_flag = true;
    return true;
}

fn setQwenGdn(value: []const u8) bool {
    model.qwen_gdn_f32 = model.parseQwenGdn(value) orelse return false;
    return true;
}

/// What `--mtp-head-kv-quant` does on this arch, for the boot log; null when unset.
fn mtpHeadKvLine(config: *const model.ModelConfig, use: flags.LaunchFlagUse) ?[]const u8 {
    if (!use.mtp_head_kv_quant) return null;
    if (config.isMimo()) return "MTP head KV: --mtp-head-kv-quant has no effect on mimo_v2 (its heads keep a dense sliding window)";
    return "MTP head KV: following --kv-quant (--mtp-head-kv-quant)";
}

/// What `--qwen-gdn 16` does on this arch, for the boot log; null at the f32 default.
/// GLM and MiMo have no GatedDeltaNet arm, so the flag has no effect there.
fn qwenGdnLine(config: *const model.ModelConfig, use: flags.LaunchFlagUse) ?[]const u8 {
    if (!use.qwen_gdn_16) return null;
    if (config.isGlm5()) return "GDN state: --qwen-gdn 16 has no effect on GLM (its KDA state is f32 already)";
    if (config.isMimo()) return "GDN state: --qwen-gdn 16 has no effect on mimo_v2 (no GatedDeltaNet layers)";
    return "GDN state: bf16 between tokens (--qwen-gdn 16; f32 is the default; the decay gate is f32 either way)";
}
