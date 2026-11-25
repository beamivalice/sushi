//! Hy4-Flash's launch face (the `flags` slot of `arch.zig`): the chunk bounds of its static bill, its words for
//! `--think`, and the KV widths its latent cache stores.
const std = @import("std");
const kv_quant = @import("../kv_quant.zig");
const flags = @import("../arch/flags.zig");
const settings_resolve = @import("../settings_resolve.zig");
const widest_chunk = @import("hy4_config.zig").widest_chunk;

/// The rows the load's prefill chunk is chosen within (D4); a load of another arch refuses them by name
/// (`flags.archFlagRefusal`).
pub const options = [_]flags.Option{
    .{ .name = "--prefill-chunk-max", .takes_value = true, .apply = setChunkMax, .expected = chunk_expected },
    .{ .name = "--prefill-chunk-min", .takes_value = true, .apply = setChunkMin, .expected = chunk_expected },
};

const chunk_expected = "expected " ++ settings_resolve.static_chunk_rungs_text;

fn chunkRows(value: []const u8) ?u32 {
    const n = std.fmt.parseInt(u32, value, 10) catch return null;
    return if (settings_resolve.isStaticChunkRung(n) and n <= widest_chunk) n else null;
}

fn setChunkMax(value: []const u8) bool {
    settings_resolve.launch.prefill_chunk_max = chunkRows(value) orelse return false;
    return true;
}

fn setChunkMin(value: []const u8) bool {
    settings_resolve.launch.prefill_chunk_min = chunkRows(value) orelse return false;
    return true;
}

pub const help = [_]flags.HelpChunk{
    .{ .before = "  --max-concurrent", .text =
        \\  --prefill-chunk-max <n>
        \\                      Hy4-Flash only: the widest prefill chunk its load
        \\                        may fix: 512, 1024, 2048, 4096 or 8192 rows
        \\                        (default 4096, or a wider --prefill-chunk-min);
        \\                        any other is refused. The
        \\                        load takes the widest chunk within the two
        \\                        bounds whose arena fits beside the context, and
        \\                        every prefill runs at it. Outranks the model's
        \\                        prefill_chunk_max setting. Other models refuse it.
        \\  --prefill-chunk-min <n>
        \\                      Hy4-Flash only: the narrowest chunk its load may
        \\                        fix, one of the same widths (default 512).
        \\                        Equal to the max, it forces the chunk; a load that
        \\                        cannot hold it refuses with the numbers. Outranks
        \\                        prefill_chunk_min. Other models refuse it.
        \\
    },
};

pub const think_rank = 1;
pub const think_help = "Hy4 off|high|max;\n";
pub const think_refusal = "Hy4 off|high|max";

/// Whether the latent cache stores `kv`: the engine's affine kv8 (the default) or BF16.
pub fn storesKv(kv: kv_quant.KVQuantConfig) bool {
    return kv.scheme == .off or (kv.scheme == .affine and kv.bits == 8 and kv.group_size == 64);
}

/// The `kv_quant` spellings the settings page may keep for this arch: the ones `storesKv` takes.
pub const kv_quant_choices = [_][]const u8{ "off", "0", "16", "8" };

/// A saved width the cache cannot store is ignored with this line; a launch flag naming one is refused with it.
pub fn kvQuantRequestRefusal(kv: kv_quant.KVQuantConfig) ?[]const u8 {
    if (storesKv(kv)) return null;
    return "Hy4-Flash's latent cache supports kv_quant 8 or off.";
}

test "hy4 flags: kv_quant_choices are exactly the spellings the latent cache stores" {
    for ([_][]const u8{ "off", "0", "16", "4", "6", "8" }) |spelling| {
        const kv = kv_quant.KVQuantConfig.fromJsonValue(.{ .string = spelling }).?;
        var listed = false;
        for (kv_quant_choices) |c| listed = listed or std.mem.eql(u8, c, spelling);
        try std.testing.expectEqual(storesKv(kv), listed);
        try std.testing.expectEqual(storesKv(kv), kvQuantRequestRefusal(kv) == null);
    }
}
