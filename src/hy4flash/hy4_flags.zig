//! Hy4-Flash's launch face (the `flags` slot of `arch.zig`): its words for `--think`, and the KV widths its latent cache
//! stores.
const std = @import("std");
const kv_quant = @import("../kv_quant.zig");

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
