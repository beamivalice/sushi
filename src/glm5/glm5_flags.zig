//! GLM's launch face (the `flags` slot of `arch.zig`): its words for `--think`, and where its cache differs from the shared rules.
const std = @import("std");
const kv_quant = @import("../kv_quant.zig");

pub const think_rank = 0;
pub const think_help = "GLM low|high|max (default high);\n";
pub const think_refusal = "GLM low|high|max";

pub const mtp_off_line = "MTP: off (GLM native serving has no integrated MTP head)";

/// The latent's bits for this scheme (`glm5_latent.zig`); null when the latent cache cannot store it (kv4).
pub fn latentBits(kv: kv_quant.KVQuantConfig) ?u8 {
    return switch (kv.scheme) {
        .off => 0,
        .affine => if ((kv.bits == 8 or kv.bits == 6) and kv.group_size == 64) kv.bits else null,
    };
}

/// The `kv_quant` spellings the settings page may keep for this arch: the ones `latentBits` stores.
pub const kv_quant_choices = [_][]const u8{ "off", "0", "16", "6", "8" };

/// The latent cache stores 8, 6 or off; a request asking for another width is refused.
pub fn kvQuantRequestRefusal(kv: kv_quant.KVQuantConfig) ?[]const u8 {
    if (latentBits(kv) != null) return null;
    return "GLM native MLA cache supports kv_quant 8, 6 or off.";
}

test "GLM latent stores BF16, kv8 or kv6 and names kv4 unsupported" {
    try std.testing.expectEqual(@as(?u8, 0), latentBits(kv_quant.KVQuantConfig.dense));
    try std.testing.expectEqual(@as(?u8, 8), latentBits(kv_quant.KVQuantConfig.affine(8)));
    try std.testing.expectEqual(@as(?u8, null), latentBits(kv_quant.KVQuantConfig.affine(4)));
    try std.testing.expectEqual(@as(?u8, 6), latentBits(kv_quant.KVQuantConfig.affine(6)));
    try std.testing.expectEqual(@as(?u8, 8), latentBits(kv_quant.KVQuantConfig.fromJsonValue(.{ .integer = 8 }).?));
}

test "kv_quant_choices are exactly the spellings the latent stores" {
    for ([_][]const u8{ "off", "0", "16", "4", "6", "8" }) |spelling| {
        const stored = latentBits(kv_quant.KVQuantConfig.fromJsonValue(.{ .string = spelling }).?) != null;
        var listed = false;
        for (kv_quant_choices) |c| listed = listed or std.mem.eql(u8, c, spelling);
        try std.testing.expectEqual(stored, listed);
    }
}
