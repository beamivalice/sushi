//! GLM's launch face (the `flags` slot of `arch.zig`): its words for `--think`, and where its cache differs from the shared rules.
const kv_quant = @import("../kv_quant.zig");

pub const think_rank = 0;
pub const think_help = "GLM low|high|max (default high);\n";
pub const think_refusal = "GLM low|high|max";

pub const mtp_off_line = "MTP: off (GLM native serving has no integrated MTP head)";

/// The latent cache stores 8, 6 or off; a request asking for another width is refused.
pub fn kvQuantRequestRefusal(kv: kv_quant.KVQuantConfig) ?[]const u8 {
    if (kv.glmLatentBits() != null) return null;
    return "GLM native MLA cache supports kv_quant 8, 6 or off.";
}
