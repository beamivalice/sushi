//! GLM's drafter: DFlash2 rounds (`glm5_generate`). Its MTP stays off, so there is no head for `arch/drafter.zig`.
const spec_types = @import("../spec_types.zig");

pub const kinds = [_]spec_types.Kind{.dflash2};
pub const rounds = @import("glm5_generate.zig");
pub const nextDflash = rounds.nextGlmDflash;
pub const inDirCacheValid = @import("glm5_dflash_cache.zig").cacheValid;

/// The native model runs the rounds when it is loaded and the sampling allows speculation.
pub fn nativeEligible(xfm: *const @import("../trunk.zig").Trunk, sampling: anytype, logprobs_n: u32) bool {
    return @import("glm5_state.zig").modelOf(xfm) != null and rounds.glmDflashEligible(sampling, logprobs_n);
}

pub const reserveNative = rounds.reserveGlmDflash;
