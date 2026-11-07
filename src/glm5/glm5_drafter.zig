//! GLM's drafter: DFlash2 rounds (`glm5_generate`). Its MTP stays off, so there is no head for `arch/drafter.zig`.
const spec_types = @import("../spec_types.zig");

pub const kinds = [_]spec_types.Kind{.dflash2};
pub const rounds = @import("glm5_generate.zig");
pub const nextDflash = rounds.nextGlmDflash;
