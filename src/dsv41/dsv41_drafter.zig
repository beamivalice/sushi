//! DeepSeek-V4.1's drafter: DSpark rounds over the model's own state (`dsv41_dspark`). There is no head to step, so
//! `arch/drafter.zig` answers every head operation with its default and the rounds run from `nextDspark`.
const trunk_mod = @import("../trunk.zig");
const spec_types = @import("../spec_types.zig");
const Trunk = trunk_mod.Trunk;

pub const kinds = [_]spec_types.Kind{.dspark};
pub const head_kind: spec_types.Kind = .dspark;

pub const rounds = @import("dsv41_dspark.zig");
pub const nextDspark = rounds.nextDspark;
pub const dspark_rounds = rounds;

/// The DSpark stages and their windows live on the model's own state, so a request that drafts with them holds the
/// model exclusively (`ownsModuleDecodeState`).
pub const Head = *Trunk;

pub fn headOf(xfm: *Trunk, _: bool, _: ?*anyopaque) ?Head {
    return if (xfm.arch_state.dsv41.model != null and xfm.arch_state.dsv41.model.?.dspark != null) xfm else null;
}
