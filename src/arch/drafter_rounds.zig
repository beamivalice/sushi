//! The drafter rounds as the generator and the scheduler reach them: the `drafter_rounds` of the loaded model's arch,
//! bound by the model's tag at run time (a build may hold several block-drafter archs), else a stand-in that refuses
//! (an arch whose drafter declares no rounds).
//!
//! A `drafter_rounds` module declares `nextDrafterRound`, `drafterResumeAfterForced`, `drafterEmitPending`,
//! `drafterDiag` (the diagnostic switch the arch reads) and `stageBlock` (drafts one stage block holds, from the loaded
//! model). The shared round body is `drafter_round.Rounds`, over the arch's hooks.
const std = @import("std");
const registry = @import("registry.zig");
const generate = @import("../generate.zig");
const drafter_rows = @import("../drafter_rows.zig");
const trunk = @import("../trunk.zig");
const Generator = generate.Generator;

const Refuse = struct {
    pub fn nextDrafterRound(_: *Generator, _: std.mem.Allocator) !?Generator.DrafterStepResult {
        return error.SpecDecodeUnsupported;
    }
    pub fn drafterResumeAfterForced(_: *Generator, _: std.mem.Allocator) !drafter_rows.Resume {
        return .refused;
    }
    pub fn drafterEmitPending(_: *Generator, _: std.mem.Allocator) !?u32 {
        return error.SpecDecodeUnsupported;
    }
    pub fn drafterDiag() drafter_rows.Diag {
        return .off;
    }
    pub fn stageBlock(_: *trunk.Trunk) u32 {
        return default_block;
    }
};

/// The block a drafter-round policy assumes before a model says its own.
pub const default_block: u32 = 5;

/// The drafter rounds of the arch `t`, or the refusal when its drafter declares none.
fn RoundsOf(comptime t: registry.Tag) type {
    const M = registry.Mod(t);
    if (@hasDecl(M, "drafter") and @hasDecl(M.drafter, "drafter_rounds")) return M.drafter.drafter_rounds;
    return Refuse;
}

pub fn nextDrafterRound(self: *Generator, allocator: std.mem.Allocator) !?Generator.DrafterStepResult {
    switch (self.xfm.arch) {
        inline else => |t| return RoundsOf(t).nextDrafterRound(self, allocator),
    }
}

pub fn drafterResumeAfterForced(self: *Generator, allocator: std.mem.Allocator) !drafter_rows.Resume {
    switch (self.xfm.arch) {
        inline else => |t| return RoundsOf(t).drafterResumeAfterForced(self, allocator),
    }
}

pub fn drafterEmitPending(self: *Generator, allocator: std.mem.Allocator) !?u32 {
    switch (self.xfm.arch) {
        inline else => |t| return RoundsOf(t).drafterEmitPending(self, allocator),
    }
}

/// The diagnostic switch the loaded model's arch reads.
pub fn drafterDiag(xfm: *const trunk.Trunk) drafter_rows.Diag {
    switch (xfm.arch) {
        inline else => |t| return RoundsOf(t).drafterDiag(),
    }
}

pub fn stageBlock(xfm: *trunk.Trunk) u32 {
    switch (xfm.arch) {
        inline else => |t| return RoundsOf(t).stageBlock(xfm),
    }
}
