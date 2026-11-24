//! DSpark's rounds as the generator and the scheduler reach them: the `dspark_rounds` of the loaded model's arch, bound
//! by the model's tag at run time (a build may hold several DSpark archs), else a stand-in that refuses (an arch whose
//! drafter declares no rounds).
//!
//! A `dspark_rounds` module declares `nextDspark`, `dsparkResumeAfterForced`, `dsparkEmitPending`, `dsparkDiag`
//! (the diagnostic switch the arch reads) and `stageBlock` (drafts one stage block holds, from the loaded model). The
//! shared round body is `dspark_round.Rounds`, over the arch's hooks.
const std = @import("std");
const registry = @import("registry.zig");
const generate = @import("../generate.zig");
const dspark_rows = @import("../dspark_rows.zig");
const trunk = @import("../trunk.zig");
const Generator = generate.Generator;

const Refuse = struct {
    pub fn nextDspark(_: *Generator, _: std.mem.Allocator) !?Generator.DrafterStepResult {
        return error.SpecDecodeUnsupported;
    }
    pub fn dsparkResumeAfterForced(_: *Generator, _: std.mem.Allocator) !dspark_rows.Resume {
        return .refused;
    }
    pub fn dsparkEmitPending(_: *Generator, _: std.mem.Allocator) !?u32 {
        return error.SpecDecodeUnsupported;
    }
    pub fn dsparkDiag() dspark_rows.Diag {
        return .off;
    }
    pub fn stageBlock(_: *trunk.Trunk) u32 {
        return default_block;
    }
};

/// The block a DSpark policy assumes before a model says its own.
pub const default_block: u32 = 5;

/// The DSpark rounds of the arch `t`, or the refusal when its drafter declares none.
fn RoundsOf(comptime t: registry.Tag) type {
    const M = registry.Mod(t);
    if (@hasDecl(M, "drafter") and @hasDecl(M.drafter, "dspark_rounds")) return M.drafter.dspark_rounds;
    return Refuse;
}

pub fn nextDspark(self: *Generator, allocator: std.mem.Allocator) !?Generator.DrafterStepResult {
    switch (self.xfm.arch) {
        inline else => |t| return RoundsOf(t).nextDspark(self, allocator),
    }
}

pub fn dsparkResumeAfterForced(self: *Generator, allocator: std.mem.Allocator) !dspark_rows.Resume {
    switch (self.xfm.arch) {
        inline else => |t| return RoundsOf(t).dsparkResumeAfterForced(self, allocator),
    }
}

pub fn dsparkEmitPending(self: *Generator, allocator: std.mem.Allocator) !?u32 {
    switch (self.xfm.arch) {
        inline else => |t| return RoundsOf(t).dsparkEmitPending(self, allocator),
    }
}

/// The diagnostic switch the loaded model's arch reads.
pub fn dsparkDiag(xfm: *const trunk.Trunk) dspark_rows.Diag {
    switch (xfm.arch) {
        inline else => |t| return RoundsOf(t).dsparkDiag(),
    }
}

pub fn stageBlock(xfm: *trunk.Trunk) u32 {
    switch (xfm.arch) {
        inline else => |t| return RoundsOf(t).stageBlock(xfm),
    }
}
