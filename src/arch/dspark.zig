//! DSpark's rounds as the generator and the scheduler reach them: the `dspark_rounds` of the first built arch whose
//! drafter declares one, else a stand-in that refuses (no request drafts with DSpark in a build without such an arch).
//!
//! A `dspark_rounds` module declares `nextDspark`, `dsparkResumeAfterForced`, `dsparkEmitPending`, `dsparkDiag`
//! (the diagnostic switch the arch reads) and `stageBlock` (drafts one stage block holds, from the loaded model).
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

// Binds the first built arch that declares DSpark rounds, at comptime: a build has one DSpark arch.
const Impl = blk: {
    for (std.meta.tags(registry.Tag)) |t| {
        if (@hasDecl(registry.Mod(t).drafter, "dspark_rounds")) break :blk registry.Mod(t).drafter.dspark_rounds;
    }
    break :blk Refuse;
};

pub const nextDspark = Impl.nextDspark;
pub const dsparkResumeAfterForced = Impl.dsparkResumeAfterForced;
pub const dsparkEmitPending = Impl.dsparkEmitPending;
pub const dsparkDiag = Impl.dsparkDiag;
pub const stageBlock = Impl.stageBlock;
