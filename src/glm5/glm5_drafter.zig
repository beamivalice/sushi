//! GLM's drafter: DFlash2 rounds (`glm5_generate`). Its MTP stays off, so there is no head for `arch/drafter.zig`.
const spec_types = @import("../spec_types.zig");
const log = @import("../log.zig");

pub const kinds = [_]spec_types.Kind{.dflash2};
pub const rounds = @import("glm5_generate.zig");
pub const nextDflash = rounds.nextGlmDflash;

/// The cumulative round phases of a native DFlash2 run, one `[spec-stats]` line.
pub fn logRoundStats(self: anytype) void {
    const attempts: f64 = @floatFromInt(self.dflash_attempted);
    const r = self.native_round;
    const plain_ticks: f64 = @floatFromInt(@max(self.dflash_plain.ticks, 1));
    log.info("  [spec-stats] glm_round_ms draft={d:.2} verify={d:.2} replay={d:.2} commit={d:.2} lookup={d}/{d} plain={d} serial_ms={d:.2} plain_ms verify={d:.2} replay={d:.2} commit={d:.2}\n", .{
        @as(f64, @floatFromInt(r.draft_ns)) / attempts / 1e6,
        @as(f64, @floatFromInt(r.verify_ns)) / attempts / 1e6,
        @as(f64, @floatFromInt(r.replay_ns)) / attempts / 1e6,
        @as(f64, @floatFromInt(r.commit_ns)) / attempts / 1e6,
        self.native_lookup.rounds,
        self.native_lookup.landed,
        self.dflash_plain.ticks,
        self.native_gate.serial_ms orelse 0.0,
        @as(f64, @floatFromInt(self.dflash_plain.verify_ns)) / plain_ticks / 1e6,
        @as(f64, @floatFromInt(self.dflash_plain.replay_ns)) / plain_ticks / 1e6,
        @as(f64, @floatFromInt(self.dflash_plain.commit_ns)) / plain_ticks / 1e6,
    });
}

pub const inDirCacheValid = @import("glm5_dflash_cache.zig").cacheValid;
pub const shipped_subdir = @import("glm5_dflash_cache.zig").shipped_subdir;

/// The native model runs the rounds when it is loaded and the sampling allows speculation.
pub fn nativeEligible(xfm: *const @import("../trunk.zig").Trunk, sampling: anytype, logprobs_n: u32) bool {
    return @import("glm5_state.zig").modelOf(xfm) != null and rounds.glmDflashEligible(sampling, logprobs_n);
}

pub const reserveNative = rounds.reserveGlmDflash;
