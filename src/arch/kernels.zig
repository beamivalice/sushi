//! The kernel arms of an arch that shared code sizes by: admission's batched KV, a slot's bill, and the env reads
//! resolved at startup. An arch declares any of these in a `kernels` struct in its `arch.zig`; each is asked by the
//! config's arch, and one it omits gives the dense answer.
const std = @import("std");
const model = @import("../model.zig");
const registry = @import("registry.zig");
const arch_config = @import("config.zig");
const ModelConfig = model.ModelConfig;

fn has(comptime t: registry.Tag, comptime name: []const u8) bool {
    const M = registry.Mod(t);
    return @hasDecl(M, "kernels") and @hasDecl(M.kernels, name);
}

fn K(comptime t: registry.Tag) type {
    return registry.Mod(t).kernels;
}

/// Whether a batched step of `seq_len` rows gathers only the selected KV blocks.
pub fn batchedGatherOn(config: *const ModelConfig, seq_len: c_int) bool {
    const tag = arch_config.tagOf(config) orelse return false;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "batchedGatherOn")) K(t).batchedGatherOn(seq_len) else false,
    };
}

/// The KV length a batched step of `seq_len` rows needs before it gathers.
pub fn batchedGatherFloor(config: *const ModelConfig, seq_len: c_int, quantized: bool) c_int {
    const tag = arch_config.tagOf(config) orelse return 0;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "batchedGatherFloor")) K(t).batchedGatherFloor(seq_len, quantized) else 0,
    };
}

/// Whether the indexer's block scores fuse into the selection, so a slot holds no score bank.
pub fn scoreFusedFor(config: *const ModelConfig, batch: c_int, n_idx: c_int, idx_hd: c_int) bool {
    const tag = arch_config.tagOf(config) orelse return false;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "scoreFusedFor")) K(t).scoreFusedFor(batch, n_idx, idx_hd) else false,
    };
}

/// The MoE blocks a diagnostic stand-in replaces with a free one, process-wide.
pub const MoeStandin = struct { moe_shared: bool = false, moe_router: bool = false, moe_gateup: bool = false, moe_down: bool = false };

/// The stand-ins of the built arch that reads them; none where no built arch does.
pub inline fn moeStandin() MoeStandin {
    inline for (comptime std.enums.values(registry.Tag)) |t| {
        if (comptime has(t, "moeStandin")) return K(t).moeStandin();
    }
    return .{};
}

/// The shared-expert fold ran for the first time: the arch that folds says so in the log.
pub fn sharedFoldEngaged() void {
    inline for (comptime std.enums.values(registry.Tag)) |t| {
        if (comptime has(t, "sharedFoldEngaged")) K(t).sharedFoldEngaged();
    }
}

/// A diagnostic env switch of a built arch is armed: the round-cost table is not persisted from such a run.
pub fn diagArmed() bool {
    inline for (comptime std.enums.values(registry.Tag)) |t| {
        if (comptime has(t, "diagArmed")) if (K(t).diagArmed()) return true;
    }
    return false;
}

/// Every built arch's lazily cached env reads, resolved on the main thread before other threads exist.
pub fn warmEnvCaches() void {
    inline for (comptime std.enums.values(registry.Tag)) |t| {
        if (comptime has(t, "warmEnvCaches")) K(t).warmEnvCaches();
    }
}
