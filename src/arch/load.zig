//! What an arch loads its own way, beside the generic loaders: its weights from the checkpoint's own layout, what that
//! load keeps resident, a drafting head allocated beside the trunk, and the load-time work that follows. An arch
//! declares any of these in a `load` struct in its `arch.zig`; each is asked by the config's arch.
const std = @import("std");
const model = @import("../model.zig");
const trunk = @import("../trunk.zig");
const registry = @import("registry.zig");
const arch_config = @import("config.zig");
const ModelConfig = model.ModelConfig;
const Weights = model.Weights;
const Trunk = trunk.Trunk;

fn has(comptime t: registry.Tag, comptime name: []const u8) bool {
    const M = registry.Mod(t);
    return @hasDecl(M, "load") and @hasDecl(M.load, name);
}

fn L(comptime t: registry.Tag) type {
    return registry.Mod(t).load;
}

/// The arch's own load of `config`'s weights; null takes the generic loaders.
pub fn weights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool) !?Weights {
    const tag = arch_config.tagOf(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "weights")) try L(t).weights(io, allocator, model_dir, config, load_vision) else null,
    };
}

/// A checkpoint whose routed experts are individual native tensors, read by the arch that declares that layout; null
/// when no built arch does.
pub fn individualExpertWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, vision: bool) !?Weights {
    inline for (comptime std.enums.values(registry.Tag)) |t| {
        if (comptime has(t, "individualExpertWeights")) return try L(t).individualExpertWeights(io, allocator, model_dir, vision);
    }
    return null;
}

/// The resident bytes the arch's own load holds: weights, its head when MTP is on, and what the load builds beside them;
/// null takes the indexed scan.
pub fn residentLoadBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool, mtp_on: bool) !?u64 {
    const tag = arch_config.tagOf(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "residentLoadBytes")) try L(t).residentLoadBytes(io, allocator, model_dir, config, load_vision, mtp_on) else null,
    };
}

/// Whether the pack in `model_dir` ships the head the arch drafts with; null leaves the question to the shared readers.
pub fn shipsHead(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8) ?bool {
    const tag = arch_config.tagOf(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "shipsHead")) L(t).shipsHead(io, allocator, model_dir) else null,
    };
}

/// A drafting head the arch allocates beside the trunk, handed to the drafter as `aux`; null when it has none or it did
/// not load.
pub fn loadHead(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, xfm: *Trunk, mtp_enabled: bool) ?*anyopaque {
    const tag = arch_config.tagOf(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "loadHead")) L(t).loadHead(io, allocator, model_dir, config, xfm, mtp_enabled) else null,
    };
}

/// Frees a head `loadHead` returned.
pub fn freeHead(config: *const ModelConfig, allocator: std.mem.Allocator, head: *anyopaque) void {
    const tag = arch_config.tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "freeHead")) L(t).freeHead(allocator, head),
    }
}

/// The deepest MTP round a head `loadHead` returned serves.
pub fn headDepthCap(config: *const ModelConfig, head: *anyopaque) ?u32 {
    const tag = arch_config.tagOf(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(t, "headDepthCap")) L(t).headDepthCap(config, head) else null,
    };
}

/// What the arch builds once its trunk and head are loaded, before the first request.
pub fn prepare(config: *const ModelConfig, xfm: *Trunk) void {
    const tag = arch_config.tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "prepare")) L(t).prepare(config, xfm),
    }
}

/// The arch's speculative warmup after the trunk's own, with the head `loadHead` returned.
pub fn warmSpec(io: std.Io, config: *const ModelConfig, xfm: *Trunk, head: ?*anyopaque, kv_config: trunk.KVQuantConfig) void {
    const tag = arch_config.tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "warmSpec")) L(t).warmSpec(io, config, xfm, head, kv_config),
    }
}
