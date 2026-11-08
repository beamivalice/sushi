//! What an arch loads its own way, beside the generic loaders: its weights from the checkpoint's own layout, what that
//! load keeps resident, a drafting head allocated beside the trunk, its drafter's staging and binding, and the load-time
//! work and lines that follow. An arch declares any of these in a `load` struct or module in its `arch.zig`; each is asked
//! by the config's arch, and an arch that omits one takes the default stated on its wrapper.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const trunk = @import("../trunk.zig");
const dflash_mod = @import("../dflash.zig");
const tokenizer_mod = @import("../tokenizer.zig");
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

/// A drafter the load binds in place of the one it found: `path` is owned, `resident_bytes` the bill with it.
pub const Prepared = struct { path: []u8, resident_bytes: ?u64 };

pub fn drafterOff(config: *const ModelConfig, drafter_dir: []const u8) !bool {
    const tag = arch_config.tagOf(config) orelse return false;
    switch (tag) {
        inline else => |t| return if (comptime has(t, "drafterOff")) try L(t).drafterOff(config, drafter_dir) else false,
    }
}

/// The drafter's resident bytes once its request terms are stamped; null when the arch stages none.
pub fn stageDrafter(io: std.Io, allocator: std.mem.Allocator, config: *ModelConfig, model_dir: []const u8, drafter_dir: []const u8, no_drafter: bool) !?u64 {
    const tag = arch_config.tagOf(config) orelse return null;
    switch (tag) {
        inline else => |t| return if (comptime has(t, "stageDrafter")) try L(t).stageDrafter(io, allocator, config, model_dir, drafter_dir, no_drafter) else null,
    }
}

pub fn prepareDrafter(io: std.Io, allocator: std.mem.Allocator, config: *const ModelConfig, model_dir: []const u8, drafter_dir: []const u8, resident_bytes: ?u64) !?Prepared {
    const tag = arch_config.tagOf(config) orelse return null;
    switch (tag) {
        inline else => |t| return if (comptime has(t, "prepareDrafter")) try L(t).prepareDrafter(io, allocator, config, model_dir, drafter_dir, resident_bytes) else null,
    }
}

/// The drafter read from `dir`: the arch's own form when it has one, else the generic block drafter.
pub fn loadDflash(io: std.Io, allocator: std.mem.Allocator, xfm: *Trunk, dir: []const u8) !dflash_mod.DflashModel {
    switch (xfm.arch) {
        inline else => |t| {
            if (comptime has(t, "loadDflash")) return L(t).loadDflash(io, allocator, xfm, dir);
        },
    }
    return dflash_mod.loadDflash(io, allocator, mlx.gpuStream(), dir);
}

pub fn bindDflash(allocator: std.mem.Allocator, xfm: *Trunk, d: *dflash_mod.DflashModel, kv: trunk.KVQuantConfig, tok: *tokenizer_mod.Tokenizer) bool {
    switch (xfm.arch) {
        inline else => |t| return if (comptime has(t, "bindDflash")) L(t).bindDflash(allocator, xfm, d, kv, tok) else false,
    }
}

pub fn dflashRowBounds(config: *const ModelConfig, min: ?u32, max: ?u32) !?u32 {
    const tag = arch_config.tagOf(config) orelse return null;
    switch (tag) {
        inline else => |t| return if (comptime has(t, "dflashRowBounds")) try L(t).dflashRowBounds(min, max) else null,
    }
}

pub fn logLoad(config: *const ModelConfig, kv: trunk.KVQuantConfig, mtp_on: bool, params: anytype) void {
    const tag = arch_config.tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "logLoad")) L(t).logLoad(config, kv, mtp_on, params),
    }
}

pub fn logDflashReady(config: *const ModelConfig, block_size: u32) void {
    const tag = arch_config.tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (comptime has(t, "logDflashReady")) L(t).logDflashReady(block_size),
    }
}

pub fn warmUbench(allocator: std.mem.Allocator, xfm: *Trunk, kv: trunk.KVQuantConfig, tok: *tokenizer_mod.Tokenizer) void {
    switch (xfm.arch) {
        inline else => |t| if (comptime has(t, "warmUbench")) L(t).warmUbench(allocator, xfm, kv, tok),
    }
}

test "an arch that declares no load module takes every default" {
    var config = model.ModelConfig{ .model_type = "not_a_served_arch" };
    try std.testing.expect((try weights(std.testing.io, std.testing.allocator, "", &config, false)) == null);
    try std.testing.expect((try residentLoadBytes(std.testing.io, std.testing.allocator, "", &config, false, false)) == null);
    try std.testing.expect(!try drafterOff(&config, ""));
    try std.testing.expect((try stageDrafter(std.testing.io, std.testing.allocator, &config, "", "", false)) == null);
    try std.testing.expect((try prepareDrafter(std.testing.io, std.testing.allocator, &config, "", "", null)) == null);
    try std.testing.expect((try dflashRowBounds(&config, null, null)) == null);
}
