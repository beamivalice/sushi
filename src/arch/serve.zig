//! What the server asks an arch at start and while serving, from the `serve` and `warm` parts of its `flags` slot.
//! MLX side: the host tier does not reach this file.
//!
//! A `serve` part may declare `install`/`uninstall`, `poolCapped(prev)` (no config, so every arch is asked),
//! `unloadModel(config)`, `enterVision(bill, config)`/`leaveVision(config)`, `plannedBorrowBytes(config, exl3,
//! cache_bytes)` and `loadLines(config, LoadLog)`; those take a config and only its arch is asked.
const std = @import("std");
const model = @import("../model.zig");
const registry = @import("registry.zig");
const arch_config = @import("config.zig");

fn flagsOf(comptime tag: registry.Tag) type {
    return registry.Mod(tag).flags;
}

/// Installs the hooks an arch's serving path reads for the life of `serve`; `pool_ram_cap` is the MLX pool's ceiling.
pub fn install(pool_ram_cap: u64) void {
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime @hasDecl(flagsOf(t), "serve")) flagsOf(t).serve.install(pool_ram_cap);
    }
}

pub fn uninstall() void {
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime @hasDecl(flagsOf(t), "serve")) flagsOf(t).serve.uninstall();
    }
}

/// Bytes of the n-gram table an arch is warming right now; zero when none is.
pub fn warmBytes(comptime order: std.builtin.AtomicOrder) u64 {
    var sum: u64 = 0;
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime @hasDecl(flagsOf(t), "warm")) sum += flagsOf(t).warm.bytes(order);
    }
    return sum;
}

/// What that table will hold once warm.
pub fn warmTotal(comptime order: std.builtin.AtomicOrder) u64 {
    var sum: u64 = 0;
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime @hasDecl(flagsOf(t), "warm")) sum += flagsOf(t).warm.total(order);
    }
    return sum;
}

fn serveHas(comptime tag: registry.Tag, comptime name: []const u8) bool {
    return @hasDecl(flagsOf(tag), "serve") and @hasDecl(flagsOf(tag).serve, name);
}

/// A model's MLX buffer pool was capped from the ledger; `prev` is the process-wide cap it replaced.
pub fn poolCapped(prev: u64) void {
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime serveHas(t, "poolCapped")) flagsOf(t).serve.poolCapped(prev);
    }
}

/// The `serve` part of the config's arch; each hook below asks it alone.
fn serveOf(config: *const model.ModelConfig, comptime name: []const u8, args: anytype) void {
    const tag = arch_config.tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (comptime serveHas(t, name)) @call(.auto, @field(flagsOf(t).serve, name), args),
    }
}

/// A model is unloaded: whatever pool cap its arch raised or lowered is given back.
pub fn unloadModel(config: *const model.ModelConfig) void {
    serveOf(config, "unloadModel", .{config});
}

/// An image encode of `bill` bytes starts on the vision thread.
pub fn enterVision(bill: u64, config: *const model.ModelConfig) void {
    serveOf(config, "enterVision", .{ bill, config });
}

pub fn leaveVision(config: *const model.ModelConfig) void {
    serveOf(config, "leaveVision", .{config});
}

/// The decoder pools a prefill may release, estimated at load from an expert cache of `cache_bytes`; 0 for an arch that lends none.
pub fn plannedBorrowBytes(config: *const model.ModelConfig, exl3: bool, cache_bytes: u64) u64 {
    const tag = arch_config.tagOf(config) orelse return 0;
    return switch (tag) {
        inline else => |t| if (comptime serveHas(t, "plannedBorrowBytes")) flagsOf(t).serve.plannedBorrowBytes(config, exl3, cache_bytes) else 0,
    };
}

/// What a load's arch log lines may read.
pub const LoadLog = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    /// Where the MTP choice came from (`mtp.sourceName()`).
    mtp_source: []const u8,
};

/// The arch's own lines of the load log, after the shared KV and MTP lines.
pub fn loadLines(config: *const model.ModelConfig, ctx: LoadLog) void {
    serveOf(config, "loadLines", .{ config, ctx });
}
