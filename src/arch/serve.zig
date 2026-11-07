//! What the server asks an arch at start and while serving, from the `serve` and `warm` parts of its `flags` slot.
//! MLX side: the host tier does not reach this file.
const std = @import("std");
const registry = @import("registry.zig");

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
