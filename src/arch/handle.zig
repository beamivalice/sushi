//! The engine's one door to a loaded model: the handle it holds (`Handle`, the registry's union of each arch's
//! `Model`) and every read it makes of one. Each read is one switch on the tag, resolved at compile time per prong.
const std = @import("std");
const mlx = @import("../mlx.zig");
const trunk_mod = @import("../trunk.zig");
const registry = @import("registry.zig");
const arch_request = @import("request.zig");
const arch_life = @import("life.zig");
const Trunk = trunk_mod.Trunk;

pub const Handle = registry.Arch;

/// The handle of a model built as `t.arch`.
pub fn handleOf(t: *Trunk) Handle {
    return registry.handleOf(t);
}

/// The handle of a trunk a test built by hand, which carries no arch stamp: it dispatches as the first built arch
/// wherever the tag decides, whatever its config names; every read through `trunk` is the same on each prong.
pub fn fixtureHandle(t: *Trunk) Handle {
    t.arch = std.enums.values(registry.Tag)[0];
    return handleOf(t);
}

/// The engine state the generic paths run on: an arch's `host(m)` when its `Model` is not that state, else the
/// `Model` itself.
pub fn trunk(h: Handle) *Trunk {
    return switch (h) {
        inline else => |m, t| if (comptime @hasDecl(registry.Mod(t), "host")) registry.Mod(t).host(m) else m,
    };
}

/// Frees the model, the arch's own state first, and the handle's allocation.
pub fn destroy(h: Handle, allocator: std.mem.Allocator) void {
    arch_life.deinit(h);
    switch (h) {
        inline else => |m| allocator.destroy(m),
    }
}

pub fn stream(h: Handle) mlx.mlx_stream {
    return trunk(h).s;
}

pub fn config(h: Handle) *const trunk_mod.ModelConfig {
    return &trunk(h).config;
}

/// Whether the model keeps decode state of its own, so admission lets one request in at a time.
pub fn ownsModuleDecodeState(h: Handle) bool {
    return arch_request.ownsModuleDecodeState(h);
}

/// Module archs run only their own MTP head; PLD and drafters stay off (`model_handle.moduleSpecWiring`).
pub fn moduleSpecWiring(h: Handle) bool {
    return trunk(h).moduleSpecWiring();
}

/// Whether the arch's own decode state rewinds across a speculative verify (`model_handle.moduleStateSpecRollback`).
pub fn moduleStateSpecRollback(h: Handle) bool {
    return trunk(h).moduleStateSpecRollback();
}

pub fn logitBias(h: Handle) []const @import("../logit_bias.zig").Bias {
    return trunk(h).logit_bias;
}

pub fn flushImatrix(h: Handle) void {
    trunk(h).flushImatrix();
}

test "a handle reads back the trunk it was made from, its stream, config and logit bias" {
    inline for (comptime std.enums.values(registry.Tag)) |tag| {
        var t: Trunk = undefined;
        t.arch = tag;
        t.s = mlx.gpuStream();
        t.config = .{ .model_type = "handle-test" };
        t.logit_bias = &.{};
        const h = handleOf(&t);
        try std.testing.expectEqual(tag, std.meta.activeTag(h));
        try std.testing.expectEqual(&t, trunk(h));
        try std.testing.expectEqual(t.s.ctx, stream(h).ctx);
        try std.testing.expectEqualStrings("handle-test", config(h).model_type);
        try std.testing.expectEqual(@as(usize, 0), logitBias(h).len);
    }
}
