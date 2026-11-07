//! Construction and teardown of a model. Inference thread only.
const std = @import("std");
const mlx = @import("../mlx.zig");
const trunk = @import("../trunk.zig");
const model = @import("../model.zig");
const registry = @import("registry.zig");
const Arch = registry.Arch;

/// The arch that builds `model_type`. A type the model layer serves but this build left out is refused; a type no arch
/// names takes the default arch's trunk, as an unnamed config always has (the arch gate refuses it at load).
fn route(model_type: []const u8) error{ArchitectureUnsupported}!registry.Tag {
    if (registry.ofModelType(model_type)) |tag| return tag;
    if (model.isServedArch(model_type)) return error.ArchitectureUnsupported;
    return registry.default orelse error.ArchitectureUnsupported;
}

/// Builds the model of `config.model_type` and stamps the arch it was built as, which every later dispatch reads.
pub fn init(io: std.Io, allocator: std.mem.Allocator, config: trunk.ModelConfig, weights: *trunk.Weights, s: mlx.mlx_stream) !trunk.Trunk {
    return switch (try route(config.model_type)) {
        inline else => |t| blk: {
            var m = try registry.Mod(t).init(io, allocator, config, weights, s);
            m.arch = t;
            break :blk m;
        },
    };
}

/// Frees the arch's own state, then the trunk's.
pub fn deinit(a: Arch) void {
    switch (a) {
        inline else => |m, t| registry.Mod(t).deinit(m),
    }
}

/// Empties every cache for a new request.
pub fn resetCache(a: Arch) !void {
    return switch (a) {
        inline else => |m, t| registry.Mod(t).resetCache(m),
    };
}

/// Pre-faults the weights and compiles the first-touch kernels with throwaway forwards, then resets.
pub fn warmup(a: Arch) !void {
    return switch (a) {
        inline else => |m, t| registry.Mod(t).warmup(m),
    };
}

test "a model type no arch names goes to the default arch" {
    if (registry.default == null) return;
    try std.testing.expectEqual(registry.default.?, try route("mistral"));
    try std.testing.expectEqual(registry.default.?, try route(""));
}

test "a served type the build left out is refused, a built one is routed to its arch" {
    for (model.served_model_types) |t| {
        if (registry.ofModelType(t)) |tag| {
            try std.testing.expectEqual(tag, try route(t));
        } else {
            try std.testing.expectError(error.ArchitectureUnsupported, route(t));
        }
    }
}
