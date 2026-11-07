//! The MLX-free face of an arch: what a config says, before any weight loads.
const std = @import("std");
const model = @import("../model.zig");
const registry = @import("registry.zig");
pub const Spec = registry.Spec;

/// The arch that claims the config's `model_type`; null when no enabled arch does.
pub fn of(config: *const model.ModelConfig) ?Spec {
    const tag = registry.ofModelType(config.model_type) orelse return null;
    return switch (tag) {
        inline else => |t| @unionInit(Spec, @tagName(t), config),
    };
}

test "the registry serves the model types model.zig gates on" {
    const served = registry.served_model_types;
    for (model.served_model_types) |t| {
        if (registry.ofModelType(t) == null) continue;
        try std.testing.expect(for (served) |s| {
            if (std.mem.eql(u8, s, t)) break true;
        } else false);
    }
    for (served) |t| try std.testing.expect(model.isServedArch(t));
}

test "a config resolves to the spec prong of its model type and an unknown one to none" {
    var config = model.ModelConfig{ .model_type = "mistral" };
    try std.testing.expect(of(&config) == null);
    inline for (registry.Mod(@as(registry.Tag, @enumFromInt(0))).model_types) |t| {
        config.model_type = t;
        try std.testing.expectEqual(@as(registry.Tag, @enumFromInt(0)), std.meta.activeTag(of(&config).?));
    }
}
