//! The registry's tests live apart from it: `registry.zig` is reached by the host tier, which links no MLX, and these
//! read every `arch.zig`.
const std = @import("std");
const opts = @import("build_options");
const registry = @import("registry.zig");

const Tag = registry.Tag;
const Mod = registry.Mod;
const tags = std.meta.tags(Tag);

test "every enabled arch's own tests are part of the build" {
    inline for (tags) |t| _ = Mod(t);
}

test "an empty arch list builds every arch, a named one only itself" {
    const testing = std.testing;
    try testing.expect(registry.selected(&.{}, "glm5"));
    try testing.expect(registry.selected(&.{ "qwen4", "glm5" }, "glm5"));
    try testing.expect(!registry.selected(&.{ "qwen4", "glm5" }, "dsv41"));
}

test "a model type resolves to the arch that claims it" {
    const testing = std.testing;
    inline for (.{ .{ "qwen4_exp", "qwen4" }, .{ "mimo_v2", "mimo2" }, .{ "glm5_next", "glm5" }, .{ "deepseek_v41", "dsv41" } }) |c| {
        if (comptime registry.selected(opts.archs, c[1])) try testing.expectEqualStrings(c[1], @tagName(registry.ofModelType(c[0]).?));
    }
    try testing.expectEqual(@as(?Tag, null), registry.ofModelType("mistral"));
    try testing.expectEqual(@as(?Tag, null), registry.ofModelType(""));
}

test "every enabled arch declares the operations the dispatch files call" {
    inline for (tags) |t| {
        inline for (.{ "model_types", "Config", "Model", "packs", "flags", "chat", "init", "deinit", "resetCache", "warmup", "forward", "lmHeadForDraft", "ownsModuleDecodeState", "expertStream" }) |decl| {
            if (!@hasDecl(Mod(t), decl)) @compileError(@tagName(t) ++ "/arch.zig lacks " ++ decl);
        }
    }
    comptime var defaults = 0;
    inline for (tags) |t| {
        if (@hasDecl(Mod(t), "default_arch")) defaults += 1;
    }
    try std.testing.expect(defaults <= 1);
}

test "an arch's registry line names its chat face's model types" {
    inline for (tags) |t| {
        const line = registry.lineModelTypes(t);
        try std.testing.expectEqual(Mod(t).chat.model_types.len, line.len);
        for (Mod(t).chat.model_types, line) |a, b| try std.testing.expectEqualStrings(a, b);
    }
}

test "an arch's chat face is the one its arch.zig re-exports" {
    inline for (tags) |t| try std.testing.expect(Mod(t).chat == registry.Chat(t));
}

test "served model types list each enabled arch's types once" {
    const testing = std.testing;
    var n: usize = 0;
    inline for (tags) |t| n += Mod(t).model_types.len;
    try testing.expectEqual(n, registry.served_model_types.len);
    for (registry.served_model_types, 0..) |t, i| {
        for (registry.served_model_types[i + 1 ..]) |u| try testing.expect(!std.mem.eql(u8, t, u));
    }
}
