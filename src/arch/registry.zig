//! The one file meant to name an architecture outside `src/<arch>/`. The target is that dropping one means deleting
//! its folder and its `// arch:<name>` line; shared files still name archs, and `tests/test_arch_drop.sh` lists the
//! strays. `-Darchs=` builds a subset: an arch left out has no prong in `Tag`, `Spec` or `Arch`.
const std = @import("std");
const opts = @import("build_options");

const Entry = struct { name: [:0]const u8, mod: type };

const entries = [_]Entry{
    .{ .name = "glm5", .mod = @import("../glm5/arch.zig") }, // arch:glm5
    .{ .name = "dsv41", .mod = @import("../dsv41/arch.zig") }, // arch:dsv41
    .{ .name = "qwen4", .mod = @import("../qwen4/arch.zig") }, // arch:qwen4
    .{ .name = "mimo2", .mod = @import("../mimo2/arch.zig") }, // arch:mimo2
};

pub fn selected(wanted: []const []const u8, name: []const u8) bool {
    if (wanted.len == 0) return true;
    for (wanted) |w| if (std.mem.eql(u8, w, name)) return true;
    return false;
}

fn registered(name: []const u8) bool {
    for (entries) |e| if (std.mem.eql(u8, e.name, name)) return true;
    return false;
}

const enabled = blk: {
    for (opts.archs) |w| if (!registered(w))
        @compileError("-Darchs names \"" ++ w ++ "\", which src/arch/registry.zig does not register");
    var buf: [entries.len]Entry = undefined;
    var n: usize = 0;
    for (entries) |e| if (selected(opts.archs, e.name)) {
        buf[n] = e;
        n += 1;
    };
    if (n == 0) @compileError("-Darchs leaves no architecture to build");
    const out = buf[0..n].*;
    break :blk out;
};

pub const Tag = blk: {
    var names: [enabled.len][]const u8 = undefined;
    var values: [enabled.len]u8 = undefined;
    for (enabled, &names, &values, 0..) |e, *name, *value, i| {
        name.* = e.name;
        value.* = i;
    }
    break :blk @Enum(u8, .exhaustive, &names, &values);
};

/// What an arch module declares as its `Config`: the MLX-free prong of `Spec`.
pub const Spec = Tagged("Config", true);
/// What an arch module declares as its `Model`: the runtime prong of `Arch`, touched on the inference thread only.
pub const Arch = Tagged("Model", false);

fn Tagged(comptime decl: []const u8, comptime by_const_ptr: bool) type {
    var names: [enabled.len][]const u8 = undefined;
    var types: [enabled.len]type = undefined;
    for (enabled, &names, &types) |e, *name, *ty| {
        name.* = e.name;
        ty.* = if (by_const_ptr) *const @field(e.mod, decl) else @field(e.mod, decl);
    }
    return @Union(.auto, Tag, &names, &types, &@splat(.{}));
}

/// The arch module behind a tag.
pub fn Mod(comptime tag: Tag) type {
    return enabled[@intFromEnum(tag)].mod;
}

pub fn ofModelType(model_type: []const u8) ?Tag {
    inline for (enabled, 0..) |e, i| {
        for (e.mod.model_types) |t| if (std.mem.eql(u8, t, model_type)) return @enumFromInt(i);
    }
    return null;
}

/// The arch that runs a model type no arch claims: today the generic MoE trunk, which declares `default_arch`. Null
/// when its arch is not built.
pub const default: ?Tag = blk: {
    for (enabled, 0..) |e, i| if (@hasDecl(e.mod, "default_arch")) break :blk @enumFromInt(i);
    break :blk null;
};

/// The handle of a loaded model: the arch `life.init` stamped on it, no test of its state.
pub fn handleOf(model: anytype) Arch {
    return switch (model.arch) {
        inline else => |t| @unionInit(Arch, @tagName(t), model),
    };
}

pub const served_model_types: []const []const u8 = blk: {
    var out: []const []const u8 = &.{};
    for (enabled) |e| for (e.mod.model_types) |t| {
        out = out ++ &[_][]const u8{t};
    };
    break :blk out;
};

/// Every registered arch's model types, enabled or not, in the order the arch gate's refusal lists them (`gate_rank`).
pub const registered_model_types: []const []const u8 = blk: {
    var order: [entries.len]Entry = entries;
    for (1..order.len) |i| {
        var j = i;
        while (j > 0 and order[j].mod.gate_rank < order[j - 1].mod.gate_rank) : (j -= 1) {
            const t = order[j];
            order[j] = order[j - 1];
            order[j - 1] = t;
        }
    }
    var out: []const []const u8 = &.{};
    for (order) |e| for (e.mod.model_types) |t| {
        out = out ++ &[_][]const u8{t};
    };
    break :blk out;
};

test "every enabled arch's own tests are part of the build" {
    inline for (enabled) |e| _ = e.mod;
}

test "an empty arch list builds every arch, a named one only itself" {
    const testing = std.testing;
    try testing.expect(selected(&.{}, "glm5"));
    try testing.expect(selected(&.{ "qwen4", "glm5" }, "glm5"));
    try testing.expect(!selected(&.{ "qwen4", "glm5" }, "dsv41"));
}

test "a model type resolves to the arch that claims it" {
    const testing = std.testing;
    inline for (.{ .{ "qwen4_exp", "qwen4" }, .{ "mimo_v2", "mimo2" }, .{ "glm5_next", "glm5" }, .{ "deepseek_v41", "dsv41" } }) |c| {
        if (comptime selected(opts.archs, c[1])) try testing.expectEqualStrings(c[1], @tagName(ofModelType(c[0]).?));
    }
    try testing.expectEqual(@as(?Tag, null), ofModelType("mistral"));
    try testing.expectEqual(@as(?Tag, null), ofModelType(""));
}

test "every enabled arch declares the operations the dispatch files call" {
    inline for (enabled) |e| {
        inline for (.{ "model_types", "Config", "Model", "packs", "flags", "init", "deinit", "resetCache", "warmup", "forward", "lmHeadForDraft", "ownsModuleDecodeState", "expertStream" }) |decl| {
            if (!@hasDecl(e.mod, decl)) @compileError(e.name ++ "/arch.zig lacks " ++ decl);
        }
    }
    comptime var defaults = 0;
    inline for (enabled) |e| {
        if (@hasDecl(e.mod, "default_arch")) defaults += 1;
    }
    try std.testing.expect(defaults <= 1);
}

test "served model types list each enabled arch's types once" {
    const testing = std.testing;
    try testing.expectEqual(enabled.len, served_model_types.len);
    for (served_model_types, 0..) |t, i| {
        for (served_model_types[i + 1 ..]) |u| try testing.expect(!std.mem.eql(u8, t, u));
    }
}
