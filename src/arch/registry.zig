//! The one file meant to name an architecture outside `src/<arch>/`. The target is that dropping one means deleting
//! its folder and its `// arch:<name>` line; shared files still name archs, and `tests/test_arch_drop.sh` lists the
//! strays. `-Darchs=` builds a subset: an arch left out has no prong in `Tag`, `Spec` or `Arch`.
//! No test lives here: the host tier (no MLX) reaches `Chat`, and a test in this file would pull every `arch.zig`
//! in with it. The tests are `registry_test.zig`.
const std = @import("std");
const opts = @import("build_options");

// One line per arch.
// `mod` is its `arch.zig` and `chat` its MLX-free face. They are separate decls, so a file is read only where it is
// used: the host tier uses `chat` alone, and an arch left out of `-Darchs` is read by neither. The line's own
// `model_types` and `gate_rank` are what the arch gate needs of an arch the build left out, without reading its files.
const archs = struct {
    pub const glm5 = struct { const name = "glm5"; const model_types = [_][]const u8{"glm5_next"}; const gate_rank = 2; const mod = @import("../glm5/arch.zig"); const chat = @import("../glm5/glm5_chat.zig"); }; // arch:glm5
    pub const dsv41 = struct { const name = "dsv41"; const model_types = [_][]const u8{"deepseek_v41"}; const gate_rank = 3; const mod = @import("../dsv41/arch.zig"); const chat = @import("../dsv41/dsv41_chat.zig"); }; // arch:dsv41
    pub const qwen4 = struct { const name = "qwen4"; const model_types = [_][]const u8{"qwen4_exp"}; const gate_rank = 0; const mod = @import("../qwen4/arch.zig"); const chat = @import("../qwen4/qwen4_chat.zig"); }; // arch:qwen4
    pub const mimo2 = struct { const name = "mimo2"; const model_types = [_][]const u8{"mimo_v2"}; const gate_rank = 1; const mod = @import("../mimo2/arch.zig"); const chat = @import("../mimo2/mimo_chat.zig"); }; // arch:mimo2
};

const all_names = blk: {
    const decls = std.meta.declarations(archs);
    var out: [decls.len][:0]const u8 = undefined;
    for (decls, &out) |d, *name| name.* = @field(archs, d).name;
    break :blk out;
};

pub fn selected(wanted: []const []const u8, name: []const u8) bool {
    if (wanted.len == 0) return true;
    for (wanted) |w| if (std.mem.eql(u8, w, name)) return true;
    return false;
}

fn registered(name: []const u8) bool {
    for (all_names) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

const enabled = blk: {
    for (opts.archs) |w| if (!registered(w))
        @compileError("-Darchs names \"" ++ w ++ "\", which src/arch/registry.zig does not register");
    var buf: [all_names.len][:0]const u8 = undefined;
    var n: usize = 0;
    for (all_names) |name| if (selected(opts.archs, name)) {
        buf[n] = name;
        n += 1;
    };
    if (n == 0) @compileError("-Darchs leaves no architecture to build");
    const out = buf[0..n].*;
    break :blk out;
};

pub const Tag = blk: {
    var names: [enabled.len][]const u8 = undefined;
    var values: [enabled.len]u8 = undefined;
    for (enabled, &names, &values, 0..) |name, *n, *value, i| {
        n.* = name;
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
    for (std.meta.tags(Tag), &names, &types) |t, *name, *ty| {
        name.* = @tagName(t);
        ty.* = if (by_const_ptr) *const @field(Mod(t), decl) else @field(Mod(t), decl);
    }
    return @Union(.auto, Tag, &names, &types, &@splat(.{}));
}

/// The arch module behind a tag.
pub fn Mod(comptime tag: Tag) type {
    return @field(archs, @tagName(tag)).mod;
}

/// Where the arch gate lists the arch: the `gate_rank` of its registry line.
pub fn gateRank(comptime tag: Tag) u8 {
    return @field(archs, @tagName(tag)).gate_rank;
}

/// The arch's MLX-free chat face (`<arch>_chat.zig`): effort offer, pinned template, pre-tokenizer.
pub fn Chat(comptime tag: Tag) type {
    return @field(archs, @tagName(tag)).chat;
}

pub fn ofModelType(model_type: []const u8) ?Tag {
    inline for (comptime std.meta.tags(Tag)) |t| {
        for (Chat(t).model_types) |m| if (std.mem.eql(u8, m, model_type)) return t;
    }
    return null;
}

/// The arch that runs a model type no arch claims: today the generic MoE trunk, which declares `default_arch`. Null
/// when its arch is not built.
pub const default: ?Tag = blk: {
    for (std.meta.tags(Tag)) |t| if (@hasDecl(Mod(t), "default_arch")) break :blk t;
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
    for (std.meta.tags(Tag)) |t| for (Chat(t).model_types) |m| {
        out = out ++ &[_][]const u8{m};
    };
    break :blk out;
};

/// Every registered arch's model types, enabled or not, in the order the arch gate's refusal lists them (`gate_rank`).
pub const registered_model_types: []const []const u8 = blk: {
    const decls = std.meta.declarations(archs);
    var order: [decls.len]type = undefined;
    for (decls, &order) |d, *a| a.* = @field(archs, d);
    for (1..order.len) |i| {
        var j = i;
        while (j > 0 and order[j].gate_rank < order[j - 1].gate_rank) : (j -= 1) {
            const a = order[j];
            order[j] = order[j - 1];
            order[j - 1] = a;
        }
    }
    var out: []const []const u8 = &.{};
    for (order) |a| for (a.model_types) |m| {
        out = out ++ &[_][]const u8{m};
    };
    break :blk out;
};

/// The model types `tag`'s registry line names.
pub fn lineModelTypes(comptime tag: Tag) []const []const u8 {
    return &@field(archs, @tagName(tag)).model_types;
}
