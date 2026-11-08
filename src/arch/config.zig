//! The config side of the arch seam: the arch a parsed config carries, each arch's parse, and what a config states
//! about its arch without loading it. An arch declares its part in the `slot:config` lines of its `arch.zig`.
const std = @import("std");
const model = @import("../model.zig");
const registry = @import("registry.zig");

pub const Tag = registry.Tag;

/// The arch of a config, resolved once by the parse. A config built by hand names its arch by `model_type` alone.
/// `model_type` must not be reassigned after the parse: the tag and the string would then disagree.
pub const ArchSpec = struct {
    tag: ?Tag = null,
};

/// Constants an arch states about its config; the defaults are what an arch with nothing special says.
pub const Facts = struct {
    /// A request that names no thinking preference renders with thinking on.
    thinks_by_default: bool = false,
    /// Its packs may store pair-major expert tiles.
    reads_pair_major: bool = false,
    /// The routed layers start here until the config's own dense prefix is read.
    stub_first_moe: u16 = 0,
    /// The config's `moe_layer_freq` names the dense prefix.
    dense_prefix_in_layer_freq: bool = false,
    /// Video input is wired.
    serves_video: bool = true,
    /// The residual is a set of hyper-connection streams: the layers carry no norms (the stream read normalizes),
    /// each loads its hyper-connection weights and the PLE, and the trunk keeps a weight-less RMS norm's ones.
    hyper_connected: bool = false,
};

/// `Facts.hyper_connected` of the config's arch; false for an unclaimed config.
pub fn hyperConnected(config: *const model.ModelConfig) bool {
    return if (facts(config)) |f| f.hyper_connected else false;
}

pub fn tagOf(config: *const model.ModelConfig) ?Tag {
    return config.arch.tag orelse registry.ofModelType(config.model_type);
}

pub fn is(config: *const model.ModelConfig, comptime name: []const u8) bool {
    const want = comptime std.meta.stringToEnum(Tag, name);
    return want != null and tagOf(config) == want.?;
}

fn claims(comptime types: anytype, model_type: []const u8) bool {
    inline for (types) |t| if (std.mem.eql(u8, t, model_type)) return true;
    return false;
}

/// The arch whose config parse reads `model_type`: its served types and the text-only twins of them.
pub fn ofConfigType(model_type: []const u8) ?Tag {
    inline for (comptime std.enums.values(Tag)) |tag| {
        if (claims(registry.Mod(tag).config_types, model_type)) return tag;
    }
    return null;
}

/// Whether discovery lists a directory declaring `model_type`.
pub fn discovers(model_type: []const u8) bool {
    inline for (comptime std.enums.values(Tag)) |tag| {
        if (claims(registry.Mod(tag).discovery_types, model_type)) return true;
    }
    return false;
}

/// The facts of the arch that serves `model_type`; null when no enabled arch does.
pub fn factsOfType(model_type: []const u8) ?Facts {
    const tag = registry.ofModelType(model_type) orelse return null;
    return factsOf(tag);
}

fn factsOf(tag: Tag) Facts {
    return switch (tag) {
        inline else => |t| registry.Mod(t).config_facts,
    };
}

pub fn facts(config: *const model.ModelConfig) ?Facts {
    return factsOf(tagOf(config) orelse return null);
}

/// The arch's part of the parse, on top of the fields every arch shares.
pub fn parseFamily(config: *model.ModelConfig, tag: Tag, root: std.json.ObjectMap, cfg_obj: std.json.ObjectMap) !void {
    config.arch = .{ .tag = tag };
    switch (tag) {
        inline else => |t| try registry.Mod(t).parseConfig(config, root, cfg_obj),
    }
}

/// What the arch adds once the pack directory is known (paths, environment).
pub fn parseDir(config: *model.ModelConfig, allocator: std.mem.Allocator, model_dir: []const u8) !void {
    const tag = tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "parseDir")) try registry.Mod(t).parseDir(config, allocator, model_dir),
    }
}

/// What the arch changes once the pack's expert layout is known.
pub fn adoptLayout(config: *model.ModelConfig) void {
    const tag = tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "adoptLayout")) registry.Mod(t).adoptLayout(config),
    }
}

/// The family's documented sampling truncation, for a checkpoint that ships no generation_config.json.
pub fn samplingDefaults(config: *model.ModelConfig) void {
    const tag = tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "samplingDefaults")) registry.Mod(t).samplingDefaults(config),
    }
}

/// What a streamed load keeps resident when the arch reads its own source layout; null takes the indexed scan.
pub fn streamedResidentSplit(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const model.ModelConfig) !?model.ResidentSplit {
    const tag = tagOf(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "streamedResidentSplit"))
            try registry.Mod(t).streamedResidentSplit(io, allocator, model_dir, config)
        else
            null,
    };
}

/// The names the arch gate's refusal lists: the enabled archs, in the order they were first documented.
pub const gate_text = blk: {
    var text: []const u8 = "";
    for (registry.registered_model_types) |t| {
        if (registry.ofModelType(t) != null) text = text ++ (if (text.len > 0) ", " else "") ++ t;
    }
    break :blk text;
};

test "every enabled arch declares its config side" {
    inline for (comptime std.enums.values(Tag)) |tag| {
        const M = registry.Mod(tag);
        inline for (.{ "config_types", "discovery_types", "config_facts", "parseConfig" }) |decl| {
            if (!@hasDecl(M, decl)) @compileError(@tagName(tag) ++ "/arch.zig lacks " ++ decl);
        }
        for (M.model_types) |t| try std.testing.expect(claims(M.config_types, t) and claims(M.discovery_types, t));
    }
}

test "the arch gate text lists every enabled arch's model types and no other" {
    for (registry.served_model_types) |t| try std.testing.expect(std.mem.indexOf(u8, gate_text, t) != null);
    for (registry.registered_model_types) |t| {
        if (registry.ofModelType(t) == null) try std.testing.expect(std.mem.indexOf(u8, gate_text, t) == null);
    }
}

test "a config type resolves to the arch that parses it and an unknown one to none" {
    try std.testing.expectEqual(@as(?Tag, null), ofConfigType("mistral"));
    try std.testing.expectEqual(@as(?Tag, null), ofConfigType(""));
    inline for (comptime std.enums.values(Tag)) |tag| {
        for (registry.Mod(tag).config_types) |t| try std.testing.expectEqual(@as(?Tag, tag), ofConfigType(t));
    }
}

test "the arch gate text keeps the order its refusal has always listed" {
    if (registry.served_model_types.len != 4) return;
    try std.testing.expectEqualStrings("qwen4_exp, mimo_v2, glm5_next, deepseek_v41", gate_text);
}
