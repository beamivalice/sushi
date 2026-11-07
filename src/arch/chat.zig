//! What each arch says about chat, tools and tokens: the effort words its template takes, a template it pins, and a
//! pre-tokenizer grammar of its own. The shared files read these here and never name an arch. MLX-free: an arch's
//! `chat` face is a plain file of constants and pure functions.
const std = @import("std");
const registry = @import("registry.zig");
const effort = @import("../effort.zig");

/// A chat template the arch serves instead of the one in the checkpoint (its encoder is not jinja.cpp-parseable),
/// and the atom an image becomes when that template reads `content` as a string.
pub const Pin = struct { template: []const u8, image_placeholder: ?[]const u8 = null };

fn faceOf(comptime tag: registry.Tag) type {
    return registry.Chat(tag);
}

/// The effort offer of the arch that claims `model_type`; null for an arch with none.
pub fn effortOffer(model_type: []const u8) ?*const effort.Offer {
    const tag = registry.ofModelType(model_type) orelse return null;
    switch (tag) {
        inline else => |t| {
            if (!@hasDecl(faceOf(t), "effort")) return null;
            const S = struct {
                const offer: effort.Offer = faceOf(t).effort;
            };
            return &S.offer;
        },
    }
}

/// The editable effort table filed under `section` in model-settings.json; null when its arch is not built.
pub fn tableSpec(comptime section: []const u8) ?*const effort.Spec {
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (@hasDecl(faceOf(t), "effort")) {
            const offer = faceOf(t).effort;
            if (offer == .table and std.mem.eql(u8, offer.table.section, section)) {
                const S = struct {
                    const spec: effort.Spec = offer.table;
                };
                return &S.spec;
            }
        }
    }
    return null;
}

pub fn pinOf(model_type: []const u8) ?Pin {
    const tag = registry.ofModelType(model_type) orelse return null;
    switch (tag) {
        inline else => |t| return if (@hasDecl(faceOf(t), "pin")) faceOf(t).pin else null,
    }
}

fn pretokNames() []const []const u8 {
    comptime var names: []const []const u8 = &.{"gpt2"};
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (@hasDecl(faceOf(t), "pretokenizer")) names = names ++ &[_][]const u8{faceOf(t).pretokenizer.name};
    }
    return names;
}

/// The pre-tokenizer grammars: `gpt2` (the shared rules in tokenizer.zig) and one per arch that brings its own.
pub const PretokStyle = blk: {
    const names = pretokNames();
    var values: [names.len]u8 = undefined;
    for (&values, 0..) |*v, i| v.* = i;
    break :blk @Enum(u8, .exhaustive, names, &values);
};

/// The arch grammar whose tokenizer.json spelling `pre_tokenizer` is; `gpt2` when none claims it.
pub fn pretokStyleOf(pre_tokenizer: std.json.Value) PretokStyle {
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (@hasDecl(faceOf(t), "pretokenizer")) {
            if (faceOf(t).pretokenizer.matches(pre_tokenizer)) return @field(PretokStyle, faceOf(t).pretokenizer.name);
        }
    }
    return .gpt2;
}

/// Splits `text` by an arch grammar; the `gpt2` rules are tokenizer.zig's own.
pub fn preTokenize(style: PretokStyle, allocator: std.mem.Allocator, text: []const u8, words: *std.ArrayList([]const u8)) !void {
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (@hasDecl(faceOf(t), "pretokenizer")) {
            if (style == @field(PretokStyle, faceOf(t).pretokenizer.name)) return faceOf(t).pretokenizer.split(allocator, text, words);
        }
    }
    unreachable;
}

test "every enabled arch's chat face is part of the build" {
    inline for (comptime std.meta.tags(registry.Tag)) |t| _ = faceOf(t);
}
