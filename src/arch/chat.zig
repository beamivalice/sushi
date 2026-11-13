//! What each arch says about chat, tools and tokens: the effort words its template takes, a template it pins, and a
//! pre-tokenizer grammar of its own. The shared files read these here and never name an arch. MLX-free: an arch's
//! `chat` face is a plain file of constants and pure functions.
const std = @import("std");
const registry = @import("registry.zig");
const effort = @import("../effort.zig");

/// A chat template the arch serves instead of the one in the checkpoint (which does not parse in the engine), and the
/// atom an image becomes when that template reads `content` as a string.
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

/// The committed fixtures of the arch registered as `name` (its chat face's `fixtures`), for a shared test of that
/// arch's data. An empty struct where the build left the arch out, so the test checks `@hasDecl` and skips.
pub fn Fixtures(comptime name: []const u8) type {
    if (comptime !@hasField(registry.Tag, name)) return struct {};
    const face = faceOf(@field(registry.Tag, name));
    return if (@hasDecl(face, "fixtures")) face.fixtures else struct {};
}

/// The 400 body for thinking off on an arch whose template has no off switch; null where the arch can think off.
pub fn thinkOffRefusal(model_type: []const u8) ?[]const u8 {
    const tag = registry.ofModelType(model_type) orelse return null;
    switch (tag) {
        inline else => |t| return if (@hasDecl(faceOf(t), "think_off_refusal")) faceOf(t).think_off_refusal else null,
    }
}

/// The template with a late system turn rendered in place, when `model_type`'s own face edits it (`lateSystemTemplate`
/// of its chat face); null leaves the template and the fold to the caller. No arch's edit reaches another's template.
/// The caller frees it.
pub fn lateSystemTemplate(model_type: []const u8, allocator: std.mem.Allocator, tpl: []const u8) !?[]const u8 {
    const tag = registry.ofModelType(model_type) orelse return null;
    switch (tag) {
        inline else => |t| return if (@hasDecl(faceOf(t), "lateSystemTemplate")) faceOf(t).lateSystemTemplate(allocator, tpl) else null,
    }
}

/// Whether `model_type`'s chat face edits its template so a late system turn renders in place.
pub fn editsLateSystem(model_type: []const u8) bool {
    const tag = registry.ofModelType(model_type) orelse return false;
    switch (tag) {
        inline else => |t| return @hasDecl(faceOf(t), "lateSystemTemplate"),
    }
}

/// The first `model_type` the arch `name` serves, or null when the build leaves it out (tests of shared code).
pub fn modelTypeOf(comptime name: []const u8) ?[]const u8 {
    if (comptime !@hasField(registry.Tag, name)) return null;
    return faceOf(@field(registry.Tag, name)).model_types[0];
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
