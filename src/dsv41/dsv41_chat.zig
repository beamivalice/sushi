//! DeepSeek-V4.1's chat face (`arch/chat.zig` reads it): its effort scale, the encoder-pinned template, and the
//! pre-tokenizer its tokenizer.json spells.
const std = @import("std");
const eff = @import("../effort.zig");
const arch_chat = @import("../arch/chat.zig");
const tokenizer = @import("../tokenizer.zig");
const pretok = @import("tokenizer_deepseek.zig");

pub const model_types = [_][]const u8{"deepseek_v41"};

/// The template takes a 1-100 scale; its own three steps are low 50, high 75 and max 100.
pub const effort: eff.Offer = .{ .table = eff.makeSpec("deepseek_v41", "DeepSeek", &.{}, &.{ .{ .word = .low, .value = 50 }, .{ .word = .high, .value = 75 }, .{ .word = .max, .value = 100 } }, true, .{ .default = .high, .rows = .{
    .{ .offered = true },
    .{ .offered = true, .sends = .{ .scale = 50 } },
    .{ .offered = true, .sends = .{ .scale = 62 } },
    .{ .offered = true, .sends = .{ .scale = 75 } },
    .{ .offered = true, .sends = .{ .scale = 87 } },
    .{ .offered = true, .sends = .{ .scale = 100 } },
} }) };

/// The release's image placeholder, which the processor expands into the image's span.
pub const image_atom = "<\xef\xbd\x9cdeepseek_image\xef\xbd\x9c>";

// The checkpoint's template merges dicts with `**`, which jinja.cpp cannot parse; this one is pinned to the encoder.
pub const fixtures = @import("dsv41_fixtures.zig");

pub const pin: arch_chat.Pin = .{ .template = @embedFile("../fixtures/dsv41_chat_template.jinja"), .image_placeholder = image_atom };

pub const pretokenizer = struct {
    pub const name = "deepseek";
    pub const split = pretok.preTokenize;

    /// V4.x's `Sequence`: its three Splits (digits, CJK, the punctuation grammar), each isolated, then a ByteLevel
    /// that does not split. Only that exact spelling is read as the grammar `tokenizer_deepseek` runs.
    pub fn matches(pre_tokenizer: std.json.Value) bool {
        if (pre_tokenizer != .object) return false;
        const list = pre_tokenizer.object.get("pretokenizers") orelse return false;
        if (list != .array or list.array.items.len != 4) return false;
        const regexes = [_][]const u8{ pretok.digits_regex, pretok.cjk_regex, pretok.main_regex };
        for (regexes, list.array.items[0..3]) |want, node| {
            if (node != .object) return false;
            const rx = tokenizer.splitRegexOf(node) orelse return false;
            const behavior = node.object.get("behavior") orelse return false;
            const invert = node.object.get("invert") orelse return false;
            if (!std.mem.eql(u8, rx, want) or behavior != .string or !std.mem.eql(u8, behavior.string, "Isolated") or invert != .bool or invert.bool) return false;
        }
        const byte_level = list.array.items[3];
        if (byte_level != .object) return false;
        const ty = byte_level.object.get("type") orelse return false;
        if (ty != .string or !std.mem.eql(u8, ty.string, "ByteLevel")) return false;
        const use_regex = byte_level.object.get("use_regex") orelse return false;
        return use_regex == .bool and !use_regex.bool;
    }
};

test "DeepSeek's built-in effort table is the encoder's 1-100 ladder" {
    const d = effort.table.table;
    for ([_]eff.Effort{ .low, .medium, .high, .xhigh, .max }, [_]u8{ 50, 62, 75, 87, 100 }) |w, n| try std.testing.expectEqualDeep(eff.Row{ .offered = true, .sends = .{ .scale = n } }, d.row(w));
}

test "the pinned template and image atom are the encoder's" {
    try std.testing.expectEqualStrings("<\u{ff5c}deepseek_image\u{ff5c}>", image_atom);
    try std.testing.expect(std.mem.indexOf(u8, pin.template, "thinking_mode") != null);
}

test "DeepSeek's Sequence of Splits is read as its own pre-tokenizer, any other spelling is not" {
    if (@import("../arch/registry.zig").ofModelType("deepseek_v41") == null) return error.SkipZigTest;
    const a = std.testing.allocator;
    const j = std.json.fmt;
    const deepseek = try std.fmt.allocPrint(a,
        \\{{"type":"Sequence","pretokenizers":[
        \\{{"type":"Split","pattern":{{"Regex":{f}}},"behavior":"Isolated","invert":false}},
        \\{{"type":"Split","pattern":{{"Regex":{f}}},"behavior":"Isolated","invert":false}},
        \\{{"type":"Split","pattern":{{"Regex":{f}}},"behavior":"Isolated","invert":false}},
        \\{{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}}]}}
    , .{ j(pretok.digits_regex, .{}), j(pretok.cjk_regex, .{}), j(pretok.main_regex, .{}) });
    defer a.free(deepseek);
    var d = try std.json.parseFromSlice(std.json.Value, a, deepseek, .{});
    defer d.deinit();
    try std.testing.expect(pretokenizer.matches(d.value));
    try std.testing.expectEqualStrings("deepseek", @tagName(arch_chat.pretokStyleOf(d.value)));

    var other = try std.json.parseFromSlice(std.json.Value, a, "{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"ByteLevel\"}]}", .{});
    defer other.deinit();
    try std.testing.expect(!pretokenizer.matches(other.value));
}
