//! Hy4-Flash's chat face: the effort words its template takes and its pre-tokenizer (`arch/chat.zig` reads it).
const std = @import("std");
const eff = @import("../effort.zig");
const arch_chat = @import("../arch/chat.zig");

pub const model_types = [_][]const u8{"hy_v4_flash"};

pub const fixtures = @import("hy4_fixtures.zig");

/// The release's tokenizer.json spells DeepSeek-V4.x's pre-tokenizer exactly (its three Splits and a ByteLevel that does
/// not split; no normalizer), so it runs DeepSeek's grammar, shared in `src/tokenizer_deepseek.zig`.
pub const pretokenizer = @import("../tokenizer_deepseek.zig").face;

/// The template takes `no_think`, `high` and `max`; low and medium think at `high` under a budget (owner, 2026-11-18).
pub const effort: eff.Offer = .{ .table = eff.makeSpec("hy_v4_flash", "Hy4", &.{ .high, .max }, &.{}, true, .{ .default = .high, .aliases = &.{ .{ .word = .minimal, .takes = .low }, .{ .word = .on, .takes = .high } }, .rows = .{
    .{ .offered = true },
    .{ .offered = true, .sends = .{ .word = .high }, .cap = 2048 },
    .{ .offered = true, .sends = .{ .word = .high }, .cap = 16384 },
    .{ .offered = true, .sends = .{ .word = .high } },
    .{ .offered = true, .sends = .{ .word = .max } },
    .{ .offered = true, .sends = .{ .word = .max } },
} }) };

test "Hy4's effort table: off, low and medium under budgets at high, high, and xhigh and max at max" {
    const testing = @import("std").testing;
    const t = effort.table.table;
    try testing.expectEqual(eff.Effort.high, t.default);
    try testing.expect(t.row(.off).offered);
    try testing.expectEqualDeep(eff.Row{ .offered = true, .sends = .{ .word = .high }, .cap = 2048 }, t.row(.low));
    try testing.expectEqualDeep(eff.Row{ .offered = true, .sends = .{ .word = .high }, .cap = 16384 }, t.row(.medium));
    try testing.expectEqualDeep(eff.Row{ .offered = true, .sends = .{ .word = .high } }, t.row(.high));
    try testing.expectEqualDeep(eff.Row{ .offered = true, .sends = .{ .word = .max } }, t.row(.xhigh));
    try testing.expectEqualDeep(eff.Row{ .offered = true, .sends = .{ .word = .max } }, t.row(.max));
}

test "Hy4's effort words: each Sushi word a request may name is served at the template word and budget the owner ruled" {
    // docs/hy4-flash-arch.md, "Sushi's effort words": minimal takes low's row, on takes high's, max takes xhigh's.
    const testing = std.testing;
    const arms = eff.effortArms("hy_v4_flash").?;
    const Want = struct { word: []const u8, sends: eff.Sends, budget: ?i32 };
    const cases = [_]Want{
        .{ .word = "off", .sends = .none, .budget = null },
        .{ .word = "none", .sends = .none, .budget = null },
        .{ .word = "minimal", .sends = .{ .word = .high }, .budget = 2048 },
        .{ .word = "low", .sends = .{ .word = .high }, .budget = 2048 },
        .{ .word = "medium", .sends = .{ .word = .high }, .budget = 16384 },
        .{ .word = "on", .sends = .{ .word = .high }, .budget = null },
        .{ .word = "high", .sends = .{ .word = .high }, .budget = null },
        .{ .word = "xhigh", .sends = .{ .word = .max }, .budget = null },
        .{ .word = "max", .sends = .{ .word = .max }, .budget = null },
    };
    for (cases) |c| {
        const arm = eff.armForWord(arms, c.word) orelse {
            std.debug.print("refused: {s}\n", .{c.word});
            return error.TestUnexpectedResult;
        };
        testing.expectEqualDeep(c.sends, arm.sends) catch |e| {
            std.debug.print("{s}: sends {any}\n", .{ c.word, arm.sends });
            return e;
        };
        testing.expectEqual(c.budget, arm.budget) catch |e| {
            std.debug.print("{s}: budget {any}\n", .{ c.word, arm.budget });
            return e;
        };
    }
}

test "Hy4's template teaches GLM's arg_key calls under its :opensource spelling" {
    const chat = @import("../chat.zig");
    const a = std.testing.allocator;
    const config = chat.ChatConfig{ .chat_template = release_template, .bos_token = "<\u{ff5c}hy_start:opensource\u{ff5c}>", .eos_token = "<\u{ff5c}hy_end:opensource\u{ff5c}>", .add_bos_token = false, .allocator = a, .model_type = "hy_v4_flash" };
    try std.testing.expectEqual(@as(?@import("../tool_calls.zig").Dialect, .arg_key), try chat.callDialect(a, &config));
}

test "Hy4's tokenizer.json pre-tokenizer is read as DeepSeek's grammar" {
    const a = std.testing.allocator;
    var p = try std.json.parseFromSlice(std.json.Value, a, @embedFile("fixtures/hy4_pre_tokenizer.json"), .{});
    defer p.deinit();
    try std.testing.expect(pretokenizer.matches(p.value));
    try std.testing.expectEqualStrings("deepseek", @tagName(arch_chat.pretokStyleOf(p.value)));
}

/// The release's `chat_template.jinja`, unmodified (NOTICE names it).
pub const release_template = @embedFile("fixtures/hy4_chat_template.jinja");

fn renderRelease(thinking: bool, effort_word: ?[]const u8) ![]const u8 {
    const chat = @import("../chat.zig");
    const a = std.testing.allocator;
    const config = chat.ChatConfig{ .chat_template = release_template, .bos_token = "<\u{ff5c}hy_start:opensource\u{ff5c}>", .eos_token = "<\u{ff5c}hy_end:opensource\u{ff5c}>", .add_bos_token = false, .allocator = a, .model_type = "hy_v4_flash" };
    const messages = [_]chat.Message{.{ .role = "user", .content = "hi" }};
    return chat.renderChatTemplate(a, &messages, &config, null, null, thinking, effort_word, false);
}

test "Hy4's template renders as transformers' apply_chat_template does: max for xhigh and max, high, and no_think when off" {
    // transformers c2ec8505, `apply_chat_template([{"role":"user","content":"hi"}], add_generation_prompt=True,
    // reasoning_effort=<word>)` on the release's tokenizer and template.
    const a = std.testing.allocator;
    const head = "<\u{ff5c}hy_start:opensource\u{ff5c}>system<\u{ff5c}hy_middle:opensource\u{ff5c}><\u{ff5c}reasoning_mode:opensource\u{ff5c}>reasoning_effort:";
    const tail = "<\u{ff5c}hy_end:opensource\u{ff5c}><\u{ff5c}hy_start:opensource\u{ff5c}>user<\u{ff5c}hy_middle:opensource\u{ff5c}>hi<\u{ff5c}hy_end:opensource\u{ff5c}>" ++
        "<\u{ff5c}hy_start:opensource\u{ff5c}>assistant<\u{ff5c}hy_middle:opensource\u{ff5c}><think:opensource>";
    const cases = [_]struct { thinking: bool, sends: ?[]const u8, want: []const u8 }{
        .{ .thinking = true, .sends = "max", .want = head ++ "max" ++ tail },
        .{ .thinking = true, .sends = "high", .want = head ++ "high" ++ tail },
        .{ .thinking = true, .sends = null, .want = head ++ "high" ++ tail },
        .{ .thinking = false, .sends = null, .want = head ++ "no_think" ++ tail ++ "</think:opensource>" },
    };
    for (cases) |c| {
        const out = renderRelease(c.thinking, c.sends) catch |e| {
            std.debug.print("refused: {s}\n", .{@import("../chat.zig").templateRefusal()});
            return e;
        };
        defer a.free(out);
        try std.testing.expectEqualStrings(c.want, out);
    }
}

test "Hy4's template renders each reference case as transformers does: system, history thinking, tools and results" {
    // tests/dump_hy4_template.py, transformers c2ec8505: messages, tools, thinking, effort and preserve per case.
    const chat = @import("../chat.zig");
    const a = std.testing.allocator;
    const Call = struct { id: []const u8, name: []const u8, arguments: []const u8 };
    const Msg = struct { role: []const u8, content: []const u8, reasoning_content: ?[]const u8 = null, tool_calls: ?[]const Call = null, tool_call_id: ?[]const u8 = null };
    const Case = struct { label: []const u8, messages: []const Msg, tools: ?std.json.Value = null, thinking: bool, sends: ?[]const u8, preserve: bool, want: []const u8 };
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    const doc = try std.json.parseFromSliceLeaky(struct { generator: []const u8, cases: []const Case }, ar, @embedFile("fixtures/hy4_template_cases.json"), .{ .ignore_unknown_fields = true });
    var failed: usize = 0;
    for (doc.cases) |c| {
        const msgs = try ar.alloc(chat.Message, c.messages.len);
        for (c.messages, msgs) |m, *out| {
            var calls: ?[]chat.ToolCall = null;
            if (m.tool_calls) |tc| {
                calls = try ar.alloc(chat.ToolCall, tc.len);
                for (tc, calls.?) |t, *o| o.* = .{ .id = t.id, .name = t.name, .arguments = t.arguments };
            }
            out.* = .{ .role = m.role, .content = m.content, .reasoning_content = m.reasoning_content, .tool_calls = calls, .tool_call_id = m.tool_call_id };
        }
        const tools: ?[]const u8 = if (c.tools) |t| try std.json.Stringify.valueAlloc(ar, t, .{}) else null;
        const config = chat.ChatConfig{ .chat_template = release_template, .bos_token = "<\u{ff5c}hy_start:opensource\u{ff5c}>", .eos_token = "<\u{ff5c}hy_end:opensource\u{ff5c}>", .add_bos_token = false, .allocator = a, .model_type = "hy_v4_flash", .preserve_thinking = if (c.preserve) true else null };
        const got = chat.renderChatTemplate(a, msgs, &config, tools, null, c.thinking, c.sends, false) catch |e| {
            std.debug.print("[{s}] refused: {t} {s}\n", .{ c.label, e, chat.templateRefusal() });
            failed += 1;
            continue;
        };
        defer a.free(got);
        if (!std.mem.eql(u8, got, c.want)) {
            const at = std.mem.indexOfDiff(u8, got, c.want) orelse 0;
            std.debug.print("[{s}] differs at byte {d}:\n  want: {s}\n  got:  {s}\n", .{ c.label, at, c.want[at..@min(c.want.len, at + 160)], got[at..@min(got.len, at + 160)] });
            failed += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}
