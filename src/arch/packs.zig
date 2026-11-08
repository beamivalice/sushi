//! The published packs, as each arch declares them in its `packs` slot (`product` and `aliases`). MLX-free: `cli.zig`,
//! the pull and list commands and the load refusals read the alias table from here.
const std = @import("std");
const registry = @import("registry.zig");

pub const Alias = struct {
    /// Short name before the ':', e.g. "qwen3.8-flash-next".
    name: []const u8,
    /// Tag after the ':'; empty = selectable only by full name:tag.
    tag: []const u8,
    repo: []const u8,
    /// Picked when the user gives the bare name with no tag.
    is_default: bool = false,
    /// Installed RAM, in GB, the README and docs/usage.md document for running the pack in memory.
    tier_gb: u32,
    /// Installed RAM, in GB, from which `sushi list` recommends the pack (the largest such pack of a
    /// family at or under the machine's RAM); 0 = never recommended.
    recommend_gb: u32 = 0,
    /// Its experts always stream from the SSD (DeepSeek): `tier_gb` is the memory it streams from, not "in memory".
    streams_experts: bool = false,
    /// The flags of the pack's recommended `sushi serve --model <pack>` in docs/usage.md.
    serve_flags: []const u8 = "",
};

/// The enabled archs in the order the arch gate lists them (`gate_rank`), which is the order `sushi list` shows packs in.
pub const gate_order = blk: {
    const tags = std.meta.tags(registry.Tag);
    var order: [tags.len]registry.Tag = undefined;
    for (tags, 0..) |t, i| order[i] = t;
    for (1..order.len) |i| {
        var j = i;
        while (j > 0 and registry.gateRank(order[j]) < registry.gateRank(order[j - 1])) : (j -= 1) {
            const a = order[j];
            order[j] = order[j - 1];
            order[j - 1] = a;
        }
    }
    break :blk order;
};

/// Every enabled arch's packs, in gate order.
pub const aliases = blk: {
    var out: []const Alias = &.{};
    for (gate_order) |t| out = out ++ @as([]const Alias, &registry.Mod(t).packs.aliases);
    const final = out[0..out.len].*;
    break :blk final;
};

/// "qwen4_exp (Qwen3.8-Flash-Next), mimo_v2 (MiMo-V2.6-Flash), and ...": what a build loads, for the load refusal.
pub const served_list = blk: {
    var text: []const u8 = "";
    for (gate_order, 0..) |t, i| {
        const sep = if (i == 0) "" else if (gate_order.len == 2) " and " else if (i + 1 == gate_order.len) ", and " else ", ";
        text = text ++ sep ++ registry.Mod(t).model_types[0] ++ " (" ++ registry.Mod(t).packs.product ++ ")";
    }
    break :blk text;
};

/// "qwen4_exp, mimo_v2, glm5_next or deepseek_v41": the types a refusal names where it asks for a supported checkpoint.
pub const served_types_or = blk: {
    var text: []const u8 = "";
    for (gate_order, 0..) |t, i| {
        const sep = if (i == 0) "" else if (i + 1 == gate_order.len) " or " else ", ";
        text = text ++ sep ++ registry.Mod(t).model_types[0];
    }
    break :blk text;
};

fn dflashTargets(comptime word: []const u8) []const u8 {
    comptime {
        var n = 0;
        for (gate_order) |t| {
            if (@hasDecl(registry.Mod(t).packs, "dflash_target")) n += 1;
        }
        var text: []const u8 = "";
        var i = 0;
        for (gate_order) |t| {
            if (!@hasDecl(registry.Mod(t).packs, "dflash_target")) continue;
            text = text ++ (if (i == 0) "" else if (n == 2) " " ++ word ++ " " else if (i + 1 == n) ", " ++ word ++ " " else ", ") ++ registry.Mod(t).packs.product;
            i += 1;
        }
        return text;
    }
}

/// "Qwen3.8-Flash-Next and GLM-5.3-Flash": the products of the built archs that bind a DFlash2 drafter (`dflash_target`).
pub const dflash_targets_and = dflashTargets("and");
pub const dflash_targets_or = dflashTargets("or");

test "the alias table lists every arch's packs, in the arch gate's order, with their tiers and serve flags" {
    if (registry.served_model_types.len != 4) return error.SkipZigTest;
    const Row = struct { name: []const u8, tag: []const u8, repo: []const u8, is_default: bool, tier_gb: u32, recommend_gb: u32, streams: bool, flags: []const u8 };
    const qwen_flags = "--mtp --kv-quant 8 --mtp-head-kv-quant --qwen-gdn 16 --ctx-size ";
    const rows = [_]Row{
        .{ .name = "qwen3.8-flash-next", .tag = "2bpw", .repo = "beamster/Qwen3.8-Flash-Next-Sushi-2bpw", .is_default = false, .tier_gb = 48, .recommend_gb = 32, .streams = false, .flags = qwen_flags ++ "131072 --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1" },
        .{ .name = "qwen3.8-flash-next", .tag = "2.75bpw", .repo = "beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw", .is_default = true, .tier_gb = 64, .recommend_gb = 64, .streams = false, .flags = qwen_flags ++ "250000 --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1" },
        .{ .name = "qwen3.8-flash-next", .tag = "4bpw", .repo = "beamster/Qwen3.8-Flash-Next-Sushi-4bpw", .is_default = false, .tier_gb = 96, .recommend_gb = 96, .streams = false, .flags = "--mtp --kv-quant 8 --mtp-head-kv-quant --ctx-size 500000 --prefill-chunk 2048 --max-tokens 64000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1" },
        .{ .name = "mimo-v2.6-flash", .tag = "2.3bpw", .repo = "beamster/MiMo-V2.6-Flash-Sushi-2.3bpw", .is_default = true, .tier_gb = 128, .recommend_gb = 128, .streams = false, .flags = "--ctx-size 1048576" },
        .{ .name = "glm-5.3-flash", .tag = "2bpw", .repo = "beamster/GLM-5.3-Flash-Sushi-2bpw", .is_default = false, .tier_gb = 96, .recommend_gb = 96, .streams = false, .flags = "" },
        .{ .name = "glm-5.3-flash", .tag = "2.4bpw", .repo = "beamster/GLM-5.3-Flash-Sushi-2.4bpw", .is_default = true, .tier_gb = 128, .recommend_gb = 128, .streams = false, .flags = "" },
        .{ .name = "deepseek-v4.1-flash", .tag = "1.75bpw", .repo = "beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw", .is_default = true, .tier_gb = 96, .recommend_gb = 96, .streams = true, .flags = "" },
    };
    try std.testing.expectEqual(rows.len, aliases.len);
    for (rows, aliases) |want, got| {
        try std.testing.expectEqualStrings(want.name, got.name);
        try std.testing.expectEqualStrings(want.tag, got.tag);
        try std.testing.expectEqualStrings(want.repo, got.repo);
        try std.testing.expectEqual(want.is_default, got.is_default);
        try std.testing.expectEqual(want.tier_gb, got.tier_gb);
        try std.testing.expectEqual(want.recommend_gb, got.recommend_gb);
        try std.testing.expectEqual(want.streams, got.streams_experts);
        try std.testing.expectEqualStrings(want.flags, got.serve_flags);
    }
}

test "every arch declares a product name and at least one pack" {
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        try std.testing.expect(registry.Mod(t).packs.product.len > 0);
        try std.testing.expect(registry.Mod(t).packs.aliases.len > 0);
    }
}
