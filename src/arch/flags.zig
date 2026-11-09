//! The launch flags an arch owns and what it says about the others, from the `flags` slot of each `arch.zig`
//! (`src/<arch>/<arch>_flags.zig`). The shared parser, `--help` and the load refusals ask here and never name an arch.
//!
//! A flags module declares what it has, each declaration optional:
//! - `options`: `[]const Option`, the flags only this arch takes;
//! - `help`: `[]const HelpChunk`, their `--help` text, placed before the shared flag `before` names;
//! - `think_rank`, `think_help`, `think_refusal`: the arch's words for `--think`;
//! - `command`: a subcommand (`name`, `runs_gpu`, `help`, `run`);
//! - `refusal(use)`: a launch flag this arch cannot act on;
//! - `drop_expert_elsewhere`: why `--drop-expert` has no effect on the other archs;
//! - `mtp_off_line`, `kvQuantRequestRefusal(kv)`, `kv_quant_choices` (the spellings its cache stores),
//!   `settings_kv_quant_ignored`, `settingsMtpIgnoredLine(config)`: the arch's lines and refusals where it differs
//!   from the shared rule;
//! - `serve` (`install`, `uninstall`) and `warm` (`bytes`, `total`): the MLX-side hooks `arch/serve.zig` reaches.
const std = @import("std");
const model = @import("../model.zig");
const model_settings = @import("../model_settings.zig");
const kv_quant = @import("../kv_quant.zig");
const registry = @import("registry.zig");
const arch_config = @import("config.zig");
const admission = @import("admission.zig");
const flag_value = @import("../flag_value.zig");

/// The launch flags a load can be refused for, as the process was started.
pub const LaunchFlagUse = struct {
    kv_quant: bool = false,
    qwen_gdn_16: bool = false,
    mtp_head_kv_quant: bool = false,
    drop_expert: bool = false,
    ssd_prefetch: bool = false,
    mtp_typical: bool = false,
    mtp_tokenv3: bool = false,
    mtp_greedy_tail: bool = false,
    /// `--mtp-min-depth` / `--mtp-max-depth` as given; 0 = not given.
    mtp_min_depth: u32 = 0,
    mtp_max_depth: u32 = 0,
};

/// `--help` text for one arch-only flag, written as the lines it prints, inserted before the line of shared flag `before`.
pub const HelpChunk = struct { before: []const u8, text: []const u8 };

pub const Option = struct {
    name: []const u8,
    takes_value: bool,
    /// Records the flag's value; false = the value is not one the flag takes.
    apply: *const fn (value: []const u8) bool,
    /// Said before `; got '<value>'` when `apply` refuses.
    expected: []const u8 = "",
    /// What this flag does on `config`, for the boot log; null when the flag was not given.
    boot_line: ?*const fn (config: *const model.ModelConfig, use: LaunchFlagUse) ?[]const u8 = null,
};

pub const Command = struct {
    name: []const u8,
    /// False for a command that never touches MLX, so it does not wake Metal.
    runs_gpu: bool,
};

fn hasDecl(comptime tag: registry.Tag, comptime name: []const u8) bool {
    return @hasDecl(registry.Mod(tag).flags, name);
}

fn flagsOf(comptime tag: registry.Tag) type {
    return registry.Mod(tag).flags;
}

/// Every enabled arch's options, in registry order.
pub const options = blk: {
    var out: []const Option = &.{};
    for (std.meta.tags(registry.Tag)) |t| {
        if (hasDecl(t, "options")) out = out ++ @as([]const Option, &flagsOf(t).options);
    }
    const final = out[0..out.len].*;
    break :blk final;
};

pub const Parsed = union(enum) {
    applied,
    refused: struct { name: []const u8, expected: []const u8, got: []const u8 },
};

/// Parses `args[i.*]` as an arch-owned flag and steps `i` past its value. Null when no enabled arch owns it, or when it
/// needs a value and has none (the caller then reports it as unrecognized).
pub fn parseOption(args: []const []const u8, i: *usize) ?Parsed {
    inline for (options) |o| {
        if (std.mem.eql(u8, args[i.*], o.name)) {
            if (o.takes_value and !flag_value.follows(args, i.*)) return null;
            const value: []const u8 = if (o.takes_value) args[i.* + 1] else "";
            if (o.takes_value) i.* += 1;
            if (o.apply(value)) return .applied;
            return .{ .refused = .{ .name = o.name, .expected = o.expected, .got = value } };
        }
    }
    return null;
}

/// The boot-log line of the arch flag `name` on `config`; null when the flag is not given or nothing needs saying.
pub fn bootLineFor(name: []const u8, config: *const model.ModelConfig, use: LaunchFlagUse) ?[]const u8 {
    inline for (options) |o| {
        if (std.mem.eql(u8, name, o.name)) return if (o.boot_line) |f| f(config, use) else null;
    }
    return null;
}

/// Each arch flag's boot line for `config`, in the order the flags are declared.
pub fn bootLines(config: *const model.ModelConfig, use: LaunchFlagUse, out: *[options.len]?[]const u8) []const ?[]const u8 {
    inline for (options, 0..) |o, n| out[n] = if (o.boot_line) |f| f(config, use) else null;
    return out[0..options.len];
}

/// A launch flag with nothing to act on in this arch, named with the reason; null when every flag given applies.
/// An explicit flag is refused rather than ignored, before the weights load.
pub fn archFlagRefusal(config: *const model.ModelConfig, use: LaunchFlagUse) ?[]const u8 {
    const traits = admission.expertStream(config);
    if (use.drop_expert and !traits.drop_expert) return dropExpertElsewhere();
    if (use.ssd_prefetch and !traits.prefetch) return "--ssd-prefetch has no effect on this model: it has no expert prefetcher; drop the flag";
    const tag = arch_config.tagOf(config) orelse return null;
    switch (tag) {
        inline else => |t| return if (comptime hasDecl(t, "refusal")) flagsOf(t).refusal(config, use) else null,
    }
}

fn dropExpertElsewhere() []const u8 {
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime hasDecl(t, "drop_expert_elsewhere")) return flagsOf(t).drop_expert_elsewhere;
    }
    return "--drop-expert has no effect on this model; drop the flag";
}

/// The facts `model_settings.applies` decides a key's reach from, so the settings API and the loader's ignored-key lines agree.
pub fn settingsFit(config: *const model.ModelConfig) model_settings.Fit {
    const unserved = arch_config.fact(config, "mtp_unserved");
    return .{
        .arch = config.model_type,
        .streams_experts = config.supportsExpertStreaming(),
        .has_vision = config.has_vision,
        .serves_mtp = !unserved,
        .streaming_required = config.expertStreamingRequired(),
        .head_streams = admission.streamedMtpHead(config, config.expert_layout),
        .stq_experts = admission.expertStream(config).stq_experts,
        .prefetch = admission.expertStream(config).prefetch,
        .kv_choices = kvQuantChoices(config),
        .context_cap = config.contextCap(),
        .reads_yarn = arch_config.fact(config, "reads_yarn"),
    };
}

/// The `kv_quant` spellings this arch's cache stores when it stores fewer than all of them; empty = every width.
fn kvQuantChoices(config: *const model.ModelConfig) []const []const u8 {
    const tag = arch_config.tagOf(config) orelse return &.{};
    switch (tag) {
        inline else => |t| return if (comptime hasDecl(t, "kv_quant_choices")) &flagsOf(t).kv_quant_choices else &.{},
    }
}

/// A model-settings.json `ssd_prefetch` on an arch with no expert prefetcher is ignored with this one line, as `kv_quant` is.
/// A pack of the arch that has one but streams no STQ experts says so at its own load.
pub fn settingsSsdPrefetchIgnoredLine(config: *const model.ModelConfig) ?[]const u8 {
    if (admission.expertStream(config).prefetch or model_settings.applies("ssd_prefetch", settingsFit(config)) or config.ssd_prefetch_override == null) return null;
    return "model-settings.json ssd_prefetch is ignored: this model has no expert prefetcher";
}

/// A model-settings.json `kv_quant` on an arch with no KV cache is ignored with this one line, never a refusal:
/// the file outlives the flags that wrote it, and only an explicit launch flag is a request.
pub fn settingsKvQuantIgnoredLine(config: *const model.ModelConfig) ?[]const u8 {
    if (model_settings.applies("kv_quant", settingsFit(config)) or config.kv_quant_override == null) return null;
    const tag = arch_config.tagOf(config) orelse return null;
    switch (tag) {
        inline else => |t| return if (comptime hasDecl(t, "settings_kv_quant_ignored")) flagsOf(t).settings_kv_quant_ignored else null,
    }
}

/// A model-settings.json `mtp_acceptance` or `mtp_greedy_tail` where no MTP head runs is ignored with this one line, as `kv_quant` is.
pub fn settingsMtpIgnoredLine(config: *const model.ModelConfig) ?[]const u8 {
    if (model_settings.applies("mtp_acceptance", settingsFit(config))) return null;
    if (arch_config.tagOf(config)) |tag| switch (tag) {
        inline else => |t| if (comptime hasDecl(t, "settingsMtpIgnoredLine")) return flagsOf(t).settingsMtpIgnoredLine(config),
    };
    if (config.mtp_acceptance_override != null or config.mtp_greedy_tail_override != null) return "model-settings.json mtp_acceptance and mtp_greedy_tail are ignored: this load runs no MTP head";
    return null;
}

/// The boot line saying the arch serves no MTP head at all; null where it may.
pub fn mtpOffLine(config: *const model.ModelConfig) ?[]const u8 {
    const tag = arch_config.tagOf(config) orelse return null;
    switch (tag) {
        inline else => |t| return if (comptime hasDecl(t, "mtp_off_line")) flagsOf(t).mtp_off_line else null,
    }
}

/// Why a request's KV scheme is refused on this arch's cache; null when it stores it.
pub fn kvQuantRequestRefusal(config: *const model.ModelConfig, kv: kv_quant.KVQuantConfig) ?[]const u8 {
    const tag = arch_config.tagOf(config) orelse return null;
    switch (tag) {
        inline else => |t| return if (comptime hasDecl(t, "kvQuantRequestRefusal")) flagsOf(t).kvQuantRequestRefusal(kv) else null,
    }
}

/// The subcommands the enabled archs add, in registry order.
pub const commands = blk: {
    var out: []const Command = &.{};
    for (std.meta.tags(registry.Tag)) |t| {
        if (hasDecl(t, "command")) out = out ++ &[_]Command{.{ .name = flagsOf(t).command.name, .runs_gpu = flagsOf(t).command.runs_gpu }};
    }
    const final = out[0..out.len].*;
    break :blk final;
};

/// Runs the arch subcommand `name` with the arguments after it; false when no enabled arch has it.
pub fn runCommand(allocator: std.mem.Allocator, io: std.Io, name: []const u8, rest: []const []const u8) !bool {
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime hasDecl(t, "command")) {
            if (std.mem.eql(u8, name, flagsOf(t).command.name)) {
                try flagsOf(t).command.run(allocator, io, rest);
                return true;
            }
        }
    }
    return false;
}

/// Whether the command named `name` (an arch's or not) keeps clear of MLX.
pub fn commandSkipsGpu(name: []const u8) bool {
    inline for (commands) |c| if (!c.runs_gpu and std.mem.eql(u8, name, c.name)) return true;
    return false;
}

const testing = std.testing;

fn built(model_type: []const u8) bool {
    return registry.ofModelType(model_type) != null;
}

test "an arch-owned flag parses its value, steps past it, and names what it expected when it refuses" {
    if (!built("qwen4_exp")) return error.SkipZigTest;
    const saved = model.qwen_gdn_f32;
    defer model.qwen_gdn_f32 = saved;
    var i: usize = 0;
    try testing.expectEqual(Parsed.applied, parseOption(&.{ "--qwen-gdn", "16" }, &i).?);
    try testing.expectEqual(@as(usize, 1), i);
    try testing.expect(!model.qwen_gdn_f32);
    i = 0;
    try testing.expectEqual(Parsed.applied, parseOption(&.{ "--qwen-gdn", "32" }, &i).?);
    try testing.expect(model.qwen_gdn_f32);
    i = 0;
    const refused = parseOption(&.{ "--qwen-gdn", "8" }, &i).?.refused;
    try testing.expectEqualStrings("--qwen-gdn", refused.name);
    try testing.expectEqualStrings("expected 32 or 16", refused.expected);
    try testing.expectEqualStrings("8", refused.got);
    // A value-taking flag with no value left is not consumed: the caller reports it as unrecognized.
    i = 0;
    try testing.expect(parseOption(&.{"--qwen-gdn"}, &i) == null);
    i = 0;
    try testing.expect(parseOption(&.{ "--port", "1" }, &i) == null);
}

test "--drop-expert takes a multiple of 8 and --mtp-head-kv-quant takes no value" {
    if (built("deepseek_v41")) {
        const saved = model.drop_expert;
        defer model.drop_expert = saved;
        var i: usize = 0;
        try testing.expectEqual(Parsed.applied, parseOption(&.{ "--drop-expert", "16" }, &i).?);
        try testing.expectEqual(@as(u16, 16), model.drop_expert);
        i = 0;
        const refused = parseOption(&.{ "--drop-expert", "7" }, &i).?.refused;
        try testing.expectEqualStrings("expected a multiple of 8 routed experts per layer (0 = off)", refused.expected);
        i = 0;
        try testing.expect(parseOption(&.{"--drop-expert"}, &i) == null);
    }
    if (built("qwen4_exp")) {
        const trunk = @import("../trunk.zig");
        const saved = trunk.Trunk.mtp_head_kv_quant_flag;
        defer trunk.Trunk.mtp_head_kv_quant_flag = saved;
        trunk.Trunk.mtp_head_kv_quant_flag = false;
        var i: usize = 0;
        try testing.expectEqual(Parsed.applied, parseOption(&.{ "--mtp-head-kv-quant", "--next" }, &i).?);
        try testing.expectEqual(@as(usize, 0), i);
        try testing.expect(trunk.Trunk.mtp_head_kv_quant_flag);
    }
}

test "--mtp-head-kv-quant is reported per arch" {
    if (!built("qwen4_exp")) return error.SkipZigTest;
    const qwen4 = model.ModelConfig{ .model_type = "qwen4_exp" };
    const mimo = model.ModelConfig{ .model_type = "mimo_v2" };
    const given = LaunchFlagUse{ .mtp_head_kv_quant = true };
    try testing.expect(bootLineFor("--mtp-head-kv-quant", &qwen4, .{}) == null);
    try testing.expect(bootLineFor("--mtp-head-kv-quant", &mimo, .{}) == null);
    try testing.expectEqualStrings("MTP head KV: following --kv-quant (--mtp-head-kv-quant)", bootLineFor("--mtp-head-kv-quant", &qwen4, given).?);
    // MiMo's heads keep their own dense window; the flag never reaches them.
    if (built("mimo_v2")) try testing.expectEqualStrings("MTP head KV: --mtp-head-kv-quant has no effect on mimo_v2 (its heads keep a dense sliding window)", bootLineFor("--mtp-head-kv-quant", &mimo, given).?);
}

test "the GDN-state flag is reported per arch, and has no effect where there is no GDN arm" {
    if (!built("qwen4_exp")) return error.SkipZigTest;
    const qwen4 = model.ModelConfig{ .model_type = "qwen4_exp" };
    const glm = model.ModelConfig{ .model_type = "glm5_next" };
    const mimo = model.ModelConfig{ .model_type = "mimo_v2" };
    const given = LaunchFlagUse{ .qwen_gdn_16 = true };
    try testing.expect(bootLineFor("--qwen-gdn", &qwen4, .{}) == null);
    try testing.expect(bootLineFor("--qwen-gdn", &glm, .{}) == null);
    try testing.expectEqualStrings("GDN state: bf16 between tokens (--qwen-gdn 16; f32 is the default; the decay gate is f32 either way)", bootLineFor("--qwen-gdn", &qwen4, given).?);
    // GLM's recurrent state is KDA, not GDN: the flag names a layer it does not have.
    if (built("glm5_next")) try testing.expectEqualStrings("GDN state: --qwen-gdn 16 has no effect on GLM (its KDA state is f32 already)", bootLineFor("--qwen-gdn", &glm, given).?);
    if (built("mimo_v2")) try testing.expectEqualStrings("GDN state: --qwen-gdn 16 has no effect on mimo_v2 (no GatedDeltaNet layers)", bootLineFor("--qwen-gdn", &mimo, given).?);
}

test "deepseek_v41 refuses the launch flags it cannot act on, and ignores a model-settings kv_quant with one line" {
    if (!built("deepseek_v41")) return error.SkipZigTest;
    var dsv41 = model.ModelConfig{ .model_type = "deepseek_v41" };
    const qwen4 = model.ModelConfig{ .model_type = "qwen4_exp" };
    try testing.expect(archFlagRefusal(&dsv41, .{}) == null);
    const all = LaunchFlagUse{ .kv_quant = true, .qwen_gdn_16 = true, .mtp_head_kv_quant = true, .mtp_typical = true, .mtp_tokenv3 = true, .mtp_greedy_tail = true };
    try testing.expect(archFlagRefusal(&qwen4, all) == null);
    for ([_]struct { use: LaunchFlagUse, why: []const u8 }{
        .{ .use = .{ .kv_quant = true }, .why = "--kv-quant has no effect on deepseek_v41: its compressed state is stored as the QAT round trips left it, so there is no KV cache to quantize; drop the flag" },
        .{ .use = .{ .qwen_gdn_16 = true }, .why = "--qwen-gdn has no effect on deepseek_v41: it has no GatedDeltaNet layers; drop the flag" },
        .{ .use = .{ .mtp_head_kv_quant = true }, .why = "--mtp-head-kv-quant has no effect on deepseek_v41: its DSpark stages keep their own windows; drop the flag" },
        .{ .use = .{ .mtp_typical = true }, .why = "--mtp-typical has no effect on deepseek_v41: DSpark accepts drafts by the exact rule only; drop the flag" },
        .{ .use = .{ .mtp_tokenv3 = true }, .why = "--mtp-tokenv3 has no effect on deepseek_v41: DSpark accepts drafts by the exact rule only; drop the flag" },
        .{ .use = .{ .mtp_greedy_tail = true }, .why = "--mtp-greedy-tail has no effect on deepseek_v41: DSpark drafts one block from the trunk's own state, with no MTP tail; drop the flag" },
    }) |c| {
        try testing.expectEqualStrings(c.why, archFlagRefusal(&dsv41, c.use) orelse return error.TestExpectedRefusal);
    }
    try testing.expect(settingsKvQuantIgnoredLine(&dsv41) == null);
    dsv41.kv_quant_override = kv_quant.KVQuantConfig.affine(8);
    try testing.expectEqualStrings("model-settings.json kv_quant is ignored on deepseek_v41 (no KV cache to quantize)", settingsKvQuantIgnoredLine(&dsv41).?);
    try testing.expect(settingsKvQuantIgnoredLine(&qwen4) == null);
}

test "deepseek_v41 ignores a model-settings mtp_acceptance or mtp_greedy_tail with one line naming the keys set" {
    if (!built("deepseek_v41")) return error.SkipZigTest;
    var dsv41 = model.ModelConfig{ .model_type = "deepseek_v41" };
    var qwen4 = model.ModelConfig{ .model_type = "qwen4_exp" };
    try testing.expect(settingsMtpIgnoredLine(&dsv41) == null);
    dsv41.mtp_acceptance_override = .{ .typical = .{ .delta = 0.3 } };
    try testing.expectEqualStrings("model-settings.json mtp_acceptance is ignored on deepseek_v41 (DSpark accepts drafts by the exact rule only)", settingsMtpIgnoredLine(&dsv41).?);
    dsv41.mtp_acceptance_override = null;
    dsv41.mtp_greedy_tail_override = true;
    try testing.expectEqualStrings("model-settings.json mtp_greedy_tail is ignored on deepseek_v41 (DSpark has no MTP tail)", settingsMtpIgnoredLine(&dsv41).?);
    dsv41.mtp_acceptance_override = .{ .tokenv3 = 0.95 };
    try testing.expectEqualStrings("model-settings.json mtp_acceptance and mtp_greedy_tail are ignored on deepseek_v41 (DSpark accepts drafts by the exact rule only, with no MTP tail)", settingsMtpIgnoredLine(&dsv41).?);
    qwen4.mtp_acceptance_override = .{ .tokenv3 = 0.95 };
    qwen4.mtp_greedy_tail_override = true;
    try testing.expect(settingsMtpIgnoredLine(&qwen4) == null);

    if (!built("glm5_next")) return;
    var glm = model.ModelConfig{ .model_type = "glm5_next" };
    try testing.expect(settingsMtpIgnoredLine(&glm) == null);
    glm.mtp_greedy_tail_override = true;
    try testing.expectEqualStrings("model-settings.json mtp_acceptance and mtp_greedy_tail are ignored: this load runs no MTP head", settingsMtpIgnoredLine(&glm).?);
}

test "settingsFit: a latent-cache arch stores only the latent widths and serves no MTP head; the cap is the model's window" {
    if (!built("glm5_next") or !built("deepseek_v41")) return error.SkipZigTest;
    var glm = model.ModelConfig{ .model_type = "glm5_next", .max_position_embeddings = 131072 };
    const fit = settingsFit(&glm);
    try testing.expect(fit.kv_choices.len != 0 and !fit.serves_mtp);
    try testing.expectEqual(@as(u32, 131072), fit.context_cap);
    try testing.expect(!model_settings.choiceAllowed("kv_quant", "4", fit));
    try testing.expect(model_settings.choiceAllowed("kv_quant", "6", fit));
    const qwen4 = model.ModelConfig{ .model_type = "qwen4_exp" };
    try testing.expect(settingsFit(&qwen4).kv_choices.len == 0 and settingsFit(&qwen4).serves_mtp);
    // A DSpark pack without STQ experts has no prefetcher, and says so at its own load.
    var dsv41 = model.ModelConfig{ .model_type = "deepseek_v41", .ssd_prefetch_override = 30 };
    try testing.expect(!model_settings.applies("ssd_prefetch", settingsFit(&dsv41)));
    try testing.expect(settingsSsdPrefetchIgnoredLine(&dsv41) == null);
}

test "--ssd-prefetch is refused where there is no expert prefetcher, before the weights load, and its model setting is ignored there" {
    if (!built("deepseek_v41")) return error.SkipZigTest;
    const use = LaunchFlagUse{ .ssd_prefetch = true };
    var dsv41 = model.ModelConfig{ .model_type = "deepseek_v41" };
    dsv41.ssd_prefetch_override = 30;
    try testing.expect(archFlagRefusal(&dsv41, use) == null);
    try testing.expect(settingsSsdPrefetchIgnoredLine(&dsv41) == null);
    for ([_][]const u8{ "qwen4_exp", "mimo_v2", "glm5_next" }) |arch| {
        if (!built(arch)) continue;
        var cfg = model.ModelConfig{ .model_type = arch };
        try testing.expectEqualStrings("--ssd-prefetch has no effect on this model: it has no expert prefetcher; drop the flag", archFlagRefusal(&cfg, use) orelse return error.TestExpectedRefusal);
        try testing.expect(archFlagRefusal(&cfg, .{}) == null);
        try testing.expect(settingsSsdPrefetchIgnoredLine(&cfg) == null);
        cfg.ssd_prefetch_override = 30;
        try testing.expectEqualStrings("model-settings.json ssd_prefetch is ignored: this model has no expert prefetcher", settingsSsdPrefetchIgnoredLine(&cfg).?);
    }
}

test "--drop-expert is refused on every arch but deepseek_v41, before the weights load" {
    if (!built("deepseek_v41")) return error.SkipZigTest;
    const use = LaunchFlagUse{ .drop_expert = true };
    const dsv41 = model.ModelConfig{ .model_type = "deepseek_v41" };
    try testing.expect(archFlagRefusal(&dsv41, use) == null);
    for ([_][]const u8{ "qwen4_exp", "mimo_v2", "glm5_next" }) |arch| {
        if (!built(arch)) continue;
        const cfg = model.ModelConfig{ .model_type = arch };
        try testing.expectEqualStrings("--drop-expert has no effect on this model: only deepseek_v41 packs carry an expert usage list; drop the flag", archFlagRefusal(&cfg, use) orelse return error.TestExpectedRefusal);
        try testing.expect(archFlagRefusal(&cfg, .{}) == null);
    }
}

test "a latent KV cache refuses a request width it does not store, and an arch without an MTP head says so" {
    if (!built("glm5_next")) return error.SkipZigTest;
    const glm = model.ModelConfig{ .model_type = "glm5_next" };
    const qwen4 = model.ModelConfig{ .model_type = "qwen4_exp" };
    try testing.expectEqualStrings("GLM native MLA cache supports kv_quant 8, 6 or off.", kvQuantRequestRefusal(&glm, kv_quant.KVQuantConfig.affine(4)).?);
    try testing.expect(kvQuantRequestRefusal(&glm, kv_quant.KVQuantConfig.affine(8)) == null);
    try testing.expect(kvQuantRequestRefusal(&qwen4, kv_quant.KVQuantConfig.affine(4)) == null);
    try testing.expectEqualStrings("MTP: off (GLM native serving has no integrated MTP head)", mtpOffLine(&glm).?);
    try testing.expect(mtpOffLine(&qwen4) == null);
}
