//! The launch flags an arch owns and what it says about the others, from the `flags` slot of each `arch.zig`
//! (`src/<arch>/<arch>_flags.zig`). The shared parser, `--help` and the load refusals ask here and never name an arch.
//!
//! A flags module declares what it has, each declaration optional:
//! - `options`: `[]const Option`, the flags only this arch takes;
//! - `help`: `[]const HelpChunk`, their `--help` text, placed before the shared flag `before` names;
//! - `think_rank`, `think_help`, `think_refusal`: the arch's words for `--think`;
//! - `command`: a subcommand (`name`, `runs_gpu`, `help`, `run`);
//! - `refusal(use)`: a launch flag this arch cannot act on;
//! - `mtp_off_line`, `kvQuantRequestRefusal(kv)`, `kv_quant_choices` (the spellings its cache stores),
//!   `settings_kv_quant_ignored`: the arch's lines and refusals where it differs from the shared rule;
//! - `serve` (`install`, `uninstall`) and `warm` (`bytes`, `total`): the MLX-side hooks `arch/serve.zig` reaches.
const std = @import("std");
const model = @import("../model.zig");
const model_settings = @import("../model_settings.zig");
const kv_quant = @import("../kv_quant.zig");
const registry = @import("registry.zig");
const arch_config = @import("config.zig");
const admission = @import("admission.zig");
const flag_value = @import("../flag_value.zig");
const settings_resolve = @import("../settings_resolve.zig");
const arch_sched = @import("sched.zig");
const sched_policy = @import("../sched_policy.zig");

/// The launch flags a load can be refused for, as the process was started.
pub const LaunchFlagUse = struct {
    kv_quant: bool = false,
    qwen_gdn_16: bool = false,
    mtp_head_kv_quant: bool = false,
    drop_expert: bool = false,
    ssd_prefetch: bool = false,
    /// `--drafter`: a DFlash2 assistant beside the model.
    drafter: bool = false,
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

/// The opt-out flag of arch `t`: its loads decode on MLX; another arch's model loads as it would without it.
fn coolerOffOption(comptime t: registry.Tag, comptime o: arch_config.CoolerOptOut) Option {
    return .{ .name = o.flag, .takes_value = false, .apply = struct {
        fn f(_: []const u8) bool {
            model.cooler_off.insert(t);
            return true;
        }
    }.f };
}

fn hasDecl(comptime tag: registry.Tag, comptime name: []const u8) bool {
    return @hasDecl(registry.Mod(tag).flags, name);
}

fn flagsOf(comptime tag: registry.Tag) type {
    return registry.Mod(tag).flags;
}

/// `--drop-expert`, which any arch whose streamed experts carry a usage list takes (`ExpertStreamTraits.drop_expert`).
const shared_options = [_]Option{
    .{ .name = "--drop-expert", .takes_value = true, .apply = setDropExpert, .expected = "expected a multiple of 8 routed experts per layer (0 = off)" },
    .{ .name = "--prefill-chunk-max", .takes_value = true, .apply = setChunkMax, .expected = chunk_expected },
    .{ .name = "--prefill-chunk-min", .takes_value = true, .apply = setChunkMin, .expected = chunk_expected },
};

const chunk_expected = "expected " ++ settings_resolve.static_chunk_rungs_text;

fn chunkRows(value: []const u8) ?u32 {
    const n = std.fmt.parseInt(u32, value, 10) catch return null;
    return if (settings_resolve.isStaticChunkRung(n)) n else null;
}

fn setChunkMax(value: []const u8) bool {
    settings_resolve.launch.prefill_chunk_max = chunkRows(value) orelse return false;
    return true;
}

fn setChunkMin(value: []const u8) bool {
    settings_resolve.launch.prefill_chunk_min = chunkRows(value) orelse return false;
    return true;
}

fn setDropExpert(value: []const u8) bool {
    model.drop_expert = @import("../expert_usage.zig").parseCount(value) orelse return false;
    return true;
}

/// The shared options, then every enabled arch's, in registry order, its Sushi Cooler opt-out first.
pub const options = blk: {
    var out: []const Option = &shared_options;
    for (std.meta.tags(registry.Tag)) |t| {
        if (arch_config.coolerOptOut(t)) |o| out = out ++ [_]Option{coolerOffOption(t, o)};
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
    if (use.drop_expert and !traits.drop_expert) return "--drop-expert has no effect on this model: only a quantized pack whose experts stream drops them; drop the flag";
    if (use.ssd_prefetch and !traits.prefetch) return "--ssd-prefetch has no effect on this model: it has no expert prefetcher; drop the flag";
    const tag = arch_config.tagOf(config) orelse return null;
    switch (tag) {
        inline else => |t| {
            if (comptime hasDecl(t, "refusal")) if (flagsOf(t).refusal(config, use)) |r| return r;
            return coolerRefusal(t, config, use);
        },
    }
}

/// Why a request carrying `what` (`logit_bias`, `think_penalty`) is refused by this config's arch; null when it serves it.
pub fn biasRefusal(buf: []u8, config: *const model.ModelConfig, what: []const u8) ?[]const u8 {
    if (!arch_config.fact(config, "bias_unserved")) return null;
    return std.fmt.bufPrint(buf, "{s} does not support {s}", .{ productOf(config), what }) catch what;
}

/// Why a load of `config` is refused for a think penalty set for the model itself (its saved setting) or a logit-bias
/// file its arch does not serve; null when it takes them. `--think-penalty` is a launch default and never refuses.
pub fn biasLoadRefusal(buf: []u8, config: *const model.ModelConfig) ?[]const u8 {
    if (!arch_config.fact(config, "bias_unserved")) return null;
    const product = productOf(config);
    if (config.think_penalty_override) |lambda| {
        if (lambda > 0) return std.fmt.bufPrint(buf, "{s} does not support think_penalty: remove it from this model's settings", .{product}) catch "think_penalty is not supported";
    }
    if (config.logit_bias_file_override != null) return std.fmt.bufPrint(buf, "{s} does not support logit_bias: remove logit_bias_file from this model's settings", .{product}) catch "logit_bias is not supported";
    return null;
}

fn productOf(config: *const model.ModelConfig) []const u8 {
    const tag = arch_config.tagOf(config) orelse return "this model";
    switch (tag) {
        inline else => |t| return registry.Mod(t).packs.product,
    }
}

/// What a Cooler load serves, whatever its arch: no DFlash2, which runs on MLX only. A flag that asks otherwise is
/// refused by name, with the arch's own way back to MLX.
const cooler_refused = [_]struct { field: []const u8, flag: []const u8, why: []const u8 }{
    .{ .field = "drafter", .flag = "--drafter", .why = "DFlash2 runs on MLX only" },
};

fn coolerRefusal(comptime t: registry.Tag, config: *const model.ModelConfig, use: LaunchFlagUse) ?[]const u8 {
    if (!@import("../cooler_side.zig").onEngine(config)) return null;
    const back = comptime if (arch_config.coolerOptOut(t)) |o| ", or pass " ++ o.flag else "";
    inline for (cooler_refused) |r| if (@field(use, r.field)) return r.flag ++ " is not served on Sushi Cooler: " ++ r.why ++ "; drop the flag" ++ back;
    return null;
}


const cooler_archs = blk: {
    var n: usize = 0;
    for (std.meta.tags(registry.Tag)) |t| {
        if (arch_config.coolerOptOut(t) != null) n += 1;
    }
    break :blk n;
};

/// For each Sushi Cooler opt-out flag given for another arch than the model's, the line its load says: the flag
/// keeps that arch's loads on MLX and leaves this one as it is.
pub fn coolerOffIgnoredLines(config: *const model.ModelConfig) [cooler_archs]?[]const u8 {
    var out: [cooler_archs]?[]const u8 = @splat(null);
    const own = arch_config.tagOf(config);
    var at: usize = 0;
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime arch_config.coolerOptOut(t)) |o| {
            if (model.cooler_off.contains(t) and own != t) out[at] = o.flag ++ " has no effect on this model: it keeps " ++ registry.Mod(t).model_types[0] ++ " loads on MLX";
            at += 1;
        }
    }
    return out;
}

/// The model setting that turns the model's arch off Sushi Cooler; null for an arch with no opt-out.
fn coolerSettingName(config: *const model.ModelConfig) ?[]const u8 {
    const tag = arch_config.tagOf(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime arch_config.coolerOptOut(t)) |o| o.setting else null,
    };
}

/// The model's own Sushi Cooler setting out of its settings row; null when unset or the arch has no opt-out.
pub fn coolerSettingOf(config: *const model.ModelConfig, o: model_settings.Override) ?bool {
    const tag = arch_config.tagOf(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime arch_config.coolerOptOut(t)) |opt| @field(o, opt.setting) else null,
    };
}

/// Sets the model's own Sushi Cooler setting in `o` to `v`; an arch with no opt-out has none to set.
pub fn putCoolerSetting(config: *const model.ModelConfig, o: *model_settings.Override, v: ?bool) void {
    const tag = arch_config.tagOf(config) orelse return;
    switch (tag) {
        inline else => |t| if (comptime arch_config.coolerOptOut(t)) |opt| {
            @field(o, opt.setting) = v;
        },
    }
}

/// A Sushi Cooler setting the row carries for another arch than the model's: its load names it as ignored.
pub fn strayCoolerSetting(config: *const model.ModelConfig, o: model_settings.Override) ?[]const u8 {
    const own = coolerSettingName(config);
    inline for (comptime std.meta.tags(registry.Tag)) |t| if (comptime arch_config.coolerOptOut(t)) |opt| {
        const mine = if (own) |n| std.mem.eql(u8, n, opt.setting) else false;
        if (!mine and @field(o, opt.setting) != null) return opt.setting;
    };
    return null;
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
        .cache_bias = admission.expertStream(config).cache_bias,
        .kv_choices = kvQuantChoices(config),
        .context_cap = config.contextCap(),
        .reads_yarn = arch_config.fact(config, "reads_yarn"),
        .cooler_setting = coolerSettingName(config),
        .exclusive_decode = arch_config.exclusiveDecode(config),
        .max_concurrent_cap = if (arch_config.exclusiveDecode(config)) 1 else @intCast(arch_sched.rowCap(config) orelse sched_policy.MAX_BATCH_GROUP),
        .serves_bias = !arch_config.fact(config, "bias_unserved"),
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

/// A non-zero `--expert-cache-bias` (flag or model setting) on an engine whose arch has no cache-favoring pick is ignored
/// with this one line.
pub fn expertCacheBiasIgnoredLine(config: *const model.ModelConfig) ?[]const u8 {
    if (!(config.expertCacheBias() > 0) or admission.expertStream(config).cache_bias) return null;
    return "[expert-bias] ignored: this model has no cache-favoring pick";
}

/// Another arch's Sushi Cooler setting in the model's row is ignored with this one line, as `ssd_prefetch` is.
pub fn settingsCoolerIgnoredLine(config: *const model.ModelConfig) ?[]const u8 {
    const stray = config.cooler_setting_stray orelse return null;
    inline for (comptime std.meta.tags(registry.Tag)) |t| if (comptime arch_config.coolerOptOut(t)) |opt| {
        if (std.mem.eql(u8, stray, opt.setting)) return "model-settings.json " ++ opt.setting ++ " is ignored: it turns Sushi Cooler off for " ++ registry.Mod(t).model_types[0] ++ " only";
    };
    return null;
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
    const all = LaunchFlagUse{ .kv_quant = true, .qwen_gdn_16 = true, .mtp_head_kv_quant = true };
    try testing.expect(archFlagRefusal(&qwen4, all) == null);
    for ([_]struct { use: LaunchFlagUse, why: []const u8 }{
        .{ .use = .{ .kv_quant = true }, .why = "--kv-quant has no effect on deepseek_v41: its compressed state is stored as the QAT round trips left it, so there is no KV cache to quantize; drop the flag" },
        .{ .use = .{ .qwen_gdn_16 = true }, .why = "--qwen-gdn has no effect on deepseek_v41: it has no GatedDeltaNet layers; drop the flag" },
        .{ .use = .{ .mtp_head_kv_quant = true }, .why = "--mtp-head-kv-quant has no effect on deepseek_v41: its DSpark stages keep their own windows; drop the flag" },
    }) |c| {
        try testing.expectEqualStrings(c.why, archFlagRefusal(&dsv41, c.use) orelse return error.TestExpectedRefusal);
    }
    try testing.expect(settingsKvQuantIgnoredLine(&dsv41) == null);
    dsv41.kv_quant_override = kv_quant.KVQuantConfig.affine(8);
    try testing.expectEqualStrings("model-settings.json kv_quant is ignored on deepseek_v41 (no KV cache to quantize)", settingsKvQuantIgnoredLine(&dsv41).?);
    try testing.expect(settingsKvQuantIgnoredLine(&qwen4) == null);
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

test "a cache bias on a model whose router has no cache-favoring pick is ignored with one line, and where it applies with none" {
    const saved = model_settings.expert_cache_bias_flag;
    defer model_settings.expert_cache_bias_flag = saved;
    model_settings.expert_cache_bias_flag = null;
    for ([_][]const u8{ "deepseek_v41", "qwen4_exp", "mimo_v2", "glm5_next" }) |arch| {
        if (!built(arch)) continue;
        var cfg = model.ModelConfig{ .model_type = arch, .expert_layout = .exl3_k4 };
        try testing.expect(expertCacheBiasIgnoredLine(&cfg) == null);
        cfg.expert_cache_bias_override = 0.5;
        const applies = admission.expertStream(&cfg).cache_bias;
        try testing.expectEqual(applies, expertCacheBiasIgnoredLine(&cfg) == null);
        if (!applies) try testing.expectEqualStrings("[expert-bias] ignored: this model has no cache-favoring pick", expertCacheBiasIgnoredLine(&cfg).?);
        cfg.expert_cache_bias_override = null;
        model_settings.expert_cache_bias_flag = 0.5;
        try testing.expectEqual(applies, expertCacheBiasIgnoredLine(&cfg) == null);
        model_settings.expert_cache_bias_flag = null;
    }
    var bf16 = model.ModelConfig{ .model_type = "qwen4_exp", .expert_layout = .bf16_fused, .expert_cache_bias_override = 0.5 };
    if (built("qwen4_exp")) try testing.expect(expertCacheBiasIgnoredLine(&bf16) != null);
}

test "--drop-expert is refused on every pack but a quantized one whose experts stream, before the weights load" {
    const use = LaunchFlagUse{ .drop_expert = true };
    const why = "--drop-expert has no effect on this model: only a quantized pack whose experts stream drops them; drop the flag";
    if (built("deepseek_v41")) try testing.expect(archFlagRefusal(&model.ModelConfig{ .model_type = "deepseek_v41" }, use) == null);
    if (built("qwen4_exp")) {
        try testing.expect(archFlagRefusal(&model.ModelConfig{ .model_type = "qwen4_exp", .expert_layout = .exl3_k4 }, use) == null);
        try testing.expectEqualStrings(why, archFlagRefusal(&model.ModelConfig{ .model_type = "qwen4_exp", .expert_layout = .bf16_fused }, use) orelse return error.TestExpectedRefusal);
    }
    for ([_][]const u8{ "mimo_v2", "glm5_next" }) |arch| {
        if (!built(arch)) continue;
        const cfg = model.ModelConfig{ .model_type = arch };
        try testing.expectEqualStrings(why, archFlagRefusal(&cfg, use) orelse return error.TestExpectedRefusal);
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

test "--ds41-mlx-engine takes no value and keeps DeepSeek loads on MLX; a model of another arch loads as it would, its load naming the flag, and its model setting is ignored there" {
    if (!built("deepseek_v41")) return error.SkipZigTest;
    const saved = model.cooler_off;
    defer model.cooler_off = saved;
    model.cooler_off = .empty;
    var i: usize = 0;
    try testing.expectEqual(Parsed.applied, parseOption(&.{ "--ds41-mlx-engine", "--next" }, &i).?);
    try testing.expectEqual(@as(usize, 0), i);
    try testing.expectEqual(@as(usize, 1), model.cooler_off.count());
    var dsv41 = model.ModelConfig{ .model_type = "deepseek_v41" };
    dsv41.cooler_override = coolerSettingOf(&dsv41, .{ .ds41_cooler = false });
    try testing.expectEqual(@as(?bool, false), dsv41.cooler_override);
    dsv41.cooler_setting_stray = strayCoolerSetting(&dsv41, .{ .ds41_cooler = false });
    for (coolerOffIgnoredLines(&dsv41)) |line| try testing.expect(line == null);
    try testing.expect(archFlagRefusal(&dsv41, .{}) == null);
    try testing.expect(settingsCoolerIgnoredLine(&dsv41) == null);
    try testing.expect(model_settings.applies("ds41_cooler", settingsFit(&dsv41)));
    for ([_][]const u8{ "qwen4_exp", "mimo_v2", "glm5_next" }) |arch| {
        if (!built(arch)) continue;
        var cfg = model.ModelConfig{ .model_type = arch };
        try testing.expect(archFlagRefusal(&cfg, .{}) == null);
        var named: usize = 0;
        for (coolerOffIgnoredLines(&cfg)) |line| if (line) |l| {
            try testing.expectEqualStrings("--ds41-mlx-engine has no effect on this model: it keeps deepseek_v41 loads on MLX", l);
            named += 1;
        };
        try testing.expectEqual(@as(usize, 1), named);
        try testing.expect(settingsCoolerIgnoredLine(&cfg) == null);
        try testing.expect(!model_settings.applies("ds41_cooler", settingsFit(&cfg)));
        try testing.expectEqual(@as(?bool, null), coolerSettingOf(&cfg, .{ .ds41_cooler = false }));
        cfg.cooler_setting_stray = strayCoolerSetting(&cfg, .{ .ds41_cooler = false });
        try testing.expectEqualStrings("model-settings.json ds41_cooler is ignored: it turns Sushi Cooler off for deepseek_v41 only", settingsCoolerIgnoredLine(&cfg).?);
    }
}

test "--qwen-mlx-engine takes no value and keeps Flash-Next loads on MLX; a model of another arch loads as it would, its load naming the flag, and qwen_cooler is ignored there" {
    if (!built("qwen4_exp")) return error.SkipZigTest;
    const saved = model.cooler_off;
    defer model.cooler_off = saved;
    model.cooler_off = .empty;
    var i: usize = 0;
    try testing.expectEqual(Parsed.applied, parseOption(&.{ "--qwen-mlx-engine", "--next" }, &i).?);
    try testing.expectEqual(@as(usize, 0), i);
    var qwen = model.ModelConfig{ .model_type = "qwen4_exp" };
    for (coolerOffIgnoredLines(&qwen)) |line| try testing.expect(line == null);
    try testing.expect(archFlagRefusal(&qwen, .{}) == null);
    try testing.expect(model_settings.applies("qwen_cooler", settingsFit(&qwen)));
    try testing.expect(!model_settings.applies("ds41_cooler", settingsFit(&qwen)));
    try testing.expectEqual(@as(?bool, false), coolerSettingOf(&qwen, .{ .qwen_cooler = false, .ds41_cooler = true }));
    // DeepSeek's setting, stray on a Qwen load, is named only where DeepSeek is built.
    if (built("deepseek_v41")) {
        qwen.cooler_setting_stray = strayCoolerSetting(&qwen, .{ .qwen_cooler = false, .ds41_cooler = true });
        try testing.expectEqualStrings("model-settings.json ds41_cooler is ignored: it turns Sushi Cooler off for deepseek_v41 only", settingsCoolerIgnoredLine(&qwen).?);
    }
    for ([_][]const u8{ "deepseek_v41", "mimo_v2", "glm5_next" }) |arch| {
        if (!built(arch)) continue;
        var cfg = model.ModelConfig{ .model_type = arch };
        try testing.expect(archFlagRefusal(&cfg, .{}) == null);
        var named: usize = 0;
        for (coolerOffIgnoredLines(&cfg)) |line| if (line) |l| {
            try testing.expectEqualStrings("--qwen-mlx-engine has no effect on this model: it keeps qwen4_exp loads on MLX", l);
            named += 1;
        };
        try testing.expectEqual(@as(usize, 1), named);
        try testing.expect(!model_settings.applies("qwen_cooler", settingsFit(&cfg)));
        cfg.cooler_setting_stray = strayCoolerSetting(&cfg, .{ .qwen_cooler = false });
        try testing.expectEqualStrings("model-settings.json qwen_cooler is ignored: it turns Sushi Cooler off for qwen4_exp only", settingsCoolerIgnoredLine(&cfg).?);
    }
}

test "flags: --prefill-chunk-max and --prefill-chunk-min parse 512, 1024, 2048, 4096 or 8192 rows on every model and refuse any other" {
    const saved = settings_resolve.launch;
    defer settings_resolve.launch = saved;
    settings_resolve.launch = .{};
    var i: usize = 0;
    try testing.expectEqual(Parsed.applied, parseOption(&.{ "--prefill-chunk-max", "4096" }, &i).?);
    try testing.expectEqual(@as(usize, 1), i);
    try testing.expectEqual(@as(?u32, 4096), settings_resolve.launch.prefill_chunk_max);
    i = 0;
    try testing.expectEqual(Parsed.applied, parseOption(&.{ "--prefill-chunk-min", "512" }, &i).?);
    try testing.expectEqual(@as(?u32, 512), settings_resolve.launch.prefill_chunk_min);
    for ([_][]const u8{ "0", "31", "256", "3000", "4095", "8193", "16384", "-1", "4k" }) |bad| for ([_][]const u8{ "--prefill-chunk-max", "--prefill-chunk-min" }) |flag| {
        i = 0;
        const refused = parseOption(&.{ flag, bad }, &i).?.refused;
        try testing.expectEqualStrings("expected 512, 1024, 2048, 4096 or 8192 rows", refused.expected);
    };
    i = 0;
    try testing.expect(parseOption(&.{"--prefill-chunk-min"}, &i) == null);
    // Every served model takes them, its settings page offers them, and no arch refuses them.
    for ([_][]const u8{ "qwen4_exp", "mimo_v2", "glm5_next", "deepseek_v41", "hy_v4_flash" }) |arch| {
        if (!built(arch)) continue;
        const cfg = model.ModelConfig{ .model_type = arch };
        try testing.expect(archFlagRefusal(&cfg, .{}) == null);
        try testing.expect(model_settings.applies("prefill_chunk_max", settingsFit(&cfg)));
        try testing.expect(model_settings.applies("prefill_chunk_min", settingsFit(&cfg)));
        try testing.expectEqual(@as([]const []const u8, &model_settings.chunk_choices), model_settings.findKey(&model_settings.model_keys, "prefill_chunk_max").?.choices);
    }
}

test "an arch that serves no bias refuses a request's logit_bias and think_penalty, a think penalty by setting and a bias file at load, each naming itself and the parameter; another arch serves them, and a think penalty by flag is no refusal" {
    if (!built("hy_v4_flash") or !built("qwen4_exp")) return error.SkipZigTest;
    var buf: [256]u8 = undefined;
    const hy4 = model.ModelConfig{ .model_type = "hy_v4_flash" };
    const qwen = model.ModelConfig{ .model_type = "qwen4_exp" };
    try testing.expectEqualStrings("Hy4-Flash does not support logit_bias", biasRefusal(&buf, &hy4, "logit_bias").?);
    try testing.expectEqualStrings("Hy4-Flash does not support think_penalty", biasRefusal(&buf, &hy4, "think_penalty").?);
    try testing.expect(biasRefusal(&buf, &qwen, "logit_bias") == null);
    try testing.expect(biasRefusal(&buf, &qwen, "think_penalty") == null);
    try testing.expect(biasLoadRefusal(&buf, &hy4) == null);
    var saved = hy4;
    saved.think_penalty_override = 2;
    try testing.expectEqualStrings("Hy4-Flash does not support think_penalty: remove it from this model's settings", biasLoadRefusal(&buf, &saved).?);
    var filed = hy4;
    filed.logit_bias_file_override = @import("../logit_bias.zig").FilePath.from("bias.json");
    try testing.expectEqualStrings("Hy4-Flash does not support logit_bias: remove logit_bias_file from this model's settings", biasLoadRefusal(&buf, &filed).?);
    try testing.expect(biasLoadRefusal(&buf, &qwen) == null);
    var qwen_filed = qwen;
    qwen_filed.logit_bias_file_override = @import("../logit_bias.zig").FilePath.from("bias.json");
    qwen_filed.think_penalty_override = 2;
    try testing.expect(biasLoadRefusal(&buf, &qwen_filed) == null);
}
