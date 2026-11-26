//! Where a launch flag, a `model-settings.json` value and a default meet: an explicit flag outranks the file, which
//! outranks the default. A flag the operator did not pass is null, never a value beside a "given" bit.
const std = @import("std");
const model_settings = @import("model_settings.zig");
const model_mod = @import("model.zig");
const arch_config = @import("arch/config.zig");
const prefill_plan = @import("prefill_plan.zig");

/// Where a load's value for one key came from. A request's own field outranks all of them.
pub const Source = enum { flag, model_settings, default };

pub fn Setting(comptime T: type) type {
    return struct { value: T, source: Source };
}

/// `flag` null = not given; `file` null = the file says nothing.
pub fn resolve(comptime T: type, flag: ?T, file: ?T, default: T) Setting(T) {
    if (flag) |f| return .{ .value = f, .source = .flag };
    if (file) |s| return .{ .value = s, .source = .model_settings };
    return .{ .value = default, .source = .default };
}

pub fn sourceLabel(source: Source, flag_name: []const u8) []const u8 {
    return switch (source) {
        .flag => flag_name,
        .model_settings => "model-settings.json",
        .default => "default",
    };
}

/// The launch flags no load params carry: set once by `main` before the first load, null = not given.
pub const Launch = struct {
    /// `--drafter` was given.
    drafter: bool = false,
    /// `--prefill-chunk-max` / `--prefill-chunk-min`: the rows every model's prefill chunk is chosen within.
    prefill_chunk_max: ?u32 = null,
    prefill_chunk_min: ?u32 = null,
};

pub var launch: Launch = .{};

/// The widths a load's prefill chunk may take, widest first. Every bound of it (`--prefill-chunk-max`,
/// `--prefill-chunk-min`, and their model-settings keys) is one of them, else it is refused by name.
pub const static_chunk_rungs = [_]u32{ 8192, 4096, 2048, 1024, 512 };
/// `static_chunk_rungs` as a refusal names them.
pub const static_chunk_rungs_text = "512, 1024, 2048, 4096 or 8192 rows";

pub fn isStaticChunkRung(rows: u64) bool {
    for (static_chunk_rungs) |r| if (rows == r) return true;
    return false;
}

/// The rows a model's prefill chunk is chosen within. `max_given` / `min_given`: a flag or a model-settings key named
/// it; else it is the model's own default.
pub const ChunkBounds = struct {
    min: u32,
    max: u32,
    max_given: bool = false,
    min_given: bool = false,
    /// A given min above the model's default max raised the max to it.
    raised_by_min: bool = false,
    /// The given min is model-settings.json's, not `--prefill-chunk-min`.
    min_from_file: bool = false,
};

/// The widest chunk a model's prefill runs when no bound names its max: the static bill's own default (Hy4's 4096),
/// else the widest ladder rung at or below the width the arch starts its prefill at.
pub fn defaultChunkMax(config: *const model_mod.ModelConfig) u32 {
    const widest = arch_config.staticChunkMax(config);
    if (widest != 0) return arch_config.staticChunkDefault(config);
    const start = config.prefillStartWidth();
    for (static_chunk_rungs) |r| if (r <= start) return r;
    return static_chunk_rungs[static_chunk_rungs.len - 1];
}

/// An omitted `--prefill-chunk-min`: the narrowest rung, never above the max.
pub fn defaultChunkMin(max: u32) u32 {
    return @min(static_chunk_rungs[static_chunk_rungs.len - 1], max);
}

/// A model's chunk bounds from the launch flag, else the model-settings.json key, else the defaults. The ONE decision:
/// `bills` formats its refusal, and the forward and the bill read its `.bounds`. A given min above the model's DEFAULT max
/// raises the max to the min (`raised_by_min`); above a GIVEN max it is refused by name.
pub const ChunkBoundRefusal = union(enum) {
    not_a_rung: struct { rows: u32, name: []const u8 },
    min_above_given_max: struct { min: u32, min_name: []const u8, max: u32, max_name: []const u8 },
    min_above_arch_ceiling: struct { min: u32, min_name: []const u8, ceiling: u32 },

    /// A bound model-settings.json names is at fault: the file is edited without a restart, so the next load retries.
    pub fn fromFile(self: ChunkBoundRefusal) bool {
        const eql = std.mem.eql;
        return switch (self) {
            .not_a_rung => |g| eql(u8, g.name, file_max_name) or eql(u8, g.name, file_min_name),
            .min_above_given_max => |m| eql(u8, m.min_name, file_min_name) or eql(u8, m.max_name, file_max_name),
            .min_above_arch_ceiling => |m| eql(u8, m.min_name, file_min_name),
        };
    }
};

const file_max_name = "model-settings.json prefill_chunk_max";
const file_min_name = "model-settings.json prefill_chunk_min";

/// The widest chunk this model's prefill width reaches at a short prompt: a named min above it is refused.
pub fn chunkCeiling(config: *const model_mod.ModelConfig) u32 {
    return @intCast(prefill_plan.boundedPrefillChunk(chunk_rung_top, config.prefillScoreHeadDim(), config.num_attention_heads, 1, config.has_sliding_window, config.isMoe(), config.longCtxGated()));
}

pub const BoundsPick = union(enum) {
    bounds: ChunkBounds,
    refused: ChunkBoundRefusal,
};

pub fn chunkBounds(config: *const model_mod.ModelConfig) BoundsPick {
    const Bound = struct { rows: u32, name: []const u8 };
    const max: ?Bound = if (launch.prefill_chunk_max) |v| .{ .rows = v, .name = "--prefill-chunk-max" } else if (config.prefill_chunk_max_override) |v| .{ .rows = v, .name = file_max_name } else null;
    const min: ?Bound = if (launch.prefill_chunk_min) |v| .{ .rows = v, .name = "--prefill-chunk-min" } else if (config.prefill_chunk_min_override) |v| .{ .rows = v, .name = file_min_name } else null;
    for ([_]?Bound{ max, min }) |given| if (given) |g| if (!isStaticChunkRung(g.rows)) {
        return .{ .refused = .{ .not_a_rung = .{ .rows = g.rows, .name = g.name } } };
    };
    const default_max = defaultChunkMax(config);
    const raised = max == null and min != null and min.?.rows > default_max;
    const hi: u32 = if (max) |g| g.rows else if (raised) min.?.rows else default_max;
    const lo: u32 = if (min) |g| g.rows else defaultChunkMin(hi);
    if (lo > hi) return .{ .refused = .{ .min_above_given_max = .{ .min = lo, .min_name = min.?.name, .max = hi, .max_name = max.?.name } } };
    // A named min the width funnel caps at a short prompt would be under-run by every request: refuse the load instead.
    if (min) |named| {
        const ceiling = chunkCeiling(config);
        if (ceiling < lo) return .{ .refused = .{ .min_above_arch_ceiling = .{ .min = lo, .min_name = named.name, .ceiling = ceiling } } };
    }
    return .{ .bounds = .{ .min = lo, .max = hi, .max_given = max != null or raised, .min_given = min != null, .raised_by_min = raised, .min_from_file = if (min) |g| std.mem.eql(u8, g.name, file_min_name) else false } };
}

/// The widest chunk a no-flag load ran with: the model's max where a bound named it, else the widest rung (8192, the
/// old global default). A bound-free load keeps every figure it had, and the checkpoint stride it must cover.
pub const chunk_rung_top: u32 = static_chunk_rungs[0];

const testing = std.testing;
const model_load = @import("model_load.zig");
const kv_quant = @import("kv_quant.zig");
const bills = @import("bills.zig");
const http = @import("http.zig");
const server_state = @import("server_state.zig");

const KV = kv_quant.KVQuantConfig;

fn expectSetting(comptime T: type, want_value: T, want_source: Source, got: Setting(T)) !void {
    try testing.expectEqualDeep(want_value, got.value);
    try testing.expectEqual(want_source, got.source);
}

test "resolve: the flag outranks the file, which outranks the default" {
    try expectSetting(u8, 3, .default, resolve(u8, null, null, 3));
    try expectSetting(u8, 2, .model_settings, resolve(u8, null, 2, 3));
    try expectSetting(u8, 1, .flag, resolve(u8, 1, 2, 3));
    try expectSetting(bool, false, .flag, resolve(bool, false, true, true));
}

// The precedence invariant (CLAUDE.md, docs/server-lifecycle.md#settings) as a table: for every key, each
// (flag, file) pair resolves to the value and source below.

test "settings precedence: mtp is flag > file > on" {
    const rows = [_]struct { flag: ?bool, file: ?bool, on: bool, source: Source }{
        .{ .flag = null, .file = null, .on = true, .source = .default },
        .{ .flag = null, .file = true, .on = true, .source = .model_settings },
        .{ .flag = null, .file = false, .on = false, .source = .model_settings },
        .{ .flag = true, .file = null, .on = true, .source = .flag },
        .{ .flag = true, .file = false, .on = true, .source = .flag },
        .{ .flag = false, .file = null, .on = false, .source = .flag },
        .{ .flag = false, .file = true, .on = false, .source = .flag },
    };
    for (rows) |r| {
        const config = model_mod.ModelConfig{ .mtp_override = r.file };
        const got = model_load.mtpChoiceFor(r.flag, &config);
        try testing.expectEqual(r.on, got.on);
        try testing.expectEqual(r.source, got.source);
    }
}

test "settings precedence: kv_quant is flag > file > the engine default" {
    const rows = [_]struct { flag: ?KV, file: ?KV, want: KV, source: Source }{
        .{ .flag = null, .file = null, .want = KV.engine_default, .source = .default },
        .{ .flag = null, .file = KV.dense, .want = KV.dense, .source = .model_settings },
        .{ .flag = null, .file = KV.affine(4), .want = KV.affine(4), .source = .model_settings },
        .{ .flag = KV.affine(6), .file = null, .want = KV.affine(6), .source = .flag },
        .{ .flag = KV.affine(6), .file = KV.dense, .want = KV.affine(6), .source = .flag },
        .{ .flag = KV.dense, .file = KV.affine(4), .want = KV.dense, .source = .flag },
    };
    for (rows) |r| {
        const got = kv_quant.KvCacheChoice.resolve(r.flag, r.file);
        try testing.expectEqualDeep(r.want, got.config);
        try testing.expectEqual(r.source, got.source);
    }
}

test "settings precedence: ctx_size is flag > file > auto, and 0 is auto on both sides" {
    const rows = [_]struct { flag: u32, file: u32, want: u32, source: Source }{
        .{ .flag = 0, .file = 0, .want = 0, .source = .default },
        .{ .flag = 0, .file = 8192, .want = 8192, .source = .model_settings },
        .{ .flag = 4096, .file = 0, .want = 4096, .source = .flag },
        .{ .flag = 4096, .file = 8192, .want = 4096, .source = .flag },
    };
    for (rows) |r| try expectSetting(u32, r.want, r.source, model_settings.contextPick(r.flag, r.file));
}

test "settings precedence: preserve_thinking and think_penalty are flag > file > default" {
    const bool_rows = [_]struct { flag: ?bool, file: ?bool, want: bool, source: Source }{
        .{ .flag = null, .file = null, .want = true, .source = .default },
        .{ .flag = null, .file = false, .want = false, .source = .model_settings },
        .{ .flag = false, .file = true, .want = false, .source = .flag },
        .{ .flag = true, .file = false, .want = true, .source = .flag },
    };
    for (bool_rows) |r| try expectSetting(bool, r.want, r.source, resolve(bool, r.flag, r.file, model_settings.default_preserve_thinking));
    const penalty_rows = [_]struct { flag: ?f32, file: ?f32, want: f32, source: Source }{
        .{ .flag = null, .file = null, .want = 0, .source = .default },
        .{ .flag = null, .file = 3, .want = 3, .source = .model_settings },
        .{ .flag = 0, .file = 3, .want = 0, .source = .flag },
        .{ .flag = 5, .file = null, .want = 5, .source = .flag },
    };
    for (penalty_rows) |r| try expectSetting(f32, r.want, r.source, resolve(f32, r.flag, r.file, model_settings.default_think_penalty));
}

test "settings precedence: vision is flag > file > the load's default" {
    const rows = [_]struct { flag: ?bool, file: ?bool, default_on: bool, want: bool, source: Source }{
        .{ .flag = null, .file = null, .default_on = true, .want = true, .source = .default },
        .{ .flag = null, .file = null, .default_on = false, .want = false, .source = .default },
        .{ .flag = null, .file = true, .default_on = false, .want = true, .source = .model_settings },
        .{ .flag = null, .file = false, .default_on = true, .want = false, .source = .model_settings },
        .{ .flag = true, .file = false, .default_on = false, .want = true, .source = .flag },
        .{ .flag = false, .file = true, .default_on = true, .want = false, .source = .flag },
    };
    for (rows) |r| {
        const got = model_settings.VisionChoice.resolve(r.flag, r.file, r.default_on);
        try testing.expectEqual(r.want, got.on);
        try testing.expectEqual(r.source, got.source);
    }
}

test "settings precedence: the server keys are flag > file > default" {
    const check_rows = [_]struct { no_flag: bool, file: ?bool, want: bool, source: Source }{
        .{ .no_flag = false, .file = null, .want = true, .source = .default },
        .{ .no_flag = false, .file = false, .want = false, .source = .model_settings },
        .{ .no_flag = true, .file = true, .want = false, .source = .flag },
        .{ .no_flag = true, .file = null, .want = false, .source = .flag },
    };
    for (check_rows) |r| try expectSetting(bool, r.want, r.source, model_settings.autoCheck(r.no_flag, r.file));
    const skip_rows = [_]struct { flag: bool, file: ?bool, want: bool, source: Source }{
        .{ .flag = false, .file = null, .want = false, .source = .default },
        .{ .flag = false, .file = true, .want = true, .source = .model_settings },
        .{ .flag = true, .file = false, .want = true, .source = .flag },
    };
    for (skip_rows) |r| try expectSetting(bool, r.want, r.source, model_settings.skipMemPreflight(r.flag, r.file));
    const bind_rows = [_]struct { host: ?[]const u8, port: ?u16, file_port: u16, want_host: []const u8, want_port: u16 }{
        .{ .host = null, .port = null, .file_port = 12345, .want_host = server_state.default_host, .want_port = 12345 },
        .{ .host = null, .port = null, .file_port = 9000, .want_host = server_state.default_host, .want_port = 9000 },
        .{ .host = "0.0.0.0", .port = 8080, .file_port = 9000, .want_host = "0.0.0.0", .want_port = 8080 },
    };
    for (bind_rows) |r| {
        const bind = http.resolveBind(r.host, r.port, r.file_port);
        try testing.expectEqualStrings(r.want_host, bind.host);
        try testing.expectEqual(r.want_port, bind.port);
    }
}

test "settings precedence: a prefix-cache number the operator named is used as given" {
    const config = model_mod.ModelConfig{};
    const default = server_state.PREFIX_CACHE_MEM_DEFAULT;
    try testing.expectEqual(default * 4, bills.defaultPrefixCacheAsk(default, false, default * 4));
    try testing.expectEqual(default, bills.defaultPrefixCacheAsk(default, true, default * 4));
    try testing.expectEqual(default * 2, bills.defaultPrefixCacheAsk(default * 2, false, default * 4));
    try testing.expectEqual(default, bills.defaultPrefixCacheAskFor(&config, default, true, default * 4));
}

test "settings precedence: the SSD ask names who set it" {
    const saved_bytes = server_state.prefix_cache_disk_bytes;
    const saved_source = server_state.prefix_cache_disk_source;
    defer {
        server_state.prefix_cache_disk_bytes = saved_bytes;
        server_state.prefix_cache_disk_source = saved_source;
    }
    const rows = [_]struct { source: Source, explicit: bool, name: []const u8 }{
        .{ .source = .default, .explicit = false, .name = "--prefix-cache-disk" },
        .{ .source = .flag, .explicit = true, .name = "--prefix-cache-disk" },
        .{ .source = .model_settings, .explicit = true, .name = "model-settings.json" },
    };
    for (rows) |r| {
        server_state.setDiskAsk(40 << 30, r.source);
        const ask = bills.diskAsk();
        try testing.expectEqual(@as(u64, 40 << 30), ask.bytes);
        try testing.expectEqual(r.explicit, ask.explicit());
        try testing.expectEqualStrings(r.name, ask.sourceName());
    }
}

test "settings precedence: the chunk bounds are flag > file > the model's defaults" {
    const saved = launch;
    defer launch = saved;
    var config = model_mod.ModelConfig{};
    config.arch = .{ .tag = null };
    config.model_type = "qwen4_exp";
    launch = .{};
    const quiet = chunkBounds(&config).bounds;
    try testing.expectEqual(defaultChunkMax(&config), quiet.max);
    try testing.expectEqual(@as(u32, 512), quiet.min);
    try testing.expect(!quiet.max_given and !quiet.min_given);
    config.prefill_chunk_max_override = 2048;
    try testing.expectEqual(@as(u32, 2048), chunkBounds(&config).bounds.max);
    try testing.expect(chunkBounds(&config).bounds.max_given);
    launch.prefill_chunk_max = 4096;
    launch.prefill_chunk_min = 1024;
    const picked = chunkBounds(&config).bounds;
    try testing.expectEqual(@as(u32, 4096), picked.max);
    try testing.expectEqual(@as(u32, 1024), picked.min);
    try testing.expect(picked.max_given and picked.min_given);
}
