//! Where a launch flag, a `model-settings.json` value and a default meet: an explicit flag outranks the file, which
//! outranks the default. A flag the operator did not pass is null, never a value beside a "given" bit.
const std = @import("std");
const mtp_acceptance = @import("mtp_acceptance.zig");
const model_settings = @import("model_settings.zig");

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
    mtp_acceptance: ?mtp_acceptance.Mode = null,
    mtp_greedy_tail: ?bool = null,
};

pub var launch: Launch = .{};

/// The prefill width. `--prefill-chunk` outranks the per-model chunk the memory sizer pins
/// (`ModelConfig.pinned_prefill_chunk`); without the flag 8192 is only the default.
pub var prefill_chunk: Setting(usize) = .{ .value = 8192, .source = .default };

/// THIS model's acceptance mode: the launch flag > its `mtp_acceptance` > exact.
pub fn acceptanceFor(file: ?mtp_acceptance.Mode) Setting(mtp_acceptance.Mode) {
    return resolve(mtp_acceptance.Mode, launch.mtp_acceptance, file, model_settings.default_mtp_acceptance);
}

/// THIS model's greedy tail: the launch flag > its `mtp_greedy_tail` > off.
pub fn greedyTailFor(file: ?bool) Setting(bool) {
    return resolve(bool, launch.mtp_greedy_tail, file, model_settings.default_mtp_greedy_tail);
}

const testing = std.testing;
const model_mod = @import("model.zig");
const model_load = @import("model_load.zig");
const kv_quant = @import("kv_quant.zig");
const bills = @import("bills.zig");
const http = @import("http.zig");
const server_state = @import("server_state.zig");

const KV = kv_quant.KVQuantConfig;
const Mode = mtp_acceptance.Mode;

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

test "settings precedence: mtp_acceptance is flag > file > exact" {
    const saved = launch;
    defer launch = saved;
    const typical: Mode = .{ .typical = .{ .delta = 0.1 } };
    const tokenv3: Mode = .{ .tokenv3 = 0.9 };
    const rows = [_]struct { flag: ?Mode, file: ?Mode, want: Mode, source: Source }{
        .{ .flag = null, .file = null, .want = .exact, .source = .default },
        .{ .flag = null, .file = tokenv3, .want = tokenv3, .source = .model_settings },
        .{ .flag = typical, .file = null, .want = typical, .source = .flag },
        .{ .flag = typical, .file = tokenv3, .want = typical, .source = .flag },
        .{ .flag = tokenv3, .file = typical, .want = tokenv3, .source = .flag },
    };
    for (rows) |r| {
        launch = .{ .mtp_acceptance = r.flag };
        try expectSetting(Mode, r.want, r.source, acceptanceFor(r.file));
    }
}

test "settings precedence: mtp_greedy_tail is flag > file > off, and the flag only turns it on" {
    const saved = launch;
    defer launch = saved;
    const rows = [_]struct { flag: ?bool, file: ?bool, want: bool, source: Source }{
        .{ .flag = null, .file = null, .want = false, .source = .default },
        .{ .flag = null, .file = true, .want = true, .source = .model_settings },
        .{ .flag = null, .file = false, .want = false, .source = .model_settings },
        .{ .flag = true, .file = null, .want = true, .source = .flag },
        .{ .flag = true, .file = false, .want = true, .source = .flag },
    };
    for (rows) |r| {
        launch = .{ .mtp_greedy_tail = r.flag };
        try expectSetting(bool, r.want, r.source, greedyTailFor(r.file));
    }
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

test "settings precedence: --prefill-chunk outranks the pinned width only when given" {
    const saved = prefill_chunk;
    defer prefill_chunk = saved;
    prefill_chunk = .{ .value = 4096, .source = .default };
    try testing.expectEqual(@as(u32, 0), bills.explicitPrefillChunk());
    prefill_chunk = .{ .value = 4096, .source = .flag };
    try testing.expectEqual(@as(u32, 4096), bills.explicitPrefillChunk());
}
