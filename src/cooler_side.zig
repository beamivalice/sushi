//! Where a load runs, decided in one place for every arch: Sushi Cooler, the engine of our own, or MLX. An arch on
//! Sushi Cooler declares it (`cooler`); one with an opt-out also declares the loads it cannot run there yet
//! (`coolerGap`). The user's opt-out, the arch's gap and a test's side are read here and nowhere else (lint
//! `cooler-side-query`). The answer is a pure function of the config, so a load's side is fixed for its life.
const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const registry = @import("arch/registry.zig");
const arch_config = @import("arch/config.zig");

pub const Side = union(enum) {
    engine,
    /// The arch has no Sushi Cooler.
    mlx_no_engine,
    /// The arch's launch flag (`cooler.opt_out.flag`).
    mlx_by_flag,
    /// The model setting (`cooler.opt_out.setting`) turned Sushi Cooler off.
    mlx_by_setting,
    /// The arch cannot run this load on Sushi Cooler yet, for the reason it gives.
    mlx_gap: []const u8,
    /// A test that names no side keeps MLX.
    mlx_in_test,
};

comptime {
    for (std.meta.tags(registry.Tag)) |t| if (arch_config.cooler(t)) |c| {
        const gap = @hasDecl(registry.Mod(t), "coolerGap");
        if (c.opt_out) |o| {
            if (!gap) @compileError(@tagName(t) ++ " declares the Sushi Cooler opt-out " ++ o.flag ++ " without coolerGap");
        } else if (gap) @compileError(@tagName(t) ++ " has no MLX route for coolerGap to send a load to");
    };
}

pub fn sideOf(config: *const model.ModelConfig) Side {
    const tag = arch_config.tagOf(config) orelse return .mlx_no_engine;
    return switch (tag) {
        inline else => |t| sideFor(comptime arch_config.cooler(t), registry.Mod(t), config),
    };
}

/// The side of a load whose arch declares `cooler` and, beside an opt-out, `M.coolerGap`.
fn sideFor(comptime cooler: ?arch_config.Cooler, comptime M: type, config: *const model.ModelConfig) Side {
    const c = cooler orelse return .mlx_no_engine;
    if (c.opt_out == null) return .engine;
    switch (config.cooler_ask) {
        .flag => return .mlx_by_flag,
        .setting => return .mlx_by_setting,
        .none => {},
    }
    if (M.coolerGap(config)) |why| return .{ .mlx_gap = why };
    // The tests written against MLX keep it; the engine's own tests name the engine.
    if (comptime builtin.is_test) return if (config.cooler_for_test orelse false) .engine else .mlx_in_test;
    return .engine;
}

/// Whether the load of `config` runs on the engine of our own. Every hook that differs by side asks here.
pub fn onEngine(config: *const model.ModelConfig) bool {
    return sideOf(config) == .engine;
}

/// The stream a load of `config` runs on when it is the engine's; null for the thread's GPU stream.
pub fn stream(config: *const model.ModelConfig) ?mlx.mlx_stream {
    return if (onEngine(config)) mlx.engine.stream() else null;
}

/// Who kept a load on MLX: the user's flag, else the model setting, else nobody. Stamped once at the load.
pub fn stamp(config: *model.ModelConfig, on: bool, by_flag: bool) void {
    config.cooler_ask = if (on) .none else if (by_flag) .flag else .setting;
}

/// A test names the side its load runs on, before the load.
pub fn forTest(config: *model.ModelConfig, on: bool) void {
    config.cooler_for_test = on;
}

/// The load line of `config`'s side after `what` (the arch's own words for the forwards it moves), naming the arch's
/// flag and setting.
pub fn line(buf: []u8, what: []const u8, config: *const model.ModelConfig) []const u8 {
    const o = optOutOf(config) orelse arch_config.CoolerOptOut{ .flag = "", .setting = "" };
    return switch (sideOf(config)) {
        .engine => std.fmt.bufPrint(buf, "{s}: on Sushi Cooler, the engine of our own", .{what}),
        .mlx_no_engine => std.fmt.bufPrint(buf, "{s}: on MLX", .{what}),
        .mlx_by_flag => std.fmt.bufPrint(buf, "{s}: on MLX because {s} was given (Sushi Cooler is off)", .{ what, o.flag }),
        .mlx_by_setting => std.fmt.bufPrint(buf, "{s}: on MLX because model-settings.json {s} is off (Sushi Cooler is off)", .{ what, o.setting }),
        .mlx_gap => |why| std.fmt.bufPrint(buf, "{s}: on MLX: {s}", .{ what, why }),
        .mlx_in_test => std.fmt.bufPrint(buf, "{s}: on MLX (a test that names no side)", .{what}),
    } catch what;
}

fn optOutOf(config: *const model.ModelConfig) ?arch_config.CoolerOptOut {
    const tag = arch_config.tagOf(config) orelse return null;
    return switch (tag) {
        inline else => |t| arch_config.coolerOptOut(t),
    };
}

test "cooler side: an arch on Sushi Cooler with no opt-out runs every load there, whatever the asks or a test's side" {
    const only: arch_config.Cooler = .{ .opt_out = null };
    var c: model.ModelConfig = .{ .model_type = "cooler_only" };
    try std.testing.expect(sideFor(only, struct {}, &c) == .engine);
    c.cooler_ask = .flag;
    try std.testing.expect(sideFor(only, struct {}, &c) == .engine);
    c.cooler_ask = .setting;
    try std.testing.expect(sideFor(only, struct {}, &c) == .engine);
    c.cooler_for_test = false;
    try std.testing.expect(sideFor(only, struct {}, &c) == .engine);
}

test "cooler side: an opt-out arch reads the asks, then its gap, then a test's side; an arch without Sushi Cooler stays on MLX" {
    const Gapped = struct {
        pub fn coolerGap(config: *const model.ModelConfig) ?[]const u8 {
            return if (config.quant_bits == 4) "no 4-bit packs" else null;
        }
    };
    const opt: arch_config.Cooler = .{ .opt_out = .{ .flag = "--x-mlx-engine", .setting = "x_cooler" } };
    var c: model.ModelConfig = .{ .model_type = "opt_out", .cooler_for_test = true };
    try std.testing.expect(sideFor(opt, Gapped, &c) == .engine);
    c.cooler_ask = .flag;
    try std.testing.expect(sideFor(opt, Gapped, &c) == .mlx_by_flag);
    c.cooler_ask = .setting;
    try std.testing.expect(sideFor(opt, Gapped, &c) == .mlx_by_setting);
    c.cooler_ask = .none;
    c.quant_bits = 4;
    try std.testing.expectEqualStrings("no 4-bit packs", sideFor(opt, Gapped, &c).mlx_gap);
    c.quant_bits = 0;
    c.cooler_for_test = null;
    try std.testing.expect(sideFor(opt, Gapped, &c) == .mlx_in_test);
    try std.testing.expect(sideFor(null, Gapped, &c) == .mlx_no_engine);
}
