//! Where a load runs, decided in one place for every arch: Sushi Cooler, the engine of our own, or MLX. An arch on
//! Sushi Cooler declares its opt-out (`cooler_opt_out`) and the loads it cannot run there yet (`coolerGap`); the user's
//! opt-out, the arch's gap and a test's side are read here and nowhere else (lint `cooler-side-query`). The answer is a
//! pure function of the config, so a load's side is fixed for its life.
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
    /// The arch's launch flag (`cooler_opt_out.flag`).
    mlx_by_flag,
    /// The model setting (`cooler_opt_out.setting`) turned Sushi Cooler off.
    mlx_by_setting,
    /// The arch cannot run this load on Sushi Cooler yet, for the reason it gives.
    mlx_gap: []const u8,
    /// A test that names no side keeps MLX.
    mlx_in_test,
};

comptime {
    for (std.meta.tags(registry.Tag)) |t| if (arch_config.coolerOptOut(t)) |o| {
        if (!@hasDecl(registry.Mod(t), "coolerGap")) @compileError(@tagName(t) ++ " declares the Sushi Cooler opt-out " ++ o.flag ++ " without coolerGap");
    };
}

pub fn sideOf(config: *const model.ModelConfig) Side {
    const tag = arch_config.tagOf(config) orelse return .mlx_no_engine;
    switch (tag) {
        inline else => |t| {
            if (comptime arch_config.coolerOptOut(t) == null) return .mlx_no_engine;
            switch (config.cooler_ask) {
                .flag => return .mlx_by_flag,
                .setting => return .mlx_by_setting,
                .none => {},
            }
            if (registry.Mod(t).coolerGap(config)) |why| return .{ .mlx_gap = why };
            // The tests written against MLX keep it; the engine's own tests name the engine.
            if (comptime builtin.is_test) return if (config.cooler_for_test orelse false) .engine else .mlx_in_test;
            return .engine;
        },
    }
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
