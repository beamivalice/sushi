//! A registry of fake benches for the report tests: names and scopes in the shape the harness lists them.
const report = @import("ubench_report.zig");

fn fakeRun(_: *report.Run) anyerror!void {}

const fake_axes = [_]report.Axis{
    .{ .name = "rows", .values = &.{ "1", "4" } },
    .{ .name = "mode", .values = &.{ "uniform", "lowest" }, .numeric = false },
};
pub const fake_registry = [_]report.Bench{
    .{ .name = "dsv41.experts", .scope = .dsv41, .what = "chain", .axes = &fake_axes, .arms = &.{ "a", "b" }, .ref = "a", .opts = &.{.{ .name = "text", .help = "x" }}, .run = fakeRun },
    .{ .name = "dsv41.packed", .scope = .dsv41, .kind = .pack, .what = "p", .arms = &.{"only"}, .ref = "only", .run = fakeRun },
    .{ .name = "shared.other", .scope = .shared, .what = "o", .arms = &.{"x"}, .ref = "x", .run = fakeRun },
};
