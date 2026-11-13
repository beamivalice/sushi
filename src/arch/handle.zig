//! The engine's one door to a loaded model: the handle it holds (`Handle`, the registry's union of each arch's
//! `Model`) and every read it makes of one. Each read is one switch on the tag, resolved at compile time per prong.
const std = @import("std");
const mlx = @import("../mlx.zig");
const trunk_mod = @import("../trunk.zig");
const registry = @import("registry.zig");
const arch_request = @import("request.zig");
const arch_life = @import("life.zig");
const sushi_exl3 = @import("sushi_exl3");
const Trunk = trunk_mod.Trunk;

pub const Handle = registry.Arch;

/// The handle of a model built as `t.arch`.
pub fn handleOf(t: *Trunk) Handle {
    return registry.handleOf(t);
}

/// The handle of a trunk a test built by hand, which carries no arch stamp: it dispatches as the first built arch
/// wherever the tag decides, whatever its config names; every read through `trunk` is the same on each prong.
pub fn fixtureHandle(t: *Trunk) Handle {
    t.arch = std.enums.values(registry.Tag)[0];
    return handleOf(t);
}

/// The engine state the generic paths run on: an arch's `host(m)` when its `Model` is not that state, else the
/// `Model` itself.
pub fn trunk(h: Handle) *Trunk {
    return switch (h) {
        inline else => |m, t| if (comptime @hasDecl(registry.Mod(t), "host")) registry.Mod(t).host(m) else m,
    };
}

/// Ends one hold of the engine (`mlx.engine.hold`). The last hold's end is the last Cooler model's, whether its load
/// finished or failed: the EXL3 module's arrays on the engine's side go with it. Every hold ends here (lint
/// `engine-hold-end`): a second exit is how a failed load kept them.
pub fn engineHoldEnd() void {
    mlx.engine.unhold();
    if (!mlx.engine.holding()) sushi_exl3.dropHostArrays(1);
}

/// Models on MLX's side the registry holds (inference thread). The EXL3 module caches arrays per side for the process;
/// the last such model's unload gives MLX's back, as the last Cooler unload gives back the engine's.
const MlxModels = struct {
    n: usize = 0,

    fn held(self: *MlxModels) void {
        self.n += 1;
    }

    fn gone(self: *MlxModels) void {
        self.n -= 1;
        if (self.n == 0) sushi_exl3.dropHostArrays(0);
    }
};
var on_mlx: MlxModels = .{};

/// The handle the registry holds for the model built as `t`; `destroy` gives it back.
pub fn adopt(t: *Trunk) Handle {
    if (!mlx.engine.isStream(t.s)) on_mlx.held();
    return handleOf(t);
}

/// Frees the model of a handle `adopt` made, the arch's own state first, and the handle's allocation.
pub fn destroy(h: Handle, allocator: std.mem.Allocator) void {
    const on_engine = mlx.engine.isStream(stream(h));
    arch_life.deinit(h);
    switch (h) {
        inline else => |m| allocator.destroy(m),
    }
    if (!on_engine) on_mlx.gone();
}

pub fn stream(h: Handle) mlx.mlx_stream {
    return trunk(h).s;
}

/// The inference thread's turn for one unit of work on a model: the model of `h`, or before it has one the model
/// `cfg` will load. Beside a model whose load holds the engine's scope, the work of any other steps out of that
/// scope for its span (`engine.turnTo`); in a process no load holds, a turn is nothing.
pub fn turnFor(h: ?Handle, cfg: ?*const trunk_mod.ModelConfig) mlx.engine.Turn {
    if (!mlx.engine.holding()) return .{};
    return mlx.engine.turnTo(onEngine(h, cfg));
}

/// Whether the model of `h`, or before it has one the model `cfg` will load, runs on the engine's own stream.
pub fn onEngine(h: ?Handle, cfg: ?*const trunk_mod.ModelConfig) bool {
    return if (h) |handle| mlx.engine.isStream(stream(handle)) else if (cfg) |c| arch_life.ownStream(c) != null else false;
}

pub fn config(h: Handle) *const trunk_mod.ModelConfig {
    return &trunk(h).config;
}

/// Whether the model keeps decode state of its own, so admission lets one request in at a time.
pub fn ownsModuleDecodeState(h: Handle) bool {
    return arch_request.ownsModuleDecodeState(h);
}

/// Module archs run only their own MTP head; PLD and drafters stay off (`model_handle.moduleSpecWiring`).
pub fn moduleSpecWiring(h: Handle) bool {
    return trunk(h).moduleSpecWiring();
}

/// Whether the arch's own decode state rewinds across a speculative verify (`model_handle.moduleStateSpecRollback`).
pub fn moduleStateSpecRollback(h: Handle) bool {
    return trunk(h).moduleStateSpecRollback();
}

pub fn logitBias(h: Handle) []const @import("../logit_bias.zig").Bias {
    return trunk(h).logit_bias;
}

pub fn flushImatrix(h: Handle) void {
    trunk(h).flushImatrix();
}

/// Runs `run` on `items` in the turn of the side that owns them. Beside a load that holds the engine's scope the
/// items of each side are a run of their own, a side's items in the order they came: those on MLX (`own` false), then
/// those `onEngineOf` names. No model's work then runs in a turn of the other side. In a process no load holds, one
/// run of them all, `own` false.
pub fn perSide(comptime T: type, items: []T, comptime onEngineOf: fn (T) bool, ctx: anytype, comptime run: fn (@TypeOf(ctx), []T, bool) anyerror!void) !void {
    if (!mlx.engine.holding()) return run(ctx, items, false);
    std.sort.insertion(T, items, {}, struct {
        fn mlxFirst(_: void, a: T, b: T) bool {
            return !onEngineOf(a) and onEngineOf(b);
        }
    }.mlxFirst);
    var split: usize = 0;
    while (split < items.len and !onEngineOf(items[split])) split += 1;
    for ([_][]T{ items[0..split], items[split..] }, [_]bool{ false, true }) |side, own| {
        if (side.len == 0) continue;
        const turn = mlx.engine.turnTo(own);
        defer turn.end();
        try run(ctx, side, own);
    }
}

test "beside a held scope each side's items run in that side's turn, in their order; with no hold they run as one" {
    try mlx.requireGpu();
    const Item = struct { id: u8, own: bool };
    const Seen = struct {
        runs: usize = 0,
        ids: [2][4]u8 = @splat(@splat(0)),
        lens: [2]usize = .{ 0, 0 },
        own: [2]bool = .{ false, false },
        active: [2]bool = .{ false, false },
        rest: [2]bool = .{ false, false },

        fn run(self: *@This(), items: []Item, own: bool) anyerror!void {
            const at = self.runs;
            self.runs += 1;
            for (items, 0..) |it, i| self.ids[at][i] = it.id;
            self.lens[at] = items.len;
            self.own[at] = own;
            self.active[at] = mlx.engine.active;
            self.rest[at] = mlx.engine.mlx_at_rest.on.load(.acquire);
        }
    };
    const onEngineOf = struct {
        fn f(it: Item) bool {
            return it.own;
        }
    }.f;
    var items = [_]Item{ .{ .id = 1, .own = true }, .{ .id = 2, .own = false }, .{ .id = 3, .own = true }, .{ .id = 4, .own = false } };
    var alone: Seen = .{};
    try perSide(Item, &items, onEngineOf, &alone, Seen.run);
    try std.testing.expectEqual(@as(usize, 1), alone.runs);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, alone.ids[0][0..alone.lens[0]]);
    try std.testing.expect(!alone.own[0]);

    const was_enabled = mlx.engine.enabled;
    defer @import("../test_support.zig").restoreEngine(was_enabled);
    mlx.engine.hold();
    defer engineHoldEnd();
    const swallows = mlx.rest_swallows.load(.monotonic);
    var held: Seen = .{};
    try perSide(Item, &items, onEngineOf, &held, Seen.run);
    try std.testing.expectEqual(@as(usize, 2), held.runs);
    try std.testing.expectEqualSlices(u8, &.{ 2, 4 }, held.ids[0][0..held.lens[0]]);
    try std.testing.expectEqualSlices(u8, &.{ 1, 3 }, held.ids[1][0..held.lens[1]]);
    try std.testing.expectEqual([2]bool{ false, true }, held.own);
    try std.testing.expectEqual([2]bool{ false, true }, held.active);
    try std.testing.expectEqual([2]bool{ false, true }, held.rest);
    // Back as the hold left it.
    try std.testing.expect(mlx.engine.active and mlx.engine.mlx_at_rest.on.load(.acquire));
    try std.testing.expectEqual(swallows, mlx.rest_swallows.load(.monotonic));
}

test "the last model on MLX's side gives back the arrays the EXL3 module cached for that side; one still held keeps them" {
    try mlx.requireGpu();
    sushi_exl3.dropHostArrays(0);
    const active = struct {
        fn read() usize {
            var n: usize = 0;
            _ = mlx.mlx_get_active_memory(&n);
            return n;
        }
    }.read;
    var models: MlxModels = .{};
    const before = active();
    models.held();
    models.held();
    _ = try sushi_exl3.linear.gemmInputs(3, 70, 32);
    const cached = active();
    try std.testing.expect(cached > before);
    models.gone();
    try std.testing.expectEqual(cached, active());
    models.gone();
    try std.testing.expectEqual(before, active());
}

test "a handle reads back the trunk it was made from, its stream, config and logit bias" {
    inline for (comptime std.enums.values(registry.Tag)) |tag| {
        var t: Trunk = undefined;
        t.arch = tag;
        t.s = mlx.gpuStream();
        t.config = .{ .model_type = "handle-test" };
        t.logit_bias = &.{};
        const h = handleOf(&t);
        try std.testing.expectEqual(tag, std.meta.activeTag(h));
        try std.testing.expectEqual(&t, trunk(h));
        try std.testing.expectEqual(t.s.ctx, stream(h).ctx);
        try std.testing.expectEqualStrings("handle-test", config(h).model_type);
        try std.testing.expectEqual(@as(usize, 0), logitBias(h).len);
    }
}
