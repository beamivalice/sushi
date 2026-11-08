//! Flash-Next's arm of `shared.shared_expert_fold`: one routed layer with the shared expert separate, folded into the
//! routed launches, and absent. Run by `sushi-ubench`, never by a test; the fold's bytes are asserted in `qwen4_shared.zig`.
const std = @import("std");
const mlx = @import("../mlx.zig");
const io_util = @import("../io_util.zig");
const report = @import("../ubench_report.zig");
const api = @import("sushi_exl3");
const shared = @import("../ubench_shared.zig");
const qwen = @import("qwen4_shared.zig");

const Arr = mlx.mlx_array;
const Ops = @import("../mlx_ops.zig").Ops;

pub const benches = [_]report.Bench{};

/// The Flash-Next arm of `shared.shared_expert_fold`, run when the model axis lists qwen4: every listed layout and kind.
pub fn sharedFold(run: *report.Run, s: mlx.mlx_stream, layouts: []const []const u8) !void {
    for (run.axis("shared")) |kind| for (layouts) |layout| try runQwen(run, s, std.mem.eql(u8, layout, "grouped"), std.mem.eql(u8, kind, "trellis"));
}

/// One Flash-Next layer: 2560 -> 640, top-10 of 64 experts, the shared expert with its gate logit.
const QwenRung = struct {
    s: mlx.mlx_stream,
    rows: usize,
    layers: usize,
    dec: api.format.Decode,
    weights: []const qwen.Weights,
    chains: []const qwen.Chain,
    ids: []const Arr,
    scores: []const Arr,
};

fn qwenSample(r: *const QwenRung, arm: usize) anyerror!f64 {
    var scope = Ops{ .s = r.s };
    defer scope.deinit();
    var prng = std.Random.DefaultPrng.init(0x913);
    var x = try qwen.randomBf16(&scope, shared.alloc, prng.random(), &.{ 1, @intCast(r.rows), qwen.hidden }, 4);
    try shared.eval(x);
    const sw = io_util.Stopwatch.init(shared.io());
    for (0..r.layers) |l| {
        const chain = r.chains[l];
        x = switch (@as(shared.FoldArm, @enumFromInt(arm))) {
            .separate => try qwen.separate(&scope, x, r.weights[l], chain, r.ids[l], r.scores[l], r.dec),
            .routed_only => try scope.own(switch (chain) {
                .bank => |b| try api.moe(r.s, x, b, r.ids[l], r.scores[l], r.dec, false),
                .ragged => |rb| try api.moeRagged(r.s, x, rb, r.ids[l], r.scores[l], r.dec, false, null, null),
            }),
            .fused => blk: {
                var sh = r.weights[l].shared;
                sh.logit = r.weights[l].logit;
                const got = switch (chain) {
                    .bank => |b| try api.qwen_fold.moeGatedSharedFolded(r.s, x, b, r.ids[l], r.scores[l], r.dec, sh, null),
                    .ragged => |rb| try api.rate_groups.moeGatedSharedFolded(r.s, x, rb, r.ids[l], r.scores[l], r.dec, sh, null),
                };
                break :blk try scope.own(got orelse return error.FoldDeclined);
            },
        };
    }
    try shared.eval(x);
    return @as(f64, @floatFromInt(sw.read())) / @as(f64, @floatFromInt(r.layers));
}

const bank_experts = 64;
/// The trunk's 6-bit trellis, in packed words per tile pair.
const trellis_bits: u32 = 96;
const grouped_counts = [_]c_int{ 20, 20, 24 };

fn runQwen(run: *report.Run, s: mlx.mlx_stream, grouped: bool, trellis: bool) !void {
    const layers = @min(run.steps(shared.default_layers), shared.max_layers);
    var prng = std.Random.DefaultPrng.init(913);
    const rnd = prng.random();
    var bops: [shared.max_layers]Ops = @splat(.{ .s = s });
    defer for (bops[0..layers]) |*b| b.deinit();
    var ragged: [shared.max_layers]api.Ragged = undefined;
    var made: usize = 0;
    defer for (ragged[0..made]) |*r| r.deinit();
    var weights: [shared.max_layers]qwen.Weights = undefined;
    var weights_made: usize = 0;
    defer for (weights[0..weights_made]) |*w| w.deinit(shared.alloc);
    var chains: [shared.max_layers]qwen.Chain = undefined;
    for (0..layers) |l| {
        weights[l] = try qwen.weights(&bops[l], shared.alloc, rnd, s, qwen.hidden, qwen.inter, 128, 1, if (trellis) trellis_bits else 0);
        weights_made += 1;
        if (grouped) {
            var parts: [3]api.Bank = undefined;
            for (&parts, grouped_counts, qwen.group_rates, 0..) |*p, count, r, g| p.* = try qwen.routedBank(&bops[l], count, r[0], r[2], 500 + 100 * @as(u64, @intCast(l)) + @as(u64, @intCast(g)));
            ragged[l] = try api.Ragged.init(s, &parts);
            made += 1;
            chains[l] = .{ .ragged = &ragged[l] };
        } else chains[l] = .{ .bank = try qwen.routedBank(&bops[l], bank_experts, 64, 64, 900 + @as(u64, @intCast(l))) };
    }
    const dec = if (grouped) api.format.Decode.tabled(.spec1, .w15) else qwen.test_dec;
    for (run.axis("rows")) |rows_text| {
        const rows = try std.fmt.parseInt(usize, rows_text, 10);
        var rops = Ops{ .s = s };
        defer rops.deinit();
        var ids: [shared.max_layers]Arr = undefined;
        var scores: [shared.max_layers]Arr = undefined;
        for (0..layers) |l| {
            const route = try qwen.routing(&rops, rnd, rows, 60);
            ids[l] = route.ids;
            scores[l] = route.scores;
        }
        const rung: QwenRung = .{ .s = s, .rows = rows, .layers = layers, .dec = dec, .weights = weights[0..layers], .chains = chains[0..layers], .ids = ids[0..layers], .scores = scores[0..layers] };
        const label = try std.fmt.allocPrint(run.a, "qwen4 {s}{s} rows {d}", .{ if (grouped) "grouped" else "bank", if (trellis) " trellis" else "", rows });
        try run.interleave(label, &rung, qwenSample);
    }
}
