//! DeepSeek-V4.1-Flash's STQ routed experts, timed at the pack's own geometry: PAIR1 rate groups over 384 experts
//! (5120 -> 2304 -> 5120, top-6, clamped SwiGLU) on the decode chain, the verify widths and the prefill GEMMs, and the
//! DSpark stage's resident 128-expert n32 bank. Each rung gives the time per call on a chain of dependent calls and
//! the effective read rate over the picked experts' bytes. Run by `sushi-ubench`, never by a test.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const report = @import("mlx_host").report;
const exl3 = @import("expert_exl3.zig");
const kernels = @import("expert_exl3_kernels.zig");
const rate_groups = @import("rate_groups.zig");
const root = @import("root.zig");
const bench = @import("bench_util.zig");

pub const HIDDEN: usize = 5120;
pub const INTER: usize = 2304;
pub const LIMIT: c_int = 10;
pub const ROUTE_SETS: usize = 16;
/// One group per rate class, in the pack's order (highest rate first): `n` for gate, up and down, and the experts.
pub const Group = struct { n: [3]u32, experts: u32 };

pub const Layout = struct { key: []const u8, name: []const u8, groups: []const Group, topk: usize };

/// The plan's layers 2 and 5 (four groups), layer 23 (five: one expert's down at K1.625 under gate/up K2), the DSpark stage.
pub const LAYOUTS = [_]Layout{
    .{ .key = "L2", .name = "L2 40:5 32:17 26:37 24:325", .topk = 6, .groups = &.{
        .{ .n = .{ 40, 40, 40 }, .experts = 5 },  .{ .n = .{ 32, 32, 32 }, .experts = 17 },
        .{ .n = .{ 26, 26, 26 }, .experts = 37 }, .{ .n = .{ 24, 24, 24 }, .experts = 325 },
    } },
    .{ .key = "L5", .name = "L5 40:12 32:30 26:80 24:262", .topk = 6, .groups = &.{
        .{ .n = .{ 40, 40, 40 }, .experts = 12 }, .{ .n = .{ 32, 32, 32 }, .experts = 30 },
        .{ .n = .{ 26, 26, 26 }, .experts = 80 }, .{ .n = .{ 24, 24, 24 }, .experts = 262 },
    } },
    .{ .key = "L23", .name = "L23 40:51 32:75 32/26:1 26:109 24:148", .topk = 6, .groups = &.{
        .{ .n = .{ 40, 40, 40 }, .experts = 51 },  .{ .n = .{ 32, 32, 32 }, .experts = 75 },
        .{ .n = .{ 32, 32, 26 }, .experts = 1 },   .{ .n = .{ 26, 26, 26 }, .experts = 109 },
        .{ .n = .{ 24, 24, 24 }, .experts = 148 },
    } },
    .{ .key = "stage", .name = "stage n32 x128 top3", .topk = 3, .groups = &.{.{ .n = .{ 32, 32, 32 }, .experts = 128 }} },
};

pub const Mode = enum { uniform, highest, lowest, shared };

/// `rows * topk` distinct-per-row expert ids drawn per `mode`; returns the mean bytes read per call over the
/// route sets (every distinct expert once).
pub fn makeRoutes(rnd: std.Random, rb: *const rate_groups.Ragged, rows: usize, topk: usize, mode: Mode, out: *[ROUTE_SETS]mlx.mlx_array) f64 {
    var total: f64 = 0;
    for (out) |*r| {
        var ids: [256 * 6]u32 = undefined;
        var seen: [512]bool = @splat(false);
        for (0..rows) |row| {
            const ri = ids[row * topk ..][0..topk];
            if (mode == .shared and row > 0) {
                @memcpy(ri, ids[0..topk]);
                continue;
            }
            var k: usize = 0;
            while (k < topk) {
                const pool: u32 = switch (mode) {
                    .uniform => rb.experts,
                    .highest, .lowest => 12,
                    .shared => rb.experts,
                };
                var pick = rnd.uintLessThan(u32, pool);
                if (mode == .lowest) pick = rb.experts - 1 - pick;
                if (std.mem.indexOfScalar(u32, ri[0..k], pick) != null) continue;
                ri[k] = pick;
                k += 1;
                if (!seen[pick]) {
                    seen[pick] = true;
                    total += expertBytes(rb, pick);
                }
            }
        }
        r.* = mlx.mlx_array_new_data(&ids, &[_]c_int{@intCast(rows * topk)}, 1, .uint32);
    }
    return total / ROUTE_SETS;
}

/// Bytes of expert `e`: its three projections at their own n.
pub fn expertBytes(rb: *const rate_groups.Ragged, e: u32) f64 {
    const r = rb.rates[rb.groupOf(e)];
    const tiles: f64 = @floatFromInt((HIDDEN / 16) * (INTER / 16));
    return tiles * 2 * @as(f64, @floatFromInt(r[0] + r[1] + r[2]));
}

pub const Ctx = struct {
    rb: *const rate_groups.Ragged,
    routes: *const [ROUTE_SETS]mlx.mlx_array,
    scores: mlx.mlx_array,
    rows: c_int,
    topk: c_int,
    prefill: bool,
};

pub fn step(c: Ctx, s: mlx.mlx_stream, i: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    const slots = c.routes[i % ROUTE_SETS];
    return if (c.prefill)
        rate_groups.moePrefillClamped(s, x, c.rb, slots, c.scores, c.topk, LIMIT)
    else
        rate_groups.moeDecodeClamped(s, x, c.rb, slots, c.scores, c.topk, .float32, null, null, LIMIT);
}

pub const dsv41_experts: report.Bench = .{
    .name = "dsv41.experts",
    .scope = .dsv41,
    .what = "384-expert PAIR1 chain at the plan's layers: decode rows, verify widths, prefill, the DSpark bank",
    .unit = .us,
    .axes = &.{
        .{ .name = "layout", .values = &.{ "L2", "L5", "L23", "stage" }, .numeric = false },
        .{ .name = "rows", .values = &.{ "1", "2", "3", "4", "5", "6", "16", "256" } },
        .{ .name = "mode", .values = &.{ "uniform", "highest", "lowest", "shared" }, .numeric = false },
    },
    .arms = &.{"chain"},
    .ref = "chain",
    .run = runExperts,
};

const Sample = struct { s: mlx.mlx_stream, x0: mlx.mlx_array, c: Ctx, steps: usize };

/// Nanoseconds of one call: the chain less the link floor measured beside it.
fn sampleChain(smp: Sample, _: usize) anyerror!f64 {
    const ns = try bench.chainNs(smp.s, smp.x0, smp.steps, smp.c, step);
    const floor = try bench.chainNs(smp.s, smp.x0, smp.steps, {}, bench.linkOnly);
    return (@as(f64, @floatFromInt(ns)) - @as(f64, @floatFromInt(floor))) / @as(f64, @floatFromInt(smp.steps));
}

fn modeNamed(name: []const u8) Mode {
    return std.meta.stringToEnum(Mode, name).?;
}

fn runExperts(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    defer kernels.setDecodeParams(.mul1);
    var prng = std.Random.DefaultPrng.init(0xd54e);
    const rnd = prng.random();
    kernels.setDecodeParams(.{ .codebook = .pair1, .window = .w16 });
    for (LAYOUTS) |lay| {
        if (!listed(run.axis("layout"), lay.key)) continue;
        var banks: [8]root.Bank = undefined;
        for (lay.groups, 0..) |g, i| banks[i] = try bench.makeBank(rnd, HIDDEN, INTER, g.n, g.experts, 0.02, 0.9);
        var rb = try rate_groups.Ragged.init(s, banks[0..lay.groups.len]);
        defer rb.deinit();
        const st = rb.storage();
        const vec = mlx.mlx_vector_array_new_data(&st, st.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        for (banks[0..lay.groups.len]) |*b| bench.freeBank(b);
        const is_stage = lay.groups.len == 1;
        for (run.axis("rows")) |rows_text| {
            const rows = try std.fmt.parseInt(usize, rows_text, 10);
            if (rows == 256 and is_stage) continue;
            for (run.axis("mode")) |mode_text| {
                const mode = modeNamed(mode_text);
                if (is_stage and mode != .uniform) continue;
                if (mode == .shared and (rows > 6 or rows == 1)) continue;
                if (rows <= 6 and rows != 1 and (mode == .highest or mode == .lowest)) continue;
                if (rows == 256 and mode != .uniform) continue;
                var routes: [ROUTE_SETS]mlx.mlx_array = undefined;
                const bytes = makeRoutes(rnd, &rb, rows, lay.topk, mode, &routes);
                defer for (routes) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                const xh = try std.heap.page_allocator.alloc(u16, rows * HIDDEN);
                defer std.heap.page_allocator.free(xh);
                for (xh) |*v| v.* = @truncate(@as(u32, @bitCast((rnd.float(f32) * 2 - 1) * 0.5)) >> 16);
                const x0 = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), @intCast(HIDDEN) }, 2, .bfloat16);
                defer _ = mlx.mlx_array_free(x0);
                var scv: [256 * 6]f32 = undefined;
                for (&scv) |*v| v.* = 0.05 + rnd.float(f32) * 0.2;
                const scores = mlx.mlx_array_new_data(&scv, &[_]c_int{@intCast(rows * lay.topk)}, 1, .float32);
                defer _ = mlx.mlx_array_free(scores);
                const c: Ctx = .{ .rb = &rb, .routes = &routes, .scores = scores, .rows = @intCast(rows), .topk = @intCast(lay.topk), .prefill = rows > kernels.DECODE_ROWS_MAX };
                const label = try std.fmt.allocPrint(run.a, "{s} rows {d} {s}", .{ lay.key, rows, mode_text });
                try run.interleave(label, Sample{ .s = s, .x0 = x0, .c = c, .steps = run.steps(if (c.prefill) 4 else 16) }, sampleChain);
                const us = run.lastMedianNs("chain") / 1e3;
                try run.metric("chain", "MB read", "MB", bytes / 1e6);
                try run.metric("chain", "GB/s", "GB/s", bench.gbps(bytes, us));
                try run.metric("chain", "of roofline", "x", bench.gbps(bytes, us) / bench.ROOFLINE_GBPS);
            }
        }
    }
}

fn listed(values: []const []const u8, v: []const u8) bool {
    for (values) |x| if (std.mem.eql(u8, x, v)) return true;
    return false;
}
