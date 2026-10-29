//! The EXL3 kernel microbenches `sushi-ubench` runs, never a test: `dsv41.experts` (DeepSeek's PAIR1 rate groups at the
//! pack's own geometry), the shared decode, grouped-decode, rate-group, MoE-layer, prefill-GEMM and trunk benches, and
//! `glm5.routed_t1`. Each rung gives the time per call on a chain of dependent calls or a batch of independent ones.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const report = @import("mlx_host").report;
const io_util = @import("mlx_host").io_util;
const exl3 = @import("expert_exl3.zig");
const kernels = @import("expert_exl3_kernels.zig");
const rate_groups = @import("rate_groups.zig");
const root = @import("root.zig");
const bench = @import("bench_util.zig");
const group_layout = @import("layout.zig");

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

pub const dsv41_tail_routed: report.Bench = .{
    .name = "dsv41.tail_routed",
    .scope = .dsv41,
    .what = "the CED decoder tail's routed experts at the plan's layers: the prefill sorted GEMM against the multi-row decode chain, by rows",
    .unit = .us,
    .axes = &.{
        .{ .name = "layout", .values = &.{ "L2", "L5", "L23" }, .numeric = false },
        .{ .name = "rows", .values = &.{ "8", "16", "32", "64", "128", "256" } },
    },
    .arms = &.{ "gemm", "chain" },
    .ref = "gemm",
    .run = runTailRouted,
};

const TailRung = struct { s: mlx.mlx_stream, x0: mlx.mlx_array, c: Ctx, steps: usize };

fn tailStep(c: Ctx, chain: bool, s: mlx.mlx_stream, i: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    const slots = c.routes[i % ROUTE_SETS];
    return if (chain)
        rate_groups.moeDecodeClamped(s, x, c.rb, slots, c.scores, c.topk, .float32, null, null, LIMIT)
    else
        rate_groups.moePrefillClamped(s, x, c.rb, slots, c.scores, c.topk, LIMIT);
}

const TailStep = struct { c: Ctx, chain: bool };

fn tailLink(t: TailStep, s: mlx.mlx_stream, i: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    return tailStep(t.c, t.chain, s, i, x);
}

fn sampleTail(smp: TailRung, arm: usize) anyerror!f64 {
    const ns = try bench.chainNs(smp.s, smp.x0, smp.steps, TailStep{ .c = smp.c, .chain = arm == 1 }, tailLink);
    const floor = try bench.chainNs(smp.s, smp.x0, smp.steps, {}, bench.linkOnly);
    return (@as(f64, @floatFromInt(ns)) - @as(f64, @floatFromInt(floor))) / @as(f64, @floatFromInt(smp.steps));
}

fn runTailRouted(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    defer kernels.setDecodeParams(.mul1);
    var prng = std.Random.DefaultPrng.init(0xd54f);
    const rnd = prng.random();
    kernels.setDecodeParams(.{ .codebook = .pair1, .window = .w16 });
    for (LAYOUTS[0..3]) |lay| {
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
        for (run.axis("rows")) |rows_text| {
            const rows = try std.fmt.parseInt(usize, rows_text, 10);
            var routes: [ROUTE_SETS]mlx.mlx_array = undefined;
            _ = makeRoutes(rnd, &rb, rows, lay.topk, .uniform, &routes);
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
            const c: Ctx = .{ .rb = &rb, .routes = &routes, .scores = scores, .rows = @intCast(rows), .topk = @intCast(lay.topk), .prefill = true };
            const label = try std.fmt.allocPrint(run.a, "{s} rows {d}", .{ lay.key, rows });
            try run.interleave(label, TailRung{ .s = s, .x0 = x0, .c = c, .steps = run.steps(4) }, sampleTail);
        }
    }
}

fn listed(values: []const []const u8, v: []const u8) bool {
    for (values) |x| if (std.mem.eql(u8, x, v)) return true;
    return false;
}

// ---- shared.exl3_rate_groups: one Flash-Next MoE layer (2560 -> 640, E=512, top-10, bf16 rows, the gated shared
// expert) as uniform n32, the masked `moeGroups` and the one-dispatch grouped chain ----

const FLASH_HIDDEN: usize = 2560;
const FLASH_INTER: usize = 640;
const FLASH_EXPERTS: u32 = 512;
const FLASH_TOPK: usize = 10;
const FLASH_ROUTE_SETS: usize = 16;
const DECODE_STEPS: usize = 48;
const PREFILL_STEPS: usize = 4;
/// Prefill rungs run the first three layouts: the others probe the decode kernel's per-expert switch.
const PREFILL_LAYOUTS: usize = 3;

/// A layer layout: each group's (gate, up, down) n, and for routed slot j the groups it draws from.
const RgLayout = struct { key: []const u8, rates: []const [3]u32, picks: []const []const u8 };

fn rep(comptime n: usize, comptime v: []const u8) [n][]const u8 {
    return @splat(v);
}

const EVERY_N = blk: {
    var r: [21][3]u32 = undefined;
    for (&r, 0..) |*v, i| v.* = @splat(24 + 2 * i);
    break :blk r;
};

const PER_EXPERT = [_][3]u32{
    .{ 24, 24, 24 }, .{ 24, 24, 24 }, .{ 24, 24, 28 }, .{ 28, 28, 32 }, .{ 32, 32, 36 },
    .{ 36, 36, 40 }, .{ 40, 40, 40 }, .{ 48, 48, 48 }, .{ 48, 48, 48 }, .{ 64, 64, 64 },
    .{ 26, 26, 24 }, .{ 24, 24, 26 }, .{ 26, 26, 30 }, .{ 30, 30, 32 }, .{ 32, 32, 34 },
    .{ 36, 36, 44 }, .{ 44, 44, 40 }, .{ 48, 48, 44 }, .{ 44, 44, 56 }, .{ 56, 56, 64 },
};

const RG_LAYOUTS = [_]RgLayout{
    // K1.5|K3, gate_up 5:5, down shared.
    .{ .key = "g2", .rates = &.{ .{ 24, 24, 24 }, .{ 48, 48, 48 } }, .picks = &(rep(5, &.{0}) ++ rep(5, &.{1})) },
    // K1.5|K3 allocation: gate_up 5:5, down 3:7 as three pair classes.
    .{ .key = "pairs3", .rates = &.{ .{ 24, 24, 24 }, .{ 24, 24, 48 }, .{ 48, 48, 48 } }, .picks = &(rep(3, &.{0}) ++ rep(2, &.{1}) ++ rep(5, &.{2})) },
    // Per expert: slot j takes group j or its near-rate twin 10+j (the per-expert allocation's mix).
    .{ .key = "mix20", .rates = &PER_EXPERT, .picks = &.{ &.{ 0, 10 }, &.{ 1, 11 }, &.{ 2, 12 }, &.{ 3, 13 }, &.{ 4, 14 }, &.{ 5, 15 }, &.{ 6, 16 }, &.{ 7, 17 }, &.{ 8, 18 }, &.{ 9, 19 } } },
    // The per-expert mix with its n56-64 slot moved to the floor rates.
    .{ .key = "mix20_floor", .rates = &PER_EXPERT, .picks = &.{ &.{ 0, 10 }, &.{ 1, 11 }, &.{ 2, 12 }, &.{ 3, 13 }, &.{ 4, 14 }, &.{ 5, 15 }, &.{ 6, 16 }, &.{ 7, 17 }, &.{ 8, 18 }, &.{ 0, 10 } } },
    // The per-expert kernel's switch alone: the same rate set, every slot in its one n32 group.
    .{ .key = "mix20_n32", .rates = &.{
        .{ 24, 24, 24 }, .{ 24, 24, 24 }, .{ 24, 24, 28 }, .{ 28, 28, 32 }, .{ 32, 32, 32 },
        .{ 36, 36, 40 }, .{ 40, 40, 40 }, .{ 48, 48, 48 }, .{ 48, 48, 48 }, .{ 64, 64, 64 },
        .{ 26, 26, 24 }, .{ 24, 24, 26 }, .{ 26, 26, 30 }, .{ 30, 30, 32 }, .{ 32, 32, 34 },
        .{ 36, 36, 44 }, .{ 44, 44, 40 }, .{ 48, 48, 44 }, .{ 44, 44, 56 }, .{ 56, 56, 64 },
    }, .picks = &rep(10, &.{4}) },
    // Twenty-one classes, every even n from 24 to 64 (an eighth-step menu), every slot in the n32 group.
    .{ .key = "every21", .rates = &EVERY_N, .picks = &rep(10, &.{4}) },
    // The switch-free twin of the two above: twenty n32 groups, every slot in group 4.
    .{ .key = "n32x20", .rates = &@as([20][3]u32, @splat(.{ 32, 32, 32 })), .picks = &rep(10, &.{4}) },
};

const Arm = enum { uniform, masked, grouped };

fn bf16(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

/// `out.len` route sets of `rows * topk` experts, distinct within a row: each slot from its layout groups, or
/// uniform over `experts` when `rb` is null.
fn routeSets(rnd: std.Random, rows: usize, rb: ?*const rate_groups.Ragged, picks: []const []const u8, experts: u32, topk: usize, out: []mlx.mlx_array) !void {
    const ids = try std.heap.page_allocator.alloc(u32, rows * topk);
    defer std.heap.page_allocator.free(ids);
    for (out) |*r| {
        for (0..rows) |row| {
            const ri = ids[row * topk ..][0..topk];
            var k: usize = 0;
            while (k < topk) {
                const pick: u32 = if (rb) |b| blk: {
                    const choices = picks[k];
                    const g = choices[rnd.uintLessThan(usize, choices.len)];
                    break :blk b.base[g] + rnd.uintLessThan(u32, b.experts_of[g]);
                } else rnd.uintLessThan(u32, experts);
                if (std.mem.indexOfScalar(u32, ri[0..k], pick) != null) continue;
                ri[k] = pick;
                k += 1;
            }
        }
        r.* = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .uint32);
    }
}

/// A bf16 array of `count` values drawn from [lo, lo + span).
fn randomBf16(rnd: std.Random, dims: []const c_int, count: usize, lo: f32, span: f32) !mlx.mlx_array {
    const h = try std.heap.page_allocator.alloc(u16, count);
    defer std.heap.page_allocator.free(h);
    for (h) |*v| v.* = bf16(lo + rnd.float(f32) * span);
    return mlx.mlx_array_new_data(h.ptr, dims.ptr, @intCast(dims.len), .bfloat16);
}

/// sigmoid(logit) * shared + y, the separate tail every path not folding it runs.
fn addTail(s: mlx.mlx_stream, y: mlx.mlx_array, shared: mlx.mlx_array, logit: mlx.mlx_array) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, logit, s));
    var gated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gated);
    try mlx.check(mlx.mlx_multiply(&gated, sig, shared, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, y, gated, s));
    return out;
}

const RgRung = struct {
    s: mlx.mlx_stream,
    rows: usize,
    prefill: bool,
    x0: mlx.mlx_array,
    scores: mlx.mlx_array,
    shared: mlx.mlx_array,
    logit: mlx.mlx_array,
    uniform: root.Bank,
    uniform_routes: []const mlx.mlx_array,
    rb: *const rate_groups.Ragged,
    routes: []const mlx.mlx_array,
    dec: exl3.Decode,
};
const RgStep = struct { c: *const RgRung, arm: Arm };

fn rgStep(st: RgStep, s: mlx.mlx_stream, i: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    const c = st.c;
    const rows: c_int = @intCast(c.rows);
    const k: c_int = FLASH_TOPK;
    const hidden: c_int = FLASH_HIDDEN;
    const one = c.rows == 1;
    const dec = c.dec;
    const route = switch (st.arm) {
        .uniform => c.uniform_routes[i % c.uniform_routes.len],
        else => c.routes[i % c.routes.len],
    };
    if (!c.prefill and st.arm == .grouped) return groupedDecode(c, s, x, route);
    const ids = try rate_groups.reshaped(s, route, &.{ rows, 1, k });
    defer _ = mlx.mlx_array_free(ids);
    const x3 = try rate_groups.reshaped(s, x, &.{ rows, 1, hidden });
    defer _ = mlx.mlx_array_free(x3);
    if (c.prefill) return switch (st.arm) {
        .uniform => root.moe(s, x3, c.uniform, ids, c.scores, dec, false),
        .masked => root.moeGroups(s, x3, c.rb.views[0..c.rb.count], ids, c.scores, dec, false),
        .grouped => root.moeRagged(s, x3, c.rb, ids, c.scores, dec, false, null, null),
    };
    switch (st.arm) {
        .uniform => {
            if (one) return root.moeWithGatedShared(s, x3, c.uniform, ids, c.scores, dec, false, c.shared, c.logit);
            const y = try root.moe(s, x3, c.uniform, ids, c.scores, dec, true);
            defer _ = mlx.mlx_array_free(y);
            return addTail(s, y, c.shared, c.logit);
        },
        .masked => {
            const y = try root.moeGroups(s, x3, c.rb.views[0..c.rb.count], ids, c.scores, dec, !one);
            defer _ = mlx.mlx_array_free(y);
            return addTail(s, y, c.shared, c.logit);
        },
        .grouped => unreachable,
    }
}

/// The one-dispatch decode chain; one row folds the gated shared expert into it, more rows keep the separate tail.
fn groupedDecode(c: *const RgRung, s: mlx.mlx_stream, x: mlx.mlx_array, route: mlx.mlx_array) !mlx.mlx_array {
    const rows: c_int = @intCast(c.rows);
    const k: c_int = FLASH_TOPK;
    const hidden: c_int = FLASH_HIDDEN;
    kernels.setDecodeParams(c.dec);
    const flat_sc = try rate_groups.reshaped(s, c.scores, &.{rows * k});
    defer _ = mlx.mlx_array_free(flat_sc);
    if (c.rows == 1) {
        const sh = try rate_groups.reshaped(s, c.shared, &.{hidden});
        defer _ = mlx.mlx_array_free(sh);
        const lg = try rate_groups.reshaped(s, c.logit, &.{1});
        defer _ = mlx.mlx_array_free(lg);
        return rate_groups.moeDecode(s, x, c.rb, route, flat_sc, k, .bfloat16, sh, lg);
    }
    const y = try rate_groups.moeDecode(s, x, c.rb, route, flat_sc, k, .bfloat16, null, null);
    defer _ = mlx.mlx_array_free(y);
    return addTail(s, y, c.shared, c.logit);
}

const RgSample = struct { c: *const RgRung, steps: usize };

/// Nanoseconds of one layer call: the chain less the link floor measured beside it.
fn rgSample(smp: RgSample, arm: usize) anyerror!f64 {
    const c = smp.c;
    const ns = try bench.chainNs(c.s, c.x0, smp.steps, RgStep{ .c = c, .arm = @enumFromInt(arm) }, rgStep);
    const floor = try bench.chainNs(c.s, c.x0, smp.steps, {}, bench.linkOnly);
    return (@as(f64, @floatFromInt(ns)) - @as(f64, @floatFromInt(floor))) / @as(f64, @floatFromInt(smp.steps));
}

pub const exl3_rate_groups: report.Bench = .{
    .name = "shared.exl3_rate_groups",
    .scope = .shared,
    .what = "rate-grouped Flash-Next MoE layer against uniform n32 and the masked moeGroups: decode rows, prefill chunks",
    .unit = .us,
    .steps = DECODE_STEPS,
    .axes = &.{
        .{ .name = "layout", .values = &.{ "g2", "pairs3", "mix20", "mix20_floor", "mix20_n32", "every21", "n32x20" }, .numeric = false },
        .{ .name = "rows", .values = &.{ "1", "4", "512", "2048" } },
        .{ .name = "codebook", .values = &.{ "spec1", "pair1" }, .numeric = false },
    },
    .arms = &.{ "uniform", "masked", "grouped" },
    .ref = "uniform",
    .run = runRateGroups,
};

fn decodeNamed(name: []const u8) exl3.Decode {
    return if (std.mem.eql(u8, name, "spec1")) exl3.Decode.tabled(.spec1, .w15) else .{ .codebook = .pair1, .window = .w16 };
}

fn runRateGroups(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    defer kernels.setDecodeParams(.mul1);
    var prng = std.Random.DefaultPrng.init(0x7a11);
    const rnd = prng.random();
    var uniform = try bench.makeBank(rnd, FLASH_HIDDEN, FLASH_INTER, .{ 32, 32, 32 }, FLASH_EXPERTS, 0.9, 0.9);
    defer bench.freeBank(&uniform);
    for (RG_LAYOUTS, 0..) |lay, li| {
        if (!listed(run.axis("layout"), lay.key)) continue;
        var banks: [group_layout.max_groups]root.Bank = undefined;
        const per: u32 = FLASH_EXPERTS / @as(u32, @intCast(lay.rates.len));
        for (lay.rates, 0..) |r, gi| {
            const e = if (gi + 1 == lay.rates.len) FLASH_EXPERTS - per * @as(u32, @intCast(gi)) else per;
            banks[gi] = try bench.makeBank(rnd, FLASH_HIDDEN, FLASH_INTER, r, e, 0.9, 0.9);
        }
        var rb = try rate_groups.Ragged.init(s, banks[0..lay.rates.len]);
        defer rb.deinit();
        const st = rb.storage();
        const vec = mlx.mlx_vector_array_new_data(&st, st.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        for (banks[0..lay.rates.len]) |*b| bench.freeBank(b);
        for (run.axis("rows")) |rows_text| {
            const rows = try std.fmt.parseInt(usize, rows_text, 10);
            const prefill = rows > kernels.DECODE_ROWS_MAX;
            if (prefill and li >= PREFILL_LAYOUTS) continue;
            const sets: usize = if (prefill) 1 else FLASH_ROUTE_SETS;
            var routes_buf: [FLASH_ROUTE_SETS]mlx.mlx_array = undefined;
            var uroutes_buf: [FLASH_ROUTE_SETS]mlx.mlx_array = undefined;
            const routes = routes_buf[0..sets];
            const uroutes = uroutes_buf[0..sets];
            try routeSets(rnd, rows, null, &.{}, FLASH_EXPERTS, FLASH_TOPK, uroutes);
            try routeSets(rnd, rows, &rb, lay.picks, FLASH_EXPERTS, FLASH_TOPK, routes);
            defer for (routes, uroutes) |a, b| {
                _ = mlx.mlx_array_free(a);
                _ = mlx.mlx_array_free(b);
            };
            const ci: c_int = @intCast(rows);
            const hidden: c_int = FLASH_HIDDEN;
            const x0 = try randomBf16(rnd, if (prefill) &.{ ci, 1, hidden } else &.{ ci, hidden }, rows * FLASH_HIDDEN, -0.05, 0.1);
            defer _ = mlx.mlx_array_free(x0);
            const shared = try randomBf16(rnd, &.{ ci, 1, hidden }, rows * FLASH_HIDDEN, -0.05, 0.1);
            defer _ = mlx.mlx_array_free(shared);
            const logit = try randomBf16(rnd, &.{ ci, 1, 1 }, rows, -0.5, 0);
            defer _ = mlx.mlx_array_free(logit);
            const scores = try randomBf16(rnd, &.{ ci, 1, FLASH_TOPK }, rows * FLASH_TOPK, 0.05, 0.1);
            defer _ = mlx.mlx_array_free(scores);
            for (run.axis("codebook")) |cb| {
                const c: RgRung = .{ .s = s, .rows = rows, .prefill = prefill, .x0 = x0, .scores = scores, .shared = shared, .logit = logit, .uniform = uniform, .uniform_routes = uroutes, .rb = &rb, .routes = routes, .dec = decodeNamed(cb) };
                const label = try std.fmt.allocPrint(run.a, "{s} {s} rows {d}", .{ lay.key, cb, rows });
                try run.interleave(label, RgSample{ .c = &c, .steps = run.steps(if (prefill) PREFILL_STEPS else DECODE_STEPS) }, rgSample);
            }
        }
    }
}

// ---- shared.exl3_rate_groups_rows: one DeepSeek-V4.1 routed layer through the grouped prefill chain with every
// expert routed exactly r rows ----

pub const exl3_rate_groups_rows: report.Bench = .{
    .name = "shared.exl3_rate_groups_rows",
    .scope = .shared,
    .what = "DeepSeek-V4.1 layer (384 experts, four PAIR1 groups) through the grouped prefill chain by live rows per expert",
    .unit = .ms,
    .steps = PREFILL_STEPS,
    .axes = &.{
        .{ .name = "rows_per_expert", .values = &.{ "1", "4", "8", "12", "16", "17", "20", "24", "28", "32", "33", "40", "48" } },
    },
    .arms = &.{"grouped"},
    .ref = "grouped",
    .run = runRows,
};

const RowsRung = struct { s: mlx.mlx_stream, x0: mlx.mlx_array, rb: *const rate_groups.Ragged, routes: mlx.mlx_array, scores: mlx.mlx_array, steps: usize };

fn rowsStep(c: RowsRung, s: mlx.mlx_stream, _: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    return root.moeRagged(s, x, c.rb, c.routes, c.scores, .{ .codebook = .pair1, .window = .w16 }, false, null, null);
}

fn rowsSample(c: RowsRung, _: usize) anyerror!f64 {
    const ns = try bench.chainNs(c.s, c.x0, c.steps, c, rowsStep);
    const floor = try bench.chainNs(c.s, c.x0, c.steps, {}, bench.linkOnly);
    return (@as(f64, @floatFromInt(ns)) - @as(f64, @floatFromInt(floor))) / @as(f64, @floatFromInt(c.steps));
}

fn runRows(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    defer kernels.setDecodeParams(.mul1);
    var prng = std.Random.DefaultPrng.init(0xd541);
    const rnd = prng.random();
    const experts: u32 = 384;
    const topk: usize = 6;
    var banks: [4]root.Bank = undefined;
    for ([_]u32{ 24, 26, 32, 40 }, 0..) |n, g| banks[g] = try bench.makeBank(rnd, HIDDEN, INTER, .{ n, n, n }, experts / 4, 0.9, 0.9);
    var rb = try rate_groups.Ragged.init(s, &banks);
    defer rb.deinit();
    for (&banks) |*b| bench.freeBank(b);
    const st = rb.storage();
    const vec = mlx.mlx_vector_array_new_data(&st, st.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    for (run.axis("rows_per_expert")) |text| {
        const r = try std.fmt.parseInt(usize, text, 10);
        const rows = experts * r / topk;
        const ci: c_int = @intCast(rows);
        const x0 = try randomBf16(rnd, &.{ ci, 1, HIDDEN }, rows * HIDDEN, -0.05, 0.1);
        defer _ = mlx.mlx_array_free(x0);
        const ids = try std.heap.page_allocator.alloc(u32, rows * topk);
        defer std.heap.page_allocator.free(ids);
        for (ids, 0..) |*id, j| id.* = @intCast(j % experts);
        const routes = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ ci, 1, @intCast(topk) }, 3, .uint32);
        defer _ = mlx.mlx_array_free(routes);
        const scores = try randomBf16(rnd, &.{ ci, 1, @intCast(topk) }, rows * topk, 0.05, 0.1);
        defer _ = mlx.mlx_array_free(scores);
        const label = try std.fmt.allocPrint(run.a, "rows/expert {d} (rows {d})", .{ r, rows });
        try run.interleave(label, RowsRung{ .s = s, .x0 = x0, .rb = &rb, .routes = routes, .scores = scores, .steps = run.steps(PREFILL_STEPS) }, rowsSample);
        const ms = run.lastMedianNs("grouped") / 1e6;
        const flop = @as(f64, @floatFromInt(experts * r)) * 3 * 2 * @as(f64, @floatFromInt(HIDDEN * INTER));
        try run.metric("grouped", "us/expert", "us", ms * 1e3 / @as(f64, @floatFromInt(experts)));
        try run.metric("grouped", "TFLOP/s", "TFLOP/s", flop / (ms / 1e3) / 1e12);
    }
}

// ---- shared.shared_expert_fold, DeepSeek-V4.1 rungs: the shared expert beside the 384-expert routed chain ----

/// Arm indices into `shared.shared_expert_fold`'s arms.
const FoldArm = enum { separate, fused, routed_only };

const FoldCtx = struct {
    base: Ctx,
    bank: root.Bank,
    dec: exl3.Decode,
    zeros: mlx.mlx_array,
    ones: mlx.mlx_array,
    arm: FoldArm,
};

fn foldStep(c: FoldCtx, s: mlx.mlx_stream, i: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    const rows = c.base.rows;
    const pair1: exl3.Decode = .{ .codebook = .pair1, .window = .w16 };
    if (c.arm == .fused) {
        kernels.setDecodeParams(pair1);
        const se: rate_groups.SharedExpert = .{ .bank = c.bank, .dec = c.dec };
        return (try rate_groups.moeDecodeClampedSharedFused(s, x, c.base.rb, c.base.routes[i % ROUTE_SETS], c.base.scores, c.base.topk, .float32, se, LIMIT)) orelse error.SharedFuseDeclined;
    }
    var sy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sy);
    if (c.arm == .separate) {
        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        try mlx.check(mlx.mlx_reshape(&x3, x, &[_]c_int{ 1, rows, @intCast(HIDDEN) }, 3, s));
        kernels.setDecodeParams(c.dec);
        const sy3 = try root.moeClamped(s, x3, c.bank, c.zeros, c.ones, c.dec, LIMIT);
        defer _ = mlx.mlx_array_free(sy3);
        try mlx.check(mlx.mlx_reshape(&sy, sy3, &[_]c_int{ rows, @intCast(HIDDEN) }, 2, s));
    }
    kernels.setDecodeParams(pair1);
    return rate_groups.moeDecodeClamped(s, x, c.base.rb, c.base.routes[i % ROUTE_SETS], c.base.scores, c.base.topk, .float32, if (c.arm == .separate) sy else null, null, LIMIT);
}

const FoldSample = struct { s: mlx.mlx_stream, x0: mlx.mlx_array, steps: usize, c: FoldCtx };

/// Nanoseconds of one layer: the chain less the link floor measured beside it.
fn foldSample(smp: FoldSample, arm: usize) anyerror!f64 {
    var c = smp.c;
    c.arm = @enumFromInt(arm);
    const ns = try bench.chainNs(smp.s, smp.x0, smp.steps, c, foldStep);
    const floor = try bench.chainNs(smp.s, smp.x0, smp.steps, {}, bench.linkOnly);
    return (@as(f64, @floatFromInt(ns)) - @as(f64, @floatFromInt(floor))) / @as(f64, @floatFromInt(smp.steps));
}

/// The plan's layer 2 (four PAIR1 groups, top-6) with a SPEC1 n64 shared expert, at every `rows` rung in force.
pub fn sharedFoldDsv41(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    defer kernels.setDecodeParams(.mul1);
    var prng = std.Random.DefaultPrng.init(0xd54e);
    const rnd = prng.random();
    kernels.setDecodeParams(.{ .codebook = .pair1, .window = .w16 });
    const lay = LAYOUTS[0];
    var banks: [8]root.Bank = undefined;
    for (lay.groups, 0..) |g, i| banks[i] = try bench.makeBank(rnd, HIDDEN, INTER, g.n, g.experts, 0.02, 0.9);
    var rb = try rate_groups.Ragged.init(s, banks[0..lay.groups.len]);
    defer rb.deinit();
    const st = rb.storage();
    const vec = mlx.mlx_vector_array_new_data(&st, st.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    for (banks[0..lay.groups.len]) |*b| bench.freeBank(b);
    var sbank = try bench.makeBank(rnd, HIDDEN, INTER, .{ 64, 64, 64 }, 1, 0.02, 0.9);
    defer bench.freeBank(&sbank);
    for (run.axis("rows")) |rows_text| {
        const rows = try std.fmt.parseInt(usize, rows_text, 10);
        var routes: [ROUTE_SETS]mlx.mlx_array = undefined;
        _ = makeRoutes(rnd, &rb, rows, lay.topk, .uniform, &routes);
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
        var zi: [16]i32 = @splat(0);
        var one: [16]f32 = @splat(1);
        const zeros = mlx.mlx_array_new_data(&zi, &[_]c_int{ 1, @intCast(rows), 1 }, 3, .int32);
        defer _ = mlx.mlx_array_free(zeros);
        const ones = mlx.mlx_array_new_data(&one, &[_]c_int{ 1, @intCast(rows), 1 }, 3, .float32);
        defer _ = mlx.mlx_array_free(ones);
        const base: Ctx = .{ .rb = &rb, .routes = &routes, .scores = scores, .rows = @intCast(rows), .topk = @intCast(lay.topk), .prefill = false };
        const smp: FoldSample = .{ .s = s, .x0 = x0, .steps = run.steps(16), .c = .{ .base = base, .bank = sbank, .dec = exl3.Decode.tabled(.spec1, .w15), .zeros = zeros, .ones = ones, .arm = .separate } };
        var launches: [3]u32 = @splat(0);
        for (&launches, 0..) |*n, arm| {
            if (!run.hasArm(@tagName(@as(FoldArm, @enumFromInt(arm))))) continue;
            kernels.resetFusedDispatchCount();
            var c = smp.c;
            c.arm = @enumFromInt(arm);
            const y = try foldStep(c, s, 0, x0);
            try mlx.check(mlx.mlx_array_eval(y));
            _ = mlx.mlx_array_free(y);
            n.* = kernels.fusedDispatchCount();
        }
        const label = try std.fmt.allocPrint(run.a, "dsv41 grouped rows {d}", .{rows});
        try run.interleave(label, smp, foldSample);
        for (launches, 0..) |n, arm| {
            const name = @tagName(@as(FoldArm, @enumFromInt(arm)));
            if (run.hasArm(name)) try run.metric(name, "launches", "n", @floatFromInt(n));
        }
    }
}

const MCG_W15: exl3.Decode = .{ .codebook = .mcg, .window = .w15 };

const RATES_TEXT = [_][]const u8{
    "24", "26", "28", "30", "32", "34", "36", "38", "40", "42", "44", "46", "48", "50", "52", "54",
    "56", "58", "60", "62", "64", "72", "80", "96", "112", "128",
};

pub const decode_chain: report.Bench = .{
    .name = "shared.exl3_decode_chain",
    .scope = .shared,
    .what = "EXL3 decode chain (pair GEMV, middle, down, reduce) at Flash-Next geometry: a rate against n32, n48 and n64, 1 to 4 rows",
    .unit = .us,
    .steps = 32,
    .axes = &.{
        .{ .name = "codebook", .values = &.{ "mcg", "spec1" }, .numeric = false },
        .{ .name = "rate", .values = &RATES_TEXT, .numeric = false },
        .{ .name = "rows", .values = &.{ "1", "3", "4" } },
    },
    .arms = &.{ "n32", "n48", "n64", "rate" },
    .ref = "n48",
    .run = runDecodeChain,
};

const ChainStep = struct { f: *const kernels.MimoMoeFixture };

fn chainStep(c: ChainStep, s: mlx.mlx_stream, _: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    return kernels.decodeChain(s, c.f, x);
}

/// Nanoseconds of one chain step over fixture `f`: the chain less the link floor measured beside it.
fn stepNs(s: mlx.mlx_stream, f: *const kernels.MimoMoeFixture, steps: usize) !f64 {
    const x0 = f.arrays[8];
    const ns = try bench.chainNs(s, x0, steps, ChainStep{ .f = f }, chainStep);
    const floor = try bench.chainNs(s, x0, steps, {}, bench.linkOnly);
    return (@as(f64, @floatFromInt(ns)) - @as(f64, @floatFromInt(floor))) / @as(f64, @floatFromInt(steps));
}

const ChainSample = struct { s: mlx.mlx_stream, dec: exl3.Decode, f: [4]*const kernels.MimoMoeFixture, steps: usize };

fn sampleDecodeChain(smp: ChainSample, arm: usize) anyerror!f64 {
    kernels.setDecodeParams(smp.dec);
    return stepNs(smp.s, smp.f[arm], smp.steps);
}

fn flashBlock(alloc: std.mem.Allocator, dec: exl3.Decode, n: u32, rows: usize) !kernels.CarriedBlock {
    return kernels.carriedBlock(alloc, .{ .e = FLASH_EXPERTS, .hidden = FLASH_HIDDEN, .inter = FLASH_INTER, .topk = FLASH_TOPK, .rows = rows, .rate = .{ .n = n }, .dec = dec, .seed = 5200 + n + 97 * rows, .banks = kernels.MIMO_BANKS, .x_scale = 3 }, 0);
}

fn runDecodeChain(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    defer kernels.setDecodeParams(.mul1);
    const bases = [3]u32{ 32, 48, 64 };
    for (run.axis("codebook")) |cb_text| {
        const dec: exl3.Decode = if (std.mem.eql(u8, cb_text, "spec1")) kernels.SPEC1_W15 else MCG_W15;
        try dec.requireTable();
        for (run.axis("rows")) |rows_text| {
            const rows = try std.fmt.parseInt(usize, rows_text, 10);
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            var base: [3]kernels.CarriedBlock = undefined;
            for (&base, bases) |*b, n| b.* = try flashBlock(arena.allocator(), dec, n, rows);
            defer for (&base) |*b| b.f.deinit();
            for (run.axis("rate")) |rate_text| {
                const n = try std.fmt.parseInt(u32, rate_text, 10);
                var rung_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                defer rung_arena.deinit();
                var block = try flashBlock(rung_arena.allocator(), dec, n, rows);
                defer block.f.deinit();
                const label = try std.fmt.allocPrint(run.a, "{s} n{d} rows {d}", .{ cb_text, n, rows });
                const smp: ChainSample = .{ .s = s, .dec = dec, .f = .{ &base[0].f, &base[1].f, &base[2].f, &block.f }, .steps = run.steps(32) };
                try run.interleave(label, smp, sampleDecodeChain);
                const bytes: f64 = @as(f64, @floatFromInt(block.used * 3 * (FLASH_HIDDEN / 16) * (FLASH_INTER / 16) * n)) * 2;
                const us = run.lastMedianNs("rate") / 1e3;
                try run.metric("rate", "MB read", "MB", bytes / 1e6);
                try run.metric("rate", "GB/s", "GB/s", bench.gbps(bytes, us));
            }
        }
    }
}

const GroupGeom = struct { name: []const u8, hidden: usize, inter: usize, topk: usize, e: usize };
const GROUP_GEOMS = [_]GroupGeom{
    .{ .name = "flash", .hidden = 2560, .inter = 640, .topk = 10, .e = 512 },
    .{ .name = "deepseek", .hidden = 5120, .inter = 2304, .topk = 6, .e = 384 },
};
const GroupRate = struct { name: []const u8, n: u32, dec: exl3.Decode };
const GROUP_RATES = [_]GroupRate{
    .{ .name = "mcg32", .n = 32, .dec = MCG_W15 },
    .{ .name = "mcg40", .n = 40, .dec = MCG_W15 },
    .{ .name = "mcg48", .n = 48, .dec = MCG_W15 },
    .{ .name = "mcg64", .n = 64, .dec = MCG_W15 },
    .{ .name = "stq36", .n = 36, .dec = kernels.SPEC1_W15 },
    .{ .name = "stq42", .n = 42, .dec = kernels.SPEC1_W15 },
};

pub const grouped_decode: report.Bench = .{
    .name = "shared.exl3_grouped_decode",
    .scope = .shared,
    .what = "EXL3 decode chain at 1 to 8 rows with verify rows sharing expert reads: ungrouped, prepared, pairs and the member capacity sized to the rows",
    .unit = .us,
    .steps = 32,
    .axes = &.{
        .{ .name = "geometry", .values = &.{ "flash", "deepseek" }, .numeric = false },
        .{ .name = "rate", .values = &.{ "mcg32", "mcg40", "mcg48", "mcg64", "stq36", "stq42" }, .numeric = false },
        .{ .name = "rows", .values = &.{ "1", "2", "3", "4", "6", "8" } },
        .{ .name = "carry", .values = &.{ "0", "0.5" }, .numeric = false },
    },
    .arms = &kernels.GROUP_ARMS,
    .ref = "fused",
    .run = runGroupedDecode,
};

const GroupSample = struct { s: mlx.mlx_stream, f: *const kernels.MimoMoeFixture, dec: exl3.Decode, rows: usize, steps: usize };

fn sampleGrouped(smp: GroupSample, arm: usize) anyerror!f64 {
    kernels.setDecodeParams(smp.dec);
    kernels.setGroupArm(kernels.groupArm(smp.rows, arm));
    return stepNs(smp.s, smp.f, smp.steps);
}

fn runGroupedDecode(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    defer {
        kernels.setDecodeParams(.mul1);
        kernels.clearGroupArm();
    }
    for (GROUP_GEOMS) |geom| {
        if (!listed(run.axis("geometry"), geom.name)) continue;
        for (GROUP_RATES) |rate| {
            if (!listed(run.axis("rate"), rate.name)) continue;
            try rate.dec.requireTable();
            for (run.axis("rows")) |rows_text| {
                const rows = try std.fmt.parseInt(usize, rows_text, 10);
                for (run.axis("carry")) |carry_text| {
                    if (rows == 1 and !std.mem.eql(u8, carry_text, "0")) continue;
                    const carry = try std.fmt.parseFloat(f32, carry_text);
                    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                    defer arena.deinit();
                    const alloc = arena.allocator();
                    const real: kernels.MimoMoeCase = .{ .e = geom.e, .hidden = geom.hidden, .inter = geom.inter, .topk = geom.topk, .rows = rows, .rate = .{ .n = rate.n }, .dec = rate.dec, .seed = 14000 + rate.n, .banks = kernels.MIMO_BANKS, .x_scale = 3 };
                    var block = try kernels.carriedBlock(alloc, real, carry);
                    defer block.f.deinit();
                    const counts = try alloc.alloc(u32, block.used);
                    @memset(counts, 0);
                    for (block.slots) |v| counts[v] += 1;
                    kernels.setDecodeParams(rate.dec);
                    for (kernels.GROUP_ARMS, 0..) |name, i| {
                        if (!run.hasArm(name)) continue;
                        const arm = kernels.groupArm(rows, i);
                        kernels.setGroupArm(arm);
                        kernels.grouped_widest = 0;
                        const y = try kernels.decodeChain(s, &block.f, block.f.arrays[8]);
                        defer _ = mlx.mlx_array_free(y);
                        try mlx.check(mlx.mlx_array_eval(y));
                        if (rows > 1 and kernels.grouped_widest != (arm.group orelse 0)) run.invalid("an arm did not engage its member capacity");
                    }
                    const label = try std.fmt.allocPrint(run.a, "{s} {s} rows {d} carry {s}", .{ geom.name, rate.name, rows, carry_text });
                    const smp: GroupSample = .{ .s = s, .f = &block.f, .dec = rate.dec, .rows = rows, .steps = run.steps(32) };
                    try run.interleave(label, smp, sampleGrouped);
                    try run.metric("fused", "experts", "n", @floatFromInt(block.used));
                    try run.metric("fused", "max members", "n", @floatFromInt(std.mem.max(u32, counts)));
                }
            }
        }
    }
}

const LAYER_ROUTES = 8;
const LAYER_RATE: u32 = 64;
const AFFINE_GROUP = 64;
const AFFINE_BITS = 4;

pub const moe_layer: report.Bench = .{
    .name = "shared.exl3_moe_layer",
    .scope = .shared,
    .what = "a Flash-Next MoE layer (E512, 2560 -> 640, top-10, K4) on each codebook, against stock MLX affine-4 gather_qmm x3, decode rows to prefill",
    .unit = .us,
    .steps = 8,
    .axes = &.{.{ .name = "rows", .values = &.{ "1", "2", "3", "4", "16", "512", "2048" } }},
    .arms = &.{ "mul1", "mcg", "std1", "spec1", "pair1", "affine" },
    .ref = "mcg",
    .run = runMoeLayer,
};

const LAYER_DECODES = [5]exl3.Decode{
    .mul1,
    MCG_W15,
    exl3.Decode.tabled(.std1, .w15),
    kernels.SPEC1_W15,
    .{ .codebook = .pair1, .window = .w16 },
};

const Affine = struct { gate: [3]mlx.mlx_array, down: [3]mlx.mlx_array };

/// A random dense bank quantized to affine-4: MLX's own expert layout at the same shape.
fn affineBank(s: mlx.mlx_stream, in: c_int, out: c_int) ![3]mlx.mlx_array {
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_random_normal(&dense, &[_]c_int{ @intCast(FLASH_EXPERTS), out, in }, 3, .float16, 0, 1, .{ .ctx = null }, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, dense, mlx.mlx_optional_int.some(AFFINE_GROUP), mlx.mlx_optional_int.some(AFFINE_BITS), "affine", .{ .ctx = null }, s));
    var parts: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    for (&parts, 0..) |*p, i| try mlx.check(mlx.mlx_vector_array_get(p, triple, i));
    try mlx.check(mlx.mlx_array_eval(parts[0]));
    return parts;
}

fn freeAll(arrays: []const mlx.mlx_array) void {
    for (arrays) |a| _ = mlx.mlx_array_free(a);
}

const LayerCtx = struct {
    s: mlx.mlx_stream,
    bank: root.Bank,
    x0: mlx.mlx_array,
    routes: *const [LAYER_ROUTES]mlx.mlx_array,
    scores: mlx.mlx_array,
    steps: usize,
    rows: usize,
    affine: ?Affine,
    xrep: mlx.mlx_array,
    xmid: mlx.mlx_array,
    sorted: mlx.mlx_array,
};

const LayerStep = struct { c: *const LayerCtx, dec: exl3.Decode };

fn layerStep(c: LayerStep, s: mlx.mlx_stream, i: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    return root.moe(s, x, c.c.bank, c.c.routes[i % LAYER_ROUTES], c.c.scores, c.dec, false);
}

fn gatherAffine(s: mlx.mlx_stream, x: mlx.mlx_array, w: [3]mlx.mlx_array, sorted: mlx.mlx_array, is_sorted: bool) !void {
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_gather_qmm(&q, x, w[0], w[1], w[2], .{ .ctx = null }, sorted, true, mlx.mlx_optional_int.some(AFFINE_GROUP), mlx.mlx_optional_int.some(AFFINE_BITS), "affine", is_sorted, s));
    try mlx.check(mlx.mlx_array_eval(q));
}

fn sampleLayer(c: LayerCtx, arm: usize) anyerror!f64 {
    if (arm == LAYER_DECODES.len) {
        const aff = c.affine.?;
        const io = std.Io.Threaded.global_single_threaded.io();
        var sw = io_util.Stopwatch.init(io);
        for (0..c.steps) |_| {
            try gatherAffine(c.s, c.xrep, aff.gate, c.sorted, c.rows > kernels.DECODE_ROWS_MAX);
            try gatherAffine(c.s, c.xrep, aff.gate, c.sorted, c.rows > kernels.DECODE_ROWS_MAX);
            try gatherAffine(c.s, c.xmid, aff.down, c.sorted, c.rows > kernels.DECODE_ROWS_MAX);
        }
        return @as(f64, @floatFromInt(sw.read())) / @as(f64, @floatFromInt(c.steps));
    }
    const dec = LAYER_DECODES[arm];
    kernels.setDecodeParams(dec);
    const ns = try bench.chainNs(c.s, c.x0, c.steps, LayerStep{ .c = &c, .dec = dec }, layerStep);
    const floor = try bench.chainNs(c.s, c.x0, c.steps, {}, bench.linkOnly);
    return (@as(f64, @floatFromInt(ns)) - @as(f64, @floatFromInt(floor))) / @as(f64, @floatFromInt(c.steps));
}

fn runMoeLayer(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    defer kernels.setDecodeParams(.mul1);
    var prng = std.Random.DefaultPrng.init(0x1a7e);
    const rnd = prng.random();
    for (LAYER_DECODES) |d| try d.requireTable();
    var bank = try bench.makeBank(rnd, FLASH_HIDDEN, FLASH_INTER, .{ LAYER_RATE, LAYER_RATE, LAYER_RATE }, FLASH_EXPERTS, 0.02, 0.9);
    defer bench.freeBank(&bank);
    var affine: ?Affine = null;
    defer if (affine) |a| {
        freeAll(&a.gate);
        freeAll(&a.down);
    };
    if (run.hasArm("affine")) affine = .{
        .gate = try affineBank(s, FLASH_HIDDEN, FLASH_INTER),
        .down = try affineBank(s, FLASH_INTER, FLASH_HIDDEN),
    };
    for (run.axis("rows")) |rows_text| {
        const rows = try std.fmt.parseInt(usize, rows_text, 10);
        const slots = rows * FLASH_TOPK;
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var routes: [LAYER_ROUTES]mlx.mlx_array = undefined;
        var first_sorted: []i32 = &.{};
        for (&routes, 0..) |*r, set| {
            const ids = try a.alloc(u32, slots);
            for (0..rows) |row| {
                const ri = ids[row * FLASH_TOPK ..][0..FLASH_TOPK];
                var k: usize = 0;
                while (k < FLASH_TOPK) {
                    const pick = rnd.uintLessThan(u32, FLASH_EXPERTS);
                    if (std.mem.indexOfScalar(u32, ri[0..k], pick) != null) continue;
                    ri[k] = pick;
                    k += 1;
                }
            }
            r.* = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(rows), FLASH_TOPK }, 3, .uint32);
            if (set == 0) {
                first_sorted = try a.alloc(i32, slots);
                for (first_sorted, ids) |*d, v| d.* = @intCast(v);
                std.mem.sort(i32, first_sorted, {}, std.sort.asc(i32));
            }
        }
        defer for (routes) |r| {
            _ = mlx.mlx_array_free(r);
        };
        const sc = try a.alloc(f32, slots);
        @memset(sc, 1.0 / @as(f32, FLASH_TOPK));
        const scores = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{ 1, @intCast(rows), FLASH_TOPK }, 3, .float32);
        defer _ = mlx.mlx_array_free(scores);
        const xh = try a.alloc(u16, rows * FLASH_HIDDEN);
        for (xh) |*v| v.* = @truncate(@as(u32, @bitCast((rnd.float(f32) * 2 - 1) * 0.5)) >> 16);
        const x0 = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ 1, @intCast(rows), FLASH_HIDDEN }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(x0);
        const sorted = mlx.mlx_array_new_data(first_sorted.ptr, &[_]c_int{@intCast(slots)}, 1, .int32);
        defer _ = mlx.mlx_array_free(sorted);
        var xrep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xrep);
        var xmid = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xmid);
        try mlx.check(mlx.mlx_random_normal(&xrep, &[_]c_int{ @intCast(slots), 1, FLASH_HIDDEN }, 3, .float16, 0, 0.1, .{ .ctx = null }, s));
        try mlx.check(mlx.mlx_random_normal(&xmid, &[_]c_int{ @intCast(slots), 1, FLASH_INTER }, 3, .float16, 0, 0.1, .{ .ctx = null }, s));
        const ctx: LayerCtx = .{ .s = s, .bank = bank, .x0 = x0, .routes = &routes, .scores = scores, .steps = run.steps(if (rows > kernels.DECODE_ROWS_MAX) 3 else 16), .rows = rows, .affine = affine, .xrep = xrep, .xmid = xmid, .sorted = sorted };
        const label = try std.fmt.allocPrint(run.a, "rows {d}", .{rows});
        try run.interleave(label, ctx, sampleLayer);
    }
}

const GemmShape = struct { name: []const u8, in: c_int, out: c_int, experts: c_int, dec: exl3.Decode };
const MIMO_W12: exl3.Decode = .{ .codebook = .mcg, .window = .w12 };
const GEMM_SHAPES = [_]GemmShape{
    .{ .name = "flash_gate_up", .in = 2560, .out = 640, .experts = 512, .dec = MCG_W15 },
    .{ .name = "flash_down", .in = 640, .out = 2560, .experts = 512, .dec = MCG_W15 },
    .{ .name = "mimo_gate_up", .in = 4096, .out = 2048, .experts = 256, .dec = MIMO_W12 },
    .{ .name = "mimo_down", .in = 2048, .out = 4096, .experts = 256, .dec = MIMO_W12 },
};

pub const prefill_gemm: report.Bench = .{
    .name = "shared.exl3_prefill_gemm",
    .scope = .shared,
    .what = "the sorted prefill GEMM of one projection: the served body against the reference body (NAX reference, else the scalar body), and a rate against n32 and n64",
    .unit = .us,
    .steps = 4,
    .axes = &.{
        .{ .name = "shape", .values = &.{ "flash_gate_up", "flash_down", "mimo_gate_up", "mimo_down" }, .numeric = false },
        .{ .name = "rate", .values = &.{ "24", "32", "36", "40", "42", "48", "64", "80", "96", "128" }, .numeric = false },
        .{ .name = "slots", .values = &.{ "170", "640", "2050", "5120", "20480" } },
    },
    .arms = &.{ "served", "reference", "n32", "n64" },
    .ref = "served",
    .opts = &.{
        .{ .name = "nax", .help = "off times the simdgroup-matrix and scalar bodies on a NAX chip" },
        .{ .name = "counts", .help = "a file of per-layer expert counts ([exl3-counts] lines): one rung per layer for the mimo shapes in place of slots" },
    },
    .run = runPrefillGemm,
};

const GemmCtx = struct {
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    eids: mlx.mlx_array,
    tab: kernels.WindowTable,
    trellis: [4]mlx.mlx_array,
    dec: exl3.Decode,
    nax: bool,
    calls: usize,
};

/// The reference body is the NAX reference where NAX runs, else the scalar body the simdgroup-matrix body replaced.
fn setBody(reference: bool, nax: bool) void {
    kernels.nax_reference_override = reference and nax;
    kernels.gemm_simdmat_force = if (reference and !nax) false else null;
}

fn gemmCall(c: GemmCtx, arm: usize) !mlx.mlx_array {
    setBody(arm == 1, c.nax);
    kernels.setDecodeParams(c.dec);
    return kernels.innerGemmSortedTable(c.s, c.x, c.trellis[arm], c.eids, 32, true, c.tab);
}

fn sampleGemm(c: GemmCtx, arm: usize) anyerror!f64 {
    const outs = try std.heap.page_allocator.alloc(mlx.mlx_array, c.calls);
    defer std.heap.page_allocator.free(outs);
    var made: usize = 0;
    defer for (outs[0..made]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (outs) |*o| {
        o.* = try gemmCall(c, arm);
        made += 1;
        _ = mlx.mlx_vector_array_append_value(vec, o.*);
    }
    const io = std.Io.Threaded.global_single_threaded.io();
    var sw = io_util.Stopwatch.init(io);
    try mlx.check(mlx.mlx_eval(vec));
    return @as(f64, @floatFromInt(sw.read())) / @as(f64, @floatFromInt(c.calls));
}

fn randomTrellis(rnd: std.Random, experts: c_int, in: c_int, out: c_int, n: u32) !mlx.mlx_array {
    const it = @divExact(in, 16);
    const ot = @divExact(out, 16);
    const words = try std.heap.page_allocator.alloc(u16, @as(usize, @intCast(experts * it * ot)) * n);
    defer std.heap.page_allocator.free(words);
    bench.randomWords(rnd, words);
    return mlx.mlx_array_new_data(words.ptr, &[_]c_int{ experts, it, ot, @intCast(n) }, 4, .uint16);
}

/// Per-layer expert counts from `[exl3-counts]` log lines (`... ] c0,c1,..`, 256 counts a line).
fn readCounts(a: std.mem.Allocator, path: []const u8) ![]const [256]u32 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    var out: std.ArrayList([256]u32) = .empty;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const at = std.mem.indexOf(u8, line, "] ") orelse continue;
        var c: [256]u32 = @splat(0);
        var it = std.mem.tokenizeScalar(u8, line[at + 2 ..], ',');
        var i: usize = 0;
        while (it.next()) |v| : (i += 1) {
            if (i == 256) break;
            c[i] = std.fmt.parseInt(u32, std.mem.trim(u8, v, " \r"), 10) catch 0;
        }
        try out.append(a, c);
    }
    return out.items;
}

const COUNT_LAYERS = [_]usize{ 0, 10, 23, 35, 46 };

fn runPrefillGemm(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    defer {
        kernels.setDecodeParams(.mul1);
        setBody(false, false);
        kernels.gemm_nax_failed = false;
    }
    if (run.opt("nax")) |v| {
        if (!std.mem.eql(u8, v, "off")) return error.BadOpt;
        kernels.gemm_nax_failed = true;
    }
    const nax = kernels.gemmNaxOn();
    var prng = std.Random.DefaultPrng.init(0x6e33);
    const rnd = prng.random();
    var counts: []const [256]u32 = &.{};
    if (run.opt("counts")) |path| counts = try readCounts(run.a, path);
    for (GEMM_SHAPES) |sh| {
        if (!listed(run.axis("shape"), sh.name)) continue;
        try sh.dec.requireTable();
        const mimo = sh.experts == 256;
        const bank32 = try randomTrellis(rnd, sh.experts, sh.in, sh.out, 32);
        defer _ = mlx.mlx_array_free(bank32);
        const bank64 = try randomTrellis(rnd, sh.experts, sh.in, sh.out, 64);
        defer _ = mlx.mlx_array_free(bank64);
        for (run.axis("rate")) |rate_text| {
            const n = try std.fmt.parseInt(u32, rate_text, 10);
            const bank = try randomTrellis(rnd, sh.experts, sh.in, sh.out, n);
            defer _ = mlx.mlx_array_free(bank);
            const use_counts = mimo and counts.len > 0;
            const rungs = if (use_counts) COUNT_LAYERS.len else run.axis("slots").len;
            for (0..rungs) |ri| {
                var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
                defer arena.deinit();
                const a = arena.allocator();
                var label: []const u8 = undefined;
                var ids: []u32 = undefined;
                if (use_counts) {
                    const li = COUNT_LAYERS[ri];
                    if (li >= counts.len) continue;
                    var total: usize = 0;
                    for (counts[li]) |c| total += c;
                    ids = try a.alloc(u32, total);
                    var at: usize = 0;
                    for (counts[li], 0..) |c, e| for (0..c) |_| {
                        ids[at] = @intCast(e);
                        at += 1;
                    };
                    label = try std.fmt.allocPrint(run.a, "{s} n{d} layer {d} ({d} slots)", .{ sh.name, n, li + 1, total });
                } else {
                    const slots = try std.fmt.parseInt(usize, run.axis("slots")[ri], 10);
                    ids = try a.alloc(u32, slots);
                    for (ids) |*v| v.* = rnd.uintLessThan(u32, @intCast(sh.experts));
                    std.mem.sort(u32, ids, {}, std.sort.asc(u32));
                    label = try std.fmt.allocPrint(run.a, "{s} n{d} slots {d}", .{ sh.name, n, slots });
                }
                const nslots: c_int = @intCast(ids.len);
                const eids = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{nslots}, 1, .uint32);
                defer _ = mlx.mlx_array_free(eids);
                const xh = try a.alloc(u16, ids.len * @as(usize, @intCast(sh.in)));
                for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
                const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ nslots, sh.in }, 2, .float16);
                defer _ = mlx.mlx_array_free(x);
                const tab = try kernels.gemmWindowTable(s, eids, nslots, 32, true);
                defer _ = mlx.mlx_array_free(tab.starts);
                defer _ = mlx.mlx_array_free(tab.nlives);
                const ctx: GemmCtx = .{ .s = s, .x = x, .eids = eids, .tab = tab, .trellis = .{ bank, bank, bank32, bank64 }, .dec = sh.dec, .nax = nax, .calls = run.steps(4) };
                if (nax) try checkBodies(run, ctx);
                try run.interleave(label, ctx, sampleGemm);
            }
        }
    }
}

/// The NAX reference body writes the served body's bytes; a difference makes the timing meaningless.
fn checkBodies(run: *report.Run, c: GemmCtx) !void {
    var bytes: [2][]u16 = undefined;
    for (&bytes, 0..) |*b, arm| {
        const y = try gemmCall(c, arm);
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
        const p: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(y) orelse return error.F16Unreadable);
        b.* = try run.a.dupe(u16, p[0..mlx.mlx_array_size(y)]);
    }
    if (!std.mem.eql(u16, bytes[0], bytes[1])) run.invalid("the reference NAX body wrote other bytes than the served body");
}

const TrunkShape = struct { name: []const u8, out: usize, in: usize, n: u32, groups: usize = 1 };
/// DeepSeek-V4.1-Flash's trunk linears as the STQ pack stores them (`trunk_quant` n64, the head n96).
const TRUNK_SHAPES = [_]TrunkShape{
    .{ .name = "wq_a", .out = 1280, .in = 5120, .n = 64 },
    .{ .name = "wq_b", .out = 32768, .in = 1280, .n = 64 },
    .{ .name = "wkv", .out = 512, .in = 5120, .n = 64 },
    .{ .name = "wo_a", .out = 1024, .in = 4096, .n = 64, .groups = 8 },
    .{ .name = "wo_b", .out = 5120, .in = 8192, .n = 64 },
    .{ .name = "shared_w1", .out = 2304, .in = 5120, .n = 64 },
    .{ .name = "shared_w2", .out = 5120, .in = 2304, .n = 64 },
    .{ .name = "comp_wkv", .out = 512, .in = 5120, .n = 64 },
    .{ .name = "idx_wq_b", .out = 4096, .in = 1280, .n = 64 },
    .{ .name = "idx_wk", .out = 128, .in = 512, .n = 64 },
    .{ .name = "head", .out = 129280, .in = 5120, .n = 96 },
};

pub const trunk_rows: report.Bench = .{
    .name = "shared.exl3_trunk_rows",
    .scope = .shared,
    .what = "DeepSeek-V4.1 trunk linears by width over distinct banks, a forward's cold reads: served, NAX at every width, the indexed GEMV, mxfp8, affine-4 and a streaming floor",
    .unit = .us,
    .steps = 64,
    .axes = &.{
        .{ .name = "shape", .values = &.{ "wq_a", "wq_b", "wkv", "wo_a", "wo_b", "shared_w1", "shared_w2", "comp_wkv", "idx_wq_b", "idx_wk", "head" }, .numeric = false },
        .{ .name = "rows", .values = &.{ "1", "2", "3", "5", "8", "256" } },
    },
    .arms = &.{ "served", "exact", "gemv", "mxfp8", "affine4", "floor" },
    .ref = "served",
    .run = runTrunkRows,
};

const TRUNK_ARM_FLOOR = 5;

const Trunk = struct {
    sh: TrunkShape,
    copies: usize,
    stq: []const root.Proj,
    mx_w: []const mlx.mlx_array,
    mx_s: []const mlx.mlx_array,
    a4_w: []const mlx.mlx_array,
    a4_s: []const mlx.mlx_array,
    rows: usize,
    arm: usize,
};

var stream_floor_kernel: ?mlx.mlx_fast_metal_kernel = null;
const STREAM_FLOOR_SOURCE: [:0]const u8 =
    \\const uint tid = uint(thread_position_in_grid.x);
    \\const device uint4 *p = (const device uint4 *)w;
    \\const uint total = uint(threads_per_grid.x);
    \\const uint4 a = p[tid];
    \\const uint4 b = p[tid + total];
    \\const uint4 c = p[tid + 2u * total];
    \\const uint4 d = p[tid + 3u * total];
    \\uint v = (a.x ^ a.y ^ a.z ^ a.w) + (b.x ^ b.y ^ b.z ^ b.w) + (c.x ^ c.y ^ c.z ^ c.w) + (d.x ^ d.y ^ d.z ^ d.w);
    \\if (tid == 0u && float(xin[0]) == 1.5e30f) v = 0u;
    \\v = simd_sum(v);
    \\if ((tid & 31u) == 0u) y[tid >> 5u] = v;
;

/// The same bytes as the STQ bank, streamed by 16-byte coalesced loads: the memory floor of a call.
/// `xin` orders it after the previous step.
fn streamFloor(s: mlx.mlx_stream, w: mlx.mlx_array, xin: mlx.mlx_array, bytes: usize) !mlx.mlx_array {
    const nthreads: c_int = @intCast(bytes / 16 / 4);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{@divExact(nthreads, 32)}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, nthreads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const k = try kernels.getNamedKernel(&stream_floor_kernel, "sushi_stream_ref", &.{ "w", "xin" }, &.{"y"}, STREAM_FLOOR_SOURCE, "");
    const ov = try kernels.applyOuts(s, k, &.{ w, xin }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    return y;
}

fn trunkBytes(sh: TrunkShape) usize {
    return sh.out * sh.in * sh.groups * sh.n / 128;
}

fn trunkStep(t: Trunk, s: mlx.mlx_stream, i: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    const c = i % t.copies;
    const dec = kernels.SPEC1_W15;
    const sh = t.sh;
    switch (t.arm) {
        0, 1, 2 => {
            const served = t.rows <= @as(usize, @intCast(kernels.PROJECT_ROWS_MAX));
            kernels.rows_exact = t.arm == 1;
            defer kernels.rows_exact = false;
            root.linear.force_arm = if (t.arm == 2 and served) .gemv else null;
            defer root.linear.force_arm = null;
            if (sh.groups > 1) return root.linear.linearGrouped(s, x, t.stq[c], dec);
            var x2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x2);
            try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ @intCast(t.rows), @intCast(sh.in) }, 2, s));
            return root.linear.linear(s, x2, t.stq[c], dec);
        },
        3 => {
            var o = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(o);
            try mlx.check(mlx.mlx_quantized_matmul(&o, x, t.mx_w[c], t.mx_s[c], .{ .ctx = null }, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", s));
            return o;
        },
        4 => {
            var o = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(o);
            try mlx.check(mlx.mlx_quantized_matmul(&o, x, t.a4_w[c], t.a4_s[c], t.a4_s[c], true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", s));
            return o;
        },
        else => return streamFloor(s, t.stq[c].trellis, x, trunkBytes(sh)),
    }
}

const TrunkSample = struct { s: mlx.mlx_stream, x0: mlx.mlx_array, t: Trunk, steps: usize };

fn sampleTrunk(smp: TrunkSample, arm: usize) anyerror!f64 {
    var t = smp.t;
    t.arm = arm;
    kernels.setDecodeParams(kernels.SPEC1_W15);
    const ns = try bench.chainNs(smp.s, smp.x0, smp.steps, t, trunkStep);
    const floor = try bench.chainNs(smp.s, smp.x0, smp.steps, {}, bench.linkOnly);
    return (@as(f64, @floatFromInt(ns)) - @as(f64, @floatFromInt(floor))) / @as(f64, @floatFromInt(smp.steps));
}

/// Matrix `c` of a `[copies * g, ...]` bank as its own `[g, ...]` view, evaluated (views share the bank's buffer).
fn bankView(s: mlx.mlx_stream, bank: mlx.mlx_array, c: usize, g: usize) !mlx.mlx_array {
    const sh = mlx.getShape(bank);
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = .{ 0, 0, 0, 0 };
    const nd = sh.len;
    start[0] = @intCast(c * g);
    stop[0] = @intCast((c + 1) * g);
    for (1..nd) |d| stop[d] = sh[d];
    const ones: [4]c_int = @splat(1);
    var v = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_slice(&v, bank, &start, @intCast(nd), &stop, @intCast(nd), &ones, @intCast(nd), s));
    try mlx.check(mlx.mlx_array_eval(v));
    return v;
}

fn runTrunkRows(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    defer {
        kernels.setDecodeParams(.mul1);
        kernels.rows_exact = false;
        root.linear.force_arm = null;
    }
    var prng = std.Random.DefaultPrng.init(0x7a);
    const rnd = prng.random();
    try kernels.SPEC1_W15.requireTable();
    for (TRUNK_SHAPES) |sh| {
        if (!listed(run.axis("shape"), sh.name)) continue;
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const g = sh.groups;
        const copies = std.math.clamp((768 << 20) / (sh.out * sh.in * g), 4, 64);
        var owned: std.ArrayList(mlx.mlx_array) = .empty;
        defer for (owned.items) |x| {
            _ = mlx.mlx_array_free(x);
        };
        const words = try a.alloc(u16, copies * g * (sh.in / 16) * (sh.out / 16) * sh.n);
        bench.randomWords(rnd, words);
        const bank = mlx.mlx_array_new_data(words.ptr, &[_]c_int{ @intCast(copies * g), @intCast(sh.in / 16), @intCast(sh.out / 16), @intCast(sh.n) }, 4, .uint16);
        try owned.append(a, bank);
        a.free(words);
        // A chain keeps its magnitude: suh = 1/sqrt(in) cancels the inner sum's growth.
        const scaled = try a.alloc(u16, g * sh.in);
        @memset(scaled, exl3.f32ToF16Bits(1.0 / @sqrt(@as(f32, @floatFromInt(sh.in)))));
        const ones = try a.alloc(u16, g * sh.out);
        @memset(ones, exl3.f32ToF16Bits(1.0));
        const suh = mlx.mlx_array_new_data(scaled.ptr, &[_]c_int{ @intCast(g), @intCast(sh.in) }, 2, .float16);
        const svh = mlx.mlx_array_new_data(ones.ptr, &[_]c_int{ @intCast(g), @intCast(sh.out) }, 2, .float16);
        try owned.appendSlice(a, &.{ suh, svh });
        const stq = try a.alloc(root.Proj, copies);
        for (stq, 0..) |*p, c| {
            p.* = .{ .trellis = try bankView(s, bank, c, g), .suh = suh, .svh = svh };
            try owned.append(a, p.trellis);
        }
        const mx_codes = try a.alloc(u32, copies * g * sh.out * sh.in / 4);
        for (mx_codes) |*v| v.* = rnd.int(u32) & 0x3f3f3f3f;
        const mx_exp = try a.alloc(u8, copies * g * sh.out * sh.in / 32);
        for (mx_exp) |*v| v.* = 120 + rnd.uintLessThan(u8, 8);
        const mx_bank = mlx.mlx_array_new_data(mx_codes.ptr, &[_]c_int{ @intCast(copies * g), @intCast(sh.out), @intCast(sh.in / 4) }, 3, .uint32);
        const mx_scale_bank = mlx.mlx_array_new_data(mx_exp.ptr, &[_]c_int{ @intCast(copies * g), @intCast(sh.out), @intCast(sh.in / 32) }, 3, .uint8);
        a.free(mx_codes);
        a.free(mx_exp);
        const a4_codes = try a.alloc(u32, copies * g * sh.out * sh.in / 8);
        for (a4_codes) |*v| v.* = rnd.int(u32);
        const a4_bank = mlx.mlx_array_new_data(a4_codes.ptr, &[_]c_int{ @intCast(copies * g), @intCast(sh.out), @intCast(sh.in / 8) }, 3, .uint32);
        a.free(a4_codes);
        var a4_scale_bank = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_full(&a4_scale_bank, &[_]c_int{ @intCast(copies * g), @intCast(sh.out), @intCast(sh.in / 64) }, 3, mlx.mlx_array_new_float(0.01), .bfloat16, s));
        try owned.appendSlice(a, &.{ mx_bank, mx_scale_bank, a4_bank, a4_scale_bank });
        const mx_w = try a.alloc(mlx.mlx_array, copies);
        const mx_s = try a.alloc(mlx.mlx_array, copies);
        const a4_w = try a.alloc(mlx.mlx_array, copies);
        const a4_s = try a.alloc(mlx.mlx_array, copies);
        for (0..copies) |c| {
            mx_w[c] = try bankView(s, mx_bank, c, g);
            mx_s[c] = try bankView(s, mx_scale_bank, c, g);
            a4_w[c] = try bankView(s, a4_bank, c, g);
            a4_s[c] = try bankView(s, a4_scale_bank, c, g);
            try owned.appendSlice(a, &.{ mx_w[c], mx_s[c], a4_w[c], a4_s[c] });
        }
        for (run.axis("rows")) |rows_text| {
            const rows = try std.fmt.parseInt(usize, rows_text, 10);
            if (g > 1 and rows == 256) continue;
            const xs = try a.alloc(u16, g * rows * sh.in);
            for (xs) |*v| v.* = @truncate(@as(u32, @bitCast(rnd.float(f32) - 0.5)) >> 16);
            const x0 = mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ @intCast(g), @intCast(rows), @intCast(sh.in) }, 3, .bfloat16);
            defer _ = mlx.mlx_array_free(x0);
            const t: Trunk = .{ .sh = sh, .copies = copies, .stq = stq, .mx_w = mx_w, .mx_s = mx_s, .a4_w = a4_w, .a4_s = a4_s, .rows = rows, .arm = 0 };
            const label = try std.fmt.allocPrint(run.a, "{s} G{d} [{d}x{d}] n{d} rows {d}", .{ sh.name, g, sh.out, sh.in, sh.n, rows });
            try run.interleave(label, TrunkSample{ .s = s, .x0 = x0, .t = t, .steps = run.steps(64) }, sampleTrunk);
            const bytes: f64 = @floatFromInt(trunkBytes(sh));
            try run.metric("served", "MB", "MB", bytes / 1e6);
            const us = run.lastMedianNs("served") / 1e3;
            try run.metric("served", "GB/s", "GB/s", bench.gbps(bytes, us));
        }
    }
}

const T1_ROUTES = 24;
const T1_HIDDEN: c_int = 4096;
const T1_INTER: c_int = 2048;
const T1_TOPK: usize = 8;
const T1_EXPERTS: usize = 288;
const T1_PHASES = [_][]const u8{ "link", "pair_prepare", "gate_up", "mid_prepare", "down", "finish" };

pub const routed_t1: report.Bench = .{
    .name = "glm5.routed_t1",
    .scope = .glm5,
    .what = "GLM's one-row routed chain (E288, 4096 -> 2048, top-8, clamp 10, MCG w14) up to each phase, per layer: lane GEMVs against the one-row cooperative ones",
    .unit = .us,
    .steps = 24,
    .axes = &.{
        .{ .name = "rate", .values = &.{ "36", "40" } },
        .{ .name = "phase", .values = &T1_PHASES, .numeric = false },
    },
    .arms = &.{ "lane", "one_row" },
    .ref = "one_row",
    .run = runRoutedT1,
};

const T1Ctx = struct {
    f: *const kernels.MimoMoeFixture,
    shared: mlx.mlx_array,
    routes: *const [T1_ROUTES]mlx.mlx_array,
    upto: usize,
};

/// One decode layer's chain through phase `upto` (0 = the link alone); the output is what the next layer waits on.
fn t1Step(c: T1Ctx, s: mlx.mlx_stream, i: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    const a = c.f.arrays;
    const slots = c.routes[i % T1_ROUTES];
    if (c.upto == 0) return bench.linkOnly({}, s, i, x);
    const prep = try kernels.lanePairPrepare(s, x, a[3], a[3], slots, null, T1_HIDDEN, T1_TOPK, T1_TOPK);
    defer _ = mlx.mlx_array_free(prep[1]);
    if (c.upto == 1) return prep[0];
    defer _ = mlx.mlx_array_free(prep[0]);
    const gu = (try kernels.lanePairCoopWith(s, prep[0], prep[1], a[0], a[1], slots, !kernels.one_row_coop_off)) orelse return error.LanePairDeclined;
    defer _ = mlx.mlx_array_free(gu[1]);
    if (c.upto == 2) return gu[0];
    defer _ = mlx.mlx_array_free(gu[0]);
    const mid = try kernels.downLanePrepare(s, gu[0], gu[1], a[4], a[4], a[5], slots, T1_INTER, T1_TOPK, 10);
    if (c.upto == 3) return mid;
    defer _ = mlx.mlx_array_free(mid);
    const down = try kernels.downLaneCoopWith(s, mid, a[2], slots, !kernels.one_row_coop_off);
    if (c.upto == 4) return down;
    defer _ = mlx.mlx_array_free(down);
    return kernels.downFinishReduceWithShared(s, down, a[6], slots, a[9], T1_HIDDEN, 1, T1_TOPK, .bfloat16, c.shared, null, true);
}

const T1Sample = struct { s: mlx.mlx_stream, x0: mlx.mlx_array, c: T1Ctx, steps: usize };

fn sampleT1(smp: T1Sample, arm: usize) anyerror!f64 {
    kernels.one_row_coop_off = arm == 0;
    const ns = try bench.chainNs(smp.s, smp.x0, smp.steps, smp.c, t1Step);
    return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(smp.steps));
}

fn t1Bytes(a: mlx.mlx_array) ![]const u8 {
    try mlx.check(mlx.mlx_array_eval(a));
    const n = mlx.mlx_array_size(a);
    return switch (mlx.mlx_array_dtype(a)) {
        .float32 => std.mem.sliceAsBytes((mlx.mlx_array_data_float32(a) orelse return error.F32Unreadable)[0..n]),
        .float16 => std.mem.sliceAsBytes((mlx.mlx_array_data_float16(a) orelse return error.F16Unreadable)[0..n]),
        else => error.UnexpectedDtype,
    };
}

/// The two GEMV kinds write the same bytes through the gate, up and down projections.
fn t1SameBytes(run: *report.Run, s: mlx.mlx_stream, f: *const kernels.MimoMoeFixture, x0: mlx.mlx_array, slots: mlx.mlx_array) !void {
    const a = f.arrays;
    var want: [3][]const u8 = undefined;
    for ([_]bool{ true, false }, 0..) |off, vi| {
        kernels.one_row_coop_off = off;
        const prep = try kernels.lanePairPrepare(s, x0, a[3], a[3], slots, null, T1_HIDDEN, T1_TOPK, T1_TOPK);
        defer _ = mlx.mlx_array_free(prep[0]);
        defer _ = mlx.mlx_array_free(prep[1]);
        const gu = (try kernels.lanePairCoopWith(s, prep[0], prep[1], a[0], a[1], slots, !off)).?;
        defer _ = mlx.mlx_array_free(gu[0]);
        defer _ = mlx.mlx_array_free(gu[1]);
        const mid = try kernels.downLanePrepare(s, gu[0], gu[1], a[4], a[4], a[5], slots, T1_INTER, T1_TOPK, 10);
        defer _ = mlx.mlx_array_free(mid);
        const down = try kernels.downLaneCoopWith(s, mid, a[2], slots, !off);
        defer _ = mlx.mlx_array_free(down);
        const got = [3]mlx.mlx_array{ gu[0], gu[1], down };
        for (got, 0..) |g, k| {
            const bytes = try run.a.dupe(u8, try t1Bytes(g));
            if (vi == 0) want[k] = bytes else if (!std.mem.eql(u8, want[k], bytes)) run.invalid("the lane and one-row GEMVs wrote different bytes");
        }
    }
}

fn runRoutedT1(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    const dec: exl3.Decode = .{ .codebook = .mcg, .window = .w14 };
    kernels.setDecodeParams(dec);
    defer {
        kernels.setDecodeParams(.mul1);
        kernels.one_row_coop_off = false;
    }
    for (run.axis("rate")) |rate_text| {
        const n = try std.fmt.parseInt(u32, rate_text, 10);
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var f = try kernels.mimoMoeFixture(a, .{ .hidden = @intCast(T1_HIDDEN), .inter = @intCast(T1_INTER), .e = T1_EXPERTS, .rows = 1, .topk = T1_TOPK, .rate = .{ .n = n }, .dec = dec, .seed = 7000 + n, .banks = kernels.MIMO_BANKS, .x_scale = 1 });
        defer f.deinit();
        var prng = std.Random.DefaultPrng.init(n);
        const rnd = prng.random();
        var routes: [T1_ROUTES]mlx.mlx_array = undefined;
        for (&routes) |*r| {
            var ids: [T1_TOPK]u32 = undefined;
            for (&ids, 0..) |*v, k| while (true) {
                const pick = rnd.uintLessThan(u32, T1_EXPERTS);
                if (std.mem.indexOfScalar(u32, ids[0..k], pick) == null) {
                    v.* = pick;
                    break;
                }
            };
            r.* = mlx.mlx_array_new_data(&ids, &[_]c_int{T1_TOPK}, 1, .uint32);
        }
        defer for (routes) |r| {
            _ = mlx.mlx_array_free(r);
        };
        var x0 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x0);
        try mlx.check(mlx.mlx_astype(&x0, f.arrays[8], .bfloat16, s));
        var shared = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(shared);
        try mlx.check(mlx.mlx_multiply(&shared, x0, x0, s));
        try mlx.check(mlx.mlx_array_eval(shared));
        try t1SameBytes(run, s, &f, x0, routes[0]);
        var prev: [2]f64 = .{ 0, 0 };
        var prev_phase: ?usize = null;
        for (run.axis("phase")) |phase| {
            const upto = phaseIndex(phase);
            const c: T1Ctx = .{ .f = &f, .shared = shared, .routes = &routes, .upto = upto };
            const label = try std.fmt.allocPrint(run.a, "n{d} {s}", .{ n, phase });
            try run.interleave(label, T1Sample{ .s = s, .x0 = x0, .c = c, .steps = run.steps(24) }, sampleT1);
            for ([_][]const u8{ "lane", "one_row" }, 0..) |arm, k| {
                if (!run.hasArm(arm)) continue;
                const med = run.lastMedianNs(arm) / 1e3;
                if (upto == 0 or prev_phase == upto - 1) try run.metric(arm, "phase us", "us", if (upto == 0) med else med - prev[k]);
                prev[k] = med;
            }
            prev_phase = upto;
        }
    }
}

fn phaseIndex(name: []const u8) usize {
    for (T1_PHASES, 0..) |p, i| if (std.mem.eql(u8, p, name)) return i;
    unreachable;
}
