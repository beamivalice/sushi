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
