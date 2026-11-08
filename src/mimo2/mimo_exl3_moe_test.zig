//! MiMo-shaped EXL3 MoE layers through the trunk's MoE path against a host oracle: its sigmoid routing with a selection
//! bias, the routed banks on either codebook, and two packs on different codebooks resident together.
const std = @import("std");
const mlx = @import("../mlx.zig");
const trunk = @import("../trunk.zig");
const sushi_exl3 = @import("sushi_exl3");
const expert_exl3 = sushi_exl3.format;
const expert_exl3_kernels = @import("../layer_weights.zig").expert_exl3_kernels;
const mimo_forward = @import("mimo_forward.zig");
const io_util_mod = @import("../io_util.zig");
const diagEnvOn = io_util_mod.diagEnvOn;
const Trunk = trunk.Trunk;
const MoeMlpWeights = trunk.MoeMlpWeights;
const bindLinears = @import("../layer_weights.zig").bindLinears;

fn mimoFixtureMaxAbs(a: []const f32, b: []const f32) !f32 {
    if (a.len != b.len) return error.FixtureShapeMismatch;
    var max_err: f32 = 0;
    for (a, b) |x, y| {
        if (!std.math.isFinite(x) or !std.math.isFinite(y)) return error.NonFiniteFixture;
        max_err = @max(max_err, @abs(x - y));
    }
    return max_err;
}

pub fn mimoExpectClose(label: []const u8, actual: []const f32, expected: []const f32, limit: f32) !void {
    const err = try mimoFixtureMaxAbs(actual, expected);
    if (err > limit) {
        var peak: usize = 0;
        for (actual, expected, 0..) |x, y, i| {
            if (@abs(x - y) > @abs(actual[peak] - expected[peak])) peak = i;
        }
        std.debug.print("{s}: max absolute error {d} exceeds {d}; index {d}, actual {d}, expected {d}; first-row error {d}\n", .{
            label, err, limit, peak, actual[peak], expected[peak],
            try mimoFixtureMaxAbs(actual[0..@min(384, actual.len)], expected[0..@min(384, expected.len)]),
        });
        return error.TestExpectedEqual;
    }
}

/// A MiMo-shaped EXL3 MoE layer: hidden != inter, sigmoid routing with a
/// selection bias, no shared expert. Every expert holds the SAME bank and the
/// router is flat, so the normalized routed sum is one expert's own output
/// whichever top-k the GPU picks — that is the host oracle.
pub const MimoExl3MoeHarness = struct {
    xfm: Trunk,
    x: mlx.mlx_array,
    mw: MoeMlpWeights,
    owned: [10]mlx.mlx_array,
    gate_t: []u16,
    down_t: []u16,
    suh_h: []u16,
    svh_i: []u16,
    suh_i: []u16,
    svh_h: []u16,
    xf: []f32,
    dec: expert_exl3.Decode,
    rows: usize,

    const E: usize = 4;
    const hidden: usize = 128;
    const inter: usize = 256;

    pub fn init(alloc: std.mem.Allocator, s: mlx.mlx_stream, rate: expert_exl3.Rate, rows: usize, seed: u64, dec: expert_exl3.Decode) !MimoExl3MoeHarness {
        const packed_n = rate.halfwords();
        const gu_tile = (hidden / 16) * (inter / 16) * packed_n;
        const d_tile = (inter / 16) * (hidden / 16) * packed_n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const gate_t = try alloc.alloc(u16, E * gu_tile);
        const down_t = try alloc.alloc(u16, E * d_tile);
        for (gate_t[0..gu_tile]) |*v| v.* = @truncate(rnd.int(u32));
        for (down_t[0..d_tile]) |*v| v.* = @truncate(rnd.int(u32));
        for (1..E) |e| {
            @memcpy(gate_t[e * gu_tile ..][0..gu_tile], gate_t[0..gu_tile]);
            @memcpy(down_t[e * d_tile ..][0..d_tile], down_t[0..d_tile]);
        }
        const scale = struct {
            fn fill(bank: []u16, n: usize, r: std.Random) void {
                for (bank[0..n]) |*v| v.* = expert_exl3.f32ToF16Bits(if (r.boolean()) @as(f32, 0.9) else -0.9);
                var e: usize = 1;
                while (e * n < bank.len) : (e += 1) @memcpy(bank[e * n ..][0..n], bank[0..n]);
            }
        }.fill;
        const suh_h = try alloc.alloc(u16, E * hidden);
        const svh_i = try alloc.alloc(u16, E * inter);
        const suh_i = try alloc.alloc(u16, E * inter);
        const svh_h = try alloc.alloc(u16, E * hidden);
        scale(suh_h, hidden, rnd);
        scale(svh_i, inter, rnd);
        scale(suh_i, inter, rnd);
        scale(svh_h, hidden, rnd);
        const xh = try alloc.alloc(u16, rows * hidden);
        const xf = try alloc.alloc(f32, rows * hidden);
        for (xh, xf) |*b, *v| {
            b.* = expert_exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
            v.* = expert_exl3.f16BitsToF32(b.*);
        }
        // A flat router: every expert scores the same, so the normalized
        // weights sum to one whichever top-k the selection returns.
        const rw = try alloc.alloc(u16, hidden * E);
        @memset(rw, 0);
        const bias = try alloc.alloc(f32, E);
        @memset(bias, 0);
        const ci = struct {
            fn i(v: usize) c_int {
                return @intCast(v);
            }
        }.i;
        const gt = mlx.mlx_array_new_data(gate_t.ptr, &[_]c_int{ ci(E), ci(hidden / 16), ci(inter / 16), ci(packed_n) }, 4, .uint16);
        const dt = mlx.mlx_array_new_data(down_t.ptr, &[_]c_int{ ci(E), ci(inter / 16), ci(hidden / 16), ci(packed_n) }, 4, .uint16);
        const a_suh_h = mlx.mlx_array_new_data(suh_h.ptr, &[_]c_int{ ci(E), ci(hidden) }, 2, .float16);
        const a_svh_i = mlx.mlx_array_new_data(svh_i.ptr, &[_]c_int{ ci(E), ci(inter) }, 2, .float16);
        const a_suh_i = mlx.mlx_array_new_data(suh_i.ptr, &[_]c_int{ ci(E), ci(inter) }, 2, .float16);
        const a_svh_h = mlx.mlx_array_new_data(svh_h.ptr, &[_]c_int{ ci(E), ci(hidden) }, 2, .float16);
        const router = mlx.mlx_array_new_data(rw.ptr, &[_]c_int{ ci(hidden), ci(E) }, 2, .float16);
        const eb = mlx.mlx_array_new_data(bias.ptr, &[_]c_int{ci(E)}, 1, .float32);
        const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ 1, ci(rows), ci(hidden) }, 3, .float16);
        const one = mlx.mlx_array_new_float(1.0);
        const none = mlx.mlx_array{ .ctx = null };
        var h: MimoExl3MoeHarness = .{
            .xfm = undefined,
            .x = x,
            .owned = .{ gt, dt, a_suh_h, a_svh_i, a_suh_i, a_svh_h, router, eb, x, one },
            .gate_t = gate_t,
            .down_t = down_t,
            .suh_h = suh_h,
            .svh_i = svh_i,
            .suh_i = suh_i,
            .svh_h = svh_h,
            .xf = xf,
            .dec = dec,
            .rows = rows,
            .mw = .{
                .router_w = router,
                .router_s = none,
                .router_b = none,
                .switch_gate_w = gt,
                .switch_gate_s = a_suh_h,
                .switch_gate_b = a_svh_i,
                .switch_up_w = gt,
                .switch_up_s = a_suh_h,
                .switch_up_b = a_svh_i,
                .switch_down_w = dt,
                .switch_down_s = a_suh_i,
                .switch_down_b = a_svh_h,
                .shared_gate_w = none,
                .shared_gate_s = none,
                .shared_gate_b = none,
                .shared_up_w = none,
                .shared_up_s = none,
                .shared_up_b = none,
                .shared_down_w = none,
                .shared_down_s = none,
                .shared_down_b = none,
                .expert_bias = eb,
                .route_norm = true,
                .route_scale = 1.0,
            },
        };
        h.xfm.s = s;
        h.xfm.config = .{
            .model_type = "mimo_v2",
            .expert_layout = .exl3_k4,
            .expert_quant_rate = rate,
            .expert_quant_codebook = dec.codebook,
            .expert_quant_window = dec.window,
            .expert_quant_table = dec.table,
            .num_experts = @intCast(E),
            .num_experts_per_tok = 2,
            .hidden_size = @intCast(hidden),
            .moe_intermediate_size = @intCast(inter),
            .quant_bits = 0,
            .quant_group_size = 64,
            .quant_mode = .affine,
            .hidden_act = .silu,
            .moe_sigmoid_router = true,
            .moe_n_group = 1,
            .moe_topk_group = 1,
            .moe_route_norm = true,
        };
        h.xfm.one = one;
        h.xfm.bits_cache = null;
        h.xfm.compiled_moe_routing = null;
        h.xfm.compiled_gelu = null;
        h.xfm.compiled_geglu = null;
        h.xfm.cost_trace_active = false;
        h.xfm.expert_stream = null;
        try bindLinears(&h.mw, &h.xfm.config);
        for (h.owned) |a| try mlx.check(mlx.mlx_array_eval(a));
        return h;
    }

    /// One expert's SwiGLU on the host, at the rate the trellis names.
    fn hostRow(self: *MimoExl3MoeHarness, alloc: std.mem.Allocator, rate: expert_exl3.Rate, row: usize, out: []f32) !void {
        const x = self.xf[row * hidden ..][0..hidden];
        const t_hidden = try alloc.alloc(f32, hidden);
        defer alloc.free(t_hidden);
        const t_inter = try alloc.alloc(f32, inter);
        defer alloc.free(t_inter);
        const inner = try alloc.alloc(f32, inter);
        defer alloc.free(inner);
        const gate_y = try alloc.alloc(f32, inter);
        defer alloc.free(gate_y);
        const up_y = try alloc.alloc(f32, inter);
        defer alloc.free(up_y);
        const gu_tile = (hidden / 16) * (inter / 16) * rate.halfwords();
        const d_tile = (inter / 16) * (hidden / 16) * rate.halfwords();
        expert_exl3.project(x, self.gate_t[0..gu_tile], self.suh_h[0..hidden], self.svh_i[0..inter], hidden, inter, rate, self.dec, t_hidden, inner, gate_y);
        expert_exl3.project(x, self.gate_t[0..gu_tile], self.suh_h[0..hidden], self.svh_i[0..inter], hidden, inter, rate, self.dec, t_hidden, inner, up_y);
        for (gate_y, up_y) |*g, u| g.* = (g.* / (1.0 + @exp(-g.*))) * u;
        expert_exl3.project(gate_y, self.down_t[0..d_tile], self.suh_i[0..inter], self.svh_h[0..hidden], inter, hidden, rate, self.dec, t_inter, inner[0..hidden], out);
    }

    /// One forward through the engine, scored against this harness's own
    /// codebook on the host.
    fn expectMatchesHost(self: *MimoExl3MoeHarness, alloc: std.mem.Allocator, rate: expert_exl3.Rate) !void {
        const y = try self.xfm.moeMLP(self.x, &self.mw);
        defer _ = mlx.mlx_array_free(y);
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, y, false, self.xfm.s));
        try mlx.check(mlx.mlx_array_eval(c));
        const got = mlx.mlx_array_data_float16(c) orelse return error.F16Unreadable;
        const want = try alloc.alloc(f32, hidden);
        defer alloc.free(want);
        var ss: f64 = 0;
        var ref: f64 = 0;
        for (0..self.rows) |r| {
            try self.hostRow(alloc, rate, r, want);
            for (want, 0..) |w, i| {
                const a: f64 = @floatCast(got[r * hidden + i]);
                ss += (a - w) * (a - w);
                ref += @as(f64, w) * @as(f64, w);
            }
        }
        const rel = @sqrt(ss / @max(ref, 1e-20));
        if (!(rel < 0.01)) {
            std.debug.print("mimo exl3 forward codebook={s} window={d} rel_rms={d:.6}\n", .{ @tagName(self.dec.codebook), self.dec.window.bits(), rel });
            return error.TestExpectedEqual;
        }
    }

    pub fn deinit(self: *MimoExl3MoeHarness) void {
        for (self.owned) |a| _ = mlx.mlx_array_free(a);
    }
};

pub fn mimoExl3ForwardMatchesHost(rate: expert_exl3.Rate, rows: usize, seed: u64) !void {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var h = try MimoExl3MoeHarness.init(alloc, s, rate, rows, seed, .mcg);
    defer h.deinit();
    try h.expectMatchesHost(alloc, rate);
}

/// Two EXL3 packs on different codebooks, resident together and dispatching
/// alternately: every forward must decode with ITS OWN model's codebook.
pub fn exl3CodebookFollowsModel(rows: usize) !void {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const rate: expert_exl3.Rate = .{ .n = 40 };
    var mul1 = try MimoExl3MoeHarness.init(alloc, s, rate, rows, 101, .mul1);
    defer mul1.deinit();
    var mcg = try MimoExl3MoeHarness.init(alloc, s, rate, rows, 103, .mcg);
    defer mcg.deinit();
    for (0..2) |_| {
        try mul1.expectMatchesHost(alloc, rate);
        try mcg.expectMatchesHost(alloc, rate);
    }
}

pub fn mimoExl3DistinctRouting(rows: usize) !void {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const H = MimoExl3MoeHarness.hidden;
    const I = MimoExl3MoeHarness.inter;
    const E = MimoExl3MoeHarness.E;
    const rate: expert_exl3.Rate = .{ .n = 40 };
    var h = try MimoExl3MoeHarness.init(alloc, s, rate, rows, 191, .{ .codebook = .mcg, .window = .w12 });
    defer h.deinit();
    var prng = std.Random.DefaultPrng.init(193);
    const rnd = prng.random();
    for (h.gate_t) |*v| v.* = rnd.int(u16);
    for (h.down_t) |*v| v.* = rnd.int(u16);
    const router = try alloc.alloc(f32, H * E);
    for (router) |*v| {
        const value = (rnd.float(f32) * 2 - 1) * 0.2;
        const bits: u32 = @bitCast(value);
        v.* = @bitCast((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
    }
    const bias = [_]f32{ 0.03, -0.08, 0.11, -0.02 };
    const replace = struct {
        fn array(dst: *mlx.mlx_array, src: mlx.mlx_array) void {
            _ = mlx.mlx_array_free(dst.*);
            dst.* = src;
        }
    }.array;
    replace(&h.owned[0], mlx.mlx_array_new_data(h.gate_t.ptr, &.{ E, H / 16, I / 16, 40 }, 4, .uint16));
    replace(&h.owned[1], mlx.mlx_array_new_data(h.down_t.ptr, &.{ E, I / 16, H / 16, 40 }, 4, .uint16));
    replace(&h.owned[6], mlx.mlx_array_new_data(router.ptr, &.{ H, E }, 2, .float32));
    replace(&h.owned[7], mlx.mlx_array_new_data(&bias, &.{E}, 1, .float32));
    var xb = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&xb, h.x, .bfloat16, s));
    replace(&h.owned[8], xb);
    h.x = xb;
    h.mw.switch_gate_w = h.owned[0];
    h.mw.switch_up_w = h.owned[0];
    h.mw.switch_down_w = h.owned[1];
    h.mw.router_w = h.owned[6];
    h.mw.expert_bias = h.owned[7];
    var x32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x32);
    try mlx.check(mlx.mlx_astype(&x32, h.x, .float32, s));
    try mlx.check(mlx.mlx_array_eval(x32));
    @memcpy(h.xf, (mlx.mlx_array_data_float32(x32) orelse return error.F32Unreadable)[0..h.xf.len]);
    const logits = try mimo_forward.mimoRouterLogits(&h.xfm, h.x, &h.mw);
    defer _ = mlx.mlx_array_free(logits);
    const routed = try mimo_forward.computeMimoRouting(&h.xfm, logits, h.mw.expert_bias.?);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);
    try t.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(routed.norm_scores));
    var ids32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids32);
    try mlx.check(mlx.mlx_astype(&ids32, routed.inds, .uint32, s));
    try mlx.check(mlx.mlx_array_eval(ids32));
    try mlx.check(mlx.mlx_array_eval(routed.norm_scores));
    const ids = mlx.mlx_array_data_uint32(ids32) orelse return error.ExpertIdsUnreadable;
    const scores = mlx.mlx_array_data_float32(routed.norm_scores) orelse return error.F32Unreadable;
    const y = try h.xfm.moeMLP(h.x, &h.mw);
    defer _ = mlx.mlx_array_free(y);
    try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(y));
    var y32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y32);
    try mlx.check(mlx.mlx_astype(&y32, y, .float32, s));
    try mlx.check(mlx.mlx_array_eval(y32));
    const got = mlx.mlx_array_data_float32(y32) orelse return error.F32Unreadable;
    var err2: f64 = 0;
    var ref2: f64 = 0;
    var seen: u32 = 0;
    for (0..rows) |r| {
        var probabilities: [E]f32 = undefined;
        for (0..E) |e| {
            var dot: f32 = 0;
            for (0..H) |d| dot += h.xf[r * H + d] * router[d * E + e];
            probabilities[e] = 1 / (1 + @exp(-dot));
        }
        const den = probabilities[ids[r * 2]] + probabilities[ids[r * 2 + 1]];
        var want: [H]f32 = @splat(0);
        for (0..2) |k| {
            const e = ids[r * 2 + k];
            try t.expect(e < E);
            seen |= @as(u32, 1) << @intCast(e);
            const score = probabilities[e] / den;
            try t.expectApproxEqAbs(score, scores[r * 2 + k], 1e-6);
            for (0..E) |other| {
                if (other == ids[r * 2] or other == ids[r * 2 + 1]) continue;
                try t.expect(probabilities[e] + bias[e] >= probabilities[other] + bias[other]);
            }
            var oracle = h;
            oracle.gate_t = h.gate_t[e * (h.gate_t.len / E) ..];
            oracle.down_t = h.down_t[e * (h.down_t.len / E) ..];
            oracle.suh_h = h.suh_h[e * H ..];
            oracle.svh_i = h.svh_i[e * I ..];
            oracle.suh_i = h.suh_i[e * I ..];
            oracle.svh_h = h.svh_h[e * H ..];
            var out: [H]f32 = undefined;
            try oracle.hostRow(alloc, rate, r, &out);
            for (&want, out) |*w, value| w.* += value * score;
        }
        for (want, 0..) |w, d| {
            const actual = got[r * H + d];
            try t.expect(std.math.isFinite(actual) and std.math.isFinite(w));
            const delta: f64 = actual - w;
            err2 += delta * delta;
            ref2 += @as(f64, w) * w;
        }
    }
    if (rows > 1) try t.expect(@popCount(seen) > 2);
    const relative = @sqrt(err2 / @max(ref2, 1e-20));
    if (diagEnvOn("SUSHI_EXL3_PARITY_TRACE"))
        std.debug.print("mimo distinct routed MoE rows={d} relative RMS={d:.8}\n", .{ rows, relative });
    if (!(relative < 0.01)) {
        std.debug.print("mimo distinct routed MoE rows={d} relative RMS={d:.8}\n", .{ rows, relative });
        return error.TestExpectedEqual;
    }
}

test "exl3 decode rows follow their own model's codebook with two packs resident" {
    try exl3CodebookFollowsModel(4);
}

test "exl3 prefill rows follow their own model's codebook with two packs resident" {
    try exl3CodebookFollowsModel(expert_exl3_kernels.DECODE_ROWS_MAX * 2);
}

test "mimo_v2 EXL3 resident MoE decode rows match the host SwiGLU oracle at K2.5 MCG" {
    try mimoExl3ForwardMatchesHost(.{ .n = 40 }, 4, 71);
}

test "mimo_v2 EXL3 resident MoE prefill rows match the host SwiGLU oracle at K2.5 MCG" {
    try mimoExl3ForwardMatchesHost(.{ .n = 40 }, expert_exl3_kernels.DECODE_ROWS_MAX * 2, 73);
}

test "mimo_v2 EXL3 distinct routed experts match host with BF16 inputs" {
    for ([_]usize{ 1, 4, 14, 17, 33 }) |rows| try mimoExl3DistinctRouting(rows);
}

// ── Tests ──
