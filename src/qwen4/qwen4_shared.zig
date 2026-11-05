//! Qwen3.8-Flash-Next decode: the shared expert, its gate logit and the sigmoid-gated add inside the routed EXL3
//! launches (`exl3/qwen_fold.zig` for a one-bank layer, `exl3/rate_groups.zig` for a rate-grouped one).
const std = @import("std");
const mlx = @import("../mlx.zig");
const api = @import("sushi_exl3");
const tfm = @import("../transformer.zig");
const trunk_stq = @import("../trunk_stq.zig");
const Arr = mlx.mlx_array;

var logged = false;

fn affine(w: Arr, s: ?Arr, b: ?Arr) api.qwen_riders.Affine {
    return .{ .weight = w, .scales = s orelse .{ .ctx = null }, .biases = b orelse .{ .ctx = null } };
}

/// The layer output with the shared expert folded into the routed launches, for one to eight rows of `x` [1, rows, hidden];
/// null where the separate chain runs. `router_logit` is the shared-expert gate logit when the router kernel already
/// computed it (null ctx otherwise: the fold computes it). `trunk_dec` decodes a trellis shared expert.
pub fn folded(s: mlx.mlx_stream, x: Arr, mw: *const tfm.MoeMlpWeights, inds: Arr, scores: Arr, dec: api.format.Decode, trunk_dec: ?api.format.Decode, router_logit: Arr) !?Arr {
    const gate_w = mw.shared_expert_gate_w orelse return null;
    const own_logit = router_logit.ctx == null;
    const trellis: ?api.shared_trellis.Banks = if (mw.shared_gate_up_stack) |stack| blk: {
        if (stack.count != 2 or !trunk_stq.isStq(mw.shared_down_w)) break :blk null;
        break :blk .{
            .gate_up = stack.p,
            .down = .{ .trellis = mw.shared_down_w, .suh = mw.shared_down_s, .svh = mw.shared_down_b },
            .dec = trunk_dec orelse return null,
        };
    } else null;
    const shared: api.qwen_riders.Shared = .{
        .gate = affine(mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b),
        .up = affine(mw.shared_up_w, mw.shared_up_s, mw.shared_up_b),
        .down = affine(mw.shared_down_w, mw.shared_down_s, mw.shared_down_b),
        .trellis = trellis,
        .sigmoid = try tfm.swigluSigTable(s, .bfloat16, std.heap.c_allocator),
        .logit = if (own_logit) affine(gate_w, mw.shared_expert_gate_s, mw.shared_expert_gate_b) else null,
    };
    const gate: ?Arr = if (own_logit) null else router_logit;
    const result = if (mw.exl3_ragged) |*rb|
        try api.rate_groups.moeGatedSharedFolded(s, x, rb, inds, scores, dec, shared, gate)
    else blk: {
        const bank = api.Bank{
            .gate = .{ .trellis = mw.switch_gate_w, .suh = mw.switch_gate_s, .svh = mw.switch_gate_b },
            .up = .{ .trellis = mw.switch_up_w, .suh = mw.switch_up_s, .svh = mw.switch_up_b },
            .down = .{ .trellis = mw.switch_down_w, .suh = mw.switch_down_s, .svh = mw.switch_down_b },
        };
        break :blk try api.qwen_fold.moeGatedSharedFolded(s, x, bank, inds, scores, dec, shared, gate);
    };
    if (result != null and !logged) {
        logged = true;
        @import("../log.zig").info("[qwen] shared expert rides the routed launches engaged\n", .{});
    }
    return result;
}

pub const Ops = @import("../glm5/glm5_model.zig").Ops;
pub const test_dec: api.format.Decode = .{ .codebook = .mcg, .window = .w15 };
pub const hidden: usize = 2560;
pub const inter: usize = 640;
pub const topk: usize = 10;

fn bf16(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

/// An affine8 bank at group `group`: random codes, positive scales and centred biases at `gain`.
fn affineBank(ops: *Ops, a: std.mem.Allocator, rnd: std.Random, rows: usize, k: usize, group: usize, gain: f32) !api.qwen_riders.Affine {
    const codes = try a.alloc(u32, rows * k / 4);
    defer a.free(codes);
    const scales = try a.alloc(u16, rows * k / group);
    defer a.free(scales);
    const biases = try a.alloc(u16, scales.len);
    defer a.free(biases);
    for (codes) |*v| v.* = rnd.int(u32);
    for (scales, biases) |*sc, *b| {
        sc.* = bf16(gain * (rnd.float(f32) + 0.125) / 128);
        b.* = bf16(gain * (rnd.float(f32) - 0.5));
    }
    const shape = [_]c_int{ @intCast(rows), @intCast(k / 4) };
    const grid = [_]c_int{ @intCast(rows), @intCast(k / group) };
    return .{
        .weight = try ops.own(mlx.mlx_array_new_data(codes.ptr, &shape, 2, .uint32)),
        .scales = try ops.own(mlx.mlx_array_new_data(scales.ptr, &grid, 2, .bfloat16)),
        .biases = try ops.own(mlx.mlx_array_new_data(biases.ptr, &grid, 2, .bfloat16)),
    };
}

pub fn randomBf16(ops: *Ops, a: std.mem.Allocator, rnd: std.Random, shape: []const c_int, spread: f32) !Arr {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const values = try a.alloc(u16, n);
    defer a.free(values);
    for (values) |*v| v.* = bf16((rnd.float(f32) - 0.5) * spread);
    return ops.own(mlx.mlx_array_new_data(values.ptr, shape.ptr, @intCast(shape.len), .bfloat16));
}

/// A routed bank of `experts` at gate/up rate `n` and down rate `down_n`, random codewords at Qwen's width.
pub fn routedBank(ops: *Ops, experts: c_int, n: c_int, down_n: c_int, seed: u64) !api.Bank {
    var projections: [3]api.Proj = undefined;
    for (&projections, 0..) |*projection, i| {
        const input: c_int = if (i == 2) inter else hidden;
        const output: c_int = if (i == 2) hidden else inter;
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, seed + i));
        const codes = try ops.slot();
        try mlx.check(mlx.mlx_random_bits(codes, &[_]c_int{ experts, @divExact(input, 16), @divExact(output, 16), if (i == 2) down_n else n }, 4, 2, key.*, ops.s));
        const su = try ops.slot();
        try mlx.check(mlx.mlx_random_uniform(su, try ops.scalar(-0.2, .float16), try ops.scalar(0.2, .float16), &[_]c_int{ experts, input }, 2, .float16, key.*, ops.s));
        const sv = try ops.slot();
        try mlx.check(mlx.mlx_random_uniform(sv, try ops.scalar(-0.05, .float16), try ops.scalar(0.05, .float16), &[_]c_int{ experts, output }, 2, .float16, key.*, ops.s));
        projection.* = .{ .trellis = codes.*, .suh = su.*, .svh = sv.* };
    }
    return .{ .gate = projections[0], .up = projections[1], .down = projections[2] };
}

pub fn routing(ops: *Ops, rnd: std.Random, rows: usize, experts: usize) !struct { ids: Arr, scores: Arr } {
    var ids: [8 * topk]u32 = undefined;
    var probs: [8 * topk]f32 = undefined;
    for (0..rows) |r| {
        var pool: [64]u32 = undefined;
        for (pool[0..experts], 0..) |*p, i| p.* = @intCast(i);
        rnd.shuffle(u32, pool[0..experts]);
        for (0..topk) |k| {
            ids[r * topk + k] = pool[k];
            probs[r * topk + k] =(rnd.float(f32) + 0.1) / 6;
        }
    }
    const shape = [_]c_int{ 1, @intCast(rows), topk };
    const f32_scores = try ops.own(mlx.mlx_array_new_data(&probs, &shape, 3, .float32));
    return .{ .ids = try ops.own(mlx.mlx_array_new_data(&ids, &shape, 3, .uint32)), .scores = try ops.cast(f32_scores, .bfloat16) };
}

pub const Weights = struct {
    shared: api.qwen_riders.Shared,
    logit: api.qwen_riders.Affine,
    group: usize,
    trellis: ?TrellisShared = null,

    pub fn deinit(self: *Weights, a: std.mem.Allocator) void {
        if (self.trellis) |*t| t.deinit(a);
    }
};

/// A layer's shared expert at hidden width `hidden_k`: affine8 at `group`, or with `trellis_bits` other than 0 the
/// trunk's trellis banks at that packed rate (96 is 6 bits).
pub fn weights(ops: *Ops, a: std.mem.Allocator, rnd: std.Random, s: mlx.mlx_stream, hidden_k: usize, width: usize, group: usize, gain: f32, trellis_bits: u32) !Weights {
    if (trellis_bits != 0) {
        var t = try TrellisShared.init(ops, a, a, rnd, s, trellis_bits, api.format.Decode.tabled(.spec1, .w15));
        errdefer t.deinit(a);
        return .{ .shared = try t.shared(ops), .logit = t.logit, .group = group, .trellis = t };
    }
    return .{
        .shared = .{
            .gate = try affineBank(ops, a, rnd, width, hidden_k, group, gain),
            .up = try affineBank(ops, a, rnd, width, hidden_k, group, gain),
            .down = try affineBank(ops, a, rnd, hidden_k, width, group, gain),
            .sigmoid = try tfm.swigluSigTable(s, .bfloat16, std.heap.c_allocator),
        },
        .logit = try affineBank(ops, a, rnd, 1, hidden_k, 64, gain),
        .group = group,
    };
}

pub const Chain = union(enum) { bank: api.Bank, ragged: *const api.Ragged };

/// Today's chain for the layer: the shared projections as `moeSharedDown` and `moeSharedGateLogit` run them, then the
/// routed experts with the gated shared add as `moeExl3` runs them.
pub fn separate(ops: *Ops, x: Arr, w: Weights, chain: Chain, ids: Arr, scores: Arr, dec: api.format.Decode) !Arr {
    if (w.trellis) |t| return t.separate(ops, x, chain, ids, scores, dec);
    const s = ops.s;
    const g = try ops.own(try tfm.qmatmulBits(x, w.shared.gate.weight, w.shared.gate.scales, w.shared.gate.biases, 8, @intCast(w.group), .affine, s));
    const u = try ops.own(try tfm.qmatmulBits(x, w.shared.up.weight, w.shared.up.scales, w.shared.up.biases, 8, @intCast(w.group), .affine, s));
    const act = try ops.own((try tfm.fusedSwiGLU(s, g, u)) orelse return error.TestExpectedSwiglu);
    const down = try ops.own(try tfm.qmatmulBits(act, w.shared.down.weight, w.shared.down.scales, w.shared.down.biases, 8, @intCast(w.group), .affine, s));
    const logit = try ops.own(try tfm.qmatmulBits(x, w.logit.weight, w.logit.scales, w.logit.biases, 8, 64, .affine, s));
    return ops.own(switch (chain) {
        .bank => |bank| try api.moeWithGatedShared(s, x, bank, ids, scores, dec, false, down, logit),
        .ragged => |rb| try api.moeRagged(s, x, rb, ids, scores, dec, false, down, logit),
    });
}

fn expectSameBits(want: Arr, got: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(want));
    try mlx.check(mlx.mlx_array_eval(got));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    const n = mlx.mlx_array_size(got);
    const a = mlx.mlx_array_data_bfloat16(want).?[0..n];
    const b = mlx.mlx_array_data_bfloat16(got).?[0..n];
    var live: usize = 0;
    for (a) |v| if (v != 0) {
        live += 1;
    };
    // A chain of zeros would equal itself.
    try std.testing.expect(live > n / 2);
    try std.testing.expectEqualSlices(u16, a, b);
}

const Rates = struct { dec: api.format.Decode, rates: [3]c_int };
const one_bank_cases = [_]Rates{
    .{ .dec = test_dec, .rates = .{ 64, 64, 64 } },
    .{ .dec = test_dec, .rates = .{ 32, 32, 32 } },
    .{ .dec = test_dec, .rates = .{ 44, 44, 64 } },
};

test "Qwen shared expert folded into the one-bank routed launches is bit exact to the separate chain at every row count" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(901);
    const rnd = random.random();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    defer api.qwen_fold.shared_first = false;
    const experts: c_int = 24;
    for (one_bank_cases) |kind| {
        const bank = try routedBank(&ops, experts, kind.rates[0], kind.rates[2], 40 + @as(u64, @intCast(kind.rates[0])));
        for ([_]usize{ 128, 64 }) |group| for ([_]f32{ 1, 0.05 }) |gain| {
            var wops = Ops{ .s = s };
            defer wops.deinit();
            const w = try weights(&wops, std.testing.allocator, rnd, s, hidden, inter, group, gain, 0);
            for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8 }) |rows| {
                var scope = Ops{ .s = s };
                defer scope.deinit();
                const route = try routing(&scope, rnd, rows, @intCast(experts));
                const x = try randomBf16(&scope, std.testing.allocator, rnd, &.{ 1, @intCast(rows), hidden }, 4);
                const want = try separate(&scope, x, w, .{ .bank = bank }, route.ids, route.scores, kind.dec);
                // The router kernel's own logit rides in at one row, the rider's at every width.
                for ([_]bool{ false, true }) |router_logit| for ([_]bool{ false, true }) |first| {
                        api.qwen_fold.shared_first = first;
                    var sh = w.shared;
                    var gate: ?Arr = null;
                    if (router_logit) {
                        gate = try scope.own(try tfm.qmatmulBits(x, w.logit.weight, w.logit.scales, w.logit.biases, 8, 64, .affine, s));
                    } else sh.logit = w.logit;
                    const folded_before = api.qwen_fold.foldedCount();
                    const launches_before = api.qwen_fold.launchCount();
                    const dispatches_before = api.kernels.fusedDispatchCount();
                    const got = try scope.own((try api.qwen_fold.moeGatedSharedFolded(s, x, bank, route.ids, route.scores, kind.dec, sh, gate)) orelse return error.TestExpectedFold);
                    try std.testing.expectEqual(folded_before + 1, api.qwen_fold.foldedCount());
                    try std.testing.expectEqual(launches_before + 3, api.qwen_fold.launchCount());
                    try std.testing.expectEqual(dispatches_before + 3, api.kernels.fusedDispatchCount());
                    try expectSameBits(want, got);
                };
            }
        };
    }
}

const group_sizes = [_]c_int{ 5, 9, 10 };
pub const group_rates = [_][3]c_int{ .{ 32, 32, 32 }, .{ 44, 44, 44 }, .{ 64, 64, 48 } };

fn raggedLayer(ops: *Ops, s: mlx.mlx_stream) !api.Ragged {
    var banks: [3]api.Bank = undefined;
    for (&banks, group_sizes, group_rates, 0..) |*bank, count, r, g| bank.* = try routedBank(ops, count, r[0], r[2], 70 + 10 * @as(u64, @intCast(g)));
    return api.Ragged.init(s, &banks);
}

test "Qwen shared expert folded into the rate-grouped routed launches is bit exact to the separate chain at every row count" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(907);
    const rnd = random.random();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    defer api.rate_groups.fused_shared_first = false;
    var rb = try raggedLayer(&ops, s);
    defer rb.deinit();
    for ([_]api.format.Decode{ test_dec, api.format.Decode.tabled(.spec1, .w15) }) |dec| for ([_]f32{ 1, 0.05 }) |gain| {
        const w = try weights(&ops, std.testing.allocator, rnd, s, hidden, inter, 128, gain, 0);
        for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8 }) |rows| {
            const route = try routing(&ops, rnd, rows, 24);
            const x = try randomBf16(&ops, std.testing.allocator, rnd, &.{ 1, @intCast(rows), hidden }, 4);
            const want = try separate(&ops, x, w, .{ .ragged = &rb }, route.ids, route.scores, dec);
            for ([_]bool{ false, true }) |router_logit| for ([_]bool{ false, true }) |first| {
                api.rate_groups.fused_shared_first = first;
                var sh = w.shared;
                var gate: ?Arr = null;
                if (router_logit) {
                    gate = try ops.own(try tfm.qmatmulBits(x, w.logit.weight, w.logit.scales, w.logit.biases, 8, 64, .affine, s));
                } else sh.logit = w.logit;
                const folded_before = api.rate_groups.qwenFoldedCount();
                const launches_before = api.rate_groups.qwenLaunchCount();
                const dispatches_before = api.kernels.fusedDispatchCount();
                const got = try ops.own((try api.rate_groups.moeGatedSharedFolded(s, x, &rb, route.ids, route.scores, dec, sh, gate)) orelse return error.TestExpectedFold);
                try std.testing.expectEqual(folded_before + 1, api.rate_groups.qwenFoldedCount());
                try std.testing.expectEqual(launches_before + 3, api.rate_groups.qwenLaunchCount());
                try std.testing.expectEqual(dispatches_before + 3, api.kernels.fusedDispatchCount());
                try expectSameBits(want, got);
            };
        }
    };
}

test "Qwen shared expert fold declines what the separate chain must run" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(911);
    const rnd = random.random();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var rb = try raggedLayer(&ops, s);
    defer rb.deinit();
    const bank = try routedBank(&ops, 24, 64, 64, 91);
    const w = try weights(&ops, std.testing.allocator, rnd, s, hidden, inter, 128, 1, 0);
    var with_logit = w.shared;
    with_logit.logit = w.logit;
    const four = try routing(&ops, rnd, 4, 24);
    const x4 = try randomBf16(&ops, std.testing.allocator, rnd, &.{ 1, 4, hidden }, 4);
    const before = api.qwen_fold.foldedCount() + api.rate_groups.qwenFoldedCount();
    const dispatches = api.kernels.fusedDispatchCount();
    const fold = api.qwen_fold.moeGatedSharedFolded;
    const rag = api.rate_groups.moeGatedSharedFolded;
    // Nine rows are past the verify width.
    const nine = try routing(&ops, rnd, 8, 24);
    const x9 = try randomBf16(&ops, std.testing.allocator, rnd, &.{ 1, 9, hidden }, 4);
    try std.testing.expect((try fold(s, x9, bank, nine.ids, nine.scores, test_dec, with_logit, null)) == null);
    try std.testing.expect((try rag(s, x9, &rb, nine.ids, nine.scores, test_dec, with_logit, null)) == null);
    // Ids of another row count, f16 activations, a missing gate logit, a gate of the wrong width.
    try std.testing.expect((try fold(s, x4, bank, nine.ids, nine.scores, test_dec, with_logit, null)) == null);
    try std.testing.expect((try fold(s, try ops.cast(x4, .float16), bank, four.ids, four.scores, test_dec, with_logit, null)) == null);
    try std.testing.expect((try fold(s, x4, bank, four.ids, four.scores, test_dec, w.shared, null)) == null);
    try std.testing.expect((try rag(s, x4, &rb, four.ids, four.scores, test_dec, w.shared, null)) == null);
    const short = try randomBf16(&ops, std.testing.allocator, rnd, &.{3}, 1);
    try std.testing.expect((try fold(s, x4, bank, four.ids, four.scores, test_dec, w.shared, short)) == null);
    // Shared banks the riders do not read: four-bit codes, group 32, a hidden width off the 128 grid, f32 scales.
    var narrow = with_logit;
    narrow.gate.weight = try ops.slice(w.shared.gate.weight, 1, 0, @intCast(hidden / 8));
    try std.testing.expect((try fold(s, x4, bank, four.ids, four.scores, test_dec, narrow, null)) == null);
    var wide_scales = with_logit;
    wide_scales.down.scales = try ops.cast(w.shared.down.scales, .float32);
    try std.testing.expect((try fold(s, x4, bank, four.ids, four.scores, test_dec, wide_scales, null)) == null);
    // A codebook whose table the decode does not carry.
    try std.testing.expect((try fold(s, x4, bank, four.ids, four.scores, .{ .codebook = .spec1, .window = .w15 }, with_logit, null)) == null);
    try std.testing.expectEqual(before, api.qwen_fold.foldedCount() + api.rate_groups.qwenFoldedCount());
    try std.testing.expectEqual(dispatches, api.kernels.fusedDispatchCount());
}

/// A trellis shared expert at the trunk's bits `n`: the stacked gate/up and the down, with the loader's stack.
pub const TrellisShared = struct {
    held: std.ArrayList(Arr) = .empty,
    stack: trunk_stq.Shared,
    down: trunk_stq.Triple,
    logit: api.qwen_riders.Affine,
    dec: api.format.Decode,

    /// `bytes` backs the host banks and must outlive the weights' build only; `a` holds the arrays' list.
    pub fn init(ops: *Ops, a: std.mem.Allocator, bytes: std.mem.Allocator, rnd: std.Random, s: mlx.mlx_stream, n: u32, dec: api.format.Decode) !TrellisShared {
        var self: TrellisShared = undefined;
        self.held = .empty;
        self.dec = dec;
        var parts: [2]trunk_stq.Triple = undefined;
        for (&parts) |*part| {
            const bank = try trunk_stq.TestBank.init(bytes, rnd, hidden, inter, n);
            part.* = try bank.triple(s);
            try self.held.appendSlice(a, &.{ part.w, part.s, part.b });
        }
        self.stack = (try trunk_stq.fuse(&.{ &parts[0], &parts[1] }, &self.held, a, s)) orelse return error.TestExpectedStack;
        const down = try trunk_stq.TestBank.init(bytes, rnd, inter, hidden, n);
        self.down = try down.triple(s);
        try self.held.appendSlice(a, &.{ self.down.w, self.down.s, self.down.b });
        self.logit = try affineBank(ops, bytes, rnd, 1, hidden, 64, 1);
        return self;
    }

    pub fn deinit(self: *TrellisShared, a: std.mem.Allocator) void {
        for (self.held.items) |v| _ = mlx.mlx_array_free(v);
        self.held.deinit(a);
    }

    pub fn shared(self: *const TrellisShared, ops: *Ops) !api.qwen_riders.Shared {
        return .{
            .sigmoid = try tfm.swigluSigTable(ops.s, .bfloat16, std.heap.c_allocator),
            .trellis = .{ .gate_up = self.stack.p, .down = .{ .trellis = self.down.w, .suh = self.down.s, .svh = self.down.b }, .dec = self.dec },
        };
    }

    /// Today's chain for the layer: the stacked gate/up, SwiGLU and down as `moeSharedDown` runs them.
    pub fn separate(self: *const TrellisShared, ops: *Ops, x: Arr, chain: Chain, ids: Arr, scores: Arr, dec: api.format.Decode) !Arr {
        const s = ops.s;
        var gu: [2]Arr = undefined;
        if (!try trunk_stq.sharedProject(s, x, self.stack, self.dec, &gu, false)) return error.TestExpectedStack;
        const g = try ops.own(gu[0]);
        const u = try ops.own(gu[1]);
        const act = try ops.own((try tfm.fusedSwiGLU(s, g, u)) orelse return error.TestExpectedSwiglu);
        const down = try ops.own(try trunk_stq.matmul(s, act, self.down.w, self.down.s, self.down.b, self.dec, false));
        const logit = try ops.own(try tfm.qmatmulBits(x, self.logit.weight, self.logit.scales, self.logit.biases, 8, 64, .affine, s));
        return ops.own(switch (chain) {
            .bank => |bank| try api.moeWithGatedShared(s, x, bank, ids, scores, dec, false, down, logit),
            .ragged => |rb| try api.moeRagged(s, x, rb, ids, scores, dec, false, down, logit),
        });
    }
};

test "Qwen trellis shared expert folded into the routed launches is bit exact to the separate chain at every row count" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s) or !api.kernels.gemmNaxOn()) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(913);
    const rnd = random.random();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var rb = try raggedLayer(&ops, s);
    defer rb.deinit();
    const bank = try routedBank(&ops, 24, 44, 64, 53);
    const trunk_dec = api.format.Decode.tabled(.spec1, .w15);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for ([_]u32{ 96, 64 }) |bits| {
        var w = try TrellisShared.init(&ops, std.testing.allocator, arena.allocator(), rnd, s, bits, trunk_dec);
        defer w.deinit(std.testing.allocator);
        for (0..2) |chain_kind| for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8 }) |rows| {
            const dec = if (chain_kind == 0) test_dec else api.format.Decode.tabled(.spec1, .w15);
            const route = try routing(&ops, rnd, rows, 24);
            const x = try randomBf16(&ops, std.testing.allocator, rnd, &.{ 1, @intCast(rows), hidden }, 4);
            const chain: Chain = if (chain_kind == 0) .{ .bank = bank } else .{ .ragged = &rb };
            const want = try w.separate(&ops, x, chain, route.ids, route.scores, dec);
            for ([_]bool{ false, true }) |router_logit| {
                var sh = try w.shared(&ops);
                var gate: ?Arr = null;
                if (router_logit) {
                    gate = try ops.own(try tfm.qmatmulBits(x, w.logit.weight, w.logit.scales, w.logit.biases, 8, 64, .affine, s));
                } else sh.logit = w.logit;
                const dispatches_before = api.kernels.fusedDispatchCount();
                const got = try ops.own((if (chain_kind == 0)
                    try api.qwen_fold.moeGatedSharedFolded(s, x, bank, route.ids, route.scores, dec, sh, gate)
                else
                    try api.rate_groups.moeGatedSharedFolded(s, x, &rb, route.ids, route.scores, dec, sh, gate)) orelse return error.TestExpectedFold);
                try std.testing.expectEqual(dispatches_before + 3, api.kernels.fusedDispatchCount());
                try expectSameBits(want, got);
            }
        };
    }
}

test "Qwen trellis shared expert fold declines what the separate chain must run" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s) or !api.kernels.gemmNaxOn()) return error.SkipZigTest;
    var random = std.Random.DefaultPrng.init(917);
    const rnd = random.random();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const bank = try routedBank(&ops, 24, 64, 64, 91);
    var w = try TrellisShared.init(&ops, std.testing.allocator, arena.allocator(), rnd, s, 96, api.format.Decode.tabled(.spec1, .w15));
    defer w.deinit(std.testing.allocator);
    var sh = try w.shared(&ops);
    sh.logit = w.logit;
    const four = try routing(&ops, rnd, 4, 24);
    const x4 = try randomBf16(&ops, std.testing.allocator, rnd, &.{ 1, 4, hidden }, 4);
    const before = api.qwen_fold.foldedCount();
    const dispatches = api.kernels.fusedDispatchCount();
    // A trunk codebook whose table the decode does not carry.
    var no_table = sh;
    no_table.trellis.?.dec = .{ .codebook = .spec1, .window = .w15 };
    try std.testing.expect((try api.qwen_fold.moeGatedSharedFolded(s, x4, bank, four.ids, four.scores, test_dec, no_table, null)) == null);
    // A down bank off the gate/up width, a gate/up bank off the hidden width.
    var wrong_down = sh;
    wrong_down.trellis.?.down = sh.trellis.?.gate_up;
    try std.testing.expect((try api.qwen_fold.moeGatedSharedFolded(s, x4, bank, four.ids, four.scores, test_dec, wrong_down, null)) == null);
    const narrow = try randomBf16(&ops, std.testing.allocator, rnd, &.{ 1, 4, hidden / 2 }, 4);
    try std.testing.expect((try api.qwen_fold.moeGatedSharedFolded(s, narrow, bank, four.ids, four.scores, test_dec, sh, null)) == null);
    try std.testing.expectEqual(before, api.qwen_fold.foldedCount());
    try std.testing.expectEqual(dispatches, api.kernels.fusedDispatchCount());
}
