//! Hy4-Flash STQ packs (part 1w) on the tiny STQ pack: `HY4_STQ_TINY` (the identity-Hessian pack `sushi convert exl3`
//! writes, beamster test-fixtures/hy4-stq-tiny) beside `HY4_BF16_TINY_STQ` (its BF16 source, test-fixtures/
//! hy4-bf16-tiny-stq, the dumper's `--geometry stq`: every trunk linear and expert side on the 128 grid).
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const model_mod = @import("../model.zig");
const hy4_config = @import("hy4_config.zig");
const hy4_stq = @import("hy4_stq.zig");
const fwd = @import("hy4_forward.zig");
const Arr = mlx.mlx_array;

/// A measurement line, printed only under `HY4_STQ_DIAG` (a passing test prints nothing).
fn diag(comptime fmt: []const u8, args: anytype) void {
    if (std.c.getenv("HY4_STQ_DIAG") != null) std.debug.print(fmt, args);
}

const Packs = struct {
    stq_dir: []const u8 = "",
    stq: model_mod.Weights,
    src: model_mod.Weights,
    config: hy4_config.Config,
    quant: hy4_stq.Quant,

    fn open(s: mlx.mlx_stream) !Packs {
        const stq_dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
        const src_dir = std.mem.span(std.c.getenv("HY4_BF16_TINY_STQ") orelse return error.SkipZigTest);
        const a = testing.allocator;
        var p: Packs = .{ .stq = model_mod.Weights.init(a), .src = model_mod.Weights.init(a), .config = undefined, .quant = undefined };
        errdefer p.close();
        // The pack's own shards (the Engram clone is read by the index alone).
        for ([_][]const u8{ "model-globals", "model-trunk-00", "model-trunk-01", "model-trunk-02", "model-trunk-03", "model-trunk-04", "model-trunk-05", "model-layer-01", "model-layer-02", "model-layer-03", "model-layer-04", "model-layer-05" }) |shard|
            try loadInto(&p.stq, stq_dir, shard, s);
        try loadInto(&p.src, src_dir, "model", s);
        const io = std.Io.Threaded.global_single_threaded.io();
        var d = try std.Io.Dir.openDirAbsolute(io, stq_dir, .{});
        defer d.close(io);
        const text = try d.readFileAlloc(io, "config.json", a, .limited(1 << 20));
        defer a.free(text);
        p.config = try hy4_config.parseText(text);
        p.quant = try hy4_stq.parseQuantText(a, text);
        p.stq_dir = stq_dir;
        return p;
    }

    fn close(p: *Packs) void {
        p.stq.deinit();
        p.src.deinit();
    }
};

fn loadInto(w: *model_mod.Weights, dir: []const u8, shard: []const u8, s: mlx.mlx_stream) !void {
    const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/{s}.safetensors", .{ dir, shard }, 0);
    defer testing.allocator.free(path);
    try model_mod.loadSafetensorsFile(testing.allocator, w, path.ptr, s, true);
}

/// Deterministic rows in [-1, 1).
fn rows(o: *Ops, t: usize, in: usize, seed: u64) !Arr {
    const v = try testing.allocator.alloc(f32, t * in);
    defer testing.allocator.free(v);
    var r = std.Random.DefaultPrng.init(seed);
    for (v) |*d| d.* = r.random().float(f32) * 2 - 1;
    return o.own(mlx.mlx_array_new_data(v.ptr, &[_]c_int{ @intCast(t), @intCast(in) }, 2, .float32));
}

/// ‖got − want‖ / ‖want‖ over every value.
fn relRms(o: *Ops, got: Arr, want: Arr) !f64 {
    const g = try o.contiguous(try o.cast(got, .float32));
    const r = try o.contiguous(try o.cast(want, .float32));
    try mlx.check(mlx.mlx_array_eval(g));
    try mlx.check(mlx.mlx_array_eval(r));
    const gv = mlx.mlx_array_data_float32(g).?[0..mlx.mlx_array_size(g)];
    const rv = mlx.mlx_array_data_float32(r).?[0..mlx.mlx_array_size(r)];
    var num: f64 = 0;
    var den: f64 = 0;
    for (gv, rv) |x, y| {
        num += (@as(f64, x) - y) * (@as(f64, x) - y);
        den += @as(f64, y) * y;
    }
    return @sqrt(num / den);
}

/// The trunk bases an STQ pack stores as trellis banks, for layer `l`.
fn trunkBases(c: hy4_config.Config, l: usize, buf: *[8][96]u8, names: *[8][]const u8) ![]const []const u8 {
    var n: usize = 0;
    const attn = [_][]const u8{ "q_a_proj", "q_b_proj", "linear_gate", "o_proj" };
    for (attn) |b| {
        names[n] = try std.fmt.bufPrint(&buf[n], "model.layers.{d}.self_attn.{s}", .{ l, b });
        n += 1;
    }
    const ffn: []const []const u8 = if (l < c.dense_layers) &.{ "gate_proj", "up_proj", "down_proj" } else &.{ "shared_experts.gate_proj", "shared_experts.up_proj", "shared_experts.down_proj", "routed_expert_down_proj" };
    for (ffn) |b| {
        names[n] = try std.fmt.bufPrint(&buf[n], "model.layers.{d}.mlp.{s}", .{ l, b });
        n += 1;
    }
    return names[0..n];
}

/// The bar a K6 SPEC1 bank's product is held to against its BF16 source on uniform rows: relative RMS error (an
/// identity-Hessian round of random weights; the tiny's measured worst is printed beside it).
const k6_bar: f64 = 2.5e-2;

test "hy4 stq: every trunk linear the tiny STQ pack stores as a trellis bank (attention, dense and shared FFN, latent projections, head) multiplies within the K6 bar of its BF16 source, and kv_b decodes to its source (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var p = try Packs.open(es);
    defer p.close();
    const dec = p.quant.trunk orelse return error.TrunkQuantMissing;
    try testing.expect(p.quant.expert != null and !p.quant.synthetic);
    var owned: std.ArrayList(Arr) = .empty;
    defer {
        for (owned.items) |x| _ = mlx.mlx_array_free(x);
        owned.deinit(testing.allocator);
    }
    var o: Ops = .{ .s = es };
    defer o.deinit();
    var worst: f64 = 0;
    var worst_name: [96]u8 = undefined;
    var worst_len: usize = 0;
    var count: usize = 0;
    for (0..p.config.layers + 1) |l| {
        var buf: [8][96]u8 = undefined;
        var names: [8][]const u8 = undefined;
        const bases: []const []const u8 = if (l == p.config.layers) &.{"lm_head"} else try trunkBases(p.config, l, &buf, &names);
        for (bases) |base| {
            const bank = (try hy4_stq.load(&p.stq, base, dec, &owned, testing.allocator, es)) orelse {
                std.debug.print("{s} is not a trellis bank in the STQ tiny\n", .{base});
                return error.NotTrellis;
            };
            var wn: [112]u8 = undefined;
            const src = p.src.get(try std.fmt.bufPrint(&wn, "{s}.weight", .{base})).?;
            const x = try rows(&o, 7, @intCast(bank.inDim()), count + 1);
            const got = try hy4_stq.apply(&o, x, bank);
            const want = try o.binary(.mm, try o.cast(try o.cast(x, .bfloat16), .float32), try o.transpose(try o.cast(src, .float32), &.{ 1, 0 }));
            const e = try relRms(&o, got, want);
            if (e > worst) {
                worst = e;
                worst_len = base.len;
                @memcpy(worst_name[0..base.len], base);
            }
            count += 1;
        }
        if (l < p.config.layers) {
            var kb: [96]u8 = undefined;
            const base = try std.fmt.bufPrint(&kb, "model.layers.{d}.self_attn.kv_b_proj", .{l});
            const bank = (try hy4_stq.load(&p.stq, base, dec, &owned, testing.allocator, es)).?;
            const dense = try o.own(try hy4_stq.decodeDense(testing.allocator, es, bank));
            var wn: [112]u8 = undefined;
            const src = p.src.get(try std.fmt.bufPrint(&wn, "{s}.weight", .{base})).?;
            try testing.expectEqualSlices(c_int, mlx.getShape(src), mlx.getShape(dense));
            const e = try relRms(&o, dense, src);
            if (e > worst) {
                worst = e;
                worst_len = base.len;
                @memcpy(worst_name[0..base.len], base);
            }
            count += 1;
        }
    }
    diag("[hy4-stq] {d} trunk banks: worst relative RMS {e:.3} ({s}), bar {e:.1}\n", .{ count, worst, worst_name[0..worst_len], k6_bar });
    try testing.expect(worst <= k6_bar);
}

/// Host rows of `a` as f32.
fn hostRows(o: *Ops, a: Arr) ![]const f32 {
    const c = try o.contiguous(try o.cast(a, .float32));
    try mlx.check(mlx.mlx_array_eval(c));
    return mlx.mlx_array_data_float32(c).?[0..mlx.mlx_array_size(c)];
}

/// The bar the tiny STQ pack's logits are held to against its BF16 source's: mean KL(BF16 || STQ) per row (K6 trunk,
/// K4 PAIR1 experts, identity Hessian, random weights; the measured mean and top-1 agreement are printed beside it).
/// A pack whose narrowest expert group runs below K4 scales it as the KLD scales, by the square of its error
/// (`expertBar`): the two-group tiny (K3 and K4 experts) measures 5.2e-2.
const model_kld_bar: f64 = 5e-2;

test "hy4 stq: the tiny STQ pack (K6 trunk, PAIR1 experts resident in their rate groups, K6 head and embedding) prefills a prompt and decodes a row within the KLD bar of its rates against its BF16 source (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var p = try Packs.open(es);
    defer p.close();
    const vocab: usize = p.config.vocab;
    var ids: [41]u32 = undefined;
    var r = std.Random.DefaultPrng.init(0x4f7);
    for (&ids) |*d| d.* = r.random().uintLessThan(u32, @intCast(vocab - 1)) + 1;
    var o: Ops = .{ .s = es };
    defer o.deinit();
    // The teacher: the BF16 source on the engine, its products as a served BF16 load runs them.
    const want = blk: {
        const m = try hy4_model.Model.init(a, p.config, &p.src, es);
        defer m.deinit();
        const pre = try o.own(try m.forward(ids[0..40]));
        const dec = try o.own(try m.forward(ids[40..41]));
        break :blk try o.own(try o.result(try o.contiguous(try o.concat(&.{ pre, dec }, 0))));
    };
    var narrowest: u32 = std.math.maxInt(u32);
    const got = blk: {
        const m = try hy4_model.Model.initWith(a, p.config, &p.stq, es, .{ .dir = p.stq_dir, .quant = p.quant });
        defer m.deinit();
        for (m.stq_experts) |maybe| if (maybe) |rb| for (rb.rates[0..rb.count]) |rates| for (rates) |n| {
            narrowest = @min(narrowest, n);
        };
        const pre = try o.own(try m.forward(ids[0..40]));
        const dec = try o.own(try m.forward(ids[40..41]));
        break :blk try o.own(try o.result(try o.contiguous(try o.concat(&.{ pre, dec }, 0))));
    };
    const scale = expertBar(narrowest) / expert_bar;
    const kld_bar = model_kld_bar * scale * scale;
    const w = try hostRows(&o, want);
    const g = try hostRows(&o, got);
    var kld_sum: f64 = 0;
    var kld_max: f64 = 0;
    var agree: usize = 0;
    for (0..ids.len) |row| {
        const pw = w[row * vocab ..][0..vocab];
        const pg = g[row * vocab ..][0..vocab];
        const lw = logSumExp(pw);
        const lg = logSumExp(pg);
        var kl: f64 = 0;
        var aw: usize = 0;
        var ag: usize = 0;
        for (pw, pg, 0..) |x, y, i| {
            const lpw = @as(f64, x) - lw;
            kl += @exp(lpw) * (lpw - (@as(f64, y) - lg));
            if (x > pw[aw]) aw = i;
            if (y > pg[ag]) ag = i;
        }
        kld_sum += kl;
        kld_max = @max(kld_max, kl);
        agree += @intFromBool(aw == ag);
    }
    const mean = kld_sum / @as(f64, @floatFromInt(ids.len));
    diag("[hy4-stq] tiny STQ vs BF16 over {d} rows: KLD mean {e:.3}, max {e:.3}; top-1 {d}/{d}; bar {e:.1} (narrowest expert n {d})\n", .{ ids.len, mean, kld_max, agree, ids.len, kld_bar, narrowest });
    try testing.expect(mean <= kld_bar);
}

fn logSumExp(v: []const f32) f64 {
    var mx: f64 = -std.math.inf(f64);
    for (v) |x| mx = @max(mx, x);
    var sum: f64 = 0;
    for (v) |x| sum += @exp(@as(f64, x) - mx);
    return mx + @log(sum);
}

/// The bar the trellis expert chain's routed sum is held to against the BF16 experts' on the same rows and picks:
/// relative RMS. K6 measures 1.9% a linear here (above); K4 has two bits fewer, about 4x that, 7.7% a projection, and
/// an expert chains three (gate, up, down): about 13%. Measured on the tiny: 10.0-13.3% at both widths. An expert read
/// from the wrong bank lands near 1.4, which the test's control shows. A group at a lower rate doubles it a bit fewer
/// (`expertBar`).
const expert_bar: f64 = 1.5e-1;

/// The bar for a layer whose narrowest group runs at `n` halfwords a tile (PAIR1: K = n / 16): K4's, doubled for each
/// bit below 4.
fn expertBar(n: u32) f64 {
    const k: f64 = @as(f64, @floatFromInt(n)) / 16.0;
    return expert_bar * std.math.pow(f64, 2, @max(0, 4 - k));
}

/// The source expert behind each of the pack's experts: the pack writes the router's rows in its group order, each a
/// source row as stored.
fn sourceOrder(o: *Ops, pack_router: Arr, src_router: Arr, out: []u32) !void {
    const pr = try o.contiguous(pack_router);
    const sr = try o.contiguous(src_router);
    try mlx.check(mlx.mlx_array_eval(pr));
    try mlx.check(mlx.mlx_array_eval(sr));
    const d: usize = @intCast(mlx.getShape(pr)[1]);
    const pv = mlx.mlx_array_data_bfloat16(pr).?[0 .. out.len * d];
    const sv = mlx.mlx_array_data_bfloat16(sr).?[0 .. out.len * d];
    for (out, 0..) |*dst, i| {
        dst.* = for (0..out.len) |j| {
            if (std.mem.eql(u16, pv[i * d ..][0..d], sv[j * d ..][0..d])) break @intCast(j);
        } else return error.RouterRowNotInSource;
    }
}

test "hy4 stq: each MoE layer's routed experts through the clamped trellis chain (PAIR1, its rate groups in the router's order) sum the same picks within the bar of their rates against the BF16 experts, at a decode width and a prefill width (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {

    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var p = try Packs.open(es);
    defer p.close();
    const c = p.config;
    const g: fwd.MoeGeometry = .{ .top_k = @intCast(c.top_k), .inter = @intCast(c.moe_inter), .scale = c.route_scale, .swiglu_limit = c.swiglu_limit, .eps = c.eps };
    var o: Ops = .{ .s = es };
    defer o.deinit();
    var worst: f64 = 0;
    var r = std.Random.DefaultPrng.init(0xe4);
    for (c.dense_layers..c.layers) |l| {
        var rb = (try hy4_stq.loadExperts(&p.stq, l, es, a)).?;
        defer rb.deinit();
        var nb: [96]u8 = undefined;
        const gate_up = p.src.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.experts.gate_up_proj", .{l})).?;
        const down = p.src.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.experts.down_proj", .{l})).?;
        // The pack's expert i is the source's `order[i]` (the router rows in group order).
        var order: [64]u32 = undefined;
        const pack_router = p.stq.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.gate.weight", .{l})).?;
        try sourceOrder(&o, pack_router, p.src.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.gate.weight", .{l})).?, order[0..c.experts]);
        var narrowest: u32 = std.math.maxInt(u32);
        for (rb.rates[0..rb.count]) |rates| for (rates) |n| {
            narrowest = @min(narrowest, n);
        };
        const bar = expertBar(narrowest);
        const src_picks = try a.alloc(u32, c.top_k * 40);
        defer a.free(src_picks);
        // 3 rows take the decode chain, 40 the prefill GEMMs.
        for ([_]usize{ 3, 40 }) |t| {
            const k: usize = c.top_k;
            const picks = try a.alloc(u32, t * k);
            defer a.free(picks);
            const wts = try a.alloc(f32, t * k);
            defer a.free(wts);
            for (0..t) |row| {
                // k distinct experts a row.
                var chosen: usize = 0;
                while (chosen < k) {
                    const e = r.random().uintLessThan(u32, c.experts);
                    if (std.mem.indexOfScalar(u32, picks[row * k ..][0..chosen], e) != null) continue;
                    picks[row * k + chosen] = e;
                    wts[row * k + chosen] = 0.25 + r.random().float(f32);
                    chosen += 1;
                }
            }
            const lat = try o.binary(.mul, try rows(&o, t, c.moe_inter, l * 100 + t), try o.scalar(4, .float32));
            const pk = try o.own(mlx.mlx_array_new_data(picks.ptr, &[_]c_int{ @intCast(t), @intCast(k) }, 2, .uint32));
            const wa = try o.own(mlx.mlx_array_new_data(wts.ptr, &[_]c_int{ @intCast(t), @intCast(k) }, 2, .float32));
            const got = try o.own(try hy4_stq.routed(es, p.quant.expert.?, c.swiglu_limit, &rb, lat, pk, wa));
            for (picks, src_picks[0..picks.len]) |v, *d| d.* = order[v];
            const want = try o.own(try fwd.routedExperts(a, es, lat, wa, null, src_picks[0..picks.len], g, gate_up, down, null));
            const e = try relRms(&o, got, want);
            worst = @max(worst, e / bar * expert_bar);
            // The route split into two expert groups (experts below half, the rest), each at weight 0 outside its
            // group, sums to the whole route's: the chain sums in f32, so the groups add as one.
            {
                const half = c.experts / 2;
                const lo = try a.alloc(f32, t * k);
                defer a.free(lo);
                const hi = try a.alloc(f32, t * k);
                defer a.free(hi);
                for (picks, wts, lo, hi) |pv, wv, *l_, *h_| {
                    l_.* = if (pv < half) wv else 0;
                    h_.* = if (pv < half) 0 else wv;
                }
                const wl = try o.own(mlx.mlx_array_new_data(lo.ptr, &[_]c_int{ @intCast(t), @intCast(k) }, 2, .float32));
                const wh = try o.own(mlx.mlx_array_new_data(hi.ptr, &[_]c_int{ @intCast(t), @intCast(k) }, 2, .float32));
                const parts = try o.binary(.add, try o.own(try hy4_stq.routed(es, p.quant.expert.?, c.swiglu_limit, &rb, lat, pk, wl)), try o.own(try hy4_stq.routed(es, p.quant.expert.?, c.swiglu_limit, &rb, lat, pk, wh)));
                try testing.expect(try relRms(&o, parts, got) <= 1e-6);
            }
            // The control: the BF16 experts one bank over.
            for (picks, src_picks[0..picks.len]) |v, *d| d.* = order[(v + 1) % c.experts];
            const off = try o.own(try fwd.routedExperts(a, es, lat, wa, null, src_picks[0..picks.len], g, gate_up, down, null));
            const e_off = try relRms(&o, got, off);
            diag("[hy4-stq] layer {d} rows {d} ({d} groups, narrowest n {d}): routed relative RMS {e:.3} against the bar {e:.2} (one bank over: {e:.3})\n", .{ l, t, rb.count, narrowest, e, bar, e_off });
            try testing.expect(e_off > 3 * bar);
        }
    }
    try testing.expect(worst <= expert_bar);
}

test "hy4 stq served: the tiny STQ pack loaded through the shared loader, its experts resident, answers the logits the model gives on the pack's tensors, prefill and decode (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    try servedStqSaid(null, exact_bar);
}

test "hy4 stq served: its experts streamed from SSD through a cache of 8 slots a layer (every route one group) answer the same logits (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    try servedStqSaid(8, exact_bar);
}

test "hy4 stq served: through a cache of 4 slots a layer (the prefill's route in groups, each refilling the cache) answer the same logits (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    try servedStqSaid(4, exact_bar);
}

test "hy4 stq served: a decode row over resident experts (the pack's tensors, or a stream holding every expert) waits for the GPU only at its readout (HY4_STQ_TINY)" {
    try residentWaits(null);
    try residentWaits(8);
}

/// The engine's waits over one decode row after a prompt: the row's forward and its logits read to the host.
fn residentWaits(slots: ?u64) !void {
    const Trunk = @import("../trunk.zig").Trunk;
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = slots != null;
    config.expert_source_dir = try a.dupe(u8, dir);
    if (slots) |n| config.expert_cache_bytes = n * config.num_hidden_layers * (try expertBytes(io, dir));
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    if (xfm.expert_stream) |e| for (0..config.num_hidden_layers) |l| if (l > 0) try testing.expect(e.residentRoute(@intCast(l)) != null);
    var o: Ops = .{ .s = xfm.s };
    defer o.deinit();
    var ids: [41]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast(1 + (i * 7919) % 1000);
    _ = try hostRows(&o, try o.own(try xfm.forward(try o.own(mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, 40 }, 2, .uint32)))));
    const before = mlx.engine.stats().waits;
    _ = try hostRows(&o, try o.own(try xfm.forward(try o.own(mlx.mlx_array_new_data(&ids[40], &[_]c_int{ 1, 1 }, 2, .uint32)))));
    const waits = mlx.engine.stats().waits - before;
    if (waits != 1) std.debug.print("a resident decode row waited {d} times (slots {?d})\n", .{ waits, slots });
    try testing.expectEqual(@as(u64, 1), waits);
}

test "hy4 stq served: --expert-cache-bias on a streamed load raises the decode's cache hit rate, and at its most no decode route misses (HY4_STQ_TINY)" {
    const settings = @import("../model_settings.zig");
    const exact = try streamedDecode(4, 0);
    const biased = try streamedDecode(4, settings.expert_cache_bias_max);
    // The exact picks miss the cache: the case has something to raise.
    try testing.expect(exact.hits < exact.routes);
    try testing.expect(biased.hits * exact.routes > exact.hits * biased.routes);
    // 8 outranks any sigmoid score plus the router's bias: every pick is an expert the layer's cache holds.
    try testing.expectEqual(biased.routes, biased.hits);
}

test "hy4 stq served: a resident stream's log carries no [expert-stream] breakdown, a streamed one's and a resident one's at debug do (HY4_STQ_TINY)" {
    try testing.expectEqual(@as(usize, 0), try breakdownLines(8, .info));
    try testing.expect(try breakdownLines(4, .info) > 0);
    try testing.expect(try breakdownLines(8, .debug) > 0);
}

/// The `[expert-stream]` breakdown lines the server log gets over a 40-row prompt and two decode rows (the forwards the
/// breakdown's schedule names), the tiny STQ pack's experts streamed through `slots` cache slots a layer (8 holds every
/// expert resident), at log level `level`.
fn breakdownLines(slots: u64, level: @import("../log.zig").Level) !usize {
    const log = @import("../log.zig");
    const Trunk = @import("../trunk.zig").Trunk;
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [512]u8 = undefined;
    var path_buf: [600]u8 = undefined;
    const log_path = try std.fmt.bufPrint(&path_buf, "{s}/server.log", .{dir_buf[0..try tmp.dir.realPath(testing.io, &dir_buf)]});
    const level_was = if (log.isDebug()) log.Level.debug else log.Level.info;
    defer log.setLevel(level_was);
    log.setLevel(level);
    {
        try log.openFile(log_path, 0);
        defer log.closeFile();
        var config = try model_mod.parseConfig(io, a, dir);
        defer config.deinit(a);
        config.expert_streaming = true;
        config.expert_source_dir = try a.dupe(u8, dir);
        config.expert_cache_bytes = slots * config.num_hidden_layers * (try expertBytes(io, dir));
        var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
        defer weights.deinit();
        model_mod.resolveWeightPrefix(&config, &weights);
        var xfm = try Trunk.init(io, a, config, &weights);
        defer xfm.deinit();
        for (1..config.num_hidden_layers) |l| try testing.expectEqual(slots >= 8, xfm.expert_stream.?.residentRoute(@intCast(l)) != null);
        var o: Ops = .{ .s = xfm.s };
        defer o.deinit();
        var ids: [42]u32 = undefined;
        for (&ids, 0..) |*d, i| d.* = @intCast(1 + (i * 7919) % 1000);
        _ = try hostRows(&o, try o.own(try xfm.forward(try o.own(mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, 40 }, 2, .uint32)))));
        for (ids[40..]) |*id| _ = try hostRows(&o, try o.own(try xfm.forward(try o.own(mlx.mlx_array_new_data(id, &[_]c_int{ 1, 1 }, 2, .uint32)))));
    }
    const text = try tmp.dir.readFileAlloc(testing.io, "server.log", a, .limited(1 << 22));
    defer a.free(text);
    return std.mem.count(u8, text, "[expert-stream] forward=") + std.mem.count(u8, text, "[expert-stream] route forward=");
}

const DecodeRoutes = struct { routes: u64, hits: u64 };

/// The expert stream's routes and cache hits over 24 decode rows after a 40-row prompt, the tiny STQ pack's experts
/// streamed through `slots` cache slots a layer under `--expert-cache-bias` `bias` (as the model setting).
fn streamedDecode(slots: u64, bias: f32) !DecodeRoutes {
    const Trunk = @import("../trunk.zig").Trunk;
    const settings = @import("../model_settings.zig");
    const expert_stats = @import("../expert_stats.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const flag_was = settings.expert_cache_bias_flag;
    defer settings.expert_cache_bias_flag = flag_was;
    settings.expert_cache_bias_flag = null;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.expert_source_dir = try a.dupe(u8, dir);
    config.expert_cache_bytes = slots * config.num_hidden_layers * (try expertBytes(io, dir));
    config.expert_cache_bias_override = bias;
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    for (1..config.num_hidden_layers) |l| try testing.expect(xfm.expert_stream.?.residentRoute(@intCast(l)) == null);
    var o: Ops = .{ .s = xfm.s };
    defer o.deinit();
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast(1 + (i * 7919) % 1000);
    _ = try hostRows(&o, try o.own(try xfm.forward(try o.own(mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, 40 }, 2, .uint32)))));
    const before = expert_stats.process.snapshot();
    for (ids[40..]) |*id| _ = try hostRows(&o, try o.own(try xfm.forward(try o.own(mlx.mlx_array_new_data(id, &[_]c_int{ 1, 1 }, 2, .uint32)))));
    const after = expert_stats.process.snapshot();
    const got: DecodeRoutes = .{ .routes = after.routes - before.routes, .hits = after.hits - before.hits };
    diag("streamed decode, {d} slots, bias {d}: {d}/{d} hits\n", .{ slots, bias, got.hits, got.routes });
    return got;
}

/// One routed expert's stored bytes in the pack at `dir`: layer 1's group 0, its banks and scales, over its experts.
fn expertBytes(io: std.Io, dir: []const u8) !u64 {
    _ = io;
    var w = model_mod.Weights.init(testing.allocator);
    defer w.deinit();
    const es = mlx.engine.stream();
    try loadInto(&w, dir, "model-layer-01", es);
    var total: u64 = 0;
    for ([_][]const u8{ "gate", "up", "down" }) |proj| for ([_][]const u8{ "trellis_pm", "suh", "svh" }) |leaf| {
        var nb: [96]u8 = undefined;
        const t = w.get(try std.fmt.bufPrint(&nb, "model.layers.1.mlp.switch_mlp.{s}_proj.g0.{s}", .{ proj, leaf })).?;
        total += mlx.mlx_array_size(t) * mlx.mlx_array_itemsize(t) / @as(u64, @intCast(mlx.getShape(t)[0]));
    };
    return total;
}

test "hy4 attention spans: on Sushi Cooler a chunk's spans share one launch of the gate, of W_o and of the widened kv_b, with each span's own launches' bytes, BF16 and STQ, windowed, gathered and short last spans (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer fwd.spans_share = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    // 300 rows: a sliding layer's spans 256 + 44, a sparse layer's past its selection threshold 9 x 32 + 12; then 64
    // (two whole gathered spans) and two decode rows.
    var ids: [366]u32 = undefined;
    for ([_]bool{ false, true }) |stq| {
        var logits: [2][]f32 = .{ &.{}, &.{} };
        defer for (logits) |l| a.free(l);
        var dispatches: [2]u64 = undefined;
        for (&logits, &dispatches, [_]bool{ false, true }) |*out, *count, share| {
            fwd.spans_share = share;
            var p = try Packs.open(es);
            defer p.close();
            for (&ids, 0..) |*d, i| d.* = @intCast(1 + (i * 7919) % (p.config.vocab - 1));
            const m = if (stq)
                try hy4_model.Model.initWith(a, p.config, &p.stq, es, .{ .dir = p.stq_dir, .quant = p.quant })
            else
                try hy4_model.Model.init(a, p.config, &p.src, es);
            defer m.deinit();
            var o: Ops = .{ .s = es };
            defer o.deinit();
            var r0: usize = 0;
            for ([_]usize{ 300, 64, 1, 1 }) |w| {
                const before = mlx.engine.stats().dispatches;
                const row = try hostRows(&o, try o.own(try m.forwardReadout(ids[r0 .. r0 + w], .all)));
                if (r0 == 0) count.* = mlx.engine.stats().dispatches - before;
                const grown = try a.realloc(out.*, out.len + row.len);
                @memcpy(grown[out.len..], row);
                out.* = grown;
                r0 += w;
            }
        }
        diag("spans (stq {}): a 300-row chunk's dispatches {d} each span alone, {d} shared\n", .{ stq, dispatches[0], dispatches[1] });
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(logits[0]), std.mem.sliceAsBytes(logits[1]));
        // Each sparse layer's ten gathered spans launch the gate and W_o twice each (the whole spans, the short one).
        try testing.expect(dispatches[0] >= dispatches[1] + 3 * 2 * (10 - 2));
    }
}

test "hy4 stq: a decode row's absorbed attention and f32 linears read their BF16 weights as stored, kv_b held BF16, and answer the bytes of the products over their f32 widening, dense and past the selection (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer fwd.absorb_kernels_on = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var ids: [60]u32 = undefined;
    var logits: [2][]f32 = .{ &.{}, &.{} };
    defer for (logits) |l| a.free(l);
    for (&logits, [_]bool{ false, true }) |*out, on| {
        fwd.absorb_kernels_on = on;
        // A load decodes and drops the pack's banks it reads by row: each load opens the pack anew.
        var p = try Packs.open(es);
        defer p.close();
        for (&ids, 0..) |*d, i| d.* = @intCast(1 + (i * 7919) % (p.config.vocab - 1));
        const m = try hy4_model.Model.initWith(a, p.config, &p.stq, es, .{ .dir = p.stq_dir, .quant = p.quant });
        defer m.deinit();
        for (0..p.config.layers) |l| try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(m.layers[l].attn.kv_b));
        var o: Ops = .{ .s = es };
        defer o.deinit();
        // A prompt wider than the kernels' verify widths (its products the same on both sides), then decode rows on
        // both sides of the sparse layers' selection threshold.
        _ = try hostRows(&o, try o.own(try m.forwardReadout(ids[0..12], .last)));
        for (12..ids.len) |i| {
            const row = try hostRows(&o, try o.own(try m.forwardReadout(ids[i .. i + 1], .last)));
            const grown = try a.realloc(out.*, out.len + row.len);
            @memcpy(grown[out.len..], row);
            out.* = grown;
        }
    }
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(logits[0]), std.mem.sliceAsBytes(logits[1]));
}

test "hy4 stq: trunk linears whose launches round their rows and outputs through bf16, the norm after q_a, o and the dense down in their finish, answer the bytes of the casts around the bf16 launches and the f32 norm after, a prefill past the rows arm and decode rows (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer hy4_stq.thru_bf16_on = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var ids: [24]u32 = undefined;
    var logits: [2][]f32 = .{ &.{}, &.{} };
    defer for (logits) |l| a.free(l);
    for (&logits, [_]bool{ false, true }) |*out, on| {
        hy4_stq.thru_bf16_on = on;
        var p = try Packs.open(es);
        defer p.close();
        for (&ids, 0..) |*d, i| d.* = @intCast(1 + (i * 7919) % (p.config.vocab - 1));
        const m = try hy4_model.Model.initWith(a, p.config, &p.stq, es, .{ .dir = p.stq_dir, .quant = p.quant });
        defer m.deinit();
        var o: Ops = .{ .s = es };
        defer o.deinit();
        // A 12-row prompt (past the multi-row arm's 8), then decode rows.
        out.* = try a.dupe(f32, try hostRows(&o, try o.own(try m.forwardReadout(ids[0..12], .all))));
        for (12..ids.len) |i| {
            const row = try hostRows(&o, try o.own(try m.forwardReadout(ids[i .. i + 1], .last)));
            const grown = try a.realloc(out.*, out.len + row.len);
            @memcpy(grown[out.len..], row);
            out.* = grown;
        }
    }
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(logits[0]), std.mem.sliceAsBytes(logits[1]));
}

test "hy4 stq: a decode row's trunk linear through a trellis bank is its one through-bf16 launch, the bytes of the casts around the bf16 launch (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var p = try Packs.open(es);
    defer p.close();
    var owned: std.ArrayList(Arr) = .empty;
    defer {
        for (owned.items) |x| _ = mlx.mlx_array_free(x);
        owned.deinit(testing.allocator);
    }
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const bank = (try hy4_stq.load(&p.stq, "model.layers.1.self_attn.q_b_proj", p.quant.trunk.?, &owned, testing.allocator, es)).?;
    const x = try rows(&o, 1, @intCast(bank.inDim()), 3);
    try mlx.check(mlx.mlx_array_eval(x));
    const want = try o.cast(try hy4_stq.applyStored(&o, x, bank), .float32);
    try mlx.check(mlx.mlx_array_eval(want));
    const before = mlx.engine.stats().dispatches;
    const got = try fwd.linear(&o, x, .{ .w = undefined, .stq = bank });
    try mlx.check(mlx.mlx_array_eval(got));
    try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(try hostRows(&o, want)), std.mem.sliceAsBytes(try hostRows(&o, got)));
}

test "hy4 stq: the attention's write-back expanded inside the FFN's mix launch answers the bytes of the expand on its own, a prefill and decode rows (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer fwd.mix_expand_on = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var ids: [20]u32 = undefined;
    var logits: [2][]f32 = .{ &.{}, &.{} };
    defer for (logits) |l| a.free(l);
    for (&logits, [_]bool{ false, true }) |*out, on| {
        fwd.mix_expand_on = on;
        var p = try Packs.open(es);
        defer p.close();
        for (&ids, 0..) |*d, i| d.* = @intCast(1 + (i * 7919) % (p.config.vocab - 1));
        const m = try hy4_model.Model.initWith(a, p.config, &p.stq, es, .{ .dir = p.stq_dir, .quant = p.quant });
        defer m.deinit();
        var o: Ops = .{ .s = es };
        defer o.deinit();
        out.* = try a.dupe(f32, try hostRows(&o, try o.own(try m.forwardReadout(ids[0..8], .all))));
        for (8..ids.len) |i| {
            const row = try hostRows(&o, try o.own(try m.forwardReadout(ids[i .. i + 1], .last)));
            const grown = try a.realloc(out.*, out.len + row.len);
            @memcpy(grown[out.len..], row);
            out.* = grown;
        }
    }
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(logits[0]), std.mem.sliceAsBytes(logits[1]));
}

/// The served path runs the model's own kernels, a route split into groups summing in f32: within rounding of the
/// reference's.
const exact_bar: f32 = 1e-3;

fn servedStqSaid(slots: ?u64, bar: f32) !void {
    servedStq(slots, bar) catch |e| {
        if (e == error.SkipZigTest) return e;
        var said: [512]u8 = undefined;
        std.debug.print("\nserved stq: {t}: {s}\n", .{ e, mlx.takeError(&said) orelse "(no message)" });
        if (@errorReturnTrace()) |tr| std.debug.dumpStackTrace(tr.*);
        return e;
    };
}

const served_steps = [_]usize{ 40, 1, 1, 1, 1, 1, 1, 1, 1 };

fn servedStq(slots: ?u64, bar: f32) !void {
    const streamed = slots != null;
    const hy4_model = @import("hy4_model.zig");
    const Trunk = @import("../trunk.zig").Trunk;
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    var ids: [48]u32 = undefined;
    var vocab: usize = 0;
    var dir_buf: [512]u8 = undefined;
    var dir: []const u8 = undefined;
    var expert_bytes: u64 = 0;
    // The reference: the model over the pack's tensors (the part-1 tests hold it to the BF16 source), each step's last
    // row read to the host before the served load.
    const want = blk: {
        const es = mlx.engine.stream();
        const scope = mlx.engine.enter(es);
        defer scope.end();
        var p = try Packs.open(es);
        defer p.close();
        dir = dir_buf[0..p.stq_dir.len];
        @memcpy(dir_buf[0..p.stq_dir.len], p.stq_dir);
        vocab = p.config.vocab;
        // One routed expert's stored bytes: layer 1's group 0, its banks and scales, over its experts.
        for ([_][]const u8{ "gate", "up", "down" }) |proj| for ([_][]const u8{ "trellis_pm", "suh", "svh" }) |leaf| {
            var nb: [96]u8 = undefined;
            const t = p.stq.get(try std.fmt.bufPrint(&nb, "model.layers.1.mlp.switch_mlp.{s}_proj.g0.{s}", .{ proj, leaf })).?;
            expert_bytes += mlx.mlx_array_size(t) * mlx.mlx_array_itemsize(t) / @as(u64, @intCast(mlx.getShape(t)[0]));
        };
        var r = std.Random.DefaultPrng.init(0x5e7);
        for (&ids) |*d| d.* = r.random().uintLessThan(u32, @intCast(vocab - 1)) + 1;
        const out = try a.alloc(f32, served_steps.len * vocab);
        errdefer a.free(out);
        const m = try hy4_model.Model.initWith(a, p.config, &p.stq, es, .{ .dir = p.stq_dir, .quant = p.quant });
        defer m.deinit();
        m.kv = .dense;
        var o: Ops = .{ .s = es };
        defer o.deinit();
        var at: usize = 0;
        for (served_steps, 0..) |rows_n, i| {
            // The last row's readout alone, as the served forward reads a step: the head's row arm, not its GEMM.
            const last = try hostRows(&o, try o.own(try m.forwardReadout(ids[at .. at + rows_n], .last)));
            @memcpy(out[i * vocab ..][0..vocab], last);
            at += rows_n;
        }
        break :blk out;
    };
    defer a.free(want);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = streamed;
    config.expert_source_dir = try a.dupe(u8, dir);
    if (slots) |n| config.expert_cache_bytes = n * config.num_hidden_layers * expert_bytes;
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    try testing.expectEqual(!streamed, weights.get("model.layers.1.mlp.switch_mlp.gate_proj.g0.trellis_pm") != null);
    const billed = (try @import("hy4_load.zig").residentLoadBytes(io, a, dir, &config, false, false)).?;
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    try testing.expectEqual(streamed, xfm.expert_stream != null);
    // The bill's resident bytes are the bytes the load holds: the pack's tensors (the experts' as views of their
    // ragged storage, once) and the banks decoded at load in place of their stored bytes.
    {
        const st = &xfm.arch_state.hy4flash;
        var held: u64 = st.model.?.stq_decoded_bytes;
        var it = st.weights.?.map.iterator();
        while (it.next()) |e| held += mlx.mlx_array_size(e.value_ptr.*) * mlx.mlx_array_itemsize(e.value_ptr.*);
        try testing.expectEqual(billed, held);
    }
    var o: Ops = .{ .s = xfm.s };
    defer o.deinit();
    // The served request reads its latent width from the cache's config: a test load's is BF16, as the reference's.
    try testing.expectEqual(@as(@TypeOf(xfm.cache.config.scheme), .off), xfm.cache.config.scheme);
    var at: usize = 0;
    for (served_steps, 0..) |rows_n, i| {
        const arr = try o.own(mlx.mlx_array_new_data(&ids[at], &[_]c_int{ 1, @intCast(rows_n) }, 2, .uint32));
        const got = try hostRows(&o, try o.own(try xfm.forward(arr)));
        at += rows_n;
        const ref = want[i * vocab ..][0..vocab];
        var peak: f32 = 0;
        var worst: f32 = 0;
        for (got[got.len - vocab ..], ref) |x, y| {
            peak = @max(peak, @abs(y));
            worst = @max(worst, @abs(x - y));
        }
        if (worst > bar * peak) {
            std.debug.print("served stq at {d}: worst |Δ| {e:.3} of the peak {e:.3}\n", .{ at - 1, worst, peak });
            return error.OutsideBar;
        }
    }
}

test "hy4 stq: the K6 embedding gathered on the GPU gives EmbedGather.rows' bytes, at block edges, across blocks, in any order and over more blocks than one launch takes (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    try mlx.requireGpu();
    if (!@import("sushi_exl3").kernels.gemmNaxOn()) return error.SkipZigTest;
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var p = try Packs.open(es);
    defer p.close();
    var owned: std.ArrayList(Arr) = .empty;
    defer {
        for (owned.items) |x| _ = mlx.mlx_array_free(x);
        owned.deinit(a);
    }
    const bank = (try hy4_stq.load(&p.stq, "model.embed_tokens", p.quant.trunk, &owned, a, es)).?;
    var host = try hy4_stq.EmbedGather.of(a, bank);
    defer host.deinit(a);
    const gpu: @import("hy4_embed.zig").Gather = .{ .bank = bank };
    const v: u32 = p.config.vocab;
    const d: usize = p.config.dim;
    const edges = [_]u32{ 128, 0, 127, v - 1, 5000, 129, 0, v - 128, 77777 };
    // One id from each of 70 blocks, more than one launch's group of blocks, in a scrambled order.
    var spread: [70]u32 = undefined;
    for (&spread, 0..) |*id, i| id.* = @intCast(((i * 37) % 70) * (v / 70) + i % 128);
    // No ids: no rows, as the host gather gives.
    const empty = try gpu.rows(a, es, &.{});
    defer _ = mlx.mlx_array_free(empty);
    try testing.expectEqualSlices(c_int, &.{ 0, @intCast(d) }, mlx.getShape(empty));
    for ([_][]const u32{ &edges, &spread }) |ids| {
        const want = try a.alloc(f32, ids.len * d);
        defer a.free(want);
        try host.rows(a, ids, want);
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const got = try hostRows(&o, try o.own(try gpu.rows(a, es, ids)));
        try testing.expectEqualSlices(u32, @ptrCast(want), @ptrCast(got));
    }
}

test "hy4 stq: after warmKernels a GPU embedding gather of any block count, in any order, makes no pipeline (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    try mlx.requireGpu();
    if (!@import("sushi_exl3").kernels.gemmNaxOn()) return error.SkipZigTest;
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var p = try Packs.open(es);
    defer p.close();
    const m = try @import("hy4_model.zig").Model.initWith(a, p.config, &p.stq, es, .{ .dir = p.stq_dir, .quant = p.quant });
    defer m.deinit();
    const g = m.embed_gpu orelse return error.TestUnexpectedResult;
    try m.warmKernels();
    // One id a block, so a gather of n ids decodes n blocks: one launch group and past it.
    var ids: [72]u32 = undefined;
    for (&ids, 0..) |*id, b| id.* = @intCast(b * 128 + b % 128);
    const before = mlx.engine.pipelinesMade().count;
    for ([_]usize{ 1, 2, 3, 5, 9, 17, 31, 32, 33, 40 }) |n| {
        const e = try g.rows(a, es, ids[0..n]);
        defer _ = mlx.mlx_array_free(e);
        try mlx.check(mlx.mlx_array_eval(e));
    }
    // A prompt's ids are in no block order: past one launch's group the parts are joined and taken back into the ids'.
    for ([_]usize{ 33, 40, 65, 72 }) |n| {
        var shuffled: [72]u32 = undefined;
        for (shuffled[0..n], 0..) |*id, i| id.* = ids[(i * 7) % n];
        const e = try g.rows(a, es, shuffled[0..n]);
        defer _ = mlx.mlx_array_free(e);
        try mlx.check(mlx.mlx_array_eval(e));
    }
    const made = mlx.engine.pipelinesMade().count - before;
    errdefer std.debug.print("gathers after the warm made {d} pipelines\n", .{made});
    try testing.expectEqual(@as(u64, 0), made);
}

test "hy4 stq: the K6 embedding gathered by 128-id block gives each token the row of the whole bank's decode, at block edges and across blocks, in any order (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var p = try Packs.open(es);
    defer p.close();
    var owned: std.ArrayList(Arr) = .empty;
    defer {
        for (owned.items) |x| _ = mlx.mlx_array_free(x);
        owned.deinit(a);
    }
    const bank = (try hy4_stq.load(&p.stq, "model.embed_tokens", p.quant.trunk, &owned, a, es)).?;
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const whole = try o.own(try hy4_stq.decodeDense(a, es, bank)); // [vocab, dim] BF16
    const v: u32 = p.config.vocab;
    const ids = [_]u32{ 128, 0, 127, v - 1, 5000, 129, 0, v - 128, 77777 };
    const d: usize = p.config.dim;
    const got = try a.alloc(f32, ids.len * d);
    defer a.free(got);
    var g = try hy4_stq.EmbedGather.of(a, bank);
    defer g.deinit(a);
    try g.rows(a, &ids, got);
    const idx = try o.own(mlx.mlx_array_new_data(&ids, &[_]c_int{ids.len}, 1, .uint32));
    const want = try hostRows(&o, try o.take(whole, idx, 0));
    // The whole decode is the kernel's product (its f16 Hadamard planes) rounded to BF16; the gather is the exact f32
    // decode. Held per row, to the row's peak: a few f16 and one BF16 rounding apart.
    var worst: f32 = 0;
    var num: f64 = 0;
    var den: f64 = 0;
    for (0..ids.len) |r| {
        var peak: f32 = 0;
        var dmax: f32 = 0;
        for (got[r * d ..][0..d], want[r * d ..][0..d]) |x, y| {
            peak = @max(peak, @abs(y));
            dmax = @max(dmax, @abs(x - y));
            num += (@as(f64, x) - y) * (@as(f64, x) - y);
            den += @as(f64, y) * y;
        }
        worst = @max(worst, dmax / peak);
    }
    const rel = @sqrt(num / den);
    diag("[hy4-stq] embedding gather against the whole decode: worst |Δ| {e:.3} of its row's peak, relative RMS {e:.3}\n", .{ worst, rel });
    try testing.expect(worst <= 1.0 / 64.0 and rel <= 1.0 / 256.0);
}

test "hy4 stq: a dry recording gathers the host embedding's rows as zeros, decoding no block, and still refuses an id past the vocabulary (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var p = try Packs.open(es);
    defer p.close();
    var owned: std.ArrayList(Arr) = .empty;
    defer {
        for (owned.items) |x| _ = mlx.mlx_array_free(x);
        owned.deinit(a);
    }
    const bank = (try hy4_stq.load(&p.stq, "model.embed_tokens", p.quant.trunk, &owned, a, es)).?;
    var g = try hy4_stq.EmbedGather.of(a, bank);
    defer g.deinit(a);
    const v: u32 = p.config.vocab;
    const d: usize = p.config.dim;
    const ids = [_]u32{ 0, 128, 5000, v - 1 };
    const out = try a.alloc(f32, ids.len * d);
    defer a.free(out);
    @memset(out, 1);
    const plan = @import("../tensor/plan/plan_api.zig");
    var driver: plan.Driver = .{ .gpa = a };
    defer driver.deinit();
    plan.session = .{ .mode = .dry, .driver = &driver };
    defer plan.session = .{};
    const before = hy4_stq.EmbedGather.blocks_decoded;
    try g.rows(a, &ids, out);
    try testing.expectEqual(before, hy4_stq.EmbedGather.blocks_decoded);
    for (out) |x| try testing.expectEqual(@as(f32, 0), x);
    try testing.expectError(error.TokenOutOfRange, g.rows(a, &[_]u32{v}, out[0..d]));
    plan.session = .{};
    try g.rows(a, &ids, out);
    try testing.expectEqual(before + 4, hy4_stq.EmbedGather.blocks_decoded);
}

test "hy4 stq: the host embedding gather keeps the blocks it decoded to its byte cap: a block asked again is read back, not decoded, past the cap the least recently used goes, and every row keeps its bytes (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var p = try Packs.open(es);
    defer p.close();
    var owned: std.ArrayList(Arr) = .empty;
    defer {
        for (owned.items) |x| _ = mlx.mlx_array_free(x);
        owned.deinit(a);
    }
    const bank = (try hy4_stq.load(&p.stq, "model.embed_tokens", p.quant.trunk, &owned, a, es)).?;
    const d: usize = p.config.dim;
    const block_bytes: u64 = d * 128 * @sizeOf(f32);
    var plain = try hy4_stq.EmbedGather.ofWith(a, bank, 0);
    defer plain.deinit(a);
    var g = try hy4_stq.EmbedGather.ofWith(a, bank, 2 * block_bytes);
    defer g.deinit(a);
    // Held at its cap: two blocks' rows, with their ids and last uses.
    const kept = hy4_stq.EmbedGather.cacheBytes(d, 2 * block_bytes);
    try testing.expect(kept >= 2 * block_bytes and kept < 2 * block_bytes + 1024);
    try testing.expectEqual(plain.bytes() + kept, g.bytes());
    // Blocks 0, 1 and 2; two fit.
    const steps = [_]struct { ids: []const u32, decodes: usize }{
        .{ .ids = &.{ 0, 1 }, .decodes = 1 },
        .{ .ids = &.{ 1, 128, 0 }, .decodes = 1 },
        .{ .ids = &.{5}, .decodes = 0 },
        .{ .ids = &.{300}, .decodes = 1 }, // block 1 was used least recently: it goes
        .{ .ids = &.{ 2, 301 }, .decodes = 0 },
        .{ .ids = &.{129}, .decodes = 1 },
    };
    for (steps) |st| {
        const got = try a.alloc(f32, st.ids.len * d);
        defer a.free(got);
        const want = try a.alloc(f32, st.ids.len * d);
        defer a.free(want);
        try plain.rows(a, st.ids, want);
        const before = hy4_stq.EmbedGather.blocks_decoded;
        try g.rows(a, st.ids, got);
        try testing.expectEqual(st.decodes, hy4_stq.EmbedGather.blocks_decoded - before);
        try testing.expectEqualSlices(u32, @ptrCast(want), @ptrCast(got));
    }
}

test "hy4 stq served: a server's streamed load reads the STQ pack's rate-group banks as the stream's layout, not a refusal (HY4_STQ_TINY)" {
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const io = std.Io.Threaded.global_single_threaded.io();
    const layout = try @import("../expert_store.zig").streamingLayoutOfDir(testing.allocator, io, "hy_v4_flash", dir, 6, 1);
    try testing.expectEqual(@import("../expert_store.zig").Layout.exl3_k4, layout);
}

test "hy4 stq served: the pack's expert banks are checked at the experts' own width, the latent, not the model's (HY4_STQ_TINY)" {
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    // The release's experts run at 1536 under a 3072-wide model; the tiny's latent is its width, so the model is
    // widened here to tell the two apart.
    try testing.expectEqual(@as(u32, 128), config.expertGeometry().hidden);
    config.hidden_size = 256;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const pack_index = @import("../pack_index.zig");
    const source = try pack_index.loadSourceIndex(io, arena.allocator(), dir);
    _ = try pack_index.validateExl3Layer(&source, arena.allocator(), &config, "model.layers.1");
}

test "hy4 stq served: a streamed load's budget and fill peak price no union workspace at the release's geometry, as the engine allocates none (Hy4 routes within its cache)" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const model_load = @import("../model_load.zig");
    const es = @import("../expert_stream.zig");
    var config = try model_mod.parseConfigText(io, a, "", @import("hy4_config.zig").release_config);
    defer config.deinit(a);
    // A K2.25 PAIR1 pack's store as it reports itself: one union set, its experts 2.0 MB; the K6 trunk 6.28 GB.
    config.expert_shape = .{ .union_sets = 1 };
    const per: u64 = 2_009_000;
    const r = try model_load.resolveExpertCache(18 << 30, &config, .{ .trunk = 6_280_000_000, .mtp = 0 }, false, per);
    try testing.expectEqual(@as(u64, 0), r.ledger.?.workspace_bytes);
    const cache = try es.cachePlanBytesForGeometry(r.cache_bytes, config.expertGeometry(), per);
    try testing.expectEqual(@as(u64, 0), cache.workspace_bytes);
    const slots = cache.slots_per_layer;
    try testing.expectEqual(es.fillPeakBytes(per, slots, slots, cache.bounce_bytes), cache.prefill_peak_bytes);
}

test "hy4 stq served: a request that omits max_tokens gets the release's served output cap, 65,536" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigText(io, a, "", hy4_config.release_config);
    defer config.deinit(a);
    try testing.expectEqual(@as(u32, 65_536), config.defaultMaxTokens());
    config.max_tokens_override = 8_000;
    try testing.expectEqual(@as(u32, 8_000), config.defaultMaxTokens());
}

test "hy4 stq served: with no flag and no setting the release takes the RAM-tier default of its Mac" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const model_load = @import("../model_load.zig");
    var config = try model_mod.parseConfigText(io, a, "", hy4_config.release_config);
    defer config.deinit(a);
    try testing.expect(config.serve_defaults.ram_budget);
    try testing.expectEqual(model_load.ramTier(@import("../status.zig").getTotalMemBytes()), model_load.ramDefaultTierFor(&config));
}

test "hy4 stq served: the RAM-tier default holds every routed expert resident, its vision tower beside them, on a 192 or 256 GB Mac and on a 128 GB one, and 390 of 512 slots on a 96 GB one" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const model_load = @import("../model_load.zig");
    var config = try model_mod.parseConfigText(io, a, "", hy4_config.release_config);
    defer config.deinit(a);
    // The 4bpw pack: a 6.28 GB trunk and 512 slots a layer of 3.557 MB experts (74.68 GB), none of Engram's tables
    // (read by row from SSD), and the tower at affine 8 (0.63 GB), which a no-flag load keeps.
    const per: u64 = 3_557_000;
    const vision = model_load.coldLoadVision(&config, true);
    try testing.expect(vision);
    const all = @as(u64, config.expertLayerCount()) * config.keptExperts() * per;
    config.expert_shape = .{ .union_sets = 1, .all_bytes = all };
    const split: model_mod.ResidentSplit = .{ .trunk = 6_280_000_000, .mtp = 0, .vision = 630_000_000 };
    const r = try model_load.resolveExpertCache(1 << 50, &config, split, vision, per);
    const need = r.ledger.?.trunk_bytes + r.ledger.?.vision_bytes + r.ledger.?.cache_bytes;
    try testing.expect(r.ledger.?.resident);
    try testing.expectEqual(@as(u16, 512), r.ledger.?.slots_per_layer);
    const GiB: u64 = 1 << 30;
    const Flat = struct {
        pub fn needAt(_: @This(), budget: u64) ?u64 {
            return budget;
        }
    };
    // Each tier's working set is 75% of its RAM, less an 8 GiB margin and the 4 GiB request bill beside the budget.
    for ([_]struct { tier: model_load.RamTier, ram: u64, slots: u16 }{
        .{ .tier = .gb256, .ram = 256, .slots = 512 },
        .{ .tier = .gb192, .ram = 192, .slots = 512 },
        .{ .tier = .gb128, .ram = 128, .slots = 512 },
        .{ .tier = .gb96, .ram = 96, .slots = 390 },
    }) |c| {
        const limits = model_load.BudgetLimits{ .ceiling = c.ram * GiB * 3 / 4, .ceiling_name = "the GPU working-set limit", .margin = 8 * GiB, .user_wire = 0, .user_wire_name = "", .user_wire_margin = GiB };
        const d = model_load.ramDefaultBudget(c.tier, need, 4 * GiB, limits, Flat{});
        const planned = try model_load.resolveExpertCache(d.bytes, &config, split, vision, per);
        errdefer std.debug.print("{s}: budget {d} GiB, slots {d}, cache {d} B\n", .{ @tagName(c.tier), d.bytes >> 30, planned.ledger.?.slots_per_layer, planned.cache_bytes });
        try testing.expectEqual(c.slots, planned.ledger.?.slots_per_layer);
    }
}

test "hy4 stq: a served model's deinit leaves the host arrays the EXL3 kernels cached to its handle, which drops them with the last model of the side (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var p = try Packs.open(es);
    defer p.close();
    const exl3 = @import("sushi_exl3");
    // Another model's cache: arrays the EXL3 module made for its kernels, which a Hy4 model's unload must not free. A
    // later ask for them answers with the same arrays while the cache holds them.
    const other = try exl3.linear.gemmInputs(3, 70, 32);
    const served = try hy4_model.Model.initWith(a, p.config, &p.stq, es, .{ .dir = p.stq_dir, .quant = p.quant, .served = true });
    served.deinit();
    const kept = try exl3.linear.gemmInputs(3, 70, 32);
    try testing.expectEqual(other.slots.ctx, kept.slots.ctx);
}

test "hy4 stq: a verify block under the mark of a DFly2 draft at and under the selection threshold holds no more than its new rows and the mark's small state (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    const hy4_model = @import("hy4_model.zig");
    const hy4_ledger = @import("hy4_ledger.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var p = try Packs.open(es);
    defer p.close();
    const m = try hy4_model.Model.initWith(a, p.config, &p.stq, es, .{ .dir = p.stq_dir, .quant = p.quant });
    defer m.deinit();
    try loadInto(&p.stq, p.stq_dir, "model-dfly", es);
    var d = try @import("hy4_dfly.zig").Drafter.init(a, m, &p.stq);
    defer d.deinit();
    var rng = std.Random.DefaultPrng.init(0x5e9);
    const inUse = struct {
        fn now() u64 {
            mlx.engine.close();
            mlx.engine.trim();
            return mlx.engine.poolTotal() - mlx.engine.poolFree();
        }
    }.now;
    // At and under the selection threshold (16 keys on the tiny) a draft layer reads every row of its target.
    for ([_]usize{ 8, 16 }) |len| {
        m.reset();
        const ids = try a.alloc(u32, len + 6);
        defer a.free(ids);
        for (ids) |*x| x.* = rng.random().uintLessThan(u32, @intCast(p.config.vocab - 1)) + 1;
        var at: usize = 0;
        while (at < len) : (at += 512) _ = mlx.mlx_array_free(try m.forwardReadout(ids[at..@min(len, at + 512)], .none));
        const in0 = inUse();
        var block = try d.draft(m, 7);
        block.deinit();
        // The verify block's forward under the round's mark: the mark keeps the small state the block replaces (the rings it
        // copies, the open groups, the hidden state), the rows grow in place.
        var mk = try m.mark();
        defer m.dropMark(&mk);
        _ = mlx.mlx_array_free(try m.forwardReadout(ids[len .. len + 6], .none));
        const grown = hy4_ledger.stateBytes(&p.config, 8, len + 6) - hy4_ledger.stateBytes(&p.config, 8, len);
        try testing.expect(inUse() - in0 <= grown + hy4_ledger.pointBytes(&p.config, 8) + hy4_ledger.hiddenBytes(&p.config) + hy4_ledger.pickBytes(&p.config));
    }
}

/// One site's W_o and the iHC mix after it, on random inputs: `rows` gated rows of `in` through a K6 bank to `d`, four
/// streams of `d`, the mix's weights and both norms.
const OSite = struct {
    bank: hy4_stq.Bank,
    gated: Arr,
    h: Arr,
    post: Arr,
    mix: fwd.Mix,
    norm_w: Arr,
    attn: fwd.Attn,

    fn init(o: *Ops, alloc: std.mem.Allocator, rnd: std.Random, in: usize, d: usize, t: c_int) !OSite {
        const exl3 = @import("sushi_exl3");
        const tb = try exl3.trunk.TestBank.init(alloc, rnd, in, d, 96);
        const tr = try tb.triple(o.s);
        const bank: hy4_stq.Bank = .{ .w = try o.own(tr.w), .suh = try o.own(tr.s), .svh = try o.own(tr.b), .dec = exl3.format.Decode.tabled(.spec1, .w15) };
        const tu: usize = @intCast(t);
        const fill = struct {
            fn f(a: std.mem.Allocator, r: std.Random, n: usize, lo: f32, span: f32) ![]f32 {
                const v = try a.alloc(f32, n);
                for (v) |*e| e.* = lo + r.float(f32) * span;
                return v;
            }
        };
        const gv = try fill.f(alloc, rnd, tu * in, -2, 4);
        const hv = try fill.f(alloc, rnd, tu * 4 * d, -1, 2);
        const pv = try fill.f(alloc, rnd, tu * 4, -1, 3);
        const fv = try fill.f(alloc, rnd, 8 * 4 * d, -0.05, 0.1);
        const pnv = try fill.f(alloc, rnd, d, 0.5, 1);
        const nwv = try fill.f(alloc, rnd, d, 0.5, 1);
        const sv = [_]f32{ 0.9, 1.1, 0.7 };
        const basev = [_]f32{ 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8 };
        var c: OSite = undefined;
        c.bank = bank;
        c.gated = try o.own(mlx.mlx_array_new_data(gv.ptr, &[_]c_int{ t, @intCast(in) }, 2, .float32));
        c.h = try o.own(mlx.mlx_array_new_data(hv.ptr, &[_]c_int{ t, 4, @intCast(d) }, 3, .float32));
        c.post = try o.own(mlx.mlx_array_new_data(pv.ptr, &[_]c_int{ t, 4 }, 2, .float32));
        c.mix = .{ .fn_ = try o.own(mlx.mlx_array_new_data(fv.ptr, &[_]c_int{ 8, @intCast(4 * d) }, 2, .float32)), .scale = try o.own(mlx.mlx_array_new_data(&sv, &[_]c_int{3}, 1, .float32)), .base = try o.own(mlx.mlx_array_new_data(&basev, &[_]c_int{8}, 1, .float32)) };
        c.norm_w = try o.cast(try o.own(mlx.mlx_array_new_data(nwv.ptr, &[_]c_int{@intCast(d)}, 1, .float32)), .bfloat16);
        c.attn.o = .{ .w = undefined, .stq = bank };
        c.attn.post_norm = try o.cast(try o.own(mlx.mlx_array_new_data(pnv.ptr, &[_]c_int{@intCast(d)}, 1, .float32)), .bfloat16);
        for ([_]Arr{ c.gated, c.h, c.post, c.norm_w, c.attn.post_norm }) |a| try mlx.check(mlx.mlx_array_eval(a));
        return c;
    }

    /// W_o's partials and the mix that finishes them: today's pair of launches.
    fn parts(c: OSite, o: *Ops) !fwd.Expanded {
        const p = (try hy4_stq.applyParts(o, c.gated, c.bank)).?;
        return fwd.expandMixNormParts(o, c.h, c.post, p.part, p.nsplit, c.attn, 1e-5, c.mix, 2.0, 1e-6, 1e-6, c.norm_w);
    }
};

fn expectSameExpanded(o: *Ops, want: fwd.Expanded, got: fwd.Expanded) !void {
    for ([_][2]Arr{ .{ want.m, got.m }, .{ want.mixed.post, got.mixed.post }, .{ want.mixed.x, got.mixed.x } }) |pair|
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(try hostRows(o, pair[0])), std.mem.sliceAsBytes(try hostRows(o, pair[1])));
}

fn evalExpanded(e: fwd.Expanded) !void {
    for ([_]Arr{ e.m, e.mixed.post, e.mixed.x }) |a| try mlx.check(mlx.mlx_array_eval(a));
}

/// The release's sparse and sliding W_o, then a small block.
const o_shapes = [_]struct { in: usize, d: usize }{ .{ .in = 8192, .d = 3072 }, .{ .in = 16384, .d = 3072 }, .{ .in = 512, .d = 384 } };

test "hy4 stq: a decode block's iHC mix finishes and norms W_o's partials in its launch, one launch fewer with the bytes of W_o's finish and post norm and the mix after it, at the release's 8,192 and 16,384 to 3,072 and a small block, one row and three, 1,024 lanes and 512, on the machine's arm and SIMD" {
    const exl3 = @import("sushi_exl3");
    const hck = @import("../hc_kernels.zig");
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const tg_was = hck.hc_tg;
    defer hck.hc_tg = tg_was;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x0f17);
    var served: usize = 0;
    for ([_]bool{ false, true }) |simd| {
        const prof: ?exl3.device.Scope = if (simd) exl3.device.withProfile(exl3.device.simdOnly()) else null;
        defer if (prof) |p| p.end();
        for (o_shapes, 0..) |sh, si| for ([_]c_int{ 1, 3 }) |t| for ([_]usize{ 1024, 512 }) |tg| {
            hck.hc_tg = tg;
            var o: Ops = .{ .s = es };
            defer o.deinit();
            const c = try OSite.init(&o, arena.allocator(), prng.random(), sh.in, sh.d, t);
            try testing.expect(fwd.oPartsReady(&o, c.h, c.mix, c.attn));
            // W_o's launch and its finish_rms, as a launch that does not finish its own partials makes them.
            exl3.trunk.self_finish_on = false;
            defer exl3.trunk.self_finish_on = true;
            const c0 = mlx.engine.stats().dispatches;
            const y = try fwd.linearNormed(&o, c.gated, c.attn.o, c.attn.post_norm, 1e-5);
            const want = try fwd.expandMixNorm(&o, c.h, c.post, y, c.mix, 4, 2.0, 1e-6, 1e-6, c.norm_w);
            try evalExpanded(want);
            const chain = mlx.engine.stats().dispatches - c0;
            exl3.trunk.self_finish_on = true;
            const g0 = mlx.engine.stats().dispatches;
            const got = try c.parts(&o);
            try evalExpanded(got);
            const launches = mlx.engine.stats().dispatches - g0;
            try expectSameExpanded(&o, want, got);
            // W_o's launch, then the mix's: its finish_rms launch is none.
            try testing.expectEqual(chain - 1, launches);
            if (si < 2) try testing.expectEqual(@as(u64, 2), launches);
            served += 1;
        };
    }
    try testing.expectEqual(@as(usize, 24), served);
}

test "hy4 stq: on Sushi Cooler W_o's launch finishes its own partials (each output block's last threadgroup to count in) and the iHC mix norms its rows, the bytes and the two launches of W_o's partials and the mix that finishes them, at the release's 8,192 and 16,384 to 3,072 and a small block, one, three and eight rows, 1,024 lanes and 512, on the machine's arm and SIMD" {
    const exl3 = @import("sushi_exl3");
    const hck = @import("../hc_kernels.zig");
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const tg_was = hck.hc_tg;
    defer hck.hc_tg = tg_was;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x0f18);
    var served: usize = 0;
    for ([_]bool{ false, true }) |simd| {
        const prof: ?exl3.device.Scope = if (simd) exl3.device.withProfile(exl3.device.simdOnly()) else null;
        defer if (prof) |p| p.end();
        for (o_shapes) |sh| for ([_]c_int{ 1, 3, 8 }) |t| for ([_]usize{ 1024, 512 }) |tg| {
            hck.hc_tg = tg;
            var o: Ops = .{ .s = es };
            defer o.deinit();
            const c = try OSite.init(&o, arena.allocator(), prng.random(), sh.in, sh.d, t);
            const c0 = mlx.engine.stats().dispatches;
            const want = try c.parts(&o);
            try evalExpanded(want);
            const pair = mlx.engine.stats().dispatches - c0;
            const g0 = mlx.engine.stats().dispatches;
            const y = (try hy4_stq.applyFinished(&o, c.gated, c.bank)).?;
            const got = try fwd.expandMixNormRows(&o, c.h, c.post, y, c.attn, 1e-5, c.mix, 2.0, 1e-6, 1e-6, c.norm_w);
            try evalExpanded(got);
            try testing.expectEqual(pair, mlx.engine.stats().dispatches - g0);
            try testing.expectEqual(@as(u64, 2), pair);
            try expectSameExpanded(&o, want, got);
            // W_o's rows are its through launch's (`apply`): the finish it took in is the finish's bytes.
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(try hostRows(&o, try hy4_stq.apply(&o, c.gated, c.bank))), std.mem.sliceAsBytes(try hostRows(&o, y)));
            served += 1;
        };
    }
    try testing.expectEqual(@as(usize, 36), served);
    // MLX's stream has no counter page: there W_o's launch leaves its partials for the mix, as before.
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const c = try OSite.init(&o, arena.allocator(), prng.random(), 512, 384, 1);
    var mo: Ops = .{ .s = mlx.gpuStream() };
    defer mo.deinit();
    try testing.expect(try hy4_stq.applyFinished(&mo, c.gated, c.bank) == null);
}

/// A trunk site at the release's geometry: `in` to `out`, finished as `kind` (`project_rows.Counted`'s tags).
const TrunkSite = struct { in: u32, out: u32, kind: enum { thru, stored, normed } };

/// Every trunk site a decode row reaches that leaves partials at the release's geometry: q_a and its norm, q_b, the gate
/// (stored bf16), W_o through its post norm, the shared expert's and the latents' linears, the dense layer's FFN and its
/// post norm. The head's 944 output blocks are past the counter page's counters and keep their finish.
fn releaseTrunkSites() ![11]TrunkSite {
    const c = try hy4_config.parseText(hy4_config.release_config);
    const qk = c.qkDim();
    return .{
        .{ .in = c.dim, .out = c.q_lora, .kind = .normed },
        .{ .in = c.q_lora, .out = c.sparse_heads * qk, .kind = .thru },
        .{ .in = c.q_lora, .out = c.sliding_heads * qk, .kind = .thru },
        .{ .in = c.dim, .out = c.sparse_heads * c.v_dim, .kind = .stored },
        .{ .in = c.dim, .out = c.sliding_heads * c.v_dim, .kind = .stored },
        .{ .in = c.sparse_heads * c.v_dim, .out = c.dim, .kind = .normed },
        .{ .in = c.sliding_heads * c.v_dim, .out = c.dim, .kind = .normed },
        .{ .in = c.dim, .out = c.moe_inter, .kind = .thru },
        .{ .in = c.moe_inter, .out = c.dim, .kind = .thru },
        .{ .in = c.dim, .out = c.dense_inter, .kind = .thru },
        .{ .in = c.dense_inter, .out = c.dim, .kind = .normed },
    };
}

/// `n` values in `[lo, lo + span)`, as `dt`.
fn randomArr(o: *Ops, alloc: std.mem.Allocator, rnd: std.Random, shape: []const c_int, lo: f32, span: f32, dt: mlx.mlx_dtype) !Arr {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try alloc.alloc(f32, n);
    for (v) |*e| e.* = lo + rnd.float(f32) * span;
    const a = try o.cast(try o.own(mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32)), dt);
    try mlx.check(mlx.mlx_array_eval(a));
    return a;
}

/// A trunk site's linear through the trunk's own entry (`hy4_stq`'s): the rows, and the dispatches it made.
fn siteRows(o: *Ops, x: Arr, b: hy4_stq.Bank, kind: @FieldType(TrunkSite, "kind"), norm_w: Arr, eps: f32) !struct { y: Arr, launches: u64 } {
    const c0 = mlx.engine.stats().dispatches;
    const y = switch (kind) {
        .thru => try hy4_stq.apply(o, x, b),
        .stored => try hy4_stq.applyStored(o, x, b),
        .normed => (try hy4_stq.applyNormed(o, x, b, norm_w, eps)).?,
    };
    try mlx.check(mlx.mlx_array_eval(y));
    return .{ .y = y, .launches = mlx.engine.stats().dispatches - c0 };
}

test "hy4 stq: on Sushi Cooler every trunk site a decode row reaches finishes its own partials in its launch at one to three rows (q_a and W_o through their norm at one, the gate stored bf16 from its f32 rows), the bytes of today's launch, finish and cast, at the release's shapes, one, three and eight rows, on the machine's arm and SIMD" {
    const exl3 = @import("sushi_exl3");
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer exl3.trunk.self_finish_on = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x5e1f);
    const rnd = prng.random();
    const dec = exl3.format.Decode.tabled(.spec1, .w15);
    const eps = (try hy4_config.parseText(hy4_config.release_config)).eps;
    var served: usize = 0;
    for ([_]bool{ false, true }) |simd| {
        const prof: ?exl3.device.Scope = if (simd) exl3.device.withProfile(exl3.device.simdOnly()) else null;
        defer if (prof) |p| p.end();
        for (try releaseTrunkSites()) |site| {
            var o: Ops = .{ .s = es };
            defer o.deinit();
            const tb = try exl3.trunk.TestBank.init(alloc, rnd, site.in, site.out, 96);
            const tr = try tb.triple(es);
            const bank: hy4_stq.Bank = .{ .w = try o.own(tr.w), .suh = try o.own(tr.s), .svh = try o.own(tr.b), .dec = dec };
            const norm_w = try randomArr(&o, alloc, rnd, &.{@intCast(site.out)}, 0.5, 1, .bfloat16);
            for ([_]c_int{ 1, 3, 8 }) |t| {
                const x = try randomArr(&o, alloc, rnd, &.{ t, @intCast(site.in) }, -2, 4, .float32);
                exl3.trunk.self_finish_on = false;
                const want = try siteRows(&o, x, bank, site.kind, norm_w, eps);
                exl3.trunk.self_finish_on = true;
                const got = try siteRows(&o, x, bank, site.kind, norm_w, eps);
                try testing.expectEqual(mlx.mlx_array_dtype(want.y), mlx.mlx_array_dtype(got.y));
                try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(try hostRows(&o, want.y)), std.mem.sliceAsBytes(try hostRows(&o, got.y)));
                const kind: exl3.project_rows.Counted = switch (site.kind) {
                    .thru => .thru,
                    .stored => .stored,
                    .normed => .{ .normed = .{ .w = norm_w, .eps = try o.own(try mlx.newData(&[_]f32{eps}, &.{1}, .float32)), .cols = @intCast(site.out) } },
                };
                // Today: the launch and its finish, the stored gate's cast before them. Up to three rows (a normed
                // launch one): one launch.
                try testing.expectEqual(@as(u64, if (site.kind == .stored) 3 else 2), want.launches);
                try testing.expectEqual(if (t <= exl3.project_rows.countedRowsMax(kind)) @as(u64, 1) else want.launches, got.launches);
                // The counted launch itself at three and eight rows too, where the trunk keeps the finish.
                const c0 = mlx.engine.stats().dispatches;
                const direct = (try exl3.linear.linearCounted(es, x, .{ .trellis = bank.w, .suh = bank.suh, .svh = bank.svh }, dec, .{ .rows = .{ .exact = true, .thru_bf16 = true }, .decoded_billed = true }, kind)).?;
                _ = try o.own(direct);
                try mlx.check(mlx.mlx_array_eval(direct));
                try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - c0);
                try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(try hostRows(&o, want.y)), std.mem.sliceAsBytes(try hostRows(&o, direct)));
                served += 1;
            }
        }
    }
    try testing.expectEqual(@as(usize, 66), served);
    // MLX's stream has no counter page: there every site keeps its launch and finish.
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const tb = try exl3.trunk.TestBank.init(alloc, rnd, 1024, 1024, 96);
    const tr = try tb.triple(es);
    const p: exl3.Proj = .{ .trellis = try o.own(tr.w), .suh = try o.own(tr.s), .svh = try o.own(tr.b) };
    const x = try randomArr(&o, alloc, rnd, &.{ 1, 1024 }, -2, 4, .float32);
    try testing.expect(try exl3.linear.linearCounted(mlx.gpuStream(), x, p, dec, .{ .rows = .{ .exact = true, .thru_bf16 = true } }, .thru) == null);
}

test "hy4 stq: on Sushi Cooler a SwiGLU's gate and up of up to three rows are one launch over their one input that finishes both banks and leaves the SwiGLU, the bytes of today's two launches, two finishes and hy4_chain.swiglu, at the release's shared expert and dense layer, one, three and eight rows, on the machine's arm and SIMD" {
    const exl3 = @import("sushi_exl3");
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer exl3.trunk.self_finish_on = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x5e20);
    const rnd = prng.random();
    const dec = exl3.format.Decode.tabled(.spec1, .w15);
    const c = try hy4_config.parseText(hy4_config.release_config);
    var served: usize = 0;
    for ([_]bool{ false, true }) |simd| {
        const prof: ?exl3.device.Scope = if (simd) exl3.device.withProfile(exl3.device.simdOnly()) else null;
        defer if (prof) |p| p.end();
        for ([_]u32{ c.moe_inter, c.dense_inter }) |inter| {
            var o: Ops = .{ .s = es };
            defer o.deinit();
            var banks: [2]hy4_stq.Bank = undefined;
            for (&banks) |*b| {
                const tb = try exl3.trunk.TestBank.init(alloc, rnd, c.dim, inter, 96);
                const tr = try tb.triple(es);
                b.* = .{ .w = try o.own(tr.w), .suh = try o.own(tr.s), .svh = try o.own(tr.b), .dec = dec };
            }
            const w: fwd.Swiglu = .{ .gate = .{ .w = undefined, .stq = banks[0] }, .up = .{ .w = undefined, .stq = banks[1] }, .down = undefined };
            for ([_]c_int{ 1, 3, 8 }) |t| {
                const x = try randomArr(&o, alloc, rnd, &.{ t, @intCast(c.dim) }, -2, 4, .float32);
                exl3.trunk.self_finish_on = false;
                var c0 = mlx.engine.stats().dispatches;
                const want = try fwd.swigluAct(&o, x, w, null);
                try mlx.check(mlx.mlx_array_eval(want));
                const today = mlx.engine.stats().dispatches - c0;
                exl3.trunk.self_finish_on = true;
                c0 = mlx.engine.stats().dispatches;
                const got = try fwd.swigluAct(&o, x, w, null);
                try mlx.check(mlx.mlx_array_eval(got));
                const launches = mlx.engine.stats().dispatches - c0;
                try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(try hostRows(&o, want)), std.mem.sliceAsBytes(try hostRows(&o, got)));
                // Today: gate and up, their finishes, the SwiGLU. Up to three rows: the pair alone.
                try testing.expectEqual(@as(u64, 5), today);
                try testing.expectEqual(if (t <= exl3.project_rows.COUNTED_ROWS_MAX) @as(u64, 1) else today, launches);
                // The pair's launch itself at eight rows too, where the trunk keeps the five.
                c0 = mlx.engine.stats().dispatches;
                const direct = (try exl3.linear.linearPairCounted(es, x, .{ .trellis = banks[0].w, .suh = banks[0].suh, .svh = banks[0].svh }, .{ .trellis = banks[1].w, .suh = banks[1].suh, .svh = banks[1].svh }, dec, .{ .rows = .{ .exact = true, .thru_bf16 = true }, .decoded_billed = true })).?;
                _ = try o.own(direct);
                try mlx.check(mlx.mlx_array_eval(direct));
                try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - c0);
                try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(try hostRows(&o, want)), std.mem.sliceAsBytes(try hostRows(&o, direct)));
                served += 1;
            }
        }
    }
    try testing.expectEqual(@as(usize, 12), served);
}

test "hy4 stq: on Sushi Cooler an STQ MoE's latent leaves W_lat_down as stored bf16, one launch up to three rows, the bytes of today's f32 latent and the routed experts' cast of it, at the release's shape, one, three and eight rows, on the machine's arm and SIMD" {
    const exl3 = @import("sushi_exl3");
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer exl3.trunk.self_finish_on = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x5e21);
    const rnd = prng.random();
    const c = try hy4_config.parseText(hy4_config.release_config);
    var served: usize = 0;
    for ([_]bool{ false, true }) |simd| {
        const prof: ?exl3.device.Scope = if (simd) exl3.device.withProfile(exl3.device.simdOnly()) else null;
        defer if (prof) |p| p.end();
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const tb = try exl3.trunk.TestBank.init(alloc, rnd, c.dim, c.moe_inter, 96);
        const tr = try tb.triple(es);
        const lin: fwd.Lin = .{ .w = undefined, .stq = .{ .w = try o.own(tr.w), .suh = try o.own(tr.s), .svh = try o.own(tr.b), .dec = exl3.format.Decode.tabled(.spec1, .w15) } };
        for ([_]c_int{ 1, 3, 8 }) |t| {
            const x = try randomArr(&o, alloc, rnd, &.{ t, @intCast(c.dim) }, -2, 4, .float32);
            // Today: the latent's launch and finish, then the routed experts' cast of it.
            exl3.trunk.self_finish_on = false;
            var c0 = mlx.engine.stats().dispatches;
            const want = try o.cast(try fwd.linear(&o, x, lin), .bfloat16);
            try mlx.check(mlx.mlx_array_eval(want));
            try testing.expectEqual(@as(u64, 3), mlx.engine.stats().dispatches - c0);
            exl3.trunk.self_finish_on = true;
            c0 = mlx.engine.stats().dispatches;
            const got = try o.cast(try fwd.expertLatent(&o, x, lin, null), .bfloat16);
            try mlx.check(mlx.mlx_array_eval(got));
            try testing.expectEqual(@as(u64, if (t <= exl3.project_rows.COUNTED_ROWS_MAX) 1 else 3), mlx.engine.stats().dispatches - c0);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(try hostRows(&o, want)), std.mem.sliceAsBytes(try hostRows(&o, got)));
            served += 1;
        }
    }
    try testing.expectEqual(@as(usize, 6), served);
}

test "hy4 stq: attendGatedCausal and attendRowsGated drop the attention core's glue launches and answer the op chain's bytes" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer fwd.chain_kernels_on = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x51a7);
    const rnd = prng.random();
    const fill = struct {
        fn f(a: std.mem.Allocator, r: std.Random, n: usize, lo: f32, span: f32) ![]f32 {
            const v = try a.alloc(f32, n);
            for (v) |*e| e.* = lo + r.float(f32) * span;
            return v;
        }
    };

    const g_slide: fwd.AttnGeometry = .{ .heads = 64, .nope = 192, .rope = 64, .v = 256, .latent = 512, .theta = 1e4, .eps = 1e-5 };
    const g_sparse: fwd.AttnGeometry = .{ .heads = 32, .nope = 192, .rope = 64, .v = 256, .latent = 512, .theta = 1e7, .eps = 1e-5 };

    for ([_]struct { g: fwd.AttnGeometry, t: c_int, n: c_int, gathered: bool }{
        .{ .g = g_slide, .t = 1, .n = 512, .gathered = false },
        .{ .g = g_slide, .t = 6, .n = 512, .gathered = false },
        .{ .g = g_sparse, .t = 1, .n = 2048, .gathered = true },
        .{ .g = g_sparse, .t = 6, .n = 2048, .gathered = true },
    }) |case| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const h = case.g.heads;
        const t = case.t;
        const n = case.n;
        const qk = case.g.nope + case.g.rope;
        const qv = try fill.f(alloc, rnd, @intCast(h * t * qk), -1.0, 2.0);
        const cv = try fill.f(alloc, rnd, @intCast((if (case.gathered) t * n else n) * case.g.latent), -0.5, 1.0);
        const pv = try fill.f(alloc, rnd, @intCast((if (case.gathered) t * n else n) * case.g.rope), -0.5, 1.0);
        const kvbv = try fill.f(alloc, rnd, @intCast(h * (case.g.nope + case.g.v) * case.g.latent), -0.1, 0.2);
        const gatev = try fill.f(alloc, rnd, @intCast(t * h * case.g.v), -1.0, 2.0);
        const sinkv = try fill.f(alloc, rnd, @intCast(h), -1.0, 2.0);

        const q = try o.own(mlx.mlx_array_new_data(qv.ptr, &[_]c_int{ h, t, qk }, 3, .float32));
        const c_arr = try o.own(mlx.mlx_array_new_data(cv.ptr, &[_]c_int{ if (case.gathered) t * n else n, case.g.latent }, 2, .float32));
        const pe_arr = try o.cast(try o.own(mlx.mlx_array_new_data(pv.ptr, &[_]c_int{ if (case.gathered) t * n else n, case.g.rope }, 2, .float32)), .bfloat16);
        const kv_b = try o.cast(try o.own(mlx.mlx_array_new_data(kvbv.ptr, &[_]c_int{ h * (case.g.nope + case.g.v), case.g.latent }, 2, .float32)), .bfloat16);
        const gate = try o.cast(try o.own(mlx.mlx_array_new_data(gatev.ptr, &[_]c_int{ t, h * case.g.v }, 2, .float32)), .bfloat16);
        const sinks = try o.own(mlx.mlx_array_new_data(sinkv.ptr, &[_]c_int{h}, 1, .float32));
        var attn: fwd.Attn = undefined;
        attn.kv_b = kv_b;
        attn.sinks = sinks;
        attn.o = .{ .w = try o.zeros(&.{ 1, 1 }, .bfloat16) };
        const kv_b32 = if (t > 1) try o.cast(kv_b, .float32) else null;
        const in: fwd.SpanIn = .{ .gate = gate, .kv_b32 = kv_b32, .whole = false };

        var keep_arr: Arr = undefined;
        if (case.gathered) {
            const kv = try alloc.alloc(bool, @intCast(t * n));
            for (kv, 0..) |*d, i| d.* = (i % 3) != 0;
            keep_arr = try o.own(mlx.mlx_array_new_data(kv.ptr, &[_]c_int{ t, n }, 2, .bool_));
            try mlx.check(mlx.mlx_array_eval(keep_arr));
        }
        for ([_]Arr{ q, c_arr, pe_arr, kv_b, gate, sinks }) |arr| try mlx.check(mlx.mlx_array_eval(arr));
        if (kv_b32) |k32| try mlx.check(mlx.mlx_array_eval(k32));

        fwd.chain_kernels_on = false;
        const d0 = mlx.engine.stats().dispatches;
        const want = if (case.gathered)
            try fwd.attendRowsGated(&o, in, q, c_arr, pe_arr, keep_arr, attn, case.g, null)
        else
            try fwd.attendGatedCausal(&o, in, q, c_arr, pe_arr, attn, case.g, if (t == 1) n - 1 else n - t, if (t == 1) 0 else n, null);
        try mlx.check(mlx.mlx_array_eval(want));
        const off_launches = mlx.engine.stats().dispatches - d0;

        fwd.chain_kernels_on = true;
        const d1 = mlx.engine.stats().dispatches;
        const got = if (case.gathered)
            try fwd.attendRowsGated(&o, in, q, c_arr, pe_arr, keep_arr, attn, case.g, null)
        else
            try fwd.attendGatedCausal(&o, in, q, c_arr, pe_arr, attn, case.g, if (t == 1) n - 1 else n - t, if (t == 1) 0 else n, null);
        try mlx.check(mlx.mlx_array_eval(got));
        const on_launches = mlx.engine.stats().dispatches - d1;

        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(try hostRows(&o, want)), std.mem.sliceAsBytes(try hostRows(&o, got)));
        const min_drop: u64 = if (case.gathered) 6 else 8;
        try testing.expect(off_launches >= on_launches + min_drop);
    }
}

const RowTap = struct {
    const Rec = struct { layer: usize, kind: fwd.TapKind, rows: usize, data: []f32 };
    a: std.mem.Allocator,
    recs: std.ArrayList(Rec) = .empty,
    tap: fwd.Tap = undefined,

    fn init(self: *RowTap) void {
        self.tap = .{ .ctx = self, .input = input, .routes = routes, .expert_act = expertAct };
    }

    fn deinit(self: *RowTap) void {
        for (self.recs.items) |r| self.a.free(r.data);
        self.recs.deinit(self.a);
    }

    fn input(ctx: *anyopaque, layer: usize, kind: fwd.TapKind, x: Arr) anyerror!void {
        const self: *RowTap = @ptrCast(@alignCast(ctx));
        if (kind == .expert_in) return;
        var o: Ops = .{ .s = mlx.engine.stream() };
        defer o.deinit();
        const v = try hostRows(&o, x);
        try self.recs.append(self.a, .{ .layer = layer, .kind = kind, .rows = @intCast(mlx.getShape(x)[0]), .data = try self.a.dupe(f32, v) });
    }

    fn routes(_: *anyopaque, _: usize, _: []const u32, _: Arr) anyerror!void {}
    fn expertAct(_: *anyopaque, _: usize, _: Arr, _: []const u32) anyerror!void {}

    fn row(self: *const RowTap, layer: usize, kind: fwd.TapKind, r: usize) ?[]const f32 {
        var at: usize = 0;
        for (self.recs.items) |rec| {
            if (rec.layer != layer or rec.kind != kind) continue;
            const w = rec.data.len / rec.rows;
            if (r < at + rec.rows) return rec.data[(r - at) * w ..][0..w];
            at += rec.rows;
        }
        return null;
    }
};

test "hy4 stq: a 6-row verify block's rows are their one-row steps' bytes at every linear's input and in the logits (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var p = try Packs.open(es);
    defer p.close();
    var ids: [48]u32 = undefined;
    var rng = std.Random.DefaultPrng.init(0x5e7);
    for (&ids) |*d| d.* = rng.random().uintLessThan(u32, @intCast(p.config.vocab - 1)) + 1;
    const block = 6;
    var o: Ops = .{ .s = es };
    defer o.deinit();
    var steps: [block]RowTap = undefined;
    var step_logits: [block][]const f32 = undefined;
    {
        const m = try hy4_model.Model.initWith(a, p.config, &p.stq, es, .{ .dir = p.stq_dir, .quant = p.quant });
        defer m.deinit();
        _ = try o.own(try m.forwardReadout(ids[0..40], .none));
        for (0..block) |r| {
            steps[r] = .{ .a = a };
            steps[r].init();
            m.tap = &steps[r].tap;
            step_logits[r] = try hostRows(&o, try o.own(try m.forwardReadout(ids[40 + r .. 41 + r], .last)));
            m.tap = null;
        }
    }
    defer for (&steps) |*t| t.deinit();
    var wide: RowTap = .{ .a = a };
    wide.init();
    defer wide.deinit();
    var p2 = try Packs.open(es);
    defer p2.close();
    const m = try hy4_model.Model.initWith(a, p2.config, &p2.stq, es, .{ .dir = p2.stq_dir, .quant = p2.quant });
    defer m.deinit();
    _ = try o.own(try m.forwardReadout(ids[0..40], .none));
    m.tap = &wide.tap;
    const lg = try hostRows(&o, try o.own(try m.forwardReadout(ids[40 .. 40 + block], .all)));
    m.tap = null;
    var parted: usize = 0;
    for (wide.recs.items) |rec| for (0..rec.rows) |r| {
        const want = steps[r].row(rec.layer, rec.kind, 0) orelse continue;
        const got = wide.row(rec.layer, rec.kind, r).?;
        if (std.mem.eql(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got))) continue;
        if (parted == 0) {
            var worst: f32 = 0;
            for (want, got) |x, y| worst = @max(worst, @abs(x - y));
            std.debug.print("width probe: layer {d} {t} row {d} parts first, worst |d| {e:.3}\n", .{ rec.layer, rec.kind, r, worst });
        }
        parted += 1;
    };
    const vocab = p.config.vocab;
    var rows_parted: usize = 0;
    for (0..block) |r| {
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(step_logits[r]), std.mem.sliceAsBytes(lg[r * vocab ..][0..vocab]))) rows_parted += 1;
    }
    errdefer std.debug.print("width probe: {d} tap rows parted, {d} of {d} logit rows\n", .{ parted, rows_parted, block });
    try testing.expectEqual(@as(usize, 0), parted);
    try testing.expectEqual(@as(usize, 0), rows_parted);
}

fn copyF32Bytes(a: std.mem.Allocator, arr: Arr) ![]u8 {
    var o: Ops = .{ .s = mlx.engine.stream() };
    defer o.deinit();
    const c = try o.contiguous(try o.cast(arr, .float32));
    try mlx.check(mlx.mlx_array_eval(c));
    const raw = (mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..mlx.mlx_array_size(c)];
    return a.dupe(u8, std.mem.sliceAsBytes(raw));
}

test "hy4 hc: a decode step and a 3-row verify block make as many dispatches with the wide mix as with one threadgroup a row, and give its logits (HY4_STQ_TINY)" {
    const contract = @import("../cooler_contract_test.zig");
    const hy4_contract = @import("hy4_contract.zig");
    const hy4_hc = @import("hy4_hc.zig");
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    var cooler: contract.Load = undefined;
    try cooler.open(hy4_contract, dir, true);
    defer _ = cooler.close();
    const m = cooler.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const turn = hy4_contract.openTurn(cooler.h, &cooler.config);
    defer turn.end();
    defer hy4_hc.wide_on = true;
    var ids: [44]u32 = undefined;
    var rng = std.Random.DefaultPrng.init(0x4c3);
    for (&ids) |*t| t.* = rng.random().uintLessThan(u32, @intCast(m.config.vocab - 1)) + 1;
    var got: [2][2][]u8 = undefined;
    var made: usize = 0;
    defer for (0..made) |i| a.free(got[i / 2][i % 2]);
    var counts: [2][2]u64 = undefined;
    for ([_]bool{ false, true }, 0..) |on, arm| {
        hy4_hc.wide_on = on;
        m.reset();
        _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..40], .none));
        for ([_][2]usize{ .{ 40, 41 }, .{ 41, 44 } }, 0..) |span, i| {
            const wide_before = hy4_hc.wide_launches;
            const before = mlx.engine.stats().dispatches;
            const lg = try m.forwardReadout(ids[span[0]..span[1]], .all);
            defer _ = mlx.mlx_array_free(lg);
            counts[arm][i] = mlx.engine.stats().dispatches - before;
            try testing.expectEqual(on, hy4_hc.wide_launches > wide_before);
            got[arm][i] = try copyF32Bytes(a, lg);
            made += 1;
        }
    }
    for (0..2) |i| {
        try testing.expectEqual(counts[0][i], counts[1][i]);
        try testing.expectEqualSlices(u8, got[0][i], got[1][i]);
    }
}

test "hy4 hc: a decode step and 3- and 4-row verify blocks make as many dispatches with W_o finishing its own partials (up to o_rows_max rows) as with the iHC mix finishing them, and give its logits (HY4_STQ_TINY)" {
    const contract = @import("../cooler_contract_test.zig");
    const hy4_contract = @import("hy4_contract.zig");
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    var cooler: contract.Load = undefined;
    try cooler.open(hy4_contract, dir, true);
    defer _ = cooler.close();
    const m = cooler.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const turn = hy4_contract.openTurn(cooler.h, &cooler.config);
    defer turn.end();
    defer fwd.o_rows_on = true;
    var ids: [48]u32 = undefined;
    var rng = std.Random.DefaultPrng.init(0x4c4);
    for (&ids) |*t| t.* = rng.random().uintLessThan(u32, @intCast(m.config.vocab - 1)) + 1;
    var got: [2][3][]u8 = undefined;
    var made: usize = 0;
    defer for (0..made) |i| a.free(got[i / 3][i % 3]);
    var counts: [2][3]u64 = undefined;
    for ([_]bool{ false, true }, 0..) |on, arm| {
        fwd.o_rows_on = on;
        m.reset();
        _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..40], .none));
        for ([_][2]usize{ .{ 40, 41 }, .{ 41, 44 }, .{ 44, 48 } }, 0..) |span, i| {
            const finished_before = hy4_stq.finished_launches;
            const before = mlx.engine.stats().dispatches;
            const lg = try m.forwardReadout(ids[span[0]..span[1]], .all);
            defer _ = mlx.mlx_array_free(lg);
            counts[arm][i] = mlx.engine.stats().dispatches - before;
            // Every layer's W_o, when on and within its rows.
            const finishes = on and span[1] - span[0] <= fwd.o_rows_max;
            try testing.expectEqual(@as(u64, if (finishes) m.config.layers else 0), hy4_stq.finished_launches - finished_before);
            got[arm][i] = try copyF32Bytes(a, lg);
            made += 1;
        }
    }
    for (0..3) |i| {
        try testing.expectEqual(counts[0][i], counts[1][i]);
        try testing.expectEqualSlices(u8, got[0][i], got[1][i]);
    }
}

test "hy4 hc: a decode step makes one dispatch fewer a dense layer with every trunk launch finishing its own partials (the tiny's other trunk launches fold at its 128 width, W_o finishes itself already; a normed launch finishes itself at one row), 3- and 4-row verify blocks as many, and all give the same logits (HY4_STQ_TINY)" {
    const contract = @import("../cooler_contract_test.zig");
    const hy4_contract = @import("hy4_contract.zig");
    const exl3 = @import("sushi_exl3");
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    var cooler: contract.Load = undefined;
    try cooler.open(hy4_contract, dir, true);
    defer _ = cooler.close();
    const m = cooler.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const turn = hy4_contract.openTurn(cooler.h, &cooler.config);
    defer turn.end();
    defer exl3.trunk.self_finish_on = true;
    // The dense layer's down and its norm (its 256-wide input splits k in two) is the only trunk launch past W_o that
    // leaves partials.
    try testing.expectEqual(@as(u32, 1), m.config.dense_layers);
    var ids: [48]u32 = undefined;
    var rng = std.Random.DefaultPrng.init(0x5e1f);
    for (&ids) |*t| t.* = rng.random().uintLessThan(u32, @intCast(m.config.vocab - 1)) + 1;
    var got: [2][3][]u8 = undefined;
    var made: usize = 0;
    defer for (0..made) |i| a.free(got[i / 3][i % 3]);
    var counts: [2][3]u64 = undefined;
    for ([_]bool{ false, true }, 0..) |on, arm| {
        exl3.trunk.self_finish_on = on;
        m.reset();
        _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..40], .none));
        for ([_][2]usize{ .{ 40, 41 }, .{ 41, 44 }, .{ 44, 48 } }, 0..) |span, i| {
            const before = mlx.engine.stats().dispatches;
            const lg = try m.forwardReadout(ids[span[0]..span[1]], .all);
            defer _ = mlx.mlx_array_free(lg);
            counts[arm][i] = mlx.engine.stats().dispatches - before;
            got[arm][i] = try copyF32Bytes(a, lg);
            made += 1;
        }
    }
    for (0..3) |i| try testing.expectEqualSlices(u8, got[0][i], got[1][i]);
    try testing.expectEqual(counts[0][0] - m.config.dense_layers, counts[1][0]);
    try testing.expectEqual(counts[0][1], counts[1][1]);
    try testing.expectEqual(counts[0][2], counts[1][2]);
}

test "hy4 verify: 2..8-row blocks match one-row steps and absorb_kernels_on toggle when hy4_absorb does not serve (HY4_BF16_TINY, HY4_STQ_TINY_MIXED)" {
    const contract = @import("../cooler_contract_test.zig");
    const hy4_contract = @import("hy4_contract.zig");
    const a = testing.allocator;
    const saved_a = fwd.absorb_kernels_on;
    defer fwd.absorb_kernels_on = saved_a;
    var ran: usize = 0;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY_MIXED" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran += 1;
        var cooler: contract.Load = undefined;
        try cooler.open(hy4_contract, dir, true);
        defer _ = cooler.close();
        const m = cooler.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
        const turn = hy4_contract.openTurn(cooler.h, &cooler.config);
        defer turn.end();

        const vocab = m.config.vocab;
        const row_bytes = vocab * 4;
        var ids: [64]u32 = undefined;
        var rng = std.Random.DefaultPrng.init(0x9a11);
        for (&ids) |*t| t.* = rng.random().uintLessThan(u32, @intCast(vocab - 1)) + 1;

        for ([_]usize{ 5, 20 }) |pre| {
            for ([_]usize{ 6, 8 }) |block| {
                m.reset();
                const pre_lg = try m.forwardReadout(ids[0..pre], .none);
                _ = mlx.mlx_array_free(pre_lg);
                var step_bytes: [8][]u8 = undefined;
                var made: usize = 0;
                defer for (step_bytes[0..made]) |s| a.free(s);
                for (0..block) |r| {
                    const lg = try m.forwardReadout(ids[pre + r .. pre + r + 1], .last);
                    defer _ = mlx.mlx_array_free(lg);
                    step_bytes[r] = try copyF32Bytes(a, lg);
                    made += 1;
                }
                m.reset();
                const pre_lg2 = try m.forwardReadout(ids[0..pre], .none);
                _ = mlx.mlx_array_free(pre_lg2);
                const blg = try m.forwardReadout(ids[pre .. pre + block], .all);
                defer _ = mlx.mlx_array_free(blg);
                const wide = try copyF32Bytes(a, blg);
                defer a.free(wide);
                for (0..block) |r| {
                    errdefer std.debug.print("verify width diff on {s} pre={d} block={d} row={d}\n", .{ env, pre, block, r });
                    try testing.expectEqualSlices(u8, step_bytes[r], wide[r * row_bytes ..][0..row_bytes]);
                }
            }
        }

        fwd.absorb_kernels_on = true;
        m.reset();
        const p_on = try m.forwardReadout(ids[0..40], .last);
        defer _ = mlx.mlx_array_free(p_on);
        const pb_on = try copyF32Bytes(a, p_on);
        defer a.free(pb_on);
        const b_on = try m.forwardReadout(ids[40..46], .all);
        defer _ = mlx.mlx_array_free(b_on);
        const bb_on = try copyF32Bytes(a, b_on);
        defer a.free(bb_on);
        const d_on = try m.forwardReadout(ids[46..47], .last);
        defer _ = mlx.mlx_array_free(d_on);
        const db_on = try copyF32Bytes(a, d_on);
        defer a.free(db_on);

        fwd.absorb_kernels_on = false;
        m.reset();
        const p_off = try m.forwardReadout(ids[0..40], .last);
        defer _ = mlx.mlx_array_free(p_off);
        const pb_off = try copyF32Bytes(a, p_off);
        defer a.free(pb_off);
        const b_off = try m.forwardReadout(ids[40..46], .all);
        defer _ = mlx.mlx_array_free(b_off);
        const bb_off = try copyF32Bytes(a, b_off);
        defer a.free(bb_off);
        const d_off = try m.forwardReadout(ids[46..47], .last);
        defer _ = mlx.mlx_array_free(d_off);
        const db_off = try copyF32Bytes(a, d_off);
        defer a.free(db_off);
        fwd.absorb_kernels_on = true;

        errdefer std.debug.print("absorb_kernels_on seam diff on {s}\n", .{env});
        try testing.expectEqualSlices(u8, pb_on, pb_off);
        try testing.expectEqualSlices(u8, bb_on, bb_off);
        try testing.expectEqualSlices(u8, db_on, db_off);
    }
    if (ran == 0) return error.SkipZigTest;
}

test "hy4 attention spans: multi-span prefill chunks with 2, 4 and 8-row tails match spans_share off (HY4_STQ_TINY, HY4_BF16_TINY, HY4_STQ_TINY_MIXED)" {
    const contract = @import("../cooler_contract_test.zig");
    const hy4_contract = @import("hy4_contract.zig");
    const a = testing.allocator;
    const saved_share = fwd.spans_share;
    const saved_prod = fwd.products;
    defer {
        fwd.spans_share = saved_share;
        fwd.products = saved_prod;
    }
    var ran: usize = 0;
    for ([_][:0]const u8{ "HY4_STQ_TINY", "HY4_BF16_TINY", "HY4_STQ_TINY_MIXED" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran += 1;
        // A load measures its region with the served seams, not the last fixture's arm.
        fwd.spans_share = saved_share;
        fwd.products = saved_prod;
        var cooler: contract.Load = undefined;
        try cooler.open(hy4_contract, dir, true);
        defer _ = cooler.close();
        const m = cooler.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
        const turn = hy4_contract.openTurn(cooler.h, &cooler.config);
        defer turn.end();

        const vocab = m.config.vocab;
        var ids: [300]u32 = undefined;
        var rng = std.Random.DefaultPrng.init(0x9a11);
        for (&ids) |*t| t.* = rng.random().uintLessThan(u32, @intCast(vocab - 1)) + 1;

        for ([_]fwd.Products{ .bf16, .f32 }) |prod| {
            if (prod == .f32 and !std.mem.eql(u8, env, "HY4_BF16_TINY")) continue;
            fwd.products = prod;
            for ([_]usize{ 34, 36, 40, 264 }) |len| {
                fwd.spans_share = true;
                m.reset();
                const lg_on = try m.forwardReadout(ids[0..len], .all);
                defer _ = mlx.mlx_array_free(lg_on);
                const b_on = try copyF32Bytes(a, lg_on);
                defer a.free(b_on);

                fwd.spans_share = false;
                m.reset();
                const lg_off = try m.forwardReadout(ids[0..len], .all);
                defer _ = mlx.mlx_array_free(lg_off);
                const b_off = try copyF32Bytes(a, lg_off);
                defer a.free(b_off);

                errdefer std.debug.print("spans_share diff on {s} prod={t} len={d}\n", .{ env, prod, len });
                try testing.expectEqualSlices(u8, b_off, b_on);
            }
        }
    }
    if (ran == 0) return error.SkipZigTest;
}

test "hy4 verify launches: a 6-row verify router makes one launch past logits as one row does" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();

    const a = testing.allocator;
    const e: c_int = 512;
    const d: c_int = 3072;
    const k: c_int = 16;
    var o: Ops = .{ .s = es };
    defer o.deinit();
    var prng = std.Random.DefaultPrng.init(0x7007);
    const rnd = prng.random();
    const wv = try a.alloc(f32, @intCast(e * d));
    defer a.free(wv);
    for (wv) |*v| v.* = rnd.floatNorm(f32) * 0.05;
    const bv = try a.alloc(f32, @intCast(e));
    defer a.free(bv);
    for (bv) |*v| v.* = rnd.floatNorm(f32) * 0.1;
    const mv = try a.alloc(i32, @intCast(e));
    defer a.free(mv);
    for (mv, 0..) |*v, i| v.* = @intCast((i * 19 + 3) % @as(usize, @intCast(e)));
    const xv = try a.alloc(f32, @intCast(6 * d));
    defer a.free(xv);
    for (xv) |*v| v.* = rnd.floatNorm(f32);

    const w = try o.cast(try o.own(mlx.mlx_array_new_data(wv.ptr, &[_]c_int{ e, d }, 2, .float32)), .bfloat16);
    const bias = try o.own(mlx.mlx_array_new_data(bv.ptr, &[_]c_int{e}, 1, .float32));
    const map = try o.own(mlx.mlx_array_new_data(mv.ptr, &[_]c_int{e}, 1, .int32));
    const x6 = try o.own(mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ 6, d }, 2, .float32));
    const x1 = try o.contiguous(try o.slice(x6, 0, 0, 1));
    for ([_]Arr{ w, bias, map, x6, x1 }) |arr| try mlx.check(mlx.mlx_array_eval(arr));

    const router: fwd.Router = .{ .weight = w, .bias = bias };
    for ([_]?Arr{ map, null }) |m| {
        const d0 = mlx.engine.stats().dispatches;
        const r1 = try fwd.routeMapped(&o, x1, router, k, 3.998, m);
        try mlx.check(mlx.mlx_array_eval(r1.routed.weights));
        if (r1.slots) |sl| try mlx.check(mlx.mlx_array_eval(sl));
        const one = mlx.engine.stats().dispatches - d0;

        const d1 = mlx.engine.stats().dispatches;
        const r6 = try fwd.routeMapped(&o, x6, router, k, 3.998, m);
        try mlx.check(mlx.mlx_array_eval(r6.routed.weights));
        if (r6.slots) |sl| try mlx.check(mlx.mlx_array_eval(sl));
        const six = mlx.engine.stats().dispatches - d1;

        try testing.expectEqual(@as(u64, 2), one);
        try testing.expectEqual(one, six);
    }
}

test "hy4 verify launches: a 6-row verify attention builds its ring keys and row joins once per block, not per row" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x51a8);
    const rnd = prng.random();
    const fill = struct {
        fn f(a: std.mem.Allocator, r: std.Random, n: usize, lo: f32, span: f32) ![]f32 {
            const v = try a.alloc(f32, n);
            for (v) |*e| e.* = lo + r.float(f32) * span;
            return v;
        }
    };

    const g_slide: fwd.AttnGeometry = .{ .heads = 64, .nope = 192, .rope = 64, .v = 256, .latent = 512, .theta = 1e4, .eps = 1e-5 };
    const g_sparse: fwd.AttnGeometry = .{ .heads = 32, .nope = 192, .rope = 64, .v = 256, .latent = 512, .theta = 1e7, .eps = 1e-5 };

    {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const t: c_int = 6;
        const w: c_int = 512;
        const n: c_int = w - 1 + t;
        const h = g_slide.heads;
        const qk = g_slide.nope + g_slide.rope;
        const q = try o.own(mlx.mlx_array_new_data((try fill.f(alloc, rnd, @intCast(h * t * qk), -1, 2)).ptr, &[_]c_int{ h, t, qk }, 3, .float32));
        const c_arr = try o.own(mlx.mlx_array_new_data((try fill.f(alloc, rnd, @intCast(n * g_slide.latent), -0.5, 1)).ptr, &[_]c_int{ n, g_slide.latent }, 2, .float32));
        const pe_arr = try o.cast(try o.own(mlx.mlx_array_new_data((try fill.f(alloc, rnd, @intCast(n * g_slide.rope), -0.5, 1)).ptr, &[_]c_int{ n, g_slide.rope }, 2, .float32)), .bfloat16);
        const kv_b = try o.cast(try o.own(mlx.mlx_array_new_data((try fill.f(alloc, rnd, @intCast(h * (g_slide.nope + g_slide.v) * g_slide.latent), -0.1, 0.2)).ptr, &[_]c_int{ h * (g_slide.nope + g_slide.v), g_slide.latent }, 2, .float32)), .bfloat16);
        const gate = try o.cast(try o.own(mlx.mlx_array_new_data((try fill.f(alloc, rnd, @intCast(t * h * g_slide.v), -1, 2)).ptr, &[_]c_int{ t, h * g_slide.v }, 2, .float32)), .bfloat16);
        const sinks = try o.own(mlx.mlx_array_new_data((try fill.f(alloc, rnd, @intCast(h), -1, 2)).ptr, &[_]c_int{h}, 1, .float32));
        var attn: fwd.Attn = undefined;
        attn.kv_b = kv_b;
        attn.sinks = sinks;
        attn.o = .{ .w = try o.zeros(&.{ 1, 1 }, .bfloat16) };
        for ([_]Arr{ q, c_arr, pe_arr, kv_b, gate, sinks, attn.o.w }) |arr| try mlx.check(mlx.mlx_array_eval(arr));

        const d0 = mlx.engine.stats().dispatches;
        const out = try fwd.attendGatedRing(&o, .{ .gate = gate, .kv_b32 = null, .whole = false }, q, c_arr, pe_arr, attn, g_slide, w - 1, w, 4096, null);
        try mlx.check(mlx.mlx_array_eval(out));
        const launches = mlx.engine.stats().dispatches - d0;
        errdefer std.debug.print("sliding 6-row verify attention launches: {d}\n", .{launches});
        try testing.expect(launches <= 41);
    }

    {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const t: c_int = 6;
        const n: c_int = 2048;
        const h = g_sparse.heads;
        const qk = g_sparse.nope + g_sparse.rope;
        const q = try o.own(mlx.mlx_array_new_data((try fill.f(alloc, rnd, @intCast(h * t * qk), -1, 2)).ptr, &[_]c_int{ h, t, qk }, 3, .float32));
        const c_arr = try o.own(mlx.mlx_array_new_data((try fill.f(alloc, rnd, @intCast(t * n * g_sparse.latent), -0.5, 1)).ptr, &[_]c_int{ t * n, g_sparse.latent }, 2, .float32));
        const pe_arr = try o.cast(try o.own(mlx.mlx_array_new_data((try fill.f(alloc, rnd, @intCast(t * n * g_sparse.rope), -0.5, 1)).ptr, &[_]c_int{ t * n, g_sparse.rope }, 2, .float32)), .bfloat16);
        const kv_b = try o.cast(try o.own(mlx.mlx_array_new_data((try fill.f(alloc, rnd, @intCast(h * (g_sparse.nope + g_sparse.v) * g_sparse.latent), -0.1, 0.2)).ptr, &[_]c_int{ h * (g_sparse.nope + g_sparse.v), g_sparse.latent }, 2, .float32)), .bfloat16);
        const gate = try o.cast(try o.own(mlx.mlx_array_new_data((try fill.f(alloc, rnd, @intCast(t * h * g_sparse.v), -1, 2)).ptr, &[_]c_int{ t, h * g_sparse.v }, 2, .float32)), .bfloat16);
        const sinks = try o.own(mlx.mlx_array_new_data((try fill.f(alloc, rnd, @intCast(h), -1, 2)).ptr, &[_]c_int{h}, 1, .float32));
        const kv = try alloc.alloc(bool, @intCast(t * n));
        for (kv, 0..) |*d, i| d.* = (i % 3) != 0;
        const keep = try o.own(mlx.mlx_array_new_data(kv.ptr, &[_]c_int{ t, n }, 2, .bool_));
        var attn: fwd.Attn = undefined;
        attn.kv_b = kv_b;
        attn.sinks = sinks;
        attn.o = .{ .w = try o.zeros(&.{ 1, 1 }, .bfloat16) };
        for ([_]Arr{ q, c_arr, pe_arr, kv_b, gate, sinks, keep, attn.o.w }) |arr| try mlx.check(mlx.mlx_array_eval(arr));

        const d0 = mlx.engine.stats().dispatches;
        const out = try fwd.attendRowsGated(&o, .{ .gate = gate, .kv_b32 = null, .whole = false }, q, c_arr, pe_arr, keep, attn, g_sparse, null);
        try mlx.check(mlx.mlx_array_eval(out));
        const launches = mlx.engine.stats().dispatches - d0;
        errdefer std.debug.print("sparse 6-row verify attention launches: {d}\n", .{launches});
        try testing.expect(launches <= 30);
    }
}

test "hy4 verify: a 6..8-row BF16 linear at W_kva's 576x3072 shape gives each row its one-row step's bytes" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();

    const a = testing.allocator;
    const n: c_int = 576;
    const k: c_int = 3072;
    const t: c_int = 8;
    var o: Ops = .{ .s = es };
    defer o.deinit();
    var prng = std.Random.DefaultPrng.init(0x5760);
    const wv = try a.alloc(f32, @intCast(n * k));
    defer a.free(wv);
    for (wv) |*v| v.* = prng.random().floatNorm(f32) * 0.05;
    const xv = try a.alloc(f32, @intCast(t * k));
    defer a.free(xv);
    for (xv) |*v| v.* = prng.random().floatNorm(f32);
    const w = try o.cast(try o.own(mlx.mlx_array_new_data(wv.ptr, &[_]c_int{ n, k }, 2, .float32)), .bfloat16);
    const x = try o.own(mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ t, k }, 2, .float32));
    try mlx.check(mlx.mlx_array_eval(w));
    try mlx.check(mlx.mlx_array_eval(x));

    const lin: fwd.Lin = .{ .w = w };
    const y_blk = try copyF32Bytes(a, try fwd.linear(&o, x, lin));
    defer a.free(y_blk);
    const row_b = @as(usize, @intCast(n)) * 4;
    for (0..@as(usize, @intCast(t))) |r| {
        const ri: c_int = @intCast(r);
        const y1 = try copyF32Bytes(a, try fwd.linear(&o, try o.slice(x, 0, ri, ri + 1), lin));
        defer a.free(y1);
        try testing.expectEqualSlices(u8, y1, y_blk[r * row_b ..][0..row_b]);
    }
}

test "hy4 verify launches: a 2..8-row BF16 linear at W_kva's shape is one launch whatever the rows, each row its one-row step's bytes" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();

    const a = testing.allocator;
    const k_max: c_int = 8;
    var o: Ops = .{ .s = es };
    defer o.deinit();
    var prng = std.Random.DefaultPrng.init(0x5761);
    for ([_][2]c_int{ .{ 576, 3072 }, .{ 144, 128 } }) |sh| {
        const n = sh[0];
        const k = sh[1];
        const wv = try a.alloc(f32, @intCast(n * k));
        defer a.free(wv);
        for (wv) |*v| v.* = prng.random().floatNorm(f32) * 0.05;
        const xv = try a.alloc(f32, @intCast(k_max * k));
        defer a.free(xv);
        for (xv) |*v| v.* = prng.random().floatNorm(f32);
        const w = try o.cast(try o.own(mlx.mlx_array_new_data(wv.ptr, &[_]c_int{ n, k }, 2, .float32)), .bfloat16);
        const x_all = try o.own(mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ k_max, k }, 2, .float32));
        try mlx.check(mlx.mlx_array_eval(w));
        try mlx.check(mlx.mlx_array_eval(x_all));
        const lin: fwd.Lin = .{ .w = w };
        const row_b = @as(usize, @intCast(n)) * 4;
        var t: c_int = 2;
        while (t <= k_max) : (t += 1) {
            const x = try o.contiguous(try o.slice(x_all, 0, 0, t));
            try mlx.check(mlx.mlx_array_eval(x));
            const d0 = mlx.engine.stats().dispatches;
            const y = try o.contiguous(try fwd.linear(&o, x, lin));
            try mlx.check(mlx.mlx_array_eval(y));
            const launches = mlx.engine.stats().dispatches - d0;
            errdefer std.debug.print("{d}-row linear at {d}x{d}: {d} launches\n", .{ t, n, k, launches });
            try testing.expectEqual(@as(u64, 1), launches);
            const y_blk = try copyF32Bytes(a, y);
            defer a.free(y_blk);
            for (0..@as(usize, @intCast(t))) |r| {
                const ri: c_int = @intCast(r);
                const y1 = try copyF32Bytes(a, try fwd.linear(&o, try o.contiguous(try o.slice(x_all, 0, ri, ri + 1)), lin));
                defer a.free(y1);
                try testing.expectEqualSlices(u8, y1, y_blk[r * row_b ..][0..row_b]);
            }
        }
    }
}

test "hy4 verify: an 8-row block on HY4_BF16_TINY gives each row its one-row step's logits" {
    const contract = @import("../cooler_contract_test.zig");
    const hy4_contract = @import("hy4_contract.zig");
    const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    var cooler: contract.Load = undefined;
    try cooler.open(hy4_contract, dir, true);
    defer _ = cooler.close();
    const m = cooler.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const turn = hy4_contract.openTurn(cooler.h, &cooler.config);
    defer turn.end();

    const vocab = m.config.vocab;
    const row_bytes = vocab * 4;
    var ids: [32]u32 = undefined;
    var rng = std.Random.DefaultPrng.init(0x55aa);
    for (&ids) |*tok| tok.* = rng.random().uintLessThan(u32, @intCast(vocab - 1)) + 1;

    const pre: usize = 15;
    const block: usize = 8;
    m.reset();
    const pre_lg = try m.forwardReadout(ids[0..pre], .none);
    _ = mlx.mlx_array_free(pre_lg);
    var step_bytes: [8][]u8 = undefined;
    var made: usize = 0;
    defer for (step_bytes[0..made]) |s| a.free(s);
    for (0..block) |r| {
        const lg = try m.forwardReadout(ids[pre + r .. pre + r + 1], .last);
        defer _ = mlx.mlx_array_free(lg);
        step_bytes[r] = try copyF32Bytes(a, lg);
        made += 1;
    }

    m.reset();
    const pre_lg2 = try m.forwardReadout(ids[0..pre], .none);
    _ = mlx.mlx_array_free(pre_lg2);
    const blg = try m.forwardReadout(ids[pre .. pre + block], .all);
    defer _ = mlx.mlx_array_free(blg);
    const wide = try copyF32Bytes(a, blg);
    defer a.free(wide);
    for (0..block) |r| {
        try testing.expectEqualSlices(u8, step_bytes[r], wide[r * row_bytes ..][0..row_bytes]);
    }
}

test "hy4 stq: every verify width 2..8 is its one-row steps' bytes at every linear's input and in the logits, below and past the selection threshold and across the sliding window's wrap (HY4_STQ_TINY, HY4_BF16_TINY_STQ)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var p = try Packs.open(es);
    defer p.close();
    const m = try hy4_model.Model.initWith(a, p.config, &p.stq, es, .{ .dir = p.stq_dir, .quant = p.quant });
    defer m.deinit();
    var ids: [48]u32 = undefined;
    var rng = std.Random.DefaultPrng.init(0x5e7);
    for (&ids) |*d| d.* = rng.random().uintLessThan(u32, @intCast(p.config.vocab - 1)) + 1;
    const max_w = 8;
    const vocab = p.config.vocab;
    const thr: usize = p.config.indexer.group_topk * p.config.indexer.group;
    var starts: [12]usize = undefined;
    var n_starts: usize = 0;
    starts[n_starts] = 5;
    n_starts += 1;
    for (thr - 8..thr + 2) |p0| {
        starts[n_starts] = p0;
        n_starts += 1;
    }
    starts[n_starts] = 21;
    n_starts += 1;
    for (starts[0..n_starts]) |p0| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        var steps: [max_w]RowTap = undefined;
        var made: usize = 0;
        defer for (steps[0..made]) |*t| t.deinit();
        var step_logits: [max_w][]const f32 = undefined;
        m.reset();
        _ = try o.own(try m.forwardReadout(ids[0..p0], .none));
        for (0..max_w) |r| {
            steps[r] = .{ .a = a };
            steps[r].init();
            made += 1;
            m.tap = &steps[r].tap;
            step_logits[r] = try hostRows(&o, try o.own(try m.forwardReadout(ids[p0 + r .. p0 + r + 1], .last)));
            m.tap = null;
        }
        for (2..max_w + 1) |width| {
            var wo: Ops = .{ .s = es };
            defer wo.deinit();
            var wide: RowTap = .{ .a = a };
            wide.init();
            defer wide.deinit();
            m.reset();
            _ = try wo.own(try m.forwardReadout(ids[0..p0], .none));
            m.tap = &wide.tap;
            const lg = try hostRows(&wo, try wo.own(try m.forwardReadout(ids[p0 .. p0 + width], .all)));
            m.tap = null;
            var parted: usize = 0;
            for (wide.recs.items) |rec| for (0..rec.rows) |r| {
                const want = steps[r].row(rec.layer, rec.kind, 0) orelse continue;
                const got = wide.row(rec.layer, rec.kind, r).?;
                if (std.mem.eql(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got))) continue;
                if (parted == 0) {
                    var worst: f32 = 0;
                    for (want, got) |x, y| worst = @max(worst, @abs(x - y));
                    std.debug.print("width probe: p0={d} width={d} layer {d} {t} row {d} parts first, worst |d| {e:.3}\n", .{ p0, width, rec.layer, rec.kind, r, worst });
                }
                parted += 1;
            };
            var rows_parted: usize = 0;
            for (0..width) |r| {
                if (!std.mem.eql(u8, std.mem.sliceAsBytes(step_logits[r]), std.mem.sliceAsBytes(lg[r * vocab ..][0..vocab]))) rows_parted += 1;
            }
            diag("width probe: p0={d} width={d} taps={d} parted={d} logits_parted={d}\n", .{ p0, width, wide.recs.items.len, parted, rows_parted });
            errdefer std.debug.print("width probe: p0={d} width={d}: {d} tap rows parted, {d} of {d} logit rows\n", .{ p0, width, parted, rows_parted, width });
            try testing.expectEqual(@as(usize, 0), parted);
            try testing.expectEqual(@as(usize, 0), rows_parted);
        }
    }
}
