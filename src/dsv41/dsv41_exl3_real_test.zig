//! DeepSeek-V4.1-Flash STQ kernels on real shards (`DSV41_RT_PACK`, a pack with `model-trunk-LL.safetensors` and
//! `model-layer-LL.safetensors`): the trunk linears and two experts of every rate group of a converted layer, each scored
//! against a dense f64 product of the host tile decode, the f32 truth. Skipped without the pack.
const std = @import("std");
const mlx = @import("../mlx.zig");
const exl3 = @import("sushi_exl3");
const format = exl3.format;
const kernels = exl3.kernels;
const rate_groups = exl3.rate_groups;
const linear = exl3.linear;
const truth = exl3.f32_truth;

const relRms = truth.relRms;
const toF32 = truth.toF32;
const evalWords = truth.evalWords;
const evalHalves = truth.evalHalves;
const activations = truth.activations;
const matvec = truth.matvec;
const reshaped = truth.reshaped;
const routedErr = truth.routedErr;
const Dense = truth.Dense;
const PAIR1: format.Decode = .{ .codebook = .pair1, .window = .w16 };

/// Rows the chain is held at: one decode row, DSpark's verify block, a decode-width batch, and a prefill chunk.
const ROW_COUNTS = [_]usize{ 1, 5, 16, 33 };

fn packDir() ?[:0]const u8 {
    const raw = std.c.getenv("DSV41_RT_PACK") orelse return null;
    return std.mem.span(raw);
}

const Shard = struct {
    map: mlx.mlx_map_string_to_array,

    /// The safetensors load runs on the CPU stream: a GPU stream has no read for it.
    fn open(alloc: std.mem.Allocator, dir: []const u8, kind: []const u8, layer: usize) !Shard {
        const path = try std.fmt.allocPrintSentinel(alloc, "{s}/model-{s}-{d:0>2}.safetensors", .{ dir, kind, layer }, 0);
        defer alloc.free(path);
        var map = mlx.mlx_map_string_to_array_new();
        errdefer _ = mlx.mlx_map_string_to_array_free(map);
        var meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try mlx.check(mlx.mlx_load_safetensors(&map, &meta, path.ptr, mlx.mlx_default_cpu_stream_new()));
        return .{ .map = map };
    }

    fn deinit(self: *Shard) void {
        _ = mlx.mlx_map_string_to_array_free(self.map);
    }

    fn has(self: *const Shard, alloc: std.mem.Allocator, name: []const u8) !bool {
        const it = mlx.mlx_map_string_to_array_iterator_new(self.map);
        defer _ = mlx.mlx_map_string_to_array_iterator_free(it);
        while (true) {
            var key: ?[*:0]const u8 = null;
            var value = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(value);
            if (mlx.mlx_map_string_to_array_iterator_next(&key, &value, it) != 0 or key == null) return false;
            if (std.mem.eql(u8, std.mem.span(key.?), name)) return true;
        }
        _ = alloc;
    }

    fn get(self: *const Shard, alloc: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !mlx.mlx_array {
        const name = try std.fmt.allocPrintSentinel(alloc, fmt, args, 0);
        defer alloc.free(name);
        var a = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(a);
        try mlx.check(mlx.mlx_map_string_to_array_get(&a, self.map, name.ptr));
        return a;
    }
};


/// One stored trunk linear `<base>` of a trunk shard through the served arms at every row count, against the dense
/// twin of its own trellis: the worst relative RMS.
fn linearErr(alloc: std.mem.Allocator, s: mlx.mlx_stream, shard: *const Shard, base: []const u8, rnd: std.Random) !f64 {
    const dec = format.Decode.tabled(.spec1, .w15);
    kernels.setDecodeParams(dec);
    const tr0 = try shard.get(alloc, "{s}.trellis", .{base});
    defer _ = mlx.mlx_array_free(tr0);
    const su0 = try shard.get(alloc, "{s}.suh", .{base});
    defer _ = mlx.mlx_array_free(su0);
    const sv0 = try shard.get(alloc, "{s}.svh", .{base});
    defer _ = mlx.mlx_array_free(sv0);
    const sh = mlx.getShape(tr0);
    const groups: usize = if (sh.len == 4) @intCast(sh[0]) else 1;
    const in: usize = @intCast(sh[sh.len - 3] * 16);
    const out: usize = @intCast(sh[sh.len - 2] * 16);
    const n: u32 = @intCast(sh[sh.len - 1]);
    const g: c_int = @intCast(groups);
    const tr = try reshaped(s, tr0, &.{ g, @intCast(in / 16), @intCast(out / 16), @intCast(n) });
    defer _ = mlx.mlx_array_free(tr);
    const su = try reshaped(s, su0, &.{ g, @intCast(in) });
    defer _ = mlx.mlx_array_free(su);
    const sv = try reshaped(s, sv0, &.{ g, @intCast(out) });
    defer _ = mlx.mlx_array_free(sv);
    const proj: exl3.Proj = .{ .trellis = tr, .suh = su, .svh = sv };
    const words = try evalWords(tr);
    const suh = try evalHalves(su);
    const svh = try evalHalves(sv);
    const per = (in / 16) * (out / 16) * n;
    const w = try alloc.alloc([]f32, groups);
    for (w, 0..) |*wg, gi| {
        wg.* = try alloc.alloc(f32, in * out);
        try linear.decodeDense(alloc, words[gi * per ..][0..per], suh[gi * in ..][0..in], svh[gi * out ..][0..out], in, out, n, dec, wg.*);
    }
    var worst: f64 = 0;
    for ([_]usize{ 1, 5, 8, 40 }) |rows| {
        if (groups > 1 and rows > 8) continue;
        const act = try activations(alloc, rnd, groups * rows, in, 1);
        defer _ = mlx.mlx_array_free(act.x);
        const x3 = try reshaped(s, act.x, &.{ g, @intCast(rows), @intCast(in) });
        defer _ = mlx.mlx_array_free(x3);
        const y = if (groups == 1) try linear.linear(s, act.x, proj, dec) else try linear.linearGrouped(s, x3, proj, dec);
        defer _ = mlx.mlx_array_free(y);
        var cy = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cy);
        const got = try toF32(s, y, &cy);
        const want = try alloc.alloc(f64, got.len);
        for (0..groups) |gi| for (0..rows) |r| {
            matvec(w[gi], act.vals[(gi * rows + r) * in ..][0..in], out, want[(gi * rows + r) * out ..][0..out]);
        };
        for (got) |v| try std.testing.expect(std.math.isFinite(v));
        worst = @max(worst, relRms(got, want));
    }
    return worst;
}

test "dsv41 real: the trunk linears of converted layers 0 and 2 hold the f32-truth bar on the served arms (DSV41_RT_PACK)" {
    const dir = packDir() orelse return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x7e41);
    const rnd = prng.random();
    const names = [_][]const u8{
        "attn.wq_a",             "attn.wkv",              "attn.wq_b",             "attn.wo_a",           "attn.wo_b",
        "ffn.shared_experts.w1", "ffn.shared_experts.w3", "ffn.shared_experts.w2", "attn.compressor.wkv", "attn.compressor.wgate",
        "attn.indexer.wq_b",     "attn.indexer.wk",
    };
    for ([_]usize{ 0, 2 }) |layer| {
        var shard = try Shard.open(alloc, dir, "trunk", layer);
        defer shard.deinit();
        for (names) |name| {
            const base = try std.fmt.allocPrint(alloc, "layers.{d}.{s}", .{ layer, name });
            if (!try shard.has(alloc, try std.fmt.allocPrint(alloc, "{s}.trellis", .{base}))) continue;
            const err = try linearErr(alloc, s, &shard, base, rnd);
            if (err >= 0.002) std.debug.print("dsv41 real {s}: relRms {d}\n", .{ base, err });
            try std.testing.expect(err < 0.002);
        }
    }
}

/// One stored linear of a trunk shard: its bank and the dense f32 twin of its own trellis.
const RealLin = struct { p: exl3.Proj, w: []f32, in: usize, out: usize };

fn realLin(alloc: std.mem.Allocator, s: mlx.mlx_stream, shard: *const Shard, base: []const u8) !RealLin {
    const dec = format.Decode.tabled(.spec1, .w15);
    const tr0 = try shard.get(alloc, "{s}.trellis", .{base});
    defer _ = mlx.mlx_array_free(tr0);
    const su0 = try shard.get(alloc, "{s}.suh", .{base});
    defer _ = mlx.mlx_array_free(su0);
    const sv0 = try shard.get(alloc, "{s}.svh", .{base});
    defer _ = mlx.mlx_array_free(sv0);
    const sh = mlx.getShape(tr0);
    const in: usize = @intCast(sh[0] * 16);
    const out: usize = @intCast(sh[1] * 16);
    const n: u32 = @intCast(sh[2]);
    const p: exl3.Proj = .{
        .trellis = try reshaped(s, tr0, &.{ 1, @intCast(in / 16), @intCast(out / 16), @intCast(n) }),
        .suh = try reshaped(s, su0, &.{ 1, @intCast(in) }),
        .svh = try reshaped(s, sv0, &.{ 1, @intCast(out) }),
    };
    const w = try alloc.alloc(f32, in * out);
    try linear.decodeDense(alloc, try evalWords(p.trellis), try evalHalves(p.suh), try evalHalves(p.svh), in, out, n, dec, w);
    return .{ .p = p, .w = w, .in = in, .out = out };
}

test "dsv41 real: the attention input's and the compressor's linears in one stacked launch hold the f32-truth bar (DSV41_RT_PACK)" {
    const dir = packDir() orelse return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x5ab1);
    const rnd = prng.random();
    const dec = format.Decode.tabled(.spec1, .w15);
    var shard = try Shard.open(alloc, dir, "trunk", 2);
    defer shard.deinit();
    for ([_][]const []const u8{ &.{ "attn.wq_a", "attn.wkv" }, &.{ "attn.compressor.wkv", "attn.compressor.wgate" } }) |names| {
        var lins: [linear.SHARED_MAX]RealLin = undefined;
        var cols: [linear.SHARED_MAX + 1]c_int = @splat(0);
        for (names, 0..) |name, i| {
            lins[i] = try realLin(alloc, s, &shard, try std.fmt.allocPrint(alloc, "layers.2.{s}", .{name}));
            cols[i + 1] = cols[i] + @as(c_int, @intCast(lins[i].out));
        }
        defer for (lins[0..names.len]) |l| for ([_]mlx.mlx_array{ l.p.trellis, l.p.suh, l.p.svh }) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var cat: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (cat) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for (0..3) |k| {
            var parts: [linear.SHARED_MAX]mlx.mlx_array = undefined;
            for (lins[0..names.len], 0..) |l, i| parts[i] = if (k == 0) l.p.trellis else if (k == 1) l.p.suh else l.p.svh;
            const v = mlx.mlx_vector_array_new_data(&parts, names.len);
            defer _ = mlx.mlx_vector_array_free(v);
            try mlx.check(mlx.mlx_concatenate_axis(&cat[k], v, @as(c_int, if (k == 0) 2 else if (k == 1) 0 else 1), s));
        }
        const sh: linear.Shared = .{ .p = .{ .trellis = cat[0], .suh = cat[1], .svh = cat[2] }, .cols = cols, .count = names.len };
        for ([_]usize{ 1, 5, 8 }) |rows| {
            const act = try activations(alloc, rnd, rows, lins[0].in, 1);
            defer _ = mlx.mlx_array_free(act.x);
            var outs: [linear.SHARED_MAX]mlx.mlx_array = undefined;
            try linear.linearShared(s, act.x, sh, dec, outs[0..names.len]);
            defer for (outs[0..names.len]) |o| {
                _ = mlx.mlx_array_free(o);
            };
            for (lins[0..names.len], outs[0..names.len], names) |l, o, name| {
                var cy = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(cy);
                const got = try toF32(s, o, &cy);
                const want = try alloc.alloc(f64, got.len);
                for (0..rows) |r| matvec(l.w, act.vals[r * l.in ..][0..l.in], l.out, want[r * l.out ..][0..l.out]);
                for (got) |v| try std.testing.expect(std.math.isFinite(v));
                const err = relRms(got, want);
                if (err >= 0.002) std.debug.print("dsv41 real stacked {s} rows {d}: relRms {d}\n", .{ name, rows, err });
                try std.testing.expect(err < 0.002);
            }
        }
    }
}

test "dsv41 real: two experts of every rate group of converted layer 2 hold the f32-truth bar (DSV41_RT_PACK)" {
    const dir = packDir() orelse return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    kernels.setDecodeParams(PAIR1);
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x7e42);
    const rnd = prng.random();
    const layer: usize = 2;
    const hidden: usize = 5120;
    const inter: usize = 2304;
    var shard = try Shard.open(alloc, dir, "layer", layer);
    defer shard.deinit();
    var banks: [8]exl3.Bank = undefined;
    var dense: [16]Dense = undefined;
    var count: usize = 0;
    var ndense: usize = 0;
    defer for (banks[0..count]) |*b| {
        for ([_]*exl3.Proj{ &b.gate, &b.up, &b.down }) |p| {
            _ = mlx.mlx_array_free(p.trellis);
            _ = mlx.mlx_array_free(p.suh);
            _ = mlx.mlx_array_free(p.svh);
        }
    };
    var gi: usize = 0;
    while (try shard.has(alloc, try std.fmt.allocPrint(alloc, "model.layers.{d}.mlp.switch_mlp.gate_proj.g{d}.trellis", .{ layer, gi }))) : (gi += 1) {
        var n3: [3]u32 = undefined;
        var bank: exl3.Bank = undefined;
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, pi| {
            const p = switch (pi) {
                0 => &bank.gate,
                1 => &bank.up,
                else => &bank.down,
            };
            const tr = try shard.get(alloc, "model.layers.{d}.mlp.switch_mlp.{s}.g{d}.trellis", .{ layer, proj, gi });
            const su = try shard.get(alloc, "model.layers.{d}.mlp.switch_mlp.{s}.g{d}.suh", .{ layer, proj, gi });
            const sv = try shard.get(alloc, "model.layers.{d}.mlp.switch_mlp.{s}.g{d}.svh", .{ layer, proj, gi });
            n3[pi] = @intCast(mlx.getShape(tr)[3]);
            // The first two experts of the group, as an array of their own.
            const view = struct {
                fn of(sm: mlx.mlx_stream, a: mlx.mlx_array) !mlx.mlx_array {
                    const sh = mlx.getShape(a);
                    var start: [4]c_int = @splat(0);
                    var stop: [4]c_int = @splat(0);
                    const strides: [4]c_int = @splat(1);
                    @memcpy(stop[0..sh.len], sh);
                    stop[0] = @min(2, sh[0]);
                    var o = mlx.mlx_array_new();
                    errdefer _ = mlx.mlx_array_free(o);
                    try mlx.check(mlx.mlx_slice(&o, a, &start, sh.len, &stop, sh.len, &strides, sh.len, sm));
                    return o;
                }
            }.of;
            p.* = .{ .trellis = try view(s, tr), .suh = try view(s, su), .svh = try view(s, sv) };
            for ([_]mlx.mlx_array{ tr, su, sv }) |a| _ = mlx.mlx_array_free(a);
        }
        banks[count] = bank;
        count += 1;
        const e: usize = @intCast(mlx.getShape(bank.gate.trellis)[0]);
        for (0..e) |l| {
            dense[ndense] = try Dense.of(alloc, bank, l, hidden, inter, n3, PAIR1);
            ndense += 1;
        }
    }
    try std.testing.expect(count >= 2);
    var rb = try rate_groups.Ragged.init(s, banks[0..count]);
    defer rb.deinit();
    for (ROW_COUNTS) |rows| {
        const err = try routedErr(alloc, s, &rb, dense[0..ndense], rnd, hidden, inter, rows, 4, 1);
        if (err >= 0.005) std.debug.print("dsv41 real experts rows {d}: relRms {d}\n", .{ rows, err });
        try std.testing.expect(err < 0.005);
    }
}
