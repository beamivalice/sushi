const std = @import("std");
const mlx = @import("mlx_host").mlx;
const convert = @import("convert.zig");
const source = @import("source.zig");

pub const Error = error{ HessianBadShape, HessianExpertOutOfRange } || std.mem.Allocator.Error;

pub const RowsHessians = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    src: *const source.SafetensorsSource,
    layer: u32,
    mtp: bool,
    geom: convert.Geometry,
    x: mlx.mlx_array,
    starts: []usize,
    order: []i32,

    pub fn init(
        allocator: std.mem.Allocator,
        s: mlx.mlx_stream,
        src: *const source.SafetensorsSource,
        layer: u32,
        mtp: bool,
        geom: convert.Geometry,
        x: mlx.mlx_array,
        ids: []const i32,
        topk: usize,
    ) !RowsHessians {
        const xs = mlx.getShape(x);
        if (xs.len != 2 or xs[1] != geom.hidden or topk == 0 or ids.len != @as(usize, @intCast(xs[0])) * topk) return error.HessianBadShape;
        const starts = try allocator.alloc(usize, geom.experts + 1);
        errdefer allocator.free(starts);
        @memset(starts, 0);
        for (ids) |id| {
            if (id < 0 or @as(usize, @intCast(id)) >= geom.experts) return error.HessianExpertOutOfRange;
            starts[@as(usize, @intCast(id)) + 1] += 1;
        }
        for (1..starts.len) |e| starts[e] += starts[e - 1];
        const order = try allocator.alloc(i32, ids.len);
        errdefer allocator.free(order);
        const cursor = try allocator.dupe(usize, starts[0..geom.experts]);
        defer allocator.free(cursor);
        for (ids, 0..) |id, i| {
            const e: usize = @intCast(id);
            order[cursor[e]] = @intCast(i / topk);
            cursor[e] += 1;
        }
        return .{ .allocator = allocator, .s = s, .src = src, .layer = layer, .mtp = mtp, .geom = geom, .x = x, .starts = starts, .order = order };
    }

    pub fn deinit(self: *RowsHessians) void {
        self.allocator.free(self.starts);
        self.allocator.free(self.order);
    }

    pub fn hessianSource(self: *RowsHessians) convert.HessianSource {
        return .{ .ctx = self, .gate_up = gateUp, .down = down };
    }

    fn routedRows(self: *RowsHessians, expert: usize) !mlx.mlx_array {
        if (expert >= self.geom.experts) return error.HessianExpertOutOfRange;
        const lo = self.starts[expert];
        const hi = self.starts[expert + 1];
        const idx = mlx.mlx_array_new_data(self.order[lo..hi].ptr, &[_]c_int{@intCast(hi - lo)}, 1, .int32);
        defer _ = mlx.mlx_array_free(idx);
        var rows = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_take_axis(&rows, self.x, idx, 0, self.s));
        return rows;
    }

    fn gram(self: *RowsHessians, a: mlx.mlx_array, out: []f32) !void {
        var at = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(at);
        try mlx.check(mlx.mlx_transpose(&at, a, self.s));
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_matmul(&g, at, a, self.s));
        try mlx.check(mlx.mlx_array_eval(g));
        const data = mlx.mlx_array_data_float32(g) orelse return error.HessianBadShape;
        @memcpy(out, data[0..out.len]);
    }

    fn gateUp(ctx: *anyopaque, expert: usize, out: []f32) anyerror!u64 {
        const self: *RowsHessians = @ptrCast(@alignCast(ctx));
        if (out.len != self.geom.hidden * self.geom.hidden) return error.HessianBadShape;
        const m = self.starts[expert + 1] - self.starts[expert];
        if (m == 0) return 0;
        const rows = try self.routedRows(expert);
        defer _ = mlx.mlx_array_free(rows);
        try self.gram(rows, out);
        return m;
    }

    fn projected(self: *RowsHessians, rows: mlx.mlx_array, expert: usize, role: source.ProjRole) !mlx.mlx_array {
        const h = self.geom.hidden;
        const i = self.geom.inter;
        const w = try self.allocator.alloc(f32, h * i);
        defer self.allocator.free(w);
        try self.src.readExpertPublic(self.allocator, .qwen4_exp, self.layer, self.mtp, role, expert, h, i, w);
        const wa = mlx.mlx_array_new_data(w.ptr, &[_]c_int{ @intCast(h), @intCast(i) }, 2, .float32);
        defer _ = mlx.mlx_array_free(wa);
        var y = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_matmul(&y, rows, wa, self.s));
        try mlx.check(mlx.mlx_array_eval(y));
        return y;
    }

    fn down(ctx: *anyopaque, expert: usize, out: []f32) anyerror!u64 {
        const self: *RowsHessians = @ptrCast(@alignCast(ctx));
        if (out.len != self.geom.inter * self.geom.inter) return error.HessianBadShape;
        const m = self.starts[expert + 1] - self.starts[expert];
        if (m == 0) return 0;
        const rows = try self.routedRows(expert);
        defer _ = mlx.mlx_array_free(rows);
        const g = try self.projected(rows, expert, .gate);
        defer _ = mlx.mlx_array_free(g);
        const u = try self.projected(rows, expert, .up);
        defer _ = mlx.mlx_array_free(u);
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, g, self.s));
        var silu = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(silu);
        try mlx.check(mlx.mlx_multiply(&silu, g, sig, self.s));
        var act = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act);
        try mlx.check(mlx.mlx_multiply(&act, silu, u, self.s));
        try self.gram(act, out);
        return m;
    }
};

fn expectGramClose(want: []const f64, got: []const f32, n: usize) !void {
    for (0..n) |i| for (0..n) |j| {
        const scale = @sqrt(want[i * n + i] * want[j * n + j]);
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want[i * n + j])), got[i * n + j], 4e-3 * @as(f32, @floatCast(scale)) + 1e-6);
    };
}

test "hessians: per-expert gate/up and down Hessians match a naive sum over the routed rows" {
    const t = std.testing;
    const io = t.io;
    const E = convert.tiny.experts;
    const H = convert.tiny.hidden;
    const I = convert.tiny.inter;
    const topk = 2;
    const n_rows = 150;
    var prng = std.Random.DefaultPrng.init(91);
    const r = prng.random();

    var src_dir = t.tmpDir(.{});
    defer src_dir.cleanup();
    try convert.tiny.writeSource(t.allocator, io, src_dir.dir, &prng, "model.language_model.layers");
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir.dir);
    defer src.deinit();

    const x = try t.allocator.alloc(f32, n_rows * H);
    defer t.allocator.free(x);
    for (x) |*v| v.* = r.floatNorm(f32);
    var ids: [n_rows * topk]i32 = undefined;
    for (0..n_rows) |row| {
        const a = r.intRangeLessThan(i32, 0, E - 1);
        ids[row * topk] = a;
        ids[row * topk + 1] = a + 1;
    }
    const xa = mlx.mlx_array_new_data(x.ptr, &[_]c_int{ n_rows, H }, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);

    var rows = try RowsHessians.init(t.allocator, mlx.gpuStream(), &src, 0, false, .{ .experts = E, .hidden = H, .inter = I }, xa, &ids, topk);
    defer rows.deinit();
    const hs = rows.hessianSource();

    const wg = try t.allocator.alloc(f32, H * I);
    defer t.allocator.free(wg);
    const wu = try t.allocator.alloc(f32, H * I);
    defer t.allocator.free(wu);
    const got_gu = try t.allocator.alloc(f32, H * H);
    defer t.allocator.free(got_gu);
    const got_dn = try t.allocator.alloc(f32, I * I);
    defer t.allocator.free(got_dn);
    for (0..E) |e| {
        var want_gu = try t.allocator.alloc(f64, H * H);
        defer t.allocator.free(want_gu);
        var want_dn = try t.allocator.alloc(f64, I * I);
        defer t.allocator.free(want_dn);
        @memset(want_gu, 0);
        @memset(want_dn, 0);
        try src.readExpertPublic(t.allocator, .qwen4_exp, 0, false, .gate, e, H, I, wg);
        try src.readExpertPublic(t.allocator, .qwen4_exp, 0, false, .up, e, H, I, wu);
        var count: u64 = 0;
        for (0..n_rows) |row| {
            var hit = false;
            for (0..topk) |j| if (ids[row * topk + j] == @as(i32, @intCast(e))) {
                hit = true;
            };
            if (!hit) continue;
            count += 1;
            const xr = x[row * H ..][0..H];
            for (0..H) |i| for (0..H) |j| {
                want_gu[i * H + j] += @as(f64, xr[i]) * xr[j];
            };
            var act: [I]f64 = undefined;
            for (0..I) |j| {
                var g: f64 = 0;
                var u: f64 = 0;
                for (0..H) |h| {
                    g += @as(f64, xr[h]) * wg[h * I + j];
                    u += @as(f64, xr[h]) * wu[h * I + j];
                }
                act[j] = g / (1.0 + @exp(-g)) * u;
            }
            for (0..I) |i| for (0..I) |j| {
                want_dn[i * I + j] += act[i] * act[j];
            };
        }
        try t.expectEqual(count, try hs.gate_up(hs.ctx, e, got_gu));
        try t.expectEqual(count, try hs.down(hs.ctx, e, got_dn));
        try expectGramClose(want_gu, got_gu, H);
        try expectGramClose(want_dn, got_dn, I);
    }
}

test "hessians: a down Hessian at Flash-Next geometry over a ragged row count is finite and matches a naive sum" {
    const t = std.testing;
    const io = t.io;
    const E = 2;
    const H = 2560;
    const I = 640;
    const topk = 2;
    const n_rows = 292;
    var prng = std.Random.DefaultPrng.init(92);
    const r = prng.random();

    var src_dir = t.tmpDir(.{});
    defer src_dir.cleanup();
    try convert.tiny.writeSourceDims(t.allocator, io, src_dir.dir, &prng, "model.language_model.layers", E, H, I);
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir.dir);
    defer src.deinit();

    const x = try t.allocator.alloc(f32, n_rows * H);
    defer t.allocator.free(x);
    for (x) |*v| v.* = r.floatNorm(f32);
    var ids: [n_rows * topk]i32 = undefined;
    for (0..n_rows) |row| {
        ids[row * topk] = 0;
        ids[row * topk + 1] = 1;
    }
    const xa = mlx.mlx_array_new_data(x.ptr, &[_]c_int{ n_rows, H }, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);
    var rows = try RowsHessians.init(t.allocator, mlx.gpuStream(), &src, 0, false, .{ .experts = E, .hidden = H, .inter = I }, xa, &ids, topk);
    defer rows.deinit();
    const hs = rows.hessianSource();

    const wg = try t.allocator.alloc(f32, H * I);
    defer t.allocator.free(wg);
    const wu = try t.allocator.alloc(f32, H * I);
    defer t.allocator.free(wu);
    try src.readExpertPublic(t.allocator, .qwen4_exp, 0, false, .gate, 0, H, I, wg);
    try src.readExpertPublic(t.allocator, .qwen4_exp, 0, false, .up, 0, H, I, wu);
    const want = try t.allocator.alloc(f64, I * I);
    defer t.allocator.free(want);
    @memset(want, 0);
    for (0..n_rows) |row| {
        const xr = x[row * H ..][0..H];
        var act: [I]f64 = undefined;
        for (0..I) |j| {
            var g: f64 = 0;
            var u: f64 = 0;
            for (0..H) |h| {
                g += @as(f64, xr[h]) * wg[h * I + j];
                u += @as(f64, xr[h]) * wu[h * I + j];
            }
            act[j] = g / (1.0 + @exp(-g)) * u;
        }
        for (0..I) |i| for (0..I) |j| {
            want[i * I + j] += act[i] * act[j];
        };
    }
    const got = try t.allocator.alloc(f32, I * I);
    defer t.allocator.free(got);
    try t.expectEqual(@as(u64, n_rows), try hs.down(hs.ctx, 0, got));
    for (got) |v| try t.expect(std.math.isFinite(v));
    try expectGramClose(want, got, I);
}
