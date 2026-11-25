const std = @import("std");
const mlx = @import("mlx_host").mlx;
const convert = @import("convert.zig");
const source = @import("source.zig");
const stages_mod = @import("stages.zig");

pub const Error = error{ HessianBadShape, HessianExpertOutOfRange } || std.mem.Allocator.Error;

pub const RowsHessians = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    src: *const source.SafetensorsSource,
    layer: u32,
    mtp: bool,
    geom: convert.Geometry,
    x: mlx.mlx_array,
    /// The same tokens' rows on the BF16 path, for the compensated target (`target.zig`); routing stays `x`'s.
    xf: ?mlx.mlx_array = null,
    /// The down projections' own rows and routing, when they differ from gate/up's.
    down_rows: ?*RowsHessians = null,
    starts: []usize,
    order: []i32,
    /// The checkpoint the source experts come from, and its SwiGLU clamp (0 = none): gate capped at the limit,
    /// up clipped to +-limit, as DeepSeek-V4.1 runs them.
    arch: source.Arch = .qwen4_exp,
    limit: f32 = 0,

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
        if (self.xf != null) return .{ .ctx = self, .gate_up = gateUp, .down = down, .gate_up_target = gateUpTarget, .down_target = downTarget };
        return .{ .ctx = self, .gate_up = gateUp, .down = down };
    }

    fn routedRows(self: *RowsHessians, expert: usize) !mlx.mlx_array {
        return self.routedRowsOf(self.x, expert);
    }

    fn routedRowsOf(self: *RowsHessians, from: mlx.mlx_array, expert: usize) !mlx.mlx_array {
        if (expert >= self.geom.experts) return error.HessianExpertOutOfRange;
        const lo = self.starts[expert];
        const hi = self.starts[expert + 1];
        const idx = mlx.mlx_array_new_data(self.order[lo..hi].ptr, &[_]c_int{@intCast(hi - lo)}, 1, .int32);
        defer _ = mlx.mlx_array_free(idx);
        var rows = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_take_axis(&rows, from, idx, 0, self.s));
        if (mlx.mlx_array_dtype(rows) == .float32) return rows;
        var wide = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(wide);
        try mlx.check(mlx.mlx_astype(&wide, rows, .float32, self.s));
        _ = mlx.mlx_array_free(rows);
        return wide;
    }

    /// a' b into `out`.
    fn crossInto(self: *RowsHessians, a: mlx.mlx_array, b: mlx.mlx_array, out: []f32) !void {
        var at = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(at);
        try mlx.check(mlx.mlx_transpose(&at, a, self.s));
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_matmul(&g, at, b, self.s));
        try evalWaited(g);
        const data = mlx.mlx_array_data_float32(g) orelse return error.HessianBadShape;
        @memcpy(out, data[0..out.len]);
    }

    fn matmulHost(self: *RowsHessians, rows: mlx.mlx_array, w: []const f32, in: usize, out: usize) !mlx.mlx_array {
        const wa = mlx.mlx_array_new_data(w.ptr, &[_]c_int{ @intCast(in), @intCast(out) }, 2, .float32);
        defer _ = mlx.mlx_array_free(wa);
        var y = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_matmul(&y, rows, wa, self.s));
        return y;
    }

    fn swiglu(self: *RowsHessians, g_in: mlx.mlx_array, u_in: mlx.mlx_array) !mlx.mlx_array {
        var g = g_in;
        var u = u_in;
        var capped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(capped);
        var clipped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(clipped);
        if (self.limit > 0) {
            const hi = mlx.mlx_array_new_float(self.limit);
            defer _ = mlx.mlx_array_free(hi);
            const lo = mlx.mlx_array_new_float(-self.limit);
            defer _ = mlx.mlx_array_free(lo);
            try mlx.check(mlx.mlx_minimum(&capped, g, hi, self.s));
            try mlx.check(mlx.mlx_clip(&clipped, u, lo, hi, self.s));
            g = capped;
            u = clipped;
        }
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, g, self.s));
        var silu = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(silu);
        try mlx.check(mlx.mlx_multiply(&silu, g, sig, self.s));
        var act = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(act);
        try mlx.check(mlx.mlx_multiply(&act, silu, u, self.s));
        return act;
    }

    /// Gate/up: X~'X~ into `h`, and X~'X [Wg | Wu] into `cw` ([hidden, 2 inter], each row gate then up).
    fn gateUpTarget(ctx: *anyopaque, expert: usize, h: []f32, cw: []f32) anyerror!u64 {
        const self: *RowsHessians = @ptrCast(@alignCast(ctx));
        const hid = self.geom.hidden;
        const i = self.geom.inter;
        if (h.len != hid * hid or cw.len != hid * 2 * i) return error.HessianBadShape;
        const m = self.starts[expert + 1] - self.starts[expert];
        if (m == 0) return 0;
        const xq = try self.routedRows(expert);
        defer _ = mlx.mlx_array_free(xq);
        const xf = try self.routedRowsOf(self.xf.?, expert);
        defer _ = mlx.mlx_array_free(xf);
        try self.gram(xq, h);
        const w = try self.allocator.alloc(f32, hid * 2 * i);
        defer self.allocator.free(w);
        const part = try self.allocator.alloc(f32, hid * i);
        defer self.allocator.free(part);
        for ([_]source.ProjRole{ .gate, .up }, 0..) |role, side| {
            try self.src.readExpertPublic(self.allocator, self.arch, self.layer, self.mtp, role, expert, hid, i, part);
            for (0..hid) |r| @memcpy(w[r * 2 * i + side * i ..][0..i], part[r * i ..][0..i]);
        }
        const p = try self.matmulHost(xf, w, hid, 2 * i);
        defer _ = mlx.mlx_array_free(p);
        try self.crossInto(xq, p, cw);
        return m;
    }

    /// Down: over the SwiGLU activations, X~ through this expert's quantized gate/up (`gate_q`, `up_q`, public
    /// [hidden, inter]) and X through the source's; X~'X~ into `h`, X~'X Wd into `cw` ([inter, hidden]).
    fn downOf(ctx: *anyopaque) *RowsHessians {
        const self: *RowsHessians = @ptrCast(@alignCast(ctx));
        return self.down_rows orelse self;
    }

    fn downTarget(ctx: *anyopaque, expert: usize, gate_q: []const f32, up_q: []const f32, h: []f32, cw: []f32) anyerror!u64 {
        const self = downOf(ctx);
        const hid = self.geom.hidden;
        const i = self.geom.inter;
        if (h.len != i * i or cw.len != i * hid or gate_q.len != hid * i or up_q.len != hid * i) return error.HessianBadShape;
        const m = self.starts[expert + 1] - self.starts[expert];
        if (m == 0) return 0;
        const xq = try self.routedRows(expert);
        defer _ = mlx.mlx_array_free(xq);
        const xf = try self.routedRowsOf(self.xf.?, expert);
        defer _ = mlx.mlx_array_free(xf);
        const gq = try self.matmulHost(xq, gate_q, hid, i);
        defer _ = mlx.mlx_array_free(gq);
        const uq = try self.matmulHost(xq, up_q, hid, i);
        defer _ = mlx.mlx_array_free(uq);
        const act_q = try self.swiglu(gq, uq);
        defer _ = mlx.mlx_array_free(act_q);
        const gf = try self.projected(xf, expert, .gate);
        defer _ = mlx.mlx_array_free(gf);
        const uf = try self.projected(xf, expert, .up);
        defer _ = mlx.mlx_array_free(uf);
        const act_f = try self.swiglu(gf, uf);
        defer _ = mlx.mlx_array_free(act_f);
        try self.gram(act_q, h);
        const wd = try self.allocator.alloc(f32, i * hid);
        defer self.allocator.free(wd);
        try self.src.readExpertPublic(self.allocator, self.arch, self.layer, self.mtp, .down, expert, i, hid, wd);
        const p = try self.matmulHost(act_f, wd, i, hid);
        defer _ = mlx.mlx_array_free(p);
        try self.crossInto(act_q, p, cw);
        return m;
    }

    fn gram(self: *RowsHessians, a: mlx.mlx_array, out: []f32) !void {
        var at = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(at);
        try mlx.check(mlx.mlx_transpose(&at, a, self.s));
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_matmul(&g, at, a, self.s));
        try evalWaited(g);
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
        try self.src.readExpertPublic(self.allocator, self.arch, self.layer, self.mtp, role, expert, h, i, w);
        const wa = mlx.mlx_array_new_data(w.ptr, &[_]c_int{ @intCast(h), @intCast(i) }, 2, .float32);
        defer _ = mlx.mlx_array_free(wa);
        var y = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_matmul(&y, rows, wa, self.s));
        try evalWaited(y);
        return y;
    }

    fn down(ctx: *anyopaque, expert: usize, out: []f32) anyerror!u64 {
        const self = downOf(ctx);
        if (out.len != self.geom.inter * self.geom.inter) return error.HessianBadShape;
        const m = self.starts[expert + 1] - self.starts[expert];
        if (m == 0) return 0;
        const rows = try self.routedRows(expert);
        defer _ = mlx.mlx_array_free(rows);
        const g = try self.projected(rows, expert, .gate);
        defer _ = mlx.mlx_array_free(g);
        const u = try self.projected(rows, expert, .up);
        defer _ = mlx.mlx_array_free(u);
        const act = try self.swiglu(g, u);
        defer _ = mlx.mlx_array_free(act);
        try self.gram(act, out);
        return m;
    }
};

fn evalWaited(a: mlx.mlx_array) !void {
    const t0 = stages_mod.nowNs();
    defer stages_mod.waitedSince(t0);
    try mlx.check(mlx.mlx_array_eval(a));
}

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
