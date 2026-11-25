//! PAIR1's Metal pair search: sashimi's v2 kernel (`sashimi/tinyquant/pair_search.py`, `_source_v2`), written per
//! rate. A threadgroup searches `tiles` tiles at once; each step keeps half costs for both of a tile's rows in
//! threadgroup memory and every step's half costs in device scratch, so the traceback re-scores the traced state's
//! candidates (lowest candidate at the strict minimum) instead of storing back-pointers. Its codewords are
//! `pair_search.Search.run`'s, byte for byte.
const std = @import("std");
const format = @import("sushi_exl3").format;
const pair_search = @import("pair_search.zig");

pub const Plan = struct {
    /// Tiles per threadgroup.
    tiles: u32,
    threads: u32,
    /// Candidate rows read from the half2 decode table; the rest hash their codeword.
    table_rows: u32,
};

/// Per-thread state limits a plan, so the wide-state low rates run one tile on many threads and hash, the high rates
/// several tiles on few threads and read the table. sashimi's per-rate plans, with n32, n36 and n38 re-timed in the
/// Zig launcher (`SUSHI_ENCODER_PAIR1_UBENCH`).
pub fn planFor(n: u32) Plan {
    return switch (n) {
        0...28 => .{ .tiles = 1, .threads = 1024, .table_rows = 0 },
        30 => .{ .tiles = 1, .threads = 512, .table_rows = 0 },
        32 => .{ .tiles = 2, .threads = 256, .table_rows = 256 },
        34 => .{ .tiles = 2, .threads = 512, .table_rows = 256 },
        36, 38 => .{ .tiles = 2, .threads = 128, .table_rows = 256 },
        40...46 => .{ .tiles = 4, .threads = 256, .table_rows = 256 },
        48...62 => .{ .tiles = 4, .threads = 128, .table_rows = 256 },
        else => .{ .tiles = 2, .threads = 128, .table_rows = 256 },
    };
}

const tg_bytes = 32768;

pub const Shape = struct {
    ks: [8]u8,
    /// Cost slots per tile row: the most states any step has.
    sm: u32,
    /// The pins live in device scratch when the threadgroup is full of costs.
    pins_device: bool,
};

pub fn shapeOf(n: u32, plan: Plan) Shape {
    const ks = pair_search.ksteps(.{ .n = n });
    const sm = @as(u32, 1) << @intCast(16 - std.mem.min(u8, &ks));
    return .{ .ks = ks, .sm = sm, .pins_device = plan.tiles * 2 * sm * 2 + 4 * plan.tiles > tg_bytes };
}

/// The (fresh bits of this step, of the next) variants of a rate, in sashimi's order, and each phase's variant.
fn variants(ks: [8]u8, out: *[8][2]u8) struct { count: usize, of: [8]u8 } {
    var count: usize = 0;
    for (0..8) |i| {
        const v = [2]u8{ ks[i], ks[(i + 1) & 7] };
        var seen = false;
        for (out[0..count]) |o| seen = seen or std.mem.eql(u8, &o, &v);
        if (!seen) {
            out[count] = v;
            count += 1;
        }
    }
    std.mem.sort([2]u8, out[0..count], {}, struct {
        fn lt(_: void, a: [2]u8, b: [2]u8) bool {
            return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1]);
        }
    }.lt);
    var of: [8]u8 = undefined;
    for (0..8) |i| {
        for (out[0..count], 0..) |o, vi| if (o[0] == ks[i] and o[1] == ks[(i + 1) & 7]) {
            of[i] = @intCast(vi);
        };
    }
    return .{ .count = count, .of = of };
}

pub fn header(allocator: std.mem.Allocator, n: u32) ![:0]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    const ks = pair_search.ksteps(.{ .n = n });
    var vs: [8][2]u8 = undefined;
    const v = variants(ks, &vs);
    try out.appendSlice(allocator,
        \\static inline float2 pair_dec_pm(uint p, uint rot, uint m0, uint m1) {
        \\    const half4 h = as_type<half4>(uint2(p & m0, rotate(p, rot) & m1));
        \\    return float2(float(h.x + h.y), float(h.z + h.w)) * 4096.0f;
        \\}
        \\
    );
    try out.print(allocator, "constant uchar KST[8] = {{{d}, {d}, {d}, {d}, {d}, {d}, {d}, {d}}};\n", .{ ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7] });
    try out.print(allocator, "constant uchar VAR[8] = {{{d}, {d}, {d}, {d}, {d}, {d}, {d}, {d}}};\n", .{ v.of[0], v.of[1], v.of[2], v.of[3], v.of[4], v.of[5], v.of[6], v.of[7] });
    return out.toOwnedSliceSentinel(allocator, 0);
}

const Ctx = struct {
    out: *std.ArrayList(u8),
    a: std.mem.Allocator,
    plan: Plan,
    shape: Shape,
    p: format.Pair,
    pin_pass: u32,

    fn w(self: Ctx, comptime fmt: []const u8, args: anytype) !void {
        try self.out.print(self.a, fmt, args);
    }

    fn dec(self: Ctx, cw: []const u8) !void {
        try self.w("pair_dec_pm(({s}) * 0x{x}u, {d}u, 0x{x}u, 0x{x}u)", .{ cw, self.p.mult, self.p.rot, self.p.mask0, self.p.mask1 });
    }

    fn pinAt(self: Ctx, j: []const u8) !void {
        if (self.shape.pins_device) try self.w("((device uint*)temp_costs)[grp * {d}u + {s}]", .{ self.plan.tiles, j }) else try self.w("sh_pin[{s}]", .{j});
    }

    fn firstBody(self: Ctx, kc: u32, kn: u32) !void {
        const s_bits = 16 - kn;
        const nb = @as(u32, 1) << @intCast(kn);
        const gsz = (@as(u32, 1) << @intCast(s_bits)) / self.plan.threads;
        for (0..self.plan.tiles) |j| {
            try self.w(
                \\
                \\            for (uint g = 0u; g < {d}u; g++) {{
                \\                const uint e = tid * {d}u + g;
                \\                float best = INF;
                \\                {s}
                \\                for (uint a = 0u; a < {d}u; a++) {{
                \\                    const uint cw = (a << {d}u) | e;
                \\                    float err;
                \\                    if (pinned && (cw >> {d}u) != pin{d}) err = INF;
                \\                    else {{
                \\                        const float2 v =
            , .{ gsz, gsz, if (nb <= 16) "#pragma clang loop unroll(full)" else "", nb, s_bits, kc, j });
            try self.dec("cw");
            try self.w(
                \\;
                \\                        const float d0 = v.x - t0_{d}, d1 = v.y - t1_{d};
                \\                        err = fma(d1, d1, d0 * d0);
                \\                        if (err > INF) err = INF;
                \\                    }}
                \\                    if (err < best) best = err;
                \\                }}
                \\                costs[{d}][0u][e] = half(best);
                \\                if (keep) row{d}[ulong(pi) * {d}u + e] = half(best);
                \\            }}
            , .{ j, j, j, j, self.shape.sm });
        }
    }

    fn stepBody(self: Ctx, kc: u32, kn: u32) !void {
        const s_bits = 16 - kn;
        const nb = @as(u32, 1) << @intCast(kn);
        const gsz = (@as(u32, 1) << @intCast(s_bits)) / self.plan.threads;
        const gi = @min(gsz, @as(u32, 1) << @intCast(kc));
        const rows = @min(self.plan.table_rows, nb);
        const m = self.plan.tiles;
        const pa_step: u32 = (@as(u32, 1) << @intCast(s_bits)) *% self.p.mult;
        try self.w("\n            const uint ebase = tid * {d}u;\n            float best[{d}];\n            for (uint g = 0u; g < {d}u; g++) best[g] = INF;\n", .{ gsz, m * gsz, m * gsz });
        if (rows < nb) {
            try self.w("            uint pe[{d}];", .{gsz});
            for (0..gsz) |g| try self.w(" pe[{d}] = (ebase + {d}u) * 0x{x}u;", .{ g, g, self.p.mult });
            try self.w("\n", .{});
        }
        for ([_]bool{ true, false }) |table| {
            if (table and rows == 0) continue;
            if (!table and rows == nb) continue;
            if (table) {
                try self.w("            for (uint a = 0u; a < {d}u; a++) {{\n", .{rows});
            } else {
                try self.w("            uint pa = 0x{x}u;\n            {s}\n            for (uint a = {d}u; a < {d}u; a++) {{\n", .{ rows *% pa_step, if (nb - rows <= 16) "#pragma clang loop unroll(full)" else "", rows, nb });
            }
            var gb: u32 = 0;
            while (gb * gi < gsz) : (gb += 1) {
                try self.w("                ", .{});
                for (0..m) |j| try self.w("const float prev{d}_{d} = float(costs[{d}][prev_buf][(a << {d}u) | ((ebase + {d}u) >> {d}u)]); ", .{ j, gb, j, s_bits - kc, gb * gi, kc });
                try self.w("\n", .{});
                for (gb * gi..(gb + 1) * gi) |g| {
                    if (table) {
                        try self.w("                {{ const float2 v = float2(as_type<half2>(dq_tab[(a << {d}u) | (ebase + {d}u)]));", .{ s_bits, g });
                    } else {
                        try self.w("                {{ const uint p = pa + pe[{d}]; const half4 h = as_type<half4>(uint2(p & 0x{x}u, rotate(p, {d}u) & 0x{x}u)); const float s0 = float(h.x + h.y), s1 = float(h.z + h.w);", .{ g, self.p.mask0, self.p.rot, self.p.mask1 });
                    }
                    for (0..m) |j| {
                        const bi = j * gsz + g;
                        if (table) {
                            try self.w(" {{ const float d0 = v.x - t0_{d}, d1 = v.y - t1_{d};", .{ j, j });
                        } else {
                            try self.w(" {{ const float d0 = fma(s0, 4096.0f, -t0_{d}), d1 = fma(s1, 4096.0f, -t1_{d});", .{ j, j });
                        }
                        try self.w(" best[{d}] = fmin(best[{d}], fma(d1, d1, fma(d0, d0, prev{d}_{d}))); }}", .{ bi, bi, j, gb });
                    }
                    try self.w(" }}\n", .{});
                }
            }
            if (!table) try self.w("                pa += 0x{x}u;\n", .{pa_step});
            try self.w("            }}\n", .{});
        }
        for (0..m) |j| {
            try self.w("            ", .{});
            for (0..gsz) |g| try self.w("costs[{d}][curr][ebase + {d}u] = half(best[{d}]); ", .{ j, g, j * gsz + g });
            try self.w("\n            if (keep) {{ ", .{});
            for (0..gsz) |g| try self.w("row{d}[ulong(pi) * {d}u + ebase + {d}u] = half(best[{d}]); ", .{ j, self.shape.sm, g, j * gsz + g });
            try self.w("}}\n", .{});
        }
    }
};

/// The kernel body for rate `n` and pin `pin_pass` pairs: inputs `tiles` [rows, 256] f32 (trellis order) and `dq_tab`
/// (65536 half2 as u32, both weights of every codeword); outputs `q_tiles` [rows, 256] f32, `indices` [rows, 128] pair
/// codewords, `temp_edges` (each tile's per-step half costs) and `temp_costs` (the pins, when in device memory).
pub fn source(allocator: std.mem.Allocator, n: u32, plan: Plan, pin_pass: u32) ![:0]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    const p = format.pair1Pair(.{ .n = n }) orelse return error.RateUnsupported;
    const shape = shapeOf(n, plan);
    const c = Ctx{ .out = &out, .a = allocator, .plan = plan, .shape = shape, .p = p, .pin_pass = pin_pass };
    const m = plan.tiles;
    const sm = shape.sm;
    var vs: [8][2]u8 = undefined;
    const v = variants(shape.ks, &vs);

    try c.w("    const uint grp = threadgroup_position_in_grid.x;\n    const uint tid = thread_position_in_threadgroup.x;\n", .{});
    for (0..m) |j| try c.w(
        \\    const device float* in{d} = tiles + (grp * {d}u + {d}u) * 256u;
        \\    device half* row{d} = (device half*)temp_edges + ulong(grp * {d}u + {d}u) * 128u * {d}u;
        \\    uint pin{d} = 0u;
        \\
    , .{ j, m, j, j, m, j, sm, j });
    try c.w("    threadgroup half costs[{d}][2][{d}];\n", .{ m, sm });
    if (!shape.pins_device) try c.w("    threadgroup uint sh_pin[{d}];\n", .{m});
    try c.w("    bool pinned = false;\n    const float INF = 65504.0f;\n", .{});
    var nsteps_buf: [48]u8 = undefined;
    const nsteps = if (pin_pass == pair_search.full_pin_pass) "128u" else std.fmt.bufPrint(&nsteps_buf, "(pass == 0u ? {d}u : 128u)", .{2 * pin_pass}) catch unreachable;
    try c.w(
        \\    for (uint pass = 0u; pass < 2u; pass++) {{
        \\        const bool last = pass == 1u;
        \\        const uint roll = pass == 0u ? {d}u : 0u;
        \\        const uint nsteps = {s};
        \\        uint curr = 0u;
        \\        {{
        \\            const uint pi = roll;
        \\            const bool keep = last || pi < {d}u || pi == 127u;
        \\
    , .{ pair_search.pairs - pin_pass, nsteps, pin_pass });
    const targets = struct {
        fn emit(cc: Ctx) !void {
            try cc.w("            ", .{});
            for (0..cc.plan.tiles) |j| try cc.w("const float t0_{d} = in{d}[2u * pi], t1_{d} = in{d}[2u * pi + 1u]; ", .{ j, j, j, j });
            try cc.w("\n", .{});
        }
    };
    try targets.emit(c);
    try c.w("            switch (VAR[pi & 7u]) {{\n", .{});
    for (vs[0..v.count], 0..) |vv, vi| {
        try c.w("            case {d}: {{", .{vi});
        try c.firstBody(vv[0], vv[1]);
        try c.w(" }} break;\n", .{});
    }
    try c.w(
        \\            }}
        \\        }}
        \\        threadgroup_barrier(mem_flags::mem_threadgroup);
        \\        for (uint step = 1u; step < nsteps; step++) {{
        \\            const uint pi = (step + roll) & 127u;
        \\            const uint prev_buf = curr;
        \\            curr = 1u - curr;
        \\            const bool keep = last || pi < {d}u || pi == 127u;
        \\
    , .{pin_pass});
    try targets.emit(c);
    try c.w("            switch (VAR[pi & 7u]) {{\n", .{});
    for (vs[0..v.count], 0..) |vv, vi| {
        try c.w("            case {d}: {{", .{vi});
        try c.stepBody(vv[0], vv[1]);
        try c.w(" }} break;\n", .{});
    }
    try c.w(
        \\            }}
        \\            if (step == nsteps - 1u) {{ threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device); }} else {{ threadgroup_barrier(mem_flags::mem_threadgroup); }}
        \\        }}
        \\        if ((tid >> 5u) < {d}u) {{
        \\            const uint j = tid >> 5u;
        \\            const uint tile = grp * {d}u + j;
        \\            uint edge =
    , .{ m, m });
    try c.pinAt("j");
    try c.w(
        \\;
        \\            if (pass == 0u) {{
        \\                float lbest = INF; uint lidx = 0xFFFFFFFFu;
        \\                const uint s_last = 1u << (16u - uint(KST[roll & 7u]));
        \\                for (uint e = tid & 31u; e < s_last; e += 32u) {{
        \\                    const float v = float(costs[j][curr][e]);
        \\                    if (v < lbest) {{ lbest = v; lidx = e; }}
        \\                }}
        \\                const float sv = simd_min(lbest);
        \\                const uint si = simd_min(lbest == sv ? lidx : 0xFFFFFFFFu);
        \\                edge = si == 0xFFFFFFFFu ? 0u : si;
        \\            }}
        \\            const device half* row = (const device half*)temp_edges + ulong(tile) * 128u * {d}u;
        \\            const device float* in_t = tiles + tile * 256u;
        \\            const uint lane = tid & 31u, pin_j =
    , .{sm});
    try c.pinAt("j");
    try c.w(
        \\;
        \\            for (int st = int(nsteps) - 1; st >= 0; st--) {{
        \\                const uint pi = (uint(st) + roll) & 127u;
        \\                const uint kn = uint(KST[(pi + 1u) & 7u]), kc = uint(KST[pi & 7u]), s_bits = 16u - kn;
        \\                const float t0 = in_t[2u * pi], t1 = in_t[2u * pi + 1u];
        \\                const ulong prow = ulong((pi + 127u) & 127u) * {d}u;
        \\                float lb = INF; uint la = 0xFFFFFFFFu;
        \\                for (uint a = lane; a < (1u << kn); a += 32u) {{
        \\                    const uint cw = (a << s_bits) | edge;
        \\                    const float2 v =
    , .{sm});
    try c.dec("cw");
    try c.w(
        \\;
        \\                    const float d0 = v.x - t0, d1 = v.y - t1;
        \\                    float err;
        \\                    if (st == 0) {{
        \\                        if (pinned && (cw >> kc) != pin_j) err = INF;
        \\                        else {{ err = fma(d1, d1, d0 * d0); if (err > INF) err = INF; }}
        \\                    }} else {{
        \\                        err = fma(d1, d1, fma(d0, d0, float(row[prow + (cw >> kc)])));
        \\                    }}
        \\                    if (err < lb) {{ lb = err; la = a; }}
        \\                }}
        \\                const float sv = simd_min(lb);
        \\                const uint sa = simd_min(lb == sv ? la : 0xFFFFFFFFu);
        \\                const uint cw = ((sa == 0xFFFFFFFFu ? 0u : sa) << s_bits) | edge;
        \\                edge = cw >> kc;
        \\                if (last) {{
        \\                    if (lane == 0u) {{
        \\                        indices[tile * 128u + pi] = ushort(cw);
        \\                        const float2 v =
    , .{});
    try c.dec("cw");
    try c.w(
        \\;
        \\                        q_tiles[tile * 256u + 2u * pi] = v.x;
        \\                        q_tiles[tile * 256u + 2u * pi + 1u] = v.y;
        \\                    }}
        \\                }} else if (pi == 0u) {{
        \\                    break;
        \\                }}
        \\            }}
        \\            if (lane == 0u)
    , .{});
    try c.pinAt("j");
    try c.w(
        \\ = edge;
        \\        }}
        \\        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        \\
    , .{});
    for (0..m) |j| {
        var jb: [8]u8 = undefined;
        try c.w("        pin{d} = ", .{j});
        try c.pinAt(std.fmt.bufPrint(&jb, "{d}u", .{j}) catch unreachable);
        try c.w(";\n", .{});
    }
    try c.w("        pinned = true;\n    }}\n", .{});
    return out.toOwnedSliceSentinel(allocator, 0);
}

/// Both weights of every codeword as one half2 (low = the even weight), as u32: the table the high rates read.
pub fn decodeTableWords(allocator: std.mem.Allocator, p: format.Pair) ![]u32 {
    const words = try allocator.alloc(u32, 1 << 16);
    for (words, 0..) |*v, cw| {
        const h = format.decodePair(@intCast(cw), p);
        v.* = @as(u32, h[0]) | (@as(u32, h[1]) << 16);
    }
    return words;
}

test "pair_kernel: every rate's plan fits the threadgroup and its states" {
    const t = std.testing;
    var n: u32 = 24;
    while (n <= 64) : (n += 2) {
        const plan = planFor(n);
        const shape = shapeOf(n, plan);
        try t.expect(plan.tiles * 2 * shape.sm * 2 <= tg_bytes);
        try t.expect(plan.threads >= 32 * plan.tiles);
        try t.expect(plan.threads <= @as(u32, 1) << @intCast(16 - std.mem.max(u8, &shape.ks)));
        const src = try source(t.allocator, n, plan, pair_search.full_pin_pass);
        t.allocator.free(src);
        const hdr = try header(t.allocator, n);
        t.allocator.free(hdr);
    }
}
