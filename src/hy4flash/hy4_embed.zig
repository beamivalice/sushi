//! Hy4-Flash's STQ embedding gathered on the GPU. The blocks of 128 ids a chunk reads are decoded
//! (`trunk_gemm.decodeDense` over each block's 8 output tiles), their Hadamard taken down every 128-row chunk and along
//! each token's column, then scaled by `suh` and `svh`: `exl3.linear.decodeDense`'s f32 operations in its order, so a row
//! is `EmbedGather.rows`' bytes.
const std = @import("std");
const mlx = @import("../mlx.zig");
const exl3 = @import("sushi_exl3");
const hy4_stq = @import("hy4_stq.zig");
const metal_def = @import("../metal_def.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const Arr = mlx.mlx_array;
const plan = @import("../tensor/plan/plan_api.zig");
const mlx_pages = @import("../mlx_pages.zig");

const block = exl3.format.HAD_DIM;
/// The blocks one column launch transforms: a group holds `group_blocks · dim · 128` f16 decoded and f32 transformed.
pub const group_blocks: usize = 32;

/// The blocks a group of `n` launches at: a power of two, so the decode's width (a kernel template) takes one of six
/// values, each made at load (`Model.warmKernels`). The rows past `n` repeat its last block and are read by no token.
pub fn launchBlocks(n: usize) usize {
    return std.math.ceilPowerOfTwoAssert(usize, @max(n, 1));
}

/// `hadamard128` down a column: each output `c` the sum over `k` ascending of `had[k][c] · v[k]`, one product then
/// one add a term. Thread (c, o, ch) writes block `o / 128`'s row `ch · 128 + c` at column `o % 128`.
const COLS_SOURCE =
    \\#pragma clang fp contract(off)
    \\const uint c = thread_position_in_grid.x, o = thread_position_in_grid.y, ch = thread_position_in_grid.z;
    \\const device half *col = d + (size_t)o * uint(DIM) + ch * 128u;
    \\float acc = 0.0f;
    \\for (uint k = 0; k < 128u; ++k) acc = acc + had[k * 128u + c] * float(col[k]);
    \\x[((size_t)(o / 128u) * uint(DIM) + ch * 128u + c) * 128u + o % 128u] = acc;
;

/// `hadamard128` along a row at the token's column, then the row's `suh` times the column's `svh`.
const ROWS_SOURCE =
    \\#pragma clang fp contract(off)
    \\const uint i = thread_position_in_grid.x, t = thread_position_in_grid.y;
    \\if (i >= uint(DIM)) return;
    \\const uint slot = uint(sel[2u * t]), id = uint(sel[2u * t + 1u]), col = id % 128u;
    \\const device float *row = x + ((size_t)slot * uint(DIM) + i) * 128u;
    \\float acc = 0.0f;
    \\for (uint k = 0; k < 128u; ++k) acc = acc + had[k * 128u + col] * row[k];
    \\e[(size_t)t * uint(DIM) + i] = acc * (float(suh[i]) * float(svh[id]));
;

var cols_def: metal_def.Def = .{ .name = "hy4_embed_cols", .ins = &.{ "d", "had" }, .outs = &.{"x"}, .source = COLS_SOURCE };
var rows_def: metal_def.Def = .{ .name = "hy4_embed_rows", .ins = &.{ "x", "had", "suh", "svh", "sel" }, .outs = &.{"e"}, .source = ROWS_SOURCE };

/// The signed Hadamard matrix `hadamard128` multiplies by, `had[k][c]` at `k · 128 + c`.
const had_table: [block * block]f32 = blk: {
    @setEvalBranchQuota(100_000);
    var t: [block * block]f32 = undefined;
    for (0..block) |k| for (0..block) |c| {
        t[k * block + c] = (if (@popCount(k & c) % 2 == 0) @as(f32, 1) else -1) * exl3.format.HAD_SCALE;
    };
    break :blk t;
};

fn ci(v: usize) c_int {
    return @intCast(v);
}

/// The device side of an STQ embedding: its bank's views.
pub const Gather = struct {
    bank: hy4_stq.Bank,

    /// Each id's row, f32 `[ids.len, dim]`, owned by the caller. An id past the vocabulary is refused.
    pub fn rows(self: Gather, gpa: std.mem.Allocator, s: mlx.mlx_stream, ids: []const u32) !Arr {
        const b = self.bank;
        const sh = mlx.getShape(b.w);
        const dim: usize = @intCast(sh[1] * 16);
        const vocab: usize = @intCast(sh[2] * 16);
        for (ids) |id| if (id >= vocab) return error.TokenOutOfRange;
        if (ids.len == 0) {
            var none = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(none);
            try mlx.check(mlx.mlx_zeros(&none, &[_]c_int{ 0, @intCast(dim) }, 2, .float32, s));
            return none;
        }
        // The distinct blocks, ascending.
        const blocks = try gpa.alloc(u32, ids.len);
        defer gpa.free(blocks);
        for (ids, blocks) |id, *k| k.* = id / @as(u32, block);
        std.mem.sort(u32, blocks, {}, std.sort.asc(u32));
        var nb_all: usize = 0;
        for (blocks) |k| {
            if (nb_all > 0 and blocks[nb_all - 1] == k) continue;
            blocks[nb_all] = k;
            nb_all += 1;
        }
        // The ids decide how many blocks: a dry recording plans the most they can name (`plan.bound`), its ids' own
        // blocks and the lowest others beside them, ascending, so each id finds its block.
        const most_blocks = @min(ids.len, (vocab + block - 1) / block);
        const planned = try plan.bound(nb_all, most_blocks);
        if (planned != nb_all) {
            var other: u32 = 0;
            var k = nb_all;
            while (k < planned) : (other += 1) {
                if (std.mem.indexOfScalar(u32, blocks[0..nb_all], other) != null) continue;
                blocks[k] = other;
                k += 1;
            }
            std.mem.sort(u32, blocks[0..planned], {}, std.sort.asc(u32));
            nb_all = planned;
        }
        // What a forward asks follows its rows alone, never how its ids fall in the blocks (a dry recording's ids are
        // zeros): every group's (slot, id) pairs go up in one array, each group's rows land in one block of the ids'
        // rows in the groups' order, and the result is that block's permutation back to the ids' order. Whether the
        // groups land and the width each launches at follow the most blocks the rows can name, so a wide forward asks
        // what its dry recording asked, its region's first fit included, whichever blocks its ids fall in.
        const n_groups = (nb_all + group_blocks - 1) / group_blocks;
        const launch = launchBlocks(@min(group_blocks, most_blocks));
        const starts = try gpa.alloc(usize, n_groups + 1);
        defer gpa.free(starts);
        @memset(starts, 0);
        const home = try gpa.alloc(u32, ids.len);
        defer gpa.free(home);
        for (ids, home) |id, *h| {
            const k = std.sort.binarySearch(u32, blocks[0..nb_all], id / @as(u32, block), struct {
                fn cmp(want: u32, have: u32) std.math.Order {
                    return std.math.order(want, have);
                }
            }.cmp).?;
            h.* = @intCast(k);
            starts[k / group_blocks + 1] += 1;
        }
        for (1..starts.len) |g| starts[g] += starts[g - 1];
        const sel = try gpa.alloc(i32, 2 * ids.len);
        defer gpa.free(sel);
        // Row `inv[ti]` of the groups' rows is id `ti`'s.
        const inv = try gpa.alloc(i32, ids.len);
        defer gpa.free(inv);
        const next = try gpa.dupe(usize, starts[0..n_groups]);
        defer gpa.free(next);
        for (ids, home, 0..) |id, k, ti| {
            const g = k / group_blocks;
            const at = next[g];
            next[g] += 1;
            sel[2 * at] = @intCast(k % group_blocks);
            sel[2 * at + 1] = @intCast(id);
            inv[ti] = @intCast(at);
        }
        var outer: Ops = .{ .s = s };
        defer outer.deinit();
        const had = try outer.own(mlx.mlx_array_new_data(&had_table, &[_]c_int{block * block}, 1, .float32));
        const pairs_all = try outer.own(mlx.mlx_array_new_data(sel.ptr, &[_]c_int{ci(2 * ids.len)}, 1, .int32));
        // On the engine every group's rows land in one block; MLX's stream joins them, where a write copies the block.
        var landed: ?mlx_pages.InPlace = null;
        defer if (landed) |*l| l.deinit();
        var parts: std.ArrayList(Arr) = .empty;
        defer parts.deinit(gpa);
        if (most_blocks > group_blocks and mlx.engine.isStream(s)) {
            var whole = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(whole);
            try mlx.check(mlx.engine.freshArray(&whole, &.{ ci(ids.len), ci(dim) }, .float32));
            landed = mlx_pages.InPlace.adopt(whole, .ring);
        }
        const tiles = try gpa.alloc(i32, group_blocks * 8);
        defer gpa.free(tiles);
        for (0..n_groups) |gi| {
            const g0 = gi * group_blocks;
            const group = blocks[g0..@min(g0 + group_blocks, nb_all)];
            const nb = launch;
            var o: Ops = .{ .s = s };
            defer o.deinit();
            // Each block's 8 output tiles over every input tile, decoded to f16 `[nb · 128, dim]`.
            for (0..nb) |q| for (0..8) |r| {
                tiles[q * 8 + r] = @intCast(group[@min(q, group.len - 1)] * 8 + r);
            };
            const idx = try o.own(mlx.mlx_array_new_data(tiles.ptr, &[_]c_int{ci(nb * 8)}, 1, .int32));
            const sub = try o.contiguous(try o.take(b.w, idx, 2));
            const dense = blk: {
                const previous = exl3.kernels.active_decode;
                defer exl3.kernels.active_decode = previous;
                exl3.kernels.active_decode = b.dec;
                break :blk try o.own(try exl3.trunk_gemm.decodeDense(s, sub));
            };
            var xo: [1]Arr = undefined;
            try metal_def.launch(s, &cols_def, &.{ dense, had }, &.{.{ .shape = &.{ ci(nb), ci(dim), ci(block) }, .dt = .float32 }}, .{ block, nb * block, dim / block }, .{ block, 1, 1 }, &.{.{ .name = "DIM", .v = ci(dim) }}, &.{}, &xo);
            const x = try o.own(xo[0]);
            const t0 = starts[gi];
            const tn = starts[gi + 1] - t0;
            // How many of the ids fall in the group follows the ids: a dry recording plans every id there.
            const launched = try plan.bound(tn, ids.len);
            // A landing group asks every id's row, as its recording does, and computes its own.
            const out_rows = if (landed != null) ids.len else launched;
            const pairs = try o.contiguous(try o.slice(pairs_all, 0, ci(2 * t0), ci(2 * (t0 + tn))));
            var eo: [1]Arr = undefined;
            try metal_def.launch(s, &rows_def, &.{ x, had, b.suh, b.svh, pairs }, &.{.{ .shape = &.{ ci(out_rows), ci(dim) }, .dt = .float32 }}, .{ dim, launched, 1 }, .{ @min(dim, 256), 1, 1 }, &.{.{ .name = "DIM", .v = ci(dim) }}, &.{}, &eo);
            if (landed) |l| {
                const part = try o.slice(try o.own(eo[0]), 0, 0, ci(tn));
                var up = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(up);
                const moved = try l.write(&up, part, &.{ ci(t0), 0 }, &.{ ci(t0 + tn), ci(dim) }, s);
                _ = mlx.mlx_array_free(l.array);
                landed = moved;
                try moved.landed(s);
                continue;
            }
            // Evaluated in its scope: the group's decode and transform go with it, and `tiles` is host data. Owned first,
            // so a failed evaluation frees it. The engine copies host data as it is made, so its last group is left to
            // run behind the forward.
            const part = try outer.own(eo[0]);
            if (!mlx.engine.isStream(s) or gi + 1 < n_groups) try mlx.check(mlx.mlx_array_eval(part));
            try parts.append(gpa, part);
        }
        const all = if (landed) |l| l.array else if (parts.items.len == 1) parts.items[0] else try outer.concat(parts.items, 0);
        // One row's order is its own whatever its id: a decode step takes no permutation.
        if (ids.len == 1) return outer.result(all);
        const perm = try outer.own(mlx.mlx_array_new_data(inv.ptr, &[_]c_int{ci(ids.len)}, 1, .int32));
        const out = try outer.take(all, perm, 0);
        if (!mlx.engine.isStream(s)) try mlx.check(mlx.mlx_array_eval(out));
        return outer.result(out);
    }

    pub fn rowFromId(self: Gather, s: mlx.mlx_stream, id: Arr) !Arr {
        const b = self.bank;
        const sh = mlx.getShape(b.w);
        const dim: usize = @intCast(sh[1] * 16);
        var o: Ops = .{ .s = s };
        defer o.deinit();
        const id_i32 = try o.reshape(try o.cast(id, .int32), &.{1});
        const c128 = try o.own(mlx.mlx_array_new_data(&[_]i32{block}, &[_]c_int{1}, 1, .int32));
        const c8 = try o.own(mlx.mlx_array_new_data(&[_]i32{8}, &[_]c_int{1}, 1, .int32));
        const off = try o.own(mlx.mlx_array_new_data(&[_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 }, &[_]c_int{8}, 1, .int32));
        const zero = try o.own(mlx.mlx_array_new_data(&[_]i32{0}, &[_]c_int{1}, 1, .int32));
        const base = try o.binary(.mul, try o.binary(.floor_div, id_i32, c128), c8);
        const idx = try o.binary(.add, base, off);
        const sub = try o.contiguous(try o.take(b.w, idx, 2));
        const dense = blk: {
            const previous = exl3.kernels.active_decode;
            defer exl3.kernels.active_decode = previous;
            exl3.kernels.active_decode = b.dec;
            break :blk try o.own(try exl3.trunk_gemm.decodeDense(s, sub));
        };
        const had = try o.own(mlx.mlx_array_new_data(&had_table, &[_]c_int{block * block}, 1, .float32));
        var xo: [1]Arr = undefined;
        try metal_def.launch(s, &cols_def, &.{ dense, had }, &.{.{ .shape = &.{ 1, ci(dim), ci(block) }, .dt = .float32 }}, .{ block, block, dim / block }, .{ block, 1, 1 }, &.{.{ .name = "DIM", .v = ci(dim) }}, &.{}, &xo);
        const x = try o.own(xo[0]);
        const pairs = try o.concat(&.{ zero, id_i32 }, 0);
        var eo: [1]Arr = undefined;
        try metal_def.launch(s, &rows_def, &.{ x, had, b.suh, b.svh, pairs }, &.{.{ .shape = &.{ 1, ci(dim) }, .dt = .float32 }}, .{ dim, 1, 1 }, .{ @min(dim, 256), 1, 1 }, &.{.{ .name = "DIM", .v = ci(dim) }}, &.{}, &eo);
        const part = try o.own(eo[0]);
        if (!mlx.engine.isStream(s)) try mlx.check(mlx.mlx_array_eval(part));
        return o.result(part);
    }
};
