const std = @import("std");
const mlx = @import("mlx_host").mlx;
const format = @import("sushi_exl3").format;
const encode = @import("encode.zig");
const quantize = @import("quantize.zig");
const st = @import("safetensors.zig");
const stages_mod = @import("stages.zig");
const gpu_factor = @import("gpu_factor.zig");
pub const Pool = @import("pool.zig").Pool;

pub const tile_len = encode.tile_len;
const max_scratch_bytes: usize = 2048 * 1024 * 1024;
const launch_tiles_cap: usize = 4096;

pub const Error = encode.Error || error{
    EmptyTensor,
    GeometryUnsupported,
    MetalKernelCompileFailed,
    MetalKernelBadOutputCount,
    MlxError,
    HessianNotPositiveDefinite,
};

const Phase = struct {
    k: u32,
    kr: u32,
    kk: u32,
    edges: u32,
    gsz: u32,
    blocks: u32,
    cache_prev: bool,
    bits: u32,
    pack: u32,
    elem_bits: u32,
    row_bytes: u32,
};

const Plan = struct {
    period: usize,
    phases: [8]Phase,
    edges_max: u32,
    threads: u32,
    row_bytes: u32,
    row_words: u32,
    global_costs: bool,
    single_costs: bool,
    table: bool,
};

const single_cost_bytes = 16 * 1024;
const single_prev_regs = 32;
// The most a threadgroup can hold on any Apple GPU; a pipeline may allow fewer.
const device_threads = 1024;
const single_threads = device_threads;
const plain_threads = 256;
// The smallest pipeline limit MLX has reported so far; a launch never asks for more.
var learned_threads: u32 = std.math.maxInt(u32);

/// Wide windows keep one cost row in threadgroup memory, read a decode table and run 1024 threads per tile
/// when every phase's predecessor costs fit in registers; other plans keep both rows and hash every codeword.
fn buildPlan(rate: format.Rate, dec: format.Decode, max_threads: u32) Error!Plan {
    const wide = try planFor(rate, dec, @min(single_threads, floorPow2(max_threads)));
    if (wide.single_costs) return wide;
    return planFor(rate, dec, @min(plain_threads, floorPow2(max_threads)));
}

fn floorPow2(x: u32) u32 {
    return if (x == 0) 0 else @as(u32, 1) << @intCast(31 - @clz(x));
}

fn planFor(rate: format.Rate, dec: format.Decode, max_threads: u32) Error!Plan {
    if (rate.n % 2 != 0 or rate.n < 24 or rate.n > format.Rate.max_n) return error.RateUnsupported;
    const w: u32 = dec.window.bits();
    if (w < 12 or w > 16) return error.WindowUnsupported;
    if (!encode.searchable(dec.codebook)) return error.CodebookUnsupported;
    const period: usize = 16 / std.math.gcd(@as(u32, 16), rate.n);
    var pattern: [8]u32 = undefined;
    var kmin: u32 = 32;
    var kmax: u32 = 0;
    for (0..period) |i| {
        const k: u32 = @intCast(rate.freshBits(i));
        pattern[i] = k;
        kmin = @min(kmin, k);
        kmax = @max(kmax, k);
    }
    if (kmax >= w) return error.WindowTooNarrow;
    const edges_max = @as(u32, 1) << @intCast(w - kmin);
    const threads = @min(max_threads, @as(u32, 1) << @intCast(w - kmax));
    var phases: [8]Phase = undefined;
    var max_rb: u32 = 0;
    for (0..period) |i| {
        const k = pattern[i];
        const k_next = pattern[(i + 1) % period];
        const kr = w - k_next;
        const edges = @as(u32, 1) << @intCast(kr);
        const kk = @as(u32, 1) << @intCast(k_next);
        const gsz_raw = @min(edges / threads, @as(u32, 1) << @intCast(k));
        const cache_prev = gsz_raw > 1 and kk <= 32;
        const gsz: u32 = if (cache_prev) gsz_raw else 1;
        const blocks = edges / (threads * gsz);
        const bits = k_next;
        var pack: u32 = 1;
        while (pack * 2 <= gsz and pack * 2 * bits <= 32) pack *= 2;
        const width = pack * bits;
        const elem_bits: u32 = if (width <= 8) 8 else if (width <= 16) 16 else 32;
        const row_bytes = (edges / pack) * (elem_bits / 8);
        max_rb = @max(max_rb, row_bytes);
        phases[i] = .{
            .k = k,
            .kr = kr,
            .kk = kk,
            .edges = edges,
            .gsz = gsz,
            .blocks = blocks,
            .cache_prev = cache_prev,
            .bits = bits,
            .pack = pack,
            .elem_bits = elem_bits,
            .row_bytes = row_bytes,
        };
    }
    const row_bytes = ((max_rb + 3) / 4) * 4;
    var single = max_threads == single_threads and (w - kmin) > 12 and edges_max * 2 <= single_cost_bytes;
    for (phases[0..period]) |ph| {
        if (!ph.cache_prev or ph.blocks * ph.kk > single_prev_regs) single = false;
    }
    return .{
        .period = period,
        .phases = phases,
        .edges_max = edges_max,
        .threads = threads,
        .row_bytes = row_bytes,
        .row_words = row_bytes / 4,
        .global_costs = (w - kmin) > 12 and !single,
        .single_costs = single,
        .table = single,
    };
}

fn appendDecodeExpr(out: *std.ArrayList(u8), allocator: std.mem.Allocator, cb: format.Codebook, cw_in: []const u8, dst: []const u8) !void {
    switch (cb) {
        .mcg => try out.print(allocator,
            \\                uint dq_cw = {s} * 0xCBAC1FEDu;
            \\                uint dq_r = (dq_cw & 0x8FFF8FFFu) ^ 0x3B603B60u;
            \\                half2 dq_h2 = as_type<half2>(dq_r);
            \\                float {s} = float(dq_h2.x + dq_h2.y);
            \\
        , .{ cw_in, dst }),
        .mul1 => try out.print(allocator,
            \\                uint dq_cw = {s} * 0x83DCD12Du;
            \\                uint dq_sum = 0x6400u;
            \\                for (uint dq_lane = 0u; dq_lane < 4u; dq_lane++) {{
            \\                    dq_sum += (dq_cw >> (8u * dq_lane)) & 0xFFu;
            \\                }}
            \\                half dq_h = as_type<half>(ushort(dq_sum & 0xFFFFu));
            \\                half dq_k_inv = as_type<half>(ushort(0x1EEEu));
            \\                half dq_k_bias = as_type<half>(ushort(0xC931u));
            \\                float {s} = float(fma(dq_h, dq_k_inv, dq_k_bias));
            \\
        , .{ cw_in, dst }),
        .spec1, .pair1, .std1 => unreachable,
    }
}

fn appendPtrElem(out: *std.ArrayList(u8), allocator: std.mem.Allocator, ph: Phase, row: []const u8) !void {
    if (ph.elem_bits == 8) {
        try out.print(allocator, "(edge_ptr + ulong({s}) * ROW_BYTES)", .{row});
    } else if (ph.elem_bits == 16) {
        try out.print(allocator, "((device ushort*)(edge_ptr + ulong({s}) * ROW_BYTES))", .{row});
    } else {
        try out.print(allocator, "((device uint*)(edge_ptr + ulong({s}) * ROW_BYTES))", .{row});
    }
}

fn appendPtrRead(out: *std.ArrayList(u8), allocator: std.mem.Allocator, ph: Phase, row: []const u8, edge: []const u8) !void {
    const shift = std.math.log2_int(u32, ph.pack);
    try out.appendSlice(allocator, "(");
    if (ph.pack > 1) try out.appendSlice(allocator, "(");
    try out.appendSlice(allocator, "uint(");
    try appendPtrElem(out, allocator, ph, row);
    if (shift > 0) {
        try out.print(allocator, "[({s} >> {d}u)])", .{ edge, shift });
    } else {
        try out.print(allocator, "[{s}])", .{edge});
    }
    if (ph.pack > 1) {
        try out.print(allocator, " >> (({s} & {d}u) * {d}u))", .{ edge, ph.pack - 1, ph.bits });
    }
    try out.print(allocator, " & {d}u)", .{ph.kk - 1});
}

fn appendDecode(out: *std.ArrayList(u8), allocator: std.mem.Allocator, cb: format.Codebook, table: bool, cw_in: []const u8, dst: []const u8) !void {
    if (table) return out.print(allocator, "                float {s} = float(dq_tab[{s}]);\n", .{ dst, cw_in });
    try appendDecodeExpr(out, allocator, cb, cw_in, dst);
}

/// A single-buffer step whose edges are scored a packed word at a time: every fresh value reads the
/// word's codewords as half4 loads from the decode table, and each edge still compares its candidates
/// in ascending fresh order, so the choices are the scalar loop's.
fn appendVecStep(out: *std.ArrayList(u8), allocator: std.mem.Allocator, ph: Phase, cost_store: []const u8, pos: []const u8) !void {
    const cast = if (ph.elem_bits == 8) "uchar" else if (ph.elem_bits == 16) "ushort" else "uint";
    try out.print(allocator,
        \\            for (uint bi = 0u; bi < {d}u; bi++) {{
        \\                uint e0 = (bi * THREADS + tid) * {d}u;
        \\                for (uint dw = 0u; dw < {d}u; dw++) {{
        \\                    uint ew = e0 + dw * {d}u;
        \\                    float best[{d}];
        \\                    uint best_fresh[{d}];
        \\                    #pragma clang loop unroll(full)
        \\                    for (uint dd = 0u; dd < {d}u; dd++) {{ best[dd] = H_INF_F; best_fresh[dd] = 0u; }}
        \\                    #pragma clang loop unroll(full)
        \\                    for (uint fresh = 0u; fresh < {d}u; fresh++) {{
        \\                        float pv = prevs[bi][fresh];
        \\                        const device half4* row = (const device half4*)(dq_tab + (fresh << {d}u) + ew);
        \\                        #pragma clang loop unroll(full)
        \\                        for (uint v = 0u; v < {d}u; v++) {{
        \\                            half4 q = row[v];
        \\                            #pragma clang loop unroll(full)
        \\                            for (uint j = 0u; j < 4u; j++) {{
        \\                                float dh = float(q[j]) - target;
        \\                                float err = fma(dh, dh, pv);
        \\                                uint dd = v * 4u + j;
        \\                                if (err < best[dd]) {{
        \\                                    best[dd] = err;
        \\                                    best_fresh[dd] = fresh;
        \\                                }}
        \\                            }}
        \\                        }}
        \\                    }}
        \\                    uint pk = 0u;
        \\                    #pragma clang loop unroll(full)
        \\                    for (uint dd = 0u; dd < {d}u; dd++) {{
        \\                        uint e = ew + dd;
        \\                        {s} = half(best[dd]);
        \\                        pk |= best_fresh[dd] << (dd * {d}u);
        \\                    }}
        \\                    uint widx = (bi * THREADS + tid) * {d}u + dw;
        \\                    
    , .{ ph.blocks, ph.gsz, ph.gsz / ph.pack, ph.pack, ph.pack, ph.pack, ph.pack, ph.kk, ph.kr, ph.pack / 4, ph.pack, cost_store, ph.bits, ph.gsz / ph.pack });
    try appendPtrElem(out, allocator, ph, pos);
    try out.print(allocator, "[widx] = {s}(pk);\n                }}\n            }}\n", .{cast});
}

fn appendEdgeLoop(
    out: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    ph: Phase,
    cb: format.Codebook,
    plan: Plan,
    cost_store: []const u8,
    prev_cost_load: []const u8,
    first: bool,
    pos: []const u8,
) !void {
    const unroll = if (!first and ph.cache_prev) "#pragma clang loop unroll(full)" else "";
    const single = plan.single_costs and !first;
    if (single) {
        try out.print(allocator,
            \\            float prevs[{d}][{d}];
            \\            #pragma clang loop unroll(full)
            \\            for (uint bi = 0u; bi < {d}u; bi++) {{
            \\                uint e0 = (bi * THREADS + tid) * {d}u;
            \\                #pragma clang loop unroll(full)
            \\                for (uint pf = 0u; pf < {d}u; pf++) {{
            \\                    uint pred = ((pf << {d}u) | e0) >> {d}u;
            \\                    prevs[bi][pf] = float({s});
            \\                }}
            \\            }}
            \\            threadgroup_barrier(mem_flags::mem_threadgroup);
            \\            #pragma clang loop unroll(full)
            \\
        , .{ ph.blocks, ph.kk, ph.blocks, ph.gsz, ph.kk, ph.kr, ph.k, prev_cost_load });
        if (plan.table and ph.pack % 4 == 0) return appendVecStep(out, allocator, ph, cost_store, pos);
    }
    try out.print(allocator, "            for (uint bi = 0u; bi < {d}u; bi++) {{\n                uint e0 = (bi * THREADS + tid) * {d}u;\n", .{ ph.blocks, ph.gsz });
    if (!first and ph.cache_prev and !single) {
        try out.print(allocator,
            \\                float prevs[{d}];
            \\                {s}
            \\                for (uint pf = 0u; pf < {d}u; pf++) {{
            \\                    uint pred = ((pf << {d}u) | e0) >> {d}u;
            \\                    prevs[pf] = float({s});
            \\                }}
            \\
        , .{ ph.kk, unroll, ph.kk, ph.kr, ph.k, prev_cost_load });
    }
    try out.print(allocator,
        \\                for (uint dw = 0u; dw < {d}u; dw++) {{
        \\                    uint pk = 0u;
        \\                    for (uint dd = 0u; dd < {d}u; dd++) {{
        \\                    uint e = e0 + dw * {d}u + dd;
        \\                    float best = H_INF_F;
        \\                    uint best_fresh = 0u;
        \\                    {s}
        \\                    for (uint fresh = 0u; fresh < {d}u; fresh++) {{
        \\                        uint cw = (fresh << {d}u) + e;
        \\
    , .{ ph.gsz / ph.pack, ph.pack, ph.pack, unroll, ph.kk, ph.kr });
    if (first or !ph.cache_prev) {
        try out.print(allocator, "                        uint pred = cw >> {d}u;\n", .{ph.k});
    }
    if (first) {
        try out.appendSlice(allocator,
            \\                    float err;
            \\                    if (pinned && pred != pin) {
            \\                        err = H_INF_F;
            \\                    } else {
            \\
        );
        try appendDecode(out, allocator, cb, plan.table, "cw", "dq_val");
        try out.appendSlice(allocator,
            \\                        float dh = dq_val - target;
            \\                        err = dh * dh;
            \\                        if (err > H_INF_F) err = H_INF_F;
            \\                    }
            \\
        );
    } else {
        try appendDecode(out, allocator, cb, plan.table, "cw", "dq_val");
        const prev_val = if (single) "prevs[bi][fresh]" else if (ph.cache_prev) "prevs[fresh]" else prev_cost_load;
        if (ph.cache_prev) {
            try out.print(allocator,
                \\                        float dh = dq_val - target;
                \\                        float err = fma(dh, dh, {s});
                \\
            , .{prev_val});
        } else {
            try out.print(allocator,
                \\                        float dh = dq_val - target;
                \\                        float err = fma(dh, dh, float({s}));
                \\
            , .{prev_val});
        }
    }
    const cast = if (ph.elem_bits == 8) "uchar" else if (ph.elem_bits == 16) "ushort" else "uint";
    try out.print(allocator,
        \\                        if (err < best) {{
        \\                            best = err;
        \\                            best_fresh = fresh;
        \\                        }}
        \\                    }}
        \\                    {s} = half(best);
        \\                    pk |= best_fresh << (dd * {d}u);
        \\                    }}
        \\                    uint widx = (bi * THREADS + tid) * {d}u + dw;
        \\                    
    , .{ cost_store, ph.bits, ph.gsz / ph.pack });
    try appendPtrElem(out, allocator, ph, pos);
    try out.print(allocator, "[widx] = {s}(pk);\n                }}\n            }}\n", .{cast});
}

fn buildKernelSource(allocator: std.mem.Allocator, plan: Plan, cb: format.Codebook) ![:0]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    const cost_store = if (plan.single_costs) "costs[e]" else if (plan.global_costs) "costs[ulong(curr) * EDGES + e]" else "costs[curr][e]";
    const prev_cost_load = if (plan.single_costs) "costs[pred]" else if (plan.global_costs) "costs[ulong(prev_buf) * EDGES + pred]" else "costs[prev_buf][pred]";
    const barrier = if (plan.global_costs)
        "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"
    else
        "threadgroup_barrier(mem_flags::mem_threadgroup);";

    try out.print(allocator,
        \\#define EDGES {d}u
        \\#define ROW_BYTES {d}u
        \\#define THREADS {d}u
        \\#define H_INF_F 65504.0f
        \\
        \\    uint tile = threadgroup_position_in_grid.x;
        \\    uint tid = thread_position_in_threadgroup.x;
        \\
    , .{ plan.edges_max, plan.row_bytes, plan.threads });
    if (plan.period > 1) {
        try out.print(allocator, "    const uchar K_TAB[{d}] = {{", .{plan.period});
        for (0..plan.period) |i| try out.print(allocator, "{s}{d}", .{ if (i == 0) "" else ", ", plan.phases[i].k });
        try out.print(allocator, "}};\n    const uchar KR_TAB[{d}] = {{", .{plan.period});
        for (0..plan.period) |i| try out.print(allocator, "{s}{d}", .{ if (i == 0) "" else ", ", plan.phases[i].kr });
        try out.appendSlice(allocator, "};\n");
    }
    try out.appendSlice(allocator,
        \\    const device float* in_tile = tiles + tile * 256u;
        \\    device float* out_tile = q_tiles + tile * 256u;
        \\    device ushort* out_idx = indices + tile * 256u;
        \\    device uchar* edge_ptr = (device uchar*)temp_edges + ulong(tile) * 256u * ROW_BYTES;
        \\
    );
    if (plan.single_costs) {
        try out.appendSlice(allocator, "    threadgroup half costs[EDGES];\n");
    } else if (plan.global_costs) {
        try out.appendSlice(allocator, "    device half* costs = temp_costs + ulong(tile) * 2u * EDGES;\n");
    } else {
        try out.appendSlice(allocator, "    threadgroup half costs[2][EDGES];\n");
    }
    try out.appendSlice(allocator,
        \\    threadgroup uint sh_pin;
        \\    uint pin = 0u;
        \\    bool pinned = false;
        \\
        \\    for (uint pass = 0u; pass < 2u; pass++) {
        \\        uint curr = 0u;
        \\        uint roll = pass == 0u ? 128u : 0u;
        \\        float target = in_tile[roll];
        \\
    );
    try appendEdgeLoop(&out, allocator, plan.phases[0], cb, plan, cost_store, prev_cost_load, true, "roll");
    try out.print(allocator,
        \\        {s}
        \\        for (uint step = 1u; step < 256u; step++) {{
        \\            uint ri = (step + roll) & 255u;
        \\            uint prev_buf = curr;
        \\            curr = 1u - curr;
        \\            float target = in_tile[ri];
        \\
    , .{barrier});
    if (plan.period == 1) {
        try appendEdgeLoop(&out, allocator, plan.phases[0], cb, plan, cost_store, prev_cost_load, false, "ri");
    } else {
        try out.print(allocator, "            uint ph = ri & {d}u;\n            ", .{plan.period - 1});
        for (0..plan.period) |i| {
            if (i > 0) try out.appendSlice(allocator, " else ");
            if (i + 1 < plan.period) try out.print(allocator, "if (ph == {d}u) ", .{i});
            try out.appendSlice(allocator, "{\n");
            try appendEdgeLoop(&out, allocator, plan.phases[i], cb, plan, cost_store, prev_cost_load, false, "ri");
            try out.appendSlice(allocator, "            }");
        }
        try out.appendSlice(allocator, "\n");
    }
    try out.print(allocator,
        \\            {s}
        \\        }}
        \\        if (tid == 0u) {{
        \\            uint edge = pin;
        \\            if (pass == 0u) {{
        \\                float best = H_INF_F;
        \\                edge = 0u;
        \\                for (uint e = 0u; e < {d}u; e++) {{
        \\                    float v = float({s});
        \\                    if (v < best) {{
        \\                        best = v;
        \\                        edge = e;
        \\                    }}
        \\                }}
        \\            }}
        \\            for (int step_i = 255; step_i >= 0; step_i--) {{
        \\                uint ri = (uint(step_i) + roll) & 255u;
        \\
    , .{ barrier, plan.phases[plan.period - 1].edges, cost_store });
    if (plan.period == 1) {
        try out.appendSlice(allocator, "                uint state = (");
        try appendPtrRead(&out, allocator, plan.phases[0], "ri", "edge");
        try out.print(allocator, " << {d}u) | edge;\n                edge = state >> {d}u;\n", .{ plan.phases[0].kr, plan.phases[0].k });
    } else {
        try out.print(allocator, "                uint bph = ri & {d}u;\n                uint bfresh;\n                ", .{plan.period - 1});
        for (0..plan.period) |i| {
            if (i > 0) try out.appendSlice(allocator, " else ");
            if (i + 1 < plan.period) try out.print(allocator, "if (bph == {d}u) ", .{i});
            try out.appendSlice(allocator, "{ bfresh = ");
            try appendPtrRead(&out, allocator, plan.phases[i], "ri", "edge");
            try out.appendSlice(allocator, "; }");
        }
        try out.appendSlice(allocator,
            \\
            \\                uint state = (bfresh << uint(KR_TAB[bph])) | edge;
            \\                edge = state >> uint(K_TAB[bph]);
            \\
        );
    }
    try out.appendSlice(allocator,
        \\                if (pass == 1u) {
        \\                    out_idx[ri] = ushort(state);
        \\
    );
    try appendDecodeExpr(&out, allocator, cb, "state", "dq_out");
    try out.print(allocator,
        \\                    out_tile[ri] = dq_out;
        \\                }} else if (ri == 0u) {{
        \\                    break;
        \\                }}
        \\            }}
        \\            sh_pin = edge;
        \\        }}
        \\        {s}
        \\        pin = sh_pin;
        \\        pinned = true;
        \\    }}
        \\
    , .{barrier});
    return out.toOwnedSliceSentinel(allocator, 0);
}

const KernelKey = struct {
    n: u32,
    cb: format.Codebook,
    win: format.Window,
    single: bool,
    threads: u32,
};

var cached_keys: [64]KernelKey = undefined;
var cached_kernels: [64]?mlx.mlx_fast_metal_kernel = @splat(null);
var cached_len: usize = 0;

fn getKernel(allocator: std.mem.Allocator, rate: format.Rate, dec: format.Decode, plan: Plan) Error!mlx.mlx_fast_metal_kernel {
    const key = KernelKey{ .n = rate.n, .cb = dec.codebook, .win = dec.window, .single = plan.single_costs, .threads = plan.threads };
    for (cached_kernels[0..cached_len], 0..) |k_opt, i| {
        if (k_opt) |k| if (std.meta.eql(cached_keys[i], key)) return k;
    }
    const source = try buildKernelSource(allocator, plan, dec.codebook);
    defer allocator.free(source);
    var name_buf: [80]u8 = undefined;
    const name_s = std.fmt.bufPrint(&name_buf, "sushi_exl3_search_n{d}_{s}_w{d}{s}", .{ rate.n, @tagName(dec.codebook), dec.window.bits(), if (plan.single_costs) "_single" else "" }) catch unreachable;
    name_buf[name_s.len] = 0;
    const name: [:0]const u8 = name_buf[0..name_s.len :0];
    const ins_all = [_][*:0]const u8{ "tiles", "dq_tab" };
    const ins = ins_all[0..if (plan.table) 2 else 1];
    const outs = [_][*:0]const u8{ "q_tiles", "indices", "temp_edges", "temp_costs" };
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(name.ptr, in_vec, out_vec, source.ptr, "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    if (cached_len < cached_kernels.len) {
        cached_keys[cached_len] = key;
        cached_kernels[cached_len] = kernel;
        cached_len += 1;
    } else {
        _ = mlx.mlx_fast_metal_kernel_free(cached_kernels[0].?);
        cached_keys[0] = key;
        cached_kernels[0] = kernel;
    }
    return kernel;
}

var cached_tables: [4]?mlx.mlx_array = @splat(null);
var cached_table_keys: [4]format.Decode = undefined;

/// Every codeword's decoded value as F16, from the same expression the search hashes, so a table read
/// scores exactly what the hash would.
fn decodeTable(allocator: std.mem.Allocator, s: mlx.mlx_stream, dec: format.Decode) Error!mlx.mlx_array {
    for (cached_tables, 0..) |t_opt, i| if (t_opt) |t| if (std.meta.eql(cached_table_keys[i], dec)) return t;
    var src = std.ArrayList(u8).empty;
    defer src.deinit(allocator);
    try src.appendSlice(allocator, "    uint cw = thread_position_in_grid.x;\n");
    try appendDecodeExpr(&src, allocator, dec.codebook, "cw", "value");
    try src.appendSlice(allocator, "    table[cw] = half(value);\n");
    try src.append(allocator, 0);
    var name_buf: [64]u8 = undefined;
    const name = std.fmt.bufPrintSentinel(&name_buf, "sushi_exl3_decode_table_{s}", .{@tagName(dec.codebook)}, 0) catch unreachable;
    const ins = [_][*:0]const u8{"like"};
    const outs = [_][*:0]const u8{"table"};
    const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(name.ptr, in_vec, out_vec, @ptrCast(src.items.ptr), "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    defer _ = mlx.mlx_fast_metal_kernel_free(kernel);
    const codewords: c_int = @as(c_int, 1) << @intCast(dec.window.bits());
    const like_data = [_]f32{0};
    const like = mlx.mlx_array_new_data(&like_data, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(like);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{codewords}, 1, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, codewords, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const inputs = [_]mlx.mlx_array{like};
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    var table = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&table, outputs_vec, 0));
    try mlx.check(mlx.mlx_array_eval(table));
    for (&cached_tables, 0..) |*slot, i| if (slot.* == null) {
        slot.* = table;
        cached_table_keys[i] = dec;
        return table;
    };
    _ = mlx.mlx_array_free(cached_tables[0].?);
    cached_tables[0] = table;
    cached_table_keys[0] = dec;
    return table;
}

/// Searches `tiles` (`n_tiles * 256` row-major 16x16 tiles) on the Metal GPU, packs the chosen
/// codewords into `packed_out` (`n_tiles * rate.n`), and if `recon` is provided writes the decoded
/// row-major values into `recon`.
pub fn encodeTiles(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    rate: format.Rate,
    dec: format.Decode,
    tiles: []const f32,
    packed_out: []u16,
    recon: ?[]f32,
) Error!void {
    if (packed_out.len != tiles.len / tile_len * rate.n) {
        _ = try buildPlan(rate, dec, device_threads);
        return if (tiles.len == 0) error.EmptyTensor else error.ShapeMismatch;
    }
    var search = try Search.submit(allocator, s, null, rate, dec, tiles, recon != null);
    defer search.deinit();
    try search.collect(null, packed_out, recon);
}

/// One `encodeTiles` call split in two: `submit` queues every launch on the GPU and returns, so the
/// caller can work on the host until `collect` waits for the codes.
pub const Search = struct {
    const Pending = struct { start: usize, count: usize, q: mlx.mlx_array, idx: mlx.mlx_array };

    allocator: std.mem.Allocator,
    rate: format.Rate,
    n_tiles: usize,
    launches: []Pending,

    pub fn submit(allocator: std.mem.Allocator, s: mlx.mlx_stream, pool: ?*Pool, rate: format.Rate, dec: format.Decode, tiles: []const f32, want_recon: bool) Error!Search {
        return submitCapped(allocator, s, pool, rate, dec, tiles, want_recon, device_threads);
    }

    /// Asks for `max_threads` per threadgroup, or for the pipeline limit MLX reports when a kernel cannot take that many.
    pub fn submitCapped(allocator: std.mem.Allocator, s: mlx.mlx_stream, pool: ?*Pool, rate: format.Rate, dec: format.Decode, tiles: []const f32, want_recon: bool, max_threads: u32) Error!Search {
        var cap = @min(max_threads, learned_threads);
        while (true) {
            return submitAt(allocator, s, pool, rate, dec, tiles, want_recon, cap) catch |e| {
                const limit = (if (e == error.MlxError) mlx.takeThreadgroupLimit() else null) orelse return e;
                learned_threads = @min(learned_threads, limit);
                if (limit == 0 or limit >= cap) return e;
                cap = limit;
                continue;
            };
        }
    }

    fn submitAt(allocator: std.mem.Allocator, s: mlx.mlx_stream, pool: ?*Pool, rate: format.Rate, dec: format.Decode, tiles: []const f32, want_recon: bool, cap: u32) Error!Search {
        const plan = try buildPlan(rate, dec, cap);
        if (tiles.len == 0) return error.EmptyTensor;
        if (tiles.len % tile_len != 0) return error.ShapeMismatch;
        const n_tiles = tiles.len / tile_len;
        var perm: [tile_len]usize = undefined;
        format.tensorCorePerm(&perm);

        const kernel = try getKernel(allocator, rate, dec, plan);
        const table = if (plan.table) try decodeTable(allocator, s, dec) else mlx.mlx_array_new();
        defer if (!plan.table) {
            _ = mlx.mlx_array_free(table);
        };
        const per_tile_bytes: usize = 256 * @as(usize, plan.row_bytes) +
            (if (plan.global_costs) 2 * @as(usize, plan.edges_max) * 2 else 0);
        const max_batch = @max(@as(usize, 1), @min(launch_tiles_cap, max_scratch_bytes / per_tile_bytes));
        const n_launch = (n_tiles + max_batch - 1) / max_batch;
        const launches = try allocator.alloc(Pending, n_launch);
        var made: usize = 0;
        errdefer {
            for (launches[0..made]) |l| {
                _ = mlx.mlx_array_free(l.q);
                _ = mlx.mlx_array_free(l.idx);
            }
            allocator.free(launches);
        }
        const k_tiles_buf = try allocator.alloc(f32, @min(n_tiles, max_batch) * tile_len);
        defer allocator.free(k_tiles_buf);
        const evals = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(evals);

        var start: usize = 0;
        while (start < n_tiles) : (made += 1) {
            const count = @min(max_batch, n_tiles - start);
            var gather = Permute{ .src = tiles[start * tile_len ..][0 .. count * tile_len], .dst = k_tiles_buf[0 .. count * tile_len], .perm = &perm };
            runChunks(pool, count, &gather, Permute.gather);
            const nt_c: c_int = @intCast(count);
            const in_arr = mlx.mlx_array_new_data(k_tiles_buf.ptr, &[_]c_int{ nt_c, 256 }, 2, .float32);
            defer _ = mlx.mlx_array_free(in_arr);

            const cfg = mlx.mlx_fast_metal_kernel_config_new();
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ nt_c, 256 }, 2, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ nt_c, 256 }, 2, .uint16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ nt_c, @intCast(256 * plan.row_words) }, 2, .uint32));
            const cost_shape = if (plan.global_costs) [_]c_int{ nt_c, 2, @intCast(plan.edges_max) } else [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &cost_shape, 3, .float16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, nt_c * @as(c_int, @intCast(plan.threads)), 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @intCast(plan.threads), 1, 1));

            const inputs = [_]mlx.mlx_array{ in_arr, table };
            const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, if (plan.table) 2 else 1);
            defer _ = mlx.mlx_vector_array_free(inputs_vec);
            var outputs_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(outputs_vec);
            try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
            if (mlx.mlx_vector_array_size(outputs_vec) != 4) return error.MetalKernelBadOutputCount;

            var q_arr = mlx.mlx_array_new();
            var idx_arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&q_arr, outputs_vec, 0));
            try mlx.check(mlx.mlx_vector_array_get(&idx_arr, outputs_vec, 1));
            launches[made] = .{ .start = start, .count = count, .q = q_arr, .idx = idx_arr };
            try mlx.check(mlx.mlx_vector_array_append_value(evals, idx_arr));
            if (want_recon) try mlx.check(mlx.mlx_vector_array_append_value(evals, q_arr));
            start += count;
        }
        try mlx.check(mlx.mlx_async_eval(evals));
        return .{ .allocator = allocator, .rate = rate, .n_tiles = n_tiles, .launches = launches };
    }

    pub fn collect(self: *Search, pool: ?*Pool, packed_out: []u16, recon: ?[]f32) Error!void {
        if (packed_out.len != self.n_tiles * self.rate.n) return error.ShapeMismatch;
        if (recon) |r| if (r.len != self.n_tiles * tile_len) return error.ShapeMismatch;
        var perm: [tile_len]usize = undefined;
        format.tensorCorePerm(&perm);
        var ks: [tile_len]u8 = undefined;
        for (&ks, 0..) |*k, i| k.* = @intCast(self.rate.freshBits(i));
        for (self.launches) |l| {
            if (recon != null) try mlx.check(mlx.mlx_array_eval(l.q));
            try mlx.check(mlx.mlx_array_eval(l.idx));
        }
        // The pool may still run work started before the wait; unpacking needs it.
        if (pool) |p| p.wait();
        for (self.launches) |l| {
            var unpack = Unpack{
                .rate = self.rate,
                .ks = &ks,
                .perm = &perm,
                .states = mlx.mlx_array_data_uint16(l.idx).?,
                .values = if (recon != null) mlx.mlx_array_data_float32(l.q).? else undefined,
                .packed_out = packed_out[l.start * self.rate.n ..][0 .. l.count * self.rate.n],
                .recon = if (recon) |r| r[l.start * tile_len ..][0 .. l.count * tile_len] else null,
            };
            runChunks(pool, l.count, &unpack, Unpack.run);
        }
    }

    pub fn deinit(self: *Search) void {
        for (self.launches) |l| {
            _ = mlx.mlx_array_free(l.q);
            _ = mlx.mlx_array_free(l.idx);
        }
        self.allocator.free(self.launches);
    }
};

const Permute = struct {
    src: []const f32,
    dst: []f32,
    perm: *const [tile_len]usize,

    fn gather(self: *Permute, lo: usize, hi: usize) void {
        for (lo..hi) |ti| {
            const from = self.src[ti * tile_len ..][0..tile_len];
            for (self.dst[ti * tile_len ..][0..tile_len], self.perm) |*v, p| v.* = from[p];
        }
    }
};

const Unpack = struct {
    rate: format.Rate,
    ks: *const [tile_len]u8,
    perm: *const [tile_len]usize,
    states: [*]const u16,
    values: [*]const f32,
    packed_out: []u16,
    recon: ?[]f32,

    fn run(self: *Unpack, lo: usize, hi: usize) void {
        const n = self.rate.n;
        for (lo..hi) |ti| {
            const states_ptr: *const [tile_len]u16 = @ptrCast(self.states + ti * tile_len);
            encode.Encoder.packStates(self.rate, self.ks, states_ptr, self.packed_out[ti * n ..][0..n]);
            if (self.recon) |r| {
                const q = self.values + ti * tile_len;
                const dst = r[ti * tile_len ..][0..tile_len];
                for (self.perm, 0..) |p, i| dst[p] = q[i];
            }
        }
    }
};

const chunk_tiles = 64;

/// body(ctx, lo, hi) over `count` tiles in chunks of `chunk_tiles`, on the pool when there is one.
fn runChunks(pool: ?*Pool, count: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), usize, usize) void) void {
    const Wrap = struct {
        ctx: @TypeOf(ctx),
        count: usize,
        fn one(self: *const @This(), c: usize) void {
            body(self.ctx, c * chunk_tiles, @min(self.count, (c + 1) * chunk_tiles));
        }
    };
    const w = Wrap{ .ctx = ctx, .count = count };
    const chunks = (count + chunk_tiles - 1) / chunk_tiles;
    if (pool) |p| p.run(chunks, &w, Wrap.one) else for (0..chunks) |c| w.one(c);
}

const sample_width = 3;
const sample_tiles = 128;

fn sampleTiles(allocator: std.mem.Allocator, inner: []const f32, rows: usize, cols: usize) ![]f32 {
    const tk = rows / 16;
    const tn = cols / 16;
    const pair_count = @max(tk, tn) * sample_width;
    const stride = @max(1, pair_count / sample_tiles);
    var tiles: std.ArrayList(f32) = .empty;
    errdefer tiles.deinit(allocator);
    var p: usize = 0;
    var n_sample: usize = 0;
    while (p < pair_count and n_sample < sample_tiles) : (p += stride) {
        const i = p / sample_width;
        const wv = p % sample_width;
        const r0 = (i % tk) * 16;
        const c0 = ((i + wv) % tn) * 16;
        for (0..16) |r| try tiles.appendSlice(allocator, inner[(r0 + r) * cols + c0 ..][0..16]);
        n_sample += 1;
    }
    return tiles.toOwnedSlice(allocator);
}

/// One global scale per tensor for several same-shaped tensors (`quantize.globalScale` on each): every
/// golden-section probe of all tensors is one search launch, and each tensor walks the probes it would alone.
pub fn globalScaleBatch(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    pool: ?*Pool,
    rate: format.Rate,
    dec: format.Decode,
    inners: []const []const f32,
    rows: usize,
    cols: usize,
    out: []f32,
) Error!void {
    if (rows == 0 or cols == 0) return error.EmptyTensor;
    if (rows % 128 != 0 or cols % 128 != 0) return error.GeometryUnsupported;
    if (out.len != inners.len) return error.ShapeMismatch;
    for (inners) |inner| if (inner.len != rows * cols) return error.ShapeMismatch;
    _ = try buildPlan(rate, dec, device_threads);
    const count = inners.len;
    if (count == 0) return;

    const samples = try allocator.alloc([]f32, count);
    defer allocator.free(samples);
    var made: usize = 0;
    defer for (samples[0..made]) |sm| allocator.free(sm);
    while (made < count) : (made += 1) samples[made] = try sampleTiles(allocator, inners[made], rows, cols);
    const per = samples[0].len;
    const scaled = try allocator.alloc(f32, per * count);
    defer allocator.free(scaled);
    const recon = try allocator.alloc(f32, per * count);
    defer allocator.free(recon);
    const packed_scratch = try allocator.alloc(u16, per / tile_len * rate.n * count);
    defer allocator.free(packed_scratch);
    const scores = try allocator.alloc(f64, count);
    defer allocator.free(scores);

    const Probe = struct {
        fn run(a: std.mem.Allocator, st_: mlx.mlx_stream, pl: ?*Pool, r: format.Rate, d: format.Decode, sm: []const []f32, gs: []const f32, scl: []f32, rec: []f32, scr: []u16, sc: []f64) Error!void {
            const n = sm[0].len;
            for (sm, gs, 0..) |sample, g, b| for (scl[b * n ..][0..n], sample) |*o, v| {
                o.* = v * g;
            };
            var search = try Search.submit(a, st_, pl, r, d, scl, true);
            defer search.deinit();
            try search.collect(pl, scr, rec);
            for (sm, gs, 0..) |sample, g, b| {
                var sse: f64 = 0;
                for (rec[b * n ..][0..n], sample) |rv, v| {
                    const dd = rv / g - v;
                    sse += @as(f64, dd * dd);
                }
                sc[b] = sse / @as(f64, @floatFromInt(n));
            }
        }
    };

    const resphi64: f64 = 2.0 - (1.0 + @sqrt(5.0)) / 2.0;
    const resphi: f32 = @floatCast(resphi64);
    const bracket = quantize.gScaleBracket(dec.codebook);
    var width: f64 = bracket[1] - bracket[0];
    var steps: usize = 0;
    while (width > 0.03 and steps < 8) {
        width *= 1.0 - resphi64;
        steps += 1;
    }
    const State = struct { a: f32, b: f32, x1: f32, x2: f32, f1: f64, f2: f64 };
    const states = try allocator.alloc(State, count);
    defer allocator.free(states);
    const probes = try allocator.alloc(f32, count);
    defer allocator.free(probes);
    for (states) |*st_| {
        const a: f32 = bracket[0];
        const b: f32 = bracket[1];
        st_.* = .{ .a = a, .b = b, .x1 = a + resphi * (b - a), .x2 = b - resphi * (b - a), .f1 = 0, .f2 = 0 };
    }
    for (states, probes) |st_, *p| p.* = st_.x1;
    try Probe.run(allocator, s, pool, rate, dec, samples, probes, scaled, recon, packed_scratch, scores);
    for (states, scores) |*st_, sc| st_.f1 = sc;
    for (states, probes) |st_, *p| p.* = st_.x2;
    try Probe.run(allocator, s, pool, rate, dec, samples, probes, scaled, recon, packed_scratch, scores);
    for (states, scores) |*st_, sc| st_.f2 = sc;
    for (0..steps) |_| {
        for (states, probes) |*st_, *p| {
            const left = st_.f1 < st_.f2;
            const a_n = if (left) st_.a else st_.x1;
            const b_n = if (left) st_.x2 else st_.b;
            p.* = if (left) a_n + resphi * (b_n - a_n) else b_n - resphi * (b_n - a_n);
            st_.a = a_n;
            st_.b = b_n;
        }
        try Probe.run(allocator, s, pool, rate, dec, samples, probes, scaled, recon, packed_scratch, scores);
        for (states, scores, probes) |*st_, fp, probe| {
            const left = st_.f1 < st_.f2;
            if (left) {
                st_.x2 = st_.x1;
                st_.f2 = st_.f1;
                st_.x1 = probe;
                st_.f1 = fp;
            } else {
                st_.x1 = st_.x2;
                st_.f1 = st_.f2;
                st_.x2 = probe;
                st_.f2 = fp;
            }
        }
    }
    for (states, out) |st_, *o| o.* = (st_.a + st_.b) / 2.0;
}

const LdlqStep = struct { i: usize, chunk: usize, bi: usize, bj: usize, height: usize, tk: usize, last_in_chunk: bool };

fn ldlqSteps(allocator: std.mem.Allocator, rows: usize) ![]LdlqStep {
    var list: std.ArrayList(LdlqStep) = .empty;
    errdefer list.deinit(allocator);
    var j = rows;
    while (j > 0) {
        const i = j -| 128;
        const chunk = j - i;
        var bj = chunk;
        while (bj > 0) : (bj -= 16) {
            const bi = bj -| 16;
            try list.append(allocator, .{ .i = i, .chunk = chunk, .bi = bi, .bj = bj, .height = bj - bi, .tk = (i + bi) / 16, .last_in_chunk = bi == 0 });
        }
        j = i;
    }
    return list.toOwnedSlice(allocator);
}

const Ldlq = struct {
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    inner: []const f32,
    l: []const f32,
    rec: []f32,
    prod: []f32,
    comp: []f32,
    err_row: []f32,
    err_blk: []f32,
    l_blk: []f32,
    defer_next: usize = 0,
    defer_end: usize = 0,
    defer_rows: usize = 0,
    defer_chunk: usize = 0,

    fn init(allocator: std.mem.Allocator, inner: []const f32, l: []const f32, rows: usize, cols: usize) !Ldlq {
        const rec = try allocator.alloc(f32, rows * cols);
        errdefer allocator.free(rec);
        @memset(rec, 0);
        const prod = try allocator.alloc(f32, rows * cols);
        errdefer allocator.free(prod);
        @memset(prod, 0);
        const comp = try allocator.alloc(f32, 16 * cols);
        errdefer allocator.free(comp);
        const err_row = try allocator.alloc(f32, cols);
        errdefer allocator.free(err_row);
        const err_blk = try allocator.alloc(f32, ldlq_chunk * cols);
        errdefer allocator.free(err_blk);
        const l_blk = try allocator.alloc(f32, rows * ldlq_chunk);
        return .{ .allocator = allocator, .rows = rows, .cols = cols, .inner = inner, .l = l, .rec = rec, .prod = prod, .comp = comp, .err_row = err_row, .err_blk = err_blk, .l_blk = l_blk };
    }

    fn deinit(self: *Ldlq) void {
        self.allocator.free(self.rec);
        self.allocator.free(self.prod);
        self.allocator.free(self.comp);
        self.allocator.free(self.err_row);
        self.allocator.free(self.err_blk);
        self.allocator.free(self.l_blk);
    }

    fn prepare(self: *Ldlq, step: LdlqStep, tiles_out: []f32) void {
        const cols = self.cols;
        const rows = self.rows;
        const i = step.i;
        const height = step.height;
        const comp = self.comp;
        @memcpy(comp[0 .. height * cols], self.prod[(i + step.bi) * cols ..][0 .. height * cols]);
        for (step.bj..step.chunk) |m| {
            for (self.err_row, self.inner[(i + m) * cols ..][0..cols], self.rec[(i + m) * cols ..][0..cols]) |*e, wv, rv| e.* = wv - rv;
            for (0..height) |r| {
                const lv = self.l[(i + m) * rows + i + step.bi + r];
                for (comp[r * cols ..][0..cols], self.err_row) |*c, e| c.* += lv * e;
            }
        }
        for (0..height) |r| for (comp[r * cols ..][0..cols], self.inner[(i + step.bi + r) * cols ..][0..cols]) |*c, wv| {
            c.* = wv + c.*;
        };
        const out_tiles = cols / 16;
        for (0..out_tiles) |tn| {
            const dst = tiles_out[tn * tile_len ..][0..tile_len];
            for (0..16) |r| @memcpy(dst[r * 16 ..][0..16], comp[r * cols + tn * 16 ..][0..16]);
        }
    }

    fn commit(self: *Ldlq, step: LdlqStep, rec_tiles: []const f32) void {
        const cols = self.cols;
        const i = step.i;
        const out_tiles = cols / 16;
        for (0..out_tiles) |tn| {
            const src = rec_tiles[tn * tile_len ..][0..tile_len];
            for (0..16) |r| @memcpy(self.rec[(i + step.bi + r) * cols + tn * 16 ..][0..16], src[r * 16 ..][0..16]);
        }
        // A finished chunk's error reaches the next chunk's rows at once and the rows above it in `feedDeferred`
        // slices over that chunk's steps, off the step's critical path; each row still takes the chunks in order.
        if (step.last_in_chunk and self.defer_next < self.defer_end) {
            self.feedRows(self.defer_next, self.defer_end, self.defer_chunk);
            self.defer_next = self.defer_end;
        }
        if (step.last_in_chunk and i > 0) {
            const chunk = step.chunk;
            for (0..chunk) |m| {
                for (self.err_blk[m * cols ..][0..cols], self.inner[(i + m) * cols ..][0..cols], self.rec[(i + m) * cols ..][0..cols]) |*e, wv, rv| e.* = wv - rv;
            }
            transposeBlock(self.l_blk[0 .. i * chunk], self.l[i * self.rows ..], self.rows, i, chunk);
            const near = i -| ldlq_chunk;
            self.feedRows(near, i, chunk);
            const steps_next = (i - near + ldlq_feedback_rows - 1) / ldlq_feedback_rows;
            self.defer_next = 0;
            self.defer_end = near;
            self.defer_rows = (near + steps_next - 1) / steps_next;
            self.defer_chunk = chunk;
        }
    }

    fn feedDeferred(self: *Ldlq) void {
        if (self.defer_next >= self.defer_end) return;
        const hi = @min(self.defer_end, self.defer_next + self.defer_rows);
        self.feedRows(self.defer_next, hi, self.defer_chunk);
        self.defer_next = hi;
    }

    fn feedRows(self: *Ldlq, lo: usize, hi: usize, chunk: usize) void {
        const cols = self.cols;
        feedbackUpdate(self.prod[lo * cols .. hi * cols], cols, self.err_blk[0 .. chunk * cols], self.l_blk[lo * chunk .. hi * chunk], chunk);
    }
};

const ldlq_feedback_rows = 16;

/// dst[r * chunk + m] = src[m * stride + r] for r < rows, m < chunk, in 32x32 tiles so both sides stay in cache.
fn transposeBlock(dst: []f32, src: []const f32, stride: usize, rows: usize, chunk: usize) void {
    const t = 32;
    var m0: usize = 0;
    while (m0 < chunk) : (m0 += t) {
        var r0: usize = 0;
        while (r0 < rows) : (r0 += t) {
            for (m0..@min(m0 + t, chunk)) |m| {
                const from = src[m * stride ..];
                for (r0..@min(r0 + t, rows)) |r| dst[r * chunk + m] = from[r];
            }
        }
    }
}

const ldlq_chunk = 128;
const FeedLanes = @Vector(16, f32);

/// prod[r] += sum over m of lt[r][m] * err[m], each element accumulated in ascending m with a rounded product
/// then a rounded add, so every value matches the row-at-a-time loop bit for bit.
fn feedbackUpdate(prod: []f32, cols: usize, err: []const f32, lt: []const f32, chunk: usize) void {
    const rows = prod.len / cols;
    var r: usize = 0;
    while (r < rows) : (r += 2) {
        const pair = r + 1 < rows;
        var c0: usize = 0;
        while (c0 < cols) : (c0 += 32) {
            const p0 = prod[r * cols + c0 ..];
            var a00: FeedLanes = p0[0..16].*;
            var a01: FeedLanes = p0[16..32].*;
            var a10: FeedLanes = if (pair) prod[(r + 1) * cols + c0 ..][0..16].* else @splat(0);
            var a11: FeedLanes = if (pair) prod[(r + 1) * cols + c0 + 16 ..][0..16].* else @splat(0);
            const l0row = lt[r * chunk ..][0..chunk];
            const l1row = if (pair) lt[(r + 1) * chunk ..][0..chunk] else l0row;
            for (0..chunk) |m| {
                const e0: FeedLanes = err[m * cols + c0 ..][0..16].*;
                const e1: FeedLanes = err[m * cols + c0 + 16 ..][0..16].*;
                const l0: FeedLanes = @splat(l0row[m]);
                const l1: FeedLanes = @splat(l1row[m]);
                a00 += l0 * e0;
                a01 += l0 * e1;
                a10 += l1 * e0;
                a11 += l1 * e1;
            }
            p0[0..16].* = a00;
            p0[16..32].* = a01;
            if (pair) {
                prod[(r + 1) * cols + c0 ..][0..16].* = a10;
                prod[(r + 1) * cols + c0 + 16 ..][0..16].* = a11;
            }
        }
    }
}

pub const TensorJob = struct {
    w: []const f32,
    h: []const f32,
    sign_in: []const f32,
    sign_out: []const f32,
};

fn workerCount(jobs: usize) usize {
    const cpus = std.Thread.getCpuCount() catch 1;
    return @max(1, @min(jobs, @min(cpus, 16)));
}

fn parallelFor(count: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), usize) void) void {
    const workers = workerCount(count);
    if (workers <= 1) {
        for (0..count) |i| body(ctx, i);
        return;
    }
    const Shared = struct {
        next: std.atomic.Value(usize) = .init(0),
        count: usize,
        ctx: @TypeOf(ctx),
        fn run(self: *@This()) void {
            while (true) {
                const i = self.next.fetchAdd(1, .monotonic);
                if (i >= self.count) return;
                body(self.ctx, i);
            }
        }
    };
    var shared = Shared{ .count = count, .ctx = ctx };
    var threads: [16]std.Thread = undefined;
    var spawned: usize = 0;
    while (spawned + 1 < workers) : (spawned += 1) {
        threads[spawned] = std.Thread.spawn(.{}, Shared.run, .{&shared}) catch break;
    }
    shared.run();
    for (threads[0..spawned]) |th| th.join();
}

/// The host half of `quantizeBatch` for `jobs`: regularized inner matrices and LDL factors. It makes no MLX
/// call, so it can run on any thread beside a GPU search. At float32 precision the factors stay unbuilt: the
/// damped Hessians wait in `damped` for `finishBatch` to factor them on the GPU.
pub const Prepared = struct {
    allocator: std.mem.Allocator,
    in: usize,
    out: usize,
    reg: []quantize.Regularized,
    l: [][]f32,
    damped: [][]f32,
    means: []f64,
    precision: gpu_factor.Precision,
    factor_stats: gpu_factor.Stats = .{},

    pub fn deinit(self: *Prepared) void {
        for (self.reg, self.l, self.damped) |*r, l, d| {
            r.deinit(self.allocator);
            self.allocator.free(l);
            self.allocator.free(d);
        }
        self.allocator.free(self.reg);
        self.allocator.free(self.l);
        self.allocator.free(self.damped);
        self.allocator.free(self.means);
    }
};

const Prep = struct {
    jobs: []const TensorJob,
    in: usize,
    out: usize,
    mode: quantize.OutScales,
    precision: gpu_factor.Precision,
    allocator: std.mem.Allocator,
    reg: []quantize.Regularized,
    l: [][]f32,
    damped: [][]f32,
    means: []f64,
    failed: []?Error,

    fn run(self: *Prep, b: usize) void {
        const job = self.jobs[b];
        const hdiag = self.allocator.alloc(f32, self.in) catch |e| return self.fail(b, e);
        defer self.allocator.free(hdiag);
        for (hdiag, 0..) |*d, i| d.* = job.h[i * self.in + i];
        self.reg[b] = quantize.regularize(self.allocator, job.w, self.in, self.out, job.sign_in, job.sign_out, self.mode, hdiag) catch |e| return self.fail(b, e);
        const rotated = quantize.rotatedHessian(self.allocator, job.h, self.reg[b].suh) catch |e| return self.fail(b, e);
        if (self.precision == .f32) {
            self.means[b] = quantize.dampDiagonal(rotated, self.in);
            self.damped[b] = rotated;
            return;
        }
        defer self.allocator.free(rotated);
        self.l[b] = quantize.ldlFactor(self.allocator, rotated, self.in) catch |e| return self.fail(b, e);
    }

    fn fail(self: *Prep, b: usize, e: Error) void {
        self.failed[b] = e;
    }
};

pub fn prepareBatch(allocator: std.mem.Allocator, pool: ?*Pool, jobs: []const TensorJob, in: usize, out: usize, mode: quantize.OutScales, precision: gpu_factor.Precision) Error!Prepared {
    if (in == 0 or out == 0) return error.EmptyTensor;
    if (in % 128 != 0 or out % 128 != 0) return error.GeometryUnsupported;
    for (jobs) |job| {
        if (job.w.len != in * out or job.h.len != in * in or job.sign_in.len != in or job.sign_out.len != out) return error.ShapeMismatch;
    }
    const count = jobs.len;
    const reg = try allocator.alloc(quantize.Regularized, count);
    errdefer allocator.free(reg);
    const l = try allocator.alloc([]f32, count);
    errdefer allocator.free(l);
    const damped = try allocator.alloc([]f32, count);
    errdefer allocator.free(damped);
    const means = try allocator.alloc(f64, count);
    errdefer allocator.free(means);
    @memset(means, 0);
    const failed = try allocator.alloc(?Error, count);
    defer allocator.free(failed);
    @memset(failed, null);
    for (reg, l, damped) |*r, *lb, *d| {
        r.* = .{ .inner = &.{}, .suh = &.{}, .svh = &.{} };
        lb.* = &.{};
        d.* = &.{};
    }
    var prep = Prep{ .jobs = jobs, .in = in, .out = out, .mode = mode, .precision = precision, .allocator = allocator, .reg = reg, .l = l, .damped = damped, .means = means, .failed = failed };
    runParallel(pool, count, &prep, Prep.run);
    var result = Prepared{ .allocator = allocator, .in = in, .out = out, .reg = reg, .l = l, .damped = damped, .means = means, .precision = precision };
    for (failed) |f| if (f) |e| {
        result.deinit();
        return e;
    };
    return result;
}

fn runParallel(pool: ?*Pool, count: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), usize) void) void {
    if (pool) |p| p.run(count, ctx, body) else parallelFor(count, ctx, body);
}

/// Main-thread work the LDLQ runs between its steps while the GPU has a search queued: `step` does one
/// piece and returns whether more remain. It charges its own stage time.
pub const Side = struct {
    ctx: *anyopaque,
    step: *const fn (*anyopaque) anyerror!bool,
};

/// The GPU half of `quantizeBatch`: the global scale and the LDLQ search of every prepared tensor.
pub fn finishBatch(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    pool: ?*Pool,
    prepared: *Prepared,
    rate: format.Rate,
    dec: format.Decode,
    stages: ?*stages_mod.Stages,
    results: []quantize.Quantized,
    side: ?Side,
) anyerror!void {
    const count = prepared.reg.len;
    if (results.len != count) return error.ShapeMismatch;
    _ = try buildPlan(rate, dec, device_threads);
    if (count == 0) return;
    const in = prepared.in;
    const out = prepared.out;
    var lap = stages_mod.Lap.start(stages);
    if (prepared.precision == .f32) {
        try gpu_factor.factorBatch(allocator, s, @ptrCast(prepared.damped), prepared.means, in, .{}, &prepared.factor_stats, prepared.l);
        for (prepared.damped) |*d| {
            allocator.free(d.*);
            d.* = &.{};
        }
        lap.mark(.factor);
    }
    const inners = try allocator.alloc([]const f32, count);
    defer allocator.free(inners);
    for (inners, prepared.reg) |*o, r| o.* = r.inner;
    const gs = try allocator.alloc(f32, count);
    defer allocator.free(gs);
    try globalScaleBatch(allocator, s, pool, rate, dec, inners, in, out, gs);
    for (prepared.reg, gs) |r, g| for (r.inner) |*v| {
        v.* *= g;
    };
    lap.mark(.scale);

    var made: usize = 0;
    const packed_out = try allocator.alloc([]u16, count);
    defer allocator.free(packed_out);
    errdefer for (packed_out[0..made]) |p| allocator.free(p);
    while (made < count) : (made += 1) packed_out[made] = try allocator.alloc(u16, in / 16 * (out / 16) * rate.n);
    const ls = try allocator.alloc([]const f32, count);
    defer allocator.free(ls);
    for (ls, prepared.l) |*o, l| o.* = l;
    try ldlqBatch(allocator, s, pool, rate, dec, inners, ls, in, out, packed_out, &lap, side);

    var done: usize = 0;
    errdefer for (results[0..done]) |r| {
        allocator.free(r.suh);
        allocator.free(r.svh);
    };
    while (done < count) : (done += 1) {
        const suh = try allocator.alloc(u16, in);
        errdefer allocator.free(suh);
        for (suh, prepared.reg[done].suh) |*o, v| o.* = format.f32ToF16Bits(v / gs[done]);
        const svh = try allocator.alloc(u16, out);
        for (svh, prepared.reg[done].svh) |*o, v| o.* = format.f32ToF16Bits(v);
        results[done] = .{ .packed_out = packed_out[done], .suh = suh, .svh = svh, .g = gs[done] };
    }
    made = 0;
}

/// Quantizes `jobs` (all `in` x `out`) together: the per-tensor host stages run on worker threads and
/// the search launches carry one 16-row step of many tensors. Each tensor's bytes equal `quantizeTensor`
/// on that tensor alone.
pub fn quantizeBatch(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    jobs: []const TensorJob,
    in: usize,
    out: usize,
    rate: format.Rate,
    dec: format.Decode,
    mode: quantize.OutScales,
    stages: ?*stages_mod.Stages,
    results: []quantize.Quantized,
) Error!void {
    if (results.len != jobs.len) return error.ShapeMismatch;
    _ = try buildPlan(rate, dec, device_threads);
    var lap = stages_mod.Lap.start(stages);
    var prepared = try prepareBatch(allocator, null, jobs, in, out, mode, .f64);
    defer prepared.deinit();
    lap.mark(.prep);
    // Without a side job only the search's own errors can surface.
    finishBatch(allocator, s, null, &prepared, rate, dec, stages, results, null) catch |e| return @errorCast(e);
}

/// One lane's tensors through the LDLQ steps: host feedback before each step's search, commit after.
const Lane = struct {
    states: []Ldlq,
    packed_out: []const []u16,
    tiles: []f32,
    rec: []f32,
    packed_all: []u16,
    stride: usize,
    step: LdlqStep = undefined,
    search: ?Search = null,

    fn prepare(self: *Lane, b: usize) void {
        self.states[b].prepare(self.step, self.tiles[b * self.stride ..][0..self.stride]);
    }

    fn commit(self: *Lane, b: usize) void {
        self.states[b].commit(self.step, self.rec[b * self.stride ..][0..self.stride]);
    }

    fn feedDeferred(self: *Lane, b: usize) void {
        self.states[b].feedDeferred();
    }
};

const ldlq_lanes = 2;

/// Two lanes alternate so the GPU searches one lane's step while the host feeds the other.
fn ldlqBatch(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    pool: ?*Pool,
    rate: format.Rate,
    dec: format.Decode,
    inners: []const []const f32,
    ls: []const []const f32,
    rows: usize,
    cols: usize,
    packed_out: []const []u16,
    lap: *stages_mod.Lap,
    side: ?Side,
) anyerror!void {
    const count = inners.len;
    const states = try allocator.alloc(Ldlq, count);
    defer allocator.free(states);
    var made: usize = 0;
    defer for (states[0..made]) |*st_| st_.deinit();
    while (made < count) : (made += 1) states[made] = try Ldlq.init(allocator, inners[made], ls[made], rows, cols);
    const steps = try ldlqSteps(allocator, rows);
    defer allocator.free(steps);
    const out_tiles = cols / 16;
    const stride = out_tiles * tile_len;
    const tiles = try allocator.alloc(f32, count * stride);
    defer allocator.free(tiles);
    const rec = try allocator.alloc(f32, count * stride);
    defer allocator.free(rec);
    const packed_all = try allocator.alloc(u16, count * out_tiles * rate.n);
    defer allocator.free(packed_all);

    const n_split = @max(1, @min(ldlq_lanes, count));
    var lanes_buf: [ldlq_lanes]Lane = undefined;
    var n_lanes: usize = 0;
    for (0..n_split) |li| {
        const range = [2]usize{ li * count / n_split, (li + 1) * count / n_split };
        if (range[0] == range[1]) continue;
        lanes_buf[n_lanes] = .{
            .states = states[range[0]..range[1]],
            .packed_out = packed_out[range[0]..range[1]],
            .tiles = tiles[range[0] * stride .. range[1] * stride],
            .rec = rec[range[0] * stride .. range[1] * stride],
            .packed_all = packed_all[range[0] * out_tiles * rate.n .. range[1] * out_tiles * rate.n],
            .stride = stride,
        };
        n_lanes += 1;
    }
    const lanes = lanes_buf[0..n_lanes];
    defer for (lanes) |*ln| if (ln.search) |*sr| sr.deinit();
    defer if (pool) |p| p.wait();
    const tile_words = out_tiles * rate.n;
    var side_left = side != null;

    for (lanes) |*ln| {
        ln.step = steps[0];
        runParallel(pool, ln.states.len, ln, Lane.prepare);
        ln.search = try Search.submit(allocator, s, pool, rate, dec, ln.tiles, true);
    }
    lap.mark(.feedback);
    for (steps, 0..) |step, k| {
        for (lanes) |*ln| {
            var sr = ln.search.?;
            ln.search = null;
            defer sr.deinit();
            try sr.collect(pool, ln.packed_all, ln.rec);
            lap.mark(.search);
            for (ln.packed_out, 0..) |dst, b| @memcpy(dst[step.tk * tile_words ..][0..tile_words], ln.packed_all[b * tile_words ..][0..tile_words]);
            runParallel(pool, ln.states.len, ln, Lane.commit);
            if (k + 1 < steps.len) {
                ln.step = steps[k + 1];
                runParallel(pool, ln.states.len, ln, Lane.prepare);
                ln.search = try Search.submit(allocator, s, pool, rate, dec, ln.tiles, true);
            }
            if (pool) |p| p.start(ln.states.len, ln, Lane.feedDeferred) else for (0..ln.states.len) |b| ln.feedDeferred(b);
            lap.mark(.feedback);
        }
        if (side_left) {
            side_left = try side.?.step(side.?.ctx);
            lap.skip();
        }
    }
}

pub fn quantizeTensor(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    w: []const f32,
    h: []const f32,
    in: usize,
    out: usize,
    sign_in: []const f32,
    sign_out: []const f32,
    rate: format.Rate,
    dec: format.Decode,
    mode: quantize.OutScales,
    stages: ?*stages_mod.Stages,
) Error!quantize.Quantized {
    var result: [1]quantize.Quantized = undefined;
    try quantizeBatch(allocator, s, &.{.{ .w = w, .h = h, .sign_in = sign_in, .sign_out = sign_out }}, in, out, rate, dec, mode, stages, &result);
    return result[0];
}

pub fn quantizeNamed(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    w: []const f32,
    h: []const f32,
    in: usize,
    out: usize,
    seed: u64,
    name: []const u8,
    rate: format.Rate,
    dec: format.Decode,
    mode: quantize.OutScales,
    stages: ?*stages_mod.Stages,
) Error!quantize.Quantized {
    if (in == 0 or out == 0) return error.EmptyTensor;
    if (in % 128 != 0 or out % 128 != 0) return error.GeometryUnsupported;
    if (w.len != in * out or h.len != in * in) return error.ShapeMismatch;
    _ = try buildPlan(rate, dec, device_threads);
    const both = try quantize.signs(allocator, seed, name, in + out);
    defer allocator.free(both);
    return quantizeTensor(allocator, s, w, h, in, out, both[0..in], both[in..], rate, dec, mode, stages);
}

test "metal_search: invalid rate, window or slice length is refused by name" {
    const t = std.testing;
    const s = mlx.gpuStream();
    var tile: [tile_len]f32 = @splat(0);
    var out: [32]u16 = undefined;
    try t.expectError(error.RateUnsupported, encodeTiles(t.allocator, s, .{ .n = 22 }, .mcg, &tile, out[0..22], null));
    try t.expectError(error.RateUnsupported, encodeTiles(t.allocator, s, .{ .n = 33 }, .mcg, &tile, &out, null));
    try t.expectError(error.RateUnsupported, encodeTiles(t.allocator, s, .{ .n = 130 }, .mcg, &tile, &out, null));
    try t.expectError(error.WindowUnsupported, encodeTiles(t.allocator, s, .{ .n = 32 }, .{ .codebook = .mcg, .window = .w11 }, &tile, &out, null));
    try t.expectError(error.EmptyTensor, encodeTiles(t.allocator, s, .{ .n = 32 }, .mcg, tile[0..0], out[0..0], null));
    try t.expectError(error.ShapeMismatch, encodeTiles(t.allocator, s, .{ .n = 32 }, .mcg, tile[0..100], &out, null));
    try t.expectError(error.ShapeMismatch, encodeTiles(t.allocator, s, .{ .n = 32 }, .mcg, &tile, out[0..16], null));
}

test "metal_search: every admitted rate and window matches the CPU search code for code" {
    const t = std.testing;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(20261017);
    const r = prng.random();
    const n_tiles = 4;
    var tiles: [n_tiles * tile_len]f32 = undefined;
    for (0..tile_len * 2) |i| tiles[i] = r.floatNorm(f32);
    @memset(tiles[tile_len * 2 .. tile_len * 3], 0);
    for (tile_len * 3..tile_len * 4) |i| tiles[i] = r.floatNorm(f32) * (if (i % 7 == 0) @as(f32, 3.0) else @as(f32, 0.5));

    const rates = [_]u32{ 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 80, 96, 128 };
    const windows = [_]format.Window{ .w12, .w13, .w14, .w15, .w16 };
    for ([_]format.Codebook{ .mcg, .mul1 }) |cb| {
        for (windows) |win| {
            for (rates) |n| {
                if (win != .w12 and win != .w15 and !(win == .w16 and (n == 40 or n == 64)) and !(win == .w13 and n == 40) and !(win == .w14 and n == 48)) continue;
                const rate = format.Rate{ .n = n };
                const dec = format.Decode{ .codebook = cb, .window = win };
                var enc = try encode.Encoder.init(t.allocator, rate, dec);
                defer enc.deinit();

                const got_packed = try t.allocator.alloc(u16, n_tiles * n);
                defer t.allocator.free(got_packed);
                const got_recon = try t.allocator.alloc(f32, n_tiles * tile_len);
                defer t.allocator.free(got_recon);
                try encodeTiles(t.allocator, s, rate, dec, &tiles, got_packed, got_recon);

                for (0..n_tiles) |ti| {
                    var want_packed: [128]u16 = undefined;
                    var want_recon: [tile_len]f32 = undefined;
                    _ = try enc.encodeTile(tiles[ti * tile_len ..][0..tile_len], want_packed[0..n], &want_recon);
                    try t.expectEqualSlices(u16, want_packed[0..n], got_packed[ti * n ..][0..n]);
                    try t.expectEqualSlices(f32, &want_recon, got_recon[ti * tile_len ..][0..tile_len]);

                    var decoded: [tile_len]u16 = undefined;
                    format.decodeTile(got_packed[ti * n ..][0..n], rate, dec, &decoded);
                    for (got_recon[ti * tile_len ..][0..tile_len], decoded) |rv, dv| {
                        try t.expectEqual(format.f32ToF16Bits(rv), dv);
                    }
                }
            }
        }
    }
}

test "metal_search: the 1024-thread single-cost plan needs 1024 threads and a lower pipeline limit plans another" {
    const t = std.testing;
    const rate = format.Rate{ .n = 32 };
    const dec = format.Decode{ .codebook = .mcg, .window = .w15 };
    const wide = try buildPlan(rate, dec, 1024);
    try t.expect(wide.single_costs);
    try t.expectEqual(@as(u32, 1024), wide.threads);
    for ([_]u32{ 832, 640, 384, 256 }) |limit| {
        const plan = try buildPlan(rate, dec, limit);
        try t.expect(!plan.single_costs);
        try t.expect(plan.threads <= limit);
    }
}

test "metal_search: a pipeline limit below the launch size searches the same codes at a smaller threadgroup" {
    const t = std.testing;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(20261019);
    const r = prng.random();
    const n_tiles = 3;
    var tiles: [n_tiles * tile_len]f32 = undefined;
    for (&tiles) |*v| v.* = r.floatNorm(f32);
    const cases = [_]struct { n: u32, cb: format.Codebook, win: format.Window }{
        .{ .n = 32, .cb = .mcg, .win = .w15 },
        .{ .n = 40, .cb = .mul1, .win = .w15 },
        .{ .n = 32, .cb = .mcg, .win = .w12 },
    };
    // 832 and 384 are limits an M2 Pro reports. An M5 never refuses a plan, so the retry itself runs in the gpu_factor test.
    for (cases) |c| {
        const rate = format.Rate{ .n = c.n };
        const dec = format.Decode{ .codebook = c.cb, .window = c.win };
        var enc = try encode.Encoder.init(t.allocator, rate, dec);
        defer enc.deinit();
        for ([_]u32{ 2048, 832, 384, 128 }) |cap| {
            const got_packed = try t.allocator.alloc(u16, n_tiles * c.n);
            defer t.allocator.free(got_packed);
            const got_recon = try t.allocator.alloc(f32, n_tiles * tile_len);
            defer t.allocator.free(got_recon);
            var search = try Search.submitCapped(t.allocator, s, null, rate, dec, &tiles, true, cap);
            defer search.deinit();
            try search.collect(null, got_packed, got_recon);
            for (0..n_tiles) |ti| {
                var want_packed: [128]u16 = undefined;
                var want_recon: [tile_len]f32 = undefined;
                _ = try enc.encodeTile(tiles[ti * tile_len ..][0..tile_len], want_packed[0..c.n], &want_recon);
                try t.expectEqualSlices(u16, want_packed[0..c.n], got_packed[ti * c.n ..][0..c.n]);
                try t.expectEqualSlices(f32, &want_recon, got_recon[ti * tile_len ..][0..tile_len]);
            }
        }
    }
    try t.expect(!mlx.errorPending());
}

test "metal_search: full tensor LDLQ and global scale match the CPU quantizer code for code" {
    const t = std.testing;
    const s = mlx.gpuStream();
    var file = try st.parse(t.allocator, st.tensor);
    defer file.deinit(t.allocator);
    const fx_in = 256;
    const fx_out = 128;
    const w_raw = st.slice(f16, file.tensors.get("W").?);
    const h_raw = st.slice(f16, file.tensors.get("H").?);
    const w = try t.allocator.alloc(f32, fx_in * fx_out);
    defer t.allocator.free(w);
    for (w, w_raw) |*o, v| o.* = @floatCast(v);
    const h_full = try t.allocator.alloc(f32, fx_in * fx_in);
    defer t.allocator.free(h_full);
    for (h_full, h_raw) |*o, v| o.* = @floatCast(v);

    const Config = struct { name: []const u8, dec: format.Decode, n: u32, rows: usize, mode: quantize.OutScales };
    const configs = [_]Config{
        .{ .name = "mcg_n32_w12_r256_auto", .dec = .{ .codebook = .mcg, .window = .w12 }, .n = 32, .rows = 256, .mode = .auto },
        .{ .name = "mul1_n40_w12_r256_always", .dec = .{ .codebook = .mul1, .window = .w12 }, .n = 40, .rows = 256, .mode = .always },
        .{ .name = "mcg_n24_w12_r256_never", .dec = .{ .codebook = .mcg, .window = .w12 }, .n = 24, .rows = 256, .mode = .never },
        .{ .name = "mcg_n40_w15_r128_auto", .dec = .{ .codebook = .mcg, .window = .w15 }, .n = 40, .rows = 128, .mode = .auto },
        .{ .name = "mul1_n64_w12_r128_always", .dec = .{ .codebook = .mul1, .window = .w12 }, .n = 64, .rows = 128, .mode = .always },
    };
    const sign_out = st.slice(f32, file.tensors.get("sign_out").?);
    for (configs) |c| {
        const sign_in = st.slice(f32, file.tensors.get(if (c.rows == 256) "sign_in_r256" else "sign_in_r128").?);
        const h = try t.allocator.alloc(f32, c.rows * c.rows);
        defer t.allocator.free(h);
        for (0..c.rows) |i| @memcpy(h[i * c.rows ..][0..c.rows], h_full[i * fx_in ..][0..c.rows]);
        var q = try quantizeTensor(t.allocator, s, w[0 .. c.rows * fx_out], h, c.rows, fx_out, sign_in, sign_out, .{ .n = c.n }, c.dec, c.mode, null);
        defer q.deinit(t.allocator);

        var key: [96]u8 = undefined;
        const want_packed = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&key, "{s}.packed", .{c.name})).?);
        const want_suh = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&key, "{s}.suh", .{c.name})).?);
        const want_svh = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&key, "{s}.svh", .{c.name})).?);
        const want_g = st.slice(f32, file.tensors.get(try std.fmt.bufPrint(&key, "{s}.g", .{c.name})).?)[0];
        try t.expectEqualSlices(u16, want_packed, q.packed_out);
        try t.expectEqualSlices(u16, want_suh, q.suh);
        try t.expectEqualSlices(u16, want_svh, q.svh);
        try t.expectApproxEqAbs(want_g, q.g, 1e-5);
    }
}

test "metal_search: a Flash-Next sized tensor quantizes near its source at K1.5 and K2.5" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const in = 2560;
    const out = 640;
    var prng = std.Random.DefaultPrng.init(2560);
    const r = prng.random();
    const w = try t.allocator.alloc(f32, in * out);
    defer t.allocator.free(w);
    for (w) |*v| v.* = r.floatNorm(f32) * 0.02;
    const h = try t.allocator.alloc(f32, in * in);
    defer t.allocator.free(h);
    @memset(h, 0);
    for (0..in) |i| h[i * in + i] = 1;
    for ([_]u32{ 24, 40 }) |n| {
        var q = try quantizeNamed(t.allocator, s, w, h, in, out, 7, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.e0", .{ .n = n }, .{ .codebook = .mcg, .window = .w12 }, .auto, null);
        defer q.deinit(t.allocator);
        const decoded = try t.allocator.alloc(u16, in * out);
        defer t.allocator.free(decoded);
        try format.reconstructPublic(t.allocator, q.packed_out, q.suh, q.svh, in, out, .{ .n = n }, .{ .codebook = .mcg, .window = .w12 }, decoded);
        var num: f64 = 0;
        var den: f64 = 0;
        for (w, decoded) |wv, bits| {
            const d = @as(f64, format.f16BitsToF32(bits)) - wv;
            num += d * d;
            den += @as(f64, wv) * wv;
        }
        try t.expect(@sqrt(num / den) < 0.6);
    }
}

test "metal_search: a batch of tensors quantizes to the same bytes as one tensor at a time" {
    const t = std.testing;
    const s = mlx.gpuStream();
    var file = try st.parse(t.allocator, st.tensor);
    defer file.deinit(t.allocator);
    const in = 256;
    const out = 128;
    const w_raw = st.slice(f16, file.tensors.get("W").?);
    const h_raw = st.slice(f16, file.tensors.get("H").?);
    const batch = 3;
    var ws: [batch][]f32 = undefined;
    var signs_in: [batch][]f32 = undefined;
    var signs_out: [batch][]f32 = undefined;
    var jobs: [batch]TensorJob = undefined;
    const h = try t.allocator.alloc(f32, in * in);
    defer t.allocator.free(h);
    for (h, h_raw) |*o, v| o.* = @floatCast(v);
    for (0..batch) |b| {
        ws[b] = try t.allocator.alloc(f32, in * out);
        for (ws[b], w_raw, 0..) |*o, v, i| o.* = @as(f32, @floatCast(v)) * (1.0 + 0.25 * @as(f32, @floatFromInt(b))) + 0.001 * @as(f32, @floatFromInt((i * 7 + b) % 5));
        const both = try quantize.signs(t.allocator, 100 + b, "batch", in + out);
        signs_in[b] = both;
        signs_out[b] = both[in..];
        jobs[b] = .{ .w = ws[b], .h = h, .sign_in = both[0..in], .sign_out = both[in..] };
    }
    defer for (0..batch) |b| {
        t.allocator.free(ws[b]);
        t.allocator.free(signs_in[b]);
    };

    const rate = format.Rate{ .n = 32 };
    const dec = format.Decode{ .codebook = .mcg, .window = .w12 };
    var got: [batch]quantize.Quantized = undefined;
    try quantizeBatch(t.allocator, s, &jobs, in, out, rate, dec, .auto, null, &got);
    defer for (&got) |*q| q.deinit(t.allocator);
    for (jobs, got) |job, g| {
        var want = try quantizeTensor(t.allocator, s, job.w, job.h, in, out, job.sign_in, job.sign_out, rate, dec, .auto, null);
        defer want.deinit(t.allocator);
        try t.expectEqualSlices(u16, want.packed_out, g.packed_out);
        try t.expectEqualSlices(u16, want.suh, g.suh);
        try t.expectEqualSlices(u16, want.svh, g.svh);
        try t.expectEqual(want.g, g.g);
    }
}

test "metal_search: a tensor of several 128-row chunks quantizes to the CPU quantizer's bytes" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const in = 512;
    const out = 128;
    var prng = std.Random.DefaultPrng.init(512);
    const r = prng.random();
    const w = try t.allocator.alloc(f32, in * out);
    defer t.allocator.free(w);
    for (w) |*v| v.* = r.floatNorm(f32) * 0.02;
    const x = try t.allocator.alloc(f32, in * 24);
    defer t.allocator.free(x);
    for (x) |*v| v.* = r.floatNorm(f32);
    const h = try t.allocator.alloc(f32, in * in);
    defer t.allocator.free(h);
    for (0..in) |i| for (0..in) |j| {
        var acc: f32 = if (i == j) 0.05 else 0;
        for (0..24) |m| acc += x[i * 24 + m] * x[j * 24 + m];
        h[i * in + j] = acc;
    };
    const rate = format.Rate{ .n = 32 };
    const dec = format.Decode{ .codebook = .mcg, .window = .w12 };
    var want = try quantize.quantizeNamed(t.allocator, w, h, in, out, 5, "chunks", rate, dec, .auto);
    defer want.deinit(t.allocator);
    var got = try quantizeNamed(t.allocator, s, w, h, in, out, 5, "chunks", rate, dec, .auto, null);
    defer got.deinit(t.allocator);
    try t.expectEqualSlices(u16, want.packed_out, got.packed_out);
    try t.expectEqualSlices(u16, want.suh, got.suh);
    try t.expectEqualSlices(u16, want.svh, got.svh);
}

/// tr((W - What)^T H (W - What)) / tr(W^T H W): the error LDLQ minimizes, in the public basis.
fn hessianError(allocator: std.mem.Allocator, w: []const f32, decoded: []const u16, h: []const f32, in: usize, out: usize) !f64 {
    const d = try allocator.alloc(f64, in * out);
    defer allocator.free(d);
    for (d, w, decoded) |*o, wv, bits| o.* = @as(f64, format.f16BitsToF32(bits)) - wv;
    var num: f64 = 0;
    var den: f64 = 0;
    for (0..in) |i| for (0..in) |k| {
        const hv: f64 = h[i * in + k];
        if (hv == 0) continue;
        for (0..out) |c| {
            num += d[i * out + c] * hv * d[k * out + c];
            den += @as(f64, w[i * out + c]) * hv * w[k * out + c];
        }
    };
    return num / den;
}

test "metal_search: a float32 GPU factor quantizes as well as the float64 factor" {
    const t = std.testing;
    const s = mlx.gpuStream();
    var file = try st.parse(t.allocator, st.tensor);
    defer file.deinit(t.allocator);
    const in = 256;
    const out = 128;
    const w_raw = st.slice(f16, file.tensors.get("W").?);
    const h_raw = st.slice(f16, file.tensors.get("H").?);
    const h = try t.allocator.alloc(f32, in * in);
    defer t.allocator.free(h);
    for (h, h_raw) |*o, v| o.* = @floatCast(v);
    const w = try t.allocator.alloc(f32, in * out);
    defer t.allocator.free(w);
    for (w, w_raw) |*o, v| o.* = @floatCast(v);
    const both = try quantize.signs(t.allocator, 5, "factor32", in + out);
    defer t.allocator.free(both);
    const jobs = [_]TensorJob{.{ .w = w, .h = h, .sign_in = both[0..in], .sign_out = both[in..] }};
    const rate = format.Rate{ .n = 32 };
    const dec = format.Decode{ .codebook = .mcg, .window = .w12 };

    var want: [1]quantize.Quantized = undefined;
    try quantizeBatch(t.allocator, s, &jobs, in, out, rate, dec, .auto, null, &want);
    defer want[0].deinit(t.allocator);
    var prepared = try prepareBatch(t.allocator, null, &jobs, in, out, .auto, .f32);
    defer prepared.deinit();
    var got: [1]quantize.Quantized = undefined;
    try finishBatch(t.allocator, s, null, &prepared, rate, dec, null, &got, null);
    defer got[0].deinit(t.allocator);
    try t.expectEqual(@as(usize, 0), prepared.factor_stats.fallbacks);

    var errs: [2]f64 = undefined;
    for ([_]quantize.Quantized{ want[0], got[0] }, &errs) |q, *e| {
        const decoded = try t.allocator.alloc(u16, in * out);
        defer t.allocator.free(decoded);
        try format.reconstructPublic(t.allocator, q.packed_out, q.suh, q.svh, in, out, rate, dec, decoded);
        e.* = try hessianError(t.allocator, w, decoded, h, in, out);
    }
    try t.expect(errs[1] <= errs[0] * 1.02);
}

test "metal_search: the blocked feedback update equals the row-at-a-time loop bit for bit" {
    const t = std.testing;
    var prng = std.Random.DefaultPrng.init(91);
    const r = prng.random();
    const rows = 7;
    const cols = 64;
    const chunk = 128;
    const prod = try t.allocator.alloc(f32, rows * cols);
    defer t.allocator.free(prod);
    const want = try t.allocator.alloc(f32, rows * cols);
    defer t.allocator.free(want);
    const err = try t.allocator.alloc(f32, chunk * cols);
    defer t.allocator.free(err);
    const lt = try t.allocator.alloc(f32, rows * chunk);
    defer t.allocator.free(lt);
    for (prod, want) |*p, *w| {
        p.* = r.floatNorm(f32);
        w.* = p.*;
    }
    for (err) |*v| v.* = r.floatNorm(f32) * 1e-2;
    for (lt) |*v| v.* = r.floatNorm(f32) * 3;
    for (0..chunk) |m| for (0..rows) |row| {
        const lv = lt[row * chunk + m];
        for (want[row * cols ..][0..cols], err[m * cols ..][0..cols]) |*c, e| c.* += lv * e;
    };
    feedbackUpdate(prod, cols, err, lt, chunk);
    for (want, prod) |a, b| try t.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
}

test "metal_search: search throughput (SUSHI_ENCODER_SEARCH_UBENCH)" {
    if (std.c.getenv("SUSHI_ENCODER_SEARCH_UBENCH") == null) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    const n_tiles: usize = 4096;
    var prng = std.Random.DefaultPrng.init(4096);
    const r = prng.random();
    const tiles = try t.allocator.alloc(f32, n_tiles * tile_len);
    defer t.allocator.free(tiles);
    for (tiles) |*v| v.* = r.floatNorm(f32);
    const rate = format.Rate{ .n = 40 };
    const dec = format.Decode{ .codebook = .mcg, .window = .w15 };
    const packed_out = try t.allocator.alloc(u16, n_tiles * rate.n);
    defer t.allocator.free(packed_out);
    const recon = try t.allocator.alloc(f32, n_tiles * tile_len);
    defer t.allocator.free(recon);
    try encodeTiles(t.allocator, s, rate, dec, tiles, packed_out, recon);
    // Sustained, not burst: a conversion keeps the GPU busy for minutes.
    const t0 = stages_mod.nowNs();
    var calls: usize = 0;
    while (stages_mod.nowNs() - t0 < 20 * std.time.ns_per_s) : (calls += 1) try encodeTiles(t.allocator, s, rate, dec, tiles, packed_out, recon);
    const us = @as(f64, @floatFromInt(stages_mod.nowNs() - t0)) / 1e3 / @as(f64, @floatFromInt(calls * n_tiles));
    std.debug.print("search K2.5 mcg w15: {d:.2} us/tile sustained over {d} tiles\n", .{ us, calls * n_tiles });
}
