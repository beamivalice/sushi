//! Hy4Flash precompiled custom kernels catalog (`sushi.metallib`).
const std = @import("std");
const custom = @import("../tensor/metallib_custom.zig");
const exl3 = @import("sushi_exl3");
const sparse = @import("../sparse_select.zig");

const hy4_attn = @import("hy4_attn.zig");
const hy4_hc = @import("hy4_hc.zig");
const hy4_row = @import("hy4_row.zig");
const hy4_gemv = @import("hy4_gemv.zig");
const hy4_norm = @import("hy4_norm.zig");
const hy4_rope = @import("hy4_rope.zig");
const hy4_absorb = @import("hy4_absorb.zig");
const hy4_engram_gate = @import("hy4_engram_gate.zig");
const hy4_chain = @import("hy4_chain.zig");
const hy4_embed = @import("hy4_embed.zig");
const hy4_expert_mv = @import("hy4_expert_mv.zig");
const hy4_front = @import("hy4_front.zig");
const hy4_index = @import("hy4_index.zig");
const hy4_latent = @import("hy4_latent.zig");
const hy4_route = @import("hy4_route.zig");

const KernelSpec = custom.KernelSpec;
const ParamSpec = custom.ParamSpec;
const InputSlot = custom.InputSlot;
const OutputSlot = custom.OutputSlot;
const CtVal = custom.CtVal;
const Dtype = custom.Dtype;

const f32_bf16: []const Dtype = &.{ .float32, .bfloat16 };
const bool_f32: []const Dtype = &.{ .bool_, .float32 };
const u32_i32: []const Dtype = &.{ .uint32, .int32 };
const map_dts: []const Dtype = &.{ .int32, .uint32, .float32, .bfloat16 };
const rows_1_16: []const i32 = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
const rows_1_8: []const i32 = &.{ 1, 2, 3, 4, 5, 6, 7, 8 };

const hc_narrow_td = [_][2]i32{
    .{ 1024, 128 }, .{ 512, 128 }, .{ 256, 128 }, .{ 128, 128 }, .{ 64, 128 },
    .{ 1024, 256 }, .{ 512, 256 }, .{ 256, 256 },
    .{ 1024, 384 }, .{ 512, 384 }, .{ 256, 384 },
    .{ 1024, 512 }, .{ 512, 512 }, .{ 256, 512 },
    .{ 1024, 3072 }, .{ 512, 3072 }, .{ 256, 3072 }, .{ 128, 3072 },
};
const hc_wide_td = [_][2]i32{
    .{ 1024, 128 }, .{ 512, 128 }, .{ 256, 128 },
    .{ 1024, 256 }, .{ 512, 256 },
    .{ 1024, 384 }, .{ 512, 384 },
    .{ 1024, 512 }, .{ 512, 512 },
    .{ 1024, 3072 }, .{ 512, 3072 }, .{ 256, 3072 },
};

fn hcTuples(comptime wide: bool, comptime n_dt: usize) []const []const CtVal {
    const tds = if (wide) hc_wide_td else hc_narrow_td;
    const n_dt_combos: usize = @as(usize, 1) << @intCast(n_dt);
    const total = tds.len * n_dt_combos;
    comptime {
        @setEvalBranchQuota(20000);
        var out: [total][]const CtVal = undefined;
        var idx: usize = 0;
        for (tds) |td| {
            for (0..n_dt_combos) |mask| {
                var row: [4 + n_dt]CtVal = undefined;
                row[0] = .{ .int = td[0] };
                row[1] = .{ .int = 4 };
                row[2] = .{ .int = 8 };
                row[3] = .{ .int = td[1] };
                for (0..n_dt) |b| {
                    row[4 + b] = .{ .dtype = if (((mask >> @intCast(b)) & 1) == 0) .float32 else .bfloat16 };
                }
                const frozen = row;
                out[idx] = &frozen;
                idx += 1;
            }
        }
        const final = out;
        return &final;
    }
}

fn hcSpec(comptime f: hy4_hc.Form, comptime wide: bool) KernelSpec {
    const name = switch (f) {
        .plain => "sushi_hy4_hc_mix_norm",
        .expand => "sushi_hy4_hc_mix_expand_norm",
        .back => "sushi_hy4_hc_back_mix_norm",
        .back_inner => "sushi_hy4_hc_back_inner_mix_norm",
        .expand_norm => "sushi_hy4_hc_expand_norm_mix_norm",
        .expand_parts => "sushi_hy4_hc_expand_parts_mix_norm",
        .expand_rows => "sushi_hy4_hc_expand_rows_mix_norm",
    } ++ (if (wide) "_wide" else "");

    const ins: []const InputSlot = switch (f) {
        .plain => &.{
            .{ .name = "x", .ty = .{ .fixed = .float32 } },
            .{ .name = "fn", .ty = .{ .fixed = .float32 } },
            .{ .name = "scale", .ty = .{ .fixed = .float32 } },
            .{ .name = "base", .ty = .{ .param = "TBASE" } },
            .{ .name = "eps", .ty = .{ .fixed = .float32 } },
            .{ .name = "w", .ty = .{ .param = "TW" } },
        },
        .expand => &.{
            .{ .name = "x", .ty = .{ .fixed = .float32 } },
            .{ .name = "y", .ty = .{ .param = "TY" } },
            .{ .name = "pin", .ty = .{ .fixed = .float32 } },
            .{ .name = "fn", .ty = .{ .fixed = .float32 } },
            .{ .name = "scale", .ty = .{ .fixed = .float32 } },
            .{ .name = "base", .ty = .{ .param = "TBASE" } },
            .{ .name = "eps", .ty = .{ .fixed = .float32 } },
            .{ .name = "w", .ty = .{ .param = "TW" } },
        },
        .back => &.{
            .{ .name = "x", .ty = .{ .fixed = .float32 } },
            .{ .name = "pin", .ty = .{ .fixed = .float32 } },
            .{ .name = "a", .ty = .{ .param = "TA" } },
            .{ .name = "b", .ty = .{ .param = "TB" } },
            .{ .name = "wn", .ty = .{ .param = "TW" } },
            .{ .name = "fn", .ty = .{ .fixed = .float32 } },
            .{ .name = "scale", .ty = .{ .fixed = .float32 } },
            .{ .name = "base", .ty = .{ .param = "TBASE" } },
            .{ .name = "eps", .ty = .{ .fixed = .float32 } },
            .{ .name = "w", .ty = .{ .param = "TW" } },
        },
        .back_inner => &.{
            .{ .name = "x", .ty = .{ .fixed = .float32 } },
            .{ .name = "pin", .ty = .{ .fixed = .float32 } },
            .{ .name = "a", .ty = .{ .fixed = .float16 } },
            .{ .name = "asvh", .ty = .{ .fixed = .float16 } },
            .{ .name = "b", .ty = .{ .fixed = .float16 } },
            .{ .name = "bsvh", .ty = .{ .fixed = .float16 } },
            .{ .name = "wn", .ty = .{ .param = "TW" } },
            .{ .name = "fn", .ty = .{ .fixed = .float32 } },
            .{ .name = "scale", .ty = .{ .fixed = .float32 } },
            .{ .name = "base", .ty = .{ .param = "TBASE" } },
            .{ .name = "eps", .ty = .{ .fixed = .float32 } },
            .{ .name = "w", .ty = .{ .param = "TW" } },
        },
        .expand_norm => &.{
            .{ .name = "x", .ty = .{ .fixed = .float32 } },
            .{ .name = "pin", .ty = .{ .fixed = .float32 } },
            .{ .name = "a", .ty = .{ .param = "TA" } },
            .{ .name = "wn", .ty = .{ .param = "TW" } },
            .{ .name = "fn", .ty = .{ .fixed = .float32 } },
            .{ .name = "scale", .ty = .{ .fixed = .float32 } },
            .{ .name = "base", .ty = .{ .param = "TBASE" } },
            .{ .name = "eps", .ty = .{ .fixed = .float32 } },
            .{ .name = "w", .ty = .{ .param = "TW" } },
        },
        .expand_parts => &.{
            .{ .name = "x", .ty = .{ .fixed = .float32 } },
            .{ .name = "op", .ty = .{ .fixed = .float32 } },
            .{ .name = "osvh", .ty = .{ .fixed = .float16 } },
            .{ .name = "ow", .ty = .{ .param = "TW" } },
            .{ .name = "pin", .ty = .{ .fixed = .float32 } },
            .{ .name = "fn", .ty = .{ .fixed = .float32 } },
            .{ .name = "scale", .ty = .{ .fixed = .float32 } },
            .{ .name = "base", .ty = .{ .param = "TBASE" } },
            .{ .name = "eps", .ty = .{ .fixed = .float32 } },
            .{ .name = "w", .ty = .{ .param = "TW" } },
        },
        .expand_rows => &.{
            .{ .name = "x", .ty = .{ .fixed = .float32 } },
            .{ .name = "oy", .ty = .{ .fixed = .float32 } },
            .{ .name = "ow", .ty = .{ .param = "TW" } },
            .{ .name = "pin", .ty = .{ .fixed = .float32 } },
            .{ .name = "fn", .ty = .{ .fixed = .float32 } },
            .{ .name = "scale", .ty = .{ .fixed = .float32 } },
            .{ .name = "base", .ty = .{ .param = "TBASE" } },
            .{ .name = "eps", .ty = .{ .fixed = .float32 } },
            .{ .name = "w", .ty = .{ .param = "TW" } },
        },
    };

    const outs: []const OutputSlot = if (f == .plain) (if (wide) &.{
        .{ .name = "post", .ty = .{ .fixed = .float32 } },
        .{ .name = "h_out", .ty = .{ .fixed = .float32 } },
        .{ .name = "ctr", .ty = .{ .fixed = .uint32 } },
    } else &.{
        .{ .name = "post", .ty = .{ .fixed = .float32 } },
        .{ .name = "h_out", .ty = .{ .fixed = .float32 } },
    }) else (if (wide) &.{
        .{ .name = "post", .ty = .{ .fixed = .float32 } },
        .{ .name = "h_out", .ty = .{ .fixed = .float32 } },
        .{ .name = "m", .ty = .{ .fixed = .float32 } },
        .{ .name = "ctr", .ty = .{ .fixed = .uint32 } },
    } else &.{
        .{ .name = "post", .ty = .{ .fixed = .float32 } },
        .{ .name = "h_out", .ty = .{ .fixed = .float32 } },
        .{ .name = "m", .ty = .{ .fixed = .float32 } },
    });

    const base_params: []const ParamSpec = &.{
        .{ .name = "TG", .kind = .ct_int },
        .{ .name = "HC", .kind = .ct_int },
        .{ .name = "MIXN", .kind = .ct_int },
        .{ .name = "D", .kind = .ct_int },
    };
    const dt_params: []const ParamSpec = switch (f) {
        .plain, .back_inner, .expand_parts, .expand_rows => &.{
            .{ .name = "TBASE", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "TW", .kind = .io_dtype, .dtypes = f32_bf16 },
        },
        .expand => &.{
            .{ .name = "TY", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "TBASE", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "TW", .kind = .io_dtype, .dtypes = f32_bf16 },
        },
        .expand_norm => &.{
            .{ .name = "TA", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "TBASE", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "TW", .kind = .io_dtype, .dtypes = f32_bf16 },
        },
        .back => &.{
            .{ .name = "TA", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "TB", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "TBASE", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "TW", .kind = .io_dtype, .dtypes = f32_bf16 },
        },
    };
    const ctr_params: []const ParamSpec = if (wide) &.{.{ .name = "CTR", .kind = .fc_int }} else &.{};
    const extra_params: []const ParamSpec = switch (f) {
        .plain, .expand => &.{},
        .back, .back_inner, .expand_norm => &.{.{ .name = "EPS", .kind = .fc_int }},
        .expand_rows => &.{.{ .name = "OEPS", .kind = .fc_int }},
        .expand_parts => &.{ .{ .name = "OSPLIT", .kind = .fc_int }, .{ .name = "OEPS", .kind = .fc_int } },
    };

    return .{
        .name = name,
        .header = if (custom.is_codegen and (f == .expand_parts or f == .back_inner)) exl3.project_rows.ROWS_HELPERS else "",
        .source = if (custom.is_codegen) hy4_hc.kernelSource(f, wide) else "",
        .inputs = ins,
        .outputs = outs,
        .params = base_params ++ dt_params ++ ctr_params ++ extra_params,
        .ct_tuples = hcTuples(wide, dt_params.len),
    };
}

// Steel attention (NPART, U) pairs: `NPART` in {1,2,4,8,16,32}, `U <= max(1, 16/NPART)` even (plus 1).
const steel_nu_pairs = [_][2]i32{
    .{ 1, 1 }, .{ 1, 2 }, .{ 1, 4 }, .{ 1, 6 }, .{ 1, 8 }, .{ 1, 10 }, .{ 1, 12 }, .{ 1, 14 }, .{ 1, 16 },
    .{ 2, 1 }, .{ 2, 2 }, .{ 2, 4 }, .{ 2, 6 }, .{ 2, 8 },
    .{ 4, 1 }, .{ 4, 2 }, .{ 4, 4 },
    .{ 8, 1 }, .{ 8, 2 },
    .{ 16, 1 },
    .{ 32, 1 },
};

const steel_val_tuples: []const []const CtVal = blk: {
    var out: [steel_nu_pairs.len][]const CtVal = undefined;
    for (steel_nu_pairs, 0..) |p, i| {
        const row = [2]CtVal{ .{ .int = p[0] }, .{ .int = p[1] } };
        out[i] = &row;
    }
    const final = out;
    break :blk &final;
};

const steel_score_tuples: []const []const CtVal = blk: {
    const rs = [_]i32{ 16, 32, 64, 128 };
    var out: [rs.len * steel_nu_pairs.len][]const CtVal = undefined;
    var i: usize = 0;
    for (rs) |r| {
        for (steel_nu_pairs) |p| {
            const row = [3]CtVal{ .{ .int = r }, .{ .int = p[0] }, .{ .int = p[1] } };
            out[i] = &row;
            i += 1;
        }
    }
    const final = out;
    break :blk &final;
};

const index_score_tuples: []const []const CtVal = blk: {
    const shapes = [_][3]i32{
        .{ 256, 16, 128 },
        .{ 256, 4, 32 },
        .{ 256, 4, 64 },
        .{ 256, 8, 64 },
        .{ 256, 8, 128 },
    };
    var out: [shapes.len * 2][]const CtVal = undefined;
    var i: usize = 0;
    for (shapes) |sh| {
        for (f32_bf16) |dt| {
            const row = [4]CtVal{ .{ .int = sh[0] }, .{ .int = sh[1] }, .{ .int = sh[2] }, .{ .dtype = dt } };
            out[i] = &row;
            i += 1;
        }
    }
    const final = out;
    break :blk &final;
};

fn engramHeadsSource(a: std.mem.Allocator, nh: usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "auto hw = ");
    for (0..nh) |i| {
        if (i + 1 < nh) {
            try out.print(a, "(threadgroup_position_in_grid.x % uint(NH)) == {d} ? w{d} : ", .{ i, i });
        } else {
            try out.print(a, "w{d};\n", .{i});
        }
    }
    try out.appendSlice(a, hy4_engram_gate.heads_body);
    return out.toOwnedSlice(a);
}

fn engramDeltaSource(a: std.mem.Allocator, nc: usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "#pragma clang fp contract(off)\n");
    inline for (.{ .{ "d", "ds" }, .{ "s", "sc" } }) |pair| {
        try out.print(a, "auto {s} = ", .{pair[1]});
        for (0..nc) |i| {
            if (i + 1 < nc) {
                try out.print(a, "thread_position_in_grid.y == {d} ? {s}{d} : ", .{ i, pair[0], i });
            } else {
                try out.print(a, "{s}{d};\n", .{ pair[0], i });
            }
        }
    }
    try out.appendSlice(a, hy4_engram_gate.delta_body);
    return out.toOwnedSlice(a);
}

fn engramHeadsInputs(comptime nh: usize) []const InputSlot {
    comptime {
        var ins: [1 + nh]InputSlot = undefined;
        ins[0] = .{ .name = "x", .ty = .{ .param = "T" } };
        for (0..nh) |i| {
            ins[1 + i] = .{ .name = std.fmt.comptimePrint("w{d}", .{i}), .ty = .{ .param = "TW" } };
        }
        const final = ins;
        return &final;
    }
}

fn engramDeltaInputs(comptime nc: usize) []const InputSlot {
    comptime {
        var ins: [5 + 2 * nc]InputSlot = undefined;
        ins[0] = .{ .name = "h", .ty = .{ .param = "TH" } };
        ins[1] = .{ .name = "v", .ty = .{ .fixed = .float32 } };
        ins[2] = .{ .name = "vsum", .ty = .{ .fixed = .float32 } };
        ins[3] = .{ .name = "gamma", .ty = .{ .param = "TG" } };
        ins[4] = .{ .name = "live", .ty = .{ .fixed = .float32 } };
        for (0..nc) |i| {
            ins[5 + i] = .{ .name = std.fmt.comptimePrint("d{d}", .{i}), .ty = .{ .fixed = .float32 } };
            ins[5 + nc + i] = .{ .name = std.fmt.comptimePrint("s{d}", .{i}), .ty = .{ .param = "TS" } };
        }
        const final = ins;
        return &final;
    }
}

pub const specs: []const KernelSpec = &.{
    // ── hy4_attn (NAX) ──────────────────────────────────────────────────────────
    .{
        .name = "sushi_hy4_attn_range",
        .nax = true,
        .header = hy4_attn.header,
        .source = hy4_attn.range_source,
        .inputs = &.{
            .{ .name = "ql", .ty = .{ .fixed = .float32 } },
            .{ .name = "qp", .ty = .{ .fixed = .float32 } },
            .{ .name = "kc", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "kp", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "sink", .ty = .{ .param = "TSINK" } },
            .{ .name = "meta", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "L", .kind = .fc_int },
            .{ .name = "R", .kind = .fc_int },
            .{ .name = "HB", .kind = .ct_int, .ints = &.{16} },
            .{ .name = "KT", .kind = .ct_int, .ints = &.{32} },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{ 2, 3, 5, 9 } },
            .{ .name = "TSINK", .kind = .io_dtype, .dtypes = f32_bf16 },
        },
    },
    .{
        .name = "sushi_hy4_attn_picks",
        .nax = true,
        .header = hy4_attn.header,
        .source = hy4_attn.picks_source,
        .inputs = &.{
            .{ .name = "ql", .ty = .{ .fixed = .float32 } },
            .{ .name = "qp", .ty = .{ .fixed = .float32 } },
            .{ .name = "kc", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "kp", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "rows", .ty = .{ .fixed = .uint32 } },
            .{ .name = "keep", .ty = .{ .fixed = .bool_ } },
            .{ .name = "sink", .ty = .{ .param = "TSINK" } },
            .{ .name = "meta", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "L", .kind = .fc_int },
            .{ .name = "R", .kind = .fc_int },
            .{ .name = "HB", .kind = .ct_int, .ints = &.{16} },
            .{ .name = "KT", .kind = .ct_int, .ints = &.{32} },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{ 2, 3, 5, 9 } },
            .{ .name = "TSINK", .kind = .io_dtype, .dtypes = f32_bf16 },
        },
    },

    // ── hy4_hc (14 SIMD kernels: 7 forms × narrow/wide) ─────────────────────────
    hcSpec(.plain, false),
    hcSpec(.plain, true),
    hcSpec(.expand, false),
    hcSpec(.expand, true),
    hcSpec(.back, false),
    hcSpec(.back, true),
    hcSpec(.back_inner, false),
    hcSpec(.back_inner, true),
    hcSpec(.expand_norm, false),
    hcSpec(.expand_norm, true),
    hcSpec(.expand_parts, false),
    hcSpec(.expand_parts, true),
    hcSpec(.expand_rows, false),
    hcSpec(.expand_rows, true),

    // ── hy4_row ─────────────────────────────────────────────────────────────────
    .{
        .name = "sushi_hy4_row_f32",
        .source = hy4_row.source,
        .inputs = &.{
            .{ .name = "w", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "x", .ty = .{ .fixed = .float32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "TM", .kind = .ct_int, .ints = &.{2} },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{2} },
            .{ .name = "R", .kind = .ct_int, .ints = rows_1_16 },
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "N", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_hy4_row_f32_wide",
        .source = hy4_row.wide_source,
        .inputs = &.{
            .{ .name = "w", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "x", .ty = .{ .fixed = .float32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "R", .kind = .ct_int, .ints = rows_1_16 },
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "N", .kind = .fc_int },
        },
    },

    // ── hy4_gemv ────────────────────────────────────────────────────────────────
    .{
        .name = "sushi_hy4_gemv_thru",
        .source = hy4_gemv.source,
        .inputs = &.{
            .{ .name = "w", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "x", .ty = .{ .fixed = .float32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "R", .kind = .ct_int, .ints = rows_1_16 },
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "N", .kind = .fc_int },
        },
    },

    // ── hy4_norm ────────────────────────────────────────────────────────────────
    .{
        .name = "sushi_hy4_rms",
        .source = hy4_norm.source,
        .inputs = &.{
            .{ .name = "x", .ty = .{ .param = "T" } },
            .{ .name = "y", .ty = .{ .param = "TY" } },
            .{ .name = "z", .ty = .{ .param = "TZ" } },
            .{ .name = "w", .ty = .{ .param = "TW" } },
            .{ .name = "hs", .ty = .{ .param = "THP" } },
            .{ .name = "post", .ty = .{ .param = "THP" } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .param = "OT" } }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = f32_bf16 },
            .{ .name = "TY", .kind = .ct_dtype, .dtypes = f32_bf16 },
            .{ .name = "TZ", .kind = .ct_dtype, .dtypes = f32_bf16 },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = f32_bf16 },
            .{ .name = "TW", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "THP", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "D", .kind = .fc_int },
            .{ .name = "EPS", .kind = .fc_int },
            .{ .name = "ADD", .kind = .fc_bool },
            .{ .name = "YROW", .kind = .fc_bool },
            .{ .name = "MUL", .kind = .fc_bool },
        },
    },
    .{
        .name = "sushi_hy4_rms_expand_inner",
        .source = hy4_norm.expand_inner_source,
        .inputs = &.{
            .{ .name = "a", .ty = .{ .fixed = .float16 } },
            .{ .name = "asvh", .ty = .{ .fixed = .float16 } },
            .{ .name = "b", .ty = .{ .fixed = .float16 } },
            .{ .name = "bsvh", .ty = .{ .fixed = .float16 } },
            .{ .name = "w", .ty = .{ .param = "TW" } },
            .{ .name = "hs", .ty = .{ .fixed = .float32 } },
            .{ .name = "post", .ty = .{ .fixed = .float32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "TW", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "D", .kind = .fc_int },
            .{ .name = "TG", .kind = .fc_int },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "EPS", .kind = .fc_int },
        },
    },

    // ── hy4_rope ────────────────────────────────────────────────────────────────
    .{
        .name = "sushi_hy4_rope_tail",
        .source = hy4_rope.source,
        .inputs = &.{
            .{ .name = "x", .ty = .{ .param = "T" } },
            .{ .name = "pos", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .param = "OT" } }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = f32_bf16 },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = f32_bf16 },
            .{ .name = "NOPE", .kind = .fc_int },
            .{ .name = "HALF", .kind = .fc_int },
            .{ .name = "BASE", .kind = .fc_int },
            .{ .name = "SCALE", .kind = .fc_int },
            .{ .name = "ROW_MAJOR", .kind = .fc_bool },
            .{ .name = "PER_ROW", .kind = .fc_bool },
        },
    },

    // ── hy4_absorb ──────────────────────────────────────────────────────────────
    .{
        .name = "sushi_hy4_absorb_q",
        .source = hy4_absorb.q_source,
        .inputs = &.{
            .{ .name = "kvb", .ty = .{ .param = "TW" } },
            .{ .name = "q", .ty = .{ .fixed = .float32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "TW", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "RT", .kind = .ct_int, .ints = rows_1_8 },
            .{ .name = "NSG", .kind = .fc_int },
            .{ .name = "HEADS", .kind = .fc_int },
            .{ .name = "ROWS", .kind = .fc_int },
            .{ .name = "NOPE", .kind = .fc_int },
            .{ .name = "VDIM", .kind = .ignored },
            .{ .name = "LAT", .kind = .fc_int },
            .{ .name = "QS", .kind = .ignored },
            .{ .name = "R", .kind = .fc_int },
            .{ .name = "ROW_MAJOR", .kind = .fc_bool },
        },
    },
    .{
        .name = "sushi_hy4_absorb_v",
        .source = hy4_absorb.v_source,
        .inputs = &.{
            .{ .name = "kvb", .ty = .{ .param = "TW" } },
            .{ .name = "u", .ty = .{ .fixed = .float32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "TW", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "RT", .kind = .ct_int, .ints = rows_1_8 },
            .{ .name = "NSG", .kind = .ignored },
            .{ .name = "HEADS", .kind = .fc_int },
            .{ .name = "ROWS", .kind = .fc_int },
            .{ .name = "NOPE", .kind = .fc_int },
            .{ .name = "VDIM", .kind = .fc_int },
            .{ .name = "LAT", .kind = .fc_int },
            .{ .name = "QS", .kind = .ignored },
            .{ .name = "R", .kind = .fc_int },
            .{ .name = "ROW_MAJOR", .kind = .ignored },
        },
    },
    .{
        .name = "sushi_hy4_absorb_v_gated",
        .header = hy4_absorb.sigmoid_header,
        .source = hy4_absorb.v_source_gated,
        .inputs = &.{
            .{ .name = "kvb", .ty = .{ .param = "TW" } },
            .{ .name = "u", .ty = .{ .fixed = .float32 } },
            .{ .name = "gate", .ty = .{ .param = "TG" } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "TW", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "RT", .kind = .ct_int, .ints = rows_1_8 },
            .{ .name = "TG", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "NSG", .kind = .ignored },
            .{ .name = "HEADS", .kind = .fc_int },
            .{ .name = "ROWS", .kind = .fc_int },
            .{ .name = "NOPE", .kind = .fc_int },
            .{ .name = "VDIM", .kind = .fc_int },
            .{ .name = "LAT", .kind = .fc_int },
            .{ .name = "QS", .kind = .ignored },
            .{ .name = "R", .kind = .fc_int },
            .{ .name = "ROW_MAJOR", .kind = .ignored },
        },
    },

    // ── hy4_engram_gate ─────────────────────────────────────────────────────────
    .{
        .name = "sushi_hy4_engram_heads",
        .disambig = "nh4",
        .header = if (custom.is_codegen) hy4_engram_gate.header else "",
        .source_fn = custom.codegenFn(engramHeadsSource),
        .source_arg = 4,
        .inputs = engramHeadsInputs(4),
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = f32_bf16 },
            .{ .name = "TW", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "D", .kind = .fc_int },
            .{ .name = "NH", .kind = .fc_int },
            .{ .name = "EPS", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_hy4_engram_heads",
        .disambig = "nh16",
        .header = if (custom.is_codegen) hy4_engram_gate.header else "",
        .source_fn = custom.codegenFn(engramHeadsSource),
        .source_arg = 16,
        .inputs = engramHeadsInputs(16),
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = f32_bf16 },
            .{ .name = "TW", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "D", .kind = .fc_int },
            .{ .name = "NH", .kind = .fc_int },
            .{ .name = "EPS", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_hy4_engram_value",
        .header = if (custom.is_codegen) hy4_engram_gate.header else "",
        .source = if (custom.is_codegen) hy4_engram_gate.value_source else "",
        .inputs = &.{
            .{ .name = "lin", .ty = .{ .fixed = .float32 } },
            .{ .name = "b", .ty = .{ .param = "TB" } },
        },
        .outputs = &.{
            .{ .name = "out_v", .ty = .{ .fixed = .float32 } },
            .{ .name = "out_sq", .ty = .{ .fixed = .float32 } },
        },
        .params = &.{
            .{ .name = "TB", .kind = .io_dtype, .dtypes = f32_bf16 },
        },
    },
    .{
        .name = "sushi_hy4_engram_delta",
        .disambig = "nc4",
        .header = if (custom.is_codegen) hy4_engram_gate.header else "",
        .source_fn = custom.codegenFn(engramDeltaSource),
        .source_arg = 4,
        .inputs = engramDeltaInputs(4),
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "TH", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "TG", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "TS", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "D", .kind = .fc_int },
            .{ .name = "NC", .kind = .fc_int },
            .{ .name = "EPS", .kind = .fc_int },
            .{ .name = "INV_N", .kind = .fc_int },
            .{ .name = "INV_SQRT_D", .kind = .fc_int },
        },
    },

    // ── hy4_chain (14 SIMD kernels) ─────────────────────────────────────────────
    .{
        .name = "sushi_hy4_swiglu",
        .header = hy4_chain.header,
        .source = hy4_chain.swiglu_source,
        .inputs = &.{
            .{ .name = "g", .ty = .{ .param = "TG" } },
            .{ .name = "u", .ty = .{ .param = "TU" } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .param = "OT" } }},
        .params = &.{
            .{ .name = "TG", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "TU", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = f32_bf16 },
        },
    },
    .{
        .name = "sushi_hy4_swiglu_finish_prepare",
        .header = hy4_chain.header,
        .source = hy4_chain.swiglu_finish_prepare_source,
        .inputs = &.{
            .{ .name = "g_inner", .ty = .{ .fixed = .float16 } },
            .{ .name = "g_svh", .ty = .{ .fixed = .float16 } },
            .{ .name = "u_inner", .ty = .{ .fixed = .float16 } },
            .{ .name = "u_svh", .ty = .{ .fixed = .float16 } },
            .{ .name = "d_suh", .ty = .{ .fixed = .float16 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float16 } }},
        .params = &.{.{ .name = "DIM", .kind = .fc_int }},
    },
    .{
        .name = "sushi_hy4_recall_groups",
        .header = hy4_chain.header,
        .source = hy4_chain.groups_source,
        .inputs = &.{
            .{ .name = "order", .ty = .{ .fixed = .int32 } },
            .{ .name = "meta", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
    },
    .{
        .name = "sushi_hy4_recalled_rows",
        .header = hy4_chain.header,
        .source = hy4_chain.rows_source,
        .inputs = &.{
            .{ .name = "groups", .ty = .{ .fixed = .float32 } },
            .{ .name = "meta", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{
            .{ .name = "rows", .ty = .{ .fixed = .uint32 } },
            .{ .name = "keep", .ty = .{ .fixed = .bool_ } },
        },
    },
    .{
        .name = "sushi_hy4_gated",
        .header = hy4_chain.header,
        .source = hy4_chain.gate_source,
        .inputs = &.{
            .{ .name = "u", .ty = .{ .fixed = .float32 } },
            .{ .name = "gate", .ty = .{ .param = "TG" } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "TG", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "V", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_hy4_gated_into",
        .header = hy4_chain.header,
        .source = hy4_chain.gate_into_source,
        .inputs = &.{
            .{ .name = "u", .ty = .{ .fixed = .float32 } },
            .{ .name = "gate", .ty = .{ .param = "TG" } },
            .{ .name = "row0", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .param = "OT" } }},
        .params = &.{
            .{ .name = "TG", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = f32_bf16 },
            .{ .name = "V", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_hy4_gated_prepared_into",
        .header = hy4_chain.header,
        .source = hy4_chain.gate_prepared_into_source,
        .inputs = &.{
            .{ .name = "u", .ty = .{ .fixed = .float32 } },
            .{ .name = "gate", .ty = .{ .param = "TG" } },
            .{ .name = "suh", .ty = .{ .fixed = .float16 } },
            .{ .name = "row0", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float16 } }},
        .params = &.{
            .{ .name = "TG", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "V", .kind = .fc_int },
            .{ .name = "HV", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_hy4_sink_softmax_causal",
        .header = hy4_chain.header,
        .source = hy4_chain.sink_softmax_causal_source,
        .inputs = &.{
            .{ .name = "a", .ty = .{ .fixed = .float32 } },
            .{ .name = "b", .ty = .{ .fixed = .float32 } },
            .{ .name = "sinks", .ty = .{ .param = "TS" } },
            .{ .name = "meta", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "TS", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "HEADS_FIRST", .kind = .fc_bool },
        },
    },
    .{
        .name = "sushi_hy4_sink_softmax_mask",
        .header = hy4_chain.header,
        .source = hy4_chain.sink_softmax_mask_source,
        .inputs = &.{
            .{ .name = "a", .ty = .{ .fixed = .float32 } },
            .{ .name = "b", .ty = .{ .fixed = .float32 } },
            .{ .name = "mask", .ty = .{ .param = "TM" } },
            .{ .name = "sinks", .ty = .{ .param = "TS" } },
            .{ .name = "meta", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "TM", .kind = .io_dtype, .dtypes = bool_f32 },
            .{ .name = "TS", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "HEADS_FIRST", .kind = .fc_bool },
            .{ .name = "MASK_BOOL", .kind = .fc_bool },
        },
    },
    .{
        .name = "sushi_hy4_ring_keys",
        .header = hy4_chain.header,
        .source = hy4_chain.ring_keys_source,
        .inputs = &.{
            .{ .name = "c", .ty = .{ .param = "TC" } },
            .{ .name = "k_pe", .ty = .{ .param = "TP" } },
            .{ .name = "meta", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{
            .{ .name = "c_out", .ty = .{ .fixed = .float32 } },
            .{ .name = "pe_out", .ty = .{ .fixed = .float32 } },
        },
        .params = &.{
            .{ .name = "TC", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "TP", .kind = .io_dtype, .dtypes = f32_bf16 },
        },
    },
    .{
        .name = "sushi_hy4_steel_scores",
        .source = hy4_chain.steel_scores_source,
        .inputs = &.{
            .{ .name = "ql", .ty = .{ .fixed = .float32 } },
            .{ .name = "qp", .ty = .{ .fixed = .float32 } },
            .{ .name = "c", .ty = .{ .fixed = .float32 } },
            .{ .name = "pe", .ty = .{ .fixed = .bfloat16 } },
        },
        .outputs = &.{
            .{ .name = "s1", .ty = .{ .fixed = .float32 } },
            .{ .name = "s2", .ty = .{ .fixed = .float32 } },
        },
        .params = &.{
            .{ .name = "R", .kind = .ct_int },
            .{ .name = "NPART", .kind = .ct_int },
            .{ .name = "U", .kind = .ct_int },
            .{ .name = "N", .kind = .fc_int },
            .{ .name = "QA", .kind = .fc_int },
            .{ .name = "PA", .kind = .fc_int },
            .{ .name = "CA", .kind = .fc_int },
            .{ .name = "EA", .kind = .fc_int },
            .{ .name = "H", .kind = .fc_int },
            .{ .name = "PK", .kind = .fc_int },
        },
        .ct_tuples = steel_score_tuples,
    },
    .{
        .name = "sushi_hy4_steel_values",
        .source = hy4_chain.steel_values_source,
        .inputs = &.{
            .{ .name = "pp", .ty = .{ .fixed = .float32 } },
            .{ .name = "c", .ty = .{ .fixed = .float32 } },
        },
        .outputs = &.{.{ .name = "u", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "NPART", .kind = .ct_int },
            .{ .name = "U", .kind = .ct_int },
            .{ .name = "W", .kind = .fc_int },
            .{ .name = "PA", .kind = .fc_int },
            .{ .name = "CA", .kind = .fc_int },
            .{ .name = "H", .kind = .fc_int },
            .{ .name = "PK", .kind = .fc_int },
        },
        .ct_tuples = steel_val_tuples,
    },
    .{
        .name = "sushi_hy4_steel_scores_groups",
        .source = hy4_chain.steel_scores_groups_source,
        .inputs = &.{
            .{ .name = "ql", .ty = .{ .fixed = .float32 } },
            .{ .name = "qp", .ty = .{ .fixed = .float32 } },
            .{ .name = "c", .ty = .{ .fixed = .float32 } },
            .{ .name = "pe", .ty = .{ .fixed = .bfloat16 } },
        },
        .outputs = &.{
            .{ .name = "s1", .ty = .{ .fixed = .float32 } },
            .{ .name = "s2", .ty = .{ .fixed = .float32 } },
        },
        .params = &.{
            .{ .name = "R", .kind = .ct_int },
            .{ .name = "NPART", .kind = .ct_int },
            .{ .name = "U", .kind = .ct_int },
            .{ .name = "N", .kind = .fc_int },
            .{ .name = "H", .kind = .fc_int },
            .{ .name = "PK", .kind = .fc_int },
        },
        .ct_tuples = steel_score_tuples,
    },
    .{
        .name = "sushi_hy4_steel_values_groups",
        .source = hy4_chain.steel_values_groups_source,
        .inputs = &.{
            .{ .name = "pp", .ty = .{ .fixed = .float32 } },
            .{ .name = "c", .ty = .{ .fixed = .float32 } },
        },
        .outputs = &.{.{ .name = "u", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "NPART", .kind = .ct_int },
            .{ .name = "U", .kind = .ct_int },
            .{ .name = "W", .kind = .fc_int },
            .{ .name = "N", .kind = .fc_int },
            .{ .name = "H", .kind = .fc_int },
            .{ .name = "PK", .kind = .fc_int },
        },
        .ct_tuples = steel_val_tuples,
    },

    // ── hy4_embed ───────────────────────────────────────────────────────────────
    .{
        .name = "hy4_embed_cols",
        .source = hy4_embed.COLS_SOURCE,
        .inputs = &.{
            .{ .name = "d", .ty = .{ .fixed = .float16 } },
            .{ .name = "had", .ty = .{ .fixed = .float32 } },
        },
        .outputs = &.{.{ .name = "x", .ty = .{ .fixed = .float32 } }},
        .params = &.{.{ .name = "DIM", .kind = .fc_int }},
    },
    .{
        .name = "hy4_embed_rows",
        .source = hy4_embed.ROWS_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = .{ .fixed = .float32 } },
            .{ .name = "had", .ty = .{ .fixed = .float32 } },
            .{ .name = "suh", .ty = .{ .fixed = .float16 } },
            .{ .name = "svh", .ty = .{ .fixed = .float16 } },
            .{ .name = "sel", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{.{ .name = "e", .ty = .{ .fixed = .float32 } }},
        .params = &.{.{ .name = "DIM", .kind = .fc_int }},
    },

    // ── hy4_expert_mv ───────────────────────────────────────────────────────────
    .{
        .name = "sushi_hy4_expert_mv",
        .source = hy4_expert_mv.source,
        .inputs = &.{
            .{ .name = "bank", .ty = .{ .param = "TW" } },
            .{ .name = "rows", .ty = .{ .param = "T" } },
            .{ .name = "groups", .ty = .{ .fixed = .uint32 } },
            .{ .name = "picks", .ty = .{ .fixed = .uint32 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .param = "T" } }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = f32_bf16 },
            .{ .name = "TW", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "NV", .kind = .ct_int, .ints = &.{4} },
            .{ .name = "NDIM", .kind = .fc_int },
            .{ .name = "KDIM", .kind = .fc_int },
        },
    },

    // ── hy4_front ───────────────────────────────────────────────────────────────
    .{
        .name = "sushi_hy4_front",
        .source = hy4_front.source,
        .inputs = &.{
            .{ .name = "ckv", .ty = .{ .param = "T" } },
            .{ .name = "w", .ty = .{ .param = "TW" } },
            .{ .name = "pos", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{
            .{ .name = "c_out", .ty = .{ .param = "OT" } },
            .{ .name = "k_out", .ty = .{ .fixed = .bfloat16 } },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = f32_bf16 },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = f32_bf16 },
            .{ .name = "TW", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "LAT", .kind = .fc_int },
            .{ .name = "CKW", .kind = .fc_int },
            .{ .name = "HALF", .kind = .fc_int },
            .{ .name = "BASE", .kind = .fc_int },
            .{ .name = "EPS", .kind = .fc_int },
            .{ .name = "SCALE", .kind = .fc_int },
            .{ .name = "PER_ROW", .kind = .fc_bool },
        },
    },

    // ── hy4_index ───────────────────────────────────────────────────────────────
    .{
        .name = "sushi_hy4_index_scores",
        .source = hy4_index.score_source,
        .inputs = &.{
            .{ .name = "q", .ty = .{ .fixed = .float32 } },
            .{ .name = "keys", .ty = .{ .fixed = .float32 } },
            .{ .name = "w", .ty = .{ .param = "TW" } },
            .{ .name = "meta", .ty = .{ .fixed = .int32 } },
            .{ .name = "wsc", .ty = .{ .fixed = .float32 } },
        },
        .outputs = &.{.{ .name = "scores", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "TG", .kind = .ct_int },
            .{ .name = "IH", .kind = .ct_int },
            .{ .name = "IHD", .kind = .ct_int },
            .{ .name = "TW", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "NOPE", .kind = .fc_int },
            .{ .name = "HALF", .kind = .fc_int },
            .{ .name = "BASE", .kind = .fc_int },
            .{ .name = "QSCALE", .kind = .fc_int },
            .{ .name = "KSCALE", .kind = .fc_int },
        },
        .ct_tuples = index_score_tuples,
    },
    .{
        .name = "sushi_hy4_index_pick",
        .header = sparse.TOPK_HEADER,
        .source = hy4_index.pick_source,
        .inputs = &.{
            .{ .name = "x", .ty = .{ .fixed = .float32 } },
            .{ .name = "kk", .ty = .{ .fixed = .int32 } },
            .{ .name = "meta", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{
            .{ .name = "groups", .ty = .{ .fixed = .float32 } },
            .{ .name = "rows", .ty = .{ .fixed = .uint32 } },
            .{ .name = "keep", .ty = .{ .fixed = .bool_ } },
        },
        .params = &.{.{ .name = "TG", .kind = .fc_int }},
    },
    .{
        .name = "sushi_hy4_index_wide",
        .source = hy4_index.wide_score_source,
        .inputs = &.{
            .{ .name = "q", .ty = .{ .fixed = .float32 } },
            .{ .name = "keys", .ty = .{ .fixed = .float32 } },
            .{ .name = "w", .ty = .{ .fixed = .float32 } },
            .{ .name = "meta", .ty = .{ .fixed = .int32 } },
            .{ .name = "wsc", .ty = .{ .fixed = .float32 } },
        },
        .outputs = &.{.{ .name = "scores", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "IH", .kind = .ct_int, .ints = &.{ 4, 8, 12, 16, 20, 24, 28, 32 } },
            .{ .name = "IHD", .kind = .fc_int },
        },
    },

    // ── hy4_latent ──────────────────────────────────────────────────────────────
    .{
        .name = "sushi_hy4_latent_read",
        .source = hy4_latent.source,
        .inputs = &.{
            .{ .name = "q", .ty = .{ .fixed = .uint32 } },
            .{ .name = "scales", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "biases", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "idx", .ty = .{ .param = "TI" } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .float32 } }},
        .params = &.{
            .{ .name = "TI", .kind = .io_dtype, .dtypes = u32_i32 },
            .{ .name = "GATHER", .kind = .fc_bool },
            .{ .name = "WORDS", .kind = .fc_int },
            .{ .name = "GS_WORDS", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_hy4_latent_read_bf16",
        .source = hy4_latent.source_bf16,
        .inputs = &.{
            .{ .name = "q", .ty = .{ .fixed = .uint32 } },
            .{ .name = "scales", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "biases", .ty = .{ .fixed = .bfloat16 } },
        },
        .outputs = &.{.{ .name = "out", .ty = .{ .fixed = .bfloat16 } }},
        .params = &.{
            .{ .name = "GATHER", .kind = .ignored },
            .{ .name = "WORDS", .kind = .fc_int },
            .{ .name = "GS_WORDS", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_hy4_latent_append",
        .source = hy4_latent.append_source,
        .inputs = &.{
            .{ .name = "c", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "k_pe", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "meta", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{
            .{ .name = "q_dst", .ty = .{ .fixed = .uint32 } },
            .{ .name = "s_dst", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "b_dst", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "pe_dst", .ty = .{ .fixed = .bfloat16 } },
        },
        .params = &.{
            .{ .name = "GROUPS", .kind = .fc_int },
            .{ .name = "ROPE", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_hy4_latent_ring_windows",
        .source = hy4_latent.ring_windows_source,
        .inputs = &.{
            .{ .name = "q", .ty = .{ .fixed = .uint32 } },
            .{ .name = "scales", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "biases", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "k_pe", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "saved_c", .ty = .{ .fixed = .float32 } },
            .{ .name = "saved_pe", .ty = .{ .fixed = .bfloat16 } },
            .{ .name = "meta", .ty = .{ .fixed = .int32 } },
        },
        .outputs = &.{
            .{ .name = "c_out", .ty = .{ .fixed = .float32 } },
            .{ .name = "pe_out", .ty = .{ .fixed = .bfloat16 } },
        },
        .params = &.{
            .{ .name = "ROWS", .kind = .fc_int },
            .{ .name = "WORDS", .kind = .fc_int },
            .{ .name = "GS_WORDS", .kind = .fc_int },
            .{ .name = "ROPE", .kind = .fc_int },
        },
    },

    // ── hy4_route ───────────────────────────────────────────────────────────────
    .{
        .name = "sushi_hy4_route",
        .header = hy4_route.header,
        .source = hy4_route.source,
        .inputs = &.{
            .{ .name = "logits", .ty = .{ .fixed = .float32 } },
            .{ .name = "bias", .ty = .{ .param = "TB" } },
            .{ .name = "map", .ty = .{ .param = "TM" } },
        },
        .outputs = &.{
            .{ .name = "picks", .ty = .{ .fixed = .uint32 } },
            .{ .name = "weights", .ty = .{ .fixed = .float32 } },
            .{ .name = "slots", .ty = .{ .fixed = .uint32 } },
        },
        .params = &.{
            .{ .name = "FAST", .kind = .ct_bool },
            .{ .name = "TB", .kind = .io_dtype, .dtypes = f32_bf16 },
            .{ .name = "TM", .kind = .io_dtype, .dtypes = map_dts },
            .{ .name = "E", .kind = .fc_int },
            .{ .name = "N", .kind = .fc_int },
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "SCALE", .kind = .fc_int },
            .{ .name = "MAP", .kind = .fc_bool },
        },
    },
};
