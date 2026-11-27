//! Shared Sushi Cooler custom kernels catalog (`sushi.metallib`).
//! Covers every shared custom kernel across `src/*.zig` and `src/tensor/matmul.zig`.
//! Single source of truth for both `gen-metallib` and runtime `kernel.zig` resolution.

const std = @import("std");
const custom = @import("metallib_custom.zig");
const matmul = @import("matmul.zig");
const metal_def = @import("../metal_def.zig");
const hc = @import("../hc_kernels.zig");
const sparse = @import("../sparse_select.zig");
const sampler = @import("../sampler_draw.zig");
const trunk = @import("../trunk.zig");
const attn = @import("../attn_kernels.zig");
const router = @import("../moe_router.zig");
const stream_compute = @import("../moe_stream_compute.zig");
const gqmv = @import("../moe_gqmv.zig");
const expert_bf16 = @import("../expert_bf16_kernels.zig");
const fp8 = @import("../fp8_block.zig");
const mtp = @import("../mtp.zig");
const mtp_dense = @import("../mtp_dense_rows.zig");
const mtp_qmv = @import("../mtp_qmv.zig");
const verify_moe = @import("../verify_moe.zig");
const verify_qmm = @import("../verify_qmm.zig");
const dflash_conv = @import("../dflash_conv.zig");
const dflash_qmv = @import("../dflash_qmv.zig");

const KernelSpec = custom.KernelSpec;
const ParamSpec = custom.ParamSpec;
const InputSlot = custom.InputSlot;
const OutputSlot = custom.OutputSlot;
const CtVal = custom.CtVal;
const Dtype = custom.Dtype;

const floats = [_]Dtype{ .bfloat16, .float16, .float32 };
const f16s = [_]Dtype{ .bfloat16, .float16 };
const i32_u32 = [_]Dtype{ .int32, .uint32 };

fn fix(d: Dtype) custom.TypeRef {
    return .{ .fixed = d };
}
fn par(n: []const u8) custom.TypeRef {
    return .{ .param = n };
}

fn ci(v: i32) CtVal {
    return .{ .int = v };
}
fn cd(d: Dtype) CtVal {
    return .{ .dtype = d };
}

// ── Filters ──

fn hcMixFilter(vals: []const CtVal) bool {
    // vals: [R, M, HC, XDT]
    const r = vals[0].int;
    const m = vals[1].int;
    return (r == 2 and (m == 12 or m == 8 or m == 1)) or
        (r == 1 and (m == 12 or m == 8 or m == 4 or m == 1)) or
        (r == 4 and (m == 12 or m == 8 or m == 1));
}

fn hcIdentFilter(vals: []const CtVal) bool {
    // vals: [T, WT]
    return vals[1].dtype == vals[0].dtype or vals[0].dtype == .float32;
}

fn indexTileFilter(vals: []const CtVal) bool {
    // vals: [IH, IHD, KT, T]
    const kt = vals[2].int;
    const t = vals[3].dtype;
    return (t == .float32 and kt == 32) or (t != .float32 and kt == 64);
}

fn gdnStepFilter(vals: []const CtVal) bool {
    // vals: [InT, StT, OutT, Dk, GT, BT]
    const in_t = vals[0].dtype;
    const st_t = vals[1].dtype;
    const out_t = vals[2].dtype;
    const gt = vals[4].dtype;
    const bt = vals[5].dtype;
    if (st_t == .float16) return false;
    if (out_t == .float16 and in_t != .float16) return false;
    // g and beta are either f32 or match InT (beta can also be bf16 when InT=bf16).
    const g_ok = gt == .float32 or gt == in_t;
    const b_ok = bt == .float32 or bt == in_t;
    return g_ok and b_ok;
}

fn gqmvPlainFilter(vals: []const CtVal) bool {
    // vals: [T, BITS, ST]
    const t = vals[0].dtype;
    const st = vals[2].dtype;
    return st == t or t == .float32;
}

fn routerGateFilter(vals: []const CtVal) bool {
    // vals: [TX, TOUT, LT]
    return vals[0].dtype == vals[1].dtype;
}

fn fp8GemvFilter(vals: []const CtVal) bool {
    // vals: [M, NR, T]
    const m = vals[0].int;
    const nr = vals[1].int;
    if (m <= 4) return nr == 1 or nr == 2;
    return nr == 2 or nr == 4;
}

fn mtpQmvTileFilter(vals: []const CtVal) bool {
    // vals: [T, NV, FAST, RPS]
    const fast = vals[2].int;
    const rps = vals[3].int;
    return (fast == 1 and rps == 2) or (fast == 0 and rps == 4);
}

fn mtpQmvStreamFilter(vals: []const CtVal) bool {
    // vals: [O, MT]
    const o = vals[0].int;
    const mt = vals[1].int;
    return (o == 1 and (mt == 5 or mt == 7)) or
        (o == 4 and (mt == 3 or mt == 4)) or
        (o == 2 and mt == 4);
}

// ── Dynamic source builders (build-time only) ──

fn replaceAllAlloc(a: std.mem.Allocator, src: []const u8, needle: []const u8, rep: []const u8) ![]u8 {
    return std.mem.replaceOwned(u8, a, src, needle, rep);
}

fn vqmmFcClean(a: std.mem.Allocator, raw: []const u8) ![]const u8 {
    var s = try replaceAllAlloc(a, raw, "if constexpr (", "if (");
    s = try replaceAllAlloc(a, s, "partials[K_PARTS * ", "partials[8 * ");
    s = try replaceAllAlloc(a, s, "constexpr int K = KCONST;", "const int K = KCONST;");
    s = try replaceAllAlloc(a, s, "constexpr int K_bytes = K * BITS / 8;", "const int K_bytes = K * BITS / 8;");
    s = try replaceAllAlloc(a, s, "constexpr int K_by_gs = K / GS;", "const int K_by_gs = K / GS;");
    s = try replaceAllAlloc(a, s, "constexpr int K_chunk = K / NSG;", "const int K_chunk = K / NSG;");
    s = try replaceAllAlloc(a, s, "static_assert(BITS == 8, \"unsupported NAX affine width\");", "");
    return s;
}

fn vqmmKsSource(a: std.mem.Allocator, arg: usize) ![]const u8 {
    const m = arg / 16;
    const bn = arg % 16;
    return vqmmFcClean(a, try verify_qmm.verifyQmmSource(a, m, bn));
}

fn vqmmMsgSource(a: std.mem.Allocator, arg: usize) ![]const u8 {
    return vqmmFcClean(a, try verify_qmm.verifyQmmMsgSource(a, arg, @intCast(verify_qmm.vqmmMsgBn(@intCast(arg)))));
}

fn vqmmNaxSource(a: std.mem.Allocator, _: usize) ![]const u8 {
    return vqmmFcClean(a, verify_qmm.VQMM_NAX_SOURCE);
}

const gdn_params = [_]ParamSpec{
    .{ .name = "InT", .kind = .ct_dtype, .dtypes = &floats },
    .{ .name = "StT", .kind = .ct_dtype, .dtypes = &.{ .bfloat16, .float32 } },
    .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats },
    .{ .name = "Dk", .kind = .ct_int, .ints = &.{ 64, 128, 256 } },
    .{ .name = "Dv", .kind = .fc_int },
    .{ .name = "Hk", .kind = .fc_int },
    .{ .name = "Hv", .kind = .fc_int },
    .{ .name = "GT", .kind = .io_dtype, .dtypes = &floats },
    .{ .name = "BT", .kind = .io_dtype, .dtypes = &floats },
};

const gdn_ins_plain = [_]InputSlot{
    .{ .name = "q", .ty = par("InT") },
    .{ .name = "k", .ty = par("InT") },
    .{ .name = "v", .ty = par("InT") },
    .{ .name = "g", .ty = par("GT") },
    .{ .name = "beta", .ty = par("BT") },
    .{ .name = "state_in", .ty = par("StT") },
    .{ .name = "T", .ty = fix(.int32), .scalar = true },
};

const gdn_ins_seq = gdn_ins_plain ++ [_]InputSlot{
    .{ .name = "seq_stride", .ty = fix(.int32), .scalar = true },
};

const gdn_outs_plain = [_]OutputSlot{
    .{ .name = "y", .ty = par("OutT") },
    .{ .name = "state_out", .ty = par("StT") },
};

const gdn_outs_seq = [_]OutputSlot{
    .{ .name = "y", .ty = par("OutT") },
    .{ .name = "state_seq", .ty = par("StT") },
    .{ .name = "state_out", .ty = par("StT") },
};

const attn_pd_ins = [_]InputSlot{
    .{ .name = "q", .ty = par("T") },
    .{ .name = "k", .ty = par("T") },
    .{ .name = "v", .ty = par("T") },
    .{ .name = "scl", .ty = fix(.float32) },
    .{ .name = "win", .ty = fix(.int32) },
    .{ .name = "kr", .ty = fix(.int32) },
    .{ .name = "phase", .ty = fix(.int32) },
    .{ .name = "m_in", .ty = fix(.float32) },
    .{ .name = "l_in", .ty = fix(.float32) },
    .{ .name = "o_in", .ty = fix(.float32) },
    .{ .name = "mask", .ty = fix(.bool_) },
    .{ .name = "skip", .ty = fix(.bool_) },
    .{ .name = "sinks", .ty = fix(.float32) },
};

const attn_pd_outs = [_]OutputSlot{
    .{ .name = "out", .ty = par("T") },
    .{ .name = "m_out", .ty = fix(.float32) },
    .{ .name = "l_out", .ty = fix(.float32) },
    .{ .name = "o_out", .ty = fix(.float32) },
};

const fp8_ins_x = [_]InputSlot{
    .{ .name = "x", .ty = par("T") },
    .{ .name = "w", .ty = fix(.uint8) },
    .{ .name = "scales", .ty = fix(.float32) },
};

const fp8_ins_dq = [_]InputSlot{
    .{ .name = "w", .ty = fix(.uint8) },
    .{ .name = "scales", .ty = fix(.float32) },
};

const fp8_outs_1 = [_]OutputSlot{.{ .name = "y", .ty = par("T") }};
const fp8_outs_3 = [_]OutputSlot{
    .{ .name = "yq", .ty = par("T") },
    .{ .name = "yk", .ty = par("T") },
    .{ .name = "yv", .ty = par("T") },
};

const fp8_common_fcs = [_]ParamSpec{
    .{ .name = "K", .kind = .fc_int },
    .{ .name = "TP", .kind = .fc_int },
    .{ .name = "P0", .kind = .fc_int },
    .{ .name = "P1", .kind = .fc_int },
    .{ .name = "P2", .kind = .fc_int },
    .{ .name = "BPR", .kind = .fc_int },
    .{ .name = "VSB", .kind = .fc_int },
};

const vqmm_ins = [_]InputSlot{
    .{ .name = "x", .ty = par("T") },
    .{ .name = "w_q", .ty = fix(.uint32) },
    .{ .name = "scales", .ty = par("T") },
    .{ .name = "biases", .ty = par("T") },
    .{ .name = "K_size", .ty = fix(.int32), .scalar = true },
    .{ .name = "N_size", .ty = fix(.int32), .scalar = true },
};

const vqmm_outs = [_]OutputSlot{.{ .name = "y", .ty = par("T") }};

const vqmm_ks_specs: [18]KernelSpec = blk: {
    var out: [18]KernelSpec = undefined;
    var idx: usize = 0;
    for (2..8) |m| {
        for ([_]usize{ 2, 4, 8 }) |bn| {
            out[idx] = .{
                .name = std.fmt.comptimePrint("sushi_vqmm_ks_m{d}_n{d}", .{ m, bn }),
                .source_fn = vqmmKsSource,
                .source_arg = m * 16 + bn,
                .inputs = &vqmm_ins,
                .outputs = &vqmm_outs,
                .params = &.{
                    .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
                    .{ .name = "GS", .kind = .fc_int },
                    .{ .name = "BITS", .kind = .fc_int },
                    .{ .name = "K_PARTS", .kind = .fc_int },
                },
            };
            idx += 1;
        }
    }
    break :blk out;
};

const vqmm_msg_specs: [6]KernelSpec = blk: {
    var out: [6]KernelSpec = undefined;
    for (2..8, 0..) |m, idx| {
        out[idx] = .{
            .name = std.fmt.comptimePrint("sushi_vqmm_msg_m{d}", .{m}),
            .source_fn = vqmmMsgSource,
            .source_arg = m,
            .inputs = &vqmm_ins,
            .outputs = &vqmm_outs,
            .params = &.{
                .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
                .{ .name = "GS", .kind = .fc_int },
                .{ .name = "BITS", .kind = .fc_int },
                .{ .name = "NSG", .kind = .fc_int },
            },
        };
    }
    break :blk out;
};

const static_specs = [_]KernelSpec{
    // ── src/tensor/matmul.zig ──
    .{
        .name = "sushi_group_splitk_accum",
        .source = matmul.group_accum_source,
        .inputs = &.{.{ .name = "parts", .ty = fix(.float32) }},
        .outputs = &.{.{ .name = "out", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "PARTS", .kind = .fc_int },
            .{ .name = "MN", .kind = .fc_int },
        },
    },

    // ── src/hc_kernels.zig (12 kernels, 1 NAX) ──
    .{
        .name = "dsv41_hc_mix",
        .header = hc.HC_HEADER,
        .source = hc.HC_MIX_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("XDT") },
            .{ .name = "fn", .ty = fix(.float32) },
            .{ .name = "scale", .ty = fix(.float32) },
            .{ .name = "base", .ty = fix(.float32) },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "z", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "TG", .kind = .fc_int },
            .{ .name = "R", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "M", .kind = .ct_int, .ints = &.{ 1, 4, 8, 12 } },
            .{ .name = "HC", .kind = .ct_int, .ints = &.{ 2, 4, 8 } },
            .{ .name = "XDT", .kind = .io_dtype, .dtypes = &floats },
        },
        .filter = hcMixFilter,
    },
    .{
        .name = "dsv41_hc_mix_nax",
        .nax = true,
        .header = metal_def.NAX_HEADER ++ hc.HC_HEADER,
        .source = hc.HC_MIX_NAX_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "fn", .ty = fix(.float32) },
            .{ .name = "scale", .ty = fix(.float32) },
            .{ .name = "base", .ty = fix(.float32) },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "z", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "NK", .kind = .fc_int },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{16} },
            .{ .name = "MIXN", .kind = .ct_int, .ints = &.{ 8, 12, 16, 24 } },
            .{ .name = "HC", .kind = .ct_int, .ints = &.{ 2, 4 } },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
        },
    },
    .{
        .name = "dsv41_hc_stage",
        .header = hc.HC_HEADER,
        .source = hc.HC_STAGE_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "cpre", .ty = fix(.float32) },
            .{ .name = "w", .ty = par("T") },
            .{ .name = "fn", .ty = fix(.float32) },
            .{ .name = "scale", .ty = fix(.float32) },
            .{ .name = "base", .ty = fix(.float32) },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "h_out", .ty = par("T") },
            .{ .name = "z", .ty = fix(.float32) },
        },
        .params = &.{
            .{ .name = "TG", .kind = .fc_int },
            .{ .name = "HC", .kind = .ct_int, .ints = &.{ 2, 4 } },
            .{ .name = "ND", .kind = .ct_int, .ints = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16 } },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
        },
    },
    .{
        .name = "dsv41_hc_stage_x",
        .header = hc.HC_HEADER,
        .source = hc.HC_STAGE_X_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "y", .ty = par("T") },
            .{ .name = "spost", .ty = fix(.float32) },
            .{ .name = "scomb", .ty = fix(.float32) },
            .{ .name = "cpre", .ty = fix(.float32) },
            .{ .name = "w", .ty = par("T") },
            .{ .name = "fn", .ty = fix(.float32) },
            .{ .name = "scale", .ty = fix(.float32) },
            .{ .name = "base", .ty = fix(.float32) },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "h_out", .ty = par("T") },
            .{ .name = "z", .ty = fix(.float32) },
            .{ .name = "x1", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "TG", .kind = .fc_int },
            .{ .name = "HC", .kind = .ct_int, .ints = &.{ 2, 4 } },
            .{ .name = "ND", .kind = .ct_int, .ints = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16 } },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
        },
    },
    .{
        .name = "dsv41_hc_stage_z",
        .header = hc.HC_HEADER,
        .source = hc.HC_STAGE_Z_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "y", .ty = par("T") },
            .{ .name = "za", .ty = fix(.float32) },
            .{ .name = "w", .ty = par("T") },
            .{ .name = "fn", .ty = fix(.float32) },
            .{ .name = "scale", .ty = fix(.float32) },
            .{ .name = "base", .ty = fix(.float32) },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "h_out", .ty = par("T") },
            .{ .name = "z", .ty = fix(.float32) },
            .{ .name = "x1", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "TG", .kind = .fc_int },
            .{ .name = "HC", .kind = .ct_int, .ints = &.{ 2, 4 } },
            .{ .name = "ND", .kind = .ct_int, .ints = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16 } },
            .{ .name = "ITERS", .kind = .fc_int },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
        },
    },
    .{
        .name = "dsv41_hc_split",
        .disambig = "2in",
        .source = hc.HC_SPLIT_SOURCE,
        .inputs = &.{
            .{ .name = "z", .ty = fix(.float32) },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "pre", .ty = fix(.float32) },
            .{ .name = "post", .ty = fix(.float32) },
            .{ .name = "comb", .ty = fix(.float32) },
        },
        .params = &.{
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "ITERS", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_hc_norm",
        .source = hc.HC_NORM_SOURCE,
        .inputs = &.{
            .{ .name = "z", .ty = fix(.float32) },
            .{ .name = "x", .ty = par("T") },
            .{ .name = "cpre", .ty = fix(.float32) },
            .{ .name = "w", .ty = par("T") },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "pre", .ty = fix(.float32) },
            .{ .name = "post", .ty = fix(.float32) },
            .{ .name = "comb", .ty = fix(.float32) },
            .{ .name = "h_out", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "TG", .kind = .fc_int },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "ITERS", .kind = .fc_int },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
        },
    },
    .{
        .name = "dsv41_hc_expand",
        .source = hc.HC_EXPAND_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "y", .ty = par("T") },
            .{ .name = "post", .ty = fix(.float32) },
            .{ .name = "comb", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "HC", .kind = .ct_int, .ints = &.{4} },
            .{ .name = "COPY", .kind = .fc_int },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
        },
    },
    .{
        .name = "hc_ident_norm",
        .source = hc.HC_IDENT_NORM_SOURCE,
        .inputs = &.{
            .{ .name = "z", .ty = fix(.float32) },
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w", .ty = par("WT") },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "pre", .ty = fix(.float32) },
            .{ .name = "post", .ty = fix(.float32) },
            .{ .name = "h_out", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "TG", .kind = .fc_int },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "ZW", .kind = .fc_int },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "WT", .kind = .io_dtype, .dtypes = &floats },
        },
        .filter = hcIdentFilter,
    },
    .{
        .name = "hc_ident_head",
        .source = hc.HC_IDENT_HEAD_SOURCE,
        .inputs = &.{
            .{ .name = "z", .ty = fix(.float32) },
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w", .ty = par("WT") },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "pre", .ty = fix(.float32) },
            .{ .name = "h_out", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "TG", .kind = .fc_int },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "ZW", .kind = .fc_int },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "WT", .kind = .io_dtype, .dtypes = &floats },
        },
        .filter = hcIdentFilter,
    },
    .{
        .name = "hc_ident_expand",
        .source = hc.HC_IDENT_EXPAND_SOURCE,
        .inputs = &.{
            .{ .name = "h", .ty = fix(.float32) },
            .{ .name = "y", .ty = par("YDT") },
            .{ .name = "post", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "out", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "YDT", .kind = .io_dtype, .dtypes = &floats },
        },
    },
    .{
        .name = "hc_ident_mix_expand",
        .header = hc.HC_HEADER,
        .source = hc.HC_IDENT_MIX_EXPAND_SOURCE,
        .inputs = &.{
            .{ .name = "h", .ty = fix(.float32) },
            .{ .name = "y", .ty = par("YDT") },
            .{ .name = "post", .ty = fix(.float32) },
            .{ .name = "fn", .ty = fix(.float32) },
            .{ .name = "scale", .ty = fix(.float32) },
            .{ .name = "base", .ty = fix(.float32) },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "m", .ty = fix(.float32) },
            .{ .name = "z", .ty = fix(.float32) },
        },
        .params = &.{
            .{ .name = "TG", .kind = .fc_int },
            .{ .name = "R", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "M", .kind = .ct_int, .ints = &.{ 1, 4, 8, 12 } },
            .{ .name = "HC", .kind = .ct_int, .ints = &.{ 2, 4, 8 } },
            .{ .name = "YDT", .kind = .io_dtype, .dtypes = &floats },
        },
        .filter = hcMixFilter,
    },

    // ── src/hc_prefill.zig (2 kernels) ──
    .{
        .name = "sushi_hc_prefill_norm",
        .source = @embedFile("../kernels/hc_prefill_norm.metal"),
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w", .ty = par("T") },
            .{ .name = "eps", .ty = fix(.float32), .scalar = true },
            .{ .name = "wo", .ty = par("T") },
            .{ .name = "wi", .ty = par("T") },
        },
        .outputs = &.{
            .{ .name = "normed", .ty = par("T") },
            .{ .name = "stream", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "H", .kind = .fc_int },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "WR", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_hc_prefill_mix",
        .source = @embedFile("../kernels/hc_prefill_mix.metal"),
        .inputs = &.{
            .{ .name = "up", .ty = par("T") },
            .{ .name = "normed", .ty = par("T") },
            .{ .name = "sigtab", .ty = par("T") },
            .{ .name = "rows", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "H", .kind = .fc_int },
        },
    },

    // ── src/sparse_select.zig (6 kernels) ──
    .{
        .name = "dsv41_topk",
        .header = sparse.TOPK_HEADER,
        .source = sparse.TOPK_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = fix(.float32) },
            .{ .name = "kk", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "out", .ty = fix(.int32) }},
        .params = &.{.{ .name = "TG", .kind = .fc_int }},
    },
    .{
        .name = "dsv41_topk_below",
        .header = sparse.TOPK_HEADER,
        .source = sparse.TOPK_BELOW_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = fix(.float32) },
            .{ .name = "kk", .ty = fix(.int32) },
            .{ .name = "lens", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "out", .ty = fix(.int32) }},
        .params = &.{.{ .name = "TG", .kind = .fc_int }},
    },
    .{
        .name = "dsv41_topk_chunk",
        .header = sparse.TOPK_HEADER,
        .source = sparse.TOPK_CHUNK_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = fix(.float32) },
            .{ .name = "kk", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "out", .ty = fix(.int32) }},
        .params = &.{.{ .name = "TG", .kind = .fc_int }},
    },
    .{
        .name = "dsv41_topk_merge",
        .header = sparse.TOPK_HEADER,
        .source = sparse.TOPK_MERGE_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = fix(.float32) },
            .{ .name = "ids", .ty = fix(.int32) },
            .{ .name = "kk", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "out", .ty = fix(.int32) }},
        .params = &.{.{ .name = "TG", .kind = .fc_int }},
    },
    .{
        .name = "dsv41_index_scores",
        .source = sparse.IDX_SCORE_SOURCE,
        .inputs = &.{
            .{ .name = "q", .ty = fix(.float32) },
            .{ .name = "keys", .ty = par("KT") },
            .{ .name = "w", .ty = par("WDT") },
            .{ .name = "meta", .ty = fix(.int32) },
            .{ .name = "wsc", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "scores", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "IH", .kind = .ct_int, .ints = &.{ 4, 8, 16, 32, 64 } },
            .{ .name = "IHD", .kind = .ct_int, .ints = &.{ 32, 64, 128 } },
            .{ .name = "KT", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "WDT", .kind = .io_dtype, .dtypes = &.{ .float32, .bfloat16 } },
        },
    },
    .{
        .name = "dsv41_index_tile",
        .source = sparse.IDX_TILE_SOURCE,
        .inputs = &.{
            .{ .name = "q", .ty = par("T") },
            .{ .name = "keys", .ty = par("T") },
            .{ .name = "w", .ty = par("WDT") },
            .{ .name = "meta", .ty = fix(.int32) },
            .{ .name = "wsc", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "scores", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "IH", .kind = .ct_int, .ints = &.{ 8, 16, 32, 64 } },
            .{ .name = "IHD", .kind = .ct_int, .ints = &.{ 32, 64, 128 } },
            .{ .name = "KT", .kind = .ct_int, .ints = &.{ 32, 64 } },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "WDT", .kind = .io_dtype, .dtypes = &.{ .float32, .bfloat16 } },
        },
        .filter = indexTileFilter,
    },

    // ── src/sampler_draw.zig (1 kernel) ──
    .{
        .name = "sushi_sampler_draw",
        .header = sampler.HEADER,
        .source = sampler.SOURCE,
        .inputs = &.{
            .{ .name = "logits", .ty = par("T") },
            .{ .name = "fparams", .ty = fix(.float32) },
            .{ .name = "uparams", .ty = fix(.uint32) },
            .{ .name = "keys", .ty = fix(.uint32) },
        },
        .outputs = &.{
            .{ .name = "tok", .ty = fix(.uint32) },
            .{ .name = "route_of", .ty = fix(.uint32) },
        },
        .params = &.{
            .{ .name = "V", .kind = .fc_int },
            .{ .name = "TG", .kind = .ct_int, .ints = &.{ 256, 512, 1024 } },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
        },
    },

    // ── src/trunk.zig (5 kernels) ──
    .{
        .name = "gated_delta_step",
        .source = trunk.gdnKernelSource(false, false),
        .inputs = &gdn_ins_plain,
        .outputs = &gdn_outs_plain,
        .params = &gdn_params,
        .filter = gdnStepFilter,
    },
    .{
        .name = "gated_delta_step_vec",
        .source = trunk.gdnKernelSource(true, false),
        .inputs = &gdn_ins_plain,
        .outputs = &gdn_outs_plain,
        .params = &gdn_params,
        .filter = gdnStepFilter,
    },
    .{
        .name = "gated_delta_step_seq",
        .source = trunk.gdnKernelSource(false, true),
        .inputs = &gdn_ins_seq,
        .outputs = &gdn_outs_seq,
        .params = &gdn_params,
        .filter = gdnStepFilter,
    },
    .{
        .name = "gated_delta_step_seq_vec",
        .source = trunk.gdnKernelSource(true, true),
        .inputs = &gdn_ins_seq,
        .outputs = &gdn_outs_seq,
        .params = &gdn_params,
        .filter = gdnStepFilter,
    },
    .{
        .name = "sushi_swiglu",
        .source = trunk.SWIGLU_SOURCE,
        .inputs = &.{
            .{ .name = "gate", .ty = par("T") },
            .{ .name = "up", .ty = par("T") },
            .{ .name = "sigtab", .ty = par("T") },
            .{ .name = "N_size", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{.{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s }},
    },

    // ── src/attn_kernels.zig (4 kernels, 1 NAX) ──
    .{
        .name = "sushi_attn_pd",
        .header = attn.ATTN_PD_KERNEL_HEADER,
        .source = attn.ATTN_PD_KERNEL_SOURCE,
        .inputs = &attn_pd_ins,
        .outputs = &attn_pd_outs,
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "BDK", .kind = .ct_int, .ints = &.{ 192, 256 } },
            .{ .name = "BDV", .kind = .ct_int, .ints = &.{ 128, 256 } },
            .{ .name = "QSA", .kind = .fc_int },
            .{ .name = "SINK", .kind = .fc_int },
        },
        .ct_tuples = &.{
            &.{ cd(.bfloat16), ci(256), ci(256) },
            &.{ cd(.bfloat16), ci(192), ci(128) },
        },
    },
    .{
        .name = "sushi_attn_pd_nax",
        .nax = true,
        .header = @embedFile("../kernels/attn_pd_nax_header.metal"),
        .source = @embedFile("../kernels/attn_pd_nax.metal"),
        .inputs = &attn_pd_ins,
        .outputs = &attn_pd_outs,
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "BDK", .kind = .ct_int, .ints = &.{192} },
            .{ .name = "BDV", .kind = .ct_int, .ints = &.{128} },
            .{ .name = "SINK", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_attn_qsa256_qsplit",
        .disambig = "q",
        .header = attn.KV_UNPACK_HEADER ++ attn.ATTN_QSA256_KERNEL_HEADER,
        .source = attn.QSA_ATTN_Q_SOURCE,
        .inputs = &.{
            .{ .name = "q", .ty = par("T") },
            .{ .name = "scl", .ty = fix(.float32) },
            .{ .name = "blocks", .ty = fix(.int32) },
            .{ .name = "kq", .ty = fix(.uint32) },
            .{ .name = "ksc", .ty = par("T") },
            .{ .name = "kbi", .ty = par("T") },
            .{ .name = "vq", .ty = fix(.uint32) },
            .{ .name = "vsc", .ty = par("T") },
            .{ .name = "vbi", .ty = par("T") },
        },
        .outputs = &.{
            .{ .name = "pacc", .ty = fix(.float32) },
            .{ .name = "pml", .ty = fix(.float32) },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
            .{ .name = "NSG", .kind = .fc_int },
            .{ .name = "NSGT", .kind = .fc_int },
            .{ .name = "BK", .kind = .ct_int, .ints = &.{ 16, 32 } },
            .{ .name = "RATIO", .kind = .fc_int },
            .{ .name = "BITS", .kind = .ct_int, .ints = &.{ 4, 6, 8 } },
            .{ .name = "GS", .kind = .fc_int },
            .{ .name = "QUANT", .kind = .ct_int, .ints = &.{1} },
            .{ .name = "DK", .kind = .ct_int, .ints = &.{ 128, 192, 256 } },
            .{ .name = "DV", .kind = .ct_int, .ints = &.{ 128, 256 } },
            .{ .name = "SPARSE", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_attn_qsa256_qsplit",
        .disambig = "d",
        .header = attn.KV_UNPACK_HEADER ++ attn.ATTN_QSA256_KERNEL_HEADER,
        .source = attn.QSA_ATTN_Q_SOURCE,
        .inputs = &.{
            .{ .name = "q", .ty = par("T") },
            .{ .name = "scl", .ty = fix(.float32) },
            .{ .name = "blocks", .ty = fix(.int32) },
            .{ .name = "kq", .ty = par("T") },
            .{ .name = "ksc", .ty = par("T") },
            .{ .name = "kbi", .ty = par("T") },
            .{ .name = "vq", .ty = par("T") },
            .{ .name = "vsc", .ty = par("T") },
            .{ .name = "vbi", .ty = par("T") },
        },
        .outputs = &.{
            .{ .name = "pacc", .ty = fix(.float32) },
            .{ .name = "pml", .ty = fix(.float32) },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
            .{ .name = "NSG", .kind = .fc_int },
            .{ .name = "NSGT", .kind = .fc_int },
            .{ .name = "BK", .kind = .ct_int, .ints = &.{ 16, 32 } },
            .{ .name = "RATIO", .kind = .fc_int },
            .{ .name = "BITS", .kind = .ct_int, .ints = &.{ 4, 6, 8 } },
            .{ .name = "GS", .kind = .fc_int },
            .{ .name = "QUANT", .kind = .ct_int, .ints = &.{0} },
            .{ .name = "DK", .kind = .ct_int, .ints = &.{ 128, 192, 256 } },
            .{ .name = "DV", .kind = .ct_int, .ints = &.{ 128, 256 } },
            .{ .name = "SPARSE", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_attn_qsa256_qmerge",
        .header = attn.ATTN_QSA256_KERNEL_HEADER,
        .source = attn.QSA_ATTN_MERGE_SOURCE,
        .inputs = &.{
            .{ .name = "pacc", .ty = fix(.float32) },
            .{ .name = "pml", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{.{ .name = "T", .kind = .ct_dtype, .dtypes = &floats }},
    },

    // ── src/moe_router.zig (4 kernels) ──
    .{
        .name = "sushi_moe_router_softmax",
        .source = router.ROUTER_SOURCES[0],
        .inputs = &.{.{ .name = "logits", .ty = par("LT") }},
        .outputs = &.{
            .{ .name = "inds", .ty = fix(.uint32) },
            .{ .name = "scores", .ty = par("TOUT") },
        },
        .params = &.{
            .{ .name = "TOUT", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "NE", .kind = .fc_int },
            .{ .name = "NK", .kind = .fc_int },
            .{ .name = "TG", .kind = .fc_int },
            .{ .name = "NORM", .kind = .fc_int },
            .{ .name = "LT", .kind = .io_dtype, .dtypes = &floats },
        },
    },
    .{
        .name = "sushi_moe_router_sigmoid",
        .source = router.ROUTER_SOURCES[1],
        .inputs = &.{
            .{ .name = "logits", .ty = par("LT") },
            .{ .name = "bias", .ty = par("BT") },
            .{ .name = "scale", .ty = fix(.float32), .scalar = true },
        },
        .outputs = &.{
            .{ .name = "inds", .ty = fix(.uint32) },
            .{ .name = "scores", .ty = par("TOUT") },
        },
        .params = &.{
            .{ .name = "TOUT", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "NE", .kind = .fc_int },
            .{ .name = "NK", .kind = .fc_int },
            .{ .name = "TG", .kind = .fc_int },
            .{ .name = "NORM", .kind = .fc_int },
            .{ .name = "LT", .kind = .io_dtype, .dtypes = &floats },
            .{ .name = "BT", .kind = .io_dtype, .dtypes = &floats },
        },
    },
    .{
        .name = "sushi_moe_router_sigmoid_grouped",
        .source = router.ROUTER_SOURCES[2],
        .inputs = &.{
            .{ .name = "logits", .ty = par("LT") },
            .{ .name = "bias", .ty = par("BT") },
            .{ .name = "scale", .ty = fix(.float32), .scalar = true },
        },
        .outputs = &.{
            .{ .name = "inds", .ty = fix(.uint32) },
            .{ .name = "scores", .ty = par("TOUT") },
        },
        .params = &.{
            .{ .name = "TOUT", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "NE", .kind = .fc_int },
            .{ .name = "NK", .kind = .fc_int },
            .{ .name = "TG", .kind = .fc_int },
            .{ .name = "NORM", .kind = .fc_int },
            .{ .name = "NG", .kind = .fc_int },
            .{ .name = "GSZ", .kind = .fc_int },
            .{ .name = "TOPKG", .kind = .fc_int },
            .{ .name = "LT", .kind = .io_dtype, .dtypes = &floats },
            .{ .name = "BT", .kind = .io_dtype, .dtypes = &floats },
        },
    },
    .{
        .name = "sushi_moe_router_softmax_gate",
        .source = router.ROUTER_SOURCES[3],
        .inputs = &.{
            .{ .name = "logits", .ty = par("LT") },
            .{ .name = "gx", .ty = par("TX") },
            .{ .name = "gw", .ty = par("TX") },
        },
        .outputs = &.{
            .{ .name = "inds", .ty = fix(.uint32) },
            .{ .name = "scores", .ty = par("TOUT") },
            .{ .name = "glogit", .ty = par("TOUT") },
        },
        .params = &.{
            .{ .name = "GK", .kind = .fc_int },
            .{ .name = "TX", .kind = .ct_dtype, .dtypes = &f16s },
            .{ .name = "TOUT", .kind = .ct_dtype, .dtypes = &f16s },
            .{ .name = "NE", .kind = .fc_int },
            .{ .name = "NK", .kind = .fc_int },
            .{ .name = "TG", .kind = .fc_int },
            .{ .name = "NORM", .kind = .fc_int },
            .{ .name = "LT", .kind = .io_dtype, .dtypes = &floats },
        },
        .filter = routerGateFilter,
    },

    // ── src/moe_stream_compute.zig (1 kernel) ──
    .{
        .name = "sushi_expert_pick",
        .source = stream_compute.EXPERT_PICK_SOURCE,
        .inputs = &.{
            .{ .name = "ids", .ty = par("IND") },
            .{ .name = "logits", .ty = par("LT") },
            .{ .name = "map", .ty = fix(.int32) },
            .{ .name = "starved", .ty = fix(.uint8) },
            .{ .name = "gap", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "slots", .ty = fix(.int32) },
            .{ .name = "picked", .ty = fix(.int32) },
        },
        .params = &.{
            .{ .name = "IND", .kind = .io_dtype, .dtypes = &i32_u32 },
            .{ .name = "LT", .kind = .io_dtype, .dtypes = &floats },
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "E", .kind = .fc_int },
            .{ .name = "LIMIT", .kind = .fc_int },
        },
    },

    // ── src/moe_gqmv.zig (7 kernels) ──
    .{
        .name = "sushi_moe_gather_qmv",
        .header = gqmv.GQMV_AFFINE_HEADER,
        .source = gqmv.GQMV_SOURCES[0],
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w_q", .ty = fix(.uint32) },
            .{ .name = "scales", .ty = par("ST") },
            .{ .name = "biases", .ty = par("ST") },
            .{ .name = "inds", .ty = fix(.uint32) },
            .{ .name = "K_size", .ty = fix(.int32), .scalar = true },
            .{ .name = "N_size", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "GS", .kind = .fc_int },
            .{ .name = "BITS", .kind = .ct_int, .ints = &.{ 2, 3, 4, 8 } },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &floats },
        },
        .filter = gqmvPlainFilter,
    },
    .{
        .name = "sushi_moe_gather_qmv_px",
        .header = gqmv.GQMV_AFFINE_HEADER,
        .source = gqmv.GQMV_SOURCES[1],
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w_q", .ty = fix(.uint32) },
            .{ .name = "scales", .ty = par("ST") },
            .{ .name = "biases", .ty = par("ST") },
            .{ .name = "inds", .ty = fix(.uint32) },
            .{ .name = "K_size", .ty = fix(.int32), .scalar = true },
            .{ .name = "N_size", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "GS", .kind = .fc_int },
            .{ .name = "BITS", .kind = .ct_int, .ints = &.{ 2, 3, 4, 8 } },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &floats },
        },
        .filter = gqmvPlainFilter,
    },
    .{
        .name = "sushi_moe_gather_qmv_gateup",
        .header = gqmv.GQMV_AFFINE_HEADER,
        .source = gqmv.GQMV_GATEUP_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "wg_q", .ty = fix(.uint32) },
            .{ .name = "g_scales", .ty = par("T") },
            .{ .name = "g_biases", .ty = par("T") },
            .{ .name = "wu_q", .ty = fix(.uint32) },
            .{ .name = "u_scales", .ty = par("T") },
            .{ .name = "u_biases", .ty = par("T") },
            .{ .name = "inds", .ty = fix(.uint32) },
            .{ .name = "sigtab", .ty = par("T") },
            .{ .name = "K_size", .ty = fix(.int32), .scalar = true },
            .{ .name = "N_size", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
            .{ .name = "GS", .kind = .fc_int },
            .{ .name = "BITS", .kind = .ct_int, .ints = &.{ 2, 3, 4, 8 } },
            .{ .name = "KP", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_moe_gather_qmv_gateup_rows",
        .header = gqmv.GQMV_AFFINE_HEADER,
        .source = gqmv.GQMV_GATEUP_ROWS_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "wg_q", .ty = fix(.uint32) },
            .{ .name = "g_scales", .ty = par("T") },
            .{ .name = "g_biases", .ty = par("T") },
            .{ .name = "wu_q", .ty = fix(.uint32) },
            .{ .name = "u_scales", .ty = par("T") },
            .{ .name = "u_biases", .ty = par("T") },
            .{ .name = "inds", .ty = fix(.uint32) },
            .{ .name = "sigtab", .ty = par("T") },
            .{ .name = "K_size", .ty = fix(.int32), .scalar = true },
            .{ .name = "N_size", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
            .{ .name = "GS", .kind = .fc_int },
            .{ .name = "BITS", .kind = .ct_int, .ints = &.{ 2, 3, 4, 8 } },
            .{ .name = "KP", .kind = .fc_int },
            .{ .name = "TOPK", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_moe_gather_qmv_downred",
        .header = gqmv.GQMV_AFFINE_HEADER,
        .source = gqmv.GQMV_DOWNRED_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w_q", .ty = fix(.uint32) },
            .{ .name = "scales", .ty = par("T") },
            .{ .name = "biases", .ty = par("T") },
            .{ .name = "inds", .ty = fix(.uint32) },
            .{ .name = "scores", .ty = par("T") },
            .{ .name = "K_size", .ty = fix(.int32), .scalar = true },
            .{ .name = "N_size", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
            .{ .name = "GS", .kind = .fc_int },
            .{ .name = "BITS", .kind = .ct_int, .ints = &.{ 2, 3, 4, 8 } },
            .{ .name = "TOPK", .kind = .fc_int },
            .{ .name = "ROWS", .kind = .fc_int },
            .{ .name = "LPR", .kind = .fc_int },
            .{ .name = "KP", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_moe_gather_qmv_downred_rows",
        .header = gqmv.GQMV_AFFINE_HEADER,
        .source = gqmv.GQMV_DOWNRED_ROWS_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w_q", .ty = fix(.uint32) },
            .{ .name = "scales", .ty = par("T") },
            .{ .name = "biases", .ty = par("T") },
            .{ .name = "inds", .ty = fix(.uint32) },
            .{ .name = "scores", .ty = par("T") },
            .{ .name = "K_size", .ty = fix(.int32), .scalar = true },
            .{ .name = "N_size", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
            .{ .name = "GS", .kind = .fc_int },
            .{ .name = "BITS", .kind = .ct_int, .ints = &.{ 2, 3, 4, 8 } },
            .{ .name = "TOPK", .kind = .fc_int },
            .{ .name = "ROWS", .kind = .fc_int },
            .{ .name = "LPR", .kind = .fc_int },
            .{ .name = "KP", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_qmv_rows",
        .source = gqmv.SUSHI_QMV_ROWS_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w_q", .ty = fix(.uint32) },
            .{ .name = "scales", .ty = par("T") },
            .{ .name = "biases", .ty = par("T") },
            .{ .name = "K_size", .ty = fix(.int32), .scalar = true },
            .{ .name = "N_size", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
            .{ .name = "BITS", .kind = .ct_int, .ints = &.{ 3, 4, 8 } },
            .{ .name = "GS", .kind = .fc_int },
        },
    },

    // ── src/expert_bf16_kernels.zig (2 kernels) ──
    .{
        .name = "sushi_expert_bf16_gateup_swiglu",
        .source = expert_bf16.GATEUP_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "slab", .ty = par("T") },
            .{ .name = "slots", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "h", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "TOPK", .kind = .fc_int },
            .{ .name = "NDIM", .kind = .fc_int },
            .{ .name = "KDIM", .kind = .fc_int },
            .{ .name = "LPR", .kind = .fc_int },
            .{ .name = "SGS", .kind = .fc_int },
            .{ .name = "NPT", .kind = .ct_int, .ints = &.{ 1, 2, 4, 8 } },
        },
    },
    .{
        .name = "sushi_expert_bf16_down_reduce",
        .source = expert_bf16.DOWNRED_SOURCE,
        .inputs = &.{
            .{ .name = "h", .ty = par("T") },
            .{ .name = "down", .ty = par("T") },
            .{ .name = "slots", .ty = fix(.int32) },
            .{ .name = "weights", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "TOPK", .kind = .fc_int },
            .{ .name = "NDIM", .kind = .fc_int },
            .{ .name = "KDIM", .kind = .fc_int },
            .{ .name = "LPR", .kind = .fc_int },
            .{ .name = "NPT", .kind = .ct_int, .ints = &.{ 1, 2, 4, 8 } },
            .{ .name = "SSG", .kind = .fc_int },
            .{ .name = "STAGE", .kind = .ct_int, .ints = &.{ 0, 1 } },
        },
    },

    // ── src/fp8_block.zig (8 kernels, 2 NAX) ──
    .{
        .name = "sushi_fp8_block_gemv",
        .header = fp8.E4M3_HEADER,
        .source = fp8.GEMV_HEAD ++ fp8.GEMV_DIRECT ++ fp8.GEMV_TAIL ++ fp8.GEMV_STORE_1,
        .inputs = &fp8_ins_x,
        .outputs = &fp8_outs_1,
        .params = &([_]ParamSpec{
            .{ .name = "M", .kind = .ct_int, .ints = &.{ 1, 2, 3, 4, 5, 6, 7, 8 } },
            .{ .name = "NR", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "SGS", .kind = .fc_int },
            .{ .name = "S", .kind = .ignored },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
        } ++ fp8_common_fcs),
        .filter = fp8GemvFilter,
    },
    .{
        .name = "sushi_fp8_block_gemv3",
        .header = fp8.E4M3_HEADER,
        .source = fp8.GEMV_HEAD ++ fp8.GEMV_DIRECT ++ fp8.GEMV_TAIL ++ fp8.GEMV_STORE_3,
        .inputs = &fp8_ins_x,
        .outputs = &fp8_outs_3,
        .params = &([_]ParamSpec{
            .{ .name = "M", .kind = .ct_int, .ints = &.{ 1, 2, 3, 4, 5, 6, 7, 8 } },
            .{ .name = "NR", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "SGS", .kind = .fc_int },
            .{ .name = "S", .kind = .ignored },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
        } ++ fp8_common_fcs),
        .filter = fp8GemvFilter,
    },
    .{
        .name = "sushi_fp8_block_gemv_staged",
        .header = fp8.E4M3_HEADER,
        .source = fp8.GEMV_HEAD ++ fp8.GEMV_STAGED ++ fp8.GEMV_TAIL ++ fp8.GEMV_STORE_1,
        .inputs = &fp8_ins_x,
        .outputs = &fp8_outs_1,
        .params = &([_]ParamSpec{
            .{ .name = "M", .kind = .ct_int, .ints = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 } },
            .{ .name = "NR", .kind = .ct_int, .ints = &.{4} },
            .{ .name = "SGS", .kind = .fc_int },
            .{ .name = "S", .kind = .ct_int, .ints = &.{1} },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
        } ++ fp8_common_fcs),
    },
    .{
        .name = "sushi_fp8_block_gemv_staged3",
        .header = fp8.E4M3_HEADER,
        .source = fp8.GEMV_HEAD ++ fp8.GEMV_STAGED ++ fp8.GEMV_TAIL ++ fp8.GEMV_STORE_3,
        .inputs = &fp8_ins_x,
        .outputs = &fp8_outs_3,
        .params = &([_]ParamSpec{
            .{ .name = "M", .kind = .ct_int, .ints = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 } },
            .{ .name = "NR", .kind = .ct_int, .ints = &.{4} },
            .{ .name = "SGS", .kind = .fc_int },
            .{ .name = "S", .kind = .ct_int, .ints = &.{1} },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
        } ++ fp8_common_fcs),
    },
    .{
        .name = "sushi_fp8_block_dequant",
        .header = fp8.E4M3_HEADER,
        .source = fp8.DEQUANT_BODY ++ fp8.DEQUANT_DST_1 ++ fp8.DEQUANT_TAIL,
        .inputs = &fp8_ins_dq,
        .outputs = &fp8_outs_1,
        .params = &([_]ParamSpec{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
        } ++ fp8_common_fcs),
    },
    .{
        .name = "sushi_fp8_block_dequant3",
        .header = fp8.E4M3_HEADER,
        .source = fp8.DEQUANT_BODY ++ fp8.DEQUANT_DST_3 ++ fp8.DEQUANT_TAIL,
        .inputs = &fp8_ins_dq,
        .outputs = &fp8_outs_3,
        .params = &([_]ParamSpec{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats },
        } ++ fp8_common_fcs),
    },
    .{
        .name = "sushi_fp8_block_tile",
        .nax = true,
        .header = fp8.TILE_HEADER,
        .source = fp8.tileSource(false),
        .inputs = &fp8_ins_x,
        .outputs = &fp8_outs_1,
        .params = &([_]ParamSpec{
            .{ .name = "M", .kind = .fc_int },
            .{ .name = "MT", .kind = .fc_int },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
        } ++ fp8_common_fcs),
    },
    .{
        .name = "sushi_fp8_block_tile3",
        .nax = true,
        .header = fp8.TILE_HEADER,
        .source = fp8.tileSource(true),
        .inputs = &fp8_ins_x,
        .outputs = &fp8_outs_3,
        .params = &([_]ParamSpec{
            .{ .name = "M", .kind = .fc_int },
            .{ .name = "MT", .kind = .fc_int },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
        } ++ fp8_common_fcs),
    },

    // ── src/mtp.zig (2 kernels) ──
    .{
        .name = "sushi_mtp_top32_partial",
        .header = mtp.TOP32_ORDINAL_HEADER,
        .source = mtp.TOP32_PARTIAL_SRC,
        .inputs = &.{.{ .name = "logits", .ty = par("LDT") }},
        .outputs = &.{
            .{ .name = "cand_ord", .ty = fix(.uint32) },
            .{ .name = "cand_idx", .ty = fix(.uint32) },
        },
        .params = &.{
            .{ .name = "RC", .kind = .fc_int },
            .{ .name = "LDT", .kind = .io_dtype, .dtypes = &floats },
        },
    },
    .{
        .name = "sushi_mtp_top32_finalize",
        .source = mtp.TOP32_FINAL_SRC,
        .inputs = &.{
            .{ .name = "cand_ord", .ty = fix(.uint32) },
            .{ .name = "cand_idx", .ty = fix(.uint32) },
        },
        .outputs = &.{.{ .name = "token_ids", .ty = fix(.uint32) }},
    },

    // ── src/mtp_dense_rows.zig (2 kernels) ──
    .{
        .name = "mtp_dense_router_rows",
        .source = mtp_dense.ROUTER_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = fix(.bfloat16) },
            .{ .name = "w", .ty = fix(.bfloat16) },
        },
        .outputs = &.{.{ .name = "y", .ty = fix(.bfloat16) }},
    },
    .{
        .name = "mtp_dense_gate_rows",
        .source = mtp_dense.GATE_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = fix(.bfloat16) },
            .{ .name = "w", .ty = fix(.bfloat16) },
        },
        .outputs = &.{.{ .name = "y", .ty = fix(.bfloat16) }},
    },

    // ── src/mtp_qmv.zig (3 kernels) ──
    .{
        .name = "mtp_serial_qmv8_rows",
        .source = mtp_qmv.SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w", .ty = fix(.uint32) },
            .{ .name = "sc", .ty = par("T") },
            .{ .name = "bi", .ty = par("T") },
            .{ .name = "K_size", .ty = fix(.int32), .scalar = true },
            .{ .name = "N_size", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "M", .kind = .fc_int },
            .{ .name = "GS", .kind = .fc_int },
            .{ .name = "NV", .kind = .ct_int, .ints = &.{ 1, 2, 3, 4 } },
            .{ .name = "FAST", .kind = .ct_int, .ints = &.{ 0, 1 } },
            .{ .name = "RPS", .kind = .ct_int, .ints = &.{ 2, 4 } },
        },
        .filter = mtpQmvTileFilter,
    },
    .{
        .name = "mtp_serial_qmv8_rows_stream",
        .source = mtp_qmv.STREAM_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = fix(.bfloat16) },
            .{ .name = "w", .ty = fix(.uint32) },
            .{ .name = "scales", .ty = fix(.bfloat16) },
            .{ .name = "biases", .ty = fix(.bfloat16) },
        },
        .outputs = &.{.{ .name = "y", .ty = fix(.bfloat16) }},
        .params = &.{
            .{ .name = "M", .kind = .fc_int },
            .{ .name = "N", .kind = .fc_int },
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "GS", .kind = .fc_int },
            .{ .name = "O", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "SG", .kind = .fc_int },
            .{ .name = "MT", .kind = .ct_int, .ints = &.{ 3, 4, 5, 7 } },
            .{ .name = "SPLIT", .kind = .fc_int },
        },
        .filter = mtpQmvStreamFilter,
    },
    .{
        .name = "mtp_serial_gemv_f32_rows",
        .source = mtp_qmv.F32_GEMV_ROWS_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = fix(.float32) },
            .{ .name = "w", .ty = fix(.float32) },
            .{ .name = "K_size", .ty = fix(.int32), .scalar = true },
            .{ .name = "N_size", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "y", .ty = fix(.float32) }},
    },

    // ── src/verify_moe.zig (4 kernels) ──
    .{
        .name = "verify_expert_reuse",
        .source = verify_moe.VERIFY_EXPERT_REUSE_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w", .ty = fix(.uint32) },
            .{ .name = "sc", .ty = par("T") },
            .{ .name = "bi", .ty = par("T") },
            .{ .name = "ids", .ty = fix(.uint32) },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "N", .kind = .fc_int },
        },
    },
    .{
        .name = "verify_expert_reduce",
        .source = verify_moe.VERIFY_EXPERT_REDUCE_SOURCE,
        .inputs = &.{
            .{ .name = "down", .ty = par("T") },
            .{ .name = "inverse", .ty = fix(.uint32) },
            .{ .name = "scores", .ty = par("T") },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{.{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} }},
    },
    .{
        .name = "mtp_coarse_pairs",
        .source = verify_moe.MTP_COARSE_PAIRS_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w", .ty = fix(.uint32) },
            .{ .name = "sc", .ty = par("T") },
            .{ .name = "bi", .ty = par("T") },
            .{ .name = "K_size", .ty = fix(.int32), .scalar = true },
            .{ .name = "N_size", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{.{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} }},
    },
    .{
        .name = "verify_route_pack",
        .source = verify_moe.VERIFY_ROUTE_PACK_SOURCE,
        .inputs = &.{.{ .name = "ids", .ty = fix(.uint32) }},
        .outputs = &.{
            .{ .name = "inverse", .ty = fix(.uint32) },
            .{ .name = "sorted", .ty = fix(.uint32) },
            .{ .name = "lhs", .ty = fix(.uint32) },
        },
    },

    // ── src/verify_qmm.zig NAX kernel (24 SIMD kernels appended below) ──
    .{
        .name = "sushi_vqmm_nax_m16",
        .nax = true,
        .header = verify_qmm.VQMM_NAX_HEADER,
        .source_fn = vqmmNaxSource,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w_q", .ty = fix(.uint32) },
            .{ .name = "scales", .ty = par("T") },
            .{ .name = "biases", .ty = par("T") },
            .{ .name = "N_size", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
            .{ .name = "GS", .kind = .fc_int },
            .{ .name = "BITS", .kind = .fc_int },
            .{ .name = "KCONST", .kind = .fc_int },
        },
    },

    // ── src/dflash_conv.zig (2 kernels) ──
    .{
        .name = "sushi_dflash_two_tap_bf16",
        .source = dflash_conv.SOURCE,
        .inputs = &.{
            .{ .name = "hidden", .ty = fix(.bfloat16) },
            .{ .name = "dynamic", .ty = fix(.bfloat16) },
            .{ .name = "base", .ty = fix(.bfloat16) },
        },
        .outputs = &.{.{ .name = "y", .ty = fix(.bfloat16) }},
        .params = &.{
            .{ .name = "L", .kind = .fc_int },
            .{ .name = "H", .kind = .fc_int },
            .{ .name = "GS", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_dflash_two_tap_bf16_add",
        .source = dflash_conv.SOURCE_ADD,
        .inputs = &.{
            .{ .name = "hidden", .ty = fix(.bfloat16) },
            .{ .name = "dynamic", .ty = fix(.bfloat16) },
            .{ .name = "base", .ty = fix(.bfloat16) },
            .{ .name = "residual", .ty = fix(.bfloat16) },
        },
        .outputs = &.{.{ .name = "y", .ty = fix(.bfloat16) }},
        .params = &.{
            .{ .name = "L", .kind = .fc_int },
            .{ .name = "H", .kind = .fc_int },
            .{ .name = "GS", .kind = .fc_int },
        },
    },

    // ── src/dflash_qmv.zig (2 kernels) ──
    .{
        .name = "sushi_dflash_a4_wide",
        .source = dflash_qmv.SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w", .ty = fix(.uint32) },
            .{ .name = "sc", .ty = par("T") },
            .{ .name = "bi", .ty = par("T") },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "M", .kind = .fc_int },
            .{ .name = "N", .kind = .fc_int },
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "NV", .kind = .ct_int, .ints = &.{8} },
        },
    },
    .{
        .name = "sushi_dflash_a4_gate_up_act",
        .header = dflash_qmv.ACT_HEADER,
        .source = dflash_qmv.ACT_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = fix(.bfloat16) },
            .{ .name = "gw", .ty = fix(.uint32) },
            .{ .name = "gsc", .ty = fix(.bfloat16) },
            .{ .name = "gbi", .ty = fix(.bfloat16) },
            .{ .name = "uw", .ty = fix(.uint32) },
            .{ .name = "usc", .ty = fix(.bfloat16) },
            .{ .name = "ubi", .ty = fix(.bfloat16) },
            .{ .name = "sigmoid", .ty = fix(.bfloat16) },
        },
        .outputs = &.{.{ .name = "y", .ty = fix(.bfloat16) }},
        .params = &.{
            .{ .name = "M", .kind = .fc_int },
            .{ .name = "N", .kind = .fc_int },
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "NV", .kind = .ct_int, .ints = &.{ 3, 8 } },
        },
    },

    // ── Engine / Metal / Route contract self-test kernels (precompiled so tests compile 0 Metal source) ──
    .{
        .name = "sushi_test_scale_f32",
        .source = "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] * scale;",
        .inputs = &.{
            .{ .name = "x", .ty = fix(.float32) },
            .{ .name = "scale", .ty = fix(.float32), .scalar = true },
        },
        .outputs = &.{.{ .name = "out", .ty = fix(.float32) }},
    },
    .{
        .name = "sushi_test_scale",
        .source = "out[thread_position_in_grid.x] = float(x[thread_position_in_grid.x]) * float(SCALE);",
        .inputs = &.{.{ .name = "x", .ty = par("X") }},
        .outputs = &.{.{ .name = "out", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "X", .kind = .io_dtype, .dtypes = &.{ .float32, .int32 } },
            .{ .name = "SCALE", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_test_add",
        .disambig = "k",
        .source = "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] + float(K);",
        .inputs = &.{.{ .name = "x", .ty = fix(.float32) }},
        .outputs = &.{.{ .name = "out", .ty = fix(.float32) }},
        .params = &.{.{ .name = "K", .kind = .fc_int }},
    },
    .{
        .name = "sushi_test_add",
        .disambig = "kl",
        .source = "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] + float(K) + float(L) * 0.0f;",
        .inputs = &.{.{ .name = "x", .ty = fix(.float32) }},
        .outputs = &.{.{ .name = "out", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "L", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_test_wide_targs",
        .source = "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] + float(A0 + A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10 + A11 + A12 + A13 + A14 + A15 + A16);",
        .inputs = &.{.{ .name = "x", .ty = fix(.float32) }},
        .outputs = &.{.{ .name = "out", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "A0", .kind = .fc_int },  .{ .name = "A1", .kind = .fc_int },
            .{ .name = "A2", .kind = .fc_int },  .{ .name = "A3", .kind = .fc_int },
            .{ .name = "A4", .kind = .fc_int },  .{ .name = "A5", .kind = .fc_int },
            .{ .name = "A6", .kind = .fc_int },  .{ .name = "A7", .kind = .fc_int },
            .{ .name = "A8", .kind = .fc_int },  .{ .name = "A9", .kind = .fc_int },
            .{ .name = "A10", .kind = .fc_int }, .{ .name = "A11", .kind = .fc_int },
            .{ .name = "A12", .kind = .fc_int }, .{ .name = "A13", .kind = .fc_int },
            .{ .name = "A14", .kind = .fc_int }, .{ .name = "A15", .kind = .fc_int },
            .{ .name = "A16", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_test_range_copy",
        .source = "uint i = thread_position_in_grid.x; if (i >= uint(r[0]) && i < uint(r[1])) out[i] = x[i];",
        .inputs = &.{
            .{ .name = "x", .ty = fix(.float32) },
            .{ .name = "r", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "out", .ty = fix(.float32) }},
    },
    .{
        .name = "sushi_test_counted_sum",
        .source =
        \\const uint g = threadgroup_position_in_grid.x;
        \\device atomic_uint *c = (device atomic_uint *)ctr;
        \\threadgroup bool last[1];
        \\if (thread_index_in_threadgroup == 0) {
        \\  atomic_store_explicit(c + CTR + g, x[g], memory_order_relaxed);
        \\  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        \\  last[0] = atomic_fetch_add_explicit(c, 1u, memory_order_relaxed) == uint(G - 1);
        \\  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        \\}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\if (!last[0] || thread_index_in_threadgroup != 0) return;
        \\uint sum = 0;
        \\for (uint i = 0; i < uint(G); i++) sum += atomic_load_explicit(c + CTR + i, memory_order_relaxed);
        \\out[0] = sum;
        \\atomic_store_explicit(c, 0u, memory_order_relaxed);
        ,
        .inputs = &.{.{ .name = "x", .ty = fix(.uint32) }},
        .outputs = &.{
            .{ .name = "out", .ty = fix(.uint32) },
            .{ .name = "ctr", .ty = fix(.uint32) },
        },
        .params = &.{
            .{ .name = "G", .kind = .fc_int },
            .{ .name = "CTR", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_test_engine_probe",
        .source =
        \\uint i = thread_position_in_grid.x;
        \\out[i] = T(float(x[i]) * float(SCALE) + float(w[i % uint(w_shape[0])]) + bias);
        \\tail[i] = x_ndim + int(w_strides[0]);
        ,
        .inputs = &.{
            .{ .name = "x", .ty = fix(.float32) },
            .{ .name = "w", .ty = fix(.float32) },
            .{ .name = "bias", .ty = fix(.float32), .scalar = true },
        },
        .outputs = &.{
            .{ .name = "out", .ty = par("T") },
            .{ .name = "tail", .ty = fix(.int32) },
        },
        .params = &.{
            .{ .name = "SCALE", .kind = .fc_int },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
        },
    },
    .{
        .name = "sushi_test_native_probe",
        .source =
        \\uint i = thread_position_in_grid.x;
        \\out[i] = x[i] * float(SCALE) + float(threads_per_threadgroup.x > 0);
        ,
        .inputs = &.{.{ .name = "x", .ty = fix(.float32) }},
        .outputs = &.{.{ .name = "out", .ty = fix(.float32) }},
        .params = &.{.{ .name = "SCALE", .kind = .fc_int }},
    },
    .{
        .name = "sushi_engine_bfloat_math",
        .source =
        \\uint i = thread_position_in_grid.x;
        \\T v = T(x[i]);
        \\out[i] = metal::abs(v) + metal::precise::exp(v) * T(0.5f) + simd_sum(v) + simd_shuffle_down(v, 1u) + metal::fast::tanh(v);
        ,
        .inputs = &.{.{ .name = "x", .ty = fix(.float32) }},
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{.{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} }},
    },
};

const all_specs = static_specs ++ vqmm_ks_specs ++ vqmm_msg_specs;
pub const specs: []const KernelSpec = &all_specs;
