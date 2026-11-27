//! Qwen4 precompiled custom kernels catalog (`sushi.metallib`).
//! Covers every Qwen4 custom kernel across `src/qwen4/*.zig` and `src/kernels/{gdn_pipelined,qsa_nax*}.metal`.
//! Single source of truth for `gen-metallib`, runtime `kernel.zig` resolution, and `stock.zig` completeness test.

const custom = @import("../tensor/metallib_custom.zig");
const trunk = @import("../trunk.zig");
const attn = @import("../attn_kernels.zig");
const gdn_dec = @import("gdn_decode.zig");
const qgdn = @import("qwen4_gdn.zig");
const qsel = @import("qwen4_qsa_select.zig");
const qgat = @import("qwen4_qsa_gather.zig");
const qsa = @import("qwen4_qsa.zig");
const qhc = @import("qwen4_hc.zig");
const qfwd = @import("qwen4_forward.zig");

const KernelSpec = custom.KernelSpec;
const ParamSpec = custom.ParamSpec;
const InputSlot = custom.InputSlot;
const OutputSlot = custom.OutputSlot;
const CtVal = custom.CtVal;
const Dtype = custom.Dtype;

const floats = [_]Dtype{ .bfloat16, .float16, .float32 };
const f16s = [_]Dtype{ .bfloat16, .float16 };
const bf_f32 = [_]Dtype{ .bfloat16, .float32 };

fn fix(d: Dtype) custom.TypeRef {
    return .{ .fixed = d };
}
fn par(n: []const u8) custom.TypeRef {
    return .{ .param = n };
}
fn ci(v: i32) CtVal {
    return .{ .int = v };
}

fn gdnBlockedFilter(vals: []const CtVal) bool {
    // vals: [InT, StT, OutT, Dk, GT, BT]
    const in_t = vals[0].dtype;
    const st_t = vals[1].dtype;
    const out_t = vals[2].dtype;
    const gt = vals[4].dtype;
    const bt = vals[5].dtype;
    if (st_t != in_t and st_t != .float32) return false;
    if (out_t != in_t and out_t != .bfloat16) return false;
    if (gt != in_t and gt != .float32) return false;
    if (bt != in_t and bt != .float32) return false;
    return true;
}

fn gdnFoldFilter(vals: []const CtVal) bool {
    // vals: [T, StT, GT, DK, DV, NT, SPLIT, TL, AL]
    const nt = vals[5].int;
    const tl = vals[7].int;
    return nt == 1024 or (nt == 2048 and tl == 2);
}

const gdn_blocked_ins = [_]InputSlot{
    .{ .name = "q", .ty = par("InT") },
    .{ .name = "k", .ty = par("InT") },
    .{ .name = "v", .ty = par("InT") },
    .{ .name = "g", .ty = par("GT") },
    .{ .name = "beta", .ty = par("BT") },
    .{ .name = "state_in", .ty = par("StT") },
    .{ .name = "T", .ty = fix(.int32), .scalar = true },
};
const gdn_blocked_outs = [_]OutputSlot{
    .{ .name = "y", .ty = par("OutT") },
    .{ .name = "state_out", .ty = par("StT") },
};

const gdn_blocked_params_narrow = [_]ParamSpec{
    .{ .name = "InT", .kind = .ct_dtype, .dtypes = &f16s },
    .{ .name = "StT", .kind = .ct_dtype, .dtypes = &floats },
    .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats },
    .{ .name = "Dk", .kind = .ct_int, .ints = &.{128} },
    .{ .name = "Dv", .kind = .fc_int },
    .{ .name = "Hk", .kind = .fc_int },
    .{ .name = "Hv", .kind = .fc_int },
    .{ .name = "GT", .kind = .io_dtype, .dtypes = &floats },
    .{ .name = "BT", .kind = .io_dtype, .dtypes = &floats },
};

const gdn_blocked_params_all = [_]ParamSpec{
    .{ .name = "InT", .kind = .ct_dtype, .dtypes = &floats },
    .{ .name = "StT", .kind = .ct_dtype, .dtypes = &floats },
    .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats },
    .{ .name = "Dk", .kind = .ct_int, .ints = &.{128} },
    .{ .name = "Dv", .kind = .fc_int },
    .{ .name = "Hk", .kind = .fc_int },
    .{ .name = "Hv", .kind = .fc_int },
    .{ .name = "GT", .kind = .io_dtype, .dtypes = &floats },
    .{ .name = "BT", .kind = .io_dtype, .dtypes = &floats },
};

const gdn_dec_ins = [_]InputSlot{
    .{ .name = "qkv", .ty = par("T") },
    .{ .name = "a_in", .ty = par("T") },
    .{ .name = "b_in", .ty = par("T") },
    .{ .name = "conv_state", .ty = par("T") },
    .{ .name = "state_in", .ty = par("StT") },
    .{ .name = "conv_w", .ty = par("T") },
    .{ .name = "A_log", .ty = par("AL") },
    .{ .name = "dt_bias", .ty = par("T") },
    .{ .name = "q_scale", .ty = par("T"), .scalar = true },
    .{ .name = "k_scale", .ty = par("T"), .scalar = true },
};

const qsa_select_params = [_]ParamSpec{
    .{ .name = "TGS", .kind = .fc_int },
    .{ .name = "K", .kind = .fc_int },
    .{ .name = "DIGITS", .kind = .ct_int, .ints = &.{ 8, 11 } },
};

const hc_du_params = [_]ParamSpec{
    .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
    .{ .name = "GS", .kind = .fc_int },
    .{ .name = "BITS", .kind = .ct_int, .ints = &.{ 2, 4, 8 } },
    .{ .name = "HC", .kind = .fc_int },
    .{ .name = "H", .kind = .fc_int },
    .{ .name = "R", .kind = .fc_int },
    .{ .name = "ROWS", .kind = .ct_int, .ints = &.{ 1, 2, 3, 4, 5, 6, 7, 8 } },
    .{ .name = "NROWS", .kind = .fc_int },
};

pub const specs: []const KernelSpec = &.{
    // ── gdn_decode.zig (4 SIMD kernels) ──
    .{
        .name = "sushi_gdn_decode_recur",
        .header = trunk.GDN_KERNEL_HEADER,
        .source = gdn_dec.K1_SOURCE,
        .inputs = &gdn_dec_ins,
        .outputs = &.{
            .{ .name = "y", .ty = par("T") },
            .{ .name = "conv_out", .ty = par("T") },
            .{ .name = "state_out", .ty = par("StT") },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "StT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "GT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "HK", .kind = .fc_int },
            .{ .name = "HV", .kind = .fc_int },
            .{ .name = "DK", .kind = .ct_int, .ints = &.{128} },
            .{ .name = "DV", .kind = .ct_int, .ints = &.{128} },
            .{ .name = "C", .kind = .fc_int },
            .{ .name = "NT", .kind = .ct_int, .ints = &.{256} },
            .{ .name = "SPLIT", .kind = .ct_int, .ints = &.{4} },
            .{ .name = "AL", .kind = .io_dtype, .dtypes = &bf_f32 },
        },
    },
    .{
        .name = "sushi_gdn_decode_recur_seq",
        .header = trunk.GDN_KERNEL_HEADER,
        .source = gdn_dec.K1S_SOURCE,
        .inputs = &gdn_dec_ins,
        .outputs = &.{
            .{ .name = "y", .ty = par("T") },
            .{ .name = "conv_out", .ty = par("T") },
            .{ .name = "state_out", .ty = par("StT") },
            .{ .name = "state_seq", .ty = par("StT") },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "StT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "GT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "HK", .kind = .fc_int },
            .{ .name = "HV", .kind = .fc_int },
            .{ .name = "DK", .kind = .ct_int, .ints = &.{128} },
            .{ .name = "DV", .kind = .ct_int, .ints = &.{128} },
            .{ .name = "C", .kind = .fc_int },
            .{ .name = "NT", .kind = .ct_int, .ints = &.{256} },
            .{ .name = "SPLIT", .kind = .ct_int, .ints = &.{4} },
            .{ .name = "TL", .kind = .ct_int, .ints = &.{ 2, 3, 4, 5, 6, 7, 8, 9 } },
            .{ .name = "AL", .kind = .io_dtype, .dtypes = &bf_f32 },
        },
    },
    .{
        .name = "sushi_gdn_decode_recur_seq_nocap",
        .header = trunk.GDN_KERNEL_HEADER,
        .source = gdn_dec.K1S_NOCAP_SOURCE,
        .inputs = &gdn_dec_ins,
        .outputs = &.{
            .{ .name = "y", .ty = par("T") },
            .{ .name = "conv_out", .ty = par("T") },
            .{ .name = "state_out", .ty = par("StT") },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "StT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "GT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "HK", .kind = .fc_int },
            .{ .name = "HV", .kind = .fc_int },
            .{ .name = "DK", .kind = .ct_int, .ints = &.{128} },
            .{ .name = "DV", .kind = .ct_int, .ints = &.{128} },
            .{ .name = "C", .kind = .fc_int },
            .{ .name = "NT", .kind = .ct_int, .ints = &.{256} },
            .{ .name = "SPLIT", .kind = .ct_int, .ints = &.{4} },
            .{ .name = "TL", .kind = .ct_int, .ints = &.{ 2, 3, 4, 5, 6, 7, 8, 9 } },
            .{ .name = "AL", .kind = .io_dtype, .dtypes = &bf_f32 },
        },
    },
    .{
        .name = "sushi_gdn_verify_fold",
        .header = trunk.GDN_KERNEL_HEADER,
        .source = gdn_dec.K1S_FOLD_SOURCE,
        .inputs = &(gdn_dec_ins ++ [_]InputSlot{
            .{ .name = "z", .ty = par("T") },
            .{ .name = "norm_w", .ty = par("T") },
            .{ .name = "eps", .ty = fix(.float32), .scalar = true },
        }),
        .outputs = &.{
            .{ .name = "gated", .ty = par("T") },
            .{ .name = "conv_out", .ty = par("T") },
            .{ .name = "state_out", .ty = par("StT") },
            .{ .name = "state_seq", .ty = par("StT") },
            .{ .name = "conv_in", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "StT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "GT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "HK", .kind = .fc_int },
            .{ .name = "HV", .kind = .fc_int },
            .{ .name = "DK", .kind = .ct_int, .ints = &.{128} },
            .{ .name = "DV", .kind = .ct_int, .ints = &.{128} },
            .{ .name = "C", .kind = .fc_int },
            .{ .name = "NT", .kind = .ct_int, .ints = &.{ 1024, 2048 } },
            .{ .name = "SPLIT", .kind = .ct_int, .ints = &.{1} },
            .{ .name = "TL", .kind = .ct_int, .ints = &.{ 2, 3, 4, 5, 6, 7, 8, 9 } },
            .{ .name = "SWISH", .kind = .fc_int },
            .{ .name = "AL", .kind = .io_dtype, .dtypes = &bf_f32 },
        },
        .filter = gdnFoldFilter,
    },

    // ── qwen4_gdn.zig (6 SIMD kernels) ──
    .{
        .name = "gated_delta_blocked_tb16",
        .source = qgdn.gdnBlockedSource(16),
        .inputs = &gdn_blocked_ins,
        .outputs = &gdn_blocked_outs,
        .params = &gdn_blocked_params_all,
        .filter = gdnBlockedFilter,
    },
    .{
        .name = "gated_delta_blocked_tb32",
        .source = qgdn.gdnBlockedSource(32),
        .inputs = &gdn_blocked_ins,
        .outputs = &gdn_blocked_outs,
        .params = &gdn_blocked_params_narrow,
        .filter = gdnBlockedFilter,
    },
    .{
        .name = "gated_delta_blocked_tb48",
        .source = qgdn.gdnBlockedSource(48),
        .inputs = &gdn_blocked_ins,
        .outputs = &gdn_blocked_outs,
        .params = &gdn_blocked_params_narrow,
        .filter = gdnBlockedFilter,
    },
    .{
        .name = "gated_delta_pipelined_tb12",
        .source = @embedFile("../kernels/gdn_pipelined.metal"),
        .inputs = &gdn_blocked_ins,
        .outputs = &gdn_blocked_outs,
        .params = &gdn_blocked_params_narrow,
        .filter = gdnBlockedFilter,
    },
    .{
        .name = "sushi_gdn_prework2",
        .header = trunk.GDN_KERNEL_HEADER,
        .source = qgdn.GDN_PREWORK_SOURCE,
        .inputs = &.{
            .{ .name = "qkv", .ty = par("T") },
            .{ .name = "conv_state", .ty = par("T") },
            .{ .name = "conv_w", .ty = par("T") },
            .{ .name = "q_scale", .ty = par("T"), .scalar = true },
            .{ .name = "k_scale", .ty = par("T"), .scalar = true },
            .{ .name = "b_in", .ty = par("T") },
            .{ .name = "a_in", .ty = par("T") },
            .{ .name = "A_log", .ty = par("AL") },
            .{ .name = "dt_bias", .ty = par("T") },
            .{ .name = "sigtab", .ty = par("T") },
            .{ .name = "seq", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{
            .{ .name = "q_out", .ty = par("T") },
            .{ .name = "k_out", .ty = par("T") },
            .{ .name = "v_out", .ty = par("T") },
            .{ .name = "conv_out", .ty = par("T") },
            .{ .name = "g_out", .ty = par("GT") },
            .{ .name = "beta_out", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "GT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "HK", .kind = .fc_int },
            .{ .name = "HV", .kind = .fc_int },
            .{ .name = "DK", .kind = .fc_int },
            .{ .name = "DV", .kind = .fc_int },
            .{ .name = "NKEEP", .kind = .fc_int },
            .{ .name = "C", .kind = .fc_int },
            .{ .name = "TAB", .kind = .fc_int },
            .{ .name = "QSTRIDE", .kind = .fc_int },
            .{ .name = "QOFF", .kind = .fc_int },
            .{ .name = "BSTRIDE", .kind = .fc_int },
            .{ .name = "BOFF", .kind = .fc_int },
            .{ .name = "ASTRIDE", .kind = .fc_int },
            .{ .name = "AOFF", .kind = .fc_int },
            .{ .name = "AL", .kind = .io_dtype, .dtypes = &bf_f32 },
        },
    },
    .{
        .name = "sushi_gdn_normgate",
        .source = qgdn.GDN_NORMGATE_SOURCE,
        .inputs = &.{
            .{ .name = "y", .ty = par("T") },
            .{ .name = "z", .ty = par("T") },
            .{ .name = "norm_w", .ty = par("T") },
            .{ .name = "eps", .ty = fix(.float32), .scalar = true },
            .{ .name = "sigtab", .ty = par("T") },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "HV", .kind = .fc_int },
            .{ .name = "DV", .kind = .fc_int },
            .{ .name = "ZSTRIDE", .kind = .fc_int },
            .{ .name = "ZOFF", .kind = .fc_int },
            .{ .name = "TAB", .kind = .fc_int },
            .{ .name = "SWISH", .kind = .fc_int },
        },
    },

    // ── qwen4_qsa_select.zig (1 NAX + 1 SIMD) ──
    .{
        .name = "sushi_qsa_score",
        .nax = true,
        .header = attn.QSA_SCORE_KERNEL_HEADER,
        .source = qsel.QSA_SCORE_KERNEL_SOURCE,
        .inputs = &.{
            .{ .name = "Q", .ty = fix(.bfloat16) },
            .{ .name = "POOL", .ty = fix(.bfloat16) },
        },
        .outputs = &.{.{ .name = "OUT", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "NSGVAL", .kind = .ct_int },
            .{ .name = "USEH4VAL", .kind = .ct_int },
        },
        .ct_tuples = &.{
            &.{ ci(4), ci(0) },
            &.{ ci(8), ci(1) },
        },
    },
    .{
        .name = "sushi_qsa_pool_rope",
        .source = qsel.QSA_POOL_ROPE_SOURCE,
        .inputs = &.{
            .{ .name = "kb", .ty = par("T") },
            .{ .name = "norm_w", .ty = par("T") },
            .{ .name = "cosv", .ty = par("T") },
            .{ .name = "sinv", .ty = par("T") },
            .{ .name = "eps", .ty = fix(.float32), .scalar = true },
            .{ .name = "inv_r", .ty = fix(.float32), .scalar = true },
            .{ .name = "nblk", .ty = fix(.int32), .scalar = true },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "R", .kind = .fc_int },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "HALF", .kind = .fc_int },
        },
    },

    // ── qwen4_qsa_gather.zig (2 SIMD + 2 NAX) ──
    .{
        .name = "sushi_attn_qsa256",
        .header = attn.ATTN_QSA256_KERNEL_HEADER,
        .source = qgat.ATTN_QSA256_KERNEL_SOURCE,
        .inputs = &.{
            .{ .name = "q", .ty = par("T") },
            .{ .name = "k", .ty = par("T") },
            .{ .name = "v", .ty = par("T") },
            .{ .name = "scl", .ty = fix(.float32) },
            .{ .name = "blocks", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{ 1, 2, 3, 4, 5, 6, 7, 8 } },
            .{ .name = "BK", .kind = .ct_int, .ints = &.{32} },
            .{ .name = "RATIO", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_attn_qsa256_packed",
        .header = qgat.QSA_PACKED_HEADER ++ attn.ATTN_QSA256_KERNEL_HEADER,
        .source = qgat.ATTN_QSA256_KERNEL_SOURCE,
        .inputs = &.{
            .{ .name = "q", .ty = par("T") },
            .{ .name = "k", .ty = fix(.uint32) },
            .{ .name = "v", .ty = fix(.uint32) },
            .{ .name = "scl", .ty = fix(.float32) },
            .{ .name = "blocks", .ty = fix(.int32) },
            .{ .name = "ksc", .ty = par("T") },
            .{ .name = "kbi", .ty = par("T") },
            .{ .name = "vsc", .ty = par("T") },
            .{ .name = "vbi", .ty = par("T") },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{ 1, 2, 3, 4, 5, 6, 7, 8 } },
            .{ .name = "BK", .kind = .ct_int, .ints = &.{32} },
            .{ .name = "RATIO", .kind = .fc_int },
            .{ .name = "BITS", .kind = .ct_int, .ints = &.{ 2, 3, 4, 6, 8 } },
            .{ .name = "GS", .kind = .ct_int, .ints = &.{ 32, 64, 128 } },
        },
    },
    .{
        .name = "sushi_qsa_nax_precise",
        .nax = true,
        .header = @embedFile("../kernels/qsa_nax_header.metal"),
        .source = @embedFile("../kernels/qsa_nax.metal"),
        .inputs = &.{
            .{ .name = "q", .ty = par("T") },
            .{ .name = "k", .ty = par("T") },
            .{ .name = "v", .ty = par("T") },
            .{ .name = "scl", .ty = fix(.float32) },
            .{ .name = "blocks", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "NSG", .kind = .ignored },
            .{ .name = "BK", .kind = .ignored },
            .{ .name = "RATIO", .kind = .ignored },
        },
    },
    .{
        .name = "sushi_qsa_nax_packed",
        .nax = true,
        .header = qgat.QSA_PACKED_HEADER ++ @embedFile("../kernels/qsa_nax_header.metal"),
        .source = @embedFile("../kernels/qsa_nax.metal"),
        .inputs = &.{
            .{ .name = "q", .ty = par("T") },
            .{ .name = "k", .ty = fix(.uint32) },
            .{ .name = "v", .ty = fix(.uint32) },
            .{ .name = "scl", .ty = fix(.float32) },
            .{ .name = "blocks", .ty = fix(.int32) },
            .{ .name = "ksc", .ty = par("T") },
            .{ .name = "kbi", .ty = par("T") },
            .{ .name = "vsc", .ty = par("T") },
            .{ .name = "vbi", .ty = par("T") },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{2} },
            .{ .name = "BK", .kind = .ct_int, .ints = &.{32} },
            .{ .name = "RATIO", .kind = .fc_int },
            .{ .name = "BITS", .kind = .ct_int, .ints = &.{ 2, 3, 4, 6, 8 } },
            .{ .name = "GS", .kind = .ct_int, .ints = &.{ 32, 64, 128 } },
        },
    },

    // ── qwen4_qsa.zig (3 SIMD kernels) ──
    .{
        .name = "sushi_qsa_select",
        .header = qsa.QSA_SELECT_KERNEL_HEADER,
        .source = qsa.QSA_SELECT_KERNEL_SOURCE,
        .inputs = &.{
            .{ .name = "scores", .ty = fix(.float32) },
            .{ .name = "bounds", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "ids", .ty = fix(.int32) }},
        .params = &qsa_select_params,
    },
    .{
        .name = "sushi_qsa_select_split_local",
        .header = qsa.QSA_SELECT_KERNEL_HEADER,
        .source = qsa.QSA_SELECT_SPLIT_LOCAL_SOURCE,
        .inputs = &.{
            .{ .name = "scores", .ty = fix(.float32) },
            .{ .name = "bounds", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "ids", .ty = fix(.int32) }},
        .params = &qsa_select_params,
    },
    .{
        .name = "sushi_qsa_select_split_merge",
        .header = qsa.QSA_SELECT_KERNEL_HEADER,
        .source = qsa.QSA_SELECT_SPLIT_MERGE_SOURCE,
        .inputs = &.{
            .{ .name = "scores", .ty = fix(.float32) },
            .{ .name = "local_ids", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "ids", .ty = fix(.int32) }},
        .params = &qsa_select_params,
    },

    // ── qwen4_hc.zig (3 SIMD kernels) ──
    .{
        .name = "sushi_hc_read_n",
        .source = qhc.HC_FUSED_N_SOURCE,
        .inputs = &.{
            .{ .name = "x_in", .ty = par("T") },
            .{ .name = "nw", .ty = par("T") },
            .{ .name = "iw", .ty = par("T") },
            .{ .name = "eps", .ty = fix(.float32) },
            .{ .name = "wo_in", .ty = par("T") },
            .{ .name = "wi_in", .ty = par("T") },
        },
        .outputs = &.{
            .{ .name = "xn_out", .ty = par("T") },
            .{ .name = "ipart_out", .ty = fix(.float32) },
            .{ .name = "xs_out", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &f16s },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "H", .kind = .fc_int },
            .{ .name = "INJ", .kind = .fc_int },
            .{ .name = "WR", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_hc_read_d",
        .source = qhc.HC_FUSED_D_SOURCE,
        .inputs = &.{
            .{ .name = "xn_in", .ty = par("T") },
            .{ .name = "dw_q", .ty = fix(.uint32) },
            .{ .name = "dw_s", .ty = par("T") },
            .{ .name = "dw_b", .ty = par("T") },
            .{ .name = "ipart_in", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "act_out", .ty = par("T") },
            .{ .name = "inj_out", .ty = par("T") },
        },
        .params = &hc_du_params,
    },
    .{
        .name = "sushi_hc_read_u",
        .source = qhc.HC_FUSED_U_SOURCE,
        .inputs = &.{
            .{ .name = "xn_in", .ty = par("T") },
            .{ .name = "act_in", .ty = par("T") },
            .{ .name = "uw_q", .ty = fix(.uint32) },
            .{ .name = "uw_s", .ty = par("T") },
            .{ .name = "uw_b", .ty = par("T") },
        },
        .outputs = &.{.{ .name = "mixed_out", .ty = par("T") }},
        .params = &hc_du_params,
    },

    // ── qwen4_forward.zig (2 SIMD kernels) ──
    .{
        .name = "sushi_qk_norm_rope",
        .source = qfwd.QK_NORM_ROPE_SOURCE,
        .inputs = &.{
            .{ .name = "q", .ty = par("T") },
            .{ .name = "k", .ty = par("T") },
            .{ .name = "qw", .ty = par("T") },
            .{ .name = "kw", .ty = par("T") },
            .{ .name = "angles", .ty = fix(.float32) },
            .{ .name = "ms", .ty = par("T") },
            .{ .name = "eps", .ty = fix(.float32), .scalar = true },
        },
        .outputs = &.{
            .{ .name = "oq", .ty = par("T") },
            .{ .name = "ok", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "HQ", .kind = .fc_int },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "SCALED", .kind = .fc_int },
        },
    },
    .{
        .name = "sushi_qk_norm_rope_256",
        .source = qfwd.QK_NORM_ROPE_256_SOURCE,
        .inputs = &.{
            .{ .name = "q", .ty = par("T") },
            .{ .name = "k", .ty = par("T") },
            .{ .name = "qw", .ty = par("T") },
            .{ .name = "kw", .ty = par("T") },
            .{ .name = "angles", .ty = fix(.float32) },
            .{ .name = "eps", .ty = fix(.float32), .scalar = true },
        },
        .outputs = &.{
            .{ .name = "oq", .ty = par("T") },
            .{ .name = "ok", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "HQ", .kind = .fc_int },
            .{ .name = "HK", .kind = .fc_int },
            .{ .name = "RD", .kind = .fc_int },
        },
    },
};
