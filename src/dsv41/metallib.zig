//! DSv41 precompiled custom kernels catalog (`sushi.metallib`).
//! Covers every DSv41 custom kernel across `src/dsv41/{dsv41_qat,dsv41_kernels,ubench}.zig`.
//! Single source of truth for `gen-metallib`, runtime `kernel.zig` resolution, and `stock.zig` completeness test.

const custom = @import("../tensor/metallib_custom.zig");
const qat = @import("dsv41_qat.zig");
const k = @import("dsv41_kernels.zig");
const ub = @import("ubench.zig");

const KernelSpec = custom.KernelSpec;
const InputSlot = custom.InputSlot;
const OutputSlot = custom.OutputSlot;
const CtVal = custom.CtVal;
const Dtype = custom.Dtype;

const bf_f32 = [_]Dtype{ .bfloat16, .float32 };
const i32_u32 = [_]Dtype{ .uint32, .int32 };

fn fix(d: Dtype) custom.TypeRef {
    return .{ .fixed = d };
}
fn par(n: []const u8) custom.TypeRef {
    return .{ .param = n };
}
fn ci(v: i32) CtVal {
    return .{ .int = v };
}

fn sameFirstTwoDtypes(vals: []const CtVal) bool {
    return vals[0].dtype == vals[1].dtype;
}

fn attnDecFilter(vals: []const CtVal) bool {
    // vals: [D, NSG, KT, OT, QT, ST]
    const kt = vals[2].dtype;
    const ot = vals[3].dtype;
    const qt = vals[4].dtype;
    return qt == kt or qt == ot;
}

fn attnSplitFilter(vals: []const CtVal) bool {
    // vals: [D, NSG, KT, QT, ST]
    const kt = vals[2].dtype;
    const qt = vals[3].dtype;
    return qt == kt;
}

const attn_dec_ins = [_]InputSlot{
    .{ .name = "q", .ty = par("QT") },
    .{ .name = "ring", .ty = par("KT") },
    .{ .name = "comp", .ty = par("KT") },
    .{ .name = "topk", .ty = fix(.int32) },
    .{ .name = "sink", .ty = par("ST") },
    .{ .name = "meta", .ty = fix(.int32) },
    .{ .name = "scal", .ty = fix(.float32) },
    .{ .name = "freqs", .ty = fix(.float32) },
};

const attn_dec_packed_ins = [_]InputSlot{
    .{ .name = "q", .ty = par("QT") },
    .{ .name = "ring", .ty = par("KT") },
    .{ .name = "ccodes", .ty = fix(.uint8) },
    .{ .name = "cscales", .ty = fix(.uint8) },
    .{ .name = "topk", .ty = fix(.int32) },
    .{ .name = "sink", .ty = par("ST") },
    .{ .name = "meta", .ty = fix(.int32) },
    .{ .name = "scal", .ty = fix(.float32) },
    .{ .name = "freqs", .ty = fix(.float32) },
};

const attn_split_ins = [_]InputSlot{
    .{ .name = "q", .ty = par("QT") },
    .{ .name = "ring", .ty = par("KT") },
    .{ .name = "comp", .ty = par("KT") },
    .{ .name = "topk", .ty = fix(.int32) },
    .{ .name = "sink", .ty = par("ST") },
    .{ .name = "meta", .ty = fix(.int32) },
    .{ .name = "scal", .ty = fix(.float32) },
};

const attn_split_packed_ins = [_]InputSlot{
    .{ .name = "q", .ty = par("QT") },
    .{ .name = "ring", .ty = par("KT") },
    .{ .name = "ccodes", .ty = fix(.uint8) },
    .{ .name = "cscales", .ty = fix(.uint8) },
    .{ .name = "topk", .ty = fix(.int32) },
    .{ .name = "sink", .ty = par("ST") },
    .{ .name = "meta", .ty = fix(.int32) },
    .{ .name = "scal", .ty = fix(.float32) },
};

const ride_extra_ins = [_]InputSlot{
    .{ .name = "z", .ty = fix(.float32) },
    .{ .name = "eps", .ty = fix(.float32) },
};

const ride_extra_outs = [_]OutputSlot{
    .{ .name = "pre", .ty = fix(.float32) },
    .{ .name = "post", .ty = fix(.float32) },
    .{ .name = "comb", .ty = fix(.float32) },
};

const attn_fused_ins = [_]InputSlot{
    .{ .name = "q", .ty = par("T") },
    .{ .name = "keys", .ty = par("T") },
    .{ .name = "comp", .ty = par("T") },
    .{ .name = "topk", .ty = fix(.int32) },
    .{ .name = "sink", .ty = par("ST") },
    .{ .name = "meta", .ty = fix(.int32) },
    .{ .name = "scal", .ty = fix(.float32) },
    .{ .name = "freqs", .ty = fix(.float32) },
};

const attn_fused_fp4_ins = [_]InputSlot{
    .{ .name = "q", .ty = par("T") },
    .{ .name = "keys", .ty = par("T") },
    .{ .name = "ccodes", .ty = fix(.uint8) },
    .{ .name = "cscales", .ty = fix(.uint8) },
    .{ .name = "topk", .ty = fix(.int32) },
    .{ .name = "sink", .ty = par("ST") },
    .{ .name = "meta", .ty = fix(.int32) },
    .{ .name = "scal", .ty = fix(.float32) },
    .{ .name = "freqs", .ty = fix(.float32) },
};

const idx_nax_ins = [_]InputSlot{
    .{ .name = "qa", .ty = fix(.float16) },
    .{ .name = "pq", .ty = fix(.float32) },
    .{ .name = "sqr", .ty = fix(.int32) },
    .{ .name = "qf", .ty = fix(.float32) },
    .{ .name = "codes", .ty = fix(.uint8) },
    .{ .name = "scales", .ty = fix(.uint8) },
    .{ .name = "w", .ty = par("W") },
    .{ .name = "meta", .ty = fix(.int32) },
    .{ .name = "wsc", .ty = fix(.float32) },
};

const idx_nax_tuples = [_][]const CtVal{
    &.{ ci(1), ci(6), .{ .dtype = .bfloat16 } },
    &.{ ci(2), ci(3), .{ .dtype = .bfloat16 } },
    &.{ ci(3), ci(2), .{ .dtype = .bfloat16 } },
    &.{ ci(4), ci(1), .{ .dtype = .bfloat16 } },
    &.{ ci(1), ci(6), .{ .dtype = .float32 } },
    &.{ ci(2), ci(3), .{ .dtype = .float32 } },
    &.{ ci(3), ci(2), .{ .dtype = .float32 } },
    &.{ ci(4), ci(1), .{ .dtype = .float32 } },
};

pub const specs: []const KernelSpec = &.{
    // ── dsv41_qat.zig (5 SIMD kernels) ──
    .{
        .name = "dsv41_qat",
        .header = qat.QAT_HEADER,
        .source = qat.QAT_SOURCE,
        .inputs = &.{.{ .name = "x", .ty = par("T") }},
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "G", .kind = .fc_int },
            .{ .name = "MODE", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_pack_fp4",
        .header = qat.PACK_HEADER,
        .source = qat.PACK_SOURCE,
        .inputs = &.{.{ .name = "x", .ty = par("X") }},
        .outputs = &.{
            .{ .name = "codes", .ty = fix(.uint8) },
            .{ .name = "scales", .ty = fix(.uint8) },
        },
        .params = &.{
            .{ .name = "X", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "G", .kind = .fc_int },
            .{ .name = "MODE", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_unpack_fp4",
        .header = qat.PACK_HEADER,
        .source = qat.UNPACK_SOURCE,
        .inputs = &.{
            .{ .name = "codes", .ty = fix(.uint8) },
            .{ .name = "scales", .ty = fix(.uint8) },
        },
        .outputs = &.{.{ .name = "y", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "G", .kind = .fc_int },
            .{ .name = "MODE", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_unpack_picks",
        .header = qat.PACK_HEADER,
        .source = qat.UNPACK_PICKS_SOURCE,
        .inputs = &.{
            .{ .name = "codes", .ty = fix(.uint8) },
            .{ .name = "scales", .ty = fix(.uint8) },
            .{ .name = "topk", .ty = fix(.int32) },
        },
        .outputs = &.{
            .{ .name = "y", .ty = par("T") },
            .{ .name = "ids", .ty = fix(.int32) },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "G", .kind = .fc_int },
            .{ .name = "MODE", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_hc_split",
        .disambig = "4in",
        .source = qat.HC_SPLIT_SOURCE,
        .inputs = &.{
            .{ .name = "mixes", .ty = fix(.float32) },
            .{ .name = "scale", .ty = fix(.float32) },
            .{ .name = "base", .ty = fix(.float32) },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "pre", .ty = fix(.float32) },
            .{ .name = "post", .ty = fix(.float32) },
            .{ .name = "comb", .ty = fix(.float32) },
        },
        .params = &.{
            .{ .name = "ITERS", .kind = .fc_int },
        },
    },

    // ── dsv41_kernels.zig: MoE (4 SIMD kernels) ──
    .{
        .name = "dsv41_moe_gu",
        .header = k.EXPERT_HEADER,
        .source = k.MOE_GU_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "gw", .ty = fix(.uint32) },
            .{ .name = "gs", .ty = fix(.uint8) },
            .{ .name = "uw", .ty = fix(.uint32) },
            .{ .name = "us", .ty = fix(.uint8) },
            .{ .name = "sgw", .ty = fix(.uint32) },
            .{ .name = "sgs", .ty = fix(.uint8) },
            .{ .name = "suw", .ty = fix(.uint32) },
            .{ .name = "sus", .ty = fix(.uint8) },
            .{ .name = "inds", .ty = par("IND") },
            .{ .name = "wts", .ty = fix(.float32) },
            .{ .name = "lim", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "RPS", .kind = .ct_int, .ints = &.{2} },
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "N", .kind = .fc_int },
            .{ .name = "TOPK", .kind = .fc_int },
            .{ .name = "IND", .kind = .io_dtype, .dtypes = &i32_u32 },
        },
    },
    .{
        .name = "dsv41_moe_dn",
        .header = k.EXPERT_HEADER,
        .source = k.MOE_DN_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "dw", .ty = fix(.uint32) },
            .{ .name = "ds", .ty = fix(.uint8) },
            .{ .name = "sdw", .ty = fix(.uint32) },
            .{ .name = "sds", .ty = fix(.uint8) },
            .{ .name = "inds", .ty = par("IND") },
            .{ .name = "res", .ty = par("T") },
            .{ .name = "post", .ty = fix(.float32) },
            .{ .name = "comb", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "RPS", .kind = .ct_int, .ints = &.{2} },
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "N", .kind = .fc_int },
            .{ .name = "TOPK", .kind = .fc_int },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "IND", .kind = .io_dtype, .dtypes = &i32_u32 },
        },
    },
    .{
        .name = "dsv41_moe_act",
        .source = k.MOE_ACT_SOURCE,
        .inputs = &.{
            .{ .name = "g", .ty = par("T") },
            .{ .name = "u", .ty = par("T") },
            .{ .name = "wts", .ty = fix(.float32) },
            .{ .name = "lim", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "N", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_moe_combine",
        .source = k.MOE_COMBINE_SOURCE,
        .inputs = &.{
            .{ .name = "dn", .ty = par("T") },
            .{ .name = "inv", .ty = par("INV") },
            .{ .name = "sh", .ty = par("T") },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "INV", .kind = .io_dtype, .dtypes = &i32_u32 },
            .{ .name = "D", .kind = .fc_int },
            .{ .name = "K", .kind = .fc_int },
        },
    },

    // ── dsv41_kernels.zig: router, top-k, dense f32 (5 SIMD kernels) ──
    .{
        .name = "dsv41_gate",
        .source = k.GATE_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("X") },
            .{ .name = "w", .ty = par("W") },
        },
        .outputs = &.{.{ .name = "logits", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "X", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "W", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "SGS", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_route",
        .source = k.ROUTE_SOURCE,
        .inputs = &.{
            .{ .name = "logits", .ty = fix(.float32) },
            .{ .name = "bias", .ty = fix(.float32) },
            .{ .name = "scal", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "idx", .ty = fix(.uint32) },
            .{ .name = "wts", .ty = fix(.float32) },
        },
        .params = &.{
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "NORM", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_route_split",
        .source = k.ROUTE_SPLIT_SOURCE,
        .inputs = &.{
            .{ .name = "logits", .ty = fix(.float32) },
            .{ .name = "bias", .ty = fix(.float32) },
            .{ .name = "scal", .ty = fix(.float32) },
            .{ .name = "z", .ty = fix(.float32) },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "idx", .ty = fix(.uint32) },
            .{ .name = "wts", .ty = fix(.float32) },
            .{ .name = "pre", .ty = fix(.float32) },
            .{ .name = "post", .ty = fix(.float32) },
            .{ .name = "comb", .ty = fix(.float32) },
        },
        .params = &.{
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "NORM", .kind = .fc_int },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "ITERS", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_route_split_slots",
        .source = k.ROUTE_SPLIT_SLOTS_SOURCE,
        .inputs = &.{
            .{ .name = "logits", .ty = fix(.float32) },
            .{ .name = "bias", .ty = fix(.float32) },
            .{ .name = "scal", .ty = fix(.float32) },
            .{ .name = "z", .ty = fix(.float32) },
            .{ .name = "eps", .ty = fix(.float32) },
            .{ .name = "map", .ty = fix(.int32) },
        },
        .outputs = &.{
            .{ .name = "idx", .ty = fix(.uint32) },
            .{ .name = "wts", .ty = fix(.float32) },
            .{ .name = "pre", .ty = fix(.float32) },
            .{ .name = "post", .ty = fix(.float32) },
            .{ .name = "comb", .ty = fix(.float32) },
            .{ .name = "local", .ty = fix(.int32) },
        },
        .params = &.{
            .{ .name = "K", .kind = .fc_int },
            .{ .name = "NORM", .kind = .fc_int },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "ITERS", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_dense_f32x",
        .source = k.DENSE_F32X_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("OT") },
            .{ .name = "w", .ty = par("W") },
        },
        .outputs = &.{.{ .name = "out", .ty = par("OT") }},
        .params = &.{
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "W", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "RPS", .kind = .ct_int, .ints = &.{4} },
        },
    },

    // ── dsv41_kernels.zig: rotary & kv row (4 SIMD kernels) ──
    .{
        .name = "dsv41_rope_tail",
        .header = qat.QAT_HEADER,
        .source = k.ROPE_TAIL_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "freqs", .ty = fix(.float32) },
            .{ .name = "pos", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "out", .ty = par("OT") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "D", .kind = .fc_int },
            .{ .name = "ROW", .kind = .fc_int },
            .{ .name = "QK", .kind = .fc_int },
            .{ .name = "INV", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_kv_row",
        .header = qat.QAT_HEADER,
        .source = k.KV_ROW_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("T") },
            .{ .name = "w", .ty = par("T") },
            .{ .name = "freqs", .ty = fix(.float32) },
            .{ .name = "pos", .ty = fix(.int32) },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "D", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_rope_kv",
        .header = qat.QAT_HEADER,
        .source = k.ROPE_KV_SOURCE,
        .inputs = &.{
            .{ .name = "qx", .ty = par("T") },
            .{ .name = "kx", .ty = par("T") },
            .{ .name = "w", .ty = par("T") },
            .{ .name = "freqs", .ty = fix(.float32) },
            .{ .name = "pos", .ty = fix(.int32) },
            .{ .name = "eps", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "qout", .ty = par("OT") },
            .{ .name = "kout", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "D", .kind = .fc_int },
            .{ .name = "ROW", .kind = .fc_int },
            .{ .name = "QK", .kind = .fc_int },
            .{ .name = "INV", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
        },
        .filter = sameFirstTwoDtypes,
    },
    .{
        .name = "dsv41_rope_kv_ring",
        .header = qat.QAT_HEADER,
        .source = k.ROPE_KV_RING_SOURCE,
        .inputs = &.{
            .{ .name = "qx", .ty = par("T") },
            .{ .name = "kx", .ty = par("T") },
            .{ .name = "w", .ty = par("T") },
            .{ .name = "freqs", .ty = fix(.float32) },
            .{ .name = "pos", .ty = fix(.int32) },
            .{ .name = "eps", .ty = fix(.float32) },
            .{ .name = "ring", .ty = par("T") },
        },
        .outputs = &.{
            .{ .name = "qout", .ty = par("OT") },
            .{ .name = "rout", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "D", .kind = .fc_int },
            .{ .name = "ROW", .kind = .fc_int },
            .{ .name = "QK", .kind = .fc_int },
            .{ .name = "INV", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
            .{ .name = "QTG", .kind = .fc_int },
        },
        .filter = sameFirstTwoDtypes,
    },
    .{
        .name = "dsv41_rope_tail_ip",
        .source = k.ROPE_TAIL_IP_SOURCE,
        .inputs = &.{
            .{ .name = "freqs", .ty = fix(.float32) },
            .{ .name = "pos", .ty = fix(.int32) },
            .{ .name = "meta", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "qout", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "D", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_kv_row_ring_ip",
        .header = qat.QAT_HEADER,
        .source = k.KV_ROW_RING_IP_SOURCE,
        .inputs = &.{
            .{ .name = "kx", .ty = par("T") },
            .{ .name = "w", .ty = par("T") },
            .{ .name = "freqs", .ty = fix(.float32) },
            .{ .name = "pos", .ty = fix(.int32) },
            .{ .name = "eps", .ty = fix(.float32) },
            .{ .name = "meta", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "rout", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "D", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_rope_kv_ring_ip",
        .header = qat.QAT_HEADER,
        .source = k.ROPE_KV_RING_IP_SOURCE,
        .inputs = &.{
            .{ .name = "kx", .ty = par("T") },
            .{ .name = "w", .ty = par("T") },
            .{ .name = "freqs", .ty = fix(.float32) },
            .{ .name = "pos", .ty = fix(.int32) },
            .{ .name = "eps", .ty = fix(.float32) },
            .{ .name = "meta", .ty = fix(.int32) },
        },
        .outputs = &.{
            .{ .name = "qout", .ty = par("T") },
            .{ .name = "rout", .ty = par("T") },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "D", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
        },
    },

    // ── dsv41_kernels.zig: attention decode/split/merge/softmax (10 SIMD kernels) ──
    .{
        .name = "dsv41_attn_decode",
        .header = k.HEADS_HEADER,
        .source = k.ATTN_DEC_SOURCE,
        .inputs = &attn_dec_ins,
        .outputs = &.{.{ .name = "out", .ty = par("OT") }},
        .params = &.{
            .{ .name = "D", .kind = .ct_int, .ints = &.{ 64, 128, 256, 512 } },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{8} },
            .{ .name = "KT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "QT", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
            .{ .name = "OG", .kind = .fc_int },
        },
        .filter = attnDecFilter,
    },
    .{
        .name = "dsv41_attn_decode_ride",
        .header = k.HEADS_HEADER,
        .source = k.ATTN_DEC_RIDE_SOURCE,
        .inputs = &(attn_dec_ins ++ ride_extra_ins),
        .outputs = &([_]OutputSlot{.{ .name = "out", .ty = par("OT") }} ++ ride_extra_outs),
        .params = &.{
            .{ .name = "D", .kind = .ct_int, .ints = &.{ 64, 128, 256, 512 } },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{8} },
            .{ .name = "KT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "QT", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
            .{ .name = "OG", .kind = .fc_int },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "ITERS", .kind = .fc_int },
        },
        .filter = attnDecFilter,
    },
    .{
        .name = "dsv41_attn_split",
        .source = k.ATTN_SPLIT_SOURCE,
        .inputs = &attn_split_ins,
        .outputs = &.{
            .{ .name = "pacc", .ty = fix(.float32) },
            .{ .name = "pm", .ty = fix(.float32) },
            .{ .name = "pl", .ty = fix(.float32) },
        },
        .params = &.{
            .{ .name = "D", .kind = .ct_int, .ints = &.{ 64, 128, 256, 512 } },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{8} },
            .{ .name = "KT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "QT", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
        },
        .filter = attnSplitFilter,
    },
    .{
        .name = "dsv41_attn_split_ride",
        .source = k.ATTN_SPLIT_RIDE_SOURCE,
        .inputs = &(attn_split_ins ++ ride_extra_ins),
        .outputs = &([_]OutputSlot{
            .{ .name = "pacc", .ty = fix(.float32) },
            .{ .name = "pm", .ty = fix(.float32) },
            .{ .name = "pl", .ty = fix(.float32) },
        } ++ ride_extra_outs),
        .params = &.{
            .{ .name = "D", .kind = .ct_int, .ints = &.{ 64, 128, 256, 512 } },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{8} },
            .{ .name = "KT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "QT", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "ITERS", .kind = .fc_int },
        },
        .filter = attnSplitFilter,
    },
    .{
        .name = "dsv41_attn_decode_packed",
        .header = k.HEADS_HEADER ++ "\n" ++ qat.PACK_HEADER,
        .source = k.ATTN_DEC_PACKED_SOURCE,
        .inputs = &attn_dec_packed_ins,
        .outputs = &.{.{ .name = "out", .ty = par("OT") }},
        .params = &.{
            .{ .name = "D", .kind = .ct_int, .ints = &.{ 64, 128, 256, 512 } },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{8} },
            .{ .name = "KT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "QT", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
            .{ .name = "OG", .kind = .fc_int },
            .{ .name = "G", .kind = .fc_int },
            .{ .name = "MODE", .kind = .fc_int },
        },
        .filter = attnDecFilter,
    },
    .{
        .name = "dsv41_attn_decode_packed_ride",
        .header = k.HEADS_HEADER ++ "\n" ++ qat.PACK_HEADER,
        .source = k.ATTN_DEC_PACKED_RIDE_SOURCE,
        .inputs = &(attn_dec_packed_ins ++ ride_extra_ins),
        .outputs = &([_]OutputSlot{.{ .name = "out", .ty = par("OT") }} ++ ride_extra_outs),
        .params = &.{
            .{ .name = "D", .kind = .ct_int, .ints = &.{ 64, 128, 256, 512 } },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{8} },
            .{ .name = "KT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "QT", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
            .{ .name = "OG", .kind = .fc_int },
            .{ .name = "G", .kind = .fc_int },
            .{ .name = "MODE", .kind = .fc_int },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "ITERS", .kind = .fc_int },
        },
        .filter = attnDecFilter,
    },
    .{
        .name = "dsv41_attn_split_packed",
        .header = qat.PACK_HEADER,
        .source = k.ATTN_SPLIT_PACKED_SOURCE,
        .inputs = &attn_split_packed_ins,
        .outputs = &.{
            .{ .name = "pacc", .ty = fix(.float32) },
            .{ .name = "pm", .ty = fix(.float32) },
            .{ .name = "pl", .ty = fix(.float32) },
        },
        .params = &.{
            .{ .name = "D", .kind = .ct_int, .ints = &.{ 64, 128, 256, 512 } },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{8} },
            .{ .name = "KT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "QT", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "G", .kind = .fc_int },
            .{ .name = "MODE", .kind = .fc_int },
        },
        .filter = attnSplitFilter,
    },
    .{
        .name = "dsv41_attn_split_packed_ride",
        .header = qat.PACK_HEADER,
        .source = k.ATTN_SPLIT_PACKED_RIDE_SOURCE,
        .inputs = &(attn_split_packed_ins ++ ride_extra_ins),
        .outputs = &([_]OutputSlot{
            .{ .name = "pacc", .ty = fix(.float32) },
            .{ .name = "pm", .ty = fix(.float32) },
            .{ .name = "pl", .ty = fix(.float32) },
        } ++ ride_extra_outs),
        .params = &.{
            .{ .name = "D", .kind = .ct_int, .ints = &.{ 64, 128, 256, 512 } },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{8} },
            .{ .name = "KT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "QT", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "G", .kind = .fc_int },
            .{ .name = "MODE", .kind = .fc_int },
            .{ .name = "HC", .kind = .fc_int },
            .{ .name = "ITERS", .kind = .fc_int },
        },
        .filter = attnSplitFilter,
    },
    .{
        .name = "dsv41_attn_merge",
        .header = k.HEADS_HEADER,
        .source = k.ATTN_MERGE_SOURCE,
        .inputs = &.{
            .{ .name = "pacc", .ty = fix(.float32) },
            .{ .name = "pm", .ty = fix(.float32) },
            .{ .name = "pl", .ty = fix(.float32) },
            .{ .name = "sink", .ty = par("ST") },
            .{ .name = "meta", .ty = fix(.int32) },
            .{ .name = "freqs", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "out", .ty = par("OT") }},
        .params = &.{
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "D", .kind = .fc_int },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
            .{ .name = "OG", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_attn_softmax",
        .source = k.ATTN_SOFTMAX_SOURCE,
        .inputs = &.{
            .{ .name = "lw", .ty = fix(.float16) },
            .{ .name = "lc", .ty = fix(.float16) },
            .{ .name = "topk", .ty = fix(.int32) },
            .{ .name = "sink", .ty = par("ST") },
            .{ .name = "meta", .ty = fix(.int32) },
            .{ .name = "scal", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "pw", .ty = par("OT") },
            .{ .name = "pc", .ty = par("OT") },
        },
        .params = &.{
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &.{.float16} },
            .{ .name = "NJ", .kind = .ct_int, .ints = &.{ 4, 8, 12, 16, 20, 24, 28, 32 } },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
        },
    },

    // ── dsv41_kernels.zig: fused prefill attention (2 SIMD + 2 NAX) ──
    .{
        .name = "dsv41_attn_fused",
        .header = k.ATTN_EPI_HEADER,
        .source = k.ATTN_FUSED_SOURCE,
        .inputs = &attn_fused_ins,
        .outputs = &.{.{ .name = "out", .ty = par("OT") }},
        .params = &.{
            .{ .name = "D", .kind = .ct_int, .ints = &.{ 64, 128, 256, 512 } },
            .{ .name = "HB", .kind = .ct_int, .ints = &.{16} },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{8} },
            .{ .name = "KT", .kind = .ct_int, .ints = &.{16} },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "EPI", .kind = .fc_int },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
            .{ .name = "GIN", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_attn_fused_fp4",
        .header = k.ATTN_EPI_HEADER ++ qat.PACK_HEADER,
        .source = k.ATTN_FUSED_FP4_SOURCE,
        .inputs = &attn_fused_fp4_ins,
        .outputs = &.{.{ .name = "out", .ty = par("OT") }},
        .params = &.{
            .{ .name = "D", .kind = .ct_int, .ints = &.{ 64, 128, 256, 512 } },
            .{ .name = "HB", .kind = .ct_int, .ints = &.{16} },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{8} },
            .{ .name = "KT", .kind = .ct_int, .ints = &.{16} },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "EPI", .kind = .fc_int },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
            .{ .name = "GIN", .kind = .fc_int },
            .{ .name = "G", .kind = .fc_int },
            .{ .name = "MODE", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_attn_nax",
        .nax = true,
        .header = k.ATTN_NAX_HEADER ++ k.ATTN_EPI_HEADER,
        .source = k.ATTN_NAX_SOURCE,
        .inputs = &attn_fused_ins,
        .outputs = &.{.{ .name = "out", .ty = par("OT") }},
        .params = &.{
            .{ .name = "D", .kind = .ct_int, .ints = &.{ 256, 512 } },
            .{ .name = "HB", .kind = .ct_int, .ints = &.{16} },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{8} },
            .{ .name = "KT", .kind = .ct_int, .ints = &.{16} },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "EPI", .kind = .fc_int },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
            .{ .name = "GIN", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_attn_nax_fp4",
        .nax = true,
        .header = k.ATTN_NAX_HEADER ++ k.ATTN_EPI_HEADER ++ qat.PACK_HEADER,
        .source = k.ATTN_NAX_FP4_SOURCE,
        .inputs = &attn_fused_fp4_ins,
        .outputs = &.{.{ .name = "out", .ty = par("OT") }},
        .params = &.{
            .{ .name = "D", .kind = .ct_int, .ints = &.{ 256, 512 } },
            .{ .name = "HB", .kind = .ct_int, .ints = &.{16} },
            .{ .name = "NSG", .kind = .ct_int, .ints = &.{8} },
            .{ .name = "KT", .kind = .ct_int, .ints = &.{16} },
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "ST", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "EPI", .kind = .fc_int },
            .{ .name = "RD", .kind = .fc_int },
            .{ .name = "QAT", .kind = .fc_int },
            .{ .name = "GIN", .kind = .fc_int },
            .{ .name = "G", .kind = .fc_int },
            .{ .name = "MODE", .kind = .fc_int },
        },
    },

    // ── dsv41_kernels.zig: indexer (4 SIMD + 2 NAX) ──
    .{
        .name = "dsv41_index_scores_fp4",
        .header = k.IDX_PACKED_HEADER,
        .source = k.IDX_SCORE_PACKED_SOURCE,
        .inputs = &.{
            .{ .name = "q", .ty = fix(.float32) },
            .{ .name = "codes", .ty = fix(.uint8) },
            .{ .name = "scales", .ty = fix(.uint8) },
            .{ .name = "w", .ty = par("W") },
            .{ .name = "meta", .ty = fix(.int32) },
            .{ .name = "wsc", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "scores", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "IH", .kind = .ct_int, .ints = &.{ 4, 8, 16, 32 } },
            .{ .name = "IHD", .kind = .ct_int, .ints = &.{ 32, 64, 128 } },
            .{ .name = "W", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "G", .kind = .fc_int },
            .{ .name = "MODE", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_index_scores_gather",
        .header = k.IDX_PACKED_HEADER,
        .source = k.IDX_GATHER_SOURCE,
        .inputs = &.{
            .{ .name = "q", .ty = fix(.float32) },
            .{ .name = "codes", .ty = fix(.uint8) },
            .{ .name = "scales", .ty = fix(.uint8) },
            .{ .name = "w", .ty = par("W") },
            .{ .name = "meta", .ty = fix(.int32) },
            .{ .name = "wsc", .ty = fix(.float32) },
            .{ .name = "blocks", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "scores", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "IH", .kind = .ct_int, .ints = &.{ 4, 8, 16, 32 } },
            .{ .name = "IHD", .kind = .ct_int, .ints = &.{ 32, 64, 128 } },
            .{ .name = "W", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "G", .kind = .fc_int },
            .{ .name = "MODE", .kind = .fc_int },
            .{ .name = "BS", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_cand_keys",
        .source = k.CAND_KEYS_SOURCE,
        .inputs = &.{
            .{ .name = "pos", .ty = fix(.int32) },
            .{ .name = "blocks", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "out", .ty = fix(.int32) }},
        .params = &.{
            .{ .name = "BS", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_index_qprep",
        .source = k.IDX_NAX_PREP_SOURCE,
        .inputs = &.{
            .{ .name = "q", .ty = fix(.float32) },
            .{ .name = "w", .ty = par("W") },
            .{ .name = "wsc", .ty = fix(.float32) },
        },
        .outputs = &.{
            .{ .name = "qa", .ty = fix(.float16) },
            .{ .name = "pq", .ty = fix(.float32) },
            .{ .name = "sqr", .ty = fix(.int32) },
        },
        .params = &.{
            .{ .name = "W", .kind = .io_dtype, .dtypes = &bf_f32 },
        },
    },
    .{
        .name = "dsv41_index_scores_nax",
        .nax = true,
        .header = k.ATTN_NAX_HEADER ++ k.IDX_NAX_HEADER,
        .source = k.IDX_NAX_SOURCE,
        .inputs = &idx_nax_ins,
        .outputs = &.{.{ .name = "scores", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "RB", .kind = .ct_int },
            .{ .name = "SPR", .kind = .ct_int },
            .{ .name = "W", .kind = .io_dtype },
            .{ .name = "IT", .kind = .fc_int },
            .{ .name = "BS", .kind = .ignored },
        },
        .ct_tuples = &idx_nax_tuples,
    },
    .{
        .name = "dsv41_index_scores_nax_gather",
        .nax = true,
        .header = k.ATTN_NAX_HEADER ++ k.IDX_NAX_HEADER,
        .source = k.IDX_NAX_GATHER_SOURCE,
        .inputs = &(idx_nax_ins ++ [_]InputSlot{.{ .name = "blocks", .ty = fix(.int32) }}),
        .outputs = &.{.{ .name = "scores", .ty = fix(.float32) }},
        .params = &.{
            .{ .name = "RB", .kind = .ct_int },
            .{ .name = "SPR", .kind = .ct_int },
            .{ .name = "W", .kind = .io_dtype },
            .{ .name = "IT", .kind = .fc_int },
            .{ .name = "BS", .kind = .fc_int },
        },
        .ct_tuples = &idx_nax_tuples,
    },

    // ── dsv41_kernels.zig: DSpark draft & Engram (3 SIMD kernels) ──
    .{
        .name = "dsv41_markov_row",
        .source = k.MARKOV_ROW_SOURCE,
        .inputs = &.{
            .{ .name = "logits", .ty = fix(.float32) },
            .{ .name = "head", .ty = par("T") },
            .{ .name = "embed", .ty = par("T") },
            .{ .name = "prev", .ty = fix(.int32) },
            .{ .name = "hout", .ty = par("HT") },
            .{ .name = "cw", .ty = fix(.float32) },
            .{ .name = "step", .ty = fix(.int32) },
        },
        .outputs = &.{
            .{ .name = "row", .ty = fix(.float32) },
            .{ .name = "part_val", .ty = fix(.float32) },
            .{ .name = "part_idx", .ty = fix(.int32) },
            .{ .name = "conf", .ty = fix(.float32) },
        },
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "HT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "TG", .kind = .ct_int, .ints = &.{256} },
            .{ .name = "R", .kind = .fc_int },
        },
    },
    .{
        .name = "dsv41_markov_pick",
        .source = k.MARKOV_PICK_SOURCE,
        .inputs = &.{
            .{ .name = "part_val", .ty = fix(.float32) },
            .{ .name = "part_idx", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "pick", .ty = fix(.int32) }},
    },
    .{
        .name = "dsv41_engram",
        .source = k.ENGRAM_SOURCE,
        .inputs = &.{
            .{ .name = "x", .ty = par("OT") },
            .{ .name = "kv", .ty = par("KT") },
            .{ .name = "qk", .ty = fix(.float32) },
            .{ .name = "cst", .ty = fix(.float32) },
        },
        .outputs = &.{.{ .name = "out", .ty = par("OT") }},
        .params = &.{
            .{ .name = "OT", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "KT", .kind = .io_dtype, .dtypes = &bf_f32 },
        },
    },

    // ── dsv41/ubench.zig (2 SIMD link helpers) ──
    .{
        .name = "dsv41_ubench_picks_link",
        .source = ub.LINK_SOURCE,
        .inputs = &.{
            .{ .name = "topk", .ty = fix(.int32) },
            .{ .name = "rows", .ty = par("R") },
            .{ .name = "ids", .ty = fix(.int32) },
        },
        .outputs = &.{.{ .name = "out", .ty = fix(.int32) }},
        .params = &.{
            .{ .name = "R", .kind = .io_dtype, .dtypes = &bf_f32 },
        },
    },
    .{
        .name = "dsv41_ubench_fusion_link",
        .source = ub.FUSION_LINK_SOURCE,
        .inputs = &.{
            .{ .name = "base", .ty = par("T") },
            .{ .name = "dep_a", .ty = par("DA") },
            .{ .name = "dep_b", .ty = par("DB") },
        },
        .outputs = &.{.{ .name = "out", .ty = par("T") }},
        .params = &.{
            .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 },
            .{ .name = "DA", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "DB", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "N", .kind = .fc_int },
        },
    },
};
