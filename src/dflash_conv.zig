//! Two-tap DFlash2 convolution with the reference's BF16 multiply/add boundaries.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
const BODY =
    \\#pragma clang fp contract(off)
    \\#pragma clang fp reassociate(off)
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=uint(L*H))return;
    \\const uint row=i/uint(H),col=i%uint(H),group=col/uint(GS);
    \\bfloat result=bfloat(0.0f);
    \\for(uint tap=0;tap<2u;++tap){
    \\ const bfloat value=row>=tap?hidden[(row-tap)*hidden_strides[1]+col*hidden_strides[2]]:bfloat(0.0f);
    \\ const bfloat a=base[tap*base_strides[0]+col*base_strides[1]];
    \\ const bfloat b=dynamic[row*dynamic_strides[1]+tap*dynamic_strides[2]+group*dynamic_strides[3]];
    \\ const bfloat fixed_product=bfloat(float(a)*float(value));
    \\ result=bfloat(float(result)+float(fixed_product));
    \\ const bfloat dynamic_product=bfloat(float(b)*float(value));
    \\ result=bfloat(float(result)+float(dynamic_product));
    \\}
    \\
;
const SOURCE = BODY ++ "y[i]=result;\n";
// The sublayer's residual add rounds like MLX's BF16 add of the stored convolution output.
const SOURCE_ADD = BODY ++ "y[i]=bfloat(float(residual[row*residual_strides[1]+col*residual_strides[2]])+float(result));\n";
var kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);
const Key = struct { len: c_int, hidden: c_int, group: c_int };
const Config = struct { key: Key, value: mlx.mlx_fast_metal_kernel_config };
var configs: [16]?Config = @splat(null);

pub fn apply(s: mlx.mlx_stream, hidden: Arr, dynamic: Arr, base: Arr, group_size: u32) !?Arr {
    return run(s, hidden, dynamic, base, group_size, null);
}

/// `residual + conv(hidden)` in one dispatch, bit for bit the convolution then MLX's add.
pub fn applyAdd(s: mlx.mlx_stream, hidden: Arr, dynamic: Arr, base: Arr, group_size: u32, residual: Arr) !?Arr {
    if (residual.ctx == null or mlx.mlx_array_dtype(residual) != .bfloat16 or !std.mem.eql(c_int, mlx.getShape(residual), mlx.getShape(hidden))) return null;
    return run(s, hidden, dynamic, base, group_size, residual);
}

fn run(s: mlx.mlx_stream, hidden: Arr, dynamic: Arr, base: Arr, group_size: u32, residual: ?Arr) !?Arr {
    if (!mlx.streamIsGpu(s) or group_size == 0) return null;
    for ([_]Arr{ hidden, dynamic, base }) |a| if (a.ctx == null or mlx.mlx_array_dtype(a) != .bfloat16) return null;
    const shape = mlx.getShape(hidden);
    if (shape.len != 3 or shape[0] != 1 or shape[1] < 1 or shape[1] > 8 or shape[2] < 1 or shape[2] > 4096 or group_size > @as(u32, @intCast(shape[2]))) return null;
    const group: c_int = @intCast(group_size);
    if (@mod(shape[2], group) != 0 or !std.mem.eql(c_int, mlx.getShape(dynamic), &.{ 1, shape[1], 2, @divExact(shape[2], group) }) or
        !std.mem.eql(c_int, mlx.getShape(base), &.{ 2, shape[2] })) return null;
    const variant: usize = if (residual == null) 0 else 1;
    if (kernels[variant] == null) {
        const ins = if (residual == null) mlx.mlx_vector_string_new_data(&.{ "hidden", "dynamic", "base" }, 3) else mlx.mlx_vector_string_new_data(&.{ "hidden", "dynamic", "base", "residual" }, 4);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{"y"}, 1);
        defer _ = mlx.mlx_vector_string_free(outs);
        const k = if (residual == null)
            mlx.mlx_fast_metal_kernel_new("sushi_dflash_two_tap_bf16", ins, outs, SOURCE, "", false, false)
        else
            mlx.mlx_fast_metal_kernel_new("sushi_dflash_two_tap_bf16_add", ins, outs, SOURCE_ADD, "", false, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernels[variant] = k;
    }
    const key = Key{ .len = shape[1], .hidden = shape[2], .group = group };
    var cached: ?mlx.mlx_fast_metal_kernel_config = null;
    for (configs) |item| if (item) |entry| if (std.meta.eql(key, entry.key)) {
        cached = entry.value;
        break;
    };
    const cfg = cached orelse mlx.mlx_fast_metal_kernel_config_new();
    var retained = cached != null;
    defer if (!retained) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    };
    if (cached == null) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.ptr, shape.len, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, key.len * key.hidden, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
        inline for (.{ "L", "H", "GS" }, .{ key.len, key.hidden, key.group }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, value));
        for (&configs) |*item| if (item.* == null) {
            item.* = .{ .key = key, .value = cfg };
            retained = true;
            break;
        };
    }
    const ins = if (residual) |r| mlx.mlx_vector_array_new_data(&.{ hidden, dynamic, base, r }, 4) else mlx.mlx_vector_array_new_data(&.{ hidden, dynamic, base }, 3);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kernels[variant].?, ins, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    return out;
}

test "DFlash fused convolution preserves BF16 boundaries and strided tap sets" {
    try mlx.requireGpu();
    const Ops = @import("glm5/glm5_model.zig").Ops;
    const fixture = @import("dflash.zig").TinyFix;
    for ([_]c_int{ 128, 4096 }) |width| for ([_]c_int{ 1, 2, 3, 8 }) |len| {
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const hidden = try ops.own(try fixture.bf16ArrShaped(&.{ 1, len, width }, 791, ops.s));
        const bases = try ops.own(try fixture.bf16ArrShaped(&.{ 2, 2, width }, 793, ops.s));
        const dynamics = try ops.own(try fixture.bf16ArrShaped(&.{ 1, len, 2, 2, @divExact(width, 16) }, 797, ops.s));
        for (0..2) |set| {
            const at: c_int = @intCast(set);
            const base = try ops.reshape(try ops.slice(bases, 0, at, at + 1), &.{ 2, width });
            const cut = try ops.slot();
            try mlx.check(mlx.mlx_slice(cut, dynamics, &.{ 0, 0, at, 0, 0 }, 5, &.{ 1, len, at + 1, 2, @divExact(width, 16) }, 5, &.{ 1, 1, 1, 1, 1 }, 5, ops.s));
            const dynamic = try ops.reshape(cut.*, &.{ 1, len, 2, @divExact(width, 16) });
            for ([_]bool{ false, true }) |zero| {
                const x = if (zero) try ops.zeros(&.{ 1, len, width }, .bfloat16) else hidden;
                const want = try ops.own(try @import("dflash.zig").groupedDynConvReference(x, dynamic, base, 16, ops.s));
                const got = try ops.own((try apply(ops.s, x, dynamic, base, 16)) orelse return error.TestExpectedFusedConv);
                try mlx.check(mlx.mlx_array_eval(want));
                try mlx.check(mlx.mlx_array_eval(got));
                const count: usize = @intCast(len * width);
                try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(want).?[0..count], mlx.mlx_array_data_bfloat16(got).?[0..count]);
                // A residual read as a row view of a wider block.
                const wide = try ops.own(try fixture.bf16ArrShaped(&.{ 1, 8, width }, 799, ops.s));
                const residual = try ops.slice(wide, 1, 8 - len, 8);
                const want_sum = try ops.binary(.add, residual, want);
                const got_sum = try ops.own((try applyAdd(ops.s, x, dynamic, base, 16, residual)) orelse return error.TestExpectedFusedConv);
                try mlx.check(mlx.mlx_array_eval(want_sum));
                try mlx.check(mlx.mlx_array_eval(got_sum));
                try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(want_sum).?[0..count], mlx.mlx_array_data_bfloat16(got_sum).?[0..count]);
            }
        }
        try std.testing.expect((try apply(ops.s, try ops.cast(hidden, .float32), try ops.zeros(&.{ 1, len, 2, @divExact(width, 16) }, .bfloat16), try ops.zeros(&.{ 2, width }, .bfloat16), 16)) == null);
    };
}
