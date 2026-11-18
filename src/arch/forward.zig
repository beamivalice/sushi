//! Forward dispatch over the loaded arch. Inference thread only.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model_handle = @import("../model_handle.zig");
const registry = @import("registry.zig");
const transient_peak = @import("../transient_peak.zig");
const arch_cooler = @import("cooler.zig");
const Arch = registry.Arch;

/// One prefill chunk, decode step or verify block, as `ctx` says.
/// The late-compile line's words for the forward in flight (`engine.lateWhere`).
var where_buf: [48]u8 = undefined;

pub fn forward(a: Arch, ctx: *model_handle.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
    const config = @import("handle.zig").config(a);
    const rows = arch_cooler.enterForward(config, ctx, ids, @import("handle.zig").trunk(a), transient_peak.enabled());
    const where = if (rows) |r| std.fmt.bufPrint(&where_buf, "a forward of {d} rows to {d}", .{ r.rows, r.seq }) catch "a forward" else "a forward";
    const was = mlx.engine.lateWhere(where);
    defer _ = mlx.engine.lateWhere(was);
    // Paired: `rows` is read whenever the peak diagnostic is on.
    transient_peak.begin();
    defer if (rows) |r| transient_peak.end(.ofRows(r.rows), r.rows, r.cap(config));
    return switch (a) {
        inline else => |m, t| registry.Mod(t).forward(m, ctx, ids),
    };
}

/// Draft logits: the hidden state through the trunk's bare head.
pub fn lmHeadForDraft(a: Arch, x: mlx.mlx_array) !mlx.mlx_array {
    return switch (a) {
        inline else => |m, t| registry.Mod(t).lmHeadForDraft(m, x),
    };
}
