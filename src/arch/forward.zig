//! Forward dispatch over the loaded arch. Inference thread only.
const mlx = @import("../mlx.zig");
const model_handle = @import("../model_handle.zig");
const registry = @import("registry.zig");
const Arch = registry.Arch;

/// One prefill chunk, decode step or verify block, as `ctx` says.
pub fn forward(a: Arch, ctx: *model_handle.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
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
