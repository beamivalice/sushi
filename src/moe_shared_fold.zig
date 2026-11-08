//! The shared expert, its gate logit and the sigmoid-gated add inside the routed EXL3 launches (`exl3/fold_gated.zig`
//! for a one-bank layer, `exl3/rate_groups.zig` for a rate-grouped one), for any trunk whose MoE layer gates it.
const std = @import("std");
const mlx = @import("mlx.zig");
const api = @import("sushi_exl3");
const trunk_mod = @import("trunk.zig");
const Arr = mlx.mlx_array;

var logged = false;

/// The stored geometry of the layer for `x`'s row count, or null where none was bound: the gate's logit must be handed
/// in or computed by the fold as the load assumed.
fn storedGeo(mw: *const trunk_mod.MoeMlpWeights, x: Arr, own_logit: bool) ?*const api.riders_gated.Geometry {
    const gate_s = mw.shared_expert_gate_s orelse return null;
    if (own_logit != (gate_s.ctx != null)) return null;
    const shape = mlx.getShape(x);
    if (shape.len != 3 or shape[1] < 1 or shape[1] > mw.shared_geo.len) return null;
    return if (mw.shared_geo[@intCast(shape[1] - 1)]) |*geo| geo else null;
}

/// The layer output with the shared expert folded into the routed launches, for one to eight rows of `x` [1, rows, hidden];
/// null where the separate chain runs. `router_logit` is the shared-expert gate logit when the router kernel already
/// computed it (null ctx otherwise: the fold computes it). `trunk_dec` decodes a trellis shared expert.
pub fn folded(s: mlx.mlx_stream, x: Arr, mw: *const trunk_mod.MoeMlpWeights, inds: Arr, scores: Arr, dec: api.format.Decode, trunk_dec: ?api.format.Decode, router_logit: Arr) !?Arr {
    if (mw.shared_expert_gate_w == null) return null;
    const own_logit = router_logit.ctx == null;
    const geo = storedGeo(mw, x, own_logit);
    const shared = mw.foldShared(try trunk_mod.swigluSigTable(s, .bfloat16, std.heap.c_allocator), trunk_dec, own_logit, geo != null) orelse return null;
    const gate: ?Arr = if (own_logit) null else router_logit;
    const result = if (mw.exl3_ragged) |*rb|
        (if (geo) |g| try api.rate_groups.moeGatedSharedFoldedGeo(s, x, rb, inds, scores, dec, shared, gate, g) else try api.rate_groups.moeGatedSharedFolded(s, x, rb, inds, scores, dec, shared, gate))
    else blk: {
        const bank = mw.routedBank();
        const g = geo orelse break :blk try api.fold_gated.moeGatedSharedFolded(s, x, bank, inds, scores, dec, shared, gate);
        if (mw.routed_shape) |*shape| break :blk try api.fold_gated.moeGatedSharedFoldedBound(s, x, bank, shape, inds, scores, dec, shared, gate, g);
        break :blk try api.fold_gated.moeGatedSharedFoldedGeo(s, x, bank, inds, scores, dec, shared, gate, g);
    };
    if (result != null and !logged) {
        logged = true;
        @import("log.zig").info("[qwen] shared expert rides the routed launches engaged\n", .{});
    }
    return result;
}
