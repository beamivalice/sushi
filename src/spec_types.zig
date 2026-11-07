//! Types the speculative-decoding files share, so the round, the policy and the drafters need not import
//! `generate.zig` or `scheduler.zig`.
const std = @import("std");
const mlx = @import("mlx.zig");
const mtp_mod = @import("mtp.zig");

/// A head's last healthy acceptance surface; it dies with the head, so a seed never crosses models.
pub const EvSeed = struct { accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32 };

/// Launch settings the rounds read. A generator copies them when it starts, so a round never reads a process global.
pub const RoundOpts = struct {
    /// `--no-mtp-lookup`: no prompt-lookup drafts in MTP rounds or GLM DFlash2 chains.
    lookup_disabled: bool = false,
    /// `--max-mtp-ctx N`: MTP stays off past N context tokens (0 = unlimited).
    max_mtp_ctx: u32 = 0,
};

/// How an arch drafts. `mtp` and `dspark` run from a `Drafter` head; `dflash2` runs from the generator's block
/// drafter state.
pub const Kind = enum { mtp, dspark, dflash2 };

/// A scalar in the array's own dtype, so a masked bf16 block is not widened to
/// f32 by its own `-inf`.
fn scalarLike(v: f32, like: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, raw, mlx.mlx_array_dtype(like), s));
    return out;
}

/// A greedy spec-verify argmax whose entries are COMMITTED as next tokens.
/// `verifyArgmax` is its only constructor and every commit site takes this type,
/// so a raw `mlx_argmax_axis` cannot reach a commit without the reserved-id mask.
pub const CommittedArgmax = struct {
    arr: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *CommittedArgmax) void {
        if (self.arr.ctx != null) _ = mlx.mlx_array_free(self.arr);
        self.arr = .{ .ctx = null };
    }

    pub fn present(self: CommittedArgmax) bool {
        return self.arr.ctx != null;
    }

    /// The lazy handle, for an eval vector or a forced eval - never for a commit.
    pub fn lazy(self: CommittedArgmax) mlx.mlx_array {
        return self.arr;
    }

    /// The realized ids; the caller must have evaluated `lazy()` first.
    pub fn ids(self: CommittedArgmax, len: usize) ![]const i32 {
        const data = mlx.mlx_array_data_int32(self.arr) orelse return error.MlxArrayDataNull;
        return data[0..len];
    }
};

/// The greedy spec-verify argmax, over the reserved-id mask when the model has
/// one. The sampled verify filters through `applySuppressMask` and the rerank
/// draft select takes the mask, so without this the greedy verify was the one
/// path that could commit a flagged special or a padding row past the defined
/// vocabulary — ids the serial sampler can never draw.
pub fn verifyArgmax(logits: mlx.mlx_array, mask: ?mlx.mlx_array, s: mlx.mlx_stream) !CommittedArgmax {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (mask) |m| {
        const neg_inf = try scalarLike(-std.math.inf(f32), logits, s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits, s));
        try mlx.check(mlx.mlx_argmax_axis(&out, masked, 2, false, s));
    } else {
        try mlx.check(mlx.mlx_argmax_axis(&out, logits, 2, false, s));
    }
    return .{ .arr = out };
}

/// The `[1+m, V]` rows a round reads from its verify block: a batched group
/// right-pads every row to its widest draft, so the block may be longer.
pub fn verifyRows2d(probs_all: mlx.mlx_array, m: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(probs_all);
    const rows: c_int = @intCast(1 + m);
    if (shape.len != 3 or shape[0] != 1 or shape[1] < rows) return error.MtpVerifyBlockShape;
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, probs_all, &.{ 0, 0, 0 }, 3, &.{ 1, rows, shape[2] }, 3, &.{ 1, 1, 1 }, 3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, head, &.{ rows, shape[2] }, 2, s));
    return out;
}

pub fn isEosId(id: u32, eos: []const u32) bool {
    for (eos) |e| if (id == e) return true;
    return false;
}

/// Truthy if the env var is exactly "1". Anything else (unset, "0", "true",
/// "yes") is false — keep matching surface tight to avoid surprises.
pub fn readEnvBool(name: [:0]const u8) bool {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return false;
    const slice = std.mem.sliceTo(raw.?, 0);
    return std.mem.eql(u8, slice, "1");
}

/// How far a head's position has drifted from where the next round would read it.
pub fn headPositionDrift(pos_base: i64, seq_offset: usize, next_pos_offset: i64) i64 {
    if (seq_offset == 0) return 0;
    return next_pos_offset - (pos_base + @as(i64, @intCast(seq_offset)));
}
