const std = @import("std");
const mlx = @import("mlx.zig");

/// A repeat penalty of 0 or below is off: it would divide by zero or flip signs.
pub fn repeatOn(repeat_penalty: f32) bool {
    return repeat_penalty > 0.0 and repeat_penalty != 1.0;
}

/// The three penalties of a request. `repeat` scales a seen id's logit (llama.cpp's); `presence` is
/// subtracted once per seen id and `frequency` once per occurrence (OpenAI's definitions).
pub const Penalties = struct {
    repeat: f32 = 1.0,
    presence: f32 = 0.0,
    frequency: f32 = 0.0,

    pub fn active(self: Penalties) bool {
        return repeatOn(self.repeat) or self.presence != 0.0 or self.frequency != 0.0;
    }
};

/// `[vocab]` int32 occurrences of each id in `ids`.
pub fn buildCounts(allocator: std.mem.Allocator, ids: []const u32, vocab: usize) !mlx.mlx_array {
    const buf = try allocator.alloc(i32, vocab);
    defer allocator.free(buf);
    @memset(buf, 0);
    for (ids) |id| {
        if (id < vocab) buf[id] += 1;
    }
    const shape = [_]c_int{@intCast(vocab)};
    return mlx.mlx_array_new_data(buf.ptr, &shape, 1, .int32);
}

fn asFloat32(out: *mlx.mlx_array, a: mlx.mlx_array, s: mlx.mlx_stream) !void {
    try mlx.check(mlx.mlx_astype(out, a, .float32, s));
}

/// The penalties on every logit whose id has a nonzero entry in `counts` (int32, broadcast against
/// `logits`). The serial sampler and the MTP verify rows both come here, so a row equals its serial tick.
pub fn apply(res: *mlx.mlx_array, logits: mlx.mlx_array, counts: mlx.mlx_array, pen: Penalties, s: mlx.mlx_stream) !void {
    var current = logits;
    const zero_i = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero_i);
    var seen = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(seen);
    try mlx.check(mlx.mlx_greater(&seen, counts, zero_i, s));

    // Repeat penalty: seen tokens divide by the penalty when positive and multiply by it when negative.
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    if (repeatOn(pen.repeat)) {
        const rp = mlx.mlx_array_new_float(pen.repeat);
        defer _ = mlx.mlx_array_free(rp);
        const inv_rp = mlx.mlx_array_new_float(1.0 / pen.repeat);
        defer _ = mlx.mlx_array_free(inv_rp);
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);

        var positive_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(positive_mask);
        try mlx.check(mlx.mlx_greater(&positive_mask, current, zero, s));

        var pen_pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_pos);
        try mlx.check(mlx.mlx_multiply(&pen_pos, current, inv_rp, s));

        var pen_neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pen_neg);
        try mlx.check(mlx.mlx_multiply(&pen_neg, current, rp, s));

        var sign_selected = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sign_selected);
        try mlx.check(mlx.mlx_where(&sign_selected, positive_mask, pen_pos, pen_neg, s));

        try mlx.check(mlx.mlx_where(&penalized, seen, sign_selected, current, s));
        current = penalized;
        penalized_owned = true;
    }

    if (pen.presence == 0.0 and pen.frequency == 0.0) {
        try mlx.check(mlx.mlx_copy(res, current, s));
        return;
    }
    // Additive part in float32: presence * seen + frequency * count.
    var seen_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(seen_f);
    try asFloat32(&seen_f, seen, s);
    var counts_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(counts_f);
    try asFloat32(&counts_f, counts, s);
    const pp = mlx.mlx_array_new_float(pen.presence);
    defer _ = mlx.mlx_array_free(pp);
    const fp = mlx.mlx_array_new_float(pen.frequency);
    defer _ = mlx.mlx_array_free(fp);
    var by_presence = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(by_presence);
    try mlx.check(mlx.mlx_multiply(&by_presence, seen_f, pp, s));
    var by_count = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(by_count);
    try mlx.check(mlx.mlx_multiply(&by_count, counts_f, fp, s));
    var subtract = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(subtract);
    try mlx.check(mlx.mlx_add(&subtract, by_presence, by_count, s));
    try mlx.check(mlx.mlx_subtract(res, current, subtract, s));
}

/// Row j of `logits` `[1, L, V]` under the penalties as its serial tick applies them: after the
/// committed ids (`committed`, `[V]` int32 counts) and `ids[0..j]` (`[1, L]` int32, inclusive, lazy is fine).
pub fn shiftRows(out: *mlx.mlx_array, logits: mlx.mlx_array, ids: mlx.mlx_array, committed: mlx.mlx_array, pen: Penalties, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    var vocab_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vocab_ids);
    try mlx.check(mlx.mlx_arange(&vocab_ids, 0, @floatFromInt(shape[shape.len - 1]), 1, .int32, s));
    var column = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(column);
    try mlx.check(mlx.mlx_expand_dims(&column, ids, 2, s));
    var hit = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hit);
    try mlx.check(mlx.mlx_equal(&hit, column, vocab_ids, s));
    var hit_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hit_i);
    try mlx.check(mlx.mlx_astype(&hit_i, hit, .int32, s));
    var through = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(through);
    try mlx.check(mlx.mlx_cumsum(&through, hit_i, 1, false, true, s));
    var counts = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(counts);
    try mlx.check(mlx.mlx_add(&counts, through, committed, s));
    try apply(out, logits, counts, pen, s);
}
