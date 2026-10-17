const std = @import("std");
const mlx = @import("mlx.zig");

/// A repeat penalty of 0 or below is off: it would divide by zero or flip signs.
pub fn repeatOn(repeat_penalty: f32) bool {
    return repeat_penalty > 0.0 and repeat_penalty != 1.0;
}

/// The repeat and presence penalties on every logit where `seen` (bool, broadcast against `logits`)
/// is true. The serial sampler and the MTP verify rows both come here, so a row equals its serial tick.
pub fn apply(res: *mlx.mlx_array, logits: mlx.mlx_array, seen: mlx.mlx_array, repeat_penalty: f32, presence_penalty: f32, s: mlx.mlx_stream) !void {
    var current = logits;

    // Repeat penalty: seen tokens divide by the penalty when positive and multiply by it when negative.
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    if (repeatOn(repeat_penalty)) {
        const rp = mlx.mlx_array_new_float(repeat_penalty);
        defer _ = mlx.mlx_array_free(rp);
        const inv_rp = mlx.mlx_array_new_float(1.0 / repeat_penalty);
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

    // Presence penalty: subtract from seen tokens
    if (presence_penalty != 0.0) {
        const pp = mlx.mlx_array_new_float(presence_penalty);
        defer _ = mlx.mlx_array_free(pp);

        var seen_float = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(seen_float);
        try mlx.check(mlx.mlx_astype(&seen_float, seen, .float16, s));

        var subtract = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(subtract);
        try mlx.check(mlx.mlx_multiply(&subtract, seen_float, pp, s));

        try mlx.check(mlx.mlx_subtract(res, current, subtract, s));
    } else {
        try mlx.check(mlx.mlx_copy(res, current, s));
    }
}

/// Row j of `logits` `[1, L, V]` under the penalties as its serial tick applies them: after the
/// committed ids (`committed`, `[V]` bool) and `ids[0..j]` (`[1, L]` int32, inclusive, lazy is fine).
pub fn shiftRows(out: *mlx.mlx_array, logits: mlx.mlx_array, ids: mlx.mlx_array, committed: mlx.mlx_array, repeat_penalty: f32, presence_penalty: f32, s: mlx.mlx_stream) !void {
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
    const zero = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero);
    var in_round = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(in_round);
    try mlx.check(mlx.mlx_greater(&in_round, through, zero, s));
    var seen = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(seen);
    try mlx.check(mlx.mlx_logical_or(&seen, in_round, committed, s));
    try apply(out, logits, seen, repeat_penalty, presence_penalty, s);
}
