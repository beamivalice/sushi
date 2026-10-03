//! Rotated-codebook (TBQ) constant tables, ported from the llama.cpp-turboq-mtp
//! fork (ref/ggml-turboq-tables.h, seed-42 FWHT sign arrays). Block size is
//! 128 for both schemes; the 3-bit codebook lives in the UNNORMALIZED FWHT
//! domain (its dequant applies the 1/sqrt(128) factor), the 4-bit one in the
//! normalized domain. Do not rescale or swap tables between schemes.

const std = @import("std");

/// 1/sqrt(128), matching `inv_sqrt_128` in the C reference.
pub const inv_sqrt_128: f32 = 0.08838834764831845;
/// Rotation block width (both TBQ schemes).
pub const block: usize = 128;


pub const centroids4 = [_]f32{
    -0.241556, -0.182907, -0.143047, -0.111065, -0.083317, -0.058069, -0.034311, -0.011353,
    0.011353, 0.034311, 0.058069, 0.083317, 0.111065, 0.143047, 0.182907, 0.241556,
};

pub const midpoints4 = [_]f32{
    -0.212232, -0.162977, -0.127056, -0.097191, -0.070693, -0.046190, -0.022832, 0.000000,
    0.022832, 0.046190, 0.070693, 0.097191, 0.127056, 0.162977, 0.212232,
};

pub const centroids3 = [_]f32{
    -1.646828, -0.895384, -0.491349, -0.157976, 0.157976, 0.491349, 0.895384, 1.646828,
};

pub const midpoints3 = [_]f32{
    -1.271106, -0.693367, -0.324663, 0.0, 0.324663, 0.693367, 1.271106,
};

pub const signs1_4 = [_]f32{
    -1, 1, 1, -1, -1, 1,
    -1, 1, -1, -1, 1, 1,
    1, 1, 1, 1, 1, -1,
    1, -1, 1, -1, -1, 1,
    1, 1, -1, 1, 1, -1,
    -1, -1, -1, 1, 1, -1,
    1, 1, -1, 1, -1, 1,
    1, -1, -1, 1, -1, 1,
    1, 1, 1, -1, -1, -1,
    -1, -1, 1, -1, 1, 1,
    1, 1, -1, 1, -1, -1,
    1, -1, -1, -1, 1, -1,
    -1, -1, 1, -1, -1, -1,
    1, 1, 1, -1, -1, 1,
    1, 1, -1, -1, 1, 1,
    -1, 1, 1, -1, 1, -1,
    -1, 1, 1, -1, 1, -1,
    1, -1, 1, 1, 1, 1,
    -1, 1, -1, 1, 1, -1,
    1, 1, -1, -1, -1, -1,
    -1, 1, 1, -1, 1, 1,
    -1, 1,
};

pub const signs2_4 = [_]f32{
    1, 1, 1, 1, -1, 1,
    1, -1, 1, -1, -1, -1,
    1, -1, -1, -1, 1, 1,
    -1, -1, 1, -1, 1, -1,
    1, -1, -1, 1, -1, 1,
    1, 1, 1, 1, -1, -1,
    -1, 1, -1, -1, -1, -1,
    -1, -1, 1, 1, 1, -1,
    1, -1, 1, 1, 1, -1,
    -1, 1, -1, -1, -1, -1,
    -1, -1, 1, 1, 1, -1,
    1, -1, -1, -1, -1, 1,
    -1, 1, -1, 1, -1, -1,
    1, 1, -1, 1, -1, 1,
    1, -1, 1, -1, -1, -1,
    -1, 1, -1, -1, 1, -1,
    1, -1, 1, 1, 1, -1,
    -1, 1, -1, 1, -1, 1,
    1, -1, -1, 1, -1, 1,
    -1, 1, 1, -1, 1, -1,
    1, -1, -1, -1, -1, -1,
    1, -1,
};

pub const signs1_3 = [_]f32{
    -1, 1, 1, 1, -1, 1,
    -1, 1, -1, -1, 1, 1,
    1, 1, -1, -1, 1, -1,
    -1, 1, 1, -1, -1, 1,
    -1, -1, 1, 1, 1, -1,
    1, -1, 1, -1, 1, 1,
    -1, -1, 1, 1, -1, 1,
    -1, -1, 1, 1, -1, 1,
    1, 1, 1, -1, 1, -1,
    -1, 1, -1, 1, 1, 1,
    -1, -1, 1, -1, -1, 1,
    -1, -1, 1, -1, 1, 1,
    -1, 1, -1, 1, 1, -1,
    -1, 1, 1, -1, -1, -1,
    1, 1, -1, -1, -1, -1,
    -1, -1, 1, 1, -1, 1,
    -1, -1, 1, 1, -1, 1,
    1, 1, -1, -1, -1, 1,
    1, -1, 1, -1, -1, 1,
    -1, 1, -1, -1, -1, -1,
    1, 1, 1, -1, 1, 1,
    1, -1,
};

pub const signs2_3 = [_]f32{
    1, 1, 1, -1, -1, -1,
    -1, -1, -1, 1, -1, 1,
    1, 1, -1, -1, 1, -1,
    -1, 1, 1, -1, 1, -1,
    -1, 1, 1, -1, -1, 1,
    -1, -1, 1, 1, 1, -1,
    -1, -1, 1, -1, -1, 1,
    -1, -1, -1, -1, -1, -1,
    -1, 1, 1, 1, 1, 1,
    1, -1, 1, -1, -1, -1,
    1, -1, 1, 1, -1, 1,
    1, 1, -1, -1, -1, -1,
    1, -1, 1, -1, 1, 1,
    1, -1, 1, -1, -1, -1,
    -1, 1, 1, -1, 1, -1,
    1, 1, -1, -1, 1, -1,
    -1, -1, 1, 1, 1, 1,
    1, 1, 1, 1, -1, -1,
    -1, -1, -1, -1, 1, -1,
    -1, 1, 1, 1, -1, -1,
    1, -1, 1, 1, -1, -1,
    -1, -1,
};


/// Sylvester Hadamard 128, row-major, entries ±1: `fwht(x) = x @ h128` for a
/// row-vector x (the matrix is symmetric, so orientation is irrelevant).
/// `h128[i * 128 + j]`.
pub const h128: [128 * 128]f32 = blk: {
    @setEvalBranchQuota(20000);
    // Sylvester construction: H[i,j] = (-1)^{popcount(i & j)}; the 128×128
    // matrix equals the 256×256 one restricted to indices < 128.
    var h: [128 * 128]f32 = undefined;
    for (0..128) |i| {
        for (0..128) |j| {
            h[i * 128 + j] = if (@popCount(i & j) & 1 == 0) 1 else -1;
        }
    }
    break :blk h;
};

test "h128 is symmetric Hadamard" {
    std.debug.assert(h128[0] == 1 and h128[128 * 128 - 1] == -1);
    for (0..128) |i| {
        for (0..128) |j| {
            std.debug.assert(h128[i * 128 + j] == h128[j * 128 + i]);
            std.debug.assert(h128[i * 128 + j] == 1 or h128[i * 128 + j] == -1);
        }
        var dot: f32 = 0;
        for (0..128) |k| dot += h128[i * 128 + k] * h128[i * 128 + k];
        std.debug.assert(dot == 128);
    }
}
test "midpoints separate centroids (both schemes, ascending)" {
    for ([_]struct { c: []const f32, m: []const f32 }{
        .{ .c = &centroids4, .m = &midpoints4 },
        .{ .c = &centroids3, .m = &midpoints3 },
    }) |t_| {
        for (t_.m, 0..) |mp, j| {
            std.debug.assert(mp > t_.c[j] and mp < t_.c[j + 1]);
        }
    }
}
