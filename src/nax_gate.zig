const trunk = @import("trunk.zig");
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const device = @import("sushi_exl3").device;
const testing = std.testing;
const verifyQmmEnabled = trunk.verifyQmmEnabled;

/// Human-readable NAX status for `--version` / the app's Settings ("nax"
/// line, first token on/off, remainder the reason). Hardware + OS is the
/// whole story for OUR binaries: the bundled MLX always ships the NAX
/// kernels (build-mlx.sh + tests/test_mlx_staged_nax.sh assert them), so
/// this mirrors MLX's stock-op is_nax_available() gate (GPU gen >= 17,
/// macOS >= 26.2 — that symbol is not exported, hence the mirror).
/// Deliberately ignores the verifyQmm-lane QA env switches: those scope
/// our custom kernel lane, not MLX's stock dispatch.
pub fn naxStatusOf(p: device.Profile) []const u8 {
    if (!p.chip.contains(.matrix_g17)) return "off (requires M5-class GPU)";
    if (p.osTooOldFor(.matrix_g17)) return "off (requires macOS 26.2+)";
    return "on (M5 neural accelerators)";
}

pub fn naxStatus() []const u8 {
    return naxStatusOf(device.current());
}

/// "26.4"-style product version, whole, for a store's identity.
pub fn macosProductVersion(buf: []u8) ?[]const u8 {
    const text = device.osVersion(buf);
    return if (text.len == 0) null else text;
}

/// The matrix units are this GPU's and usable: generation 17 or later under macOS 26.2, and not withheld by
/// `SUSHI_FORCE_GPU_FAMILY_FALLBACK=1`, which has an M5 rehearse the plain-SIMD path of an M1 to M4.
pub fn verifyQmmNaxAvailable() bool {
    // The MLX-free test tier reaches this through the bill and answers for no device, whatever the box.
    if (@import("build_options").host_tier) return false;
    return device.current().has(.matrix_g17);
}

var nax_sdpa_logged = false;

/// mlx >= 0.32.2 serves hd-256 attention on NAX with a fused full-attention
/// kernel (`sdpa_full_self_attention_nax`), auto only at >= 1024 causal rows
/// and reachable below that through `force_fused`. Preferred wherever NAX is
/// available; elsewhere sushi_attn_p256 and the split/vector path serve.
pub fn naxSdpaPreferred() bool {
    const on = verifyQmmNaxAvailable();
    if (on and !nax_sdpa_logged) log.info("[nax-sdpa] engaged: stock fused sdpa (force_fused) serves hd-256 causal prefill and verify blocks > 8 rows; sushi_attn_p256 causal arm declined\n", .{});
    nax_sdpa_logged = on or nax_sdpa_logged;
    return on;
}

/// Family-wide NAX readiness: the hardware probe passes. This intentionally says nothing about a specific
/// M/K/N dispatch.
pub fn verifyQmmNaxEnabled() bool {
    return verifyQmmEnabled() and verifyQmmNaxAvailable();
}

/// M width where the NAX tile takes over from the plain-SIMD lanes.
pub fn naxMinMForBits(bits: u32) c_int {
    // The byte-per-value q8 NAX tile wins already at M=7 (+6.09% over the
    // plain lane, five thermally-settled counterbalanced pairs). Packed q4/q6
    // still lose there, so their measured takeover remains M=8.
    return if (bits == 8) 7 else 8;
}

/// QSA's NAX arm takes cooperative input tensors, which Metal has from macOS 26.3.
pub fn qsaNaxOsSupported(version: []const u8) bool {
    return device.Os.parse(version).atLeast(26, 3);
}

pub fn qsaNaxOsOk() bool {
    return device.current().os.atLeast(26, 3);
}

pub fn qsaProbeLcgBf16(s: mlx.mlx_stream, shape: []const c_int, seed: u32) ?mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = std.heap.page_allocator.alloc(f32, n) catch return null;
    defer std.heap.page_allocator.free(buf);
    var state = seed;
    for (buf) |*x| {
        state = state *% 1664525 +% 1013904223;
        x.* = @as(f32, @floatFromInt(state >> 8)) * (1.0 / 16777216.0) - 0.5;
    }
    const host = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(host);
    var out = mlx.mlx_array_new();
    if (mlx.mlx_astype(&out, host, .bfloat16, s) != 0) {
        _ = mlx.mlx_array_free(out);
        return null;
    }
    if (mlx.mlx_array_eval(out) != 0) {
        _ = mlx.mlx_array_free(out);
        return null;
    }
    return out;
}

pub fn qsaProbeCorruptFirst(s: mlx.mlx_stream, arr: mlx.mlx_array) ?mlx.mlx_array {
    var contig = mlx.mlx_array_new();
    if (mlx.mlx_contiguous(&contig, arr, false, s) != 0) {
        _ = mlx.mlx_array_free(contig);
        return null;
    }
    defer _ = mlx.mlx_array_free(contig);
    var f32a = mlx.mlx_array_new();
    if (mlx.mlx_astype(&f32a, contig, .float32, s) != 0) {
        _ = mlx.mlx_array_free(f32a);
        return null;
    }
    defer _ = mlx.mlx_array_free(f32a);
    if (mlx.mlx_array_eval(f32a) != 0) return null;
    const n = mlx.mlx_array_size(f32a);
    if (n == 0) return null;
    const src = mlx.mlx_array_data_float32(f32a) orelse return null;
    const buf = std.heap.page_allocator.alloc(f32, n) catch return null;
    defer std.heap.page_allocator.free(buf);
    @memcpy(buf, src[0..n]);
    buf[0] += 1.0;
    const shape = mlx.getShape(arr);
    const host = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(host);
    var out = mlx.mlx_array_new();
    if (mlx.mlx_astype(&out, host, mlx.mlx_array_dtype(arr), s) != 0) {
        _ = mlx.mlx_array_free(out);
        return null;
    }
    return out;
}

pub fn qsaProbeAllFinite(s: mlx.mlx_stream, a: mlx.mlx_array) bool {
    var fin = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fin);
    if (mlx.mlx_isfinite(&fin, a, s) != 0) return false;
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    if (mlx.mlx_all(&all, fin, false, s) != 0) return false;
    if (mlx.mlx_array_eval(all) != 0) return false;
    var ok: bool = false;
    if (mlx.mlx_array_item_bool(&ok, all) != 0) return false;
    return ok;
}

pub fn qsaProbeMaxAbsDiff(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) ?f32 {
    var sub = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sub);
    if (mlx.mlx_subtract(&sub, a, b, s) != 0) return null;
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    if (mlx.mlx_abs(&ab, sub, s) != 0) return null;
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    if (mlx.mlx_max(&mx, ab, false, s) != 0) return null;
    if (mlx.mlx_array_eval(mx) != 0) return null;
    var v: f32 = 0;
    if (mlx.mlx_array_item_float32(&v, mx) != 0) return null;
    return v;
}

test "naxStatusOf: on for G17 + 26.2, off names the missing leg (--version / Settings display)" {
    const Os = device.Os;
    try testing.expectEqualStrings("on (M5 neural accelerators)", naxStatusOf(.from("applegpu_g17s", Os.parse("26.2"), false)));
    try testing.expectEqualStrings("on (M5 neural accelerators)", naxStatusOf(.from("applegpu_g17d", Os.parse("27.0"), false)));
    // The QA switch scopes our own kernel arms, not what the chip and MLX's stock ops have.
    try testing.expectEqualStrings("on (M5 neural accelerators)", naxStatusOf(.from("applegpu_g17s", Os.parse("27.0"), true)));
    try testing.expectEqualStrings("off (requires M5-class GPU)", naxStatusOf(.from("applegpu_g16s", Os.parse("26.4"), false)));
    try testing.expectEqualStrings("off (requires M5-class GPU)", naxStatusOf(.from("", Os.parse("26.4"), false)));
    try testing.expectEqualStrings("off (requires macOS 26.2+)", naxStatusOf(.from("applegpu_g17s", Os.parse("26.1"), false)));
}

test "verifyQmmNaxAvailable and qsaNaxOsOk answer for the profile a test puts in place" {
    const host_tier = @import("build_options").host_tier;
    const Os = device.Os;
    const cases = [_]struct { arch: []const u8, os: []const u8, held: bool, nax: bool, qsa: bool }{
        .{ .arch = "applegpu_g17s", .os = "26.3", .held = false, .nax = true, .qsa = true },
        .{ .arch = "applegpu_g17s", .os = "26.2", .held = false, .nax = true, .qsa = false },
        .{ .arch = "applegpu_g17s", .os = "27.0", .held = true, .nax = false, .qsa = true },
        .{ .arch = "applegpu_g16s", .os = "27.0", .held = false, .nax = false, .qsa = true },
        .{ .arch = "applegpu_g13c", .os = "26.5", .held = false, .nax = false, .qsa = true },
        .{ .arch = "", .os = "", .held = false, .nax = false, .qsa = false },
    };
    for (cases) |c| {
        const scope = device.withProfile(.from(c.arch, Os.parse(c.os), c.held));
        defer scope.end();
        try testing.expectEqual(c.nax and !host_tier, verifyQmmNaxAvailable());
        try testing.expectEqual(c.qsa, qsaNaxOsOk());
    }
}

test "naxMinMForBits: q8 takes over at the measured M7, packed widths at M8" {
    try testing.expectEqual(@as(c_int, 8), naxMinMForBits(4));
    try testing.expectEqual(@as(c_int, 8), naxMinMForBits(6));
    try testing.expectEqual(@as(c_int, 7), naxMinMForBits(8));
}
