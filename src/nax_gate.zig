const trf = @import("transformer.zig");
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const testing = std.testing;
const verifyQmmEnabled = trf.verifyQmmEnabled;

/// Case-insensitive prefix match on the M5-class GPU family identifier.
/// Prefix (not equality) is MTPLX's shipping behavior — device variants
/// report suffixed forms ("applegpu_g17s", "applegpu_g17d").
pub fn naxArchGeneration(arch: []const u8) struct { gen: u32, phone: bool } {
    if (arch.len < 3) return .{ .gen = 0, .phone = false };
    var buf: [128]u8 = undefined;
    const n = @min(arch.len, buf.len);
    for (arch[0..n], 0..) |c, i| buf[i] = std.ascii.toLower(c);
    const a = buf[0..n];
    // Generic compiler targets such as air64_v27 do not name a GPU generation.
    const suffix = if (std.mem.startsWith(u8, a, "applegpu_g")) a[10..] else if (a[0] == 'g') a[1..] else return .{ .gen = 0, .phone = false };
    var digits: usize = 0;
    while (digits < suffix.len and std.ascii.isDigit(suffix[digits])) digits += 1;
    if (digits == 0 or suffix.len - digits > 1) return .{ .gen = 0, .phone = false };
    if (digits < suffix.len and !std.ascii.isAlphabetic(suffix[digits])) return .{ .gen = 0, .phone = false };
    return .{
        .gen = std.fmt.parseInt(u32, suffix[0..digits], 10) catch 0,
        .phone = digits < suffix.len and suffix[digits] == 'p',
    };
}

pub fn naxArchSupportedFrom(arch: []const u8) bool {
    const parsed = naxArchGeneration(arch);
    const floor: u32 = if (parsed.phone) 18 else 17;
    return parsed.gen >= floor;
}

pub fn naxArchIsG17(arch: []const u8) bool {
    return naxArchSupportedFrom(arch);
}

/// "26.4"/"26.4.1"-style product version at least req_major.req_minor.
/// Unparseable components read as 0 (mirrors MTPLX: int() failures fall
/// back to 0, so garbage can never satisfy the floor).
pub fn macosVersionAtLeast(ver: []const u8, req_major: u32, req_minor: u32) bool {
    var it = std.mem.splitScalar(u8, ver, '.');
    const major = std.fmt.parseInt(u32, it.first(), 10) catch 0;
    const minor: u32 = if (it.next()) |mn| (std.fmt.parseInt(u32, mn, 10) catch 0) else 0;
    return major > req_major or (major == req_major and minor >= req_minor);
}

/// The whole availability gate, pure over its inputs (mirror of MTPLX's
/// nax_available()): not force-fallback, G17-class GPU, macOS >= 26.2 (the
/// MetalPerformancePrimitives floor).
pub fn naxAvailableFrom(force_fallback: bool, arch: []const u8, os_ver: []const u8) bool {
    if (force_fallback) return false;
    if (!naxArchIsG17(arch)) return false;
    return macosVersionAtLeast(os_ver, 26, 2);
}

/// Human-readable NAX status for `--version` / the app's Settings ("nax"
/// line, first token on/off, remainder the reason). Hardware + OS is the
/// whole story for OUR binaries: the bundled MLX always ships the NAX
/// kernels (build-mlx.sh + tests/test_mlx_staged_nax.sh assert them), so
/// this mirrors MLX's stock-op is_nax_available() gate (GPU gen >= 17,
/// macOS >= 26.2 — that symbol is not exported, hence the mirror).
/// Deliberately ignores the verifyQmm-lane QA env switches: those scope
/// our custom kernel lane, not MLX's stock dispatch.
pub fn naxStatusFrom(arch: []const u8, os_ver: []const u8) []const u8 {
    if (!naxArchIsG17(arch)) return "off (requires M5-class GPU)";
    if (!macosVersionAtLeast(os_ver, 26, 2)) return "off (requires macOS 26.2+)";
    return "on (M5 neural accelerators)";
}

/// naxStatusFrom over the real device: mlx device-info arch + sysctl OS
/// version (same sources as the verifyQmm probe).
pub fn naxStatus() []const u8 {
    var arch_buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&arch_buf) orelse "";
    var ver_buf: [64]u8 = undefined;
    const ver = macosProductVersion(&ver_buf) orelse "";
    return naxStatusFrom(arch, ver);
}

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// "kern.osproductversion" → "26.4"-style string (the sysctl mirror of
/// Python's platform.mac_ver()[0]).
pub fn macosProductVersion(buf: []u8) ?[]const u8 {
    var len: usize = buf.len;
    if (sysctlbyname("kern.osproductversion", buf.ptr, &len, null, 0) != 0) return null;
    var n = @min(len, buf.len);
    while (n > 0 and buf[n - 1] == 0) n -= 1;
    if (n == 0) return null;
    return buf[0..n];
}

/// GPU architecture identifier off mlx device info ("applegpu_g16" on the
/// M4 Max). The returned pointer from mlx is borrowed from the info object,
/// so the string is copied into the caller's buffer before the info frees.
pub fn gpuArchitecture(buf: []u8) ?[]const u8 {
    var dev = mlx.mlx_device{ .ctx = null };
    if (mlx.mlx_get_default_device(&dev) != 0) return null;
    defer _ = mlx.mlx_device_free(dev);
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return null;
    var cstr: [*:0]const u8 = undefined;
    if (mlx.mlx_device_info_get_string(&cstr, info, "architecture") != 0) return null;
    const arch = std.mem.span(cstr);
    if (arch.len == 0 or arch.len > buf.len) return null;
    @memcpy(buf[0..arch.len], arch);
    return buf[0..arch.len];
}

/// Test seam: force the NAX availability probe (null = real probe). Only
/// ever force FALSE on non-G17 machines — forcing true would let dispatch
/// build a kernel whose matmul2d pipeline cannot be created off M5-class
/// hardware.
pub var vqmm_nax_probe_override: ?bool = null;

var vqmm_nax_avail_cache: ?bool = null;

/// M5-class NAX units present: "applegpu_g17" GPU + macOS >= 26.2.
/// SUSHI_FORCE_GPU_FAMILY_FALLBACK=1 pretends the units are absent so
/// an M5 can rehearse the exact M1-M4 plain-SIMD path (QA switch, mirrored
/// from MTPLX). Cached for the process lifetime (inference-thread caller
/// discipline, same as the kernel caches).
pub fn verifyQmmNaxAvailable() bool {
    if (vqmm_nax_probe_override) |v| return v;
    if (vqmm_nax_avail_cache) |v| return v;
    const ok = computeNaxAvailable();
    vqmm_nax_avail_cache = ok;
    return ok;
}

fn computeNaxAvailable() bool {
    var force = false;
    if (std.c.getenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK")) |p| {
        const v = std.mem.span(p);
        force = v.len > 0 and v[0] == '1';
    }
    var arch_buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&arch_buf) orelse "";
    var ver_buf: [64]u8 = undefined;
    const ver = macosProductVersion(&ver_buf) orelse "";
    return naxAvailableFrom(force, arch, ver);
}

var nax_sdpa_avail_cache: ?bool = null;
pub var nax_sdpa_override: ?bool = null; // test seam

/// mlx >= 0.32.2 serves hd-256 attention on NAX with a fused full-attention
/// kernel (`sdpa_full_self_attention_nax`), auto only at >= 1024 causal rows
/// and reachable below that through `force_fused`. Preferred wherever NAX is
/// available; elsewhere sushi_attn_p256 and the split/vector path serve.
pub fn naxSdpaPreferred() bool {
    if (nax_sdpa_override) |v| return v;
    if (nax_sdpa_avail_cache) |v| return v;
    const on = computeNaxAvailable();
    nax_sdpa_avail_cache = on;
    if (on) log.info("[nax-sdpa] engaged: stock fused sdpa (force_fused) serves hd-256 causal prefill and verify blocks > 8 rows; sushi_attn_p256 causal arm declined\n", .{});
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

var qsa_nax_os_cached: ?bool = null;

pub fn qsaNaxOsSupported(version: []const u8) bool {
    return macosVersionAtLeast(version, 26, 3);
}

pub fn qsaNaxOsOk() bool {
    if (qsa_nax_os_cached) |v| return v;
    var version_buf: [64]u8 = undefined;
    const supported = qsaNaxOsSupported(macosProductVersion(&version_buf) orelse "");
    qsa_nax_os_cached = supported;
    return supported;
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

test "NAX availability rejects compiler versions and malformed GPU identifiers" {
    for ([_][]const u8{ "air64_v27", "AIR64_V27", "metal_27", "foo17", "gfoobar27", "g17dd", "applegpu_g", "applegpu_g17_extra" }) |arch| {
        try testing.expect(!naxAvailableFrom(false, arch, "26.6.2"));
    }
    try testing.expect(naxAvailableFrom(false, "AppleGPU_G17S", "26.6.2"));
    try testing.expect(naxAvailableFrom(false, "g18p", "26.6.2"));
    try testing.expect(!naxAvailableFrom(false, "g17p", "26.6.2"));
}

test "NAX availability probe: G17 prefix + macOS 26.2 floor + fallback rehearsal (pure parts)" {
    // Mirrors MTPLX's shipping nax_available() gate exactly (nax_verify.py).
    // Arch: case-insensitive prefix match on "applegpu_g17".
    try testing.expect(naxArchIsG17("applegpu_g17s"));
    try testing.expect(naxArchIsG17("AppleGPU_G17"));
    try testing.expect(naxArchIsG17("applegpu_g17"));
    try testing.expect(!naxArchIsG17("applegpu_g16")); // M4-class
    try testing.expect(!naxArchIsG17("applegpu_g13"));
    try testing.expect(!naxArchIsG17(""));
    try testing.expect(naxArchIsG17("g17"));

    // macOS floor: >= 26.2 (where MetalPerformancePrimitives ships).
    try testing.expect(macosVersionAtLeast("26.2", 26, 2));
    try testing.expect(macosVersionAtLeast("26.4.1", 26, 2));
    try testing.expect(macosVersionAtLeast("27.0", 26, 2));
    try testing.expect(macosVersionAtLeast("27", 26, 2));
    try testing.expect(!macosVersionAtLeast("26.1", 26, 2));
    try testing.expect(!macosVersionAtLeast("26", 26, 2));
    try testing.expect(!macosVersionAtLeast("25.5", 26, 2));
    // Unparseable components read as 0 and never satisfy the floor (MTPLX
    // semantics: int() failures fall back to 0).
    try testing.expect(!macosVersionAtLeast("", 26, 2));
    try testing.expect(!macosVersionAtLeast("garbage", 26, 2));
    try testing.expect(!macosVersionAtLeast("26.x", 26, 2));

    // The combined gate.
    try testing.expect(naxAvailableFrom(false, "applegpu_g17d", "26.2"));
    try testing.expect(!naxAvailableFrom(false, "applegpu_g16", "26.2"));
    try testing.expect(!naxAvailableFrom(false, "applegpu_g17d", "26.1"));
    // SUSHI_FORCE_GPU_FAMILY_FALLBACK=1 QA rehearsal: an M5 pretends the
    // units are absent so the exact M1-M4 plain-SIMD path runs there.
    try testing.expect(!naxAvailableFrom(true, "applegpu_g17d", "26.4"));
}

test "naxStatusFrom: on for G17 + 26.2, off names the missing leg (--version / Settings display)" {
    // Hardware + OS both satisfied → active. Kernels-present is a build-time
    // invariant for our binaries (build-mlx.sh + tests/test_mlx_staged_nax.sh),
    // so hardware capability == MLX's stock-op is_nax_available() gate.
    try testing.expectEqualStrings("on (M5 neural accelerators)", naxStatusFrom("applegpu_g17s", "26.2"));
    try testing.expectEqualStrings("on (M5 neural accelerators)", naxStatusFrom("applegpu_g17d", "27.0"));
    // Pre-M5 GPU: the GPU is the blocker regardless of OS.
    try testing.expectEqualStrings("off (requires M5-class GPU)", naxStatusFrom("applegpu_g16s", "26.4"));
    try testing.expectEqualStrings("off (requires M5-class GPU)", naxStatusFrom("", "26.4"));
    // M5 on an old OS: the OS is the blocker (bundle min is 26.2 anyway, but
    // a dev build can run anywhere).
    try testing.expectEqualStrings("off (requires macOS 26.2+)", naxStatusFrom("applegpu_g17s", "26.1"));
}

test "verifyQmmNaxAvailable: false on every non-G17 device (kernel is never built here)" {
    // The real-device leg of the probe: on this M4 (and every CI runner) the
    // architecture is not applegpu_g17, so the probe must read false and the
    // NAX kernel object can never be constructed. On a real M5 the arch IS
    // G17 and this test only checks the plumbing returned something.
    var buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&buf) orelse {
        // No Metal device info at all: the probe must be false.
        try testing.expect(!verifyQmmNaxAvailable());
        return;
    };
    try testing.expect(arch.len > 0);
    var vbuf: [64]u8 = undefined;
    const ver = macosProductVersion(&vbuf) orelse "<none>";
    errdefer std.debug.print("[nax-probe] arch={s} macos={s} available={}\n", .{ arch, ver, verifyQmmNaxAvailable() });
    if (!naxArchIsG17(arch)) {
        try testing.expect(!verifyQmmNaxAvailable());
    }
}

test "naxMinMForBits: q8 takes over at the measured M7, packed widths at M8" {
    try testing.expectEqual(@as(c_int, 8), naxMinMForBits(4));
    try testing.expectEqual(@as(c_int, 8), naxMinMForBits(6));
    try testing.expectEqual(@as(c_int, 7), naxMinMForBits(8));
}

test "qsaNaxOsSupported: cooperative input tensors require macOS 26.3" {
    try std.testing.expect(!qsaNaxOsSupported("26.2"));
    try std.testing.expect(qsaNaxOsSupported("26.3"));
    try std.testing.expect(qsaNaxOsSupported("26.5"));
    try std.testing.expect(!qsaNaxOsSupported(""));
}

test "naxArchSupportedFrom: desktop gen>=17, phone gen>=18" {
    try std.testing.expect(naxArchSupportedFrom("applegpu_g17s"));
    try std.testing.expect(naxArchSupportedFrom("applegpu_g17"));
    try std.testing.expect(naxArchSupportedFrom("g17"));
    try std.testing.expect(naxArchSupportedFrom("applegpu_g18s"));
    try std.testing.expect(naxArchSupportedFrom("g18"));
    try std.testing.expect(!naxArchSupportedFrom("applegpu_g16s"));
    try std.testing.expect(!naxArchSupportedFrom("g16"));
    try std.testing.expect(!naxArchSupportedFrom("applegpu_g17p"));
    try std.testing.expect(!naxArchSupportedFrom("g17p"));
    try std.testing.expect(naxArchSupportedFrom("applegpu_g18p"));
    try std.testing.expect(naxArchSupportedFrom("g18p"));
    try std.testing.expect(!naxArchSupportedFrom("garbage"));
    try std.testing.expect(!naxArchSupportedFrom(""));
}
