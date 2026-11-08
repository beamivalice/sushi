//! The YaRN geometry of Flash-Next's shipped rope and the reference's inv_freq for it, for the rope tests.
const Yarn = @import("mrope.zig").Yarn;

/// The shipped Qwen3.8-Flash-Next rope geometry, from the checkpoint's own
/// config.json: head_dim 256 × partial_rotary_factor 0.25 → rotary_dim 64 (so
/// 32 frequencies, which is what `mrope_section` [11,11,10] sums to), theta 1e7,
/// pre-trained window 262144.
pub const QWEN4_YARN = Yarn{
    .theta = 10_000_000.0,
    .factor = 4.0,
    .orig_max = 262_144,
    .rotary_dim = 64,
};

/// `_compute_yarn_parameters` output for QWEN4_YARN, produced by the reference
/// (tests/qwen4/dump_qwen4_yarn_fixtures.py, transformers 5.9) — and independently by
/// a transcription of vLLM's `YaRNScalingRotaryEmbedding._compute_inv_freq`,
/// which agrees with HF bit-for-bit here. f64: the references compute in f32,
/// which is 1e-7 sloppier than these.
pub const QWEN4_YARN_GOLDEN = [_]f64{
    1.0, // 0   extrapolated band (i < low): unscaled
    0.6042963902381328, // 1
    0.36517412725483767, // 2
    0.220673406908459, // 3
    0.1333521432163324, // 4
    0.08058421877614819, // 5
    0.04869675251658631, // 6
    0.029427271762092817, // 7
    0.01778279410038923, // 8
    0.010746078283213174, // 9
    0.006493816315762114, // 10
    0.003924189758484536, // 11
    0.002371373705661655, // 12
    0.0014330125702369627, // 13
    0.0008659643233600654, // 14  last ramp-free extrapolation
    0.0004742398226801046, // 15  blended: 0.90625 of unscaled
    0.00025693505988868084, // 16
    0.0001373497450760004, // 17
    0.00007217387404309113, // 18
    0.0000370722498206804, // 19
    0.000018449222025000475, // 20
    0.000008759770071179004, // 21  blended: 0.34375 of unscaled
    0.0000038498163151487305, // 22 interpolated band (i >= high): exactly /4
    0.0000023264301023242476, // 23
    0.0000014058533129758728, // 24
    0.0000008495520822356398, // 25
    0.0000005133812566142865, // 26
    0.0000003102344401879299, // 27
    0.00000018747355233311396, // 28
    0.00000011328959094002045, // 29
    0.00000006846049085660903, // 30
    0.00000004137042749857954, // 31
};
