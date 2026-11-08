//! The test files of `src/qwen4/`, read by `src/tests.zig` and `src/tests_host.zig` through the registry. One
//! `@import("<file>.zig"),` a line: build.zig reads `host_tests` as text.

/// The test files the MLX root `src/tests.zig` runs for this arch.
pub const tests = .{
    @import("qwen4_store_test.zig"),
    @import("qwen_vision.zig"),
    @import("qwen4_gdn.zig"),
    @import("qwen4_config.zig"),
    @import("qwen4_forward.zig"),
    @import("qwen4_fixture_test.zig"),
    @import("qwen4_trunk_stq_test.zig"),
    @import("qwen4_hc.zig"),
    @import("gdn_decode.zig"),
    @import("qwen4_qsa.zig"),
    @import("qwen4_qsa_attn.zig"),
    @import("qwen4_qsa_gather.zig"),
    @import("qwen4_qsa_select.zig"),
    @import("qwen4_qsa_state.zig"),
    @import("qwen4_state.zig"),
    @import("qwen4_ssm_group.zig"),
    @import("qwen4_prefix_cache_test.zig"),
    @import("qwen4_disk_qsa.zig"),
    @import("qwen4_exp.zig"),
    @import("qwen4_generate_test.zig"),
    @import("qwen4_shared.zig"),
    @import("qwen4_scheduler_test.zig"),
    @import("qwen4_drafter.zig"),
};
