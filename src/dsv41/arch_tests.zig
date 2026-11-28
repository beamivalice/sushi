//! The test files of `src/dsv41/`, read by `src/tests.zig` and `src/tests_host.zig` through the registry. One
//! `@import("<file>.zig"),` a line: build.zig reads `host_tests` as text.

/// The test files the MLX root `src/tests.zig` runs for this arch.
pub const tests = .{
    @import("dsv41_vision.zig"),
    @import("dsv41_bills_live_test.zig"),
    @import("dsv41_config.zig"),
    @import("dsv41_wide_bytes.zig"),
    @import("dsv41_qat.zig"),
    @import("dsv41_engram.zig"),
    @import("dsv41_source.zig"),
    @import("dsv41_model.zig"),
    @import("dsv41_plan.zig"),
    @import("dsv41_plan_test.zig"),
    @import("dsv41_sample_plan.zig"),
    @import("dsv41_sample_plan_test.zig"),
    @import("dsv41_encode_plan.zig"),
    @import("dsv41_encode_plan_test.zig"),
    @import("dsv41_close_test.zig"),
    @import("dsv41_stream.zig"),
    @import("dsv41_state.zig"),
    @import("dsv41_borrow.zig"),
    @import("dsv41_prefix.zig"),
    @import("dsv41_prefix_cache_test.zig"),
    @import("dsv41_serving.zig"),
    @import("dsv41_pipelines.zig"),
    @import("dsv41_query_count_test.zig"),
    @import("dsv41_exl3_real_test.zig"),
    @import("dsv41_kernels.zig"),
    @import("kld_capture.zig"),
    @import("dsv41_thought_point.zig"),
    @import("repack_test.zig"),
    @import("arch.zig"),
    @import("dsv41_sched.zig"),
    @import("dsv41_scope.zig"),
    @import("dsv41_generate_test.zig"),
};

/// The MLX-free test files `src/tests_host.zig` runs for this arch.
pub const host_tests = .{
    @import("dsv41_bills_test.zig"),
    @import("dsv41_flags_test.zig"),
    @import("dsv41_replay_test.zig"),
    @import("dsv41_chat.zig"),
};
