//! The test files of `src/mimo2/`, read by `src/tests.zig` and `src/tests_host.zig` through the registry. One
//! `@import("<file>.zig"),` a line: build.zig reads `host_tests` as text.

/// The test files the MLX root `src/tests.zig` runs for this arch.
pub const tests = .{
    @import("mimo_quant_test.zig"),
    @import("mimo_config.zig"),
    @import("mimo_generate_test.zig"),
    @import("mimo_vision.zig"),
    @import("mimo_forward.zig"),
    @import("mimo_fixture_test.zig"),
    @import("mimo_exl3_moe_test.zig"),
    @import("mimo2_state.zig"),
    @import("mimo2_shortlist.zig"),
    @import("mimo_mtp.zig"),
    @import("mimo_source.zig"),
    @import("mimo_scheduler_test.zig"),
    @import("mimo2_load.zig"),
    @import("mimo2_drafter.zig"),
    @import("mimo_bill.zig"),
};
