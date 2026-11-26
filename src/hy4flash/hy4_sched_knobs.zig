//! Hy4-Flash scheduler switches and counters a test build reads (`hy4_sched` publishes only dispatch hooks).

/// A group tick verifies each drafting request's block beside the others' in one forward. Drafting requests in company
/// decode as plain joined rows until groups beat them live, so only a test turns groups on.
pub var groups_on_for_test: bool = false;
pub var group_ticks_for_test: u64 = 0;
pub var groups_refused_for_test: u64 = 0;

pub const CopyStreamForTest = struct { request: *@import("hy4_model.zig").Request, tokens: []const u32 };
pub var copy_streams_for_test: ?[]const CopyStreamForTest = null;
