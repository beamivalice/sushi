//! What a DeepSeek-V4.1 model owns beyond the trunk shell: its native model, whose one request's state it keeps, and
//! the streams its routed experts come through. They live in `Trunk.arch_state.dsv41` (`arch/arch_state.zig`).
const std = @import("std");
const trunk = @import("../trunk.zig");
const arch_state = @import("../arch/arch_state.zig");
const dsv41_model = @import("dsv41_model.zig");
const dsv41_stream = @import("dsv41_stream.zig");
const Trunk = trunk.Trunk;

/// The model-lifetime state.
pub const State = struct {
    model: ?*dsv41_model.Model = null,
    experts: ?*dsv41_stream.Stream = null,
    stq_experts: ?*dsv41_stream.StqStream = null,

    /// `Model.deinit` frees the model itself, and a `Stream` holds no resource of its own (the engine it reads is the
    /// trunk's `expert_stream`): neither takes more than the destroy below.
    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        if (self.model) |m| {
            m.deinit();
            self.model = null;
        }
        if (self.experts) |st| {
            allocator.destroy(st);
            self.experts = null;
        }
        if (self.stq_experts) |st| {
            st.deinit();
            allocator.destroy(st);
            self.stq_experts = null;
        }
    }

    pub fn reset(self: *State) void {
        if (self.model) |m| dsv41_model.resetState(m);
    }
};

/// DeepSeek's state on `t`; null in a build without dsv41.
pub fn stateOf(t: *Trunk) ?*State {
    return if (comptime @hasField(arch_state.State, "dsv41")) &t.arch_state.dsv41 else null;
}

/// DeepSeek's native model on `t`: null when the model is not DeepSeek or the build has no dsv41.
pub fn modelOf(t: *const Trunk) ?*dsv41_model.Model {
    return if (comptime @hasField(arch_state.State, "dsv41")) t.arch_state.dsv41.model else null;
}

/// The mxfp4 expert stream on `t`; null on an STQ pack, a non-DeepSeek model or a build without dsv41.
pub fn expertsOf(t: *const Trunk) ?*dsv41_stream.Stream {
    return if (comptime @hasField(arch_state.State, "dsv41")) t.arch_state.dsv41.experts else null;
}

/// The STQ expert stream on `t`; null on an mxfp4 pack, a non-DeepSeek model or a build without dsv41.
pub fn stqExpertsOf(t: *const Trunk) ?*dsv41_stream.StqStream {
    return if (comptime @hasField(arch_state.State, "dsv41")) t.arch_state.dsv41.stq_experts else null;
}

test "DeepSeek state: an empty state releases and resets as a no-op, twice" {
    var state: State = .{};
    state.reset();
    state.deinit(std.testing.allocator);
    state.deinit(std.testing.allocator);
    try std.testing.expect(state.model == null and state.experts == null and state.stq_experts == null);
}
