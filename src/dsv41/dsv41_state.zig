//! What a DeepSeek-V4.1 model owns beyond the trunk shell: its native model, which parks the decode state between
//! requests, and the streams its routed experts come through. They live in `Trunk.arch_state.dsv41` (`arch/arch_state.zig`).
const std = @import("std");
const trunk = @import("../trunk.zig");
const arch_state = @import("../arch/arch_state.zig");
const dsv41_model = @import("dsv41_model.zig");
const dsv41_stream = @import("dsv41_stream.zig");
const dsv41_borrow = @import("dsv41_borrow.zig");
const mlx = @import("../mlx.zig");
const Trunk = trunk.Trunk;

/// The model-lifetime state.
pub const State = struct {
    model: ?*dsv41_model.Model = null,
    experts: ?*dsv41_stream.Stream = null,
    stq_experts: ?*dsv41_stream.StqStream = null,
    /// What `lendOwn` replaced, for `endOwn`.
    own_run: ?dsv41_model.Lent = null,

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

    /// DeepSeek cuts its own rows back to the accepted prefix (`dsv41_model.finishVerify`) when it drafts with DSpark.
    pub fn specRollsBack(self: *const State) bool {
        return if (self.model) |m| m.dspark != null else false;
    }

    /// The stream the sampler and the token arrays run on after a forward: the engine's while decode runs there.
    pub fn outStream(self: *const State, s: mlx.mlx_stream) mlx.mlx_stream {
        const m = self.model orelse return s;
        return if (m.onEngine()) mlx.engine.stream() else s;
    }

    /// The served warm's forwards run under the caps a request of their rows has (`dsv41_borrow.enterWarm`), and its
    /// requests, which no slot carries, read the model's own state: with joined rows it is lent for the warm.
    pub fn servedWarmBegin(self: *State) void {
        dsv41_borrow.warm_budget = true;
        const m = self.model orelse return;
        if (m.serves_rows and m.bound == null) self.own_run = dsv41_model.lendOwn(m);
    }

    pub fn servedWarmEnd(self: *State) void {
        dsv41_borrow.endWarmMlx();
        if (self.own_run) |l| l.end();
        self.own_run = null;
    }

    /// The state a forward with no slot request runs is reset: the model's own, with joined rows lent for the reset.
    pub fn reset(self: *State) void {
        const m = self.model orelse return;
        const own: ?dsv41_model.Lent = if (m.serves_rows and m.bound == null) dsv41_model.lendOwn(m) else null;
        defer if (own) |l| l.end();
        dsv41_model.resetState(m);
    }
};

/// A forward's DeepSeek part: the slot request whose decode state it runs (`dsv41_model.bind`).
pub const Ctx = struct {
    request: ?*dsv41_model.Request = null,
};

/// The slot request a forward context carries; null when it carries none or the build has no dsv41.
pub fn requestOf(ctx: *const trunk.ForwardCtx) ?*dsv41_model.Request {
    return if (comptime @hasField(arch_state.Ctx, "dsv41")) ctx.arch_ctx.dsv41.request else null;
}

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
