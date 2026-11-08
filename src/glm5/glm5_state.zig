//! What a GLM-5 model owns beyond the trunk shell: its native model and the default request's KDA/MLA state. They
//! live in `Trunk.arch_state.glm5` and `ForwardCtx.arch_ctx.glm5` (`arch/arch_state.zig`).
const std = @import("std");
const glm5_config = @import("glm5_config.zig");
const mlx = @import("../mlx.zig");
const trunk = @import("../trunk.zig");
const arch_state = @import("../arch/arch_state.zig");
const forward = @import("glm5_forward.zig");
const Trunk = trunk.Trunk;
const ForwardCtx = trunk.ForwardCtx;

/// The model-lifetime state: the native model, and the request the default context (no slot) runs.
pub const State = struct {
    model: ?*forward.Model = null,
    request: ?forward.Request = null,

    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        if (self.request) |*r| r.deinit();
        self.request = null;
        if (self.model) |m| {
            m.deinit();
            allocator.destroy(m);
            self.model = null;
        }
    }

    pub fn reset(self: *State) void {
        if (self.request) |*request| request.reset();
    }

    /// The default context points GLM at this state's own request.
    pub fn ctx(self: *State) Ctx {
        return .{ .request = if (self.request) |*request| request else null };
    }

    /// GLM's native model, not the trunk's layers, runs the forward.
    pub fn runsOwnModel(self: *const State) bool {
        return self.model != null;
    }

    /// The weights are read-only and every slot's request reads them, so slots interleave and batch.
    pub fn sharesReadonly(self: *const State) bool {
        return self.model != null;
    }

    pub fn rawEmbedding(self: *const State, ids: mlx.mlx_array) !?mlx.mlx_array {
        const m = self.model orelse return null;
        return try m.rawEmbedding(ids);
    }
};

/// What a forward context carries for GLM: the request whose KDA/MLA state the forward reads and advances.
pub const Ctx = struct {
    request: ?*forward.Request = null,
};

/// GLM's state on `t`; null in a build without glm5.
pub fn stateOf(t: *Trunk) ?*State {
    return if (comptime @hasField(arch_state.State, "glm5")) &t.arch_state.glm5 else null;
}

/// GLM's native model on `t`: null when the model is not GLM or the build has no glm5.
pub fn modelOf(t: *const Trunk) ?*forward.Model {
    return if (comptime @hasField(arch_state.State, "glm5")) t.arch_state.glm5.model else null;
}

/// GLM's own request on `t` (the one the default context runs); null when it holds none or the build has no glm5.
pub fn ownRequestOf(t: *Trunk) ?*forward.Request {
    const s = stateOf(t) orelse return null;
    return if (s.request) |*request| request else null;
}

/// A forward context's GLM part, pointed at `request`.
pub fn ctxFor(request: *forward.Request) arch_state.Ctx {
    var out: arch_state.Ctx = .{};
    if (comptime @hasField(arch_state.Ctx, "glm5")) out.glm5.request = request;
    return out;
}

/// The batched row tick serves a resident GLM model whose config admits it; a streamed teacher runs one row.
pub fn supportsBatchedRows(t: *const Trunk) bool {
    const glm = modelOf(t) orelse return false;
    return glm5_config.supportsBatchedRows(t.config) and glm.expert_stream == null;
}

/// The GLM request a forward context carries; null when it carries none or the build has no glm5.
pub fn requestOf(ctx: *const ForwardCtx) ?*forward.Request {
    return if (comptime @hasField(arch_state.Ctx, "glm5")) ctx.arch_ctx.glm5.request else null;
}

test "GLM state: a request-only state resets in place and releases through deinit" {
    const testing = std.testing;
    var state: State = .{ .request = try forward.Request.init(testing.allocator, 3) };
    state.request.?.offset = 7;
    state.request.?.failed = true;
    state.reset();
    try testing.expectEqual(@as(usize, 0), state.request.?.offset);
    try testing.expect(!state.request.?.failed);
    try testing.expectEqual(@as(usize, 3), state.request.?.layers.len);
    state.deinit(testing.allocator);
    try testing.expect(state.request == null and state.model == null);
    state.deinit(testing.allocator);
    state.reset();
}

test "GLM state: the default context points at the state's own request, and an empty state at none" {
    const testing = std.testing;
    var state: State = .{};
    try testing.expect(state.ctx().request == null);
    try testing.expect(!state.runsOwnModel() and !state.sharesReadonly());
    state.request = try forward.Request.init(testing.allocator, 2);
    defer state.deinit(testing.allocator);
    try testing.expectEqual(&state.request.?, state.ctx().request.?);
}
