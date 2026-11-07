//! What a GLM-5 model owns beyond the trunk shell: its native model and the default request's KDA/MLA state.
const std = @import("std");
const trunk = @import("../trunk.zig");
const forward = @import("glm5_forward.zig");
const Trunk = trunk.Trunk;

/// The model-lifetime state. Until it replaces `Trunk.glm5` and `Trunk.glm5_request`, those fields are its two members.
pub const State = struct {
    model: ?*forward.Model = null,
    request: ?forward.Request = null,

    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        release(allocator, &self.model, &self.request);
    }

    pub fn reset(self: *State) void {
        if (self.request) |*request| request.reset();
    }
};

/// What a forward context carries for GLM: the request whose KDA/MLA state the forward reads and advances.
pub const Ctx = struct {
    request: ?*forward.Request = null,
};

fn release(allocator: std.mem.Allocator, model: *?*forward.Model, request: *?forward.Request) void {
    if (request.*) |*r| r.deinit();
    request.* = null;
    if (model.*) |m| {
        m.deinit();
        allocator.destroy(m);
        model.* = null;
    }
}

pub fn deinit(self: *Trunk) void {
    release(self.allocator, &self.glm5, &self.glm5_request);
}

pub fn reset(self: *Trunk) void {
    if (self.glm5_request) |*request| request.reset();
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

test "GLM state: State and Ctx hold the types the trunk's GLM fields hold today" {
    const testing = std.testing;
    try testing.expect(@FieldType(trunk.ForwardCtx, "glm5_request") == @FieldType(Ctx, "request"));
    try testing.expect(@FieldType(Trunk, "glm5_request") == @FieldType(State, "request"));
    try testing.expect(@FieldType(Trunk, "glm5") == @FieldType(State, "model"));
    const ctx: Ctx = .{};
    try testing.expect(ctx.request == null);
}
