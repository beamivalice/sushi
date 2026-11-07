//! What a DeepSeek-V4.1 model owns beyond the trunk shell: its native model, whose one request's state it keeps, and
//! the streams its routed experts come through.
const std = @import("std");
const trunk = @import("../trunk.zig");
const dsv41_model = @import("dsv41_model.zig");
const dsv41_stream = @import("dsv41_stream.zig");
const Trunk = trunk.Trunk;

/// The model-lifetime state. Until it replaces the three `Trunk.dsv41*` fields, those are its three members.
pub const State = struct {
    model: ?*dsv41_model.Model = null,
    experts: ?*dsv41_stream.Stream = null,
    stq_experts: ?*dsv41_stream.StqStream = null,

    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        release(allocator, &self.model, &self.experts, &self.stq_experts);
    }

    pub fn reset(self: *State) void {
        if (self.model) |m| dsv41_model.resetState(m);
    }
};

fn release(
    allocator: std.mem.Allocator,
    model: *?*dsv41_model.Model,
    experts: *?*dsv41_stream.Stream,
    stq_experts: *?*dsv41_stream.StqStream,
) void {
    if (model.*) |m| {
        m.deinit();
        model.* = null;
    }
    if (experts.*) |st| {
        allocator.destroy(st);
        experts.* = null;
    }
    if (stq_experts.*) |st| {
        st.deinit();
        allocator.destroy(st);
        stq_experts.* = null;
    }
}

pub fn deinit(self: *Trunk) void {
    release(self.allocator, &self.dsv41, &self.dsv41_experts, &self.dsv41_stq_experts);
}

pub fn reset(self: *Trunk) void {
    if (self.dsv41) |m| dsv41_model.resetState(m);
}

test "DeepSeek state: an empty state releases and resets as a no-op, twice" {
    var state: State = .{};
    state.reset();
    state.deinit(std.testing.allocator);
    state.deinit(std.testing.allocator);
    try std.testing.expect(state.model == null and state.experts == null and state.stq_experts == null);
}

test "DeepSeek state: State holds the types the trunk's DeepSeek fields hold today" {
    try std.testing.expect(@FieldType(Trunk, "dsv41") == @FieldType(State, "model"));
    try std.testing.expect(@FieldType(Trunk, "dsv41_experts") == @FieldType(State, "experts"));
    try std.testing.expect(@FieldType(Trunk, "dsv41_stq_experts") == @FieldType(State, "stq_experts"));
}
