//! What a DeepSeek-V4.1 model owns beyond the trunk shell: its native model, whose one request's state it keeps, and
//! the streams its routed experts come through.
const trunk = @import("../trunk.zig");
const dsv41_model = @import("dsv41_model.zig");
const Trunk = trunk.Trunk;

pub fn deinit(self: *Trunk) void {
    if (self.dsv41) |m| {
        m.deinit();
        self.dsv41 = null;
    }
    if (self.dsv41_experts) |st| {
        self.allocator.destroy(st);
        self.dsv41_experts = null;
    }
    if (self.dsv41_stq_experts) |st| {
        st.deinit();
        self.allocator.destroy(st);
        self.dsv41_stq_experts = null;
    }
}

pub fn reset(self: *Trunk) void {
    if (self.dsv41) |m| dsv41_model.resetState(m);
}
