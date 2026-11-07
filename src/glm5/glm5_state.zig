//! What a GLM-5 model owns beyond the trunk shell: its native model and the default request's KDA/MLA state.
const trunk = @import("../trunk.zig");
const Trunk = trunk.Trunk;

pub fn deinit(self: *Trunk) void {
    if (self.glm5_request) |*request| request.deinit();
    self.glm5_request = null;
    if (self.glm5) |mdl| {
        mdl.deinit();
        self.allocator.destroy(mdl);
        self.glm5 = null;
    }
}

pub fn reset(self: *Trunk) void {
    if (self.glm5_request) |*request| request.reset();
}
