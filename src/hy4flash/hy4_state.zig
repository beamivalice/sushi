//! What a Hy4-Flash model owns beyond the trunk shell (`Trunk.arch_state.hy4flash`, `arch/arch_state.zig`): its
//! composed model and the pack's tensors the model reads, moved out of the loader's map so they live as long as it.
const std = @import("std");
const model_mod = @import("../model.zig");
const hy4_model = @import("hy4_model.zig");
const hy4_dfly = @import("hy4_dfly.zig");

pub const State = struct {
    model: ?*hy4_model.Model = null,
    weights: ?*model_mod.Weights = null,
    /// The loader's map, which keeps the vision tower's tensors (`takeWeights`) for as long as the model: the plans
    /// record the encode over them (`hy4_encode_plan`).
    loader_weights: ?*const model_mod.Weights = null,
    /// DFly2, when the load keeps it (`dsparkOn`).
    drafter: ?hy4_dfly.Drafter = null,
    /// A DSpark round's open verify: the state before the block and the block's ids (`hy4_dspark`).
    verify: ?Verify = null,

    pub const Verify = struct { mark: ?hy4_model.Model.Mark = null, ids: [16]u32, n: usize };

    /// Whether a served request may draft: DFly2 rolls the model back to the verify's mark itself.
    pub fn specRollsBack(self: *const State) bool {
        return self.model != null and self.drafter != null;
    }

    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        dropVerify(self, false);
        if (self.drafter) |*d| {
            d.deinit();
            self.drafter = null;
        }
        if (self.model) |m| {
            m.deinit();
            self.model = null;
        }
        if (self.weights) |w| {
            w.deinit();
            allocator.destroy(w);
            self.weights = null;
        }
    }

    /// The composed model, not the trunk's layers, runs every forward.
    pub fn runsOwnModel(self: *const State) bool {
        return self.model != null;
    }
};

/// Moves every entry of `from` but the vision tower's into a new map the caller owns. The tower stays in `from`, the
/// map the shared loader builds its vision encoder from; `deinit` of `from` frees nothing the model reads.
pub fn takeWeights(allocator: std.mem.Allocator, from: *model_mod.Weights, c: *const @import("hy4_config.zig").Config) !*model_mod.Weights {
    const own = try allocator.create(model_mod.Weights);
    own.* = model_mod.Weights.init(allocator);
    errdefer {
        own.deinit();
        allocator.destroy(own);
    }
    try own.map.ensureTotalCapacity(from.map.count());
    var it = from.map.iterator();
    while (it.next()) |e| {
        if (@import("hy4_census.zig").roleOf(c, e.key_ptr.*)) |r| if (r.role == .vision) continue else {} else |_| {}
        own.map.putAssumeCapacity(e.key_ptr.*, e.value_ptr.*);
    }
    var moved = own.map.iterator();
    while (moved.next()) |e| _ = from.map.remove(e.key_ptr.*);
    return own;
}

/// Drops an open verify's mark, if any; with `reset` the model's state too (a round that failed mid-way).
pub fn dropVerify(st: *State, reset: bool) void {
    var v = st.verify orelse return;
    st.verify = null;
    const m = st.model orelse return;
    if (v.mark) |*mk| m.dropMark(mk);
    if (reset) m.reset();
}

test "hy4 state: the model takes the load's tensors but the vision tower's, which the shared loader's encoder reads" {
    const testing = std.testing;
    const mlx = @import("../mlx.zig");
    const a = testing.allocator;
    const cfg = try @import("hy4_config.zig").parseText(@import("hy4_config.zig").release_config);
    var from = model_mod.Weights.init(a);
    defer from.deinit();
    for ([_][]const u8{ "vit.embeddings.patch_embedding.weight", "model.embed_tokens.weight" }) |k| {
        try from.map.put(try a.dupe(u8, k), try mlx.newData(&[_]f32{1}, &.{1}, .float32));
    }
    const own = try takeWeights(a, &from, &cfg);
    defer {
        own.deinit();
        a.destroy(own);
    }
    try testing.expect(own.get("model.embed_tokens.weight") != null);
    try testing.expect(own.get("vit.embeddings.patch_embedding.weight") == null);
    try testing.expect(from.get("vit.embeddings.patch_embedding.weight") != null);
}
