//! What a MiMo-V2.6-Flash model keeps beside the trunk: the greedy readout's shortlist cell, and on a streamed load the
//! expert engine (the trunk's `expert_stream`). It lives in `Trunk.arch_state.mimo2` (`arch/arch_state.zig`).
const std = @import("std");
const mlx = @import("../mlx.zig");
const mtp_mod = @import("../mtp.zig");
const trunk = @import("../trunk.zig");
const arch_state = @import("../arch/arch_state.zig");
const shortlist_mod = @import("mimo2_shortlist.zig");
const mimo_mtp = @import("mimo_mtp.zig");
const Trunk = trunk.Trunk;

/// The model-lifetime state.
pub const State = struct {
    /// Held by pointer: a readout holds `*const Trunk`, which the compiler may assume is never written, and the cell's
    /// pointee is not const. One heap object per model, so a copy of the state never duplicates the coarse head.
    shortlist: ?*shortlist_mod.Cell = null,

    /// Makes the shortlist cell; a streamed load also gets its expert engine.
    pub fn init(self: *State, args: arch_state.InitArgs) !?Trunk.ExpertStreamParts {
        const cell = try args.allocator.create(shortlist_mod.Cell);
        errdefer args.allocator.destroy(cell);
        cell.* = .{};
        const parts = if (args.config.expert_streaming) try Trunk.initExpertStream(args.allocator, args.config, args.s) else null;
        self.shortlist = cell;
        return parts;
    }

    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        const cell = self.shortlist orelse return;
        mtp_mod.retireCoarse(&cell.coarse);
        allocator.destroy(cell);
        self.shortlist = null;
    }

    pub fn headShortlist(self: *const State, t: *const Trunk, x: mlx.mlx_array) !?mlx.mlx_array {
        return shortlist_mod.readout(t, self.shortlist orelse return null, x);
    }
};

/// What a forward context carries for MiMo.
pub const Ctx = struct {
    /// A spec verify: every row of this multi-row forward computes with its own decode tick's arithmetic
    /// (`mimoVerifyRowsAttn`), so greedy MTP is serial.
    verify_rows: bool = false,
    /// The batched decode (`forwardMimoBatchedDecode`): row i of this `[1, N]` forward is slot i's decode tick, its
    /// attention and KV append on `batch_rows[i]`'s cache at `batch_row_offsets[i]`.
    batch_rows: ?[]const *trunk.ForwardCtx = null,
    batch_row_offsets: []const u32 = &.{},

    /// Marks the forward a speculative verify block (`on`) or not.
    pub fn setVerifyRows(self: *Ctx, on: bool) void {
        self.verify_rows = on;
    }

    /// The widest verify block whose rows keep their decode tick's arithmetic.
    pub fn verifyRowsMax() usize {
        return @intCast(@import("mimo_forward.zig").MIMO_VERIFY_ROWS_MAX);
    }
};

/// The coarse lm_head copy's slot in `t`'s shortlist cell; null on a trunk without one or a build without mimo2.
pub fn coarseHead(t: *const Trunk) ?*?mtp_mod.RerankCoarse {
    if (comptime !@hasField(arch_state.State, "mimo2")) return null;
    const cell = t.arch_state.mimo2.shortlist orelse return null;
    return &cell.coarse;
}

test "MiMo state: a verify block is marked on the context and cleared" {
    var ctx: Ctx = .{};
    try std.testing.expect(!ctx.verify_rows and ctx.batch_rows == null and ctx.batch_row_offsets.len == 0);
    ctx.setVerifyRows(true);
    try std.testing.expect(ctx.verify_rows);
    ctx.setVerifyRows(false);
    try std.testing.expect(!ctx.verify_rows);
}

test "MiMo state: an empty state releases as a no-op, twice, and holds no coarse head" {
    var state: State = .{};
    state.deinit(std.testing.allocator);
    state.deinit(std.testing.allocator);
    try std.testing.expect(state.shortlist == null);
}

test "MiMo state: a resident load's init makes the shortlist cell and starts no expert engine" {
    var state: State = .{};
    var name_buf: [256]u8 = undefined;
    var owned: ?[]mlx.mlx_array = null;
    var weights: @import("../model.zig").Weights = undefined;
    const parts = try state.init(.{
        .allocator = std.testing.allocator,
        .config = &.{ .model_type = "mimo_v2" },
        .weights = &weights,
        .prefix = "",
        .name_buf = &name_buf,
        .s = undefined,
        .owned_bf16 = &owned,
    });
    defer state.deinit(std.testing.allocator);
    try std.testing.expect(parts == null);
    try std.testing.expect(state.shortlist != null and state.shortlist.?.coarse == null and !state.shortlist.?.logged);
}

/// A RerankFixture trunk as MiMo: its state holds `cell`, whose coarse head is built at `bits`.
fn mimoFixture(fx: *mtp_mod.RerankFixture, cell: *shortlist_mod.Cell, bits: u32) !void {
    fx.xfm.config.model_type = "mimo_v2";
    fx.xfm.suppress_mask = null;
    fx.xfm.bits_cache = null;
    fx.xfm.arch_state = .{};
    fx.xfm.arch_state.mimo2.shortlist = cell;
    if (bits > 0) cell.coarse = mtp_mod.buildRerankCoarse(mlx.gpuStream(), &fx.xfm, bits) orelse return error.CoarseBuildDeclined;
}

fn bf16Row(s: mlx.mlx_stream, rows: c_int, buf: []const f32) !mlx.mlx_array {
    const xf = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, rows, 512 }, 3, .float32);
    defer _ = mlx.mlx_array_free(xf);
    var x = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, s));
    return x;
}

test "a MiMo shortlist cell filled on first use is written once and read back through the state's pointer" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var fx = try mtp_mod.RerankFixture.init(s, mtp_mod.TOP32_MIN_ROWS + 96, 512, 8, 64, 0x1A4D);
    defer fx.deinit();
    var cell: shortlist_mod.Cell = .{};
    try mimoFixture(&fx, &cell, 0);
    defer mtp_mod.retireCoarse(&cell.coarse);

    // The coarse head: built by the first ask, kept by the second.
    try std.testing.expect(mimo_mtp.buildCoarseHead(s, &fx.xfm));
    const built = cell.coarse.?.q.w.ctx;
    try std.testing.expect(mimo_mtp.buildCoarseHead(s, &fx.xfm));
    try std.testing.expectEqual(built, cell.coarse.?.q.w.ctx);

    // The one-time log flag: a readout holds `*const Trunk` and still sets it, once.
    var buf: [512]f32 = undefined;
    for (&buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.03125;
    const x = try bf16Row(s, 1, &buf);
    defer _ = mlx.mlx_array_free(x);
    const ro: *const Trunk = &fx.xfm;
    for (0..2) |_| {
        const out = try ro.lmHeadProject(x, true);
        defer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        try std.testing.expect(fx.xfm.arch_state.mimo2.shortlist.?.logged);
        try std.testing.expectEqual(built, coarseHead(ro).?.*.?.q.w.ctx);
    }
}

test "a retired coarse head is gone for every reader and is not freed twice, and another trunk's stays" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var a = try mtp_mod.RerankFixture.init(s, mtp_mod.TOP32_MIN_ROWS + 96, 512, 8, 64, 0x1A4D);
    defer a.deinit();
    var b = try mtp_mod.RerankFixture.init(s, mtp_mod.TOP32_MIN_ROWS + 96, 512, 8, 64, 0x1A4E);
    defer b.deinit();
    var cells: [2]shortlist_mod.Cell = .{ .{ .logged = true }, .{ .logged = true } };
    for ([_]*mtp_mod.RerankFixture{ &a, &b }, &cells) |fx, *cell| try mimoFixture(fx, cell, 2);
    defer for (&cells) |*cell| mtp_mod.retireCoarse(&cell.coarse);

    // What a failed shortlist does: the MiMo readers go through `coarseHead`, which now reads null.
    mtp_mod.retireCoarse(coarseHead(&a.xfm).?);
    try std.testing.expect(coarseHead(&a.xfm).?.* == null);
    try std.testing.expect(coarseHead(&b.xfm).?.* != null);
    // Retiring again, as the deinit does, frees nothing twice.
    mtp_mod.retireCoarse(coarseHead(&a.xfm).?);
    // The readout falls back to the full head on A and keeps its shortlist on B.
    var buf: [512]f32 = undefined;
    for (&buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) * 0.03125;
    const x = try bf16Row(s, 1, &buf);
    defer _ = mlx.mlx_array_free(x);
    const full = try a.xfm.lmHeadProject(x, true);
    defer _ = mlx.mlx_array_free(full);
    const sparse = try b.xfm.lmHeadProject(x, true);
    defer _ = mlx.mlx_array_free(sparse);
    try mlx.check(mlx.mlx_array_eval(full));
    try mlx.check(mlx.mlx_array_eval(sparse));
}

test "a MiMo argmax-only readout holds the full head's bit-exact logits at its top-32, one row or four" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    // K 512 and a vocab that is a multiple of 8 keep both reads on the served head's qmv kernel.
    var fx = try mtp_mod.RerankFixture.init(s, mtp_mod.TOP32_MIN_ROWS + 96, 512, 8, 64, 0x1A4D);
    defer fx.deinit();
    var cell: shortlist_mod.Cell = .{ .logged = true };
    try mimoFixture(&fx, &cell, 2);
    defer mtp_mod.retireCoarse(&cell.coarse);
    const vocab: usize = @intCast(fx.vocab);
    var prng = std.Random.DefaultPrng.init(0x7E57);
    for ([_]c_int{ 1, 4 }) |rows| {
        const n: usize = @intCast(rows);
        var buf: [4 * 512]f32 = undefined;
        for (buf[0 .. n * 512]) |*v| v.* = @as(f32, @floatFromInt(prng.random().intRangeAtMost(i32, -64, 63))) * 0.015625;
        const x = try bf16Row(s, rows, buf[0 .. n * 512]);
        defer _ = mlx.mlx_array_free(x);
        const sparse = try fx.xfm.lmHeadProject(x, true);
        defer _ = mlx.mlx_array_free(sparse);
        const full = try fx.xfm.lmHeadProject(x, false);
        defer _ = mlx.mlx_array_free(full);
        var sf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sf);
        try mlx.check(mlx.mlx_astype(&sf, sparse, .float32, s));
        var ff = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ff);
        try mlx.check(mlx.mlx_astype(&ff, full, .float32, s));
        try mlx.check(mlx.mlx_array_eval(sf));
        try mlx.check(mlx.mlx_array_eval(ff));
        const sp = mlx.mlx_array_data_float32(sf) orelse return error.Unreadable;
        const fp = mlx.mlx_array_data_float32(ff) orelse return error.Unreadable;
        for (0..n) |r| {
            var kept: usize = 0;
            for (sp[r * vocab ..][0..vocab], fp[r * vocab ..][0..vocab]) |va, vb| {
                if (va == -std.math.inf(f32)) continue;
                kept += 1;
                try std.testing.expectEqual(vb, va);
            }
            try std.testing.expectEqual(@as(usize, 32), kept);
        }
    }
}
