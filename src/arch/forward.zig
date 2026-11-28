//! Forward dispatch over the loaded arch. Inference thread only.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model_handle = @import("../model_handle.zig");
const registry = @import("registry.zig");
const transient_peak = @import("../transient_peak.zig");
const arch_cooler = @import("cooler.zig");
const Arch = registry.Arch;

/// One prefill chunk, decode step or verify block, as `ctx` says.
/// The late-compile line's words for the forward in flight (`engine.lateWhere`).
var where_buf: [48]u8 = undefined;

pub fn forward(a: Arch, ctx: *model_handle.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
    const config = @import("handle.zig").config(a);
    const rows = arch_cooler.enterForward(config, ctx, ids, @import("handle.zig").trunk(a), transient_peak.enabled());
    const where = if (rows) |r| std.fmt.bufPrint(&where_buf, "a forward of {d} rows to {d}", .{ r.rows, r.seq }) catch "a forward" else "a forward";
    const was = mlx.engine.lateWhere(where);
    defer _ = mlx.engine.lateWhere(was);
    // Paired: `rows` is read whenever the peak diagnostic is on.
    transient_peak.begin();
    defer if (rows) |r| transient_peak.end(.ofRows(r.rows), r.rows, r.cap(config));
    const one = if (recordings) |r| try r.begin(ids) else null;
    defer if (one) |o| recordings.?.end(o);
    return switch (a) {
        inline else => |m, t| registry.Mod(t).forward(m, ctx, ids),
    };
}

const core = @import("../tensor/core.zig");
const plan = @import("../tensor/plan/plan_api.zig");

/// While set, every forward is recorded as an execution of its own on the pool (docs/bill_redesign.md, stage 1), its
/// root the forward's site and rows.
pub var recordings: ?*Recordings = null;

pub const Recordings = struct {
    gpa: std.mem.Allocator,
    list: std.ArrayList(*Recorded) = .empty,
    /// A recording whose levels are short of some dispatches (a level map could not grow).
    short: bool = false,

    /// One forward's recording, not yet ended: the recorder holds what it asked, and the pool's count of asks then.
    pub const Recorded = struct {
        recorder: plan.Recorder,
        pool_asks: u64 = 0,
        pool_ask_bytes: u64 = 0,
        stats: core.Stats = .{},
    };

    fn begin(r: *Recordings, ids: mlx.mlx_array) !*Recorded {
        _ = ids;
        const one = try r.gpa.create(Recorded);
        errdefer r.gpa.destroy(one);
        one.* = .{ .recorder = plan.Recorder.begin(r.gpa) };
        errdefer one.recorder.deinit();
        try r.list.append(r.gpa, one);
        errdefer _ = r.list.pop();
        try core.beginRecording(.real, &one.recorder);
        one.stats = core.stats();
        return one;
    }

    fn end(r: *Recordings, one: *Recorded) void {
        const now = core.stats();
        one.pool_asks = now.asks - one.stats.asks;
        one.pool_ask_bytes = now.ask_bytes - one.stats.ask_bytes;
        core.endRecording() catch {
            r.short = true;
        };
    }

    pub fn deinit(r: *Recordings) void {
        for (r.list.items) |one| {
            one.recorder.deinit();
            r.gpa.destroy(one);
        }
        r.list.deinit(r.gpa);
    }
};

/// Draft logits: the hidden state through the trunk's bare head.
pub fn lmHeadForDraft(a: Arch, x: mlx.mlx_array) !mlx.mlx_array {
    return switch (a) {
        inline else => |m, t| registry.Mod(t).lmHeadForDraft(m, x),
    };
}
