//! Hy4-Flash's capture and restore on the static bill (docs/bill_redesign.md part 8): a point is a `Points` slot written
//! in place, and a restore installs into the request's own state, so neither asks the engine's pool once it is closed.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const core = @import("../tensor/core.zig");
const fwd = @import("hy4_forward.zig");
const hy4_config = @import("hy4_config.zig");
const hy4_model = @import("hy4_model.zig");
const prefix = @import("hy4_prefix.zig");
const pc = @import("../prefix_cache.zig");
const kv_disk_cache = @import("../kv_disk_cache.zig");
const KVCache = @import("../kv_cache.zig").KVCache;
const Arr = mlx.mlx_array;

/// The BF16 tiny's pack and geometry, its arrays on the CPU stream.
const Tiny = struct {
    pack: model_mod.Weights,
    config: hy4_config.Config,
    cpu: mlx.mlx_stream,

    fn open() !Tiny {
        const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
        const a = testing.allocator;
        var t: Tiny = .{ .pack = model_mod.Weights.init(a), .config = undefined, .cpu = mlx.mlx_default_cpu_stream_new() };
        errdefer _ = mlx.mlx_stream_free(t.cpu);
        errdefer t.pack.deinit();
        const path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{dir}, 0);
        defer a.free(path);
        try model_mod.loadSafetensorsFile(a, &t.pack, path.ptr, t.cpu, true);
        const io = std.Io.Threaded.global_single_threaded.io();
        var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
        defer d.close(io);
        const text = try d.readFileAlloc(io, "config.json", a, .limited(1 << 20));
        defer a.free(text);
        t.config = try hy4_config.parseText(text);
        return t;
    }

    fn close(t: *Tiny) void {
        t.pack.deinit();
        _ = mlx.mlx_stream_free(t.cpu);
    }
};

/// The closed pool's asks `f` makes, served by the census.
fn closedAsks(comptime f: anytype, args: anytype) !u64 {
    try core.closePool(.census);
    defer core.closePool(.open) catch {};
    const before = core.stats().closed_asks;
    try @call(.auto, f, args);
    return core.stats().closed_asks - before;
}

/// The last row's logits of `ids` run on `m`, as host values the caller frees.
fn logitsOf(o: *Ops, m: *hy4_model.Model, ids: []const u32) ![]f32 {
    const l = try o.own(try m.forwardReadout(ids, .last));
    const c = try o.contiguous(try o.cast(l, .float32));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_float32(c) orelse return error.HostRead;
    return testing.allocator.dupe(f32, p[0..mlx.mlx_array_size(c)]);
}

/// The state as a stored point holds it (`Model.capturePoint`): every row below the position and the small state, as
/// bytes appended to `out`.
fn stateBytes(o: *Ops, m: *hy4_model.Model, out: *std.ArrayList(u8)) !void {
    const a = testing.allocator;
    const rows = try a.alloc([hy4_model.Model.point_arrays]?Arr, m.config.layers);
    defer a.free(rows);
    try m.capturePoint(rows);
    defer hy4_model.Model.freeArrays(rows);
    for (rows) |row| for (row) |maybe| if (maybe) |x| {
        const c = try o.contiguous(x);
        try mlx.check(mlx.mlx_array_eval(c));
        const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
        const p: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint8(c) orelse return error.HostRead);
        try out.appendSlice(a, p[0..n]);
    };
}

const Rig = struct {
    t: Tiny,
    m: *hy4_model.Model,
    hc: pc.HotPrefixCache,
    es: mlx.mlx_stream,
    o: Ops,

    /// A lookup of `prompt` as the scheduler runs it on Hy4's target.
    fn lookup(r: *Rig, prompt: []const u32) !pc.LookupResult {
        var target = try KVCache.initWithConfig(testing.allocator, r.m.config.layers, .engine_default);
        defer target.deinit();
        var off: usize = 0;
        return r.hc.lookupImpl(&target, &off, @import("../arch/prefix.zig").recurrent("hy4flash", r.m), r.es, prompt, false, 0, null, &.{}, null, null, null, false);
    }
};

/// The prefix cache's SSD tier alone, its row segments `span` tokens.
fn diskCache(io: std.Io, base: []const u8, c: *const hy4_config.Config, span: u32) !pc.HotPrefixCache {
    var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 0, 0);
    errdefer hc.deinit();
    hc.ssd_first = true;
    hc.hybrid = true;
    hc.cp_thin = .min_span_recency;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "hy4-points", 0, 128);
    hc.disk.?.ssd_first = true;
    hc.disk.?.cp_thin = .min_span_recency;
    hc.disk.?.ssm_max_per_entry = kv_disk_cache.SSM_DISK_MAX_PER_ENTRY;
    hc.disk.?.armTestSpace(256 << 30, 1 << 40);
    hc.disk.?.enableBackgroundWriter();
    hc.disk.?.points_hold_state = true;
    try prefix.setRowSpan(testing.allocator, c, &hc.disk.?, span, 8);
    hc.arrays_on = mlx.engine.stream();
    hc.disk.?.arrays_on = hc.arrays_on;
    return hc;
}

test "hy4 points: a live point, a stored point and its write-through, a publish, and a restore in place, from a live point, from a RAM entry's point and from SSD ask nothing of a closed pool, and each restore gives a cold prefill's state and next logits (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    var r: Rig = .{ .t = try Tiny.open(), .m = undefined, .hc = undefined, .es = undefined, .o = undefined };
    defer r.t.close();
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    r.es = mlx.engine.stream();
    const scope = mlx.engine.enter(r.es);
    defer scope.end();
    r.o = .{ .s = r.es };
    defer r.o.deinit();
    const ids = try a.alloc(u32, 1024);
    defer a.free(ids);
    for (ids, 0..) |*d, i| d.* = @intCast((i * 7919 + 17) % 120000);
    const reach = 1024;
    const p = 512;
    // Cold: the state at `p` and the logits of the chunk past it and of the decode row after.
    var want_state: std.ArrayList(u8) = .empty;
    defer want_state.deinit(a);
    const want_chunk, const want_step = blk: {
        const ref = try hy4_model.Model.init(a, r.t.config, &r.t.pack, r.es);
        defer ref.deinit();
        ref.req().kv = .engine_default;
        try ref.reserveFor(reach);
        _ = mlx.mlx_array_free(try ref.forwardReadout(ids[0..256], .none));
        _ = mlx.mlx_array_free(try ref.forwardReadout(ids[256..p], .none));
        try stateBytes(&r.o, ref, &want_state);
        const chunk = try logitsOf(&r.o, ref, ids[p .. p + 40]);
        break :blk .{ chunk, try logitsOf(&r.o, ref, ids[p + 40 .. p + 41]) };
    };
    defer a.free(want_chunk);
    defer a.free(want_step);
    r.m = try hy4_model.Model.init(a, r.t.config, &r.t.pack, r.es);
    defer r.m.deinit();
    r.m.req().kv = .engine_default;
    r.hc = try diskCache(std.testing.io, base, &r.t.config, 512);
    defer r.hc.deinit();
    const m = r.m;
    const Check = struct {
        /// The state where a restore left it at `p`, and the chunk and decode row past it, are the cold run's.
        fn same(rr: *Rig, ws: []const u8, wc: []const f32, wst: []const f32, at: []const u32) !void {
            var got: std.ArrayList(u8) = .empty;
            defer got.deinit(testing.allocator);
            try testing.expectEqual(@as(usize, p), rr.m.req().pos);
            try stateBytes(&rr.o, rr.m, &got);
            try testing.expectEqualSlices(u8, ws, got.items);
            const c = try logitsOf(&rr.o, rr.m, at[p .. p + 40]);
            defer testing.allocator.free(c);
            try testing.expectEqualSlices(f32, wc, c);
            const s = try logitsOf(&rr.o, rr.m, at[p + 40 .. p + 41]);
            defer testing.allocator.free(s);
            try testing.expectEqualSlices(f32, wst, s);
        }
    };
    var asks: [8]u64 = @splat(0);
    // The prefill as served: its state reserved at the reach, a stored point at each chunk end, the one on the SSD
    // grid written through.
    try m.reserveFor(reach);
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..256], .none));
    asks[0] = try closedAsks(hy4_model.Model.keepLivePoint, .{m});
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[256..p], .none));
    var kv = try KVCache.initWithConfig(a, m.config.layers, .engine_default);
    defer kv.deinit();
    kv.step = p;
    var cps: [1]@import("../ssm_cache.zig").SSMCheckpoint = undefined;
    const Capture = struct {
        fn run(mm: *hy4_model.Model, out: *@import("../ssm_cache.zig").SSMCheckpoint) !void {
            out.* = try prefix.captureKept(testing.allocator, mm, p);
        }
        fn through(rr: *Rig, k: *const KVCache, prompt: []const u32, list: []@import("../ssm_cache.zig").SSMCheckpoint) !void {
            prefix.writeThrough(&rr.hc, rr.m, k, prompt, false, .{}, null, list, rr.es);
            rr.hc.disk.?.drainWriter();
        }
        fn publish(rr: *Rig, tokens: []const u32) !void {
            prefix.publishLive(&rr.hc, rr.m, tokens, false, .{});
        }
    };
    asks[1] = try closedAsks(Capture.run, .{ m, &cps[0] });
    defer cps[0].deinit(a);
    asks[2] = try closedAsks(Capture.through, .{ &r, &kv, ids[0..p], &cps });
    // A point the RAM tier keeps whole (`capture`): taken with the pool open, as the RAM entry holds it.
    var ram_cp = try prefix.capture(a, m, p);
    defer ram_cp.deinit(a);
    try Capture.publish(&r, ids[0..p]);
    const Restore = struct {
        fn live(rr: *Rig, prompt: []const u32, want: usize) !void {
            const res = try rr.lookup(prompt);
            try testing.expectEqual(want, res.matched);
        }
        fn ram(rr: *Rig, cp: *const @import("../ssm_cache.zig").SSMCheckpoint, prompt: []const u32) !void {
            try prefix.restore(rr.m, cp, prompt);
        }
    };
    // In place: the state stands at the record's end.
    asks[4] = try closedAsks(Restore.live, .{ &r, ids[0 .. p + 41], p });
    try Check.same(&r, want_state.items, want_chunk, want_step, ids);
    // The request's publish keeps a point behind its reply; the next turn rolls back to the record's point at `p`.
    asks[3] = try closedAsks(Capture.publish, .{ &r, ids[0 .. p + 41] });
    asks[5] = try closedAsks(Restore.live, .{ &r, ids[0 .. p + 41], p });
    try Check.same(&r, want_state.items, want_chunk, want_step, ids);
    // From the RAM entry's point, over a state run past it.
    asks[6] = try closedAsks(Restore.ram, .{ &r, &ram_cp, ids[0 .. p + 41] });
    try Check.same(&r, want_state.items, want_chunk, want_step, ids);
    // From SSD, over a state run past it: the point the tier reads back (its own arrays, `kv_disk_cache`), installed.
    const d = &r.hc.disk.?;
    const hm = d.bestHybridMatch(ids[0 .. p + 41], false, .engine_default, p + 41, .{}) orelse return error.TestNoDiskPoint;
    try testing.expectEqual(@as(u32, p), hm.cp);
    var target = try KVCache.initWithConfig(a, m.config.layers, .engine_default);
    defer target.deinit();
    var disk_cp = try d.restoreCheckpointInto(&target, hm.idx, hm.cp, 4 * m.config.layers, m.s);
    defer disk_cp.deinit(d.allocator);
    asks[7] = try closedAsks(Restore.ram, .{ &r, &disk_cp, ids[0 .. p + 41] });
    try Check.same(&r, want_state.items, want_chunk, want_step, ids);
    errdefer std.debug.print("[hy4-points] closed asks: live point {d}, stored point {d}, write-through {d}, publish {d}; restore in place {d}, from a live point {d}, from RAM {d}, from SSD {d}\n", .{ asks[0], asks[1], asks[2], asks[3], asks[4], asks[5], asks[6], asks[7] });
    try testing.expectEqualSlices(u64, &@as([8]u64, @splat(0)), &asks);
}

test "hy4 bill: the Points slots a request's state reserves hold exactly pointsBytes, with the hidden state where it drafts, at 8 and 16 bits; a take past them is Hy4PointsFull (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const B = @import("hy4_ledger.zig");
    var t = try Tiny.open();
    defer t.close();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    for ([_]u64{ 8, 16 }) |kv| for ([_]bool{ false, true }) |drafts| {
        const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, es);
        defer m.deinit();
        m.req().kv = if (kv == 8) .engine_default else .dense;
        m.keep_hidden = drafts;
        try m.reserveFor(64);
        const held = try prefix.pointsHeldBytes(m);
        errdefer std.debug.print("[hy4-points] kv{d} drafts={}: the slots hold {d} B, billed {d} B\n", .{ kv, drafts, held, B.pointsBytes(&t.config, kv, drafts) });
        try testing.expectEqual(B.pointsBytes(&t.config, kv, drafts), held);
        for (0..hy4_model.Model.point_slots) |_| _ = try m.point();
        try testing.expectError(error.Hy4PointsFull, m.point());
    };
}

/// The bytes of a stored point's small state, appended to `out`.
fn pointBytes(o: *Ops, cp: *const @import("../ssm_cache.zig").SSMCheckpoint, out: *std.ArrayList(u8)) !void {
    for (cp.layers) |l| for ([_]Arr{ l.conv_state, l.ssm_state }) |x| {
        const c = try o.contiguous(x);
        try mlx.check(mlx.mlx_array_eval(c));
        const p: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint8(c) orelse return error.HostRead);
        try out.appendSlice(testing.allocator, p[0 .. mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c)]);
    };
}

test "hy4 points: a live point taken while a stored point still shares a free slot's blocks takes another slot, asks nothing of a closed pool and leaves the stored point as it was (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var t = try Tiny.open();
    defer t.close();
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const m = try hy4_model.Model.init(a, t.config, &t.pack, es);
    defer m.deinit();
    m.req().kv = .engine_default;
    var ids: [300]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast((i * 7919 + 17) % 120000);
    try m.reserveFor(ids.len);
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..256], .none));
    // A stored point over the live one's slot, the live one let go: the slot is free while the stored point reads it.
    var cp = try prefix.captureKept(a, m, 256);
    defer cp.deinit(a);
    const shared = m.req().live_points.items[0].slot;
    m.dropLive();
    var held: std.ArrayList(u8) = .empty;
    defer held.deinit(a);
    try pointBytes(&o, &cp, &held);
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[256..270], .none));
    const asks = try closedAsks(hy4_model.Model.keepLivePoint, .{m});
    errdefer std.debug.print("[hy4-points] the live point asked the closed pool {d} times\n", .{asks});
    try testing.expectEqual(@as(u64, 0), asks);
    try testing.expect(m.req().live_points.items[0].slot != shared);
    var after: std.ArrayList(u8) = .empty;
    defer after.deinit(a);
    try pointBytes(&o, &cp, &after);
    try testing.expectEqualSlices(u8, held.items, after.items);
}

test "hy4 points: a request's state reserved before its lookup at the request's end takes a restore on a fresh model with no ask of the pool, and gives a cold prefill's state (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var t = try Tiny.open();
    defer t.close();
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    var ids: [600]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast((i * 7919 + 17) % 120000);
    const p = 512;
    const ref = try hy4_model.Model.init(a, t.config, &t.pack, es);
    defer ref.deinit();
    ref.req().kv = .engine_default;
    try ref.reserveFor(ids.len);
    _ = mlx.mlx_array_free(try ref.forwardReadout(ids[0..p], .none));
    var want: std.ArrayList(u8) = .empty;
    defer want.deinit(a);
    try stateBytes(&o, ref, &want);
    var cp = try prefix.capture(a, ref, p);
    defer cp.deinit(a);
    // A fresh model, as a fresh load leaves it; the request of the prompt and 40 tokens is reserved before its lookup.
    const m = try hy4_model.Model.init(a, t.config, &t.pack, es);
    defer m.deinit();
    try prefix.reserveRequest(m, p + 48, 40, 0);
    try testing.expectEqual(@as(usize, p + 48 + 40), m.req().reach);
    const before = core.stats().asks;
    try prefix.restore(m, &cp, ids[0..p]);
    const asks = core.stats().asks - before;
    errdefer std.debug.print("[hy4-points] the restore asked the pool {d} times\n", .{asks});
    try testing.expectEqual(@as(u64, 0), asks);
    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(a);
    try stateBytes(&o, m, &got);
    try testing.expectEqualSlices(u8, want.items, got.items);
}

test "hy4 points: a restore from SSD reads the point's files straight into the request's state, at a point on the row-segment grid and one past it, asks nothing of a closed pool and gives a cold prefill's state and next logits (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    var r: Rig = .{ .t = try Tiny.open(), .m = undefined, .hc = undefined, .es = undefined, .o = undefined };
    defer r.t.close();
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    r.es = mlx.engine.stream();
    const scope = mlx.engine.enter(r.es);
    defer scope.end();
    r.o = .{ .s = r.es };
    defer r.o.deinit();
    const ids = try a.alloc(u32, 1024);
    defer a.free(ids);
    for (ids, 0..) |*d, i| d.* = @intCast((i * 7919 + 17) % 120000);
    const reach = 1024;
    // The prefill's chunks; a point at each of their ends: 512 on the 256-token segment grid, 602 past it, its rows
    // past the segments in its own file and an open group of two rows.
    const ends = [_]usize{ 256, 512, 602 };
    const points = [_]usize{ 512, 602 };
    const Cold = struct { state: std.ArrayList(u8) = .empty, chunk: []f32 = &.{}, step: []f32 = &.{} };
    var cold: [points.len]Cold = @splat(.{});
    defer for (&cold) |*c| {
        c.state.deinit(a);
        a.free(c.chunk);
        a.free(c.step);
    };
    for (points, &cold) |p, *c| {
        const ref = try hy4_model.Model.init(a, r.t.config, &r.t.pack, r.es);
        defer ref.deinit();
        ref.req().kv = .engine_default;
        try ref.reserveFor(reach);
        var from: usize = 0;
        for (ends) |e| if (e <= p) {
            _ = mlx.mlx_array_free(try ref.forwardReadout(ids[from..e], .none));
            from = e;
        };
        try stateBytes(&r.o, ref, &c.state);
        c.chunk = try logitsOf(&r.o, ref, ids[p .. p + 40]);
        c.step = try logitsOf(&r.o, ref, ids[p + 40 .. p + 41]);
    }
    r.m = try hy4_model.Model.init(a, r.t.config, &r.t.pack, r.es);
    defer r.m.deinit();
    const m = r.m;
    m.req().kv = .engine_default;
    r.hc = try diskCache(std.testing.io, base, &r.t.config, 256);
    defer r.hc.deinit();
    const d = &r.hc.disk.?;
    // The point on the grid is written through with its segments; the one past it reaches SSD with the live record.
    try m.reserveFor(reach);
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..256], .none));
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[256..512], .none));
    {
        var kv = try KVCache.initWithConfig(a, m.config.layers, .engine_default);
        defer kv.deinit();
        kv.step = 512;
        var cps: [1]@import("../ssm_cache.zig").SSMCheckpoint = .{try prefix.captureKept(a, m, 512)};
        defer cps[0].deinit(a);
        prefix.writeThrough(&r.hc, m, &kv, ids[0..512], false, .{}, null, &cps, r.es);
        d.drainWriter();
    }
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[512..602], .none));
    try m.keepLivePoint();
    prefix.publishLive(&r.hc, m, ids[0..602], false, .{});
    prefix.flushLive(&r.hc, m, "test");
    d.drainWriter();
    const Restore = struct {
        fn disk(rr: *Rig, prompt: []const u32, want: usize) !void {
            var target = try KVCache.initWithConfig(testing.allocator, rr.m.config.layers, .engine_default);
            defer target.deinit();
            const res = prefix.restoreDisk(&rr.hc, &rr.hc.disk.?, rr.m, &target, 0, prompt, false, .{}, null, rr.es) orelse return error.TestNoDiskRestore;
            try testing.expectEqual(want, res.matched);
        }
    };
    var asks: [points.len]u64 = @splat(0);
    for (points, &cold, &asks) |p, *c, *n| {
        // Over a state run past the point.
        _ = mlx.mlx_array_free(try m.forwardReadout(ids[m.req().pos .. m.req().pos + 30], .none));
        // Nothing beside the state, wide phases included: the state was reserved at the reach (D61).
        const made = core.stats().asks;
        n.* = try closedAsks(Restore.disk, .{ &r, ids[0 .. p + 41], p });
        try testing.expectEqual(made, core.stats().asks);
        try testing.expectEqual(p, m.req().pos);
        var got: std.ArrayList(u8) = .empty;
        defer got.deinit(a);
        try stateBytes(&r.o, m, &got);
        try testing.expectEqualSlices(u8, c.state.items, got.items);
        const chunk = try logitsOf(&r.o, m, ids[p .. p + 40]);
        defer a.free(chunk);
        try testing.expectEqualSlices(f32, c.chunk, chunk);
        const step = try logitsOf(&r.o, m, ids[p + 40 .. p + 41]);
        defer a.free(step);
        try testing.expectEqualSlices(f32, c.step, step);
    }
    errdefer std.debug.print("[hy4-points] a restore from SSD asked the closed pool {d} times at 512, {d} at 602\n", .{ asks[0], asks[1] });
    try testing.expectEqualSlices(u64, &@as([points.len]u64, @splat(0)), &asks);
}

/// A tiny pack's model on the engine's stream, its arrays and ids as the cells above set them.
const Session = struct {
    r: Rig,
    was: bool,
    scope: @typeInfo(@TypeOf(mlx.engine.enter)).@"fn".return_type.?,
    ids: []u32,

    fn begin(s: *Session, ids_len: usize) !void {
        const a = testing.allocator;
        s.r = .{ .t = try Tiny.open(), .m = undefined, .hc = undefined, .es = undefined, .o = undefined };
        errdefer s.r.t.close();
        fwd.products = .bf16;
        s.was = mlx.engine.enabled;
        mlx.engine.enabled = true;
        errdefer @import("../test_support.zig").restoreEngine(s.was);
        s.r.es = mlx.engine.stream();
        s.scope = mlx.engine.enter(s.r.es);
        s.r.o = .{ .s = s.r.es };
        s.ids = try a.alloc(u32, ids_len);
        for (s.ids, 0..) |*d, i| d.* = @intCast((i * 7919 + 17) % 120000);
        s.r.m = try hy4_model.Model.init(a, s.r.t.config, &s.r.t.pack, s.r.es);
        s.r.m.req().kv = .engine_default;
    }

    fn end(s: *Session) void {
        s.r.m.deinit();
        testing.allocator.free(s.ids);
        s.r.o.deinit();
        s.scope.end();
        @import("../test_support.zig").restoreEngine(s.was);
        s.r.t.close();
    }

    /// Runs `ids[from..to]` through the model and keeps a live point at its end.
    fn prefillKeep(s: *Session, from: usize, to: usize) !void {
        _ = mlx.mlx_array_free(try s.r.m.forwardReadout(s.ids[from..to], .none));
        try s.r.m.keepLivePoint();
    }
};

const k_img1: u64 = 0x1111_2222_3333_4444;
const k_img2: u64 = 0x5555_6666_7777_8888;

test "hy4 points: a write-through or end point that stops before writing leaves no media chain for the next consumer (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    var s: Session = undefined;
    try s.begin(64);
    defer s.end();
    var hc = pc.HotPrefixCache.init(testing.allocator, 4);
    defer hc.deinit();
    var kv = try KVCache.initWithConfig(testing.allocator, s.r.m.config.layers, .engine_default);
    defer kv.deinit();
    const chain = [_]pc.MediaSpan{.{ .start = 8, .key = k_img1 }};
    // No SSD tier: nothing is written, whatever the chain.
    hc.commit_media_chain = &chain;
    prefix.writeThrough(&hc, s.r.m, &kv, s.ids[0..16], false, .{}, null, &.{}, s.r.es);
    try testing.expectEqual(@as(usize, 0), hc.commit_media_chain.len);
    hc.commit_media_chain = &chain;
    prefix.writeEndPoint(&hc, s.r.m, &kv, s.ids[0..16], false, .{}, null);
    try testing.expectEqual(@as(usize, 0), hc.commit_media_chain.len);
}

test "hy4 points: a restore from SSD reports disk_restore, so a turn cut before its second image links the row segments it restored from the two-image record (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    var s: Session = undefined;
    try s.begin(1024);
    defer s.end();
    const m = s.r.m;
    s.r.hc = try diskCache(std.testing.io, base, &s.r.t.config, 256);
    defer s.r.hc.deinit();
    const hc = &s.r.hc;
    const d = &hc.disk.?;
    const img1_at: usize = 40;
    const chain = [_]pc.MediaSpan{ .{ .start = img1_at, .key = k_img1 }, .{ .start = 540, .key = k_img2 } };

    // Turn 1: images at 40 and 540, points at 256, 512 and 700, the record flushed to SSD under both images' key.
    try m.reserveFor(1024);
    try s.prefillKeep(0, 256);
    try s.prefillKeep(256, 512);
    _ = mlx.mlx_array_free(try m.forwardReadout(s.ids[512..700], .none));
    hc.commit_media_chain = &chain;
    prefix.publishLive(hc, m, s.ids[0..700], false, .{ .key = k_img2, .start = img1_at });
    prefix.flushLive(hc, m, "test");
    d.drainWriter();
    try testing.expectEqual(@as(usize, 1), d.entryCount());
    const donor_id = d.entries.items[0].id;

    // Turn 2 repeats the first 520 tokens: it restores the point at 512 from SSD.
    _ = mlx.mlx_array_free(try m.forwardReadout(s.ids[700..730], .none));
    var target = try KVCache.initWithConfig(testing.allocator, m.config.layers, .engine_default);
    defer target.deinit();
    const res = prefix.restoreDisk(hc, d, m, &target, 0, s.ids[0..521], false, .{}, null, s.r.es) orelse return error.TestNoDiskRestore;
    try testing.expectEqual(@as(usize, 512), res.matched);
    const dr = res.disk_restore orelse return error.TestExpectedDiskRestore;
    try testing.expectEqual(donor_id, dr.id);
    try testing.expectEqual(@as(u32, 512), dr.rows);
    try testing.expectEqual(dr, prefix.liveOf(hc).restored_disk orelse return error.TestExpectedRestoredDisk);

    // It ends at 520, before the second image: its stamp is image 1's, which the record does not carry, so only the
    // restored rows say its first two segments are the record's.
    try s.prefillKeep(512, 520);
    hc.commit_media_chain = &chain;
    prefix.publishLive(hc, m, s.ids[0..520], false, .{ .key = k_img2, .start = img1_at });
    prefix.flushLive(hc, m, "test");
    d.drainWriter();
    try testing.expectEqual(@as(usize, 2), d.entryCount());
    try testing.expectEqual(@as(u32, 2), d.entries.items[1].inherited_grown);
}

test "hy4 points: a rollback before the second image restamps the published record to the images it still covers, and its flush stores that stamp (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    var s: Session = undefined;
    try s.begin(1024);
    defer s.end();
    const m = s.r.m;
    s.r.hc = try diskCache(std.testing.io, base, &s.r.t.config, 256);
    defer s.r.hc.deinit();
    const hc = &s.r.hc;
    const live = prefix.liveOf(hc);
    const img1_at: usize = 40;
    const chain = [_]pc.MediaSpan{ .{ .start = img1_at, .key = k_img1 }, .{ .start = 580, .key = k_img2 } };

    try m.reserveFor(1024);
    try s.prefillKeep(0, 540);
    _ = mlx.mlx_array_free(try m.forwardReadout(s.ids[540..660], .none));
    hc.commit_media_chain = &chain;
    prefix.publishLive(hc, m, s.ids[0..660], false, .{ .key = k_img2, .start = img1_at });
    try testing.expectEqual(k_img2, live.vision_key);

    // The next prompt shares 600 tokens: the state rolls back to its point at 540, before the second image.
    var target = try KVCache.initWithConfig(testing.allocator, m.config.layers, .engine_default);
    defer target.deinit();
    var moe: usize = 0;
    const res = prefix.rollbackLive(hc, m, &target, &moe, s.r.es, s.ids[0..600], false, .{}, null) orelse return error.TestExpectedRollback;
    try testing.expectEqual(@as(usize, 540), res.matched);
    try testing.expectEqual(@as(usize, 540), live.tokens.items.len);
    try testing.expectEqual(k_img1, live.vision_key);
    prefix.flushLive(hc, m, "handover");
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expectEqual(k_img1, hc.disk.?.entries.items[0].vision_key);
}
