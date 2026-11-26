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
