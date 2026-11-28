//! The Sushi Cooler contract's manifest cells: what a load's pipeline manifest records of its served warm, under
//! which pack, shape and settings a later load takes it, and what a request compiled late leaves for the next load.
//! The runner and the shared helpers are in `cooler_contract_test.zig`.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const arch_handle = @import("arch/handle.zig");
const test_support = @import("test_support.zig");
const model_load = @import("model_load.zig");
const arch_sched = @import("arch/sched.zig");
const io = std.Io.Threaded.global_single_threaded.io();
const contract = @import("cooler_contract_test.zig");
const Load = contract.Load;
const Served = contract.Served;
const eachArch = contract.eachArch;

fn isLedgerArch(comptime C: type, dir: []const u8) !bool {
    var cfg = try contract.prepareConfig(C, dir, true);
    defer cfg.deinit(testing.allocator);
    return @import("bills.zig").ledgerArch(&cfg);
}

test "cooler contract: a pipeline a Cooler request compiles late is written to the load's manifest, for the next load to make ahead" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            if (try isLedgerArch(C, dir)) return error.SkipZigTest;
            const shared = @import("arch/cooler_pipelines.zig");
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            const key = shared.writtenKey() orelse return error.NoManifestWritten;
            {
                // A kernel no warm makes: a unary op over int16, made by a cell's own ops, which a served request on
                // the ledger never asks of its closed pool.
                const turn = test_support.openTurn(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                const vals = [_]i16{ -3, 4, -5, 6 };
                const x = mlx.mlx_array_new_data(&vals, &[_]c_int{4}, 1, .int16);
                defer _ = mlx.mlx_array_free(x);
                var y = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(y);
                try mlx.check(mlx.mlx_abs(&y, x, mlx.engine.stream()));
                try mlx.check(mlx.mlx_array_eval(y));
            }
            var found = try @import("pipeline_manifest.zig").lookup(testing.allocator, io, tmp.dir, key);
            defer found.deinit();
            const recipes = switch (found) {
                .same => |l| l.recipes,
                else => return error.NoManifestWritten,
            };
            for (recipes) |r| if (std.mem.indexOf(u8, r.function, "Abs") != null and std.mem.indexOf(u8, r.function, "int16") != null) return;
            return error.LatePipelineNotWritten;
        }
    }.run);
}

test "cooler contract: a Cooler load whose manifest holds a served warm of another shape or another scope still runs it in full" {
    try contract.qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, _: []const u32) !void {
            const shared = @import("arch/cooler_pipelines.zig");
            const cooler_warm = @import("cooler_warm.zig");
            const dir_len = @import("pipeline_manifest.zig").Key.dir_len;
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            var runs: [2]u32 = undefined;
            for (&runs, 0..) |*r, i| {
                const before = cooler_warm.runs_for_test;
                var cooler: Load = undefined;
                try cooler.open(Q, dir, true);
                defer _ = cooler.close();
                r.* = cooler_warm.runs_for_test - before;
                if (i == 0) {
                    // The manifest of another scope: the build stays, what selects the pack's kernels does not.
                    var it = tmp.dir.iterate();
                    var from: [dir_len]u8 = undefined;
                    while (try it.next(io)) |e| if (e.kind == .directory and e.name.len == dir_len) @memcpy(&from, e.name);
                    var to = from;
                    @memset(to[0..16], '0');
                    try tmp.dir.rename(&from, tmp.dir, &to, io);
                }
            }
            try testing.expectEqual(@as(u32, 1), runs[0]);
            try testing.expectEqual(@as(u32, 1), runs[1]);
        }
    }.run);
}

/// The served warms that ran their requests in one served load of `dir`, the warm's chunk pinned to `chunk` (0: not
/// pinned).
fn warmRuns(comptime C: type, dir: []const u8, chunk: u32) !u32 {
    const cooler_warm = @import("cooler_warm.zig");
    const before = cooler_warm.runs_for_test;
    var cooler: Load = undefined;
    try cooler.prepare(C, dir, true);
    const settings = @import("settings_resolve.zig");
    const was = settings.launch;
    if (chunk != 0) settings.launch = .{ .prefill_chunk_max = chunk, .prefill_chunk_min = chunk };
    defer settings.launch = was;
    try cooler.loadServed(dir);
    defer _ = cooler.close();
    return cooler_warm.runs_for_test - before;
}

test "cooler contract: a served warm this build ran whole is skipped by a later load of its shape, whatever loaded in between" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            if (try isLedgerArch(C, dir)) return error.SkipZigTest;
            const shared = @import("arch/cooler_pipelines.zig");
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            mlx.engine.pipeline_registry.resetForTest();
            // A at the default shape, B at a narrower chunk (another marker, the same manifest), A again.
            const first = try warmRuns(C, dir, 0);
            const between = try warmRuns(C, dir, 2048);
            const again = try warmRuns(C, dir, 0);
            errdefer std.debug.print("served warms that ran: first={d} between={d} again={d}\n", .{ first, between, again });
            try testing.expectEqual(@as(u32, 1), first);
            try testing.expectEqual(@as(u32, 1), between);
            try testing.expectEqual(@as(u32, 0), again);
        }
    }.run);
}

test "cooler contract: a load that fails after its manifest settled leaves the next load a manifest it finds, trusts and writes" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            if (try isLedgerArch(C, dir)) return error.SkipZigTest;
            const shared = @import("arch/cooler_pipelines.zig");
            const cooler_warm = @import("cooler_warm.zig");
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            mlx.engine.pipeline_registry.resetForTest();
            const first = try warmRuns(C, dir, 0);
            if ((cooler_warm.longest_for_test orelse return error.TestNoWarmReading).asked == 0) return;
            // Another checkpoint stride is a warm this manifest never held on every arch (a static bill fixes its own
            // chunk), and a bill it exceeds fails the load in it.
            {
                const state = @import("server_state.zig");
                const stride_was = state.ssm_checkpoint_stride;
                defer state.ssm_checkpoint_stride = stride_was;
                state.ssm_checkpoint_stride = stride_was + 1;
                cooler_warm.bill_for_test = 0;
                defer cooler_warm.bill_for_test = null;
                try testing.expectError(error.WarmOverBill, warmRuns(C, dir, 2048));
            }
            var msg: [512]u8 = undefined;
            _ = mlx.takeError(&msg);
            // The next load of A's shape finds the manifest and skips the warm, and its own write lands.
            const before = cooler_warm.runs_for_test;
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            const again = cooler_warm.runs_for_test - before;
            errdefer std.debug.print("served warms that ran: first={d} again={d}\n", .{ first, again });
            try testing.expectEqual(@as(u32, 1), first);
            try testing.expectEqual(@as(u32, 0), again);
            const key = shared.writtenKey() orelse return error.NoManifestWritten;
            var found = try @import("pipeline_manifest.zig").lookup(testing.allocator, io, tmp.dir, key);
            defer found.deinit();
            try testing.expect(found == .same);
        }
    }.run);
}

/// Whether the manifest of `key` holds the marker recipe `name`.
fn holdsMarker(dir: std.Io.Dir, key: @import("pipeline_manifest.zig").Key, name: []const u8) !bool {
    var found = try @import("pipeline_manifest.zig").lookup(testing.allocator, io, dir, key);
    defer found.deinit();
    const recipes = switch (found) {
        .same => |l| l.recipes,
        else => return error.NoManifest,
    };
    for (recipes) |r| if (r.kind == .marker and std.mem.eql(u8, r.key, name)) return true;
    return false;
}

test "cooler contract: a marker recorded while a load's weights loaded lands in its manifest, and one recorded after unload writes none" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            if (try isLedgerArch(C, dir)) return error.SkipZigTest;
            const shared = @import("arch/cooler_pipelines.zig");
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            mlx.engine.pipeline_registry.resetForTest();
            mlx.engine.pipeline_registry.recordMarker("vision tower", "tower/this-load");
            var key: @import("pipeline_manifest.zig").Key = undefined;
            {
                var cooler: Load = undefined;
                try cooler.open(C, dir, true);
                defer _ = cooler.close();
                key = shared.writtenKey() orelse return error.NoManifestWritten;
            }
            try testing.expect(try holdsMarker(tmp.dir, key, "tower/this-load"));
            // The next load's vision tower warms while its weights load, before its manifest is opened.
            mlx.engine.pipeline_registry.recordMarker("vision tower", "tower/next-load");
            shared.rewriteLate();
            try testing.expect(!try holdsMarker(tmp.dir, key, "tower/next-load"));
        }
    }.run);
}

test "cooler contract: a manifest rewritten after its root directory was removed and made again lands in the new one" {
    const shared = @import("arch/cooler_pipelines.zig");
    const pm = @import("pipeline_manifest.zig");
    try mlx.requireGpu();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    try pm.setRootFromHome(buf[0..try tmp.dir.realPath(io, &buf)]);
    defer pm.clearRootForTest();
    defer shared.forget();
    mlx.engine.pipeline_registry.resetForTest();
    var plan = shared.beginScope(testing.allocator, io, @as([16]u8, "aaaaaaaaaaaaaaaa".*));
    defer plan.deinit();
    _ = plan.settle();
    plan.store();
    const key = shared.writtenKey() orelse return error.NoManifestWritten;
    // Another process or a cleanup removes the root; the next load makes it again.
    try tmp.dir.deleteTree(io, ".sushi");
    mlx.engine.pipeline_registry.record(.{ .kind = .source, .function = "late_kernel", .text = "kernel void late_kernel() {}\n" });
    shared.rewriteLate();
    var root = tmp.dir.openDir(io, ".sushi/cache/pipelines", .{ .iterate = true }) catch |e| {
        std.debug.print("the root was not made again: {s}\n", .{@errorName(e)});
        return e;
    };
    defer root.close(io);
    var found = try pm.lookup(testing.allocator, io, root, key);
    defer found.deinit();
    try testing.expect(found == .same);
}

test "cooler contract: a plan's kept shapes land in a root removed and made again" {
    const shared = @import("arch/cooler_pipelines.zig");
    const pm = @import("pipeline_manifest.zig");
    try mlx.requireGpu();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    try pm.setRootFromHome(buf[0..try tmp.dir.realPath(io, &buf)]);
    defer pm.clearRootForTest();
    defer shared.forget();
    mlx.engine.pipeline_registry.resetForTest();
    var plan = shared.beginScope(testing.allocator, io, @as([16]u8, "aaaaaaaaaaaaaaaa".*));
    defer plan.deinit();
    _ = plan.settle();
    const cache = plan.shapeCache() orelse return error.NoShapeCache;
    try cache.keep(testing.allocator, .{ .phase = 1 });
    // Another process or a cleanup removes the root; the load's later misses are kept in the one made again.
    try tmp.dir.deleteTree(io, ".sushi");
    cache.keep(testing.allocator, .{ .phase = 2 }) catch |e| {
        std.debug.print("a shape was not kept in the root made again: {s}\n", .{@errorName(e)});
        return e;
    };
    const kept = try cache.read(testing.allocator);
    defer testing.allocator.free(kept);
    try testing.expectEqual(@as(usize, 1), kept.len);
    try testing.expectEqual(@as(u32, 2), kept[0].phase);
}

/// Two DeepSeek packs whose scopes differ (the tile order does) and whose served warm has one shape.
fn twoPacks() ?struct { a: []const u8, b: []const u8 } {
    const a = std.mem.span(std.c.getenv("DSV41_STQ_TINY_PM") orelse return null);
    const b = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return null);
    return .{ .a = a, .b = b };
}

/// The process the two-pack tests share: a manifest root of its own, the engine on.
const TwoPacks = struct {
    tmp: testing.TmpDir,
    was_enabled: bool,
    packs: @TypeOf(twoPacks().?),

    fn begin(self: *TwoPacks) !void {
        try mlx.requireGpu();
        self.packs = twoPacks() orelse return error.SkipZigTest;
        self.was_enabled = mlx.engine.enabled;
        mlx.engine.enabled = true;
        self.tmp = testing.tmpDir(.{ .iterate = true });
        @import("arch/cooler_pipelines.zig").root_hook = self.tmp.dir;
        mlx.engine.pipeline_registry.resetForTest();
    }

    fn end(self: *TwoPacks) void {
        const shared = @import("arch/cooler_pipelines.zig");
        shared.root_hook = null;
        shared.forget();
        self.tmp.cleanup();
        test_support.restoreEngine(self.was_enabled);
    }

    /// A new process: it knows the manifests' files and nothing the last one made.
    fn restart(_: *TwoPacks) void {
        mlx.engine.pipeline_registry.resetForTest();
    }
};

test "cooler contract: a pack's served warm is skipped by its next load in the process, whatever pack of another scope loaded in between" {
    const C = @import("dsv41/arch.zig").cooler_contract;
    var two: TwoPacks = undefined;
    try two.begin();
    defer two.end();
    const a_first = try warmRuns(C, two.packs.a, 0);
    const b_between = try warmRuns(C, two.packs.b, 0);
    const a_again = try warmRuns(C, two.packs.a, 0);
    errdefer std.debug.print("served warms that ran: a_first={d} b_between={d} a_again={d}\n", .{ a_first, b_between, a_again });
    try testing.expectEqual(@as(u32, 1), a_first);
    try testing.expectEqual(@as(u32, 1), b_between);
    try testing.expectEqual(@as(u32, 0), a_again);
}

test "cooler contract: a served-warm marker one pack's warm recorded is not written into another pack's manifest" {
    const C = @import("dsv41/arch.zig").cooler_contract;
    const cooler_warm = @import("cooler_warm.zig");
    var two: TwoPacks = undefined;
    try two.begin();
    defer two.end();
    // Process 1: A at the default shape, then B at a narrower chunk: B's warm never ran at the default shape.
    const a_first = try warmRuns(C, two.packs.a, 0);
    const b_narrow = try warmRuns(C, two.packs.b, 2048);
    // Process 2: B at the default shape enters its warm instead of skipping (a tiny bill fails it fast).
    two.restart();
    cooler_warm.bill_for_test = 1 << 20;
    defer cooler_warm.bill_for_test = null;
    try testing.expectError(error.WarmOverBill, warmRuns(C, two.packs.b, 0));
    var msg: [512]u8 = undefined;
    _ = mlx.takeError(&msg);
    errdefer std.debug.print("served warms that ran: a_first={d} b_narrow={d}\n", .{ a_first, b_narrow });
    try testing.expectEqual(@as(u32, 1), a_first);
    try testing.expectEqual(@as(u32, 1), b_narrow);
}

test "cooler contract: two packs whose served warms have one shape and one bill each measure their widest request against it" {
    const C = @import("dsv41/arch.zig").cooler_contract;
    const cooler_warm = @import("cooler_warm.zig");
    var two: TwoPacks = undefined;
    try two.begin();
    defer two.end();
    // Process 1: B at a narrower chunk, then A at the shape both share.
    const b_narrow = try warmRuns(C, two.packs.b, 2048);
    const a_first = try warmRuns(C, two.packs.a, 0);
    try testing.expectEqual(@as(u32, 1), b_narrow);
    try testing.expectEqual(@as(u32, 1), a_first);
    // Process 2: A skips its warm, as its own manifest says. B, whose manifest holds no warm of this shape, runs it, and
    // the measure of its widest request runs with it: a bill it exceeds is refused by name.
    two.restart();
    try testing.expectEqual(@as(u32, 0), try warmRuns(C, two.packs.a, 0));
    cooler_warm.bill_for_test = 1 << 20;
    defer cooler_warm.bill_for_test = null;
    try testing.expectError(error.WarmOverBill, warmRuns(C, two.packs.b, 0));
    var msg: [512]u8 = undefined;
    _ = mlx.takeError(&msg);
}

const mtp_settings = .{ LookupSetting, MaxMtpCtxSetting, MtpMaxDepthSetting };

/// Every MTP setting a warm reads at its `written` value, or at `loaded` for the one `flipped` names.
fn setMtpSettings(comptime flipped: ?type) void {
    inline for (mtp_settings) |Setting| Setting.set(if (flipped == Setting) Setting.loaded else Setting.written);
}

/// The files under `src`, copied into `dst`: a manifest's directory as one process left it.
fn copyTree(src: std.Io.Dir, dst: std.Io.Dir) !void {
    var walker = try src.walk(testing.allocator);
    defer walker.deinit();
    while (try walker.next(io)) |e| switch (e.kind) {
        .directory => try dst.createDirPath(io, e.path),
        .file => {
            const bytes = try src.readFileAlloc(io, e.path, testing.allocator, .limited(1 << 30));
            defer testing.allocator.free(bytes);
            try dst.writeFile(io, .{ .sub_path = e.path, .data = bytes });
        },
        else => {},
    };
}

/// A process that loads from a copy of `written`, the manifest a process wrote under every setting's `written` value,
/// with `flipped`'s setting at `loaded` (none: the control); when skipped (`runs == 0`), runs a full warm from no
/// manifest to verify nothing was left late.
fn settingProbe(comptime C: type, dir: []const u8, written: std.Io.Dir, comptime flipped: ?type) !struct { runs: u32, late: u64, drafts: bool } {
    const shared = @import("arch/cooler_pipelines.zig");
    const cooler_warm = @import("cooler_warm.zig");
    const reg = mlx.engine.pipeline_registry;
    var copy = testing.tmpDir(.{ .iterate = true });
    defer copy.cleanup();
    try copyTree(written, copy.dir);
    var fresh = testing.tmpDir(.{ .iterate = true });
    defer fresh.cleanup();
    defer shared.root_hook = null;
    defer shared.forget();
    shared.root_hook = copy.dir;
    reg.resetForTest();
    setMtpSettings(flipped);
    const before = cooler_warm.runs_for_test;
    var cooler: Load = undefined;
    try cooler.open(C, dir, true);
    defer _ = cooler.close();
    const runs = cooler_warm.runs_for_test - before;
    const drafts = @import("generate.zig").MtpHeadRef.fromModel(cooler.xfm, cooler.mtpOn(), null) != null;
    var added: u64 = 0;
    if (runs == 0) {
        shared.root_hook = fresh.dir;
        reg.settleMarkers(false);
        const late = mlx.engine.latePipelines();
        mlx.engine.late_names.clear();
        {
            const turn = arch_handle.turnFor(cooler.h, &cooler.config);
            defer turn.end();
            const verify_cap: ?u32 = if (@hasDecl(C, "specTokens")) @import("generate.zig").Generator.mtpVerifyDraftsMax(&cooler.config) else null;
            try model_load.warmLoad(io, testing.allocator, &cooler.config, cooler.xfm, verify_cap, null, cooler.xfm.cache.config);
        }
        added = mlx.engine.latePipelines() - late;
        if (added > 0) std.debug.print("{s} left {d} kernels late:\n{s}\n", .{ if (flipped) |F| @typeName(F) else "the control", added, mlx.engine.late_names.text() });
    }
    return .{ .runs = runs, .late = added, .drafts = drafts };
}

/// `--no-mtp-lookup`: the manifest's process ran with it, the loading one with prompt lookup on.
const LookupSetting = struct {
    const written = true;
    const loaded = false;
    fn get() bool {
        return @import("generate.zig").mtp_lookup_disabled;
    }
    fn set(v: bool) void {
        @import("generate.zig").mtp_lookup_disabled = v;
    }
};

/// `--max-mtp-ctx 24`: the manifest's process ran with it, the loading one with no ceiling.
const MaxMtpCtxSetting = struct {
    const written: u32 = 24;
    const loaded: u32 = 0;
    fn get() u32 {
        return @import("generate.zig").max_mtp_ctx;
    }
    fn set(v: u32) void {
        @import("generate.zig").max_mtp_ctx = v;
    }
};

/// `--mtp-max-depth 2`: the manifest's process ran with it, the loading one with the planner's own cap.
const MtpMaxDepthSetting = struct {
    const written: u32 = 2;
    const loaded: u32 = 0;
    fn get() u32 {
        return @import("mtp_depth_bounds.zig").active.max;
    }
    fn set(v: u32) void {
        @import("mtp_depth_bounds.zig").active.max = v;
    }
};

test "cooler contract: a manifest's served-warm marker written under one MTP setting is not taken under another the warm reads" {
    try contract.qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, _: []const u32) !void {
            const shared = @import("arch/cooler_pipelines.zig");
            const saved = .{ LookupSetting.get(), MaxMtpCtxSetting.get(), MtpMaxDepthSetting.get() };
            defer LookupSetting.set(saved[0]);
            defer MaxMtpCtxSetting.set(saved[1]);
            defer MtpMaxDepthSetting.set(saved[2]);
            // One process writes the manifest under every setting's `written` value; each later process loads a copy.
            var written = testing.tmpDir(.{ .iterate = true });
            defer written.cleanup();
            {
                defer shared.root_hook = null;
                defer shared.forget();
                shared.root_hook = written.dir;
                mlx.engine.pipeline_registry.resetForTest();
                setMtpSettings(null);
                var first: Load = undefined;
                try first.open(Q, dir, true);
                defer _ = first.close();
                // A load with no drafter: its warm reads no MTP setting, so there is nothing for its marker to keep apart.
                if (@import("generate.zig").MtpHeadRef.fromModel(first.xfm, true, null) == null) return error.SkipZigTest;
            }
            // The control: no setting flipped, the manifest's warm skipped and nothing left late.
            const control = try settingProbe(Q, dir, written.dir, null);
            try testing.expectEqual(@as(u32, 0), control.runs);
            try testing.expectEqual(@as(u64, 0), control.late);
            inline for (mtp_settings) |Setting| {
                const flipped = try settingProbe(Q, dir, written.dir, Setting);
                errdefer std.debug.print("{s}: drafts={}; control runs={d} late={d}; flipped runs={d} late={d}\n", .{ @typeName(Setting), flipped.drafts, control.runs, control.late, flipped.runs, flipped.late });
                try testing.expectEqual(control.drafts, flipped.drafts);
                // Only a drafter's rounds read these settings (`cooler_warm.warmMarker`): a load with none takes its
                // marker under any, and the warm's requests made nothing late under it.
                try testing.expectEqual(@as(u32, if (flipped.drafts) 1 else 0), flipped.runs);
                try testing.expectEqual(@as(u64, 0), flipped.late);
            }
        }
    }.run);
}

test "cooler contract: a reload of a pack in its own scope skips its served warm, answers as a warmed load does and leaves its warm's requests no kernel to compile" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const shared = @import("arch/cooler_pipelines.zig");
            const cooler_warm = @import("cooler_warm.zig");
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            var empty = testing.tmpDir(.{ .iterate = true });
            defer empty.cleanup();
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            mlx.engine.pipeline_registry.resetForTest();
            const prompt = ids[0..@min(ids.len, 20)];
            var full: std.ArrayList(u32) = .empty;
            defer full.deinit(a);
            {
                var cooler: Load = undefined;
                const before = cooler_warm.runs_for_test;
                try cooler.open(C, dir, true);
                defer _ = cooler.close();
                try testing.expectEqual(@as(u32, 1), cooler_warm.runs_for_test - before);
                var sv: Served = undefined;
                try sv.init(&cooler, dir);
                defer sv.deinit();
                var from: u32 = 0;
                full = try sv.request(prompt, 12, &from);
            }
            // The reload runs as a fresh process: it knows only the manifest's file and nothing the first made.
            mlx.engine.pipeline_registry.resetForTest();
            var cooler: Load = undefined;
            const before = cooler_warm.runs_for_test;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            try testing.expectEqual(@as(u32, 0), cooler_warm.runs_for_test - before);
            {
                const late = mlx.engine.latePipelines();
                var sv: Served = undefined;
                try sv.init(&cooler, dir);
                defer sv.deinit();
                var from: u32 = 0;
                var skipped = try sv.request(prompt, 12, &from);
                defer skipped.deinit(a);
                try testing.expectEqualSlices(u32, full.items, skipped.items);
                try testing.expectEqual(late, mlx.engine.latePipelines());
            }
            // The skipped warm's own requests, run now: the pipelines the manifest made are all they ask for.
            shared.root_hook = empty.dir;
            mlx.engine.pipeline_registry.settleMarkers(false);
            const late = mlx.engine.latePipelines();
            mlx.engine.late_names.clear();
            {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                const verify_cap: ?u32 = if (@hasDecl(C, "specTokens")) @import("generate.zig").Generator.mtpVerifyDraftsMax(&cooler.config) else null;
                try model_load.warmLoad(io, a, &cooler.config, cooler.xfm, verify_cap, null, cooler.xfm.cache.config);
            }
            errdefer std.debug.print("the skipped warm left these to its requests:\n{s}", .{mlx.engine.late_names.text()});
            try testing.expectEqual(late, mlx.engine.latePipelines());
        }
    }.run);
}
