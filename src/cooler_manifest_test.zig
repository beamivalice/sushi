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

test "cooler contract: a pipeline a Cooler request compiles late is written to the load's manifest, for the next load to make ahead" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const shared = @import("arch/cooler_pipelines.zig");
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                try @import("model_load.zig").warmLoad(io, testing.allocator, &cooler.config, cooler.xfm, null, null, cooler.xfm.cache.config);
            }
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

test "cooler contract: a Cooler load whose manifest holds a complete served warm runs none, in the process that wrote it and in a fresh one" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const shared = @import("arch/cooler_pipelines.zig");
            const cooler_warm = @import("cooler_warm.zig");
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            var runs: [3]u32 = undefined;
            for (&runs, 0..) |*r, i| {
                // The third load is a fresh process: it knows the manifest's file and nothing the first two made.
                if (i == 2) mlx.engine.pipeline_registry.resetForTest();
                // `open` is the served load, its warm included.
                const before = cooler_warm.runs_for_test;
                var cooler: Load = undefined;
                try cooler.open(C, dir, true);
                defer _ = cooler.close();
                r.* = cooler_warm.runs_for_test - before;
            }
            try testing.expectEqual(@as(u32, 1), runs[0]);
            try testing.expectEqual(@as(u32, 0), runs[1]);
            try testing.expectEqual(@as(u32, 0), runs[2]);
        }
    }.run);
}

test "cooler contract: a Cooler load whose manifest holds a served warm of another shape or another scope still runs it in full" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const shared = @import("arch/cooler_pipelines.zig");
            const cooler_warm = @import("cooler_warm.zig");
            const dir_len = @import("pipeline_manifest.zig").Key.dir_len;
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            var runs: [3]u32 = undefined;
            for (&runs, 0..) |*r, i| {
                const before = cooler_warm.runs_for_test;
                var cooler: Load = undefined;
                try cooler.prepare(C, dir, true);
                // A narrower `--prefill-chunk-max`: the same pack, another chunk for the warm's requests.
                const settings = @import("settings_resolve.zig");
                const was = settings.launch;
                if (i == 1) settings.launch = .{ .prefill_chunk_max = 2048, .prefill_chunk_min = 2048 };
                defer settings.launch = was;
                try cooler.loadServed(dir);
                defer _ = cooler.close();
                r.* = cooler_warm.runs_for_test - before;
                if (i == 1) {
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
            try testing.expectEqual(@as(u32, 1), runs[2]);
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

test "cooler contract: a served-warm marker one pack's manifest holds is not taken under another pack whose own manifest never held it" {
    const C = @import("dsv41/arch.zig").cooler_contract;
    var two: TwoPacks = undefined;
    try two.begin();
    defer two.end();
    // Process 1: pack B at a narrower chunk, so B's manifest holds that shape's marker only.
    const b_narrow = try warmRuns(C, two.packs.b, 2048);
    // Process 2: pack A at its own shape.
    two.restart();
    const a_first = try warmRuns(C, two.packs.a, 0);
    // Process 3: A again, trusted and skipped as designed, then B at the default shape: B's manifest holds no warm of it.
    two.restart();
    const a_again = try warmRuns(C, two.packs.a, 0);
    const b_default = try warmRuns(C, two.packs.b, 0);
    errdefer std.debug.print("served warms that ran: b_narrow={d} a_first={d} a_again={d} b_default={d}\n", .{ b_narrow, a_first, a_again, b_default });
    try testing.expectEqual(@as(u32, 1), b_narrow);
    try testing.expectEqual(@as(u32, 1), a_first);
    try testing.expectEqual(@as(u32, 0), a_again);
    try testing.expectEqual(@as(u32, 1), b_default);
}

test "cooler contract: a served-warm marker one pack's warm recorded is not written into another pack's manifest" {
    const C = @import("dsv41/arch.zig").cooler_contract;
    var two: TwoPacks = undefined;
    try two.begin();
    defer two.end();
    // Process 1: A at the default shape, then B at a narrower chunk: B's warm never ran at the default shape.
    const a_first = try warmRuns(C, two.packs.a, 0);
    const b_narrow = try warmRuns(C, two.packs.b, 2048);
    // Process 2: B at the default shape warms in full.
    two.restart();
    const b_default = try warmRuns(C, two.packs.b, 0);
    errdefer std.debug.print("served warms that ran: a_first={d} b_narrow={d} b_default={d}\n", .{ a_first, b_narrow, b_default });
    try testing.expectEqual(@as(u32, 1), a_first);
    try testing.expectEqual(@as(u32, 1), b_narrow);
    try testing.expectEqual(@as(u32, 1), b_default);
}

test "cooler contract: two packs whose served warms have one shape and one bill each measure their widest request against it" {
    const C = @import("dsv41/arch.zig").cooler_contract;
    const cooler_warm = @import("cooler_warm.zig");
    var two: TwoPacks = undefined;
    try two.begin();
    defer two.end();
    // Process 1: B at a narrower chunk, then A at the shape both share.
    _ = try warmRuns(C, two.packs.b, 2048);
    _ = try warmRuns(C, two.packs.a, 0);
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
/// with `flipped`'s setting at `loaded` (none: the control); then a full warm under the same settings from no manifest.
/// The warms its load ran, the kernels the full warm left late, and whether the load drafts.
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
    const added = mlx.engine.latePipelines() - late;
    if (added > 0) std.debug.print("{s} left {d} kernels late:\n{s}\n", .{ if (flipped) |F| @typeName(F) else "the control", added, mlx.engine.late_names.text() });
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
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
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
                try first.open(C, dir, true);
                defer _ = first.close();
                // A load with no drafter: its warm reads no MTP setting, so there is nothing for its marker to keep apart.
                if (@import("generate.zig").MtpHeadRef.fromModel(first.xfm, true, null) == null) return error.SkipZigTest;
            }
            // The control: no setting flipped, the manifest's warm skipped and nothing left late.
            const control = try settingProbe(C, dir, written.dir, null);
            try testing.expectEqual(@as(u32, 0), control.runs);
            try testing.expectEqual(@as(u64, 0), control.late);
            inline for (mtp_settings) |Setting| {
                const flipped = try settingProbe(C, dir, written.dir, Setting);
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
