//! DeepSeek's load and the engine's pipeline manifest (`src/pipeline_manifest.zig`): which manifest a load starts from,
//! what it makes ahead of the warm, and the manifest it leaves behind. Every case is one `[pipelines]` line.

const std = @import("std");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const metal = @import("../metal.zig");
const load_clock = @import("../load_clock.zig");
const manifest = @import("../pipeline_manifest.zig");
const model_mod = @import("../model.zig");
const stream_mod = @import("../expert_stream.zig");
const dsv41_model = @import("dsv41_model.zig");

const cache = mlx.engine.pipeline_cache;
const registry = mlx.engine.pipeline_registry;
const ModelConfig = model_mod.ModelConfig;

/// Variables that select which kernels the engine makes or what their source says. The profile, report and probe
/// switches change no recipe and are left out, so a debug session does not write a second manifest.
const recipe_env = [_][*:0]const u8{
    "SUSHI_DSV41_COMPOSED",
    "SUSHI_DSV41_DSPARK_EXACT",
    "SUSHI_DSV41_DSPARK_ROWS",
    "SUSHI_DSV41_HEAD_ROWS",
    "SUSHI_DSV41_NO_DECODE_ENGRAM_AHEAD",
    "SUSHI_DSV41_ENGRAM_REUSE",
    "SUSHI_DSV41_STQ_ROUTER_SCALE",
    "SUSHI_DSV41_CAPTURE_LEAN",
    "SUSHI_EXL3_GEMM_WIN",
    "SUSHI_EXL3_WIN_ALIGN",
    "SUSHI_EXL3_SWIGLU_MAXABS",
    "SUSHI_FORCE_GPU_FAMILY_FALLBACK",
    "MLX_METAL_GPU_ARCH",
    "MLX_ENABLE_TF32",
};

/// Which pack and which warm; the build part is `buildKey`.
pub fn scopeKey(a: std.mem.Allocator, io: std.Io, config: *const ModelConfig, engine: *const stream_mod.Engine) ![16]u8 {
    var d: manifest.Digest = .{};
    try manifest.addPack(&d, a, io, config.expert_source_dir orelse return error.MissingExpertSourceDir);
    const c = &config.dsv41.?;
    d.addInt("context", c.max_pos);
    d.addInt("window", c.window);
    d.addInt("yarn", @bitCast(c.yarn_factor));
    d.addInt("candidate blocks", c.cand_blocks);
    d.addInt("dspark", @intFromBool(config.dsparkOn()));
    d.addInt("tile order", @intFromEnum(config.expert_tile_order));
    d.addInt("expert cache", config.expert_cache_bytes);
    d.addInt("ssd budget", config.expert_ssd_budget_bytes);
    d.addInt("resident", @intFromBool(engine.plan.resident));
    d.addInt("dropped experts", engine.geometry.drop_n);
    for (recipe_env) |name| if (std.c.getenv(name)) |v| d.add(std.mem.span(name), std.mem.span(v));
    return d.hex();
}

pub fn buildKey(a: std.mem.Allocator, io: std.Io) ![16]u8 {
    var d: manifest.Digest = .{};
    var arch_buf: [64]u8 = undefined;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    try manifest.addBuild(&d, a, io, (try metal.device()).architecture(&arch_buf), try mlx.engine.metallibPath(&path_buf));
    return d.hex();
}

pub const Case = enum { off, none, unreadable, other_build, same };

/// A test's directory, in place of the process's root.
pub var root_hook: ?std.Io.Dir = null;

/// The root directory the process was given (`manifest.setRootFromHome`), made if it is missing. `reason` names why
/// there is none: nothing derives a default, so a process that was not given one keeps no manifest.
pub fn openRoot(io: std.Io, reason: *[]const u8) ?std.Io.Dir {
    if (root_hook) |dir| return dir;
    const path = manifest.configuredRoot() orelse {
        reason.* = "no manifest root was given";
        return null;
    };
    return std.Io.Dir.cwd().createDirPathOpen(io, path, .{}) catch |e| {
        reason.* = @errorName(e);
        return null;
    };
}

pub fn threadCount() usize {
    return @min(std.Thread.getCpuCount() catch 4, 8);
}

pub const Plan = struct {
    a: std.mem.Allocator,
    io: std.Io,
    root: ?std.Io.Dir,
    key: manifest.Key,
    key_ms: u64,
    found: manifest.Lookup,
    case: Case,
    why: []const u8 = "",
    threads: usize,
    /// What the last `settle` spent making pipelines ahead, and how many it made.
    made_ns: u64 = 0,
    made_count: u64 = 0,

    pub fn deinit(p: *Plan) void {
        p.found.deinit();
    }

    /// Makes the pipelines of the manifest found, on worker threads, and says which case runs. The warm that follows is
    /// the scope returned.
    pub fn settle(p: *Plan) dsv41_model.WarmScope {
        // A later warm adds to a manifest only where a complete one stands: the one found, or the one `store` writes.
        last = null;
        var dir_buf: [manifest.Key.dir_len]u8 = undefined;
        const dir = p.key.dirName(&dir_buf);
        const recipes: []const manifest.Recipe = switch (p.found) {
            .same, .other => |l| l.recipes,
            else => &.{},
        };
        if (recipes.len > 0) {
            const t0 = load_clock.now();
            const report = if (cache.start(recipes, p.threads)) |job| cache.finish(job) else |e| {
                log.warn("[pipelines] manifest {s}: the workers did not start ({s}); the warm forwards run and rebuild it\n", .{ dir, @errorName(e) });
                p.case = .unreadable;
                return .all;
            };
            const wall = load_clock.now() - t0;
            p.made_ns = wall;
            p.made_count = report.made;
            const wall_ms = wall / std.time.ns_per_ms;
            if (report.failed > 0) {
                registry.settleMarkers(false);
                log.warn("[pipelines] manifest {s}: {d} of {d} recipes did not make ({s}); not trusted, the warm forwards run and rebuild it\n", .{ dir, report.failed, recipes.len, report.firstError() });
                p.case = .unreadable;
                return .all;
            }
            registry.settleMarkers(p.case == .same);
            if (p.case == .same) {
                // Every recipe of the file made, so every one is a pipeline of this process: a manifest rewritten
                // after a late warm keeps the ones this load never launched.
                for (recipes) |r| registry.record(r);
                remember(p);
                log.info("[pipelines] made {d} pipelines from the manifest {s} on {d} threads in {d} ms; the engine's warm forwards are skipped (key {d} ms)\n", .{ report.made, dir, p.threads, wall_ms, p.key_ms });
                return .mlx;
            }
            log.info("[pipelines] manifest of another build, recompiled {d} on {d} threads in {d} ms; the warm forwards run and the manifest {s} is rewritten (key {d} ms)\n", .{ report.made, p.threads, wall_ms, dir, p.key_ms });
            return .all;
        }
        registry.settleMarkers(false);
        switch (p.case) {
            .none => log.info("[pipelines] no manifest; the warm forwards run and write {s} (key {d} ms)\n", .{ dir, p.key_ms }),
            .unreadable => log.warn("[pipelines] manifest unreadable ({s}); the warm forwards run and rebuild it (key {d} ms)\n", .{ p.why, p.key_ms }),
            else => log.info("[pipelines] manifest off ({s}); the warm forwards run\n", .{p.why}),
        }
        return .all;
    }

    /// After a complete warm: the recipes the process made, written for this key, and the older builds of the scope
    /// removed. A failure is a warning, not a failed load.
    pub fn store(p: *Plan) void {
        const root = p.root orelse return;
        remember(p);
        if (!write(p.a, p.io, root, p.key)) return;
        var removed: std.ArrayList([manifest.Key.dir_len]u8) = .empty;
        defer removed.deinit(p.a);
        manifest.prune(p.a, p.io, root, p.key, &removed) catch |e| log.warn("[pipelines] older manifests not removed: {s}\n", .{@errorName(e)});
        var dir_buf: [manifest.Key.dir_len]u8 = undefined;
        log.info("[pipelines] manifest {s} written: {d} recipes\n", .{ p.key.dirName(&dir_buf), registry.recipes().len });
        for (removed.items) |name| log.info("[pipelines] removed the manifest {s} of an older build\n", .{name});
    }
};

/// Where this load's manifest lives once it holds a complete warm, for a warm that runs after the load's own (the
/// vision tower's).
var last: ?struct { a: std.mem.Allocator, io: std.Io, root: std.Io.Dir, key: manifest.Key } = null;

fn remember(p: *const Plan) void {
    last = if (p.root) |root| .{ .a = p.a, .io = p.io, .root = root, .key = p.key } else null;
}

fn write(a: std.mem.Allocator, io: std.Io, root: std.Io.Dir, key: manifest.Key) bool {
    if (registry.lost > 0) {
        log.warn("[pipelines] manifest not written: {d} recipes could not be kept\n", .{registry.lost});
        return false;
    }
    manifest.write(a, io, root, key, registry.recipes()) catch |e| {
        log.warn("[pipelines] manifest not written: {s}\n", .{@errorName(e)});
        return false;
    };
    return true;
}

/// A test's directory is gone with the test: nothing is rewritten after it.
pub fn forget() void {
    last = null;
}

/// Pipelines made after the load's manifest was settled or written: the manifest of the load's key, written again with
/// them. A load with no manifest root writes nothing.
pub fn rewriteLate() void {
    const l = last orelse return;
    if (!write(l.a, l.io, l.root, l.key)) return;
    var dir_buf: [manifest.Key.dir_len]u8 = undefined;
    log.info("[pipelines] manifest {s} written again with a later warm: {d} recipes\n", .{ l.key.dirName(&dir_buf), registry.recipes().len });
}

/// Looks up this key's manifest. A problem with the directory or the
/// file is a case, never an error.
pub fn begin(a: std.mem.Allocator, io: std.Io, root: ?std.Io.Dir, why_no_root: []const u8, key: manifest.Key, key_ms: u64) Plan {
    var plan: Plan = .{ .a = a, .io = io, .root = root, .key = key, .key_ms = key_ms, .found = .none, .case = .off, .why = why_no_root, .threads = threadCount() };
    const dir = root orelse return plan;
    plan.found = manifest.lookup(a, io, dir, key) catch |e| {
        plan.why = @errorName(e);
        plan.found = .none;
        return plan;
    };
    switch (plan.found) {
        .none => plan.case = .none,
        .unreadable => |u| {
            plan.case = .unreadable;
            plan.why = u.reason;
        },
        .same => plan.case = .same,
        .other => plan.case = .other_build,
    }
    return plan;
}

// ── Tests ──

const testing = std.testing;
const trunk_mod = @import("../trunk.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const generate = @import("../generate.zig");
const dsv41_state = @import("dsv41_state.zig");
const test_support = @import("../test_support.zig");
const dsv41_serving = @import("dsv41_serving.zig");
const dsv41_borrow = @import("dsv41_borrow.zig");
const dsv41_scope = @import("dsv41_scope.zig");

/// A tiny STQ DeepSeek served on the engine of our own, as a load builds it, without the warm.
const Served = struct {
    a: std.mem.Allocator,
    config: ModelConfig,
    weights: model_mod.Weights,
    xfm: trunk_mod.Trunk,
    tok: tokenizer_mod.Tokenizer,
    ids: []u32,
    was_enabled: bool,

    fn init(self: *Served, a: std.mem.Allocator) !void {
        const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
        const io = std.Io.Threaded.global_single_threaded.io();
        self.a = a;
        self.was_enabled = mlx.engine.enabled;
        mlx.engine.enabled = true;
        errdefer test_support.restoreEngine(self.was_enabled);
        var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
        defer fx.deinit();
        var sc = dsv41_scope.Scope.plain(a, mlx.gpuStream());
        defer sc.deinit();
        self.ids = try dsv41_model.tokens(&sc, &fx);
        errdefer a.free(self.ids);
        self.config = try model_mod.parseConfig(io, a, dir);
        errdefer self.config.deinit(a);
        self.config.expert_streaming = true;
        self.config.stream_mtp_head = true;
        self.config.expert_cache_bytes = 64 << 20;
        self.config.expert_source_dir = try a.dupe(u8, dir);
        self.weights = try model_mod.loadWeightsForConfig(io, a, dir, &self.config, false);
        errdefer self.weights.deinit();
        self.xfm = try trunk_mod.Trunk.init(io, a, self.config, &self.weights);
        errdefer self.xfm.deinit();
        self.tok = tokenizer_mod.Tokenizer.initEmpty(a);
        dsv41_state.modelOf(&self.xfm).?.engine_decode = true;
        dsv41_borrow.own_override = true;
    }

    fn deinit(self: *Served) void {
        dsv41_borrow.own_override = null;
        mlx.engine.budget(std.math.maxInt(u64)) catch {};
        dsv41_state.modelOf(&self.xfm).?.engine_decode = false;
        dsv41_state.modelOf(&self.xfm).?.engine_prefill = false;
        self.tok.deinit();
        self.xfm.deinit();
        self.weights.deinit();
        self.config.deinit(self.a);
        self.a.free(self.ids);
        test_support.restoreEngine(self.was_enabled);
    }

    fn model(self: *Served) *dsv41_model.Model {
        return dsv41_state.modelOf(&self.xfm).?;
    }

    /// One load's warm as `initDsv41` runs it: settle the plan, warm what it says, store after a full warm.
    fn load(self: *Served, plan: *Plan) !dsv41_model.WarmScope {
        const scope = plan.settle();
        try dsv41_serving.warmAllAt(&self.xfm, self.model(), null, scope, 256);
        if (scope == .all) plan.store();
        return scope;
    }

    /// One wide chunk through the forward a request takes, from the start of the cache when `ids` is the first.
    fn prefillChunk(self: *Served, ids: []const u32) !void {
        var ctx = self.xfm.defaultCtx();
        if (self.model().dec_state == null) try self.xfm.resetCache();
        ctx.cache.step = if (self.model().dec_state) |st| st.n else 0;
        const arr = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32);
        defer _ = mlx.mlx_array_free(arr);
        const out = try dsv41_serving.forwardDsv41With(&self.xfm, &ctx, arr, self.model());
        defer _ = mlx.mlx_array_free(out);
        if (out.ctx != null) try mlx.check(mlx.mlx_array_eval(out));
    }

    /// 24 greedy tokens of the first prompt, through DSpark rounds.
    fn tokens(self: *Served) ![]u32 {
        const a = self.a;
        try self.xfm.resetCache();
        var gen = try generate.Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, &self.xfm, &self.tok, self.ids[0..30], 24, .{ .temperature = 0.0 }, &.{}, .{ .dspark_enabled = true });
        defer gen.deinit(a);
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a);
        while (try gen.nextDspark(a)) |r| {
            defer a.free(r.tokens);
            try out.appendSlice(a, r.tokens);
        }
        return out.toOwnedSlice(a);
    }
};

fn testKey(build: *const [16]u8) manifest.Key {
    return .{ .scope = "5c0pe5c0pe5c0pe5".*, .build = build.* };
}

fn plainBegin(root: std.Io.Dir, key: manifest.Key) Plan {
    return begin(testing.allocator, std.Io.Threaded.global_single_threaded.io(), root, "", key, 0);
}

fn engineDispatches() u64 {
    return mlx.engine.stats().dispatches;
}

test "dsv41 pipelines: a load with no manifest writes one, and a load from it skips the engine's forwards and serves the same tokens (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    var served: Served = undefined;
    try served.init(testing.allocator);
    defer served.deinit();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const key = testKey("b1b1b1b1b1b1b1b1");

    var first = plainBegin(tmp.dir, key);
    defer first.deinit();
    try testing.expectEqual(Case.none, first.case);
    const before_full = engineDispatches();
    try testing.expectEqual(dsv41_model.WarmScope.all, try served.load(&first));
    const full = engineDispatches() - before_full;
    mlx.engine.seal(true);
    defer mlx.engine.seal(false);
    const warmed = try served.tokens();
    defer testing.allocator.free(warmed);

    var written = try manifest.lookup(testing.allocator, std.Io.Threaded.global_single_threaded.io(), tmp.dir, key);
    defer written.deinit();
    try testing.expect(written == .same);
    try testing.expect(written.same.recipes.len > 0);

    mlx.engine.seal(false);
    var second = plainBegin(tmp.dir, key);
    defer second.deinit();
    try testing.expectEqual(Case.same, second.case);
    const before_skip = engineDispatches();
    try testing.expectEqual(dsv41_model.WarmScope.mlx, try served.load(&second));
    try testing.expect(engineDispatches() - before_skip < full);
    mlx.engine.seal(true);
    const late = mlx.engine.latePipelines();
    const from_manifest = try served.tokens();
    defer testing.allocator.free(from_manifest);
    try testing.expectEqualSlices(u32, warmed, from_manifest);
    try testing.expectEqual(late, mlx.engine.latePipelines());
}

test "dsv41 pipelines: a load whose warm did not finish leaves no manifest, and a later warm writes none in its place" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    defer forget();
    const io = std.Io.Threaded.global_single_threaded.io();
    const key = testKey("c2c2c2c2c2c2c2c2");
    var plan = plainBegin(tmp.dir, key);
    defer plan.deinit();
    try testing.expectEqual(dsv41_model.WarmScope.all, plan.settle());
    rewriteLate();
    var unfinished = try manifest.lookup(testing.allocator, io, tmp.dir, key);
    defer unfinished.deinit();
    try testing.expect(unfinished == .none);

    plan.store();
    rewriteLate();
    var stored = try manifest.lookup(testing.allocator, io, tmp.dir, key);
    defer stored.deinit();
    try testing.expect(stored == .same);
}

test "dsv41 pipelines: the wide chunks' recipes are in the manifest, and a load from it runs none of the wide forwards and compiles none inside a prompt (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    var served: Served = undefined;
    try served.init(testing.allocator);
    defer served.deinit();
    const m = served.model();
    m.engine_prefill = true;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const key = testKey("d1d1d1d1d1d1d1d1");

    var first = plainBegin(tmp.dir, key);
    defer first.deinit();
    const state_only = m.state_only_used;
    try testing.expectEqual(dsv41_model.WarmScope.all, try served.load(&first));
    // The wide warm ran before the manifest was stored, so it holds the wide recipes.
    try testing.expect(m.state_only_used > state_only);
    var written = try manifest.lookup(testing.allocator, std.Io.Threaded.global_single_threaded.io(), tmp.dir, key);
    defer written.deinit();
    try testing.expect(written == .same);
    const with_wide = written.same.recipes.len;

    mlx.engine.seal(false);
    var second = plainBegin(tmp.dir, key);
    defer second.deinit();
    try testing.expectEqual(Case.same, second.case);
    const state_only_second = m.state_only_used;
    try testing.expectEqual(dsv41_model.WarmScope.mlx, try served.load(&second));
    try testing.expectEqual(state_only_second, m.state_only_used);
    mlx.engine.seal(true);
    defer mlx.engine.seal(false);
    const late = mlx.engine.latePipelines();
    var prompt: [97]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = served.ids[(i * 5 + 3) % served.ids.len];
    try served.prefillChunk(prompt[0..64]);
    try served.prefillChunk(prompt[64..97]);
    try testing.expectEqual(late, mlx.engine.latePipelines());
    try testing.expect(with_wide > 0);
}

test "dsv41 pipelines: a chunk whose index top-k streams over key tiles compiles nothing inside a prompt, after a full warm and after a load from the manifest (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    var served: Served = undefined;
    try served.init(testing.allocator);
    defer served.deinit();
    const m = served.model();
    m.engine_prefill = true;
    // The fixture's window of 8 rows is a span the indexer scans a row at a time: a wider one is scored in tiles.
    m.cfg.window = 32;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const key = testKey("e2e2e2e2e2e2e2e2");
    var prompt: [97]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = served.ids[(i * 5 + 3) % served.ids.len];
    const served_tile = m.index_stream_tile;
    defer m.index_stream_tile = served_tile;
    for ([_]dsv41_model.WarmScope{ .all, .mlx }) |scope| {
        mlx.engine.seal(false);
        m.index_stream_tile = served_tile;
        var plan = plainBegin(tmp.dir, key);
        defer plan.deinit();
        try testing.expectEqual(scope, try served.load(&plan));
        mlx.engine.seal(true);
        defer mlx.engine.seal(false);
        const late = mlx.engine.latePipelines();
        const streams = dsv41_model.indexStreams();
        // The fixture's context is under one served tile of keys: the smallest tile makes its second chunk span two.
        m.index_stream_tile = dsv41_model.indexStreamTileMin(m);
        try served.prefillChunk(prompt[0..64]);
        try served.prefillChunk(prompt[64..97]);
        try testing.expect(dsv41_model.indexStreams() > streams);
        try testing.expectEqual(late, mlx.engine.latePipelines());
    }
}

test "dsv41 pipelines: a manifest of another build is recompiled, the warm runs in full, and the older build's directory goes (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    var served: Served = undefined;
    try served.init(testing.allocator);
    defer served.deinit();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const old = testKey("0a0a0a0a0a0a0a0a");
    const new = testKey("0b0b0b0b0b0b0b0b");

    var first = plainBegin(tmp.dir, old);
    defer first.deinit();
    _ = try served.load(&first);

    var second = plainBegin(tmp.dir, new);
    defer second.deinit();
    try testing.expectEqual(Case.other_build, second.case);
    try testing.expectEqual(dsv41_model.WarmScope.all, try served.load(&second));

    const io = std.Io.Threaded.global_single_threaded.io();
    var now = try manifest.lookup(testing.allocator, io, tmp.dir, new);
    defer now.deinit();
    try testing.expect(now == .same);
    var gone = try manifest.lookup(testing.allocator, io, tmp.dir, old);
    defer gone.deinit();
    try testing.expect(gone == .other);
}

test "dsv41 pipelines: a truncated manifest is reported by name and rebuilt, never trusted (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    var served: Served = undefined;
    try served.init(testing.allocator);
    defer served.deinit();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    const key = testKey("c1c1c1c1c1c1c1c1");

    var first = plainBegin(tmp.dir, key);
    defer first.deinit();
    _ = try served.load(&first);

    var name_buf: [manifest.Key.dir_len]u8 = undefined;
    var path_buf: [64]u8 = undefined;
    const file = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ key.dirName(&name_buf), manifest.file_name });
    const whole = try tmp.dir.readFileAlloc(io, file, testing.allocator, .limited(1 << 30));
    defer testing.allocator.free(whole);
    try tmp.dir.writeFile(io, .{ .sub_path = file, .data = whole[0 .. whole.len / 2] });

    var second = plainBegin(tmp.dir, key);
    defer second.deinit();
    try testing.expectEqual(Case.unreadable, second.case);
    try testing.expectEqualStrings("ManifestChecksum", second.why);
    try testing.expectEqual(dsv41_model.WarmScope.all, try served.load(&second));

    var rebuilt = try manifest.lookup(testing.allocator, io, tmp.dir, key);
    defer rebuilt.deinit();
    try testing.expect(rebuilt == .same);
    try testing.expectEqual(whole.len, (try tmp.dir.statFile(io, file, .{})).size);
}

test "dsv41 pipelines: a load given no manifest root keeps no manifest and says so (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    var served: Served = undefined;
    try served.init(testing.allocator);
    defer served.deinit();
    var plan = dsv41_serving.beginPipelines(testing.allocator, &served.config, served.xfm.expert_stream.?, true);
    defer plan.deinit();
    try testing.expectEqual(Case.off, plan.case);
    try testing.expect(plan.root == null);
    try testing.expectEqualStrings("no manifest root was given", plan.why);
    try testing.expectEqual(dsv41_model.WarmScope.all, try served.load(&plan));
}
