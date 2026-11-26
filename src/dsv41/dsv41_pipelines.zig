//! DeepSeek's load and the engine's pipeline manifest (`src/pipeline_manifest.zig`): which manifest a load starts from,
//! what it makes ahead of the warm, and the manifest it leaves behind. Every case is one `[pipelines]` line.

const std = @import("std");
const mlx = @import("../mlx.zig");
const manifest = @import("../pipeline_manifest.zig");
const model_mod = @import("../model.zig");
const stream_mod = @import("../expert_stream.zig");
const dsv41_model = @import("dsv41_model.zig");
const shared = @import("../arch/cooler_pipelines.zig");
const Plan = shared.Plan;
const forget = shared.forget;
const rewriteLate = shared.rewriteLate;
const begin = shared.begin;
const Case = shared.Case;

const ModelConfig = model_mod.ModelConfig;

/// DeepSeek's variables that select which kernels the engine makes or what their source says (beside
/// `cooler_pipelines.recipe_env`).
const recipe_env = [_][*:0]const u8{
    "SUSHI_DSV41_COMPOSED",
    "SUSHI_DSV41_DSPARK_EXACT",
    "SUSHI_DSV41_DSPARK_ROWS",
    "SUSHI_DSV41_HEAD_ROWS",
    "SUSHI_DSV41_NO_DECODE_ENGRAM_AHEAD",
    "SUSHI_DSV41_ENGRAM_REUSE",
    "SUSHI_DSV41_STQ_ROUTER_SCALE",
    "SUSHI_DSV41_CAPTURE_LEAN",
};

/// Which pack and which warm; the build part is `cooler_pipelines.buildKey`.
pub fn scopeKey(a: std.mem.Allocator, io: std.Io, config: *const ModelConfig, engine: *const stream_mod.Engine) ![16]u8 {
    var d: manifest.Digest = .{};
    try manifest.addPack(&d, a, io, config.expert_source_dir orelse return error.MissingExpertSourceDir);
    const c = &config.dsv41.?;
    d.addInt("context", c.max_pos);
    d.addInt("window", c.window);
    d.addInt("yarn", @bitCast(c.yarn_factor));
    d.addInt("candidate blocks", c.cand_blocks);
    d.addInt("dspark", @intFromBool(config.drafterOn()));
    d.addInt("tile order", @intFromEnum(config.expert_tile_order));
    d.addInt("expert cache", config.expert_cache_bytes);
    d.addInt("ssd budget", config.expert_ssd_budget_bytes);
    d.addInt("resident", @intFromBool(engine.plan.resident));
    d.addInt("dropped experts", engine.geometry.drop_n);
    for (recipe_env) |name| if (std.c.getenv(name)) |v| d.add(std.mem.span(name), std.mem.span(v));
    shared.addRecipeEnv(&d);
    return d.hex();
}

// ── Tests ──

const testing = std.testing;
const trunk_mod = @import("../trunk.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const generate = @import("../generate.zig");
const dsv41_state = @import("dsv41_state.zig");
const test_support = @import("../test_support.zig");
const dsv41_serving = @import("dsv41_serving.zig");
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
        @import("../cooler_side.zig").forTest(&self.config, true);
        self.weights = try model_mod.loadWeightsForConfig(io, a, dir, &self.config, false);
        errdefer self.weights.deinit();
        self.xfm = try trunk_mod.Trunk.init(io, a, self.config, &self.weights);
        errdefer self.xfm.deinit();
        self.tok = tokenizer_mod.Tokenizer.initEmpty(a);
    }

    fn deinit(self: *Served) void {
        mlx.engine.budget(std.math.maxInt(u64)) catch {};
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
        const scope = dsv41_serving.warmScopeOf(plan.settle());
        try dsv41_serving.warmAllAt(&self.xfm, self.model(), null, scope, 256);
        if (scope == .all) plan.store();
        return scope;
    }

    /// One wide chunk through the forward a request takes, from the start of the cache when `ids` is the first.
    fn prefillChunk(self: *Served, ids: []const u32) !void {
        var ctx = self.xfm.defaultCtx();
        if ((try dsv41_model.currentState(self.model())) == null) try self.xfm.resetCache();
        ctx.cache.step = if (try dsv41_model.currentState(self.model())) |st| st.n else 0;
        const arr = try mlx.newData(ids.ptr, &[_]c_int{ 1, @intCast(ids.len) }, .int32);
        defer _ = mlx.mlx_array_free(arr);
        const out = try dsv41_serving.forwardDsv41With(&self.xfm, &ctx, arr, self.model());
        defer _ = mlx.mlx_array_free(out);
        if (out.ctx != null) try mlx.check(mlx.mlx_array_eval(out));
    }

    /// 24 greedy tokens of the first prompt, through DSpark rounds.
    fn tokens(self: *Served) ![]u32 {
        const a = self.a;
        try self.xfm.resetCache();
        var gen = try generate.Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, &self.xfm, &self.tok, self.ids[0..30], 24, .{ .temperature = 0.0 }, &.{}, .{ .drafter_enabled = true });
        defer gen.deinit(a);
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a);
        while (try gen.nextDrafterRound(a)) |r| {
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

test "dsv41 pipelines: a load with no manifest writes one with the wide chunks' recipes, and a load from it skips the engine's forwards, the wide ones included, serves the same tokens and compiles none inside a prompt (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    var served: Served = undefined;
    try served.init(testing.allocator);
    defer served.deinit();
    const m = served.model();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const key = testKey("b1b1b1b1b1b1b1b1");

    // A load given no manifest root keeps none, says so, and warms in full.
    var rootless = dsv41_serving.beginPipelines(testing.allocator, &served.config, served.xfm.expert_stream.?, true);
    defer rootless.deinit();
    try testing.expectEqual(Case.off, rootless.case);
    try testing.expect(rootless.root == null);
    try testing.expectEqualStrings("no manifest root was given", rootless.why);
    try testing.expectEqual(dsv41_model.WarmScope.all, dsv41_serving.warmScopeOf(rootless.settle()));

    var first = plainBegin(tmp.dir, key);
    defer first.deinit();
    try testing.expectEqual(Case.none, first.case);
    const before_full = engineDispatches();
    const state_only = m.state_only_used;
    try testing.expectEqual(dsv41_model.WarmScope.all, try served.load(&first));
    const full = engineDispatches() - before_full;
    // The wide warm ran before the manifest was stored, so it holds the wide recipes.
    try testing.expect(m.state_only_used > state_only);
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
    const state_only_second = m.state_only_used;
    try testing.expectEqual(dsv41_model.WarmScope.mlx, try served.load(&second));
    try testing.expect(engineDispatches() - before_skip < full);
    try testing.expectEqual(state_only_second, m.state_only_used);
    mlx.engine.seal(true);
    const late = mlx.engine.latePipelines();
    const from_manifest = try served.tokens();
    defer testing.allocator.free(from_manifest);
    try testing.expectEqualSlices(u32, warmed, from_manifest);
    try served.xfm.resetCache();
    var prompt: [97]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = served.ids[(i * 5 + 3) % served.ids.len];
    try served.prefillChunk(prompt[0..64]);
    try served.prefillChunk(prompt[64..97]);
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
    try testing.expect(!plan.settle());
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

test "dsv41 pipelines: a chunk whose index top-k streams over key tiles compiles nothing inside a prompt, after a full warm and after a load from the manifest (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    var served: Served = undefined;
    try served.init(testing.allocator);
    defer served.deinit();
    const m = served.model();
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

test "dsv41 pipelines: a manifest of another build is recompiled, the warm runs in full, and the older builds' directories go but the newest other's" {
    try mlx.requireGpu();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    defer forget();
    const oldest = testKey("0a0a0a0a0a0a0a0a");
    const old = testKey("0c0c0c0c0c0c0c0c");
    const new = testKey("0b0b0b0b0b0b0b0b");
    const io = std.Io.Threaded.global_single_threaded.io();

    var first = plainBegin(tmp.dir, oldest);
    defer first.deinit();
    try testing.expect(!first.settle());
    first.store();
    // Another binary's manifest, written later: the file times tick finer than this wait.
    std.Io.sleep(io, .fromMilliseconds(40), .real) catch {};
    var middle = plainBegin(tmp.dir, old);
    defer middle.deinit();
    _ = middle.settle();
    middle.store();
    std.Io.sleep(io, .fromMilliseconds(40), .real) catch {};

    var second = plainBegin(tmp.dir, new);
    defer second.deinit();
    try testing.expectEqual(Case.other_build, second.case);
    try testing.expectEqual(dsv41_model.WarmScope.all, dsv41_serving.warmScopeOf(second.settle()));
    second.store();

    var now = try manifest.lookup(testing.allocator, io, tmp.dir, new);
    defer now.deinit();
    try testing.expect(now == .same);
    var kept = try manifest.lookup(testing.allocator, io, tmp.dir, old);
    defer kept.deinit();
    try testing.expect(kept == .same);
    var gone = try manifest.lookup(testing.allocator, io, tmp.dir, oldest);
    defer gone.deinit();
    try testing.expect(gone == .other);
}

test "dsv41 pipelines: a truncated manifest is reported by name and rebuilt, never trusted" {
    try mlx.requireGpu();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    defer forget();
    const io = std.Io.Threaded.global_single_threaded.io();
    const key = testKey("c1c1c1c1c1c1c1c1");

    var first = plainBegin(tmp.dir, key);
    defer first.deinit();
    try testing.expect(!first.settle());
    first.store();

    var name_buf: [manifest.Key.dir_len]u8 = undefined;
    var path_buf: [64]u8 = undefined;
    const file = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ key.dirName(&name_buf), manifest.file_name });
    const whole = try tmp.dir.readFileAlloc(io, file, testing.allocator, .limited(1 << 30));
    defer testing.allocator.free(whole);
    // A tail cut, whatever size the recipes this GPU recorded make the file: it keeps the fixed header and a digest's
    // length, so the checksum, not the header check, names it.
    try tmp.dir.writeFile(io, .{ .sub_path = file, .data = whole[0 .. whole.len - 1] });

    var second = plainBegin(tmp.dir, key);
    defer second.deinit();
    try testing.expectEqual(Case.unreadable, second.case);
    try testing.expectEqualStrings("ManifestChecksum", second.why);
    try testing.expectEqual(dsv41_model.WarmScope.all, dsv41_serving.warmScopeOf(second.settle()));
    second.store();

    var rebuilt = try manifest.lookup(testing.allocator, io, tmp.dir, key);
    defer rebuilt.deinit();
    try testing.expect(rebuilt == .same);
    try testing.expectEqual(whole.len, (try tmp.dir.statFile(io, file, .{})).size);
}
