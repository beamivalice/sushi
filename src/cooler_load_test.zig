//! The Sushi Cooler contract's load cells: where a load's weights live, its warm and seal, its manifest, its refusals
//! and its teardown. The runner and the shared helpers are in `cooler_contract_test.zig`.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const Trunk = @import("trunk.zig").Trunk;
const registry = @import("arch/registry.zig");
const arch_handle = @import("arch/handle.zig");
const test_support = @import("test_support.zig");
const model_load = @import("model_load.zig");
const io = std.Io.Threaded.global_single_threaded.io();
const contract = @import("cooler_contract_test.zig");
const Load = contract.Load;
const Served = contract.Served;
const Sheet = contract.Sheet;
const eachArch = contract.eachArch;
const eachFixture = contract.eachFixture;
const eachLayout = contract.eachLayout;
const greedy = contract.greedy;
const withUiSettings = contract.withUiSettings;


test "cooler contract: a Cooler load that drops experts by usage serves, routing only the kept ones on the GPU" {
    // A route to a dropped expert fails the request (`ExpertDropped`): the router must never pick one.
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "dropExperts")) return error.SkipZigTest;
            const a = testing.allocator;
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            var ld: Load = undefined;
            try ld.prepare(C, dir, true);
            const root = @import("test_support.zig").usagePack(&tmp, dir, ld.config.num_hidden_layers, ld.config.num_experts, ld.config.num_experts_per_tok) catch |e| {
                ld.config.deinit(a);
                return e;
            };
            defer a.free(root);
            const n = (C.dropExperts(a, &ld.config, root) catch |e| {
                ld.config.deinit(a);
                return e;
            }) orelse {
                ld.config.deinit(a);
                return error.SkipZigTest;
            };
            ld.config.drop_expert = n;
            try ld.load(root);
            defer _ = ld.close();
            try testing.expect(ld.onEngine());
            // Every expert the router can pick is resident: a one-row request reads no route on the host.
            const engine = ld.xfm.expert_stream orelse return error.TestExpectedExpertStream;
            if (engine.plan.resident) {
                const before = engine.hostRoutes();
                const handed = mlx.engine.stats().handovers;
                const tokens = 4;
                var one = try greedy(&ld, ids[0..1], tokens);
                one.deinit(a);
                try testing.expectEqual(before, engine.hostRoutes());
                // Nor does a layer hand its pass over early: each boundary leaves the GPU idle, and the kicks start it.
                // A token hands over two: the step's own next forward, and its sample's pass before the next
                // forward reads the ids (`qwen4_hc.handOverIdsWriter`).
                try testing.expect(mlx.engine.stats().handovers - handed <= 2 * tokens);
            }
            var said = try greedy(&ld, ids[0..@min(ids.len, 9)], 6);
            said.deinit(a);
        }
    }.run);
}

test "cooler contract: a Cooler load runs on the engine and holds every weight in an owned region, none in the forward pool" {
    try eachLayout(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            var cooler: Load = undefined;
            try cooler.openLight(C, dir, true);
            defer _ = cooler.close();
            try testing.expect(cooler.onEngine());
            try testing.expectEqual(@as(usize, 0), try C.weightsOutsideRegions(cooler.xfm));
            if (!cooler.config.expert_streaming or cooler.config.stream_mtp_head) try testing.expect(C.headLoaded(cooler.xfm));
        }
    }.run);
}

test "cooler contract: a Cooler load opens under whatever engine cap was in force before it, and its unload leaves the pool as it found it" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const base = Sheet.now();
            // A cap an earlier forward left in force, below the smallest buffer the load makes.
            try mlx.engine.budget(4 << 10);
            defer mlx.engine.budget(std.math.maxInt(u64)) catch {};
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            _ = ld.close();
            try testing.expectEqual(base, Sheet.now());
        }
    }.run);
}

test "cooler contract: a second Cooler load beside a sealed one, of another arch, warms unsealed and ends sealed" {
    // The load scope is shared: the second load's warm makes its own pipelines, none of them counted as late.
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    var ran = false;
    inline for (comptime std.meta.tags(registry.Tag)) |ta| {
        if (comptime !@hasDecl(registry.Mod(ta), "cooler_contract")) continue;
        ran = (try besideSealed(ta)) or ran;
    }
    if (!ran) return error.SkipZigTest;
}

/// A load of `ta`'s first fixture, warmed and sealed, then a load of each other arch's beside it, one at a time; false
/// when it ran none. Beside a load on the ledger each is refused by name.
fn besideSealed(comptime ta: registry.Tag) !bool {
    const A = registry.Mod(ta).cooler_contract;
    // A run that names its fixtures takes each ordered pair either of whose arches it names (`pairChosen`); each first
    // arch still loads once, for every pair it starts.
    const sel = contract.selection();
    if (!pairsWith(ta, sel)) return false;
    const da = std.mem.span(std.c.getenv(A.fixtures[0]) orelse return false);
    const Warm = struct {
        fn run(ld: *Load) !void {
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            return @import("model_load.zig").warmLoad(io, testing.allocator, &ld.config, ld.xfm, null, null, ld.xfm.cache.config);
        }
    };
    var first: Load = undefined;
    try first.open(A, da, true);
    defer _ = first.close();
    try Warm.run(&first);
    try testing.expect(mlx.engine.sealed());
    var ran = false;
    inline for (comptime std.meta.tags(registry.Tag)) |tb| {
        if (comptime ta == tb or !@hasDecl(registry.Mod(tb), "cooler_contract")) continue;
        const B = registry.Mod(tb).cooler_contract;
        const db_z = std.c.getenv(B.fixtures[0]);
        if (db_z != null and contract.pairChosen(sel, &A.fixtures, &B.fixtures, true, true)) {
            const db = std.mem.span(db_z.?);
            ran = true;
            if (first.config.static_ledger != null) {
                try contract.expectRefusedBesideLedger(B, db);
            } else {
                const late = mlx.engine.latePipelines();
                mlx.engine.late_names.clear();
                var second: Load = undefined;
                try second.open(B, db, true);
                defer _ = second.close();
                try Warm.run(&second);
                errdefer std.debug.print("cooler contract: the second load's warm ({s} beside {s}) counted late:\n{s}", .{ B.fixtures[0], A.fixtures[0], mlx.engine.late_names.text() });
                try testing.expectEqual(late, mlx.engine.latePipelines());
                try testing.expect(mlx.engine.sealed());
            }
        }
    }
    return ran;
}

fn pairsWith(comptime ta: registry.Tag, sel: ?[]const u8) bool {
    const A = registry.Mod(ta).cooler_contract;
    const a_set = std.c.getenv(A.fixtures[0]) != null;
    inline for (comptime std.meta.tags(registry.Tag)) |tb| {
        if (comptime ta == tb or !@hasDecl(registry.Mod(tb), "cooler_contract")) continue;
        const B = registry.Mod(tb).cooler_contract;
        if (contract.pairChosen(sel, &A.fixtures, &B.fixtures, a_set, std.c.getenv(B.fixtures[0]) != null)) return true;
    }
    return false;
}

test "cooler contract: a Cooler load's prefix cache keeps its arrays on the engine, and an MLX load's on MLX" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const prefix_arch = @import("arch/prefix.zig");
            inline for (.{ true, false }) |cool| {
                // An arch with no MLX arm has no MLX load to hold to its side.
                if (comptime !cool and !contract.hasMlxArm(C)) continue;
                var ld: Load = undefined;
                try ld.openLight(C, dir, cool);
                defer _ = ld.close();
                const on = prefix_arch.arraysOn(&ld.config);
                errdefer std.debug.print("cooler contract: a load {s} keeps its prefix arrays on {s}\n", .{ if (cool) "on Cooler" else "on MLX", if (on == null) "MLX" else "the engine" });
                if (cool) try testing.expect(on != null and mlx.engine.isStream(on.?)) else try testing.expect(on == null);
            }
        }
    }.run);
}

test "cooler contract: a Cooler load whose widest served request holds more than its bill is refused by name" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const shared = @import("arch/cooler_pipelines.zig");
            const cooler_warm = @import("cooler_warm.zig");
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            // A fresh manifest, so the load runs its served warm.
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            cooler_warm.bill_for_test = 1 << 20;
            defer cooler_warm.bill_for_test = null;
            var cooler: Load = undefined;
            if (cooler.open(C, dir, true)) |_| {
                _ = cooler.close();
                return error.TestUnexpectedResult;
            } else |e| try testing.expectEqual(error.WarmOverBill, e);
            var msg: [512]u8 = undefined;
            _ = mlx.takeError(&msg);
        }
    }.run);
}

test "cooler contract: a Cooler load under each KV precision and context extension the web UI offers takes it and loads on the engine" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            _ = ids;
            const rows = [_]struct { []const u8, []const u8, []const u8 }{
                .{ "{\"kv_quant\": \"off\"}", "kv_quant", "off" },
                .{ "{\"kv_quant\": \"4\"}", "kv_quant", "4" },
                .{ "{\"kv_quant\": \"6\"}", "kv_quant", "6" },
                .{ "{\"yarn_factor\": 2}", "yarn_factor", "2" },
            };
            inline for (rows) |r| {
                const row = r[0];
                const Cell = struct {
                    var d: []const u8 = undefined;
                    fn body() !void {
                        var ld: Load = undefined;
                        try ld.openLight(C, d, true);
                        defer _ = ld.close();
                        // A row the web UI does not offer for this model (DeepSeek keeps no KV cache to quantize, a
                        // RoPE with no YaRN) has nothing to hold; one it offers must take effect, never skip.
                        const fit = @import("arch/flags.zig").settingsFit(&ld.config);
                        const model_settings = @import("model_settings.zig");
                        if (!model_settings.applies(r[1], fit) or !model_settings.choiceAllowed(r[1], r[2], fit)) return;
                        errdefer std.debug.print("cooler contract: {s} is offered for this model and did not take effect\n", .{row});
                        if (comptime std.mem.eql(u8, r[1], "kv_quant")) {
                            try testing.expect(ld.config.kv_quant_override != null);
                        } else try testing.expect(ld.config.yarn_factor_override != null);
                        try testing.expect(ld.onEngine());
                    }
                };
                Cell.d = dir;
                try withUiSettings(dir, row, Cell.body);
            }
        }
    }.run);
}

/// The process's memory as a headroom read sees it: MLX's bytes and the engine's pool.
const MemSnap = struct {
    mlx_cache: u64,
    pool: u64,
    pool_free: u64,

    fn now() MemSnap {
        mlx.engine.close();
        var cache: usize = 0;
        _ = mlx.mlx_get_cache_memory(&cache);
        const st = mlx.engine.stats();
        return .{ .mlx_cache = cache, .pool = st.pool_bytes, .pool_free = st.pool_free };
    }
};

/// A borrow whose freed pools unwire late (issue 105): the first request's first wide forward reads its headroom a byte
/// short until the read after its first release has passed, then has room.
const LateUnwireHeadroom = struct {
    var req: usize = 0;
    var reads: u32 = 0;
    var borrowed = false;
    fn headroom(_: *const model_mod.ModelConfig) @import("bills.zig").Headroom {
        const need = @import("dsv41/dsv41_borrow.zig").forward_need;
        if (req == 0 and !borrowed and need > 0) {
            reads += 1;
            if (reads <= 3) return .{ .live = need - 1 };
            borrowed = true;
        }
        return .{ .live = std.math.maxInt(u64) / 4 };
    }
};

test "cooler contract: the first served DeepSeek request after a Cooler load's warm starts from the state the second does, and a streamed load's first wide prefill is served through a late-unwiring borrow (issue 107)" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            if (!@import("arch/config.zig").isOf(&ld.config, "dsv41")) return error.SkipZigTest;
            const borrow = @import("dsv41/dsv41_borrow.zig");
            const was_fn = borrow.headroom_fn;
            borrow.headroom_fn = &LateUnwireHeadroom.headroom;
            defer borrow.headroom_fn = was_fn;
            LateUnwireHeadroom.reads = 0;
            LateUnwireHeadroom.borrowed = false;
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            var starts: [2]MemSnap = undefined;
            var grew: [2]u64 = undefined;
            for (0..2) |r| {
                const prompt = try a.alloc(u32, 3000);
                defer a.free(prompt);
                for (prompt, 0..) |*t, i| t.* = ids[(i * (r + 1) + r * 7) % ids.len];
                LateUnwireHeadroom.req = r;
                starts[r] = MemSnap.now();
                mlx.engine.resetPoolPeak(.process);
                var from: u32 = 0;
                var got = try sv.requestWith(prompt, 8, .{ .temperature = 0 }, false, &from);
                defer got.deinit(a);
                try testing.expectEqual(@as(usize, 8), got.items.len);
                // From what was in use at the start: a headroom read counts the pool's free buffers as free.
                grew[r] = mlx.engine.poolHeldPeak(.process) -| (starts[r].pool -| starts[r].pool_free);
            }
            // A start's free buffers are reused in other sizes than a cold start makes: its peak moves by up to them.
            const rest = @max(starts[0].pool_free, starts[1].pool_free);
            const in_use = [2]u64{ starts[0].pool -| starts[0].pool_free, starts[1].pool -| starts[1].pool_free };
            errdefer std.debug.print("first request: MLX cache {d} KB, pool in use {d} KB, grew {d} KB, pool rest {d} KB; second: MLX cache {d} KB, pool in use {d} KB, grew {d} KB, pool rest {d} KB; borrowed {}\n", .{ starts[0].mlx_cache >> 10, in_use[0] >> 10, grew[0] >> 10, starts[0].pool_free >> 10, starts[1].mlx_cache >> 10, in_use[1] >> 10, grew[1] >> 10, starts[1].pool_free >> 10, LateUnwireHeadroom.borrowed });
            // The late-unwiring borrow ran in the first request, and the request was served. Only a streamed load's
            // decoder holds expert-cache pools a prefill can borrow: a resident one never takes the borrow path.
            const state = &arch_handle.trunk(sv.entry.handle.?).arch_state;
            // A build without DeepSeek (`-Darchs`) has no streamed decoder that lends.
            const lends = if (comptime @hasField(@TypeOf(state.*), "dsv41")) (if (state.dsv41.stq_experts) |x| x.engine.borrow_from < x.engine.layers.len else false) else false;
            try testing.expectEqual(lends, LateUnwireHeadroom.borrowed);
            // A headroom read counts MLX's cache and the pool's free buffers as free: the first request starts as the second.
            try testing.expectEqual(starts[1].mlx_cache, starts[0].mlx_cache);
            // The warm leaves no more of the pool in use than a finished request does.
            try testing.expect(in_use[0] <= in_use[1]);
            try testing.expect(@max(grew[0], grew[1]) - @min(grew[0], grew[1]) <= rest);
        }
    }.run);
}

test "cooler contract: a resident load beside buffers already past a declared wired limit's floor is refused by the preflight, which names that limit" {
    try mlx.requireGpu();
    const dir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const bills = @import("bills.zig");
    const status = @import("status.zig");
    const registry_mod = @import("model_registry.zig");
    const scheduler_mod = @import("scheduler.zig");
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    defer mlx.fault.disarm();

    // A declared iogpu.wired_limit_mb (above 75% of RAM) whose floor, less the unplanned margin, is 1 GiB.
    const total = status.getTotalMemBytes();
    const mb_was = bills.wired_limit_mb_override;
    defer bills.wired_limit_mb_override = mb_was;
    bills.wired_limit_mb_override = (total - (1 << 30)) >> 20;
    const margin_was = bills.launch_state.wired_margin_flag_bytes;
    defer bills.launch_state.wired_margin_flag_bytes = margin_was;
    bills.launch_state.wired_margin_flag_bytes = total - (2 << 30);
    const floor_was = bills.gpu_floor_bytes.load(.monotonic);
    defer bills.gpu_floor_bytes.store(floor_was, .monotonic);

    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    @import("cooler_side.zig").forTest(&config, true);
    const floor = bills.ceilingNow().floor().?;
    try testing.expect(floor > 0 and floor <= 1 << 30);

    // 1.5 GiB already in use: past the declared floor, well under Metal's working set.
    const shape = [_]c_int{3 << 27};
    var held: mlx.mlx_array = .{ .ctx = null };
    try mlx.check(mlx.mlx_zeros(&held, &shape, 1, .float32, mlx.gpuStream()));
    defer _ = mlx.mlx_array_free(held);
    try mlx.check(mlx.mlx_array_eval(held));

    const reg = try registry_mod.ModelRegistry.init(a, io, null, 3, 0, null);
    defer reg.deinit();
    const entry = try reg.registerStub("past-floor", dir, 0);
    reg.mutex.lockUncancelable(io);
    try testing.expect(reg.tryBeginLoadLocked(entry));
    reg.mutex.unlock(io);
    var sch: scheduler_mod.Scheduler = undefined;
    sch.registry = reg;
    sch.io = io;
    sch.allocator = a;
    sch.ctx_size_flag = 0;
    var req: scheduler_mod.LoadRequest = .{ .entry = entry, .config = &config, .tok = undefined, .chat_config = undefined, .model_dir = dir, .evict_entries = &.{} };
    // Past the preflight the first checked MLX call is the weights': failing it ends an admitted load before it needs a
    // tokenizer.
    mlx.fault.arm(1);
    try testing.expect(!model_load.runLoadRequest(&sch, &req));
    const admitted = mlx.fault.didFire();
    mlx.fault.disarm();
    if (admitted) {
        var msg: [300]u8 = undefined;
        _ = mlx.takeError(&msg);
    }
    errdefer std.debug.print("a load beside {d} MB in use under a {d} MB floor ended {s}; the preflight admitted it: {}\n", .{ @import("gpu_mem.zig").activeBytes() >> 20, floor >> 20, req.outcome.errorName() orelse "loaded", admitted });
    try testing.expect(!admitted);
    try testing.expectEqualStrings("InsufficientMemory", req.outcome.errorName().?);
    try testing.expect(std.mem.indexOf(u8, req.refusal_text[0..req.refusal_text_len], "iogpu.wired_limit_mb = ") != null);
}

/// What a load's preflight saw of MLX's pool when it read the memory (the serving-bill hook runs inside it).
const PreflightSawMlx = struct {
    var mlx_cache: ?usize = null;
    fn bill(_: *const model_mod.ModelConfig, _: u64) ?model_load.LoadServingBill {
        var c: usize = 0;
        _ = mlx.mlx_get_cache_memory(&c);
        mlx_cache = c;
        // Past the eviction: the first checked MLX call from here is the new load's weights'.
        mlx.fault.arm(1);
        return null;
    }
};

/// A load of the tiny Qwen (on the engine or on MLX) that evicts the same pack loaded on MLX, failed at its first
/// weights call; returns what MLX's pool held when its preflight read the memory.
fn evictMlxModel(new_on_engine: bool) !usize {
    try mlx.requireGpu();
    const dir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const registry_mod = @import("model_registry.zig");
    const scheduler_mod = @import("scheduler.zig");
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    defer mlx.fault.disarm();
    _ = mlx.mlx_clear_cache();

    const reg = try registry_mod.ModelRegistry.init(a, io, null, 3, 0, null);
    defer reg.deinit();
    // The victim: the same pack loaded on MLX (the user's opt-out), as a served load leaves it on the heap.
    const victim = try reg.registerStub("victim", dir, 0);
    const config = try a.create(model_mod.ModelConfig);
    victim.config = config;
    {
        const off = model_mod.cooler_off;
        defer model_mod.cooler_off = off;
        model_mod.cooler_off = .full;
        config.* = try model_mod.parseConfig(io, a, dir);
        model_load.applyModelSettings(config, .{});
    }
    @import("cooler_side.zig").forTest(config, true);
    try testing.expect(!arch_handle.onEngine(null, config));
    const weights = try a.create(model_mod.Weights);
    weights.* = try model_mod.loadWeightsForConfig(io, a, dir, config, false);
    victim.weights = weights;
    const t = try a.create(Trunk);
    t.* = try Trunk.init(io, a, config.*, weights);
    victim.handle = arch_handle.adopt(t);
    var active_victim: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_victim);
    reg.mutex.lockUncancelable(io);
    reg.markReadyLocked(victim, active_victim);
    reg.markEvictingLocked(victim);
    reg.mutex.unlock(io);

    // A second id has a folder of its own (`FolderId`); the load reads the pack from `req.model_dir`.
    var other_tmp = testing.tmpDir(.{});
    defer other_tmp.cleanup();
    var ob: [std.fs.max_path_bytes]u8 = undefined;
    const other = try reg.registerStub("other", ob[0..try other_tmp.dir.realPath(io, &ob)], 0);
    reg.mutex.lockUncancelable(io);
    try testing.expect(reg.tryBeginLoadLocked(other));
    reg.mutex.unlock(io);
    var other_config = blk: {
        const off = model_mod.cooler_off;
        defer model_mod.cooler_off = off;
        if (!new_on_engine) model_mod.cooler_off = .full;
        var c = try model_mod.parseConfig(io, a, dir);
        model_load.applyModelSettings(&c, .{});
        break :blk c;
    };
    defer other_config.deinit(a);
    @import("cooler_side.zig").forTest(&other_config, true);
    try testing.expectEqual(new_on_engine, arch_handle.onEngine(null, &other_config));
    var victims = [_]*registry_mod.LoadedModel{victim};
    var sch: scheduler_mod.Scheduler = undefined;
    sch.registry = reg;
    sch.io = io;
    sch.allocator = a;
    sch.ctx_size_flag = 0;
    const hook_was = scheduler_mod.load_serving_bill;
    defer scheduler_mod.load_serving_bill = hook_was;
    scheduler_mod.load_serving_bill = &PreflightSawMlx.bill;
    PreflightSawMlx.mlx_cache = null;
    var req: scheduler_mod.LoadRequest = .{ .entry = other, .config = &other_config, .tok = undefined, .chat_config = undefined, .model_dir = dir, .evict_entries = &victims };
    try testing.expect(!model_load.runLoadRequest(&sch, &req));
    mlx.fault.disarm();
    var msg: [300]u8 = undefined;
    _ = mlx.takeError(&msg);
    try testing.expectEqual(registry_mod.LoadState.unloaded, victim.state);
    _ = mlx.mlx_clear_cache();
    return PreflightSawMlx.mlx_cache orelse error.TestUnexpectedResult;
}

test "cooler contract: a load that evicts a model on MLX reads its preflight with the victim's freed buffers given back, on the engine as on MLX" {
    for ([_]bool{ true, false }) |new_on_engine| {
        const parked = try evictMlxModel(new_on_engine);
        errdefer std.debug.print("a load {s} that evicted a model on MLX read its preflight beside {d} KB in MLX's pool\n", .{ if (new_on_engine) "on the engine" else "on MLX", parked >> 10 });
        try testing.expect(parked < 1 << 20);
    }
}
