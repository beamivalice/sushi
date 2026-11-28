//! The Sushi Cooler contract's request-level cells on a held load (`contract.heldLoad`): one served load per fixture
//! for the step, read back at rest after every test, and that read's own test. The runner and the shared helpers are in
//! `cooler_contract_test.zig`.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const arch_handle = @import("arch/handle.zig");
const test_support = @import("test_support.zig");
const arch_sched = @import("arch/sched.zig");
const io = std.Io.Threaded.global_single_threaded.io();
const contract = @import("cooler_contract_test.zig");
const heldLoad = contract.heldLoad;
const Load = contract.Load;
const Served = contract.Served;
const Sheet = contract.Sheet;
const eachArch = contract.eachArch;
const eachFixture = contract.eachFixture;
const eachLayout = contract.eachLayout;
const onTheCut = contract.onTheCut;
const firstAsks = contract.firstAsks;
const greedy = contract.greedy;
const Generator = @import("generate.zig").Generator;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const SamplingParams = @import("sampler.zig").SamplingParams;
const ContractOf = contract.ContractOf;
const qwenTinyServed = contract.qwenTinyServed;
const eachDraftingArch = contract.eachDraftingArch;
const servedOn = contract.servedOn;
const serial = contract.serial;
const heldRest = contract.heldRest;
const Group = @import("cooler_requests_test.zig").Group;
const capInForce = @import("cooler_requests_test.zig").capInForce;
const poolInUse = @import("cooler_requests_test.zig").poolInUse;

test "cooler contract: a wide forward that reads out every row's logits, as `sushi kld` runs one, runs under its cap" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            {
                var cfg = try contract.prepareConfig(C, dir, true);
                defer cfg.deinit(testing.allocator);
                // On the ledger a wide forward runs in the region its load measured (`hy4 plan: a prefill chunk of every
                // width ... runs in the region its load measured`); a `sushi kld` load is off the ledger and uncapped.
                try contract.requirePoolCap(&cfg);
            }
            const ld = try heldLoad(C, dir);
            const a = testing.allocator;
            for ([_]usize{ 418, 2048 }) |width| {
                const wide = try a.alloc(u32, width);
                defer a.free(wide);
                for (wide, 0..) |*t, i| t.* = ids[i % ids.len];
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                defer arch_sched.endRequest(&ld.config);
                try ld.xfm.resetCache();
                ld.xfm.bindOwnRows(width);
                const arr = mlx.mlx_array_new_data(@ptrCast(wide.ptr), &[_]c_int{ 1, @intCast(width) }, 2, .int32);
                defer _ = mlx.mlx_array_free(arr);
                var ctx = ld.xfm.defaultCtx();
                errdefer std.debug.print("cooler contract: a {d}-row forward reading out every row\n", .{width});
                const logits = try @import("model_handle.zig").forwardWith(ld.xfm, &ctx, arr);
                defer _ = mlx.mlx_array_free(logits);
                try mlx.check(mlx.mlx_array_eval(logits));
                const read: usize = if (@hasDecl(C, "reads_out_last_row_only")) 1 else width;
                try testing.expectEqual(@as(c_int, @intCast(read)), mlx.getShape(logits)[1]);
            }
        }
    }.run);
}

test "cooler contract: a decode after a long Cooler prefill grows the engine's pool by no more than its forward's cap" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const cooler = try heldLoad(C, dir);
            const a = testing.allocator;
            const len: usize = 4097;
            const long = try a.alloc(u32, len);
            defer a.free(long);
            for (long, 0..) |*t, i| t.* = ids[i % ids.len];
            const Body = struct {
                fn decode(ld: *Load, gen: *Generator) anyerror!void {
                    // From what the prefill left in use: each step's scratch, within a one-row forward's cap.
                    mlx.engine.close();
                    const in0 = Sheet.now().in_use;
                    mlx.engine.resetPoolPeak(.process);
                    var steps: usize = 0;
                    while (try gen.next(testing.allocator)) |_| steps += 1;
                    mlx.engine.close();
                    const grew = mlx.engine.poolHeldPeak(.process) -| in0;
                    const cap = @import("arch/cooler.zig").forwardCap(&ld.config, 1, len + steps);
                    if (grew > cap) std.debug.print("cooler contract: {d} decode steps after a {d}-token prefill grew the pool {d} KB over its {d} KB cap\n", .{ steps, len, grew >> 10, cap >> 10 });
                    try testing.expect(grew <= cap);
                }
            };
            try withPrefilled(cooler, long, 4, 4096, cooler, Body.decode);
        }
    }.run);
}

test "cooler contract: a decode after a wide forward that left free buffers of sizes it never asks for grows the engine's pool by no more than its forward's cap" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const cooler = @import("arch/cooler.zig");
            {
                var cfg = try contract.prepareConfig(C, dir, true);
                defer cfg.deinit(testing.allocator);
                // On the ledger a request ends with no free buffer in the pool and asks it nothing (`hy4 close`, and the
                // lines `hy4 ledger` sums after each request).
                try contract.requirePoolCap(&cfg);
                // An arch that installs its own hard caps refuses a buffer past them; its decode cap has its own cells.
                if (cooler.ownsCaps(&cfg)) {
                    @import("test_note.zig").notePartSkipped("an arch with its own pool caps");
                    return error.SkipZigTest;
                }
            }
            const ld = try heldLoad(C, dir);
            const a = testing.allocator;
            const wide_rows: usize = 2048;
            const wide = try a.alloc(u32, wide_rows);
            defer a.free(wide);
            for (wide, 0..) |*t, i| t.* = ids[i % ids.len];
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            defer arch_sched.endRequest(&ld.config);
            try ld.xfm.resetCache();
            ld.xfm.bindOwnRows(wide_rows + 1);
            const cap = cooler.forwardCap(&ld.config, 1, wide_rows + 1);
            {
                const arr = mlx.mlx_array_new_data(@ptrCast(wide.ptr), &[_]c_int{ 1, wide_rows }, 2, .int32);
                defer _ = mlx.mlx_array_free(arr);
                var ctx = ld.xfm.defaultCtx();
                ctx.skip_lm_head = true;
                const out = try @import("model_handle.zig").forwardWith(ld.xfm, &ctx, arr);
                defer _ = mlx.mlx_array_free(out);
                try mlx.check(mlx.mlx_array_eval(out));
            }
            // The wide forward's leftovers, and then some: a decode asks buffers far smaller than these, so none of
            // them serves it and what stays free past the lift sits on top of its own asks.
            {
                const parked_each: usize = 1 << 20;
                const n = cap / parked_each + 2;
                const held = try a.alloc(mlx.mlx_array, n);
                defer a.free(held);
                var made: usize = 0;
                defer for (held[0..made]) |h| {
                    _ = mlx.mlx_array_free(h);
                };
                while (made < n) : (made += 1) {
                    held[made] = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_zeros(&held[made], &[_]c_int{parked_each / 4}, 1, .float32, ld.xfm.s));
                    try mlx.check(mlx.mlx_array_eval(held[made]));
                }
            }
            {
                const one = mlx.mlx_array_new_data(@ptrCast(wide.ptr), &[_]c_int{ 1, 1 }, 2, .int32);
                defer _ = mlx.mlx_array_free(one);
                var ctx = ld.xfm.defaultCtx();
                // The decode forward begins: its model's wide cap is lifted and the free buffers past its cap go.
                // The window opens after that, so it holds the forward's own asks over what the lift left parked.
                _ = cooler.enterForward(&ld.config, &ctx, one, ld.xfm, false);
                mlx.engine.close();
                const in0 = Sheet.now().in_use;
                mlx.engine.resetPoolPeak(.process);
                const out = try @import("model_handle.zig").forwardWith(ld.xfm, &ctx, one);
                defer _ = mlx.mlx_array_free(out);
                try mlx.check(mlx.mlx_array_eval(out));
                mlx.engine.close();
                const grew = mlx.engine.poolHeldPeak(.process) -| in0;
                if (grew > cap) std.debug.print("cooler contract: a decode forward after a {d}-row one grew the pool {d} KB over its {d} KB cap\n", .{ wide_rows, grew >> 10, cap >> 10 });
                try testing.expect(grew <= cap);
            }
        }
    }.run);
}

test "cooler contract: a served cold prefill past the reservation floor, whose first chunk grows the request's whole reserved cache, runs under its chunks' caps" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            // The owner's agent turn, 40,388 tokens cold at 4,096 a chunk. A cut's one attention layer reserves the
            // context's whole rest to outgrow its first chunk's cap, as the real pack's twelve do at 256,000.
            const len: usize = 40388;
            const max_tokens: u32 = 1_000_000;
            {
                var cfg = try contract.prepareConfig(C, dir, true);
                defer cfg.deinit(a);
                if (!cfg.reservesKvCapacity() or cfg.max_position_embeddings < len + max_tokens) return error.SkipZigTest;
            }
            const ld = try heldLoad(C, dir);
            const long = try a.alloc(u32, len);
            defer a.free(long);
            for (long, 0..) |*t, i| t.* = ids[i % ids.len];
            const Body = struct {
                fn reserved(l: *Load, _: *Generator) anyerror!void {
                    try testing.expect(l.xfm.cache.reserve_tokens >= len + max_tokens);
                }
            };
            errdefer std.debug.print("cooler contract: a cold {d}-token prefill at 4096 rows a chunk reserving {d} more\n", .{ len, max_tokens });
            try withPrefilled(ld, long, max_tokens, 4096, ld, Body.reserved);
        }
    }.run);
}

test "cooler contract: a Cooler request refused at any pool buffer site fails as TransientCapExceeded once the scheduler settles it, leaving no latch, and every later request on the same load says the same tokens" {
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer seen.deinit(testing.allocator);
    const S = struct {
        var sites: *std.AutoHashMapUnmanaged(u64, void) = undefined;
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            defer mlx.engine.fault.disarm();
            const ld = try heldLoad(C, dir);
            // A speculative round's plan follows measured costs and live sync times, so its draft count, and the asks
            // with it, change from request to request; pinned (as --mtp-min-depth 3 --mtp-max-depth 3) every request
            // makes the same asks and the sweep proves each one.
            const depth_bounds = @import("mtp_depth_bounds.zig");
            const bounds = depth_bounds.active;
            defer depth_bounds.active = bounds;
            depth_bounds.active = .{ .min = 3, .max = 3 };
            for ([_]usize{ 1, 9 }) |plen| for ([_]bool{ false, true }) |spec| {
                if (spec and !@hasDecl(C, "specTokens")) continue;
                const prompt = ids[0..@min(ids.len, plen)];
                var want = try requestOf(C, ld, prompt, 4, spec);
                defer want.deinit(a);
                const Req = struct {
                    ld: *Load,
                    prompt: []const u32,
                    spec: bool,
                    fn go(q: @This()) anyerror!void {
                        var got = try requestOf(C, q.ld, q.prompt, 4, q.spec);
                        got.deinit(testing.allocator);
                    }
                };
                // One clean request records every pool buffer's site; each is refused at its first ask, on the first arm
                // and fixture of the arch that reaches it. A speculative request's asks do not come in the same order
                // every time, so a point is a site's n-th ask.
                const points = try firstAsks(a, sites, C, Req{ .ld = ld, .prompt = prompt, .spec = spec }, Req.go);
                defer a.free(points);
                const base = Sheet.now();
                for (points, 0..) |pt, at| {
                    errdefer std.debug.print("cooler contract: a {s} request of {d} tokens refused at point {d} (a site's ask {d})\n", .{ if (spec) "speculative" else "plain", prompt.len, at, pt.n });
                    mlx.engine.fault.armAt(pt);
                    const r = requestOf(C, ld, prompt, 4, spec);
                    const fired = !mlx.engine.fault.armed();
                    mlx.engine.fault.disarm();
                    if (r) |got| {
                        var g = got;
                        g.deinit(a);
                    } else |e| try testing.expectEqualStrings("TransientCapExceeded", mlx.settledErrorName(e));
                    try testing.expect(!mlx.errorPending());
                    if (!fired and pt.strict) {
                        std.debug.print("cooler contract: a {s} request never made ask {d} of a site its recording made\n", .{ if (spec) "speculative" else "plain", pt.n });
                        return error.TestUnexpectedResult;
                    }
                    var after = try requestOf(C, ld, prompt, 4, spec);
                    defer after.deinit(a);
                    try testing.expectEqualSlices(u32, want.items, after.items);
                    try testing.expectEqual(base, Sheet.now());
                }
            };
        }
    };
    S.sites = &seen;
    try eachLayout(S.run);
}

test "cooler contract: a Cooler prefill hands each chunk's free buffers back, so a long prompt holds no more free than one chunk" {
    const Q = ContractOf("qwen4");
    if (comptime @hasDecl(Q, "prompt")) return prefillHandsChunksBack(Q);
    return error.SkipZigTest;
}

test "cooler contract: a cache cut to its rows at a request's end holds one buffer's copy past itself, as its bill prices, and gives the reservation back" {
    const Q = ContractOf("qwen4");
    if (comptime @hasDecl(Q, "prompt")) return commitCutHoldsOneBuffer(Q);
    return error.SkipZigTest;
}

test "cooler contract: a served speculative request refused at any pool buffer site its plain request lacks leaves no dispatch open, leaves the pool as it was, and the next says its tokens" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            const depth_bounds = @import("mtp_depth_bounds.zig");
            const saved = depth_bounds.active;
            defer depth_bounds.active = saved;
            depth_bounds.active = .{ .min = 3, .max = 3 };
            var p: [40]u32 = undefined;
            for (&p, 0..) |*t, i| t.* = ids[(i * 5 + 1) % ids.len];
            var from: u32 = 0;
            var want = try sv.requestWith(&p, 6, .{ .temperature = 0 }, true, &from);
            defer want.deinit(a);
            const Req = struct {
                sv: *Served,
                p: []const u32,
                mtp: bool,
                fn go(q: @This()) anyerror!void {
                    _ = try q.sv.requestToCleanup(q.p, 6, q.mtp);
                }
            };
            // A round's verify holds what its rollback needs on the request's own state, and its forward asks at
            // the sites a prefill chunk does: eight frames tell the two apart, whose unwinding differs above it.
            const sites = &mlx.engine.fault.sites;
            defer sites.* = .{};
            sites.* = .{ .lasts = false, .frames = 8 };
            const plain = try mlx.sweepSites(a, sites, Req{ .sv = &sv, .p = &p, .mtp = false }, Req.go);
            defer a.free(plain);
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(a);
            for (plain) |pt| try seen.put(a, pt.site, {});
            const all = try mlx.sweepSites(a, sites, Req{ .sv = &sv, .p = &p, .mtp = true }, Req.go);
            defer a.free(all);
            const base = Sheet.now();
            for (all, 0..) |pt, at| {
                if (seen.contains(pt.site)) continue;
                errdefer std.debug.print("cooler contract: a speculative request refused at pool point {d} of {d} (ask {d} of its site)\n", .{ at, all.len, pt.n });
                mlx.engine.fault.armAt(pt);
                const open = try sv.requestToCleanup(&p, 6, true);
                const fired = !mlx.engine.fault.armed();
                mlx.engine.fault.disarm();
                try testing.expect(!mlx.errorPending());
                try testing.expect(fired or !pt.strict);
                try testing.expectEqual(@as(u32, 0), open);
                try testing.expect(Sheet.now().in_use <= base.in_use);
                var got = try sv.requestWith(&p, 6, .{ .temperature = 0 }, true, &from);
                defer got.deinit(a);
                try testing.expectEqualSlices(u32, want.items, got.items);
                const now = Sheet.now();
                try testing.expectEqual(base.in_use, now.in_use);
                try testing.expectEqual(@as(u32, 0), now.open);
            }
        }
    }.run);
}

test "cooler contract: a prompt token id outside the model's vocabulary is refused before any forward reads past the embedding" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const cooler = try heldLoad(C, dir);
            const past: u32 = @intCast(cooler.config.vocab_size);
            try testing.expectError(error.PromptTokenOutOfVocab, greedy(cooler, &.{ ids[0], past }, 1));
        }
    }.run);
}

test "cooler contract held loads: a held load on the ledger holds its whole budget, so the next held load of any fixture opens once it is unloaded" {
    const H = ContractOf("hy4flash");
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(H, "prompt") or !@hasDecl(Q, "prompt")) return error.SkipZigTest;
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const envs = [_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY", "QWEN4_STQ_TINY" };
    var dirs: [envs.len][]const u8 = undefined;
    for (envs, &dirs) |env, *dir| {
        if (!contract.fixtureChosen(contract.selection(), env)) return error.SkipZigTest;
        dir.* = std.mem.span(std.c.getenv(env) orelse return error.SkipZigTest);
    }
    // Each opens beside the one before and, once the ledger's has gone, serves a request.
    const Open = struct {
        fn serves(comptime C: type, dir: []const u8) !void {
            const ids = try C.prompt(testing.allocator, dir);
            defer testing.allocator.free(ids);
            const ld = heldLoad(C, dir) catch |e| {
                std.debug.print("cooler contract: the held load of {s} beside the held loads before it: {s}\n", .{ std.fs.path.basename(dir), @errorName(e) });
                return e;
            };
            var got = try greedy(ld, ids[0..9], 3);
            got.deinit(testing.allocator);
        }
    };
    try Open.serves(H, dirs[0]);
    try Open.serves(H, dirs[1]);
    try Open.serves(Q, dirs[2]);
}

test "cooler contract: a Cooler image request refused at any pool buffer site a text request lacks frees what it made, the embedding its rows splice into too" {
    try servedOn("qwen4", &.{"QWEN4_BF16_TINY"}, struct {
        const Req = struct {
            ld: *Load,
            prompt: []const u32,
            fn go(q: @This()) anyerror!void {
                var got = try greedy(q.ld, q.prompt, 2);
                got.deinit(testing.allocator);
            }
        };

        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            defer mlx.engine.fault.disarm();
            // A cut fixture's vocabulary may end before its image placeholder: no prompt can carry one.
            {
                var cfg = try contract.prepareConfig(Q, dir, true);
                defer cfg.deinit(a);
                if (cfg.image_token_id == 0 or cfg.image_token_id >= cfg.vocab_size) return error.SkipZigTest;
            }
            const ld = try heldLoad(Q, dir);
            var prompt: [12]u32 = undefined;
            for (&prompt, 0..) |*t, i| t.* = if (i >= 3 and i < 7) ld.config.image_token_id else ids[i % ids.len];
            {
                // The rows are the cell's own, not a served request's state (the tower's encode writes them).
                const turn = test_support.openTurn(ld.h, &ld.config);
                defer turn.end();
                const scope = mlx.engine.enter(ld.xfm.s);
                defer scope.end();
                var rows = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(rows);
                try mlx.check(mlx.mlx_zeros(&rows, &[_]c_int{ 1, 4, @intCast(ld.config.hidden_size) }, 3, .bfloat16, ld.xfm.s));
                try mlx.check(mlx.mlx_array_eval(rows));
                ld.xfm.vision_embeddings = rows;
            }
            defer {
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                _ = mlx.mlx_array_free(ld.xfm.vision_embeddings.?);
                ld.xfm.vision_embeddings = null;
            }
            // A text request's sites are the request sweep's; this one refuses those the image's rows add.
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(a);
            var text: [12]u32 = undefined;
            for (&text, 0..) |*t, i| t.* = ids[i % ids.len];
            a.free(try firstAsks(a, &seen, Q, Req{ .ld = ld, .prompt = &text }, Req.go));
            const points = try firstAsks(a, &seen, Q, Req{ .ld = ld, .prompt = &prompt }, Req.go);
            defer a.free(points);
            if (points.len == 0) {
                @import("test_note.zig").notePartSkipped("an image request with no pool buffer site of its own");
                return error.SkipZigTest;
            }
            const base = Sheet.now();
            var msg: [300]u8 = undefined;
            for (points, 0..) |pt, at| {
                mlx.engine.fault.armAt(pt);
                const r = Req.go(.{ .ld = ld, .prompt = &prompt });
                const fired = !mlx.engine.fault.armed();
                mlx.engine.fault.disarm();
                if (r) |_| try testing.expect(!mlx.errorPending()) else |_| _ = mlx.takeError(&msg);
                errdefer std.debug.print("cooler contract: an image request refused at point {d} of {d} left {any} -> {any}\n", .{ at, points.len, base, Sheet.now() });
                try testing.expect(fired or !pt.strict);
                // A request leaves the load's caches as it found them only once one runs whole after the refusal.
                try Req.go(.{ .ld = ld, .prompt = &prompt });
                try testing.expectEqual(base, Sheet.now());
            }
        }
    }.run);
}

test "cooler contract: a Cooler load's /props names Sushi Cooler as the engine its forwards run on" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            _ = ids;
            const ld = try heldLoad(C, dir);
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            try testing.expectEqualStrings("sushi-cooler", @import("models_api.zig").propsSettingsFor(sv.entry).engine);
        }
    }.run);
}

test "cooler contract: two unseeded sampled requests under one sampler decoded at once serve, plain and speculative, each row drawn under its own request's key (#106)" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld_own: Load = undefined;
            try ld_own.openLight(Q, dir, true);
            defer _ = ld_own.close();
            const ld = &ld_own;
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            var long: [120]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
            // Rows that sample alike with no seed draw as one: the issue's sampler, temperature alone, and min-p.
            for ([_]SamplingParams{ .{ .temperature = 0.6, .top_p = 0.95, .top_k = 20 }, .{ .temperature = 0.8 }, .{ .temperature = 0.7, .min_p = 0.05 } }) |sp| {
                for ([_]bool{ false, true }) |mtp| try sv.concurrent(&.{ long[0..40], long[0..101] }, 12, sp, mtp);
            }
        }
    }.run);
}

test "cooler contract: a grouped tick after another request's prefill chunk runs under no cap that chunk installed, and both requests serve, plain (#106)" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld_own: Load = undefined;
            try ld_own.openLight(Q, dir, true);
            defer _ = ld_own.close();
            const ld = &ld_own;
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            var long: [300]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[(i * 3 + 1) % ids.len];
            try groupedAfterPrefill(&sv, &long, false);
        }
    }.run);
}

test "cooler contract: four requests, three speculative and one plain, the last prefilled beside three decoding, tick under no cap a prefill chunk installed, at every tick and every turn of the model after it, and all serve (#106)" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld_own: Load = undefined;
            try ld_own.openLight(Q, dir, true);
            defer _ = ld_own.close();
            const ld = &ld_own;
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            sv.sch.queue_cap = 4;
            var long: [101]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[(i * 3 + 1) % ids.len];
            // The issue's widths: every speculative row at depth 2. The plain row's tick joins the speculative rounds.
            try joinedAfterPrefill(&sv, &.{ long[0..40], long[0..60], long[0..80], long[0..101] }, &.{ true, true, true, false }, 12);
        }
    }.run);
}

test "cooler contract: three speculative requests, one restored from its last turn and prefilled beside two decoding, tick under no cap that prefill installed, at every tick and every turn of the model after it, and all serve (#112)" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld_own: Load = undefined;
            try ld_own.openLight(Q, dir, true);
            defer _ = ld_own.close();
            const ld = &ld_own;
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            sv.sch.queue_cap = 3;
            const a = testing.allocator;
            var conv: std.ArrayList(u32) = .empty;
            defer conv.deinit(a);
            for (0..1500) |i| try conv.append(a, ids[(i * 7 + i / ids.len) % ids.len]);
            var f: u32 = 0;
            var reply = try sv.request(conv.items, 16, &f);
            defer reply.deinit(a);
            try conv.appendSlice(a, reply.items);
            for (0..200) |i| try conv.append(a, ids[(i * 13 + 101) % ids.len]);
            // Rows of like lengths, as the issue's long sessions: the planner groups them only within its pad waste.
            var other: [1600]u32 = undefined;
            for (&other, 0..) |*t, i| t.* = ids[(i * 3 + 1) % ids.len];
            try joinedAfterPrefill(&sv, &.{ other[0..1500], other[0..1600], conv.items }, &.{ true, true, true }, 24);
            try testing.expect(sv.reused > 0);
        }
    }.run);
}

test "cooler contract: a Cooler load's sampled speculative requests, seeded or not, serve and ask nothing of MLX" {
    try eachDraftingArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specTokensWith")) return error.SkipZigTest;
            const a = testing.allocator;
            const cooler = try heldLoad(C, dir);
            const cases = [_]SamplingParams{
                .{ .temperature = 0.8, .top_p = 0.95 },
                .{ .temperature = 0.8, .top_p = 0.95, .seed = 7 },
                .{ .temperature = 1.0, .top_k = 20 },
            };
            // The sampler's draw follows no prompt length: each length's forward is the greedy cell's.
            const prompt = ids[0..@min(ids.len, 9)];
            for (cases) |sampling| {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                const calls = mlx.engine.mlxCalls();
                // A fixture with no drafter skips, which is no failure to name.
                var skipped = false;
                errdefer if (!skipped) std.debug.print("cooler contract: sampled speculative request over {d} tokens, seed {?d}, top_k {d}\n", .{ prompt.len, sampling.seed, sampling.top_k });
                var got = C.specTokensWith(a, cooler.xfm, prompt, 12, sampling) catch |e| {
                    skipped = e == error.SkipZigTest;
                    return e;
                };
                defer got.deinit(a);
                try testing.expectEqual(@as(usize, 12), got.items.len);
                try testing.expectEqual(calls, mlx.engine.mlxCalls());
            }
        }
    }.run);
}

test "cooler contract: a seeded sampled speculative request says the seeded serial request's tokens, again after another request and at every pinned depth" {
    try eachDraftingArch(struct {
        fn spec(ld: *Load, comptime C: type, prompt: []const u32, n: u32, sampling: SamplingParams, depth: u32) !std.ArrayList(u32) {
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            defer arch_sched.endRequest(&ld.config);
            return C.specTokensAt(testing.allocator, ld.xfm, prompt, n, sampling, depth);
        }

        fn differs(what: []const u8, depth: u32, sampling: SamplingParams, len: usize, want: []const u32, got: []const u32) bool {
            const at = std.mem.indexOfDiff(u32, want, got) orelse return false;
            std.debug.print("cooler contract: seed {?d} over {d} tokens: {s} (pinned depth {d}) differs from token {d}\n", .{ sampling.seed, len, what, depth, at });
            return true;
        }

        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specTokensAt") or !@hasDecl(C, "specIsGreedy")) return error.SkipZigTest;
            const a = testing.allocator;
            // Verify rows another plan computes at each row count give no position one token.
            {
                var cfg = try contract.prepareConfig(C, dir, true);
                defer cfg.deinit(a);
                if (!C.specIsGreedy(&cfg)) return error.SkipZigTest;
            }
            const ld = try heldLoad(C, dir);
            const depth_bounds = @import("mtp_depth_bounds.zig");
            const saved = depth_bounds.active;
            defer depth_bounds.active = saved;
            var long: [128]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[(i * 7 + i / ids.len) % ids.len];
            const cases = [_]SamplingParams{
                .{ .temperature = 0.8, .top_k = 20, .seed = 42 },
                .{ .temperature = 1.0, .top_p = 0.95, .top_k = 40, .min_p = 0.05, .seed = 11 },
            };
            var bad: u32 = 0;
            const len = long.len;
            const prompt: []const u32 = &long;
            for (cases, 0..) |sampling, ci| {
                var want = try serial(ld, prompt, 20, sampling);
                defer want.deinit(a);
                var first = try spec(ld, C, prompt, 20, sampling, 0);
                defer first.deinit(a);
                if (differs("the speculative request against the serial one", 0, sampling, len, want.items, first.items)) bad += 1;
                if (ci > 0) continue;
                // Another request between publishes the acceptance the next one plans from.
                {
                    var other = try spec(ld, C, long[0..64], 20, .{ .temperature = 0 }, 0);
                    other.deinit(a);
                    var again = try spec(ld, C, prompt, 20, sampling, 0);
                    defer again.deinit(a);
                    if (differs("the speculative request after another against the first", 0, sampling, len, first.items, again.items)) bad += 1;
                }
                for ([_]u32{ 1, 3, 6 }) |depth| {
                    depth_bounds.active = .{ .min = depth, .max = depth };
                    defer depth_bounds.active = saved;
                    var pinned = try spec(ld, C, prompt, 20, sampling, depth);
                    defer pinned.deinit(a);
                    if (differs("the speculative request against the first", depth, sampling, len, first.items, pinned.items)) bad += 1;
                }
            }
            try testing.expectEqual(@as(u32, 0), bad);
        }
    }.run);
}

test "cooler contract: a greedy request's thought closed by its budget says the serial request's tokens, whether the close lands on a serial tick or a speculative round's exit" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specOptions") or !@hasDecl(C, "specIsGreedy")) return error.SkipZigTest;
            const a = testing.allocator;
            {
                var cfg = try contract.prepareConfig(C, dir, true);
                defer cfg.deinit(a);
                if (!C.specIsGreedy(&cfg)) return error.SkipZigTest;
            }
            const ld = try heldLoad(C, dir);
            const depth_bounds = @import("mtp_depth_bounds.zig");
            const saved = depth_bounds.active;
            defer depth_bounds.active = saved;
            // As wide as a served close (an early-stop line, the closer, a separator), past the verify rows.
            var forced: [24]u32 = undefined;
            for (&forced, 0..) |*t, i| t.* = ids[(i * 5 + 3) % ids.len];
            forced[forced.len - 2] = @intCast(ld.config.vocab_size - 1);
            var bad: u32 = 0;
            for ([_]u32{ 3, 10 }) |budget| {
                const n = budget + 28;
                var want = try BoundRun.run(ld, C, ids, n, budget, &forced, .serial);
                defer want.deinit(a);
                try testing.expect(!want.at_exit);
                const modes = [_]BoundRun.Mode{ .drained, .{ .mtp = 1 }, .{ .mtp = 3 }, .{ .mtp = 6 } };
                for (modes) |mode| {
                    if (mode == .mtp) depth_bounds.active = .{ .min = mode.mtp, .max = mode.mtp };
                    defer depth_bounds.active = saved;
                    var got = try BoundRun.run(ld, C, ids, n, budget, &forced, mode);
                    defer got.deinit(a);
                    if (mode == .drained) try testing.expect(got.at_exit);
                    const tokens_at = std.mem.indexOfDiff(u32, want.tokens.items, got.tokens.items);
                    const logits_same = std.mem.eql(u8, std.mem.sliceAsBytes(want.logits), std.mem.sliceAsBytes(got.logits));
                    if (tokens_at == null and logits_same) continue;
                    bad += 1;
                    const depth = if (mode == .mtp) mode.mtp else 0;
                    const flight = if (got.at_exit) "no token in flight" else "a token in flight";
                    const logits = if (logits_same) "equal" else "differ";
                    std.debug.print("cooler contract: budget {d}, {s} depth {d}, closed with {s}: tokens differ from {?d}, logits after the close {s}\n", .{ budget, @tagName(mode), depth, flight, tokens_at, logits });
                }
            }
            try testing.expectEqual(@as(u32, 0), bad);
        }
    }.run);
}

test "cooler contract: an exact Cooler speculative request says its greedy tokens, over one row, past the verify edge and a chunk" {
    try eachLayout(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specIsGreedy")) return error.SkipZigTest;
            const a = testing.allocator;
            {
                var cfg = try contract.prepareConfig(C, dir, true);
                defer cfg.deinit(a);
                if (!C.specIsGreedy(&cfg)) return error.SkipZigTest;
            }
            const ld = try heldLoad(C, dir);
            for ([_]usize{ 1, 9, 40 }) |len| {
                const prompt = ids[0..@min(ids.len, len)];
                errdefer std.debug.print("cooler contract: a prompt of {d} tokens\n", .{prompt.len});
                var g = try greedy(ld, prompt, 12);
                defer g.deinit(a);
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                defer arch_sched.endRequest(&ld.config);
                var sp = try C.specTokens(a, ld.xfm, prompt, 12);
                defer sp.deinit(a);
                try testing.expectEqualSlices(u32, g.items, sp.items);
            }
        }
    }.run);
}

test "cooler contract: a plain request, or one near its token budget, beside long speculative rows leaves them the planner's speculation" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld_own: Load = undefined;
            try ld_own.openLight(Q, dir, true);
            defer _ = ld_own.close();
            const ld = &ld_own;
            var bufs: [3][120]u32 = undefined;
            for (&bufs, 0..) |*b, r| for (b, 0..) |*t, i| {
                t.* = ids[(i * (r + 3) + 5 * r + 1) % ids.len];
            };
            var solo0 = try Group.solo(ld, dir, .{ .prompt = bufs[0][0..23], .max = 128, .mtp = false });
            defer solo0.deinit(testing.allocator);
            var solo1 = try Group.solo(ld, dir, .{ .prompt = bufs[1][0..37], .max = 128, .mtp = false });
            defer solo1.deinit(testing.allocator);
            // A speculative third row is the control: the long rows speculate beside it. A plain third row cannot
            // speculate, and a short-budget one is always under the 96 tokens a calibration round needs.
            const thirds = [_]Group.Req{
                .{ .prompt = bufs[2][0..51], .max = 128 },
                .{ .prompt = bufs[2][0..51], .max = 128, .mtp = false },
            };
            for (thirds, 0..) |third, ti| {
                var sv: Served = undefined;
                try sv.init(ld, dir);
                defer sv.deinit();
                const set = [_]Group.Req{ .{ .prompt = bufs[0][0..23], .max = 128 }, .{ .prompt = bufs[1][0..37], .max = 128 }, third };
                var got = try Group.run(&sv, &set);
                defer got.deinit();
                errdefer std.debug.print("cooler contract: third row {s}: planner plain/prime/spec {any}\n", .{ if (ti == 0) "speculative" else "plain", got.planner[0..3] });
                try testing.expect(got.planner[0][2] > 0);
                try testing.expect(got.planner[1][2] > 0);
                try expectCachedSolo(&solo0, &solo1, &got);
                const tgot = Group.Out{ .published = .{ got.published[2], .empty, .empty, .empty, .empty, .empty, .empty, .empty }, .failed = .{ got.failed[2], null, null, null, null, null, null, null } };
                try Group.expectSolo(ld, dir, set[2..3], &tgot);
            }
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            var shorts: [8]Group.Req = undefined;
            shorts[0] = .{ .prompt = bufs[0][0..23], .max = 128 };
            shorts[1] = .{ .prompt = bufs[1][0..37], .max = 128 };
            for (shorts[2..], 0..) |*r, k| r.* = .{ .prompt = bufs[2][k .. 40 + k], .max = 40, .at_tick = @intCast(k * 40) };
            var got = try Group.run(&sv, &shorts);
            defer got.deinit();
            errdefer std.debug.print("cooler contract: short-budget rows beside: planner plain/prime/spec {any}\n", .{got.planner[0..2]});
            try testing.expect(got.planner[0][2] > 0);
            try testing.expect(got.planner[1][2] > 0);
            try expectCachedSolo(&solo0, &solo1, &got);
        }
    }.run);
}

test "cooler contract: a request that cannot join the planner's tick leaves the rows that can their speculation" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld_own: Load = undefined;
            try ld_own.openLight(Q, dir, true);
            defer _ = ld_own.close();
            const ld = &ld_own;
            var bufs: [3][700]u32 = undefined;
            for (&bufs, 0..) |*b, r| for (b, 0..) |*t, i| {
                t.* = ids[(i * (r + 3) + 5 * r + 1) % ids.len];
            };
            var solo0 = try Group.solo(ld, dir, .{ .prompt = bufs[0][0..23], .max = 128, .mtp = false });
            defer solo0.deinit(testing.allocator);
            var solo1 = try Group.solo(ld, dir, .{ .prompt = bufs[1][0..37], .max = 128, .mtp = false });
            defer solo1.deinit(testing.allocator);
            // Each third row has a reason of its own to decode outside the planner: a penalty its plain tick does not
            // sample, a position table it does not read, a context so much longer than
            // its neighbours' that padding them to it wastes the verify.
            const thirds = [_]struct { name: []const u8, req: Group.Req }{
                .{ .name = "penalty", .req = .{ .prompt = bufs[2][0..51], .max = 64, .sampling = .{ .temperature = 0, .presence_penalty = 0.5 } } },
                .{ .name = "mrope", .req = .{ .prompt = bufs[2][0..51], .max = 64, .mrope = true } },
                .{ .name = "pad waste", .req = .{ .prompt = bufs[2][0..650], .max = 64 } },
            };
            var bad: u32 = 0;
            for (thirds) |third| {
                var sv: Served = undefined;
                try sv.init(ld, dir);
                defer sv.deinit();
                const set = [_]Group.Req{ .{ .prompt = bufs[0][0..23], .max = 128 }, .{ .prompt = bufs[1][0..37], .max = 128 }, third.req };
                var got = try Group.run(&sv, &set);
                defer got.deinit();
                // Every row says its solo tokens: the two that group, and the third that decodes in the regular path.
                if (got.planner[0][2] == 0 or got.planner[1][2] == 0) {
                    std.debug.print("cooler contract: a {s} row beside: planner plain/prime/spec {any}\n", .{ third.name, got.planner[0..3] });
                    bad += 1;
                }
                expectCachedSolo(&solo0, &solo1, &got) catch |e| {
                    std.debug.print("cooler contract: a {s} row beside: {s}\n", .{ third.name, @errorName(e) });
                    bad += 1;
                };
                const tgot = Group.Out{ .published = .{ got.published[2], .empty, .empty, .empty, .empty, .empty, .empty, .empty }, .failed = .{ got.failed[2], null, null, null, null, null, null, null } };
                Group.expectSolo(ld, dir, set[2..3], &tgot) catch |e| {
                    std.debug.print("cooler contract: a {s} row beside: {s}\n", .{ third.name, @errorName(e) });
                    bad += 1;
                };
            }
            try testing.expectEqual(@as(u32, 0), bad);
        }
    }.run);
}

test "cooler contract: a served request on a load that holds its drafter drafts, greedy and seeded" {
    try eachDraftingArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(C, dir);
            // A load with no drafter has nothing to draft with.
            const head = @import("generate.zig").MtpHeadRef.fromModel(ld.xfm, true, null) orelse return error.SkipZigTest;
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            for ([_]SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0.8, .top_k = 20, .seed = 42 } }) |sp| {
                var drafted: u64 = 0;
                var got = try contract.servedDrafts(&sv, ids[0..@min(ids.len, 30)], 16, sp, head, &drafted);
                defer got.deinit(a);
                errdefer std.debug.print("cooler contract: a served request at temperature {d} drafted nothing over {d} tokens\n", .{ sp.temperature, got.items.len });
                try testing.expect(drafted > 0);
            }
        }
    }.run);
}

test "cooler contract: the decode meter's prefill, whose chunks nobody reads, runs on Sushi Cooler as a served prefill does" {
    try onTheCut(struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const ld = try heldLoad(C, dir);
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            defer arch_sched.endRequest(&ld.config);
            const tokens: usize = 4096;
            try ld.xfm.resetCache();
            ld.xfm.bindOwnRows(tokens + 1);
            var ctx = ld.xfm.defaultCtx();
            mlx.engine.close();
            const in0 = Sheet.now().in_use;
            mlx.engine.resetPoolPeak(.process);
            try testing.expectEqual(tokens, try @import("model_load.zig").meterPrefill(testing.allocator, ld.xfm, &ctx, tokens));
            mlx.engine.close();
            // A served chunk reads out no logits, so no chunk holds more than its forward's cap beyond the request
            // state the prefill commits (the cache it grew), which a cap counts as in use.
            const cap = @import("arch/cooler.zig").forwardCap(&ld.config, 2048, tokens);
            const committed = Sheet.now().in_use -| in0;
            const scratch = mlx.engine.poolHeldPeak(.process) -| in0 -| committed;
            if (scratch > cap) std.debug.print("cooler contract: the decode meter's {d}-token prefill held {d} KB of scratch over its chunks' {d} KB cap\n", .{ tokens, scratch >> 10, cap >> 10 });
            try testing.expect(scratch <= cap);
        }
    }.run);
}

test "cooler contract: on real text, joined decode rows say their solo rows' tokens for a long chunked prompt beside a short one (cut)" {
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(Q, "prompt")) return error.SkipZigTest;
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    errdefer {
        var msg: [512]u8 = undefined;
        std.debug.print("cooler contract: fixture {s}: {s}\n", .{ dir, mlx.takeError(&msg) orelse "no MLX or engine error latched" });
    }
    const a = testing.allocator;
    const ids = try realTextIds(a, dir, 4200);
    defer a.free(ids);
    contract.unloadHeldDir(dir);
    var ld_own: Load = undefined;
    try ld_own.openLight(Q, dir, true);
    defer _ = ld_own.close();
    const ld = &ld_own;
    // A long chunked row past the two-pass context beside a short one; three short rows are the tiny's.
    const sets = [_][]const usize{&.{ 4097, 17 }};
    for (sets) |lens| {
        var prompts: [8][]const u32 = undefined;
        for (lens, 0..) |l, i| prompts[i] = ids[i * 7 .. i * 7 + l];
        const got = blk: {
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            defer arch_sched.endRequest(&ld.config);
            break :blk try Q.greedyRows(a, ld.xfm, prompts[0..lens.len], 40);
        };
        defer {
            for (got) |*g| g.deinit(a);
            a.free(got);
        }
        for (prompts[0..lens.len], got) |p, row| {
            const one = blk: {
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                defer arch_sched.endRequest(&ld.config);
                break :blk try Q.greedyRows(a, ld.xfm, &.{p}, 40);
            };
            defer {
                for (one) |*g| g.deinit(a);
                a.free(one);
            }
            errdefer std.debug.print("cooler contract: rows {any}: the row of {d} tokens differs from its solo run at {?d}\n", .{ lens, p.len, std.mem.indexOfDiff(u32, one[0].items, row.items) });
            try testing.expectEqualSlices(u32, one[0].items, row.items);
        }
    }
}

test "cooler contract: on real text, greedy speculative at depths 1, 3 and 6 says greedy's tokens, short and past the two-pass context (cut)" {
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(Q, "prompt")) return error.SkipZigTest;
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    errdefer {
        var msg: [512]u8 = undefined;
        std.debug.print("cooler contract: fixture {s}: {s}\n", .{ dir, mlx.takeError(&msg) orelse "no MLX or engine error latched" });
    }
    const a = testing.allocator;
    const ids = try realTextIds(a, dir, 4097);
    defer a.free(ids);
    const ld = try heldLoad(Q, dir);
    // Below and past the 2048 context where verify takes the two-pass kernel, the second a chunked prefill; the short
    // prompt arms are the tiny's.
    for ([_]usize{ 17, 4097 }) |len| {
        var greedy_said = try greedyOn(ld, ids[0..len], 24);
        defer greedy_said.deinit(a);
        const depths: []const u32 = if (len == 17) &.{ 1, 6 } else &.{ 1, 3, 6 };
        // The shallowest and deepest round and one between: each depth re-prefills the prompt.
        for (depths) |depth| {
            var spec = blk: {
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                defer arch_sched.endRequest(&ld.config);
                break :blk try Q.specTokensAt(a, ld.xfm, ids[0..len], 24, .{ .temperature = 0.0 }, depth);
            };
            defer spec.deinit(a);
            errdefer std.debug.print("cooler contract: real text of {d} tokens at depth {d}: speculative differs from greedy at {?d}\n", .{ len, depth, std.mem.indexOfDiff(u32, greedy_said.items, spec.items) });
            try testing.expectEqualSlices(u32, greedy_said.items, spec.items);
        }
    }
}

test "cooler contract: a cold prompt admitted at its 512-row bill holds no more than that bill through its prefill, under the served width hook" {
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(Q, "prompt")) return error.SkipZigTest;
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const bills = @import("bills.zig");
    const ids = try Q.prompt(a, dir);
    defer a.free(ids);
    contract.unloadHeldDir(dir);
    var ld_own: Load = undefined;
    try ld_own.openLight(Q, dir, true);
    defer _ = ld_own.close();
    const ld = &ld_own;
    // Long enough that the request's reservation outgrows the cache's floor.
    const len: usize = @intCast(@import("kv_cache.zig").KVCache.RESERVE_MIN_TOKENS + 1024);
    const n: u32 = 16;
    const long = try a.alloc(u32, len);
    defer a.free(long);
    for (long, 0..) |*t, i| t.* = ids[i % ids.len];
    const c = &ld.config;
    const bits = bills.defaultKvBits(c);
    // Admission at the edge: the memory it reads is the 512-row bill, and the width it picks there.
    const bill = bills.prefillNeededAtChunk(c, len, n, bits, 512, .{});
    const picked = bills.chooseRequestPrefillChunk(c, len, n, bits, bill, 0, .{});
    const turn = arch_handle.turnFor(ld.h, c);
    defer turn.end();
    defer arch_sched.endRequest(c);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    mlx.engine.close();
    mlx.engine.trim();
    const held0 = mlx.engine.poolTotal();
    var box: AdmittedBox = .{ .cfg = c, .bits = bits, .avail = bill, .in0 = mlx.engine.poolTotal() - mlx.engine.poolFree() };
    mlx.engine.resetPoolPeak(.process);
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, long, n, .{ .temperature = 0.0 }, &.{}, .{
        .pinned_prefill_chunk = picked,
        .adaptive_chunk_width = true,
        .chunk_width_hook = .{ .ctx = &box, .call = AdmittedBox.call, .confirm = AdmittedBox.confirm },
    });
    defer gen.deinit(a);
    _ = try gen.next(a);
    mlx.engine.close();
    const held = mlx.engine.poolHeldPeak(.process) -| held0;
    errdefer std.debug.print("a cold {d}-token prompt admitted at its {d} MB bill (width {d}), the hook's widths {any}, {d} widens confirmed: {d} MB held at the peak\n", .{ len, bill >> 20, picked, box.widths[0..@min(box.n, box.widths.len)], box.widened, held >> 20 });
    try testing.expect(held <= bill);
}

test "cooler contract held loads: the read after each test passes a held load between requests and names one whose config changed or beside which a dispatch stayed open" {
    try servedOn("qwen4", &.{"QWEN4_STQ_TINY"}, struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const ld = try heldLoad(Q, dir);
            var got = try greedy(ld, ids, 4);
            got.deinit(testing.allocator);
            var buf: [512]u8 = undefined;
            var w = std.Io.Writer.fixed(&buf);
            try testing.expect(!heldRest(&w));
            // A config the cell changed and did not put back.
            const was = ld.config.num_hidden_layers;
            ld.config.num_hidden_layers +%= 1;
            try testing.expect(heldRest(&w));
            try testing.expect(std.mem.indexOf(u8, w.buffered(), "config changed") != null);
            ld.config.num_hidden_layers = was;
            // A dispatch the cell encoded and never closed.
            w = std.Io.Writer.fixed(&buf);
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            const id = [_]i32{ 1, 2, 3, 4 };
            const x = try @import("generate.zig").idsArray(ld.xfm, &id, &.{ 1, 4 });
            defer _ = mlx.mlx_array_free(x);
            var y = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(y);
            try mlx.check(mlx.mlx_add(&y, x, x, ld.xfm.outStream()));
            turn.end();
            try testing.expect(mlx.engine.openDispatches() > 0);
            try testing.expect(heldRest(&w));
            try testing.expect(std.mem.indexOf(u8, w.buffered(), "dispatches open") != null);
            w = std.Io.Writer.fixed(&buf);
            try testing.expect(!heldRest(&w));
        }
    }.run);
}

/// A request of `ids` whose prefill runs at `chunk` rows, its Generator made (the prefill runs in it) and handed to `body`.
fn withPrefilled(ld: *Load, ids: []const u32, n: u32, chunk: usize, ctx: anytype, comptime body: fn (@TypeOf(ctx), *Generator) anyerror!void) !void {
    const a = testing.allocator;
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, ids, n, .{ .temperature = 0.0 }, &.{}, .{ .pinned_prefill_chunk = chunk });
    defer gen.deinit(a);
    try body(ctx, &gen);
}

/// A greedy request, plain or through the arch's speculative path, on a load that stays.
fn requestOf(comptime C: type, ld: *Load, prompt: []const u32, n: u32, spec: bool) !std.ArrayList(u32) {
    if (!spec) return greedy(ld, prompt, n);
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    try ld.xfm.resetCache();
    if (comptime !@hasDecl(C, "specTokens")) return error.SkipZigTest;
    return C.specTokens(testing.allocator, ld.xfm, prompt, n);
}

fn prefillHandsChunksBack(comptime Q: type) !void {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const ids = try Q.prompt(a, dir);
    defer a.free(ids);
    const ld = try heldLoad(Q, dir);
    var long: [3600]u32 = undefined;
    for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
    var free_after: [2]u64 = undefined;
    for ([_]usize{ 512, long.len }, &free_after) |len, *free| {
        const turn = arch_handle.turnFor(ld.h, &ld.config);
        defer turn.end();
        try ld.xfm.resetCache();
        mlx.engine.close();
        mlx.engine.trim();
        var tok = Tokenizer.initEmpty(a);
        defer tok.deinit();
        var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, long[0..len], 1, .{ .temperature = 0.0 }, &.{}, .{ .pinned_prefill_chunk = 512, .skip_lazy_preforward = true });
        defer gen.deinit(a);
        mlx.engine.close();
        free.* = mlx.engine.stats().pool_free;
        arch_sched.endRequest(&ld.config);
    }
    // Beyond one chunk's, only what the whole prompt sizes may stay free: its ids, a pool class of their bytes.
    try testing.expect(free_after[1] <= free_after[0] + std.math.ceilPowerOfTwoAssert(usize, long.len * @sizeOf(u32)));
}

fn commitCutHoldsOneBuffer(comptime Q: type) !void {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try Q.prompt(a, dir);
    defer a.free(ids);
    const ld = try heldLoad(Q, dir);
    const bills = @import("bills.zig");
    const len: usize = 4097;
    const n: u32 = 3072;
    const long = try a.alloc(u32, len);
    defer a.free(long);
    for (long, 0..) |*t, i| t.* = ids[i % ids.len];
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, long, n, .{ .temperature = 0.0 }, &.{}, .{});
    defer gen.deinit(a);
    // The reply is stopped a hundred tokens in: the request reserved its last row (the prompt and all 3,072 asked).
    for (0..100) |_| _ = try gen.next(a);
    const KV = @import("kv_cache.zig").KVCache;
    const cache = &ld.xfm.cache;
    try testing.expect(cache.end_tokens >= len + n);
    var slack: u64 = 0;
    for (cache.entries) |entry| if (entry.initialized) {
        slack = @max(slack, KV.bufferCapacity(entry.keys.array) - entry.offset);
    };
    try testing.expect(slack > 0);
    mlx.engine.close();
    mlx.engine.trim();
    const before = mlx.engine.poolTotal() - mlx.engine.poolFree();
    mlx.engine.resetPoolPeak(.process);
    try cache.shrinkToRows(mlx.engine.stream());
    mlx.engine.close();
    const over = mlx.engine.poolPeak(.process) -| before;
    mlx.engine.trim();
    const after = mlx.engine.poolTotal() - mlx.engine.poolFree();
    // Every buffer holds its rows, and the cut held beside the cache the one widest buffer of those rows.
    var widest: u64 = 0;
    for (cache.entries) |entry| if (entry.initialized) {
        try testing.expectEqual(entry.offset, KV.bufferCapacity(entry.keys.array));
        for ([_]mlx.mlx_array{ entry.keys.array, entry.values.array, entry.keys_scales.array, entry.keys_biases.array, entry.values_scales.array, entry.values_biases.array }) |buf| {
            if (buf.ctx != null) widest = @max(widest, mlx.engine.poolBytes(mlx.mlx_array_size(buf) * mlx.mlx_array_itemsize(buf)));
        }
    };
    const freed = before -| after;
    const kv = bills.configuredKvQuantFor(&ld.config);
    // Billed for the request as the engine bound it, in a context that holds it.
    var wide = ld.config;
    wide.pinned_context = 1 << 20;
    const billed = bills.commitTrimBytes(&wide, len, n, if (kv.scheme == .off) 16 else kv.bits, wide.prefillStartWidth(), .{});
    errdefer std.debug.print("cooler contract: a cache cut to its rows held {d} B past itself, its widest buffer is {d} B, billed {d} B; the cut gave {d} B back\n", .{ over, widest, billed, freed });
    try testing.expectEqual(widest, over);
    try testing.expect(over <= billed);
    try testing.expect(freed > 0);
}

/// One greedy request under a thinking bound, ticked in the scheduler's order (`loopGuardTick` before every tick,
/// `commitThoughtClose` when it fires).
const BoundRun = struct {
    /// `drained` decodes serial but leaves the pipeline before the thought's last token, as an MTP serial block's exit
    /// does, so the bound fires with no token in flight whatever the model accepts; `mtp` runs rounds at a pinned depth.
    const Mode = union(enum) { serial, drained, mtp: u32 };

    tokens: std.ArrayList(u32) = .empty,
    /// The logits the forced commit leaves for the token after it.
    logits: []f32 = &.{},
    /// The bound fired on a speculative round's exit (no pipelined token in flight).
    at_exit: bool = false,

    fn deinit(self: *BoundRun, a: std.mem.Allocator) void {
        self.tokens.deinit(a);
        a.free(self.logits);
    }

    fn run(ld: *Load, comptime C: type, prompt: []const u32, n: u32, budget: u32, forced: []const u32, mode: Mode) !BoundRun {
        const a = testing.allocator;
        const scheduler = @import("scheduler.zig");
        const turn = arch_handle.turnFor(ld.h, &ld.config);
        defer turn.end();
        defer arch_sched.endRequest(&ld.config);
        var bound = @import("mtp_plan.zig").ThinkBound{ .budget = budget, .opener_id = null, .closer_id = forced[forced.len - 2], .forced = forced, .in_think = true };
        var tok = Tokenizer.initEmpty(a);
        defer tok.deinit();
        try ld.xfm.resetCache();
        const options: Generator.InitOptions = if (mode == .mtp) C.specOptions(ld.xfm, mode.mtp) else .{};
        var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, prompt, n, .{ .temperature = 0.0, .think_bound = &bound }, &.{}, options);
        defer gen.deinit(a);
        var out: BoundRun = .{};
        errdefer out.deinit(a);
        while (true) {
            bound.observe(gen.generated_ids.items);
            if (bound.due()) {
                bound.fired = true;
                out.at_exit = gen.pending == null;
                const r = try gen.commitForcedTokens(a, forced);
                defer a.free(r.emitted);
                try out.tokens.appendSlice(a, r.emitted);
                if (r.stopped) break;
                out.logits = try test_support.qwen4ReadF32(a, gen.pending.?.logits, ld.xfm.s);
                if (!try gen.mtpResumeAfterForced(a)) {
                    gen.spec_disabled_runtime = true;
                    gen.spec_disable_reason = .think_bound;
                }
                continue;
            }
            if (mode == .mtp and !scheduler.thinkBoundForcesPlain(&gen)) {
                gen.mtp_group_cap = 0;
                scheduler.capMtpRoundToThinkBound(&gen);
                const r = (try gen.nextMtp(a)) orelse break;
                defer a.free(r.tokens);
                try out.tokens.appendSlice(a, r.tokens);
            } else if (mode == .drained and bound.room() == 1) switch (try gen.drainPipelineForSpec(a)) {
                .drained => |t| try out.tokens.append(a, t),
                .already_clean => return error.NoPipelineToDrain,
                .stopped => break,
            } else try out.tokens.append(a, (try gen.next(a)) orelse break);
        }
        if (!bound.fired) return error.BoundNeverFired;
        return out;
    }
};

fn expectCachedSolo(solo0: *const std.ArrayList(u32), solo1: *const std.ArrayList(u32), got: *const Group.Out) !void {
    const wants = [2][]const u32{ solo0.items, solo1.items };
    for (wants, 0..) |want, i| {
        if (got.failed[i]) |e| {
            std.debug.print("cooler contract: row {d} failed ({s})\n", .{ i, e });
            return error.RequestFailed;
        }
        if (std.mem.indexOfDiff(u32, want, got.published[i].items)) |at| {
            std.debug.print("cooler contract: row {d} differs from its solo tokens at {d}\n", .{ i, at });
            return error.RowDiffersFromSolo;
        }
    }
}

/// A request decodes alone; a second arrives and prefills (wide forwards, each installing its chunk's cap); then the
/// two decode as a group, to their ends.
fn groupedAfterPrefill(sv: *Served, long: []const u32, mtp: bool) !void {
    const a = testing.allocator;
    const scheduler = Served.scheduler;
    const sch = &sv.sch;
    const head = if (mtp) @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(sv.entry.handle.?), true, null) else null;
    var slots: [2]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer {
        for (slots[0..made]) |slot| sch.complete(slot);
        for (sch.cleanup_queue.items) |s| s.deinit();
        sch.cleanup_queue.clearRetainingCapacity();
    }
    const uncapped = std.math.maxInt(u64);
    for ([_][]const u32{ long[0..40], long }) |p| {
        const slot = try sch.submit(.{ .model = sv.entry, .prompt_ids = p, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = 24, .enable_mtp = mtp, .mtp = head });
        slots[made] = slot;
        made += 1;
        _ = sch.pending.orderedRemove(0);
        try scheduler.runPrefill(sch, slot);
        if (!scheduler.postPrefillTerminal(sch, slot)) try sch.decoding.append(a, slot);
        if (made == 1) try scheduler.runDecodeTick(sch, slots[0..1]);
    }
    // The second request's last chunk left its cap in force.
    try testing.expect((mlx.engine.poolLimit() orelse uncapped) < uncapped);
    try scheduler.runDecodeTick(sch, slots[0..2]);
    // A grouped tick is a narrow forward: the prefill's cap is lifted for it, as for a one-request step.
    try testing.expectEqual(uncapped, mlx.engine.poolLimit() orelse uncapped);
    while (true) {
        var live: [2]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (slots[0..made]) |slot| if (slot.joinsTick()) {
            live[k] = slot;
            k += 1;
        };
        if (k == 0) break;
        try scheduler.runDecodeTick(sch, live[0..k]);
        scheduler.cullDecoding(sch);
    }
    for (slots[0..made]) |slot| if (slot.error_code) |e| {
        std.debug.print("cooler contract: a served request failed ({s})\n", .{e});
        return error.RequestFailed;
    };
}

/// Requests arrive one at a time, each prefilled (wide forwards, each installing its chunk's cap) while those before it
/// decode, speculative rows at depth 2 where `mtp` says; then all of them tick together to their ends. After every
/// tick, and at the model's next turn, no cap is in force: a tick's forwards are narrow, and lift a chunk's cap.
fn joinedAfterPrefill(sv: *Served, prompts: []const []const u32, mtp: []const bool, max: u32) !void {
    const a = testing.allocator;
    const scheduler = Served.scheduler;
    const sch = &sv.sch;
    const depth_bounds = @import("mtp_depth_bounds.zig");
    const saved = depth_bounds.active;
    defer depth_bounds.active = saved;
    depth_bounds.active = .{ .min = 2, .max = 2 };
    const head = @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(sv.entry.handle.?), true, null);
    var slots: [4]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer {
        for (slots[0..made]) |slot| sch.complete(slot);
        for (sch.cleanup_queue.items) |s| s.deinit();
        sch.cleanup_queue.clearRetainingCapacity();
    }
    const refusals = mlx.engine.poolRefusals();
    var ticks: u32 = 0;
    for (prompts, mtp) |p, spec| {
        const slot = try sch.submit(.{ .model = sv.entry, .prompt_ids = p, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = max, .enable_mtp = spec, .mtp = if (spec) head else null });
        slots[made] = slot;
        made += 1;
        _ = sch.pending.orderedRemove(0);
        try scheduler.runPrefill(sch, slot);
        sv.reused = slot.cached_tokens;
        if (!scheduler.postPrefillTerminal(sch, slot)) try sch.decoding.append(a, slot);
        if (made == prompts.len) try testing.expect(capInForce() != null);
        while (true) {
            var live: [4]*scheduler.Slot = undefined;
            var k: usize = 0;
            for (slots[0..made]) |s| if (s.joinsTick()) {
                live[k] = s;
                k += 1;
            };
            if (k == 0) break;
            try scheduler.runDecodeTick(sch, live[0..k]);
            scheduler.cullDecoding(sch);
            ticks += 1;
            errdefer std.debug.print("cooler contract: after tick {d} of {d} rows a cap of {d} KB is in force over {d} KB the pool's tensors hold\n", .{ ticks, k, (capInForce() orelse 0) >> 10, poolInUse() >> 10 });
            try testing.expectEqual(@as(?u64, null), capInForce());
            {
                const turn = arch_handle.turnFor(sv.entry.handle, sv.entry.config);
                defer turn.end();
                errdefer std.debug.print("cooler contract: at the model's turn after tick {d} a cap of {d} KB is in force over {d} KB the pool's tensors hold\n", .{ ticks, (capInForce() orelse 0) >> 10, poolInUse() >> 10 });
                try testing.expectEqual(@as(?u64, null), capInForce());
            }
            // Requests before the last decode a few ticks while the next arrives; the last group ticks to its end.
            if (made < prompts.len and ticks % 3 == 0) break;
        }
    }
    try testing.expectEqual(refusals, mlx.engine.poolRefusals());
    for (slots[0..made]) |slot| if (slot.error_code) |e| {
        std.debug.print("cooler contract: a served request failed ({s})\n", .{e});
        return error.RequestFailed;
    };
}

/// `n` ids of real text: Pride and Prejudice (public domain, the bench's corpus) through the fixture's own tokenizer.
fn realTextIds(a: std.mem.Allocator, dir: []const u8, n: usize) ![]u32 {
    const prose = @embedFile("fixtures/prose_pride_and_prejudice.txt");
    var tok = try @import("tokenizer.zig").loadTokenizer(io, a, dir);
    defer tok.deinit();
    const text = try tok.encode(a, prose[0..@min(prose.len, n * 8)]);
    defer a.free(text);
    try testing.expect(text.len >= n);
    return a.dupe(u32, text[0..n]);
}

fn greedyOn(ld: *Load, ids: []const u32, n: u32) !std.ArrayList(u32) {
    const a = testing.allocator;
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, ids, n, .{ .temperature = 0.0 }, &.{}, .{});
    defer gen.deinit(a);
    var got: std.ArrayList(u32) = .empty;
    errdefer got.deinit(a);
    while (try gen.next(a)) |t| try got.append(a, t);
    return got;
}

/// The served width hook (`bills.adaptivePrefillWidth`) and its widen confirmation (`adaptivePrefillWidenStillFits`'s
/// test) on a box whose free memory is what admission read, less what the request has taken from the engine's pool
/// since. Records the width each boundary chose, and the widens it confirmed.
const AdmittedBox = struct {
    cfg: *const @import("model.zig").ModelConfig,
    bits: u64,
    avail: u64,
    in0: u64,
    widths: [64]u32 = @splat(0),
    n: usize = 0,
    widened: u32 = 0,
    /// The cache's room at the boundary the hook was last asked at, which the confirmation prices a widen from.
    room: @import("kv_cache.zig").Room = .{},

    fn call(ctx: *anyopaque, pos: usize, cur: u32, cap: u32, st: *@import("mtp_plan.zig").AdaptiveWidthState) u32 {
        const box: *AdmittedBox = @ptrCast(@alignCast(ctx));
        const used = (mlx.engine.poolTotal() - mlx.engine.poolFree()) -| box.in0;
        box.room = st.room;
        const w = @import("bills.zig").adaptivePrefillWidth(box.cfg, box.bits, pos, box.avail -| used, cur, cap, st);
        if (box.n < box.widths.len) box.widths[box.n] = if (w != 0) w else cur;
        box.n += 1;
        return w;
    }
    fn confirm(ctx: *anyopaque, pos: usize, want: u32) bool {
        const box: *AdmittedBox = @ptrCast(@alignCast(ctx));
        const bills = @import("bills.zig");
        const used = (mlx.engine.poolTotal() - mlx.engine.poolFree()) -| box.in0;
        const fits = bills.prefillChunkHeldMax(box.cfg, box.bits, want, pos, box.room) <= box.avail -| used;
        if (fits) box.widened += 1;
        return fits;
    }
};

