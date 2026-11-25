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
const ContractOf = contract.ContractOf;
const qwenTinyServed = contract.qwenTinyServed;
const servedOn = contract.servedOn;
const serial = contract.serial;
const heldRest = contract.heldRest;

test "cooler contract: a wide forward that reads out every row's logits, as `sushi kld` runs one, runs under its cap" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const ld = try heldLoad(C, dir);
            // On the ledger a wide forward runs in the region its load measured (`hy4 plan: a prefill chunk of every
            // width ... runs in the region its load measured`); a `sushi kld` load is off the ledger and uncapped.
            try contract.requirePoolCap(&ld.config);
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
            try withPrefilled(cooler, long, 16, 4096, cooler, Body.decode);
        }
    }.run);
}

test "cooler contract: a decode after a wide forward that left free buffers of sizes it never asks for grows the engine's pool by no more than its forward's cap" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const ld = try heldLoad(C, dir);
            const cooler = @import("arch/cooler.zig");
            // On the ledger a request ends with no free buffer in the pool and asks it nothing (`hy4 close`, and the
            // lines `hy4 ledger` sums after each request).
            try contract.requirePoolCap(&ld.config);
            // An arch that installs its own hard caps refuses a buffer past them; its decode cap has its own cells.
            if (cooler.ownsCaps(&ld.config)) {
                @import("test_note.zig").notePartSkipped("an arch with its own pool caps");
                return error.SkipZigTest;
            }
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

test "cooler contract: a served cold prefill past the reservation floor, whose first chunk grows the request's whole reserved cache, runs under its chunks' caps" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(C, dir);
            // The owner's agent turn, 40,388 tokens cold at 4,096 a chunk. A cut's one attention layer reserves the
            // context's whole rest to outgrow its first chunk's cap, as the real pack's twelve do at 256,000.
            const len: usize = 40388;
            const max_tokens: u32 = 1_000_000;
            if (!ld.config.reservesKvCapacity() or ld.config.max_position_embeddings < len + max_tokens) return error.SkipZigTest;
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
                var want = try requestOf(C, ld, prompt, 6, spec);
                defer want.deinit(a);
                const Req = struct {
                    ld: *Load,
                    prompt: []const u32,
                    spec: bool,
                    fn go(q: @This()) anyerror!void {
                        var got = try requestOf(C, q.ld, q.prompt, 6, q.spec);
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
                    const r = requestOf(C, ld, prompt, 6, spec);
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
                    var after = try requestOf(C, ld, prompt, 6, spec);
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

test "cooler contract: a served speculative request refused at any pool buffer site its plain request lacks, at its first ask or its last, leaves the pool as it was and the next says its tokens" {
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
            var want = try sv.requestWith(&p, 12, .{ .temperature = 0 }, true, &from);
            defer want.deinit(a);
            const Req = struct {
                sv: *Served,
                p: []const u32,
                mtp: bool,
                fn go(q: @This()) anyerror!void {
                    var f: u32 = 0;
                    var got = try q.sv.requestWith(q.p, 12, .{ .temperature = 0 }, q.mtp, &f);
                    got.deinit(testing.allocator);
                }
            };
            // A round's verify holds what its rollback needs on the request's own state, and its forward asks at
            // the sites a prefill chunk does: eight frames tell the two apart, whose unwinding differs above it.
            const sites = &mlx.engine.fault.sites;
            defer sites.* = .{};
            sites.* = .{ .frames = 8 };
            const plain = try mlx.sweepSites(a, sites, Req{ .sv = &sv, .p = &p, .mtp = false }, Req.go);
            defer a.free(plain);
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(a);
            for (plain) |pt| try seen.put(a, pt.site, {});
            sites.* = .{ .lasts = true, .frames = 8 };
            const all = try mlx.sweepSites(a, sites, Req{ .sv = &sv, .p = &p, .mtp = true }, Req.go);
            defer a.free(all);
            sv.quiet_failures = true;
            const base = Sheet.now();
            for (all, 0..) |pt, at| {
                if (seen.contains(pt.site)) continue;
                errdefer std.debug.print("cooler contract: a speculative request refused at pool point {d} of {d} (ask {d} of its site)\n", .{ at, all.len, pt.n });
                mlx.engine.fault.armAt(pt);
                const r = Req.go(.{ .sv = &sv, .p = &p, .mtp = true });
                const fired = !mlx.engine.fault.armed();
                mlx.engine.fault.disarm();
                r catch {};
                try testing.expect(!mlx.errorPending());
                try testing.expect(fired or !pt.strict);
                var got = try sv.requestWith(&p, 12, .{ .temperature = 0 }, true, &from);
                defer got.deinit(a);
                try testing.expectEqualSlices(u32, want.items, got.items);
                const now = Sheet.now();
                try testing.expectEqual(base.in_use, now.in_use);
                try testing.expectEqual(@as(u32, 0), now.open);
            }
        }
    }.run);
}

test "cooler contract: a speculative request failed at any pool buffer site its plain request lacks leaves no dispatch open once the cleanup drain takes its slot, before the slot's memory goes" {
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
            const Req = struct {
                sv: *Served,
                p: []const u32,
                mtp: bool,
                fn go(q: @This()) anyerror!void {
                    _ = try q.sv.requestToCleanup(q.p, 12, q.mtp);
                }
            };
            const sites = &mlx.engine.fault.sites;
            defer sites.* = .{};
            sites.* = .{ .frames = 8 };
            const plain = try mlx.sweepSites(a, sites, Req{ .sv = &sv, .p = &p, .mtp = false }, Req.go);
            defer a.free(plain);
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(a);
            for (plain) |pt| try seen.put(a, pt.site, {});
            sites.* = .{ .lasts = true, .frames = 8 };
            const all = try mlx.sweepSites(a, sites, Req{ .sv = &sv, .p = &p, .mtp = true }, Req.go);
            defer a.free(all);
            const base = Sheet.now();
            for (all, 0..) |pt, at| {
                if (seen.contains(pt.site)) continue;
                errdefer std.debug.print("cooler contract: a speculative request failed at pool point {d} of {d} (ask {d} of its site)\n", .{ at, all.len, pt.n });
                mlx.engine.fault.armAt(pt);
                const open = try sv.requestToCleanup(&p, 12, true);
                mlx.engine.fault.disarm();
                try testing.expect(!mlx.errorPending());
                try testing.expectEqual(@as(u32, 0), open);
                try testing.expect(Sheet.now().in_use <= base.in_use);
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

test "cooler contract: a Cooler image request refused at any pool buffer site a text request lacks frees what it made, the embedding its rows splice into too" {
    try eachArch(struct {
        const Req = struct {
            ld: *Load,
            prompt: []const u32,
            fn go(q: @This()) anyerror!void {
                var got = try greedy(q.ld, q.prompt, 2);
                got.deinit(testing.allocator);
            }
        };

        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            defer mlx.engine.fault.disarm();
            const ld = try heldLoad(C, dir);
            // A cut fixture's vocabulary may end before its image placeholder: no prompt can carry one.
            if (ld.config.image_token_id == 0 or ld.config.image_token_id >= ld.config.vocab_size) return error.SkipZigTest;
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
            a.free(try firstAsks(a, &seen, C, Req{ .ld = ld, .prompt = &text }, Req.go));
            const points = try firstAsks(a, &seen, C, Req{ .ld = ld, .prompt = &prompt }, Req.go);
            defer a.free(points);
            // A load on the ledger takes an image's rows as the request's Media, not from the pool: its request asks
            // the closed pool the sites a text request does, and none of its own (`hy4 ledger`, `hy4 close`).
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
    try Open.serves(H, dirs[0]);
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

