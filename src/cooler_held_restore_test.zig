//! The Sushi Cooler contract's restore cells on a held load (`contract.heldLoad`): one served load per fixture for
//! the step, read back at rest after every test. The runner and the shared helpers are in `cooler_contract_test.zig`.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const arch_handle = @import("arch/handle.zig");
const io = std.Io.Threaded.global_single_threaded.io();
const contract = @import("cooler_contract_test.zig");
const heldLoad = contract.heldLoad;
const Load = contract.Load;
const Served = contract.Served;
const eachArch = contract.eachArch;
const eachQwenServed = contract.eachQwenServed;
const firstAsks = contract.firstAsks;
const qwenTinyServed = contract.qwenTinyServed;
const inUsePeak = contract.inUsePeak;
const PartialShare = @import("cooler_restore_test.zig").PartialShare;
const fitsOnlyTaken = @import("cooler_restore_test.zig").fitsOnlyTaken;
const heldPrompt = @import("cooler_restore_test.zig").heldPrompt;
const takeoverAtRest = @import("cooler_restore_test.zig").takeoverAtRest;

test "cooler contract: a served Cooler request that restores from the SSD tier says a cold request's tokens and asks nothing of MLX" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            // Long enough that the tier holds restore points under it.
            var long: [1300]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
            const ld = try heldLoad(C, dir);
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            // Held restore points would answer from RAM; with none held every point is on the SSD tier.
            sv.entry.prefix_cache.?.disk.?.held_cap_for_test = 0;
            var cold_from: u32 = 0;
            var cold = try sv.request(&long, 12, &cold_from);
            defer cold.deinit(a);
            try testing.expectEqual(@as(u32, 0), cold_from);
            // Another conversation takes the model, so the first one's state stands only on SSD.
            var other_from: u32 = 0;
            var other = try sv.request(ids[0..@min(ids.len, 20)], 4, &other_from);
            other.deinit(a);
            const calls = mlx.engine.mlxCalls();
            var warm_from: u32 = 0;
            var warm = try sv.request(&long, 12, &warm_from);
            defer warm.deinit(a);
            try testing.expectEqual(calls, mlx.engine.mlxCalls());
            errdefer std.debug.print("cooler contract: the warm request restored {d} tokens from SSD\n", .{warm_from});
            try testing.expect(warm_from > 0);
            try testing.expectEqualSlices(u32, cold.items, warm.items);
        }
    }.run);
}

test "cooler contract: with the SSD tier off a served model reuses a repeated prompt from RAM, or keeps no RAM entries and says so where its RAM points would hold no state" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const log = @import("log.zig");
            const path = "/tmp/sushi-ram-only-restore.log";
            _ = std.c.unlink(path);
            try log.openFile(path, 0);
            defer {
                log.closeFile();
                _ = std.c.unlink(path);
            }
            var long: [700]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
            const ld = try heldLoad(C, dir);
            var sv: Served = undefined;
            try sv.initRamOnly(ld, dir, 1 << 30);
            defer sv.deinit();
            if (@import("arch/prefix.zig").ramHoldsNoState(ld.xfm)) {
                try testing.expect(sv.entry.prefix_cache == null);
                var buf: [4096]u8 = undefined;
                const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
                try testing.expect(fd >= 0);
                defer _ = std.c.close(fd);
                const n = std.c.read(fd, &buf, buf.len);
                const text = buf[0..if (n > 0) @intCast(n) else 0];
                errdefer std.debug.print("cooler contract: the load said:\n{s}\n", .{text});
                var want_buf: [160]u8 = undefined;
                const want = try std.fmt.bufPrint(&want_buf, "[prefix-cache] {s} restores a prompt only from the SSD tier, which is off: it keeps no RAM prefix entries", .{ld.config.model_type});
                try testing.expect(std.mem.indexOf(u8, text, want) != null);
                return;
            }
            var cold_from: u32 = 0;
            var cold = try sv.request(&long, 8, &cold_from);
            defer cold.deinit(a);
            var warm_from: u32 = 0;
            var warm = try sv.request(&long, 8, &warm_from);
            defer warm.deinit(a);
            errdefer std.debug.print("cooler contract: the repeated prompt reused {d} tokens from RAM\n", .{sv.reused});
            try testing.expect(sv.reused > 0);
            try testing.expectEqualSlices(u32, cold.items, warm.items);
        }
    }.run);
}

test "cooler contract: a served request restored from a shorter prompt's SSD record takes it, and says the same tokens every time" {
    // Bytewise cold holds only for a restore at the request's own point (the SSD contract above): another prompt's
    // record sits off this prompt's cold split ([0, N - 31) then the tail), so its state reaches N - 31 in two
    // spans where cold forwards one, and bf16 near-ties may flip. What holds is that the restore is taken and is
    // deterministic.
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var long: [1300]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[(i * 7 + i / ids.len) % ids.len];
            const ld = try heldLoad(C, dir);
            var said: [2]std.ArrayList(u32) = undefined;
            for (&said) |*got| {
                // A fresh tier each time: the shorter prompt's record is the only one the longer request can restore.
                var sv: Served = undefined;
                try sv.init(ld, dir);
                defer sv.deinit();
                var from: u32 = 0;
                var shorter = try sv.request(long[0..1299], 24, &from);
                shorter.deinit(a);
                got.* = try sv.request(&long, 24, &from);
                errdefer std.debug.print("cooler contract: the longer request reused {d} tokens of the shorter's\n", .{sv.reused});
                try testing.expectEqual(@as(u32, 1299 - 31), sv.reused);
            }
            defer for (&said) |*got| got.deinit(a);
            try testing.expectEqualSlices(u32, said[0].items, said[1].items);
        }
    }.run);
}

test "cooler contract: a served SSD restore refused at any pool buffer site its cold requests lack falls back to a cold prefill and says the cold tokens, and the next conversation says them" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var long: [1300]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
            const ld = try heldLoad(Q, dir);
            var cold: std.ArrayList(u32) = undefined;
            const plain = SsdRestore{ .ld = ld, .dir = dir, .long = &long, .other = ids[0..@min(ids.len, 20)], .restored = &cold };
            try plain.go();
            defer cold.deinit(a);
            // Each point replays the whole conversation on a fresh tier, so every recorded point is reached again. The
            // restore's own sites only: the cold requests' are the request sweep's.
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(a);
            a.free(try firstAsks(a, &seen, Q, SsdRestore{ .ld = ld, .dir = dir, .long = &long, .other = ids[0..@min(ids.len, 20)], .cold_only = true }, SsdRestore.go));
            const points = try firstAsks(a, &seen, Q, SsdRestore{ .ld = ld, .dir = dir, .long = &long, .other = ids[0..@min(ids.len, 20)] }, SsdRestore.go);
            defer a.free(points);
            for (points, 0..) |pt, at| {
                var got: std.ArrayList(u32) = .empty;
                mlx.engine.fault.armAt(pt);
                const r = (SsdRestore{ .ld = ld, .dir = dir, .long = &long, .other = ids[0..@min(ids.len, 20)], .restored = &got }).go();
                const fired = !mlx.engine.fault.armed();
                mlx.engine.fault.disarm();
                errdefer std.debug.print("cooler contract: SSD restore point {d} of {d} (fired {})\n", .{ at, points.len, fired });
                try r;
                defer got.deinit(a);
                try testing.expectEqualSlices(u32, cold.items, got.items);
                try testing.expect(!mlx.errorPending());
                if (pt.strict) try testing.expect(fired);
                var after: std.ArrayList(u32) = undefined;
                try (SsdRestore{ .ld = ld, .dir = dir, .long = &long, .other = ids[0..@min(ids.len, 20)], .restored = &after }).go();
                defer after.deinit(a);
                try testing.expectEqualSlices(u32, cold.items, after.items);
            }
        }
    }.run);
}

test "cooler contract: a served Qwen conversation keeps its newest off-grid restore points in RAM, writes none while the next turn extends them, and a flush writes the held ones" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const dws = @import("disk_write_stats.zig");
            const ssm = @backingInt(dws.Kind.ssm);
            const ld = try heldLoad(Q, dir);
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            var prompt: std.ArrayList(u32) = .empty;
            defer prompt.deinit(a);
            for (0..3000) |i| try prompt.append(a, ids[(i * 7 + i / ids.len) % ids.len]);
            const disk = &sv.entry.prefix_cache.?.disk.?;
            const before = dws.process.snapshot().written[ssm];
            var f: u32 = 0;
            var last_point: u32 = 0;
            // Four turns under the 8192 grid: every restore point is off-grid, and each turn restores the one before.
            for (0..4) |turn| {
                var reply = try sv.request(prompt.items, 16, &f);
                defer reply.deinit(a);
                disk.drainWriter();
                if (turn > 0) {
                    try testing.expectEqual(last_point, sv.reused);
                    // A held point is restored from RAM: the hit counts on the RAM tier, never the SSD's.
                    try testing.expectEqual(@import("cache_stats.zig").Tier.hot, sv.entry.prefix_cache.?.lastRestoredTier());
                }
                last_point = @intCast(prompt.items.len - 31);
                try prompt.appendSlice(a, reply.items);
                for (0..1500) |i| try prompt.append(a, ids[(i * 13 + turn * 101) % ids.len]);
            }
            const during = dws.process.snapshot().written[ssm] - before;
            if (during != 0) std.debug.print("cooler contract: the conversation wrote {d} KB of restore points while each turn extended the last\n", .{during >> 10});
            try testing.expectEqual(@as(u64, 0), during);
            @import("scheduler.zig").flushLiveState(sv.entry, "test");
            disk.drainWriter();
            try testing.expect(dws.process.snapshot().written[ssm] - before > 0);
        }
    }.run);
}

test "cooler contract: a served Qwen turn that leaves the last turn's text between two restore points inherits the one below the cut, the engine refusing nothing" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const route = @import("tensor/route.zig");
            const ld = try heldLoad(Q, dir);
            var sv: Served = undefined;
            try sv.initTiers(ld, dir, 1 << 30);
            defer sv.deinit();
            const refused0 = route.refusals;
            var f: u32 = 0;
            var first = try heldPrompt(a, ids, 1500, 0);
            defer first.deinit(a);
            var reply = try sv.request(first.items, 16, &f);
            defer reply.deinit(a);
            // Turn 2 extends turn 1: its entry holds both prompt ends' restore points.
            var second: std.ArrayList(u32) = .empty;
            defer second.deinit(a);
            try second.appendSlice(a, first.items);
            try second.appendSlice(a, reply.items);
            var said = try heldPrompt(a, ids, 1500, 2);
            defer said.deinit(a);
            try second.appendSlice(a, said.items);
            var reply2 = try sv.request(second.items, 16, &f);
            defer reply2.deinit(a);
            // Turn 3 cuts between them: its commit inherits turn 1's point, the QSA history moved onto it.
            var next: std.ArrayList(u32) = .empty;
            defer next.deinit(a);
            try next.appendSlice(a, second.items[0 .. second.items.len - 100]);
            var more = try heldPrompt(a, ids, 200, 1);
            defer more.deinit(a);
            try next.appendSlice(a, more.items);
            var got = try sv.request(next.items, 16, &f);
            defer got.deinit(a);
            try testing.expect(sv.reused > 0 and sv.reused < first.items.len + reply.items.len);
            try testing.expectEqual(@as(u64, 0), route.refusals - refused0);
        }
    }.run);
}

test "cooler contract: a served Qwen turn that shares part of the last turn's RAM entry, and fits only by taking it over, takes it and says the shared restore's tokens, the engine refusing nothing (#109)" {
    try eachQwenServed(struct {
        /// Admission that refuses a warm share the request did not take over: a copy of the prefix does not fit.
        fn onlyTaken(_: *const @import("model.zig").ModelConfig, _: usize, _: u32, _: @import("trunk.zig").KVQuantConfig, _: bool, matched: u64, _: u64, will_donate: bool, _: bool, _: @import("scheduler.zig").CheckpointPlan) bool {
            return matched == 0 or will_donate;
        }

        /// Turn 1, a turn 2 that extends it (its entry holds both prompt ends' restore points), then a turn 3 that
        /// leaves turn 2's text between those points and says more; turn 3's tokens. Every engine refusal on the way
        /// is counted in `refused`.
        fn conversation(ld: *Load, dir: []const u8, ids: []const u32, taken: bool, reused: *u32, entries: *usize, refused: *u64) !std.ArrayList(u32) {
            const a = testing.allocator;
            const route = @import("tensor/route.zig");
            const refused0 = route.refusals;
            defer refused.* = route.refusals - refused0;
            var sv: Served = undefined;
            try sv.initTiers(ld, dir, 1 << 30);
            defer sv.deinit();
            var f: u32 = 0;
            var first = try heldPrompt(a, ids, 1500, 0);
            defer first.deinit(a);
            var reply = try sv.request(first.items, 16, &f);
            defer reply.deinit(a);
            var second: std.ArrayList(u32) = .empty;
            defer second.deinit(a);
            try second.appendSlice(a, first.items);
            try second.appendSlice(a, reply.items);
            var said = try heldPrompt(a, ids, 1500, 2);
            defer said.deinit(a);
            try second.appendSlice(a, said.items);
            var reply2 = try sv.request(second.items, 16, &f);
            defer reply2.deinit(a);
            // The entry's point at turn 2's prompt end is past the cut, another text's: the share keeps turn 1's.
            var next: std.ArrayList(u32) = .empty;
            defer next.deinit(a);
            try next.appendSlice(a, second.items[0 .. second.items.len - 100]);
            var more = try heldPrompt(a, ids, 200, 1);
            defer more.deinit(a);
            try next.appendSlice(a, more.items);
            const scheduler = @import("scheduler.zig");
            const fits_was = scheduler.prefill_admission_fits;
            defer scheduler.prefill_admission_fits = fits_was;
            if (taken) scheduler.prefill_admission_fits = onlyTaken;
            var got = try sv.request(next.items, 16, &f);
            errdefer got.deinit(a);
            reused.* = sv.reused;
            // A turn 4 that leaves turn 3's text between its restore points: a partial share of the entry turn 3
            // committed, taken over again on the takeover arm.
            var fourth: std.ArrayList(u32) = .empty;
            defer fourth.deinit(a);
            try fourth.appendSlice(a, next.items[0 .. next.items.len - 50]);
            try fourth.appendSlice(a, said.items[0..150]);
            var got4 = try sv.request(fourth.items, 16, &f);
            defer got4.deinit(a);
            if (sv.reused == 0 or sv.reused >= next.items.len - 50) return error.TestTurnFourNotPartial;
            try got.appendSlice(a, got4.items);
            entries.* = sv.entry.prefix_cache.?.entries.items.len;
            return got;
        }

        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            var shared_reused: u32 = 0;
            var taken_reused: u32 = 0;
            var entries: usize = 0;
            var refused: u64 = 0;
            var shared = try conversation(ld, dir, ids, false, &shared_reused, &entries, &refused);
            defer shared.deinit(a);
            // The share's commit inherits the entry's points below the cut on the engine's stream.
            try testing.expectEqual(@as(u64, 0), refused);
            var taken = try conversation(ld, dir, ids, true, &taken_reused, &entries, &refused);
            defer taken.deinit(a);
            try testing.expectEqual(@as(u64, 0), refused);
            // A partial share: turn 1's restore point, below the cut.
            try testing.expect(taken_reused > 0 and taken_reused < 1516);
            try testing.expectEqual(shared_reused, taken_reused);
            try testing.expectEqualSlices(u32, shared.items, taken.items);
            // Turn 2's commit replaced the entry it took over.
            try testing.expectEqual(@as(usize, 1), entries);
        }
    }.run);
}

test "cooler contract: a served Qwen turn no longer than the RAM entry it partly shares, a regenerate or a turn that drops more of the last one than it adds, fits by taking it over and says the shared restore's tokens (#109)" {
    try eachQwenServed(struct {
        /// Admission that refuses a warm share the request did not take over: a copy of the prefix does not fit.
        fn onlyTaken(_: *const @import("model.zig").ModelConfig, _: usize, _: u32, _: @import("trunk.zig").KVQuantConfig, _: bool, matched: u64, _: u64, will_donate: bool, _: bool, _: @import("scheduler.zig").CheckpointPlan) bool {
            return matched == 0 or will_donate;
        }

        const Shape = enum { shorter_turn, regenerate };

        /// Turn 1, a turn 2 that extends it, then turn 3: turn 2 less its last 100 tokens and 50 new ones, or turn 2's
        /// prompt again. Returns turn 3's tokens; its prompt (`prompt_len`) is shorter than turn 2's entry either way.
        fn conversation(ld: *Load, dir: []const u8, ids: []const u32, shape: Shape, taken: bool, reused: *u32, prompt_len: *usize, refused: *u64) !std.ArrayList(u32) {
            const a = testing.allocator;
            const route = @import("tensor/route.zig");
            const refused0 = route.refusals;
            defer refused.* = route.refusals - refused0;
            var sv: Served = undefined;
            try sv.initTiers(ld, dir, 1 << 30);
            defer sv.deinit();
            var f: u32 = 0;
            var first = try heldPrompt(a, ids, 1500, 0);
            defer first.deinit(a);
            var reply = try sv.request(first.items, 16, &f);
            defer reply.deinit(a);
            var second: std.ArrayList(u32) = .empty;
            defer second.deinit(a);
            try second.appendSlice(a, first.items);
            try second.appendSlice(a, reply.items);
            var said = try heldPrompt(a, ids, 1500, 2);
            defer said.deinit(a);
            try second.appendSlice(a, said.items);
            var reply2 = try sv.request(second.items, 16, &f);
            defer reply2.deinit(a);
            var next: std.ArrayList(u32) = .empty;
            defer next.deinit(a);
            switch (shape) {
                .shorter_turn => {
                    try next.appendSlice(a, second.items[0 .. second.items.len - 100]);
                    var more = try heldPrompt(a, ids, 50, 1);
                    defer more.deinit(a);
                    try next.appendSlice(a, more.items);
                },
                .regenerate => try next.appendSlice(a, second.items),
            }
            if (next.items.len >= second.items.len + reply2.items.len) return error.TestTurnThreeNotShorter;
            prompt_len.* = next.items.len;
            const scheduler = @import("scheduler.zig");
            const fits_was = scheduler.prefill_admission_fits;
            defer scheduler.prefill_admission_fits = fits_was;
            if (taken) scheduler.prefill_admission_fits = onlyTaken;
            const got = try sv.request(next.items, 16, &f);
            reused.* = sv.reused;
            return got;
        }

        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            for ([_]Shape{ .shorter_turn, .regenerate }) |shape| {
                var shared_reused: u32 = 0;
                var taken_reused: u32 = 0;
                var prompt_len: usize = 0;
                var refused: u64 = 0;
                var shared = try conversation(ld, dir, ids, shape, false, &shared_reused, &prompt_len, &refused);
                defer shared.deinit(a);
                try testing.expectEqual(@as(u64, 0), refused);
                var taken = try conversation(ld, dir, ids, shape, true, &taken_reused, &prompt_len, &refused);
                defer taken.deinit(a);
                try testing.expectEqual(@as(u64, 0), refused);
                // The slot restores at a point below its prompt's end, the regenerate's whole-prompt share included.
                try testing.expect(taken_reused > 0 and taken_reused < prompt_len);
                try testing.expectEqual(shared_reused, taken_reused);
                try testing.expectEqualSlices(u32, shared.items, taken.items);
            }
        }
    }.run);
}

test "cooler contract: a served Qwen turn whose partial share fits once an idle entry is evicted is served as a share, and the entry it restores from stays whole (#109)" {
    try eachQwenServed(struct {
        var cache: ?*const @import("prefix_cache.zig").HotPrefixCache = null;

        /// Admission that refuses a warm share while the cache holds another entry than the one shared.
        fn fitsAlone(_: *const @import("model.zig").ModelConfig, _: usize, _: u32, _: @import("trunk.zig").KVQuantConfig, _: bool, matched: u64, _: u64, will_donate: bool, _: bool, _: @import("scheduler.zig").CheckpointPlan) bool {
            return matched == 0 or will_donate or cache.?.entries.items.len <= 1;
        }

        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            // The hot cache keeps max_concurrent conversations: this cell needs an idle one beside the shared one.
            const was = ld.config.max_concurrent;
            ld.config.max_concurrent = 2;
            defer ld.config.max_concurrent = was;
            var sv: Served = undefined;
            try sv.initTiers(ld, dir, 1 << 30);
            defer sv.deinit();
            // RAM-first: an idle entry stays resident, so there is one to evict.
            sv.entry.prefix_cache.?.ssd_first = false;
            var f: u32 = 0;
            var first = try heldPrompt(a, ids, 1500, 0);
            defer first.deinit(a);
            var reply = try sv.request(first.items, 16, &f);
            defer reply.deinit(a);
            var second: std.ArrayList(u32) = .empty;
            defer second.deinit(a);
            try second.appendSlice(a, first.items);
            try second.appendSlice(a, reply.items);
            var said = try heldPrompt(a, ids, 1500, 2);
            defer said.deinit(a);
            try second.appendSlice(a, said.items);
            var reply2 = try sv.request(second.items, 16, &f);
            defer reply2.deinit(a);
            // Another conversation's entry, idle.
            var other = try heldPrompt(a, ids, 600, 7);
            defer other.deinit(a);
            var other_reply = try sv.request(other.items, 8, &f);
            other_reply.deinit(a);
            const hc = &sv.entry.prefix_cache.?;
            try testing.expectEqual(@as(usize, 2), hc.entries.items.len);
            var next: std.ArrayList(u32) = .empty;
            defer next.deinit(a);
            try next.appendSlice(a, second.items[0 .. second.items.len - 100]);
            var more = try heldPrompt(a, ids, 200, 1);
            defer more.deinit(a);
            try next.appendSlice(a, more.items);
            const scheduler = @import("scheduler.zig");
            const fits_was = scheduler.prefill_admission_fits;
            defer scheduler.prefill_admission_fits = fits_was;
            cache = hc;
            defer cache = null;
            scheduler.prefill_admission_fits = fitsAlone;
            var got = try sv.request(next.items, 16, &f);
            defer got.deinit(a);
            try testing.expect(sv.reused > 0);
            // The idle entry made the room; the entry turn 3 shared still holds turn 2 whole.
            var whole = false;
            for (hc.entries.items) |*e| whole = whole or e.tokens.len == second.items.len + reply2.items.len;
            try testing.expect(whole);
        }
    }.run);
}

test "cooler contract: a served Qwen conversation of takeovers, partial, regenerated and diverging, says its share arm's tokens with the pool poisoned, under a roomy and a tight RAM budget" {
    try qwenTinyServed(struct {
        /// Turn 1, turn 2, a partial takeover turn 3, a regenerate of turn 3, a turn 5 that diverges inside the taken
        /// prefix, a turn 6 that extends turn 5: every reply, in order. Taken, every free buffer is poisoned between turns.
        fn conversation(ld: *Load, dir: []const u8, ids: []const u32, taken: bool, ram: u64, out: *std.ArrayList(u32)) !void {
            const a = testing.allocator;
            var sv: Served = undefined;
            try sv.initTiers(ld, dir, ram);
            defer sv.deinit();
            const poison: ?u8 = if (taken) 0xFF else null;
            mlx.engine.poison.byte = poison;
            defer mlx.engine.poison.byte = null;
            var f: u32 = 0;
            const fits_was = Served.scheduler.prefill_admission_fits;
            defer Served.scheduler.prefill_admission_fits = fits_was;
            if (taken) Served.scheduler.prefill_admission_fits = fitsOnlyTaken;
            var prompt = try heldPrompt(a, ids, 1500, 0);
            defer prompt.deinit(a);
            const plans = [_]struct { cut: usize, say: usize, seed: usize }{
                .{ .cut = 0, .say = 1500, .seed = 2 }, // turn 2 extends turn 1 and its reply
                .{ .cut = 100, .say = 200, .seed = 1 }, // turn 3 re-renders the reply's tail: a partial takeover
                .{ .cut = 0, .say = 0, .seed = 0 }, // turn 4 regenerates turn 3
                .{ .cut = 1200, .say = 300, .seed = 3 }, // turn 5 diverges well inside the taken prefix
                .{ .cut = 0, .say = 400, .seed = 4 }, // turn 6 extends turn 5
            };
            var last: std.ArrayList(u32) = .empty;
            defer last.deinit(a);
            for (0..plans.len + 1) |k| {
                if (k > 0 and k != 3) {
                    // This prompt, its reply, cut, then new text; turn 4 resends turn 3's prompt.
                    const pl = plans[k - 1];
                    try prompt.appendSlice(a, last.items);
                    prompt.shrinkRetainingCapacity(prompt.items.len - pl.cut);
                    var said = try heldPrompt(a, ids, pl.say, pl.seed);
                    defer said.deinit(a);
                    try prompt.appendSlice(a, said.items);
                }
                if (poison) |b| _ = mlx.engine.poison.free(b);
                var r = try sv.request(prompt.items, 16, &f);
                defer r.deinit(a);
                try out.appendSlice(a, r.items);
                last.clearRetainingCapacity();
                try last.appendSlice(a, r.items);
                try testing.expect(takeoverAtRest(&sv.entry.prefix_cache.?));
            }
        }

        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            for ([_]u64{ 1 << 30, 24 << 20 }) |ram| {
                errdefer std.debug.print("cooler contract: the takeover conversation under a {d} MB RAM budget\n", .{ram >> 20});
                var shared: std.ArrayList(u32) = .empty;
                defer shared.deinit(a);
                try conversation(ld, dir, ids, false, ram, &shared);
                var taken: std.ArrayList(u32) = .empty;
                defer taken.deinit(a);
                try conversation(ld, dir, ids, true, ram, &taken);
                try testing.expectEqualSlices(u32, shared.items, taken.items);
            }
        }
    }.run);
}

test "cooler contract: a served Qwen turn of every prompt shape against a RAM entry, shorter, as long and longer, on the share and diverging, is taken over when its share does not fit, SSD-first or not, and says its share arm's tokens" {
    try qwenTinyServed(struct {
        /// Turn 3's prompt: the entry E (turns 1 and 2 and their replies) less its last `cut` tokens, then `say` new
        /// ones. A share needs a restore point at or below it: E holds one at each prompt's end.
        const Shape = struct { cut: usize, say: usize };
        const shapes = [_]Shape{
            .{ .cut = 16, .say = 0 }, // shorter, on the share: turn 2 regenerated
            .{ .cut = 0, .say = 0 }, // as long: turn 2's cut reply continued
            .{ .cut = 0, .say = 200 }, // longer: the next turn
            .{ .cut = 300, .say = 200 }, // shorter, diverging
            .{ .cut = 100, .say = 100 }, // as long, diverging
            .{ .cut = 100, .say = 300 }, // longer, diverging
        };

        fn turn(ld: *Load, dir: []const u8, ids: []const u32, taken: bool, sh: Shape, out: *std.ArrayList(u32)) !void {
            const a = testing.allocator;
            var sv: Served = undefined;
            try sv.initTiers(ld, dir, 1 << 30);
            defer sv.deinit();
            var conv = try PartialShare.serve(&sv, ids, 0);
            defer conv.deinit();
            const e = conv.entry.items;
            var prompt: std.ArrayList(u32) = .empty;
            defer prompt.deinit(a);
            try prompt.appendSlice(a, e[0 .. e.len - sh.cut]);
            var said = try heldPrompt(a, ids, sh.say, 7);
            defer said.deinit(a);
            try prompt.appendSlice(a, said.items);
            // A diverging shape really leaves the entry where its new text starts.
            if (sh.cut > 0 and sh.say > 0) try testing.expect(said.items[0] != e[e.len - sh.cut]);
            const fits_was = Served.scheduler.prefill_admission_fits;
            defer Served.scheduler.prefill_admission_fits = fits_was;
            if (taken) Served.scheduler.prefill_admission_fits = fitsOnlyTaken;
            var f: u32 = 0;
            var got = try sv.request(prompt.items, 16, &f);
            defer got.deinit(a);
            try testing.expect(sv.reused > 0);
            try out.appendSlice(a, got.items);
            try testing.expect(takeoverAtRest(&sv.entry.prefix_cache.?));
        }

        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const prefix_cache = @import("prefix_cache.zig");
            const ld = try heldLoad(Q, dir);
            defer prefix_cache.ssd_first_override = null;
            for ([_]bool{ true, false }) |ssd_first| {
                prefix_cache.ssd_first_override = ssd_first;
                for (shapes) |sh| {
                    errdefer std.debug.print("cooler contract: the turn of E less {d} tokens then {d} new, SSD-first {any}\n", .{ sh.cut, sh.say, ssd_first });
                    var shared: std.ArrayList(u32) = .empty;
                    defer shared.deinit(a);
                    try turn(ld, dir, ids, false, sh, &shared);
                    var taken: std.ArrayList(u32) = .empty;
                    defer taken.deinit(a);
                    try turn(ld, dir, ids, true, sh, &taken);
                    try testing.expectEqualSlices(u32, shared.items, taken.items);
                }
            }
        }
    }.run);
}

test "cooler contract: a served Qwen takeover turn cancelled as its prefill ends leaves the hot cache at rest, and the next turn restores and says what the share arm's does" {
    try qwenTinyServed(struct {
        /// The #109 conversation, its turn 3 cancelled once its prefill ends, then a turn 4 that extends turn 3's prompt.
        fn conversation(ld: *Load, dir: []const u8, ids: []const u32, taken: bool, reused: *u32, out: *std.ArrayList(u32)) !void {
            const a = testing.allocator;
            const scheduler = Served.scheduler;
            var sv: Served = undefined;
            try sv.initTiers(ld, dir, 1 << 30);
            defer sv.deinit();
            var c = try PartialShare.serve(&sv, ids, 200);
            defer c.deinit();
            {
                const fits_was = scheduler.prefill_admission_fits;
                defer scheduler.prefill_admission_fits = fits_was;
                if (taken) scheduler.prefill_admission_fits = fitsOnlyTaken;
                const sch = &sv.sch;
                const slot = try sch.submit(.{ .model = sv.entry, .prompt_ids = c.third.items, .sampling = .{ .temperature = 0.0 }, .eos_token_ids = &.{}, .max_tokens = 16 });
                _ = sch.pending.orderedRemove(0);
                try scheduler.runPrefill(sch, slot);
                slot.cancelled.store(true, .release);
                try testing.expect(scheduler.postPrefillTerminal(sch, slot));
                sch.complete(slot);
                for (sch.cleanup_queue.items) |s| {
                    scheduler.cleanupSlot(sch, s);
                    const turn = arch_handle.turnFor(s.model.handle, s.model.config);
                    defer turn.end();
                    s.deinit();
                }
                sch.cleanup_queue.clearRetainingCapacity();
                const hc = &sv.entry.prefix_cache.?;
                hc.flushPendingDisk(arch_handle.trunk(sv.entry.handle.?).s);
                hc.disk.?.drainWriter();
                try testing.expect(takeoverAtRest(hc));
            }
            var fourth: std.ArrayList(u32) = .empty;
            defer fourth.deinit(a);
            try fourth.appendSlice(a, c.third.items);
            var said = try heldPrompt(a, ids, 300, 6);
            defer said.deinit(a);
            try fourth.appendSlice(a, said.items);
            var f: u32 = 0;
            var got = try sv.request(fourth.items, 16, &f);
            defer got.deinit(a);
            reused.* = sv.reused;
            try out.appendSlice(a, got.items);
        }

        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            var shared_reused: u32 = 0;
            var taken_reused: u32 = 0;
            var shared: std.ArrayList(u32) = .empty;
            defer shared.deinit(a);
            var taken: std.ArrayList(u32) = .empty;
            defer taken.deinit(a);
            try conversation(ld, dir, ids, false, &shared_reused, &shared);
            try conversation(ld, dir, ids, true, &taken_reused, &taken);
            try testing.expect(taken_reused > 0);
            try testing.expectEqual(shared_reused, taken_reused);
            try testing.expectEqualSlices(u32, shared.items, taken.items);
        }
    }.run);
}

test "cooler contract: a served speculative Qwen conversation of takeovers, partial and whole, says its share arm's tokens with the pool poisoned" {
    try qwenTinyServed(struct {
        fn conversation(ld: *Load, dir: []const u8, ids: []const u32, taken: bool, out: *std.ArrayList(u32)) !void {
            const a = testing.allocator;
            var sv: Served = undefined;
            try sv.initTiers(ld, dir, 1 << 30);
            defer sv.deinit();
            const poison: ?u8 = if (taken) 0xFF else null;
            mlx.engine.poison.byte = poison;
            defer mlx.engine.poison.byte = null;
            var f: u32 = 0;
            const fits_was = Served.scheduler.prefill_admission_fits;
            defer Served.scheduler.prefill_admission_fits = fits_was;
            if (taken) Served.scheduler.prefill_admission_fits = fitsOnlyTaken;
            var prompt = try heldPrompt(a, ids, 1500, 0);
            defer prompt.deinit(a);
            const plans = [_]struct { cut: usize, say: usize, seed: usize }{
                .{ .cut = 0, .say = 1500, .seed = 2 },
                .{ .cut = 10, .say = 200, .seed = 1 }, // re-renders the reply's tail: a partial share of a speculative entry
                .{ .cut = 0, .say = 300, .seed = 4 }, // extends: a whole-entry hit
                .{ .cut = 300, .say = 400, .seed = 5 }, // re-renders inside the last prompt: partial again
            };
            var last: std.ArrayList(u32) = .empty;
            defer last.deinit(a);
            for (0..plans.len + 1) |k| {
                if (k > 0) {
                    const pl = plans[k - 1];
                    try prompt.appendSlice(a, last.items);
                    prompt.shrinkRetainingCapacity(prompt.items.len - pl.cut);
                    var said = try heldPrompt(a, ids, pl.say, pl.seed);
                    defer said.deinit(a);
                    try prompt.appendSlice(a, said.items);
                }
                if (poison) |b| _ = mlx.engine.poison.free(b);
                var r = try sv.requestWith(prompt.items, 24, .{ .temperature = 0.0 }, true, &f);
                defer r.deinit(a);
                try out.appendSlice(a, r.items);
                last.clearRetainingCapacity();
                try last.appendSlice(a, r.items);
                try testing.expect(takeoverAtRest(&sv.entry.prefix_cache.?));
            }
        }

        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            var shared: std.ArrayList(u32) = .empty;
            defer shared.deinit(a);
            try conversation(ld, dir, ids, false, &shared);
            var taken: std.ArrayList(u32) = .empty;
            defer taken.deinit(a);
            try conversation(ld, dir, ids, true, &taken);
            try testing.expectEqualSlices(u32, shared.items, taken.items);
        }
    }.run);
}

test "cooler contract: a served Qwen takeover turn, whole or partial, is credited the capacity it writes in place, and holds no more in use than the exact part of its bill" {
    try eachQwenServed(struct {
        /// The warm inputs of the last admission probe that took the entry over.
        var taken: ?@import("bills.zig").WarmPrefix = null;
        var seq: usize = 0;

        fn fitsTaken(_: *const @import("model.zig").ModelConfig, prompt_len: usize, _: u32, _: @import("trunk.zig").KVQuantConfig, _: bool, matched: u64, capacity: u64, will_donate: bool, mtp: bool, cps: @import("scheduler.zig").CheckpointPlan) bool {
            if (matched > 0 and will_donate) {
                taken = .{ .matched_tokens = matched, .capacity_tokens = capacity, .will_donate = true, .mtp_on = mtp, .checkpoints = cps.keep, .released = cps.released, .thought = cps.thought };
                seq = prompt_len;
            }
            return matched == 0 or will_donate;
        }

        const Turn = struct { sv: *Served, prompt: []const u32 };

        fn go(q: Turn) !void {
            var f: u32 = 0;
            var got = try q.sv.request(q.prompt, 16, &f);
            got.deinit(testing.allocator);
        }

        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const bills = @import("bills.zig");
            const ld = try heldLoad(Q, dir);
            const c = &ld.config;
            const kv = bills.configuredKvQuantFor(c);
            const kv_bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
            const chunk: u64 = c.prefillStartWidth();
            // The unnamed small tensors of a request: well under one recurrent state set.
            const small: u64 = 4 << 20;
            var sv: Served = undefined;
            try sv.initTiers(ld, dir, 1 << 30);
            defer sv.deinit();
            var conv = try PartialShare.serve(&sv, ids, 200);
            defer conv.deinit();
            const fits_was = Served.scheduler.prefill_admission_fits;
            defer Served.scheduler.prefill_admission_fits = fits_was;
            Served.scheduler.prefill_admission_fits = fitsTaken;
            defer taken = null;
            // A whole take (turn 3 extends turn 2's entry), then a partial one (turn 4 re-renders turn 3's tail).
            var third: std.ArrayList(u32) = .empty;
            defer third.deinit(a);
            try third.appendSlice(a, conv.entry.items);
            var said = try heldPrompt(a, ids, 100, 4);
            defer said.deinit(a);
            try third.appendSlice(a, said.items);
            var fourth: std.ArrayList(u32) = .empty;
            defer fourth.deinit(a);
            for ([_]bool{ false, true }) |partial| {
                errdefer std.debug.print("cooler contract: {s}: the {s} takeover turn\n", .{ dir, if (partial) "partial" else "whole" });
                const prompt = if (partial) blk: {
                    try fourth.appendSlice(a, third.items[0 .. third.items.len - 50]);
                    var more = try heldPrompt(a, ids, 100, 5);
                    defer more.deinit(a);
                    try fourth.appendSlice(a, more.items);
                    break :blk fourth.items;
                } else third.items;
                taken = null;
                const peak = try inUsePeak(Turn{ .sv = &sv, .prompt = prompt }, go);
                const warm = taken orelse return error.TestTurnNotTakenOver;
                // The buffers the turn took hold the last turn's reservation, past the rows it shares.
                try testing.expect(warm.capacity_tokens > warm.matched_tokens);
                const reserved = bills.cacheStorage(c, seq, 16, kv_bits, chunk, warm).rows;
                try testing.expectEqual(@min(warm.capacity_tokens, reserved), warm.creditedRows(reserved));
                const exact = bills.requestHeldBytes(c, seq, 16, kv_bits, chunk, warm);
                if (peak > exact + small) std.debug.print("cooler contract: {d} KB in use at the peak, {d} KB past the exact bill ({d} KB)\n", .{ peak >> 10, (peak - exact) >> 10, exact >> 10 });
                try testing.expect(peak <= exact + small);
            }
        }
    }.run);
}

test "cooler contract: a served Qwen partial takeover refused before it donates leaves no cut on the entry, so the next turn takes the entry whole and keeps its restore points" {
    try qwenTinyServed(struct {
        /// Admission that refuses every warm request, taken over or shared.
        fn fitsNoWarm(_: *const @import("model.zig").ModelConfig, _: usize, _: u32, _: @import("trunk.zig").KVQuantConfig, _: bool, matched: u64, _: u64, _: bool, _: bool, _: @import("scheduler.zig").CheckpointPlan) bool {
            return matched == 0;
        }

        /// The #109 conversation, (its turn 3 taken over and then refused), and a turn 4 that extends turn 2's entry:
        /// the restore points of the entry turn 4 leaves.
        fn conversation(ld: *Load, dir: []const u8, ids: []const u32, refused_turn: bool, out: *std.ArrayList(usize)) !void {
            const a = testing.allocator;
            var sv: Served = undefined;
            try sv.initTiers(ld, dir, 1 << 30);
            defer sv.deinit();
            sv.quiet_failures = true;
            var c = try PartialShare.serve(&sv, ids, 200);
            defer c.deinit();
            const hc = &sv.entry.prefix_cache.?;
            if (refused_turn) {
                const fits_was = Served.scheduler.prefill_admission_fits;
                defer Served.scheduler.prefill_admission_fits = fits_was;
                Served.scheduler.prefill_admission_fits = fitsNoWarm;
                _ = try sv.requestToCleanup(c.third.items, 16, false);
                // Refused before its append donated: turn 2's entry is back, whole.
                try testing.expect(takeoverAtRest(hc));
                var whole = false;
                for (hc.entries.items) |*e| whole = whole or std.mem.eql(u32, e.tokens, c.entry.items);
                try testing.expect(whole);
            }
            var fourth: std.ArrayList(u32) = .empty;
            defer fourth.deinit(a);
            try fourth.appendSlice(a, c.entry.items);
            var said = try heldPrompt(a, ids, 1500, 5);
            defer said.deinit(a);
            try fourth.appendSlice(a, said.items);
            var f: u32 = 0;
            var got = try sv.request(fourth.items, 16, &f);
            defer got.deinit(a);
            for (hc.entries.items) |*e| {
                if (e.tokens.len < fourth.items.len) continue;
                if (e.ssm_checkpoints) |cps| for (cps) |*cp| try out.append(a, cp.pos);
            }
        }

        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            var clean: std.ArrayList(usize) = .empty;
            defer clean.deinit(a);
            var after_refusal: std.ArrayList(usize) = .empty;
            defer after_refusal.deinit(a);
            try conversation(ld, dir, ids, false, &clean);
            try conversation(ld, dir, ids, true, &after_refusal);
            try testing.expect(clean.items.len > 0);
            try testing.expectEqualSlices(usize, clean.items, after_refusal.items);
        }
    }.run);
}

test "cooler contract: a held Qwen restore point another conversation pushes out of RAM is written, not dropped, and restores" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            const disk = &sv.entry.prefix_cache.?.disk.?;
            var prompts: [3]std.ArrayList(u32) = undefined;
            for (&prompts, 0..) |*p, k| p.* = try heldPrompt(a, ids, 3000, k + 1);
            defer for (&prompts) |*p| p.deinit(a);
            var f: u32 = 0;
            var r0 = try sv.request(prompts[0].items, 8, &f);
            r0.deinit(a);
            disk.drainWriter();
            try testing.expectEqual(@as(usize, 1), disk.held.items.len);
            const one = disk.held_bytes;
            disk.held_cap_for_test = 2 * one + one / 2;
            var r1 = try sv.request(prompts[1].items, 8, &f);
            r1.deinit(a);
            disk.drainWriter();
            try testing.expectEqual(@as(usize, 2), disk.held.items.len);
            const before = ssmWritten();
            // A third conversation's point: the oldest (the first conversation's) is written, not dropped.
            var r2 = try sv.request(prompts[2].items, 8, &f);
            r2.deinit(a);
            disk.drainWriter();
            try testing.expectEqual(@as(usize, 2), disk.held.items.len);
            try testing.expectEqual(one, ssmWritten() - before);
            var again = try sv.request(prompts[0].items, 8, &f);
            again.deinit(a);
            try testing.expectEqual(@as(u32, 3000 - 31), sv.reused);
        }
    }.run);
}

test "cooler contract: a held Qwen restore point restores the same tokens as its file after a flush" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            const disk = &sv.entry.prefix_cache.?.disk.?;
            var prompt = try heldPrompt(a, ids, 3000, 7);
            defer prompt.deinit(a);
            var f: u32 = 0;
            var cold = try sv.request(prompt.items, 8, &f);
            cold.deinit(a);
            disk.drainWriter();
            const before = ssmWritten();
            var from_ram = try sv.request(prompt.items, 24, &f);
            defer from_ram.deinit(a);
            disk.drainWriter();
            try testing.expectEqual(@as(u32, 3000 - 31), sv.reused);
            try testing.expectEqual(before, ssmWritten());
            // The clean unload's flush: the held point is written, and the next request restores it from its file.
            @import("scheduler.zig").flushLiveState(sv.entry, "test");
            disk.drainWriter();
            try testing.expectEqual(@as(usize, 0), disk.held.items.len);
            try testing.expect(ssmWritten() > before);
            var from_file = try sv.request(prompt.items, 24, &f);
            defer from_file.deinit(a);
            try testing.expectEqual(@as(u32, 3000 - 31), sv.reused);
            try testing.expectEqualSlices(u32, from_ram.items, from_file.items);
        }
    }.run);
}

test "cooler contract: a regenerated Qwen turn restores from its own prompt end, live and after a flush" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            const disk = &sv.entry.prefix_cache.?.disk.?;
            var prompt = try heldPrompt(a, ids, 3000, 11);
            defer prompt.deinit(a);
            var f: u32 = 0;
            var first = try sv.request(prompt.items, 16, &f);
            first.deinit(a);
            disk.drainWriter();
            // The same conversation's next point extends the prompt end (a reply end, or the turn after).
            var longer = try prompt.clone(a);
            defer longer.deinit(a);
            for (0..64) |i| try longer.append(a, ids[(i * 13 + 5) % ids.len]);
            var next = try sv.request(longer.items, 16, &f);
            next.deinit(a);
            disk.drainWriter();
            // Regenerate: the first prompt again restores from its own end, which the newer point did not take away.
            var live = try sv.request(prompt.items, 16, &f);
            defer live.deinit(a);
            disk.drainWriter();
            try testing.expectEqual(@as(u32, 3000 - 31), sv.reused);
            // After the flush an unload or a restart runs, it restores from the written point.
            @import("scheduler.zig").flushLiveState(sv.entry, "test");
            disk.drainWriter();
            try testing.expectEqual(@as(usize, 0), disk.held.items.len);
            var after = try sv.request(prompt.items, 16, &f);
            defer after.deinit(a);
            try testing.expectEqual(@as(u32, 3000 - 31), sv.reused);
            try testing.expectEqualSlices(u32, live.items, after.items);
        }
    }.run);
}

test "cooler contract: a Qwen prompt that re-renders an earlier reply restores from that turn's own point, written when the held points pass the cap" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            const disk = &sv.entry.prefix_cache.?.disk.?;
            // Over the 8192 grid, so a written grid point lies under the turns' held points.
            var conv = try heldPrompt(a, ids, 9000, 41);
            defer conv.deinit(a);
            var f: u32 = 0;
            // The second turn's prompt and the first 8 tokens of its reply: an agent's later prompt that re-renders
            // the rest of that reply (a coerced tool call) shares this much.
            var shared: usize = 0;
            for (0..3) |turn| {
                var reply = try sv.request(conv.items, 16, &f);
                defer reply.deinit(a);
                try expectListedPointsLive(disk);
                // Room for one point and a half: a commit keeps its newest point held and writes the spare it supersedes.
                if (turn == 0) disk.held_cap_for_test = disk.held_bytes / disk.held.items.len * 3 / 2;
                if (turn == 1) shared = conv.items.len + 8;
                try conv.appendSlice(a, reply.items);
                for (0..600) |i| try conv.append(a, ids[(i * 13 + turn * 101) % ids.len]);
            }
            var next = try conv.clone(a);
            defer next.deinit(a);
            next.shrinkRetainingCapacity(shared);
            for (0..200) |i| try next.append(a, ids[(i * 17 + 3) % ids.len]);
            var last = try sv.request(next.items, 16, &f);
            last.deinit(a);
            // The second turn's prompt-end point, not the 8192 grid point under it.
            try testing.expectEqual(@as(u32, @intCast(shared - 8 - 31)), sv.reused);
        }
    }.run);
}

test "cooler contract: a held Qwen restore point whose write fails as it gives way to a staged write mid-commit leaves no listed point without bytes" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            const disk = &sv.entry.prefix_cache.?.disk.?;
            var conv = try heldPrompt(a, ids, 3000, 51);
            defer conv.deinit(a);
            var f: u32 = 0;
            var first = try sv.request(conv.items, 16, &f);
            try conv.appendSlice(a, first.items);
            first.deinit(a);
            try expectListedPointsLive(disk);
            const one = disk.held_bytes;
            try testing.expect(one > 0);
            // The next turn's commit stages its chunks with too little permit beside the held points, and their writes fail.
            const permit = disk.writer.?.permit_bytes;
            defer disk.writer.?.permit_bytes = permit;
            disk.writer.?.permit_bytes = one + (1 << 20);
            disk.fail_held_flush_for_test = true;
            defer disk.fail_held_flush_for_test = false;
            for (0..1500) |i| try conv.append(a, ids[(i * 13 + 7) % ids.len]);
            var next = try sv.request(conv.items, 16, &f);
            next.deinit(a);
            try testing.expectEqual(@as(usize, 0), disk.held.items.len);
            try expectListedPointsLive(disk);
        }
    }.run);
}

test "cooler contract: held Qwen restore points give way to a staged write that needs the writer's permit" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            var sv: Served = undefined;
            try sv.init(ld, dir);
            defer sv.deinit();
            const disk = &sv.entry.prefix_cache.?.disk.?;
            var p0 = try heldPrompt(a, ids, 3000, 11);
            defer p0.deinit(a);
            var p1 = try heldPrompt(a, ids, 3000, 12);
            defer p1.deinit(a);
            var f: u32 = 0;
            var r0 = try sv.request(p0.items, 8, &f);
            r0.deinit(a);
            disk.drainWriter();
            const one = disk.held_bytes;
            try testing.expect(one > 0);
            // Room for the held points, but not beside the next conversation's chunk writes.
            disk.held_cap_for_test = 4 * one;
            const permit = disk.writer.?.permit_bytes;
            defer disk.writer.?.permit_bytes = permit;
            disk.writer.?.permit_bytes = one + (1 << 20);
            const before = ssmWritten();
            var r1 = try sv.request(p1.items, 8, &f);
            r1.deinit(a);
            disk.drainWriter();
            try testing.expectEqual(@as(usize, 0), disk.held.items.len);
            try testing.expect(ssmWritten() - before >= one);
        }
    }.run);
}

test "cooler contract: a rewind past a Qwen conversation's held restore points restores from the grid point, the same tokens every time" {
    // Its tokens are not the cold run's: the grid point's state came from a prefill that started at the turn before's
    // point, off a cold prefill's split, as any restore from another prompt's record is (the shorter-prompt SSD cell).
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const ld = try heldLoad(Q, dir);
            var said: [2]std.ArrayList(u32) = undefined;
            for (&said) |*got| {
                var reused: u32 = 0;
                got.* = try rewindPastHeld(ld, dir, ids, &reused);
                try testing.expectEqual(@as(u32, 8192), reused);
            }
            defer for (&said) |*got| got.deinit(a);
            try testing.expectEqualSlices(u32, said[0].items, said[1].items);
        }
    }.run);
}

/// One served conversation on a fresh tier: `long` cold, another conversation, then `long` again, restored from SSD.
const SsdRestore = struct {
    ld: *Load,
    dir: []const u8,
    long: []const u32,
    other: []const u32,
    restored: ?*std.ArrayList(u32) = null,
    /// The two cold requests alone, no restore.
    cold_only: bool = false,

    fn go(self: SsdRestore) anyerror!void {
        const a = testing.allocator;
        var sv: Served = undefined;
        try sv.init(self.ld, self.dir);
        defer sv.deinit();
        sv.quiet_failures = true;
        var f: u32 = 0;
        var cold = try sv.request(self.long, 12, &f);
        cold.deinit(a);
        var o = try sv.request(self.other, 4, &f);
        o.deinit(a);
        if (self.cold_only) return;
        const got = try sv.request(self.long, 12, &f);
        if (self.restored) |out| out.* = got else {
            var g = got;
            g.deinit(a);
        }
    }
};

fn ssmWritten() u64 {
    return @import("disk_write_stats.zig").process.snapshot().written[@backingInt(@import("disk_write_stats.zig").Kind.ssm)];
}

/// Every restore point the SSD tier's index names restores from where it lives: its held bytes or its file.
fn expectListedPointsLive(disk: *@import("kv_disk_cache.zig").DiskTier) !void {
    disk.drainWriter();
    for (disk.entries.items) |e| for (e.ssm_positions) |p| {
        if (disk.heldIndex(e.id, p) != null) continue;
        var buf: [1024]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/e{d}/s{d:0>7}.safetensors", .{ disk.root, e.id, p });
        errdefer std.debug.print("cooler contract: e{d} lists a restore point at {d} that is neither held nor on disk\n", .{ e.id, p });
        try testing.expect(disk.fs.size(path) != null);
    };
}

/// A four-turn Qwen conversation over the 8192 grid on a fresh tier, then a rewind to its second turn's prompt, whose
/// own held point the turns after it superseded: the rewind's tokens, and the prompt tokens it took from a tier.
fn rewindPastHeld(ld: *Load, dir: []const u8, ids: []const u32, reused: *u32) !std.ArrayList(u32) {
    const a = testing.allocator;
    var conv = try heldPrompt(a, ids, 7000, 21);
    defer conv.deinit(a);
    var rewind: std.ArrayList(u32) = .empty;
    defer rewind.deinit(a);
    var sv: Served = undefined;
    try sv.init(ld, dir);
    defer sv.deinit();
    const disk = &sv.entry.prefix_cache.?.disk.?;
    var f: u32 = 0;
    for (0..4) |turn| {
        var reply = try sv.request(conv.items, 8, &f);
        defer reply.deinit(a);
        disk.drainWriter();
        try conv.appendSlice(a, reply.items);
        for (0..1500) |i| try conv.append(a, ids[(i * 13 + turn * 101) % ids.len]);
        if (turn == 1) try rewind.appendSlice(a, conv.items[0 .. conv.items.len - 1500]);
    }
    const back = try sv.request(rewind.items, 16, &f);
    reused.* = sv.reused;
    return back;
}
