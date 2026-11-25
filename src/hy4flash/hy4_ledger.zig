//! Hy4-Flash's bill (the `bill` slot of `arch.zig`, docs/bill_redesign.md part 9): its state rates, what a request's
//! stores hold at the pool's size classes, and a request's bill on the ledger in closed form over its reach. Its forwards
//! run on its plans and in the region its load measured (D97, D98), which the model bill holds.
const std = @import("std");
const model_mod = @import("../model.zig");
const ledger = @import("../ledger.zig");
const bills = @import("../bills.zig");
const hy4_config = @import("hy4_config.zig");
const hy4_model = @import("hy4_model.zig");
const pool_class = @import("../tensor/pool_class.zig");
const prefill_plan = @import("../prefill_plan.zig");
const settings_resolve = @import("../settings_resolve.zig");

pub const Terms = ledger.RequestTerms;

const f32b: u64 = 4;
const bf16b: u64 = 2;

/// Bytes the state keeps a token, at `kv_bits` (8: the engine's affine kv8; 16: BF16): each sparse layer's latent and
/// rope row, and its share of a pooled index key (f32, one a group).
pub fn stateBytesPerToken(c: *const hy4_config.Config, kv_bits: u64) u64 {
    return @as(u64, c.sparseCount()) * (latentRowBytes(c, kv_bits) + @as(u64, c.indexer.head_dim) * f32b / c.indexer.group);
}

/// The part of a token's state a KV width narrows, as BF16: each sparse layer's latent (the arch's `kvBytesPerToken`;
/// `bills.kvBytesPerTokenAtBits` takes it to the 8-bit codes with a group's scale and bias exactly).
pub fn latentBytesPerToken(c: *const hy4_config.Config) u64 {
    return @as(u64, c.sparseCount()) * c.kv_lora * bf16b;
}

/// A latent row as stored: codes with a BF16 scale and bias a group of 64 at 8 bits, else BF16; the rope values BF16.
pub fn latentRowBytes(c: *const hy4_config.Config, kv_bits: u64) u64 {
    const lat: u64 = c.kv_lora;
    const latent = if (kv_bits == 8) lat + lat / 64 * 2 * bf16b else lat * bf16b;
    return latent + @as(u64, c.rope_dim) * bf16b;
}

/// What a request keeps whatever its length: each sliding layer's ring of `window` rows, each sparse layer's open
/// group (fewer than `ratio` f32 token keys).
pub fn stateFixedBytes(c: *const hy4_config.Config, kv_bits: u64) u64 {
    const sliding: u64 = c.layers - c.sparseCount();
    return sliding * c.window * latentRowBytes(c, kv_bits) + @as(u64, c.sparseCount()) * (c.indexer.group - 1) * c.indexer.head_dim * f32b;
}

/// A sum of arrays at the pool's size classes.
const Sum = struct {
    total: u64 = 0,
    fn add(self: *Sum, count: u64, bytes: u64) void {
        self.total +|= count *| pool_class.poolBytes(@intCast(bytes));
    }
};

/// One complete state of a request `rows` tokens deep, each array a pool buffer at its size class: a sparse layer's
/// latent (its scales and biases at 8 bits), rope rows and pooled keys, each a store grown in place to at most
/// `capRows` of the rows it holds, and its open group; a sliding layer's ring of `window` rows in the same arrays.
pub fn stateBytes(c: *const hy4_config.Config, kv_bits: u64, rows: u64) u64 {
    const sparse: u64 = c.sparseCount();
    const g: u64 = c.indexer.group;
    var s: Sum = .{};
    if (rows > 0) rowArrays(&s, c, kv_bits, sparse, capRows(rows));
    rowArrays(&s, c, kv_bits, c.layers - sparse, c.window);
    if (rows / g > 0) s.add(sparse, capRows(rows / g) * c.indexer.head_dim * f32b);
    // Each sparse layer's open group block, `ratio - 1` keys whatever the rows (`Model.reserveFor`).
    if (g > 1) s.add(sparse, (g - 1) * c.indexer.head_dim * f32b);
    return s.total;
}

/// The most rows a store that has held `rows` has (`row_store.capBound`): the rows and the slack of its last grow, a
/// `hy4_model.rows_grow_div`th of them, 64 at least.
pub fn capRows(rows: u64) u64 {
    return @import("../row_store.zig").capBound(@intCast(rows), hy4_model.rows_grow_div);
}

/// A point within the state (`hy4_model.Model.Point`), whatever its position: each sliding layer's ring, its own once
/// the state's next write copies it, and each sparse layer's open group at its widest. A drafter's hidden state apart.
pub fn pointBytes(c: *const hy4_config.Config, kv_bits: u64) u64 {
    var s: Sum = .{};
    rowArrays(&s, c, kv_bits, c.layers - c.sparseCount(), c.window);
    if (c.indexer.group > 1) s.add(c.sparseCount(), (c.indexer.group - 1) * @as(u64, c.indexer.head_dim) * f32b);
    return s.total;
}

/// The `Points` slots made with a request's state (`Model.point_slots`: the running request's points and the live
/// record's), each a point's small state at its widest, with the hidden state where the model keeps it
/// (`Model.keep_hidden`); whatever the context (`hy4_prefix.ram_tier_max_bytes` bounds it).
pub fn pointsBytes(c: *const hy4_config.Config, kv_bits: u64, keep_hidden: bool) u64 {
    return @as(u64, hy4_model.Model.point_slots) *| (pointBytes(c, kv_bits) +| (if (keep_hidden) hiddenBytes(c) else 0));
}

/// A drafting load's mark blocks (`Model.mark_fixed`, `mark_hidden`), reserved with every request's state: a point's
/// small state at its widest and the hidden state.
pub fn markBytes(c: *const hy4_config.Config, kv_bits: u64) u64 {
    return pointBytes(c, kv_bits) +| hiddenBytes(c);
}

/// `layers` layers' latent (codes, scales and biases at 8 bits; BF16 at 16) and rope rows, `held` rows each.
fn rowArrays(s: *Sum, c: *const hy4_config.Config, kv_bits: u64, layers: u64, held: u64) void {
    const lat: u64 = c.kv_lora;
    if (kv_bits == 8) {
        s.add(layers, held * lat);
        s.add(2 * layers, held * (lat / 64) * bf16b);
    } else s.add(layers, held * lat * bf16b);
    s.add(layers, held * c.rope_dim * bf16b);
}

/// What a model that drafts keeps of its last forward beside the state: each sparse layer's selection for the last row
/// (`Model.last_picks`: the key rows, `group_topk · group` u32, and which it reads, one byte each), and the hidden state
/// the draft starts from, which a round's mark holds beside the new one.
pub fn pickBytes(c: *const hy4_config.Config) u64 {
    const width: u64 = @as(u64, c.indexer.group_topk) * c.indexer.group;
    var s: Sum = .{};
    s.add(c.sparseCount(), width * 4);
    s.add(c.sparseCount(), width);
    const rows: u64 = @intCast(@import("hy4_forward.zig").verify_rows_max);
    s.add(c.sparseCount(), rows * width * 4);
    s.add(c.sparseCount(), rows * width);
    s.add(c.sparseCount(), rows * c.indexer.head_dim * f32b);
    s.add(1, rows * c.hc_mult * c.dim * f32b);
    return s.total;
}

pub fn hiddenBytes(c: *const hy4_config.Config) u64 {
    var s: Sum = .{};
    s.add(1, @as(u64, c.dim) * f32b);
    return s.total;
}

/// What a prefill's stored restore points may copy beside the state (`hy4_prefix.captureCheckpoint`): with a
/// write-through (SSD-first), the point a chunk takes shares the state's rows until the writer copies it into its own
/// bounded host buffer, at most the writer's permit (`RequestTerms.copy_cap`: the copy is the request's state, up to
/// this); with none, a point is its position alone, its live point a slot `pointsBytes` bills. 0 where no checkpoint
/// is taken.
pub fn checkpointCap(config: *const model_mod.ModelConfig, chunk: u64) u64 {
    const base = bills.effectiveSsmCheckpointStride(bills.launch_state.ssm_checkpoint_stride, bills.launch_state.prefix_cache_capacity, bills.launch_state.prefix_cache_ram_enabled, bills.diskBytesFor(config));
    const stride: u64 = prefill_plan.effectiveSsmCheckpointStride(base, @max(@as(usize, @intCast(chunk)), settings_resolve.prefill_chunk.value));
    if (stride == 0 or bills.launch_state.prefix_cache_ram_enabled or bills.diskBytesFor(config) == 0) return 0;
    return bills.launch_state.disk_writer_permit_bytes;
}

/// A state store reserved for `rows` holds `capRows` of them (`rows + rows / rows_grow_div`, at least 64), each array a
/// pool buffer at its size class (under 1/16 over, and a page): the rate a token as held, both slacks counted.
fn heldRate(rate: u64) u64 {
    const capped = rate +| (std.math.divCeil(u64, rate, hy4_model.rows_grow_div) catch unreachable);
    return capped +| (std.math.divCeil(u64, capped, pool_class.steps) catch unreachable);
}

/// What a Hy4 load's requests hold on the ledger: every load plans, a streamed one with the cache bias too (D102); null
/// for another arch. `chunk` is the load's prefill chunk, which decides where its restore points are written.
pub fn ledgerTerms(config: *const model_mod.ModelConfig, kv_bits: u64, chunk: u64) ?Terms {
    const c = config.hy4flash orelse return null;
    const drafts = @import("arch.zig").drafterOn(config) and bills.mtpHeadDefaultOn(config);
    const rate = heldRate(stateBytesPerToken(&c, kv_bits));
    // Each sparse layer's stores (latent parts, rope rows, pooled keys): their 64-row floor and the one-row grow of a
    // short reach, a pooled row a group of tokens, and a page apiece.
    const sparse: u64 = c.sparseCount();
    const floor_rows: u64 = 65 * @as(u64, c.indexer.group);
    const floors = heldRate(floor_rows *| stateBytesPerToken(&c, kv_bits)) +| sparse * 5 * pool_class.page_bytes;
    // A speculative round writes up to a verify block's rows past the reply, and keeps its mark and handover.
    const spec_rows: u64 = if (drafts) @import("../sampler_draw.zig").max_rows - 1 else 0;
    // The handover, the mark and each point's hidden row are held by every request of a load that drafts (D56).
    const holds = @import("arch.zig").drafterOn(config);
    const handover = if (holds) pickBytes(&c) +| hiddenBytes(&c) +| markBytes(&c, kv_bits) else 0;
    return .{
        .per_token = rate,
        .fixed = stateBytes(&c, kv_bits, 0) +| floors +| rate *| spec_rows +| handover,
        .points = pointsBytes(&c, kv_bits, holds),
        .copy_cap = checkpointCap(config, chunk),
    };
}

const testing = std.testing;
const static_bill = @import("../static_bill.zig");

test "hy4 bill: a refusal beside a running request names the max_tokens a re-send is held for and then served at, on the connection thread's wire too" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigText(io, a, "", @import("hy4_config.zig").release_config);
    defer config.deinit(a);
    var l: ledger.Ledger = .{};
    config.static_prefill_chunk = 4096;
    config.pinned_prefill_chunk = 4096;
    config.static_ledger = &l;
    config.pinned_context = 1 << 20;
    l.terms = ledgerTerms(&config, bills.defaultKvBits(&config), 4096).?;
    l.setBudget(64 << 30);
    try l.reserve(.weights, 56 << 30, null);
    // The running request's line.
    try l.reserve(.state, l.terms.?.at(1000 + 200_000), null);
    var prompt: [1000]u32 = undefined;
    for (&prompt, 0..) |*p, i| p.* = @intCast(i + 10);
    const asked: u32 = 1_000_000;
    const refused = switch (static_bill.check(&config, prompt.len, asked, null)) {
        .refused => |r| r,
        .proceed => return error.TestExpectedRefusal,
    };
    try testing.expect(refused.fitting_max_tokens > 0 and refused.fitting_max_tokens < asked);
    // The room is what the running request's end frees: a re-send at the named max_tokens is held for it and fits then.
    try testing.expect(static_bill.check(&config, prompt.len, refused.fitting_max_tokens, null) == .proceed);
    try testing.expect(static_bill.check(&config, prompt.len, refused.fitting_max_tokens + 1, null) == .refused);
    // The connection thread answers with those figures, and lets the re-send through.
    const server = @import("../server.zig");
    var fds: [2]std.posix.fd_t = undefined;
    try testing.expect(std.c.socketpair(1, 1, 0, &fds) == 0);
    defer _ = std.c.close(fds[1]);
    var conn: @import("../http_conn.zig").Conn = undefined;
    @import("../http_conn.zig").Conn.init(&conn, .{ .socket = .{ .handle = fds[0], .address = undefined } }, testing.io);
    try testing.expect(!try server.checkAttentionMemory(a, &conn, &prompt, asked, &config, null, false, null, false, false, null));
    try testing.expect(try server.checkAttentionMemory(a, &conn, &prompt, refused.fitting_max_tokens, &config, null, false, null, false, false, null));
    _ = std.c.close(fds[0]);
    var buf: [8192]u8 = undefined;
    const body = server.readAllFromPair(fds[1], &buf);
    var named: [64]u8 = undefined;
    const wire = try std.fmt.bufPrint(&named, "max_tokens {d} fits (asked {d})", .{ refused.fitting_max_tokens, asked });
    try testing.expect(std.mem.indexOf(u8, body, wire) != null);
    // Once the running request ends the same re-send is admitted outright.
    l.release(.state, l.of(.state));
    const again = static_bill.reservation(&config, prompt.len, refused.fitting_max_tokens, null);
    try testing.expectEqual(ledger.Verdict.admit, ledger.decide(l.snapshot(), again.total()));
}

test "hy4 bill: a load that streams with a cache bias plans too (D102), so it is on the ledger" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigText(io, a, "", @import("hy4_config.zig").release_config);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.expert_resident = false;
    config.expert_cache_bias_override = 0.2;
    try testing.expect(ledgerTerms(&config, 8, 4096) != null);
}

/// An arena that grows with the chunk, as a load's measured one does (`arch.planArena`).
const LinearArena = struct {
    fn at(_: *anyopaque, chunk: u32) anyerror!u64 {
        return (96 << 20) + @as(u64, chunk) * (640 << 10);
    }

    var token: u8 = 0;
    const arena: bills.ArenaAt = .{ .ctx = &token, .at = at };

    fn of(chunk: u32) u64 {
        return at(&token, chunk) catch unreachable;
    }
};

test "hy4 bill: a load on the ledger prices its context first at the min's terms and arena, then fixes the widest chunk whose measured arena and request fit the room, and refuses with the numbers when the min's do not" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigText(io, a, "", @import("hy4_config.zig").release_config);
    defer config.deinit(a);
    const kv = bills.defaultKvBits(&config);
    const b: bills.ChunkBounds = .{ .min = 512, .max = 8192 };
    var cand_buf: [@import("../settings_resolve.zig").static_chunk_rungs.len]u32 = undefined;
    const candidates = bills.staticChunkCandidates(b, &cand_buf);
    const narrow = ledgerTerms(&config, kv, b.min).?;
    const narrow_arena = LinearArena.of(b.min);
    var seen: [6]bool = @splat(false);
    var last: u32 = 0;
    var room: u64 = 128 << 20;
    while (room < 64 << 30) : (room += 64 << 20) {
        const ctx = ledger.contextFor(room -| narrow_arena, narrow, bills.ledgerWindow(&config));
        switch (try bills.chooseLedgerPrefillChunk(&config, b, .{ .room = room, .arena = LinearArena.arena })) {
            .chunk => |chunk| {
                try testing.expect(ctx > 0);
                const t = ledgerTerms(&config, kv, chunk).?;
                try testing.expect(LinearArena.of(chunk) + t.at(ctx) <= room);
                for (candidates) |w| if (w > chunk) {
                    const tw = ledgerTerms(&config, kv, w).?;
                    try testing.expect(LinearArena.of(w) + tw.at(ctx) > room);
                };
                try testing.expect(chunk >= last);
                last = chunk;
                for (candidates, 0..) |w, i| seen[i] = seen[i] or w == chunk;
            },
            .refused => |r| {
                try testing.expectEqual(@as(u32, 0), last);
                try testing.expectEqual(bills.ContextSource.auto, r.context);
                try testing.expectEqual(narrow_arena + narrow.at(@max(ctx, 1)), r.needed);
                try testing.expect(r.needed > r.free);
                seen[5] = true;
            },
        }
    }
    try testing.expect(seen[5] and seen[4] and seen[0]);
    try testing.expect(seen[1] or seen[2] or seen[3]);
    // An explicit context is priced as given, and its refusal names where it came from: the model's setting, then the flag.
    config.ctx_override = 65536;
    const exact = narrow_arena + narrow.at(65536);
    const setting = try bills.chooseLedgerPrefillChunk(&config, b, .{ .room = exact - 1, .arena = LinearArena.arena });
    try testing.expectEqual(bills.ContextSource.setting, setting.refused.context);
    try testing.expectEqual(exact, setting.refused.needed);
    const launch = &@import("../server_state.zig").server_config;
    const flag0 = launch.max_context_size;
    defer launch.max_context_size = flag0;
    launch.max_context_size = 65536;
    const flagged = try bills.chooseLedgerPrefillChunk(&config, b, .{ .room = exact - 1, .arena = LinearArena.arena });
    try testing.expectEqual(bills.ContextSource.flag, flagged.refused.context);
    try testing.expectEqual(@as(u32, 512), (try bills.chooseLedgerPrefillChunk(&config, b, .{ .room = exact, .arena = LinearArena.arena })).chunk);
}

test "hy4 bill: a load on the ledger refuses before it records its plans only where its choice refuses with no arena at all, a refusal every measured arena then gives too; an auto context only when not one token fits" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigText(io, a, "", @import("hy4_config.zig").release_config);
    defer config.deinit(a);
    const b: bills.ChunkBounds = .{ .min = 512, .max = 8192 };
    const narrow = ledgerTerms(&config, bills.defaultKvBits(&config), b.min).?;
    const launch = &@import("../server_state.zig").server_config;
    const flag0 = launch.max_context_size;
    defer launch.max_context_size = flag0;
    for ([_]u32{ 0, 65536, 1 << 20 }) |manual| {
        launch.max_context_size = manual;
        var early_seen = false;
        var plans_decide_seen = false;
        var room: u64 = 0;
        while (room < 48 << 30) : (room += 16 << 20) {
            const full = try bills.chooseLedgerPrefillChunk(&config, b, .{ .room = room, .arena = LinearArena.arena });
            const early = try bills.refusedBeforePlans(&config, b, room) orelse {
                if (full == .refused) plans_decide_seen = true;
                continue;
            };
            early_seen = true;
            try testing.expect(full == .refused);
            try testing.expectEqual(full.refused.context, early.context);
            try testing.expectEqual(narrow.at(@max(1, if (manual > 0) manual else 0)), early.needed);
            try testing.expect(early.needed <= full.refused.needed and early.needed > room);
        }
        try testing.expect(early_seen and plans_decide_seen);
    }
    // A manual context no chunk's request holds in the whole room is refused before the plans, one byte short; at the
    // byte, the plans decide.
    launch.max_context_size = 65536;
    try testing.expect(try bills.refusedBeforePlans(&config, b, narrow.at(65536) - 1) != null);
    try testing.expect(try bills.refusedBeforePlans(&config, b, narrow.at(65536)) == null);
}

test "hy4 bill: before its load a request on the ledger is priced by one function at the widest chunk the load may fix: the default and the planned context's, and the warm's longest" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigText(io, a, "", @import("hy4_config.zig").release_config);
    defer config.deinit(a);
    @import("../cooler_side.zig").forTest(&config, true);
    const widest = @import("../arch/config.zig").staticChunkMax(&config);
    const terms = bills.ledgerTermsAt(&config, bills.defaultKvBits(&config), widest).?;
    const planned = terms.at(bills.expertStreamingPlannedSeq(&config));
    try testing.expectEqual(terms.at(config.defaultContext()), bills.defaultRequestBill(&config));
    try testing.expectEqual(planned, bills.expertStreamingPlannedKvBytes(&config));
    try testing.expectEqual(planned, bills.expertStreamingServingBytes(&config));
    // The warm runs its longest request at the chunk the load fixes, which the widest bounds.
    try testing.expectEqual(terms.at(bills.warmLongestPrompt(&config, widest) + bills.WARM_REPLY_TOKENS), bills.warmNeedBytes(&config));
}

test "hy4 bill: the widest chunk below the chosen one whose arena and request hold a context beside the model bill, and none when no narrower rung does" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigText(io, a, "", @import("hy4_config.zig").release_config);
    defer config.deinit(a);
    const kv = bills.defaultKvBits(&config);
    var b: bills.ChunkBounds = .{ .min = 512, .max = 8192 };
    const ctx: u64 = 65536;
    const held: u64 = 1 << 30;
    const exact = held + LinearArena.of(2048) + ledgerTerms(&config, kv, 2048).?.at(ctx);
    try testing.expectEqual(@as(?u32, 2048), try bills.chunkHoldingContext(&config, b, 8192, ctx, held, exact, LinearArena.arena));
    try testing.expectEqual(@as(?u32, 1024), try bills.chunkHoldingContext(&config, b, 8192, ctx, held, exact - 1, LinearArena.arena));
    // Only the rungs below the chosen one count, and none below the min.
    try testing.expectEqual(@as(?u32, null), try bills.chunkHoldingContext(&config, b, 512, ctx, held, exact, LinearArena.arena));
    b.min = 4096;
    try testing.expectEqual(@as(?u32, null), try bills.chunkHoldingContext(&config, b, 8192, ctx, held, exact, LinearArena.arena));
}

test "hy4 bill: a request's copy of its state is billed at its own size up to the writer's permit, not at the permit whatever its length" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigText(io, a, "", @import("hy4_config.zig").release_config);
    defer config.deinit(a);
    const kv = bills.defaultKvBits(&config);
    const c = config.hy4flash.?;
    const terms = ledgerTerms(&config, kv, 8192).?;
    try testing.expectEqual(bills.launch_state.disk_writer_permit_bytes, terms.copy_cap);
    // A short request's copy is as short as its state; the points are the Points slots alone, whatever the reach.
    const state_4k = terms.per_token * 4096 + terms.fixed;
    try testing.expectEqual(pointsBytes(&c, kv, @import("arch.zig").drafterOn(&config)), terms.points);
    try testing.expectEqual(2 * state_4k + terms.points, terms.at(4096));
    try testing.expectEqual(terms.per_token * 900_000 + terms.fixed + terms.copy_cap + terms.points, terms.at(900_000));
    // Where no checkpoint is written there is no copy.
    var ram = config;
    ram.prefix_cache_disk_declined = true;
    try testing.expectEqual(@as(u64, 0), ledgerTerms(&ram, kv, 8192).?.copy_cap);
}
