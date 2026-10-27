//! Grammar masks for the rows of one speculative verify chunk.
//!
//! A chunk `[t1, d1 .. dk]` is verified in one pass, so row r's mask is the one serial decode would
//! build after `t1, d1 .. dr`. `plan` walks the constraint (JSON grammar and reasoning protocol)
//! over the drafts on a scratch copy of its state, writes each row's mask and cuts the chunk at the
//! first draft its row's mask forbids. The live state is untouched; `advance` commits tokens to it.

const std = @import("std");
const json_grammar = @import("json_grammar.zig");
const json_schema = @import("json_schema.zig");
const token_mask = @import("token_mask.zig");
const rp = @import("reasoning_protocol.zig");
const log = @import("log.zig");

pub const View = struct {
    grammar: *json_grammar.Grammar,
    token_bytes: *const token_mask.TokenBytes,
    proto: ?*const rp.Protocol,
    pstate: *rp.State,
    eos_ids: []const u32,
};

pub const Advance = struct {
    /// The reasoning close completed inside this token; `suffix` is the payload it carried.
    crossed: bool = false,
    suffix: []const u8 = &.{},
};

pub const Error = error{ InvalidProtocolTransition, InvalidProtocolPayload, UnsupportedPhase } || std.mem.Allocator.Error;

/// Feeds one token through the protocol and grammar, as serial decode does after sampling it.
pub fn advance(v: View, token: u32) Error!Advance {
    const bytes = if (token < v.token_bytes.bytes.len) v.token_bytes.bytes[token] else null;
    switch (v.pstate.phase) {
        .reasoning => {
            const suffix = rp.observeProtocolToken(v.proto.?, v.pstate, token, bytes) orelse return .{};
            if (v.pstate.invalid) return error.InvalidProtocolTransition;
            for (suffix) |b| if (!try v.grammar.acceptByte(b)) return error.InvalidProtocolPayload;
            v.pstate.phase = .json_body;
            return .{ .crossed = true, .suffix = suffix };
        },
        .json_body => {
            if (v.grammar.dead) return .{};
            for (bytes orelse return .{}) |b| {
                if (!try v.grammar.acceptByte(b)) {
                    log.warn("[grammar] token {d} produced byte 0x{x} that was rejected — disabling further mask enforcement\n", .{ token, b });
                    v.grammar.dead = true;
                    break;
                }
            }
            return .{};
        },
        .choice, .header => return error.UnsupportedPhase,
    }
}

/// `open`: every token is legal and `out` was not written.
pub const Mask = enum { open, masked };

/// The tokens serial decode may sample next, written to `out` (`token_bytes.bytes.len` long) unless `open`.
pub fn maskFor(v: View, out: []bool) Error!Mask {
    switch (v.pstate.phase) {
        .reasoning => {
            const excluded = try rp.applyReasoningMask(v.proto.?, v.pstate, v.grammar, v.token_bytes, out);
            return if (excluded == 0) .open else .masked;
        },
        .json_body => {
            if (v.grammar.dead) return .open;
            var allowed = (try token_mask.buildMask(v.grammar, v.token_bytes, out)).allowed;
            if (v.grammar.isComplete()) for (v.eos_ids) |eos| {
                if (eos < out.len and !out[eos]) {
                    out[eos] = true;
                    allowed += 1;
                }
            };
            if (allowed == 0) {
                log.warn("[grammar] no token satisfies the schema at this position — disabling further mask enforcement\n", .{});
                v.grammar.dead = true;
                return .open;
            }
            return .masked;
        },
        .choice, .header => return error.UnsupportedPhase,
    }
}

pub const Planned = struct {
    /// Verify rows kept: `t1` and the drafts that may run.
    rows: usize,
    /// Some row's mask forbids tokens.
    masked: bool,
};

/// Row r of `masks` (`stride` bools, `stride >= token_bytes.bytes.len`) is the mask after `t1` and the first r
/// drafts, padded with false. `t1` was sampled under the live state's mask. A reasoning draft that serial decode
/// answers by forcing the boundary (EOS, pad) is cut, and so is every draft a mask forbids.
pub fn plan(v: View, t1: u32, drafts: []const u32, masks: []bool, stride: usize) Error!Planned {
    const n = v.token_bytes.bytes.len;
    std.debug.assert(stride >= n and masks.len >= (drafts.len + 1) * stride);
    const snap = try v.grammar.snapshot();
    defer v.grammar.discardSnapshot(snap);
    const saved = v.pstate.*;
    defer {
        v.grammar.restoreFrom(snap) catch unreachable; // the stack's capacity only grew
        v.pstate.* = saved;
    }
    _ = try advance(v, t1);
    var stop = false;
    var masked = false;
    var r: usize = 0;
    while (true) : (r += 1) {
        const row = masks[r * stride ..][0..stride];
        switch (try maskFor(v, row[0..n])) {
            .open => @memset(row[0..n], true),
            .masked => masked = true,
        }
        @memset(row[n..], false);
        if (stop or r == drafts.len) break;
        const d = drafts[r];
        if (d >= n or !row[d]) break;
        if (v.pstate.phase == .reasoning and (d == 0 or std.mem.indexOfScalar(u32, v.eos_ids, d) != null)) break;
        _ = try advance(v, d);
        stop = std.mem.indexOfScalar(u32, v.eos_ids, d) != null;
    }
    return .{ .rows = r + 1, .masked = masked };
}

const testing = std.testing;

const Fixture = struct {
    const eos = 1;
    const open_brace = 2;
    const quote = 4;
    const key_a = 5;
    const colon = 6;
    const yes = 7;
    const close_brace = 3;
    const prose = 10;
    const think_close = 11;
    const think_close_json = 12;
    const think_close_bad = 13;
    const words = [_]?[]const u8{ null, null, "{", "}", "\"", "a", ":", "true", "false", " ", "x", "</think>", "</think>{", "</think>x", "yz" };

    tb: token_mask.TokenBytes,
    schema: json_schema.Schema,
    grammar: json_grammar.Grammar,
    proto: rp.Protocol,
    pstate: rp.State,
    candidates: [3]u32,

    fn init(self: *Fixture, start: rp.Phase) !void {
        const a = testing.allocator;
        self.tb = try token_mask.TokenBytes.init(std.heap.ArenaAllocator.init(a), &words, eos);
        errdefer self.tb.deinit();
        const parsed = try std.json.parseFromSlice(std.json.Value, a,
            \\{"type":"object","properties":{"a":{"type":"boolean"}},"required":["a"],"additionalProperties":false}
        , .{});
        defer parsed.deinit();
        self.schema = try json_schema.parse(a, parsed.value);
        errdefer self.schema.deinit();
        self.grammar = try json_grammar.Grammar.init(a, &self.schema);
        self.candidates = .{ think_close, think_close_json, think_close_bad };
        self.proto = .{ .kind = .bare_think };
        try testing.expect(self.proto.setCloser(rp.BARE_THINK_CLOSER));
        self.proto.closer_span_candidates = &self.candidates;
        self.pstate = .{ .phase = start };
    }

    fn deinit(self: *Fixture) void {
        self.grammar.deinit();
        self.schema.deinit();
        self.tb.deinit();
    }

    fn view(self: *Fixture) View {
        return .{ .grammar = &self.grammar, .token_bytes = &self.tb, .proto = &self.proto, .pstate = &self.pstate, .eos_ids = &.{eos} };
    }
};

fn rowOf(masks: []const bool, r: usize) []const bool {
    return masks[r * Fixture.words.len ..][0..Fixture.words.len];
}

fn run(fx: *Fixture, t1: u32, drafts: []const u32, masks: []bool) !Planned {
    return plan(fx.view(), t1, drafts, masks, Fixture.words.len);
}

test "plan: each row takes the mask serial decode builds after the drafts before it, and the live state stays put" {
    var fx: Fixture = undefined;
    try fx.init(.json_body);
    defer fx.deinit();
    var masks: [4 * Fixture.words.len]bool = undefined;
    const depth = fx.grammar.stack.items.len;

    const planned = try run(&fx, Fixture.open_brace, &.{ Fixture.quote, Fixture.key_a, Fixture.prose }, &masks);
    try testing.expectEqual(@as(usize, 3), planned.rows);
    try testing.expect(planned.masked);
    try testing.expect(rowOf(&masks, 0)[Fixture.quote] and !rowOf(&masks, 0)[Fixture.prose] and !rowOf(&masks, 0)[Fixture.yes]);
    try testing.expect(rowOf(&masks, 1)[Fixture.key_a] and !rowOf(&masks, 1)[Fixture.quote]);
    try testing.expect(rowOf(&masks, 2)[Fixture.quote] and !rowOf(&masks, 2)[Fixture.key_a]);
    try testing.expectEqual(depth, fx.grammar.stack.items.len);
    try testing.expect((try maskFor(fx.view(), masks[0..Fixture.words.len])) == .masked);
    try testing.expect(masks[Fixture.open_brace] and !masks[Fixture.quote]);
}

test "plan: the first draft its row forbids is dropped with every draft after it" {
    var fx: Fixture = undefined;
    try fx.init(.json_body);
    defer fx.deinit();
    var masks: [4 * Fixture.words.len]bool = undefined;
    const planned = try run(&fx, Fixture.open_brace, &.{ Fixture.prose, Fixture.quote, Fixture.key_a }, &masks);
    try testing.expectEqual(@as(usize, 1), planned.rows);
}

test "plan: EOS is a draft only once the grammar is complete, and ends the chunk" {
    var fx: Fixture = undefined;
    try fx.init(.json_body);
    defer fx.deinit();
    var masks: [9 * Fixture.words.len]bool = undefined;
    try testing.expectEqual(@as(usize, 1), (try run(&fx, Fixture.open_brace, &.{Fixture.eos}, &masks)).rows);
    const whole = [_]u32{ Fixture.quote, Fixture.key_a, Fixture.quote, Fixture.colon, Fixture.yes, Fixture.close_brace, Fixture.eos, Fixture.prose };
    try testing.expectEqual(@as(usize, 8), (try run(&fx, Fixture.open_brace, &whole, &masks)).rows);
}

test "plan: a dead grammar leaves every row open" {
    var fx: Fixture = undefined;
    try fx.init(.json_body);
    defer fx.deinit();
    fx.grammar.dead = true;
    var masks: [3 * Fixture.words.len]bool = undefined;
    const planned = try run(&fx, Fixture.prose, &.{ Fixture.prose, Fixture.yes }, &masks);
    try testing.expectEqual(@as(usize, 3), planned.rows);
    try testing.expect(!planned.masked);
    for (masks) |ok| try testing.expect(ok);
}

test "plan: reasoning rows are open until a close crosses with a payload the schema rejects" {
    var fx: Fixture = undefined;
    try fx.init(.reasoning);
    defer fx.deinit();
    var masks: [4 * Fixture.words.len]bool = undefined;
    const planned = try run(&fx, Fixture.prose, &.{ Fixture.prose, Fixture.yes }, &masks);
    try testing.expectEqual(@as(usize, 3), planned.rows);
    try testing.expect(planned.masked);
    try testing.expect(rowOf(&masks, 0)[Fixture.prose] and rowOf(&masks, 0)[Fixture.think_close_json]);
    try testing.expect(!rowOf(&masks, 0)[Fixture.think_close_bad]);
    // A draft that would cross with a payload the schema rejects ends the chunk, as does EOS: serial decode forces the boundary.
    try testing.expectEqual(@as(usize, 2), (try run(&fx, Fixture.prose, &.{ Fixture.prose, Fixture.think_close_bad, Fixture.prose }, &masks)).rows);
    try testing.expectEqual(@as(usize, 2), (try run(&fx, Fixture.prose, &.{ Fixture.prose, Fixture.eos }, &masks)).rows);
    try testing.expectEqual(@as(usize, 2), (try run(&fx, Fixture.prose, &.{ Fixture.prose, 0 }, &masks)).rows);
    try testing.expectEqual(rp.Phase.reasoning, fx.pstate.phase);
}

test "plan: past the close the rows take the JSON mask" {
    var fx: Fixture = undefined;
    try fx.init(.reasoning);
    defer fx.deinit();
    var masks: [5 * Fixture.words.len]bool = undefined;
    const drafts = [_]u32{ Fixture.prose, Fixture.think_close_json, Fixture.quote, Fixture.prose };
    const through = try run(&fx, Fixture.prose, &drafts, &masks);
    try testing.expectEqual(@as(usize, 4), through.rows);
    try testing.expect(rowOf(&masks, 2)[Fixture.quote] and !rowOf(&masks, 2)[Fixture.prose]);
    // A pending token that is itself the close hands the JSON mask to the first row.
    const closes = try run(&fx, Fixture.think_close_json, &.{ Fixture.quote, Fixture.key_a }, &masks);
    try testing.expectEqual(@as(usize, 3), closes.rows);
    try testing.expect(rowOf(&masks, 0)[Fixture.quote] and !rowOf(&masks, 0)[Fixture.prose]);
}

test "advance: the close reports the payload it carried and the state moves into the body" {
    var fx: Fixture = undefined;
    try fx.init(.reasoning);
    defer fx.deinit();
    const v = fx.view();
    try testing.expect(!(try advance(v, Fixture.prose)).crossed);
    const crossed = try advance(v, Fixture.think_close_json);
    try testing.expect(crossed.crossed);
    try testing.expectEqualStrings("{", crossed.suffix);
    try testing.expectEqual(rp.Phase.json_body, fx.pstate.phase);
    _ = try advance(v, Fixture.quote);
    try testing.expect(!fx.grammar.dead);
}
