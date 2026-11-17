//! Overthinking-marker penalty (arXiv 2606.00206): while the reasoning span is open, every
//! marker token's logit drops by a fixed lambda before any sampler reads it.
const std = @import("std");
const mlx = @import("mlx.zig");
const bias_mod = @import("logit_bias.zig");

/// The paper's Appendix C markers as words, then four that Qwen3.8's own hesitant reasoning
/// leans on; `markerIds` maps them onto a tokenizer.
pub const WORDS = [_][]const u8{
    "perhaps",   "maybe",       "wait",        "actually",     "hold",       "hmm",       "alternatively",
    "however",   "instead",     "but",         "though",       "although",   "yet",       "rather",
    "unless",    "otherwise",   "nonetheless", "nevertheless", "regardless", "still",     "anyway",
    "or",        "either",      "whether",     "uncertain",    "unsure",     "possibly",  "might",
    "could",     "another",     "different",   "reconsider",   "rethink",    "backtrack", "retry",
    "recheck",   "revisit",     "doubt",       "confused",     "wrong",      "mistake",   "error",
    "incorrect", "alternative", "likely",      "probably",     "again",
};

/// Every single-token spelling of `words` (bare and space-led, lowercase and capitalised), sorted
/// and unique. A spelling the tokenizer splits is skipped: its first piece ("re", "back", "D")
/// also starts unrelated words.
pub fn markerIds(allocator: std.mem.Allocator, words: []const []const u8, tok: anytype) ![]u32 {
    var ids = std.ArrayList(u32).empty;
    errdefer ids.deinit(allocator);
    var buf: [64]u8 = undefined;
    for (words) |w| {
        if (w.len == 0 or w.len >= buf.len) continue;
        for ([_]usize{ 0, 1 }) |lead| {
            for ([_]bool{ false, true }) |cap| {
                buf[0] = ' ';
                @memcpy(buf[lead..][0..w.len], w);
                if (cap) buf[lead] = std.ascii.toUpper(buf[lead]);
                const enc = try tok.encode(allocator, buf[0 .. lead + w.len]);
                defer allocator.free(enc);
                if (enc.len == 1) try ids.append(allocator, enc[0]);
            }
        }
    }
    std.mem.sort(u32, ids.items, {}, std.sort.asc(u32));
    var n: usize = 0;
    for (ids.items) |id| {
        if (n > 0 and ids.items[n - 1] == id) continue;
        ids.items[n] = id;
        n += 1;
    }
    ids.shrinkRetainingCapacity(n);
    return ids.toOwnedSlice(allocator);
}

/// One request's penalty: lambda and its think markers, plus the span state after the committed
/// ids `cursor` has walked. lambda 0 = off.
pub const ThinkPenalty = struct {
    lambda: f32 = 0,
    opener_id: ?u32 = null,
    closer_id: u32 = 0,
    phase: Phase = .after,
    cursor: usize = 0,
    file_biases: []const bias_mod.Bias = &.{},
    biases: []const bias_mod.Bias = &.{},
    prepared: ?Prepared = null,

    pub fn prepare(self: *ThinkPenalty, a: std.mem.Allocator, vocab: usize, mask: ?mlx.mlx_array, s: mlx.mlx_stream) !void {
        if (self.hasBias()) self.prepared = try Prepared.init(a, vocab, mask, self.*, s);
    }

    pub fn deinitPrepared(self: *ThinkPenalty) void {
        if (self.prepared) |*p| p.deinit();
        self.prepared = null;
    }

    pub fn hasBias(self: ThinkPenalty) bool {
        if (self.prepared != null) return true;
        for (self.file_biases) |b| if (b.delta != 0) return true;
        for (self.biases) |b| if (b.delta != 0) return true;
        return false;
    }

    pub fn active(self: ThinkPenalty) bool {
        return self.lambda > 0 or self.hasBias();
    }

    /// The span never reopens: a closer ends it for the rest of the request.
    pub const Phase = enum { before, inside, after };

    pub fn armed(self: ThinkPenalty) bool {
        return self.lambda > 0 and self.phase != .after;
    }

    fn step(self: ThinkPenalty, phase: Phase, id: u32) Phase {
        if (id == self.closer_id) return .after;
        if (phase == .before and self.opener_id != null and id == self.opener_id.?) return .inside;
        return phase;
    }

    pub fn observe(self: *ThinkPenalty, ids: []const u32) void {
        while (self.cursor < ids.len) : (self.cursor += 1) self.phase = self.step(self.phase, ids[self.cursor]);
    }

    /// The phase after the observed ids, then `extra`.
    pub fn phaseAfter(self: ThinkPenalty, extra: []const u32) Phase {
        var p = self.phase;
        for (extra) |id| p = self.step(p, id);
        return p;
    }
};

/// `[vocab]` bool, true on every marker id below `vocab`. The caller owns it.
pub fn buildMask(allocator: std.mem.Allocator, ids: []const u32, vocab: usize) !mlx.mlx_array {
    const buf = try allocator.alloc(bool, vocab);
    defer allocator.free(buf);
    @memset(buf, false);
    for (ids) |id| {
        if (id < vocab) buf[id] = true;
    }
    const shape = [_]c_int{@intCast(vocab)};
    return try mlx.newData(buf.ptr, &shape, .bool_);
}

/// `out = where(cond, logits - lambda, logits)`, lambda in the logits' own dtype so a
/// shifted row keeps the dtype an unshifted one has.
fn shiftWhere(out: *mlx.mlx_array, logits: mlx.mlx_array, cond: mlx.mlx_array, lambda: f32, s: mlx.mlx_stream) !void {
    const raw = try mlx.newFloat(lambda);
    defer _ = mlx.mlx_array_free(raw);
    var lam = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lam);
    try mlx.check(mlx.mlx_astype(&lam, raw, mlx.mlx_array_dtype(logits), s));
    var lowered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lowered);
    try mlx.check(mlx.mlx_subtract(&lowered, logits, lam, s));
    try mlx.check(mlx.mlx_where(out, cond, lowered, logits, s));
}

/// Every position of `logits` follows an open span: the markers drop by `lambda`.
pub fn shift(out: *mlx.mlx_array, logits: mlx.mlx_array, mask: mlx.mlx_array, lambda: f32, s: mlx.mlx_stream) !void {
    return shiftWhere(out, logits, mask, lambda, s);
}

fn insideRows(ids: mlx.mlx_array, p: ThinkPenalty, s: mlx.mlx_stream) !mlx.mlx_array {
    if (p.phase == .after) return try mlx.newBool(false);
    const open = try seenThrough(ids, p.closer_id, s);
    defer _ = mlx.mlx_array_free(open);
    const zero = try mlx.newInt(0);
    defer _ = mlx.mlx_array_free(zero);
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_equal(&gate, open, zero, s));
    if (p.phase == .before) {
        const opener = p.opener_id orelse return try mlx.newBool(false);
        const opened = try seenThrough(ids, opener, s);
        defer _ = mlx.mlx_array_free(opened);
        var after_open = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(after_open);
        try mlx.check(mlx.mlx_greater(&after_open, opened, zero, s));
        var both = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_logical_and(&both, gate, after_open, s));
        _ = mlx.mlx_array_free(gate);
        gate = both;
    }
    var rows = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_expand_dims(&rows, gate, 2, s));
    return rows;
}

/// Row j of `logits` `[1, L, V]` follows `ids[0..j]` (`[1, L]` int32, inclusive, lazy is fine)
/// after the span state `p`; each row shifts while the span is open there.
pub fn shiftRows(out: *mlx.mlx_array, logits: mlx.mlx_array, ids: mlx.mlx_array, mask: mlx.mlx_array, p: ThinkPenalty, s: mlx.mlx_stream) !void {
    if (p.phase == .after) return mlx.check(mlx.mlx_array_set(out, logits));
    const rows = try insideRows(ids, p, s);
    defer _ = mlx.mlx_array_free(rows);
    var cond = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cond);
    try mlx.check(mlx.mlx_logical_and(&cond, rows, mask, s));
    try shiftWhere(out, logits, cond, p.lambda, s);
}

/// How many times `id` occurs in each prefix of `ids` (`[1, L]` int32).
fn seenThrough(ids: mlx.mlx_array, id: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const target = try mlx.newInt(@intCast(id));
    defer _ = mlx.mlx_array_free(target);
    var hit = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hit);
    try mlx.check(mlx.mlx_equal(&hit, ids, target, s));
    var hit_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hit_i);
    try mlx.check(mlx.mlx_astype(&hit_i, hit, .int32, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_cumsum(&out, hit_i, 1, false, true, s));
    return out;
}

const testing = std.testing;

const FakeTokenizer = struct {
    pieces: []const []const u8,

    pub fn encode(self: FakeTokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        for (self.pieces, 0..) |p, i| {
            if (std.mem.eql(u8, p, text)) {
                const out = try allocator.alloc(u32, 1);
                out[0] = @intCast(i);
                return out;
            }
        }
        const out = try allocator.alloc(u32, 2);
        @memset(out, 999);
        return out;
    }
};

test "think penalty: markerIds keeps every single-token spelling and skips the split ones" {
    const tok = FakeTokenizer{ .pieces = &.{ "Wait", " Wait", "wait", " wait", " Hmm", "Hmm", " hmm", " recheck" } };
    const ids = try markerIds(testing.allocator, &.{ "wait", "hmm", "recheck", "wait" }, tok);
    defer testing.allocator.free(ids);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, ids);
}

test "think penalty: the span opens at the opener, closes at the closer and never reopens" {
    const open: u32 = 10;
    const close: u32 = 11;
    var inside = ThinkPenalty{ .lambda = 1, .opener_id = open, .closer_id = close, .phase = .inside };
    try testing.expectEqual(ThinkPenalty.Phase.inside, inside.phaseAfter(&.{ 1, 2, open }));
    try testing.expectEqual(ThinkPenalty.Phase.after, inside.phaseAfter(&.{ 1, close }));
    try testing.expectEqual(ThinkPenalty.Phase.after, inside.phaseAfter(&.{ close, open, 3 }));
    inside.observe(&.{ 1, 2 });
    try testing.expectEqual(ThinkPenalty.Phase.inside, inside.phase);
    inside.observe(&.{ 1, 2, close, 4 });
    try testing.expectEqual(ThinkPenalty.Phase.after, inside.phase);
    try testing.expectEqual(@as(usize, 4), inside.cursor);

    const before = ThinkPenalty{ .lambda = 1, .opener_id = open, .closer_id = close, .phase = .before };
    try testing.expectEqual(ThinkPenalty.Phase.before, before.phaseAfter(&.{ 1, 2 }));
    try testing.expectEqual(ThinkPenalty.Phase.inside, before.phaseAfter(&.{ open, 2 }));
    try testing.expectEqual(ThinkPenalty.Phase.after, before.phaseAfter(&.{ close, open }));
    try testing.expect(!(ThinkPenalty{}).armed());
    try testing.expect(before.armed());
}

fn readF32(allocator: std.mem.Allocator, arr: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_astype(&wide, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(wide));
    const n: usize = @intCast(mlx.mlx_array_size(wide));
    const data = mlx.mlx_array_data_float32(wide) orelse return error.MlxArrayDataNull;
    return allocator.dupe(f32, data[0..n]);
}

fn testBlock(rows: usize, v: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const host = try testing.allocator.alloc(f32, rows * v);
    defer testing.allocator.free(host);
    for (host, 0..) |*x, i| x.* = @as(f32, @floatFromInt((i * 37) % 23)) * 0.375 - 3.0;
    const shape = [_]c_int{ 1, @intCast(rows), @intCast(v) };
    const f = try mlx.newData(host.ptr, &shape, .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

test "think penalty: shift drops only the marker logits, by lambda" {
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 64;
    const block = try testBlock(1, v, s);
    defer _ = mlx.mlx_array_free(block);
    const mask = try buildMask(a, &.{ 3, 40, 70 }, v);
    defer _ = mlx.mlx_array_free(mask);
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try shift(&out, block, mask, 1.5, s);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(out));
    const raw = try readF32(a, block, s);
    defer a.free(raw);
    const got = try readF32(a, out, s);
    defer a.free(got);
    for (raw, got, 0..) |r, g, i| {
        const want = if (i == 3 or i == 40) r - 1.5 else r;
        try testing.expectEqual(want, g);
    }
}

test "think penalty: every verify row shifts exactly as its serial tick would, across the span's edges" {
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 48;
    const open: u32 = 30;
    const close: u32 = 31;
    const mask = try buildMask(a, &.{ 2, 5, close, 47 }, v);
    defer _ = mlx.mlx_array_free(mask);
    const Case = struct { phase: ThinkPenalty.Phase, ids: []const i32 };
    const cases = [_]Case{
        .{ .phase = .inside, .ids = &.{ 7, 8, close, 9, open } },
        .{ .phase = .inside, .ids = &.{close} },
        .{ .phase = .inside, .ids = &.{ 4, 4, 4 } },
        .{ .phase = .before, .ids = &.{ 7, open, 8, close, 9 } },
        .{ .phase = .before, .ids = &.{ close, open, 1 } },
    };
    for (cases) |c| {
        const p = ThinkPenalty{ .lambda = 2.5, .opener_id = open, .closer_id = close, .phase = c.phase };
        const rows = c.ids.len;
        const block = try testBlock(rows, v, s);
        defer _ = mlx.mlx_array_free(block);
        const id_shape = [_]c_int{ 1, @intCast(rows) };
        const ids = mlx.mlx_array_new_data(c.ids.ptr, &id_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(ids);
        var shifted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(shifted);
        try shiftRows(&shifted, block, ids, mask, p, s);
        const got = try readF32(a, shifted, s);
        defer a.free(got);
        const raw = try readF32(a, block, s);
        defer a.free(raw);
        var host_ids: [8]u32 = undefined;
        for (c.ids, 0..) |id, j| host_ids[j] = @intCast(id);
        for (0..rows) |j| {
            const row_raw = raw[j * v ..][0..v];
            const row_got = got[j * v ..][0..v];
            if (p.phaseAfter(host_ids[0 .. j + 1]) != .inside) {
                try testing.expectEqualSlices(f32, row_raw, row_got);
                continue;
            }
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            try mlx.check(mlx.mlx_slice(&row, block, &.{ 0, @intCast(j), 0 }, 3, &.{ 1, @as(c_int, @intCast(j)) + 1, @intCast(v) }, 3, &.{ 1, 1, 1 }, 3, s));
            var serial = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(serial);
            try shift(&serial, row, mask, p.lambda, s);
            const want = try readF32(a, serial, s);
            defer a.free(want);
            try testing.expectEqualSlices(f32, want, row_got);
        }
    }
}

test {
    _ = @import("logit_bias.zig");
}

pub const Prepared = struct {
    inside: mlx.mlx_array,
    outside: mlx.mlx_array,

    pub fn init(a: std.mem.Allocator, vocab: usize, mask: ?mlx.mlx_array, p: ThinkPenalty, s: mlx.mlx_stream) !Prepared {
        const host = try a.alloc(f32, vocab * 2);
        defer a.free(host);
        bias_mod.fillDeltas(host[0..vocab], p.file_biases, true);
        bias_mod.addDeltas(host[0..vocab], p.biases, true);
        bias_mod.fillDeltas(host[vocab..], p.file_biases, false);
        bias_mod.addDeltas(host[vocab..], p.biases, false);
        const vshape = [_]c_int{@intCast(vocab)};
        var inside = try mlx.newData(host.ptr, &vshape, .float32);
        errdefer _ = mlx.mlx_array_free(inside);
        const outside = try mlx.newData(host[vocab..].ptr, &vshape, .float32);
        errdefer _ = mlx.mlx_array_free(outside);
        if (mask) |m| if (p.lambda > 0) {
            var combined = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(combined);
            try shift(&combined, inside, m, p.lambda, s);
            _ = mlx.mlx_array_free(inside);
            inside = combined;
        };
        try mlx.check(mlx.mlx_array_eval(inside));
        try mlx.check(mlx.mlx_array_eval(outside));
        return .{ .inside = inside, .outside = outside };
    }

    pub fn deinit(self: *Prepared) void {
        _ = mlx.mlx_array_free(self.inside);
        _ = mlx.mlx_array_free(self.outside);
    }
};

pub fn shiftScoped(out: *mlx.mlx_array, logits: mlx.mlx_array, mask: ?mlx.mlx_array, p: ThinkPenalty, ids: ?mlx.mlx_array, s: mlx.mlx_stream) !void {
    if (!p.hasBias()) {
        if (mask != null and p.lambda > 0) {
            if (ids) |rows| return shiftRows(out, logits, rows, mask.?, p, s);
            if (p.phase == .inside) return shift(out, logits, mask.?, p.lambda, s);
        }
        return mlx.check(mlx.mlx_array_set(out, logits));
    }
    const prepared = p.prepared orelse return error.LogitBiasNotPrepared;
    const inside = prepared.inside;
    const outside = prepared.outside;
    var deltas = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(deltas);
    if (ids) |rows| {
        const gate = try insideRows(rows, p, s);
        defer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_where(&deltas, gate, inside, outside, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&deltas, if (p.phase == .inside) inside else outside));
    }
    return addDeltas(out, logits, deltas, s);
}

/// Row j of `logits` `[1, L, V]` at the span state the caller decided (`inside[j]`), with
/// `shiftScoped`'s arithmetic: a draft tree's rows have no prefix order for `ids` to gate by.
pub fn shiftDecided(out: *mlx.mlx_array, logits: mlx.mlx_array, mask: ?mlx.mlx_array, p: ThinkPenalty, inside: []const bool, s: mlx.mlx_stream) !void {
    if (!p.hasBias() and (mask == null or p.lambda <= 0)) return mlx.check(mlx.mlx_array_set(out, logits));
    const gate = try mlx.newData(inside.ptr, &[_]c_int{ 1, @intCast(inside.len), 1 }, .bool_);
    defer _ = mlx.mlx_array_free(gate);
    if (!p.hasBias()) {
        var cond = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cond);
        try mlx.check(mlx.mlx_logical_and(&cond, gate, mask.?, s));
        return shiftWhere(out, logits, cond, p.lambda, s);
    }
    const prepared = p.prepared orelse return error.LogitBiasNotPrepared;
    var deltas = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(deltas);
    try mlx.check(mlx.mlx_where(&deltas, gate, prepared.inside, prepared.outside, s));
    return addDeltas(out, logits, deltas, s);
}

/// `logits + deltas` in the logits' dtype; lanes whose delta is zero keep their bytes.
fn addDeltas(out: *mlx.mlx_array, logits: mlx.mlx_array, deltas: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var typed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(typed);
    try mlx.check(mlx.mlx_astype(&typed, deltas, mlx.mlx_array_dtype(logits), s));
    var shifted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shifted);
    try mlx.check(mlx.mlx_add(&shifted, logits, typed, s));
    const zero = try mlx.newFloat(0);
    defer _ = mlx.mlx_array_free(zero);
    var changed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(changed);
    try mlx.check(mlx.mlx_equal(&changed, typed, zero, s));
    try mlx.check(mlx.mlx_where(out, changed, logits, shifted, s));
}

test "logit bias GPU: scoped verify rows equal serial across boundaries and add the preset" {
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const v = 48;
    const open: u32 = 30;
    const close: u32 = 31;
    const mask = try buildMask(a, &.{ 2, 5, 47 }, v);
    defer _ = mlx.mlx_array_free(mask);
    const file = [_]bias_mod.Bias{
        .{ .id = 2, .delta = 1, .scope = .reasoning },
        .{ .id = 5, .delta = 3, .scope = .answer },
        .{ .id = 47, .delta = -1, .scope = .all },
    };
    const request_bias = [_]bias_mod.Bias{.{ .id = 2, .delta = 0.5 }};
    const host_ids = [_]u32{ 7, open, 8, close, 9, open };
    const signed_ids = [_]i32{ 7, open, 8, close, 9, open };
    var p = ThinkPenalty{ .lambda = 2.5, .opener_id = open, .closer_id = close, .phase = .before, .file_biases = &file, .biases = &request_bias };
    try p.prepare(a, v, mask, s);
    defer p.deinitPrepared();
    const block = try testBlock(host_ids.len, v, s);
    defer _ = mlx.mlx_array_free(block);
    const ids = mlx.mlx_array_new_data(&signed_ids, &[_]c_int{ 1, host_ids.len }, 2, .int32);
    defer _ = mlx.mlx_array_free(ids);
    var shifted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shifted);
    try shiftScoped(&shifted, block, mask, p, ids, s);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(shifted));
    const got = try readF32(a, shifted, s);
    defer a.free(got);
    const raw = try readF32(a, block, s);
    defer a.free(raw);
    for (0..host_ids.len) |j| {
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, block, &.{ 0, @intCast(j), 0 }, 3, &.{ 1, @as(c_int, @intCast(j)) + 1, v }, 3, &.{ 1, 1, 1 }, 3, s));
        var state = p;
        state.phase = p.phaseAfter(host_ids[0 .. j + 1]);
        const inside = state.phase == .inside;
        for (0..v) |id| {
            const delta: f32 = switch (id) {
                2 => if (inside) -1 else 0.5,
                5 => if (inside) -2.5 else 3,
                47 => if (inside) -3.5 else -1,
                else => 0,
            };
            try testing.expectEqual(raw[j * v + id] + delta, got[j * v + id]);
        }
        var serial = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(serial);
        try shiftScoped(&serial, row, mask, state, null, s);
        const want = try readF32(a, serial, s);
        defer a.free(want);
        try testing.expectEqualSlices(f32, want, got[j * v ..][0..v]);
    }
}

test "logit bias GPU: rows at caller-decided span states equal their serial rows" {
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const v = 48;
    const mask = try buildMask(a, &.{ 2, 5, 47 }, v);
    defer _ = mlx.mlx_array_free(mask);
    const file = [_]bias_mod.Bias{
        .{ .id = 2, .delta = 1, .scope = .reasoning },
        .{ .id = 5, .delta = 0.375, .scope = .answer },
    };
    const request_bias = [_]bias_mod.Bias{.{ .id = 9, .delta = -100 }};
    // Tree siblings: no prefix order for `ids` to gate by.
    const inside = [_]bool{ true, false, true, false };
    for ([_]bool{ false, true }) |with_bias| {
        var p = ThinkPenalty{ .lambda = 2.5, .opener_id = 30, .closer_id = 31, .phase = .inside };
        if (with_bias) {
            p.file_biases = &file;
            p.biases = &request_bias;
        }
        try p.prepare(a, v, mask, s);
        defer p.deinitPrepared();
        const block = try testBlock(inside.len, v, s);
        defer _ = mlx.mlx_array_free(block);
        var shifted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(shifted);
        try shiftDecided(&shifted, block, mask, p, &inside, s);
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(shifted));
        const got = try readF32(a, shifted, s);
        defer a.free(got);
        for (inside, 0..) |in_span, j| {
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            try mlx.check(mlx.mlx_slice(&row, block, &.{ 0, @intCast(j), 0 }, 3, &.{ 1, @as(c_int, @intCast(j)) + 1, v }, 3, &.{ 1, 1, 1 }, 3, s));
            var state = p;
            state.phase = if (in_span) .inside else .after;
            var serial = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(serial);
            try shiftScoped(&serial, row, mask, state, null, s);
            const want = try readF32(a, serial, s);
            defer a.free(want);
            try testing.expectEqualSlices(f32, want, got[j * v ..][0..v]);
        }
    }
}

test "logit bias GPU: preset and disabled paths preserve the original bytes" {
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const block = try testBlock(1, 48, s);
    defer _ = mlx.mlx_array_free(block);
    const mask = try buildMask(a, &.{ 2, 5, 47 }, 48);
    defer _ = mlx.mlx_array_free(mask);
    for ([_]f32{ 0, 1, 2.5 }) |lambda| {
        var original = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(original);
        if (lambda == 0) {
            try mlx.check(mlx.mlx_array_set(&original, block));
        } else try shift(&original, block, mask, lambda, s);
        var scoped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scoped);
        try shiftScoped(&scoped, block, mask, .{ .lambda = lambda, .phase = .inside }, null, s);
        const want = try readF32(a, original, s);
        defer a.free(want);
        const got = try readF32(a, scoped, s);
        defer a.free(got);
        try testing.expectEqualSlices(f32, want, got);
    }
}

pub fn needsSpan(lambda: f32, entries: []const bias_mod.Bias) bool {
    if (lambda > 0) return true;
    for (entries) |entry| if (entry.delta != 0 and entry.scope != .all) return true;
    return false;
}

test "logit bias CPU: disabled and all-scope requests do not need span tokenization" {
    try testing.expect(!needsSpan(0, &.{}));
    try testing.expect(needsSpan(1, &.{}));
    try testing.expect(!needsSpan(0, &.{.{ .id = 1, .delta = 2 }}));
    try testing.expect(!needsSpan(0, &.{.{ .id = 1, .delta = 0, .scope = .reasoning }}));
    try testing.expect(needsSpan(0, &.{.{ .id = 1, .delta = 2, .scope = .reasoning }}));
    try testing.expect(needsSpan(0, &.{.{ .id = 1, .delta = -2, .scope = .answer }}));
}
