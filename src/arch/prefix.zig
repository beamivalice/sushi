//! Prefix-cache dispatch over the arch seam. The hot cache, the SSD tier and the SSM entries keep one code path;
//! what is an arch's own (its restore target, its per-entry rows, its disk sections) is declared by that arch's
//! `prefix` module and reached here, `inline else` over the enabled archs. Inference thread only.
//!
//! An arch's `prefix` module may declare, each optional:
//! - `Rows`: per-entry rows that scale with the prompt and ride beside the KV (deinit, bytes, rowBytes, len,
//!   trimmedCopy, share, releaseHandles; module fns `keepThrough`, `persistRows`);
//! - `State` (a pointer to the live request state), with `reset`, and the hooks `handover`, `rollbackLive`,
//!   `restoreDisk` and `restoreEntry`: a lookup restores into it instead of into SSM entries;
//! - `CacheState`: what the hot cache keeps per cache for this arch (default-initialised, `deinit(allocator)`);
//! - `diskLayoutKey(allocator, config, model_dir, xfm)`: the SSD namespace key when it names more than the config's tag;
//! - `checkpointsRequired(config)`, `adoptSpecHead`, `aux` (SSM entry accelerators), `disk` (the QSA-history section).
const std = @import("std");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const registry = @import("registry.zig");
const pc = @import("../prefix_cache.zig");
const kv_disk_cache = @import("../kv_disk_cache.zig");
const ssm_cache = @import("../ssm_cache.zig");
const trunk = @import("../trunk.zig");

const Tag = registry.Tag;
const tags = std.enums.values(Tag);

fn P(comptime t: Tag) type {
    return registry.Mod(t).prefix;
}

fn Payload(comptime t: Tag, comptime name: []const u8) type {
    return if (@hasDecl(P(t), name)) @field(P(t), name) else void;
}

fn ByTag(comptime name: []const u8) type {
    var names: [tags.len][]const u8 = undefined;
    var types: [tags.len]type = undefined;
    for (tags, &names, &types) |t, *n, *ty| {
        n.* = @tagName(t);
        ty.* = Payload(t, name);
    }
    return @Union(.auto, Tag, &names, &types, &@splat(.{}));
}

fn isVoid(comptime T: type) bool {
    return T == void;
}

fn defaultOf(comptime T: type) *const anyopaque {
    const v: T = if (T == void) {} else .{};
    return &v;
}

fn firstWith(comptime name: []const u8) ?type {
    for (tags) |t| if (@hasDecl(P(t), name)) return @field(P(t), name);
    return null;
}

/// Whether the build carries this arch. An arch's own prefix code is compiled in every build that still imports its
/// folder, so it names its tag through the helpers below, which are unreachable where the arch is left out.
pub fn has(comptime name: []const u8) bool {
    return @hasField(Tag, name);
}

/// The restore target of an arch whose state is not SSM entries: one prong per enabled arch, void where it has none.
pub const State = ByTag("State");

const RowsUnion = ByTag("Rows");

/// A lookup target holding the arch's state.
pub fn recurrent(comptime name: []const u8, state: anytype) pc.Recurrent {
    if (comptime !has(name)) unreachable;
    return .{ .arch = @unionInit(State, name, state) };
}

/// An entry's arch rows, whichever arch owns them.
pub const Rows = struct {
    v: RowsUnion,

    pub fn ofName(comptime name: []const u8, rows: anytype) Rows {
        if (comptime !has(name)) unreachable;
        return .{ .v = @unionInit(RowsUnion, name, rows) };
    }

    /// `as` by name, for the code of an arch the build may leave out.
    pub fn asName(self: anytype, comptime name: []const u8) @TypeOf(&@field(self.v, name)) {
        return &@field(self.v, name);
    }

    pub fn deinit(self: *Rows) void {
        switch (self.v) {
            inline else => |*r| if (comptime !isVoid(@TypeOf(r.*))) r.deinit(),
        }
    }

    pub fn len(self: *const Rows) usize {
        switch (self.v) {
            inline else => |r| if (comptime isVoid(@TypeOf(r))) unreachable else return r.len,
        }
    }

    pub fn bytes(self: *const Rows) u64 {
        switch (self.v) {
            inline else => |r| if (comptime isVoid(@TypeOf(r))) unreachable else return r.bytes(),
        }
    }

    pub fn rowBytes(self: *const Rows) u64 {
        switch (self.v) {
            inline else => |r| if (comptime isVoid(@TypeOf(r))) unreachable else return r.rowBytes(),
        }
    }

    pub fn releaseHandles(self: *Rows) void {
        switch (self.v) {
            inline else => |*r| if (comptime !isVoid(@TypeOf(r.*))) r.releaseHandles(),
        }
    }

    pub fn share(self: *const Rows) !Rows {
        switch (self.v) {
            inline else => |r, t| if (comptime isVoid(@TypeOf(r))) unreachable else return .{ .v = @unionInit(RowsUnion, @tagName(t), try r.share()) },
        }
    }

    pub fn trimmedCopy(self: *const Rows, n: usize, s: mlx.mlx_stream) !Rows {
        switch (self.v) {
            inline else => |r, t| if (comptime isVoid(@TypeOf(r))) unreachable else return .{ .v = @unionInit(RowsUnion, @tagName(t), try r.trimmedCopy(n, s)) },
        }
    }

    /// The checkpoints these rows can restore from: `cps` without those past the rows, the rest freed.
    pub fn keepCheckpoints(self: *const Rows, allocator: std.mem.Allocator, cps: []trunk.SSMCheckpoint) []trunk.SSMCheckpoint {
        switch (self.v) {
            inline else => |r, t| if (comptime isVoid(@TypeOf(r))) unreachable else return P(t).keepThrough(allocator, cps, r.len),
        }
    }

    /// The entry as the SSD tier stores it: the arch's own layout for these rows.
    pub fn persist(
        self: *const Rows,
        d: *kv_disk_cache.DiskTier,
        tokens: []const u32,
        has_tools: bool,
        media: pc.MediaStamp,
        cps: ?[]trunk.SSMCheckpoint,
        dflash: ?kv_disk_cache.SpecCommit,
        whole: bool,
        s: mlx.mlx_stream,
    ) !kv_disk_cache.PersistOutcome {
        switch (self.v) {
            inline else => |*r, t| if (comptime isVoid(@TypeOf(r.*))) unreachable else return P(t).persistRows(d, r, tokens, has_tools, media, cps, dflash, whole, s),
        }
    }
};

/// What the hot cache keeps per cache for each arch: a field per enabled arch, void where it keeps nothing.
pub const CacheState = blk: {
    var names: [tags.len][]const u8 = undefined;
    var types: [tags.len]type = undefined;
    var attrs: [tags.len]std.lang.Type.Struct.FieldAttributes = undefined;
    for (tags, &names, &types, &attrs) |t, *n, *ty, *a| {
        n.* = @tagName(t);
        ty.* = Payload(t, "CacheState");
        a.* = .{ .default_value_ptr = defaultOf(ty.*) };
    }
    break :blk @Struct(.auto, null, &names, &types, &attrs);
};

pub fn deinitCacheState(cs: *CacheState, allocator: std.mem.Allocator) void {
    inline for (tags) |t| {
        if (comptime !isVoid(Payload(t, "CacheState"))) @field(cs, @tagName(t)).deinit(allocator);
    }
}

pub fn reset(s: State) void {
    switch (s) {
        inline else => |p, t| if (comptime isVoid(@TypeOf(p))) unreachable else P(t).reset(p),
    }
}

/// The request's state is about to be replaced: what the arch keeps of the last one reaches SSD first.
pub fn handover(hc: *pc.HotPrefixCache, s: State, st: mlx.mlx_stream) void {
    switch (s) {
        inline else => |p, t| if (comptime isVoid(@TypeOf(p))) unreachable else if (comptime @hasDecl(P(t), "handover")) P(t).handover(hc, p, st),
    }
}

/// The live state rolled back in place instead of a stored restore; null when nothing live serves the prompt.
pub fn rollbackLive(
    hc: *pc.HotPrefixCache,
    s: State,
    target_cache: *trunk.KVCache,
    target_moe_seq_offset: *usize,
    st: mlx.mlx_stream,
    prompt_ids: []const u32,
    has_tools: bool,
    media_limits: kv_disk_cache.MediaLimits,
    ram_match: ?pc.RamMatch,
) ?pc.LookupResult {
    switch (s) {
        inline else => |p, t| {
            if (comptime isVoid(@TypeOf(p))) unreachable;
            if (comptime !@hasDecl(P(t), "rollbackLive")) return null;
            return P(t).rollbackLive(hc, p, target_cache, target_moe_seq_offset, st, prompt_ids, has_tools, media_limits, ram_match);
        },
    }
}

/// The SSD arm for the arch's restore target. Null falls back to the cold path: an arch with a state never
/// restores through the dense chunks.
pub fn restoreDisk(
    hc: *pc.HotPrefixCache,
    d: *kv_disk_cache.DiskTier,
    s: State,
    target_cache: *trunk.KVCache,
    ram_eff: usize,
    prompt_ids: []const u32,
    has_tools: bool,
    media_limits: kv_disk_cache.MediaLimits,
    dflash_target: ?pc.DflashTarget,
    st: mlx.mlx_stream,
) ?pc.LookupResult {
    switch (s) {
        inline else => |p, t| {
            if (comptime isVoid(@TypeOf(p))) unreachable;
            return P(t).restoreDisk(hc, d, p, target_cache, ram_eff, prompt_ids, has_tools, media_limits, dflash_target, st);
        },
    }
}

/// Installs the arch's state at the highest checkpoint of `e` at or below `restore_cap`; returns the position it
/// stands at, 0 when nothing could be installed.
pub fn restoreEntry(hc: *pc.HotPrefixCache, s: State, e: *pc.Entry, restore_cap: usize, prompt_ids: []const u32) !usize {
    switch (s) {
        inline else => |p, t| {
            if (comptime isVoid(@TypeOf(p))) unreachable;
            return P(t).restoreEntry(hc, p, e, restore_cap, prompt_ids);
        },
    }
}

/// Whether a config of this arch restores only from checkpoints, so the cache is off without them.
pub fn checkpointsRequired(config: *const model.ModelConfig) bool {
    const tag = registry.ofModelType(config.model_type) orelse return false;
    switch (tag) {
        inline else => |t| return if (comptime @hasDecl(P(t), "checkpointsRequired")) P(t).checkpointsRequired(config) else false,
    }
}

/// The stored-state layout the SSD namespace is keyed by, when the loaded arch names more than the config's own tag.
pub fn diskLayoutKey(allocator: std.mem.Allocator, config: *const model.ModelConfig, model_dir: []const u8, xfm: *trunk.Transformer) !?[]u8 {
    switch (registry.handleOf(xfm)) {
        inline else => |_, t| return if (comptime @hasDecl(P(t), "diskLayoutKey")) P(t).diskLayoutKey(allocator, config, model_dir, xfm) else null,
    }
}

/// A committed MTP head's snapshot adopted into the loaded model's own head, as `plan` decided. Null leaves the head
/// blind: an arch with no in-checkpoint head declines every snapshot.
pub fn adoptSpecHead(xfm: *trunk.Transformer, plan: pc.SpecAdopt, snap: *const pc.DflashSnap, base_pos: *usize) ?usize {
    switch (registry.handleOf(xfm)) {
        inline else => |m, t| return if (comptime @hasDecl(P(t), "adoptSpecHead")) P(t).adoptSpecHead(m, plan, snap, base_pos) else null,
    }
}

/// SSM entry accelerators (the QSA history buffers, the GDN replay) belong to the arch that carries that state.
pub const aux = firstWith("aux") orelse struct {
    // The qwen4 sources still compile through transformer.zig in a build that leaves the arch out, and they
    // read these two; S4-qwen4 removes those imports and with them these lines.
    pub const ple_prev_len = @import("../qwen4/qwen4_exp.zig").MAX_NGRAM_SIZE;
    pub const Replay = @import("../qwen4/gdn_decode.zig").Replay;
    pub const replayState = @import("../qwen4/gdn_decode.zig").replayState;
    pub fn noteHistorySeed() void {}
    pub fn noteCapBufAlloc() void {}
    pub fn qsaHistoryRows(_: *const ssm_cache.SSMCacheEntry) c_int {
        return 0;
    }
    pub fn qsaRawKeyBytes(_: *const ssm_cache.SSMCacheEntry) u64 {
        return 0;
    }
    pub fn ssmAuxIsQsaHistory(_: *const ssm_cache.SSMCacheEntry) bool {
        return false;
    }
    pub fn ssmFreeQsaState(_: *ssm_cache.SSMCacheEntry) void {}
    pub fn qsaKeyAppendPlan(rows: usize, cap: usize, add: usize, _: usize) struct { new_cap: usize, write_at: usize, new_rows: usize } {
        return .{ .new_cap = @max(cap, rows + add), .write_at = rows, .new_rows = rows + add };
    }
    pub fn qsaLeftoverAt(_: mlx.mlx_array, _: c_int, _: c_int, _: c_int, _: mlx.mlx_stream) !mlx.mlx_array {
        return .{ .ctx = null };
    }
    pub fn qsaResliceToKeep(_: *ssm_cache.SSMCacheEntry, _: c_int, _: mlx.mlx_stream) !bool {
        return false;
    }
};

/// The QSA-history section of an SSD record: the same arch's, with the sidecar file it writes.
pub const disk = firstWith("disk") orelse struct {
    const DiskTier = kv_disk_cache.DiskTier;
    pub fn qsaHistoryRowsOf(_: *const trunk.SSMCheckpoint) c_int {
        return 0;
    }
    pub fn newestQsaCheckpoint(_: []const trunk.SSMCheckpoint) ?*const trunk.SSMCheckpoint {
        return null;
    }
    pub fn writeQsaHistoryFile(_: *DiskTier, _: []const u8, _: *const trunk.SSMCheckpoint, _: mlx.mlx_stream) !u64 {
        return 0;
    }
    pub fn loadQsaHistoryFile(_: *DiskTier, _: u64, _: usize) !trunk.SSMCheckpoint {
        return error.DiskCacheNoCheckpoint;
    }
    pub fn persistQsaHistory(
        _: *DiskTier,
        _: []const u8,
        _: ?[]const trunk.SSMCheckpoint,
        inherited: bool,
        inherited_rows: u32,
        held_bytes: u64,
        _: u32,
        _: mlx.mlx_stream,
    ) !DiskTier.QsaHistoryResult {
        return .{ .inherited = inherited, .rows = inherited_rows, .bytes = if (inherited) 0 else held_bytes };
    }
    pub fn linkInheritedQsa(_: *DiskTier, _: DiskTier.ChunkDonor, _: u64) bool {
        return false;
    }
};

test "the dispatch unions have one prong per enabled arch, in registry order" {
    inline for (.{ State, RowsUnion }) |U| {
        const names = @typeInfo(U).@"union".field_names;
        try std.testing.expectEqual(tags.len, names.len);
        inline for (names, tags) |name, t| try std.testing.expectEqualStrings(@tagName(t), name);
    }
    const names = @typeInfo(CacheState).@"struct".field_names;
    try std.testing.expectEqual(tags.len, names.len);
    inline for (names, tags) |name, t| try std.testing.expectEqualStrings(@tagName(t), name);
}

test "every enabled arch has a prefix module, and only the archs with the state declare it" {
    inline for (tags) |t| {
        const owns_rows = comptime @hasDecl(P(t), "Rows");
        const owns_state = comptime @hasDecl(P(t), "State");
        try std.testing.expectEqual(owns_rows, comptime !isVoid(Payload(t, "Rows")));
        try std.testing.expectEqual(owns_state, comptime !isVoid(Payload(t, "State")));
        // A restore target without its hooks would fall through the dispatch at runtime.
        if (owns_state) inline for (.{ "reset", "restoreDisk", "restoreEntry" }) |hook| {
            if (!@hasDecl(P(t), hook)) @compileError(@tagName(t) ++ " declares State without " ++ hook);
        };
        if (owns_rows) inline for (.{ "keepThrough", "persistRows" }) |hook| {
            if (!@hasDecl(P(t), hook)) @compileError(@tagName(t) ++ " declares Rows without " ++ hook);
        };
    }
}

test "only an arch whose request state is a checkpoint requires checkpoints" {
    const rows = [_]struct { model_type: []const u8, want: bool }{
        .{ .model_type = "deepseek_v41", .want = true },
        .{ .model_type = "glm5_next", .want = false },
        .{ .model_type = "qwen4_exp", .want = false },
        .{ .model_type = "mimo_v2", .want = false },
        .{ .model_type = "mistral", .want = false },
    };
    for (rows) |r| {
        if (registry.ofModelType(r.model_type) == null and model.isServedArch(r.model_type)) continue;
        var config = model.ModelConfig{ .model_type = r.model_type };
        try std.testing.expectEqual(r.want, checkpointsRequired(&config));
    }
}
