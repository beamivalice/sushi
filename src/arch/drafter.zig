//! The speculative drafter of a loaded model, one prong per registered architecture that drafts from a head, and the
//! committed history it keeps per request. The controller around it is head-agnostic (the EV planner, the acceptance
//! gate, the pre-draft and the horizon valve work in tokens and probabilities); everything a head decides is behind
//! the operations here, each answered by the arch's own `<arch>_drafter.zig` as the arch publishes it (`drafter` in
//! its `arch.zig`). A module that omits an operation gets the default stated on the method.
//!
//! A built `@Union` cannot carry methods, so `Drafter` and `Cache` wrap the unions the registry yields.
const std = @import("std");
const mlx = @import("../mlx.zig");
const mtp_mod = @import("../mtp.zig");
const registry = @import("registry.zig");
const spec_types = @import("../spec_types.zig");
const transformer_mod = @import("../transformer.zig");
const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;

fn drafterOf(comptime name: []const u8) type {
    return registry.Mod(@field(registry.Tag, name)).drafter;
}

fn namesWith(comptime decl: []const u8) []const []const u8 {
    comptime {
        var out: []const []const u8 = &.{};
        for (std.meta.fieldNames(registry.Tag)) |n| {
            if (@hasDecl(drafterOf(n), decl)) out = out ++ &[_][]const u8{n};
        }
        return out;
    }
}

/// One prong per registered arch whose drafter declares `decl`, typed by it.
fn UnionOf(comptime decl: []const u8) type {
    const names = comptime namesWith(decl);
    var values: [names.len]u8 = undefined;
    var types: [names.len]type = undefined;
    for (names, 0..) |n, i| {
        values[i] = i;
        types[i] = @field(drafterOf(n), decl);
    }
    const Tag = @Enum(u8, .exhaustive, names, &values);
    return @Union(.auto, Tag, names, &types, &@splat(.{}));
}

const HeadUnion = UnionOf("Head");
const CacheUnion = UnionOf("Cache");

pub const Drafter = struct {
    u: HeadUnion,

    pub const Cache = DraftCache;
    pub const Tag = std.meta.Tag(HeadUnion);

    fn D(comptime t: Tag) type {
        return drafterOf(@tagName(t));
    }

    /// Is the arch registered as `name` built, with a head to draft from?
    pub fn has(comptime name: []const u8) bool {
        return @hasField(HeadUnion, name);
    }

    /// The drafter a loaded model gets, from the first built arch that claims it: `mtp_enabled` says whether the
    /// request for MTP stands, `aux` is a head loaded beside the trunk (MiMo's), if any.
    pub fn fromModel(model: *Transformer, mtp_enabled: bool, aux: ?*anyopaque) ?Drafter {
        inline for (comptime std.meta.fieldNames(Tag)) |name| {
            const M = drafterOf(name);
            if (comptime @hasDecl(M, "headOf")) {
                if (M.headOf(model, mtp_enabled, aux)) |h| return of(name, h);
            }
        }
        return null;
    }

    /// The drafter of the arch registered as `name`.
    pub fn of(comptime name: []const u8, head: @FieldType(HeadUnion, name)) Drafter {
        return .{ .u = @unionInit(HeadUnion, name, head) };
    }

    /// How this head drafts: shared code asks this and never the arch.
    pub fn kind(self: Drafter) spec_types.Kind {
        return switch (self.u) {
            inline else => |_, t| D(t).head_kind,
        };
    }

    /// Is this head's decode state module-owned (one per model) rather than per-request? Both the scheduler and the
    /// sticky-serial arm ask this.
    pub fn moduleOwned(self: Drafter) bool {
        return switch (self.u) {
            inline else => |h, t| if (comptime @hasDecl(D(t), "moduleOwned")) D(t).moduleOwned(h) else false,
        };
    }

    /// Do several requests' chains build in one head forward?
    pub fn batchesHeadRows(self: Drafter) bool {
        return switch (self.u) {
            inline else => |_, t| @hasDecl(D(t), "batches_head_rows") and D(t).batches_head_rows,
        };
    }

    /// Builds the next draft step of several requests' chains in one head forward (see `batchesHeadRows`).
    pub fn buildBatchedChains(self: Drafter, gens: anytype, chains: anytype, from: u32, to: u32) !void {
        return switch (self.u) {
            inline else => |_, t| if (comptime @hasDecl(D(t), "buildBatchedChains"))
                D(t).buildBatchedChains(gens, chains, from, to)
            else
                error.DrafterHasNoBatchedRows,
        };
    }

    /// Frees a head the engine allocated on its own (a head the trunk owns is freed with it).
    pub fn destroyHead(self: Drafter, allocator: std.mem.Allocator) void {
        switch (self.u) {
            inline else => |h, t| if (comptime @hasDecl(D(t), "destroyHead")) D(t).destroyHead(h, allocator),
        }
    }

    pub fn makeCache(self: Drafter, allocator: std.mem.Allocator) !Cache {
        return switch (self.u) {
            inline else => |h, t| if (comptime @hasDecl(D(t), "makeCache"))
                Cache.of(@tagName(t), try D(t).makeCache(h, allocator))
            else
                error.DrafterHasNoHeadCache,
        };
    }

    /// One head forward over L positions. `.logits` returns the LAST row only. `.mixed` asks for the lm_head's INPUT
    /// vector instead, returned in `rerank_x`. `host_ids` are `id_arr`'s values when the caller holds them (a
    /// round's committed rows); the MiMo arm records them.
    pub fn forward(
        self: Drafter,
        target: *Transformer,
        cache: *Cache,
        id_arr: mlx.mlx_array,
        hidden: mlx.mlx_array,
        rope_offset: c_int,
        want: mtp_mod.StepWant,
        mrope_ctx: ?mtp_mod.MropeContext,
        host_ids: ?[]const u32,
    ) !mtp_mod.StepOut {
        return switch (self.u) {
            inline else => |h, t| if (comptime @hasDecl(D(t), "forward"))
                D(t).forward(h, target, &@field(cache.u, @tagName(t)), id_arr, hidden, rope_offset, want, mrope_ctx, host_ids)
            else
                error.DrafterHasNoHeadStep,
        };
    }

    /// Appends committed history without projecting logits.
    pub fn appendHistory(
        self: Drafter,
        target: *Transformer,
        cache: *Cache,
        token_ids: []const u32,
        hidden: mlx.mlx_array,
        rope_offset: c_int,
        mrope_ctx: ?mtp_mod.MropeContext,
        allocator: std.mem.Allocator,
    ) !void {
        return switch (self.u) {
            inline else => |h, t| if (comptime @hasDecl(D(t), "appendHistory"))
                D(t).appendHistory(h, target, &@field(cache.u, @tagName(t)), token_ids, hidden, rope_offset, mrope_ctx, allocator)
            else
                error.DrafterHasNoHeadStep,
        };
    }

    /// Draft-rerank scheme (see mtp.zig): greedy drafts pick via a coarse 2-bit readout plus a trunk-head top-32
    /// re-score instead of a full draft-head projection and argmax.
    pub fn canRerankDrafts(self: Drafter) bool {
        return switch (self.u) {
            inline else => |h, t| if (comptime @hasDecl(D(t), "canRerankDrafts")) D(t).canRerankDrafts(h) else false,
        };
    }

    /// `x` is whatever the TARGET's lm_head consumes for this arm.
    pub fn draftSelect(self: Drafter, target: *Transformer, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
        return switch (self.u) {
            inline else => |h, t| if (comptime @hasDecl(D(t), "draftSelect"))
                D(t).draftSelect(h, target, x, suppress_mask)
            else
                error.DrafterHasNoHeadStep,
        };
    }

    /// The exact re-scored top-32 shortlist a SAMPLED draft proposes from, or null when the coarse rerank head is
    /// unavailable. `x` is the same vector `draftSelect` takes.
    pub fn draftShortlist(self: Drafter, target: *Transformer, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !?mtp_mod.Shortlist {
        return switch (self.u) {
            inline else => |h, t| if (comptime @hasDecl(D(t), "draftShortlist")) D(t).draftShortlist(h, target, x, suppress_mask) else null,
        };
    }

    /// Every G17/NAX cost surface is calibrated against one exact runtime geometry, so a head without a surface of
    /// its own plans under `generic`: an off-profile head served by a calibrated surface would plan depths the
    /// measurement never covered.
    pub fn costProfile(self: Drafter, target: *const Transformer, kv: transformer_mod.KVQuantConfig) mtp_mod.MtpCostProfile {
        return switch (self.u) {
            inline else => |h, t| if (comptime @hasDecl(D(t), "costProfile")) D(t).costProfile(h, target, kv) else .generic,
        };
    }

    /// The head's last healthy acceptance surface, or null when it has never published one. The seed lives on the
    /// head object itself, so it dies with the head and can never cross models.
    pub fn evSeed(self: Drafter) ?spec_types.EvSeed {
        return switch (self.u) {
            inline else => |h, t| if (comptime @hasDecl(D(t), "evSeed")) D(t).evSeed(h) else null,
        };
    }

    pub fn setEvSeed(self: Drafter, accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32) void {
        switch (self.u) {
            inline else => |h, t| if (comptime @hasDecl(D(t), "setEvSeed")) D(t).setEvSeed(h, accept, m_lo),
        }
    }
};

/// The head's committed-history cache.
pub const DraftCache = struct {
    u: CacheUnion,

    pub fn of(comptime name: []const u8, payload: @FieldType(CacheUnion, name)) DraftCache {
        return .{ .u = @unionInit(CacheUnion, name, payload) };
    }

    /// The arch-typed state of the cache registered as `name`, for the code that is that arch's.
    pub fn as(self: *const DraftCache, comptime name: []const u8) *const @FieldType(CacheUnion, name) {
        return &@field(self.u, name);
    }

    /// Installs this request's state on the head; a no-op where the history is per-request already.
    pub fn activate(self: *const DraftCache) void {
        switch (self.u) {
            inline else => |*c| c.activate(),
        }
    }

    pub fn step(self: *const DraftCache) usize {
        return switch (self.u) {
            inline else => |*c| c.step(),
        };
    }

    /// The underlying KVCache the prefix cache's spec-snap machinery snapshots and restores. A head whose state is
    /// not KV-only also needs `head()`. Null when the head keeps nothing to snapshot.
    pub fn kv(self: *const DraftCache) ?*KVCache {
        return switch (self.u) {
            inline else => |*c| c.kv(),
        };
    }

    /// The Transformer owning an in-checkpoint head; null on the other arms and whenever `kv()` is null.
    pub fn head(self: *const DraftCache) ?*Transformer {
        return switch (self.u) {
            inline else => |*c| if (comptime @hasDecl(@TypeOf(c.*), "head")) c.head() else null,
        };
    }

    pub fn truncate(self: *DraftCache, len: usize, s: mlx.mlx_stream) !void {
        switch (self.u) {
            inline else => |*c| try c.truncate(len, s),
        }
    }

    pub fn deinit(self: *DraftCache) void {
        switch (self.u) {
            inline else => |*c| c.deinit(),
        }
    }

    /// Appends this cache's storage to a prefill eval batch, so the chunk's activation graph can be freed with the
    /// chunk instead of accumulating across the whole prompt.
    pub fn appendEvalArrays(self: *const DraftCache, vec: mlx.mlx_vector_array) void {
        switch (self.u) {
            inline else => |*c| c.appendEvalArrays(vec),
        }
    }

    /// Is the history rebuilt from the prompt's tail, so the final prefill chunk's storage need not be evaluated?
    pub fn rebuiltFromTail(self: *const DraftCache) bool {
        return switch (self.u) {
            inline else => |*c| @hasDecl(@TypeOf(c.*), "rebuilt_from_tail") and @TypeOf(c.*).rebuilt_from_tail,
        };
    }

    /// How far the head's position drifted from where the round starting at head row `off0` would read it; 0 where
    /// the head keeps no position of its own.
    pub fn positionDrift(self: *const DraftCache, off0: usize) i64 {
        return switch (self.u) {
            inline else => |*c| if (comptime @hasDecl(@TypeOf(c.*), "positionDrift")) c.positionDrift(off0) else 0,
        };
    }

    /// After a forced commit left a hole in the history: starts a head that refuses a gap over. True when it did,
    /// and the pending history stash is stale.
    pub fn reseedAfterForcedCommit(self: *const DraftCache) !bool {
        return switch (self.u) {
            inline else => |*c| if (comptime @hasDecl(@TypeOf(c.*), "reseedAfterForcedCommit")) blk: {
                try c.reseedAfterForcedCommit();
                break :blk true;
            } else false,
        };
    }
};

/// Does the arch running `model` list DFlash2 among its drafters?
pub fn supportsDflash(model: *Transformer) bool {
    return switch (registry.handleOf(model)) {
        inline else => |_, t| blk: {
            inline for (registry.Mod(t).drafter.kinds) |k| if (k == .dflash2) break :blk true;
            break :blk false;
        },
    };
}

/// Do the arch's DFlash2 rounds choose their verify rows per round (a verify row costs bytes there)?
pub fn dflashAdaptiveRows(model: *Transformer) bool {
    return switch (registry.handleOf(model)) {
        inline else => |_, t| @hasDecl(registry.Mod(t).drafter, "dflash_adaptive_rows") and registry.Mod(t).drafter.dflash_adaptive_rows,
    };
}

test "an arch has a head prong exactly when its drafter runs from a head" {
    const head_names = comptime std.meta.fieldNames(Drafter.Tag);
    comptime var heads = 0;
    inline for (comptime std.meta.fieldNames(registry.Tag)) |name| {
        const M = drafterOf(name);
        const from_head = comptime blk: {
            for (M.kinds) |k| if (k == .mtp or k == .dspark) break :blk true;
            break :blk false;
        };
        try std.testing.expectEqual(from_head, @hasDecl(M, "Head"));
        try std.testing.expectEqual(from_head, @hasField(Drafter.Tag, name));
        if (from_head) {
            heads += 1;
            comptime var declares = false;
            inline for (M.kinds) |k| declares = declares or k == M.head_kind;
            try std.testing.expect(declares);
        }
    }
    try std.testing.expectEqual(heads, head_names.len);
}

test "every registered arch publishes a drafter, and only an arch listing dflash2 names a DFlash2 round" {
    inline for (comptime std.meta.fieldNames(registry.Tag)) |name| {
        const M = drafterOf(name);
        const dflash = comptime blk: {
            for (M.kinds) |k| if (k == .dflash2) break :blk true;
            break :blk false;
        };
        try std.testing.expectEqual(dflash, @hasDecl(M, "nextDflash"));
    }
}

test "the loaded arch decides whether a model takes a DFlash sidecar and how its rounds pick rows" {
    var t: Transformer = undefined;
    // The default arch, MiMo, runs the generic round and verifies the whole block.
    if (comptime registry.default) |tag| {
        t.arch = tag;
        try std.testing.expect(supportsDflash(&t));
        try std.testing.expect(!dflashAdaptiveRows(&t));
    }
    if (registry.ofModelType("qwen4_exp")) |tag| {
        t.arch = tag;
        try std.testing.expect(supportsDflash(&t));
        try std.testing.expect(dflashAdaptiveRows(&t));
    }
    if (registry.ofModelType("glm5_next")) |tag| {
        t.arch = tag;
        try std.testing.expect(supportsDflash(&t));
    }
    if (registry.ofModelType("deepseek_v41")) |tag| {
        t.arch = tag;
        try std.testing.expect(!supportsDflash(&t));
    }
}
