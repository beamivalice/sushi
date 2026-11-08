//! What a model and a forward carry for their arch beside the trunk. `Trunk.arch_state` has a field per enabled arch,
//! the `State` its `arch.zig` declares; `ForwardCtx.arch_ctx` the same over `Ctx`. An arch that declares neither has
//! a void field, and every field is default-initialised, so a context built by hand carries each arch's empty state.
//!
//! A `State` may declare `init` (filled after the trunk's weights, from `InitArgs`), `ctx(*State) Ctx` (what the
//! default context carries), `runsOwnModel` (its own model, not the trunk's layers, runs the forward), `sharesReadonly`
//! (weights every slot reads), `specRollsBack`, `rawEmbedding`, `headShortlist` (an argmax-only readout of the lm
//! head), `clearCompiled` (closures bound to the compiling thread) and the trunk ops below that take the trunk. A `Ctx`
//! with a `request` field points at the request a slot owns beside its KV cache (`SlotRequest`); one may declare
//! `mrope`/`setMrope` (an M-RoPE position table) and `logMeters` (per-request meters).
const std = @import("std");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const mrope = @import("../mrope.zig");
const mtp = @import("../mtp.zig");
const registry = @import("registry.zig");
const arch_config = @import("config.zig");
const trunk = @import("../trunk.zig");
const Trunk = trunk.Trunk;
const ForwardCtx = trunk.ForwardCtx;

const tags = std.enums.values(registry.Tag);

fn Decl(comptime t: registry.Tag, comptime name: []const u8) type {
    return if (@hasDecl(registry.Mod(t), name)) @field(registry.Mod(t), name) else void;
}

fn defaultOf(comptime T: type) *const anyopaque {
    const v: T = if (T == void) {} else .{};
    return &v;
}

fn PerArch(comptime name: []const u8) type {
    var names: [tags.len][]const u8 = undefined;
    var types: [tags.len]type = undefined;
    var attrs: [tags.len]std.lang.Type.Struct.FieldAttributes = undefined;
    for (tags, &names, &types, &attrs) |t, *n, *ty, *a| {
        n.* = @tagName(t);
        ty.* = Decl(t, name);
        a.* = .{ .default_value_ptr = defaultOf(ty.*) };
    }
    return @Struct(.auto, null, &names, &types, &attrs);
}

/// `Trunk.arch_state`: each enabled arch's model-lifetime state.
pub const State = PerArch("State");
/// `ForwardCtx.arch_ctx`: each enabled arch's per-forward context.
pub const Ctx = PerArch("Ctx");

fn has(comptime T: type, comptime name: []const u8) bool {
    return T != void and @hasDecl(T, name);
}

/// The context a model's own state gives a forward that no slot supplies.
pub fn ctxOf(s: *State) Ctx {
    var out: Ctx = .{};
    inline for (tags) |t| {
        if (comptime has(Decl(t, "State"), "ctx")) @field(out, @tagName(t)) = @field(s, @tagName(t)).ctx();
    }
    return out;
}

pub fn runsOwnModel(s: *const State) bool {
    inline for (tags) |t| {
        if (comptime has(Decl(t, "State"), "runsOwnModel")) if (@field(s, @tagName(t)).runsOwnModel()) return true;
    }
    return false;
}

pub fn sharesReadonly(s: *const State) bool {
    inline for (tags) |t| {
        if (comptime has(Decl(t, "State"), "sharesReadonly")) if (@field(s, @tagName(t)).sharesReadonly()) return true;
    }
    return false;
}

/// Frees the arch's compiled closures, bound to the thread that compiled them.
pub fn clearCompiled(s: *State) void {
    inline for (tags) |t| {
        if (comptime has(Decl(t, "State"), "clearCompiled")) @field(s, @tagName(t)).clearCompiled();
    }
}

/// Which of an arch's kernels a load compiles: a server compiles every one, the offline tools the decode-state gates.
pub const CompileSet = enum { gates, all };

/// The trunk ops below are functions a `State` declares over the trunk. Each built arch's runs on every trunk, and one
/// whose state on `t` is empty finds nothing to do.
fn trunkHook(comptime name: []const u8, args: anytype) void {
    inline for (tags) |t| {
        if (comptime has(Decl(t, "State"), name)) @call(.auto, @field(Decl(t, "State"), name), args);
    }
}

fn trunkHookTry(comptime name: []const u8, args: anytype) !void {
    inline for (tags) |t| {
        if (comptime has(Decl(t, "State"), name)) try @call(.auto, @field(Decl(t, "State"), name), args);
    }
}

pub fn compile(t: *Trunk, set: CompileSet) void {
    trunkHook("compile", .{ t, set });
}

/// The arch's MTP head takes the trunk's KV scheme.
pub fn applyHeadKvQuant(t: *Trunk, kv: trunk.KVQuantConfig) !void {
    try trunkHookTry("applyHeadKvQuant", .{ t, kv });
}

/// Builds the MTP head's draft rerank; whether one is built.
pub fn buildDraftRerank(t: *Trunk) bool {
    inline for (tags) |tag| {
        const S = Decl(tag, "State");
        if (comptime has(S, "buildDraftRerank")) if (S.buildDraftRerank(t)) return true;
    }
    return false;
}

/// The MTP head's coarse-rerank shortlists for a group's stacked draft rows; false when the head built no rerank.
pub fn draftShortlists(t: *Trunk, x: mlx.mlx_array, suppress: ?mlx.mlx_array, out: []mtp.Shortlist) !bool {
    inline for (tags) |tag| {
        const S = Decl(tag, "State");
        if (comptime has(S, "draftShortlists")) return S.draftShortlists(t, x, suppress, out);
    }
    return error.NoMtpHead;
}

/// A checkpoint is taken at `row` of the MTP head's own rows.
pub fn markHeadCheckpoint(t: *Trunk, row: usize) !void {
    try trunkHookTry("markHeadCheckpoint", .{ t, row });
}

/// The per-model tables a forward builds from its slot's positions are stale: another slot runs next.
pub fn markTablesStale(t: *Trunk) void {
    trunkHook("markTablesStale", .{t});
}

/// Frees the per-model position tables.
pub fn resetTables(t: *Trunk) void {
    trunkHook("resetTables", .{t});
}

/// The slot's state leaves the batched decode group before a forward of its own.
pub fn releaseGroup(t: *Trunk, ctx: *ForwardCtx) !void {
    try trunkHookTry("releaseGroup", .{ t, ctx });
}

/// The slot whose `entries` these are leaves the batched decode group for good.
pub fn dropGroup(t: *Trunk, entries: []trunk.SSMCacheEntry) void {
    trunkHook("dropGroup", .{ t, entries });
}

/// Fills what a forward over lazy ids left for the host (`ForwardCtx.lazy_ids`), syncing on the ids.
pub fn flushDeferred(t: *Trunk, ctx: *ForwardCtx) !void {
    try trunkHookTry("flushDeferred", .{ t, ctx });
}

/// `flushDeferred` over the rows of one grouped forward, with one sync.
pub fn flushDeferredGroup(t: *Trunk, ctxs: []const *ForwardCtx) !void {
    try trunkHookTry("flushDeferredGroup", .{ t, ctxs });
}

/// Drops what a failed forward over lazy ids left for the host.
pub fn discardDeferred(t: *Trunk, ctx: *ForwardCtx) void {
    trunkHook("discardDeferred", .{ t, ctx });
}

/// The M-RoPE position table the forward's request carries; null for a text-only request or a build whose archs
/// rotate without one.
pub fn mropeOf(c: *const Ctx) ?mrope.PositionContext {
    inline for (tags) |t| {
        if (comptime has(Decl(t, "Ctx"), "mrope")) if (@field(c, @tagName(t)).mrope()) |p| return p;
    }
    return null;
}

/// Points the forward at the request's M-RoPE table, borrowed from the slot.
pub fn setMrope(c: *Ctx, pos: ?[]const i32, total: usize, delta: i32) void {
    inline for (tags) |t| {
        if (comptime has(Decl(t, "Ctx"), "setMrope")) @field(c, @tagName(t)).setMrope(pos, total, delta);
    }
}

/// Logs the per-request meters the forward's context tallied, and resets them.
pub fn logMeters(c: *Ctx) void {
    inline for (tags) |t| {
        if (comptime has(Decl(t, "Ctx"), "logMeters")) @field(c, @tagName(t)).logMeters();
    }
}

/// The arch's state rewinds across a speculative verify, so its own MTP head may draft.
pub fn specRollsBack(s: *const State) bool {
    inline for (tags) |t| {
        if (comptime has(Decl(t, "State"), "specRollsBack")) if (@field(s, @tagName(t)).specRollsBack()) return true;
    }
    return false;
}

/// The arch's own token embedding; null leaves the trunk's.
pub fn rawEmbedding(s: *const State, ids: mlx.mlx_array) !?mlx.mlx_array {
    inline for (tags) |t| {
        if (comptime has(Decl(t, "State"), "rawEmbedding")) if (try @field(s, @tagName(t)).rawEmbedding(ids)) |e| return e;
    }
    return null;
}

/// What a trunk arch's `State.init` reads once the trunk's weights are evaluated.
pub const InitArgs = struct {
    allocator: std.mem.Allocator,
    config: *const model.ModelConfig,
    weights: *model.Weights,
    prefix: []const u8,
    name_buf: *[256]u8,
    s: mlx.mlx_stream,
    /// Arrays the load made that the trunk frees with its own.
    owned_bf16: *?[]mlx.mlx_array,
};

/// Fills the state of the arch `args.config` names, last in the trunk's construction: nothing after it can fail, so a
/// `State.init` that fails releases what it made. Returns the expert engine of a streamed load.
pub fn initState(s: *State, args: InitArgs) !?Trunk.ExpertStreamParts {
    const tag = arch_config.tagOf(args.config) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime has(Decl(t, "State"), "init")) try @field(s, @tagName(t)).init(args) else null,
    };
}

/// The arch's argmax-only readout of `x` through `t`'s head; null reads the full head.
pub fn headShortlist(t: *const Trunk, x: mlx.mlx_array) !?mlx.mlx_array {
    inline for (tags) |tg| {
        if (comptime has(Decl(tg, "State"), "headShortlist")) if (try @field(t.arch_state, @tagName(tg)).headShortlist(t, x)) |out| return out;
    }
    return null;
}

/// The arch whose `Ctx` points at a request the slot owns; null when no built arch keeps one.
const request_arch: ?registry.Tag = blk: {
    var found: ?registry.Tag = null;
    for (tags) |t| if (Decl(t, "Ctx") != void and @hasField(Decl(t, "Ctx"), "request")) {
        if (found != null) @compileError("two archs keep a slot request; Slot holds one");
        found = t;
    };
    break :blk found;
};

/// Stands in for the slot request in a build where no arch keeps one: there is never one to hold.
pub const NoRequest = struct {
    pub fn deinit(_: *NoRequest) void {}
    pub fn reset(_: *NoRequest) void {}
    pub fn residentBytes(_: *const NoRequest) u64 {
        return 0;
    }
};

/// The request a slot owns beside its KV cache, the one `request_arch`'s `Ctx` points at.
pub const SlotRequest = if (request_arch) |t|
    @typeInfo(@typeInfo(@FieldType(Decl(t, "Ctx"), "request")).optional.child).pointer.child
else
    NoRequest;

/// The context a slot's forward carries: `request_arch` pointed at the slot's request.
pub fn ctxForSlot(request: ?*SlotRequest) Ctx {
    var out: Ctx = .{};
    if (comptime request_arch) |t| @field(out, @tagName(t)).request = request;
    return out;
}

test "arch_state and arch_ctx have a field per enabled arch, its declared type or void" {
    inline for (.{ .{ State, "State" }, .{ Ctx, "Ctx" } }) |c| {
        const names = @typeInfo(c[0]).@"struct".field_names;
        try std.testing.expectEqual(tags.len, names.len);
        inline for (names, tags) |name, t| {
            try std.testing.expectEqualStrings(@tagName(t), name);
            try std.testing.expect(@FieldType(c[0], name) == Decl(t, c[1]));
        }
    }
    const empty: Ctx = .{};
    _ = empty;
}

/// Fields of the engine handles that still name an arch: each arch's S3 stage moves its own into `arch_state` /
/// `arch_ctx` and deletes its lines here. Empty means the seam is done.
const arch_named_fields = [_][]const u8{};

/// Whether `field`'s `_`-separated words hold `term`'s words in a row: `gdn` matches `compiled_gdn_gate`, not `agdn`.
fn namesTerm(comptime field: []const u8, comptime term: []const u8) bool {
    var i: usize = 0;
    while (i + term.len <= field.len) : (i += 1) {
        const starts = i == 0 or field[i - 1] == '_';
        const ends = i + term.len == field.len or field[i + term.len] == '_';
        if (starts and ends and std.mem.eql(u8, field[i..][0..term.len], term)) return true;
    }
    return false;
}

/// The arch name without its version digits (`glm5` -> `glm`), the stem a field naming that arch carries.
fn stem(comptime name: []const u8) []const u8 {
    var n = name.len;
    while (n > 0 and std.ascii.isDigit(name[n - 1])) n -= 1;
    return name[0..n];
}

test "no field of Trunk, ForwardCtx or Slot names an arch beyond the allowlist, and every allowlisted one still exists" {
    comptime {
        @setEvalBranchQuota(200_000);
        const handles = .{ @import("../trunk.zig").Trunk, @import("../trunk.zig").ForwardCtx, @import("../scheduler.zig").Slot };
        var offenders: []const u8 = "";
        var seen: [arch_named_fields.len]bool = @splat(false);
        for (handles) |T| for (@typeInfo(T).@"struct".field_names) |field| {
            const type_name = @typeName(T);
            const qualified = type_name[if (std.mem.lastIndexOfScalar(u8, type_name, '.')) |i| i + 1 else 0..] ++ "." ++ field;
            const names_arch = for (registry.registered_names) |arch| {
                if (std.mem.indexOf(u8, field, stem(arch)) != null) break true;
            } else for (registry.registered_seam_terms) |term| {
                if (namesTerm(field, term)) break true;
            } else false;
            if (names_arch) {
                const at: ?usize = for (arch_named_fields, 0..) |allowed, i| {
                    if (std.mem.eql(u8, allowed, qualified)) break i;
                } else null;
                if (at) |i| {
                    seen[i] = true;
                } else {
                    offenders = offenders ++ " " ++ qualified;
                }
            }
        };
        for (arch_named_fields, seen) |allowed, s| {
            if (!s) offenders = offenders ++ " stale:" ++ allowed;
        }
        if (offenders.len > 0) @compileError("arch-named handle fields:" ++ offenders);
    }
}
