//! What a model and a forward carry for their arch beside the trunk. `Trunk.arch_state` has a field per enabled arch,
//! the `State` its `arch.zig` declares; `ForwardCtx.arch_ctx` the same over `Ctx`. An arch that declares neither has
//! a void field, and every field is default-initialised, so a context built by hand carries each arch's empty state.
//!
//! A `State` may declare `ctx(*State) Ctx` (what the default context carries), `runsOwnModel` (its own model, not the
//! trunk's layers, runs the forward), `sharesReadonly` (weights every slot reads) and `rawEmbedding`. A `Ctx` with a
//! `request` field points at the request a slot owns beside its KV cache (`SlotRequest`).
const std = @import("std");
const mlx = @import("../mlx.zig");
const registry = @import("registry.zig");

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

/// The arch's own token embedding; null leaves the trunk's.
pub fn rawEmbedding(s: *const State, ids: mlx.mlx_array) !?mlx.mlx_array {
    inline for (tags) |t| {
        if (comptime has(Decl(t, "State"), "rawEmbedding")) if (try @field(s, @tagName(t)).rawEmbedding(ids)) |e| return e;
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
const arch_named_fields = [_][]const u8{
    "Trunk.qwen4",
    "Trunk.qwen4_stream_f32",
    "Trunk.qwen4_mixer",
    "Trunk.qwen4_mtp",
    "Trunk.qwen4_mtp_owner",
    "Trunk.decode_async_ladder_qwen4",
    "Trunk.qsa_consts",
    "Trunk.qsa_pooled_rope",
    "Trunk.qsa_score_bank_builds",
    "Trunk.compiled_gdn_gate",
    "Trunk.compiled_hc_silu",
    "Trunk.compiled_hc_mix",
    "Trunk.compiled_hc_inj",
    "Trunk.compiled_hc_write",
    "Trunk.gdn_ones_w",
    "Trunk.gdn_q_scale",
    "Trunk.gdn_k_scale",
    "Trunk.gdn_eps",
    "ForwardCtx.qsa_mask",
    "ForwardCtx.qsa_blocks",
    "ForwardCtx.qsa_arms",
    "ForwardCtx.skip_ple",
    "ForwardCtx.qsa_entry",
    "ForwardCtx.qsa_pos_base",
    "ForwardCtx.ple_defer",
    "ForwardCtx.ple_pending",
};

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
