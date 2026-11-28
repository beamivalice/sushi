//! Profile-keyed shared held load for `test-hy4`: one Hy4 ledger load fits on the ledger at a time, kept across
//! tests of the same `(fixture, profile)` and wired to `test_rest.held` so each test starts with clean KV/SSM/request
//! state and the process's end unloads it cleanly.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const sampler = @import("../sampler.zig");
const bills = @import("../bills.zig");
const arch_handle = @import("../arch/handle.zig");
const arch_sched = @import("../arch/sched.zig");
const contract = @import("../cooler_contract_test.zig");
const hy4_contract = @import("hy4_contract.zig");
const hy4_load = @import("hy4_load.zig");
const VisionEncoder = @import("../vision.zig").VisionEncoder;

const smp = std.heap.smp_allocator;
const io = std.Io.Threaded.global_single_threaded.io();

pub const Profile = enum {
    stq_openLight,
    stq_resident,
    stq_drafting_suppress,
    bf16_openLight,
    bf16_vision,
};

/// Experts resident in the stream so concurrent decode rows join into `batchedForward`, DFly2 kept when shipped.
pub const ResidentHy4 = struct {
    pub fn prepare(a: std.mem.Allocator, config: *model_mod.ModelConfig, dir: []const u8) !void {
        try hy4_contract.prepare(a, config, dir);
        config.expert_resident = true;
        config.expert_cache_bytes = std.math.maxInt(u64) / 2;
    }
};

/// Experts resident without the expert stream, DFly2 kept where the pack ships it.
pub const Drafting = struct {
    pub fn prepare(a: std.mem.Allocator, config: *model_mod.ModelConfig, dir: []const u8) !void {
        config.expert_streaming = false;
        config.expert_source_dir = try a.dupe(u8, dir);
        config.stream_mtp_head = hy4_load.shipsHead(io, a, dir);
    }
};

pub const Held = struct {
    profile: Profile,
    dir: []u8,
    ld: *contract.Load,
    tower: ?VisionEncoder = null,
    config_hash: u64,
    sheet: contract.Sheet,
    keep_hidden: bool,
    serves_requests: bool,
};

var held_state: ?Held = null;

pub fn envOf(profile: Profile) [:0]const u8 {
    return switch (profile) {
        .stq_openLight, .stq_resident, .stq_drafting_suppress => "HY4_STQ_TINY",
        .bf16_openLight, .bf16_vision => "HY4_BF16_TINY",
    };
}

pub fn currentProfile() ?Profile {
    return if (held_state) |*h| h.profile else null;
}

/// Returns the active held load for `profile`, releasing any prior held load of a different profile first.
pub fn acquire(profile: Profile) !*Held {
    const dir = std.mem.span(std.c.getenv(envOf(profile)) orelse return error.SkipZigTest);
    if (held_state) |*h| {
        if (h.profile == profile and std.mem.eql(u8, h.dir, dir)) return h;
    }
    releaseHeld();
    mlx.engine.close();
    mlx.engine.trim();

    const ld = try smp.create(contract.Load);
    errdefer smp.destroy(ld);
    var tower: ?VisionEncoder = null;

    switch (profile) {
        .stq_openLight, .bf16_openLight => {
            try ld.prepareOn(smp, hy4_contract, dir, true);
            ld.config.test_skip_warm = true;
            try ld.loadServed(dir);
        },
        .stq_resident => {
            try ld.prepareOn(smp, ResidentHy4, dir, true);
            try ld.loadServed(dir);
        },
        .stq_drafting_suppress => {
            try ld.prepareOn(smp, Drafting, dir, true);
            ld.config.test_skip_warm = true;
            try ld.load(dir);
            errdefer ld.discard();
            const vocab: usize = ld.xfm.config.vocab_size;
            ld.xfm.suppress_mask = try sampler.buildSuppressMask(&.{ 5, 9 }, 0, vocab, ld.xfm.outStream());
            try ld.serve();
        },
        .bf16_vision => {
            try ld.prepareOn(smp, hy4_contract, dir, true);
            ld.vision = true;
            try ld.load(dir);
            tower = VisionEncoder.init(smp, ld.config, &ld.weights) catch |e| {
                ld.discard();
                return e;
            };
            ld.serve() catch |e| {
                tower.?.deinit();
                ld.discard();
                return e;
            };
        },
    }
    errdefer {
        if (tower) |*t| t.deinit();
        _ = ld.close();
    }

    const name = try smp.dupe(u8, dir);
    errdefer smp.free(name);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    held_state = .{
        .profile = profile,
        .dir = name,
        .ld = ld,
        .tower = tower,
        .config_hash = configHash(&ld.config),
        .sheet = undefined,
        .keep_hidden = m.keep_hidden,
        .serves_requests = m.serves_requests,
    };

    const rest = @import("../test_rest.zig");
    rest.held = .{ .holds = rest.held.holds + 1, .rest = heldRest, .close = heldClose };
    var none = std.Io.Writer.fixed(&.{});
    _ = heldRest(&none);
    held_state.?.sheet = contract.Sheet.now();
    return &held_state.?;
}

/// Unloads any currently held load so a test that must open, mutate, or fail its own fresh load starts clean.
pub fn releaseHeld() void {
    const rest = @import("../test_rest.zig");
    if (held_state) |*h| {
        if (h.tower) |*t| t.deinit();
        _ = h.ld.close();
        smp.destroy(h.ld);
        smp.free(h.dir);
        held_state = null;
        rest.held = .{};
    } else if (rest.held.holds > 0) {
        if (rest.held.close) |close_fn| {
            var none = std.Io.Writer.fixed(&.{});
            _ = close_fn(&none);
        } else {
            rest.held = .{};
        }
    }
}

fn configHash(config: *const model_mod.ModelConfig) u64 {
    return std.hash.Wyhash.hash(0, std.mem.asBytes(config));
}

fn heldRest(w: *std.Io.Writer) bool {
    const h = &(held_state orelse return false);
    var left = false;
    const open = mlx.engine.openDispatches();
    if (open != 0) {
        w.print(" {d} dispatches open beside the held loads;", .{open}) catch {};
        left = true;
    }
    if (configHash(&h.ld.config) != h.config_hash) {
        w.print(" the held load of {s} with its config changed;", .{std.fs.path.basename(h.dir)}) catch {};
        left = true;
    }
    if (h.ld.xfm.arch_state.hy4flash.model) |m| {
        m.bound = null;
        m.own_lent = false;
        m.keep_hidden = h.keep_hidden;
        m.serves_requests = h.serves_requests;
        m.own.kv = bills.kvCacheForLaunch(&h.ld.config, null).config;
    }
    const turn = arch_handle.turnFor(h.ld.h, &h.ld.config);
    defer turn.end();
    h.ld.xfm.resetCache() catch {
        w.writeAll(" a held load's cache not reset;") catch {};
        left = true;
    };
    h.ld.xfm.cache.reinit(h.ld.config.num_hidden_layers, bills.kvCacheForLaunch(&h.ld.config, null).config) catch {
        w.writeAll(" a held load's cache not made again;") catch {};
        left = true;
    };
    arch_sched.endRequest(&h.ld.config);
    const now = contract.Sheet.now();
    if (now.mapped != h.sheet.mapped or now.regions != h.sheet.regions) {
        w.print(" beside the held loads: {d} B mapped ({d} held), {d} owned regions ({d});", .{ now.mapped, h.sheet.mapped, now.regions, h.sheet.regions }) catch {};
        return true;
    }
    return left;
}

fn heldClose(w: *std.Io.Writer) bool {
    releaseHeld();
    const now = contract.Sheet.now();
    if (now.in_use == 0 and now.regions == 0 and now.open == 0) return false;
    w.print(" {d} B in use, {d} owned regions, {d} dispatches open after the held loads' unload;", .{ now.in_use, now.regions, now.open }) catch {};
    return true;
}
