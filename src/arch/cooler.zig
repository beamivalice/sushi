//! What every load on the engine of our own owes, whatever its arch, in one place:
//! - the first narrow forward after a wide one (decode after a prefill) leaves at most its cap's margin of free buffers,
//!   so its own asks come on top of its bill's transient and a decode does not park a prefill's scratch; nothing is refused, since the shared bill is not a measured cap
//!   of the engine's own forward (buffers an open pass binds outlive their tensors). The wide cap is its model's: that
//!   model's next narrow forward lifts it, in whatever turn it runs (`engine.budgetWide`);
//! - a request's end closes its pass and leaves the pool at most its resting cap of free buffers (what the next
//!   request's first verify forward takes, which the ledger bills as parked);
//! - an unload leaves the pool empty;
//! - a load whose warm fails does not serve (`model_load`'s warm), and a load whose warm ends is sealed: a pipeline made
//!   after it is late.
//! An arch whose own module installs its forwards' measured caps (`cooler_own_caps`, DeepSeek's beside the expert pools
//! it releases) keeps them and its unload's, and names its resting cap (`coolerRestCap`); the rest are the same for all.
const std = @import("std");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const bills = @import("../bills.zig");
const registry = @import("registry.zig");
const model_mod = @import("../model.zig");
const model_handle = @import("../model_handle.zig");
const cooler_side = @import("../cooler_side.zig");
const ModelConfig = model_mod.ModelConfig;
const Trunk = @import("../trunk.zig").Trunk;

/// A verify block's rows: every forward of a request is billed at no fewer.
const verify_rows: u64 = @import("../sampler_draw.zig").max_rows;

comptime {
    for (std.enums.values(registry.Tag)) |t| {
        const M = registry.Mod(t);
        if (@hasDecl(M, "cooler_own_caps") and M.cooler_own_caps and !@hasDecl(M, "coolerRestCap"))
            @compileError(@tagName(t) ++ " keeps its own pool caps and names no resting cap (coolerRestCap)");
    }
}

const onEngine = cooler_side.onEngine;

/// An arch on the ledger (`bills.ledgerArch`) runs its narrow forwards on its plans and its wide ones in its region: its
/// pool is its turn's gate (D64), never a forward's cap, and keeps no free buffer past a request.
const plansForwards = bills.ledgerArch;

/// Whether the arch's own module installs and lifts its caps (DeepSeek's beside the expert pools it releases).
fn ownCaps(config: *const ModelConfig) bool {
    const tag = registry.ofModelType(config.model_type) orelse return false;
    switch (tag) {
        inline else => |t| return comptime @hasDecl(registry.Mod(t), "cooler_own_caps") and registry.Mod(t).cooler_own_caps,
    }
}

/// PURE: the bytes a forward of `rows` ending at `seq` is billed to add to the pool: its transient at no fewer rows than
/// a verify block, and where that transient is exact (`bills.coolerPrefillRowBytes`), the one margin a bill carries.
pub fn forwardCap(config: *const ModelConfig, rows: u64, seq: u64) u64 {
    const kv = bills.configuredKvQuantFor(config);
    const kv_bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
    return bills.prefillTransientReserveAtKv(config, kv_bits, @max(rows, verify_rows), seq) +| forwardMargin(config);
}

/// PURE: the one margin a forward's cap carries where its transient is exact, else none. It is also what a lift leaves
/// free: the forward's own asks, which no parked buffer of another size serves, land on top of the buffers that stay,
/// so only what the transient's bill leaves over can stay parked.
fn forwardMargin(config: *const ModelConfig) u64 {
    return if (bills.coolerBillExact(config)) bills.BILL_MARGIN_BYTES else 0;
}

/// PURE: `forwardCap` of a forward that reads out every row through the head, which asks `readout` bytes
/// (`Trunk.lmHeadReadoutBytes`) past the last layer. A served prefill chunk reads out none, and so does the decode meter's
/// (`model_load.meterPrefill`); `sushi kld` reads out its whole width.
pub fn readoutCap(config: *const ModelConfig, rows: u64, seq: u64, readout: u64) u64 {
    return forwardCap(config, rows, seq) +| bills.atTrunkRows(config, rows, readout);
}

/// PURE: the free buffers a request's end leaves the pool at most.
pub fn restCap(config: *const ModelConfig) u64 {
    // A load on the ledger runs its narrow phases on its arena: the next request asks the pool for none of them, and
    // what the pool kept past the seal would be held outside every line.
    if (plansForwards(config)) return 0;
    const tag = registry.ofModelType(config.model_type) orelse return forwardCap(config, verify_rows, 1);
    switch (tag) {
        inline else => |t| if (comptime @hasDecl(registry.Mod(t), "coolerRestCap")) return registry.Mod(t).coolerRestCap(config),
    }
    return forwardCap(config, verify_rows, 1);
}

/// `readout`: the head's bytes over a wide forward's rows when it reads them out (`readoutCap`), else 0.
pub const Rows = struct {
    rows: u64,
    seq: u64,
    readout: u64 = 0,

    /// Null where the arch's forwards run under no cap (`plansForwards`).
    pub fn cap(r: Rows, config: *const ModelConfig) ?u64 {
        if (plansForwards(config)) return null;
        return readoutCap(config, r.rows, r.seq, r.readout);
    }
};

/// A forward of the model of `config`, whose trunk is `head`, begins: its rows, and the position its last row ends at,
/// read only when the shared seam acts on them or `want` asks (each read of a shape is a query of MLX's on an MLX load).
pub fn enterForward(config: *const ModelConfig, ctx: *const model_handle.ForwardCtx, ids: mlx.mlx_array, head: *Trunk, want: bool) ?Rows {
    const seam = onEngine(config) and !ownCaps(config) and !plansForwards(config);
    if (!seam and !want) return null;
    const shape = mlx.getShape(ids);
    const rows: u64 = if (shape.len == 0) 1 else @intCast(shape[shape.len - 1]);
    var at: Rows = .{ .rows = rows, .seq = @as(u64, ctx.cache.kvLenForBatching()) + rows };
    if (!seam) return at;
    if (rows > verify_rows) {
        if (!ctx.skip_lm_head) at.readout = head.lmHeadReadoutBytes(rows);
        // A prefill chunk holds no more than its bill: under the cap the pool gives up its free buffers before it asks
        // Metal for more. Uncapped, a 4096-row chunk of the Qwen cut kept ~2 GB of them on 2.9 GB in use, 1.4 GB past
        // its bill. The model's next narrow forward, or the request's end, lifts it.
        enterWide(&head.cooler_budget, at.cap(config).?);
        return at;
    }
    liftWide(config, &head.cooler_budget);
    return at;
}

/// A forward wider than a verify block of the model of `slot` begins on the engine under `bytes`: its model's cap until
/// that model's next narrow forward.
pub fn enterWide(slot: *mlx.engine.BudgetSlot, bytes: u64) void {
    mlx.engine.close();
    mlx.engine.budgetWide(slot, bytes) catch |e| log.warn("[engine] pool: the cap of {d} MB was not installed: {s}\n", .{ bytes >> 20, @errorName(e) });
}

/// A forward of several requests' rows of the model of `config` and `slot` begins (joined decode rows, a grouped
/// verify). It is a narrow forward whatever its rows: the model's wide forward's cap is lifted for it, none installed.
pub fn enterJoined(config: *const ModelConfig, slot: *mlx.engine.BudgetSlot) void {
    if (!joinedLifts(config, slot)) return;
    liftWide(config, slot);
}

/// Whether a joined forward of the model of `config` and `slot` would lift its cap now.
pub fn joinedLifts(config: *const ModelConfig, slot: *const mlx.engine.BudgetSlot) bool {
    return slot.wide and onEngine(config) and !ownCaps(config);
}

/// After a wide forward of this model, the free buffers past a narrow forward's margin go and its cap is lifted.
fn liftWide(config: *const ModelConfig, slot: *mlx.engine.BudgetSlot) void {
    if (!slot.wide) return;
    mlx.engine.close();
    const bytes = forwardMargin(config);
    mlx.engine.liftWide(slot, bytes) catch |e| log.warn("[engine] pool: the cap of {d} MB was not installed: {s}\n", .{ bytes >> 20, @errorName(e) });
}

fn install(bytes: u64) void {
    mlx.engine.budget(bytes) catch |e| log.warn("[engine] pool: the cap of {d} MB was not installed: {s}\n", .{ bytes >> 20, @errorName(e) });
}

/// The free buffers past `bytes` go, and no cap stays in force.
fn trimTo(bytes: u64) void {
    install(bytes);
    install(std.math.maxInt(u64));
}

/// A request on the model of `config` ended: its pass closes, and the free buffers past its resting cap go back. Every
/// Cooler arch's: the rest is what its ledger parks. An arch with its own caps keeps the rest as its cap in force
/// (DeepSeek lifts it around a restore); for the rest no cap stays.
pub fn endRequest(config: *const ModelConfig) void {
    closeRequest(config);
    rest(config);
}

/// The request's pass closes; a slot that frees buffers frees them after this and before `rest`.
pub fn closeRequest(config: *const ModelConfig) void {
    if (!onEngine(config)) return;
    @import("cooler_pipelines.zig").requestEnded();
    mlx.engine.close();
}

/// The free buffers past the resting cap go back, after everything the request held is freed.
pub fn rest(config: *const ModelConfig) void {
    if (!onEngine(config)) return;
    if (ownCaps(config)) install(restCap(config)) else trimTo(restCap(config));
}

/// Whether the arch installs its own forward caps, which a test's budget would not outlive.
pub fn ownsCaps(config: *const ModelConfig) bool {
    return ownCaps(config);
}

/// The model of `config` is unloaded: the pool's free buffers go, then any cap, since a cap is a forward's and the
/// next load makes tensors before any.
pub fn unloadModel(config: *const ModelConfig) void {
    if (onEngine(config)) @import("cooler_pipelines.zig").modelUnloaded();
    if (!onEngine(config) or ownCaps(config)) return;
    mlx.engine.close();
    trimTo(0);
}

/// A model went before its warm ended (`Trunk.warm_seal` still held): the seal its build lifted stands again.
pub fn warmAbandoned(seal: ?mlx.engine.Seal) void {
    if (seal) |s| s.end();
}

/// A load that plans makes its arena: on the ledger once its chunk is fixed, before its warm, so the warm runs what a
/// request runs (D107); else at its warm's end. An arch that declares `planBuild` records and packs its plans and keeps
/// the shapes its misses record in `cache` (its pipeline manifest's directory); any other serves on the pool.
pub fn planLoad(xfm: *Trunk, cache: ?@import("../tensor/plan/plan_book.zig").Cache) !void {
    // The warm's free buffers leave the pool before the arena is made, so the two never stand together.
    mlx.engine.close();
    mlx.engine.trim();
    switch (registry.handleOf(xfm)) {
        inline else => |m, t| if (comptime @hasDecl(registry.Mod(t), "planBuild")) try registry.Mod(t).planBuild(m, cache),
    }
}

/// A load on the ledger records its plans before its warm (D98), so its chunk is chosen on the arena they and the
/// region ask (`planArena`); an arch that declares no `planRecord` records none.
pub fn planRecord(xfm: *Trunk, cache: ?@import("../tensor/plan/plan_book.zig").Cache) !void {
    switch (registry.handleOf(xfm)) {
        inline else => |m, t| if (comptime @hasDecl(registry.Mod(t), "planRecord")) try registry.Mod(t).planRecord(m, cache),
    }
}

/// The arena the plans of `xfm` hold with prefill chunks of `chunk` rows, as Metal allocates it (whole pages); null for a
/// load that planned none.
pub fn planArena(xfm: *Trunk, chunk: u32) !?u64 {
    switch (registry.handleOf(xfm)) {
        inline else => |m, t| if (comptime @hasDecl(registry.Mod(t), "planArena")) {
            const bytes = (try registry.Mod(t).planArena(m, chunk)) orelse return null;
            return std.mem.alignForward(u64, bytes, @import("../tensor/pool_class.zig").page_bytes);
        },
    }
    return null;
}

/// A sample on a load whose arch plans its samples (docs/bill_redesign.md, D24): from `beginSample` to `end` (or `token`,
/// or `fail`) the sampler's work from the logits to the token is one execution of the root its mix names. An arch that
/// declares `sampleBegin` answers whether one began, and refuses by name a mix it did not plan (`PlanMiss`); every other
/// load samples as before.
pub const Sample = struct {
    xfm: *Trunk,
    on: bool,

    /// The execution ends: a miss it met is `PlanMiss`.
    pub fn end(run: Sample) !void {
        if (!run.on) return;
        switch (registry.handleOf(run.xfm)) {
            inline else => |m, t| if (comptime @hasDecl(registry.Mod(t), "sampleBegin")) return registry.Mod(t).sampleEnd(m),
        }
    }

    /// `lazy` (a draw's token, owned) as a token that outlives the execution, which ends: the arch's own block holds it.
    pub fn token(run: Sample, lazy: mlx.mlx_array) !mlx.mlx_array {
        if (!run.on) return lazy;
        switch (registry.handleOf(run.xfm)) {
            inline else => |m, t| if (comptime @hasDecl(registry.Mod(t), "sampleBegin")) return registry.Mod(t).sampleToken(m, lazy),
        }
        return lazy;
    }

    /// The sampler failed inside the execution, which goes: a miss it met is `PlanMiss`, any other failure the caller's.
    pub fn fail(run: Sample) error{PlanMiss}!void {
        if (!run.on) return;
        switch (registry.handleOf(run.xfm)) {
            inline else => |m, t| if (comptime @hasDecl(registry.Mod(t), "sampleBegin")) return registry.Mod(t).sampleFail(m),
        }
    }
};

/// A forward's input ids [1, n] where the arch of `xfm` holds them in a block of its own (`forwardIds`, a load on the
/// static bill); null for every other load, whose input the caller makes.
pub fn forwardIds(xfm: *Trunk, ids: []const i32) !?mlx.mlx_array {
    switch (registry.handleOf(xfm)) {
        inline else => |m, t| if (comptime @hasDecl(registry.Mod(t), "forwardIds")) return registry.Mod(t).forwardIds(m, ids),
    }
    return null;
}

/// A forced call's token `id` in the form the load's draws read a token in flight (`Sample.token`'s), or null where the
/// arch plans no samples and the token stays the host array of its id.
pub fn forcedToken(xfm: *Trunk, id: u32) !?mlx.mlx_array {
    switch (registry.handleOf(xfm)) {
        inline else => |m, t| if (comptime @hasDecl(registry.Mod(t), "sampleForced")) return registry.Mod(t).sampleForced(m, id),
    }
    return null;
}

pub fn beginSample(xfm: *Trunk, mix: @import("../sampler.zig").Mix) !Sample {
    switch (registry.handleOf(xfm)) {
        inline else => |m, t| if (comptime @hasDecl(registry.Mod(t), "sampleBegin")) return .{ .xfm = xfm, .on = try registry.Mod(t).sampleBegin(m, mix) },
    }
    return .{ .xfm = xfm, .on = false };
}

/// A warm of the Cooler load of `xfm` begins: it asks the pool by design until its seal (`warmEnded`), so the load's
/// gate opens, in force for the turn it runs in; a load's warm after its seal too.
pub fn warmBegins(xfm: *Trunk) void {
    xfm.cooler_budget.gate = .open;
    mlx.engine.gatePool(.open);
}

/// The Cooler load of `xfm` ended its warm: a pipeline made from here is late (`engine.seal`), and the warm's transients
/// leave the pool, so the load's memory reading is the model's.
pub fn warmEnded(xfm: *Trunk) void {
    xfm.warm_seal = null;
    mlx.engine.seal(true);
    mlx.engine.close();
    mlx.engine.trim();
    // Past the seal a load of an arch on the ledger closes the pool for its turns: an ask outside a wide phase is
    // refused by name (D64). Every other model's turns keep it open.
    xfm.cooler_budget.gate = if (plansForwards(&xfm.config)) .refuse else .open;
}

/// What a load on the engine of our own owes the engine while it reads its weights and builds its model, in one place
/// for every arch: the engine on before the first weight is wrapped, every region the load makes wrapped as the
/// engine's, no cap of an earlier model's forward in force from its first weight, and while the model builds the
/// engine's scope held (to the unload, `arch_handle.destroy`) and the seal of an earlier load lifted, standing again if
/// this one fails.
pub const LoadScope = struct {
    on: bool,
    held: bool = false,
    seal: ?mlx.engine.Seal = null,

    /// The weights of `config` are read from here to `end`, under no cap of an earlier model's forward.
    pub fn weights(config: *const ModelConfig) error{EngineBudget}!LoadScope {
        const on = cooler_side.onEngine(config);
        if (on) {
            mlx.engine.enable();
            mlx.engine.budget(std.math.maxInt(u64)) catch return error.EngineBudget;
        }
        mlx.engine.own_weights = on;
        return .{ .on = on };
    }

    /// The model of `config` is built from here to `end` (or `fail`): the hold it takes is the model's to its unload.
    pub fn build(config: *const ModelConfig) error{EngineBudget}!LoadScope {
        const on = cooler_side.onEngine(config);
        if (!on) return .{ .on = false };
        mlx.engine.hold();
        errdefer @import("handle.zig").engineHoldEnd();
        mlx.engine.budget(std.math.maxInt(u64)) catch return error.EngineBudget;
        mlx.engine.own_weights = true;
        return .{ .on = true, .held = true, .seal = mlx.engine.unsealed() };
    }

    /// The weights are read or the model is built: the model keeps its hold, and its seal (`Trunk.warm_seal`) stays
    /// lifted until its warm ends (`warmEnded`) or the model goes.
    pub fn end(_: LoadScope) void {
        mlx.engine.own_weights = false;
    }

    /// The build failed: the hold goes and the earlier seal stands again.
    pub fn fail(self: LoadScope) void {
        mlx.engine.own_weights = false;
        if (self.seal) |s| s.end();
        if (self.held) @import("handle.zig").engineHoldEnd();
    }
};

/// A step of a load that makes its own pipelines outside the model's build (an image tower's load or warm): the seal is
/// lifted for it and stands again at `end`. Null on MLX.
pub fn making(on: bool) ?mlx.engine.Seal {
    return if (on) mlx.engine.unsealed() else null;
}
