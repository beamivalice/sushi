//! What every load on the engine of our own owes, whatever its arch, in one place:
//! - the first narrow forward after a wide one (decode after a prefill) trims the free buffers to its bill's transient,
//!   so a decode does not park a prefill's scratch; nothing is refused, since the shared bill is not a measured cap
//!   of the engine's own forward (buffers an open pass binds outlive their tensors);
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
    const margin: u64 = if (bills.coolerPrefillRowBytes(config) > 0) bills.BILL_MARGIN_BYTES else 0;
    return bills.prefillTransientReserveAtKv(config, kv_bits, @max(rows, verify_rows), seq) +| margin;
}

/// PURE: `forwardCap` of a forward that reads out every row through the head, which asks `readout` bytes
/// (`Trunk.lmHeadReadoutBytes`) past the last layer. A served prefill chunk reads out none; a tool's (`sushi kld`, the
/// decode meter) reads out its whole width.
pub fn readoutCap(config: *const ModelConfig, rows: u64, seq: u64, readout: u64) u64 {
    return forwardCap(config, rows, seq) +| bills.atTrunkRows(config, rows, readout);
}

/// PURE: the free buffers a request's end leaves the pool at most.
pub fn restCap(config: *const ModelConfig) u64 {
    const tag = registry.ofModelType(config.model_type) orelse return forwardCap(config, verify_rows, 1);
    switch (tag) {
        inline else => |t| if (comptime @hasDecl(registry.Mod(t), "coolerRestCap")) return registry.Mod(t).coolerRestCap(config),
    }
    return forwardCap(config, verify_rows, 1);
}

/// A forward wider than a verify block ran since the last trim: the next narrow one trims its scratch.
var after_wide = false;

/// `readout`: the head's bytes over a wide forward's rows when it reads them out (`readoutCap`), else 0.
pub const Rows = struct {
    rows: u64,
    seq: u64,
    readout: u64 = 0,

    pub fn cap(r: Rows, config: *const ModelConfig) u64 {
        return readoutCap(config, r.rows, r.seq, r.readout);
    }
};

/// A forward of the model of `config` begins: its rows, and the position its last row ends at, read only when the
/// shared seam acts on them or `want` asks (each read of a shape is a query of MLX's on an MLX load).
pub fn enterForward(config: *const ModelConfig, ctx: *const model_handle.ForwardCtx, ids: mlx.mlx_array, head: *const Trunk, want: bool) ?Rows {
    const seam = onEngine(config) and !ownCaps(config);
    if (!seam and !want) return null;
    const shape = mlx.getShape(ids);
    const rows: u64 = if (shape.len == 0) 1 else @intCast(shape[shape.len - 1]);
    var at: Rows = .{ .rows = rows, .seq = @as(u64, ctx.cache.kvLenForBatching()) + rows };
    if (!seam) return at;
    if (rows > verify_rows) {
        if (!ctx.skip_lm_head) at.readout = head.lmHeadReadoutBytes(rows);
        // A prefill chunk holds no more than its bill: under the cap the pool gives up its free buffers before it asks
        // Metal for more. Uncapped, a 4096-row chunk of the Qwen cut kept ~2 GB of them on 2.9 GB in use, 1.4 GB past
        // its bill. The next narrow forward, or the request's end, lifts it.
        after_wide = true;
        mlx.engine.close();
        install(at.cap(config));
        return at;
    }
    liftWide(config, rows, at.seq);
    return at;
}

/// A forward of several requests' rows begins (joined decode rows, a grouped verify): `rows` in all, the longest
/// ending at `seq`. It is a narrow forward whatever its rows: a wide forward's cap is lifted for it, none installed.
pub fn enterJoined(config: *const ModelConfig, rows: u64, seq: u64) void {
    if (!joinedLifts(config)) return;
    liftWide(config, rows, seq);
}

/// Whether a joined forward of `config`'s model would lift a cap now, so its rows are worth reading.
pub fn joinedLifts(config: *const ModelConfig) bool {
    return after_wide and onEngine(config) and !ownCaps(config);
}

/// After a wide forward, the free buffers past a narrow forward's cap go and no cap stays in force.
fn liftWide(config: *const ModelConfig, rows: u64, seq: u64) void {
    if (!after_wide) return;
    after_wide = false;
    mlx.engine.close();
    trimTo(forwardCap(config, rows, seq));
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
    if (!onEngine(config)) return;
    @import("cooler_pipelines.zig").requestEnded();
    mlx.engine.close();
    after_wide = false;
    if (ownCaps(config)) install(restCap(config)) else trimTo(restCap(config));
}

/// Whether the arch installs its own forward caps, which a test's budget would not outlive.
pub fn ownsCaps(config: *const ModelConfig) bool {
    return ownCaps(config);
}

/// The model of `config` is unloaded: the pool's free buffers go, then any cap, since a cap is a forward's and the
/// next load makes tensors before any.
pub fn unloadModel(config: *const ModelConfig) void {
    if (!onEngine(config) or ownCaps(config)) return;
    mlx.engine.close();
    trimTo(0);
}

/// A Cooler load's warm ended: a pipeline made from here is late (`engine.seal`), and the warm's transients leave the
/// pool, so the load's memory reading is the model's.
pub fn warmEnded() void {
    mlx.engine.seal(true);
    mlx.engine.close();
    mlx.engine.trim();
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

    /// The weights are read or the model is built: the model keeps its hold, and the seal stays lifted until its warm
    /// ends (`warmEnded`).
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
