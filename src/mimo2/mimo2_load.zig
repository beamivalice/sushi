//! MiMo's own load (`load` in `arch.zig`, reached through `arch/load.zig`): the source loader its FP8 trunk takes under
//! either routed-expert layout, what that load keeps resident, the MTP heads it allocates beside the trunk, and the
//! verify warmup.
const std = @import("std");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const model = @import("../model.zig");
const trunk = @import("../trunk.zig");
const bills = @import("../bills.zig");
const mtp_policy = @import("../mtp_policy.zig");
const mimo_source = @import("mimo_source.zig");
const mimo_mtp = @import("mimo_mtp.zig");
const mimo_forward = @import("mimo_forward.zig");
const Trunk = trunk.Trunk;
const ModelConfig = model.ModelConfig;
const Weights = model.Weights;

fn logSourceLoad(config: *const ModelConfig, vision: bool) void {
    log.info("[mimo-source] loading original shards: {s} experts, FP8 trunk in source bytes{s}\n", .{
        if (config.expert_streaming) "SSD-streamed" else if (config.expert_layout == .exl3_k4) "resident EXL3" else "native MXFP4",
        if (vision) ", bf16 vision tower" else "",
    });
}

/// MiMo's trunk is FP8 on disk under either routed-expert layout, so both take the source loader; an EXL3 pack's
/// routed banks come resident beside it.
pub fn individualExpertWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, vision: bool) !Weights {
    var config = try model.parseConfig(io, allocator, model_dir);
    defer config.deinit(allocator);
    logSourceLoad(&config, vision);
    var weights_out = try mimo_source.loadWeights(io, allocator, model_dir, &config);
    errdefer weights_out.deinit();
    if (vision) try mimo_source.loadVisionWeightsInto(&weights_out, io, allocator, model_dir);
    return weights_out;
}

/// A source trunk's weights; null for a pack the generic loaders read.
pub fn weights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool) !?Weights {
    if (!config.usesMimoSourceTrunk()) return null;
    const vision = load_vision and config.mimo_vision;
    if (!config.expert_streaming) return try individualExpertWeights(io, allocator, model_dir, vision);
    logSourceLoad(config, vision);
    var weights_out = try mimo_source.loadWeights(io, allocator, model_dir, config);
    errdefer weights_out.deinit();
    if (vision) try mimo_source.loadVisionWeightsInto(&weights_out, io, allocator, model_dir);
    return weights_out;
}

/// Resident bytes of a MiMo pack the source loader prepares: the trunk as served (FP8 codes + scale grids, bf16 rest)
/// plus any resident routed banks, plus the vision tower as stored when it is loaded.
pub fn sourceResidentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, vision: bool) !u64 {
    const trunk_bytes = try mimo_source.residentBytes(io, allocator, model_dir);
    return if (vision) trunk_bytes + try mimo_source.visionResidentBytes(io, allocator, model_dir) else trunk_bytes;
}

/// Resident bytes of a MiMo checkpoint's MTP heads as `mimo_source` uploads them.
pub fn headResidentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !u64 {
    return mimo_source.mtpResidentBytes(io, allocator, model_dir);
}

pub fn shipsHead(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    return (headResidentBytes(io, allocator, model_dir) catch 0) > 0;
}

/// The resident bytes a MiMo source-trunk load holds: trunk and vision tower as stored, the heads when MTP is on, and
/// the coarse lm_head copy the heads and the greedy readout share. Null for a pack the indexed scan bills.
pub fn residentLoadBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool, mtp_on: bool) !?u64 {
    if (!config.usesMimoSourceTrunk()) return null;
    var bytes = try sourceResidentBytes(io, allocator, model_dir, load_vision and config.mimo_vision);
    if (mtp_on) bytes += try headResidentBytes(io, allocator, model_dir);
    return bytes + bills.coarseHeadBytes(config);
}

/// MiMo's MTP heads from the checkpoint's own shard, bound to the trunk they share.
fn loadHeads(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, xfm: *Trunk) !?*mimo_mtp.Head {
    var head_weights = try mimo_source.loadMtpWeights(io, allocator, model_dir);
    defer head_weights.deinit();
    var head = (try mimo_mtp.Head.load(allocator, mlx.gpuStream(), config, &head_weights)) orelse return null;
    errdefer head.deinit();
    head.target = xfm;
    const ptr = try allocator.create(mimo_mtp.Head);
    ptr.* = head;
    // The coarse lm_head copy is a load cost, never the first draft's.
    const rerank = ptr.canRerankDrafts();
    log.info("[mimo-mtp] {d} heads loaded ({d:.2} GB resident); draft rerank {s}\n", .{ head.heads, @as(f64, @floatFromInt(head.residentBytes())) / 1e9, if (rerank) "on" else "off" });
    return ptr;
}

/// The trained heads, resident beside a resident trunk (streaming refuses MTP before the load).
pub fn loadHead(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, xfm: *Trunk, mtp_enabled: bool) ?*anyopaque {
    if (!mtp_enabled or config.expert_streaming) return null;
    const head = loadHeads(io, allocator, model_dir, config, xfm) catch |err| {
        log.warn("[mimo-mtp] heads not loaded ({s}) — MTP off\n", .{@errorName(err)});
        return null;
    };
    return head;
}

fn headOf(p: *anyopaque) *mimo_mtp.Head {
    return @ptrCast(@alignCast(p));
}

pub fn freeHead(allocator: std.mem.Allocator, p: *anyopaque) void {
    const h = headOf(p);
    h.deinit();
    allocator.destroy(h);
}

/// The head count and the widest verify whose rows stay byte-identical to serial decode: a deeper round would hand the
/// forward more rows than the decode-shaped verify serves.
pub fn headDepthCap(config: *const ModelConfig, p: *anyopaque) u32 {
    return @min(@as(u32, @intCast(headOf(p).heads)), mtp_policy.mtpVerifyDraftsMax(config));
}

/// The heads built the coarse copy for their drafts; without them the trunk's greedy readout still takes one, under the
/// predicate of the resident bill that prices it.
pub fn prepare(config: *const ModelConfig, xfm: *Trunk) void {
    if (bills.coarseHeadBytes(config) > 0) _ = mimo_mtp.buildCoarseHead(mlx.gpuStream(), xfm);
}

/// A resident MiMo warms its verify rows with or without its heads: a PLD request verifies on them too.
pub fn verifyWarmWanted(config: *const ModelConfig) bool {
    return config.isMimo() and !config.expert_streaming;
}

pub fn warmSpec(io: std.Io, config: *const ModelConfig, xfm: *Trunk, head: ?*anyopaque, kv_config: trunk.KVQuantConfig) void {
    if (!verifyWarmWanted(config)) return;
    const start = std.Io.Timestamp.now(io, .awake);
    const rows = mimo_forward.warmupMimoVerify(xfm, kv_config) catch |err| {
        log.warn("[spec-warmup] MiMo verify failed ({s}); the first round at each width pays its kernel compile inside the round.\n", .{@errorName(err)});
        return;
    };
    const steps = if (head) |p| headOf(p).warmup(xfm) catch |err| {
        log.warn("[spec-warmup] MiMo heads failed ({s}); the first round pays their kernel compile inside the round.\n", .{@errorName(err)});
        return;
    } else 0;
    const ms: u64 = @as(u64, @intCast(start.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_ms;
    log.info("[spec-warmup] MiMo verify rows 0x{x}, {d} head steps ({d} ms).\n", .{ rows, steps, ms });
}

test "a resident MiMo cold load reserves its load preflight's requirement" {
    const model_load = @import("../model_load.zig");
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var fixture = try mimo_source.makeTinySourceFixture(io, a, &tmp);
    defer fixture.deinit();
    fixture.config.expert_layout = .mxfp4_individual;
    const loaded = (try residentLoadBytes(io, a, fixture.path, &fixture.config, false, false)).?;
    const preflight = model_load.loadRequirementBytes(&fixture.config, loaded, false, 0);
    try std.testing.expectEqual(preflight, try model_load.residentColdLoadBillBytes(io, a, &fixture.config, fixture.path, false, false, false, ""));
    try std.testing.expect(loaded > 0);
}
