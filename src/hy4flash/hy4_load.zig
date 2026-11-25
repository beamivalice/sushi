//! Hy4-Flash's own load (the `load` slot of `arch.zig`): the release's tensors read by name, as stored, each placed by
//! the census (`hy4_census.roleOf`). A streamed load leaves the routed banks to the expert stream; the vision tower
//! loads when asked; the DFly stages wait for their drafter. A name the census does not know refuses the load.
const std = @import("std");
const model = @import("../model.zig");
const log = @import("../log.zig");
const census = @import("hy4_census.zig");
const weights_where = @import("../weights_where.zig");
const ModelConfig = model.ModelConfig;

const expert_io = @import("../expert_io.zig");

/// What a load keeps beside the trunk: the tower, and DFly2 when it drafts with it.
const Keep = struct { config: *const ModelConfig, vision: bool, dfly: bool, unknown: usize = 0 };

pub fn weights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool) !?model.Weights {
    var k: Keep = .{ .config = config, .vision = load_vision and config.has_vision, .dfly = @import("arch.zig").dsparkOn(config) };
    var w = try weights_where.loadWhere("hy4", io, allocator, model_dir, &k, keeps);
    errdefer w.deinit();
    if (k.unknown > 0) return error.Hy4TensorUnknown;
    return w;
}

/// The stored bytes `weights` holds: every tensor it keeps, as its shard headers size it, DFly2's when `mtp_on`.
pub fn residentLoadBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig, load_vision: bool, mtp_on: bool) !?u64 {
    var k: Keep = .{ .config = config, .vision = load_vision and config.has_vision, .dfly = mtp_on and config.hy4flash.?.dfly.layers > 0 };
    var shards = try weights_where.shardNames(io, allocator, model_dir);
    defer weights_where.freeShardNames(allocator, &shards);
    var total: u64 = 0;
    for (shards.items) |name| {
        const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, name }, 0);
        defer allocator.free(path);
        const fd = try expert_io.openHinted(path, .{ .nocache = true, .readahead_off = false });
        defer _ = std.c.close(fd);
        var header = try expert_io.SafetensorsHeader.read(allocator, fd);
        defer header.deinit();
        if (header.parsed.value != .object) return error.InvalidSafetensorsHeader;
        var it = header.parsed.value.object.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__") or !keeps(e.key_ptr.*, &k)) continue;
            if (e.value_ptr.* != .object) return error.InvalidSafetensorsHeader;
            // A bank the model decodes at load (`decodedAtLoad`) is held as its BF16 [out, in] rows, not as stored.
            if (decodedAtLoad(e.key_ptr.*)) |part| {
                if (part == .trellis) {
                    const sh = e.value_ptr.object.get("shape") orelse return error.InvalidSafetensorsHeader;
                    if (sh != .array or sh.array.items.len != 3 or sh.array.items[0] != .integer or sh.array.items[1] != .integer) return error.InvalidSafetensorsHeader;
                    const in = 16 * (std.math.cast(u64, sh.array.items[0].integer) orelse return error.InvalidSafetensorsHeader);
                    const out = 16 * (std.math.cast(u64, sh.array.items[1].integer) orelse return error.InvalidSafetensorsHeader);
                    total += in * out * 2;
                }
                continue;
            }
            const offs = e.value_ptr.object.get("data_offsets") orelse return error.InvalidSafetensorsHeader;
            if (offs != .array or offs.array.items.len != 2 or offs.array.items[0] != .integer or offs.array.items[1] != .integer) return error.InvalidSafetensorsHeader;
            const lo = std.math.cast(u64, offs.array.items[0].integer) orelse return error.InvalidSafetensorsHeader;
            const hi = std.math.cast(u64, offs.array.items[1].integer) orelse return error.InvalidSafetensorsHeader;
            if (hi < lo) return error.InvalidSafetensorsHeader;
            total += hi - lo;
        }
    }
    if (k.unknown > 0) return error.Hy4TensorUnknown;
    return total;
}

const BankPart = enum { trellis, scale };

const pad_prefix = @import("sushi_exl3").group_layout.pad_prefix;

/// An STQ bank the model decodes once at load and drops (`Model.denseOf`): absorbed MLA's `kv_b`, which the forward
/// slices by head (the embedding stays a bank, read by 128-id block). Null for every other tensor.
pub fn decodedAtLoad(key: []const u8) ?BankPart {
    const base = for ([_][]const u8{ ".trellis", ".suh", ".svh" }) |suffix| {
        if (std.mem.endsWith(u8, key, suffix)) break key[0 .. key.len - suffix.len];
    } else return null;
    if (!std.mem.endsWith(u8, base, ".self_attn.kv_b_proj")) return null;
    return if (std.mem.endsWith(u8, key, ".trellis")) .trellis else .scale;
}

fn keeps(key: []const u8, k: *Keep) bool {
    return keptWith(&k.config.hy4flash.?, k.config.expert_streaming, k.vision, k.dfly, key) catch {
        log.err("hy_v4_flash: the pack holds {s}, a tensor the release does not ship\n", .{key});
        k.unknown += 1;
        return false;
    };
}

/// Whether the pack ships DFly2's layers, which a load drafting with them reads.
pub fn shipsHead(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return false;
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(64 << 20)) catch return false;
    defer allocator.free(raw);
    return std.mem.indexOf(u8, raw, "\"model.dfly_layers.0.") != null;
}

/// What a streamed load of `config` keeps resident, for the preflight: the trunk as `weights` keeps it (the routed
/// banks and Engram's tables stay on SSD), DFly2 as the head (the planner drops it when the load does not draft), and the
/// tower as stored where the release ships one.
pub fn streamedSplit(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig) !model.ResidentSplit {
    var streamed = config.*;
    streamed.expert_streaming = true;
    const trunk = (try residentLoadBytes(io, allocator, model_dir, &streamed, false, false)).?;
    const with_tower = if (config.has_vision) (try residentLoadBytes(io, allocator, model_dir, &streamed, true, false)).? else trunk;
    const with_dfly = if (streamed.hy4flash.?.dfly.layers > 0) (try residentLoadBytes(io, allocator, model_dir, &streamed, false, true)).? else trunk;
    return .{ .trunk = trunk, .mtp = with_dfly - trunk, .vision = with_tower - trunk };
}

/// Whether a load keeps `key` resident. A streamed load reads the routed banks through the expert stream and Engram's
/// tables by row; the tower loads when asked; the DFly stages wait for their drafter.
pub fn kept(c: *const @import("hy4_config.zig").Config, streamed: bool, vision: bool, key: []const u8) !bool {
    return keptWith(c, streamed, vision, false, key);
}

/// `kept`, with DFly2's tensors kept when the load drafts with it (`dsparkOn`).
pub fn keptWith(c: *const @import("hy4_config.zig").Config, streamed: bool, vision: bool, dfly: bool, key: []const u8) !bool {
    // The zero tensors a repack lays between pair-major trellises (docs/pack-format.md#tile-order): never loaded or billed.
    if (std.mem.startsWith(u8, key, pad_prefix)) return false;
    return switch ((try census.roleOf(c, key)).role) {
        // A streamed load reads the routed banks through the expert stream; Engram's tables are read by row from the
        // shards on every load, never held in memory.
        .routed_experts => !streamed,
        .engram_table => false,
        .vision => vision,
        .dfly2 => dfly,
        else => true,
    };
}

/// DIAGNOSTIC (`SUSHI_HY4_DFLY_PROBE=<tokens>[,<every>]`, `SUSHI_HY4_DFLY_PROBE_TEXT=<abs path>`, read once at load):
/// DFly2's teacher-forced acceptance on a load that keeps its drafter (`hy4_dfly.probe`), logged as `[dfly-probe]`.
pub fn warmUbench(allocator: std.mem.Allocator, xfm: *@import("../trunk.zig").Trunk, kv: @import("../kv_quant.zig").KVQuantConfig, tok: *@import("../tokenizer.zig").Tokenizer) void {
    const raw = std.c.getenv("SUSHI_HY4_DFLY_PROBE") orelse return;
    dflyProbe(allocator, xfm, kv, std.mem.sliceTo(raw, 0), tok) catch |err| log.warn("[dfly-probe] failed: {s}\n", .{@errorName(err)});
}

pub fn ubenchMark(xfm: *@import("../trunk.zig").Trunk) !bool {
    const st = &xfm.arch_state.hy4flash;
    const m = st.model orelse return false;
    st.ubench_mark = try m.mark();
    return true;
}

pub fn ubenchRollback(xfm: *@import("../trunk.zig").Trunk, ctx: *@import("../trunk.zig").ForwardCtx) !void {
    const st = &xfm.arch_state.hy4flash;
    const m = st.model orelse return error.Hy4ModelMissing;
    const mk = &(st.ubench_mark orelse return error.Hy4PointMismatch);
    try m.rollbackTo(mk, &.{});
    ctx.cache.step = m.pos;
}

pub fn ubenchDropMark(xfm: *@import("../trunk.zig").Trunk) void {
    const st = &xfm.arch_state.hy4flash;
    if (st.ubench_mark) |*mk| {
        if (st.model) |m| m.dropMark(mk);
        st.ubench_mark = null;
    }
}

fn dflyProbe(gpa: std.mem.Allocator, xfm: *@import("../trunk.zig").Trunk, kv: @import("../kv_quant.zig").KVQuantConfig, spec: []const u8, tok: *@import("../tokenizer.zig").Tokenizer) !void {
    const dfly = @import("hy4_dfly.zig");
    const st = &xfm.arch_state.hy4flash;
    const m = st.model orelse return error.Hy4ModelMissing;
    const d = &(st.drafter orelse return error.Hy4DflyMissing);
    var fields = std.mem.splitScalar(u8, spec, ',');
    const tokens = try std.fmt.parseInt(usize, fields.first(), 10);
    const every = if (fields.next()) |f| try std.fmt.parseInt(usize, f, 10) else 1;
    if (@import("hy4_flags.zig").storesKv(kv)) m.kv = kv;
    m.dropLive();
    const io = std.Io.Threaded.global_single_threaded.io();
    const text: ?[]u8 = if (std.c.getenv("SUSHI_HY4_DFLY_PROBE_TEXT")) |path| try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(path), gpa, .limited(64 << 20)) else null;
    defer if (text) |t| gpa.free(t);
    const prompts: []const []const u8 = if (text) |t| &.{t} else &dfly.probe_prompts;
    const slots: usize = m.config.dfly.draft;
    var all: dfly.Tally = .{};
    for (prompts, 1..) |p, i| {
        const ids = try tok.encode(gpa, p);
        defer gpa.free(ids);
        var tally: dfly.Tally = .{};
        try dfly.probe(d, m, ids, tokens, every, &tally);
        var buf: [32]u8 = undefined;
        tally.log(try std.fmt.bufPrint(&buf, "prompt {d}", .{i}), slots);
        all.add(&tally);
    }
    all.log("all", slots);
}

test "hy4 load: the release's resident bytes, streamed and resident, from its 1,964 tensors" {
    const hy4_config = @import("hy4_config.zig");
    const c = try hy4_config.parseText(hy4_config.release_config);
    var sums: [2]u64 = .{ 0, 0 };
    var tower: u64 = 0;
    var lines = std.mem.tokenizeScalar(u8, @embedFile("fixtures/hy4_tensors.txt"), '\n');
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const name = f.next().?;
        const dt = f.next().?;
        var n: u64 = if (std.mem.eql(u8, dt, "BF16")) 2 else 4;
        var dims = std.mem.tokenizeScalar(u8, f.next().?, ',');
        while (dims.next()) |d| n *= try std.fmt.parseInt(u64, d, 10);
        for (&sums, [_]bool{ true, false }) |*sum, streamed| {
            if (try kept(&c, streamed, false, name)) sum.* += n;
        }
        if (try kept(&c, true, true, name) and !try kept(&c, true, false, name)) tower += n;
    }
    // Streamed: the census's total less the routed banks, Engram's tables, DFly2 and the tower.
    try std.testing.expectEqual(@as(u64, 14_016_920_660), sums[0]);
    // Resident experts add the routed banks; Engram's tables stay on disk either way.
    try std.testing.expectEqual(@as(u64, 311_174_970_452), sums[1]);
    // A load asked for the tower adds its 441 tensors, as stored.
    try std.testing.expectEqual(@as(u64, 933_258_720), tower);
}

test "hy4 load: a streamed load keeps images on by default, its tower resident beside the experts as the rest but Engram" {
    const model_load = @import("../model_load.zig");
    const config = ModelConfig{ .model_type = "hy_v4_flash", .has_vision = true };
    try std.testing.expect(model_load.visionDefaultOn(&config, true));
    try std.testing.expect(model_load.coldLoadVision(&config, true));
}

test "hy4 load: DFly2 drafts by default: a launch with no --mtp and no setting turns it on, --no-mtp turns it off" {
    const model_load = @import("../model_load.zig");
    const config = ModelConfig{ .model_type = "hy_v4_flash" };
    try std.testing.expect(model_load.mtpChoiceFor(null, &config).on);
    try std.testing.expect(!model_load.mtpChoiceFor(false, &config).on);
}
