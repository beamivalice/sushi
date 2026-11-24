const std = @import("std");
const mlx = @import("mlx_host").mlx;
const api = @import("sushi_exl3");
const format = api.format;
const convert = @import("convert.zig");
const pack = @import("pack.zig");
const shard = @import("shard.zig");
const source = @import("source.zig");
const st = @import("safetensors.zig");
const stages_mod = @import("stages.zig");
const gpu_factor = @import("gpu_factor.zig");
const encode = @import("encode.zig");
const rate_plan = @import("rate_plan.zig");
const trunk_from = @import("trunk_from.zig");
const fisher = @import("fisher.zig");
const trunk_mod = @import("trunk.zig");
const layer_stage = @import("layer_stage.zig");
const gptq = @import("gptq.zig");

pub const Error = error{
    OutputNotEmpty,
    ResumeNothingToResume,
    ResumePolicyMismatch,
    ResumePackComplete,
    ArchitectureNotConvertible,
    SourceConfigInvalid,
    SourceHasNoExperts,
    RateUnsupported,
    PinPassUnsupported,
    TrunkFromMissing,
    TrunkModeConflict,
    TrunkNeedsCalibration,
    TrunkHeadMissing,
    TrunkFromRateGrouped,
    TrunkWidthConflict,
    TrunkBitsUnsupported,
    FisherFileMissing,
    TargetBoundaryMissing,
} || std.mem.Allocator.Error;

pub const Info = struct {
    pack_dir: std.Io.Dir,
    pack_abs: []const u8,
    source_dir_abs: []const u8,
    source_config: []const u8,
    geom: convert.Geometry,
    layers: u32,
    layer_full: []const bool,
    ple_layer: u32,
    rate: format.Rate,
    dec: format.Decode,
    trunk_bits: u32,
    trunk_group: u32,
    overrides: []const pack.Override,
    /// The pack's n-gram table spec; null when it has none.
    ngram: ?pack.NgramSpec,
    eq: pack.ExpertQuantSpec,
    /// Per layer, the source expert id of each stored router row; null where the rows keep source order.
    perms: []const ?[]const u32 = &.{},
    /// A directory of the BF16 path's stored layer inputs (`boundary-LL.u16`) over the same calibration: the
    /// routed experts then round the compensated target.
    target_boundaries: ?[]const u8 = null,
    /// The compensated target with the BF16 path carried by the run itself, one boundary at a time beside the
    /// quantized one, instead of read from `target_boundaries`.
    target_in_run: bool = false,
    /// Re-round every trunk linear by GPTQ on the quantized path's Grams (`trunk.zig`).
    trunk_gptq: bool = false,
    /// Round the trunk's linears onto the trellis on the same Grams (`trunk_stq.zig`).
    trunk_stq: ?trunk_mod.Stq = null,
};

/// How the trunk is made: re-rounded by GPTQ along the quantized path over a base trunk (the `--trunk-from` pack's
/// layout, else round-to-nearest at the trunk bits), taken as stored from `--trunk-from`, or round-to-nearest.
/// `stq` rounds the trunk of the `--trunk-from` pack onto the trellis and keeps its experts (`trunk_stq.zig`).
pub const TrunkMode = enum { gptq, stored, rtn, stq };

pub const Calibrator = struct {
    ctx: *anyopaque,
    prepare: *const fn (ctx: *anyopaque, info: Info) anyerror!void,
    ready: *const fn (ctx: *anyopaque, layer: u32, last: bool) bool,
    begin: *const fn (ctx: *anyopaque, layer: u32, mtp: bool) anyerror!?convert.HessianSource,
    done: *const fn (ctx: *anyopaque, layer: u32, mtp: bool, reused: bool) anyerror!void,
    finish: *const fn (ctx: *anyopaque) void,
};

pub const Sink = struct {
    ctx: ?*anyopaque = null,
    write: *const fn (ctx: ?*anyopaque, line: []const u8) void,
};

pub const Options = struct {
    model_dir: []const u8,
    out_dir: []const u8,
    rate: format.Rate,
    dec: format.Decode,
    trunk_bits: ?u32 = null,
    trunk_group: ?u32 = null,
    calib_sha256: []const u8 = "none",
    calib_tokens: usize = 0,
    resuming: bool = false,
    version: []const u8,
    min_rows: u64 = 64,
    batch: usize = 16,
    factor_precision: gpu_factor.Precision = .f64,
    pin_pass: u32 = encode.default_pin_pass,
    plan: ?*const rate_plan.Plan = null,
    /// A pack of the same model whose non-expert tensors are taken as stored.
    trunk_from: ?[]const u8 = null,
    /// A directory of per-layer output Fishers (`fisher-LL.npz`); routed experts then take Fisher-weighted output
    /// scales.
    fisher_dir: ?[]const u8 = null,
    /// See `Info.target_boundaries` and `Info.target_in_run`; at most one is set.
    target_boundaries: ?[]const u8 = null,
    target_in_run: bool = false,
    trunk: TrunkMode = .gptq,
    /// Seconds of idle after each written layer; not part of the resume policy key.
    layer_cooldown: u32 = 0,
};

pub const Summary = struct {
    layers_written: usize = 0,
    layers_reused: usize = 0,
    mtp_written: bool = false,
    gate_up_identity: usize = 0,
    down_identity: usize = 0,
};

const progress_name = "convert-progress.json";

const OverrideJson = struct { base: []const u8, bits: u32, group: u32 };
const Progress = struct {
    policy: []const u8,
    trunk_done: bool = false,
    ngram_done: bool = false,
    ngram_present: bool = false,
    overrides: []const OverrideJson = &.{},
};

fn say(sink: ?Sink, comptime fmt: []const u8, args: anytype) void {
    const s = sink orelse return;
    var buf: [512]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, fmt, args) catch return;
    s.write(s.ctx, line);
}

fn sleepSeconds(seconds: u32) void {
    var req: std.c.timespec = .{ .sec = seconds, .nsec = 0 };
    var left: std.c.timespec = undefined;
    while (std.c.nanosleep(&req, &left) != 0) req = left;
}

fn rest(sink: ?Sink, layer: u32, seconds: u32, sleep: *const fn (u32) void) void {
    if (seconds == 0) return;
    say(sink, "layer {d}: cooldown {d}s", .{ layer, seconds });
    sleep(seconds);
}

fn sayFactor(sink: ?Sink, layer: u32, precision: gpu_factor.Precision, stats: *const convert.LayerStats) void {
    if (precision != .f32) return;
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    say(sink, "layer {d}: factor f32: gate/up {d:.1}s, down {d:.1}s, {d} retries, {d} float64 fallbacks, GPU peak {d:.2} GB", .{
        layer,
        @as(f64, @floatFromInt(stats.factor_gate_up_ns)) / 1e9,
        @as(f64, @floatFromInt(stats.factor_down_ns)) / 1e9,
        stats.factor_retries,
        stats.factor_fallbacks,
        @as(f64, @floatFromInt(peak)) / 1e9,
    });
}

fn sayStages(sink: ?Sink, label: []const u8, times: *const stages_mod.Stages) void {
    const s = sink orelse return;
    var buf: [1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    w.print("{s}: stage times ", .{label}) catch return;
    times.write(&w) catch return;
    s.write(s.ctx, w.buffered());
}

/// The float64 factor and the full pin pass add nothing to the key, so a conversion begun before either choice
/// existed still resumes.
fn policyKey(allocator: std.mem.Allocator, opts: Options, bits: u32, group: u32) ![]u8 {
    const factor = if (opts.factor_precision == .f32) "|factor=32" else "";
    const mode = if (opts.trunk == .gptq) "|trunk_mode=gptq" else "";
    var pin_buf: [16]u8 = undefined;
    const pin = if (opts.pin_pass == encode.full_pin_pass) "" else try std.fmt.bufPrint(&pin_buf, "|pin={d}", .{opts.pin_pass});
    const base = try std.fmt.allocPrint(allocator, "{s}|n={d}|{s}|w={d}|trunk={d}/{d}|calib={s}:{d}|min_rows={d}{s}{s}{s}", .{
        opts.version, opts.rate.n, @tagName(opts.dec.codebook), opts.dec.window.bits(), bits, group, opts.calib_sha256, opts.calib_tokens, opts.min_rows, factor, mode, pin,
    });
    if (opts.dec.table == null and opts.plan == null and opts.trunk_from == null and opts.fisher_dir == null and opts.target_boundaries == null and !opts.target_in_run) return base;
    defer allocator.free(base);
    const table = if (opts.dec.table) |i| api.codebook_tables.tables[i].name else "";
    const plan: []const u8 = if (opts.plan) |p| &p.sha256 else "";
    return std.fmt.allocPrint(allocator, "{s}|table={s}|plan={s}|trunk_from={s}|fisher={s}|target={s}", .{ base, table, plan, opts.trunk_from orelse "", opts.fisher_dir orelse "", if (opts.target_in_run) "in-run" else opts.target_boundaries orelse "" });
}

pub const Geometry = struct { layers: u32, experts: usize, hidden: usize, inter: usize, mtp: bool, ple_layer: u32 = 0 };

fn textConfigInt(obj: std.json.ObjectMap, key: []const u8) !usize {
    const v = obj.get(key) orelse return error.SourceConfigInvalid;
    if (v != .integer or v.integer <= 0) return error.SourceConfigInvalid;
    return @intCast(v.integer);
}

/// The routed experts per layer and the layer count a source config states, for reading a rate plan.
pub fn planShape(allocator: std.mem.Allocator, config_json: []const u8) !struct { experts: usize, layers: u32 } {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, config_json, .{}) catch return error.SourceConfigInvalid;
    defer parsed.deinit();
    if (parsed.value != .object) return error.SourceConfigInvalid;
    const text = parsed.value.object.get("text_config") orelse return error.SourceConfigInvalid;
    if (text != .object) return error.SourceConfigInvalid;
    // DeepSeek-V4.1 names its routed experts `n_routed_experts`.
    const experts = textConfigInt(text.object, "num_experts") catch try textConfigInt(text.object, "n_routed_experts");
    return .{ .experts = experts, .layers = @intCast(try textConfigInt(text.object, "num_hidden_layers")) };
}

fn pleLayer(text: std.json.ObjectMap) u32 {
    const v = text.get("ple_layer_ids") orelse return 0;
    if (v != .array or v.array.items.len == 0 or v.array.items[0] != .integer or v.array.items[0].integer < 1) return 0;
    return @intCast(v.array.items[0].integer - 1);
}

pub fn layerKinds(allocator: std.mem.Allocator, config_json: []const u8, layers: u32) ![]bool {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, config_json, .{});
    defer parsed.deinit();
    const out = try allocator.alloc(bool, layers);
    errdefer allocator.free(out);
    @memset(out, false);
    const types = parsed.value.object.get("text_config").?.object.get("layer_types") orelse return out;
    if (types != .array or types.array.items.len < layers) return error.SourceConfigInvalid;
    for (out, types.array.items[0..layers]) |*o, k| o.* = k == .string and std.mem.eql(u8, k.string, "full_attention");
    return out;
}

pub fn readGeometry(allocator: std.mem.Allocator, config_json: []const u8, src: *const source.SafetensorsSource) !Geometry {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, config_json, .{}) catch return error.SourceConfigInvalid;
    defer parsed.deinit();
    if (parsed.value != .object) return error.SourceConfigInvalid;
    const root = parsed.value.object;
    const text_v = root.get("text_config") orelse return error.SourceConfigInvalid;
    if (text_v != .object) return error.SourceConfigInvalid;
    const text = text_v.object;
    const mtp_layers: usize = if (text.get("mtp_num_hidden_layers")) |v| (if (v == .integer and v.integer > 0) @intCast(v.integer) else 0) else 0;
    var key: [128]u8 = undefined;
    const has_mtp = mtp_layers > 0 and src.tensors.contains(try std.fmt.bufPrint(&key, "mtp.layers.0.mlp.experts.gate_up_proj", .{}));
    return .{
        .layers = @intCast(try textConfigInt(text, "num_hidden_layers")),
        .experts = try textConfigInt(text, "num_experts"),
        .hidden = try textConfigInt(text, "hidden_size"),
        .inter = try textConfigInt(text, "moe_intermediate_size"),
        .mtp = has_mtp,
        .ple_layer = pleLayer(text),
    };
}

fn layerShardComplete(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8, prefix: []const u8, layer: u32) bool {
    var f = dir.openFile(io, name, .{}) catch return false;
    defer f.close(io);
    const size = (f.stat(io) catch return false).size;
    var len_buf: [8]u8 = undefined;
    if ((f.readPositionalAll(io, &len_buf, 0) catch return false) != 8) return false;
    const header_len = std.mem.readInt(u64, &len_buf, .little);
    if (header_len > size - 8) return false;
    const header = allocator.alloc(u8, header_len) catch return false;
    defer allocator.free(header);
    if ((f.readPositionalAll(io, header, 8) catch return false) != header_len) return false;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, header, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    var key: [160]u8 = undefined;
    const want = std.fmt.bufPrint(&key, "{s}.layers.{d}.mlp.switch_mlp.", .{ prefix, layer }) catch return false;
    var end: u64 = 0;
    var banks: usize = 0;
    var it = parsed.value.object.iterator();
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
        if (!std.mem.startsWith(u8, e.key_ptr.*, want) or e.value_ptr.* != .object) return false;
        const offsets = e.value_ptr.object.get("data_offsets") orelse return false;
        if (offsets != .array or offsets.array.items.len != 2 or offsets.array.items[1] != .integer or offsets.array.items[1].integer < 0) return false;
        end = @max(end, @as(u64, @intCast(offsets.array.items[1].integer)));
        banks += 1;
    }
    return banks != 0 and banks % 9 == 0 and end + 8 + header_len == size;
}

/// The file of the BF16 path's input to layer `layer` in a `--target-boundaries` directory.
pub fn storedBoundaryName(buf: []u8, layer: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "boundary-{d:0>2}.u16", .{layer});
}

fn gptqRounds(bits: u32, overrides: []const pack.Override) bool {
    if (!gptq.bitsRounded(bits)) return false;
    for (overrides) |o| if (!gptq.bitsRounded(o.bits)) return false;
    return true;
}

/// Names the first per-layer input file a run reads that is not there, before the run writes anything. A layer a
/// resumed run keeps (its shard is complete in `out`) reads none.
fn checkLayerInputs(allocator: std.mem.Allocator, io: std.Io, opts: Options, src: *const source.SafetensorsSource, layers: u32, out: ?std.Io.Dir, sink: ?Sink) !void {
    if (opts.fisher_dir == null and opts.target_boundaries == null) return;
    const cwd = std.Io.Dir.cwd();
    var layer: u32 = 0;
    while (layer < layers) : (layer += 1) {
        var kb: [128]u8 = undefined;
        const probe = try std.fmt.bufPrint(&kb, "model.language_model.layers.{d}.mlp.experts.gate_up_proj", .{layer});
        if (!src.tensors.contains(probe)) continue;
        if (out) |dir| {
            if (layerKept(allocator, io, dir, opts.trunk, layers, layer)) continue;
        }
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        if (opts.fisher_dir) |dir| {
            const path = try fisher.layerPath(&path_buf, dir, layer);
            cwd.access(io, path, .{}) catch {
                say(sink, "--fisher-dir: {s} is missing", .{path});
                return error.FisherFileMissing;
            };
        }
        if (opts.target_boundaries) |dir| {
            var name_buf: [48]u8 = undefined;
            const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, try storedBoundaryName(&name_buf, layer) });
            cwd.access(io, path, .{}) catch {
                say(sink, "--target-boundaries: {s} is missing", .{path});
                return error.TargetBoundaryMissing;
            };
        }
    }
}

fn trunkWritten(io: std.Io, dir: std.Io.Dir, layer: u32) bool {
    var buf: [64]u8 = undefined;
    _ = dir.statFile(io, trunk_mod.shardName(&buf, layer) catch return false, .{}) catch return false;
    return true;
}

fn headWritten(io: std.Io, dir: std.Io.Dir) bool {
    _ = dir.statFile(io, trunk_mod.head_shard, .{}) catch return false;
    return true;
}

/// A layer a resume keeps. The re-rounded trunk adds its layer shard and, for the last layer, the head, which only
/// that layer's advance writes.
fn layerKept(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, trunk: TrunkMode, layers: u32, layer: u32) bool {
    var name_buf: [64]u8 = undefined;
    const name = shard.shardFileName(&name_buf, layer) catch return false;
    if (!layerShardComplete(allocator, io, dir, name, shard.trunk_prefix, layer)) return false;
    if (trunk != .gptq) return true;
    return trunkWritten(io, dir, layer) and (layer + 1 < layers or headWritten(io, dir));
}

/// Rewrites every shard that holds a tensor a re-rounded shard (`trunk-*`) also holds, without it.
fn dropRerounded(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !usize {
    const files = try safetensorsFiles(allocator, io, dir);
    defer {
        for (files) |f| allocator.free(f);
        allocator.free(files);
    }
    var replaced = std.StringHashMap(void).init(allocator);
    defer {
        var it = replaced.keyIterator();
        while (it.next()) |k| allocator.free(k.*);
        replaced.deinit();
    }
    for (files) |f| {
        if (!layer_stage.isRerounded(f)) continue;
        var h = try trunk_from.readHeader(allocator, io, dir, f);
        defer h.parsed.deinit();
        var it = h.root.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
            const g = try replaced.getOrPut(e.key_ptr.*);
            if (!g.found_existing) g.key_ptr.* = try allocator.dupe(u8, e.key_ptr.*);
        }
    }
    var rewritten: usize = 0;
    for (files) |f| {
        if (layer_stage.isRerounded(f)) continue;
        const dropped = try trunk_from.rewriteWithout(allocator, io, dir, f, &replaced, struct {
            fn drop(set: *const std.StringHashMap(void), name: []const u8) bool {
                return set.contains(name);
            }
        }.drop);
        if (dropped > 0) rewritten += 1;
    }
    return rewritten;
}

fn writeProgress(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, p: Progress) !void {
    const text = try std.json.Stringify.valueAlloc(allocator, p, .{});
    defer allocator.free(text);
    try dir.writeFile(io, .{ .sub_path = ".convert-progress.tmp", .data = text });
    try dir.rename(".convert-progress.tmp", dir, progress_name, io);
}

fn dirIsEmpty(io: std.Io, dir: std.Io.Dir) !bool {
    var it = dir.iterate();
    return (try it.next(io)) == null;
}

fn safetensorsFiles(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) ![][]u8 {
    var names = std.ArrayList([]u8).empty;
    errdefer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or entry.name[0] == '.' or !std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        try names.ensureUnusedCapacity(allocator, 1);
        names.appendAssumeCapacity(try allocator.dupe(u8, entry.name));
    }
    std.mem.sortUnstable([]u8, names.items, {}, struct {
        fn lt(_: void, a: []u8, b: []u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    return names.toOwnedSlice(allocator);
}

pub fn indexOfDir(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) ![]u8 {
    const files = try safetensorsFiles(allocator, io, dir);
    defer {
        for (files) |f| allocator.free(f);
        allocator.free(files);
    }
    var entries = std.ArrayList(pack.IndexEntry).empty;
    defer {
        for (entries.items) |e| {
            allocator.free(e.tensor);
            allocator.free(e.file);
        }
        entries.deinit(allocator);
    }
    var total: u64 = 0;
    for (files) |name| {
        var f = try dir.openFile(io, name, .{});
        defer f.close(io);
        var len_buf: [8]u8 = undefined;
        if (try f.readPositionalAll(io, &len_buf, 0) != 8) return error.TruncatedSafetensors;
        const header_len = std.mem.readInt(u64, &len_buf, .little);
        const stat = try f.stat(io);
        total += stat.size - 8 - header_len;
        const header = try allocator.alloc(u8, header_len);
        defer allocator.free(header);
        if (try f.readPositionalAll(io, header, 8) != header_len) return error.TruncatedSafetensors;
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, header, .{});
        defer parsed.deinit();
        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
            const t = try allocator.dupe(u8, e.key_ptr.*);
            errdefer allocator.free(t);
            const fl = try allocator.dupe(u8, name);
            errdefer allocator.free(fl);
            try entries.append(allocator, .{ .tensor = t, .file = fl });
        }
    }
    return pack.buildIndex(allocator, entries.items, total);
}

pub fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    s: mlx.mlx_stream,
    opts: Options,
    calibrator: ?Calibrator,
    sink: ?Sink,
) !Summary {
    const cwd = std.Io.Dir.cwd();
    var src_dir = try cwd.openDir(io, opts.model_dir, .{ .iterate = true });
    defer src_dir.close(io);
    const arch = try source.identifySourceDir(allocator, io, src_dir);
    if (arch != .qwen4_exp) return error.ArchitectureNotConvertible;
    switch (opts.trunk) {
        .stored => if (opts.trunk_from == null) return error.TrunkFromMissing,
        .rtn => if (opts.trunk_from != null) return error.TrunkModeConflict,
        .gptq => if (calibrator == null) return error.TrunkNeedsCalibration,
        .stq => return error.TrunkModeConflict,
    }
    if (opts.trunk_from != null and (opts.trunk_bits != null or opts.trunk_group != null)) return error.TrunkWidthConflict;
    if (opts.target_in_run and opts.target_boundaries != null) return error.TargetConflict;
    if (opts.target_in_run and calibrator == null) return error.TargetNeedsCalibration;
    var trunk_bits = opts.trunk_bits orelse arch.defaultTrunkBits();
    var trunk_group = opts.trunk_group orelse arch.defaultTrunkGroup();
    if (opts.trunk == .gptq and opts.trunk_from == null and !gptqRounds(trunk_bits, &.{})) return error.TrunkBitsUnsupported;

    var src = try source.SafetensorsSource.open(allocator, io, src_dir);
    defer src.deinit();
    const config_json = try src_dir.readFileAlloc(io, "config.json", allocator, .limited(16 << 20));
    defer allocator.free(config_json);
    const geom = try readGeometry(allocator, config_json, &src);

    var ceiling = opts.rate;
    if (!encode.runnableRate(opts.rate) or !encode.searchable(opts.dec, opts.rate)) return error.RateUnsupported;
    if (!encode.validPinPass(opts.pin_pass)) return error.PinPassUnsupported;
    const perms = try allocator.alloc(?[]const u32, geom.layers);
    defer allocator.free(perms);
    @memset(perms, null);
    defer for (perms) |p| if (p) |v| allocator.free(v);
    if (opts.plan) |plan| {
        if (plan.layers.len != geom.layers) return error.RateUnsupported;
        for (0..geom.layers) |l| {
            const classes = plan.classes(@intCast(l)) orelse continue;
            for (classes) |c| for ([2]format.Rate{ c.gate_up, c.down }) |r| {
                if (!encode.runnableRate(r) or !encode.searchable(opts.dec, r)) return error.RateUnsupported;
            };
            perms[l] = try rate_plan.permutation(allocator, classes);
        }
        ceiling.n = @max(ceiling.n, plan.widest.n);
    }

    var served: ?trunk_from.Served = null;
    defer if (served) |*v| v.deinit();
    if (opts.trunk_from) |path| {
        var from = try cwd.openDir(io, path, .{ .iterate = true });
        defer from.close(io);
        served = try trunk_from.read(allocator, io, from);
        try trunk_from.refuseRateGrouped(allocator, io, from, served.?.config);
        if (opts.trunk == .gptq and !gptqRounds(served.?.bits, served.?.overrides)) return error.TrunkBitsUnsupported;
    }
    {
        var existing = if (opts.resuming) cwd.openDir(io, opts.out_dir, .{ .iterate = true }) catch null else null;
        defer if (existing) |*d| d.close(io);
        try checkLayerInputs(allocator, io, opts, &src, geom.layers, existing, sink);
    }

    try cwd.createDirPath(io, opts.out_dir);
    var out = try cwd.openDir(io, opts.out_dir, .{ .iterate = true });
    defer out.close(io);
    const fresh = try dirIsEmpty(io, out);
    if (!fresh and !opts.resuming) return error.OutputNotEmpty;

    const key = try policyKey(allocator, opts, trunk_bits, trunk_group);
    defer allocator.free(key);
    var progress = Progress{ .policy = key };
    var stored_overrides: ?std.json.Parsed(Progress) = null;
    defer if (stored_overrides) |p| p.deinit();
    if (!fresh) {
        if (out.access(io, "config.json", .{})) |_| return error.ResumePackComplete else |_| {}
        const raw = out.readFileAlloc(io, progress_name, allocator, .limited(16 << 20)) catch return error.ResumeNothingToResume;
        defer allocator.free(raw);
        const parsed = std.json.parseFromSlice(Progress, allocator, raw, .{ .allocate = .alloc_always }) catch return error.ResumeNothingToResume;
        errdefer parsed.deinit();
        if (!std.mem.eql(u8, parsed.value.policy, key)) return error.ResumePolicyMismatch;
        stored_overrides = parsed;
        progress = parsed.value;
        progress.policy = key;
    } else {
        try writeProgress(allocator, io, out, progress);
    }

    var summary = Summary{};
    var overrides = std.ArrayList(pack.Override).empty;
    defer {
        for (overrides.items) |o| allocator.free(o.base);
        overrides.deinit(allocator);
    }
    var ngram_spec: ?pack.NgramSpec = null;

    if (opts.trunk_from) |path| {
        var from = try cwd.openDir(io, path, .{ .iterate = true });
        defer from.close(io);
        const sv = &served.?;
        if (!progress.trunk_done) {
            say(sink, "trunk: copying the non-expert shards of {s}", .{path});
            const copied = try trunk_from.copy(allocator, io, from, out, perms, sv.ngram != null, opts.trunk == .gptq);
            say(sink, "trunk: {d} shards copied", .{copied});
            progress.trunk_done = true;
            progress.ngram_done = true;
            progress.ngram_present = sv.ngram != null;
            try writeProgress(allocator, io, out, progress);
        }
        trunk_bits = sv.bits;
        trunk_group = sv.group;
        for (sv.overrides) |o| {
            const base = try allocator.dupe(u8, o.base);
            errdefer allocator.free(base);
            try overrides.append(allocator, .{ .base = base, .bits = o.bits, .group = o.group });
        }
    } else if (!progress.trunk_done) {
        say(sink, "trunk: writing affine {d}-bit shards", .{trunk_bits});
        var trunk = try pack.writeQwenTrunk(allocator, io, out, &src, trunk_bits, trunk_group, perms);
        defer trunk.deinit();
        const stored = try allocator.alloc(OverrideJson, trunk.overrides.items.len);
        defer allocator.free(stored);
        for (trunk.overrides.items, stored) |o, *dst| dst.* = .{ .base = o.base, .bits = o.bits, .group = o.group };
        progress.overrides = stored;
        progress.trunk_done = true;
        try writeProgress(allocator, io, out, progress);
        for (trunk.overrides.items) |o| {
            try overrides.ensureUnusedCapacity(allocator, 1);
            overrides.appendAssumeCapacity(.{ .base = try allocator.dupe(u8, o.base), .bits = o.bits, .group = o.group });
        }
    } else {
        for (progress.overrides) |o| {
            try overrides.ensureUnusedCapacity(allocator, 1);
            overrides.appendAssumeCapacity(.{ .base = try allocator.dupe(u8, o.base), .bits = o.bits, .group = o.group });
        }
    }

    if (!progress.ngram_done) {
        say(sink, "n-gram table: merging shards", .{});
        const result = try pack.writeNgramTable(allocator, io, out, &src, .{});
        progress.ngram_present = result != null;
        progress.ngram_done = true;
        const carried = try allocator.alloc(OverrideJson, overrides.items.len);
        defer allocator.free(carried);
        for (overrides.items, carried) |o, *dst| dst.* = .{ .base = o.base, .bits = o.bits, .group = o.group };
        progress.overrides = carried;
        try writeProgress(allocator, io, out, progress);
    }
    if (progress.ngram_present) ngram_spec = if (served) |sv| sv.ngram else .{};

    var k_buf: [16]u8 = undefined;
    const k_text = ceiling.kText(&k_buf);
    var tok_buf: [24]u8 = undefined;
    var rows_buf: [24]u8 = undefined;
    var pin_buf: [16]u8 = undefined;
    var provenance_buf: [7][2][]const u8 = .{
        .{ "encoder", opts.version },
        .{ "quantizer", "ldlq-rotated" },
        .{ "global_scale_search", "on" },
        .{ "calib_sha256", opts.calib_sha256 },
        .{ "calib_tokens", try std.fmt.bufPrint(&tok_buf, "{d}", .{opts.calib_tokens}) },
        .{ "hessian_min_rows", try std.fmt.bufPrint(&rows_buf, "{d}", .{opts.min_rows}) },
        undefined,
    };
    var provenance_len: usize = 6;
    if (opts.pin_pass != encode.full_pin_pass) {
        provenance_buf[provenance_len] = .{ "pin_pass", try std.fmt.bufPrint(&pin_buf, "{d}", .{opts.pin_pass}) };
        provenance_len += 1;
    }
    const policy = convert.Policy{ .rate = opts.rate, .dec = opts.dec, .min_rows = opts.min_rows, .batch = opts.batch, .factor_precision = opts.factor_precision, .pin_pass = opts.pin_pass, .provenance = provenance_buf[0..provenance_len] };
    const geometry = convert.Geometry{ .experts = geom.experts, .hidden = geom.hidden, .inter = geom.inter };

    var out_abs_buf: [std.fs.max_path_bytes]u8 = undefined;
    const out_abs = out_abs_buf[0..try out.realPath(io, &out_abs_buf)];
    const kinds = try layerKinds(allocator, config_json, geom.layers);
    defer allocator.free(kinds);
    if (calibrator) |c| {
        try c.prepare(c.ctx, .{
            .pack_dir = out,
            .pack_abs = out_abs,
            .source_dir_abs = opts.model_dir,
            .source_config = config_json,
            .geom = .{ .experts = geom.experts, .hidden = geom.hidden, .inter = geom.inter },
            .layers = geom.layers,
            .layer_full = kinds,
            .ple_layer = geom.ple_layer,
            .rate = ceiling,
            .dec = opts.dec,
            .trunk_bits = trunk_bits,
            .trunk_group = trunk_group,
            .overrides = overrides.items,
            .ngram = ngram_spec,
            .eq = pack.expertQuant(opts.dec, k_text, &.{}),
            .perms = perms,
            .target_boundaries = opts.target_boundaries,
            .target_in_run = opts.target_in_run,
            .trunk_gptq = opts.trunk == .gptq,
        });
    }

    var layer: u32 = 0;
    var any_experts = false;
    while (layer < geom.layers) : (layer += 1) {
        var kb: [128]u8 = undefined;
        const probe = try std.fmt.bufPrint(&kb, "model.language_model.layers.{d}.mlp.experts.gate_up_proj", .{layer});
        if (!src.tensors.contains(probe)) continue;
        any_experts = true;
        var name_buf: [64]u8 = undefined;
        const name = try shard.shardFileName(&name_buf, layer);
        const hidden_ready = if (calibrator) |c| c.ready(c.ctx, layer, layer + 1 == geom.layers) else true;
        if (hidden_ready and layerKept(allocator, io, out, opts.trunk, geom.layers, layer)) {
            summary.layers_reused += 1;
            if (calibrator) |c| try c.done(c.ctx, layer, false, true);
            say(sink, "layer {d}: reused", .{layer});
            continue;
        }
        out.deleteFile(io, name) catch {};
        const begin_at = stages_mod.nowNs();
        const hs = if (calibrator) |c| try c.begin(c.ctx, layer, false) else null;
        const convert_at = stages_mod.nowNs();
        var layer_policy = policy;
        layer_policy.classes = if (opts.plan) |p| p.classes(layer) else null;
        layer_policy.target = opts.target_boundaries != null or opts.target_in_run;
        var lf: ?fisher.LayerFisher = null;
        defer if (lf) |*f| f.deinit();
        if (opts.fisher_dir) |dir| {
            lf = try fisher.loadLayer(allocator, io, dir, layer, geom.experts, geom.hidden, geom.inter);
            layer_policy.fisher = &lf.?;
        }
        const stats = try convert.convertQwenLayer(allocator, s, io, out, &src, layer, false, geometry, layer_policy, hs);
        const convert_done = stages_mod.nowNs();
        summary.layers_written += 1;
        summary.gate_up_identity += stats.gate_up_identity;
        summary.down_identity += stats.down_identity;
        say(sink, "layer {d}: written (identity Hessian: {d} gate/up, {d} down of {d} experts)", .{ layer, stats.gate_up_identity, stats.down_identity, stats.experts });
        var label_buf: [32]u8 = undefined;
        sayStages(sink, std.fmt.bufPrint(&label_buf, "layer {d}", .{layer}) catch "layer", &stats.stages);
        sayFactor(sink, layer, policy.factor_precision, &stats);
        const advance_at = stages_mod.nowNs();
        if (calibrator) |c| try c.done(c.ctx, layer, false, false);
        say(sink, "layer {d}: capture {d:.1}s, convert {d:.1}s, advance {d:.1}s", .{
            layer,
            @as(f64, @floatFromInt(convert_at - begin_at)) / 1e9,
            @as(f64, @floatFromInt(convert_done - convert_at)) / 1e9,
            @as(f64, @floatFromInt(stages_mod.nowNs() - advance_at)) / 1e9,
        });
        if (opts.layer_cooldown != 0) _ = mlx.mlx_synchronize(s);
        rest(sink, layer, opts.layer_cooldown, sleepSeconds);
    }
    if (!any_experts) return error.SourceHasNoExperts;

    if (geom.mtp) {
        var name_buf: [64]u8 = undefined;
        const name = try shard.mtpShardFileName(&name_buf, 0);
        if (!layerShardComplete(allocator, io, out, name, shard.mtp_prefix, 0)) {
            out.deleteFile(io, name) catch {};
            const hs = if (calibrator) |c| try c.begin(c.ctx, 0, true) else null;
            const stats = try convert.convertQwenLayer(allocator, s, io, out, &src, 0, true, geometry, policy, hs);
            summary.mtp_written = true;
            summary.gate_up_identity += stats.gate_up_identity;
            summary.down_identity += stats.down_identity;
            say(sink, "mtp: written (identity Hessian: {d} gate/up, {d} down of {d} experts)", .{ stats.gate_up_identity, stats.down_identity, stats.experts });
            sayStages(sink, "mtp", &stats.stages);
        }
    }

    if (opts.trunk == .gptq) {
        _ = out.statFile(io, trunk_mod.head_shard, .{}) catch return error.TrunkHeadMissing;
        const rewritten = try dropRerounded(allocator, io, out);
        say(sink, "trunk: re-rounded linears replace the base trunk's in {d} shards", .{rewritten});
    }
    const sidecars = try pack.copySidecars(allocator, io, src_dir, out);
    say(sink, "sidecars: {d} files copied", .{sidecars});
    const index = try indexOfDir(allocator, io, out);
    defer allocator.free(index);
    try out.writeFile(io, .{ .sub_path = ".model.safetensors.index.json.tmp", .data = index });
    try out.rename(".model.safetensors.index.json.tmp", out, "model.safetensors.index.json", io);

    // The shards' keys, then the ones that describe the whole pack.
    var extra = std.ArrayList([2][]const u8).empty;
    defer extra.deinit(allocator);
    try extra.appendSlice(allocator, provenance_buf[0..provenance_len]);
    if (opts.plan) |p| try extra.append(allocator, .{ "rate_plan_sha256", &p.sha256 });
    if (opts.fisher_dir != null) try extra.append(allocator, .{ "out_scales_mode", "fisher" });
    if (opts.target_boundaries != null or opts.target_in_run) try extra.append(allocator, .{ "target", "compensated" });
    if (opts.trunk == .gptq) try extra.append(allocator, .{ "trunk", "gptq" });
    const base_config = if (served) |sv| sv.config else config_json;
    const config = try pack.buildConfig(allocator, base_config, trunk_bits, trunk_group, overrides.items, ngram_spec, pack.expertQuant(opts.dec, k_text, extra.items));
    defer allocator.free(config);
    try out.writeFile(io, .{ .sub_path = ".config.json.tmp", .data = config });
    try out.rename(".config.json.tmp", out, "config.json", io);
    if (calibrator) |c| c.finish(c.ctx);
    out.deleteFile(io, progress_name) catch {};
    out.deleteTree(io, ".capture") catch {};
    say(sink, "pack complete", .{});
    return summary;
}

const Poison = struct {
    child: std.mem.Allocator,

    fn allocator(self: *Poison) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *Poison = @ptrCast(@alignCast(ctx));
        return self.child.vtable.alloc(self.child.ptr, len, alignment, ret_addr);
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *Poison = @ptrCast(@alignCast(ctx));
        return self.child.vtable.resize(self.child.ptr, memory, alignment, new_len, ret_addr);
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *Poison = @ptrCast(@alignCast(ctx));
        return self.child.vtable.remap(self.child.ptr, memory, alignment, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *Poison = @ptrCast(@alignCast(ctx));
        @memset(memory, 1);
        self.child.vtable.free(self.child.ptr, memory, alignment, ret_addr);
    }
};

const Fake = struct {
    begun: std.ArrayList(u32) = .empty,
    kinds: std.ArrayList(bool) = .empty,
    info: ?Info = null,
    fail_on: ?u32 = null,
    /// Info.perms as prepare saw it: the slices live only for the run.
    perms: std.ArrayList(?[]u32) = .empty,
    /// What a re-rounded trunk leaves in the pack: a shard per layer from `begin`, the head from the last layer's
    /// advance.
    write_trunk: bool = false,
    write_head: bool = false,

    fn begin(ctx: *anyopaque, layer: u32, mtp: bool) anyerror!?convert.HessianSource {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        if (mtp) return null;
        if (self.fail_on) |f| if (f == layer) return error.InjectedFailure;
        try self.begun.append(std.testing.allocator, layer);
        try self.kinds.append(std.testing.allocator, self.info.?.layer_full[layer]);
        if (self.write_trunk) {
            var name_buf: [64]u8 = undefined;
            try self.rerounded(try trunk_mod.shardName(&name_buf, layer));
        }
        return null;
    }

    fn rerounded(self: *Fake, name: []const u8) !void {
        var b = pack.ShardBuilder.init(std.testing.allocator);
        defer b.deinit();
        var tensor_buf: [96]u8 = undefined;
        try b.add(try std.fmt.bufPrint(&tensor_buf, "language_model.model.{s}.weight", .{name}), "BF16", &.{1}, &.{ 0, 0 });
        try b.flush(std.testing.io, self.info.?.pack_dir, name, &.{.{ "format", "mlx" }});
    }

    fn done(ctx: *anyopaque, layer: u32, mtp: bool, reused: bool) anyerror!void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        if (self.write_head and !mtp and !reused and layer + 1 == self.info.?.layers) try self.rerounded(trunk_mod.head_shard);
    }

    fn prepare(ctx: *anyopaque, info: Info) anyerror!void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.info = info;
        for (info.perms) |p| {
            const owned = if (p) |v| try std.testing.allocator.dupe(u32, v) else null;
            errdefer if (owned) |o| std.testing.allocator.free(o);
            try self.perms.append(std.testing.allocator, owned);
        }
    }

    fn ready(_: *anyopaque, _: u32, _: bool) bool {
        return true;
    }

    fn finish(_: *anyopaque) void {}

    fn deinit(self: *Fake) void {
        self.begun.deinit(std.testing.allocator);
        self.kinds.deinit(std.testing.allocator);
        for (self.perms.items) |p| if (p) |v| std.testing.allocator.free(v);
        self.perms.deinit(std.testing.allocator);
    }

    fn calibrator(self: *Fake) Calibrator {
        return .{ .ctx = self, .prepare = prepare, .ready = ready, .begin = begin, .done = done, .finish = finish };
    }
};

pub fn writeTinySource(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !void {
    const cfg =
        \\{"model_type":"qwen4_exp","text_config":{"model_type":"qwen4_exp_text","num_hidden_layers":2,"num_experts":2,"hidden_size":128,"moe_intermediate_size":128,"mtp_num_hidden_layers":1,"layer_types":["full_attention","linear_attention"]}}
    ;
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg });
    try dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = "{}" });
    const specs = [_]pack.fixture.Spec{
        .{ .name = "model.language_model.embed_tokens.weight", .dtype = "BF16", .shape = &.{ 64, 128 } },
        .{ .name = "model.language_model.layers.0.self_attn.q_proj.weight", .dtype = "BF16", .shape = &.{ 128, 128 } },
        .{ .name = "model.language_model.layers.0.mlp.gate.weight", .dtype = "BF16", .shape = &.{ 2, 128 } },
        .{ .name = "model.language_model.layers.1.mlp.gate.weight", .dtype = "BF16", .shape = &.{ 2, 128 } },
        .{ .name = "model.language_model.layers.0.mlp.experts.gate_up_proj", .dtype = "BF16", .shape = &.{ 2, 256, 128 } },
        .{ .name = "model.language_model.layers.0.mlp.experts.down_proj", .dtype = "BF16", .shape = &.{ 2, 128, 128 } },
        .{ .name = "model.language_model.layers.1.mlp.experts.gate_up_proj", .dtype = "BF16", .shape = &.{ 2, 256, 128 } },
        .{ .name = "model.language_model.layers.1.mlp.experts.down_proj", .dtype = "BF16", .shape = &.{ 2, 128, 128 } },
        .{ .name = "mtp.layers.0.mlp.experts.gate_up_proj", .dtype = "BF16", .shape = &.{ 2, 256, 128 } },
        .{ .name = "mtp.layers.0.mlp.experts.down_proj", .dtype = "BF16", .shape = &.{ 2, 128, 128 } },
        .{ .name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", .dtype = "BF16", .shape = &.{ 8, 64 } },
        .{ .name = "lm_head.weight", .dtype = "BF16", .shape = &.{ 64, 128 } },
    };
    try pack.fixture.write(allocator, io, dir, &specs);
}

fn readShard(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) !st.File {
    const raw = try dir.readFileAlloc(io, name, allocator, .limited(64 << 20));
    defer allocator.free(raw);
    return st.parse(allocator, raw);
}

pub fn tinyOptions(model_dir: []const u8, out_dir: []const u8, again: bool) Options {
    return .{
        .model_dir = model_dir,
        .out_dir = out_dir,
        .rate = .{ .n = 32 },
        .dec = .{ .codebook = .mcg, .window = .w12 },
        .calib_sha256 = "abc123",
        .calib_tokens = 4096,
        .resuming = again,
        .version = "sushi-encoder-test",
        .trunk = .rtn,
        .pin_pass = encode.full_pin_pass,
    };
}

pub fn tmpPath(allocator: std.mem.Allocator, tmp: *std.testing.TmpDir, sub: []const u8) ![]u8 {
    const io = std.testing.io;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &buf);
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ buf[0..n], sub });
}

test "driver: a tiny Qwen source becomes a complete pack, and a second run into it is refused" {
    try mlx.requireGpu();
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "src");
    var src_dir = try work.dir.openDir(io, "src", .{});
    defer src_dir.close(io);
    try writeTinySource(t.allocator, io, src_dir);
    const model = try tmpPath(t.allocator, &work, "src");
    defer t.allocator.free(model);
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);

    var fake = Fake{};
    defer fake.deinit();
    const summary = try run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, out_path, false), fake.calibrator(), null);
    try t.expectEqual(@as(usize, 2), summary.layers_written);
    try t.expect(summary.mtp_written);
    try t.expectEqualSlices(u32, &.{ 0, 1 }, fake.begun.items);

    var pack_dir = try work.dir.openDir(io, "pack", .{ .iterate = true });
    defer pack_dir.close(io);
    for ([_][]const u8{ "config.json", "model.safetensors.index.json", "layer-0.safetensors", "layer-1.safetensors", "mtp-0.safetensors", "ngram_table.bin", "tokenizer.json", "model-trunk-00000.safetensors" }) |n| {
        try pack_dir.access(io, n, .{});
    }
    try t.expectError(error.FileNotFound, pack_dir.access(io, progress_name, .{}));

    const cfg_raw = try pack_dir.readFileAlloc(io, "config.json", t.allocator, .limited(1 << 20));
    defer t.allocator.free(cfg_raw);
    const cfg = try std.json.parseFromSlice(std.json.Value, t.allocator, cfg_raw, .{});
    defer cfg.deinit();
    const parsed = try api.parseExpertQuant(cfg.value.object);
    try t.expectEqual(@as(u32, 32), parsed.rate.n);
    try t.expectEqual(format.Window.w12, parsed.window);
    try t.expectEqualStrings("sushi-encoder-test", cfg.value.object.get("expert_quant").?.object.get("encoder").?.string);

    const idx_raw = try pack_dir.readFileAlloc(io, "model.safetensors.index.json", t.allocator, .limited(1 << 20));
    defer t.allocator.free(idx_raw);
    const idx = try std.json.parseFromSlice(std.json.Value, t.allocator, idx_raw, .{});
    defer idx.deinit();
    const wm = idx.value.object.get("weight_map").?.object;
    try t.expectEqualStrings("layer-1.safetensors", wm.get("language_model.model.layers.1.mlp.switch_mlp.up_proj.trellis").?.string);
    try t.expectEqualStrings("mtp-0.safetensors", wm.get("language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.suh").?.string);
    try t.expect(wm.get("language_model.lm_head.scales") != null);

    try t.expectError(error.OutputNotEmpty, run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, out_path, false), null, null));
    try t.expect(cfg.value.object.get("expert_quant").?.object.get("pin_pass") == null);
    var shard_file = try readShard(t.allocator, io, pack_dir, "layer-0.safetensors");
    defer shard_file.deinit(t.allocator);
    try t.expect(shard_file.metadata.get("pin_pass") == null);
}

test "driver: the default pin pass is 64, stamped on the shards, and 128 is the unstamped full search" {
    try std.testing.expectEqual(@as(u32, 64), (Options{ .model_dir = "", .out_dir = "", .rate = .{ .n = 32 }, .dec = .{ .codebook = .mcg }, .version = "" }).pin_pass);
    try std.testing.expectEqual(@as(u32, 64), (convert.Policy{ .rate = .{ .n = 32 }, .dec = .{ .codebook = .mcg } }).pin_pass);
}

test "driver: a shortened pin pass is stamped on its shards and config, and a resume at any other refuses" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "src");
    var src_dir = try work.dir.openDir(io, "src", .{});
    defer src_dir.close(io);
    try writeTinySource(t.allocator, io, src_dir);
    const model = try tmpPath(t.allocator, &work, "src");
    defer t.allocator.free(model);
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);

    var bad = tinyOptions(model, out_path, false);
    bad.pin_pass = 24;
    try t.expectError(error.PinPassUnsupported, run(t.allocator, io, mlx.gpuStream(), bad, null, null));

    var pinned = tinyOptions(model, out_path, false);
    pinned.pin_pass = 64;
    var broken = Fake{ .fail_on = 1 };
    defer broken.deinit();
    try t.expectError(error.InjectedFailure, run(t.allocator, io, mlx.gpuStream(), pinned, broken.calibrator(), null));
    try t.expectError(error.ResumePolicyMismatch, run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, out_path, true), null, null));
    var other = pinned;
    other.resuming = true;
    other.pin_pass = 32;
    try t.expectError(error.ResumePolicyMismatch, run(t.allocator, io, mlx.gpuStream(), other, null, null));

    pinned.resuming = true;
    var fake = Fake{};
    defer fake.deinit();
    _ = try run(t.allocator, io, mlx.gpuStream(), pinned, fake.calibrator(), null);
    var pack_dir = try work.dir.openDir(io, "pack", .{ .iterate = true });
    defer pack_dir.close(io);
    for ([_][]const u8{ "layer-0.safetensors", "layer-1.safetensors" }) |name| {
        var shard_file = try readShard(t.allocator, io, pack_dir, name);
        defer shard_file.deinit(t.allocator);
        try t.expectEqualStrings("64", shard_file.metadata.get("pin_pass").?);
    }
    const cfg_raw = try pack_dir.readFileAlloc(io, "config.json", t.allocator, .limited(1 << 20));
    defer t.allocator.free(cfg_raw);
    const cfg = try std.json.parseFromSlice(std.json.Value, t.allocator, cfg_raw, .{});
    defer cfg.deinit();
    try t.expectEqualStrings("64", cfg.value.object.get("expert_quant").?.object.get("pin_pass").?.string);
}

test "driver: a run that fails after layer 0 resumes at layer 1, and a changed policy refuses to resume" {
    try mlx.requireGpu();
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "src");
    var src_dir = try work.dir.openDir(io, "src", .{});
    defer src_dir.close(io);
    try writeTinySource(t.allocator, io, src_dir);
    const model = try tmpPath(t.allocator, &work, "src");
    defer t.allocator.free(model);
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);

    var broken = Fake{ .fail_on = 1 };
    defer broken.deinit();
    try t.expectError(error.InjectedFailure, run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, out_path, false), broken.calibrator(), null));
    var pack_dir = try work.dir.openDir(io, "pack", .{ .iterate = true });
    defer pack_dir.close(io);
    try pack_dir.access(io, "layer-0.safetensors", .{});
    try t.expectError(error.FileNotFound, pack_dir.access(io, "config.json", .{}));
    try t.expectError(error.OutputNotEmpty, run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, out_path, false), null, null));

    var changed = tinyOptions(model, out_path, true);
    changed.rate = .{ .n = 48 };
    try t.expectError(error.ResumePolicyMismatch, run(t.allocator, io, mlx.gpuStream(), changed, null, null));
    var gpu_factored = tinyOptions(model, out_path, true);
    gpu_factored.factor_precision = .f32;
    try t.expectError(error.ResumePolicyMismatch, run(t.allocator, io, mlx.gpuStream(), gpu_factored, null, null));
    var shortened = tinyOptions(model, out_path, true);
    shortened.pin_pass = 64;
    try t.expectError(error.ResumePolicyMismatch, run(t.allocator, io, mlx.gpuStream(), shortened, null, null));

    var fake = Fake{};
    defer fake.deinit();
    const summary = try run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, out_path, true), fake.calibrator(), null);
    try t.expectEqual(@as(usize, 1), summary.layers_written);
    try t.expectEqual(@as(usize, 1), summary.layers_reused);
    try t.expectEqualSlices(u32, &.{1}, fake.begun.items);
    try pack_dir.access(io, "config.json", .{});

    var empty = t.tmpDir(.{});
    defer empty.cleanup();
    const nothing = try tmpPath(t.allocator, &empty, "x");
    defer t.allocator.free(nothing);
    try empty.dir.createDirPath(io, "x");
    try empty.dir.writeFile(io, .{ .sub_path = "x/stray.txt", .data = "x" });
    try t.expectError(error.ResumeNothingToResume, run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, nothing, true), null, null));
}

test "driver: the layer kinds a calibrator reads from Info outlive the call that prepared it" {
    try mlx.requireGpu();
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "src");
    var src_dir = try work.dir.openDir(io, "src", .{});
    defer src_dir.close(io);
    try writeTinySource(t.allocator, io, src_dir);
    const model = try tmpPath(t.allocator, &work, "src");
    defer t.allocator.free(model);
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);

    var poison = Poison{ .child = t.allocator };
    var fake = Fake{};
    defer fake.deinit();
    _ = try run(poison.allocator(), io, mlx.gpuStream(), tinyOptions(model, out_path, false), fake.calibrator(), null);
    try t.expectEqualSlices(bool, &.{ true, false }, fake.kinds.items);
}

fn tensorBytes(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) ![]u8 {
    const idx_raw = try dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(1 << 20));
    defer allocator.free(idx_raw);
    const idx = try std.json.parseFromSlice(std.json.Value, allocator, idx_raw, .{});
    defer idx.deinit();
    const file = idx.value.object.get("weight_map").?.object.get(name).?.string;
    const raw = try dir.readFileAlloc(io, file, allocator, .limited(1 << 26));
    defer allocator.free(raw);
    const header_len = std.mem.readInt(u64, raw[0..8], .little);
    const header = try std.json.parseFromSlice(std.json.Value, allocator, raw[8..][0..header_len], .{});
    defer header.deinit();
    const offs = header.value.object.get(name).?.object.get("data_offsets").?.array.items;
    return allocator.dupe(u8, raw[8 + header_len ..][@intCast(offs[0].integer)..@intCast(offs[1].integer)]);
}

test "driver: a rate plan over another pack's trunk writes table-codebook rate groups, reorders the router and stamps the ceiling" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "src");
    var src_dir = try work.dir.openDir(io, "src", .{});
    defer src_dir.close(io);
    try writeTinySource(t.allocator, io, src_dir);
    const model = try tmpPath(t.allocator, &work, "src");
    defer t.allocator.free(model);
    const served_path = try tmpPath(t.allocator, &work, "served");
    defer t.allocator.free(served_path);
    _ = try run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, served_path, false), null, null);

    var plan = try rate_plan.parse(t.allocator,
        \\{"format":"exls-rate-plan-1","experts":2,"layers":{"0":{"gate_up":[2,3],"down":[2,2]}}}
    , 2, 2);
    defer plan.deinit();
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);
    var opts = tinyOptions(model, out_path, false);
    opts.dec = .{ .codebook = .std1, .window = .w15, .table = api.codebook_tables.indexOf("std1") };
    opts.plan = &plan;
    opts.trunk_from = served_path;
    opts.trunk = .stored;
    var fake = Fake{};
    defer fake.deinit();
    _ = try run(t.allocator, io, mlx.gpuStream(), opts, fake.calibrator(), null);
    try t.expectEqualSlices(u32, &.{ 1, 0 }, fake.perms.items[0].?);
    try t.expect(fake.perms.items[1] == null);
    try t.expectEqual(@as(u32, 48), fake.info.?.rate.n);

    var served = try work.dir.openDir(io, "served", .{});
    defer served.close(io);
    var out = try work.dir.openDir(io, "pack", .{});
    defer out.close(io);
    const cfg_raw = try out.readFileAlloc(io, "config.json", t.allocator, .limited(1 << 20));
    defer t.allocator.free(cfg_raw);
    const cfg = try std.json.parseFromSlice(std.json.Value, t.allocator, cfg_raw, .{});
    defer cfg.deinit();
    const spec = try api.parseExpertQuant(cfg.value.object);
    try t.expectEqual(@as(u32, 48), spec.rate.n);
    try t.expectEqual(opts.dec.table, spec.table);
    try t.expectEqualStrings(&plan.sha256, cfg.value.object.get("expert_quant").?.object.get("rate_plan_sha256").?.string);

    const router = "language_model.model.layers.0.mlp.gate.weight";
    const before = try tensorBytes(t.allocator, io, served, router);
    defer t.allocator.free(before);
    const after = try tensorBytes(t.allocator, io, out, router);
    defer t.allocator.free(after);
    try t.expectEqualSlices(u8, before[256..512], after[0..256]);
    try t.expectEqualSlices(u8, before[0..256], after[256..512]);
    for ([_][]const u8{ "language_model.model.layers.1.mlp.gate.weight", "language_model.lm_head.scales", "language_model.model.layers.0.self_attn.q_proj.weight" }) |name| {
        const a = try tensorBytes(t.allocator, io, served, name);
        defer t.allocator.free(a);
        const b = try tensorBytes(t.allocator, io, out, name);
        defer t.allocator.free(b);
        try t.expectEqualSlices(u8, a, b);
    }
    const g0 = try tensorBytes(t.allocator, io, out, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.g0.trellis");
    defer t.allocator.free(g0);
    try t.expectEqual(@as(usize, 1 * 8 * 8 * 48 * 2), g0.len);
    const g1 = try tensorBytes(t.allocator, io, out, "language_model.model.layers.0.mlp.switch_mlp.down_proj.g1.trellis");
    defer t.allocator.free(g1);
    try t.expectEqual(@as(usize, 1 * 8 * 8 * 32 * 2), g1.len);
    const uniform = try tensorBytes(t.allocator, io, out, "language_model.model.layers.1.mlp.switch_mlp.up_proj.trellis");
    defer t.allocator.free(uniform);
    try t.expectEqual(@as(usize, 2 * 8 * 8 * 32 * 2), uniform.len);
}

test "driver: a rate plan over the trunk written from the source reorders the router into group order" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "src");
    var src_dir = try work.dir.openDir(io, "src", .{});
    defer src_dir.close(io);
    try writeTinySource(t.allocator, io, src_dir);
    const model = try tmpPath(t.allocator, &work, "src");
    defer t.allocator.free(model);
    const unplanned_path = try tmpPath(t.allocator, &work, "unplanned");
    defer t.allocator.free(unplanned_path);
    _ = try run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, unplanned_path, false), null, null);

    var plan = try rate_plan.parse(t.allocator,
        \\{"format":"exls-rate-plan-1","experts":2,"layers":{"0":{"gate_up":[2,3],"down":[2,2]}}}
    , 2, 2);
    defer plan.deinit();
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);
    var opts = tinyOptions(model, out_path, false);
    opts.dec = .{ .codebook = .std1, .window = .w15, .table = api.codebook_tables.indexOf("std1") };
    opts.plan = &plan;
    var fake = Fake{};
    defer fake.deinit();
    _ = try run(t.allocator, io, mlx.gpuStream(), opts, fake.calibrator(), null);
    try t.expectEqualSlices(u32, &.{ 1, 0 }, fake.perms.items[0].?);

    var unplanned = try work.dir.openDir(io, "unplanned", .{});
    defer unplanned.close(io);
    var out = try work.dir.openDir(io, "pack", .{});
    defer out.close(io);
    const router = "language_model.model.layers.0.mlp.gate.weight";
    const before = try tensorBytes(t.allocator, io, unplanned, router);
    defer t.allocator.free(before);
    const after = try tensorBytes(t.allocator, io, out, router);
    defer t.allocator.free(after);
    try t.expectEqualSlices(u8, before[256..512], after[0..256]);
    try t.expectEqualSlices(u8, before[0..256], after[256..512]);
    const unplanned_layer1 = try tensorBytes(t.allocator, io, unplanned, "language_model.model.layers.1.mlp.gate.weight");
    defer t.allocator.free(unplanned_layer1);
    const layer1 = try tensorBytes(t.allocator, io, out, "language_model.model.layers.1.mlp.gate.weight");
    defer t.allocator.free(layer1);
    try t.expectEqualSlices(u8, unplanned_layer1, layer1);
}

test "driver: a trunk mode without what it needs is refused before anything is written" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "src");
    var src_dir = try work.dir.openDir(io, "src", .{});
    defer src_dir.close(io);
    try writeTinySource(t.allocator, io, src_dir);
    const model = try tmpPath(t.allocator, &work, "src");
    defer t.allocator.free(model);
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);
    var opts = tinyOptions(model, out_path, false);
    opts.trunk = .stored;
    try t.expectError(error.TrunkFromMissing, run(t.allocator, io, mlx.gpuStream(), opts, null, null));
    opts.trunk = .rtn;
    opts.trunk_from = model;
    try t.expectError(error.TrunkModeConflict, run(t.allocator, io, mlx.gpuStream(), opts, null, null));
    opts.trunk = .gptq;
    opts.trunk_from = null;
    try t.expectError(error.TrunkNeedsCalibration, run(t.allocator, io, mlx.gpuStream(), opts, null, null));
    opts.trunk = .rtn;
    opts.target_in_run = true;
    try t.expectError(error.TargetNeedsCalibration, run(t.allocator, io, mlx.gpuStream(), opts, null, null));
    var fake = Fake{};
    defer fake.deinit();
    opts.target_boundaries = model;
    try t.expectError(error.TargetConflict, run(t.allocator, io, mlx.gpuStream(), opts, fake.calibrator(), null));
    try t.expectError(error.FileNotFound, work.dir.access(io, "pack", .{}));
}

test "driver: re-rounded shards win over the base trunk's copies, which leave its shards at the end" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{ .iterate = true });
    defer work.cleanup();
    const a = [_]u16{ 1, 2, 3, 4 };
    const b = [_]u16{ 5, 6 };
    const a2 = [_]u16{ 9, 9, 9, 9 };
    var base = pack.ShardBuilder.init(t.allocator);
    defer base.deinit();
    try base.add("language_model.model.layers.0.x.weight", "BF16", &.{ 2, 2 }, std.mem.sliceAsBytes(&a));
    try base.add("language_model.model.layers.0.norm.weight", "BF16", &.{2}, std.mem.sliceAsBytes(&b));
    try base.flush(io, work.dir, "model-trunk-00001.safetensors", &.{.{ "format", "mlx" }});
    var re = pack.ShardBuilder.init(t.allocator);
    defer re.deinit();
    try re.add("language_model.model.layers.0.x.weight", "BF16", &.{ 2, 2 }, std.mem.sliceAsBytes(&a2));
    var name_buf: [64]u8 = undefined;
    try re.flush(io, work.dir, try trunk_mod.shardName(&name_buf, 0), &.{.{ "format", "mlx" }});

    var map = try layer_stage.PackMap.scan(t.allocator, io, work.dir);
    const seen = try map.bytes(t.allocator, io, "language_model.model.layers.0.x.weight");
    map.deinit();
    defer t.allocator.free(seen);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&a2), seen);

    try t.expectEqual(@as(usize, 1), try dropRerounded(t.allocator, io, work.dir));
    var h = try trunk_from.readHeader(t.allocator, io, work.dir, "model-trunk-00001.safetensors");
    defer h.parsed.deinit();
    try t.expect(h.parsed.value.object.get("language_model.model.layers.0.x.weight") == null);
    try t.expect(h.parsed.value.object.get("language_model.model.layers.0.norm.weight") != null);
    try t.expectEqualStrings("mlx", h.parsed.value.object.get("__metadata__").?.object.get("format").?.string);
    var after = try layer_stage.PackMap.scan(t.allocator, io, work.dir);
    defer after.deinit();
    const norm = try after.bytes(t.allocator, io, "language_model.model.layers.0.norm.weight");
    defer t.allocator.free(norm);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&b), norm);
}

const RestLog = struct {
    slept: u32 = 0,
    calls: u32 = 0,
    lines: [4][64]u8 = undefined,
    lens: [4]usize = @splat(0),
    n: usize = 0,

    fn write(ctx: ?*anyopaque, line: []const u8) void {
        const self: *RestLog = @ptrCast(@alignCast(ctx.?));
        @memcpy(self.lines[self.n][0..line.len], line);
        self.lens[self.n] = line.len;
        self.n += 1;
    }
};

var rest_log: RestLog = .{};

fn recordSleep(seconds: u32) void {
    rest_log.slept += seconds;
    rest_log.calls += 1;
}

test "driver: a layer rests for the cooldown through the injected sleep, and a zero cooldown neither sleeps nor logs" {
    const t = std.testing;
    rest_log = .{};
    const sink = Sink{ .ctx = &rest_log, .write = RestLog.write };
    rest(sink, 3, 0, recordSleep);
    try t.expectEqual(@as(u32, 0), rest_log.calls);
    try t.expectEqual(@as(usize, 0), rest_log.n);
    rest(sink, 3, 15, recordSleep);
    rest(null, 4, 2, recordSleep);
    try t.expectEqual(@as(u32, 2), rest_log.calls);
    try t.expectEqual(@as(u32, 17), rest_log.slept);
    try t.expectEqual(@as(usize, 1), rest_log.n);
    try t.expectEqualStrings("layer 3: cooldown 15s", rest_log.lines[0][0..rest_log.lens[0]]);
}

test "driver: a base shard whose every tensor was re-rounded is removed, not left empty" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{ .iterate = true });
    defer work.cleanup();
    const w = [_]u16{ 1, 2, 3, 4 };
    const w2 = [_]u16{ 7, 7, 7, 7 };
    var base = pack.ShardBuilder.init(t.allocator);
    defer base.deinit();
    try base.add("language_model.lm_head.weight", "BF16", &.{ 2, 2 }, std.mem.sliceAsBytes(&w));
    try base.flush(io, work.dir, "model-lm-head.safetensors", &.{.{ "format", "mlx" }});
    var re = pack.ShardBuilder.init(t.allocator);
    defer re.deinit();
    try re.add("language_model.lm_head.weight", "BF16", &.{ 2, 2 }, std.mem.sliceAsBytes(&w2));
    try re.flush(io, work.dir, trunk_mod.head_shard, &.{.{ "format", "mlx" }});

    try t.expectEqual(@as(usize, 1), try dropRerounded(t.allocator, io, work.dir));
    try t.expectError(error.FileNotFound, work.dir.access(io, "model-lm-head.safetensors", .{}));
    var map = try layer_stage.PackMap.scan(t.allocator, io, work.dir);
    defer map.deinit();
    const seen = try map.bytes(t.allocator, io, "language_model.lm_head.weight");
    defer t.allocator.free(seen);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&w2), seen);
}

test "driver: a finished layer shard over 1 GiB counts as complete on resume" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    // A sparse file stands in for a 4-bit Qwen layer, which is larger than 1 GiB.
    const payload: u64 = (1 << 30) + (64 << 20);
    var header = std.ArrayList(u8).empty;
    defer header.deinit(t.allocator);
    try header.appendSlice(t.allocator, "{");
    const parts = [_][]const u8{ "trellis", "suh", "svh" };
    var at: u64 = 0;
    var i: usize = 0;
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| for (parts) |part| {
        const len: u64 = if (i == 8) payload - at else 64;
        if (i != 0) try header.appendSlice(t.allocator, ",");
        try header.print(t.allocator, "\"language_model.model.layers.3.mlp.switch_mlp.{s}.{s}\":{{\"dtype\":\"U16\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ proj, part, len / 2, at, at + len });
        at += len;
        i += 1;
    };
    try header.appendSlice(t.allocator, "}");
    while (header.items.len % 8 != 0) try header.append(t.allocator, ' ');
    var f = try work.dir.createFile(io, "layer-3.safetensors", .{});
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.items.len, .little);
    try f.writePositionalAll(io, &len_buf, 0);
    try f.writePositionalAll(io, header.items, 8);
    try f.writePositionalAll(io, &[_]u8{0}, 8 + header.items.len + payload - 1);
    f.close(io);
    try t.expect(layerShardComplete(t.allocator, io, work.dir, "layer-3.safetensors", shard.trunk_prefix, 3));
    try t.expect(!layerShardComplete(t.allocator, io, work.dir, "layer-3.safetensors", shard.trunk_prefix, 4));

    try work.dir.rename("layer-3.safetensors", work.dir, "layer-5.safetensors", io);
    var g = try work.dir.openFile(io, "layer-5.safetensors", .{ .mode = .read_write });
    defer g.close(io);
    try g.setLength(io, 8 + header.items.len + payload - 1);
    try t.expect(!layerShardComplete(t.allocator, io, work.dir, "layer-5.safetensors", shard.trunk_prefix, 3));
}

test "driver: a resume into a finished pack stops, whether or not its progress file or capture is left" {
    try mlx.requireGpu();
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "src");
    var src_dir = try work.dir.openDir(io, "src", .{});
    defer src_dir.close(io);
    try writeTinySource(t.allocator, io, src_dir);
    const model = try tmpPath(t.allocator, &work, "src");
    defer t.allocator.free(model);
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);
    _ = try run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, out_path, false), null, null);

    var pack_dir = try work.dir.openDir(io, "pack", .{ .iterate = true });
    defer pack_dir.close(io);
    const layer_before = try pack_dir.readFileAlloc(io, "layer-0.safetensors", t.allocator, .limited(1 << 24));
    defer t.allocator.free(layer_before);

    try t.expectError(error.ResumePackComplete, run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, out_path, true), null, null));
    try pack_dir.writeFile(io, .{ .sub_path = progress_name, .data = "{\"policy\":\"x\"}" });
    try t.expectError(error.ResumePackComplete, run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, out_path, true), null, null));

    const layer_after = try pack_dir.readFileAlloc(io, "layer-0.safetensors", t.allocator, .limited(1 << 24));
    defer t.allocator.free(layer_after);
    try t.expectEqualSlices(u8, layer_before, layer_after);
    try pack_dir.access(io, progress_name, .{});
}

fn tinyWork(work: *std.testing.TmpDir) ![]u8 {
    const io = std.testing.io;
    try work.dir.createDirPath(io, "src");
    var src_dir = try work.dir.openDir(io, "src", .{});
    defer src_dir.close(io);
    try writeTinySource(std.testing.allocator, io, src_dir);
    return tmpPath(std.testing.allocator, work, "src");
}

fn expectNothingWritten(work: *std.testing.TmpDir) !void {
    try std.testing.expectError(error.FileNotFound, work.dir.access(std.testing.io, "pack", .{}));
}

test "driver: a gptq resume that finds the last layer finished but no head redoes that layer, which writes the head" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    const model = try tinyWork(&work);
    defer t.allocator.free(model);
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);

    var stopped = Fake{ .write_trunk = true };
    defer stopped.deinit();
    var opts = tinyOptions(model, out_path, false);
    opts.trunk = .gptq;
    try t.expectError(error.TrunkHeadMissing, run(t.allocator, io, mlx.gpuStream(), opts, stopped.calibrator(), null));
    try t.expectEqualSlices(u32, &.{ 0, 1 }, stopped.begun.items);

    var resumed = Fake{ .write_trunk = true, .write_head = true };
    defer resumed.deinit();
    opts.resuming = true;
    const summary = try run(t.allocator, io, mlx.gpuStream(), opts, resumed.calibrator(), null);
    try t.expectEqualSlices(u32, &.{1}, resumed.begun.items);
    try t.expectEqual(@as(usize, 1), summary.layers_reused);
    try t.expectEqual(@as(usize, 1), summary.layers_written);
    var pack_dir = try work.dir.openDir(io, "pack", .{});
    defer pack_dir.close(io);
    try pack_dir.access(io, "config.json", .{});
    try pack_dir.access(io, trunk_mod.head_shard, .{});
}

test "driver: a gptq run does not take the trunk-from pack's re-rounded head as its own" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    const model = try tinyWork(&work);
    defer t.allocator.free(model);
    const served_path = try tmpPath(t.allocator, &work, "served");
    defer t.allocator.free(served_path);
    _ = try run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, served_path, false), null, null);
    var served = try work.dir.openDir(io, "served", .{});
    defer served.close(io);
    var head = pack.ShardBuilder.init(t.allocator);
    defer head.deinit();
    try head.add("language_model.lm_head.weight", "BF16", &.{1}, &.{ 0, 0 });
    try head.flush(io, served, trunk_mod.head_shard, &.{.{ "format", "mlx" }});

    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);
    var fake = Fake{ .write_trunk = true };
    defer fake.deinit();
    var opts = tinyOptions(model, out_path, false);
    opts.trunk = .gptq;
    opts.trunk_from = served_path;
    try t.expectError(error.TrunkHeadMissing, run(t.allocator, io, mlx.gpuStream(), opts, fake.calibrator(), null));
}

test "driver: --trunk-from a rate-grouped pack is refused before anything is written" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    const model = try tinyWork(&work);
    defer t.allocator.free(model);
    const served_path = try tmpPath(t.allocator, &work, "served");
    defer t.allocator.free(served_path);
    _ = try run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, served_path, false), null, null);

    var plan = try rate_plan.parse(t.allocator,
        \\{"format":"exls-rate-plan-1","experts":2,"layers":{"0":{"gate_up":[2,3],"down":[2,2]}}}
    , 2, 2);
    defer plan.deinit();
    const grouped_path = try tmpPath(t.allocator, &work, "grouped");
    defer t.allocator.free(grouped_path);
    var opts = tinyOptions(model, grouped_path, false);
    opts.plan = &plan;
    opts.trunk_from = served_path;
    opts.trunk = .stored;
    _ = try run(t.allocator, io, mlx.gpuStream(), opts, null, null);

    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);
    var again = tinyOptions(model, out_path, false);
    again.trunk_from = grouped_path;
    again.trunk = .stored;
    try t.expectError(error.TrunkFromRateGrouped, run(t.allocator, io, mlx.gpuStream(), again, null, null));
    try expectNothingWritten(&work);
}

test "driver: a rate the GPU search cannot run is refused before anything is written" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    const model = try tinyWork(&work);
    defer t.allocator.free(model);
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);
    for ([_]u32{ 16, 22, 33 }) |n| {
        var opts = tinyOptions(model, out_path, false);
        opts.rate = .{ .n = n };
        try t.expectError(error.RateUnsupported, run(t.allocator, io, mlx.gpuStream(), opts, null, null));
        try expectNothingWritten(&work);
    }
}

test "driver: a GPTQ trunk at a width it cannot pack is refused before anything is written, and so is a trunk-from pack at one" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    const model = try tinyWork(&work);
    defer t.allocator.free(model);
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);
    var fake = Fake{};
    defer fake.deinit();
    for ([_]u32{ 3, 5, 6 }) |bits| {
        var opts = tinyOptions(model, out_path, false);
        opts.trunk = .gptq;
        opts.trunk_bits = bits;
        try t.expectError(error.TrunkBitsUnsupported, run(t.allocator, io, mlx.gpuStream(), opts, fake.calibrator(), null));
        try expectNothingWritten(&work);
    }

    const served_path = try tmpPath(t.allocator, &work, "served");
    defer t.allocator.free(served_path);
    var narrow = tinyOptions(model, served_path, false);
    narrow.trunk_bits = 3;
    _ = try run(t.allocator, io, mlx.gpuStream(), narrow, null, null);
    var from = tinyOptions(model, out_path, false);
    from.trunk = .gptq;
    from.trunk_from = served_path;
    try t.expectError(error.TrunkBitsUnsupported, run(t.allocator, io, mlx.gpuStream(), from, fake.calibrator(), null));
    try expectNothingWritten(&work);

    var base_name = "language_model.model.layers.0.self_attn.q_proj".*;
    const overrides = [_]pack.Override{.{ .base = &base_name, .bits = 5, .group = 64 }};
    try t.expect(gptqRounds(8, &.{}));
    try t.expect(!gptqRounds(8, &overrides));
}

test "driver: --trunk-bits and --trunk-group beside --trunk-from are refused, not replaced by the served pack's width" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    const model = try tinyWork(&work);
    defer t.allocator.free(model);
    const served_path = try tmpPath(t.allocator, &work, "served");
    defer t.allocator.free(served_path);
    _ = try run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, served_path, false), null, null);
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);
    var opts = tinyOptions(model, out_path, false);
    opts.trunk_from = served_path;
    opts.trunk = .stored;
    opts.trunk_bits = 4;
    try t.expectError(error.TrunkWidthConflict, run(t.allocator, io, mlx.gpuStream(), opts, null, null));
    opts.trunk_bits = null;
    opts.trunk_group = 128;
    try t.expectError(error.TrunkWidthConflict, run(t.allocator, io, mlx.gpuStream(), opts, null, null));
    try expectNothingWritten(&work);
}

test "driver: a missing Fisher or stored-boundary file is named before anything is written, and a resume skips the layers it keeps" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    const model = try tinyWork(&work);
    defer t.allocator.free(model);
    try work.dir.createDirPath(io, "inputs");
    const inputs = try tmpPath(t.allocator, &work, "inputs");
    defer t.allocator.free(inputs);
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);

    var opts = tinyOptions(model, out_path, false);
    opts.fisher_dir = inputs;
    try t.expectError(error.FisherFileMissing, run(t.allocator, io, mlx.gpuStream(), opts, null, null));
    try expectNothingWritten(&work);
    opts.fisher_dir = null;
    opts.target_boundaries = inputs;
    try t.expectError(error.TargetBoundaryMissing, run(t.allocator, io, mlx.gpuStream(), opts, null, null));
    try expectNothingWritten(&work);

    _ = try run(t.allocator, io, mlx.gpuStream(), tinyOptions(model, out_path, false), null, null);
    var pack_dir = try work.dir.openDir(io, "pack", .{ .iterate = true });
    defer pack_dir.close(io);
    var src_dir = try work.dir.openDir(io, "src", .{ .iterate = true });
    defer src_dir.close(io);
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir);
    defer src.deinit();
    var resume_opts = tinyOptions(model, out_path, true);
    resume_opts.target_boundaries = inputs;
    try checkLayerInputs(t.allocator, io, resume_opts, &src, 2, pack_dir, null);
    try pack_dir.deleteFile(io, "layer-1.safetensors");
    try t.expectError(error.TargetBoundaryMissing, checkLayerInputs(t.allocator, io, resume_opts, &src, 2, pack_dir, null));
    try work.dir.writeFile(io, .{ .sub_path = "inputs/boundary-01.u16", .data = "x" });
    try checkLayerInputs(t.allocator, io, resume_opts, &src, 2, pack_dir, null);
}

test "driver: config.json's expert_quant carries every provenance key a shard does" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    const model = try tinyWork(&work);
    defer t.allocator.free(model);
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);
    var opts = tinyOptions(model, out_path, false);
    opts.pin_pass = 96;
    _ = try run(t.allocator, io, mlx.gpuStream(), opts, null, null);
    var pack_dir = try work.dir.openDir(io, "pack", .{});
    defer pack_dir.close(io);
    var layer = try readShard(t.allocator, io, pack_dir, "layer-0.safetensors");
    defer layer.deinit(t.allocator);
    const cfg_raw = try pack_dir.readFileAlloc(io, "config.json", t.allocator, .limited(1 << 20));
    defer t.allocator.free(cfg_raw);
    const cfg = try std.json.parseFromSlice(std.json.Value, t.allocator, cfg_raw, .{});
    defer cfg.deinit();
    const eq = cfg.value.object.get("expert_quant").?.object;
    var checked: usize = 0;
    var it = layer.metadata.iterator();
    while (it.next()) |kv| {
        var engine = false;
        for ([_][]const u8{ "format", "k", "codebook", "window" }) |name| engine = engine or std.mem.eql(u8, kv.key_ptr.*, name);
        if (engine) continue;
        try t.expectEqualStrings(kv.value_ptr.*, eq.get(kv.key_ptr.*).?.string);
        checked += 1;
    }
    try t.expect(eq.contains("global_scale_search") and eq.contains("hessian_min_rows") and eq.contains("pin_pass"));
    try t.expectEqual(@as(usize, 7), checked);
}

test "driver: a PAIR1 pack stamps pair1-w16-v1, and the engine's served MoE decodes its experts as the host decoder does" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "src");
    var src_dir = try work.dir.openDir(io, "src", .{});
    defer src_dir.close(io);
    try writeTinySource(t.allocator, io, src_dir);
    const model = try tmpPath(t.allocator, &work, "src");
    defer t.allocator.free(model);
    const out_path = try tmpPath(t.allocator, &work, "pack");
    defer t.allocator.free(out_path);
    var opts = tinyOptions(model, out_path, false);
    const dec = format.Decode{ .codebook = .pair1, .window = .w16, .table = api.codebook_tables.indexOf("pair1-w16-v1") };
    opts.dec = dec;
    opts.pin_pass = encode.default_pin_pass;
    _ = try run(t.allocator, io, mlx.gpuStream(), opts, null, null);

    var pack_dir = try work.dir.openDir(io, "pack", .{});
    defer pack_dir.close(io);
    const cfg_raw = try pack_dir.readFileAlloc(io, "config.json", t.allocator, .limited(1 << 20));
    defer t.allocator.free(cfg_raw);
    const cfg = try std.json.parseFromSlice(std.json.Value, t.allocator, cfg_raw, .{});
    defer cfg.deinit();
    const eq = cfg.value.object.get("expert_quant").?.object;
    try t.expectEqualStrings("stq", eq.get("format").?.string);
    try t.expectEqualStrings("pair1", eq.get("codebook").?.string);
    try t.expectEqualStrings("pair1-w16-v1", eq.get("codebook_table").?.string);
    try t.expectEqualStrings("c089b4eb", eq.get("codebook_digest").?.string);
    const spec = try api.parseExpertQuant(cfg.value.object);
    try t.expectEqual(@as(u32, 32), spec.rate.n);
    try t.expectEqual(format.Window.w16, spec.window);
    try t.expectEqual(dec.table, spec.table);
    _ = try api.codebook_tables.resolve(eq, .pair1, .w16);

    const raw = try pack_dir.readFileAlloc(io, "layer-0.safetensors", t.allocator, .limited(64 << 20));
    defer t.allocator.free(raw);
    const aligned = try t.allocator.alignedAlloc(u8, .of(u16), raw.len);
    defer t.allocator.free(aligned);
    @memcpy(aligned, raw);
    var shard_file = try st.parse(t.allocator, aligned);
    defer shard_file.deinit(t.allocator);
    try t.expectEqualStrings("pair1", shard_file.metadata.get("codebook").?);
    try t.expectEqualStrings("16", shard_file.metadata.get("window").?);

    const s = mlx.gpuStream();
    const dim = 128;
    const rate = format.Rate{ .n = 32 };
    const expert = 1;
    var prng = std.Random.DefaultPrng.init(5);
    var x: [dim]f32 = undefined;
    for (&x) |*v| v.* = prng.random().floatNorm(f32);
    var projs: [3]api.Proj = undefined;
    var weights: [3][dim * dim]f32 = undefined;
    defer for (projs) |p| {
        _ = mlx.mlx_array_free(p.trellis);
        _ = mlx.mlx_array_free(p.suh);
        _ = mlx.mlx_array_free(p.svh);
    };
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, pi| {
        var key: [160]u8 = undefined;
        const prefix = "language_model.model.layers.0.mlp.switch_mlp.";
        const tv = shard_file.tensors.get(try std.fmt.bufPrint(&key, "{s}{s}.trellis", .{ prefix, proj })).?;
        const trellis = st.slice(u16, tv);
        const suh = st.slice(u16, shard_file.tensors.get(try std.fmt.bufPrint(&key, "{s}{s}.suh", .{ prefix, proj })).?);
        const svh = st.slice(u16, shard_file.tensors.get(try std.fmt.bufPrint(&key, "{s}{s}.svh", .{ prefix, proj })).?);
        const tiles = (dim / 16) * (dim / 16);
        var public: [dim * dim]u16 = undefined;
        try format.reconstructPublic(t.allocator, trellis[expert * tiles * rate.n ..][0 .. tiles * rate.n], suh[expert * dim ..][0..dim], svh[expert * dim ..][0..dim], dim, dim, rate, dec, &public);
        for (&weights[pi], public) |*w, b| w.* = format.f16BitsToF32(b);
        var shape: [4]c_int = undefined;
        for (tv.shape, 0..) |d, i| shape[i] = @intCast(d);
        projs[pi] = .{
            .trellis = mlx.mlx_array_new_data(trellis.ptr, &shape, 4, .uint16),
            .suh = mlx.mlx_array_new_data(suh.ptr, &[_]c_int{ 2, dim }, 2, .float16),
            .svh = mlx.mlx_array_new_data(svh.ptr, &[_]c_int{ 2, dim }, 2, .float16),
        };
    }
    var want: [dim]f32 = undefined;
    var mid: [dim]f32 = undefined;
    for (0..dim) |c| {
        var g: f32 = 0;
        var u: f32 = 0;
        for (0..dim) |r| {
            g += x[r] * weights[0][r * dim + c];
            u += x[r] * weights[1][r * dim + c];
        }
        mid[c] = g / (1 + @exp(-g)) * u;
    }
    for (0..dim) |c| {
        var acc: f32 = 0;
        for (0..dim) |r| acc += mid[r] * weights[2][r * dim + c];
        want[c] = acc;
    }
    const xa = mlx.mlx_array_new_data(&x, &[_]c_int{ 1, 1, dim }, 3, .float32);
    defer _ = mlx.mlx_array_free(xa);
    const ind = [_]u32{expert};
    const inds = mlx.mlx_array_new_data(&ind, &[_]c_int{ 1, 1, 1 }, 3, .uint32);
    defer _ = mlx.mlx_array_free(inds);
    const one = [_]f32{1};
    const scores = mlx.mlx_array_new_data(&one, &[_]c_int{ 1, 1, 1 }, 3, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const y = try api.moe(s, xa, .{ .gate = projs[0], .up = projs[1], .down = projs[2] }, inds, scores, dec, false);
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));
    const got = mlx.mlx_array_data_float32(y).?;
    var err: f64 = 0;
    var energy: f64 = 0;
    for (want, 0..) |w, i| {
        err += (got[i] - w) * (got[i] - w);
        energy += w * w;
    }
    try t.expect(energy > 0 and err / energy < 1e-4);
}
