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

pub const Error = error{
    OutputNotEmpty,
    ResumeNothingToResume,
    ResumePolicyMismatch,
    ResumePackComplete,
    ArchitectureNotConvertible,
    SourceConfigInvalid,
    SourceHasNoExperts,
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
    has_ngram: bool,
    eq: pack.ExpertQuantSpec,
};

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
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    w.print("{s}: stage times ", .{label}) catch return;
    times.write(&w) catch return;
    s.write(s.ctx, w.buffered());
}

/// The float64 factor adds nothing to the key, so a float64 conversion begun before the precision choice
/// existed still resumes.
fn policyKey(allocator: std.mem.Allocator, opts: Options, bits: u32, group: u32) ![]u8 {
    const factor = if (opts.factor_precision == .f32) "|factor=32" else "";
    return std.fmt.allocPrint(allocator, "{s}|n={d}|{s}|w={d}|trunk={d}/{d}|calib={s}:{d}|min_rows={d}{s}", .{
        opts.version, opts.rate.n, @tagName(opts.dec.codebook), opts.dec.window.bits(), bits, group, opts.calib_sha256, opts.calib_tokens, opts.min_rows, factor,
    });
}

const Geometry = struct { layers: u32, experts: usize, hidden: usize, inter: usize, mtp: bool, ple_layer: u32 = 0 };

fn textConfigInt(obj: std.json.ObjectMap, key: []const u8) !usize {
    const v = obj.get(key) orelse return error.SourceConfigInvalid;
    if (v != .integer or v.integer <= 0) return error.SourceConfigInvalid;
    return @intCast(v.integer);
}

fn pleLayer(text: std.json.ObjectMap) u32 {
    const v = text.get("ple_layer_ids") orelse return 0;
    if (v != .array or v.array.items.len == 0 or v.array.items[0] != .integer or v.array.items[0].integer < 1) return 0;
    return @intCast(v.array.items[0].integer - 1);
}

fn layerKinds(allocator: std.mem.Allocator, config_json: []const u8, layers: u32) ![]bool {
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

fn readGeometry(allocator: std.mem.Allocator, config_json: []const u8, src: *const source.SafetensorsSource) !Geometry {
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
    const probe = std.fmt.bufPrint(&key, "{s}.layers.{d}.mlp.switch_mlp.down_proj.svh", .{ prefix, layer }) catch return false;
    const entry = parsed.value.object.get(probe) orelse return false;
    if (entry != .object) return false;
    const offsets = entry.object.get("data_offsets") orelse return false;
    if (offsets != .array or offsets.array.items.len != 2 or offsets.array.items[1] != .integer or offsets.array.items[1].integer < 0) return false;
    return @as(u64, @intCast(offsets.array.items[1].integer)) + 8 + header_len == size;
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
        try names.append(allocator, try allocator.dupe(u8, entry.name));
    }
    std.mem.sortUnstable([]u8, names.items, {}, struct {
        fn lt(_: void, a: []u8, b: []u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    return names.toOwnedSlice(allocator);
}

fn indexOfDir(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) ![]u8 {
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
    const trunk_bits = opts.trunk_bits orelse arch.defaultTrunkBits();
    const trunk_group = opts.trunk_group orelse arch.defaultTrunkGroup();

    var src = try source.SafetensorsSource.open(allocator, io, src_dir);
    defer src.deinit();
    const config_json = try src_dir.readFileAlloc(io, "config.json", allocator, .limited(16 << 20));
    defer allocator.free(config_json);
    const geom = try readGeometry(allocator, config_json, &src);

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

    if (!progress.trunk_done) {
        say(sink, "trunk: writing affine {d}-bit shards", .{trunk_bits});
        var trunk = try pack.writeQwenTrunk(allocator, io, out, &src, trunk_bits, trunk_group);
        defer trunk.deinit();
        const stored = try allocator.alloc(OverrideJson, trunk.overrides.items.len);
        defer allocator.free(stored);
        for (trunk.overrides.items, stored) |o, *dst| dst.* = .{ .base = o.base, .bits = o.bits, .group = o.group };
        progress.overrides = stored;
        progress.trunk_done = true;
        try writeProgress(allocator, io, out, progress);
        for (trunk.overrides.items) |o| {
            try overrides.append(allocator, .{ .base = try allocator.dupe(u8, o.base), .bits = o.bits, .group = o.group });
        }
    } else {
        for (progress.overrides) |o| {
            try overrides.append(allocator, .{ .base = try allocator.dupe(u8, o.base), .bits = o.bits, .group = o.group });
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
    if (progress.ngram_present) ngram_spec = .{};

    var k_buf: [16]u8 = undefined;
    const k_text = opts.rate.kText(&k_buf);
    var tok_buf: [24]u8 = undefined;
    var rows_buf: [24]u8 = undefined;
    const provenance = [_][2][]const u8{
        .{ "encoder", opts.version },
        .{ "quantizer", "ldlq-rotated" },
        .{ "global_scale_search", "on" },
        .{ "calib_sha256", opts.calib_sha256 },
        .{ "calib_tokens", try std.fmt.bufPrint(&tok_buf, "{d}", .{opts.calib_tokens}) },
        .{ "hessian_min_rows", try std.fmt.bufPrint(&rows_buf, "{d}", .{opts.min_rows}) },
    };
    const policy = convert.Policy{ .rate = opts.rate, .dec = opts.dec, .min_rows = opts.min_rows, .batch = opts.batch, .factor_precision = opts.factor_precision, .provenance = &provenance };
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
            .rate = opts.rate,
            .dec = opts.dec,
            .trunk_bits = trunk_bits,
            .trunk_group = trunk_group,
            .overrides = overrides.items,
            .has_ngram = ngram_spec != null,
            .eq = .{ .k_text = k_text, .codebook = @tagName(opts.dec.codebook), .window = opts.dec.window.bits() },
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
        if (hidden_ready and layerShardComplete(allocator, io, out, name, shard.trunk_prefix, layer)) {
            summary.layers_reused += 1;
            if (calibrator) |c| try c.done(c.ctx, layer, false, true);
            say(sink, "layer {d}: reused", .{layer});
            continue;
        }
        out.deleteFile(io, name) catch {};
        const begin_at = stages_mod.nowNs();
        const hs = if (calibrator) |c| try c.begin(c.ctx, layer, false) else null;
        const convert_at = stages_mod.nowNs();
        const stats = try convert.convertQwenLayer(allocator, s, io, out, &src, layer, false, geometry, policy, hs);
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

    const sidecars = try pack.copySidecars(allocator, io, src_dir, out);
    say(sink, "sidecars: {d} files copied", .{sidecars});
    const index = try indexOfDir(allocator, io, out);
    defer allocator.free(index);
    try out.writeFile(io, .{ .sub_path = ".model.safetensors.index.json.tmp", .data = index });
    try out.rename(".model.safetensors.index.json.tmp", out, "model.safetensors.index.json", io);

    const config = try pack.buildConfig(allocator, config_json, trunk_bits, trunk_group, overrides.items, ngram_spec, .{
        .k_text = k_text,
        .codebook = @tagName(opts.dec.codebook),
        .window = opts.dec.window.bits(),
        .provenance = &.{
            .{ "encoder", opts.version },
            .{ "quantizer", "ldlq-rotated" },
            .{ "calib_sha256", opts.calib_sha256 },
            .{ "calib_tokens", try std.fmt.bufPrint(&tok_buf, "{d}", .{opts.calib_tokens}) },
        },
    });
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

    fn begin(ctx: *anyopaque, layer: u32, mtp: bool) anyerror!?convert.HessianSource {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        if (mtp) return null;
        if (self.fail_on) |f| if (f == layer) return error.InjectedFailure;
        try self.begun.append(std.testing.allocator, layer);
        try self.kinds.append(std.testing.allocator, self.info.?.layer_full[layer]);
        return null;
    }

    fn done(_: *anyopaque, _: u32, _: bool, _: bool) anyerror!void {}

    fn prepare(ctx: *anyopaque, info: Info) anyerror!void {
        const self: *Fake = @ptrCast(@alignCast(ctx));
        self.info = info;
    }

    fn ready(_: *anyopaque, _: u32, _: bool) bool {
        return true;
    }

    fn finish(_: *anyopaque) void {}

    fn deinit(self: *Fake) void {
        self.begun.deinit(std.testing.allocator);
        self.kinds.deinit(std.testing.allocator);
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
    };
}

pub fn tmpPath(allocator: std.mem.Allocator, tmp: *std.testing.TmpDir, sub: []const u8) ![]u8 {
    const io = std.testing.io;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &buf);
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ buf[0..n], sub });
}

test "driver: a tiny Qwen source becomes a complete pack, and a second run into it is refused" {
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
}

test "driver: a run that fails after layer 0 resumes at layer 1, and a changed policy refuses to resume" {
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
