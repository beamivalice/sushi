//! Qwen3.8-Flash-Next's dense trunk rounded onto the trellis (STQ, SPEC1 w15) in a pack whose routed experts stay as
//! they are. The pack's expert shards, routers, embedding, n-gram table, MTP head and vision tower are cloned; the
//! linears of each layer (attention, GDN, shared expert, the PLE pair) are K4 banks and `lm_head` a K6 bank, each
//! rounded by LDLQ on the Gram of its input as the model computes it: layers are rounded in order, every layer's
//! input is the previous layer's output through the new trunk and the pack's experts (`Capture.stqLayer`). What a
//! 128 trellis tile cannot hold (the GDN `a` and `b` gates, the hyper-connection mixes) stays as the pack stores it.
const std = @import("std");
const api = @import("sushi_exl3");
const capture_mod = @import("capture.zig");
const driver = @import("driver.zig");
const encode = @import("encode.zig");
const pack = @import("pack.zig");
const source = @import("source.zig");
const trunk_from = @import("trunk_from.zig");
const trunk_mod = @import("trunk.zig");
const stages_mod = @import("stages.zig");

pub const Error = error{
    ArchitectureNotConvertible,
    TrunkBaseUnsupported,
    TrunkBaseAlreadyStq,
    TrunkBaseShardMissing,
    ResumeNothingToResume,
    ResumePolicyMismatch,
    ResumePackComplete,
    ResumeBoundaryMissing,
    OutputNotEmpty,
    PinPassUnsupported,
} || std.mem.Allocator.Error;

pub const Options = struct {
    /// The BF16 checkpoint the trunk is rounded from.
    model_dir: []const u8,
    /// The pack whose experts and every other tensor stay; its trunk is the GPTQ-rounded affine one (`trunk-LNN`).
    base_dir: []const u8,
    out_dir: []const u8,
    stq: trunk_mod.Stq = .{},
    calib_sha256: []const u8 = "none",
    calib_tokens: usize = 0,
    resuming: bool = false,
    version: []const u8,
    /// Round only the first `n` layers and stop: no head, no config (a timing or test run).
    only_layers: ?u32 = null,
};

pub const Summary = struct {
    layers_written: usize = 0,
    layers_resumed: usize = 0,
    banks: usize = 0,
    head_written: bool = false,
};

pub const Sink = driver.Sink;

fn say(sink: ?Sink, comptime fmt: []const u8, args: anytype) void {
    const s = sink orelse return;
    var buf: [512]u8 = undefined;
    const line = std.fmt.bufPrint(&buf, fmt, args) catch return;
    s.write(s.ctx, line);
}

const progress_name = "convert-progress.json";

fn policyKey(allocator: std.mem.Allocator, opts: Options, base_config: []const u8) ![]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(base_config, &digest, .{});
    return std.fmt.allocPrint(allocator, "qwen-stq-trunk-1|base {x}|trunk n{d} head n{d}|pin {d}|calib {s} {d}", .{
        digest[0..8], opts.stq.rate.n, opts.stq.head_rate.n, opts.stq.pin_pass, opts.calib_sha256, opts.calib_tokens,
    });
}

fn exists(io: std.Io, dir: std.Io.Dir, name: []const u8) bool {
    _ = dir.statFile(io, name, .{}) catch return false;
    return true;
}

/// True when `name` holds a whole safetensors file: its header parses and the data it declares is all there.
fn shardWhole(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) bool {
    var h = trunk_from.readHeader(allocator, io, dir, name) catch return false;
    defer h.parsed.deinit();
    var end: u64 = 0;
    var it = h.root.iterator();
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
        const t = trunk_from.Tensor.of(allocator, e.value_ptr.*) catch return false;
        defer t.deinit(allocator);
        end = @max(end, t.hi);
    }
    var f = dir.openFile(io, name, .{}) catch return false;
    defer f.close(io);
    const stat = f.stat(io) catch return false;
    return stat.size == 8 + h.len + end;
}

/// The base of a tensor name: everything before the last dot.
fn baseOf(name: []const u8) []const u8 {
    const dot = std.mem.lastIndexOfScalar(u8, name, '.') orelse return name;
    return name[0..dot];
}

/// Rewrites shard `file` of `dir` without the tensors of the given bases (`<base>.weight|scales|biases`), keeping
/// the rest byte for byte and the shard's metadata. Returns how many tensors it dropped.
pub fn dropBases(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, file: []const u8, bases: []const []const u8) !usize {
    return trunk_from.rewriteWithout(allocator, io, dir, file, bases, struct {
        fn drop(names: []const []const u8, name: []const u8) bool {
            const own = baseOf(name);
            for (names) |base| if (std.mem.eql(u8, base, own)) return true;
            return false;
        }
    }.drop);
}

/// `config` with the converted bases removed from its affine `quantization` blocks and a `trunk_quant` block added.
pub fn buildConfig(allocator: std.mem.Allocator, base_config: []const u8, converted: []const []const u8, stq: trunk_mod.Stq, provenance: []const [2][]const u8) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var root = (try std.json.parseFromSliceLeaky(std.json.Value, a, base_config, .{})).object;
    for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
        const q = root.getPtr(key) orelse continue;
        if (q.* != .object) continue;
        for (converted) |base| _ = q.object.orderedRemove(base);
    }
    const dec = stq.dec();
    const table = &api.codebook_tables.tables[dec.table.?];
    var k_buf: [16]u8 = undefined;
    var tq = std.json.ObjectMap.empty;
    try tq.put(a, "format", .{ .string = "stq" });
    const k_text = try a.dupe(u8, (api.format.Rate{ .n = @max(stq.rate.n, stq.head_rate.n) }).kText(&k_buf));
    try tq.put(a, "k", .{ .number_string = k_text });
    try tq.put(a, "codebook", .{ .string = @tagName(dec.codebook) });
    try tq.put(a, "window", .{ .integer = dec.window.bits() });
    try tq.put(a, "codebook_table", .{ .string = table.name });
    try tq.put(a, "codebook_digest", .{ .string = table.digest });
    for (provenance) |pair| try tq.put(a, pair[0], .{ .string = pair[1] });
    try root.put(a, "trunk_quant", .{ .object = tq });
    return std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = root }, .{ .whitespace = .indent_2 });
}

const Progress = struct { policy: []const u8 };

fn writeProgress(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, p: Progress) !void {
    const text = try std.json.Stringify.valueAlloc(allocator, p, .{});
    defer allocator.free(text);
    try dir.writeFile(io, .{ .sub_path = ".convert-progress.tmp", .data = text });
    try dir.rename(".convert-progress.tmp", dir, progress_name, io);
}

/// The highest `hidden-N.bin` the capture scratch holds: the boundary a resumed run goes on from.
fn storedBoundary(io: std.Io, out: std.Io.Dir) ?u32 {
    var scratch = out.openDir(io, ".capture", .{ .iterate = true }) catch return null;
    defer scratch.close(io);
    var best: ?u32 = null;
    var it = scratch.iterate();
    while (it.next(io) catch return best) |e| {
        if (e.kind != .file or !std.mem.startsWith(u8, e.name, "hidden-") or !std.mem.endsWith(u8, e.name, ".bin")) continue;
        const at = std.fmt.parseInt(u32, e.name["hidden-".len .. e.name.len - ".bin".len], 10) catch continue;
        if (best == null or at > best.?) best = at;
    }
    return best;
}

fn seconds(from: u64, to: u64) f64 {
    return @as(f64, @floatFromInt(to - from)) / 1e9;
}

pub fn run(allocator: std.mem.Allocator, io: std.Io, opts: Options, cap: *capture_mod.Capture, sink: ?Sink) !Summary {
    const cwd = std.Io.Dir.cwd();
    if (!encode.validPinPass(opts.stq.pin_pass)) return error.PinPassUnsupported;
    var src_dir = try cwd.openDir(io, opts.model_dir, .{ .iterate = true });
    defer src_dir.close(io);
    if (try source.identifySourceDir(allocator, io, src_dir) != .qwen4_exp) return error.ArchitectureNotConvertible;
    const source_config = try src_dir.readFileAlloc(io, "config.json", allocator, .limited(16 << 20));
    defer allocator.free(source_config);
    const geom = try driver.readGeometry(allocator, source_config, cap.src);

    var base = try cwd.openDir(io, opts.base_dir, .{ .iterate = true });
    defer base.close(io);
    var served = try trunk_from.read(allocator, io, base);
    defer served.deinit();
    const base_parsed = try std.json.parseFromSlice(std.json.Value, allocator, served.config, .{});
    defer base_parsed.deinit();
    if (base_parsed.value.object.contains("trunk_quant")) return error.TrunkBaseAlreadyStq;
    const experts = try api.parseExpertQuant(base_parsed.value.object);
    const dec = api.format.Decode{ .codebook = experts.codebook, .window = experts.window, .table = experts.table };

    // The GPTQ trunk of a `--trunk gptq` pack lives in `trunk-LNN` and `trunk-head`; they are what this rewrites.
    var layer: u32 = 0;
    while (layer < geom.layers) : (layer += 1) {
        var nb: [64]u8 = undefined;
        if (!exists(io, base, try trunk_mod.shardName(&nb, layer))) return error.TrunkBaseUnsupported;
        var eb: [64]u8 = undefined;
        if (!exists(io, base, try std.fmt.bufPrint(&eb, "layer-{d}.safetensors", .{layer}))) return error.TrunkBaseShardMissing;
    }
    if (!exists(io, base, trunk_mod.head_shard)) return error.TrunkBaseUnsupported;

    try cwd.createDirPath(io, opts.out_dir);
    var out = try cwd.openDir(io, opts.out_dir, .{ .iterate = true });
    defer out.close(io);
    var empty = true;
    {
        var it = out.iterate();
        empty = (try it.next(io)) == null;
    }
    if (!empty and !opts.resuming) return error.OutputNotEmpty;
    const key = try policyKey(allocator, opts, served.config);
    defer allocator.free(key);
    if (empty) {
        try writeProgress(allocator, io, out, .{ .policy = key });
    } else {
        if (exists(io, out, "config.json")) return error.ResumePackComplete;
        const raw = out.readFileAlloc(io, progress_name, allocator, .limited(1 << 20)) catch return error.ResumeNothingToResume;
        defer allocator.free(raw);
        const stored = std.json.parseFromSlice(Progress, allocator, raw, .{}) catch return error.ResumeNothingToResume;
        defer stored.deinit();
        if (!std.mem.eql(u8, stored.value.policy, key)) return error.ResumePolicyMismatch;
    }

    // Every shard of the base is cloned (copy on write): the experts and everything else stay as they are, and the
    // base's trunk shards are what a layer's staging reads until its own are rewritten.
    var cloned: usize = 0;
    {
        var it = base.iterate();
        while (try it.next(io)) |e| {
            if (e.kind != .file or e.name[0] == '.') continue;
            const wanted = std.mem.endsWith(u8, e.name, ".safetensors") or std.mem.eql(u8, e.name, "ngram_table.bin");
            if (!wanted or exists(io, out, e.name)) continue;
            try trunk_from.cloneOrCopy(io, base, out, e.name);
            cloned += 1;
        }
    }
    if (cloned != 0) say(sink, "trunk: {d} shards of the base cloned", .{cloned});

    var out_abs_buf: [std.fs.max_path_bytes]u8 = undefined;
    const out_abs = out_abs_buf[0..try out.realPath(io, &out_abs_buf)];
    const kinds = try driver.layerKinds(allocator, source_config, geom.layers);
    defer allocator.free(kinds);
    var k_buf: [16]u8 = undefined;
    const c = cap.calibrator();
    try c.prepare(c.ctx, .{
        .pack_dir = out,
        .pack_abs = out_abs,
        .source_dir_abs = opts.model_dir,
        .source_config = source_config,
        .geom = .{ .experts = geom.experts, .hidden = geom.hidden, .inter = geom.inter },
        .layers = geom.layers,
        .layer_full = kinds,
        .ple_layer = geom.ple_layer,
        .rate = experts.rate,
        .dec = dec,
        .trunk_bits = served.bits,
        .trunk_group = served.group,
        .overrides = served.overrides,
        .ngram = served.ngram,
        .eq = pack.expertQuant(dec, experts.rate.kText(&k_buf), &.{}),
        .trunk_stq = opts.stq,
    });
    defer c.finish(c.ctx);

    var summary = Summary{};
    var converted = std.ArrayList([]u8).empty;
    defer {
        for (converted.items) |n| allocator.free(n);
        converted.deinit(allocator);
    }
    var tok_buf: [24]u8 = undefined;
    var pin_buf: [16]u8 = undefined;
    const provenance = [_][2][]const u8{
        .{ "encoder", opts.version },
        .{ "quantizer", "ldlq-rotated" },
        .{ "calib_sha256", opts.calib_sha256 },
        .{ "calib_tokens", try std.fmt.bufPrint(&tok_buf, "{d}", .{opts.calib_tokens}) },
        .{ "pin_pass", try std.fmt.bufPrint(&pin_buf, "{d}", .{opts.stq.pin_pass}) },
    };

    // A resumed run goes on from the newest stored boundary; every layer before it has its shards.
    const first: u32 = storedBoundary(io, out) orelse 0;
    if (first > geom.layers) return error.ResumeBoundaryMissing;
    var at: u32 = 0;
    while (at < first) : (at += 1) {
        var nb: [64]u8 = undefined;
        if (!shardWhole(allocator, io, out, try trunk_mod.stqShardName(&nb, at))) return error.ResumeBoundaryMissing;
        summary.layers_resumed += 1;
    }
    if (first != 0) say(sink, "resume: layers 0 to {d} are done, the boundary of layer {d} is stored", .{ first - 1, first });

    const last: u32 = if (opts.only_layers) |n| @min(n, geom.layers) else geom.layers;
    layer = first;
    while (layer < last) : (layer += 1) {
        const t0 = stages_mod.nowNs();
        try cap.stqLayer(layer);
        const t1 = stages_mod.nowNs();
        var nb: [64]u8 = undefined;
        const trunk = &cap.trunk.?;
        const banks = try trunk.writeBanks(io, out, try trunk_mod.stqShardName(&nb, layer), &provenance);
        var kb: [64]u8 = undefined;
        var names = std.ArrayList([]const u8).empty;
        defer names.deinit(allocator);
        for (trunk.tensors.items) |t| try names.append(allocator, t.base);
        _ = try dropBases(allocator, io, out, try trunk_mod.shardName(&kb, layer), names.items);
        for (trunk.tensors.items) |t| {
            const owned = try allocator.dupe(u8, t.base);
            errdefer allocator.free(owned);
            try converted.append(allocator, owned);
        }
        summary.banks += banks;
        const t2 = stages_mod.nowNs();
        try cap.stqAdvance(layer);
        summary.layers_written += 1;
        say(sink, "layer {d}: {d} banks; trunk {d:.1}s, shards {d:.1}s, advance {d:.1}s", .{
            layer,
            banks,
            seconds(t0, t1),
            seconds(t1, t2),
            seconds(t2, stages_mod.nowNs()),
        });
    }
    if (opts.only_layers != null) return summary;

    // The names of the layers this run did not round come from their shards.
    for (0..first) |l| {
        var nb: [64]u8 = undefined;
        try namesOfBanks(allocator, io, out, try trunk_mod.stqShardName(&nb, @intCast(l)), &converted);
    }
    {
        const t0 = stages_mod.nowNs();
        var head = try cap.stqHead();
        defer head.deinit();
        const banks = try head.writeBanks(io, out, trunk_mod.stq_head_shard, &provenance);
        var names = std.ArrayList([]const u8).empty;
        defer names.deinit(allocator);
        for (head.tensors.items) |t| if (t.bank != null) try names.append(allocator, t.base);
        _ = try dropBases(allocator, io, out, trunk_mod.head_shard, names.items);
        for (names.items) |n| {
            const owned = try allocator.dupe(u8, n);
            errdefer allocator.free(owned);
            try converted.append(allocator, owned);
        }
        summary.banks += banks;
        summary.head_written = true;
        say(sink, "head: {d} bank, {d:.1}s", .{ banks, seconds(t0, stages_mod.nowNs()) });
    }

    const sidecars = try pack.copySidecars(allocator, io, base, out);
    out.deleteFile(io, "README.md") catch {};
    say(sink, "sidecars: {d} files copied", .{sidecars});
    const index = try driver.indexOfDir(allocator, io, out);
    defer allocator.free(index);
    try out.writeFile(io, .{ .sub_path = ".model.safetensors.index.json.tmp", .data = index });
    try out.rename(".model.safetensors.index.json.tmp", out, "model.safetensors.index.json", io);

    const config = try buildConfig(allocator, served.config, converted.items, opts.stq, &provenance);
    defer allocator.free(config);
    try out.writeFile(io, .{ .sub_path = ".config.json.tmp", .data = config });
    try out.rename(".config.json.tmp", out, "config.json", io);
    out.deleteFile(io, progress_name) catch {};
    out.deleteTree(io, ".capture") catch {};
    say(sink, "pack complete", .{});
    return summary;
}

/// Adds the bases of the `.trellis` tensors `file` holds to `names`.
fn namesOfBanks(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, file: []const u8, names: *std.ArrayList([]u8)) !void {
    var h = try trunk_from.readHeader(allocator, io, dir, file);
    defer h.parsed.deinit();
    var it = h.root.iterator();
    while (it.next()) |e| {
        if (!std.mem.endsWith(u8, e.key_ptr.*, ".trellis")) continue;
        const base = try allocator.dupe(u8, baseOf(e.key_ptr.*));
        errdefer allocator.free(base);
        try names.append(allocator, base);
    }
}

fn writeAffine(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, file: []const u8, bases: []const []const u8) !void {
    var b = pack.ShardBuilder.init(allocator);
    defer b.deinit();
    for (bases, 0..) |base, i| {
        var nb: [96]u8 = undefined;
        const w: [4]u8 = @splat(@intCast(i + 1));
        try b.add(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}), "U32", &.{1}, &w);
        try b.add(try std.fmt.bufPrint(&nb, "{s}.scales", .{base}), "BF16", &.{1}, w[0..2]);
        try b.add(try std.fmt.bufPrint(&nb, "{s}.biases", .{base}), "BF16", &.{1}, w[0..2]);
    }
    try b.flush(io, dir, file, &.{.{ "format", "mlx" }});
}

test "trunk stq: a kept shard loses the converted bases and nothing else, and a second drop finds nothing" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try writeAffine(t.allocator, io, work.dir, "trunk-L00.safetensors", &.{ "m.a_proj", "m.b_proj", "m.gate" });
    const before = try work.dir.readFileAlloc(io, "trunk-L00.safetensors", t.allocator, .limited(1 << 20));
    defer t.allocator.free(before);
    var f0 = try @import("safetensors.zig").parse(t.allocator, before);
    defer f0.deinit(t.allocator);

    try t.expectEqual(@as(usize, 6), try dropBases(t.allocator, io, work.dir, "trunk-L00.safetensors", &.{ "m.a_proj", "m.b_proj" }));
    const after = try work.dir.readFileAlloc(io, "trunk-L00.safetensors", t.allocator, .limited(1 << 20));
    defer t.allocator.free(after);
    var f1 = try @import("safetensors.zig").parse(t.allocator, after);
    defer f1.deinit(t.allocator);
    try t.expectEqual(@as(usize, 3), f1.tensors.count());
    for ([_][]const u8{ "m.gate.weight", "m.gate.scales", "m.gate.biases" }) |n| {
        try t.expectEqualSlices(u8, f0.tensors.get(n).?.bytes, f1.tensors.get(n).?.bytes);
    }
    try t.expectEqualStrings("mlx", f1.metadata.get("format").?);
    try t.expectEqual(@as(usize, 0), try dropBases(t.allocator, io, work.dir, "trunk-L00.safetensors", &.{ "m.a_proj", "m.b_proj" }));

    // A shard with nothing left goes away.
    try t.expectEqual(@as(usize, 3), try dropBases(t.allocator, io, work.dir, "trunk-L00.safetensors", &.{"m.gate"}));
    try t.expect(!exists(io, work.dir, "trunk-L00.safetensors"));
}

test "trunk stq: the config drops the converted bases and gains a trunk_quant block the engine's parser takes" {
    const t = std.testing;
    const base =
        \\{"model_type":"qwen4_exp","quantization":{"group_size":64,"bits":8,"mode":"affine",
        \\"language_model.model.layers.0.linear_attn.in_proj_qkv":{"bits":8,"group_size":128,"mode":"affine"},
        \\"language_model.model.layers.0.linear_attn.in_proj_a":{"bits":8,"group_size":128,"mode":"affine"}},
        \\"quantization_config":{"group_size":64,"bits":8,"mode":"affine",
        \\"language_model.model.layers.0.linear_attn.in_proj_qkv":{"bits":8,"group_size":128,"mode":"affine"},
        \\"language_model.model.layers.0.linear_attn.in_proj_a":{"bits":8,"group_size":128,"mode":"affine"}},
        \\"expert_quant":{"format":"stq","k":4,"codebook":"spec1","window":15,"codebook_table":"spec1-w15","codebook_digest":"37f72431"}}
    ;
    const text = try buildConfig(t.allocator, base, &.{"language_model.model.layers.0.linear_attn.in_proj_qkv"}, .{}, &.{.{ "encoder", "test" }});
    defer t.allocator.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, text, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
        const q = root.get(key).?.object;
        try t.expect(q.get("language_model.model.layers.0.linear_attn.in_proj_qkv") == null);
        try t.expect(q.get("language_model.model.layers.0.linear_attn.in_proj_a") != null);
        try t.expectEqual(@as(i64, 8), q.get("bits").?.integer);
    }
    const experts = try api.parseExpertQuant(root);
    try t.expectEqual(@as(u32, 64), experts.rate.n);
    const trunk = try api.parseQuantBlock(root.get("trunk_quant").?);
    try t.expectEqual(api.format.Codebook.spec1, trunk.codebook);
    try t.expectEqual(api.format.Window.w15, trunk.window);
    try t.expectEqual(@as(u32, 96), trunk.rate.n);
    try t.expectEqual(experts.table, trunk.table);
    try t.expectEqualStrings("test", root.get("trunk_quant").?.object.get("encoder").?.string);
}
