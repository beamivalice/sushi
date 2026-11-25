//! DeepSeek-V4.1 (`deepseek_v41`) conversion from a stored calibration capture (`sushi kld capture --layer-major`'s
//! `calib/`): no forward runs for the backbone. Each layer's routed experts take their gate/up Hessians from the
//! routed rows of `mlp-in-LL` (`route-LL`), their down Hessians from the source expert's clamped SwiGLU over the same
//! rows, then the LDLQ and the PAIR1 search at the planned rates, written as rate groups with the router in group
//! order. The trunk linears take their Hessians from the stored X^T X sums (`trunk-LL`) and round to SPEC1 K4, the
//! head to K6, the embedding to MLX affine 4-bit. The DSpark stages are calibrated in process: their forward runs over
//! the stored main rows (`dspark-in-LL`), and their experts round to PAIR1 K2, their trunk to the trunk's rate.
const std = @import("std");
const io_util = @import("mlx_host").io_util;
const host = @import("mlx_host");
const mlx = host.mlx;
const api = @import("sushi_exl3");
const format = api.format;
const codebook_tables = api.codebook_tables;
const source = @import("source.zig");
const convert = @import("convert.zig");
const hessians = @import("hessians.zig");
const metal_search = @import("metal_search.zig");
const quantize = @import("quantize.zig");
const pack = @import("pack.zig");
const st = @import("safetensors.zig");
const rate_plan = @import("rate_plan.zig");
const stages_mod = @import("stages.zig");
const encode = @import("encode.zig");
const pool_mod = @import("pool.zig");

const dsv = host.dsv41_model;
const dsv_cfg = host.dsv41_config;
const A = mlx.mlx_array;

/// The recipe every DeepSeek-V4.1 STQ pack of this converter follows, unless `Options` says another rate.
pub const default_trunk_rate = format.Rate{ .n = 64 };
pub const head_rate = format.Rate{ .n = 96 };
pub const default_stage_rate = format.Rate{ .n = 32 };
pub const embed_bits: u32 = 4;
pub const embed_group: u32 = 64;
pub const trunk_dec = format.Decode{ .codebook = .spec1, .window = .w15, .table = codebook_tables.indexOf("spec1-w15") };
/// The seed the expert converter's signs use (`convert.Policy.seed`).
const sign_seed: u64 = 0x5eed_e3c0_de00_0001;
pub const expert_dec = format.Decode{ .codebook = .pair1, .window = .w16, .table = codebook_tables.indexOf("pair1-w16-v1") };

pub const Options = struct {
    model_dir: []const u8,
    out_dir: []const u8,
    /// The capture's `calib/` directory.
    capture_dir: []const u8,
    /// The routed experts' rate where the plan names none.
    rate: format.Rate,
    plan: ?*const rate_plan.Plan = null,
    pin_pass: u32 = encode.default_pin_pass,
    batch: usize = 16,
    min_rows: u64 = 64,
    resuming: bool = false,
    version: []const u8,
    /// Only these backbone layers and nothing else (a timing run); the pack is left without `config.json`.
    only_layers: ?[]const u32 = null,
    /// Every `anchor_stride`-th position of a window drafts a DSpark block in the stages' calibration.
    anchor_stride: usize = 8,
    /// Anchors per DSpark calibration forward.
    anchor_batch: usize = 128,
    /// The rate of every block's trunk banks; the head stays `head_rate`.
    trunk_rate: format.Rate = default_trunk_rate,
    /// The DSpark stages' trunk banks' rate; null: `trunk_rate`.
    drafter_rate: ?format.Rate = null,
    /// The DSpark stages' routed experts' rate (PAIR1).
    drafter_experts_rate: format.Rate = default_stage_rate,

    fn drafterTrunkRate(self: *const Options) format.Rate {
        return self.drafter_rate orelse self.trunk_rate;
    }
};

pub const Summary = struct {
    layers_written: usize = 0,
    layers_reused: usize = 0,
    stages_written: usize = 0,
    gate_up_identity: usize = 0,
    down_identity: usize = 0,
    trunk_identity: usize = 0,
};

pub const Sink = struct {
    ctx: ?*anyopaque = null,
    write: *const fn (?*anyopaque, []const u8) void,
};

fn say(sink: ?Sink, comptime fmt: []const u8, args: anytype) void {
    const s = sink orelse return;
    var buf: [512]u8 = undefined;
    s.write(s.ctx, std.fmt.bufPrint(&buf, fmt, args) catch return);
}

const progress_name = "convert-progress.json";

/// The trunk linears of a block, each with the stored input it reads (the capture's sum names).
const Linear = struct { tail: []const u8, gram: []const u8 };
const block_linears = [_]Linear{
    .{ .tail = "attn.wq_a", .gram = "attn_in" },
    .{ .tail = "attn.wkv", .gram = "attn_in" },
    .{ .tail = "attn.compressor.wkv", .gram = "attn_in" },
    .{ .tail = "attn.compressor.wgate", .gram = "attn_in" },
    .{ .tail = "attn.indexer.weights_proj", .gram = "attn_in" },
    .{ .tail = "attn.wq_b", .gram = "q_lora" },
    .{ .tail = "attn.indexer.wq_b", .gram = "q_lora" },
    .{ .tail = "attn.indexer.wk", .gram = "index_latent" },
    .{ .tail = "attn.wo_b", .gram = "wo_b_in" },
    .{ .tail = "ffn.shared_experts.w1", .gram = "ffn_in" },
    .{ .tail = "ffn.shared_experts.w3", .gram = "ffn_in" },
    .{ .tail = "ffn.shared_experts.w2", .gram = "shared_act" },
    .{ .tail = "engram.wkv", .gram = "engram_in" },
    .{ .tail = "main_proj", .gram = "main_in" },
};

// ── files ────────────────────────────────────────────────────────────────────────────────────────────

fn layerExpertsFile(buf: []u8, layer: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "model-layer-{d:0>2}.safetensors", .{layer});
}

fn layerTrunkFile(buf: []u8, layer: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "model-trunk-{d:0>2}.safetensors", .{layer});
}

fn stageExpertsFile(buf: []u8, stage: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "model-mtp-{d}.safetensors", .{stage});
}

fn stageTrunkFile(buf: []u8, stage: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "model-mtp-trunk-{d}.safetensors", .{stage});
}

const globals_file = "model-globals.safetensors";

fn exists(io: std.Io, dir: std.Io.Dir, name: []const u8) bool {
    _ = dir.statFile(io, name, .{}) catch return false;
    return true;
}

/// A file's bytes as a GPU array `[rows, cols]` without a copy where Metal takes the mapping.
pub fn fileArray(a: std.mem.Allocator, path: []const u8, rows: usize, cols: usize, dtype: mlx.mlx_dtype, itemsize: usize) !A {
    const z = try a.dupeSentinel(u8, path, 0);
    defer a.free(z);
    const fd = std.c.open(z.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.CaptureFileMissing;
    defer _ = std.c.close(fd);
    var reader = host.expert_io.ParallelReader.init(a, host.expert_io.ParallelReader.default_chunk, host.expert_io.ParallelReader.default_workers);
    defer reader.deinit();
    const shape = [_]c_int{ @intCast(rows), @intCast(cols) };
    const t = try host.model.readTensorArray(&reader, fd, 0, rows * cols * itemsize, &shape, dtype);
    return t.array;
}

pub fn fileSize(a: std.mem.Allocator, path: []const u8) !u64 {
    const z = try a.dupeSentinel(u8, path, 0);
    defer a.free(z);
    const fd = std.c.open(z.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.CaptureFileMissing;
    defer _ = std.c.close(fd);
    var s: std.c.Stat = undefined;
    if (std.c.fstat(fd, &s) != 0) return error.CaptureFileMissing;
    return @intCast(s.size);
}

fn readWhole(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    const len = try fileSize(a, path);
    const buf = try a.alloc(u8, len);
    errdefer a.free(buf);
    var f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    try pack.readAllChunked(f, io, buf, 0, pack.read_chunk);
    return buf;
}

/// Routed ids `[tokens * top_k]` from a `route-*.u16` file.
pub fn readRoutes(a: std.mem.Allocator, io: std.Io, path: []const u8, experts: usize) ![]i32 {
    const raw = try readWhole(a, io, path);
    defer a.free(raw);
    const ids = try a.alloc(i32, raw.len / 2);
    errdefer a.free(ids);
    for (ids, 0..) |*o, i| {
        const v = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
        if (v >= experts) return error.CaptureRouteOutOfRange;
        o.* = v;
    }
    return ids;
}

// ── the stored X^T X sums ───────────────────────────────────────────────────────────────────────────

pub const Sums = struct {
    raw: []u8,
    file: st.File,

    pub fn open(a: std.mem.Allocator, io: std.Io, path: []const u8) !Sums {
        const raw = try readWhole(a, io, path);
        errdefer a.free(raw);
        return .{ .raw = raw, .file = try st.parse(a, raw) };
    }

    pub fn deinit(self: *Sums, a: std.mem.Allocator) void {
        self.file.deinit(a);
        a.free(self.raw);
    }

    /// `name`'s sum (f32) and its row count, or null when the capture holds none.
    pub fn get(self: *const Sums, name: []const u8) ?struct { gram: []const f32, rows: u64 } {
        var buf: [192]u8 = undefined;
        const g = self.file.tensors.get(name) orelse return null;
        const r = self.file.tensors.get(std.fmt.bufPrint(&buf, "{s}.rows", .{name}) catch return null) orelse return null;
        return .{ .gram = st.slice(f32, g), .rows = @intCast(st.slice(i64, r)[0]) };
    }
};

// ── one dense trellis linear ────────────────────────────────────────────────────────────────────────

pub const Dense = struct {
    trellis: []u16,
    suh: []u16,
    svh: []u16,

    pub fn deinit(self: *Dense, a: std.mem.Allocator) void {
        a.free(self.trellis);
        a.free(self.suh);
        a.free(self.svh);
    }
};

/// One tensor `w` (public `[in, out]`) rounded with input Hessian `h` (identity when null), signs seeded by `name`.
pub fn quantizeDense(a: std.mem.Allocator, s: mlx.mlx_stream, pool: *pool_mod.Pool, w: []const f32, h_in: ?[]const f32, in: usize, out: usize, rate: format.Rate, pin_pass: u32, name: []const u8, stages: *stages_mod.Stages) !Dense {
    const h = try a.alloc(f32, in * in);
    defer a.free(h);
    if (h_in) |given| @memcpy(h, given) else {
        @memset(h, 0);
        for (0..in) |i| h[i * in + i] = 1;
    }
    const signs = try quantize.signs(a, sign_seed, name, in + out);
    defer a.free(signs);
    const job = [_]metal_search.TensorJob{.{ .w = w, .h = h, .sign_in = signs[0..in], .sign_out = signs[in..] }};
    var prepared = try metal_search.prepareBatch(a, pool, &job, in, out, .auto, .f64);
    defer prepared.deinit();
    var result: [1]quantize.Quantized = undefined;
    try metal_search.finishBatch(a, s, pool, &prepared, rate, trunk_dec, pin_pass, stages, &result, null);
    return .{ .trellis = result[0].packed_out, .suh = result[0].suh, .svh = result[0].svh };
}

/// A trunk shard under construction: the converted linears and every tensor kept as stored.
const TrunkWriter = struct {
    a: std.mem.Allocator,
    s: mlx.mlx_stream,
    src: *const source.SafetensorsSource,
    pool: *pool_mod.Pool,
    opts: *const Options,
    builder: pack.ShardBuilder,
    /// Source tensors already written (converted or rewritten); the rest of the unit's prefix is copied.
    done: std.StringHashMap(void),
    stages: stages_mod.Stages = .{},
    identity: usize = 0,
    min_rows: u64,
    /// The rate of this shard's trunk banks.
    rate: format.Rate,

    fn init(a: std.mem.Allocator, s: mlx.mlx_stream, src: *const source.SafetensorsSource, pool: *pool_mod.Pool, opts: *const Options, rate: format.Rate) TrunkWriter {
        return .{ .a = a, .s = s, .src = src, .pool = pool, .opts = opts, .builder = pack.ShardBuilder.init(a), .done = std.StringHashMap(void).init(a), .min_rows = opts.min_rows, .rate = rate };
    }

    fn deinit(self: *TrunkWriter) void {
        var it = self.done.keyIterator();
        while (it.next()) |k| self.a.free(k.*);
        self.done.deinit();
        self.builder.deinit();
    }

    fn mark(self: *TrunkWriter, name: []const u8) !void {
        if (self.done.contains(name)) return;
        const key = try self.a.dupe(u8, name);
        errdefer self.a.free(key);
        try self.done.put(key, {});
    }

    fn markBase(self: *TrunkWriter, base: []const u8) !void {
        var buf: [192]u8 = undefined;
        try self.mark(try std.fmt.bufPrint(&buf, "{s}.weight", .{base}));
        try self.mark(try std.fmt.bufPrint(&buf, "{s}.scale", .{base}));
    }

    fn gramFor(self: *TrunkWriter, sums: ?*const Sums, key: []const u8) ?[]const f32 {
        const sm = sums orelse return null;
        const g = sm.get(key) orelse return null;
        if (g.rows < self.min_rows) return null;
        return g.gram;
    }

    fn addBank(self: *TrunkWriter, base: []const u8, d: Dense, shape_lead: []const usize, in: usize, out: usize, rate: format.Rate) !void {
        var buf: [192]u8 = undefined;
        var shape: [5]usize = undefined;
        @memcpy(shape[0..shape_lead.len], shape_lead);
        const k = shape_lead.len;
        shape[k] = in / 16;
        shape[k + 1] = out / 16;
        shape[k + 2] = rate.n;
        try self.builder.add(try std.fmt.bufPrint(&buf, "{s}.trellis", .{base}), "U16", shape[0 .. k + 3], std.mem.sliceAsBytes(d.trellis));
        shape[k] = in;
        try self.builder.add(try std.fmt.bufPrint(&buf, "{s}.suh", .{base}), "F16", shape[0 .. k + 1], std.mem.sliceAsBytes(d.suh));
        shape[k] = out;
        try self.builder.add(try std.fmt.bufPrint(&buf, "{s}.svh", .{base}), "F16", shape[0 .. k + 1], std.mem.sliceAsBytes(d.svh));
    }

    /// `base` as a trellis bank when both sides are on the 128 grid; off the grid it stays as stored.
    fn linear(self: *TrunkWriter, base: []const u8, sums: ?*const Sums, gram_key: []const u8) !bool {
        var buf: [192]u8 = undefined;
        const loc = self.src.tensors.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{base})) orelse return false;
        if (loc.shape.len != 2) return false;
        const out = loc.shape[0];
        const in = loc.shape[1];
        if (out % 128 != 0 or in % 128 != 0) return false;
        const w = try self.a.alloc(f32, in * out);
        defer self.a.free(w);
        try self.src.readLinearPublic(self.a, base, out, in, w);
        const h = self.gramFor(sums, gram_key);
        if (h == null) self.identity += 1;
        var d = try quantizeDense(self.a, self.s, self.pool, w, h, in, out, self.rate, self.opts.pin_pass, base, &self.stages);
        defer d.deinit(self.a);
        try self.addBank(base, d, &.{}, in, out, self.rate);
        try self.markBase(base);
        return true;
    }

    /// The grouped output projection `wo_a` (`[groups * o_lora, in]`) as one bank of `groups` matrices, each with
    /// its own input sum (`wo_a_in` is `[groups, in, in]`).
    fn groupedWoA(self: *TrunkWriter, base: []const u8, sums: ?*const Sums, gram_key: []const u8, groups: usize) !void {
        var buf: [192]u8 = undefined;
        const loc = self.src.tensors.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{base})) orelse return error.SourceTensorMissing;
        const rows = loc.shape[0];
        const in = loc.shape[1];
        const ol = rows / groups;
        const w = try self.a.alloc(f32, in * rows);
        defer self.a.free(w);
        try self.src.readLinearPublic(self.a, base, rows, in, w);
        const all = self.gramFor(sums, gram_key);
        if (all == null) self.identity += 1;
        const tiles = (in / 16) * (ol / 16) * self.rate.n;
        const tr = try self.a.alloc(u16, groups * tiles);
        defer self.a.free(tr);
        const suh = try self.a.alloc(u16, groups * in);
        defer self.a.free(suh);
        const svh = try self.a.alloc(u16, groups * ol);
        defer self.a.free(svh);
        const part = try self.a.alloc(f32, in * ol);
        defer self.a.free(part);
        for (0..groups) |g| {
            for (0..in) |r| @memcpy(part[r * ol ..][0..ol], w[r * rows + g * ol ..][0..ol]);
            var nb: [192]u8 = undefined;
            const name = try std.fmt.bufPrint(&nb, "{s}@{d}", .{ base, g });
            var d = try quantizeDense(self.a, self.s, self.pool, part, if (all) |x| x[g * in * in ..][0 .. in * in] else null, in, ol, self.rate, self.opts.pin_pass, name, &self.stages);
            defer d.deinit(self.a);
            @memcpy(tr[g * tiles ..][0..tiles], d.trellis);
            @memcpy(suh[g * in ..][0..in], d.suh);
            @memcpy(svh[g * ol ..][0..ol], d.svh);
        }
        try self.addBank(base, .{ .trellis = tr, .suh = suh, .svh = svh }, &.{groups}, in, ol, self.rate);
        try self.markBase(base);
    }

    /// The router's rows (weight, bias, bias_vl) in stored group order: row i is source row `perm[i]`.
    fn router(self: *TrunkWriter, prefix: []const u8, perm: []const u32) !void {
        for ([_][]const u8{ "weight", "bias", "bias_vl" }) |leaf| {
            var buf: [192]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, "{s}.ffn.gate.{s}", .{ prefix, leaf });
            const loc = self.src.tensors.get(name) orelse continue;
            const raw = try self.src.readAllAlloc(self.a, name);
            defer self.a.free(raw);
            const row = raw.len / perm.len;
            if (loc.shape[0] != perm.len or row * perm.len != raw.len) return error.RouterShape;
            const out = try self.a.alloc(u8, raw.len);
            defer self.a.free(out);
            for (perm, 0..) |p, i| @memcpy(out[i * row ..][0..row], raw[p * row ..][0..row]);
            try self.builder.add(name, loc.dtype, loc.shape, out);
            try self.mark(name);
        }
    }

    /// Every source tensor under `prefix.` not yet written and not `skip`ped, as stored.
    fn rest(self: *TrunkWriter, prefix: ?[]const u8, skip: *const fn ([]const u8) bool) !void {
        var names = std.ArrayList([]const u8).empty;
        defer names.deinit(self.a);
        var it = self.src.tensors.keyIterator();
        while (it.next()) |k| {
            const name = k.*;
            if (prefix) |p| {
                if (!std.mem.startsWith(u8, name, p) or name.len <= p.len or name[p.len] != '.') continue;
            }
            if (self.done.contains(name) or skip(name)) continue;
            try names.append(self.a, name);
        }
        std.mem.sortUnstable([]const u8, names.items, {}, struct {
            fn lt(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.lessThan(u8, x, y);
            }
        }.lt);
        for (names.items) |name| {
            const loc = self.src.tensors.get(name).?;
            const raw = try self.src.readAllAlloc(self.a, name);
            defer self.a.free(raw);
            try self.builder.add(name, loc.dtype, loc.shape, raw);
        }
    }

    fn flush(self: *TrunkWriter, io: std.Io, dir: std.Io.Dir, file: []const u8) !void {
        var k_buf: [16]u8 = undefined;
        try self.builder.flush(io, dir, file, &.{
            .{ "format", "stq" },
            .{ "codebook", "spec1" },
            .{ "codebook_table", "spec1-w15" },
            .{ "k", self.rate.kText(&k_buf) },
            .{ "encoder", self.opts.version },
        });
    }
};

fn isRoutedOrEngramTable(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".ffn.experts.") != null or std.mem.indexOf(u8, name, ".engram.embed.") != null;
}

fn isLayerOrStage(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "layers.") or std.mem.startsWith(u8, name, "mtp.");
}

/// A block's trunk: every on-grid linear as a SPEC1 bank at the writer's rate, the grouped `wo_a`, the router in group order, the
/// rest as stored.
fn writeBlockTrunk(tw: *TrunkWriter, io: std.Io, out: std.Io.Dir, prefix: []const u8, sums: ?*const Sums, sum_prefix: []const u8, groups: usize, perm: []const u32, file: []const u8) !void {
    for (block_linears) |l| {
        var bb: [192]u8 = undefined;
        var gb: [192]u8 = undefined;
        const base = try std.fmt.bufPrint(&bb, "{s}.{s}", .{ prefix, l.tail });
        const key = try std.fmt.bufPrint(&gb, "{s}.{s}", .{ sum_prefix, l.gram });
        _ = try tw.linear(base, sums, key);
    }
    var bb: [192]u8 = undefined;
    var gb: [192]u8 = undefined;
    try tw.groupedWoA(try std.fmt.bufPrint(&bb, "{s}.attn.wo_a", .{prefix}), sums, try std.fmt.bufPrint(&gb, "{s}.wo_a_in", .{sum_prefix}), groups);
    try tw.router(prefix, perm);
    try tw.rest(prefix, isRoutedOrEngramTable);
    try tw.flush(io, out, file);
}

// ── the routed experts of one block ─────────────────────────────────────────────────────────────────

const ExpertUnit = struct {
    /// `layers.L` rows (`mlp-in-LL.bf16`, `route-LL.u16`) or a DSpark stage's own.
    rows_path: []const u8,
    routes_path: []const u8,
    layer: u32,
    stage: bool,
    experts: usize,
    top_k: usize,
    classes: []const convert.Class,
    nest: []const u8,
    file: []const u8,
};

fn convertExperts(a: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, out: std.Io.Dir, src: *const source.SafetensorsSource, cfg: *const dsv_cfg.Config, opts: *const Options, unit: ExpertUnit) !convert.LayerStats {
    const hidden: usize = cfg.dim;
    const ids = try readRoutes(a, io, unit.routes_path, unit.experts);
    defer a.free(ids);
    const tokens = ids.len / unit.top_k;
    if (try fileSize(a, unit.rows_path) != tokens * hidden * 2) return error.CaptureRowsMismatch;
    const x = try fileArray(a, unit.rows_path, tokens, hidden, .bfloat16, 2);
    defer _ = mlx.mlx_array_free(x);
    const geom = convert.Geometry{ .experts = unit.experts, .hidden = hidden, .inter = cfg.moe_inter };
    var rh = try hessians.RowsHessians.init(a, s, src, unit.layer, unit.stage, geom, x, ids, unit.top_k);
    defer rh.deinit();
    rh.arch = .deepseek_v41;
    rh.limit = cfg.swiglu_limit;
    var widest = format.Rate{ .n = 0 };
    for (unit.classes) |c| widest.n = @max(widest.n, @max(c.gate_up.n, c.down.n));
    var tok_buf: [24]u8 = undefined;
    var rows_buf: [24]u8 = undefined;
    var pin_buf: [24]u8 = undefined;
    const provenance = [_][2][]const u8{
        .{ "encoder", opts.version },
        .{ "quantizer", "ldlq" },
        .{ "global_scale_search", "golden" },
        .{ "calib_tokens", try std.fmt.bufPrint(&tok_buf, "{d}", .{tokens}) },
        .{ "hessian_min_rows", try std.fmt.bufPrint(&rows_buf, "{d}", .{opts.min_rows}) },
        .{ "pin_pass", try std.fmt.bufPrint(&pin_buf, "{d}", .{opts.pin_pass}) },
    };
    return convert.convertQwenLayer(a, s, io, out, src, unit.layer, unit.stage, geom, .{
        .rate = widest,
        .dec = expert_dec,
        .min_rows = opts.min_rows,
        .batch = opts.batch,
        .pin_pass = opts.pin_pass,
        .provenance = &provenance,
        .classes = unit.classes,
        .arch = .deepseek_v41,
        .naming = .{ .nest = unit.nest, .file = unit.file, .pair_major = true },
    }, rh.hessianSource());
}

// ── the DSpark stages' calibration ──────────────────────────────────────────────────────────────────

/// What the stages' calibration forward hands the converter: each stage's FFN rows and routes on disk, every linear
/// input's X^T X on the GPU, and the main projection's.
const StageTap = struct {
    a: std.mem.Allocator,
    s: mlx.mlx_stream,
    layers: usize,
    stages: usize,
    acc: [max_stages][kinds]A = @splat(@splat(.{ .ctx = null })),
    rows: [max_stages][kinds]u64 = @splat(@splat(0)),
    main_acc: A = .{ .ctx = null },
    main_rows: u64 = 0,
    ffn_fd: [max_stages]std.c.fd_t = @splat(-1),
    route_fd: [max_stages]std.c.fd_t = @splat(-1),

    const max_stages = 8;
    const kinds = std.enums.values(dsv.TapKind).len;

    fn tap(self: *StageTap) dsv.Tap {
        return .{ .ctx = self, .input = input, .routes = routes, .expert_act = expertAct };
    }

    fn deinit(self: *StageTap) void {
        for (&self.acc) |*row| for (row) |*x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        };
        if (self.main_acc.ctx != null) _ = mlx.mlx_array_free(self.main_acc);
        for (self.ffn_fd, self.route_fd) |f, r| for ([2]std.c.fd_t{ f, r }) |fd| if (fd >= 0) {
            _ = std.c.close(fd);
        };
    }

    fn gramInto(self: *StageTap, acc: *A, rows: *u64, x: A, grouped: bool) !void {
        var sc = host.dsv41_scope.Scope.plain(self.a, self.s);
        defer sc.deinit();
        const shape = mlx.getShape(x);
        var g: A = undefined;
        if (grouped) {
            const xf = try sc.astype(try sc.transpose(x, &.{ 1, 0, 2 }), .float32);
            g = try sc.matmul(try sc.transpose(xf, &.{ 0, 2, 1 }), xf);
            rows.* += @intCast(shape[0]);
        } else {
            const width: usize = @intCast(shape[shape.len - 1]);
            const n = mlx.mlx_array_size(x) / width;
            const xf = try sc.astype(try sc.reshape(x, &.{ @intCast(n), @intCast(width) }), .float32);
            g = try sc.matmul(try sc.t2(xf), xf);
            rows.* += n;
        }
        const total = if (acc.ctx != null) sc.out(try sc.add(acc.*, g)) else sc.out(g);
        try mlx.check(mlx.mlx_array_eval(total));
        if (acc.ctx != null) _ = mlx.mlx_array_free(acc.*);
        acc.* = total;
    }

    fn input(ctx: *anyopaque, layer: usize, kind: dsv.TapKind, x: A) anyerror!void {
        const self: *StageTap = @ptrCast(@alignCast(ctx));
        if (kind == .dspark_in) {
            if (layer != self.layers) return error.StageTapOutOfOrder;
            return self.gramInto(&self.main_acc, &self.main_rows, x, false);
        }
        if (layer < self.layers or layer >= self.layers + self.stages) return error.StageTapOutOfOrder;
        const si = layer - self.layers;
        if (kind == .ffn_in) {
            const bytes = try hostBytes(self.a, self.s, x);
            defer self.a.free(bytes);
            try writeFd(self.ffn_fd[si], bytes);
        }
        try self.gramInto(&self.acc[si][@intFromEnum(kind)], &self.rows[si][@intFromEnum(kind)], x, kind == .wo_a_in);
    }

    fn routes(ctx: *anyopaque, layer: usize, idx: A, wts: A) anyerror!void {
        _ = wts;
        const self: *StageTap = @ptrCast(@alignCast(ctx));
        if (layer < self.layers or layer >= self.layers + self.stages) return error.StageTapOutOfOrder;
        const raw = try hostBytes(self.a, self.s, idx);
        defer self.a.free(raw);
        const n = raw.len / 4;
        const ids = try self.a.alloc(u16, n);
        defer self.a.free(ids);
        for (ids, 0..) |*o, i| o.* = @intCast(std.mem.readInt(i32, raw[i * 4 ..][0..4], .little));
        try writeFd(self.route_fd[layer - self.layers], std.mem.sliceAsBytes(ids));
    }

    fn expertAct(_: *anyopaque, _: usize, _: A, _: A) anyerror!void {}

    /// The sums in the capture's own layout: `layers.{cfg.layers + si}.<kind>` and `.rows`.
    fn write(self: *StageTap, io: std.Io, dir: std.Io.Dir, file: []const u8) !void {
        var b = pack.ShardBuilder.init(self.a);
        defer b.deinit();
        for (0..self.stages) |si| for (0..kinds) |k| {
            const acc = self.acc[si][k];
            if (acc.ctx == null) continue;
            var nb: [96]u8 = undefined;
            const name = try std.fmt.bufPrint(&nb, "layers.{d}.{s}", .{ self.layers + si, @tagName(@as(dsv.TapKind, @enumFromInt(k))) });
            try addSum(&b, self.a, self.s, name, acc, self.rows[si][k]);
        };
        // Stage 0 holds the main projection.
        if (self.main_acc.ctx != null) {
            var nb: [96]u8 = undefined;
            try addSum(&b, self.a, self.s, try std.fmt.bufPrint(&nb, "layers.{d}.main_in", .{self.layers}), self.main_acc, self.main_rows);
        }
        try b.flush(io, dir, file, &.{.{ "producer", "dsv41 stage calibration" }});
    }
};

fn addSum(b: *pack.ShardBuilder, a: std.mem.Allocator, s: mlx.mlx_stream, name: []const u8, acc: A, rows: u64) !void {
    const bytes = try hostBytes(a, s, acc);
    defer a.free(bytes);
    const shape = mlx.getShape(acc);
    var dims: [3]usize = undefined;
    for (shape, 0..) |d, i| dims[i] = @intCast(d);
    try b.add(name, "F32", dims[0..shape.len], bytes);
    var nb: [128]u8 = undefined;
    const r: i64 = @intCast(rows);
    try b.add(try std.fmt.bufPrint(&nb, "{s}.rows", .{name}), "I64", &.{1}, std.mem.asBytes(&r));
}

fn hostBytes(a: std.mem.Allocator, s: mlx.mlx_stream, x: A) ![]u8 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
    const data: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint8(c) orelse return error.MlxArrayDataNull);
    return a.dupe(u8, data[0..n]);
}

fn writeFd(fd: std.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const got = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (got < 0) {
            if (std.c._errno().* == @intFromEnum(std.c.E.INTR)) continue;
            return error.StageWriteFailed;
        }
        done += @intCast(got);
    }
}

fn openTrunc(a: std.mem.Allocator, dir: []const u8, name: []const u8) !std.c.fd_t {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0);
    defer a.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.StageWriteFailed;
    return fd;
}

const Window = struct { offset: usize, len: usize };

/// The capture's windows in order (`windows.jsonl` beside `calib/`).
fn readWindows(a: std.mem.Allocator, io: std.Io, capture_dir: []const u8) ![]Window {
    const path = try std.fmt.allocPrint(a, "{s}/../windows.jsonl", .{capture_dir});
    defer a.free(path);
    const raw = try readWhole(a, io, path);
    defer a.free(raw);
    var out = std.ArrayList(Window).empty;
    errdefer out.deinit(a);
    var at: usize = 0;
    var lines = std.mem.splitScalar(u8, raw, '\n');
    while (lines.next()) |line| {
        if (std.mem.trim(u8, line, " \r\t").len == 0) continue;
        const parsed = try std.json.parseFromSlice(std.json.Value, a, line, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.CaptureWindowsInvalid;
        const n = parsed.value.object.get("tokens") orelse return error.CaptureWindowsInvalid;
        if (n != .integer or n.integer <= 0) return error.CaptureWindowsInvalid;
        try out.append(a, .{ .offset = at, .len = @intCast(n.integer) });
        at += @intCast(n.integer);
    }
    return out.toOwnedSlice(a);
}

/// Runs the stages over every window's stored main rows; writes `<dspark_dir>/ffn-in-S.bf16`, `route-S.u16` and
/// `sums.safetensors`, and `done` last.
fn calibrateStages(a: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, opts: *const Options, cfg: *const dsv_cfg.Config, dspark_dir: []const u8, sink: ?Sink) !void {
    const c = cfg;
    var w = try host.dsv41_source.loadResident(io, a, opts.model_dir, .{ .experts = false, .dspark = true });
    defer w.deinit();
    const m = try dsv.init(a, c.*, &w, opts.model_dir, s, .{ .act = .bfloat16, .qat = true, .experts = false, .dspark = true });
    defer m.deinit();
    var stap = StageTap{ .a = a, .s = s, .layers = c.layers, .stages = c.n_mtp };
    defer stap.deinit();
    if (c.n_mtp > StageTap.max_stages) return error.TooManyStages;
    for (0..c.n_mtp) |si| {
        var nb: [64]u8 = undefined;
        stap.ffn_fd[si] = try openTrunc(a, dspark_dir, try std.fmt.bufPrint(&nb, "ffn-in-{d}.bf16", .{si}));
        stap.route_fd[si] = try openTrunc(a, dspark_dir, try std.fmt.bufPrint(&nb, "route-{d}.u16", .{si}));
    }
    m.tap = stap.tap();
    defer m.tap = null;

    const windows = try readWindows(a, io, opts.capture_dir);
    defer a.free(windows);
    const tokens_path = try std.fmt.allocPrint(a, "{s}/tokens.bin", .{opts.capture_dir});
    defer a.free(tokens_path);
    const tok_raw = try readWhole(a, io, tokens_path);
    defer a.free(tok_raw);
    const all_ids: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, tok_raw));
    const targets = c.dspark.targets[0..c.dspark.n_targets];
    var maps: [dsv_cfg.max_targets]A = undefined;
    var mapped: usize = 0;
    defer for (maps[0..mapped]) |x| {
        _ = mlx.mlx_array_free(x);
    };
    for (targets) |t| {
        const p = try std.fmt.allocPrint(a, "{s}/dspark-in-{d:0>2}.bf16", .{ opts.capture_dir, t });
        defer a.free(p);
        if (try fileSize(a, p) != all_ids.len * c.dim * 2) return error.CaptureRowsMismatch;
        maps[mapped] = try fileArray(a, p, all_ids.len, c.dim, .bfloat16, 2);
        mapped += 1;
    }
    const anchors = try a.alloc(usize, opts.anchor_batch);
    defer a.free(anchors);
    var done_windows: usize = 0;
    for (windows) |win| {
        if (win.offset + win.len > all_ids.len) return error.CaptureWindowsInvalid;
        var sc = host.dsv41_scope.Scope.plain(a, s);
        defer sc.deinit();
        var parts: [dsv_cfg.max_targets]A = undefined;
        for (maps[0..mapped], 0..) |mp, i| parts[i] = try sc.slice(mp, 0, win.offset, win.offset + win.len);
        const main_rows = try sc.contiguous(try sc.concat(parts[0..mapped], -1));
        try mlx.check(mlx.mlx_array_eval(main_rows));
        const ids = all_ids[win.offset..][0..win.len];
        var p: usize = 1;
        while (p < win.len) {
            var n: usize = 0;
            while (n < anchors.len and p < win.len) : (p += opts.anchor_stride) {
                anchors[n] = p;
                n += 1;
            }
            try dsv.dsparkCalibrate(m, ids, main_rows, anchors[0..n]);
        }
        done_windows += 1;
        if (done_windows % 32 == 0) say(sink, "dspark calibration: {d}/{d} windows", .{ done_windows, windows.len });
    }
    var dir = try std.Io.Dir.cwd().openDir(io, dspark_dir, .{});
    defer dir.close(io);
    try stap.write(io, dir, "sums.safetensors");
    for (stap.ffn_fd, stap.route_fd) |f, r| for ([2]std.c.fd_t{ f, r }) |fd| if (fd >= 0) if (std.c.fsync(fd) != 0) return error.StageWriteFailed;
    try dir.writeFile(io, .{ .sub_path = "done", .data = "" });
}

// ── the pack ────────────────────────────────────────────────────────────────────────────────────────

fn policyKey(a: std.mem.Allocator, opts: *const Options, tokens: u64) ![]u8 {
    var k_buf: [16]u8 = undefined;
    const base = try std.fmt.allocPrint(a, "{s}|dsv41|experts {s} {s}|plan {s}|pin {d}|rows {d}|trunk spec1-w15 n{d} head n{d} stage n{d} embed {d}/{d}|calib {d} stride {d}", .{
        opts.version,
        opts.rate.kText(&k_buf),
        codebook_tables.tables[expert_dec.table.?].name,
        if (opts.plan) |p| p.sha256[0..] else "none",
        opts.pin_pass,
        opts.min_rows,
        opts.trunk_rate.n,
        head_rate.n,
        opts.drafter_experts_rate.n,
        embed_bits,
        embed_group,
        tokens,
        opts.anchor_stride,
    });
    defer a.free(base);
    // A drafter trunk at the trunk's rate leaves the key as it was before the flag existed; the order the shards are in is part of it.
    if (opts.drafterTrunkRate().n == opts.trunk_rate.n) return std.fmt.allocPrint(a, "{s}|pair_major", .{base});
    return std.fmt.allocPrint(a, "{s}|drafter trunk n{d}|pair_major", .{ base, opts.drafterTrunkRate().n });
}

/// The source's `config.json` plus `expert_quant` (the routed experts) and `trunk_quant` (the trunk banks), each stamped
/// with its widest rate.
fn buildConfig(a: std.mem.Allocator, source_config: []const u8, widest: format.Rate, trunk_widest: format.Rate, provenance: []const [2][]const u8) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, ar, source_config, .{}) catch return error.UnknownSourceFormat;
    if (parsed != .object) return error.UnknownSourceFormat;
    var root = parsed.object;
    var k_buf: [2][16]u8 = undefined;
    for ([_]struct { key: []const u8, dec: format.Decode, k: []const u8, pair_major: bool }{
        .{ .key = "expert_quant", .dec = expert_dec, .k = widest.kText(&k_buf[0]), .pair_major = true },
        .{ .key = "trunk_quant", .dec = trunk_dec, .k = trunk_widest.kText(&k_buf[1]), .pair_major = false },
    }) |q| {
        const spec = pack.expertQuant(q.dec, q.k, provenance);
        var o = std.json.ObjectMap.empty;
        if (q.pair_major) try o.put(ar, "tile_order", .{ .string = "pair_major" });
        try o.put(ar, "format", .{ .string = spec.format });
        try o.put(ar, "k", .{ .number_string = spec.k_text });
        try o.put(ar, "codebook", .{ .string = spec.codebook });
        try o.put(ar, "window", .{ .integer = spec.window });
        try o.put(ar, "codebook_table", .{ .string = spec.table.? });
        try o.put(ar, "codebook_digest", .{ .string = spec.digest.? });
        for (provenance) |pair| try o.put(ar, pair[0], .{ .string = pair[1] });
        try root.put(ar, q.key, .{ .object = o });
    }
    return std.json.Stringify.valueAlloc(a, std.json.Value{ .object = root }, .{ .whitespace = .indent_2 });
}

extern "c" fn clonefile(src: [*:0]const u8, dst: [*:0]const u8, flags: u32) c_int;

/// The Engram tables, untouched: each source shard holding one is cloned (copy-on-write) under a name the
/// resident loader skips; the index points the table at it.
fn cloneEngram(a: std.mem.Allocator, io: std.Io, src: *const source.SafetensorsSource, opts: *const Options, out: std.Io.Dir, entries: *std.ArrayList(pack.IndexEntry)) !void {
    var it = src.tensors.iterator();
    var cloned = std.StringHashMap([]const u8).init(a);
    defer {
        var vit = cloned.valueIterator();
        while (vit.next()) |v| a.free(v.*);
        cloned.deinit();
    }
    while (it.next()) |e| {
        if (std.mem.indexOf(u8, e.key_ptr.*, ".engram.embed.") == null) continue;
        const shard_name = src.shards[e.value_ptr.shard_idx];
        const gop = try cloned.getOrPut(shard_name);
        if (!gop.found_existing) {
            const dst_name = try std.fmt.allocPrint(a, "engram-{s}", .{shard_name});
            gop.value_ptr.* = dst_name;
            if (!exists(io, out, dst_name)) {
                const from = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ opts.model_dir, shard_name }, 0);
                defer a.free(from);
                const to = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ opts.out_dir, dst_name }, 0);
                defer a.free(to);
                if (clonefile(from.ptr, to.ptr, 0) != 0) {
                    var sd = try std.Io.Dir.cwd().openDir(io, opts.model_dir, .{});
                    defer sd.close(io);
                    try sd.copyFile(shard_name, out, dst_name, io, .{});
                }
            }
        }
        const tensor = try a.dupe(u8, e.key_ptr.*);
        errdefer a.free(tensor);
        const file = try a.dupe(u8, gop.value_ptr.*);
        errdefer a.free(file);
        try entries.append(a, .{ .tensor = tensor, .file = file });
    }
}

/// The index over every `model-*.safetensors` the pack wrote, and the Engram clones.
/// The index of a pack whose routed trellis tensors are `trellis_pm`: tensors in the sorted order of their pack-order names, then
/// every shard's pads in file-name order. This is the text `sushi repack` writes (docs/pack-format.md#repack), kept byte for
/// byte; `total_size` counts the tensors, not the pads.
pub fn pairMajorIndex(a: std.mem.Allocator, entries: []const pack.IndexEntry, pads: []const pack.IndexEntry, total: u64) ![]u8 {
    const order = try a.alloc(usize, entries.len);
    defer a.free(order);
    for (order, 0..) |*o, i| o.* = i;
    const Ctx = struct {
        e: []const pack.IndexEntry,
        fn packName(t: []const u8) []const u8 {
            return if (std.mem.endsWith(u8, t, ".trellis_pm")) t[0 .. t.len - "_pm".len] else t;
        }
        fn lt(c: @This(), x: usize, y: usize) bool {
            return std.mem.lessThan(u8, packName(c.e[x].tensor), packName(c.e[y].tensor));
        }
    };
    std.mem.sortUnstable(usize, order, Ctx{ .e = entries }, Ctx.lt);
    const pad_order = try a.alloc(usize, pads.len);
    defer a.free(pad_order);
    for (pad_order, 0..) |*o, i| o.* = i;
    // A stable sort by file keeps each shard's pads in data order.
    std.mem.sort(usize, pad_order, pads, struct {
        fn lt(p: []const pack.IndexEntry, x: usize, y: usize) bool {
            return std.mem.lessThan(u8, p[x].file, p[y].file);
        }
    }.lt);
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(a);
    try out.print(a, "{{\n \"metadata\": {{\"total_size\":{d}}},\n \"weight_map\": {{\n", .{total});
    var n: usize = 0;
    for (order) |i| {
        try out.print(a, "{s}  {f}: {f}", .{ if (n == 0) "" else ",\n", std.json.fmt(entries[i].tensor, .{}), std.json.fmt(entries[i].file, .{}) });
        n += 1;
    }
    for (pad_order) |i| {
        try out.print(a, "{s}  {f}: {f}", .{ if (n == 0) "" else ",\n", std.json.fmt(pads[i].tensor, .{}), std.json.fmt(pads[i].file, .{}) });
        n += 1;
    }
    try out.appendSlice(a, "\n }\n}\n");
    return out.toOwnedSlice(a);
}

fn writeIndex(a: std.mem.Allocator, io: std.Io, src: *const source.SafetensorsSource, opts: *const Options, out: std.Io.Dir) !void {
    var entries = std.ArrayList(pack.IndexEntry).empty;
    defer {
        for (entries.items) |e| {
            a.free(e.tensor);
            a.free(e.file);
        }
        entries.deinit(a);
    }
    var pads = std.ArrayList(pack.IndexEntry).empty;
    defer {
        for (pads.items) |e| {
            a.free(e.tensor);
            a.free(e.file);
        }
        pads.deinit(a);
    }
    var total: u64 = 0;
    var it = out.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.startsWith(u8, entry.name, "model-") or !std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ opts.out_dir, entry.name });
        defer a.free(path);
        const z = try a.dupeSentinel(u8, path, 0);
        defer a.free(z);
        total += try pack.appendShardEntries(a, z, entry.name, &entries);
        try pack.appendPadEntries(a, z, entry.name, &pads);
    }
    try cloneEngram(a, io, src, opts, out, &entries);
    const body = try pairMajorIndex(a, entries.items, pads.items, total);
    defer a.free(body);
    try out.writeFile(io, .{ .sub_path = ".model.safetensors.index.json.tmp", .data = body });
    try out.rename(".model.safetensors.index.json.tmp", out, "model.safetensors.index.json", io);
}

fn uniformClasses(a: std.mem.Allocator, experts: usize, rate: format.Rate) ![]convert.Class {
    const ids = try a.alloc(u32, experts);
    for (ids, 0..) |*v, e| v.* = @intCast(e);
    const out = try a.alloc(convert.Class, 1);
    out[0] = .{ .gate_up = rate, .down = rate, .ids = ids };
    return out;
}

fn freeClasses(a: std.mem.Allocator, classes: []convert.Class) void {
    for (classes) |c| a.free(c.ids);
    a.free(classes);
}

fn addStats(sum: *Summary, stats: convert.LayerStats) void {
    sum.gate_up_identity += stats.gate_up_identity;
    sum.down_identity += stats.down_identity;
}

fn sayStages(sink: ?Sink, label: []const u8, unit: usize, times: *const stages_mod.Stages, elapsed: f64) void {
    var buf: [384]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    w.print("{s} {d}: {d:.1} s", .{ label, unit, elapsed }) catch return;
    for (std.enums.values(stages_mod.Stage)) |stg| {
        const ns = times.get(stg);
        if (ns > 0) w.print(", {s} {d:.1} s", .{ @tagName(stg), @as(f64, @floatFromInt(ns)) / 1e9 }) catch return;
    }
    say(sink, "{s}", .{w.buffered()});
}

fn seconds(since: std.Io.Timestamp, io: std.Io) f64 {
    return @as(f64, @floatFromInt(since.untilNow(io, .awake).nanoseconds)) / 1e9;
}

pub fn run(a: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, opts: Options, sink: ?Sink) !Summary {
    const cwd = std.Io.Dir.cwd();
    var src_dir = try cwd.openDir(io, opts.model_dir, .{ .iterate = true });
    defer src_dir.close(io);
    if (try source.identifySourceDir(a, io, src_dir) != .deepseek_v41) return error.ArchitectureNotConvertible;
    var cfg = try dsv_cfg.readFile(io, a, opts.model_dir);
    _ = &cfg;
    if (cfg.swiglu_limit != @round(cfg.swiglu_limit)) return error.SwigluLimitNotWhole;
    if (!encode.runnableRate(opts.rate) or !encode.searchable(expert_dec, opts.rate)) return error.RateUnsupported;
    if (!encode.validPinPass(opts.pin_pass)) return error.PinPassUnsupported;
    for ([_]format.Rate{ opts.trunk_rate, opts.drafterTrunkRate() }) |r| if (!encode.searchable(trunk_dec, r)) return error.RateUnsupported;
    if (!encode.searchable(expert_dec, opts.drafter_experts_rate)) return error.RateUnsupported;
    if (opts.plan) |plan| {
        if (plan.layers.len != cfg.layers) return error.RateUnsupported;
        for (plan.layers) |cl| if (cl) |classes| for (classes) |c| for ([2]format.Rate{ c.gate_up, c.down }) |r| {
            if (!encode.searchable(expert_dec, r)) return error.RateUnsupported;
        };
    }
    var src = try source.SafetensorsSource.open(a, io, src_dir);
    defer src.deinit();

    var route_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tokens = (try fileSize(a, try std.fmt.bufPrint(&route_buf, "{s}/route-00.u16", .{opts.capture_dir}))) / (2 * cfg.top_k);

    try io_util.makeDirPath(io, opts.out_dir);
    var out = try cwd.openDir(io, opts.out_dir, .{ .iterate = true });
    defer out.close(io);
    const key = try policyKey(a, &opts, tokens);
    defer a.free(key);
    if (out.readFileAlloc(io, progress_name, a, .limited(1 << 20))) |stored| {
        defer a.free(stored);
        if (!opts.resuming) return error.OutputNotEmpty;
        if (!std.mem.eql(u8, stored, key)) return error.ResumePolicyMismatch;
    } else |_| {
        var it = out.iterate();
        if (try it.next(io) != null and !opts.resuming) return error.OutputNotEmpty;
        try out.writeFile(io, .{ .sub_path = progress_name, .data = key });
    }
    if (exists(io, out, "config.json")) return error.ResumePackComplete;

    const pool = try pool_mod.Pool.init(a, @max(1, @min(pool_mod.cpuCount(), 16) - 1));
    defer pool.deinit();
    var summary = Summary{};

    const layers: []const u32 = opts.only_layers orelse blk: {
        const all = try a.alloc(u32, cfg.layers);
        for (all, 0..) |*v, i| v.* = @intCast(i);
        break :blk all;
    };
    defer if (opts.only_layers == null) a.free(layers);

    for (layers) |layer| {
        var eb: [64]u8 = undefined;
        var tb: [64]u8 = undefined;
        const ef = try layerExpertsFile(&eb, layer);
        const tf = try layerTrunkFile(&tb, layer);
        if (exists(io, out, ef) and exists(io, out, tf)) {
            summary.layers_reused += 1;
            continue;
        }
        const started = std.Io.Timestamp.now(io, .awake);
        var owned: ?[]convert.Class = null;
        defer if (owned) |c| freeClasses(a, c);
        const classes: []const convert.Class = if (opts.plan) |p| (p.classes(layer) orelse blk: {
            owned = try uniformClasses(a, cfg.experts, opts.rate);
            break :blk owned.?;
        }) else blk: {
            owned = try uniformClasses(a, cfg.experts, opts.rate);
            break :blk owned.?;
        };
        const perm = try rate_plan.permutation(a, classes);
        defer a.free(perm);
        if (!exists(io, out, ef)) {
            var nest_buf: [64]u8 = undefined;
            const rows_path = try std.fmt.allocPrint(a, "{s}/mlp-in-{d:0>2}.bf16", .{ opts.capture_dir, layer });
            defer a.free(rows_path);
            const routes_path = try std.fmt.allocPrint(a, "{s}/route-{d:0>2}.u16", .{ opts.capture_dir, layer });
            defer a.free(routes_path);
            const stats = try convertExperts(a, io, s, out, &src, &cfg, &opts, .{
                .rows_path = rows_path,
                .routes_path = routes_path,
                .layer = layer,
                .stage = false,
                .experts = cfg.experts,
                .top_k = cfg.top_k,
                .classes = classes,
                .nest = try std.fmt.bufPrint(&nest_buf, "model.layers.{d}", .{layer}),
                .file = ef,
            });
            addStats(&summary, stats);
            sayStages(sink, "layer experts", layer, &stats.stages, seconds(started, io));
            say(sink, "layer {d}: {d} experts, identity Hessians gate/up {d}, down {d}", .{ layer, stats.experts, stats.gate_up_identity, stats.down_identity });
        }
        if (!exists(io, out, tf)) {
            const t0 = std.Io.Timestamp.now(io, .awake);
            const sums_path = try std.fmt.allocPrint(a, "{s}/trunk-{d:0>2}.safetensors", .{ opts.capture_dir, layer });
            defer a.free(sums_path);
            var sums = try Sums.open(a, io, sums_path);
            defer sums.deinit(a);
            var tw = TrunkWriter.init(a, s, &src, pool, &opts, opts.trunk_rate);
            defer tw.deinit();
            var pb: [32]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&pb, "layers.{d}", .{layer});
            try writeBlockTrunk(&tw, io, out, prefix, &sums, prefix, cfg.o_groups, perm, tf);
            summary.trunk_identity += tw.identity;
            sayStages(sink, "layer trunk", layer, &tw.stages, seconds(t0, io));
        }
        summary.layers_written += 1;
    }
    if (opts.only_layers != null) return summary;

    if (cfg.n_mtp > 0 and cfg.dspark.block > 0) try convertStages(a, io, s, &opts, &cfg, &src, out, pool, &summary, sink);

    if (!exists(io, out, globals_file)) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        var tw = TrunkWriter.init(a, s, &src, pool, &opts, head_rate);
        defer tw.deinit();
        const head_path = try std.fmt.allocPrint(a, "{s}/trunk-head.safetensors", .{opts.capture_dir});
        defer a.free(head_path);
        var sums = try Sums.open(a, io, head_path);
        defer sums.deinit(a);
        if (!try tw.linear("head", &sums, "head_in")) return error.HeadOffGrid;
        try writeEmbed(&tw);
        try tw.rest(null, isLayerOrStage);
        try tw.flush(io, out, globals_file);
        sayStages(sink, "globals", 0, &tw.stages, seconds(t0, io));
    }

    try writeIndex(a, io, &src, &opts, out);
    _ = try pack.copySidecars(a, io, src_dir, out);
    const source_config = try src_dir.readFileAlloc(io, "config.json", a, .limited(16 << 20));
    defer a.free(source_config);
    var widest = opts.rate;
    if (opts.plan) |p| widest.n = @max(widest.n, p.widest.n);
    widest.n = @max(widest.n, opts.drafter_experts_rate.n);
    var tok_buf: [24]u8 = undefined;
    const provenance = [_][2][]const u8{
        .{ "encoder", opts.version },
        .{ "calib_tokens", try std.fmt.bufPrint(&tok_buf, "{d}", .{tokens}) },
    };
    const trunk_widest = format.Rate{ .n = @max(head_rate.n, @max(opts.trunk_rate.n, opts.drafterTrunkRate().n)) };
    const config = try buildConfig(a, source_config, widest, trunk_widest, &provenance);
    defer a.free(config);
    out.deleteFile(io, progress_name) catch {};
    try out.writeFile(io, .{ .sub_path = ".config.json.tmp", .data = config });
    try out.rename(".config.json.tmp", out, "config.json", io);
    return summary;
}

/// The embedding as MLX's affine triple (row gather).
fn writeEmbed(tw: *TrunkWriter) !void {
    const loc = tw.src.tensors.get("embed.weight") orelse return error.SourceTensorMissing;
    if (!std.mem.eql(u8, loc.dtype, "BF16") or loc.shape.len != 2) return error.EmbedDtype;
    const raw = try tw.src.readAllAlloc(tw.a, "embed.weight");
    defer tw.a.free(raw);
    const rows = loc.shape[0];
    const cols = loc.shape[1];
    var q = try source.quantizeAffineBf16(tw.a, @alignCast(std.mem.bytesAsSlice(u16, raw)), rows, cols, embed_bits, embed_group);
    defer q.deinit(tw.a);
    try tw.builder.add("embed.weight", "U32", &.{ rows, q.w_cols }, std.mem.sliceAsBytes(q.weight));
    try tw.builder.add("embed.scales", "BF16", &.{ rows, q.s_cols }, std.mem.sliceAsBytes(q.scales));
    try tw.builder.add("embed.biases", "BF16", &.{ rows, q.s_cols }, std.mem.sliceAsBytes(q.biases));
    try tw.mark("embed.weight");
}

fn convertStages(a: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, opts: *const Options, cfg: *const dsv_cfg.Config, src: *const source.SafetensorsSource, out: std.Io.Dir, pool: *pool_mod.Pool, summary: *Summary, sink: ?Sink) !void {
    const dspark_dir = try std.fmt.allocPrint(a, "{s}/.dspark", .{opts.out_dir});
    defer a.free(dspark_dir);
    try io_util.makeDirPath(io, dspark_dir);
    var all_done = true;
    for (0..cfg.n_mtp) |si| {
        var eb: [64]u8 = undefined;
        var tb: [64]u8 = undefined;
        if (!exists(io, out, try stageExpertsFile(&eb, @intCast(si))) or !exists(io, out, try stageTrunkFile(&tb, @intCast(si)))) all_done = false;
    }
    if (all_done) return;
    {
        var dd = try std.Io.Dir.cwd().openDir(io, dspark_dir, .{});
        defer dd.close(io);
        if (!exists(io, dd, "done")) {
            const t0 = std.Io.Timestamp.now(io, .awake);
            try calibrateStages(a, io, s, opts, cfg, dspark_dir, sink);
            say(sink, "dspark calibration: {d:.1} s", .{seconds(t0, io)});
        }
    }
    const sums_path = try std.fmt.allocPrint(a, "{s}/sums.safetensors", .{dspark_dir});
    defer a.free(sums_path);
    var sums = try Sums.open(a, io, sums_path);
    defer sums.deinit(a);
    const classes = try uniformClasses(a, cfg.dspark.experts, opts.drafter_experts_rate);
    defer freeClasses(a, classes);
    for (0..cfg.n_mtp) |si| {
        const stage: u32 = @intCast(si);
        var eb: [64]u8 = undefined;
        var tb: [64]u8 = undefined;
        const ef = try stageExpertsFile(&eb, stage);
        const tf = try stageTrunkFile(&tb, stage);
        if (!exists(io, out, ef)) {
            const t0 = std.Io.Timestamp.now(io, .awake);
            var nest_buf: [32]u8 = undefined;
            const rows_path = try std.fmt.allocPrint(a, "{s}/ffn-in-{d}.bf16", .{ dspark_dir, si });
            defer a.free(rows_path);
            const routes_path = try std.fmt.allocPrint(a, "{s}/route-{d}.u16", .{ dspark_dir, si });
            defer a.free(routes_path);
            const stats = try convertExperts(a, io, s, out, src, cfg, opts, .{
                .rows_path = rows_path,
                .routes_path = routes_path,
                .layer = stage,
                .stage = true,
                .experts = cfg.dspark.experts,
                .top_k = cfg.dspark.top_k,
                .classes = classes,
                .nest = try std.fmt.bufPrint(&nest_buf, "mtp.{d}", .{si}),
                .file = ef,
            });
            addStats(summary, stats);
            sayStages(sink, "stage experts", si, &stats.stages, seconds(t0, io));
        }
        if (!exists(io, out, tf)) {
            const t0 = std.Io.Timestamp.now(io, .awake);
            var tw = TrunkWriter.init(a, s, src, pool, opts, opts.drafterTrunkRate());
            defer tw.deinit();
            var pb: [32]u8 = undefined;
            var sb: [32]u8 = undefined;
            const prefix = try std.fmt.bufPrint(&pb, "mtp.{d}", .{si});
            const sum_prefix = try std.fmt.bufPrint(&sb, "layers.{d}", .{cfg.layers + si});
            try writeBlockTrunk(&tw, io, out, prefix, &sums, sum_prefix, cfg.o_groups, classes[0].ids, tf);
            summary.trunk_identity += tw.identity;
            sayStages(sink, "stage trunk", si, &tw.stages, seconds(t0, io));
        }
        summary.stages_written += 1;
    }
}

// ── tests ───────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;

const tiny_tokenizer =
    \\{"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE",
    \\ "vocab":{"a":0,"b":1,"c":2,"d":3},"merges":[]}}
;

/// The checkpoint's own files and a tokenizer, in a directory of their own.
fn stageTinySource(a: std.mem.Allocator, io: std.Io, dst: std.Io.Dir, src_dir: []const u8) !void {
    var src = try std.Io.Dir.openDirAbsolute(io, src_dir, .{ .iterate = true });
    defer src.close(io);
    var it = src.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        const n = entry.name;
        const wanted = std.mem.eql(u8, n, "config.json") or std.mem.eql(u8, n, "engram_token_map.json") or std.mem.eql(u8, n, "model.safetensors.index.json") or
            (std.mem.startsWith(u8, n, "model-") and std.mem.endsWith(u8, n, ".safetensors"));
        if (!wanted) continue;
        const bytes = try src.readFileAlloc(io, n, a, .limited(64 << 20));
        defer a.free(bytes);
        try dst.writeFile(io, .{ .sub_path = n, .data = bytes });
    }
    try dst.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{}" });
    try dst.writeFile(io, .{ .sub_path = "tokenizer.json", .data = tiny_tokenizer });
}

/// The checkpoint (or pack) at `root` through the engine's one loader, its experts streamed.
fn loadTiny(a: std.mem.Allocator, io: std.Io, root: []const u8) !*host.kld.Loaded {
    const l = try a.create(host.kld.Loaded);
    errdefer a.destroy(l);
    var config = try host.model.parseConfig(io, a, root);
    errdefer config.deinit(a);
    config.expert_streaming = true;
    config.expert_cache_bytes = 64 << 20;
    config.expert_source_dir = try a.dupe(u8, root);
    l.* = .{ .allocator = a, .io = io, .config = config, .tok = try host.tokenizer.loadTokenizer(io, a, root), .chat_config = undefined, .weights = undefined, .xfm = undefined };
    errdefer l.tok.deinit();
    l.weights = try host.model.loadWeightsForConfig(io, a, root, &l.config, false);
    errdefer l.weights.deinit();
    l.xfm = try host.trunk.Trunk.init(io, a, l.config, &l.weights);
    return l;
}

fn freeTiny(a: std.mem.Allocator, l: *host.kld.Loaded) void {
    l.xfm.deinit();
    l.weights.deinit();
    l.tok.deinit();
    l.config.deinit(a);
    a.destroy(l);
}

fn lastLogits(a: std.mem.Allocator, l: *host.kld.Loaded, ids: []const u32) ![]f32 {
    const m = l.xfm.arch_state.dsv41.model.?;
    const logits = try dsv.extend(m, ids, .last);
    defer _ = mlx.mlx_array_free(logits);
    const n = mlx.mlx_array_size(logits);
    const data = mlx.mlx_array_data_float32(logits) orelse return error.MlxArrayDataNull;
    return a.dupe(f32, data[0..n]);
}

/// Every trellis bank of the shard `file` is `n` wide in its last axis and the shard stamps that rate as its `k`.
fn expectShardRate(a: std.mem.Allocator, io: std.Io, out_dir: []const u8, file: []const u8, rate: format.Rate) !void {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ out_dir, file });
    defer a.free(path);
    const raw = try a.alignedAlloc(u8, .of(u16), try fileSize(a, path));
    defer a.free(raw);
    {
        var f = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer f.close(io);
        try pack.readAllChunked(f, io, raw, 0, pack.read_chunk);
    }
    var shard = try st.parse(a, raw);
    defer shard.deinit(a);
    var k_buf: [16]u8 = undefined;
    try testing.expectEqualStrings(rate.kText(&k_buf), shard.metadata.get("k").?);
    var banks: usize = 0;
    var it = shard.tensors.iterator();
    while (it.next()) |e| {
        if (!std.mem.endsWith(u8, e.key_ptr.*, ".trellis") and !std.mem.endsWith(u8, e.key_ptr.*, ".trellis_pm")) continue;
        banks += 1;
        try testing.expectEqual(@as(usize, rate.n), e.value_ptr.shape[e.value_ptr.shape.len - 1]);
    }
    try testing.expect(banks > 0);
}

const tile_page: usize = 16384;

/// A routed shard laid out as docs/pack-format.md#tile-order says: every trellis is `trellis_pm` `[E, OT/2, IT, 2, n]` on a
/// 16 KiB page of the file, the gaps are zero `__pad__.<stem>.<i>` tensors the index lists, the data starts on a page and the
/// shard stamps `tile_order`.
fn expectPairMajorShard(a: std.mem.Allocator, io: std.Io, out_dir: []const u8, file: []const u8, index: std.json.ObjectMap) !void {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ out_dir, file });
    defer a.free(path);
    const raw = try a.alignedAlloc(u8, .of(u16), try fileSize(a, path));
    defer a.free(raw);
    {
        var f = try std.Io.Dir.cwd().openFile(io, path, .{});
        defer f.close(io);
        try pack.readAllChunked(f, io, raw, 0, pack.read_chunk);
    }
    var shard = try st.parse(a, raw);
    defer shard.deinit(a);
    try testing.expectEqualStrings("pair_major", shard.metadata.get("tile_order").?);
    const stem = file[0 .. file.len - ".safetensors".len];
    var first: usize = raw.len;
    var pm: usize = 0;
    var pads: usize = 0;
    var it = shard.tensors.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        const view = e.value_ptr.*;
        const at = @intFromPtr(view.bytes.ptr) - @intFromPtr(raw.ptr);
        first = @min(first, at);
        if (std.mem.endsWith(u8, key, ".trellis")) return error.PackOrderTrellis;
        if (std.mem.startsWith(u8, key, "__pad__.")) {
            pads += 1;
            try testing.expect(std.mem.startsWith(u8, key["__pad__.".len..], stem));
            try testing.expectEqualStrings("U8", view.dtype);
            try testing.expect(view.shape.len == 1 and view.bytes.len > 0 and view.bytes.len < tile_page);
            for (view.bytes) |b| try testing.expectEqual(@as(u8, 0), b);
            try testing.expectEqualStrings(file, index.get("weight_map").?.object.get(key).?.string);
            continue;
        }
        if (!std.mem.endsWith(u8, key, ".trellis_pm")) continue;
        pm += 1;
        try testing.expect(view.shape.len == 5 and view.shape[3] == 2);
        try testing.expectEqual(@as(usize, 0), at % tile_page);
        try testing.expectEqualStrings(file, index.get("weight_map").?.object.get(key).?.string);
    }
    try testing.expect(pm > 0);
    try testing.expectEqual(@as(usize, 0), first % tile_page);
}

/// The shards of the tiny pack carry the recipe's rates, and `config.json` stamps the widest trunk rate.
fn expectTinyRates(a: std.mem.Allocator, io: std.Io, out_dir: []const u8, recipe: TinyRecipe) !void {
    const drafter = recipe.drafter orelse recipe.trunk;
    try expectShardRate(a, io, out_dir, "model-trunk-00.safetensors", recipe.trunk);
    try expectShardRate(a, io, out_dir, "model-mtp-trunk-0.safetensors", drafter);
    try expectShardRate(a, io, out_dir, "model-mtp-0.safetensors", recipe.drafter_experts);
    try expectShardRate(a, io, out_dir, globals_file, head_rate);
    const config_path = try std.fmt.allocPrint(a, "{s}/config.json", .{out_dir});
    defer a.free(config_path);
    const config = try readWhole(a, io, config_path);
    defer a.free(config);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, config, .{});
    defer parsed.deinit();
    const widest = @max(head_rate.n, @max(recipe.trunk.n, drafter.n));
    try testing.expectEqual(@as(i64, widest / 16), parsed.value.object.get("trunk_quant").?.object.get("k").?.integer);
    try testing.expectEqualStrings("pair_major", parsed.value.object.get("expert_quant").?.object.get("tile_order").?.string);
    const index_path = try std.fmt.allocPrint(a, "{s}/model.safetensors.index.json", .{out_dir});
    defer a.free(index_path);
    const index_raw = try readWhole(a, io, index_path);
    defer a.free(index_raw);
    const index = try std.json.parseFromSlice(std.json.Value, a, index_raw, .{});
    defer index.deinit();
    for ([_][]const u8{ "model-layer-00.safetensors", "model-layer-07.safetensors", "model-mtp-0.safetensors", "model-mtp-2.safetensors" }) |file| {
        try expectPairMajorShard(a, io, out_dir, file, index.value.object);
    }
}

/// The three rates `sushi convert` takes for a DeepSeek pack: the trunk, the DSpark trunk (null: the trunk's) and the
/// DSpark experts.
const TinyRecipe = struct {
    trunk: format.Rate = .{ .n = 64 },
    drafter: ?format.Rate = null,
    drafter_experts: format.Rate = .{ .n = 32 },
    /// Also convert into a second pack in two runs, the first cut after two layers and the second `--resume`, and require its files
    /// to equal the uninterrupted pack's.
    resumed: bool = false,
};

fn expectSamePack(a: std.mem.Allocator, io: std.Io, want_dir: []const u8, got_dir: []const u8) !void {
    var dir = try std.Io.Dir.openDirAbsolute(io, want_dir, .{ .iterate = true });
    defer dir.close(io);
    var files: usize = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or std.mem.startsWith(u8, entry.name, "convert-progress")) continue;
        const want_path = try std.fmt.allocPrint(a, "{s}/{s}", .{ want_dir, entry.name });
        defer a.free(want_path);
        const got_path = try std.fmt.allocPrint(a, "{s}/{s}", .{ got_dir, entry.name });
        defer a.free(got_path);
        const want = try readWhole(a, io, want_path);
        defer a.free(want);
        const got = try readWhole(a, io, got_path);
        defer a.free(got);
        try testing.expectEqualSlices(u8, want, got);
        files += 1;
    }
    try testing.expect(files > 20);
}

fn convertTiny(recipe: TinyRecipe) !void {
    const src_env = std.c.getenv("DSV41_STQ_SRC") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try tmp.dir.createDirPath(io, "src");
    {
        var d = try tmp.dir.openDir(io, "src", .{});
        defer d.close(io);
        try stageTinySource(a, io, d, std.mem.span(src_env));
    }
    const src_root = try std.fmt.allocPrint(a, "{s}/src", .{root});
    defer a.free(src_root);
    var cfg = try dsv_cfg.readFile(io, a, src_root);
    _ = &cfg;

    var prng = std.Random.DefaultPrng.init(11);
    var body: std.Io.Writer.Allocating = .init(a);
    defer body.deinit();
    for (0..6) |w| {
        try body.writer.print("{{\"id\":\"w{d}\",\"prompt_ids\":[", .{w});
        for (0..48) |k| try body.writer.print("{s}{d}", .{ if (k == 0) "" else ",", prng.random().uintLessThan(u32, cfg.vocab - 8) });
        try body.writer.writeAll("]}\n");
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "prompts.jsonl", .data = body.written() });
    const prompts = try std.fmt.allocPrint(a, "{s}/prompts.jsonl", .{root});
    defer a.free(prompts);
    const cap_dir = try std.fmt.allocPrint(a, "{s}/cap", .{root});
    defer a.free(cap_dir);
    {
        const l = try loadTiny(a, io, src_root);
        defer freeTiny(a, l);
        var quiet: host.kld.Out = .{ .silent = true };
        try host.dsv41_kld_capture.runLoaded(a, io, l, .{ .command = .capture, .layer_major = true, .model_dir = src_root, .prompts = prompts, .out_dir = cap_dir, .tokens = 1, .no_template = true }, 3, 64, &quiet);
    }

    // Three rate groups per layer, so the router's group order differs from the source's.
    var plan_text: std.Io.Writer.Allocating = .init(a);
    defer plan_text.deinit();
    try plan_text.writer.print("{{\"format\":\"exls-rate-plan-1\",\"layers\":{{", .{});
    const gate_up = [_][]const u8{ "2", "1.5", "3", "2", "1.5", "3", "2", "1.5" };
    const down = [_][]const u8{ "2", "2", "4", "2", "2", "4", "2", "2" };
    for (0..cfg.layers) |li| {
        try plan_text.writer.print("{s}\"{d}\":{{\"gate_up\":[", .{ if (li == 0) "" else ",", li });
        for (0..cfg.experts) |e| try plan_text.writer.print("{s}{s}", .{ if (e == 0) "" else ",", gate_up[e % gate_up.len] });
        try plan_text.writer.writeAll("],\"down\":[");
        for (0..cfg.experts) |e| try plan_text.writer.print("{s}{s}", .{ if (e == 0) "" else ",", down[e % down.len] });
        try plan_text.writer.writeAll("]}");
    }
    try plan_text.writer.writeAll("}}");
    var plan = try rate_plan.parse(a, plan_text.written(), cfg.experts, cfg.layers);
    defer plan.deinit();

    const kept = std.c.getenv("SUSHI_DSV41_CONVERT_OUT");
    const out_dir = if (kept) |k| try a.dupe(u8, std.mem.span(k)) else try std.fmt.allocPrint(a, "{s}/pack", .{root});
    defer a.free(out_dir);
    const calib = try std.fmt.allocPrint(a, "{s}/calib", .{cap_dir});
    defer a.free(calib);
    const summary = try run(a, io, mlx.gpuStream(), .{
        .model_dir = src_root,
        .out_dir = out_dir,
        .capture_dir = calib,
        .rate = .{ .n = 32 },
        .plan = &plan,
        .min_rows = 1,
        .version = "test",
        .anchor_stride = 3,
        .anchor_batch = 8,
        .trunk_rate = recipe.trunk,
        .drafter_rate = recipe.drafter,
        .drafter_experts_rate = recipe.drafter_experts,
    }, null);
    try testing.expectEqual(@as(usize, cfg.layers), summary.layers_written);
    try expectTinyRates(a, io, out_dir, recipe);
    if (recipe.resumed) {
        const out2 = try std.fmt.allocPrint(a, "{s}/pack-resumed", .{root});
        defer a.free(out2);
        const cut = [_]u32{ 0, 4 };
        const again: Options = .{ .model_dir = src_root, .out_dir = out2, .capture_dir = calib, .rate = .{ .n = 32 }, .plan = &plan, .min_rows = 1, .version = "test", .anchor_stride = 3, .anchor_batch = 8, .trunk_rate = recipe.trunk, .drafter_rate = recipe.drafter, .drafter_experts_rate = recipe.drafter_experts };
        var first = again;
        first.only_layers = &cut;
        _ = try run(a, io, mlx.gpuStream(), first, null);
        var second = again;
        second.resuming = true;
        const rest = try run(a, io, mlx.gpuStream(), second, null);
        try testing.expectEqual(@as(usize, cut.len), rest.layers_reused);
        try expectSamePack(a, io, out_dir, out2);
    }
    try testing.expectEqual(@as(usize, cfg.n_mtp), summary.stages_written);
    try testing.expectEqual(@as(usize, 0), summary.trunk_identity);

    // A rerun into the finished pack is refused and leaves it as it is.
    try testing.expectError(error.ResumePackComplete, run(a, io, mlx.gpuStream(), .{ .model_dir = src_root, .out_dir = out_dir, .capture_dir = calib, .rate = .{ .n = 32 }, .plan = &plan, .min_rows = 1, .version = "test", .resuming = true, .anchor_stride = 3, .anchor_batch = 8, .trunk_rate = recipe.trunk, .drafter_rate = recipe.drafter, .drafter_experts_rate = recipe.drafter_experts }, null));

    const ids = [_]u32{ 3, 17, 42, 9, 100, 5, 77, 31, 64, 12, 8, 150 };
    const want = blk: {
        const l = try loadTiny(a, io, src_root);
        defer freeTiny(a, l);
        break :blk try lastLogits(a, l, &ids);
    };
    defer a.free(want);
    const l = try loadTiny(a, io, out_dir);
    defer freeTiny(a, l);
    try testing.expect(l.config.dsv41.?.expert_dec != null and l.config.dsv41.?.trunk_dec != null);
    const got = try lastLogits(a, l, &ids);
    defer a.free(got);
    try testing.expectEqual(want.len, got.len);
    for (got) |v| try testing.expect(std.math.isFinite(v));
    if (kept != null) {
        const link = try std.fmt.allocPrintSentinel(a, "{s}/source", .{out_dir}, 0);
        defer a.free(link);
        const target = try std.fmt.allocPrintSentinel(a, "{s}", .{std.mem.span(src_env)}, 0);
        defer a.free(target);
        _ = std.c.unlink(link.ptr);
        _ = std.c.symlink(target.ptr, link.ptr);
        // The serving parity tests read the fixture beside the pack.
        for ([_][]const u8{ "fixture.safetensors", "fixture_noqat.safetensors", "qat.safetensors" }) |name| {
            const fx_link = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ out_dir, name }, 0);
            defer a.free(fx_link);
            const fx_target = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ std.mem.span(src_env), name }, 0);
            defer a.free(fx_target);
            _ = std.c.unlink(fx_link.ptr);
            _ = std.c.symlink(fx_target.ptr, fx_link.ptr);
        }
    }
}

test "dsv41 convert: a tiny checkpoint's layer-major capture converts end to end into a pack the engine serves (DSV41_STQ_SRC)" {
    try convertTiny(.{});
}

test "dsv41 convert: the trunk, DSpark trunk and DSpark expert rates a run names are the rates its shards stamp and the engine serves (DSV41_STQ_SRC)" {
    try convertTiny(.{ .trunk = .{ .n = 80 }, .drafter = .{ .n = 96 }, .drafter_experts = .{ .n = 40 } });
}

test "dsv41 convert: a stopped run resumed writes the pack an uninterrupted run writes (DSV41_STQ_SRC)" {
    try convertTiny(.{ .resumed = true });
}

test "dsv41 convert: a DSpark trunk with no rate of its own is written at the trunk's (DSV41_STQ_SRC)" {
    try convertTiny(.{ .trunk = .{ .n = 80 } });
}

// `DSV41_RT_PACK` (a pack with `model-layer-LL.safetensors`), `DSV41_RT_SRC` (its source checkpoint), `DSV41_RT_PLAN`
// (the rate plan it was converted with), `DSV41_RT_CAPTURE` (the capture's calib dir) and `DSV41_RT_LAYER`: a real
// layer's banks decode through the engine's reconstruction to the source weights, per rate group.
test "dsv41 convert: a real layer shard decodes to its source weights and keeps the routed rows' outputs (DSV41_RT_PACK)" {
    const pack_env = std.c.getenv("DSV41_RT_PACK") orelse return error.SkipZigTest;
    const src_env = std.c.getenv("DSV41_RT_SRC") orelse return error.SkipZigTest;
    const plan_env = std.c.getenv("DSV41_RT_PLAN") orelse return error.SkipZigTest;
    const calib_env = std.c.getenv("DSV41_RT_CAPTURE") orelse return error.SkipZigTest;
    const layer: u32 = if (std.c.getenv("DSV41_RT_LAYER")) |l| try std.fmt.parseInt(u32, std.mem.span(l), 10) else 0;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const cwd = std.Io.Dir.cwd();
    const report = std.c.getenv("SUSHI_DSV41_RT_REPORT") != null;

    var src_dir = try cwd.openDir(io, std.mem.span(src_env), .{ .iterate = true });
    defer src_dir.close(io);
    var src = try source.SafetensorsSource.open(a, io, src_dir);
    defer src.deinit();
    const plan_text = try readWhole(a, io, std.mem.span(plan_env));
    defer a.free(plan_text);
    var plan = try rate_plan.parse(a, plan_text, 384, 40);
    defer plan.deinit();
    const classes = plan.classes(layer) orelse return error.SkipZigTest;

    var name_buf: [64]u8 = undefined;
    const shard_path = try std.fmt.allocPrint(a, "{s}/{s}", .{ std.mem.span(pack_env), try layerExpertsFile(&name_buf, layer) });
    defer a.free(shard_path);
    const shard_len = try fileSize(a, shard_path);
    const raw = try a.alignedAlloc(u8, .of(u16), shard_len);
    defer a.free(raw);
    {
        var f = try cwd.openFile(io, shard_path, .{});
        defer f.close(io);
        try pack.readAllChunked(f, io, raw, 0, pack.read_chunk);
    }
    var file = try st.parse(a, raw);
    defer file.deinit(a);

    const hidden: usize = 5120;
    const inter: usize = 2304;
    // The first routed rows of the capture, for the output-space error of gate and up.
    const sample_tokens: usize = 8192;
    const calib = std.mem.span(calib_env);
    const routes_path = try std.fmt.allocPrint(a, "{s}/route-{d:0>2}.u16", .{ calib, layer });
    defer a.free(routes_path);
    const rows_path = try std.fmt.allocPrint(a, "{s}/mlp-in-{d:0>2}.bf16", .{ calib, layer });
    defer a.free(rows_path);
    const ids = try readRoutes(a, io, routes_path, 384);
    defer a.free(ids);
    const x_bf16 = try a.alloc(u16, sample_tokens * hidden);
    defer a.free(x_bf16);
    {
        var f = try cwd.openFile(io, rows_path, .{});
        defer f.close(io);
        try pack.readAllChunked(f, io, std.mem.sliceAsBytes(x_bf16), 0, pack.read_chunk);
    }

    const weight = try a.alloc(f32, hidden * inter);
    defer a.free(weight);
    const decoded = try a.alloc(u16, hidden * inter);
    defer a.free(decoded);
    const xf = try a.alloc(f32, hidden);
    defer a.free(xf);
    const y_src = try a.alloc(f32, inter);
    defer a.free(y_src);
    const y_err = try a.alloc(f32, inter);
    defer a.free(y_err);
    for (classes, 0..) |cls, g| {
        const picks = [_]usize{ 0, cls.ids.len / 2, cls.ids.len - 1 };
        for (picks) |local| {
            const e = cls.ids[local];
            for ([_]source.ProjRole{ .gate, .up, .down }) |role| {
                const in: usize = if (role == .down) inter else hidden;
                const out: usize = if (role == .down) hidden else inter;
                const rate = if (role == .down) cls.down else cls.gate_up;
                const tiles = (in / 16) * (out / 16) * rate.n;
                var key: [160]u8 = undefined;
                const trellis = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&key, "model.layers.{d}.mlp.switch_mlp.{s}.g{d}.trellis", .{ layer, role.suffix(), g })) orelse return error.MissingTensor);
                const suh = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&key, "model.layers.{d}.mlp.switch_mlp.{s}.g{d}.suh", .{ layer, role.suffix(), g })) orelse return error.MissingTensor);
                const svh = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&key, "model.layers.{d}.mlp.switch_mlp.{s}.g{d}.svh", .{ layer, role.suffix(), g })) orelse return error.MissingTensor);
                try src.readExpertPublic(a, .deepseek_v41, layer, false, role, e, in, out, weight);
                try format.reconstructPublic(a, trellis[local * tiles ..][0..tiles], suh[local * in ..][0..in], svh[local * out ..][0..out], in, out, rate, expert_dec, decoded);
                var num: f64 = 0;
                var den: f64 = 0;
                for (weight, decoded) |w, bits| {
                    const d = @as(f64, format.f16BitsToF32(bits)) - w;
                    num += d * d;
                    den += @as(f64, w) * w;
                }
                const rel = @sqrt(num / den);
                // Output space (gate, up): the error of the routed rows' projections over their size.
                var out_num: f64 = 0;
                var out_den: f64 = 0;
                var used: usize = 0;
                if (role != .down) {
                    for (0..sample_tokens) |t| {
                        var routed = false;
                        for (ids[t * 6 ..][0..6]) |id| routed = routed or @as(u32, @intCast(id)) == e;
                        if (!routed or used >= 64) continue;
                        used += 1;
                        for (xf, x_bf16[t * hidden ..][0..hidden]) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
                        @memset(y_src, 0);
                        @memset(y_err, 0);
                        for (xf, 0..) |xi, i| {
                            if (xi == 0) continue;
                            const wrow = weight[i * out ..][0..out];
                            const drow = decoded[i * out ..][0..out];
                            for (wrow, drow, y_src, y_err) |w, dbits, *ys, *ye| {
                                ys.* += xi * w;
                                ye.* += xi * (format.f16BitsToF32(dbits) - w);
                            }
                        }
                        for (y_src, y_err) |ys, ye| {
                            out_den += @as(f64, ys) * ys;
                            out_num += @as(f64, ye) * ye;
                        }
                    }
                }
                const out_rel = if (out_den > 0) @sqrt(out_num / out_den) else 0;
                if (report) std.debug.print("rt layer {d} group {d} (gate_up n{d}, down n{d}) expert {d} {s}: weight rel {d:.4}, output rel {d:.4} over {d} rows\n", .{ layer, g, cls.gate_up.n, cls.down.n, e, role.suffix(), rel, out_rel, used });
                // White-noise K1.5 on N(0,1) is 0.37 relative; LDLQ trades weight error for output error.
                try testing.expect(rel < 0.9);
                try testing.expect(out_rel < 0.6);
            }
        }
    }
}
