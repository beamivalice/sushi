//! Hy4-Flash (`hy_v4_flash`) STQ packs. The routed experts are PAIR1 rate groups, the fused `gate_up_proj`
//! `[E, 2I, L]` split into gate (rows 0..I) and up (rows I..2I), the router rows in group order. The trunk linears on
//! the 128 grid (attention, dense layer 0, shared expert, latent down/up, the DFly2 drafter's), the embedding and the
//! head are SPEC1 K6 banks. The Engram tables are the source shards cloned (BF16, as stored). Every other tensor is
//! copied as stored.
//! The codes (`Codes`): `synthetic` writes random trellis codes, scales matched to each source tensor's RMS, with no
//! Hessian and no search: the real format byte for byte, weights that mean nothing, for building kernels and loaders.
//! `identity_hessian` rounds every bank by LDLQ and the trellis search against an identity Hessian (no calibration
//! capture): real weights, a correctness fixture rather than a quality pack. `capture` takes the Hessians from a stored
//! calibration capture (`calib/`, DeepSeek-V4.1's contract): each MoE layer's routed experts from the routed rows of
//! `mlp-in-LL` (the latent `lat` the experts read, `route-LL`), down from the source expert's clamped SwiGLU over the
//! same rows; the trunk linears from the X^T X sums `trunk-LL` (`layers.L.<tap kind>`) and the head from `trunk-head`
//! (`head_in`). The embedding and the DFly2 drafter have no capture and round against the identity.
const std = @import("std");
const host = @import("mlx_host");
const api = @import("sushi_exl3");
const format = api.format;
const codebook_tables = api.codebook_tables;
const source = @import("source.zig");
const convert = @import("convert.zig");
const shard = @import("shard.zig");
const pack = @import("pack.zig");
const st = @import("safetensors.zig");
const rate_plan = @import("rate_plan.zig");
const pool_mod = @import("pool.zig");
const encode = @import("encode.zig");
const hessians = @import("hessians.zig");
const stages_mod = @import("stages.zig");
const dsv41 = @import("dsv41.zig");
const trunk_from = @import("trunk_from.zig");
const quantize = @import("quantize.zig");
const metal_search = @import("metal_search.zig");
const mlx = host.mlx;

const hy4_config = host.hy4_config;
const census = host.hy4_census;

pub const default_trunk_rate = format.Rate{ .n = 96 };
pub const default_min_rows = 64;
/// The seed the expert converter's signs use (`convert.Policy.seed`).
const sign_seed: u64 = 0x5eed_e3c0_de00_0001;
/// The working memory one trunk search batch may take (`trunkJobBytes` per bank).
const trunk_batch_bytes: usize = 24 << 30;

/// One bank's share of a batch: its weight, Hessian and signs here, and `prepareBatch`'s regularized weight, factor
/// and damped Hessian.
fn trunkJobBytes(in: usize, out: usize) usize {
    return 4 * (3 * in * in + 2 * in * out + in + out);
}
pub const trunk_dec = format.Decode{ .codebook = .spec1, .window = .w15, .table = codebook_tables.indexOf("spec1-w15") };
pub const default_expert_dec = format.Decode{ .codebook = .pair1, .window = .w16, .table = codebook_tables.indexOf("pair1-w16-v1") };

pub const Codes = enum {
    synthetic,
    identity_hessian,
    capture,

    /// The `hessian` stamp of the pack's banks.
    fn hessian(self: Codes) []const u8 {
        return switch (self) {
            .synthetic => "none",
            .identity_hessian => "identity",
            .capture => "capture",
        };
    }
};

pub const Options = struct {
    model_dir: []const u8,
    out_dir: []const u8,
    /// The routed experts' rate where the plan names none.
    rate: format.Rate,
    plan: ?*const rate_plan.Plan = null,
    /// The routed experts' codec: PAIR1 (K1.5 to K4) or SPEC1 `spec1-w15` (any K the table holds).
    expert_dec: format.Decode = default_expert_dec,
    /// Every trunk, embedding and head bank's rate.
    trunk_rate: format.Rate = default_trunk_rate,
    /// The DFly2 drafter's banks' rate; null: `trunk_rate`.
    drafter_rate: ?format.Rate = null,
    /// Only these layers' expert and trunk shards, nothing else (a timing run): no drafter, globals, index or config.
    only_layers: ?[]const u32 = null,
    codes: Codes,
    /// The capture's `calib/` directory (`codes` `.capture`).
    capture_dir: []const u8 = "",
    /// Fewer routed rows (or summed rows) than this round against the identity, counted in the summary.
    min_rows: u64 = default_min_rows,
    /// Experts per search batch: the converter's peak memory.
    batch: usize = 16,
    pin_pass: u32 = encode.default_pin_pass,
    /// Continue an interrupted pack in `out_dir`: a shard already stamped with this run's `pack_revision` is kept.
    resuming: bool = false,
    /// Seeds every synthetic tensor's codes, with its name.
    seed: u64 = 0x4879_3453_594e_5448,
    version: []const u8,

    pub fn drafterRate(self: *const Options) format.Rate {
        return self.drafter_rate orelse self.trunk_rate;
    }
};

pub const Summary = struct {
    expert_layers: usize = 0,
    expert_groups: usize = 0,
    trunk_banks: usize = 0,
    stored: usize = 0,
    engram_tables: usize = 0,
    /// With a capture, banks rounded against the identity because the capture taps no input of theirs: the embedding's
    /// and the DFly2 drafter's.
    untapped_banks: usize = 0,
    /// With a capture, banks rounded against the identity because their routed or summed rows number under `min_rows`.
    short_banks: usize = 0,
    bytes_written: u64 = 0,
    /// With `resuming`: layers whose experts shard, and trunk, drafter or globals shards, this run's revision already held.
    experts_resumed: usize = 0,
    units_resumed: usize = 0,
    pack_revision: [64]u8 = undefined,
    pack_revision_len: usize = 0,

    pub fn revision(self: *const Summary) []const u8 {
        return self.pack_revision[0..self.pack_revision_len];
    }
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

pub const globals_file = "model-globals.safetensors";
pub const dfly_file = "model-dfly.safetensors";

pub fn layerExpertsFile(buf: []u8, layer: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "model-layer-{d:0>2}.safetensors", .{layer});
}

pub fn layerTrunkFile(buf: []u8, layer: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "model-trunk-{d:0>2}.safetensors", .{layer});
}

/// The source's `config.json`, parsed by the engine's own Hy4 parser.
fn readConfig(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !struct { raw: []u8, cfg: hy4_config.Config } {
    const raw = try dir.readFileAlloc(io, "config.json", a, .limited(16 << 20));
    errdefer a.free(raw);
    if (try source.identifyConfigJson(a, raw) != .hy_v4_flash) return error.ArchitectureNotConvertible;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.SourceConfigInvalid;
    const text = parsed.value.object.get("text_config") orelse return error.SourceConfigInvalid;
    if (text != .object) return error.SourceConfigInvalid;
    return .{ .raw = raw, .cfg = try hy4_config.parse(text.object, parsed.value.object) };
}

/// Where a source tensor goes.
const Form = enum {
    /// A SPEC1 K6 bank `<base>.{trellis,suh,svh}` replacing `<base>.weight`.
    trunk_bank,
    /// A layer's fused routed banks, written as rate groups in the layer's experts shard.
    routed,
    /// Router rows, reordered into group order.
    router,
    /// The source shard is cloned; the index points the table at the clone.
    engram_table,
    stored,
};

fn onGrid(shape: []const usize) bool {
    return shape.len == 2 and shape[0] % 128 == 0 and shape[1] % 128 == 0;
}

/// A linear of the trunk, the drafter, the embedding or the head becomes a bank when both sides sit on the 128 grid
/// (`kv_a_proj_with_mqa`'s 576 rows do not, and stay as stored). The indexer and Engram projections stay as stored.
fn formOf(role: census.Role, name: []const u8, loc: source.TensorLoc) Form {
    return switch (role) {
        .routed_experts => .routed,
        .router => .router,
        .engram_table => .engram_table,
        .embedding, .lm_head, .attention, .dense_ffn, .shared_experts, .latent_proj, .dfly2 => if (std.mem.endsWith(u8, name, ".weight") and
            std.mem.eql(u8, loc.dtype, "BF16") and onGrid(loc.shape)) .trunk_bank else .stored,
        .final_norm, .hc_head, .layer_norm, .hc_mix, .indexer, .ffn_norm, .engram, .vision => .stored,
    };
}

/// The shard a tensor's unit writes: its layer's trunk, the drafter's, or the globals.
fn unitOf(entry: census.Entry) Unit {
    if (entry.role == .dfly2) return .dfly;
    if (entry.layer) |l| return .{ .layer = l };
    return .globals;
}

const Unit = union(enum) {
    layer: u32,
    dfly,
    globals,

    fn eql(x: Unit, y: Unit) bool {
        return switch (x) {
            .layer => |l| y == .layer and y.layer == l,
            .dfly => y == .dfly,
            .globals => y == .globals,
        };
    }
};

// ── synthetic codes ─────────────────────────────────────────────────────────────────────────────────

fn tensorSeed(seed: u64, name: []const u8) u64 {
    return std.hash.Wyhash.hash(seed, name);
}

fn fillCodes(dst: []u16, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    prng.random().bytes(std.mem.sliceAsBytes(dst));
}

/// The RMS of the weights random codes decode to at unit scales: a scale of `rms / codeRms` gives a bank of RMS `rms`.
fn codeRms(a: std.mem.Allocator, rate: format.Rate, dec: format.Decode) !f32 {
    const side = 256;
    const trellis = try a.alloc(u16, (side / 16) * (side / 16) * rate.n);
    defer a.free(trellis);
    fillCodes(trellis, 0x5ca1e ^ rate.n);
    var ones: [side]u16 = undefined;
    @memset(&ones, format.f32ToF16Bits(1));
    const w = try a.alloc(u16, side * side);
    defer a.free(w);
    try format.reconstructPublic(a, trellis, &ones, &ones, side, side, rate, dec, w);
    var sum: f64 = 0;
    for (w) |b| {
        const v: f64 = format.f16BitsToF32(b);
        sum += v * v;
    }
    const rms: f32 = @floatCast(@sqrt(sum / @as(f64, @floatFromInt(w.len))));
    if (!(rms > 0) or !std.math.isFinite(rms)) return error.SyntheticCodeRms;
    return rms;
}

fn rmsBf16(raw: []const u8) f32 {
    var sum: f64 = 0;
    const n = raw.len / 2;
    for (0..n) |i| {
        const bits = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
        const v: f64 = @as(f32, @bitCast(@as(u32, bits) << 16));
        sum += v * v;
    }
    return @floatCast(@sqrt(sum / @as(f64, @floatFromInt(@max(n, 1)))));
}

/// One synthetic bank `[experts, in/16, out/16, n]`: random codes, `suh` the RMS-matching scale, `svh` one.
const SynthBank = struct {
    trellis: []u16,
    suh: []u16,
    svh: []u16,

    fn init(a: std.mem.Allocator, experts: usize, in: usize, out: usize, rate: format.Rate, scale: f32, seed: u64) !SynthBank {
        const bank = try alloc(a, experts, in, out, rate);
        fillCodes(bank.trellis, seed);
        @memset(bank.suh, format.f32ToF16Bits(scale));
        @memset(bank.svh, format.f32ToF16Bits(1));
        return bank;
    }

    fn alloc(a: std.mem.Allocator, experts: usize, in: usize, out: usize, rate: format.Rate) !SynthBank {
        const trellis = try a.alloc(u16, experts * (in / 16) * (out / 16) * rate.n);
        errdefer a.free(trellis);
        const suh = try a.alloc(u16, experts * in);
        errdefer a.free(suh);
        return .{ .trellis = trellis, .suh = suh, .svh = try a.alloc(u16, experts * out) };
    }

    fn deinit(self: *SynthBank, a: std.mem.Allocator) void {
        a.free(self.trellis);
        a.free(self.suh);
        a.free(self.svh);
    }
};

// ── the writer ──────────────────────────────────────────────────────────────────────────────────────

const Writer = struct {
    a: std.mem.Allocator,
    io: std.Io,
    src: *const source.SafetensorsSource,
    out: std.Io.Dir,
    opts: *const Options,
    /// Each source tensor's census entry, by name.
    entries: std.StringHashMap(census.Entry),
    s: mlx.mlx_stream,
    pool: *pool_mod.Pool,
    trunk_code_rms: f32,
    drafter_code_rms: f32,
    /// `codeRms` of the expert codec by rate, filled on first use.
    expert_code_rms: [format.Rate.max_n + 1]f32 = @splat(0),
    revision: []const u8,
    summary: *Summary,
    stages: stages_mod.Stages = .{},
    cfg: *const hy4_config.Config,
    /// Whether the routed experts' shards are pair-major (`Naming.pair_major`).
    pair_major: bool,
    /// Every trunk bank `roundTrunk` rounded, by base name, until its shard takes it.
    rounded: std.StringHashMap(dsv41.Dense),

    fn deinit(self: *Writer) void {
        var it = self.rounded.iterator();
        while (it.next()) |e| {
            self.a.free(e.key_ptr.*);
            e.value_ptr.deinit(self.a);
        }
        self.rounded.deinit();
    }

    /// The rate of the bank `base`: the drafter's (`model.dfly_layers.*`) or the trunk's.
    fn rateOf(self: *const Writer, base: []const u8) format.Rate {
        return if (std.mem.startsWith(u8, base, "model.dfly_layers.")) self.opts.drafterRate() else self.opts.trunk_rate;
    }

    fn metadata(self: *const Writer, k_buf: []u8, rate: format.Rate) [9][2][]const u8 {
        return .{
            .{ "format", "stq" },
            .{ "codebook", "spec1" },
            .{ "codebook_table", "spec1-w15" },
            .{ "window", "15" },
            .{ "k", rate.kText(k_buf) },
            .{ "encoder", self.opts.version },
            .{ "synthetic", if (self.opts.codes == .synthetic) "true" else "false" },
            .{ "hessian", self.opts.codes.hessian() },
            .{ "pack_revision", self.revision },
        };
    }

    /// The tensors of `unit` in name order, each with its form.
    fn unitNames(self: *const Writer, unit: Unit) ![][]const u8 {
        var names = std.ArrayList([]const u8).empty;
        errdefer names.deinit(self.a);
        var it = self.entries.iterator();
        while (it.next()) |e| {
            if (!unitOf(e.value_ptr.*).eql(unit)) continue;
            const f = formOf(e.value_ptr.role, e.key_ptr.*, self.src.tensors.get(e.key_ptr.*).?);
            if (f == .routed or f == .engram_table) continue;
            try names.append(self.a, e.key_ptr.*);
        }
        std.mem.sortUnstable([]const u8, names.items, {}, struct {
            fn lt(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.lessThan(u8, x, y);
            }
        }.lt);
        return names.toOwnedSlice(self.a);
    }

    /// With `resuming`, whether `file` is already written by this run's recipe: a shard is published whole (written
    /// under a temp name, then renamed or linked), so one present is complete; one of another revision is refused, and so
    /// is a `routed` experts shard in the other tile order than this run writes (the pack never holds both).
    fn resumable(self: *const Writer, file: []const u8, routed: bool) !bool {
        if (!self.opts.resuming) return false;
        var header = trunk_from.readHeader(self.a, self.io, self.out, file) catch |err| return if (err == error.FileNotFound) false else err;
        defer header.parsed.deinit();
        const md = header.root.get("__metadata__") orelse return error.ResumeRevisionMismatch;
        if (md != .object) return error.ResumeRevisionMismatch;
        const rev = md.object.get("pack_revision") orelse return error.ResumeRevisionMismatch;
        if (rev != .string or !std.mem.eql(u8, rev.string, self.revision)) return error.ResumeRevisionMismatch;
        const want: ?[]const u8 = if (routed and self.pair_major) "pair_major" else null;
        const have: ?[]const u8 = if (md.object.get("tile_order")) |t| (if (t == .string) t.string else "") else null;
        if ((want == null) != (have == null) or (want != null and !std.mem.eql(u8, want.?, have.?))) return error.ResumeTileOrderMismatch;
        return true;
    }

    /// A trunk, drafter or globals shard: its banks, its router in group order, the rest as stored.
    fn writeUnit(self: *Writer, unit: Unit, file: []const u8, perm: ?[]const u32) !void {
        const names = try self.unitNames(unit);
        defer self.a.free(names);
        var b = pack.ShardBuilder.init(self.a);
        defer b.deinit();
        for (names) |name| {
            const loc = self.src.tensors.get(name).?;
            switch (formOf(self.entries.get(name).?.role, name, loc)) {
                .trunk_bank => try self.addTrunkBank(&b, name, loc),
                .router => try self.addRouter(&b, name, loc, perm orelse return error.RouterWithoutExperts),
                .stored => {
                    const raw = try self.src.readAllAlloc(self.a, name);
                    defer self.a.free(raw);
                    try b.add(name, loc.dtype, loc.shape, raw);
                    self.summary.stored += 1;
                },
                .routed, .engram_table => unreachable,
            }
        }
        self.summary.bytes_written += b.size();
        var k_buf: [16]u8 = undefined;
        const md = self.metadata(&k_buf, if (unit == .dfly) self.opts.drafterRate() else self.opts.trunk_rate);
        try b.flush(self.io, self.out, file, &md);
    }

    fn addTrunkBank(self: *Writer, b: *pack.ShardBuilder, name: []const u8, loc: source.TensorLoc) !void {
        const out = loc.shape[0];
        const in = loc.shape[1];
        const base = name[0 .. name.len - ".weight".len];
        const rate = self.rateOf(base);
        var bank: SynthBank = undefined;
        switch (self.opts.codes) {
            .synthetic => {
                const raw = try self.src.readAllAlloc(self.a, name);
                defer self.a.free(raw);
                bank = try SynthBank.init(self.a, 1, in, out, rate, rmsBf16(raw) / (if (rate.n == self.opts.trunk_rate.n) self.trunk_code_rms else self.drafter_code_rms), tensorSeed(self.opts.seed, name));
            },
            .identity_hessian, .capture => {
                const kv = self.rounded.fetchRemove(base) orelse return error.TrunkBankNotRounded;
                self.a.free(kv.key);
                bank = .{ .trellis = kv.value.trellis, .suh = kv.value.suh, .svh = kv.value.svh };
            },
        }
        defer bank.deinit(self.a);
        var nb: [192]u8 = undefined;
        try b.add(try std.fmt.bufPrint(&nb, "{s}.trellis", .{base}), "U16", &.{ in / 16, out / 16, rate.n }, std.mem.sliceAsBytes(bank.trellis));
        try b.add(try std.fmt.bufPrint(&nb, "{s}.suh", .{base}), "F16", &.{in}, std.mem.sliceAsBytes(bank.suh));
        try b.add(try std.fmt.bufPrint(&nb, "{s}.svh", .{base}), "F16", &.{out}, std.mem.sliceAsBytes(bank.svh));
        self.summary.trunk_banks += 1;
    }

    /// The captured X^T X of `base`'s input into `h` [in, in], or why the bank rounds against the identity: no capture,
    /// a bank the capture has no tap for (the embedding, the drafter), or a sum of fewer than `min_rows` rows. A tapped
    /// input the capture lacks is an error. One sum is read at a time: a layer's sums hold `o_in` at heads·v squared.
    fn trunkHessian(self: *const Writer, base: []const u8, h: []f32) !enum { captured, no_capture, untapped, short } {
        if (self.opts.codes != .capture) return .no_capture;
        const kind = tapKindOf(base) orelse return .untapped;
        var pb: [std.fs.max_path_bytes]u8 = undefined;
        const file = if (kind.layer) |l| try std.fmt.bufPrint(&pb, "trunk-{d:0>2}.safetensors", .{l}) else "trunk-head.safetensors";
        var nb: [96]u8 = undefined;
        const key = if (kind.layer) |l| try std.fmt.bufPrint(&nb, "layers.{d}.{s}", .{ l, kind.name }) else kind.name;
        var dir = try std.Io.Dir.cwd().openDir(self.io, self.opts.capture_dir, .{});
        defer dir.close(self.io);
        var header = try trunk_from.readHeader(self.a, self.io, dir, file);
        defer header.parsed.deinit();
        var rb: [128]u8 = undefined;
        const rows_v = header.root.get(try std.fmt.bufPrint(&rb, "{s}.rows", .{key})) orelse return error.CaptureSumMissing;
        const gram_v = header.root.get(key) orelse return error.CaptureSumMissing;
        const rows_t = try trunk_from.Tensor.of(self.a, rows_v);
        defer rows_t.deinit(self.a);
        const gram_t = try trunk_from.Tensor.of(self.a, gram_v);
        defer gram_t.deinit(self.a);
        if (!std.mem.eql(u8, gram_t.dtype, "F32") or gram_t.hi - gram_t.lo != h.len * 4 or rows_t.hi - rows_t.lo != 8) return error.CaptureSumShape;
        var f = try dir.openFile(self.io, file, .{});
        defer f.close(self.io);
        var rows: i64 = 0;
        if (try f.readPositionalAll(self.io, std.mem.asBytes(&rows), 8 + header.len + rows_t.lo) != 8) return error.TruncatedSafetensors;
        if (rows < self.opts.min_rows) return .short;
        if (try f.readPositionalAll(self.io, std.mem.sliceAsBytes(h), 8 + header.len + gram_t.lo) != h.len * 4) return error.TruncatedSafetensors;
        return .captured;
    }

    /// Rounds every trunk bank of `units` before any trunk shard is written: the banks of one shape share one LDLQ and
    /// search batch, as many as `trunk_batch_bytes` of working memory holds. One bank alone leaves the GPU idle: each
    /// LDLQ step searches only that bank's `out/16` tiles.
    fn roundTrunk(self: *Writer, units: []const Unit) !void {
        const Job = struct { base: []const u8, in: usize, out: usize, n: u32 };
        var jobs = std.ArrayList(Job).empty;
        defer jobs.deinit(self.a);
        var it = self.entries.iterator();
        while (it.next()) |e| {
            const loc = self.src.tensors.get(e.key_ptr.*).?;
            if (formOf(e.value_ptr.role, e.key_ptr.*, loc) != .trunk_bank) continue;
            const u = unitOf(e.value_ptr.*);
            var wanted = false;
            for (units) |x| wanted = wanted or x.eql(u);
            if (!wanted) continue;
            const base = e.key_ptr.*[0 .. e.key_ptr.len - ".weight".len];
            try jobs.append(self.a, .{ .base = base, .in = loc.shape[1], .out = loc.shape[0], .n = self.rateOf(base).n });
        }
        std.mem.sortUnstable(Job, jobs.items, {}, struct {
            fn lt(_: void, x: Job, y: Job) bool {
                if (x.n != y.n) return x.n < y.n;
                if (x.in != y.in) return x.in < y.in;
                if (x.out != y.out) return x.out < y.out;
                return std.mem.lessThan(u8, x.base, y.base);
            }
        }.lt);
        var i: usize = 0;
        while (i < jobs.items.len) {
            const first = jobs.items[i];
            const cost = trunkJobBytes(first.in, first.out);
            var j = i + 1;
            while (j < jobs.items.len and jobs.items[j].n == first.n and jobs.items[j].in == first.in and jobs.items[j].out == first.out and (j - i + 1) * cost <= trunk_batch_bytes) j += 1;
            const bases = try self.a.alloc([]const u8, j - i);
            defer self.a.free(bases);
            for (bases, jobs.items[i..j]) |*b, job| b.* = job.base;
            try self.roundBatch(bases, first.in, first.out, .{ .n = first.n });
            i = j;
        }
    }

    /// The banks `bases` (all `[out, in]`) in one LDLQ and search batch, into `rounded`.
    fn roundBatch(self: *Writer, bases: []const []const u8, in: usize, out: usize, rate: format.Rate) !void {
        const n = bases.len;
        const bufs = try self.a.alloc([]f32, 3 * n);
        defer self.a.free(bufs);
        var made: usize = 0;
        defer for (bufs[0..made]) |b| self.a.free(b);
        const jobs = try self.a.alloc(metal_search.TensorJob, n);
        defer self.a.free(jobs);
        for (bases, jobs) |base, *job| {
            const w = try self.a.alloc(f32, in * out);
            bufs[made] = w;
            made += 1;
            const h = try self.a.alloc(f32, in * in);
            bufs[made] = h;
            made += 1;
            const sg = try quantize.signs(self.a, sign_seed, base, in + out);
            bufs[made] = sg;
            made += 1;
            try self.src.readLinearPublic(self.a, base, out, in, w);
            const why = try self.trunkHessian(base, h);
            if (why != .captured) {
                @memset(h, 0);
                for (0..in) |r| h[r * in + r] = 1;
            }
            self.summary.untapped_banks += @intFromBool(why == .untapped);
            self.summary.short_banks += @intFromBool(why == .short);
            job.* = .{ .w = w, .h = h, .sign_in = sg[0..in], .sign_out = sg[in..] };
        }
        var prepared = try metal_search.prepareBatch(self.a, self.pool, jobs, in, out, .auto, .f64);
        defer prepared.deinit();
        const results = try self.a.alloc(quantize.Quantized, n);
        defer self.a.free(results);
        try metal_search.finishBatch(self.a, self.s, self.pool, &prepared, rate, trunk_dec, self.opts.pin_pass, &self.stages, results, null);
        var kept: usize = 0;
        errdefer for (results[kept..]) |*r| r.deinit(self.a);
        try self.rounded.ensureUnusedCapacity(@intCast(n));
        for (bases, results) |base, r| {
            const key = try self.a.dupe(u8, base);
            self.rounded.putAssumeCapacity(key, .{ .trellis = r.packed_out, .suh = r.suh, .svh = r.svh });
            kept += 1;
        }
    }

    /// Row i of the stored router is the source's row `perm[i]` (weight and correction bias alike).
    fn addRouter(self: *Writer, b: *pack.ShardBuilder, name: []const u8, loc: source.TensorLoc, perm: []const u32) !void {
        const raw = try self.src.readAllAlloc(self.a, name);
        defer self.a.free(raw);
        if (loc.shape[0] != perm.len or raw.len % perm.len != 0) return error.RouterShape;
        const row = raw.len / perm.len;
        const out = try self.a.alloc(u8, raw.len);
        defer self.a.free(out);
        for (perm, 0..) |p, i| @memcpy(out[i * row ..][0..row], raw[p * row ..][0..row]);
        try b.add(name, loc.dtype, loc.shape, out);
        self.summary.stored += 1;
    }

    /// Layer `layer`'s routed experts as rate groups (`classes` in stored order) in `model-layer-LL.safetensors`.
    fn writeExperts(self: *Writer, layer: u32, classes: []const convert.Class, file: []const u8) !void {
        var nb: [128]u8 = undefined;
        const gu = self.src.tensors.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.experts.gate_up_proj", .{layer})) orelse return error.SourceTensorMissing;
        const dn = self.src.tensors.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.experts.down_proj", .{layer})) orelse return error.SourceTensorMissing;
        const g = try expertGeometry(gu, dn);
        if (self.opts.codes != .synthetic) return self.searchExperts(layer, classes, file, g);
        // Each projection's scale from expert 0's rows: reading every expert would double the run's I/O.
        const first = try self.a.alloc(u8, 2 * g.inter * g.latent * 2);
        defer self.a.free(first);
        try self.src.readRange(gu, 0, first);
        const down0 = try self.a.alloc(u8, g.latent * g.inter * 2);
        defer self.a.free(down0);
        try self.src.readRange(dn, 0, down0);
        const half = g.inter * g.latent * 2;
        const rms = [3]f32{ rmsBf16(first[0..half]), rmsBf16(first[half..]), rmsBf16(down0) };
        const dims = [3][2]usize{ .{ g.latent, g.inter }, .{ g.latent, g.inter }, .{ g.inter, g.latent } };
        const roles = [3][]const u8{ "gate", "up", "down" };

        const banks = try self.a.alloc([3]SynthBank, classes.len);
        defer self.a.free(banks);
        var made: usize = 0;
        defer for (banks[0..made]) |*set| for (set) |*bk| bk.deinit(self.a);
        const groups = try self.a.alloc(shard.Group, classes.len);
        defer self.a.free(groups);
        for (classes, 0..) |c, gi| {
            const rates = [3]format.Rate{ c.gate_up, c.gate_up, c.down };
            for (0..3) |p| {
                var sb: [96]u8 = undefined;
                const seed_name = try std.fmt.bufPrint(&sb, "model.layers.{d}.mlp.experts.{s}.g{d}", .{ layer, roles[p], gi });
                if (self.expert_code_rms[rates[p].n] == 0) self.expert_code_rms[rates[p].n] = try codeRms(self.a, rates[p], self.opts.expert_dec);
                const scale = rms[p] / self.expert_code_rms[rates[p].n];
                banks[gi][p] = SynthBank.init(self.a, c.ids.len, dims[p][0], dims[p][1], rates[p], scale, tensorSeed(self.opts.seed, seed_name)) catch |err| {
                    for (banks[gi][0..p]) |*bk| bk.deinit(self.a);
                    return err;
                };
            }
            made += 1;
            var gb: [3]shard.Bank = undefined;
            for (&gb, banks[gi], dims) |*o, bk, d| o.* = .{ .experts = c.ids.len, .in_features = d[0], .out_features = d[1], .trellis = bk.trellis, .suh = bk.suh, .svh = bk.svh };
            groups[gi] = .{ .banks = gb, .rates = rates };
        }
        var nest_buf: [48]u8 = undefined;
        const provenance = [_][2][]const u8{
            .{ "encoder", self.opts.version },
            .{ "synthetic", if (self.opts.codes == .synthetic) "true" else "false" },
            .{ "hessian", self.opts.codes.hessian() },
            .{ "pack_revision", self.revision },
        };
        try shard.writeGroupsNamed(self.a, self.io, self.out, .{ .nest = try std.fmt.bufPrint(&nest_buf, "model.layers.{d}", .{layer}), .file = file, .pair_major = self.pair_major }, groups, self.opts.expert_dec, &provenance);
        for (groups) |gr| for (gr.banks) |bk| {
            self.summary.bytes_written += (bk.trellis.len + bk.suh.len + bk.svh.len) * 2;
        };
        self.summary.expert_layers += 1;
        self.summary.expert_groups += groups.len;
    }

    /// Layer `layer`'s experts rounded by the expert converter in batches of `batch`, Hessians from the capture's
    /// routed rows or the identity.
    fn searchExperts(self: *Writer, layer: u32, classes: []const convert.Class, file: []const u8, g: ExpertGeometry) !void {
        var widest = format.Rate{ .n = 0 };
        for (classes) |c| widest.n = @max(widest.n, @max(c.gate_up.n, c.down.n));
        var tok_buf: [24]u8 = undefined;
        var rows_buf: [24]u8 = undefined;
        var pin_buf: [24]u8 = undefined;
        var nest_buf: [48]u8 = undefined;
        const geom = convert.Geometry{ .experts = g.experts, .hidden = g.latent, .inter = g.inter };
        var rh: ?hessians.RowsHessians = null;
        var ids: []i32 = &.{};
        defer if (ids.len > 0) self.a.free(ids);
        var x: ?mlx.mlx_array = null;
        defer if (rh) |*r| r.deinit();
        defer if (x) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        if (self.opts.codes == .capture) {
            var pb: [2][std.fs.max_path_bytes]u8 = undefined;
            const routes_path = try std.fmt.bufPrint(&pb[0], "{s}/route-{d:0>2}.u16", .{ self.opts.capture_dir, layer });
            const rf = try rowsFile(self.a, &pb[1], self.opts.capture_dir, layer);
            ids = try dsv41.readRoutes(self.a, self.io, routes_path, g.experts);
            const tokens = ids.len / self.cfg.top_k;
            if (ids.len % self.cfg.top_k != 0 or try dsv41.fileSize(self.a, rf.path) != tokens * g.latent * rf.itemsize) return error.CaptureRowsMismatch;
            x = try dsv41.fileArray(self.a, rf.path, tokens, g.latent, rf.dtype, rf.itemsize);
            rh = try hessians.RowsHessians.init(self.a, self.s, self.src, layer, false, geom, x.?, ids, self.cfg.top_k);
            rh.?.arch = .hy_v4_flash;
            rh.?.limit = self.cfg.swiglu_limit;
        }
        const provenance = [_][2][]const u8{
            .{ "encoder", self.opts.version },
            .{ "quantizer", "ldlq" },
            .{ "synthetic", "false" },
            .{ "hessian", self.opts.codes.hessian() },
            .{ "calib_tokens", try std.fmt.bufPrint(&tok_buf, "{d}", .{ids.len / self.cfg.top_k}) },
            .{ "hessian_min_rows", try std.fmt.bufPrint(&rows_buf, "{d}", .{self.opts.min_rows}) },
            .{ "pin_pass", try std.fmt.bufPrint(&pin_buf, "{d}", .{self.opts.pin_pass}) },
            .{ "pack_revision", self.revision },
        };
        const stats = try convert.convertQwenLayer(self.a, self.s, self.io, self.out, self.src, layer, false, geom, .{
            .rate = widest,
            .dec = self.opts.expert_dec,
            .min_rows = self.opts.min_rows,
            .batch = self.opts.batch,
            .pin_pass = self.opts.pin_pass,
            .provenance = &provenance,
            .classes = classes,
            .arch = .hy_v4_flash,
            .naming = .{ .nest = try std.fmt.bufPrint(&nest_buf, "model.layers.{d}", .{layer}), .file = file, .pair_major = self.pair_major },
        }, if (rh) |*r| r.hessianSource() else null);
        self.stages.merge(&stats.stages);
        self.summary.short_banks += if (self.opts.codes == .capture) stats.gate_up_identity + stats.down_identity else 0;
        self.summary.bytes_written += try fileBytes(self.io, self.out, file);
        self.summary.expert_layers += 1;
        self.summary.expert_groups += classes.len;
    }
};

fn fileBytes(io: std.Io, dir: std.Io.Dir, name: []const u8) !u64 {
    const stat = try dir.statFile(io, name, .{});
    return stat.size;
}

/// The capture's tap kind whose X^T X a linear reads (`hy4_forward.TapKind`), and its layer; null for the embedding, the
/// drafter and the indexer's f32 pooling projections, which the capture does not tap. Linears kept as stored map too, so
/// the capture is checked whole against the source.
fn tapKindOf(base: []const u8) ?struct { name: []const u8, layer: ?u32 } {
    if (std.mem.eql(u8, base, "lm_head")) return .{ .name = "head_in", .layer = null };
    const pre = "model.layers.";
    if (!std.mem.startsWith(u8, base, pre)) return null;
    const dot = std.mem.indexOfScalarPos(u8, base, pre.len, '.') orelse return null;
    const layer = std.fmt.parseInt(u32, base[pre.len..dot], 10) catch return null;
    const leaf = base[dot + 1 ..];
    const table = [_]struct { []const u8, []const u8 }{
        .{ "self_attn.q_a_proj", "attn_in" },
        .{ "self_attn.kv_a_proj_with_mqa", "attn_in" },
        .{ "self_attn.compressed_indexer.token_key_projection", "attn_in" },
        .{ "self_attn.compressed_indexer.head_weight_projection", "attn_in" },
        .{ "self_attn.compressed_indexer.query_projection", "q_lora" },
        .{ "mlp.gate", "ffn_in" },
        .{ "engram_layer.value_proj", "engram_in" },
        .{ "self_attn.linear_gate", "attn_in" },
        .{ "self_attn.q_b_proj", "q_lora" },
        .{ "self_attn.kv_b_proj", "kv_latent" },
        .{ "self_attn.o_proj", "o_in" },
        .{ "mlp.routed_expert_down_proj", "ffn_in" },
        .{ "mlp.shared_experts.gate_proj", "ffn_in" },
        .{ "mlp.shared_experts.up_proj", "ffn_in" },
        .{ "mlp.gate_proj", "ffn_in" },
        .{ "mlp.up_proj", "ffn_in" },
        .{ "mlp.routed_expert_up_proj", "lat_up_in" },
        .{ "mlp.shared_experts.down_proj", "shared_act" },
        .{ "mlp.down_proj", "shared_act" },
    };
    for (table) |row| if (std.mem.eql(u8, leaf, row[0])) return .{ .name = row[1], .layer = layer };
    if (std.mem.startsWith(u8, leaf, "engram_layer.key_projs.")) return .{ .name = "engram_in", .layer = layer };
    return null;
}

/// The capture against the source before anything is written: every MoE layer of `layers` has its routes and rows, and
/// every linear of `layers` (and the head, with `head`) the X^T X sum of its input at its width.
/// A MoE layer's captured rows: `mlp-in-LL.bf16` at the BF16 products, `mlp-in-LL.f32` at the f32 products a lossless
/// teacher runs. The Hessians widen either to f32. A layer holding both is refused, as is one holding neither.
const RowsFile = struct { path: []const u8, dtype: mlx.mlx_dtype, itemsize: usize };

fn rowsFile(a: std.mem.Allocator, buf: []u8, capture_dir: []const u8, layer: usize) !RowsFile {
    const forms = [_]struct { ext: []const u8, dtype: mlx.mlx_dtype, itemsize: usize }{
        .{ .ext = "bf16", .dtype = .bfloat16, .itemsize = 2 },
        .{ .ext = "f32", .dtype = .float32, .itemsize = 4 },
    };
    var found: ?usize = null;
    for (forms, 0..) |f, i| {
        const path = try std.fmt.bufPrint(buf, "{s}/mlp-in-{d:0>2}.{s}", .{ capture_dir, layer, f.ext });
        _ = dsv41.fileSize(a, path) catch |err| if (err == error.CaptureFileMissing) continue else return err;
        if (found != null) return error.CaptureRowsAmbiguous;
        found = i;
    }
    const f = forms[found orelse return error.CaptureFileMissing];
    return .{ .path = try std.fmt.bufPrint(buf, "{s}/mlp-in-{d:0>2}.{s}", .{ capture_dir, layer, f.ext }), .dtype = f.dtype, .itemsize = f.itemsize };
}

fn checkCapture(a: std.mem.Allocator, io: std.Io, src: *const source.SafetensorsSource, cfg: *const hy4_config.Config, capture_dir: []const u8, layers: []const u32, head: bool) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, capture_dir, .{});
    defer dir.close(io);
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    var nb: [128]u8 = undefined;
    for (layers) |l| if (l >= cfg.dense_layers) {
        const gu = src.tensors.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.experts.gate_up_proj", .{l})) orelse return error.SourceTensorMissing;
        const latent = gu.shape[2];
        const routes = try dsv41.fileSize(a, try std.fmt.bufPrint(&pb, "{s}/route-{d:0>2}.u16", .{ capture_dir, l }));
        if (routes % (2 * cfg.top_k) != 0) return error.CaptureRowsMismatch;
        const rf = try rowsFile(a, &pb, capture_dir, l);
        if (try dsv41.fileSize(a, rf.path) != routes / (2 * cfg.top_k) * latent * rf.itemsize) return error.CaptureRowsMismatch;
    };
    for (0..layers.len + @intFromBool(head)) |i| {
        const layer: ?u32 = if (i < layers.len) layers[i] else null;
        const file = if (layer) |l| try std.fmt.bufPrint(&pb, "trunk-{d:0>2}.safetensors", .{l}) else "trunk-head.safetensors";
        var header = trunk_from.readHeader(a, io, dir, file) catch |err| return if (err == error.FileNotFound) error.CaptureSumMissing else err;
        defer header.parsed.deinit();
        var it = src.tensors.iterator();
        while (it.next()) |e| {
            const name = e.key_ptr.*;
            if (!std.mem.endsWith(u8, name, ".weight") or e.value_ptr.shape.len != 2) continue;
            const kind = tapKindOf(name[0 .. name.len - ".weight".len]) orelse continue;
            if (!std.meta.eql(kind.layer, layer)) continue;
            const key = if (kind.layer) |l| try std.fmt.bufPrint(&nb, "layers.{d}.{s}", .{ l, kind.name }) else kind.name;
            var rb: [136]u8 = undefined;
            if (header.root.get(try std.fmt.bufPrint(&rb, "{s}.rows", .{key})) == null) return error.CaptureSumMissing;
            const gram = try trunk_from.Tensor.of(a, header.root.get(key) orelse return error.CaptureSumMissing);
            defer gram.deinit(a);
            const in = e.value_ptr.shape[1];
            if (!std.mem.eql(u8, gram.dtype, "F32") or !std.mem.eql(usize, gram.shape, &.{ in, in })) return error.CaptureSumShape;
        }
    }
}

pub const ExpertGeometry = struct { experts: usize, latent: usize, inter: usize };

/// The fused banks' widths: `gate_up_proj` `[E, 2I, L]`, `down_proj` `[E, L, I]`, both sides on the 128 grid.
fn expertGeometry(gu: source.TensorLoc, dn: source.TensorLoc) !ExpertGeometry {
    if (gu.shape.len != 3 or dn.shape.len != 3 or !std.mem.eql(u8, gu.dtype, "BF16") or !std.mem.eql(u8, dn.dtype, "BF16")) return error.Hy4ExpertLayout;
    const g = ExpertGeometry{ .experts = gu.shape[0], .latent = gu.shape[2], .inter = gu.shape[1] / 2 };
    if (gu.shape[1] % 2 != 0 or dn.shape[0] != g.experts or dn.shape[1] != g.latent or dn.shape[2] != g.inter) return error.Hy4ExpertLayout;
    if (g.latent % 128 != 0 or g.inter % 128 != 0) return error.Hy4ExpertsOffGrid;
    return g;
}

fn uniformClasses(a: std.mem.Allocator, experts: usize, rate: format.Rate) ![]convert.Class {
    const ids = try a.alloc(u32, experts);
    errdefer a.free(ids);
    for (ids, 0..) |*v, e| v.* = @intCast(e);
    const out = try a.alloc(convert.Class, 1);
    out[0] = .{ .gate_up = rate, .down = rate, .ids = ids };
    return out;
}

fn freeClasses(a: std.mem.Allocator, classes: []convert.Class) void {
    for (classes) |c| a.free(c.ids);
    a.free(classes);
}

extern "c" fn clonefile(src: [*:0]const u8, dst: [*:0]const u8, flags: u32) c_int;

/// Each source shard holding an Engram table, cloned copy-on-write as `engram-<shard>`; the clone's other tensors are
/// not indexed. A clone needs the pack on the source's volume: a copy of 127 GB is not done in its place.
fn cloneEngram(w: *Writer, entries: *std.ArrayList(pack.IndexEntry), clones: *std.ArrayList([]u8)) !void {
    var it = w.entries.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.role != .engram_table) continue;
        const shard_name = w.src.shards[w.src.tensors.get(e.key_ptr.*).?.shard_idx];
        const dst_name = try std.fmt.allocPrint(w.a, "engram-{s}", .{shard_name});
        var known = false;
        for (clones.items) |c| known = known or std.mem.eql(u8, c, dst_name);
        if (known) w.a.free(dst_name) else {
            const from = try std.fmt.allocPrintSentinel(w.a, "{s}/{s}", .{ w.opts.model_dir, shard_name }, 0);
            defer w.a.free(from);
            const to = try std.fmt.allocPrintSentinel(w.a, "{s}/{s}", .{ w.opts.out_dir, dst_name }, 0);
            defer w.a.free(to);
            // An interrupted run may have cloned it already; a clone is the source's bytes, so it is made again.
            if (w.opts.resuming) w.out.deleteFile(w.io, dst_name) catch |err| if (err != error.FileNotFound) return err;
            if (clonefile(from.ptr, to.ptr, 0) != 0) {
                w.a.free(dst_name);
                return error.EngramCloneFailed;
            }
            try clones.ensureUnusedCapacity(w.a, 1);
            clones.appendAssumeCapacity(dst_name);
        }
        try entries.ensureUnusedCapacity(w.a, 1);
        const tensor = try w.a.dupe(u8, e.key_ptr.*);
        errdefer w.a.free(tensor);
        entries.appendAssumeCapacity(.{ .tensor = tensor, .file = try std.fmt.allocPrint(w.a, "engram-{s}", .{shard_name}) });
        w.summary.engram_tables += 1;
    }
}

/// The index over every `model-*.safetensors` written and the Engram clones.
fn writeIndex(w: *Writer, clones: *std.ArrayList([]u8)) !void {
    const a = w.a;
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
    var it = w.out.iterate();
    while (try it.next(w.io)) |entry| {
        if (entry.kind != .file or !std.mem.startsWith(u8, entry.name, "model-") or !std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        const z = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ w.opts.out_dir, entry.name }, 0);
        defer a.free(z);
        total += try pack.appendShardEntries(a, z, entry.name, &entries);
        try pack.appendPadEntries(a, z, entry.name, &pads);
    }
    try cloneEngram(w, &entries, clones);
    var eit = w.entries.iterator();
    while (eit.next()) |e| if (e.value_ptr.role == .engram_table) {
        const loc = w.src.tensors.get(e.key_ptr.*).?;
        total += loc.end - loc.start;
    };
    const body = if (w.pair_major) try dsv41.pairMajorIndex(a, entries.items, pads.items, total) else try pack.buildIndex(a, entries.items, total);
    defer a.free(body);
    try w.out.writeFile(w.io, .{ .sub_path = ".model.safetensors.index.json.tmp", .data = body });
    try w.out.rename(".model.safetensors.index.json.tmp", w.out, "model.safetensors.index.json", w.io);
}

/// The source's `config.json` with the pack's stamps: `expert_quant` (the routed experts), `trunk_quant` (the trunk,
/// embedding and head banks), `drafter_quant` (the DFly2 drafter's, when it ships), `engram_tables`, `synthetic` and
/// `pack_revision` (the clones unstamped). `expert_quant` carries `tile_order` when the routed experts are `pair_major`
/// and the plan's `rate_plan_sha256` when a rate plan wrote them.
fn buildConfig(a: std.mem.Allocator, source_config: []const u8, widest: format.Rate, opts: *const Options, revision: []const u8, unstamped: []const []u8, version: []const u8, has_dfly: bool, pair_major: bool) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, ar, source_config, .{}) catch return error.SourceConfigInvalid;
    if (parsed != .object) return error.SourceConfigInvalid;
    var root = parsed.object;
    var k_buf: [3][16]u8 = undefined;
    for ([_]struct { key: []const u8, dec: format.Decode, k: []const u8, pair_major: bool }{
        .{ .key = "expert_quant", .dec = opts.expert_dec, .k = widest.kText(&k_buf[0]), .pair_major = pair_major },
        .{ .key = "trunk_quant", .dec = trunk_dec, .k = opts.trunk_rate.kText(&k_buf[1]), .pair_major = false },
        .{ .key = "drafter_quant", .dec = trunk_dec, .k = opts.drafterRate().kText(&k_buf[2]), .pair_major = false },
    }) |q| {
        if (std.mem.eql(u8, q.key, "drafter_quant") and !has_dfly) continue;
        const spec = pack.expertQuant(q.dec, q.k, &.{});
        var o = std.json.ObjectMap.empty;
        if (q.pair_major) try o.put(ar, "tile_order", .{ .string = "pair_major" });
        try o.put(ar, "format", .{ .string = spec.format });
        try o.put(ar, "k", .{ .number_string = spec.k_text });
        try o.put(ar, "codebook", .{ .string = spec.codebook });
        try o.put(ar, "window", .{ .integer = spec.window });
        try o.put(ar, "codebook_table", .{ .string = spec.table.? });
        try o.put(ar, "codebook_digest", .{ .string = spec.digest.? });
        try o.put(ar, "encoder", .{ .string = version });
        try o.put(ar, "hessian", .{ .string = opts.codes.hessian() });
        if (std.mem.eql(u8, q.key, "expert_quant")) if (opts.plan) |p| try o.put(ar, "rate_plan_sha256", .{ .string = &p.sha256 });
        try root.put(ar, q.key, .{ .object = o });
    }
    try root.put(ar, "engram_tables", .{ .string = "bf16" });
    try root.put(ar, "synthetic", .{ .bool = opts.codes == .synthetic });
    var rev = std.json.ObjectMap.empty;
    try rev.put(ar, "id", .{ .string = revision });
    var list = std.json.Array.init(ar);
    for (unstamped) |u| try list.append(.{ .string = u });
    try rev.put(ar, "unstamped", .{ .array = list });
    try root.put(ar, "pack_revision", .{ .object = rev });
    return std.json.Stringify.valueAlloc(a, std.json.Value{ .object = root }, .{ .whitespace = .indent_2 });
}

/// `synthetic-<12 hex>` of the encoder, the recipe and the source's config and index: the same inputs give the same id.
fn revisionId(buf: []u8, opts: *const Options, source_config: []const u8, index: []const u8, capture_head: []const u8) ![]const u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var k_buf: [16]u8 = undefined;
    h.update(opts.version);
    h.update(opts.rate.kText(&k_buf));
    if (opts.plan) |p| h.update(&p.sha256);
    h.update(std.mem.asBytes(&opts.seed));
    h.update(@tagName(opts.codes));
    h.update(@tagName(opts.expert_dec.codebook));
    h.update(opts.trunk_rate.kText(&k_buf));
    h.update(opts.drafterRate().kText(&k_buf));
    h.update(source_config);
    h.update(index);
    h.update(capture_head);
    h.update(std.mem.asBytes(&opts.min_rows));
    const hex = std.fmt.bytesToHex(h.finalResult(), .lower);
    return std.fmt.bufPrint(buf, "{s}-{s}", .{ switch (opts.codes) {
        .synthetic => "synthetic",
        .identity_hessian => "identity",
        .capture => "capture",
    }, hex[0..12] });
}

fn sayStages(sink: ?Sink, label: []const u8, stages: *const stages_mod.Stages) void {
    var buf: [448]u8 = undefined;
    var sw = std.Io.Writer.fixed(&buf);
    stages.write(&sw) catch {};
    say(sink, "{s}: {s}", .{ label, sw.buffered() });
}

fn secondsBetween(from: std.Io.Timestamp, to: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(from.durationTo(to).nanoseconds)) / 1e9;
}

/// Converts the checkpoint into a pack whose routed experts are pair-major (`trellis_pm`, docs/pack-format.md#tile-order): the one
/// layout a Hy4-Flash pack is made in.
pub fn run(a: std.mem.Allocator, io: std.Io, opts: Options, sink: ?Sink) !Summary {
    return runLayout(a, io, opts, sink, true);
}

/// `run` with the routed experts' layout named. `pair_major` false writes the pack order, which no pack is made in: only the tests
/// call it, as the reference the pair-major write is held against.
fn runLayout(a: std.mem.Allocator, io: std.Io, opts: Options, sink: ?Sink, pair_major: bool) !Summary {
    if (!encode.searchable(opts.expert_dec, opts.rate) or !encode.searchable(trunk_dec, opts.trunk_rate) or
        !encode.searchable(trunk_dec, opts.drafterRate())) return error.RateUnsupported;
    const cwd = std.Io.Dir.cwd();
    var src_dir = try cwd.openDir(io, opts.model_dir, .{ .iterate = true });
    defer src_dir.close(io);
    const conf = try readConfig(a, io, src_dir);
    defer a.free(conf.raw);
    const cfg = conf.cfg;
    if (cfg.swiglu_limit != @round(cfg.swiglu_limit)) return error.SwigluLimitNotWhole;
    if (opts.plan) |plan| {
        if (plan.layers.len != cfg.layers) return error.RateUnsupported;
        for (plan.layers) |cl| if (cl) |classes| for (classes) |c| for ([2]format.Rate{ c.gate_up, c.down }) |r| {
            if (!encode.searchable(opts.expert_dec, r)) return error.RateUnsupported;
        };
    }

    var src = try source.SafetensorsSource.open(a, io, src_dir);
    defer src.deinit();

    var entries = std.StringHashMap(census.Entry).init(a);
    defer entries.deinit();
    var tit = src.tensors.iterator();
    while (tit.next()) |t| try entries.put(t.key_ptr.*, try census.roleOf(&cfg, t.key_ptr.*));

    // Every layer's experts are checked before anything is written.
    for (cfg.dense_layers..cfg.layers) |l| {
        var nb: [2][128]u8 = undefined;
        const gu = src.tensors.get(try std.fmt.bufPrint(&nb[0], "model.layers.{d}.mlp.experts.gate_up_proj", .{l})) orelse return error.SourceTensorMissing;
        const dn = src.tensors.get(try std.fmt.bufPrint(&nb[1], "model.layers.{d}.mlp.experts.down_proj", .{l})) orelse return error.SourceTensorMissing;
        const g = try expertGeometry(gu, dn);
        if (g.experts != cfg.experts) return error.Hy4ExpertLayout;
    }

    try cwd.createDirPath(io, opts.out_dir);
    var out = try cwd.openDir(io, opts.out_dir, .{ .iterate = true });
    defer out.close(io);
    {
        var it = out.iterate();
        if (try it.next(io) != null and !opts.resuming) return error.OutputNotEmpty;
    }
    if (opts.resuming) if (out.statFile(io, "config.json", .{})) |_| return error.ResumePackComplete else |_| {};

    const index_raw = src_dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 << 20)) catch try a.dupe(u8, "");
    defer a.free(index_raw);
    // A capture is named by its head sum, which every window's rows reach.
    var head_sums: ?dsv41.Sums = null;
    defer if (head_sums) |*hs| hs.deinit(a);
    if (opts.codes == .capture) {
        const head_path = try std.fmt.allocPrint(a, "{s}/trunk-head.safetensors", .{opts.capture_dir});
        defer a.free(head_path);
        head_sums = try dsv41.Sums.open(a, io, head_path);
    }
    var summary = Summary{};
    const layers: []const u32 = opts.only_layers orelse blk: {
        const all = try a.alloc(u32, cfg.layers);
        for (all, 0..) |*v, i| v.* = @intCast(i);
        break :blk all;
    };
    defer if (opts.only_layers == null) a.free(layers);
    for (layers) |layer| if (layer >= cfg.layers) return error.LayerOutOfRange;
    if (opts.codes == .capture) try checkCapture(a, io, &src, &cfg, opts.capture_dir, layers, opts.only_layers == null);
    const revision = try revisionId(&summary.pack_revision, &opts, conf.raw, index_raw, if (head_sums) |hs| hs.raw else "");
    summary.pack_revision_len = revision.len;

    const pool = try pool_mod.Pool.init(a, @max(1, @min(pool_mod.cpuCount(), 16) - 1));
    defer pool.deinit();
    var w = Writer{
        .s = mlx.gpuStream(),
        .pool = pool,
        .a = a,
        .io = io,
        .src = &src,
        .out = out,
        .opts = &opts,
        .entries = entries,
        .trunk_code_rms = try codeRms(a, opts.trunk_rate, trunk_dec),
        .drafter_code_rms = try codeRms(a, opts.drafterRate(), trunk_dec),
        .revision = revision,
        .summary = &summary,
        .cfg = &cfg,
        .pair_major = pair_major,
        .rounded = std.StringHashMap(dsv41.Dense).init(a),
    };
    defer w.deinit();
    var widest = opts.rate;
    const perms = try a.alloc(?[]u32, layers.len);
    defer a.free(perms);
    @memset(perms, null);
    defer for (perms) |p| if (p) |x| a.free(x);
    for (layers, perms) |layer, *perm| {
        if (layer < cfg.dense_layers) continue;
        const t0 = std.Io.Timestamp.now(io, .awake);
        var owned: ?[]convert.Class = null;
        defer if (owned) |c| freeClasses(a, c);
        const classes: []const convert.Class = (if (opts.plan) |p| p.classes(layer) else null) orelse blk: {
            owned = try uniformClasses(a, cfg.experts, opts.rate);
            break :blk owned.?;
        };
        for (classes) |c| widest.n = @max(widest.n, @max(c.gate_up.n, c.down.n));
        perm.* = try rate_plan.permutation(a, classes);
        var eb: [64]u8 = undefined;
        const file = try layerExpertsFile(&eb, layer);
        if (try w.resumable(file, true)) {
            summary.experts_resumed += 1;
            continue;
        }
        try w.writeExperts(layer, classes, file);
        say(sink, "layer {d}/{d} experts: {d:.1} s", .{ layer + 1, cfg.layers, secondsBetween(t0, std.Io.Timestamp.now(io, .awake)) });
        sayStages(sink, "experts stages", &w.stages);
        w.stages = .{};
    }

    var has_dfly = false;
    var eit = entries.valueIterator();
    while (eit.next()) |e| has_dfly = has_dfly or e.role == .dfly2;
    const Pending = struct { unit: Unit, file: []const u8, perm: ?[]const u32 };
    const name_bufs = try a.alloc([32]u8, layers.len);
    defer a.free(name_bufs);
    var pending = std.ArrayList(Pending).empty;
    defer pending.deinit(a);
    for (layers, perms, name_bufs) |layer, perm, *nb| {
        try pending.append(a, .{ .unit = .{ .layer = layer }, .file = try layerTrunkFile(nb, layer), .perm = perm });
    }
    if (opts.only_layers == null) {
        if (has_dfly) try pending.append(a, .{ .unit = .dfly, .file = dfly_file, .perm = null });
        try pending.append(a, .{ .unit = .globals, .file = globals_file, .perm = null });
    }
    var units = std.ArrayList(Unit).empty;
    defer units.deinit(a);
    var kept: usize = 0;
    for (pending.items) |p| {
        if (try w.resumable(p.file, false)) {
            summary.units_resumed += 1;
            continue;
        }
        try units.append(a, p.unit);
        pending.items[kept] = p;
        kept += 1;
    }
    pending.shrinkRetainingCapacity(kept);
    if (opts.codes != .synthetic) {
        const t0 = std.Io.Timestamp.now(io, .awake);
        try w.roundTrunk(units.items);
        say(sink, "trunk banks: {d:.1} s", .{secondsBetween(t0, std.Io.Timestamp.now(io, .awake))});
        sayStages(sink, "trunk stages", &w.stages);
    }
    for (pending.items) |p| try w.writeUnit(p.unit, p.file, p.perm);
    if (opts.only_layers != null) return summary;

    var clones = std.ArrayList([]u8).empty;
    defer {
        for (clones.items) |c| a.free(c);
        clones.deinit(a);
    }
    try writeIndex(&w, &clones);
    _ = try pack.copySidecars(a, io, src_dir, out);
    const config = try buildConfig(a, conf.raw, widest, &opts, revision, clones.items, opts.version, has_dfly, pair_major);
    defer a.free(config);
    try out.writeFile(io, .{ .sub_path = ".config.json.tmp", .data = config });
    try out.rename(".config.json.tmp", out, "config.json", io);
    return summary;
}

// ── tests ───────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;

test "hy4 convert: the release-layout tiny's 32-wide experts are refused by name before anything is written (HY4_BF16_TINY)" {
    const src_env = std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const out_dir = try std.fmt.allocPrint(a, "{s}/pack", .{root});
    defer a.free(out_dir);
    try testing.expectError(error.Hy4ExpertsOffGrid, run(a, io, .{ .model_dir = std.mem.span(src_env), .out_dir = out_dir, .rate = .{ .n = 64 }, .codes = .synthetic, .version = "test" }, null));
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "pack", .{}));
}

/// A whole pack file, its header parsed.
const PackFile = struct {
    raw: []align(4) u8,
    file: st.File,

    fn open(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) !PackFile {
        const bytes = try dir.readFileAlloc(io, name, a, .limited(1 << 30));
        defer a.free(bytes);
        const raw = try a.alignedAlloc(u8, .of(u32), bytes.len);
        errdefer a.free(raw);
        @memcpy(raw, bytes);
        return .{ .raw = raw, .file = try st.parse(a, raw) };
    }

    fn deinit(self: *PackFile, a: std.mem.Allocator) void {
        self.file.deinit(a);
        a.free(self.raw);
    }
};

/// The written pack as its index maps it: each name read from the shard the index names, never from a header scan
/// (a clone's other tensors are in its header but not in the index).
const Pack = struct {
    index: std.json.Parsed(std.json.Value),
    weight_map: std.json.ObjectMap,
    files: std.StringHashMap(PackFile),

    fn open(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !Pack {
        const raw = try dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 << 20));
        defer a.free(raw);
        const index = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
        errdefer index.deinit();
        if (index.value != .object) return error.TestIndexShape;
        const wm = index.value.object.get("weight_map") orelse return error.TestIndexShape;
        if (wm != .object) return error.TestIndexShape;
        var self = Pack{ .index = index, .weight_map = wm.object, .files = std.StringHashMap(PackFile).init(a) };
        errdefer {
            var fit = self.files.valueIterator();
            while (fit.next()) |f| f.deinit(a);
            self.files.deinit();
        }
        var it = self.weight_map.iterator();
        while (it.next()) |e| {
            const file = e.value_ptr.*;
            if (file != .string) return error.TestIndexShape;
            const g = try self.files.getOrPut(file.string);
            if (!g.found_existing) g.value_ptr.* = try PackFile.open(a, io, dir, file.string);
        }
        return self;
    }

    fn deinit(self: *Pack, a: std.mem.Allocator) void {
        var it = self.files.valueIterator();
        while (it.next()) |f| f.deinit(a);
        self.files.deinit();
        self.index.deinit();
    }

    fn fileOf(self: *const Pack, name: []const u8) ?[]const u8 {
        const f = self.weight_map.get(name) orelse return null;
        return if (f == .string) f.string else null;
    }

    fn get(self: *const Pack, name: []const u8) ?st.TensorView {
        return self.files.get(self.fileOf(name) orelse return null).?.file.tensors.get(name);
    }
};

fn expectK(v: std.json.Value, want: f64) !void {
    const k: f64 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return error.TestKNotANumber,
    };
    try testing.expectEqual(want, k);
}

fn expectLoc(loc: ?st.TensorView, dtype: []const u8, shape: []const usize) !void {
    const l = loc orelse return error.TestTensorMissing;
    try testing.expectEqualStrings(dtype, l.dtype);
    try testing.expectEqualSlices(usize, shape, l.shape);
}

// `HY4_BF16_TINY_STQ`: the tiny at the trellis geometry (`dump_hy4_fixtures.py --geometry stq`), every expert and trunk
// linear on the 128 grid, with DFly2 tensors.
test "hy4 convert: a synthetic pack holds every census tensor in its stated form and round-trips its stamps (HY4_BF16_TINY_STQ)" {
    const src_env = std.c.getenv("HY4_BF16_TINY_STQ") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src_root = std.mem.span(src_env);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const out_dir = try std.fmt.allocPrint(a, "{s}/pack", .{root});
    defer a.free(out_dir);

    var src_dir = try std.Io.Dir.cwd().openDir(io, src_root, .{ .iterate = true });
    defer src_dir.close(io);
    const conf = try readConfig(a, io, src_dir);
    defer a.free(conf.raw);
    const cfg = conf.cfg;
    var src = try source.SafetensorsSource.open(a, io, src_dir);
    defer src.deinit();

    // Three rate groups a layer, so the router's group order is not the source's.
    var plan_text: std.Io.Writer.Allocating = .init(a);
    defer plan_text.deinit();
    const gate_up = [_][]const u8{ "4", "3", "2.5", "4", "3", "2.5", "4", "3" };
    const down = [_][]const u8{ "4", "4", "3", "4", "4", "3", "4", "4" };
    try plan_text.writer.writeAll("{\"format\":\"exls-rate-plan-1\",\"layers\":{");
    for (cfg.dense_layers..cfg.layers) |li| {
        try plan_text.writer.print("{s}\"{d}\":{{\"gate_up\":[", .{ if (li == cfg.dense_layers) "" else ",", li });
        for (0..cfg.experts) |e| try plan_text.writer.print("{s}{s}", .{ if (e == 0) "" else ",", gate_up[e % gate_up.len] });
        try plan_text.writer.writeAll("],\"down\":[");
        for (0..cfg.experts) |e| try plan_text.writer.print("{s}{s}", .{ if (e == 0) "" else ",", down[e % down.len] });
        try plan_text.writer.writeAll("]}");
    }
    try plan_text.writer.writeAll("}}");
    var plan = try rate_plan.parse(a, plan_text.written(), cfg.experts, cfg.layers);
    defer plan.deinit();

    const summary = try run(a, io, .{ .model_dir = src_root, .out_dir = out_dir, .rate = .{ .n = 64 }, .plan = &plan, .codes = .synthetic, .version = "test" }, null);
    try testing.expectEqual(@as(usize, cfg.layers - cfg.dense_layers), summary.expert_layers);
    try testing.expect(std.mem.startsWith(u8, summary.revision(), "synthetic-"));
    // A second run over the same inputs names the same revision.
    {
        const again_dir = try std.fmt.allocPrint(a, "{s}/again", .{root});
        defer a.free(again_dir);
        const again = try run(a, io, .{ .model_dir = src_root, .out_dir = again_dir, .rate = .{ .n = 64 }, .plan = &plan, .codes = .synthetic, .version = "test" }, null);
        try testing.expectEqualStrings(summary.revision(), again.revision());
    }

    try verifyPack(a, io, &src, &cfg, out_dir, &plan, &summary, .synthetic);
}

// `HY4_BF16_TINY_STQ` rounded for real against an identity Hessian: the same layout, weights near the source's.
test "hy4 convert: an identity-Hessian pack holds every census tensor and decodes near the source (HY4_BF16_TINY_STQ)" {
    const src_env = std.c.getenv("HY4_BF16_TINY_STQ") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src_root = std.mem.span(src_env);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const out_dir = try std.fmt.allocPrint(a, "{s}/pack", .{root});
    defer a.free(out_dir);
    var src_dir = try std.Io.Dir.cwd().openDir(io, src_root, .{ .iterate = true });
    defer src_dir.close(io);
    const conf = try readConfig(a, io, src_dir);
    defer a.free(conf.raw);
    var src = try source.SafetensorsSource.open(a, io, src_dir);
    defer src.deinit();
    const summary = try run(a, io, .{ .model_dir = src_root, .out_dir = out_dir, .rate = .{ .n = 64 }, .codes = .identity_hessian, .version = "test" }, null);
    try verifyPack(a, io, &src, &conf.cfg, out_dir, null, &summary, .identity_hessian);

    // A bank rounded in its family's batch is the bank rounded alone.
    var out = try std.Io.Dir.cwd().openDir(io, out_dir, .{});
    defer out.close(io);
    var pk = try Pack.open(a, io, out);
    defer pk.deinit(a);
    const pool = try pool_mod.Pool.init(a, 4);
    defer pool.deinit();
    var stages: stages_mod.Stages = .{};
    var nb: [96]u8 = undefined;
    for ([_][]const u8{ "self_attn.q_a_proj", "self_attn.o_proj" }) |leaf| {
        const base = try std.fmt.bufPrint(&nb, "model.layers.{d}.{s}", .{ conf.cfg.dense_layers, leaf });
        var wb: [128]u8 = undefined;
        const loc = src.tensors.get(try std.fmt.bufPrint(&wb, "{s}.weight", .{base})).?;
        const w = try a.alloc(f32, loc.shape[0] * loc.shape[1]);
        defer a.free(w);
        try src.readLinearPublic(a, base, loc.shape[0], loc.shape[1], w);
        var alone = try dsv41.quantizeDense(a, mlx.gpuStream(), pool, w, null, loc.shape[1], loc.shape[0], default_trunk_rate, encode.default_pin_pass, base, &stages);
        defer alone.deinit(a);
        try testing.expectEqualSlices(u16, alone.trellis, st.slice(u16, pk.get(try std.fmt.bufPrint(&wb, "{s}.trellis", .{base})).?));
        try testing.expectEqualSlices(u16, alone.suh, st.slice(u16, pk.get(try std.fmt.bufPrint(&wb, "{s}.suh", .{base})).?));
        try testing.expectEqualSlices(u16, alone.svh, st.slice(u16, pk.get(try std.fmt.bufPrint(&wb, "{s}.svh", .{base})).?));
    }
}

/// Every source tensor in its form in the pack at `out_dir`, and its stamps: synthetic banks at about the source's
/// RMS, identity-Hessian banks near the source's weights.
fn verifyPack(a: std.mem.Allocator, io: std.Io, src: *const source.SafetensorsSource, cfg: *const hy4_config.Config, out_dir: []const u8, plan: ?*const rate_plan.Plan, summary: *const Summary, codes: Codes) !void {
    const uniform = try uniformClasses(a, cfg.experts, .{ .n = 64 });
    defer freeClasses(a, uniform);
    var out = try std.Io.Dir.cwd().openDir(io, out_dir, .{ .iterate = true });
    defer out.close(io);
    var pk = try Pack.open(a, io, out);
    defer pk.deinit(a);

    // config.json: the stamps.
    const config_raw = try out.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    defer a.free(config_raw);
    const config = try std.json.parseFromSlice(std.json.Value, a, config_raw, .{});
    defer config.deinit();
    const c = config.value.object;
    try testing.expectEqual(codes == .synthetic, c.get("synthetic").?.bool);
    try testing.expectEqualStrings("bf16", c.get("engram_tables").?.string);
    const rev = c.get("pack_revision").?.object;
    try testing.expectEqualStrings(summary.revision(), rev.get("id").?.string);
    const eq = c.get("expert_quant").?.object;
    if (plan) |p| try testing.expectEqualStrings(&p.sha256, (eq.get("rate_plan_sha256") orelse return error.RatePlanUnstamped).string);
    if (plan == null) try testing.expect(eq.get("rate_plan_sha256") == null);
    try testing.expectEqualStrings("stq", eq.get("format").?.string);
    try testing.expectEqualStrings("pair1", eq.get("codebook").?.string);
    try testing.expectEqualStrings("pair1-w16-v1", eq.get("codebook_table").?.string);
    try expectK(eq.get("k").?, 4);
    const tq = c.get("trunk_quant").?.object;
    try testing.expectEqualStrings("spec1", tq.get("codebook").?.string);
    try testing.expectEqualStrings("spec1-w15", tq.get("codebook_table").?.string);
    try expectK(tq.get("k").?, 6);
    // The engine's Hy4 parser still reads the stamped config.
    _ = try hy4_config.parse(c.get("text_config").?.object, c);

    // Every shard but the unstamped clones carries the revision and the synthetic stamp.
    var fit = pk.files.iterator();
    while (fit.next()) |fe| {
        const name = fe.key_ptr.*;
        var unstamped = false;
        for (rev.get("unstamped").?.array.items) |u| unstamped = unstamped or std.mem.eql(u8, u.string, name);
        if (unstamped) {
            try testing.expect(std.mem.startsWith(u8, name, "engram-"));
            continue;
        }
        const pf = fe.value_ptr;
        try testing.expectEqualStrings(summary.revision(), pf.file.metadata.get("pack_revision").?);
        try testing.expectEqualStrings(if (codes == .synthetic) "true" else "false", pf.file.metadata.get("synthetic").?);
        try testing.expectEqualStrings(codes.hessian(), pf.file.metadata.get("hessian").?);
        try testing.expectEqualStrings("stq", pf.file.metadata.get("format").?);
        if (std.mem.startsWith(u8, name, "model-layer-")) {
            try testing.expectEqualStrings("pair1", pf.file.metadata.get("codebook").?);
            try testing.expectEqualStrings("16", pf.file.metadata.get("window").?);
        } else try testing.expectEqualStrings("spec1", pf.file.metadata.get("codebook").?);
    }

    // Every tensor the census names, in its form.
    var nb: [192]u8 = undefined;
    var forms: std.EnumArray(Form, usize) = .initFill(0);
    var it = src.tensors.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        const loc = e.value_ptr.*;
        const entry = try census.roleOf(cfg, name);
        const form = formOf(entry.role, name, loc);
        forms.getPtr(form).* += 1;
        switch (form) {
            .stored, .router => {
                const got = pk.get(name) orelse return error.TestTensorMissing;
                try testing.expectEqualStrings(loc.dtype, got.dtype);
                try testing.expectEqualSlices(usize, loc.shape, got.shape);
                const want = try src.readAllAlloc(a, name);
                defer a.free(want);
                const have = got.bytes;
                if (form == .stored) {
                    try testing.expectEqualSlices(u8, want, have);
                } else {
                    const perm = try rate_plan.permutation(a, (if (plan) |p| p.classes(entry.layer.?).? else uniform));
                    defer a.free(perm);
                    const row = want.len / perm.len;
                    for (perm, 0..) |p, i| try testing.expectEqualSlices(u8, want[p * row ..][0..row], have[i * row ..][0..row]);
                }
            },
            .engram_table => {
                const got = pk.get(name) orelse return error.TestTensorMissing;
                try testing.expect(std.mem.startsWith(u8, pk.fileOf(name).?, "engram-"));
                const want = try src.readAllAlloc(a, name);
                defer a.free(want);
                try testing.expectEqualSlices(u8, want, got.bytes);
            },
            .trunk_bank => {
                const base = name[0 .. name.len - ".weight".len];
                try testing.expect(pk.get(name) == null);
                const out_dim = loc.shape[0];
                const in_dim = loc.shape[1];
                try expectLoc(pk.get(try std.fmt.bufPrint(&nb, "{s}.trellis", .{base})), "U16", &.{ in_dim / 16, out_dim / 16, default_trunk_rate.n });
                try expectLoc(pk.get(try std.fmt.bufPrint(&nb, "{s}.suh", .{base})), "F16", &.{in_dim});
                try expectLoc(pk.get(try std.fmt.bufPrint(&nb, "{s}.svh", .{base})), "F16", &.{out_dim});
                // The bank decodes to finite weights at about the source's RMS.
                const t_raw = pk.get(try std.fmt.bufPrint(&nb, "{s}.trellis", .{base})).?.bytes;
                const s_raw = pk.get(try std.fmt.bufPrint(&nb, "{s}.suh", .{base})).?.bytes;
                const v_raw = pk.get(try std.fmt.bufPrint(&nb, "{s}.svh", .{base})).?.bytes;
                const dec = try a.alloc(u16, in_dim * out_dim);
                defer a.free(dec);
                try format.reconstructPublic(a, @alignCast(std.mem.bytesAsSlice(u16, t_raw)), @alignCast(std.mem.bytesAsSlice(u16, s_raw)), @alignCast(std.mem.bytesAsSlice(u16, v_raw)), in_dim, out_dim, default_trunk_rate, trunk_dec, dec);
                var sum: f64 = 0;
                for (dec) |bits| {
                    const v = format.f16BitsToF32(bits);
                    try testing.expect(std.math.isFinite(v));
                    sum += @as(f64, v) * v;
                }
                const want = try src.readAllAlloc(a, name);
                defer a.free(want);
                switch (codes) {
                    .synthetic => {
                        const ratio = @sqrt(sum / @as(f64, @floatFromInt(dec.len))) / rmsBf16(want);
                        try testing.expect(ratio > 0.8 and ratio < 1.25);
                    },
                    .identity_hessian, .capture => {
                        const w = try a.alloc(f32, in_dim * out_dim);
                        defer a.free(w);
                        try src.readLinearPublic(a, base, out_dim, in_dim, w);
                        try testing.expect(relError(w, dec) < 0.05);
                    },
                }
            },
            .routed => {
                try testing.expect(pk.get(name) == null);
                const classes = (if (plan) |p| p.classes(entry.layer.?).? else uniform);
                const fused = std.mem.endsWith(u8, name, "gate_up_proj");
                const latent = loc.shape[2];
                const inter = if (fused) loc.shape[1] / 2 else loc.shape[2];
                const lat = if (fused) latent else loc.shape[1];
                var experts: usize = 0;
                for (classes, 0..) |cl, gi| {
                    experts += cl.ids.len;
                    const projs: []const []const u8 = if (fused) &.{ "gate_proj", "up_proj" } else &.{"down_proj"};
                    for (projs) |proj| {
                        const in_dim = if (fused) lat else inter;
                        const out_dim = if (fused) inter else lat;
                        const n = if (fused) cl.gate_up.n else cl.down.n;
                        const pre = try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.switch_mlp.{s}.g{d}", .{ entry.layer.?, proj, gi });
                        var kb: [224]u8 = undefined;
                        try testing.expect(pk.get(try std.fmt.bufPrint(&kb, "{s}.trellis", .{pre})) == null);
                        try expectLoc(pk.get(try std.fmt.bufPrint(&kb, "{s}.trellis_pm", .{pre})), "U16", &.{ cl.ids.len, out_dim / 32, in_dim / 16, 2, n });
                        try expectLoc(pk.get(try std.fmt.bufPrint(&kb, "{s}.suh", .{pre})), "F16", &.{ cl.ids.len, in_dim });
                        try expectLoc(pk.get(try std.fmt.bufPrint(&kb, "{s}.svh", .{pre})), "F16", &.{ cl.ids.len, out_dim });
                    }
                }
                try testing.expectEqual(cfg.experts, @as(u32, @intCast(experts)));
                // Each group's first expert decodes near its source weight.
                if (codes == .identity_hessian) for (classes, 0..) |cl, gi| {
                    const projs: []const source.ProjRole = if (fused) &.{ .gate, .up } else &.{.down};
                    for (projs) |role| {
                        const in_dim = if (fused) lat else inter;
                        const out_dim = if (fused) inter else lat;
                        const n = if (fused) cl.gate_up.n else cl.down.n;
                        var kb: [224]u8 = undefined;
                        const pre = try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.switch_mlp.{s}.g{d}", .{ entry.layer.?, role.suffix(), gi });
                        const tr = try packOrderTiles(a, pk.get(try std.fmt.bufPrint(&kb, "{s}.trellis_pm", .{pre})).?, 0, in_dim / 16, out_dim / 16, n);
                        defer a.free(tr);
                        const su = st.slice(u16, pk.get(try std.fmt.bufPrint(&kb, "{s}.suh", .{pre})).?)[0..in_dim];
                        const sv = st.slice(u16, pk.get(try std.fmt.bufPrint(&kb, "{s}.svh", .{pre})).?)[0..out_dim];
                        const dec = try a.alloc(u16, in_dim * out_dim);
                        defer a.free(dec);
                        try format.reconstructPublic(a, tr, su, sv, in_dim, out_dim, .{ .n = n }, default_expert_dec, dec);
                        const w = try a.alloc(f32, in_dim * out_dim);
                        defer a.free(w);
                        try src.readExpertPublic(a, .hy_v4_flash, entry.layer.?, false, role, cl.ids[0], in_dim, out_dim, w);
                        try testing.expect(relError(w, dec) < 0.1);
                    }
                };
            },
        }
    }
    // The fixture reaches every form, the drafter's banks among the trunk's.
    for (std.enums.values(Form)) |f| try testing.expect(forms.get(f) > 0);
    try testing.expect(pk.get("model.embed_tokens.trellis") != null and pk.get("lm_head.trellis") != null);
    var dfly_banks: usize = 0;
    var pit = pk.weight_map.iterator();
    while (pit.next()) |kv| {
        const k = kv.key_ptr;
        dfly_banks += @intFromBool(std.mem.startsWith(u8, k.*, "model.dfly_layers.") and std.mem.endsWith(u8, k.*, ".trellis"));
    }
    try testing.expect(dfly_banks > 0);
}

/// ||decoded - w|| / ||w|| of a public weight and its decoded f16 bits.
fn relError(w: []const f32, decoded: []const u16) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (w, decoded) |x, bits| {
        const d = @as(f64, format.f16BitsToF32(bits)) - x;
        num += d * d;
        den += @as(f64, x) * x;
    }
    return @sqrt(num / den);
}

/// A `calib/` of `tokens` rows for the source at `src_root`, in the capture's layout: per MoE layer `route-LL.u16`
/// (`top_k` distinct picks a token) and `mlp-in-LL.bf16` (latent rows, column scales spread over two decades); per
/// layer `trunk-LL.safetensors` and `trunk-head.safetensors` with a sum for every tapped input. `identity` writes
/// every sum as the identity; otherwise each is X^T X of 256 rows with columns spread like the latent rows'. The sum
/// named `omit` is left out.
fn writeCalib(a: std.mem.Allocator, io: std.Io, calib: []const u8, src: *const source.SafetensorsSource, cfg: *const hy4_config.Config, tokens: usize, identity: bool, omit: []const u8) !void {
    var prng = std.Random.DefaultPrng.init(0xca11b);
    const r = prng.random();
    try std.Io.Dir.cwd().createDirPath(io, calib);
    var dir = try std.Io.Dir.cwd().openDir(io, calib, .{});
    defer dir.close(io);
    var nb: [96]u8 = undefined;
    const lat = cfg.moe_inter;
    for (cfg.dense_layers..cfg.layers) |l| {
        const picks = try a.alloc(u16, tokens * cfg.top_k);
        defer a.free(picks);
        for (0..tokens) |t| {
            var k: usize = 0;
            while (k < cfg.top_k) {
                const e: u16 = r.uintLessThan(u16, @intCast(cfg.experts));
                if (std.mem.indexOfScalar(u16, picks[t * cfg.top_k ..][0..k], e) != null) continue;
                picks[t * cfg.top_k + k] = e;
                k += 1;
            }
        }
        try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&nb, "route-{d:0>2}.u16", .{l}), .data = std.mem.sliceAsBytes(picks) });
        const rows = try a.alloc(u16, tokens * lat);
        defer a.free(rows);
        for (rows, 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(r.floatNorm(f32) * columnScale(i % lat))) >> 16);
        try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&nb, "mlp-in-{d:0>2}.bf16", .{l}), .data = std.mem.sliceAsBytes(rows) });
    }
    for (0..cfg.layers + 1) |l| {
        var b = pack.ShardBuilder.init(a);
        defer b.deinit();
        var seen = std.StringHashMap(void).init(a);
        defer {
            var kit = seen.keyIterator();
            while (kit.next()) |k| a.free(k.*);
            seen.deinit();
        }
        var it = src.tensors.iterator();
        while (it.next()) |e| {
            const name = e.key_ptr.*;
            if (!std.mem.endsWith(u8, name, ".weight")) continue;
            const kind = tapKindOf(name[0 .. name.len - ".weight".len]) orelse continue;
            if ((kind.layer orelse cfg.layers) != l) continue;
            const key = if (kind.layer) |kl| try std.fmt.bufPrint(&nb, "layers.{d}.{s}", .{ kl, kind.name }) else kind.name;
            if (seen.contains(key) or std.mem.eql(u8, key, omit)) continue;
            const owned = try a.dupe(u8, key);
            seen.put(owned, {}) catch |err| {
                a.free(owned);
                return err;
            };
            const w = e.value_ptr.shape[1];
            const gram = try a.alloc(f32, w * w);
            defer a.free(gram);
            @memset(gram, 0);
            if (identity) {
                for (0..w) |i| gram[i * w + i] = 1;
            } else {
                const x = try a.alloc(f32, w);
                defer a.free(x);
                for (0..256) |_| {
                    for (x, 0..) |*v, i| v.* = r.floatNorm(f32) * columnScale(i);
                    for (0..w) |i| for (0..w) |j| {
                        gram[i * w + j] += x[i] * x[j];
                    };
                }
            }
            try b.add(key, "F32", &.{ w, w }, std.mem.sliceAsBytes(gram));
            var rb: [128]u8 = undefined;
            const n: i64 = if (identity) 1000 else 256;
            try b.add(try std.fmt.bufPrint(&rb, "{s}.rows", .{key}), "I64", &.{1}, std.mem.asBytes(&n));
        }
        const file = if (l == cfg.layers) "trunk-head.safetensors" else try std.fmt.bufPrint(&nb, "trunk-{d:0>2}.safetensors", .{l});
        try b.flush(io, dir, file, &.{});
    }
}

fn columnScale(i: usize) f32 {
    return std.math.pow(f32, 10, @as(f32, @floatFromInt(i % 17)) / 8 - 1);
}

/// tr(Eᵀ H E) of the error E = decoded - w, both public `[in, out]`.
fn weightedError(w: []const f32, decoded: []const u16, h: []const f32, in: usize, out: usize) f64 {
    var sum: f64 = 0;
    for (0..out) |c| for (0..in) |i| {
        const ei = @as(f64, format.f16BitsToF32(decoded[i * out + c])) - w[i * out + c];
        for (0..in) |j| {
            const ej = @as(f64, format.f16BitsToF32(decoded[j * out + c])) - w[j * out + c];
            sum += ei * h[i * in + j] * ej;
        }
    };
    return sum;
}

fn capturePaths(a: std.mem.Allocator, root: []const u8) ![3][]u8 {
    var out: [3][]u8 = undefined;
    var made: usize = 0;
    errdefer for (out[0..made]) |p| a.free(p);
    for (&out, [_][]const u8{ "calib", "pack", "identity" }) |*p, leaf| {
        p.* = try std.fmt.allocPrint(a, "{s}/{s}", .{ root, leaf });
        made += 1;
    }
    return out;
}

// With every sum the identity and too few routed rows for any expert, the capture path rounds exactly as the identity
// path: the same banks, byte for byte, and every expert projection counted against the identity.
test "hy4 convert: a capture of identity sums and starved experts gives the identity pack's banks (HY4_BF16_TINY_STQ)" {
    const src_env = std.c.getenv("HY4_BF16_TINY_STQ") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src_root = std.mem.span(src_env);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const paths = try capturePaths(a, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
    defer for (paths) |p| a.free(p);
    var src_dir = try std.Io.Dir.cwd().openDir(io, src_root, .{ .iterate = true });
    defer src_dir.close(io);
    const conf = try readConfig(a, io, src_dir);
    defer a.free(conf.raw);
    var src = try source.SafetensorsSource.open(a, io, src_dir);
    defer src.deinit();
    try writeCalib(a, io, paths[0], &src, &conf.cfg, 8, true, "");

    const got = try run(a, io, .{ .model_dir = src_root, .out_dir = paths[1], .rate = .{ .n = 64 }, .codes = .capture, .capture_dir = paths[0], .version = "test" }, null);
    _ = try run(a, io, .{ .model_dir = src_root, .out_dir = paths[2], .rate = .{ .n = 64 }, .codes = .identity_hessian, .version = "test" }, null);
    try testing.expect(std.mem.startsWith(u8, got.revision(), "capture-"));
    const moe_layers = conf.cfg.layers - conf.cfg.dense_layers;
    // Each expert's gate and up share one Hessian, its down has its own: two counts an expert.
    try testing.expectEqual(moe_layers * conf.cfg.experts * 2, got.short_banks);
    try testing.expectEqual(try embedAndDflyBanks(&src, &conf.cfg), got.untapped_banks);

    var pack_dir = try std.Io.Dir.cwd().openDir(io, paths[1], .{});
    defer pack_dir.close(io);
    var id_dir = try std.Io.Dir.cwd().openDir(io, paths[2], .{});
    defer id_dir.close(io);
    var pk = try Pack.open(a, io, pack_dir);
    defer pk.deinit(a);
    var ik = try Pack.open(a, io, id_dir);
    defer ik.deinit(a);
    var banks: usize = 0;
    var it = ik.weight_map.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        const want = ik.get(name).?;
        const have = pk.get(name) orelse return error.TestTensorMissing;
        try testing.expectEqualSlices(u8, want.bytes, have.bytes);
        banks += @intFromBool(std.mem.endsWith(u8, name, ".trellis"));
    }
    try testing.expect(banks > 0);
    try testing.expectEqualStrings("capture", pk.files.get(pk.fileOf("lm_head.trellis").?).?.file.metadata.get("hessian").?);
}

// The capture writes `mlp-in-LL.f32` at the f32 products its teacher runs: the rows read as f32, so rows the BF16 twin
// holds give the same pack byte for byte; a capture holding both forms is refused before anything is written.
test "hy4 convert: a capture at f32 products (mlp-in-LL.f32) gives its BF16 twin's pack, and both forms are refused (HY4_BF16_TINY_STQ)" {
    const src_env = std.c.getenv("HY4_BF16_TINY_STQ") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src_root = std.mem.span(src_env);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const paths = try capturePaths(a, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
    defer for (paths) |p| a.free(p);
    var src_dir = try std.Io.Dir.cwd().openDir(io, src_root, .{ .iterate = true });
    defer src_dir.close(io);
    const conf = try readConfig(a, io, src_dir);
    defer a.free(conf.raw);
    var src = try source.SafetensorsSource.open(a, io, src_dir);
    defer src.deinit();
    try writeCalib(a, io, paths[0], &src, &conf.cfg, 512, false, "");
    const opts = Options{ .model_dir = src_root, .out_dir = paths[1], .rate = .{ .n = 64 }, .codes = .capture, .capture_dir = paths[0], .min_rows = 8, .version = "test" };
    _ = try run(a, io, opts, null);

    var calib = try std.Io.Dir.cwd().openDir(io, paths[0], .{});
    defer calib.close(io);
    var nb: [2][32]u8 = undefined;
    for (conf.cfg.dense_layers..conf.cfg.layers) |l| {
        const bf = try std.fmt.bufPrint(&nb[0], "mlp-in-{d:0>2}.bf16", .{l});
        const raw = try calib.readFileAlloc(io, bf, a, .limited(1 << 30));
        defer a.free(raw);
        const wide = try a.alloc(u32, raw.len / 2);
        defer a.free(wide);
        for (wide, std.mem.bytesAsSlice(u16, raw)) |*w, h| w.* = @as(u32, h) << 16;
        try calib.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&nb[1], "mlp-in-{d:0>2}.f32", .{l}), .data = std.mem.sliceAsBytes(wide) });
    }
    var both = opts;
    both.out_dir = paths[2];
    try testing.expectError(error.CaptureRowsAmbiguous, run(a, io, both, null));
    var out_dir = try std.Io.Dir.cwd().openDir(io, paths[2], .{ .iterate = true });
    var it_out = out_dir.iterate();
    try testing.expect(try it_out.next(io) == null);
    out_dir.close(io);
    for (conf.cfg.dense_layers..conf.cfg.layers) |l| try calib.deleteFile(io, try std.fmt.bufPrint(&nb[0], "mlp-in-{d:0>2}.bf16", .{l}));
    _ = try run(a, io, both, null);

    var bf_dir = try std.Io.Dir.cwd().openDir(io, paths[1], .{});
    defer bf_dir.close(io);
    var f32_dir = try std.Io.Dir.cwd().openDir(io, paths[2], .{});
    defer f32_dir.close(io);
    var bk = try Pack.open(a, io, bf_dir);
    defer bk.deinit(a);
    var fk = try Pack.open(a, io, f32_dir);
    defer fk.deinit(a);
    var experts: usize = 0;
    var it = bk.weight_map.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        try testing.expectEqualSlices(u8, bk.get(name).?.bytes, (fk.get(name) orelse return error.TestTensorMissing).bytes);
        experts += @intFromBool(std.mem.indexOf(u8, name, ".mlp.switch_mlp.") != null and std.mem.endsWith(u8, name, ".trellis_pm"));
    }
    try testing.expect(experts > 0);
}

// An interrupted pack resumes: the shards stamped with this run's revision stay, the rest are written as a whole run
// writes them, so the resumed pack is the uninterrupted one byte for byte. A complete pack, a shard of another revision
// and a non-empty directory without --resume are refused by name.
test "hy4 convert: --resume keeps the shards of this revision and writes the rest to the uninterrupted pack's bytes (HY4_BF16_TINY_STQ)" {
    const src_env = std.c.getenv("HY4_BF16_TINY_STQ") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src_root = std.mem.span(src_env);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const paths = try capturePaths(a, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
    defer for (paths) |p| a.free(p);
    const whole = Options{ .model_dir = src_root, .out_dir = paths[1], .rate = .{ .n = 64 }, .codes = .identity_hessian, .version = "test" };
    _ = try run(a, io, whole, null);
    var cut = whole;
    cut.out_dir = paths[2];
    _ = try run(a, io, cut, null);

    var a_dir = try std.Io.Dir.cwd().openDir(io, paths[1], .{ .iterate = true });
    defer a_dir.close(io);
    var b_dir = try std.Io.Dir.cwd().openDir(io, paths[2], .{ .iterate = true });
    defer b_dir.close(io);
    try testing.expectError(error.OutputNotEmpty, run(a, io, cut, null));
    cut.resuming = true;
    try testing.expectError(error.ResumePackComplete, run(a, io, cut, null));
    // Cut where a run can stop: before the config and index, an expert shard and a trunk shard not yet written.
    var eb: [64]u8 = undefined;
    var tb: [64]u8 = undefined;
    const dropped = [_][]const u8{ "config.json", "model.safetensors.index.json", try layerExpertsFile(&eb, 2), try layerTrunkFile(&tb, 4), globals_file };
    for (dropped) |f| try b_dir.deleteFile(io, f);
    var other = cut;
    other.rate = .{ .n = 48 };
    try testing.expectError(error.ResumeRevisionMismatch, run(a, io, other, null));
    const resumed = try run(a, io, cut, null);
    var files: usize = 0;
    var it = a_dir.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file) continue;
        const want = try a_dir.readFileAlloc(io, e.name, a, .limited(1 << 30));
        defer a.free(want);
        const got = b_dir.readFileAlloc(io, e.name, a, .limited(1 << 30)) catch return error.TestFileMissing;
        defer a.free(got);
        try testing.expectEqualSlices(u8, want, got);
        files += 1;
    }
    try testing.expect(files > dropped.len);
    var cfg_dir = try std.Io.Dir.cwd().openDir(io, src_root, .{ .iterate = true });
    defer cfg_dir.close(io);
    const conf = try readConfig(a, io, cfg_dir);
    defer a.free(conf.raw);
    // Every MoE layer's experts but layer 2's, and every trunk shard but layer 4's and the globals.
    try testing.expectEqual(conf.cfg.layers - conf.cfg.dense_layers - 1, resumed.experts_resumed);
    // Every layer's trunk shard but layer 4's, and the drafter's.
    try testing.expectEqual(conf.cfg.layers - 1 + 1, resumed.units_resumed);
}

// The DFly2 drafter's banks take their own rate: every `model.dfly_layers.*` bank at `drafter_rate` (K4 here), stamped
// so in its shard and the config's `drafter_quant`, decoding near the source; every other bank keeps the trunk's K6.
test "hy4 convert: the drafter's banks round at their own rate, the trunk's at its own (HY4_BF16_TINY_STQ)" {
    const src_env = std.c.getenv("HY4_BF16_TINY_STQ") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src_root = std.mem.span(src_env);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const out_dir = try std.fmt.allocPrint(a, "{s}/pack", .{path_buf[0..try tmp.dir.realPath(io, &path_buf)]});
    defer a.free(out_dir);
    var src_dir = try std.Io.Dir.cwd().openDir(io, src_root, .{ .iterate = true });
    defer src_dir.close(io);
    var src = try source.SafetensorsSource.open(a, io, src_dir);
    defer src.deinit();
    const k4 = format.Rate{ .n = 64 };
    _ = try run(a, io, .{ .model_dir = src_root, .out_dir = out_dir, .rate = k4, .drafter_rate = k4, .codes = .identity_hessian, .version = "test" }, null);
    var out = try std.Io.Dir.cwd().openDir(io, out_dir, .{});
    defer out.close(io);
    var pk = try Pack.open(a, io, out);
    defer pk.deinit(a);
    var drafter: usize = 0;
    var trunk: usize = 0;
    var it = pk.weight_map.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        if (!std.mem.endsWith(u8, name, ".trellis") or std.mem.indexOf(u8, name, ".mlp.switch_mlp.") != null) continue;
        const t = pk.get(name).?;
        const is_dfly = std.mem.startsWith(u8, name, "model.dfly_layers.");
        try testing.expectEqual(@as(usize, if (is_dfly) k4.n else default_trunk_rate.n), t.shape[2]);
        drafter += @intFromBool(is_dfly);
        trunk += @intFromBool(!is_dfly);
        if (!is_dfly) continue;
        const base = name[0 .. name.len - ".trellis".len];
        const in_dim = t.shape[0] * 16;
        const out_dim = t.shape[1] * 16;
        const w = try a.alloc(f32, in_dim * out_dim);
        defer a.free(w);
        try src.readLinearPublic(a, base, out_dim, in_dim, w);
        var nb: [2][160]u8 = undefined;
        const dec = try a.alloc(u16, in_dim * out_dim);
        defer a.free(dec);
        try format.reconstructPublic(a, @alignCast(std.mem.bytesAsSlice(u16, t.bytes)), @alignCast(std.mem.bytesAsSlice(u16, pk.get(try std.fmt.bufPrint(&nb[0], "{s}.suh", .{base})).?.bytes)), @alignCast(std.mem.bytesAsSlice(u16, pk.get(try std.fmt.bufPrint(&nb[1], "{s}.svh", .{base})).?.bytes)), in_dim, out_dim, k4, trunk_dec, dec);
        try testing.expect(relError(w, dec) < 0.10);
    }
    try testing.expect(drafter > 0 and trunk > 0);
    try testing.expectEqualStrings("4", pk.files.get(dfly_file).?.file.metadata.get("k").?);
    try testing.expectEqualStrings("6", pk.files.get(globals_file).?.file.metadata.get("k").?);
    const raw = try out.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    defer a.free(raw);
    const cfg = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer cfg.deinit();
    try expectK(cfg.value.object.get("drafter_quant").?.object.get("k").?, 4);
    try expectK(cfg.value.object.get("trunk_quant").?.object.get("k").?, 6);
}

/// The banks a capture has no sum for: the embedding's and the drafter's.
fn embedAndDflyBanks(src: *const source.SafetensorsSource, cfg: *const hy4_config.Config) !usize {
    var n: usize = 0;
    var it = src.tensors.iterator();
    while (it.next()) |e| {
        const entry = try census.roleOf(cfg, e.key_ptr.*);
        if (entry.role != .embedding and entry.role != .dfly2) continue;
        n += @intFromBool(formOf(entry.role, e.key_ptr.*, e.value_ptr.*) == .trunk_bank);
    }
    return n;
}

// With real sums and enough routed rows, each bank is rounded for its own input: under the captured Hessian, a trunk
// bank's and an expert's error are below the identity pack's.
test "hy4 convert: a capture pack's banks beat the identity pack's under the captured Hessians (HY4_BF16_TINY_STQ)" {
    const src_env = std.c.getenv("HY4_BF16_TINY_STQ") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src_root = std.mem.span(src_env);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const paths = try capturePaths(a, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
    defer for (paths) |p| a.free(p);
    var src_dir = try std.Io.Dir.cwd().openDir(io, src_root, .{ .iterate = true });
    defer src_dir.close(io);
    const conf = try readConfig(a, io, src_dir);
    defer a.free(conf.raw);
    const cfg = conf.cfg;
    var src = try source.SafetensorsSource.open(a, io, src_dir);
    defer src.deinit();
    const tokens = 512;
    try writeCalib(a, io, paths[0], &src, &cfg, tokens, false, "");

    const got = try run(a, io, .{ .model_dir = src_root, .out_dir = paths[1], .rate = .{ .n = 64 }, .codes = .capture, .capture_dir = paths[0], .min_rows = 16, .version = "test" }, null);
    _ = try run(a, io, .{ .model_dir = src_root, .out_dir = paths[2], .rate = .{ .n = 64 }, .codes = .identity_hessian, .version = "test" }, null);
    try testing.expectEqual(try embedAndDflyBanks(&src, &cfg), got.untapped_banks);
    try testing.expectEqual(@as(usize, 0), got.short_banks);
    var pack_dir = try std.Io.Dir.cwd().openDir(io, paths[1], .{});
    defer pack_dir.close(io);
    var id_dir = try std.Io.Dir.cwd().openDir(io, paths[2], .{});
    defer id_dir.close(io);
    var pk = try Pack.open(a, io, pack_dir);
    defer pk.deinit(a);
    var ik = try Pack.open(a, io, id_dir);
    defer ik.deinit(a);
    var nb: [160]u8 = undefined;

    // The first MoE layer's q_a against its attn_in sum.
    const layer = cfg.dense_layers;
    var sums_path: [std.fs.max_path_bytes]u8 = undefined;
    var sums = try dsv41.Sums.open(a, io, try std.fmt.bufPrint(&sums_path, "{s}/trunk-{d:0>2}.safetensors", .{ paths[0], layer }));
    defer sums.deinit(a);
    const base = try std.fmt.bufPrint(&nb, "model.layers.{d}.self_attn.q_a_proj", .{layer});
    try testing.expect(try bankWins(a, &src, &pk, &ik, base, sums.get(try std.fmt.bufPrint(&sums_path, "layers.{d}.attn_in", .{layer})).?.gram));

    // Expert 0's gate against the latent rows routed to it.
    const lat = cfg.moe_inter;
    var rp: [std.fs.max_path_bytes]u8 = undefined;
    const ids = try dsv41.readRoutes(a, io, try std.fmt.bufPrint(&rp, "{s}/route-{d:0>2}.u16", .{ paths[0], layer }), cfg.experts);
    defer a.free(ids);
    const rows_raw = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&rp, "{s}/mlp-in-{d:0>2}.bf16", .{ paths[0], layer }), a, .limited(1 << 30));
    defer a.free(rows_raw);
    const h = try a.alloc(f32, lat * lat);
    defer a.free(h);
    @memset(h, 0);
    for (ids, 0..) |e, p| if (e == 0) {
        const t = p / cfg.top_k;
        for (0..lat) |i| for (0..lat) |j| {
            h[i * lat + j] += bf16At(rows_raw, t * lat + i) * bf16At(rows_raw, t * lat + j);
        };
    };
    const w = try a.alloc(f32, lat * lat);
    defer a.free(w);
    try src.readExpertPublic(a, .hy_v4_flash, layer, false, .gate, 0, lat, lat, w);
    const pre = try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.switch_mlp.gate_proj.g0", .{layer});
    var errs: [2]f64 = undefined;
    for ([2]*const Pack{ &pk, &ik }, &errs) |p, *e| {
        const dec = try decodeBank(a, p, pre, 0, lat, lat, .{ .n = 64 }, default_expert_dec);
        defer a.free(dec);
        e.* = weightedError(w, dec, h, lat, lat);
    }
    try testing.expect(errs[0] < errs[1]);
}

fn bf16At(raw: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, raw[i * 2 ..][0..2], .little)) << 16);
}

/// Matrix `index` of the leading axis of a `trellis_pm` tensor, back in the pack order `[IT][OT][n]` the decoder reads.
fn packOrderTiles(a: std.mem.Allocator, pm: st.TensorView, index: usize, it: usize, ot: usize, n: usize) ![]u16 {
    const per = it * ot * n;
    const out = try a.alloc(u16, per);
    api.group_layout.tileMajorInverse(u16, out, st.slice(u16, pm)[index * per ..][0..per], it, ot, n);
    return out;
}

/// Bank `pre` (`.trellis`/`.trellis_pm`, `.suh`/`.svh`, matrix `index` of its leading axis) decoded to public `[in, out]` f16 bits.
fn decodeBank(a: std.mem.Allocator, p: *const Pack, pre: []const u8, index: usize, in: usize, out: usize, rate: format.Rate, dec: format.Decode) ![]u16 {
    var kb: [192]u8 = undefined;
    const tiles = (in / 16) * (out / 16) * rate.n;
    var pm_buf: [200]u8 = undefined;
    const pm = p.get(try std.fmt.bufPrint(&pm_buf, "{s}.trellis_pm", .{pre}));
    const owned: ?[]u16 = if (pm) |v| try packOrderTiles(a, v, index, in / 16, out / 16, rate.n) else null;
    defer if (owned) |o| a.free(o);
    const tr = owned orelse st.slice(u16, p.get(try std.fmt.bufPrint(&kb, "{s}.trellis", .{pre})) orelse return error.TestTensorMissing)[index * tiles ..][0..tiles];
    const su = st.slice(u16, p.get(try std.fmt.bufPrint(&kb, "{s}.suh", .{pre})) orelse return error.TestTensorMissing)[index * in ..][0..in];
    const sv = st.slice(u16, p.get(try std.fmt.bufPrint(&kb, "{s}.svh", .{pre})) orelse return error.TestTensorMissing)[index * out ..][0..out];
    const got = try a.alloc(u16, in * out);
    errdefer a.free(got);
    try format.reconstructPublic(a, tr, su, sv, in, out, rate, dec, got);
    return got;
}

/// Whether trunk bank `base` of `pk` has a lower error under `h` than the same bank of `ik`.
fn bankWins(a: std.mem.Allocator, src: *const source.SafetensorsSource, pk: *const Pack, ik: *const Pack, base: []const u8, h: []const f32) !bool {
    var wb: [192]u8 = undefined;
    const loc = src.tensors.get(try std.fmt.bufPrint(&wb, "{s}.weight", .{base})) orelse return error.TestTensorMissing;
    const out = loc.shape[0];
    const in = loc.shape[1];
    const w = try a.alloc(f32, in * out);
    defer a.free(w);
    try src.readLinearPublic(a, base, out, in, w);
    var errs: [2]f64 = undefined;
    for ([2]*const Pack{ pk, ik }, &errs) |p, *e| {
        const dec = try decodeBank(a, p, base, 0, in, out, default_trunk_rate, trunk_dec);
        defer a.free(dec);
        e.* = weightedError(w, dec, h, in, out);
    }
    return errs[0] < errs[1];
}

// The capture writes a sum for every linear's input; the reader maps each to it, the banks and the linears kept as stored
// alike (kv_a off the grid, the router, the indexer's query, token key and head weights, Engram's projections).
test "hy4 convert: every BF16 linear of a layer maps to the capture sum of its input (HY4_BF16_TINY_STQ)" {
    const src_env = std.c.getenv("HY4_BF16_TINY_STQ") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var src_dir = try std.Io.Dir.cwd().openDir(io, std.mem.span(src_env), .{ .iterate = true });
    defer src_dir.close(io);
    const conf = try readConfig(a, io, src_dir);
    defer a.free(conf.raw);
    var src = try source.SafetensorsSource.open(a, io, src_dir);
    defer src.deinit();
    var linears: usize = 0;
    var it = src.tensors.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        const loc = e.value_ptr.*;
        const entry = try census.roleOf(&conf.cfg, name);
        if (entry.layer == null or entry.role == .routed_experts or entry.role == .engram_table) continue;
        if (!std.mem.endsWith(u8, name, ".weight") or loc.shape.len != 2 or !std.mem.eql(u8, loc.dtype, "BF16")) continue;
        linears += 1;
        const kind = tapKindOf(name[0 .. name.len - ".weight".len]) orelse {
            std.log.err("no capture sum maps to {s}", .{name});
            return error.TestUnmappedLinear;
        };
        try testing.expectEqual(entry.layer.?, kind.layer.?);
    }
    try testing.expect(linears > 0);
}

// A capture is checked whole before anything is written: a sum the source's linears read and every MoE layer's rows.
test "hy4 convert: a capture missing a tapped sum or a layer's rows is refused before anything is written (HY4_BF16_TINY_STQ)" {
    const src_env = std.c.getenv("HY4_BF16_TINY_STQ") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src_root = std.mem.span(src_env);
    var src_dir = try std.Io.Dir.cwd().openDir(io, src_root, .{ .iterate = true });
    defer src_dir.close(io);
    const conf = try readConfig(a, io, src_dir);
    defer a.free(conf.raw);
    const cfg = conf.cfg;
    var src = try source.SafetensorsSource.open(a, io, src_dir);
    defer src.deinit();
    const engram_layer = for (0..cfg.layers) |l| {
        var nb: [96]u8 = undefined;
        if (src.tensors.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.engram_layer.value_proj.weight", .{l})) != null) break l;
    } else return error.TestNoEngram;
    var kb: [64]u8 = undefined;
    var last: [64]u8 = undefined;
    const cases = [_]struct { omit: []const u8, drop: []const u8, err: anyerror }{
        .{ .omit = try std.fmt.bufPrint(&kb, "layers.{d}.engram_in", .{engram_layer}), .drop = "", .err = error.CaptureSumMissing },
        .{ .omit = "", .drop = try std.fmt.bufPrint(&last, "route-{d:0>2}.u16", .{cfg.layers - 1}), .err = error.CaptureFileMissing },
    };
    for (cases) |cs| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const paths = try capturePaths(a, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
        defer for (paths) |p| a.free(p);
        try writeCalib(a, io, paths[0], &src, &cfg, 8, true, cs.omit);
        if (cs.drop.len > 0) {
            var cd = try std.Io.Dir.cwd().openDir(io, paths[0], .{});
            defer cd.close(io);
            try cd.deleteFile(io, cs.drop);
        }
        try testing.expectError(cs.err, run(a, io, .{ .model_dir = src_root, .out_dir = paths[1], .rate = .{ .n = 64 }, .codes = .capture, .capture_dir = paths[0], .version = "test" }, null));
        var out = try std.Io.Dir.cwd().openDir(io, paths[1], .{ .iterate = true });
        defer out.close(io);
        var oit = out.iterate();
        try testing.expect(try oit.next(io) == null);
    }
}

// ── pair-major ──────────────────────────────────────────────────────────────────────────────────────

/// The file at `dir/name`, whole.
fn readPackFile(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) ![]u8 {
    return dir.readFileAlloc(io, name, a, .limited(1 << 30));
}

/// `pm` holds the pack order `po` holds, in the layout docs/pack-format.md#tile-order gives, and nothing else differs: every routed
/// trellis is `trellis_pm`, each expert's tiles `tileMajor` of the pack order's, on a page of the file after a header padded to one,
/// the gaps are zero `__pad__` tensors the index lists, the routed shards stamp `tile_order`, the config stamps it in `expert_quant`,
/// and every other file (the trunk, the drafter, the globals, the clones, the sidecars) is the same bytes.
fn expectPairMajorOf(a: std.mem.Allocator, io: std.Io, po: std.Io.Dir, pm: std.Io.Dir, routed_shards: usize) !void {
    var routed: usize = 0;
    var same: usize = 0;
    var it = po.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or std.mem.startsWith(u8, entry.name, "convert-progress")) continue;
        if (std.mem.eql(u8, entry.name, "config.json") or std.mem.eql(u8, entry.name, "model.safetensors.index.json")) continue;
        const want = try readPackFile(a, io, po, entry.name);
        defer a.free(want);
        const got = try readPackFile(a, io, pm, entry.name);
        defer a.free(got);
        const is_routed = std.mem.startsWith(u8, entry.name, "model-layer-") and std.mem.endsWith(u8, entry.name, ".safetensors");
        if (!is_routed) {
            try testing.expectEqualSlices(u8, want, got);
            same += 1;
            continue;
        }
        routed += 1;
        const stem = entry.name[0 .. entry.name.len - ".safetensors".len];
        const raw_po = try a.alignedAlloc(u8, .of(u32), want.len);
        defer a.free(raw_po);
        @memcpy(raw_po, want);
        const raw_pm = try a.alignedAlloc(u8, .of(u32), got.len);
        defer a.free(raw_pm);
        @memcpy(raw_pm, got);
        var fo = try st.parse(a, raw_po);
        defer fo.deinit(a);
        var fm = try st.parse(a, raw_pm);
        defer fm.deinit(a);
        try testing.expectEqualStrings("pair_major", fm.metadata.get("tile_order") orelse return error.TestNoTileOrder);
        try testing.expect(fo.metadata.get("tile_order") == null);
        try testing.expectEqual(fo.metadata.count() + 1, fm.metadata.count());
        var mit = fo.metadata.iterator();
        while (mit.next()) |m| try testing.expectEqualStrings(m.value_ptr.*, fm.metadata.get(m.key_ptr.*) orelse return error.TestMetadataLost);
        var first: usize = raw_pm.len;
        var pads: usize = 0;
        var tit = fm.tensors.iterator();
        while (tit.next()) |t| {
            first = @min(first, @intFromPtr(t.value_ptr.bytes.ptr) - @intFromPtr(raw_pm.ptr));
            if (!std.mem.startsWith(u8, t.key_ptr.*, shard.pad_prefix)) continue;
            pads += 1;
            try testing.expect(std.mem.startsWith(u8, t.key_ptr.*[shard.pad_prefix.len..], stem));
            try testing.expectEqualStrings("U8", t.value_ptr.dtype);
            try testing.expect(t.value_ptr.shape.len == 1 and t.value_ptr.bytes.len > 0 and t.value_ptr.bytes.len < shard.page);
            for (t.value_ptr.bytes) |b| try testing.expectEqual(@as(u8, 0), b);
        }
        try testing.expectEqual(@as(usize, 0), first % shard.page);
        try testing.expectEqual(fo.tensors.count() + pads, fm.tensors.count());
        var pm_banks: usize = 0;
        var oit = fo.tensors.iterator();
        while (oit.next()) |t| {
            const name = t.key_ptr.*;
            const ref = t.value_ptr.*;
            if (!std.mem.endsWith(u8, name, ".trellis")) {
                const have = fm.tensors.get(name) orelse return error.TestTensorMissing;
                try testing.expectEqualStrings(ref.dtype, have.dtype);
                try testing.expectEqualSlices(usize, ref.shape, have.shape);
                try testing.expectEqualSlices(u8, ref.bytes, have.bytes);
                continue;
            }
            try testing.expect(fm.tensors.get(name) == null);
            var nb: [192]u8 = undefined;
            const have = fm.tensors.get(try std.fmt.bufPrint(&nb, "{s}_pm", .{name})) orelse return error.TestTensorMissing;
            const experts, const tiles_in, const tiles_out, const n = .{ ref.shape[0], ref.shape[1], ref.shape[2], ref.shape[3] };
            try testing.expectEqualStrings("U16", have.dtype);
            try testing.expectEqualSlices(usize, &.{ experts, tiles_out / 2, tiles_in, 2, n }, have.shape);
            try testing.expectEqual(@as(usize, 0), (@intFromPtr(have.bytes.ptr) - @intFromPtr(raw_pm.ptr)) % shard.page);
            const per = tiles_in * tiles_out * n;
            const want_pm = try a.alloc(u16, experts * per);
            defer a.free(want_pm);
            for (0..experts) |e| api.group_layout.tileMajor(u16, want_pm[e * per ..][0..per], st.slice(u16, ref)[e * per ..][0..per], tiles_in, tiles_out, n);
            try testing.expectEqualSlices(u16, want_pm, st.slice(u16, have));
            pm_banks += 1;
        }
        try testing.expect(pm_banks > 0);
    }
    try testing.expectEqual(routed_shards, routed);
    try testing.expect(same > 0);
    try expectPairMajorConfigAndIndex(a, io, po, pm);
}

/// The config is the pack order's with `expert_quant.tile_order`, and the index names every tensor of the pack order's (a routed
/// trellis as `trellis_pm`) in the shard it was in, plus every shard's pads, with the same `total_size`.
fn expectPairMajorConfigAndIndex(a: std.mem.Allocator, io: std.Io, po: std.Io.Dir, pm: std.Io.Dir) !void {
    const cfg_po = try readPackFile(a, io, po, "config.json");
    defer a.free(cfg_po);
    const cfg_pm = try readPackFile(a, io, pm, "config.json");
    defer a.free(cfg_pm);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    var parsed = try std.json.parseFromSliceLeaky(std.json.Value, ar, cfg_pm, .{});
    const eq = parsed.object.getPtr("expert_quant").?;
    try testing.expectEqualStrings("pair_major", eq.object.get("tile_order").?.string);
    try testing.expect(parsed.object.get("trunk_quant").?.object.get("tile_order") == null);
    _ = eq.object.orderedRemove("tile_order");
    const without = try std.json.Stringify.valueAlloc(a, parsed, .{ .whitespace = .indent_2 });
    defer a.free(without);
    try testing.expectEqualStrings(cfg_po, without);

    const idx_po = try readPackFile(a, io, po, "model.safetensors.index.json");
    defer a.free(idx_po);
    const idx_pm = try readPackFile(a, io, pm, "model.safetensors.index.json");
    defer a.free(idx_pm);
    const ip = try std.json.parseFromSliceLeaky(std.json.Value, ar, idx_po, .{});
    const im = try std.json.parseFromSliceLeaky(std.json.Value, ar, idx_pm, .{});
    try testing.expectEqual(ip.object.get("metadata").?.object.get("total_size").?.integer, im.object.get("metadata").?.object.get("total_size").?.integer);
    const wp = ip.object.get("weight_map").?.object;
    const wm = im.object.get("weight_map").?.object;
    var pads: usize = 0;
    var mit = wm.iterator();
    while (mit.next()) |e| pads += @intFromBool(std.mem.startsWith(u8, e.key_ptr.*, shard.pad_prefix));
    try testing.expect(pads > 0);
    try testing.expectEqual(wp.count() + pads, wm.count());
    var pit = wp.iterator();
    while (pit.next()) |e| {
        var nb: [256]u8 = undefined;
        const name = if (isRoutedTrellis(e.key_ptr.*))
            try std.fmt.bufPrint(&nb, "{s}_pm", .{e.key_ptr.*})
        else
            e.key_ptr.*;
        try testing.expectEqualStrings(e.value_ptr.string, (wm.get(name) orelse return error.TestIndexMissing).string);
    }
}

fn isRoutedTrellis(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".mlp.switch_mlp.") != null and std.mem.endsWith(u8, name, ".trellis");
}

fn convertBothLayouts(codes: Codes) !void {
    const src_env = std.c.getenv("HY4_BF16_TINY_STQ") orelse return error.SkipZigTest;
    if (codes != .synthetic and mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src_root = std.mem.span(src_env);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const paths = try capturePaths(a, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
    defer for (paths) |p| a.free(p);
    const opts = Options{ .model_dir = src_root, .out_dir = paths[1], .rate = .{ .n = 64 }, .codes = codes, .version = "test" };
    const pm_summary = try run(a, io, opts, null);
    var reference = opts;
    reference.out_dir = paths[2];
    const po_summary = try runLayout(a, io, reference, null, false);
    try testing.expectEqualStrings(po_summary.revision(), pm_summary.revision());
    var pm_dir = try std.Io.Dir.cwd().openDir(io, paths[1], .{ .iterate = true });
    defer pm_dir.close(io);
    var po_dir = try std.Io.Dir.cwd().openDir(io, paths[2], .{ .iterate = true });
    defer po_dir.close(io);
    var src_dir = try std.Io.Dir.cwd().openDir(io, src_root, .{});
    defer src_dir.close(io);
    const conf = try readConfig(a, io, src_dir);
    defer a.free(conf.raw);
    try expectPairMajorOf(a, io, po_dir, pm_dir, conf.cfg.layers - conf.cfg.dense_layers);
}

// Hy4-Flash is converted pair-major always: a pack-order write is made by no run, so the pack order is the reference. Both code
// sources write the routed shards (the synthetic banks here, the searched ones in the identity pack).
test "hy4 convert: a synthetic pack's routed experts are the pack order's tiles pair-major, every other file the same (HY4_BF16_TINY_STQ)" {
    try convertBothLayouts(.synthetic);
}

test "hy4 convert: an identity-Hessian pack's routed experts are the pack order's tiles pair-major, every other file the same (HY4_BF16_TINY_STQ)" {
    try convertBothLayouts(.identity_hessian);
}

// A pair-major run goes on from the routed shards of its own order only: a pack-order shard of the same revision is refused by
// name, as a pair-major one is by a run in the pack order, never kept beside shards of the other order.
test "hy4 convert: --resume across the two orders is refused by name, in both directions (HY4_BF16_TINY_STQ)" {
    const src_env = std.c.getenv("HY4_BF16_TINY_STQ") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const paths = try capturePaths(a, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
    defer for (paths) |p| a.free(p);
    const first_layer = [_]u32{2};
    var cut = Options{ .model_dir = std.mem.span(src_env), .out_dir = paths[1], .rate = .{ .n = 64 }, .codes = .synthetic, .version = "test", .only_layers = &first_layer };
    _ = try runLayout(a, io, cut, null, false);
    cut.out_dir = paths[2];
    _ = try run(a, io, cut, null);
    cut.only_layers = null;
    cut.resuming = true;
    cut.out_dir = paths[1];
    try testing.expectError(error.ResumeTileOrderMismatch, run(a, io, cut, null));
    cut.out_dir = paths[2];
    try testing.expectError(error.ResumeTileOrderMismatch, runLayout(a, io, cut, null, false));
}
