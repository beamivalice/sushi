const std = @import("std");
const io_util = @import("mlx_host").io_util;
const host = @import("mlx_host");
const mlx = host.mlx;
const kld = host.kld;
const imatrix = host.imatrix;
const calib = @import("calib.zig");
const convert = @import("convert.zig");
const driver = @import("driver.zig");
const hessians = @import("hessians.zig");
const layer_stage = @import("layer_stage.zig");
const pack = @import("pack.zig");
const source = @import("source.zig");
const stages_mod = @import("stages.zig");
const trunk_mod = @import("trunk.zig");
const trunk_stq = @import("trunk_stq.zig");
const trunk_from = @import("trunk_from.zig");
const sft = @import("safetensors.zig");
const api = @import("sushi_exl3");
const format = api.format;
const pool_mod = @import("pool.zig");
const rate_plan = @import("rate_plan.zig");

const stage_a = "stage-capture";
const stage_b = "stage-forward";
const stage_bf16 = "stage-bf16";

const max_rank = 4;

/// One layer boundary's hidden state for every calibration window, back to back. Axis 0 is the batch and axis 1
/// the token axis; a batch of windows takes its count and their length there.
pub const Hidden = struct {
    allocator: std.mem.Allocator,
    starts: []const usize,
    rank: usize,
    shape: [max_rank]c_int,
    dtype: mlx.mlx_dtype,
    itemsize: usize,
    token_bytes: usize,
    bytes: []u8,
    boundary: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator, starts: []const usize, extents: []const c_int, dtype: mlx.mlx_dtype, itemsize: usize) !Hidden {
        if (extents.len < 2 or extents.len > max_rank) return error.HiddenShapeMismatch;
        var shape: [max_rank]c_int = @splat(1);
        var per: usize = itemsize;
        for (extents, 0..) |d, i| {
            shape[i] = d;
            if (i > 1) per *= @intCast(d);
        }
        return .{ .allocator = allocator, .starts = starts, .rank = extents.len, .shape = shape, .dtype = dtype, .itemsize = itemsize, .token_bytes = per, .bytes = try allocator.alloc(u8, per * starts[starts.len - 1]) };
    }

    pub fn dims(self: *const Hidden) []const c_int {
        return self.shape[0..self.rank];
    }

    pub fn deinit(self: *Hidden) void {
        self.allocator.free(self.bytes);
    }

    pub fn window(self: *const Hidden, i: usize) []u8 {
        return self.bytes[self.starts[i] * self.token_bytes .. self.starts[i + 1] * self.token_bytes];
    }

    fn batchBytes(self: *const Hidden, b: trunk_mod.Batch) []u8 {
        return self.bytes[self.starts[b.first] * self.token_bytes .. self.starts[b.first + b.count] * self.token_bytes];
    }

    pub fn array(self: *const Hidden, i: usize) mlx.mlx_array {
        return self.batchArray(.{ .first = i, .count = 1, .len = self.starts[i + 1] - self.starts[i] });
    }

    pub fn batchArray(self: *const Hidden, b: trunk_mod.Batch) mlx.mlx_array {
        var shape = self.shape;
        shape[0] = @intCast(b.count);
        shape[1] = @intCast(b.len);
        return mlx.mlx_array_new_data(self.batchBytes(b).ptr, &shape, @intCast(self.rank), self.dtype);
    }

    pub fn store(self: *Hidden, s: mlx.mlx_stream, i: usize, arr: mlx.mlx_array) !void {
        return self.storeBatch(s, .{ .first = i, .count = 1, .len = self.starts[i + 1] - self.starts[i] }, arr);
    }

    pub fn storeBatch(self: *Hidden, s: mlx.mlx_stream, b: trunk_mod.Batch, arr: mlx.mlx_array) !void {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_contiguous(&flat, arr, false, s));
        try mlx.check(mlx.mlx_array_eval(flat));
        const dst = self.batchBytes(b);
        if (mlx.mlx_array_size(flat) * mlx.mlx_array_itemsize(flat) != dst.len) return error.HiddenShapeMismatch;
        const data = mlx.mlx_array_data_uint8(flat) orelse return error.HiddenUnreadable;
        @memcpy(dst, data[0..dst.len]);
    }
};

fn configInt(allocator: std.mem.Allocator, config: []const u8, key: []const u8) !c_int {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, config, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.SourceConfigInvalid;
    const text = parsed.value.object.get("text_config") orelse return error.SourceConfigInvalid;
    const v = text.object.get(key) orelse return error.SourceConfigInvalid;
    if (v != .integer or v.integer <= 0) return error.SourceConfigInvalid;
    return @intCast(v.integer);
}

fn secs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

fn configFloat(allocator: std.mem.Allocator, config: []const u8, key: []const u8) !f32 {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, config, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.SourceConfigInvalid;
    const text = parsed.value.object.get("text_config") orelse return error.SourceConfigInvalid;
    const v = text.object.get(key) orelse return error.SourceConfigInvalid;
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => error.SourceConfigInvalid,
    };
}

pub const TrunkError = struct { name: []u8, rel_err: f64 };

/// Every trunk linear the staged layer runs, by the input it reads.
fn watchTrunk(grams: *trunk_mod.Grams, xfm: *host.trunk.Trunk, ple: bool) !void {
    const lw = &(xfm.moe_layers orelse return error.TrunkLinearMissing)[0];
    const hc_attn = lw.hc_attn orelse return error.TrunkLinearMissing;
    const hc_mlp = lw.hc_mlp orelse return error.TrunkLinearMissing;
    try grams.watch(hc_attn.down_w, .attn_n);
    try grams.watch(hc_attn.up_w, .attn_mix);
    try grams.watch(hc_mlp.down_w, .mlp_n);
    try grams.watch(hc_mlp.up_w, .mlp_mix);
    switch (lw.attn) {
        .linear => |la| {
            for ([_]mlx.mlx_array{ la.qkv_w, la.z_w, la.a_w, la.b_w }) |w| try grams.watch(w, .attn_in);
            try grams.watch(la.out_w, .attn_out);
        },
        .full => |fa| {
            for ([_]mlx.mlx_array{ fa.q_w, fa.k_w, fa.v_w, fa.idx_qk_w }) |w| try grams.watch(w, .attn_in);
            try grams.watch(fa.o_w, .attn_out);
        },
    }
    switch (lw.mlp) {
        .moe => |mw| {
            try grams.watch(mw.shared_gate_w, .moe_in);
            try grams.watch(mw.shared_up_w, .moe_in);
            try grams.watch(mw.shared_down_w, .shared_down);
        },
        .dense => return error.TrunkLinearMissing,
    }
    if (ple) {
        const pw = lw.ple orelse return error.TrunkLinearMissing;
        try grams.watch(pw.key_w, .ple);
        try grams.watch(pw.value_w, .ple);
    }
}

/// The final mixer's inputs over one window's last boundary `x` ([..., hc * hidden] rows): the group-normalized
/// streams, the low-rank mix input and the mixed row lm_head reads.
fn headGrams(grams: *trunk_mod.Grams, s: mlx.mlx_stream, x: mlx.mlx_array, norm: mlx.mlx_array, down: mlx.mlx_array, up: mlx.mlx_array, hc: c_int, hidden: c_int, eps: f32) !void {
    const rows: c_int = @intCast(@divExact(mlx.mlx_array_size(x), @as(usize, @intCast(hc * hidden))));
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_reshape(&x3, x, &[_]c_int{ rows, hc, hidden }, 3, s));
    var ones = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones);
    try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{hidden}, 1, .bfloat16, s));
    var rn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rn);
    try mlx.check(mlx.mlx_fast_rms_norm(&rn, x3, ones, eps, s));
    var n3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n3);
    try mlx.check(mlx.mlx_multiply(&n3, rn, norm, s));
    var n = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n);
    try mlx.check(mlx.mlx_reshape(&n, n3, &[_]c_int{ rows, hc * hidden }, 2, s));
    var dt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dt);
    try mlx.check(mlx.mlx_transpose(&dt, down, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_matmul(&d, n, dt, s));
    const inv = mlx.mlx_array_new_float(1.0 / @as(f32, @floatFromInt(hc)));
    defer _ = mlx.mlx_array_free(inv);
    var inv_bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv_bf);
    try mlx.check(mlx.mlx_astype(&inv_bf, inv, .bfloat16, s));
    var ds = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ds);
    try mlx.check(mlx.mlx_multiply(&ds, d, inv_bf, s));
    var dsig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dsig);
    try mlx.check(mlx.mlx_sigmoid(&dsig, ds, s));
    var mix_in = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mix_in);
    try mlx.check(mlx.mlx_multiply(&mix_in, ds, dsig, s));
    var ut = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ut);
    try mlx.check(mlx.mlx_transpose(&ut, up, s));
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_matmul(&u, mix_in, ut, s));
    var mix = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mix);
    try mlx.check(mlx.mlx_sigmoid(&mix, u, s));
    var m3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m3);
    try mlx.check(mlx.mlx_reshape(&m3, mix, &[_]c_int{ rows, hc, hidden }, 3, s));
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    try mlx.check(mlx.mlx_multiply(&prod, m3, n3, s));
    var mixed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mixed);
    try mlx.check(mlx.mlx_mean_axis(&mixed, prod, 1, false, s));
    if (grams.want.contains(.final_n)) try grams.add(.final_n, n);
    if (grams.want.contains(.final_mix)) try grams.add(.final_mix, mix_in);
    try grams.add(.head_in, mixed);
}

fn hiddenName(buf: []u8, boundary: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "hidden-{d}.bin", .{boundary});
}

/// A BF16-path boundary file, stored or carried in-run (`.capture/`); `tmp` is the in-run one being written.
fn boundaryName(buf: []u8, boundary: u32, tmp: bool) ![]const u8 {
    if (!tmp) return driver.storedBoundaryName(buf, boundary);
    return std.fmt.bufPrint(buf, ".boundary-{d:0>2}.u16.tmp", .{boundary});
}

const embed_key = "model.language_model.embed_tokens.weight";

pub const RowsTrace = struct {
    x: []f32,
    ids: []i32,
    topk: usize,
};

/// `inputs[L - 1]`: window 0's rows of the reference input to layer L >= 1 (layer 0 reads the run's own embedding
/// boundary); `outputs[L]`: the BF16 layer's output rows for it.
pub const Isolate = struct { inputs: []const []const u8, outputs: std.ArrayList([]u8) = .empty };

pub const Capture = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    s: mlx.mlx_stream,
    windows: *const calib.Windows,
    src: *const source.SafetensorsSource,
    scratch: std.Io.Dir = undefined,
    scratch_abs: []u8 = &.{},
    trace: ?*std.ArrayList([]u8) = null,
    /// Test seam: window 0's rows of each in-run BF16 boundary, as published.
    bf16_trace: ?*std.ArrayList([]u8) = null,
    /// Test seam: each BF16 layer also runs window 0 alone from a given input, to measure one layer apart from the
    /// layers before it.
    bf16_isolate: ?*Isolate = null,
    rows_trace: ?*std.ArrayList(RowsTrace) = null,
    /// Test seam: the BF16 path's MLP input rows of each layer's target.
    rows_f_trace: ?*std.ArrayList(RowsTrace) = null,
    /// Test seam: `saveBoundaries` fails with `InjectedStop` after this many of its steps.
    stop_after: ?usize = null,
    steps: usize = 0,
    ssd_budget_bytes: u64,
    info: ?driver.Info = null,
    hidden: ?Hidden = null,
    rows: ?imatrix.CapturedRows = null,
    /// The same tokens' MLP input rows on the BF16 path, for the compensated target.
    rows_f: ?imatrix.CapturedRows = null,
    rows_h: ?hessians.RowsHessians = null,
    hessian_stream: ?mlx.mlx_stream = null,
    /// Re-rounded trunk (`Info.trunk_gptq`): the layer's linears from `begin` until `done` stages them, the Gram
    /// sums, and the windows the Grams read.
    trunk: ?trunk_mod.Trunk = null,
    grams: ?trunk_mod.Grams = null,
    /// Every forward runs these batches of windows; the trunk Grams read the strided ones (`trunk.strided`).
    batches: []trunk_mod.Batch = &.{},
    strided_rows: u64 = 0,
    pool: ?*pool_mod.Pool = null,
    /// The down projections' rows and routing, through the layer's re-rounded MLP input side.
    rows_c: ?imatrix.CapturedRows = null,
    rows_hd: ?hessians.RowsHessians = null,
    /// Each re-rounded tensor's tr(E H E') / tr(W H W'), by served name, as the log reports it.
    trunk_errors: ?*std.ArrayList(TrunkError) = null,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, windows: *const calib.Windows, src: *const source.SafetensorsSource, ssd_budget_bytes: u64) Capture {
        return .{ .allocator = allocator, .io = io, .s = s, .windows = windows, .src = src, .ssd_budget_bytes = ssd_budget_bytes };
    }

    /// The Hessians run on a stream of their own so their grams overlap the search queued on `s`.
    fn hessianStream(self: *Capture) mlx.mlx_stream {
        if (self.hessian_stream) |hs| return hs;
        const dev = mlx.mlx_device_new_type(.gpu, 0);
        defer _ = mlx.mlx_device_free(dev);
        const hs = if (mlx.streamIsGpu(self.s)) mlx.mlx_stream_new_device(dev) else self.s;
        self.hessian_stream = hs;
        return hs;
    }

    pub fn deinit(self: *Capture) void {
        self.dropRows();
        self.dropTrunk();
        if (self.grams) |*g| g.deinit();
        if (self.pool) |p| p.deinit();
        if (self.batches.len != 0) self.allocator.free(self.batches);
        if (self.hessian_stream) |hs| if (hs.ctx != self.s.ctx) {
            _ = mlx.mlx_stream_free(hs);
        };
        if (self.hidden) |*h| h.deinit();
        if (self.scratch_abs.len != 0) {
            self.allocator.free(self.scratch_abs);
            self.scratch.close(self.io);
        }
    }

    pub fn calibrator(self: *Capture) driver.Calibrator {
        return .{ .ctx = self, .prepare = prepare, .ready = ready, .begin = begin, .done = done, .finish = finish };
    }

    fn dropTrunk(self: *Capture) void {
        if (self.trunk) |*t| t.deinit();
        self.trunk = null;
    }

    fn dropRows(self: *Capture) void {
        if (self.rows_h) |*r| r.deinit();
        self.rows_h = null;
        if (self.rows_hd) |*r| r.deinit();
        self.rows_hd = null;
        for ([_]*?imatrix.CapturedRows{ &self.rows, &self.rows_f, &self.rows_c }) |slot| {
            if (slot.*) |r| {
                _ = mlx.mlx_array_free(r.x);
                self.allocator.free(r.ids);
            }
            slot.* = null;
        }
    }

    fn prepare(ctx: *anyopaque, info: driver.Info) anyerror!void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        self.info = info;
        try info.pack_dir.createDirPath(self.io, ".capture");
        self.scratch = try info.pack_dir.openDir(self.io, ".capture", .{ .iterate = true });
        var abs_buf: [std.fs.max_path_bytes]u8 = undefined;
        self.scratch_abs = try self.allocator.dupe(u8, abs_buf[0..try self.scratch.realPath(self.io, &abs_buf)]);
        var map = try layer_stage.PackMap.scan(self.allocator, self.io, info.pack_dir);
        defer map.deinit();
        try layer_stage.writeGlobals(self.allocator, self.io, &map, self.scratch);
        try layer_stage.writePlaceholderExperts(self.allocator, self.io, self.scratch, info.geom, info.rate, info.dec);
        if (self.batches.len == 0) {
            // The re-rounded trunk batches windows as its reference does; the other trunks keep one window per forward.
            self.batches = try trunk_mod.batchPlan(self.allocator, self.windows.starts, if (info.trunk_gptq or info.trunk_stq != null) trunk_mod.batch_tokens else 0);
            self.strided_rows = 0;
            for (self.batches, 0..) |b, bi| {
                if (trunk_mod.strided(bi)) self.strided_rows += b.count * b.len;
            }
        }
        if ((info.trunk_gptq or info.trunk_stq != null) and self.pool == null) {
            self.pool = try pool_mod.Pool.init(self.allocator, pool_mod.cpuCount());
            self.grams = trunk_mod.Grams.init(self.allocator, self.s);
        }
    }

    /// A layer needs no capture when a later boundary is saved: each save drops the one before, so a resumed run
    /// finds only the boundary of its first unfinished layer.
    fn ready(ctx: *anyopaque, layer: u32, last: bool) bool {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        if (last) return true;
        var it = self.scratch.iterate();
        while (it.next(self.io) catch return false) |e| {
            if (e.kind != .file or !std.mem.startsWith(u8, e.name, "hidden-") or !std.mem.endsWith(u8, e.name, ".bin")) continue;
            const at = std.fmt.parseInt(u32, e.name["hidden-".len .. e.name.len - ".bin".len], 10) catch continue;
            if (at > layer) return true;
        }
        return false;
    }

    fn loadBoundary(self: *Capture, layer: u32) !void {
        if (self.hidden) |h| if (h.boundary != null and h.boundary.? == layer) return;
        var buf: [48]u8 = undefined;
        const name = try hiddenName(&buf, layer);
        var file = try self.scratch.openFile(self.io, name, .{});
        defer file.close(self.io);
        var header: [max_rank + 3]c_int = undefined;
        const header_bytes = std.mem.sliceAsBytes(&header);
        const size = (try file.stat(self.io)).size;
        if (size < header_bytes.len or try file.readPositionalAll(self.io, header_bytes, 0) != header_bytes.len) return error.HiddenScratchCorrupt;
        if (header[0] < 1 or header[0] > max_rank) return error.HiddenScratchCorrupt;
        const dtype: mlx.mlx_dtype = @fromBackingInt(@intCast(header[max_rank + 1]));
        var h = try Hidden.init(self.allocator, self.windows.starts, header[1..][0..@intCast(header[0])], dtype, @intCast(header[max_rank + 2]));
        errdefer h.deinit();
        if (size - header_bytes.len != h.bytes.len) return error.HiddenScratchCorrupt;
        try pack.readAllChunked(file, self.io, h.bytes, header_bytes.len, pack.read_chunk);
        h.boundary = layer;
        if (self.hidden) |*old| old.deinit();
        self.hidden = h;
    }

    fn saveBoundary(self: *Capture, layer: u32) !void {
        const h = &self.hidden.?;
        var buf: [48]u8 = undefined;
        const name = try hiddenName(&buf, layer);
        var tmp_buf: [64]u8 = undefined;
        const tmp = try std.fmt.bufPrint(&tmp_buf, ".{s}.tmp", .{name});
        var file = try self.scratch.createFile(self.io, tmp, .{});
        {
            defer file.close(self.io);
            const header = [max_rank + 3]c_int{ @intCast(h.rank), h.shape[0], h.shape[1], h.shape[2], h.shape[3], @backingInt(h.dtype), @intCast(h.itemsize) };
            try file.writeStreamingAll(self.io, std.mem.sliceAsBytes(&header));
            try pack.writeAllChunked(file, self.io, h.bytes);
        }
        try self.scratch.rename(tmp, self.scratch, name, self.io);
        try self.stepDone();
        if (self.trace) |list| {
            try list.ensureUnusedCapacity(self.allocator, 1);
            list.appendAssumeCapacity(try self.allocator.dupe(u8, h.window(0)));
        }
        if (layer > 0) {
            var old_buf: [48]u8 = undefined;
            self.scratch.deleteFile(self.io, try hiddenName(&old_buf, layer - 1)) catch {};
        }
        try self.stepDone();
    }

    fn stepDone(self: *Capture) !void {
        self.steps += 1;
        if (self.stop_after == self.steps) return error.InjectedStop;
    }

    fn stageFor(self: *Capture, name: []const u8, layer: u32, experts: layer_stage.Stage.Experts) ![]u8 {
        return self.stageWith(name, layer, experts, if (self.trunk) |*t| t else null);
    }

    /// `trunk`'s linears, as far as it has rounded them, in place of the pack's.
    fn stageWith(self: *Capture, name: []const u8, layer: u32, experts: layer_stage.Stage.Experts, trunk: ?*const trunk_mod.Trunk) ![]u8 {
        const linears = if (trunk) |t| try t.staged(self.allocator) else &.{};
        defer if (linears.len != 0) self.allocator.free(linears);
        const info = self.info.?;
        var map = try layer_stage.PackMap.scan(self.allocator, self.io, info.pack_dir);
        defer map.deinit();
        try layer_stage.stageLayer(self.allocator, self.io, &map, self.scratch, name, .{
            .layer = layer,
            .kind = if (info.layer_full[layer]) .full else .linear,
            .ple_layer = info.ple_layer,
            .experts = experts,
            .source_config = info.source_config,
            .source_dir_abs = info.source_dir_abs,
            .root_abs = self.scratch_abs,
            .pack_abs = info.pack_abs,
            .trunk_bits = info.trunk_bits,
            .trunk_group = info.trunk_group,
            .overrides = info.overrides,
            .ngram = info.ngram,
            .eq = info.eq,
            .linears = linears,
        });
        return std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.scratch_abs, name });
    }

    /// The BF16 path's stored boundary (`boundary-LL.u16`: one bf16 row of every stream per token, windows end to
    /// end in calibration order) as a forward's input, in the forward's own [batch, len, streams x hidden] layout:
    /// only the PLE layer reshapes a four-axis stream. With `next`, the in-run path writes the layer's output there
    /// as the next boundary, in the same layout.
    const Stored = struct {
        file: std.Io.File,
        hc: c_int,
        hidden: c_int,
        buf: []u8,
        next: ?std.Io.File = null,
    };

    fn writeNext(self: *Capture, st: Stored, b: trunk_mod.Batch, slot: mlx.mlx_array) !void {
        if (mlx.mlx_array_dtype(slot) != .bfloat16) return error.BoundaryDtype;
        const row_bytes: usize = @intCast(st.hc * st.hidden * 2);
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_contiguous(&flat, slot, false, self.s));
        try mlx.check(mlx.mlx_array_eval(flat));
        const len = b.count * b.len * row_bytes;
        if (mlx.mlx_array_size(flat) * 2 != len) return error.BoundaryRowsMismatch;
        const data = mlx.mlx_array_data_uint8(flat) orelse return error.HiddenUnreadable;
        try st.next.?.writePositionalAll(self.io, data[0..len], self.windows.starts[b.first] * row_bytes);
    }

    /// The BF16 path's first boundary, as `name` in `dir`: each calibration token's source embedding row, once per
    /// stream. `chunk` tokens are written at a time.
    fn writeEmbeddingBoundary(self: *Capture, dir: std.Io.Dir, name: []const u8, hc: usize, hidden: usize, chunk: usize) !void {
        const loc = self.src.tensors.get(embed_key) orelse return error.EmbeddingMissing;
        if (!std.mem.eql(u8, loc.dtype, "BF16") or loc.shape.len != 2 or loc.shape[1] != hidden) return error.BoundaryFileShape;
        const table = try self.allocator.alloc(u8, loc.end - loc.start);
        defer self.allocator.free(table);
        try self.src.readRange(loc, 0, table);
        const row = hidden * 2;
        const buf = try self.allocator.alloc(u8, chunk * hc * row);
        defer self.allocator.free(buf);
        var tmp_buf: [64]u8 = undefined;
        const tmp = try std.fmt.bufPrint(&tmp_buf, ".{s}.tmp", .{name});
        {
            var file = try dir.createFile(self.io, tmp, .{});
            defer file.close(self.io);
            const ids = self.windows.ids[0..self.windows.tokens()];
            var at: usize = 0;
            while (at < ids.len) : (at += chunk) {
                const part = ids[at..@min(at + chunk, ids.len)];
                for (part, 0..) |id, i| {
                    if (id >= loc.shape[0]) return error.TokenOutOfRange;
                    for (0..hc) |s| @memcpy(buf[(i * hc + s) * row ..][0..row], table[id * row ..][0..row]);
                }
                try file.writePositionalAll(self.io, buf[0 .. part.len * hc * row], at * hc * row);
            }
        }
        try dir.rename(tmp, dir, name, self.io);
    }

    fn firstWindowRows(self: *Capture, name: []const u8) ![]u8 {
        const hc = try configInt(self.allocator, self.info.?.source_config, "hc_count");
        const len = (self.windows.starts[1] - self.windows.starts[0]) * @as(usize, @intCast(hc)) * self.info.?.geom.hidden * 2;
        var f = try self.scratch.openFile(self.io, name, .{});
        defer f.close(self.io);
        const out = try self.allocator.alloc(u8, len);
        errdefer self.allocator.free(out);
        if (try f.readPositionalAll(self.io, out, 0) != len) return error.BoundaryFileShape;
        return out;
    }

    /// Publishes the in-run BF16 path's next boundary, then the quantized one, and only then drops the BF16 path's
    /// current one: a run stopped between any two steps resumes with both boundaries of its layer on disk.
    fn saveBoundaries(self: *Capture, next: u32, in_run: bool) !void {
        var a: [48]u8 = undefined;
        var b: [48]u8 = undefined;
        if (in_run) {
            const published = try boundaryName(&b, next, false);
            try self.scratch.rename(try boundaryName(&a, next, true), self.scratch, published, self.io);
            try self.stepDone();
            if (self.bf16_trace) |list| try list.append(self.allocator, try self.firstWindowRows(published));
        }
        try self.saveBoundary(next);
        if (in_run) self.scratch.deleteFile(self.io, try boundaryName(&a, next - 1, false)) catch {};
        try self.stepDone();
    }

    fn storedWindow(self: *Capture, st: Stored, w: usize) !mlx.mlx_array {
        return self.storedBatch(st, .{ .first = w, .count = 1, .len = self.windows.starts[w + 1] - self.windows.starts[w] });
    }

    fn storedBatch(self: *Capture, st: Stored, b: trunk_mod.Batch) !mlx.mlx_array {
        const row_bytes: usize = @intCast(st.hc * st.hidden * 2);
        const lo = self.windows.starts[b.first];
        const bytes = st.buf[0 .. b.count * b.len * row_bytes];
        if (try st.file.readPositionalAll(self.io, bytes, lo * row_bytes) != bytes.len) return error.BoundaryFileShort;
        return mlx.mlx_array_new_data(bytes.ptr, &[_]c_int{ @intCast(b.count), @intCast(b.len), st.hc * st.hidden }, 3, .bfloat16);
    }

    fn forwardWindows(self: *Capture, loaded: *kld.Loaded, layer: u32, out: ?*?Hidden) !void {
        return self.forwardFrom(loaded, layer, out, null, .{});
    }

    /// `grams` sums its watched linears' inputs over the strided windows; `only_tapped` runs just those.
    /// For batches forwarded the Grams' first (`trunk_mod.strided`) and then the rest, each in plan order: the
    /// forward that ran plan batch i.
    fn stridedFirstOrder(allocator: std.mem.Allocator, batches: usize) ![]usize {
        const order = try allocator.alloc(usize, batches);
        var at: usize = 0;
        for (0..batches) |bi| if (trunk_mod.strided(bi)) {
            order[bi] = at;
            at += 1;
        };
        for (0..batches) |bi| if (!trunk_mod.strided(bi)) {
            order[bi] = at;
            at += 1;
        };
        return order;
    }

    /// Which batches a forward runs: all, only those the Grams read (`trunk_mod.strided`), or only the others.
    const Tap = struct { grams: ?*trunk_mod.Grams = null, only_tapped: bool = false, skip_tapped: bool = false };

    fn forwardFrom(self: *Capture, loaded: *kld.Loaded, layer: u32, out: ?*?Hidden, stored: ?Stored, tap: Tap) !void {
        const info = self.info.?;
        for (self.batches, 0..) |b, bi| {
            if (tap.skip_tapped and trunk_mod.strided(bi)) continue;
            const tapped = tap.grams != null and trunk_mod.strided(bi);
            if (tap.only_tapped and !tapped) continue;
            if (tapped) tap.grams.?.arm();
            defer if (tapped) tap.grams.?.disarm();
            try loaded.xfm.resetCache();
            loaded.xfm.bindOwnRows(calib.window_len);
            var ctx = loaded.xfm.defaultCtx();
            ctx.skip_lm_head = true;
            const input = if (stored) |st| try self.storedBatch(st, b) else if (layer > 0) self.hidden.?.batchArray(b) else mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(input);
            host.model_handle.setLayerCapture(&ctx, if (layer > 0 or stored != null) input else null, layer != info.ple_layer);
            var slot = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(slot);
            const ids = [_]u32{0};
            var cl = host.model_handle.CaptureLayers{ .ids = &ids, .out = (&slot)[0..1] };
            const next = if (stored) |st| st.next != null else false;
            if (out != null or next) ctx.capture_layers = &cl;
            const tokens = batchTokensOf(self.allocator, self.windows, b);
            defer _ = mlx.mlx_array_free(tokens);
            const result = try loaded.xfm.forwardWith(&ctx, tokens);
            defer _ = mlx.mlx_array_free(result);
            try mlx.check(mlx.mlx_array_eval(result));
            if (tapped) try tap.grams.?.settle();
            if (next) try self.writeNext(stored.?, b, slot);
            if (out) |o| {
                if (o.* == null) {
                    o.* = try Hidden.init(self.allocator, self.windows.starts, mlx.getShape(slot), mlx.mlx_array_dtype(slot), mlx.mlx_array_itemsize(slot));
                }
                try o.*.?.storeBatch(self.s, b, slot);
            }
        }
    }

    fn begin(ctx: *anyopaque, layer: u32, mtp: bool) anyerror!?convert.HessianSource {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        if (mtp) return null;
        if (self.info.?.trunk_gptq) return self.beginRerounded(layer);
        self.dropRows();
        if (layer > 0) try self.loadBoundary(layer);
        const dir = try self.stageFor(stage_a, layer, .placeholder);
        defer self.allocator.free(dir);
        imatrix.rows_request = 0;
        defer imatrix.rows_request = null;
        var loaded = try kld.loadModel(self.io, self.allocator, .{ .model_dir = dir, .ssd_budget_bytes = self.ssd_budget_bytes });
        defer loaded.deinit();
        try self.forwardWindows(loaded, layer, null);
        const tap = loaded.xfm.imatrix orelse return error.CaptureTapNotArmed;
        self.rows = try tap.takeRows();
        const info = self.info.?;
        if (layer < info.perms.len) if (info.perms[layer]) |perm| try toSourceIds(self.rows.?.ids, perm);
        if (self.rows_trace) |list| try list.append(self.allocator, try copyRows(self.allocator, self.s, self.rows.?));
        if (self.targetDir()) |bdir| {
            self.rows_f = try self.targetRows(loaded, dir, layer, bdir);
            try self.logDrift(layer);
        }
        self.rows_h = try hessians.RowsHessians.init(self.allocator, self.hessianStream(), self.src, layer, false, info.geom, self.rows.?.x, self.rows.?.ids, self.rows.?.topk);
        if (self.rows_f) |f| self.rows_h.?.xf = f.x;
        return self.rows_h.?.hessianSource();
    }

    fn trunkSpec(self: *const Capture) trunk_mod.Spec {
        const info = self.info.?;
        return .{ .bits = info.trunk_bits, .group = info.trunk_group, .overrides = info.overrides };
    }

    /// Passes A-C of the re-rounded trunk (`trunk.zig`). Each pass loads the layer staged through the linears rounded
    /// so far and the source weights of the rest, sums its groups' Grams along the quantized path and rounds them.
    /// The routed experts' gate/up rows and routing come from pass B, their down rows and routing from pass C, and
    /// the BF16 path's rows from the all-source layer on the stored boundary.
    fn beginRerounded(self: *Capture, layer: u32) !?convert.HessianSource {
        const info = self.info.?;
        self.dropRows();
        self.dropTrunk();
        if (layer > 0) try self.loadBoundary(layer);
        const has_ple = layer == info.ple_layer;
        self.trunk = try trunk_mod.Trunk.forLayer(self.allocator, self.src, layer, info.layer_full[layer], has_ple, self.trunkSpec());
        const grams = &self.grams.?;
        imatrix.rows_request = 0;
        defer imatrix.rows_request = null;
        for ([_]trunk_mod.Pass{ .a, .b, .c }) |pass| {
            const t0 = stages_mod.nowNs();
            const dir = try self.stageFor(stage_a, layer, .placeholder);
            defer self.allocator.free(dir);
            var loaded = try kld.loadModel(self.io, self.allocator, .{ .model_dir = dir, .ssd_budget_bytes = self.ssd_budget_bytes });
            defer loaded.deinit();
            const tap = loaded.xfm.imatrix orelse return error.CaptureTapNotArmed;
            grams.reset();
            grams.forget();
            try watchTrunk(grams, &loaded.xfm, has_ple);
            grams.want = trunk_mod.groupsOf(pass);
            // Passes A and B round on a host thread once their Grams are complete, beside the forwards that are
            // left: pass A's BF16 path, and pass B's batches the Grams do not read.
            var rounding: ?trunk_mod.Rounding = null;
            defer if (rounding) |*r| r.deinit();
            var t1: u64 = 0;
            switch (pass) {
                .a => {
                    try self.forwardFrom(loaded, layer, null, null, .{ .grams = grams, .only_tapped = true });
                    const strided = try tap.takeRows();
                    _ = mlx.mlx_array_free(strided.x);
                    self.allocator.free(strided.ids);
                    try grams.expectRows(self.strided_rows);
                    t1 = stages_mod.nowNs();
                    rounding = try trunk_mod.Rounding.begin(&self.trunk.?, self.pool, grams, pass);
                    try rounding.?.start();
                    if (self.targetDir()) |bdir| self.rows_f = try self.targetRows(loaded, dir, layer, bdir);
                },
                .b => {
                    try self.forwardFrom(loaded, layer, null, null, .{ .grams = grams, .only_tapped = true });
                    try grams.expectRows(self.strided_rows);
                    t1 = stages_mod.nowNs();
                    rounding = try trunk_mod.Rounding.begin(&self.trunk.?, self.pool, grams, pass);
                    try rounding.?.start();
                    try self.forwardFrom(loaded, layer, null, null, .{ .skip_tapped = true });
                    const order = try stridedFirstOrder(self.allocator, self.batches.len);
                    defer self.allocator.free(order);
                    self.rows = try tap.takeRowsInOrder(order);
                },
                .c => {
                    try self.forwardFrom(loaded, layer, null, null, .{ .grams = grams });
                    self.rows_c = try tap.takeRows();
                    try grams.expectRows(self.strided_rows);
                    t1 = stages_mod.nowNs();
                },
                .head => unreachable,
            }
            if (pass != .a) if (layer < info.perms.len) if (info.perms[layer]) |perm| try toSourceIds((if (pass == .b) self.rows else self.rows_c).?.ids, perm);
            const t2 = stages_mod.nowNs();
            if (rounding) |*r| try r.finish(grams) else try self.trunk.?.round(self.pool, grams, pass);
            host.log.info("[convert] layer {d}: trunk pass {s}: load and forwards {d:.1}s, rounding {d:.1}s ({d:.1}s past the forwards)\n", .{ layer, @tagName(pass), secs(t2 - t0), secs(stages_mod.nowNs() - t1), secs(stages_mod.nowNs() - t2) });
        }
        grams.reset();
        var name_buf: [64]u8 = undefined;
        try self.trunk.?.write(self.io, info.pack_dir, try trunk_mod.shardName(&name_buf, layer));
        self.reportTrunk(layer, &self.trunk.?);

        if (self.rows_trace) |list| try list.append(self.allocator, try copyRows(self.allocator, self.s, self.rows.?));
        if (self.rows_f != null) try self.logDrift(layer);
        const hs = self.hessianStream();
        self.rows_h = try hessians.RowsHessians.init(self.allocator, hs, self.src, layer, false, info.geom, self.rows.?.x, self.rows.?.ids, self.rows.?.topk);
        self.rows_hd = try hessians.RowsHessians.init(self.allocator, hs, self.src, layer, false, info.geom, self.rows_c.?.x, self.rows_c.?.ids, self.rows_c.?.topk);
        if (self.rows_f) |f| {
            self.rows_h.?.xf = f.x;
            self.rows_hd.?.xf = f.x;
        }
        self.rows_h.?.down_rows = &self.rows_hd.?;
        return self.rows_h.?.hessianSource();
    }

    /// The STQ trunk of `layer` (`trunk_stq.zig`): the passes of `beginRerounded` over the tensors that go onto the
    /// trellis, the layer's routed experts left as the pack holds them, so nothing but the Grams is taken from the
    /// forwards. Leaves `self.trunk` rounded; `stqAdvance` stages it.
    pub fn stqLayer(self: *Capture, layer: u32) !void {
        const info = self.info.?;
        self.dropRows();
        self.dropTrunk();
        if (layer > 0) try self.loadBoundary(layer);
        const has_ple = layer == info.ple_layer;
        self.trunk = try trunk_mod.Trunk.forLayerStq(self.allocator, self.src, layer, info.layer_full[layer], has_ple, info.trunk_stq.?);
        const grams = &self.grams.?;
        const active = self.trunk.?.activeGroups();
        for ([_]trunk_mod.Pass{ .a, .b, .c }) |pass| {
            const want = trunk_mod.groupsOf(pass).intersectWith(active);
            if (want.count() == 0) continue;
            const t0 = stages_mod.nowNs();
            const dir = try self.stageFor(stage_a, layer, .placeholder);
            defer self.allocator.free(dir);
            var loaded = try kld.loadModel(self.io, self.allocator, .{ .model_dir = dir, .ssd_budget_bytes = self.ssd_budget_bytes });
            defer loaded.deinit();
            grams.reset();
            grams.forget();
            try watchTrunk(grams, &loaded.xfm, has_ple);
            grams.want = want;
            try self.forwardFrom(loaded, layer, null, null, .{ .grams = grams, .only_tapped = true });
            try grams.expectRows(self.strided_rows);
            const t1 = stages_mod.nowNs();
            try self.trunk.?.round(self.pool, grams, pass);
            host.log.info("[convert] layer {d}: trunk pass {s}: load and forwards {d:.1}s, rounding {d:.1}s\n", .{ layer, @tagName(pass), secs(t1 - t0), secs(stages_mod.nowNs() - t1) });
        }
        grams.reset();
        self.reportTrunk(layer, &self.trunk.?);
    }

    /// Runs `layer` with its rounded trunk and the pack's own experts over the calibration, and stores the next
    /// boundary.
    pub fn stqAdvance(self: *Capture, layer: u32) !void {
        defer self.dropTrunk();
        var name_buf: [64]u8 = undefined;
        const shard_name = try std.fmt.bufPrint(&name_buf, "layer-{d}.safetensors", .{layer});
        const dir = try self.stageFor(stage_b, layer, .{ .shard = shard_name });
        defer self.allocator.free(dir);
        var loaded = try kld.loadModel(self.io, self.allocator, .{ .model_dir = dir });
        defer loaded.deinit();
        if (layer > 0) {
            try self.loadBoundary(layer);
            self.hidden.?.boundary = null;
            try self.forwardWindows(loaded, layer, &self.hidden);
            self.hidden.?.boundary = layer + 1;
        } else {
            var next: ?Hidden = null;
            errdefer if (next) |*h| h.deinit();
            try self.forwardWindows(loaded, layer, &next);
            next.?.boundary = layer + 1;
            if (self.hidden) |*old| old.deinit();
            self.hidden = next.?;
            next = null;
        }
        try self.saveBoundaries(layer + 1, false);
    }

    /// `lm_head` onto the trellis on the Gram of the mixed row it reads, taken on the last boundary through the
    /// source mixer. The caller writes and frees the returned trunk.
    pub fn stqHead(self: *Capture) !trunk_mod.Trunk {
        const info = self.info.?;
        var t = try trunk_mod.Trunk.forHeadStq(self.allocator, self.src, info.trunk_stq.?);
        errdefer t.deinit();
        try self.loadBoundary(info.layers);
        const grams = &self.grams.?;
        grams.reset();
        grams.forget();
        grams.want = trunk_mod.groupsOf(.head).intersectWith(t.activeGroups());
        try self.headGramsOver(&t, grams);
        try t.round(self.pool, grams, .head);
        grams.reset();
        self.reportTrunk(null, &t);
        return t;
    }

    fn reportTrunk(self: *Capture, layer: ?u32, t: *const trunk_mod.Trunk) void {
        for (t.tensors.items) |x| {
            if (x.round == .keep) continue;
            if (layer) |l| {
                host.log.info("[convert] layer {d}: trunk {s} rel err {e:.6}\n", .{ l, x.base, x.rel_err });
            } else host.log.info("[convert] head: trunk {s} rel err {e:.6}\n", .{ x.base, x.rel_err });
            if (self.trunk_errors) |list| {
                const name = self.allocator.dupe(u8, x.base) catch continue;
                list.append(self.allocator, .{ .name = name, .rel_err = x.rel_err }) catch self.allocator.free(name);
            }
        }
    }

    /// The head groups `grams.want` names, summed over the strided windows of the last boundary through the source
    /// mixer (`t`'s first two tensors).
    fn headGramsOver(self: *Capture, t: *const trunk_mod.Trunk, grams: *trunk_mod.Grams) !void {
        const info = self.info.?;
        const hc = try configInt(self.allocator, info.source_config, "hc_count");
        const eps = try configFloat(self.allocator, info.source_config, "rms_norm_eps");
        var map = try layer_stage.PackMap.scan(self.allocator, self.io, info.pack_dir);
        defer map.deinit();
        const norm_raw = try map.bytes(self.allocator, self.io, "language_model.model.hyper_connection_mixer.hc_norm.weight");
        defer self.allocator.free(norm_raw);
        const hidden: c_int = @intCast(info.geom.hidden);
        const norm = mlx.mlx_array_new_data(norm_raw.ptr, &[_]c_int{ hc, hidden }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(norm);
        const down = &t.tensors.items[0];
        const up = &t.tensors.items[1];
        const dw = mlx.mlx_array_new_data(down.source.ptr, &[_]c_int{ @intCast(down.out), @intCast(down.in) }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(dw);
        const uw = mlx.mlx_array_new_data(up.source.ptr, &[_]c_int{ @intCast(up.out), @intCast(up.in) }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(uw);
        for (self.batches, 0..) |b, bi| {
            if (!trunk_mod.strided(bi)) continue;
            const x = self.hidden.?.batchArray(b);
            defer _ = mlx.mlx_array_free(x);
            try headGrams(grams, self.s, x, norm, dw, uw, hc, hidden, eps);
            try grams.settle();
        }
        for ([_]trunk_mod.Group{ .final_n, .final_mix, .head_in }) |g| {
            if (grams.want.contains(g) and grams.rows.get(g) != self.strided_rows) return error.TrunkTapRows;
        }
    }

    /// The final mixer and lm_head re-rounded on Grams of the last boundary along the quantized path, read through
    /// the source mixer.
    fn head(self: *Capture) !void {
        const info = self.info.?;
        var t = try trunk_mod.Trunk.forHead(self.allocator, self.src, self.trunkSpec());
        defer t.deinit();
        const grams = &self.grams.?;
        grams.reset();
        grams.forget();
        grams.want = trunk_mod.groupsOf(.head);
        try self.headGramsOver(&t, grams);
        try t.round(self.pool, grams, .head);
        grams.reset();
        try t.write(self.io, info.pack_dir, trunk_mod.head_shard);
        self.reportTrunk(null, &t);
    }

    /// Where the BF16 path's boundaries are read: the stored ones, or the run's own scratch when it carries them.
    fn targetDir(self: *const Capture) ?[]const u8 {
        const info = self.info.?;
        return if (info.target_in_run) self.scratch_abs else info.target_boundaries;
    }

    /// The BF16 path's MLP input rows: the staged layer `loaded` (from `dir`) over the stored boundary. The PLE layer
    /// reads the source's BF16 n-gram shards there, as the in-run BF16 boundary does, not the pack's table.
    fn targetRows(self: *Capture, loaded: *kld.Loaded, dir: []const u8, layer: u32, bdir: []const u8) !imatrix.CapturedRows {
        const info = self.info.?;
        var result: imatrix.CapturedRows = undefined;
        if (layer == info.ple_layer) {
            const bf16 = try kld.loadModel(self.io, self.allocator, .{ .model_dir = dir, .ssd_budget_bytes = self.ssd_budget_bytes, .ngram_bf16_dir = info.source_dir_abs });
            defer bf16.deinit();
            try self.forwardStoredBoundary(bf16, layer, bdir, null);
            result = try (bf16.xfm.imatrix orelse return error.CaptureTapNotArmed).takeRows();
        } else {
            try self.forwardStoredBoundary(loaded, layer, bdir, null);
            result = try (loaded.xfm.imatrix orelse return error.CaptureTapNotArmed).takeRows();
        }
        if (self.rows_f_trace) |list| try list.append(self.allocator, try copyRows(self.allocator, self.s, result));
        return result;
    }

    fn forwardStoredBoundary(self: *Capture, loaded: *kld.Loaded, layer: u32, dir: []const u8, next: ?std.Io.File) !void {
        const info = self.info.?;
        const hc = try configInt(self.allocator, info.source_config, "hc_count");
        var name_buf: [48]u8 = undefined;
        if (info.target_in_run and layer == 0) {
            const first = try boundaryName(&name_buf, 0, false);
            self.scratch.access(self.io, first, .{}) catch try self.writeEmbeddingBoundary(self.scratch, first, @intCast(hc), info.geom.hidden, 4096);
        }
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, try boundaryName(&name_buf, layer, false) });
        var file = try std.Io.Dir.cwd().openFile(self.io, path, .{});
        defer file.close(self.io);
        const width: usize = @intCast(hc * @as(c_int, @intCast(info.geom.hidden)));
        if ((try file.stat(self.io)).size != self.windows.tokens() * width * 2) return error.BoundaryFileShape;
        var longest: usize = 0;
        for (self.batches) |b| longest = @max(longest, b.count * b.len);
        const buf = try self.allocator.alloc(u8, longest * width * 2);
        defer self.allocator.free(buf);
        try self.forwardFrom(loaded, layer, null, .{ .file = file, .hc = hc, .hidden = @intCast(info.geom.hidden), .buf = buf, .next = next }, .{});
    }

    /// How far the BF16 path's MLP inputs sit from the quantized path's, over all calibration rows: a sanity line
    /// (a misaligned boundary file reads as a drift near 1).
    fn logDrift(self: *Capture, layer: u32) !void {
        const q = self.rows.?.x;
        const f = self.rows_f.?.x;
        if (mlx.mlx_array_size(q) != mlx.mlx_array_size(f)) return error.BoundaryRowsMismatch;
        var d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d);
        try mlx.check(mlx.mlx_subtract(&d, f, q, self.s));
        var sq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sq);
        try mlx.check(mlx.mlx_multiply(&sq, d, d, self.s));
        var dn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dn);
        try mlx.check(mlx.mlx_sum(&dn, sq, false, self.s));
        var qq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qq);
        try mlx.check(mlx.mlx_multiply(&qq, q, q, self.s));
        var qn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qn);
        try mlx.check(mlx.mlx_sum(&qn, qq, false, self.s));
        try mlx.check(mlx.mlx_array_eval(dn));
        try mlx.check(mlx.mlx_array_eval(qn));
        const a = mlx.mlx_array_data_float32(dn) orelse return error.HiddenUnreadable;
        const b = mlx.mlx_array_data_float32(qn) orelse return error.HiddenUnreadable;
        host.log.info("[convert] layer {d}: BF16-path MLP inputs differ from the quantized path's by {d:.4} (relative)\n", .{ layer, @sqrt(a[0] / b[0]) });
    }

    /// The BF16 model's layer `layer` over the in-run boundary of its input, written as the next one: the source's
    /// trunk linears and routed experts, staged and forwarded like the capture's layers.
    fn forwardBf16Layer(self: *Capture, layer: u32) !void {
        const info = self.info.?;
        var source_trunk = try trunk_mod.Trunk.forLayer(self.allocator, self.src, layer, info.layer_full[layer], layer == info.ple_layer, self.trunkSpec());
        defer source_trunk.deinit();
        const perm = if (layer < info.perms.len) info.perms[layer] else null;
        const dir = try self.stageWith(stage_bf16, layer, .{ .source_bf16 = .{ .src = self.src, .perm = perm } }, &source_trunk);
        defer self.allocator.free(dir);
        // The PLE layer reads the source's BF16 n-gram shards, not the pack's table, which may be quantized.
        var loaded = try kld.loadModel(self.io, self.allocator, .{ .model_dir = dir, .ngram_bf16_dir = if (layer == info.ple_layer) info.source_dir_abs else "" });
        defer loaded.deinit();
        var name_buf: [48]u8 = undefined;
        var next = try self.scratch.createFile(self.io, try boundaryName(&name_buf, layer + 1, true), .{});
        defer next.close(self.io);
        try self.forwardStoredBoundary(loaded, layer, self.scratch_abs, next);
        if (self.bf16_isolate) |iso| try self.isolateWindow0(loaded, layer, iso);
    }

    fn isolateWindow0(self: *Capture, loaded: *kld.Loaded, layer: u32, iso: *Isolate) !void {
        const info = self.info.?;
        const hc = try configInt(self.allocator, info.source_config, "hc_count");
        var name_buf: [48]u8 = undefined;
        const first = if (layer == 0) try self.firstWindowRows(try boundaryName(&name_buf, 0, false)) else null;
        defer if (first) |f| self.allocator.free(f);
        const rows = first orelse iso.inputs[layer - 1];
        const len: c_int = @intCast(self.windows.starts[1] - self.windows.starts[0]);
        const input = mlx.mlx_array_new_data(rows.ptr, &[_]c_int{ 1, len, hc * @as(c_int, @intCast(info.geom.hidden)) }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(input);
        try loaded.xfm.resetCache();
        var ctx = loaded.xfm.defaultCtx();
        ctx.skip_lm_head = true;
        host.model_handle.setLayerCapture(&ctx, input, layer != info.ple_layer);
        var slot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(slot);
        const ids = [_]u32{0};
        var cl = host.model_handle.CaptureLayers{ .ids = &ids, .out = (&slot)[0..1] };
        ctx.capture_layers = &cl;
        const tokens = windowTokensOf(self.allocator, self.windows, 0);
        defer _ = mlx.mlx_array_free(tokens);
        const result = try loaded.xfm.forwardWith(&ctx, tokens);
        defer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_array_eval(result));
        try iso.outputs.append(self.allocator, try bf16Bytes(self.allocator, slot, rows.len));
    }

    fn done(ctx: *anyopaque, layer: u32, mtp: bool, reused: bool) anyerror!void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        if (mtp or reused) return;
        self.dropRows();
        const info = self.info.?;
        const last = layer + 1 >= info.layers;
        if (info.target_in_run and !last) {
            const t0 = stages_mod.nowNs();
            try self.forwardBf16Layer(layer);
            host.log.info("[convert] layer {d}: BF16 boundary {d:.1}s\n", .{ layer, secs(stages_mod.nowNs() - t0) });
        }
        if (last and !info.trunk_gptq) return;
        defer self.dropTrunk();
        var name_buf: [64]u8 = undefined;
        const shard_name = try std.fmt.bufPrint(&name_buf, "layer-{d}.safetensors", .{layer});
        const dir = try self.stageFor(stage_b, layer, .{ .shard = shard_name });
        defer self.allocator.free(dir);
        var loaded = try kld.loadModel(self.io, self.allocator, .{ .model_dir = dir });
        defer loaded.deinit();
        if (layer > 0) {
            // Each window's input is copied into its array before the window runs, so the next boundary
            // overwrites this one in place: one boundary is held, not two.
            try self.loadBoundary(layer);
            self.hidden.?.boundary = null;
            try self.forwardWindows(loaded, layer, &self.hidden);
            self.hidden.?.boundary = layer + 1;
        } else {
            var next: ?Hidden = null;
            errdefer if (next) |*h| h.deinit();
            try self.forwardWindows(loaded, layer, &next);
            next.?.boundary = layer + 1;
            if (self.hidden) |*old| old.deinit();
            self.hidden = next.?;
            next = null;
        }
        if (last) return self.head();
        try self.saveBoundaries(layer + 1, info.target_in_run);
    }

    fn finish(ctx: *anyopaque) void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        self.scratch.deleteTree(self.io, stage_a) catch {};
        self.scratch.deleteTree(self.io, stage_b) catch {};
        self.scratch.deleteTree(self.io, stage_bf16) catch {};
    }
};

/// The staged router's rows are in stored order; the Hessians are keyed by source expert id.
fn toSourceIds(ids: []i32, perm: []const u32) !void {
    for (ids) |*id| {
        if (id.* < 0 or id.* >= perm.len) return error.HessianExpertOutOfRange;
        id.* = @intCast(perm[@intCast(id.*)]);
    }
}

pub fn copyRows(allocator: std.mem.Allocator, s: mlx.mlx_stream, rows: imatrix.CapturedRows) !RowsTrace {
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_contiguous(&flat, rows.x, false, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    const data = mlx.mlx_array_data_float32(flat) orelse return error.HiddenUnreadable;
    const x = try allocator.dupe(f32, data[0..mlx.mlx_array_size(flat)]);
    errdefer allocator.free(x);
    return .{ .x = x, .ids = try allocator.dupe(i32, rows.ids), .topk = rows.topk };
}

fn modelsPath(buf: []u8, name: []const u8) ![]const u8 {
    if (std.c.getenv("SUSHI_ENCODER_REAL_MODEL") == null) return error.SkipZigTest;
    const root = std.c.getenv("SUSHI_MODELS_DIR") orelse return error.SkipZigTest;
    const text = std.mem.span(root);
    if (text.len == 0 or !std.fs.path.isAbsolute(text)) return error.SkipZigTest;
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ text, name }) catch error.SkipZigTest;
}

fn batchTokensOf(allocator: std.mem.Allocator, windows: *const calib.Windows, b: trunk_mod.Batch) mlx.mlx_array {
    const ids = windows.ids[windows.starts[b.first]..windows.starts[b.first + b.count]];
    const as_i32 = allocator.alloc(i32, ids.len) catch unreachable;
    defer allocator.free(as_i32);
    for (ids, as_i32) |id, *o| o.* = @intCast(id);
    return mlx.mlx_array_new_data(as_i32.ptr, &[_]c_int{ @intCast(b.count), @intCast(b.len) }, 2, .int32);
}

/// The first `len` bytes of `arr` as contiguous bf16.
fn bf16Bytes(allocator: std.mem.Allocator, arr: mlx.mlx_array, len: usize) ![]u8 {
    const s = mlx.gpuStream();
    var cast = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cast);
    try mlx.check(mlx.mlx_astype(&cast, arr, .bfloat16, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_contiguous(&flat, cast, false, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    if (mlx.mlx_array_size(flat) * 2 < len) return error.BoundaryRowsMismatch;
    return allocator.dupe(u8, mlx.mlx_array_data_uint8(flat).?[0..len]);
}

fn windowTokensOf(allocator: std.mem.Allocator, windows: *const calib.Windows, i: usize) mlx.mlx_array {
    const ids = windows.window(i);
    const as_i32 = allocator.alloc(i32, ids.len) catch unreachable;
    defer allocator.free(as_i32);
    for (ids, as_i32) |id, *o| o.* = @intCast(id);
    return mlx.mlx_array_new_data(as_i32.ptr, &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32);
}

fn freeRowsTrace(allocator: std.mem.Allocator, list: *std.ArrayList(RowsTrace)) void {
    for (list.items) |r| {
        allocator.free(r.x);
        allocator.free(r.ids);
    }
    list.deinit(allocator);
}

fn fullForwardRows(allocator: std.mem.Allocator, io: std.Io, pack_abs: []const u8, windows: *const calib.Windows, layers: usize, out: *std.ArrayList(RowsTrace)) !void {
    imatrix.rows_request = 0;
    defer imatrix.rows_request = null;
    var full = try kld.loadModel(io, allocator, .{ .model_dir = pack_abs, .ssd_budget_bytes = 4 << 30 });
    defer full.deinit();
    const tap = full.xfm.imatrix orelse return error.CaptureTapNotArmed;
    for (0..layers) |layer| {
        tap.armRows(layer);
        for (0..windows.count) |w| {
            try full.xfm.resetCache();
            full.xfm.bindOwnRows(calib.window_len);
            var ctx = full.xfm.defaultCtx();
            ctx.skip_lm_head = true;
            const tokens = windowTokensOf(allocator, windows, w);
            defer _ = mlx.mlx_array_free(tokens);
            const result = try full.xfm.forwardWith(&ctx, tokens);
            defer _ = mlx.mlx_array_free(result);
            try mlx.check(mlx.mlx_array_eval(result));
        }
        const rows = try tap.takeRows();
        defer {
            _ = mlx.mlx_array_free(rows.x);
            allocator.free(rows.ids);
        }
        try out.append(allocator, try copyRows(allocator, mlx.gpuStream(), rows));
    }
}

const rows_rel_tolerance: f32 = 1e-3;
const hessian_rel_tolerance: f64 = 1e-3;
const ids_agreement_floor: f64 = 1.0;

fn rowsAgree(want: RowsTrace, got: RowsTrace) !void {
    const t = std.testing;
    try t.expectEqual(want.topk, got.topk);
    try t.expectEqual(want.x.len, got.x.len);
    try t.expectEqual(want.ids.len, got.ids.len);
    var same: usize = 0;
    for (want.ids, got.ids) |a, b| same += @intFromBool(a == b);
    try t.expect(@as(f64, @floatFromInt(same)) >= ids_agreement_floor * @as(f64, @floatFromInt(want.ids.len)));
    var peak: f32 = 0;
    var worst: f32 = 0;
    for (want.x, got.x) |a, b| {
        peak = @max(peak, @abs(a));
        worst = @max(worst, @abs(a - b));
    }
    try t.expect(worst <= rows_rel_tolerance * peak);
}

fn hessianRelDiff(allocator: std.mem.Allocator, s: mlx.mlx_stream, src: *const source.SafetensorsSource, layer: u32, geom: convert.Geometry, want: RowsTrace, got: RowsTrace, expert: usize, down: bool) !f64 {
    const n = if (down) geom.inter else geom.hidden;
    var sides: [2][]f32 = undefined;
    for (&sides, [2]RowsTrace{ want, got }) |*side, r| {
        const rows_n = r.x.len / geom.hidden;
        const x = mlx.mlx_array_new_data(r.x.ptr, &[_]c_int{ @intCast(rows_n), @intCast(geom.hidden) }, 2, .float32);
        defer _ = mlx.mlx_array_free(x);
        var rh = try hessians.RowsHessians.init(allocator, s, src, layer, false, geom, x, r.ids, r.topk);
        defer rh.deinit();
        const hs = rh.hessianSource();
        side.* = try allocator.alloc(f32, n * n);
        errdefer allocator.free(side.*);
        const rows_seen = if (down) try hs.down(hs.ctx, expert, side.*) else try hs.gate_up(hs.ctx, expert, side.*);
        if (rows_seen == 0) return error.ExpertSawNoRows;
    }
    defer for (sides) |side| allocator.free(side);
    var num: f64 = 0;
    var den: f64 = 0;
    for (sides[0], sides[1]) |a, b| {
        const d = @as(f64, a) - b;
        num += d * d;
        den += @as(f64, a) * a;
    }
    return @sqrt(num / den);
}

test "capture: batches forwarded strided-first map back to plan order" {
    const order = try Capture.stridedFirstOrder(std.testing.allocator, 6);
    defer std.testing.allocator.free(order);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 3, 4, 1, 5 }, order);
}

test "capture (real Qwen checkpoint, SUSHI_ENCODER_REAL_MODEL=1 and SUSHI_MODELS_DIR): the layer-streamed capture of a real four-layer slice reproduces the full forward's hidden boundaries, MLP input rows, routing and Hessians" {
    const t = std.testing;
    const io = t.io;
    const slice = @import("slice.zig");
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const src_path = try modelsPath(&path_buf, "Qwen/Qwen3.8-Flash-Next");
    var real = std.Io.Dir.cwd().openDir(io, src_path, .{ .iterate = true }) catch return error.SkipZigTest;
    defer real.close(io);
    host.log.enableStderr();
    var real_src = try source.SafetensorsSource.open(t.allocator, io, real);
    defer real_src.deinit();

    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "slice");
    var slice_dir = try work.dir.openDir(io, "slice", .{ .iterate = true });
    defer slice_dir.close(io);
    const layers = 4;
    try slice.sliceQwen(t.allocator, io, real, &real_src, slice_dir, .{ .layers = layers, .experts = 16 });
    var slice_src = try source.SafetensorsSource.open(t.allocator, io, slice_dir);
    defer slice_src.deinit();
    var abs_buf: [std.fs.max_path_bytes]u8 = undefined;
    const work_abs = abs_buf[0..try work.dir.realPath(io, &abs_buf)];
    const slice_abs = try std.fmt.allocPrint(t.allocator, "{s}/slice", .{work_abs});
    defer t.allocator.free(slice_abs);
    const pack_abs = try std.fmt.allocPrint(t.allocator, "{s}/pack", .{work_abs});
    defer t.allocator.free(pack_abs);

    var tokenizer = try host.tokenizer.loadTokenizerAny(io, t.allocator, src_path);
    defer tokenizer.deinit();
    const code_ids = try calib.tokenizeText(t.allocator, &tokenizer, calib.code_text);
    defer t.allocator.free(code_ids);
    var windows = try calib.fromTokens(t.allocator, code_ids[0 .. calib.window_len * 2], calib.window_len * 2);
    defer windows.deinit();

    var trace = std.ArrayList([]u8).empty;
    defer {
        for (trace.items) |b| t.allocator.free(b);
        trace.deinit(t.allocator);
    }
    var streamed = std.ArrayList(RowsTrace).empty;
    defer freeRowsTrace(t.allocator, &streamed);
    var cap = Capture.init(t.allocator, io, mlx.gpuStream(), &windows, &slice_src, 4 << 30);
    defer cap.deinit();
    cap.trace = &trace;
    cap.rows_trace = &streamed;
    const summary = try driver.run(t.allocator, io, mlx.gpuStream(), .{
        .model_dir = slice_abs,
        .out_dir = pack_abs,
        .rate = .{ .n = 24 },
        .dec = .{ .codebook = .mcg, .window = .w12 },
        .calib_sha256 = &windows.sha256,
        .calib_tokens = windows.tokens(),
        .version = "sushi-encoder-test",
        .trunk = .rtn,
    }, cap.calibrator(), null);
    try t.expectEqual(@as(usize, layers), summary.layers_written);
    try t.expectEqual(@as(usize, layers - 1), trace.items.len);
    try t.expectEqual(@as(usize, layers), streamed.items.len);

    var full = try kld.loadModel(io, t.allocator, .{ .model_dir = pack_abs });
    defer full.deinit();
    try full.xfm.resetCache();
    full.xfm.bindOwnRows(calib.window_len);
    var ctx = full.xfm.defaultCtx();
    ctx.skip_lm_head = true;
    const layer_ids = [_]u32{ 0, 1, 2, 3 };
    var outs: [layers]mlx.mlx_array = undefined;
    for (&outs) |*o| o.* = mlx.mlx_array_new();
    defer for (outs) |o| {
        _ = mlx.mlx_array_free(o);
    };
    var cl = host.model_handle.CaptureLayers{ .ids = &layer_ids, .out = &outs };
    ctx.capture_layers = &cl;
    const tokens = windowTokensOf(t.allocator, &windows, 0);
    defer _ = mlx.mlx_array_free(tokens);
    const result = try full.xfm.forwardWith(&ctx, tokens);
    defer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_array_eval(result));
    for (trace.items, 0..) |want, i| {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_contiguous(&flat, outs[i], false, mlx.gpuStream()));
        try mlx.check(mlx.mlx_array_eval(flat));
        const got = mlx.mlx_array_data_uint8(flat).?;
        try t.expectEqualSlices(u8, want, got[0..want.len]);
    }

    var reference = std.ArrayList(RowsTrace).empty;
    defer freeRowsTrace(t.allocator, &reference);
    try fullForwardRows(t.allocator, io, pack_abs, &windows, layers, &reference);
    const geom = convert.Geometry{ .experts = 16, .hidden = 2560, .inter = 640 };
    for (streamed.items, reference.items, 0..) |want, got, layer| {
        try rowsAgree(want, got);
        for ([_]usize{ 0, 5, 11 }) |e| {
            try t.expect(try hessianRelDiff(t.allocator, mlx.gpuStream(), &slice_src, @intCast(layer), geom, want, got, e, false) <= hessian_rel_tolerance);
            try t.expect(try hessianRelDiff(t.allocator, mlx.gpuStream(), &slice_src, @intCast(layer), geom, want, got, e, true) <= hessian_rel_tolerance);
        }
    }
}

test "capture (real Qwen checkpoint, SUSHI_ENCODER_REAL_MODEL=1 and SUSHI_MODELS_DIR): the BF16 path's rows at the PLE layer read the source's n-gram shards, not the pack's table" {
    const t = std.testing;
    const io = t.io;
    const slice = @import("slice.zig");
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const src_path = try modelsPath(&path_buf, "Qwen/Qwen3.8-Flash-Next");
    var real = std.Io.Dir.cwd().openDir(io, src_path, .{ .iterate = true }) catch return error.SkipZigTest;
    defer real.close(io);
    var real_src = try source.SafetensorsSource.open(t.allocator, io, real);
    defer real_src.deinit();

    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "slice");
    var slice_dir = try work.dir.openDir(io, "slice", .{ .iterate = true });
    defer slice_dir.close(io);
    const layers = 2;
    try slice.sliceQwen(t.allocator, io, real, &real_src, slice_dir, .{ .layers = layers, .experts = 16 });
    var slice_src = try source.SafetensorsSource.open(t.allocator, io, slice_dir);
    defer slice_src.deinit();
    var abs_buf: [std.fs.max_path_bytes]u8 = undefined;
    const work_abs = abs_buf[0..try work.dir.realPath(io, &abs_buf)];
    const slice_abs = try std.fmt.allocPrint(t.allocator, "{s}/slice", .{work_abs});
    defer t.allocator.free(slice_abs);

    var tokenizer = try host.tokenizer.loadTokenizerAny(io, t.allocator, src_path);
    defer tokenizer.deinit();
    const code_ids = try calib.tokenizeText(t.allocator, &tokenizer, calib.code_text);
    defer t.allocator.free(code_ids);
    var windows = try calib.fromTokens(t.allocator, code_ids[0 .. calib.window_len * 2], calib.window_len * 2);
    defer windows.deinit();

    // Two served packs of the slice that differ only in their n-gram table's codes.
    const names = [_][]const u8{ "served_a", "served_b" };
    for (names) |name| {
        const served_abs = try std.fmt.allocPrint(t.allocator, "{s}/{s}", .{ work_abs, name });
        defer t.allocator.free(served_abs);
        _ = try driver.run(t.allocator, io, mlx.gpuStream(), .{
            .model_dir = slice_abs,
            .out_dir = served_abs,
            .rate = .{ .n = 24 },
            .dec = .{ .codebook = .mcg, .window = .w12 },
            .calib_sha256 = &windows.sha256,
            .calib_tokens = windows.tokens(),
            .version = "sushi-encoder-test",
            .trunk = .rtn,
        }, null, null);
    }
    {
        var served_b = try work.dir.openDir(io, "served_b", .{});
        defer served_b.close(io);
        const table = try served_b.readFileAlloc(io, "ngram_table.bin", t.allocator, .limited(1 << 30));
        defer t.allocator.free(table);
        const header_len = std.mem.readInt(u64, table[0..8], .little);
        const data = table[8 + header_len ..];
        for (data[0 .. data.len / 10 * 7]) |*b| b.* ^= 0x11;
        try served_b.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = table });
    }

    var rows: [2]std.ArrayList(RowsTrace) = .{ .empty, .empty };
    var target: [2]std.ArrayList(RowsTrace) = .{ .empty, .empty };
    defer for (&rows) |*r| freeRowsTrace(t.allocator, r);
    defer for (&target) |*r| freeRowsTrace(t.allocator, r);
    for (names, 0..) |name, arm| {
        const served_abs = try std.fmt.allocPrint(t.allocator, "{s}/{s}", .{ work_abs, name });
        defer t.allocator.free(served_abs);
        const pack_abs = try std.fmt.allocPrint(t.allocator, "{s}/pack_{d}", .{ work_abs, arm });
        defer t.allocator.free(pack_abs);
        var cap = Capture.init(t.allocator, io, mlx.gpuStream(), &windows, &slice_src, 4 << 30);
        defer cap.deinit();
        cap.rows_trace = &rows[arm];
        cap.rows_f_trace = &target[arm];
        _ = try driver.run(t.allocator, io, mlx.gpuStream(), .{
            .model_dir = slice_abs,
            .out_dir = pack_abs,
            .rate = .{ .n = 24 },
            .dec = .{ .codebook = .mcg, .window = .w12 },
            .trunk_from = served_abs,
            .trunk = .stored,
            .target_in_run = true,
            .calib_sha256 = &windows.sha256,
            .calib_tokens = windows.tokens(),
            .version = "sushi-encoder-test",
            .min_rows = 1,
            .batch = 4,
        }, cap.calibrator(), null);
    }
    for (0..layers) |layer| {
        try t.expectEqualSlices(f32, target[0].items[layer].x, target[1].items[layer].x);
    }
    try t.expectEqualSlices(f32, rows[0].items[0].x, rows[1].items[0].x);
    // The quantized path does read the pack's table, so the two arms are not the same run.
    try t.expect(!std.mem.eql(f32, rows[0].items[1].x, rows[1].items[1].x));
}

/// Relative RMS of each hyper-connection stream of `got` from `want`, both bf16 rows of `hc` x `hidden` per token.
fn streamRelDiff(want: []const u8, got: []const u8, hc: usize, hidden: usize, out: []f64) void {
    const a: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, want));
    const b: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, got));
    for (out, 0..) |*o, st| {
        var num: f64 = 0;
        var den: f64 = 0;
        var tok: usize = 0;
        while (tok * hc * hidden < a.len) : (tok += 1) {
            for (a[(tok * hc + st) * hidden ..][0..hidden], b[(tok * hc + st) * hidden ..][0..hidden]) |x, y| {
                const fx: f64 = @as(f32, @bitCast(@as(u32, x) << 16));
                const fy: f64 = @as(f32, @bitCast(@as(u32, y) << 16));
                num += (fy - fx) * (fy - fx);
                den += fx * fx;
            }
        }
        o.* = @sqrt(num / den);
    }
}

/// Per stream, against the BF16 slice's own layer outputs. One layer from the reference's own input must sit within
/// the kernels' gap (measured up to 0.006; a wrong expert order or n-gram table moves it 0.05-0.3). Chained over
/// layers, that gap compounds as the slice's own forward does under the conversion's batch shape, so a chained
/// boundary may exceed the slice's batched-against-alone gap only by a little.
const bf16_one_layer_rel_tolerance = 0.01;
const bf16_chained_excess_tolerance = 0.003;

test "capture (real Qwen checkpoint, SUSHI_ENCODER_REAL_MODEL=1 and SUSHI_MODELS_DIR): the in-run BF16 boundaries are the BF16 slice's own layer outputs, every stream, under a reordered router" {
    const t = std.testing;
    const io = t.io;
    const slice = @import("slice.zig");
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const src_path = try modelsPath(&path_buf, "Qwen/Qwen3.8-Flash-Next");
    var real = std.Io.Dir.cwd().openDir(io, src_path, .{ .iterate = true }) catch return error.SkipZigTest;
    defer real.close(io);
    var real_src = try source.SafetensorsSource.open(t.allocator, io, real);
    defer real_src.deinit();

    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "slice");
    var slice_dir = try work.dir.openDir(io, "slice", .{ .iterate = true });
    defer slice_dir.close(io);
    const layers = 4;
    try slice.sliceQwen(t.allocator, io, real, &real_src, slice_dir, .{ .layers = layers, .experts = 16 });
    var slice_src = try source.SafetensorsSource.open(t.allocator, io, slice_dir);
    defer slice_src.deinit();
    var abs_buf: [std.fs.max_path_bytes]u8 = undefined;
    const work_abs = abs_buf[0..try work.dir.realPath(io, &abs_buf)];
    const slice_abs = try std.fmt.allocPrint(t.allocator, "{s}/slice", .{work_abs});
    defer t.allocator.free(slice_abs);
    const pack_abs = try std.fmt.allocPrint(t.allocator, "{s}/pack", .{work_abs});
    defer t.allocator.free(pack_abs);

    var tokenizer = try host.tokenizer.loadTokenizerAny(io, t.allocator, src_path);
    defer tokenizer.deinit();
    const code_ids = try calib.tokenizeText(t.allocator, &tokenizer, calib.code_text);
    defer t.allocator.free(code_ids);
    var windows = try calib.fromTokens(t.allocator, code_ids[0 .. calib.window_len * 2], calib.window_len * 2);
    defer windows.deinit();

    var bf16 = std.ArrayList([]u8).empty;
    defer {
        for (bf16.items) |b| t.allocator.free(b);
        bf16.deinit(t.allocator);
    }
    // A served pack of the slice, then a planned conversion over its trunk, as a real run: the plan's rate groups
    // reorder every layer's router, and the PLE layer's n-gram table is the served pack's 4-bit one.
    const served_abs = try std.fmt.allocPrint(t.allocator, "{s}/served", .{work_abs});
    defer t.allocator.free(served_abs);
    _ = try driver.run(t.allocator, io, mlx.gpuStream(), .{
        .model_dir = slice_abs,
        .out_dir = served_abs,
        .rate = .{ .n = 24 },
        .dec = .{ .codebook = .mcg, .window = .w12 },
        .calib_sha256 = &windows.sha256,
        .calib_tokens = windows.tokens(),
        .version = "sushi-encoder-test",
        .trunk = .rtn,
    }, null, null);
    const row = "{\"gate_up\":[2,1.5,2,1.5,2,1.5,2,1.5,2,1.5,2,1.5,2,1.5,2,1.5],\"down\":[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]}";
    var plan = try rate_plan.parse(t.allocator, "{\"format\":\"exls-rate-plan-1\",\"experts\":16,\"layers\":{\"0\":" ++ row ++ ",\"1\":" ++ row ++ ",\"2\":" ++ row ++ ",\"3\":" ++ row ++ "}}", 16, layers);
    defer plan.deinit();

    // The slice's own layer outputs for window 0: run alone (the reference) and batched beside window 1, the shape
    // the conversion forwards (how far batching alone moves a BF16 forward).
    const width = 4 * 2560;
    const w0 = (windows.starts[1] - windows.starts[0]) * width * 2;
    var ref: [2][layers - 1][]u8 = undefined;
    {
        var full = try kld.loadModel(io, t.allocator, .{ .model_dir = slice_abs, .ssd_budget_bytes = 4 << 30 });
        defer full.deinit();
        for (0..2) |arm| {
            try full.xfm.resetCache();
            var ctx = full.xfm.defaultCtx();
            ctx.skip_lm_head = true;
            const layer_ids = [_]u32{ 0, 1, 2 };
            var outs: [layers - 1]mlx.mlx_array = undefined;
            for (&outs) |*o| o.* = mlx.mlx_array_new();
            defer for (outs) |o| {
                _ = mlx.mlx_array_free(o);
            };
            var cl = host.model_handle.CaptureLayers{ .ids = &layer_ids, .out = &outs };
            ctx.capture_layers = &cl;
            const tokens = if (arm == 0) windowTokensOf(t.allocator, &windows, 0) else batchTokensOf(t.allocator, &windows, .{ .first = 0, .count = 2, .len = calib.window_len });
            defer _ = mlx.mlx_array_free(tokens);
            const result = try full.xfm.forwardWith(&ctx, tokens);
            defer _ = mlx.mlx_array_free(result);
            try mlx.check(mlx.mlx_array_eval(result));
            for (outs, 0..) |o, i| ref[arm][i] = try bf16Bytes(t.allocator, o, w0);
        }
    }
    defer for (ref) |arm| for (arm) |b| t.allocator.free(b);

    var iso = Isolate{ .inputs = ref[0][0 .. layers - 2] };
    defer {
        for (iso.outputs.items) |b| t.allocator.free(b);
        iso.outputs.deinit(t.allocator);
    }
    var cap = Capture.init(t.allocator, io, mlx.gpuStream(), &windows, &slice_src, 4 << 30);
    defer cap.deinit();
    cap.bf16_trace = &bf16;
    cap.bf16_isolate = &iso;
    _ = try driver.run(t.allocator, io, mlx.gpuStream(), .{
        .model_dir = slice_abs,
        .out_dir = pack_abs,
        .rate = .{ .n = 24 },
        .dec = .{ .codebook = .std1, .window = .w15, .table = @import("sushi_exl3").codebook_tables.indexOf("std1") },
        .plan = &plan,
        .trunk_from = served_abs,
        .calib_sha256 = &windows.sha256,
        .calib_tokens = windows.tokens(),
        .version = "sushi-encoder-test",
        .min_rows = 1,
        .target_in_run = true,
        // The target rounds a layer's gate/up banks ahead of its down banks: three or more of each.
        .batch = 4,
    }, cap.calibrator(), null);
    try t.expectEqual(@as(usize, layers - 1), bf16.items.len);
    try t.expectEqual(@as(usize, layers - 1), iso.outputs.items.len);

    var chained: [layers - 1][4]f64 = undefined;
    var isolated: [layers - 1][4]f64 = undefined;
    var batched: [layers - 1][4]f64 = undefined;
    for (0..layers - 1) |i| {
        streamRelDiff(ref[0][i], bf16.items[i][0..w0], 4, 2560, &chained[i]);
        streamRelDiff(ref[0][i], iso.outputs.items[i], 4, 2560, &isolated[i]);
        streamRelDiff(ref[0][i], ref[1][i], 4, 2560, &batched[i]);
        if (std.c.getenv("SUSHI_ENCODER_BF16_DIFF") != null) {
            const c = chained[i];
            const o = isolated[i];
            const b = batched[i];
            std.debug.print("boundary {d}: chained {d:.5} {d:.5} {d:.5} {d:.5} | one layer {d:.5} {d:.5} {d:.5} {d:.5} | slice batched {d:.5} {d:.5} {d:.5} {d:.5}\n", .{ i + 1, c[0], c[1], c[2], c[3], o[0], o[1], o[2], o[3], b[0], b[1], b[2], b[3] });
        }
    }
    for (isolated) |rel| for (rel) |r| try t.expect(r <= bf16_one_layer_rel_tolerance);
    for (chained, batched) |c, b| for (c, b) |rc, rb| try t.expect(rc <= rb + bf16_chained_excess_tolerance);
}

test "capture (real Qwen checkpoint, SUSHI_ENCODER_REAL_MODEL=1 and SUSHI_MODELS_DIR): a re-rounded trunk writes every layer's and the head's linears, the pack it makes reproduces the streamed boundaries, and the taps see exactly the forward's inputs without changing it" {
    const t = std.testing;
    const io = t.io;
    const slice = @import("slice.zig");
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const src_path = try modelsPath(&path_buf, "Qwen/Qwen3.8-Flash-Next");
    var real = std.Io.Dir.cwd().openDir(io, src_path, .{ .iterate = true }) catch return error.SkipZigTest;
    defer real.close(io);
    var real_src = try source.SafetensorsSource.open(t.allocator, io, real);
    defer real_src.deinit();
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "slice");
    var slice_dir = try work.dir.openDir(io, "slice", .{ .iterate = true });
    defer slice_dir.close(io);
    const layers = 4;
    try slice.sliceQwen(t.allocator, io, real, &real_src, slice_dir, .{ .layers = layers, .experts = 16 });
    var slice_src = try source.SafetensorsSource.open(t.allocator, io, slice_dir);
    defer slice_src.deinit();
    var abs_buf: [std.fs.max_path_bytes]u8 = undefined;
    const work_abs = abs_buf[0..try work.dir.realPath(io, &abs_buf)];
    const slice_abs = try std.fmt.allocPrint(t.allocator, "{s}/slice", .{work_abs});
    defer t.allocator.free(slice_abs);
    const pack_abs = try std.fmt.allocPrint(t.allocator, "{s}/pack", .{work_abs});
    defer t.allocator.free(pack_abs);
    var tokenizer = try host.tokenizer.loadTokenizerAny(io, t.allocator, src_path);
    defer tokenizer.deinit();
    const code_ids = try calib.tokenizeText(t.allocator, &tokenizer, calib.code_text);
    defer t.allocator.free(code_ids);
    var windows = try calib.fromTokens(t.allocator, code_ids[0 .. calib.window_len * 2], calib.window_len * 2);
    defer windows.deinit();

    var trace = std.ArrayList([]u8).empty;
    defer {
        for (trace.items) |b| t.allocator.free(b);
        trace.deinit(t.allocator);
    }
    var errors = std.ArrayList(TrunkError).empty;
    defer {
        for (errors.items) |e| t.allocator.free(e.name);
        errors.deinit(t.allocator);
    }
    var cap = Capture.init(t.allocator, io, mlx.gpuStream(), &windows, &slice_src, 4 << 30);
    defer cap.deinit();
    cap.trace = &trace;
    cap.trunk_errors = &errors;
    const summary = try driver.run(t.allocator, io, mlx.gpuStream(), .{
        .model_dir = slice_abs,
        .out_dir = pack_abs,
        .rate = .{ .n = 24 },
        .dec = .{ .codebook = .mcg, .window = .w12 },
        .calib_sha256 = &windows.sha256,
        .calib_tokens = windows.tokens(),
        .version = "sushi-encoder-test",
        .min_rows = 1,
    }, cap.calibrator(), null);
    try t.expectEqual(@as(usize, layers), summary.layers_written);
    // Layers 0-2 are linear attention (layer 1 carries the PLE pair), layer 3 full attention; then the head.
    try t.expectEqual(@as(usize, 12 + 14 + 12 + 12 + 3), errors.items.len);
    for (errors.items) |e| try t.expect(e.rel_err > 0 and e.rel_err < 1e-3);

    var full = try kld.loadModel(io, t.allocator, .{ .model_dir = pack_abs });
    defer full.deinit();
    const tokens = windowTokensOf(t.allocator, &windows, 0);
    defer _ = mlx.mlx_array_free(tokens);
    // Both windows fit one 8192-token batch, the shape the capture ran them in.
    const both = batchTokensOf(t.allocator, &windows, .{ .first = 0, .count = 2, .len = calib.window_len });
    defer _ = mlx.mlx_array_free(both);
    const layer_ids = [_]u32{ 0, 1, 2, 3 };
    var outs: [layers]mlx.mlx_array = undefined;
    for (&outs) |*o| o.* = mlx.mlx_array_new();
    defer for (outs) |o| {
        _ = mlx.mlx_array_free(o);
    };
    {
        try full.xfm.resetCache();
        var ctx = full.xfm.defaultCtx();
        ctx.skip_lm_head = true;
        var cl = host.model_handle.CaptureLayers{ .ids = &layer_ids, .out = &outs };
        ctx.capture_layers = &cl;
        const result = try full.xfm.forwardWith(&ctx, both);
        defer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_array_eval(result));
    }
    try t.expectEqual(@as(usize, layers - 1), trace.items.len);
    for (trace.items, 0..) |want, i| {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_contiguous(&flat, outs[i], false, mlx.gpuStream()));
        try mlx.check(mlx.mlx_array_eval(flat));
        try t.expectEqualSlices(u8, want, mlx.mlx_array_data_uint8(flat).?[0..want.len]);
    }

    imatrix.rows_request = 0;
    defer imatrix.rows_request = null;
    var tapped = try kld.loadModel(io, t.allocator, .{ .model_dir = pack_abs, .ssd_budget_bytes = 4 << 30 });
    defer tapped.deinit();
    var grams = trunk_mod.Grams.init(t.allocator, mlx.gpuStream());
    defer grams.deinit();
    try watchTrunk(&grams, &tapped.xfm, false);
    grams.want = std.EnumSet(trunk_mod.Group).full;
    var mixed: [2][]u8 = undefined;
    for (&mixed, [_]bool{ false, true }) |*o, arm| {
        try tapped.xfm.resetCache();
        var ctx = tapped.xfm.defaultCtx();
        ctx.skip_lm_head = true;
        if (arm) grams.arm();
        defer grams.disarm();
        const result = try tapped.xfm.forwardWith(&ctx, tokens);
        defer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_array_eval(result));
        if (arm) try grams.settle();
        o.* = try t.allocator.dupe(u8, mlx.mlx_array_data_uint8(result).?[0 .. mlx.mlx_array_size(result) * mlx.mlx_array_itemsize(result)]);
    }
    defer for (mixed) |m| t.allocator.free(m);
    try t.expectEqualSlices(u8, mixed[0], mixed[1]);
    try grams.expectRows(calib.window_len);
    const rows = try tapped.xfm.imatrix.?.takeRows();
    defer {
        _ = mlx.mlx_array_free(rows.x);
        t.allocator.free(rows.ids);
    }
    var once = trunk_mod.Grams.init(t.allocator, mlx.gpuStream());
    defer once.deinit();
    var first = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(first);
    try mlx.check(mlx.mlx_slice(&first, rows.x, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ calib.window_len, 2560 }, 2, &[_]c_int{ 1, 1 }, 2, mlx.gpuStream()));
    try once.add(.moe_in, first);
    try once.settle();
    const want = mlx.mlx_array_data_float32(once.acc.get(.moe_in)).?;
    const got = mlx.mlx_array_data_float32(grams.acc.get(.moe_in)).?;
    for (0..2560) |i| try t.expectApproxEqRel(want[i * 2560 + i], got[i * 2560 + i], 1e-4);
}

/// `pack_abs` with every STQ bank replaced by its dense twin (the bank's weight as the kernels apply it, BF16, under
/// `<base>.weight`), in `twin_abs`: what the engine's own Qwen forward can run without the trellis kernels.
fn writeDenseTwin(allocator: std.mem.Allocator, io: std.Io, pack_abs: []const u8, twin_abs: []const u8, dec: format.Decode) !usize {
    const cwd = std.Io.Dir.cwd();
    var from = try cwd.openDir(io, pack_abs, .{ .iterate = true });
    defer from.close(io);
    try io_util.makeDirPath(io, twin_abs);
    var to = try cwd.openDir(io, twin_abs, .{ .iterate = true });
    defer to.close(io);
    var dense_builder = pack.ShardBuilder.init(allocator);
    defer dense_builder.deinit();
    var gone = std.StringHashMap(void).init(allocator);
    defer {
        var it = gone.keyIterator();
        while (it.next()) |k| allocator.free(k.*);
        gone.deinit();
    }
    var banks: usize = 0;
    var it = from.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file or e.name[0] == '.') continue;
        if (!std.mem.startsWith(u8, e.name, "trunk-stq-")) {
            if (std.mem.endsWith(u8, e.name, ".safetensors") or std.mem.eql(u8, e.name, "ngram_table.bin")) {
                var src_buf: [std.fs.max_path_bytes]u8 = undefined;
                try to.symLink(io, try std.fmt.bufPrint(&src_buf, "{s}/{s}", .{ pack_abs, e.name }), e.name, .{});
            } else if (!std.mem.eql(u8, e.name, "model.safetensors.index.json")) {
                try from.copyFile(e.name, to, e.name, io, .{});
            }
            continue;
        }
        const raw = try from.readFileAlloc(io, e.name, allocator, .unlimited);
        defer allocator.free(raw);
        var file = try sft.parse(allocator, raw);
        defer file.deinit(allocator);
        var names = file.tensors.keyIterator();
        while (names.next()) |name| {
            try gone.put(try allocator.dupe(u8, name.*), {});
            if (!std.mem.endsWith(u8, name.*, ".trellis")) continue;
            const base = name.*[0 .. name.*.len - ".trellis".len];
            var nb: [240]u8 = undefined;
            const tr = file.tensors.get(name.*).?;
            const su = file.tensors.get(try std.fmt.bufPrint(&nb, "{s}.suh", .{base})).?;
            const sv = file.tensors.get(try std.fmt.bufPrint(&nb, "{s}.svh", .{base})).?;
            const in = tr.shape[0] * 16;
            const out = tr.shape[1] * 16;
            const bank = trunk_mod.Bank{ .trellis = @constCast(sft.slice(u16, tr)), .suh = @constCast(sft.slice(u16, su)), .svh = @constCast(sft.slice(u16, sv)), .rate = .{ .n = @intCast(tr.shape[2]) } };
            const dense = try trunk_mod.denseOfBank(allocator, bank, dec, out, in);
            defer allocator.free(dense);
            try dense_builder.add(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}), "BF16", &.{ out, in }, std.mem.sliceAsBytes(dense));
            banks += 1;
        }
    }
    try dense_builder.flush(io, to, "trunk-dense.safetensors", &.{.{ "format", "mlx" }});

    const index_raw = try from.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(64 << 20));
    defer allocator.free(index_raw);
    const index = try std.json.parseFromSlice(std.json.Value, allocator, index_raw, .{});
    defer index.deinit();
    var entries = std.ArrayList(pack.IndexEntry).empty;
    defer {
        for (entries.items) |en| {
            allocator.free(en.tensor);
            allocator.free(en.file);
        }
        entries.deinit(allocator);
    }
    var wit = index.value.object.get("weight_map").?.object.iterator();
    while (wit.next()) |w| {
        if (gone.contains(w.key_ptr.*)) continue;
        try entries.append(allocator, .{ .tensor = try allocator.dupe(u8, w.key_ptr.*), .file = try allocator.dupe(u8, w.value_ptr.string) });
    }
    var dense_file = try to.openFile(io, "trunk-dense.safetensors", .{});
    defer dense_file.close(io);
    var dh = try trunk_from.readHeader(allocator, io, to, "trunk-dense.safetensors");
    defer dh.parsed.deinit();
    var dit = dh.parsed.value.object.iterator();
    while (dit.next()) |d| {
        if (std.mem.eql(u8, d.key_ptr.*, "__metadata__")) continue;
        try entries.append(allocator, .{ .tensor = try allocator.dupe(u8, d.key_ptr.*), .file = try allocator.dupe(u8, "trunk-dense.safetensors") });
    }
    const text = try pack.buildIndex(allocator, entries.items, 0);
    defer allocator.free(text);
    try to.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = text });
    return banks;
}

test "capture (real Qwen checkpoint, SUSHI_ENCODER_REAL_MODEL=1 and SUSHI_MODELS_DIR): an STQ trunk keeps the pack's experts byte for byte, writes banks whose dense twin reproduces the streamed boundaries, and keeps what the trellis cannot hold" {
    const t = std.testing;
    const io = t.io;
    const slice = @import("slice.zig");
    if (std.c.getenv("SUSHI_ENCODER_TEST_LOG") != null) host.log.enableStderr();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const src_path = try modelsPath(&path_buf, "Qwen/Qwen3.8-Flash-Next");
    var real = std.Io.Dir.cwd().openDir(io, src_path, .{ .iterate = true }) catch return error.SkipZigTest;
    defer real.close(io);
    var real_src = try source.SafetensorsSource.open(t.allocator, io, real);
    defer real_src.deinit();
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "slice");
    var slice_dir = try work.dir.openDir(io, "slice", .{ .iterate = true });
    defer slice_dir.close(io);
    const layers = 4;
    try slice.sliceQwen(t.allocator, io, real, &real_src, slice_dir, .{ .layers = layers, .experts = 16 });
    var slice_src = try source.SafetensorsSource.open(t.allocator, io, slice_dir);
    defer slice_src.deinit();
    var abs_buf: [std.fs.max_path_bytes]u8 = undefined;
    const work_abs = abs_buf[0..try work.dir.realPath(io, &abs_buf)];
    const slice_abs = try std.fmt.allocPrint(t.allocator, "{s}/slice", .{work_abs});
    defer t.allocator.free(slice_abs);
    const base_abs = try std.fmt.allocPrint(t.allocator, "{s}/base", .{work_abs});
    defer t.allocator.free(base_abs);
    const pack_abs = try std.fmt.allocPrint(t.allocator, "{s}/pack", .{work_abs});
    defer t.allocator.free(pack_abs);
    const twin_abs = try std.fmt.allocPrint(t.allocator, "{s}/twin", .{work_abs});
    defer t.allocator.free(twin_abs);
    var tokenizer = try host.tokenizer.loadTokenizerAny(io, t.allocator, src_path);
    defer tokenizer.deinit();
    const code_ids = try calib.tokenizeText(t.allocator, &tokenizer, calib.code_text);
    defer t.allocator.free(code_ids);
    var windows = try calib.fromTokens(t.allocator, code_ids[0 .. calib.window_len * 2], calib.window_len * 2);
    defer windows.deinit();

    // The base: a pack with the GPTQ-rounded affine trunk, as the served ones have.
    {
        var cap = Capture.init(t.allocator, io, mlx.gpuStream(), &windows, &slice_src, 4 << 30);
        defer cap.deinit();
        _ = try driver.run(t.allocator, io, mlx.gpuStream(), .{
            .model_dir = slice_abs,
            .out_dir = base_abs,
            .rate = .{ .n = 24 },
            .dec = .{ .codebook = .mcg, .window = .w12 },
            .calib_sha256 = &windows.sha256,
            .calib_tokens = windows.tokens(),
            .version = "sushi-encoder-test",
            .min_rows = 1,
        }, cap.calibrator(), null);
    }

    var trace = std.ArrayList([]u8).empty;
    defer {
        for (trace.items) |b| t.allocator.free(b);
        trace.deinit(t.allocator);
    }
    var errors = std.ArrayList(TrunkError).empty;
    defer {
        for (errors.items) |e| t.allocator.free(e.name);
        errors.deinit(t.allocator);
    }
    var cap = Capture.init(t.allocator, io, mlx.gpuStream(), &windows, &slice_src, 4 << 30);
    defer cap.deinit();
    cap.trace = &trace;
    cap.trunk_errors = &errors;
    const summary = try trunk_stq.run(t.allocator, io, .{
        .model_dir = slice_abs,
        .base_dir = base_abs,
        .out_dir = pack_abs,
        .calib_sha256 = &windows.sha256,
        .calib_tokens = windows.tokens(),
        .version = "sushi-encoder-test",
    }, &cap, null);
    try t.expectEqual(@as(usize, layers), summary.layers_written);
    try t.expect(summary.head_written);
    // Per layer: the GDN layers' qkv, z, out and the shared expert's three; layer 3 (full attention) has q, k, v,
    // the indexer, o and the shared three; then lm_head. The PLE pair stays affine.
    try t.expectEqual(@as(usize, 6 + 6 + 6 + 8 + 1), summary.banks);
    try t.expectEqual(@as(usize, 6 + 6 + 6 + 8 + 1), errors.items.len);
    for (errors.items) |e| try t.expect(e.rel_err > 0 and e.rel_err < 0.1);

    var base = try std.Io.Dir.cwd().openDir(io, base_abs, .{ .iterate = true });
    defer base.close(io);
    var made = try std.Io.Dir.cwd().openDir(io, pack_abs, .{ .iterate = true });
    defer made.close(io);
    // The experts, the routers, the embedding and the n-gram table are the base's bytes.
    var untouched: usize = 0;
    var it = base.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file or e.name[0] == '.') continue;
        if (std.mem.startsWith(u8, e.name, "trunk-")) continue;
        if (!std.mem.endsWith(u8, e.name, ".safetensors") and !std.mem.eql(u8, e.name, "ngram_table.bin")) continue;
        const a = try base.readFileAlloc(io, e.name, t.allocator, .unlimited);
        defer t.allocator.free(a);
        const b = try made.readFileAlloc(io, e.name, t.allocator, .unlimited);
        defer t.allocator.free(b);
        try t.expectEqualSlices(u8, a, b);
        untouched += 1;
    }
    try t.expect(untouched >= layers + 1);

    // The index names banks for the converted linears and keeps the affine triple of the rest.
    const index_raw = try made.readFileAlloc(io, "model.safetensors.index.json", t.allocator, .limited(64 << 20));
    defer t.allocator.free(index_raw);
    const index = try std.json.parseFromSlice(std.json.Value, t.allocator, index_raw, .{});
    defer index.deinit();
    const wm = index.value.object.get("weight_map").?.object;
    const p = "language_model.model.layers.0.";
    try t.expectEqualStrings("trunk-stq-L00.safetensors", wm.get(p ++ "linear_attn.in_proj_qkv.trellis").?.string);
    try t.expect(wm.get(p ++ "linear_attn.in_proj_qkv.weight") == null and wm.get(p ++ "linear_attn.in_proj_qkv.scales") == null);
    try t.expectEqualStrings("trunk-stq-L03.safetensors", wm.get("language_model.model.layers.3.self_attn.q_proj.svh").?.string);
    try t.expectEqualStrings("trunk-stq-head.safetensors", wm.get("language_model.lm_head.trellis").?.string);
    try t.expect(wm.get("language_model.lm_head.weight") == null);
    try t.expectEqualStrings("trunk-L00.safetensors", wm.get(p ++ "linear_attn.in_proj_a.weight").?.string);
    try t.expectEqualStrings("trunk-L00.safetensors", wm.get(p ++ "attn_hyper_connection.input_mix_weight_down.scales").?.string);
    try t.expectEqualStrings("trunk-head.safetensors", wm.get("language_model.model.hyper_connection_mixer.input_mix_weight_down.weight").?.string);
    try t.expectEqualStrings("trunk-L01.safetensors", wm.get("language_model.model.layers.1.ple.key_proj.weight").?.string);
    try t.expect(wm.get("language_model.model.layers.1.ple.key_proj.trellis") == null);

    const cfg_raw = try made.readFileAlloc(io, "config.json", t.allocator, .limited(16 << 20));
    defer t.allocator.free(cfg_raw);
    const cfg = try std.json.parseFromSlice(std.json.Value, t.allocator, cfg_raw, .{});
    defer cfg.deinit();
    const trunk_spec = try api.parseQuantBlock(cfg.value.object.get("trunk_quant").?);
    try t.expectEqual(@as(u32, 96), trunk_spec.rate.n);
    try t.expect(cfg.value.object.get("quantization").?.object.get(p ++ "linear_attn.in_proj_qkv") == null);

    // The shards stamp what they hold: K4 banks, and the head's K6.
    const layer_shard = try made.readFileAlloc(io, "trunk-stq-L00.safetensors", t.allocator, .unlimited);
    defer t.allocator.free(layer_shard);
    var f0 = try sft.parse(t.allocator, layer_shard);
    defer f0.deinit(t.allocator);
    try t.expectEqualStrings("4", f0.metadata.get("k").?);
    try t.expectEqualSlices(usize, &.{ 160, 640, 64 }, f0.tensors.get(p ++ "linear_attn.in_proj_qkv.trellis").?.shape);
    var head_header = try trunk_from.readHeader(t.allocator, io, made, "trunk-stq-head.safetensors");
    defer head_header.parsed.deinit();
    try t.expectEqualStrings("6", head_header.parsed.value.object.get("__metadata__").?.object.get("k").?.string);

    // Dev lever: keep the pack (a four-layer, sixteen-expert slice of the real model) as a whole-model fixture.
    if (std.c.getenv("SUSHI_ENCODER_STQ_KEEP_DIR")) |keep| {
        const keep_abs = std.mem.span(keep);
        try io_util.makeDirPath(io, keep_abs);
        var keep_dir = try std.Io.Dir.cwd().openDir(io, keep_abs, .{});
        defer keep_dir.close(io);
        var kit = made.iterate();
        while (try kit.next(io)) |e| {
            if (e.kind == .file and e.name[0] != '.') try trunk_from.cloneOrCopy(io, made, keep_dir, e.name);
        }
    }

    // The pack's banks, decoded the way the kernels apply them, run the whole slice to the boundaries the streamed
    // run stored.
    const dec = (trunk_mod.Stq{}).dec();
    try t.expectEqual(@as(usize, 6 + 6 + 6 + 8 + 1), try writeDenseTwin(t.allocator, io, pack_abs, twin_abs, dec));
    var full = try kld.loadModel(io, t.allocator, .{ .model_dir = twin_abs });
    defer full.deinit();
    const both = batchTokensOf(t.allocator, &windows, .{ .first = 0, .count = 2, .len = calib.window_len });
    defer _ = mlx.mlx_array_free(both);
    const layer_ids = [_]u32{ 0, 1, 2, 3 };
    var outs: [layers]mlx.mlx_array = undefined;
    for (&outs) |*o| o.* = mlx.mlx_array_new();
    defer for (outs) |o| {
        _ = mlx.mlx_array_free(o);
    };
    {
        try full.xfm.resetCache();
        var ctx = full.xfm.defaultCtx();
        ctx.skip_lm_head = true;
        var cl = host.model_handle.CaptureLayers{ .ids = &layer_ids, .out = &outs };
        ctx.capture_layers = &cl;
        const result = try full.xfm.forwardWith(&ctx, both);
        defer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_array_eval(result));
    }
    try t.expectEqual(@as(usize, layers), trace.items.len);
    for (trace.items, 0..) |want, i| {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_contiguous(&flat, outs[i], false, mlx.gpuStream()));
        try mlx.check(mlx.mlx_array_eval(flat));
        try t.expectEqualSlices(u8, want, mlx.mlx_array_data_uint8(flat).?[0..want.len]);
    }
}

test "capture (real Qwen checkpoint, SUSHI_ENCODER_REAL_MODEL=1 and SUSHI_MODELS_DIR): windows of one length forwarded as one batch give each window's own boundaries and MLP rows" {
    const t = std.testing;
    const io = t.io;
    const slice = @import("slice.zig");
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const src_path = try modelsPath(&path_buf, "Qwen/Qwen3.8-Flash-Next");
    var real = std.Io.Dir.cwd().openDir(io, src_path, .{ .iterate = true }) catch return error.SkipZigTest;
    defer real.close(io);
    var real_src = try source.SafetensorsSource.open(t.allocator, io, real);
    defer real_src.deinit();
    var work = t.tmpDir(.{});
    defer work.cleanup();
    try work.dir.createDirPath(io, "slice");
    var slice_dir = try work.dir.openDir(io, "slice", .{ .iterate = true });
    defer slice_dir.close(io);
    const layers = 4;
    try slice.sliceQwen(t.allocator, io, real, &real_src, slice_dir, .{ .layers = layers, .experts = 16 });
    var abs_buf: [std.fs.max_path_bytes]u8 = undefined;
    const work_abs = abs_buf[0..try work.dir.realPath(io, &abs_buf)];
    const slice_abs = try std.fmt.allocPrint(t.allocator, "{s}/slice", .{work_abs});
    defer t.allocator.free(slice_abs);
    const pack_abs = try std.fmt.allocPrint(t.allocator, "{s}/pack", .{work_abs});
    defer t.allocator.free(pack_abs);
    _ = try driver.run(t.allocator, io, mlx.gpuStream(), .{
        .model_dir = slice_abs,
        .out_dir = pack_abs,
        .rate = .{ .n = 24 },
        .dec = .{ .codebook = .mcg, .window = .w12 },
        .version = "sushi-encoder-test",
        .trunk = .rtn,
    }, null, null);
    var tokenizer = try host.tokenizer.loadTokenizerAny(io, t.allocator, src_path);
    defer tokenizer.deinit();
    const code_ids = try calib.tokenizeText(t.allocator, &tokenizer, calib.code_text);
    defer t.allocator.free(code_ids);
    const len = 512;
    const count = 3;

    imatrix.rows_request = 0;
    defer imatrix.rows_request = null;
    var m = try kld.loadModel(io, t.allocator, .{ .model_dir = pack_abs, .ssd_budget_bytes = 4 << 30 });
    defer m.deinit();
    const layer_ids = [_]u32{ 0, 1, 2, 3 };
    var each: [count][layers][]f32 = undefined;
    for (0..count) |w| {
        const tok = mlx.mlx_array_new_data(code_ids[w * len ..].ptr, &[_]c_int{ 1, len }, 2, .uint32);
        defer _ = mlx.mlx_array_free(tok);
        each[w] = try forwardLayers(t.allocator, &m.xfm, tok, &layer_ids);
    }
    defer for (each) |e| for (e) |l| t.allocator.free(l);
    const rows_each = try m.xfm.imatrix.?.takeRows();
    defer {
        _ = mlx.mlx_array_free(rows_each.x);
        t.allocator.free(rows_each.ids);
    }
    const tok = mlx.mlx_array_new_data(code_ids.ptr, &[_]c_int{ count, len }, 2, .uint32);
    defer _ = mlx.mlx_array_free(tok);
    const batched = try forwardLayers(t.allocator, &m.xfm, tok, &layer_ids);
    defer for (batched) |l| t.allocator.free(l);
    const rows_batched = try m.xfm.imatrix.?.takeRows();
    defer {
        _ = mlx.mlx_array_free(rows_batched.x);
        t.allocator.free(rows_batched.ids);
    }
    for (0..layers) |l| {
        const per = batched[l].len / count;
        for (0..count) |w| try expectRel(each[w][l], batched[l][w * per ..][0..per], 2e-2);
    }
    const a = try readF32(t.allocator, rows_each.x);
    defer t.allocator.free(a);
    const b = try readF32(t.allocator, rows_batched.x);
    defer t.allocator.free(b);
    try expectRel(a, b, 2e-2);

    // A stored boundary enters a layer past the PLE layer as a batched stream: the capture's BF16 path at layer >= 1.
    const per = batched[0].len / count;
    const bf = try t.allocator.alloc(u16, batched[0].len);
    defer t.allocator.free(bf);
    for (bf, batched[0]) |*o, v| o.* = @truncate(@as(u32, @bitCast(v)) >> 16);
    const stream = mlx.mlx_array_new_data(bf.ptr, &[_]c_int{ count, len, @intCast(per / len) }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(stream);
    try m.xfm.resetCache();
    var ctx = m.xfm.defaultCtx();
    ctx.skip_lm_head = true;
    host.model_handle.setLayerCapture(&ctx, stream, true);
    const out = try m.xfm.forwardWith(&ctx, tok);
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    const rows_stream = try m.xfm.imatrix.?.takeRows();
    _ = mlx.mlx_array_free(rows_stream.x);
    t.allocator.free(rows_stream.ids);
}

fn readF32(allocator: std.mem.Allocator, x: mlx.mlx_array) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, x, .float32, mlx.gpuStream()));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, f, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(c));
    return allocator.dupe(f32, mlx.mlx_array_data_float32(c).?[0..mlx.mlx_array_size(c)]);
}

fn forwardLayers(allocator: std.mem.Allocator, xfm: *host.trunk.Trunk, tok: mlx.mlx_array, ids: []const u32) ![4][]f32 {
    try xfm.resetCache();
    var ctx = xfm.defaultCtx();
    ctx.skip_lm_head = true;
    var outs: [4]mlx.mlx_array = undefined;
    for (&outs) |*o| o.* = mlx.mlx_array_new();
    defer for (outs) |o| {
        _ = mlx.mlx_array_free(o);
    };
    var cl = host.model_handle.CaptureLayers{ .ids = ids, .out = &outs };
    ctx.capture_layers = &cl;
    const result = try xfm.forwardWith(&ctx, tok);
    defer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_array_eval(result));
    var got: [4][]f32 = undefined;
    for (outs, &got) |o, *g| g.* = try readF32(allocator, o);
    return got;
}

/// ||a - b|| / ||a|| under `tol`: a batch runs other kernel shapes, so its bf16 roundings differ.
fn expectRel(a: []const f32, b: []const f32, tol: f64) !void {
    try std.testing.expectEqual(a.len, b.len);
    var num: f64 = 0;
    var den: f64 = 0;
    for (a, b) |x, y| {
        num += (@as(f64, x) - y) * (@as(f64, x) - y);
        den += @as(f64, x) * x;
    }
    try std.testing.expect(@sqrt(num / den) < tol);
}

test "capture: a hidden boundary survives a save and a load, an older boundary is dropped, and ready holds for every layer before the saved one" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "scratch");
    var ids: [12]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i);
    var windows = try calib.fromLengths(t.allocator, &ids, &.{ 4, 8 });
    defer windows.deinit();
    var cap = Capture.init(t.allocator, io, mlx.gpuStream(), &windows, undefined, 0);
    cap.scratch = try tmp.dir.openDir(io, "scratch", .{ .iterate = true });
    var abs: [std.fs.max_path_bytes]u8 = undefined;
    cap.scratch_abs = try t.allocator.dupe(u8, abs[0..try cap.scratch.realPath(io, &abs)]);
    defer cap.deinit();
    var trace = std.ArrayList([]u8).empty;
    defer {
        for (trace.items) |b| t.allocator.free(b);
        trace.deinit(t.allocator);
    }
    cap.trace = &trace;

    var h = try Hidden.init(t.allocator, windows.starts, &.{ 1, 4, 16 }, .bfloat16, 2);
    for (h.bytes, 0..) |*b, i| b.* = @truncate(i *% 7);
    h.boundary = 3;
    cap.hidden = h;
    try cap.saveBoundary(3);
    try t.expectEqual(@as(usize, 1), trace.items.len);
    try t.expectEqualSlices(u8, h.window(0), trace.items[0]);
    try t.expectEqual(@as(usize, 12 * 16 * 2), h.bytes.len);
    const want = try t.allocator.dupe(u8, h.bytes);
    defer t.allocator.free(want);

    try t.expect(Capture.ready(&cap, 2, false));
    try t.expect(Capture.ready(&cap, 0, false));
    try t.expect(!Capture.ready(&cap, 3, false));
    try t.expect(Capture.ready(&cap, 9, true));

    cap.hidden.?.deinit();
    cap.hidden = null;
    try cap.loadBoundary(3);
    try t.expectEqualSlices(u8, want, cap.hidden.?.bytes);
    try t.expectEqual(mlx.mlx_dtype.bfloat16, cap.hidden.?.dtype);
    try t.expectEqualSlices(c_int, &.{ 1, 4, 16 }, cap.hidden.?.dims());
    try t.expectError(error.FileNotFound, cap.loadBoundary(4));

    cap.hidden.?.boundary = 4;
    try cap.saveBoundary(4);
    try t.expect(Capture.ready(&cap, 2, false));
    try t.expect(Capture.ready(&cap, 3, false));
    try t.expect(!Capture.ready(&cap, 4, false));
    try cap.scratch.writeFile(io, .{ .sub_path = ".hidden-6.bin.tmp", .data = "x" });
    try t.expect(!Capture.ready(&cap, 4, false));

    var copy = try Hidden.init(t.allocator, windows.starts, &.{ 1, 4, 16 }, .bfloat16, 2);
    defer copy.deinit();
    const arr = cap.hidden.?.array(1);
    defer _ = mlx.mlx_array_free(arr);
    try t.expectEqualSlices(c_int, &.{ 1, 8, 16 }, mlx.getShape(arr));
    try copy.store(mlx.gpuStream(), 1, arr);
    try t.expectEqualSlices(u8, want[4 * 16 * 2 ..], copy.window(1));
    try t.expectError(error.HiddenShapeMismatch, copy.store(mlx.gpuStream(), 0, arr));
}

test "capture: a stored BF16 boundary window is the window's own rows of every stream, shaped as the forward's [batch, len, streams x hidden] stream" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var ids: [12]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i);
    var windows = try calib.fromLengths(t.allocator, &ids, &.{ 4, 8 });
    defer windows.deinit();
    var cap = Capture.init(t.allocator, io, mlx.gpuStream(), &windows, undefined, 0);
    defer cap.deinit();
    const hc = 2;
    const hidden = 3;
    var rows: [12 * hc * hidden]u16 = undefined;
    for (&rows, 0..) |*v, i| v.* = @intCast(i);
    try tmp.dir.writeFile(io, .{ .sub_path = "boundary-00.u16", .data = std.mem.sliceAsBytes(&rows) });
    var file = try tmp.dir.openFile(io, "boundary-00.u16", .{});
    defer file.close(io);
    var buf: [8 * hc * hidden * 2]u8 = undefined;
    const arr = try cap.storedWindow(.{ .file = file, .hc = hc, .hidden = hidden, .buf = &buf }, 1);
    defer _ = mlx.mlx_array_free(arr);
    try t.expectEqualSlices(c_int, &.{ 1, 8, hc * hidden }, mlx.getShape(arr));
    try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(arr));
    try mlx.check(mlx.mlx_array_eval(arr));
    const got: [*]const u16 = @ptrCast(@alignCast(mlx.mlx_array_data_uint8(arr).?));
    try t.expectEqualSlices(u16, rows[4 * hc * hidden ..], got[0 .. 8 * hc * hidden]);
}

test "capture: the in-run BF16 path's first boundary is each calibration token's source embedding row in every stream" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try pack.fixture.write(t.allocator, io, tmp.dir, &.{.{ .name = embed_key, .dtype = "BF16", .shape = &.{ 16, 8 } }});
    var src = try source.SafetensorsSource.open(t.allocator, io, tmp.dir);
    defer src.deinit();
    const ids = [_]u32{ 3, 0, 15, 3, 7, 9, 1, 2, 4, 4, 11 };
    var windows = try calib.fromLengths(t.allocator, &ids, &.{ 4, 7 });
    defer windows.deinit();
    var cap = Capture.init(t.allocator, io, mlx.gpuStream(), &windows, &src, 0);
    defer cap.deinit();
    const hc = 3;
    try cap.writeEmbeddingBoundary(tmp.dir, "boundary-00.u16", hc, 8, 4);
    const got = try tmp.dir.readFileAlloc(io, "boundary-00.u16", t.allocator, .limited(1 << 20));
    defer t.allocator.free(got);
    const table = try t.allocator.alloc(u8, 16 * 8 * 2);
    defer t.allocator.free(table);
    try src.readRange(src.tensors.get(embed_key).?, 0, table);
    try t.expectEqual(ids.len * hc * 8 * 2, got.len);
    for (ids, 0..) |id, i| for (0..hc) |s| {
        try t.expectEqualSlices(u8, table[id * 16 ..][0..16], got[(i * hc + s) * 16 ..][0..16]);
    };
    try t.expectError(error.FileNotFound, tmp.dir.access(io, ".boundary-00.u16.tmp", .{}));
    try t.expectError(error.BoundaryFileShape, cap.writeEmbeddingBoundary(tmp.dir, "boundary-00.u16", hc, 6, 4));
}

test "capture: a batch's layer output rows land at the batch's own rows of the next boundary, and only BF16 is taken" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var ids: [12]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i);
    var windows = try calib.fromLengths(t.allocator, &ids, &.{ 4, 4, 4 });
    defer windows.deinit();
    var cap = Capture.init(t.allocator, io, mlx.gpuStream(), &windows, undefined, 0);
    defer cap.deinit();
    const hc = 2;
    const hidden = 3;
    var next = try tmp.dir.createFile(io, "next.u16", .{ .read = true });
    defer next.close(io);
    var buf: [1]u8 = undefined;
    const st: Capture.Stored = .{ .file = undefined, .hc = hc, .hidden = hidden, .buf = &buf, .next = next };
    var rows: [8 * hc * hidden]u16 = undefined;
    for (&rows, 0..) |*v, i| v.* = @intCast(1000 + i);
    // The forward's own four-axis [batch, len, streams, hidden] output of windows 1 and 2.
    const slot = mlx.mlx_array_new_data(&rows, &[_]c_int{ 2, 4, hc, hidden }, 4, .bfloat16);
    defer _ = mlx.mlx_array_free(slot);
    try cap.writeNext(st, .{ .first = 1, .count = 2, .len = 4 }, slot);
    var got: [8 * hc * hidden]u16 = undefined;
    try t.expectEqual(got.len * 2, try next.readPositionalAll(io, std.mem.sliceAsBytes(&got), 4 * hc * hidden * 2));
    try t.expectEqualSlices(u16, &rows, &got);
    const wide = mlx.mlx_array_new_data(&rows, &[_]c_int{ 1, 4, hc, hidden }, 4, .float32);
    defer _ = mlx.mlx_array_free(wide);
    try t.expectError(error.BoundaryDtype, cap.writeNext(st, .{ .first = 0, .count = 1, .len = 4 }, wide));
    const short = mlx.mlx_array_new_data(&rows, &[_]c_int{ 1, 4, hc, hidden }, 4, .bfloat16);
    defer _ = mlx.mlx_array_free(short);
    try t.expectError(error.BoundaryRowsMismatch, cap.writeNext(st, .{ .first = 1, .count = 2, .len = 4 }, short));
}

test "capture: a run stopped after any step of the boundary roll resumes with both boundaries of its layer on disk" {
    const t = std.testing;
    const io = t.io;
    var ids: [4]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i);
    var windows = try calib.fromLengths(t.allocator, &ids, &.{4});
    defer windows.deinit();
    // Layer 3 is finished and its output is being published as boundary 4; step k is the k-th file operation.
    var step: usize = 1;
    while (step <= 4) : (step += 1) {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        var cap = Capture.init(t.allocator, io, mlx.gpuStream(), &windows, undefined, 0);
        defer cap.deinit();
        cap.scratch = try tmp.dir.openDir(io, ".", .{ .iterate = true });
        cap.scratch_abs = try t.allocator.dupe(u8, "x");
        cap.hidden = try Hidden.init(t.allocator, windows.starts, &.{ 1, 4, 2 }, .bfloat16, 2);
        try tmp.dir.writeFile(io, .{ .sub_path = "boundary-03.u16", .data = "a" });
        try tmp.dir.writeFile(io, .{ .sub_path = "hidden-3.bin", .data = "q" });
        try tmp.dir.writeFile(io, .{ .sub_path = ".boundary-04.u16.tmp", .data = "b" });
        cap.stop_after = step;
        try t.expectError(error.InjectedStop, cap.saveBoundaries(4, true));
        // Layer 3 is redone from boundary 3 unless hidden-4 is there, and then layer 4 starts from boundary 4.
        const advanced = Capture.ready(&cap, 3, false);
        const from: []const u8 = if (advanced) "4" else "3";
        var name: [32]u8 = undefined;
        try tmp.dir.access(io, try std.fmt.bufPrint(&name, "hidden-{s}.bin", .{from}), .{});
        try tmp.dir.access(io, try std.fmt.bufPrint(&name, "boundary-0{s}.u16", .{from}), .{});
        try t.expectEqual(step >= 2, advanced);
    }
}

test "capture: the in-run BF16 boundary rolls with the quantized one and drops the one before" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var ids: [4]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i);
    var windows = try calib.fromLengths(t.allocator, &ids, &.{4});
    defer windows.deinit();
    var cap = Capture.init(t.allocator, io, mlx.gpuStream(), &windows, undefined, 0);
    defer cap.deinit();
    cap.scratch = try tmp.dir.openDir(io, ".", .{ .iterate = true });
    cap.scratch_abs = try t.allocator.dupe(u8, "x");
    cap.hidden = try Hidden.init(t.allocator, windows.starts, &.{ 1, 4, 2 }, .bfloat16, 2);
    try tmp.dir.writeFile(io, .{ .sub_path = "boundary-03.u16", .data = "a" });
    try tmp.dir.writeFile(io, .{ .sub_path = "hidden-3.bin", .data = "q" });
    try tmp.dir.writeFile(io, .{ .sub_path = ".boundary-04.u16.tmp", .data = "b" });
    try cap.saveBoundaries(4, true);
    try tmp.dir.access(io, "hidden-4.bin", .{});
    try t.expectError(error.FileNotFound, tmp.dir.access(io, "hidden-3.bin", .{}));
    try t.expectError(error.FileNotFound, tmp.dir.access(io, "boundary-03.u16", .{}));
    const next = try tmp.dir.readFileAlloc(io, "boundary-04.u16", t.allocator, .limited(16));
    defer t.allocator.free(next);
    try t.expectEqualStrings("b", next);
    // No next BF16 boundary: nothing is published or dropped.
    try t.expectError(error.FileNotFound, cap.saveBoundaries(5, true));
    try tmp.dir.access(io, "boundary-04.u16", .{});
    try tmp.dir.access(io, "hidden-4.bin", .{});
}

test "capture: a calibration window's forward runs on the cut Qwen pack, bound at the window's rows (QWEN4_TEST_MODEL)" {
    const t = std.testing;
    const dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    try mlx.requireGpu();
    var loaded = try kld.loadModel(t.io, t.allocator, .{ .model_dir = dir });
    defer loaded.deinit();
    var ids: [calib.window_len]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast((i * 7 + 3) % @as(usize, loaded.config.vocab_size));
    var windows = try calib.fromTokens(t.allocator, &ids, ids.len);
    defer windows.deinit();
    var cap = Capture.init(t.allocator, t.io, mlx.gpuStream(), &windows, undefined, 0);
    defer cap.deinit();
    var info: driver.Info = undefined;
    info.ple_layer = std.math.maxInt(u32);
    cap.info = info;
    // One window a forward, the plan a trunk that is not re-rounded takes.
    cap.batches = try trunk_mod.batchPlan(t.allocator, windows.starts, 0);
    var hidden: ?Hidden = null;
    defer if (hidden) |*h| h.deinit();
    try cap.forwardWindows(loaded, 0, &hidden);
    try t.expect(hidden != null);
}
