//! Hy4-Flash's layer-major calibration capture: `sushi kld capture --layer-major --tokens 1` on the BF16 release. A batch
//! of W windows runs layer 0, then layer 1, and so on, so each MoE layer's 512 streamed experts are read once per batch
//! (one pinned union), never once per window. Between layers a window carries only its four f32 streams and its Engram
//! rows; a layer's attention state goes once the window has run it. Beside the standard one-row fixture it writes, in
//! `<out>/calib/`, DeepSeek's contract (`dsv41/kld_capture.zig`) with Hy4's inputs:
//!   mlp-in-LL.bf16      what layer LL's routed experts read: the latent row (`lat_down` of the FFN input), [tokens, 1536],
//!                       BF16 when the routed products are BF16 (the values are BF16's, so nothing is lost), else .f32
//!   route-LL.u16        the router's top-k global expert ids, route-w-LL.f32 the weights the layer applied
//!   trunk-LL.safetensors  sum x x^T over every row of each trunk linear's input (f32), trunk-head.safetensors the head's
//!   tokens.bin, layout.json
//!   head-in.f32         opt-in (`SUSHI_HY4_HEAD_ROWS=1`): the head's input for every row, the final norm's output,
//!                       [tokens, hidden] f32, tokens in window order; f32 is the forward's own dtype, so nothing is rounded
//! `SUSHI_HY4_CAPTURE_LEAN=1` (only with HEAD_ROWS) writes nothing but tokens.bin, layout.json and head-in.f32 beside the
//! fixture: no mlp-in, route or trunk sums, and none of their GPU work.
//! The forward's per-layer entry, pinned union and taps are `hy4_model`'s and `hy4_forward`'s.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const kld = @import("../kld.zig");
const fixture_mod = @import("../kld_fixture.zig");
const util = @import("../kld_capture_util.zig");
const calib = @import("../kld_calib.zig");
const expert_stream = @import("../expert_stream.zig");
const hy4_config = @import("hy4_config.zig");
const hy4_model = @import("hy4_model.zig");
const fwd = @import("hy4_forward.zig");
const hy4_ledger = @import("hy4_ledger.zig");
const Arr = mlx.mlx_array;

/// Windows a batch holds at most.
pub const batch_cap: u32 = 32;

/// Rows of pieces whose FFN runs together: bounds the batched experts' planes, not their speed (a test narrows it).
var ffn_rows: usize = 16384;

/// The trunk inputs the sums cover, by what reads them.
pub const Kind = enum {
    /// The input norm's output: q_a, kv_a, linear_gate, the indexer's token key and head weights.
    attn_in,
    /// q_a's normed output: q_b, the indexer's query.
    q_lora,
    /// The normed KV latent: kv_b.
    kv_latent,
    /// The gated attention rows: o.
    o_in,
    /// The FFN input: lat_down, the router, the shared expert's gate and up; layer 0's dense gate and up.
    ffn_in,
    /// What the routed experts read, `lat_down(ffn_in)`: stored as rows (`mlp-in-LL`), never summed.
    expert_in,
    /// The routed sum in the latent: lat_up.
    lat_up_in,
    /// The shared expert's SwiGLU activation (layer 0: the dense FFN's): its down.
    shared_act,
    /// Engram's normed table rows: its value and key projections.
    engram_in,
    /// The final norm's output: the head.
    head_in,
};
const kinds = std.enums.values(Kind).len;

fn isMoe(c: *const hy4_config.Config, layer: usize) bool {
    return layer >= c.dense_layers;
}

fn hasEngram(c: *const hy4_config.Config, layer: usize) bool {
    for (c.engram.layers[0..c.engram.n]) |l| if (l == layer) return true;
    return false;
}

/// The input width of `kind` at `layer`, 0 where the layer has no such input.
fn width(c: *const hy4_config.Config, layer: usize, kind: Kind) u64 {
    return switch (kind) {
        .attn_in, .ffn_in, .head_in => c.dim,
        .q_lora => c.q_lora,
        .kv_latent => c.kv_lora,
        .o_in => @as(u64, c.heads(layer)) * c.v_dim,
        .lat_up_in => if (isMoe(c, layer)) c.moe_inter else 0,
        .expert_in => 0,
        .shared_act => if (isMoe(c, layer)) c.moe_inter else c.dense_inter,
        .engram_in => if (hasEngram(c, layer)) @as(u64, c.engram.primeCount()) * c.engram.head_dim else 0,
    };
}

// ── the bill ─────────────────────────────────────────────────────────────────────────────────────

/// The f32 sums one layer holds (the head's for `layer` = layers).
pub fn layerSumsBytes(c: *const hy4_config.Config, layer: usize) u64 {
    var total: u64 = 0;
    if (layer == c.layers) return 4 * @as(u64, c.dim) * c.dim;
    for (std.enums.values(Kind)) |k| if (k != .head_in) {
        const w = width(c, layer, k);
        total += 4 * w * w;
    };
    return total;
}

/// The largest layer's sums: the running total on the GPU, the grams submitted and not yet waited for, the host copy
/// handed to the folder, the folder's two queued jobs and the previous total it reads back.
fn sumsInFlight(c: *const hy4_config.Config) u64 {
    var widest: u64 = 0;
    for (0..c.layers + 1) |l| widest = @max(widest, layerSumsBytes(c, l));
    return 5 * widest;
}

/// What one window of `rows` tokens carries between layers: its four f32 streams and its Engram rows.
pub fn carriedPerWindow(c: *const hy4_config.Config, rows: u64) u64 {
    const streams = rows * c.hc_mult * c.dim * 4;
    const engram = if (c.engram.n == 0) 0 else rows * (@as(u64, c.engram.n) * c.engram.primeCount() * 4 + 4);
    return streams + engram;
}

/// What the FFN of up to `ffn_rows` rows of pieces holds at once: their streams after the attention and FFN inputs in
/// f32, every pick's activation and output at f32 (the capture's products), and one block's four f32 SwiGLU planes.
fn ffnBatchBytes(c: *const hy4_config.Config) u64 {
    const rows: u64 = ffn_rows;
    const picks = rows * c.top_k;
    return rows * (@as(u64, c.hc_mult) * c.dim * 4 + @as(u64, c.dim) * 4) + picks * c.moe_inter * 4 * 2 + @as(u64, fwd.batched_block) * 2 * c.moe_inter * 4 * 4;
}

/// The one window running a layer: what a forward of its `rows` asks past its state (`window`, `windowBytes`), that
/// layer's state at BF16, the sums in flight, the batched FFN and the bill's margin.
pub fn reserveBytes(c: *const hy4_config.Config, rows: u64, window: u64) u64 {
    const layer_state = rows * hy4_ledger.latentRowBytes(c, 16) + rows / c.indexer.group * c.indexer.head_dim * 4 +
        @as(u64, c.window) * hy4_ledger.latentRowBytes(c, 16);
    return window + layer_state + sumsInFlight(c) + ffnBatchBytes(c) + @import("../bills.zig").BILL_MARGIN_BYTES;
}

/// What the forwards of a window of `rows` tokens ask past its state, at BF16 KV, measured on the loaded model dry
/// (D104): the capture runs the window in pieces of `chunk` rows (`runBatch`), each after the ones before it, so the
/// most the recordings of those pieces hold at once (`plan.region.need`).
pub fn windowBytes(m: *hy4_model.Model, gpa: std.mem.Allocator, rows: u64, chunk: u64) !u64 {
    const hy4_plan = @import("hy4_plan.zig");
    const plan = @import("../tensor/plan/plan_api.zig");
    // A piece of a few rows takes the wide hyper-connection pipelines, which a dry recording needs latched (D20).
    try m.probeWide();
    var most: plan.region.Need = .{};
    var at: u64 = 0;
    while (at < rows) : (at += chunk) {
        var passes = try hy4_plan.recordChecked(m, gpa, .{ .rows = @intCast(@min(chunk, rows - at)), .readout = .none, .keep_hidden = false, .kv = .dense, .n = @intCast(at), .phase = .wide }, @intCast(rows));
        defer passes.deinit();
        for (passes.traces) |*t| most = most.max(try plan.region.need(gpa, t, plan.region.default_alignment));
    }
    return most.scratch +| most.host;
}

/// The cheapest budget the streamed load accepts: what stays resident, the all-experts union a batch pins, one token's
/// picks, the bounce buffers and two slots a layer.
pub fn floorBudget(c: *const hy4_config.Config, resident: u64, per_expert: u64) !u64 {
    const moe_layers: u64 = c.layers - c.dense_layers;
    const workspace = try std.math.mul(u64, c.experts, per_expert);
    const selected = try std.math.mul(u64, c.top_k, per_expert);
    const cache = try std.math.mul(u64, 2 * moe_layers, per_expert);
    return resident + workspace + selected + cache + expert_stream.BOUNCE_BYTES;
}

/// The largest batch, at most `batch_cap` windows of `rows` tokens, that fits `budget` beside the floor.
pub fn batchWindows(c: *const hy4_config.Config, budget: u64, floor: u64, rows: u64, window: u64) !u32 {
    const reserve = reserveBytes(c, rows, window);
    if (budget <= floor or budget - floor <= reserve) return error.Hy4CaptureBudgetTooSmall;
    const fits = (budget - floor - reserve) / carriedPerWindow(c, rows);
    if (fits == 0) return error.Hy4CaptureBudgetTooSmall;
    return @intCast(@min(fits, batch_cap));
}

/// A batch `w` windows wide fits `budget`, or is refused by name.
pub fn checkBatch(c: *const hy4_config.Config, budget: u64, floor: u64, rows: u64, window: u64, w: u32) !void {
    const need = floor +| reserveBytes(c, rows, window) +| @as(u64, w) *| carriedPerWindow(c, rows);
    if (need > budget) {
        @import("../log.zig").err("[kld] layer-major: --batch-windows {d} of {d}-token windows needs {d:.2} GB, the budget gives {d:.2}\n", .{ w, rows, @as(f64, @floatFromInt(need)) / 1e9, @as(f64, @floatFromInt(budget)) / 1e9 });
        return error.Hy4CaptureBudgetExceeded;
    }
}

/// Bytes the remaining windows still write: rows, routes, token ids, every layer's sums (two versions in flight), logits;
/// `head_row_bytes` is what the head's input rows add a token (0 when they are off).
pub fn diskBytes(c: *const hy4_config.Config, tokens: u64, windows: u64, mlp_in_itemsize: u64, head_row_bytes: u64, lean: bool) u64 {
    if (lean) return tokens * (head_row_bytes + 4) + windows * @as(u64, c.vocab) * 4;
    const moe_layers: u64 = c.layers - c.dense_layers;
    const per_token = moe_layers * (@as(u64, c.moe_inter) * mlp_in_itemsize + @as(u64, c.top_k) * 6) + 4 + head_row_bytes;
    var sums: u64 = 0;
    for (0..c.layers + 1) |l| sums += layerSumsBytes(c, l);
    return tokens * per_token + 2 * sums + windows * @as(u64, c.vocab) * 4;
}

/// The head's input rows, in `calib/`.
const head_rows_file = "head-in.f32";

/// What a run writes beyond the standard outputs.
pub const Mode = struct {
    head_rows: bool = false,
    lean: bool = false,

    /// The manifest's mode, which a resume must match; null for the default outputs.
    fn tag(self: Mode) ?[]const u8 {
        if (!self.head_rows) return null;
        return if (self.lean) "head-rows-lean" else "head-rows";
    }

    /// A lean run keeps only the head rows, so without them it would write nothing a tuner can use.
    pub fn check(self: Mode) !Mode {
        if (self.lean and !self.head_rows) return error.Hy4CaptureLeanNeedsHeadRows;
        return self;
    }
};

pub fn modeFromEnv() !Mode {
    const diagEnvOn = @import("../trunk.zig").diagEnvOn;
    return (Mode{ .head_rows = diagEnvOn("SUSHI_HY4_HEAD_ROWS"), .lean = diagEnvOn("SUSHI_HY4_CAPTURE_LEAN") }).check();
}

// ── the calib writer ─────────────────────────────────────────────────────────────────────────────

pub const Capture = struct {
    /// A lean capture keeps only the head rows: it opens no other file and reads no other input.
    lean: bool,
    a: std.mem.Allocator,
    s: mlx.mlx_stream,
    cfg: *const hy4_config.Config,
    /// Per MoE layer: mlp-in, route ids, route weights (index `layer - dense_layers`); empty when lean.
    fds: []std.c.fd_t,
    folder: *calib.Folder,
    /// `mlp-in-LL` as BF16 (the BF16 routed products' values) or f32.
    bf16: bool,
    /// `head-in.f32`, or -1 when the head's input rows are not written.
    head_fd: std.c.fd_t = -1,
    /// The forward's tap, bound to this capture where it lives (`bind`).
    tap_value: fwd.Tap = undefined,
    /// The layer in flight; `cfg.layers` is the head.
    layer: usize = 0,
    acc: [kinds]Arr = @splat(.{ .ctx = null }),
    rows: [kinds]u64 = @splat(0),
    /// Nanoseconds the capture itself worked, after the forward made its inputs.
    extra_ns: u64 = 0,
    io: std.Io = std.Io.Threaded.global_single_threaded.io(),

    pub fn open(a: std.mem.Allocator, s: mlx.mlx_stream, cfg: *const hy4_config.Config, dir: []const u8, folder: *calib.Folder, bf16: bool, mode: Mode) !Capture {
        const moe: usize = if (mode.lean) 0 else cfg.layers - cfg.dense_layers;
        const fds = try a.alloc(std.c.fd_t, 3 * moe);
        errdefer a.free(fds);
        @memset(fds, -1);
        errdefer for (fds) |fd| if (fd >= 0) {
            _ = std.c.close(fd);
        };
        var name: [48]u8 = undefined;
        const ext = if (bf16) "bf16" else "f32";
        for (0..moe) |i| {
            const l = i + cfg.dense_layers;
            fds[3 * i] = try calib.openAppend(dir, try std.fmt.bufPrint(&name, "mlp-in-{d:0>2}.{s}", .{ l, ext }));
            fds[3 * i + 1] = try calib.openAppend(dir, try std.fmt.bufPrint(&name, "route-{d:0>2}.u16", .{l}));
            fds[3 * i + 2] = try calib.openAppend(dir, try std.fmt.bufPrint(&name, "route-w-{d:0>2}.f32", .{l}));
        }
        const head_fd = if (mode.head_rows) try calib.openAppend(dir, head_rows_file) else -1;
        return .{ .lean = mode.lean, .a = a, .s = s, .cfg = cfg, .fds = fds, .folder = folder, .bf16 = bf16, .head_fd = head_fd };
    }

    pub fn close(self: *Capture) void {
        self.resetSums();
        for (self.fds) |fd| _ = std.c.close(fd);
        if (self.head_fd >= 0) _ = std.c.close(self.head_fd);
        self.a.free(self.fds);
    }

    fn fileOf(self: *const Capture, layer: usize, which: usize) !std.c.fd_t {
        if (!isMoe(self.cfg, layer) or layer >= self.cfg.layers) return error.Hy4CaptureOutOfOrder;
        return self.fds[3 * (layer - self.cfg.dense_layers) + which];
    }

    fn resetSums(self: *Capture) void {
        for (&self.acc) |*x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x.*);
            x.* = .{ .ctx = null };
        }
        self.rows = @splat(0);
    }

    pub fn begin(self: *Capture, layer: usize) void {
        self.resetSums();
        self.layer = layer;
    }

    pub fn itemsize(self: *const Capture) u64 {
        return if (self.bf16) 2 else 4;
    }

    /// What one token's head row adds on disk.
    pub fn headRowBytes(self: *const Capture) u64 {
        return if (self.head_fd >= 0) @as(u64, self.cfg.dim) * 4 else 0;
    }

    /// Back to the committed tokens: an interrupted batch's rows and routes go.
    pub fn truncate(self: *Capture, tokens: u64) !void {
        const k: u64 = self.cfg.top_k;
        if (self.head_fd >= 0) try calib.truncateFd(self.head_fd, tokens * self.headRowBytes());
        for (0..self.fds.len / 3) |i| {
            try calib.truncateFd(self.fds[3 * i], tokens * self.cfg.moe_inter * self.itemsize());
            try calib.truncateFd(self.fds[3 * i + 1], tokens * k * 2);
            try calib.truncateFd(self.fds[3 * i + 2], tokens * k * 4);
        }
    }

    pub fn sync(self: *Capture) !void {
        for (self.fds) |f| if (std.c.fsync(f) != 0) return error.CalibCaptureWriteFailed;
        if (self.head_fd >= 0 and std.c.fsync(self.head_fd) != 0) return error.CalibCaptureWriteFailed;
    }

    fn timed(self: *Capture) std.Io.Timestamp {
        return std.Io.Timestamp.now(self.io, .awake);
    }

    fn charge(self: *Capture, started: std.Io.Timestamp) void {
        self.extra_ns += @intCast(started.untilNow(self.io, .awake).nanoseconds);
    }

    /// Adds `x x^T` over the rows of `x` [..., in] to the layer's running sum of `kind`, in f32.
    pub fn input(self: *Capture, layer: usize, kind: Kind, x: Arr) !void {
        if (self.lean and kind != .head_in) return;
        if (layer != self.layer or (kind == .head_in) != (layer == self.cfg.layers)) return error.Hy4CaptureOutOfOrder;
        if (kind == .expert_in) return self.expertIn(layer, x);
        const started = self.timed();
        defer self.charge(started);
        if (kind == .head_in and self.head_fd >= 0) try self.headRows(x);
        if (self.lean) return;
        const shape = mlx.getShape(x);
        const in: usize = @intCast(shape[shape.len - 1]);
        if (in != width(self.cfg, layer, kind)) return error.Hy4CaptureWidth;
        const rows = mlx.mlx_array_size(x) / in;
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ @intCast(rows), @intCast(in) }, 2, self.s));
        var xf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xf);
        try mlx.check(mlx.mlx_astype(&xf, flat, .float32, self.s));
        var xt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xt);
        try mlx.check(mlx.mlx_transpose(&xt, xf, self.s));
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_matmul(&g, xt, xf, self.s));
        const k = @intFromEnum(kind);
        var total = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(total);
        if (self.acc[k].ctx != null) try mlx.check(mlx.mlx_add(&total, self.acc[k], g, self.s)) else try mlx.check(mlx.mlx_array_set(&total, g));
        // Submitted, not waited for: the kind's previous sum is the one waited on, so a kind keeps two grams in flight
        // and the forward goes on while the GPU multiplies. `end` waits for the last.
        if (self.acc[k].ctx != null) try mlx.check(mlx.mlx_array_eval(self.acc[k]));
        const pending = mlx.mlx_vector_array_new_value(total);
        defer _ = mlx.mlx_vector_array_free(pending);
        try mlx.check(mlx.mlx_async_eval(pending));
        if (self.acc[k].ctx != null) _ = mlx.mlx_array_free(self.acc[k]);
        self.acc[k] = total;
        self.rows[k] += rows;
    }

    /// Appends the head's input rows `x` [T, hidden], which the forward makes in f32: written whole, never rounded.
    fn headRows(self: *Capture, x: Arr) !void {
        if (mlx.mlx_array_dtype(x) != .float32) return error.Hy4CaptureHeadRowsDtype;
        const bytes = try calib.hostBytes(self.a, self.s, x);
        defer self.a.free(bytes);
        if (bytes.len % (@as(usize, self.cfg.dim) * 4) != 0) return error.Hy4CaptureWidth;
        try calib.writeAll(self.head_fd, bytes);
    }

    /// What a MoE layer's routed experts read, `lat` [T, latent] f32: appended to `mlp-in-LL`, as BF16 when the products
    /// are (each value is then a BF16's, checked, so the file loses nothing).
    fn expertIn(self: *Capture, layer: usize, lat: Arr) !void {
        if (layer != self.layer) return error.Hy4CaptureOutOfOrder;
        if (mlx.mlx_array_dtype(lat) != .float32) return error.Hy4CaptureMlpInDtype;
        const started = self.timed();
        defer self.charge(started);
        const rows = try calib.hostBytes(self.a, self.s, lat);
        defer self.a.free(rows);
        if (self.bf16) {
            const halves = try self.a.alloc(u16, rows.len / 4);
            defer self.a.free(halves);
            for (halves, 0..) |*o, i| {
                const bits = std.mem.readInt(u32, rows[i * 4 ..][0..4], .little);
                if (bits & 0xffff != 0) return error.Hy4CaptureLatentNotBf16;
                o.* = @intCast(bits >> 16);
            }
            try calib.writeAll(try self.fileOf(layer, 0), std.mem.sliceAsBytes(halves));
        } else try calib.writeAll(try self.fileOf(layer, 0), rows);
    }

    /// A MoE layer's picks (host, [T·k] global expert ids) and the weights [T, k] it applies: appended to its files.
    pub fn routes(self: *Capture, layer: usize, picks: []const u32, weights: Arr) !void {
        if (self.lean) return;
        if (layer != self.layer) return error.Hy4CaptureOutOfOrder;
        const started = self.timed();
        defer self.charge(started);
        const k: usize = self.cfg.top_k;
        if (picks.len % k != 0) return error.Hy4CaptureWidth;
        const packed_ids = try self.a.alloc(u16, picks.len);
        defer self.a.free(packed_ids);
        for (picks, packed_ids) |p, *o| {
            if (p >= self.cfg.experts) return error.ExpertOutOfRange;
            o.* = @intCast(p);
        }
        try calib.writeAll(try self.fileOf(layer, 1), std.mem.sliceAsBytes(packed_ids));
        var wf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wf);
        try mlx.check(mlx.mlx_astype(&wf, weights, .float32, self.s));
        const w = try calib.hostBytes(self.a, self.s, wf);
        defer self.a.free(w);
        if (w.len != picks.len * 4) return error.Hy4CaptureWidth;
        try calib.writeAll(try self.fileOf(layer, 2), w);
    }

    /// Hands the layer's (or the head's) sums to the folder as the running total of windows `[0, to)`.
    pub fn end(self: *Capture, from: usize, to: usize) !void {
        if (self.lean) return;
        const started = self.timed();
        defer self.charge(started);
        var entries: std.ArrayList(calib.Entry) = .empty;
        errdefer {
            for (entries.items) |*e| e.deinit(self.a);
            entries.deinit(self.a);
        }
        const head = self.layer == self.cfg.layers;
        for (std.enums.values(Kind)) |kind| {
            const k = @intFromEnum(kind);
            const x = self.acc[k];
            if (x.ctx == null) continue;
            try mlx.check(mlx.mlx_array_eval(x));
            const n = mlx.mlx_array_size(x);
            const src = mlx.mlx_array_data_float32(x) orelse return error.MlxArrayDataNull;
            const data = try self.a.dupe(f32, src[0..n]);
            errdefer self.a.free(data);
            const shape = try self.a.alloc(usize, mlx.mlx_array_ndim(x));
            errdefer self.a.free(shape);
            for (shape, mlx.getShape(x)) |*o, d| o.* = @intCast(d);
            const name = if (head) try self.a.dupe(u8, "head_in") else try std.fmt.allocPrint(self.a, "layers.{d}.{s}", .{ self.layer, @tagName(kind) });
            errdefer self.a.free(name);
            try entries.append(self.a, .{ .name = name, .shape = shape, .data = data, .rows = self.rows[k] });
        }
        self.resetSums();
        try self.folder.submit(.{ .layer = if (head) null else self.layer, .from = from, .to = to, .entries = try entries.toOwnedSlice(self.a) });
    }

    /// The forward's tap, bound to this capture.
    /// The forward's tap, bound to this capture where it now lives; the model holds its address.
    pub fn bind(self: *Capture) *const fwd.Tap {
        self.tap_value = .{ .ctx = self, .input = Bind.input, .routes = Bind.routes, .expert_act = Bind.expertAct };
        return &self.tap_value;
    }

    const Bind = struct {
        fn input(ctx: *anyopaque, layer: usize, kind: fwd.TapKind, x: Arr) anyerror!void {
            const self: *Capture = @ptrCast(@alignCast(ctx));
            try self.input(layer, std.meta.stringToEnum(Kind, @tagName(kind)) orelse return error.Hy4CaptureUnknownTap, x);
        }
        fn routes(ctx: *anyopaque, layer: usize, picks: []const u32, weights: Arr) anyerror!void {
            const self: *Capture = @ptrCast(@alignCast(ctx));
            try self.routes(layer, picks, weights);
        }
        /// The down inputs feed no file: the encoder recomputes them from `mlp-in` and the source experts.
        fn expertAct(ctx: *anyopaque, layer: usize, act: Arr, experts: []const u32) anyerror!void {
            _ = .{ ctx, layer, act, experts };
        }
    };
};

// ── the run ──────────────────────────────────────────────────────────────────────────────────────

const Manifest = struct {
    schema: []const u8 = "sushi-hy4-layer-major-capture-v1",
    model: []const u8,
    config_sha256: []const u8,
    index_sha256: []const u8,
    tokenizer_sha256: []const u8,
    shard_stat_sha256: []const u8,
    prompts: []const u8,
    inputs_sha256: []const u8,
    prompt_count: usize,
    chunk: usize,
    top_k: u32,
    label: []const u8,
    engine_sha256: []const u8,
    /// Absent for the default outputs, so earlier runs' manifests still match; a resume in another mode is refused.
    mode: ?[]const u8 = null,
};

const Layout = struct {
    schema: []const u8 = "sushi-hy4-calibration-v1",
    layers: u32,
    dense_layers: u32,
    hidden: u32,
    latent: u32,
    experts: u32,
    top_k: u32,
    /// Present only when the run wrote rows, routes and sums (it is not lean).
    mlp_in: ?[]const u8,
    route_ids: ?[]const u8,
    route_weights: ?[]const u8,
    tokens: []const u8 = "tokens.bin: u32 ids of every window in order; ../windows.jsonl holds each window's length",
    /// Present only when the run wrote `head-in.f32`.
    head_rows: ?[]const u8 = null,
    trunk_sums: ?[]const u8,
};

const route_ids_text = "route-LL.u16: [tokens, top_k] global expert ids, MoE layers only";
const route_weights_text = "route-w-LL.f32: [tokens, top_k] the weights the layer applied (normalized, times the routed scale)";
const trunk_sums_text = "trunk-LL.safetensors, trunk-head.safetensors: layers.LL.<input> = sum over rows of x x^T in f32 ([in, in]) beside <name>.rows (I64); inputs: attn_in (q_a, kv_a, linear_gate, indexer token key and head weights), q_lora (q_b, indexer query), kv_latent (kv_b), o_in (o), ffn_in (lat_down, router, shared gate/up; layer 0 dense gate/up), lat_up_in (lat_up), shared_act (shared down; layer 0 dense down), engram_in (engram value and key projections), head_in (lm_head)";

/// What the finished capture holds; a lean run lists only what it wrote.
fn layoutOf(c: *const hy4_config.Config, bf16: bool, mode: Mode) Layout {
    const full = !mode.lean;
    return .{
        .head_rows = if (mode.head_rows) "head-in.f32: [tokens, hidden] f32 little-endian, tokens in window order; row r is the final norm's output after the head's collapse, in f32 as the forward makes it, which the head multiplies: logits = head @ row" else null,
        .layers = c.layers,
        .dense_layers = c.dense_layers,
        .hidden = c.dim,
        .latent = c.moe_inter,
        .experts = c.experts,
        .top_k = c.top_k,
        .mlp_in = if (!full) null else if (bf16)
            "mlp-in-LL.bf16: [tokens, latent] bf16 little-endian, the latent row lat_down(ffn_in) the routed experts read (BF16 products: its values are BF16's, stored whole), MoE layers only, tokens in window order"
        else
            "mlp-in-LL.f32: [tokens, latent] f32 little-endian, the latent row lat_down(ffn_in) the routed experts read, MoE layers only, tokens in window order",
        .route_ids = if (full) route_ids_text else null,
        .route_weights = if (full) route_weights_text else null,
        .trunk_sums = if (full) trunk_sums_text else null,
    };
}

/// The layer-major capture of `l`'s prompts into `opts.out_dir`, `batch` windows at a time, prefilled in `chunk`-row
/// pieces as the window-major capture prefills them; `mode` adds the head's input rows and can keep only those.
pub fn runLoaded(a: std.mem.Allocator, io: std.Io, l: *kld.Loaded, opts: kld.Options, batch: u32, chunk: usize, out: *kld.Out, mode: Mode) !void {
    _ = try mode.check();
    const m = l.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const c = &m.config;
    const bf16 = fwd.products == .bf16;
    // Each window starts from no state, and the teacher's latent is BF16 (`accepts` refuses a quantized one).
    m.reset();
    m.req().kv = @import("../kv_quant.zig").KVQuantConfig.dense;
    const cwd = std.Io.Dir.cwd();
    if (cwd.access(io, opts.out_dir, .{})) |_| return error.Hy4CaptureOutputExists else |err| if (err != error.FileNotFound) return err;
    const staging = try std.fmt.allocPrint(a, "{s}.partial", .{opts.out_dir});
    defer a.free(staging);
    const calib_dir = try std.fmt.allocPrint(a, "{s}/calib", .{staging});
    defer a.free(calib_dir);

    var prompts = try fixture_mod.loadPrompts(a, io, opts.prompts, opts.limit);
    defer prompts.deinit();
    if (prompts.items.len == 0) return error.NoPromptsFound;
    const inputs = try a.alloc([]u32, prompts.items.len);
    defer a.free(inputs);
    var encoded: usize = 0;
    defer for (inputs[0..encoded]) |ids| a.free(ids);
    for (prompts.items, inputs) |p, *ids| {
        ids.* = if (p.ids) |given| try a.dupe(u32, given) else try l.tok.encode(a, p.text);
        encoded += 1;
        if (ids.len == 0) return error.EmptyPrompt;
        for (ids.*) |id| if (id >= c.vocab) return error.InvalidHy4Prompt;
        if (ids.len > c.max_pos) return error.Hy4ContextExceeded;
    }

    const config_sha = try util.metadataHash(a, io, opts.model_dir, "config.json");
    const stat_sha = try calib.sourceStatSha(a, io, opts.model_dir);
    const inputs_sha = util.inputsSha(prompts.items, inputs);
    const manifest = try std.json.Stringify.valueAlloc(a, Manifest{
        .model = opts.model_dir,
        .config_sha256 = &config_sha,
        .index_sha256 = &try util.metadataHash(a, io, opts.model_dir, "model.safetensors.index.json"),
        .tokenizer_sha256 = &try util.metadataHash(a, io, opts.model_dir, "tokenizer.json"),
        .shard_stat_sha256 = &stat_sha,
        .prompts = opts.prompts,
        .inputs_sha256 = &inputs_sha,
        .prompt_count = prompts.items.len,
        .chunk = chunk,
        .top_k = opts.top_k,
        .label = opts.label,
        .engine_sha256 = &try util.engineSha(a, io),
        .mode = mode.tag(),
    }, .{ .whitespace = .indent_2, .emit_null_optional_fields = false });
    defer a.free(manifest);

    var ledger = try calib.Ledger.open(a, io, staging, calib_dir, manifest);
    defer ledger.close();
    try ledger.verify(prompts.items, inputs);
    var folder = calib.Folder{ .a = a, .io = io, .dir = calib_dir };
    if (!mode.lean) try folder.start();
    defer folder.stop();
    var capture = try Capture.open(a, m.s, c, calib_dir, &folder, bf16, mode);
    defer capture.close();
    m.tap = capture.bind();
    defer m.tap = null;

    var run = Run{ .a = a, .io = io, .m = m, .ledger = &ledger, .capture = &capture, .folder = &folder, .prompts = prompts.items, .inputs = inputs, .chunk = chunk, .out = out, .pause_file = opts.pause_file, .row = try a.alloc(f32, c.vocab) };
    defer a.free(run.row);
    try util.json(a, io, staging, "progress.json", .{ .complete = false, .phase = "validated", .prompts = prompts.items.len });
    var records: std.ArrayList(fixture_mod.PromptRecord) = .empty;
    defer {
        for (records.items) |r| r.deinit(a);
        records.deinit(a);
    }
    const started = std.Io.Timestamp.now(io, .awake);
    try run.capturePrompts(batch, &records);
    ledger.dropStale(prompts.items.len);
    const elapsed = @as(f64, @floatFromInt(started.untilNow(io, .awake).nanoseconds)) / 1e9;
    try fixture_mod.writeBaseline(a, io, staging, .{ .rope = kld.ropeOf(&l.config), .label = opts.label, .model = opts.model_dir, .run = opts.label, .kv_cache_format = "bf16", .inference_profile = "greedy", .prompt_set = opts.prompts, .ssd_budget_gb = kld.capturedSsdBudgetGb(&l.config, opts.ssd_budget_bytes), .tokens_per_prompt = opts.tokens, .top_k = opts.top_k, .elapsed_secs = elapsed }, records.items);
    try util.completeBaseline(a, io, staging, records.items.len, opts.tokens);
    if (!std.mem.eql(u8, &config_sha, &try util.metadataHash(a, io, opts.model_dir, "config.json")) or !std.mem.eql(u8, &stat_sha, &try calib.sourceStatSha(a, io, opts.model_dir))) return error.Hy4CaptureSourceChanged;
    if (!mode.lean) try calib.finalizeSums(a, io, calib_dir, c.layers, prompts.items.len);
    try util.json(a, io, calib_dir, "layout.json", layoutOf(c, bf16, mode));
    try util.json(a, io, staging, "progress.json", .{ .complete = true, .phase = "complete", .completed_prompts = records.items.len, .tokens = ledger.committed_tokens });
    try cwd.renamePreserve(staging, cwd, opts.out_dir, io);
    out.print("[kld] layer-major Hy4-Flash captured {d} windows into {s}\n", .{ records.items.len, opts.out_dir });
}

const Run = struct {
    a: std.mem.Allocator,
    io: std.Io,
    m: *hy4_model.Model,
    ledger: *calib.Ledger,
    capture: *Capture,
    folder: *calib.Folder,
    prompts: []const fixture_mod.Prompt,
    inputs: []const []u32,
    chunk: usize,
    out: *kld.Out,
    pause_file: []const u8,
    row: []f32,

    const Result = struct { record: ?fixture_mod.PromptRecord = null, chosen: u32 = 0, logits_sha256: [64]u8 = undefined };
    const Stats = struct { fill_bytes: u64, fill_ns: u64, wall_ns: u64, pin_ns: u64, extra_ns: u64 };

    /// One window's standard fixture row from its logits [1, vocab].
    fn writeWindow(self: *Run, w: usize, logits: Arr) !Result {
        if (mlx.mlx_array_size(logits) != self.row.len) return error.InvalidHy4Logits;
        var wide = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wide);
        try mlx.check(mlx.mlx_astype(&wide, logits, .float32, self.m.s));
        try mlx.check(mlx.mlx_array_eval(wide));
        @memcpy(self.row, (mlx.mlx_array_data_float32(wide) orelse return error.MlxArrayDataNull)[0..self.row.len]);
        for (self.row) |v| if (!std.math.isFinite(v)) return error.NonfiniteHy4Logits;
        const chosen = util.greedy(self.row);
        const p = self.prompts[w];
        var writer = try fixture_mod.PromptWriter.begin(self.a, self.io, self.ledger.staging, w, p.id);
        defer writer.deinit();
        try writer.appendRow(self.row, chosen);
        const record = try writer.finish(p.text, p.text, self.inputs[w], &[_]u32{chosen});
        return .{ .record = record, .chosen = chosen, .logits_sha256 = util.shaHex(std.mem.sliceAsBytes(self.row)) };
    }

    /// One window's chunk between layers: its streams and Engram rows, and where it sits in the window.
    const Piece = struct { hs: Arr, rows: ?hy4_model.Model.EngramRows, p0: usize, t: usize };

    /// Windows `[first, end)` through every layer, then their heads.
    fn runBatch(self: *Run, first: usize, end: usize, results: []Result) !Stats {
        const a = self.a;
        const m = self.m;
        const c = &m.config;
        const engine = m.experts;
        const fill_bytes = if (engine) |e| e.fill_bytes_total else 0;
        const fill_ns = if (engine) |e| e.fill_ns_total else 0;
        const extra_ns = self.capture.extra_ns;
        var pin_ns: u64 = 0;
        var clock = expert_stream.Clock.init();
        var pieces: std.ArrayList(Piece) = .empty;
        defer {
            for (pieces.items) |*p| {
                _ = mlx.mlx_array_free(p.hs);
                if (p.rows) |*r| r.deinit(m.gpa);
            }
            pieces.deinit(a);
        }
        const counts = try a.alloc(usize, end - first);
        defer a.free(counts);
        for (self.inputs[first..end], counts) |ids, *n| {
            n.* = 0;
            var at: usize = 0;
            while (at < ids.len) : (at += self.chunk) {
                const part = ids[at..@min(ids.len, at + self.chunk)];
                try pieces.ensureUnusedCapacity(a, 1);
                const hs = try m.embedStreams(part);
                errdefer _ = mlx.mlx_array_free(hs);
                const history = try historyIds(a, m, ids[0..at]);
                defer a.free(history);
                pieces.appendAssumeCapacity(.{ .hs = hs, .rows = try m.engramRowsAfter(history, part), .p0 = at, .t = part.len });
                n.* += 1;
            }
        }
        for (0..c.layers) |li| {
            const pin = isMoe(c, li) and engine != null;
            if (pin) {
                const t0 = std.Io.Timestamp.now(self.io, .awake);
                try m.pin(@intCast(li));
                pin_ns += @intCast(t0.untilNow(self.io, .awake).nanoseconds);
            }
            defer if (pin) m.unpin();
            self.capture.begin(li);
            if (per_piece_for_test or !m.ffnBatched(li)) try self.passPieces(li, pieces.items, counts) else try self.passBatched(li, pieces.items, counts);
            try self.capture.end(first, end);
        }
        self.capture.begin(c.layers);
        var k: usize = 0;
        for (counts, results, first..) |n, *res, index| {
            k += n;
            // Each earlier piece reads out nothing, but the tap takes its head rows.
            for (pieces.items[k - n .. k - 1]) |p| _ = mlx.mlx_array_free(try m.readoutStreams(p.hs, .none));
            const logits = try m.readoutStreams(pieces.items[k - 1].hs, .last);
            defer _ = mlx.mlx_array_free(logits);
            res.* = try self.writeWindow(index, logits);
        }
        try self.capture.end(first, end);
        return .{ .fill_bytes = (if (engine) |e| e.fill_bytes_total else 0) - fill_bytes, .fill_ns = (if (engine) |e| e.fill_ns_total else 0) - fill_ns, .wall_ns = clock.lap(), .pin_ns = pin_ns, .extra_ns = self.capture.extra_ns - extra_ns };
    }

    /// Layer `li` through every piece, one piece at a time with its FFN.
    fn passPieces(self: *Run, li: usize, pieces: []Piece, counts: []const usize) !void {
        const m = self.m;
        var k: usize = 0;
        for (counts) |n| {
            for (pieces[k .. k + n]) |*p| {
                try interruptForTest();
                const next = try m.passLayer(li, p.hs, p.rows, @intCast(p.t), @intCast(p.p0 + p.t), @intCast(p.p0));
                _ = mlx.mlx_array_free(p.hs);
                p.hs = next;
            }
            m.releaseLayer(li);
            k += n;
        }
    }

    /// Layer `li` with the FFN of up to `ffn_rows` rows of pieces taken together (`Model.ffnBatch`): each piece's
    /// attention runs on its window's state, and the routed experts read each expert's matrix once for all of them.
    fn passBatched(self: *Run, li: usize, pieces: []Piece, counts: []const usize) !void {
        const m = self.m;
        var fronts: std.ArrayList(hy4_model.Model.Front) = .empty;
        defer {
            for (fronts.items) |f| f.deinit();
            fronts.deinit(self.a);
        }
        const first = try self.a.alloc(usize, pieces.len);
        defer self.a.free(first);
        var rows: usize = 0;
        var k: usize = 0;
        for (counts) |n| {
            for (pieces[k .. k + n], k..) |*p, i| {
                try interruptForTest();
                try fronts.ensureUnusedCapacity(self.a, 1);
                fronts.appendAssumeCapacity(try m.layerFront(li, p.hs, p.rows, @intCast(p.t), @intCast(p.p0 + p.t), @intCast(p.p0)));
                first[fronts.items.len - 1] = i;
                rows += p.t;
            }
            m.releaseLayer(li);
            k += n;
            if (rows >= ffn_rows) try self.finishFfn(li, pieces, &fronts, first, &rows);
        }
        if (fronts.items.len > 0) try self.finishFfn(li, pieces, &fronts, first, &rows);
    }

    /// The FFN of the pending fronts together, then each piece's write-back into its streams.
    fn finishFfn(self: *Run, li: usize, pieces: []Piece, fronts: *std.ArrayList(hy4_model.Model.Front), index: []const usize, rows: *usize) !void {
        const m = self.m;
        const n = fronts.items.len;
        const xs = try self.a.alloc(Arr, 2 * n);
        defer self.a.free(xs);
        const fy = xs[n..];
        @memset(fy, .{ .ctx = null });
        defer for (fy) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        for (fronts.items, xs[0..n]) |f, *x| x.* = f.x;
        try m.ffnBatch(li, xs[0..n], fy);
        for (fronts.items, fy, index[0..n]) |f, y, i| {
            const next = try m.layerBack(f, y);
            _ = mlx.mlx_array_free(pieces[i].hs);
            pieces[i].hs = next;
        }
        for (fronts.items) |f| f.deinit();
        fronts.clearRetainingCapacity();
        rows.* = 0;
    }

    /// Every uncommitted window, a batch at a time.
    fn capturePrompts(self: *Run, batch: u32, records: *std.ArrayList(fixture_mod.PromptRecord)) !void {
        const a = self.a;
        const c = &self.m.config;
        for (self.ledger.records) |r| {
            const nll: f64 = @bitCast(r.nll_bits);
            const id = try a.dupe(u8, r.id);
            errdefer a.free(id);
            try records.ensureUnusedCapacity(a, 1);
            records.appendAssumeCapacity(.{ .id = id, .dir = try a.dupe(u8, r.dir), .prompt_tokens = r.tokens, .generated_tokens = 1, .strict_nll_mean = nll, .strict_perplexity = @exp(nll) });
        }
        try self.capture.truncate(self.ledger.committed_tokens);
        try self.ledger.truncateTokens();
        var next = self.ledger.records.len;
        self.ledger.dropStale(next);
        var remaining: u64 = 0;
        for (self.inputs[next..]) |ids| remaining += ids.len;
        const needed = diskBytes(c, remaining, self.prompts.len - next, self.capture.itemsize(), self.capture.headRowBytes(), self.capture.lean);
        if (@import("../disk_placement.zig").volumeSpace(self.ledger.calib)) |space| if (space.free < needed + (16 << 30)) {
            @import("../log.zig").err("[kld] layer-major: {d} GB still to write, {d} GB free (16 GiB margin)\n", .{ needed / 1_000_000_000, space.free / 1_000_000_000 });
            return error.Hy4CaptureDiskTooSmall;
        };
        const run_clock = expert_stream.Clock.init();
        var run_tokens: u64 = 0;
        const results = try a.alloc(Result, batch);
        defer a.free(results);
        var lines: std.ArrayList(calib.WindowRecord) = .empty;
        defer lines.deinit(a);
        while (next < self.prompts.len) {
            try util.waitWhilePaused(a, self.io, self.pause_file, next, self.out);
            const end = @min(self.prompts.len, next + batch);
            const batch_results = results[0 .. end - next];
            @memset(batch_results, .{});
            defer for (batch_results) |r| if (r.record) |rec| rec.deinit(a);
            const stats = try self.runBatch(next, end, batch_results);
            try self.folder.drain();
            try self.capture.sync();
            lines.clearRetainingCapacity();
            var offset = self.ledger.committed_tokens;
            var tokens: u64 = 0;
            const ids_shas = try a.alloc([64]u8, end - next);
            defer a.free(ids_shas);
            for (batch_results, next.., ids_shas) |*res, i, *ids_sha| {
                const record = res.record orelse return error.Hy4CaptureRowCountMismatch;
                for ([_][]const u8{ "logits.f32", "prompt_tokens.txt", "generated_tokens.txt" }) |leaf| {
                    const path = try std.fmt.allocPrint(a, "{s}/{s}/{s}", .{ self.ledger.staging, record.dir, leaf });
                    defer a.free(path);
                    try util.fsyncPath(path);
                }
                ids_sha.* = util.shaHex(std.mem.sliceAsBytes(self.inputs[i]));
                try lines.append(a, .{ .window = i, .id = record.id, .dir = record.dir, .tokens = self.inputs[i].len, .token_offset = offset, .chosen = res.chosen, .nll_bits = @bitCast(record.strict_nll_mean), .logits_dtype = "float32", .ids_sha256 = ids_sha, .logits_sha256 = &res.logits_sha256 });
                offset += self.inputs[i].len;
                tokens += self.inputs[i].len;
            }
            try self.ledger.commit(self.inputs[next..end], lines.items);
            self.ledger.dropStale(end);
            for (batch_results) |*res| {
                try records.append(a, res.record.?);
                res.record = null;
            }
            const wall = @as(f64, @floatFromInt(stats.wall_ns)) / 1e9;
            const pin = @as(f64, @floatFromInt(stats.pin_ns)) / 1e9;
            const extra = @as(f64, @floatFromInt(stats.extra_ns)) / 1e9;
            const forward = wall - pin - extra;
            run_tokens += tokens;
            remaining -= tokens;
            const rate = @as(f64, @floatFromInt(run_tokens)) / (@as(f64, @floatFromInt(run_clock.start.untilNow(run_clock.io, .boot).nanoseconds)) / 1e9);
            try util.json(a, self.io, self.ledger.staging, "progress.json", .{ .complete = false, .phase = "layer-major", .committed_windows = end, .total_windows = self.prompts.len, .batch_windows = batch, .batch_seconds = wall, .batch_forward_seconds = forward, .batch_pin_seconds = pin, .batch_extra_seconds = extra, .batch_fill_seconds = @as(f64, @floatFromInt(stats.fill_ns)) / 1e9, .batch_fill_bytes = stats.fill_bytes, .batch_tokens = tokens, .tokens_per_second = rate, .eta_hours = @as(f64, @floatFromInt(remaining)) / rate / 3600 });
            self.out.print("[kld] layer-major windows {d}..{d} of {d}: {d} tokens in {d:.1} s ({d:.2} tok/s), read {d:.1} GB; forward {d:.1} s, expert pin {d:.1} s, capture extras {d:.1} s; {d:.2} h left\n", .{ next, end, self.prompts.len, tokens, wall, @as(f64, @floatFromInt(tokens)) / wall, @as(f64, @floatFromInt(stats.fill_bytes)) / 1e9, forward, pin, extra, @as(f64, @floatFromInt(remaining)) / rate / 3600 });
            next = end;
        }
    }
};

/// The Engram history of a window's later chunk: its earlier ids as the model holds them (`Model.historyId`).
fn historyIds(a: std.mem.Allocator, m: *const hy4_model.Model, ids: []const u32) ![]u32 {
    const out = try a.alloc(u32, ids.len);
    for (ids, out) |id, *d| d.* = m.historyId(id);
    return out;
}

var interrupt_layers_for_test: ?usize = null;

/// A test's reference: every layer through `passLayer` a piece at a time, as the run did before the FFN was batched.
var per_piece_for_test: bool = false;

fn interruptForTest() !void {
    if (@import("builtin").is_test) if (interrupt_layers_for_test) |*left| {
        if (left.* == 0) return error.TestInterrupted;
        left.* -= 1;
    };
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;

test "hy4 layer-major bill: a window carries its four f32 streams and its Engram rows, the sums are the widest layer's, and the budget buys the batch" {
    const c = try hy4_config.parseText(hy4_config.release_file);
    // 4 x 3072 f32 streams and 4 modules x 16 row ids and a live flag: 49,412 B a token.
    try testing.expectEqual(@as(u64, 2048 * 49_412), carriedPerWindow(&c, 2048));
    // A sliding layer with Engram (11): attn_in, q_lora, kv_latent, o_in (64 x 256), ffn_in, lat_up_in, shared_act,
    // engram_in; layer 0, dense: its 13,824-wide activation; the head: 3072 x 3072.
    const sq = struct {
        fn f(w: u64) u64 {
            return 4 * w * w;
        }
    }.f;
    try testing.expectEqual(sq(3072) * 2 + sq(1024) + sq(512) + sq(16384) + sq(1536) * 3, layerSumsBytes(&c, 11));
    try testing.expectEqual(sq(3072) * 2 + sq(1024) + sq(512) + sq(8192) + sq(13824), layerSumsBytes(&c, 0));
    try testing.expectEqual(sq(3072), layerSumsBytes(&c, c.layers));
    const per_expert: u64 = 14_155_776;
    const floor = try floorBudget(&c, 14 << 30, per_expert);
    const window: u64 = 3 << 30;
    try testing.expectEqual(batch_cap, try batchWindows(&c, 200 << 30, floor, 2048, window));
    try testing.expectError(error.Hy4CaptureBudgetTooSmall, batchWindows(&c, floor, floor, 2048, window));
    const half = floor + reserveBytes(&c, 2048, window) + carriedPerWindow(&c, 2048) * (batch_cap / 2);
    try testing.expectEqual(batch_cap / 2, try batchWindows(&c, half, floor, 2048, window));
    try testing.expectEqual(batch_cap / 2 - 1, try batchWindows(&c, half - 1, floor, 2048, window));
    try checkBatch(&c, half, floor, 2048, window, batch_cap / 2);
    try testing.expectError(error.Hy4CaptureBudgetExceeded, checkBatch(&c, half, floor, 2048, window, batch_cap / 2 + 1));
    // The head's input rows are one f32 row of 3072 a token.
    try testing.expectEqual(@as(u64, 1000 * 3072 * 4), diskBytes(&c, 1000, 3, 4, 3072 * 4, false) - diskBytes(&c, 1000, 3, 4, 0, false));
    // A lean run writes the token ids, the head rows and the logits, no rows, routes or sums.
    try testing.expectEqual(@as(u64, 1000 * (3072 * 4 + 4) + 3 * c.vocab * 4), diskBytes(&c, 1000, 3, 4, 3072 * 4, true));
}

/// A small geometry the writer test feeds by hand: 3 layers (a dense one, two MoE), 8 experts, top 2.
fn writerConfig() hy4_config.Config {
    var c: hy4_config.Config = .{ .vocab = 64, .dim = 16, .layers = 3, .dense_layers = 1, .dense_inter = 24, .sparse_heads = 2, .sliding_heads = 4, .q_lora = 8, .kv_lora = 8, .v_dim = 4, .experts = 8, .top_k = 2, .moe_inter = 12, .hc_mult = 4 };
    c.kinds[1] = .sparse;
    c.engram.n = 1;
    c.engram.layers[0] = 1;
    c.engram.heads = 2;
    c.engram.max_ngram = 3;
    c.engram.head_dim = 4;
    return c;
}

fn seeded(a: std.mem.Allocator, n: usize, seed: u64) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const out = try a.alloc(f32, n);
    for (out) |*v| v.* = prng.random().float(f32) * 2 - 1;
    return out;
}

/// The f64 sum of x x^T over the rows of each batch's `x` [rows, w].
fn gram(a: std.mem.Allocator, xs: []const []const f32, w: usize) ![]f64 {
    const g = try a.alloc(f64, w * w);
    @memset(g, 0);
    for (xs) |x| for (0..x.len / w) |r| for (0..w) |i| for (0..w) |j| {
        g[i * w + j] += @as(f64, x[r * w + i]) * x[r * w + j];
    };
    return g;
}

fn readSums(a: std.mem.Allocator, io: std.Io, path: []const u8, name: []const u8, rows: *u64) ![]f32 {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24));
    defer a.free(raw);
    const hlen = std.mem.readInt(u64, raw[0..8], .little);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, raw[8 .. 8 + hlen], .{});
    defer parsed.deinit();
    const body = raw[8 + hlen ..];
    const off = parsed.value.object.get(name).?.object.get("data_offsets").?.array.items;
    const bytes = body[@intCast(off[0].integer)..@intCast(off[1].integer)];
    const out = try a.alloc(f32, bytes.len / 4);
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    var key: [96]u8 = undefined;
    const roff = parsed.value.object.get(try std.fmt.bufPrint(&key, "{s}.rows", .{name})).?.object.get("data_offsets").?.array.items;
    rows.* = std.mem.readInt(u64, body[@intCast(roff[0].integer)..][0..8], .little);
    return out;
}

test "hy4 layer-major calib writer: two batches give the rows, routes and head rows in window order and the sums of every row, and a resume truncates an interrupted batch" {
    writerCase() catch |e| {
        std.debug.print("\nwriter: {t}\n", .{e});
        if (@errorReturnTrace()) |tr| std.debug.dumpStackTrace(tr.*);
        return e;
    };
}

fn writerCase() !void {
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const c = writerConfig();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try a.dupe(u8, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
    defer a.free(dir);
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Two batches of 5 and 3 rows: each MoE layer's ffn_in, picks and experts' input rows (BF16 values in f32, as the
    // BF16 products give them), and the head's input.
    const batches = [_]usize{ 5, 3 };
    var ffn: [2][]f32 = undefined;
    var lat: [2][]f32 = undefined;
    var head: [2][]f32 = undefined;
    var picks: [2][]u32 = undefined;
    var wts: [2][]f32 = undefined;
    for (batches, 0..) |rows, b| {
        ffn[b] = try seeded(arena, rows * c.dim, 10 + b);
        lat[b] = try seeded(arena, rows * c.moe_inter, 20 + b);
        for (lat[b]) |*v| v.* = @bitCast(@as(u32, @bitCast(v.*)) & 0xffff_0000);
        head[b] = try seeded(arena, rows * c.dim, 30 + b);
        wts[b] = try seeded(arena, rows * c.top_k, 40 + b);
        picks[b] = try arena.alloc(u32, rows * c.top_k);
        for (picks[b], 0..) |*p, i| p.* = @intCast((i * 3 + b) % c.experts);
    }
    const arr = struct {
        fn f(x: anytype, shape: []const c_int, dtype: mlx.mlx_dtype) Arr {
            return mlx.mlx_array_new_data(x.ptr, shape.ptr, @intCast(shape.len), dtype);
        }
    }.f;

    const feed = struct {
        fn batch(cap: *Capture, cfg: *const hy4_config.Config, b: usize, rows: usize, f: []f32, l: []f32, h: []f32, p: []u32, w: []f32, from: usize, to: usize) !void {
            const r: c_int = @intCast(rows);
            for (0..cfg.layers) |li| {
                cap.begin(li);
                if (li >= cfg.dense_layers) {
                    const x = arr(f, &.{ r, @intCast(cfg.dim) }, .float32);
                    defer _ = mlx.mlx_array_free(x);
                    try cap.input(li, .ffn_in, x);
                    const wa = arr(w, &.{ r, @intCast(cfg.top_k) }, .float32);
                    defer _ = mlx.mlx_array_free(wa);
                    try cap.routes(li, p, wa);
                    const la = arr(l, &.{ r, @intCast(cfg.moe_inter) }, .float32);
                    defer _ = mlx.mlx_array_free(la);
                    try cap.input(li, .expert_in, la);
                }
                try cap.end(from, to);
            }
            cap.begin(cfg.layers);
            const x = arr(h, &.{ r, @intCast(cfg.dim) }, .float32);
            defer _ = mlx.mlx_array_free(x);
            try cap.input(cfg.layers, .head_in, x);
            try cap.end(from, to);
            _ = b;
        }
    };

    {
        var folder = calib.Folder{ .a = a, .io = io, .dir = dir };
        try folder.start();
        defer folder.stop();
        var cap = try Capture.open(a, s, &c, dir, &folder, true, .{ .head_rows = true });
        defer cap.close();
        try feed.batch(&cap, &c, 0, 5, ffn[0], lat[0], head[0], picks[0], wts[0], 0, 1);
        try folder.drain();
        try cap.sync();
        // An interrupted second batch: its rows reach the files, its sums never commit.
        try feed.batch(&cap, &c, 1, 3, ffn[1], lat[1], head[1], picks[1], wts[1], 1, 2);
        try folder.drain();
        // A latent value no BF16 holds is refused, not rounded.
        cap.begin(1);
        var off: [12]f32 = @splat(1.0 + 1.0 / 1024.0);
        const bad = arr(&off, &.{ 1, 12 }, .float32);
        defer _ = mlx.mlx_array_free(bad);
        try testing.expectError(error.Hy4CaptureLatentNotBf16, cap.input(1, .expert_in, bad));
        // A head row that is not f32 is refused, not widened or rounded into the file.
        cap.begin(c.layers);
        var narrow = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(narrow);
        const wide = arr(head[0], &.{ 5, @intCast(c.dim) }, .float32);
        defer _ = mlx.mlx_array_free(wide);
        try mlx.check(mlx.mlx_astype(&narrow, wide, .bfloat16, s));
        try testing.expectError(error.Hy4CaptureHeadRowsDtype, cap.input(c.layers, .head_in, narrow));
    }

    {
        var folder = calib.Folder{ .a = a, .io = io, .dir = dir };
        try folder.start();
        defer folder.stop();
        var cap = try Capture.open(a, s, &c, dir, &folder, true, .{ .head_rows = true });
        defer cap.close();
        try cap.truncate(5);
        try feed.batch(&cap, &c, 1, 3, ffn[1], lat[1], head[1], picks[1], wts[1], 1, 2);
        try folder.drain();
        try cap.sync();
    }
    try calib.finalizeSums(a, io, dir, c.layers, 2);

    const read = struct {
        fn f(al: std.mem.Allocator, d: std.Io.Dir, name: []const u8) ![]u8 {
            return d.readFileAlloc(std.Io.Threaded.global_single_threaded.io(), name, al, .limited(1 << 20));
        }
    }.f;
    // The experts' input rows, BF16 halves of the f32 values, in window order.
    const mlp_in = try read(arena, tmp.dir, "mlp-in-01.bf16");
    try testing.expectEqual((lat[0].len + lat[1].len) * 2, mlp_in.len);
    for (0..mlp_in.len / 2) |i| {
        const v = if (i < lat[0].len) lat[0][i] else lat[1][i - lat[0].len];
        try testing.expectEqual(@as(u16, @intCast(@as(u32, @bitCast(v)) >> 16)), std.mem.readInt(u16, mlp_in[i * 2 ..][0..2], .little));
    }
    const route = try read(arena, tmp.dir, "route-01.u16");
    try testing.expectEqual((5 + 3) * c.top_k * 2, route.len);
    for (0..route.len / 2) |i| {
        const p = if (i < picks[0].len) picks[0][i] else picks[1][i - picks[0].len];
        try testing.expectEqual(@as(u16, @intCast(p)), std.mem.readInt(u16, route[i * 2 ..][0..2], .little));
    }
    const route_w = try read(arena, tmp.dir, "route-w-01.f32");
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(wts[0]), route_w[0 .. wts[0].len * 4]);
    try testing.expectEqual((5 + 3) * c.top_k * 4, route_w.len);
    // Each MoE layer has its own files.
    try testing.expectEqualSlices(u8, route, try read(arena, tmp.dir, "route-02.u16"));

    // The head's input rows, f32 whole, in window order: the interrupted batch's rows went with its truncate.
    const head_in = try read(arena, tmp.dir, "head-in.f32");
    try testing.expectEqual((5 + 3) * c.dim * 4, head_in.len);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(head[0]), head_in[0 .. head[0].len * 4]);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(head[1]), head_in[head[0].len * 4 ..]);

    for ([_]struct { file: []const u8, name: []const u8, xs: [2][]f32 }{
        .{ .file = "trunk-01.safetensors", .name = "layers.1.ffn_in", .xs = ffn },
        .{ .file = "trunk-head.safetensors", .name = "head_in", .xs = head },
    }) |want| {
        var rows: u64 = 0;
        const got = try readSums(arena, io, try std.fmt.allocPrint(arena, "{s}/{s}", .{ dir, want.file }), want.name, &rows);
        try testing.expectEqual(@as(u64, 8), rows);
        const ref = try gram(arena, &want.xs, c.dim);
        var peak: f64 = 0;
        for (ref) |r| peak = @max(peak, @abs(r));
        // A test process's GEMM may take TF32 tiles; a batch missing or summed twice is off by its whole share.
        for (got, ref) |g, r| try testing.expect(@abs(g - r) <= 1e-2 * peak);
    }
}

test "hy4 layer-major capture equals the window-major forward byte for byte at several batch sizes and chunks, resumes to the same bytes, and its routes count tokens x top-k (HY4_BF16_TINY)" {
    tinyLayerMajor() catch |e| {
        if (e == error.SkipZigTest) return e;
        var said: [512]u8 = undefined;
        std.debug.print("\nlayer-major: {t}: {s}\n", .{ e, mlx.takeError(&said) orelse "(no message)" });
        if (@errorReturnTrace()) |tr| std.debug.dumpStackTrace(tr.*);
        return e;
    };
}

test "hy4 layer-major window: a forward of the longest window's rows from no state at BF16 KV asks the pool no more past its state than the window the capture measures on its load, dry (D104, HY4_BF16_TINY)" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const t = try Tiny.open(a, io);
    defer t.close(a);
    const m = t.loaded.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const longest = std.mem.max(usize, &tiny_lengths);
    const ids = for (t.windows) |w| {
        if (w.len == longest) break w;
    } else unreachable;
    const window = try windowBytes(m, a, longest, kld.promptChunk(&t.loaded.config));
    defer m.reset();
    m.reset();
    m.req().kv = @import("../kv_quant.zig").KVQuantConfig.dense;
    try m.reserveFor(longest);
    mlx.engine.close();
    const in0 = mlx.engine.poolTotal() - mlx.engine.poolFree();
    mlx.engine.resetPoolPeak(.process);
    _ = mlx.mlx_array_free(try m.forwardReadout(ids, .none));
    mlx.engine.close();
    const asked = mlx.engine.poolPeak(.process) -| in0;
    errdefer std.debug.print("[hy4-window] a {d}-row forward asked {d} B past its state, the measured window {d} B\n", .{ longest, asked, window });
    try testing.expect(asked > 0);
    try testing.expect(asked <= window);
}

/// The tiny's windows: lengths across the window (8), a group (4) and the selection threshold (16).
const tiny_lengths = [_]usize{ 5, 13, 12, 3, 17, 7, 9 };

/// The tiny's loaded model, its windows written as a prompts file, and the options a capture of them takes.
const Tiny = struct {
    arena: std.heap.ArenaAllocator,
    tmp: testing.TmpDir,
    root: []const u8,
    cfg: model.ModelConfig,
    loaded: *kld.Loaded,
    opts: kld.Options,
    tokens: u64,
    windows: [tiny_lengths.len][]u32,

    fn open(a: std.mem.Allocator, io: std.Io) !*Tiny {
        @import("hy4_held.zig").releaseHeld();
        const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
        try mlx.requireGpu();
        const self = try a.create(Tiny);
        errdefer a.destroy(self);
        self.arena = std.heap.ArenaAllocator.init(a);
        errdefer self.arena.deinit();
        const arena = self.arena.allocator();
        self.tmp = testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        self.root = try arena.dupe(u8, path_buf[0..try self.tmp.dir.realPath(io, &path_buf)]);
        var prng = std.Random.DefaultPrng.init(11);
        var body: std.Io.Writer.Allocating = .init(arena);
        self.tokens = 0;
        for (&self.windows, tiny_lengths, 0..) |*w, n, i| {
            w.* = try arena.alloc(u32, n);
            for (w.*) |*id| id.* = prng.random().uintLessThan(u32, 200);
            try body.writer.print("{{\"id\":\"w{d}\",\"prompt_ids\":[", .{i});
            for (w.*, 0..) |id, k| try body.writer.print("{s}{d}", .{ if (k == 0) "" else ",", id });
            try body.writer.writeAll("]}\n");
            self.tokens += n;
        }
        try self.tmp.dir.writeFile(io, .{ .sub_path = "prompts.jsonl", .data = body.written() });
        const prompts = try std.fmt.allocPrint(arena, "{s}/prompts.jsonl", .{self.root});

        const kc = @import("hy4_kld_capture.zig");
        self.cfg = try model.parseConfig(io, a, dir);
        errdefer self.cfg.deinit(a);
        self.opts = .{ .command = .capture, .model_dir = dir, .prompts = prompts, .tokens = 1, .no_template = true, .layer_major = true, .ssd_budget_bytes = 0 };
        var load = self.opts;
        var lcfg = self.cfg;
        lcfg.expert_union_within_cache = false;
        load.ssd_budget_bytes = try kc.captureFloor(io, a, &lcfg, dir);
        self.loaded = try kld.loadModel(io, a, load);
        return self;
    }

    fn close(self: *Tiny, a: std.mem.Allocator) void {
        self.loaded.deinit();
        self.cfg.deinit(a);
        self.tmp.cleanup();
        self.arena.deinit();
        a.destroy(self);
    }
};

fn tinyLayerMajor() !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const tiny = try Tiny.open(a, io);
    defer tiny.close(a);
    const arena = tiny.arena.allocator();
    const root = tiny.root;
    const windows = tiny.windows;
    const tokens = tiny.tokens;
    const cfg = tiny.cfg;
    const opts = tiny.opts;
    const loaded = tiny.loaded;
    const m = loaded.xfm.arch_state.hy4flash.model.?;
    var quiet: kld.Out = .{ .silent = true };

    // The window-major logits of each window at chunk width `chunk`.
    const reference = struct {
        fn logits(al: std.mem.Allocator, mm: *hy4_model.Model, ids: []const u32, chunk: usize) ![]u8 {
            mm.reset();
            mm.req().kv = @import("../kv_quant.zig").KVQuantConfig.dense;
            var at: usize = 0;
            while (at + chunk < ids.len) : (at += chunk) _ = mlx.mlx_array_free(try mm.forwardReadout(ids[at .. at + chunk], .none));
            const last = try mm.forwardReadout(ids[at..], .last);
            defer _ = mlx.mlx_array_free(last);
            return calib.hostBytes(al, mm.s, last);
        }
    };
    for ([_]usize{ 4, 4096 }) |chunk| {
        const want = try arena.alloc([]u8, windows.len);
        for (windows, want) |w, *o| o.* = try reference.logits(arena, m, w, chunk);
        for ([_]u32{ 1, 3 }) |batch| {
            const out_dir = try std.fmt.allocPrint(arena, "{s}/lm-{d}-{d}", .{ root, chunk, batch });
            var run = opts;
            run.out_dir = out_dir;
            try runLoaded(a, io, loaded, run, batch, chunk, &quiet, .{});
            for (want, 0..) |bytes, i| {
                const got = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/prompts/{d:0>2}_w{d}/logits.f32", .{ out_dir, i, i }), arena, .limited(1 << 24));
                try testing.expectEqualSlices(u8, bytes, got);
            }
            // Every MoE layer routed every token to top-k experts.
            for (cfg.hy4flash.?.dense_layers..cfg.hy4flash.?.layers) |li| {
                const route = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/calib/route-{d:0>2}.u16", .{ out_dir, li }), arena, .limited(1 << 24));
                try testing.expectEqual(tokens * cfg.hy4flash.?.top_k * 2, route.len);
            }
        }
    }

    // Interrupted inside its second batch, then rerun at another batch size: the uninterrupted bytes.
    const resumed = try std.fmt.allocPrint(arena, "{s}/resumed", .{root});
    var run = opts;
    run.out_dir = resumed;
    var first_pieces: usize = 0;
    for (tiny_lengths[0..3]) |n| first_pieces += (n + 3) / 4;
    interrupt_layers_for_test = first_pieces * cfg.hy4flash.?.layers + 10;
    try testing.expectError(error.TestInterrupted, runLoaded(a, io, loaded, run, 3, 4, &quiet, .{}));
    interrupt_layers_for_test = null;
    try runLoaded(a, io, loaded, run, 2, 4, &quiet, .{});
    for (0..windows.len) |i| {
        const leaf = try std.fmt.allocPrint(arena, "prompts/{d:0>2}_w{d}/logits.f32", .{ i, i });
        const x = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/{s}", .{ resumed, leaf }), arena, .limited(1 << 24));
        const y = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/lm-4-3/{s}", .{ root, leaf }), arena, .limited(1 << 24));
        try testing.expectEqualSlices(u8, y, x);
    }
    const mlp_in = if (fwd.products == .bf16) "calib/mlp-in-01.bf16" else "calib/mlp-in-01.f32";
    for ([_][]const u8{ "calib/tokens.bin", mlp_in, "calib/route-01.u16", "calib/route-w-01.f32" }) |leaf| {
        const x = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/{s}", .{ resumed, leaf }), arena, .limited(1 << 26));
        const y = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/lm-4-3/{s}", .{ root, leaf }), arena, .limited(1 << 26));
        try testing.expectEqualSlices(u8, y, x);
    }
    // The sums' bytes follow the batch partition (f32 addition order); their values and row counts do not.
    // `head_in` holds every position's final row, the head's input when it serves, not only the row read out.
    const sums = [_]struct { file: []const u8, name: []const u8, rows: u64 }{
        .{ .file = "calib/trunk-01.safetensors", .name = "layers.1.ffn_in", .rows = tokens },
        .{ .file = "calib/trunk-head.safetensors", .name = "head_in", .rows = tokens },
    };
    for (sums) |f| {
        var rows_a: u64 = 0;
        var rows_b: u64 = 0;
        const x = try readSums(arena, io, try std.fmt.allocPrint(arena, "{s}/{s}", .{ resumed, f.file }), f.name, &rows_a);
        const y = try readSums(arena, io, try std.fmt.allocPrint(arena, "{s}/lm-4-3/{s}", .{ root, f.file }), f.name, &rows_b);
        try testing.expectEqual(rows_b, rows_a);
        try testing.expectEqual(f.rows, rows_a);
        var peak: f32 = 0;
        for (y) |v| peak = @max(peak, @abs(v));
        for (x, y) |u, v| try testing.expect(@abs(u - v) <= 1e-5 * peak);
    }
}

/// Every file under `a` is under `b` with the same bytes, and the other way round, bar the run's timings.
fn expectSameTree(arena: std.mem.Allocator, io: std.Io, a: []const u8, b: []const u8, sub: []const u8) !void {
    return expectSameTreeBut(arena, io, a, b, sub, &.{});
}

/// `expectSameTree`, but the files named in `skip` (by leaf name) are not compared.
fn expectSameTreeBut(arena: std.mem.Allocator, io: std.Io, a: []const u8, b: []const u8, sub: []const u8, skip: []const []const u8) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, try std.fmt.allocPrint(arena, "{s}/{s}", .{ a, sub }), .{ .iterate = true });
    defer dir.close(io);
    var count: usize = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        const rel = if (sub.len == 0) try arena.dupe(u8, entry.name) else try std.fmt.allocPrint(arena, "{s}/{s}", .{ sub, entry.name });
        if (entry.kind == .directory) {
            try expectSameTreeBut(arena, io, a, b, rel, skip);
            continue;
        }
        if (std.mem.eql(u8, entry.name, "progress.json") or std.mem.eql(u8, entry.name, "baseline.json")) continue;
        if (for (skip) |name| {
            if (std.mem.eql(u8, entry.name, name)) break true;
        } else false) continue;
        const x = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/{s}", .{ a, rel }), arena, .limited(1 << 26));
        const y = std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/{s}", .{ b, rel }), arena, .limited(1 << 26)) catch |e| {
            std.debug.print("\nmissing under b: {s} ({t})\n", .{ rel, e });
            return e;
        };
        testing.expectEqualSlices(u8, x, y) catch |e| {
            std.debug.print("\ndiffers: {s}\n", .{rel});
            return e;
        };
        count += 1;
    }
    if (sub.len == 0) try testing.expect(count > 0);
}

test "hy4 layer-major batched FFN: a capture is the per-piece FFN's byte for byte in every file, and a capture resumed from one into the other is the same (HY4_BF16_TINY)" {
    batchedParity() catch |e| {
        if (e == error.SkipZigTest) return e;
        var said: [512]u8 = undefined;
        std.debug.print("\nbatched parity: {t}: {s}\n", .{ e, mlx.takeError(&said) orelse "(no message)" });
        if (@errorReturnTrace()) |tr| std.debug.dumpStackTrace(tr.*);
        return e;
    };
}

fn batchedParity() !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const tiny = try Tiny.open(a, io);
    defer tiny.close(a);
    const arena = tiny.arena.allocator();
    const cfg = tiny.cfg.hy4flash.?;
    var quiet: kld.Out = .{ .silent = true };
    defer ffn_rows = 16384;
    const products = fwd.products;
    defer fwd.products = products;

    // The capture's products (f32) and the served ones (BF16), over windows of different lengths, three to a batch, in
    // 4-row and window-wide pieces; a small `ffn_rows` makes a layer's FFN run in several groups.
    for ([_]struct { kind: fwd.Products, chunk: usize, batch: u32, rows: usize }{
        .{ .kind = .f32, .chunk = 4, .batch = 3, .rows = 16384 },
        .{ .kind = .bf16, .chunk = 4096, .batch = 3, .rows = 12 },
        .{ .kind = .f32, .chunk = 4, .batch = 7, .rows = 16384 },
    }) |cell| {
        fwd.products = cell.kind;
        ffn_rows = cell.rows;
        const want = try std.fmt.allocPrint(arena, "{s}/per-piece-{t}-{d}-{d}-{d}", .{ tiny.root, cell.kind, cell.chunk, cell.batch, cell.rows });
        const got = try std.fmt.allocPrint(arena, "{s}/batched-{t}-{d}-{d}-{d}", .{ tiny.root, cell.kind, cell.chunk, cell.batch, cell.rows });
        var run = tiny.opts;
        run.out_dir = want;
        per_piece_for_test = true;
        defer per_piece_for_test = false;
        try runLoaded(a, io, tiny.loaded, run, cell.batch, cell.chunk, &quiet, .{});
        per_piece_for_test = false;
        run.out_dir = got;
        try runLoaded(a, io, tiny.loaded, run, cell.batch, cell.chunk, &quiet, .{});
        try expectSameTree(arena, io, want, got, "");
        try expectSameTree(arena, io, got, want, "");
    }

    // The first batch committed by the per-piece FFN, the rest by the batched one.
    fwd.products = .f32;
    ffn_rows = 16384;
    const want = try std.fmt.allocPrint(arena, "{s}/per-piece-f32-4-3-16384", .{tiny.root});
    const resumed = try std.fmt.allocPrint(arena, "{s}/resumed", .{tiny.root});
    var run = tiny.opts;
    run.out_dir = resumed;
    var first_pieces: usize = 0;
    for (tiny_lengths[0..3]) |n| first_pieces += (n + 3) / 4;
    per_piece_for_test = true;
    interrupt_layers_for_test = first_pieces * cfg.layers + 10;
    try testing.expectError(error.TestInterrupted, runLoaded(a, io, tiny.loaded, run, 3, 4, &quiet, .{}));
    interrupt_layers_for_test = null;
    per_piece_for_test = false;
    try runLoaded(a, io, tiny.loaded, run, 3, 4, &quiet, .{});
    try expectSameTree(arena, io, want, resumed, "");
    try expectSameTree(arena, io, resumed, want, "");
}

/// Every head input a window-major forward reads, as the f32 bytes of its rows, in the order the forward gives them.
const HeadSink = struct {
    a: std.mem.Allocator,
    s: mlx.mlx_stream,
    bytes: std.ArrayList(u8) = .empty,
    tap: fwd.Tap = undefined,

    fn bind(self: *HeadSink) *const fwd.Tap {
        self.tap = .{ .ctx = self, .input = input, .routes = routes, .expert_act = expertAct };
        return &self.tap;
    }

    fn input(ctx: *anyopaque, layer: usize, kind: fwd.TapKind, x: Arr) anyerror!void {
        _ = layer;
        const self: *HeadSink = @ptrCast(@alignCast(ctx));
        if (kind != .head_in) return;
        if (mlx.mlx_array_dtype(x) != .float32) return error.Hy4CaptureHeadRowsDtype;
        const rows = try calib.hostBytes(self.a, self.s, x);
        defer self.a.free(rows);
        try self.bytes.appendSlice(self.a, rows);
    }

    fn routes(ctx: *anyopaque, layer: usize, picks: []const u32, weights: Arr) anyerror!void {
        _ = .{ ctx, layer, picks, weights };
    }

    fn expertAct(ctx: *anyopaque, layer: usize, act: Arr, experts: []const u32) anyerror!void {
        _ = .{ ctx, layer, act, experts };
    }
};

test "hy4 layer-major head rows: head-in.f32 is the head's input of every token in window order, the window-major forward's bytes, the sums do not move with the tap on, and a resume in another mode is refused (HY4_BF16_TINY)" {
    headRowsCell() catch |e| {
        if (e == error.SkipZigTest) return e;
        var said: [512]u8 = undefined;
        std.debug.print("\nhead rows: {t}: {s}\n", .{ e, mlx.takeError(&said) orelse "(no message)" });
        if (@errorReturnTrace()) |tr| std.debug.dumpStackTrace(tr.*);
        return e;
    };
}

fn headRowsCell() !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const tiny = try Tiny.open(a, io);
    defer tiny.close(a);
    const arena = tiny.arena.allocator();
    const cfg = tiny.cfg.hy4flash.?;
    const m = tiny.loaded.xfm.arch_state.hy4flash.model.?;
    var quiet: kld.Out = .{ .silent = true };
    const cwd = std.Io.Dir.cwd();
    const row_bytes = @as(usize, cfg.dim) * 4;

    // Chunk widths of 4 (several pieces a window) and 4096 (one), at batches of 3 and 1.
    for ([_]struct { chunk: usize, batch: u32 }{ .{ .chunk = 4, .batch = 3 }, .{ .chunk = 4096, .batch = 1 } }) |cell| {
        // The window-major forward's head inputs, window after window, from a tap on the same model.
        var sink: HeadSink = .{ .a = a, .s = m.s };
        defer sink.bytes.deinit(a);
        m.tap = sink.bind();
        for (tiny.windows) |ids| {
            m.reset();
            m.req().kv = @import("../kv_quant.zig").KVQuantConfig.dense;
            var at: usize = 0;
            while (at + cell.chunk < ids.len) : (at += cell.chunk) _ = mlx.mlx_array_free(try m.forwardReadout(ids[at .. at + cell.chunk], .none));
            _ = mlx.mlx_array_free(try m.forwardReadout(ids[at..], .last));
        }
        m.tap = null;
        try testing.expectEqual(@as(usize, tiny.tokens) * row_bytes, sink.bytes.items.len);

        const rows_dir = try std.fmt.allocPrint(arena, "{s}/rows-{d}-{d}", .{ tiny.root, cell.chunk, cell.batch });
        const plain_dir = try std.fmt.allocPrint(arena, "{s}/plain-{d}-{d}", .{ tiny.root, cell.chunk, cell.batch });
        var run = tiny.opts;
        run.out_dir = rows_dir;
        try runLoaded(a, io, tiny.loaded, run, cell.batch, cell.chunk, &quiet, .{ .head_rows = true });
        run.out_dir = plain_dir;
        try runLoaded(a, io, tiny.loaded, run, cell.batch, cell.chunk, &quiet, .{});

        const got = try cwd.readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/calib/{s}", .{ rows_dir, head_rows_file }), arena, .limited(1 << 26));
        try testing.expectEqualSlices(u8, sink.bytes.items, got);
        // A run without the switch writes no head rows, and the tap changes no other byte: sums, rows, routes, logits.
        try testing.expectError(error.FileNotFound, cwd.access(io, try std.fmt.allocPrint(arena, "{s}/calib/{s}", .{ plain_dir, head_rows_file }), .{}));
        const skip = [_][]const u8{ head_rows_file, "layout.json", "layer-major.json" };
        try expectSameTreeBut(arena, io, plain_dir, rows_dir, "", &skip);
        try expectSameTreeBut(arena, io, rows_dir, plain_dir, "", &skip);
    }

    // Interrupted inside its second batch, resumed in the same mode at another batch size: the same rows; another mode is refused.
    const resumed = try std.fmt.allocPrint(arena, "{s}/rows-resumed", .{tiny.root});
    var run = tiny.opts;
    run.out_dir = resumed;
    var first_pieces: usize = 0;
    for (tiny_lengths[0..3]) |n| first_pieces += (n + 3) / 4;
    interrupt_layers_for_test = first_pieces * cfg.layers + 10;
    defer interrupt_layers_for_test = null;
    try testing.expectError(error.TestInterrupted, runLoaded(a, io, tiny.loaded, run, 3, 4, &quiet, .{ .head_rows = true }));
    try testing.expectError(error.CalibResumeMismatch, runLoaded(a, io, tiny.loaded, run, 3, 4, &quiet, .{}));
    interrupt_layers_for_test = null;
    try runLoaded(a, io, tiny.loaded, run, 2, 4, &quiet, .{ .head_rows = true });
    const whole = try cwd.readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/rows-4-3/calib/{s}", .{ tiny.root, head_rows_file }), arena, .limited(1 << 26));
    const again = try cwd.readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/calib/{s}", .{ resumed, head_rows_file }), arena, .limited(1 << 26));
    try testing.expectEqualSlices(u8, whole, again);
}

/// The names directly under `path`, sorted.
fn namesIn(arena: std.mem.Allocator, io: std.Io, path: []const u8) ![]const []const u8 {
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |entry| try names.append(arena, try arena.dupe(u8, entry.name));
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    return names.items;
}

fn expectNames(arena: std.mem.Allocator, io: std.Io, path: []const u8, want: []const []const u8) !void {
    const got = try namesIn(arena, io, path);
    try testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try testing.expectEqualStrings(w, g);
}

test "hy4 layer-major lean: a lean run without head rows is refused, and each mode has its manifest tag" {
    try testing.expectError(error.Hy4CaptureLeanNeedsHeadRows, (Mode{ .lean = true }).check());
    try testing.expectEqual(Mode{ .head_rows = true, .lean = true }, try (Mode{ .head_rows = true, .lean = true }).check());
    try testing.expectEqualStrings("head-rows-lean", (Mode{ .head_rows = true, .lean = true }).tag().?);
    try testing.expectEqualStrings("head-rows", (Mode{ .head_rows = true }).tag().?);
    try testing.expectEqual(@as(?[]const u8, null), (Mode{}).tag());
}

test "hy4 layer-major lean writer: only the head rows are written, and no sum is built for any other input" {
    leanWriterCase() catch |e| {
        std.debug.print("\nlean writer: {t}\n", .{e});
        if (@errorReturnTrace()) |tr| std.debug.dumpStackTrace(tr.*);
        return e;
    };
}

fn leanWriterCase() !void {
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const c = writerConfig();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try a.dupe(u8, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
    defer a.free(dir);
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const rows: usize = 5;
    const r: c_int = @intCast(rows);
    const ffn = try seeded(arena, rows * c.dim, 10);
    const lat = try seeded(arena, rows * c.moe_inter, 20);
    for (lat) |*v| v.* = @bitCast(@as(u32, @bitCast(v.*)) & 0xffff_0000);
    const head = try seeded(arena, rows * c.dim, 30);
    const wts = try seeded(arena, rows * c.top_k, 40);
    const picks = try arena.alloc(u32, rows * c.top_k);
    for (picks, 0..) |*p, i| p.* = @intCast((i * 3) % c.experts);
    const arr = struct {
        fn f(x: []const f32, shape: []const c_int) Arr {
            return mlx.mlx_array_new_data(x.ptr, shape.ptr, @intCast(shape.len), .float32);
        }
    }.f;

    var folder = calib.Folder{ .a = a, .io = io, .dir = dir };
    var cap = try Capture.open(a, s, &c, dir, &folder, true, .{ .head_rows = true, .lean = true });
    defer cap.close();
    for (c.dense_layers..c.layers) |li| {
        cap.begin(li);
        const x = arr(ffn, &.{ r, @intCast(c.dim) });
        defer _ = mlx.mlx_array_free(x);
        try cap.input(li, .ffn_in, x);
        const wa = arr(wts, &.{ r, @intCast(c.top_k) });
        defer _ = mlx.mlx_array_free(wa);
        try cap.routes(li, picks, wa);
        const la = arr(lat, &.{ r, @intCast(c.moe_inter) });
        defer _ = mlx.mlx_array_free(la);
        try cap.input(li, .expert_in, la);
        // A gram would be a pending array here; a lean capture builds none.
        for (cap.acc) |g| try testing.expect(g.ctx == null);
        try testing.expectEqual(@as(@TypeOf(cap.rows), @splat(0)), cap.rows);
        try cap.end(0, 1);
    }
    cap.begin(c.layers);
    const hx = arr(head, &.{ r, @intCast(c.dim) });
    defer _ = mlx.mlx_array_free(hx);
    try cap.input(c.layers, .head_in, hx);
    for (cap.acc) |g| try testing.expect(g.ctx == null);
    try cap.end(0, 1);
    try folder.drain();
    try cap.sync();
    try expectNames(arena, io, dir, &.{head_rows_file});
    const got = try tmp.dir.readFileAlloc(io, head_rows_file, arena, .limited(1 << 20));
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(head), got);
}

test "hy4 layer-major lean: head-in.f32, the tokens, the windows and every logits row equal the full run's byte for byte, and calib holds nothing else (HY4_BF16_TINY)" {
    leanCell() catch |e| {
        if (e == error.SkipZigTest) return e;
        var said: [512]u8 = undefined;
        std.debug.print("\nlean: {t}: {s}\n", .{ e, mlx.takeError(&said) orelse "(no message)" });
        if (@errorReturnTrace()) |tr| std.debug.dumpStackTrace(tr.*);
        return e;
    };
}

fn leanCell() !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const tiny = try Tiny.open(a, io);
    defer tiny.close(a);
    const arena = tiny.arena.allocator();
    var quiet: kld.Out = .{ .silent = true };
    const cwd = std.Io.Dir.cwd();
    const full = try std.fmt.allocPrint(arena, "{s}/full", .{tiny.root});
    const lean = try std.fmt.allocPrint(arena, "{s}/lean", .{tiny.root});
    var run = tiny.opts;
    run.out_dir = full;
    try runLoaded(a, io, tiny.loaded, run, 3, 4, &quiet, .{ .head_rows = true });
    run.out_dir = lean;
    try runLoaded(a, io, tiny.loaded, run, 3, 4, &quiet, .{ .head_rows = true, .lean = true });

    // The fixture rows, windows.jsonl and the head rows are the full run's; the manifest and layout name the mode.
    const same = [_][]const u8{ "calib/head-in.f32", "calib/tokens.bin", "windows.jsonl" };
    for (same) |leaf| {
        const x = try cwd.readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/{s}", .{ full, leaf }), arena, .limited(1 << 26));
        const y = try cwd.readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/{s}", .{ lean, leaf }), arena, .limited(1 << 26));
        try testing.expectEqualSlices(u8, x, y);
    }
    try expectSameTree(arena, io, full, lean, "prompts");
    try expectSameTree(arena, io, lean, full, "prompts");
    try expectNames(arena, io, try std.fmt.allocPrint(arena, "{s}/calib", .{lean}), &.{ head_rows_file, "layout.json", "tokens.bin" });

    // layout.json lists what the run holds: the lean one no rows, routes or sums.
    const Held = struct { head_rows: ?[]const u8 = null, mlp_in: ?[]const u8 = null, route_ids: ?[]const u8 = null, route_weights: ?[]const u8 = null, trunk_sums: ?[]const u8 = null };
    for ([_]struct { dir: []const u8, extras: bool }{ .{ .dir = full, .extras = true }, .{ .dir = lean, .extras = false } }) |case| {
        const raw = try cwd.readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/calib/layout.json", .{case.dir}), arena, .limited(1 << 20));
        const held = (try std.json.parseFromSlice(Held, arena, raw, .{ .ignore_unknown_fields = true })).value;
        try testing.expect(held.head_rows != null);
        try testing.expectEqual(case.extras, held.mlp_in != null);
        try testing.expectEqual(case.extras, held.route_ids != null);
        try testing.expectEqual(case.extras, held.route_weights != null);
        try testing.expectEqual(case.extras, held.trunk_sums != null);
    }
}

test "hy4 layer-major lean resume: a lean run interrupted and resumed gives the uninterrupted bytes, a rerun in another mode is refused both ways, and lean without head rows writes nothing (HY4_BF16_TINY)" {
    leanResumeCell() catch |e| {
        if (e == error.SkipZigTest) return e;
        var said: [512]u8 = undefined;
        std.debug.print("\nlean resume: {t}: {s}\n", .{ e, mlx.takeError(&said) orelse "(no message)" });
        if (@errorReturnTrace()) |tr| std.debug.dumpStackTrace(tr.*);
        return e;
    };
}

fn leanResumeCell() !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const tiny = try Tiny.open(a, io);
    defer tiny.close(a);
    const arena = tiny.arena.allocator();
    const cfg = tiny.cfg.hy4flash.?;
    var quiet: kld.Out = .{ .silent = true };
    const cwd = std.Io.Dir.cwd();
    const lean_mode: Mode = .{ .head_rows = true, .lean = true };
    const rows_mode: Mode = .{ .head_rows = true };
    var first_pieces: usize = 0;
    for (tiny_lengths[0..3]) |n| first_pieces += (n + 3) / 4;
    defer interrupt_layers_for_test = null;

    const whole = try std.fmt.allocPrint(arena, "{s}/lean-whole", .{tiny.root});
    var run = tiny.opts;
    run.out_dir = whole;
    try runLoaded(a, io, tiny.loaded, run, 3, 4, &quiet, lean_mode);

    // Interrupted inside its second batch: the other modes are refused, the same mode resumes at another batch size.
    const resumed = try std.fmt.allocPrint(arena, "{s}/lean-resumed", .{tiny.root});
    run.out_dir = resumed;
    interrupt_layers_for_test = first_pieces * cfg.layers + 10;
    try testing.expectError(error.TestInterrupted, runLoaded(a, io, tiny.loaded, run, 3, 4, &quiet, lean_mode));
    try testing.expectError(error.CalibResumeMismatch, runLoaded(a, io, tiny.loaded, run, 3, 4, &quiet, rows_mode));
    try testing.expectError(error.CalibResumeMismatch, runLoaded(a, io, tiny.loaded, run, 3, 4, &quiet, .{}));
    interrupt_layers_for_test = null;
    try runLoaded(a, io, tiny.loaded, run, 2, 4, &quiet, lean_mode);
    for ([_][]const u8{ "calib/head-in.f32", "calib/tokens.bin", "windows.jsonl" }) |leaf| {
        const x = try cwd.readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/{s}", .{ whole, leaf }), arena, .limited(1 << 26));
        const y = try cwd.readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/{s}", .{ resumed, leaf }), arena, .limited(1 << 26));
        try testing.expectEqualSlices(u8, x, y);
    }
    try expectSameTree(arena, io, whole, resumed, "prompts");
    try expectSameTree(arena, io, resumed, whole, "prompts");
    try expectNames(arena, io, try std.fmt.allocPrint(arena, "{s}/calib", .{resumed}), &.{ head_rows_file, "layout.json", "tokens.bin" });

    // A head-rows run interrupted likewise is not resumed lean.
    const other = try std.fmt.allocPrint(arena, "{s}/rows-resumed", .{tiny.root});
    run.out_dir = other;
    interrupt_layers_for_test = first_pieces * cfg.layers + 10;
    try testing.expectError(error.TestInterrupted, runLoaded(a, io, tiny.loaded, run, 3, 4, &quiet, rows_mode));
    try testing.expectError(error.CalibResumeMismatch, runLoaded(a, io, tiny.loaded, run, 3, 4, &quiet, lean_mode));
    interrupt_layers_for_test = null;

    // Lean without head rows is refused before the run creates anything.
    const refused = try std.fmt.allocPrint(arena, "{s}/refused", .{tiny.root});
    run.out_dir = refused;
    try testing.expectError(error.Hy4CaptureLeanNeedsHeadRows, runLoaded(a, io, tiny.loaded, run, 3, 4, &quiet, .{ .lean = true }));
    try testing.expectError(error.FileNotFound, cwd.access(io, refused, .{}));
    try testing.expectError(error.FileNotFound, cwd.access(io, try std.fmt.allocPrint(arena, "{s}.partial", .{refused}), .{}));
}

test "hy4 layer-major window: a window past the widest chunk, which the capture runs in pieces of its prompt chunk, is measured and not refused (HY4_BF16_TINY)" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const t = try Tiny.open(a, io);
    defer t.close(a);
    const m = t.loaded.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const rows = hy4_config.widest_chunk + 1;
    const chunk = kld.promptChunk(&t.loaded.config);
    const window = windowBytes(m, a, rows, chunk) catch |e| {
        var said: [512]u8 = undefined;
        std.debug.print("the capture's bill refuses a window of {d} rows in pieces of {d}: {t} ({s})\n", .{ rows, chunk, e, mlx.takeError(&said) orelse "" });
        return e;
    };
    try testing.expect(window > 0);
}

test "hy4 layer-major window: a window of several pieces is billed by the pieces it runs, and a forward of them asks no more (HY4_BF16_TINY)" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const t = try Tiny.open(a, io);
    defer t.close(a);
    const m = t.loaded.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const chunk = kld.promptChunk(&t.loaded.config);
    const rows = chunk + chunk / 4;
    const ids = try a.alloc(u32, rows);
    defer a.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast((i * 7) % 200);
    const window = try windowBytes(m, a, rows, chunk);
    // One forward of all the rows, which the capture never runs: it asks more than its pieces.
    const hy4_plan = @import("hy4_plan.zig");
    const plan = @import("../tensor/plan/plan_api.zig");
    var passes = try hy4_plan.recordChecked(m, a, .{ .rows = rows, .readout = .none, .keep_hidden = false, .kv = .dense, .phase = .wide }, rows);
    defer passes.deinit();
    var whole: plan.region.Need = .{};
    for (passes.traces) |*tr| whole = whole.max(try plan.region.need(a, tr, plan.region.default_alignment));
    errdefer std.debug.print("a window of {d} rows is billed {d} B, one forward of them {d} B\n", .{ rows, window, whole.scratch + whole.host });
    try testing.expect(window < whole.scratch + whole.host);
    // The pieces as `runBatch` runs them.
    defer m.reset();
    m.reset();
    m.req().kv = @import("../kv_quant.zig").KVQuantConfig.dense;
    try m.reserveFor(rows);
    mlx.engine.close();
    const in0 = mlx.engine.poolTotal() - mlx.engine.poolFree();
    mlx.engine.resetPoolPeak(.process);
    var at: usize = 0;
    while (at < rows) : (at += chunk) _ = mlx.mlx_array_free(try m.forwardReadout(ids[at..@min(rows, at + chunk)], .none));
    mlx.engine.close();
    try testing.expect(mlx.engine.poolPeak(.process) -| in0 <= window);
}

/// A Hy4 load with every expert resident, as a served load without a stream.
const ServedResident = struct {
    pub fn prepare(a: std.mem.Allocator, config: *model.ModelConfig, dir: []const u8) !void {
        config.expert_streaming = false;
        config.expert_source_dir = try a.dupe(u8, dir);
    }
};

/// The region a served load of `dir` measures (`hy4_plan.regionAt` at its chunk).
fn servedRegion(dir: []const u8) !@import("../tensor/plan/plan_api.zig").region.Need {
    @import("hy4_held.zig").releaseHeld();
    const contract = @import("../cooler_contract_test.zig");
    var ld: contract.Load = undefined;
    try ld.openLight(ServedResident, dir, true);
    defer _ = ld.close();
    return ld.xfm.arch_state.hy4flash.model.?.plans.?.region;
}

test "hy4 layer-major: a served load measures the region it measured before a capture load ran in the process (HY4_BF16_TINY)" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
    try mlx.requireGpu();
    const was = fwd.products;
    defer fwd.products = was;
    fwd.products = .bf16;
    const before = try servedRegion(dir);
    {
        // The capture's loads and window run at f32 products.
        const t = try Tiny.open(a, io);
        defer t.close(a);
        try testing.expectEqual(fwd.Products.f32, fwd.products);
    }
    const after = try servedRegion(dir);
    errdefer std.debug.print("a served load measured scratch {d} B, host {d} B before a capture load and {d} B, {d} B after it\n", .{ before.scratch, before.host, after.scratch, after.host });
    try testing.expectEqual(before, after);
}
