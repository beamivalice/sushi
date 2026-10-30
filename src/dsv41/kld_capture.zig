//! Layer-major DeepSeek-V4.1 calibration capture: `sushi kld capture --layer-major --tokens 1` on a `deepseek_v41`
//! checkpoint. A batch of windows runs layer 0, then layer 1, and so on, so the 289 GB of streamed FP4 experts are read
//! once per batch (one pinned union per layer), never once per window. Next to the standard one-row fixture it writes
//! what a quantizer needs, in `<out>/calib/`:
//!   mlp-in-LL.bf16     the FFN input every routed and shared expert of layer LL reads, [tokens, dim] bf16
//!   route-LL.u16       the router's top-k expert ids per token, route-w-LL.f32 the weights it applied
//!   trunk-LL.safetensors  X^T X of the input of every trunk linear of layer LL, f32, summed over all tokens
//!   trunk-head.safetensors  the same for the head
//!   imatrix.safetensors   the per-expert second moments (`SUSHI_IMATRIX_OUT` names another file)
//!   dspark-in-LL.bf16  the mean of the hc copies entering each DSpark target layer, [tokens, dim] bf16
//!   tokens.bin         the window token ids (u32), whose lengths `windows.jsonl` records
//!   head-in.bf16       opt-in (`SUSHI_DSV41_HEAD_ROWS=1`): the array the head multiplies for every row, the final RMSNorm
//!                      output after the last hyper-connection collapse, [tokens, dim] bf16 (the engine's own activation
//!                      dtype, so the rounding is the forward's, not the file's); head(row) is the teacher's logits
//! `SUSHI_DSV41_CAPTURE_LEAN=1` (only with HEAD_ROWS) writes nothing but tokens.bin, windows.jsonl, the one-row fixture,
//! layout.json and head-in.bf16: no mlp-in, route, trunk sums, imatrix or dspark-in, and none of their GPU work.
//! Rows are appended in window order; a window commits (windows.jsonl) only after every file of its batch is synced,
//! and a rerun of the same command truncates to the committed windows and continues.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const kld = @import("../kld.zig");
const dsv = @import("dsv41_model.zig");
const dsv_cfg = @import("dsv41_config.zig");
const dsv_stream = @import("dsv41_stream.zig");
const dsv_source = @import("dsv41_source.zig");
const sc_mod = @import("dsv41_scope.zig");
const imatrix_capture = @import("../imatrix.zig");
const expert_stream = @import("../expert_stream.zig");
const glm = @import("../glm5/glm5_kld_capture.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const none = sc_mod.none;

/// Rows one forward chunk holds: a window of this many tokens or fewer is one forward, as in the window-major capture.
pub const chunk_rows: usize = 2048;
/// Windows a batch holds at most.
const batch_cap: u32 = 32;
const kinds = std.enums.values(dsv.TapKind).len;
const head_layer = std.math.maxInt(usize);

/// What a run writes beyond the standard outputs.
pub const Mode = struct {
    head_rows: bool = false,
    lean: bool = false,

    fn tag(self: Mode) ?[]const u8 {
        if (!self.head_rows) return null;
        return if (self.lean) "head-rows-lean" else "head-rows";
    }
};

fn modeFromEnv() !Mode {
    const diagEnvOn = @import("../transformer.zig").diagEnvOn;
    return checkedMode(diagEnvOn("SUSHI_DSV41_HEAD_ROWS"), diagEnvOn("SUSHI_DSV41_CAPTURE_LEAN"));
}

/// A lean run keeps only the head rows, so without them it would write nothing a quantizer can use.
fn checkedMode(head_rows: bool, lean: bool) !Mode {
    if (lean and !head_rows) return error.Dsv41CaptureLeanNeedsHeadRows;
    return .{ .head_rows = head_rows, .lean = lean };
}

/// DeepSeek-V4.1 keeps its own state in the activation dtype and has no hidden-boundary or imatrix tap:
/// a flag that would silently do nothing is refused by name.
pub fn refuse(config: *const model.ModelConfig, opts: kld.Options) !void {
    if (!config.isDsv41()) return;
    if (opts.kv_quant_config.isQuant()) return error.Dsv41KvQuantUnsupported;
    if (!opts.qwen_gdn_f32) return error.Dsv41HasNoQwenGdnState;
    if (opts.hidden_out.len > 0) return error.Dsv41HiddenCaptureUnsupported;
    if (opts.imatrix_out.len > 0) return error.Dsv41ImatrixCaptureUnsupported;
}

pub fn accepts(cfg: *const model.ModelConfig, opts: kld.Options) !bool {
    if (!cfg.isDsv41() or opts.command != .capture or !opts.layer_major) return false;
    if (opts.kv_quant_config.isQuant()) return error.Dsv41KvQuantUnsupported;
    if (!opts.qwen_gdn_f32) return error.Dsv41HasNoQwenGdnState;
    if (opts.hidden_out.len > 0) return error.Dsv41HiddenCaptureUnsupported;
    if (opts.imatrix_windows != 0) return error.Dsv41ImatrixWindowsUnsupported;
    if (opts.tokens != 1) return error.Dsv41LayerMajorNeedsOneRow;
    if (!opts.no_template or opts.ssd_budget_bytes == 0 or opts.pick_tolerance != 0 or opts.enable_mtp or opts.wired_margin_bytes != 0) return error.Dsv41CaptureNeedsLosslessStreaming;
    return true;
}

pub fn tryRun(a: std.mem.Allocator, io: std.Io, opts: kld.Options, out: *kld.Out) !bool {
    if (opts.command != .capture or !opts.layer_major) return false;
    var cfg = try model.parseConfig(io, a, opts.model_dir);
    defer cfg.deinit(a);
    if (!try accepts(&cfg, opts)) return false;
    _ = try modeFromEnv();
    const dc = cfg.dsv41.?;
    const per_expert = try dsv_stream.expertBytes(&dc);
    const trunk = try dsv_source.residentBytes(io, a, opts.model_dir);
    // The batch pins one layer's whole union at a time, so the cache itself stays at its two-slot floor.
    const floor = try floorBudget(&dc, trunk, per_expert);
    const load = loadOptions(opts, floor);
    const batch = if (opts.batch_windows != 0) opts.batch_windows else try batchWindows(&dc, opts.ssd_budget_bytes, floor, chunk_rows);
    const loaded = try kld.loadModel(io, a, load);
    defer loaded.deinit();
    try runLoaded(a, io, loaded, opts, batch, chunk_rows, out);
    return true;
}

/// The capture owns the imatrix itself, which the generic DeepSeek load refuses to arm.
fn loadOptions(opts: kld.Options, floor: u64) kld.Options {
    var load = opts;
    load.ssd_budget_bytes = floor;
    load.imatrix_out = "";
    return load;
}

/// The cheapest budget the expert engine accepts: trunk, the all-experts workspace, one token's picks, the bounce
/// buffers and two slots a layer.
fn floorBudget(dc: *const dsv_cfg.Config, trunk: u64, per_expert: u64) !u64 {
    const workspace = try std.math.mul(u64, dc.experts, per_expert);
    const selected = try std.math.mul(u64, dc.top_k, per_expert);
    const cache = try std.math.mul(u64, 2 * @as(u64, dc.layers), per_expert);
    return trunk + workspace + selected + cache + expert_stream.BOUNCE_BYTES + (1 << 30);
}

/// Bytes of the trunk sums one layer holds on the GPU while a batch runs (a window's term and the running total).
fn sumsBytes(dc: *const dsv_cfg.Config) u64 {
    const d: u64 = dc.dim;
    const og: u64 = dc.o_groups;
    const gin: u64 = @as(u64, dc.heads) * dc.head_dim / dc.o_groups;
    const wide: u64 = og * dc.o_lora;
    const e: u64 = @as(u64, dc.engram.cols()) * dc.engram.head_dim;
    return 4 * (2 * d * d + @as(u64, dc.q_lora) * dc.q_lora + @as(u64, dc.index_head_dim) * dc.index_head_dim + og * gin * gin + wide * wide +
        @as(u64, dc.moe_inter) * dc.moe_inter + e * e) * 2;
}

/// What one window of `rows` tokens holds between layers.
fn carriedPerWindow(dc: *const dsv_cfg.Config, rows: usize) u64 {
    const r: u64 = rows;
    return r * dc.hc_mult * dc.dim * 2 * 2 + r * dc.hc_mult * 4 + r * dc.stateBytesPerToken() + dc.stateFixedBytes(false) +
        r * dc.index_topk * 4 + r * dc.engram.n * dc.engram.cols() * 4;
}

/// The largest batch, at most `batch_cap` windows of `rows` tokens, that fits `budget`.
pub fn batchWindows(dc: *const dsv_cfg.Config, budget: u64, floor: u64, rows: usize) !u32 {
    const reserve = dc.chunkCap(chunk_rows, rows) + sumsBytes(dc) + (2 << 30);
    if (budget <= floor or budget - floor <= reserve) return error.Dsv41CaptureBudgetTooSmall;
    const fits = (budget - floor - reserve) / carriedPerWindow(dc, rows);
    if (fits == 0) return error.Dsv41CaptureBudgetTooSmall;
    return @intCast(@min(fits, batch_cap));
}

// ── what a resumed run must match ────────────────────────────────────────────────────────────────

const Manifest = struct {
    schema: []const u8 = "sushi-dsv41-layer-major-capture-v1",
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
    /// Absent for the default outputs, so earlier runs' manifests still match.
    mode: ?[]const u8 = null,
};

/// One line of `windows.jsonl`, written only after the window's rows, routes, sums and logits reached the device.
const WindowRecord = struct {
    window: usize,
    id: []const u8,
    dir: []const u8,
    tokens: usize,
    token_offset: u64,
    chosen: u32,
    nll_bits: u64,
    logits_dtype: []const u8,
    ids_sha256: []const u8,
    logits_sha256: []const u8,
};

fn sourceStatSha(a: std.mem.Allocator, io: std.Io, dir_path: []const u8) ![64]u8 {
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| a.free(n);
        names.deinit(a);
    }
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.startsWith(u8, entry.name, "model") or !std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        try names.append(a, try a.dupe(u8, entry.name));
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn less(_: void, x: []u8, y: []u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    if (names.items.len == 0) return error.MissingIndexedDsv41Weight;
    var hash = Sha256.init(.{});
    for (names.items) |name| {
        const stat = try dir.statFile(io, name, .{});
        const stamp = try std.fmt.allocPrint(a, "{s}:{d}:{d}\n", .{ name, stat.size, stat.mtime.nanoseconds });
        defer a.free(stamp);
        hash.update(stamp);
    }
    return std.fmt.bytesToHex(hash.finalResult(), .lower);
}

// ── small file helpers ──────────────────────────────────────────────────────────────────────────

fn openAppend(dir: []const u8, name: []const u8) !std.c.fd_t {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.Dsv41CaptureOpenFailed;
    return fd;
}

fn writeAll(fd: std.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const got = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (got < 0) {
            if (std.c._errno().* == @intFromEnum(std.c.E.INTR)) continue;
            return error.Dsv41CaptureWriteFailed;
        }
        if (got == 0) return error.Dsv41CaptureWriteFailed;
        done += @intCast(got);
    }
}

/// A file shorter than `bytes` lost committed rows; longer is an interrupted batch's tail.
fn truncateFd(fd: std.c.fd_t, bytes: u64) !void {
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0 or @as(u64, @intCast(st.size)) < bytes) return error.Dsv41CaptureRowsMissing;
    if (std.c.ftruncate(fd, @intCast(bytes)) != 0) return error.Dsv41CaptureOpenFailed;
}

/// The array's bytes on the host (the caller frees).
fn hostBytes(a: std.mem.Allocator, s: mlx.mlx_stream, x: A) ![]u8 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
    const data: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint8(c) orelse return error.MlxArrayDataNull);
    return a.dupe(u8, data[0..n]);
}

// ── the trunk sums: one safetensors file per layer, folded on a worker thread ────────────────────────

const Entry = struct {
    name: []u8,
    shape: []usize,
    data: []f32,
    rows: u64,

    fn deinit(self: *Entry, a: std.mem.Allocator) void {
        a.free(self.name);
        a.free(self.shape);
        a.free(self.data);
    }
};

const Job = struct { layer: usize, from: usize, to: usize, entries: []Entry };

fn sumsPath(a: std.mem.Allocator, dir: []const u8, layer: usize, windows: usize, tmp: bool) ![]u8 {
    const suffix = if (tmp) ".tmp.safetensors" else ".safetensors";
    if (layer == head_layer) return std.fmt.allocPrint(a, "{s}/trunk-head.{d:0>6}{s}", .{ dir, windows, suffix });
    return std.fmt.allocPrint(a, "{s}/trunk-{d:0>2}.{d:0>6}{s}", .{ dir, layer, windows, suffix });
}

/// Adds the previous file's sums into the job's, writes the new file whole and renames it into place.
fn foldJob(a: std.mem.Allocator, dir: []const u8, job: Job) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    if (job.from > 0) {
        const prev_path = try sumsPath(a, dir, job.layer, job.from, false);
        defer a.free(prev_path);
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, prev_path, a, .limited(16 << 30));
        defer a.free(raw);
        if (raw.len < 8) return error.Dsv41SumsCorrupt;
        const hlen = std.mem.readInt(u64, raw[0..8], .little);
        if (8 + hlen > raw.len) return error.Dsv41SumsCorrupt;
        const parsed = try std.json.parseFromSlice(std.json.Value, a, raw[8 .. 8 + hlen], .{});
        defer parsed.deinit();
        const body = raw[8 + hlen ..];
        for (job.entries) |*e| {
            const t = (parsed.value.object.get(e.name) orelse return error.Dsv41SumsCorrupt).object;
            const off = t.get("data_offsets").?.array.items;
            const lo: usize = @intCast(off[0].integer);
            const hi: usize = @intCast(off[1].integer);
            if (hi - lo != e.data.len * 4 or hi > body.len) return error.Dsv41SumsCorrupt;
            for (e.data, 0..) |*v, i| v.* += @bitCast(std.mem.readInt(u32, body[lo + i * 4 ..][0..4], .little));
            var key_buf: [160]u8 = undefined;
            const rt = (parsed.value.object.get(try std.fmt.bufPrint(&key_buf, "{s}.rows", .{e.name})) orelse return error.Dsv41SumsCorrupt).object;
            const roff = rt.get("data_offsets").?.array.items;
            e.rows += std.mem.readInt(u64, body[@intCast(roff[0].integer)..][0..8], .little);
        }
    }
    var header: std.Io.Writer.Allocating = .init(a);
    defer header.deinit();
    const w = &header.writer;
    try w.print("{{\"__metadata__\":{{\"windows\":\"{d}\",\"producer\":\"sushi kld capture --layer-major\"}}", .{job.to});
    var offset: u64 = 0;
    for (job.entries) |e| {
        try w.print(",\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{e.name});
        for (e.shape, 0..) |d, i| try w.print("{s}{d}", .{ if (i == 0) "" else ",", d });
        try w.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + e.data.len * 4 });
        offset += e.data.len * 4;
    }
    for (job.entries) |e| {
        try w.print(",\"{s}.rows\":{{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[{d},{d}]}}", .{ e.name, offset, offset + 8 });
        offset += 8;
    }
    try w.writeAll("}");
    while (header.written().len % 8 != 0) try w.writeByte(' ');

    const tmp = try sumsPath(a, dir, job.layer, job.to, true);
    defer a.free(tmp);
    const final = try sumsPath(a, dir, job.layer, job.to, false);
    defer a.free(final);
    const tmp_z = try std.fmt.allocPrintSentinel(a, "{s}", .{tmp}, 0);
    defer a.free(tmp_z);
    const fd = std.c.open(tmp_z.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.Dsv41CaptureOpenFailed;
    {
        defer _ = std.c.close(fd);
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, header.written().len, .little);
        try writeAll(fd, &len);
        try writeAll(fd, header.written());
        for (job.entries) |e| try writeAll(fd, std.mem.sliceAsBytes(e.data));
        for (job.entries) |e| try writeAll(fd, std.mem.asBytes(&e.rows));
        if (std.c.fsync(fd) != 0) return error.Dsv41CaptureWriteFailed;
    }
    const cwd = std.Io.Dir.cwd();
    try cwd.rename(tmp, cwd, final, io);
}

/// Folds at most two layers' sums at a time off the inference thread; MLX stays on the caller.
const Folder = struct {
    a: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    mu: std.Io.Mutex = .init,
    work: std.Io.Condition = .init,
    idle: std.Io.Condition = .init,
    queue: [2]?Job = .{ null, null },
    busy: bool = false,
    stopping: bool = false,
    failure: ?anyerror = null,
    thread: ?std.Thread = null,

    fn start(self: *Folder) !void {
        self.thread = try std.Thread.spawn(.{}, loop, .{self});
    }

    fn stop(self: *Folder) void {
        self.mu.lockUncancelable(self.io);
        self.stopping = true;
        self.work.broadcast(self.io);
        self.mu.unlock(self.io);
        if (self.thread) |t| t.join();
        self.thread = null;
        for (&self.queue) |*slot| if (slot.*) |job| {
            for (job.entries) |*e| e.deinit(self.a);
            self.a.free(job.entries);
            slot.* = null;
        };
    }

    /// Takes the job's entries; waits while both slots are taken.
    fn submit(self: *Folder, job: Job) !void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        while (self.failure == null and self.queue[0] != null and self.queue[1] != null) self.idle.waitUncancelable(self.io, &self.mu);
        if (self.failure) |e| {
            for (job.entries) |*x| @constCast(x).deinit(self.a);
            self.a.free(job.entries);
            return e;
        }
        const slot: usize = if (self.queue[0] == null) 0 else 1;
        self.queue[slot] = job;
        self.work.signal(self.io);
    }

    /// Returns once every submitted job has been written.
    fn drain(self: *Folder) !void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        while (self.failure == null and (self.busy or self.queue[0] != null or self.queue[1] != null)) self.idle.waitUncancelable(self.io, &self.mu);
        if (self.failure) |e| return e;
    }

    fn loop(self: *Folder) void {
        while (true) {
            self.mu.lockUncancelable(self.io);
            while (!self.stopping and self.queue[0] == null and self.queue[1] == null) self.work.waitUncancelable(self.io, &self.mu);
            if (self.stopping) {
                self.mu.unlock(self.io);
                return;
            }
            const slot: usize = if (self.queue[0] != null) 0 else 1;
            const job = self.queue[slot].?;
            self.busy = true;
            self.mu.unlock(self.io);
            const result = foldJob(self.a, self.dir, job);
            for (job.entries) |*e| e.deinit(self.a);
            self.a.free(job.entries);
            self.mu.lockUncancelable(self.io);
            self.queue[slot] = null;
            self.busy = false;
            if (result) |_| {} else |err| self.failure = err;
            self.idle.broadcast(self.io);
            self.mu.unlock(self.io);
        }
    }
};

// ── the tap sink ─────────────────────────────────────────────────────────────────────────────────

const Capture = struct {
    a: std.mem.Allocator,
    s: mlx.mlx_stream,
    dim: usize,
    top_k: usize,
    layers: usize,
    fds: []std.c.fd_t,
    collector: ?*imatrix_capture.Collector,
    folder: *Folder,
    lean: bool,
    head_fd: std.c.fd_t = -1,
    /// The backbone layers whose mean-hc input is dumped, in the order of their files.
    dspark: []const u32,
    io: std.Io = std.Io.Threaded.global_single_threaded.io(),
    /// Nanoseconds the sink itself worked (sums, row and route writes, imatrix), after the forward made its inputs.
    extra_ns: u64 = 0,
    layer: usize = 0,
    acc: [kinds]A = @splat(none),
    rows: [kinds]u64 = @splat(0),
    /// The FFN input of the layer in flight, until its routes arrive.
    ffn_x: A = none,

    fn open(a: std.mem.Allocator, s: mlx.mlx_stream, dc: *const dsv_cfg.Config, calib: []const u8, collector: ?*imatrix_capture.Collector, folder: *Folder, mode: Mode) !Capture {
        const n = dc.layers;
        const files: usize = if (mode.lean) 0 else 3 * n + dc.dspark.n_targets;
        const fds = try a.alloc(std.c.fd_t, files);
        errdefer a.free(fds);
        @memset(fds, -1);
        errdefer for (fds) |fd| if (fd >= 0) {
            _ = std.c.close(fd);
        };
        var name: [48]u8 = undefined;
        if (!mode.lean) {
            for (0..n) |li| {
                fds[li] = try openAppend(calib, try std.fmt.bufPrint(&name, "mlp-in-{d:0>2}.bf16", .{li}));
                fds[n + li] = try openAppend(calib, try std.fmt.bufPrint(&name, "route-{d:0>2}.u16", .{li}));
                fds[2 * n + li] = try openAppend(calib, try std.fmt.bufPrint(&name, "route-w-{d:0>2}.f32", .{li}));
            }
            for (0..dc.dspark.n_targets) |i| fds[3 * n + i] = try openAppend(calib, try std.fmt.bufPrint(&name, "dspark-in-{d:0>2}.bf16", .{dc.dspark.targets[i]}));
        }
        const head_fd = if (mode.head_rows) try openAppend(calib, "head-in.bf16") else -1;
        return .{ .a = a, .s = s, .dim = dc.dim, .top_k = dc.top_k, .layers = n, .fds = fds, .collector = collector, .folder = folder, .lean = mode.lean, .head_fd = head_fd, .dspark = dc.dspark.targets[0..dc.dspark.n_targets] };
    }

    fn close(self: *Capture) void {
        self.resetSums();
        if (self.ffn_x.ctx != null) _ = mlx.mlx_array_free(self.ffn_x);
        for (self.fds) |fd| _ = std.c.close(fd);
        if (self.head_fd >= 0) _ = std.c.close(self.head_fd);
        self.a.free(self.fds);
    }

    fn mlpFd(self: *const Capture, layer: usize) std.c.fd_t {
        return self.fds[layer];
    }
    fn routeFd(self: *const Capture, layer: usize) std.c.fd_t {
        return self.fds[self.layers + layer];
    }
    fn weightFd(self: *const Capture, layer: usize) std.c.fd_t {
        return self.fds[2 * self.layers + layer];
    }

    fn tap(self: *Capture) dsv.Tap {
        return .{ .ctx = self, .input = input, .routes = routes, .expert_act = expertAct };
    }

    fn resetSums(self: *Capture) void {
        for (&self.acc) |*x| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x.*);
            x.* = none;
        }
        self.rows = @splat(0);
    }

    fn dsparkFd(self: *const Capture, layer: usize) ?std.c.fd_t {
        for (self.dspark, 0..) |t, i| if (t == layer) return self.fds[3 * self.layers + i];
        return null;
    }

    fn truncate(self: *Capture, tokens: u64) !void {
        if (self.head_fd >= 0) try truncateFd(self.head_fd, tokens * self.dim * 2);
        if (self.lean) return;
        for (self.dspark, 0..) |_, i| try truncateFd(self.fds[3 * self.layers + i], tokens * self.dim * 2);
        for (0..self.layers) |li| {
            try truncateFd(self.mlpFd(li), tokens * self.dim * 2);
            try truncateFd(self.routeFd(li), tokens * self.top_k * 2);
            try truncateFd(self.weightFd(li), tokens * self.top_k * 4);
        }
    }

    fn sync(self: *Capture) !void {
        for (self.fds) |fd| if (std.c.fsync(fd) != 0) return error.Dsv41CaptureWriteFailed;
        if (self.head_fd >= 0 and std.c.fsync(self.head_fd) != 0) return error.Dsv41CaptureWriteFailed;
    }

    /// The head's input rows as the forward made them; bf16 is the activation dtype, so this rounds nothing.
    fn writeHeadRows(self: *Capture, x: A) !void {
        var sc = Scope.init(self.a, self.s);
        defer sc.deinit();
        const bytes = try hostBytes(self.a, self.s, try sc.astype(x, .bfloat16));
        defer self.a.free(bytes);
        try writeAll(self.head_fd, bytes);
    }

    fn input(ctx: *anyopaque, layer: usize, kind: dsv.TapKind, x: A) anyerror!void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        if (self.lean and kind != .head_in) return;
        try mlx.check(mlx.mlx_array_eval(x));
        const started = std.Io.Timestamp.now(self.io, .awake);
        defer self.extra_ns += @intCast(started.untilNow(self.io, .awake).nanoseconds);
        if (kind == .head_in) {
            if (self.layer != head_layer) return error.Dsv41CaptureOutOfOrder;
            if (self.head_fd >= 0) try self.writeHeadRows(x);
            if (self.lean) return;
        } else if (layer != self.layer) return error.Dsv41CaptureOutOfOrder;
        if (kind == .dspark_in) {
            const bytes = try hostBytes(self.a, self.s, x);
            defer self.a.free(bytes);
            return writeAll(self.dsparkFd(layer) orelse return error.Dsv41CaptureOutOfOrder, bytes);
        }
        if (kind == .ffn_in) {
            const bytes = try hostBytes(self.a, self.s, x);
            defer self.a.free(bytes);
            try writeAll(self.mlpFd(layer), bytes);
            if (self.ffn_x.ctx != null) _ = mlx.mlx_array_free(self.ffn_x);
            self.ffn_x = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&self.ffn_x, x));
        }
        try self.accumulate(kind, x);
    }

    /// `acc += X^T X` over the rows of `x` (one `[in, in]` matrix, `[groups, in, in]` for the grouped output projection).
    fn accumulate(self: *Capture, kind: dsv.TapKind, x: A) !void {
        var sc = Scope.init(self.a, self.s);
        defer sc.deinit();
        const shape = mlx.getShape(x);
        var g: A = undefined;
        var rows: usize = undefined;
        if (kind == .wo_a_in) {
            const xf = try sc.astype(try sc.transpose(x, &.{ 1, 0, 2 }), .float32);
            g = try sc.matmul(try sc.transpose(xf, &.{ 0, 2, 1 }), xf);
            rows = @intCast(shape[0]);
        } else {
            const width: usize = @intCast(shape[shape.len - 1]);
            rows = mlx.mlx_array_size(x) / width;
            const xf = try sc.astype(try sc.reshape(x, &.{ sc_mod.ci(rows), sc_mod.ci(width) }), .float32);
            g = try sc.matmul(try sc.t2(xf), xf);
        }
        const k = @intFromEnum(kind);
        const total = if (self.acc[k].ctx != null) sc.out(try sc.add(self.acc[k], g)) else sc.out(g);
        try mlx.check(mlx.mlx_array_eval(total));
        if (self.acc[k].ctx != null) _ = mlx.mlx_array_free(self.acc[k]);
        self.acc[k] = total;
        self.rows[k] += rows;
    }

    fn routes(ctx: *anyopaque, layer: usize, idx: A, wts: A) anyerror!void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        if (self.lean) return;
        try mlx.check(mlx.mlx_array_eval(idx));
        try mlx.check(mlx.mlx_array_eval(wts));
        const started = std.Io.Timestamp.now(self.io, .awake);
        defer self.extra_ns += @intCast(started.untilNow(self.io, .awake).nanoseconds);
        if (layer != self.layer or self.ffn_x.ctx == null) return error.Dsv41CaptureOutOfOrder;
        const ids = try hostBytes(self.a, self.s, idx);
        defer self.a.free(ids);
        const w = try hostBytes(self.a, self.s, wts);
        defer self.a.free(w);
        const n = ids.len / 4;
        const packed_ids = try self.a.alloc(u16, n);
        defer self.a.free(packed_ids);
        for (packed_ids, 0..) |*o, i| {
            const id = std.mem.readInt(i32, ids[i * 4 ..][0..4], .little);
            if (id < 0 or id > std.math.maxInt(u16)) return error.ExpertOutOfRange;
            o.* = @intCast(id);
        }
        try writeAll(self.routeFd(layer), std.mem.sliceAsBytes(packed_ids));
        try writeAll(self.weightFd(layer), w);
        try self.collector.?.observeGateUp(layer, self.ffn_x, idx);
        _ = mlx.mlx_array_free(self.ffn_x);
        self.ffn_x = none;
    }

    fn expertAct(ctx: *anyopaque, layer: usize, act: A, ids: A) anyerror!void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        if (self.lean) return;
        try mlx.check(mlx.mlx_array_eval(act));
        const started = std.Io.Timestamp.now(self.io, .awake);
        defer self.extra_ns += @intCast(started.untilNow(self.io, .awake).nanoseconds);
        try self.collector.?.observeDown(layer, act, ids);
    }

    fn begin(self: *Capture, layer: usize) void {
        self.resetSums();
        self.layer = layer;
    }

    /// Hands the layer's (or the head's) sums to the folder as the running total of windows `[0, to)`.
    fn end(self: *Capture, from: usize, to: usize) !void {
        if (self.lean) return;
        const started = std.Io.Timestamp.now(self.io, .awake);
        defer self.extra_ns += @intCast(started.untilNow(self.io, .awake).nanoseconds);
        const layer = self.layer;
        var entries: std.ArrayList(Entry) = .empty;
        errdefer {
            for (entries.items) |*e| e.deinit(self.a);
            entries.deinit(self.a);
        }
        for (std.enums.values(dsv.TapKind)) |kind| {
            const k = @intFromEnum(kind);
            const x = self.acc[k];
            if (x.ctx != null) {
                try mlx.check(mlx.mlx_array_eval(x));
                const n = mlx.mlx_array_size(x);
                const src = mlx.mlx_array_data_float32(x) orelse return error.MlxArrayDataNull;
                const data = try self.a.alloc(f32, n);
                errdefer self.a.free(data);
                @memcpy(data, src[0..n]);
                const shape = try self.a.alloc(usize, mlx.mlx_array_ndim(x));
                errdefer self.a.free(shape);
                for (shape, mlx.getShape(x)) |*o, d| o.* = @intCast(d);
                const name = if (kind == .head_in) try self.a.dupe(u8, "head_in") else try std.fmt.allocPrint(self.a, "layers.{d}.{s}", .{ layer, @tagName(kind) });
                errdefer self.a.free(name);
                try entries.append(self.a, .{ .name = name, .shape = shape, .data = data, .rows = self.rows[k] });
            }
        }
        self.resetSums();
        try self.folder.submit(.{ .layer = layer, .from = from, .to = to, .entries = try entries.toOwnedSlice(self.a) });
    }
};

// ── the run ──────────────────────────────────────────────────────────────────────────────────────

/// One window between layers: its state and one pass per chunk.
const Window = struct {
    st: dsv.State,
    passes: []dsv.Pass,
    /// Each pass's CED tail (`dsv.cedFork`), run after it: the decoder state and the window's logits are the served
    /// prefill's, while the passes feed the calibration files every row.
    tails: []?dsv.Pass = &.{},
};

const Run = struct {
    a: std.mem.Allocator,
    io: std.Io,
    m: *dsv.Model,
    stream: *dsv_stream.Stream,
    staging: []const u8,
    calib: []const u8,
    prompts: []const kld.Prompt,
    inputs: []const []u32,
    capture: *Capture,
    collector: ?*imatrix_capture.Collector,
    folder: *Folder,
    mode: Mode,
    log_fd: std.c.fd_t = -1,
    tokens_fd: std.c.fd_t = -1,
    records: []WindowRecord = &.{},
    arena: std.heap.ArenaAllocator,
    committed_tokens: u64 = 0,
    row: []f32,
    chunk: usize,
    out: *kld.Out,
    pause_file: []const u8,
    pin_ns: u64 = 0,

    fn statePath(self: *Run, buf: []u8, n: usize, tmp: bool) ![]const u8 {
        return std.fmt.bufPrint(buf, "{s}/imatrix-{d}{s}.safetensors", .{ self.staging, n, if (tmp) ".tmp" else "" });
    }

    fn saveImatrix(self: *Run, n: usize) !void {
        var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
        var final_buf: [std.fs.max_path_bytes]u8 = undefined;
        const tmp = try self.statePath(&tmp_buf, n, true);
        const final = try self.statePath(&final_buf, n, false);
        try self.collector.?.saveState(tmp);
        try glm.fsyncPath(tmp);
        const cwd = std.Io.Dir.cwd();
        try cwd.rename(tmp, cwd, final, self.io);
    }

    /// Removes this run's recoverable leftovers except the files of the first `keep` windows.
    fn dropStale(self: *Run, keep: usize) void {
        const cwd = std.Io.Dir.cwd();
        var keep_state: [64]u8 = undefined;
        const keep_name = std.fmt.bufPrint(&keep_state, "imatrix-{d}.safetensors", .{keep}) catch return;
        var keep_suffix: [32]u8 = undefined;
        const keep_sums = std.fmt.bufPrint(&keep_suffix, ".{d:0>6}.safetensors", .{keep}) catch return;
        if (cwd.openDir(self.io, self.staging, .{ .iterate = true })) |opened| {
            var dir = opened;
            defer dir.close(self.io);
            var it = dir.iterate();
            while (it.next(self.io) catch null) |entry| {
                if (std.mem.startsWith(u8, entry.name, "imatrix-") and !std.mem.eql(u8, entry.name, keep_name)) dir.deleteFile(self.io, entry.name) catch {};
            }
        } else |_| {}
        if (cwd.openDir(self.io, self.calib, .{ .iterate = true })) |opened| {
            var dir = opened;
            defer dir.close(self.io);
            var it = dir.iterate();
            while (it.next(self.io) catch null) |entry| {
                if (std.mem.startsWith(u8, entry.name, "trunk-") and !std.mem.endsWith(u8, entry.name, keep_sums)) dir.deleteFile(self.io, entry.name) catch {};
            }
        } else |_| {}
    }

    fn openLedger(self: *Run, expected: []const u8) !void {
        const a = self.a;
        const cwd = std.Io.Dir.cwd();
        const manifest = try std.fmt.allocPrint(a, "{s}/layer-major.json", .{self.staging});
        defer a.free(manifest);
        if (cwd.access(self.io, self.staging, .{})) |_| {
            const stored = cwd.readFileAlloc(self.io, manifest, a, .limited(1 << 20)) catch |err| return if (err == error.FileNotFound) error.Dsv41LayerMajorStagingUnknown else err;
            defer a.free(stored);
            if (!std.mem.eql(u8, stored, expected)) {
                @import("../log.zig").err("[kld] layer-major: {s} belongs to another run.\nstored:\n{s}\nthis run:\n{s}\n", .{ self.staging, stored, expected });
                return error.Dsv41LayerMajorResumeMismatch;
            }
        } else |err| {
            if (err != error.FileNotFound) return err;
            try cwd.createDir(self.io, self.staging, .default_dir);
            const tmp = try std.fmt.allocPrint(a, "{s}.tmp", .{manifest});
            defer a.free(tmp);
            try cwd.writeFile(self.io, .{ .sub_path = tmp, .data = expected });
            try glm.fsyncPath(tmp);
            try cwd.rename(tmp, cwd, manifest, self.io);
        }
        try cwd.createDirPath(self.io, self.calib);
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const log_path = try std.fmt.bufPrintSentinel(&buf, "{s}/windows.jsonl", .{self.staging}, 0);
        self.log_fd = std.c.open(log_path.ptr, .{ .ACCMODE = .RDWR, .CREAT = true, .APPEND = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o644));
        if (self.log_fd < 0) return error.GlmLayerMajorLogUnreadable;
        const arena = self.arena.allocator();
        const body = try cwd.readFileAlloc(self.io, log_path, arena, .limited(1 << 30));
        // A line without its newline was being written when the run died; it commits nothing.
        const whole = if (std.mem.lastIndexOfScalar(u8, body, '\n')) |e| e + 1 else 0;
        if (whole != body.len and std.c.ftruncate(self.log_fd, @intCast(whole)) != 0) return error.GlmLayerMajorLogUnreadable;
        var lines: std.ArrayList(WindowRecord) = .empty;
        var it = std.mem.splitScalar(u8, body[0..whole], '\n');
        while (it.next()) |line| {
            if (line.len == 0) continue;
            try lines.append(arena, std.json.parseFromSliceLeaky(WindowRecord, arena, line, .{}) catch return error.GlmLayerMajorLogCorrupt);
        }
        self.records = lines.items;
        self.tokens_fd = try openAppend(self.calib, "tokens.bin");
    }

    fn closeLedger(self: *Run) void {
        if (self.log_fd >= 0) _ = std.c.close(self.log_fd);
        if (self.tokens_fd >= 0) _ = std.c.close(self.tokens_fd);
        self.arena.deinit();
    }

    /// Every committed window must still be the same input and logits row.
    fn verify(self: *Run) !void {
        var offset: u64 = 0;
        for (self.records, 0..) |r, i| {
            if (r.window != i or i >= self.prompts.len or !std.mem.eql(u8, r.id, self.prompts[i].id) or r.tokens != self.inputs[i].len or r.token_offset != offset) return error.GlmLayerMajorWindowChanged;
            const ids_sha = glm.shaHex(std.mem.sliceAsBytes(self.inputs[i]));
            const path = try std.fmt.allocPrint(self.a, "{s}/{s}/logits.f32", .{ self.staging, r.dir });
            defer self.a.free(path);
            const logits = std.Io.Dir.cwd().readFileAlloc(self.io, path, self.a, .limited(64 << 20)) catch return error.GlmLayerMajorWindowChanged;
            defer self.a.free(logits);
            const logits_sha = glm.shaHex(logits);
            if (!std.mem.eql(u8, r.ids_sha256, &ids_sha) or !std.mem.eql(u8, r.logits_sha256, &logits_sha)) return error.GlmLayerMajorWindowChanged;
            offset += r.tokens;
        }
        self.committed_tokens = offset;
        if (self.records.len == 0) return;
        const bytes = try self.a.alloc(u8, @intCast(offset * 4));
        defer self.a.free(bytes);
        const path = try std.fmt.allocPrint(self.a, "{s}/tokens.bin", .{self.calib});
        defer self.a.free(path);
        var file = try std.Io.Dir.cwd().openFile(self.io, path, .{});
        defer file.close(self.io);
        if (file.readPositionalAll(self.io, bytes, 0) catch 0 != bytes.len) return error.GlmLayerMajorWindowChanged;
        var at: usize = 0;
        for (self.records, 0..) |r, i| {
            if (!std.mem.eql(u8, bytes[at * 4 ..][0 .. r.tokens * 4], std.mem.sliceAsBytes(self.inputs[i]))) return error.GlmLayerMajorWindowChanged;
            at += r.tokens;
        }
    }

    const Result = struct { record: ?kld.PromptRecord = null, chosen: u32 = 0, logits_sha256: [64]u8 = undefined };

    /// Writes one window's standard fixture row from its logits.
    fn writeWindow(self: *Run, w: usize, logits: A) !Result {
        if (mlx.mlx_array_size(logits) != self.row.len) return error.InvalidDsv41Logits;
        var wide = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wide);
        try mlx.check(mlx.mlx_astype(&wide, logits, .float32, self.m.s));
        try mlx.check(mlx.mlx_array_eval(wide));
        @memcpy(self.row, (mlx.mlx_array_data_float32(wide) orelse return error.MlxArrayDataNull)[0..self.row.len]);
        var nonzero = false;
        for (self.row) |v| {
            if (!std.math.isFinite(v)) return error.NonfiniteDsv41Logits;
            nonzero = nonzero or v != 0;
        }
        if (!nonzero) return error.Dsv41ZeroNormLogits;
        const chosen = glm.greedy(self.row);
        const p = self.prompts[w];
        var writer = try kld.PromptWriter.begin(self.a, self.io, self.staging, w, p.id);
        defer writer.deinit();
        try writer.appendRow(self.row, chosen);
        const record = try writer.finish(p.text, p.text, self.inputs[w], &[_]u32{chosen});
        return .{ .record = record, .chosen = chosen, .logits_sha256 = glm.shaHex(std.mem.sliceAsBytes(self.row)) };
    }

    const BatchStats = struct { fill_bytes: u64, fill_ns: u64, wall_ns: u64, pin_ns: u64, extra_ns: u64 };

    /// Windows `[first, end)` through every layer, then their heads.
    fn runBatch(self: *Run, first: usize, end: usize, results: []Result) !BatchStats {
        const a = self.a;
        const m = self.m;
        const engine = self.stream.engine;
        const fill_bytes = engine.fill_bytes_total;
        const fill_ns = engine.fill_ns_total;
        const pin_ns = self.pin_ns;
        const extra_ns = self.capture.extra_ns;
        var clock = expert_stream.Clock.init();
        const windows = try a.alloc(Window, end - first);
        defer a.free(windows);
        var made: usize = 0;
        defer for (windows[0..made]) |*w| {
            for (w.tails) |*t| if (t.*) |*x| x.deinit(a);
            a.free(w.tails);
            for (w.passes) |*p| p.deinit(a);
            a.free(w.passes);
            w.st.deinit();
        };
        for (windows, self.inputs[first..end]) |*w, ids| {
            w.* = .{ .st = try dsv.newState(m), .passes = &.{} };
            made += 1;
            var passes: std.ArrayList(dsv.Pass) = .empty;
            errdefer {
                for (passes.items) |*p| p.deinit(a);
                passes.deinit(a);
            }
            var at: usize = 0;
            while (at < ids.len) : (at += self.chunk) {
                try passes.append(a, try dsv.beginPass(m, &w.st, ids[at..@min(ids.len, at + self.chunk)], at, dsv.SPAN));
            }
            w.passes = try passes.toOwnedSlice(a);
            w.tails = try a.alloc(?dsv.Pass, w.passes.len);
            @memset(w.tails, null);
        }
        for (0..m.cfg.layers) |li| {
            const pin_started = std.Io.Timestamp.now(self.io, .awake);
            try self.stream.pin(li);
            self.pin_ns += @intCast(pin_started.untilNow(self.io, .awake).nanoseconds);
            defer self.stream.unpin();
            self.capture.begin(li);
            for (windows) |*w| for (w.passes, w.tails) |*p, *tail| {
                if (@import("builtin").is_test) if (interrupt_layers_for_test) |*left| {
                    if (left.* == 0) return error.TestInterrupted;
                    left.* -= 1;
                };
                if (m.ced_split != null and li == m.ced_split.?) tail.* = try dsv.cedFork(m, p, .last);
                try dsv.passLayer(m, p, li);
                // A tail feeds no calibration file: its rows are the pass's own again.
                if (tail.*) |*t| {
                    const tap = m.tap;
                    m.tap = null;
                    defer m.tap = tap;
                    try dsv.passLayer(m, t, li);
                }
            };
            try self.capture.end(first, end);
        }
        self.capture.begin(head_layer);
        for (windows, results, first..) |*w, *res, index| {
            for (w.passes, 0..) |*p, k| {
                const logits = try dsv.finishPass(m, p, .last);
                defer _ = mlx.mlx_array_free(logits);
                if (k + 1 < w.passes.len) continue;
                if (w.tails[k]) |*t| {
                    const tap = m.tap;
                    m.tap = null;
                    defer m.tap = tap;
                    const served = try dsv.finishPass(m, t, .last);
                    defer _ = mlx.mlx_array_free(served);
                    res.* = try self.writeWindow(index, served);
                } else res.* = try self.writeWindow(index, logits);
            }
        }
        try self.capture.end(first, end);
        return .{ .fill_bytes = engine.fill_bytes_total - fill_bytes, .fill_ns = engine.fill_ns_total - fill_ns, .wall_ns = clock.lap(), .pin_ns = self.pin_ns - pin_ns, .extra_ns = self.capture.extra_ns - extra_ns };
    }

    /// Every uncommitted window, a batch at a time.
    fn capturePrompts(self: *Run, batch: u32, records: *std.ArrayList(kld.PromptRecord)) !void {
        const a = self.a;
        for (self.records) |r| {
            const nll: f64 = @bitCast(r.nll_bits);
            const id = try a.dupe(u8, r.id);
            errdefer a.free(id);
            try records.append(a, .{ .id = id, .dir = try a.dupe(u8, r.dir), .prompt_tokens = r.tokens, .generated_tokens = 1, .strict_nll_mean = nll, .strict_perplexity = @exp(nll) });
        }
        try self.capture.truncate(self.committed_tokens);
        var next = self.records.len;
        var offset = self.committed_tokens;
        if (!self.mode.lean) try self.restoreImatrix(next);
        self.dropStale(next);
        const run_clock = expert_stream.Clock.init();
        var run_tokens: u64 = 0;
        var remaining: u64 = 0;
        for (self.inputs[next..]) |ids| remaining += ids.len;
        const needed = diskBytes(&self.m.cfg, remaining, self.prompts.len - next, self.row.len, self.mode);
        if (@import("../disk_placement.zig").volumeSpace(self.calib)) |space| if (space.free < needed + (16 << 30)) {
            @import("../log.zig").err("[kld] layer-major: {d} GB still to write, {d} GB free (16 GiB margin)\n", .{ needed / 1_000_000_000, space.free / 1_000_000_000 });
            return error.GlmLayerMajorDiskTooSmall;
        };
        const results = try a.alloc(Result, batch);
        defer a.free(results);
        while (next < self.prompts.len) {
            try glm.waitWhilePaused(a, self.io, self.pause_file, next, self.out);
            const end = @min(self.prompts.len, next + batch);
            const batch_results = results[0 .. end - next];
            @memset(batch_results, .{});
            defer for (batch_results) |r| if (r.record) |rec| rec.deinit(a);
            const stats = try self.runBatch(next, end, batch_results);
            try self.folder.drain();
            for (self.inputs[next..end]) |ids| try writeAll(self.tokens_fd, std.mem.sliceAsBytes(ids));
            if (std.c.fsync(self.tokens_fd) != 0) return error.Dsv41CaptureWriteFailed;
            try self.capture.sync();
            if (!self.mode.lean) try self.saveImatrix(end);
            var tokens: u64 = 0;
            for (batch_results, next..) |*res, i| {
                const record = res.record orelse return error.Dsv41CaptureRowCountMismatch;
                for ([_][]const u8{ "logits.f32", "prompt_tokens.txt", "generated_tokens.txt" }) |leaf| {
                    const path = try std.fmt.allocPrint(a, "{s}/{s}/{s}", .{ self.staging, record.dir, leaf });
                    defer a.free(path);
                    try glm.fsyncPath(path);
                }
                const ids_sha = glm.shaHex(std.mem.sliceAsBytes(self.inputs[i]));
                const line = try std.json.Stringify.valueAlloc(a, WindowRecord{
                    .window = i,
                    .id = record.id,
                    .dir = record.dir,
                    .tokens = self.inputs[i].len,
                    .token_offset = offset,
                    .chosen = res.chosen,
                    .nll_bits = @bitCast(record.strict_nll_mean),
                    .logits_dtype = "float32",
                    .ids_sha256 = &ids_sha,
                    .logits_sha256 = &res.logits_sha256,
                }, .{});
                defer a.free(line);
                try glm.appendLine(self.log_fd, line);
                offset += self.inputs[i].len;
                tokens += self.inputs[i].len;
            }
            if (std.c.fsync(self.log_fd) != 0) return error.GlmLayerMajorSyncFailed;
            self.committed_tokens = offset;
            self.dropStale(end);
            for (batch_results) |*res| {
                try records.append(a, res.record.?);
                res.record = null;
            }
            const wall = @as(f64, @floatFromInt(stats.wall_ns)) / 1e9;
            const fill = @as(f64, @floatFromInt(stats.fill_ns)) / 1e9;
            const pin = @as(f64, @floatFromInt(stats.pin_ns)) / 1e9;
            const extra = @as(f64, @floatFromInt(stats.extra_ns)) / 1e9;
            const forward = wall - pin - extra;
            run_tokens += tokens;
            remaining -= tokens;
            const rate = @as(f64, @floatFromInt(run_tokens)) / (@as(f64, @floatFromInt(run_clock.start.untilNow(run_clock.io, .boot).nanoseconds)) / 1e9);
            try glm.json(a, self.io, self.staging, "progress.json", .{ .complete = false, .phase = "layer-major", .committed_windows = end, .total_windows = self.prompts.len, .batch_windows = batch, .batch_seconds = wall, .batch_forward_seconds = forward, .batch_pin_seconds = pin, .batch_extra_seconds = extra, .batch_fill_seconds = fill, .batch_fill_bytes = stats.fill_bytes, .batch_tokens = tokens, .tokens_per_second = rate, .eta_hours = @as(f64, @floatFromInt(remaining)) / rate / 3600 });
            self.out.print("[kld] layer-major windows {d}..{d} of {d}: {d} tokens in {d:.1} s ({d:.2} tok/s), read {d:.1} GB in {d:.1} s; this run {d:.2} tok/s, {d:.2} h left\n", .{ next, end, self.prompts.len, tokens, wall, @as(f64, @floatFromInt(tokens)) / wall, @as(f64, @floatFromInt(stats.fill_bytes)) / 1e9, fill, rate, @as(f64, @floatFromInt(remaining)) / rate / 3600 });
            self.out.print("[kld] layer-major batch: forward {d:.1} s ({d:.2} tok/s), expert pin {d:.1} s ({d:.2} s a layer), capture extras {d:.1} s\n", .{ forward, @as(f64, @floatFromInt(tokens)) / forward, pin, pin / @as(f64, @floatFromInt(self.m.cfg.layers)), extra });
            next = end;
        }
    }

    fn restoreImatrix(self: *Run, n: usize) !void {
        if (n == 0) return;
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try self.statePath(&buf, n, false);
        std.Io.Dir.cwd().access(self.io, path, .{}) catch return error.GlmLayerMajorImatrixStateMissing;
        try self.collector.?.loadState(path);
    }
};

/// Bytes the remaining windows still write: rows, routes, token ids, every layer's sums (two versions in flight), logits.
fn diskBytes(dc: *const dsv_cfg.Config, tokens: u64, windows: u64, vocab: u64, mode: Mode) u64 {
    const head: u64 = if (mode.head_rows) @as(u64, dc.dim) * 2 else 0;
    if (mode.lean) return tokens * (head + 4) + windows * vocab * 4;
    const per_token = @as(u64, dc.layers) * (@as(u64, dc.dim) * 2 + dc.top_k * 6) + 4 + head;
    return tokens * per_token + @as(u64, dc.layers) * sumsBytes(dc) + windows * vocab * 4;
}

var interrupt_layers_for_test: ?usize = null;

/// The layer-major capture of `l`'s prompts into `opts.out_dir`, `batch` windows at a time.
pub fn runLoaded(a: std.mem.Allocator, io: std.Io, l: *kld.Loaded, opts: kld.Options, batch: u32, chunk: usize, out: *kld.Out) !void {
    return runLoadedMode(a, io, l, opts, batch, chunk, out, try modeFromEnv());
}

pub fn runLoadedMode(a: std.mem.Allocator, io: std.Io, l: *kld.Loaded, opts: kld.Options, batch: u32, chunk: usize, out: *kld.Out, mode: Mode) !void {
    const m = l.xfm.dsv41 orelse return error.Dsv41ModelMissing;
    const stream = l.xfm.dsv41_experts orelse return error.Dsv41ModelMissing;
    const dc = &m.cfg;
    const cwd = std.Io.Dir.cwd();
    if (cwd.access(io, opts.out_dir, .{})) |_| return error.Dsv41CaptureOutputExists else |err| if (err != error.FileNotFound) return err;
    const staging = try std.fmt.allocPrint(a, "{s}.partial", .{opts.out_dir});
    defer a.free(staging);
    const calib = try std.fmt.allocPrint(a, "{s}/calib", .{staging});
    defer a.free(calib);

    var prompts = try kld.loadPrompts(a, io, opts.prompts, opts.limit);
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
        for (ids.*) |id| if (id >= dc.vocab) return error.InvalidDsv41Prompt;
        if (ids.len > dc.max_pos) return error.Dsv41ContextExceeded;
    }

    const config_sha = try glm.metadataHash(a, io, opts.model_dir, "config.json");
    const index_sha = try glm.metadataHash(a, io, opts.model_dir, "model.safetensors.index.json");
    const tokenizer_sha = try glm.metadataHash(a, io, opts.model_dir, "tokenizer.json");
    const stat_sha = try sourceStatSha(a, io, opts.model_dir);
    const inputs_sha = glm.inputsSha(prompts.items, inputs);
    const engine_sha = try glm.engineSha(a, io);
    const manifest = try std.json.Stringify.valueAlloc(a, Manifest{
        .model = opts.model_dir,
        .config_sha256 = &config_sha,
        .index_sha256 = &index_sha,
        .tokenizer_sha256 = &tokenizer_sha,
        .shard_stat_sha256 = &stat_sha,
        .prompts = opts.prompts,
        .inputs_sha256 = &inputs_sha,
        .prompt_count = prompts.items.len,
        .chunk = chunk,
        .top_k = opts.top_k,
        .label = opts.label,
        .engine_sha256 = &engine_sha,
        .mode = mode.tag(),
    }, .{ .whitespace = .indent_2, .emit_null_optional_fields = false });
    defer a.free(manifest);

    const imatrix_path = if (opts.imatrix_out.len != 0) opts.imatrix_out else try std.fmt.allocPrint(a, "{s}/imatrix.safetensors", .{calib});
    defer if (opts.imatrix_out.len == 0) a.free(imatrix_path);
    const collector: ?*imatrix_capture.Collector = if (mode.lean) null else try imatrix_capture.Collector.init(a, m.s, imatrix_path, dc.layers, @intCast(dc.experts), .ffn_experts);
    defer if (collector) |c| c.deinit();
    var folder = Folder{ .a = a, .io = io, .dir = calib };
    var run = Run{
        .a = a,
        .io = io,
        .m = m,
        .stream = stream,
        .staging = staging,
        .calib = calib,
        .prompts = prompts.items,
        .inputs = inputs,
        .capture = undefined,
        .collector = collector,
        .folder = &folder,
        .mode = mode,
        .arena = .init(a),
        .row = try a.alloc(f32, dc.vocab),
        .chunk = chunk,
        .out = out,
        .pause_file = opts.pause_file,
    };
    defer a.free(run.row);
    try run.openLedger(manifest);
    defer run.closeLedger();
    try run.verify();
    if (!mode.lean) try folder.start();
    defer folder.stop();
    var capture = try Capture.open(a, m.s, dc, calib, collector, &folder, mode);
    defer capture.close();
    run.capture = &capture;
    m.tap = capture.tap();
    defer m.tap = null;

    try glm.json(a, io, staging, "progress.json", .{ .complete = false, .phase = "validated", .prompts = prompts.items.len });
    var records: std.ArrayList(kld.PromptRecord) = .empty;
    defer {
        for (records.items) |r| r.deinit(a);
        records.deinit(a);
    }
    const started = std.Io.Timestamp.now(io, .awake);
    try run.capturePrompts(batch, &records);
    if (collector) |c| _ = try c.flush();
    run.dropStale(prompts.items.len);
    const elapsed = @as(f64, @floatFromInt(started.untilNow(io, .awake).nanoseconds)) / 1e9;
    try kld.writeBaseline(a, io, staging, .{ .label = opts.label, .model = opts.model_dir, .run = opts.label, .kv_cache_format = "bf16", .inference_profile = "greedy", .prompt_set = opts.prompts, .ssd_budget_gb = kld.capturedSsdBudgetGb(&l.config, opts.ssd_budget_bytes), .tokens_per_prompt = opts.tokens, .top_k = opts.top_k, .elapsed_secs = elapsed }, records.items);
    try glm.completeBaseline(a, io, staging, records.items.len, opts.tokens);
    if (!std.mem.eql(u8, &config_sha, &try glm.metadataHash(a, io, opts.model_dir, "config.json")) or !std.mem.eql(u8, &stat_sha, &try sourceStatSha(a, io, opts.model_dir))) return error.Dsv41CaptureSourceChanged;
    if (!mode.lean) try finalizeSums(a, io, calib, prompts.items.len, dc);
    try glm.json(a, io, staging, "layout.json", layout(dc, mode));
    try glm.json(a, io, staging, "progress.json", .{ .complete = true, .phase = "complete", .completed_prompts = records.items.len, .tokens = run.committed_tokens });
    try cwd.renamePreserve(staging, cwd, opts.out_dir, io);
    out.print("[kld] layer-major DeepSeek-V4.1 captured {d} windows into {s}\n", .{ records.items.len, opts.out_dir });
}

/// The finished running totals become the layers' sums files.
fn finalizeSums(a: std.mem.Allocator, io: std.Io, calib: []const u8, windows: usize, dc: *const dsv_cfg.Config) !void {
    const cwd = std.Io.Dir.cwd();
    for (0..dc.layers + 1) |k| {
        const layer = if (k == dc.layers) head_layer else k;
        const from = try sumsPath(a, calib, layer, windows, false);
        defer a.free(from);
        const to = if (layer == head_layer)
            try std.fmt.allocPrint(a, "{s}/trunk-head.safetensors", .{calib})
        else
            try std.fmt.allocPrint(a, "{s}/trunk-{d:0>2}.safetensors", .{ calib, layer });
        defer a.free(to);
        try cwd.rename(from, cwd, to, io);
    }
}

const Layout = struct {
    schema: []const u8,
    layers: u32,
    hidden: u32,
    experts: u32,
    top_k: u32,
    mlp_in: ?[]const u8,
    route_ids: ?[]const u8,
    route_weights: ?[]const u8,
    tokens: []const u8,
    trunk_sums: ?[]const u8,
    imatrix: ?[]const u8,
    head_rows: ?[]const u8,
};

/// What the finished capture holds; a lean run lists only what it wrote.
fn layout(dc: *const dsv_cfg.Config, mode: Mode) Layout {
    const full = !mode.lean;
    return .{
        .schema = "sushi-dsv41-calibration-v1",
        .layers = dc.layers,
        .hidden = dc.dim,
        .experts = dc.experts,
        .top_k = dc.top_k,
        .mlp_in = if (full) "mlp-in-LL.bf16: [tokens, hidden] bf16 little-endian, tokens in window order" else null,
        .route_ids = if (full) "route-LL.u16: [tokens, top_k] expert ids" else null,
        .route_weights = if (full) "route-w-LL.f32: [tokens, top_k] the weights the forward applied (normalized, times the routed scale)" else null,
        .tokens = "tokens.bin: u32 ids of every window in order; ../windows.jsonl holds each window's length",
        .trunk_sums = if (full) "trunk-LL.safetensors, trunk-head.safetensors: layers.LL.<input> = sum over rows of x x^T in f32 ([in, in]; wo_a_in is [groups, in, in]) beside <name>.rows (I64); inputs: attn_in (wq_a, wkv, compressor, indexer weights), q_lora (wq_b, indexer wq_b), index_latent (indexer wk), wo_a_in, wo_b_in, ffn_in (shared w1/w3), shared_act (shared w2), engram_in (engram wkv), head_in (head)" else null,
        .imatrix = if (full) "imatrix.safetensors: layers.LL.ffn.experts.{gate_up_proj,down_proj}[.rows], per-expert second moments" else null,
        .head_rows = if (mode.head_rows) "head-in.bf16: [tokens, hidden] bf16 little-endian, tokens in window order; row r is the final RMSNorm output after the last hyper-connection collapse in the activation dtype (bf16), which the head multiplies after an f32 cast: logits = head @ row" else null,
    };
}

// ── tests: a tiny seeded DeepSeek pack (DSV41_TINY, tests/dsv41/dump_dsv41_fixtures.py) ──────────────────

const testing = std.testing;
const Transformer = @import("../transformer.zig").Transformer;

const tiny_tokenizer =
    \\{"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE",
    \\ "vocab":{"a":0,"b":1,"c":2,"d":3},"merges":[]}}
;
const tiny_lengths = [_]usize{ 5, 13, 12, 3, 10, 7, 11, 16, 9 };

/// A model directory of its own: the fixture pack's files and a tokenizer.
fn stageTiny(a: std.mem.Allocator, io: std.Io, dst: std.Io.Dir, src_dir: []const u8) !void {
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

fn tinyWindows(a: std.mem.Allocator) ![][]u32 {
    var prng = std.Random.DefaultPrng.init(7);
    const out = try a.alloc([]u32, tiny_lengths.len);
    for (out, tiny_lengths) |*w, n| {
        w.* = try a.alloc(u32, n);
        for (w.*) |*id| id.* = prng.random().uintLessThan(u32, 96);
    }
    return out;
}

fn writeTinyPrompts(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, windows: []const []const u32) !void {
    var body: std.Io.Writer.Allocating = .init(a);
    defer body.deinit();
    for (windows, 0..) |w, i| {
        try body.writer.print("{{\"id\":\"w{d}\",\"prompt_ids\":[", .{i});
        for (w, 0..) |id, k| try body.writer.print("{s}{d}", .{ if (k == 0) "" else ",", id });
        try body.writer.writeAll("]}\n");
    }
    try dir.writeFile(io, .{ .sub_path = "prompts.jsonl", .data = body.written() });
}

const TinyLoaded = struct {
    loaded: kld.Loaded,

    fn init(a: std.mem.Allocator, io: std.Io, root: []const u8) !*TinyLoaded {
        const self = try a.create(TinyLoaded);
        errdefer a.destroy(self);
        var config = try model.parseConfig(io, a, root);
        errdefer config.deinit(a);
        config.expert_streaming = true;
        config.expert_cache_bytes = 3 * config.num_hidden_layers * try dsv_stream.expertBytes(&config.dsv41.?);
        config.expert_source_dir = try a.dupe(u8, root);
        self.loaded = .{ .allocator = a, .io = io, .config = config, .tok = try @import("../tokenizer.zig").loadTokenizer(io, a, root), .chat_config = undefined, .weights = undefined, .xfm = undefined };
        errdefer self.loaded.tok.deinit();
        self.loaded.weights = try model.loadWeightsForConfig(io, a, root, &self.loaded.config, false);
        errdefer self.loaded.weights.deinit();
        self.loaded.xfm = try Transformer.init(io, a, self.loaded.config, &self.loaded.weights);
        return self;
    }

    fn deinit(self: *TinyLoaded) void {
        const a = self.loaded.allocator;
        self.loaded.xfm.deinit();
        self.loaded.weights.deinit();
        self.loaded.tok.deinit();
        self.loaded.config.deinit(a);
        a.destroy(self);
    }
};

/// Everything a forward feeds the linears, summed in f64 from the raw inputs: the capture's oracle.
const Recorder = struct {
    a: std.mem.Allocator,
    s: mlx.mlx_stream,
    layers: usize,
    sums: [][kinds]?[]f64,
    rows: [][kinds]u64,
    mlp: []std.ArrayList(u8),
    ids: []std.ArrayList(u8),
    weights: []std.ArrayList(u8),
    dspark_rows: [64]std.ArrayList(u8) = @splat(.empty),
    /// The head's input rows, bf16, in the order the forward made them.
    head: std.ArrayList(u8) = .empty,
    /// Per layer, the sum of the squared down-projection inputs of each expert, `[experts * inter]`.
    down: [][]f64,
    experts: usize,
    inter: usize,

    fn init(a: std.mem.Allocator, s: mlx.mlx_stream, dc: *const dsv_cfg.Config) !Recorder {
        const layers: usize = dc.layers;
        const sums = try a.alloc([kinds]?[]f64, layers + 1);
        for (sums) |*x| x.* = @splat(null);
        const rows = try a.alloc([kinds]u64, layers + 1);
        for (rows) |*x| x.* = @splat(0);
        const lists = try a.alloc(std.ArrayList(u8), 3 * layers);
        for (lists) |*l| l.* = .empty;
        const down = try a.alloc([]f64, layers);
        for (down) |*d| {
            d.* = try a.alloc(f64, dc.experts * dc.moe_inter);
            @memset(d.*, 0);
        }
        return .{ .a = a, .s = s, .layers = layers, .sums = sums, .rows = rows, .mlp = lists[0..layers], .ids = lists[layers .. 2 * layers], .weights = lists[2 * layers ..], .down = down, .experts = dc.experts, .inter = dc.moe_inter };
    }

    fn deinit(self: *Recorder) void {
        for (self.sums) |row| for (row) |x| if (x) |v| self.a.free(v);
        self.a.free(self.sums);
        self.a.free(self.rows);
        for (&self.dspark_rows) |*l| l.deinit(self.a);
        self.head.deinit(self.a);
        for (self.down) |d| self.a.free(d);
        self.a.free(self.down);
        const all = self.mlp.ptr[0 .. 3 * self.layers];
        for (all) |*l| l.deinit(self.a);
        self.a.free(all);
    }

    fn tap(self: *Recorder) dsv.Tap {
        return .{ .ctx = self, .input = input, .routes = routes, .expert_act = expertAct };
    }

    fn input(ctx: *anyopaque, layer: usize, kind: dsv.TapKind, x: A) anyerror!void {
        const self: *Recorder = @ptrCast(@alignCast(ctx));
        if (kind == .head_in) {
            var hsc = Scope.init(self.a, self.s);
            defer hsc.deinit();
            const bytes = try hostBytes(self.a, self.s, try hsc.astype(x, .bfloat16));
            defer self.a.free(bytes);
            try self.head.appendSlice(self.a, bytes);
        }
        if (kind == .ffn_in or kind == .dspark_in) {
            const bytes = try hostBytes(self.a, self.s, x);
            defer self.a.free(bytes);
            try (if (kind == .ffn_in) &self.mlp[layer] else &self.dspark_rows[layer]).appendSlice(self.a, bytes);
            if (kind == .dspark_in) return;
        }
        var sc = Scope.init(self.a, self.s);
        defer sc.deinit();
        const shape = mlx.getShape(x);
        const host = try sc_mod.hostF32(try sc.contiguous(try sc.astype(x, .float32)));
        const k = @intFromEnum(kind);
        const groups: usize = if (kind == .wo_a_in) @intCast(shape[1]) else 1;
        const width: usize = @intCast(shape[shape.len - 1]);
        const rows: usize = host.len / (groups * width);
        const sums = self.sums[layer][k] orelse blk: {
            const fresh = try self.a.alloc(f64, groups * width * width);
            @memset(fresh, 0);
            self.sums[layer][k] = fresh;
            break :blk fresh;
        };
        for (0..groups) |g| for (0..rows) |r| {
            const row = host[(r * groups + g) * width ..][0..width];
            const acc = sums[g * width * width ..][0 .. width * width];
            for (0..width) |i| for (0..width) |j| {
                acc[i * width + j] += @as(f64, row[i]) * row[j];
            };
        };
        self.rows[layer][k] += rows;
    }

    fn routes(ctx: *anyopaque, layer: usize, idx: A, wts: A) anyerror!void {
        const self: *Recorder = @ptrCast(@alignCast(ctx));
        const ids = try hostBytes(self.a, self.s, idx);
        defer self.a.free(ids);
        for (0..ids.len / 4) |i| {
            const id: u16 = @intCast(std.mem.readInt(i32, ids[i * 4 ..][0..4], .little));
            try self.ids[layer].appendSlice(self.a, std.mem.asBytes(&id));
        }
        const w = try hostBytes(self.a, self.s, wts);
        defer self.a.free(w);
        try self.weights[layer].appendSlice(self.a, w);
    }

    fn expertAct(ctx: *anyopaque, layer: usize, act: A, ids: A) anyerror!void {
        const self: *Recorder = @ptrCast(@alignCast(ctx));
        var sc = Scope.init(self.a, self.s);
        defer sc.deinit();
        const x = try sc_mod.hostF32(try sc.contiguous(try sc.astype(act, .float32)));
        const e = try sc_mod.hostI32(try sc.contiguous(try sc.astype(ids, .int32)));
        for (e, 0..) |id, r| for (0..self.inter) |c| {
            const v = x[r * self.inter + c];
            self.down[layer][@as(usize, @intCast(id)) * self.inter + c] += @as(f64, v) * v;
        };
    }
};

/// The imatrix a capture wrote: the gate/up second moments from the rows and routes it dumped, the down ones from the recorder.
fn expectImatrixMatches(a: std.mem.Allocator, io: std.Io, root: []const u8, name: []const u8, rec: *const Recorder, hidden: usize, top_k: usize) !void {
    var file = try SumsFile.open(a, io, root, "{s}/calib/imatrix.safetensors", .{name});
    defer file.deinit(a);
    for (0..rec.layers) |layer| {
        const rows_bytes = try readFile(a, io, root, "{s}/calib/mlp-in-{d:0>2}.bf16", .{ name, layer });
        defer a.free(rows_bytes);
        const route_bytes = try readFile(a, io, root, "{s}/calib/route-{d:0>2}.u16", .{ name, layer });
        defer a.free(route_bytes);
        const tokens = rows_bytes.len / (hidden * 2);
        const want_gu = try a.alloc(f64, rec.experts * hidden);
        defer a.free(want_gu);
        @memset(want_gu, 0);
        const want_rows = try a.alloc(f64, rec.experts);
        defer a.free(want_rows);
        @memset(want_rows, 0);
        for (0..tokens) |t| for (0..top_k) |k| {
            const e = std.mem.readInt(u16, route_bytes[(t * top_k + k) * 2 ..][0..2], .little);
            want_rows[e] += 1;
            for (0..hidden) |c| {
                const bits: u32 = @as(u32, std.mem.readInt(u16, rows_bytes[(t * hidden + c) * 2 ..][0..2], .little)) << 16;
                const v: f32 = @bitCast(bits);
                want_gu[e * hidden + c] += @as(f64, v) * v / @as(f64, @floatFromInt(tokens));
            }
        };
        var key: [96]u8 = undefined;
        const gu = file.tensor(try std.fmt.bufPrint(&key, "layers.{d}.ffn.experts.gate_up_proj", .{layer})) orelse return error.ImatrixMissing;
        var peak: f64 = 0;
        for (want_gu) |v| peak = @max(peak, v);
        for (want_gu, 0..) |v, i| {
            const g: f32 = @bitCast(std.mem.readInt(u32, gu[i * 4 ..][0..4], .little));
            if (@abs(@as(f64, g) - v) > 1e-3 * peak) {
                std.debug.print("layer {d} gate_up[{d}] is {e}, the rows say {e}\n", .{ layer, i, g, v });
                return error.ImatrixMismatch;
            }
        }
        const counts = file.tensor(try std.fmt.bufPrint(&key, "layers.{d}.ffn.experts.gate_up_proj.rows", .{layer})) orelse return error.ImatrixMissing;
        for (want_rows, 0..) |v, e| try testing.expectEqual(@as(f32, @floatCast(v)), @as(f32, @bitCast(std.mem.readInt(u32, counts[e * 4 ..][0..4], .little))));
        const down = file.tensor(try std.fmt.bufPrint(&key, "layers.{d}.ffn.experts.down_proj", .{layer})) orelse return error.ImatrixMissing;
        var down_peak: f64 = 0;
        for (rec.down[layer]) |v| down_peak = @max(down_peak, v / @as(f64, @floatFromInt(tokens)));
        for (rec.down[layer], 0..) |v, i| {
            const g: f32 = @bitCast(std.mem.readInt(u32, down[i * 4 ..][0..4], .little));
            if (@abs(@as(f64, g) - v / @as(f64, @floatFromInt(tokens))) > 1e-3 * down_peak) {
                std.debug.print("layer {d} down[{d}] is {e}, the activations say {e}\n", .{ layer, i, g, v / @as(f64, @floatFromInt(tokens)) });
                return error.ImatrixMismatch;
            }
        }
    }
}

/// The window-major forward: a fresh state per window and `chunk`-row extends, the last row's logits as f32 bytes.
fn windowMajorLogits(a: std.mem.Allocator, m: *dsv.Model, windows: []const []const u32, chunk: usize) ![][]u8 {
    const out = try a.alloc([]u8, windows.len);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |b| a.free(b);
        a.free(out);
    }
    for (windows, out) |ids, *slot| {
        dsv.resetState(m);
        var at: usize = 0;
        while (at < ids.len) : (at += chunk) {
            const logits = try dsv.extendSpan(m, ids[at..@min(ids.len, at + chunk)], .last, dsv.SPAN);
            defer _ = mlx.mlx_array_free(logits);
            if (at + chunk >= ids.len) slot.* = try hostBytes(a, m.s, logits);
        }
        made += 1;
    }
    dsv.resetState(m);
    return out;
}

fn readFile(a: std.mem.Allocator, io: std.Io, root: []const u8, comptime fmt: []const u8, args: anytype) ![]u8 {
    const path = try std.fmt.allocPrint(a, "{s}/" ++ fmt, .{root} ++ args);
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
}

const SumsFile = struct {
    raw: []u8,
    header: std.json.Parsed(std.json.Value),
    body_at: usize,

    fn open(a: std.mem.Allocator, io: std.Io, path_root: []const u8, comptime fmt: []const u8, args: anytype) !SumsFile {
        const raw = try readFile(a, io, path_root, fmt, args);
        errdefer a.free(raw);
        const hlen = std.mem.readInt(u64, raw[0..8], .little);
        return .{ .raw = raw, .header = try std.json.parseFromSlice(std.json.Value, a, raw[8 .. 8 + hlen], .{}), .body_at = @intCast(8 + hlen) };
    }

    fn deinit(self: *SumsFile, a: std.mem.Allocator) void {
        self.header.deinit();
        a.free(self.raw);
    }

    fn tensor(self: *const SumsFile, name: []const u8) ?[]const u8 {
        const t = (self.header.value.object.get(name) orelse return null).object;
        const off = t.get("data_offsets").?.array.items;
        return self.raw[self.body_at + @as(usize, @intCast(off[0].integer)) ..][0..@intCast(off[1].integer - off[0].integer)];
    }
};

fn expectSumsMatch(a: std.mem.Allocator, io: std.Io, calib: []const u8, rec: *const Recorder) !void {
    var checked: usize = 0;
    for (0..rec.layers + 1) |layer| {
        var file = if (layer == rec.layers)
            try SumsFile.open(a, io, calib, "trunk-head.safetensors", .{})
        else
            try SumsFile.open(a, io, calib, "trunk-{d:0>2}.safetensors", .{layer});
        defer file.deinit(a);
        for (std.enums.values(dsv.TapKind)) |kind| {
            if ((kind == .head_in) != (layer == rec.layers)) continue;
            const k = @intFromEnum(kind);
            var name_buf: [64]u8 = undefined;
            const name = if (layer == rec.layers) "head_in" else try std.fmt.bufPrint(&name_buf, "layers.{d}.{s}", .{ layer, @tagName(kind) });
            const want = rec.sums[layer][k] orelse {
                if (file.tensor(name) != null) {
                    std.debug.print("{s} is in the file and was never fed\n", .{name});
                    return error.SumsUnexpected;
                }
                continue;
            };
            const got = file.tensor(name) orelse return error.SumsMissing;
            try testing.expectEqual(want.len * 4, got.len);
            var peak: f64 = 0;
            for (want) |v| peak = @max(peak, @abs(v));
            for (want, 0..) |v, i| {
                const g: f32 = @bitCast(std.mem.readInt(u32, got[i * 4 ..][0..4], .little));
                if (@abs(@as(f64, g) - v) > 2e-5 * peak) {
                    std.debug.print("{s}: element {d} is {e}, X^T X is {e} (peak {e})\n", .{ name, i, g, v, peak });
                    return error.SumsMismatch;
                }
            }
            var rows_buf: [72]u8 = undefined;
            const rows = file.tensor(try std.fmt.bufPrint(&rows_buf, "{s}.rows", .{name})) orelse return error.SumsMissing;
            try testing.expectEqual(rec.rows[layer][k], std.mem.readInt(u64, rows[0..8], .little));
            checked += 1;
        }
    }
    try testing.expect(checked > rec.layers);
}

fn runTiny(a: std.mem.Allocator, io: std.Io, root: []const u8, tiny: *TinyLoaded, name: []const u8, batch: u32, chunk: usize) !void {
    var quiet: kld.Out = .{ .silent = true };
    const out_dir = try std.fmt.allocPrint(a, "{s}/{s}", .{ root, name });
    defer a.free(out_dir);
    const prompts = try std.fmt.allocPrint(a, "{s}/prompts.jsonl", .{root});
    defer a.free(prompts);
    try runLoaded(a, io, &tiny.loaded, .{ .command = .capture, .layer_major = true, .model_dir = root, .prompts = prompts, .out_dir = out_dir, .tokens = 1, .no_template = true }, batch, chunk, &quiet);
}

fn expectSameFile(a: std.mem.Allocator, io: std.Io, root: []const u8, left: []const u8, right: []const u8, comptime rel: []const u8) !void {
    const x = try readFile(a, io, root, "{s}/" ++ rel, .{left});
    defer a.free(x);
    const y = try readFile(a, io, root, "{s}/" ++ rel, .{right});
    defer a.free(y);
    try testing.expectEqualSlices(u8, x, y);
}

/// Every row, route and imatrix byte of two captures, and every window's logits.
fn expectSameCapture(a: std.mem.Allocator, io: std.Io, root: []const u8, left: []const u8, right: []const u8, layers: usize, windows: usize) !void {
    for (0..layers) |li| {
        inline for (.{ "calib/mlp-in-{d:0>2}.bf16", "calib/route-{d:0>2}.u16", "calib/route-w-{d:0>2}.f32" }) |pattern| {
            const x = try readFile(a, io, root, "{s}/" ++ pattern, .{ left, li });
            defer a.free(x);
            const y = try readFile(a, io, root, "{s}/" ++ pattern, .{ right, li });
            defer a.free(y);
            try testing.expect(x.len > 0);
            try testing.expectEqualSlices(u8, x, y);
        }
    }
    try expectSameFile(a, io, root, left, right, "calib/imatrix.safetensors");
    try expectSameFile(a, io, root, left, right, "calib/tokens.bin");
    for (0..windows) |w| {
        const x = try readFile(a, io, root, "{s}/prompts/{d:0>2}_w{d}/logits.f32", .{ left, w, w });
        defer a.free(x);
        const y = try readFile(a, io, root, "{s}/prompts/{d:0>2}_w{d}/logits.f32", .{ right, w, w });
        defer a.free(y);
        try testing.expectEqualSlices(u8, x, y);
    }
}

test "deepseek_v41 layer-major capture equals the window-major forward byte for byte at any batch size and chunk, and its trunk sums are X^T X (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src = try dsv.fixtureDir();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try stageTiny(a, io, tmp.dir, src);
    const windows = try tinyWindows(a);
    defer {
        for (windows) |w| a.free(w);
        a.free(windows);
    }
    try writeTinyPrompts(a, io, tmp.dir, windows);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const tiny = try TinyLoaded.init(a, io, root);
    defer tiny.deinit();
    const m = tiny.loaded.xfm.dsv41.?;
    const layers = m.cfg.layers;

    var first_dir: ?[]const u8 = null;
    defer if (first_dir) |d| a.free(d);
    for ([_]usize{ 4, 64 }) |chunk| {
        var rec = try Recorder.init(a, m.s, &m.cfg);
        defer rec.deinit();
        // A tapped forward runs every layer over every row, as the calibration files need. A window whose last chunk
        // is wider than the attention window takes the CED replay's logits, the served forward's, which the capture
        // forks beside its tapped pass.
        m.tap = rec.tap();
        const want_logits = try windowMajorLogits(a, m, windows, chunk);
        m.tap = null;
        const served = try windowMajorLogits(a, m, windows, chunk);
        defer {
            for (served) |b| a.free(b);
            a.free(served);
        }
        for (windows, want_logits, served) |ids, *x, y| if ((ids.len - 1) % chunk + 1 > m.cfg.window) @memcpy(x.*, y);
        defer {
            for (want_logits) |b| a.free(b);
            a.free(want_logits);
        }
        var base: ?[]const u8 = null;
        defer if (base) |b| a.free(b);
        for ([_]u32{ 1, 3 }) |batch| {
            const name = try std.fmt.allocPrint(a, "lm-c{d}-b{d}", .{ chunk, batch });
            defer if (base != null) a.free(name);
            try runTiny(a, io, root, tiny, name, batch, chunk);
            for (windows, 0..) |_, w| {
                const got = try readFile(a, io, root, "{s}/prompts/{d:0>2}_w{d}/logits.f32", .{ name, w, w });
                defer a.free(got);
                try testing.expectEqualSlices(u8, want_logits[w], got);
            }
            for (0..layers) |li| {
                inline for (.{ .{ "mlp-in-{d:0>2}.bf16", "mlp" }, .{ "route-{d:0>2}.u16", "ids" }, .{ "route-w-{d:0>2}.f32", "weights" } }) |pair| {
                    const got = try readFile(a, io, root, "{s}/calib/" ++ pair[0], .{ name, li });
                    defer a.free(got);
                    try testing.expect(got.len > 0);
                    try testing.expectEqualSlices(u8, @field(rec, pair[1])[li].items, got);
                }
            }
            for (m.cfg.dspark.targets[0..m.cfg.dspark.n_targets]) |t| {
                const got = try readFile(a, io, root, "{s}/calib/dspark-in-{d:0>2}.bf16", .{ name, t });
                defer a.free(got);
                try testing.expect(got.len > 0);
                try testing.expectEqualSlices(u8, rec.dspark_rows[t].items, got);
            }
            const calib = try std.fmt.allocPrint(a, "{s}/{s}/calib", .{ root, name });
            defer a.free(calib);
            try expectSumsMatch(a, io, calib, &rec);
            try expectImatrixMatches(a, io, root, name, &rec, m.cfg.dim, m.cfg.top_k);
            if (base) |b| {
                try expectSameCapture(a, io, root, b, name, layers, windows.len);
            } else {
                base = try a.dupe(u8, name);
                if (first_dir == null) first_dir = try a.dupe(u8, name);
            }
        }
    }
    try testing.expect(tiny.loaded.xfm.dsv41_experts.?.pinned_leases > 0);
}

test "deepseek_v41 layer-major logits are the generic window-major capture's fixture, file for file (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src = try dsv.fixtureDir();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try stageTiny(a, io, tmp.dir, src);
    const windows = try tinyWindows(a);
    defer {
        for (windows) |w| a.free(w);
        a.free(windows);
    }
    try writeTinyPrompts(a, io, tmp.dir, windows);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const tiny = try TinyLoaded.init(a, io, root);
    defer tiny.deinit();

    var quiet: kld.Out = .{ .silent = true };
    const prompts = try std.fmt.allocPrint(a, "{s}/prompts.jsonl", .{root});
    defer a.free(prompts);
    const generic = try std.fmt.allocPrint(a, "{s}/generic", .{root});
    defer a.free(generic);
    try kld.runCapture(io, a, &tiny.loaded, .{ .model_dir = root, .prompts = prompts, .out_dir = generic, .tokens = 1, .no_template = true }, &quiet);
    try runTiny(a, io, root, tiny, "lm", 4, chunk_rows);
    for (0..windows.len) |w| {
        inline for (.{ "logits.f32", "prompt_tokens.txt", "generated_tokens.txt" }) |leaf| {
            const x = try readFile(a, io, root, "generic/prompts/{d:0>2}_w{d}/" ++ leaf, .{ w, w });
            defer a.free(x);
            const y = try readFile(a, io, root, "lm/prompts/{d:0>2}_w{d}/" ++ leaf, .{ w, w });
            defer a.free(y);
            try testing.expectEqualSlices(u8, x, y);
        }
    }
    // The fixture reads back through compare's own loader.
    const lm = try std.fmt.allocPrint(a, "{s}/lm", .{root});
    defer a.free(lm);
    var base = try kld.readBaseline(a, io, lm);
    defer base.deinit();
    try testing.expectEqual(windows.len, base.prompts.len);
}

test "deepseek_v41 layer-major capture interrupted inside a batch resumes to the uninterrupted bytes, at another batch size too (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src = try dsv.fixtureDir();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try stageTiny(a, io, tmp.dir, src);
    const windows = try tinyWindows(a);
    defer {
        for (windows) |w| a.free(w);
        a.free(windows);
    }
    try writeTinyPrompts(a, io, tmp.dir, windows);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const tiny = try TinyLoaded.init(a, io, root);
    defer tiny.deinit();
    const layers = tiny.loaded.xfm.dsv41.?.cfg.layers;

    try runTiny(a, io, root, tiny, "whole", 3, 4);
    // Batch one is windows 0..2: 2 + 4 + 3 chunk passes per layer.
    const batch_one = (2 + 4 + 3) * layers;
    for ([_]struct { name: []const u8, again: u32 }{ .{ .name = "same", .again = 3 }, .{ .name = "other", .again = 2 } }) |c| {
        interrupt_layers_for_test = batch_one + 20;
        defer interrupt_layers_for_test = null;
        try testing.expectError(error.TestInterrupted, runTiny(a, io, root, tiny, c.name, 3, 4));
        interrupt_layers_for_test = null;
        // A different run's command refuses to adopt the staging.
        try testing.expectError(error.Dsv41LayerMajorResumeMismatch, runTiny(a, io, root, tiny, c.name, 3, 5));
        try runTiny(a, io, root, tiny, c.name, c.again, 4);
        try expectSameCapture(a, io, root, "whole", c.name, layers, windows.len);
        if (c.again == 3) {
            for (0..layers) |li| {
                const x = try readFile(a, io, root, "whole/calib/trunk-{d:0>2}.safetensors", .{li});
                defer a.free(x);
                const y = try readFile(a, io, root, "{s}/calib/trunk-{d:0>2}.safetensors", .{ c.name, li });
                defer a.free(y);
                try testing.expectEqualSlices(u8, x, y);
            }
        }
    }
}

test "deepseek_v41 layer-major capture: flags it takes and refuses, and the batch the budget buys" {
    const cfg = model.ModelConfig{ .model_type = "deepseek_v41" };
    const ok = kld.Options{ .command = .capture, .layer_major = true, .no_template = true, .tokens = 1, .ssd_budget_bytes = 100 << 30 };
    try testing.expect(try accepts(&cfg, ok));
    try testing.expect(!try accepts(&cfg, .{ .command = .capture, .no_template = true, .tokens = 1, .ssd_budget_bytes = 100 << 30 }));
    try testing.expect(!try accepts(&model.ModelConfig{ .model_type = "glm5_next" }, ok));
    var bad = ok;
    bad.tokens = 4;
    try testing.expectError(error.Dsv41LayerMajorNeedsOneRow, accepts(&cfg, bad));
    bad = ok;
    bad.hidden_out = "/x";
    try testing.expectError(error.Dsv41HiddenCaptureUnsupported, accepts(&cfg, bad));
    bad = ok;
    bad.ssd_budget_bytes = 0;
    try testing.expectError(error.Dsv41CaptureNeedsLosslessStreaming, accepts(&cfg, bad));
    bad = ok;
    bad.imatrix_out = "/x.safetensors";
    try testing.expect(try accepts(&cfg, bad));
    try refuse(&cfg, loadOptions(bad, 1 << 30));

    // The release's geometry.
    var dc = dsv_cfg.Config{ .vocab = 129280, .dim = 5120, .layers = 40, .heads = 64, .head_dim = 512, .q_lora = 1280, .o_lora = 1024, .o_groups = 8, .window = 128, .experts = 384, .top_k = 6, .moe_inter = 2304, .hc_mult = 4, .index_heads = 64, .index_head_dim = 128, .index_topk = 512 };
    dc.engram = .{ .n = 2, .max_ngram = 4, .heads = 8, .head_dim = 256 };
    const per_expert = try dsv_stream.expertBytes(&dc);
    try testing.expectEqual(@as(u64, 18_800_640), per_expert);
    const floor = try floorBudget(&dc, 12 << 30, per_expert);
    try testing.expectEqual(@as(u32, batch_cap), try batchWindows(&dc, 100 << 30, floor, chunk_rows));
    try testing.expectError(error.Dsv41CaptureBudgetTooSmall, batchWindows(&dc, floor, floor, chunk_rows));
    const small = try batchWindows(&dc, floor + (14 << 30), floor, chunk_rows);
    try testing.expect(small >= 1 and small < batch_cap);
}

/// The head applied to row `row` of the stored head inputs must give the logits the capture wrote for that window.
fn expectHeadRowLogits(a: std.mem.Allocator, io: std.Io, root: []const u8, name: []const u8, m: *dsv.Model, head_rows: []const u8, windows: []const []const u32) !void {
    const d = m.cfg.dim;
    var at: usize = 0;
    for (windows, 0..) |ids, w| {
        at += ids.len;
        const raw = head_rows[(at - 1) * d * 2 ..][0 .. d * 2];
        const row = try a.alloc(f32, d);
        defer a.free(row);
        for (row, 0..) |*o, i| o.* = @bitCast(@as(u32, std.mem.readInt(u16, raw[i * 2 ..][0..2], .little)) << 16);
        var sc = Scope.init(a, m.s);
        defer sc.deinit();
        const got = try sc_mod.hostF32(try sc.contiguous(try m.head.apply(&sc, try sc.floats(row, &.{ 1, sc_mod.ci(d) }))));
        const want_bytes = try readFile(a, io, root, "{s}/prompts/{d:0>2}_w{d}/logits.f32", .{ name, w, w });
        defer a.free(want_bytes);
        const want = std.mem.bytesAsSlice(f32, want_bytes);
        try testing.expectEqual(want.len, got.len);
        var scale: f32 = 0;
        var worst: f32 = 0;
        for (want, got) |x, y| {
            scale = @max(scale, @abs(x));
            worst = @max(worst, @abs(x - y));
        }
        try testing.expect(scale > 0);
        try testing.expect(worst <= 1e-3 * scale);
    }
}

fn expectMissing(a: std.mem.Allocator, io: std.Io, root: []const u8, comptime fmt: []const u8, args: anytype) !void {
    const path = try std.fmt.allocPrint(a, "{s}/" ++ fmt, .{root} ++ args);
    defer a.free(path);
    try testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, path, .{}));
}

fn runTinyMode(a: std.mem.Allocator, io: std.Io, root: []const u8, tiny: *TinyLoaded, name: []const u8, batch: u32, chunk: usize, mode: Mode) !void {
    var quiet: kld.Out = .{ .silent = true };
    const out_dir = try std.fmt.allocPrint(a, "{s}/{s}", .{ root, name });
    defer a.free(out_dir);
    const prompts = try std.fmt.allocPrint(a, "{s}/prompts.jsonl", .{root});
    defer a.free(prompts);
    try runLoadedMode(a, io, &tiny.loaded, .{ .command = .capture, .layer_major = true, .model_dir = root, .prompts = prompts, .out_dir = out_dir, .tokens = 1, .no_template = true }, batch, chunk, &quiet, mode);
}

test "deepseek_v41 layer-major capture: head rows are the head's input for every row, a lean run writes only those, and both resume (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const src = try dsv.fixtureDir();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try stageTiny(a, io, tmp.dir, src);
    const windows = try tinyWindows(a);
    defer {
        for (windows) |w| a.free(w);
        a.free(windows);
    }
    try writeTinyPrompts(a, io, tmp.dir, windows);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const tiny = try TinyLoaded.init(a, io, root);
    defer tiny.deinit();
    const m = tiny.loaded.xfm.dsv41.?;
    const layers = m.cfg.layers;
    var tokens: usize = 0;
    for (windows) |w| tokens += w.len;

    var rec = try Recorder.init(a, m.s, &m.cfg);
    defer rec.deinit();
    m.tap = rec.tap();
    const want_logits = try windowMajorLogits(a, m, windows, 4);
    m.tap = null;
    defer {
        for (want_logits) |b| a.free(b);
        a.free(want_logits);
    }
    try testing.expectEqual(tokens * m.cfg.dim * 2, rec.head.items.len);

    // Off by default.
    try runTinyMode(a, io, root, tiny, "off", 3, 4, .{});
    try expectMissing(a, io, root, "off/calib/head-in.bf16", .{});

    try runTinyMode(a, io, root, tiny, "full", 3, 4, .{ .head_rows = true });
    const full = try readFile(a, io, root, "full/calib/head-in.bf16", .{});
    defer a.free(full);
    try testing.expectEqualSlices(u8, rec.head.items, full);
    try expectHeadRowLogits(a, io, root, "full", m, full, windows);
    const layout_json = try readFile(a, io, root, "full/calib/../layout.json", .{});
    defer a.free(layout_json);
    try testing.expect(std.mem.indexOf(u8, layout_json, "\"head_rows\": \"head-in.bf16") != null);
    for (0..windows.len) |w| {
        const got = try readFile(a, io, root, "full/prompts/{d:0>2}_w{d}/logits.f32", .{ w, w });
        defer a.free(got);
        try testing.expectEqualSlices(u8, want_logits[w], got);
    }

    // Lean: the same rows and logits, and none of the other calibration outputs.
    try runTinyMode(a, io, root, tiny, "lean", 2, 4, .{ .head_rows = true, .lean = true });
    const lean = try readFile(a, io, root, "lean/calib/head-in.bf16", .{});
    defer a.free(lean);
    try testing.expectEqualSlices(u8, full, lean);
    try expectSameFile(a, io, root, "full", "lean", "calib/tokens.bin");
    try expectSameFile(a, io, root, "full", "lean", "windows.jsonl");
    for (0..windows.len) |w| {
        const x = try readFile(a, io, root, "full/prompts/{d:0>2}_w{d}/logits.f32", .{ w, w });
        defer a.free(x);
        const y = try readFile(a, io, root, "lean/prompts/{d:0>2}_w{d}/logits.f32", .{ w, w });
        defer a.free(y);
        try testing.expectEqualSlices(u8, x, y);
    }
    try expectMissing(a, io, root, "lean/calib/mlp-in-00.bf16", .{});
    try expectMissing(a, io, root, "lean/calib/route-00.u16", .{});
    try expectMissing(a, io, root, "lean/calib/route-w-00.f32", .{});
    try expectMissing(a, io, root, "lean/calib/trunk-00.safetensors", .{});
    try expectMissing(a, io, root, "lean/calib/trunk-head.safetensors", .{});
    try expectMissing(a, io, root, "lean/calib/imatrix.safetensors", .{});
    for (m.cfg.dspark.targets[0..m.cfg.dspark.n_targets]) |t| try expectMissing(a, io, root, "lean/calib/dspark-in-{d:0>2}.bf16", .{t});
    const lean_dir = try std.fmt.allocPrint(a, "{s}/lean", .{root});
    defer a.free(lean_dir);
    var base = try kld.readBaseline(a, io, lean_dir);
    base.deinit();

    // A run interrupted inside the second batch resumes to the uninterrupted rows, lean or not; another mode is refused.
    const batch_one = (2 + 4 + 3) * layers;
    for ([_]struct { name: []const u8, mode: Mode }{ .{ .name = "cut-full", .mode = .{ .head_rows = true } }, .{ .name = "cut-lean", .mode = .{ .head_rows = true, .lean = true } } }) |c| {
        interrupt_layers_for_test = batch_one + 20;
        defer interrupt_layers_for_test = null;
        try testing.expectError(error.TestInterrupted, runTinyMode(a, io, root, tiny, c.name, 3, 4, c.mode));
        interrupt_layers_for_test = null;
        const other: Mode = if (c.mode.lean) .{ .head_rows = true } else .{};
        try testing.expectError(error.Dsv41LayerMajorResumeMismatch, runTinyMode(a, io, root, tiny, c.name, 3, 4, other));
        try runTinyMode(a, io, root, tiny, c.name, 3, 4, c.mode);
        const got = try readFile(a, io, root, "{s}/calib/head-in.bf16", .{c.name});
        defer a.free(got);
        try testing.expectEqualSlices(u8, full, got);
    }
}

test "deepseek_v41 capture: lean needs head rows" {
    try testing.expectError(error.Dsv41CaptureLeanNeedsHeadRows, checkedMode(false, true));
}
