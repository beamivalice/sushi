const std = @import("std");
const host = @import("mlx_host");
const mlx = host.mlx;
const kld = host.kld;
const imatrix = host.imatrix;
const format = @import("sushi_exl3").format;
const calib = @import("calib.zig");
const convert = @import("convert.zig");
const driver = @import("driver.zig");
const hessians = @import("hessians.zig");
const layer_stage = @import("layer_stage.zig");
const pack = @import("pack.zig");
const source = @import("source.zig");

const stage_a = "stage-capture";
const stage_b = "stage-forward";

const max_rank = 4;

pub const Hidden = struct {
    allocator: std.mem.Allocator,
    count: usize,
    rank: usize,
    shape: [max_rank]c_int,
    dtype: mlx.mlx_dtype,
    itemsize: usize,
    window_bytes: usize,
    bytes: []u8,
    boundary: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator, count: usize, extents: []const c_int, dtype: mlx.mlx_dtype, itemsize: usize) !Hidden {
        if (extents.len == 0 or extents.len > max_rank) return error.HiddenShapeMismatch;
        var shape: [max_rank]c_int = @splat(1);
        var per: usize = itemsize;
        for (extents, 0..) |d, i| {
            shape[i] = d;
            per *= @intCast(d);
        }
        return .{ .allocator = allocator, .count = count, .rank = extents.len, .shape = shape, .dtype = dtype, .itemsize = itemsize, .window_bytes = per, .bytes = try allocator.alloc(u8, per * count) };
    }

    pub fn dims(self: *const Hidden) []const c_int {
        return self.shape[0..self.rank];
    }

    pub fn deinit(self: *Hidden) void {
        self.allocator.free(self.bytes);
    }

    pub fn array(self: *const Hidden, i: usize) mlx.mlx_array {
        return mlx.mlx_array_new_data(self.bytes[i * self.window_bytes ..].ptr, &self.shape, @intCast(self.rank), self.dtype);
    }

    pub fn store(self: *Hidden, s: mlx.mlx_stream, i: usize, arr: mlx.mlx_array) !void {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_contiguous(&flat, arr, false, s));
        try mlx.check(mlx.mlx_array_eval(flat));
        if (mlx.mlx_array_size(flat) * mlx.mlx_array_itemsize(flat) != self.window_bytes) return error.HiddenShapeMismatch;
        const data = mlx.mlx_array_data_uint8(flat) orelse return error.HiddenUnreadable;
        @memcpy(self.bytes[i * self.window_bytes ..][0..self.window_bytes], data[0..self.window_bytes]);
    }
};

fn hiddenName(buf: []u8, boundary: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "hidden-{d}.bin", .{boundary});
}

pub const RowsTrace = struct {
    x: []f32,
    ids: []i32,
    topk: usize,
};

pub const Capture = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    s: mlx.mlx_stream,
    windows: *const calib.Windows,
    src: *const source.SafetensorsSource,
    scratch: std.Io.Dir = undefined,
    scratch_abs: []u8 = &.{},
    trace: ?*std.ArrayList([]u8) = null,
    rows_trace: ?*std.ArrayList(RowsTrace) = null,
    ssd_budget_bytes: u64,
    info: ?driver.Info = null,
    hidden: ?Hidden = null,
    rows: ?imatrix.CapturedRows = null,
    rows_h: ?hessians.RowsHessians = null,
    hessian_stream: ?mlx.mlx_stream = null,

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

    fn dropRows(self: *Capture) void {
        if (self.rows_h) |*r| r.deinit();
        self.rows_h = null;
        if (self.rows) |r| {
            _ = mlx.mlx_array_free(r.x);
            self.allocator.free(r.ids);
        }
        self.rows = null;
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
    }

    fn ready(ctx: *anyopaque, layer: u32, last: bool) bool {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        if (last) return true;
        var buf: [48]u8 = undefined;
        const name = hiddenName(&buf, layer + 1) catch return false;
        _ = self.scratch.statFile(self.io, name, .{}) catch return false;
        return true;
    }

    fn windowTokens(self: *Capture, i: usize) mlx.mlx_array {
        return windowTokensOf(self.allocator, self.windows, i);
    }

    fn loadBoundary(self: *Capture, layer: u32) !void {
        if (self.hidden) |h| if (h.boundary != null and h.boundary.? == layer) return;
        var buf: [48]u8 = undefined;
        const name = try hiddenName(&buf, layer);
        const raw = try self.scratch.readFileAlloc(self.io, name, self.allocator, .limited(1 << 40));
        defer self.allocator.free(raw);
        var header: [max_rank + 3]c_int = undefined;
        if (raw.len < @sizeOf(@TypeOf(header))) return error.HiddenScratchCorrupt;
        @memcpy(std.mem.sliceAsBytes(&header), raw[0..@sizeOf(@TypeOf(header))]);
        if (header[0] < 1 or header[0] > max_rank) return error.HiddenScratchCorrupt;
        const dtype: mlx.mlx_dtype = @enumFromInt(header[max_rank + 1]);
        var h = try Hidden.init(self.allocator, self.windows.count, header[1..][0..@intCast(header[0])], dtype, @intCast(header[max_rank + 2]));
        errdefer h.deinit();
        const payload = raw[@sizeOf(@TypeOf(header))..];
        if (payload.len != h.bytes.len) return error.HiddenScratchCorrupt;
        @memcpy(h.bytes, payload);
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
            const header = [max_rank + 3]c_int{ @intCast(h.rank), h.shape[0], h.shape[1], h.shape[2], h.shape[3], @intFromEnum(h.dtype), @intCast(h.itemsize) };
            try file.writeStreamingAll(self.io, std.mem.sliceAsBytes(&header));
            try pack.writeAllChunked(file, self.io, h.bytes);
        }
        try self.scratch.rename(tmp, self.scratch, name, self.io);
        if (self.trace) |list| try list.append(self.allocator, try self.allocator.dupe(u8, h.bytes[0..h.window_bytes]));
        if (layer > 0) {
            var old_buf: [48]u8 = undefined;
            self.scratch.deleteFile(self.io, try hiddenName(&old_buf, layer - 1)) catch {};
        }
    }

    fn stageFor(self: *Capture, name: []const u8, layer: u32, experts: layer_stage.Stage.Experts) ![]u8 {
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
            .has_ngram = info.has_ngram,
            .eq = info.eq,
        });
        return std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ self.scratch_abs, name });
    }

    fn forwardWindows(self: *Capture, loaded: *kld.Loaded, layer: u32, out: ?*?Hidden) !void {
        const info = self.info.?;
        for (0..self.windows.count) |w| {
            try loaded.xfm.resetCache();
            var ctx = loaded.xfm.defaultCtx();
            ctx.skip_lm_head = true;
            ctx.skip_ple = layer != info.ple_layer;
            const input = if (layer > 0) self.hidden.?.array(w) else mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(input);
            if (layer > 0) ctx.stream_in = input;
            var slot = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(slot);
            const ids = [_]u32{0};
            var cl = host.transformer.CaptureLayers{ .ids = &ids, .out = (&slot)[0..1] };
            if (out != null) ctx.capture_layers = &cl;
            const tokens = self.windowTokens(w);
            defer _ = mlx.mlx_array_free(tokens);
            const result = try loaded.xfm.forwardWith(&ctx, tokens);
            defer _ = mlx.mlx_array_free(result);
            try mlx.check(mlx.mlx_array_eval(result));
            if (out) |o| {
                if (o.* == null) {
                    o.* = try Hidden.init(self.allocator, self.windows.count, mlx.getShape(slot), mlx.mlx_array_dtype(slot), mlx.mlx_array_itemsize(slot));
                }
                try o.*.?.store(self.s, w, slot);
            }
        }
    }

    fn begin(ctx: *anyopaque, layer: u32, mtp: bool) anyerror!?convert.HessianSource {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        if (mtp) return null;
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
        if (self.rows_trace) |list| try list.append(self.allocator, try copyRows(self.allocator, self.s, self.rows.?));
        const info = self.info.?;
        self.rows_h = try hessians.RowsHessians.init(self.allocator, self.hessianStream(), self.src, layer, false, info.geom, self.rows.?.x, self.rows.?.ids, self.rows.?.topk);
        return self.rows_h.?.hessianSource();
    }

    fn done(ctx: *anyopaque, layer: u32, mtp: bool, reused: bool) anyerror!void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        if (mtp or reused) return;
        self.dropRows();
        const info = self.info.?;
        if (layer + 1 >= info.layers) return;
        var name_buf: [64]u8 = undefined;
        const shard_name = try std.fmt.bufPrint(&name_buf, "layer-{d}.safetensors", .{layer});
        const dir = try self.stageFor(stage_b, layer, .{ .shard = shard_name });
        defer self.allocator.free(dir);
        var loaded = try kld.loadModel(self.io, self.allocator, .{ .model_dir = dir });
        defer loaded.deinit();
        if (layer > 0) try self.loadBoundary(layer);
        var next: ?Hidden = null;
        errdefer if (next) |*h| h.deinit();
        try self.forwardWindows(loaded, layer, &next);
        next.?.boundary = layer + 1;
        if (self.hidden) |*old| old.deinit();
        self.hidden = next.?;
        next = null;
        try self.saveBoundary(layer + 1);
    }

    fn finish(ctx: *anyopaque) void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        self.scratch.deleteTree(self.io, stage_a) catch {};
        self.scratch.deleteTree(self.io, stage_b) catch {};
    }
};

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
    }, cap.calibrator(), null);
    try t.expectEqual(@as(usize, layers), summary.layers_written);
    try t.expectEqual(@as(usize, layers - 1), trace.items.len);
    try t.expectEqual(@as(usize, layers), streamed.items.len);

    var full = try kld.loadModel(io, t.allocator, .{ .model_dir = pack_abs });
    defer full.deinit();
    try full.xfm.resetCache();
    var ctx = full.xfm.defaultCtx();
    ctx.skip_lm_head = true;
    const layer_ids = [_]u32{ 0, 1, 2, 3 };
    var outs: [layers]mlx.mlx_array = undefined;
    for (&outs) |*o| o.* = mlx.mlx_array_new();
    defer for (outs) |o| {
        _ = mlx.mlx_array_free(o);
    };
    var cl = host.transformer.CaptureLayers{ .ids = &layer_ids, .out = &outs };
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

test "capture: a hidden boundary survives a save and a load, an older boundary is dropped, and ready names only a saved one" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "scratch");
    var ids: [calib.window_len * 2]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i);
    var windows = try calib.fromTokens(t.allocator, &ids, ids.len);
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

    var h = try Hidden.init(t.allocator, 2, &.{ 1, 4, 16 }, .bfloat16, 2);
    for (h.bytes, 0..) |*b, i| b.* = @truncate(i *% 7);
    h.boundary = 3;
    cap.hidden = h;
    try cap.saveBoundary(3);
    try t.expectEqual(@as(usize, 1), trace.items.len);
    try t.expectEqualSlices(u8, h.bytes[0..h.window_bytes], trace.items[0]);
    const want = try t.allocator.dupe(u8, h.bytes);
    defer t.allocator.free(want);

    try t.expect(Capture.ready(&cap, 2, false));
    try t.expect(!Capture.ready(&cap, 1, false));
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
    try t.expect(!Capture.ready(&cap, 2, false));
    try t.expect(Capture.ready(&cap, 3, false));

    var copy = try Hidden.init(t.allocator, 2, &.{ 1, 4, 16 }, .bfloat16, 2);
    defer copy.deinit();
    const arr = cap.hidden.?.array(1);
    defer _ = mlx.mlx_array_free(arr);
    try copy.store(mlx.gpuStream(), 1, arr);
    try t.expectEqualSlices(u8, want[cap.hidden.?.window_bytes..], copy.bytes[copy.window_bytes..]);
}
