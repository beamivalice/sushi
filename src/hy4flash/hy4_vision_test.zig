//! Hy4-Flash's vision (part 1v) on the tiny: `HY4_BF16_TINY` with `oracle_vision.safetensors`
//! (tests/dump_hy4_fixtures.py --vision), transformers' f32 forward with llama.cpp's tanh GELU.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const model_mod = @import("../model.zig");
const hy4_config = @import("hy4_config.zig");
const hy4_vision = @import("hy4_vision.zig");
const fwd = @import("hy4_forward.zig");
const Arr = mlx.mlx_array;

/// The bar the tower's stages are held to in f32, relative to each stage's peak. A tanh GELU read as erf moves the
/// merged rows by 2.7e-4 of their peak (the dump checks it), so this bar tells the two apart thirteen times over.
const tower_bar: f64 = 2e-5;

const Fixture = struct {
    pack: model_mod.Weights,
    oracle: model_mod.Weights,
    config: hy4_config.Config,

    fn open(s: mlx.mlx_stream) !Fixture {
        const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
        const a = testing.allocator;
        var f: Fixture = .{ .pack = undefined, .oracle = undefined, .config = undefined };
        f.pack = try load(dir, "model.safetensors", s);
        errdefer f.pack.deinit();
        f.oracle = load(dir, "oracle_vision.safetensors", s) catch |e| {
            if (e == error.FileNotFound) return error.SkipZigTest;
            return e;
        };
        errdefer f.oracle.deinit();
        const io = std.Io.Threaded.global_single_threaded.io();
        var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
        defer d.close(io);
        const text = try d.readFileAlloc(io, "config.json", a, .limited(1 << 20));
        defer a.free(text);
        f.config = try hy4_config.parseText(text);
        return f;
    }

    fn close(f: *Fixture) void {
        f.pack.deinit();
        f.oracle.deinit();
    }

    fn ref(f: *const Fixture, name: []const u8) !Arr {
        return f.oracle.get(name) orelse {
            std.debug.print("oracle_vision has no {s}\n", .{name});
            return error.MissingTensor;
        };
    }
};

fn load(dir: []const u8, name: []const u8, s: mlx.mlx_stream) !model_mod.Weights {
    const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/{s}", .{ dir, name }, 0);
    defer testing.allocator.free(path);
    std.Io.Dir.cwd().access(std.Io.Threaded.global_single_threaded.io(), path, .{}) catch return error.FileNotFound;
    var w = model_mod.Weights.init(testing.allocator);
    errdefer w.deinit();
    try model_mod.loadSafetensorsFile(testing.allocator, &w, path.ptr, s, true);
    return w;
}

fn host(o: *Ops, a: Arr) ![]const f32 {
    const c = try o.contiguous(try o.cast(a, .float32));
    try mlx.check(mlx.mlx_array_eval(c));
    return mlx.mlx_array_data_float32(c).?[0..mlx.mlx_array_size(c)];
}

fn expectNear(label: []const u8, o: *Ops, got: Arr, want: Arr, bar: f64) !void {
    const g = try host(o, got);
    const r = try host(o, want);
    if (g.len != r.len) {
        std.debug.print("{s}: {d} values, the oracle has {d}\n", .{ label, g.len, r.len });
        return error.ShapeMismatch;
    }
    var peak: f64 = 1e-30;
    for (r) |v| peak = @max(peak, @abs(v));
    var worst: f64 = 0;
    for (g, r) |x, y| worst = @max(worst, @abs(@as(f64, x) - y));
    if (std.c.getenv("HY4_VISION_DIAG") != null) std.debug.print("[hy4-vision] {s}: worst {e:.3} of the peak {e:.3}\n", .{ label, worst / peak, peak });
    if (!(worst <= bar * peak)) {
        std.debug.print("{s}: worst |Δ| {e:.3} of the peak {e:.3}, bar {e:.1}\n", .{ label, worst / peak, peak, bar });
        return error.OutsideBar;
    }
}

test "hy4 vision: the tiny's image through the tower (patch embedding and positions, each block) and the merger gives the oracle's rows within the tower bar, the merged rows (H/2) x (W/2 + 1) with the newline closing each (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const products = fwd.products;
    fwd.products = .f32;
    defer fwd.products = products;
    var f = try Fixture.open(es);
    defer f.close();
    const v = f.config.vision orelse return error.Hy4VisionMissing;
    var tower = try hy4_vision.Tower.init(a, v, &f.pack);
    defer tower.deinit();
    // Attention by chunks of 5 query rows: the tiny's 24 patches take five, the last one short.
    tower.attn_rows = 5;
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const grid = try host(&o, try f.ref("vis.grid"));
    const gh: u32 = @intFromFloat(grid[1]);
    const gw: u32 = @intFromFloat(grid[2]);
    var blocks: [8]Arr = undefined;
    var stages: hy4_vision.Tower.Stages = .{ .blocks = blocks[0..v.layers] };
    const rows = try tower.forward(&o, try f.ref("vis.pixels"), gh, gw, &stages);
    try testing.expectEqual(tower.rowsOf(gh, gw), @as(u32, @intCast(mlx.getShape(rows)[0])));
    try expectNear("patch embedding and positions", &o, stages.embed.?, try f.ref("vis.embed"), tower_bar);
    for (0..v.layers) |i| {
        var lb: [32]u8 = undefined;
        var nb: [32]u8 = undefined;
        try expectNear(try std.fmt.bufPrint(&lb, "block {d}", .{i}), &o, blocks[i], try f.ref(try std.fmt.bufPrint(&nb, "vis.block{d}", .{i})), tower_bar);
    }
    try expectNear("merged rows", &o, rows, try f.ref("vis.merged"), tower_bar);
}

/// The bar the logits are held to, as the text oracle's (`hy4_oracle_test`): within 2e-3 of their peak.
const logits_bar: f64 = 2e-3;

test "hy4 vision: the tiny's image prompt (the tower's rows spliced at the image ids, consecutive positions, the image ids dead and read as the pad id in Engram's history) prefills and decodes to the oracle's logits (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const products = fwd.products;
    fwd.products = .f32;
    defer fwd.products = products;
    var f = try Fixture.open(es);
    defer f.close();
    const v = f.config.vision orelse return error.Hy4VisionMissing;
    var tower = try hy4_vision.Tower.init(a, v, &f.pack);
    defer tower.deinit();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const grid = try host(&o, try f.ref("vis.grid"));
    const image = try tower.forward(&o, try f.ref("vis.pixels"), @intFromFloat(grid[1]), @intFromFloat(grid[2]), null);
    const ids_f = try host(&o, try f.ref("vis.ids"));
    var ids: [64]u32 = undefined;
    for (ids_f, ids[0..ids_f.len]) |x, *d| d.* = @intFromFloat(x);
    const steps = 4;
    const prompt = ids[0 .. ids_f.len - steps];
    const m = try hy4_model.Model.init(a, f.config, &f.pack, es);
    defer m.deinit();
    // The oracle's latent is BF16, as a BF16 cache holds it.
    m.req().kv = .dense;
    const want = try host(&o, try f.ref("vis.logits"));
    const vocab = f.config.vocab;
    const pre = try o.own(try m.forwardSpliced(prompt, .{ .rows = image }, .all));
    if (std.c.getenv("HY4_VISION_DIAG") != null) {
        const g = try host(&o, pre);
        for (0..prompt.len) |r| {
            var worst: f64 = 0;
            var peak: f64 = 1e-30;
            for (g[r * vocab ..][0..vocab], want[r * vocab ..][0..vocab]) |x, y| {
                worst = @max(worst, @abs(@as(f64, x) - y));
                peak = @max(peak, @abs(y));
            }
            std.debug.print("[hy4-vision] row {d} id {d}: worst {e:.3} of the peak\n", .{ r, prompt[r], worst / peak });
        }
    }
    try expectNearSlice("image prompt prefill", try host(&o, pre), want[0 .. prompt.len * vocab], logits_bar);
    for (0..steps) |k| {
        const at = prompt.len + k;
        const row = try o.own(try m.forwardReadout(ids[at .. at + 1], .last));
        var nb: [48]u8 = undefined;
        try expectNearSlice(try std.fmt.bufPrint(&nb, "decode step {d}", .{k}), try host(&o, row), want[at * vocab ..][0..vocab], logits_bar);
    }
}

fn expectNearSlice(label: []const u8, g: []const f32, r: []const f32, bar: f64) !void {
    if (g.len != r.len) {
        std.debug.print("{s}: {d} values, the oracle has {d}\n", .{ label, g.len, r.len });
        return error.ShapeMismatch;
    }
    var peak: f64 = 1e-30;
    for (r) |x| peak = @max(peak, @abs(x));
    var worst: f64 = 0;
    for (g, r) |x, y| worst = @max(worst, @abs(@as(f64, x) - y));
    if (std.c.getenv("HY4_VISION_DIAG") != null) std.debug.print("[hy4-vision] {s}: worst {e:.3} of the peak {e:.3}\n", .{ label, worst / peak, peak });
    if (!(worst <= bar * peak)) {
        std.debug.print("{s}: worst |Δ| {e:.3} of the peak {e:.3}, bar {e:.1}\n", .{ label, worst / peak, peak, bar });
        return error.OutsideBar;
    }
}

test "hy4 vision: the image prompt prefilled in two chunks cut inside the image, the second carrying the rows the first consumed, gives the oracle's logits on every row (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const products = fwd.products;
    fwd.products = .f32;
    defer fwd.products = products;
    var f = try Fixture.open(es);
    defer f.close();
    const v = f.config.vision orelse return error.Hy4VisionMissing;
    var tower = try hy4_vision.Tower.init(a, v, &f.pack);
    defer tower.deinit();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const grid = try host(&o, try f.ref("vis.grid"));
    const image = try tower.forward(&o, try f.ref("vis.pixels"), @intFromFloat(grid[1]), @intFromFloat(grid[2]), null);
    const ids_f = try host(&o, try f.ref("vis.ids"));
    var ids: [64]u32 = undefined;
    for (ids_f, ids[0..ids_f.len]) |x, *d| d.* = @intFromFloat(x);
    const prompt = ids[0 .. ids_f.len - 4];
    // The cut falls after the image's third row.
    const first = std.mem.indexOfScalar(u32, prompt, f.config.image_id) orelse return error.Hy4ImageMissing;
    const cut = first + 3;
    const m = try hy4_model.Model.init(a, f.config, &f.pack, es);
    defer m.deinit();
    m.req().kv = .dense;
    const want = try host(&o, try f.ref("vis.logits"));
    const vocab = f.config.vocab;
    const one = try o.own(try m.forwardSpliced(prompt[0..cut], .{ .rows = image }, .all));
    try expectNearSlice("first chunk", try host(&o, one), want[0 .. cut * vocab], logits_bar);
    const two = try o.own(try m.forwardSpliced(prompt[cut..], .{ .rows = image, .offset = 3 }, .all));
    try expectNearSlice("second chunk", try host(&o, two), want[cut * vocab .. prompt.len * vocab], logits_bar);
}

test "hy4 vision: an encode on Sushi Cooler holds no more than encodeHeldBytes bills, and within 5% of it, at grids that take several attention chunks with a short last one, as served, its matrices BF16 or affine 8 (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try AffinePack.write(&tmp);
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const products = fwd.products;
    fwd.products = .bf16;
    defer fwd.products = products;
    var f = try Fixture.open(es);
    defer f.close();
    var p = try AffinePack.open(dir, es);
    defer p.close();
    for ([_]struct { v: hy4_config.Vision, w: *model_mod.Weights }{
        .{ .v = f.config.vision orelse return error.Hy4VisionMissing, .w = &f.pack },
        .{ .v = p.config.vision.?, .w = &p.weights },
    }) |arm| {
        var tower = try hy4_vision.Tower.init(a, arm.v, arm.w);
        defer tower.deinit();
        try encodeHeld(es, &tower, arm.v);
    }
}

fn encodeHeld(es: mlx.mlx_stream, tower: *hy4_vision.Tower, v: hy4_config.Vision) !void {
    const a = testing.allocator;
    // 1,536 patches by 256 query rows, six even chunks; 320 by 96, the last chunk 32 rows.
    for ([_][3]u32{ .{ 32, 48, 256 }, .{ 16, 20, 96 } }) |case| {
        const gh = case[0];
        const gw = case[1];
        tower.attn_rows = case[2];
        const n: usize = gh * gw;
        const feat: usize = 3 * v.patch * v.patch;
        const host_pv = try a.alloc(f32, n * feat);
        defer a.free(host_pv);
        var rng = std.Random.DefaultPrng.init(7);
        for (host_pv) |*x| x.* = rng.random().floatNorm(f32);
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const pv = try o.own(mlx.mlx_array_new_data(host_pv.ptr, &[_]c_int{ @intCast(n), @intCast(feat) }, 2, .float32));
        try mlx.check(mlx.mlx_array_eval(pv));
        try mlx.check(mlx.mlx_synchronize(es));
        // The pool starts with no free buffer: the encode makes every one it holds.
        mlx.engine.trim();
        const held0 = mlx.engine.poolTotal();
        mlx.engine.resetPoolPeak(.process);
        {
            var eo: Ops = .{ .s = es };
            defer eo.deinit();
            const rows = try tower.forward(&eo, pv, gh, gw, null);
            try mlx.check(mlx.mlx_array_eval(rows));
            try mlx.check(mlx.mlx_synchronize(es));
        }
        const held = mlx.engine.poolHeldPeak(.process) -| held0;
        const bill = try hy4_vision.encodeHeldBytes(v, tower.attn_rows, gh, gw);
        // The bill is the pool reusing every freed buffer once its pass is handed over; a run's timing can let it
        // reuse a few more.
        if (held > bill or bill - held > bill / 20) {
            std.debug.print("an encode of {d} x {d} patches by {d} query rows ({s}) held {d} B of the pool, billed {d} B\n", .{ gh, gw, case[2], if (v.quant != null) "affine 8" else "BF16", held, bill });
            return error.EncodeBillOff;
        }
    }
}

test "hy4 media: a 70 x 45 image decodes to the patches and grid of transformers' PIL processor at the release's settings (HY4_BF16_TINY)" {
    const a = testing.allocator;
    const hy4_media = @import("hy4_media.zig");
    const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var w = load(dir, "oracle_processor.safetensors", cpu) catch |e| {
        if (e == error.FileNotFound) return error.SkipZigTest;
        return e;
    };
    defer w.deinit();
    var config = try model_mod.parseConfig(std.testing.io, a, dir);
    defer config.deinit(a);
    const rgb_arr = w.get("proc.rgb") orelse return error.MissingTensor;
    try mlx.check(mlx.mlx_array_eval(rgb_arr));
    const shape = mlx.getShape(rgb_arr);
    const rgb = mlx.mlx_array_data_uint8(rgb_arr).?[0..mlx.mlx_array_size(rgb_arr)];
    const vp = hy4_media.preproc(&config);
    const img = hy4_media.decodeImage(a, rgb, @intCast(shape[0]), @intCast(shape[1]), vp) orelse return error.Undecodable;
    defer a.free(img.pixels);
    var o: Ops = .{ .s = cpu };
    defer o.deinit();
    const grid = try host(&o, w.get("proc.grid").?);
    try testing.expectEqual(@as(u32, @intFromFloat(grid[1])), img.grid_h);
    try testing.expectEqual(@as(u32, @intFromFloat(grid[2])), img.grid_w);
    try testing.expectEqual(@as(usize, 8), hy4_media.imageRows(vp.merge, img.grid_h, img.grid_w));
    const want = try host(&o, w.get("proc.pixels").?);
    const got = @as([*]const f32, @ptrCast(@alignCast(img.pixels.ptr)))[0 .. img.pixels.len / 4];
    try testing.expectEqual(want.len, got.len);
    // One ulp at 1: the reference rescales by 1/255 then normalizes, the engine takes x / 127.5 - 1.
    for (got, want, 0..) |g, r, k| if (@abs(g - r) > 1.2e-7) {
        std.debug.print("patch value {d}: {d}, the processor's {d}\n", .{ k, g, r });
        return error.OutsideBar;
    };
}

test "hy4 vision: after warmKernels an image prompt's prefill makes no pipeline a text prefill of its width did not (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var f = try Fixture.open(es);
    defer f.close();
    const m = try hy4_model.Model.init(a, f.config, &f.pack, es);
    defer m.deinit();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const ids_f = try host(&o, try f.ref("vis.ids"));
    var ids: [64]u32 = undefined;
    for (ids_f, ids[0..ids_f.len]) |x, *d| d.* = @intFromFloat(x);
    const prompt = ids[0 .. ids_f.len - 4];
    var text: [64]u32 = undefined;
    for (prompt, text[0..prompt.len]) |t, *d| d.* = if (t == f.config.image_id) 7 else t;
    try m.warmKernels();
    _ = try o.own(try m.forwardSpliced(text[0..prompt.len], null, .last));
    m.reset();
    var n_img: c_int = 0;
    for (prompt) |t| n_img += @intFromBool(t == f.config.image_id);
    const rows = try o.own(mlx.mlx_array_new());
    var r = rows;
    try mlx.check(mlx.mlx_zeros(&r, &[_]c_int{ 1, n_img, @intCast(f.config.dim) }, 3, .float32, es));
    try mlx.check(mlx.mlx_array_eval(r));
    const before = mlx.engine.pipelinesMade().count;
    _ = try o.own(try m.forwardSpliced(prompt, .{ .rows = r }, .last));
    const made = mlx.engine.pipelinesMade().count - before;
    if (made != 0) {
        std.debug.print("an image prefill after the warm made {d} pipelines\n", .{made});
        return error.CompiledInRequest;
    }
}

// ── The tower stored as MLX affine 8 (`sushi convert vision-affine8`) ──

/// The tiny after `sushi convert vision-affine8` into `tmp`: its config and every shard's tensors.
const AffinePack = struct {
    weights: model_mod.Weights,
    config: hy4_config.Config,
    dir: []u8,

    /// Runs the pass, on MLX's CPU stream: before the test enters the engine.
    fn write(tmp: *testing.TmpDir) ![]u8 {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const src = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir = try std.fmt.allocPrint(a, "{s}/va8", .{buf[0..try tmp.dir.realPath(io, &buf)]});
        errdefer a.free(dir);
        _ = try @import("sushi_encoder").vision_affine.run(a, io, .{ .model_dir = src, .out_dir = dir, .version = "test" });
        return dir;
    }

    fn open(dir: []u8, s: mlx.mlx_stream) !AffinePack {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        errdefer a.free(dir);
        var w = model_mod.Weights.init(a);
        errdefer w.deinit();
        for ([_][]const u8{ "model.safetensors", "model-vision.safetensors" }) |name| {
            const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0);
            defer a.free(path);
            try model_mod.loadSafetensorsFile(a, &w, path.ptr, s, true);
        }
        var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
        defer d.close(io);
        const text = try d.readFileAlloc(io, "config.json", a, .limited(1 << 20));
        defer a.free(text);
        return .{ .weights = w, .config = try hy4_config.parseText(text), .dir = dir };
    }

    fn close(p: *AffinePack) void {
        p.weights.deinit();
        testing.allocator.free(p.dir);
    }
};

/// Replaces `name`'s array in `w` (the map keeps its key).
fn replace(w: *model_mod.Weights, name: []const u8, value: Arr) void {
    const slot = w.map.getPtr(name).?;
    _ = mlx.mlx_array_free(slot.*);
    slot.* = value;
}

test "hy4 vision: a tower stored affine 8 multiplies its codes: every stage equals the tower over the codes' f32 values within the tower bar (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try AffinePack.write(&tmp);
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const products = fwd.products;
    fwd.products = .f32;
    defer fwd.products = products;
    var f = try Fixture.open(es);
    defer f.close();
    var p = try AffinePack.open(dir, es);
    defer p.close();
    const vq = p.config.vision.?;
    const q = vq.quant orelse return error.TestVisionNotAffine;
    // The reference: the BF16 tiny with each matrix the pack stores affine replaced by its codes' f32 values.
    var deq_count: usize = 0;
    var it = p.weights.map.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        const m = @import("hy4_census.zig").visionAffine(vq, name) orelse continue;
        const base = name[0 .. name.len - ".weight".len];
        var nb: [2][160]u8 = undefined;
        // `q·s + b` in f32 on the host: MLX's dequantize computes in the scales' BF16.
        var ho: Ops = .{ .s = es };
        defer ho.deinit();
        const codes_arr = try ho.contiguous(e.value_ptr.*);
        try mlx.check(mlx.mlx_array_eval(codes_arr));
        const codes: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint32(codes_arr).?);
        const sc = try host(&ho, p.weights.get(try std.fmt.bufPrint(&nb[0], "{s}.scales", .{base})).?);
        const bi = try host(&ho, p.weights.get(try std.fmt.bufPrint(&nb[1], "{s}.biases", .{base})).?);
        const rows: usize = @intCast(m[0]);
        const in: usize = @intCast(m[1]);
        const deq = try a.alloc(f32, rows * in);
        defer a.free(deq);
        for (0..rows) |r| for (0..in) |c| {
            const g = r * (in / q.group) + c / q.group;
            deq[r * in + c] = @as(f32, @floatFromInt(codes[r * in + c])) * sc[g] + bi[g];
        };
        // The source keeps a convolution's weight 4-D; the reference takes it as the [out, in] matrix the codes hold.
        replace(&f.pack, name, mlx.mlx_array_new_data(deq.ptr, &[_]c_int{ @intCast(rows), @intCast(in) }, 2, .float32));
        deq_count += 1;
    }
    try testing.expectEqual(@as(usize, 2 * 6 + 2), deq_count);
    var ref_tower = try hy4_vision.Tower.init(a, f.config.vision.?, &f.pack);
    defer ref_tower.deinit();
    var tower = try hy4_vision.Tower.init(a, vq, &p.weights);
    defer tower.deinit();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const grid = try host(&o, try f.ref("vis.grid"));
    const gh: u32 = @intFromFloat(grid[1]);
    const gw: u32 = @intFromFloat(grid[2]);
    var blocks: [2][8]Arr = undefined;
    var stages = [2]hy4_vision.Tower.Stages{ .{ .blocks = blocks[0][0..vq.layers] }, .{ .blocks = blocks[1][0..vq.layers] } };
    const got = try tower.forward(&o, try f.ref("vis.pixels"), gh, gw, &stages[0]);
    const want = try ref_tower.forward(&o, try f.ref("vis.pixels"), gh, gw, &stages[1]);
    try expectNear("patch embedding and positions", &o, stages[0].embed.?, stages[1].embed.?, tower_bar);
    for (0..vq.layers) |i| {
        var lb: [32]u8 = undefined;
        try expectNear(try std.fmt.bufPrint(&lb, "block {d}", .{i}), &o, blocks[0][i], blocks[1][i], tower_bar);
    }
    try expectNear("merged rows", &o, got, want, tower_bar);
}

test "hy4 vision: a tower matrix in a form the pack does not state is refused by name: codes with no vision_quant, BF16 under it, grids beside a BF16 weight, F16, an affine patch embedding, grids of the wrong width, a stray grid (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try AffinePack.write(&tmp);
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var f = try Fixture.open(es);
    defer f.close();
    var p = try AffinePack.open(dir, es);
    defer p.close();
    const affine = p.config.vision.?;
    var plain = affine;
    plain.quant = null;
    const q_name = "vit.layers.1.self_attn.q_proj.weight";
    // The affine pack and the BF16 tiny each load as they are.
    {
        var t = try hy4_vision.Tower.init(a, affine, &p.weights);
        t.deinit();
        var u = try hy4_vision.Tower.init(a, plain, &f.pack);
        u.deinit();
    }
    // Codes the config does not state.
    try testing.expectError(error.Hy4VisionTensorFormat, hy4_vision.Tower.init(a, plain, &p.weights));
    // A matrix the stamp says is affine, stored BF16 (its grids gone too).
    try testing.expectError(error.Hy4VisionTensorFormat, hy4_vision.Tower.init(a, affine, &f.pack));
    const Edit = struct { name: []const u8, value: Arr };
    const bf16_q = f.pack.get(q_name).?;
    const f16_q = blk: {
        var c = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&c, bf16_q, .float16, es));
        break :blk c;
    };
    defer _ = mlx.mlx_array_free(f16_q);
    const scales = p.weights.get("vit.layers.1.self_attn.q_proj.scales").?;
    const narrow = blk: {
        var c = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&c, scales, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 32, 0 }, 2, &[_]c_int{ 1, 1 }, 2, es));
        break :blk c;
    };
    defer _ = mlx.mlx_array_free(narrow);
    const cases = [_]struct { v: hy4_config.Vision, on_pack: bool, edits: []const Edit }{
        // Grids beside a BF16 weight.
        .{ .v = plain, .on_pack = false, .edits = &.{.{ .name = "vit.layers.1.self_attn.q_proj.scales", .value = scales }} },
        // An F16 matrix.
        .{ .v = plain, .on_pack = false, .edits = &.{.{ .name = q_name, .value = f16_q }} },
        // The patch embedding as codes.
        .{ .v = affine, .on_pack = true, .edits = &.{.{ .name = "vit.embeddings.patch_embedding.weight", .value = p.weights.get(q_name).? }} },
        // Grids of no width the codes take.
        .{ .v = affine, .on_pack = true, .edits = &.{.{ .name = "vit.layers.1.self_attn.q_proj.scales", .value = narrow }} },
        // A grid beside no matrix the tower holds.
        .{ .v = affine, .on_pack = true, .edits = &.{.{ .name = "vit.perceive.norm.scales", .value = scales }} },
    };
    for (cases, 0..) |c, i| {
        const w = if (c.on_pack) &p.weights else &f.pack;
        var saved: [1]?Arr = .{null};
        for (c.edits, &saved) |e, *s| {
            var copy = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&copy, e.value));
            if (w.map.getPtr(e.name)) |slot| {
                s.* = slot.*;
                slot.* = copy;
            } else try w.map.put(try a.dupe(u8, e.name), copy);
        }
        defer for (c.edits, saved) |e, s| {
            if (s) |old| replace(w, e.name, old) else _ = mlx.mlx_array_free(w.take(e.name).?);
        };
        if (hy4_vision.Tower.init(a, c.v, w)) |t_ok| {
            var t = t_ok;
            t.deinit();
            std.debug.print("case {d} loaded\n", .{i});
            return error.TestFormatAccepted;
        } else |err| try testing.expectEqual(error.Hy4VisionTensorFormat, err);
    }
}


/// `‖a - b‖ / ‖b‖` over all values.
fn relRms(g: []const f32, r: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 1e-300;
    for (g, r) |x, y| {
        num += (@as(f64, x) - y) * (@as(f64, x) - y);
        den += @as(f64, y) * y;
    }
    return @sqrt(num / den);
}

/// Each stored-affine matrix's relative weight error `‖W - Ŵ‖_F / ‖W‖_F`, its codes `Ŵ = q·s + b` against the BF16
/// source `W`, in the tower's order: per block q, k, v, o, MLP up, MLP down, then the merger's two.
fn weightErrors(a: std.mem.Allocator, o: *Ops, v: hy4_config.Vision, src: *const model_mod.Weights, pack: *const model_mod.Weights) ![]f64 {
    var names = std.ArrayList([]const u8).empty;
    defer {
        for (names.items) |n| a.free(n);
        names.deinit(a);
    }
    for (0..v.layers) |l| for ([_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h" }) |leaf| {
        try names.append(a, try std.fmt.allocPrint(a, "vit.layers.{d}.{s}", .{ l, leaf }));
    };
    for ([_][]const u8{ "vit.perceive.proj.0", "vit.perceive.proj.2" }) |n| try names.append(a, try a.dupe(u8, n));
    const q = v.quant.?;
    const out = try a.alloc(f64, names.items.len);
    errdefer a.free(out);
    for (names.items, out) |base, *eps| {
        var nb: [4][168]u8 = undefined;
        const wname = try std.fmt.bufPrint(&nb[0], "{s}.weight", .{base});
        const m = @import("hy4_census.zig").visionAffine(v, wname) orelse return error.TestNotAffine;
        const w = try host(o, src.get(wname).?);
        const codes_arr = try o.contiguous(pack.get(wname).?);
        try mlx.check(mlx.mlx_array_eval(codes_arr));
        const codes: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint32(codes_arr).?);
        const sc = try host(o, pack.get(try std.fmt.bufPrint(&nb[1], "{s}.scales", .{base})).?);
        const bi = try host(o, pack.get(try std.fmt.bufPrint(&nb[2], "{s}.biases", .{base})).?);
        const in: usize = @intCast(m[1]);
        var num: f64 = 0;
        var den: f64 = 0;
        for (w, 0..) |x, i| {
            const g = (i / in) * (in / q.group) + (i % in) / q.group;
            const d = @as(f64, @floatFromInt(codes[i])) * sc[g] + bi[g] - x;
            num += d * d;
            den += @as(f64, x) * x;
        }
        eps.* = @sqrt(num / den);
    }
    return out;
}

/// The quantization error's allowance over the unit-gain model: a block's attention softmax and tanh GELU may amplify
/// a perturbation of their input, by no more than this (set before measuring, not fitted).
const gain_allowance: f64 = 2;

test "hy4 vision: the tiny's pack after vision-affine8, loaded as served, keeps the oracle's image rows within the affine-8 error of its matrices, and its image prompt's logits within what noise of that size moves them (HY4_BF16_TINY)" {
    const hy4_media = @import("hy4_media.zig");
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try AffinePack.write(&tmp);
    defer a.free(dir);
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const products = fwd.products;
    fwd.products = .f32;
    defer fwd.products = products;
    var f = try Fixture.open(es);
    defer f.close();
    // The served load: the config, the weights the loader keeps with the tower asked for, the media slot's tower.
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, true);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    const v = config.hy4flash.?.vision.?;
    try testing.expect(v.quant != null);
    // The bill's resident bytes are the bytes the load holds, the tower's codes and grids as stored.
    {
        var held: u64 = 0;
        var it = weights.map.iterator();
        while (it.next()) |e| held += mlx.mlx_array_size(e.value_ptr.*) * mlx.mlx_array_itemsize(e.value_ptr.*);
        try testing.expectEqual((try @import("hy4_load.zig").residentLoadBytes(io, a, dir, &config, true, false)).?, held);
    }
    var tower = try hy4_media.initTower(a, config, &weights);
    defer tower.deinit();
    var o: Ops = .{ .s = es };
    defer o.deinit();

    // The model: errors of independent roundings add in quadrature, through a gain of one (`gain_allowance` over it).
    const eps = try weightErrors(a, &o, v, &f.pack, &weights);
    defer a.free(eps);
    const per_block = 6;
    var sum: f64 = 0;
    for (eps) |e| sum += e * e;
    const bar_rows = gain_allowance * @sqrt(sum);

    const grid = try host(&o, try f.ref("vis.grid"));
    const gh: u32 = @intFromFloat(grid[1]);
    const gw: u32 = @intFromFloat(grid[2]);
    var blocks: [8]Arr = undefined;
    var stages: hy4_vision.Tower.Stages = .{ .blocks = blocks[0..v.layers] };
    const rows = try tower.tower.forward(&o, try f.ref("vis.pixels"), gh, gw, &stages);
    // Nothing quantized precedes the blocks: the patch embedding is BF16.
    try expectNear("patch embedding and positions", &o, stages.embed.?, try f.ref("vis.embed"), tower_bar);
    var upstream: f64 = 0;
    for (0..v.layers) |i| {
        for (eps[i * per_block ..][0..per_block]) |e| upstream += e * e;
        var nb: [32]u8 = undefined;
        const e = relRms(try host(&o, blocks[i]), try host(&o, try f.ref(try std.fmt.bufPrint(&nb, "vis.block{d}", .{i}))));
        if (std.c.getenv("HY4_VISION_DIAG") != null) std.debug.print("[hy4-vision] affine block {d}: {e:.3}, the model's {e:.3}\n", .{ i, e, @sqrt(upstream) });
        if (!(e <= gain_allowance * @sqrt(upstream))) {
            std.debug.print("affine block {d}: relative error {e:.3}, bar {e:.3}\n", .{ i, e, gain_allowance * @sqrt(upstream) });
            return error.OutsideBar;
        }
    }
    const e_rows = relRms(try host(&o, rows), try host(&o, try f.ref("vis.merged")));

    // The image prompt end to end: the rows spliced into the model over the served weights, prefill and 4 decode steps.
    const ids_f = try host(&o, try f.ref("vis.ids"));
    var ids: [64]u32 = undefined;
    for (ids_f, ids[0..ids_f.len]) |x, *d| d.* = @intFromFloat(x);
    // Engram's tables stay in the shards, read by row, as served.
    const m = try hy4_model.Model.initWith(a, config.hy4flash.?, &weights, es, .{ .dir = dir });
    defer m.deinit();
    m.req().kv = .dense;
    const want = try host(&o, try f.ref("vis.logits"));
    const got = try promptLogits(a, &o, m, ids[0..ids_f.len], rows, want.len);
    defer a.free(got);
    const e_logits = relRms(got, want);
    // The tiny's random trunk is no unit gain: its logits move by many times what an image row moves. Its own gain sets
    // the logits' bar: the oracle's rows each moved by Gaussian noise of the rows' bar move them by `e_noise`.
    const ref_rows = try host(&o, try f.ref("vis.merged"));
    const noisy = try a.alloc(f32, ref_rows.len);
    defer a.free(noisy);
    var rng = std.Random.DefaultPrng.init(0x7e57);
    const width: usize = v.out_hidden;
    for (0..ref_rows.len / width) |r| {
        const ref_row = ref_rows[r * width ..][0..width];
        const row = noisy[r * width ..][0..width];
        var ref_sq: f64 = 0;
        var noise_sq: f64 = 0;
        for (row, ref_row) |*x, y| {
            x.* = rng.random().floatNorm(f32);
            noise_sq += @as(f64, x.*) * x.*;
            ref_sq += @as(f64, y) * y;
        }
        const k: f32 = @floatCast(bar_rows * @sqrt(ref_sq / noise_sq));
        for (row, ref_row) |*x, y| x.* = y + k * x.*;
    }
    const noisy_arr = try o.own(mlx.mlx_array_new_data(noisy.ptr, &[_]c_int{ @intCast(ref_rows.len / width), @intCast(width) }, 2, .float32));
    m.reset();
    const moved = try promptLogits(a, &o, m, ids[0..ids_f.len], noisy_arr, want.len);
    defer a.free(moved);
    const e_noise = relRms(moved, want);
    if (std.c.getenv("HY4_VISION_DIAG") != null) {
        std.debug.print("[hy4-vision] affine eps: ", .{});
        for (eps) |e| std.debug.print("{e:.2} ", .{e});
        std.debug.print("\n[hy4-vision] affine merged rows {e:.3} (bar {e:.3}), logits {e:.3} (bar {e:.3})\n", .{ e_rows, bar_rows, e_logits, e_noise });
    }
    if (!(e_rows <= bar_rows and e_logits <= e_noise)) {
        std.debug.print("affine: merged rows {e:.3} (bar {e:.3}), logits {e:.3} (bar {e:.3})\n", .{ e_rows, bar_rows, e_logits, e_noise });
        return error.OutsideBar;
    }
}

/// The logits of the image prompt `ids` (its last 4 ids decoded one at a time), the image's `rows` spliced in.
fn promptLogits(a: std.mem.Allocator, o: *Ops, m: anytype, ids: []const u32, rows: Arr, len: usize) ![]f32 {
    const steps = 4;
    const prompt = ids[0 .. ids.len - steps];
    const vocab = len / ids.len;
    const out = try a.alloc(f32, len);
    errdefer a.free(out);
    const spliced = try o.reshape(rows, &.{ 1, mlx.getShape(rows)[0], mlx.getShape(rows)[1] });
    @memcpy(out[0 .. prompt.len * vocab], try host(o, try o.own(try m.forwardSpliced(prompt, .{ .rows = spliced }, .all))));
    for (0..steps) |k| {
        const at = prompt.len + k;
        @memcpy(out[at * vocab ..][0..vocab], try host(o, try o.own(try m.forwardReadout(ids[at .. at + 1], .last))));
    }
    return out;
}

/// One block of the release's tower (1,152 wide, 16 heads, a 4,304-wide MLP, 2 x 2 merge, 3,072 out), every value zero: the
/// kernels an encode picks follow its shapes and dtypes. Matrices are affine codes where `v.quant` stores them so.
fn releaseTower(w: *model_mod.Weights, v: hy4_config.Vision) !void {
    const a = testing.allocator;
    const put = struct {
        fn f(map: *model_mod.Weights, name: []const u8, shape: []const c_int, dtype: mlx.mlx_dtype) !void {
            var count: usize = 1;
            for (shape) |d| count *= @intCast(d);
            const bytes = try testing.allocator.alloc(u8, count * if (dtype == .uint32) @as(usize, 4) else 2);
            defer testing.allocator.free(bytes);
            @memset(bytes, 0);
            const arr = try mlx.newData(bytes.ptr, shape, dtype);
            errdefer _ = mlx.mlx_array_free(arr);
            try map.map.put(try testing.allocator.dupe(u8, name), arr);
        }
    }.f;
    var buf: [160]u8 = undefined;
    const h: c_int = @intCast(v.hidden);
    const cells: c_int = h * @as(c_int, @intCast(v.merge * v.merge));
    const matrices = [_]struct { []const u8, c_int, c_int }{
        .{ "vit.layers.0.self_attn.q_proj", h, h },
        .{ "vit.layers.0.self_attn.k_proj", h, h },
        .{ "vit.layers.0.self_attn.v_proj", h, h },
        .{ "vit.layers.0.self_attn.o_proj", h, h },
        .{ "vit.layers.0.mlp.dense_h_to_4h", @intCast(v.inter), h },
        .{ "vit.layers.0.mlp.dense_4h_to_h", h, @intCast(v.inter) },
        .{ "vit.perceive.proj.0", cells, cells },
        .{ "vit.perceive.proj.2", @intCast(v.out_hidden), cells },
    };
    for (matrices) |m| {
        const wname = try std.fmt.bufPrint(&buf, "{s}.weight", .{m[0]});
        if (v.quant) |q| if (@import("hy4_census.zig").visionAffine(v, wname) != null) {
            try put(w, wname, &.{ m[1], @divExact(m[2] * @as(c_int, @intCast(q.bits)), 32) }, .uint32);
            for ([_][]const u8{ "scales", "biases" }) |leaf| {
                const gname = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ m[0], leaf });
                try put(w, gname, &.{ m[1], @divExact(m[2], @as(c_int, @intCast(q.group))) }, .bfloat16);
            }
            continue;
        };
        try put(w, wname, &.{ m[1], m[2] }, .bfloat16);
    }
    const out: c_int = @intCast(v.out_hidden);
    const vectors = [_]struct { []const u8, c_int }{
        .{ "vit.layers.0.input_layernorm.weight", h },      .{ "vit.layers.0.input_layernorm.bias", h },
        .{ "vit.layers.0.post_attention_layernorm.weight", h }, .{ "vit.layers.0.post_attention_layernorm.bias", h },
        .{ "vit.layers.0.self_attn.q_proj.bias", h },       .{ "vit.layers.0.self_attn.k_proj.bias", h },
        .{ "vit.layers.0.self_attn.v_proj.bias", h },       .{ "vit.layers.0.self_attn.o_proj.bias", h },
        .{ "vit.layers.0.mlp.dense_h_to_4h.bias", @intCast(v.inter) }, .{ "vit.layers.0.mlp.dense_4h_to_h.bias", h },
        .{ "vit.embeddings.patch_embedding.bias", h },      .{ "vit.perceive.norm.weight", h },
        .{ "vit.perceive.proj.0.bias", cells },             .{ "vit.perceive.proj.2.bias", out },
        .{ "vit.perceive.image_newline", out },
    };
    for (vectors) |vec| try put(w, vec[0], &.{vec[1]}, .bfloat16);
    try put(w, "vit.embeddings.patch_embedding.weight", &.{ h, @intCast(3 * v.patch * v.patch) }, .bfloat16);
    try put(w, "vit.embeddings.position_embedding.weight", &.{ 128 * 128 + 1, h }, .bfloat16);
    _ = a;
}

/// Encodes at grids of every kind the release's processor lands on (whole merge cells, 16 to 9,216 patches): one
/// attention chunk, whole chunks of 512 query rows and a last short one, key counts aligned and not, past 4,096 and
/// 8,192 keys, a row's width and a column's.
const release_grids = [_][2]u32{
    .{ 2, 2 }, .{ 6, 10 }, .{ 16, 16 }, .{ 32, 32 }, .{ 2, 96 }, .{ 96, 2 }, .{ 64, 66 }, .{ 96, 96 },
};

fn towerAfterWarm(quantized: bool) !void {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var v: hy4_config.Vision = .{ .layers = 1, .hidden = 1152, .heads = 16, .inter = 4304, .patch = 16, .merge = 2, .out_hidden = 3072 };
    if (quantized) v.quant = .{ .bits = 8, .group = hy4_config.visionGroup(v.hidden).? };
    var w = model_mod.Weights.init(a);
    defer w.deinit();
    try releaseTower(&w, v);
    var tower = try hy4_vision.Tower.init(a, v, &w);
    defer tower.deinit();
    try tower.warmEngine();
    mlx.engine.seal(true);
    defer mlx.engine.seal(false);
    const late = mlx.engine.latePipelines();
    mlx.engine.late_names.clear();
    var first: ?[2]u32 = null;
    for (release_grids) |g| {
        const n: usize = @as(usize, g[0]) * g[1];
        const feat: usize = 3 * v.patch * v.patch;
        const zeros = try a.alloc(f32, n * feat);
        defer a.free(zeros);
        @memset(zeros, 0);
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const patches = try o.own(try mlx.newData(zeros.ptr, &[_]c_int{ @intCast(n), @intCast(feat) }, .float32));
        const rows = try tower.forward(&o, patches, g[0], g[1], null);
        try mlx.check(mlx.mlx_array_eval(rows));
        if (first == null and mlx.engine.latePipelines() != late) first = g;
    }
    errdefer std.debug.print("hy4 vision: from the grid of {d} x {d} patches on, an encode compiled:\n{s}", .{ first.?[0], first.?[1], mlx.engine.late_names.text() });
    try testing.expectEqual(late, mlx.engine.latePipelines());
}

test "hy4 vision: a Cooler load's tower warms before the seal, so an image of any grid compiles nothing inside a request, at the release's widths, its matrices BF16" {
    try towerAfterWarm(false);
}

test "hy4 vision: a Cooler load's tower warms before the seal, so an image of any grid compiles nothing inside a request, at the release's widths, its matrices affine 8" {
    try towerAfterWarm(true);
}
