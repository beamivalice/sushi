//! What `sushi kld` asks of Hy4-Flash (`arch.zig` `kld`, reached through `src/arch/offline.zig`). Capture and compare
//! run the generic path, streamed: the model takes the request's latent width from its cache config at step 0
//! (`arch.zig` `forward`), so a capture or compare at `--kv-quant 8` or `16` runs the width it names.
const std = @import("std");
const model = @import("../model.zig");
const kld = @import("../kld.zig");
const kv_quant = @import("../kv_quant.zig");
const layer_major = @import("hy4_layer_major.zig");
const model_load = @import("../model_load.zig");
const bills = @import("../bills.zig");

/// The launch options Hy4 cannot honor, refused by name before the load: a flag that would silently do nothing, or
/// a latent width the model does not serve.
pub fn refuse(config: *const model.ModelConfig, opts: kld.Options) !void {
    _ = config;
    if (!servesKv(opts.kv_quant_config)) return error.Hy4KvQuantUnsupported;
    if (!opts.qwen_gdn_f32) return error.Hy4HasNoQwenGdnState;
    if (opts.hidden_out.len > 0) return error.Hy4HiddenCaptureUnsupported;
    if (opts.imatrix_out.len > 0) return error.Hy4ImatrixCaptureUnsupported;
    // The routing is the host's exact top-k, so the swap has nothing to act on; a compare takes the cache-favoring pick.
    if (opts.pick_tolerance > 0) return error.Hy4ExpertPickToleranceUnsupported;
    if (opts.drop_expert != 0) return error.Hy4DropExpertUnsupported;
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// What a run needs before its load plans.
/// - The teacher carries no lossy step of its own: a capture's products are f32 (`hy4_forward.products`), and on a NAX
///   GPU MLX and the engine run f32 products on TF32 tiles unless `MLX_ENABLE_TF32=0`, which MLX reads at its first
///   product: it is set here and MLX's reading taken again (the CLI reads the GPU before any subcommand), and a process
///   whose MLX already multiplied with it on is refused by name. A compare runs the student as it is served: BF16 products.
/// - A layer-major capture pins a layer's whole union, so its load keeps a union workspace beside the cache.
pub fn beforeLoad(config: *model.ModelConfig, opts: kld.Options) !void {
    if (std.c.getenv("MLX_ENABLE_TF32")) |v| {
        if (!std.mem.eql(u8, std.mem.span(v), "0")) return error.Hy4KldNeedsTf32Off;
    } else if (setenv("MLX_ENABLE_TF32", "0", 0) != 0) return error.Hy4KldEnvironmentFailed;
    if (try @import("../tensor/matmul.zig").rereadTf32()) return error.Hy4KldTf32AlreadyRead;
    @import("hy4_forward.zig").products = if (opts.command == .capture) .f32 else .bf16;
    if (opts.layer_major) layerMajorLoad(config);
}

/// The products a load serves with, which the capture's f32 products must not outlive.
pub fn afterUnload(_: *const model.ModelConfig) void {
    @import("hy4_forward.zig").products = .bf16;
}

fn layerMajorLoad(config: *model.ModelConfig) void {
    config.expert_union_within_cache = false;
}

/// Whether the layer-major calibration capture takes this run; the flags it cannot honor are refused by name.
pub fn accepts(config: *const model.ModelConfig, opts: kld.Options) !bool {
    if (!@import("../arch/config.zig").isOf(config, "hy4flash") or opts.command != .capture or !opts.layer_major) return false;
    if (opts.kv_quant_config.isQuant()) return error.Hy4LayerMajorNeedsBf16Latent;
    if (opts.hidden_out.len > 0) return error.Hy4HiddenCaptureUnsupported;
    if (opts.imatrix_windows != 0) return error.Hy4ImatrixWindowsUnsupported;
    // The encoder's Hessians come from the stored rows; the imatrix file's writer cannot take an engine load's arrays.
    if (opts.imatrix_out.len > 0) return error.Hy4ImatrixCaptureUnsupported;
    if (opts.tokens != 1) return error.Hy4LayerMajorNeedsOneRow;
    if (!opts.no_template or (opts.mtp orelse false) or (opts.wired_margin_bytes orelse 0) != 0) return error.Hy4CaptureNeedsLosslessStreaming;
    return true;
}

/// The layer-major capture (`hy4_layer_major`): the load keeps its expert cache at the two-slot floor, and the rest of
/// the budget (the flag, else the setting, else serve's RAM default) buys windows a batch.
pub fn tryRun(a: std.mem.Allocator, io: std.Io, opts: kld.Options, out: *kld.Out) !bool {
    if (opts.command != .capture or !opts.layer_major) return false;
    var cfg = try model.parseConfig(io, a, opts.model_dir);
    defer cfg.deinit(a);
    if (!try accepts(&cfg, opts)) return false;
    const c = cfg.hy4flash orelse return error.Hy4ConfigMissing;
    const mode = try layer_major.modeFromEnv();
    layerMajorLoad(&cfg);
    const floor = try captureFloor(io, a, &cfg, opts.model_dir);
    const chunk = kld.promptChunk(&cfg);
    const rows: u64 = try longestWindow(a, io, opts);
    const asked = model_load.resolveSsdBudget(opts.ssd_budget_bytes, cfg.ssd_budget_gb_override, cfg.supportsExpertStreaming());
    var settle: bills.SettleBudget = .{};
    const budget = (try model_load.withRamDefault(io, a, &cfg, opts.model_dir, asked, model_load.ramDefaultTierFor(&cfg), false, 0, &settle)).bytes;
    var load = opts;
    load.ssd_budget_bytes = floor;
    const loaded = try kld.loadModel(io, a, load);
    defer loaded.deinit();
    // A window's pieces are measured on the load, dry (D104): the batch is what the budget holds beside it.
    const window = try layer_major.windowBytes(loaded.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing, a, rows, chunk);
    const batch = if (opts.batch_windows != 0) opts.batch_windows else try layer_major.batchWindows(&c, budget, floor, rows, window);
    try layer_major.checkBatch(&c, budget, floor, rows, window, batch);
    try layer_major.runLoaded(a, io, loaded, opts, batch, chunk, out, mode);
    return true;
}

/// The longest window the prompts hold, which a batch is billed by.
fn longestWindow(a: std.mem.Allocator, io: std.Io, opts: kld.Options) !u64 {
    var prompts = try @import("../kld_fixture.zig").loadPrompts(a, io, opts.prompts, opts.limit);
    defer prompts.deinit();
    var most: u64 = 0;
    for (prompts.items) |p| most = @max(most, if (p.ids) |ids| ids.len else p.text.len);
    if (most == 0) return error.NoPromptsFound;
    return most;
}

/// The cheapest `--ssd-budget-gb` the streamed load accepts with a layer's whole union pinned: the trunk it keeps, the
/// expert pool, the union, one token's picks, the bounce buffers and two slots a layer.
pub fn captureFloor(io: std.Io, a: std.mem.Allocator, cfg: *const model.ModelConfig, model_dir: []const u8) !u64 {
    const c = cfg.hy4flash orelse return error.Hy4ConfigMissing;
    const split = try @import("hy4_load.zig").streamedSplit(io, a, model_dir, cfg);
    const stream = @import("../expert_stream.zig");
    const layout = try stream.quant.streamingLayoutOfDir(a, io, cfg.model_type, model_dir, @intCast(cfg.num_hidden_layers), cfg.expertGeometry().first_moe_layer);
    const per_expert = (try stream.expertBillFor(a, model_dir, cfg.expertGeometry(), layout)).expert_bytes;
    return layer_major.floorBudget(&c, split.trunk + cfg.expertPoolBytes(), per_expert);
}

/// The latent widths the model stores (`arch.zig` `kvWidth`): BF16, or the engine's affine kv8 in groups of 64.
fn servesKv(kv: kv_quant.KVQuantConfig) bool {
    return kv.scheme == .off or std.meta.eql(kv, kv_quant.KVQuantConfig.engine_default);
}

const testing = std.testing;
const mlx = @import("../mlx.zig");
const fixture_mod = @import("../kld_fixture.zig");
const util = @import("../kld_capture_util.zig");

test "kld: hy_v4_flash takes its two latent widths and refuses by name what it cannot honor" {
    const config = model.ModelConfig{ .model_type = "hy_v4_flash" };
    try refuse(&config, .{});
    try refuse(&config, .{ .kv_quant_config = kv_quant.KVQuantConfig.affine(8) });
    for ([_]u8{ 4, 6 }) |bits| try testing.expectError(error.Hy4KvQuantUnsupported, refuse(&config, .{ .kv_quant_config = kv_quant.KVQuantConfig.affine(bits) }));
    var wide = kv_quant.KVQuantConfig.affine(8);
    wide.group_size = 128;
    try testing.expectError(error.Hy4KvQuantUnsupported, refuse(&config, .{ .kv_quant_config = wide }));
    try testing.expectError(error.Hy4HasNoQwenGdnState, refuse(&config, .{ .qwen_gdn_f32 = false }));
    try testing.expectError(error.Hy4HiddenCaptureUnsupported, refuse(&config, .{ .hidden_out = "/x" }));
    try testing.expectError(error.Hy4ImatrixCaptureUnsupported, refuse(&config, .{ .imatrix_out = "/x.safetensors" }));
    try testing.expectError(error.Hy4ExpertPickToleranceUnsupported, refuse(&config, .{ .command = .compare, .pick_tolerance = 0.2 }));
    try refuse(&config, .{ .command = .compare, .cache_bias = 0.5 });
    try testing.expectError(error.Hy4DropExpertUnsupported, refuse(&config, .{ .command = .compare, .drop_expert = 8 }));
}

test "kld: a hy_v4_flash run refuses TF32 products by name" {
    const a = testing.allocator;
    const fwd = @import("hy4_forward.zig");
    const was = fwd.products;
    defer fwd.products = was;
    const saved = if (std.c.getenv("MLX_ENABLE_TF32")) |v| try a.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer if (saved) |v| {
        _ = setenv("MLX_ENABLE_TF32", v, 1);
        a.free(v);
    } else {
        _ = unsetenv("MLX_ENABLE_TF32");
    };
    _ = setenv("MLX_ENABLE_TF32", "1", 1);
    var config = model.ModelConfig{ .model_type = "hy_v4_flash" };
    try testing.expectError(error.Hy4KldNeedsTf32Off, beforeLoad(&config, .{}));
}

extern "c" fn unsetenv(name: [*:0]const u8) c_int;

// The CLI reads the GPU before any subcommand, with the variable unset: TF32 on. Before MLX runs a product the run turns
// it off and MLX's own reading is taken again; once MLX has read it on, the run is refused by name.
test "kld: a hy_v4_flash run turns TF32 off after an early read, unless MLX has already multiplied with it on" {
    const matmul = @import("../tensor/matmul.zig");
    if (!(try matmul.gpu()).nax) return error.SkipZigTest;
    const a = testing.allocator;
    const saved = if (std.c.getenv("MLX_ENABLE_TF32")) |v| try a.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer if (saved) |v| {
        _ = setenv("MLX_ENABLE_TF32", v, 1);
        a.free(v);
    } else {
        _ = unsetenv("MLX_ENABLE_TF32");
    };
    defer {
        matmul.cacheTf32ForTest(null);
        matmul.mlx_tf32_for_test = null;
    }
    var config = model.ModelConfig{ .model_type = "hy_v4_flash" };
    for ([_]bool{ false, true }) |mlx_read_on| {
        _ = unsetenv("MLX_ENABLE_TF32");
        matmul.cacheTf32ForTest(true);
        matmul.mlx_tf32_for_test = mlx_read_on;
        if (mlx_read_on) {
            try testing.expectError(error.Hy4KldTf32AlreadyRead, beforeLoad(&config, .{}));
        } else {
            try beforeLoad(&config, .{});
            try testing.expect(!(try matmul.gpu()).tf32);
        }
    }
}

test "kld: a hy_v4_flash layer-major capture takes one row a window of lossless streaming and refuses the rest by name" {
    const cfg = model.ModelConfig{ .model_type = "hy_v4_flash" };
    const ok = kld.Options{ .command = .capture, .layer_major = true, .no_template = true, .tokens = 1, .ssd_budget_bytes = 100 << 30 };
    try testing.expect(try accepts(&cfg, ok));
    // No --ssd-budget-gb is no refusal: the batch is sized from the setting, else serve's RAM default.
    var no_flag = ok;
    no_flag.ssd_budget_bytes = 0;
    try testing.expect(try accepts(&cfg, no_flag));
    try testing.expect(!try accepts(&cfg, .{ .command = .capture, .no_template = true, .tokens = 1, .ssd_budget_bytes = 100 << 30 }));
    try testing.expect(!try accepts(&model.ModelConfig{ .model_type = "deepseek_v41" }, ok));
    var bad = ok;
    bad.tokens = 4;
    try testing.expectError(error.Hy4LayerMajorNeedsOneRow, accepts(&cfg, bad));
    bad = ok;
    bad.kv_quant_config = kv_quant.KVQuantConfig.affine(8);
    try testing.expectError(error.Hy4LayerMajorNeedsBf16Latent, accepts(&cfg, bad));
    bad = ok;
    bad.no_template = false;
    try testing.expectError(error.Hy4CaptureNeedsLosslessStreaming, accepts(&cfg, bad));
    // A wired margin is refused where one is given and is not zero; an omitted flag gives none.
    var margin = ok;
    margin.wired_margin_bytes = 0;
    try testing.expect(try accepts(&cfg, margin));
    margin.wired_margin_bytes = 1 << 30;
    try testing.expectError(error.Hy4CaptureNeedsLosslessStreaming, accepts(&cfg, margin));
    bad = ok;
    bad.imatrix_windows = 8;
    try testing.expectError(error.Hy4ImatrixWindowsUnsupported, accepts(&cfg, bad));
    bad = ok;
    bad.hidden_out = "/x";
    try testing.expectError(error.Hy4HiddenCaptureUnsupported, accepts(&cfg, bad));
    bad = ok;
    bad.imatrix_out = "/x.safetensors";
    try testing.expectError(error.Hy4ImatrixCaptureUnsupported, accepts(&cfg, bad));
}

test "kld: the hy_v4_flash capture floor is the least budget the streamed load's plan accepts, a whole layer's union beside two slots a layer (HY4_BF16_TINY)" {
    const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var cfg = try model.parseConfig(io, a, dir);
    defer cfg.deinit(a);
    layerMajorLoad(&cfg);
    const floor = try captureFloor(io, a, &cfg, dir);
    const plan = try model_load.planExpertStreaming(io, a, &cfg, dir, floor, false);
    try testing.expectEqual(@as(u64, cfg.num_experts) * plan.resolved.ledger.?.cache_bytes / (2 * cfg.expertLayerCount()), plan.resolved.ledger.?.workspace_bytes);
    try testing.expectEqual(@as(u16, 2), plan.cache.slots_per_layer);
    try testing.expectError(error.SsdBudgetBelowResident, model_load.planExpertStreaming(io, a, &cfg, dir, floor - 1, false));
}

/// The tiny's oracle, read on the CPU before any load: its 48 ids and every position's logits [48, vocab].
const Oracle = struct {
    ids: [48]u32,
    logits: []f32,
    vocab: usize,

    fn read(a: std.mem.Allocator, dir: []const u8) !Oracle {
        const path = try std.fmt.allocPrintSentinel(a, "{s}/oracle.safetensors", .{dir}, 0);
        defer a.free(path);
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        var w = model.Weights.init(a);
        defer w.deinit();
        try model.loadSafetensorsFile(a, &w, path.ptr, cpu, true);
        var o: Oracle = .{ .ids = undefined, .logits = undefined, .vocab = 0 };
        const ids = try hostF32(a, w.get("ids") orelse return error.Hy4OracleIds, cpu);
        defer a.free(ids);
        for (&o.ids, ids[0..48]) |*d, v| d.* = @intFromFloat(v);
        const logits = w.get("logits") orelse return error.Hy4OracleLogits;
        o.vocab = @intCast(mlx.getShape(logits)[1]);
        o.logits = try hostF32(a, logits, cpu);
        return o;
    }
};

fn hostF32(a: std.mem.Allocator, x: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_astype(&c, x, .float32, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_float32(c) orelse return error.HostRead;
    return a.dupe(f32, p[0..mlx.mlx_array_size(c)]);
}

/// The worst |got - want| over a row, as a fraction of the row's peak.
fn rowError(got: []const f32, want: []const f32) f64 {
    var peak: f64 = 1e-30;
    var worst: f64 = 0;
    for (got, want) |x, y| {
        peak = @max(peak, @abs(y));
        worst = @max(worst, @abs(@as(f64, x) - y));
    }
    return worst / peak;
}

const Compared = struct { mean_kld: f64, top1: f64, positions: f64 };

fn compareJson(a: std.mem.Allocator, io: std.Io, path: []const u8) !Compared {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    const o = parsed.value.object;
    return .{ .mean_kld = try util.jsonNumber(o.get("mean_kld").?), .top1 = try util.jsonNumber(o.get("mean_top1").?), .positions = try util.jsonNumber(o.get("positions").?) };
}

test "kld: a hy_v4_flash teacher captured streamed holds the oracle's logits, a compare of the same load reads KLD 0, and a kv8 student a small non-zero KLD (HY4_BF16_TINY)" {
    const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
    try mlx.requireGpu();
    const fwd = @import("hy4_forward.zig");
    const was = fwd.products;
    defer fwd.products = was;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const oracle = try Oracle.read(arena, dir);

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = try arena.dupe(u8, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
    // The oracle's prompt with its own continuation forced, so each captured row is an oracle row; and a greedy one.
    var body: std.Io.Writer.Allocating = .init(arena);
    try body.writer.writeAll("{\"id\":\"oracle\",\"prompt_ids\":[");
    for (oracle.ids[0..40], 0..) |id, i| try body.writer.print("{s}{d}", .{ if (i == 0) "" else ",", id });
    try body.writer.writeAll("],\"continuation_ids\":[");
    for (oracle.ids[40..48], 0..) |id, i| try body.writer.print("{s}{d}", .{ if (i == 0) "" else ",", id });
    try body.writer.writeAll("]}\n{\"id\":\"greedy\",\"prompt_ids\":[");
    for (oracle.ids[3..20], 0..) |id, i| try body.writer.print("{s}{d}", .{ if (i == 0) "" else ",", id });
    try body.writer.writeAll("]}\n");
    try tmp.dir.writeFile(io, .{ .sub_path = "prompts.jsonl", .data = body.written() });

    var probe = model.ModelConfig{ .model_type = "hy_v4_flash" };
    beforeLoad(&probe, .{}) catch |e| {
        if (e == error.Hy4KldTf32AlreadyRead) std.debug.print("this test runs the teacher: run it with MLX_ENABLE_TF32=0 (this process read TF32 on before it)\n", .{});
        return e;
    };
    var quiet: kld.Out = .{ .silent = true };
    const budget: u64 = 1 << 30;
    const teacher = try std.fmt.allocPrint(arena, "{s}/teacher", .{root});
    {
        const opts: kld.Options = .{ .model_dir = dir, .prompts = try std.fmt.allocPrint(arena, "{s}/prompts.jsonl", .{root}), .out_dir = teacher, .tokens = 6, .no_template = true, .ssd_budget_bytes = budget };
        const loaded = try kld.loadModel(io, a, opts);
        defer loaded.deinit();
        try testing.expect(loaded.xfm.expert_stream != null);
        try testing.expectEqual(fwd.Products.f32, fwd.products);
        try kld.runCapture(io, a, loaded, opts, &quiet);
    }
    var base = try fixture_mod.readBaseline(a, io, teacher);
    defer base.deinit();
    try testing.expectEqualStrings("bf16", base.kv_cache_format);
    const logits = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/prompts/00_oracle/logits.f32", .{teacher}), arena, .limited(1 << 26));
    const rows = std.mem.bytesAsSlice(f32, @as([]align(1) const u8, logits));
    try testing.expectEqual(8 * oracle.vocab, rows.len);
    for (0..8) |r| {
        const got: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, logits[r * oracle.vocab * 4 ..][0 .. oracle.vocab * 4]));
        const err = rowError(got, oracle.logits[(39 + r) * oracle.vocab ..][0..oracle.vocab]);
        if (!(err <= 2e-3)) {
            std.debug.print("captured row at {d}: {e} of the oracle's peak\n", .{ 39 + r, err });
            return error.OutsideBar;
        }
    }

    // A compare loads the student as served (BF16 products); these cells run it at the teacher's f32 so each measures
    // one thing: the teacher's own width reads it back exactly, the 8-bit latent costs a little.
    var kv8: f64 = 0;
    for ([_]kv_quant.KVQuantConfig{ kv_quant.KVQuantConfig.dense, kv_quant.KVQuantConfig.engine_default }) |kv| {
        const json_path = try std.fmt.allocPrint(arena, "{s}/compare-{s}.json", .{ root, kld.kvCacheFormat(kv) });
        const opts: kld.Options = .{ .command = .compare, .model_dir = dir, .fixture = teacher, .json_out = json_path, .kv_quant_config = kv, .ssd_budget_bytes = budget };
        {
            const loaded = try kld.loadModel(io, a, opts);
            defer loaded.deinit();
            try testing.expectEqual(fwd.Products.bf16, fwd.products);
            fwd.products = .f32;
            try kld.runCompare(io, a, loaded, opts, &quiet);
        }
        const c = try compareJson(arena, io, json_path);
        try testing.expectEqual(@as(f64, 8 + 6), c.positions);
        if (kv.scheme == .off) {
            try testing.expectEqual(@as(f64, 0), c.mean_kld);
            try testing.expectEqual(@as(f64, 1), c.top1);
        } else kv8 = c.mean_kld;
    }
    // The tiny's random routing magnifies any rounding: the bound says the 8-bit latent moves the logits, not by how much.
    try testing.expect(kv8 > 0 and kv8 < 0.1);
}
