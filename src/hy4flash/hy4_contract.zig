//! What the shared Sushi Cooler contract (`src/cooler_contract_test.zig`) asks of a Hy4-Flash load: its tiny fixture
//! and the reads only this arch can make. The STQ tiny ships DFly2, so the speculative cells run on it.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const Trunk = @import("../trunk.zig").Trunk;
const Generator = @import("../generate.zig").Generator;
const Tokenizer = @import("../tokenizer.zig").Tokenizer;

pub const fixtures = [_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" };

pub const openTurn = @import("../test_support.zig").openTurn;

/// Every Hy4 load runs on Sushi Cooler: there is no MLX load to hold one to, so the cells that need one skip.
pub const mlx_arm = false;

/// A forward reads out its last row alone (`hy4_model.Readout.last`), however wide; a kld run asks for every row.
pub const reads_out_last_row_only = true;

/// One row, a group of 4, the window of 8, the 16-key selection threshold and past it, the oracle's whole prompt.
pub const prompt_lengths = [_]usize{ 1, 2, 7, 16, 17, 40 };

/// A test load streams the experts, as the release's on a 32 GB Mac, and keeps DFly2 where the fixture ships it.
pub fn prepare(a: std.mem.Allocator, config: *model_mod.ModelConfig, dir: []const u8) !void {
    config.expert_streaming = true;
    config.stream_mtp_head = @import("hy4_load.zig").shipsHead(std.Io.Threaded.global_single_threaded.io(), a, dir);
    config.expert_cache_bytes = 4 << 20;
    config.expert_source_dir = try a.dupe(u8, dir);
}

/// The oracle's 40 prompt ids (allocated with `a`), read before any load; a pack without an oracle (the STQ tiny), 40
/// seeded ids under its vocabulary.
pub fn prompt(a: std.mem.Allocator, dir: []const u8) ![]u32 {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/oracle.safetensors", .{dir}, 0);
    defer a.free(path);
    const io = std.Io.Threaded.global_single_threaded.io();
    std.Io.Dir.cwd().access(io, path, .{}) catch return seededPrompt(a, dir);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var w = model_mod.Weights.init(a);
    defer w.deinit();
    try model_mod.loadSafetensorsFile(a, &w, path.ptr, cpu, true);
    // Read as stored, with no op: the engine refuses an MLX stream while a test's held load stands.
    const ids = w.get("ids") orelse return error.Hy4OracleIds;
    if (mlx.mlx_array_dtype(ids) != .int32) return error.Hy4OracleIds;
    try mlx.check(mlx.mlx_array_eval(ids));
    const v = (mlx.mlx_array_data_int32(ids) orelse return error.HostRead)[0..40];
    const out = try a.alloc(u32, v.len);
    for (v, out) |x, *d| d.* = @intCast(x);
    return out;
}

fn seededPrompt(a: std.mem.Allocator, dir: []const u8) ![]u32 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const raw = try d.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    defer a.free(raw);
    const parsed = try std.json.parseFromSlice(struct { text_config: struct { vocab_size: u32 } }, a, raw, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const out = try a.alloc(u32, 40);
    var r = std.Random.DefaultPrng.init(0x4d4);
    for (out) |*id| id.* = r.random().uintLessThan(u32, parsed.value.text_config.vocab_size - 1) + 1;
    return out;
}

/// The weights of a Cooler load held outside an owned region: every tensor the model reads is one the load placed.
pub fn weightsOutsideRegions(xfm: *Trunk) !usize {
    const w = xfm.arch_state.hy4flash.weights orelse return error.Hy4ModelMissing;
    var outside: usize = 0;
    var it = w.map.iterator();
    while (it.next()) |e| {
        try mlx.check(mlx.mlx_array_eval(e.value_ptr.*));
        if (!model_mod.ownsPointer(mlx.engine.storePointer(e.value_ptr.*))) outside += 1;
    }
    return outside;
}

/// Whether the load holds DFly2, which a fixture that ships it drafts with.
pub fn headLoaded(xfm: *Trunk) bool {
    return xfm.arch_state.hy4flash.drafter != null;
}

/// `n` greedy tokens over `ids` in DFly2 rounds, every round width swept: the spec path a request takes.
pub fn specTokens(a: std.mem.Allocator, xfm: *Trunk, ids: []const u32, n: u32) !std.ArrayList(u32) {
    return specTokensWith(a, xfm, ids, n, .{ .temperature = 0.0 });
}

/// Whether a verify row is its serial step's bytes, so a greedy round says greedy's tokens: not yet (`hy4_drafter`
/// serves a fixed width until the forward is width-exact).
pub fn specIsGreedy(config: *const model_mod.ModelConfig) bool {
    return config.trunk_quant_rate != null and config.expert_layout == .exl3_k4;
}

/// `specTokens` under `sampling`.
pub fn specTokensWith(a: std.mem.Allocator, xfm: *Trunk, ids: []const u32, n: u32, sampling: @import("../sampler.zig").SamplingParams) !std.ArrayList(u32) {
    return specRounds(a, xfm, ids, n, sampling, .sweep);
}

/// `specTokensWith`, every round verifying `depth` drafts.
pub fn specTokensAt(a: std.mem.Allocator, xfm: *Trunk, ids: []const u32, n: u32, sampling: @import("../sampler.zig").SamplingParams, depth: u32) !std.ArrayList(u32) {
    return specRounds(a, xfm, ids, n, sampling, if (depth == 0) .off else .{ .rows = @intCast(depth + 1) });
}

fn specRounds(a: std.mem.Allocator, xfm: *Trunk, ids: []const u32, n: u32, sampling: @import("../sampler.zig").SamplingParams, diag: @import("../drafter_rows.zig").Diag) !std.ArrayList(u32) {
    if (xfm.arch_state.hy4flash.drafter == null) return error.SkipZigTest;
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try xfm.resetCache();
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, xfm, &tok, ids, n, sampling, &.{}, .{ .drafter_enabled = true });
    defer gen.deinit(a);
    gen.drafter_diag = diag;
    var got: std.ArrayList(u32) = .empty;
    errdefer got.deinit(a);
    while (try gen.nextDrafterRound(a)) |r| {
        defer a.free(r.tokens);
        try got.appendSlice(a, r.tokens);
    }
    return got;
}
