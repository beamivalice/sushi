//! What the shared Sushi Cooler contract (`src/cooler_contract_test.zig`) asks of a Flash-Next load: its tiny fixture
//! and the reads only this arch can make.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const Trunk = @import("../trunk.zig").Trunk;

/// One per format Sushi Cooler runs (`qwen4_cooler.formatOf`): the STQ toy, the served-geometry STQ cut a full-suite
/// run names (`docs/testsuite.md`: only it reaches the served kernels' widths), and the BF16 tiny.
pub const fixtures = [_][:0]const u8{ "QWEN4_STQ_TINY", "QWEN4_TEST_MODEL", "QWEN4_BF16_TINY" };

/// MTP's verdict is exact and an STQ pack's verify rows are a serial step's bytes, so its speculative request says its
/// greedy tokens. A dense BF16 product takes another plan at each row count, on MLX as on the engine.
pub fn specIsGreedy(config: *const model_mod.ModelConfig) bool {
    return @import("qwen4_cooler.zig").formatOf(config) == .stq;
}

/// The kinds of layer an exact Cooler prefill bill names (`qwen4_bill.PrefillRows`). The PLE layer has no stand-in, so
/// every set holds it.
pub const PrefillKinds = struct { gdn: bool = true, attn: bool = true, moe: bool = true };

/// The sets whose chunks the contract measures against their rows: all, and each kind left as the peak by standing in
/// the others.
pub const prefill_kind_sets = [_]PrefillKinds{ .{}, .{ .attn = false, .moe = false }, .{ .gdn = false, .moe = false }, .{ .gdn = false, .attn = false, .moe = false } };

/// The forward runs only `kinds` (the rest stood in); null runs every layer.
pub fn standInAllBut(kinds: ?PrefillKinds) void {
    const fwd = @import("qwen4_forward.zig");
    fwd.qwen4_standin_override = if (kinds) |k| .{ .gdn = !k.gdn, .attn = !k.attn, .mlp = !k.moe } else null;
}

/// Whether a forward that runs only `kinds` selects QSA blocks (its attention layers do).
pub fn selectsBlocks(kinds: PrefillKinds) bool {
    return kinds.attn;
}

/// The bytes a prefill row of a forward that runs only `kinds` holds at its peak.
pub fn prefillRowsOf(config: *const model_mod.ModelConfig, kinds: PrefillKinds) u64 {
    const r = @import("qwen4_bill.zig").coolerPrefillRows(config);
    var top = r.ple;
    if (kinds.gdn) top = @max(top, r.gdn);
    if (kinds.attn) top = @max(top, r.attn);
    if (kinds.moe) top = @max(top, r.moe);
    return top + r.trunk + r.inflight;
}

/// The bytes a row of the bill names beyond what a chunk asks: room for a plane the GPU still binds.
pub fn prefillAllowanceRows(config: *const model_mod.ModelConfig) u64 {
    return @import("qwen4_bill.zig").coolerPrefillRows(config).inflight;
}

/// Prompt lengths a request of the contract takes: one row, the verify widths' edge, a chunk past it.
pub const prompt_lengths = [_]usize{ 1, 2, 7, 8, 9, 40 };

/// A test load is resident, as the tiny pack is.
/// The BF16 checkpoint streams its experts from the pack, as a served load of it does: its fused HF expert banks are
/// read only by the expert stream.
pub fn prepare(a: std.mem.Allocator, config: *model_mod.ModelConfig, dir: []const u8) !void {
    if (@import("qwen4_cooler.zig").formatOf(config) != .bf16) return;
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 64 << 20;
    config.expert_source_dir = try a.dupe(u8, dir);
}

/// The experts a contract load of `config` drops a layer (`--drop-expert`), streamed as a drop must be; null where the
/// format keeps every expert. An STQ pack keeps a quarter.
pub fn dropExperts(a: std.mem.Allocator, config: *model_mod.ModelConfig, dir: []const u8) !?u16 {
    if (@import("qwen4_cooler.zig").formatOf(config) != .stq) return null;
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    // A cap: the kept set of every fixture fits, so a pack whose layers are all rate-grouped plans them resident.
    config.expert_cache_bytes = 4 << 30;
    config.expert_source_dir = try a.dupe(u8, dir);
    return @intCast(config.num_experts / 4 * 3);
}

/// The served tiny's chat and Responses replies run on for hundreds of tokens.
pub const server_long_replies = true;

/// What a server test serves (`src/server_test.zig`), linked into `dir`: the tiny's weights with the cut fixture's
/// tokenizer and chat template, which the tiny lacks.
pub fn serverDir(dir: std.Io.Dir) !void {
    const tiny = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    const cut = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    const words = [_][]const u8{ "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "vocab.json", "merges.txt" };
    try @import("../test_support.zig").linkInto(dir, tiny, &words);
    const io = std.Io.Threaded.global_single_threaded.io();
    for (words) |name| {
        var target: [1024]u8 = undefined;
        try dir.symLink(io, try std.fmt.bufPrint(&target, "{s}/{s}", .{ cut, name }), name, .{});
    }
}

/// The tiny pack ships no tokenizer: ids under the 128 its calibration drew from.
pub fn prompt(a: std.mem.Allocator, dir: []const u8) ![]u32 {
    _ = dir;
    const ids = try a.alloc(u32, 48);
    for (ids, 0..) |*t, i| t.* = @intCast((i * 37 + 11) % 128);
    return ids;
}

/// The weights of a Cooler load held outside an owned region: zero when every weight is mapped or built in one.
pub fn weightsOutsideRegions(xfm: *Trunk) !usize {
    const all = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(all);
    try @import("qwen4_cooler.zig").appendWeights(xfm, all);
    var outside: usize = 0;
    for (0..mlx.mlx_vector_array_size(all)) |i| {
        var arr: mlx.mlx_array = .{};
        defer _ = mlx.mlx_array_free(arr);
        try mlx.check(mlx.mlx_vector_array_get(&arr, all, i));
        try mlx.check(mlx.mlx_array_eval(arr));
        if (!model_mod.ownsPointer(mlx.engine.storePointer(arr))) outside += 1;
    }
    return outside;
}

/// `n` greedy tokens over `ids` in MTP draft and verify rounds: the spec path a request takes.
pub fn specTokens(a: std.mem.Allocator, xfm: *Trunk, ids: []const u32, n: u32) !std.ArrayList(u32) {
    return specTokensWith(a, xfm, ids, n, .{ .temperature = 0.0 });
}

/// `specTokens` under `sampling`.
pub fn specTokensWith(a: std.mem.Allocator, xfm: *Trunk, ids: []const u32, n: u32, sampling: @import("../sampler.zig").SamplingParams) !std.ArrayList(u32) {
    return specTokensAt(a, xfm, ids, n, sampling, 0);
}

/// `specTokensWith` with the request's draft depth cap at `depth` (0: the load's).
pub fn specTokensAt(a: std.mem.Allocator, xfm: *Trunk, ids: []const u32, n: u32, sampling: @import("../sampler.zig").SamplingParams, depth: u32) !std.ArrayList(u32) {
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try xfm.resetCache();
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, xfm, &tok, ids, n, sampling, &.{}, specOptions(xfm, depth));
    defer gen.deinit(a);
    var got: std.ArrayList(u32) = .empty;
    errdefer got.deinit(a);
    while (try gen.nextMtp(a)) |r| {
        defer a.free(r.tokens);
        try got.appendSlice(a, r.tokens);
    }
    return got;
}

/// A speculative request's generator options with its draft depth cap at `depth` (0: the load's).
pub fn specOptions(xfm: *Trunk, depth: u32) Generator.InitOptions {
    return .{
        .mtp = generate.MtpHeadRef.of("qwen4", xfm),
        .mtp_enabled = true,
        .model_has_mtp = true,
        .mtp_depth = depth,
        // The load's acceptance rule and tail, as the scheduler hands a served request.
    };
}

/// The rows of KV the MTP head holds for the request whose half of the head is installed.
pub fn headCacheTokens(xfm: *Trunk) usize {
    return xfm.arch_state.qwen4.mtp.?.cache.residentCapacityTokens();
}

/// `n` greedy tokens for each of `prompts`, every prompt prefilled in a slot of its own and every tick one forward of
/// all the rows: the joined decode a served Cooler load runs (`qwen4_forward.forwardQwen4DecodeRows`).
pub fn greedyRows(a: std.mem.Allocator, xfm: *Trunk, prompts: []const []const u32, n: u32) ![]std.ArrayList(u32) {
    const Slot = @import("qwen4_test_support.zig").Qwen4TestSlot;
    const rows = prompts.len;
    const slots = try a.alloc(*Slot, rows);
    defer a.free(slots);
    var made: usize = 0;
    defer for (slots[0..made]) |sl| sl.deinit(a);
    const got = try a.alloc(std.ArrayList(u32), rows);
    for (got) |*g| g.* = .empty;
    errdefer {
        for (got) |*g| g.deinit(a);
        a.free(got);
    }
    for (prompts, 0..) |p, i| {
        slots[i] = try Slot.initWith(a, xfm.config.num_hidden_layers, xfm.cache.config);
        made += 1;
        const ids = try a.alloc(i32, p.len);
        defer a.free(ids);
        for (ids, p) |*d, t| d.* = @intCast(t);
        // As a request prefills: all but the last token without a readout (no row of it is sampled), then the last
        // alone, whose logits are a decode row's.
        if (ids.len > 1) {
            slots[i].ctx.skip_lm_head = true;
            defer slots[i].ctx.skip_lm_head = false;
            const hidden = try slots[i].forward(xfm, ids[0 .. ids.len - 1]);
            defer _ = mlx.mlx_array_free(hidden);
            try mlx.check(mlx.mlx_array_eval(hidden));
        }
        const logits = try slots[i].forward(xfm, ids[ids.len - 1 ..]);
        defer _ = mlx.mlx_array_free(logits);
        try got[i].append(a, try lastArgmax(logits, xfm.s));
    }
    const ctxs = try a.alloc(*@import("../trunk.zig").ForwardCtx, rows);
    defer a.free(ctxs);
    const toks = try a.alloc(u32, rows);
    defer a.free(toks);
    for (1..n) |_| {
        for (0..rows) |i| {
            ctxs[i] = &slots[i].ctx;
            toks[i] = got[i].items[got[i].items.len - 1];
        }
        const out = try @import("qwen4_forward.zig").forwardQwen4DecodeRows(xfm, toks, ctxs, null);
        defer {
            for (out) |o| _ = mlx.mlx_array_free(o);
            a.free(out);
        }
        for (out, 0..) |o, i| try got[i].append(a, try lastArgmax(o, xfm.s));
    }
    return got;
}

fn lastArgmax(logits: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    const shape = mlx.getShape(logits);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, logits, &[_]c_int{ -1, shape[shape.len - 1] }, 2, s));
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, flat, 1, false, s));
    try mlx.check(mlx.mlx_array_eval(am));
    const ids = mlx.mlx_array_data_uint32(am) orelse return error.NoArgmax;
    return ids[mlx.mlx_array_size(am) - 1];
}

const generate = @import("../generate.zig");
const Generator = generate.Generator;
const Tokenizer = @import("../tokenizer.zig").Tokenizer;

/// Whether the load holds the MTP head the fixture ships.
pub fn headLoaded(xfm: *Trunk) bool {
    return xfm.arch_state.qwen4.mtp != null;
}
