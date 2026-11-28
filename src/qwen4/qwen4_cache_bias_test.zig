//! `--expert-cache-bias` on a streamed Flash-Next STQ load, on the tiny pack (QWEN4_STQ_TINY).
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const model_load = @import("../model_load.zig");
const arch_handle = @import("../arch/handle.zig");
const arch_sched = @import("../arch/sched.zig");
const prefix_arch = @import("../arch/prefix.zig");
const contract = @import("../cooler_contract_test.zig");
const expert_stream = @import("../expert_stream.zig");
const Slot = @import("qwen4_test_support.zig").Qwen4TestSlot;
const io = std.Io.Threaded.global_single_threaded.io();

const decode_steps = 10;

/// What one streamed load of the tiny pack made of the same prompt and the same forced decode tokens.
const Run = struct {
    /// Hash of the prefill's hidden rows, and of every decode step's logits.
    prefill: u64,
    decode: u64,
    /// Experts the SSD read for the decode steps.
    decode_fills: u64,
    /// The SSD cache namespace the load keys by, when the arch names more than the config's tag.
    key: ?[]u8,

    fn deinit(self: *Run) void {
        if (self.key) |k| testing.allocator.free(k);
    }
};

fn bytesHash(arr: mlx.mlx_array, s: mlx.mlx_stream) !u64 {
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_astype(&wide, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(wide));
    const n: usize = @intCast(mlx.mlx_array_size(wide));
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(wide).?[0..n]));
}

fn resetGroupCache(c: *expert_stream.GroupCache) void {
    @memset(c.expert_to_slot, -1);
    @memset(c.slot_to_expert, std.math.maxInt(u16));
    @memset(c.ages, 0);
    @memset(c.ready, false);
    @memset(c.swap_starved, 0);
    c.tick = 0;
}

fn resetEngineCaches(engine: *expert_stream.Engine) void {
    for (engine.layers) |*layer| {
        resetGroupCache(&layer.cache);
        if (layer.grouped) |gl| for (gl.caches[0..gl.caches_made]) |*c| resetGroupCache(c);
        if (layer.spec_slots.ctx != null) {
            _ = mlx.mlx_array_free(layer.spec_slots);
            layer.spec_slots = .{ .ctx = null };
        }
        if (layer.spec_raw.ctx != null) {
            _ = mlx.mlx_array_free(layer.spec_raw);
            layer.spec_raw = .{ .ctx = null };
        }
        if (layer.spec_pen.ctx != null) {
            _ = mlx.mlx_array_free(layer.spec_pen);
            layer.spec_pen = .{ .ctx = null };
        }
        if (layer.spec_host.len != 0) @memset(layer.spec_host, -2);
    }
}

fn runOnLoaded(ld: *contract.Load, dir: []const u8, ids: []const u32) !Run {
    const a = testing.allocator;
    const xfm = ld.xfm;
    const engine = xfm.expert_stream.?;
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    try xfm.resetCache();

    const slot = try Slot.initWith(a, xfm.config.num_hidden_layers, xfm.cache.config);
    defer slot.deinit(a);
    const prompt = try a.alloc(i32, ids.len);
    defer a.free(prompt);
    for (prompt, ids) |*d, t| d.* = @intCast(t);
    slot.ctx.skip_lm_head = true;
    const hidden = try slot.forward(xfm, prompt);
    defer _ = mlx.mlx_array_free(hidden);
    const prefill = try bytesHash(hidden, xfm.s);
    slot.ctx.skip_lm_head = false;

    var hasher = std.hash.Wyhash.init(0);
    const fills_before = engine.fill_experts_total;
    for (0..decode_steps) |step| {
        const forced = [_]i32{prompt[(step * 5) % prompt.len]};
        const logits = try slot.forward(xfm, &forced);
        defer _ = mlx.mlx_array_free(logits);
        hasher.update(std.mem.asBytes(&try bytesHash(logits, xfm.s)));
    }
    return .{
        .prefill = prefill,
        .decode = hasher.final(),
        .decode_fills = engine.fill_experts_total - fills_before,
        .key = try prefix_arch.diskLayoutKey(a, &ld.config, dir, xfm),
    };
}

test "qwen4 cache bias: bias 0 routes as no flag, a bias favours cached experts at one row and never at a wide chunk, and its states keep a namespace (QWEN4_STQ_TINY)" {
    try contract.qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: contract.Load = undefined;
            try ld.prepare(Q, dir, true);
            {
                errdefer ld.config.deinit(a);
                var budget: u64 = 1 << 30;
                const plan = while (budget <= 64 << 30) : (budget += 1 << 30) {
                    break model_load.planExpertStreamingWith(io, a, &ld.config, dir, budget, false, 0) catch continue;
                } else return error.TestUnexpectedResult;
                model_load.applyStreamingPlan(&ld.config, plan, budget);
                // A cache of twenty experts a layer, of the pack's sixty-four: a one-row step misses.
                ld.config.expert_cache_bytes = 20 * ld.config.expertLayerCount() * plan.cache.expert_bytes;
                if (ld.config.expert_source_dir == null) ld.config.expert_source_dir = try a.dupe(u8, dir);
                try testing.expect(ld.config.expert_streaming and !ld.config.expert_resident);
                ld.config.test_skip_warm = true;
                ld.config.expert_cache_bias_override = null;
                try testing.expectEqual(@as(f32, 0), ld.config.expertCacheBias());
                ld.config.expert_cache_bias_override = 0;
                try testing.expectEqual(@as(f32, 0), ld.config.expertCacheBias());
            }
            try ld.loadServed(dir);
            defer _ = ld.close();

            var exact = try runOnLoaded(&ld, dir, ids);
            defer exact.deinit();
            try testing.expect(exact.key == null);

            resetEngineCaches(ld.xfm.expert_stream.?);
            ld.xfm.expert_stream.?.cache_bias = 8;
            ld.config.expert_cache_bias_override = 8;
            var biased = try runOnLoaded(&ld, dir, ids);
            defer biased.deinit();
            // A wide chunk routes exactly; a one-row step takes a cached expert over a better one it would read.
            errdefer std.debug.print("exact fills {d}, bias 8 fills {d}, decode hashes {x} {x}\n", .{ exact.decode_fills, biased.decode_fills, exact.decode, biased.decode });
            try testing.expectEqual(exact.prefill, biased.prefill);
            try testing.expect(biased.decode != exact.decode);
            try testing.expect(biased.decode_fills < exact.decode_fills);
            try testing.expect(std.mem.indexOf(u8, biased.key orelse return error.TestUnexpectedResult, "-cbias") != null);
        }
    }.run);
}

test "qwen4 pick tolerance: a lossy pick keeps its states out of the exact namespace, and a tolerance and a bias do not share one (QWEN4_STQ_TINY)" {
    const a = testing.allocator;
    const saved = expert_stream.pick_tolerance;
    defer expert_stream.pick_tolerance = saved;
    var xfm = @import("../test_support.zig").testTrunk("qwen4_exp");
    xfm.arch = .qwen4;
    var engine = expert_stream.Engine{
        .allocator = a,
        .geometry = undefined,
        .plan = undefined,
        .store = undefined,
        .layers = &.{},
        .s = .{},
        .forward_clock = undefined,
        .cache_bias = 0,
    };
    xfm.expert_stream = &engine;
    const cfg = model_mod.ModelConfig{ .model_type = "qwen4_exp", .expert_layout = .exl3_k4 };

    expert_stream.pick_tolerance = 0;
    try testing.expect((try prefix_arch.diskLayoutKey(a, &cfg, "", &xfm)) == null);

    expert_stream.pick_tolerance = 0.3;
    const picked = (try prefix_arch.diskLayoutKey(a, &cfg, "", &xfm)) orelse return error.TestUnexpectedResult;
    defer a.free(picked);
    engine.cache_bias = 8;
    const both = (try prefix_arch.diskLayoutKey(a, &cfg, "", &xfm)) orelse return error.TestUnexpectedResult;
    defer a.free(both);
    expert_stream.pick_tolerance = 0.2;
    engine.cache_bias = 0;
    const other = (try prefix_arch.diskLayoutKey(a, &cfg, "", &xfm)) orelse return error.TestUnexpectedResult;
    defer a.free(other);
    expert_stream.pick_tolerance = 0;
    engine.cache_bias = 8;
    const biased = (try prefix_arch.diskLayoutKey(a, &cfg, "", &xfm)) orelse return error.TestUnexpectedResult;
    defer a.free(biased);

    const keys = [_][]const u8{ picked, both, other, biased };
    try testing.expect(std.mem.indexOf(u8, picked, "-pick") != null);
    for (keys, 0..) |x, i| for (keys[i + 1 ..]) |y| try testing.expect(!std.mem.eql(u8, x, y));
}
