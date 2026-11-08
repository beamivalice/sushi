//! Fixtures the bill tests share: the served shapes as configs, and the pin fixture over all four archs.
const std = @import("std");
const arch_config = @import("arch/config.zig");
const model_mod = @import("model.zig");
const nax_gate = @import("nax_gate.zig");
const PinCfg = model_mod.ModelConfig;
const bills = @import("bills.zig");
const testing = std.testing;

/// A stand-in for the deployed long-context pack: a GDN+MoE trunk with a QSA
/// indexer. Declares `qwen4_exp`: every long-context mechanism is gated on it.
pub fn longCtxTestConfig() model_mod.ModelConfig {
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "qwen4_exp"; // arch-scan-exempt: test fixture, not a gate
    cfg.num_hidden_layers = 40;
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 256;
    cfg.hidden_size = 2560;
    cfg.intermediate_size = 9216;
    cfg.intermediate_size_declared = true;
    cfg.quant_bits = 4;
    cfg.max_position_embeddings = 1048576;
    cfg.indexer_budget = 2048;
    cfg.indexer_n_heads = 4;
    cfg.indexer_head_dim = 128;
    cfg.indexer_compress_ratio = 4;
    return cfg;
}

pub fn qwen4RequestTestConfig() model_mod.ModelConfig {
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "qwen4_exp"; // arch-scan-exempt: test fixture, not a gate
    cfg.num_hidden_layers = 48;
    cfg.full_attention_interval = 4;
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 256;
    // No sliding-band layers: the field defaults to true, which routes a different chunk policy.
    cfg.has_sliding_window = false;
    cfg.hidden_size = 2560;
    cfg.moe_intermediate_size = 1024;
    cfg.num_experts_per_tok = 8;
    cfg.num_experts = 512;
    cfg.quant_bits = 4;
    cfg.max_position_embeddings = 1048576;
    cfg.indexer_budget = 2048;
    cfg.indexer_n_heads = 4;
    cfg.indexer_head_dim = 128;
    cfg.indexer_compress_ratio = 4;
    cfg.pinned_prefill_chunk = 1024; // what load-time sizing gives at 1M ctx
    return cfg;
}

/// The geometry the served pack actually ships (`Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit`):
/// a narrower expert with a shared one beside it, and the GatedDeltaNet head counts
/// `qwen4RequestTestConfig` leaves at zero. Both chunk-scaled terms of the bill read them, so
/// the fixture's step is not the deployed one.
pub fn qwen4DeployedTestConfig() model_mod.ModelConfig {
    var cfg = qwen4RequestTestConfig();
    cfg.moe_intermediate_size = 640;
    cfg.num_experts_per_tok = 10;
    cfg.shared_expert_intermediate_size = 640;
    cfg.linear_num_key_heads = 16;
    cfg.linear_num_value_heads = 48;
    cfg.linear_key_head_dim = 128;
    cfg.linear_value_head_dim = 128;
    cfg.linear_conv_kernel_dim = 4;
    cfg.quant_group_size = 64;
    cfg.quant_mode = .affine;
    return cfg;
}

/// The served Flash-Next geometry at `--qwen-gdn 16`, as the M1 Max runs measured it.
pub fn qwen4MeasuredTestConfig() model_mod.ModelConfig {
    var cfg = qwen4DeployedTestConfig();
    cfg.gdn_state_f32 = false;
    return cfg;
}

/// The qwen4_exp shape of the live #353 report: Qwen3.8-Flash-Next mixed-4-8bit, `--ctx-size 1048576 --kv-quant 8`.
pub fn qwen4ExpOomConfig() model_mod.ModelConfig {
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "qwen4_exp"; // arch-scan-exempt: test fixture, not a gate
    cfg.num_hidden_layers = 48;
    cfg.full_attention_interval = 4; // 12 caching layers, 36 GatedDeltaNet
    cfg.num_attention_heads = 24;
    cfg.num_key_value_heads = 2;
    cfg.head_dim = 256;
    cfg.hidden_size = 2560;
    cfg.moe_intermediate_size = 640;
    cfg.num_experts = 512;
    cfg.num_experts_per_tok = 10;
    cfg.shared_expert_intermediate_size = 640;
    cfg.linear_num_key_heads = 16;
    cfg.linear_key_head_dim = 128;
    cfg.linear_num_value_heads = 48;
    cfg.linear_value_head_dim = 128;
    cfg.linear_conv_kernel_dim = 4;
    cfg.indexer_budget = 512;
    cfg.indexer_head_dim = 128;
    cfg.indexer_compress_ratio = 4;
    // Pinned: with nothing pinned `getEffectiveContextLength` reads live GPU memory and every
    // expectation below depends on the box.
    cfg.pinned_context = 1_048_576;
    return cfg;
}

/// The live 364k agent session: `qwen4ExpOomConfig` plus the deployed pack's indexer budget 2048,
/// four indexer heads, affine-4 weights and `--ctx-size 786432`; reproduces the incident's bills to the MB.
pub fn qwen4ExpLive364kConfig() model_mod.ModelConfig {
    var cfg = qwen4ExpOomConfig();
    cfg.pinned_context = 786_432;
    cfg.indexer_budget = 2048;
    cfg.indexer_n_heads = 4;
    cfg.quant_bits = 4;
    cfg.quant_mode = .affine;
    cfg.max_position_embeddings = 1_048_576;
    return cfg;
}

pub fn mimoV2BillConfig() model_mod.ModelConfig {
    var c = model_mod.ModelConfig{};
    c.model_type = "mimo_v2";
    c.num_hidden_layers = 4;
    c.hidden_size = 384;
    c.num_attention_heads = 4;
    c.has_sliding_window = true;
    c.has_explicit_layer_types = true;
    c.sliding_window = 128;
    c.layer_is_global[0] = true;
    c.layer_is_global[3] = true;
    c.head_dim = 192;
    c.v_head_dim = 128;
    c.num_key_value_heads = 3;
    c.global_head_dim = 192;
    c.global_v_head_dim = 128;
    c.num_global_key_value_heads = 2;
    // Pinned as a load pins it: unpinned, the reservation's headroom follows an auto-context
    // sized from live GPU memory and from the attention switches a test flips between bills.
    c.pinned_context = 1 << 20;
    return c;
}

/// The Flash geometry the band sheet scales with: 64 query heads, 39 of 48 layers sliding.
pub fn mimoV2FlashBillConfig() model_mod.ModelConfig {
    var cfg = mimoV2BillConfig();
    cfg.num_hidden_layers = 48;
    cfg.hidden_size = 4096;
    cfg.num_attention_heads = 64;
    cfg.num_key_value_heads = 8;
    cfg.num_global_key_value_heads = 4;
    cfg.layer_is_global[3] = false;
    for ([_]u32{ 0, 5, 11, 17, 23, 29, 35, 41, 47 }) |li| cfg.layer_is_global[li] = true;
    return cfg;
}

pub fn glmAssistantFixture() !model_mod.ModelConfig {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    cfg.dflash_capture_bytes_per_token = (5 * 4096 + 2 * 4096) * 2 + 5 * 8 * 128 * 4;
    return cfg;
}

pub const PinFx = struct {
    qwen: PinCfg,
    mimo: PinCfg,
    glm: PinCfg,
    dsv: PinCfg,
    saved_nax: ?bool,

    // The NAX probe reads the device: the pins price the SIMD arm on every machine.
    pub fn init() !PinFx {
        const saved_nax = nax_gate.vqmm_nax_probe_override;
        nax_gate.vqmm_nax_probe_override = false;
        errdefer nax_gate.vqmm_nax_probe_override = saved_nax;
        const a = std.testing.allocator;
        var qwen = try model_mod.parseConfigFromJson(a, @embedFile("fixtures/model-configs/qwen4_exp.json"));
        errdefer qwen.deinit(a);
        var mimo = try model_mod.parseConfigFromJson(a, arch_config.fixtureConfig("mimo2") orelse arch_config.unbuilt_config);
        errdefer mimo.deinit(a);
        var glm = try model_mod.parseConfigFromJson(a, @embedFile("fixtures/glm5_config.json"));
        errdefer glm.deinit(a);
        var dsv = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
        dsv.dsv41 = try arch_config.releaseOwn("dsv41");
        dsv.serve_defaults = arch_config.releaseServeDefaults("dsv41");
        dsv.max_position_embeddings = 1_048_576;
        // Pinned as a load pins them: unpinned, the bills read the live GPU.
        for ([_]*PinCfg{ &qwen, &mimo, &glm, &dsv }) |c| {
            c.pinned_context = 262_144;
            c.pinned_prefill_chunk = 2048;
        }
        return .{ .qwen = qwen, .mimo = mimo, .glm = glm, .dsv = dsv, .saved_nax = saved_nax };
    }

    pub fn deinit(self: *PinFx) void {
        nax_gate.vqmm_nax_probe_override = self.saved_nax;
        const a = std.testing.allocator;
        self.qwen.deinit(a);
        self.mimo.deinit(a);
        self.glm.deinit(a);
    }

    pub fn all(self: *PinFx) [4]*const PinCfg {
        return .{ &self.qwen, &self.mimo, &self.glm, &self.dsv };
    }
};

pub const LoadBillFixtures = struct {
    qwen: model_mod.ModelConfig,
    mimo: model_mod.ModelConfig,
    glm: model_mod.ModelConfig,

    pub fn init() !LoadBillFixtures {
        const a = std.testing.allocator;
        var qwen = try model_mod.parseConfigFromJson(a, @embedFile("fixtures/model-configs/qwen4_exp.json"));
        errdefer qwen.deinit(a);
        var mimo = try model_mod.parseConfigFromJson(a, arch_config.fixtureConfig("mimo2") orelse arch_config.unbuilt_config);
        errdefer mimo.deinit(a);
        return .{ .qwen = qwen, .mimo = mimo, .glm = try model_mod.parseConfigFromJson(a, @embedFile("fixtures/glm5_config.json")) };
    }

    pub fn deinit(self: *LoadBillFixtures) void {
        self.qwen.deinit(std.testing.allocator);
        self.mimo.deinit(std.testing.allocator);
        self.glm.deinit(std.testing.allocator);
    }
};

const pin_archs = [_][]const u8{ "qwen4", "mimo2", "glm5", "dsv41" };

/// A pin lists one run of values per served pack, in `pin_archs` order, equal runs unless `lens` says otherwise. The runs
/// of an arch `-Darchs` leaves out are not checked; a build with none of them skips.
fn pinRows(comptime name: []const u8, want: []const u64, lens: ?[4]usize, cfgs: [4]*const PinCfg, comptime run: fn (*const PinCfg, *bills.PinGot) void) !void {
    try testing.expect(lens != null or want.len % pin_archs.len == 0);
    var start: usize = 0;
    var checked: usize = 0;
    inline for (pin_archs, 0..) |arch, i| {
        const n = if (lens) |l| l[i] else want.len / pin_archs.len;
        if (bills.archBuilt(arch)) {
            var got = bills.PinGot{};
            run(cfgs[i], &got);
            try bills.pinEq(name, want[start..][0..n], got.slice());
            checked += 1;
        }
        start += n;
    }
    try testing.expectEqual(want.len, start);
    if (checked == 0) return error.SkipZigTest;
}

pub fn pinConfigs(comptime name: []const u8, want: []const u64, comptime run: fn (*const PinCfg, *bills.PinGot) void) !void {
    var fx = try PinFx.init();
    defer fx.deinit();
    try pinRows(name, want, null, fx.all(), run);
}

/// `pinConfigs` for a bill whose run adds a different number of values per pack.
pub fn pinConfigsRagged(comptime name: []const u8, want: []const u64, lens: [4]usize, comptime run: fn (*const PinCfg, *bills.PinGot) void) !void {
    var fx = try PinFx.init();
    defer fx.deinit();
    try pinRows(name, want, lens, fx.all(), run);
}


pub fn pinLoadConfigs(comptime name: []const u8, want: []const u64, comptime run: fn (*const PinCfg, *bills.PinGot) void) !void {
    var fx = try LoadBillFixtures.init();
    defer fx.deinit();
    var dsv = PinCfg{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    dsv.dsv41 = try arch_config.releaseOwn("dsv41");
    try pinRows(name, want, null, .{ &fx.qwen, &fx.mimo, &fx.glm, &dsv }, run);
}
