//! The fused hyper-connection gate's decision on a pack: for every hyper-connection read of the trunk and the MTP head,
//! the (bits, group size) the fused kernels run it at, or none.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const trunk_mod = @import("../trunk.zig");
const qwen4_hc = @import("qwen4_hc.zig");
const Trunk = trunk_mod.Trunk;
const HcWeights = trunk_mod.HcWeights;
const testing = std.testing;

const Decisions = struct { fused_8_64: usize = 0, fused_other: usize = 0, declined: usize = 0 };

const Pinned = struct { pack: []const u8, want: Decisions };

const pinned = [_]Pinned{
    .{ .pack = "Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw", .want = .{ .fused_8_64 = 100 } },
    .{ .pack = "Qwen3.8-Flash-Next-Sushi-4bpw", .want = .{ .fused_8_64 = 100 } },
};

fn note(d: *Decisions, xfm: *const Trunk, w: *const HcWeights) void {
    const qp = qwen4_hc.fusedQuant(xfm, w) orelse {
        d.declined += 1;
        return;
    };
    if (qp.bits == 8 and qp.group_size == 64) d.fused_8_64 += 1 else d.fused_other += 1;
}

test "qwen4 hyper-connection gate: the fused kernels' decision on every read of a pack (QWEN4_TEST_MODEL)" {
    const model_dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    const name = std.fs.path.basename(std.mem.trimEnd(u8, model_dir, "/"));
    const want = for (pinned) |p| {
        if (std.mem.eql(u8, p.pack, name)) break p.want;
    } else return error.SkipZigTest;
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, model_dir);
    defer if (config.ngram_table_path) |path| a.free(path);
    var weights = try model_mod.loadWeights(io, a, model_dir);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const q4 = &xfm.arch_state.qwen4;
    var got: Decisions = .{};
    for (xfm.moe_layers.?) |*lw| {
        if (lw.hc_attn) |*w| note(&got, &xfm, w);
        if (lw.hc_mlp) |*w| note(&got, &xfm, w);
    }
    note(&got, &xfm, &q4.mixer.?);
    if (q4.mtp) |*m| {
        if (m.layer.hc_attn) |*w| note(&got, &xfm, w);
        if (m.layer.hc_mlp) |*w| note(&got, &xfm, w);
        note(&got, &xfm, &m.mixer);
    }
    if (!std.meta.eql(want, got)) {
        std.debug.print("hyper-connection gate {any}, pinned {any}\n", .{ got, want });
        return error.TestExpectedEqual;
    }
}
