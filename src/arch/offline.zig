//! What the offline tools (`sushi kld`, the imatrix capture, the hidden-state capture) ask of an arch. An arch's
//! `kld` module (declared in its `arch.zig`) holds the ops it overrides; an arch without one takes the default.
//!
//! Ops an arch `kld` module may declare:
//! - `hiddenCaptureWidth(config)`: the width of one hidden-capture row; default `hidden_size`.
//! - `refuse(config, opts)`: launch options the arch cannot honor.
//! - `tryRun(allocator, io, opts, out)`: the arch's own capture path; true when it ran.
//! - `afterLoad(xfm, config, kv)`: arch state to set up once the model is loaded.
//! - `capturePromptChunks(...)`: the prompt forward that appends every block boundary; null falls back to the
//!   trunk's capture.
//! - `imatrix_naming`: how the source checkpoint names the expert blocks, for a model_type the engine arms from.
const std = @import("std");
const model = @import("../model.zig");
const mlx = @import("../mlx.zig");
const trf = @import("../transformer.zig");
const hidden_capture = @import("../hidden_capture.zig");
const imatrix = @import("../imatrix.zig");
const kld = @import("../kld.zig");
const registry = @import("registry.zig");
const spec = @import("spec.zig");

fn hasOp(comptime tag: registry.Tag, comptime name: []const u8) bool {
    const Mod = registry.Mod(tag);
    return @hasDecl(Mod, "kld") and @hasDecl(Mod.kld, name);
}

pub fn hiddenCaptureWidth(config: *const model.ModelConfig) usize {
    const s = spec.of(config) orelse return config.hidden_size;
    switch (s) {
        inline else => |c, t| {
            if (comptime hasOp(t, "hiddenCaptureWidth")) return registry.Mod(t).kld.hiddenCaptureWidth(c);
            return c.hidden_size;
        },
    }
}

pub fn refuse(config: *const model.ModelConfig, opts: kld.Options) !void {
    const s = spec.of(config) orelse return;
    switch (s) {
        inline else => |c, t| {
            if (comptime hasOp(t, "refuse")) try registry.Mod(t).kld.refuse(c, opts);
        },
    }
}

pub fn tryRun(allocator: std.mem.Allocator, io: std.Io, opts: kld.Options, out: *kld.Out) !bool {
    inline for (comptime std.meta.tags(registry.Tag)) |t| {
        if (comptime hasOp(t, "tryRun")) {
            if (try registry.Mod(t).kld.tryRun(allocator, io, opts, out)) return true;
        }
    }
    return false;
}

pub fn afterLoad(xfm: *trf.Transformer, config: model.ModelConfig, kv: trf.KVQuantConfig) !void {
    switch (registry.handleOf(xfm)) {
        inline else => |m, t| {
            if (comptime hasOp(t, "afterLoad")) try registry.Mod(t).kld.afterLoad(m, config, kv);
        },
    }
}

pub fn capturePrompt(
    allocator: std.mem.Allocator,
    l: *kld.Loaded,
    ctx: *trf.ForwardCtx,
    ids: []const u32,
    writer: *hidden_capture.Writer,
    chunk: usize,
) !?mlx.mlx_array {
    const s = spec.of(&l.config) orelse return null;
    switch (s) {
        inline else => |_, t| {
            if (comptime hasOp(t, "capturePromptChunks")) return try registry.Mod(t).kld.capturePromptChunks(allocator, l, ctx, ids, writer, chunk);
            return null;
        },
    }
}

/// The expert-block naming of a model_type the engine arms an imatrix capture for; null when its arch has none.
pub fn imatrixNaming(model_type: []const u8) ?imatrix.Naming {
    const tag = registry.ofModelType(model_type) orelse return null;
    switch (tag) {
        inline else => |t| {
            if (comptime hasOp(t, "imatrix_naming")) return registry.Mod(t).kld.imatrix_naming;
            return null;
        },
    }
}

test "a hidden capture row keeps every residual stream the arch carries and defaults to the hidden size" {
    const testing = std.testing;
    const rows = [_]struct { model_type: []const u8, hc: u32, want: usize }{
        .{ .model_type = "qwen4_exp", .hc = 4, .want = 10240 },
        .{ .model_type = "glm5_next", .hc = 4, .want = 10240 },
        .{ .model_type = "mimo_v2", .hc = 0, .want = 2560 },
        .{ .model_type = "not_a_served_arch", .hc = 4, .want = 2560 },
    };
    for (rows) |r| {
        if (registry.ofModelType(r.model_type) == null and model.isServedArch(r.model_type)) continue;
        const config = model.ModelConfig{ .model_type = r.model_type, .hidden_size = 2560, .hc_count = r.hc };
        try testing.expectEqual(r.want, hiddenCaptureWidth(&config));
    }
}

test "every served architecture prefills a prompt in its serving start width, whatever the box" {
    const testing = std.testing;
    const rows = [_]struct { config: model.ModelConfig, want: usize }{
        .{ .config = .{ .model_type = "qwen4_exp" }, .want = 4096 },
        .{ .config = .{ .model_type = "glm5_next" }, .want = 2048 },
        .{ .config = .{ .model_type = "mimo_v2", .has_sliding_window = true, .sliding_window = 128, .head_dim = 192 }, .want = 2048 },
    };
    for (rows) |r| {
        if (registry.ofModelType(r.config.model_type) == null) continue;
        try testing.expectEqual(r.want, kld.promptChunk(&r.config));
    }
}

test "an imatrix naming resolves through the arch that claims the model type" {
    const testing = std.testing;
    if (registry.ofModelType("qwen4_exp") != null) {
        try testing.expectEqualStrings("model.language_model.layers.", imatrixNaming("qwen4_exp").?.layer_prefix);
    }
    if (registry.ofModelType("mimo_v2") != null) {
        try testing.expectEqualStrings("model.layers.", imatrixNaming("mimo_v2").?.layer_prefix);
    }
    try testing.expect(imatrixNaming("not_a_served_arch") == null);
}
