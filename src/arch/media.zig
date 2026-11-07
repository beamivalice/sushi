//! Media of the loaded arch: its vision tower, its image processor and the prompt rows an image or a video holds.
//! A config reaches the arch whose `media.claims` holds, a processor mode the arch that declares it; a config no
//! enabled arch claims has no tower, no M-RoPE and the fixed-square processor.
//!
//! An arch's `media` declares `mode`, `claims`, `preproc`, `Tower` with `initTower`, `scratchBytes`, `resize` and
//! `normalizeChw` (patch-grid towers) or `planImage` and `decodeImage` (an arch that decodes on its own). It may add
//! `maxImagePatches`, `imageRows`, `expandVideo`, `videoResize`, `mropePositions`; each has a default.
const std = @import("std");
const mlx = @import("../mlx.zig");
const chat = @import("../chat.zig");
const model = @import("../model.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const vision_common = @import("../vision_common.zig");
const registry = @import("registry.zig");
const media_mode = @import("media_mode.zig");

pub const Mode = media_mode.Mode;
pub const modeNamed = media_mode.modeNamed;

/// The size a processor resizes an image to and the patch grid it lands on.
pub const ImagePlan = struct { width: u32, height: u32, grid_h: u32, grid_w: u32 };

/// M-RoPE position ids of a prompt: `pos` holds 3 axes of `total` ids; null when the arch has none.
pub const Mrope = struct {
    pos: ?[]const i32 = null,
    total: usize = 0,
    delta: i32 = 0,
};

/// Soft-token rows a patch-grid tower produces for one block: its merged grid.
pub fn gridRows(grid_t: u32, grid_h: u32, grid_w: u32, merge: u32) usize {
    const m = @max(merge, 1);
    return @as(usize, grid_t) * (grid_h / m) * (grid_w / m);
}

fn M(comptime tag: registry.Tag) type {
    return registry.Mod(tag).media;
}

fn ofConfig(config: *const model.ModelConfig) ?registry.Tag {
    inline for (comptime std.meta.tags(registry.Tag)) |tag| if (M(tag).claims(config)) return tag;
    return null;
}

fn ofMode(mode: Mode) ?registry.Tag {
    inline for (comptime std.meta.tags(registry.Tag)) |tag| if (M(tag).mode == mode) return tag;
    return null;
}

/// Whether an enabled arch claims the config's tower; tests that build a tower config by hand skip when none does.
pub fn claimed(config: *const model.ModelConfig) bool {
    return ofConfig(config) != null;
}

/// Whether an enabled arch declares this processor mode.
pub fn servesMode(mode: Mode) bool {
    return ofMode(mode) != null;
}

pub fn preproc(config: *const model.ModelConfig) chat.VisionPreproc {
    const tag = ofConfig(config) orelse return .{};
    return switch (tag) {
        inline else => |t| M(t).preproc(config),
    };
}

const TowerUnion = blk: {
    const tags = std.meta.tags(registry.Tag);
    var names: [tags.len][]const u8 = undefined;
    var types: [tags.len]type = undefined;
    for (tags, &names, &types) |tag, *name, *ty| {
        name.* = @tagName(tag);
        ty.* = M(tag).Tower;
    }
    break :blk @Union(.auto, registry.Tag, &names, &types, &@splat(.{}));
};

/// The resident image tower of a served model, owning its weights and GPU stream.
pub const Tower = struct {
    arch: TowerUnion,

    /// `error.MissingVisionWeights` when no arch claims the config.
    pub fn init(allocator: std.mem.Allocator, config: model.ModelConfig, weights: *model.Weights) !Tower {
        const tag = ofConfig(&config) orelse return error.MissingVisionWeights;
        return switch (tag) {
            inline else => |t| .{ .arch = @unionInit(TowerUnion, @tagName(t), try M(t).initTower(allocator, config, weights)) },
        };
    }

    pub fn deinit(self: *Tower) void {
        switch (self.arch) {
            inline else => |*t| t.deinit(),
        }
    }

    pub fn forward(self: *Tower, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        switch (self.arch) {
            inline else => |*t| return t.forward(patches, grid_h, grid_w),
        }
    }

    pub fn forwardVideo(self: *Tower, patches: mlx.mlx_array, grid_t: u32, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        switch (self.arch) {
            inline else => |*t| {
                if (comptime !@hasDecl(@TypeOf(t.*), "forwardVideo")) return error.NoVideoEncoder;
                return t.forwardVideo(patches, grid_t, grid_h, grid_w);
            },
        }
    }
};

/// Peak GPU scratch of one ViT call over `patches` rows, from the tower's measured bill.
pub fn scratchBytes(config: *const model.ModelConfig, patches: u64) u64 {
    const tag = ofConfig(config) orelse return 0;
    return switch (tag) {
        inline else => |t| M(t).scratchBytes(config, patches),
    };
}

/// Patches of the largest single image this processor admits.
pub fn maxImagePatches(vp: chat.VisionPreproc) u64 {
    const pixel_cap = vision_common.effectivePixelBounds(vp.min_pixels, vp.max_pixels).max / @max(@as(u64, vp.patch) * vp.patch, 1);
    const tag = ofMode(vp.mode) orelse return pixel_cap;
    return switch (tag) {
        inline else => |t| if (@hasDecl(M(t), "maxImagePatches")) M(t).maxImagePatches(vp) else pixel_cap,
    };
}

/// Prompt rows one image of this patch grid holds.
pub fn imageRows(config: *const model.ModelConfig, merge: u32, grid_h: u32, grid_w: u32) usize {
    const tag = ofConfig(config) orelse return gridRows(1, grid_h, grid_w, merge);
    return switch (tag) {
        inline else => |t| if (@hasDecl(M(t), "imageRows")) M(t).imageRows(merge, grid_h, grid_w) else gridRows(1, grid_h, grid_w, merge),
    };
}

/// Appends the `rows` prompt tokens of one video block whose placeholder is `token`.
pub fn expandVideo(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u32),
    config: *const model.ModelConfig,
    tokenizer: ?*const tokenizer_mod.Tokenizer,
    video: chat.VideoData,
    rows: usize,
    token: u32,
) !void {
    const tag = ofConfig(config) orelse return out.appendNTimes(allocator, token, rows);
    switch (tag) {
        inline else => |t| {
            if (comptime @hasDecl(M(t), "expandVideo")) return M(t).expandVideo(allocator, out, config, tokenizer, video, rows);
            return out.appendNTimes(allocator, token, rows);
        },
    }
}

/// M-RoPE positions over the final prompt, in prompt order of every block; empty when the arch has none.
pub fn mropePositions(
    allocator: std.mem.Allocator,
    prompt_ids: []const u32,
    images: []const chat.ImageData,
    videos: []const chat.VideoData,
    config: *const model.ModelConfig,
) !Mrope {
    const tag = ofConfig(config) orelse return .{};
    switch (tag) {
        inline else => |t| {
            if (comptime !@hasDecl(M(t), "mropePositions")) return .{};
            return M(t).mropePositions(allocator, prompt_ids, images, videos, config);
        },
    }
}

/// What a processor makes of a `src_w` x `src_h` image: the size it resizes to and its patch grid. Null for the
/// fixed-square processor and for a size no plan exists for.
pub fn planImage(vp: chat.VisionPreproc, src_w: u32, src_h: u32) ?ImagePlan {
    const tag = ofMode(vp.mode) orelse return null;
    switch (tag) {
        inline else => |t| {
            if (comptime @hasDecl(M(t), "planImage")) return M(t).planImage(vp, src_w, src_h);
            const factor = std.math.mul(u32, vp.patch, vp.merge) catch return null;
            if (factor == 0 or vp.tps == 0) return null;
            const rs = M(t).resize(vp, factor, src_h, src_w);
            return .{ .width = rs.w, .height = rs.h, .grid_h = rs.h / vp.patch, .grid_w = rs.w / vp.patch };
        },
    }
}

/// Whether the processor decodes a whole image itself, instead of resizing into a normalized CHW plane.
pub fn decodesItself(vp: chat.VisionPreproc) bool {
    const tag = ofMode(vp.mode) orelse return false;
    return switch (tag) {
        inline else => |t| @hasDecl(M(t), "decodeImage"),
    };
}

pub fn decodeImage(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32, vp: chat.VisionPreproc) ?chat.ImageData {
    const tag = ofMode(vp.mode) orelse return null;
    switch (tag) {
        inline else => |t| {
            if (comptime !@hasDecl(M(t), "decodeImage")) return null;
            return M(t).decodeImage(allocator, rgb, src_h, src_w, vp);
        },
    }
}

/// Resizes RGB8 `rgb` into the normalized CHW plane `dst` the processor's patch layout reads. `frames` is the
/// temporal extent the resize is planned over: `vp.tps` for an image.
pub fn normalizeChw(
    allocator: std.mem.Allocator,
    vp: chat.VisionPreproc,
    dst: []f32,
    rgb: []const u8,
    src_h: u32,
    src_w: u32,
    dst_h: u32,
    dst_w: u32,
    frames: u32,
) !void {
    const tag = ofMode(vp.mode) orelse return error.NoPatchGridProcessor;
    switch (tag) {
        inline else => |t| {
            if (comptime !@hasDecl(M(t), "normalizeChw")) return error.NoPatchGridProcessor;
            return M(t).normalizeChw(allocator, vp, dst, rgb, src_h, src_w, dst_h, dst_w, frames);
        },
    }
}

pub fn hasVideo(vp: chat.VisionPreproc) bool {
    const tag = ofMode(vp.mode) orelse return false;
    return switch (tag) {
        inline else => |t| @hasDecl(M(t), "videoResize"),
    };
}

/// The size every frame of a `frames`-frame video resizes to; null when the processor refuses the video.
pub fn videoResize(vp: chat.VisionPreproc, frames: u32, first_h: u32, first_w: u32, factor: u32) ?vision_common.Resized {
    const tag = ofMode(vp.mode) orelse return null;
    switch (tag) {
        inline else => |t| {
            if (comptime !@hasDecl(M(t), "videoResize")) return null;
            return M(t).videoResize(vp, frames, first_h, first_w, factor);
        },
    }
}

/// A config an enabled arch serving `model_type` makes for tests, or null when the build left that arch out.
pub fn testConfig(model_type: []const u8) ?model.ModelConfig {
    const tag = registry.ofModelType(model_type) orelse return null;
    return switch (tag) {
        inline else => |t| if (@hasDecl(M(t), "testConfig")) M(t).testConfig() else null,
    };
}

test "at most one arch claims a parsed config, whatever the claim order" {
    const docs = [_][]const u8{
        @embedFile("../fixtures/model-configs/qwen4_exp.json"),
        @embedFile("../fixtures/model-configs/mimo_v2.json"),
        @embedFile("../fixtures/glm5_config.json"),
    };
    for (docs) |doc| {
        const config = try model.parseConfigFromJson(std.testing.allocator, doc);
        try std.testing.expect(claimCount(&config) <= 1);
    }
    const dsv = testConfig("deepseek_v41") orelse return;
    try std.testing.expect(claimCount(&dsv) <= 1);
}

fn claimCount(config: *const model.ModelConfig) usize {
    var n: usize = 0;
    inline for (comptime std.meta.tags(registry.Tag)) |tag| {
        if (M(tag).claims(config)) n += 1;
    }
    return n;
}

test "a config no arch claims has no tower, no scratch and the fixed-square processor" {
    const config = model.ModelConfig{};
    try std.testing.expectEqual(Mode.gemma, preproc(&config).mode);
    try std.testing.expectEqual(@as(u64, 0), scratchBytes(&config, 1000));
    try std.testing.expectEqual(@as(usize, 9), imageRows(&config, 2, 6, 6));
    try std.testing.expectEqual(@as(?ImagePlan, null), planImage(.{}, 100, 100));
    try std.testing.expect(!hasVideo(.{}));
    try std.testing.expect(!decodesItself(.{}));
    const none = try mropePositions(std.testing.allocator, &.{ 1, 2 }, &.{}, &.{}, &config);
    try std.testing.expectEqual(@as(?[]const i32, null), none.pos);
}
