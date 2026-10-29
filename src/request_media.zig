const std = @import("std");
const mlx = @import("mlx.zig");
const tokenizer_mod = @import("tokenizer.zig");
const chat_mod = @import("chat.zig");
const model_mod = @import("model.zig");
const dsv41_config = @import("dsv41/dsv41_config.zig");
const qwen_vision = @import("qwen4/qwen_vision.zig");
const vision_common = @import("vision_common.zig");
const mimo_vision = @import("mimo2/mimo_vision.zig");
const glm5_vision = @import("glm5/glm5_vision.zig");
const dsv41_vision = @import("dsv41/dsv41_vision.zig");
const mrope_mod = @import("mrope.zig");
const log = @import("log.zig");
const responses_mod = @import("responses.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const tokenize_cache_mod = @import("tokenize_cache.zig");
const scheduler_mod = @import("scheduler.zig");
const model_registry_mod = @import("model_registry.zig");
const model_settings = @import("model_settings.zig");
const stb = @import("stb");
const webp = @import("webp");
const Tokenizer = tokenizer_mod.Tokenizer;
const LoadedModel = model_registry_mod.LoadedModel;
const testing = std.testing;
const server = @import("server.zig");
const MropeData = server.MropeData;
const WirePos = server.WirePos;
const currentGpuMemoryCeiling = server.currentGpuMemoryCeiling;
const mapGenerationError = server.mapGenerationError;
const testDsv41Vp = server.testDsv41Vp;
const testWire = server.testWire;
const test_png_blue = server.test_png_blue;
const test_png_red = server.test_png_red;
const towerFitFault = server.towerFitFault;

const test_png_0 = "data:image/png;base64,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";
const test_png_1 = "data:image/png;base64,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";
const test_png_2 = "data:image/png;base64,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";

/// A grayscale PNG of `w` x `h` as a data URL: stored deflate blocks, so any size builds without an encoder.
fn testPngUrl(a: std.mem.Allocator, w: u32, h: u32) ![]u8 {
    var raw = std.ArrayList(u8).empty;
    defer raw.deinit(a);
    for (0..h) |y| {
        try raw.append(a, 0);
        for (0..w) |x| try raw.append(a, @truncate(x *% 7 +% y *% 13));
    }
    var z = std.ArrayList(u8).empty;
    defer z.deinit(a);
    try z.appendSlice(a, &.{ 0x78, 0x01 });
    var off: usize = 0;
    while (true) {
        const n = @min(raw.items.len - off, 65535);
        const last = off + n == raw.items.len;
        try z.append(a, @intFromBool(last));
        var len: [4]u8 = undefined;
        std.mem.writeInt(u16, len[0..2], @intCast(n), .little);
        std.mem.writeInt(u16, len[2..4], @intCast(~@as(u16, @intCast(n))), .little);
        try z.appendSlice(a, &len);
        try z.appendSlice(a, raw.items[off..][0..n]);
        off += n;
        if (last) break;
    }
    var adler: [4]u8 = undefined;
    std.mem.writeInt(u32, &adler, std.hash.Adler32.hash(raw.items), .big);
    try z.appendSlice(a, &adler);

    var png = std.ArrayList(u8).empty;
    defer png.deinit(a);
    try png.appendSlice(a, &.{ 0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a });
    var ihdr: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr[0..4], w, .big);
    std.mem.writeInt(u32, ihdr[4..8], h, .big);
    @memcpy(ihdr[8..13], &[_]u8{ 8, 0, 0, 0, 0 });
    for ([_]struct { []const u8, []const u8 }{ .{ "IHDR", &ihdr }, .{ "IDAT", z.items }, .{ "IEND", "" } }) |chunk| {
        var word: [4]u8 = undefined;
        std.mem.writeInt(u32, &word, @intCast(chunk[1].len), .big);
        try png.appendSlice(a, &word);
        const body_at = png.items.len;
        try png.appendSlice(a, chunk[0]);
        try png.appendSlice(a, chunk[1]);
        std.mem.writeInt(u32, &word, std.hash.Crc32.hash(png.items[body_at..]), .big);
        try png.appendSlice(a, &word);
    }
    const prefix = "data:image/png;base64,";
    const out = try a.alloc(u8, prefix.len + std.base64.standard.Encoder.calcSize(png.items.len));
    @memcpy(out[0..prefix.len], prefix);
    _ = std.base64.standard.Encoder.encode(out[prefix.len..], png.items);
    return out;
}

const test_vision_modes = [_]chat_mod.VisionPreproc{
    .{ .mode = .qwen, .patch = 16, .tps = 2, .merge = 2, .min_pixels = 65536, .max_pixels = 1048576 },
    .{ .mode = .mimo, .patch = 16, .tps = 2, .merge = 2, .min_pixels = 8192, .max_pixels = 8388608 },
    .{ .mode = .glm5, .patch = 14, .tps = 2, .merge = 2, .min_tokens = 16, .max_tokens = 8000 },
};

test "a request holds its images as the files it was sent, not the patches they decode to" {
    const a = std.testing.allocator;
    const vp = try testDsv41Vp();
    const urls = [_][]const u8{ test_png_0, test_png_1, test_png_2 };
    var media = RequestMedia.init(a);
    defer media.deinit();
    const list = media.images(try media.openImages());
    var file_bytes: u64 = 0;
    var decoded_bytes: u64 = 0;
    const feat: u64 = 3 * @as(u64, vp.patch) * vp.patch * 4;
    for (0..300) |i| {
        const url = urls[i % urls.len];
        try std.testing.expect(appendImageUrlContent(a, list, url, vp));
        const img = list.items[list.items.len - 1];
        try std.testing.expect(img.undecoded);
        file_bytes += std.base64.standard.Decoder.calcSizeForSlice(url[std.mem.indexOf(u8, url, ";base64,").? + 8 ..]) catch unreachable;
        decoded_bytes += @as(u64, img.grid_h) * img.grid_w * feat;
    }
    var held: u64 = 0;
    for (list.items) |img| held += img.pixels.len;
    try std.testing.expectEqual(file_bytes, held);
    // Every one of these upscales to the processor's minimum canvas, so a decode is thousands of times the file.
    try std.testing.expect(held * 100 < decoded_bytes);
}

test "the plan read from an image's header is the grid and size its full decode reaches" {
    const a = std.testing.allocator;
    const sizes = [_][2]u32{
        .{ 1, 1 },      .{ 7, 5 },     .{ 32, 32 },  .{ 40, 30 },   .{ 100, 3 },
        .{ 3, 100 },    .{ 640, 480 }, .{ 480, 640 }, .{ 1280, 720 }, .{ 3000, 20 },
        .{ 20, 3000 },  .{ 1920, 1080 },
    };
    var modes = std.ArrayList(chat_mod.VisionPreproc).empty;
    defer modes.deinit(a);
    try modes.append(a, try testDsv41Vp());
    try modes.appendSlice(a, &test_vision_modes);
    for (sizes) |size| {
        const url = try testPngUrl(a, size[0], size[1]);
        defer a.free(url);
        const b64 = url[std.mem.indexOf(u8, url, ";base64,").? + 8 ..];
        for (modes.items) |vp| {
            const eager = parseImageUrlContent(a, url, vp);
            const lazy = planEncodedImage(a, b64, vp);
            defer if (lazy) |img| a.free(img.pixels);
            defer if (eager) |img| a.free(img.pixels);
            try std.testing.expectEqual(eager == null, lazy == null);
            const want = eager orelse continue;
            const got = lazy.?;
            try std.testing.expect(got.undecoded);
            try std.testing.expectEqual(want.width, got.width);
            try std.testing.expectEqual(want.height, got.height);
            try std.testing.expectEqual(want.grid_h, got.grid_h);
            try std.testing.expectEqual(want.grid_w, got.grid_w);
            // The rows the prompt places for the plan are the rows the tower will produce.
            if (vp.mode == .dsv41) try std.testing.expectEqual(
                dsv41_vision.spanRows(want.grid_h, want.grid_w, vp.merge),
                dsv41_vision.spanRows(got.grid_h, got.grid_w, vp.merge),
            );
        }
    }
}

test "decoding a held image later gives the patches an up-front decode gave" {
    const a = std.testing.allocator;
    const vp = try testDsv41Vp();
    var modes = std.ArrayList(chat_mod.VisionPreproc).empty;
    defer modes.deinit(a);
    try modes.append(a, vp);
    try modes.appendSlice(a, &test_vision_modes);
    for ([_][]const u8{ test_png_0, test_png_1, test_png_2 }) |url| {
        for (modes.items) |mode| {
            const eager = parseImageUrlContent(a, url, mode) orelse return error.DecodeFailed;
            defer a.free(eager.pixels);
            const held = imageUrlContent(a, url, mode) orelse return error.DecodeFailed;
            defer a.free(held.pixels);
            try std.testing.expect(held.undecoded);
            const later = decodeImageToPixels(a, held.pixels, mode) orelse return error.DecodeFailed;
            defer a.free(later.pixels);
            try std.testing.expectEqualSlices(u8, eager.pixels, later.pixels);
            try std.testing.expectEqual(eager.grid_h, later.grid_h);
            try std.testing.expectEqual(eager.grid_w, later.grid_w);
        }
    }
}

test "the encode bill of a held image is the bill of its decode" {
    const a = std.testing.allocator;
    var cfg: model_mod.ModelConfig = .{ .qv_patch = 14, .qv_temporal_patch = 1, .qv_heads = 16, .qv_hidden = 1024, .qv_out_hidden = 4096 };
    cfg.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    const vp = visionPreprocFromConfig(&cfg);
    for ([_][]const u8{ test_png_0, test_png_1, test_png_2 }) |url| {
        const eager = parseImageUrlContent(a, url, vp) orelse return error.DecodeFailed;
        defer a.free(eager.pixels);
        const held = imageUrlContent(a, url, vp) orelse return error.DecodeFailed;
        defer a.free(held.pixels);
        const rows = dsv41_vision.spanRows(eager.grid_h, eager.grid_w, vp.merge);
        const want = visionEncodeBill(&cfg, &.{eager}, &.{}, rows);
        const got = visionEncodeBill(&cfg, &.{held}, &.{}, rows);
        try std.testing.expectEqual(want.bytes, got.bytes);
        try std.testing.expectEqual(want.largest_group_patches, got.largest_group_patches);
    }
}

test "an image the header cannot place is refused when the request is read" {
    const a = std.testing.allocator;
    const vp = try testDsv41Vp();
    var media = RequestMedia.init(a);
    defer media.deinit();
    const list = media.images(try media.openImages());
    try std.testing.expect(!appendImageUrlContent(a, list, "data:image/png;base64,AAAA", vp));
    try std.testing.expect(!appendImageUrlContent(a, list, "data:image/png;base64,!!!!", vp));
    try std.testing.expectEqual(@as(usize, 0), list.items.len);
}

test "an undecoded image's chain key is the key its decoded pixels always had" {
    const a = std.testing.allocator;
    var dsv: model_mod.ModelConfig = .{};
    dsv.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    const qwen: model_mod.ModelConfig = .{ .qwen_vision = true, .qv_patch = 16, .qv_temporal_patch = 2, .qv_merge = 2, .qv_min_pixels = 65536, .qv_max_pixels = 1048576 };
    const urls = [_][]const u8{ test_png_0, test_png_1, test_png_2 };
    for ([_]*const model_mod.ModelConfig{ &dsv, &qwen }) |config| {
        const vp = visionPreprocFromConfig(config);
        // More images than one decode window, so the key crosses a refill.
        const n = 2 * scheduler_mod.decode_workers + 3;
        var eager: [n]chat_mod.ImageData = undefined;
        var held: [n]chat_mod.ImageData = undefined;
        var placed: [n]PlacedMedia = undefined;
        for (0..n) |i| {
            eager[i] = parseImageUrlContent(a, urls[i % urls.len], vp) orelse return error.DecodeFailed;
            held[i] = imageUrlContent(a, urls[i % urls.len], vp) orelse return error.DecodeFailed;
            try std.testing.expect(held[i].undecoded);
            placed[i] = .{ .kind = .image, .start = 3 + 10 * i, .rows = 1 };
        }
        defer for (0..n) |i| {
            a.free(eager[i].pixels);
            a.free(held[i].pixels);
        };
        // `mediaChain` over decoded pixels is main's key code, unchanged.
        const want = try mediaChain(a, &placed, &eager, &.{}, null);
        defer a.free(want);
        var bad: usize = 0;
        const got = try keyMedia(a, config, &placed, &held, &.{}, &bad);
        defer a.free(got);
        try std.testing.expectEqualSlices(prefix_cache_mod.MediaSpan, want, got);
    }
}

test "the chain key of a decoded image is pinned: existing prefix-cache records must keep matching" {
    const a = std.testing.allocator;
    const vp = try testDsv41Vp();
    const img = parseImageUrlContent(a, test_png_0, vp) orelse return error.DecodeFailed;
    defer a.free(img.pixels);
    const chain = try mediaChain(a, &.{.{ .kind = .image, .start = 3, .rows = 1 }}, &.{img}, &.{}, null);
    defer a.free(chain);
    try std.testing.expectEqual(@as(u64, 16185945656631848181), chain[0].key);
}

test "a request whose image will not decode is refused naming it when it is keyed" {
    const a = std.testing.allocator;
    var cfg: model_mod.ModelConfig = .{};
    cfg.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    const vp = visionPreprocFromConfig(&cfg);
    const good = imageUrlContent(a, test_png_0, vp) orelse return error.DecodeFailed;
    defer a.free(good.pixels);
    var bad_file = good;
    bad_file.pixels = "not an image";
    const images = [_]chat_mod.ImageData{ good, bad_file };
    const placed = [_]PlacedMedia{ .{ .kind = .image, .start = 3, .rows = 1 }, .{ .kind = .image, .start = 9, .rows = 1 } };
    var bad: usize = 0;
    try std.testing.expectError(error.ImageUndecodable, keyMedia(a, &cfg, &placed, &images, &.{}, &bad));
    try std.testing.expectEqual(@as(usize, 1), bad);
}

test "a media encode refused after the restore reaches the client with the text and status it was given" {
    var buf: [320]u8 = undefined;
    const fit = scheduler_mod.MediaRefusal{ .server_error = false, .buf = undefined, .len = 0 };
    var refused = fit;
    refused.len = (std.fmt.bufPrint(&refused.buf, "needs ~9 MB, ~1 MB is available", .{}) catch unreachable).len;
    server.slot_media_refusal = refused;
    const wire = mapGenerationError(error.MediaRefused, &buf);
    try std.testing.expectEqual(@as(u32, 400), wire.code);
    try std.testing.expectEqualStrings("needs ~9 MB, ~1 MB is available", wire.message);
    try std.testing.expectEqual(@as(?scheduler_mod.MediaRefusal, null), server.slot_media_refusal);
    refused.server_error = true;
    server.slot_media_refusal = refused;
    try std.testing.expectEqual(@as(u32, 500), mapGenerationError(error.MediaRefused, &buf).code);
}

/// A request carrying images/video/audio on a model serving WITHOUT its tower
/// (`--no-vision`, or a checkpoint with no vision weights) is refused by name.
/// Before this the media parts were parsed and then silently dropped — the
/// model answered the text alone (a 200 with a hallucinated "Sky" for a house).
/// A streamed load left its tower off only because nothing asked for it (not `--no-vision`).
fn streamedTowerOffByDefault(config: *const model_mod.ModelConfig) bool {
    const choice = scheduler_mod.visionChoiceFor(config, true);
    return config.expert_streaming and config.has_vision and !choice.on and choice.source == .default;
}

fn mediaRejectReason(messages: []const chat_mod.Message, streamed_tower_off: bool) ?[]const u8 {
    for (messages) |m| {
        if (m.images != null or m.videos != null) return if (streamed_tower_off)
            "This model streams its experts from SSD and was loaded without its vision tower; relaunch with --vision to accept image/video content"
        else
            "This model is serving without its vision tower (--no-vision or no vision weights); image/video content is not supported";
        if (m.audio != null) return "This model is serving without its audio embedder; input_audio content is not supported";
    }
    return null;
}

test "mediaRejectReason: media on a tower-less model is refused by name, text passes" {
    const t = std.testing;
    const text = [_]chat_mod.Message{.{ .role = "user", .content = "hi" }};
    try t.expect(mediaRejectReason(&text, false) == null);
    const img = [_]chat_mod.Message{ .{ .role = "user", .content = "hi" }, .{ .role = "user", .content = "look", .images = &[_]chat_mod.ImageData{} } };
    try t.expect(std.mem.indexOf(u8, mediaRejectReason(&img, false).?, "vision tower") != null);
    const aud = [_]chat_mod.Message{.{ .role = "user", .content = "listen", .audio = &[_]chat_mod.AudioData{} }};
    try t.expect(std.mem.indexOf(u8, mediaRejectReason(&aud, false).?, "audio") != null);
    // A streamed load that left its tower off names the flag that loads it.
    const streamed = mediaRejectReason(&img, true).?;
    try t.expect(std.mem.indexOf(u8, streamed, "streams its experts") != null);
    try t.expect(std.mem.indexOf(u8, streamed, "--vision") != null);
    try t.expect(std.mem.indexOf(u8, streamed, "--no-vision") == null);
}

test "only a streamed load whose tower is off by default names --vision in its media refusal" {
    const t = std.testing;
    const saved = model_settings.vision_flag;
    defer model_settings.vision_flag = saved;
    var cfg = model_mod.ModelConfig{ .has_vision = true, .expert_streaming = true };
    model_settings.vision_flag = null;
    try t.expect(streamedTowerOffByDefault(&cfg));
    model_settings.vision_flag = false;
    try t.expect(!streamedTowerOffByDefault(&cfg));
    model_settings.vision_flag = null;
    cfg.vision_override = false;
    try t.expect(!streamedTowerOffByDefault(&cfg));
    cfg.vision_override = null;
    cfg.expert_streaming = false;
    try t.expect(!streamedTowerOffByDefault(&cfg));
    cfg.expert_streaming = true;
    cfg.has_vision = false;
    try t.expect(!streamedTowerOffByDefault(&cfg));
    // DeepSeek-V4.1 streams WITH its tower, so there is nothing to name.
    var ds = model_mod.ModelConfig{ .has_vision = true, .expert_streaming = true, .model_type = "deepseek_v41" };
    try t.expect(!streamedTowerOffByDefault(&ds));
    model_settings.vision_flag = false;
    try t.expect(!streamedTowerOffByDefault(&ds));
}

/// A named refusal of a request's media: the status and the text the
/// surface's error sender ships.
pub const MediaFault = struct {
    server_error: bool = false,
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn init(server_error: bool, comptime fmt: []const u8, args: anytype) MediaFault {
        var f: MediaFault = .{ .server_error = server_error };
        f.len = (std.fmt.bufPrint(&f.buf, fmt, args) catch @as([]u8, &f.buf)).len;
        return f;
    }

    pub fn text(self: *const MediaFault) []const u8 {
        return self.buf[0..self.len];
    }

    pub fn status(self: *const MediaFault) []const u8 {
        return if (self.server_error) "500 Internal Server Error" else "400 Bad Request";
    }

    pub fn code(self: *const MediaFault) u16 {
        return if (self.server_error) 500 else 400;
    }

    pub fn errorType(self: *const MediaFault) []const u8 {
        return if (self.server_error) "server_error" else "invalid_request_error";
    }
};

fn imageDecodeFault(pos: WirePos, reason: []const u8) MediaFault {
    return MediaFault.init(false, "{s}[{d}]: image could not be decoded: {s}", .{ pos.list, pos.index, reason });
}

fn imageCountFault(n_images: usize) MediaFault {
    return MediaFault.init(false, "the request carries {d} images; at most {d} are accepted", .{ n_images, chat_mod.MAX_REQUEST_IMAGES });
}

/// Why an image URL did not decode, in words a client can act on.
fn imageRejectReason(url: []const u8) []const u8 {
    if (!std.mem.startsWith(u8, url, "data:")) return "remote URLs are not fetched; send a base64 data URL (data:image/jpeg|png|gif|bmp|webp;base64,...)";
    const sep = std.mem.indexOf(u8, url, ";base64,") orelse return "the data URL is not base64 (data:image/jpeg|png|gif|bmp|webp;base64,...)";
    for (url[sep + 8 ..]) |c| {
        if (!std.ascii.isAlphanumeric(c) and c != '+' and c != '/' and c != '=') return "the base64 payload is malformed";
    }
    return "the payload is not a readable JPEG, PNG, GIF, BMP or WebP image";
}

pub fn responsesMediaRefusal(pi: *const responses_mod.ParsedInput) ?MediaFault {
    if (pi.n_images > chat_mod.MAX_REQUEST_IMAGES) return imageCountFault(pi.n_images);
    const f = pi.media_fault orelse return null;
    return switch (f.kind) {
        .undecodable_image => imageDecodeFault(.{ .list = "input", .index = f.item }, f.reason),
        .unsupported_audio => MediaFault.init(false, "input[{d}]: input_audio is not supported by this model", .{f.item}),
    };
}

/// `responses.ImageUrlDecoder`: decode into `list`, or say why not.
pub fn decodeImageUrlInto(allocator: std.mem.Allocator, list: *std.ArrayList(chat_mod.ImageData), url: []const u8, vp: chat_mod.VisionPreproc) ?[]const u8 {
    if (appendImageUrlContent(allocator, list, url, vp)) return null;
    return imageRejectReason(url);
}

pub fn openAiImageUrl(part: std.json.Value) ?[]const u8 {
    const img = part.object.get("image_url") orelse return null;
    return switch (img) {
        .string => |s| s,
        .object => |o| if (o.get("url")) |u| (if (u == .string) u.string else null) else null,
        else => null,
    };
}

/// Every message's images and videos in the order the template renders their
/// placeholders: message order, each message's own order within a kind.
const FlatMedia = struct {
    images: []chat_mod.ImageData,
    videos: []chat_mod.VideoData,
    has_audio: bool,

    pub fn deinit(self: *FlatMedia, allocator: std.mem.Allocator) void {
        allocator.free(self.images);
        allocator.free(self.videos);
    }
};

pub fn flattenMedia(allocator: std.mem.Allocator, messages: []const chat_mod.Message) !FlatMedia {
    var n_images: usize = 0;
    var n_videos: usize = 0;
    var has_audio = false;
    for (messages) |m| {
        if (m.images) |imgs| n_images += imgs.len;
        if (m.videos) |vids| n_videos += vids.len;
        if (m.audio != null) has_audio = true;
    }
    const images = try allocator.alloc(chat_mod.ImageData, n_images);
    errdefer allocator.free(images);
    const videos = try allocator.alloc(chat_mod.VideoData, n_videos);
    var next_image: usize = 0;
    var next_video: usize = 0;
    for (messages) |m| {
        if (m.images) |imgs| {
            @memcpy(images[next_image..][0..imgs.len], imgs);
            next_image += imgs.len;
        }
        if (m.videos) |vids| {
            @memcpy(videos[next_video..][0..vids.len], vids);
            next_video += vids.len;
        }
    }
    return .{ .images = images, .videos = videos, .has_audio = has_audio };
}

/// A media kind the loaded model cannot encode is refused by name, never
/// answered from the text alone.
fn mediaKindRefusal(messages: []const chat_mod.Message, flat: FlatMedia, has_tower: bool, streamed_tower_off: bool) ?MediaFault {
    if (!has_tower) return MediaFault.init(false, "{s}", .{mediaRejectReason(messages, streamed_tower_off).?});
    if (flat.has_audio) return MediaFault.init(false, "input_audio is not supported by this model: its tower encodes images and video only", .{});
    return null;
}

/// Peak GPU scratch of one ViT call over `patches` rows, from each tower's measured bill.
fn visionScratchBytes(config: *const model_mod.ModelConfig, patches: u64) u64 {
    if (config.mimo_vision) return mimo_vision.encodeScratchBytes(config, patches);
    if (config.glm5_vision) return glm5_vision.encodeScratchBytes(config, patches);
    if (config.isDsv41()) return dsv41_vision.encodeScratchBytes(config, patches);
    return qwen_vision.encodeScratchBytes(config, patches);
}

/// GPU bytes a request's media encode peaks at: the largest block's tower scratch
/// (a video's block is one temporal group, `forwardVideo` evaluates each alone),
/// every block's f32 pixels, and three bf16 copies of the soft-token rows.
pub fn visionEncodeBill(config: *const model_mod.ModelConfig, images: []const chat_mod.ImageData, videos: []const chat_mod.VideoData, rows: usize) struct { bytes: u64, largest_group_patches: u64 } {
    var worst: u64 = 0;
    var input_bytes: u64 = 0;
    const patch_bytes: u64 = 3 * @as(u64, config.qv_temporal_patch) * config.qv_patch * config.qv_patch * 4;
    for (images) |im| {
        const patches = @as(u64, im.grid_h) * im.grid_w;
        worst = @max(worst, patches);
        input_bytes += @max(im.pixels.len, patches * patch_bytes);
    }
    for (videos) |vd| {
        const patches = @as(u64, vd.grid_h) * vd.grid_w;
        worst = @max(worst, patches);
        input_bytes += @max(vd.pixels.len, @as(u64, vd.grid_t) * patches * patch_bytes);
    }
    const output_bytes = @as(u64, rows) * @max(config.qv_out_hidden, 1) * 2;
    return .{ .bytes = visionScratchBytes(config, worst) + input_bytes + 3 * output_bytes, .largest_group_patches = worst };
}

/// The encode bill of the largest single image this tower's processor admits: a streamed
/// `--vision` load proves it fits beside its budget. Every request is still checked live.
pub fn largestImageEncodeBytes(config: *const model_mod.ModelConfig) u64 {
    const vp = visionPreprocFromConfig(config);
    const merge: u64 = @max(vp.merge, 1);
    const patches: u64 = if (vp.mode == .glm5)
        @as(u64, vp.max_tokens) * merge * merge
    else if (vp.mode == .dsv41)
        merge * merge * (vp.max_tokens - 2)
    else
        vision_common.effectivePixelBounds(vp.min_pixels, vp.max_pixels).max / @max(@as(u64, vp.patch) * vp.patch, 1);
    const largest = [_]chat_mod.ImageData{.{ .pixels = &.{}, .width = 0, .height = 0, .grid_h = 1, .grid_w = @intCast(patches) }};
    return visionEncodeBill(config, &largest, &.{}, @intCast(patches / (merge * merge))).bytes;
}

pub fn visionEncodeBillBytes(config: *const model_mod.ModelConfig, images: []const chat_mod.ImageData, videos: []const chat_mod.VideoData, rows: usize) u64 {
    return visionEncodeBill(config, images, videos, rows).bytes;
}

/// What `scheduler_mod.vision_encode_available` reads on the inference thread, right before an encode.
pub fn visionEncodeAvailable(config: *const model_mod.ModelConfig) u64 {
    var active_mem: usize = 0;
    _ = mlx.mlx_get_active_memory(&active_mem);
    return currentGpuMemoryCeiling(config, active_mem) -| active_mem;
}

/// One media block of the expanded prompt, in prompt order.
const PlacedMedia = struct { kind: chat_mod.MediaPart.Kind, start: usize, rows: usize };

const MediaPlacement = struct {
    ids: []u32,
    items: []PlacedMedia,
    rows: usize,

    pub fn deinit(self: *MediaPlacement, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.items);
    }
};

/// How many image / video placeholders the rendered prompt holds: one pad
/// token per block, which `placeMedia` expands.
fn countMediaPlaceholders(ids: []const u32, config: *const model_mod.ModelConfig) struct { images: usize, videos: usize } {
    var images: usize = 0;
    var videos: usize = 0;
    for (ids) |t| {
        if (config.image_token_id > 0 and t == config.image_token_id) images += 1;
        if (config.video_token_id > 0 and t == config.video_token_id) videos += 1;
    }
    return .{ .images = images, .videos = videos };
}

/// Soft-token rows a patch-grid tower produces for one block: its merged grid.
fn mediaRows(grid_t: u32, grid_h: u32, grid_w: u32, merge: u32) usize {
    const m = @max(merge, 1);
    return @as(usize, grid_t) * (grid_h / m) * (grid_w / m);
}

/// Expand each template placeholder to its block's run of pad tokens: the
/// k-th image pad takes `images[k]`, the k-th video pad `videos[k]`. The
/// caller has checked the counts match (`countMediaPlaceholders`).
fn placeMedia(
    allocator: std.mem.Allocator,
    ids: []const u32,
    config: *const model_mod.ModelConfig,
    images: []const chat_mod.ImageData,
    videos: []const chat_mod.VideoData,
    tokenizer: ?*const Tokenizer,
) !MediaPlacement {
    const merge = visionPreprocFromConfig(config).merge;
    var out = std.ArrayList(u32).empty;
    errdefer out.deinit(allocator);
    var items = std.ArrayList(PlacedMedia).empty;
    errdefer items.deinit(allocator);
    var next_image: usize = 0;
    var next_video: usize = 0;
    var rows_total: usize = 0;
    for (ids) |t| {
        const is_image = config.image_token_id > 0 and t == config.image_token_id;
        const is_video = config.video_token_id > 0 and t == config.video_token_id;
        if (!is_image and !is_video) {
            try out.append(allocator, t);
            continue;
        }
        const rows = if (is_image) blk: {
            const im = images[next_image];
            next_image += 1;
            // A gridless (fixed-square) tower has no per-image row count here.
            if (im.grid_h == 0) return error.MediaLayoutUnsupported;
            break :blk if (config.isDsv41()) dsv41_vision.spanRows(im.grid_h, im.grid_w, merge) else mediaRows(1, im.grid_h, im.grid_w, merge);
        } else blk: {
            const vd = videos[next_video];
            next_video += 1;
            break :blk mediaRows(vd.grid_t, vd.grid_h, vd.grid_w, merge);
        };
        try items.append(allocator, .{ .kind = if (is_image) .image else .video, .start = out.items.len, .rows = rows });
        if (is_video and config.glm5_vision) {
            const tok = tokenizer orelse return error.MediaLayoutUnsupported;
            const video = videos[next_video - 1];
            const per_group = rows / video.grid_t;
            for (0..video.grid_t) |group| {
                try out.append(allocator, config.vision_start_token_id);
                try out.appendNTimes(allocator, config.image_token_id, per_group);
                try out.append(allocator, config.vision_end_token_id);
                var stamp: [64]u8 = undefined;
                const seconds = @as(f64, @floatFromInt(group * config.qv_temporal_patch)) / video.fps;
                const text = try std.fmt.bufPrint(&stamp, "{d:.1} seconds", .{seconds});
                const stamp_ids = try tok.encode(allocator, text);
                defer allocator.free(stamp_ids);
                try out.appendSlice(allocator, stamp_ids);
            }
        } else try out.appendNTimes(allocator, t, rows);
        rows_total += rows;
    }
    const owned_ids = try out.toOwnedSlice(allocator);
    errdefer allocator.free(owned_ids);
    return .{ .ids = owned_ids, .items = try items.toOwnedSlice(allocator), .rows = rows_total };
}

/// Prefix-cache identity of a request's media: per block in prompt order, a
/// hash of that block and every block before it, pixels and positions both.
fn mediaChain(
    allocator: std.mem.Allocator,
    placed: []const PlacedMedia,
    images: []const chat_mod.ImageData,
    videos: []const chat_mod.VideoData,
    window: ?*scheduler_mod.ImageDecodeWindow,
) ![]prefix_cache_mod.MediaSpan {
    const out = try allocator.alloc(prefix_cache_mod.MediaSpan, placed.len);
    errdefer allocator.free(out);
    var prev: u64 = 0x5ec0de;
    var next_image: usize = 0;
    var next_video: usize = 0;
    for (placed, out) |p, *span| {
        var h = std.hash.Wyhash.init(prev);
        h.update(std.mem.asBytes(&p.start));
        switch (p.kind) {
            .image => {
                // An undecoded image is keyed by the pixels its decode makes, a window of them at a time.
                const im = images[next_image];
                const at = next_image;
                const px: scheduler_mod.VisionImagePixels = if (window) |w|
                    w.view(at) orelse return error.ImageUndecodable
                else
                    .{ .pixels = im.pixels, .width = im.width, .height = im.height, .grid_h = im.grid_h, .grid_w = im.grid_w };
                defer if (window) |w| w.release(at);
                next_image += 1;
                h.update(std.mem.asBytes(&px.width));
                h.update(std.mem.asBytes(&px.height));
                h.update(std.mem.asBytes(&px.grid_h));
                h.update(std.mem.asBytes(&px.grid_w));
                h.update(px.pixels);
            },
            .video => {
                const vd = videos[next_video];
                next_video += 1;
                h.update(std.mem.asBytes(&vd.grid_t));
                h.update(std.mem.asBytes(&vd.grid_h));
                h.update(std.mem.asBytes(&vd.grid_w));
                h.update(std.mem.asBytes(&vd.fps));
                h.update(vd.pixels);
            },
        }
        prev = h.final() | 1;
        span.* = .{ .start = p.start, .key = prev };
    }
    return out;
}

/// `mediaChain` over a request whose images may still be files: each is decoded for its key, a window
/// at a time, and the pixels are freed as soon as the key has taken them. The encode decodes again.
fn keyMedia(
    allocator: std.mem.Allocator,
    config: *const model_mod.ModelConfig,
    placed: []const PlacedMedia,
    images: []const chat_mod.ImageData,
    videos: []const chat_mod.VideoData,
    bad_image: *usize,
) ![]prefix_cache_mod.MediaSpan {
    const views = try allocator.alloc(scheduler_mod.VisionImagePixels, images.len);
    defer allocator.free(views);
    var any_undecoded = false;
    for (images, views) |im, *v| {
        v.* = .{ .pixels = im.pixels, .width = im.width, .height = im.height, .grid_h = im.grid_h, .grid_w = im.grid_w, .undecoded = im.undecoded };
        any_undecoded = any_undecoded or im.undecoded;
    }
    if (!any_undecoded) return mediaChain(allocator, placed, images, videos, null);
    var window = scheduler_mod.ImageDecodeWindow{ .allocator = allocator, .config = config, .decode = &decodeEncodedImage, .images = views };
    defer window.deinit();
    defer media_decode_ns += window.decode_ns;
    return mediaChain(allocator, placed, images, videos, &window) catch |err| {
        bad_image.* = window.failed;
        return err;
    };
}

/// What a request's media hands its slot. Whatever the slot has not taken is
/// freed by `deinit`.
pub const PreparedMedia = struct {
    deferred: ?*scheduler_mod.DeferredMedia = null,
    chain: []const prefix_cache_mod.MediaSpan = &.{},
    mrope: MropeData = .{},

    pub fn deinit(self: *PreparedMedia, allocator: std.mem.Allocator) void {
        if (self.deferred) |d| d.deinit();
        allocator.free(self.chain);
        if (self.mrope.pos) |p| allocator.free(p);
        self.* = .{};
    }
};

const MediaOutcome = union(enum) { ready: PreparedMedia, refused: MediaFault };

/// The CPU half of `prepareRequestMedia`: the request-shape refusals and the placeholder
/// expansion, so count_tokens counts the rows a prefill would hold without running the tower.
pub fn placeRequestMedia(allocator: std.mem.Allocator, lm: *LoadedModel, messages: []const chat_mod.Message, flat: FlatMedia, ids: []const u32) !union(enum) { placed: MediaPlacement, refused: MediaFault } {
    const config = lm.config.?;
    if (mediaKindRefusal(messages, flat, lm.vision_encoder != null, streamedTowerOffByDefault(config))) |fault| return .{ .refused = fault };
    const found = countMediaPlaceholders(ids, config);
    if (found.images != flat.images.len or found.videos != flat.videos.len) {
        return .{ .refused = MediaFault.init(false, "the rendered prompt holds {d} image and {d} video placeholders for {d} images and {d} videos (media in a system message, or placeholder text in the content)", .{ found.images, found.videos, flat.images.len, flat.videos.len }) };
    }
    const placed = placeMedia(allocator, ids, config, flat.images, flat.videos, lm.tokenizer.?) catch |err| switch (err) {
        error.MediaLayoutUnsupported => return .{ .refused = MediaFault.init(false, "this model's vision tower cannot place images inside a conversation", .{}) },
        else => return err,
    };
    return .{ .placed = placed };
}

/// Place every message's media at its template placeholder, refuse by name
/// what cannot be served and key it for the prefix cache. The slot encodes it
/// once the prefix restore says which blocks it still needs. On `.ready`,
/// `ids` holds the expanded prompt.
pub fn prepareRequestMedia(
    allocator: std.mem.Allocator,
    lm: *LoadedModel,
    messages: []const chat_mod.Message,
    ids: *[]u32,
    ctx_limit: usize,
) !MediaOutcome {
    var flat = try flattenMedia(allocator, messages);
    defer flat.deinit(allocator);
    if (flat.images.len == 0 and flat.videos.len == 0 and !flat.has_audio) return .{ .ready = .{} };
    const config = lm.config.?;
    var placed = switch (try placeRequestMedia(allocator, lm, messages, flat, ids.*)) {
        .placed => |p| p,
        .refused => |fault| return .{ .refused = fault },
    };
    defer placed.deinit(allocator);
    if (placed.ids.len > ctx_limit) {
        return .{ .refused = MediaFault.init(false, "prompt ({d} tokens, {d} of them from {d} images and {d} videos) exceeds the {d}-token context", .{ placed.ids.len, placed.rows, flat.images.len, flat.videos.len, ctx_limit }) };
    }
    var bad_image: usize = 0;
    const chain = keyMedia(allocator, config, placed.items, flat.images, flat.videos, &bad_image) catch |err| switch (err) {
        error.ImageUndecodable => return .{ .refused = MediaFault.init(false, "image {d} could not be decoded: the payload is not a readable JPEG, PNG, GIF, BMP or WebP image", .{bad_image + 1}) },
        else => return err,
    };
    var ready: PreparedMedia = .{ .chain = chain };
    errdefer ready.deinit(allocator);

    log.info("Multimodal: processing {d} image(s), {d} video(s)\n", .{ flat.images.len, flat.videos.len });
    const images = try allocator.dupe(chat_mod.ImageData, flat.images);
    errdefer allocator.free(images);
    const videos = try allocator.dupe(chat_mod.VideoData, flat.videos);
    errdefer allocator.free(videos);
    const blocks = try allocator.alloc(scheduler_mod.MediaBlock, placed.items.len);
    errdefer allocator.free(blocks);
    for (placed.items, blocks) |p, *blk| blk.* = .{ .kind = p.kind, .rows = p.rows };
    const deferred = try allocator.create(scheduler_mod.DeferredMedia);
    deferred.* = .{ .allocator = allocator, .images = images, .videos = videos, .blocks = blocks, .decode_ns = takeMediaDecodeNs() };
    ready.deferred = deferred;
    ready.mrope = computeQwenMrope(allocator, placed.ids, flat.images, flat.videos, config) catch MropeData{};
    log.info("  Multimodal: {d} soft tokens placed in a {d}-token prompt\n", .{ placed.rows, placed.ids.len });

    allocator.free(ids.*);
    ids.* = placed.ids;
    placed.ids = placed.ids[0..0];
    return .{ .ready = ready };
}

/// Image decode time this connection thread has spent since the last request took it.
threadlocal var media_decode_ns: u64 = 0;

fn takeMediaDecodeNs() u64 {
    defer media_decode_ns = 0;
    return media_decode_ns;
}

fn monotonicNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Compute the interleaved-M-RoPE table from the FINAL prompt_ids (after pad
/// expansion) + every image and video grid in prompt order. Returns an empty
/// bundle when the model isn't Qwen-vision or there are no grids. Caller owns `pos`.
fn computeQwenMrope(allocator: std.mem.Allocator, prompt_ids: []const u32, images: []const chat_mod.ImageData, videos: []const chat_mod.VideoData, config: *const model_mod.ModelConfig) !MropeData {
    if (!config.qwen_vision) return .{};
    // Each modality's grids in its own document order: getRopeIndex
    // interleaves the two lists by whichever marker occurs first in tokens.
    var image_grids = std.ArrayList(mrope_mod.ImageGrid).empty;
    defer image_grids.deinit(allocator);
    var video_grids = std.ArrayList(mrope_mod.ImageGrid).empty;
    defer video_grids.deinit(allocator);
    for (images) |im| {
        if (im.grid_h > 0) try image_grids.append(allocator, .{ .t = 1, .h = im.grid_h, .w = im.grid_w });
    }
    for (videos) |vd| {
        try video_grids.append(allocator, .{ .t = vd.grid_t, .h = vd.grid_h, .w = vd.grid_w });
    }
    if (image_grids.items.len == 0 and video_grids.items.len == 0) return .{};

    var ri = mrope_mod.getRopeIndex(allocator, prompt_ids, image_grids.items, video_grids.items, config.image_token_id, config.video_token_id, config.vision_start_token_id, config.qv_merge) catch |err| {
        log.warn("M-RoPE get_rope_index failed ({s}); falling back to scalar RoPE\n", .{@errorName(err)});
        return .{};
    };
    defer ri.deinit();
    const total = prompt_ids.len;
    const flat = try allocator.alloc(i32, 3 * total);
    @memcpy(flat[0..total], ri.pos[0]);
    @memcpy(flat[total .. 2 * total], ri.pos[1]);
    @memcpy(flat[2 * total .. 3 * total], ri.pos[2]);
    log.info("  M-RoPE: {d} images, {d} videos, position ids over {d} tokens, decode delta {d}\n", .{ image_grids.items.len, video_grids.items.len, total, ri.delta });
    return .{ .pos = flat, .total = total, .delta = ri.delta };
}

/// Decode an `input_audio.data` payload into raw float32-LE PCM samples for the
/// Gemma 4 12B unified audio embedder. Accepts a bare base64 string or a
/// `data:audio/...;base64,...` URL. The decoded bytes are interpreted as
/// little-endian float32 mono samples at 16 kHz (the client resamples). Returns
/// null on decode failure or a non-multiple-of-4 byte length.
fn parseAudioContent(allocator: std.mem.Allocator, data: []const u8) ?chat_mod.AudioData {
    const b64 = if (std.mem.indexOf(u8, data, ";base64,")) |sep| data[sep + 8 ..] else data;
    const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(b64) catch return null;
    if (decoded_size == 0 or decoded_size % 4 != 0) return null;
    const raw_buf = allocator.alloc(u8, decoded_size) catch return null;
    std.base64.standard.Decoder.decode(raw_buf, b64) catch {
        allocator.free(raw_buf);
        return null;
    };
    return .{ .samples = raw_buf };
}

/// Decode a JPEG/PNG/GIF/BMP image buffer to float32 CHW pixels, resized to target_size.
/// Uses stb_image for decoding, then nearest-neighbor resize + CHW conversion.
/// Decode an image content URL into preprocessed float32 CHW pixels. Supports:
///   data:image/x-mlx-pixels;base64,... (already-preprocessed float32 CHW)
///   data:image/jpeg|png|webp|...;base64,... (decoded + resized via stb_image / libwebp)
/// Returns null on any decode failure (caller treats as missing image).
/// Derive per-request image preprocessing params from the loaded model config.
pub fn visionPreprocFromConfig(config: *const model_mod.ModelConfig) chat_mod.VisionPreproc {
    if (config.mimo_vision) return .{
        .mode = .mimo,
        .patch = config.qv_patch,
        .tps = config.qv_temporal_patch,
        .merge = config.qv_merge,
        .min_pixels = config.qv_min_pixels,
        .max_pixels = config.qv_max_pixels,
    };
    if (config.dsv41) |c| if (c.vision) |v| return .{
        .mode = .dsv41,
        .patch = v.patch,
        .merge = v.downsample,
        .min_pixels = v.min_pixels,
        .max_tokens = v.max_tokens,
        .max_wh_ratio = v.max_wh_ratio,
    };
    if (config.glm5_vision) return .{
        .mode = .glm5,
        .patch = config.qv_patch,
        .tps = config.qv_temporal_patch,
        .merge = config.qv_merge,
        .min_tokens = config.glmv_min_image_tokens,
        .max_tokens = config.glmv_max_image_tokens,
        .max_video_tokens = config.glmv_max_video_tokens,
    };
    if (!config.qwen_vision) return .{};
    return .{
        .mode = .qwen,
        .patch = config.qv_patch,
        .tps = config.qv_temporal_patch,
        .merge = config.qv_merge,
        .min_pixels = config.qv_min_pixels,
        .max_pixels = config.qv_max_pixels,
    };
}

var vision_pixel_clamp_logged: bool = false;
/// One-shot: the checkpoint declared more pixels than the ViT can attend.
fn logVisionPixelClamp(declared: u32) void {
    if (vision_pixel_clamp_logged) return;
    vision_pixel_clamp_logged = true;
    log.info("[vision] image area capped at {d} px (checkpoint declares {d}): the ViT materializes its full attention\n", .{ vision_common.ENGINE_MAX_PIXELS, declared });
}

test "visionPreprocFromConfig threads each tower's processor bounds" {
    const qwen = visionPreprocFromConfig(&.{
        .qwen_vision = true,
        .qv_patch = 14,
        .qv_temporal_patch = 2,
        .qv_merge = 2,
        .qv_min_pixels = 65536,
        .qv_max_pixels = 16777216,
    });
    try std.testing.expectEqual(.qwen, qwen.mode);
    try std.testing.expectEqual(@as(u32, 14), qwen.patch);
    try std.testing.expectEqual(@as(u32, 2), qwen.tps);
    try std.testing.expectEqual(@as(u32, 2), qwen.merge);
    try std.testing.expectEqual(@as(u32, 65536), qwen.min_pixels);
    try std.testing.expectEqual(@as(u32, 16777216), qwen.max_pixels);

    try std.testing.expectEqual(.gemma, visionPreprocFromConfig(&.{}).mode);

    const mimo = visionPreprocFromConfig(&.{
        .mimo_vision = true,
        .qv_patch = 16,
        .qv_temporal_patch = 2,
        .qv_merge = 2,
        .qv_min_pixels = 8192,
        .qv_max_pixels = 8388608,
    });
    try std.testing.expectEqual(.mimo, mimo.mode);
    try std.testing.expectEqual(@as(u32, 8192), mimo.min_pixels);
    try std.testing.expectEqual(@as(u32, 8388608), mimo.max_pixels);
}

test "an x-mlx-pixels payload is refused by a patch-grid tower (it is a Gemma format)" {
    // Live crash 2026-08-11: the app preprocesses images into Gemma's square
    // `x-mlx-pixels` buffer unless it recognises the arch, so a patch-grid model got
    // one. That buffer carries NO patch grid, so the encode routed to the SigLIP
    // `forward` — whose weights are null sentinels on a patch-grid encoder —
    // and killed the server. The original image is unrecoverable from it (already
    // resized and normalized), so the only honest answer is to refuse the block.
    const url = "data:image/x-mlx-pixels;base64,AAAAAM3MzD3NzEw+mpmZPs3MzD4AAAA/mpkZPzMzMz/NzEw/ZmZmPwAAgD/NzIw/";
    const gemma = parseImageUrlContent(std.testing.allocator, url, .{}) orelse return error.GemmaMustAccept;
    defer std.testing.allocator.free(gemma.pixels);
    try std.testing.expectEqual(@as(u32, 2), gemma.width);
    try std.testing.expectEqual(@as(u32, 0), gemma.grid_h);

    for ([_]chat_mod.VisionPreproc{ .{ .mode = .mimo }, .{ .mode = .qwen } }) |vp| {
        try std.testing.expect(parseImageUrlContent(std.testing.allocator, url, vp) == null);
    }
}

pub fn parseImageUrlContent(allocator: std.mem.Allocator, url: []const u8, vp: chat_mod.VisionPreproc) ?chat_mod.ImageData {
    const sep = std.mem.indexOf(u8, url, ";base64,") orelse return null;
    const b64_data = url[sep + 8 ..];
    const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(b64_data) catch return null;
    const raw_buf = allocator.alloc(u8, decoded_size) catch return null;
    std.base64.standard.Decoder.decode(raw_buf, b64_data) catch {
        allocator.free(raw_buf);
        return null;
    };

    const mime = url[0..sep];
    if (std.mem.eql(u8, mime, "data:image/x-mlx-pixels")) {
        // Already preprocessed float32 CHW pixels — a GEMMA format. It carries
        // no patch grid and the source image cannot be recovered from it, so a
        // patch-grid tower has to refuse rather than hand a gridless buffer to
        // an encoder that has no SigLIP weights.
        if (vp.mode != .gemma) {
            log.warn("Dropping x-mlx-pixels image: {s} needs server-side preprocessing (send the encoded image instead)\n", .{@tagName(vp.mode)});
            allocator.free(raw_buf);
            return null;
        }
        const n_pixels = raw_buf.len / 4;
        const per_channel = n_pixels / 3;
        const side = std.math.sqrt(per_channel);
        if (side * side == per_channel and n_pixels == 3 * side * side) {
            return .{
                .pixels = raw_buf,
                .width = @intCast(side),
                .height = @intCast(side),
            };
        }
        allocator.free(raw_buf);
        return null;
    }

    if (std.mem.startsWith(u8, mime, "data:image/")) {
        // JPEG/PNG/GIF/BMP/WebP — decode + resize + convert to float32 CHW (Gemma) or
        // smart-resized merge-order pixel_values (Qwen3-VL).
        const started = monotonicNs();
        const img = decodeImageToPixels(allocator, raw_buf, vp);
        media_decode_ns += monotonicNs() - started;
        allocator.free(raw_buf);
        return img;
    }

    allocator.free(raw_buf);
    return null;
}

/// The one heap buffer every decoded media entry carries. `ImageData` and
/// `VideoData` spell it `pixels`, `AudioData` spells it `samples`; a fourth
/// modality that spells it something else fails to COMPILE here rather than
/// leaking quietly.
inline fn mediaBuffer(item: anytype) []const u8 {
    return if (@hasField(@TypeOf(item), "pixels")) item.pixels else item.samples;
}

fn isDecodedImageMime(mime: []const u8) bool {
    return std.mem.startsWith(u8, mime, "data:image/") and !std.mem.eql(u8, mime, "data:image/x-mlx-pixels");
}

/// The pixel size of a JPEG/PNG/GIF/BMP/WebP file, read from its header: stb first, as the decode tries them.
fn imageHeaderSize(encoded: []const u8) ?struct { w: u32, h: u32 } {
    var w: c_int = 0;
    var h: c_int = 0;
    var channels: c_int = 0;
    if (stb.stbi_info_from_memory(encoded.ptr, @intCast(encoded.len), &w, &h, &channels) == 0) {
        if (webp.WebPGetInfo(encoded.ptr, encoded.len, &w, &h) == 0) return null;
    }
    if (w <= 0 or h <= 0) return null;
    return .{ .w = @intCast(w), .h = @intCast(h) };
}

/// What a patch-grid tower's processor makes of a `src_w` x `src_h` image: the size it resizes to and its
/// patch grid. `decodeImageToPixels` builds its buffers from the same numbers.
const ImagePlan = struct { width: u32, height: u32, grid_h: u32, grid_w: u32 };

fn planImage(vp: chat_mod.VisionPreproc, src_w: u32, src_h: u32) ?ImagePlan {
    if (vp.mode == .gemma) return null;
    if (vp.mode == .dsv41) {
        const plan = dsv41_vision.planGrid(src_w, src_h, .{ .patch = vp.patch, .downsample = vp.merge, .max_tokens = vp.max_tokens, .min_pixels = vp.min_pixels, .max_wh_ratio = vp.max_wh_ratio }) orelse return null;
        return .{ .width = plan.w, .height = plan.h, .grid_h = plan.vit_h, .grid_w = plan.vit_w };
    }
    const factor = std.math.mul(u32, vp.patch, vp.merge) catch return null;
    if (factor == 0 or vp.tps == 0) return null;
    const rs = imageResize(vp, factor, src_h, src_w);
    return .{ .width = rs.w, .height = rs.h, .grid_h = rs.h / vp.patch, .grid_w = rs.w / vp.patch };
}

/// An image the tower will decode later: its file bytes and the plan the decode reaches, so the request
/// holds the compressed size and the prompt still places the rows of the grid.
fn planEncodedImage(allocator: std.mem.Allocator, b64: []const u8, vp: chat_mod.VisionPreproc) ?chat_mod.ImageData {
    const started = monotonicNs();
    defer media_decode_ns += monotonicNs() - started;
    const size = std.base64.standard.Decoder.calcSizeForSlice(b64) catch return null;
    const raw = allocator.alloc(u8, size) catch return null;
    var kept = false;
    defer if (!kept) allocator.free(raw);
    std.base64.standard.Decoder.decode(raw, b64) catch return null;
    const dims = imageHeaderSize(raw) orelse return null;
    const plan = planImage(vp, dims.w, dims.h) orelse return null;
    kept = true;
    return .{ .pixels = raw, .width = plan.width, .height = plan.height, .grid_h = plan.grid_h, .grid_w = plan.grid_w, .undecoded = true };
}

/// The tower-time decode of an `undecoded` image (`scheduler.vision_image_decode`).
pub fn decodeEncodedImage(allocator: std.mem.Allocator, encoded: []const u8, config: *const model_mod.ModelConfig) ?chat_mod.ImageData {
    return decodeImageToPixels(allocator, encoded, visionPreprocFromConfig(config));
}

/// One `image_url` as the request holds it: a patch-grid tower's JPEG/PNG/GIF/BMP/WebP stays encoded until its
/// tower call, anything else decodes here.
fn imageUrlContent(allocator: std.mem.Allocator, url: []const u8, vp: chat_mod.VisionPreproc) ?chat_mod.ImageData {
    if (vp.mode != .gemma) if (std.mem.indexOf(u8, url, ";base64,")) |sep| {
        if (isDecodedImageMime(url[0..sep])) return planEncodedImage(allocator, url[sep + 8 ..], vp);
    };
    return parseImageUrlContent(allocator, url, vp);
}

/// One modality's decoded entries, grouped per message. Slots are INDICES, not
/// pointers: opening another slot may grow the backing list and move every
/// `ArrayList` header, while the buffers those headers point at stay put.
fn MediaBag(comptime T: type) type {
    return struct {
        const Self = @This();
        lists: std.ArrayList(std.ArrayList(T)) = .empty,

        fn open(self: *Self, allocator: std.mem.Allocator) !usize {
            try self.lists.append(allocator, .empty);
            return self.lists.items.len - 1;
        }

        fn at(self: *Self, slot: usize) *std.ArrayList(T) {
            return &self.lists.items[slot];
        }

        /// Borrowed view for a `chat_mod.Message` field. Null when nothing
        /// decoded, so a message with no media keeps a null field rather than
        /// an empty slice (every downstream reader tests the optional).
        fn slice(self: *Self, slot: usize) ?[]const T {
            const l = self.lists.items[slot];
            return if (l.items.len == 0) null else l.items;
        }

        fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            for (self.lists.items) |*l| {
                for (l.items) |item| allocator.free(mediaBuffer(item));
                l.deinit(allocator);
            }
            self.lists.deinit(allocator);
        }
    };
}

/// Every image / video / audio buffer decoded out of ONE request body, owned in
/// ONE place.
///
/// `chat_mod.Message.images/videos/audio` BORROW these slices — the handler
/// hands the message a view and keeps ownership here — so the `defer
/// messages.deinit(allocator)` every handler already has, which frees the
/// Message array and nothing it points at, can never be the whole story. Both
/// `handleChatCompletions` and the Anthropic `/v1/messages` handler used to
/// hand-roll a local list per message and `toOwnedSlice` it into the Message,
/// which leaked the full decoded CHW buffer of every image (megabytes each) on
/// the SUCCESS path and on every return after the parse loop.
///
/// Ownership is by PROVENANCE: a list only ever comes from `openImages` /
/// `openVideos` / `openAudio`, and is owned from its first append — so a `try`
/// between the decode and the message append cannot leak either, and a new
/// early return cannot leak by omission. A handler takes media lists from here only:
/// a media list of its own is never freed.
pub const RequestMedia = struct {
    allocator: std.mem.Allocator,
    image_bag: MediaBag(chat_mod.ImageData) = .{},
    video_bag: MediaBag(chat_mod.VideoData) = .{},
    audio_bag: MediaBag(chat_mod.AudioData) = .{},
    part_lists: std.ArrayList([]chat_mod.MediaPart) = .empty,
    /// Images the body names anywhere, decoded or not.
    n_images: usize = 0,
    /// The first media part that did not decode.
    fault: ?MediaFault = null,
    /// Whether the loaded model encodes `input_audio`; otherwise the part is refused.
    accepts_audio: bool = false,

    pub fn init(allocator: std.mem.Allocator) RequestMedia {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *RequestMedia) void {
        self.image_bag.deinit(self.allocator);
        self.video_bag.deinit(self.allocator);
        self.audio_bag.deinit(self.allocator);
        for (self.part_lists.items) |l| self.allocator.free(l);
        self.part_lists.deinit(self.allocator);
    }

    /// The body's media refusal, if any; too many images outranks a decode failure.
    pub fn refusal(self: *const RequestMedia) ?MediaFault {
        if (self.n_images > chat_mod.MAX_REQUEST_IMAGES) return imageCountFault(self.n_images);
        return self.fault;
    }

    /// Decode `url` into image slot `slot`. Past the image limit or after a
    /// fault nothing more decodes: the request is refused either way.
    pub fn addImage(self: *RequestMedia, slot: usize, url: ?[]const u8, vp: chat_mod.VisionPreproc, pos: WirePos) bool {
        self.n_images += 1;
        if (self.n_images > chat_mod.MAX_REQUEST_IMAGES or self.fault != null) return false;
        const u = url orelse {
            self.fault = imageDecodeFault(pos, "the part carries no image URL or base64 data");
            return false;
        };
        if (appendImageUrlContent(self.allocator, self.images(slot), u, vp)) return true;
        self.fault = imageDecodeFault(pos, imageRejectReason(u));
        return false;
    }

    pub fn addVideo(self: *RequestMedia, slot: usize, frame_urls: []const []const u8, vp: chat_mod.VisionPreproc, pos: WirePos) bool {
        if (self.fault != null) return false;
        if (appendVideoUrlContent(self.allocator, self.videos(slot), frame_urls, vp)) return true;
        self.fault = MediaFault.init(false, "{s}[{d}]: video could not be decoded: its frames must be base64 image data URLs of one size", .{ pos.list, pos.index });
        return false;
    }

    pub fn addAudio(self: *RequestMedia, slot: usize, data: ?[]const u8, pos: WirePos) void {
        if (self.fault != null) return;
        if (!self.accepts_audio) {
            self.fault = MediaFault.init(false, "{s}[{d}]: input_audio is not supported by this model", .{ pos.list, pos.index });
            return;
        }
        if (data) |d| if (parseAudioContent(self.allocator, d)) |aud| {
            self.audio(slot).append(self.allocator, aud) catch self.allocator.free(aud.samples);
            return;
        };
        self.fault = MediaFault.init(false, "{s}[{d}]: input_audio could not be decoded: data must be base64 float32 PCM", .{ pos.list, pos.index });
    }

    /// Take ownership of a message's placed parts; null when it placed none.
    pub fn ownParts(self: *RequestMedia, list: *std.ArrayList(chat_mod.MediaPart)) !?[]const chat_mod.MediaPart {
        if (list.items.len == 0) return null;
        try self.part_lists.ensureUnusedCapacity(self.allocator, 1);
        const owned = try list.toOwnedSlice(self.allocator);
        self.part_lists.appendAssumeCapacity(owned);
        return owned;
    }

    pub fn openImages(self: *RequestMedia) !usize {
        return self.image_bag.open(self.allocator);
    }
    fn images(self: *RequestMedia, slot: usize) *std.ArrayList(chat_mod.ImageData) {
        return self.image_bag.at(slot);
    }
    pub fn imagesSlice(self: *RequestMedia, slot: usize) ?[]const chat_mod.ImageData {
        return self.image_bag.slice(slot);
    }

    pub fn openVideos(self: *RequestMedia) !usize {
        return self.video_bag.open(self.allocator);
    }
    pub fn videos(self: *RequestMedia, slot: usize) *std.ArrayList(chat_mod.VideoData) {
        return self.video_bag.at(slot);
    }
    pub fn videosSlice(self: *RequestMedia, slot: usize) ?[]const chat_mod.VideoData {
        return self.video_bag.slice(slot);
    }

    pub fn openAudio(self: *RequestMedia) !usize {
        return self.audio_bag.open(self.allocator);
    }
    fn audio(self: *RequestMedia, slot: usize) *std.ArrayList(chat_mod.AudioData) {
        return self.audio_bag.at(slot);
    }
    pub fn audioSlice(self: *RequestMedia, slot: usize) ?[]const chat_mod.AudioData {
        return self.audio_bag.slice(slot);
    }
};

/// Decode one `image_url` into as many entries as the model's processor makes
/// of it. Returns false when nothing could be decoded (a remote
/// URL, bad base64, an unreadable payload) so the caller can refuse by name
/// instead of answering a prompt the image silently fell out of.
pub fn appendImageUrlContent(
    allocator: std.mem.Allocator,
    list: *std.ArrayList(chat_mod.ImageData),
    url: []const u8,
    vp: chat_mod.VisionPreproc,
) bool {
    const before = list.items.len;
    if (imageUrlContent(allocator, url, vp)) |img| {
        list.append(allocator, img) catch allocator.free(img.pixels);
    }
    return list.items.len > before;
}

/// The resample filter belongs to the arch's reference processor; every served
/// patch-grid tower resizes bicubic.
fn resampleFilterFor(vp: chat_mod.VisionPreproc) vision_common.Filter {
    _ = vp;
    return .bicubic;
}

/// A decoded source image as packed RGB8, owned by `allocator`. One owner and
/// one free path — stb and libwebp buffers are copied out and released here, so
/// callers do not juggle three different deallocators.
const DecodedRgb = struct {
    rgb: []u8,
    w: u32,
    h: u32,

    fn deinit(self: DecodedRgb, allocator: std.mem.Allocator) void {
        allocator.free(self.rgb);
    }
};

fn decodeRgbOwned(allocator: std.mem.Allocator, encoded: []const u8) ?DecodedRgb {
    var w: c_int = 0;
    var h: c_int = 0;
    var channels: c_int = 0;
    // stb first (JPEG/PNG/GIF/BMP) — 4 channels so transparency composites onto white.
    if (stb.stbi_load_from_memory(encoded.ptr, @intCast(encoded.len), &w, &h, &channels, 4)) |rgba| {
        defer stb.stbi_image_free(rgba);
        const total_px: usize = @intCast(w * h);
        const rgb = allocator.alloc(u8, total_px * 3) catch return null;
        for (0..total_px) |i| {
            const a = @as(u16, rgba[i * 4 + 3]);
            const inv_a = 255 - a;
            rgb[i * 3 + 0] = @intCast((a * @as(u16, rgba[i * 4 + 0]) + inv_a * 255) / 255);
            rgb[i * 3 + 1] = @intCast((a * @as(u16, rgba[i * 4 + 1]) + inv_a * 255) / 255);
            rgb[i * 3 + 2] = @intCast((a * @as(u16, rgba[i * 4 + 2]) + inv_a * 255) / 255);
        }
        return .{ .rgb = rgb, .w = @intCast(w), .h = @intCast(h) };
    }

    var webp_w: c_int = 0;
    var webp_h: c_int = 0;
    const decoded = webp.WebPDecodeRGB(encoded.ptr, encoded.len, &webp_w, &webp_h) orelse return null;
    defer webp.WebPFree(decoded);
    const total: usize = @as(usize, @intCast(webp_w)) * @as(usize, @intCast(webp_h)) * 3;
    const rgb = allocator.alloc(u8, total) catch return null;
    @memcpy(rgb, decoded[0..total]);
    return .{ .rgb = rgb, .w = @intCast(webp_w), .h = @intCast(webp_h) };
}

/// The pixel size a patch-grid tower's processor resizes an `src_h` x `src_w` image to.
fn imageResize(vp: chat_mod.VisionPreproc, factor: u32, src_h: u32, src_w: u32) vision_common.Resized {
    const bounds = vision_common.effectivePixelBounds(vp.min_pixels, vp.max_pixels);
    if (bounds.clamped and vp.mode == .qwen) logVisionPixelClamp(vp.max_pixels);
    return switch (vp.mode) {
        .mimo => mimo_vision.smartResize(src_h, src_w, factor, bounds.min, bounds.max),
        .glm5 => glm5_vision.smartResize(vp.tps, src_h, src_w, vp.tps, factor, vp.min_tokens, vp.max_tokens),
        else => vision_common.smartResizeImage(src_h, src_w, factor, bounds.min, bounds.max),
    };
}

fn decodeImageToPixels(allocator: std.mem.Allocator, encoded: []const u8, vp: chat_mod.VisionPreproc) ?chat_mod.ImageData {
    const target: u32 = 768; // Gemma 4 default for square images

    const src = decodeRgbOwned(allocator, encoded) orelse return null;
    defer src.deinit(allocator);
    const px = src.rgb.ptr;
    const src_w: u32 = src.w;
    const src_h: u32 = src.h;

    // DeepSeek-V4.1 letterboxes into a canvas of whole patches; the span's size is `dsv41_vision.spanRows`.
    if (vp.mode == .dsv41) {
        const pr: dsv41_vision.Processor = .{ .patch = vp.patch, .downsample = vp.merge, .max_tokens = vp.max_tokens, .min_pixels = vp.min_pixels, .max_wh_ratio = vp.max_wh_ratio };
        const pv = dsv41_vision.preprocess(allocator, px[0 .. @as(usize, src_h) * src_w * 3], src_h, src_w, pr) catch return null;
        log.info("  Decoded {d}x{d} image → deepseek grid {d}x{d} ({d} tokens, canvas {d}x{d})\n", .{ src_w, src_h, pv.grid_h, pv.grid_w, dsv41_vision.spanRows(pv.grid_h, pv.grid_w, vp.merge), pv.w, pv.h });
        return .{ .pixels = pv.bytes, .width = pv.w, .height = pv.h, .grid_h = pv.grid_h, .grid_w = pv.grid_w };
    }

    // Patch-grid towers: smart-resize to a multiple of patch·merge, normalize
    // (x/255−0.5)/0.5 (both processors use mean/std 0.5), then emit that
    // processor's pixel_values. `grid_h/grid_w` carry the full patch grid;
    // token count = (gh/merge)·(gw/merge). Qwen bounds the PIXEL area.
    if (vp.mode != .gemma) {
        const factor = std.math.mul(u32, vp.patch, vp.merge) catch return null;
        const plan = planImage(vp, src_w, src_h) orelse return null;
        const rh = plan.height;
        const rw = plan.width;
        const C: u32 = 3;
        const gh = rh / vp.patch;
        const gw = rw / vp.patch;
        const n: usize = @as(usize, gh) * gw;
        const feat: usize = @as(usize, C) * vp.tps * vp.patch * vp.patch;
        const plane: usize = @as(usize, rh) * rw;

        const chw = allocator.alloc(f32, @as(usize, C) * plane) catch return null;
        defer allocator.free(chw);
        const source_len: usize = @as(usize, src_h) * src_w * C;
        if (vp.mode == .mimo) {
            mimo_vision.resizeNormalizedChw(chw, px[0..source_len], src_h, src_w, rh, rw) catch return null;
        } else if (vp.mode == .glm5) {
            glm5_vision.resizeNormalizedChw(allocator, chw, px[0..source_len], src_h, src_w, rh, rw, vp.tps, vp.tps, factor, vp.min_tokens) catch return null;
        } else vision_common.resizeRgbNormalizedChw(
            allocator,
            chw,
            px[0..source_len],
            src_h,
            src_w,
            rh,
            rw,
            resampleFilterFor(vp),
        ) catch return null;

        const pv_bytes = allocator.alloc(u8, n * feat * 4) catch return null;
        const pv_f32 = @as([*]f32, @ptrCast(@alignCast(pv_bytes.ptr)))[0 .. n * feat];
        switch (vp.mode) {
            else => vision_common.buildPixelValues(pv_f32, chw, C, rh, rw, vp.patch, vp.tps, vp.merge),
        }
        log.info("  Decoded {d}x{d} image → {s} grid {d}x{d} ({d} tokens, resized {d}x{d})\n", .{ src_w, src_h, @tagName(vp.mode), gh, gw, n / (@as(usize, vp.merge) * vp.merge), rw, rh });
        return .{ .pixels = pv_bytes, .width = rw, .height = rh, .grid_h = gh, .grid_w = gw };
    }

    // Allocate float32 CHW output: [3, target, target]
    const out_size = 3 * target * target;
    const out_buf = allocator.alloc(u8, out_size * 4) catch return null;
    const float_buf: [*]f32 = @ptrCast(@alignCast(out_buf.ptr));

    // Bilinear resize + HWC→CHW + rescale to [0,1]
    for (0..target) |ty| {
        for (0..target) |tx| {
            // Map target pixel to source coordinates
            const sx_f: f32 = @as(f32, @floatFromInt(tx)) * @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(target));
            const sy_f: f32 = @as(f32, @floatFromInt(ty)) * @as(f32, @floatFromInt(src_h)) / @as(f32, @floatFromInt(target));

            // Nearest-neighbor for simplicity (bilinear adds complexity for marginal benefit here)
            const sx: u32 = @min(@as(u32, @intFromFloat(sx_f)), src_w - 1);
            const sy: u32 = @min(@as(u32, @intFromFloat(sy_f)), src_h - 1);

            const src_idx = (sy * src_w + sx) * 3;
            const dst_idx = ty * target + tx;

            // CHW: channel 0 (R), channel 1 (G), channel 2 (B)
            float_buf[0 * target * target + dst_idx] = @as(f32, @floatFromInt(px[src_idx + 0])) / 255.0;
            float_buf[1 * target * target + dst_idx] = @as(f32, @floatFromInt(px[src_idx + 1])) / 255.0;
            float_buf[2 * target * target + dst_idx] = @as(f32, @floatFromInt(px[src_idx + 2])) / 255.0;
        }
    }

    log.info("  Decoded {d}x{d} image → {d}x{d} float32 CHW\n", .{ src_w, src_h, target, target });
    return .{ .pixels = out_buf, .width = target, .height = target };
}

/// Decode a `video_url` block's `frames` array — already-decoded-by-the-client
/// JPEG/PNG data URLs, one per sampled frame; no video codec exists anywhere in
/// this codebase, so frame extraction is the client's job — into ONE
/// `chat_mod.VideoData`. All frames share ONE smart-resize target, computed
/// from the FIRST frame and applied to every frame (a video's whole patch grid
/// must be identical across frames), then grouped into `vp.tps`-sized temporal-
/// patch groups — the last group pads by repeating its final frame, matching
/// HF's video processor. Qwen-only: the only family declaring `video_token_id`.
fn decodeVideoUrlContent(allocator: std.mem.Allocator, frame_urls: []const []const u8, vp: chat_mod.VisionPreproc) ?chat_mod.VideoData {
    if ((vp.mode != .qwen and vp.mode != .glm5) or frame_urls.len == 0) return null;
    const factor = std.math.mul(u32, vp.patch, vp.merge) catch return null;
    if (factor == 0 or vp.tps == 0 or vp.tps > 8) return null;

    // Decode every frame to RGB8 first — frame 0's natural size decides the
    // resize target every later frame must also resize to.
    var decoded = std.ArrayList(DecodedRgb).empty;
    defer {
        for (decoded.items) |d| d.deinit(allocator);
        decoded.deinit(allocator);
    }
    for (frame_urls) |url| {
        const sep = std.mem.indexOf(u8, url, ";base64,") orelse return null;
        const b64 = url[sep + 8 ..];
        const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(b64) catch return null;
        const raw = allocator.alloc(u8, decoded_size) catch return null;
        defer allocator.free(raw);
        std.base64.standard.Decoder.decode(raw, b64) catch return null;
        const rgb = decodeRgbOwned(allocator, raw) orelse return null;
        decoded.append(allocator, rgb) catch {
            rgb.deinit(allocator);
            return null;
        };
    }

    const bounds = vision_common.effectivePixelBounds(vp.min_pixels, vp.max_pixels);
    const min_pixels = bounds.min;
    const max_pixels = bounds.max;
    if (bounds.clamped and vp.mode == .qwen) logVisionPixelClamp(vp.max_pixels);
    const first = decoded.items[0];
    if (vp.mode == .glm5 and @as(u64, (frame_urls.len + vp.tps - 1) / vp.tps) > vp.max_video_tokens) return null;
    const rs = if (vp.mode == .glm5)
        glm5_vision.smartResize(@intCast(frame_urls.len), first.h, first.w, vp.tps, factor, vp.min_tokens, vp.max_video_tokens)
    else
        vision_common.smartResizeImage(first.h, first.w, factor, min_pixels, max_pixels);
    const rh = rs.h;
    const rw = rs.w;
    const C: u32 = 3;
    const gh = rh / vp.patch;
    const gw = rw / vp.patch;
    const n_per_group: usize = @as(usize, gh) * gw;
    const feat: usize = @as(usize, C) * vp.tps * vp.patch * vp.patch;

    // Resize every frame to the shared (rh, rw) target.
    var frames_chw = std.ArrayList([]f32).empty;
    defer {
        for (frames_chw.items) |f| allocator.free(f);
        frames_chw.deinit(allocator);
    }
    for (decoded.items) |d| {
        const source_len: usize = @as(usize, d.h) * d.w * C;
        const chw = allocator.alloc(f32, @as(usize, C) * rh * rw) catch return null;
        const resize = if (vp.mode == .glm5)
            glm5_vision.resizeNormalizedChw(allocator, chw, d.rgb[0..source_len], d.h, d.w, rh, rw, @intCast(frame_urls.len), vp.tps, factor, vp.min_tokens)
        else
            vision_common.resizeRgbNormalizedChw(allocator, chw, d.rgb[0..source_len], d.h, d.w, rh, rw, resampleFilterFor(vp));
        resize catch {
            allocator.free(chw);
            return null;
        };
        frames_chw.append(allocator, chw) catch {
            allocator.free(chw);
            return null;
        };
    }

    // Group into tps-sized temporal patches, padding the last group by
    // repeating its final frame.
    const grid_t: usize = (frames_chw.items.len + vp.tps - 1) / vp.tps;
    const pv_bytes = allocator.alloc(u8, grid_t * n_per_group * feat * 4) catch return null;
    const pv_f32 = @as([*]f32, @ptrCast(@alignCast(pv_bytes.ptr)))[0 .. grid_t * n_per_group * feat];

    var group_frames: [8][]const f32 = undefined;
    var g: usize = 0;
    while (g < grid_t) : (g += 1) {
        var k: usize = 0;
        while (k < vp.tps) : (k += 1) {
            const idx = @min(g * vp.tps + k, frames_chw.items.len - 1);
            group_frames[k] = frames_chw.items[idx];
        }
        const out_slice = pv_f32[g * n_per_group * feat ..][0 .. n_per_group * feat];
        vision_common.buildPixelValuesVideo(out_slice, group_frames[0..vp.tps], C, rh, rw, vp.patch, vp.merge);
    }

    log.info("  Decoded {d} frames → {s} video grid_t={d} grid {d}x{d} ({d} tokens, resized {d}x{d})\n", .{
        frame_urls.len, @tagName(vp.mode), grid_t, gh, gw, grid_t * n_per_group / (@as(usize, vp.merge) * vp.merge), rw, rh,
    });
    return .{ .pixels = pv_bytes, .grid_t = @intCast(grid_t), .grid_h = gh, .grid_w = gw };
}

/// Decode a `video_url` block's `frames` array into `list`, mirroring
/// `appendImageUrlContent`'s append-or-report shape.
pub fn appendVideoUrlContent(
    allocator: std.mem.Allocator,
    list: *std.ArrayList(chat_mod.VideoData),
    frame_urls: []const []const u8,
    vp: chat_mod.VisionPreproc,
) bool {
    const vid = decodeVideoUrlContent(allocator, frame_urls, vp) orelse return false;
    list.append(allocator, vid) catch {
        allocator.free(vid.pixels);
        return false;
    };
    return true;
}

test "RequestMedia frees every decoded buffer it was handed" {
    // std.testing.allocator fails this test on a leak, which is the whole
    // point: the owner is the only thing that frees `pixels`/`samples`, so a
    // gutted `deinit` is caught here rather than in a week of RSS growth.
    const a = std.testing.allocator;
    var media = RequestMedia.init(a);

    const img_slot = try media.openImages();
    try media.images(img_slot).append(a, .{
        .pixels = try a.alloc(u8, 4096),
        .width = 32,
        .height = 32,
    });
    try media.images(img_slot).append(a, .{
        .pixels = try a.alloc(u8, 2048),
        .width = 16,
        .height = 32,
    });

    const vid_slot = try media.openVideos();
    try media.videos(vid_slot).append(a, .{
        .pixels = try a.alloc(u8, 1024),
        .grid_t = 1,
        .grid_h = 2,
        .grid_w = 2,
    });

    const aud_slot = try media.openAudio();
    try media.audio(aud_slot).append(a, .{ .samples = try a.alloc(u8, 512) });

    // A borrowed slice for `Message.images`; empty slots read back as null so a
    // message with no media keeps a null field rather than an empty slice.
    try std.testing.expectEqual(@as(usize, 2), (media.imagesSlice(img_slot) orelse return error.NoImages).len);
    try std.testing.expectEqual(@as(usize, 1), (media.videosSlice(vid_slot) orelse return error.NoVideos).len);
    try std.testing.expectEqual(@as(usize, 1), (media.audioSlice(aud_slot) orelse return error.NoAudio).len);
    const empty = try media.openImages();
    try std.testing.expect(media.imagesSlice(empty) == null);

    media.deinit();
}

test "readOpenAiMessages: every message's media decodes and sits at its content offset" {
    // An agent turn: the human's screenshot two turns back, then a tool result
    // carrying a new one. Both reach the prompt, each where it was sent.
    var t = try testWire("{\"messages\":[" ++
        "{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64," ++ test_png_red ++ "\"}},{\"type\":\"text\",\"text\":\"look\"}]}," ++
        "{\"role\":\"assistant\",\"content\":\"seen\"}," ++
        "{\"role\":\"user\",\"content\":\"take a screenshot\"}," ++
        "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"screenshot\",\"arguments\":\"{}\"}}]}," ++
        "{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":[{\"type\":\"text\",\"text\":\"shot:\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64," ++ test_png_blue ++ "\"}}]}" ++
        "]}", .openai);
    defer t.parsed.deinit();
    defer t.wire.deinit();
    try std.testing.expect(t.wire.media.refusal() == null);
    const msgs = t.wire.messages.items;
    try std.testing.expectEqual(@as(usize, 5), msgs.len);
    try std.testing.expectEqualStrings("look", msgs[0].content);
    try std.testing.expectEqual(@as(usize, 1), msgs[0].images.?.len);
    try std.testing.expectEqualSlices(chat_mod.MediaPart, &.{.{ .at = 0, .kind = .image }}, msgs[0].media_parts.?);
    try std.testing.expectEqualStrings("tool", msgs[4].role);
    try std.testing.expectEqualStrings("shot:", msgs[4].content);
    try std.testing.expectEqual(@as(usize, 1), msgs[4].images.?.len);
    try std.testing.expectEqualSlices(chat_mod.MediaPart, &.{.{ .at = 5, .kind = .image }}, msgs[4].media_parts.?);
    try std.testing.expect(!std.mem.eql(u8, msgs[0].images.?[0].pixels, msgs[4].images.?[0].pixels));

    var flat = try flattenMedia(std.testing.allocator, msgs);
    defer flat.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 2), flat.images.len);
    try std.testing.expectEqual(msgs[0].images.?[0].pixels.ptr, flat.images[0].pixels.ptr);
}

test "an input_audio part sent to a vision-only model is refused by name" {
    // In the walk, where the part is named...
    var t = try testWire("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"listen\"},{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"AAAAAA==\",\"format\":\"wav\"}}]}]}", .openai);
    defer t.parsed.deinit();
    defer t.wire.deinit();
    try std.testing.expectEqualStrings("messages[1]: input_audio is not supported by this model", t.wire.media.refusal().?.text());
    // ...and at placement, for decoded audio a vision tower cannot encode.
    const audio = [_]chat_mod.AudioData{.{ .samples = "\x00\x00\x00\x00" }};
    const msgs = [_]chat_mod.Message{.{ .role = "user", .content = "listen", .audio = &audio }};
    var flat = try flattenMedia(std.testing.allocator, &msgs);
    defer flat.deinit(std.testing.allocator);
    const fault = mediaKindRefusal(&msgs, flat, true, false) orelse return error.TestExpectedRefusal;
    try std.testing.expect(std.mem.startsWith(u8, fault.text(), "input_audio is not supported by this model"));
    try std.testing.expect(std.mem.indexOf(u8, mediaKindRefusal(&msgs, flat, false, false).?.text(), "audio") != null);
}

fn testTowerConfigs() [3]model_mod.ModelConfig {
    return .{
        // Qwen3.8 packs: no processor config, so the 1,003,520 px default.
        .{ .has_vision = true, .qwen_vision = true, .qv_heads = 16, .qv_hidden = 1152, .qv_intermediate = 4304, .qv_out_hidden = 2560, .qv_patch = 16, .qv_temporal_patch = 2, .qv_merge = 2 },
        .{ .has_vision = true, .mimo_vision = true, .qv_heads = 32, .mvit_kv_heads = 8, .qv_head_dim = 64, .qv_hidden = 1280, .qv_intermediate = 4608, .qv_out_hidden = 4096, .qv_patch = 16, .qv_temporal_patch = 2, .qv_merge = 2, .qv_min_pixels = 8192, .qv_max_pixels = 8388608 },
        .{ .has_vision = true, .glm5_vision = true, .qv_heads = 16, .qv_head_dim = 64, .qv_hidden = 1024, .qv_intermediate = 4096, .qv_out_hidden = 4096, .glmv_projection_intermediate = 10240, .qv_patch = 14, .qv_temporal_patch = 2, .qv_merge = 2, .glmv_min_image_tokens = 16, .glmv_max_image_tokens = 8000 },
    };
}

test "the largest-image encode bill covers any single image the tower's processor admits" {
    const t = std.testing;
    const sides = [_][2]u32{ .{ 1080, 1920 }, .{ 20000, 20000 }, .{ 30000, 64 }, .{ 64, 30000 }, .{ 4321, 3210 }, .{ 7, 9 } };
    for (testTowerConfigs()) |config| {
        const reserve = largestImageEncodeBytes(&config);
        const vp = visionPreprocFromConfig(&config);
        var largest: u64 = 0;
        for (sides) |s| {
            const rs = imageResize(vp, vp.patch * vp.merge, s[0], s[1]);
            const img = [_]chat_mod.ImageData{testGridImage(rs.h / vp.patch, rs.w / vp.patch, "")};
            const bill = visionEncodeBill(&config, &img, &.{}, mediaRows(1, img[0].grid_h, img[0].grid_w, vp.merge)).bytes;
            try t.expect(bill <= reserve);
            largest = @max(largest, bill);
        }
        try t.expect(largest * 10 >= reserve * 9);
    }
}

test "an image request bills its encode the same on a streamed load as on a resident one" {
    const t = std.testing;
    const img = [_]chat_mod.ImageData{testGridImage(46, 82, "")};
    for (testTowerConfigs()) |resident| {
        var streamed = resident;
        streamed.expert_streaming = true;
        streamed.expert_vision_tower_bytes = 1 << 30;
        streamed.expert_vision_encode_bytes = largestImageEncodeBytes(&resident);
        const bill = visionEncodeBill(&resident, &img, &.{}, 943);
        try t.expectEqual(bill.bytes, visionEncodeBill(&streamed, &img, &.{}, 943).bytes);
        try t.expect(towerFitFault(&streamed, &img, &.{}, 943, bill.bytes) == null);
        try t.expect(towerFitFault(&streamed, &img, &.{}, 943, bill.bytes - 1) != null);
    }
}

test "towerFitFault admits an image at the 1536^2 cap when 9 GB is free" {
    var config = model_mod.ModelConfig{};
    config.qv_heads = 16;
    config.qv_hidden = 1152;
    config.qv_intermediate = 4304;
    config.qv_out_hidden = 2560;
    // Measured per-block peak at 96x96 patches: 6.1 GB (`qwen vision ubench`).
    const cap = [_]chat_mod.ImageData{testGridImage(96, 96, "")};
    try std.testing.expect(towerFitFault(&config, &cap, &.{}, 2304, 9_000_000_000) == null);
}

test "towerFitFault bills a MiMo image by the MiMo-ViT's measured scratch, not score sheets" {
    var config = model_mod.ModelConfig{};
    config.mimo_vision = true;
    config.qv_heads = 32;
    config.mvit_kv_heads = 8;
    config.qv_head_dim = 64;
    config.qv_hidden = 1280;
    config.qv_intermediate = 4608;
    config.qv_out_hidden = 4096;
    // The engine's 1536^2 cap: 96x96 patches. N^2 score sheets would bill ~33 GB.
    const cap = [_]chat_mod.ImageData{testGridImage(96, 96, "")};
    const need = mimo_vision.encodeScratchBytes(&config, 9216);
    try std.testing.expect(need < 3 * (1 << 30));
    try std.testing.expect(towerFitFault(&config, &cap, &.{}, 2304, 3 * (1 << 30)) == null);
    try std.testing.expect(towerFitFault(&config, &cap, &.{}, 2304, need / 2) != null);
}

test "towerFitFault refuses an encode whose scratch does not fit, naming both sizes" {
    var config = model_mod.ModelConfig{};
    config.qv_heads = 16;
    config.qv_hidden = 1152;
    config.qv_intermediate = 4304;
    config.qv_out_hidden = 2560;
    // ~8100 patches is a 1920x1080 screenshot at the 1536^2 cap: one block's f32 score sheet is 4.2 GB.
    const shot = [_]chat_mod.ImageData{ testGridImage(90, 90, ""), testGridImage(8, 8, "") };
    const need = visionScratchBytes(&config, 8100);
    try std.testing.expect(need > 4 * (1 << 30));
    const fault = towerFitFault(&config, &shot, &.{}, 2041, need / 2) orelse return error.TestExpectedRefusal;
    try std.testing.expect(std.mem.indexOf(u8, fault.text(), "8100 patches") != null);
    try std.testing.expect(!fault.server_error);
    try std.testing.expect(towerFitFault(&config, &shot, &.{}, 2041, need * 2) == null);
    // A tower this check does not model is not refused by it.
    config.qv_heads = 0;
    try std.testing.expect(towerFitFault(&config, &shot, &.{}, 2041, 0) == null);
}

test "tokenize cache keyFor: a media part's position keys the render, its pixels do not" {
    // The template renders a placeholder per part; the pixels never reach
    // the rendered text, so two screenshots at the same spot share a prompt.
    const a = [_]chat_mod.ImageData{.{ .pixels = "a", .width = 8, .height = 8 }};
    const b = [_]chat_mod.ImageData{.{ .pixels = "b", .width = 8, .height = 8 }};
    const lead = [_]chat_mod.MediaPart{.{ .at = 0, .kind = .image }};
    const tail = [_]chat_mod.MediaPart{.{ .at = 15, .kind = .image }};
    const with_a = [_]chat_mod.Message{.{ .role = "user", .content = "what's in this?", .images = &a, .media_parts = &lead }};
    const with_b = [_]chat_mod.Message{.{ .role = "user", .content = "what's in this?", .images = &b, .media_parts = &lead }};
    const moved = [_]chat_mod.Message{.{ .role = "user", .content = "what's in this?", .images = &a, .media_parts = &tail }};
    const text = [_]chat_mod.Message{.{ .role = "user", .content = "what's in this?" }};
    const k_a = tokenize_cache_mod.TokenizeCache.keyFor(&with_a, null, null, false, null, false, null).?;
    try std.testing.expectEqual(k_a, tokenize_cache_mod.TokenizeCache.keyFor(&with_b, null, null, false, null, false, null).?);
    try std.testing.expect(k_a != tokenize_cache_mod.TokenizeCache.keyFor(&moved, null, null, false, null, false, null).?);
    try std.testing.expect(k_a != tokenize_cache_mod.TokenizeCache.keyFor(&text, null, null, false, null, false, null).?);
}

fn testGridImage(grid_h: u32, grid_w: u32, pixels: []const u8) chat_mod.ImageData {
    return .{ .pixels = pixels, .width = grid_w * 16, .height = grid_h * 16, .grid_h = grid_h, .grid_w = grid_w };
}

test "placeMedia expands each placeholder to its block's rows, in prompt order" {
    var config = model_mod.ModelConfig{};
    config.qwen_vision = true;
    config.qv_merge = 2;
    config.image_token_id = 9;
    config.video_token_id = 8;
    // <vs> image <ve> text <vs> video <ve> text <vs> image <ve>
    const ids = [_]u32{ 1, 7, 9, 6, 2, 7, 8, 6, 3, 7, 9, 6 };
    const images = [_]chat_mod.ImageData{ testGridImage(4, 4, "a"), testGridImage(2, 6, "b") };
    const videos = [_]chat_mod.VideoData{.{ .pixels = "v", .grid_t = 2, .grid_h = 2, .grid_w = 2 }};
    const found = countMediaPlaceholders(&ids, &config);
    try std.testing.expectEqual(@as(usize, 2), found.images);
    try std.testing.expectEqual(@as(usize, 1), found.videos);
    var placed = try placeMedia(std.testing.allocator, &ids, &config, &images, &videos, null);
    defer placed.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(u32, &.{ 1, 7, 9, 9, 9, 9, 6, 2, 7, 8, 8, 6, 3, 7, 9, 9, 9, 6 }, placed.ids);
    try std.testing.expectEqual(@as(usize, 9), placed.rows);
    try std.testing.expectEqualSlices(PlacedMedia, &.{
        .{ .kind = .image, .start = 2, .rows = 4 },
        .{ .kind = .video, .start = 9, .rows = 2 },
        .{ .kind = .image, .start = 14, .rows = 3 },
    }, placed.items);
}

test "deepseek_v41: the image token spelled in a text-only prompt is text, with no placeholder check" {
    // The released tokenizer splits `<｜deepseek_image｜>` out of any text (an AGENTS.md that names it);
    // only attached media make an id a placeholder, and this arch serves none.
    var cfg = model_mod.ModelConfig{ .model_type = "deepseek_v41" };
    cfg.image_token_id = 129264;
    var entry = model_registry_mod.LoadedModel{
        .allocator = std.testing.allocator,
        .id = "m",
        .path = "",
        .bytes_on_disk = 0,
        .arch_hint = "deepseek_v41",
        .config = &cfg,
        .weights = null,
        .transformer = null,
        .tokenizer = null,
        .chat_config = null,
        .vision_encoder = null,
        .drafter_path = "",
        .drafter_block_size = 0,
        .prefix_cache = null,
        .refcount = std.atomic.Value(u32).init(0),
        .last_used_ns = 0,
        .bytes_resident = 0,
        .state = .ready,
        .error_name = null,
    };
    const messages = [_]chat_mod.Message{.{ .role = "user", .content = "the token <｜deepseek_image｜> is reserved" }};
    var ids = try std.testing.allocator.dupe(u32, &.{ 0, 128803, 5, 129264, 6, 128804 });
    defer std.testing.allocator.free(ids);
    const before = ids;
    switch (try prepareRequestMedia(std.testing.allocator, &entry, &messages, &ids, 1 << 20)) {
        .ready => |ready| {
            var r = ready;
            defer r.deinit(std.testing.allocator);
        },
        .refused => return error.TextOnlyPromptRefused,
    }
    try std.testing.expectEqualSlices(u32, &.{ 0, 128803, 5, 129264, 6, 128804 }, ids);
    try std.testing.expectEqual(before.ptr, ids.ptr);
    const found = countMediaPlaceholders(ids, &cfg);
    try std.testing.expectEqual(@as(usize, 1), found.images);
}

test "mediaChain keys every block by its pixels, its position and the blocks before it" {
    const a = testGridImage(2, 2, "pixels-a");
    const b = testGridImage(2, 2, "pixels-b");
    const chainOf = struct {
        fn f(placed: []const PlacedMedia, images: []const chat_mod.ImageData) ![]prefix_cache_mod.MediaSpan {
            return mediaChain(std.testing.allocator, placed, images, &.{}, null);
        }
    }.f;
    const one = try chainOf(&.{.{ .kind = .image, .start = 3, .rows = 1 }}, &.{a});
    defer std.testing.allocator.free(one);
    const two = try chainOf(&.{ .{ .kind = .image, .start = 3, .rows = 1 }, .{ .kind = .image, .start = 10, .rows = 1 } }, &.{ a, b });
    defer std.testing.allocator.free(two);
    const moved = try chainOf(&.{.{ .kind = .image, .start = 4, .rows = 1 }}, &.{a});
    defer std.testing.allocator.free(moved);
    const other = try chainOf(&.{.{ .kind = .image, .start = 3, .rows = 1 }}, &.{b});
    defer std.testing.allocator.free(other);
    const swapped = try chainOf(&.{ .{ .kind = .image, .start = 3, .rows = 1 }, .{ .kind = .image, .start = 10, .rows = 1 } }, &.{ b, a });
    defer std.testing.allocator.free(swapped);

    // Appending a block keeps the earlier keys: the prefix cache matches on them.
    try std.testing.expectEqual(one[0], two[0]);
    try std.testing.expectEqual(@as(usize, 10), two[1].start);
    try std.testing.expect(two[1].key != one[0].key);
    try std.testing.expect(moved[0].key != one[0].key);
    try std.testing.expect(other[0].key != one[0].key);
    try std.testing.expect(swapped[1].key != two[1].key);
    for ([_][]const prefix_cache_mod.MediaSpan{ one, two, moved, other, swapped }) |chain| {
        for (chain) |span| try std.testing.expect(span.key != 0);
    }
}

test "parseAudioContent decodes base64 float32 PCM and rejects bad lengths" {
    // 2 float32 samples = 8 bytes. base64 of 8 zero bytes = "AAAAAAAAAAA=".
    const eight_zeros = "AAAAAAAAAAA=";
    const aud = parseAudioContent(testing.allocator, eight_zeros) orelse return error.TestUnexpectedNull;
    defer testing.allocator.free(aud.samples);
    try testing.expectEqual(@as(usize, 8), aud.samples.len);

    // A data-URL prefix is tolerated.
    const with_prefix = "data:audio/x-mlx-pcm;base64," ++ eight_zeros;
    const aud2 = parseAudioContent(testing.allocator, with_prefix) orelse return error.TestUnexpectedNull;
    defer testing.allocator.free(aud2.samples);
    try testing.expectEqual(@as(usize, 8), aud2.samples.len);

    // 6 decoded bytes is not a whole number of float32s → rejected.
    try testing.expect(parseAudioContent(testing.allocator, "AAAAAAAA") == null);
}

test "video admission bills the largest temporal group" {
    const config = model_mod.ModelConfig{ .qv_heads = 16, .qv_hidden = 1152, .qv_intermediate = 4304, .qv_out_hidden = 2560 };
    const videos = [_]chat_mod.VideoData{.{ .pixels = &.{}, .grid_t = 8, .grid_h = 46, .grid_w = 82 }};
    const patches: u64 = 46 * 82;
    const rows = 8 * patches / 4;
    const old_bill = visionScratchBytes(&config, 8 * patches) + rows * config.qv_out_hidden * 2;
    const floor = visionScratchBytes(&config, patches) + 8 * patches * 3 * 2 * 16 * 16 * 4 + rows * config.qv_out_hidden * 2;
    const bill = visionEncodeBill(&config, &.{}, &videos, rows);
    try std.testing.expectEqual(patches, bill.largest_group_patches);
    try std.testing.expect(bill.bytes < old_bill / 8);
    try std.testing.expect(bill.bytes >= floor);
    try std.testing.expect(towerFitFault(&config, &.{}, &videos, rows, bill.bytes) == null);
    try std.testing.expect(towerFitFault(&config, &.{}, &videos, rows, bill.bytes - 1) != null);
    try std.testing.expect(towerFitFault(&config, &.{}, &videos, rows, old_bill / 8) == null);
    try std.testing.expect(towerFitFault(&config, &.{}, &videos, rows, floor - 1) != null);
}

test "visionEncodeBill covers each measured video peak by >= 25% and within 1.5x" {
    const config = model_mod.ModelConfig{ .qv_heads = 16, .qv_hidden = 1152, .qv_intermediate = 4304, .qv_out_hidden = 2560 };
    // Peak from pixel upload to evaluated output, bytes (`qwen vision ubench`, Sushi-3bpw tower).
    const measured = [_]struct { t: u32, h: u32, w: u32, peak: u64 }{
        .{ .t = 1, .h = 46, .w = 82, .peak = 1_248_628_098 },
        .{ .t = 2, .h = 46, .w = 82, .peak = 1_276_628_354 },
        .{ .t = 4, .h = 46, .w = 82, .peak = 1_332_645_250 },
        .{ .t = 8, .h = 46, .w = 82, .peak = 1_444_679_042 },
        .{ .t = 2, .h = 24, .w = 42, .peak = 172_203_426 },
        .{ .t = 8, .h = 24, .w = 42, .peak = 217_128_354 },
        .{ .t = 2, .h = 96, .w = 96, .peak = 6_237_093_916 },
        .{ .t = 8, .h = 96, .w = 96, .peak = 6_647_611_420 },
    };
    for (measured) |m| {
        const video = [_]chat_mod.VideoData{.{ .pixels = &.{}, .grid_t = m.t, .grid_h = m.h, .grid_w = m.w }};
        const bill = visionEncodeBill(&config, &.{}, &video, m.t * m.h * m.w / 4).bytes;
        try std.testing.expect(bill * 4 >= m.peak * 5);
        try std.testing.expect(bill * 2 <= m.peak * 3);
    }
}

test "GLM vision serving processor carries image and video token budgets without M-RoPE" {
    const cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const processor = visionPreprocFromConfig(&cfg);
    try std.testing.expectEqual(.glm5, processor.mode);
    try std.testing.expectEqual(@as(u32, 14), processor.patch);
    try std.testing.expectEqual(@as(u32, 2), processor.tps);
    try std.testing.expectEqual(@as(u32, 16), processor.min_tokens);
    try std.testing.expectEqual(@as(u32, 8000), processor.max_tokens);
    try std.testing.expectEqual(@as(u32, 240000), processor.max_video_tokens);
    try std.testing.expect(!cfg.qwen_vision);
    const images = [_]chat_mod.ImageData{.{ .pixels = &.{}, .width = 56, .height = 56, .grid_h = 4, .grid_w = 4 }};
    const videos = [_]chat_mod.VideoData{.{ .pixels = &.{}, .grid_t = 2, .grid_h = 4, .grid_w = 4, .fps = 2 }};
    const ids = [_]u32{ 154830, 154854, 154831, 154832, 154855, 154833 };
    const counts = countMediaPlaceholders(&ids, &cfg);
    try std.testing.expectEqual(@as(usize, 1), counts.images);
    try std.testing.expectEqual(@as(usize, 1), counts.videos);
    var tok = Tokenizer.initEmptyForTests(std.testing.allocator);
    defer tok.deinit();
    try tok.special_tokens.put(try std.testing.allocator.dupe(u8, "0.0 seconds"), 1000);
    try tok.special_tokens.put(try std.testing.allocator.dupe(u8, "1.0 seconds"), 1001);
    var placed = try placeMedia(std.testing.allocator, &ids, &cfg, &images, &videos, &tok);
    defer placed.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 12), placed.rows);
    try std.testing.expectEqualSlices(u32, &.{ 154830, 154854, 154854, 154854, 154854, 154831, 154832, 154830, 154854, 154854, 154854, 154854, 154831, 1000, 154830, 154854, 154854, 154854, 154854, 154831, 1001, 154833 }, placed.ids);
    const bill = visionEncodeBill(&cfg, &images, &videos, placed.rows);
    try std.testing.expectEqual(@as(u64, 16), bill.largest_group_patches);
    const input_bytes: u64 = 3 * 16 * 3 * 2 * 14 * 14 * 4;
    const output_bytes: u64 = 12 * 4096 * 2;
    try std.testing.expectEqual(glm5_vision.encodeScratchBytes(&cfg, 16) + input_bytes + 3 * output_bytes, bill.bytes);
}

test "DeepSeek vision serving: the processor carries the release's budgets, a placeholder expands to the whole span, and its encode is billed" {
    const a = std.testing.allocator;
    const cfg = try model_mod.parseConfigFromJson(a, dsv41_config.release_config);
    const vp = visionPreprocFromConfig(&cfg);
    try std.testing.expectEqual(.dsv41, vp.mode);
    try std.testing.expectEqual(@as(u32, 14), vp.patch);
    try std.testing.expectEqual(@as(u32, 3), vp.merge);
    try std.testing.expectEqual(@as(u32, 1024), vp.max_tokens);
    try std.testing.expectEqual(@as(u32, 295936), vp.min_pixels);
    try std.testing.expect(cfg.has_vision and !cfg.qwen_vision);

    // 520x390 plans a 34x45 patch grid: 12x15 aligner cells, a newline per row, start and end.
    const images = [_]chat_mod.ImageData{testGridImage(34, 45, "")};
    const ids = [_]u32{ 1, cfg.image_token_id, 2 };
    const counts = countMediaPlaceholders(&ids, &cfg);
    try std.testing.expectEqual(@as(usize, 1), counts.images);
    var placed = try placeMedia(a, &ids, &cfg, &images, &.{}, null);
    defer placed.deinit(a);
    try std.testing.expectEqual(@as(usize, 194), placed.rows);
    try std.testing.expectEqual(@as(usize, 196), placed.ids.len);
    for (placed.ids[1..195]) |t| try std.testing.expectEqual(cfg.image_token_id, t);
    try std.testing.expectEqual(@as(usize, 1), placed.items.len);
    try std.testing.expectEqual(@as(usize, 1), placed.items[0].start);

    const bill = visionEncodeBill(&cfg, &images, &.{}, placed.rows);
    try std.testing.expectEqual(@as(u64, 1530), bill.largest_group_patches);
    const input_bytes: u64 = 1530 * 588 * 4;
    const output_bytes: u64 = 194 * 5120 * 2;
    try std.testing.expectEqual(dsv41_vision.encodeScratchBytes(&cfg, 1530) + input_bytes + 3 * output_bytes, bill.bytes);

    // The reserve a streamed load keeps covers the largest image the processor admits.
    const reserve = largestImageEncodeBytes(&cfg);
    const proc = dsv41_vision.Processor.of(cfg.dsv41.?.vision.?);
    var largest: u64 = 0;
    for ([_][2]u32{ .{ 1920, 1080 }, .{ 4000, 3000 }, .{ 7000, 1000 }, .{ 1000, 7000 }, .{ 640, 640 }, .{ 14, 14 } }) |s| {
        const plan = dsv41_vision.planGrid(s[0], s[1], proc).?;
        const img = [_]chat_mod.ImageData{testGridImage(plan.vit_h, plan.vit_w, "")};
        const rows = dsv41_vision.spanTokens(plan.llm_h, plan.llm_w);
        const one = visionEncodeBill(&cfg, &img, &.{}, rows).bytes;
        try std.testing.expect(one <= reserve);
        largest = @max(largest, one);
    }
    try std.testing.expect(largest * 10 >= reserve * 8);
}

test "DeepSeek's image-encode bill, which is also its transient cap, covers a quarter over the measured peak" {
    const cfg = try model_mod.parseConfigFromJson(std.testing.allocator, dsv41_config.release_config);
    // One 1031-token image of 8463 patches peaked at 1.111 GiB of MLX's own bytes (binary 26a28040, budget 108).
    const images = [_]chat_mod.ImageData{testGridImage(91, 93, "")};
    // Less the 0.052 GiB a decode forward holds before the encode starts: the cap is over that level.
    const added: u64 = 1_137_000_000;
    try std.testing.expect(visionEncodeBill(&cfg, &images, &.{}, 1031).bytes >= added * 5 / 4);
    try std.testing.expect(largestImageEncodeBytes(&cfg) >= added * 5 / 4);
}

const test_gif_2x2 = [_]u8{ 71, 73, 70, 56, 57, 97, 2, 0, 2, 0, 129, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 44, 0, 0, 0, 0, 2, 0, 2, 0, 0, 2, 3, 68, 52, 5, 0, 59 };
const test_bmp_2x2 = [_]u8{ 66, 77, 70, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 16, 0, 0, 0, 19, 11, 0, 0, 19, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0 };

test "image input decodes GIF and BMP: red, green over blue, white" {
    const a = std.testing.allocator;
    const want = [_]u8{ 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255 };
    for ([_][]const u8{ &test_gif_2x2, &test_bmp_2x2 }) |bytes| {
        const img = decodeRgbOwned(a, bytes) orelse return error.DecodeFailed;
        defer img.deinit(a);
        try std.testing.expectEqual(@as(u32, 2), img.w);
        try std.testing.expectEqual(@as(u32, 2), img.h);
        try std.testing.expectEqualSlices(u8, &want, img.rgb);
        const size = imageHeaderSize(bytes) orelse return error.DecodeFailed;
        try std.testing.expectEqual(@as(u32, 2), size.w);
        try std.testing.expectEqual(@as(u32, 2), size.h);
    }
}
