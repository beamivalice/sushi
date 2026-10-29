const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const model_mod = @import("model.zig");
const vision_mod = @import("vision.zig");
const chat_mod = @import("chat.zig");
const model_registry_mod = @import("model_registry.zig");
const log = @import("log.zig");
const io_util = @import("io_util.zig");
const ModelConfig = model_mod.ModelConfig;
const VisionEncoder = vision_mod.VisionEncoder;
const LoadedModel = model_registry_mod.LoadedModel;
const testing = std.testing;
const scheduler_mod = @import("scheduler.zig");
const Scheduler = scheduler_mod.Scheduler;
const Slot = scheduler_mod.Slot;
const destroyAbandoned = scheduler_mod.destroyAbandoned;
const submitTestScheduler = scheduler_mod.submitTestScheduler;
const takeOrphanedVisionLocked = scheduler_mod.takeOrphanedVisionLocked;
const takeTeardownLocked = scheduler_mod.takeTeardownLocked;

pub var slot_vision_free_test_hook: ?*const fn (mlx.mlx_array) void = null;

/// The `VisionEncodeRequest.error_name` of an encode whose bill no longer fits when its turn comes.
pub const vision_headroom_error = "VisionHeadroomExceeded";

/// The `VisionEncodeRequest.error_name` of an image whose file would not decode.
pub const image_undecodable_error = "ImageUndecodable";

/// One media block of the prompt, in prompt order: the soft-token rows it takes from the encoded array.
pub const MediaBlock = struct { kind: chat_mod.MediaPart.Kind, rows: usize };

/// A request's media, encoded on the inference thread once the prefix restore has fixed how much of it
/// the restored KV already holds. `images`/`videos` borrow buffers the handler keeps alive until
/// `Scheduler.complete` (an `undecoded` image's is its file, decoded only if the encode reaches it);
/// `deinit` frees only the lists.
pub const DeferredMedia = struct {
    allocator: std.mem.Allocator,
    images: []const chat_mod.ImageData,
    videos: []const chat_mod.VideoData,
    blocks: []const MediaBlock,
    /// Connection-thread time spent reading this request's images, for the `Multimodal:` line.
    decode_ns: u64 = 0,

    pub fn deinit(self: *DeferredMedia) void {
        const allocator = self.allocator;
        allocator.free(self.images);
        allocator.free(self.videos);
        allocator.free(self.blocks);
        allocator.destroy(self);
    }
};

/// A media encode the engine could not run: what the client is told, and with which status.
pub const MediaRefusal = struct {
    server_error: bool,
    buf: [320]u8 = undefined,
    len: usize = 0,

    fn init(server_error: bool, comptime fmt: []const u8, args: anytype) MediaRefusal {
        var r: MediaRefusal = .{ .server_error = server_error };
        r.len = (std.fmt.bufPrint(&r.buf, fmt, args) catch @as([]u8, &r.buf)).len;
        return r;
    }

    pub fn text(self: *const MediaRefusal) []const u8 {
        return self.buf[0..self.len];
    }
};

/// What a restored prefix lets the encode leave out.
pub const MediaSkip = struct { blocks: usize = 0, images: usize = 0, videos: usize = 0, rows: usize = 0 };

/// The leading blocks whose every soft-token row lies inside the restored prefix (`consumed_rows`
/// placeholders): their KV is restored, so their embeddings are never read. A block the restore point
/// cuts through is not skipped.
pub fn skippedMedia(blocks: []const MediaBlock, consumed_rows: usize) MediaSkip {
    var skip: MediaSkip = .{};
    for (blocks) |b| {
        if (skip.rows + b.rows > consumed_rows) break;
        skip.rows += b.rows;
        skip.blocks += 1;
        switch (b.kind) {
            .image => skip.images += 1,
            .video => skip.videos += 1,
        }
    }
    return skip;
}

const EncodedMedia = struct { embeddings: ?mlx.mlx_array, skipped_rows: usize };

/// Encodes what `consumed_rows` of restored prefix leaves uncovered. `embeddings` is null when the
/// restore covers every block; its row 0 is placeholder row `skipped_rows` of the prompt.
pub fn encodeMediaAfterRestore(sch: *Scheduler, model: *LoadedModel, d: *const DeferredMedia, consumed_rows: usize, refusal: *?MediaRefusal) !EncodedMedia {
    const skip = skippedMedia(d.blocks, consumed_rows);
    const blocks = d.blocks[skip.blocks..];
    const images = d.images[skip.images..];
    const videos = d.videos[skip.videos..];
    var rows: usize = 0;
    for (blocks) |b| rows += b.rows;
    const sw = io_util.Stopwatch.init(sch.io);
    var decode_ns = d.decode_ns;
    defer log.info("  Multimodal: encoded {d} of {d} image(s) and {d} of {d} video(s), {d} of {d} soft tokens (the restored prefix holds {d}): decode {d} ms, encode {d} ms\n", .{
        images.len, d.images.len,
        videos.len, d.videos.len,
        rows,       rows + skip.rows,
        skip.rows,  decode_ns / std.time.ns_per_ms,
        sw.read() / std.time.ns_per_ms,
    });
    if (blocks.len == 0) return .{ .embeddings = null, .skipped_rows = skip.rows };

    const allocator = sch.allocator;
    const pix = try allocator.alloc(VisionImagePixels, images.len);
    defer allocator.free(pix);
    for (images, pix) |img, *p| p.* = .{ .pixels = img.pixels, .width = @intCast(img.width), .height = @intCast(img.height), .grid_h = img.grid_h, .grid_w = img.grid_w, .undecoded = img.undecoded };
    const vids = try allocator.alloc(VisionVideoPixels, videos.len);
    defer allocator.free(vids);
    for (videos, vids) |vid, *v| v.* = .{ .pixels = vid.pixels, .grid_t = vid.grid_t, .grid_h = vid.grid_h, .grid_w = vid.grid_w };
    const order = try allocator.alloc(chat_mod.MediaPart.Kind, blocks.len);
    defer allocator.free(order);
    for (blocks, order) |b, *k| k.* = b.kind;

    const cfg = model.config.?;
    var req = VisionEncodeRequest{
        .model = model,
        .images = pix,
        .videos = vids,
        .order = order,
        .bill_bytes = if (cfg.qv_heads == 0) 0 else if (scheduler_mod.vision_encode_bill) |bill| bill(cfg, images, videos, rows) else 0,
        .allocator = allocator,
    };
    runVisionEncode(sch, &req);
    decode_ns += req.decode_ns;
    if (req.error_name) |name| {
        defer allocator.free(name);
        if (std.mem.eql(u8, name, image_undecodable_error)) {
            refusal.* = MediaRefusal.init(false, "image {d} could not be decoded: the payload is not a readable JPEG, PNG, GIF, BMP or WebP image", .{skip.images + req.failed_image + 1});
            return error.MediaRefused;
        }
        var text: [320]u8 = undefined;
        const fit = if (std.mem.eql(u8, name, vision_headroom_error)) (if (scheduler_mod.vision_fit_fault) |f| f(&text, cfg, images, videos, rows, req.available_bytes) else null) else null;
        refusal.* = if (fit) |msg| MediaRefusal.init(false, "{s}", .{msg}) else MediaRefusal.init(true, "vision encoding failed: {s}", .{name});
        return error.MediaRefused;
    }
    const arr = req.result orelse return error.MediaRefused;
    if (req.n_vision_tokens + req.n_video_tokens != rows) {
        refusal.* = MediaRefusal.init(true, "vision encoder produced {d} rows for {d} placeholder tokens", .{ req.n_vision_tokens + req.n_video_tokens, rows });
        _ = mlx.mlx_array_free(arr);
        return error.MediaRefused;
    }
    return .{ .embeddings = arr, .skipped_rows = skip.rows };
}

/// Tests only: stands in for the tower on one image block.
var vision_block_test_hook: ?*const fn (VisionImagePixels) anyerror!mlx.mlx_array = null;

pub const orphan_vision_cap = 16;

/// Inference thread only (the sole mlx caller), like every other free of a slot's arrays.
pub fn freeVisionArray(ve: mlx.mlx_array) void {
    if (@import("builtin").is_test and slot_vision_free_test_hook != null) {
        slot_vision_free_test_hook.?(ve);
    } else {
        _ = mlx.mlx_array_free(ve);
    }
}

/// Phase A4: pixel data for a single image, decoded by the connection
/// thread (CPU only — stb_image / libwebp). The inference thread wraps this
/// in an `mlx_array` via `mlx_array_new_data` and runs the vision encoder.
pub const VisionImagePixels = struct {
    /// Raw bytes holding float32 pixel data. Qwen3-VL:
    /// merge-order pixel_values (N × C·tps·ps·ps × 4). Borrowed; must outlive the
    /// encodeVision call (which blocks until completion).
    pixels: []const u8,
    width: u32,
    height: u32,
    /// Qwen3-VL only: full patch grid. Selects QwenVision.
    grid_h: u32 = 0,
    grid_w: u32 = 0,
    /// `pixels` is still the encoded file; the encode decodes it (`ImageDecodeWindow`).
    undecoded: bool = false,
};

/// Qwen3-VL video: pre-patchified pixel_values for ALL `grid_t` temporal-patch
/// groups, concatenated (see `vision_common.buildPixelValuesVideo`). Borrowed;
/// must outlive the encodeVision call.
pub const VisionVideoPixels = struct {
    pixels: []const u8,
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
};

/// Phase A4: vision-encode work item. Conn thread fills `images` (raw pixel
/// data, CPU-only) and calls `Scheduler.encodeVision`, which posts the
/// request and blocks until the inference thread fills `result` and signals
/// `done`. Ownership of `result` transfers to the caller on success — pass
/// to `scheduler.submit(.{ .vision_embeddings = arr, ... })` and the slot's
/// `deinit` will free it.
pub const VisionEncodeRequest = struct {
    /// Plan 05 Phase D: target model whose `vision_encoder` services this
    /// request. The conn thread holds a refcount (via `ensureLoaded`) for
    /// the duration of the call.
    model: *model_registry_mod.LoadedModel,
    /// Per-image float32 CHW pixel buffers. Borrowed; must outlive the call.
    images: []const VisionImagePixels,
    /// Per-video pre-patchified pixel buffers. Borrowed; must outlive the call.
    /// Qwen-only (video_token_id != 0) — empty on every other arch.
    videos: []const VisionVideoPixels = &.{},
    /// Audio: per-clip raw float32-LE 16 kHz mono sample
    /// buffers. Borrowed; must outlive the call. The inference thread frames
    /// each into 640-sample tokens and projects them through the audio embedder.
    audio: []const []const u8 = &.{},
    /// The prompt order of the image and video blocks (each kind in its own
    /// list order); empty means every image, then every video.
    order: []const chat_mod.MediaPart.Kind = &.{},
    /// Output: encoded embedding tensor on success — image and video soft
    /// tokens in `order`, then audio soft tokens, concatenated along the token
    /// axis so the splice scatters each row into its placeholder.
    /// Ownership transfers to the caller.
    result: ?mlx.mlx_array = null,
    /// Output: number of vision / video / audio soft tokens in `result` (in
    /// that order). The caller inserts exactly this many image / video / audio
    /// placeholders.
    n_vision_tokens: usize = 0,
    n_video_tokens: usize = 0,
    /// `server.visionEncodeBill` of this request; the inference thread refuses the encode when
    /// the GPU has less left when its turn comes. 0 = unbilled.
    bill_bytes: u64 = 0,
    /// Output: what the GPU had left when `vision_headroom_error` refused the encode.
    available_bytes: u64 = 0,
    /// Output: error name on failure. Owned by `allocator`; caller frees.
    error_name: ?[]const u8 = null,
    /// Output: which image (an index into `images`) `image_undecodable_error` names.
    failed_image: usize = 0,
    /// Output: time spent decoding `undecoded` images.
    decode_ns: u64 = 0,
    /// Done flag (under done_mu). Caller's wait-loop drains the cond when
    /// this flips true.
    done: bool = false,
    allocator: std.mem.Allocator,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

/// Most image decodes that run at once, and so most decoded images the host holds for one request.
pub const decode_workers = 8;

/// An `undecoded` image's file to its pixels; null cannot decode.
pub const ImageDecodeFn = *const fn (std.mem.Allocator, []const u8, *const model_mod.ModelConfig) ?chat_mod.ImageData;

/// Decodes the `undecoded` images of a request `decode_workers` at a time, one thread each, and frees
/// each decoded image once its reader (the media key, the tower call) is done with it: a request of
/// many screenshots never holds more than a window of decoded patches.
pub const ImageDecodeWindow = struct {
    allocator: std.mem.Allocator,
    config: ?*const model_mod.ModelConfig,
    decode: ?ImageDecodeFn,
    images: []const VisionImagePixels,
    first: usize = 0,
    len: usize = 0,
    slots: [decode_workers]?chat_mod.ImageData = @splat(null),
    decode_ns: u64 = 0,
    /// The index of the image `view` last failed to decode.
    failed: usize = 0,

    fn nowNs() u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    const Pool = struct {
        window: *ImageDecodeWindow,
        next: std.atomic.Value(usize) = .init(0),

        fn work(self: *Pool) void {
            const w = self.window;
            const decode = w.decode orelse return;
            const config = w.config orelse return;
            while (true) {
                const i = self.next.fetchAdd(1, .monotonic);
                if (i >= w.len) return;
                const src = w.images[w.first + i];
                if (src.undecoded) w.slots[i] = decode(w.allocator, src.pixels, config);
            }
        }
    };

    fn refill(self: *ImageDecodeWindow, from: usize) void {
        const started = nowNs();
        self.deinit();
        self.first = from;
        self.len = @min(decode_workers, self.images.len - from);
        var pool = Pool{ .window = self };
        var threads: [decode_workers - 1]std.Thread = undefined;
        var n_threads: usize = 0;
        for (0..self.len - 1) |_| {
            threads[n_threads] = std.Thread.spawn(.{}, Pool.work, .{&pool}) catch break;
            n_threads += 1;
        }
        pool.work();
        for (threads[0..n_threads]) |t| t.join();
        self.decode_ns += nowNs() - started;
    }

    /// The pixels of `images[index]`, decoding its window when it is not in the current one. Null when
    /// the file does not decode.
    pub fn view(self: *ImageDecodeWindow, index: usize) ?VisionImagePixels {
        const src = self.images[index];
        if (!src.undecoded) return src;
        if (index < self.first or index >= self.first + self.len) self.refill(index);
        const img = self.slots[index - self.first] orelse {
            self.failed = index;
            return null;
        };
        return .{ .pixels = img.pixels, .width = img.width, .height = img.height, .grid_h = img.grid_h, .grid_w = img.grid_w };
    }

    pub fn release(self: *ImageDecodeWindow, index: usize) void {
        if (index < self.first or index >= self.first + self.len) return;
        const slot = &self.slots[index - self.first];
        if (slot.*) |img| self.allocator.free(img.pixels);
        slot.* = null;
    }

    pub fn deinit(self: *ImageDecodeWindow) void {
        for (&self.slots) |*slot| {
            if (slot.*) |img| self.allocator.free(img.pixels);
            slot.* = null;
        }
        self.len = 0;
    }
};

/// One image through the tower: a patch-grid ViT takes pixel_values [N, feat]
/// and yields [1, N/merge², hidden].
fn encodeImageBlock(vision_enc: *VisionEncoder, img: VisionImagePixels) !mlx.mlx_array {
    if (@import("builtin").is_test) if (vision_block_test_hook) |hook| return hook(img);
    if (img.grid_h > 0) {
        const n: usize = @as(usize, img.grid_h) * img.grid_w;
        const feat: usize = (img.pixels.len / 4) / n;
        const shape = [_]c_int{ @intCast(n), @intCast(feat) };
        const pixel_arr = mlx.mlx_array_new_data(img.pixels.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(pixel_arr);
        return vision_enc.forwardPatches(pixel_arr, img.grid_h, img.grid_w);
    }
    return error.NoPatchGridEncoder;
}

pub fn encodeVideoBlock(vision_enc: *VisionEncoder, vid: VisionVideoPixels) !mlx.mlx_array {
    const n: usize = @as(usize, vid.grid_t) * vid.grid_h * vid.grid_w;
    const feat: usize = (vid.pixels.len / 4) / n;
    const shape = [_]c_int{ @intCast(n), @intCast(feat) };
    const pixel_arr = mlx.mlx_array_new_data(vid.pixels.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(pixel_arr);
    return vision_enc.forwardVideoPatches(pixel_arr, vid.grid_t, vid.grid_h, vid.grid_w);
}

/// Phase A4: encode one or more images on the inference thread. Writes the
/// result into a request struct + signals done, so the conn thread (blocked
/// in `encodeVision`) gets the output. On error, sets `req.error_name` and
/// still signals done.
/// Plan 05 Phase D: routes the encode through `req.model.vision_encoder`,
/// not the scheduler's borrowed-view singleton — each LoadedModel has its
/// own vision encoder when applicable.
pub fn runVisionEncode(sch: *Scheduler, req: *VisionEncodeRequest) void {
    const vision_enc = req.model.vision_encoder orelse {
        finishVisionRequest(sch, req, "VisionEncoderNotLoaded");
        return;
    };
    if (req.images.len == 0 and req.videos.len == 0 and req.audio.len == 0) {
        finishVisionRequest(sch, req, "EmptyImages");
        return;
    }
    // The connection thread's fit check ran before the queue: a prefill's cache growth, or the
    // output of an encode ahead of this one, may have taken the headroom since.
    if (req.bill_bytes > 0) if (scheduler_mod.vision_encode_available) |available_fn| if (req.model.config) |cfg| {
        const available = available_fn(cfg);
        if (req.bill_bytes > available) {
            req.available_bytes = available;
            finishVisionRequest(sch, req, vision_headroom_error);
            return;
        }
    };

    if (req.model.config) |cfg| @import("dsv41/dsv41_borrow.zig").enterVisionMlx(req.bill_bytes, cfg);
    defer if (req.model.config) |cfg| @import("dsv41/dsv41_borrow.zig").leaveVisionMlx(cfg);

    // Encode all soft tokens into `emb_parts`: images and videos in prompt
    // `order`, then audio, so the single splice channel scatters them in the
    // order of the placeholder rows.
    var emb_parts = std.ArrayList(mlx.mlx_array).empty;
    defer emb_parts.deinit(req.allocator);
    const failParts = struct {
        fn f(s: *Scheduler, r: *VisionEncodeRequest, parts: []mlx.mlx_array, name: []const u8) void {
            for (parts) |e| _ = mlx.mlx_array_free(e);
            finishVisionRequest(s, r, name);
        }
    }.f;

    if (req.order.len > 0 and req.order.len != req.images.len + req.videos.len) {
        finishVisionRequest(sch, req, "MediaOrderMismatch");
        return;
    }
    var n_vision: usize = 0;
    var n_video: usize = 0;
    var next_image: usize = 0;
    var next_video: usize = 0;
    var window = ImageDecodeWindow{ .allocator = req.allocator, .config = req.model.config, .decode = scheduler_mod.vision_image_decode, .images = req.images };
    defer window.deinit();
    for (0..req.images.len + req.videos.len) |block| {
        const is_image = if (req.order.len > 0) req.order[block] == .image else block < req.images.len;
        if ((is_image and next_image == req.images.len) or (!is_image and next_video == req.videos.len)) {
            failParts(sch, req, emb_parts.items, "MediaOrderMismatch");
            return;
        }
        const emb = (if (is_image) blk: {
            defer req.decode_ns = window.decode_ns;
            const px = window.view(next_image) orelse {
                req.failed_image = next_image;
                failParts(sch, req, emb_parts.items, image_undecodable_error);
                return;
            };
            defer window.release(next_image);
            break :blk encodeImageBlock(vision_enc, px);
        } else encodeVideoBlock(vision_enc, req.videos[next_video])) catch |err| {
            failParts(sch, req, emb_parts.items, @errorName(err));
            return;
        };
        const rows: usize = @intCast(mlx.getShape(emb)[1]);
        if (is_image) {
            next_image += 1;
            n_vision += rows;
        } else {
            next_video += 1;
            n_video += rows;
        }
        emb_parts.append(req.allocator, emb) catch |err| {
            _ = mlx.mlx_array_free(emb);
            failParts(sch, req, emb_parts.items, @errorName(err));
            return;
        };
    }

    if (emb_parts.items.len == 0) {
        finishVisionRequest(sch, req, "EmptyImages");
        return;
    }

    // Single modality/clip: pass through. Multiple: concatenate along token dim.
    var combined: mlx.mlx_array = undefined;
    if (emb_parts.items.len == 1) {
        combined = emb_parts.items[0];
        emb_parts.items[0] = mlx.mlx_array_new(); // sentinel so the deferred-free path is a no-op
    } else {
        const cat_vec = mlx.mlx_vector_array_new_data(emb_parts.items.ptr, emb_parts.items.len);
        defer _ = mlx.mlx_vector_array_free(cat_vec);
        combined = mlx.mlx_array_new();
        if (mlx.mlx_concatenate_axis(&combined, cat_vec, 1, vision_enc.s) != 0) {
            _ = mlx.mlx_array_free(combined);
            failParts(sch, req, emb_parts.items, "ConcatenateFailed");
            return;
        }
    }
    for (emb_parts.items) |e| _ = mlx.mlx_array_free(e);

    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    req.result = combined;
    req.n_vision_tokens = n_vision;
    req.n_video_tokens = n_video;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

fn finishVisionRequest(sch: *Scheduler, req: *VisionEncodeRequest, err_name: []const u8) void {
    req.done_mu.lockUncancelable(sch.io);
    defer req.done_mu.unlock(sch.io);
    if (req.error_name) |old| req.allocator.free(old);
    req.error_name = req.allocator.dupe(u8, err_name) catch null;
    req.done = true;
    req.done_cond.broadcast(sch.io);
}

test "a cleanup allocation failure never frees MLX on the connection thread" {
    const Probe = struct {
        var inference_id: std.Thread.Id = undefined;
        var frees: usize = 0;
        var off_thread_frees: usize = 0;
        fn free(_: mlx.mlx_array) void {
            frees += 1;
            if (std.Thread.getCurrentId() != inference_id) off_thread_frees += 1;
        }
        fn complete(sch: *Scheduler, slots: []const *Slot) void {
            for (slots) |slot| sch.complete(slot);
        }
    };
    Probe.inference_id = std.Thread.getCurrentId();
    Probe.frees = 0;
    Probe.off_thread_frees = 0;
    slot_vision_free_test_hook = Probe.free;
    defer slot_vision_free_test_hook = null;
    var failing = testing.FailingAllocator.init(testing.allocator, .{});
    const allocator = failing.allocator();
    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.transformer = null;
    var sch: Scheduler = undefined;
    sch.allocator = allocator;
    sch.io = testing.io;
    sch.kv_quant_config = .dense;
    sch.kv_quant_explicit = false;
    sch.queue_mu = .init;
    sch.queue_cond = .init;
    sch.submit_cond = .init;
    sch.shutdown = .init(false);
    sch.queue_cap = 2;
    sch.in_flight = 0;
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.cleanup_queue = .empty;
    sch.prefilling = .empty;
    defer sch.pending.deinit(allocator);
    defer sch.decoding.deinit(allocator);
    defer sch.cleanup_queue.deinit(allocator);
    defer sch.prefilling.deinit(allocator);
    defer for (sch.cleanup_queue.items) |slot| slot.deinit();
    for (0..12) |_| {
        failing.fail_index = std.math.maxInt(usize);
        failing.resize_fail_index = std.math.maxInt(usize);
        var slots: [2]*Slot = undefined;
        for (&slots) |*slot| slot.* = try sch.submit(.{
            .model = &model,
            .prompt_ids = &.{1},
            .sampling = .{},
            .eos_token_ids = &.{},
            .max_tokens = 1,
            .vision_embeddings = .{ .ctx = @ptrFromInt(1) },
        });
        failing.fail_index = failing.alloc_index;
        failing.resize_fail_index = failing.resize_index;
        const conn = try std.Thread.spawn(.{}, Probe.complete, .{ &sch, &slots });
        conn.join();
        try testing.expectEqual(@as(usize, 0), Probe.off_thread_frees);
        try testing.expectEqual(@as(usize, 0), Probe.frees);
        try testing.expectEqual(@as(usize, 0), sch.in_flight);
    }
    try testing.expectEqual(@as(usize, 24), sch.cleanup_queue.items.len);
    while (sch.cleanup_queue.items.len > 0) sch.cleanup_queue.orderedRemove(0).deinit();
    try testing.expectEqual(@as(usize, 24), Probe.frees);
    try testing.expectEqual(@as(usize, 0), Probe.off_thread_frees);
}

test "an image request keeps its drafter: verify carries the M-RoPE positions, the drafter reads the sequence" {
    const Probe = struct {
        fn free(_: mlx.mlx_array) void {}
    };
    slot_vision_free_test_hook = Probe.free;
    defer slot_vision_free_test_hook = null;
    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.transformer = null;
    var sch = submitTestScheduler(testing.allocator, 2);
    defer sch.pending.deinit(testing.allocator);
    defer sch.cleanup_queue.deinit(testing.allocator);
    defer sch.prefilling.deinit(testing.allocator);
    const slot = try sch.submit(.{
        .model = &model,
        .prompt_ids = &.{1},
        .sampling = .{},
        .eos_token_ids = &.{},
        .max_tokens = 1,
        .enable_drafter = true,
        .vision_embeddings = .{ .ctx = @ptrFromInt(1) },
    });
    try testing.expect(slot.enable_drafter);
    sch.complete(slot);
    var batch: [16]*Slot = undefined;
    const take = takeTeardownLocked(&sch, &batch);
    for (batch[0..take.cleanup_n]) |s| s.deinit();
    destroyAbandoned(take.abandoned);
}

test "a submit that fails once its slot is built hands the slot to the inference thread" {
    const Probe = struct {
        var inference_id: std.Thread.Id = undefined;
        var frees: usize = 0;
        var off_thread_frees: usize = 0;
        fn free(_: mlx.mlx_array) void {
            frees += 1;
            if (std.Thread.getCurrentId() != inference_id) off_thread_frees += 1;
        }
        fn submit(sch: *Scheduler, model: *LoadedModel, slot: *?*Slot, err: *?anyerror) void {
            slot.* = sch.submit(.{
                .model = model,
                .prompt_ids = &.{1},
                .sampling = .{},
                .eos_token_ids = &.{},
                .max_tokens = 1,
                .vision_embeddings = .{ .ctx = @ptrFromInt(1) },
            }) catch |e| {
                err.* = e;
                return;
            };
        }
    };
    Probe.inference_id = std.Thread.getCurrentId();
    slot_vision_free_test_hook = Probe.free;
    defer slot_vision_free_test_hook = null;
    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.transformer = null;
    var handed_over: usize = 0;
    var fail_at: usize = 0;
    while (fail_at < 32) : (fail_at += 1) {
        var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = fail_at });
        const allocator = failing.allocator();
        var sch = submitTestScheduler(allocator, 2);
        defer sch.pending.deinit(allocator);
        defer sch.cleanup_queue.deinit(allocator);
        defer sch.prefilling.deinit(allocator);
        Probe.frees = 0;
        Probe.off_thread_frees = 0;
        var slot: ?*Slot = null;
        var err: ?anyerror = null;
        const conn = try std.Thread.spawn(.{}, Probe.submit, .{ &sch, &model, &slot, &err });
        conn.join();
        try testing.expectEqual(@as(usize, 0), Probe.off_thread_frees);
        if (slot) |s| sch.complete(s);
        // The inference thread's pass over what the connection left behind.
        var batch: [16]*Slot = undefined;
        const take = takeTeardownLocked(&sch, &batch);
        for (batch[0..take.cleanup_n]) |s| s.deinit();
        const built = slot != null or take.abandoned != null;
        if (take.abandoned != null) handed_over += 1;
        destroyAbandoned(take.abandoned);
        try testing.expectEqual(@as(usize, if (built) 1 else 0), Probe.frees);
        try testing.expectEqual(@as(usize, 0), Probe.off_thread_frees);
    }
    // cleanup_queue, prefilling and pending each grow once after the slot is built.
    try testing.expect(handed_over >= 3);

    var shut = submitTestScheduler(testing.allocator, 2);
    shut.shutdown.store(true, .release);
    Probe.frees = 0;
    var shut_slot: ?*Slot = null;
    var shut_err: ?anyerror = null;
    const shut_conn = try std.Thread.spawn(.{}, Probe.submit, .{ &shut, &model, &shut_slot, &shut_err });
    shut_conn.join();
    try testing.expect(shut_err.? == error.Shutdown);
    try testing.expectEqual(@as(usize, 0), Probe.off_thread_frees);
    destroyAbandoned(shut.abandoned);
    try testing.expectEqual(@as(usize, 1), Probe.frees);
}

test "a submit that fails before its slot exists hands the vision array to the inference thread" {
    const Probe = struct {
        var inference_id: std.Thread.Id = undefined;
        var frees: usize = 0;
        var off_thread_frees: usize = 0;
        fn free(_: mlx.mlx_array) void {
            frees += 1;
            if (std.Thread.getCurrentId() != inference_id) off_thread_frees += 1;
        }
        fn submit(sch: *Scheduler, model: *LoadedModel, err: *?anyerror) void {
            _ = sch.submit(.{
                .model = model,
                .prompt_ids = &.{1},
                .sampling = .{},
                .eos_token_ids = &.{},
                .max_tokens = 1,
                .vision_embeddings = .{ .ctx = @ptrFromInt(1) },
            }) catch |e| {
                err.* = e;
            };
        }
        /// One connection-thread submit that must fail, then the inference thread's drain.
        fn expectHandedOver(allocator: std.mem.Allocator, model: *LoadedModel, kv: transformer_mod.KVQuantConfig, want: ?anyerror) !void {
            var sch: Scheduler = undefined;
            sch.allocator = allocator;
            sch.io = testing.io;
            sch.kv_quant_config = kv;
            sch.kv_quant_explicit = true;
            sch.queue_mu = .init;
            sch.queue_cond = .init;
            sch.submit_cond = .init;
            sch.shutdown = .init(false);
            sch.queue_cap = 2;
            sch.in_flight = 0;
            sch.req_seq = 0;
            sch.pending = .empty;
            sch.cleanup_queue = .empty;
            sch.prefilling = .empty;
            sch.orphan_vision_n = 0;
            defer sch.pending.deinit(allocator);
            defer sch.cleanup_queue.deinit(allocator);
            defer sch.prefilling.deinit(allocator);
            frees = 0;
            off_thread_frees = 0;
            var err: ?anyerror = null;
            const conn = try std.Thread.spawn(.{}, submit, .{ &sch, model, &err });
            conn.join();
            if (want) |w| try testing.expectEqual(w, err.?) else try testing.expect(err != null);
            try testing.expectEqual(@as(usize, 0), off_thread_frees);
            try testing.expectEqual(@as(usize, 0), frees);
            var buf: [orphan_vision_cap]mlx.mlx_array = undefined;
            const n = takeOrphanedVisionLocked(&sch, &buf);
            for (buf[0..n]) |ve| freeVisionArray(ve);
            try testing.expectEqual(@as(usize, 1), n);
            try testing.expectEqual(@as(usize, 1), frees);
            try testing.expectEqual(@as(usize, 0), off_thread_frees);
            try testing.expectEqual(@as(usize, 0), takeOrphanedVisionLocked(&sch, &buf));
        }
    };
    Probe.inference_id = std.Thread.getCurrentId();
    slot_vision_free_test_hook = Probe.free;
    defer slot_vision_free_test_hook = null;

    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.transformer = null;

    model.config = null;
    try Probe.expectHandedOver(testing.allocator, &model, .dense, error.ModelNotReady);

    var glm_cfg = ModelConfig{ .num_hidden_layers = 0, .model_type = "glm5_next" };
    model.config = &glm_cfg;
    try Probe.expectHandedOver(testing.allocator, &model, .affine(4), error.GlmKvQuantUnsupported);

    // Every allocation Slot.init makes is a point where it can fail.
    model.config = &cfg;
    var counting = testing.FailingAllocator.init(testing.allocator, .{});
    const probe_slot = try Slot.init(counting.allocator(), testing.io, &cfg, .{
        .prompt_ids = &.{1},
        .sampling = .{},
        .eos_token_ids = &.{},
        .max_tokens = 1,
        .vision_embeddings = null,
        .model = &model,
    }, .dense);
    probe_slot.deinit();
    try testing.expect(counting.alloc_index > 0);
    for (0..counting.alloc_index) |fail_index| {
        var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = fail_index });
        try Probe.expectHandedOver(failing.allocator(), &model, .dense, error.OutOfMemory);
    }
}

test "a queued vision encode is refused when its bill no longer fits, and an earlier output counts as spent" {
    const Probe = struct {
        var capacity: u64 = 0;
        var resident: u64 = 0;
        var tower_runs: usize = 0;
        fn available(_: *const ModelConfig) u64 {
            return capacity -| resident;
        }
        fn tower(_: VisionImagePixels) anyerror!mlx.mlx_array {
            tower_runs += 1;
            resident += 100;
            const rows = [_]f32{ 0, 0, 0, 0 };
            return mlx.mlx_array_new_data(&rows, &[_]c_int{ 1, 1, 4 }, 3, .float32);
        }
        fn run(sch: *Scheduler, model: *LoadedModel, bill: u64) !VisionEncodeRequest {
            const pixels = [_]VisionImagePixels{.{ .pixels = &.{}, .width = 1, .height = 1 }};
            var req = VisionEncodeRequest{ .model = model, .images = &pixels, .bill_bytes = bill, .allocator = testing.allocator };
            runVisionEncode(sch, &req);
            return req;
        }
    };
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.vision_encoder = @ptrFromInt(@alignOf(VisionEncoder));
    vision_block_test_hook = Probe.tower;
    scheduler_mod.vision_encode_available = Probe.available;
    defer {
        vision_block_test_hook = null;
        scheduler_mod.vision_encode_available = null;
    }

    // Headroom shrank after the connection thread's check: the tower never runs.
    Probe.capacity = 50;
    Probe.resident = 0;
    Probe.tower_runs = 0;
    const refused = try Probe.run(&sch, &model, 100);
    defer if (refused.error_name) |e| testing.allocator.free(e);
    try testing.expectEqualStrings(vision_headroom_error, refused.error_name orelse return error.TestExpectedRefusal);
    try testing.expectEqual(@as(u64, 50), refused.available_bytes);
    try testing.expect(refused.done);
    try testing.expect(refused.result == null);
    try testing.expectEqual(@as(usize, 0), Probe.tower_runs);

    // Two encodes queued together: the first fits, and its output (held by its connection thread)
    // leaves the second without room.
    Probe.capacity = 150;
    Probe.resident = 0;
    const first = try Probe.run(&sch, &model, 100);
    defer if (first.result) |r| {
        _ = mlx.mlx_array_free(r);
    };
    try testing.expect(first.error_name == null);
    try testing.expectEqual(@as(usize, 1), Probe.tower_runs);
    const second = try Probe.run(&sch, &model, 100);
    defer if (second.error_name) |e| testing.allocator.free(e);
    try testing.expectEqualStrings(vision_headroom_error, second.error_name orelse return error.TestExpectedRefusal);
    try testing.expectEqual(@as(u64, 50), second.available_bytes);
    try testing.expectEqual(@as(usize, 1), Probe.tower_runs);

    // An unbilled request (a tower with no bill) is never refused here.
    const unbilled = try Probe.run(&sch, &model, 0);
    defer if (unbilled.result) |r| {
        _ = mlx.mlx_array_free(r);
    };
    try testing.expect(unbilled.error_name == null);
}

test "skippedMedia leaves out only the leading blocks the restored prefix holds whole" {
    const blocks = [_]MediaBlock{ .{ .kind = .image, .rows = 4 }, .{ .kind = .video, .rows = 2 }, .{ .kind = .image, .rows = 3 } };
    const none = skippedMedia(&blocks, 0);
    try testing.expectEqual(@as(usize, 0), none.blocks);
    try testing.expectEqual(@as(usize, 0), none.rows);
    // A 5-row prefix ends inside the video: it is encoded.
    const straddle = skippedMedia(&blocks, 5);
    try testing.expectEqual(MediaSkip{ .blocks = 1, .images = 1, .videos = 0, .rows = 4 }, straddle);
    try testing.expectEqual(MediaSkip{ .blocks = 2, .images = 1, .videos = 1, .rows = 6 }, skippedMedia(&blocks, 6));
    try testing.expectEqual(MediaSkip{ .blocks = 2, .images = 1, .videos = 1, .rows = 6 }, skippedMedia(&blocks, 8));
    try testing.expectEqual(MediaSkip{ .blocks = 3, .images = 2, .videos = 1, .rows = 9 }, skippedMedia(&blocks, 9));
}

test "a restored prefix lets the encode skip its images, and the rest match a full encode byte for byte" {
    const Probe = struct {
        var tower_runs: usize = 0;
        var billed_images: usize = 0;
        var billed_rows: usize = 0;
        /// One row per `width`, four floats each: the image's first byte, then the row index.
        fn tower(img: VisionImagePixels) anyerror!mlx.mlx_array {
            tower_runs += 1;
            var data: [64]f32 = undefined;
            const rows: usize = img.width;
            for (0..rows) |r| {
                data[r * 4 + 0] = @floatFromInt(img.pixels[0]);
                data[r * 4 + 1] = @floatFromInt(r);
                data[r * 4 + 2] = @floatFromInt(img.pixels[0] * 3);
                data[r * 4 + 3] = 0.1 * @as(f32, @floatFromInt(img.pixels[0]));
            }
            return mlx.mlx_array_new_data(&data, &[_]c_int{ 1, @intCast(rows), 4 }, 3, .float32);
        }
        fn available(_: *const ModelConfig) u64 {
            return std.math.maxInt(u64);
        }
        fn bill(_: *const ModelConfig, images: []const chat_mod.ImageData, _: []const chat_mod.VideoData, rows: usize) u64 {
            billed_images = images.len;
            billed_rows = rows;
            return 1;
        }
        fn rowsOf(arr: mlx.mlx_array) []const f32 {
            _ = mlx.mlx_array_eval(arr);
            const shape = mlx.getShape(arr);
            return mlx.mlx_array_data_float32(arr).?[0 .. @as(usize, @intCast(shape[1])) * 4];
        }
    };
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.allocator = testing.allocator;
    var cfg = ModelConfig{ .num_hidden_layers = 0, .qv_heads = 1 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    var encoder: VisionEncoder = .{ .allocator = testing.allocator, .s = mlx.mlx_default_cpu_stream_new() };
    model.vision_encoder = &encoder;
    vision_block_test_hook = Probe.tower;
    scheduler_mod.vision_encode_available = Probe.available;
    scheduler_mod.vision_encode_bill = Probe.bill;
    defer {
        vision_block_test_hook = null;
        scheduler_mod.vision_encode_available = null;
        scheduler_mod.vision_encode_bill = null;
    }

    const images = [_]chat_mod.ImageData{
        .{ .pixels = "\x01", .width = 4, .height = 1 },
        .{ .pixels = "\x02", .width = 2, .height = 1 },
        .{ .pixels = "\x03", .width = 3, .height = 1 },
    };
    const blocks = [_]MediaBlock{ .{ .kind = .image, .rows = 4 }, .{ .kind = .image, .rows = 2 }, .{ .kind = .image, .rows = 3 } };
    const media = DeferredMedia{ .allocator = testing.allocator, .images = &images, .videos = &.{}, .blocks = &blocks };
    var refusal: ?MediaRefusal = null;

    const full = try encodeMediaAfterRestore(&sch, &model, &media, 0, &refusal);
    defer _ = mlx.mlx_array_free(full.embeddings.?);
    try testing.expectEqual(@as(usize, 3), Probe.tower_runs);
    try testing.expectEqual(@as(usize, 3), Probe.billed_images);
    try testing.expectEqual(@as(usize, 9), Probe.billed_rows);
    const all_rows = Probe.rowsOf(full.embeddings.?);

    // The first image is inside the 4-row prefix: the tower runs twice and the bill sees two images.
    Probe.tower_runs = 0;
    const tail = try encodeMediaAfterRestore(&sch, &model, &media, 4, &refusal);
    defer _ = mlx.mlx_array_free(tail.embeddings.?);
    try testing.expectEqual(@as(usize, 2), Probe.tower_runs);
    try testing.expectEqual(@as(usize, 2), Probe.billed_images);
    try testing.expectEqual(@as(usize, 5), Probe.billed_rows);
    try testing.expectEqual(@as(usize, 4), tail.skipped_rows);
    try testing.expectEqualSlices(f32, all_rows[4 * 4 ..], Probe.rowsOf(tail.embeddings.?));

    // The prefix ends inside the second image: it is encoded whole, and the splice resumes at its row 1.
    Probe.tower_runs = 0;
    const straddled = try encodeMediaAfterRestore(&sch, &model, &media, 5, &refusal);
    defer _ = mlx.mlx_array_free(straddled.embeddings.?);
    try testing.expectEqual(@as(usize, 2), Probe.tower_runs);
    try testing.expectEqual(@as(usize, 4), straddled.skipped_rows);
    try testing.expectEqualSlices(f32, all_rows[4 * 4 ..], Probe.rowsOf(straddled.embeddings.?));

    // A prefix holding every image encodes nothing.
    Probe.tower_runs = 0;
    const covered = try encodeMediaAfterRestore(&sch, &model, &media, 9, &refusal);
    try testing.expect(covered.embeddings == null);
    try testing.expectEqual(@as(usize, 9), covered.skipped_rows);
    try testing.expectEqual(@as(usize, 0), Probe.tower_runs);
}

test "undecoded images are decoded a window at a time, freed after their tower call, and a restored prefix never decodes its own" {
    const Probe = struct {
        var live = std.atomic.Value(isize).init(0);
        var peak = std.atomic.Value(isize).init(0);
        var decodes = std.atomic.Value(usize).init(0);
        var fail_id: u8 = 0;
        fn decode(allocator: std.mem.Allocator, file: []const u8, _: *const ModelConfig) ?chat_mod.ImageData {
            _ = decodes.fetchAdd(1, .monotonic);
            if (file[0] == fail_id) return null;
            const pixels = allocator.alloc(u8, 1) catch return null;
            pixels[0] = file[0];
            const now = live.fetchAdd(1, .monotonic) + 1;
            _ = peak.fetchMax(now, .monotonic);
            return .{ .pixels = pixels, .width = file[1], .height = 1 };
        }
        fn tower(img: VisionImagePixels) anyerror!mlx.mlx_array {
            if (img.undecoded) return error.TowerGotAFile;
            var data: [64]f32 = undefined;
            const rows: usize = img.width;
            for (0..rows) |r| {
                data[r * 4 + 0] = @floatFromInt(img.pixels[0]);
                data[r * 4 + 1] = @floatFromInt(r);
                data[r * 4 + 2] = 0;
                data[r * 4 + 3] = 0;
            }
            _ = live.fetchSub(1, .monotonic);
            return mlx.mlx_array_new_data(&data, &[_]c_int{ 1, @intCast(rows), 4 }, 3, .float32);
        }
        fn available(_: *const ModelConfig) u64 {
            return std.math.maxInt(u64);
        }
        fn rowsOf(arr: mlx.mlx_array) []const f32 {
            _ = mlx.mlx_array_eval(arr);
            const shape = mlx.getShape(arr);
            return mlx.mlx_array_data_float32(arr).?[0 .. @as(usize, @intCast(shape[1])) * 4];
        }
    };
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.allocator = testing.allocator;
    var cfg = ModelConfig{ .num_hidden_layers = 0 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    var encoder: VisionEncoder = .{ .allocator = testing.allocator, .s = mlx.mlx_default_cpu_stream_new() };
    model.vision_encoder = &encoder;
    vision_block_test_hook = Probe.tower;
    scheduler_mod.vision_encode_available = Probe.available;
    scheduler_mod.vision_image_decode = Probe.decode;
    defer {
        vision_block_test_hook = null;
        scheduler_mod.vision_encode_available = null;
        scheduler_mod.vision_image_decode = null;
    }

    const n = 20;
    var eager: [n]chat_mod.ImageData = undefined;
    var files: [n]chat_mod.ImageData = undefined;
    var file_bytes: [n][2]u8 = undefined;
    var blocks: [n]MediaBlock = undefined;
    var rows_total: usize = 0;
    for (0..n) |i| {
        const id: u8 = @intCast(i + 1);
        const width: u8 = @intCast(1 + i % 3);
        file_bytes[i] = .{ id, width };
        eager[i] = .{ .pixels = file_bytes[i][0..1], .width = width, .height = 1 };
        files[i] = .{ .pixels = &file_bytes[i], .width = width, .height = 1, .undecoded = true };
        blocks[i] = .{ .kind = .image, .rows = width };
        rows_total += width;
    }
    var refusal: ?MediaRefusal = null;

    const whole = DeferredMedia{ .allocator = testing.allocator, .images = &eager, .videos = &.{}, .blocks = &blocks };
    const want = try encodeMediaAfterRestore(&sch, &model, &whole, 0, &refusal);
    defer _ = mlx.mlx_array_free(want.embeddings.?);
    try testing.expectEqual(@as(usize, 0), Probe.decodes.load(.monotonic));

    Probe.live.store(0, .monotonic);
    const lazy = DeferredMedia{ .allocator = testing.allocator, .images = &files, .videos = &.{}, .blocks = &blocks };
    const got = try encodeMediaAfterRestore(&sch, &model, &lazy, 0, &refusal);
    defer _ = mlx.mlx_array_free(got.embeddings.?);
    try testing.expectEqualSlices(f32, Probe.rowsOf(want.embeddings.?), Probe.rowsOf(got.embeddings.?));
    try testing.expectEqual(@as(usize, n), Probe.decodes.load(.monotonic));
    try testing.expectEqual(@as(isize, decode_workers), Probe.peak.load(.monotonic));
    try testing.expectEqual(@as(isize, 0), Probe.live.load(.monotonic));

    // The first four images sit inside the restored prefix: their files are never opened.
    var prefix_rows: usize = 0;
    for (blocks[0..4]) |b| prefix_rows += b.rows;
    Probe.decodes.store(0, .monotonic);
    const tail = try encodeMediaAfterRestore(&sch, &model, &lazy, prefix_rows, &refusal);
    defer _ = mlx.mlx_array_free(tail.embeddings.?);
    try testing.expectEqual(@as(usize, n - 4), Probe.decodes.load(.monotonic));
    try testing.expectEqualSlices(f32, Probe.rowsOf(want.embeddings.?)[prefix_rows * 4 ..], Probe.rowsOf(tail.embeddings.?));

    // A file that will not decode refuses the request by its place in it.
    Probe.fail_id = 7;
    try testing.expectError(error.MediaRefused, encodeMediaAfterRestore(&sch, &model, &lazy, 0, &refusal));
    try testing.expect(std.mem.startsWith(u8, refusal.?.text(), "image 7 could not be decoded"));
    try testing.expect(!refusal.?.server_error);
}

test "an encode refused for headroom after the restore names the bill of what it encodes" {
    const Probe = struct {
        fn available(_: *const ModelConfig) u64 {
            return 10;
        }
        fn bill(_: *const ModelConfig, images: []const chat_mod.ImageData, _: []const chat_mod.VideoData, _: usize) u64 {
            return 100 * images.len;
        }
        fn fault(out: []u8, _: *const ModelConfig, images: []const chat_mod.ImageData, _: []const chat_mod.VideoData, _: usize, free: u64) ?[]const u8 {
            return std.fmt.bufPrint(out, "{d} images, {d} free", .{ images.len, free }) catch null;
        }
        fn tower(_: VisionImagePixels) anyerror!mlx.mlx_array {
            return error.TowerMustNotRun;
        }
    };
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.allocator = testing.allocator;
    var cfg = ModelConfig{ .num_hidden_layers = 0, .qv_heads = 1 };
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.vision_encoder = @ptrFromInt(@alignOf(VisionEncoder));
    vision_block_test_hook = Probe.tower;
    scheduler_mod.vision_encode_available = Probe.available;
    scheduler_mod.vision_encode_bill = Probe.bill;
    scheduler_mod.vision_fit_fault = Probe.fault;
    defer {
        vision_block_test_hook = null;
        scheduler_mod.vision_encode_available = null;
        scheduler_mod.vision_encode_bill = null;
        scheduler_mod.vision_fit_fault = null;
    }
    const images = [_]chat_mod.ImageData{ .{ .pixels = "\x01", .width = 4, .height = 1 }, .{ .pixels = "\x02", .width = 2, .height = 1 } };
    const blocks = [_]MediaBlock{ .{ .kind = .image, .rows = 4 }, .{ .kind = .image, .rows = 2 } };
    const media = DeferredMedia{ .allocator = testing.allocator, .images = &images, .videos = &.{}, .blocks = &blocks };
    var refusal: ?MediaRefusal = null;
    try testing.expectError(error.MediaRefused, encodeMediaAfterRestore(&sch, &model, &media, 0, &refusal));
    try testing.expectEqualStrings("2 images, 10 free", refusal.?.text());
    try testing.expect(!refusal.?.server_error);
    refusal = null;
    try testing.expectError(error.MediaRefused, encodeMediaAfterRestore(&sch, &model, &media, 4, &refusal));
    try testing.expectEqualStrings("1 images, 10 free", refusal.?.text());
}
