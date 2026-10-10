//! Native streamed GLM capture; standard KLD files, independent engine provenance.
const std = @import("std");
const model = @import("model.zig");
const kld = @import("kld.zig");
const mlx = @import("mlx.zig");
const native = @import("glm5_diagnostic.zig");
const forward = @import("glm5_forward.zig");
const streaming = @import("glm5_stream.zig");
const hidden_capture = @import("hidden_capture.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Arr = mlx.mlx_array;
pub fn accepts(cfg: *const model.ModelConfig, opts: kld.Options) !bool {
    if (!cfg.isGlm5() or opts.command != .capture) return false;
    // GLM's recurrent state is KDA and always f32; the Qwen GDN arm's flag has
    // nothing to switch here, so name it rather than take it and do nothing.
    if (opts.qwen_gdn_fp32) return error.GlmTeacherHasNoQwenGdnState;
    // The FP8 release is a block quantization of the BF16 one, never a teacher.
    if (cfg.expert_layout == .fp8_individual) return error.NativeGlmTeacherRequiresLosslessStreaming;
    // A pack captures its own served path (a student reference, not the teacher) at a BF16 latent.
    if (cfg.expert_layout != .bf16_individual) {
        if (opts.kv_quant_config.isQuant()) return error.GlmStudentReferenceNeedsBf16Latent;
        return false;
    }
    if (cfg.quant_bits != 0 or
        opts.tokens == 0 or opts.ssd_budget_bytes == 0 or opts.expert_cache_bytes != 0 or opts.pick_tolerance != 0 or
        !opts.no_template or opts.enable_mtp or
        opts.kv_quant_config.scheme != .off or opts.wired_margin_bytes != 0)
        return error.NativeGlmTeacherRequiresLosslessStreaming;
    // A layer-major window ends with its prefill: no layer's cache outlives that layer.
    if (opts.layer_major and opts.tokens != 1) return error.GlmLayerMajorNeedsOneRow;
    return true;
}
pub fn tryRun(a: std.mem.Allocator, io: std.Io, opts: kld.Options, out: *kld.Out) !bool {
    if (opts.command != .capture) return false;
    var cfg = try model.parseConfig(io, a, opts.model_dir);
    defer cfg.deinit(a);
    if (!try accepts(&cfg, opts)) return false;
    try run(a, io, &cfg, opts, out);
    return true;
}
test "GLM native KLD capture takes the lossless teacher before the generic loader" {
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .expert_layout = .bf16_individual };
    const opts = kld.Options{ .command = .capture, .no_template = true, .tokens = 512, .ssd_budget_bytes = 100 << 30 };
    if (!(try accepts(&cfg, opts))) return error.MissingNativeGlmCapture;
}

test "GLM native KLD capture refuses --qwen-gdn: its KDA state is f32 already" {
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .expert_layout = .bf16_individual };
    var opts = kld.Options{ .command = .capture, .no_template = true, .tokens = 512, .ssd_budget_bytes = 100 << 30, .qwen_gdn_fp32 = true };
    try std.testing.expectError(error.GlmTeacherHasNoQwenGdnState, accepts(&cfg, opts));
    opts.qwen_gdn_fp32 = false;
    try std.testing.expect(try accepts(&cfg, opts));
}

const HeaderAudit = struct { trunk_bytes: u64 = 0, trunk_bf16: usize = 0, trunk_f32: usize = 0, expert_bf16: usize = 0, shard_stat_sha256: [64]u8 = @splat(0) };

fn textKey(key: []const u8, layers: usize) bool {
    if (std.mem.startsWith(u8, key, "model.language_model.layers.")) {
        const tail = key["model.language_model.layers.".len..];
        const end = std.mem.indexOfScalar(u8, tail, '.') orelse return false;
        if ((std.fmt.parseInt(usize, tail[0..end], 10) catch return false) >= layers) return false;
    }
    return std.mem.indexOf(u8, key, ".mtp.") == null and
        (std.mem.startsWith(u8, key, "model.language_model.") or std.mem.startsWith(u8, key, "lm_head."));
}
fn expertKey(key: []const u8) bool {
    return std.mem.indexOf(u8, key, ".mlp.experts.") != null or std.mem.indexOf(u8, key, ".mlp.switch_mlp.") != null;
}
fn auditTensor(value: std.json.Value, expert: bool, file_bytes: u64, base: u64, audit: *HeaderAudit) !void {
    if (value != .object) return error.InvalidGlmTeacherHeader;
    const dtype = value.object.get("dtype") orelse return error.InvalidGlmTeacherHeader;
    if (dtype != .string) return error.InvalidGlmTeacherHeader;
    const bf16 = std.mem.eql(u8, dtype.string, "BF16");
    const fp32 = std.mem.eql(u8, dtype.string, "F32");
    if ((!bf16 and !fp32) or (expert and !bf16)) return error.NativeGlmTeacherUnsupportedDtype;
    const shape = value.object.get("shape") orelse return error.InvalidGlmTeacherHeader;
    if (shape != .array) return error.InvalidGlmTeacherHeader;
    var elements: u64 = 1;
    for (shape.array.items) |dim| {
        if (dim != .integer or dim.integer <= 0) return error.InvalidGlmTeacherHeader;
        elements = try std.math.mul(u64, elements, @intCast(dim.integer));
    }
    const offsets = value.object.get("data_offsets") orelse return error.InvalidGlmTeacherHeader;
    if (offsets != .array or offsets.array.items.len != 2) return error.InvalidGlmTeacherHeader;
    const start = offsets.array.items[0];
    const end = offsets.array.items[1];
    if (start != .integer or end != .integer or start.integer < 0 or end.integer < start.integer) return error.InvalidGlmTeacherHeader;
    const bytes: u64 = @intCast(end.integer - start.integer);
    if (bytes != try std.math.mul(u64, elements, if (bf16) 2 else 4) or
        try std.math.add(u64, base, @intCast(end.integer)) > file_bytes) return error.InvalidGlmTeacherHeader;
    if (expert) audit.expert_bf16 += 1 else {
        audit.trunk_bytes = try std.math.add(u64, audit.trunk_bytes, bytes);
        if (bf16) audit.trunk_bf16 += 1 else audit.trunk_f32 += 1;
    }
}

/// Read only index-owned text tensor headers before the lazy loader can evaluate.
fn auditHeaders(a: std.mem.Allocator, io: std.Io, directory: []const u8, cfg: *const model.ModelConfig, check_tensors: bool) !HeaderAudit {
    var dir = try std.Io.Dir.openDirAbsolute(io, directory, .{});
    defer dir.close(io);
    const raw = try dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 * 1024 * 1024));
    defer a.free(raw);
    const index = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer index.deinit();
    if (index.value != .object) return error.InvalidGlmWeightIndex;
    const wm = index.value.object.get("weight_map") orelse return error.InvalidGlmWeightIndex;
    if (wm != .object) return error.InvalidGlmWeightIndex;
    var files = std.StringHashMap(void).init(a);
    defer files.deinit();
    var entries = wm.object.iterator();
    while (entries.next()) |entry| {
        if (!textKey(entry.key_ptr.*, cfg.num_hidden_layers)) continue;
        const file = entry.value_ptr.*;
        if (file != .string or file.string.len == 0 or std.mem.indexOfAny(u8, file.string, "/\\") != null or std.mem.eql(u8, file.string, "..")) return error.InvalidGlmShardName;
        try files.put(file.string, {});
    }
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(a);
    var shards = files.keyIterator();
    while (shards.next()) |shard| try names.append(a, shard.*);
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, left: []const u8, right: []const u8) bool {
            return std.mem.lessThan(u8, left, right);
        }
    }.less);
    var audit: HeaderAudit = .{};
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (names.items) |shard| {
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ directory, shard }, 0);
        defer a.free(path);
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.MissingIndexedGlmWeight;
        defer _ = std.c.close(fd);
        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0 or st.size < 8) return error.InvalidGlmTeacherHeader;
        const mt = st.mtime();
        const stamp = try std.fmt.allocPrint(a, "{s}:{d}:{d}:{d}:{d}\n", .{ shard, st.ino, st.size, mt.sec, mt.nsec });
        defer a.free(stamp);
        hash.update(stamp);
        if (!check_tensors) continue;
        var length: [8]u8 = undefined;
        try @import("expert_io.zig").readExact(fd, &length, 0);
        const n = std.mem.readInt(u64, &length, .little);
        if (n == 0 or n > 128 * 1024 * 1024 or n > @as(u64, @intCast(st.size)) - 8) return error.InvalidGlmTeacherHeader;
        const header = try a.alloc(u8, @intCast(n));
        defer a.free(header);
        try @import("expert_io.zig").readExact(fd, header, 8);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, header, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidGlmTeacherHeader;
        entries = wm.object.iterator();
        while (entries.next()) |entry| {
            if (!textKey(entry.key_ptr.*, cfg.num_hidden_layers) or !std.mem.eql(u8, entry.value_ptr.string, shard)) continue;
            const tensor = parsed.value.object.get(entry.key_ptr.*) orelse return error.MissingIndexedGlmWeight;
            try auditTensor(tensor, expertKey(entry.key_ptr.*), @intCast(st.size), n + 8, &audit);
        }
    }
    if (check_tensors and (audit.trunk_bf16 + audit.trunk_f32 == 0 or audit.expert_bf16 == 0)) return error.MissingIndexedGlmWeight;
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    audit.shard_stat_sha256 = std.fmt.bytesToHex(digest, .lower);
    return audit;
}

/// Dispatches of every arm a NAX capture can take, so a fixture states which route made it.
fn armDispatches() struct { hc_prefill: usize, hc_collapse_simd32: usize, kda_value_rows: usize, kda_prefill_cluster: usize, mla_head_batch: usize, packed_attention: usize, index_scores_nax: usize, decode_b1: usize } {
    return .{
        .hc_prefill = @import("glm5_hc_prefill.zig").dispatchCount(),
        .hc_collapse_simd32 = @import("glm5_hc_collapse_simd32.zig").dispatchCount(),
        .kda_value_rows = @import("glm5_kda_value_rows.zig").dispatchCount(),
        .kda_prefill_cluster = @import("glm5_kda_prefill_cluster.zig").dispatchCount(),
        .mla_head_batch = @import("glm5_mla_prefill_batch.zig").dispatchCount(),
        .packed_attention = @import("glm5_attention_nax_packed.zig").dispatchCount(),
        .index_scores_nax = @import("glm5_indexpool_nax.zig").dispatchCount(),
        .decode_b1 = @import("glm5_attention_decode_batch.zig").b1Calls(),
    };
}

fn json(a: std.mem.Allocator, io: std.Io, directory: []const u8, name: []const u8, value: anytype) !void {
    const raw = try std.json.Stringify.valueAlloc(a, value, .{ .whitespace = .indent_2 });
    defer a.free(raw);
    var dir = try std.Io.Dir.cwd().openDir(io, directory, .{});
    defer dir.close(io);
    const tmp = try std.fmt.allocPrint(a, "{s}.tmp", .{name});
    defer a.free(tmp);
    try dir.writeFile(io, .{ .sub_path = tmp, .data = raw });
    try dir.rename(tmp, dir, name, io);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

fn normalizeTeacherTf32() !void {
    if (std.c.getenv("MLX_ENABLE_TF32")) |value| {
        if (!std.mem.eql(u8, std.mem.span(value), "0")) return error.NativeGlmTeacherRequiresTf32Off;
    } else if (setenv("MLX_ENABLE_TF32", "0", 0) != 0) return error.NativeGlmTeacherEnvironmentFailed;
}

fn teacherEnvironment() !void {
    try normalizeTeacherTf32();
    if (model.getConfigOverrides() != null) return error.NativeGlmTeacherOverrides;
    @import("glm5_model.zig").enterTeacher();
}

/// Widening BF16 to F32 is exact. Any other native dtype is refused, never repaired.
fn exportRow(s: mlx.mlx_stream, logits: Arr, row: []f32) !mlx.mlx_dtype {
    if (mlx.mlx_array_size(logits) != row.len) return error.InvalidGlmDiagnosticLogits;
    const dtype = mlx.mlx_array_dtype(logits);
    if (dtype != .bfloat16 and dtype != .float32) return error.NativeGlmTeacherUnsupportedLogits;
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_astype(&wide, logits, .float32, s));
    try mlx.check(mlx.mlx_array_eval(wide));
    @memcpy(row, (mlx.mlx_array_data_float32(wide) orelse return error.MlxArrayDataNull)[0..row.len]);
    var nonzero = false;
    for (row) |v| {
        if (!std.math.isFinite(v)) return error.NonfiniteGlmLogits;
        nonzero = nonzero or v != 0;
    }
    if (!nonzero) return error.NativeGlmTeacherZeroNormLogits;
    return dtype;
}
fn greedy(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = i;
    };
    return @intCast(best);
}
fn activeBound(limit: u64, remaining: u64) !usize {
    var active: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&active));
    if (active > limit or remaining > limit - active) return error.GlmResidentBudgetExceeded;
    return active;
}

/// What a resumed layer-major capture must match byte for byte (`layer-major.json`).
const LayerMajorRun = struct {
    schema: []const u8 = "sushi-glm-layer-major-capture-v1",
    model: []const u8,
    config_sha256: []const u8,
    index_sha256: []const u8,
    tokenizer_sha256: []const u8,
    shard_stat_sha256: []const u8,
    prompts: []const u8,
    inputs_sha256: []const u8,
    prompt_count: usize,
    tokens: u32,
    chunk: usize,
    top_k: u32,
    label: []const u8,
    hidden_out: []const u8,
    hidden_width: usize,
    hidden_boundaries: usize,
    engine_sha256: []const u8,
};

/// One line of `windows.jsonl`, written only after the window's logits and boundary rows reached the device.
const WindowRecord = struct {
    window: usize,
    id: []const u8,
    dir: []const u8,
    tokens: usize,
    token_offset: u64,
    chosen: u32,
    nll_bits: u64,
    logits_dtype: []const u8,
    ids_sha256: []const u8,
    logits_sha256: []const u8,
    hidden_sha256: []const u8,
};

fn shaHex(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn inputsSha(prompts: []const kld.Prompt, inputs: []const []u32) [64]u8 {
    var h = Sha256.init(.{});
    for (prompts, inputs) |p, ids| {
        h.update(std.mem.asBytes(&@as(u64, p.id.len)));
        h.update(p.id);
        h.update(std.mem.asBytes(&@as(u64, ids.len)));
        h.update(std.mem.sliceAsBytes(ids));
    }
    return std.fmt.bytesToHex(h.finalResult(), .lower);
}

fn engineSha(a: std.mem.Allocator, io: std.Io) ![64]u8 {
    const path = try std.process.executablePathAlloc(io, a);
    defer a.free(path);
    return @import("update.zig").fileSha256Hex(io, path);
}

fn fsyncPath(path: []const u8) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const fd = std.c.open(try std.fmt.bufPrintSentinel(&buf, "{s}", .{path}, 0), .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.GlmLayerMajorSyncFailed;
    defer _ = std.c.close(fd);
    if (std.c.fsync(fd) != 0) return error.GlmLayerMajorSyncFailed;
}

/// While `pause` exists the capture idles between batches, `<pause>.ack` saying so.
fn waitWhilePaused(a: std.mem.Allocator, io: std.Io, pause: []const u8, committed: usize, out: *kld.Out) !void {
    if (pause.len == 0) return;
    const cwd = std.Io.Dir.cwd();
    cwd.access(io, pause, .{}) catch return;
    const ack = try std.fmt.allocPrint(a, "{s}.ack", .{pause});
    defer a.free(ack);
    const body = try std.fmt.allocPrint(a, "{{\"committed_windows\":{d}}}\n", .{committed});
    defer a.free(body);
    try cwd.writeFile(io, .{ .sub_path = ack, .data = body });
    out.print("[kld] layer-major paused with {d} windows committed; remove {s} to continue\n", .{ committed, pause });
    while (true) {
        cwd.access(io, pause, .{}) catch break;
        var tick = std.c.timespec{ .sec = 0, .nsec = 250_000_000 };
        _ = std.c.nanosleep(&tick, null);
    }
    cwd.deleteFile(io, ack) catch {};
}

/// The resumable state of a layer-major capture inside `<out>.partial`: the run it belongs to and the windows it
/// committed. A rerun of the same command verifies every committed window by hash and continues after them.
const Ledger = struct {
    a: std.mem.Allocator,
    staging: []const u8,
    run: LayerMajorRun,
    arena: std.heap.ArenaAllocator,
    records: []WindowRecord = &.{},
    committed_tokens: u64 = 0,
    log_fd: std.c.fd_t = -1,

    fn open(a: std.mem.Allocator, io: std.Io, staging: []const u8, run_value: LayerMajorRun) !Ledger {
        const expected = try std.json.Stringify.valueAlloc(a, run_value, .{ .whitespace = .indent_2 });
        defer a.free(expected);
        const cwd = std.Io.Dir.cwd();
        const manifest = try std.fmt.allocPrint(a, "{s}/layer-major.json", .{staging});
        defer a.free(manifest);
        if (cwd.access(io, staging, .{})) |_| {
            const stored = cwd.readFileAlloc(io, manifest, a, .limited(1 << 20)) catch |err| return if (err == error.FileNotFound) error.GlmLayerMajorStagingUnknown else err;
            defer a.free(stored);
            if (!std.mem.eql(u8, stored, expected)) {
                @import("log.zig").err("[kld] layer-major: {s} belongs to another run.\nstored:\n{s}\nthis run:\n{s}\n", .{ staging, stored, expected });
                return error.GlmLayerMajorResumeMismatch;
            }
        } else |err| {
            if (err != error.FileNotFound) return err;
            // Only a fresh run refuses rows it did not write; a resume owns its hidden files and truncates them.
            if (run_value.hidden_out.len != 0 and try hidden_capture.holdsData(run_value.hidden_out, run_value.hidden_boundaries)) return error.GlmLayerMajorHiddenNotEmpty;
            try cwd.createDir(io, staging, .default_dir);
            const tmp = try std.fmt.allocPrint(a, "{s}.tmp", .{manifest});
            defer a.free(tmp);
            try cwd.writeFile(io, .{ .sub_path = tmp, .data = expected });
            try fsyncPath(tmp);
            try cwd.rename(tmp, cwd, manifest, io);
        }
        var self = Ledger{ .a = a, .staging = staging, .run = run_value, .arena = .init(a) };
        errdefer self.deinit();
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const log_path = try std.fmt.bufPrintSentinel(&buf, "{s}/windows.jsonl", .{staging}, 0);
        self.log_fd = std.c.open(log_path, .{ .ACCMODE = .RDWR, .CREAT = true, .APPEND = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o644));
        if (self.log_fd < 0) return error.GlmLayerMajorLogUnreadable;
        const arena = self.arena.allocator();
        const body = try cwd.readFileAlloc(io, log_path, arena, .limited(1 << 30));
        // A line without its newline was being written when the run died; it commits nothing.
        const whole = if (std.mem.lastIndexOfScalar(u8, body, '\n')) |end| end + 1 else 0;
        if (whole != body.len and std.c.ftruncate(self.log_fd, @intCast(whole)) != 0) return error.GlmLayerMajorLogUnreadable;
        var lines: std.ArrayList(WindowRecord) = .empty;
        var it = std.mem.splitScalar(u8, body[0..whole], '\n');
        while (it.next()) |line| {
            if (line.len == 0) continue;
            try lines.append(arena, std.json.parseFromSliceLeaky(WindowRecord, arena, line, .{}) catch return error.GlmLayerMajorLogCorrupt);
        }
        self.records = lines.items;
        return self;
    }

    fn deinit(self: *Ledger) void {
        if (self.log_fd >= 0) _ = std.c.close(self.log_fd);
        self.arena.deinit();
    }

    /// Every committed window must still be the same input, logits row and boundary rows.
    fn verify(self: *Ledger, io: std.Io, prompts: []const kld.Prompt, inputs: []const []u32) !void {
        var offset: u64 = 0;
        for (self.records, 0..) |r, i| {
            if (r.window != i or i >= prompts.len or !std.mem.eql(u8, r.id, prompts[i].id) or r.tokens != inputs[i].len or r.token_offset != offset) return error.GlmLayerMajorWindowChanged;
            const ids_sha = shaHex(std.mem.sliceAsBytes(inputs[i]));
            const path = try std.fmt.allocPrint(self.a, "{s}/{s}/logits.f32", .{ self.staging, r.dir });
            defer self.a.free(path);
            const logits = std.Io.Dir.cwd().readFileAlloc(io, path, self.a, .limited(64 << 20)) catch return error.GlmLayerMajorWindowChanged;
            defer self.a.free(logits);
            const logits_sha = shaHex(logits);
            if (!std.mem.eql(u8, r.ids_sha256, &ids_sha) or !std.mem.eql(u8, r.logits_sha256, &logits_sha)) return error.GlmLayerMajorWindowChanged;
            if (!try self.idListIs(io, r.dir, "prompt_tokens.txt", inputs[i]) or !try self.idListIs(io, r.dir, "generated_tokens.txt", &.{r.chosen})) return error.GlmLayerMajorWindowChanged;
            offset += r.tokens;
        }
        self.committed_tokens = offset;
        if (self.run.hidden_out.len != 0 and self.records.len != 0) try self.verifyHidden(inputs);
    }

    /// A committed window's token file, byte for byte what the fixture writer emits for `ids`; a missing one differs.
    fn idListIs(self: *Ledger, io: std.Io, dir: []const u8, leaf: []const u8, ids: []const u32) !bool {
        const path = try std.fmt.allocPrint(self.a, "{s}/{s}/{s}", .{ self.staging, dir, leaf });
        defer self.a.free(path);
        const stored = std.Io.Dir.cwd().readFileAlloc(io, path, self.a, .limited(64 << 20)) catch return false;
        defer self.a.free(stored);
        var want: std.Io.Writer.Allocating = .init(self.a);
        defer want.deinit();
        try kld.writeIdList(&want.writer, ids);
        return std.mem.eql(u8, stored, want.written());
    }

    fn verifyHidden(self: *Ledger, inputs: []const []u32) !void {
        const Check = struct {
            ledger: *const Ledger,
            inputs: []const []u32,
            fds: []std.c.fd_t,
            changed: std.atomic.Value(bool) = .init(false),
            fn worker(c: *@This(), first: usize, step: usize) void {
                const buf = std.heap.page_allocator.alloc(u8, 8 << 20) catch return c.changed.store(true, .release);
                defer std.heap.page_allocator.free(buf);
                var i = first;
                while (i < c.ledger.records.len and !c.changed.load(.acquire)) : (i += step) {
                    if (!(c.window(i, buf) catch false)) c.changed.store(true, .release);
                }
            }
            fn window(c: *@This(), i: usize, buf: []u8) !bool {
                const r = c.ledger.records[i];
                const row_bytes: u64 = c.ledger.run.hidden_width * 2;
                var h = Sha256.init(.{});
                for (c.fds[0 .. c.fds.len - 1]) |fd| {
                    var at = r.token_offset * row_bytes;
                    const end = at + r.tokens * row_bytes;
                    while (at < end) {
                        const n: usize = @intCast(@min(buf.len, end - at));
                        try @import("expert_io.zig").readExact(fd, buf[0..n], at);
                        h.update(buf[0..n]);
                        at += n;
                    }
                }
                const digest = std.fmt.bytesToHex(h.finalResult(), .lower);
                const ids = std.mem.sliceAsBytes(c.inputs[i]);
                try @import("expert_io.zig").readExact(c.fds[c.fds.len - 1], buf[0..ids.len], r.token_offset * 4);
                return std.mem.eql(u8, buf[0..ids.len], ids) and std.mem.eql(u8, r.hidden_sha256, &digest);
            }
        };
        const fds = try self.a.alloc(std.c.fd_t, self.run.hidden_boundaries + 1);
        defer self.a.free(fds);
        @memset(fds, -1);
        defer for (fds) |fd| if (fd >= 0) {
            _ = std.c.close(fd);
        };
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        for (fds, 0..) |*fd, b| {
            const path = if (b == self.run.hidden_boundaries) try std.fmt.bufPrintSentinel(&buf, "{s}/tokens.bin", .{self.run.hidden_out}, 0) else try std.fmt.bufPrintSentinel(&buf, "{s}/boundary-{d:0>2}.bin", .{ self.run.hidden_out, b }, 0);
            fd.* = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
            if (fd.* < 0) return error.GlmLayerMajorWindowChanged;
        }
        var check = Check{ .ledger = self, .inputs = inputs, .fds = fds };
        var threads: [8]std.Thread = undefined;
        const count = @min(threads.len, self.records.len);
        var spawned: usize = 0;
        defer for (threads[0..spawned]) |t| t.join();
        while (spawned < count) : (spawned += 1) threads[spawned] = try std.Thread.spawn(.{}, Check.worker, .{ &check, spawned, count });
        for (threads[0..spawned]) |t| t.join();
        spawned = 0;
        if (check.changed.load(.acquire)) return error.GlmLayerMajorWindowChanged;
    }

    const Batch = struct {
        io: std.Io,
        s: mlx.mlx_stream,
        net: *const forward.Model,
        prompts: []const kld.Prompt,
        inputs: []const []u32,
        chunk: usize,
        batch: u32,
        hidden: ?*hidden_capture.Writer,
        row: []f32,
        limit: usize,
        pause_file: []const u8,
        out: *kld.Out,
    };

    /// Runs every uncommitted window, a batch at a time, and leaves `records` with every window's fixture record.
    fn capture(self: *Ledger, b: Batch, records: *std.ArrayList(kld.PromptRecord), logits_dtype: *?mlx.mlx_dtype) !void {
        const a = self.a;
        for (self.records) |r| {
            const dtype = std.meta.stringToEnum(mlx.mlx_dtype, r.logits_dtype) orelse return error.GlmLayerMajorLogCorrupt;
            if (logits_dtype.*) |prior| if (prior != dtype) return error.NativeGlmTeacherLogitsDtypeChanged;
            logits_dtype.* = dtype;
            const nll: f64 = @bitCast(r.nll_bits);
            const id = try a.dupe(u8, r.id);
            errdefer a.free(id);
            try records.append(a, .{ .id = id, .dir = try a.dupe(u8, r.dir), .prompt_tokens = r.tokens, .generated_tokens = 1, .strict_nll_mean = nll, .strict_perplexity = @exp(nll) });
        }
        if (b.hidden) |w| try w.truncateTo(self.committed_tokens);
        var next = self.records.len;
        var offset = self.committed_tokens;
        const run_clock = @import("expert_stream.zig").Clock.init();
        var run_tokens: u64 = 0;
        var remaining: u64 = 0;
        for (b.inputs[next..]) |ids| remaining += ids.len;
        // A full disk would stop the run mid-batch; refuse before the first batch instead, with a 16 GiB margin.
        const needed = layerMajorDiskBytes(remaining, b.prompts.len - next, if (b.hidden != null) self.run.hidden_boundaries else 0, self.run.hidden_width, b.row.len);
        if (@import("kv_disk_cache.zig").volumeSpace(if (b.hidden != null) self.run.hidden_out else self.staging)) |space| if (space.free < needed + (16 << 30)) {
            @import("log.zig").err("[kld] layer-major: {d} GB still to write, {d} GB free (16 GiB margin)\n", .{ needed / 1_000_000_000, space.free / 1_000_000_000 });
            return error.GlmLayerMajorDiskTooSmall;
        };
        while (next < b.prompts.len) {
            try waitWhilePaused(a, b.io, b.pause_file, next, b.out);
            const end = @min(b.prompts.len, next + b.batch);
            var sink = try BatchSink.init(a, b, self.staging, next, end, logits_dtype);
            defer sink.deinit();
            const stats = try @import("glm5_layer_major.zig").prefill(a, b.net, b.inputs[next..end], b.chunk, sink.sink());
            if (b.hidden) |w| {
                for (b.inputs[next..end]) |ids| try w.appendTokens(ids);
                try w.sync();
            }
            var tokens: u64 = 0;
            for (sink.results, next..) |*res, i| {
                const record = res.record orelse return error.NativeGlmTeacherRowCountMismatch;
                for ([_][]const u8{ "logits.f32", "prompt_tokens.txt", "generated_tokens.txt" }) |leaf| {
                    const path = try std.fmt.allocPrint(a, "{s}/{s}/{s}", .{ self.staging, record.dir, leaf });
                    defer a.free(path);
                    try fsyncPath(path);
                }
                const ids_sha = shaHex(std.mem.sliceAsBytes(b.inputs[i]));
                const hidden_sha = std.fmt.bytesToHex(sink.hashes[i - next].finalResult(), .lower);
                const line = try std.json.Stringify.valueAlloc(a, WindowRecord{
                    .window = i,
                    .id = record.id,
                    .dir = record.dir,
                    .tokens = b.inputs[i].len,
                    .token_offset = offset,
                    .chosen = res.chosen,
                    .nll_bits = @bitCast(record.strict_nll_mean),
                    .logits_dtype = @tagName(res.dtype),
                    .ids_sha256 = &ids_sha,
                    .logits_sha256 = &res.logits_sha256,
                    .hidden_sha256 = if (b.hidden != null) &hidden_sha else "",
                }, .{});
                defer a.free(line);
                try appendLine(self.log_fd, line);
                offset += b.inputs[i].len;
                tokens += b.inputs[i].len;
            }
            if (std.c.fsync(self.log_fd) != 0) return error.GlmLayerMajorSyncFailed;
            for (sink.results) |*res| {
                try records.append(a, res.record.?);
                res.record = null;
            }
            const wall = @as(f64, @floatFromInt(stats.wall_ns)) / 1e9;
            const fill = @as(f64, @floatFromInt(stats.fill_ns)) / 1e9;
            run_tokens += tokens;
            remaining -= tokens;
            const rate = @as(f64, @floatFromInt(run_tokens)) / (@as(f64, @floatFromInt(run_clock.start.untilNow(run_clock.io, .boot).nanoseconds)) / 1e9);
            try json(a, b.io, self.staging, "progress.json", .{ .complete = false, .phase = "layer-major", .committed_windows = end, .total_windows = b.prompts.len, .batch_windows = b.batch, .batch_seconds = wall, .batch_fill_seconds = fill, .batch_fill_bytes = stats.fill_bytes, .batch_tokens = tokens, .tokens_per_second = rate, .eta_hours = @as(f64, @floatFromInt(remaining)) / rate / 3600 });
            b.out.print("[kld] layer-major windows {d}..{d} of {d}: {d} tokens in {d:.1} s ({d:.2} tok/s), read {d:.1} GB in {d:.1} s; this run {d:.2} tok/s, {d:.2} h left\n", .{ next, end, b.prompts.len, tokens, wall, @as(f64, @floatFromInt(tokens)) / wall, @as(f64, @floatFromInt(stats.fill_bytes)) / 1e9, fill, rate, @as(f64, @floatFromInt(remaining)) / rate / 3600 });
            next = end;
        }
    }
};

/// Bytes the remaining windows still write: every boundary row, their token ids and one F32 logits row each.
fn layerMajorDiskBytes(tokens: u64, windows: u64, boundaries: u64, width: u64, vocab: u64) u64 {
    return tokens * (boundaries * width * 2 + 4) + windows * vocab * 4;
}

fn appendLine(fd: std.c.fd_t, line: []const u8) !void {
    for ([_][]const u8{ line, "\n" }) |part| {
        var done: usize = 0;
        while (done < part.len) {
            const got = std.c.write(fd, part[done..].ptr, part.len - done);
            if (got <= 0) return error.GlmLayerMajorLogWriteFailed;
            done += @intCast(got);
        }
    }
}

/// Receives one layer-major batch: boundary rows go to the hidden capture as they are made, each window's
/// logits row becomes its fixture directory.
const BatchSink = struct {
    a: std.mem.Allocator,
    b: Ledger.Batch,
    staging: []const u8,
    first: usize,
    logits_dtype: *?mlx.mlx_dtype,
    hashes: []Sha256,
    results: []Result,

    const Result = struct { record: ?kld.PromptRecord = null, chosen: u32 = 0, logits_sha256: [64]u8 = undefined, dtype: mlx.mlx_dtype = .float32 };

    fn init(a: std.mem.Allocator, b: Ledger.Batch, staging: []const u8, first: usize, end: usize, logits_dtype: *?mlx.mlx_dtype) !BatchSink {
        const hashes = try a.alloc(Sha256, end - first);
        errdefer a.free(hashes);
        for (hashes) |*h| h.* = Sha256.init(.{});
        const results = try a.alloc(Result, end - first);
        @memset(results, .{});
        return .{ .a = a, .b = b, .staging = staging, .first = first, .logits_dtype = logits_dtype, .hashes = hashes, .results = results };
    }
    fn deinit(self: *BatchSink) void {
        for (self.results) |r| if (r.record) |record| record.deinit(self.a);
        self.a.free(self.results);
        self.a.free(self.hashes);
    }
    fn sink(self: *BatchSink) @import("glm5_layer_major.zig").Sink {
        return .{ .ctx = self, .boundary = boundary, .logits = logits };
    }
    fn boundary(ctx: *anyopaque, window: usize, index: usize, rows: Arr) anyerror!void {
        const self: *BatchSink = @ptrCast(@alignCast(ctx));
        if (@import("builtin").is_test) if (interrupt_boundaries_for_test) |*left| {
            if (left.* == 0) return error.TestInterrupted;
            left.* -= 1;
        };
        if (self.b.hidden) |w| _ = try w.appendRows(self.b.s, index, rows, &self.hashes[window]);
    }
    fn logits(ctx: *anyopaque, window: usize, value: Arr) anyerror!void {
        const self: *BatchSink = @ptrCast(@alignCast(ctx));
        const i = self.first + window;
        const dtype = try exportRow(self.b.s, value, self.b.row);
        if (self.logits_dtype.*) |prior| {
            if (prior != dtype) return error.NativeGlmTeacherLogitsDtypeChanged;
        } else self.logits_dtype.* = dtype;
        const chosen = greedy(self.b.row);
        const p = self.b.prompts[i];
        var writer = try kld.PromptWriter.begin(self.a, self.b.io, self.staging, i, p.id);
        defer writer.deinit();
        try writer.appendRow(self.b.row, chosen);
        self.results[window] = .{ .record = try writer.finish(p.text, p.text, self.b.inputs[i], &[_]u32{chosen}), .chosen = chosen, .logits_sha256 = shaHex(std.mem.sliceAsBytes(self.b.row)), .dtype = dtype };
        _ = try activeBound(self.b.limit, 0);
    }
};

fn run(a: std.mem.Allocator, io: std.Io, cfg: *const model.ModelConfig, opts: kld.Options, out: *kld.Out) !void {
    try teacherEnvironment();
    out.print("[glm] NAX arms {s}\n", .{if (@import("glm5_model.zig").naxArms()) "on: B1 decode attention, HC and KDA fused prefill, MLA head batches, packed sparse attention and NAX index scores past 2051 tokens" else "off: reference arms (scalar latent attention, staged HC and KDA)"});
    const cwd = std.Io.Dir.cwd();
    if (cwd.access(io, opts.out_dir, .{})) |_| return error.NativeGlmTeacherOutputExists else |err| if (err != error.FileNotFound) return err;
    const staging = try std.fmt.allocPrint(a, "{s}.partial", .{opts.out_dir});
    defer a.free(staging);
    if (!opts.layer_major) try cwd.createDir(io, staging, .default_dir);
    var prompts = try kld.loadPrompts(a, io, opts.prompts, opts.limit);
    defer prompts.deinit();
    if (prompts.items.len == 0) return error.NoPromptsFound;
    var tok = try @import("tokenizer.zig").loadTokenizer(io, a, opts.model_dir);
    defer tok.deinit();
    const inputs = try a.alloc([]u32, prompts.items.len);
    defer a.free(inputs);
    var encoded: usize = 0;
    defer for (inputs[0..encoded]) |ids| a.free(ids);
    var max_tokens: usize = 0;
    for (prompts.items, inputs) |p, *ids| {
        ids.* = if (p.ids) |given| try a.dupe(u32, given) else try tok.encode(a, p.text);
        encoded += 1;
        if (ids.len == 0) return error.EmptyPrompt;
        for (ids.*) |id| if (id >= cfg.vocab_size) return error.InvalidGlmPrompt;
        const capacity = try std.math.add(usize, ids.len, opts.tokens);
        if (capacity > cfg.max_position_embeddings or (opts.ctx_size != 0 and capacity > opts.ctx_size)) return error.GlmContextExceeded;
        max_tokens = @max(max_tokens, capacity);
    }
    const chunk = @min(@as(usize, 512), max_tokens);
    const reserve_floor: u64 = if (@import("builtin").is_test) reserve_floor_for_test else 8 << 30;
    const reserve = @max(reserve_floor, try streaming.minimumReserve(cfg, max_tokens, chunk) + try streaming.armsReserve(cfg, chunk));
    const config_sha = try metadataHash(a, io, opts.model_dir, "config.json");
    const index_sha = try metadataHash(a, io, opts.model_dir, "model.safetensors.index.json");
    const tokenizer_sha = try metadataHash(a, io, opts.model_dir, "tokenizer.json");
    const headers = try auditHeaders(a, io, opts.model_dir, cfg, true);
    const batch: u32 = if (!opts.layer_major) 0 else if (opts.batch_windows != 0) opts.batch_windows else try streaming.layerMajorWindows(cfg, opts.ssd_budget_bytes, headers.trunk_bytes, reserve, max_tokens, chunk, 32);
    const planned = if (opts.layer_major) try streaming.layerMajorBudget(cfg, opts.ssd_budget_bytes, headers.trunk_bytes, reserve, max_tokens, chunk, batch) else null;
    const trunk_limit = try streaming.trunkLimit(cfg, opts.ssd_budget_bytes, reserve + if (planned) |p| p.carried else 0);
    if (headers.trunk_bytes > trunk_limit) return error.GlmResidentBudgetExceeded;
    const inputs_sha = inputsSha(prompts.items, inputs);
    const engine_sha: [64]u8 = if (opts.layer_major) try engineSha(a, io) else @splat('0');
    var ledger: ?Ledger = if (opts.layer_major) try Ledger.open(a, io, staging, .{
        .model = opts.model_dir,
        .config_sha256 = &config_sha,
        .index_sha256 = &index_sha,
        .tokenizer_sha256 = &tokenizer_sha,
        .shard_stat_sha256 = &headers.shard_stat_sha256,
        .prompts = opts.prompts,
        .inputs_sha256 = &inputs_sha,
        .prompt_count = prompts.items.len,
        .tokens = opts.tokens,
        .chunk = chunk,
        .top_k = opts.top_k,
        .label = opts.label,
        .hidden_out = opts.hidden_out,
        .hidden_width = kld.hiddenCaptureWidth(cfg),
        .hidden_boundaries = cfg.num_hidden_layers + 1,
        .engine_sha256 = &engine_sha,
    }) else null;
    defer if (ledger) |*l| l.deinit();
    if (ledger) |*l| try l.verify(io, prompts.items, inputs);
    try json(a, io, staging, "source-header-audit.json", headers);
    try json(a, io, staging, "progress.json", .{ .complete = false, .phase = "validated", .prompts = prompts.items.len, .tokens_per_prompt = opts.tokens });
    const limit: usize = std.math.cast(usize, opts.ssd_budget_bytes) orelse return error.InvalidGlmStreamBudget;
    const recommended = mlx.maxRecommendedWorkingSet();
    if (recommended == 0 or limit > recommended) return error.InvalidGlmWiredLimit;
    var old_memory: usize = 0;
    var old_cache: usize = 0;
    var old_wired: usize = 0;
    var ignored: usize = 0;
    try mlx.check(mlx.mlx_set_memory_limit(&old_memory, limit));
    defer _ = mlx.mlx_set_memory_limit(&ignored, old_memory);
    try mlx.check(mlx.mlx_set_cache_limit(&old_cache, 0));
    defer _ = mlx.mlx_set_cache_limit(&ignored, old_cache);
    try mlx.check(mlx.mlx_set_wired_limit(&old_wired, limit));
    defer _ = mlx.mlx_set_wired_limit(&ignored, old_wired);
    try mlx.check(mlx.mlx_clear_cache());
    try mlx.check(mlx.mlx_reset_peak_memory());
    const s = mlx.gpuStream();
    const started = std.Io.Timestamp.now(io, .awake);
    try json(a, io, staging, "progress.json", .{ .complete = false, .phase = "loading" });
    var weights = try native.loadWeightsBounded(io, a, opts.model_dir, s, true, trunk_limit);
    defer weights.deinit();
    const payload = native.storedBytes(&weights);
    if (payload != headers.trunk_bytes) return error.NativeGlmTeacherStoredBytesChanged;
    const budget = planned orelse try streaming.captureBudget(cfg, opts.ssd_budget_bytes, payload, reserve, max_tokens, chunk);
    var engine = try @import("expert_stream.zig").Engine.initWithOptions(a, opts.model_dir, cfg.expertGeometry(), budget.cache, s, .{ .layout = .bf16_individual });
    defer engine.deinit();
    var net = try forward.Model.loadStreamed(a, cfg.*, &weights, s, .{ .engine = &engine, .max_tokens = max_tokens, .max_chunk = chunk });
    defer net.deinit();
    const loaded_active = try activeBound(limit, reserve + budget.carried);
    const hidden = if (opts.hidden_out.len > 0) try hidden_capture.Writer.open(a, io, opts.hidden_out, cfg.num_hidden_layers, kld.hiddenCaptureWidth(cfg)) else null;
    defer if (hidden) |w| w.close();
    var records: std.ArrayList(kld.PromptRecord) = .empty;
    defer {
        for (records.items) |r| r.deinit(a);
        records.deinit(a);
    }
    const row = try a.alloc(f32, cfg.vocab_size);
    defer a.free(row);
    var logits_dtype: ?mlx.mlx_dtype = null;
    var final_offset: usize = 0;
    if (ledger) |*l| {
        try l.capture(.{ .io = io, .s = s, .net = &net, .prompts = prompts.items, .inputs = inputs, .chunk = chunk, .batch = batch, .hidden = hidden, .row = row, .limit = limit, .pause_file = opts.pause_file, .out = out }, &records, &logits_dtype);
        final_offset = inputs[inputs.len - 1].len;
    } else try windowMajor(a, io, s, &net, staging, prompts.items, inputs, chunk, opts.tokens, hidden, row, limit, &records, &logits_dtype, &final_offset);
    const elapsed = @as(f64, @floatFromInt(started.untilNow(io, .awake).nanoseconds)) / 1e9;
    try kld.writeBaseline(a, io, staging, .{ .label = opts.label, .model = opts.model_dir, .run = opts.label, .kv_cache_format = "bf16", .inference_profile = "greedy-native-glm-streamed", .prompt_set = opts.prompts, .ssd_budget_gb = opts.ssd_budget_bytes >> 30, .tokens_per_prompt = opts.tokens, .top_k = opts.top_k, .elapsed_secs = elapsed }, records.items);
    var peak: usize = 0;
    var cached: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    try mlx.check(mlx.mlx_get_cache_memory(&cached));
    const active = try activeBound(limit, 0);
    if (!std.mem.eql(u8, &config_sha, &try metadataHash(a, io, opts.model_dir, "config.json")) or
        !std.mem.eql(u8, &index_sha, &try metadataHash(a, io, opts.model_dir, "model.safetensors.index.json")) or
        !std.mem.eql(u8, &tokenizer_sha, &try metadataHash(a, io, opts.model_dir, "tokenizer.json"))) return error.NativeGlmTeacherSourceChanged;
    const final_shards = try auditHeaders(a, io, opts.model_dir, cfg, false);
    if (!std.mem.eql(u8, &headers.shard_stat_sha256, &final_shards.shard_stat_sha256)) return error.NativeGlmTeacherSourceChanged;
    try completeBaseline(a, io, staging, records.items.len, opts.tokens);
    if (cached != 0 or peak > limit) return error.GlmResidentBudgetExceeded;
    try json(a, io, staging, "identity.json", .{ .schema = "sushi-native-glm-capture-v1", .complete = true, .engine = "sushi-native-glm", .model = opts.model_dir, .source_storage = "indexed BF16/F32 trunk; individual BF16 experts, as stored", .config_sha256 = &config_sha, .index_sha256 = &index_sha, .tokenizer_sha256 = &tokenizer_sha, .kda_unary_modes = try @import("glm5_kda_fused.zig").unaryModes(s), .kda_body_dispatches = @import("glm5_kda_fused.zig").dispatchCount(), .kda_post_dispatches = @import("glm5_kda_fused.zig").postDispatchCount(), .kda_prework_dispatches = @import("glm5_kda_prework.zig").dispatchCount(), .trunk_header_audit = headers, .shard_stat_sha256 = &headers.shard_stat_sha256, .logits_dtype = @tagName(logits_dtype.?), .logits_export = "exact full-vocabulary little-endian float32", .vocab_size = cfg.vocab_size, .tokens_per_prompt = opts.tokens, .prompt_count = records.items.len, .prefix_chunk = chunk, .final_request_offset = final_offset, .kv_cache_format = "bf16", .kda_state_format = "float32", .dense_prefill = true, .synchronous_layers = true, .mtp = false, .dflash = false, .tf32 = false, .nax_arms = @import("glm5_model.zig").naxArms(), .reference_numerics = @import("glm5_model.zig").reference_numerics, .arm_dispatches = armDispatches(), .template = false, .prefix_reuse = false, .layer_major = opts.layer_major, .batch_windows = batch, .hidden_out = opts.hidden_out, .hidden_width = if (hidden != null) kld.hiddenCaptureWidth(cfg) else 0, .stream_budget = budget, .stream_cache_slots = engine.plan.slots_per_layer, .stream_fill_bytes = engine.fill_bytes_total, .loaded_active_bytes = loaded_active, .active_bytes = active, .peak_bytes = peak, .memory_limit_bytes = limit, .wired_limit_bytes = limit, .allocator_cache_bytes = cached, .elapsed_seconds = elapsed });
    try json(a, io, staging, "progress.json", .{ .complete = true, .phase = "complete", .completed_prompts = records.items.len, .completed_rows = records.items.len * opts.tokens });
    try cwd.renamePreserve(staging, cwd, opts.out_dir, io);
    out.print("[kld] native GLM captured {d} prompts x {d} full-vocabulary rows into {s}\n", .{ records.items.len, opts.tokens, opts.out_dir });
}

/// One prompt after another: prefill chunks, then greedy rows; a hidden capture takes the prefill's boundaries.
fn windowMajor(a: std.mem.Allocator, io: std.Io, s: mlx.mlx_stream, net: *const forward.Model, staging: []const u8, prompts: []const kld.Prompt, inputs: []const []u32, chunk: usize, tokens: u32, hidden: ?*hidden_capture.Writer, row: []f32, limit: usize, records: *std.ArrayList(kld.PromptRecord), logits_dtype: *?mlx.mlx_dtype, final_offset: *usize) !void {
    var request = try forward.Request.init(a, net.layers.len);
    defer request.deinit();
    request.dense_prefill = true;
    request.prefill_async = false;
    request.decode_async = false;
    request.prefill_sync_layers = 1;
    const generated = try a.alloc(u32, tokens);
    defer a.free(generated);
    const Rows = struct {
        writer: *hidden_capture.Writer,
        s: mlx.mlx_stream,
        fn append(ctx: *anyopaque, boundary: usize, rows: Arr) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            _ = try self.writer.appendRows(self.s, boundary, rows, null);
        }
    };
    for (prompts, inputs, 0..) |p, ids, index| {
        request.reset();
        var writer = try kld.PromptWriter.begin(a, io, staging, index, p.id);
        defer writer.deinit();
        var rows: Rows = undefined;
        if (hidden) |w| {
            rows = .{ .writer = w, .s = s };
            request.boundaries = .{ .ctx = &rows, .append = Rows.append };
        }
        defer request.boundaries = null;
        var logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(logits);
        var cursor: usize = 0;
        while (cursor < ids.len) {
            const end = @min(ids.len, cursor + chunk);
            const input = mlx.mlx_array_new_data(ids[cursor..end].ptr, &.{ 1, @intCast(end - cursor) }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const next = try net.forwardLast(&request, input, true);
            _ = mlx.mlx_array_free(logits);
            logits = next;
            try mlx.check(mlx.mlx_array_eval(logits));
            _ = try activeBound(limit, 0);
            cursor = end;
            try json(a, io, staging, "progress.json", .{ .complete = false, .phase = "prefill", .prompt = p.id, .prefill_tokens = cursor, .completed_rows = 0 });
        }
        request.boundaries = null;
        if (hidden) |w| try w.appendTokens(ids);
        for (generated, 0..) |*chosen, step| {
            const dtype = try exportRow(s, logits, row);
            if (logits_dtype.*) |prior| {
                if (prior != dtype) return error.NativeGlmTeacherLogitsDtypeChanged;
            } else logits_dtype.* = dtype;
            chosen.* = greedy(row);
            try writer.appendRow(row, chosen.*);
            _ = try activeBound(limit, 0);
            try json(a, io, staging, "progress.json", .{ .complete = false, .phase = "capture", .prompt = p.id, .completed_prompts = index, .completed_rows = step + 1, .total_rows = tokens, .request_offset = request.offset });
            if (step + 1 == generated.len) break;
            const input = mlx.mlx_array_new_data(chosen, &.{ 1, 1 }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const next = try net.forwardLast(&request, input, true);
            _ = mlx.mlx_array_free(logits);
            logits = next;
        }
        if (writer.rows != tokens or request.offset != ids.len + tokens - 1) return error.NativeGlmTeacherRowCountMismatch;
        final_offset.* = request.offset;
        const record = try writer.finish(p.text, p.text, ids, generated);
        errdefer record.deinit(a);
        try records.append(a, record);
    }
}

test "GLM native KLD capture rejects lossy options" {
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .expert_layout = .bf16_individual };
    var opts = kld.Options{ .command = .capture, .no_template = true, .tokens = 512, .ssd_budget_bytes = 100 << 30 };
    opts.enable_mtp = true;
    try std.testing.expectError(error.NativeGlmTeacherRequiresLosslessStreaming, accepts(&cfg, opts));
    opts.enable_mtp = false;
    opts.expert_cache_bytes = 1;
    try std.testing.expectError(error.NativeGlmTeacherRequiresLosslessStreaming, accepts(&cfg, opts));
    opts.expert_cache_bytes = 0;
    opts.kv_quant_config = @import("transformer.zig").KVQuantConfig.affine(8);
    try std.testing.expectError(error.NativeGlmTeacherRequiresLosslessStreaming, accepts(&cfg, opts));
    opts.command = .compare;
    try std.testing.expect(!try accepts(&cfg, opts));
}

test "GLM native KLD capture CPU header audit refuses nonteacher storage" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "BF16", "F32", "F16", "U16", "F8_E4M3" }) |dtype| {
        const raw = try std.fmt.allocPrint(a, "{{\"dtype\":\"{s}\",\"shape\":[2],\"data_offsets\":[0,{d}]}}", .{ dtype, if (std.mem.eql(u8, dtype, "F32")) @as(u8, 8) else 4 });
        defer a.free(raw);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
        defer parsed.deinit();
        var audit: HeaderAudit = .{};
        if (std.mem.eql(u8, dtype, "BF16") or std.mem.eql(u8, dtype, "F32")) {
            try auditTensor(parsed.value, false, 64, 8, &audit);
            try std.testing.expect(audit.trunk_bytes == 4 or audit.trunk_bytes == 8);
        } else try std.testing.expectError(error.NativeGlmTeacherUnsupportedDtype, auditTensor(parsed.value, false, 64, 8, &audit));
        if (std.mem.eql(u8, dtype, "BF16")) {
            try auditTensor(parsed.value, true, 64, 8, &audit);
            try std.testing.expectEqual(@as(usize, 1), audit.expert_bf16);
        } else try std.testing.expectError(error.NativeGlmTeacherUnsupportedDtype, auditTensor(parsed.value, true, 64, 8, &audit));
    }
    try std.testing.expect(!textKey("model.language_model.layers.40.mlp.gate.weight", 40));
    try std.testing.expect(!textKey("model.language_model.mtp.weight", 40));
    try std.testing.expect(!textKey("model.visual.weight", 40));
}

test "GLM native KLD capture CPU exact logits export and greedy full vocabulary" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const bits = [_]u16{ 0xbf80, 0x3f80, 0x4080, 0x4080, 0x0001 };
    const x = mlx.mlx_array_new_data(&bits, &.{ 1, 1, 5 }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    var row: [5]f32 = undefined;
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, try exportRow(s, x, &row));
    for (bits, row) |b, f| try std.testing.expectEqual(@as(u32, b) << 16, @as(u32, @bitCast(f)));
    try std.testing.expectEqual(@as(u32, 2), greedy(&row));
    const half = [_]f16{ 1, 2 };
    const y = mlx.mlx_array_new_data(&half, &.{ 1, 1, 2 }, 3, .float16);
    defer _ = mlx.mlx_array_free(y);
    try std.testing.expectError(error.NativeGlmTeacherUnsupportedLogits, exportRow(s, y, row[0..2]));
}

fn metadataHash(a: std.mem.Allocator, io: std.Io, directory: []const u8, name: []const u8) ![64]u8 {
    var dir = try std.Io.Dir.openDirAbsolute(io, directory, .{});
    defer dir.close(io);
    const bytes = try dir.readFileAlloc(io, name, a, .limited(128 * 1024 * 1024));
    defer a.free(bytes);
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    return std.fmt.bytesToHex(hash, .lower);
}
fn completeBaseline(a: std.mem.Allocator, io: std.Io, directory: []const u8, count: usize, tokens: u32) !void {
    var dir = try std.Io.Dir.cwd().openDir(io, directory, .{});
    defer dir.close(io);
    const raw = try dir.readFileAlloc(io, "baseline.json", a, .limited(16 * 1024 * 1024));
    defer a.free(raw);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadBaselineJson;
    const arena = parsed.arena.allocator();
    try parsed.value.object.put(arena, "complete", .{ .bool = true });
    try parsed.value.object.put(arena, "requested_prompt_count", .{ .integer = @intCast(count) });
    try parsed.value.object.put(arena, "completed_prompt_count", .{ .integer = @intCast(count) });
    try parsed.value.object.put(arena, "requested_tokens_per_prompt", .{ .integer = tokens });
    try parsed.value.object.put(arena, "actual_positions", .{ .integer = @intCast(count * tokens) });
    try parsed.value.object.put(arena, "human_truncated", .{ .bool = false });
    try parsed.value.object.put(arena, "quality_scope", .{ .string = "Requested prompt study; full fixed-length capture. The standard release verdict requires sixteen prompts." });
    try json(a, io, directory, "baseline.json", parsed.value);
}

test "GLM native KLD capture CPU rejects zero norm logits" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const data = [_]f32{ 0, 0 };
    const x = mlx.mlx_array_new_data(&data, &.{ 1, 1, 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(x);
    var row: [2]f32 = undefined;
    try std.testing.expectError(error.NativeGlmTeacherZeroNormLogits, exportRow(s, x, &row));
}

test "GLM teacher capture takes the NAX arms on a NAX GPU and the reference arms without one" {
    const a = std.testing.allocator;
    const base = @import("glm5_model.zig");
    const transformer = @import("transformer.zig");
    const saved = if (std.c.getenv("MLX_ENABLE_TF32")) |value| try a.dupeSentinel(u8, std.mem.span(value), 0) else null;
    const gate = transformer.vqmm_nax_probe_override;
    defer {
        transformer.vqmm_nax_probe_override = gate;
        if (saved) |value| {
            _ = setenv("MLX_ENABLE_TF32", value, 1);
            a.free(value);
        } else _ = unsetenv("MLX_ENABLE_TF32");
        base.leaveTeacher();
    }
    for ([_]?bool{ gate, false }) |nax| {
        transformer.vqmm_nax_probe_override = nax;
        _ = unsetenv("MLX_ENABLE_TF32");
        try std.testing.expect(!base.teacher);
        try teacherEnvironment();
        try std.testing.expect(base.teacher);
        try std.testing.expectEqual(!base.naxArms(), base.reference_numerics);
        try std.testing.expectEqualStrings("0", std.mem.span(std.c.getenv("MLX_ENABLE_TF32").?));
        const arms = [_]bool{
            @import("glm5_hc_prefill.zig").enabled(),
            @import("glm5_hc_collapse_simd32.zig").enabled(),
            @import("glm5_kda_prefill_cluster.zig").enabled(),
            @import("glm5_attention_nax_packed.zig").enabled(),
            @import("glm5_indexpool_nax.zig").enabled(),
            @import("glm5_attention_decode_batch.zig").enabled(),
        };
        for (arms) |on| try std.testing.expectEqual(base.naxArms(), on);
        base.leaveTeacher();
        try std.testing.expect(!base.teacher and !base.reference_numerics);
    }
    _ = setenv("MLX_ENABLE_TF32", "1", 1);
    try std.testing.expectError(error.NativeGlmTeacherRequiresTf32Off, teacherEnvironment());
}

test "GLM KLD capture sends a pack to the generic student reference, at a BF16 latent only" {
    var cfg = model.ModelConfig{ .model_type = "glm5_next", .expert_layout = .exl3_k4 };
    const kv = @import("kv_quant.zig").KVQuantConfig;
    var opts = kld.Options{ .command = .capture, .no_template = true, .tokens = 256 };
    try std.testing.expect(!(try accepts(&cfg, opts)));
    for ([_]kv{ kv.affine(4), kv.affine(8) }) |quant| {
        opts.kv_quant_config = quant;
        try std.testing.expectError(error.GlmStudentReferenceNeedsBf16Latent, accepts(&cfg, opts));
    }
    opts.command = .compare;
    try std.testing.expect(!(try accepts(&cfg, opts)));
    cfg.expert_layout = .fp8_individual;
    try std.testing.expect(!(try accepts(&cfg, opts)));
    opts.command = .capture;
    opts.kv_quant_config = .dense;
    try std.testing.expectError(error.NativeGlmTeacherRequiresLosslessStreaming, accepts(&cfg, opts));
    cfg.expert_layout = .bf16_individual;
    opts = .{ .command = .capture, .no_template = true, .tokens = 256 };
    try std.testing.expectError(error.NativeGlmTeacherRequiresLosslessStreaming, accepts(&cfg, opts));
    opts.ssd_budget_bytes = 100 << 30;
    try std.testing.expect(try accepts(&cfg, opts));
}

test "GLM native KLD capture refuses a kv8 latent cache" {
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .expert_layout = .bf16_individual };
    const opts = kld.Options{ .command = .capture, .no_template = true, .tokens = 512, .ssd_budget_bytes = 100 << 30, .kv_quant_config = @import("kv_quant.zig").KVQuantConfig.affine(8) };
    try std.testing.expectError(error.NativeGlmTeacherRequiresLosslessStreaming, accepts(&cfg, opts));
}

test "GLM layer-major CPU disk bill counts every boundary row, token id and logits row still to write" {
    // The first tune portion plus the held-out windows: 550 windows of 500 tokens, 46 boundaries of four 4096 streams.
    try std.testing.expectEqual(@as(u64, 414_857_036_000), layerMajorDiskBytes(275_000, 550, 46, 16384, 154880));
    try std.testing.expectEqual(@as(u64, 550 * 154880 * 4), layerMajorDiskBytes(275_000, 550, 0, 0, 154880) - 275_000 * 4);
}

test "GLM layer-major capture CPU flags: one row per window, native teacher only" {
    const t = std.testing;
    const opts = try kld.parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p.jsonl", "--out", "/o", "--tokens", "1", "--no-template", "--layer-major", "--batch-windows", "4", "--pause-file", "/pause" });
    try t.expect(opts.layer_major);
    try t.expectEqual(@as(u32, 4), opts.batch_windows);
    try t.expectEqualStrings("/pause", opts.pause_file);
    try t.expectError(error.LayerMajorFlagWithoutLayerMajor, kld.parseArgs(&.{ "capture", "--model", "/m", "--prompts", "/p", "--out", "/o", "--batch-windows", "4" }));
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .expert_layout = .bf16_individual };
    var teacher = kld.Options{ .command = .capture, .no_template = true, .tokens = 1, .ssd_budget_bytes = 100 << 30, .layer_major = true, .hidden_out = "/h" };
    try t.expect(try accepts(&cfg, teacher));
    teacher.tokens = 2;
    try t.expectError(error.GlmLayerMajorNeedsOneRow, accepts(&cfg, teacher));
}

/// Test-only: the number of boundary appends a layer-major capture makes before it fails as if killed.
var interrupt_boundaries_for_test: ?usize = null;
var reserve_floor_for_test: u64 = 8 << 30;

/// A tiny GLM BF16 checkpoint and seven token-id windows of different lengths, under one temporary root.
const TinyCapture = struct {
    tmp: std.testing.TmpDir,
    root: []const u8,
    model: []const u8,
    prompts: []const u8,
    arena: std.heap.ArenaAllocator,

    fn init(a: std.mem.Allocator) !TinyCapture {
        var self = TinyCapture{ .tmp = std.testing.tmpDir(.{}), .root = "", .model = "", .prompts = "", .arena = .init(a) };
        errdefer self.deinit();
        const arena = self.arena.allocator();
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        self.root = try arena.dupe(u8, buf[0..try self.tmp.dir.realPath(std.testing.io, &buf)]);
        try self.tmp.dir.createDirPath(std.testing.io, "model");
        var model_dir = try self.tmp.dir.openDir(std.testing.io, "model", .{});
        defer model_dir.close(std.testing.io);
        try @import("glm_stream_fixture.zig").writeCheckpoint(a, model_dir, 0x61a7);
        self.model = try std.fmt.allocPrint(arena, "{s}/model", .{self.root});
        var lines: std.ArrayList(u8) = .empty;
        var prng = std.Random.DefaultPrng.init(3);
        for ([_]usize{ 9, 13, 5, 12, 10, 7, 11 }, 0..) |n, w| {
            try lines.print(arena, "{{\"id\":\"w{d:0>5}\",\"prompt_ids\":[", .{w});
            for (0..n) |i| try lines.print(arena, "{s}{d}", .{ if (i == 0) "" else ",", prng.random().uintLessThan(u32, 16) });
            try lines.appendSlice(arena, "]}\n");
        }
        try self.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "windows.jsonl", .data = lines.items });
        self.prompts = try std.fmt.allocPrint(arena, "{s}/windows.jsonl", .{self.root});
        return self;
    }
    fn deinit(self: *TinyCapture) void {
        self.arena.deinit();
        self.tmp.cleanup();
    }
    fn opts(self: *TinyCapture, name: []const u8, batch: u32) !kld.Options {
        const arena = self.arena.allocator();
        return .{
            .command = .capture,
            .model_dir = self.model,
            .prompts = self.prompts,
            .out_dir = try std.fmt.allocPrint(arena, "{s}/{s}", .{ self.root, name }),
            .hidden_out = try std.fmt.allocPrint(arena, "{s}/{s}-hidden", .{ self.root, name }),
            .tokens = 1,
            .no_template = true,
            .ssd_budget_bytes = 1 << 30,
            .label = "tiny",
            .layer_major = batch != 0,
            .batch_windows = batch,
        };
    }
    fn capture(self: *TinyCapture, a: std.mem.Allocator, opts_value: kld.Options) !void {
        _ = self;
        var out = kld.Out{ .silent = true };
        defer @import("glm5_model.zig").leaveTeacher();
        // The tiny checkpoint needs no production-sized floor; all computed bills and Metal limits still apply.
        const prior = reserve_floor_for_test;
        reserve_floor_for_test = 128 << 20;
        defer reserve_floor_for_test = prior;
        if (!try tryRun(a, std.testing.io, opts_value, &out)) return error.NativeGlmCaptureNotTaken;
    }
    fn read(self: *TinyCapture, sub: []const u8) ![]u8 {
        return self.tmp.dir.readFileAlloc(std.testing.io, sub, self.arena.allocator(), .limited(64 << 20));
    }
    /// Every fixture and boundary byte the two captures wrote, the baseline apart from its wall time.
    fn expectSame(self: *TinyCapture, want: []const u8, got: []const u8) !void {
        const arena = self.arena.allocator();
        for (0..7) |w| for ([_][]const u8{ "logits.f32", "generated_tokens.txt", "prompt_tokens.txt", "id.txt" }) |leaf| {
            const sub = try std.fmt.allocPrint(arena, "prompts/{d:0>2}_w{d:0>5}/{s}", .{ w, w, leaf });
            try std.testing.expectEqualSlices(u8, try self.read(try std.fmt.allocPrint(arena, "{s}/{s}", .{ want, sub })), try self.read(try std.fmt.allocPrint(arena, "{s}/{s}", .{ got, sub })));
        };
        for (0..@import("glm_stream_fixture.zig").Tiny.layers + 2) |b| {
            const leaf = if (b == 0) try arena.dupe(u8, "tokens.bin") else try std.fmt.allocPrint(arena, "boundary-{d:0>2}.bin", .{b - 1});
            const bytes = try self.read(try std.fmt.allocPrint(arena, "{s}-hidden/{s}", .{ want, leaf }));
            try std.testing.expect(bytes.len > 0);
            try std.testing.expectEqualSlices(u8, bytes, try self.read(try std.fmt.allocPrint(arena, "{s}-hidden/{s}", .{ got, leaf })));
        }
        var lines = [2]std.ArrayList(u8){ .empty, .empty };
        for ([_][]const u8{ want, got }, &lines) |name, *kept| {
            var it = std.mem.splitScalar(u8, try self.read(try std.fmt.allocPrint(arena, "{s}/baseline.json", .{name})), '\n');
            while (it.next()) |line| if (std.mem.indexOf(u8, line, "\"elapsed_secs\"") == null and std.mem.indexOf(u8, line, "\"label\"") == null) try kept.appendSlice(arena, line);
        }
        try std.testing.expectEqualStrings(lines[0].items, lines[1].items);
    }
};

test "GLM native teacher refuses a budget above Metal's recommended working set" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var tiny = try TinyCapture.init(a);
    defer tiny.deinit();
    var opts = try tiny.opts("above-wired", 0);
    opts.ssd_budget_bytes = @max(@as(u64, 10) << 30, @as(u64, mlx.maxRecommendedWorkingSet()) + 1);
    var out = kld.Out{ .silent = true };
    defer @import("glm5_model.zig").leaveTeacher();
    try std.testing.expectError(error.InvalidGlmWiredLimit, tryRun(a, std.testing.io, opts, &out));
}

test "GLM layer-major capture writes window-major's fixture and boundaries byte for byte, across batches and a resume" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var tiny = try TinyCapture.init(a);
    defer tiny.deinit();
    try tiny.capture(a, try tiny.opts("window-major", 0));
    try tiny.capture(a, try tiny.opts("batch2", 2));
    try tiny.expectSame("window-major", "batch2");
    try tiny.capture(a, try tiny.opts("batch3", 3));
    try tiny.expectSame("window-major", "batch3");

    // Interrupted in the middle of its second batch (each window appends six boundaries per batch),
    // then resumed with another batch size.
    interrupt_boundaries_for_test = 3 * 6 + 8;
    try std.testing.expectError(error.TestInterrupted, tiny.capture(a, try tiny.opts("resumed", 3)));
    interrupt_boundaries_for_test = null;
    var it = std.mem.splitScalar(u8, std.mem.trimEnd(u8, try tiny.read("resumed.partial/windows.jsonl"), "\n"), '\n');
    var committed: usize = 0;
    while (it.next()) |_| committed += 1;
    try std.testing.expectEqual(@as(usize, 3), committed);
    try tiny.capture(a, try tiny.opts("resumed", 2));
    try tiny.expectSame("window-major", "resumed");
}

test "GLM capture identity records whether the NAX arms were on and what each dispatched" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var tiny = try TinyCapture.init(a);
    defer tiny.deinit();
    try tiny.capture(a, try tiny.opts("identity", 0));
    const parsed = try std.json.parseFromSlice(std.json.Value, a, try tiny.read("identity/identity.json"), .{});
    defer parsed.deinit();
    const nax = @import("glm5_model.zig").naxArms();
    try std.testing.expectEqual(nax, parsed.value.object.get("nax_arms").?.bool);
    try std.testing.expectEqual(!nax, parsed.value.object.get("reference_numerics").?.bool);
    const arms = parsed.value.object.get("arm_dispatches").?.object;
    for ([_][]const u8{ "hc_prefill", "hc_collapse_simd32", "kda_value_rows", "kda_prefill_cluster", "mla_head_batch", "packed_attention", "index_scores_nax", "decode_b1" }) |name| {
        try std.testing.expect(arms.get(name).? == .integer);
    }
}

test "GLM layer-major resume refuses a changed committed window or a different run" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var tiny = try TinyCapture.init(a);
    defer tiny.deinit();
    interrupt_boundaries_for_test = 2 * 6 + 3;
    try std.testing.expectError(error.TestInterrupted, tiny.capture(a, try tiny.opts("run", 2)));
    interrupt_boundaries_for_test = null;
    var other = try tiny.opts("run", 2);
    other.label = "another";
    try std.testing.expectError(error.GlmLayerMajorResumeMismatch, tiny.capture(a, other));
    // Flip one bit in the first window's rows of one boundary.
    const path = try std.fmt.allocPrint(tiny.arena.allocator(), "{s}/run-hidden/boundary-03.bin", .{tiny.root});
    const bytes = try tiny.read("run-hidden/boundary-03.bin");
    bytes[17] ^= 1;
    try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = bytes });
    try std.testing.expectError(error.GlmLayerMajorWindowChanged, tiny.capture(a, try tiny.opts("run", 2)));
}

test "GLM layer-major resume refuses a mutated, deleted or truncated committed token file" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var tiny = try TinyCapture.init(a);
    defer tiny.deinit();
    interrupt_boundaries_for_test = 2 * 6 + 3;
    try std.testing.expectError(error.TestInterrupted, tiny.capture(a, try tiny.opts("run", 2)));
    interrupt_boundaries_for_test = null;
    const arena = tiny.arena.allocator();
    for ([_][]const u8{ "prompt_tokens.txt", "generated_tokens.txt" }) |leaf| {
        const sub = try std.fmt.allocPrint(arena, "run.partial/prompts/01_w00001/{s}", .{leaf});
        const path = try std.fmt.allocPrint(arena, "{s}/{s}", .{ tiny.root, sub });
        const original = try tiny.read(sub);
        try std.testing.expect(original.len > 0);
        const altered = try arena.dupe(u8, original);
        altered[0] = if (altered[0] == '9') '8' else altered[0] + 1;
        for ([_][]const u8{ altered, original[0 .. original.len - 1], "" }) |body| {
            try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = body });
            try std.testing.expectError(error.GlmLayerMajorWindowChanged, tiny.capture(a, try tiny.opts("run", 2)));
        }
        try std.Io.Dir.cwd().deleteFile(std.testing.io, path);
        try std.testing.expectError(error.GlmLayerMajorWindowChanged, tiny.capture(a, try tiny.opts("run", 2)));
        try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = path, .data = original });
    }
    try tiny.capture(a, try tiny.opts("run", 2));
    try tiny.capture(a, try tiny.opts("window-major", 0));
    try tiny.expectSame("window-major", "run");
}

test "GLM layer-major capture interrupted inside its first batch resumes to the uninterrupted bytes" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var tiny = try TinyCapture.init(a);
    defer tiny.deinit();
    try tiny.capture(a, try tiny.opts("window-major", 0));
    interrupt_boundaries_for_test = 3 + 5;
    try std.testing.expectError(error.TestInterrupted, tiny.capture(a, try tiny.opts("resumed", 3)));
    interrupt_boundaries_for_test = null;
    try std.testing.expectEqual(@as(usize, 0), (try tiny.read("resumed.partial/windows.jsonl")).len);
    try std.testing.expect((try tiny.read("resumed-hidden/boundary-00.bin")).len > 0);
    try tiny.capture(a, try tiny.opts("resumed", 2));
    try tiny.expectSame("window-major", "resumed");
}

test "GLM layer-major capture refuses a hidden directory that already holds rows, and keeps refusing" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var tiny = try TinyCapture.init(a);
    defer tiny.deinit();
    try tiny.tmp.dir.createDirPath(std.testing.io, "run-hidden");
    try tiny.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "run-hidden/boundary-02.bin", .data = "stale" });
    for (0..2) |_| try std.testing.expectError(error.GlmLayerMajorHiddenNotEmpty, tiny.capture(a, try tiny.opts("run", 2)));
    try std.testing.expectEqualStrings("stale", try tiny.read("run-hidden/boundary-02.bin"));
}

test "GLM layer-major capture idles on its pause file and continues when it is removed" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var tiny = try TinyCapture.init(a);
    defer tiny.deinit();
    var opts = try tiny.opts("paused", 4);
    opts.pause_file = try std.fmt.allocPrint(tiny.arena.allocator(), "{s}/pause", .{tiny.root});
    try tiny.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "pause", .data = "" });
    const Release = struct {
        fn run(dir: std.Io.Dir, stop: *std.atomic.Value(bool)) void {
            while (!stop.load(.acquire)) {
                if (dir.access(std.testing.io, "pause.ack", .{})) |_| break else |_| {}
                var tick = std.c.timespec{ .sec = 0, .nsec = 10_000_000 };
                _ = std.c.nanosleep(&tick, null);
            }
            dir.deleteFile(std.testing.io, "pause") catch {};
        }
    };
    var stop = std.atomic.Value(bool).init(false);
    const thread = try std.Thread.spawn(.{}, Release.run, .{ tiny.tmp.dir, &stop });
    defer {
        stop.store(true, .release);
        thread.join();
    }
    try tiny.capture(a, opts);
    if (tiny.tmp.dir.access(std.testing.io, "pause.ack", .{})) |_| return error.TestUnexpectedResult else |_| {}
    try tiny.capture(a, try tiny.opts("window-major", 0));
    try tiny.expectSame("window-major", "paused");
}
