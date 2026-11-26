//! GLP (GGUF Layer Projection) control vectors: the reader contract of
//! `docs/spec/GLP.md` (https://github.com/msuiche/weightless/blob/main/docs/spec/GLP.md).
//!
//! A GLP file is a GGUF v3 container with a `glp.*` metadata namespace and `direction.N`
//! tensors. This build implements exactly one contract — `spec_version = 1`, `mode = project`,
//! `hook_point = residual_stream_post_layer`, `rank = 1` — and refuses every other file by name
//! rather than applying it somewhere it was not calibrated for. The apply itself lives in the
//! arch (`dsv41_model.zig`); this module is the container and the gates, and it links no MLX.
//!
//! Fail-closed rules implemented here (GLP.md, "Reader conformance"):
//! 1. `glp.mode` is read; absence means `add`, which this build does not implement, so it is
//!    refused (there is no additive path to fall back to).
//! 2. An unrecognised mode is fatal; it never falls back to `add`.
//! 3. A `glp.hook_point` this build does not apply at is fatal.
//! 4. `direction.0` is rejected (this build implements no layer-0 apply); `direction.N` applies
//!    at layer `N`.
//! 5. `alpha` is never folded into the data: it is returned beside the unit directions.
//! 6. A project file is never merged with another control vector (the caller refuses a second
//!    `--glp`).
//! 7. `glp.spec_version` is read and an unimplemented version is fatal.
//! 8. `n_embd` must match the model's hidden size and every layer id must be inside the model.
const std = @import("std");
const gguf = @import("gguf.zig");

pub const Code = enum {
    gguf,
    mode_missing,
    mode_add,
    mode_unsupported,
    spec_version_missing,
    spec_version_unsupported,
    hook_point_unsupported,
    rank_unsupported,
    v2_key_under_v1,
    direction_zero,
    direction_name,
    direction_rank,
    direction_dtype,
    direction_shape,
    direction_count,
    direction_norm,
    direction_duplicate,
    embedding_mismatch,
    layer_out_of_range,
    alpha_missing,
    alpha_invalid,
    content_sha256_mismatch,
};

pub const Refusal = struct {
    code: Code,
    /// Owned, actionable text: what was wrong and what this build implements.
    message: []u8,

    pub fn deinit(self: *Refusal, a: std.mem.Allocator) void {
        a.free(self.message);
        self.* = undefined;
    }
};

/// A parsed, validated GLP vector: one unit-norm f32 direction per steered layer, or null.
pub const Loaded = struct {
    allocator: std.mem.Allocator,
    n_embd: usize,
    /// `glp.alpha_default`, or the runtime override when one was given.
    alpha: f32,
    /// `dirs[layer]` is the unit direction applied at that layer, or null.
    dirs: []?[]f32,

    pub fn deinit(self: *Loaded) void {
        for (self.dirs) |d| if (d) |v| self.allocator.free(v);
        self.allocator.free(self.dirs);
        self.* = undefined;
    }
};

pub const Result = union(enum) {
    loaded: Loaded,
    refused: Refusal,
};

/// How many layers the vector steers (the load line's count).
pub fn steeredLayers(l: *const Loaded) usize {
    var n: usize = 0;
    for (l.dirs) |d| {
        if (d != null) n += 1;
    }
    return n;
}

pub const Options = struct {
    /// The loaded model's hidden size (`config.hidden_size`).
    n_embd: usize,
    /// The loaded model's layer count (`config.num_hidden_layers`).
    n_layers: usize,
    /// `--glp-alpha` or `WEIGHTLESS_STEER_ALPHA`; null keeps `glp.alpha_default`.
    alpha_override: ?f32 = null,
};

/// Reads and validates `path`. Only I/O and allocation errors escape; a contract violation is a
/// `Result.refused` with its own message.
pub fn load(io: std.Io, allocator: std.mem.Allocator, path: []const u8, opts: Options) !Result {
    const raw = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(64 << 20)) catch |err| {
        return refuse(allocator, .gguf, "cannot read '{s}': {s}", .{ path, @errorName(err) });
    };
    defer allocator.free(raw);
    return parse(allocator, raw, opts);
}

/// Validates an in-memory GGUF. `raw` is borrowed for the call only.
pub fn parse(allocator: std.mem.Allocator, raw: []const u8, opts: Options) !Result {
    var f = gguf.parse(allocator, raw) catch |err| {
        return refuse(allocator, .gguf, "not a GGUF v3 file: {s}", .{@errorName(err)});
    };
    defer f.deinit();
    return parseFile(allocator, &f, opts);
}

fn parseFile(allocator: std.mem.Allocator, f: *const gguf.File, opts: Options) !Result {
    var has_glp = false;
    var has_dspark = false;
    var it = f.metadata.iterator();
    while (it.next()) |e| {
        if (std.mem.startsWith(u8, e.key_ptr.*, "glp.")) has_glp = true;
        if (std.mem.startsWith(u8, e.key_ptr.*, "dspark.")) has_dspark = true;
    }
    if (!has_glp) {
        if (has_dspark) return refuse(allocator, .mode_missing, "no glp.* metadata, only the pre-GLP dspark.* namespace: re-export or re-download the vector (the GLP rename has no alias)", .{});
        return refuse(allocator, .mode_missing, "no glp.* metadata: this is a legacy llama.cpp additive control vector, and this build implements GLP project mode only (spec_version 1, project, residual_stream_post_layer, rank 1) — see https://github.com/msuiche/weightless/blob/main/docs/spec/GLP.md", .{});
    }

    const mode = switch (f.get("glp.mode") orelse return refuse(allocator, .mode_missing, "glp.mode absent: absence means the legacy additive operation, which this build does not implement (project only)", .{})) {
        .string => |s| s,
        else => return refuse(allocator, .mode_unsupported, "glp.mode is not a string", .{}),
    };
    if (std.mem.eql(u8, mode, "add")) return refuse(allocator, .mode_add, "glp.mode = add: additive control vectors are not implemented in this build (project only); a projective direction applied additively is silently wrong", .{});
    if (!std.mem.eql(u8, mode, "project")) return refuse(allocator, .mode_unsupported, "glp.mode = '{s}' not supported: this build implements 'project' only", .{mode});

    const version = switch (f.get("glp.spec_version") orelse return refuse(allocator, .spec_version_missing, "glp.spec_version missing: a project file must name the contract it was written to (this build implements 1)", .{})) {
        else => |v| asU32(v) orelse return refuse(allocator, .spec_version_unsupported, "glp.spec_version is not an integer", .{}),
    };
    if (version != 1) return refuse(allocator, .spec_version_unsupported, "glp.spec_version {d} not supported: this build implements v1 (project, residual_stream_post_layer, rank 1)", .{version});

    const hook = if (f.get("glp.hook_point")) |v| switch (v) {
        .string => |s| s,
        else => return refuse(allocator, .hook_point_unsupported, "glp.hook_point is not a string", .{}),
    } else "residual_stream_post_layer";
    if (!std.mem.eql(u8, hook, "residual_stream_post_layer")) {
        const extra: []const u8 = if (std.mem.eql(u8, hook, "hc_fold_pre_engram")) " (an hc fold site requires spec_version 3 and the companion attention-writer arm)" else "";
        return refuse(allocator, .hook_point_unsupported, "glp.hook_point = '{s}' not supported: this build applies at residual_stream_post_layer only{s}", .{ hook, extra });
    }

    const rank = if (f.get("glp.rank")) |v| asU32(v) orelse return refuse(allocator, .rank_unsupported, "glp.rank is not an integer", .{}) else 1;
    if (rank != 1) return refuse(allocator, .rank_unsupported, "glp.rank {d} not supported: this build implements rank 1 (one direction per layer)", .{rank});
    for ([_][]const u8{ "glp.dir_scales", "glp.layer_scales" }) |key| {
        if (f.get(key) != null) return refuse(allocator, .v2_key_under_v1, "{s} requires spec_version 2, but the file says 1", .{key});
    }

    const alpha_file = switch (f.get("glp.alpha_default") orelse return refuse(allocator, .alpha_missing, "glp.alpha_default missing: a project file must carry its dose", .{})) {
        else => |v| asF32(v) orelse return refuse(allocator, .alpha_invalid, "glp.alpha_default is not a number", .{}),
    };
    if (!std.math.isFinite(alpha_file) or alpha_file < 0) return refuse(allocator, .alpha_invalid, "glp.alpha_default {d} must be finite and >= 0", .{alpha_file});
    const alpha = opts.alpha_override orelse alpha_file;
    if (!std.math.isFinite(alpha) or alpha < 0) return refuse(allocator, .alpha_invalid, "alpha override {d} must be finite and >= 0", .{alpha});

    var dirs = try allocator.alloc(?[]f32, opts.n_layers);
    @memset(dirs, null);
    var keep = false;
    defer if (!keep) {
        for (dirs) |d| if (d) |v| allocator.free(v);
        allocator.free(dirs);
    };
    const seen = try allocator.alloc(bool, opts.n_layers);
    defer allocator.free(seen);
    @memset(seen, false);
    const raw_of = try allocator.alloc(?[]const u8, opts.n_layers);
    defer allocator.free(raw_of);
    @memset(raw_of, null);

    var count: usize = 0;
    for (f.tensors) |*t| {
        const name = parseDirectionName(t.name) orelse return refuse(allocator, .direction_name, "tensor '{s}' is not a GLP direction: expected direction.N (N >= 1)", .{t.name});
        if (name.dir != 0) return refuse(allocator, .direction_rank, "tensor '{s}' is a rank-k direction (direction.N.j): this build implements rank 1", .{t.name});
        if (name.layer == 0) return refuse(allocator, .direction_zero, "direction.0 is invalid: this build implements no layer-0 apply; re-export without layer 0", .{});
        if (name.layer >= opts.n_layers) return refuse(allocator, .layer_out_of_range, "direction.{d} applies at layer {d}, but the model has {d} layers", .{ name.layer, name.layer, opts.n_layers });
        if (seen[name.layer]) return refuse(allocator, .direction_duplicate, "direction.{d} appears twice", .{name.layer});
        if (t.type_ != 0) return refuse(allocator, .direction_dtype, "tensor '{s}' is not F32 (ggml type {d}): GLP directions are F32", .{ t.name, t.type_ });
        if (t.dims.len != 1) return refuse(allocator, .direction_shape, "tensor '{s}' has {d} dimensions, expected 1", .{ t.name, t.dims.len });
        if (t.dims[0] != opts.n_embd) return refuse(allocator, .embedding_mismatch, "direction '{s}' has {d} elements; the model's hidden size is {d}", .{ t.name, t.dims[0], opts.n_embd });
        const bytes = f.tensorBytes(t) catch |err| return refuse(allocator, .gguf, "tensor '{s}': {s}", .{ t.name, @errorName(err) });

        const dir = try allocator.alloc(f32, opts.n_embd);
        dirs[name.layer] = dir;
        var sum: f64 = 0;
        for (dir, 0..) |*v, k| {
            const bits = std.mem.readInt(u32, bytes[k * 4 ..][0..4], .little);
            v.* = @bitCast(bits);
            if (!std.math.isFinite(v.*)) return refuse(allocator, .direction_norm, "tensor '{s}' has a non-finite element", .{t.name});
            sum += @as(f64, v.*) * @as(f64, v.*);
        }
        const norm = @sqrt(sum);
        if (!(norm > 1e-12)) return refuse(allocator, .direction_norm, "tensor '{s}' has norm {d}: a direction must be non-zero", .{ t.name, norm });
        for (dir) |*v| v.* = @floatCast(@as(f64, v.*) / norm);
        raw_of[name.layer] = bytes;
        seen[name.layer] = true;
        count += 1;
    }
    if (count == 0) return refuse(allocator, .direction_count, "no direction.N tensors: a GLP file carries at least one", .{});

    if (f.get("glp.content_sha256")) |v| {
        const want = switch (v) {
            .string => |s| s,
            else => return refuse(allocator, .content_sha256_mismatch, "glp.content_sha256 is not a string", .{}),
        };
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        for (raw_of) |rb| if (rb) |b| h.update(b);
        const got = std.fmt.bytesToHex(h.finalResult(), .lower);
        if (!std.ascii.eqlIgnoreCase(want, &got)) return refuse(allocator, .content_sha256_mismatch, "glp.content_sha256 does not match the tensor bytes: the file is corrupt or was edited", .{});
    }

    keep = true;
    return .{ .loaded = .{ .allocator = allocator, .n_embd = opts.n_embd, .alpha = alpha, .dirs = dirs } };
}

const DirectionName = struct { layer: usize, dir: usize };

/// `direction.N` -> (N, 0); `direction.N.j` -> (N, j); null for anything else.
fn parseDirectionName(name: []const u8) ?DirectionName {
    const prefix = "direction.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    var it = std.mem.splitScalar(u8, name[prefix.len..], '.');
    const layer_s = it.next() orelse return null;
    if (layer_s.len == 0) return null;
    const layer = std.fmt.parseInt(usize, layer_s, 10) catch return null;
    const dir_s = it.next();
    if (it.next() != null) return null;
    const dir = if (dir_s) |s| blk: {
        if (s.len == 0) return null;
        break :blk std.fmt.parseInt(usize, s, 10) catch return null;
    } else 0;
    return .{ .layer = layer, .dir = dir };
}

fn asU32(v: gguf.Value) ?u32 {
    return switch (v) {
        .uint8 => |n| n,
        .uint16 => |n| n,
        .uint32 => |n| n,
        .uint64 => |n| std.math.cast(u32, n),
        .int8 => |n| if (n >= 0) @intCast(n) else null,
        .int16 => |n| if (n >= 0) @intCast(n) else null,
        .int32 => |n| if (n >= 0) @intCast(n) else null,
        .int64 => |n| if (n >= 0) std.math.cast(u32, n) else null,
        else => null,
    };
}

fn asF32(v: gguf.Value) ?f32 {
    return switch (v) {
        .float32 => |n| n,
        .float64 => |n| @floatCast(n),
        else => null,
    };
}

fn refuse(a: std.mem.Allocator, code: Code, comptime fmt: []const u8, args: anytype) !Result {
    return .{ .refused = .{ .code = code, .message = try std.fmt.allocPrint(a, fmt, args) } };
}

// ── the load's hand-off to the arch ──────────────────────────────────────
//
// The load validates the file before any weight is read (`model_load.zig`) and parks the result
// here; the arch's model init takes it and builds its MLX arrays. Inference thread only, one
// model load at a time, like the other launch decisions in `model.zig`.

var pending: ?Loaded = null;

pub fn setPending(l: Loaded) void {
    pending = l;
}

pub fn takePending() ?Loaded {
    const p = pending;
    pending = null;
    return p;
}

pub fn clearPending() void {
    if (pending) |*p| p.deinit();
    pending = null;
}

// ── tests ────────────────────────────────────────────────────────────────
//
// The fixtures are synthetic GGUF v3 files written by `tests/glp/make_fixtures.py` (stdlib only,
// no `gguf` package): a valid v1 project vector and one file per refusal path.

const fixtures = struct {
    const valid = @embedFile("glp/fixtures/valid.gguf");
    const mode_add = @embedFile("glp/fixtures/mode_add.gguf");
    const mode_unknown = @embedFile("glp/fixtures/mode_unknown.gguf");
    const spec_v2 = @embedFile("glp/fixtures/spec_v2.gguf");
    const hook_hc = @embedFile("glp/fixtures/hook_hc.gguf");
    const rank2 = @embedFile("glp/fixtures/rank2.gguf");
    const direction0 = @embedFile("glp/fixtures/direction0.gguf");
    const no_glp = @embedFile("glp/fixtures/no_glp.gguf");
    const bad_dtype = @embedFile("glp/fixtures/bad_dtype.gguf");
    const bad_shape = @embedFile("glp/fixtures/bad_shape.gguf");
    const bad_norm = @embedFile("glp/fixtures/bad_norm.gguf");
    const bad_sha = @embedFile("glp/fixtures/bad_sha.gguf");
    const layer_oob = @embedFile("glp/fixtures/layer_oob.gguf");
    const dir_scales = @embedFile("glp/fixtures/dir_scales.gguf");
};

const test_opts = Options{ .n_embd = 8, .n_layers = 4 };

fn expectRefused(raw: []const u8, opts: Options, want: Code) !void {
    var r = try parse(std.testing.allocator, raw, opts);
    switch (r) {
        .loaded => |*l| {
            l.deinit();
            return error.TestExpectedRefusal;
        },
        .refused => |*ref| {
            defer ref.deinit(std.testing.allocator);
            try std.testing.expectEqual(want, ref.code);
            try std.testing.expect(ref.message.len > 0);
        },
    }
}

test "glp: a valid v1 project vector parses, normalises and keeps alpha out of the data" {
    const r = try parse(std.testing.allocator, fixtures.valid, test_opts);
    var l = switch (r) {
        .loaded => |l| l,
        .refused => |ref| {
            std.debug.print("unexpected refusal: {s}\n", .{ref.message});
            return error.TestUnexpectedRefusal;
        },
    };
    defer l.deinit();
    try std.testing.expectEqual(@as(usize, 8), l.n_embd);
    try std.testing.expectEqual(@as(f32, 4.0), l.alpha);
    try std.testing.expect(l.dirs[0] == null);
    try std.testing.expect(l.dirs[1] != null);
    try std.testing.expect(l.dirs[2] != null);
    try std.testing.expect(l.dirs[3] == null);
    // The fixture's layer-1 direction is (3,4,0,...): unit-norm is (0.6,0.8,0,...).
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), l.dirs[1].?[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), l.dirs[1].?[1], 1e-6);
    var norm: f64 = 0;
    for (l.dirs[1].?) |v| norm += @as(f64, v) * @as(f64, v);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), @sqrt(norm), 1e-6);
}

test "glp: the runtime alpha override replaces glp.alpha_default" {
    const r = try parse(std.testing.allocator, fixtures.valid, .{ .n_embd = 8, .n_layers = 4, .alpha_override = 2.5 });
    var l = switch (r) {
        .loaded => |l| l,
        .refused => |ref| {
            std.debug.print("unexpected refusal: {s}\n", .{ref.message});
            return error.TestUnexpectedRefusal;
        },
    };
    defer l.deinit();
    try std.testing.expectEqual(@as(f32, 2.5), l.alpha);
}

test "glp: every refusal path is named" {
    try expectRefused(fixtures.mode_add, test_opts, .mode_add);
    try expectRefused(fixtures.mode_unknown, test_opts, .mode_unsupported);
    try expectRefused(fixtures.spec_v2, test_opts, .spec_version_unsupported);
    try expectRefused(fixtures.hook_hc, test_opts, .hook_point_unsupported);
    try expectRefused(fixtures.rank2, test_opts, .rank_unsupported);
    try expectRefused(fixtures.direction0, test_opts, .direction_zero);
    try expectRefused(fixtures.no_glp, test_opts, .mode_missing);
    try expectRefused(fixtures.bad_dtype, test_opts, .direction_dtype);
    try expectRefused(fixtures.bad_shape, test_opts, .direction_shape);
    try expectRefused(fixtures.bad_norm, test_opts, .direction_norm);
    try expectRefused(fixtures.bad_sha, test_opts, .content_sha256_mismatch);
    try expectRefused(fixtures.layer_oob, test_opts, .layer_out_of_range);
    try expectRefused(fixtures.dir_scales, test_opts, .v2_key_under_v1);
}

test "glp: a direction wider than the model's hidden size is refused" {
    try expectRefused(fixtures.valid, .{ .n_embd = 16, .n_layers = 4 }, .embedding_mismatch);
}

test "glp: a direction past the model's layer count is refused" {
    try expectRefused(fixtures.valid, .{ .n_embd = 8, .n_layers = 2 }, .layer_out_of_range);
}

test "glp: direction names map N to layer N, with no offset" {
    try std.testing.expectEqual(DirectionName{ .layer = 3, .dir = 0 }, parseDirectionName("direction.3").?);
    try std.testing.expectEqual(DirectionName{ .layer = 3, .dir = 2 }, parseDirectionName("direction.3.2").?);
    try std.testing.expect(parseDirectionName("direction.") == null);
    try std.testing.expect(parseDirectionName("direction.3.") == null);
    try std.testing.expect(parseDirectionName("direction.3.2.1") == null);
    try std.testing.expect(parseDirectionName("other.3") == null);
}
