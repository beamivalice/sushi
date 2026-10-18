const std = @import("std");
const mlx = @import("mlx_host").mlx;
const format = @import("sushi_exl3").format;
const st = @import("safetensors.zig");

pub const Arch = enum {
    qwen4_exp,
    mimo_v2,
    glm5_next,

    pub fn defaultTrunkBits(self: Arch) u32 {
        return switch (self) {
            .qwen4_exp, .mimo_v2 => 8,
            .glm5_next => 6,
        };
    }

    pub fn defaultTrunkGroup(self: Arch) u32 {
        return switch (self) {
            .qwen4_exp, .mimo_v2 => 64,
            .glm5_next => 128,
        };
    }

    pub fn expertPrefix(self: Arch, mtp: bool) []const u8 {
        return switch (self) {
            .qwen4_exp => if (mtp) "language_model.mtp" else "language_model.model",
            .mimo_v2 => "model",
            .glm5_next => "model.language_model",
        };
    }
};

pub const ProjRole = enum(u8) {
    gate = 0,
    up = 1,
    down = 2,

    pub fn suffix(self: ProjRole) []const u8 {
        return switch (self) {
            .gate => "gate_proj",
            .up => "up_proj",
            .down => "down_proj",
        };
    }

    pub fn shortName(self: ProjRole) []const u8 {
        return switch (self) {
            .gate => "gate",
            .up => "up",
            .down => "down",
        };
    }
};

pub const TrunkAction = union(enum) {
    retain_source,
    retain_plus_one,
    retain_conv_last,
    retain_channels_last,
    affine: struct { bits: u32, group_size: u32, override_in_config: bool },
};

pub const AffineTriple = struct {
    weight: []u32,
    scales: []u16,
    biases: []u16,
    rows: usize,
    w_cols: usize,
    s_cols: usize,

    pub fn deinit(self: *AffineTriple, allocator: std.mem.Allocator) void {
        allocator.free(self.weight);
        allocator.free(self.scales);
        allocator.free(self.biases);
    }
};

pub fn identifyConfigJson(allocator: std.mem.Allocator, raw_json: []const u8) !Arch {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw_json, .{}) catch return error.UnknownSourceFormat;
    defer parsed.deinit();
    if (parsed.value != .object) return error.UnknownSourceFormat;
    const root = parsed.value.object;
    const mt_val = root.get("model_type") orelse blk: {
        if (root.get("text_config")) |tc| {
            if (tc == .object) if (tc.object.get("model_type")) |v| break :blk v;
        }
        return error.UnknownSourceFormat;
    };
    if (mt_val != .string) return error.UnknownSourceFormat;
    const mt = mt_val.string;
    if (std.mem.eql(u8, mt, "qwen4_exp") or std.mem.eql(u8, mt, "qwen4_exp_text")) return .qwen4_exp;
    if (std.mem.eql(u8, mt, "mimo_v2")) return .mimo_v2;
    if (std.mem.eql(u8, mt, "glm5_next") or std.mem.eql(u8, mt, "glm5_next_text")) return .glm5_next;
    return error.ArchitectureUnsupported;
}

pub fn identifySourceDir(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !Arch {
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (std.mem.endsWith(u8, entry.name, ".gguf")) return error.ModelFormatUnsupported;
    }
    const cfg_bytes = dir.readFileAlloc(io, "config.json", allocator, .limited(16 * 1024 * 1024)) catch return error.UnknownSourceFormat;
    defer allocator.free(cfg_bytes);
    return identifyConfigJson(allocator, cfg_bytes);
}

pub fn decodeBf16ToF32(raw_u16: []const u16, out: []f32) !void {
    if (raw_u16.len == 0) return error.EmptyTensor;
    if (out.len != raw_u16.len) return error.ShapeMismatch;
    for (raw_u16, out) |w, *dst| {
        if ((w & 0x7F80) == 0x7F80) return error.NonFiniteSourceWeight;
        dst.* = @bitCast(@as(u32, w) << 16);
    }
}

pub fn publicFromBf16(raw_u16: []const u16, out_rows: usize, in_cols: usize, out_public: []f32) !void {
    if (out_rows == 0 or in_cols == 0) return error.EmptyTensor;
    const total = std.math.mul(usize, out_rows, in_cols) catch return error.ShapeMismatch;
    if (raw_u16.len != total or out_public.len != total) return error.ShapeMismatch;
    for (0..out_rows) |r| {
        const row = raw_u16[r * in_cols ..][0..in_cols];
        for (row, 0..) |w, c| {
            if ((w & 0x7F80) == 0x7F80) return error.NonFiniteSourceWeight;
            out_public[c * out_rows + r] = @bitCast(@as(u32, w) << 16);
        }
    }
}

pub fn decodeFp8E4m3Byte(b: u8) !f32 {
    if ((b & 0x7F) == 0x7F) return error.NonFiniteSourceWeight;
    const h_bits: u16 = (@as(u16, b & 0x7F) << 7) | (@as(u16, b & 0x80) << 8);
    return format.f16BitsToF32(h_bits) * 256.0;
}

pub fn decodeFp8Block(codes: []const u8, scales: []const f32, rows: usize, cols: usize, block: usize, out: []f32) !void {
    if (rows == 0 or cols == 0 or block == 0) return error.EmptyTensor;
    const total = std.math.mul(usize, rows, cols) catch return error.ShapeMismatch;
    if (codes.len != total or out.len != total) return error.ShapeMismatch;
    const s_rows = (rows + block - 1) / block;
    const s_cols = (cols + block - 1) / block;
    if (scales.len != s_rows * s_cols) return error.ShapeMismatch;
    for (scales) |s| if (!std.math.isFinite(s)) return error.NonFiniteSourceWeight;

    for (0..rows) |r| {
        const sr = (r / block) * s_cols;
        for (0..cols) |c| {
            const v = (try decodeFp8E4m3Byte(codes[r * cols + c])) * scales[sr + c / block];
            if (!std.math.isFinite(v)) return error.NonFiniteSourceWeight;
            out[r * cols + c] = v;
        }
    }
}

const e2m1_mag = [8]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };

pub fn e2m1Value(nibble: u4) f32 {
    const mag = e2m1_mag[nibble & 7];
    return if ((nibble & 8) != 0) -mag else mag;
}

pub fn e8m0Value(code: u8) !f32 {
    if (code == 0) return 0.0;
    if (code == 255) return error.NonFiniteSourceWeight;
    return @bitCast(@as(u32, code) << 23);
}

pub fn decodeMxfp4(weight: []const u8, scale: []const u8, out_rows: usize, in_cols: usize, out: []f32) !void {
    if (out_rows == 0 or in_cols == 0) return error.EmptyTensor;
    if (in_cols % 32 != 0) return error.GeometryUnsupported;
    const half_cols = in_cols / 2;
    const scale_cols = in_cols / 32;
    if (weight.len != out_rows * half_cols or scale.len != out_rows * scale_cols or out.len != out_rows * in_cols) {
        return error.ShapeMismatch;
    }
    for (0..out_rows) |r| {
        const w_row = weight[r * half_cols ..][0..half_cols];
        const s_row = scale[r * scale_cols ..][0..scale_cols];
        const dst_row = out[r * in_cols ..][0..in_cols];
        for (0..scale_cols) |sb| {
            const sc = try e8m0Value(s_row[sb]);
            for (0..16) |b_idx| {
                const byte = w_row[sb * 16 + b_idx];
                const lo = e2m1Value(@truncate(byte & 0x0F)) * sc;
                const hi = e2m1Value(@truncate(byte >> 4)) * sc;
                if (!std.math.isFinite(lo) or !std.math.isFinite(hi)) return error.NonFiniteSourceWeight;
                dst_row[sb * 32 + b_idx * 2] = lo;
                dst_row[sb * 32 + b_idx * 2 + 1] = hi;
            }
        }
    }
}

pub fn publicFromMxfp4(weight: []const u8, scale: []const u8, out_rows: usize, in_cols: usize, out_public: []f32) !void {
    if (out_rows == 0 or in_cols == 0) return error.EmptyTensor;
    if (in_cols % 32 != 0) return error.GeometryUnsupported;
    const half_cols = in_cols / 2;
    const scale_cols = in_cols / 32;
    if (weight.len != out_rows * half_cols or scale.len != out_rows * scale_cols or out_public.len != in_cols * out_rows) {
        return error.ShapeMismatch;
    }
    for (0..out_rows) |r| {
        const w_row = weight[r * half_cols ..][0..half_cols];
        const s_row = scale[r * scale_cols ..][0..scale_cols];
        for (0..scale_cols) |sb| {
            const sc = try e8m0Value(s_row[sb]);
            for (0..16) |b_idx| {
                const byte = w_row[sb * 16 + b_idx];
                const lo = e2m1Value(@truncate(byte & 0x0F)) * sc;
                const hi = e2m1Value(@truncate(byte >> 4)) * sc;
                if (!std.math.isFinite(lo) or !std.math.isFinite(hi)) return error.NonFiniteSourceWeight;
                const c0 = sb * 32 + b_idx * 2;
                out_public[c0 * out_rows + r] = lo;
                out_public[(c0 + 1) * out_rows + r] = hi;
            }
        }
    }
}

pub fn quantizeAffineBf16(
    allocator: std.mem.Allocator,
    raw_bf16: []const u16,
    rows: usize,
    cols: usize,
    bits: u32,
    group_size: u32,
) !AffineTriple {
    if (rows == 0 or cols == 0) return error.EmptyTensor;
    if (raw_bf16.len != rows * cols) return error.ShapeMismatch;
    if (bits != 2 and bits != 3 and bits != 4 and bits != 5 and bits != 6 and bits != 8) return error.GeometryUnsupported;
    if (group_size != 32 and group_size != 64 and group_size != 128) return error.GeometryUnsupported;
    if (cols % group_size != 0 or (cols * bits) % 32 != 0) return error.GeometryUnsupported;
    for (raw_bf16) |w| if ((w & 0x7F80) == 0x7F80) return error.NonFiniteSourceWeight;

    const w_cols = cols * bits / 32;
    const s_cols = cols / group_size;
    const weight = try allocator.alloc(u32, rows * w_cols);
    errdefer allocator.free(weight);
    const scales = try allocator.alloc(u16, rows * s_cols);
    errdefer allocator.free(scales);
    const biases = try allocator.alloc(u16, rows * s_cols);
    errdefer allocator.free(biases);

    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);

    const chunk_rows: usize = 4096;
    var r0: usize = 0;
    while (r0 < rows) {
        const r1 = @min(rows, r0 + chunk_rows);
        const nr: c_int = @intCast(r1 - r0);
        const nc: c_int = @intCast(cols);
        const wb = mlx.mlx_array_new_data(raw_bf16[r0 * cols ..].ptr, &[_]c_int{ nr, nc }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(wb);

        var vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_quantize(
            &vec,
            wb,
            mlx.mlx_optional_int.some(@intCast(group_size)),
            mlx.mlx_optional_int.some(@intCast(bits)),
            "affine",
            .{},
            cpu,
        ));
        if (mlx.mlx_vector_array_size(vec) != 3) return error.UnexpectedQuantizeOutput;
        var q_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_arr);
        var s_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(s_arr);
        var b_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b_arr);
        try mlx.check(mlx.mlx_vector_array_get(&q_arr, vec, 0));
        try mlx.check(mlx.mlx_vector_array_get(&s_arr, vec, 1));
        try mlx.check(mlx.mlx_vector_array_get(&b_arr, vec, 2));
        try mlx.check(mlx.mlx_array_eval(q_arr));
        try mlx.check(mlx.mlx_array_eval(s_arr));
        try mlx.check(mlx.mlx_array_eval(b_arr));

        const q_ptr = mlx.mlx_array_data_uint32(q_arr).?;
        const s_ptr: [*]const u16 = @ptrCast(mlx.mlx_array_data_bfloat16(s_arr).?);
        const b_ptr: [*]const u16 = @ptrCast(mlx.mlx_array_data_bfloat16(b_arr).?);
        const w_count = (r1 - r0) * w_cols;
        const s_count = (r1 - r0) * s_cols;
        @memcpy(weight[r0 * w_cols ..][0..w_count], q_ptr[0..w_count]);
        @memcpy(scales[r0 * s_cols ..][0..s_count], s_ptr[0..s_count]);
        @memcpy(biases[r0 * s_cols ..][0..s_count], b_ptr[0..s_count]);
        for (scales[r0 * s_cols ..][0..s_count]) |sv| if ((sv & 0x7F80) == 0x7F80) return error.NonFiniteAffineGrid;
        for (biases[r0 * s_cols ..][0..s_count]) |bv| if ((bv & 0x7F80) == 0x7F80) return error.NonFiniteAffineGrid;
        r0 = r1;
    }
    return .{
        .weight = weight,
        .scales = scales,
        .biases = biases,
        .rows = rows,
        .w_cols = w_cols,
        .s_cols = s_cols,
    };
}

fn qwenIsVisual(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "model.visual.") or std.mem.startsWith(u8, key, "vision_tower.");
}

fn qwenRetainedAction(key: []const u8, dtype: []const u8, shape: []const usize) ?TrunkAction {
    if (!std.mem.eql(u8, dtype, "BF16")) return null;
    if (shape.len == 1 and std.mem.endsWith(u8, key, ".weight") and !qwenIsVisual(key) and !std.mem.endsWith(u8, key, ".linear_attn.norm.weight")) {
        const stem = key[0 .. key.len - ".weight".len];
        const leaf = stem[(std.mem.lastIndexOfScalar(u8, stem, '.') orelse 0)..];
        if (std.mem.indexOf(u8, leaf, "norm") != null) return .retain_plus_one;
    }
    if (shape.len == 3 and shape[1] == 1 and std.mem.endsWith(u8, key, ".conv1d.weight")) return .retain_conv_last;
    if (shape.len == 5 and std.mem.endsWith(u8, key, "patch_embed.proj.weight")) return .retain_channels_last;
    return null;
}

pub fn trunkSpecForTensor(arch: Arch, key: []const u8, dtype: []const u8, shape: []const usize, trunk_bits: u32, trunk_group: u32) TrunkAction {
    if (arch == .qwen4_exp) if (qwenRetainedAction(key, dtype, shape)) |action| return action;
    if (!std.mem.eql(u8, dtype, "BF16") or shape.len != 2 or !std.mem.endsWith(u8, key, ".weight")) {
        return .retain_source;
    }
    switch (arch) {
        .qwen4_exp => {
            if (std.mem.startsWith(u8, key, "model.visual.") or
                std.mem.startsWith(u8, key, "vision_tower.") or
                std.mem.endsWith(u8, key, ".block_inject_weight.weight") or
                std.mem.endsWith(u8, key, ".mlp.gate.weight") or
                std.mem.endsWith(u8, key, ".mlp.shared_expert_gate.weight") or
                shape[0] < 32 or shape[1] % 64 != 0)
            {
                return .retain_source;
            }
            if (std.mem.endsWith(u8, key, ".embed_tokens.weight")) {
                return .{ .affine = .{ .bits = trunk_bits, .group_size = 64, .override_in_config = trunk_group != 64 } };
            }
            const gs: u32 = if (shape[1] % trunk_group == 0) trunk_group else 64;
            return .{ .affine = .{ .bits = trunk_bits, .group_size = gs, .override_in_config = gs != trunk_group } };
        },
        .glm5_next => {
            const numel = shape[0] * shape[1];
            if (numel <= 1048576 or
                shape[1] % trunk_group != 0 or
                std.mem.indexOf(u8, key, ".mlp.gate.") != null or
                std.mem.indexOf(u8, key, ".hc_") != null or
                std.mem.indexOf(u8, key, ".mlp.experts.") != null)
            {
                return .retain_source;
            }
            return .{ .affine = .{ .bits = trunk_bits, .group_size = trunk_group, .override_in_config = true } };
        },
        .mimo_v2 => {
            if (std.mem.eql(u8, key, "lm_head.weight") or
                std.mem.eql(u8, key, "model.embed_tokens.weight") or
                (std.mem.startsWith(u8, key, "model.layers.") and std.mem.endsWith(u8, key, ".self_attn.o_proj.weight")))
            {
                if (shape[1] % trunk_group != 0) return .retain_source;
                return .{ .affine = .{ .bits = trunk_bits, .group_size = trunk_group, .override_in_config = false } };
            }
            return .retain_source;
        },
    }
}

pub const TensorLoc = struct {
    shard_idx: usize,
    data_start: u64,
    start: u64,
    end: u64,
    dtype: []u8,
    shape: []usize,
};

pub const SafetensorsSource = struct {
    allocator: std.mem.Allocator,
    root_path: []u8,
    shards: [][]u8,
    tensors: std.StringHashMap(TensorLoc),

    pub fn open(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !SafetensorsSource {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try dir.realPath(io, &path_buf);
        const root_path = try allocator.dupe(u8, path_buf[0..path_len]);
        errdefer allocator.free(root_path);

        var shard_names = std.ArrayList([]u8).empty;
        errdefer {
            for (shard_names.items) |s| allocator.free(s);
            shard_names.deinit(allocator);
        }

        if (dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(32 * 1024 * 1024))) |idx_raw| {
            defer allocator.free(idx_raw);
            const parsed = try std.json.parseFromSlice(std.json.Value, allocator, idx_raw, .{});
            defer parsed.deinit();
            if (parsed.value != .object) return error.BadSafetensorsIndex;
            const wm = parsed.value.object.get("weight_map") orelse return error.BadSafetensorsIndex;
            if (wm != .object) return error.BadSafetensorsIndex;
            var seen = std.StringHashMap(void).init(allocator);
            defer seen.deinit();
            var it = wm.object.iterator();
            while (it.next()) |e| {
                if (e.value_ptr.* != .string) return error.BadSafetensorsIndex;
                const fname = e.value_ptr.*.string;
                if (!seen.contains(fname)) {
                    const dup = try allocator.dupe(u8, fname);
                    try shard_names.append(allocator, dup);
                    try seen.put(dup, {});
                }
            }
        } else |_| {
            try shard_names.append(allocator, try allocator.dupe(u8, "model.safetensors"));
        }

        var tensors = std.StringHashMap(TensorLoc).init(allocator);
        errdefer {
            var it = tensors.iterator();
            while (it.next()) |e| {
                allocator.free(e.key_ptr.*);
                allocator.free(e.value_ptr.dtype);
                allocator.free(e.value_ptr.shape);
            }
            tensors.deinit();
        }

        for (shard_names.items, 0..) |fname, s_idx| {
            const full = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ root_path, fname }, 0);
            defer allocator.free(full);
            const fd = std.c.open(full.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
            if (fd < 0) return error.SourceOpenFailed;
            defer _ = std.c.close(fd);

            var len_buf: [8]u8 = undefined;
            try preadAll(fd, &len_buf, 0);
            const hdr_len = std.mem.readInt(u64, &len_buf, .little);
            if (hdr_len > 64 * 1024 * 1024) return error.BadSafetensorsHeader;
            const hdr = try allocator.alloc(u8, hdr_len);
            defer allocator.free(hdr);
            try preadAll(fd, hdr, 8);
            const parsed = try std.json.parseFromSlice(std.json.Value, allocator, hdr, .{});
            defer parsed.deinit();
            if (parsed.value != .object) return error.BadSafetensorsHeader;
            var it = parsed.value.object.iterator();
            while (it.next()) |entry| {
                if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;
                if (tensors.contains(entry.key_ptr.*)) continue;
                if (entry.value_ptr.* != .object) return error.BadSafetensorsHeader;
                const obj = entry.value_ptr.*.object;
                const dt = obj.get("dtype") orelse return error.BadSafetensorsHeader;
                const sh = obj.get("shape") orelse return error.BadSafetensorsHeader;
                const offs = obj.get("data_offsets") orelse return error.BadSafetensorsHeader;
                if (dt != .string or sh != .array or offs != .array or offs.array.items.len != 2) return error.BadSafetensorsHeader;
                const shape = try allocator.alloc(usize, sh.array.items.len);
                errdefer allocator.free(shape);
                for (sh.array.items, 0..) |d, i| {
                    if (d != .integer or d.integer < 0) return error.BadSafetensorsHeader;
                    shape[i] = @intCast(d.integer);
                }
                const start: u64 = @intCast(offs.array.items[0].integer);
                const end: u64 = @intCast(offs.array.items[1].integer);
                try tensors.put(try allocator.dupe(u8, entry.key_ptr.*), .{
                    .shard_idx = s_idx,
                    .data_start = 8 + hdr_len,
                    .start = start,
                    .end = end,
                    .dtype = try allocator.dupe(u8, dt.string),
                    .shape = shape,
                });
            }
        }
        return .{
            .allocator = allocator,
            .root_path = root_path,
            .shards = try shard_names.toOwnedSlice(allocator),
            .tensors = tensors,
        };
    }

    pub fn deinit(self: *SafetensorsSource) void {
        var it = self.tensors.iterator();
        while (it.next()) |e| {
            self.allocator.free(e.key_ptr.*);
            self.allocator.free(e.value_ptr.dtype);
            self.allocator.free(e.value_ptr.shape);
        }
        self.tensors.deinit();
        for (self.shards) |s| self.allocator.free(s);
        self.allocator.free(self.shards);
        self.allocator.free(self.root_path);
    }

    pub fn readRange(self: *const SafetensorsSource, loc: TensorLoc, rel_offset: u64, dst: []u8) !void {
        if (rel_offset + dst.len > loc.end - loc.start) return error.TruncatedSafetensors;
        var path_buf: [std.fs.max_path_bytes + 1]u8 = undefined;
        const s = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ self.root_path, self.shards[loc.shard_idx] });
        path_buf[s.len] = 0;
        const fd = std.c.open(path_buf[0..s.len :0].ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.SourceOpenFailed;
        defer _ = std.c.close(fd);
        try preadAll(fd, dst, loc.data_start + loc.start + rel_offset);
    }

    /// Streams one routed expert's public weight `[in_dim, out_dim]` in f32 without loading the whole shard.
    pub fn readExpertPublic(
        self: *const SafetensorsSource,
        allocator: std.mem.Allocator,
        arch: Arch,
        layer: u32,
        mtp: bool,
        role: ProjRole,
        expert: usize,
        in_dim: usize,
        out_dim: usize,
        dst_public: []f32,
    ) !void {
        if (dst_public.len != in_dim * out_dim) return error.ShapeMismatch;
        var key_buf: [192]u8 = undefined;
        switch (arch) {
            .qwen4_exp => {
                const pfx = if (mtp) "mtp.layers" else "model.language_model.layers";
                const leaf = if (role == .down) "down_proj" else "gate_up_proj";
                const key = try std.fmt.bufPrint(&key_buf, "{s}.{d}.mlp.experts.{s}", .{ pfx, layer, leaf });
                const loc = self.tensors.get(key) orelse return error.MissingTensor;
                if (!std.mem.eql(u8, loc.dtype, "BF16") or loc.shape.len != 3) return error.UnknownDtype;
                const exp_rows: usize = if (role == .down) out_dim else out_dim * 2;
                if (expert >= loc.shape[0] or loc.shape[1] != exp_rows or loc.shape[2] != in_dim) return error.BadShape;
                const row_offset: usize = if (role == .up) out_dim else 0;
                const byte_offset: u64 = @intCast(((expert * exp_rows) + row_offset) * in_dim * 2);
                const raw_u16 = try allocator.alloc(u16, out_dim * in_dim);
                defer allocator.free(raw_u16);
                try self.readRange(loc, byte_offset, std.mem.sliceAsBytes(raw_u16));
                try publicFromBf16(raw_u16, out_dim, in_dim, dst_public);
            },
            .glm5_next => {
                const key = try std.fmt.bufPrint(&key_buf, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.weight", .{ layer, expert, role.suffix() });
                const loc = self.tensors.get(key) orelse return error.MissingTensor;
                if (!std.mem.eql(u8, loc.dtype, "BF16") or loc.shape.len != 2) return error.UnknownDtype;
                if (loc.shape[0] != out_dim or loc.shape[1] != in_dim) return error.BadShape;
                const raw_u16 = try allocator.alloc(u16, out_dim * in_dim);
                defer allocator.free(raw_u16);
                try self.readRange(loc, 0, std.mem.sliceAsBytes(raw_u16));
                try publicFromBf16(raw_u16, out_dim, in_dim, dst_public);
            },
            .mimo_v2 => {
                var skey_buf: [192]u8 = undefined;
                const wkey = try std.fmt.bufPrint(&key_buf, "model.layers.{d}.mlp.experts.{d}.{s}.weight", .{ layer, expert, role.suffix() });
                const skey = try std.fmt.bufPrint(&skey_buf, "model.layers.{d}.mlp.experts.{d}.{s}.weight_scale", .{ layer, expert, role.suffix() });
                const wloc = self.tensors.get(wkey) orelse return error.MissingTensor;
                const sloc = self.tensors.get(skey) orelse return error.MissingTensor;
                if (!std.mem.eql(u8, wloc.dtype, "U8") or !std.mem.eql(u8, sloc.dtype, "U8")) return error.UnknownDtype;
                if (wloc.shape.len != 2 or sloc.shape.len != 2) return error.BadShape;
                if (wloc.shape[0] != out_dim or wloc.shape[1] != in_dim / 2 or sloc.shape[0] != out_dim or sloc.shape[1] != in_dim / 32) return error.BadShape;
                const w_bytes = try allocator.alloc(u8, out_dim * (in_dim / 2));
                defer allocator.free(w_bytes);
                const s_bytes = try allocator.alloc(u8, out_dim * (in_dim / 32));
                defer allocator.free(s_bytes);
                try self.readRange(wloc, 0, w_bytes);
                try self.readRange(sloc, 0, s_bytes);
                try publicFromMxfp4(w_bytes, s_bytes, out_dim, in_dim, dst_public);
            },
        }
    }
};

fn preadAll(fd: std.c.fd_t, dst: []u8, offset: u64) !void {
    var done: usize = 0;
    while (done < dst.len) {
        const got = std.c.pread(fd, dst[done..].ptr, dst.len - done, @intCast(offset + done));
        if (got < 0) {
            if (std.c._errno().* == @backingInt(std.c.E.INTR)) continue;
            return error.SourceReadFailed;
        }
        if (got == 0) return error.TruncatedSafetensors;
        done += @intCast(got);
    }
}

test "source: BF16, FP8-block and MXFP4 readers match the reference dequant fixtures" {
    const t = std.testing;
    var file = try st.parse(t.allocator, st.sources);
    defer file.deinit(t.allocator);

    const bf16_raw = st.slice(u16, file.tensors.get("bf16.raw").?);
    const bf16_want = st.slice(f32, file.tensors.get("bf16.f32").?);
    const bf16_pub_want = st.slice(f32, file.tensors.get("bf16.public").?);
    const bf16_got = try t.allocator.alloc(f32, bf16_want.len);
    defer t.allocator.free(bf16_got);
    try decodeBf16ToF32(bf16_raw, bf16_got);
    try t.expectEqualSlices(f32, bf16_want, bf16_got);

    const bf16_pub_got = try t.allocator.alloc(f32, bf16_pub_want.len);
    defer t.allocator.free(bf16_pub_got);
    try publicFromBf16(bf16_raw, 32, 128, bf16_pub_got);
    try t.expectEqualSlices(f32, bf16_pub_want, bf16_pub_got);

    const fp8_codes = file.tensors.get("fp8.weight").?.bytes;
    const fp8_scales = st.slice(f32, file.tensors.get("fp8.scale").?);
    const fp8_want = st.slice(f32, file.tensors.get("fp8.f32").?);
    const fp8_got = try t.allocator.alloc(f32, fp8_want.len);
    defer t.allocator.free(fp8_got);
    try decodeFp8Block(fp8_codes, fp8_scales, 160, 256, 128, fp8_got);
    try t.expectEqualSlices(f32, fp8_want, fp8_got);

    const mxfp4_w = file.tensors.get("mxfp4.weight").?.bytes;
    const mxfp4_s = file.tensors.get("mxfp4.scale").?.bytes;
    const mxfp4_want = st.slice(f32, file.tensors.get("mxfp4.f32").?);
    const mxfp4_pub_want = st.slice(f32, file.tensors.get("mxfp4.public").?);
    const mxfp4_got = try t.allocator.alloc(f32, mxfp4_want.len);
    defer t.allocator.free(mxfp4_got);
    try decodeMxfp4(mxfp4_w, mxfp4_s, 64, 128, mxfp4_got);
    try t.expectEqualSlices(f32, mxfp4_want, mxfp4_got);

    const mxfp4_pub_got = try t.allocator.alloc(f32, mxfp4_pub_want.len);
    defer t.allocator.free(mxfp4_pub_got);
    try publicFromMxfp4(mxfp4_w, mxfp4_s, 64, 128, mxfp4_pub_got);
    try t.expectEqualSlices(f32, mxfp4_pub_want, mxfp4_pub_got);

    try t.expectError(error.NonFiniteSourceWeight, decodeFp8E4m3Byte(0x7F));
    try t.expectError(error.NonFiniteSourceWeight, decodeFp8E4m3Byte(0xFF));
    try t.expectError(error.NonFiniteSourceWeight, e8m0Value(0xFF));
}

test "source: trunk affine quantization matches mlx_quantize bit for bit" {
    const t = std.testing;
    var file = try st.parse(t.allocator, st.sources);
    defer file.deinit(t.allocator);

    const bf16_raw = st.slice(u16, file.tensors.get("bf16.raw").?);
    const Case = struct { prefix: []const u8, bits: u32, gs: u32 };
    const cases = [_]Case{
        .{ .prefix = "affine_q8_g64", .bits = 8, .gs = 64 },
        .{ .prefix = "affine_q6_g128", .bits = 6, .gs = 128 },
        .{ .prefix = "affine_q4_g64", .bits = 4, .gs = 64 },
    };
    for (cases) |c| {
        var triple = try quantizeAffineBf16(t.allocator, bf16_raw, 32, 128, c.bits, c.gs);
        defer triple.deinit(t.allocator);
        var kbuf: [64]u8 = undefined;
        const want_w = st.slice(u32, file.tensors.get(try std.fmt.bufPrint(&kbuf, "{s}.weight", .{c.prefix})).?);
        const want_s = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&kbuf, "{s}.scales", .{c.prefix})).?);
        const want_b = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&kbuf, "{s}.biases", .{c.prefix})).?);
        try t.expectEqualSlices(u32, want_w, triple.weight);
        try t.expectEqualSlices(u16, want_s, triple.scales);
        try t.expectEqualSlices(u16, want_b, triple.biases);
    }
}

test "source: architecture identification and per-tensor trunk rules match served packs and refuse unknown sources" {
    const t = std.testing;
    try t.expectEqual(Arch.qwen4_exp, try identifyConfigJson(t.allocator, "{\"model_type\":\"qwen4_exp\"}"));
    try t.expectEqual(Arch.mimo_v2, try identifyConfigJson(t.allocator, "{\"model_type\":\"mimo_v2\"}"));
    try t.expectEqual(Arch.glm5_next, try identifyConfigJson(t.allocator, "{\"model_type\":\"glm5_next\"}"));
    try t.expectError(error.ArchitectureUnsupported, identifyConfigJson(t.allocator, "{\"model_type\":\"llama\"}"));
    try t.expectError(error.UnknownSourceFormat, identifyConfigJson(t.allocator, "{}"));

    const q_in_proj_a = trunkSpecForTensor(.qwen4_exp, "language_model.model.layers.0.linear_attn.in_proj_a.weight", "BF16", &.{ 48, 2560 }, 8, 64);
    try t.expectEqual(@as(u32, 64), q_in_proj_a.affine.group_size);
    try t.expect(!q_in_proj_a.affine.override_in_config);
    const q_lm_head = trunkSpecForTensor(.qwen4_exp, "lm_head.weight", "BF16", &.{ 248320, 2560 }, 8, 64);
    try t.expectEqual(@as(u32, 64), q_lm_head.affine.group_size);
    const q_hc_up = trunkSpecForTensor(.qwen4_exp, "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up.weight", "BF16", &.{ 10240, 320 }, 8, 64);
    try t.expectEqual(@as(u32, 64), q_hc_up.affine.group_size);
    const q_router = trunkSpecForTensor(.qwen4_exp, "language_model.model.layers.0.mlp.gate.weight", "BF16", &.{ 512, 2560 }, 8, 64);
    try t.expect(q_router == .retain_source);

    const glm_q = trunkSpecForTensor(.glm5_next, "model.language_model.layers.0.self_attn.q_proj.weight", "BF16", &.{ 4096, 4096 }, 6, 128);
    try t.expectEqual(@as(u32, 6), glm_q.affine.bits);
    try t.expectEqual(@as(u32, 128), glm_q.affine.group_size);
    const glm_small = trunkSpecForTensor(.glm5_next, "model.language_model.layers.3.self_attn.indexer.wk.weight", "BF16", &.{ 128, 4096 }, 6, 128);
    try t.expect(glm_small == .retain_source);

    const mimo_o = trunkSpecForTensor(.mimo_v2, "model.layers.0.self_attn.o_proj.weight", "BF16", &.{ 4096, 12288 }, 8, 64);
    try t.expectEqual(@as(u32, 8), mimo_o.affine.bits);
    const mimo_qkv = trunkSpecForTensor(.mimo_v2, "model.layers.0.self_attn.qkv_proj.weight", "F8_E4M3", &.{ 14848, 4096 }, 8, 64);
    try t.expect(mimo_qkv == .retain_source);
}

test "source: SafetensorsSource streams per-expert public weights for Qwen, GLM and MiMo and refuses GGUF" {
    const t = std.testing;
    const io = t.io;
    var file = try st.parse(t.allocator, st.sources);
    defer file.deinit(t.allocator);

    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.gguf", .data = "GGUF" });
    try t.expectError(error.ModelFormatUnsupported, identifySourceDir(t.allocator, io, tmp.dir));
    try tmp.dir.deleteFile(io, "model.gguf");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"mimo_v2\"}" });
    try t.expectEqual(Arch.mimo_v2, try identifySourceDir(t.allocator, io, tmp.dir));

    const bf16_raw = file.tensors.get("bf16.raw").?.bytes;
    const mxfp4_w = file.tensors.get("mxfp4.weight").?.bytes;
    const mxfp4_s = file.tensors.get("mxfp4.scale").?.bytes;
    var payload = std.ArrayList(u8).empty;
    defer payload.deinit(t.allocator);
    try payload.appendSlice(t.allocator, bf16_raw);
    try payload.appendSlice(t.allocator, mxfp4_w);
    try payload.appendSlice(t.allocator, mxfp4_s);

    const b_end = bf16_raw.len;
    const mw_end = b_end + mxfp4_w.len;
    const ms_end = mw_end + mxfp4_s.len;
    var hdr = std.ArrayList(u8).empty;
    defer hdr.deinit(t.allocator);
    try hdr.print(t.allocator,
        \\{{"model.language_model.layers.0.mlp.experts.gate_up_proj":{{"dtype":"BF16","shape":[1,32,128],"data_offsets":[0,{d}]}},
        \\"model.language_model.layers.3.mlp.experts.0.gate_proj.weight":{{"dtype":"BF16","shape":[32,128],"data_offsets":[0,{d}]}},
        \\"model.layers.1.mlp.experts.0.gate_proj.weight":{{"dtype":"U8","shape":[64,64],"data_offsets":[{d},{d}]}},
        \\"model.layers.1.mlp.experts.0.gate_proj.weight_scale":{{"dtype":"U8","shape":[64,4],"data_offsets":[{d},{d}]}}}}
    , .{ b_end, b_end, b_end, mw_end, mw_end, ms_end });
    while (hdr.items.len % 8 != 0) try hdr.append(t.allocator, ' ');

    var st_file = std.ArrayList(u8).empty;
    defer st_file.deinit(t.allocator);
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, hdr.items.len, .little);
    try st_file.appendSlice(t.allocator, &len_buf);
    try st_file.appendSlice(t.allocator, hdr.items);
    try st_file.appendSlice(t.allocator, payload.items);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = st_file.items });

    var src = try SafetensorsSource.open(t.allocator, io, tmp.dir);
    defer src.deinit();

    const qwen_gate = try t.allocator.alloc(f32, 128 * 16);
    defer t.allocator.free(qwen_gate);
    try src.readExpertPublic(t.allocator, .qwen4_exp, 0, false, .gate, 0, 128, 16, qwen_gate);
    const qwen_up = try t.allocator.alloc(f32, 128 * 16);
    defer t.allocator.free(qwen_up);
    try src.readExpertPublic(t.allocator, .qwen4_exp, 0, false, .up, 0, 128, 16, qwen_up);

    const glm_gate = try t.allocator.alloc(f32, 128 * 32);
    defer t.allocator.free(glm_gate);
    try src.readExpertPublic(t.allocator, .glm5_next, 3, false, .gate, 0, 128, 32, glm_gate);
    try t.expectEqualSlices(f32, st.slice(f32, file.tensors.get("bf16.public").?), glm_gate);

    for (0..128) |r| {
        try t.expectEqualSlices(f32, glm_gate[r * 32 ..][0..16], qwen_gate[r * 16 ..][0..16]);
        try t.expectEqualSlices(f32, glm_gate[r * 32 + 16 ..][0..16], qwen_up[r * 16 ..][0..16]);
    }

    const mimo_gate = try t.allocator.alloc(f32, 128 * 64);
    defer t.allocator.free(mimo_gate);
    try src.readExpertPublic(t.allocator, .mimo_v2, 1, false, .gate, 0, 128, 64, mimo_gate);
    try t.expectEqualSlices(f32, st.slice(f32, file.tensors.get("mxfp4.public").?), mimo_gate);
}
