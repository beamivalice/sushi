//! Per-input-channel activation statistics ("imatrix") captured while the engine
//! serves a checkpoint with expert streaming, in the exact contract
//! sashimi writes and reads.
//!
//! Per layer the file carries three entries, whatever the architecture:
//!   <experts>.gate_up_proj        [E * hidden], expert e at [e*hidden, (e+1)*hidden)
//!   <experts>.down_proj           [E * inter]
//!   <experts>.gate_up_proj.rows   [E] tokens routed to each expert
//! The values are sum(x^2) over the rows routed to the expert divided by the
//! LAYER's token count, so a busy expert keeps its larger vote.
//!
//! `<experts>` is the SOURCE checkpoint's own name for the layer's expert
//! block, which is per-arch (`Arch`): Qwen3.8-Flash-Next stores the routed
//! experts as one fused pair of tensors under
//! `model.language_model.layers.{L}.mlp.experts.{gate_up_proj,down_proj}`,
//! MiMo V2.6 Flash one tensor per expert under
//! `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight`. Both are keyed
//! here by the layer's `…mlp.experts.` prefix and the flat per-layer layout
//! above; a converter slices expert e out of it. gate and up share one statistic
//! because they read the same MLP input row.
//!
//! A dense linear the forward reports (`observeLinear`: MiMo's o_proj and
//! lm_head) is keyed by its source weight name, e.g.
//! `model.layers.{L}.self_attn.o_proj.weight` [in] = sum(x^2) over the rows it
//! read divided by that row count, beside `<name>.rows` [1].
//!
//! Opt-in through `SUSHI_IMATRIX_OUT=<abs>.safetensors`; absent or empty = off,
//! and nothing is allocated.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

pub const ENV_VAR = "SUSHI_IMATRIX_OUT";

/// The architectures with an expert-naming contract. The converter looks the
/// imatrix up by SOURCE HF weight name, never by the engine's internal module
/// names, and those names differ per checkpoint.
pub const Arch = enum {
    qwen4_exp,
    mimo_v2,
    glm5_next,

    /// The SOURCE checkpoint's decoder-layer prefix.
    pub fn layerPrefix(self: Arch) []const u8 {
        return switch (self) {
            .qwen4_exp, .glm5_next => "model.language_model.layers.",
            .mimo_v2 => "model.layers.",
        };
    }

    pub fn fromModelType(model_type: []const u8) ?Arch {
        if (std.mem.eql(u8, model_type, "qwen4_exp")) return .qwen4_exp;
        if (std.mem.eql(u8, model_type, "mimo_v2")) return .mimo_v2;
        if (std.mem.eql(u8, model_type, "glm5_next")) return .glm5_next;
        return null;
    }
};

/// Absolute output path from the environment, or null when capture is off.
pub fn envPath() ?[]const u8 {
    const raw = std.c.getenv(ENV_VAR) orelse return null;
    const s = std.mem.sliceTo(raw, 0);
    return if (s.len == 0) null else s;
}

const Layer = struct {
    gu: mlx.mlx_array = .{ .ctx = null },
    down: mlx.mlx_array = .{ .ctx = null },
    rows: mlx.mlx_array = .{ .ctx = null },
    tokens: u64 = 0,

    fn deinit(self: *Layer) void {
        if (self.gu.ctx != null) _ = mlx.mlx_array_free(self.gu);
        if (self.down.ctx != null) _ = mlx.mlx_array_free(self.down);
        if (self.rows.ctx != null) _ = mlx.mlx_array_free(self.rows);
        self.* = .{};
    }
};

/// A dense linear whose input rows the forward reports.
pub const Linear = union(enum) {
    o_proj: usize,
    lm_head,
};

const Dense = struct {
    acc: mlx.mlx_array = .{ .ctx = null },
    rows: u64 = 0,

    fn deinit(self: *Dense) void {
        if (self.acc.ctx != null) _ = mlx.mlx_array_free(self.acc);
        self.* = .{};
    }
};

pub const Collector = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    path: []u8,
    experts: c_int,
    arch: Arch,
    /// arange [1, E] int32 — the comparand every one-hot count broadcasts against.
    axis: mlx.mlx_array,
    layers: []Layer,
    o_proj: []Dense,
    lm_head: Dense = .{},

    pub fn init(allocator: std.mem.Allocator, s: mlx.mlx_stream, path: []const u8, num_layers: usize, experts: c_int, arch: Arch) !*Collector {
        if (experts <= 0 or num_layers == 0) return error.ImatrixBadGeometry;
        const self = try allocator.create(Collector);
        errdefer allocator.destroy(self);
        const owned = try allocator.dupe(u8, path);
        errdefer allocator.free(owned);
        const layers = try allocator.alloc(Layer, num_layers);
        errdefer allocator.free(layers);
        for (layers) |*l| l.* = .{};
        const o_proj = try allocator.alloc(Dense, num_layers);
        errdefer allocator.free(o_proj);
        for (o_proj) |*d| d.* = .{};
        var axis = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(axis);
        try mlx.check(mlx.mlx_arange(&axis, 0, @floatFromInt(experts), 1, .int32, s));
        var axis_2d = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(axis_2d);
        try mlx.check(mlx.mlx_reshape(&axis_2d, axis, &[_]c_int{ 1, experts }, 2, s));
        _ = mlx.mlx_array_free(axis);
        self.* = .{
            .allocator = allocator,
            .s = s,
            .path = owned,
            .experts = experts,
            .arch = arch,
            .axis = axis_2d,
            .layers = layers,
            .o_proj = o_proj,
        };
        return self;
    }

    /// `init` from the environment; null when capture is off or the model type
    /// has no expert-naming contract.
    pub fn forModel(allocator: std.mem.Allocator, s: mlx.mlx_stream, model_type: []const u8, num_layers: usize, experts: c_int) !?*Collector {
        const path = envPath() orelse return null;
        const arch = Arch.fromModelType(model_type) orelse {
            log.warn("[imatrix] no expert naming contract for model_type {s}: capture off\n", .{model_type});
            return null;
        };
        const self = try init(allocator, s, path, num_layers, experts, arch);
        log.info("[imatrix] expert activation capture armed: {s} ({s}, {d} layers, {d} experts)\n", .{ path, @tagName(arch), num_layers, experts });
        return self;
    }

    pub fn deinit(self: *Collector) void {
        for (self.layers) |*l| l.deinit();
        self.allocator.free(self.layers);
        for (self.o_proj) |*d| d.deinit();
        self.allocator.free(self.o_proj);
        self.lm_head.deinit();
        _ = mlx.mlx_array_free(self.axis);
        self.allocator.free(self.path);
        self.allocator.destroy(self);
    }

    /// `counts[r, e]` = how many of row r's top-k slots picked expert e. One
    /// [rows, E] compare per slot rather than one [rows, k, E] tensor: k is small
    /// and the 3-D form costs k times the transient at prefill widths.
    fn counts(self: *Collector, ids: mlx.mlx_array, rows: c_int, k: c_int) !mlx.mlx_array {
        var acc = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(acc);
        try mlx.check(mlx.mlx_zeros(&acc, &[_]c_int{ rows, self.experts }, 2, .float32, self.s));
        var j: c_int = 0;
        while (j < k) : (j += 1) {
            var col = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(col);
            try mlx.check(mlx.mlx_slice(&col, ids, &[_]c_int{ 0, j }, 2, &[_]c_int{ rows, j + 1 }, 2, &[_]c_int{ 1, 1 }, 2, self.s));
            var eq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eq);
            try mlx.check(mlx.mlx_equal(&eq, col, self.axis, self.s));
            var hot = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(hot);
            try mlx.check(mlx.mlx_astype(&hot, eq, .float32, self.s));
            var sum = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(sum);
            try mlx.check(mlx.mlx_add(&sum, acc, hot, self.s));
            _ = mlx.mlx_array_free(acc);
            acc = sum;
        }
        return acc;
    }

    /// `acc[e, c] += sum over rows routed to e of values[r, c]^2`, as one
    /// [E, rows] x [rows, C] matmul against the one-hot count matrix.
    fn addOuter(self: *Collector, acc: *mlx.mlx_array, cnt: mlx.mlx_array, values: mlx.mlx_array) !void {
        var v32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v32);
        try mlx.check(mlx.mlx_astype(&v32, values, .float32, self.s));
        var sq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sq);
        try mlx.check(mlx.mlx_square(&sq, v32, self.s));
        var cnt_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cnt_t);
        try mlx.check(mlx.mlx_transpose(&cnt_t, cnt, self.s));
        var prod = mlx.mlx_array_new();
        mlx.check(mlx.mlx_matmul(&prod, cnt_t, sq, self.s)) catch |e| {
            _ = mlx.mlx_array_free(prod);
            return e;
        };
        try self.fold(acc, prod);
    }

    /// Fold `term` (ownership taken) into `acc` and evaluate: the accumulator is
    /// outside the forward's own graph, so without the eval every chunk's inputs
    /// would stay pinned by an ever-growing add chain.
    fn fold(self: *Collector, acc: *mlx.mlx_array, term: mlx.mlx_array) !void {
        if (acc.ctx == null) {
            acc.* = term;
        } else {
            var sum = mlx.mlx_array_new();
            mlx.check(mlx.mlx_add(&sum, acc.*, term, self.s)) catch |e| {
                _ = mlx.mlx_array_free(sum);
                _ = mlx.mlx_array_free(term);
                return e;
            };
            _ = mlx.mlx_array_free(term);
            _ = mlx.mlx_array_free(acc.*);
            acc.* = sum;
        }
        try mlx.check(mlx.mlx_array_eval(acc.*));
    }

    /// The MLP input rows behind one MoE layer's chunk. `x_rows` is [rows, hidden],
    /// `ids` the [rows, top_k] GLOBAL expert ids (never remapped slab slots).
    pub fn observeGateUp(self: *Collector, layer: usize, x_rows: mlx.mlx_array, ids: mlx.mlx_array) !void {
        if (layer >= self.layers.len) return error.ImatrixLayerOutOfRange;
        const xs = mlx.getShape(x_rows);
        const is = mlx.getShape(ids);
        if (xs.len != 2 or is.len != 2 or is[0] != xs[0] or xs[0] <= 0 or is[1] <= 0) return error.ImatrixBadShape;
        const cnt = try self.counts(ids, is[0], is[1]);
        defer _ = mlx.mlx_array_free(cnt);
        const slot = &self.layers[layer];
        try self.addOuter(&slot.gu, cnt, x_rows);
        var routed = mlx.mlx_array_new();
        mlx.check(mlx.mlx_sum_axis(&routed, cnt, 0, false, self.s)) catch |e| {
            _ = mlx.mlx_array_free(routed);
            return e;
        };
        try self.fold(&slot.rows, routed);
        slot.tokens += @intCast(xs[0]);
    }

    /// The SwiGLU activation rows feeding down. `act_rows` is [n, inter] and `ids`
    /// the [n, 1] global expert id of each of those rows, in the SAME row order.
    pub fn observeDown(self: *Collector, layer: usize, act_rows: mlx.mlx_array, ids: mlx.mlx_array) !void {
        if (layer >= self.layers.len) return error.ImatrixLayerOutOfRange;
        const as = mlx.getShape(act_rows);
        const is = mlx.getShape(ids);
        if (as.len != 2 or is.len != 2 or is[0] != as[0] or is[1] != 1 or as[0] <= 0) return error.ImatrixBadShape;
        const cnt = try self.counts(ids, is[0], 1);
        defer _ = mlx.mlx_array_free(cnt);
        try self.addOuter(&self.layers[layer].down, cnt, act_rows);
    }

    /// A dense linear's input `x` ([..., in], any leading shape): its per-channel
    /// sum of squares over every row.
    pub fn observeLinear(self: *Collector, which: Linear, x: mlx.mlx_array) !void {
        const slot = switch (which) {
            .o_proj => |layer| if (layer < self.o_proj.len) &self.o_proj[layer] else return error.ImatrixLayerOutOfRange,
            .lm_head => &self.lm_head,
        };
        const xs = mlx.getShape(x);
        if (xs.len == 0 or xs[xs.len - 1] <= 0) return error.ImatrixBadShape;
        const channels = xs[xs.len - 1];
        const rows = mlx.mlx_array_size(x) / @as(usize, @intCast(channels));
        if (rows == 0) return;
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ @intCast(rows), channels }, 2, self.s));
        var v32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v32);
        try mlx.check(mlx.mlx_astype(&v32, flat, .float32, self.s));
        var sq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sq);
        try mlx.check(mlx.mlx_square(&sq, v32, self.s));
        var sum = mlx.mlx_array_new();
        mlx.check(mlx.mlx_sum_axis(&sum, sq, 0, false, self.s)) catch |e| {
            _ = mlx.mlx_array_free(sum);
            return e;
        };
        try self.fold(&slot.acc, sum);
        slot.rows += rows;
    }

    /// `name` = sum / rows beside `name.rows`; nothing for a linear that never ran.
    fn putDense(self: *Collector, map: mlx.mlx_map_string_to_array, held: *std.ArrayList(mlx.mlx_array), name: []const u8, d: *const Dense) !usize {
        if (d.rows == 0 or d.acc.ctx == null) return 0;
        var key_buf: [192]u8 = undefined;
        const denom = mlx.mlx_array_new_float(@floatFromInt(d.rows));
        defer _ = mlx.mlx_array_free(denom);
        var mean = mlx.mlx_array_new();
        mlx.check(mlx.mlx_divide(&mean, d.acc, denom, self.s)) catch |e| {
            _ = mlx.mlx_array_free(mean);
            return e;
        };
        held.append(self.allocator, mean) catch |e| {
            _ = mlx.mlx_array_free(mean);
            return e;
        };
        try mlx.check(mlx.mlx_array_eval(mean));
        try mlx.check(mlx.mlx_map_string_to_array_insert(map, try std.fmt.bufPrintSentinel(&key_buf, "{s}", .{name}, 0), mean));
        const count: f32 = @floatFromInt(d.rows);
        const rows = mlx.mlx_array_new_data(&count, &[_]c_int{1}, 1, .float32);
        held.append(self.allocator, rows) catch |e| {
            _ = mlx.mlx_array_free(rows);
            return e;
        };
        try mlx.check(mlx.mlx_map_string_to_array_insert(map, try std.fmt.bufPrintSentinel(&key_buf, "{s}.rows", .{name}, 0), rows));
        return 2;
    }

    fn scaledFlat(self: *Collector, acc: mlx.mlx_array, denom: mlx.mlx_array) !mlx.mlx_array {
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_divide(&scaled, acc, denom, self.s));
        const shape = mlx.getShape(scaled);
        var flat = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, scaled, &[_]c_int{shape[0] * shape[1]}, 1, self.s));
        try mlx.check(mlx.mlx_array_eval(flat));
        return flat;
    }

    /// The raw accumulators of every expert layer (no normalization), so a capture that resumes continues the sums.
    /// `path` must end in `.safetensors`: MLX appends the suffix otherwise.
    pub fn saveState(self: *Collector, path: []const u8) !void {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const tokens = try self.allocator.alloc(u32, self.layers.len);
        defer self.allocator.free(tokens);
        var key_buf: [64]u8 = undefined;
        for (self.layers, tokens, 0..) |*slot, *t, li| {
            t.* = @intCast(slot.tokens);
            if (slot.tokens == 0) continue;
            inline for (.{ .{ "gu", &slot.gu }, .{ "down", &slot.down }, .{ "rows", &slot.rows } }) |entry| {
                try mlx.check(mlx.mlx_array_eval(entry[1].*));
                try mlx.check(mlx.mlx_map_string_to_array_insert(map, try std.fmt.bufPrintSentinel(&key_buf, "{s}.{d}", .{ entry[0], li }, 0), entry[1].*));
            }
        }
        const count = mlx.mlx_array_new_data(tokens.ptr, &[_]c_int{@intCast(tokens.len)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(count);
        try mlx.check(mlx.mlx_map_string_to_array_insert(map, "tokens", count));
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        const path_z = try std.fmt.allocPrintSentinel(self.allocator, "{s}", .{path}, 0);
        defer self.allocator.free(path_z);
        try mlx.check(mlx.mlx_save_safetensors(path_z.ptr, map, meta));
    }

    /// Replaces every accumulator with the ones `saveState` wrote; a file for other geometry is refused.
    pub fn loadState(self: *Collector, path: []const u8) !void {
        var loaded = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(loaded);
        var meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        const path_z = try std.fmt.allocPrintSentinel(self.allocator, "{s}", .{path}, 0);
        defer self.allocator.free(path_z);
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        try mlx.check(mlx.mlx_load_safetensors(&loaded, &meta, path_z.ptr, cpu));
        var count = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(count);
        try mlx.check(mlx.mlx_map_string_to_array_get(&count, loaded, "tokens"));
        try mlx.check(mlx.mlx_array_eval(count));
        const tokens = mlx.mlx_array_data_uint32(count) orelse return error.ImatrixStateCorrupt;
        if (mlx.mlx_array_size(count) != self.layers.len) return error.ImatrixStateGeometry;
        var key_buf: [64]u8 = undefined;
        for (self.layers, 0..) |*slot, li| {
            slot.deinit();
            slot.tokens = tokens[li];
            if (slot.tokens == 0) continue;
            inline for (.{ .{ "gu", &slot.gu, @as(c_int, 1) }, .{ "down", &slot.down, @as(c_int, 1) }, .{ "rows", &slot.rows, @as(c_int, 1) } }) |entry| {
                try mlx.check(mlx.mlx_map_string_to_array_get(entry[1], loaded, try std.fmt.bufPrintSentinel(&key_buf, "{s}.{d}", .{ entry[0], li }, 0)));
                try mlx.check(mlx.mlx_array_eval(entry[1].*));
                if (mlx.getShape(entry[1].*)[0] != self.experts) return error.ImatrixStateGeometry;
            }
        }
    }

    /// Write the safetensors file. Returns its byte count.
    pub fn flush(self: *Collector) !u64 {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        var held: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (held.items) |a| _ = mlx.mlx_array_free(a);
            held.deinit(self.allocator);
        }
        var key_buf: [192]u8 = undefined;
        var entries: usize = 0;
        for (self.layers, 0..) |*slot, li| {
            if (slot.tokens == 0 or slot.gu.ctx == null or slot.down.ctx == null) continue;
            const denom = mlx.mlx_array_new_float(@floatFromInt(slot.tokens));
            defer _ = mlx.mlx_array_free(denom);
            const gu = try self.scaledFlat(slot.gu, denom);
            try held.append(self.allocator, gu);
            const down = try self.scaledFlat(slot.down, denom);
            try held.append(self.allocator, down);
            try mlx.check(mlx.mlx_array_eval(slot.rows));
            const prefix = self.arch.layerPrefix();
            const gu_key = try std.fmt.bufPrintSentinel(&key_buf, "{s}{d}.mlp.experts.gate_up_proj", .{ prefix, li }, 0);
            try mlx.check(mlx.mlx_map_string_to_array_insert(map, gu_key.ptr, gu));
            const rows_key = try std.fmt.bufPrintSentinel(&key_buf, "{s}{d}.mlp.experts.gate_up_proj.rows", .{ prefix, li }, 0);
            try mlx.check(mlx.mlx_map_string_to_array_insert(map, rows_key.ptr, slot.rows));
            const down_key = try std.fmt.bufPrintSentinel(&key_buf, "{s}{d}.mlp.experts.down_proj", .{ prefix, li }, 0);
            try mlx.check(mlx.mlx_map_string_to_array_insert(map, down_key.ptr, down));
            entries += 3;
        }
        var name_buf: [192]u8 = undefined;
        for (self.o_proj, 0..) |*d, li| {
            const name = try std.fmt.bufPrint(&name_buf, "{s}{d}.self_attn.o_proj.weight", .{ self.arch.layerPrefix(), li });
            entries += try self.putDense(map, &held, name, d);
        }
        entries += try self.putDense(map, &held, "lm_head.weight", &self.lm_head);
        if (entries == 0) return error.ImatrixNothingCaptured;

        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        _ = mlx.mlx_map_string_to_string_insert(meta, "keys", "SOURCE checkpoint weight names");
        _ = mlx.mlx_map_string_to_string_insert(meta, "values", "experts: sum(x^2)/layer tokens, per expert concatenated; dense linears: sum(x^2)/rows");
        _ = mlx.mlx_map_string_to_string_insert(meta, "producer", "sushi " ++ ENV_VAR);

        const path_z = try std.fmt.allocPrintSentinel(self.allocator, "{s}", .{self.path}, 0);
        defer self.allocator.free(path_z);
        try mlx.check(mlx.mlx_save_safetensors(path_z.ptr, map, meta));
        const io = std.Io.Threaded.global_single_threaded.io();
        const stat = std.Io.Dir.cwd().statFile(io, self.path, .{}) catch |e| {
            log.warn("[imatrix] wrote {s}: size unavailable ({s})\n", .{ self.path, @errorName(e) });
            return 0;
        };
        log.info("[imatrix] wrote {s}: {d} entries, {d} bytes\n", .{ self.path, entries, stat.size });
        return stat.size;
    }
};

// ── tests ──

const testing = std.testing;
const QWEN_PREFIX = Arch.qwen4_exp.layerPrefix();

fn f32At(arr: mlx.mlx_array, i: usize) !f32 {
    try mlx.check(mlx.mlx_array_eval(arr));
    const p = mlx.mlx_array_data_float32(arr) orelse return error.Unreadable;
    return p[i];
}

test "imatrix accumulates routed sums, rows and token counts by expert" {
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    const E: c_int = 3;
    const H: c_int = 2;
    const I: c_int = 2;

    const col = try Collector.init(alloc, s, "/dev/null", 2, E, .qwen4_exp);
    defer col.deinit();

    // rows 0,1; top-2 routing: row0 -> {0,2}, row1 -> {2,2} (a duplicate slot counts twice).
    const x = [_]f32{ 1, 2, 3, 4 };
    const xa = mlx.mlx_array_new_data(&x, &[_]c_int{ 2, H }, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);
    const ids = [_]i32{ 0, 2, 2, 2 };
    const ida = mlx.mlx_array_new_data(&ids, &[_]c_int{ 2, 2 }, 2, .int32);
    defer _ = mlx.mlx_array_free(ida);
    try col.observeGateUp(1, xa, ida);

    const gu = col.layers[1].gu;
    // expert 0: row0 only -> 1, 4;  expert 1: none;  expert 2: row0 + 2*row1 -> 1+18, 4+32
    try testing.expectEqual(@as(f32, 1), try f32At(gu, 0));
    try testing.expectEqual(@as(f32, 4), try f32At(gu, 1));
    try testing.expectEqual(@as(f32, 0), try f32At(gu, 2));
    try testing.expectEqual(@as(f32, 0), try f32At(gu, 3));
    try testing.expectEqual(@as(f32, 19), try f32At(gu, 4));
    try testing.expectEqual(@as(f32, 36), try f32At(gu, 5));

    const rows = col.layers[1].rows;
    try testing.expectEqual(@as(f32, 1), try f32At(rows, 0));
    try testing.expectEqual(@as(f32, 0), try f32At(rows, 1));
    try testing.expectEqual(@as(f32, 3), try f32At(rows, 2));
    try testing.expectEqual(@as(u64, 2), col.layers[1].tokens);

    // The activation rows carry one id each, in row order.
    const act = [_]f32{ 1, 1, 2, 2, 3, 3, 4, 4 };
    const acta = mlx.mlx_array_new_data(&act, &[_]c_int{ 4, I }, 2, .float32);
    defer _ = mlx.mlx_array_free(acta);
    const aids = [_]i32{ 0, 2, 2, 2 };
    const aida = mlx.mlx_array_new_data(&aids, &[_]c_int{ 4, 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(aida);
    try col.observeDown(1, acta, aida);

    const dn = col.layers[1].down;
    try testing.expectEqual(@as(f32, 1), try f32At(dn, 0));
    try testing.expectEqual(@as(f32, 0), try f32At(dn, 2));
    try testing.expectEqual(@as(f32, 4 + 9 + 16), try f32At(dn, 4));

    // A second chunk adds on top and keeps counting tokens.
    try col.observeGateUp(1, xa, ida);
    try testing.expectEqual(@as(f32, 2), try f32At(col.layers[1].gu, 0));
    try testing.expectEqual(@as(f32, 6), try f32At(col.layers[1].rows, 2));
    try testing.expectEqual(@as(u64, 4), col.layers[1].tokens);

    // An untouched layer stays empty.
    try testing.expect(col.layers[0].gu.ctx == null);
    try testing.expectEqual(@as(u64, 0), col.layers[0].tokens);
}

test "imatrix writes the converter's keys, shapes and layer-normalized values" {
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    const E: c_int = 3;

    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const out = try std.fs.path.join(alloc, &.{ dir, "imatrix.safetensors" });
    defer alloc.free(out);

    const col = try Collector.init(alloc, s, out, 2, E, .qwen4_exp);
    defer col.deinit();

    const x = [_]f32{ 1, 2, 3, 4 };
    const xa = mlx.mlx_array_new_data(&x, &[_]c_int{ 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);
    const ids = [_]i32{ 0, 2, 2, 2 };
    const ida = mlx.mlx_array_new_data(&ids, &[_]c_int{ 2, 2 }, 2, .int32);
    defer _ = mlx.mlx_array_free(ida);
    try col.observeGateUp(1, xa, ida);
    const act = [_]f32{ 1, 1, 2, 2, 3, 3, 4, 4 };
    const acta = mlx.mlx_array_new_data(&act, &[_]c_int{ 4, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(acta);
    const aids = [_]i32{ 0, 2, 2, 2 };
    const aida = mlx.mlx_array_new_data(&aids, &[_]c_int{ 4, 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(aida);
    try col.observeDown(1, acta, aida);

    const bytes = try col.flush();
    try testing.expect(bytes > 0);

    const path_z = try std.fmt.allocPrintSentinel(alloc, "{s}", .{out}, 0);
    defer alloc.free(path_z);
    var loaded = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(loaded);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    // safetensors Load has no GPU eval — the reader runs on the CPU stream.
    const cpu = mlx.mlx_default_cpu_stream_new();
    try mlx.check(mlx.mlx_load_safetensors(&loaded, &meta, path_z, cpu));

    var gu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gu);
    try mlx.check(mlx.mlx_map_string_to_array_get(&gu, loaded, QWEN_PREFIX ++ "1.mlp.experts.gate_up_proj"));
    try testing.expectEqualSlices(c_int, &[_]c_int{6}, mlx.getShape(gu));
    // 2 tokens in the layer: sum(x^2) per expert channel divided by that count.
    try testing.expectEqual(@as(f32, 0.5), try f32At(gu, 0));
    try testing.expectEqual(@as(f32, 9.5), try f32At(gu, 4));

    var dn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dn);
    try mlx.check(mlx.mlx_map_string_to_array_get(&dn, loaded, QWEN_PREFIX ++ "1.mlp.experts.down_proj"));
    try testing.expectEqualSlices(c_int, &[_]c_int{6}, mlx.getShape(dn));
    try testing.expectEqual(@as(f32, 14.5), try f32At(dn, 4));

    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_map_string_to_array_get(&rows, loaded, QWEN_PREFIX ++ "1.mlp.experts.gate_up_proj.rows"));
    try testing.expectEqualSlices(c_int, &[_]c_int{3}, mlx.getShape(rows));
    try testing.expectEqual(@as(f32, 1), try f32At(rows, 0));
    try testing.expectEqual(@as(f32, 3), try f32At(rows, 2));

    // A layer that never routed contributes no entries.
    var absent = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(absent);
    try testing.expect(mlx.mlx_map_string_to_array_get(&absent, loaded, QWEN_PREFIX ++ "0.mlp.experts.gate_up_proj") != 0);
}

test "imatrix capture is off without the environment variable" {
    try testing.expect(envPath() == null);
    const alloc = testing.allocator;
    try testing.expect(try Collector.forModel(alloc, mlx.gpuStream(), "qwen4_exp", 4, 8) == null);
}

test "the mimo_v2 arch keys the same per-layer layout by MiMo's own expert names" {
    const s = mlx.gpuStream();
    const alloc = testing.allocator;
    const E: c_int = 3;
    const hidden: c_int = 2;
    const inter: c_int = 4;

    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const out = try std.fs.path.join(alloc, &.{ dir, "mimo.safetensors" });
    defer alloc.free(out);

    try testing.expectEqual(Arch.mimo_v2, Arch.fromModelType("mimo_v2").?);
    const col = try Collector.init(alloc, s, out, 3, E, .mimo_v2);
    defer col.deinit();

    const x = [_]f32{ 1, 2, 3, 4 };
    const xa = mlx.mlx_array_new_data(&x, &[_]c_int{ 2, hidden }, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);
    const ids = [_]i32{ 0, 2, 2, 2 };
    const ida = mlx.mlx_array_new_data(&ids, &[_]c_int{ 2, 2 }, 2, .int32);
    defer _ = mlx.mlx_array_free(ida);
    try col.observeGateUp(2, xa, ida);
    const act = [_]f32{ 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 };
    const acta = mlx.mlx_array_new_data(&act, &[_]c_int{ 4, inter }, 2, .float32);
    defer _ = mlx.mlx_array_free(acta);
    const aida = mlx.mlx_array_new_data(&ids, &[_]c_int{ 4, 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(aida);
    try col.observeDown(2, acta, aida);
    try testing.expect(try col.flush() > 0);

    const path_z = try std.fmt.allocPrintSentinel(alloc, "{s}", .{out}, 0);
    defer alloc.free(path_z);
    var loaded = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(loaded);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    const cpu = mlx.mlx_default_cpu_stream_new();
    try mlx.check(mlx.mlx_load_safetensors(&loaded, &meta, path_z, cpu));

    var gu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gu);
    try mlx.check(mlx.mlx_map_string_to_array_get(&gu, loaded, "model.layers.2.mlp.experts.gate_up_proj"));
    try testing.expectEqualSlices(c_int, &[_]c_int{E * hidden}, mlx.getShape(gu));
    try testing.expectEqual(@as(f32, 0.5), try f32At(gu, 0));
    var dn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dn);
    try mlx.check(mlx.mlx_map_string_to_array_get(&dn, loaded, "model.layers.2.mlp.experts.down_proj"));
    try testing.expectEqualSlices(c_int, &[_]c_int{E * inter}, mlx.getShape(dn));
    try testing.expectEqual(@as(f32, 14.5), try f32At(dn, 2 * inter));
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_map_string_to_array_get(&rows, loaded, "model.layers.2.mlp.experts.gate_up_proj.rows"));
    try testing.expectEqualSlices(c_int, &[_]c_int{E}, mlx.getShape(rows));
    try testing.expectEqual(@as(f32, 3), try f32At(rows, 2));

    // The Qwen prefix is NOT what a MiMo file is keyed by.
    var absent = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(absent);
    try testing.expect(mlx.mlx_map_string_to_array_get(&absent, loaded, QWEN_PREFIX ++ "2.mlp.experts.gate_up_proj") != 0);
}

test "imatrix records a dense linear's input rows under its source weight name" {
    const s = mlx.gpuStream();
    const alloc = testing.allocator;

    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const out = try std.fs.path.join(alloc, &.{ dir, "trunk.safetensors" });
    defer alloc.free(out);

    const col = try Collector.init(alloc, s, out, 3, 4, .mimo_v2);
    defer col.deinit();

    // o_proj of layer 1 sees [batch 1, 2 rows, 2 channels] twice; lm_head one [1, 3, 2].
    const x = [_]f32{ 1, 2, 3, 4 };
    const xa = mlx.mlx_array_new_data(&x, &[_]c_int{ 1, 2, 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(xa);
    try col.observeLinear(.{ .o_proj = 1 }, xa);
    try col.observeLinear(.{ .o_proj = 1 }, xa);
    const h = [_]f32{ 1, 0, 2, 0, 3, 6 };
    const ha = mlx.mlx_array_new_data(&h, &[_]c_int{ 1, 3, 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(ha);
    try col.observeLinear(.lm_head, ha);
    try testing.expectError(error.ImatrixLayerOutOfRange, col.observeLinear(.{ .o_proj = 3 }, xa));
    try testing.expect(try col.flush() > 0);

    const path_z = try std.fmt.allocPrintSentinel(alloc, "{s}", .{out}, 0);
    defer alloc.free(path_z);
    var loaded = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(loaded);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    const cpu = mlx.mlx_default_cpu_stream_new();
    try mlx.check(mlx.mlx_load_safetensors(&loaded, &meta, path_z, cpu));

    const Want = struct { key: [:0]const u8, vals: []const f32 };
    for ([_]Want{
        // (1+9)*2 / 4 rows, (4+16)*2 / 4 rows
        .{ .key = "model.layers.1.self_attn.o_proj.weight", .vals = &.{ 5, 10 } },
        .{ .key = "model.layers.1.self_attn.o_proj.weight.rows", .vals = &.{4} },
        .{ .key = "lm_head.weight", .vals = &.{ 14.0 / 3.0, 12 } },
        .{ .key = "lm_head.weight.rows", .vals = &.{3} },
    }) |want| {
        var arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(arr);
        try mlx.check(mlx.mlx_map_string_to_array_get(&arr, loaded, want.key));
        try testing.expectEqualSlices(c_int, &[_]c_int{@intCast(want.vals.len)}, mlx.getShape(arr));
        for (want.vals, 0..) |v, i| try testing.expectApproxEqRel(v, try f32At(arr, i), 1e-6);
    }
    // A layer whose o_proj never ran, and the expert blocks nobody routed, write nothing.
    var absent = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(absent);
    try testing.expect(mlx.mlx_map_string_to_array_get(&absent, loaded, "model.layers.0.self_attn.o_proj.weight") != 0);
    try testing.expect(mlx.mlx_map_string_to_array_get(&absent, loaded, "model.layers.1.mlp.experts.gate_up_proj") != 0);
}
