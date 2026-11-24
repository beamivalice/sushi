//! The trunk re-rounded along the quantized path: every trunk linear a layer runs is rounded by GPTQ (`gptq.zig`)
//! in its served affine format, on the Gram of its input as the quantized model computes it. A layer rounds in the
//! order it runs: the attention half (pass A), then the MLP's input side (pass B), then the shared expert's down
//! projection (pass C), each pass running through the linears the passes before it rounded.
const std = @import("std");
const host = @import("mlx_host");
const mlx = host.mlx;
const api = @import("sushi_exl3");
const format = api.format;
const codebook_tables = api.codebook_tables;
const encode = @import("encode.zig");
const gptq = @import("gptq.zig");
const metal_search = @import("metal_search.zig");
const pack = @import("pack.zig");
const pool_mod = @import("pool.zig");
const quantize = @import("quantize.zig");
const source = @import("source.zig");
const stages_mod = @import("stages.zig");
const layer_stage = @import("layer_stage.zig");
const st = @import("safetensors.zig");

pub const Error = error{ TrunkLinearMissing, TrunkSourceShape, TrunkGramEmpty, TrunkTapRows } || std.mem.Allocator.Error;

pub const Group = enum { ple, attn_n, attn_mix, attn_in, attn_out, mlp_n, mlp_mix, moe_in, shared_down, final_n, final_mix, head_in };

pub const Pass = enum { a, b, c, head };

pub fn passOf(g: Group) Pass {
    return switch (g) {
        .ple, .attn_n, .attn_mix, .attn_in, .attn_out => .a,
        .mlp_n, .mlp_mix, .moe_in => .b,
        .shared_down => .c,
        .final_n, .final_mix, .head_in => .head,
    };
}

pub fn groupsOf(p: Pass) std.EnumSet(Group) {
    var set: std.EnumSet(Group) = .empty;
    for (std.enums.values(Group)) |g| if (passOf(g) == p) set.insert(g);
    return set;
}

/// A trunk linear by its source name after `layers.L.` (or the whole source name for the head) and the input it reads.
pub const Linear = struct { suffix: []const u8, group: Group };

const hc_linears = [_]Linear{
    .{ .suffix = "attn_hyper_connection.input_mix_weight_down", .group = .attn_n },
    .{ .suffix = "attn_hyper_connection.input_mix_weight_up", .group = .attn_mix },
    .{ .suffix = "mlp_hyper_connection.input_mix_weight_down", .group = .mlp_n },
    .{ .suffix = "mlp_hyper_connection.input_mix_weight_up", .group = .mlp_mix },
};
const gdn_linears = [_]Linear{
    .{ .suffix = "linear_attn.in_proj_qkv", .group = .attn_in },
    .{ .suffix = "linear_attn.in_proj_z", .group = .attn_in },
    .{ .suffix = "linear_attn.in_proj_a", .group = .attn_in },
    .{ .suffix = "linear_attn.in_proj_b", .group = .attn_in },
    .{ .suffix = "linear_attn.out_proj", .group = .attn_out },
};
const qsa_linears = [_]Linear{
    .{ .suffix = "self_attn.q_proj", .group = .attn_in },
    .{ .suffix = "self_attn.k_proj", .group = .attn_in },
    .{ .suffix = "self_attn.v_proj", .group = .attn_in },
    .{ .suffix = "self_attn.indexer.index_qk_proj", .group = .attn_in },
    .{ .suffix = "self_attn.o_proj", .group = .attn_out },
};
const shared_linears = [_]Linear{
    .{ .suffix = "mlp.shared_expert.gate_proj", .group = .moe_in },
    .{ .suffix = "mlp.shared_expert.up_proj", .group = .moe_in },
    .{ .suffix = "mlp.shared_expert.down_proj", .group = .shared_down },
};
const ple_linears = [_]Linear{
    .{ .suffix = "ple.key_proj", .group = .ple },
    .{ .suffix = "ple.value_proj", .group = .ple },
};
pub const head_linears = [_]Linear{
    .{ .suffix = "model.language_model.hyper_connection_mixer.input_mix_weight_down", .group = .final_n },
    .{ .suffix = "model.language_model.hyper_connection_mixer.input_mix_weight_up", .group = .final_mix },
    .{ .suffix = "lm_head", .group = .head_in },
};

pub const max_layer_linears = hc_linears.len + qsa_linears.len + shared_linears.len + ple_linears.len;

pub fn layerLinears(buf: *[max_layer_linears]Linear, full: bool, ple: bool) []const Linear {
    var n: usize = 0;
    const parts = [_][]const Linear{ &hc_linears, if (full) &qsa_linears else &gdn_linears, &shared_linears, if (ple) &ple_linears else &.{} };
    for (parts) |part| for (part) |l| {
        buf[n] = l;
        n += 1;
    };
    return buf[0..n];
}

/// The re-rounded linears' shards; `layer_stage.isRerounded` names them.
pub fn shardName(buf: []u8, layer: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "trunk-L{d:0>2}.safetensors", .{layer});
}
pub const head_shard = "trunk-head.safetensors";

/// The STQ trunk's banks, apart from the affine linears a layer keeps.
pub fn stqShardName(buf: []u8, layer: u32) ![]const u8 {
    return std.fmt.bufPrint(buf, "trunk-stq-L{d:0>2}.safetensors", .{layer});
}
pub const stq_head_shard = "trunk-stq-head.safetensors";

/// How a trunk linear is rounded: GPTQ in its affine format, onto the trellis (STQ), or left to the pack as stored.
pub const Round = enum { gptq, stq, keep };

/// The STQ trunk recipe: SPEC1 w15 banks at `rate`, the head at `head_rate`. A linear off the 128 grid on either
/// side, the hyper-connection mixes and the PLE pair (which the engine reads affine) stay in the pack's affine format.
pub const Stq = struct {
    rate: format.Rate = .{ .n = 64 },
    head_rate: format.Rate = .{ .n = 96 },
    pin_pass: u32 = encode.default_pin_pass,
    decode: format.Decode = .{ .codebook = .spec1, .window = .w15, .table = codebook_tables.indexOf("spec1-w15") },

    pub fn dec(self: Stq) format.Decode {
        return self.decode;
    }
};

/// The seed the expert and DeepSeek-V4.1 trunk converters draw signs from (`quantize.signs`).
pub const sign_seed: u64 = 0x5eed_e3c0_de00_0001;

pub fn stqGroup(g: Group) bool {
    return switch (g) {
        .attn_in, .attn_out, .moe_in, .shared_down, .head_in => true,
        .ple, .attn_n, .attn_mix, .mlp_n, .mlp_mix, .final_n, .final_mix => false,
    };
}

pub fn onGrid(out: usize, in: usize) bool {
    return out % 128 == 0 and in % 128 == 0;
}

pub const batch_tokens: usize = 8192;
pub const gram_stride: usize = 4;

/// Consecutive windows of one length forwarded together.
pub const Batch = struct { first: usize, count: usize, len: usize };

/// Windows `starts` packed in order into batches of one length and at most `max_tokens` tokens (a longer window is a
/// batch of its own; 0 makes every window its own batch).
pub fn batchPlan(allocator: std.mem.Allocator, starts: []const usize, max_tokens: usize) ![]Batch {
    var out = std.ArrayList(Batch).empty;
    errdefer out.deinit(allocator);
    const count = starts.len - 1;
    var i: usize = 0;
    while (i < count) {
        const len = starts[i + 1] - starts[i];
        var j = i;
        while (j < count and starts[j + 1] - starts[j] == len and (j - i + 1) * len <= @max(max_tokens, len)) j += 1;
        try out.append(allocator, .{ .first = i, .count = j - i, .len = len });
        i = j;
    }
    return out.toOwnedSlice(allocator);
}

/// The trunk Grams read every `gram_stride`-th batch.
pub fn strided(batch_index: usize) bool {
    return batch_index % gram_stride == 0;
}

/// Each watched group's X'X (float32) over the rows a forward feeds its linears, through `linear_tap`.
pub const Grams = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    targets: std.ArrayList(Target) = .empty,
    want: std.EnumSet(Group) = .empty,
    acc: std.EnumArray(Group, mlx.mlx_array) = .initFill(.{ .ctx = null }),
    rows: std.EnumArray(Group, u64) = .initFill(0),
    /// Groups recorded since the last `settle`: the linears of a group read one input, which the forward may hand
    /// each of them as a different array.
    seen: std.EnumSet(Group) = .empty,
    failed: ?anyerror = null,

    const Target = struct { w: ?*anyopaque, group: Group };

    pub fn init(allocator: std.mem.Allocator, s: mlx.mlx_stream) Grams {
        return .{ .allocator = allocator, .s = s };
    }

    pub fn deinit(self: *Grams) void {
        self.targets.deinit(self.allocator);
        self.reset();
    }

    /// Drops every sum.
    pub fn reset(self: *Grams) void {
        for (std.enums.values(Group)) |g| {
            const a = self.acc.getPtr(g);
            if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
            a.* = .{ .ctx = null };
            self.rows.set(g, 0);
        }
    }

    pub fn watch(self: *Grams, w: mlx.mlx_array, g: Group) !void {
        if (w.ctx == null) return error.TrunkLinearMissing;
        try self.targets.append(self.allocator, .{ .w = w.ctx, .group = g });
    }

    pub fn forget(self: *Grams) void {
        self.targets.clearRetainingCapacity();
    }

    pub fn arm(self: *Grams) void {
        host.trunk.linear_tap = .{ .ctx = self, .record = record };
    }

    pub fn disarm(_: *Grams) void {
        host.trunk.linear_tap = null;
    }

    fn record(ctx: *anyopaque, w: mlx.mlx_array, x: mlx.mlx_array) void {
        const self: *Grams = @ptrCast(@alignCast(ctx));
        const g = for (self.targets.items) |t| {
            if (t.w == w.ctx) break t.group;
        } else return;
        if (!self.want.contains(g) or self.seen.contains(g)) return;
        self.seen.insert(g);
        self.add(g, x) catch |e| {
            if (self.failed == null) self.failed = e;
        };
    }

    /// Adds X'X of `x`'s rows ([..., in]) to group `g`'s sum, lazily.
    pub fn add(self: *Grams, g: Group, x: mlx.mlx_array) !void {
        const shape = mlx.getShape(x);
        const in = shape[shape.len - 1];
        const rows: c_int = @intCast(@divExact(mlx.mlx_array_size(x), @as(usize, @intCast(in))));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ rows, in }, 2, self.s));
        var f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&f, flat, .float32, self.s));
        var ft = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ft);
        try mlx.check(mlx.mlx_transpose(&ft, f, self.s));
        var gm = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(gm);
        try mlx.check(mlx.mlx_matmul(&gm, ft, f, self.s));
        const a = self.acc.getPtr(g);
        if (a.ctx == null) {
            a.* = gm;
        } else {
            var sum = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(sum);
            try mlx.check(mlx.mlx_add(&sum, a.*, gm, self.s));
            _ = mlx.mlx_array_free(gm);
            _ = mlx.mlx_array_free(a.*);
            a.* = sum;
        }
        self.rows.set(g, self.rows.get(g) + @as(u64, @intCast(rows)));
    }

    /// Evaluates the sums so far; the next forward's inputs count afresh.
    pub fn settle(self: *Grams) !void {
        self.seen = .empty;
        if (self.failed) |e| {
            self.failed = null;
            return e;
        }
        var live: [std.enums.values(Group).len]mlx.mlx_array = undefined;
        var n: usize = 0;
        for (std.enums.values(Group)) |g| {
            const a = self.acc.get(g);
            if (a.ctx != null) {
                live[n] = a;
                n += 1;
            }
        }
        if (n == 0) return;
        const vec = mlx.mlx_vector_array_new_data(&live, n);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
    }

    /// Every wanted group that has a watched linear saw exactly `rows` rows.
    pub fn expectRows(self: *const Grams, rows: u64) !void {
        for (self.targets.items) |t| if (self.want.contains(t.group) and self.rows.get(t.group) != rows) {
            host.log.info("[convert] trunk tap {s}: {d} rows where the forward ran {d}\n", .{ @tagName(t.group), self.rows.get(t.group), rows });
            return error.TrunkTapRows;
        };
    }
};

/// The (bits, group) a served tensor is stored at: its override, else the trunk default.
pub const Spec = struct {
    bits: u32,
    group: u32,
    overrides: []const pack.Override,

    pub fn grid(self: Spec, base: []const u8) gptq.Grid {
        for (self.overrides) |o| if (std.mem.eql(u8, o.base, base)) return .{ .bits = @intCast(o.bits), .group = o.group };
        return .{ .bits = @intCast(self.bits), .group = self.group };
    }
};

pub const Rounded = struct { words: []u32, scales: []u16, biases: []u16 };

/// A tensor rounded onto the trellis: `[in/16, out/16, n]` codes and the two sign-scale vectors.
pub const Bank = struct {
    trellis: []u16,
    suh: []u16,
    svh: []u16,
    rate: format.Rate,

    pub fn deinit(self: *Bank, allocator: std.mem.Allocator) void {
        allocator.free(self.trellis);
        allocator.free(self.suh);
        allocator.free(self.svh);
    }
};

pub const Tensor = struct {
    /// Served name without `.weight`.
    base: []u8,
    group: Group,
    out: usize,
    in: usize,
    grid: gptq.Grid = .{ .bits = 8, .group = 128 },
    /// The source weight, bf16 [out, in].
    source: []u16,
    round: Round = .gptq,
    rate: format.Rate = .{ .n = 0 },
    rounded: ?Rounded = null,
    bank: ?Bank = null,
    /// A `bank`'s weight as the trellis decodes it, bf16 [out, in]: what a staged layer runs.
    dense: ?[]u16 = null,
    rel_err: f64 = 0,
};

pub const Mode = union(enum) { gptq: Spec, stq: format.Rate, keep };

/// A bank's weight as the serving kernels apply it (`linear.decodeDense`'s twin), bf16 [out, in].
pub fn denseOfBank(allocator: std.mem.Allocator, bank: Bank, dec: format.Decode, out: usize, in: usize) ![]u16 {
    const public = try allocator.alloc(f32, in * out);
    defer allocator.free(public);
    try api.linear.decodeDense(allocator, bank.trellis, bank.suh, bank.svh, in, out, bank.rate.n, dec, public);
    const dense = try allocator.alloc(u16, out * in);
    errdefer allocator.free(dense);
    for (0..in) |i| {
        for (0..out) |o| dense[o * in + i] = pack.bf16FromF32(public[i * out + o]);
    }
    return dense;
}

/// One layer's (or the head's) trunk linears: source weights until a pass rounds them.
pub const Trunk = struct {
    allocator: std.mem.Allocator,
    tensors: std.ArrayList(Tensor) = .empty,
    stq: Stq = .{},

    pub fn deinit(self: *Trunk) void {
        for (self.tensors.items) |*t| {
            self.allocator.free(t.base);
            self.allocator.free(t.source);
            if (t.rounded) |r| {
                self.allocator.free(r.words);
                self.allocator.free(r.scales);
                self.allocator.free(r.biases);
            }
            if (t.bank) |*b| b.deinit(self.allocator);
            if (t.dense) |d| self.allocator.free(d);
        }
        self.tensors.deinit(self.allocator);
    }

    pub fn forLayer(allocator: std.mem.Allocator, src: *const source.SafetensorsSource, layer: u32, full: bool, ple: bool, spec: Spec) !Trunk {
        var buf: [max_layer_linears]Linear = undefined;
        var self = Trunk{ .allocator = allocator };
        errdefer self.deinit();
        for (layerLinears(&buf, full, ple)) |l| {
            var key_buf: [192]u8 = undefined;
            const key = try std.fmt.bufPrint(&key_buf, "model.language_model.layers.{d}.{s}", .{ layer, l.suffix });
            try self.add(src, key, l.group, .{ .gptq = spec });
        }
        return self;
    }

    pub fn forHead(allocator: std.mem.Allocator, src: *const source.SafetensorsSource, spec: Spec) !Trunk {
        var self = Trunk{ .allocator = allocator };
        errdefer self.deinit();
        for (head_linears) |l| try self.add(src, l.suffix, l.group, .{ .gptq = spec });
        return self;
    }

    /// The layer's linears that go onto the trellis: every one of a group `stqGroup` names that sits on the 128
    /// grid. The rest are not listed; the pack keeps them as stored.
    pub fn forLayerStq(allocator: std.mem.Allocator, src: *const source.SafetensorsSource, layer: u32, full: bool, ple: bool, stq: Stq) !Trunk {
        var buf: [max_layer_linears]Linear = undefined;
        var self = Trunk{ .allocator = allocator, .stq = stq };
        errdefer self.deinit();
        for (layerLinears(&buf, full, ple)) |l| {
            if (!stqGroup(l.group)) continue;
            var key_buf: [192]u8 = undefined;
            const key = try std.fmt.bufPrint(&key_buf, "model.language_model.layers.{d}.{s}", .{ layer, l.suffix });
            try self.add(src, key, l.group, .{ .stq = stq.rate });
        }
        return self;
    }

    /// The final mixer's linears, listed so the head's Gram reads through the source mixer but never rounded, and
    /// `lm_head` onto the trellis at the head rate.
    pub fn forHeadStq(allocator: std.mem.Allocator, src: *const source.SafetensorsSource, stq: Stq) !Trunk {
        var self = Trunk{ .allocator = allocator, .stq = stq };
        errdefer self.deinit();
        for (head_linears) |l| try self.add(src, l.suffix, l.group, if (stqGroup(l.group)) .{ .stq = stq.head_rate } else .keep);
        return self;
    }

    fn add(self: *Trunk, src: *const source.SafetensorsSource, key: []const u8, g: Group, mode: Mode) !void {
        var wkey_buf: [200]u8 = undefined;
        const wkey = try std.fmt.bufPrint(&wkey_buf, "{s}.weight", .{key});
        const loc = src.tensors.get(wkey) orelse return error.TrunkLinearMissing;
        if (!std.mem.eql(u8, loc.dtype, "BF16") or loc.shape.len != 2) return error.TrunkSourceShape;
        const out = loc.shape[0];
        const in = loc.shape[1];
        if (mode == .stq and !onGrid(out, in)) return;
        const w = try self.allocator.alloc(u16, out * in);
        errdefer self.allocator.free(w);
        try src.readRange(loc, 0, std.mem.sliceAsBytes(w));
        var name_buf: [200]u8 = undefined;
        const served = try pack.qwenServedName(&name_buf, wkey);
        const base = try self.allocator.dupe(u8, served[0 .. served.len - ".weight".len]);
        errdefer self.allocator.free(base);
        var t = Tensor{ .base = base, .group = g, .out = out, .in = in, .source = w };
        switch (mode) {
            .gptq => |spec| {
                t.grid = spec.grid(base);
                if (in % t.grid.group != 0) return error.TrunkSourceShape;
            },
            .stq => |rate| {
                t.round = .stq;
                t.rate = rate;
            },
            .keep => t.round = .keep,
        }
        try self.tensors.append(self.allocator, t);
    }

    /// The groups some tensor of this trunk is rounded in.
    pub fn activeGroups(self: *const Trunk) std.EnumSet(Group) {
        var set: std.EnumSet(Group) = .empty;
        for (self.tensors.items) |t| if (t.round != .keep) set.insert(t.group);
        return set;
    }

    /// Rounds the linears of `pass` on `grams`' sums (each group's factor shared by its linears) and measures each
    /// one's tr(E H E') / tr(W H W').
    pub fn round(self: *Trunk, pool: ?*pool_mod.Pool, grams: *Grams, pass: Pass) !void {
        var r = try Rounding.begin(self, pool, grams, pass);
        defer r.deinit();
        const t0 = stages_mod.nowNs();
        r.run();
        const t1 = stages_mod.nowNs();
        try r.finish(grams);
        if (self.stq.rate.n != 0 and self.tensors.items.len != 0 and self.tensors.items[0].round == .stq) {
            var buf: [512]u8 = undefined;
            var w = std.Io.Writer.fixed(&buf);
            r.stages.write(&w) catch {};
            host.log.info("[convert] stq rounding {s}: host prep {d:.1}s, search and decode {d:.1}s ({s})\n", .{ @tagName(pass), @as(f64, @floatFromInt(t1 - t0)) / 1e9, @as(f64, @floatFromInt(stages_mod.nowNs() - t1)) / 1e9, w.buffered() });
        }
    }

    /// The trunk linears as a stage holds them: rounded ones affine, the rest as their source weights.
    pub fn staged(self: *const Trunk, allocator: std.mem.Allocator) ![]layer_stage.Linear {
        var kept: usize = 0;
        for (self.tensors.items) |t| kept += @intFromBool(t.round == .keep);
        const out = try allocator.alloc(layer_stage.Linear, self.tensors.items.len - kept);
        var at: usize = 0;
        for (self.tensors.items) |t| {
            if (t.round == .keep) continue;
            const o = &out[at];
            at += 1;
            if (t.dense) |d| {
                o.* = .{ .base = t.base, .weight = .{ .dtype = "BF16", .shape = .{ t.out, t.in }, .bytes = std.mem.sliceAsBytes(d) } };
            } else if (t.rounded) |r| {
                const groups = t.in / t.grid.group;
                o.* = .{
                    .base = t.base,
                    .weight = .{ .dtype = "U32", .shape = .{ t.out, r.words.len / t.out }, .bytes = std.mem.sliceAsBytes(r.words) },
                    .scales = .{ .dtype = "BF16", .shape = .{ t.out, groups }, .bytes = std.mem.sliceAsBytes(r.scales) },
                    .biases = .{ .dtype = "BF16", .shape = .{ t.out, groups }, .bytes = std.mem.sliceAsBytes(r.biases) },
                };
            } else {
                o.* = .{ .base = t.base, .weight = .{ .dtype = "BF16", .shape = .{ t.out, t.in }, .bytes = std.mem.sliceAsBytes(t.source) } };
            }
        }
        return out;
    }

    /// The rounded linears as one shard.
    pub fn write(self: *const Trunk, io: std.Io, dir: std.Io.Dir, name: []const u8) !void {
        var b = pack.ShardBuilder.init(self.allocator);
        defer b.deinit();
        const parts = try self.staged(self.allocator);
        defer self.allocator.free(parts);
        for (parts) |p| {
            if (p.scales == null) continue;
            var nb: [240]u8 = undefined;
            try b.add(try std.fmt.bufPrint(&nb, "{s}.weight", .{p.base}), p.weight.dtype, &p.weight.shape, p.weight.bytes);
            try b.add(try std.fmt.bufPrint(&nb, "{s}.scales", .{p.base}), p.scales.?.dtype, &p.scales.?.shape, p.scales.?.bytes);
            try b.add(try std.fmt.bufPrint(&nb, "{s}.biases", .{p.base}), p.biases.?.dtype, &p.biases.?.shape, p.biases.?.bytes);
        }
        try b.flush(io, dir, name, &.{.{ "format", "mlx" }});
    }

    /// The tensors rounded onto the trellis as one STQ shard: `<base>.{trellis,suh,svh}` under the stamp the engine
    /// reads, `k` the widest bank in the shard. Returns how many banks it holds.
    pub fn writeBanks(self: *const Trunk, io: std.Io, dir: std.Io.Dir, name: []const u8, provenance: []const [2][]const u8) !usize {
        var b = pack.ShardBuilder.init(self.allocator);
        defer b.deinit();
        var widest = format.Rate{ .n = 0 };
        for (self.tensors.items) |t| {
            const bank = t.bank orelse continue;
            var nb: [240]u8 = undefined;
            try b.add(try std.fmt.bufPrint(&nb, "{s}.trellis", .{t.base}), "U16", &.{ t.in / 16, t.out / 16, bank.rate.n }, std.mem.sliceAsBytes(bank.trellis));
            try b.add(try std.fmt.bufPrint(&nb, "{s}.suh", .{t.base}), "F16", &.{t.in}, std.mem.sliceAsBytes(bank.suh));
            try b.add(try std.fmt.bufPrint(&nb, "{s}.svh", .{t.base}), "F16", &.{t.out}, std.mem.sliceAsBytes(bank.svh));
            widest.n = @max(widest.n, bank.rate.n);
        }
        const banks = b.count / 3;
        if (banks == 0) return 0;
        const dec = self.stq.dec();
        var k_buf: [16]u8 = undefined;
        var w_buf: [4]u8 = undefined;
        var meta = std.ArrayList([2][]const u8).empty;
        defer meta.deinit(self.allocator);
        try meta.appendSlice(self.allocator, &.{
            .{ "format", "stq" },
            .{ "k", widest.kText(&k_buf) },
            .{ "codebook", @tagName(dec.codebook) },
            .{ "window", try std.fmt.bufPrint(&w_buf, "{d}", .{dec.window.bits()}) },
            .{ "codebook_table", codebook_tables.tables[dec.table.?].name },
        });
        try meta.appendSlice(self.allocator, provenance);
        try b.flush(io, dir, name, meta.items);
        return banks;
    }
};

/// tr(E H E') / tr(W H W') with E = deq - w, both [out, in], and H the group's X'X.
/// `Trunk.round` in three parts, so the host work can run on a thread of its own beside the GPU: `begin` reads the
/// pass's Grams and `finish` measures and keeps the results (both MLX, on the caller's thread); `run` factors and
/// rounds on the host only.
pub const Rounding = struct {
    trunk: *Trunk,
    pool: ?*pool_mod.Pool,
    pass: Pass,
    /// Each group's X'X / rows; empty for a group outside the pass.
    h: std.EnumArray(Group, []f32) = .initFill(&.{}),
    n: std.EnumArray(Group, usize) = .initFill(0),
    /// Per tensor of the pass: its source weight in float32 and its rounding.
    w: [][]f32,
    out: []?gptq.Rounded,
    /// Per tensor rounded onto the trellis: its host half, until `finish` runs the search.
    prep: []?metal_search.Prepared,
    stages: stages_mod.Stages = .{},
    err: ?anyerror = null,
    thread: ?std.Thread = null,

    pub fn begin(trunk: *Trunk, pool: ?*pool_mod.Pool, grams: *Grams, pass: Pass) !Rounding {
        const a = trunk.allocator;
        const n = trunk.tensors.items.len;
        var self = Rounding{ .trunk = trunk, .pool = pool, .pass = pass, .w = try a.alloc([]f32, n), .out = undefined, .prep = undefined };
        @memset(self.w, &.{});
        self.out = a.alloc(?gptq.Rounded, n) catch |e| {
            a.free(self.w);
            return e;
        };
        @memset(self.out, null);
        self.prep = a.alloc(?metal_search.Prepared, n) catch |e| {
            a.free(self.w);
            a.free(self.out);
            return e;
        };
        @memset(self.prep, null);
        errdefer self.deinit();
        for (std.enums.values(Group)) |g| {
            if (passOf(g) != pass) continue;
            const fi = for (trunk.tensors.items, 0..) |t, i| {
                if (t.group == g and t.round != .keep) break i;
            } else continue;
            const in = trunk.tensors.items[fi].in;
            const rows = grams.rows.get(g);
            const acc = grams.acc.get(g);
            if (rows == 0 or acc.ctx == null) return error.TrunkGramEmpty;
            try mlx.check(mlx.mlx_array_eval(acc));
            const data = mlx.mlx_array_data_float32(acc) orelse return error.TrunkGramEmpty;
            const h = try a.alloc(f32, in * in);
            for (h, data[0 .. in * in]) |*o, v| o.* = @floatCast(@as(f64, v) / @as(f64, @floatFromInt(rows)));
            self.h.set(g, h);
            self.n.set(g, in);
        }
        return self;
    }

    pub fn deinit(self: *Rounding) void {
        const a = self.trunk.allocator;
        if (self.thread) |th| th.join();
        self.thread = null;
        for (std.enums.values(Group)) |g| a.free(self.h.get(g));
        for (self.w) |w| a.free(w);
        for (self.out) |*o| if (o.*) |*r| r.deinit(a);
        for (self.prep) |*p| if (p.*) |*v| v.deinit();
        a.free(self.w);
        a.free(self.out);
        a.free(self.prep);
    }

    pub fn run(self: *Rounding) void {
        self.runOrFail() catch |e| {
            self.err = e;
        };
    }

    fn runOrFail(self: *Rounding) !void {
        const a = self.trunk.allocator;
        for (std.enums.values(Group)) |g| {
            const h = self.h.get(g);
            if (h.len == 0) continue;
            const in = self.n.get(g);
            var factor: ?gptq.Factor = null;
            defer if (factor) |*f| f.deinit(a);
            for (self.trunk.tensors.items, 0..) |t, i| {
                if (t.group != g) continue;
                switch (t.round) {
                    .keep => continue,
                    .stq => try self.prepareStq(i, h),
                    .gptq => {
                        if (factor == null) factor = try gptq.Factor.init(a, self.pool, h, in);
                        self.w[i] = try a.alloc(f32, t.out * t.in);
                        try source.decodeBf16ToF32(t.source, self.w[i]);
                        self.out[i] = try gptq.quantizeRows(a, self.pool, self.w[i], t.out, t.in, &factor.?, t.grid);
                    },
                }
            }
        }
    }

    /// The host half of rounding tensor `i` onto the trellis against Gram `h`: regularize, rotate and factor. It
    /// makes no MLX call, so it can run on the rounding thread.
    fn prepareStq(self: *Rounding, i: usize, h: []const f32) !void {
        const a = self.trunk.allocator;
        const t = self.trunk.tensors.items[i];
        self.w[i] = try a.alloc(f32, t.out * t.in);
        try source.decodeBf16ToF32(t.source, self.w[i]);
        const public = try a.alloc(f32, t.in * t.out);
        defer a.free(public);
        try source.publicFromBf16(t.source, t.out, t.in, public);
        const signs = try quantize.signs(a, sign_seed, t.base, t.in + t.out);
        defer a.free(signs);
        const job = [_]metal_search.TensorJob{.{ .w = public, .h = h, .sign_in = signs[0..t.in], .sign_out = signs[t.in..] }};
        self.prep[i] = try metal_search.prepareBatch(a, self.pool, &job, t.in, t.out, .auto, .f64);
    }

    /// The search of tensor `i`, then its decoded weight and the error the Gram weighs. Runs on the MLX thread.
    fn finishStq(self: *Rounding, i: usize, grams: *Grams) !void {
        const a = self.trunk.allocator;
        const t = &self.trunk.tensors.items[i];
        const dec = self.trunk.stq.dec();
        var result: [1]quantize.Quantized = undefined;
        try metal_search.finishBatch(a, grams.s, self.pool, &self.prep[i].?, t.rate, dec, self.trunk.stq.pin_pass, &self.stages, &result, null);
        self.prep[i].?.deinit();
        self.prep[i] = null;
        t.bank = .{ .trellis = result[0].packed_out, .suh = result[0].suh, .svh = result[0].svh, .rate = t.rate };
        const dense = try denseOfBank(a, t.bank.?, dec, t.out, t.in);
        t.dense = dense;
        const deq = try a.alloc(f32, dense.len);
        defer a.free(deq);
        for (deq, dense) |*o, d| o.* = @bitCast(@as(u32, d) << 16);
        t.rel_err = try relErr(grams.s, deq, self.w[i], t.out, t.in, grams.acc.get(t.group));
    }

    pub fn start(self: *Rounding) !void {
        self.thread = try std.Thread.spawn(.{}, run, .{self});
    }

    pub fn finish(self: *Rounding, grams: *Grams) !void {
        if (self.thread) |th| th.join();
        self.thread = null;
        if (self.err) |e| return e;
        const a = self.trunk.allocator;
        for (self.trunk.tensors.items, 0..) |*t, i| {
            if (self.prep[i] != null) {
                try self.finishStq(i, grams);
                continue;
            }
            const r = &(self.out[i] orelse continue);
            t.rel_err = try relErr(grams.s, r.deq, self.w[i], t.out, t.in, grams.acc.get(t.group));
            const words = try r.packed32(a, t.grid.bits);
            errdefer a.free(words);
            const scales = try a.dupe(u16, r.scales);
            errdefer a.free(scales);
            t.rounded = .{ .words = words, .scales = scales, .biases = try a.dupe(u16, r.biases) };
        }
    }
};

fn relErr(s: mlx.mlx_stream, deq: []const f32, w: []const f32, out: usize, in: usize, h: mlx.mlx_array) !f64 {
    const shape = [_]c_int{ @intCast(out), @intCast(in) };
    const wa = mlx.mlx_array_new_data(w.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(wa);
    const da = mlx.mlx_array_new_data(deq.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(da);
    var e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_subtract(&e, da, wa, s));
    const num = try quadratic(s, e, h);
    const den = try quadratic(s, wa, h);
    return num / den;
}

fn quadratic(s: mlx.mlx_stream, a: mlx.mlx_array, h: mlx.mlx_array) !f64 {
    var ah = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ah);
    try mlx.check(mlx.mlx_matmul(&ah, a, h, s));
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_multiply(&p, ah, a, s));
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_sum(&sum, p, false, s));
    try mlx.check(mlx.mlx_array_eval(sum));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, sum));
    return v;
}

test "trunk: batches pack consecutive equal-length windows to 8192 tokens, and the Grams read every fourth" {
    const t = std.testing;
    // 2048 x 6 (batches of 4 and 2), 512 x 20 (16 + 4), 9000 x 2 (one each).
    var starts = std.ArrayList(usize).empty;
    defer starts.deinit(t.allocator);
    var at: usize = 0;
    try starts.append(t.allocator, 0);
    for ([_][2]usize{ .{ 6, 2048 }, .{ 20, 512 }, .{ 2, 9000 } }) |run| for (0..run[0]) |_| {
        at += run[1];
        try starts.append(t.allocator, at);
    };
    const plan = try batchPlan(t.allocator, starts.items, batch_tokens);
    defer t.allocator.free(plan);
    const want = [_]Batch{ .{ .first = 0, .count = 4, .len = 2048 }, .{ .first = 4, .count = 2, .len = 2048 }, .{ .first = 6, .count = 16, .len = 512 }, .{ .first = 22, .count = 4, .len = 512 }, .{ .first = 26, .count = 1, .len = 9000 }, .{ .first = 27, .count = 1, .len = 9000 } };
    try t.expectEqualSlices(Batch, &want, plan);
    try t.expect(strided(0) and !strided(3) and strided(4));
    const single = try batchPlan(t.allocator, starts.items, 0);
    defer t.allocator.free(single);
    try t.expectEqual(@as(usize, 28), single.len);
}

test "trunk: a layer's linears follow its attention kind, and the PLE pair only on the PLE layer" {
    const t = std.testing;
    var buf: [max_layer_linears]Linear = undefined;
    const gdn = layerLinears(&buf, false, false);
    try t.expectEqual(@as(usize, 12), gdn.len);
    var buf2: [max_layer_linears]Linear = undefined;
    const qsa = layerLinears(&buf2, true, true);
    try t.expectEqual(@as(usize, 14), qsa.len);
    var by_pass = std.EnumArray(Pass, usize).initFill(0);
    for (qsa) |l| by_pass.set(passOf(l.group), by_pass.get(passOf(l.group)) + 1);
    try t.expectEqual(@as(usize, 9), by_pass.get(.a));
    try t.expectEqual(@as(usize, 4), by_pass.get(.b));
    try t.expectEqual(@as(usize, 1), by_pass.get(.c));
}

/// A trunk of one STQ tensor `base` ([out, in], random bf16) in `group`, and Grams of `rows` random inputs for it.
const StqCase = struct {
    trunk: Trunk,
    grams: Grams,
    rows: usize,

    fn init(allocator: std.mem.Allocator, s: mlx.mlx_stream, base: []const u8, group: Group, out: usize, in: usize, rows: usize, seed: u64) !StqCase {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        var self = StqCase{ .trunk = .{ .allocator = allocator }, .grams = Grams.init(allocator, s), .rows = rows };
        errdefer self.deinit();
        const w = try allocator.alloc(u16, out * in);
        errdefer allocator.free(w);
        for (w) |*v| v.* = pack.bf16FromF32((rnd.float(f32) - 0.5) * 0.1);
        const name = try allocator.dupe(u8, base);
        errdefer allocator.free(name);
        try self.trunk.tensors.append(allocator, .{ .base = name, .group = group, .out = out, .in = in, .source = w, .round = .stq, .rate = self.trunk.stq.rate });
        // Inputs with a few loud channels, so the Gram is far from the identity.
        const x = try allocator.alloc(f32, rows * in);
        defer allocator.free(x);
        for (x, 0..) |*v, i| v.* = (rnd.float(f32) - 0.5) * (if (i % in < 8) @as(f32, 4) else 1);
        const arr = mlx.mlx_array_new_data(x.ptr, &[_]c_int{ @intCast(rows), @intCast(in) }, 2, .float32);
        defer _ = mlx.mlx_array_free(arr);
        try self.grams.add(group, arr);
        try self.grams.settle();
        return self;
    }

    fn deinit(self: *StqCase) void {
        self.grams.deinit();
        self.trunk.deinit();
    }

    /// The Gram `Rounding.begin` reads: the sum over the rows, divided by them.
    fn gram(self: *StqCase, allocator: std.mem.Allocator, group: Group, in: usize) ![]f32 {
        const acc = self.grams.acc.get(group);
        try mlx.check(mlx.mlx_array_eval(acc));
        const data = mlx.mlx_array_data_float32(acc).?;
        const h = try allocator.alloc(f32, in * in);
        for (h, data[0 .. in * in]) |*o, v| o.* = @floatCast(@as(f64, v) / @as(f64, @floatFromInt(self.rows)));
        return h;
    }
};

test "trunk stq: a tensor rounds to the bytes the DeepSeek-V4.1 trunk's search gives on the same Gram, and its dense twin is what the kernels apply" {
    const t = std.testing;
    const dsv41 = @import("dsv41.zig");
    const s = mlx.gpuStream();
    const out: usize = 256;
    const in: usize = 384;
    var case = try StqCase.init(t.allocator, s, "language_model.model.layers.0.self_attn.k_proj", .attn_in, out, in, 2048, 7);
    defer case.deinit();
    const pool = try pool_mod.Pool.init(t.allocator, 4);
    defer pool.deinit();
    try case.trunk.round(pool, &case.grams, .a);

    const tensor = case.trunk.tensors.items[0];
    const bank = tensor.bank.?;
    try t.expectEqual(@as(usize, in / 16 * (out / 16) * 64), bank.trellis.len);
    try t.expectEqual(in, bank.suh.len);
    try t.expectEqual(out, bank.svh.len);
    try t.expect(tensor.rounded == null);

    const h = try case.gram(t.allocator, .attn_in, in);
    defer t.allocator.free(h);
    const public = try t.allocator.alloc(f32, in * out);
    defer t.allocator.free(public);
    try source.publicFromBf16(tensor.source, out, in, public);
    var stages = stages_mod.Stages{};
    var want = try dsv41.quantizeDense(t.allocator, s, pool, public, h, in, out, .{ .n = 64 }, case.trunk.stq.pin_pass, tensor.base, &stages);
    defer want.deinit(t.allocator);
    try t.expectEqualSlices(u16, want.trellis, bank.trellis);
    try t.expectEqualSlices(u16, want.suh, bank.suh);
    try t.expectEqualSlices(u16, want.svh, bank.svh);

    // The dense twin: bf16 of the f32 decode, [out, in], and within f16 rounding of the engine's own reconstruction.
    const dense = tensor.dense.?;
    try t.expectEqual(out * in, dense.len);
    const recon = try t.allocator.alloc(u16, in * out);
    defer t.allocator.free(recon);
    try format.reconstructPublic(t.allocator, bank.trellis, bank.suh, bank.svh, in, out, bank.rate, case.trunk.stq.dec(), recon);
    var diff: f64 = 0;
    var norm: f64 = 0;
    for (0..out) |o| for (0..in) |i| {
        const d: f64 = @as(f32, @bitCast(@as(u32, dense[o * in + i]) << 16));
        const r: f64 = format.f16BitsToF32(recon[i * out + o]);
        diff += (d - r) * (d - r);
        norm += r * r;
    };
    try t.expect(@sqrt(diff / norm) < 5e-3);
    // K4 on a Gram-weighted rounding: the error the Gram weighs is a small share of the weight's.
    try t.expect(tensor.rel_err > 0 and tensor.rel_err < 0.05);
}

test "trunk stq: at the K6 trunk rate a tensor rounds to the K6 bytes of the same search, closer than K4" {
    const t = std.testing;
    const dsv41 = @import("dsv41.zig");
    const s = mlx.gpuStream();
    const out: usize = 256;
    const in: usize = 384;
    const pool = try pool_mod.Pool.init(t.allocator, 4);
    defer pool.deinit();
    var errs: [2]f64 = undefined;
    for ([_]u32{ 64, 96 }, 0..) |n, i| {
        var case = try StqCase.init(t.allocator, s, "language_model.model.layers.0.self_attn.k_proj", .attn_in, out, in, 2048, 7);
        defer case.deinit();
        case.trunk.stq.rate = .{ .n = n };
        case.trunk.tensors.items[0].rate = .{ .n = n };
        try case.trunk.round(pool, &case.grams, .a);
        const tensor = case.trunk.tensors.items[0];
        const bank = tensor.bank.?;
        try t.expectEqual(n, bank.rate.n);
        try t.expectEqual(@as(usize, in / 16 * (out / 16) * n), bank.trellis.len);
        errs[i] = tensor.rel_err;
        if (n != 96) continue;
        const h = try case.gram(t.allocator, .attn_in, in);
        defer t.allocator.free(h);
        const public = try t.allocator.alloc(f32, in * out);
        defer t.allocator.free(public);
        try source.publicFromBf16(tensor.source, out, in, public);
        var stages = stages_mod.Stages{};
        var want = try dsv41.quantizeDense(t.allocator, s, pool, public, h, in, out, .{ .n = 96 }, case.trunk.stq.pin_pass, tensor.base, &stages);
        defer want.deinit(t.allocator);
        try t.expectEqualSlices(u16, want.trellis, bank.trellis);
    }
    try t.expect(errs[1] < errs[0]);
}

test "trunk stq: the shard holds a trellis, suh and svh per bank under the engine's stamp, with the widest k" {
    const t = std.testing;
    const io = t.io;
    var work = t.tmpDir(.{});
    defer work.cleanup();
    var trunk = Trunk{ .allocator = t.allocator };
    defer trunk.deinit();
    const rates = [_]u32{ 64, 96 };
    for (rates, 0..) |n, i| {
        const bank = Bank{
            .trellis = try t.allocator.alloc(u16, 8 * 8 * n),
            .suh = try t.allocator.alloc(u16, 128),
            .svh = try t.allocator.alloc(u16, 128),
            .rate = .{ .n = n },
        };
        @memset(bank.trellis, @intCast(i + 1));
        @memset(bank.suh, 0x3c00);
        @memset(bank.svh, 0x3c00);
        var name_buf: [64]u8 = undefined;
        const base = try t.allocator.dupe(u8, try std.fmt.bufPrint(&name_buf, "language_model.model.layers.0.linear_attn.w{d}", .{i}));
        try trunk.tensors.append(t.allocator, .{ .base = base, .group = .attn_in, .out = 128, .in = 128, .source = try t.allocator.alloc(u16, 0), .round = .stq, .rate = .{ .n = n }, .bank = bank });
    }
    const banks = try trunk.writeBanks(io, work.dir, "trunk-stq-L00.safetensors", &.{.{ "encoder", "test" }});
    try t.expectEqual(@as(usize, 2), banks);
    const raw = try work.dir.readFileAlloc(io, "trunk-stq-L00.safetensors", t.allocator, .limited(1 << 24));
    defer t.allocator.free(raw);
    var file = try st.parse(t.allocator, raw);
    defer file.deinit(t.allocator);
    try t.expectEqualStrings("stq", file.metadata.get("format").?);
    try t.expectEqualStrings("6", file.metadata.get("k").?);
    try t.expectEqualStrings("spec1", file.metadata.get("codebook").?);
    try t.expectEqualStrings("15", file.metadata.get("window").?);
    try t.expectEqualStrings("spec1-w15", file.metadata.get("codebook_table").?);
    try t.expectEqualStrings("test", file.metadata.get("encoder").?);
    const tr = file.tensors.get("language_model.model.layers.0.linear_attn.w1.trellis").?;
    try t.expectEqualStrings("U16", tr.dtype);
    try t.expectEqualSlices(usize, &.{ 8, 8, 96 }, tr.shape);
    const su = file.tensors.get("language_model.model.layers.0.linear_attn.w0.suh").?;
    try t.expectEqualStrings("F16", su.dtype);
    try t.expectEqualSlices(usize, &.{128}, su.shape);
    try t.expectEqual(@as(usize, 6), file.tensors.count());
}

test "trunk stq: the groups the trellis takes and the grid it needs" {
    const t = std.testing;
    inline for (.{ Group.attn_in, Group.attn_out, Group.moe_in, Group.shared_down, Group.head_in }) |g| try t.expect(stqGroup(g));
    inline for (.{ Group.ple, Group.attn_n, Group.attn_mix, Group.mlp_n, Group.mlp_mix, Group.final_n, Group.final_mix }) |g| try t.expect(!stqGroup(g));
    // The GDN gates (48 rows), the hyper-connection mixes (320 on one side) and Flash-Next's real projections.
    try t.expect(!onGrid(48, 2560));
    try t.expect(!onGrid(320, 10240));
    try t.expect(!onGrid(10240, 320));
    for ([_][2]usize{ .{ 10240, 2560 }, .{ 6144, 2560 }, .{ 2560, 6144 }, .{ 12288, 2560 }, .{ 512, 2560 }, .{ 640, 2560 }, .{ 2560, 640 }, .{ 248320, 2560 } }) |d| try t.expect(onGrid(d[0], d[1]));
}
