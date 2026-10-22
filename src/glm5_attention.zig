//! Request-local IndexPool and NoPE latent attention. IndexPool state is lossless; the
//! latent is BF16, kv8 or kv6 as the request asks (`State.latent_bits`).
const std = @import("std");
const mlx = @import("mlx.zig");
const latent_store = @import("glm5_latent.zig");
pub const Latent = latent_store.Latent;
const partials = @import("glm5_attention_prefill.zig");
const packed_nax = @import("glm5_attention_nax_packed.zig");
const latent_overlay = @import("glm5_attention_overlay.zig");
const Arr = mlx.mlx_array;
const Ops = @import("glm5_model.zig").Ops;
const nil = Arr{ .ctx = null };
const pool_size = 4;
const pool_budget = 512;
const selected_width = pool_size * pool_budget + pool_size - 1;
pub const score_scratch_bytes: usize = 2 * 1024 * 1024;
pub const attention_scratch_bytes: usize = 8 * 1024 * 1024;
/// Scalar latent-attention dispatches, [dense, sparse].
pub var scalar_calls: [2]usize = .{ 0, 0 };
/// Packed prefill selection planes at `history` rows: two 32-row tiles in flight, each two
/// 16-row selections holding their FP32 scores (ranked selection keeps no other plane).
pub fn selectionScratchBytes(history: usize) !usize {
    return std.math.mul(usize, 2 * 2 * packed_nax.max_rows * 4, history / pool_size);
}
/// Paired packed tiles keep a second tile live beside the first.
pub fn packedCadenceTransientBudget(chunk: usize, pending_layers: usize) !usize {
    if (!packed_nax.enabled() or chunk <= packed_nax.wide_rows) return 0;
    return std.math.mul(usize, packed_nax.scratch_limit, pending_layers);
}

const Scope = struct {
    s: mlx.mlx_stream,
    values: [96]Arr = undefined,
    count: usize = 0,
    fn deinit(self: *Scope) void {
        for (self.values[0..self.count]) |a| _ = mlx.mlx_array_free(a);
    }
    fn slot(self: *Scope) !*Arr {
        if (self.count == self.values.len) return error.AttentionGraphTooLarge;
        self.values[self.count] = mlx.mlx_array_new();
        self.count += 1;
        return &self.values[self.count - 1];
    }
    fn own(self: *Scope, a: Arr) !Arr {
        const out = self.slot() catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
        _ = mlx.mlx_array_free(out.*);
        out.* = a;
        return a;
    }
    fn result(_: *Scope, a: Arr) !Arr {
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_set(&out, a));
        return out;
    }
    fn cast(self: *Scope, a: Arr, dtype: mlx.mlx_dtype) !Arr {
        if (mlx.mlx_array_dtype(a) == dtype) return a;
        const out = try self.slot();
        try mlx.check(mlx.mlx_astype(out, a, dtype, self.s));
        return out.*;
    }
    fn shape(self: *Scope, a: Arr, dims: []const c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_reshape(out, a, dims.ptr, dims.len, self.s));
        return out.*;
    }
    fn cut(self: *Scope, a: Arr, start: c_int, end: c_int) !Arr {
        const sh = mlx.getShape(a);
        var lo: [4]c_int = @splat(0);
        var hi: [4]c_int = @splat(0);
        const step: [4]c_int = @splat(1);
        if (sh.len > 4 or start < 0 or end < start or end > sh[0]) return error.InvalidGlmAttentionShape;
        @memcpy(hi[0..sh.len], sh);
        lo[0] = start;
        hi[0] = end;
        const out = try self.slot();
        try mlx.check(mlx.mlx_slice(out, a, &lo, sh.len, &hi, sh.len, &step, sh.len, self.s));
        return out.*;
    }
    fn join(self: *Scope, a: Arr, b: Arr) !Arr {
        if (a.ctx == null or mlx.getShape(a)[0] == 0) return b;
        const parts = [_]Arr{ a, b };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        const out = try self.slot();
        try mlx.check(mlx.mlx_concatenate_axis(out, vec, 0, self.s));
        return out.*;
    }
    fn zeros(self: *Scope, dims: []const c_int, dtype: mlx.mlx_dtype) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_zeros(out, dims.ptr, dims.len, dtype, self.s));
        return out.*;
    }
    fn copy(self: *Scope, a: Arr) !Arr {
        const zero = try self.zeros(&.{}, mlx.mlx_array_dtype(a));
        const out = try self.slot();
        try mlx.check(mlx.mlx_add(out, a, zero, self.s));
        return out.*;
    }
    fn appendRows(self: *Scope, buffer: Arr, used: usize, chunk: Arr) !Arr {
        const sh = mlx.getShape(chunk);
        const needed = try std.math.add(usize, used, @intCast(sh[0]));
        if (needed > std.math.maxInt(c_int) - 255) return error.InvalidGlmAttentionShape;
        const current: usize = if (buffer.ctx == null) 0 else @intCast(mlx.getShape(buffer)[0]);
        var target = buffer;
        if (needed > current) {
            const capacity = (needed + 255) / 256 * 256;
            const padding = try self.zeros(&.{ @intCast(capacity - current), sh[1] }, mlx.mlx_array_dtype(chunk));
            target = try self.join(buffer, padding);
        }
        const out = try self.slot();
        try mlx.check(mlx.mlx_slice_update(out, target, chunk, &.{ @intCast(used), 0 }, 2, &.{ @intCast(needed), sh[1] }, 2, &.{ 1, 1 }, 2, self.s));
        return out.*;
    }
};

fn supported(dtype: mlx.mlx_dtype) bool {
    return dtype == .bfloat16 or dtype == .float32;
}

fn compress(scope: *Scope, keys: Arr, gates: Arr, ape: Arr, ready: c_int) !Arr {
    const width = mlx.getShape(keys)[1];
    const k = try scope.shape(try scope.cut(keys, 0, ready), &.{ @divExact(ready, 4), 4, width });
    const g = try scope.shape(try scope.cut(gates, 0, ready), &.{ @divExact(ready, 4), 4, width });
    const logits = try scope.slot();
    try mlx.check(mlx.mlx_add(logits, g, ape, scope.s));
    const probs = try scope.slot();
    try mlx.check(mlx.mlx_softmax_axis(probs, logits.*, 1, false, scope.s));
    const weighted = try scope.slot();
    try mlx.check(mlx.mlx_multiply(weighted, probs.*, k, scope.s));
    const pooled = try scope.slot();
    try mlx.check(mlx.mlx_sum_axis(pooled, weighted.*, 1, false, scope.s));
    return scope.cast(pooled.*, mlx.mlx_array_dtype(keys));
}

pub const State = struct {
    /// BF16 rows, or kv8/kv6 u32 codes with `latent_scales` and `latent_biases` beside them.
    latent: Arr = nil,
    latent_scales: Arr = nil,
    latent_biases: Arr = nil,
    pooled: Arr = nil,
    tail_keys: Arr = nil,
    tail_gates: Arr = nil,
    /// Pools an index-only append completed past `pooled`'s rows: sibling branches share the
    /// reserved buffer, so writing it would copy it whole for every branch.
    pool_tail: Arr = nil,
    processed: usize = 0,
    /// 0 stores BF16 latent rows, 8 stores kv8, 6 stores kv6; kept across reset.
    latent_bits: u8 = 0,

    pub fn init() State {
        return .{};
    }
    pub fn deinit(self: *State) void {
        for (self.arrays()) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        self.* = .{ .latent_bits = self.latent_bits };
    }
    pub fn reset(self: *State) void {
        self.deinit();
    }
    pub fn arrays(self: *const State) [7]Arr {
        return .{ self.latent, self.latent_scales, self.latent_biases, self.pooled, self.tail_keys, self.tail_gates, self.pool_tail };
    }
    fn poolTailRows(self: *const State) usize {
        return if (self.pool_tail.ctx == null) 0 else @intCast(mlx.getShape(self.pool_tail)[0]);
    }
    fn indexKeys(self: *const State) Arr {
        return if (self.pooled.ctx != null) self.pooled else self.pool_tail;
    }
    pub fn latentView(self: *const State) Latent {
        return .{ .data = self.latent, .scales = self.latent_scales, .biases = self.latent_biases };
    }
    /// A second owner of the same immutable arrays; appends replace handles, never contents.
    pub fn share(self: *const State) !State {
        var copy = State{ .processed = self.processed, .latent_bits = self.latent_bits };
        errdefer copy.deinit();
        inline for (.{ "latent", "latent_scales", "latent_biases", "pooled", "tail_keys", "tail_gates", "pool_tail" }) |name| {
            const value = @field(self.*, name);
            if (value.ctx != null) {
                @field(copy, name) = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&@field(copy, name), value));
            }
        }
        return copy;
    }
    pub fn evaluate(self: *const State) !void {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (self.arrays()) |a| if (a.ctx != null) try mlx.check(mlx.mlx_vector_array_append_value(vec, a));
        try mlx.check(mlx.mlx_eval(vec));
    }
    pub fn append(self: *State, latent: Arr, keys: Arr, gates: Arr, ape: Arr, s: mlx.mlx_stream) !usize {
        return self.appendImpl(latent, keys, gates, ape, s, true);
    }
    /// Verification forks retain their latent prefix; attention reads the supplied overlay.
    pub fn appendIndexOnly(self: *State, latent: Arr, keys: Arr, gates: Arr, ape: Arr, s: mlx.mlx_stream) !usize {
        return self.appendImpl(latent, keys, gates, ape, s, false);
    }
    fn appendImpl(self: *State, latent: Arr, keys: Arr, gates: Arr, ape: Arr, s: mlx.mlx_stream, store_latent: bool) !usize {
        if (latent.ctx == null or keys.ctx == null or gates.ctx == null or ape.ctx == null) return error.InvalidGlmAttentionShape;
        const ls = mlx.getShape(latent);
        const ks = mlx.getShape(keys);
        if (ls.len != 2 or ks.len != 2 or ls[0] <= 0 or ls[1] <= 0 or ks[1] <= 0 or ls[0] != ks[0] or
            !std.mem.eql(c_int, ks, mlx.getShape(gates)) or !std.mem.eql(c_int, &.{ 4, ks[1] }, mlx.getShape(ape)) or
            !supported(mlx.mlx_array_dtype(latent)) or !supported(mlx.mlx_array_dtype(keys)) or
            mlx.mlx_array_dtype(gates) != mlx.mlx_array_dtype(keys) or mlx.mlx_array_dtype(ape) != mlx.mlx_array_dtype(keys)) return error.InvalidGlmAttentionShape;
        if (self.latent_bits != 0 and (!latent_store.storable(self.latent_bits) or mlx.mlx_array_dtype(latent) != .bfloat16 or
            @mod(ls[1], @as(c_int, latent_store.group_size)) != 0)) return error.InvalidGlmAttentionShape;
        if (self.processed != 0 and (self.latentView().width() != ls[1] or self.latentView().dtype() != mlx.mlx_array_dtype(latent))) return error.InvalidGlmAttentionShape;
        if (self.tail_keys.ctx != null and (mlx.getShape(self.tail_keys)[1] != ks[1] or mlx.mlx_array_dtype(self.tail_keys) != mlx.mlx_array_dtype(keys))) return error.InvalidGlmAttentionShape;
        for ([_]Arr{ self.pooled, self.pool_tail }) |a| if (a.ctx != null and (mlx.getShape(a)[1] != ks[1] or mlx.mlx_array_dtype(a) != mlx.mlx_array_dtype(keys))) return error.InvalidGlmAttentionShape;
        if (store_latent and self.pool_tail.ctx != null) return error.InvalidGlmAttentionShape;
        const next = try std.math.add(usize, self.processed, @intCast(ls[0]));
        var scope = Scope{ .s = s };
        defer scope.deinit();
        var l = self.latent;
        var l_scales = self.latent_scales;
        var l_biases = self.latent_biases;
        if (store_latent and self.latent_bits == 0) l = try scope.appendRows(self.latent, self.processed, latent);
        if (store_latent and self.latent_bits != 0) {
            var q = try latent_store.quantize(latent, self.latent_bits, s);
            defer q.deinit();
            l = try scope.appendRows(self.latent, self.processed, q.q);
            l_scales = try scope.appendRows(self.latent_scales, self.processed, q.scales);
            l_biases = try scope.appendRows(self.latent_biases, self.processed, q.biases);
        }
        const k = try scope.join(self.tail_keys, keys);
        const g = try scope.join(self.tail_gates, gates);
        const ready = @divTrunc(mlx.getShape(k)[0], 4) * 4;
        const fresh = if (ready > 0) try compress(&scope, k, g, ape, ready) else nil;
        const p = if (ready > 0 and store_latent) try scope.appendRows(self.pooled, self.processed / 4, fresh) else self.pooled;
        const pt = if (ready > 0 and !store_latent) try scope.join(self.pool_tail, fresh) else self.pool_tail;
        const tail_k = try scope.copy(try scope.cut(k, ready, mlx.getShape(k)[0]));
        const tail_g = try scope.copy(try scope.cut(g, ready, mlx.getShape(g)[0]));
        var new = State{ .processed = next, .latent_bits = self.latent_bits };
        errdefer new.deinit();
        if (l.ctx != null) new.latent = try scope.result(l);
        if (l_scales.ctx != null) new.latent_scales = try scope.result(l_scales);
        if (l_biases.ctx != null) new.latent_biases = try scope.result(l_biases);
        if (p.ctx != null) new.pooled = try scope.result(p);
        if (pt.ctx != null) new.pool_tail = try scope.result(pt);
        new.tail_keys = try scope.result(tail_k);
        new.tail_gates = try scope.result(tail_g);
        const before = self.processed;
        self.deinit();
        self.* = new;
        return before;
    }
};

var score_kernel: ?mlx.mlx_fast_metal_kernel = null;
var score_tail_kernel: ?mlx.mlx_fast_metal_kernel = null;
var expand_kernel: ?mlx.mlx_fast_metal_kernel = null;
var expand_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;
var score_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;
var attention_kernels: [latent_store.variants]?mlx.mlx_fast_metal_kernel = @splat(null);
var merge_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn kernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    return kernelWithHeader(slot, name, inputs, outputs, source, "");
}
fn kernelWithHeader(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [:0]const u8, header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}
fn apply(k: mlx.mlx_fast_metal_kernel, inputs: []const Arr, cfg: mlx.mlx_fast_metal_kernel_config, s: mlx.mlx_stream) !mlx.mlx_vector_array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    errdefer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    return ov;
}
fn kernelOutput(scope: *Scope, ov: mlx.mlx_vector_array, i: usize) !Arr {
    const out = try scope.slot();
    try mlx.check(mlx.mlx_vector_array_get(out, ov, i));
    return out.*;
}
fn template(cfg: mlx.mlx_fast_metal_kernel_config, name: [*:0]const u8, value: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, value));
}

// The SIMD-sum scorer's arithmetic without its shuffles: each lane owns one head, forms that
// scorer's 32 lane partials in lane order (each sequential over d = l, l+32, ...) and joins them in
// `simd_sum`'s order, an xor butterfly over 1, 2, 4, 8, 16 lanes: balanced pairs in lane order.
const SCORE_HEADER: [:0]const u8 =
    \\#define SUSHI_TREE_PUSH(v) { float c = (v); \
    \\  if ((l & 1u) == 0u) s0 = c; else { c = s0 + c; \
    \\  if ((l & 2u) == 0u) s1 = c; else { c = s1 + c; \
    \\  if ((l & 4u) == 0u) s2 = c; else { c = s2 + c; \
    \\  if ((l & 8u) == 0u) s3 = c; else { c = s3 + c; \
    \\  if ((l & 16u) == 0u) s4 = c; else dot = s4 + c; } } } } }
    \\
;
const SCORE_KEYS: [:0]const u8 = "#define SUSHI_POOL_KEYS(p) (keys + size_t(p) * uint(I))\n";
const SCORE_TAIL_KEYS: [:0]const u8 = "#define SUSHI_POOL_KEYS(p) ((p) < uint(tail_base) ? keys + size_t(p) * uint(I) : tail + size_t((p) - uint(tail_base)) * uint(I))\n";
const SCORE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint lane = thread_position_in_threadgroup.x;
    \\const uint ty = thread_position_in_threadgroup.y;
    \\const uint row = threadgroup_position_in_grid.y * uint(RT) + ty;
    \\const uint pools = uint(count);
    \\const uint first = threadgroup_position_in_grid.x * 32u;
    \\const uint last = min(pools, first + 32u);
    \\const uint limit = uint(offset) + row + 1u;
    \\// Lanes past the last head repeat it; only heads below J are summed.
    \\const uint head = min(lane, uint(J) - 1u);
    \\threadgroup float terms[RT][8][33];
    \\InT qv[I];
    \\for (uint d = 0; d < uint(I); ++d) qv[d] = q[(row * uint(J) + head) * uint(I) + d];
    \\const float weight = float(weights[row * uint(J) + head]);
    \\for (uint base = first; base < last; base += 8u) {
    \\  const uint block = min(8u, last - base);
    \\  for (uint j = 0; j < block; ++j) {
    \\    const uint p = base + j;
    \\    if ((p + 1u) * 4u > limit) continue;
    \\    const device InT* kp = SUSHI_POOL_KEYS(p);
    \\    float s0, s1, s2, s3, s4, dot;
    \\    #pragma unroll
    \\    for (uint l = 0; l < 32u; ++l) {
    \\      float part = 0.0f;
    \\      if (l < uint(I)) part += float(qv[l]) * float(kp[l]);
    \\      if (l + 32u < uint(I)) part += float(qv[l + 32u]) * float(kp[l + 32u]);
    \\      if (l + 64u < uint(I)) part += float(qv[l + 64u]) * float(kp[l + 64u]);
    \\      if (l + 96u < uint(I)) part += float(qv[l + 96u]) * float(kp[l + 96u]);
    \\      SUSHI_TREE_PUSH(part)
    \\    }
    \\    terms[ty][j][lane] = float(InT(max(float(InT(dot)), 0.0f) * weight));
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (lane < block) {
    \\    const uint p = base + lane;
    \\    float total = -INFINITY;
    \\    if ((p + 1u) * 4u <= limit) {
    \\      total = 0.0f;
    \\      for (uint h = 0; h < uint(J); ++h) total += terms[ty][lane][h];
    \\      total = float(InT(total));
    \\    }
    \\    out[row * pools + p] = total;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
;
const EXPAND: [:0]const u8 =
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=uint(ROWS)*2051u) return;
    \\const uint row=i/2051u;
    \\const uint col=i%2051u;
    \\const uint pos=uint(offset)+row;
    \\int token=-1;
    \\if(col<uint(K)*4u) {
    \\ const int p=int(selected[row*uint(K)+col/4u]);
    \\ if(p>=0 && (uint(p)+1u)*4u<=pos+1u) token=p*4+int(col%4u);
    \\} else if(col>=2048u) {
    \\ const uint rem=(pos+1u)%4u;
    \\ const uint j=col-2048u;
    \\ if(j<rem) token=int(pos+1u-rem+j);
    \\}
    \\out[i]=token;
;
fn replaceOnce(comptime source: []const u8, comptime old: []const u8, comptime value: []const u8) [:0]const u8 {
    const at = comptime std.mem.indexOf(u8, source, old) orelse @compileError("missing " ++ old);
    return source[0..at] ++ value ++ source[at + old.len ..] ++ "";
}
// The verifier's rows sit at their own offsets and carry their own tails.
const SCORE_ROWS_KEYS: [:0]const u8 = "#define SUSHI_POOL_KEYS(p) ((p) < uint(tail_base) ? keys + size_t(p) * uint(I) : tail + (size_t(row) * uint(TAIL_ROWS) + ((p) - uint(tail_base))) * uint(I))\n";
const SCORE_ROWS: [:0]const u8 = replaceOnce(SCORE, "const uint limit = uint(offset) + row + 1u;", "const uint limit = uint(offsets[row]) + 1u;");
const EXPAND_ROWS: [:0]const u8 = replaceOnce(EXPAND, "const uint pos=uint(offset)+row;", "const uint pos=uint(offsets[row]);");
const ATTENTION: [:0]const u8 = partials.common ++ partials.partial_tail;
const MERGE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=uint(ROWS)*uint(H)*uint(D)) return;
    \\const uint rowhead=i/uint(D),d=i%uint(D);
    \\float maximum=-INFINITY;
    \\for(uint p=0;p<uint(SPLITS);++p) maximum=max(maximum,stats[(rowhead*uint(SPLITS)+p)*2u]);
    \\float sum=0.0f,denom=0.0f;
    \\for(uint p=0;p<uint(SPLITS);++p) {
    \\ const uint base=rowhead*uint(SPLITS)+p;
    \\ const float n=stats[base*2u+1u];
    \\ if(n>0.0f) {const float w=precise::exp(stats[base*2u]-maximum);sum+=w*partial[base*uint(D)+d];denom+=w*n;}
    \\}
    \\out[i]=OutT(denom>0.0f?sum/denom:0.0f);
;

var tree_score_calls: usize = 0;

fn indexScores(scope: *Scope, state: *const State, index_q: Arr, weights: Arr, offset: usize) !Arr {
    const pools: c_int = @intCast(state.processed / 4);
    if (state.pool_tail.ctx != null) {
        const base: c_int = pools - @as(c_int, @intCast(state.poolTailRows()));
        return treeScoresTail(scope, state.indexKeys(), state.pool_tail, base, index_q, weights, offset, pools);
    }
    if (try @import("glm5_indexpool_nax.zig").tryScores(index_q, state.pooled, weights, offset, @intCast(pools), scope.s)) |out|
        return scope.own(out);
    return treeScores(scope, state.pooled, index_q, weights, offset, pools);
}

fn treeScores(scope: *Scope, pooled: Arr, index_q: Arr, weights: Arr, offset: usize, pools: c_int) !Arr {
    return treeScoresTail(scope, pooled, nil, 0, index_q, weights, offset, pools);
}

/// `tail` nil reads every pool from `pooled`; otherwise pools from `tail_base` on come from `tail`.
fn treeScoresTail(scope: *Scope, pooled: Arr, tail: Arr, tail_base: c_int, index_q: Arr, weights: Arr, offset: usize, pools: c_int) !Arr {
    const sh = mlx.getShape(index_q);
    const rows = sh[0];
    if (sh[1] > 32 or sh[2] > 128) return error.InvalidGlmAttentionShape;
    // Non-NAX pipelines can cap this kernel below 512 threads.
    const row_group = @min(rows, if (@import("glm5_model.zig").naxArms()) @as(c_int, 16) else 8);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ rows, pools }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(pools + 31, 32) * 32, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, row_group, 1));
    try template(cfg, "J", sh[1]);
    try template(cfg, "I", sh[2]);
    try template(cfg, "RT", row_group);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", mlx.mlx_array_dtype(index_q)));
    const off = try scope.own(mlx.mlx_array_new_int(@intCast(offset)));
    const count = try scope.own(mlx.mlx_array_new_int(pools));
    const ov = if (tail.ctx == null) blk: {
        const k = try kernelWithHeader(&score_kernel, "sushi_glm_index_scores_tree", &.{ "q", "keys", "weights", "offset", "count" }, &.{"out"}, SCORE, SCORE_HEADER ++ SCORE_KEYS);
        break :blk try apply(k, &.{ index_q, pooled, weights, off, count }, cfg, scope.s);
    } else blk: {
        const first = try scope.own(mlx.mlx_array_new_int(tail_base));
        const k = try kernelWithHeader(&score_tail_kernel, "sushi_glm_index_scores_tree_tail", &.{ "q", "keys", "tail", "weights", "offset", "count", "tail_base" }, &.{"out"}, SCORE, SCORE_HEADER ++ SCORE_TAIL_KEYS);
        break :blk try apply(k, &.{ index_q, pooled, tail, weights, off, count, first }, cfg, scope.s);
    };
    defer _ = mlx.mlx_vector_array_free(ov);
    tree_score_calls += 1;
    if (tree_score_calls == 1) @import("log.zig").info("[glm-index] tree scorer engaged\n", .{});
    return kernelOutput(scope, ov, 0);
}

/// The first `min(pools, 512)` of a stable ascending sort of the negated scores.
fn partitionTop(scope: *Scope, scores: Arr, rows: c_int) !Arr {
    const k = @min(mlx.getShape(scores)[1], pool_budget);
    const neg = try scope.slot();
    try mlx.check(mlx.mlx_negative(neg, scores, scope.s));
    const partition = try scope.slot();
    try mlx.check(mlx.mlx_argpartition_axis(partition, neg.*, k - 1, -1, scope.s));
    const selected = try scope.slot();
    try mlx.check(mlx.mlx_slice(selected, partition.*, &.{ 0, 0 }, 2, &.{ rows, k }, 2, &.{ 1, 1 }, 2, scope.s));
    return selected.*;
}

// `partitionTop` in one dispatch per row: radix-16 search for the 512th key, every larger key plus
// the lowest-pool ties at it, then each candidate's rank. Keys order like the stable sort: higher
// score first, NaN last, -0 equal to +0. BF16-valued scores need only the top 16 key bits.
const RANK_HEADER: [:0]const u8 =
    \\template <int KB>
    \\inline uint sushi_rank_key(float v) {
    \\  if (isnan(v)) return 0u;
    \\  const uint b = as_type<uint>(v == 0.0f ? 0.0f : v);
    \\  const uint u = (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
    \\  return KB == 16 ? (u >> 16) : u;
    \\}
    \\
;
const RANK: [:0]const u8 =
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint row = threadgroup_position_in_grid.x;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint n = uint(count);
    \\const device float* s = scores + size_t(row) * n;
    \\threadgroup uint partial[32][16];
    \\threadgroup uint bins[16];
    \\threadgroup uint pick[2];
    \\threadgroup ulong cand[512];
    \\uint prefix = 0u, mask = 0u, want = 512u;
    \\for (int shift = KB - 4; shift >= 0; shift -= 4) {
    \\  uint4 c0 = 0u, c1 = 0u, c2 = 0u, c3 = 0u;
    \\  for (uint i = tid; i < n; i += 1024u) {
    \\    const uint k = sushi_rank_key<KB>(s[i]);
    \\    if ((k & mask) != prefix) continue;
    \\    const uint4 d = uint4((k >> uint(shift)) & 15u);
    \\    c0 += uint4(d == uint4(0u, 1u, 2u, 3u));
    \\    c1 += uint4(d == uint4(4u, 5u, 6u, 7u));
    \\    c2 += uint4(d == uint4(8u, 9u, 10u, 11u));
    \\    c3 += uint4(d == uint4(12u, 13u, 14u, 15u));
    \\  }
    \\  c0 = simd_sum(c0); c1 = simd_sum(c1); c2 = simd_sum(c2); c3 = simd_sum(c3);
    \\  if (lane == 0u) for (uint j = 0; j < 4u; ++j) {
    \\    partial[sg][j] = c0[j]; partial[sg][4u + j] = c1[j]; partial[sg][8u + j] = c2[j]; partial[sg][12u + j] = c3[j];
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (tid < 16u) { uint t = 0u; for (uint g = 0; g < 32u; ++g) t += partial[g][tid]; bins[tid] = t; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (tid == 0u) {
    \\    uint above = 0u, chosen = 0u;
    \\    for (int d = 15; d >= 0; --d) {
    \\      if (above + bins[d] >= want) { chosen = uint(d); break; }
    \\      above += bins[d];
    \\    }
    \\    pick[0] = chosen;
    \\    pick[1] = want - above;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  prefix |= pick[0] << uint(shift);
    \\  mask |= 15u << uint(shift);
    \\  want = pick[1];
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\const uint per = (n + 1023u) / 1024u;
    \\const uint lo = min(n, tid * per), hi = min(n, lo + per);
    \\uint gt = 0u, eq = 0u;
    \\for (uint i = lo; i < hi; ++i) {
    \\  const uint k = sushi_rank_key<KB>(s[i]);
    \\  gt += k > prefix ? 1u : 0u;
    \\  eq += k == prefix ? 1u : 0u;
    \\}
    \\const uint gt_lane = simd_prefix_exclusive_sum(gt), eq_lane = simd_prefix_exclusive_sum(eq);
    \\if (lane == 31u) { partial[sg][0] = gt_lane + gt; partial[sg][1] = eq_lane + eq; }
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\uint gt_at = gt_lane, eq_at = eq_lane;
    \\for (uint g = 0; g < sg; ++g) { gt_at += partial[g][0]; eq_at += partial[g][1]; }
    \\const uint above = 512u - want;
    \\for (uint i = lo; i < hi; ++i) {
    \\  const uint k = sushi_rank_key<KB>(s[i]);
    \\  const ulong entry = (ulong(k) << 32) | ulong(0xFFFFFFFFu - i);
    \\  if (k > prefix) cand[gt_at++] = entry;
    \\  else if (k == prefix) { if (eq_at < want) cand[above + eq_at] = entry; ++eq_at; }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (tid < 512u) {
    \\  const ulong mine = cand[tid];
    \\  uint rank = 0u;
    \\  for (uint j = 0; j < 512u; ++j) rank += cand[j] > mine ? 1u : 0u;
    \\  out[row * 512u + rank] = 0xFFFFFFFFu - uint(mine & 0xFFFFFFFFul);
    \\}
;
var rank_kernel: ?mlx.mlx_fast_metal_kernel = null;
var rank_top_calls: usize = 0;

/// Needs at least 512 pools; `key_bits` 16 only for BF16-valued scores.
fn rankTop(scope: *Scope, scores: Arr, key_bits: u8) !Arr {
    const sh = mlx.getShape(scores);
    if (sh.len != 2 or sh[1] < pool_budget or mlx.mlx_array_dtype(scores) != .float32 or (key_bits != 16 and key_bits != 32)) return error.InvalidGlmAttentionShape;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ sh[0], pool_budget }, 2, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, sh[0] * 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
    try template(cfg, "KB", key_bits);
    const count = try scope.own(mlx.mlx_array_new_int(sh[1]));
    const k = try kernelWithHeader(&rank_kernel, "sushi_glm_rank_top512", &.{ "scores", "count" }, &.{"out"}, RANK, RANK_HEADER);
    const ov = try apply(k, &.{ scores, count }, cfg, scope.s);
    defer _ = mlx.mlx_vector_array_free(ov);
    rank_top_calls += 1;
    if (rank_top_calls == 1) @import("log.zig").info("[glm-index] ranked top-512 engaged\n", .{});
    return kernelOutput(scope, ov, 0);
}

/// The top pools of each row, [rows, min(pools, 512)], best first for ranked rows.
fn selectPools(scope: *Scope, state: *const State, index_q: Arr, weights: Arr, offset: usize) !Arr {
    const rows = mlx.getShape(index_q)[0];
    const pools: c_int = @intCast(state.processed / 4);
    const scores = try indexScores(scope, state, index_q, weights, offset);
    return if (rows > 1 and pools >= pool_budget)
        try rankTop(scope, scores, if (mlx.mlx_array_dtype(index_q) == .bfloat16) 16 else 32)
    else
        try partitionTop(scope, scores, rows);
}

fn selectChunk(scope: *Scope, state: *const State, index_q: Arr, weights: Arr, offset: usize) !Arr {
    const rows = mlx.getShape(index_q)[0];
    const pools: c_int = @intCast(state.processed / 4);
    const off = try scope.own(mlx.mlx_array_new_int(@intCast(offset)));
    const selected = try selectPools(scope, state, index_q, weights, offset);
    const ec = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(ec);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(ec, &.{ rows, selected_width }, 2, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(ec, rows * selected_width, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(ec, 256, 1, 1));
    try template(ec, "ROWS", rows);
    try template(ec, "K", @min(pools, pool_budget));
    const ek = try kernel(&expand_kernel, "sushi_glm_expand_pools", &.{ "selected", "offset" }, &.{"out"}, EXPAND);
    const ev = try apply(ek, &.{ selected, off }, ec, scope.s);
    defer _ = mlx.mlx_vector_array_free(ev);
    return kernelOutput(scope, ev, 0);
}

fn selectPackedChunk(scope: *Scope, state: *const State, index_q: Arr, weights: Arr, offset: usize) !Arr {
    const rows = mlx.getShape(index_q)[0];
    if (rows > packed_nax.max_rows and rows != packed_nax.wide_rows) return error.InvalidGlmAttentionShape;
    return selectChunk(scope, state, index_q, weights, offset);
}

/// The 16-row selector geometry a 32-row tile used to join.
fn selectPackedHalves(scope: *Scope, state: *const State, index_q: Arr, weights: Arr, offset: usize) !Arr {
    const first = try selectChunk(scope, state, try scope.cut(index_q, 0, 16), try scope.cut(weights, 0, 16), offset);
    const second = try selectChunk(scope, state, try scope.cut(index_q, 16, 32), try scope.cut(weights, 16, 32), offset + 16);
    return scope.join(first, second);
}

/// `decodeSelected` before the token expansion: the 512 pools of a serial row past the dense window,
/// null inside it (every token is selected there).
fn decodePools(state: *const State, index_q: Arr, weights: Arr, offset: usize, s: mlx.mlx_stream) !?Arr {
    if (offset + 1 <= pool_size * (pool_budget + 1) - 1) return null;
    var scope = Scope{ .s = s };
    defer scope.deinit();
    var node = state.*;
    node.processed = offset + 1;
    return try scope.result(try selectPools(&scope, &node, index_q, weights, offset));
}

/// The same per-node rule is used for serial decode and verifier ancestry.
pub fn decodeSelected(state: *const State, index_q: Arr, weights: Arr, offset: usize, s: mlx.mlx_stream) !Arr {
    var scope = Scope{ .s = s };
    defer scope.deinit();
    if (offset + 1 > pool_size * (pool_budget + 1) - 1) {
        var node = state.*;
        node.processed = offset + 1;
        return scope.result(try selectChunk(&scope, &node, index_q, weights, offset));
    }
    const ids = try scope.slot();
    try mlx.check(mlx.mlx_arange(ids, 0, selected_width, 1, .int32, s));
    return scope.result(try scope.shape(ids.*, &.{ 1, selected_width }));
}

pub const SelectRow = struct { state: *const State, index_q: Arr, weights: Arr, offset: usize };

/// `decodeSelected` for the rows of one verifier tree. Rows that share a pool count and tail length score, partition
/// and expand together, which the one-row geometry of each stays bit-identical to; the others select alone. All rows
/// must branch from the same prefix. `out[i]` is owned by the caller.
pub fn decodeSelectedRows(rows: []const SelectRow, out: []Arr, s: mlx.mlx_stream) !void {
    if (rows.len > 16 or out.len < rows.len) return error.InvalidGlmAttentionShape;
    var done: [16]bool = @splat(false);
    for (rows, 0..) |row, i| {
        if (done[i]) continue;
        var members: [16]usize = undefined;
        var count: usize = 0;
        if (row.offset + 1 > pool_size * (pool_budget + 1) - 1) for (rows[i..], i..) |other, j| {
            if (!done[j] and (other.offset + 1) / 4 == (row.offset + 1) / 4 and other.state.poolTailRows() == row.state.poolTailRows() and
                mlx.mlx_array_dtype(other.index_q) == mlx.mlx_array_dtype(row.index_q))
            {
                members[count] = j;
                count += 1;
            }
        };
        if (count < 2) {
            done[i] = true;
            out[i] = try decodeSelected(row.state, row.index_q, row.weights, row.offset, s);
            continue;
        }
        for (members[0..count]) |j| done[j] = true;
        try selectGroup(rows, members[0..count], out, s);
    }
}

fn stacked(scope: *Scope, parts: []const Arr) !Arr {
    const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    const joined = try scope.slot();
    try mlx.check(mlx.mlx_concatenate_axis(joined, vec, 0, scope.s));
    return joined.*;
}

var group_selects: usize = 0;

fn selectGroup(rows: []const SelectRow, members: []const usize, out: []Arr, s: mlx.mlx_stream) !void {
    group_selects += 1;
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const g: c_int = @intCast(members.len);
    const first = rows[members[0]];
    const pools: c_int = @intCast((first.offset + 1) / 4);
    const tail_rows: c_int = @intCast(first.state.poolTailRows());
    var queries: [16]Arr = undefined;
    var weights: [16]Arr = undefined;
    var tails: [16]Arr = undefined;
    var offsets: [16]u32 = undefined;
    for (members, 0..) |j, k| {
        queries[k] = rows[j].index_q;
        weights[k] = rows[j].weights;
        tails[k] = rows[j].state.pool_tail;
        offsets[k] = @intCast(rows[j].offset);
    }
    const keys = first.state.indexKeys();
    const tail = if (tail_rows > 0) try scope.shape(try stacked(&scope, tails[0..members.len]), &.{ g, tail_rows, mlx.getShape(keys)[1] }) else keys;
    const offset_array = try scope.own(mlx.mlx_array_new_data(&offsets, &.{g}, 1, .uint32));
    const scores = try treeScoresRows(&scope, keys, tail, tail_rows, try stacked(&scope, queries[0..members.len]), try stacked(&scope, weights[0..members.len]), offset_array, pools);
    const pools_top = try partitionTop(&scope, scores, g);
    const ec = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(ec);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(ec, &.{ g, selected_width }, 2, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(ec, g * selected_width, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(ec, 256, 1, 1));
    try template(ec, "ROWS", g);
    try template(ec, "K", @min(pools, pool_budget));
    const ek = try kernel(&expand_rows_kernel, "sushi_glm_expand_pools_rows", &.{ "selected", "offsets" }, &.{"out"}, EXPAND_ROWS);
    const ev = try apply(ek, &.{ pools_top, offset_array }, ec, s);
    defer _ = mlx.mlx_vector_array_free(ev);
    const tokens = try kernelOutput(&scope, ev, 0);
    for (members, 0..) |j, k| out[j] = try scope.result(try scope.cut(tokens, @intCast(k), @intCast(k + 1)));
}

/// `treeScoresTail` for rows at their own offsets, each with its own tail of `tail_rows` pools.
fn treeScoresRows(scope: *Scope, pooled: Arr, tail: Arr, tail_rows: c_int, index_q: Arr, weights: Arr, offsets: Arr, pools: c_int) !Arr {
    const sh = mlx.getShape(index_q);
    const rows = sh[0];
    if (sh[1] > 32 or sh[2] > 128) return error.InvalidGlmAttentionShape;
    const row_group = @min(rows, if (@import("glm5_model.zig").naxArms()) @as(c_int, 16) else 8);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ rows, pools }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(pools + 31, 32) * 32, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, row_group, 1));
    try template(cfg, "J", sh[1]);
    try template(cfg, "I", sh[2]);
    try template(cfg, "RT", row_group);
    try template(cfg, "TAIL_ROWS", tail_rows);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", mlx.mlx_array_dtype(index_q)));
    const count = try scope.own(mlx.mlx_array_new_int(pools));
    const first = try scope.own(mlx.mlx_array_new_int(pools - tail_rows));
    const k = try kernelWithHeader(&score_rows_kernel, "sushi_glm_index_scores_tree_rows", &.{ "q", "keys", "tail", "weights", "offsets", "count", "tail_base" }, &.{"out"}, SCORE_ROWS, SCORE_HEADER ++ SCORE_ROWS_KEYS);
    const ov = try apply(k, &.{ index_q, pooled, tail, weights, offsets, count, first }, cfg, scope.s);
    defer _ = mlx.mlx_vector_array_free(ov);
    return kernelOutput(scope, ov, 0);
}

fn attentionChunk(scope: *Scope, state: *const State, q: Arr, selected: ?Arr, offset: usize, scale: f32, splits: c_int, headpack: bool, overlay: ?latent_overlay.View) !Arr {
    if (headpack) {
        var ops = @import("glm5_model.zig").Ops{ .s = scope.s };
        defer ops.deinit();
        if (try packed_nax.run(&ops, q, state.latentView(), selected.?, offset, state.processed, scale)) |out| {
            // Materialize the small result before releasing the gathered bank.
            try mlx.check(mlx.mlx_array_eval(out));
            return scope.own(try ops.result(out));
        }
    }
    const kind = @intFromBool(selected != null);
    scalar_calls[kind] += 1;
    if (scalar_calls[kind] == 1) @import("log.zig").info("[glm-attn] scalar {s} latent attention engaged\n", .{if (selected != null) "sparse" else "dense"});
    const sh = mlx.getShape(q);
    const rows = sh[0];
    const heads = sh[1];
    const dim = sh[2];
    const off = try scope.own(mlx.mlx_array_new_int(@intCast(offset)));
    const length = try scope.own(mlx.mlx_array_new_int(@intCast(state.processed)));
    const scaling = try scope.own(mlx.mlx_array_new_float(scale));
    const indices = selected orelse try scope.zeros(&.{1}, .int32);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ rows, heads, splits, dim }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ rows, heads, splits, 2 }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, heads, rows * splits));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try template(cfg, "H", heads);
    try template(cfg, "D", dim);
    try template(cfg, "SPLITS", splits);
    try template(cfg, "SELECTED", @intFromBool(selected != null));
    var partial: Arr = undefined;
    var stats: Arr = undefined;
    if (overlay) |view| {
        const result = try latent_overlay.partials(q, view, indices, off, length, scaling, splits, selected != null, scope.s);
        defer result.deinit();
        partial = try scope.own(try scope.result(result.partial));
        stats = try scope.own(try scope.result(result.stats));
    } else {
        const cache = state.latentView();
        const bits = cache.bits();
        const slot = &attention_kernels[latent_store.variant(bits)];
        const k = if (cache.quantized())
            try kernelWithHeader(slot, latent_store.kernelName("sushi_glm_latent_partial", bits), &.{ "q", "cache", "cache_scales", "cache_biases", "selected", "offset", "length", "scale" }, &.{ "partial", "stats" }, ATTENTION, latent_store.header(bits))
        else
            try kernelWithHeader(slot, latent_store.kernelName("sushi_glm_latent_partial", bits), &.{ "q", "cache", "selected", "offset", "length", "scale" }, &.{ "partial", "stats" }, ATTENTION, latent_store.header(bits));
        const ov = if (cache.quantized())
            try apply(k, &.{ q, cache.data, cache.scales, cache.biases, indices, off, length, scaling }, cfg, scope.s)
        else
            try apply(k, &.{ q, cache.data, indices, off, length, scaling }, cfg, scope.s);
        defer _ = mlx.mlx_vector_array_free(ov);
        partial = try kernelOutput(scope, ov, 0);
        stats = try kernelOutput(scope, ov, 1);
    }
    const mc = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(mc);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mc, sh.ptr, 3, mlx.mlx_array_dtype(q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mc, rows * heads * dim, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mc, 256, 1, 1));
    try template(mc, "ROWS", rows);
    try template(mc, "H", heads);
    try template(mc, "D", dim);
    try template(mc, "SPLITS", splits);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(mc, "OutT", mlx.mlx_array_dtype(q)));
    const mk = try kernel(&merge_kernel, "sushi_glm_latent_merge", &.{ "partial", "stats" }, &.{"out"}, MERGE);
    const mv = try apply(mk, &.{ partial, stats }, mc, scope.s);
    defer _ = mlx.mlx_vector_array_free(mv);
    return kernelOutput(scope, mv, 0);
}

pub fn attend(state: *const State, q: Arr, index_q: ?Arr, weights: ?Arr, offset: usize, scale: f32, s: mlx.mlx_stream) !Arr {
    return attendImpl(state, q, index_q, weights, offset, scale, null, s);
}
pub fn attendOverlay(state: *const State, q: Arr, index_q: ?Arr, weights: ?Arr, offset: usize, scale: f32, view: latent_overlay.View, s: mlx.mlx_stream) !Arr {
    try view.validate(q);
    if (state.processed != view.length() or offset + 1 != state.processed) return error.InvalidGlmOverlay;
    return attendImpl(state, q, index_q, weights, offset, scale, view, s);
}

const PackedTile = struct {
    scope: Scope,
    ops: Ops,
    out: Arr = nil,
    pending: bool = false,
    fn deinit(self: *PackedTile) void {
        // Error paths must also settle submitted work before releasing its bank.
        if (self.pending) _ = mlx.mlx_array_eval(self.out);
        self.ops.deinit();
        self.scope.deinit();
    }
};

fn packedTileRows(remaining: usize, max_rows: usize) usize {
    const cap = if (max_rows >= 32 and remaining >= 32) @as(usize, 32) else @min(max_rows, packed_nax.max_rows);
    return @min(remaining, cap);
}

test "GLM packed32 scheduling keeps every remainder in original selector geometry" {
    try std.testing.expectEqual(@as(usize, 32), packedTileRows(49, 32));
    try std.testing.expectEqual(@as(usize, 32), packedTileRows(32, 32));
    try std.testing.expectEqual(@as(usize, 16), packedTileRows(31, 32));
    try std.testing.expectEqual(@as(usize, 16), packedTileRows(17, 32));
    try std.testing.expectEqual(@as(usize, 1), packedTileRows(1, 32));
    try std.testing.expectEqual(@as(usize, 16), packedTileRows(17, 16));
    try std.testing.expectEqual(@as(usize, 5), packedTileRows(5, 16));
}

/// Settles `tile` and keeps its output; the graph it built is released only now.
fn settleTile(tile: *PackedTile, parts: mlx.mlx_vector_array, s: mlx.mlx_stream) !void {
    if (!tile.pending) return;
    try mlx.check(mlx.mlx_array_eval(tile.out));
    tile.pending = false;
    try mlx.check(mlx.mlx_vector_array_append_value(parts, tile.out));
    tile.deinit();
    tile.* = .{ .scope = .{ .s = s }, .ops = .{ .s = s } };
}

// Two tiles stay in flight: the older settles only after its successor is queued, so the GPU never
// waits for the host to build the next graph.
fn attendPackedTiles(state: *const State, q: Arr, iq: Arr, weights: Arr, offset: usize, scale: f32, max_rows: usize, s: mlx.mlx_stream) !Arr {
    const rows: usize = @intCast(mlx.getShape(q)[0]);
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    var tiles = [_]PackedTile{
        .{ .scope = .{ .s = s }, .ops = .{ .s = s } },
        .{ .scope = .{ .s = s }, .ops = .{ .s = s } },
    };
    defer for (&tiles) |*tile| tile.deinit();
    var start: usize = 0;
    var slot: usize = 0;
    while (start < rows) : (slot ^= 1) {
        const tile = &tiles[slot];
        try settleTile(tile, parts, s);
        const end = start + packedTileRows(rows - start, max_rows);
        const qc = try tile.scope.cut(q, @intCast(start), @intCast(end));
        const selected = try selectPackedChunk(&tile.scope, state, try tile.scope.cut(iq, @intCast(start), @intCast(end)), try tile.scope.cut(weights, @intCast(start), @intCast(end)), offset + start);
        tile.out = (try packed_nax.run(&tile.ops, qc, state.latentView(), selected, offset + start, state.processed, scale)) orelse
            try attentionChunk(&tile.scope, state, qc, selected, offset + start, scale, 1, false, null);
        const submit = mlx.mlx_vector_array_new_data(&.{tile.out}, 1);
        defer _ = mlx.mlx_vector_array_free(submit);
        tile.pending = true;
        try mlx.check(mlx.mlx_async_eval(submit));
        start = end;
    }
    for (0..tiles.len) |_| {
        try settleTile(&tiles[slot], parts, s);
        slot ^= 1;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, parts, 0, s));
    return out;
}
fn attendImpl(state: *const State, q: Arr, index_q: ?Arr, weights: ?Arr, offset: usize, scale: f32, overlay: ?latent_overlay.View, s: mlx.mlx_stream) !Arr {
    if (!mlx.streamIsGpu(s)) return error.GlmAttentionGpuRequired;
    if (q.ctx == null) return error.InvalidGlmAttentionShape;
    const sh = mlx.getShape(q);
    const cache = if (overlay) |view| view.storage() else state.latentView();
    if (sh.len != 3 or sh[0] <= 0 or sh[1] <= 0 or sh[2] <= 0 or !supported(mlx.mlx_array_dtype(q)) or
        !std.math.isFinite(scale) or scale <= 0 or cache.data.ctx == null or
        sh[2] != cache.width() or mlx.mlx_array_dtype(q) != cache.dtype() or
        offset > state.processed or @as(usize, @intCast(sh[0])) > state.processed - offset) return error.InvalidGlmAttentionShape;
    const sparse = offset + @as(usize, @intCast(sh[0])) > pool_size * (pool_budget + 1) - 1;
    if (sparse) {
        if (state.indexKeys().ctx == null) return error.InvalidGlmAttentionShape;
        const iq = index_q orelse return error.MissingGlmIndexQuery;
        const w = weights orelse return error.MissingGlmIndexQuery;
        const is = mlx.getShape(iq);
        if (is.len != 3 or is[0] != sh[0] or is[1] <= 0 or is[2] != mlx.getShape(state.indexKeys())[1] or
            !std.mem.eql(c_int, is[0..2], mlx.getShape(w)) or mlx.mlx_array_dtype(iq) != mlx.mlx_array_dtype(state.indexKeys()) or
            mlx.mlx_array_dtype(w) != mlx.mlx_array_dtype(iq)) return error.InvalidGlmAttentionShape;
    }
    const native = @import("glm5_attention_decode_batch.zig");
    if (native.enabled() and sh[0] <= 8 and native.supportedQuery(sh, mlx.mlx_array_dtype(q), s))
        return attendNativeDecode(state, q, index_q, weights, offset, scale, overlay, s);
    const splits: c_int = if (sh[0] <= 8) 8 else 1;
    const headpack = overlay == null and sparse and splits == 1 and sh[1] == 64 and sh[2] == 512 and
        mlx.mlx_array_dtype(q) == .bfloat16 and packed_nax.enabled();
    const per_row = try std.math.mul(usize, @intCast(sh[1]), try std.math.mul(usize, @intCast(splits), (@as(usize, @intCast(sh[2])) + 2) * 4));
    const pool_bytes = @max(@as(usize, 4), state.processed / 4 * 4);
    if (per_row > attention_scratch_bytes or (sparse and pool_bytes > score_scratch_bytes)) return error.GlmAttentionScratchBudget;
    const wide_chunk = headpack and sh[0] >= 32 and @import("glm5_model.zig").naxArms();
    const packed_rows = if (wide_chunk) packed_nax.wide_rows else packed_nax.tileRows();
    // Packed tiles select 16 rows at a time; their planes are billed by `selectionScratchBytes`.
    const score_rows = if (sparse and !headpack) score_scratch_bytes / pool_bytes else std.math.maxInt(usize);
    const max_rows = @max(@as(usize, 1), @min(@min(attention_scratch_bytes / per_row, score_rows), if (headpack) packed_rows else 128));
    if (headpack and (@as(usize, @intCast(sh[0])) > max_rows or (wide_chunk and max_rows == 32)))
        return attendPackedTiles(state, q, index_q.?, weights.?, offset, scale, max_rows, s);
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    var start: usize = 0;
    while (start < sh[0]) {
        const end = @min(@as(usize, @intCast(sh[0])), start + max_rows);
        var scope = Scope{ .s = s };
        defer scope.deinit();
        const qc = try scope.cut(q, @intCast(start), @intCast(end));
        const selected = if (sparse) try selectChunk(&scope, state, try scope.cut(index_q.?, @intCast(start), @intCast(end)), try scope.cut(weights.?, @intCast(start), @intCast(end)), offset + start) else null;
        const out = try attentionChunk(&scope, state, qc, selected, offset + start, scale, splits, headpack, overlay);
        // Settle bounded chunks before dropping their score/partial buffers.
        if (@as(usize, @intCast(sh[0])) > max_rows) try mlx.check(mlx.mlx_array_eval(out));
        try mlx.check(mlx.mlx_vector_array_append_value(parts, out));
        start = end;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, parts, 0, s));
    return out;
}

fn attendNativeDecode(state: *const State, q: Arr, iq: ?Arr, weights: ?Arr, offset: usize, scale: f32, view: ?latent_overlay.View, s: mlx.mlx_stream) !Arr {
    const native = @import("glm5_attention_decode_batch.zig");
    const rows: usize = @intCast(mlx.getShape(q)[0]);
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    var first: usize = 0;
    while (first < rows) {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const end = @min(rows, first + 3);
        const outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        for (first..end) |row| {
            const pos = offset + row;
            const query = try ops.slice(q, 0, @intCast(row), @intCast(row + 1));
            const row_iq = if (iq) |a| try ops.slice(a, 0, @intCast(row), @intCast(row + 1)) else nil;
            const row_weights = if (weights) |a| try ops.slice(a, 0, @intCast(row), @intCast(row + 1)) else nil;
            if (rows == 1 and view == null and native.rowSupported(query, state.latentView(), pos, s)) {
                const selection: native.RowSelection = if (try decodePools(state, row_iq, row_weights, pos, s)) |pools|
                    .{ .pools = try ops.own(pools) }
                else
                    .{ .ids = try ops.own(try decodeSelected(state, row_iq, row_weights, pos, s)) };
                const out = (try native.runRow(&ops, query, state.latentView(), pos, selection, scale)) orelse return error.GlmDecodeNativeUnsupported;
                try mlx.check(mlx.mlx_vector_array_append_value(parts, out));
                continue;
            }
            const indices = try ops.own(try decodeSelected(state, row_iq, row_weights, pos, s));
            const prefix = if (view) |v| v.storage() else state.latentView();
            const prefix_rows = if (view) |v| v.prefix_rows else pos;
            const tail = if (view) |v| v.tail else try ops.own(try state.latentView().dense(@intCast(pos), @intCast(pos + 1), s));
            const branch = native.Branch{ .offset = pos, .length = pos + 1, .path = native.pathOf(&.{ 0, 1, 2, 3 }) };
            const out = (try native.run(&ops, query, prefix, prefix_rows, tail, &.{branch}, indices, scale)) orelse return error.GlmDecodeNativeUnsupported;
            try mlx.check(mlx.mlx_vector_array_append_value(outputs, out));
            try mlx.check(mlx.mlx_vector_array_append_value(parts, out));
        }
        // Ordinary short chunks cannot retain more than three gathered B1 banks.
        if (rows > 3) try mlx.check(mlx.mlx_eval(outputs));
        first = end;
    }
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_concatenate_axis(&result, parts, 0, s));
    return result;
}

fn array(data: []const f32, shape: []const c_int) Arr {
    return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
}

test "GLM IndexPool append matches scalar pooling across irregular chunks" {
    const s = mlx.gpuStream();
    const n = 9;
    const d = 4;
    var keys: [n * d]f32 = undefined;
    var gates: [n * d]f32 = undefined;
    var bias: [4 * d]f32 = undefined;
    for (&keys, &gates, 0..) |*k, *g, i| {
        k.* = @as(f32, @floatFromInt(i)) / 16 - 1;
        g.* = @as(f32, @floatFromInt(i % 7)) / 8 - 0.5;
    }
    for (&bias, 0..) |*b, i| b.* = @as(f32, @floatFromInt(i % 5)) / 16;
    const ape = array(&bias, &.{ 4, d });
    defer _ = mlx.mlx_array_free(ape);
    var state = State.init();
    defer state.deinit();
    var offset: usize = 0;
    for ([_]usize{ 1, 3, 1, 4 }) |rows| {
        const k = array(keys[offset * d ..][0 .. rows * d], &.{ @intCast(rows), d });
        defer _ = mlx.mlx_array_free(k);
        const g = array(gates[offset * d ..][0 .. rows * d], &.{ @intCast(rows), d });
        defer _ = mlx.mlx_array_free(g);
        try std.testing.expectEqual(offset, try state.append(k, k, g, ape, s));
        offset += rows;
        try std.testing.expectEqual(offset, state.processed);
        try state.evaluate();
        if (offset >= 4) {
            const got = mlx.mlx_array_data_float32(state.pooled).?;
            for (0..offset / 4) |p| for (0..d) |c| {
                var sum: f64 = 0;
                var value: f64 = 0;
                for (0..4) |j| {
                    const e = @exp(@as(f64, gates[(p * 4 + j) * d + c] + bias[j * d + c]));
                    sum += e;
                    value += e * keys[(p * 4 + j) * d + c];
                }
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(value / sum)), got[p * d + c], 1e-6);
            };
        }
        try std.testing.expectEqual(@as(usize, offset % 4 * d), if (state.tail_keys.ctx != null) mlx.mlx_array_size(state.tail_keys) else 0);
    }
    state.reset();
    try std.testing.expectEqual(@as(usize, 0), state.processed);
    for (state.arrays()) |a| try std.testing.expect(a.ctx == null);
}

test "GLM latent attention matches scalar causal attention at pool boundaries" {
    const s = mlx.gpuStream();
    for ([_]usize{ 1, 3, 4, 5, 2048, 2049 }) |n| {
        const a = std.testing.allocator;
        const d = 4;
        const keys = try a.alloc(f32, n * d);
        defer a.free(keys);
        for (keys, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 19)) / 32 - 0.25;
        const latent = array(keys, &.{ @intCast(n), d });
        defer _ = mlx.mlx_array_free(latent);
        const bias: [16]f32 = @splat(0);
        const ape = array(&bias, &.{ 4, d });
        defer _ = mlx.mlx_array_free(ape);
        var state = State.init();
        defer state.deinit();
        _ = try state.append(latent, latent, latent, ape, s);
        const qv = [_]f32{ 0.5, -0.25, 0.75, 0.125 };
        const q = array(&qv, &.{ 1, 1, d });
        defer _ = mlx.mlx_array_free(q);
        const output = try attend(&state, q, null, null, n - 1, 1.0 / 16.0, s);
        defer _ = mlx.mlx_array_free(output);
        try mlx.check(mlx.mlx_array_eval(output));
        const got = mlx.mlx_array_data_float32(output).?;
        var denominator: f64 = 0;
        var want: [d]f64 = @splat(0);
        for (0..n) |i| {
            var dot: f64 = 0;
            for (qv, keys[i * d ..][0..d]) |x, k| dot += @as(f64, x) * k / 16;
            const e = @exp(dot);
            denominator += e;
            for (&want, keys[i * d ..][0..d]) |*v, k| v.* += e * k;
        }
        for (want, 0..) |v, c| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(v / denominator)), got[c], 2e-6);
    }
}

test "GLM sparse pool selection excludes future pools and appends tail exactly once" {
    const s = mlx.gpuStream();
    const a = std.testing.allocator;
    const n = 2055;
    const d = 4;
    const heads = 2;
    const rows = 8;
    const offset = n - rows;
    const lat = try a.alloc(f32, n * d);
    defer a.free(lat);
    const key = try a.alloc(f32, n * d);
    defer a.free(key);
    const zeros = try a.alloc(f32, n * d);
    defer a.free(zeros);
    @memset(zeros, 0);
    for (lat, key, 0..) |*v, *k, i| {
        v.* = @as(f32, @floatFromInt(i % 23)) / 32 - 0.25;
        k.* = if (i % d == 0) @as(f32, @floatFromInt(i / d)) / 2048 else 0;
    }
    const la = array(lat, &.{ n, d });
    defer _ = mlx.mlx_array_free(la);
    const ka = array(key, &.{ n, d });
    defer _ = mlx.mlx_array_free(ka);
    const ga = array(zeros, &.{ n, d });
    defer _ = mlx.mlx_array_free(ga);
    const av: [16]f32 = @splat(0);
    const ape = array(&av, &.{ 4, d });
    defer _ = mlx.mlx_array_free(ape);
    var state = State.init();
    defer state.deinit();
    _ = try state.append(la, ka, ga, ape, s);
    const query: [rows * heads * d]f32 = @splat(0.125);
    const q = array(&query, &.{ rows, heads, d });
    defer _ = mlx.mlx_array_free(q);
    const weight: [rows * heads]f32 = @splat(0.5);
    const w = array(&weight, &.{ rows, heads });
    defer _ = mlx.mlx_array_free(w);
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const selected = try selectChunk(&scope, &state, q, w, offset);
    try mlx.check(mlx.mlx_array_eval(selected));
    const ids = mlx.mlx_array_data_int32(selected).?;
    for (0..rows) |r| {
        const pos = offset + r;
        const completed = (pos + 1) / 4;
        const first = if (completed > 512) (completed - 512) * 4 else 0;
        var seen: [n]bool = @splat(false);
        var count: usize = 0;
        for (ids[r * selected_width ..][0..selected_width]) |id| {
            if (id < 0) continue;
            try std.testing.expect(id >= first and id <= pos);
            try std.testing.expect(!seen[@intCast(id)]);
            seen[@intCast(id)] = true;
            count += 1;
        }
        try std.testing.expectEqual(pos + 1 - first, count);
        for (seen[first .. pos + 1]) |present| try std.testing.expect(present);
    }
    const y = try attend(&state, q, q, w, offset, 1.0 / 16.0, s);
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));
    const got = mlx.mlx_array_data_float32(y).?;
    for (0..rows) |r| {
        const pos = offset + r;
        const completed = (pos + 1) / 4;
        const first = if (completed > 512) (completed - 512) * 4 else 0;
        var sum: f64 = 0;
        var wanted: [d]f64 = @splat(0);
        for (first..pos + 1) |token| {
            var dot: f64 = 0;
            for (lat[token * d ..][0..d]) |v| dot += @as(f64, v) * 0.125 / 16;
            const p = @exp(dot);
            sum += p;
            for (&wanted, lat[token * d ..][0..d]) |*v, k| v.* += p * k;
        }
        for (0..heads) |h| for (wanted, 0..) |v, c| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(v / sum)), got[(r * heads + h) * d + c], 2e-6);
    }
}

test "GLM latent BF16 attention covers width512 and prefill chunk continuation" {
    const s = mlx.gpuStream();
    const n = 5;
    const d = 512;
    const h = 2;
    var data: [n * d]f32 = undefined;
    var queries: [n * h * d]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) / 32 - 0.125;
    for (&queries, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) / 32 - 0.0625;
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const lat = try scope.cast(try scope.own(array(&data, &.{ n, d })), .bfloat16);
    const q = try scope.cast(try scope.own(array(&queries, &.{ n, h, d })), .bfloat16);
    const key = try scope.zeros(&.{ n, 4 }, .bfloat16);
    const ape = try scope.zeros(&.{ 4, 4 }, .bfloat16);
    var state = State.init();
    defer state.deinit();
    _ = try state.append(lat, key, key, ape, s);
    const whole = try scope.own(try attend(&state, q, null, null, 0, 1.0 / 16.0, s));
    const full = try scope.cast(whole, .float32);
    try mlx.check(mlx.mlx_array_eval(full));
    var streamed = State.init();
    defer streamed.deinit();
    var start: c_int = 0;
    for ([_]c_int{ 1, 3, 1 }) |len| {
        const end = start + len;
        _ = try streamed.append(try scope.cut(lat, start, end), try scope.cut(key, start, end), try scope.cut(key, start, end), ape, s);
        const out = try scope.own(try attend(&streamed, try scope.cut(q, start, end), null, null, @intCast(start), 1.0 / 16.0, s));
        const got = try scope.cast(out, .float32);
        try mlx.check(mlx.mlx_array_eval(got));
        try std.testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(full).?[@as(usize, @intCast(start)) * h * d ..][0 .. @as(usize, @intCast(len)) * h * d], mlx.mlx_array_data_float32(got).?[0 .. @as(usize, @intCast(len)) * h * d]);
        start = end;
    }
}

fn bf16Round(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    return @bitCast((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
}

test "GLM index score matches rounded per-head scalar math with negative weights" {
    const s = mlx.gpuStream();
    const rows = 2;
    const heads = 32;
    const dim = 128;
    const pools = 7;
    var kv: [pools * dim]f32 = undefined;
    var query: [rows * heads * dim]f32 = undefined;
    var weight: [rows * heads]f32 = undefined;
    for (&kv, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 7)) - 3) / 8;
    for (&query, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 11)) - 5) / 16;
    for (&weight, 0..) |*v, i| v.* = if (i % 2 == 0) 0.25 else -0.5;
    inline for (.{ mlx.mlx_dtype.float32, mlx.mlx_dtype.bfloat16 }) |dtype| {
        var scope = Scope{ .s = s };
        defer scope.deinit();
        const k = try scope.cast(try scope.own(array(&kv, &.{ pools, dim })), dtype);
        const q = try scope.cast(try scope.own(array(&query, &.{ rows, heads, dim })), dtype);
        const w = try scope.cast(try scope.own(array(&weight, &.{ rows, heads })), dtype);
        const state = State{ .pooled = k, .processed = pools * 4 };
        const scores = try indexScores(&scope, &state, q, w, pools * 4 - 2);
        try mlx.check(mlx.mlx_array_eval(scores));
        const got = mlx.mlx_array_data_float32(scores).?;
        for (0..rows) |r| for (0..pools) |p| {
            if ((p + 1) * 4 > pools * 4 - 1 + r) {
                try std.testing.expect(got[r * pools + p] == -std.math.inf(f32));
                continue;
            }
            var total: f32 = 0;
            for (0..heads) |h| {
                var dot: f32 = 0;
                for (0..dim) |d| dot += query[(r * heads + h) * dim + d] * kv[p * dim + d];
                if (dtype == .bfloat16) dot = bf16Round(dot);
                var product = @max(dot, 0) * weight[r * heads + h];
                if (dtype == .bfloat16) product = bf16Round(product);
                total += product;
            }
            if (dtype == .bfloat16) total = bf16Round(total);
            try std.testing.expectEqual(total, got[r * pools + p]);
        };
    }
}

/// The scorer the tree scorer replaced: one 32-lane group per (row, pool) and a `simd_sum` per head.
const SCORE_SIMD_SUM: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint lane=thread_position_in_threadgroup.x;
    \\const uint p=threadgroup_position_in_grid.y;
    \\const uint row=threadgroup_position_in_grid.z;
    \\const uint pools=uint(count);
    \\if ((p+1u)*4u > uint(offset)+row+1u) { if(lane==0) out[row*pools+p]=-INFINITY; return; }
    \\float total=0.0f;
    \\for(uint h=0;h<uint(J);++h) {
    \\  float dot=0.0f;
    \\  for(uint d=lane;d<uint(I);d+=32u) dot+=float(q[(row*uint(J)+h)*uint(I)+d])*float(keys[p*uint(I)+d]);
    \\  dot=simd_sum(dot);
    \\  const float rounded=float(InT(dot));
    \\  total+=float(InT(max(rounded,0.0f)*float(weights[row*uint(J)+h])));
    \\}
    \\if(lane==0) out[row*pools+p]=float(InT(total));
;
var simd_sum_score_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn simdSumScores(scope: *Scope, pooled: Arr, index_q: Arr, weights: Arr, offset: usize, pools: c_int) !Arr {
    const sh = mlx.getShape(index_q);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ sh[0], pools }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, pools, sh[0]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try template(cfg, "J", sh[1]);
    try template(cfg, "I", sh[2]);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", mlx.mlx_array_dtype(index_q)));
    const off = try scope.own(mlx.mlx_array_new_int(@intCast(offset)));
    const count = try scope.own(mlx.mlx_array_new_int(pools));
    const k = try kernel(&simd_sum_score_kernel, "sushi_glm_index_scores_simd_sum", &.{ "q", "keys", "weights", "offset", "count" }, &.{"out"}, SCORE_SIMD_SUM);
    const ov = try apply(k, &.{ index_q, pooled, weights, off, count }, cfg, scope.s);
    defer _ = mlx.mlx_vector_array_free(ov);
    return kernelOutput(scope, ov, 0);
}

test "GLM tree index scorer equals the SIMD-sum scorer bit for bit at real magnitudes and long key counts" {
    const s = mlx.gpuStream();
    const Case = struct { rows: c_int, pools: c_int, back: c_int, heads: c_int = 32, dim: c_int = 128, dtype: mlx.mlx_dtype = .bfloat16 };
    // Causal edges land inside and at the end of eight-pool blocks; 17 rows leave a partial row group.
    for ([_]Case{
        .{ .rows = 1, .pools = 513, .back = 1 },
        .{ .rows = 1, .pools = 32771, .back = 2 },
        .{ .rows = 3, .pools = 1027, .back = 9 },
        .{ .rows = 16, .pools = 4099, .back = 16 },
        .{ .rows = 17, .pools = 2053, .back = 40 },
        .{ .rows = 16, .pools = 32768, .back = 70 },
        .{ .rows = 5, .pools = 37, .back = 5, .heads = 2, .dim = 8, .dtype = .float32 },
        .{ .rows = 2, .pools = 300, .back = 3, .dtype = .float32 },
    }, 0..) |case, i| {
        var scope = Scope{ .s = s };
        defer scope.deinit();
        const seed: u64 = 100 + 4 * i;
        // Index queries reach a few units, pooled keys are layer-normed with a few large pools,
        // and head weights carry the 1/sqrt(4096) scale; signed zeros come from zero weights.
        const outliers = try scope.own(mlx.mlx_array_new_float(16));
        const spikes = try scope.slot();
        try mlx.check(mlx.mlx_multiply(spikes, try normal(&scope, &.{ case.pools, case.dim }, seed, 1), outliers, s));
        const pick = try scope.slot();
        try mlx.check(mlx.mlx_greater(pick, try normal(&scope, &.{ case.pools, 1 }, seed + 1, 1), try scope.own(mlx.mlx_array_new_float(2)), s));
        const keys = try scope.slot();
        try mlx.check(mlx.mlx_where(keys, pick.*, spikes.*, try normal(&scope, &.{ case.pools, case.dim }, seed + 2, 1), s));
        const raw_w = try normal(&scope, &.{ case.rows, case.heads }, seed + 3, 0.1);
        const zero_w = try scope.slot();
        try mlx.check(mlx.mlx_greater(zero_w, raw_w, try scope.own(mlx.mlx_array_new_float(0.15)), s));
        const w = try scope.slot();
        try mlx.check(mlx.mlx_where(w, zero_w.*, try scope.zeros(&.{}, .bfloat16), raw_w, s));
        const q = try scope.cast(try normal(&scope, &.{ case.rows, case.heads, case.dim }, seed + 4, 3), case.dtype);
        const k = try scope.cast(keys.*, case.dtype);
        const weights = try scope.cast(w.*, case.dtype);
        const offset: usize = @intCast(case.pools * 4 - case.rows - case.back);
        const want = try simdSumScores(&scope, k, q, weights, offset, case.pools);
        const got = try treeScores(&scope, k, q, weights, offset, case.pools);
        try expectSameBits(want, got);
    }
    // Decode and verify rows route to the tree scorer.
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const calls = tree_score_calls;
    const pooled = try normal(&scope, &.{ 600, 128 }, 7, 1);
    const state = State{ .pooled = pooled, .processed = 2400 };
    _ = try indexScores(&scope, &state, try normal(&scope, &.{ 1, 32, 128 }, 8, 3), try normal(&scope, &.{ 1, 32 }, 9, 0.1), 2399);
    try std.testing.expectEqual(calls + 1, tree_score_calls);
}

test "GLM verifier rows select exactly what each row selects alone, at every tail length and pool boundary" {
    const s = mlx.gpuStream();
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const prefix_pools = 1100;
    const pooled = try normal(&scope, &.{ prefix_pools, 128 }, 31, 4);
    // Pool count and tail length per row: rows 0, 1 and 6 share a group, as do 2 and 3, and 4 and 5; row 7 is dense.
    const cases = [_]struct { tail: usize, processed: usize }{
        .{ .tail = 0, .processed = 4400 }, .{ .tail = 0, .processed = 4403 }, .{ .tail = 1, .processed = 4404 }, .{ .tail = 1, .processed = 4406 },
        .{ .tail = 2, .processed = 4408 }, .{ .tail = 2, .processed = 4409 }, .{ .tail = 0, .processed = 4402 }, .{ .tail = 0, .processed = 2000 },
    };
    var states: [cases.len]State = undefined;
    var rows: [cases.len]SelectRow = undefined;
    for (cases, 0..) |case, i| {
        const seed: u64 = 500 + 10 * i;
        states[i] = .{ .pooled = pooled, .processed = case.processed };
        if (case.tail > 0) states[i].pool_tail = try normal(&scope, &.{ @intCast(case.tail), 128 }, seed, 4);
        rows[i] = .{ .state = &states[i], .index_q = try normal(&scope, &.{ 1, 32, 128 }, seed + 1, 3), .weights = try normal(&scope, &.{ 1, 32 }, seed + 2, 0.1), .offset = case.processed - 1 };
        if (case.processed < 2100) states[i].pooled = try scope.cut(pooled, 0, @intCast(case.processed / 4));
    }
    var grouped: [cases.len]Arr = undefined;
    const before = group_selects;
    try decodeSelectedRows(&rows, &grouped, s);
    try std.testing.expectEqual(before + 3, group_selects);
    defer for (grouped) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (rows, 0..) |row, i| {
        const alone = try scope.own(try decodeSelected(row.state, row.index_q, row.weights, row.offset, s));
        try mlx.check(mlx.mlx_array_eval(alone));
        try mlx.check(mlx.mlx_array_eval(grouped[i]));
        try std.testing.expectEqualSlices(c_int, mlx.getShape(alone), mlx.getShape(grouped[i]));
        const n = mlx.mlx_array_size(alone);
        try std.testing.expectEqualSlices(i32, mlx.mlx_array_data_int32(alone).?[0..n], mlx.mlx_array_data_int32(grouped[i]).?[0..n]);
    }
}

test "GLM attention rejects invalid append without advancing request state" {
    const s = mlx.gpuStream();
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const lat = try scope.zeros(&.{ 3, 4 }, .bfloat16);
    const key = try scope.zeros(&.{ 3, 4 }, .bfloat16);
    const ape = try scope.zeros(&.{ 4, 4 }, .bfloat16);
    var state = State.init();
    defer state.deinit();
    _ = try state.append(lat, key, key, ape, s);
    const old = state.arrays();
    const bad = try scope.zeros(&.{ 3, 5 }, .bfloat16);
    try std.testing.expectError(error.InvalidGlmAttentionShape, state.append(bad, key, key, ape, s));
    try std.testing.expectEqual(@as(usize, 3), state.processed);
    for (old, state.arrays()) |before, after| try std.testing.expect(before.ctx == after.ctx);
    const q = try scope.zeros(&.{ 1, 1, 4 }, .bfloat16);
    try std.testing.expectError(error.InvalidGlmAttentionShape, attend(&state, q, null, null, 3, 1.0 / 16.0, s));
    var other = State.init();
    defer other.deinit();
    try std.testing.expectEqual(@as(usize, 0), other.processed);
}

test "GLM native decode falls back for unsupported tiny attention bits" {
    const native = @import("glm5_attention_decode_batch.zig");
    const s = mlx.gpuStream();
    const latent = array(&.{ 0.25, -0.5, 1, 2 }, &.{ 2, 2 });
    defer _ = mlx.mlx_array_free(latent);
    const keys = array(&.{ 0, 0, 0, 0 }, &.{ 2, 2 });
    defer _ = mlx.mlx_array_free(keys);
    const ape = array(&.{ 0, 0, 0, 0, 0, 0, 0, 0 }, &.{ 4, 2 });
    defer _ = mlx.mlx_array_free(ape);
    const q = array(&.{ 0.5, 1, -0.25, 0.75 }, &.{ 1, 2, 2 });
    defer _ = mlx.mlx_array_free(q);
    var state = State.init();
    defer state.deinit();
    _ = try state.append(latent, keys, keys, ape, s);
    const model = @import("glm5_model.zig");
    const reference = blk: {
        model.reference_numerics = true;
        defer model.reference_numerics = false;
        break :blk try attend(&state, q, null, null, 1, 0.5, s);
    };
    defer _ = mlx.mlx_array_free(reference);
    native.resetCalls();
    const actual = try attend(&state, q, null, null, 1, 0.5, s);
    defer _ = mlx.mlx_array_free(actual);
    const evals = mlx.mlx_vector_array_new_data(&.{ reference, actual }, 2);
    defer _ = mlx.mlx_vector_array_free(evals);
    try mlx.check(mlx.mlx_eval(evals));
    const count = mlx.mlx_array_size(actual);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(reference).?[0..count]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(actual).?[0..count]));
    try std.testing.expectEqual(@as(usize, 0), native.b1Calls());
    try std.testing.expectEqual(@as(usize, 0), native.b3Calls());
}

test "GLM ranked top-512 equals argpartition's stable order on ties, -inf and causal edges" {
    const s = mlx.gpuStream();
    // `future` pools at the end are -inf: 600/200 cuts inside the -inf ties, 1000 inside the zeros.
    const Case = struct { rows: c_int, pools: c_int, bits: u8, future: usize = 1 };
    for ([_]Case{
        .{ .rows = 2, .pools = 513, .bits = 16 },
        .{ .rows = 2, .pools = 600, .bits = 16, .future = 200 },
        .{ .rows = 3, .pools = 1000, .bits = 16 },
        .{ .rows = 3, .pools = 2049, .bits = 16 },
        .{ .rows = 16, .pools = 8192, .bits = 16 },
        .{ .rows = 16, .pools = 32771, .bits = 16 },
        .{ .rows = 4, .pools = 131072, .bits = 16 },
        .{ .rows = 5, .pools = 3001, .bits = 32 },
    }, 0..) |case, i| {
        var scope = Scope{ .s = s };
        defer scope.deinit();
        const seed: u64 = 200 + 3 * i;
        // BF16-valued scores with a block of exact zeros (ties), a NaN pool and future pools at -inf.
        const raw = try scope.cast(try normal(&scope, &.{ case.rows, case.pools }, seed, 0.5), .float32);
        var values = raw;
        if (case.bits == 32) {
            const fine = try scope.slot();
            try mlx.check(mlx.mlx_add(fine, raw, try scope.cast(try normal(&scope, &.{ case.rows, case.pools }, seed + 1, 0.001), .float32), s));
            values = fine.*;
        }
        const low = try scope.slot();
        try mlx.check(mlx.mlx_less(low, values, try scope.own(mlx.mlx_array_new_float(-0.2)), s));
        const tied = try scope.slot();
        try mlx.check(mlx.mlx_where(tied, low.*, try scope.zeros(&.{}, .float32), values, s));
        var column: [131072]f32 = undefined;
        const n: usize = @intCast(case.pools);
        for (column[0..n], 0..) |*v, p| v.* = if (p + case.future + i * 3 >= n) -std.math.inf(f32) else if (p == n / 2) std.math.nan(f32) else 0;
        const mask = try scope.own(mlx.mlx_array_new_data(&column, &.{ 1, case.pools }, 2, .float32));
        const scores = try scope.slot();
        try mlx.check(mlx.mlx_add(scores, tied.*, mask, s));
        const want = try partitionTop(&scope, scores.*, case.rows);
        const got = try rankTop(&scope, scores.*, case.bits);
        try expectSameBits(want, got);
    }
    // Prefill tiles rank; a decode row keeps the partition.
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const state = State{ .pooled = try normal(&scope, &.{ 600, 128 }, 230, 1), .processed = 2400 };
    const calls = rank_top_calls;
    _ = try selectChunk(&scope, &state, try normal(&scope, &.{ 16, 32, 128 }, 231, 3), try normal(&scope, &.{ 16, 32 }, 232, 0.1), 2384);
    var node = state;
    node.processed = 2400;
    _ = try selectChunk(&scope, &node, try normal(&scope, &.{ 1, 32, 128 }, 233, 3), try normal(&scope, &.{ 1, 32 }, 234, 0.1), 2399);
    try std.testing.expectEqual(calls + 1, rank_top_calls);
}

test "GLM packed prefill keeps 32-row tiles at long history and equals 16-row tiles bit for bit" {
    if (!@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var scope = Scope{ .s = s };
    defer scope.deinit();
    // Past 65536 rows the 2 MiB score plane used to cap a tile below 32 rows.
    const n = 70000;
    var twins = try kvTwins(latent_store.kv8_bits, &scope, try normal(&scope, &.{ n, 512 }, 40, 1), try normal(&scope, &.{ n, 128 }, 41, 1), &.{ 32768, 32768, n - 65536 });
    defer for (&twins) |*st| st.deinit();
    const rows = 64;
    const offset: usize = n - rows;
    const q = try normal(&scope, &.{ rows, 64, 512 }, 42, 0.05);
    const iq = try normal(&scope, &.{ rows, 32, 128 }, 43, 1);
    const w = try normal(&scope, &.{ rows, 32 }, 44, 0.1);
    const narrow = try scope.own(try attendPackedTiles(&twins[0], q, iq, w, offset, 1.0 / 16.0, packed_nax.max_rows, s));
    packed_nax.resetDispatchCount();
    const wide = try scope.own(try attend(&twins[0], q, iq, w, offset, 1.0 / 16.0, s));
    try mlx.check(mlx.mlx_array_eval(wide));
    try std.testing.expectEqual(@as(usize, rows / packed_nax.wide_rows), packed_nax.dispatchCount());
    try expectSameBits(narrow, wide);
    try std.testing.expectEqual(@as(usize, 64 * 131072), try selectionScratchBytes(131072));
}

test "GLM a 32-row selection equals two 16-row selections at tree and NAX index scoring" {
    if (!@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const n = 32768;
    var twins = try kvTwins(latent_store.kv8_bits, &scope, try normal(&scope, &.{ n, 512 }, 60, 1), try normal(&scope, &.{ n, 128 }, 61, 1), &.{ 16384, 16384 });
    defer for (&twins) |*st| st.deinit();
    const iq = try normal(&scope, &.{ n, 32, 128 }, 62, 1);
    const w = try normal(&scope, &.{ n, 32 }, 63, 0.1);
    const nax_calls = @import("glm5_indexpool_nax.zig").dispatchCount();
    // History 4160 (tree scorer), 14400 (NAX window), 32768 (NAX at its widest).
    for ([_]usize{ 4160, 14400, n }) |history| {
        var round = Scope{ .s = s };
        defer round.deinit();
        var state = twins[0];
        state.processed = history;
        const first = history - 64;
        const rows: c_int = 32;
        const cut_q = try round.cut(iq, @intCast(first), @intCast(first + 32));
        const cut_w = try round.cut(w, @intCast(first), @intCast(first + 32));
        const whole = try selectPackedChunk(&round, &state, cut_q, cut_w, first);
        const halves = try selectPackedHalves(&round, &state, cut_q, cut_w, first);
        try std.testing.expectEqual(rows, mlx.getShape(whole)[0]);
        try expectSameBits(halves, whole);
        try expectSameBits(try indexScores(&round, &state, cut_q, cut_w, first), try round.join(
            try indexScores(&round, &state, try round.cut(cut_q, 0, 16), try round.cut(cut_w, 0, 16), first),
            try indexScores(&round, &state, try round.cut(cut_q, 16, 32), try round.cut(cut_w, 16, 32), first + 16),
        ));
    }
    try std.testing.expect(@import("glm5_indexpool_nax.zig").dispatchCount() > nax_calls);
}

/// One tile at a time, each settled before the next is built.
fn attendPackedSerial(state: *const State, q: Arr, iq: Arr, weights: Arr, offset: usize, max_rows: usize, s: mlx.mlx_stream) !Arr {
    const rows: usize = @intCast(mlx.getShape(q)[0]);
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    var start: usize = 0;
    while (start < rows) {
        var scope = Scope{ .s = s };
        defer scope.deinit();
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const end = start + packedTileRows(rows - start, max_rows);
        const selected = try selectPackedChunk(&scope, state, try scope.cut(iq, @intCast(start), @intCast(end)), try scope.cut(weights, @intCast(start), @intCast(end)), offset + start);
        const out = (try packed_nax.run(&ops, try scope.cut(q, @intCast(start), @intCast(end)), state.latentView(), selected, offset + start, state.processed, 1.0 / 16.0)) orelse return error.ExpectedPackedNax;
        try mlx.check(mlx.mlx_array_eval(out));
        try mlx.check(mlx.mlx_vector_array_append_value(parts, out));
        start = end;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, parts, 0, s));
    return out;
}

test "GLM in-flight packed tiles equal serial tiles bit for bit at every tile count and remainder" {
    if (!@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const n = 16384;
    var twins = try kvTwins(latent_store.kv8_bits, &scope, try normal(&scope, &.{ n, 512 }, 50, 1), try normal(&scope, &.{ n, 128 }, 51, 1), &.{ 8192, 8192 });
    defer for (&twins) |*st| st.deinit();
    for ([_]c_int{ 16, 32, 48, 80, 96, 112 }) |rows| for ([_]usize{ packed_nax.max_rows, packed_nax.wide_rows }) |width| {
        var round = Scope{ .s = s };
        defer round.deinit();
        const offset: usize = n - @as(usize, @intCast(rows));
        const q = try normal(&round, &.{ rows, 64, 512 }, 52, 0.05);
        const iq = try normal(&round, &.{ rows, 32, 128 }, 53, 1);
        const w = try normal(&round, &.{ rows, 32 }, 54, 0.1);
        const serial = try round.own(try attendPackedSerial(&twins[0], q, iq, w, offset, width, s));
        packed_nax.resetDispatchCount();
        const tiled = try round.own(try attendPackedTiles(&twins[0], q, iq, w, offset, 1.0 / 16.0, width, s));
        try mlx.check(mlx.mlx_array_eval(tiled));
        try std.testing.expect(packed_nax.dispatchCount() >= @as(usize, @intCast(rows)) / width);
        try expectSameBits(serial, tiled);
    };
}

test "GLM packed cadence bills a second tile only beyond one wide tile" {
    const packed_attention = @import("glm5_attention_nax_packed.zig");
    try std.testing.expectEqual(@as(usize, 0), try packedCadenceTransientBudget(32, 2));
    try std.testing.expectEqual(packed_attention.scratch_limit, try packedCadenceTransientBudget(33, 1));
    try std.testing.expectEqual(packed_attention.scratch_limit * 2, try packedCadenceTransientBudget(2048, 2));
    try std.testing.expectError(error.Overflow, packedCadenceTransientBudget(2048, std.math.maxInt(usize)));
    const model = @import("glm5_model.zig");
    model.reference_numerics = true;
    defer model.reference_numerics = false;
    try std.testing.expectEqual(@as(usize, 0), try packedCadenceTransientBudget(2048, 2));
}

fn normal(scope: *Scope, shape: []const c_int, seed: u64, deviation: f32) !Arr {
    const key = try scope.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    const out = try scope.slot();
    try mlx.check(mlx.mlx_random_normal(out, shape.ptr, shape.len, .bfloat16, 0, deviation, key.*, scope.s));
    return out.*;
}

/// A quantized state and its BF16 twin fed the round trip of the same rows, in the same chunks.
fn kvTwins(bits: u8, scope: *Scope, latent: Arr, keys: Arr, chunks: []const c_int) ![2]State {
    const ape = try scope.zeros(&.{ 4, mlx.getShape(keys)[1] }, .bfloat16);
    var states = [2]State{ .{ .latent_bits = bits }, .{} };
    errdefer for (&states) |*st| st.deinit();
    var start: c_int = 0;
    for (chunks) |len| {
        const rows = try scope.cut(latent, start, start + len);
        const k = try scope.cut(keys, start, start + len);
        _ = try states[0].append(rows, k, k, ape, scope.s);
        _ = try states[1].append(try scope.own(try latent_store.readable(rows, bits, scope.s)), k, k, ape, scope.s);
        start += len;
    }
    return states;
}

pub fn expectSameBits(a: Arr, b: Arr) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const x = try ops.contiguous(a);
    const y = try ops.contiguous(b);
    try mlx.check(mlx.mlx_array_eval(x));
    try mlx.check(mlx.mlx_array_eval(y));
    const n = mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x);
    const left: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint8(x) orelse return error.MlxArrayDataNull);
    const right: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint8(y) orelse return error.MlxArrayDataNull);
    try std.testing.expectEqualSlices(u8, left[0..n], right[0..n]);
}

test "GLM kv8 and kv6 latent append quantizes each row once across prefill chunks and decode rows" {
    for ([_]u8{ latent_store.kv8_bits, 6 }) |bits| try appendOnceCase(bits);
}

fn appendOnceCase(bits: u8) !void {
    const s = mlx.gpuStream();
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const n = 600;
    const latent = try scope.own(try latent_store.randomRows(n, 512, 3, s));
    var twins = try kvTwins(bits, &scope, latent, try normal(&scope, &.{ n, 8 }, 4, 1), &.{ 1, 255, 1, 300, 3, 40 });
    defer for (&twins) |*st| st.deinit();
    const kv8 = twins[0].latentView();
    try std.testing.expect(kv8.quantized() and kv8.rowMajor());
    try std.testing.expectEqual(mlx.getShape(twins[1].latent)[0], kv8.rows());
    try expectSameBits(try scope.own(try kv8.dense(0, n, s)), try scope.cut(twins[1].latent, 0, n));
    var whole = try latent_store.quantize(latent, bits, s);
    defer whole.deinit();
    for ([_]Arr{ whole.q, whole.scales, whole.biases }, [_]Arr{ kv8.data, kv8.scales, kv8.biases }) |want, got| try expectSameBits(want, try scope.cut(got, 0, n));
    twins[0].reset();
    try std.testing.expectEqual(bits, twins[0].latent_bits);
    for (twins[0].arrays()) |a| try std.testing.expect(a.ctx == null);
}

test "GLM kv8 and kv6 scalar latent attention matches BF16 attention over the round-tripped rows" {
    for ([_]u8{ latent_store.kv8_bits, 6 }) |bits| try scalarAttentionCase(bits);
}

fn scalarAttentionCase(bits: u8) !void {
    const s = mlx.gpuStream();
    for ([_]c_int{ 40, 2055 }) |n| {
        var scope = Scope{ .s = s };
        defer scope.deinit();
        var twins = try kvTwins(bits, &scope, try normal(&scope, &.{ n, 64 }, 5, 1), try normal(&scope, &.{ n, 8 }, 6, 1), &.{ n - 8, 8 });
        defer for (&twins) |*st| st.deinit();
        for ([_]c_int{ 1, 8, 12 }) |rows| {
            const offset: usize = @intCast(n - rows);
            const q = try normal(&scope, &.{ rows, 2, 64 }, 7, 0.5);
            const iq = try normal(&scope, &.{ rows, 2, 8 }, 8, 1);
            const w = try normal(&scope, &.{ rows, 2 }, 9, 1);
            const got = try scope.own(try attend(&twins[0], q, iq, w, offset, 1.0 / 16.0, s));
            const want = try scope.own(try attend(&twins[1], q, iq, w, offset, 1.0 / 16.0, s));
            try expectSameBits(want, got);
        }
    }
}

test "GLM kv8 and kv6 native decode and packed prefill attention match BF16 over the round-tripped rows" {
    for ([_]u8{ latent_store.kv8_bits, 6 }) |bits| try nativeDecodeCase(bits);
}

fn nativeDecodeCase(bits: u8) !void {
    const native = @import("glm5_attention_decode_batch.zig");
    const s = mlx.gpuStream();
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const n = 2100;
    var twins = try kvTwins(bits, &scope, try normal(&scope, &.{ n, 512 }, 10, 1), try normal(&scope, &.{ n, 8 }, 11, 1), &.{ 2048, n - 2048 });
    defer for (&twins) |*st| st.deinit();
    native.resetCalls();
    packed_nax.resetDispatchCount();
    for ([_]c_int{ 1, 3, 16, 32 }) |rows| {
        const offset: usize = @intCast(n - rows);
        const q = try normal(&scope, &.{ rows, 64, 512 }, 12, 0.05);
        const iq = try normal(&scope, &.{ rows, 2, 8 }, 13, 1);
        const w = try normal(&scope, &.{ rows, 2 }, 14, 1);
        const got = try scope.own(try attend(&twins[0], q, iq, w, offset, 1.0 / 16.0, s));
        const want = try scope.own(try attend(&twins[1], q, iq, w, offset, 1.0 / 16.0, s));
        try expectSameBits(want, got);
    }
    try std.testing.expectEqual(@as(usize, 2 * (1 + 3)), native.b1Calls());
    try std.testing.expectEqual(@as(usize, if (@import("glm5_model.zig").naxArms()) 2 * 2 else 2 * (2 + 4)), packed_nax.dispatchCount());
}

test "GLM serial decode row equals the B1 tail-tape path bit for bit on kv8, kv6 and BF16 latents" {
    for ([_]u8{ latent_store.kv8_bits, 6 }) |bits| try serialRowCase(bits);
}

fn serialRowCase(bits: u8) !void {
    if (!@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    const native = @import("glm5_attention_decode_batch.zig");
    const s = mlx.gpuStream();
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const n = 9000;
    var twins = try kvTwins(bits, &scope, try normal(&scope, &.{ n, 512 }, 70, 1), try normal(&scope, &.{ n, 128 }, 71, 1), &.{ 4000, 1500, 3500 });
    defer for (&twins) |*st| st.deinit();
    const calls = native.rowCalls();
    // 1500 rows sit in the dense window, 2052 is its first sparse row, 9000 selects among 2250 pools.
    // The small queries scale into FP32 subnormals.
    for ([_]f32{ 0.05, 1e-37 }) |deviation| for ([_]usize{ 1500, 2052, 5000, n }) |length| for (&twins) |*twin| {
        var round = Scope{ .s = s };
        defer round.deinit();
        var state = twin.*;
        state.processed = length;
        const pos = length - 1;
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const q = try normal(&round, &.{ 1, 64, 512 }, 72, deviation);
        const iq = try normal(&round, &.{ 1, 32, 128 }, 73, 1);
        const w = try normal(&round, &.{ 1, 32 }, 74, 0.1);
        const ids = try ops.own(try decodeSelected(&state, iq, w, pos, s));
        const tail = try ops.own(try state.latentView().dense(@intCast(pos), @intCast(length), s));
        const want = (try native.run(&ops, q, state.latentView(), pos, tail, &.{.{ .offset = pos, .length = length, .path = native.pathOf(&.{ 0, 1, 2, 3 }) }}, ids, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
        const got = (try native.runRow(&ops, q, state.latentView(), pos, .{ .ids = ids }, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
        try expectSameBits(want, got);
        const pools = try decodePools(&state, iq, w, pos, s);
        try std.testing.expectEqual(length > 2051, pools != null);
        if (pools) |p| {
            const from_pools = (try native.runRow(&ops, q, state.latentView(), pos, .{ .pools = try ops.own(p) }, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
            try expectSameBits(want, from_pools);
        }
    };
    try std.testing.expectEqual(calls + 2 * (8 + 6), native.rowCalls());
}

fn bf16Values(a: Arr, out: []f32) !void {
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const x = try ops.contiguous(try ops.cast(a, .float32));
    try mlx.check(mlx.mlx_array_eval(x));
    @memcpy(out, mlx.mlx_array_data_float32(x).?[0..out.len]);
}

/// FP64 attention over explicit latent ids: the ground truth the latent arms are held to.
fn attentionOracle(a: std.mem.Allocator, latent: []const f32, q: []const f32, ids: []const i32, rows: usize, scale: f64) ![]f64 {
    const out = try a.alloc(f64, rows * 64 * 512);
    @memset(out, 0);
    const p = try a.alloc(f64, 2051);
    defer a.free(p);
    for (0..rows) |r| for (0..64) |h| {
        const row = ids[r * 2051 ..][0..2051];
        const qh = q[(r * 64 + h) * 512 ..][0..512];
        var top = -std.math.inf(f64);
        for (row, p) |id, *v| {
            v.* = -std.math.inf(f64);
            if (id < 0) continue;
            var dot: f64 = 0;
            for (qh, latent[@as(usize, @intCast(id)) * 512 ..][0..512]) |x, k| dot += @as(f64, x) * k;
            v.* = dot * scale;
            top = @max(top, v.*);
        }
        if (top == -std.math.inf(f64)) continue;
        var den: f64 = 0;
        for (p) |*v| {
            v.* = @exp(v.* - top);
            den += v.*;
        }
        const o = out[(r * 64 + h) * 512 ..][0..512];
        for (row, p) |id, w| if (id >= 0) for (o, latent[@as(usize, @intCast(id)) * 512 ..][0..512]) |*acc, x| {
            acc.* += w * x;
        };
        for (o) |*acc| acc.* /= den;
    };
    return out;
}

/// MLX runs FP32 GEMMs as TF32 on NAX GPUs unless MLX_ENABLE_TF32=0; elsewhere they are full FP32.
fn fp32GemmExact(s: mlx.mlx_stream) !bool {
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const one_plus: f32 = 1 + 1.0 / 4096.0;
    const filled: [16]f32 = @splat(one_plus);
    const identity = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    const x = try ops.binary(.mm, try ops.own(mlx.mlx_array_new_data(&filled, &.{ 4, 4 }, 2, .float32)), try ops.own(mlx.mlx_array_new_data(&identity, &.{ 4, 4 }, 2, .float32)));
    try mlx.check(mlx.mlx_array_eval(x));
    return mlx.mlx_array_data_float32(x).?[0] == one_plus;
}

test "GLM decode attention: the FP32 composite B3 is three B1, held to an FP64 oracle on every GPU" {
    const transformer = @import("transformer.zig");
    const saved = transformer.vqmm_nax_probe_override;
    defer transformer.vqmm_nax_probe_override = saved;
    // A NAX GPU's composite runs block-masked GEMMs, exact FP32 whatever MLX's TF32 default: within one
    // BF16 ulp of the oracle (the fused D512 SDPA misses it on ~70% of the cells, a TF32 GEMM on a few).
    if (@import("glm5_model.zig").naxArms()) try decodeCompositeCase(null);
    transformer.vqmm_nax_probe_override = false;
    try decodeCompositeCase(if (try fp32GemmExact(mlx.gpuStream())) 1.0 / 2048.0 else 1.0 / 64.0);
}

/// `fraction` of the peak |V| an off-NAX cell may add beyond 1/128 of its value; null holds every cell
/// to one BF16 ulp of the oracle.
fn decodeCompositeCase(fraction: ?f64) !void {
    const native = @import("glm5_attention_decode_batch.zig");
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const prefix_rows = 3000;
    const prefix = Latent{ .data = try normal(&scope, &.{ prefix_rows, 512 }, 30, 1) };
    const tape = try normal(&scope, &.{ 3, 512 }, 31, 1);
    const branches = [_]native.Branch{
        .{ .offset = prefix_rows, .length = prefix_rows + 1, .path = native.pathOf(&.{0}) },
        .{ .offset = prefix_rows + 1, .length = prefix_rows + 2, .path = native.pathOf(&.{ 0, 1 }) },
        .{ .offset = prefix_rows + 1, .length = prefix_rows + 2, .path = native.pathOf(&.{ 0, 2 }) },
    };
    var ids: [3 * 2051]i32 = undefined;
    for (branches, 0..) |branch, row| for (0..2051) |k| {
        ids[row * 2051 + k] = if (k % 41 == 7) -1 else @intCast(branch.length - 2051 + k);
    };
    const selected = try scope.own(mlx.mlx_array_new_data(&ids, &.{ 3, 2051 }, 2, .int32));
    const q = try normal(&scope, &.{ 3, 64, 512 }, 32, 2);
    var ops = Ops{ .s = s };
    defer ops.deinit();
    native.resetCalls();
    const b3 = (try native.run(&ops, q, prefix, prefix_rows, tape, &branches, selected, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
    var b1: [3]Arr = undefined;
    for (0..3) |r| {
        const at: c_int = @intCast(r);
        b1[r] = (try native.run(&ops, try ops.slice(q, 0, at, at + 1), prefix, prefix_rows, tape, branches[r .. r + 1], try ops.slice(selected, 0, at, at + 1), 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
        try expectSameBits(b1[r], try ops.slice(b3, 0, at, at + 1));
    }
    try std.testing.expectEqual(@as(usize, 1), native.b3Calls());
    try std.testing.expectEqual(@as(usize, 3), native.b1Calls());
    const rows = try a.alloc(f32, (prefix_rows + 3) * 512);
    defer a.free(rows);
    try bf16Values(prefix.data, rows[0 .. prefix_rows * 512]);
    var tape_rows: [3 * 512]f32 = undefined;
    try bf16Values(tape, &tape_rows);
    var peak: f32 = 0;
    for (rows[0 .. prefix_rows * 512]) |x| peak = @max(peak, @abs(x));
    var qv: [64 * 512]f32 = undefined;
    var got: [64 * 512]f32 = undefined;
    for (branches, 0..) |branch, r| {
        for (0..3) |t| @memcpy(rows[(prefix_rows + t) * 512 ..][0..512], tape_rows[branch.path[t] * 512 ..][0..512]);
        const at: c_int = @intCast(r);
        try bf16Values(try ops.slice(q, 0, at, at + 1), &qv);
        try bf16Values(b1[r], &got);
        const want = try attentionOracle(a, rows, &qv, ids[r * 2051 ..][0..2051], 1, 1.0 / 16.0);
        defer a.free(want);
        for (got, want) |c, o| {
            const ulp = std.math.pow(f64, 2, @floor(std.math.log2(@max(@abs(o), 1e-30))) - 7);
            const bar = if (fraction) |f| @abs(o) / 128 + peak * f else ulp + peak / 65536.0;
            try std.testing.expect(std.math.isFinite(c) and @abs(c - o) <= bar);
        }
    }
}

test "GLM sparse prefill without NAX takes the FP32 composite, no worse than the scalar arm" {
    const transformer = @import("transformer.zig");
    const saved = transformer.vqmm_nax_probe_override;
    defer transformer.vqmm_nax_probe_override = saved;
    transformer.vqmm_nax_probe_override = false;
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var scope = Scope{ .s = s };
    defer scope.deinit();
    const n = 2100;
    const rows = packed_nax.composite_rows;
    const offset = n - rows;
    var state = State{ .processed = n };
    defer state.deinit();
    state.latent = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&state.latent, try normal(&scope, &.{ n, 512 }, 20, 1)));
    var ids: [rows * 2051]i32 = undefined;
    for (0..rows) |r| for (0..2051) |k| {
        ids[r * 2051 + k] = if (r == rows - 1 or k % 37 == 5) -1 else @intCast(offset + r - 2050 + k);
    };
    const selected = try scope.own(mlx.mlx_array_new_data(&ids, &.{ rows, 2051 }, 2, .int32));
    const q = try normal(&scope, &.{ rows, 64, 512 }, 22, 2);
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const before = packed_nax.compositeCount();
    const composite = (try packed_nax.run(&ops, q, state.latentView(), selected, offset, n, 1.0 / 16.0)) orelse return error.ExpectedComposite;
    try std.testing.expectEqual(before + 1, packed_nax.compositeCount());
    const scalar = try attentionChunk(&scope, &state, q, selected, offset, 1.0 / 16.0, 1, false, null);
    const latent = try a.alloc(f32, n * 512);
    defer a.free(latent);
    try bf16Values(state.latent, latent);
    const qv = try a.alloc(f32, rows * 64 * 512);
    defer a.free(qv);
    try bf16Values(q, qv);
    const want = try attentionOracle(a, latent, qv, &ids, rows, 1.0 / 16.0);
    defer a.free(want);
    var peak: f32 = 0;
    for (latent) |x| peak = @max(peak, @abs(x));
    const got = try a.alloc(f32, want.len);
    defer a.free(got);
    const ref = try a.alloc(f32, want.len);
    defer a.free(ref);
    try bf16Values(composite, got);
    try bf16Values(scalar, ref);
    // A store rounding flip plus 2^-11 of max|V| (TF32 scores widen it to 2^-6 on a NAX GPU's MLX).
    const fraction: f64 = if (try fp32GemmExact(s)) 1.0 / 2048.0 else 1.0 / 64.0;
    const slack = peak * fraction;
    for (got, ref, want) |c, r, o| {
        try std.testing.expect(std.math.isFinite(c));
        try std.testing.expect(@abs(c - o) <= @abs(r - o) + @abs(o) / 128 + slack);
    }
    for (got[(rows - 1) * 64 * 512 ..]) |c| try std.testing.expectEqual(@as(f32, 0), c);
}

test "GLM sparse attention arms per chunk (SUSHI_GLM_ATTN_UBENCH)" {
    if (!@import("transformer.zig").diagEnvOn("SUSHI_GLM_ATTN_UBENCH")) return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    const history: usize = 16384;
    var scope = Scope{ .s = s };
    defer scope.deinit();
    var state = State{ .processed = history };
    defer state.deinit();
    state.latent = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&state.latent, try normal(&scope, &.{ @intCast(history), 512 }, 71, 1)));
    try mlx.check(mlx.mlx_array_eval(state.latent));
    const io = std.Io.Threaded.global_single_threaded.io();
    const transformer = @import("transformer.zig");
    const saved = transformer.vqmm_nax_probe_override;
    defer transformer.vqmm_nax_probe_override = saved;
    const nax = @import("glm5_model.zig").naxArms();
    for ([_]c_int{ 1, 8 }) |rows| {
        const offset = history - @as(usize, @intCast(rows));
        const ids = try a.alloc(i32, @as(usize, @intCast(rows)) * 2051);
        defer a.free(ids);
        for (0..@intCast(rows)) |r| for (0..2051) |k| {
            ids[r * 2051 + k] = if (k % 37 == 5) -1 else @intCast(offset + r - 2050 + k);
        };
        const selected = try scope.own(mlx.mlx_array_new_data(ids.ptr, &.{ rows, 2051 }, 2, .int32));
        const q = try normal(&scope, &.{ rows, 64, 512 }, 72 + @as(u64, @intCast(rows)), 2);
        try mlx.check(mlx.mlx_array_eval(q));
        var outs: [3]Arr = .{ nil, nil, nil };
        var times: [3][12]f64 = undefined;
        for (0..14) |rep| for (0..3) |arm| {
            if (arm == 2 and !nax) continue;
            var arm_scope = Scope{ .s = s };
            defer arm_scope.deinit();
            var ops = Ops{ .s = s };
            defer ops.deinit();
            transformer.vqmm_nax_probe_override = if (arm == 1) false else saved;
            const sw = @import("io_util.zig").Stopwatch.init(io);
            const out = if (arm == 0)
                try attentionChunk(&arm_scope, &state, q, selected, offset, 1.0 / 16.0, if (rows == 1) 8 else 1, false, null)
            else
                (try packed_nax.run(&ops, q, state.latentView(), selected, offset, history, 1.0 / 16.0)).?;
            try mlx.check(mlx.mlx_array_eval(out));
            if (rep >= 2) times[arm][rep - 2] = @as(f64, @floatFromInt(sw.read())) / 1e6;
            if (rep == 0) outs[arm] = try scope.own(try arm_scope.result(out));
        };
        const n: usize = @intCast(64 * 512);
        const want = blk: {
            const lat = try a.alloc(f32, history * 512);
            defer a.free(lat);
            try bf16Values(state.latent, lat);
            const qv = try a.alloc(f32, n);
            defer a.free(qv);
            try bf16Values(try scope.cut(q, 0, 1), qv);
            break :blk try attentionOracle(a, lat, qv, ids[0..2051], 1, 1.0 / 16.0);
        };
        defer a.free(want);
        const got = try a.alloc(f32, n);
        defer a.free(got);
        for (0..3) |arm| {
            if (arm == 2 and !nax) continue;
            try bf16Values(try scope.cut(outs[arm], 0, 1), got);
            var worst: f64 = 0;
            var peak: f64 = 0;
            var sq: f64 = 0;
            for (got, want) |g, w| {
                worst = @max(worst, @abs(g - w));
                peak = @max(peak, @abs(w));
                sq += (g - w) * (g - w);
            }
            std.mem.sort(f64, &times[arm], {}, std.sort.asc(f64));
            std.debug.print("[glm-attn-ubench] rows={d} arm={s} median {d:.3} ms min {d:.3} ms | row0 vs f64: max abs {e:.3} rms {e:.3} peak {d:.3}\n", .{ rows, ([_][]const u8{ "scalar", "composite", "nax" })[arm], times[arm][6], times[arm][0], worst, @sqrt(sq / @as(f64, @floatFromInt(n))), peak });
        }
    }
}

const UbenchStage = enum { scores, select, attend };

fn ubenchRun(stage: UbenchStage, state: *const State, q: Arr, iq: Arr, w: Arr, offset: usize, s: mlx.mlx_stream) !void {
    const rows: usize = @intCast(mlx.getShape(q)[0]);
    if (stage == .attend) {
        const out = try attend(state, q, iq, w, offset, 1.0 / 16.0, s);
        defer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        return;
    }
    var start: usize = 0;
    while (start < rows) {
        var scope = Scope{ .s = s };
        defer scope.deinit();
        const outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        for (0..2) |_| {
            if (start == rows) break;
            const end = @min(rows, start + packed_nax.max_rows);
            var node = state.*;
            if (rows == 1) node.processed = offset + 1;
            const iqc = try scope.cut(iq, @intCast(start), @intCast(end));
            const wc = try scope.cut(w, @intCast(start), @intCast(end));
            const out = if (stage == .scores) try indexScores(&scope, &node, iqc, wc, offset + start) else try selectChunk(&scope, &node, iqc, wc, offset + start);
            try mlx.check(mlx.mlx_vector_array_append_value(outputs, out));
            start = end;
        }
        try mlx.check(mlx.mlx_eval(outputs));
    }
}

fn ubenchMedianMs(io: std.Io, reps: usize, stage: UbenchStage, state: *const State, q: Arr, iq: Arr, w: Arr, offset: usize, s: mlx.mlx_stream) !f64 {
    var samples: [16]f64 = undefined;
    const n = @min(reps, samples.len);
    try ubenchRun(stage, state, q, iq, w, offset, s);
    for (samples[0..n]) |*sample| {
        const sw = @import("io_util.zig").Stopwatch.init(io);
        try ubenchRun(stage, state, q, iq, w, offset, s);
        sample.* = @as(f64, @floatFromInt(sw.read())) / 1e6;
    }
    std.mem.sort(f64, samples[0..n], {}, std.sort.asc(f64));
    return samples[n / 2];
}

// DIAGNOSTIC (SUSHI_GLM_LONGCTX_UBENCH=1): one MLA layer's long-context attention on a synthetic kv8
// state of `_CTX` rows (default 2048,8192,32768,131072): decode (one row) and a 2048-row prefill
// chunk, split into index scores, top-512 selection and the whole attention. Synchronizing stages
// rank costs; multiply by 11 MLA layers for a token or a chunk.
test "GLM long-context attention components (SUSHI_GLM_LONGCTX_UBENCH=1)" {
    if (!@import("transformer.zig").diagEnvOn("SUSHI_GLM_LONGCTX_UBENCH")) return error.SkipZigTest;
    if (!@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    const raw = if (std.c.getenv("SUSHI_GLM_LONGCTX_UBENCH_CTX")) |r| std.mem.sliceTo(r, 0) else "2048,8192,32768,131072";
    const reps: usize = if (std.c.getenv("SUSHI_GLM_LONGCTX_UBENCH_REPS")) |r| try std.fmt.parseInt(usize, std.mem.sliceTo(r, 0), 10) else 5;
    var it = std.mem.tokenizeScalar(u8, raw, ',');
    while (it.next()) |token| {
        const n = try std.fmt.parseInt(c_int, token, 10);
        var scope = Scope{ .s = s };
        defer scope.deinit();
        var state = State{ .latent_bits = latent_store.kv8_bits };
        defer state.deinit();
        const ape = try scope.zeros(&.{ 4, 128 }, .bfloat16);
        var filled: c_int = 0;
        while (filled < n) {
            const rows = @min(n - filled, 16384);
            var part = Scope{ .s = s };
            defer part.deinit();
            const seed: u64 = @intCast(filled);
            _ = try state.append(try normal(&part, &.{ rows, 512 }, seed + 1, 1), try normal(&part, &.{ rows, 128 }, seed + 2, 1), try normal(&part, &.{ rows, 128 }, seed + 3, 1), ape, s);
            try state.evaluate();
            filled += rows;
        }
        const total: usize = @intCast(n);
        const decode_q = try normal(&scope, &.{ 1, 64, 512 }, 21, 0.05);
        const decode_iq = try normal(&scope, &.{ 1, 32, 128 }, 22, 1);
        const decode_w = try normal(&scope, &.{ 1, 32 }, 23, 0.2);
        var ms: [3]f64 = undefined;
        for (std.enums.values(UbenchStage), &ms) |stage, *m| m.* = try ubenchMedianMs(io, reps, stage, &state, decode_q, decode_iq, decode_w, total - 1, s);
        std.debug.print("[glm-attn-ubench] ctx={d} decode: scores {d:.3} ms, select {d:.3} ms, attend {d:.3} ms\n", .{ n, ms[0], ms[1], ms[2] });
        {
            // One decode row appended to the state it owns, then to a shared copy (no donation).
            const row_latent = try normal(&scope, &.{ 1, 512 }, 24, 1);
            const row_key = try normal(&scope, &.{ 1, 128 }, 25, 1);
            for (0..2) |shared| {
                var line: [8]f64 = undefined;
                for (&line) |*m| {
                    var other = if (shared == 1) try state.share() else State{};
                    defer other.deinit();
                    const sw = @import("io_util.zig").Stopwatch.init(io);
                    _ = try state.append(row_latent, row_key, row_key, ape, s);
                    try state.evaluate();
                    m.* = @as(f64, @floatFromInt(sw.read())) / 1e6;
                }
                std.debug.print("[glm-attn-ubench] ctx={d} decode append {s} ms:", .{ n, if (shared == 1) "shared" else "owned" });
                for (line) |m| std.debug.print(" {d:.3}", .{m});
                std.debug.print("\n", .{});
            }
        }
        if (n < 2051 + 2048) continue;
        {
            // A prefill chunk appended to the owned state: each one regrows capacity.
            var part = Scope{ .s = s };
            defer part.deinit();
            const chunk_latent = try normal(&part, &.{ 2048, 512 }, 26, 1);
            const chunk_key = try normal(&part, &.{ 2048, 128 }, 27, 1);
            try mlx.check(mlx.mlx_array_eval(chunk_latent));
            try mlx.check(mlx.mlx_array_eval(chunk_key));
            std.debug.print("[glm-attn-ubench] ctx={d} chunk append ms:", .{n});
            for (0..4) |_| {
                const sw = @import("io_util.zig").Stopwatch.init(io);
                _ = try state.append(chunk_latent, chunk_key, chunk_key, ape, s);
                try state.evaluate();
                std.debug.print(" {d:.3}", .{@as(f64, @floatFromInt(sw.read())) / 1e6});
            }
            std.debug.print("\n", .{});
        }
        const prefill_q = try normal(&scope, &.{ 2048, 64, 512 }, 31, 0.05);
        const prefill_iq = try normal(&scope, &.{ 2048, 32, 128 }, 32, 1);
        const prefill_w = try normal(&scope, &.{ 2048, 32 }, 33, 0.2);
        for (std.enums.values(UbenchStage), &ms) |stage, *m| m.* = try ubenchMedianMs(io, @min(reps, 3), stage, &state, prefill_q, prefill_iq, prefill_w, total - 2048, s);
        std.debug.print("[glm-attn-ubench] ctx={d} prefill chunk 2048: scores {d:.1} ms, select {d:.1} ms, attend {d:.1} ms\n", .{ n, ms[0], ms[1], ms[2] });
    }
}

const DecodeStage = enum { null_chain, scores, select, composite_cast, composite_qk, composite_softmax, composite, full };

fn decodeComposite(ops: *Ops, q: Arr, kv_bank: Arr, mask: Arr, scale: f32, stage: DecodeStage) !Arr {
    const kv = try ops.cast(kv_bank, .float32);
    if (stage == .composite_cast) return kv;
    const queries = try ops.binary(.mul, try ops.cast(q, .float32), try ops.scalar(scale, .float32));
    const kt = try ops.transpose(kv, &.{ 0, 2, 1 });
    const blocks = try ops.ones(&.{ 1, 2, 65 }, .bool_);
    const qk = try ops.slot();
    try mlx.check(mlx.mlx_block_masked_mm(qk, queries, kt, 32, blocks, .{ .ctx = null }, .{ .ctx = null }, ops.s));
    const masked = try ops.slot();
    try mlx.check(mlx.mlx_where(masked, try ops.reshape(mask, &.{ 1, 1, 2051 }), qk.*, try ops.scalar(-std.math.inf(f32), .float32), ops.s));
    if (stage == .composite_qk) return masked.*;
    const probs = try ops.softmax(masked.*, -1);
    if (stage == .composite_softmax) return probs;
    const blocks2 = try ops.ones(&.{ 1, 2, 16 }, .bool_);
    const pv = try ops.slot();
    try mlx.check(mlx.mlx_block_masked_mm(pv, probs, kv, 32, blocks2, .{ .ctx = null }, .{ .ctx = null }, ops.s));
    return ops.cast(pv.*, .bfloat16);
}

// DIAGNOSTIC (SUSHI_GLM_DECODE_UBENCH=1, `_CTX`): one serial decode row's MLA attention as a dependent chain of
// 16 layers (each layer's query and index inputs come from the previous output), cumulative stages; the
// per-layer figure includes the chain's token plumbing, which `null_chain` measures.
test "GLM decode row attention stages (SUSHI_GLM_DECODE_UBENCH=1)" {
    if (!@import("transformer.zig").diagEnvOn("SUSHI_GLM_DECODE_UBENCH")) return error.SkipZigTest;
    if (!@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    const raw = if (std.c.getenv("SUSHI_GLM_DECODE_UBENCH_CTX")) |r| std.mem.sliceTo(r, 0) else "1024,8192,32768,131072";
    var it = std.mem.tokenizeScalar(u8, raw, ',');
    while (it.next()) |token| {
        const n = try std.fmt.parseInt(c_int, token, 10);
        var scope = Scope{ .s = s };
        defer scope.deinit();
        var state = State{ .latent_bits = latent_store.kv8_bits };
        defer state.deinit();
        const ape = try scope.zeros(&.{ 4, 128 }, .bfloat16);
        var filled: c_int = 0;
        while (filled < n) {
            const rows = @min(n - filled, 16384);
            var part = Scope{ .s = s };
            defer part.deinit();
            const seed: u64 = @intCast(filled);
            _ = try state.append(try normal(&part, &.{ rows, 512 }, seed + 1, 1), try normal(&part, &.{ rows, 128 }, seed + 2, 1), try normal(&part, &.{ rows, 128 }, seed + 3, 1), ape, s);
            try state.evaluate();
            filled += rows;
        }
        const pos: usize = @intCast(n - 1);
        const q0 = try normal(&scope, &.{ 1, 64, 512 }, 21, 0.05);
        const bank = try normal(&scope, &.{ 1, 2051, 512 }, 22, 1);
        const mask = try scope.slot();
        try mlx.check(mlx.mlx_ones(mask, &.{ 1, 2051 }, 2, .bool_, s));
        try mlx.check(mlx.mlx_array_eval(q0));
        try mlx.check(mlx.mlx_array_eval(bank));
        try mlx.check(mlx.mlx_array_eval(mask.*));
        var line: [8]f64 = undefined;
        for (std.enums.values(DecodeStage), &line) |stage, *out| {
            var samples: [9]f64 = undefined;
            for (&samples, 0..) |*sample, rep| {
                _ = rep;
                var opses: [16]Ops = undefined;
                var scopes: [16]Scope = undefined;
                for (&opses, &scopes) |*o, *sc| {
                    o.* = Ops{ .s = s };
                    sc.* = Scope{ .s = s };
                }
                defer for (&opses, &scopes) |*o, *sc| {
                    o.deinit();
                    sc.deinit();
                };
                var q = q0;
                const sw = @import("io_util.zig").Stopwatch.init(io);
                for (&opses, &scopes) |*ops, *sc| {
                    const flat = try ops.reshape(q, &.{32768});
                    const iq = try ops.reshape(try ops.slice(flat, 0, 0, 4096), &.{ 1, 32, 128 });
                    const w = try ops.reshape(try ops.slice(flat, 0, 4096, 4128), &.{ 1, 32 });
                    var node = state;
                    node.processed = pos + 1;
                    const product: Arr = switch (stage) {
                        .null_chain => q,
                        .scores => try indexScores(sc, &node, iq, w, pos),
                        .select => try ops.own(try decodeSelected(&state, iq, w, pos, s)),
                        .composite_cast, .composite_qk, .composite_softmax => try decodeComposite(ops, q, bank, mask.*, 1.0 / 16.0, stage),
                        .composite => try packed_nax.compositeWith(ops, q, bank, mask.*, 1.0 / 16.0, .block_masked),
                        .full => try ops.own(try attend(&state, q, iq, w, pos, 1.0 / 16.0, s)),
                    };
                    const first = try ops.cast(try ops.reshape(try ops.slice(try ops.reshape(product, &.{-1}), 0, 0, 1), &.{1}), .float32);
                    const token_value = try ops.cast(try ops.binary(.mul, first, try ops.scalar(0, .float32)), .bfloat16);
                    q = try ops.binary(.add, q, token_value);
                }
                try mlx.check(mlx.mlx_array_eval(q));
                sample.* = @as(f64, @floatFromInt(sw.read())) / 1e3 / 16.0;
            }
            std.mem.sort(f64, &samples, {}, std.sort.asc(f64));
            out.* = samples[4];
        }
        std.debug.print("[glm-decode-ubench] ctx={d} us/layer:", .{n});
        for (std.enums.values(DecodeStage), line) |stage, v| std.debug.print(" {s}={d:.0}", .{ @tagName(stage), v });
        std.debug.print("\n", .{});
    }
}
