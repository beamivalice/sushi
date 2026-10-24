//! DeepSeek-V4.1-Flash (`deepseek_v41`) served from MLX ops, loaded from the release's own tensors:
//! hyper-connections whose mix each sublayer takes from the one before it, attention over a sliding
//! window plus compressed KV that layers share through an indexer, Engram, and routed experts.
//! Reference: DeepSeek's `inference/model.py`. Ported in part from mlx-serve's `deepseek_v41.zig` (MIT).
//!
//! The request's state lives on the model, so one request runs at a time. A chunk runs one layer at
//! a time over all its rows: a layer's experts are read once per chunk.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const cfg_mod = @import("dsv41_config.zig");
const sc_mod = @import("dsv41_scope.zig");
const qat_mod = @import("dsv41_qat.zig");
const kernels = @import("dsv41_kernels.zig");
const engram = @import("dsv41_engram.zig");
const source = @import("dsv41_source.zig");
const model_mod = @import("model.zig");
const exl3 = @import("sushi_exl3");
const expert_usage = @import("expert_usage.zig");

const A = sc_mod.A;
const S = sc_mod.S;
const Scope = sc_mod.Scope;
const none = sc_mod.none;
const dim = sc_mod.dim;
const ci = sc_mod.ci;
const evalAll = sc_mod.evalAll;
const Config = cfg_mod.Config;

/// Rows one attention span handles: the compressed rows each query gathers are its largest
/// transient (`[span, index_topk, head_dim]` in f32).
pub const SPAN: usize = 256;
/// Index keys scored per matmul: `[rows, index_heads, tile]` in f32.
const INDEX_TILE: usize = 4096;
/// Keys one pass of a wide span's streamed top-k scores (`indexStream`).
const INDEX_STREAM_TILE: usize = @intCast(cfg_mod.index_stream_tile);
/// Tests: the spans whose index top-k streamed.
var index_streams: std.atomic.Value(usize) = .init(0);

// ── weights ──────────────────────────────────────────────────────────────

/// Tests: MLX's quantized matmul rounds its f32 inputs (about 5e-4 relative), so the tight f32 bar
/// dequantizes the weights and runs a plain matmul instead.
pub var exact_matmul: bool = false;

fn dequant(sc: *Scope, w: A, s: A, bits: c_int, mode: [*:0]const u8, dt: mlx.mlx_dtype) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_dequantize(&o, w, s, none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(bits), mode, none, .{ .value = dt, .has_value = true }, sc.s), &o);
}

/// One linear: a dense `[out, in]` weight, or the release's fp8 as MLX `mxfp8`: e4m3 codes four to a
/// u32 and one e8m0 scale per 32 columns of every row (the release's 32x32 block scale repeated down
/// its rows).
pub const Lin = struct {
    w: A,
    s: A = none,
    /// A trellis bank (`trunk_quant`); `w` and `s` are then unused.
    stq: ?Stq = null,

    pub fn outDim(self: *const Lin) usize {
        return if (self.stq) |q| q.outDim() else dim(self.w, 0);
    }

    pub fn apply(self: *const Lin, sc: *Scope, x: A) !A {
        if (self.stq) |q| return q.apply(sc, x);
        if (self.s.ctx == null) return sc.matmul(x, try sc.t2(self.w));
        if (exact_matmul) return sc.matmul(x, try sc.t2(try dequant(sc, self.w, self.s, 8, "mxfp8", mlx.mlx_array_dtype(x))));
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_quantized_matmul(&o, x, self.w, self.s, none, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", sc.s), &o);
    }
};

/// A trunk weight as a trellis bank: `p.trellis` `[G, in/16, out/16, n]`, G > 1 only for the grouped `wo_a`.
pub const Stq = struct {
    p: exl3.Proj,
    dec: exl3.format.Decode,

    pub fn outDim(self: Stq) usize {
        return dim(self.p.trellis, 2) * 16;
    }

    /// `x` `[..., in]` through the one-bank projection, in x's dtype.
    pub fn apply(self: Stq, sc: *Scope, x: A) !A {
        const sh = mlx.getShape(x);
        const in = sh[sh.len - 1];
        const rows: c_int = @intCast(@divExact(mlx.mlx_array_size(x), @as(usize, @intCast(in))));
        const y = try sc.keep(try exl3.linear.linear(sc.s, try sc.reshape(x, &.{ rows, in }), self.p, self.dec));
        var shape: [8]c_int = undefined;
        @memcpy(shape[0..sh.len], sh);
        shape[sh.len - 1] = @intCast(self.outDim());
        return sc.astype(try sc.reshape(y, shape[0..sh.len]), mlx.mlx_array_dtype(x));
    }

    /// `x` `[G, rows, in]`, bank g serving `x[g]`.
    pub fn applyGrouped(self: Stq, sc: *Scope, x: A) !A {
        return sc.astype(try sc.keep(try exl3.linear.linearGrouped(sc.s, x, self.p, self.dec)), mlx.mlx_array_dtype(x));
    }
};

/// The embedding as MLX's affine triple, gathered by row and dequantized.
pub const Embed = struct {
    w: A,
    scales: A = none,
    biases: A = none,
    bits: c_int = 0,
    group: c_int = 0,

    pub fn rows(self: *const Embed, sc: *Scope, ids: A, dt: mlx.mlx_dtype) !A {
        if (self.scales.ctx == null) return sc.take(self.w, ids, 0);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_dequantize(&o, try sc.take(self.w, ids, 0), try sc.take(self.scales, ids, 0), try sc.take(self.biases, ids, 0), mlx.mlx_optional_int.some(self.group), mlx.mlx_optional_int.some(self.bits), "affine", none, .{ .value = dt, .has_value = true }, sc.s), &o);
    }

    pub fn vocab(self: *const Embed) usize {
        return dim(self.w, 0);
    }
};

/// Routed experts of one projection stacked `[E, out, in]`, as MLX `mxfp4` (e2m1 codes eight to a u32,
/// one e8m0 per 32 columns).
pub const Bank = kernels.Quant;
pub const BankSet = kernels.Experts;

pub fn gatherQmm(sc: *Scope, x: A, b: Bank, idx: A) !A {
    var o = mlx.mlx_array_new();
    // A dense bank (tests: the decoded twin of a trellis pack).
    if (b.s.ctx == null) return sc.res(mlx.mlx_gather_mm(&o, x, try sc.transpose(b.w, &.{ 0, 2, 1 }), none, idx, false, sc.s), &o);
    if (exact_matmul) {
        const w = try sc.transpose(try dequant(sc, b.w, b.s, 4, "mxfp4", mlx.mlx_array_dtype(x)), &.{ 0, 2, 1 });
        return sc.res(mlx.mlx_gather_mm(&o, x, w, none, idx, false, sc.s), &o);
    }
    return sc.res(mlx.mlx_gather_qmm(&o, x, b.w, b.s, none, none, idx, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "mxfp4", false, sc.s), &o);
}

/// `x` through the routed experts `idx` (`[L, k]`, ids of `banks`' leading axis) weighted by `wts`
/// (`[L, k]` f32): the clamped SwiGLU of each, summed over the k in f32.
pub fn routedSum(sc: *Scope, cfg: *const Config, act: mlx.mlx_dtype, x: A, banks: BankSet, idx: A, wts: A) !A {
    const L = dim(x, 0);
    const k = dim(idx, 1);
    const d = dim(x, 1);
    const uidx = try sc.astype(idx, .uint32);
    const xe = try sc.reshape(x, &.{ ci(L), 1, 1, ci(d) });
    const gate = try gatherQmm(sc, xe, banks.gate, uidx); // [L, k, 1, inter]
    const up = try gatherQmm(sc, xe, banks.up, uidx);
    const a = try sc.astype(try sc.mul(try clampedSwiglu(sc, gate, up, cfg.swiglu_limit), try sc.reshape(wts, &.{ ci(L), ci(k), 1, 1 })), act);
    const dn = try gatherQmm(sc, a, banks.down, uidx); // [L, k, 1, d]
    return sc.sum(try sc.astype(try sc.reshape(dn, &.{ ci(L), ci(k), ci(d) }), .float32), 1, false);
}

pub fn clampedSwiglu(sc: *Scope, gate: A, up: A, limit: f32) !A {
    var g = try sc.astype(gate, .float32);
    var u = try sc.astype(up, .float32);
    if (limit > 0) {
        g = try sc.min(g, try sc.f(limit));
        u = try sc.clip(u, -limit, limit);
    }
    return sc.mul(try sc.mul(g, try sc.sigmoid(g)), u);
}

/// Where a layer's routed experts come from: stacked banks in memory, or the expert stream.
pub const Experts = union(enum) {
    resident: []BankSet,
    hook: Hook,

    pub const Hook = struct {
        ctx: *anyopaque,
        /// The routed sum (f32 `[L, dim]`) of normed `x` over router picks `idx` (`[L, k]`) weighted `wts`, plus `shared`
        /// (the shared expert's `[L, dim]`) added in f32 when given. `logits` are the router's (`Routes.logits`).
        apply: *const fn (ctx: *anyopaque, sc: *Scope, layer: usize, x: A, idx: A, wts: A, logits: A, shared: ?A) anyerror!A,
        /// The layer's picks `idx` (`[L, k]`) made readable as banks, with the ids as slots of those banks. The banks
        /// stay valid until `release`, which the caller calls once everything reading them is evaluated.
        lease: *const fn (ctx: *anyopaque, sc: *Scope, layer: usize, idx: A) anyerror!Lease,
        release: *const fn (ctx: *anyopaque) void,
        /// Profiling: the stream's cumulative host time (SSD reads and cache copies), in ns.
        host_ns: ?*const fn (ctx: *anyopaque) u64 = null,
    };

    pub const Lease = struct { banks: BankSet, local: A };
};

const Hc = struct {
    fn_w: A, // f32 [mix, hc * dim]
    scale: A, // f32 [3]
    base: A, // f32 [mix]
};

/// `shared`: wkv and wgate stacked into one bank (a pooling compressor's two f32 projections of the same rows).
const CompW = struct { wkv: Lin, wgate: ?Lin, norm: A, shared: ?exl3.linear.Shared = null };
const IdxW = struct { wq_b: Lin, weights_proj: Lin, wk: ?Lin, k_norm: A };

const Layer = struct {
    attn_norm: A,
    ffn_norm: A,
    q_norm: A,
    kv_norm: A,
    hc_attn: Hc,
    hc_ffn: Hc,
    wq_a: Lin,
    wq_b: Lin,
    wkv: Lin,
    /// Grouped output projection as `[o_groups, o_lora, in]` views.
    wo_a: Lin,
    wo_b: Lin,
    sink: A, // f32 [1, heads, 1]
    ratio: u8,
    comp: ?CompW,
    idx: ?IdxW,
    gate_w: A, // f32 [E, dim]
    gate_bias: A, // f32 [1, E]
    /// The bias an image span's rows pick experts by (`noaux_tc_for_vl`); none on a pack without a tower.
    gate_bias_vl: A = none,
    shared: [3]Lin, // w1 (gate), w3 (up), w2 (down)
    top_k: usize,
    /// wq_a and wkv (and a ratio-1 compressor's wkv) stacked into one bank: they all project the attention input.
    in_shared: ?exl3.linear.Shared = null,
};

const EngramLayer = struct {
    layer: usize,
    wkv: Lin,
    qk: A, // f32 [hc, dim]: q_weight * k_weight
    table: engram.Table,
};

/// The DSpark drafter around its stages (`Model.layers[cfg.layers..]`): the projection of the target layers' main
/// stream, the last stage's norm, the Markov head and the confidence head.
const DsparkW = struct {
    main_proj: Lin,
    main_norm: A,
    norm: A,
    /// `[vocab, rank]`, the previous token's embedding.
    markov_embed: A,
    /// f32 `[vocab, rank]`: the embedding's bigram bias over the vocabulary.
    markov_head: A,
    /// f32 `[1, dim + rank]`: the confidence logit over `[collapsed stream, markov embedding]`.
    conf: A,
    /// Each stage's routed experts, stacked (FP4), or as trellis banks (`expert_quant`; `banks` then empty).
    banks: []BankSet,
    stq: []exl3.Ragged = &.{},
};

/// Rotary tables as inverse frequencies `[rope_dim / 2]` f32.
const Rope = struct { freqs: A };

// ── state ────────────────────────────────────────────────────────────────

/// Rows appended one chunk at a time, in a buffer that grows by `rows_grow_div`'s fraction.
const Rows = struct {
    /// The rows `[cap, width]` in `dtype`, or packed: E2M1 codes `[cap, width/2]` u8.
    buf: A = none,
    /// Packed: the scale bytes `[cap, width/group]`.
    sbuf: A = none,
    used: usize = 0,
    cap: usize = 0,
    width: usize,
    dtype: mlx.mlx_dtype,
    /// The rows are an FP4 round trip's output, stored in its own format (`qat_mod.packFp4`).
    pack: ?Pack = null,

    const Pack = struct { mode: qat_mod.Qat, group: usize };

    fn deinit(self: *Rows) void {
        if (self.buf.ctx != null) _ = mlx.mlx_array_free(self.buf);
        if (self.sbuf.ctx != null) _ = mlx.mlx_array_free(self.sbuf);
    }

    /// `buf` grown to `cap` rows of `cols`, its first `used` rows kept.
    fn grow(buf: *A, used: usize, cap: usize, cols: usize, dt: mlx.mlx_dtype, s: S) !void {
        var z = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ ci(cap), ci(cols) }, 2, dt, s));
        if (used > 0) {
            var old = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(old);
            try mlx.check(mlx.mlx_slice(&old, buf.*, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(used), ci(cols) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            const zeros = z;
            defer _ = mlx.mlx_array_free(zeros);
            var up = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice_update(&up, zeros, old, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(used), ci(cols) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            z = up;
        }
        if (buf.ctx != null) _ = mlx.mlx_array_free(buf.*);
        buf.* = z;
    }

    fn put(buf: *A, x: A, at: usize, s: S) !void {
        const n = dim(x, 0);
        const cols = dim(x, 1);
        var up = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice_update(&up, buf.*, x, &[_]c_int{ ci(at), 0 }, 2, &[_]c_int{ ci(at + n), ci(cols) }, 2, &[_]c_int{ 1, 1 }, 2, s));
        _ = mlx.mlx_array_free(buf.*);
        buf.* = up;
    }

    fn append(self: *Rows, sc: *Scope, x: A) !void {
        const n = dim(x, 0);
        if (n == 0) return;
        const s = sc.s;
        if (self.used + n > self.cap) {
            const cap = @max(self.used + n, @max(64, self.cap + self.cap / cfg_mod.rows_grow_div));
            if (self.pack) |pk| {
                try grow(&self.buf, self.used, cap, self.width / 2, .uint8, s);
                try grow(&self.sbuf, self.used, cap, self.width / pk.group, .uint8, s);
            } else try grow(&self.buf, self.used, cap, self.width, self.dtype, s);
            self.cap = cap;
        }
        if (self.pack) |pk| {
            const p = try qat_mod.packFp4(sc, try sc.astype(x, self.dtype), pk.mode, pk.group);
            try put(&self.buf, p.codes, self.used, s);
            try put(&self.sbuf, p.scales, self.used, s);
        } else try put(&self.buf, try sc.astype(x, self.dtype), self.used, s);
        self.used += n;
    }

    /// Rows [0, n) as values.
    fn rows(self: *const Rows, sc: *Scope, n: usize) !A {
        return self.rowsIn(sc, 0, n);
    }

    /// Rows [lo, hi) as values.
    fn rowsIn(self: *const Rows, sc: *Scope, lo: usize, hi: usize) !A {
        const head = try sc.slice(self.buf, 0, lo, hi);
        const pk = self.pack orelse return head;
        return qat_mod.unpackFp4(sc, .{ .codes = head, .scales = try sc.slice(self.sbuf, 0, lo, hi) }, pk.mode, pk.group, self.dtype);
    }

    /// Rows `idx` (flat ids below `used`) as values.
    fn gather(self: *const Rows, sc: *Scope, idx: A) !A {
        const pk = self.pack orelse return sc.take(self.buf, idx, 0);
        return qat_mod.unpackFp4(sc, .{ .codes = try sc.take(self.buf, idx, 0), .scales = try sc.take(self.sbuf, idx, 0) }, pk.mode, pk.group, self.dtype);
    }
};

/// The window's rows by absolute position, in a ring of `cap` rows.
const Ring = struct {
    buf: A,
    cap: usize,
    width: usize,

    fn init(cap: usize, width: usize, dt: mlx.mlx_dtype, s: S) !Ring {
        var z = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ ci(cap), ci(width) }, 2, dt, s));
        return .{ .buf = z, .cap = cap, .width = width };
    }

    fn deinit(self: *Ring) void {
        _ = mlx.mlx_array_free(self.buf);
    }

    /// Rows for positions [p0, p0 + n); only the last `cap` survive.
    fn write(self: *Ring, x: A, p0: usize, s: S) !void {
        const n = dim(x, 0);
        const keep_n = @min(n, self.cap);
        const first = p0 + n - keep_n;
        var done: usize = 0;
        while (done < keep_n) {
            const slot = (first + done) % self.cap;
            const run = @min(keep_n - done, self.cap - slot);
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, x, &[_]c_int{ ci(n - keep_n + done), 0 }, 2, &[_]c_int{ ci(n - keep_n + done + run), ci(self.width) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            var cast = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cast);
            try mlx.check(mlx.mlx_astype(&cast, part, mlx.mlx_array_dtype(self.buf), s));
            var up = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice_update(&up, self.buf, cast, &[_]c_int{ ci(slot), 0 }, 2, &[_]c_int{ ci(slot + run), ci(self.width) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            _ = mlx.mlx_array_free(self.buf);
            self.buf = up;
            done += run;
        }
    }

    /// Rows for positions [lo, hi) in order; `hi - lo <= cap`.
    fn read(self: *const Ring, sc: *Scope, lo: usize, hi: usize) !A {
        std.debug.assert(hi - lo <= self.cap);
        var idx: [512]i32 = undefined;
        for (lo..hi, 0..) |p, k| idx[k] = @intCast(p % self.cap);
        return sc.take(self.buf, try sc.ints(idx[0 .. hi - lo], &.{ci(hi - lo)}), 0);
    }
};

const LState = struct {
    win: Ring,
    comp: ?Rows = null,
    idx_k: ?Rows = null,
    /// Ratio > 1 compressors: the open group's kv and gate rows, f32.
    pend_kv: A = none,
    pend_gate: A = none,

    fn deinit(self: *LState) void {
        self.win.deinit();
        if (self.comp) |*r| r.deinit();
        if (self.idx_k) |*r| r.deinit();
        if (self.pend_kv.ctx != null) _ = mlx.mlx_array_free(self.pend_kv);
        if (self.pend_gate.ctx != null) _ = mlx.mlx_array_free(self.pend_gate);
    }
};

/// What a verify chunk keeps to give back its rejected rows: each compressor's row counts and open pooling group
/// as the chunk began, and the main rows the stages' windows take only for the rows that stay.
const Verify = struct {
    n0: usize,
    rows: usize,
    comp_used: []usize,
    idx_used: []usize,
    /// A pooling compressor's rows from its open group's first token on, as the chunk left them before it cut a new group.
    pend_kv: []A,
    pend_gate: []A,
    mh: A = none,
};

pub const State = struct {
    gpa: std.mem.Allocator,
    n: usize = 0,
    layers: []LState,
    /// The request's compressed token ids (Engram look-back).
    hist: std.ArrayList(u32) = .empty,
    /// Per DSpark stage: the window of the main stream's keys for the committed positions.
    ds: []Ring = &.{},
    verify: ?Verify = null,

    pub fn deinit(self: *State) void {
        for (self.layers) |*l| l.deinit();
        self.gpa.free(self.layers);
        for (self.ds) |*r| r.deinit();
        self.gpa.free(self.ds);
        self.hist.deinit(self.gpa);
        self.dropVerify();
    }

    fn dropVerify(self: *State) void {
        const v = self.verify orelse return;
        for (v.pend_kv) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        for (v.pend_gate) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        if (v.mh.ctx != null) _ = mlx.mlx_array_free(v.mh);
        self.gpa.free(v.comp_used);
        self.gpa.free(v.idx_used);
        self.gpa.free(v.pend_kv);
        self.gpa.free(v.pend_gate);
        self.verify = null;
    }
};

fn arrayBytes(a: A) u64 {
    if (a.ctx == null) return 0;
    return @as(u64, mlx.mlx_array_size(a)) * @as(u64, mlx.mlx_array_itemsize(a));
}

/// The state's resident bytes as allocated: every layer's window ring, compressed and index rows
/// (their capacity) and open pooling group.
pub fn stateBytes(st: *const State) u64 {
    var bytes: u64 = 0;
    for (st.layers) |*l| {
        bytes += arrayBytes(l.win.buf) + arrayBytes(l.pend_kv) + arrayBytes(l.pend_gate);
        if (l.comp) |r| bytes += arrayBytes(r.buf) + arrayBytes(r.sbuf);
        if (l.idx_k) |r| bytes += arrayBytes(r.buf) + arrayBytes(r.sbuf);
    }
    for (st.ds) |r| bytes += arrayBytes(r.buf);
    return bytes;
}

// ── the model ────────────────────────────────────────────────────────────

pub const Opts = struct {
    /// bfloat16 serves; float32 is the tests' tight bar.
    act: mlx.mlx_dtype = .bfloat16,
    /// The activation round trips off (plain math, the reference's `DISABLE_FAKE_QUANT`).
    qat: bool = true,
    experts: bool = true,
    /// Load the DSpark drafter's stages (`mtp.*`); their experts are resident.
    dspark: bool = false,
    /// `--drop-expert`: the backbone routers never pick these. Read while the layers load.
    drop: ?*const expert_usage.Drop = null,
};

pub const Want = enum { last, all };

/// Named intermediates, kept as handles of their own.
pub const Trace = struct {
    gpa: std.mem.Allocator,
    items: std.ArrayList(struct { name: []u8, arr: A }) = .empty,

    pub fn put(self: *Trace, comptime fmt: []const u8, args: anytype, x: A) !void {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        errdefer _ = mlx.mlx_array_free(o);
        try self.items.append(self.gpa, .{ .name = try std.fmt.allocPrint(self.gpa, fmt, args), .arr = o });
    }

    pub fn get(self: *const Trace, name: []const u8) ?A {
        for (self.items.items) |it| if (std.mem.eql(u8, it.name, name)) return it.arr;
        return null;
    }

    pub fn deinit(self: *Trace) void {
        for (self.items.items) |it| {
            self.gpa.free(it.name);
            _ = mlx.mlx_array_free(it.arr);
        }
        self.items.deinit(self.gpa);
    }
};

/// Host wall time of a wide chunk per stage, each stage evaluated before the next is built (so graph building counts
/// with the GPU work it queues). `routed` includes the expert fill the `[expert-stream]` line splits out.
pub const Profile = struct {
    pub const Stage = enum { engram, hc_attn, attn_in, q, kv_row, compress, indexer, core, attn_out, hc_ffn, router, shared, fill, routed, expand, head };
    ns: [std.enums.values(Stage).len]u64 = @splat(0),
    /// The CED decoder's layers (from the split on), reported on their own line.
    dec: [std.enums.values(Stage).len]u64 = @splat(0),
    decoder: bool = false,
    io: std.Io,
    last: std.Io.Timestamp,
    /// Decode-width chunks (a verify) are timed too, which gives up their submit-without-waiting schedule.
    narrow: bool = false,

    pub fn init(narrow: bool) Profile {
        const io = std.Io.Threaded.global_single_threaded.io();
        return .{ .io = io, .last = std.Io.Timestamp.now(io, .boot), .narrow = narrow };
    }

    fn restart(self: *Profile) void {
        self.last = std.Io.Timestamp.now(self.io, .boot);
    }

    fn mark(self: *Profile, stage: Stage, xs: []const A) !void {
        try evalAll(xs);
        const now = std.Io.Timestamp.now(self.io, .boot);
        const into = if (self.decoder) &self.dec else &self.ns;
        into[@intFromEnum(stage)] += @intCast(self.last.durationTo(now).nanoseconds);
        self.last = now;
    }

    /// Moves the expert stream's host time (SSD reads, cache copies) out of the routed stage it was timed in.
    fn hostShare(self: *Profile, host_ns: u64) void {
        const into = if (self.decoder) &self.dec else &self.ns;
        const moved = @min(host_ns, into[@intFromEnum(Stage.routed)]);
        into[@intFromEnum(Stage.routed)] -= moved;
        into[@intFromEnum(Stage.fill)] += moved;
    }

    fn report(self: *Profile, rows: usize, p0: usize) void {
        for ([_]*[std.enums.values(Stage).len]u64{ &self.ns, &self.dec }, [_][]const u8{ "", "-dec" }) |arr, tag| {
            var buf: [768]u8 = undefined;
            var w = std.Io.Writer.fixed(&buf);
            var total: u64 = 0;
            for (arr, 0..) |v, i| {
                total += v;
                w.print(" {s}={d:.1}", .{ @tagName(@as(Stage, @enumFromInt(i))), @as(f64, @floatFromInt(v)) / 1e6 }) catch {};
            }
            if (total > 0) log.info("[dsv41-prof{s}] pos={d} rows={d} total_ms={d:.1}{s}\n", .{ tag, p0, rows, @as(f64, @floatFromInt(total)) / 1e6, w.buffered() });
            arr.* = @splat(0);
        }
        self.decoder = false;
    }
};

/// The profile when it times this chunk (wide chunks only: decode keeps its schedule).
fn prof(m: *const Model, rows: usize) ?*Profile {
    const p = m.profile orelse return null;
    return if (rows > NARROW_ROWS or p.narrow) p else null;
}

pub const Model = struct {
    gpa: std.mem.Allocator,
    s: S,
    cfg: Config,
    act: mlx.mlx_dtype,
    qat_on: bool,
    embed: Embed,
    /// f32 `[vocab, dim]`, or a trellis bank.
    head: Lin,
    norm: A,
    layers: []Layer,
    experts: Experts = .{ .resident = &.{} },
    engrams: []EngramLayer = &.{},
    hash: ?engram.Hash = null,
    token_map: []u32 = &.{},
    rope_plain: Rope,
    rope_yarn: Rope,
    dspark: ?DsparkW = null,
    /// Set while the backbone layers load (`Opts.drop`).
    drop: ?*const expert_usage.Drop = null,
    /// Arrays made at load (casts, views, stacks); freed in `deinit`.
    owned: std.ArrayList(A) = .empty,
    dec_state: ?State = null,
    /// Tests: named intermediates of a chunk.
    trace: ?*Trace = null,
    /// The fused Metal kernels of `dsv41_kernels.zig` serve where they apply; off, the op chains they replace
    /// run (the tests' reference arm). The kernels need the GPU stream either way.
    fused: bool = true,
    /// Wide spans take the f16 core (`kernels.attnWide`): faster, and f16 scores change the output, so it is off
    /// until a KLD run clears it.
    attn_f16: bool = false,
    /// TEMPORARY A/B: a decode-width pass reads its Engram rows ahead (`prefetchEngram`).
    decode_engram_ahead: bool = true,
    /// Host time the forwards waited on Engram rows, and the decode forwards since the last report.
    engram_wait_ns: u64 = 0,
    engram_forwards: u64 = 0,
    /// Keys per pass of the streamed index top-k; a test lowers it to stream a tiny context.
    index_stream_tile: usize = INDEX_STREAM_TILE,
    /// Tests: false keeps the compressed and index rows in the activation dtype, the packed form's reference.
    pack_kv: bool = true,
    /// Tests: each layer completes on the GPU before the next is built, the decode schedule's reference.
    sync_layers: bool = false,
    /// CED: the decoder's first layer (`Config.cedSplit`); null where the layout has none (and in tests, the full
    /// forward a CED prefill's encoder state is checked against).
    ced_split: ?usize = null,
    /// Diagnostic (`SUSHI_DSV41_PREFILL_PROFILE`): a wide chunk evaluates after every stage and logs where its time went.
    profile: ?*Profile = null,

    pub fn deinit(self: *Model) void {
        if (self.profile) |pf| self.gpa.destroy(pf);
        if (self.dec_state) |*st| st.deinit();
        for (self.engrams) |*e| closeEngram(e);
        self.gpa.free(self.engrams);
        self.gpa.free(self.token_map);
        switch (self.experts) {
            .resident => |r| self.gpa.free(r),
            .hook => {},
        }
        if (self.dspark) |ds| {
            self.gpa.free(ds.banks);
            for (ds.stq) |*rb| rb.deinit();
            self.gpa.free(ds.stq);
        }
        for (self.owned.items) |x| _ = mlx.mlx_array_free(x);
        self.owned.deinit(self.gpa);
        self.gpa.free(self.layers);
        self.gpa.destroy(self);
    }

    fn own(self: *Model, x: A) !A {
        self.owned.append(self.gpa, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }

    /// `x` cast to `dt` and evaluated, owned by the model.
    fn ownCast(self: *Model, x: A, dt: mlx.mlx_dtype) !A {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&o, x, dt, self.s));
        _ = try self.own(o);
        try mlx.check(mlx.mlx_array_eval(o));
        return o;
    }

    /// The view `x[start..stop]`, evaluated and owned by the model.
    fn ownSlice(self: *Model, x: A, start: []const c_int, stop: []const c_int) !A {
        var ones: [4]c_int = @splat(1);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&o, x, start.ptr, start.len, stop.ptr, stop.len, &ones, start.len, self.s));
        _ = try self.own(o);
        try mlx.check(mlx.mlx_array_eval(o));
        return o;
    }

    /// Frees `x`, an array the model owns, ahead of `deinit`.
    fn disown(self: *Model, x: A) void {
        for (self.owned.items, 0..) |a, i| if (a.ctx == x.ctx) {
            _ = mlx.mlx_array_free(self.owned.swapRemove(i));
            return;
        };
    }

    fn ownReshape(self: *Model, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, self.s));
        _ = try self.own(o);
        try mlx.check(mlx.mlx_array_eval(o));
        return o;
    }

    fn isKvSource(self: *const Model, l: usize) bool {
        return self.cfg.isKvSource(l);
    }
    fn isIndexSource(self: *const Model, l: usize) bool {
        return self.cfg.isIndexSource(l);
    }
    fn ratio(self: *const Model, l: usize) usize {
        return self.cfg.ratios[l];
    }
    fn ropeFor(self: *const Model, li: usize) *const Rope {
        return if (self.ratio(li) > 0) &self.rope_yarn else &self.rope_plain;
    }
    /// A token's Engram id; an image span's positions take no part in an n-gram.
    fn compressedId(self: *const Model, t: u32) u32 {
        if (self.cfg.image_token_id != 0 and t == self.cfg.image_token_id) return engram.Hash.DEAD;
        return if (t < self.token_map.len) self.token_map[t] else 0;
    }
};

const NameBuf = [192]u8;

fn key(buf: *NameBuf, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch error.NameTooLong;
}

/// `<pfx>.<id>.<tail>`: a backbone layer (`layers`) or a DSpark stage (`mtp`).
fn lkey(buf: *NameBuf, pfx: []const u8, id: usize, comptime tail: []const u8) ![]const u8 {
    return key(buf, "{s}.{d}." ++ tail, .{ pfx, id });
}

fn getW(w: *const model_mod.Weights, name: []const u8) !A {
    return w.get(name) orelse {
        log.err("deepseek_v41: missing weight {s}\n", .{name});
        return error.MissingWeight;
    };
}

/// `<base>.weight` as a `Lin`: the release's fp8 + 32x32 block scale becomes mxfp8, a dense one is
/// cast to `dense` (null keeps its dtype).
fn loadLin(m: *Model, w: *model_mod.Weights, base: []const u8, dense: ?mlx.mlx_dtype) !Lin {
    if (try loadStq(m, w, base)) |q| return .{ .w = none, .stq = q };
    var b1: NameBuf = undefined;
    var b2: NameBuf = undefined;
    const wname = try key(&b1, "{s}.weight", .{base});
    const sname = try key(&b2, "{s}.scale", .{base});
    const wt = try getW(w, wname);
    defer {
        source.drop(w, wname);
        source.drop(w, sname);
    }
    if (w.get(sname)) |scales| {
        if (mlx.mlx_array_dtype(wt) != .uint8 or mlx.mlx_array_ndim(wt) != 2 or mlx.mlx_array_dtype(scales) != .uint8) {
            log.err("deepseek_v41: {s} is not an fp8 weight with an e8m0 scale\n", .{base});
            return error.Dsv41WeightLayout;
        }
        const out = dim(wt, 0);
        const inp = dim(wt, 1);
        if (inp % 32 != 0 or dim(scales, 0) != (out + 31) / 32 or dim(scales, 1) != inp / 32) {
            log.err("deepseek_v41: {s} scale grid {d}x{d} does not tile a {d}x{d} weight by 32\n", .{ base, dim(scales, 0), dim(scales, 1), out, inp });
            return error.Dsv41WeightLayout;
        }
        var codes = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_view(&codes, wt, .uint32, m.s));
        _ = try m.own(codes);
        var rep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rep);
        try mlx.check(mlx.mlx_repeat_axis(&rep, scales, 32, 0, m.s));
        var rows = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&rows, rep, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(out), ci(inp / 32) }, 2, &[_]c_int{ 1, 1 }, 2, m.s));
        _ = try m.own(rows);
        try evalAll(&.{ codes, rows });
        return .{ .w = codes, .s = rows };
    }
    return .{ .w = if (dense) |dt| try m.ownCast(wt, dt) else try m.ownCast(wt, mlx.mlx_array_dtype(wt)) };
}

/// `embed.weight` dense, or MLX's affine triple with `embed.scales` and `embed.biases` (bits and group from the shapes).
fn loadEmbed(m: *Model, w: *model_mod.Weights) !Embed {
    const scales = w.get("embed.scales") orelse return .{ .w = try loadDense(m, w, "embed.weight", m.act) };
    defer for ([_][]const u8{ "embed.weight", "embed.scales", "embed.biases" }) |n| source.drop(w, n);
    const wt = try getW(w, "embed.weight");
    const biases = try getW(w, "embed.biases");
    const d: usize = m.cfg.dim;
    const groups = dim(scales, 1);
    const words = dim(wt, 1);
    const ok = mlx.mlx_array_dtype(wt) == .uint32 and groups > 0 and d % groups == 0 and (words * 32) % d == 0 and
        dim(wt, 0) == dim(scales, 0) and mlx.mlx_array_size(biases) == mlx.mlx_array_size(scales);
    if (!ok) {
        log.err("deepseek_v41: embed.weight/scales/biases are not an affine triple over dim {d}\n", .{d});
        return error.Dsv41WeightLayout;
    }
    return .{
        .w = try m.ownCast(wt, .uint32),
        .scales = try m.ownCast(scales, m.act),
        .biases = try m.ownCast(biases, m.act),
        .bits = @intCast(words * 32 / d),
        .group = @intCast(d / groups),
    };
}

/// `<base>.{trellis,suh,svh}` as a trellis bank under the pack's `trunk_quant`, or null when the pack stores `<base>`
/// another way. A one-bank weight is stored `[in/16, out/16, n]` with `[in]` and `[out]` scales.
fn loadStq(m: *Model, w: *model_mod.Weights, base: []const u8) !?Stq {
    var bt: NameBuf = undefined;
    var bu: NameBuf = undefined;
    var bv: NameBuf = undefined;
    const tname = try key(&bt, "{s}.trellis", .{base});
    const uname = try key(&bu, "{s}.suh", .{base});
    const vname = try key(&bv, "{s}.svh", .{base});
    const tr = w.get(tname) orelse return null;
    defer for ([_][]const u8{ tname, uname, vname }) |n| source.drop(w, n);
    const dec = m.cfg.trunk_dec orelse {
        log.err("deepseek_v41: {s} is a trellis bank and config.json has no trunk_quant\n", .{base});
        return error.Dsv41WeightLayout;
    };
    const su = try getW(w, uname);
    const sv = try getW(w, vname);
    const one = mlx.mlx_array_ndim(tr) == 3;
    const tsh = mlx.getShape(tr);
    const g: c_int = if (one) 1 else tsh[0];
    const t4 = if (one) [4]c_int{ 1, tsh[0], tsh[1], tsh[2] } else [4]c_int{ tsh[0], tsh[1], tsh[2], tsh[3] };
    const in = t4[1] * 16;
    const out = t4[2] * 16;
    if (mlx.mlx_array_dtype(tr) != .uint16 or (!one and mlx.mlx_array_ndim(tr) != 4) or mlx.mlx_array_dtype(su) != .float16 or
        mlx.mlx_array_dtype(sv) != .float16 or mlx.mlx_array_size(su) != @as(usize, @intCast(g * in)) or mlx.mlx_array_size(sv) != @as(usize, @intCast(g * out)) or
        exl3.format.kFromPackedDim(@intCast(t4[3])) == null)
    {
        log.err("deepseek_v41: {s} is not a trellis bank with f16 scales\n", .{base});
        return error.Dsv41WeightLayout;
    }
    return .{ .dec = dec, .p = .{
        .trellis = try m.ownReshape(tr, &t4),
        .suh = try m.ownReshape(su, &.{ g, in }),
        .svh = try m.ownReshape(sv, &.{ g, out }),
    } };
}

fn loadDense(m: *Model, w: *model_mod.Weights, name: []const u8, dt: mlx.mlx_dtype) !A {
    const x = try m.ownCast(try getW(w, name), dt);
    source.drop(w, name);
    return x;
}

fn loadHc(m: *Model, w: *model_mod.Weights, pfx: []const u8, id: usize, comptime which: []const u8) !Hc {
    var b: NameBuf = undefined;
    const fn_w = try loadDense(m, w, try lkey(&b, pfx, id, "hc_" ++ which ++ "_fn"), .float32);
    const scale = try loadDense(m, w, try lkey(&b, pfx, id, "hc_" ++ which ++ "_scale"), .float32);
    const base = try loadDense(m, w, try lkey(&b, pfx, id, "hc_" ++ which ++ "_base"), .float32);
    return .{ .fn_w = fn_w, .scale = scale, .base = base };
}

/// The stacked mxfp4 bank of one projection (`w1` gate, `w3` up, `w2` down) over every expert.
fn loadBank(m: *Model, w: *model_mod.Weights, pfx: []const u8, id: usize, comptime proj: []const u8, experts: usize, rows: usize, cols: usize) !Bank {
    var codes = try m.gpa.alloc(A, experts);
    defer m.gpa.free(codes);
    var scales = try m.gpa.alloc(A, experts);
    defer m.gpa.free(scales);
    var made: usize = 0;
    defer for (0..made) |e| {
        _ = mlx.mlx_array_free(codes[e]);
        _ = mlx.mlx_array_free(scales[e]);
    };
    for (0..experts) |e| {
        var b1: NameBuf = undefined;
        var b2: NameBuf = undefined;
        const wname = try key(&b1, "{s}.{d}.ffn.experts.{d}." ++ proj ++ ".weight", .{ pfx, id, e });
        const sname = try key(&b2, "{s}.{d}.ffn.experts.{d}." ++ proj ++ ".scale", .{ pfx, id, e });
        const wt = try getW(w, wname);
        const sc_ = try getW(w, sname);
        if (mlx.mlx_array_dtype(wt) != .int8 or dim(wt, 0) != rows or dim(wt, 1) * 2 != cols or dim(sc_, 0) != rows or dim(sc_, 1) * 32 != cols) {
            log.err("deepseek_v41: {s} is not a {d}x{d} fp4 expert with an e8m0 scale per 32\n", .{ wname, rows, cols });
            return error.Dsv41WeightLayout;
        }
        codes[e] = mlx.mlx_array_new();
        scales[e] = mlx.mlx_array_new();
        made = e + 1;
        try mlx.check(mlx.mlx_view(&codes[e], wt, .uint32, m.s));
        try mlx.check(mlx.mlx_array_set(&scales[e], sc_));
    }
    var out: Bank = .{ .w = undefined, .s = undefined };
    inline for (.{ .{ "w", codes }, .{ "s", scales } }) |f| {
        const vec = mlx.mlx_vector_array_new_data(f[1].ptr, f[1].len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_stack_axis(&o, vec, 0, m.s));
        @field(out, f[0]) = try m.own(o);
        try mlx.check(mlx.mlx_array_eval(o));
    }
    for (0..experts) |e| {
        var b1: NameBuf = undefined;
        var b2: NameBuf = undefined;
        source.drop(w, try key(&b1, "{s}.{d}.ffn.experts.{d}." ++ proj ++ ".weight", .{ pfx, id, e }));
        source.drop(w, try key(&b2, "{s}.{d}.ffn.experts.{d}." ++ proj ++ ".scale", .{ pfx, id, e }));
    }
    return out;
}

/// `bias` `[1, E]` with -inf on the experts `drop` leaves out of layer `layer`: the router picks on score plus bias and
/// weights the picks by the unbiased score, so the kept picks and their weights are the normal path's.
fn maskDropped(m: *Model, bias: A, drop: *const expert_usage.Drop, layer: usize, experts: usize) !A {
    var mask: [expert_usage.max_experts]f32 = undefined;
    for (mask[0..experts], 0..) |*v, e| v.* = if (drop.isDropped(layer, e)) -std.math.inf(f32) else 0;
    const arr = mlx.mlx_array_new_data(&mask, &[_]c_int{ 1, ci(experts) }, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, bias, arr, m.s));
    _ = try m.own(o);
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

/// Layer `li` of `m.layers` from `<pfx>.<id>.*`: backbone layer `id == li` under `layers`, or DSpark stage `id`.
fn loadLayer(m: *Model, w: *model_mod.Weights, pfx: []const u8, id: usize, li: usize) !Layer {
    const c = &m.cfg;
    const act = m.act;
    var b: NameBuf = undefined;
    var base: NameBuf = undefined;
    const og: c_int = ci(c.o_groups);
    const ol: c_int = ci(c.o_lora);
    const r = c.ratios[li];
    const stage = li >= c.layers;
    const experts: usize = if (stage) c.dspark.experts else c.experts;

    var wo_a = try loadLin(m, w, try lkey(&base, pfx, id, "attn.wo_a"), act);
    if (wo_a.stq) |q| {
        if (dim(q.p.trellis, 0) != og or q.outDim() != ol) return error.Dsv41WeightLayout;
    } else {
        wo_a.w = try m.ownReshape(wo_a.w, &.{ og, ol, ci(dim(wo_a.w, 1)) });
        if (wo_a.s.ctx != null) wo_a.s = try m.ownReshape(wo_a.s, &.{ og, ol, ci(dim(wo_a.s, 1)) });
    }

    const sink = try loadDense(m, w, try lkey(&b, pfx, id, "attn.attn_sink"), .float32);
    var ly: Layer = .{
        .attn_norm = try loadDense(m, w, try lkey(&b, pfx, id, "attn_norm.weight"), act),
        .ffn_norm = try loadDense(m, w, try lkey(&b, pfx, id, "ffn_norm.weight"), act),
        .q_norm = try loadDense(m, w, try lkey(&b, pfx, id, "attn.q_norm.weight"), act),
        .kv_norm = try loadDense(m, w, try lkey(&b, pfx, id, "attn.kv_norm.weight"), act),
        .hc_attn = try loadHc(m, w, pfx, id, "attn"),
        .hc_ffn = try loadHc(m, w, pfx, id, "ffn"),
        .wq_a = try loadLin(m, w, try lkey(&base, pfx, id, "attn.wq_a"), act),
        .wq_b = try loadLin(m, w, try lkey(&base, pfx, id, "attn.wq_b"), act),
        .wkv = try loadLin(m, w, try lkey(&base, pfx, id, "attn.wkv"), act),
        .wo_a = wo_a,
        .wo_b = try loadLin(m, w, try lkey(&base, pfx, id, "attn.wo_b"), act),
        .sink = try m.ownReshape(sink, &.{ 1, ci(c.heads), 1 }),
        .ratio = r,
        .comp = null,
        .idx = null,
        .gate_w = try loadDense(m, w, try lkey(&b, pfx, id, "ffn.gate.weight"), .float32),
        .gate_bias = undefined,
        .shared = .{
            try loadLin(m, w, try lkey(&base, pfx, id, "ffn.shared_experts.w1"), act),
            try loadLin(m, w, try lkey(&base, pfx, id, "ffn.shared_experts.w3"), act),
            try loadLin(m, w, try lkey(&base, pfx, id, "ffn.shared_experts.w2"), act),
        },
        .top_k = if (stage) c.dspark.top_k else c.top_k,
    };
    ly.gate_bias = try m.ownReshape(try loadDense(m, w, try lkey(&b, pfx, id, "ffn.gate.bias"), .float32), &.{ 1, ci(experts) });
    var vl_name: NameBuf = undefined;
    const vl_key = try lkey(&vl_name, pfx, id, "ffn.gate.bias_vl");
    if (!stage and w.get(vl_key) != null)
        ly.gate_bias_vl = try m.ownReshape(try loadDense(m, w, vl_key, .float32), &.{ 1, ci(experts) });
    if (m.drop) |d| {
        if (!stage) {
            ly.gate_bias = try maskDropped(m, ly.gate_bias, d, li, experts);
            if (ly.gate_bias_vl.ctx != null) ly.gate_bias_vl = try maskDropped(m, ly.gate_bias_vl, d, li, experts);
        }
    }
    if (c.isKvSource(li)) {
        // Pooling runs in f32, weights included (the reference promotes them); a ratio-1 latent is a plain projection.
        const pooled = r > 1;
        ly.comp = .{
            .wkv = try loadLin(m, w, try lkey(&base, pfx, id, "attn.compressor.wkv"), if (pooled) .float32 else act),
            .wgate = if (pooled) try loadLin(m, w, try lkey(&base, pfx, id, "attn.compressor.wgate"), .float32) else null,
            .norm = try loadDense(m, w, try lkey(&b, pfx, id, "attn.compressor.norm.weight"), act),
        };
    }
    if (c.isIndexSource(li)) {
        const owner = c.isKvSource(li);
        ly.idx = .{
            .wq_b = try loadLin(m, w, try lkey(&base, pfx, id, "attn.indexer.wq_b"), act),
            .weights_proj = try loadLin(m, w, try lkey(&base, pfx, id, "attn.indexer.weights_proj"), act),
            .wk = if (owner) try loadLin(m, w, try lkey(&base, pfx, id, "attn.indexer.wk"), act) else null,
            .k_norm = if (owner) try loadDense(m, w, try lkey(&b, pfx, id, "attn.indexer.k_norm.weight"), act) else none,
        };
    }
    var in_lins: [3]*Lin = .{ &ly.wq_a, &ly.wkv, undefined };
    var n_in: usize = 2;
    if (ly.comp) |*cw| if (cw.wgate == null) {
        in_lins[2] = &cw.wkv;
        n_in = 3;
    };
    ly.in_shared = try fuseShared(m, in_lins[0..n_in]);
    if (ly.comp) |*cw| if (cw.wgate) |*g| {
        cw.shared = try fuseShared(m, &.{ &cw.wkv, g });
    };
    return ly;
}

/// Stacks STQ linears that read the same rows into one bank and points each at its view of it, so one launch serves
/// them all at decode widths (`exl3.linear.linearShared`) while a wider chunk still runs each alone. Null, with nothing
/// changed, unless every one is a one-bank STQ linear of the same input width, rate and codebook, its width a multiple
/// of 128.
fn fuseShared(m: *Model, lins: []const *Lin) !?exl3.linear.Shared {
    const L = exl3.linear;
    if (lins.len < 2 or lins.len > L.SHARED_MAX) return null;
    const q0 = lins[0].stq orelse return null;
    // A copy: the shape lives in the array, which is freed below.
    const t0: [4]c_int = mlx.getShape(q0.p.trellis)[0..4].*;
    var cols: [L.SHARED_MAX + 1]c_int = @splat(0);
    for (lins, 0..) |l, i| {
        const q = l.stq orelse return null;
        const t = mlx.getShape(q.p.trellis);
        if (t.len != 4 or t[0] != 1 or t[1] != t0[1] or t[3] != t0[3] or !std.meta.eql(q.dec, q0.dec) or @rem(t[2] * 16, 128) != 0) return null;
        cols[i + 1] = cols[i] + t[2] * 16;
    }
    var parts: [3][L.SHARED_MAX]A = undefined;
    for (lins, 0..) |l, i| {
        parts[0][i] = l.stq.?.p.trellis;
        parts[1][i] = l.stq.?.p.suh;
        parts[2][i] = l.stq.?.p.svh;
    }
    var cat: [3]A = undefined;
    for (0..3) |k| {
        const v = mlx.mlx_vector_array_new_data(&parts[k], lins.len);
        defer _ = mlx.mlx_vector_array_free(v);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&o, v, @as(c_int, if (k == 0) 2 else if (k == 1) 0 else 1), m.s));
        cat[k] = try m.own(o);
    }
    try evalAll(&cat);
    const in = t0[1] * 16;
    for (lins, 0..) |l, i| {
        const c0 = cols[i];
        const c1 = cols[i + 1];
        const q = &l.stq.?;
        const old = [3]A{ q.p.trellis, q.p.suh, q.p.svh };
        q.p.trellis = try m.ownSlice(cat[0], &.{ 0, 0, @divExact(c0, 16), 0 }, &.{ 1, t0[1], @divExact(c1, 16), t0[3] });
        q.p.suh = try m.ownSlice(cat[1], &.{ @intCast(i), 0 }, &.{ @intCast(i + 1), in });
        q.p.svh = try m.ownSlice(cat[2], &.{ 0, c0 }, &.{ 1, c1 });
        for (old) |a| m.disown(a);
    }
    return .{ .p = .{ .trellis = cat[0], .suh = cat[1], .svh = cat[2] }, .cols = cols, .count = lins.len };
}

/// `x` through the stacked bank `sh`: `outs[s]` is linear s's `[rows, width]` in x's dtype, owned by `sc`.
fn sharedApply(sc: *Scope, sh: exl3.linear.Shared, dec: exl3.format.Decode, x: A, outs: []A) !void {
    try exl3.linear.linearShared(sc.s, x, sh, dec, outs);
    for (outs) |o| _ = try sc.keep(o);
}

/// Builds the model from `weights` (the resident tensors, `source.loadResident`); `dir_path` holds the
/// Engram tables. The weight map is emptied of every tensor the model takes.
pub fn init(gpa: std.mem.Allocator, cfg: Config, weights: *model_mod.Weights, dir_path: []const u8, s: S, opts: Opts) !*Model {
    const m = try gpa.create(Model);
    m.* = .{
        .gpa = gpa,
        .s = s,
        .cfg = cfg,
        .act = opts.act,
        .qat_on = opts.qat,
        .embed = .{ .w = none },
        .head = .{ .w = none },
        .norm = none,
        .layers = &.{},
        .rope_plain = .{ .freqs = none },
        .rope_yarn = .{ .freqs = none },
        .drop = opts.drop,
    };
    errdefer m.deinit();
    const c = &m.cfg;
    m.embed = try loadEmbed(m, weights);
    m.head = if (try loadStq(m, weights, "head")) |q| .{ .w = none, .stq = q } else .{ .w = try loadDense(m, weights, "head.weight", .float32) };
    m.norm = try loadDense(m, weights, "norm.weight", m.act);
    if (m.embed.vocab() != c.vocab or m.head.outDim() != c.vocab) {
        log.err("deepseek_v41: embed/head have {d} / {d} rows, config says vocab {d}\n", .{ m.embed.vocab(), m.head.outDim(), c.vocab });
        return error.Dsv41WeightLayout;
    }
    var freqs: [256]f32 = undefined;
    const half = c.rope_dim / 2;
    qat_mod.invFreqs(freqs[0..half], c.rope_dim, 0, c.rope_theta, 1, c.beta_fast, c.beta_slow);
    m.rope_plain.freqs = try m.own(mlx.mlx_array_new_data(&freqs, &[_]c_int{ci(half)}, 1, .float32));
    qat_mod.invFreqs(freqs[0..half], c.rope_dim, c.yarn_orig, c.compress_rope_theta, c.yarn_factor, c.beta_fast, c.beta_slow);
    m.rope_yarn.freqs = try m.own(mlx.mlx_array_new_data(&freqs, &[_]c_int{ci(half)}, 1, .float32));

    m.layers = try gpa.alloc(Layer, c.layers);
    @memset(m.layers, std.mem.zeroes(Layer));
    for (0..c.layers) |li| m.layers[li] = try loadLayer(m, weights, "layers", li, li);
    m.drop = null;
    if (opts.experts) {
        const banks = try gpa.alloc(BankSet, c.layers);
        m.experts = .{ .resident = banks };
        for (banks, 0..) |*bs, li| {
            bs.gate = try loadBank(m, weights, "layers", li, "w1", c.experts, c.moe_inter, c.dim);
            bs.up = try loadBank(m, weights, "layers", li, "w3", c.experts, c.moe_inter, c.dim);
            bs.down = try loadBank(m, weights, "layers", li, "w2", c.experts, c.dim, c.moe_inter);
        }
    }
    if (opts.dspark) try initDspark(m, weights);
    if (c.engram.n > 0) try initEngram(m, weights, dir_path);
    if (mlx.streamIsGpu(s)) try kernels.probeHc(gpa, s, c.hc_mult, c.dim, c.hc_iters);
    m.ced_split = c.cedSplit();
    return m;
}

fn initEngram(m: *Model, w: *model_mod.Weights, dir_path: []const u8) !void {
    const c = &m.cfg;
    const e = c.engram;
    m.token_map = try engram.loadTokenMap(m.gpa, dir_path, c.vocab, e.compressed_vocab);
    if (e.pad_id >= m.token_map.len) return error.EngramTokenMap;
    m.hash = try engram.Hash.init(e, m.token_map[e.pad_id]);
    m.engrams = try m.gpa.alloc(EngramLayer, e.n);
    var opened: usize = 0;
    errdefer for (m.engrams[0..opened]) |*t| closeEngram(t);
    var b: NameBuf = undefined;
    var base: NameBuf = undefined;
    for (0..e.n) |i| {
        const l = e.layers[i];
        const wkv = try loadLin(m, w, try key(&base, "layers.{d}.engram.wkv", .{l}), m.act);
        const qw = try loadDense(m, w, try key(&b, "layers.{d}.engram.q_weight", .{l}), .float32);
        const kw = try loadDense(m, w, try key(&b, "layers.{d}.engram.k_weight", .{l}), .float32);
        var prod = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&prod, qw, kw, m.s));
        _ = try m.own(prod);
        try mlx.check(mlx.mlx_array_eval(prod));
        var table = try engram.Table.open(m.gpa, dir_path, l, e.rows[i], e.head_dim);
        errdefer table.close();
        const rows = @min(engram.cache_rows, e.rows[i]);
        if (engram.RowCache.bytesFor(rows, e.head_dim, e.head_dim / 32) > 0) table.cache = try engram.RowCache.create(m.gpa, rows, e.head_dim, e.head_dim / 32);
        m.engrams[i] = .{ .layer = l, .wkv = wkv, .qk = prod, .table = table };
        opened = i + 1;
    }
}

fn closeEngram(e: *EngramLayer) void {
    if (e.table.cache) |c| c.destroy();
    e.table.close();
}

pub fn newState(m: *const Model) !State {
    const c = &m.cfg;
    const layers = try m.gpa.alloc(LState, c.layers);
    var made: usize = 0;
    errdefer {
        for (layers[0..made]) |*l| l.deinit();
        m.gpa.free(layers);
    }
    // With DSpark on, the rings hold the rows of a verify chunk beyond the window, so a rejected row never
    // overwrites a row a later query still reads.
    const slack: usize = if (m.dspark != null) cfg_mod.ring_slack else 0;
    for (layers, 0..) |*l, li| {
        l.* = .{ .win = try Ring.init(c.window + slack, c.head_dim, m.act, m.s) };
        made += 1;
        // With the round trips on, the rows are FP4 values and are kept in that format.
        const packed_kv = m.qat_on and m.pack_kv;
        if (m.isKvSource(li)) l.comp = .{ .width = c.head_dim, .dtype = m.act, .pack = if (packed_kv) .{ .mode = .fp4_e4m3, .group = 16 } else null };
        if (c.isOwner(li)) l.idx_k = .{ .width = c.index_head_dim, .dtype = m.act, .pack = if (packed_kv) .{ .mode = .fp4_ue8m0, .group = 32 } else null };
    }
    var ds: []Ring = &.{};
    var rings: usize = 0;
    errdefer {
        for (ds[0..rings]) |*r| r.deinit();
        m.gpa.free(ds);
    }
    if (m.dspark != null) {
        ds = try m.gpa.alloc(Ring, c.n_mtp);
        for (ds) |*r| {
            r.* = try Ring.init(c.window, c.head_dim, m.act, m.s);
            rings += 1;
        }
    }
    return .{ .gpa = m.gpa, .layers = layers, .ds = ds };
}

// ── hyper-connections ────────────────────────────────────────────────────

/// Collapse the hc copies `[L, hc, D]` with `pre` `[L, hc]`, in f32.
pub fn hcCollapse(sc: *Scope, x: A, pre: A) !A {
    const y = try sc.matmul(try sc.expand(pre, 1), try sc.astype(x, .float32)); // [L, 1, D]
    return sc.astype(try sc.reshape(y, &.{ ci(dim(x, 0)), ci(dim(x, 2)) }), mlx.mlx_array_dtype(x));
}

/// `out[k] = post[k] * y + sum_j comb[j, k] * x[j]` — the new stream.
pub fn hcExpand(sc: *Scope, x: A, y: A, post: A, comb: A) !A {
    const xf = try sc.astype(x, .float32);
    const branch = try sc.mul(try sc.expand(post, 2), try sc.expand(try sc.astype(y, .float32), 1));
    const resid = try sc.matmul(try sc.transpose(comb, &.{ 0, 2, 1 }), xf);
    return sc.astype(try sc.add(branch, resid), mlx.mlx_array_dtype(x));
}

/// The pre / post / comb coefficients of the stream `x`, from one projection of its flattened,
/// rms-normalized copies.
fn hcMixes(sc: *Scope, m: *const Model, x: A, hc: *const Hc) !qat_mod.Split {
    const L = dim(x, 0);
    const flat = try sc.reshape(try sc.astype(x, .float32), &.{ ci(L), ci(dim(x, 1) * dim(x, 2)) });
    const rs = try sc.op1(mlx.mlx_rsqrt, try sc.add(try sc.mean(try sc.op1(mlx.mlx_square, flat), -1, true), try sc.f(m.cfg.eps)));
    const mixes = try sc.mul(try sc.matmul(flat, try sc.t2(hc.fn_w)), rs);
    return qat_mod.hcSplit(sc, mixes, hc.scale, hc.base, m.cfg.hc_iters, m.cfg.hc_eps);
}

// ── attention ────────────────────────────────────────────────────────────

fn posArray(sc: *Scope, first: usize, n: usize, step: usize) !A {
    const host = try sc.a.alloc(i32, n);
    defer sc.a.free(host);
    for (host, 0..) |*v, i| v.* = @intCast(first + i * step);
    return sc.ints(host, &.{ci(n)});
}

fn qatIf(sc: *Scope, m: *const Model, x: A, mode: qat_mod.Qat, group: usize) !A {
    return if (m.qat_on) qat_mod.qat(sc, x, mode, group) else x;
}

/// The tail of `x` rotated at `pos` (`inverse` turns it back) and, with the QAT round trips on, `qat`'s round trip,
/// in `x`'s dtype or `out_dt`: one kernel, or the op chain.
fn ropeQ(sc: *Scope, m: *const Model, x: A, freqs: A, pos: A, inverse: bool, qat: kernels.RopeQat, out_dt: ?mlx.mlx_dtype) !A {
    const rd = m.cfg.rope_dim;
    if (fusedOn(sc, m)) return kernels.ropeTail(sc, x, freqs, pos, rd, inverse, if (m.qat_on) qat else .none, out_dt);
    const r = try qat_mod.rotate(sc, x, freqs, pos, rd, inverse);
    const q = switch (qat) {
        .none => r,
        .fp8_ue8m0 => try qatIf(sc, m, r, .fp8_ue8m0, 32),
        .fp4_ue8m0 => try qatIf(sc, m, r, .fp4_ue8m0, 32),
        .fp4_e4m3 => try qatIf(sc, m, r, .fp4_e4m3, 16),
    };
    return if (out_dt) |dt| sc.astype(q, dt) else q;
}

/// `x` (f32) through `l`: a dense weight at decode width is summed in f32 by the kernel, as the reference does after
/// widening it; MLX's f32 matmul carries noise of its own.
fn denseLin(sc: *Scope, m: *const Model, l: *const Lin, x: A) !A {
    if (fusedOn(sc, m) and l.stq == null and l.s.ctx == null and dim(x, 0) <= NARROW_ROWS and kernels.denseF32xFits(dim(l.w, 0), dim(l.w, 1)))
        return kernels.denseF32x(sc, x, l.w);
    return l.apply(sc, x);
}

/// A KV source's compressor: pools each completed group of `ratio` tokens into one latent (a softmax
/// gate in f32 above ratio 1), publishes index keys from the latent before rope when this layer owns
/// them, then ropes the latent at its group's first position and stores it fp4-simulated.
/// `kv_in`: a ratio-1 compressor's projection of `x` when the stacked attention bank made it, else none.
fn compress(sc: *Scope, m: *const Model, st: *State, li: usize, x: A, p0: usize, kv_in: A) !void {
    const c = &m.cfg;
    const ly = &m.layers[li];
    const cw = &ly.comp.?;
    const ls = &st.layers[li];
    const r: usize = ly.ratio;
    var latent: A = undefined;
    var g0: usize = p0;
    var groups: usize = dim(x, 0);
    if (r == 1) {
        latent = try sc.rms(if (kv_in.ctx != null) kv_in else try cw.wkv.apply(sc, x), cw.norm, c.eps);
    } else {
        const hd = c.head_dim;
        const xf = try sc.astype(x, .float32);
        var proj: [2]A = undefined;
        if (cw.shared != null and dim(xf, 0) <= exl3.kernels.PROJECT_ROWS_MAX)
            try sharedApply(sc, cw.shared.?, cw.wkv.stq.?.dec, xf, &proj)
        else
            proj = .{ try denseLin(sc, m, &cw.wkv, xf), try denseLin(sc, m, &cw.wgate.?, xf) };
        var kv = proj[0];
        var gate = proj[1];
        var pend_rows: usize = 0;
        if (ls.pend_kv.ctx != null) {
            pend_rows = dim(ls.pend_kv, 0);
            kv = try sc.concat(&.{ ls.pend_kv, kv }, 0);
            gate = try sc.concat(&.{ ls.pend_gate, gate }, 0);
        }
        const total = dim(kv, 0);
        groups = total / r;
        const full = groups * r;
        const new_kv = if (full < total) sc.out(try sc.slice(kv, 0, full, total)) else none;
        const new_gate = if (full < total) sc.out(try sc.slice(gate, 0, full, total)) else none;
        if (st.verify) |*v| {
            v.pend_kv[li] = sc.out(kv);
            v.pend_gate[li] = sc.out(gate);
        }
        if (ls.pend_kv.ctx != null) _ = mlx.mlx_array_free(ls.pend_kv);
        if (ls.pend_gate.ctx != null) _ = mlx.mlx_array_free(ls.pend_gate);
        ls.pend_kv = new_kv;
        ls.pend_gate = new_gate;
        if (groups == 0) return;
        g0 = (p0 - pend_rows) / r;
        const kg = try sc.reshape(try sc.slice(kv, 0, 0, full), &.{ ci(groups), ci(r), ci(hd) });
        const gg = try sc.reshape(try sc.slice(gate, 0, 0, full), &.{ ci(groups), ci(r), ci(hd) });
        const pooled = try sc.sum(try sc.mul(kg, try sc.softmax(gg, 1)), 1, false);
        latent = try sc.rms(try sc.astype(pooled, m.act), cw.norm, c.eps);
    }
    const pos = try posArray(sc, g0 * r, groups, r);
    const rope = m.rope_yarn.freqs;
    if (ly.idx) |*iw| if (iw.wk) |*wk| {
        var k = try sc.rms(try wk.apply(sc, latent), iw.k_norm, c.eps);
        k = try ropeQ(sc, m, k, rope, pos, false, .fp4_ue8m0, null);
        try ls.idx_k.?.append(sc, k);
    };
    const rot = try ropeQ(sc, m, latent, rope, pos, false, .fp4_e4m3, null);
    try ls.comp.?.append(sc, rot);
}

/// The `k` largest of each row of `x` as int32 indices: ascending with ties to the lower index on the kernel, else
/// unordered.
fn topkRows(sc: *Scope, m: *const Model, x: A, k: usize) !A {
    if (fusedOn(sc, m) and mlx.mlx_array_ndim(x) == 2) return kernels.topkIndices(sc, try sc.astype(x, .float32), k);
    return sc.astype(try sc.topkIdx(x, k), .int32);
}

/// Level one of the two-level top-k: the `kb` highest-scoring blocks of `bs` index groups per query
/// (`scores` `[L, nb]`, unreachable groups -inf; `lens` `[L, 1]` int32 the groups each query sees),
/// the block holding the query's newest group pinned. A bool mask over `nb`, or none when every
/// block is kept.
fn candidateBlocks(sc: *Scope, m: *const Model, li: usize, scores: A, lens: A, bs: usize, kb: usize) !A {
    if ((dim(scores, 1) + bs - 1) / bs <= kb) return none;
    return candidateMask(sc, m, li, try blockMax(sc, scores, bs), lens, bs, kb);
}

/// The highest score of each block of `bs` keys of `scores` `[L, nb]`: `[L, ceil(nb / bs)]`.
fn blockMax(sc: *Scope, scores: A, bs: usize) !A {
    const L = dim(scores, 0);
    const nb = dim(scores, 1);
    const n_blocks = (nb + bs - 1) / bs;
    var padded = scores;
    if (n_blocks * bs > nb) padded = try sc.concat(&.{ scores, try sc.broadcast(try sc.f(-std.math.inf(f32)), &.{ ci(L), ci(n_blocks * bs - nb) }) }, -1);
    return sc.max(try sc.reshape(padded, &.{ ci(L), ci(n_blocks), ci(bs) }), -1, false);
}

/// `candidateBlocks` from the block maxima `blocks_in` `[L, n_blocks]`, more than `kb` of them.
fn candidateMask(sc: *Scope, m: *const Model, li: usize, blocks_in: A, lens: A, bs: usize, kb: usize) !A {
    const L = dim(blocks_in, 0);
    const n_blocks = dim(blocks_in, 1);
    const ninf = try sc.f(-std.math.inf(f32));
    var blocks = blocks_in;
    const last = try sc.op1(mlx.mlx_floor, try sc.div(try sc.astype(try sc.sub(lens, try sc.i(1)), .float32), try sc.f(@floatFromInt(bs))));
    const bidx = try sc.astype(try sc.reshape(try sc.arange(0, n_blocks), &.{ 1, ci(n_blocks) }), .float32);
    blocks = try sc.where(try sc.op2(mlx.mlx_equal, bidx, last), try sc.f(std.math.inf(f32)), blocks);
    const chosen = try topkRows(sc, m, blocks, kb);
    if (m.trace) |t| {
        try t.put("cand_scores_{d}", .{li}, blocks);
        try t.put("cand_idx_{d}", .{li}, chosen);
    }
    const reach = try sc.op2(mlx.mlx_greater, try sc.takeAlong(blocks, chosen, -1), ninf);
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_put_along_axis(&o, try sc.zeros(&.{ ci(L), ci(n_blocks) }, .bool_), chosen, reach, -1, sc.s), &o);
}

/// The indexer's scores `[Ls, nb]` of `q` (f32 `[Ls, ih, ihd]`) against `keys`: sum over heads of the weight (the
/// projection `w` times `wscale`) times relu(q . k), -inf past the groups a query sees. A kernel for a few rows, the
/// matrix kernel for a wide block, else the op chain over key tiles.
fn indexScores(sc: *Scope, m: *const Model, q: A, keys: A, w: A, wscale: f32, p0: usize, r: usize) !A {
    const Ls = dim(q, 0);
    const nb = dim(keys, 0);
    const ih = dim(q, 1);
    const ihd = dim(q, 2);
    if (fusedOn(sc, m) and nb > 0) {
        if (Ls <= NARROW_ROWS and ih * ihd <= 8192) return kernels.indexScores(sc, q, keys, w, wscale, p0, r);
        if (kernels.indexTileFits(ih, ihd, nb)) return kernels.indexScoresTile(sc, q, keys, w, wscale, p0, r, 0);
    }
    const weights = try sc.mul(try sc.astype(w, .float32), try sc.f(wscale));
    const keysf = try sc.astype(keys, .float32);
    const q2 = try sc.reshape(q, &.{ ci(Ls * ih), ci(ihd) });
    const zero = try sc.f(0);
    var tiles: std.ArrayList(A) = .empty;
    defer tiles.deinit(sc.a);
    var t0: usize = 0;
    while (t0 < nb) : (t0 += INDEX_TILE) {
        const t1 = @min(nb, t0 + INDEX_TILE);
        const dots = try sc.reshape(try sc.matmul(q2, try sc.t2(try sc.slice(keysf, 0, t0, t1))), &.{ ci(Ls), ci(ih), ci(t1 - t0) });
        const rel = try sc.maximum(dots, zero);
        try tiles.append(sc.a, try sc.sum(try sc.mul(rel, try sc.expand(weights, 2)), 1, false));
    }
    const scores = try sc.concat(tiles.items, -1);
    const lens_h = try sc.a.alloc(i32, Ls);
    defer sc.a.free(lens_h);
    for (lens_h, 0..) |*v, i| v.* = @intCast((p0 + i + 1) / r);
    const lens = try sc.ints(lens_h, &.{ ci(Ls), 1 });
    return sc.where(try sc.lt(try sc.reshape(try sc.arange(0, nb), &.{ 1, ci(nb) }), lens), scores, try sc.f(-std.math.inf(f32)));
}

/// What `indexStream` returns: the top-k key ids per query (ascending) and the candidate-block mask, or none.
const Streamed = struct { idx: A, cand: A };

/// A running top-k of `[L, k]` ids and values, folded over tiles of scores whose columns arrive in id order.
const Running = struct {
    ids: A = none,
    vals: A = none,

    fn free(self: *Running) void {
        if (self.ids.ctx != null) _ = mlx.mlx_array_free(self.ids);
        if (self.vals.ctx != null) _ = mlx.mlx_array_free(self.vals);
        self.* = .{};
    }

    /// The running set with the top-`k` of `scores` (columns are ids `base` on) folded in, as handles outliving `ts`.
    /// The running ids all precede the tile's and both lists ascend, so a tie on the merged row goes to the lower id
    /// as on the whole row.
    fn folded(self: *const Running, ts: *Scope, scores: A, base: usize, k: usize) !Running {
        const local = try kernels.topkIndices(ts, scores, @min(k, dim(scores, 1)));
        var ids = try ts.add(local, try ts.i(ci(base)));
        var vals = try ts.takeAlong(scores, local, -1);
        if (self.ids.ctx != null) {
            ids = try ts.concat(&.{ self.ids, ids }, -1);
            vals = try ts.concat(&.{ self.vals, vals }, -1);
            const keep = try kernels.topkIndices(ts, vals, @min(k, dim(vals, 1)));
            ids = try ts.takeAlong(ids, keep, -1);
            vals = try ts.takeAlong(vals, keep, -1);
        }
        return .{ .ids = ts.out(ids), .vals = ts.out(vals) };
    }
};

/// The `k` best keys per query of `nb` index keys, scored `tile` keys at a time by the matrix kernel and folded into a
/// running top-k, so no `[Ls, nb]` scores exist: the picks are the whole row's, bit for bit. `mask` (a candidate-block
/// mask or none) keeps only its blocks of `bs` keys. With `kb` > 0 it also returns this layer's candidate mask, the
/// same fold over the block maxima (`lens` pins each query's newest block as `candidateMask` does).
fn indexStream(sc: *Scope, rows: *const Rows, nb: usize, q: A, w: A, wscale: f32, p0: usize, r: usize, k: usize, tile: usize, mask: A, bs: usize, kb: usize, lens: A) !Streamed {
    std.debug.assert(tile % kernels.indexKeyTile(rows.dtype) == 0 and (bs == 0 or tile % bs == 0));
    const n_blocks = if (bs > 0) (nb + bs - 1) / bs else 0;
    var picks = Running{};
    defer picks.free();
    var blocks = Running{};
    defer blocks.free();
    var t0: usize = 0;
    while (t0 < nb) : (t0 += tile) {
        const t1 = @min(nb, t0 + tile);
        var ts = Scope.init(sc.a, sc.s);
        defer ts.deinit();
        var scores = try kernels.indexScoresTile(&ts, q, try rows.rowsIn(&ts, t0, t1), w, wscale, p0, r, t0);
        var next_blocks = Running{};
        errdefer next_blocks.free();
        if (kb > 0) {
            var piece = try blockMax(&ts, scores, bs);
            const last = try ts.op1(mlx.mlx_floor, try ts.div(try ts.astype(try ts.sub(lens, try ts.i(1)), .float32), try ts.f(@floatFromInt(bs))));
            const bidx = try ts.astype(try ts.reshape(try ts.arange(t0 / bs, t0 / bs + dim(piece, 1)), &.{ 1, ci(dim(piece, 1)) }), .float32);
            piece = try ts.where(try ts.op2(mlx.mlx_equal, bidx, last), try ts.f(std.math.inf(f32)), piece);
            next_blocks = try blocks.folded(&ts, piece, t0 / bs, kb);
        }
        if (mask.ctx != null) {
            var o = mlx.mlx_array_new();
            const rep = try ts.res(mlx.mlx_repeat_axis(&o, try ts.slice(mask, -1, t0 / bs, (t1 + bs - 1) / bs), ci(bs), -1, ts.s), &o);
            scores = try ts.where(try ts.slice(rep, -1, 0, t1 - t0), scores, try ts.f(-std.math.inf(f32)));
        }
        var next = try picks.folded(&ts, scores, t0, k);
        errdefer next.free();
        const done = [_]A{ next.ids, next.vals, next_blocks.ids, next_blocks.vals };
        try evalAll(if (kb > 0) &done else done[0..2]);
        picks.free();
        picks = next;
        if (kb > 0) {
            blocks.free();
            blocks = next_blocks;
        }
    }
    const idx = try sc.keep(picks.ids);
    picks.ids = none;
    if (kb == 0) return .{ .idx = idx, .cand = none };
    const reach = try sc.op2(mlx.mlx_greater, blocks.vals, try sc.f(-std.math.inf(f32)));
    var o = mlx.mlx_array_new();
    const cand = try sc.res(mlx.mlx_put_along_axis(&o, try sc.zeros(&.{ ci(dim(lens, 0)), ci(n_blocks) }, .bool_), blocks.ids, reach, -1, sc.s), &o);
    return .{ .idx = idx, .cand = cand };
}

/// The top-`index_topk` compressed groups each of a span's queries keeps (`[Ls, k]` int32, -1 = none),
/// scored against the latest index-key owner's keys. The candidate source also publishes its pinned
/// blocks in `cand`; later index sources score only inside them.
fn indexer(sc: *Scope, m: *const Model, st: *State, li: usize, x: A, qr: A, p0: usize, cand: *A) !A {
    const c = &m.cfg;
    const Ls = dim(x, 0);
    const r: usize = m.ratio(li);
    const nb = (p0 + Ls) / r;
    const ih = c.index_heads;
    const ihd = c.index_head_dim;
    const iw = &m.layers[li].idx.?;
    const owner = c.latest(li, Config.isOwner);
    const rows = &st.layers[owner].idx_k.?;
    const pos = try posArray(sc, p0, Ls, 1);
    const q = try ropeQ(sc, m, try sc.reshape(try iw.wq_b.apply(sc, qr), &.{ ci(Ls), ci(ih), ci(ihd) }), m.rope_yarn.freqs, pos, false, .fp4_ue8m0, .float32);
    const wscale = 1.0 / @sqrt(@as(f32, @floatFromInt(ihd))) / @sqrt(@as(f32, @floatFromInt(ih)));
    const wts = try iw.weights_proj.apply(sc, x);
    const lens_h = try sc.a.alloc(i32, Ls);
    defer sc.a.free(lens_h);
    for (lens_h, 0..) |*v, i| v.* = @intCast((p0 + i + 1) / r);
    const lens = try sc.ints(lens_h, &.{ ci(Ls), 1 });
    const ninf = try sc.f(-std.math.inf(f32));
    const kk = @min(c.index_topk, nb);

    const cs = c.cand_source;
    const is_source = cs >= 0 and li == @as(usize, @intCast(cs));
    const masked = cs >= 0 and li > @as(usize, @intCast(cs)) and cand.ctx != null;
    const bs = if (cs >= 0) c.cand_block_size else 0;
    var idx: A = undefined;
    if (streamsIndex(sc, m, Ls, nb)) {
        _ = index_streams.fetchAdd(1, .monotonic);
        const kb = if (is_source and (nb + bs - 1) / bs > c.cand_blocks) c.cand_blocks else 0;
        const got = try indexStream(sc, rows, nb, q, wts, wscale, p0, r, kk, m.index_stream_tile, if (masked) cand.* else none, bs, kb, lens);
        if (is_source) {
            if (cand.ctx != null) _ = mlx.mlx_array_free(cand.*);
            cand.* = if (kb > 0) sc.out(got.cand) else none;
        }
        idx = got.idx;
    } else {
        var scores = if (rows.pack != null and narrowIndex(sc, m, Ls, nb)) // [Ls, nb]
            try kernels.indexScoresPacked(sc, q, try sc.slice(rows.buf, 0, 0, nb), try sc.slice(rows.sbuf, 0, 0, nb), wts, wscale, p0, r, rows.pack.?.mode, rows.pack.?.group)
        else
            try indexScores(sc, m, q, try rows.rows(sc, nb), wts, wscale, p0, r);
        if (is_source) {
            const mask = try candidateBlocks(sc, m, li, scores, lens, bs, c.cand_blocks);
            if (cand.ctx != null) _ = mlx.mlx_array_free(cand.*);
            cand.* = if (mask.ctx != null) sc.out(mask) else none;
        } else if (masked) {
            var o = mlx.mlx_array_new();
            const rep = try sc.res(mlx.mlx_repeat_axis(&o, cand.*, ci(bs), -1, sc.s), &o);
            scores = try sc.where(try sc.slice(rep, -1, 0, nb), scores, ninf);
        }
        idx = try topkRows(sc, m, scores, kk);
        if (m.trace) |t| try t.put("index_scores_{d}", .{li}, scores);
    }
    const picked = try sc.where(try sc.lt(idx, lens), idx, try sc.i(-1));
    if (m.trace) |t| try t.put("index_idx_{d}", .{li}, picked);
    return picked;
}

/// Whether a span's index top-k streams over key tiles: the matrix kernel scores it (so a tile scores the bits of
/// the whole list's columns), the keys outnumber one tile, and no trace wants the whole scores.
fn streamsIndex(sc: *const Scope, m: *const Model, Ls: usize, nb: usize) bool {
    const c = &m.cfg;
    return fusedOn(sc, m) and m.trace == null and !narrowIndex(sc, m, Ls, nb) and nb > m.index_stream_tile and kernels.indexTileFits(c.index_heads, c.index_head_dim, nb);
}

/// Whether `indexScores` runs the scalar kernel (a few rows), which reads packed keys as they are.
fn narrowIndex(sc: *const Scope, m: *const Model, Ls: usize, nb: usize) bool {
    return fusedOn(sc, m) and nb > 0 and Ls <= NARROW_ROWS and m.cfg.index_heads * m.cfg.index_head_dim <= 8192;
}

/// One span's attention in f32: the window band over `keys_w`, then the selected compressed rows
/// (`topk` < 0 = none); the per-head sink joins the softmax denominator only. `[Ls, H, hd]` in `act`.
pub fn attnCore(sc: *Scope, sink: A, q: A, keys_w: A, band: A, comp_rows: A, topk: A, scale: f32, act: mlx.mlx_dtype) !A {
    const qf = try sc.astype(q, .float32);
    const kw = try sc.astype(keys_w, .float32);
    var logits = try sc.matmul(qf, try sc.t2(kw)); // [Ls, H, Wk]
    var mask = try sc.expand(band, 1);
    var g = none;
    if (comp_rows.ctx != null) {
        const safe = try sc.maximum(topk, try sc.i(0));
        g = try sc.astype(try sc.take(comp_rows, safe, 0), .float32); // [Ls, K, hd]
        const lc = try sc.matmul(qf, try sc.transpose(g, &.{ 0, 2, 1 })); // [Ls, H, K]
        logits = try sc.concat(&.{ logits, lc }, -1);
        mask = try sc.concat(&.{ mask, try sc.expand(try sc.ge(topk, try sc.i(0)), 1) }, -1);
    }
    logits = try sc.where(mask, try sc.mul(logits, try sc.f(scale)), try sc.f(-std.math.inf(f32)));
    const mx = try sc.maximum(try sc.max(logits, -1, true), sink);
    const p = try sc.op1(mlx.mlx_exp, try sc.sub(logits, mx));
    const denom = try sc.add(try sc.sum(p, -1, true), try sc.op1(mlx.mlx_exp, try sc.sub(sink, mx)));
    const wk = dim(keys_w, 0);
    var o = try sc.matmul(try sc.slice(p, -1, 0, wk), kw);
    if (g.ctx != null) o = try sc.add(o, try sc.matmul(try sc.slice(p, -1, wk, dim(p, -1)), g));
    return sc.astype(try sc.div(o, denom), act);
}

/// What a chunk's layers hand down, per span: the selected compressed groups and the candidate blocks.
const Shared = struct {
    topk: []A,
    cand: []A,
    /// A CED replay's first position: the decoder has no window rows before it, and its compressor already ran.
    floor: ?usize = null,
    /// `[L]` bool, the chunk's rows inside an image span (none for text): borrowed from the pass.
    img: A = none,

    fn init(gpa: std.mem.Allocator, spans: usize) !Shared {
        const t = try gpa.alloc(A, spans);
        errdefer gpa.free(t);
        const cd = try gpa.alloc(A, spans);
        @memset(t, none);
        @memset(cd, none);
        return .{ .topk = t, .cand = cd };
    }

    fn deinit(self: *Shared, gpa: std.mem.Allocator) void {
        for (self.topk) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        for (self.cand) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        gpa.free(self.topk);
        gpa.free(self.cand);
    }
};

const Picks = struct { comp_rows: A = none, topk: A = none };

/// The compressed rows span `si` (chunk rows [a, b)) attends to and the groups it picked: the indexer's top-k
/// when this layer owns one, else the one a layer above handed down (`sh`).
fn compressedPicks(sc: *Scope, m: *const Model, st: *State, li: usize, h: A, qr: A, p0: usize, a: usize, b: usize, sh: *Shared, si: usize) !Picks {
    const c = &m.cfg;
    const ly = &m.layers[li];
    const r = ly.ratio;
    if (r == 0) return .{};
    const clen = (p0 + b) / r;
    if (ly.idx != null and clen > 0) {
        const sp = try indexer(sc, m, st, li, try sc.slice(h, 0, a, b), try sc.slice(qr, 0, a, b), p0 + a, &sh.cand[si]);
        if (sh.topk[si].ctx != null) _ = mlx.mlx_array_free(sh.topk[si]);
        sh.topk[si] = sc.out(sp);
    }
    if (clen == 0 or sh.topk[si].ctx == null) return .{};
    const src = c.latest(li, Config.isKvSource);
    const comp = &st.layers[src].comp.?;
    if (comp.pack == null) return .{ .topk = sh.topk[si], .comp_rows = try comp.rows(sc, clen) };
    // Packed rows: only the picks are unpacked, and the picks then index them in order.
    const topk = sh.topk[si];
    const K = dim(topk, 1);
    const n = dim(topk, 0) * K;
    const picked = try comp.gather(sc, try sc.reshape(try sc.maximum(topk, try sc.i(0)), &.{ci(n)}));
    const order = try sc.astype(try sc.reshape(try sc.arange(0, n), mlx.getShape(topk)), mlx.mlx_array_dtype(topk));
    return .{ .topk = try sc.where(try sc.ge(topk, try sc.i(0)), order, topk), .comp_rows = picked };
}

/// The window rows of layer `ly` for `proj` (`wkv` of the normed input, `[L, head_dim]`) at `pos`: rms-normed, the tail
/// rotated, and with the QAT round trips on the fp8 round trip.
fn windowRow(sc: *Scope, m: *const Model, ly: *const Layer, proj: A, rope: A, pos: A) !A {
    const c = &m.cfg;
    if (fusedOn(sc, m) and kernels.kvRowFits(c.head_dim)) return kernels.kvRow(sc, proj, ly.kv_norm, rope, pos, c.rope_dim, c.eps, m.qat_on);
    return ropeQ(sc, m, try sc.rms(proj, ly.kv_norm, c.eps), rope, pos, false, .fp8_ue8m0, null);
}

/// The attention sublayer over a chunk: `h` the normed input `[L, dim]` at positions [p0, p0 + L).
fn attention(sc: *Scope, m: *const Model, st: *State, li: usize, h: A, p0: usize, sh: *Shared, span: usize) !A {
    const c = &m.cfg;
    const ly = &m.layers[li];
    const ls = &st.layers[li];
    const L = dim(h, 0);
    const hd = c.head_dim;
    const H = c.heads;
    const rd = c.rope_dim;
    const rope = m.ropeFor(li).freqs;
    const pos = try posArray(sc, p0, L, 1);

    const fus = fusedOn(sc, m);
    const pf = prof(m, L);
    const in = try attnIn(sc, m, ly, h);
    if (pf) |p| try p.mark(.attn_in, in[0..if (in[2].ctx != null) 3 else 2]);
    const qr = try sc.rms(in[0], ly.q_norm, c.eps);
    const q = try ropeQ(sc, m, try sc.reshape(try ly.wq_b.apply(sc, qr), &.{ ci(L), ci(H), ci(hd) }), rope, pos, false, .none, null);
    if (pf) |p| try p.mark(.q, &.{ qr, q });
    const kv = try windowRow(sc, m, ly, in[1], rope, pos);
    if (pf) |p| try p.mark(.kv_row, &.{kv});

    if (ly.comp != null and sh.floor == null) try compress(sc, m, st, li, h, p0, in[2]);
    if (pf) |p| {
        var ev: std.ArrayList(A) = .empty;
        defer ev.deinit(sc.a);
        try appendStateArrays(sc.a, &ev, ls);
        try p.mark(.compress, ev.items);
    }

    const window = c.window;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    if (fus and L <= NARROW_ROWS and rd > 0 and kernels.attnDecodeFits(hd) and ls.win.cap >= window + L - 1) {
        // The ring holds this chunk's rows before the kernel reads them; the heads leave inverse-rotated.
        try ls.win.write(kv, p0, sc.s);
        const picks = try compressedPicks(sc, m, st, li, h, qr, p0, 0, L, sh, 0);
        if (pf) |p| try p.mark(.indexer, if (picks.comp_rows.ctx != null) &.{ picks.comp_rows, picks.topk } else &.{ls.win.buf});
        const o = try kernels.attnDecode(sc, .{
            .q = q,
            .ring = ls.win.buf,
            .comp = picks.comp_rows,
            .topk = picks.topk,
            .sink = ly.sink,
            .p0 = p0,
            .window = window,
            .scale = scale,
            .freqs = rope,
            .rd = rd,
            .out_dt = m.act,
        });
        if (pf) |p| try p.mark(.core, &.{o});
        return attnOut(sc, m, ly, o);
    }
    const n_spans = (L + span - 1) / span;
    const heads = try sc.a.alloc(A, n_spans);
    defer sc.a.free(heads);
    for (0..n_spans) |si| {
        // A span's intermediates (the gathered rows above all) are released once its heads are made, not held
        // to the layer's end.
        var ssc = Scope.init(sc.a, sc.s);
        defer ssc.deinit();
        const a = si * span;
        const b = @min(L, a + span);
        const Ls = b - a;
        const lo = @max(sh.floor orelse 0, if (p0 + a >= window - 1) p0 + a - (window - 1) else 0);
        const wk = p0 + b - lo;
        var parts: [2]A = undefined;
        var np: usize = 0;
        if (lo < p0) {
            parts[np] = try ls.win.read(&ssc, lo, p0);
            np += 1;
        }
        parts[np] = try ssc.slice(kv, 0, if (lo > p0) lo - p0 else 0, b);
        np += 1;
        const keys_w = try ssc.concat(parts[0..np], 0);
        const kpos = try ssc.add(try ssc.reshape(try ssc.arange(0, wk), &.{ 1, ci(wk) }), try ssc.i(ci(lo)));
        const qpos = try ssc.add(try ssc.reshape(try ssc.arange(0, Ls), &.{ ci(Ls), 1 }), try ssc.i(ci(p0 + a)));
        const band = try ssc.op2(mlx.mlx_logical_and, try ssc.op2(mlx.mlx_less_equal, kpos, qpos), try ssc.op2(mlx.mlx_greater, kpos, try ssc.sub(qpos, try ssc.i(ci(window)))));

        const picks = try compressedPicks(&ssc, m, st, li, h, qr, p0, a, b, sh, si);
        if (pf) |p| {
            const got = [_]A{ keys_w, picks.comp_rows, picks.topk };
            try p.mark(.indexer, got[0..if (picks.comp_rows.ctx != null) 3 else 1]);
        }
        const K = if (picks.comp_rows.ctx != null) dim(picks.topk, 1) else 0;
        const head = if (fus and L > NARROW_ROWS and mlx.mlx_array_dtype(q) == .bfloat16 and kernels.attnFusedFits(H, hd))
            try kernels.attnFused(&ssc, ly.sink, try ssc.slice(q, 0, a, b), keys_w, p0 + a - lo, window, picks.comp_rows, picks.topk, scale, m.act)
        else if (m.attn_f16 and fus and (m.act == .bfloat16 or m.act == .float16) and wk + K <= kernels.attn_softmax_max_cols)
            try kernels.attnWide(&ssc, ly.sink, try ssc.slice(q, 0, a, b), keys_w, p0 + a - lo, window, picks.comp_rows, picks.topk, scale, m.act)
        else
            try attnCore(&ssc, ly.sink, try ssc.slice(q, 0, a, b), keys_w, band, picks.comp_rows, picks.topk, scale, m.act);
        heads[si] = try sc.keep(ssc.out(head));
        if (L > NARROW_ROWS) try submit(heads[si]);
        if (pf) |p| try p.mark(.core, &.{heads[si]});
        if (m.trace) |t| if (si == 0) {
            try t.put("sa_q_{d}", .{li}, try ssc.slice(q, 0, a, b));
            try t.put("sa_kv_{d}", .{li}, keys_w);
            try t.put("sa_o_{d}", .{li}, heads[si]);
        };
    }
    try ls.win.write(kv, p0, sc.s);

    const out = try attnOut(sc, m, ly, try ropeQ(sc, m, try sc.concat(heads, 0), rope, pos, true, .none, null));
    if (pf) |p| try p.mark(.attn_out, &.{ out, ls.win.buf });
    return out;
}

/// What a DSpark stage's block attends over: `keys`, its window of committed main rows, then the block's own rows.
/// `band` (`[rows, keys + rows]` bool) and `pos` (the rows' positions) default to one block at [p0, p0 + B) that sees
/// everything.
pub const DsKeys = struct { keys: A, band: A = none, pos: A = none };

/// The attention input's projections: wq_a, wkv and (none past the stacked bank) a ratio-1 compressor's wkv, from one
/// launch at decode widths.
fn attnIn(sc: *Scope, m: *const Model, ly: *const Layer, h: A) ![3]A {
    if (ly.in_shared) |sh| if (dim(h, 0) <= exl3.kernels.PROJECT_ROWS_MAX) {
        var outs: [3]A = .{ none, none, none };
        try sharedApply(sc, sh, ly.wq_a.stq.?.dec, h, outs[0..sh.count]);
        return outs;
    };
    _ = m;
    return .{ try ly.wq_a.apply(sc, h), try ly.wkv.apply(sc, h), none };
}

/// A DSpark stage's attention for the draft block `h` `[B, dim]`: every row sees the stage's window of committed main
/// rows and every row of the block, its own too (there is no causal mask).
fn dsparkAttention(sc: *Scope, m: *const Model, li: usize, h: A, ds: DsKeys, p0: usize) !A {
    const c = &m.cfg;
    const ly = &m.layers[li];
    const B = dim(h, 0);
    const hd = c.head_dim;
    const rope = m.rope_plain.freqs;
    const pos = if (ds.pos.ctx != null) ds.pos else try posArray(sc, p0, B, 1);
    const in = try attnIn(sc, m, ly, h);
    const qr = try sc.rms(in[0], ly.q_norm, c.eps);
    const q = try ropeQ(sc, m, try sc.reshape(try ly.wq_b.apply(sc, qr), &.{ ci(B), ci(c.heads), ci(hd) }), rope, pos, false, .none, null);
    const keys = try sc.concat(&.{ ds.keys, try windowRow(sc, m, ly, in[1], rope, pos) }, 0);
    const band = if (ds.band.ctx != null) ds.band else try sc.ge(try sc.zeros(&.{ ci(B), ci(dim(keys, 0)) }, .float32), try sc.f(0));
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const heads = try attnCore(sc, ly.sink, q, keys, band, none, none, scale, m.act);
    return attnOut(sc, m, ly, try ropeQ(sc, m, heads, rope, pos, true, .none, null));
}

/// The heads `o` `[L, H, hd]` (already inverse-rotated) through the output projections.
fn attnOut(sc: *Scope, m: *const Model, ly: *const Layer, o: A) !A {
    const c = &m.cfg;
    const L = dim(o, 0);
    const og = c.o_groups;
    const gin = c.heads * c.head_dim / og;
    const ob = try qatIf(sc, m, try sc.reshape(o, &.{ ci(L), ci(og), ci(gin) }), .fp8_ue8m0, 32);
    const obg = try sc.transpose(ob, &.{ 1, 0, 2 }); // [og, L, in]
    const oa = if (ly.wo_a.stq) |q|
        try q.applyGrouped(sc, obg)
    else if (ly.wo_a.s.ctx == null)
        try sc.matmul(obg, try sc.transpose(ly.wo_a.w, &.{ 0, 2, 1 }))
    else if (exact_matmul)
        try sc.matmul(obg, try sc.transpose(try dequant(sc, ly.wo_a.w, ly.wo_a.s, 8, "mxfp8", mlx.mlx_array_dtype(obg)), &.{ 0, 2, 1 }))
    else blk: {
        var ro = mlx.mlx_array_new();
        break :blk try sc.res(mlx.mlx_quantized_matmul(&ro, obg, ly.wo_a.w, ly.wo_a.s, none, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", sc.s), &ro);
    };
    const mid = try sc.reshape(try sc.transpose(oa, &.{ 1, 0, 2 }), &.{ ci(L), -1 });
    return ly.wo_b.apply(sc, mid);
}

// ── routed experts, Engram, the block ────────────────────────────────────

/// Rows up to which the per-row kernels (router GEMV, hyper-connection stage, Engram gate) serve; a wider chunk
/// is throughput-bound and keeps the matmul chains.
const NARROW_ROWS: usize = @intCast(cfg_mod.narrow_rows);

/// Whether the fused kernels serve this scope: the model allows them and the ops run on the GPU stream.
fn fusedOn(sc: *const Scope, m: *const Model) bool {
    return m.fused and mlx.streamIsGpu(sc.s);
}

/// The router's picks `idx` (`[L, k]` ids) and their weights `wts` (`[L, k]` f32): sqrt-softplus scores, the f32
/// bias picks only, normalized, times the routed scale.
fn router(sc: *Scope, m: *const Model, li: usize, h: A, img: A) !kernels.Routes {
    const c = &m.cfg;
    const ly = &m.layers[li];
    const K = ly.top_k;
    const norm = c.norm_topk and K > 1;
    // `img` rows pick by the image bias; the kernel reads a `[L, E]` bias row by row.
    const bias = if (img.ctx != null and ly.gate_bias_vl.ctx != null) try sc.where(try sc.expand(img, -1), ly.gate_bias_vl, ly.gate_bias) else ly.gate_bias;
    if (fusedOn(sc, m) and c.experts <= 1024) {
        const logits = if (dim(h, 0) <= NARROW_ROWS and c.dim % 8 == 0)
            try kernels.gateLogits(sc, h, ly.gate_w)
        else
            try sc.matmul(try sc.astype(h, .float32), try sc.t2(ly.gate_w));
        var r = try kernels.route(sc, logits, bias, K, norm, c.route_scale);
        r.logits = logits;
        try traceRoutes(m, li, logits, r);
        return r;
    }
    const logits = try sc.matmul(try sc.astype(h, .float32), try sc.t2(ly.gate_w)); // [L, E]
    const scores = try sc.op1(mlx.mlx_sqrt, try sc.op2(mlx.mlx_logaddexp, logits, try sc.f(0)));
    const idx = try topkRows(sc, m, try sc.add(scores, bias), K);
    var wts = try sc.takeAlong(scores, idx, -1);
    if (norm) wts = try sc.div(wts, try sc.add(try sc.sum(wts, -1, true), try sc.f(1e-20)));
    const r: kernels.Routes = .{ .idx = idx, .wts = try sc.mul(wts, try sc.f(c.route_scale)), .logits = logits };
    try traceRoutes(m, li, logits, r);
    return r;
}

/// The router's logits and picks, for tests that compare two forwards: a pick may differ only where scores tie.
fn traceRoutes(m: *const Model, li: usize, logits: A, r: kernels.Routes) !void {
    if (m.trace) |t| {
        try t.put("route_logits_{d}", .{li}, logits);
        try t.put("route_idx_{d}", .{li}, r.idx);
    }
}

/// The layer's banks for the picks `idx`: its stacked experts, or the slabs the expert stream leased (the caller
/// `releaseBanks` once the arrays reading them are evaluated).
fn leaseBanks(sc: *Scope, m: *const Model, li: usize, idx: A) !Experts.Lease {
    // A DSpark stage's experts are always resident.
    if (li >= m.cfg.layers) return .{ .banks = m.dspark.?.banks[li - m.cfg.layers], .local = idx };
    return switch (m.experts) {
        .resident => |banks| .{ .banks = banks[li], .local = idx },
        .hook => |hk| hk.lease(hk.ctx, sc, li, idx),
    };
}

fn releaseBanks(m: *const Model, li: usize) void {
    if (li >= m.cfg.layers) return;
    switch (m.experts) {
        .resident => {},
        .hook => |hk| hk.release(hk.ctx),
    }
}

fn sharedQuant(ly: *const Layer) ?kernels.SharedLins {
    for (ly.shared) |l| if (l.s.ctx == null or l.stq != null) return null;
    return .{
        .gate = .{ .w = ly.shared[0].w, .s = ly.shared[0].s },
        .up = .{ .w = ly.shared[1].w, .s = ly.shared[1].s },
        .down = .{ .w = ly.shared[2].w, .s = ly.shared[2].s },
    };
}

/// One token's FFN as two kernels (routed and shared gate-up, then routed and shared down summed and expanded
/// into `x1` by `sf`): the new stream, or null where they do not apply.
fn ffnDecode(sc: *Scope, m: *const Model, li: usize, h: A, x1: A, sf: qat_mod.Split) !?A {
    const c = &m.cfg;
    if (c.expert_dec != null or !fusedOn(sc, m) or dim(h, 0) != 1 or !kernels.moeDecodeFits(c.dim, c.moe_inter)) return null;
    const sh = sharedQuant(&m.layers[li]) orelse return null;
    const r = try router(sc, m, li, h, none);
    const lease = try leaseBanks(sc, m, li, r.idx);
    defer releaseBanks(m, li);
    errdefer _ = mlx.mlx_synchronize(sc.s);
    const inds = try sc.reshape(lease.local, &.{ci(m.layers[li].top_k)});
    const act = try kernels.moeGateUp(sc, h, lease.banks, sh, inds, try sc.reshape(r.wts, &.{ci(m.layers[li].top_k)}), c.swiglu_limit);
    const out = try kernels.moeDownExpand(sc, act, lease.banks, sh, inds, x1, try sc.reshape(sf.post, &.{ci(c.hc_mult)}), try sc.reshape(sf.comb, &.{ ci(c.hc_mult), ci(c.hc_mult) }));
    // Every slab reader completes before another layer can refill the union.
    try mlx.check(mlx.mlx_array_eval(out));
    if (prof(m, 1)) |p| try p.mark(.routed, &.{out});
    return try sc.reshape(out, mlx.getShape(x1));
}

/// The routed experts over many rows at once, sorted by expert so each bank streams once, plus the shared expert
/// `sy`: `[L, dim]` in the activation dtype.
fn moeWide(sc: *Scope, m: *const Model, li: usize, h: A, r: kernels.Routes, sy: A) !A {
    const c = &m.cfg;
    const L = dim(h, 0);
    const k = m.layers[li].top_k;
    const n = L * k;
    const lease = try leaseBanks(sc, m, li, r.idx);
    defer releaseBanks(m, li);
    errdefer _ = mlx.mlx_synchronize(sc.s);
    const flat = try sc.reshape(try sc.astype(lease.local, .uint32), &.{ci(n)});
    const order = try sc.argsort(flat);
    const inv = try sc.argsort(order);
    const rows = try sc.op2(mlx.mlx_floor_divide, order, try sc.astype(try sc.i(@intCast(k)), .uint32));
    const sidx = try sc.take(flat, order, 0);
    const xs = try sc.reshape(try sc.take(h, rows, 0), &.{ ci(n), 1, ci(dim(h, 1)) });
    const g = try gatherSorted(sc, xs, lease.banks.gate, sidx);
    const u = try gatherSorted(sc, xs, lease.banks.up, sidx);
    const w_sorted = try sc.take(try sc.reshape(r.wts, &.{ci(n)}), order, 0);
    const act = try kernels.moeAct(sc, g, u, w_sorted, c.swiglu_limit);
    const dn = try gatherSorted(sc, act, lease.banks.down, sidx);
    const out = try kernels.moeCombine(sc, try sc.reshape(dn, &.{ ci(n), ci(dim(h, 1)) }), inv, sy, k);
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// `x` `[L, dim]` through the trellis banks `rb` at the picks `slots` (`[L, k]` ids of `rb`) weighted `wts`: the clamped
/// SwiGLU of each, summed, plus the shared expert's `shared` (`[L, dim]`, f32). Decode-width chunks ride the
/// one-dispatch-per-stage decode chain, whose reduce adds `shared`, wider ones the grouped GEMM.
pub fn stqRouted(sc: *Scope, dec: exl3.format.Decode, limit: c_int, rb: *const exl3.Ragged, x: A, slots: A, wts: A, shared: A) !A {
    const L = dim(x, 0);
    const k: c_int = @intCast(dim(slots, 1));
    const flat = try sc.reshape(try sc.astype(slots, .uint32), &.{ci(L) * k});
    const scores = try sc.reshape(try sc.astype(wts, .float32), &.{ci(L) * k});
    try dec.requireTable();
    exl3.kernels.setDecodeParams(dec);
    // The decode reduce returns one row as `[dim]`.
    if (L <= exl3.kernels.DECODE_ROWS_MAX) return sc.reshape(try sc.keep(try exl3.rate_groups.moeDecodeClamped(sc.s, x, rb, flat, scores, k, .float32, shared, null, limit)), &.{ ci(L), ci(dim(x, 1)) });
    const y = try sc.astype(try sc.keep(try exl3.rate_groups.moePrefillClamped(sc.s, x, rb, flat, scores, k, limit)), .float32);
    return sc.add(y, try sc.astype(shared, .float32));
}

/// Queues `x` on the GPU without waiting for it.
fn submit(x: A) !void {
    const v = mlx.mlx_vector_array_new_data(&[_]A{x}, 1);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_async_eval(v));
}

fn gatherSorted(sc: *Scope, x: A, b: Bank, sidx: A) !A {
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_gather_qmm(&o, x, b.w, b.s, none, none, sidx, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "mxfp4", true, sc.s), &o);
}

/// A trellis shared expert at decode width is one expert of the fused chain (pair GEMV, mid and down, reduce):
/// the clamped SwiGLU stays in f32 registers. Null where the chain does not apply.
fn sharedFused(sc: *Scope, m: *const Model, ly: *const Layer, h: A) !?A {
    const rows = dim(h, 0);
    if (rows > exl3.kernels.DECODE_ROWS_MAX) return null;
    const q = [3]Stq{ ly.shared[0].stq orelse return null, ly.shared[1].stq orelse return null, ly.shared[2].stq orelse return null };
    const r: c_int = @intCast(rows);
    const zero_ids: [exl3.kernels.DECODE_ROWS_MAX]i32 = @splat(0);
    const unit: [exl3.kernels.DECODE_ROWS_MAX]f32 = @splat(1);
    const zeros = try sc.ints(zero_ids[0..rows], &.{ 1, r, 1 });
    const ones = try sc.floats(unit[0..rows], &.{ 1, r, 1 });
    const x3 = try sc.reshape(h, &.{ 1, r, ci(dim(h, 1)) });
    const bank: exl3.Bank = .{ .gate = q[0].p, .up = q[1].p, .down = q[2].p };
    const y = try sc.keep(try exl3.moeClamped(sc.s, x3, bank, zeros, ones, q[0].dec, @intFromFloat(m.cfg.swiglu_limit)));
    return try sc.reshape(y, &.{ r, ci(dim(h, 1)) });
}

fn sharedExpert(sc: *Scope, m: *const Model, ly: *const Layer, h: A) !A {
    if (try sharedFused(sc, m, ly, h)) |y| return y;
    const g = try ly.shared[0].apply(sc, h);
    const u = try ly.shared[1].apply(sc, h);
    const act = try sc.astype(try clampedSwiglu(sc, g, u, m.cfg.swiglu_limit), m.act);
    return ly.shared[2].apply(sc, act);
}

fn moe(sc: *Scope, m: *const Model, li: usize, h: A, img: A) !A {
    const c = &m.cfg;
    const ly = &m.layers[li];
    const r = try router(sc, m, li, h, img);
    const pf = prof(m, dim(h, 0));
    if (pf) |p| try p.mark(.router, &.{ r.idx, r.wts });
    if (c.expert_dec) |dec| {
        if (li >= c.layers) {
            const sy = try sharedExpert(sc, m, ly, h);
            return sc.astype(try stqRouted(sc, dec, @intFromFloat(c.swiglu_limit), &m.dspark.?.stq[li - c.layers], h, r.idx, r.wts, sy), m.act);
        }
        const hk = switch (m.experts) {
            .resident => return error.Dsv41StqExpertsStream,
            .hook => |hk| hk,
        };
        // At decode widths the GPU runs the shared expert while the host reads the picks and fills the misses, and the
        // layer is only submitted: the next layer's picks depend on it, so their read is the one wait per layer.
        const early = !m.sync_layers and dim(h, 0) <= exl3.kernels.DECODE_ROWS_MAX;
        if (early) try submit(r.idx);
        const sy = try sharedExpert(sc, m, ly, h);
        if (early) try submit(sy);
        if (pf) |p| try p.mark(.shared, &.{sy});
        const host0 = if (pf != null and hk.host_ns != null) hk.host_ns.?(hk.ctx) else 0;
        const out = try sc.astype(try hk.apply(hk.ctx, sc, li, h, r.idx, r.wts, r.logits, sy), m.act);
        if (pf) |p| {
            try p.mark(.routed, &.{out});
            if (hk.host_ns) |f| p.hostShare(f(hk.ctx) - host0);
        }
        try if (early) submit(out) else mlx.check(mlx.mlx_array_eval(out));
        return out;
    }
    if (fusedOn(sc, m) and dim(h, 0) > 1 and c.dim % 8 == 0) return moeWide(sc, m, li, h, r, try sharedExpert(sc, m, ly, h));
    const y = if (li >= c.layers) try routedSum(sc, c, m.act, h, m.dspark.?.banks[li - c.layers], r.idx, r.wts) else switch (m.experts) {
        .resident => |banks| try routedSum(sc, c, m.act, h, banks[li], r.idx, r.wts),
        .hook => |hk| try hk.apply(hk.ctx, sc, li, h, r.idx, r.wts, r.logits, null),
    };
    const sy = try sharedExpert(sc, m, ly, h);
    return sc.astype(try sc.add(y, try sc.astype(sy, .float32)), m.act);
}

/// The gated write of the n-gram rows into the stream `[L, hc, dim]`: `wkv` turns the rows into one key
/// per copy plus a shared value, and the gate is the sigmoid of the signed sqrt of a per-copy
/// normalized stream-key dot.
fn engramApply(sc: *Scope, m: *const Model, e: *const EngramLayer, x: A, rows: A) !A {
    const wkv_in = try sc.astype(try sc.reshape(rows, &.{ ci(dim(x, 0)), -1 }), m.act);
    const kv = try e.wkv.apply(sc, wkv_in);
    if (fusedOn(sc, m) and dim(x, 0) <= NARROW_ROWS) return kernels.engramGate(sc, x, kv, e.qk, m.cfg.eps);
    return engramGateOps(sc, &m.cfg, e.qk, x, kv);
}

/// `engramApply`, except on an image span's rows (`img`, none for text): the reference shuts their gate, so
/// they pass through.
fn engramWrite(sc: *Scope, m: *const Model, e: *const EngramLayer, x: A, rows: A, img: A) !A {
    const y = try engramApply(sc, m, e, x, rows);
    return if (img.ctx != null) sc.where(try sc.expand(try sc.expand(img, -1), -1), x, y) else y;
}

/// `engramApply` after `wkv`, as the op chain.
pub fn engramGateOps(sc: *Scope, c: *const Config, qk: A, x: A, kv: A) !A {
    const L = dim(x, 0);
    const hc = c.hc_mult;
    const d = c.dim;
    const key_ = try sc.reshape(try sc.astype(try sc.slice(kv, -1, 0, hc * d), .float32), &.{ ci(L), ci(hc), ci(d) });
    const value = try sc.astype(try sc.slice(kv, -1, hc * d, (hc + 1) * d), .float32);
    const hf = try sc.astype(x, .float32);
    const eps = try sc.f(c.eps);
    const rs_h = try sc.op1(mlx.mlx_rsqrt, try sc.add(try sc.mean(try sc.op1(mlx.mlx_square, hf), -1, false), eps));
    const rs_k = try sc.op1(mlx.mlx_rsqrt, try sc.add(try sc.mean(try sc.op1(mlx.mlx_square, key_), -1, false), eps));
    const dot = try sc.mul(try sc.mul(try sc.sum(try sc.mul(try sc.mul(hf, qk), key_), -1, false), try sc.mul(rs_h, rs_k)), try sc.f(1.0 / @sqrt(@as(f32, @floatFromInt(d)))));
    const mag = try sc.op1(mlx.mlx_sqrt, try sc.maximum(try sc.op1(mlx.mlx_abs, dot), try sc.f(1e-6)));
    const gate = try sc.sigmoid(try sc.where(try sc.lt(dot, try sc.f(0)), try sc.op1(mlx.mlx_negative, mag), mag));
    return sc.astype(try sc.add(hf, try sc.mul(try sc.expand(gate, -1), try sc.expand(value, 1))), mlx.mlx_array_dtype(x));
}

const Pair = struct { x: A, pre: A };

const HcPre = struct { sp: qat_mod.Split, h: A, x1: A };

/// The coefficients of the stream `x` (of `hcExpand(x, ex.y, ex.post, ex.comb)` with `ex`, returned as `x1`) and its
/// collapse by the previous mix `cpre`, rms-normed by `norm_w`: one stage and one split kernel at a few rows, the
/// mix and norm kernels on a wide block, else the op chains.
fn hcPre(sc: *Scope, m: *const Model, x: A, ex: ?kernels.HcExpand, cpre: A, hc: *const Hc, norm_w: A) !HcPre {
    const c = &m.cfg;
    if (fusedOn(sc, m) and dim(hc.fn_w, 0) == (2 + c.hc_mult) * c.hc_mult and dim(x, 2) % 32 == 0) {
        const eps = try kernels.hcEps(sc, c.eps, c.hc_eps);
        const hw = kernels.HcWeights{ .fn_w = hc.fn_w, .scale = hc.scale, .base = hc.base };
        if (dim(x, 0) <= NARROW_ROWS) {
            const st = try kernels.hcStage(sc, x, ex, cpre, norm_w, hw, eps);
            const sp = try kernels.hcSplitZ(sc, st.z, c.hc_mult, c.hc_iters, eps);
            return .{ .sp = .{ .pre = sp.pre, .post = sp.post, .comb = sp.comb }, .h = st.h, .x1 = st.x1 };
        }
        const x1 = if (ex) |e| try hcExpand(sc, x, e.y, e.post, e.comb) else x;
        const n = try kernels.hcNorm(sc, try kernels.hcMix(sc, x1, hw, eps), x1, cpre, norm_w, c.hc_iters, eps);
        return .{ .sp = .{ .pre = n.sp.pre, .post = n.sp.post, .comb = n.sp.comb }, .h = n.h, .x1 = x1 };
    }
    const x1 = if (ex) |e| try hcExpand(sc, x, e.y, e.post, e.comb) else x;
    return .{ .sp = try hcMixes(sc, m, x1, hc), .h = try sc.rms(try hcCollapse(sc, x1, cpre), norm_w, c.eps), .x1 = x1 };
}

/// A block: attention, then the FFN, each between a collapse of the stream and an expand back. The
/// collapse weights an attention takes are the previous FFN's; the FFN's are this attention's. A DSpark stage
/// (`ds` its window of main rows) attends over those and its own block instead of a backbone window.
fn block(sc: *Scope, m: *const Model, st: *State, li: usize, x: A, cpre: A, p0: usize, sh: *Shared, span: usize, ds: ?DsKeys) !Pair {
    const ly = &m.layers[li];
    const pa = try hcPre(sc, m, x, null, cpre, &ly.hc_attn, ly.attn_norm);
    const sa = pa.sp;
    const h = pa.h;
    if (prof(m, dim(x, 0))) |p| try p.mark(.hc_attn, &.{ h, sa.pre, sa.post, sa.comb });
    if (m.trace) |t| {
        try t.put("hc_pre_{d}", .{li}, sa.pre);
        try t.put("hc_post_{d}", .{li}, sa.post);
        try t.put("hc_comb_{d}", .{li}, sa.comb);
        try t.put("attn_in_{d}", .{li}, h);
    }
    // The attention's and the FFN's intermediates live in scopes of their own, so evaluating the layer can free them.
    const a = blk: {
        var asc = Scope.init(sc.a, sc.s);
        defer asc.deinit();
        const o = if (ds) |d| try dsparkAttention(&asc, m, li, h, d, p0) else try attention(&asc, m, st, li, h, p0, sh, span);
        break :blk try sc.keep(asc.out(o));
    };
    if (m.trace) |t| try t.put("attn_{d}", .{li}, a);
    const pf = try hcPre(sc, m, x, .{ .y = a, .post = sa.post, .comb = sa.comb }, sa.pre, &ly.hc_ffn, ly.ffn_norm);
    const x1 = pf.x1;
    const sf = pf.sp;
    const h2 = pf.h;
    if (prof(m, dim(x, 0))) |p| try p.mark(.hc_ffn, &.{ h2, x1, sf.pre, sf.post, sf.comb });
    if (m.trace) |t| try t.put("ffn_in_{d}", .{li}, h2);
    if (sh.img.ctx == null) if (try ffnDecode(sc, m, li, h2, x1, sf)) |x2| return .{ .x = sc.out(x2), .pre = sc.out(sf.pre) };
    const y = blk: {
        var fsc = Scope.init(sc.a, sc.s);
        defer fsc.deinit();
        break :blk try sc.keep(fsc.out(try moe(&fsc, m, li, h2, sh.img)));
    };
    if (m.trace) |t| try t.put("ffn_{d}", .{li}, y);
    return .{ .x = sc.out(try hcExpand(sc, x1, y, sf.post, sf.comb)), .pre = sc.out(sf.pre) };
}

// ── the forward ──────────────────────────────────────────────────────────

fn appendStateArrays(gpa: std.mem.Allocator, ev: *std.ArrayList(A), ls: *const LState) !void {
    try ev.append(gpa, ls.win.buf);
    for ([_]?Rows{ ls.comp, ls.idx_k }) |rows_| if (rows_) |r| {
        if (r.buf.ctx != null) try ev.append(gpa, r.buf);
        if (r.sbuf.ctx != null) try ev.append(gpa, r.sbuf);
    };
    if (ls.pend_kv.ctx != null) try ev.append(gpa, ls.pend_kv);
    if (ls.pend_gate.ctx != null) try ev.append(gpa, ls.pend_gate);
}

fn appendVerifyArrays(gpa: std.mem.Allocator, ev: *std.ArrayList(A), st: *const State, li: usize) !void {
    const v = st.verify orelse return;
    if (v.pend_kv[li].ctx != null) try ev.append(gpa, v.pend_kv[li]);
    if (v.pend_gate[li].ctx != null) try ev.append(gpa, v.pend_gate[li]);
}

pub fn ensureState(m: *Model) !*State {
    if (m.dec_state == null) m.dec_state = try newState(m);
    return &m.dec_state.?;
}

pub fn resetState(m: *Model) void {
    if (m.dec_state) |*st| st.deinit();
    m.dec_state = null;
}

/// Every decode width a request can take (1..16 rows, after a short prefill so compressed rows exist), with DSpark a
/// draft and a verify of every block width, and the prefill paths, run once on a throwaway state: a row count is a
/// template arg of the row kernels, so each variant compiles here instead of inside a request. Leaves no state.
pub fn warmKernels(m: *Model) !void {
    resetState(m);
    defer resetState(m);
    var ids: [exl3.kernels.DECODE_ROWS_MAX]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(i % m.cfg.vocab);
    // The first steps hold one compressed row a layer: an input under 8 elements binds as another library.
    for ([_]usize{ 1, 1, 2 }) |w| try warmRun(try extend(m, ids[0..w], .last));
    for (1..ids.len + 1) |w| try warmRun(try extend(m, ids[0..w], .last));
    if (m.dspark != null) for (2..m.cfg.dspark.block + 2) |w| try warmRound(m, ids[0], w);
    // The prefill path: a chunk past the window runs CED and its 256 rows fill 8 GEMM windows (an input of 8 binds as a
    // device array), a narrower chunk runs every layer over 2 windows; at the context they leave a decode step's rows
    // span two attention splits.
    resetState(m);
    var wide: [256]u32 = undefined;
    for (&wide, 0..) |*t, i| t.* = @intCast((i * 7 + 3) % m.cfg.vocab);
    const n = @min(wide.len, m.cfg.max_pos / 4);
    try warmRun(try extend(m, wide[0..n], .last));
    try warmRun(try extend(m, wide[0 .. n / 6], .last));
    for ([_]usize{ 1, 6 }) |w| try warmRun(try extend(m, ids[0..w], .last));
    if (m.dspark != null) try warmRound(m, ids[0], m.cfg.dspark.block + 1);
}

/// A DSpark draft from `t1` and a verify of its first `w` rows, all kept.
fn warmRound(m: *Model, t1: u32, w: usize) !void {
    var d = try draftHeads(m, t1, .conf);
    defer d.deinit();
    var round: [exl3.kernels.DECODE_ROWS_MAX]u32 = undefined;
    round[0] = t1;
    @memcpy(round[1..w], d.ids[0 .. w - 1]);
    try warmRun(try verify(m, round[0..w]));
    try finishVerify(m, w - 1);
}

fn warmRun(logits: A) !void {
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_eval(logits));
}

// ── restore points ───────────────────────────────────────────────────────

/// The arrays one slot of a restore point keeps; the prefix cache stores them in an SSM snapshot's four fields.
pub const Slot = [4]A;
pub const nil_slot: Slot = @splat(none);

/// Slots a restore point holds: per layer the window ring with the open pooling group, then the compressed and
/// index rows; then one per DSpark stage window.
pub fn slotCount(m: *const Model) usize {
    return 2 * m.cfg.layers + (if (m.dspark != null) m.cfg.n_mtp else 0);
}

/// An owned copy of rows [0, n) that keeps every bit (a float add of zero would turn -0.0 into +0.0).
/// The first `n` rows of `x` as a view.
fn headRows(x: A, n: usize, s: S) !A {
    if (x.ctx == null or n == 0) return none;
    const sh = mlx.getShape(x);
    var head = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, x, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(n), sh[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return head;
}

fn ownedRows(x: A, n: usize, s: S) !A {
    if (x.ctx == null or n == 0) return none;
    const head = try headRows(x, n, s);
    defer _ = mlx.mlx_array_free(head);
    const dt = mlx.mlx_array_dtype(x);
    const raw_dt: mlx.mlx_dtype = switch (dt) {
        .bfloat16, .float16 => .uint16,
        .float32 => .uint32,
        else => dt,
    };
    var raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(raw);
    try mlx.check(mlx.mlx_view(&raw, head, raw_dt, s));
    var zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero);
    try mlx.check(mlx.mlx_zeros(&zero, &[_]c_int{1}, 1, raw_dt, s));
    var copy = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&copy, raw, zero, s));
    if (raw_dt == dt) return copy;
    defer _ = mlx.mlx_array_free(copy);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_view(&out, copy, dt, s));
    return out;
}

/// A slot array the checkpoint holds; every real one is 2-D, and an absent one is stored as a one-byte placeholder.
fn present(a: A) bool {
    return a.ctx != null and mlx.mlx_array_ndim(a) == 2 and mlx.mlx_array_size(a) > 0;
}

fn shareInto(dst: *A, src: A) !void {
    if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
    dst.* = none;
    if (!present(src)) return;
    dst.* = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(dst, src));
}

/// The request's state at `st.n` as owned copies in `out` (`slotCount` nil slots); frees them on a failure.
pub fn captureState(m: *Model, out: []Slot) !void {
    return captureInto(m, out, false);
}

/// `captureState` whose compressed rows and index keys are views of the request's own rows: no copy, so the caller
/// must drop them before the next forward appends, or that append copies the whole store.
pub fn captureStateShared(m: *Model, out: []Slot) !void {
    return captureInto(m, out, true);
}

fn captureInto(m: *Model, out: []Slot, share: bool) !void {
    const st = try ensureState(m);
    if (st.verify != null) return error.Dsv41VerifyOpen;
    if (out.len != slotCount(m)) return error.Dsv41CheckpointMismatch;
    errdefer for (out) |*sl| freeSlot(sl);
    const evals = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(evals);
    for (st.layers, 0..) |*ls, li| {
        const a = &out[2 * li];
        a[0] = try ownedRows(ls.win.buf, ls.win.cap, m.s);
        a[1] = try ownedRows(ls.pend_kv, if (ls.pend_kv.ctx != null) dim(ls.pend_kv, 0) else 0, m.s);
        a[2] = try ownedRows(ls.pend_gate, if (ls.pend_gate.ctx != null) dim(ls.pend_gate, 0) else 0, m.s);
        const b = &out[2 * li + 1];
        if (ls.comp) |r| {
            b[0] = try if (share) headRows(r.buf, r.used, m.s) else ownedRows(r.buf, r.used, m.s);
            b[1] = try if (share) headRows(r.sbuf, r.used, m.s) else ownedRows(r.sbuf, r.used, m.s);
        }
        if (ls.idx_k) |r| {
            b[2] = try if (share) headRows(r.buf, r.used, m.s) else ownedRows(r.buf, r.used, m.s);
            b[3] = try if (share) headRows(r.sbuf, r.used, m.s) else ownedRows(r.sbuf, r.used, m.s);
        }
    }
    for (st.ds, 0..) |r, si| out[2 * st.layers.len + si][0] = try ownedRows(r.buf, r.cap, m.s);
    for (out) |sl| for (sl) |x| if (x.ctx != null) try mlx.check(mlx.mlx_vector_array_append_value(evals, x));
    try mlx.check(mlx.mlx_eval(evals));
}

pub fn freeSlot(sl: *Slot) void {
    for (sl) |*x| {
        if (x.ctx != null) _ = mlx.mlx_array_free(x.*);
        x.* = none;
    }
}

fn restoreRows(r: *Rows, buf: A, sbuf: A, want: usize) !void {
    const used: usize = if (present(buf)) dim(buf, 0) else 0;
    if (used != want or (r.pack != null and used > 0 and !present(sbuf))) return error.Dsv41CheckpointMismatch;
    if (used == 0) return;
    try shareInto(&r.buf, buf);
    try shareInto(&r.sbuf, sbuf);
    r.used = used;
    r.cap = used;
}

/// Makes the request's state the restore point `slots` at position `n`, over the first `n` of the request's
/// token ids (the Engram history); the arrays are shared, never written. A failure leaves the state reset.
pub fn restoreState(m: *Model, slots: []const Slot, n: usize, ids: []const u32) !void {
    if (slots.len != slotCount(m) or ids.len < n) return error.Dsv41CheckpointMismatch;
    resetState(m);
    errdefer resetState(m);
    const st = try ensureState(m);
    for (st.layers, 0..) |*ls, li| {
        const a = slots[2 * li];
        const b = slots[2 * li + 1];
        if (!present(a[0]) or dim(a[0], 0) != ls.win.cap) return error.Dsv41CheckpointMismatch;
        try shareInto(&ls.win.buf, a[0]);
        try shareInto(&ls.pend_kv, a[1]);
        try shareInto(&ls.pend_gate, a[2]);
        const r = m.ratio(li);
        if (ls.comp) |*rows| try restoreRows(rows, b[0], b[1], n / r);
        if (ls.idx_k) |*rows| try restoreRows(rows, b[2], b[3], n / r);
    }
    for (st.ds, 0..) |*ring, si| {
        const a = slots[2 * st.layers.len + si][0];
        if (!present(a) or dim(a, 0) != ring.cap) return error.Dsv41CheckpointMismatch;
        try shareInto(&ring.buf, a);
    }
    st.n = n;
    if (m.hash != null) {
        try st.hist.ensureTotalCapacity(m.gpa, n);
        for (ids[0..n]) |t| st.hist.appendAssumeCapacity(m.compressedId(t));
    }
}

/// Sets the kernels off (the diagnostic `SUSHI_DSV41_COMPOSED`, the KLD A/B's teacher arm) or on, and says which.
pub fn setFused(m: *Model, composed: bool) void {
    m.fused = !composed;
    log.info("[dsv41] fused kernels {s}{s}\n", .{ if (m.fused) "on" else "off", if (composed) " (diag)" else "" });
}

/// One chunk of a request in flight: the residual streams between layers, the Engram rows and what layers hand
/// down, over the state `st` the chunk extends. `extendSpan` runs every layer in turn.
pub const Pass = struct {
    st: *State,
    ids: []const u32,
    p0: usize,
    span: usize,
    rows: []u32,
    x: A,
    cpre: A,
    sh: Shared,
    /// DSpark: the mean of the hyper-connection copies entering each target layer, `[L, dim]` apiece.
    targets: [cfg_mod.max_targets]A = @splat(none),
    /// CED: the rows the decoder runs over, set by `planCed` (null: every layer runs every row).
    tail: ?usize = null,
    /// The chunk's Engram rows read on a thread while the layers before each Engram layer run.
    engram_pf: ?*engram.Prefetch = null,
    /// `[L]` bool, the rows inside an image span (none for text): they route by the image bias and take no Engram write.
    img: A = none,

    pub fn deinit(self: *Pass, gpa: std.mem.Allocator) void {
        if (self.engram_pf) |pf| pf.finish();
        if (self.img.ctx != null) _ = mlx.mlx_array_free(self.img);
        gpa.free(self.rows);
        if (self.x.ctx != null) _ = mlx.mlx_array_free(self.x);
        if (self.cpre.ctx != null) _ = mlx.mlx_array_free(self.cpre);
        for (self.targets) |t| if (t.ctx != null) {
            _ = mlx.mlx_array_free(t);
        };
        self.sh.deinit(gpa);
    }
};

/// The soft tokens of a request's images, `[1, N, dim]` or `[N, dim]` in prompt order (the tower's whole spans,
/// delimiters included), and how many earlier chunks consumed: the chunk's `image_token_id` positions take the
/// next rows.
pub const VisionRows = struct { rows: A, offset: usize };

/// Embeds `ids` as the chunk of `st` that starts at position `p0` (extending its Engram history) and returns the
/// pass before layer 0. `ids` must outlive the pass.
pub fn beginPass(m: *Model, st: *State, ids: []const u32, p0: usize, span: usize) !Pass {
    return beginPassWith(m, st, ids, p0, span, null);
}

/// How many of `ids` sit in an image span.
pub fn imageRows(m: *const Model, ids: []const u32) usize {
    return if (m.cfg.image_token_id == 0) 0 else std.mem.count(u32, ids, &.{m.cfg.image_token_id});
}

/// `beginPass` with the images of the request: the span positions of `ids` take `vision`'s rows.
pub fn beginPassWith(m: *Model, st: *State, ids: []const u32, p0: usize, span: usize, vision: ?VisionRows) !Pass {
    const gpa = m.gpa;
    const c = &m.cfg;
    const L = ids.len;
    const hc = c.hc_mult;

    var rows: []u32 = &.{};
    errdefer gpa.free(rows);
    if (m.hash) |*hs| {
        const comp = try gpa.alloc(u32, L);
        defer gpa.free(comp);
        for (ids, comp) |t, *o| o.* = m.compressedId(t);
        rows = try gpa.alloc(u32, L * hs.n_layers * hs.cols());
        hs.rows(st.hist.items, comp, rows);
        try st.hist.appendSlice(gpa, comp);
    }

    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    const host = try sc.a.alloc(i32, L);
    defer sc.a.free(host);
    for (host, ids) |*o, t| o.* = @intCast(t);
    const id_arr = try sc.ints(host, &.{ci(L)});
    var emb = try m.embed.rows(&sc, id_arr, m.act); // [L, dim]
    var img: A = none;
    errdefer if (img.ctx != null) {
        _ = mlx.mlx_array_free(img);
    };
    // Once the request's rows are all consumed an image id is one the model emitted itself: a plain token.
    if (imageRows(m, ids) > 0 and (vision == null or vision.?.offset * c.dim < mlx.mlx_array_size(vision.?.rows))) {
        const vr = vision orelse return error.Dsv41VisionRowsMissing;
        for (m.layers[0..c.layers]) |*ly| if (ly.gate_bias_vl.ctx == null) return error.Dsv41VisionBiasMissing;
        const take = try sc.a.alloc(i32, L);
        defer sc.a.free(take);
        var k = vr.offset;
        for (ids, take) |t, *o| {
            o.* = if (t == c.image_token_id) ci(k) else 0;
            k += @intFromBool(t == c.image_token_id);
        }
        if (k * c.dim > mlx.mlx_array_size(vr.rows)) return error.Dsv41VisionRowsShort;
        const is_img = try sc.op2(mlx.mlx_equal, id_arr, try sc.i(@intCast(c.image_token_id)));
        img = sc.out(is_img);
        const picked = try sc.astype(try sc.take(try sc.reshape(vr.rows, &.{ -1, ci(c.dim) }), try sc.ints(take, &.{ci(L)}), 0), m.act);
        emb = try sc.where(try sc.expand(is_img, -1), picked, emb);
    }
    const ex = try sc.contiguous(try sc.broadcast(try sc.expand(emb, 1), &.{ ci(L), ci(hc), ci(c.dim) }));
    const x = sc.out(ex);
    errdefer _ = mlx.mlx_array_free(x);
    const first = try sc.op2(mlx.mlx_equal, try sc.arange(0, hc), try sc.i(0));
    const cpre = sc.out(try sc.broadcast(try sc.astype(first, .float32), &.{ ci(L), ci(hc) }));
    errdefer _ = mlx.mlx_array_free(cpre);
    try evalAll(if (img.ctx != null) &.{ x, cpre, img } else &.{ x, cpre });
    return .{ .st = st, .ids = ids, .p0 = p0, .span = span, .rows = rows, .x = x, .cpre = cpre, .sh = try Shared.init(gpa, (L + span - 1) / span), .img = img };
}

/// Layer `li` over the pass's rows, evaluated, with the new state arrays.
pub fn passLayer(m: *Model, p: *Pass, li: usize) !void {
    const gpa = m.gpa;
    if (p.tail) |t| if (li == m.ced_split.?) {
        try cedReplay(m, p, li, t);
        p.tail = null;
    };
    const L = p.ids.len;
    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    if (prof(m, L)) |pf| {
        pf.restart();
        pf.decoder = if (m.ced_split) |split| li >= split else false;
    }
    if (m.cfg.engram.layerAt(li)) |ei| {
        const hs = &m.hash.?;
        const nc = hs.cols();
        const mine = try gpa.alloc(u32, L * nc);
        defer gpa.free(mine);
        for (0..L) |t| @memcpy(mine[t * nc ..][0..nc], p.rows[(t * hs.n_layers + ei) * nc ..][0..nc]);
        const e = &m.engrams[ei];
        const io = std.Io.Threaded.global_single_threaded.io();
        const t0 = std.Io.Timestamp.now(io, .boot);
        defer m.engram_wait_ns += @intCast(t0.untilNow(io, .boot).nanoseconds);
        const table_rows = if (p.engram_pf) |pf| blk: {
            const raw = try pf.take(p.p0, ei, 0);
            defer raw.deinit(gpa);
            break :blk try sc.keep(try e.table.dequant(raw, mine.len, m.s));
        } else try sc.keep(try e.table.gather(gpa, mine, m.s)); // [L * cols, head_dim]
        const nx = sc.out(try engramWrite(&sc, m, e, p.x, table_rows, p.img));
        _ = mlx.mlx_array_free(p.x);
        p.x = nx;
        if (prof(m, L)) |pf| try pf.mark(.engram, &.{nx});
    }
    const c = &m.cfg;
    // At decode widths a layer is submitted, not waited on: the next layer's router picks are the one host read.
    const wait = m.sync_layers or L > exl3.kernels.DECODE_ROWS_MAX;
    if (p.st.ds.len > 0) for (c.dspark.targets[0..c.dspark.n_targets], 0..) |tl, t| if (tl == li) {
        p.targets[t] = sc.out(try sc.astype(try sc.mean(try sc.astype(p.x, .float32), 1, false), m.act));
        try if (wait) mlx.check(mlx.mlx_array_eval(p.targets[t])) else submit(p.targets[t]);
    };
    p.sh.img = p.img;
    const out = try block(&sc, m, p.st, li, p.x, p.cpre, p.p0, &p.sh, p.span, null);
    _ = mlx.mlx_array_free(p.x);
    _ = mlx.mlx_array_free(p.cpre);
    p.x = out.x;
    p.cpre = out.pre;
    var ev: std.ArrayList(A) = .empty;
    defer ev.deinit(gpa);
    try ev.append(gpa, p.x);
    try ev.append(gpa, p.cpre);
    try appendStateArrays(gpa, &ev, &p.st.layers[li]);
    try appendVerifyArrays(gpa, &ev, p.st, li);
    if (m.trace) |t| try t.put("stream_{d}", .{li}, p.x);
    if (prof(m, L)) |pf| return pf.mark(.expand, ev.items);
    if (wait) return evalAll(ev.items);
    const vec = mlx.mlx_vector_array_new_data(ev.items.ptr, ev.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_async_eval(vec));
}

/// The pass's logits (f32; `[1, vocab]` for the last position or `[L, vocab]` for every one; the caller frees);
/// the state now covers the chunk.
pub fn finishPass(m: *Model, p: *Pass, want: Want) !A {
    const c = &m.cfg;
    const L = p.ids.len;
    reportEngramWait(m, L);
    p.st.n = p.p0 + L;
    if (p.st.ds.len > 0) {
        var msc = Scope.init(m.gpa, m.s);
        defer msc.deinit();
        const main_rows = try msc.concat(p.targets[0..c.dspark.n_targets], -1); // [L, targets * dim]
        if (p.st.verify) |*v| {
            v.mh = msc.out(main_rows);
            try mlx.check(mlx.mlx_array_eval(v.mh));
        } else try commitMain(&msc, m, p.st, main_rows, p.p0);
    }
    var sc = Scope.init(m.gpa, m.s);
    defer sc.deinit();
    const xs = if (want == .last) try sc.slice(p.x, 0, L - 1, L) else p.x;
    const ps = if (want == .last) try sc.slice(p.cpre, 0, L - 1, L) else p.cpre;
    const hn = try sc.rms(try hcCollapse(&sc, xs, ps), m.norm, c.eps);
    const logits = try m.head.apply(&sc, try sc.astype(hn, .float32));
    const res = sc.out(logits);
    errdefer _ = mlx.mlx_array_free(res);
    if (prof(m, L)) |pf| {
        pf.restart();
        try pf.mark(.head, &.{res});
        pf.report(L, p.p0);
    }
    try mlx.check(mlx.mlx_array_eval(res));
    return res;
}

/// Advances the request by `ids` and returns the logits (f32; `[1, vocab]` for the last position or
/// `[L, vocab]` for every one; the caller frees). `span` narrows the attention spans (tests).
pub fn extendSpan(m: *Model, ids: []const u32, want: Want, span: usize) !A {
    return extendVision(m, ids, want, span, null);
}

/// `extendSpan` over a chunk that may hold image spans, whose rows `vision` carries.
pub fn extendVision(m: *Model, ids: []const u32, want: Want, span: usize, vision: ?VisionRows) !A {
    const st = try ensureState(m);
    var pass = try beginPassWith(m, st, ids, st.n, span, vision);
    defer pass.deinit(m.gpa);
    planCed(m, &pass, want);
    try prefetchEngram(m, &pass);
    for (0..m.cfg.layers) |li| try passLayer(m, &pass, li);
    return finishPass(m, &pass, want);
}

/// At debug level: the Engram wait of each wide chunk, and the mean of every 64 decode forwards.
fn reportEngramWait(m: *Model, L: usize) void {
    if (m.engrams.len == 0 or !log.isDebug()) return;
    const wide = L > NARROW_ROWS;
    if (!wide) {
        m.engram_forwards += 1;
        if (m.engram_forwards < 64) return;
    }
    const per = @as(f64, @floatFromInt(m.engram_wait_ns)) / 1e6 / @as(f64, @floatFromInt(if (wide) 1 else m.engram_forwards));
    log.debug("[dsv41] engram wait_ms={d:.3} per {s} ({d} rows)\n", .{ per, if (wide) "chunk" else "decode forward, mean of 64", L });
    m.engram_wait_ns = 0;
    m.engram_forwards = 0;
}

/// A chunk's Engram rows start reading now, one grid of one span per table, so each table's reads hide behind the
/// layers before it (a decode token's: layer 0 for layer 1, layers 0-13 for layer 14).
fn prefetchEngram(m: *Model, p: *Pass) !void {
    const hs = &(m.hash orelse return);
    if (m.engrams.len == 0 or (!m.decode_engram_ahead and p.ids.len <= NARROW_ROWS)) return;
    var tables: [cfg_mod.max_engram]*const engram.Table = undefined;
    for (m.engrams, 0..) |*e, i| tables[i] = &e.table;
    p.engram_pf = try engram.Prefetch.start(m.gpa, tables[0..m.engrams.len], &.{.{ .at = p.p0, .len = p.ids.len, .ids = p.rows }}, hs.cols(), p.ids.len);
}

/// Whether the pass's chunk runs the CED decoder over its last rows only: it does when it is wider than the window and
/// `finishPass` will take the last row's logits.
pub fn planCed(m: *const Model, p: *Pass, want: Want) void {
    p.tail = cedRows(m, p.st, p.ids.len, want);
}

fn cedRows(m: *const Model, st: *const State, L: usize, want: Want) ?usize {
    if (m.ced_split == null or want != .last or st.verify != null) return null;
    const w: usize = m.cfg.window;
    return if (L > w) w else null;
}

/// CED: the decoder's first layer `li` compresses every row of the pass (the ratio-1 rows every later layer reads),
/// then the pass keeps its last `t` rows. The decoder writes no window rows before them, and none of its queries
/// reaches one: the replay's band starts at its first row.
fn cedReplay(m: *Model, p: *Pass, li: usize, t: usize) !void {
    const gpa = m.gpa;
    const L = p.ids.len;
    const ly = &m.layers[li];
    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    const pa = try hcPre(&sc, m, p.x, null, p.cpre, &ly.hc_attn, ly.attn_norm);
    try compress(&sc, m, p.st, li, pa.h, p.p0, none);
    const x = sc.out(try sc.slice(p.x, 0, L - t, L));
    errdefer _ = mlx.mlx_array_free(x);
    const pre = sc.out(try sc.slice(p.cpre, 0, L - t, L));
    errdefer _ = mlx.mlx_array_free(pre);
    const img = if (p.img.ctx != null) sc.out(try sc.slice(p.img, 0, L - t, L)) else none;
    errdefer if (img.ctx != null) {
        _ = mlx.mlx_array_free(img);
    };
    var ev: std.ArrayList(A) = .empty;
    defer ev.deinit(gpa);
    try ev.appendSlice(gpa, &.{ x, pre });
    if (img.ctx != null) try ev.append(gpa, img);
    try appendStateArrays(gpa, &ev, &p.st.layers[li]);
    try evalAll(ev.items);
    var sh = try Shared.init(gpa, (t + p.span - 1) / p.span);
    sh.floor = p.p0 + L - t;
    p.sh.deinit(gpa);
    p.sh = sh;
    _ = mlx.mlx_array_free(p.x);
    _ = mlx.mlx_array_free(p.cpre);
    p.x = x;
    p.cpre = pre;
    if (p.img.ctx != null) _ = mlx.mlx_array_free(p.img);
    p.img = img;
    p.ids = p.ids[L - t ..];
    p.p0 += L - t;
    log.debug("[dsv41] CED prefill: {d} rows encoder-only, replay {d} rows\n", .{ L - t, t });
}

pub fn extend(m: *Model, ids: []const u32, want: Want) !A {
    return extendSpan(m, ids, want, SPAN);
}

// ── DSpark ───────────────────────────────────────────────────────────────

/// The stages' window rows for the committed positions [p0, p0 + n): the target layers' main stream
/// `main_norm(main_proj(rows))`, `rows` `[n, targets * dim]`, through each stage's own key path.
fn commitMain(sc: *Scope, m: *Model, st: *State, rows: A, p0: usize) !void {
    const ds = &m.dspark.?;
    const n = dim(rows, 0);
    const main_x = try sc.rms(try ds.main_proj.apply(sc, rows), ds.main_norm, m.cfg.eps);
    if (m.trace) |t| try t.put("ds_mainx", .{}, main_x);
    const pos = try posArray(sc, p0, n, 1);
    for (st.ds, 0..) |*ring, si| {
        const ly = &m.layers[m.cfg.layers + si];
        try ring.write(try windowRow(sc, m, ly, try ly.wkv.apply(sc, main_x), m.rope_plain.freqs, pos), p0, sc.s);
    }
    var ev: std.ArrayList(A) = .empty;
    defer ev.deinit(sc.a);
    for (st.ds) |r| try ev.append(sc.a, r.buf);
    try evalAll(ev.items);
}

/// A drafted block: `ids[0]` the token the draft started from and `ids[1..]` its `block` picks; with the heads' rows
/// on the host when asked for (`logits` `[block, vocab]`, the Markov-biased scores each pick came from; `conf`
/// `[block]`, the confidence head's logit of each).
pub const Draft = struct {
    gpa: std.mem.Allocator,
    ids: []u32,
    logits: []f32,
    conf: []f32,

    pub fn deinit(self: *Draft) void {
        self.gpa.free(self.ids);
        self.gpa.free(self.logits);
        self.gpa.free(self.conf);
    }
};

/// What `draft` hands back besides the picks: the confidence logits only (serving), or also the biased scores (tests).
pub const Heads = enum { none, conf, full };

pub fn draft(m: *Model, t1: u32, with_heads: bool) !Draft {
    return draftHeads(m, t1, if (with_heads) .full else .none);
}

/// One draft pass for the block `[t1, noise...]` at the positions the request is about to fill: the stages over
/// the windows of committed main rows, the trunk head over the last stage's norm, then the Markov bigram bias chained
/// token to token.
pub fn draftHeads(m: *Model, t1: u32, heads_wanted: Heads) !Draft {
    const gpa = m.gpa;
    const saved_profile = m.profile;
    m.profile = null;
    defer m.profile = saved_profile;
    const with_conf = heads_wanted != .none;
    const with_heads = heads_wanted == .full;
    const c = &m.cfg;
    const st = try ensureState(m);
    const ds = &(m.dspark orelse return error.Dsv41NoDspark);
    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    const B = c.dspark.block;
    const n = st.n;
    const hc = c.hc_mult;
    var in: [cfg_mod.max_block]i32 = undefined;
    for (in[0..B], 0..) |*v, k| v.* = @intCast(if (k == 0) t1 else c.dspark.noise_id);
    const emb = try m.embed.rows(&sc, try sc.ints(in[0..B], &.{ci(B)}), m.act);
    var x = try sc.contiguous(try sc.broadcast(try sc.expand(emb, 1), &.{ ci(B), ci(hc), ci(c.dim) }));
    var pre = try sc.broadcast(try sc.astype(try sc.op2(mlx.mlx_equal, try sc.arange(0, hc), try sc.i(0)), .float32), &.{ ci(B), ci(hc) });
    var dummy = try Shared.init(gpa, 1);
    defer dummy.deinit(gpa);
    const window = @min(c.window, n);
    for (0..c.n_mtp) |si| {
        const keys = try st.ds[si].read(&sc, n - window, n);
        const out = try block(&sc, m, st, c.layers + si, x, pre, n, &dummy, SPAN, .{ .keys = keys });
        x = try sc.keep(out.x);
        pre = try sc.keep(out.pre);
    }
    const hout = try hcCollapse(&sc, x, pre); // [B, dim]
    const logits = try m.head.apply(&sc, try sc.astype(try sc.rms(hout, ds.norm, c.eps), .float32)); // [B, vocab] f32
    var cur = try sc.reshape(try sc.i(@intCast(t1)), &.{1});
    var picks: [cfg_mod.max_block]A = undefined;
    var rows: [cfg_mod.max_block]A = undefined;
    var confs: [cfg_mod.max_block]A = undefined;
    for (0..B) |k| {
        const me = try sc.astype(try sc.take(ds.markov_embed, cur, 0), .float32); // [1, rank]
        const row = try sc.add(try sc.slice(logits, 0, k, k + 1), try sc.matmul(me, try sc.t2(ds.markov_head)));
        rows[k] = row;
        if (with_conf) confs[k] = try sc.matmul(try sc.concat(&.{ try sc.slice(try sc.astype(hout, .float32), 0, k, k + 1), me }, -1), try sc.t2(ds.conf));
        var am = mlx.mlx_array_new();
        cur = try sc.astype(try sc.reshape(try sc.res(mlx.mlx_argmax_axis(&am, row, -1, false, sc.s), &am), &.{1}), .int32);
        picks[k] = cur;
    }
    const all = try sc.concat(picks[0..B], 0);
    const heads = if (with_heads) try sc.concat(rows[0..B], 0) else all;
    const conf = if (with_conf) try sc.reshape(try sc.concat(confs[0..B], 0), &.{ci(B)}) else all;
    try evalAll(&.{ all, heads, conf });
    var out: Draft = .{ .gpa = gpa, .ids = &.{}, .logits = &.{}, .conf = &.{} };
    errdefer out.deinit();
    out.ids = try gpa.alloc(u32, B + 1);
    out.ids[0] = t1;
    for (try sc_mod.hostI32(all), out.ids[1..]) |v, *o| o.* = @intCast(v);
    out.logits = try gpa.alloc(f32, if (with_heads) B * c.vocab else 0);
    out.conf = try gpa.alloc(f32, if (with_conf) B else 0);
    if (with_heads) @memcpy(out.logits, try sc_mod.hostF32(heads));
    if (with_conf) @memcpy(out.conf, try sc_mod.hostF32(conf));
    return out;
}

/// Advances the request by `ids` (`[t1, drafts...]`) as a chunk it can give back: the logits of every row (f32
/// `[L, vocab]`; the caller frees). `finishVerify` keeps a prefix.
pub fn verify(m: *Model, ids: []const u32) !A {
    const st = try ensureState(m);
    if (st.verify != null) return error.Dsv41VerifyOpen;
    const nl = m.cfg.layers;
    var v: Verify = .{
        .n0 = st.n,
        .rows = ids.len,
        .comp_used = try m.gpa.alloc(usize, nl),
        .idx_used = try m.gpa.alloc(usize, nl),
        .pend_kv = try m.gpa.alloc(A, nl),
        .pend_gate = try m.gpa.alloc(A, nl),
    };
    for (st.layers, 0..) |*ls, li| {
        v.comp_used[li] = if (ls.comp) |r| r.used else 0;
        v.idx_used[li] = if (ls.idx_k) |r| r.used else 0;
        v.pend_kv[li] = none;
        v.pend_gate[li] = none;
    }
    st.verify = v;
    errdefer st.dropVerify();
    if (ids.len > cfg_mod.ring_slack) return error.Dsv41VerifyTooWide;
    return extend(m, ids, .all);
}

/// Keeps `accepted + 1` of the verify chunk's rows: the positions, the compressed rows and index keys past them, and each
/// pooling compressor's open group are cut back to the kept rows, and their main rows join the stages' windows.
pub fn finishVerify(m: *Model, accepted: usize) !void {
    const st = try ensureState(m);
    defer st.dropVerify();
    const v = &(st.verify orelse return error.Dsv41NoVerify);
    const keep = accepted + 1;
    if (keep > v.rows) return error.Dsv41VerifyTooWide;
    var sc = Scope.init(m.gpa, m.s);
    defer sc.deinit();
    const end = v.n0 + keep;
    if (keep < v.rows) {
        for (st.layers, 0..) |*ls, li| {
            const r = m.ratio(li);
            if (r == 0) continue;
            if (ls.comp) |*rows| rows.used = v.comp_used[li] + (end / r - v.n0 / r);
            if (ls.idx_k) |*rows| rows.used = v.idx_used[li] + (end / r - v.n0 / r);
            if (r > 1 and v.pend_kv[li].ctx != null) {
                const first = v.n0 - v.n0 % r; // the position of row 0 of the saved open group
                const lo = end - end % r - first;
                const hi = end - first;
                if (ls.pend_kv.ctx != null) _ = mlx.mlx_array_free(ls.pend_kv);
                if (ls.pend_gate.ctx != null) _ = mlx.mlx_array_free(ls.pend_gate);
                ls.pend_kv = if (hi > lo) sc.out(try sc.slice(v.pend_kv[li], 0, lo, hi)) else none;
                ls.pend_gate = if (hi > lo) sc.out(try sc.slice(v.pend_gate[li], 0, lo, hi)) else none;
                if (ls.pend_kv.ctx != null) try evalAll(&.{ ls.pend_kv, ls.pend_gate });
            }
        }
        st.n = end;
        if (st.hist.items.len > end) st.hist.shrinkRetainingCapacity(end);
    }
    if (v.mh.ctx != null) try commitMain(&sc, m, st, try sc.slice(v.mh, 0, 0, keep), v.n0);
}

/// DSpark stage `si`'s routed experts as trellis banks, `mtp.<si>.mlp.switch_mlp.<P>[.g<N>].{trellis,suh,svh}`, held as one
/// ragged bank (router order is group order).
fn loadStqExperts(m: *Model, w: *model_mod.Weights, si: usize) !exl3.Ragged {
    const c = &m.cfg;
    var banks: [exl3.group_layout.max_groups]exl3.Bank = undefined;
    var names: [exl3.group_layout.max_groups * 9]NameBuf = undefined;
    var lens: [exl3.group_layout.max_groups * 9]usize = undefined;
    var named: usize = 0;
    defer for (names[0..named], lens[0..named]) |*b, n| source.drop(w, b[0..n]);
    var probe: NameBuf = undefined;
    const grouped = w.get(try key(&probe, "mtp.{d}.mlp.switch_mlp.gate_proj.g0.trellis", .{si})) != null;
    var count: usize = 0;
    while (count < banks.len) : (count += 1) {
        if (grouped and w.get(try key(&probe, "mtp.{d}.mlp.switch_mlp.gate_proj.g{d}.trellis", .{ si, count })) == null) break;
        if (!grouped and count == 1) break;
        var g: [16]u8 = undefined;
        const tag = if (grouped) try std.fmt.bufPrint(&g, ".g{d}", .{count}) else "";
        inline for (.{ "gate", "up", "down" }) |pn| {
            var parts: [3]A = undefined;
            inline for (.{ "trellis", "suh", "svh" }, 0..) |part, pi| {
                const name = try key(&names[named], "mtp.{d}.mlp.switch_mlp." ++ pn ++ "_proj{s}." ++ part, .{ si, tag });
                lens[named] = name.len;
                named += 1;
                parts[pi] = try getW(w, name);
            }
            @field(banks[count], pn) = .{ .trellis = parts[0], .suh = parts[1], .svh = parts[2] };
        }
    }
    var rb = try exl3.Ragged.init(m.s, banks[0..count]);
    errdefer rb.deinit();
    if (rb.experts != c.dspark.experts or rb.hidden != c.dim or rb.inter != c.moe_inter) {
        log.err("deepseek_v41: DSpark stage {d}'s trellis experts are {d} of {d}x{d}, config says {d} of {d}x{d}\n", .{ si, rb.experts, rb.hidden, rb.inter, c.dspark.experts, c.dim, c.moe_inter });
        return error.Dsv41WeightLayout;
    }
    // The ragged storage is a copy: evaluated, it outlives the weight map's arrays.
    try evalAll(&rb.storage());
    return rb;
}

fn initDspark(m: *Model, w: *model_mod.Weights) !void {
    const c = &m.cfg;
    if (c.dspark.block == 0) return error.Dsv41NoDspark;
    const stages = c.n_mtp;
    m.layers = try m.gpa.realloc(m.layers, c.layers + stages);
    @memset(m.layers[c.layers..], std.mem.zeroes(Layer));
    const banks = try m.gpa.alloc(BankSet, if (c.expert_dec != null) 0 else stages);
    errdefer m.gpa.free(banks);
    const stq = try m.gpa.alloc(exl3.Ragged, if (c.expert_dec != null) stages else 0);
    var stq_made: usize = 0;
    errdefer {
        for (stq[0..stq_made]) |*rb| rb.deinit();
        m.gpa.free(stq);
    }
    for (0..stages) |si| {
        m.layers[c.layers + si] = try loadLayer(m, w, "mtp", si, c.layers + si);
        if (c.expert_dec != null) {
            stq[si] = try loadStqExperts(m, w, si);
            stq_made += 1;
            continue;
        }
        const ex = c.dspark.experts;
        banks[si].gate = try loadBank(m, w, "mtp", si, "w1", ex, c.moe_inter, c.dim);
        banks[si].up = try loadBank(m, w, "mtp", si, "w3", ex, c.moe_inter, c.dim);
        banks[si].down = try loadBank(m, w, "mtp", si, "w2", ex, c.dim, c.moe_inter);
    }
    var b: NameBuf = undefined;
    var base: NameBuf = undefined;
    const last = stages - 1;
    const rank = c.dspark.rank;
    const conf = try loadDense(m, w, try lkey(&b, "mtp", last, "confidence_head.proj.weight"), .float32);
    m.dspark = .{
        .main_proj = try loadLin(m, w, try lkey(&base, "mtp", 0, "main_proj"), m.act),
        .main_norm = try loadDense(m, w, try lkey(&b, "mtp", 0, "main_norm.weight"), m.act),
        .norm = try loadDense(m, w, try lkey(&b, "mtp", last, "norm.weight"), m.act),
        .markov_embed = try loadDense(m, w, try lkey(&b, "mtp", last, "markov_head.embed.weight"), m.act),
        .markov_head = try loadDense(m, w, try lkey(&b, "mtp", last, "markov_head.head.weight"), .float32),
        .conf = conf,
        .banks = banks,
        .stq = stq,
    };
    if (dim(m.dspark.?.markov_embed, 1) != rank or dim(m.dspark.?.markov_head, 1) != rank or m.dspark.?.main_proj.outDim() != c.dim or dim(conf, 1) != c.dim + rank) {
        log.err("deepseek_v41: the DSpark heads do not match rank {d} over dim {d}\n", .{ rank, c.dim });
        return error.Dsv41WeightLayout;
    }
}

// ── tests: DeepSeek's reference on a tiny checkpoint (tests/dump_dsv41_fixtures.py) ──

const testing = std.testing;

pub fn fixtureDir() ![]const u8 {
    const raw = std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest;
    return std.mem.span(raw);
}

pub fn loadFixture(dir: []const u8, name: []const u8) !model_mod.Weights {
    const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/{s}", .{ dir, name }, 0);
    defer testing.allocator.free(path);
    var w = model_mod.Weights.init(testing.allocator);
    errdefer w.deinit();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    try model_mod.loadSafetensorsFile(testing.allocator, &w, path.ptr, s, true);
    return w;
}

pub fn tinyModel(dir: []const u8, opts: Opts, s: S) !*Model {
    const io = std.Io.Threaded.global_single_threaded.io();
    const cfg = try cfg_mod.readFile(io, testing.allocator, dir);
    var w = try source.loadResident(io, testing.allocator, dir, .{ .experts = opts.experts, .dspark = opts.dspark });
    defer w.deinit();
    return init(testing.allocator, cfg, &w, dir, s, opts);
}

pub fn hostRows(sc: *Scope, a: A) ![]f32 {
    return testing.allocator.dupe(f32, try sc_mod.hostF32(try sc.astype(a, .float32)));
}

/// Max |a - b| against `tol` times the tensor's peak magnitude (at least 1), and the cosine.
fn compare(label: []const u8, got: []const f32, want: []const f32, tol: f32) !void {
    try testing.expectEqual(want.len, got.len);
    var worst: f32 = 0;
    var dot: f64 = 0;
    var ng: f64 = 0;
    var nw: f64 = 0;
    var peak: f32 = 1;
    for (got, want) |a, b| {
        peak = @max(peak, @abs(b));
        worst = @max(worst, @abs(a - b));
        dot += @as(f64, a) * b;
        ng += @as(f64, a) * a;
        nw += @as(f64, b) * b;
    }
    const cos = dot / @sqrt(ng * nw + 1e-30);
    const atol = tol * peak;
    if (worst > atol or cos < 0.99999) {
        std.debug.print("{s}: max |diff| {e} (bar {e}), cosine {d:.7}\n", .{ label, worst, atol, cos });
        return error.FixtureMismatch;
    }
}

pub fn tokens(sc: *Scope, fx: *model_mod.Weights) ![]u32 {
    const ids = try sc_mod.hostI32(try sc.astype(fx.get("ids").?, .int32));
    const out = try testing.allocator.alloc(u32, ids.len);
    for (ids, out) |v, *o| o.* = @intCast(v);
    return out;
}

const TINY_S = 40;

test "dsv41 fixture: prefill matches DeepSeek's reference in f32 with the QAT round trips off, whole and in chunks (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const want = try hostRows(&sc, fx.get("logits_full").?);
    defer testing.allocator.free(want);
    // The GPU's f32 matmul keeps about 11 bits: the tight bar runs on the CPU stream, which is exact.
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    exact_matmul = true;
    defer exact_matmul = false;
    const m = try tinyModel(dir, .{ .act = .float32, .qat = false }, cpu);
    defer m.deinit();
    var trace: Trace = .{ .gpa = testing.allocator };
    defer trace.deinit();
    m.trace = &trace;
    {
        const logits = try extendSpan(m, ids, .all, 64);
        defer _ = mlx.mlx_array_free(logits);
        // the ladder, in the order the reference computes it: the first name to disagree is the bug
        for (trace.items.items) |it| {
            if (std.mem.startsWith(u8, it.name, "sa_") and !std.mem.endsWith(u8, it.name, "_0")) continue;
            if (std.mem.startsWith(u8, it.name, "hc_") or std.mem.startsWith(u8, it.name, "attn_in")) {
                if (!std.mem.endsWith(u8, it.name, "_0")) continue;
            }
            const ref = fx.get(it.name) orelse continue;
            const got = try hostRows(&sc, it.arr);
            defer testing.allocator.free(got);
            const want_ref = try hostRows(&sc, ref);
            defer testing.allocator.free(want_ref);
            try compare(it.name, got, want_ref, 1e-4);
        }
        const got = try hostRows(&sc, logits);
        defer testing.allocator.free(got);
        try compare("logits_full", got, want, 2e-3);
    }
    m.trace = null;
    resetState(m);
    // 13-row chunks (an odd split, not a multiple of the ratio) and 5-row spans
    var done: usize = 0;
    var all: std.ArrayList(f32) = .empty;
    defer all.deinit(testing.allocator);
    while (done < ids.len) {
        const n = @min(13, ids.len - done);
        const logits = try extendSpan(m, ids[done..][0..n], .all, 5);
        defer _ = mlx.mlx_array_free(logits);
        const got = try hostRows(&sc, logits);
        defer testing.allocator.free(got);
        try all.appendSlice(testing.allocator, got);
        done += n;
    }
    try compare("logits_chunked", all.items, want, 2e-3);
}

test "dsv41 fixture: decode one token at a time after a prefill matches the reference (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const want = try hostRows(&sc, fx.get("logits_decode").?);
    defer testing.allocator.free(want);
    // The GPU's f32 matmul keeps about 11 bits: the tight bar runs on the CPU stream, which is exact.
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    exact_matmul = true;
    defer exact_matmul = false;
    const m = try tinyModel(dir, .{ .act = .float32, .qat = false }, cpu);
    defer m.deinit();
    // The reference runs every layer over every row; CED runs the decoder over the last window only.
    m.ced_split = null;
    const pre = try extend(m, ids[0..TINY_S], .last);
    _ = mlx.mlx_array_free(pre);
    const vocab = m.head.outDim();
    for (ids[TINY_S..], 0..) |t, i| {
        const logits = try extend(m, &.{t}, .last);
        defer _ = mlx.mlx_array_free(logits);
        const got = try hostRows(&sc, logits);
        defer testing.allocator.free(got);
        try compare("logits_decode", got, want[i * vocab ..][0..vocab], 2e-3);
    }
}

test "dsv41 fixture: with the QAT round trips on, the layers up to the first rounding flip match the reference (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture.safetensors");
    defer fx.deinit();
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    exact_matmul = true;
    defer exact_matmul = false;
    const m = try tinyModel(dir, .{ .act = .float32, .qat = true }, cpu);
    defer m.deinit();
    var trace: Trace = .{ .gpa = testing.allocator };
    defer trace.deinit();
    m.trace = &trace;
    const logits = try extendSpan(m, ids, .all, 64);
    defer _ = mlx.mlx_array_free(logits);
    // A quantizer step is a jump: an input within an ulp of a midpoint lands on the other side, so
    // only the first layers are held tight and the logits to a cosine.
    for (trace.items.items) |it| {
        const layer = std.fmt.parseInt(usize, it.name[std.mem.lastIndexOfScalar(u8, it.name, '_').? + 1 ..], 10) catch continue;
        if (layer > 1 or !(std.mem.startsWith(u8, it.name, "attn_") or std.mem.startsWith(u8, it.name, "ffn_") or std.mem.startsWith(u8, it.name, "stream_"))) continue;
        const ref = fx.get(it.name) orelse continue;
        const got = try hostRows(&sc, it.arr);
        defer testing.allocator.free(got);
        const want_ref = try hostRows(&sc, ref);
        defer testing.allocator.free(want_ref);
        try compare(it.name, got, want_ref, 1e-3);
    }
    const got = try hostRows(&sc, logits);
    defer testing.allocator.free(got);
    const want = try hostRows(&sc, fx.get("logits_full").?);
    defer testing.allocator.free(want);
    var dot: f64 = 0;
    var ng: f64 = 0;
    var nw: f64 = 0;
    for (got, want) |a, b| {
        dot += @as(f64, a) * b;
        ng += @as(f64, a) * a;
        nw += @as(f64, b) * b;
    }
    const cos = dot / @sqrt(ng * nw);
    if (cos < 0.99) {
        std.debug.print("logits_full: cosine {d:.5}\n", .{cos});
        return error.FixtureMismatch;
    }
}

/// Logits of a prefill over `ids[0..TINY_S]` (every row, composed) then single-token decodes of the rest with the
/// kernels `fused` or not, concatenated; `first` collects the first decode step's layer traces.
fn runTiny(m: *Model, ids: []const u32, fused: bool, first: *Trace) ![]f32 {
    var all: std.ArrayList(f32) = .empty;
    errdefer all.deinit(testing.allocator);
    resetState(m);
    m.fused = false;
    const pre = try extend(m, ids[0..TINY_S], .last);
    _ = mlx.mlx_array_free(pre);
    m.fused = fused;
    for (ids[TINY_S..], 0..) |t, i| {
        m.trace = if (i == 0) first else null;
        const logits = try extend(m, &.{t}, .last);
        defer _ = mlx.mlx_array_free(logits);
        try all.appendSlice(testing.allocator, try sc_mod.hostF32(logits));
    }
    m.trace = null;
    return all.toOwnedSlice(testing.allocator);
}

/// The layers before a routing near-tie can flip an expert: the two arms agree there to the GPU's f32 matmul
/// noise, past it only the logits' direction is held.
fn expectTracesAgree(sc: *Scope, got: *const Trace, want: *const Trace, tol: f32) !void {
    var checked: usize = 0;
    for (want.items.items) |it| {
        const layer = std.fmt.parseInt(usize, it.name[std.mem.lastIndexOfScalar(u8, it.name, '_').? + 1 ..], 10) catch continue;
        if (layer > 2 or !(std.mem.startsWith(u8, it.name, "attn_") or std.mem.startsWith(u8, it.name, "ffn_") or std.mem.startsWith(u8, it.name, "stream_"))) continue;
        const a = try hostRows(sc, got.get(it.name) orelse continue);
        defer testing.allocator.free(a);
        const b = try hostRows(sc, it.arr);
        defer testing.allocator.free(b);
        try compare(it.name, a, b, tol);
        checked += 1;
    }
    try testing.expect(checked > 0);
}

test "dsv41 fused: decode with the kernels follows the composed path on the GPU in f32 (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const m = try tinyModel(dir, .{ .act = .float32, .qat = false }, mlx.gpuStream());
    defer m.deinit();
    var tr: [2]Trace = .{ .{ .gpa = testing.allocator }, .{ .gpa = testing.allocator } };
    defer for (&tr) |*t| t.deinit();
    const want = try runTiny(m, ids, false, &tr[0]);
    defer testing.allocator.free(want);
    const got = try runTiny(m, ids, true, &tr[1]);
    defer testing.allocator.free(got);
    try expectTracesAgree(&sc, &tr[1], &tr[0], 4e-3);
    var dot: f64 = 0;
    var ng: f64 = 0;
    var nw: f64 = 0;
    for (got, want) |a, b| {
        dot += @as(f64, a) * b;
        ng += @as(f64, a) * a;
        nw += @as(f64, b) * b;
    }
    if (dot / @sqrt(ng * nw) < 0.99) return error.FixtureMismatch;
}

test "dsv41 fused: prefill with the kernels follows the composed path on the GPU in f32 (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const m = try tinyModel(dir, .{ .act = .float32, .qat = false }, mlx.gpuStream());
    defer m.deinit();
    var tr: [2]Trace = .{ .{ .gpa = testing.allocator }, .{ .gpa = testing.allocator } };
    defer for (&tr) |*t| t.deinit();
    var logits: [2][]f32 = undefined;
    for ([_]bool{ false, true }, 0..) |fused, k| {
        resetState(m);
        m.fused = fused;
        m.trace = &tr[k];
        const lg = try extendSpan(m, ids[0..TINY_S], .all, 64);
        defer _ = mlx.mlx_array_free(lg);
        logits[k] = try hostRows(&sc, lg);
    }
    m.trace = null;
    defer for (logits) |l| testing.allocator.free(l);
    try expectTracesAgree(&sc, &tr[1], &tr[0], 4e-3);
    var dot: f64 = 0;
    var ng: f64 = 0;
    var nw: f64 = 0;
    for (logits[0], logits[1]) |a, b| {
        dot += @as(f64, a) * b;
        ng += @as(f64, a) * a;
        nw += @as(f64, b) * b;
    }
    if (dot / @sqrt(ng * nw) < 0.99) return error.FixtureMismatch;
}

test "dsv41 loader: the composed diagnostic turns the fused kernels off and says so (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const m = try tinyModel(try fixtureDir(), .{ .act = .float32, .qat = false }, mlx.gpuStream());
    defer m.deinit();
    try testing.expect(m.fused);
    setFused(m, true);
    try testing.expect(!m.fused);
    setFused(m, false);
    try testing.expect(m.fused);
}

test "dsv41 fused: the f16 attention arm gives the f32 chain's first-layer heads to bf16 ulps (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const m = try tinyModel(dir, .{ .act = .bfloat16, .qat = false }, mlx.gpuStream());
    defer m.deinit();
    var tr: [2]Trace = .{ .{ .gpa = testing.allocator }, .{ .gpa = testing.allocator } };
    defer for (&tr) |*t| t.deinit();
    for ([_]bool{ false, true }, 0..) |f16_arm, k| {
        resetState(m);
        m.attn_f16 = f16_arm;
        m.trace = &tr[k];
        const lg = try extendSpan(m, ids[0..TINY_S], .all, 16);
        _ = mlx.mlx_array_free(lg);
    }
    m.trace = null;
    // Later layers drift apart through bf16 routing near-ties; the first layer is the wiring check.
    const a = try hostRows(&sc, tr[0].get("sa_o_0").?);
    defer testing.allocator.free(a);
    const b = try hostRows(&sc, tr[1].get("sa_o_0").?);
    defer testing.allocator.free(b);
    try compare("sa_o_0", b, a, 1.0 / 64.0);
}

// ── DSpark: the drafter's heads against the reference, and a verify round's rollback ──

fn fixtureIds(sc: *Scope, fx: *model_mod.Weights, name: []const u8) ![]u32 {
    const host = try sc_mod.hostI32(try sc.astype(fx.get(name).?, .int32));
    const out = try testing.allocator.alloc(u32, host.len);
    for (host, out) |v, *o| o.* = @intCast(v);
    return out;
}

fn argmaxRow(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = i;
    };
    return @intCast(best);
}

test "dsv41 DSpark: the drafted block, its Markov bias and its confidence match DeepSeek's reference, QAT off (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var sc = Scope.init(testing.allocator, cpu);
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const t1 = try fixtureIds(&sc, &fx, "ds_t1");
    defer testing.allocator.free(t1);
    const want_ids = try fixtureIds(&sc, &fx, "ds_ids");
    defer testing.allocator.free(want_ids);
    const want_logits = try hostRows(&sc, fx.get("ds_logits").?);
    defer testing.allocator.free(want_logits);
    const want_conf = try hostRows(&sc, fx.get("ds_conf").?);
    defer testing.allocator.free(want_conf);
    const want_mainx = try hostRows(&sc, fx.get("ds_mainx_prefill").?);
    defer testing.allocator.free(want_mainx);
    exact_matmul = true;
    defer exact_matmul = false;
    const m = try tinyModel(dir, .{ .act = .float32, .qat = false, .dspark = true }, cpu);
    defer m.deinit();
    // The reference runs every layer over every row; CED runs the decoder over the last window only.
    m.ced_split = null;
    const B = m.cfg.dspark.block;
    const V = m.cfg.vocab;

    var trace: Trace = .{ .gpa = testing.allocator };
    defer trace.deinit();
    m.trace = &trace;
    const pre = try extend(m, ids[0..TINY_S], .last);
    _ = mlx.mlx_array_free(pre);
    m.trace = null;
    const got_mainx = try hostRows(&sc, trace.get("ds_mainx").?);
    defer testing.allocator.free(got_mainx);
    try compare("ds_mainx_prefill", got_mainx, want_mainx, 1e-4);

    for (ids[TINY_S..], 0..) |tok, k| {
        const lg = try extend(m, &.{tok}, .last);
        _ = mlx.mlx_array_free(lg);
        var d = try draft(m, t1[k], true);
        defer d.deinit();
        try testing.expectEqualSlices(u32, want_ids[k * (B + 1) ..][0 .. B + 1], d.ids);
        try compare("ds_logits", d.logits, want_logits[k * B * V ..][0 .. B * V], 2e-3);
        try compare("ds_conf", d.conf, want_conf[k * B ..][0..B], 2e-3);
    }
}

test "dsv41 DSpark: with the QAT round trips on, the stages draft the reference's block from the reference's main rows (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture.safetensors");
    defer fx.deinit();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var sc = Scope.init(testing.allocator, cpu);
    defer sc.deinit();
    const t1 = try fixtureIds(&sc, &fx, "ds_t1");
    defer testing.allocator.free(t1);
    const want_logits = try hostRows(&sc, fx.get("ds_logits").?);
    defer testing.allocator.free(want_logits);
    exact_matmul = true;
    defer exact_matmul = false;
    const m = try tinyModel(dir, .{ .act = .float32, .qat = true, .dspark = true }, cpu);
    defer m.deinit();
    const B = m.cfg.dspark.block;
    const V = m.cfg.vocab;
    // The trunk's own rounding flips would move the main rows, so the reference's rows feed the stages directly.
    const st = try ensureState(m);
    const prefill_rows = fx.get("ds_mh_prefill").?;
    try commitMain(&sc, m, st, prefill_rows, 0);
    st.n = dim(prefill_rows, 0);
    const step_rows = fx.get("ds_mh").?;
    for (0..dim(step_rows, 0)) |k| {
        try commitMain(&sc, m, st, try sc.slice(step_rows, 0, k, k + 1), st.n);
        st.n += 1;
        var d = try draft(m, t1[k], true);
        defer d.deinit();
        var dot: f64 = 0;
        var ng: f64 = 0;
        var nw: f64 = 0;
        for (d.logits, want_logits[k * B * V ..][0 .. B * V]) |a, b| {
            dot += @as(f64, a) * b;
            ng += @as(f64, a) * a;
            nw += @as(f64, b) * b;
        }
        // a rounding flip is a jump, so the logits are held to their direction
        const cos = dot / @sqrt(ng * nw);
        if (cos <= 0.99) std.debug.print("ds step {d} cos {d:.5}\n", .{ k, cos });
        try testing.expect(cos > 0.99);
    }
}

/// A model over the tiny pack prefilled with `ids[0..n]`; the logits of its last row.
fn prefilledTiny(dir: []const u8, ids: []const u32, n: usize, s: S, last: *[]f32) !*Model {
    const m = try tinyModel(dir, .{ .act = .float32, .qat = false, .dspark = true }, s);
    errdefer m.deinit();
    const lg = try extend(m, ids[0..n], .last);
    defer _ = mlx.mlx_array_free(lg);
    last.* = try testing.allocator.dupe(f32, try sc_mod.hostF32(lg));
    return m;
}

test "dsv41 DSpark: a verify round keeps exactly its accepted prefix, whatever the count, and the next steps follow serial decode (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var sc = Scope.init(testing.allocator, cpu);
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    exact_matmul = true;
    defer exact_matmul = false;
    const P = TINY_S - 5; // an odd pooling-group boundary against the ratio-2 layers
    var first: []f32 = undefined;
    const serial = try prefilledTiny(dir, ids, P, cpu, &first);
    defer serial.deinit();
    defer testing.allocator.free(first);
    const B = serial.cfg.dspark.block;
    const V = serial.cfg.vocab;
    // the serial continuation: greedy tokens and the logits that picked them
    const G = 24;
    var chain: [G]u32 = undefined;
    var chain_logits: [G][]f32 = undefined;
    var made: usize = 0;
    defer for (chain_logits[0..made]) |l| testing.allocator.free(l);
    chain[0] = argmaxRow(first);
    for (0..G - 1) |i| {
        const lg = try extend(serial, chain[i .. i + 1], .last);
        defer _ = mlx.mlx_array_free(lg);
        chain_logits[i] = try testing.allocator.dupe(f32, try sc_mod.hostF32(lg));
        made = i + 1;
        chain[i + 1] = argmaxRow(chain_logits[i]);
    }

    for (0..B + 1) |accept| {
        var l0: []f32 = undefined;
        const m = try prefilledTiny(dir, ids, P, cpu, &l0);
        defer m.deinit();
        defer testing.allocator.free(l0);
        // [t1, the next `accept` serial tokens, then a token the trunk will not pick]
        var round: [16]u32 = undefined;
        round[0] = chain[0];
        for (0..B) |k| round[1 + k] = if (k < accept) chain[1 + k] else (chain[1 + k] + 1) % @as(u32, @intCast(V));
        const vl = try verify(m, round[0 .. B + 1]);
        defer _ = mlx.mlx_array_free(vl);
        const rows = try sc_mod.hostF32(vl);
        var accepted: usize = 0;
        while (accepted < B and argmaxRow(rows[accepted * V ..][0..V]) == round[1 + accepted]) accepted += 1;
        try testing.expectEqual(accept, accepted);
        try finishVerify(m, accepted);
        try testing.expectEqual(P + 1 + accepted, (try ensureState(m)).n);
        // the rolled-back model decodes on as serial does
        for (accepted + 1..accepted + 5) |i| {
            const lg = try extend(m, chain[i .. i + 1], .last);
            defer _ = mlx.mlx_array_free(lg);
            try compare("rolled-back decode", try sc_mod.hostF32(lg), chain_logits[i], 2e-3);
        }
        // and a second round over the rolled-back state agrees too
        const at = accepted + 5;
        var round2: [16]u32 = undefined;
        round2[0] = chain[at];
        for (0..B) |k| round2[1 + k] = chain[at + 1 + k];
        const vl2 = try verify(m, round2[0 .. B + 1]);
        defer _ = mlx.mlx_array_free(vl2);
        const rows2 = try sc_mod.hostF32(vl2);
        for (0..B + 1) |k| try compare("second round row", rows2[k * V ..][0..V], chain_logits[at + k], 2e-3);
        try finishVerify(m, B);
    }
}

test "dsv41 DSpark: after a rollback the next draft is the serial model's draft (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var sc = Scope.init(testing.allocator, cpu);
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    exact_matmul = true;
    defer exact_matmul = false;
    const P = TINY_S - 3;
    var l0: []f32 = undefined;
    const spec = try prefilledTiny(dir, ids, P, cpu, &l0);
    defer spec.deinit();
    defer testing.allocator.free(l0);
    var l1: []f32 = undefined;
    const serial = try prefilledTiny(dir, ids, P, cpu, &l1);
    defer serial.deinit();
    defer testing.allocator.free(l1);
    const B = spec.cfg.dspark.block;
    const V = spec.cfg.vocab;
    // three rounds with real drafts, the serial model walking the committed tokens one at a time
    var t1 = argmaxRow(l0);
    for (0..3) |_| {
        var d = try draft(spec, t1, false);
        defer d.deinit();
        const vl = try verify(spec, d.ids[0 .. B + 1]);
        defer _ = mlx.mlx_array_free(vl);
        const rows = try sc_mod.hostF32(vl);
        var accepted: usize = 0;
        while (accepted < B and argmaxRow(rows[accepted * V ..][0..V]) == d.ids[1 + accepted]) accepted += 1;
        const next = argmaxRow(rows[accepted * V ..][0..V]);
        try finishVerify(spec, accepted);
        for (d.ids[0 .. accepted + 1]) |tok| {
            const lg = try extend(serial, &.{tok}, .last);
            _ = mlx.mlx_array_free(lg);
        }
        var want = try draft(serial, next, true);
        defer want.deinit();
        var got = try draft(spec, next, true);
        defer got.deinit();
        try testing.expectEqualSlices(u32, want.ids, got.ids);
        try compare("draft after rollback", got.logits, want.logits, 2e-3);
        t1 = next;
    }
}

/// Bytes of every array a DSpark load holds beyond the trunk.
fn dsparkHeldBytes(m: *const Model) u64 {
    const lin = struct {
        fn of(l: Lin) u64 {
            if (l.stq) |q| return arrayBytes(q.p.trellis) + arrayBytes(q.p.suh) + arrayBytes(q.p.svh);
            return arrayBytes(l.w) + arrayBytes(l.s);
        }
    }.of;
    var bytes: u64 = 0;
    for (m.layers[m.cfg.layers..]) |ly| {
        for ([_]A{ ly.attn_norm, ly.ffn_norm, ly.q_norm, ly.kv_norm, ly.sink, ly.gate_w, ly.gate_bias, ly.hc_attn.fn_w, ly.hc_attn.scale, ly.hc_attn.base, ly.hc_ffn.fn_w, ly.hc_ffn.scale, ly.hc_ffn.base }) |a| bytes += arrayBytes(a);
        for ([_]Lin{ ly.wq_a, ly.wq_b, ly.wkv, ly.wo_a, ly.wo_b, ly.shared[0], ly.shared[1], ly.shared[2] }) |l| bytes += lin(l);
    }
    const ds = m.dspark.?;
    for (ds.banks) |b| for ([_]Bank{ b.gate, b.up, b.down }) |q| {
        bytes += arrayBytes(q.w) + arrayBytes(q.s);
    };
    for (ds.stq) |rb| for (rb.storage()[0..9]) |a| {
        bytes += arrayBytes(a);
    };
    for ([_]A{ ds.main_norm, ds.norm, ds.markov_embed, ds.markov_head, ds.conf }) |a| bytes += arrayBytes(a);
    return bytes + lin(ds.main_proj);
}

test "dsv41 DSpark bills: the load's resident bytes and the state's windows are what the model holds (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    const io = std.Io.Threaded.global_single_threaded.io();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const with = try tinyModel(dir, .{ .act = .bfloat16, .dspark = true }, cpu);
    defer with.deinit();
    const without = try tinyModel(dir, .{ .act = .bfloat16 }, cpu);
    defer without.deinit();
    try testing.expectEqual(try source.dsparkBytes(io, testing.allocator, dir), dsparkHeldBytes(with));
    try testing.expect(source.shipsDspark(io, testing.allocator, dir));

    var st_with = try newState(with);
    defer st_with.deinit();
    var st_without = try newState(without);
    defer st_without.deinit();
    try testing.expectEqual(with.cfg.stateFixedBytes(true) - with.cfg.stateFixedBytes(false), stateBytes(&st_with) - stateBytes(&st_without));
}

test "dsv41 index stream: a prefill whose index top-k streams over key tiles gives the whole scores' logits bit for bit, packed keys, the candidate masks and decode after it (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const m = try tinyModel(dir, .{ .act = .float32 }, s);
    defer m.deinit();
    var rows: [2][]f32 = undefined;
    var streamed: [2]usize = undefined;
    for (&rows, &streamed, [_]usize{ INDEX_STREAM_TILE, 32 }) |*r, *n, tile| {
        resetState(m);
        m.index_stream_tile = tile;
        const before = index_streams.load(.monotonic);
        var all: std.ArrayList(f32) = .empty;
        errdefer all.deinit(testing.allocator);
        for ([_]usize{ TINY_S, 1, 1, 3 }, [_]usize{ 0, TINY_S, TINY_S + 1, TINY_S + 2 }) |len, at| {
            const lg = try extend(m, ids[at..][0..len], .all);
            defer _ = mlx.mlx_array_free(lg);
            const h = try hostRows(&sc, lg);
            defer testing.allocator.free(h);
            try all.appendSlice(testing.allocator, h);
        }
        r.* = try all.toOwnedSlice(testing.allocator);
        n.* = index_streams.load(.monotonic) - before;
    }
    defer for (rows) |r| testing.allocator.free(r);
    try testing.expectEqual(@as(usize, 0), streamed[0]);
    try testing.expect(streamed[1] >= 2);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(rows[0]), std.mem.sliceAsBytes(rows[1]));
}

test "dsv41 packed KV: compressed rows and index keys kept as FP4 give the bf16 rows' logits bit for bit, through decode, a verify and its rollback (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    var ms: [2]*Model = undefined;
    for (&ms, [_]bool{ false, true }) |*m, pk| {
        m.* = try tinyModel(dir, .{ .act = .bfloat16, .dspark = true }, s);
        m.*.pack_kv = pk;
    }
    defer for (ms) |m| m.deinit();
    const B = ms[0].cfg.dspark.block;
    var at: usize = 0;
    for ([_]usize{ 29, 1, 1, 3 }) |n| {
        var rows: [2][]f32 = undefined;
        for (ms, &rows) |m, *r| {
            const lg = try extend(m, ids[at..][0..n], .all);
            defer _ = mlx.mlx_array_free(lg);
            r.* = try hostRows(&sc, lg);
        }
        defer for (rows) |r| testing.allocator.free(r);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(rows[0]), std.mem.sliceAsBytes(rows[1]));
        at += n;
    }
    // A verify chunk that keeps two of its rows, then decode on.
    var round: [16]u32 = undefined;
    @memcpy(round[0 .. B + 1], ids[at..][0 .. B + 1]);
    for (ms) |m| {
        const vl = try verify(m, round[0 .. B + 1]);
        _ = mlx.mlx_array_free(vl);
        try finishVerify(m, 1);
    }
    at += 2;
    var rows: [2][]f32 = undefined;
    for (ms, &rows) |m, *r| {
        const lg = try extend(m, ids[at..][0..1], .last);
        defer _ = mlx.mlx_array_free(lg);
        r.* = try hostRows(&sc, lg);
    }
    defer for (rows) |r| testing.allocator.free(r);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(rows[0]), std.mem.sliceAsBytes(rows[1]));
    // The packed state stays within its bill: the rows per token, the fixed rings and groups, and the growth slack.
    const st = try ensureState(ms[1]);
    const c = &ms[1].cfg;
    try testing.expect(stateBytes(st) <= c.stateFixedBytes(true) + 64 * c.stateBytesPerToken() * (cfg_mod.rows_grow_div + 1) / cfg_mod.rows_grow_div + 64 * 1024);
    try testing.expect(stateBytes(st) < stateBytes(try ensureState(ms[0])));
}

const Run = struct {
    rows: std.ArrayList(f32) = .empty,
    slots: []Slot = &.{},

    fn deinit(self: *Run) void {
        self.rows.deinit(testing.allocator);
        for (self.slots) |*sl| freeSlot(sl);
        testing.allocator.free(self.slots);
    }

    fn take(self: *Run, sc: *Scope, lg: A) !void {
        defer _ = mlx.mlx_array_free(lg);
        const h = try hostRows(sc, lg);
        defer testing.allocator.free(h);
        try self.rows.appendSlice(testing.allocator, h);
    }
};

/// What a request does from position `p` of `ids`: odd-width prefill chunks, decode, a verify round that keeps two
/// rows, decode on; every logit row and the state it ends in.
fn continueFrom(m: *Model, sc: *Scope, ids: []const u32, p: usize) !Run {
    var run: Run = .{};
    errdefer run.deinit();
    var at = p;
    for ([_]usize{ 7, 4, 1, 1 }) |n| {
        try run.take(sc, try extend(m, ids[at..][0..n], .all));
        at += n;
    }
    const B = m.cfg.dspark.block;
    try run.take(sc, try verify(m, ids[at..][0 .. B + 1]));
    try finishVerify(m, 1);
    at += 2;
    try run.take(sc, try extend(m, ids[at..][0..1], .last));
    run.slots = try testing.allocator.alloc(Slot, slotCount(m));
    @memset(run.slots, nil_slot);
    try captureState(m, run.slots);
    return run;
}

fn expectSlotsBits(sc: *Scope, want: []const Slot, got: []const Slot) !void {
    try testing.expectEqual(want.len, got.len);
    for (want, got) |x, y| for (x, y) |u, v| {
        try testing.expectEqual(u.ctx == null, v.ctx == null);
        if (u.ctx == null) continue;
        try testing.expectEqualSlices(c_int, mlx.getShape(u), mlx.getShape(v));
        try testing.expectEqual(mlx.mlx_array_dtype(u), mlx.mlx_array_dtype(v));
        const hu = try hostRows(sc, u);
        defer testing.allocator.free(hu);
        const hv = try hostRows(sc, v);
        defer testing.allocator.free(hv);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(hu), std.mem.sliceAsBytes(hv));
    };
}

test "dsv41 restore point: a state restored at a position continues bit for bit like the cold run, through prefill, decode and a verify round (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    for ([_]bool{ true, false }) |qat| {
        const m = try tinyModel(dir, .{ .act = .bfloat16, .qat = qat, .dspark = true }, s);
        defer m.deinit();
        // odd positions leave an open pooling group on the ratio-2 layers; past 8 the window ring has wrapped
        for ([_]usize{ 5, 9, 16, 21 }) |p| {
            resetState(m);
            _ = mlx.mlx_array_free(try extend(m, ids[0..p], .last));
            const cp = try testing.allocator.alloc(Slot, slotCount(m));
            @memset(cp, nil_slot);
            defer {
                for (cp) |*sl| freeSlot(sl);
                testing.allocator.free(cp);
            }
            try captureState(m, cp);
            var cold = try continueFrom(m, &sc, ids, p);
            defer cold.deinit();

            try restoreState(m, cp, p, ids);
            try testing.expectEqual(p, (try ensureState(m)).n);
            var warm = try continueFrom(m, &sc, ids, p);
            defer warm.deinit();
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(cold.rows.items), std.mem.sliceAsBytes(warm.rows.items));
            try expectSlotsBits(&sc, cold.slots, warm.slots);

            // the checkpoint was shared, never written: a second restore gives the same run again
            try restoreState(m, cp, p, ids);
            var again = try continueFrom(m, &sc, ids, p);
            defer again.deinit();
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(cold.rows.items), std.mem.sliceAsBytes(again.rows.items));
        }
    }
}

test "dsv41 restore point: a checkpoint of another position or shape is refused and leaves the state reset (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const m = try tinyModel(dir, .{ .act = .bfloat16, .dspark = true }, s);
    defer m.deinit();
    _ = mlx.mlx_array_free(try extend(m, ids[0..9], .last));
    const cp = try testing.allocator.alloc(Slot, slotCount(m));
    @memset(cp, nil_slot);
    defer {
        for (cp) |*sl| freeSlot(sl);
        testing.allocator.free(cp);
    }
    try captureState(m, cp);
    // the rows say position 9; claiming 12 would read rows that are not there
    try testing.expectError(error.Dsv41CheckpointMismatch, restoreState(m, cp, 12, ids));
    try testing.expectEqual(@as(usize, 0), (try ensureState(m)).n);
    try testing.expectError(error.Dsv41CheckpointMismatch, restoreState(m, cp[1..], 9, ids));
    try testing.expectError(error.Dsv41CheckpointMismatch, restoreState(m, cp, 9, ids[0..4]));
}

/// Cosine and the largest absolute difference over the peak of two rows.
fn rowDiff(a: []const f32, b: []const f32) struct { cos: f64, err: f64, same_max: bool } {
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    var worst: f64 = 0;
    var peak: f64 = 0;
    var ia: usize = 0;
    var ib: usize = 0;
    for (a, b, 0..) |x, y, k| {
        if (x > a[ia]) ia = k;
        if (y > b[ib]) ib = k;
        dot += @as(f64, x) * y;
        aa += @as(f64, x) * x;
        bb += @as(f64, y) * y;
        worst = @max(worst, @abs(@as(f64, x) - y));
        peak = @max(peak, @abs(@as(f64, y)));
    }
    return .{ .cos = dot / @sqrt(aa * bb), .err = worst / peak, .same_max = ia == ib };
}

/// One prefill of `ids` in `chunks`, then `steps` one-token decodes: the last prefill row and each decode row, and the
/// state the prefill left.
fn cedArm(m: *Model, sc: *Scope, ids: []const u32, chunks: []const usize, steps: usize, split: ?usize) !Run {
    m.ced_split = split;
    resetState(m);
    var run: Run = .{};
    errdefer run.deinit();
    var at: usize = 0;
    var last = none;
    for (chunks) |n| {
        if (last.ctx != null) _ = mlx.mlx_array_free(last);
        last = try extend(m, ids[at..][0..n], .last);
        at += n;
    }
    try run.take(sc, last);
    run.slots = try testing.allocator.alloc(Slot, slotCount(m));
    @memset(run.slots, nil_slot);
    try captureState(m, run.slots);
    for (0..steps) |i| try run.take(sc, try extend(m, ids[at + i ..][0..1], .last));
    return run;
}

test "dsv41 CED: a prefill's encoder state and the decoder's compressed rows are the full forward's bits, and the same prompt in other chunks gives the same state and logits (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    // The CPU stream in f32 with exact matmuls: the GPU forward is not batch-invariant, and on the random tiny model
    // its noise flips index picks under a chunking change with or without CED.
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    exact_matmul = true;
    defer exact_matmul = false;
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const m = try tinyModel(dir, .{ .act = .float32, .qat = false, .dspark = true }, s);
    defer m.deinit();
    const split = m.ced_split orelse return error.TestUnexpectedResult;
    const w: usize = m.cfg.window;
    const st0 = try ensureState(m);
    try testing.expectEqual(@as(?usize, null), cedRows(m, st0, w, .last));
    try testing.expectEqual(@as(?usize, w), cedRows(m, st0, w + 1, .last));
    try testing.expectEqual(@as(?usize, null), cedRows(m, st0, w + 1, .all));
    const report = @import("transformer.zig").diagEnvOn("SUSHI_DSV41_CED_REPORT");
    const N: usize = 40;
    const steps = ids.len - N;
    const V: usize = m.cfg.vocab;
    var whole = try cedArm(m, &sc, ids, &.{N}, steps, split);
    defer whole.deinit();
    // Every layer before the split and the split's compressor run every row: the full forward's bytes.
    var full = try cedArm(m, &sc, ids, &.{N}, 0, null);
    defer full.deinit();
    try expectSlotsBits(&sc, full.slots[0 .. 2 * split], whole.slots[0 .. 2 * split]);
    try expectSlotsBits(&sc, full.slots[2 * split + 1 ..][0..1], whole.slots[2 * split + 1 ..][0..1]);
    // A short first chunk (fewer rows than the window: the full forward) or a replayed one, then a replayed chunk whose
    // decoder window starts again at its own tail: the prefill's last row and the decode steps after it.
    const plans = [_][]const usize{ &.{ w - 3, N - w + 3 }, &.{ 13, N - 13 }, &.{ 2 * w + 4, N - 2 * w - 4 } };
    for (plans) |plan| {
        var run = try cedArm(m, &sc, ids, plan, steps, split);
        defer run.deinit();
        for (0..steps + 1) |r| {
            const d = rowDiff(run.rows.items[r * V ..][0..V], whole.rows.items[r * V ..][0..V]);
            if (report) std.debug.print("CED plan {any} row {d}: cos {d:.7} max err/peak {e:.2} argmax {} \n", .{ plan, r, d.cos, d.err, d.same_max });
            try testing.expect(d.err < 1e-4 and d.same_max);
        }
    }
}


test "dsv41 Engram: each table keeps a row cache of exactly the bytes the load bills (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    const m = try tinyModel(dir, .{ .act = .bfloat16, .qat = true }, mlx.gpuStream());
    defer m.deinit();
    var held: u64 = 0;
    for (m.engrams) |e| {
        const c = e.table.cache orelse return error.TestExpectedRowCache;
        held += c.keys.len * 4 + c.ages.len * 4 + c.w.len + c.s.len;
    }
    try testing.expect(held > 0);
    try testing.expectEqual(engram.cacheBytes(m.cfg.engram), held);
}

test "dsv41 Engram: a chunk's rows read ahead on a thread give the bytes of the rows read at the layer, wide and decode (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const m = try tinyModel(dir, .{ .act = .bfloat16, .qat = true }, s);
    defer m.deinit();
    try testing.expect(m.engrams.len > 0 and ids.len > NARROW_ROWS + 1);
    const prompt = ids[0 .. ids.len - 1];
    var rows: [2][2][]f32 = undefined;
    for (&rows, [_]bool{ true, false }) |*r, ahead| {
        resetState(m);
        const st = try ensureState(m);
        for ([_][]const u32{ prompt, ids[ids.len - 1 ..] }, r) |chunk, *out| {
            var p = try beginPass(m, st, chunk, st.n, SPAN);
            defer p.deinit(m.gpa);
            planCed(m, &p, .last);
            if (ahead) try prefetchEngram(m, &p);
            try testing.expectEqual(ahead, p.engram_pf != null);
            for (0..m.cfg.layers) |li| try passLayer(m, &p, li);
            const lg = try finishPass(m, &p, .last);
            defer _ = mlx.mlx_array_free(lg);
            out.* = try hostRows(&sc, lg);
        }
    }
    defer for (rows) |pair| for (pair) |r| testing.allocator.free(r);
    for (rows[0], rows[1]) |a, b| try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(b), std.mem.sliceAsBytes(a));
}

const AttnCase = struct { L: usize, wp: usize, K: usize, nc: usize, holes: bool = true, H: usize = 64, hd: usize = 512, window: usize = 128 };

const AttnData = struct { q: A, keys: A, comp: A, topk: A, band: A, sink: A };

/// bf16 rows at the release's attention shape: a span of `c.L` queries over the window rows from `c.wp` and `c.K`
/// picks into `c.nc` compressed rows, some picks -1 and some whole rows of them.
fn attnCaseData(sc: *Scope, c: AttnCase, seed: u64) !AttnData {
    const H: c_int = @intCast(c.H);
    const hd: c_int = @intCast(c.hd);
    const window = c.window;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const wk = c.wp + c.L;
    const rows = struct {
        fn make(s: *Scope, r: std.Random, shape: []const c_int, spread: f32) !A {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const v = try s.a.alloc(f32, n);
            defer s.a.free(v);
            for (v) |*x| x.* = (r.float(f32) * 2 - 1) * spread;
            const a = try s.astype(try s.floats(v, shape), .bfloat16);
            try mlx.check(mlx.mlx_array_eval(a));
            return a;
        }
    };
    const picks = try sc.a.alloc(i32, c.L * @max(c.K, 1));
    defer sc.a.free(picks);
    for (0..c.L) |l| for (0..c.K) |i| {
        const hole = c.holes and (l % 7 == 3 or rnd.uintLessThan(u32, 9) == 0);
        picks[l * c.K + i] = if (hole) -1 else @intCast(rnd.uintLessThan(u32, @intCast(c.nc)));
    };
    const kpos = try sc.reshape(try sc.arange(0, wk), &.{ 1, ci(wk) });
    const qpos = try sc.add(try sc.reshape(try sc.arange(0, c.L), &.{ ci(c.L), 1 }), try sc.i(ci(c.wp)));
    const band = try sc.op2(mlx.mlx_logical_and, try sc.op2(mlx.mlx_less_equal, kpos, qpos), try sc.op2(mlx.mlx_greater, kpos, try sc.sub(qpos, try sc.i(ci(window)))));
    return .{
        .q = try rows.make(sc, rnd, &.{ ci(c.L), H, hd }, 3),
        .keys = try rows.make(sc, rnd, &.{ ci(wk), hd }, 3),
        .comp = if (c.K > 0) try rows.make(sc, rnd, &.{ ci(c.nc), hd }, 3) else none,
        .topk = if (c.K > 0) try sc.ints(picks[0 .. c.L * c.K], &.{ ci(c.L), ci(c.K) }) else none,
        .band = band,
        .sink = try rows.make(sc, rnd, &.{ H, 1 }, 2),
    };
}

fn maxRelErr(got: []const f32, want: []const f32) f64 {
    var peak: f64 = 1e-30;
    for (want) |v| peak = @max(peak, @abs(v));
    var worst: f64 = 0;
    for (got, want) |g, w| {
        const e = @abs(@as(f64, g) - @as(f64, w));
        if (std.math.isNan(e)) return std.math.inf(f64);
        worst = @max(worst, e);
    }
    return worst / peak;
}

test "dsv41 fused attention: the one-pass kernel is no further from the f32 truth than the composed core (release span shapes)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const gpu = mlx.gpuStream();
    if (!mlx.streamIsGpu(gpu)) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, gpu);
    defer sc.deinit();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var cs = Scope.init(testing.allocator, cpu);
    defer cs.deinit();
    const cases = [_]AttnCase{
        .{ .L = 256, .wp = 127, .K = 512, .nc = 700 },
        .{ .L = 256, .wp = 127, .K = 0, .nc = 0 },
        .{ .L = 100, .wp = 63, .K = 512, .nc = 40 },
        .{ .L = 40, .wp = 0, .K = 48, .nc = 700, .holes = false },
        .{ .L = 17, .wp = 5, .K = 512, .nc = 9 },
        .{ .L = 40, .wp = 3, .K = 24, .nc = 30, .H = 16, .hd = 64, .window = 8 },
    };
    for (cases, 0..) |c, ci_| {
        const d = try attnCaseData(&sc, c, 0xa77 + ci_);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.hd)));
        const truth = try sc_mod.hostF32(try attnCore(&cs, d.sink, d.q, d.keys, d.band, d.comp, d.topk, scale, .float32));
        const core = try sc_mod.hostF32(try attnCore(&sc, d.sink, d.q, d.keys, d.band, d.comp, d.topk, scale, .float32));
        const fused = try sc_mod.hostF32(try kernels.attnFused(&sc, d.sink, d.q, d.keys, c.wp, c.window, d.comp, d.topk, scale, .float32));
        const err_core = maxRelErr(core, truth);
        const err_fused = maxRelErr(fused, truth);
        if (@import("transformer.zig").diagEnvOn("SUSHI_DSV41_ATTN_UBENCH")) std.debug.print("case {d}: core {e}, fused {e}\n", .{ ci_, err_core, err_fused });
        try testing.expect(err_fused <= err_core);
        try testing.expect(err_fused < 3e-5);
    }
}

const AttnArm = enum { core, wide, fused };

fn runAttnArm(sc: *Scope, arm: AttnArm, d: AttnData, c: AttnCase, scale: f32) !A {
    return switch (arm) {
        .core => attnCore(sc, d.sink, d.q, d.keys, d.band, d.comp, d.topk, scale, .bfloat16),
        .wide => kernels.attnWide(sc, d.sink, d.q, d.keys, c.wp, c.window, d.comp, d.topk, scale, .bfloat16),
        .fused => kernels.attnFused(sc, d.sink, d.q, d.keys, c.wp, c.window, d.comp, d.topk, scale, .bfloat16),
    };
}

// One attention span at DeepSeek-V4.1's geometry (64 heads x 512, window 128, 512 picks): the f32 composed core, the
// f16 wide arm and the fused kernel, interleaved in one process, best of 8 each, with the span's peak bytes beyond
// its inputs. Prints only under SUSHI_DSV41_ATTN_UBENCH.
const StreamCase = struct { L: usize, nb: usize, r: usize, tile: usize, distinct: usize, masked: bool = false };

/// Index keys at the release's width with only `distinct` different rows, so scores tie across tiles.
fn streamKeys(sc: *Scope, nb: usize, distinct: usize, seed: u64) !Rows {
    var rows = Rows{ .width = 128, .dtype = .bfloat16, .pack = .{ .mode = .fp4_ue8m0, .group = 32 } };
    errdefer rows.deinit();
    const base = try sc.astype(try sc_mod_randn(sc, &.{ ci(distinct), 128 }, seed), .bfloat16);
    var done: usize = 0;
    while (done < nb) {
        const n = @min(nb - done, distinct);
        try rows.append(sc, try sc.slice(base, 0, 0, n));
        done += n;
    }
    return rows;
}

fn sc_mod_randn(sc: *Scope, shape: []const c_int, seed: u64) !A {
    var rk = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_key(&rk, seed));
    _ = try sc.keep(rk);
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .float32, 0, 1.5, rk, sc.s), &o);
}

fn expectSameInts(want: A, got: A) !void {
    const w = try sc_mod.hostI32(want);
    const g = try sc_mod.hostI32(got);
    try testing.expectEqualSlices(i32, w, g);
}

test "dsv41 index stream: the streamed top-k is the whole scores' top-k, order and ties included, and its block maxima the candidate mask's (release shapes)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const gpu = mlx.gpuStream();
    if (!mlx.streamIsGpu(gpu)) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, gpu);
    defer sc.deinit();
    var fake: Model = undefined;
    fake.fused = true;
    fake.trace = null;
    const IH = 32;
    const IHD = 128;
    const cases = [_]StreamCase{
        .{ .L = 256, .nb = 50000, .r = 2, .tile = 8192, .distinct = 700, .masked = true },
        .{ .L = 128, .nb = 40000, .r = 1, .tile = 4096, .distinct = 997, .masked = true },
        .{ .L = 40, .nb = 300, .r = 1, .tile = 64, .distinct = 37 },
        .{ .L = 17, .nb = 8200, .r = 2, .tile = 4096, .distinct = 8200 },
    };
    for (cases, 0..) |c, ci_| {
        var rows = try streamKeys(&sc, c.nb, c.distinct, 0x1d0 + ci_);
        defer rows.deinit();
        const p0 = c.nb * c.r - c.L;
        const q = try sc.astype(try sc_mod_randn(&sc, &.{ ci(c.L), IH, IHD }, 0x2d0 + ci_), .float32);
        const w = try sc.astype(try sc_mod_randn(&sc, &.{ ci(c.L), IH }, 0x3d0 + ci_), .bfloat16);
        const wscale: f32 = 1.0 / @sqrt(@as(f32, IHD)) / @sqrt(@as(f32, IH));
        const k = @min(512, c.nb);
        var lens_h: [256]i32 = undefined;
        for (lens_h[0..c.L], 0..) |*v, i| v.* = @intCast((p0 + i + 1) / c.r);
        const lens = try sc.ints(lens_h[0..c.L], &.{ ci(c.L), 1 });
        const keys = try rows.rows(&sc, c.nb);
        const scores = try kernels.indexScoresTile(&sc, try sc.astype(q, .bfloat16), keys, w, wscale, p0, c.r, 0);

        const got = try indexStream(&sc, &rows, c.nb, q, w, wscale, p0, c.r, k, c.tile, none, 8, if (c.masked) 2048 else 0, lens);
        try expectSameInts(try kernels.topkIndices(&sc, scores, k), got.idx);
        if (!c.masked) continue;

        const want_mask = try candidateBlocks(&sc, &fake, 0, scores, lens, 8, 2048);
        try testing.expect(want_mask.ctx != null);
        try expectSameInts(try sc.astype(want_mask, .int32), try sc.astype(got.cand, .int32));

        var o = mlx.mlx_array_new();
        const rep = try sc.res(mlx.mlx_repeat_axis(&o, want_mask, 8, -1, sc.s), &o);
        const held = try sc.where(try sc.slice(rep, -1, 0, c.nb), scores, try sc.f(-std.math.inf(f32)));
        const masked = try indexStream(&sc, &rows, c.nb, q, w, wscale, p0, c.r, k, c.tile, want_mask, 8, 0, lens);
        try expectSameInts(try kernels.topkIndices(&sc, held, k), masked.idx);
    }
}

/// MLX peak bytes beyond the active ones while `f` builds and evaluates its results.
fn peakOf(sc: *Scope, comptime f: anytype, args: anytype) !usize {
    _ = mlx.mlx_synchronize(sc.s);
    _ = mlx.mlx_clear_cache();
    _ = mlx.mlx_reset_peak_memory();
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    var run = Scope.init(testing.allocator, sc.s);
    defer run.deinit();
    const out = try @call(.auto, f, .{&run} ++ args);
    try evalAll(&out);
    var pk: usize = 0;
    _ = mlx.mlx_get_peak_memory(&pk);
    return pk -| active;
}

const WholeArgs = struct { rows: *const Rows, nb: usize, q: A, w: A, wscale: f32, p0: usize, r: usize, k: usize, bs: usize, with_blocks: bool, lens: A };

/// `indexer`'s non-streamed arm: the whole row's scores, the candidate mask when asked, the top-k.
fn indexWholeArm(sc: *Scope, a: WholeArgs) ![2]A {
    var fake: Model = undefined;
    fake.fused = true;
    fake.trace = null;
    const keys = try a.rows.rows(sc, a.nb);
    const scores = try indexScores(sc, &fake, a.q, keys, a.w, a.wscale, a.p0, a.r);
    const idx = try topkRows(sc, &fake, scores, a.k);
    return .{ idx, if (a.with_blocks) try candidateBlocks(sc, &fake, 0, scores, a.lens, a.bs, 2048) else idx };
}

/// `indexer`'s decode-width arm: packed keys scored whole, a later layer's candidate mask applied, the top-k.
fn indexNarrowArm(sc: *Scope, a: WholeArgs, mask: A) ![2]A {
    var fake: Model = undefined;
    fake.fused = true;
    fake.trace = null;
    const pk = a.rows.pack.?;
    var scores = try kernels.indexScoresPacked(sc, a.q, try sc.slice(a.rows.buf, 0, 0, a.nb), try sc.slice(a.rows.sbuf, 0, 0, a.nb), a.w, a.wscale, a.p0, a.r, pk.mode, pk.group);
    if (mask.ctx != null) {
        var o = mlx.mlx_array_new();
        const rep = try sc.res(mlx.mlx_repeat_axis(&o, mask, ci(a.bs), -1, sc.s), &o);
        scores = try sc.where(try sc.slice(rep, -1, 0, a.nb), scores, try sc.f(-std.math.inf(f32)));
    }
    const idx = try topkRows(sc, &fake, scores, a.k);
    return .{ idx, idx };
}

fn indexStreamArm(sc: *Scope, a: WholeArgs, tile: usize) ![2]A {
    const got = try indexStream(sc, a.rows, a.nb, a.q, a.w, a.wscale, a.p0, a.r, a.k, tile, none, a.bs, if (a.with_blocks) 2048 else 0, a.lens);
    return .{ got.idx, if (a.with_blocks) got.cand else got.idx };
}

// The indexer's context-scaled working set at DeepSeek-V4.1's release shapes (32 heads x 128, top-512, blocks of 8): the
// MLX peak beyond the inputs of one wide span, whole scores against the streamed top-k, per context. Prints only under
// SUSHI_DSV41_INDEX_UBENCH.
test "dsv41 index stream peak bytes by context (SUSHI_DSV41_INDEX_UBENCH=1)" {
    if (!@import("transformer.zig").diagEnvOn("SUSHI_DSV41_INDEX_UBENCH")) return error.SkipZigTest;
    const gpu = mlx.gpuStream();
    if (!mlx.streamIsGpu(gpu)) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, gpu);
    defer sc.deinit();
    const tile: usize = @intCast(cfg_mod.index_stream_tile);
    // Encoder owners (ratio 2, 256-row span) and decoder owners (ratio 1, the 128-row CED tail).
    for ([_]struct { L: usize, r: usize }{ .{ .L = 256, .r = 2 }, .{ .L = 128, .r = 1 } }) |g| {
        for ([_]usize{ 16384, 65536, 262144, 524288, 1048576 }) |ctx| {
            const nb = ctx / g.r;
            var rows = try streamKeys(&sc, nb, 4096, 0x4d0);
            defer rows.deinit();
            const q = try sc.astype(try sc_mod_randn(&sc, &.{ ci(g.L), 32, 128 }, 0x5d0), .float32);
            const w = try sc.astype(try sc_mod_randn(&sc, &.{ ci(g.L), 32 }, 0x6d0), .bfloat16);
            const lens_h = try testing.allocator.alloc(i32, g.L);
            defer testing.allocator.free(lens_h);
            for (lens_h, 0..) |*v, i| v.* = @intCast((ctx - g.L + i + 1) / g.r);
            const lens = try sc.ints(lens_h, &.{ ci(g.L), 1 });
            try evalAll(&.{ q, w, rows.buf, rows.sbuf, lens });
            const a = WholeArgs{ .rows = &rows, .nb = nb, .q = q, .w = w, .wscale = 0.01, .p0 = ctx - g.L, .r = g.r, .k = 512, .bs = 8, .with_blocks = nb / 8 > 2048, .lens = lens };
            const whole = try peakOf(&sc, indexWholeArm, .{a});
            const streamed = try peakOf(&sc, indexStreamArm, .{ a, tile });
            std.debug.print("dsv41-index L {d} ratio {d} ctx {d}: whole {d} MB, streamed {d} MB\n", .{ g.L, g.r, ctx, whole >> 20, streamed >> 20 });
        }
    }
    // A decode or verify block: 16 rows over every key, with and without the candidate mask a later layer applies.
    for ([_]usize{ 65536, 262144, 524288, 1048576 }) |ctx| {
        const L = 16;
        var rows = try streamKeys(&sc, ctx, 4096, 0x4d0);
        defer rows.deinit();
        const q = try sc.astype(try sc_mod_randn(&sc, &.{ L, 32, 128 }, 0x5d0), .float32);
        const w = try sc.astype(try sc_mod_randn(&sc, &.{ L, 32 }, 0x6d0), .bfloat16);
        var lens_h: [L]i32 = undefined;
        for (&lens_h, 0..) |*v, i| v.* = @intCast(ctx - L + i + 1);
        const lens = try sc.ints(&lens_h, &.{ L, 1 });
        var fake: Model = undefined;
        fake.fused = true;
        fake.trace = null;
        const full = try kernels.indexScoresPacked(&sc, q, try sc.slice(rows.buf, 0, 0, ctx), try sc.slice(rows.sbuf, 0, 0, ctx), w, 0.01, ctx - L, 1, rows.pack.?.mode, rows.pack.?.group);
        const mask = try candidateBlocks(&sc, &fake, 0, full, lens, 8, 2048);
        try evalAll(&.{ q, w, rows.buf, rows.sbuf, lens, mask });
        const a = WholeArgs{ .rows = &rows, .nb = ctx, .q = q, .w = w, .wscale = 0.01, .p0 = ctx - L, .r = 1, .k = 512, .bs = 8, .with_blocks = false, .lens = lens };
        const plain = try peakOf(&sc, indexNarrowArm, .{ a, none });
        const masked = try peakOf(&sc, indexNarrowArm, .{ a, mask });
        std.debug.print("dsv41-index narrow L {d} ctx {d}: packed keys {d} MB, with the mask {d} MB\n", .{ L, ctx, plain >> 20, masked >> 20 });
    }
}

fn growArm(sc: *Scope, rows: *Rows, chunk: A) ![2]A {
    try rows.append(sc, chunk);
    return .{ rows.buf, rows.sbuf };
}

// What a row store's growth holds beyond its rows: the peak while a full buffer grows by its fraction. Prints only under
// SUSHI_DSV41_INDEX_UBENCH.
test "dsv41 rows growth peak bytes (SUSHI_DSV41_INDEX_UBENCH=1)" {
    if (!@import("transformer.zig").diagEnvOn("SUSHI_DSV41_INDEX_UBENCH")) return error.SkipZigTest;
    const gpu = mlx.gpuStream();
    if (!mlx.streamIsGpu(gpu)) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, gpu);
    defer sc.deinit();
    for ([_]struct { width: usize, mode: qat_mod.Qat, group: usize }{ .{ .width = 512, .mode = .fp4_e4m3, .group = 16 }, .{ .width = 128, .mode = .fp4_ue8m0, .group = 32 } }) |g| {
        for ([_]usize{ 262144, 1048576 }) |n| {
            var rows = Rows{ .width = g.width, .dtype = .bfloat16, .pack = .{ .mode = g.mode, .group = g.group } };
            defer rows.deinit();
            const base = try sc.astype(try sc_mod_randn(&sc, &.{ 4096, ci(g.width) }, 0x7d0), .bfloat16);
            const full = try sc.broadcast(try sc.expand(base, 0), &.{ ci(n / 4096), 4096, ci(g.width) });
            try rows.append(&sc, try sc.reshape(full, &.{ ci(n), ci(g.width) }));
            try evalAll(&.{ rows.buf, rows.sbuf });
            const chunk = try sc.slice(base, 0, 0, 2048);
            try evalAll(&.{chunk});
            const pk = try peakOf(&sc, growArm, .{ &rows, chunk });
            const bytes = n * (g.width / 2 + g.width / g.group);
            std.debug.print("dsv41-rows width {d} used {d}: {d} MB held, growth peak beyond that {d} MB\n", .{ g.width, n, bytes >> 20, pk >> 20 });
        }
    }
}

test "dsv41 attention span timing and peak bytes (SUSHI_DSV41_ATTN_UBENCH=1)" {
    if (!@import("transformer.zig").diagEnvOn("SUSHI_DSV41_ATTN_UBENCH")) return error.SkipZigTest;
    const gpu = mlx.gpuStream();
    if (!mlx.streamIsGpu(gpu)) return error.SkipZigTest;
    var base = Scope.init(testing.allocator, gpu);
    defer base.deinit();
    const io = std.Io.Threaded.global_single_threaded.io();
    const scale = 1.0 / @sqrt(@as(f32, 512));
    for ([_]AttnCase{ .{ .L = 256, .wp = 127, .K = 512, .nc = 4096, .holes = false }, .{ .L = 128, .wp = 127, .K = 512, .nc = 4096, .holes = false }, .{ .L = 256, .wp = 127, .K = 0, .nc = 0 } }) |c| {
        const d = try attnCaseData(&base, c, 0xa77);
        try evalAll(&.{ d.q, d.keys, d.band, d.sink });
        if (c.K > 0) try evalAll(&.{ d.comp, d.topk });
        var best: [3]f64 = @splat(std.math.inf(f64));
        var peak: [3]usize = @splat(0);
        for (0..17) |rep| for (std.enums.values(AttnArm)) |arm| {
            var sc = Scope.init(testing.allocator, gpu);
            defer sc.deinit();
            _ = mlx.mlx_synchronize(gpu);
            _ = mlx.mlx_clear_cache();
            _ = mlx.mlx_reset_peak_memory();
            var active: usize = 0;
            _ = mlx.mlx_get_active_memory(&active);
            const t0 = std.Io.Timestamp.now(io, .boot);
            const y = try runAttnArm(&sc, arm, d, c, scale);
            try mlx.check(mlx.mlx_array_eval(y));
            const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .boot)).nanoseconds)) / 1e6;
            var pk: usize = 0;
            _ = mlx.mlx_get_peak_memory(&pk);
            if (rep > 0) best[@intFromEnum(arm)] = @min(best[@intFromEnum(arm)], ms);
            peak[@intFromEnum(arm)] = pk -| active;
        };
        std.debug.print("dsv41-attn span {d} K {d}: core {d:.2} ms {d} MB, wide f16 {d:.2} ms {d} MB, fused {d:.2} ms {d} MB\n", .{
            c.L,                              c.K,
            best[@intFromEnum(AttnArm.core)], peak[@intFromEnum(AttnArm.core)] >> 20,
            best[@intFromEnum(AttnArm.wide)], peak[@intFromEnum(AttnArm.wide)] >> 20,
            best[@intFromEnum(AttnArm.fused)], peak[@intFromEnum(AttnArm.fused)] >> 20,
        });
    }
}

/// The ids `--drop-expert` leaves out of layer `li` in the router tests: three per layer, one from each of the tiny
/// fixture's rate groups (ids 0..2, 3..5, 6..7), so no group empties.
pub fn testDropIds(li: usize) [3]u16 {
    return .{ @intCast(li % 3), @intCast(3 + (li + 1) % 3), @intCast(6 + li % 2) };
}

pub fn testDrop(layers: usize) !expert_usage.Drop {
    var ids: [64][3]u16 = undefined;
    var rows: [64][]const u16 = undefined;
    for (0..layers) |li| {
        ids[li] = testDropIds(li);
        rows[li] = &ids[li];
    }
    return expert_usage.fixedDrop(testing.allocator, 0, rows[0..layers]);
}

/// Every row of `h` through layer `li`'s router on `cut` (experts dropped) against `full`'s normal path: the picks are
/// the top-k of the kept experts, none dropped, and the weights are the normal path's on those picks.
fn expectMaskedRoute(sc: *Scope, full: *Model, cut: *Model, d: *const expert_usage.Drop, li: usize, h: A, exact: bool) !void {
    const c = &full.cfg;
    const k = c.top_k;
    const rows: usize = @intCast(dim(h, 0));
    const r = try router(sc, cut, li, h, none);
    const idx = try sc_mod.hostI32(try sc.astype(r.idx, .int32));
    const wts = try sc_mod.hostF32(r.wts);

    const logits = try sc.matmul(try sc.astype(h, .float32), try sc.t2(full.layers[li].gate_w));
    const scores = try sc.op1(mlx.mlx_sqrt, try sc.op2(mlx.mlx_logaddexp, logits, try sc.f(0)));
    const sel = try sc_mod.hostF32(try sc.add(scores, full.layers[li].gate_bias));
    for (0..rows) |row| {
        var want: [8]bool = @splat(false);
        const best = sel[row * c.experts ..][0..8];
        for (0..k) |_| {
            var at: ?usize = null;
            for (0..c.experts) |e| {
                if (d.isDropped(li, e) or want[e]) continue;
                if (at == null or best[e] > best[at.?]) at = e;
            }
            want[at.?] = true;
        }
        var got: [8]bool = @splat(false);
        for (idx[row * k ..][0..k]) |e| {
            try testing.expect(!d.isDropped(li, @intCast(e)));
            got[@intCast(e)] = true;
        }
        try testing.expectEqualSlices(bool, &want, &got);
    }
    // The normal path's weights for these picks: sqrt-softplus score, normalized over the picks, times the scale.
    var w = try sc.takeAlong(scores, r.idx, -1);
    if (c.norm_topk and k > 1) w = try sc.div(w, try sc.add(try sc.sum(w, -1, true), try sc.f(1e-20)));
    const ref = try sc_mod.hostF32(try sc.mul(w, try sc.f(c.route_scale)));
    for (ref, wts) |a, b| {
        if (exact) try testing.expectEqual(a, b) else try testing.expectApproxEqRel(a, b, 1e-5);
    }
}

fn routerRows(sc: *Scope, dim_: usize, rows: usize, seed: u64) !A {
    const data = try testing.allocator.alloc(f32, rows * dim_);
    defer testing.allocator.free(data);
    var prng = std.Random.DefaultPrng.init(seed);
    for (data) |*v| v.* = prng.random().floatNorm(f32);
    return sc.floats(data, &.{ ci(rows), ci(dim_) });
}

test "dsv41 drop: the router with experts masked picks the top-k of the kept experts and weights them as the normal path, bit for bit on the CPU stream (DSV41_TINY)" {
    const dir = try fixtureDir();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const full = try tinyModel(dir, .{ .act = .float32, .experts = false }, s);
    defer full.deinit();
    var d = try testDrop(full.cfg.layers);
    defer d.deinit();
    const cut = try tinyModel(dir, .{ .act = .float32, .experts = false, .drop = &d }, s);
    defer cut.deinit();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const h = try routerRows(&sc, full.cfg.dim, 24, 0xd20b);
    for (0..full.cfg.layers) |li| try expectMaskedRoute(&sc, full, cut, &d, li, h, true);
}

test "dsv41 drop: the fused router kernel masks the same experts and weights the picks within the kernel's rounding (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try fixtureDir();
    const s = mlx.gpuStream();
    const full = try tinyModel(dir, .{ .act = .float32, .experts = false }, s);
    defer full.deinit();
    var d = try testDrop(full.cfg.layers);
    defer d.deinit();
    const cut = try tinyModel(dir, .{ .act = .float32, .experts = false, .drop = &d }, s);
    defer cut.deinit();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    // One row is the GEMV width, more is the matmul one.
    for ([_]usize{ 1, 24 }) |rows| {
        const h = try routerRows(&sc, full.cfg.dim, rows, 0xd20c + rows);
        for (0..full.cfg.layers) |li| try expectMaskedRoute(&sc, full, cut, &d, li, h, false);
    }
}

test "dsv41 drop: the backbone routers mask their dropped experts and a DSpark stage's router masks none (DSV41_TINY)" {
    const dir = try fixtureDir();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var d = try testDrop(8);
    defer d.deinit();
    const m = try tinyModel(dir, .{ .act = .float32, .experts = false, .dspark = true, .drop = &d }, s);
    defer m.deinit();
    try testing.expectEqual(@as(usize, 8), m.cfg.layers);
    for (m.layers, 0..) |*ly, li| {
        const bias = try sc_mod.hostF32(ly.gate_bias);
        for (bias, 0..) |v, e| {
            const masked = li < m.cfg.layers and d.isDropped(li, e);
            try testing.expectEqual(masked, v == -std.math.inf(f32));
            try testing.expect(masked or std.math.isFinite(v));
        }
    }
    try testing.expect(m.layers.len > m.cfg.layers);
}

// ── image input ──

/// A tiny model on the CPU stream whose layers all carry an image routing bias far from the text one, and the
/// token id its image spans use.
fn visionTinyModel(dir: []const u8, s: S) !struct { m: *Model, image_id: u32 } {
    const m = try tinyModel(dir, .{ .act = .float32, .qat = false }, s);
    errdefer m.deinit();
    const image_id: u32 = m.cfg.vocab - 1;
    m.cfg.image_token_id = image_id;
    for (m.layers[0..m.cfg.layers]) |*ly| {
        const bias = try sc_mod.hostF32(ly.gate_bias);
        const shifted = try testing.allocator.alloc(f32, bias.len);
        defer testing.allocator.free(shifted);
        for (bias, shifted, 0..) |b, *o, e| o.* = b + (if (e % 2 == 0) @as(f32, 6) else -6);
        ly.gate_bias_vl = try m.own(mlx.mlx_array_new_data(shifted.ptr, &[_]c_int{ 1, ci(bias.len) }, 2, .float32));
    }
    return .{ .m = m, .image_id = image_id };
}

test "dsv41 vision: the span rows replace the embeddings, and the logits are the same however the prompt is chunked (DSV41_TINY)" {
    const dir = try fixtureDir();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = Scope.init(testing.allocator, cpu);
    defer sc.deinit();
    const base = try tokens(&sc, &fx);
    defer testing.allocator.free(base);
    exact_matmul = true;
    defer exact_matmul = false;
    const t = try visionTinyModel(dir, cpu);
    defer t.m.deinit();
    const m = t.m;
    const ids = try testing.allocator.dupe(u32, base[0..TINY_S]);
    defer testing.allocator.free(ids);
    for (ids[5..15]) |*id| id.* = t.image_id; // ten image rows, split by every chunk cut below
    const rows = try routerRows(&sc, m.cfg.dim, 10, 0x1a6e);

    const whole = try extendVision(m, ids, .last, 64, .{ .rows = rows, .offset = 0 });
    defer _ = mlx.mlx_array_free(whole);
    const want = try hostRows(&sc, whole);
    defer testing.allocator.free(want);

    resetState(m);
    var got: []f32 = &.{};
    defer testing.allocator.free(got);
    var at: usize = 0;
    var consumed: usize = 0;
    for ([_]usize{ 8, 16, 40 }) |end| {
        const chunk = ids[at..end];
        const logits = try extendVision(m, chunk, .last, 5, .{ .rows = rows, .offset = consumed });
        defer _ = mlx.mlx_array_free(logits);
        consumed += imageRows(m, chunk);
        testing.allocator.free(got);
        got = try hostRows(&sc, logits);
        at = end;
    }
    try testing.expectEqual(@as(usize, 10), consumed);
    try compare("image prompt in chunks", got, want, 2e-3);

    // The rows are what the trunk reads: other rows, other logits.
    resetState(m);
    const other = try extendVision(m, ids, .last, 64, .{ .rows = try routerRows(&sc, m.cfg.dim, 10, 0x2b7f), .offset = 0 });
    defer _ = mlx.mlx_array_free(other);
    const moved = try hostRows(&sc, other);
    defer testing.allocator.free(moved);
    var worst: f32 = 0;
    for (moved, want) |a, b| worst = @max(worst, @abs(a - b));
    try testing.expect(worst > 1e-3);
}

test "dsv41 vision: a span without rows, or a pack without the image bias, is refused by name (DSV41_TINY)" {
    const dir = try fixtureDir();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var sc = Scope.init(testing.allocator, cpu);
    defer sc.deinit();
    const t = try visionTinyModel(dir, cpu);
    defer t.m.deinit();
    const ids = [_]u32{ 3, t.image_id, t.image_id, 4 };
    try testing.expectError(error.Dsv41VisionRowsMissing, extendVision(t.m, &ids, .last, 64, null));
    resetState(t.m);
    const bias = t.m.layers[2].gate_bias_vl;
    t.m.layers[2].gate_bias_vl = none;
    const rows = try routerRows(&sc, t.m.cfg.dim, 2, 7);
    try testing.expectError(error.Dsv41VisionBiasMissing, extendVision(t.m, &ids, .last, 64, .{ .rows = rows, .offset = 0 }));
    t.m.layers[2].gate_bias_vl = bias;
    resetState(t.m);
    try testing.expectError(error.Dsv41VisionRowsShort, extendVision(t.m, &ids, .last, 64, .{ .rows = rows, .offset = 1 }));
}

test "dsv41 vision: an image row picks its experts by the image bias, a text row by the text one (DSV41_TINY)" {
    const dir = try fixtureDir();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var sc = Scope.init(testing.allocator, cpu);
    defer sc.deinit();
    const t = try visionTinyModel(dir, cpu);
    defer t.m.deinit();
    const m = t.m;
    const n = 6;
    const h = try routerRows(&sc, m.cfg.dim, n, 0x51de);
    const flags = [n]bool{ true, false, true, false, false, true };
    const img = try sc.keep(mlx.mlx_array_new_data(&flags, &[_]c_int{n}, 1, .bool_));
    var moved = false;
    for (0..m.cfg.layers) |li| {
        const ly = &m.layers[li];
        const k = ly.top_k;
        const text = try router(&sc, m, li, h, none);
        const mixed = try router(&sc, m, li, h, img);
        const saved = ly.gate_bias;
        ly.gate_bias = ly.gate_bias_vl;
        const vl = try router(&sc, m, li, h, none);
        ly.gate_bias = saved;
        const mixed_idx = try sc_mod.hostI32(try sc.astype(mixed.idx, .int32));
        const text_idx = try sc_mod.hostI32(try sc.astype(text.idx, .int32));
        const vl_idx = try sc_mod.hostI32(try sc.astype(vl.idx, .int32));
        const mixed_wts = try sc_mod.hostF32(mixed.wts);
        const text_wts = try sc_mod.hostF32(text.wts);
        for (flags, 0..) |is_img, row| {
            const want = if (is_img) vl_idx else text_idx;
            try testing.expectEqualSlices(i32, want[row * k ..][0..k], mixed_idx[row * k ..][0..k]);
            if (!is_img) try testing.expectEqualSlices(f32, text_wts[row * k ..][0..k], mixed_wts[row * k ..][0..k]);
            if (is_img and !std.mem.eql(i32, vl_idx[row * k ..][0..k], text_idx[row * k ..][0..k])) moved = true;
        }
    }
    try testing.expect(moved);
}

test "dsv41 vision: the Engram write leaves an image row's stream untouched (DSV41_TINY)" {
    const dir = try fixtureDir();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var sc = Scope.init(testing.allocator, cpu);
    defer sc.deinit();
    const t = try visionTinyModel(dir, cpu);
    defer t.m.deinit();
    const m = t.m;
    const e = &m.engrams[0];
    const n = 5;
    const cols = m.cfg.engram.cols() * m.cfg.engram.head_dim;
    const x = try sc.reshape(try routerRows(&sc, m.cfg.hc_mult * m.cfg.dim, n, 0xe9), &.{ n, ci(m.cfg.hc_mult), ci(m.cfg.dim) });
    const rows = try routerRows(&sc, cols, n, 0xe10);
    const flags = [n]bool{ false, true, true, false, true };
    const img = try sc.keep(mlx.mlx_array_new_data(&flags, &[_]c_int{n}, 1, .bool_));
    const text = try sc_mod.hostF32(try engramWrite(&sc, m, e, x, rows, none));
    const mixed = try sc_mod.hostF32(try engramWrite(&sc, m, e, x, rows, img));
    const plain = try sc_mod.hostF32(x);
    const per = m.cfg.hc_mult * m.cfg.dim;
    for (flags, 0..) |is_img, row| {
        const want = if (is_img) plain else text;
        try testing.expectEqualSlices(f32, want[row * per ..][0..per], mixed[row * per ..][0..per]);
    }
    try testing.expect(!std.mem.eql(f32, text[0..per], plain[0..per]));
}

test "dsv41 vision: an image token takes no part in an n-gram, in the pass and in a restored history (DSV41_TINY)" {
    const dir = try fixtureDir();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const t = try visionTinyModel(dir, cpu);
    defer t.m.deinit();
    try testing.expectEqual(engram.Hash.DEAD, t.m.compressedId(t.image_id));
    try testing.expect(t.m.compressedId(t.image_id - 1) != engram.Hash.DEAD);
    try testing.expectEqual(@as(usize, 2), imageRows(t.m, &.{ 1, t.image_id, 2, t.image_id }));
}

test "dsv41 vision: the CED replay keeps the image mask of the rows it replays (DSV41_TINY)" {
    const dir = try fixtureDir();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var fx = try loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = Scope.init(testing.allocator, cpu);
    defer sc.deinit();
    const base = try tokens(&sc, &fx);
    defer testing.allocator.free(base);
    const t = try visionTinyModel(dir, cpu);
    defer t.m.deinit();
    const m = t.m;
    const split = m.ced_split orelse return error.TestUnexpectedResult;
    const w: usize = m.cfg.window;
    const ids = try testing.allocator.dupe(u32, base[0..TINY_S]);
    defer testing.allocator.free(ids);
    try testing.expect(ids.len > w);
    // The span straddles the replay's first row: encoder rows and replayed rows both hold image positions.
    const first = ids.len - w - 3;
    for (ids[first .. first + 8]) |*id| id.* = t.image_id;
    const rows = try routerRows(&sc, m.cfg.dim, 8, 0xced1);
    const st = try ensureState(m);
    var pass = try beginPassWith(m, st, ids, st.n, 5, .{ .rows = rows, .offset = 0 });
    defer pass.deinit(testing.allocator);
    try testing.expectEqual(ids.len, dim(pass.img, 0));
    planCed(m, &pass, .last);
    try testing.expectEqual(@as(?usize, w), pass.tail);
    for (0..split + 1) |li| try passLayer(m, &pass, li);
    try testing.expectEqual(w, dim(pass.img, 0));
    const got = try sc_mod.hostI32(try sc.astype(pass.img, .int32));
    for (got, ids[ids.len - w ..]) |g, id| try testing.expectEqual(@as(i32, @intFromBool(id == t.image_id)), g);
    try testing.expect(std.mem.indexOfScalar(i32, got, 1) != null and std.mem.indexOfScalar(i32, got, 0) != null);
}
