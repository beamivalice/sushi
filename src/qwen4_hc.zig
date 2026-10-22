//! Flash-Next hyper-connections and the n-gram PLE table.

const trf = @import("transformer.zig");
const Transformer = trf.Transformer;
const std = @import("std");
const sushi_exl3 = @import("sushi_exl3");
const expert_exl3_kernels = sushi_exl3.kernels;
const mlx = @import("mlx.zig");
const qwen4_forward = @import("qwen4_forward.zig");

const ForwardCtx = trf.ForwardCtx;
const HcWeights = trf.HcWeights;
const ModelConfig = trf.ModelConfig;
const PleWeights = trf.PleWeights;
const SSMCacheEntry = trf.SSMCacheEntry;
const getHcFusedKernel = qwen4_forward.getHcFusedKernel;
const log = trf.log;
const materializedOwnedCopy = trf.materializedOwnedCopy;
const scalarOf = trf.scalarOf;
const ssmFreeQsaState = trf.ssmFreeQsaState;
const standinOnes = trf.standinOnes;
const standinRef = trf.standinRef;
const verifySharedHardware = trf.verifySharedHardware;

pub fn claimPleSpecCapture(entry: *SSMCacheEntry, n: usize, ctx_len: usize, capturing: bool) bool {
    if (capturing and ctx_len + n <= entry.spec_ple_tokens.len) {
        entry.spec_ple_len = @intCast(ctx_len + n);
        return true;
    }
    entry.spec_ple_len = 0;
    return false;
}

pub fn hcSiluCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Transformer = @ptrCast(@alignCast(payload.?));
    const x = Transformer.closureIn(input, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(x);
    const y = self.silu(x) catch return -1;
    return Transformer.closureOut(res, y);
}

pub fn hcMixCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Transformer = @ptrCast(@alignCast(payload.?));
    const up4 = Transformer.closureIn(input, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(up4);
    const n4 = Transformer.closureIn(input, 1) orelse return -1;
    defer _ = mlx.mlx_array_free(n4);
    var mix = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mix);
    if (mlx.mlx_sigmoid(&mix, up4, self.s) != 0) return -1;
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    if (mlx.mlx_multiply(&prod, mix, n4, self.s) != 0) return -1;
    var mixed = mlx.mlx_array_new();
    if (mlx.mlx_mean_axis(&mixed, prod, 2, false, self.s) != 0) return -1;
    return Transformer.closureOut(res, mixed);
}

pub fn hcInjCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Transformer = @ptrCast(@alignCast(payload.?));
    const x = Transformer.closureIn(input, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(x);
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    if (mlx.mlx_sigmoid(&sig, x, self.s) != 0) return -1;
    const two = scalarOf(2.0, mlx.mlx_array_dtype(x), self.s) catch return -1;
    defer _ = mlx.mlx_array_free(two);
    var g = mlx.mlx_array_new();
    if (mlx.mlx_multiply(&g, sig, two, self.s) != 0) return -1;
    return Transformer.closureOut(res, g);
}

pub fn hcWriteCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Transformer = @ptrCast(@alignCast(payload.?));
    const stream4 = Transformer.closureIn(input, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(stream4);
    const out4 = Transformer.closureIn(input, 1) orelse return -1;
    defer _ = mlx.mlx_array_free(out4);
    const inj = Transformer.closureIn(input, 2) orelse return -1;
    defer _ = mlx.mlx_array_free(inj);
    var add4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(add4);
    if (mlx.mlx_multiply(&add4, out4, inj, self.s) != 0) return -1;
    var next = mlx.mlx_array_new();
    if (mlx.mlx_add(&next, stream4, add4, self.s) != 0) return -1;
    return Transformer.closureOut(res, next);
}

pub const HcRead = struct {
    mixed: mlx.mlx_array, // [B, S, hidden]
    inj: mlx.mlx_array, // [B, S, hc, 1] (null-ctx on the mixer)
    pub fn deinit(self: *@This()) void {
        _ = mlx.mlx_array_free(self.mixed);
        if (self.inj.ctx != null) _ = mlx.mlx_array_free(self.inj);
    }
};

/// `hcRead` that first applies a deferred write (`*pending`, consumed):
/// fused when the N kernel takes it, else the chain's hcWrite. `h` is
/// replaced by the written stream either way.
/// Diagnostic (`QWEN4_STANDIN=hc`): stream 0 as the block input, unit
/// gates — the read kernels drop out, the writes stay live.
pub fn hcReadStandin(self: *Transformer, h: mlx.mlx_array, batch: c_int, seq_len: c_int) !HcRead {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const start = [_]c_int{ 0, 0, 0 };
    const stop = [_]c_int{ batch, seq_len, hidden };
    const strides = [_]c_int{ 1, 1, 1 };
    var mixed = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&mixed, h, &start, 3, &stop, 3, &strides, 3, self.s));
    errdefer _ = mlx.mlx_array_free(mixed);
    const inj_shape = [_]c_int{ batch, seq_len, hc, 1 };
    return .{ .mixed = mixed, .inj = try standinOnes(&inj_shape, self.s) };
}

pub fn hcReadPending(self: *Transformer, h: *mlx.mlx_array, w: *const HcWeights, batch: c_int, seq_len: c_int, pending: *?HcPending) !HcRead {
    if (pending.*) |*pd| {
        defer pending.* = null;
        defer pd.deinit();
        if (try self.hcReadFusedFor(h.*, w, batch, seq_len, pd.*)) |o| {
            _ = mlx.mlx_array_free(h.*);
            h.* = o.stream;
            return .{ .mixed = o.mixed, .inj = o.inj };
        }
        h.* = try self.hcWrite(h.*, pd.out, pd.inj, batch, seq_len);
    }
    return self.hcRead(h.*, w, batch, seq_len);
}

pub fn hcReadFusedFor(self: *Transformer, stream: mlx.mlx_array, w: *const HcWeights, batch: c_int, seq_len: c_int, pend: ?HcPending) !?HcFusedOut {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const hp = @import("hc_prefill.zig");
    if (hp.eligible(batch, seq_len, self.config.hc_count, self.config.hidden_size, w.inject_flat, mlx.mlx_array_dtype(stream))) {
        if (try hp.norm(self.s, stream, w.norm_w, w.inject_flat, self.rms_eps_arr, batch, seq_len, if (pend) |p| .{ .out = p.out, .inj = p.inj } else null)) |n| {
            defer n.deinit();
            var read = try self.hcReadNormed(n.normalized, w, batch, seq_len, n.raw_inject);
            errdefer read.deinit();
            return .{ .stream = try standinRef(n.stream), .mixed = read.mixed, .inj = read.inj };
        }
    }
    if (batch * seq_len > HC_FUSED_MAX_ROWS or w.down_s.ctx == null or w.up_s.ctx == null or (w.inject_w.ctx != null and w.inject_flat.ctx == null)) return null;
    const dqp = self.quantParamsHinted(w.down_w, w.down_s, @intCast(hc * hidden));
    const uqp = self.quantParamsFor(w.up_w, w.up_s);
    if (dqp.bits != uqp.bits or dqp.group_size != uqp.group_size or dqp.mode != .affine or uqp.mode != .affine) return null;
    if (self.verifyFeatureEnabled(.hc_graph, qwen4_forward.mtp_verify_hc_prepared_active, batch, seq_len) and batch == 1 and hc == 4 and hidden == 2560 and dqp.bits == 8 and dqp.group_size == 64) {
        if (try hcReadPrepared(self.s, stream, w.*, seq_len, self.config.rms_norm_eps, pend)) |result| {
            qwen4_forward.mtp_verify_hc_prepared_calls +%= 1;
            const Once = struct {
                var logged = false;
            };
            if (!Once.logged) {
                Once.logged = true;
                log.info("[mtp-verify] prepared hyper-connection verifier graphs engaged\n", .{});
            }
            return result;
        }
    }
    return hcReadFused(self.s, stream, batch, seq_len, w.norm_w, w.down_w, w.down_s, w.down_b, w.up_w, w.up_s, w.up_b, w.inject_flat, self.config.rms_norm_eps, hc, hidden, dqp.bits, dqp.group_size, pend);
}

pub fn addVerifyPle(self: *Transformer, row: anytype, pw: *const PleWeights, layer: usize, index: usize) !void {
    try self.hcFlush(&row.h, 1, row.seq, &row.pending);
    const add = try self.pleForward(row.ctx, row.h, row.tokens, pw, &row.entries[layer], layer, 1, row.seq);
    defer _ = mlx.mlx_array_free(add);
    var with_ple = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(with_ple);
    if (comptime @import("builtin").is_test) {
        if (qwen4_forward.verify_fault_after_ple_row == index) mlx.fault.arm(1);
    }
    try mlx.check(mlx.mlx_add(&with_ple, row.h, add, self.s));
    _ = mlx.mlx_array_free(row.h);
    row.h = with_ple;
}

pub fn hcReadVerifyRows(self: *Transformer, rows: anytype, w: *const HcWeights) !?[]HcRead {
    if (self.qwen4 == null or self.config.hc_count != 4 or self.config.hidden_size != 2560 or rows.len < 2 or rows.len > 8) return null;
    const width = rows[0].seq;
    const has_pending = rows[0].pending != null;
    var inputs: [8]mlx.mlx_array = undefined;
    var pending: [8]HcPending = undefined;
    for (rows, 0..) |row, i| {
        if (row.seq != width or (row.pending != null) != has_pending) return null;
        inputs[i] = row.h;
        if (row.pending) |p| pending[i] = p;
    }
    const dqp = self.quantParamsHinted(w.down_w, w.down_s, 10240);
    const uqp = self.quantParamsFor(w.up_w, w.up_s);
    if (dqp.bits != 8 or uqp.bits != 8 or dqp.group_size != 64 or uqp.group_size != 64 or dqp.mode != .affine or uqp.mode != .affine) return null;
    const joined = (try hcReadJoined(self.allocator, self.s, inputs[0..rows.len], w.*, width, self.config.rms_norm_eps, if (has_pending) pending[0..rows.len] else null)) orelse return null;
    defer self.allocator.free(joined);
    errdefer for (joined) |value| inline for (.{ "mixed", "inj", "stream" }) |name| {
        if (@field(value, name).ctx != null) _ = mlx.mlx_array_free(@field(value, name));
    };
    const result = try self.allocator.alloc(HcRead, rows.len);
    for (rows, joined, result) |*row, value, *read| {
        read.* = .{ .mixed = value.mixed, .inj = value.inj };
        if (has_pending) {
            _ = mlx.mlx_array_free(row.h);
            row.h = value.stream;
            row.pending.?.deinit();
            row.pending = null;
        }
    }
    trf.mtp_verify_kernel_calls[1] +%= 1;
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[batched] joined hyper-connection verify reads engaged\n", .{});
    }
    return result;
}

/// Defer `stream += out * inj` to the next read when the fused read will
/// take it (decode/verify/batched widths); otherwise write now.
pub fn hcWriteOrDefer(self: *Transformer, h: *mlx.mlx_array, out: mlx.mlx_array, inj: mlx.mlx_array, batch: c_int, seq_len: c_int, inject_flat: mlx.mlx_array, pending: *?HcPending) !void {
    std.debug.assert(pending.* == null);
    const prefill = @import("hc_prefill.zig").eligible(batch, seq_len, self.config.hc_count, self.config.hidden_size, inject_flat, mlx.mlx_array_dtype(h.*)) and mlx.mlx_array_dtype(inj) == .bfloat16;
    if ((prefill or (batch * seq_len <= HC_FUSED_MAX_ROWS)) and mlx.mlx_array_dtype(h.*) == mlx.mlx_array_dtype(out)) {
        var pd: HcPending = undefined;
        pd.out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(pd.out);
        try mlx.check(mlx.mlx_array_set(&pd.out, out));
        pd.inj = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(pd.inj);
        try mlx.check(mlx.mlx_array_set(&pd.inj, inj));
        pending.* = pd;
        return;
    }
    h.* = try self.hcWrite(h.*, out, inj, batch, seq_len);
}

pub fn hcFlush(self: *Transformer, h: *mlx.mlx_array, batch: c_int, seq_len: c_int, pending: *?HcPending) !void {
    if (pending.*) |*pd| {
        defer pending.* = null;
        defer pd.deinit();
        h.* = try self.hcWrite(h.*, pd.out, pd.inj, batch, seq_len);
    }
}

/// Grouped RMS norm over the last `hidden` of each of the `hc` streams:
/// `x [B,S,hc*hidden]` → `[B,S,hc,hidden]`, weight `[hc,hidden]` (already
/// carrying the reference's +1). Caller frees.
pub fn hcGroupNorm(self: *Transformer, x: mlx.mlx_array, w: mlx.mlx_array, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const shape4 = [_]c_int{ batch, seq_len, hc, hidden };
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_reshape(&x4, x, &shape4, 4, self.s));
    var n4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n4);
    try mlx.check(mlx.mlx_fast_rms_norm(&n4, x4, self.ones_hidden.?, self.config.rms_norm_eps, self.s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, n4, w, self.s));
    return out;
}

/// Qwen4ExpTextGatedResidual.forward: the block input is a sigmoid-mixed
/// mean of the normalized streams; the write gates are `2·σ(inject/hc)`.
pub fn hcRead(self: *Transformer, stream: mlx.mlx_array, w: *const HcWeights, batch: c_int, seq_len: c_int) !HcRead {
    if (try self.hcReadFusedFor(stream, w, batch, seq_len, null)) |o| {
        if (o.stream.ctx != null) _ = mlx.mlx_array_free(o.stream);
        return .{ .mixed = o.mixed, .inj = o.inj };
    }
    const n4 = try self.hcGroupNorm(stream, w.norm_w, batch, seq_len);
    defer _ = mlx.mlx_array_free(n4);
    return self.hcReadNormed(n4, w, batch, seq_len, null);
}

pub fn hcReadNormed(self: *Transformer, n4: mlx.mlx_array, w: *const HcWeights, batch: c_int, seq_len: c_int, raw_inject: ?mlx.mlx_array) !HcRead {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const flat_shape = [_]c_int{ batch, seq_len, hc * hidden };
    var n_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n_flat);
    try mlx.check(mlx.mlx_reshape(&n_flat, n4, &flat_shape, 3, self.s));

    // down/inject carry the reference's `/hc` in their weights (load-time fold).
    const down = try self.qmatmul(n_flat, w.down_w, w.down_s, w.down_b);
    defer _ = mlx.mlx_array_free(down);
    const act = (try Transformer.applyClosure(self.compiled_hc_silu, &.{down})) orelse try self.silu(down);
    defer _ = mlx.mlx_array_free(act);
    const up = try self.qmatmul(act, w.up_w, w.up_s, w.up_b);
    defer _ = mlx.mlx_array_free(up);
    const shape4 = [_]c_int{ batch, seq_len, hc, hidden };
    var up4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up4);
    try mlx.check(mlx.mlx_reshape(&up4, up, &shape4, 4, self.s));
    var mixed: mlx.mlx_array = undefined;
    const prefill_mix = if (@import("hc_prefill.zig").eligible(batch, seq_len, self.config.hc_count, self.config.hidden_size, w.inject_flat, mlx.mlx_array_dtype(n4))) try @import("hc_prefill.zig").mix(self.s, up4, n4, batch, seq_len) else null;
    if (prefill_mix) |m| {
        mixed = m;
    } else if (try Transformer.applyClosure(self.compiled_hc_mix, &.{ up4, n4 })) |m| {
        mixed = m;
    } else {
        var mix = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mix);
        try mlx.check(mlx.mlx_sigmoid(&mix, up4, self.s));
        var prod = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(prod);
        try mlx.check(mlx.mlx_multiply(&prod, mix, n4, self.s));
        mixed = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_mean_axis(&mixed, prod, 2, false, self.s));
    }
    errdefer _ = mlx.mlx_array_free(mixed);

    var inj = mlx.mlx_array_new();
    if (w.inject_w.ctx != null) {
        const raw = if (raw_inject) |r| try standinRef(r) else try self.qmatmul(n_flat, w.inject_w, w.inject_s, w.inject_b);
        defer _ = mlx.mlx_array_free(raw);
        const g = (try Transformer.applyClosure(self.compiled_hc_inj, &.{raw})) orelse blk: {
            var sig = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sig);
            try mlx.check(mlx.mlx_sigmoid(&sig, raw, self.s));
            const two = try scalarOf(2.0, mlx.mlx_array_dtype(raw), self.s);
            defer _ = mlx.mlx_array_free(two);
            var g2 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&g2, sig, two, self.s));
            break :blk g2;
        };
        defer _ = mlx.mlx_array_free(g);
        const inj_shape = [_]c_int{ batch, seq_len, hc, 1 };
        try mlx.check(mlx.mlx_reshape(&inj, g, &inj_shape, 4, self.s));
    }
    return .{ .mixed = mixed, .inj = inj };
}

/// `stream + flatten(out[..., None, :] * inj)`. Consumes `stream`.
pub fn hcWrite(self: *Transformer, stream: mlx.mlx_array, out: mlx.mlx_array, inj: mlx.mlx_array, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const out_shape = [_]c_int{ batch, seq_len, 1, hidden };
    var out4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out4);
    try mlx.check(mlx.mlx_reshape(&out4, out, &out_shape, 4, self.s));
    const flat_shape = [_]c_int{ batch, seq_len, hc * hidden };
    const shape4 = [_]c_int{ batch, seq_len, hc, hidden };
    var stream4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream4);
    try mlx.check(mlx.mlx_reshape(&stream4, stream, &shape4, 4, self.s));
    var next4: mlx.mlx_array = undefined;
    if (try Transformer.applyClosure(self.compiled_hc_write, &.{ stream4, out4, inj })) |n| {
        next4 = n;
    } else {
        var add4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(add4);
        try mlx.check(mlx.mlx_multiply(&add4, out4, inj, self.s));
        next4 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&next4, stream4, add4, self.s));
    }
    defer _ = mlx.mlx_array_free(next4);
    var next = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(next);
    try mlx.check(mlx.mlx_reshape(&next, next4, &flat_shape, 3, self.s));
    _ = mlx.mlx_array_free(stream);
    return next;
}

/// Host-side n-gram gather: `[B, S, ple_embed_dim]` bf16 for this chunk's
/// token ids, advancing the token history. Serial: `entry`'s history over
/// `[1, S]`. Batched (`ctx.batch_slots`): `[N, 1]`, each row hashed
/// against ITS slot's history (`slots[i].ssm_entries[layer]`), ONE gather
/// for all N·heads rows.
pub fn pleEmbedding(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array, entry: *SSMCacheEntry, layer: usize, seq_len: c_int) !mlx.mlx_array {
    const st = self.qwen4.?;
    const emb_dim: usize = st.table.dim * st.hash.n_heads;
    const n: usize = mlx.mlx_array_size(token_ids);
    const batch: c_int = @intCast(n / @as(usize, @intCast(seq_len)));
    const shape = [_]c_int{ batch, seq_len, @intCast(emb_dim) };
    // Packed bf16 on the host (RNE) so the upload is one copy — no
    // mid-graph eval, no GPU sync inside the layer loop.
    const pk = try self.allocator.alloc(u16, n * emb_dim);
    defer self.allocator.free(pk);
    const capture = blk: {
        if (ctx.batch_slots) |slots| {
            if (!self.spec_capture_ssm) break :blk false;
            const per: usize = @intCast(seq_len);
            var any = false;
            for (slots) |sc| {
                if (self.pleClaimSpecCapture(&sc.ssm_entries.?[layer], per)) any = true;
            }
            if (any) _ = self.pleClaimSpecCapture(entry, per);
            break :blk any;
        }
        break :blk self.pleClaimSpecCapture(entry, n);
    };
    if (ctx.ple_defer and pleDeferrable(&self.config, n)) {
        if (ctx.ple_pending != null) return error.PlePendingAlreadySet;
        @memset(pk, 0);
        const emb = mlx.mlx_array_new_data(pk.ptr, &shape, 3, .bfloat16);
        errdefer _ = mlx.mlx_array_free(emb);
        var emb_ref = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(emb_ref);
        try mlx.check(mlx.mlx_array_set(&emb_ref, emb));
        var ids_ref = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(ids_ref);
        try mlx.check(mlx.mlx_array_set(&ids_ref, token_ids));
        ctx.ple_pending = .{ .emb = emb_ref, .token_ids = ids_ref, .entry = entry, .layer = layer, .seq_len = seq_len, .capture = capture };
        return emb;
    }
    try self.pleGatherBf16(ctx, token_ids, entry, layer, seq_len, pk, capture);
    return mlx.mlx_array_new_data(pk.ptr, &shape, 3, .bfloat16);
}

/// A build that reads its routing back on the HOST cannot carry a deferred
/// leaf: that read evaluates the stream, and an evaluated node never sees
/// the fill. The EXL3 sorted GEMM builds its window table from the expert
/// ids past `DECODE_ROWS_MAX` rows, so those widths gather eagerly.
pub fn pleDeferrable(cfg: *const ModelConfig, rows: usize) bool {
    if (cfg.expert_layout != .exl3_k4) return true;
    return rows <= expert_exl3_kernels.DECODE_ROWS_MAX;
}

/// Claim (or clear) `entry`'s fixed spec-PLE token slot for a gather of
/// `n` ids, reporting whether the verify history fits it. ONE predicate
/// for the eager and the deferred arm: with `ple_defer` the gather that
/// would set `spec_ple_len` runs AFTER the PLE layer's conv capture, which
/// keys on it — so the length is claimed here, at build time, and the
/// gather only fills the tokens.
pub fn pleClaimSpecCapture(self: *Transformer, entry: *SSMCacheEntry, n: usize) bool {
    const ctx_len: usize = self.qwen4.?.hash.ngram_size - 1;
    return claimPleSpecCapture(entry, n, ctx_len, self.spec_capture_ssm);
}

/// Every eval inside a `ple_defer` build reads the leaf as it stands, and
/// an evaluated node is never recomputed when the leaf's buffer is filled:
/// a capture the MTP head consumes, and anything a cadence or profiler eval
/// freezes on the way to it, must be materialized on a FILLED leaf.
pub fn plePreEval(self: *Transformer, ctx: *ForwardCtx) !void {
    if (ctx.ple_pending == null) return;
    try self.flushDeferredPle(ctx);
}

/// Qwen4ExpTextPLELayer.forward → the `[B,S,hc*hidden]` addend. Batched
/// decode hands in the MERGED conv window as `entry.aux_state`.
pub fn pleForward(self: *Transformer, ctx: *ForwardCtx, stream: mlx.mlx_array, token_ids: mlx.mlx_array, pw: *const PleWeights, entry: *SSMCacheEntry, layer: usize, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const cfg = &self.config;
    const hc: c_int = @intCast(cfg.hc_count);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const emb = try self.pleEmbedding(ctx, token_ids, entry, layer, seq_len);
    defer _ = mlx.mlx_array_free(emb);
    if (Transformer.qwen4_trace) |tr| Transformer.Qwen4Trace.set(&tr.ple_emb, emb);

    const key_raw = try self.qmatmul(emb, pw.key_w, pw.key_s, pw.key_b);
    defer _ = mlx.mlx_array_free(key_raw);
    const key4 = try self.hcGroupNorm(key_raw, pw.norm_key, batch, seq_len);
    defer _ = mlx.mlx_array_free(key4);
    const value = try self.qmatmul(emb, pw.value_w, pw.value_s, pw.value_b);
    defer _ = mlx.mlx_array_free(value);
    const query4 = try self.hcGroupNorm(stream, pw.norm_query, batch, seq_len);
    defer _ = mlx.mlx_array_free(query4);

    var kq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kq);
    try mlx.check(mlx.mlx_multiply(&kq, key4, query4, self.s));
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_sum_axis(&gate, kq, -1, true, self.s)); // [B,S,hc,1]
    // Scalars in the activation dtype: an f32 scalar promotes the gate
    // and, through the value product, the whole residual stream.
    const gate_dt = mlx.mlx_array_dtype(gate);
    const inv_sqrt_h = try scalarOf(1.0 / @sqrt(@as(f32, @floatFromInt(hidden))), gate_dt, self.s);
    defer _ = mlx.mlx_array_free(inv_sqrt_h);
    var gate_sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_sc);
    try mlx.check(mlx.mlx_multiply(&gate_sc, gate, inv_sqrt_h, self.s));
    // sqrt(max(|g|, 1e-6)) · sign(g)
    var g_abs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_abs);
    try mlx.check(mlx.mlx_abs(&g_abs, gate_sc, self.s));
    const floor = try scalarOf(1e-6, gate_dt, self.s);
    defer _ = mlx.mlx_array_free(floor);
    var g_max = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_max);
    try mlx.check(mlx.mlx_maximum(&g_max, g_abs, floor, self.s));
    var g_sqrt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_sqrt);
    try mlx.check(mlx.mlx_sqrt(&g_sqrt, g_max, self.s));
    var g_sign = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_sign);
    try mlx.check(mlx.mlx_sign(&g_sign, gate_sc, self.s));
    var g_signed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_signed);
    try mlx.check(mlx.mlx_multiply(&g_signed, g_sqrt, g_sign, self.s));
    var g_sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_sig);
    try mlx.check(mlx.mlx_sigmoid(&g_sig, g_signed, self.s));

    const v_shape = [_]c_int{ batch, seq_len, 1, hidden };
    var value4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(value4);
    try mlx.check(mlx.mlx_reshape(&value4, value, &v_shape, 4, self.s));
    var gv4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gv4);
    try mlx.check(mlx.mlx_multiply(&gv4, g_sig, value4, self.s)); // [B,S,hc,hidden]
    const flat_shape = [_]c_int{ batch, seq_len, hc * hidden };
    var gv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gv);
    try mlx.check(mlx.mlx_reshape(&gv, gv4, &flat_shape, 3, self.s));
    const gvn4 = try self.hcGroupNorm(gv, pw.norm_conv, batch, seq_len);
    defer _ = mlx.mlx_array_free(gvn4);
    var gvn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gvn);
    try mlx.check(mlx.mlx_reshape(&gvn, gvn4, &flat_shape, 3, self.s));

    // Dilated depthwise causal conv over [state | gvn].
    const dilation: c_int = @intCast(cfg.ngram_size);
    const state_len: c_int = (@as(c_int, @intCast(cfg.ple_conv_kernel)) - 1) * dilation;
    var prev_state = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prev_state);
    if (entry.aux_state.ctx != null) {
        try mlx.check(mlx.mlx_array_set(&prev_state, entry.aux_state));
    } else {
        const zshape = [_]c_int{ batch, state_len, hc * hidden };
        try mlx.check(mlx.mlx_zeros(&prev_state, &zshape, 3, .bfloat16, self.s));
    }
    const parts = [_]mlx.mlx_array{ prev_state, gvn };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 1, self.s));
    var conv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv);
    try mlx.check(mlx.mlx_conv1d(&conv, cat, pw.conv_w, 1, 0, dilation, hc * hidden, self.s));
    const conv_act = try self.silu(conv);
    defer _ = mlx.mlx_array_free(conv_act);
    // New state = the last state_len rows of the concat (materialized —
    // a view outliving `cat` pins the whole buffer).
    {
        const total: c_int = state_len + seq_len;
        const start = [_]c_int{ 0, total - state_len, 0 };
        const stop = [_]c_int{ batch, total, hc * hidden };
        const strides = [_]c_int{ 1, 1, 1 };
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, cat, &start, 3, &stop, 3, &strides, 3, self.s));
        const owned = try materializedOwnedCopy(self.s, view);
        ssmFreeQsaState(entry); // PLE layer: no QSA state to lose
        entry.aux_state = owned;
    }
    if (self.spec_capture_ssm and entry.spec_ple_len > 0) {
        if (entry.spec_ple_input.ctx != null) _ = mlx.mlx_array_free(entry.spec_ple_input);
        entry.spec_ple_input = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&entry.spec_ple_input, cat));
    }
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&out, gv, conv_act, self.s));
    if (Transformer.qwen4_trace) |tr| Transformer.Qwen4Trace.set(&tr.ple_out, out);
    return out;
}

/// `HcRead` for every row: one joined kernel pass when it serves, else each row's own read.
pub fn hcReadRows(self: *Transformer, rows: anytype, w: *const HcWeights, out: []HcRead) !void {
    if (try self.hcReadVerifyRows(rows, w)) |joined| {
        defer self.allocator.free(joined);
        @memcpy(out[0..rows.len], joined);
        return;
    }
    var built: usize = 0;
    errdefer for (out[0..built]) |*read| read.deinit();
    for (rows, out[0..rows.len]) |*row, *read| {
        self.fwd_gen = row.generation;
        read.* = try self.hcReadPending(&row.h, w, 1, row.seq, &row.pending);
        built += 1;
    }
}

// ── qwen4_exp fused hyper-connection READ at decode width ──
// Copyright (c) 2026 David Dalcu. Original kernels (sushi_hc_read_n/d/u),
// written for sushi; MIT licensed like the rest of the project — keep
// this notice when copying.
// hcRead at B*S == 1 is ~11 dispatches over 10240-wide tensors (group RMS
// norm, weight multiply, down qmv, silu, up qmv, sigmoid-mix + mean, inject
// matvec, 2·sigmoid), ×2 per layer. Three kernels: N = stats + normalized
// stream `xn` + inject gates (one threadgroup), D = down matvec + silu (one
// simdgroup per output row, gateup-shaped), U = up matvec + sigmoid-mix (one
// threadgroup per hidden column, one simdgroup per stream). Rounding sites
// mirror the chain (norm → T, ×w → T, matvec → T); only accumulation order
// differs, so the bar is per-element parity. A first cut kept `xn` in 20 KB
// of threadgroup memory inside the down kernel and cost 146 us per call
// in-situ — the device-memory `xn` is load-bearing.
pub const HC_FUSED_N_SOURCE =
    \\uint tid = thread_index_in_threadgroup;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint h = threadgroup_position_in_grid.x;
    \\uint row = threadgroup_position_in_grid.y;
    \\threadgroup float tgs[8];
    \\threadgroup float tgi[8 * HC];
    \\const int base = int(h) * H;
    \\const int PER = H / 256;
    \\// Rows (batch*seq) are independent: every per-row buffer is offset here once.
    \\const device T* x = x_in + (size_t)row * (size_t)(HC * H);
    \\device T* xn = xn_out + (size_t)row * (size_t)(HC * H);
    \\device T* xs = xs_out + (WR ? (size_t)row * (size_t)(HC * H) : 0);
    \\device float* ipart = ipart_out + (size_t)row * (size_t)(HC * HC);
    \\const device T* wo = wo_in + (size_t)row * (size_t)H;
    \\float xv[PER];
    \\if (WR) {
    \\  // Pending hcWrite: stream' = T(stream + T(out * inj)), the chain's two roundings.
    \\  // (`wi_in` can be < 8 elements and land in `constant`: no pointer rebind.)
    \\  float g = float(wi_in[(size_t)row * (size_t)HC + h]);
    \\  for (int i = 0; i < PER; ++i) {
    \\    int k = base + int(tid) + 256 * i;
    \\    T v = T(float(x[k]) + float(T(float(wo[k - base]) * g)));
    \\    xs[k] = v;
    \\    xv[i] = float(v);
    \\  }
    \\} else {
    \\  for (int i = 0; i < PER; ++i) xv[i] = float(x[base + int(tid) + 256 * i]);
    \\}
    \\float a = 0.0f;
    \\for (int i = 0; i < PER; ++i) a += xv[i] * xv[i];
    \\a = simd_sum(a);
    \\if (lane == 0) tgs[sg] = a;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float t = 0.0f;
    \\for (int g = 0; g < 8; ++g) t += tgs[g];
    \\float rsh = rsqrt(t / float(H) + eps[0]);
    \\float ip[HC];
    \\for (int c = 0; c < HC; ++c) ip[c] = 0.0f;
    \\for (int i = 0; i < PER; ++i) {
    \\  int k = base + int(tid) + 256 * i;
    \\  T v = T(float(T(xv[i] * rsh)) * float(nw[k]));
    \\  xn[k] = v;
    \\  if (INJ) { for (int c = 0; c < HC; ++c) ip[c] += float(v) * float(iw[(size_t)k * (size_t)HC + (size_t)c]); }
    \\}
    \\if (INJ) {
    \\  for (int c = 0; c < HC; ++c) { float pa = simd_sum(ip[c]); if (lane == 0) tgi[sg * HC + c] = pa; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (tid < uint(HC)) {
    \\    float tt = 0.0f;
    \\    for (int g = 0; g < 8; ++g) tt += tgi[g * HC + tid];
    \\    ipart[h * HC + tid] = tt;
    \\  }
    \\}
;

// One threadgroup per output row, its 8 simdgroups split K, and per group of
// ROWS input rows sharing each weight word; inject rows (n >= R) just reduce
// N's partials.
pub const HC_FUSED_D_SOURCE =
    \\uint tid = thread_index_in_threadgroup;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint n = threadgroup_position_in_grid.y;
    \\const uint row0 = threadgroup_position_in_grid.z * uint(ROWS);
    \\threadgroup float part[8 * ROWS];
    \\const int K = HC * H;
    \\const int VPW = 32 / BITS;
    \\const int K_by_p = K / VPW;
    \\const int K_by_gs = K / GS;
    \\const int SLICE = K_by_p / 8;
    \\const int ITERS = SLICE / 32;
    \\uint mask = (1u << BITS) - 1u;
    \\if (n < uint(R)) {
    \\  size_t wbase = (size_t)n * (size_t)K_by_p;
    \\  size_t gbase = (size_t)n * (size_t)K_by_gs;
    \\  int p0 = int(sg) * SLICE + int(lane);
    \\  uint32_t pw[ITERS];
    \\  for (int i = 0; i < ITERS; ++i) pw[i] = dw_q[wbase + (size_t)(p0 + 32 * i)];
    \\  float a0[ROWS], a1[ROWS], a2[ROWS], a3[ROWS];
    \\  for (int r = 0; r < ROWS; ++r) { a0[r] = 0.0f; a1[r] = 0.0f; a2[r] = 0.0f; a3[r] = 0.0f; }
    \\  for (int i = 0; i < ITERS; ++i) {
    \\    int k_base = (p0 + 32 * i) * VPW;
    \\    int gi = k_base / GS;
    \\    float sj = float(dw_s[gbase + (size_t)gi]);
    \\    float bj = float(dw_b[gbase + (size_t)gi]);
    \\    for (int ki = 0; ki < VPW; ki += 4) {
    \\      int k = k_base + ki;
    \\      uint32_t q = pw[i] >> (ki * BITS);
    \\      for (int r = 0; r < ROWS; ++r) {
    \\        const device T* xn = xn_in + (size_t)min(row0 + uint(r), uint(NROWS - 1)) * (size_t)K;
    \\        a0[r] += float(xn[k + 0]) * (float((q >> (0 * BITS)) & mask) * sj + bj);
    \\        a1[r] += float(xn[k + 1]) * (float((q >> (1 * BITS)) & mask) * sj + bj);
    \\        a2[r] += float(xn[k + 2]) * (float((q >> (2 * BITS)) & mask) * sj + bj);
    \\        a3[r] += float(xn[k + 3]) * (float((q >> (3 * BITS)) & mask) * sj + bj);
    \\      }
    \\    }
    \\  }
    \\  for (int r = 0; r < ROWS; ++r) {
    \\    float acc = simd_sum((a0[r] + a1[r]) + (a2[r] + a3[r]));
    \\    if (lane == 0) part[r * 8 + int(sg)] = acc;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (tid < uint(ROWS) && row0 + tid < uint(NROWS)) {
    \\    float t = 0.0f;
    \\    for (int g = 0; g < 8; ++g) t += part[tid * 8 + uint(g)];
    \\    T v = T(t);
    \\    T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
    \\    act_out[(size_t)(row0 + tid) * (size_t)R + n] = v * sig;
    \\  }
    \\} else if (tid < uint(ROWS) && row0 + tid < uint(NROWS)) {
    \\  const uint row = row0 + tid;
    \\  const device float* ipart = ipart_in + (size_t)row * (size_t)(HC * HC);
    \\  int c = int(n) - R;
    \\  float t = 0.0f;
    \\  for (int hh = 0; hh < HC; ++hh) t += ipart[hh * HC + c];
    \\  T v = T(t);
    \\  T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
    \\  inj_out[(size_t)row * (size_t)HC + c] = sig * T(2.0f);
    \\}
;

// One simdgroup per hidden column j and per group of ROWS rows sharing each
// weight word, the HC streams unrolled.
pub const HC_FUSED_U_SOURCE =
    \\uint lane = thread_index_in_simdgroup;
    \\uint j = thread_position_in_grid.y;
    \\const uint row0 = thread_position_in_grid.z * uint(ROWS);
    \\const int VPW = 32 / BITS;
    \\const int R_by_p = R / VPW;
    \\const int R_by_gs = R / GS;
    \\const int RIT = (R_by_p + 31) / 32;
    \\uint mask = (1u << BITS) - 1u;
    \\float sum[ROWS];
    \\for (int r = 0; r < ROWS; ++r) sum[r] = 0.0f;
    \\for (int h = 0; h < HC; ++h) {
    \\  size_t row = (size_t)h * (size_t)H + (size_t)j;
    \\  size_t wbase = row * (size_t)R_by_p;
    \\  size_t gbase = row * (size_t)R_by_gs;
    \\  float a0[ROWS], a1[ROWS], a2[ROWS], a3[ROWS];
    \\  for (int r = 0; r < ROWS; ++r) { a0[r] = 0.0f; a1[r] = 0.0f; a2[r] = 0.0f; a3[r] = 0.0f; }
    \\  for (int i = 0; i < RIT; ++i) {
    \\    int pack = int(lane) + 32 * i;
    \\    if (pack < R_by_p) {
    \\      uint32_t pw = uw_q[wbase + (size_t)pack];
    \\      int k_base = pack * VPW;
    \\      int gi = k_base / GS;
    \\      float sj = float(uw_s[gbase + (size_t)gi]);
    \\      float bj = float(uw_b[gbase + (size_t)gi]);
    \\      for (int ki = 0; ki < VPW; ki += 4) {
    \\        int k = k_base + ki;
    \\        uint32_t q = pw >> (ki * BITS);
    \\        for (int r = 0; r < ROWS; ++r) {
    \\          const device T* act = act_in + (size_t)min(row0 + uint(r), uint(NROWS - 1)) * (size_t)R;
    \\          a0[r] += float(act[k + 0]) * (float((q >> (0 * BITS)) & mask) * sj + bj);
    \\          a1[r] += float(act[k + 1]) * (float((q >> (1 * BITS)) & mask) * sj + bj);
    \\          a2[r] += float(act[k + 2]) * (float((q >> (2 * BITS)) & mask) * sj + bj);
    \\          a3[r] += float(act[k + 3]) * (float((q >> (3 * BITS)) & mask) * sj + bj);
    \\        }
    \\      }
    \\    }
    \\  }
    \\  for (int r = 0; r < ROWS; ++r) {
    \\    const device T* xn = xn_in + (size_t)min(row0 + uint(r), uint(NROWS - 1)) * (size_t)(HC * H);
    \\    float acc = simd_sum((a0[r] + a1[r]) + (a2[r] + a3[r]));
    \\    T u = T(acc);
    \\    T sg = T(1.0f / (1.0f + metal::exp(-float(u))));
    \\    sum[r] += float(T(float(sg) * float(xn[row])));
    \\  }
    \\}
    \\for (int r = 0; r < ROWS; ++r) {
    \\  if (lane == 0 && row0 + uint(r) < uint(NROWS)) mixed_out[(size_t)(row0 + uint(r)) * (size_t)H + j] = T(float(T(sum[r])) * float(T(1.0f / float(HC))));
    \\}
;

pub const HcFusedKey = struct { hc: c_int, h: c_int, r: c_int, inj: c_int, wr: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype, rows: c_int };

pub var hc_fused_kernels: [3]?mlx.mlx_fast_metal_kernel = .{ null, null, null };

/// One config set per row count, inject and pending write: MTP alternates widths every round, a
/// forward's first read and any read after a flush have no pending write, and the mixer no inject.
pub const HcFusedSlot = struct { key: HcFusedKey, cfgs: [3]mlx.mlx_fast_metal_kernel_config };

pub var hc_fused_slots: [HC_FUSED_MAX_ROWS + 1][2][2]?HcFusedSlot = @splat(@splat(@splat(null)));

/// Config sets built, for the slot test.
pub var hc_fused_cfg_builds: usize = 0;

/// Rows one D/U dispatch group carries (each lane holds four accumulators per row).
pub const HC_ROW_GROUP: c_int = 8;

pub var hc_fused_eps: ?mlx.mlx_array = null;

pub var hc_fused_eps_val: f32 = 0;

pub var hc_fused_engaged = false;

pub const HcFusedOut = struct { mixed: mlx.mlx_array, inj: mlx.mlx_array, stream: mlx.mlx_array };

/// A deferred `hcWrite`: the next read's N kernel applies `stream + out*inj`
/// itself (one dispatch fewer per block). Handles are retained copies.
pub const HcPending = struct {
    out: mlx.mlx_array, // [B,S,hidden]
    inj: mlx.mlx_array, // [B,S,hc,1]
    pub fn deinit(self: *HcPending) void {
        _ = mlx.mlx_array_free(self.out);
        _ = mlx.mlx_array_free(self.inj);
    }
};

/// Fused decode-width hyper-connection read over `batch*seq` rows (1..
/// HC_FUSED_MAX_ROWS: decode, verify widths, batched slots — rows are
/// independent, the grid carries groups of up to HC_ROW_GROUP rows, and each
/// weight word is read once per group).
/// `x` holds `batch*seq*hc*hidden` elements; `iw` is the row-major dense
/// `[hc*hidden, hc]` inject weight or null-ctx (mixer). With `pend`, the read
/// first applies that pending write and returns the written stream in
/// `.stream` (null-ctx otherwise). Returns `mixed [B,S,hidden]` +
/// `inj [B,S,hc,1]`, or null when the geometry/quant is outside the kernel
/// (caller keeps the chain).
pub const HC_FUSED_MAX_ROWS: c_int = 16;

pub const HcPrepared = struct {
    // Mutable tensors stay explicit inputs so the cached graph can outlive a model.
    const Key = struct { width: c_int, eps: f32, pending: bool };
    key: Key = .{ .width = 0, .eps = 0, .pending = false },
    stream: mlx.mlx_stream = .{},
    closure: mlx.mlx_closure = .{},
    traces: usize = 0,
    stamp: u64 = 0,

    pub fn callback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
        const self: *HcPrepared = @ptrCast(@alignCast(payload.?));
        self.traces += 1;
        var arrays: [11]mlx.mlx_array = @splat(.{});
        defer for (arrays) |v| {
            if (v.ctx != null) _ = mlx.mlx_array_free(v);
        };
        const n: usize = if (self.key.pending) 11 else 9;
        for (arrays[0..n], 0..) |*v, i| {
            v.* = mlx.mlx_array_new();
            if (mlx.mlx_vector_array_get(v, input, i) != 0) return -1;
        }
        const pd: ?HcPending = if (self.key.pending) .{ .out = arrays[9], .inj = arrays[10] } else null;
        const result = (hcReadFused(self.stream, arrays[0], 1, self.key.width, arrays[1], arrays[2], arrays[3], arrays[4], arrays[5], arrays[6], arrays[7], arrays[8], self.key.eps, 4, 2560, 8, 64, pd) catch return -1) orelse return -1;
        defer {
            _ = mlx.mlx_array_free(result.mixed);
            _ = mlx.mlx_array_free(result.inj);
            if (result.stream.ctx != null) _ = mlx.mlx_array_free(result.stream);
        }
        const outputs = [_]mlx.mlx_array{ result.mixed, result.inj, result.stream };
        res.* = mlx.mlx_vector_array_new_data(&outputs, if (self.key.pending) 3 else 2);
        return 0;
    }
};

pub var hc_prepared_entries: [16]HcPrepared = @splat(.{});

pub var hc_prepared_clock: u64 = 0;

pub fn hcReadPrepared(s: mlx.mlx_stream, x: mlx.mlx_array, w: HcWeights, width: c_int, eps: f32, pending: ?HcPending) !?HcFusedOut {
    return hcReadPreparedWidth(s, x, w, width, eps, pending, 8);
}

pub fn hcReadPreparedWidth(s: mlx.mlx_stream, x: mlx.mlx_array, w: HcWeights, width: c_int, eps: f32, pending: ?HcPending, max_width: c_int) !?HcFusedOut {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or width < 2 or width > max_width) return null;
    if (x.ctx == null or mlx.mlx_array_dtype(x) != .bfloat16 or !std.mem.eql(c_int, mlx.getShape(x), &.{ 1, width, 10240 })) return null;
    const arrays = [_]mlx.mlx_array{ x, w.norm_w, w.down_w, w.down_s, w.down_b, w.up_w, w.up_s, w.up_b, w.inject_flat, if (pending) |pd| pd.out else x, if (pending) |pd| pd.inj else x };
    const shapes = .{ &[_]c_int{ 4, 2560 }, &[_]c_int{ 320, 2560 }, &[_]c_int{ 320, 160 }, &[_]c_int{ 320, 160 }, &[_]c_int{ 10240, 80 }, &[_]c_int{ 10240, 5 }, &[_]c_int{ 10240, 5 }, &[_]c_int{ 10240, 4 } };
    inline for (shapes, 1..) |shape, i| {
        if (arrays[i].ctx == null or !std.mem.eql(c_int, mlx.getShape(arrays[i]), shape)) return null;
        if (mlx.mlx_array_dtype(arrays[i]) != (if (i == 2 or i == 5) mlx.mlx_dtype.uint32 else .bfloat16)) return null;
    }
    if (pending) |pd| {
        if (pd.out.ctx == null or pd.inj.ctx == null or mlx.mlx_array_dtype(pd.out) != .bfloat16 or mlx.mlx_array_dtype(pd.inj) != .bfloat16 or
            !std.mem.eql(c_int, mlx.getShape(pd.out), &.{ 1, width, 2560 }) or !std.mem.eql(c_int, mlx.getShape(pd.inj), &.{ 1, width, 4, 1 })) return null;
    }
    const key = HcPrepared.Key{ .width = width, .eps = eps, .pending = pending != null };
    var chosen: ?*HcPrepared = null;
    var victim = &hc_prepared_entries[0];
    for (&hc_prepared_entries) |*entry| {
        if (entry.closure.ctx != null and std.meta.eql(entry.key, key) and mlx.mlx_stream_equal(entry.stream, s)) {
            chosen = entry;
            break;
        }
        if (entry.stamp < victim.stamp) victim = entry;
    }
    const entry = chosen orelse blk: {
        if (victim.closure.ctx != null) _ = mlx.mlx_closure_free(victim.closure);
        if (victim.stream.ctx != null) _ = mlx.mlx_stream_free(victim.stream);
        victim.* = .{ .key = key };
        try mlx.check(mlx.mlx_stream_set(&victim.stream, s));
        const raw = mlx.mlx_closure_new_func_payload(&HcPrepared.callback, victim, null);
        defer _ = mlx.mlx_closure_free(raw);
        try mlx.check(mlx.mlx_compile(&victim.closure, raw, false));
        break :blk victim;
    };
    hc_prepared_clock +%= 1;
    entry.stamp = hc_prepared_clock;
    const input = mlx.mlx_vector_array_new_data(&arrays, if (pending != null) 11 else 9);
    defer _ = mlx.mlx_vector_array_free(input);
    var output = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(output);
    try mlx.check(mlx.mlx_closure_apply(&output, entry.closure, input));
    var result: HcFusedOut = .{ .mixed = .{}, .inj = .{}, .stream = .{} };
    errdefer inline for (.{ "mixed", "inj", "stream" }) |name| {
        if (@field(result, name).ctx != null) _ = mlx.mlx_array_free(@field(result, name));
    };
    inline for (.{ "mixed", "inj", "stream" }, 0..) |name, i| {
        if (i < 2 or pending != null) {
            @field(result, name) = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&@field(result, name), output, i));
        }
    }
    return result;
}

pub fn hcReadJoined(a: std.mem.Allocator, s: mlx.mlx_stream, inputs: []const mlx.mlx_array, w: HcWeights, width: c_int, eps: f32, pending: ?[]const HcPending) !?[]HcFusedOut {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or inputs.len < 2 or inputs.len > 8 or width < 1) return null;
    const total = @as(c_int, @intCast(inputs.len)) * width;
    if (total > HC_FUSED_MAX_ROWS or w.inject_flat.ctx == null) return null;
    for (inputs) |input| {
        if (input.ctx == null or mlx.mlx_array_dtype(input) != .bfloat16 or !std.mem.eql(c_int, mlx.getShape(input), &.{ 1, width, 10240 })) return null;
    }
    if (pending) |values| {
        if (values.len != inputs.len) return null;
        for (values) |p| {
            if (p.out.ctx == null or p.inj.ctx == null or mlx.mlx_array_dtype(p.out) != .bfloat16 or mlx.mlx_array_dtype(p.inj) != .bfloat16 or
                !std.mem.eql(c_int, mlx.getShape(p.out), &.{ 1, width, 2560 }) or !std.mem.eql(c_int, mlx.getShape(p.inj), &.{ 1, width, 4, 1 })) return null;
        }
    }
    const Join = struct {
        fn array(st: mlx.mlx_stream, values: []const mlx.mlx_array) !mlx.mlx_array {
            const vec = mlx.mlx_vector_array_new_data(values.ptr, values.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            var out = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, st));
            return out;
        }
    };
    const joined = try Join.array(s, inputs);
    defer _ = mlx.mlx_array_free(joined);
    var joined_pending: ?HcPending = null;
    defer if (joined_pending) |p| {
        _ = mlx.mlx_array_free(p.out);
        _ = mlx.mlx_array_free(p.inj);
    };
    if (pending) |values| {
        var outs: [8]mlx.mlx_array = undefined;
        var injs: [8]mlx.mlx_array = undefined;
        for (values, 0..) |p, i| {
            outs[i] = p.out;
            injs[i] = p.inj;
        }
        const out = try Join.array(s, outs[0..values.len]);
        errdefer _ = mlx.mlx_array_free(out);
        joined_pending = .{ .out = out, .inj = try Join.array(s, injs[0..values.len]) };
    }
    const full = (try hcReadPreparedWidth(s, joined, w, total, eps, joined_pending, HC_FUSED_MAX_ROWS)) orelse return null;
    defer inline for (.{ "mixed", "inj", "stream" }) |name| {
        if (@field(full, name).ctx != null) _ = mlx.mlx_array_free(@field(full, name));
    };
    const result = try a.alloc(HcFusedOut, inputs.len);
    @memset(result, .{ .mixed = .{}, .inj = .{}, .stream = .{} });
    errdefer {
        for (result) |r| inline for (.{ "mixed", "inj", "stream" }) |name| {
            if (@field(r, name).ctx != null) _ = mlx.mlx_array_free(@field(r, name));
        };
        a.free(result);
    }
    for (result, 0..) |*r, i| inline for (.{ "mixed", "inj", "stream" }) |name| {
        const value = @field(full, name);
        if (value.ctx != null) {
            const shape = mlx.getShape(value);
            var start: [4]c_int = @splat(0);
            var end: [4]c_int = @splat(1);
            @memcpy(end[0..shape.len], shape);
            start[1] = @as(c_int, @intCast(i)) * width;
            end[1] = start[1] + width;
            @field(r, name) = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&@field(r, name), value, &start, shape.len, &end, shape.len, &[_]c_int{ 1, 1, 1, 1 }, shape.len, s));
        }
    };
    return result;
}

pub fn hcReadFused(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    batch: c_int,
    seq: c_int,
    nw: mlx.mlx_array,
    dw: mlx.mlx_array,
    ds: mlx.mlx_array,
    db: mlx.mlx_array,
    uw: mlx.mlx_array,
    us: mlx.mlx_array,
    ub: mlx.mlx_array,
    iw: mlx.mlx_array,
    eps: f32,
    hc: c_int,
    hidden: c_int,
    bits: u32,
    group_size: u32,
    pend: ?HcPending,
) !?HcFusedOut {
    if (!mlx.streamIsGpu(s)) return null;
    if (bits != 2 and bits != 4 and bits != 8) return null;
    const rows = batch * seq;
    if (rows < 1 or rows > HC_FUSED_MAX_ROWS) return null;
    const vpw: c_int = @intCast(32 / bits);
    if (@rem(@as(c_int, @intCast(group_size)), vpw) != 0) return null;
    if (ds.ctx == null or db.ctx == null or us.ctx == null or ub.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(nw) != xd) return null;
    const K: c_int = hc * hidden;
    if (hc < 1 or hc > 8 or hidden < 8 or @rem(hidden, 8) != 0 or @rem(hidden, vpw) != 0) return null;
    if (mlx.mlx_array_size(x) != @as(usize, @intCast(rows * K)) or mlx.mlx_array_size(nw) != @as(usize, @intCast(K))) return null;
    const dsh = mlx.getShape(dw);
    const ush = mlx.getShape(uw);
    if (dsh.len != 2 or ush.len != 2) return null;
    const R: c_int = dsh[0];
    if (dsh[1] * vpw != K) return null;
    if (ush[0] != K or ush[1] * vpw != R) return null;
    const gsi: c_int = @intCast(group_size);
    if (@rem(K, gsi) != 0 or @rem(R, gsi) != 0) return null;
    // N: 256 threads per stream; D: 8 simdgroups × 32 lanes split each row's words.
    if (@rem(hidden, 256) != 0 or @rem(dsh[1], 256) != 0 or @rem(R, vpw) != 0) return null;
    const inj: c_int = @intFromBool(iw.ctx != null);
    const wr: c_int = @intFromBool(pend != null);
    if (pend) |pd| {
        if (mlx.mlx_array_dtype(pd.out) != xd or mlx.mlx_array_dtype(pd.inj) != xd) return null;
        if (mlx.mlx_array_size(pd.out) != @as(usize, @intCast(rows * hidden)) or mlx.mlx_array_size(pd.inj) != @as(usize, @intCast(rows * hc))) return null;
    }
    if (inj == 1) {
        if (mlx.mlx_array_dtype(iw) != xd) return null;
        const ish = mlx.getShape(iw);
        if (ish.len != 2 or ish[0] != K or ish[1] != hc) return null;
    }

    const key = HcFusedKey{ .hc = hc, .h = hidden, .r = R, .inj = inj, .wr = wr, .bits = bits, .gs = group_size, .dtype = xd, .rows = rows };
    const slot = &hc_fused_slots[@intCast(rows)][@intCast(inj)][@intCast(wr)];
    if (slot.* == null or !std.meta.eql(slot.*.?.key, key)) {
        if (slot.*) |old| for (old.cfgs) |cfg| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        };
        slot.* = null;
        const groups = @divTrunc(rows + HC_ROW_GROUP - 1, HC_ROW_GROUP);
        const rows_per = @divTrunc(rows + groups - 1, groups);
        const cn = mlx.mlx_fast_metal_kernel_config_new();
        const k_shape = [_]c_int{rows * K};
        const hc_shape = [_]c_int{rows * hc};
        const hchc_shape = [_]c_int{rows * hc * hc};
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cn, &k_shape, 1, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cn, &hchc_shape, 1, .float32));
        const xs_shape = [_]c_int{if (wr == 1) rows * K else 1};
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cn, &xs_shape, 1, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cn, 256 * hc, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cn, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cn, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cn, "HC", hc));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cn, "H", hidden));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cn, "INJ", inj));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cn, "WR", wr));
        const cd = mlx.mlx_fast_metal_kernel_config_new();
        const act_shape = [_]c_int{rows * R};
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cd, &act_shape, 1, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cd, &hc_shape, 1, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cd, 256, R + inj * hc, groups));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cd, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cd, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cd, "GS", gsi));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cd, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cd, "HC", hc));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cd, "H", hidden));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cd, "R", R));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cd, "ROWS", rows_per));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cd, "NROWS", rows));
        const cu = mlx.mlx_fast_metal_kernel_config_new();
        const mixed_shape = [_]c_int{rows * hidden};
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cu, &mixed_shape, 1, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cu, 32, hidden, groups));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cu, 32, 8, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cu, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cu, "GS", gsi));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cu, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cu, "HC", hc));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cu, "H", hidden));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cu, "R", R));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cu, "ROWS", rows_per));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cu, "NROWS", rows));
        slot.* = .{ .key = key, .cfgs = .{ cn, cd, cu } };
        hc_fused_cfg_builds += 1;
    }
    const cfgs = slot.*.?.cfgs;
    if (hc_fused_eps == null or hc_fused_eps_val != eps) {
        if (hc_fused_eps) |e| _ = mlx.mlx_array_free(e);
        const esh = [_]c_int{1};
        var ev = eps;
        hc_fused_eps = mlx.mlx_array_new_data(&ev, &esh, 1, .float32);
        hc_fused_eps_val = eps;
    }

    const apply = struct {
        fn f(st: mlx.mlx_stream, which: usize, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const mlx.mlx_array, n_out: usize, outs: []mlx.mlx_array) !void {
            const vec = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            var res = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(res);
            try mlx.check(mlx.mlx_fast_metal_kernel_apply(&res, try getHcFusedKernel(which), vec, cfg, st));
            if (mlx.mlx_vector_array_size(res) != n_out) return error.MetalKernelBadOutputCount;
            var got: usize = 0;
            errdefer {
                for (outs[0..got]) |a| _ = mlx.mlx_array_free(a);
            }
            for (0..n_out) |i| {
                outs[i] = mlx.mlx_array_new();
                got = i + 1;
                try mlx.check(mlx.mlx_vector_array_get(&outs[i], res, i));
            }
        }
    }.f;

    var n_out: [3]mlx.mlx_array = undefined;
    const wo = if (pend) |pd| pd.out else nw;
    const wi = if (pend) |pd| pd.inj else nw;
    try apply(s, 0, cfgs[0], &.{ x, nw, if (inj == 1) iw else nw, hc_fused_eps.?, wo, wi }, 3, &n_out);
    const xn = n_out[0];
    defer _ = mlx.mlx_array_free(xn);
    const ipart = n_out[1];
    defer _ = mlx.mlx_array_free(ipart);
    var stream_out = n_out[2];
    errdefer if (stream_out.ctx != null) {
        _ = mlx.mlx_array_free(stream_out);
    };
    if (wr == 0) {
        _ = mlx.mlx_array_free(stream_out);
        stream_out = .{ .ctx = null };
    } else {
        const xsh = mlx.getShape(x);
        var shaped = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(shaped);
        try mlx.check(mlx.mlx_reshape(&shaped, stream_out, xsh.ptr, @intCast(xsh.len), s));
        _ = mlx.mlx_array_free(stream_out);
        stream_out = shaped;
    }
    var d_out: [2]mlx.mlx_array = undefined;
    try apply(s, 1, cfgs[1], &.{ xn, dw, ds, db, ipart }, 2, &d_out);
    const act = d_out[0];
    defer _ = mlx.mlx_array_free(act);
    const inj_flat = d_out[1];
    defer _ = mlx.mlx_array_free(inj_flat);
    var u_out: [1]mlx.mlx_array = undefined;
    try apply(s, 2, cfgs[2], &.{ xn, act, uw, us, ub }, 1, &u_out);
    const mixed_flat = u_out[0];
    defer _ = mlx.mlx_array_free(mixed_flat);

    const mshape = [_]c_int{ batch, seq, hidden };
    var mixed = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(mixed);
    try mlx.check(mlx.mlx_reshape(&mixed, mixed_flat, &mshape, 3, s));
    var inj_out = mlx.mlx_array{ .ctx = null };
    if (inj == 1) {
        const ishape = [_]c_int{ batch, seq, hc, 1 };
        inj_out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&inj_out, inj_flat, &ishape, 4, s));
    }
    if (!hc_fused_engaged) {
        hc_fused_engaged = true;
        log.info("[qwen4] fused hyper-connection read engaged: hc={d} hidden={d} lowrank={d} {d}-bit g{d}\n", .{ hc, hidden, R, bits, group_size });
    }
    return .{ .mixed = mixed, .inj = inj_out, .stream = stream_out };
}
