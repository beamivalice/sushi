//! DFly2, Hy4-Flash's drafter (part 1x; docs/hy4-flash-arch.md#drafter-dfly2): a block `[t1, mask × draft]` drafted
//! in one pass over the target's own state. Each drafter layer is MLA in the sparse layers' shape with its own
//! projections, reading the target's cached rows at its shared layer (`dfly_target_kv_layers`) beside the block's own
//! rows; it keeps no KV of its own. The forward is the released drafter's training code, Tencent AngelSpec at
//! 0505773a (Apache-2.0; NOTICE). A drafter changes how fast the target speaks, never what it says.
const std = @import("std");
const mlx = @import("../mlx.zig");
const plan = @import("../tensor/plan/plan_api.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const model_mod = @import("../model.zig");
const fwd = @import("hy4_forward.zig");
const hy4_model = @import("hy4_model.zig");
const sushi_log = @import("../log.zig");
const Arr = mlx.mlx_array;
const Model = hy4_model.Model;

const Layer = struct {
    attn: fwd.Attn,
    geometry: fwd.AttnGeometry,
    input_norm: Arr,
    post_norm: Arr,
    mlp: fwd.Swiglu,
    /// The target layer whose cached rows it reads.
    target: usize,
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    layers: []Layer,
    enorm: Arr,
    hnorm: Arr,
    eh_proj: fwd.Lin,
    fusion_gate: Arr,
    embed_norm: Arr,
    hidden_norm: Arr,
    correction: fwd.Swiglu,
    conf_w: Arr,
    conf_b: Arr,
    final_norm: Arr,

    /// The drafter's tensors (`model.dfly_layers.*`) from `w`, trellis banks where the pack stores them. Sets the
    /// target to keep its last hidden state.
    pub fn init(gpa: std.mem.Allocator, m: *Model, w: *model_mod.Weights) !Drafter {
        const c = m.config;
        const d = c.dfly;
        if (d.layers == 0 or d.n_targets != d.layers) return error.Hy4DflyConfig;
        defer m.freeDecodeEye();
        const layers = try gpa.alloc(Layer, d.layers);
        errdefer gpa.free(layers);
        const p = "model.dfly_layers.";
        for (layers, 0..) |*ly, i| {
            const a = p ++ "{d}.self_attn.";
            const target: usize = d.targets[i];
            if (target >= c.layers or c.kinds[target] != .sparse) return error.Hy4DflyTargetNotSparse;
            ly.* = .{
                .attn = .{
                    .q_a = try m.lin(w, a ++ "q_a_proj", .{i}),
                    .q_a_norm = try get(w, a ++ "q_a_layernorm.weight", .{i}),
                    .q_b = try m.lin(w, a ++ "q_b_proj", .{i}),
                    .kv_a = try m.lin(w, a ++ "kv_a_proj_with_mqa", .{i}),
                    .kv_a_norm = try get(w, a ++ "kv_a_layernorm.weight", .{i}),
                    .kv_b = try m.denseOf(w, a ++ "kv_b_proj", .{i}),
                    .gate = try m.lin(w, a ++ "linear_gate", .{i}),
                    .o = try m.lin(w, a ++ "o_proj", .{i}),
                    .post_norm = try get(w, a ++ "post_proj_norm.weight", .{i}),
                    .sinks = try get(w, a ++ "learnable_sink_param", .{i}),
                },
                // The sparse layers' shape at the target's own rope.
                .geometry = .{
                    .heads = @intCast(c.sparse_heads),
                    .nope = @intCast(c.nope_dim),
                    .rope = @intCast(c.rope_dim),
                    .v = @intCast(c.v_dim),
                    .latent = @intCast(c.kv_lora),
                    .theta = @floatCast(c.sparse_rope.theta),
                    .eps = c.eps,
                },
                .input_norm = try get(w, p ++ "{d}.input_layernorm.weight", .{i}),
                .post_norm = try get(w, p ++ "{d}.post_attention_layernorm.weight", .{i}),
                .mlp = .{ .gate = try m.lin(w, p ++ "{d}.mlp.gate_proj", .{i}), .up = try m.lin(w, p ++ "{d}.mlp.up_proj", .{i}), .down = try m.lin(w, p ++ "{d}.mlp.down_proj", .{i}) },
                .target = target,
            };
        }
        const hc = p ++ "hidden_correction.";
        m.keep_hidden = true;
        return .{
            .gpa = gpa,
            .layers = layers,
            .enorm = try get(w, p ++ "mtp_fusion.enorm.weight", .{}),
            .hnorm = try get(w, p ++ "mtp_fusion.hnorm.weight", .{}),
            .eh_proj = try m.lin(w, p ++ "mtp_fusion.eh_proj", .{}),
            .fusion_gate = try get(w, p ++ "mtp_fusion.gate", .{}),
            .embed_norm = try get(w, hc ++ "embed_norm.weight", .{}),
            .hidden_norm = try get(w, hc ++ "hidden_norm.weight", .{}),
            .correction = .{ .gate = try m.lin(w, hc ++ "gate_proj", .{}), .up = try m.lin(w, hc ++ "up_proj", .{}), .down = try m.lin(w, hc ++ "down_proj", .{}) },
            .conf_w = try get(w, p ++ "confidence_head.proj.weight", .{}),
            .conf_b = try get(w, p ++ "confidence_head.proj.bias", .{}),
            .final_norm = try get(w, p ++ "final_norm.weight", .{}),
        };
    }

    pub fn deinit(self: *Drafter) void {
        self.gpa.free(self.layers);
    }

    /// A drafted block: `ids` the pending token then the drafts, `conf` each draft's confidence in (0, 1).
    pub const Draft = struct {
        gpa: std.mem.Allocator,
        ids: []u32,
        conf: []f32,

        pub fn deinit(self: *Draft) void {
            self.gpa.free(self.ids);
            self.gpa.free(self.conf);
        }
    };

    /// The block `[t1, mask × draft]` at the target's next positions, drafted from the target's state as it stands
    /// (its last hidden state and its shared layers' rows). Reads the target, never writes it. Each mask row's argmax
    /// is its draft (a point mass, which the verify's exact rule accepts at any temperature).
    pub fn draft(self: *const Drafter, m: *Model, t1: u32) !Draft {
        return self.draftLimit(m, t1, null);
    }

    /// `draft` capped to `max_drafts` slots (when non-null, clamped to `c.dfly.draft`), skipping the unused
    /// `[draft, vocab]` logits concat and syncing `best` + `conf` in a single GPU wait.
    pub fn draftLimit(self: *const Drafter, m: *Model, t1: u32, max_drafts: ?usize) !Draft {
        const cap: usize = m.config.dfly.draft;
        const n_draft: usize = @min(max_drafts orelse cap, cap);
        if (n_draft == 0) return error.Hy4RowsWidth;
        var o: Ops = .{ .s = m.s };
        defer o.deinit();
        const out = try self.forwardBlockFor(&o, m, t1, n_draft, false);
        const best = try o.contiguous(try o.cast(out.best, .uint32));
        const conf = try o.contiguous(try o.cast(out.conf, .float32));
        const vec = mlx.mlx_vector_array_new_data(&[_]Arr{ best, conf }, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        const ids = try self.gpa.alloc(u32, n_draft + 1);
        errdefer self.gpa.free(ids);
        ids[0] = t1;
        // The block's draft and confidence are its results: they steer the round that reads them, not the block.
        return hostDraft(self.gpa, ids, (try mlx.hostResult(u32, best, @intCast(m.config.vocab))).ptr, (try mlx.hostResult(f32, conf, null)).ptr);
    }

    /// One block's outputs in `o`, a draft slot a row: its logits [draft, vocab] f32, its confidence [draft] in (0, 1)
    /// and its draft, the logits' argmax [draft].
    pub const Block = struct { logits: Arr, conf: Arr, best: Arr };

    /// `draft`'s forward, its outputs left in `o` (the oracle test's seam).
    pub fn forwardBlock(self: *const Drafter, o: *Ops, m: *Model, t1: u32) !Block {
        return self.forwardBlockFor(o, m, t1, m.config.dfly.draft, true);
    }

    fn forwardBlockFor(self: *const Drafter, o: *Ops, m: *Model, t1: u32, n_draft: usize, need_logits: bool) !Block {
        const c = m.config;
        const t: c_int = @intCast(n_draft + 1);
        const req_ = m.reqConst();
        const h = req_.last_hidden orelse return error.Hy4DflyNoHandover;
        const p0: c_int = @intCast(req_.pos);
        const ids = try self.gpa.alloc(u32, n_draft + 1);
        defer self.gpa.free(ids);
        ids[0] = t1;
        @memset(ids[1..], c.dfly.mask_id);
        const dim: c_int = @intCast(c.dim);
        const e = try embedRows(o, m, ids); // [T, D]
        // Slot 0 is the anchor's embedding plus the gated fusion of it with the target's final hidden state, through
        // the target's final norm (AngelSpec `MTPFusion`, `slot0`); the mask slots enter as their embedding alone.
        const e0 = try o.slice(e, 0, 0, 1);
        const hn = try fwd.norm(o, try fwd.norm(o, try o.reshape(h, &.{ 1, dim }), m.final_norm, c.eps), self.hnorm, c.eps);
        const fused = try fwd.linear(o, try o.concat(&.{ try fwd.norm(o, e0, self.enorm, c.eps), hn }, 1), self.eh_proj);
        const row0 = try o.binary(.add, e0, try o.binary(.mul, fused, try o.reshape(try o.cast(self.fusion_gate, .float32), &.{ 1, dim })));
        var x = try o.concat(&.{ row0, try o.slice(e, 0, 1, t) }, 0);
        for (self.layers, 0..) |*ly, i| {
            // A layer is a frame of its own; inside it, whether it reads its target's selection (past the threshold) or
            // every row of it picks the variant (`Model.sharedRows`).
            const f = try plan.frame(@src(), .{i});
            defer f.end();
            const v = if (plan.planned()) try plan.each(@src(), req_.last_picks[ly.target] != null) else plan.Frame.none;
            defer v.end();
            var ol: Ops = .{ .s = o.s };
            defer ol.deinit();
            const a = try fwd.norm(&ol, x, ly.input_norm, c.eps);
            const pr = try fwd.project(&ol, a, ly.attn, ly.geometry, p0);
            const shared = try m.sharedRows(&ol, ly.target);
            const keys_c = try ol.concat(&.{ shared.c, pr.c }, 0);
            const keys_pe = try ol.concat(&.{ shared.k_pe, pr.k_pe }, 0);
            const mask = try blockMask(&ol, t, mlx.getShape(shared.c)[0], shared.keep);
            const attn_y = if (need_logits)
                try fwd.attend(&ol, a, pr.q, keys_c, keys_pe, ly.attn, ly.geometry, mask, null)
            else
                try fwd.attendDrafter(&ol, .{ .gate = try fwd.linearStored(&ol, a, ly.attn.gate), .whole = true }, pr.q, keys_c, keys_pe, ly.attn, ly.geometry, mask);
            const x1 = try ol.binary(.add, x, attn_y);
            const x2 = try ol.binary(.add, x1, try fwd.swiglu(&ol, try fwd.norm(&ol, x1, ly.post_norm, c.eps), ly.mlp, null));
            x = try o.own(try ol.result(x2));
        }
        const hf = try fwd.norm(o, try o.slice(x, 0, 1, t), self.final_norm, c.eps); // the mask slots
        // The hidden correction's served chain (AngelSpec `_served_draft_chain`), one slot after another: slot j's
        // hidden plus `down(silu(gate·z) ⊙ up·z)`, `z = [hidden_norm(h_j) ; embed_norm(E(prev))]`, where prev is the
        // anchor for the first slot and the slot before's draft after it. The head and the confidence read the result.
        const parts = try self.gpa.alloc(Arr, 3 * n_draft);
        defer self.gpa.free(parts);
        var prev_e = e0;
        for (0..n_draft) |j| {
            var os: Ops = .{ .s = o.s };
            defer os.deinit();
            const r: c_int = @intCast(j);
            const hj = try os.slice(hf, 0, r, r + 1);
            const z = try os.concat(&.{ try fwd.norm(&os, hj, self.hidden_norm, c.eps), try fwd.norm(&os, prev_e, self.embed_norm, c.eps) }, 1);
            const hc = try os.binary(.add, hj, try fwd.swiglu(&os, z, self.correction, null));
            const logits = try m.lmHead(&os, hc);
            const am = try os.slot();
            try mlx.check(mlx.mlx_argmax_axis(am, logits, 1, false, os.s));
            const best = try os.contiguous(try os.cast(am.*, .uint32));
            const conf = try os.unary(.sigmoid, try os.binary(.add, try fwd.linearF32(&os, hc, self.conf_w), try os.cast(self.conf_b, .float32)));
            if (need_logits) parts[j] = try o.own(try os.result(logits));
            parts[n_draft + j] = try o.own(try os.result(conf));
            parts[2 * n_draft + j] = try o.own(try os.result(best));
            if (j + 1 < n_draft) prev_e = try o.own(try os.result(try embedRowId(&os, m, best)));
        }
        return .{
            .logits = if (need_logits) try o.concat(parts[0..n_draft], 0) else mlx.mlx_array_new(),
            .conf = try o.reshape(try o.concat(parts[n_draft .. 2 * n_draft], 0), &.{t - 1}),
            .best = try o.concat(parts[2 * n_draft ..], 0),
        };
    }

    pub const max_batch_slots: usize = @intCast(@import("sushi_exl3").project_rows.NAX_ROWS_MAX);

    /// Batched DFly2 draft across `reqs.len` concurrent requests in ONE joined pass through all 5 DFly2 layers
    /// (reading DFly2 weights once instead of N times) and only `n_use = @min(max_drafts, c.dfly.draft)` steps of the
    /// hidden correction chain + shared `m.lmHead` (e.g. 1 LM head read at N=8 instead of 40), with ONE GPU wait.
    pub fn draftBatch(
        self: *const Drafter,
        m: *Model,
        reqs: []const *hy4_model.Request,
        t1s: []const u32,
        max_drafts: usize,
        out_drafts: []Draft,
    ) !void {
        const n = reqs.len;
        if (n == 0 or n > max_batch_slots or t1s.len != n or out_drafts.len != n) return error.Hy4RowsWidth;
        const c = m.config;
        const n_draft: usize = c.dfly.draft;
        const n_use: usize = @min(max_drafts, n_draft);
        if (n_use == 0) return error.Hy4RowsWidth;
        if (n == 1) {
            const lent = m.lend(reqs[0]);
            defer lent.end();
            out_drafts[0] = try self.draft(m, t1s[0]);
            return;
        }
        // `blockMask` is causal within each slot's block (`dflash_causal_attn`), so rows `0..n_use` never attend to
        // mask positions past `n_use`: running `n_use + 1` rows/slot matches `n_draft + 1` bit-for-bit while cutting
        // N=8 draft rows 3x (48 -> 16, staying inside the NAX decode arm).
        const t: c_int = @intCast(n_use + 1);
        const tu: usize = n_use + 1;
        const total: usize = n * tu;
        if (total > 64) return error.Hy4RowsWidth;
        const dim: c_int = @intCast(c.dim);
        var o: Ops = .{ .s = m.s };
        defer o.deinit();

        // Gather N anchors + 1 shared mask embedding in one call (data-independent plan bound).
        var embed_ids: [max_batch_slots + 1]u32 = undefined;
        @memcpy(embed_ids[0..n], t1s);
        embed_ids[n] = c.dfly.mask_id;
        const e_in = try embedRows(&o, m, embed_ids[0 .. n + 1]);
        const e0_all = try o.slice(e_in, 0, 0, @intCast(n));
        const mask_row = try o.slice(e_in, 0, @intCast(n), @intCast(n + 1));
        const mask_block = try o.broadcast(mask_row, &.{ @intCast(n_use), dim });

        var h_parts: [max_batch_slots]Arr = undefined;
        var flat_pos: [64]c_int = undefined;
        for (reqs, 0..) |req_i, i| {
            const h = req_i.last_hidden orelse return error.Hy4DflyNoHandover;
            h_parts[i] = try o.reshape(h, &.{ 1, dim });
            const p0: c_int = @intCast(req_i.pos);
            for (0..tu) |k| flat_pos[i * tu + k] = p0 + @as(c_int, @intCast(k));
        }
        const h_all = try o.concat(h_parts[0..n], 0);
        const hn = try fwd.norm(&o, try fwd.norm(&o, h_all, m.final_norm, c.eps), self.hnorm, c.eps);
        const fused = try fwd.linear(&o, try o.concat(&.{ try fwd.norm(&o, e0_all, self.enorm, c.eps), hn }, 1), self.eh_proj);
        const row0_all = try o.binary(.add, e0_all, try o.binary(.mul, fused, try o.reshape(try o.cast(self.fusion_gate, .float32), &.{ 1, dim })));
        var x_parts: [2 * max_batch_slots]Arr = undefined;
        for (0..n) |i| {
            const ri: c_int = @intCast(i);
            x_parts[2 * i] = try o.slice(row0_all, 0, ri, ri + 1);
            x_parts[2 * i + 1] = mask_block;
        }
        var x = try o.concat(x_parts[0 .. 2 * n], 0);

        for (self.layers, 0..) |*ly, li| {
            const f = try plan.frame(@src(), .{li});
            defer f.end();
            var ol: Ops = .{ .s = o.s };
            defer ol.deinit();
            const a = try fwd.norm(&ol, x, ly.input_norm, c.eps);
            const pr = try fwd.projectRowsJoinedFull(&ol, a, ly.attn, ly.geometry, flat_pos[0..total], .float32, .float32);
            const gate_all = try fwd.linearStored(&ol, a, ly.attn.gate);
            var gated_parts: [max_batch_slots]Arr = undefined;
            {
                const slots_f = try plan.frame(@src(), .{});
                defer slots_f.end();
                for (reqs, 0..) |req_i, i| {
                    const lent = m.lend(req_i);
                    defer lent.end();
                    const v = if (plan.planned()) try plan.each(@src(), req_i.last_picks[ly.target] != null) else plan.Frame.none;
                    defer v.end();
                    var sl: Ops = .{ .s = ol.s };
                    defer sl.deinit();
                    const r0: c_int = @intCast(i * tu);
                    const r1: c_int = r0 + t;
                    const shared = try m.sharedRows(&sl, ly.target);
                    const keys_c = try sl.concat(&.{ shared.c, try sl.slice(pr.c, 0, r0, r1) }, 0);
                    const keys_pe = try sl.concat(&.{ shared.k_pe, try sl.slice(pr.k_pe, 0, r0, r1) }, 0);
                    const mask = try blockMask(&sl, t, mlx.getShape(shared.c)[0], shared.keep);
                    const q_i = try sl.slice(pr.q_full.?, 1, r0, r1);
                    const q_raw_i = if (pr.q_raw) |qr| try sl.slice(qr, 1, r0, r1) else null;
                    const gate_i = try sl.slice(gate_all, 0, r0, r1);
                    const g_i = try fwd.attendDrafter(&sl, .{ .gate = gate_i, .whole = false, .q_raw = q_raw_i }, q_i, keys_c, keys_pe, ly.attn, ly.geometry, mask);
                    gated_parts[i] = try ol.own(try sl.result(g_i));
                }
            }
            const attn_out = try fwd.output(&ol, try ol.concat(gated_parts[0..n], 0), ly.attn, ly.geometry, null);
            const x1 = try ol.binary(.add, x, attn_out);
            const x2 = try ol.binary(.add, x1, try fwd.swiglu(&ol, try fwd.norm(&ol, x1, ly.post_norm, c.eps), ly.mlp, null));
            x = try o.own(try ol.result(x2));
        }

        // Gather the first `n_use` mask slots of every request laid out as `[n_use, n, dim]` -> `[n_use * n, dim]`.
        const x3 = try o.reshape(x, &.{ @intCast(n), t, dim });
        const masks_only = try o.slice(x3, 1, 1, @intCast(1 + n_use));
        const masks_t = try o.contiguous(try o.transpose(masks_only, &.{ 1, 0, 2 }));
        const hf_all = try fwd.norm(&o, try o.reshape(masks_t, &.{ @intCast(n_use * n), dim }), self.final_norm, c.eps);

        var best_steps: [max_probe_slots]Arr = undefined;
        var conf_steps: [max_probe_slots]Arr = undefined;
        var prev_e = e0_all;
        for (0..n_use) |j| {
            var os: Ops = .{ .s = o.s };
            defer os.deinit();
            const r0: c_int = @intCast(j * n);
            const r1: c_int = @intCast((j + 1) * n);
            const hj = try os.slice(hf_all, 0, r0, r1);
            const z = try os.concat(&.{ try fwd.norm(&os, hj, self.hidden_norm, c.eps), try fwd.norm(&os, prev_e, self.embed_norm, c.eps) }, 1);
            const hc = try os.binary(.add, hj, try fwd.swiglu(&os, z, self.correction, null));
            const logits = try m.lmHead(&os, hc);
            const am = try os.slot();
            try mlx.check(mlx.mlx_argmax_axis(am, logits, 1, false, os.s));
            const best = try os.contiguous(try os.cast(am.*, .uint32));
            const conf = try os.reshape(try os.unary(.sigmoid, try os.binary(.add, try fwd.linearF32(&os, hc, self.conf_w), try os.cast(self.conf_b, .float32))), &.{@intCast(n)});
            best_steps[j] = try o.own(try os.result(best));
            conf_steps[j] = try o.own(try os.result(conf));
            if (j + 1 < n_use) prev_e = try o.own(try os.result(try embedRowIds(&os, m, best, n)));
        }

        const all_best = try o.contiguous(try o.concat(best_steps[0..n_use], 0));
        const all_conf = try o.contiguous(try o.concat(conf_steps[0..n_use], 0));
        const vec = mlx.mlx_vector_array_new_data(&[_]Arr{ all_best, all_conf }, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        const best_host = try mlx.hostResult(u32, all_best, @intCast(c.vocab));
        const conf_host = try mlx.hostResult(f32, all_conf, null);

        var made: usize = 0;
        errdefer for (out_drafts[0..made]) |*d| d.deinit();
        for (0..n) |i| {
            const ids = try self.gpa.alloc(u32, n_use + 1);
            errdefer self.gpa.free(ids);
            const conf = try self.gpa.alloc(f32, n_use);
            ids[0] = t1s[i];
            for (0..n_use) |j| {
                ids[1 + j] = best_host[j * n + i];
                conf[j] = conf_host[j * n + i];
            }
            out_drafts[i] = .{ .gpa = self.gpa, .ids = ids, .conf = conf };
            made = i + 1;
        }
    }
};

/// The block `ids` (its drafts filled from the host rows `best`) with the confidence rows `conf` copied; null rows are
/// a failed host read, which is told before anything is allocated (`ids` stays the caller's).
pub fn hostDraft(gpa: std.mem.Allocator, ids: []u32, best: ?[*]const u32, conf: ?[*]const f32) !Drafter.Draft {
    const n_draft = ids.len - 1;
    const best_rows = (best orelse return error.HostRead)[0..n_draft];
    const conf_rows = (conf orelse return error.HostRead)[0..n_draft];
    const conf_out = try gpa.dupe(f32, conf_rows);
    @memcpy(ids[1..], best_rows);
    return .{ .gpa = gpa, .ids = ids, .conf = conf_out };
}

/// What the acceptance probe counts, over the blocks it scored (`probe`).
pub const Tally = struct {
    blocks: usize = 0,
    /// Per slot: blocks whose slots before it all matched, those in which it matched too, it matched alone, and the
    /// sum of its confidence.
    reached: [max_probe_slots]usize = @splat(0),
    accepted: [max_probe_slots]usize = @splat(0),
    alone: [max_probe_slots]usize = @splat(0),
    conf: [max_probe_slots]f64 = @splat(0),

    /// One block's `drafts` and their confidence against the greedy tokens after its pending one, `want`.
    pub fn score(self: *Tally, drafts: []const u32, conf: []const f32, want: []const u32) void {
        self.blocks += 1;
        var run = true;
        for (drafts, conf, want, 0..) |id, cf, w, k| {
            if (run) self.reached[k] += 1;
            run = run and id == w;
            if (run) self.accepted[k] += 1;
            if (id == w) self.alone[k] += 1;
            self.conf[k] += cf;
        }
    }

    pub fn add(self: *Tally, other: *const Tally) void {
        self.blocks += other.blocks;
        inline for (.{ "reached", "accepted", "alone", "conf" }) |f| {
            for (&@field(self, f), @field(other, f)) |*a, b| a.* += b;
        }
    }

    pub fn log(self: *const Tally, label: []const u8, slots: usize) void {
        var sum: usize = 0;
        for (self.accepted[0..slots]) |n| sum += n;
        const blocks: f64 = @floatFromInt(@max(self.blocks, 1));
        sushi_log.info("[dfly-probe] {s}: {d} blocks, {d:.3} accepted a block (of {d})\n", .{ label, self.blocks, @as(f64, @floatFromInt(sum)) / blocks, slots });
        for (0..slots) |k| {
            const reached: f64 = @floatFromInt(@max(self.reached[k], 1));
            sushi_log.info("[dfly-probe] {s} slot {d}: accepted {d:.3} given the slots before ({d}/{d}), matched alone {d:.3}, confidence {d:.3}\n", .{
                label,                                                     k + 1,
                @as(f64, @floatFromInt(self.accepted[k])) / reached,      self.accepted[k],
                self.reached[k],                                           @as(f64, @floatFromInt(self.alone[k])) / blocks,
                self.conf[k] / blocks,
            });
        }
    }
};
const max_probe_slots = 16;

/// The probe's built-in prompts, chat-formatted with thinking off: a code reply and a prose reply.
pub const probe_prompts = [_][]const u8{
    turn("Write a Python function that parses an ISO 8601 date string into a datetime, with a docstring and three unit tests."),
    turn("Explain in three paragraphs why the sky is blue and why sunsets are red."),
};

fn turn(comptime user: []const u8) []const u8 {
    return "<\u{ff5c}hy_start:opensource\u{ff5c}>system<\u{ff5c}hy_middle:opensource\u{ff5c}><\u{ff5c}reasoning_mode:opensource\u{ff5c}>reasoning_effort:no_think<\u{ff5c}hy_end:opensource\u{ff5c}>" ++
        "<\u{ff5c}hy_start:opensource\u{ff5c}>user<\u{ff5c}hy_middle:opensource\u{ff5c}>" ++ user ++ "<\u{ff5c}hy_end:opensource\u{ff5c}>" ++
        "<\u{ff5c}hy_start:opensource\u{ff5c}>assistant<\u{ff5c}hy_middle:opensource\u{ff5c}><think:opensource></think:opensource>";
}

/// DIAGNOSTIC (`SUSHI_HY4_DFLY_PROBE`, docs/hy4-flash-arch.md#dfly2-parity): DFly2's teacher-forced acceptance on
/// `prompt`. The target decodes `tokens` greedily alone (to its EOS at most); at every `every`-th position the
/// drafter drafts a block from the state there, scored against the greedy tokens after it into `tally`. Resets the
/// model before and after.
pub fn probe(d: *const Drafter, m: *Model, prompt: []const u32, tokens: usize, every: usize, tally: *Tally) !void {
    const gpa = d.gpa;
    const slots: usize = m.config.dfly.draft;
    if (slots > max_probe_slots or every == 0) return error.Hy4DflyProbe;
    m.reset();
    defer m.reset();
    const out = try gpa.alloc(u32, tokens);
    defer gpa.free(out);
    const at = try gpa.alloc(?Drafter.Draft, tokens);
    defer {
        for (at) |*b| if (b.*) |*v| v.deinit();
        gpa.free(at);
    }
    @memset(at, null);
    var next = try greedy(m, prompt);
    var n: usize = 0;
    while (n < tokens) {
        if (n % every == 0) at[n] = try d.draft(m, next);
        out[n] = next;
        n += 1;
        if (next == m.config.eos) break;
        next = try greedy(m, &.{next});
    }
    for (at[0..n], 0..) |b, i| {
        const blk = b orelse continue;
        if (i + slots >= n) break;
        tally.score(blk.ids[1..], blk.conf, out[i + 1 ..][0..slots]);
    }
}

/// The target's greedy next token after `ids`.
fn greedy(m: *Model, ids: []const u32) !u32 {
    var o: Ops = .{ .s = m.s };
    defer o.deinit();
    if (m.experts) |e| e.beginForward();
    const logits = try o.own(try m.forwardReadout(ids, .last));
    if (m.experts) |e| e.finishForward(ids.len);
    const am = try o.slot();
    try mlx.check(mlx.mlx_argmax_axis(am, logits, 1, false, o.s));
    const best = try o.contiguous(try o.cast(am.*, .uint32));
    try mlx.check(mlx.mlx_array_eval(best));
    return (mlx.mlx_array_data_uint32(best) orelse return error.HostRead)[0];
}

/// The embedding rows of `ids` [n, D] f32, as the target's first stream holds them.
fn embedRows(o: *Ops, m: *Model, ids: []const u32) !Arr {
    const streams = try o.own(try m.embedStreams(ids));
    return o.reshape(try o.slice(streams, 1, 0, 1), &.{ @intCast(ids.len), @intCast(m.config.dim) });
}

fn embedRowId(o: *Ops, m: *Model, id: Arr) !Arr {
    if (m.embed_gpu) |g| return o.own(try g.rowFromId(o.s, id));
    if (m.embed_gather) |_| {
        try mlx.check(mlx.mlx_array_eval(id));
        const host_id = (try mlx.hostResult(u32, id, @intCast(m.config.vocab)))[0];
        return embedRows(o, m, &.{host_id});
    }
    return o.cast(try o.take(m.embed, id, 0), .float32);
}

fn embedRowIds(o: *Ops, m: *Model, ids: Arr, n: usize) !Arr {
    if (m.embed_gpu) |g| {
        var parts: [Drafter.max_batch_slots]Arr = undefined;
        for (0..n) |i| {
            const ri: c_int = @intCast(i);
            parts[i] = try o.own(try g.rowFromId(o.s, try o.slice(ids, 0, ri, ri + 1)));
        }
        return o.concat(parts[0..n], 0);
    }
    if (m.embed_gather) |_| {
        try mlx.check(mlx.mlx_array_eval(ids));
        const host_ptr = try mlx.hostResult(u32, ids, @intCast(m.config.vocab));
        return embedRows(o, m, host_ptr[0..n]);
    }
    return o.cast(try o.take(m.embed, ids, 0), .float32);
}

/// The block's mask [T, p0 + T] over the `p0` shared rows and the block: the shared rows the target read (`keep`), and
/// the block causal within itself (the release trained with `dflash_causal_attn`).
fn blockMask(o: *Ops, t: c_int, p0: c_int, keep: ?Arr) !Arr {
    // The shared rows: every one the target's last row read (its selection's `keep`), for every block row.
    const target = if (keep) |k| try o.broadcast(try fwd.additive(o, k), &.{ t, p0 }) else try o.zeros(&.{ t, p0 }, .float32);
    return o.concat(&.{ target, try fwd.causalMask(o, t, t, 0, 0) }, 1);
}

fn get(w: *const model_mod.Weights, comptime fmt: []const u8, args: anytype) !Arr {
    var buf: [160]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, args);
    return w.get(name) orelse {
        std.log.err("hy_v4_flash: the pack has no DFly2 tensor {s}", .{name});
        return error.Hy4TensorMissing;
    };
}

test "hy4 dfly probe: a block's slots count accepted up to its first miss, each slot's match alone and its confidence" {
    var t: Tally = .{};
    t.score(&.{ 7, 8, 1, 9, 9 }, &.{ 0.5, 0.25, 0.5, 1, 1 }, &.{ 7, 8, 3, 9, 4 });
    t.score(&.{ 2, 8, 3, 9, 4 }, &.{ 0.5, 0.25, 0.5, 0, 0 }, &.{ 7, 8, 3, 9, 4 });
    try std.testing.expectEqual(@as(usize, 2), t.blocks);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 1, 0, 0 }, t.reached[0..5]);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 0, 0, 0 }, t.accepted[0..5]);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 1, 2, 1 }, t.alone[0..5]);
    try std.testing.expectEqualSlices(f64, &.{ 1, 0.5, 1, 1, 1 }, t.conf[0..5]);
    var all: Tally = .{};
    all.add(&t);
    all.add(&t);
    try std.testing.expectEqual(@as(usize, 4), all.blocks);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 0, 0, 0 }, all.accepted[0..5]);
}
