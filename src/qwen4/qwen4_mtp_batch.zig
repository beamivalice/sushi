//! Flash-Next's draft step for several requests' MTP chains in one head forward (`Drafter.buildBatchedChains`).
const std = @import("std");
const qwen4_forward = @import("qwen4_forward.zig");
const mlx = @import("../mlx.zig");
const trunk_mod = @import("../trunk.zig");
const generate = @import("../generate.zig");
const mtp_round = @import("../mtp_round.zig");
const sampler = @import("../sampler.zig");
const Generator = generate.Generator;
const MtpPreDraft = mtp_round.MtpPreDraft;

/// Chains `[from, to)` of every request, one head forward per step. Rows are row-axis stacked, so no request may
/// carry M-RoPE positions (the caller sends those down the solo build).
pub fn build(gens: []const *Generator, chains: []MtpPreDraft, from: u32, to: u32) !void {
    const xfm = gens[0].xfm;
    const n = gens.len;
    var i: u32 = from;
    while (i < to) : (i += 1) {
        var rows: [32]qwen4_forward.Qwen4MtpBatchRow = undefined;
        var merged_ids: [32]mlx.mlx_array = undefined;
        var merged_hidden: [32]mlx.mlx_array = undefined;
        var own_ids: [32]bool = @splat(false);
        var own_h: [32]bool = @splat(false);
        for (merged_ids[0..n], 0..) |*p, k| {
            p.* = .{ .ctx = null };
            merged_hidden[k] = .{ .ctx = null };
        }
        defer for (0..n) |k| {
            if (own_ids[k] and merged_ids[k].ctx != null) _ = mlx.mlx_array_free(merged_ids[k]);
            if (own_h[k] and merged_hidden[k].ctx != null) _ = mlx.mlx_array_free(merged_hidden[k]);
        };
        var modes: [32]bool = undefined;
        for (gens, chains, 0..) |g, *c, row| modes[row] = mtp_round.mtpRowRerankMode(g, c, i);
        const use_rerank = modes[0];
        const project: qwen4_forward.Qwen4MtpProject = if (use_rerank) .mixed_last_row else .last_row;
        for (gens, chains, 0..) |g, *c, k| {
            const qref = g.mtp_cache.?.as("qwen4");
            const st = qref.batchState();
            const h_prev_arg: mlx.mlx_array = if (c.h_chain) |h| h else g.last_hidden;
            const prev_tok_arr: mlx.mlx_array = if (i == 0) c.t1_arr else c.draft_arrs[i - 1];
            if (i == 0 and g.mtp_hist_stash != null) {
                var stash = g.mtp_hist_stash.?;
                g.mtp_hist_stash = null;
                defer stash.deinit();
                try qwen4_forward.qwen4MtpTruncateOn(qref.t, st, stash.off0);
                var mids = mlx.mlx_array_new();
                var mh = mlx.mlx_array_new();
                {
                    const idv = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(idv);
                    _ = mlx.mlx_vector_array_append_value(idv, stash.ids);
                    _ = mlx.mlx_vector_array_append_value(idv, prev_tok_arr);
                    try mlx.check(mlx.mlx_concatenate_axis(&mids, idv, 0, xfm.s));
                    const hv = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(hv);
                    _ = mlx.mlx_vector_array_append_value(hv, stash.hidden);
                    _ = mlx.mlx_vector_array_append_value(hv, h_prev_arg);
                    try mlx.check(mlx.mlx_concatenate_axis(&mh, hv, 1, xfm.s));
                }
                merged_ids[k] = mids;
                merged_hidden[k] = mh;
                own_ids[k] = true;
                own_h[k] = true;
            } else {
                merged_ids[k] = prev_tok_arr;
                merged_hidden[k] = h_prev_arg;
            }
            rows[k] = .{
                .st = st,
                .stream = merged_hidden[k],
                .token_ids = merged_ids[k],
                .pos_offset = mtp_round.mtpBatchHeadPosition(st.pos_base, st.seq_offset),
            };
        }
        const outs = try qwen4_forward.qwen4MtpForwardBatched(xfm, rows[0..n], project, to);
        defer {
            for (outs) |out| {
                if (out.logits.ctx != null) _ = mlx.mlx_array_free(out.logits);
                if (out.stream.ctx != null) _ = mlx.mlx_array_free(out.stream);
                if (out.mixed.ctx != null) _ = mlx.mlx_array_free(out.mixed);
            }
            xfm.allocator.free(outs);
        }
        var mixed_parts: [32]mlx.mlx_array = undefined;
        if (use_rerank) {
            for (outs, 0..) |o, k| mixed_parts[k] = o.mixed;
            const mvec = mlx.mlx_vector_array_new_data(mixed_parts[0..n].ptr, n);
            defer _ = mlx.mlx_vector_array_free(mvec);
            var stacked_x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(stacked_x);
            try mlx.check(mlx.mlx_concatenate_axis(&stacked_x, mvec, 0, xfm.s));
            var any_sampled = false;
            for (chains) |*c| {
                if (c.q_probs != null) any_sampled = true;
            }
            if (any_sampled) {
                try mtp_round.mtpGroupSampledDrafts(gens, chains, stacked_x, i);
            } else {
                const ids = try qwen4_forward.qwen4DraftSelectBatched(xfm, stacked_x, gens[0].sampling.suppress_mask);
                defer _ = mlx.mlx_array_free(ids);
                for (chains, 0..) |*c, k| {
                    const k_c: c_int = @intCast(k);
                    var one = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_slice(&one, ids, &[_]c_int{ k_c, 0 }, 2, &[_]c_int{ k_c + 1, 1 }, 2, &[_]c_int{ 1, 1 }, 2, xfm.s));
                    c.draft_arrs[i] = one;
                }
            }
            for (chains, 0..) |*c, k| {
                c.n_drafted = i + 1;
                if (c.h_chain) |h_old| _ = mlx.mlx_array_free(h_old);
                c.h_chain = outs[k].stream;
                outs[k].stream = .{ .ctx = null };
            }
        } else {
            for (gens, chains, 0..) |g, *c, k| {
                const draft_sampling = g.mtpDraftSampling(i);
                if (c.q_probs) |slots| {
                    const prop = try sampler.logitsStepProposal(outs[k].logits, g.mtpSamplingDraw(draft_sampling), xfm.s);
                    slots[i] = prop.q;
                    c.n_qp = i + 1;
                    c.draft_arrs[i] = prop.id;
                } else {
                    c.draft_arrs[i] = sampler.sampleTokenLazy(outs[k].logits, draft_sampling, xfm.s);
                }
                c.n_drafted = i + 1;
                if (c.conf_arrs != null and i < c.plan.m_lo) {
                    c.conf_arrs.?[i] = try mtp_round.draftConfidenceGraph(outs[k].logits, c.draft_arrs[i], xfm.s);
                    c.n_conf = i + 1;
                }
                if (c.h_chain) |h_old| _ = mlx.mlx_array_free(h_old);
                c.h_chain = outs[k].stream;
                outs[k].stream = .{ .ctx = null };
            }
        }
    }
}
