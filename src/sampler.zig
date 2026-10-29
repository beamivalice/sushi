const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const log = @import("log.zig");
const token_mask = @import("token_mask.zig");
const io_util = @import("io_util.zig");
const mtp_mod = @import("mtp.zig");
const mtp_acceptance = @import("mtp_acceptance.zig");
const slot_admission = @import("slot_admission.zig");
const think_penalty = @import("think_penalty.zig");
const seen_penalty = @import("seen_penalty.zig");
const logit_bias = @import("logit_bias.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const dflash_mod = @import("dflash.zig");
const ThinkPenalty = think_penalty.ThinkPenalty;
const testing = std.testing;
const mtp_plan = @import("mtp_plan.zig");
const ThinkBound = mtp_plan.ThinkBound;
const generate = @import("generate.zig");
const mtp_round = @import("mtp_round.zig");
const mtp_policy = @import("mtp_policy.zig");
const CallForce = generate.CallForce;
const Constraint = generate.Constraint;
const Generator = generate.Generator;
const SchemaConstraint = generate.SchemaConstraint;
const tokenArray = generate.tokenArray;
const verifyArgmax = generate.verifyArgmax;

/// Per-token logprob info (OpenAI format).
pub const TokenLogprob = struct {
    token_id: u32,
    logprob: f32,
};

/// Logprob result for a single generated token.
pub const LogprobResult = struct {
    token_logprob: f32, // logprob of the chosen token
    top_logprobs: []TokenLogprob, // top N alternatives (caller must free)
};

pub fn isGreedyTemperature(temperature: f32) bool {
    return temperature < 0.01;
}

/// Sampling parameters for token generation.
pub const SamplingParams = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0, // 0 = disabled
    repeat_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0, // 0.0 = disabled
    /// OpenAI's: subtracted once per earlier occurrence of a token (0.0 = disabled).
    frequency_penalty: f32 = 0.0,
    seed: ?u64 = null,
    /// Draw index under `seed`: every sample takes a fresh key.
    draw: u64 = 0,
    /// When non-null, generation is constrained to outputs that satisfy the
    /// grammar at byte level. Forces a synchronous sampling path (no lazy
    /// pipeline) since grammar advancement requires the realized token id.
    constraint: ?*Constraint = null,
    /// In-stream thinking budget (`ThinkBound`), owned by the request handler
    /// like `constraint`; null = no bound.
    think_bound: ?*ThinkBound = null,
    /// Forced tool call (`CallForce`), owned by the request handler like
    /// `think_bound`; null = the model chooses whether to call.
    call_force: ?*CallForce = null,
    /// Reserved-token suppression mask: `[vocab]` bool, true = the sampler
    /// must never draw this id (reserved specials like `<|fim_hole|>`, which
    /// a degenerate distribution can rank top-5 at a collapsed position — a
    /// reserved marker in chat output is always a bug). Model-lifetime,
    /// OWNED by the Transformer (`suppress_mask`), non-owning here; wired by
    /// `Generator.initWithOptions` so every sampling path inherits it.
    /// Applied by both samplers and both stochastic-verify filters, fully
    /// lazy (`mlx_where` + -inf, no host sync); logprobs deliberately keep
    /// reading the RAW logits — the field reports the model, the mask is
    /// sampling policy. Null = no suppression (kill switch, no-template
    /// models, every non-suppressing arch).
    suppress_mask: ?mlx.mlx_array = null,
    /// Overthinking-marker penalty; the Generator's copy tracks the span over its committed ids.
    think_penalty: ThinkPenalty = .{},
};

/// Whether a request consumes only each position's argmax: greedy (or top-1), no logit-moving
/// penalty, no logprobs, no grammar or forced call choosing among ids. Only then may a
/// forward serve pruned or shortlisted logits.
pub fn argmaxOnlyRequest(sampling: SamplingParams, logprobs_n: u32) bool {
    return (isGreedyTemperature(sampling.temperature) or sampling.top_k == 1) and
        !penaltyActive(sampling) and
        !sampling.think_penalty.active() and
        sampling.constraint == null and
        sampling.call_force == null and
        logprobs_n == 0;
}

/// Build the `[vocab]` bool suppression mask (true = never sample) on the
/// host, once per model load. Caller owns the returned array.
///
/// Two disjoint sources of "never sample this row":
///   - `ids`: the reserved specials (`tokenizer.reservedOutputIds`).
///   - `[defined_vocab, vocab)`: the checkpoint's PADDING rows. A config's
///     `vocab_size` is a padded matrix dimension, not a vocabulary — the
///     tokenizer's highest defined id + 1 is. The rows in between decode to
///     nothing, so a sample from one emits nothing while consuming a step and
///     poisoning the KV; they were the one class of unsampleable row this
///     mask never covered.
/// `defined_vocab == 0` or `>= vocab` means "no padding known" and adds
/// nothing — a mask that suppressed the whole vocab is `-inf` everywhere,
/// which argmaxes to id 0 (the all-false-mask class, from the other side).
pub fn buildSuppressMask(ids: []const u32, defined_vocab: usize, vocab: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    _ = s;
    const alloc = std.heap.page_allocator;
    const buf = try alloc.alloc(bool, vocab);
    defer alloc.free(buf);
    fillSuppressMask(buf, ids, defined_vocab);
    const shape = [_]c_int{@intCast(vocab)};
    return mlx.mlx_array_new_data(buf.ptr, &shape, 1, .bool_);
}

/// The host half of `buildSuppressMask`, pure so the row set is testable
/// without a GPU. `buf.len` is the mask's vocab.
pub fn fillSuppressMask(buf: []bool, ids: []const u32, defined_vocab: usize) void {
    @memset(buf, false);
    for (ids) |id| {
        if (id < buf.len) buf[id] = true;
    }
    if (defined_vocab > 0 and defined_vocab < buf.len) {
        @memset(buf[defined_vocab..], true);
    }
}

/// Derive + install the reserved-token suppression mask on a freshly-built
/// Transformer (both the serve load path and the CLI run path call this).
/// The id set is `tokenizer.reservedOutputIds`: `special: true` added tokens
/// minus EOS/stop ids minus template-emitted markers. Never fails a load — any error logs and
/// leaves the mask null (suppression off), and engagement is one-shot-logged
/// so a silent no-op is visible (the silent-fallback class).
pub fn installSuppressMask(xfm: *Transformer, tok: *const Tokenizer, chat_template: []const u8, eos_ids: []const u32) void {
    const ids = tokenizer_mod.reservedOutputIds(xfm.allocator, tok.flagged_specials, chat_template, eos_ids) catch |err| {
        log.warn("[suppress] derivation failed ({s}); reserved-token suppression off\n", .{@errorName(err)});
        return;
    };
    defer xfm.allocator.free(ids);
    const logits_dim: usize = xfm.config.vocab_size;
    // Padding rows past the tokenizer's last defined id.
    const defined_vocab: usize = tok.definedVocabSize();
    const pad_rows: usize = if (defined_vocab > 0 and defined_vocab < logits_dim)
        logits_dim - defined_vocab
    else
        0;
    if (ids.len == 0 and pad_rows == 0) return;
    xfm.suppress_mask = buildSuppressMask(ids, defined_vocab, logits_dim, xfm.s) catch |err| {
        log.warn("[suppress] mask build failed ({s}); reserved-token suppression off\n", .{@errorName(err)});
        return;
    };
    log.info(
        "[suppress] {d} of {d} flagged specials masked from sampling (template + eos exempt); {d} padding rows past id {d} of {d}\n",
        .{ ids.len, tok.flagged_specials.len, pad_rows, defined_vocab, logits_dim },
    );
}

/// `out = where(mask, -inf, logits)` — the masked lanes get EXACTLY -inf
/// (never an additive -inf, whose 0×-inf/NaN edge the parity rules exist
/// for). `[V]` broadcasts over both `[1, V]` and `[1, 1, V]` logits.
pub fn applySuppressMask(out: *mlx.mlx_array, logits: mlx.mlx_array, mask: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(out, mask, neg_inf, logits, s));
}

/// Only the synchronous serial sampler (`sampleToken`) applies these penalties. A repeat
/// penalty of 0 or below is off: it would divide by zero or flip signs.
pub fn penaltyActive(sampling: SamplingParams) bool {
    return penaltiesOf(sampling).active();
}

pub fn penaltiesOf(sampling: SamplingParams) seen_penalty.Penalties {
    return .{ .repeat = sampling.repeat_penalty, .presence = sampling.presence_penalty, .frequency = sampling.frequency_penalty };
}

/// Sample a token lazily from logits — returns a lazy MLX array (no eval).
/// Handles temperature scaling, top-k, and top-p, but defers materialization.
/// The returned array has shape [1] with the sampled token ID.
/// Caller must free the returned array.
/// Compute the probability distribution over the vocabulary at the LAST
/// position of `logits_3d` (shape `[B, S, V]`), with the SAME temperature +
/// top-k + top-p masking the sampler would apply. Both `target_p` and `draft_q`
/// in the stochastic-verify accept test must be computed via this function so
/// the ratio `p[draft] / q[draft]` is well-defined over the kept support.
/// Caller owns the returned array; shape `[B, V]`.
/// Batched sibling of `probsAtLastPos`: temperature → top-k → top-p →
/// softmax over EVERY position of `[1, L, V]` logits in one set of
/// row-parallel kernels. A per-position loop pays L separate ~vocab-sized
/// sort/topk kernel launches per spec-decode round; batched it's one each.
/// All filter helpers operate on the last axis, so leading dims pass through.
pub fn probsAllPositions(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var current = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(current);
    try mlx.check(mlx.mlx_array_set(&current, logits_3d));

    // Reserved-token suppression (batched MTP stochastic verify) — same
    // rationale as `probsAtLastPos`.
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applySuppressMask(&masked, current, m, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    if (sampling.temperature != 1.0) {
        const t = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(t);
        var scaled = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_divide(&scaled, current, t, s));
        _ = mlx.mlx_array_free(current);
        current = scaled;
    }
    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        var masked = mlx.mlx_array_new();
        filterTopKTopP(&masked, current, sampling.top_p, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    var probs = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, current, -1, true, s));
    _ = mlx.mlx_array_free(current);
    current = .{ .ctx = null };
    return probs;
}

/// The target densities of a whole GROUP of verify rows in one filtered block:
/// the rows' `[1, Σ(1+m_i), V]` logits filtered as one. The filter is row-wise
/// on the last axis, so a row's slice of this block is byte-for-byte the
/// `probsAllPositions` of that row alone.
pub fn groupProbsBlock(block_logits: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    return probsAllPositions(block_logits, sampling, s);
}

/// One row's `[1, 1+m, V]` densities as a VIEW into the group's block.
pub fn groupProbsRow(block_probs: mlx.mlx_array, offset: c_int, len: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(block_probs);
    const start = [_]c_int{ 0, offset, 0 };
    const stop = [_]c_int{ shape[0], offset + @as(c_int, @intCast(len)), shape[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, block_probs, &start, 3, &stop, 3, &strides, 3, s));
    return out;
}

pub fn probsAtLastPos(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(logits_3d);
    const seq_len = shape[1];
    var current = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(current);
    if (seq_len == 1) {
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, logits_3d, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ shape[0], seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits_3d, &start, 3, &stop, 3, &strides, 3, s));
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s));
    }

    // Reserved-token suppression: the filtered probs feed spec-verify
    // acceptance AND the residual corrections, so a suppressed draft's
    // acceptance probability is exactly 0 and the residual can't re-draw it.
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applySuppressMask(&masked, current, m, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    const owned = current;
    current = .{ .ctx = null };
    return filteredProbsRows(owned, sampling, s);
}

/// Temperature → top-k → top-p → softmax over the LAST axis of an owned
/// `[.., V]` array (consumed). The ONE place a proposal or target density is
/// built: a draft sampled from a distribution that is not byte-for-byte the
/// `q` handed to `specAcceptProb` breaks exactness silently, so both sides
/// come through here. Row-independent, so it serves a single `[1, V]` row and
/// a whole `[m, V]` block identically.
fn filteredProbsRows(owned_rows: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var current = owned_rows;
    errdefer _ = mlx.mlx_array_free(current);
    // Apply temperature → top-k → top-p (same order as `sampleTokenLazy`).
    if (sampling.temperature != 1.0) {
        const t = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(t);
        var scaled = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_divide(&scaled, current, t, s));
        _ = mlx.mlx_array_free(current);
        current = scaled;
    }
    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        var masked = mlx.mlx_array_new();
        filterTopKTopP(&masked, current, sampling.top_p, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    // Softmax: tokens at -inf become 0, kept tokens renormalize to sum=1.
    var probs = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, current, -1, true, s));
    _ = mlx.mlx_array_free(current);
    current = .{ .ctx = null };
    return probs;
}

/// Read `probs[0, token_id]` as f32. Forces realization with a single eval.
pub fn probAt(probs: mlx.mlx_array, token_id: u32, s: mlx.mlx_stream) !f32 {
    const idx_val: i32 = @intCast(token_id);
    const idx_shape = [_]c_int{1};
    const idx_arr = mlx.mlx_array_new_data(&idx_val, &idx_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(idx_arr);

    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, probs, idx_arr, -1, s));

    // Cast to f32 so item_float32 is exact regardless of source dtype (bf16 etc.).
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, taken, .float32, s));
    try mlx.check(mlx.mlx_array_eval(as_f32));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, as_f32));
    return v;
}

/// Sample one token from probability distribution `probs` (shape `[B, V]`).
/// Returns a u32 token id (caller can append directly).
pub fn sampleFromProbs(probs: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    // mlx_random_categorical takes logits and applies softmax. Feed log(probs)
    // so the categorical's softmax recovers the original distribution.
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_log(&log_probs, probs, s));

    const null_key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(null_key);
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, log_probs, -1, null_key, s));
    try mlx.check(mlx.mlx_array_eval(sampled));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, sampled));
    return @intCast(v);
}

/// Build a one-hot float32 row vector of shape `[1, vocab]` with 1.0 at
/// `index` and 0.0 elsewhere. Used by the DFlash stochastic-verify reject path,
/// which models the draft (an n-gram lookup, not a probabilistic model) as a
/// degenerate one-hot distribution. Caller owns the returned array.
pub fn pldOneHotRow(index: u32, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var indices = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices);
    try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));

    const target_val: i32 = @intCast(index);
    const tgt_shape = [_]c_int{1};
    const target_idx = mlx.mlx_array_new_data(&target_val, &tgt_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(target_idx);

    var mask_bool = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_bool);
    try mlx.check(mlx.mlx_equal(&mask_bool, indices, target_idx, s));

    var mask_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_f32);
    try mlx.check(mlx.mlx_astype(&mask_f32, mask_bool, .float32, s));

    const out_shape = [_]c_int{ 1, vocab };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, mask_f32, &out_shape, 2, s));
    return out;
}

/// The DFlash2 selector's proposal density for one step as a `[1, V]` row:
/// its per-candidate softmax scattered into zeros — q is zero off the
/// candidate set by construction, so this is the exact density the draft was
/// drawn from (the residual-correction contract).
pub fn selectorQRow(sp: *const dflash_mod.SelectedPath, step: usize, m: u32, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const kk = sp.cand_ids.len / @as(usize, m);
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    const z_shape = [_]c_int{ 1, vocab };
    try mlx.check(mlx.mlx_zeros(&zeros, &z_shape, 2, .float32, s));
    const row_shape = [_]c_int{ 1, @intCast(kk) };
    const ids_arr = mlx.mlx_array_new_data(sp.cand_ids.ptr + step * kk, &row_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(ids_arr);
    const vals_arr = mlx.mlx_array_new_data(sp.q.?.ptr + step * kk, &row_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(vals_arr);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_put_along_axis(&out, zeros, ids_arr, vals_arr, 1, s));
    return out;
}

/// One draft depth's token and the density it was drawn from: the `[1, V]`
/// row the verify's accept ratio and residual read for that depth.
const StepProposal = struct { id: mlx.mlx_array, q: mlx.mlx_array };

/// Sampled draft from the exact re-scored top-32: q is the draft sampler's
/// filtered softmax over the shortlist, the draft is drawn from that q, and q
/// goes to the verify as a dense `[1, V]` row that is zero off the shortlist.
/// Leviathan is exact for any q, so the residual reaches the tokens the
/// shortlist missed; filtering inside 32 rows costs acceptance at most. The
/// contract: the row the verify sees is the array the draft was drawn from.
pub fn shortlistProposal(sl: mtp_mod.Shortlist, sampling: SamplingParams, s: mlx.mlx_stream) !StepProposal {
    const shape = mlx.getShape(sl.exact);
    const n = shape[shape.len - 1];

    // f32: the density `min(1, p/q)` divides by must be the one the
    // categorical drew from, and a bf16 softmax does not sum to 1.
    var rows = mlx.mlx_array_new();
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const row_shape = [_]c_int{ 1, n };
        try mlx.check(mlx.mlx_reshape(&flat, sl.exact, &row_shape, 2, s));
        errdefer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_astype(&rows, flat, .float32, s));
    }
    // The suppress mask already applied to the coarse readout; the shortlist
    // holds no reserved id.
    var draft = sampling;
    draft.suppress_mask = null;
    const q32 = try filteredProbsRows(rows, draft, s);
    errdefer _ = mlx.mlx_array_free(q32);

    var id = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(id);
    {
        const pick = try sampleFromProbsLazy(q32, sampling, s);
        defer _ = mlx.mlx_array_free(pick);
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_take_axis(&picked, sl.cands, pick, 0, s));
        try mlx.check(mlx.mlx_astype(&id, picked, .int32, s));
    }

    const q = try scatterShortlistQ(sl, q32, s);
    _ = mlx.mlx_array_free(q32);
    return .{ .id = id, .q = q };
}

/// An argmax depth of a chain that carries densities (consumes `amax`): the
/// id reshaped to the `[1]` a sampled depth drafts, since the verify
/// concatenates a chain's ids, and its one-hot q.
pub fn argmaxStepProposal(amax: mlx.mlx_array, vocab: c_int, s: mlx.mlx_stream) !StepProposal {
    defer _ = mlx.mlx_array_free(amax);
    var id = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(id);
    try mlx.check(mlx.mlx_reshape(&id, amax, &[_]c_int{1}, 1, s));
    return .{ .id = id, .q = try lazyOneHotRow(id, vocab, s) };
}

/// A depth of a chain that carries densities, off the rerank shortlist: a
/// sampling depth draws from it (`shortlistProposal`), an argmax depth (the
/// greedy tail) takes the token a greedy chain drafts.
pub fn shortlistStepProposal(sl: mtp_mod.Shortlist, sampling: SamplingParams, s: mlx.mlx_stream) !StepProposal {
    if (!isGreedyTemperature(sampling.temperature)) return shortlistProposal(sl, sampling, s);
    return argmaxStepProposal(try mtp_mod.shortlistArgmax(s, sl), sl.rows, s);
}

/// `shortlistStepProposal` off the draft head's full `[1, 1, V]` logits.
pub fn logitsStepProposal(logits: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !StepProposal {
    if (isGreedyTemperature(sampling.temperature)) {
        const shape = mlx.getShape(logits);
        return argmaxStepProposal(sampleTokenLazy(logits, sampling, s), shape[shape.len - 1], s);
    }
    const q = try probsAtLastPos(logits, sampling, s);
    errdefer _ = mlx.mlx_array_free(q);
    return .{ .id = try sampleFromProbsLazy(q, sampling, s), .q = q };
}

/// `shortlistProposal` for K rows at once: ONE filtered block over the
/// stacked `[K, 32]` logits and ONE categorical (the kernels are launch-bound,
/// so rows are nearly free and a per-row loop is not). Requires one shared
/// sampler (`sampleRowsHomogeneous`) and no seeded row: one categorical
/// cannot carry a per-row key.
pub fn shortlistProposalRows(
    shortlists: []const mtp_mod.Shortlist,
    rows: []const usize,
    sampling: SamplingParams,
    s: mlx.mlx_stream,
) !struct {
    ids: [32]mlx.mlx_array,
    qs: [32]mlx.mlx_array,
} {
    const K = rows.len;
    std.debug.assert(K > 0 and K <= 32);
    const n = mlx.getShape(shortlists[rows[0]].exact)[mlx.getShape(shortlists[rows[0]].exact).len - 1];

    // [K, 32] f32, for the same reason as the single-row proposal.
    var block = mlx.mlx_array_new();
    {
        var parts: [32]mlx.mlx_array = @splat(.{ .ctx = null });
        defer for (parts[0..K]) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        };
        for (rows, 0..) |row, j| {
            parts[j] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&parts[j], shortlists[row].exact, &[_]c_int{ 1, n }, 2, s));
        }
        const vec = mlx.mlx_vector_array_new_data(parts[0..K].ptr, K);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
        errdefer _ = mlx.mlx_array_free(block);
        try mlx.check(mlx.mlx_astype(&block, cat, .float32, s));
    }
    // The suppress mask already applied to the coarse readout.
    var draft = sampling;
    draft.suppress_mask = null;
    const q_block = try filteredProbsRows(block, draft, s);
    defer _ = mlx.mlx_array_free(q_block);

    // ONE categorical over the block -> [K] shortlist indices.
    var picks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picks);
    {
        var logq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(logq);
        try mlx.check(mlx.mlx_log(&logq, q_block, s));
        const null_key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(null_key);
        try mlx.check(mlx.mlx_random_categorical(&picks, logq, -1, null_key, s));
    }

    var ids: [32]mlx.mlx_array = @splat(.{ .ctx = null });
    var qs: [32]mlx.mlx_array = @splat(.{ .ctx = null });
    var done: usize = 0;
    errdefer for (0..done) |j| {
        if (ids[j].ctx != null) _ = mlx.mlx_array_free(ids[j]);
        if (qs[j].ctx != null) _ = mlx.mlx_array_free(qs[j]);
    };
    for (rows, 0..) |row, j| {
        const j_c: c_int = @intCast(j);
        var pick = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pick);
        try mlx.check(mlx.mlx_slice(&pick, picks, &[_]c_int{j_c}, 1, &[_]c_int{j_c + 1}, 1, &[_]c_int{1}, 1, s));
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_take_axis(&picked, shortlists[row].cands, pick, 0, s));
        ids[j] = mlx.mlx_array_new();
        done = j + 1;
        try mlx.check(mlx.mlx_astype(&ids[j], picked, .int32, s));

        var q_row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_row);
        try mlx.check(mlx.mlx_slice(&q_row, q_block, &[_]c_int{ j_c, 0 }, 2, &[_]c_int{ j_c + 1, n }, 2, &[_]c_int{ 1, 1 }, 2, s));
        qs[j] = try scatterShortlistQ(shortlists[row], q_row, s);
    }
    return .{ .ids = ids, .qs = qs };
}

/// One row's `[1, 32]` shortlist density as the dense `[1, V]` q the verify
/// takes, zero off the shortlist.
fn scatterShortlistQ(sl: mtp_mod.Shortlist, q32: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const n = mlx.getShape(q32)[mlx.getShape(q32).len - 1];
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, &[_]c_int{ 1, sl.rows }, 2, .float32, s));
    var cand_row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_row);
    try mlx.check(mlx.mlx_reshape(&cand_row, sl.cands, &[_]c_int{ 1, n }, 2, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_put_along_axis(&out, zeros, cand_row, q32, 1, s));
    return out;
}

/// A lazy one-hot `[1, V]` f32 proposal row: the q of an argmax proposal, so
/// `min(1, p/q) == p`. The sampled path's fallback when the coarse head
/// retires mid-chain.
fn lazyOneHotRow(id: mlx.mlx_array, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var indices = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices);
    try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));
    var flat_id = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_id);
    const one = [_]c_int{1};
    try mlx.check(mlx.mlx_reshape(&flat_id, id, &one, 1, s));
    var hit = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hit);
    try mlx.check(mlx.mlx_equal(&hit, indices, flat_id, s));
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, hit, .float32, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const row_shape = [_]c_int{ 1, vocab };
    try mlx.check(mlx.mlx_reshape(&out, as_f32, &row_shape, 2, s));
    return out;
}

/// Sample from the residual distribution `residual = max(target - draft, 0)`,
/// renormalized. Used on stochastic-verify reject so the corrected token
/// preserves the target distribution (per Leviathan et al. speculative
/// decoding paper).
pub fn sampleResidual(target_probs: mlx.mlx_array, draft_probs: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, target_probs, draft_probs, s));

    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var residual = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(residual);
    try mlx.check(mlx.mlx_maximum(&residual, diff, zero, s));

    return sampleFromProbs(residual, s);
}

/// Lazy categorical sample from an already-filtered probability row
/// ([1, vocab]): log(probs) puts masked tokens at -inf, categorical draws
/// within the kept set — the same distribution as sampling the filtered
/// logits directly, but the caller keeps `probs` as the proposal density q.
pub fn sampleFromProbsLazy(probs: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var logp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logp);
    try mlx.check(mlx.mlx_log(&logp, probs, s));
    var sampled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sampled);
    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_categorical(&sampled, logp, -1, key, s));
    return sampled;
}

/// `sampleTokenLazy` under the request's penalties over `generated_ids`.
pub fn penalizedSampleLazy(allocator: std.mem.Allocator, logits: mlx.mlx_array, sampling: SamplingParams, generated_ids: []const u32, s: mlx.mlx_stream) !mlx.mlx_array {
    if (penaltyActive(sampling) and generated_ids.len > 0) {
        var penalized = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(penalized);
        try applyPenaltiesWithAllocator(allocator, &penalized, logits, generated_ids, penaltiesOf(sampling), s);
        return sampleTokenLazy(penalized, sampling, s);
    }
    return sampleTokenLazy(logits, sampling, s);
}

/// Prefill's first token under the think penalty when the prompt leaves the span open.
pub fn firstTokenSample(xfm: *Transformer, logits: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const tp = sampling.think_penalty;
    if (!tp.active()) return firstTokenLazy(logits, sampling, s);
    if (!tp.hasBias() and (xfm.think_marker_mask == null or tp.phase != .inside)) return firstTokenLazy(logits, sampling, s);
    var shifted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shifted);
    try think_penalty.shiftScoped(&shifted, logits, xfm.think_marker_mask, tp, null, s);
    return firstTokenLazy(shifted, sampling, s);
}

/// Prefill's first token: a forced call's when it owns token 0, else a sample.
fn firstTokenLazy(logits: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) mlx.mlx_array {
    if (sampling.call_force) |cf| switch (cf.decide(&.{})) {
        .force => |id| return tokenArray(id),
        .sample, .choose => {},
    };
    return sampleTokenLazy(logits, sampling, s);
}

pub fn sampleTokenLazy(logits_in: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) mlx.mlx_array {
    // Reserved-token suppression first, so every path below (greedy fast
    // path included — the collapse drew `<|fim_hole|>` at temp 0) sees the
    // masked row. Lazy like everything else here. The handle starts null-ctx
    // and is only materialized when a mask exists — the fast path below
    // exists to save ONE FFI call per decode step, so the no-mask case (every
    // non-suppressing model) must not spend two on an empty array.
    var suppressed: mlx.mlx_array = .{ .ctx = null };
    defer if (suppressed.ctx != null) {
        _ = mlx.mlx_array_free(suppressed);
    };
    const logits = if (sampling.suppress_mask) |m| blk: {
        suppressed = mlx.mlx_array_new();
        applySuppressMask(&suppressed, logits_in, m, s) catch break :blk logits_in;
        break :blk suppressed;
    } else logits_in;

    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Greedy + seq_len==1 (the decode hot path): one mlx op total. argmax_axis
    // over the vocab dim of a `[1, 1, V]` tensor yields a `[1, 1]` int array,
    // which downstream (resolvePendingToken / lazyForward / async_eval vector)
    // treats identically to `[1]`. Skipping the otherwise-needed reshape +
    // argmax-on-2D combo cuts ~one FFI call per decode step.
    if (seq_len == 1 and isGreedyTemperature(sampling.temperature)) {
        var result = mlx.mlx_array_new();
        _ = mlx.mlx_argmax_axis(&result, logits, -1, false, s);
        return result;
    }

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    // `current` is the single owned intermediate — freed before each reassignment.
    var current = mlx.mlx_array_new();

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, logits, &sq_shape, 2, s);
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        _ = mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s);

        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s);
    }

    // Greedy: argmax (no temperature)
    if (isGreedyTemperature(sampling.temperature)) {
        var result = mlx.mlx_array_new();
        _ = mlx.mlx_argmax_axis(&result, current, -1, false, s);
        _ = mlx.mlx_array_free(current);
        return result;
    }

    // Scale by 1/temperature
    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        var next = mlx.mlx_array_new();
        _ = mlx.mlx_divide(&next, current, temp_arr, s);
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Apply top-k and top-p in one pass (lazy)
    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        var next = mlx.mlx_array_new();
        filterTopKTopP(&next, current, sampling.top_p, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Sample from categorical distribution (lazy — no eval!)
    var sampled = mlx.mlx_array_new();
    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    _ = mlx.mlx_random_categorical(&sampled, current, -1, key, s);
    _ = mlx.mlx_array_free(current);

    return sampled; // Shape [1], lazy
}

pub var sample_rows_evals: u64 = 0;

/// One sampling eval for every row of a batched tick. Homogeneous unseeded rows
/// draw over one stacked `[N, V]` graph (one argmax or one categorical draw);
/// a seed, mixed filters or a lone row take the per-row arm, whose parts are
/// reshaped to `[1]` before the concat because a greedy draw is `[1, 1]` and a
/// categorical draw is `[1]`. Every row advances its `draw` before the eval, so
/// a failed eval never replays a key.
pub fn sampleRows(out: []i32, row_logits: []const mlx.mlx_array, params: []SamplingParams, s: mlx.mlx_stream) !void {
    std.debug.assert(out.len == row_logits.len and out.len == params.len);
    const n = row_logits.len;
    if (n == 0) return;
    const lazy = try sampleRowsLazy(row_logits, params, s);
    defer _ = mlx.mlx_array_free(lazy);
    for (params) |*p| p.draw +%= 1;
    sample_rows_evals += 1;
    try mlx.check(mlx.mlx_array_eval(lazy));
    if (n == 1) {
        try mlx.check(mlx.mlx_array_item_int32(&out[0], lazy));
        return;
    }
    const data = mlx.mlx_array_data_uint32(lazy) orelse return error.MlxError;
    for (out, data[0..n]) |*dst, v| dst.* = @intCast(v);
}

/// Rows may share one stacked graph only when every filter matches; a seed
/// always takes the per-row arm so its key stays per row.
pub fn sampleRowsHomogeneous(params: []const SamplingParams) bool {
    const p0 = params[0];
    for (params[1..]) |p| {
        if (p.temperature != p0.temperature or p.top_k != p0.top_k or p.top_p != p0.top_p) return false;
        const m0 = p0.suppress_mask;
        const m1 = p.suppress_mask;
        if (m0 == null and m1 == null) continue;
        if (m0 == null or m1 == null) return false;
        if (m0.?.ctx != m1.?.ctx) return false;
    }
    return true;
}

fn sampleRowsAllGreedy(params: []const SamplingParams) bool {
    for (params) |p| {
        if (!isGreedyTemperature(p.temperature)) return false;
    }
    return true;
}

fn concatLogitRows(rows: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    if (rows.len == 1) {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&out, rows[0]));
        return out;
    }
    const vec = mlx.mlx_vector_array_new_data(rows.ptr, rows.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    return out;
}

fn logitsAsNV(logits: mlx.mlx_array, n: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(logits);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (sh.len == 2) {
        try mlx.check(mlx.mlx_array_set(&out, logits));
        return out;
    }
    if (sh.len != 3 or sh[0] != n) return error.BadShape;
    if (sh[1] == 1) {
        const sq = [_]c_int{ n, sh[2] };
        try mlx.check(mlx.mlx_reshape(&out, logits, &sq, 2, s));
        return out;
    }
    const start = [_]c_int{ 0, sh[1] - 1, 0 };
    const stop = [_]c_int{ n, sh[1], sh[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));
    const sq = [_]c_int{ n, sh[2] };
    try mlx.check(mlx.mlx_reshape(&out, sliced, &sq, 2, s));
    return out;
}

fn sampleRowsLazyConcat(row_logits: []const mlx.mlx_array, params: []const SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var parts: [slot_admission.MAX_BATCH_GROUP]mlx.mlx_array = undefined;
    const n = row_logits.len;
    if (n > parts.len) return error.BatchTooWide;
    var n_parts: usize = 0;
    // One release on every exit: the concat output holds its own references.
    defer {
        for (parts[0..n_parts]) |a| _ = mlx.mlx_array_free(a);
    }
    const one = [_]c_int{1};
    for (row_logits, params) |logits, p| {
        const raw = sampleTokenLazy(logits, p, s);
        defer _ = mlx.mlx_array_free(raw);
        parts[n_parts] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&parts[n_parts], raw, &one, 1, s));
        n_parts += 1;
    }
    return concatLogitRows(parts[0..n_parts], s);
}

pub fn sampleRowsAnySeed(params: []const SamplingParams) bool {
    for (params) |p| {
        if (p.seed != null) return true;
    }
    return false;
}

fn sampleRowsLazy(row_logits: []const mlx.mlx_array, params: []const SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const n: c_int = @intCast(row_logits.len);
    if (row_logits.len == 1 or !sampleRowsHomogeneous(params) or sampleRowsAnySeed(params)) {
        return sampleRowsLazyConcat(row_logits, params, s);
    }
    const stacked = try concatLogitRows(row_logits, s);
    defer _ = mlx.mlx_array_free(stacked);
    var current = try logitsAsNV(stacked, n, s);
    var owned = true;
    errdefer if (owned) {
        _ = mlx.mlx_array_free(current);
    };
    const p0 = params[0];

    var suppressed: mlx.mlx_array = .{ .ctx = null };
    defer if (suppressed.ctx != null) {
        _ = mlx.mlx_array_free(suppressed);
    };
    if (p0.suppress_mask) |m| {
        suppressed = mlx.mlx_array_new();
        applySuppressMask(&suppressed, current, m, s) catch {
            _ = mlx.mlx_array_free(suppressed);
            suppressed = .{ .ctx = null };
        };
        if (suppressed.ctx != null) {
            _ = mlx.mlx_array_free(current);
            current = suppressed;
            suppressed = .{ .ctx = null };
        }
    }

    if (sampleRowsAllGreedy(params)) {
        var result = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_argmax_axis(&result, current, -1, false, s));
        _ = mlx.mlx_array_free(current);
        owned = false;
        return result;
    }

    if (p0.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(p0.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        var next = mlx.mlx_array_new();
        mlx.check(mlx.mlx_divide(&next, current, temp_arr, s)) catch |err| {
            _ = mlx.mlx_array_free(next);
            return err;
        };
        _ = mlx.mlx_array_free(current);
        current = next;
    }
    if (p0.top_k > 0 or p0.top_p < 1.0) {
        var next = mlx.mlx_array_new();
        filterTopKTopP(&next, current, p0.top_p, p0.top_k, s) catch {
            _ = mlx.mlx_array_free(next);
            next = current;
            current = mlx.mlx_array_new();
        };
        if (current.ctx != null) _ = mlx.mlx_array_free(current);
        current = next;
    }

    const key = seedKey(p0);
    defer _ = mlx.mlx_array_free(key);
    var sampled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, key, s));
    _ = mlx.mlx_array_free(current);
    owned = false;
    return sampled;
}

/// PRNG key for one draw: `seed` mixed with the draw index, so a seeded
/// request replays byte-for-byte and consecutive draws never share a key.
/// Null-ctx (MLX global RNG) when the request set no seed.
pub fn seedKey(sampling: SamplingParams) mlx.mlx_array {
    const seed = sampling.seed orelse return mlx.mlx_array_new();
    var key = mlx.mlx_array_new();
    _ = mlx.mlx_random_key(&key, seed +% sampling.draw *% 0x9E3779B97F4A7C15);
    return key;
}

pub const SampleResult = struct {
    token_id: u32,
    logprob_result: ?LogprobResult = null,
};

/// A host-decided think penalty for one sample: lower the `mask` ids by `lambda`.
pub const MarkerShift = struct { mask: ?mlx.mlx_array, state: ThinkPenalty };

/// Sample a token from the last position's logits.
/// temperature < 0.01 (`isGreedyTemperature`): greedy argmax. Otherwise: scale logits, apply top_p, and sample.
/// If logprobs_n > 0, also computes logprobs for the sampled token and top N alternatives.
pub fn sampleToken(allocator: std.mem.Allocator, logits: mlx.mlx_array, sampling: SamplingParams, generated_ids: ?[]const u32, logprobs_n: u32, marker_shift: ?MarkerShift, s: mlx.mlx_stream) !SampleResult {
    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    var last_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last_logits);

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, logits, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));

        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, sliced, &sq_shape, 2, s));
    }

    // Track current working logits (avoid copies when no transform needed)
    var current = last_logits;

    // Reserved-token suppression — before any other transform, so the greedy
    // arm below can't draw a suppressed id. `logprobs_logits` stays
    // `last_logits`: the reported distribution is the MODEL's, so under
    // suppression rank 1 may legitimately differ from the chosen token.
    var suppressed = mlx.mlx_array_new();
    var suppressed_owned = false;
    defer if (suppressed_owned) {
        _ = mlx.mlx_array_free(suppressed);
    };
    if (sampling.suppress_mask) |m| {
        try applySuppressMask(&suppressed, current, m, s);
        current = suppressed;
        suppressed_owned = true;
    }

    var marked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(marked);
    if (marker_shift) |m| {
        try think_penalty.shiftScoped(&marked, current, m.mask, m.state, null, s);
        current = marked;
    }

    // Apply repeat penalty to already-generated tokens
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    if (penaltyActive(sampling)) {
        if (generated_ids) |ids| {
            if (ids.len > 0) {
                try applyPenalties(&penalized, current, ids, penaltiesOf(sampling), s);
                current = penalized;
                penalized_owned = true;
            }
        }
    }

    // Greedy if temperature is ~0
    if (isGreedyTemperature(sampling.temperature)) {
        const token_id = try argmax(current, s);
        var logprob_result: ?LogprobResult = null;
        if (logprobs_n > 0) {
            logprob_result = try computeLogprobs(allocator, last_logits, token_id, logprobs_n, s);
        }
        return .{ .token_id = token_id, .logprob_result = logprob_result };
    }

    // Scale logits by 1/temperature
    var scaled = mlx.mlx_array_new();
    var scaled_owned = false;
    defer if (scaled_owned) {
        _ = mlx.mlx_array_free(scaled);
    };

    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        try mlx.check(mlx.mlx_divide(&scaled, current, temp_arr, s));
        current = scaled;
        scaled_owned = true;
    }

    // Logprobs report the MODEL's distribution, so they read the position's raw
    // logits — not `current`, which by here carries the client's temperature
    // (and any penalty) and would make the same token's logprob move with a
    // sampling knob.
    const logprobs_logits = last_logits;

    // Apply top-k and top-p in one pass
    var filtered = mlx.mlx_array_new();
    var filtered_owned = false;
    defer if (filtered_owned) {
        _ = mlx.mlx_array_free(filtered);
    };

    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        try filterTopKTopP(&filtered, current, sampling.top_p, sampling.top_k, s);
        current = filtered;
        filtered_owned = true;
    }

    // Sample from categorical distribution
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);

    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, key, s));

    // Eval and extract
    try mlx.check(mlx.mlx_array_eval(sampled));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, sampled));

    const token_id: u32 = @intCast(val);

    // Compute logprobs after sampling (we now know the token_id)
    var logprob_result: ?LogprobResult = null;
    if (logprobs_n > 0) {
        logprob_result = try computeLogprobs(allocator, logprobs_logits, token_id, logprobs_n, s);
    }

    return .{ .token_id = token_id, .logprob_result = logprob_result };
}

/// Logprobs for the FIRST generated token, taken from the prefill's
/// final-position logits — the one distribution the decode loop never sees.
/// `chosen` is the id the lazy sampler actually drew, so the reported
/// `token_logprob` belongs to the token that was really emitted (re-sampling
/// here would disagree with it at any temperature above 0).
pub fn firstTokenLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen: u32, logprobs_n: u32, s: mlx.mlx_stream) !?LogprobResult {
    if (logprobs_n == 0) return null;
    const shape = mlx.getShape(logits);
    if (shape.len != 3) return null;
    var last = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last);
    const sq_shape = [_]c_int{ 1, shape[2] };
    if (shape[1] == 1) {
        try mlx.check(mlx.mlx_reshape(&last, logits, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, shape[1] - 1, 0 };
        const stop = [_]c_int{ 1, shape[1], shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));
        try mlx.check(mlx.mlx_reshape(&last, sliced, &sq_shape, 2, s));
    }
    return try computeLogprobs(allocator, last, chosen, logprobs_n, s);
}

/// Compute log-probabilities from logits. Returns the logprob of the chosen token
/// and the top N alternatives with their token IDs and logprobs.
///
/// `logits` are the MODEL's logits for the position — before temperature,
/// penalties and top-k/top-p — so the reported distribution is the model's, as
/// OpenAI's is. Reading them after temperature made every value move with a
/// knob the client set (and at temp 0 the distribution SATURATES, so most
/// entries report exactly 0.0).
///
/// Ids travel WITH their values through `mlx_argpartition_axis` + a gather.
/// Recovering them afterwards by scanning the vocab for float equality — what
/// this did — is ambiguous the moment two logits tie, which under the
/// saturation above is everywhere: rank 1 was measured to be the chosen token
/// in 0 of 5 positions on a trivial greedy prompt.
pub fn computeLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen_token: u32, top_n: u32, s: mlx.mlx_stream) !LogprobResult {
    // log_softmax in f32: `log(softmax(x))` in bf16 lands on bf16's grid (0.125 apart
    // between -16 and -32), and f16 underflows to -inf.
    var logits32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits32);
    try mlx.check(mlx.mlx_astype(&logits32, logits, .float32, s));
    var lse = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lse);
    try mlx.check(mlx.mlx_logsumexp_axis(&lse, logits32, -1, true, s));
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_subtract(&log_probs, logits32, lse, s));

    const lp_shape = mlx.getShape(log_probs);
    const rank = lp_shape.len;
    const vocab_size: usize = @intCast(lp_shape[rank - 1]);
    const k: usize = @min(@as(usize, @min(top_n, 20)), vocab_size);

    // Top-k INDICES, carried alongside their values. Negating turns "k largest"
    // into the "k smallest" that argpartition puts in the leading slots.
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, log_probs, s));

    var part_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part_idx);
    try mlx.check(mlx.mlx_argpartition_axis(&part_idx, neg, @intCast(if (k == 0) 0 else k - 1), -1, s));

    var start_buf: [8]c_int = @splat(0);
    var stop_buf: [8]c_int = @splat(1);
    var stride_buf: [8]c_int = @splat(1);
    for (0..rank) |i| stop_buf[i] = lp_shape[i];
    stop_buf[rank - 1] = @intCast(k);

    var idx_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx_k);
    try mlx.check(mlx.mlx_slice(&idx_k, part_idx, &start_buf, rank, &stop_buf, rank, &stride_buf, rank, s));

    var vals_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals_raw);
    try mlx.check(mlx.mlx_take_axis(&vals_raw, log_probs, idx_k, @intCast(rank - 1), s));

    var vals_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals_k);
    try mlx.check(mlx.mlx_astype(&vals_k, vals_raw, .float32, s));

    var ids_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_k);
    try mlx.check(mlx.mlx_astype(&ids_k, idx_k, .int32, s));

    try mlx.check(mlx.mlx_array_eval(log_probs));
    try mlx.check(mlx.mlx_array_eval(vals_k));
    try mlx.check(mlx.mlx_array_eval(ids_k));

    // Read the logprob of the chosen token from the full array
    const lp_data = mlx.mlx_array_data_float32(log_probs);
    const chosen_logprob: f32 = if (lp_data) |ptr|
        (if (chosen_token < vocab_size) ptr[chosen_token] else -100.0)
    else
        -100.0;

    const val_ptr = mlx.mlx_array_data_float32(vals_k);
    const id_ptr = mlx.mlx_array_data_int32(ids_k);

    var top_logprobs = try allocator.alloc(TokenLogprob, k);
    errdefer allocator.free(top_logprobs);
    var filled: usize = 0;
    if (val_ptr) |vp| {
        if (id_ptr) |ip| {
            for (0..k) |i| {
                const tid = ip[i];
                if (tid < 0 or @as(usize, @intCast(tid)) >= vocab_size) continue;
                top_logprobs[filled] = .{ .token_id = @intCast(tid), .logprob = vp[i] };
                filled += 1;
            }
        }
    }
    // argpartition leaves the winners UNORDERED; ties break on the lower id so
    // the ranking is deterministic run to run.
    std.mem.sort(TokenLogprob, top_logprobs[0..filled], {}, struct {
        fn lt(_: void, a: TokenLogprob, b: TokenLogprob) bool {
            if (a.logprob != b.logprob) return a.logprob > b.logprob;
            return a.token_id < b.token_id;
        }
    }.lt);

    if (filled < top_logprobs.len) {
        top_logprobs = allocator.realloc(top_logprobs, filled) catch top_logprobs;
    }

    return .{
        .token_logprob = chosen_logprob,
        .top_logprobs = top_logprobs,
    };
}

/// Apply a grammar token mask to logits. `mask[i]==true` keeps `logits[i]`,
/// `false` replaces it with `-inf`. The mask is broadcast over leading dims so
/// `logits` can be either `[1, vocab]` or `[1, 1, vocab]`.
pub fn applyGrammarMask(allocator: std.mem.Allocator, res: *mlx.mlx_array, logits: mlx.mlx_array, mask: []const bool, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);
    const logit_mask = try maskForLogitVocab(allocator, mask, vocab_size);
    defer logit_mask.deinit(allocator);

    // Zig's `bool` is one byte and matches MLX's `.bool_` storage exactly.
    const arr_shape = [_]c_int{@intCast(vocab_size)};
    const mask_arr = mlx.mlx_array_new_data(@ptrCast(logit_mask.slice.ptr), &arr_shape, 1, .bool_);
    defer _ = mlx.mlx_array_free(mask_arr);

    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);

    try mlx.check(mlx.mlx_where(res, mask_arr, logits, neg_inf, s));
}

/// `logits` `[1, rows, vocab]` with each row's forbidden tokens at -inf; `masks` is `rows * vocab` bools.
pub fn applyRowMasks(res: *mlx.mlx_array, logits: mlx.mlx_array, masks: []const bool, rows: usize, vocab: usize, s: mlx.mlx_stream) !void {
    const shape = [_]c_int{ 1, @intCast(rows), @intCast(vocab) };
    const mask_arr = mlx.mlx_array_new_data(@ptrCast(masks.ptr), &shape, 3, .bool_);
    defer _ = mlx.mlx_array_free(mask_arr);
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, mask_arr, logits, neg_inf, s));
}

const LogitMaskView = struct {
    slice: []const bool,
    owned: ?[]bool = null,

    fn deinit(self: LogitMaskView, allocator: std.mem.Allocator) void {
        if (self.owned) |buf| allocator.free(buf);
    }
};

fn maskForLogitVocab(allocator: std.mem.Allocator, mask: []const bool, vocab_size: usize) !LogitMaskView {
    if (mask.len == vocab_size) return .{ .slice = mask };

    var adjusted = try allocator.alloc(bool, vocab_size);
    @memset(adjusted, false);
    const copy_len = @min(mask.len, vocab_size);
    @memcpy(adjusted[0..copy_len], mask[0..copy_len]);
    return .{ .slice = adjusted, .owned = adjusted };
}

/// Rank cap for the filter helpers. The widest shape a caller passes is
/// `[B, S, V]`; anything past this is a named error, never a wrong axis.
const SAMPLER_MAX_NDIM = 4;

/// `a[..., start..stop]`, every leading axis passed through whole.
fn sliceLastAxis(res: *mlx.mlx_array, a: mlx.mlx_array, start: usize, stop: usize, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(a);
    if (shape.len > SAMPLER_MAX_NDIM) return error.UnsupportedSamplerRank;
    var lo: [SAMPLER_MAX_NDIM]c_int = undefined;
    var hi: [SAMPLER_MAX_NDIM]c_int = undefined;
    var st: [SAMPLER_MAX_NDIM]c_int = undefined;
    for (shape, 0..) |dim, i| {
        const last = i == shape.len - 1;
        lo[i] = if (last) @intCast(start) else 0;
        hi[i] = if (last) @intCast(stop) else dim;
        st[i] = 1;
    }
    const n = shape.len;
    try mlx.check(mlx.mlx_slice(res, a, &lo, n, &hi, n, &st, n, s));
}

/// Column ids of `a` ranked by value DESCENDING, ties by LOWEST ORIGINAL INDEX.
///
/// mlx's merge sort is stable — `ThreadSort` swaps only on a strict less-than and
/// the merge takes from B only when `b < a` — so an ascending argsort of the
/// negated row is exactly that order, and it is the `sushi_qsa_select` tie rule.
/// Both filters and both of their routes rank through here, which is what makes a
/// shortlist cutoff agree with the whole row's on a tie-heavy row. Pinned by
/// `sampler ranks break ties by the lowest original index`.
fn ranksDescending(res: *mlx.mlx_array, a: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, a, s));
    try mlx.check(mlx.mlx_argsort_axis(res, neg, -1, s));
}

/// Candidate-chunk width for `topRanksDescending`: cost is one sort of
/// `V / chunk` chunk maxima plus one of `m * chunk` candidates, balanced at
/// sqrt(V / m).
fn shortlistChunkWidth(v: usize, m: usize) usize {
    const bal = @sqrt(@as(f64, @floatFromInt(v)) / @as(f64, @floatFromInt(@max(m, 1))));
    const c: usize = @intFromFloat(@max(16.0, @min(2048.0, bal)));
    return @min(c, v);
}

/// The top `m` of every row — values and their column ids, both in
/// `ranksDescending` order — without ranking the whole row.
///
/// Every element of a chunk is at most that chunk's max, so a chunk whose max is
/// not among the `m` largest chunk maxima holds no top-`m` element: at most `m`
/// chunks can, and ranking their `m * chunk` candidates is exact. On Metal a
/// full-row `mlx_argsort_axis` is a multi-block merge sort over the whole
/// vocabulary — and so are `mlx_topk` and `mlx_argpartition`, which mlx routes to
/// that same sort, so neither of those buys a shortlist.
fn topRanksDescending(
    vals: *mlx.mlx_array,
    ids: *mlx.mlx_array,
    logits: mlx.mlx_array,
    m: u32,
    s: mlx.mlx_stream,
) !void {
    const shape = mlx.getShape(logits);
    if (shape.len == 0 or shape.len > SAMPLER_MAX_NDIM) return error.UnsupportedSamplerRank;
    const v: usize = @intCast(shape[shape.len - 1]);
    const want: usize = @min(@as(usize, m), v);
    if (want == 0) return error.EmptyShortlist;
    const chunk = shortlistChunkWidth(v, want);
    const groups = (v + chunk - 1) / chunk;

    // Nothing left to win once the candidate set is the row itself.
    if (want >= v or want * chunk >= v) {
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try ranksDescending(&order, logits, s);
        try sliceLastAxis(ids, order, 0, want, s);
        return mlx.check(mlx.mlx_take_along_axis(vals, logits, ids.*, -1, s));
    }

    // Pad the last chunk with -inf so the reshape is exact.
    var padded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(padded);
    if (groups * chunk != v) {
        const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg_inf);
        var pad_val = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pad_val);
        try mlx.check(mlx.mlx_astype(&pad_val, neg_inf, mlx.mlx_array_dtype(logits), s));
        const axes = [_]c_int{@intCast(shape.len - 1)};
        const low = [_]c_int{0};
        const high = [_]c_int{@intCast(groups * chunk - v)};
        try mlx.check(mlx.mlx_pad(&padded, logits, &axes, 1, &low, 1, &high, 1, pad_val, "constant", s));
    } else {
        try mlx.check(mlx.mlx_array_set(&padded, logits));
    }

    var dims: [SAMPLER_MAX_NDIM + 1]c_int = undefined;
    @memcpy(dims[0 .. shape.len - 1], shape[0 .. shape.len - 1]);
    const lead = shape.len - 1;
    dims[lead] = @intCast(groups);
    dims[lead + 1] = @intCast(chunk);
    var blk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk);
    try mlx.check(mlx.mlx_reshape(&blk, padded, &dims, lead + 2, s));

    var chunk_max = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(chunk_max);
    try mlx.check(mlx.mlx_max_axis(&chunk_max, blk, -1, false, s));

    // Chunks ranked by max, ties by lowest chunk id. A tie group's contributing
    // chunks are a PREFIX of its ascending-id list — a lower id holds a lower
    // column id, which outranks — so taking the lowest ids is what keeps the
    // shortlist equal to the whole row's ranking.
    var chunk_ranks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(chunk_ranks);
    try ranksDescending(&chunk_ranks, chunk_max, s);
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try sliceLastAxis(&picked, chunk_ranks, 0, want, s);
    // Ascending chunk ids lay the candidates out in ascending COLUMN id, so the
    // stable rank below breaks candidate ties the way the whole row would.
    var win = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(win);
    try mlx.check(mlx.mlx_sort_axis(&win, picked, -1, s));
    var win_i32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(win_i32);
    try mlx.check(mlx.mlx_astype(&win_i32, win, .int32, s));

    var win_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(win_col);
    try mlx.check(mlx.mlx_expand_dims(&win_col, win_i32, -1, s));
    dims[lead] = @intCast(want);
    var gather_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gather_idx);
    try mlx.check(mlx.mlx_broadcast_to(&gather_idx, win_col, &dims, lead + 2, s));
    var cand_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_2d);
    try mlx.check(mlx.mlx_take_along_axis(&cand_2d, blk, gather_idx, @intCast(lead), s));

    // Column id of every candidate: chunk id * chunk + offset within the chunk.
    const chunk_arr = mlx.mlx_array_new_int(@intCast(chunk));
    defer _ = mlx.mlx_array_free(chunk_arr);
    var base = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(base);
    try mlx.check(mlx.mlx_multiply(&base, win_col, chunk_arr, s));
    var offsets = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(offsets);
    try mlx.check(mlx.mlx_arange(&offsets, 0, @floatFromInt(chunk), 1, .int32, s));
    var cand_ids_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ids_2d);
    try mlx.check(mlx.mlx_add(&cand_ids_2d, base, offsets, s));

    dims[lead] = @intCast(want * chunk);
    var cand_vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_vals);
    try mlx.check(mlx.mlx_reshape(&cand_vals, cand_2d, &dims, lead + 1, s));
    var cand_ids_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ids_flat);
    try mlx.check(mlx.mlx_reshape(&cand_ids_flat, cand_ids_2d, &dims, lead + 1, s));
    // A padded tail column is -inf, so it only ranks when the row holds fewer
    // than `want` finite values; clamp so the scatter stays in range regardless.
    const last_col = mlx.mlx_array_new_int(@intCast(v - 1));
    defer _ = mlx.mlx_array_free(last_col);
    var cand_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ids);
    try mlx.check(mlx.mlx_minimum(&cand_ids, cand_ids_flat, last_col, s));

    var cand_ranks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ranks);
    try ranksDescending(&cand_ranks, cand_vals, s);
    var sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel);
    try sliceLastAxis(&sel, cand_ranks, 0, want, s);
    try mlx.check(mlx.mlx_take_along_axis(vals, cand_vals, sel, -1, s));
    try mlx.check(mlx.mlx_take_along_axis(ids, cand_ids, sel, -1, s));
}

/// Apply top-k filtering: keep only the top k logits, set the rest to -inf.
fn applyTopK(res: *mlx.mlx_array, logits: mlx.mlx_array, k: u32, s: mlx.mlx_stream) !void {
    // Per-ROW top-k by RANK: the k highest-ranked column ids, scattered back as a
    // keep mask. A value cutoff (`logits >= kth largest`) keeps every token tied
    // with the k-th, and bf16 logits tie at the top constantly on a 250k vocab —
    // top_k 1 stopped being greedy.
    const shape = mlx.getShape(logits);
    const v: usize = @intCast(shape[shape.len - 1]);
    if (k == 0 or k >= v) return mlx.check(mlx.mlx_array_set(res, logits));
    var vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try topRanksDescending(&vals, &ids, logits, k, s);
    try keepByIndex(res, logits, ids, s);
}

/// Mask `logits` to -inf everywhere except the per-row `keep_idx` columns.
fn keepByIndex(res: *mlx.mlx_array, logits: mlx.mlx_array, keep_idx: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, shape.ptr, shape.len, .bool_, s));
    const t = mlx.mlx_array_new_bool(true);
    defer _ = mlx.mlx_array_free(t);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_put_along_axis(&mask, zeros, keep_idx, t, -1, s));
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, mask, logits, neg_inf, s));
}

/// Top-k and top-p in ONE pass. A bounded `applyTopP` ranks the row's top
/// `top_k` columns and scatters its keep flags at exactly those column ids, so a
/// preceding `applyTopK` selects the same shortlist and masks a SUPERSET of what
/// the nucleus keeps — one full-vocab selection and one full-row `where` of pure
/// tax on every filtered row. Byte-identical to the pair either way round.
fn filterTopKTopP(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, top_k: u32, s: mlx.mlx_stream) !void {
    if (top_p >= 1.0) return applyTopK(res, logits, top_k, s);
    return applyTopP(res, logits, top_p, top_k, s);
}

/// Apply top-p (nucleus) sampling: mask logits outside the top-p probability
/// mass. The nucleus is decided over ranked values and scattered back by column
/// id, so a token tied with the cutoff value but outside the mass is masked
/// (top_p -> 0 is greedy even on tied bf16 logits).
///
/// `nucleus_bound` is the caller's top-k. With top-k applied first the row holds
/// exactly `k` finite columns, so the nucleus cannot reach past the k-shortlist
/// and the cutoff is exact without ranking the row; 0 means the row is unfiltered
/// and the nucleus is unbounded.
fn applyTopP(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, nucleus_bound: u32, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const v: usize = @intCast(shape[shape.len - 1]);
    const bounded = nucleus_bound > 0 and nucleus_bound < v;

    var vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    if (bounded) {
        try topRanksDescending(&vals, &ids, logits, nucleus_bound, s);
    } else {
        try ranksDescending(&ids, logits, s);
        try mlx.check(mlx.mlx_take_along_axis(&vals, logits, ids, -1, s));
    }

    // f32 throughout: mlx instantiates cumsum at the INPUT dtype and a bf16
    // accumulator loses nucleus tail mass. The ranked values ARE the row's
    // finite entries (top-k left exactly k), so their softmax is the row's.
    var vals_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals_f32);
    try mlx.check(mlx.mlx_astype(&vals_f32, vals, .float32, s));
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, vals_f32, -1, true, s));

    // Mass STRICTLY above each rank — an exclusive scan down the ranking, so the
    // term is a function of the shortlist alone (everything outranking a top-k
    // column is itself top-k). Rank 0 sees 0, so the argmax is always kept —
    // the threshold floors above 0 so a literal top_p 0 is greedy, not empty.
    var above = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(above);
    try mlx.check(mlx.mlx_cumsum(&above, probs, -1, false, false, s));
    const threshold = mlx.mlx_array_new_float(@max(top_p, std.math.floatMin(f32)));
    defer _ = mlx.mlx_array_free(threshold);
    var in_nucleus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(in_nucleus);
    try mlx.check(mlx.mlx_less(&in_nucleus, above, threshold, s));

    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, shape.ptr, shape.len, .bool_, s));
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_put_along_axis(&mask, zeros, ids, in_nucleus, -1, s));
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, mask, logits, neg_inf, s));
}

/// Penalties over the already-generated tokens. Pure MLX GPU ops after the host tally: no readback,
/// the lazy graph stays intact.
fn applyPenalties(res: *mlx.mlx_array, logits: mlx.mlx_array, generated_ids: []const u32, pen: seen_penalty.Penalties, s: mlx.mlx_stream) !void {
    return applyPenaltiesWithAllocator(std.heap.page_allocator, res, logits, generated_ids, pen, s);
}

fn applyPenaltiesWithAllocator(allocator: std.mem.Allocator, res: *mlx.mlx_array, logits: mlx.mlx_array, generated_ids: []const u32, pen: seen_penalty.Penalties, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const counts = try seen_penalty.buildCounts(allocator, generated_ids, @intCast(shape[shape.len - 1]));
    defer _ = mlx.mlx_array_free(counts);
    try seen_penalty.apply(res, logits, counts, pen, s);
}

/// Greedy argmax over the last axis.
fn argmax(last_logits: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    var argmax_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(argmax_arr);
    try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, last_logits, -1, false, s));

    try mlx.check(mlx.mlx_array_eval(argmax_arr));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, argmax_arr));

    return @intCast(val);
}

test "SamplingParams defaults" {
    const params = SamplingParams{};
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 0), params.top_k);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.repeat_penalty, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), params.presence_penalty, 0.001);
    try testing.expect(params.seed == null);
}

test "SamplingParams custom values" {
    const params = SamplingParams{
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
        .repeat_penalty = 1.1,
        .presence_penalty = 0.5,
        .seed = 42,
    };
    try testing.expectApproxEqAbs(@as(f32, 0.7), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.9), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 40), params.top_k);
    try testing.expectEqual(@as(u64, 42), params.seed.?);
}

test "penalty rows: every verify row equals its serial tick over the committed ids and the round's ids through it" {
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 40;
    const Case = struct { committed: []const u32, ids: []const i32 };
    const cases = [_]Case{
        .{ .committed = &.{ 3, 7, 7, 12 }, .ids = &.{ 7, 5, 5, 31, 3 } },
        .{ .committed = &.{}, .ids = &.{ 9, 9, 2 } },
        .{ .committed = &.{ 0, 39 }, .ids = &.{ 20 } },
    };
    const Pen = seen_penalty.Penalties;
    const pens = [_]Pen{
        .{ .repeat = 1.3 },
        .{ .presence = 1.5 },
        .{ .repeat = 1.2, .presence = 0.7 },
        .{ .repeat = 0.8, .presence = 0.25 },
        .{ .frequency = 0.6 },
        .{ .repeat = 1.2, .presence = 0.3, .frequency = 0.45 },
    };
    for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dtype| {
        for (cases) |c| {
            const rows = c.ids.len;
            var host: [8 * v]f32 = undefined;
            for (host[0 .. rows * v], 0..) |*x, i| x.* = @as(f32, @floatFromInt((i * 37) % 23)) * 0.375 - 4.0;
            const shape = [_]c_int{ 1, @intCast(rows), @intCast(v) };
            const raw = mlx.mlx_array_new_data(&host, &shape, 3, .float32);
            defer _ = mlx.mlx_array_free(raw);
            var block = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(block);
            try mlx.check(mlx.mlx_astype(&block, raw, dtype, s));
            const id_shape = [_]c_int{ 1, @intCast(rows) };
            const ids = mlx.mlx_array_new_data(c.ids.ptr, &id_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(ids);
            const committed = try seen_penalty.buildCounts(a, c.committed, v);
            defer _ = mlx.mlx_array_free(committed);
            for (pens) |p| {
                var shifted = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(shifted);
                try seen_penalty.shiftRows(&shifted, block, ids, committed, p, s);
                const got = try samplerTestReadFlat(a, shifted, rows * v, s);
                defer a.free(got);
                for (0..rows) |j| {
                    var history: [16]u32 = undefined;
                    @memcpy(history[0..c.committed.len], c.committed);
                    for (c.ids[0 .. j + 1], 0..) |id, k| history[c.committed.len + k] = @intCast(id);
                    var row = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(row);
                    try mlx.check(mlx.mlx_slice(&row, block, &.{ 0, @intCast(j), 0 }, 3, &.{ 1, @as(c_int, @intCast(j)) + 1, @intCast(v) }, 3, &.{ 1, 1, 1 }, 3, s));
                    var serial = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(serial);
                    try applyPenalties(&serial, row, history[0 .. c.committed.len + j + 1], p, s);
                    const want = try samplerTestReadFlat(a, serial, v, s);
                    defer a.free(want);
                    try testing.expectEqualSlices(f32, want, got[j * v ..][0..v]);
                }
            }
        }
    }
}

test "a repeat penalty of 0 or below is off" {
    try testing.expect(!penaltyActive(.{ .repeat_penalty = 0 }));
    try testing.expect(!penaltyActive(.{ .repeat_penalty = -1.5 }));
    try testing.expect(penaltyActive(.{ .repeat_penalty = 0, .presence_penalty = 0.3 }));
    try testing.expect(!argmaxOnlyRequest(.{ .temperature = 0, .repeat_penalty = 1.1 }, 0));
    try testing.expect(argmaxOnlyRequest(.{ .temperature = 0, .repeat_penalty = 0 }, 0));
}

test "a constrained penalised step picks what sampleToken picks on the same masked logits, penalised once" {
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const v = 16;
    // Once: id 0 falls below id 1 in case A and stays above it in case B; twice would flip B too.
    const Case = struct { runner_up: f32, want: u32 };
    for ([_]Case{ .{ .runner_up = 4.8, .want = 1 }, .{ .runner_up = 4.2, .want = 0 } }) |c| {
        var host: [v]f32 = @splat(0);
        host[0] = 5.0;
        host[1] = c.runner_up;
        host[9] = 9.0;
        const shape = [_]c_int{ 1, 1, v };
        const logits = mlx.mlx_array_new_data(&host, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(logits);
        var allowed: [v]bool = @splat(true);
        allowed[9] = false;
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        try applyGrammarMask(a, &masked, logits, &allowed, s);
        const sampling = SamplingParams{ .temperature = 0, .presence_penalty = 0.5 };
        const generated = [_]u32{ 0, 3 };
        const sync = try sampleToken(a, masked, sampling, &generated, 0, null, s);
        const lazy = try penalizedSampleLazy(std.heap.page_allocator, masked, sampling, &generated, s);
        defer _ = mlx.mlx_array_free(lazy);
        const got = try samplerTestReadFlat(a, lazy, 1, s);
        defer a.free(got);
        try testing.expectEqual(c.want, sync.token_id);
        try testing.expectEqual(@as(f32, @floatFromInt(c.want)), got[0]);
    }
}

test "maskForLogitVocab pads and truncates to logits size" {
    const short_mask = [_]bool{ true, false, true };
    const padded = try maskForLogitVocab(testing.allocator, &short_mask, 5);
    defer padded.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 5), padded.slice.len);
    try testing.expect(padded.slice[0]);
    try testing.expect(!padded.slice[1]);
    try testing.expect(padded.slice[2]);
    try testing.expect(!padded.slice[3]);
    try testing.expect(!padded.slice[4]);

    const long_mask = [_]bool{ false, true, true, true };
    const truncated = try maskForLogitVocab(testing.allocator, &long_mask, 2);
    defer truncated.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), truncated.slice.len);
    try testing.expect(!truncated.slice[0]);
    try testing.expect(truncated.slice[1]);
}

test "argmax selects highest value" {
    // Create a simple logits array [1, 5] with values [0.1, 0.5, 0.9, 0.2, 0.3]
    const data = [_]f32{ 0.1, 0.5, 0.9, 0.2, 0.3 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 2), result); // index 2 has value 0.9
}

test "argmax with negative values" {
    const data = [_]f32{ -5.0, -1.0, -3.0, -0.5, -2.0 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 3), result); // index 3 has value -0.5 (highest)
}

test "applyPenalties reduces seen token logits" {
    const s = mlx.gpuStream();
    // logits: [1.0, 2.0, 3.0, 4.0, 5.0]
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const shape = [_]c_int{ 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize tokens at indices 1 and 3
    const generated = [_]u32{ 1, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyPenalties(&res, logits, &generated, .{ .repeat = 2.0 }, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: untouched → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[1], 0.01);
    // Index 2: untouched → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
    // Index 4: untouched → 5.0
    try testing.expectApproxEqAbs(@as(f32, 5.0), res_data[4], 0.01);
}

test "applyPenalties with negative logits" {
    const s = mlx.gpuStream();
    // Mix of positive and negative logits
    const data = [_]f32{ -2.0, 3.0, -1.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize all tokens
    const generated = [_]u32{ 0, 1, 2, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyPenalties(&res, logits, &generated, .{ .repeat = 2.0 }, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: negative, multiplied by 2.0 → -4.0
    try testing.expectApproxEqAbs(@as(f32, -4.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.5
    try testing.expectApproxEqAbs(@as(f32, 1.5), res_data[1], 0.01);
    // Index 2: negative, multiplied by 2.0 → -2.0
    try testing.expectApproxEqAbs(@as(f32, -2.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
}

test "applyPenalties presence penalty" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 2 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyPenalties(&res, logits, &generated, .{ .presence = 0.5 }, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: seen, presence penalty subtracted → 1.0 - 0.5 = 0.5
    try testing.expectApproxEqAbs(@as(f32, 0.5), res_data[0], 0.01);
    // Index 1: unseen → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[1], 0.01);
    // Index 2: seen → 3.0 - 0.5 = 2.5
    try testing.expectApproxEqAbs(@as(f32, 2.5), res_data[2], 0.01);
    // Index 3: unseen → 4.0
    try testing.expectApproxEqAbs(@as(f32, 4.0), res_data[3], 0.01);
}

test "frequency_penalty subtracts per occurrence, presence_penalty once, and neither scales a logit" {
    const s = mlx.gpuStream();
    var data = [_]f32{ 2, 2, 2, 2, 2, -2, 2, 2 };
    const logits = mlx.mlx_array_new_data(&data, &[_]c_int{ 1, 8 }, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try applyPenalties(&out, logits, &.{ 3, 3, 5 }, .{ .frequency = 0.5, .presence = 0.25 }, s);
    const got = try samplerTestReadFlat(testing.allocator, out, 8, s);
    defer testing.allocator.free(got);
    try testing.expectEqualSlices(f32, &.{ 2, 2, 2, 0.75, 2, -2.75, 2, 2 }, got);
    try testing.expect(penaltyActive(.{ .frequency_penalty = 0.5 }));
    try testing.expect(!penaltyActive(.{ .frequency_penalty = 0 }));
}

test "the serial and lazy samplers count every occurrence under frequency_penalty" {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var host: [4]f32 = .{ 5.0, 4.8, 0, 0 };
    const logits = mlx.mlx_array_new_data(&host, &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const generated = [_]u32{ 0, 0, 0 };
    // Three occurrences cost 0.3 (id 1 wins); a once-per-id presence of 0.1 would not.
    const Case = struct { sampling: SamplingParams, want: u32 };
    for ([_]Case{
        .{ .sampling = .{ .temperature = 0, .frequency_penalty = 0.1 }, .want = 1 },
        .{ .sampling = .{ .temperature = 0, .presence_penalty = 0.1 }, .want = 0 },
    }) |c| {
        const sync = try sampleToken(a, logits, c.sampling, &generated, 0, null, s);
        const lazy = try penalizedSampleLazy(std.heap.page_allocator, logits, c.sampling, &generated, s);
        defer _ = mlx.mlx_array_free(lazy);
        const got = try samplerTestReadFlat(a, lazy, 1, s);
        defer a.free(got);
        try testing.expectEqual(c.want, sync.token_id);
        try testing.expectEqual(@as(f32, @floatFromInt(c.want)), got[0]);
    }
}

test "applyPenalties combined penalties" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 2.0, -1.0, 3.0 };
    const shape = [_]c_int{ 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 1 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyPenalties(&res, logits, &generated, .{ .repeat = 2.0, .presence = 1.0 }, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: positive, divide by 2.0 = 1.0, then - 1.0 = 0.0
    try testing.expectApproxEqAbs(@as(f32, 0.0), res_data[0], 0.01);
    // Index 1: negative, multiply by 2.0 = -2.0, then - 1.0 = -3.0
    try testing.expectApproxEqAbs(@as(f32, -3.0), res_data[1], 0.01);
    // Index 2: unseen → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
}

test "sampleToken greedy selects argmax" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // Create logits [1, 1, 5] — 5 vocab entries, token at index 3 has highest
    const data = [_]f32{ 1.0, 0.5, 0.1, 5.0, 0.2 };
    const logits_shape = [_]c_int{ 1, 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, null, s);
    try testing.expectEqual(@as(u32, 3), result.token_id);
}

test "sampleToken with temperature produces valid token" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    const data = [_]f32{ 1.0, 2.0, 3.0 };
    const logits_shape = [_]c_int{ 1, 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.5 };
    const result = try sampleToken(allocator, logits, params, null, 0, null, s);
    // Token should be in valid range
    try testing.expect(result.token_id < 3);
}

test "seeded lazy sampling replays the same draws and advances per draw" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 0.0, 0.0 };
    const shape = [_]c_int{ 1, 1, 2 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    var runs: [2][24]i32 = undefined;
    for (&runs) |*run| {
        var params = SamplingParams{ .temperature = 1.0, .seed = 42 };
        for (run) |*out| {
            const lazy = sampleTokenLazy(logits, params, s);
            defer _ = mlx.mlx_array_free(lazy);
            try mlx.check(mlx.mlx_array_eval(lazy));
            try mlx.check(mlx.mlx_array_item_int32(out, lazy));
            params.draw +%= 1;
        }
    }
    try testing.expectEqualSlices(i32, &runs[0], &runs[1]);
    // The same key every draw replays ONE coin flip 24 times (p = 2^-23).
    var all_same = true;
    for (runs[0][1..]) |v| all_same = all_same and v == runs[0][0];
    try testing.expect(!all_same);
}

test "batched greedy ids equal two solo ids and one sampling eval" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 0.5, 0.6, 9.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const greedy = SamplingParams{ .temperature = 0.0 };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, greedy, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, greedy, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ greedy, greedy };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
}

test "batched mixed greedy and sampled rows produce two ids" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 0.5, 0.6, 9.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const greedy = SamplingParams{ .temperature = 0.0 };
    const sampled = SamplingParams{ .temperature = 0.7 };
    const solo_g = try evalLazyToken(sampleTokenLazy(r0, greedy, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ greedy, sampled };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo_g)), ids[0]);
    try testing.expect(ids[1] >= 0 and ids[1] < 4);
}

test "batched stacked seedless top_k top_p rows equal solo draws" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 0.5, 0.6, 9.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const p = SamplingParams{ .temperature = 0.7, .top_k = 1, .top_p = 0.9 };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, p, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, p, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ p, p };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
}

test "top-k and top-p break exact ties by rank, not by value" {
    // bf16 logits tie at the top constantly on a 250k vocab; a value cutoff
    // keeps every tied token and top_k 1 / top_p -> 0 stop being greedy.
    const s = mlx.gpuStream();
    const data = [_]f32{
        2.0, 5.0, 5.0, 5.0, 1.0,
        0.1, 0.2, 9.0, 0.3, 0.4,
    };
    const shape = [_]c_int{ 2, 5 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    for ([_]bool{ true, false }) |use_k| {
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        if (use_k) try applyTopK(&masked, logits, 1, s) else try applyTopP(&masked, logits, 1e-6, 0, s);
        try mlx.check(mlx.mlx_array_eval(masked));
        const out = mlx.mlx_array_data_float32(masked).?[0..10];
        var kept0: usize = 0;
        for (out[0..5]) |v| kept0 += @intFromBool(v != -std.math.inf(f32));
        try testing.expectEqual(@as(usize, 1), kept0);
        for (out[5..10], 0..) |v, i| try testing.expect((v != -std.math.inf(f32)) == (i == 2));
    }
    // A 3-of-5 top-k on the tied row keeps exactly the three tied tokens.
    var masked3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked3);
    try applyTopK(&masked3, logits, 3, s);
    try mlx.check(mlx.mlx_array_eval(masked3));
    const out3 = mlx.mlx_array_data_float32(masked3).?[0..5];
    for (out3, 0..) |v, i| try testing.expect((v != -std.math.inf(f32)) == (i >= 1 and i <= 3));
}

test "batched stacked seedless rows honour a suppress mask" {
    const s = mlx.gpuStream();
    const mask = try buildSuppressMask(&[_]u32{1}, 0, 4, s);
    defer _ = mlx.mlx_array_free(mask);
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 9.0, 0.5, 0.6,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const p = SamplingParams{ .temperature = 0.7, .top_k = 1, .top_p = 0.9, .suppress_mask = mask };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, p, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, p, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ p, p };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
    try testing.expect(ids[0] != 1);
    try testing.expect(ids[1] != 1);
}

test "batched seeded draws keep per-row keys" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const p0 = SamplingParams{ .temperature = 1.0, .seed = 7, .draw = 2 };
    const p1 = SamplingParams{ .temperature = 1.0, .seed = 99, .draw = 5 };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, p0, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, p1, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ p0, p1 };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
    try testing.expectEqual(p0.draw +% 1, params[0].draw);
    try testing.expectEqual(p1.draw +% 1, params[1].draw);
}

test "sampleToken from prefill logits (seq_len > 1)" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // [1, 3, 4] — 3 positions, 4 vocab, should take last position
    const data = [_]f32{
        0.1, 0.2, 0.3, 0.4, // pos 0
        0.5, 0.6, 0.7, 0.8, // pos 1
        9.0, 0.1, 0.1, 0.1, // pos 2 — token 0 is clearly highest
    };
    const logits_shape = [_]c_int{ 1, 3, 4 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, null, s);
    try testing.expectEqual(@as(u32, 0), result.token_id); // pos 2, index 0 = 9.0
}

/// Ids a CallForce commits after `published`, each published in turn, until it samples.
fn drainCallForce(cf: *CallForce, published: []const u32) ![]u32 {
    var gen = std.ArrayList(u32).empty;
    defer gen.deinit(std.testing.allocator);
    try gen.appendSlice(std.testing.allocator, published);
    while (true) switch (cf.decide(gen.items)) {
        .force => |id| try gen.append(std.testing.allocator, id),
        .sample => break,
        .choose => return error.UnexpectedChoice,
    };
    return std.testing.allocator.dupe(u32, gen.items[published.len..]);
}

test "CallForce: the call follows the closed thought, every other position samples" {
    const OPEN: u32 = 900;
    const CLOSE: u32 = 901;
    const NL2: u32 = 271;
    const forced = [_]u32{ NL2, 10, 11, 12 }; // separator, then the call
    const call = forced[1..];
    const t = std.testing;

    // A prompt-opened thought runs free; the closer brings the separator and the call.
    {
        var cf = CallForce{ .forced = &forced, .call_at = 1, .closer_id = CLOSE, .phase = .thinking, .last = 5 };
        try t.expect(cf.pending());
        try t.expect(cf.decide(&.{ 1, 2 }) == .sample);
        const got = try drainCallForce(&cf, &.{ 1, 2, CLOSE });
        defer t.allocator.free(got);
        try t.expectEqualSlices(u32, &forced, got);
        try t.expect(!cf.pending());
    }
    // A thought the prompt closed on the closer itself: from token 0, separator first.
    {
        var cf = CallForce{ .forced = &forced, .call_at = 1, .closer_id = CLOSE, .phase = .answer, .last = CLOSE };
        const got = try drainCallForce(&cf, &.{});
        defer t.allocator.free(got);
        try t.expectEqualSlices(u32, &forced, got);
    }
    // The prompt already wrote the separator: only the call.
    {
        var cf = CallForce{ .forced = &forced, .call_at = 1, .closer_id = CLOSE, .phase = .answer, .last = NL2 };
        const got = try drainCallForce(&cf, &.{});
        defer t.allocator.free(got);
        try t.expectEqualSlices(u32, call, got);
    }
    // A reasoning-budget commit already wrote the separator after the closer.
    {
        var cf = CallForce{ .forced = &forced, .call_at = 1, .closer_id = CLOSE, .phase = .thinking, .last = 5 };
        const got = try drainCallForce(&cf, &.{ 1, CLOSE, NL2 });
        defer t.allocator.free(got);
        try t.expectEqualSlices(u32, call, got);
    }
    // The template leaves the opener to the model: it is committed at token 0,
    // the thought then runs free until its closer.
    {
        var cf = CallForce{ .forced = &forced, .call_at = 1, .closer_id = CLOSE, .opener = .{OPEN}, .phase = .open, .last = 5 };
        const first = try drainCallForce(&cf, &.{});
        defer t.allocator.free(first);
        try t.expectEqualSlices(u32, &.{OPEN}, first);
        try t.expect(cf.decide(&.{ OPEN, 1, 2 }) == .sample);
        const got = try drainCallForce(&cf, &.{ OPEN, 1, 2, CLOSE });
        defer t.allocator.free(got);
        try t.expectEqualSlices(u32, &forced, got);
    }
}

/// One flat f32 host copy of a lazy array, for the proposal-row assertions.
pub fn proposalTestRead(allocator: std.mem.Allocator, arr: mlx.mlx_array, n: usize, s: mlx.mlx_stream) ![]f32 {
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, arr, .float32, s));
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_contiguous(&dense, as_f32, false, s));
    const flat_shape = [_]c_int{@intCast(n)};
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, dense, &flat_shape, 1, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    const out = try allocator.alloc(f32, n);
    @memcpy(out, mlx.mlx_array_data_float32(flat).?[0..n]);
    return out;
}

test "shortlistProposal: q is zero off the shortlist, sums to 1, and peaks on the best candidate" {
    // Bar: q is a normalized density over the 32 candidates only, and the
    // draft is one of them.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const V: c_int = 4096;
    const N: usize = 32;

    var ids: [N]u32 = undefined;
    var logits: [N]f32 = undefined;
    for (0..N) |i| {
        ids[i] = @intCast(37 + i * 101); // spread over the row, none at 0
        logits[i] = @as(f32, @floatFromInt(i)) * 0.25;
    }
    const best_id = ids[N - 1]; // highest logit
    const id_shape = [_]c_int{@intCast(N)};
    const cands = mlx.mlx_array_new_data(&ids, &id_shape, 1, .uint32);
    const exact_shape = [_]c_int{ 1, 1, @intCast(N) };
    const exact = mlx.mlx_array_new_data(&logits, &exact_shape, 3, .float32);
    var sl = mtp_mod.Shortlist{ .cands = cands, .exact = exact, .rows = V };
    defer sl.deinit();

    const sampling = Generator.mtpDraftSamplingFor(.{ .temperature = 1.0 }, false, mtp_round.MTP_DRAFT_TEMP);
    const prop = try shortlistProposal(sl, sampling, s);
    defer _ = mlx.mlx_array_free(prop.id);
    defer _ = mlx.mlx_array_free(prop.q);

    const q = try proposalTestRead(allocator, prop.q, @intCast(V), s);
    defer allocator.free(q);
    var sum: f64 = 0;
    var nonzero: usize = 0;
    var peak: usize = 0;
    for (q, 0..) |v, i| {
        sum += v;
        if (v != 0) {
            nonzero += 1;
            try testing.expect(std.mem.indexOfScalar(u32, &ids, @intCast(i)) != null);
        }
        if (v > q[peak]) peak = i;
    }
    try testing.expect(nonzero > 0 and nonzero <= N);
    try testing.expect(@abs(sum - 1.0) < 1e-4);
    try testing.expectEqual(best_id, @as(u32, @intCast(peak)));

    // Every draw lands on a candidate — never on a token q calls impossible.
    var draw: usize = 0;
    while (draw < 8) : (draw += 1) {
        const p2 = try shortlistProposal(sl, sampling, s);
        defer _ = mlx.mlx_array_free(p2.id);
        defer _ = mlx.mlx_array_free(p2.q);
        const drawn = try proposalTestRead(allocator, p2.id, 1, s);
        defer allocator.free(drawn);
        const tok: u32 = @intFromFloat(drawn[0]);
        try testing.expect(std.mem.indexOfScalar(u32, &ids, tok) != null);
    }
}

test "shortlistProposalRows: every row's q belongs to that row's own shortlist" {
    // The axis-bug class: one batched filter + one categorical over [K, 32] must
    // stay row-independent. Three rows with DISJOINT candidate ids and different
    // logits, so a cross-row leak or a flattened reduction is visible.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const V: c_int = 4096;
    const N: usize = 3;
    const M: usize = 32;

    var ids: [N][M]u32 = undefined;
    var logits: [N][M]f32 = undefined;
    var sl: [N]mtp_mod.Shortlist = undefined;
    for (0..N) |r| {
        for (0..M) |i| {
            ids[r][i] = @intCast(11 + r * 41 + i * 127); // disjoint per row
            // Row r peaks at a different position, so an argmax read off the
            // wrong row lands on the wrong id.
            logits[r][i] = if (i == (r * 7 + 3) % M) 6.0 else @as(f32, @floatFromInt(i)) * 0.05;
        }
        const id_shape = [_]c_int{@intCast(M)};
        const ex_shape = [_]c_int{ 1, 1, @intCast(M) };
        sl[r] = .{
            .cands = mlx.mlx_array_new_data(&ids[r], &id_shape, 1, .uint32),
            .exact = mlx.mlx_array_new_data(&logits[r], &ex_shape, 3, .float32),
            .rows = V,
        };
    }
    defer for (&sl) |*x| x.deinit();

    const sampling = Generator.mtpDraftSamplingFor(.{ .temperature = 1.0 }, false, mtp_round.MTP_DRAFT_TEMP);
    const rows = [_]usize{ 0, 1, 2 };
    const out = try shortlistProposalRows(&sl, &rows, sampling, s);
    defer for (0..N) |j| {
        _ = mlx.mlx_array_free(out.ids[j]);
        _ = mlx.mlx_array_free(out.qs[j]);
    };

    for (0..N) |r| {
        const q = try proposalTestRead(allocator, out.qs[r], @intCast(V), s);
        defer allocator.free(q);
        var sum: f64 = 0;
        var peak: usize = 0;
        for (q, 0..) |v, i| {
            sum += v;
            if (v != 0) try testing.expect(std.mem.indexOfScalar(u32, &ids[r], @intCast(i)) != null);
            if (v > q[peak]) peak = i;
        }
        try testing.expect(@abs(sum - 1.0) < 1e-4);
        try testing.expectEqual(ids[r][(r * 7 + 3) % M], @as(u32, @intCast(peak)));
        const drawn = try proposalTestRead(allocator, out.ids[r], 1, s);
        defer allocator.free(drawn);
        try testing.expect(std.mem.indexOfScalar(u32, &ids[r], @intFromFloat(drawn[0])) != null);
    }
}

test "mtpDraftStepPath: a sampled proposal keeps the rerank shortlist" {
    // Bar: a sampled proposal reads the rerank shortlist; only the chunk-A
    // confidence gate needs the full lm_head.
    const P = mtp_round.DraftStepPath;
    try testing.expectEqual(P.rerank_sampled, Generator.mtpDraftStepPath(true, true, false));
    try testing.expectEqual(P.rerank_greedy, Generator.mtpDraftStepPath(true, false, false));
    // The extension gate wants ln p_head over the whole vocabulary.
    try testing.expectEqual(P.full_logits, Generator.mtpDraftStepPath(true, true, true));
    try testing.expectEqual(P.full_logits, Generator.mtpDraftStepPath(true, false, true));
    // No coarse head: nothing to shortlist.
    try testing.expectEqual(P.full_logits, Generator.mtpDraftStepPath(false, true, false));
    try testing.expectEqual(P.full_logits, Generator.mtpDraftStepPath(false, false, false));
}

test "mtpDraftSamplingFor: sharpened fixed proposal for stochastic targets, greedy stays greedy" {
    // Stochastic target: drafts sample from the FIXED sharpened distribution
    // (temp 0.6 / top_p 0.95 / top_k 20 — oMLX Lightning's _DRAFT_SAMPLER_*
    // constants; matched-temp drafting collapses on high-entropy content).
    const target = SamplingParams{ .temperature = 1.0, .top_p = 1.0, .top_k = 0, .repeat_penalty = 1.1 };
    const d = Generator.mtpDraftSamplingFor(target, false, mtp_round.MTP_DRAFT_TEMP);
    try testing.expectEqual(@as(f32, 0.6), d.temperature);
    try testing.expectEqual(@as(f32, 0.95), d.top_p);
    try testing.expectEqual(@as(u32, 20), d.top_k);
    // Non-sampler fields ride through untouched.
    try testing.expectEqual(@as(f32, 1.1), d.repeat_penalty);

    // Greedy target keeps greedy drafts (the temp-0 identity contract).
    const greedy = SamplingParams{ .temperature = 0.0 };
    try testing.expectEqual(@as(f32, 0.0), Generator.mtpDraftSamplingFor(greedy, false, mtp_round.MTP_DRAFT_TEMP).temperature);
    const near_greedy = SamplingParams{ .temperature = 0.005 };
    try testing.expectEqual(@as(f32, 0.0), Generator.mtpDraftSamplingFor(near_greedy, false, mtp_round.MTP_DRAFT_TEMP).temperature);

    // An explicit greedy request wins.
    try testing.expectEqual(@as(f32, 0.0), Generator.mtpDraftSamplingFor(target, true, mtp_round.MTP_DRAFT_TEMP).temperature);

    // The draft temperature is the FAMILY's, passed in, never a global.
    try testing.expectEqual(@as(f32, 1.0), Generator.mtpDraftSamplingFor(target, false, 1.0).temperature);
}

test "mtpDraftGreedyFor: the proposal mode is a property of the REQUEST" {
    const sampled = SamplingParams{ .temperature = 1.0 };
    const warm = SamplingParams{ .temperature = 0.7 };
    const cold = SamplingParams{ .temperature = 0.0 };
    const near_cold = SamplingParams{ .temperature = 0.005 };
    // top_k 1 is a deterministic target: its filtered p is one-hot, so the
    // argmax proposal already accepts with probability 1.
    const one_way = SamplingParams{ .temperature = 1.0, .top_k = 1 };

    // Sampled targets propose from the shortlist, greedy ones do not.
    try testing.expect(!Generator.mtpDraftGreedyFor(sampled));
    try testing.expect(!Generator.mtpDraftGreedyFor(warm));
    try testing.expect(Generator.mtpDraftGreedyFor(cold));
    // Bar: a sharp target keeps the argmax, a flat one samples.
    try testing.expect(Generator.mtpDraftGreedyFor(.{ .temperature = 0.3 }));
    try testing.expect(Generator.mtpDraftGreedyFor(near_cold));
    try testing.expect(Generator.mtpDraftGreedyFor(one_way));
}

test "mtpDraftGreedyAt: the greedy tail samples depth 0 only; a greedy request drafts the same bytes either way" {
    const sampled = SamplingParams{ .temperature = 1.0, .top_p = 0.95, .top_k = 20 };
    for (0..4) |d| {
        const step: u32 = @intCast(d);
        try testing.expect(!Generator.mtpDraftGreedyAt(sampled, false, step));
        try testing.expectEqual(step > 0, Generator.mtpDraftGreedyAt(sampled, true, step));
    }
    // A request whose chain is greedy already stays greedy at every depth.
    try testing.expect(Generator.mtpDraftGreedyAt(.{ .temperature = 0.3 }, true, 0));

    // The byte-identity contract: a greedy target's draft sampler is the same
    // value at every depth with the tail on as off, so its chain allocates no q
    // and drafts the argmax exactly as before.
    for ([_]f32{ 0.0, 0.005 }) |t| {
        const target = SamplingParams{ .temperature = t, .top_p = 0.95, .top_k = 20, .seed = 7 };
        for (0..mtp_mod.MAX_DEPTH) |d| {
            const step: u32 = @intCast(d);
            const on = Generator.mtpDraftSamplingFor(target, Generator.mtpDraftGreedyAt(target, true, step), mtp_round.MTP_DRAFT_TEMP_QWEN4);
            const off = Generator.mtpDraftSamplingFor(target, Generator.mtpDraftGreedyAt(target, false, step), mtp_round.MTP_DRAFT_TEMP_QWEN4);
            try testing.expect(std.meta.eql(on, off));
            try testing.expect(isGreedyTemperature(on.temperature));
        }
    }
}

/// Toy-vocab helpers for the host-level exactness simulations.
const ToyDist = struct {
    fn draw(dist: []const f64, rnd: std.Random) usize {
        const u = rnd.float(f64);
        var acc: f64 = 0;
        for (dist, 0..) |mass, i| {
            acc += mass;
            if (u < acc) return i;
        }
        return dist.len - 1;
    }

    /// normalize(max(p - q, 0)): the reject residual.
    fn residual(comptime V: usize, p: [V]f64, q: [V]f64) [V]f64 {
        var r: [V]f64 = undefined;
        var sum: f64 = 0;
        for (&r, p, q) |*x, pi, qi| {
            x.* = @max(pi - qi, 0);
            sum += x.*;
        }
        for (&r) |*x| x.* /= sum;
        return r;
    }

    fn oneHot(comptime V: usize, id: usize) [V]f64 {
        var d: [V]f64 = @splat(0);
        d[id] = 1;
        return d;
    }
};

test "spec sampling exactness: a greedy tail (depth 0 sampled, argmax after it, one-hot q) reproduces the target chain (toy vocab)" {
    // Two draft depths over V = 4, run through the engine's exact acceptance
    // (`exactPrefix`): depth 0 is drawn from q0 and handed to the verify as q0,
    // depth 1 is the draft head's argmax after the depth-0 token with a one-hot q.
    // Rejecting at a depth resamples normalize(max(p - q, 0)); a reject at depth 0
    // leaves depth 1 to the next round, i.e. to the target itself. Every emitted
    // pair must follow the target's joint p0(x0) p1(x1 | x0).
    const target = SamplingParams{ .temperature = 1.0 };
    try testing.expect(!Generator.mtpDraftGreedyAt(target, true, 0));
    try testing.expect(Generator.mtpDraftGreedyAt(target, true, 1));
    const V = 4;
    const p0 = [V]f64{ 0.1, 0.2, 0.3, 0.4 };
    const p1 = [V][V]f64{
        .{ 0.7, 0.1, 0.1, 0.1 },
        .{ 0.1, 0.2, 0.3, 0.4 },
        .{ 0.25, 0.25, 0.25, 0.25 },
        .{ 0.05, 0.05, 0.1, 0.8 },
    };
    const q0 = [V]f64{ 0.4, 0.3, 0.2, 0.1 };
    // The head's depth-1 argmax per depth-0 token: the target's mode on two rows, not on the others.
    const head_argmax = [V]usize{ 0, 1, 3, 2 };

    var prng = std.Random.DefaultPrng.init(0x6A11);
    var counts = std.mem.zeroes([V][V]u64);
    var tail_accepts: u64 = 0;
    const N: usize = 400_000;
    for (0..N) |_| {
        const d0 = ToyDist.draw(&q0, prng.random());
        const d1 = head_argmax[d0];
        const p_at = [_]f32{ @floatCast(p0[d0]), @floatCast(p1[d0][d1]) };
        const q_at = [_]f32{ @floatCast(q0[d0]), 1.0 };
        const accepted = mtp_acceptance.exactPrefix(&p_at, &q_at, null, &prng);
        var x0 = d0;
        var x1 = d1;
        switch (accepted) {
            0 => {
                x0 = ToyDist.draw(&ToyDist.residual(V, p0, q0), prng.random());
                x1 = ToyDist.draw(&p1[x0], prng.random());
            },
            1 => x1 = ToyDist.draw(&ToyDist.residual(V, p1[d0], ToyDist.oneHot(V, d1)), prng.random()),
            else => tail_accepts += 1,
        }
        counts[x0][x1] += 1;
    }
    for (0..V) |a| {
        for (0..V) |b| {
            const freq = @as(f64, @floatFromInt(counts[a][b])) / @as(f64, @floatFromInt(N));
            try testing.expect(@abs(freq - p0[a] * p1[a][b]) < 0.004);
        }
    }
    // The argmax tail lands (0.17 of rounds here), so the joint is not the target by rejection alone.
    try testing.expect(tail_accepts > N / 10);
}

test "mtpRegimeObserve: averages the first 1/beta samples, then moves by the cost beta, reseeds on a new base depth" {
    var r = mtp_policy.MtpRegime{};
    Generator.mtpRegimeObserve(&r, true, 3, 45.0, 4.0); // first round counts
    try testing.expect(r.two_tok > 0.0);
    Generator.mtpRegimeObserve(&r, false, 3, 45.0, 4.0); // transition: dropped
    try testing.expect(r.one_tok == 0.0);
    Generator.mtpRegimeObserve(&r, false, 3, 50.0, 4.0);
    try testing.expectApproxEqAbs(@as(f32, 50.0), r.one_ms, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 4.0), r.one_tok, 1e-6);
    Generator.mtpRegimeObserve(&r, false, 3, 60.0, 5.0);
    try testing.expectApproxEqAbs(@as(f32, 55.0), r.one_ms, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 4.5), r.one_tok, 1e-4);
    for (0..8) |_| Generator.mtpRegimeObserve(&r, false, 3, 60.0, 5.0);
    try testing.expectApproxEqAbs(@as(f32, 59.0), r.one_ms, 1e-3);
    Generator.mtpRegimeObserve(&r, false, 3, 70.0, 5.0);
    try testing.expectApproxEqAbs(59.0 + mtp_policy.MTP_EV_COST_BETA * 11.0, r.one_ms, 1e-3);
    // The climb moved m_lo: the depth-3 rounds are not the depth-4 regime.
    Generator.mtpRegimeObserve(&r, false, 4, 70.0, 5.0);
    try testing.expectApproxEqAbs(@as(f32, 70.0), r.one_ms, 1e-6);
    // A two-chunk reading at a different base depth yields no verdict.
    Generator.mtpRegimeObserve(&r, true, 3, 80.0, 6.0);
    Generator.mtpRegimeObserve(&r, true, 3, 80.0, 6.0);
    try testing.expect(Generator.mtpRegimeTwoChunkWorse(r) == null);
}

test "mtpEmaMs: seeds on the first sample then folds nanoseconds into a ms EMA" {
    const D = Generator;
    // First sample seeds (prev <= 0): 45 ms == 45_000_000 ns, exactly.
    try testing.expectApproxEqAbs(@as(f32, 45.0), D.mtpEmaMs(0.0, 45_000_000), 1e-3);
    // A subsequent sample folds at MTP_EV_COST_BETA toward the new value.
    const folded = D.mtpEmaMs(45.0, 40_000_000); // 40 ms sample
    try testing.expectApproxEqAbs(45.0 + mtp_policy.MTP_EV_COST_BETA * (40.0 - 45.0), folded, 1e-3);
    // Env parse: default on, "0" off, "" on.
    try testing.expect(D.mtpCostTableEnabledFromEnv(null));
    try testing.expect(!D.mtpCostTableEnabledFromEnv("0"));
    try testing.expect(D.mtpCostTableEnabledFromEnv(""));
}

test "batched corrections: point-mass residuals sample deterministically, accept vec gathers draft probs" {
    const s = mlx.gpuStream();
    // V=4, m=2. Row residuals are POINT MASSES so the batched categorical
    // is deterministic end-to-end (no seed dependence):
    //   row0: p=[.5,.5,0,0], draft=0 → residual [0,.5,0,0] → corr 1
    //   row1: p=[0,.25,.75,0], draft=2 → residual [0,.25,0,0] → corr 1
    //   bonus: p=[0,0,0,1] → corr 3
    const p_data = [_]f32{ 0.5, 0.5, 0, 0, 0, 0.25, 0.75, 0, 0, 0, 0, 1.0 };
    const p_shape = [_]c_int{ 1, 3, 4 };
    const probs_all = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs_all);
    const id_shape = [_]c_int{1};
    const d0_data: i32 = 0;
    const d1_data: i32 = 2;
    const d0 = mlx.mlx_array_new_data(&d0_data, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(d0);
    const d1 = mlx.mlx_array_new_data(&d1_data, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(d1);
    const drafts = [_]mlx.mlx_array{ d0, d1 };

    // Greedy proposals (q == null): one-hot residuals.
    var g = try Generator.mtpBatchedAcceptGraph(probs_all, &drafts, null, 2, .{}, s);
    defer g.deinit();
    try mlx.check(mlx.mlx_array_eval(g.corr_samples));
    const corr = mlx.mlx_array_data_int32(g.corr_samples) orelse return error.InvalidDtype;
    try testing.expectEqual(@as(i32, 1), corr[0]);
    try testing.expectEqual(@as(i32, 1), corr[1]);
    try testing.expectEqual(@as(i32, 3), corr[2]);
    try mlx.check(mlx.mlx_array_eval(g.accept_p));
    const ap = mlx.mlx_array_data_float32(g.accept_p) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 0.5), ap[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.75), ap[1], 1e-6);
    try testing.expect(g.accept_q.ctx == null);

    // Sharp proposals: q rows equal to the one-hots reproduce the same
    // residuals, and accept_q gathers the proposal's own density.
    const q_data0 = [_]f32{ 1, 0, 0, 0 };
    const q_data1 = [_]f32{ 0, 0, 1, 0 };
    const q_shape = [_]c_int{ 1, 4 };
    const q0 = mlx.mlx_array_new_data(&q_data0, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q0);
    const q1 = mlx.mlx_array_new_data(&q_data1, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q1);
    const qs = [_]mlx.mlx_array{ q0, q1 };
    var gs = try Generator.mtpBatchedAcceptGraph(probs_all, &drafts, &qs, 2, .{}, s);
    defer gs.deinit();
    try mlx.check(mlx.mlx_array_eval(gs.corr_samples));
    const corr2 = mlx.mlx_array_data_int32(gs.corr_samples) orelse return error.InvalidDtype;
    try testing.expectEqual(@as(i32, 1), corr2[0]);
    try testing.expectEqual(@as(i32, 1), corr2[1]);
    try testing.expectEqual(@as(i32, 3), corr2[2]);
    try mlx.check(mlx.mlx_array_eval(gs.accept_q));
    const aq = mlx.mlx_array_data_float32(gs.accept_q) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 1.0), aq[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), aq[1], 1e-6);
}

test "seeded MTP draft and correction draws replay despite unrelated MLX random draws" {
    const s = mlx.gpuStream();
    const q_data = [_]f32{ 0.5, 0.5, 0.0, 0.0 };
    const q_shape = [_]c_int{ 1, 4 };
    const q = mlx.mlx_array_new_data(&q_data, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q);
    const p_data = [_]f32{ 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0 };
    const p_shape = [_]c_int{ 1, 2, 4 };
    const p = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(p);
    const d_data: i32 = 3;
    const d_shape = [_]c_int{1};
    const d = mlx.mlx_array_new_data(&d_data, &d_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(d);
    const drafts = [_]mlx.mlx_array{d};

    var runs: [2][16][3]i32 = undefined;
    for (&runs) |*run| {
        for (run, 0..) |*out, index| {
            const draft_params = SamplingParams{ .seed = 0xC0FFEE, .draw = @as(u64, @intCast(index)) * 2 };
            const lazy_draft = try sampleFromProbsLazy(q, draft_params, s);
            defer _ = mlx.mlx_array_free(lazy_draft);
            try mlx.check(mlx.mlx_array_eval(lazy_draft));
            try mlx.check(mlx.mlx_array_item_int32(&out[0], lazy_draft));

            const correction_params = SamplingParams{ .seed = 0xC0FFEE, .draw = draft_params.draw + 1 };
            var graph = try Generator.mtpBatchedAcceptGraph(p, &drafts, null, 1, correction_params, s);
            defer graph.deinit();
            try mlx.check(mlx.mlx_array_eval(graph.corr_samples));
            const corrected = mlx.mlx_array_data_int32(graph.corr_samples) orelse return error.MlxArrayDataNull;
            out[1] = corrected[0];
            out[2] = corrected[1];
        }

        // A seeded replay must not depend on unrelated requests consuming the
        // process-global MLX RNG between runs.
        const unrelated = try sampleFromProbsLazy(q, SamplingParams{}, s);
        defer _ = mlx.mlx_array_free(unrelated);
        try mlx.check(mlx.mlx_array_eval(unrelated));
    }
    try testing.expectEqualSlices([3]i32, &runs[0], &runs[1]);
    var all_same = true;
    for (runs[0][1..]) |draw| all_same = all_same and std.mem.eql(i32, &draw, &runs[0][0]);
    try testing.expect(!all_same);
}

test "typical + greedy tail: an argmax tail is accepted when it clears the entropy floor" {
    // Depth 0 is the sampled token 1 (p 0.3, floor 0.2 * exp(-H) = 0.071); depth 1
    // is the head's argmax, judged against row 1's floor of 0.130. The q rows the
    // chain carries (a density, then a one-hot) do not move a typical decision.
    const s = mlx.gpuStream();
    const p_data = [_]f32{ 0.5, 0.3, 0.2, 0.0, 0.02, 0.9, 0.04, 0.04, 0.0, 0.0, 0.0, 1.0 };
    const probs = mlx.mlx_array_new_data(&p_data, &[_]c_int{ 1, 3, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const q0_data = [_]f32{ 0.3, 0.5, 0.2, 0.0 };
    const q0 = mlx.mlx_array_new_data(&q0_data, &[_]c_int{ 1, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(q0);

    const Case = struct { tail: i32, accepted: u32 };
    // The target's own mode clears the floor; a head argmax the target puts 0.04 on does not.
    for ([_]Case{ .{ .tail = 1, .accepted = 2 }, .{ .tail = 2, .accepted = 1 } }) |case| {
        const d0_id: i32 = 1;
        const d0 = mlx.mlx_array_new_data(&d0_id, &[_]c_int{1}, 1, .int32);
        defer _ = mlx.mlx_array_free(d0);
        const d1 = mlx.mlx_array_new_data(&case.tail, &[_]c_int{1}, 1, .int32);
        defer _ = mlx.mlx_array_free(d1);
        const q1 = try lazyOneHotRow(d1, 4, s);
        defer _ = mlx.mlx_array_free(q1);
        var g = try Generator.mtpBatchedTypicalGraph(probs, &.{ d0, d1 }, &.{ q0, q1 }, 2, 0.2, .{}, s);
        defer g.deinit();
        try mlx.check(mlx.mlx_array_eval(g.accept_p));
        try mlx.check(mlx.mlx_array_eval(g.accept_q));
        const p = mlx.mlx_array_data_float32(g.accept_p) orelse return error.InvalidDtype;
        const floor = mlx.mlx_array_data_float32(g.accept_q) orelse return error.InvalidDtype;
        try testing.expectApproxEqAbs(@as(f32, 0.0714), floor[0], 1e-3);
        try testing.expectApproxEqAbs(@as(f32, 0.1300), floor[1], 1e-3);
        var rng = std.Random.DefaultPrng.init(1);
        try testing.expectEqual(case.accepted, mtp_acceptance.typicalPrefix(p[0..2], floor[0..2], null, &rng));
    }
}

test "greedy tail: each depth's verify q is the density its draft came from" {
    // Depth 0 of a sampled request draws from the shortlist; depth 1 takes its
    // argmax. The rows the chain hands the verify must be exactly those
    // proposals: a shortlist density with the draft in its support, then a
    // one-hot at the argmax, so accept_q = [q0(d0), 1] and a reject at depth 1
    // can never re-draw d1.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const V: c_int = 4096;
    const N: usize = 32;
    var ids: [N]u32 = undefined;
    var logits: [N]f32 = undefined;
    for (0..N) |i| {
        ids[i] = @intCast(37 + i * 101);
        logits[i] = @as(f32, @floatFromInt(i)) * 0.25;
    }
    var sl = mtp_mod.Shortlist{
        .cands = mlx.mlx_array_new_data(&ids, &[_]c_int{@intCast(N)}, 1, .uint32),
        .exact = mlx.mlx_array_new_data(&logits, &[_]c_int{ 1, 1, @intCast(N) }, 3, .float32),
        .rows = V,
    };
    defer sl.deinit();

    const target = SamplingParams{ .temperature = 1.0 };
    var props: [2]StepProposal = undefined;
    var built: usize = 0;
    defer for (props[0..built]) |p| {
        _ = mlx.mlx_array_free(p.id);
        _ = mlx.mlx_array_free(p.q);
    };
    for (&props, 0..) |*p, d| {
        const greedy = Generator.mtpDraftGreedyAt(target, true, @intCast(d));
        p.* = try shortlistStepProposal(sl, Generator.mtpDraftSamplingFor(target, greedy, mtp_round.MTP_DRAFT_TEMP), s);
        built += 1;
        // Every depth drafts the same `[1]` id shape: the verify concatenates them.
        try testing.expectEqualSlices(c_int, &.{1}, mlx.getShape(p.id));
    }

    const q0 = try proposalTestRead(allocator, props[0].q, @intCast(V), s);
    defer allocator.free(q0);
    const q1 = try proposalTestRead(allocator, props[1].q, @intCast(V), s);
    defer allocator.free(q1);
    const id_reads = [_][]f32{
        try proposalTestRead(allocator, props[0].id, 1, s),
        try proposalTestRead(allocator, props[1].id, 1, s),
    };
    defer for (id_reads) |r| allocator.free(r);
    const id0: usize = @intFromFloat(id_reads[0][0]);
    const id1: usize = @intFromFloat(id_reads[1][0]);
    var sum0: f64 = 0;
    var support0: usize = 0;
    for (q0) |v| {
        sum0 += v;
        if (v > 0) support0 += 1;
    }
    try testing.expect(@abs(sum0 - 1.0) < 1e-4 and support0 > 1 and q0[id0] > 0);
    try testing.expectEqual(@as(usize, ids[N - 1]), id1);
    for (q1, 0..) |v, i| try testing.expectEqual(@as(f32, if (i == id1) 1 else 0), v);

    // Target rows: row 1 splits its mass between the argmax draft and one other id.
    const Vu: usize = @intCast(V);
    const other: usize = ids[0];
    const p_data = try allocator.alloc(f32, 3 * Vu);
    defer allocator.free(p_data);
    @memset(p_data, 0);
    @memcpy(p_data[0..Vu], q0);
    p_data[Vu + id1] = 0.5;
    p_data[Vu + other] = 0.5;
    p_data[2 * Vu + other] = 1.0;
    const probs = mlx.mlx_array_new_data(p_data.ptr, &[_]c_int{ 1, 3, V }, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    var g = try Generator.mtpBatchedAcceptGraph(probs, &.{ props[0].id, props[1].id }, &.{ props[0].q, props[1].q }, 2, .{}, s);
    defer g.deinit();
    try mlx.check(mlx.mlx_array_eval(g.accept_q));
    try mlx.check(mlx.mlx_array_eval(g.accept_p));
    try mlx.check(mlx.mlx_array_eval(g.corr_samples));
    const aq = mlx.mlx_array_data_float32(g.accept_q) orelse return error.InvalidDtype;
    const ap = mlx.mlx_array_data_float32(g.accept_p) orelse return error.InvalidDtype;
    const corr = mlx.mlx_array_data_int32(g.corr_samples) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(q0[id0], aq[0], 1e-6);
    try testing.expectEqual(@as(f32, 1.0), aq[1]);
    try testing.expectEqual(@as(f32, 0.5), ap[1]);
    try testing.expectEqual(@as(i32, @intCast(other)), corr[1]);
}

test "MTP TokenV3 batched graph uses full pi for sampled and one-hot drafts" {
    const s = mlx.gpuStream();
    const p_data = [_]f32{ 0.8, 0.19, 0.01, 0, 0, 0, 0, 1 };
    const p_shape = [_]c_int{ 1, 2, 4 };
    const probs = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const id: i32 = 2;
    const id_shape = [_]c_int{1};
    const draft = mlx.mlx_array_new_data(&id, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(draft);
    const drafts = [_]mlx.mlx_array{draft};
    const q_data = [_]f32{ 0.1, 0.2, 0.7, 0 };
    const q_shape = [_]c_int{ 1, 4 };
    const q = mlx.mlx_array_new_data(&q_data, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q);
    const qs = [_]mlx.mlx_array{q};

    var sampled = try Generator.mtpBatchedTokenV3Graph(probs, &drafts, &qs, 1, 0.95, .{}, s);
    defer sampled.deinit();
    try mlx.check(mlx.mlx_array_eval(sampled.accept_p));
    try mlx.check(mlx.mlx_array_eval(sampled.accept_q));
    try mlx.check(mlx.mlx_array_eval(sampled.accept_defer));
    const pi_draft = mlx.mlx_array_data_float32(sampled.accept_p) orelse return error.InvalidDtype;
    const q_draft = mlx.mlx_array_data_float32(sampled.accept_q) orelse return error.InvalidDtype;
    const deferred = mlx.mlx_array_data_bool(sampled.accept_defer) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 0.007), pi_draft[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.7), q_draft[0], 1e-6);
    try testing.expect(deferred[0]);
    try mlx.check(mlx.mlx_array_eval(sampled.corr_samples));
    const corr = mlx.mlx_array_data_int32(sampled.corr_samples) orelse return error.InvalidDtype;
    try testing.expect(corr[0] == 0 or corr[0] == 1);
    try testing.expectEqual(@as(i32, 3), corr[1]);

    var onehot = try Generator.mtpBatchedTokenV3Graph(probs, &drafts, null, 1, 0.95, .{}, s);
    defer onehot.deinit();
    try mlx.check(mlx.mlx_array_eval(onehot.accept_p));
    const onehot_pi = mlx.mlx_array_data_float32(onehot.accept_p) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 0.01), onehot_pi[0], 1e-6);
    try testing.expect(onehot.accept_q.ctx == null);
}

fn evalLazyToken(lazy: mlx.mlx_array) !u32 {
    defer _ = mlx.mlx_array_free(lazy);
    try mlx.check(mlx.mlx_array_eval(lazy));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
    return @intCast(val);
}

test "fillSuppressMask: padding rows past the tokenizer's last defined id are suppressed" {
    // A config's `vocab_size` is a padded matrix dimension, not a vocabulary.
    // qwen4_exp declares 248320 while the tokenizer defines 248044 + 33 =
    // 248077 ids: 243 rows that decode to NOTHING, carry whatever the
    // initializer left, and were sampleable — a drawn one emits no text while
    // consuming a step and entering the KV. Same mask, second row class.
    var buf: [16]bool = undefined;
    fillSuppressMask(&buf, &[_]u32{3}, 12);
    for (buf, 0..) |v, i| {
        const want = (i == 3) or (i >= 12);
        try testing.expectEqual(want, v);
    }

    // No padding known (0) and a fully-defined vocab (== len) both add
    // nothing: a mask over the WHOLE vocab is -inf everywhere, and argmax
    // over that row returns id 0 — the all-false-mask class from the other
    // side. The reserved id must still be the only true.
    for ([_]usize{ 0, 16, 99 }) |defined| {
        fillSuppressMask(&buf, &[_]u32{3}, defined);
        for (buf, 0..) |v, i| try testing.expectEqual(i == 3, v);
    }
}

test "suppress_mask: a suppressed id is unreachable from both samplers, everything else stays" {
    // A collapsed distribution can rank `<|fim_hole|>` (a reserved FIM marker) in the
    // top-5 at every degenerate position, and greedy DREW it live. A reserved
    // marker in chat output is always a bug, so the sampler carries a
    // model-lifetime suppression mask. Contract pinned here: with the mask,
    // neither sampler can return a suppressed id on any of its paths (greedy
    // fast path, greedy general, categorical, sync sampleToken); with a null
    // mask the same logits DO pick it (red-on-revert); and logprobs keep
    // reporting the MODEL's raw distribution — the suppressed id still ranks
    // where the model put it, because the field reports the model while the
    // mask is sampling policy.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const V: usize = 8;
    const mask = try buildSuppressMask(&[_]u32{3}, 0, V, s);
    defer _ = mlx.mlx_array_free(mask);

    // id 3 (suppressed) dominates, id 5 is the runner-up, the rest are so far
    // down that after masking the categorical underflows to exactly id 5.
    var data = [_]f32{ -1000, -1000, -1000, 100, -1000, 50, -1000, -1000 };

    // Greedy fast path: [1, 1, V].
    const shape3 = [_]c_int{ 1, 1, V };
    const logits3 = mlx.mlx_array_new_data(&data, &shape3, 3, .float32);
    defer _ = mlx.mlx_array_free(logits3);
    const masked_sp = SamplingParams{ .temperature = 0.0, .suppress_mask = mask };
    const open_sp = SamplingParams{ .temperature = 0.0 };
    try testing.expectEqual(@as(u32, 5), try evalLazyToken(sampleTokenLazy(logits3, masked_sp, s)));
    try testing.expectEqual(@as(u32, 3), try evalLazyToken(sampleTokenLazy(logits3, open_sp, s)));

    // Categorical path at temp 1.0 (exercises the seq_len==1 reshape arm and
    // the sampler proper; the masked lane's probability is exactly 0 and the
    // -1000 lanes underflow, so the draw is deterministic).
    const masked_t1 = SamplingParams{ .temperature = 1.0, .suppress_mask = mask };
    try testing.expectEqual(@as(u32, 5), try evalLazyToken(sampleTokenLazy(logits3, masked_t1, s)));

    // Sync sampler (logprobs/penalty path): same policy, and the reported
    // distribution stays the model's own — rank 1 is the SUPPRESSED id.
    const r = try sampleToken(allocator, logits3, masked_sp, null, 2, null, s);
    const lp = r.logprob_result orelse return error.NoLogprobs;
    defer allocator.free(lp.top_logprobs);
    try testing.expectEqual(@as(u32, 5), r.token_id);
    try testing.expectEqual(@as(u32, 3), lp.top_logprobs[0].token_id);

    // Stochastic-verify filters: a suppressed draft's acceptance probability
    // must be exactly 0 (always rejected), and the residual it corrects from
    // can never re-draw it.
    const probs = try probsAtLastPos(logits3, masked_t1, s);
    defer _ = mlx.mlx_array_free(probs);
    try testing.expectEqual(@as(f32, 0.0), try probAt(probs, 3, s));
    try testing.expect(try probAt(probs, 5, s) > 0.99);
}

test "computeLogprobs: bf16 logits report the exact log-probability" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const raw = [_]f32{ 5, 5, 5, -20 };
    const a32 = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(a32);
    var a16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a16);
    try mlx.check(mlx.mlx_astype(&a16, a32, .bfloat16, s));
    const r = try computeLogprobs(testing.allocator, a16, 3, 4, s);
    defer testing.allocator.free(r.top_logprobs);
    // -26.0986; the bf16 grid reads -26.125 there.
    try testing.expectApproxEqAbs(-20.0 - (5.0 + @log(@as(f32, 3.0))), r.token_logprob, 1e-3);
}

test "computeLogprobs: rank 1 is the argmax and ranks descend, under a tie-saturated distribution" {
    // The producer used to recover token ids by scanning the whole vocab for
    // FLOAT EQUALITY with each `mlx_topk` value. Ties make that ambiguous, and
    // at temp 0 the post-temperature distribution saturates so ties are
    // everywhere — measured 0/5 positions where rank 1 was the chosen token on
    // a trivial prompt. The observable contract: with greedy sampling the
    // chosen token IS the argmax, so rank 1 must equal it.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    // Six exactly-tied entries, then a strict runner-up and a strict max.
    const data = [_]f32{ 1, 1, 1, 1, 1, 1, 2, 3 };
    const shape = [_]c_int{ 1, 8 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const chosen: u32 = 7; // argmax
    const r = try computeLogprobs(allocator, logits, chosen, 3, s);
    defer allocator.free(r.top_logprobs);

    try testing.expectEqual(@as(usize, 3), r.top_logprobs.len);
    try testing.expectEqual(chosen, r.top_logprobs[0].token_id);
    try testing.expectEqual(@as(u32, 6), r.top_logprobs[1].token_id);
    try testing.expect(r.top_logprobs[2].token_id < 6);
    // Strictly descending, and ids distinct.
    try testing.expect(r.top_logprobs[0].logprob > r.top_logprobs[1].logprob);
    try testing.expect(r.top_logprobs[1].logprob > r.top_logprobs[2].logprob);
    try testing.expect(r.top_logprobs[0].token_id != r.top_logprobs[1].token_id);
    try testing.expect(r.top_logprobs[1].token_id != r.top_logprobs[2].token_id);
    // The chosen token's own logprob must agree with its rank-1 entry.
    try testing.expectApproxEqAbs(r.top_logprobs[0].logprob, r.token_logprob, 1e-6);
}

test "sampleToken: reported logprobs are the model's, not the client's temperature" {
    // Same prompt, same chosen token, three temperatures used to report three
    // different logprobs (-0.2129 / -0.0607 / -2.1566 at 0 / 0.6 / 2.0). The
    // value belongs to the model, so it must not move with a sampling knob.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const data = [_]f32{ 0, 1, 2, 3 };
    const shape = [_]c_int{ 1, 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // log_softmax([0,1,2,3])[3], computed independently.
    var denom: f64 = 0;
    for (data) |v| denom += @exp(@as(f64, v));
    const expect_top: f32 = @floatCast(3.0 - @log(denom));

    const temps = [_]f32{ 0.0, 0.6, 2.0 };
    for (temps) |t| {
        const sp = SamplingParams{ .temperature = t, .seed = 7 };
        const r = try sampleToken(allocator, logits, sp, null, 2, null, s);
        const lp = r.logprob_result orelse return error.NoLogprobs;
        defer allocator.free(lp.top_logprobs);
        try testing.expectEqual(@as(usize, 2), lp.top_logprobs.len);
        try testing.expectEqual(@as(u32, 3), lp.top_logprobs[0].token_id);
        try testing.expectEqual(@as(u32, 2), lp.top_logprobs[1].token_id);
        try testing.expectApproxEqAbs(expect_top, lp.top_logprobs[0].logprob, 1e-5);
    }
}

/// Deterministic Gaussian logits (Box-Muller over an xorshift stream): diffuse
/// enough that the nucleus for top_p 0.95 spans thousands of tokens, which is
/// where a narrow cumulative accumulator loses the tail.
fn samplerTestRow(host: []f32, seed: u64, sigma: f32) void {
    var rng: u64 = seed | 1;
    var i: usize = 0;
    while (i < host.len) : (i += 2) {
        var u: [2]f64 = undefined;
        for (&u) |*d| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            d.* = (@as(f64, @floatFromInt(rng >> 11)) + 1.0) / 9007199254740994.0;
        }
        const r = @sqrt(-2.0 * @log(u[0]));
        const a = 6.283185307179586 * u[1];
        host[i] = @floatCast(r * @cos(a) * @as(f64, sigma));
        if (i + 1 < host.len) host[i + 1] = @floatCast(r * @sin(a) * @as(f64, sigma));
    }
}

/// Nucleus the sampler is meant to keep, in f64: ranked descending, a column is
/// in when less than top_p of the mass outranks it.
fn samplerTestExactKeep(allocator: std.mem.Allocator, row: []const f32, top_p: f32) !usize {
    const sorted = try allocator.alloc(f64, row.len);
    defer allocator.free(sorted);
    for (row, sorted) |x, *d| d.* = @floatCast(x);
    std.mem.sort(f64, sorted, {}, comptime std.sort.desc(f64));
    const mx = sorted[0];
    var z: f64 = 0;
    for (sorted) |d| z += @exp(d - mx);
    var above: f64 = 0;
    var keep: usize = 0;
    for (sorted) |d| {
        if (above >= @as(f64, @floatCast(top_p))) break;
        above += @exp(d - mx) / z;
        keep += 1;
    }
    return keep;
}

/// Read an mlx array back as f32 host values. A slice is a strided VIEW, so the
/// read goes through a contiguous copy and a flat reshape, never the base
/// pointer.
fn samplerTestReadFlat(allocator: std.mem.Allocator, arr: mlx.mlx_array, n: usize, s: mlx.mlx_stream) ![]f32 {
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, arr, .float32, s));
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_contiguous(&dense, as_f32, false, s));
    const flat_shape = [_]c_int{@intCast(n)};
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, dense, &flat_shape, 1, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    const out = try allocator.alloc(f32, n);
    @memcpy(out, mlx.mlx_array_data_float32(flat).?[0..n]);
    return out;
}

/// Bit equality of two mlx arrays read back as f32: both routes end in one
/// `where` over the same logits, so agreeing on the kept columns is agreeing on
/// every byte.
fn samplerTestExpectSameBits(a: mlx.mlx_array, b: mlx.mlx_array, n: usize, s: mlx.mlx_stream) !void {
    const ad = try samplerTestReadFlat(testing.allocator, a, n, s);
    defer testing.allocator.free(ad);
    const bd = try samplerTestReadFlat(testing.allocator, b, n, s);
    defer testing.allocator.free(bd);
    for (ad, bd, 0..) |x, y, i| {
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
            std.debug.print("mismatch at {d}: {d} vs {d}\n", .{ i, x, y });
            return error.FilteredLogitsDiffer;
        }
    }
}

fn samplerTestKeptCount(row: []const f32) usize {
    var n: usize = 0;
    for (row) |x| {
        if (x != -std.math.inf(f32)) n += 1;
    }
    return n;
}

test "sampler ranks break ties by the lowest original index" {
    // The contract both filters and both of their routes share. mlx's stable
    // merge sort is what delivers it; if a bump breaks stability this fails.
    const s = mlx.gpuStream();
    const data = [_]f32{ 2.0, 5.0, 5.0, 5.0, 1.0, 5.0 };
    const shape = [_]c_int{ 1, 6 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    var ranks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ranks);
    try ranksDescending(&ranks, logits, s);
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, ranks, .float32, s));
    const got = try samplerTestReadFlat(testing.allocator, as_f32, 6, s);
    defer testing.allocator.free(got);
    // The four tied 5.0s come first in ascending column order, then 2.0, then 1.0.
    try testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 5, 0, 4 }, got);
}

test "applyTopP keeps the exact nucleus on a bf16 vocab row" {
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 248320;
    const host = try allocator.alloc(f32, v);
    defer allocator.free(host);
    const rounded = try allocator.alloc(f32, v);
    defer allocator.free(rounded);

    for ([_]u64{ 1, 2, 3, 4, 5, 6 }) |seed| {
        samplerTestRow(host, seed *% 0x9E3779B97F4A7C15, 4.0);
        const shape = [_]c_int{ 1, @intCast(v) };
        const f32_row = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(f32_row);
        var bf16_row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bf16_row);
        try mlx.check(mlx.mlx_astype(&bf16_row, f32_row, .bfloat16, s));

        // The reference reads the row the kernel sees: bf16-rounded values.
        var rt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rt);
        try mlx.check(mlx.mlx_astype(&rt, bf16_row, .float32, s));
        const rt_host = try samplerTestReadFlat(allocator, rt, v, s);
        defer allocator.free(rt_host);
        @memcpy(rounded, rt_host);
        const want = try samplerTestExactKeep(allocator, rounded, 0.95);

        var res = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(res);
        try applyTopP(&res, bf16_row, 0.95, 0, s);
        const res_host = try samplerTestReadFlat(allocator, res, v, s);
        defer allocator.free(res_host);
        // The boundary column's own probability is ~1e-5 here while the f32 scan
        // carries ~1e-7 of error, so the exclusive mass can cross top_p one
        // column early or late against the f64 reference. The defect this pins
        // was 148 columns wide.
        const got = samplerTestKeptCount(res_host);
        const diff = @as(i64, @intCast(want)) - @as(i64, @intCast(got));
        try testing.expect(diff >= -1 and diff <= 1);
    }
}

test "applyTopP at top_p 0 keeps exactly the argmax" {
    // The nucleus is "mass strictly above the rank < top_p"; rank 0 sees 0, so a
    // literal 0 kept nothing and the sampler drew uniformly from the vocabulary.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 8192;
    const host = try allocator.alloc(f32, v);
    defer allocator.free(host);
    samplerTestRow(host, 0x9E3779B97F4A7C15, 4.0);
    const shape = [_]c_int{ 1, @intCast(v) };
    const row = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(row);
    for ([_]u32{ 0, 40 }) |bound| {
        var res = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(res);
        try applyTopP(&res, row, 0.0, bound, s);
        const res_host = try samplerTestReadFlat(allocator, res, v, s);
        defer allocator.free(res_host);
        try testing.expectEqual(@as(usize, 1), samplerTestKeptCount(res_host));
        var best: usize = 0;
        for (host, 0..) |x, i| if (x > host[best]) {
            best = i;
        };
        try testing.expect(std.math.isFinite(res_host[best]));
    }
}

test "topRanksDescending returns the exact top m values and column ids" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    for ([_]usize{ 8192, 248320 }) |v| {
        const rows: usize = 3;
        const host = try allocator.alloc(f32, rows * v);
        defer allocator.free(host);
        for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 1) *% 0xD1B54A32D192ED03, 4.0);
        // Row 2 quantized to 0.25 steps: thousands of exact ties.
        for (host[2 * v ..][0..v]) |*x| x.* = @round(x.* * 4.0) / 4.0;
        const shape = [_]c_int{ @intCast(rows), @intCast(v) };
        const block = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(block);

        for ([_]u32{ 1, 20, 257 }) |m| {
            var vals = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(vals);
            var ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ids);
            try topRanksDescending(&vals, &ids, block, m, s);
            // The whole row's ranking, restricted to its first m: the bar the
            // shortlist has to reproduce, ties included.
            var order = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(order);
            try ranksDescending(&order, block, s);
            var ref_ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref_ids);
            try sliceLastAxis(&ref_ids, order, 0, m, s);
            var ref_vals = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref_vals);
            try mlx.check(mlx.mlx_take_along_axis(&ref_vals, block, ref_ids, -1, s));
            var ids_f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ids_f);
            try mlx.check(mlx.mlx_astype(&ids_f, ids, .float32, s));
            var ref_ids_f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref_ids_f);
            try mlx.check(mlx.mlx_astype(&ref_ids_f, ref_ids, .float32, s));
            try samplerTestExpectSameBits(vals, ref_vals, rows * m, s);
            try samplerTestExpectSameBits(ids_f, ref_ids_f, rows * m, s);
        }
    }
}

test "applyTopK keeps exactly k columns, the highest ranked ones" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 248320;
    const rows: usize = 3;
    const host = try allocator.alloc(f32, rows * v);
    defer allocator.free(host);
    for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 7) *% 0xD1B54A32D192ED03, 4.0);
    // Row 1 quantized to 1.0 steps: a value cutoff would keep thousands here.
    for (host[v..][0..v]) |*x| x.* = @round(x.*);
    const shape = [_]c_int{ @intCast(rows), @intCast(v) };
    const block = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(block);

    for ([_]u32{ 1, 20, 64 }) |k| {
        var res = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(res);
        try applyTopK(&res, block, k, s);
        const rd = try samplerTestReadFlat(allocator, res, rows * v, s);
        defer allocator.free(rd);
        for (0..rows) |r| {
            try testing.expectEqual(@as(usize, k), samplerTestKeptCount(rd[r * v ..][0..v]));
        }
        // Same columns as ranking the whole row.
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try ranksDescending(&order, block, s);
        var ref_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref_ids);
        try sliceLastAxis(&ref_ids, order, 0, k, s);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try keepByIndex(&ref, block, ref_ids, s);
        try samplerTestExpectSameBits(res, ref, rows * v, s);
    }
}

test "shortlist top-p equals the whole row's ranking, ties and 248320-wide rows" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    for ([_]usize{ 8192, 248320 }) |v| {
        const rows: usize = 4;
        const host = try allocator.alloc(f32, rows * v);
        defer allocator.free(host);
        for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 11) *% 0xD1B54A32D192ED03, 4.0);
        // Row 2: 0.25-step ties. Row 3: every column identical — the extreme the
        // rank contract exists for.
        for (host[2 * v ..][0..v]) |*x| x.* = @round(x.* * 4.0) / 4.0;
        for (host[3 * v ..][0..v]) |*x| x.* = 1.5;

        const shape = [_]c_int{ @intCast(rows), @intCast(v) };
        const block = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(block);

        for ([_]struct { p: f32, k: u32 }{
            .{ .p = 0.95, .k = 20 },
            .{ .p = 0.8, .k = 50 },
            .{ .p = 0.99, .k = 5 },
            .{ .p = 0.5, .k = 1 },
            .{ .p = 0.95, .k = 257 },
            .{ .p = 1e-6, .k = 20 },
        }) |cfg| {
            var masked = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(masked);
            try applyTopK(&masked, block, cfg.k, s);

            var short = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(short);
            try applyTopP(&short, masked, cfg.p, cfg.k, s);
            var full = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(full);
            try applyTopP(&full, masked, cfg.p, 0, s);
            try samplerTestExpectSameBits(short, full, rows * v, s);

            // Same filtered logits and the same key must draw the same token.
            for (0..8) |d| {
                const key = seedKey(.{ .seed = 12345, .draw = @intCast(d) });
                defer _ = mlx.mlx_array_free(key);
                var a = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(a);
                var b = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(b);
                try mlx.check(mlx.mlx_random_categorical(&a, short, -1, key, s));
                try mlx.check(mlx.mlx_random_categorical(&b, full, -1, key, s));
                try samplerTestExpectSameBits(a, b, rows, s);
            }
        }
    }
}

test "shortlist top-p holds on rank-3 [B, L, V] blocks" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 8192;
    const rows: usize = 3;
    const host = try allocator.alloc(f32, rows * v);
    defer allocator.free(host);
    for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 23) *% 0xD1B54A32D192ED03, 4.0);
    const shape = [_]c_int{ 1, @intCast(rows), @intCast(v) };
    const block = mlx.mlx_array_new_data(host.ptr, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(block);

    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    try applyTopK(&masked, block, 20, s);
    var short = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(short);
    try applyTopP(&short, masked, 0.95, 20, s);
    var full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(full);
    try applyTopP(&full, masked, 0.95, 0, s);
    try samplerTestExpectSameBits(short, full, rows * v, s);

    // Per-ROW cutoffs: a block-wide filter would mask whole rows to -inf.
    const sd = try samplerTestReadFlat(allocator, short, rows * v, s);
    defer allocator.free(sd);
    for (0..rows) |r| {
        const kept = samplerTestKeptCount(sd[r * v ..][0..v]);
        try testing.expect(kept > 0 and kept <= 20);
    }
}

test "a group's joined probability block slices row-for-row into the per-row filter" {
    // Ragged widths, three rows: the axis-bug class ("one row cannot reveal an
    // axis bug"), and the offsets the projection's own walk produces.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 8192;
    const widths = [_]u32{ 3, 2, 4 };
    const sampling = SamplingParams{ .temperature = 1.0, .top_p = 0.95, .top_k = 20 };

    var rows: [widths.len]mlx.mlx_array = undefined;
    var built: usize = 0;
    defer for (rows[0..built]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (widths, 0..) |w, r| {
        const host = try allocator.alloc(f32, @as(usize, w) * v);
        defer allocator.free(host);
        for (0..w) |p| samplerTestRow(host[p * v ..][0..v], (r * 8 + p + 1) *% 0xD1B54A32D192ED03, 4.0);
        const shape = [_]c_int{ 1, @intCast(w), @intCast(v) };
        const f32_rows = mlx.mlx_array_new_data(host.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(f32_rows);
        rows[r] = mlx.mlx_array_new();
        built = r + 1;
        // bf16 like the verify block the lm_head produces.
        try mlx.check(mlx.mlx_astype(&rows[r], f32_rows, .bfloat16, s));
    }

    var joined = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(joined);
    {
        const vec = mlx.mlx_vector_array_new_data(&rows, widths.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 1, s));
    }
    const block = try groupProbsBlock(joined, sampling, s);
    defer _ = mlx.mlx_array_free(block);

    var offset: c_int = 0;
    for (widths, 0..) |w, r| {
        const got = try groupProbsRow(block, offset, w, s);
        defer _ = mlx.mlx_array_free(got);
        const want = try probsAllPositions(rows[r], sampling, s);
        defer _ = mlx.mlx_array_free(want);
        const g_shape = mlx.getShape(got);
        try testing.expectEqual(@as(c_int, 1), g_shape[0]);
        try testing.expectEqual(@as(c_int, @intCast(w)), g_shape[1]);
        try samplerTestExpectSameBits(got, want, @as(usize, w) * v, s);
        offset += @intCast(w);
    }
}

test "the group accept block declines a mixed sampler and any seeded row" {
    const a = SamplingParams{ .temperature = 1.0, .top_p = 0.95, .top_k = 20 };
    var b = a;
    b.top_k = 40;
    var seeded = a;
    seeded.seed = 7;
    try testing.expect(Generator.mtpGroupAcceptBatchable(&.{ a, a }));
    try testing.expect(!Generator.mtpGroupAcceptBatchable(&.{ a, b }));
    try testing.expect(!Generator.mtpGroupAcceptBatchable(&.{ a, seeded }));
    try testing.expect(!Generator.mtpGroupAcceptBatchable(&.{}));
}

/// The composition every sampling site ran before `filterTopKTopP`: top-k, then
/// top-p bounded by that same k. The reference the one-pass filter is pinned to.
fn samplerTestComposed(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, top_k: u32, s: mlx.mlx_stream) !void {
    var after_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(after_k);
    if (top_k > 0) {
        try applyTopK(&after_k, logits, top_k, s);
    } else {
        try mlx.check(mlx.mlx_array_set(&after_k, logits));
    }
    if (top_p < 1.0) return applyTopP(res, after_k, top_p, top_k, s);
    return mlx.check(mlx.mlx_array_set(res, after_k));
}

test "filterTopKTopP is byte-identical to the two-pass composition" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const Setting = struct { p: f32, k: u32 };
    const settings = [_]Setting{
        .{ .p = 0.95, .k = 20 }, .{ .p = 0.9, .k = 1 },  .{ .p = 1.0, .k = 20 },
        .{ .p = 0.95, .k = 0 },  .{ .p = 0.5, .k = 50 }, .{ .p = 0.95, .k = 999999 },
    };
    for ([_]usize{ 8192, 248320 }) |v| {
        const rows: usize = 3;
        const host = try allocator.alloc(f32, rows * v);
        defer allocator.free(host);
        samplerTestRow(host[0..v], 0x9E3779B97F4A7C15, 4.0);
        // Row 1 quantized to 0.25 steps: thousands of exact ties at the cutoff.
        samplerTestRow(host[v..][0..v], 0xD1B54A32D192ED03, 4.0);
        for (host[v..][0..v]) |*x| x.* = @round(x.* * 4.0) / 4.0;
        // Row 2 all equal: every column ties, so the ranking alone decides.
        @memset(host[2 * v ..][0..v], 1.25);

        // `[m, V]` and `[1, L, V]`: the mask must stay per ROW on both ranks.
        for ([_]usize{ 2, 3 }) |rank| {
            const shape2 = [_]c_int{ @intCast(rows), @intCast(v) };
            const shape3 = [_]c_int{ 1, @intCast(rows), @intCast(v) };
            const shape: []const c_int = if (rank == 2) &shape2 else &shape3;
            const f32_block = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f32_block);
            var block = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(block);
            try mlx.check(mlx.mlx_astype(&block, f32_block, .bfloat16, s));
            for (settings) |cfg| {
                var want = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(want);
                try samplerTestComposed(&want, block, cfg.p, cfg.k, s);
                var got = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(got);
                try filterTopKTopP(&got, block, cfg.p, cfg.k, s);
                samplerTestExpectSameBits(got, want, rows * v, s) catch |e| {
                    std.debug.print("v={d} rank={d} top_p={d} top_k={d}\n", .{ v, rank, cfg.p, cfg.k });
                    return e;
                };
            }
        }
    }
}

test "no block decoder commits an argmax over a suppressed id" {
    // A row whose raw argmax is a padding id past the defined vocabulary, plus a
    // flagged special: both must lose to the best legal column. Run at the block
    // widths the committing decoders verify at (MTP and the gemma drafter `1 + m`,
    // DFlash a whole block), so a per-position axis bug cannot hide. WHICH decoders
    // are covered is a compile-time property: every commit site takes a
    // `CommittedArgmax`, and only `verifyArgmax` returns one.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 512;
    const defined: usize = 500;
    const flagged = [_]u32{7};
    for ([_]usize{ 2, 3, 5 }) |rows| try suppressedArgmaxCase(allocator, s, v, defined, &flagged, rows);
}

fn suppressedArgmaxCase(allocator: std.mem.Allocator, s: mlx.mlx_stream, v: usize, defined: usize, flagged: []const u32, rows: usize) !void {
    const host = try allocator.alloc(f32, rows * v);
    defer allocator.free(host);
    @memset(host, 0.0);
    for (0..rows) |r| {
        const row = host[r * v ..][0..v];
        row[505] = 9.0; // padding row: highest of all
        row[7] = 8.0; // flagged special: second
        row[100 + r] = 7.0; // the best LEGAL id, different per position
    }
    const shape = [_]c_int{ 1, @intCast(rows), @intCast(v) };
    const f32_block = mlx.mlx_array_new_data(host.ptr, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(f32_block);
    var block = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(block);
    try mlx.check(mlx.mlx_astype(&block, f32_block, .bfloat16, s));
    const mask = try buildSuppressMask(flagged, defined, v, s);
    defer _ = mlx.mlx_array_free(mask);

    var raw = try verifyArgmax(block, null, s);
    defer raw.deinit();
    const raw_ids = try samplerTestReadFlat(allocator, raw.lazy(), rows, s);
    defer allocator.free(raw_ids);
    for (raw_ids) |id| try testing.expectEqual(@as(f32, 505), id);

    var masked = try verifyArgmax(block, mask, s);
    defer masked.deinit();
    const ids = try samplerTestReadFlat(allocator, masked.lazy(), rows, s);
    defer allocator.free(ids);
    for (ids, 0..) |id, r| try testing.expectEqual(@as(f32, @floatFromInt(100 + r)), id);
}

test "serial and MTP agree at the greedy temperature cutoff" {
    const cutoff: f32 = 0.01;
    const bits: u32 = @bitCast(cutoff);
    for ([_]f32{ 0.0, 0.005, @bitCast(bits - 1), cutoff, @bitCast(bits + 1), 0.6 }) |temperature| {
        const sampling = SamplingParams{ .temperature = temperature };
        const serial_greedy = sampleRowsAllGreedy(&.{sampling});
        const mtp_greedy = Generator.mtpDraftSamplingFor(sampling, false, 0.6).temperature == 0.0;
        try testing.expectEqual(temperature < cutoff, serial_greedy);
        try testing.expectEqual(serial_greedy, mtp_greedy);
    }
}

test "upstream bugfix: constrained logprobs describe the returned token before masking" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const s = mlx.gpuStream();
    // Each call is the final allowed token, so no model forward is needed.
    var xfm: Transformer = undefined;
    xfm.s = s;
    var bytes = try token_mask.TokenBytes.init(std.heap.ArenaAllocator.init(a), &.{ null, "true", "false", "invalid" }, null);
    defer bytes.deinit();
    const parsed = try std.json.parseFromSlice(std.json.Value, a, "{\"type\":\"boolean\"}", .{});
    defer parsed.deinit();
    var gen = Generator{
        .xfm = &xfm,
        .ctx = undefined,
        .tok = undefined,
        .next_token_id = 0,
        .step = 0,
        .max_tokens = 1,
        .sampling = .{ .temperature = 0 },
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .finish_reason = "length",
        .done = false,
        .eos_token_ids = &.{},
        .generated_ids = .empty,
        .timeout_ns = 0,
        .timer = io_util.Stopwatch.init(testing.io),
        .last_hidden = .{ .ctx = null },
        .has_last_hidden = false,
        .logprobs_n = 4,
    };
    defer gen.generated_ids.deinit(a);
    defer if (gen.last_logprob) |lp| a.free(lp.top_logprobs);
    defer if (gen.has_pending_logits) {
        _ = mlx.mlx_array_free(gen.pending_logits);
    };
    for ([_][4]f32{ .{ 0, 2, 1, 4 }, .{ 0, 1, 3, 5 } }, [_]u32{ 1, 2 }) |values, want| {
        var sc: SchemaConstraint = undefined;
        try sc.initFromValue(a, parsed.value, &bytes);
        defer sc.deinit();
        gen.sampling.constraint = &sc.constraint;
        gen.max_tokens = gen.step + 1;
        gen.done = false;
        gen.pending_logits = mlx.mlx_array_new_data(&values, &.{ 1, 1, 4 }, 3, .float32);
        gen.has_pending_logits = true;
        const expected = try computeLogprobs(a, gen.pending_logits, want, 4, s);
        defer a.free(expected.top_logprobs);
        try testing.expectEqual(@as(?u32, want), try gen.nextConstrained(a));
        const got = gen.last_logprob orelse return error.NoLogprobs;
        try testing.expectApproxEqAbs(expected.token_logprob, got.token_logprob, 1e-6);
        try testing.expectEqual(@as(u32, 3), got.top_logprobs[0].token_id);
        try testing.expectEqual(@as(?u32, null), try gen.nextConstrained(a));
    }
    try testing.expectEqualSlices(u32, &.{ 1, 2 }, gen.generated_ids.items);
}

test "argmax-only requests: greedy with nothing that reads past the argmax" {
    const greedy: SamplingParams = .{ .temperature = 0.0 };
    try std.testing.expect(argmaxOnlyRequest(greedy, 0));
    try std.testing.expect(argmaxOnlyRequest(.{ .temperature = 0.8, .top_k = 1 }, 0));
    try std.testing.expect(!argmaxOnlyRequest(.{ .temperature = 0.7 }, 0));
    try std.testing.expect(!argmaxOnlyRequest(.{ .temperature = 0.0, .repeat_penalty = 1.1 }, 0));
    try std.testing.expect(!argmaxOnlyRequest(.{ .temperature = 0.0, .presence_penalty = 0.5 }, 0));
    try std.testing.expect(!argmaxOnlyRequest(.{ .temperature = 0.0, .think_penalty = .{ .lambda = 1 } }, 0));
    try std.testing.expect(!argmaxOnlyRequest(greedy, 5));
    var cf: CallForce = undefined;
    try std.testing.expect(!argmaxOnlyRequest(.{ .temperature = 0.0, .call_force = &cf }, 0));
    var c: Constraint = undefined;
    try std.testing.expect(!argmaxOnlyRequest(.{ .temperature = 0.0, .constraint = &c }, 0));
}

test "a lazy penalty allocation failure fails the request instead of sampling unpenalized" {
    const s = mlx.gpuStream();
    const host = [_]f32{ 5, 4, 3, 2 };
    const logits = mlx.mlx_array_new_data(&host, &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    for (0..1) |fail_index| {
        var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = fail_index });
        try testing.expectError(error.OutOfMemory, penalizedSampleLazy(failing.allocator(), logits, .{ .temperature = 0, .presence_penalty = 2 }, &.{0}, s));
    }
}

test "logit bias CPU: rewards and penalties require the full head while zero stays off" {
    const t = std.testing;
    for ([_]f32{ -2, 2 }) |delta| {
        const entries = [_]logit_bias.Bias{.{ .id = 1, .delta = delta }};
        const sampling = SamplingParams{ .temperature = 0, .think_penalty = .{ .biases = &entries } };
        try t.expect(!argmaxOnlyRequest(sampling, 0));
        var from_file = sampling;
        from_file.think_penalty.biases = &.{};
        from_file.think_penalty.file_biases = &entries;
        try t.expect(!argmaxOnlyRequest(from_file, 0));
    }
    const zero = [_]logit_bias.Bias{.{ .id = 1, .delta = 0 }};
    try t.expect(argmaxOnlyRequest(.{ .temperature = 0, .think_penalty = .{ .biases = &zero } }, 0));
    try t.expect(argmaxOnlyRequest(.{ .temperature = 0 }, 0));
}
