//! Hy4-Flash's model: the release's tensors resolved by name once, and the composed forward (`hy4_forward.zig`) run
//! over a request's state. `forward(ids)` extends the state by the rows of `ids` and gives their logits: a prefill is
//! one call, each decode step another. The state per layer is its latent rows (`c` and the rotated `k_pe`) and, on a
//! sparse layer, its token keys; the request keeps its Engram history. Every array lives on the stream the model was
//! made with, so the same model runs on Sushi Cooler's stream and on the CPU stream the oracle tests use.
const std = @import("std");
const plan = @import("../tensor/plan/plan_api.zig");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const engram_mod = @import("../engram.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const mlx_pages = @import("../mlx_pages.zig");
const expert_stream = @import("../expert_stream.zig");
const kv_quant = @import("../kv_quant.zig");
const row_store = @import("../row_store.zig");
const fwd = @import("hy4_forward.zig");
const hy4_config = @import("hy4_config.zig");
const hy4_latent = @import("hy4_latent.zig");
const hy4_attn = @import("hy4_attn.zig");
const hy4_stq = @import("hy4_stq.zig");
const exl3 = @import("sushi_exl3");
const Arr = mlx.mlx_array;

const max_layers = hy4_config.max_layers;
const none: Arr = .{ .ctx = null };
/// Rows of a chunk one attention span takes: a span's scores and gathered rows are freed before the next.
pub const attn_span: c_int = 256;
/// Rows a span takes where each row gathers its own recalled keys (a sparse layer past its selection threshold):
/// 2,048 latent rows a row on the release, widened, so a narrower span.
pub const gathered_span: c_int = 32;
/// The widest chunk a forward takes, in spans at the gathered span.
const max_spans = hy4_config.widest_chunk / gathered_span;

const Ffn = union(enum) { dense: struct { sw: fwd.Swiglu, post_norm: Arr }, moe: fwd.Moe };

const EngramW = struct {
    /// The module's index in the hash (`Hash.rows` writes its columns at this slot).
    module: usize,
    /// The table resident in the pack's tensors, or on SSD read by row (a streamed load's).
    table: Arr,
    disk: ?engram_mod.Table = null,
    tensors: fwd.Engram,
    norms: [16]Arr,
    /// Per stream: key projection weight and bias, key norm, query norm, mix scale.
    streams: [5][8]Arr,
};

const Layer = struct {
    attn: fwd.Attn,
    geometry: fwd.AttnGeometry,
    indexer: ?fwd.Indexer,
    mix_attn: fwd.Mix,
    mix_ffn: fwd.Mix,
    input_norm: Arr,
    post_norm: Arr,
    ffn: Ffn,
    engram: ?*EngramW,
};

/// One layer's request state. A sliding layer's ring holds its latent rows (BF16, or affine-quantized codes, scales and
/// biases at the 8-bit width) and rotated rope rows (BF16) from position `base` on, the last `window - 1`, which no later
/// row reads past. A sparse layer keeps every row in `rows`, grown in place (`row_store`), the pooled key of each closed
/// group in `pooled`, and the token keys of its open group (fewer than `ratio`) in `open`.
const LayerState = struct {
    c: ?Arr = null,
    c_scales: ?Arr = null,
    c_biases: ?Arr = null,
    k_pe: ?Arr = null,
    base: c_int = 0,
    /// A sparse layer's rows as one store's parts: the latent's codes, scales and biases then the rope rows at the
    /// 8-bit width; the latent then the rope rows at BF16 (`sparseKeys`).
    rows: ?row_store.Store = null,
    pooled: ?row_store.Store = null,
    /// The open group: a view of its first rows in `open_buf`, the layer's fixed block of `ratio - 1` token keys, written
    /// in place each forward (docs/bill_redesign.md part 8).
    open: ?Arr = null,
    open_buf: ?Arr = null,
    /// A sliding layer's rows are a ring of `window` rows, position `pos` at row `pos % window`; `ring` counts the
    /// positions written. A window handed to the spans in ring order (a decode row, which reads every row) is
    /// `unordered`.
    ring: c_int = 0,
    unordered: bool = false,
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    config: hy4_config.Config,
    s: mlx.mlx_stream,
    layers: [max_layers]Layer = undefined,
    /// The embedding [vocab, D] (BF16), or `none` when an STQ pack stores it as a trellis bank, read by 128-id block
    /// (`embed_gpu`, or `embed_gather` off the GPU).
    embed: Arr,
    embed_gather: ?hy4_stq.EmbedGather = null,
    /// A trellis embedding read on the GPU from its bank (`hy4_embed`), where the trellis decode kernel runs.
    embed_gpu: ?@import("hy4_embed.zig").Gather = null,
    /// The head [vocab, D] (BF16, widened per block for the f32 readout), or `none` when it is a trellis bank.
    lm_head: Arr,
    lm_head_stq: ?hy4_stq.Bank = null,
    final_norm: Arr,
    /// An STQ pack's stamps (`hy4_stq.Quant`); empty for a BF16 checkpoint.
    quant: hy4_stq.Quant = .{},
    /// The arrays an STQ load made: the banks' views, `kv_b` decoded, the expert banks' ragged storage.
    stq_owned: std.ArrayList(Arr) = .empty,
    stq_experts: [max_layers]?exl3.Ragged = @splat(null),
    /// The bytes held beside the pack's tensors: the banks decoded at load (`kv_b`) and the embedding gather's copy.
    stq_decoded_bytes: u64 = 0,
    head_mix: fwd.Mix,
    hash: ?engram_mod.Hash = null,
    /// The Engram token map, as compressed ids; an id at or past its length hashes as itself (llama.cpp).
    token_map: []u32 = &.{},
    engram_store: []EngramW = &.{},
    state: [max_layers]LayerState = @splat(.{}),
    /// The request's compressed ids so far, the Engram history.
    history: std.ArrayList(u32) = .empty,
    pos: usize = 0,
    /// The routed experts' stream when they stay on SSD (a streamed load's): the layer's banks are its slots for the
    /// picks of each forward. Null when every bank is resident in the pack's tensors.
    experts: ?*expert_stream.Engine = null,
    /// Set before `init` resolves the layers: the routed banks are not in the pack's tensors.
    streamed: bool = false,
    /// A served load's model (`Sources.served`): the host arrays the EXL3 kernels cached are its handle's to drop.
    served: bool = false,
    /// The latent rows' width: the engine's affine kv8 (the default, owner's ruling) or BF16 (`KVQuantConfig.dense`).
    /// Set before a request's first forward.
    kv: kv_quant.KVQuantConfig = .engine_default,
    /// Set by a drafter (`hy4_dfly`): each forward keeps its last row's hidden state, the head's collapse of the
    /// streams before the final norm (`last_hidden`, [1, D] f32), the DFly2 handover.
    keep_hidden: bool = false,
    /// A handle on `hidden_fixed`, the block the handover is written into in place, while one stands.
    last_hidden: ?Arr = null,
    hidden_fixed: ?Arr = null,
    /// With `keep_hidden`: each sparse layer's selection for the last row a forward ran past the indexer's threshold,
    /// the key rows `[width]` and which of them it reads `[1, width]`: the rows DFly2 shares (null below the threshold).
    last_picks: [max_layers]?struct { rows: Arr, keep: Arr } = @splat(null),
    /// The blocks each sparse layer's `last_picks` are written into in place, made at their first selection.
    picks_fixed: [max_layers][2]?Arr = @splat(.{ null, null }),
    /// Every expert of one layer held in the stream's slots (`pin`): that layer's picks run as one group, no SSD read.
    pinned: ?Pinned = null,
    /// A calibration tap, called only when set: the default path makes no call of its.
    tap: ?*const fwd.Tap = null,
    /// Rows an attention span takes (`attn_span`; a test narrows it to cross span edges on the tiny).
    span_rows: c_int = attn_span,
    /// The running request's restore points, which its finish hands to the cache's live record (`hy4_prefix.Live`).
    /// A reset drops them.
    live_points: std.ArrayList(Point) = .empty,
    /// The rows the state is reserved for (`reserveFor`), the running request's reach; 0 when none is.
    reach: usize = 0,
    /// Moves whenever the request's state does (`reset`, a layer's rows appended): a record of the state stands only
    /// while the version it was published at is the model's.
    version: u64 = 0,
    /// Moves when the state is replaced whole (`reset`, a restore of a stored point): a point of the state stands only
    /// while the epoch it was taken in is the model's and no rollback went below it.
    epoch: u64 = 0,
    /// A load on the static bill: its planned roots (`hy4_plan`); null while every forward runs on the pool.
    plans: ?*@import("hy4_plan.zig").Plans = null,
    /// The plans recorded before the warm (`hy4_plan.record`, D98), with no arena yet: `plans` once the warm's end makes
    /// it.
    recorded: ?*@import("hy4_plan.zig").Plans = null,
    /// The logits a planned forward hands its caller (D19): two blocks of a verify block's rows, written in turn, so the
    /// generator may hold one forward's logits while the next is made. Made with the plans.
    logits_fixed: [2]?Arr = .{ null, null },
    /// A planned load's forward input (`idsInput`): [1, `ids_rows`] u32, written in place by the host. Made with the
    /// plans.
    ids_fixed: ?Arr = null,
    logits_turn: u1 = 0,
    /// A round's mark (`mark`): blocks of each layer's rings and open group, in `point_fields`' layout, and of the
    /// handover, each its state block's shape, written in place. Made with the state where the request drafts
    /// (`reserveFor`), else at the first mark; one mark stands at a time.
    mark_fixed: [max_layers][point_arrays]?Arr = @splat(@splat(null)),
    mark_hidden: ?Arr = null,
    marked: bool = false,
    /// The `Points` slots (docs/bill_redesign.md part 8): `point_slots` equal blocks of a point's small state, laid out
    /// as `mark_fixed` and written in place as a mark is. Made with the request's state (`reserveFor`), else at a slot's
    /// first take; a reset lets every slot go. `point_take` names the take that holds a slot, 0 when it is free.
    point_fixed: [point_slots][max_layers][point_arrays]?Arr = @splat(@splat(@splat(null))),
    point_hidden: [point_slots]?Arr = @splat(null),
    point_take: [point_slots]u32 = @splat(0),
    point_takes: u32 = 0,
    /// The one-byte array a stored point's absent arrays share (`hy4_prefix.pack`), made at load.
    placeholder: ?Arr = null,

    /// A point within the running state (`point`): its position and the `Points` slot that holds its small state, in
    /// `mark`'s form (`held`, `open`, `hidden`). The rows below `pos` are the stores' own, which only grow past it until
    /// a rollback below it or a reset. `take` is the slot's take when the point was made: a point whose slot went to a
    /// later take (a reset, a release) stands no more.
    /// `disk_id`: the SSD tier entry the point was written to (`hy4_prefix`); whether the tier still holds it there is
    /// the tier's to say (`DiskTier.holdsPoint`).
    pub const Point = struct {
        pos: usize,
        slot: u8,
        take: u32,
        hidden: bool = false,
        held: [max_layers]u8 = @splat(0),
        open: [max_layers]u8 = @splat(0),
        disk_id: ?u64 = null,
    };
    pub const Pinned = struct { layer: u16, prepared: expert_stream.Prepared };

    /// Holds every routed expert of `layer` in the stream's slots until `unpin`, so a layer-at-a-time run reads each
    /// layer's experts once (`prepared.remapped` is indexed by expert id). The layer's cache must hold them all.
    pub fn pin(self: *Model, layer: u16) !void {
        const engine = self.experts orelse return error.Hy4PinNeedsStreamedExperts;
        self.unpin();
        const ids = try self.gpa.alloc(u16, self.config.experts);
        defer self.gpa.free(ids);
        for (ids, 0..) |*id, i| id.* = @intCast(i);
        self.pinned = .{ .layer = layer, .prepared = try engine.prepareHost(layer, ids) };
    }

    pub fn unpin(self: *Model) void {
        if (self.pinned) |*p| p.prepared.deinit();
        self.pinned = null;
    }
    /// Points a running request keeps for the live record (`hy4_prefix.Live`): its newest, each a ring set and open
    /// groups (`hy4_ledger.pointBytes`).
    pub const live_point_cap = 4;
    /// The `Points` slots: the running request's points and the live record's (`hy4_prefix.record_point_cap`).
    pub const point_slots = live_point_cap + @import("hy4_prefix.zig").record_point_cap;

    /// Resolves every tensor the forward reads from `weights` (the release's names; the arrays stay owned by
    /// `weights`, which must outlive the model). A tensor the geometry needs and the pack lacks is refused by name.
    pub fn init(gpa: std.mem.Allocator, config: hy4_config.Config, weights: *model_mod.Weights, s: mlx.mlx_stream) !*Model {
        return initWith(gpa, config, weights, s, .{});
    }

    /// What a streamed load reads from SSD: the routed experts through `experts` (which the caller keeps alive), and
    /// each Engram table the pack's tensors lack, by row from the checkpoint at `dir`.
    /// `served`: a served load, whose handle drops the EXL3 module's host arrays with the last model of its side
    /// (`arch/handle`); a model made directly drops them with itself.
    pub const Sources = struct { experts: ?*expert_stream.Engine = null, dir: ?[]const u8 = null, quant: hy4_stq.Quant = .{}, served: bool = false };

    pub fn initWith(gpa: std.mem.Allocator, config: hy4_config.Config, weights: *model_mod.Weights, s: mlx.mlx_stream, src: Sources) !*Model {
        // Its f32 products keep f32 on the engine's stream (`forwardReadout`) and the CPU's; MLX's GPU may take TF32.
        if (mlx.streamIsGpu(s) and !mlx.engine.isStream(s)) return error.Hy4StreamNotCooler;
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = .{
            .gpa = gpa,
            .config = config,
            .s = s,
            .embed = none,
            .lm_head = none,
            .final_norm = try get(weights, "model.norm.weight", .{}),
            .head_mix = .{ .fn_ = try get(weights, "model.hc_head.hc_head_fn", .{}), .base = try get(weights, "model.hc_head.hc_head_base", .{}), .scale = try get(weights, "model.hc_head.hc_head_scale", .{}) },
            .experts = src.experts,
            .streamed = src.experts != null,
            .quant = src.quant,
            .served = src.served,
        };
        errdefer m.freeStq();
        if (src.quant.synthetic) std.log.warn("hy_v4_flash: this pack's codes are synthetic (random): the format is real, the weights are not", .{});
        // A trellis embedding is gathered by 128-id block: on the GPU from its bank where the trellis decode kernel
        // runs, else from the gather's own host copy at its stored bytes, the pack's tensors dropped.
        if (mlx.streamIsGpu(s) and exl3.kernels.gemmNaxOn()) {
            if (try m.bankOf(weights, "model.embed_tokens")) |b| m.embed_gpu = .{ .bank = b };
        }
        if (m.embed_gpu == null) {
            if (try m.gatherOf(weights, "model.embed_tokens")) |g| {
                m.embed_gather = g;
                m.stq_decoded_bytes += g.bytes();
            } else m.embed = try get(weights, "model.embed_tokens.weight", .{});
        }
        m.lm_head_stq = try m.bankOf(weights, "lm_head");
        if (m.lm_head_stq == null) m.lm_head = try get(weights, "lm_head.weight", .{});
        errdefer m.freeEngram();
        for (0..config.layers) |l| m.layers[l] = try m.resolveLayer(weights, l);
        if (config.engram.n > 0) try m.initEngram(weights, src.dir);
        if (fwd.hc_kernels_on and mlx.streamIsGpu(s)) {
            try @import("../hc_kernels.zig").probeIdentity(s, config.hc_mult, config.dim);
            if (config.hc_mult == 4 and mlx.engine.side(.{s}) == .engine) try @import("hy4_hc.zig").probeForms(s, @intCast(config.dim));
        }
        {
            const host = mlx.engine.enter(s);
            defer host.end();
            m.placeholder = try mlx.newData(&[_]u8{0}, &[_]c_int{1}, .uint8);
        }
        return m;
    }

    /// The wide mix's grants latched before the model's forward is recorded dry (docs/bill_redesign.md, D15), which a
    /// recording that meets a pipeline with no grant refuses by name: dry forwards of one row to a verify block's most
    /// find each wide pipeline the forward launches (`hy4_hc.Census`), then each launches once on zeros. Served launches
    /// latch their own grants as they run. The state is left empty.
    pub fn probeWide(self: *Model) !void {
        const hc = @import("hy4_hc.zig");
        const core = @import("../tensor/core.zig");
        var census: hc.Census = .{};
        defer census.deinit();
        {
            hc.census = &census;
            defer hc.census = null;
            const ids: [hc.rows_max]u32 = @splat(0);
            for (1..hc.rows_max + 1) |rows| {
                self.reset();
                defer self.reset();
                var rec = try plan.Recorder.begin(self.gpa, .{ .site = plan.siteOf(@src()) });
                defer rec.deinit();
                var driver: plan.Driver = .{ .gpa = self.gpa };
                defer driver.deinit();
                try core.beginRecording(.{ .dry = .{ .driver = &driver } }, &rec);
                const lg = self.forwardReadout(ids[0..rows], .all) catch |e| {
                    core.endRecording() catch {};
                    return e;
                };
                _ = mlx.mlx_array_free(lg);
                try core.endRecording();
            }
        }
        try hc.probeNoted(self.s, &census);
    }

    pub fn deinit(self: *Model) void {
        if (self.recorded) |p| {
            p.deinit(self.gpa);
            self.recorded = null;
        }
        if (self.plans) |p| {
            p.deinit(self.gpa);
            self.plans = null;
            @import("../tensor/core.zig").arenaRelease() catch |e| @import("../log.zig").warn("[plan] the arena stays: {t}\n", .{e});
        }
        self.freeLogitsBlocks();
        if (self.ids_fixed) |a| _ = mlx.mlx_array_free(a);
        self.ids_fixed = null;
        self.unpin();
        self.reset();
        self.history.deinit(self.gpa);
        self.dropLive();
        self.live_points.deinit(self.gpa);
        if (self.placeholder) |a| _ = mlx.mlx_array_free(a);
        self.freeEngram();
        self.freeStq();
        self.gpa.destroy(self);
    }

    fn gatherOf(self: *Model, w: *model_mod.Weights, base: []const u8) !?hy4_stq.EmbedGather {
        var views: std.ArrayList(Arr) = .empty;
        defer {
            for (views.items) |v| _ = mlx.mlx_array_free(v);
            views.deinit(self.gpa);
        }
        const b = (try hy4_stq.load(w, base, self.quant.trunk, &views, self.gpa, self.s)) orelse return null;
        const g = try hy4_stq.EmbedGather.of(self.gpa, b);
        @import("../trunk_stq.zig").drop(w, base);
        return g;
    }

    fn freeStq(self: *Model) void {
        if (self.embed_gather) |*g| g.deinit(self.gpa);
        self.embed_gather = null;
        // The trellis kernels cache host arrays keyed by the banks they read (`exl3.side_cache`): a model loaded after
        // this one could meet its freed banks' entries. A served load's handle drops a side's with its last model, so
        // another model's are left; a model made directly drops them here.
        if (!self.served and (self.quant.trunk != null or self.quant.expert != null)) for ([_]u1{ 0, 1 }) |side| exl3.dropHostArrays(side);
        for (&self.stq_experts) |*r| if (r.*) |*rb| {
            rb.deinit();
            r.* = null;
        };
        for (self.stq_owned.items) |a| _ = mlx.mlx_array_free(a);
        self.stq_owned.deinit(self.gpa);
        self.stq_owned = .empty;
    }

    /// `<base>` as a trellis bank under the pack's `trunk_quant`, or null when the pack stores `<base>.weight`.
    fn bankOf(self: *Model, w: *const model_mod.Weights, base: []const u8) !?hy4_stq.Bank {
        return hy4_stq.load(w, base, self.quant.trunk, &self.stq_owned, self.gpa, self.s);
    }

    /// A trunk linear `<base>`: its trellis bank, or its stored `<base>.weight`.
    pub fn lin(self: *Model, w: *const model_mod.Weights, comptime fmt: []const u8, args: anytype) !fwd.Lin {
        var buf: [160]u8 = undefined;
        const base = try std.fmt.bufPrint(&buf, fmt, args);
        if (try self.bankOf(w, base)) |b| return .{ .w = none, .stq = b };
        var wb: [168]u8 = undefined;
        return fwd.Lin.dense(try get(w, "{s}", .{try std.fmt.bufPrint(&wb, "{s}.weight", .{base})}));
    }

    /// `<base>` as a dense weight: its trellis bank decoded once to BF16 (a weight the forward slices), or as stored.
    /// The bank is dropped from `w` once decoded: the load holds the decoded rows alone (`hy4_load.residentLoadBytes`
    /// bills them in the bank's place).
    pub fn denseOf(self: *Model, w: *model_mod.Weights, comptime fmt: []const u8, args: anytype) !Arr {
        var buf: [160]u8 = undefined;
        const base = try std.fmt.bufPrint(&buf, fmt, args);
        var views: std.ArrayList(Arr) = .empty;
        defer {
            for (views.items) |v| _ = mlx.mlx_array_free(v);
            views.deinit(self.gpa);
        }
        if (try hy4_stq.load(w, base, self.quant.trunk, &views, self.gpa, self.s)) |b| {
            const dense = try self.keepStq(try hy4_stq.decodeDense(self.gpa, self.s, b));
            self.stq_decoded_bytes += mlx.mlx_array_size(dense) * mlx.mlx_array_itemsize(dense);
            @import("../trunk_stq.zig").drop(w, base);
            return dense;
        }
        var wb: [168]u8 = undefined;
        return get(w, "{s}", .{try std.fmt.bufPrint(&wb, "{s}.weight", .{base})});
    }

    /// `a`, owned by the model from here.
    fn keepStq(self: *Model, a: Arr) !Arr {
        errdefer _ = mlx.mlx_array_free(a);
        try self.stq_owned.append(self.gpa, a);
        return a;
    }

    fn freeEngram(self: *Model) void {
        if (self.token_map.len > 0) self.gpa.free(self.token_map);
        self.token_map = &.{};
        for (self.engram_store) |*e| if (e.disk) |*d| d.close();
        self.gpa.free(self.engram_store);
        self.engram_store = &.{};
    }

    /// Drops the request's state: the next `forward` starts at position 0.
    pub fn reset(self: *Model) void {
        self.version += 1;
        self.epoch += 1;
        self.dropLive();
        for (&self.state) |*st| freeLayer(st);
        self.history.clearRetainingCapacity();
        self.pos = 0;
        self.reach = 0;
        if (self.last_hidden) |h| _ = mlx.mlx_array_free(h);
        self.last_hidden = null;
        self.dropPicks();
        if (self.hidden_fixed) |h| _ = mlx.mlx_array_free(h);
        self.hidden_fixed = null;
        for (&self.picks_fixed) |*pf| for (pf) |*b| if (b.*) |a| {
            _ = mlx.mlx_array_free(a);
            b.* = null;
        };
        freeArrays(&self.mark_fixed);
        if (self.mark_hidden) |h| _ = mlx.mlx_array_free(h);
        self.mark_hidden = null;
        self.marked = false;
        for (&self.point_fixed, &self.point_hidden) |*slot, *h| {
            freeArrays(slot);
            if (h.*) |a| _ = mlx.mlx_array_free(a);
            h.* = null;
        }
        self.point_take = @splat(0);
    }

    pub fn dropHidden(self: *Model) void {
        if (self.last_hidden) |h| _ = mlx.mlx_array_free(h);
        self.last_hidden = null;
    }

    fn dropPicks(self: *Model) void {
        for (&self.last_picks) |*pk| if (pk.*) |p| {
            _ = mlx.mlx_array_free(p.rows);
            _ = mlx.mlx_array_free(p.keep);
            pk.* = null;
        };
    }

    /// Keeps the last of `rows` [T, width] and `keep` [T, width] as layer `l`'s `last_picks`.
    fn keepPicks(self: *Model, o: *Ops, l: usize, rows: Arr, keep: Arr) !void {
        const t = mlx.getShape(rows)[0];
        const width = mlx.getShape(rows)[1];
        const r = try o.contiguous(try o.reshape(try o.slice(rows, 0, t - 1, t), &.{width}));
        const k = try o.contiguous(try o.slice(keep, 0, t - 1, t));
        if (self.last_picks[l]) |old| {
            _ = mlx.mlx_array_free(old.rows);
            _ = mlx.mlx_array_free(old.keep);
        }
        self.last_picks[l] = null;
        const pf = &self.picks_fixed[l];
        try writeFixed(&pf[0], r, mlx.getShape(r), self.s);
        try writeFixed(&pf[1], k, mlx.getShape(k), self.s);
        var kept: [2]?Arr = .{ null, null };
        errdefer for (kept) |h| if (h) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for (&kept, pf) |*h, b| try shareInto(h, b);
        self.last_picks[l] = .{ .rows = kept[0].?, .keep = kept[1].? };
    }

    /// The last row of the streams `hs` [T, hc, D] collapsed by the head's mix (`fwd.head`), as `last_hidden`.
    fn keepHidden(self: *Model, hs: Arr) !void {
        const c = self.config;
        var o: Ops = .{ .s = self.s };
        defer o.deinit();
        const t = mlx.getShape(hs)[0];
        const h = try o.contiguous(try fwd.head(&o, try o.slice(hs, 0, t - 1, t), self.head_mix, @intCast(c.hc_mult), c.hc_eps, c.eps));
        if (self.last_hidden) |old| _ = mlx.mlx_array_free(old);
        self.last_hidden = null;
        try writeFixed(&self.hidden_fixed, h, mlx.getShape(h), self.s);
        try shareInto(&self.last_hidden, self.hidden_fixed);
    }

    /// The head's logits [T, vocab] f32 over the normed rows `x` [T, D]: the f32 head, or its trellis bank.
    pub fn lmHead(self: *Model, o: *Ops, x: Arr) !Arr {
        if (self.lm_head_stq) |b| return hy4_stq.apply(o, x, b);
        return fwd.linearF32(o, x, self.lm_head);
    }

    /// The key rows sparse layer `l`'s last row read, which DFly2 shares, in f32: past the indexer's threshold its
    /// selection (`last_picks`, with `keep` [1, width] the picks it reads), else every row `0..pos` (`keep` null).
    pub fn sharedRows(self: *Model, o: *Ops, l: usize) !struct { c: Arr, k_pe: Arr, keep: ?Arr } {
        if (self.layers[l].indexer == null) return error.Hy4DflyTargetNotSparse;
        const n: c_int = @intCast(self.pos);
        if (n == 0 or self.state[l].rows == null) return error.Hy4DflyNoContext;
        var keys = self.sparseKeys(&self.state[l]);
        if (self.last_picks[l]) |pk| return .{ .c = try self.latentAt(o, &keys, pk.rows), .k_pe = try o.cast(try o.take(keys.k_pe.?, pk.rows, 0), .float32), .keep = pk.keep };
        return .{ .c = try self.latentRange(o, &keys, 0, n), .k_pe = try o.cast(try o.slice(keys.k_pe.?, 0, 0, n), .float32), .keep = null };
    }

    fn resolveLayer(self: *Model, w: *model_mod.Weights, l: usize) !Layer {
        const c = self.config;
        const sparse = c.kinds[l] == .sparse;
        const p = "model.layers.{d}.self_attn.";
        var layer: Layer = .{
            .attn = .{
                .q_a = try self.lin(w, p ++ "q_a_proj", .{l}),
                .q_a_norm = try get(w, p ++ "q_a_layernorm.weight", .{l}),
                .q_b = try self.lin(w, p ++ "q_b_proj", .{l}),
                .kv_a = try self.lin(w, p ++ "kv_a_proj_with_mqa", .{l}),
                .kv_a_norm = try get(w, p ++ "kv_a_layernorm.weight", .{l}),
                .kv_b = try self.denseOf(w, p ++ "kv_b_proj", .{l}),
                .gate = try self.lin(w, p ++ "linear_gate", .{l}),
                .o = try self.lin(w, p ++ "o_proj", .{l}),
                .post_norm = try get(w, p ++ "post_proj_norm.weight", .{l}),
                .sinks = try get(w, p ++ "learnable_sink_param", .{l}),
            },
            .geometry = .{
                .heads = @intCast(c.heads(l)),
                .nope = @intCast(c.nope_dim),
                .rope = @intCast(c.rope_dim),
                .v = @intCast(c.v_dim),
                .latent = @intCast(c.kv_lora),
                .theta = @floatCast(if (sparse) c.sparse_rope.theta else c.sliding_rope.theta),
                .eps = c.eps,
            },
            .indexer = null,
            .mix_attn = try mixAt(w, l, "hc_attn_layer"),
            .mix_ffn = try mixAt(w, l, "hc_mlp_layer"),
            .input_norm = try get(w, "model.layers.{d}.input_layernorm.weight", .{l}),
            .post_norm = try get(w, "model.layers.{d}.post_attention_layernorm.weight", .{l}),
            .ffn = undefined,
            .engram = null,
        };
        if (sparse) {
            const ip = "model.layers.{d}.self_attn.compressed_indexer.";
            layer.indexer = .{
                .query = try get(w, ip ++ "query_projection.weight", .{l}),
                .head_weight = try get(w, ip ++ "head_weight_projection.weight", .{l}),
                .token_key = try get(w, ip ++ "token_key_projection.weight", .{l}),
                .token_key_norm = try get(w, ip ++ "token_key_norm.weight", .{l}),
                .key_value = try get(w, ip ++ "key_value_projection.weight", .{l}),
                .key_gate = try get(w, ip ++ "key_gate_projection.weight", .{l}),
                .ape = try get(w, ip ++ "intra_group_ape", .{l}),
                .key_norm = try get(w, ip ++ "key_norm.weight", .{l}),
            };
        }
        const mp = "model.layers.{d}.mlp.";
        layer.ffn = if (l < c.dense_layers) .{ .dense = .{
            .sw = .{ .gate = try self.lin(w, mp ++ "gate_proj", .{l}), .up = try self.lin(w, mp ++ "up_proj", .{l}), .down = try self.lin(w, mp ++ "down_proj", .{l}) },
            .post_norm = try get(w, mp ++ "post_fc2_norm.weight", .{l}),
        } } else .{
            .moe = .{
                .router = .{ .weight = try get(w, mp ++ "gate.weight", .{l}), .bias = try get(w, mp ++ "gate.e_score_correction_bias", .{l}) },
                // A streamed load leaves the banks on SSD: the expert stream gives each forward its picks' slots.
                // An STQ pack's experts are rate-group trellis banks (`stq_experts`); a streamed load leaves the banks
                // on SSD: the expert stream gives each forward its picks' slots.
                .gate_up = if (self.streamed or self.quant.expert != null) none else try get(w, mp ++ "experts.gate_up_proj", .{l}),
                .down = if (self.streamed or self.quant.expert != null) none else try get(w, mp ++ "experts.down_proj", .{l}),
                .lat_down = try self.lin(w, mp ++ "routed_expert_down_proj", .{l}),
                .lat_up = try self.lin(w, mp ++ "routed_expert_up_proj", .{l}),
                .shared = .{ .gate = try self.lin(w, mp ++ "shared_experts.gate_proj", .{l}), .up = try self.lin(w, mp ++ "shared_experts.up_proj", .{l}), .down = try self.lin(w, mp ++ "shared_experts.down_proj", .{l}) },
                .combine = try get(w, mp ++ "combine_output_norm.weight", .{l}),
            },
        };
        if (l >= c.dense_layers and self.quant.expert != null and !self.streamed) {
            self.stq_experts[l] = try hy4_stq.loadExperts(w, l, self.s, self.gpa) orelse {
                std.log.err("hy_v4_flash: the pack stamps expert_quant and layer {d} has no expert group", .{l});
                return error.Hy4ExpertLayout;
            };
        }
        return layer;
    }

    /// The Engram modules' tensors and the hash: the token map read to the host once (its largest id sets the
    /// multipliers' bound; every module ships the same map), the primes the config lists. A table the pack's tensors
    /// lack is read by row from `dir`.
    fn initEngram(self: *Model, w: *const model_mod.Weights, dir: ?[]const u8) !void {
        const e = self.config.engram;
        const p = "model.layers.{d}.engram_layer.";
        var c_max: u32 = 0;
        for (e.layers[0..e.n], 0..) |layer, i| {
            const map_arr = try get(w, p ++ "lookup_table", .{layer});
            try mlx.check(mlx.mlx_array_eval(map_arr));
            const raw = (mlx.mlx_array_data_int32(map_arr) orelse return error.HostRead)[0..mlx.mlx_array_size(map_arr)];
            if (i == 0) {
                self.token_map = try self.gpa.alloc(u32, raw.len);
                for (raw, self.token_map) |v, *dst| {
                    dst.* = @intCast(v);
                    c_max = @max(c_max, dst.*);
                }
            } else if (raw.len != self.token_map.len or !for (raw, self.token_map) |v, m| {
                if (@as(u32, @intCast(v)) != m) break false;
            } else true) {
                std.log.err("hy_v4_flash: the Engram module at layer {d} maps tokens unlike the one at layer {d}", .{ layer, e.layers[0] });
                return error.Hy4EngramMapsDiffer;
            }
        }
        var geo: engram_mod.Geometry = .{ .n = e.n, .seed = e.seed, .max_ngram = e.max_ngram, .vocab_size = e.vocab_size, .heads = e.heads, .head_dim = e.head_dim, .pad_id = e.pad_id, .compressed_vocab = c_max + 1 };
        for (0..e.n) |i| {
            geo.layers[i] = e.layers[i];
            geo.hash_ids[i] = e.hash_ids[i];
            geo.rows[i] = e.rows[i];
        }
        self.hash = try engram_mod.Hash.init(geo, self.compressed(e.pad_id));
        self.engram_store = try self.gpa.alloc(EngramW, e.n);
        for (self.engram_store) |*store| store.disk = null;
        const cols = self.hash.?.cols();
        const n: usize = self.config.hc_mult;
        for (self.engram_store, e.layers[0..e.n], 0..) |*store, layer, i| {
            store.module = i;
            for (0..cols) |j| store.norms[j] = try get(w, p ++ "lookup_per_head_norms.{d}.weight", .{ layer, j });
            for (0..n) |ch| {
                store.streams[0][ch] = try get(w, p ++ "key_projs.{d}.weight", .{ layer, ch });
                store.streams[1][ch] = try get(w, p ++ "key_projs.{d}.bias", .{ layer, ch });
                store.streams[2][ch] = try get(w, p ++ "norm_key.{d}.weight", .{ layer, ch });
                store.streams[3][ch] = try get(w, p ++ "norm_query.{d}.weight", .{ layer, ch });
                store.streams[4][ch] = try get(w, p ++ "unified_per_hc_scales.{d}", .{ layer, ch });
            }
            var name_buf: [160]u8 = undefined;
            const table_name = try std.fmt.bufPrint(&name_buf, p ++ "multi_head_embedding.embedding.weight", .{layer});
            store.table = w.get(table_name) orelse none;
            if (store.table.ctx == null) {
                const from = dir orelse {
                    std.log.err("hy_v4_flash: the pack has no tensor {s}", .{table_name});
                    return error.Hy4TensorMissing;
                };
                store.disk = try engram_mod.Table.openBf16(self.gpa, from, table_name, e.rows[i], e.head_dim);
            }
            store.tensors = .{
                .head_norms = store.norms[0..cols],
                .key_w = store.streams[0][0..n],
                .key_b = store.streams[1][0..n],
                .norm_key = store.streams[2][0..n],
                .norm_query = store.streams[3][0..n],
                .scales = store.streams[4][0..n],
                .value_w = try get(w, p ++ "value_proj.weight", .{layer}),
                .value_b = try get(w, p ++ "value_proj.bias", .{layer}),
                .gamma = try get(w, p ++ "unified_pre_gamma", .{layer}),
            };
            self.layers[layer].engram = store;
        }
    }

    /// Extends the request by the rows `ids` and gives their logits [T, vocab] in f32 (the head widened, llama.cpp).
    /// The caller frees the result.
    pub fn forward(self: *Model, ids: []const u32) !Arr {
        return self.forwardReadout(ids, .all);
    }

    /// Which rows a forward reads out through the head: every row [T, vocab], the last [1, vocab], none (a prefill
    /// chunk before the tail, which only extends the state; the result is then an empty array), or the last row's
    /// post-final-norm hidden [1, hidden] in place of its logits (a caller that skips the head, `skip_lm_head`).
    pub const Readout = enum { all, last, none, hidden_last };

    /// `forward` reading out `readout`'s rows: the layer-at-a-time pieces in order, one window.
    pub fn forwardReadout(self: *Model, ids: []const u32, readout: Readout) !Arr {
        return self.forwardSpliced(ids, null, readout);
    }

    /// A served forward (`hy4_plan.run`): `forwardSpliced`, and inside a plan, with the logits blocks made, the logits
    /// (or the hidden row a caller that skips the head reads) written into this forward's block (D19), the caller handed
    /// a view of it and nothing of the forward's own. A caller holds at most one view across the next forward: one still
    /// held when its block is written again makes the write copy, an ask no plan recorded, and that forward misses.
    pub fn forwardServed(self: *Model, ids: []const u32, image: ?VisionRows, readout: Readout) !Arr {
        // The blocks hold `verify_rows_max` rows: a round that reads out more (a forced `SUSHI_HY4_DSPARK_ROWS`, or a
        // drafter of more drafts) has nowhere to write its logits on a load that plans.
        if (self.logits_fixed[0] != null and readout == .all and ids.len > fwd.verify_rows_max and plan.planned()) {
            @import("../log.zig").err("[plan] a round of {d} rows reads out more than the {d} rows the planned load's logits blocks hold\n", .{ ids.len, fwd.verify_rows_max });
            return error.Hy4ReadoutPastBlock;
        }
        const logits = try self.forwardSpliced(ids, image, readout);
        if (!plan.planned() or self.logits_fixed[0] == null or readout == .none) return logits;
        defer _ = mlx.mlx_array_free(logits);
        const turn = self.logits_turn;
        self.logits_turn +%= 1;
        const block = &self.logits_fixed[turn];
        try writeFixed(block, logits, &.{ fwd.verify_rows_max, @intCast(self.config.vocab) }, self.s);
        var view = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, block.*.?, &[_]c_int{ 0, 0 }, 2, mlx.getShape(logits).ptr, 2, &[_]c_int{ 1, 1 }, 2, self.s));
        return view;
    }

    /// The logits blocks (D19), made anew.
    pub fn makeLogitsBlocks(self: *Model) !void {
        self.freeLogitsBlocks();
        for (&self.logits_fixed) |*b| b.* = (try mlx_pages.ringBuffer(&.{ fwd.verify_rows_max, @intCast(self.config.vocab) }, .float32, self.s)).array;
        self.logits_turn = 0;
    }

    pub fn freeLogitsBlocks(self: *Model) void {
        for (&self.logits_fixed) |*b| if (b.*) |a| {
            _ = mlx.mlx_array_free(a);
            b.* = null;
        };
    }

    /// The forward input's block (`ids_fixed`), made with the plans.
    pub fn makeIdsBlock(self: *Model) !void {
        if (self.ids_fixed == null) self.ids_fixed = (try mlx_pages.ringBuffer(&.{ 1, ids_rows }, .uint32, self.s)).array;
    }

    /// The rows a planned load's forward input holds (`ids_fixed`): the widest prefill chunk a load runs.
    pub const ids_rows: c_int = @import("../bills.zig").PREFILL_CHUNK_LADDER[0];

    /// `ids` [1, n] written into the input block, in place, and handed back as a view of its first `n`: a planned load's
    /// forward input asks for no buffer of its own. A wider input is refused by name.
    pub fn idsInput(self: *Model, ids: []const i32) !Arr {
        if (ids.len > ids_rows) return error.Hy4IdsPastBlock;
        const block = self.ids_fixed orelse return error.Hy4IdsBlockMissing;
        const host = std.mem.bytesAsSlice(u32, try mlx.engine.hostWritable(block));
        for (host[0..ids.len], ids) |*d, v| d.* = @bitCast(v);
        var view = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, block, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 1, @intCast(ids.len) }, 2, &[_]c_int{ 1, 1 }, 2, self.s));
        return view;
    }

    /// `forwardReadout` of a prompt whose image ids take `image`'s rows [rows, D] f32 (`hy4_vision.Tower.forward`), in
    /// order, in place of their embedding. An image position is ordinary in its position and dead to Engram: it reads
    /// as the pad id, in its own n-gram and in its neighbours' (llama.cpp's choice), with or without an image.
    /// A prompt's image rows [rows, D] (any leading shape) and how many of them earlier chunks consumed.
    pub const VisionRows = struct { rows: Arr, offset: usize = 0 };

    /// Whether a forward of `ids` takes rows of `image` in place of embeddings: an image id with no row left behind it
    /// (a text prompt's, or one the model emitted) is a plain token, and a forward of none is the text forward.
    pub fn splicesImage(self: *const Model, ids: []const u32, image: ?VisionRows) bool {
        const vr = image orelse return false;
        const n_img = mlx.mlx_array_size(vr.rows) / self.config.dim;
        return vr.offset < n_img and std.mem.indexOfScalar(u32, ids, self.config.image_id) != null;
    }

    pub fn forwardSpliced(self: *Model, ids: []const u32, image: ?VisionRows, readout: Readout) !Arr {
        const t: c_int = @intCast(ids.len);
        const p0: c_int = @intCast(self.pos);
        var hs = try self.embedStreamsWith(ids, image);
        errdefer _ = mlx.mlx_array_free(hs);
        const rows = try self.engramRowsAfter(self.history.items, ids);
        defer if (rows) |r| r.deinit(self.gpa);
        // A layer's rows are appended as it runs, so a forward that fails past its first layer leaves rows no position names.
        errdefer self.reset();
        // A layer's write-back may run the next layer's attention mix (`Passed.next`), which that layer then reads.
        var ahead: ?fwd.Mixed = null;
        defer freeMixed(ahead);
        // The selection DFly2 shares is this forward's last row's, or none below the threshold.
        self.dropPicks();
        for (0..self.config.layers) |l| {
            // A layer is a frame of its own, keyed by its index: each layer's kinds and geometry are its own. Inside it,
            // the facts of this forward's rows that decide its structure (`layerKey`) pick a variant: one iteration of
            // a loop whose key may differ from one forward to the next.
            const f = try plan.frame(@src(), .{l});
            defer f.end();
            const v = if (plan.planned()) try plan.each(@src(), self.layerKey(l, p0, t)) else plan.Frame.none;
            defer v.end();
            const next = try self.passLayerAhead(l, hs, ahead, rows, t, p0 + t, p0);
            freeMixed(ahead);
            ahead = next.next;
            _ = mlx.mlx_array_free(hs);
            hs = next.hs;
        }
        if (self.keep_hidden) try self.keepHidden(hs);
        const logits = try self.readoutStreams(hs, readout);
        errdefer if (readout != .none) {
            _ = mlx.mlx_array_free(logits);
        };
        _ = mlx.mlx_array_free(hs);
        try self.submitState(if (readout == .none) null else logits);
        try self.appendHistory(ids);
        self.pos += ids.len;
        return logits;
    }

    /// Submits the state arrays that are still graphs (a ring's write after the spans, the pooled keys and open group no
    /// span read), in one batch with the forward's `logits` when it reads any: a graph keeps the chunk's inputs alive
    /// until something reads it, and the state is billed as its arrays alone (`hy4_ledger.stateBytes`). A submit, not a
    /// wait: the caller's read of the logits is the forward's one.
    fn submitState(self: *Model, logits: ?Arr) !void {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var n: usize = 0;
        if (logits) |l| {
            try mlx.check(mlx.mlx_vector_array_append_value(vec, l));
            n += 1;
        }
        var arrays: [max_state_arrays]Arr = undefined;
        for (self.state[0..self.config.layers]) |*st| for (stateArrays(st, &arrays)) |arr| {
            var ready = false;
            try mlx.check(mlx._mlx_array_is_available(&ready, arr));
            if (!ready) {
                try mlx.check(mlx.mlx_vector_array_append_value(vec, arr));
                n += 1;
            }
        };
        if (n > 0) try mlx.check(mlx.mlx_async_eval(vec));
    }

    /// A token id as Engram reads it: an image id as the pad id. The one mapping of the raw ids every producer of the
    /// history and of the rows' ids goes through (`historyId`, `engramRowsAfter`).
    fn engramId(self: *const Model, id: u32) u32 {
        const image = self.config.image_id;
        return if (image != 0 and id == image) self.config.engram.pad_id else id;
    }

    /// A token id as Engram's history holds it.
    pub fn historyId(self: *const Model, id: u32) u32 {
        return self.compressed(self.engramId(id));
    }

    /// `ids` appended to the request's Engram history.
    fn appendHistory(self: *Model, ids: []const u32) !void {
        try self.history.ensureUnusedCapacity(self.gpa, ids.len);
        for (ids) |id| self.history.appendAssumeCapacity(self.historyId(id));
    }

    /// The `hc` f32 streams of `ids` before layer 0, [T, hc, D]: each the token's embedding. The caller frees them.
    pub fn embedStreams(self: *Model, ids: []const u32) !Arr {
        return self.embedStreamsWith(ids, null);
    }

    /// `embedStreams` with the image ids' rows taken from `image` [rows, D], one a row in order; it must hold one for
    /// each image id.
    fn embedStreamsWith(self: *Model, ids: []const u32, vision: ?VisionRows) !Arr {
        const c = self.config;
        const t: c_int = @intCast(ids.len);
        var o: Ops = .{ .s = self.s };
        defer o.deinit();
        const e = if (self.embed_gpu) |g| try o.own(try g.rows(self.gpa, self.s, ids)) else if (self.embed_gather) |g| blk: {
            const rows = try self.gpa.alloc(f32, ids.len * c.dim);
            defer self.gpa.free(rows);
            try g.rows(self.gpa, ids, rows);
            const e_rows = try o.own(mlx.mlx_array_new_data(rows.ptr, &[_]c_int{ t, @intCast(c.dim) }, 2, .float32));
            // Evaluated before `rows` is freed: an array made from host data may read it when it is evaluated.
            try mlx.check(mlx.mlx_array_eval(e_rows));
            break :blk e_rows;
        } else try o.cast(try o.take(self.embed, try o.own(mlx.mlx_array_new_data(ids.ptr, &[_]c_int{t}, 1, .uint32)), 0), .float32);
        const n_img: usize = if (vision) |vr| mlx.mlx_array_size(vr.rows) / c.dim else 0;
        const spliced = if (self.splicesImage(ids, vision)) blk: {
            const vr = vision.?;
            const pick = try self.gpa.alloc(u32, ids.len);
            defer self.gpa.free(pick);
            var k: usize = vr.offset;
            for (ids, pick, 0..) |id, *d, i| {
                if (id == c.image_id) {
                    d.* = @intCast(ids.len + k);
                    k += 1;
                } else d.* = @intCast(i);
            }
            if (k > n_img) return error.Hy4ImageRowsShort;
            const sel = try o.own(mlx.mlx_array_new_data(pick.ptr, &[_]c_int{t}, 1, .uint32));
            const img = try o.reshape(vr.rows, &.{ @intCast(n_img), @intCast(c.dim) });
            const out = try o.take(try o.concat(&.{ e, try o.cast(img, .float32) }, 0), sel, 0);
            // Evaluated before `pick` is freed: an array made from host data may read it when it is evaluated.
            try mlx.check(mlx.mlx_array_eval(out));
            break :blk out;
        } else e;
        const hc: c_int = @intCast(c.hc_mult);
        // Copied once here: layer 0's mix, its norm and its write-back each read the streams row-contiguous.
        return o.result(try o.contiguous(try o.broadcast(try o.reshape(spliced, &.{ t, 1, @intCast(c.dim) }), &.{ t, hc, @intCast(c.dim) })));
    }

    /// The head over the streams `hs` [T, hc, D]: `.all` gives [T, vocab] f32, `.last` [1, vocab], `.none` an empty
    /// array. Reads neither the position nor the history. A tap reads the head's input on every row, whatever is read
    /// out: the rows the head takes when it serves. The caller frees the result.
    pub fn readoutStreams(self: *Model, hs: Arr, readout: Readout) !Arr {
        const exact = mlx.engine.exactF32();
        defer exact.end();
        const c = self.config;
        const t = mlx.getShape(hs)[0];
        var o: Ops = .{ .s = self.s };
        defer o.deinit();
        const head_in = struct {
            fn rows(m: *Model, ops: *Ops, x: Arr) !Arr {
                return fwd.headNorm(ops, x, m.head_mix, @intCast(m.config.hc_mult), m.config.hc_eps, m.config.eps, m.final_norm);
            }
        }.rows;
        const logits = struct {
            fn of(m: *Model, ops: *Ops, x: Arr) !Arr {
                if (m.lm_head_stq) |b| return hy4_stq.apply(ops, x, b);
                return fwd.linearF32(ops, x, m.lm_head);
            }
        }.of;
        const all = if (self.tap != null or readout == .all) try head_in(self, &o, hs) else null;
        if (self.tap) |tp| try tp.input(tp.ctx, c.layers, .head_in, all.?);
        return switch (readout) {
            .none => mlx.mlx_array_new(),
            .all => o.result(try logits(self, &o, all.?)),
            // The read row on its own, as with no tap: a product over one row need not round as one over T.
            .last => o.result(try logits(self, &o, try head_in(self, &o, try o.slice(hs, 0, t - 1, t)))),
            .hidden_last => o.result(try head_in(self, &o, try o.slice(hs, 0, t - 1, t))),
        };
    }

    /// Frees layer `l`'s request state (its latent rows, ring, pooled keys and open group): a layer-at-a-time run drops
    /// a window's state once the window has run the layer.
    pub fn releaseLayer(self: *Model, l: usize) void {
        freeLayer(&self.state[l]);
    }

    /// `ids` is `[modules][T·cols]`: each module's table rows, position-major.
    pub const EngramRows = struct {
        ids: []u32,
        live: []f32,

        pub fn deinit(self: EngramRows, gpa: std.mem.Allocator) void {
            gpa.free(self.ids);
            gpa.free(self.live);
        }
    };

    /// The Engram table rows of each of the raw `ids`' positions' hash heads after `history` (`historyId`s), per module,
    /// and which positions are live (not a pad id, which an image id reads as). Reads nothing of the request's own
    /// history.
    pub fn engramRowsAfter(self: *Model, history: []const u32, ids: []const u32) !?EngramRows {
        const hash = self.hash orelse return null;
        const cols = hash.cols();
        const mods = hash.n_layers;
        const comp = try self.gpa.alloc(u32, ids.len);
        defer self.gpa.free(comp);
        for (ids, comp) |id, *dst| dst.* = self.historyId(id);
        const mixed = try self.gpa.alloc(u32, ids.len * mods * cols);
        defer self.gpa.free(mixed);
        hash.rows(history, comp, mixed);
        const out = try self.gpa.alloc(u32, ids.len * mods * cols);
        errdefer self.gpa.free(out);
        for (0..ids.len) |i| for (0..mods) |m| {
            @memcpy(out[(m * ids.len + i) * cols ..][0..cols], mixed[(i * mods + m) * cols ..][0..cols]);
        };
        const live = try self.gpa.alloc(f32, ids.len);
        for (ids, live) |id, *dst| dst.* = if (self.engramId(id) == self.config.engram.pad_id) 0 else 1;
        return .{ .ids = out, .live = live };
    }

    /// Rows `r0..r1` of the chunk's gated attention (`fwd.attendGated`), over the keys they read: a sliding row its
    /// window, a sparse row every key up to the selection threshold and its recalled groups' rows past it (`pooled` is
    /// then the layer's pooled keys). `in` holds the chunk's gate rows.
    fn attendSpan(self: *Model, o: *Ops, l: usize, x: Arr, p: fwd.Projected, in: fwd.SpanIn, pooled: ?Arr, win: *LayerState, core_keys: ?hy4_attn.Keys, r0: c_int, r1: c_int, n: c_int, p0: c_int, tap: ?fwd.At) !Arr {
        const L = &self.layers[l];
        const st = &self.state[l];
        const rows = r1 - r0;
        const at = p0 + r0;
        const last = p0 + r1; // keys past the span's last row are never read
        const qs = try o.slice(p.q, 1, r0, r1);
        const sin: fwd.SpanIn = .{ .gate = try o.slice(in.gate, 0, r0, r1), .kv_b32 = in.kv_b32, .whole = in.whole, .o_left = in.o_left, .dst = if (in.dst) |d| .{ .block = d.block, .row = r0 } else null, .index = in.index, .rows_as = in.rows_as, .q_raw = if (p.q_raw) |r| try o.slice(r, 1, r0, r1) else null };
        if (core_keys) |ck| if (rows > fwd.verify_rows_max) return self.attendSpanCore(o, l, x, p, sin, qs, pooled, ck, win.base, r0, r1, n, p0, tap);
        if (L.indexer) |iw| {
            const ig = self.indexerGeometry();
            var keys = self.sparseKeys(st);
            const thr = ig.group_topk * ig.ratio;
            const all = pooled orelse return fwd.attendGatedCausal(o, sin, qs, try self.latentRange(o, &keys, 0, last), try o.slice(keys.k_pe.?, 0, 0, last), L.attn, L.geometry, at, 0, tap);
            if (last <= thr) return fwd.attendGatedCausal(o, sin, qs, try self.latentRange(o, &keys, 0, last), try o.slice(keys.k_pe.?, 0, 0, last), L.attn, L.geometry, at, 0, tap);
            if (at < thr) {
                const cut = thr - at;
                const sel = try fwd.recalledAt(o, try o.slice(x, 0, r0 + cut, r1), try o.slice(p.q_resid, 0, r0 + cut, r1), all, iw, ig, thr, n);
                if (self.keep_hidden and last == n) try self.keepPicks(o, l, sel.rows, sel.keep);
                const flat = try o.reshape(sel.rows, &.{(rows - cut) * fwd.selectedWidth(sel.rows)});
                return fwd.attendStraddleGated(o, sin, qs, cut, try self.latentRange(o, &keys, 0, thr), try o.slice(keys.k_pe.?, 0, 0, thr), at, try self.latentAt(o, &keys, flat), try o.take(keys.k_pe.?, flat, 0), sel.keep, L.attn, L.geometry, tap);
            }
            const sel = if (sin.index != null and r0 >= sin.index.?.r0)
                try fwd.recalledPrepared(o, sin.index.?, r0, r1, mlx.getShape(all)[0], ig, at, n)
            else
                try fwd.recalledAt(o, try o.slice(x, 0, r0, r1), try o.slice(p.q_resid, 0, r0, r1), all, iw, ig, at, n);
            if (self.keep_hidden and last == n) try self.keepPicks(o, l, sel.rows, sel.keep);
            const flat = try o.reshape(sel.rows, &.{rows * fwd.selectedWidth(sel.rows)});
            return fwd.attendRowsGated(o, sin, qs, try self.latentAt(o, &keys, flat), try o.take(keys.k_pe.?, flat, 0), sel.keep, L.attn, L.geometry, tap);
        }
        const w: c_int = @intCast(self.config.window);
        if (win.unordered) {
            // A decode row reads every row the ring holds, in ring order: the keys carry their positions already.
            const k = @min(n, w);
            const keys_c = try self.latentRange(o, win, 0, k);
            const keys_pe = try o.slice(win.k_pe.?, 0, 0, k);
            if (n > w) return fwd.attendGatedFull(o, sin, qs, keys_c, keys_pe, L.attn, L.geometry, k - 1, tap);
            return fwd.attendGatedCausal(o, sin, qs, keys_c, keys_pe, L.attn, L.geometry, k - 1, 0, tap);
        }
        const s0: c_int = @max(0, at - w + 1);
        const c_win = try self.latentRange(o, win, s0 - win.base, last - win.base);
        const pe_win = try o.slice(win.k_pe.?, 0, s0 - win.base, last - win.base);
        if (fwd.isVerifyBlock(rows)) return fwd.attendGatedRing(o, sin, qs, c_win, pe_win, L.attn, L.geometry, at - s0, w, at, tap);
        return fwd.attendGatedCausal(o, sin, qs, c_win, pe_win, L.attn, L.geometry, at - s0, w, tap);
    }

    /// `attendSpan`'s rows in the matrix-unit core (`hy4_attn`) over `ck`, the chunk's keys at the dequantize's width
    /// (`coreKeys`, whose first row is position `base` on a sliding layer): the same keys each row reads in the chain, and
    /// on a sparse layer past its selection the picks read from the layer's rows where they lie.
    fn attendSpanCore(self: *Model, o: *Ops, l: usize, x: Arr, p: fwd.Projected, sin: fwd.SpanIn, qs: Arr, pooled: ?Arr, ck: hy4_attn.Keys, base: c_int, r0: c_int, r1: c_int, n: c_int, p0: c_int, tap: ?fwd.At) !Arr {
        const L = &self.layers[l];
        const at = p0 + r0;
        const last = p0 + r1; // keys past the span's last row are never read
        if (L.indexer) |iw| {
            const ig = self.indexerGeometry();
            const thr = ig.group_topk * ig.ratio;
            const all = pooled orelse return fwd.attendCoreGated(o, sin, qs, .{ .range = .{ .keys = try self.keysTo(o, ck, 0, last), .p0 = at, .window = 0 } }, L.attn, L.geometry, tap);
            if (last <= thr) return fwd.attendCoreGated(o, sin, qs, .{ .range = .{ .keys = try self.keysTo(o, ck, 0, last), .p0 = at, .window = 0 } }, L.attn, L.geometry, tap);
            if (at < thr) {
                const cut = thr - at;
                const sel = try fwd.recalledAt(o, try o.slice(x, 0, r0 + cut, r1), try o.slice(p.q_resid, 0, r0 + cut, r1), all, iw, ig, thr, n);
                if (self.keep_hidden and last == n) try self.keepPicks(o, l, sel.rows, sel.keep);
                return fwd.attendCoreGated(o, sin, qs, .{ .straddle = .{ .cut = cut, .range = .{ .keys = try self.keysTo(o, ck, 0, thr), .p0 = at, .window = 0 }, .picks = .{ .keys = ck, .rows = sel.rows, .keep = sel.keep } } }, L.attn, L.geometry, tap);
            }
            const sel = if (sin.index != null and r0 >= sin.index.?.r0)
                try fwd.recalledPrepared(o, sin.index.?, r0, r1, mlx.getShape(all)[0], ig, at, n)
            else
                try fwd.recalledAt(o, try o.slice(x, 0, r0, r1), try o.slice(p.q_resid, 0, r0, r1), all, iw, ig, at, n);
            if (self.keep_hidden and last == n) try self.keepPicks(o, l, sel.rows, sel.keep);
            return fwd.attendCoreGated(o, sin, qs, .{ .picks = .{ .keys = ck, .rows = sel.rows, .keep = sel.keep } }, L.attn, L.geometry, tap);
        }
        const w: c_int = @intCast(self.config.window);
        const s0: c_int = @max(0, at - w + 1);
        return fwd.attendCoreGated(o, sin, qs, .{ .range = .{ .keys = try self.keysTo(o, ck, s0 - base, last - base), .p0 = at - s0, .window = w } }, L.attn, L.geometry, tap);
    }

    /// Rows `start..end` of `ck`.
    fn keysTo(_: *Model, o: *Ops, ck: hy4_attn.Keys, start: c_int, end: c_int) !hy4_attn.Keys {
        return .{ .c = try o.slice(ck.c, 0, start, end), .k_pe = try o.slice(ck.k_pe, 0, start, end) };
    }

    /// The keys a chunk's wide spans read in the matrix-unit core, held by `keep`: a sliding layer's window rows (`win`,
    /// the ring's rows then the chunk's own) or a sparse layer's rows to the chunk's end, the latent at the dequantize's
    /// width (BF16 as stored, or kv8 read to BF16 once for every span).
    fn coreKeys(self: *Model, keep: *Ops, L: *const Layer, st: *const LayerState, win: *const LayerState) !hy4_attn.Keys {
        var o: Ops = .{ .s = self.s };
        defer o.deinit();
        const keys: LayerState = if (L.indexer == null) win.* else self.sparseKeys(st);
        const rows: c_int = if (L.indexer == null) mlx.getShape(win.k_pe.?)[0] else @intCast(st.rows.?.used);
        const c = try o.slice(keys.c.?, 0, 0, rows);
        const pe = try o.slice(keys.k_pe.?, 0, 0, rows);
        if (self.kv.scheme == .off) return .{ .c = try keep.own(try o.result(c)), .k_pe = try keep.own(try o.result(pe)) };
        const sc = try o.slice(keys.c_scales.?, 0, 0, rows);
        const bi = try o.slice(keys.c_biases.?, 0, 0, rows);
        const wide = if (hy4_latent.rangeApplies(o.s, c, sc, bi, self.kv.group_size, self.kv.bits))
            try o.own(try hy4_latent.rangeBf16(o.s, c, sc, bi, self.kv.group_size))
        else
            try o.own(try kv_quant.dequantizeAffine(o.s, c, sc, bi, self.kv.group_size, self.kv.bits));
        return .{ .c = try keep.own(try o.result(wide)), .k_pe = try keep.own(try o.result(pe)) };
    }

    /// Appends the new rows' latent `c` [T, latent] at the request's width (BF16, or affine-quantized in groups along the
    /// row: each row's codes, scales and biases its own, so rows append independently) and rope rows `k_pe` [T, rope]
    /// (BF16) to a sparse layer's store: only the new rows are written.
    fn appendLatent(self: *Model, o: *Ops, st: *LayerState, c: Arr, k_pe: Arr) !void {
        const rows = try o.cast(c, .bfloat16);
        const pe = try o.cast(k_pe, .bfloat16);
        if (self.kv.scheme == .off) return appendRows(&st.rows, &.{ rows, pe }, o.s);
        if (hy4_latent.appendApplies(o.s, rows, pe, self.kv.group_size, self.kv.bits, 0)) {
            const n: usize = @intCast(mlx.getShape(rows)[0]);
            const width = mlx.getShape(rows)[1];
            const rope = mlx.getShape(pe)[1];
            if (st.rows == null) st.rows = row_store.Store.init(&.{
                .{ .cols = @intCast(kv_quant.packedWords(width, self.kv.bits)), .dtype = .uint32 },
                .{ .cols = @intCast(@divExact(width, @as(c_int, @intCast(self.kv.group_size)))), .dtype = .bfloat16 },
                .{ .cols = @intCast(@divExact(width, @as(c_int, @intCast(self.kv.group_size)))), .dtype = .bfloat16 },
                .{ .cols = @intCast(rope), .dtype = .bfloat16 },
            }, rows_grow_div);
            const r = &st.rows.?;
            try r.reserveAppend(n, o.s);
            try hy4_latent.append(o.s, rows, pe, &.{ r.buf(0), r.buf(1), r.buf(2), r.buf(3) }, @intCast(r.used), 0);
            r.used += n;
            return;
        }
        var q = try kv_quant.quantizeAffine(o.s, rows, self.kv.group_size, self.kv.bits);
        defer q.deinit();
        try appendRows(&st.rows, &.{ q.q, q.scales, q.biases, pe }, o.s);
    }

    /// A sparse layer's store as the arrays the spans read (`latentRange`, `latentAt`, the rope rows), its buffers
    /// borrowed: the first `rows.used` rows of each are the layer's.
    fn sparseKeys(self: *const Model, st: *const LayerState) LayerState {
        const r = &st.rows.?;
        if (self.kv.scheme == .off) return .{ .c = r.buf(0), .k_pe = r.buf(1) };
        return .{ .c = r.buf(0), .c_scales = r.buf(1), .c_biases = r.buf(2), .k_pe = r.buf(3) };
    }

    /// The state of a request `n` rows deep, reserved for `reach`, as a dry recording leaves it, made without the forwards
    /// that would reach it (docs/bill_redesign.md, "State is dry too"): each sparse layer's store holding `n` rows and
    /// its pooled keys `n / ratio`, its open group `n % ratio` rows deep, each ring written to `n`, the history `n` ids.
    /// The next forward keeps the drafter's handover. Only inside a dry recording: its stores hold no bytes.
    pub fn dryState(self: *Model, n: usize, reach: usize) !void {
        if (!mlx.engine.dryOn()) return error.Hy4DryStateNotDry;
        self.reset();
        errdefer self.reset();
        try self.reserveFor(@max(n, reach));
        const g = self.config.indexer.group;
        for (self.state[0..self.config.layers], 0..) |*st, l| {
            if (self.layers[l].indexer == null) {
                st.ring = @intCast(n);
                continue;
            }
            st.rows.?.used = n;
            if (st.pooled) |*p| p.used = n / g;
            if (n % g > 0) st.open = try headRows(st.open_buf.?, n % g, self.s);
        }
        try self.history.appendNTimes(self.gpa, self.historyId(0), n);
        self.pos = n;
    }

    /// Room for a request whose rows reach `reach`, made between its forwards: each sparse layer's rows and pooled keys
    /// at what the request's bill holds (`hy4_ledger.capRows`), grown or cut back to it, each sliding layer's ring, each
    /// sparse layer's open block and, with a drafter, the handover's and each sparse layer's picks'. A forward within
    /// the reach then asks for no state of its own: its appends and writes land in these (docs/bill_redesign.md part 8).
    pub fn reserveFor(self: *Model, reach: usize) !void {
        if (reach < self.pos) return error.Hy4ReachBelowState;
        self.reach = reach;
        // The state is its owner's, billed with the request, never the plan's: it passes a closed pool.
        const owned = mlx.engine.widePhase();
        defer owned.end();
        const c = self.config;
        const g: usize = c.indexer.group;
        const w: c_int = @intCast(c.window);
        const latent: c_int = @intCast(c.kv_lora);
        const rope: c_int = @intCast(c.rope_dim);
        const dense = self.kv.scheme == .off;
        const words: c_int = @intCast(kv_quant.packedWords(latent, self.kv.bits));
        const groups: c_int = @divExact(latent, @as(c_int, @intCast(self.kv.group_size)));
        for (self.state[0..c.layers], 0..) |*st, l| {
            if (self.layers[l].indexer != null) {
                if (st.rows == null) st.rows = row_store.Store.init(if (dense) &.{ .{ .cols = @intCast(latent), .dtype = .bfloat16 }, .{ .cols = @intCast(rope), .dtype = .bfloat16 } } else &.{
                    .{ .cols = @intCast(words), .dtype = .uint32 },
                    .{ .cols = @intCast(groups), .dtype = .bfloat16 },
                    .{ .cols = @intCast(groups), .dtype = .bfloat16 },
                    .{ .cols = @intCast(rope), .dtype = .bfloat16 },
                }, rows_grow_div);
                try st.rows.?.reserveRows(reach, self.s);
                if (reach / g > 0) {
                    if (st.pooled == null) st.pooled = row_store.Store.init(&.{.{ .cols = c.indexer.head_dim, .dtype = .float32 }}, rows_grow_div);
                    try st.pooled.?.reserveRows(reach / g, self.s);
                } else if (st.pooled) |*p| {
                    // A reach that closes no group holds no pooled keys, as its bill does.
                    p.deinit();
                    st.pooled = null;
                }
                if (g > 1 and st.open_buf == null) st.open_buf = (try mlx_pages.ringBuffer(&.{ @intCast(g - 1), @intCast(c.indexer.head_dim) }, .float32, self.s)).array;
                continue;
            }
            const ring = [_]struct { []const u8, c_int, mlx.mlx_dtype }{
                .{ "c", if (dense) latent else words, if (dense) .bfloat16 else .uint32 },
                .{ "c_scales", groups, .bfloat16 },
                .{ "c_biases", groups, .bfloat16 },
                .{ "k_pe", rope, .bfloat16 },
            };
            inline for (.{ "c", "c_scales", "c_biases", "k_pe" }, 0..) |f, i| {
                if (@field(st, f) == null and !(dense and (i == 1 or i == 2))) @field(st, f) = (try mlx_pages.ringBuffer(&.{ w, ring[i][1] }, ring[i][2], self.s)).array;
            }
        }
        for (&self.point_fixed) |*slot| try self.smallBlocks(slot);
        if (!self.keep_hidden) return;
        if (self.hidden_fixed == null) self.hidden_fixed = (try mlx_pages.ringBuffer(&.{ 1, @intCast(c.dim) }, .float32, self.s)).array;
        for (&self.point_hidden) |*h| if (h.* == null) {
            h.* = (try mlx_pages.ringBuffer(&.{ 1, @intCast(c.dim) }, .float32, self.s)).array;
        };
        // A selection past the threshold reads `group_topk` groups of `ratio` rows (`hy4_index.pick`): its picks' blocks.
        const width: c_int = @intCast(c.indexer.group_topk * g);
        for (self.picks_fixed[0..c.layers], 0..) |*pf, l| if (self.layers[l].indexer != null and pf[0] == null) {
            pf[0] = (try mlx_pages.ringBuffer(&.{width}, .uint32, self.s)).array;
            pf[1] = (try mlx_pages.ringBuffer(&.{ 1, width }, .bool_, self.s)).array;
        };
        // A drafting request's round marks the state before its verify (`mark`).
        try self.smallBlocks(&self.mark_fixed);
        if (self.mark_hidden == null) self.mark_hidden = (try mlx_pages.ringBuffer(&.{ 1, @intCast(c.dim) }, .float32, self.s)).array;
    }

    /// Blocks for a copy of the small state (`mark_fields`), each its state block's shape, where `blocks` lacks one.
    fn smallBlocks(self: *Model, blocks: *[max_layers][point_arrays]?Arr) !void {
        for (self.state[0..self.config.layers], blocks[0..self.config.layers]) |*st, *slots| inline for (mark_fields, mark_slots) |f, i| {
            if (@field(st, f)) |b| if (slots[i] == null) {
                slots[i] = (try mlx_pages.ringBuffer(mlx.getShape(b), mlx.mlx_array_dtype(b), self.s)).array;
            };
        };
    }

    /// Keeps the state where it stands as a live point, the oldest let go past `live_point_cap`.
    pub fn keepLivePoint(self: *Model) !void {
        if (self.live_points.items.len == live_point_cap) {
            const old = self.live_points.orderedRemove(0);
            self.releasePoint(&old);
        }
        try self.live_points.ensureUnusedCapacity(self.gpa, 1);
        self.live_points.appendAssumeCapacity(try self.point());
    }

    /// Lets go of the running request's points.
    pub fn dropLive(self: *Model) void {
        for (self.live_points.items) |*p| self.releasePoint(p);
        self.live_points.clearRetainingCapacity();
    }

    /// Frees `p`'s slot, unless a reset or a later take already has.
    pub fn releasePoint(self: *Model, p: *const Point) void {
        if (self.point_take[p.slot] == p.take) self.point_take[p.slot] = 0;
    }

    /// Whether `p`'s slot still holds it.
    pub fn holdsPoint(self: *const Model, p: *const Point) bool {
        return self.point_take[p.slot] == p.take;
    }

    pub fn freeArrays(arrays: [][point_arrays]?Arr) void {
        for (arrays) |*row| for (row) |*a| if (a.*) |x| {
            _ = mlx.mlx_array_free(x);
            a.* = null;
        };
    }

    /// The state where it stands as a point: its small state copied into a free `Points` slot, in place, the rows its
    /// stores' own. Every slot held is `Hy4PointsFull`: the caps that fill them let one go first.
    pub fn point(self: *Model) !Point {
        const slot = try self.freeSlot();
        self.point_takes +%= 1;
        if (self.point_takes == 0) self.point_takes = 1;
        var p: Point = .{ .pos = self.pos, .slot = slot, .take = self.point_takes };
        p.hidden = try self.copySmall(&self.point_fixed[slot], &self.point_hidden[slot], &p.held, &p.open);
        self.point_take[slot] = p.take;
        return p;
    }

    /// A free slot no stored point still reads: a stored point taken over a live one shares its slot's blocks
    /// (`pointArrays`) past the live point's release, and a write into them would copy them on the pool.
    fn freeSlot(self: *const Model) !u8 {
        next: for (self.point_take, 0..) |take, i| {
            if (take != 0) continue;
            for (self.point_fixed[i][0..self.config.layers]) |blocks| for (blocks) |b| if (b) |x| if (!try mlx.engine.sole(x)) continue :next;
            if (self.point_hidden[i]) |h| if (!try mlx.engine.sole(h)) continue :next;
            return @intCast(i);
        }
        return error.Hy4PointsFull;
    }

    /// The small state copied into `blocks` and `hidden_block`, whole and in place: each sliding layer's ring, each sparse
    /// layer's open block (its rows a count in `open`) and the handover. `held`: which of a layer's blocks hold a copy
    /// (bit `i` for `point_fields[i]`). Whether the handover was copied.
    fn copySmall(self: *Model, blocks: *[max_layers][point_arrays]?Arr, hidden_block: *?Arr, held: *[max_layers]u8, open: *[max_layers]u8) !bool {
        const n = self.config.layers;
        for (self.state[0..n], blocks[0..n], 0..) |*st, *slots, l| inline for (mark_fields, mark_slots) |f, i| {
            // A closed group's block holds no row of the state.
            const src = if (i == 5) (if (st.open != null) st.open_buf else null) else @field(st, f);
            if (src) |b| {
                try writeFixed(&slots[i], b, mlx.getShape(b), self.s);
                held[l] |= 1 << i;
            }
            if (i == 5) if (st.open) |o| {
                open[l] = @intCast(mlx.getShape(o)[0]);
            };
        };
        const h = self.last_hidden orelse return false;
        try writeFixed(hidden_block, h, mlx.getShape(h), self.s);
        return true;
    }

    /// The blocks of `blocks` that `held` names, in `point_fields`' layout: what a rollback writes back.
    fn heldBlocks(self: *const Model, blocks: *const [max_layers][point_arrays]?Arr, held: *const [max_layers]u8) [max_layers][point_arrays]?Arr {
        var out: [max_layers][point_arrays]?Arr = @splat(@splat(null));
        for (out[0..self.config.layers], blocks[0..self.config.layers], held[0..self.config.layers]) |*row, slots, h| {
            for (row, slots, 0..) |*a, s, i| a.* = if (h & (@as(u8, 1) << @intCast(i)) != 0) s else null;
        }
        return out;
    }

    /// The point `p`'s small state in `capturePoint`'s layout, new handles on its slot's blocks: a sliding layer's ring
    /// and a sparse layer's open group (the head of its block). The caller frees them; the rows' slots stay empty.
    pub fn pointArrays(self: *const Model, p: *const Point, out: [][point_arrays]?Arr) !void {
        if (out.len != self.config.layers or !self.holdsPoint(p)) return error.Hy4PointMismatch;
        for (out) |*row| row.* = @splat(null);
        errdefer freeArrays(out);
        const blocks = self.heldBlocks(&self.point_fixed[p.slot], &p.held);
        for (out, blocks[0..out.len], p.open[0..out.len]) |*row, src, open| {
            for (0..4) |i| try shareInto(&row[i], src[i]);
            if (src[5]) |b| if (open > 0) {
                row[5] = try headRows(b, open, self.s);
            };
        }
    }

    /// Makes the state the point `p` of this state, which a forward ran past: the stores cut back to its rows (and their
    /// buffers to the bill of `for_rows` or the reservation, the larger), its rings, open groups and hidden state written
    /// back from its slot, the Engram history cut to its position. Points past it are let go: the next append writes
    /// over their rows. Checked whole before anything changes; `p` stays valid for another rollback.
    pub fn rollbackPoint(self: *Model, p: *const Point, for_rows: usize) !void {
        if (!self.holdsPoint(p)) return error.Hy4PointMismatch;
        const n = self.config.layers;
        const arrays = self.heldBlocks(&self.point_fixed[p.slot], &p.held);
        return self.rollbackFrom(.{ .pos = p.pos, .arrays = arrays[0..n], .hidden = if (p.hidden) self.point_hidden[p.slot] else null, .open_rows = p.open[0..n] }, for_rows);
    }

    /// What a rollback writes back: a point's copies, or a mark's blocks with each open group's rows (`open_rows`), whose
    /// whole block goes back.
    const Back = struct { pos: usize, arrays: []const [point_arrays]?Arr, hidden: ?Arr, open_rows: ?[]const u8 = null };

    fn rollbackFrom(self: *Model, p: Back, for_rows: usize) !void {
        const n = self.config.layers;
        const g: usize = self.config.indexer.group;
        if (p.arrays.len != n or p.pos > self.pos or for_rows < p.pos or self.history.items.len != self.pos) return error.Hy4PointMismatch;
        for (self.state[0..n], 0..) |*st, l| if (self.layers[l].indexer != null) {
            if (p.pos > 0 and (st.rows == null or st.rows.?.used < p.pos)) return error.Hy4PointMismatch;
            if (p.pos / g > 0 and (st.pooled == null or st.pooled.?.used < p.pos / g)) return error.Hy4PointMismatch;
        };
        errdefer self.reset();
        self.version += 1;
        for (self.state[0..n], p.arrays, 0..) |*st, row, l| {
            if (self.layers[l].indexer != null) {
                if (st.rows) |*r| try r.truncate(p.pos);
                if (st.pooled) |*r| try r.truncate(p.pos / g);
                if (p.open_rows) |rows| try self.openBlockFrom(st, row[5], rows[l]) else try self.openFrom(st, row[5]);
                continue;
            }
            // The point's rings written back over the state's own, in place: the point keeps its copy.
            inline for (.{ "c", "k_pe", "c_scales", "c_biases" }, 0..) |f, i| if (row[i]) |src| try writeFixed(&@field(st, f), src, mlx.getShape(src), self.s);
            st.ring = @intCast(p.pos);
        }
        self.history.shrinkRetainingCapacity(p.pos);
        self.pos = p.pos;
        if (self.last_hidden) |h| _ = mlx.mlx_array_free(h);
        self.last_hidden = null;
        if (p.hidden) |h| {
            try writeFixed(&self.hidden_fixed, h, mlx.getShape(h), self.s);
            try shareInto(&self.last_hidden, self.hidden_fixed);
        }
        var i: usize = 0;
        while (i < self.live_points.items.len) {
            if (self.live_points.items[i].pos > p.pos) {
                const q = self.live_points.orderedRemove(i);
                self.releasePoint(&q);
            } else i += 1;
        }
        // A reserved request keeps its reservation (`reserveFor`): the stores are cut back only past it.
        const keep = @max(for_rows, self.reach);
        const rows_billed = if (keep > 0) row_store.capBound(keep, rows_grow_div) else 0;
        const pooled_billed = if (keep / g > 0) row_store.capBound(keep / g, rows_grow_div) else 0;
        var need_trim = false;
        for (self.state[0..n], 0..) |*st, l| if (self.layers[l].indexer != null) {
            if (st.rows) |r| if (!r.fitsBill(rows_billed, self.s)) {
                need_trim = true;
                break;
            };
            if (st.pooled) |r| if (!r.fitsBill(pooled_billed, self.s)) {
                need_trim = true;
                break;
            };
        };
        if (need_trim) {
            const cap = mlx.engine.uncapped();
            defer mlx.engine.capped(cap);
            for (self.state[0..n], 0..) |*st, l| if (self.layers[l].indexer != null) {
                if (st.rows) |*r| if (!r.fitsBill(rows_billed, self.s)) try r.trim(self.s);
                if (st.pooled) |*r| if (!r.fitsBill(pooled_billed, self.s)) try r.trim(self.s);
            };
        }
    }

    /// The arrays a restore point keeps of each layer, in this order; an absent one is null.
    pub const point_fields = .{ "c", "k_pe", "c_scales", "c_biases", "pooled", "open" };
    pub const point_arrays = 6;

    /// The request's state as a stored restore point, `out[l]` for layer `l` (`point_fields`): the small state
    /// (`captureFixed`) and views of every row (`captureRows`). The views hold the stores' buffers: the caller drops them
    /// before the next forward appends, or that append copies every store.
    pub fn capturePoint(self: *const Model, out: [][point_arrays]?Arr) !void {
        try self.captureFixed(out);
        errdefer freeArrays(out);
        try self.captureRows(out, self.pos);
    }

    /// The small state as copies, `out[l]` for layer `l` (`point_fields`, the rows' slots empty): a sliding layer's ring,
    /// a sparse layer's open group. The state stays the sole holder of its blocks, so its next write is in place.
    pub fn captureFixed(self: *const Model, out: [][point_arrays]?Arr) !void {
        const n = self.config.layers;
        if (out.len != n) return error.Hy4PointMismatch;
        for (out) |*row| row.* = @splat(null);
        errdefer freeArrays(out);
        for (self.state[0..n], out) |*st, *row| {
            inline for (.{ "c", "k_pe", "c_scales", "c_biases", "open" }, .{ 0, 1, 2, 3, 5 }) |f, i| row[i] = try copyOf(@field(st, f), self.s);
        }
    }

    /// A sparse layer's open group made `src`'s rows (or none), written into its open block in place.
    fn openFrom(self: *Model, st: *LayerState, src: ?Arr) !void {
        if (st.open) |o| _ = mlx.mlx_array_free(o);
        st.open = null;
        const rows = src orelse return;
        const g: c_int = @intCast(self.config.indexer.group);
        try writeFixed(&st.open_buf, rows, &.{ g - 1, mlx.getShape(rows)[1] }, self.s);
        st.open = try headRows(st.open_buf.?, @intCast(mlx.getShape(rows)[0]), self.s);
    }

    /// A sparse layer's open block made `src`, whole, its open group its first `rows` (none at 0), in place.
    fn openBlockFrom(self: *Model, st: *LayerState, src: ?Arr, rows: usize) !void {
        if (st.open) |o| _ = mlx.mlx_array_free(o);
        st.open = null;
        const block = src orelse return;
        try writeFixed(&st.open_buf, block, mlx.getShape(block), self.s);
        if (rows > 0) st.open = try headRows(st.open_buf.?, rows, self.s);
    }

    /// Views of each sparse layer's rows for positions `0..pos` into `out`'s rows' slots (`point_fields`): the latent
    /// and rope rows and the pooled keys of the closed groups. The stores' rows below `pos` must be the point's own.
    pub fn captureRows(self: *const Model, out: [][point_arrays]?Arr, pos: usize) !void {
        const g: usize = self.config.indexer.group;
        for (self.state[0..self.config.layers], out) |*st, *row| {
            if (st.rows) |r| {
                if (r.used < pos) return error.Hy4PointMismatch;
                const keys = self.sparseKeys(st);
                if (pos > 0) inline for (.{ "c", "k_pe", "c_scales", "c_biases" }, 0..) |f, i| if (@field(keys, f)) |b| {
                    row[i] = try headRows(b, pos, self.s);
                };
            }
            if (st.pooled) |r| {
                if (r.used < pos / g) return error.Hy4PointMismatch;
                if (pos / g > 0) row[4] = try headRows(r.buf(0), pos / g, self.s);
            }
        }
    }

    /// Makes the request's state the point `from` (`capturePoint`'s layout; its handles stay the caller's) at `pos`,
    /// over the first `pos` of `ids`: every layer's arrays, its ring, the Engram history and the width the point keeps.
    pub fn restorePoint(self: *Model, from: []const [point_arrays]?Arr, pos: usize, ids: []const u32) !void {
        if (ids.len < pos) return error.Hy4PointMismatch;
        try self.restoreArrays(from, pos);
        errdefer self.reset();
        try self.appendHistory(ids[0..pos]);
    }

    /// The state readied for a point of `pos` rows at width `kv` whose bytes a reader writes straight into its blocks
    /// (`restoreDest`, then `restoreEnd`): emptied in place, or started over at another width, and reserved for the
    /// point's rows or the state's reach, the larger. A block another handle still reads (a stored point's view of the
    /// rows) is made anew by the reservation, so the write lands in bytes the state alone holds.
    pub fn restoreBegin(self: *Model, kv: kv_quant.KVQuantConfig, pos: usize) !void {
        if (!mlx.engine.isStream(self.s)) return error.Hy4RestoreOffEngine;
        const reach = self.reach;
        if (std.meta.eql(kv, self.kv)) self.empty() else self.reset();
        errdefer self.reset();
        self.kv = kv;
        for (self.state[0..self.config.layers]) |*st| {
            inline for (.{ "rows", "pooled" }) |f| if (@field(st, f)) |*r| {
                for (r.bufs[0..r.n]) |b| if (b.array.ctx != null and !try mlx.engine.sole(b.array)) {
                    r.deinit();
                    break;
                };
            };
            inline for (.{ "c", "c_scales", "c_biases", "k_pe", "open_buf" }) |f| if (@field(st, f)) |b| if (!try mlx.engine.sole(b)) {
                _ = mlx.mlx_array_free(b);
                @field(st, f) = null;
            };
        }
        try self.reserveFor(@max(pos, reach));
    }

    /// The bytes of the state block that holds point array `field` (`point_fields`) of layer `l`, from its row `row` on,
    /// for `t`'s rows: its dtype and columns the block's, its rows within the block (a ring or an open block whole).
    pub fn restoreDest(self: *Model, l: usize, field: usize, row: usize, dtype: mlx.mlx_dtype, shape: []const c_int) ![]u8 {
        if (l >= self.config.layers or shape.len != 2) return error.Hy4PointMismatch;
        const st = &self.state[l];
        const sparse = self.layers[l].indexer != null;
        const block: ?Arr = if (sparse) switch (field) {
            0, 1, 2, 3 => blk: {
                const keys = self.sparseKeys(st);
                break :blk switch (field) {
                    0 => keys.c,
                    1 => keys.k_pe,
                    2 => keys.c_scales,
                    else => keys.c_biases,
                };
            },
            4 => if (st.pooled) |p| p.buf(0) else null,
            5 => st.open_buf,
            else => null,
        } else switch (field) {
            0 => st.c,
            1 => st.k_pe,
            2 => st.c_scales,
            3 => st.c_biases,
            else => null,
        };
        const b = block orelse return error.Hy4PointMismatch;
        const bs = mlx.getShape(b);
        const n: usize = @intCast(shape[0]);
        const whole = !sparse or field == 5;
        if (mlx.mlx_array_dtype(b) != dtype or bs.len != 2 or bs[1] != shape[1]) return error.Hy4PointMismatch;
        if ((whole and row != 0) or (!sparse and n != @as(usize, @intCast(bs[0]))) or row + n > @as(usize, @intCast(bs[0]))) return error.Hy4PointMismatch;
        const bytes = try mlx.engine.hostWritable(b);
        const row_bytes = bytes.len / @as(usize, @intCast(bs[0]));
        return bytes[row * row_bytes ..][0 .. n * row_bytes];
    }

    /// The state stands at the point `restoreDest` wrote, `pos` rows over the first `pos` of `ids`; `rows[l][i]`: the
    /// rows written into point array `i` of layer `l`, which must be all the point holds at the state's width.
    pub fn restoreEnd(self: *Model, pos: usize, ids: []const u32, rows: []const [point_arrays]usize) !void {
        errdefer self.reset();
        const n = self.config.layers;
        if (ids.len < pos or rows.len != n) return error.Hy4PointMismatch;
        const g: usize = self.config.indexer.group;
        const scaled = self.kv.scheme != .off;
        for (self.state[0..n], rows, 0..) |*st, got, l| {
            if (self.layers[l].indexer != null) {
                const want = [point_arrays]usize{ pos, pos, if (scaled) pos else 0, if (scaled) pos else 0, pos / g, pos % g };
                if (!std.mem.eql(usize, &want, &got)) return error.Hy4PointMismatch;
                st.rows.?.used = pos;
                if (st.pooled) |*p| p.used = pos / g;
                if (pos % g > 0) st.open = try headRows(st.open_buf.?, pos % g, self.s);
                continue;
            }
            const w = self.config.window;
            const want = [point_arrays]usize{ w, w, if (scaled) w else 0, if (scaled) w else 0, 0, 0 };
            if (!std.mem.eql(usize, &want, &got)) return error.Hy4PointMismatch;
            st.ring = @intCast(pos);
        }
        self.pos = pos;
        try self.appendHistory(ids[0..pos]);
    }

    /// A verify block's mark: the state before the block, its small state copied into the model's mark blocks
    /// (`mark_fixed`), whole, so a mark asks for nothing and its bytes do not follow the position. `rollbackTo` returns
    /// to it; `dropMark` lets it go. `held`: which of a layer's blocks it copied (bit `i` for `point_fields[i]`);
    /// `open`: each sparse layer's open rows.
    pub const Mark = struct { pos: usize, epoch: u64, hidden: bool, held: [max_layers]u8 = @splat(0), open: [max_layers]u8 = @splat(0) };

    /// The state blocks a mark copies, and their places in `point_fields`.
    const mark_fields = .{ "c", "k_pe", "c_scales", "c_biases", "open_buf" };
    const mark_slots = .{ 0, 1, 2, 3, 5 };

    pub fn mark(self: *Model) !Mark {
        if (self.marked) return error.Hy4MarkHeld;
        var mk: Mark = .{ .pos = self.pos, .epoch = self.epoch, .hidden = false };
        mk.hidden = try self.copySmall(&self.mark_fixed, &self.mark_hidden, &mk.held, &mk.open);
        self.marked = true;
        return mk;
    }

    pub fn dropMark(self: *Model, mk: *Mark) void {
        if (mk.epoch == self.epoch) self.marked = false;
    }

    /// The state at `mk`, then the rows `replay` (the verdict's pending token and accepted drafts) run again from it
    /// (`hy4_plan.run`, on the arena where the load planned the replay): the request as a serial run of those rows
    /// leaves it, the rows, rings, open groups and Engram history. The mark stays valid for another rollback.
    pub fn rollbackTo(self: *Model, mk: *const Mark, replay: []const u32) !void {
        if (!self.marked or mk.epoch != self.epoch) return error.Hy4PointMismatch;
        const n = self.config.layers;
        const arrays = self.heldBlocks(&self.mark_fixed, &mk.held);
        // A verify's rollback stays inside its request, whose reservation stands (`reach`): nothing to trim.
        try self.rollbackFrom(.{ .pos = mk.pos, .arrays = arrays[0..n], .hidden = if (mk.hidden) self.mark_hidden else null, .open_rows = mk.open[0..n] }, self.pos);
        if (replay.len > 0) _ = mlx.mlx_array_free(try @import("hy4_plan.zig").run(self, replay, null, .none));
    }

    /// The point's arrays as the request's state at `pos`, with an empty history (the caller fills it).
    /// Installed into the state's own blocks: its stores reserved for the point's rows or the state's reach, the larger,
    /// the rows written into them, the rings and open groups written in place. A point at another latent width than the
    /// state's starts the state over at the point's width.
    fn restoreArrays(self: *Model, from: []const [point_arrays]?Arr, pos: usize) !void {
        const n = self.config.layers;
        if (from.len != n) return error.Hy4PointMismatch;
        // The latent's width is the point's: scales are kept only at the 8-bit width.
        const kv: kv_quant.KVQuantConfig = if (from.len > 0 and from[0][2] != null) .engine_default else .dense;
        const reach = self.reach;
        if (std.meta.eql(kv, self.kv)) self.empty() else self.reset();
        errdefer self.reset();
        self.kv = kv;
        try self.reserveFor(@max(pos, reach));
        for (self.state[0..n], from, 0..) |*st, row, l| {
            if (self.layers[l].indexer != null) {
                if (row[0]) |c| {
                    const parts: []const Arr = if (row[2] != null) &.{ c, row[2].?, row[3].?, row[1].? } else &.{ c, row[1].? };
                    try st.rows.?.append(parts, self.s);
                }
                if (row[4]) |pooled| try st.pooled.?.append(&.{pooled}, self.s);
                try self.openFrom(st, row[5]);
                continue;
            }
            inline for (point_fields, 0..) |f, i| if (comptime i < 4) if (row[i]) |src| {
                try writeFixed(&@field(st, f), src, mlx.getShape(src), self.s);
            };
            st.ring = @intCast(pos);
        }
        self.pos = pos;
    }

    /// The state emptied in place, as `reset` leaves it for every reader, its blocks, slots and reservation kept: the
    /// next state is written into them.
    fn empty(self: *Model) void {
        self.version += 1;
        self.epoch += 1;
        self.dropLive();
        self.point_take = @splat(0);
        for (self.state[0..self.config.layers]) |*st| {
            if (st.rows) |*r| r.used = 0;
            if (st.pooled) |*r| r.used = 0;
            if (st.open) |o| _ = mlx.mlx_array_free(o);
            st.open = null;
            st.ring = 0;
        }
        self.history.clearRetainingCapacity();
        self.pos = 0;
        self.dropHidden();
        self.dropPicks();
        self.marked = false;
    }

    /// The new rows `c` and `k_pe` at the stored width (as `appendLatent` stores them), owned by `keep`.
    fn freshRows(self: *Model, o: *Ops, keep: *Ops, c: Arr, k_pe: Arr) !LayerState {
        var out: LayerState = .{};
        const rows = try o.cast(c, .bfloat16);
        out.k_pe = try keep.own(try o.result(try o.cast(k_pe, .bfloat16)));
        if (self.kv.scheme == .off) {
            out.c = try keep.own(try o.result(rows));
            return out;
        }
        var q = try kv_quant.quantizeAffine(o.s, rows, self.kv.group_size, self.kv.bits);
        defer q.deinit();
        out.c = try keep.own(try o.result(q.q));
        out.c_scales = try keep.own(try o.result(q.scales));
        out.c_biases = try keep.own(try o.result(q.biases));
        return out;
    }

    /// The keys a chunk at `p0` reads: the ring's rows for positions `max(0, p0 - window + 1)..p0` in position order,
    /// then the chunk's own `fresh` rows; `base` is the first one's position. Owned by `keep`.
    fn windowKeys(self: *Model, keep: *Ops, st: *const LayerState, fresh: *const LayerState, p0: c_int) !LayerState {
        const w: c_int = @intCast(self.config.window);
        const held = @min(p0, w - 1);
        var out: LayerState = .{ .base = p0 - held };
        inline for (.{ "c", "c_scales", "c_biases", "k_pe" }) |f| if (@field(fresh, f)) |new| {
            var o: Ops = .{ .s = self.s };
            defer o.deinit();
            var parts: [3]Arr = undefined;
            var np: usize = 0;
            var pos = p0 - held;
            while (pos < p0) {
                const slot = @mod(pos, w);
                const run = @min(p0 - pos, w - slot);
                parts[np] = try o.slice(@field(st, f).?, 0, slot, slot + run);
                np += 1;
                pos += run;
            }
            parts[np] = new;
            np += 1;
            @field(out, f) = try keep.own(try o.result(if (np == 1) new else try o.concat(parts[0..np], 0)));
        };
        return out;
    }

    /// Writes the last `window` of a forward's `t` new rows at `p0` into the ring, each at its position's row, in
    /// place (the ring's handle is its only reference here).
    fn writeRing(self: *Model, st: *LayerState, fresh: *const LayerState, p0: c_int, t: c_int) !void {
        const w: c_int = @intCast(self.config.window);
        const k = @min(t, w);
        inline for (.{ "c", "c_scales", "c_biases", "k_pe" }) |f| if (@field(fresh, f)) |new| {
            const cols = mlx.getShape(new)[1];
            if (@field(st, f) == null) @field(st, f) = (try mlx_pages.ringBuffer(&.{ w, cols }, mlx.mlx_array_dtype(new), self.s)).array;
            var i = t - k;
            while (i < t) {
                const slot = @mod(p0 + i, w);
                const run = @min(t - i, w - slot);
                var o: Ops = .{ .s = self.s };
                defer o.deinit();
                const rows = try o.slice(new, 0, i, i + run);
                const next = blk: {
                    var up = mlx.mlx_array_new();
                    errdefer _ = mlx.mlx_array_free(up);
                    break :blk try mlx_pages.InPlace.adopt(@field(st, f).?, .ring).write(&up, rows, &.{ slot, 0 }, &.{ slot + run, cols }, self.s);
                };
                _ = mlx.mlx_array_free(@field(st, f).?);
                @field(st, f) = next.array;
                // The ring holds the write from here: one that fails to land is still the ring's to release.
                try next.landed(self.s);
                i += run;
            }
        };
        st.ring = p0 + t;
    }

    fn writeRingLatent(self: *Model, st: *LayerState, c: Arr, k_pe: Arr, p0: c_int, t: c_int) !void {
        const w: c_int = @intCast(self.config.window);
        const width = mlx.getShape(c)[1];
        const rope = mlx.getShape(k_pe)[1];
        const words = kv_quant.packedWords(width, self.kv.bits);
        const groups = @divExact(width, @as(c_int, @intCast(self.kv.group_size)));
        if (st.c == null) st.c = (try mlx_pages.ringBuffer(&.{ w, words }, .uint32, self.s)).array;
        if (st.c_scales == null) st.c_scales = (try mlx_pages.ringBuffer(&.{ w, groups }, .bfloat16, self.s)).array;
        if (st.c_biases == null) st.c_biases = (try mlx_pages.ringBuffer(&.{ w, groups }, .bfloat16, self.s)).array;
        if (st.k_pe == null) st.k_pe = (try mlx_pages.ringBuffer(&.{ w, rope }, .bfloat16, self.s)).array;
        try hy4_latent.append(self.s, c, k_pe, &.{ st.c.?, st.c_scales.?, st.c_biases.?, st.k_pe.? }, p0, w);
        st.ring = p0 + t;
    }

    /// The latent rows `start..end` in f32.
    fn latentRange(self: *Model, o: *Ops, st: *LayerState, start: c_int, end: c_int) !Arr {
        const c = try o.slice(st.c.?, 0, start, end);
        if (self.kv.scheme == .off) return o.cast(c, .float32);
        const sc = try o.slice(st.c_scales.?, 0, start, end);
        const bi = try o.slice(st.c_biases.?, 0, start, end);
        if (hy4_latent.rangeApplies(o.s, c, sc, bi, self.kv.group_size, self.kv.bits))
            return o.own(try hy4_latent.range(o.s, c, sc, bi, self.kv.group_size));
        return o.cast(try o.own(try kv_quant.dequantizeAffine(o.s, c, sc, bi, self.kv.group_size, self.kv.bits)), .float32);
    }

    /// The latent rows at `idx` (flat) in f32: gathered at the stored width, then widened.
    fn latentAt(self: *Model, o: *Ops, st: *LayerState, idx: Arr) !Arr {
        if (self.kv.scheme != .off and hy4_latent.gatherApplies(o.s, st.c.?, st.c_scales.?, st.c_biases.?, idx, self.kv.group_size, self.kv.bits))
            return o.own(try hy4_latent.gather(o.s, st.c.?, st.c_scales.?, st.c_biases.?, idx, self.kv.group_size));
        const c = try o.take(st.c.?, idx, 0);
        if (self.kv.scheme == .off) return o.cast(c, .float32);
        return o.cast(try o.own(try kv_quant.dequantizeAffine(o.s, c, try o.take(st.c_scales.?, idx, 0), try o.take(st.c_biases.?, idx, 0), self.kv.group_size, self.kv.bits)), .float32);
    }

    /// Makes the kernels whose variant follows a size the served warm's prompts do not reach: the sort the indexer's
    /// picks run over every pooled group picks its block by the group count, which grows with the context, so a
    /// partition at each power of two from 16 to 16,384 (past the multi-block sort) makes every variant at load; the
    /// embedding gather's block counts and a span's short tail (`warmSpanTails`) likewise.
    pub fn warmKernels(self: *Model) !void {
        // The GPU embedding gather's decode at each launch width (`hy4_embed.launchBlocks`), one id a block.
        if (self.embed_gpu) |g| {
            var ids: [@import("hy4_embed.zig").group_blocks]u32 = undefined;
            for (&ids, 0..) |*id, b| id.* = @intCast(b * 128);
            var n: usize = 1;
            while (n <= ids.len) : (n *= 2) {
                const e = try g.rows(self.gpa, self.s, ids[0..n]);
                defer _ = mlx.mlx_array_free(e);
                try mlx.check(mlx.mlx_array_eval(e));
            }
            // Past one launch's group the parts are joined and taken back into the ids' order: a second group, out of order.
            var many: [ids.len + 1]u32 = undefined;
            for (&many, 0..) |*id, b| id.* = @intCast((b * 7 % many.len) * 128);
            const e = try g.rows(self.gpa, self.s, &many);
            defer _ = mlx.mlx_array_free(e);
            try mlx.check(mlx.mlx_array_eval(e));
        }
        // An image prompt's splice takes the embedding's and the tower's rows through one take, which no text prompt makes.
        if (self.config.vision != null) {
            var o: Ops = .{ .s = self.s };
            defer o.deinit();
            const rows = try o.slot();
            try mlx.check(mlx.mlx_zeros(rows, &[_]c_int{ 1, @intCast(self.config.dim) }, 2, .float32, self.s));
            const streams = try self.embedStreamsWith(&.{self.config.image_id}, .{ .rows = rows.* });
            defer _ = mlx.mlx_array_free(streams);
            try mlx.check(mlx.mlx_array_eval(streams));
        }
        try self.warmSpanTails();
        var size: c_int = 16;
        while (size <= 16384) : (size *= 2) {
            var o: Ops = .{ .s = self.s };
            defer o.deinit();
            const range = try o.slot();
            try mlx.check(mlx.mlx_arange(range, 0, @floatFromInt(size), 1, .float32, o.s));
            const row = try o.reshape(range.*, &.{ 1, size });
            const order = try o.slot();
            try mlx.check(mlx.mlx_argpartition_axis(order, try o.unary(.negative, row), @min(size - 1, 15), -1, o.s));
            try mlx.check(mlx.mlx_array_eval(order.*));
        }
    }

    /// The short last span of a chunk launches the gate and W_o of a layer alone (`fwd.linearSpans`), at any width below
    /// its span's. A trellis bank's rows arm takes that width as a template value, so each width it serves is made at
    /// load; past it the sorted GEMM serves every width with the pipelines the whole spans make.
    fn warmSpanTails(self: *Model) !void {
        const exact = mlx.engine.exactF32();
        defer exact.end();
        // The gate reads the attention norm's rows in f32, W_o the gated rows narrow (`fwd.linearInput`).
        for (self.layers[0..self.config.layers]) |*layer| for ([_]struct { fwd.Lin, mlx.mlx_dtype }{ .{ layer.attn.gate, .float32 }, .{ layer.attn.o, .bfloat16 } }) |pair| {
            const bank = pair[0].stq orelse continue;
            var rows: c_int = 1;
            while (!exl3.trunk.gemmRows(@intCast(rows), false)) : (rows += 1) {
                var o: Ops = .{ .s = self.s };
                defer o.deinit();
                const x = try o.slot();
                try mlx.check(mlx.mlx_zeros(x, &[_]c_int{ rows, bank.inDim() }, 2, pair[1], self.s));
                try mlx.check(mlx.mlx_array_eval(try fwd.linearStored(&o, x.*, pair[0])));
            }
        };
        if (self.lm_head_stq) |bank| {
            var rows: c_int = 2;
            while (rows <= fwd.verify_rows_max) : (rows += 1) {
                var o: Ops = .{ .s = self.s };
                defer o.deinit();
                const x = try o.slot();
                try mlx.check(mlx.mlx_zeros(x, &[_]c_int{ rows, bank.inDim() }, 2, .float32, self.s));
                try mlx.check(mlx.mlx_array_eval(try hy4_stq.apply(&o, x.*, bank)));
            }
        }
    }

    /// How rows `p0..p0 + t` meet the edges where a layer's structure changes: a sparse layer's open group and pooled
    /// keys before them, its groups closing and its selection threshold (`attendSpan`), a sliding layer's window filling
    /// and its ring wrapping. A plan frame's key.
    pub const LayerKey = struct { layer: usize, open: bool, pooled: bool, closes: c_int, arm: enum { dense, straddle, selected }, cut: c_int, split: bool, lone: bool, first: bool, full: bool, wraps: bool, ring: bool, edge: c_int };

    pub fn layerKey(self: *const Model, l: usize, p0: c_int, t: c_int) LayerKey {
        const ig = self.indexerGeometry();
        const thr = ig.group_topk * ig.ratio;
        const w: c_int = @intCast(self.config.window);
        const sparse = self.layers[l].indexer != null;
        return .{
            .layer = l,
            .open = sparse and @rem(p0, ig.ratio) != 0,
            .pooled = sparse and p0 >= ig.ratio,
            // How many groups close: the pool takes their rows one at a time from two (`fwd.mmByRow`).
            .closes = if (sparse) @divFloor(p0 + t, ig.ratio) - @divFloor(p0, ig.ratio) else 0,
            .arm = if (!sparse or p0 + t <= thr) .dense else if (p0 < thr) .straddle else .selected,
            // A block across the threshold runs its rows before it dense and the rest selected: either part one row is
            // a one-row operand, so where it cuts is the key.
            .cut = if (sparse and p0 < thr and p0 + t > thr) thr - p0 else -1,
            .split = sparse and p0 + t > thr and fwd.selectionSplits(ig, p0, t),
            // One key, the row's own: an operand of one row is laid out as its transpose, which an op then never copies.
            .lone = p0 + t == 1,
            // No key before its rows: a block at the start reads none of the state, which a later one appends to.
            .first = p0 == 0,
            .full = !sparse and p0 + t > w,
            .wraps = !sparse and @rem(p0, w) + t > w,
            // A verify block reads its windows from the ring once a window precedes it (`ringKeysApplies`); before,
            // each of its rows past the window's edge rotates its keys by its own slot, so the block's start decides.
            .ring = !sparse and p0 >= w,
            .edge = if (!sparse and p0 < w and p0 + t > w) p0 else -1,
        };
    }

    /// The rows a chunk's spans run at past its closed `groups` (`loop_span`): the widest halving of `span_rows` whose
    /// indexer scores stay under 32 MB, else `span`.
    fn spanWidth(self: *const Model, span: c_int, groups: c_int, p0: c_int) c_int {
        var ws = self.span_rows;
        while (ws > span and @as(i64, ws) * @as(i64, groups) * 4 > 32 * 1024 * 1024) ws = @divTrunc(ws, 2);
        const ig = self.indexerGeometry();
        const thr = ig.group_topk * ig.ratio;
        if (ws > span and @rem(ws, span) == 0 and (p0 >= thr or @rem(thr - p0, ws) == 0)) return ws;
        return span;
    }

    /// What decides the structure of a wide forward of `t` rows at `p0` read out by `readout`, past the selection
    /// threshold and the window (`hy4_plan`'s classes): how its rows meet each kind of layer's edges, its spans' width
    /// and whether its window reads the ring in one run. Two forwards of one class ask alike, the deeper no less.
    pub fn wideClass(self: *const Model, p0: c_int, t: c_int, readout: anytype) u64 {
        var h = std.hash.Wyhash.init(0);
        std.hash.autoHash(&h, t);
        std.hash.autoHash(&h, readout);
        var kinds: [2]bool = .{ false, false };
        for (self.layers, 0..) |L, l| {
            const kind = @intFromBool(L.indexer != null);
            if (kinds[kind]) continue;
            kinds[kind] = true;
            var k = self.layerKey(l, p0, t);
            k.layer = 0;
            std.hash.autoHash(&h, k);
        }
        const ig = self.indexerGeometry();
        std.hash.autoHash(&h, self.spanWidth(@min(self.span_rows, gathered_span), @divFloor(p0 + t, ig.ratio), p0));
        std.hash.autoHash(&h, @rem(p0 + 1, @as(c_int, @intCast(self.config.window))) == 0);
        return h.final();
    }

    fn indexerGeometry(self: *const Model) fwd.IndexerGeometry {
        const c = self.config;
        return .{
            .heads = @intCast(c.indexer.heads),
            .dim = @intCast(c.indexer.head_dim),
            .rope = @intCast(c.rope_dim),
            .ratio = @intCast(c.indexer.group),
            .group_topk = @intCast(c.indexer.group_topk),
            .theta = @floatCast(c.sparse_rope.theta),
            .eps = c.eps,
        };
    }

    fn compressed(self: *const Model, id: u32) u32 {
        return if (id < self.token_map.len) self.token_map[id] else id;
    }

    /// One layer over the streams `hs` [T, n, D] of the new rows `p0..n` (`t` of them, their Engram `rows`); the result
    /// is a new array the caller owns. It reads and writes only layer `l`'s state, so a layer-at-a-time run passes each
    /// window through it before the next layer. The layer
    /// runs as phases, each in a scope of its own that hands on only its outputs (kept in `keep`): a forward holds one
    /// phase's working arrays beside what crosses phases.
    pub fn passLayer(self: *Model, l: usize, hs: Arr, rows: ?EngramRows, t: c_int, n: c_int, p0: c_int) !Arr {
        return (try self.passLayerWith(l, hs, null, rows, t, n, p0, false)).hs;
    }

    /// A layer's streams out, and the next layer's attention mix of them when the layer's last launch ran it.
    pub const Passed = struct { hs: Arr, next: ?fwd.Mixed };

    /// `passLayer` reading `pre`, layer `l`'s attention mix of `hs` when the layer before ran it, and running layer
    /// `l + 1`'s in its write-back where it can (no Engram delta comes between). The caller frees both results.
    fn passLayerAhead(self: *Model, l: usize, hs: Arr, pre: ?fwd.Mixed, rows: ?EngramRows, t: c_int, n: c_int, p0: c_int) !Passed {
        const ahead = l + 1 < self.config.layers and (self.layers[l + 1].engram == null or rows == null);
        return self.passLayerWith(l, hs, pre, rows, t, n, p0, ahead);
    }

    fn passLayerWith(self: *Model, l: usize, hs: Arr, pre: ?fwd.Mixed, rows: ?EngramRows, t: c_int, n: c_int, p0: c_int, ahead: bool) !Passed {
        const exact = mlx.engine.exactF32();
        defer exact.end();
        const front = try self.layerFrontWith(l, hs, pre, rows, t, n, p0);
        defer front.deinit();
        const c = self.config;
        const next: ?fwd.NextMix = if (ahead) .{ .mix = self.layers[l + 1].mix_attn, .n = @intCast(c.hc_mult), .magnitude = c.hc_magnitude, .eps = c.hc_eps, .norm_eps = c.eps, .norm_w = self.layers[l + 1].input_norm } else null;
        const fy = try self.ffnFor(l, front.x, .{ .h = front.mid, .post = front.post, .next = next });
        if (fy.expanded) return .{ .hs = fy.y, .next = fy.next };
        defer _ = mlx.mlx_array_free(fy.y);
        return .{ .hs = try self.layerBack(front, fy.y), .next = null };
    }

    /// What a layer's FFN and write-back read of one piece: the streams after the attention, the FFN mix's post weights
    /// and the FFN input.
    pub const Front = struct {
        mid: Arr,
        post: Arr,
        x: Arr,

        pub fn deinit(self: Front) void {
            inline for (.{ self.mid, self.post, self.x }) |a| _ = mlx.mlx_array_free(a);
        }
    };

    /// `passLayer`'s phases 0 to 3: Engram, the attention and the FFN's mix and norm. A layer-major run takes the FFN of
    /// several pieces together (`ffnBatch`) between this and `layerBack`.
    pub fn layerFront(self: *Model, l: usize, hs: Arr, rows: ?EngramRows, t: c_int, n: c_int, p0: c_int) !Front {
        return self.layerFrontWith(l, hs, null, rows, t, n, p0);
    }

    /// `layerFront` with `pre`, the attention mix and norm of `hs` the layer before ran (no Engram delta between).
    fn layerFrontWith(self: *Model, l: usize, hs: Arr, pre: ?fwd.Mixed, rows: ?EngramRows, t: c_int, n: c_int, p0: c_int) !Front {
        // The head, the router, the mixes, the indexer, Engram and the attention's f32 products are held to f32.
        const exact = mlx.engine.exactF32();
        defer exact.end();
        const at: ?fwd.At = if (self.tap) |tp| .{ .tap = tp, .layer = l } else null;
        const c = self.config;
        const L = &self.layers[l];
        const st = &self.state[l];
        const hc: c_int = @intCast(c.hc_mult);
        var keep: Ops = .{ .s = self.s };
        defer keep.deinit();
        // Phase 1: Engram, the attention mix and norm, the projections; the state appends.
        self.version += 1;
        var h: Arr = undefined;
        var post_a: Arr = undefined;
        var x: Arr = undefined;
        var x_bf: ?Arr = null;
        var p: fwd.Projected = undefined;
        var pooled: ?Arr = null;
        // A sliding layer's new rows at the stored width, written to its ring once the spans have read the window.
        var fresh: LayerState = .{};
        var ring_c: ?Arr = null;
        var ring_pe: ?Arr = null;
        var ring_written = false;
        // Phase 0: Engram's delta into the streams, in a scope of its own.
        var with_engram = hs;
        if (L.engram) |e| if (rows) |r| {
            var o: Ops = .{ .s = self.s };
            defer o.deinit();
            const cols: c_int = @intCast(self.hash.?.cols());
            const n_ids: usize = @intCast(t * cols);
            const mine = r.ids[e.module * n_ids ..][0..n_ids];
            const gathered = if (e.disk) |*d|
                try o.own(try d.gather(self.gpa, mine, self.s))
            else
                try o.take(e.table, try o.own(mlx.mlx_array_new_data(mine.ptr, &[_]c_int{t * cols}, 1, .uint32)), 0);
            const table_rows = try o.reshape(gathered, &.{ t, cols, @intCast(c.engram.head_dim) });
            const live = try o.own(mlx.mlx_array_new_data(r.live.ptr, &[_]c_int{t}, 1, .float32));
            with_engram = try keep.own(try o.result(try fwd.engramAdd(&o, hs, table_rows, live, e.tensors, at)));
        };
        {
            var o: Ops = .{ .s = self.s };
            defer o.deinit();
            const hh = with_engram;
            const a = pre orelse try fwd.mixNorm(&o, hh, L.mix_attn, hc, c.hc_magnitude, c.hc_eps, c.eps, L.input_norm);
            const xx = a.x;
            const share_bf = t > 1 and at == null and fwd.linearInputDtype(L.attn.q_a) == .bfloat16 and
                fwd.linearInputDtype(L.attn.kv_a) == .bfloat16 and fwd.linearInputDtype(L.attn.gate) == .bfloat16;
            const xx_bf = if (share_bf) try o.cast(xx, .bfloat16) else xx;
            // The latent only goes to the cache (BF16), unless a tap reads it as the norm gives it.
            const q_dt: mlx.mlx_dtype = if (t > 1 and at == null and L.indexer == null and fwd.linearInputDtype(L.attn.q_b) == .bfloat16) .bfloat16 else .float32;
            const pp = try fwd.projectAsQ(&o, xx_bf, L.attn, L.geometry, p0, if (at == null) .bfloat16 else .float32, q_dt);
            try fwd.At.input(at, .attn_in, xx);
            try fwd.At.input(at, .q_lora, pp.q_resid);
            try fwd.At.input(at, .kv_latent, pp.c);
            if (L.indexer != null) {
                try self.appendLatent(&o, st, pp.c, pp.k_pe);
            } else {
                const w: c_int = @intCast(c.window);
                const c_bf = try o.cast(pp.c, .bfloat16);
                const pe_bf = try o.cast(pp.k_pe, .bfloat16);
                if (t == 1 and self.kv.scheme != .off and hy4_latent.appendApplies(o.s, c_bf, pe_bf, self.kv.group_size, self.kv.bits, w)) {
                    try self.writeRingLatent(st, c_bf, pe_bf, p0, t);
                    ring_written = true;
                } else {
                    fresh = try self.freshRows(&o, &keep, c_bf, pe_bf);
                    if (self.kv.scheme != .off and hy4_latent.appendApplies(o.s, c_bf, pe_bf, self.kv.group_size, self.kv.bits, w)) {
                        ring_c = try keep.own(try o.result(c_bf));
                        ring_pe = try keep.own(try o.result(pe_bf));
                    }
                }
            }
            if (L.indexer) |iw| {
                const ig = self.indexerGeometry();
                try poolClosed(&o, st, try fwd.tokenKeys(&o, xx, iw, ig), iw, ig);
                if (n > ig.group_topk * ig.ratio) pooled = try keep.slice(st.pooled.?.buf(0), 0, 0, @intCast(st.pooled.?.used));
            }
            h = try keep.own(try o.result(hh));
            post_a = try keep.own(try o.result(a.post));
            x = try keep.own(try o.result(xx));
            if (share_bf) x_bf = try keep.own(try o.result(xx_bf));
            p = .{ .q = try keep.own(try o.result(pp.q)), .c = undefined, .k_pe = undefined, .q_resid = try keep.own(try o.result(pp.q_resid)), .q_raw = if (pp.q_raw) |r| try keep.own(try o.result(r)) else null };
        }
        // The window a sliding layer's spans read: a decode row writes its row into the ring and reads the ring as it
        // stands; a chunk reads the ring's last `window - 1` rows in position order before its own rows, and writes its
        // last `window` rows after the spans.
        var win: LayerState = .{};
        if (L.indexer == null) {
            if (t == 1) {
                if (!ring_written) try self.writeRing(st, &fresh, p0, t);
                win = st.*;
                win.unordered = true;
            } else win = try self.windowKeys(&keep, st, &fresh, p0);
        }
        // The keys a wide span's matrix-unit core reads, at the dequantize's width, made once for the chunk's spans.
        const core_keys: ?hy4_attn.Keys = if (fwd.coreServes(self.s, L.attn, L.geometry, t)) try self.coreKeys(&keep, L, st, &win) else null;
        // Phase 2: attention a span of rows at a time, each in a scope of its own over only the keys it reads. The gate
        // and the spans' gated rows are held until W_o has read them; W_kvb is widened once for every span.
        const span = if (pooled != null) @min(self.span_rows, gathered_span) else self.span_rows;
        const loop_span: c_int = blk: {
            if (pooled == null or core_keys == null or !mlx.engine.isStream(self.s) or !fwd.spans_share or self.span_rows <= span) break :blk span;
            break :blk self.spanWidth(span, mlx.getShape(pooled.?)[0], p0);
        };
        var held: Ops = .{ .s = self.s };
        defer held.deinit();
        var in: fwd.SpanIn = undefined;
        {
            var o: Ops = .{ .s = self.s };
            defer o.deinit();
            in.gate = try held.own(try o.result(try fwd.linearSpans(&o, x_bf orelse x, L.attn.gate, span)));
            in.kv_b32 = if (t > 1 and fwd.spans_share and !fwd.absorbServes(self.s, L.attn.kv_b, L.geometry, t)) try held.own(try o.result(try o.cast(L.attn.kv_b, .float32))) else null;
            in.whole = t <= span;
            in.index = null;
            if (L.indexer) |iw| if (pooled) |all| if (!in.whole) {
                const ig = self.indexerGeometry();
                const thr = ig.group_topk * ig.ratio;
                const sel_r0: c_int = @max(0, thr - p0);
                const idx_r0: c_int = if (sel_r0 == 0) 0 else @divTrunc(sel_r0 + span - 1, span) * span;
                if (idx_r0 < t) if (try fwd.indexPrepare(&o, x, p.q_resid, all, iw, ig, p0, idx_r0, span)) |prep| {
                    in.index = .{
                        .r0 = prep.r0,
                        .qt = try held.own(try o.result(prep.qt)),
                        .hw = try held.own(try o.result(prep.hw)),
                        .k_rot = try held.own(try o.result(prep.k_rot)),
                    };
                };
            };
        }
        // W_o's rows (or its partials) go to the iHC mix's launch, which norms them (no tap reads the attention's output).
        var o_left: fwd.OLeft = .normed;
        in.o_left = null;
        if (at == null and in.whole) {
            var o: Ops = .{ .s = self.s };
            defer o.deinit();
            if (fwd.oPartsReady(&o, h, L.mix_ffn, L.attn)) in.o_left = &o_left;
        }
        var spans: [max_spans]Arr = undefined;
        var count: usize = 0;
        // On the engine, a chunk of several spans writes each span's gated rows into one block of the chunk's rows at
        // their offset, made before the spans: nothing holds every span's rows and their join at once (D76). MLX's
        // stream joins them, where a write would copy the block.
        const placed = !in.whole and mlx.engine.isStream(self.s);
        var rows_block: ?Arr = null;
        defer if (rows_block) |b| {
            _ = mlx.mlx_array_free(b);
        };
        if (placed) {
            var b = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(b);
            try mlx.check(mlx.engine.freshArray(&b, &.{ t, L.geometry.heads * L.geometry.v }, fwd.linearInputDtype(L.attn.o)));
            rows_block = b;
        }
        in.dst = if (rows_block != null and at == null) .{ .block = rows_block.?, .row = 0 } else null;
        var r0: c_int = 0;
        while (r0 < t) {
            if (count == max_spans) return error.Hy4ChunkTooWide;
            const rem = t - r0;
            const step: c_int = if (rem >= loop_span) loop_span else if (loop_span > span and rem >= span) @divTrunc(rem, span) * span else @min(rem, span);
            const r1 = r0 + step;
            in.rows_as = if (step > span) span else null;
            var so: Ops = .{ .s = self.s };
            defer so.deinit();
            const rows_out = try self.attendSpan(&so, l, x, p, in, pooled, &win, core_keys, r0, r1, n, p0, at);
            r0 = r1;
            count += 1;
            if (rows_out.ctx == null) continue;
            const b = rows_block orelse {
                spans[count - 1] = try held.own(try so.result(rows_out));
                continue;
            };
            const next = blk: {
                var up = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(up);
                break :blk try mlx_pages.InPlace.adopt(b, .ring).write(&up, rows_out, &.{ r0, 0 }, &.{ r1, L.geometry.heads * L.geometry.v }, self.s);
            };
            _ = mlx.mlx_array_free(b);
            rows_block = next.array;
            try next.landed(self.s);
        }
        if (L.indexer == null and t > 1) {
            if (ring_c != null) try self.writeRingLatent(st, ring_c.?, ring_pe.?, p0, t) else try self.writeRing(st, &fresh, p0, t);
        }
        // W_o and its norm once over the chunk's gated rows, each row as its span's own launch gives it.
        var y: Arr = undefined;
        const fuse_o_norm = !in.whole and at == null and fwd.norm_kernel_on and fwd.mix_expand_on and fwd.mix_norm_on and fwd.hc_kernels_on and mlx.streamIsGpu(self.s);
        {
            var o: Ops = .{ .s = self.s };
            defer o.deinit();
            // The chunk's only span ran W_o and its norm itself (`SpanIn.whole`).
            y = if (in.whole) try keep.own(try o.result(spans[0])) else blk: {
                const gated = if (rows_block) |b| b else try o.concat(spans[0..count], 0);
                break :blk if (fuse_o_norm)
                    try keep.own(try o.result(try fwd.linearSpans(&o, gated, L.attn.o, span)))
                else
                    try keep.own(try o.result(try fwd.output(&o, gated, L.attn, L.geometry, span)));
            };
        }
        held.deinit();
        held.count = 0;
        if (rows_block) |b| _ = mlx.mlx_array_free(b);
        rows_block = null;
        // Phase 3: the attention write-back, the FFN mix and norm.
        var mid: Arr = undefined;
        var post_f: Arr = undefined;
        var fx: Arr = undefined;
        {
            var o: Ops = .{ .s = self.s };
            defer o.deinit();
            const em = if (fuse_o_norm)
                try fwd.expandNormMixNorm(&o, h, post_a, y, L.attn.post_norm, L.geometry.eps, L.mix_ffn, hc, c.hc_magnitude, c.hc_eps, c.eps, L.post_norm)
            else switch (o_left) {
                .parts => |osplit| try fwd.expandMixNormParts(&o, h, post_a, y, osplit, L.attn, L.geometry.eps, L.mix_ffn, c.hc_magnitude, c.hc_eps, c.eps, L.post_norm),
                .rows => try fwd.expandMixNormRows(&o, h, post_a, y, L.attn, L.geometry.eps, L.mix_ffn, c.hc_magnitude, c.hc_eps, c.eps, L.post_norm),
                .normed => try fwd.expandMixNorm(&o, h, post_a, y, L.mix_ffn, hc, c.hc_magnitude, c.hc_eps, c.eps, L.post_norm),
            };
            const m = em.m;
            const f = em.mixed;
            mid = try keep.own(try o.result(m));
            post_f = try keep.own(try o.result(f.post));
            fx = try keep.own(try o.result(f.x));
            try fwd.At.input(at, .ffn_in, fx);
        }
        var front: Front = .{ .mid = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .x = mlx.mlx_array_new() };
        errdefer front.deinit();
        try mlx.check(mlx.mlx_array_set(&front.mid, mid));
        try mlx.check(mlx.mlx_array_set(&front.post, post_f));
        try mlx.check(mlx.mlx_array_set(&front.x, fx));
        return front;
    }

    /// Phase 4: layer `l`'s FFN over the FFN input `x` of one piece (a MoE runs each routed expert in a scope of its
    /// own). The caller frees the result.
    pub fn ffnOne(self: *Model, l: usize, x: Arr) !Arr {
        return (try self.ffnFor(l, x, null)).y;
    }

    /// `ffnOne`, and with `back` the streams after the write-back where the FFN's last launch writes them (a resident
    /// STQ MoE's combine norm); `expanded` says which the result is. The caller frees it.
    fn ffnFor(self: *Model, l: usize, x: Arr, back: ?fwd.Back) !fwd.Fin {
        const exact = mlx.engine.exactF32();
        defer exact.end();
        const at: ?fwd.At = if (self.tap) |tp| .{ .tap = tp, .layer = l } else null;
        var o: Ops = .{ .s = self.s };
        defer o.deinit();
        const fin: fwd.Fin = switch (self.layers[l].ffn) {
            .dense => |d| .{ .y = try fwd.swigluNormed(&o, x, d.sw, d.post_norm, self.config.eps, at), .expanded = false },
            .moe => |m| try moeLayerImpl(self, &o, l, x, m, at, back),
        };
        const next: ?fwd.Mixed = if (fin.next) |nx| .{ .post = try o.result(nx.post), .x = try o.result(nx.x) } else null;
        errdefer freeMixed(next);
        return .{ .y = try o.result(fin.y), .expanded = fin.expanded, .next = next };
    }

    /// Whether `ffnBatch` takes layer `l`'s FFN: the BF16-bank MoE of a layer whose experts are all pinned.
    pub fn ffnBatched(self: *const Model, l: usize) bool {
        const p = self.pinned orelse return false;
        return p.layer == l and self.layers[l].ffn == .moe and self.stq_experts[l] == null and self.quant.expert == null and
            mlx.streamIsGpu(self.s);
    }

    /// Phase 4 over several pieces of layer `l` at once (`ffnBatched`), each piece's `fy[i]` the bytes `ffnOne` gives it: the
    /// route and latent per piece, the routed experts over every piece's rows together, the tail per piece. The taps see
    /// the pieces in order; the caller frees `fy`.
    pub fn ffnBatch(self: *Model, l: usize, x: []const Arr, fy: []Arr) !void {
        const exact = mlx.engine.exactF32();
        defer exact.end();
        const gpa = self.gpa;
        const at: ?fwd.At = if (self.tap) |tp| .{ .tap = tp, .layer = l } else null;
        const m = self.layers[l].ffn.moe;
        const g = self.moeGeometry();
        const prepared = self.pinned.?.prepared;
        const Piece = struct { picks: Arr = .{ .ctx = null }, weights: Arr = .{ .ctx = null }, lat: Arr = .{ .ctx = null }, sum: Arr = .{ .ctx = null } };
        const pieces = try gpa.alloc(Piece, x.len);
        defer gpa.free(pieces);
        @memset(pieces, .{});
        defer for (pieces) |p| inline for (.{ p.picks, p.weights, p.lat, p.sum }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        for (x, pieces) |xi, *p| {
            var o: Ops = .{ .s = self.s };
            defer o.deinit();
            const r = try fwd.route(&o, xi, m.router, g.top_k, g.scale);
            p.picks = try o.result(try o.contiguous(try o.cast(r.picks, .int32)));
            p.weights = try o.result(r.weights);
            p.lat = try o.result(try fwd.linear(&o, xi, m.lat_down));
        }
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (pieces) |p| for ([_]Arr{ p.picks, p.weights, p.lat }) |a| try mlx.check(mlx.mlx_vector_array_append_value(vec, a));
            try mlx.check(mlx.mlx_eval(vec));
        }
        const lats = try gpa.alloc(Arr, x.len);
        defer gpa.free(lats);
        const weights = try gpa.alloc(Arr, x.len);
        defer gpa.free(weights);
        const sums = try gpa.alloc(Arr, x.len);
        defer gpa.free(sums);
        var picks: usize = 0;
        for (pieces, lats, weights) |p, *lat, *w| {
            lat.* = p.lat;
            w.* = p.weights;
            picks += mlx.mlx_array_size(p.picks);
        }
        const experts = try gpa.alloc(u32, picks);
        defer gpa.free(experts);
        const rows = try gpa.alloc(u32, picks);
        defer gpa.free(rows);
        var off: usize = 0;
        for (pieces) |p| {
            const host = (mlx.mlx_array_data_int32(p.picks) orelse return error.HostRead)[0..mlx.mlx_array_size(p.picks)];
            for (host, experts[off..][0..host.len], rows[off..][0..host.len]) |e, *d, *r| {
                d.* = @intCast(e);
                r.* = prepared.remapped[d.*];
            }
            if (at) |a| {
                try a.tap.routes(a.tap.ctx, a.layer, experts[off..][0..host.len], p.weights);
                try fwd.At.input(at, .expert_in, p.lat);
            }
            off += host.len;
        }
        try fwd.routedBatched(gpa, self.s, lats, weights, rows, experts, g, prepared.raw_gate_up, prepared.raw_down, at, sums);
        defer for (sums) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for (x, sums, fy) |xi, sum, *out| {
            var o: Ops = .{ .s = self.s };
            defer o.deinit();
            out.* = try o.result(try o.contiguous(try fwd.moeFinish(&o, xi, m, g, sum, at)));
        }
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (fy) |a| try mlx.check(mlx.mlx_vector_array_append_value(vec, a));
        try mlx.check(mlx.mlx_eval(vec));
    }

    fn moeGeometry(self: *const Model) fwd.MoeGeometry {
        const c = self.config;
        return .{ .top_k = @intCast(c.top_k), .inter = @intCast(c.moe_inter), .scale = c.route_scale, .swiglu_limit = c.swiglu_limit, .eps = c.eps };
    }

    /// Phase 5: the FFN write-back; a new array the caller owns.
    pub fn layerBack(self: *Model, front: Front, fy: Arr) !Arr {
        var o: Ops = .{ .s = self.s };
        defer o.deinit();
        return o.result(try fwd.expand(&o, front.mid, front.post, fy));
    }
};

/// The MoE layer `l` over `x`: its banks resident, or its picks' slots from the expert stream (the picks read to the
/// host, a sync the composed path takes each MoE layer).
fn moeLayerImpl(self: *Model, o: *Ops, l: usize, x: Arr, m: fwd.Moe, at: ?fwd.At, back: ?fwd.Back) !fwd.Fin {
    const g = self.moeGeometry();
    if (self.stq_experts[l]) |*rb| return moeStqRouted(self, o, x, m, g, rb, null, at, back);
    const engine = self.experts orelse return .{ .y = (try fwd.moe(self.gpa, o, x, m, g, at)).y, .expanded = false };
    if (self.quant.expert != null) {
        // A resident layer's slots never move: its picks map to them on the GPU, with no host read.
        if (engine.residentRoute(@intCast(l))) |rr| return moeStqRouted(self, o, x, m, g, rr.bank, rr.map, at, back);
        return .{ .y = try moeStqStreamed(self, o, l, x, m, g, engine, at), .expanded = false };
    }
    return .{ .y = try moeLayerRest(self, o, l, x, m, g, engine, at), .expanded = false };
}

/// `moeLayerImpl` over a BF16 bank's experts from the stream.
fn moeLayerRest(self: *Model, o: *Ops, l: usize, x: Arr, m: fwd.Moe, g: fwd.MoeGeometry, engine: *expert_stream.Engine, at: ?fwd.At) !Arr {
    const r = try fwd.route(o, x, m.router, g.top_k, g.scale);
    const experts: usize = @intCast(mlx.getShape(m.router.weight)[0]);
    // The picks steer the groups: a dry recording makes them up and plans the most groups and picks there can be (D20).
    const host = try fwd.hostPicks(self.gpa, o, r.picks, @intCast(experts));
    defer self.gpa.free(host);
    if (at) |a| try a.tap.routes(a.tap.ctx, a.layer, host, r.weights);
    const lat = try fwd.linear(o, x, m.lat_down);
    try fwd.At.input(at, .expert_in, lat);
    const tap_all: ?fwd.ExpertTap = if (at) |a| .{ .at = a, .experts = host } else null;
    // A pinned layer's picks run as one group over its held slots.
    if (self.pinned) |*p| if (p.layer == l) {
        const rows = try self.gpa.alloc(u32, host.len);
        defer self.gpa.free(rows);
        for (host, rows) |e, *d| d.* = p.prepared.remapped[e];
        const acc = try o.own(try fwd.routedExperts(self.gpa, o.s, lat, r.weights, null, rows, g, p.prepared.raw_gate_up, p.prepared.raw_down, tap_all));
        return moeEnd(o, x, m, g, acc, at);
    };
    // The stream holds no union workspace: each route asks for at most a layer's cache slots of distinct experts. The
    // picks go in groups of distinct experts that fit; each group lands its picks at their positions in one [T·k, L]
    // sum, so the rows sum their k as one group's would.
    const slots = engine.slotsPerLayer();
    const groups = try expertGroups(self.gpa, host, slots);
    defer self.gpa.free(groups);
    const n_groups = try plan.bound(groupCount(groups), mostGroups(host.len, experts, slots));
    const occurrences = try self.gpa.alloc(u16, host.len);
    defer self.gpa.free(occurrences);
    // A group's expert ids for the tap, kept only when tapped.
    const tapped = try self.gpa.alloc(u32, if (at != null) host.len else 0);
    defer self.gpa.free(tapped);
    const picks = try self.gpa.alloc(u32, host.len);
    defer self.gpa.free(picks);
    const rows = try self.gpa.alloc(u32, host.len);
    defer self.gpa.free(rows);
    const t = mlx.getShape(lat)[0];
    var placed: ?Arr = null;
    defer if (placed) |a| {
        _ = mlx.mlx_array_free(a);
    };
    // One group reads its slots in place; more each land their picks in one sum: a variant each (`plan.choose`).
    const arm = try plan.frame(@src(), .{});
    defer arm.end();
    const many = if (mostGroups(host.len, experts, slots) > 1) plan.choose(@src(), bool, n_groups > 1) else false;
    const kind = if (plan.planned()) try plan.each(@src(), many) else plan.Frame.none;
    defer kind.end();
    // The arm keys the loop and its iterations: the one-group arm ends the layer inside its only iteration.
    const loop = try plan.frame(@src(), many);
    defer loop.end();
    // The groups land their picks in one sum made before them: no group hands one up.
    if (many) placed = try fwd.placedZeros(o.s, t, g.top_k, mlx.getShape(lat)[1]);
    var gi: u32 = 0;
    while (gi < (if (many) n_groups else 1)) : (gi += 1) {
        const it = if (plan.planned()) try plan.each(@src(), .{ many, gi > 0 }) else plan.Frame.none;
        defer it.end();
        var picked: usize = 0;
        for (host, groups, 0..) |v, gv, i| if (gv == gi) {
            occurrences[picked] = @intCast(v);
            if (at != null) tapped[picked] = v;
            picks[picked] = @intCast(i);
            picked += 1;
        };
        // A dry recording's made-up picks fill the widest group there can be: every pick.
        const n = try plan.bound(picked, host.len);
        for (picked..n) |i| {
            occurrences[i] = @intCast(host[i]);
            if (at != null) tapped[i] = host[i];
            picks[i] = @intCast(i);
        }
        var prepared = try prepareRoute(engine, @intCast(l), occurrences[0..n], @min(n, slots));
        defer prepared.deinit();
        for (prepared.remapped, rows[0..n]) |v, *d| d.* = v;
        const tap: ?fwd.ExpertTap = if (at) |a| .{ .at = a, .experts = tapped[0..n] } else null;
        // Read from the route's slots: evaluated before they are given back.
        if (!many) {
            const acc = try o.own(try fwd.routedExperts(self.gpa, o.s, lat, r.weights, null, rows[0..n], g, prepared.raw_gate_up, prepared.raw_down, tap));
            try mlx.check(mlx.mlx_array_eval(acc));
            return moeEnd(o, x, m, g, acc, at);
        }
        try fwd.placeExperts(self.gpa, o.s, &placed.?, lat, r.weights, picks[0..n], rows[0..n], g, prepared.raw_gate_up, prepared.raw_down, tap);
    }
    const acc = try o.own(try fwd.sumPlaced(o.s, placed.?, t, g.top_k));
    // Freed inside the arm's frame: both arms hand up the same arrays, the sum and what the layer's end makes.
    _ = mlx.mlx_array_free(placed.?);
    placed = null;
    return moeEnd(o, x, m, g, acc, at);
}

/// The MoE layer's end after its routed sum `acc` (in the latent): `moeFinish`, evaluated.
fn moeEnd(o: *Ops, x: Arr, m: fwd.Moe, g: fwd.MoeGeometry, acc: Arr, at: ?fwd.At) !Arr {
    const y = try o.contiguous(try fwd.moeFinish(o, x, m, g, acc, at));
    try mlx.check(mlx.mlx_array_eval(y));
    return y;
}

/// An STQ layer's MoE over resident banks `rb`: the route, then the routed experts through the clamped trellis chain
/// over the layer's rate groups, then the latent up, the shared expert and the combine norm (`fwd.moeFinish`), as the
/// BF16 path. The picks index `rb`, or map through `slot_map` [experts] int32 first (a resident stream's slots); they
/// stay on the GPU and nothing waits for the layer.
fn moeStqRouted(self: *Model, o: *Ops, x: Arr, m: fwd.Moe, g: fwd.MoeGeometry, rb: *const exl3.Ragged, slot_map: ?Arr, at: ?fwd.At, back: ?fwd.Back) !fwd.Fin {
    const rm = try fwd.routeMapped(o, x, m.router, g.top_k, g.scale, slot_map);
    const r = rm.routed;
    if (at) |a| {
        const host = try fwd.hostPicks(self.gpa, o, r.picks, @intCast(mlx.getShape(m.router.weight)[0]));
        defer self.gpa.free(host);
        try a.tap.routes(a.tap.ctx, a.layer, host, r.weights);
    }
    const share_bf = mlx.getShape(x)[0] > 1 and at == null and fwd.linearInputDtype(m.lat_down) == .bfloat16 and
        fwd.linearInputDtype(m.shared.gate) == .bfloat16 and fwd.linearInputDtype(m.shared.up) == .bfloat16;
    const x_bf = if (share_bf) try o.cast(x, .bfloat16) else x;
    const lat = try fwd.expertLatent(o, x_bf, m.lat_down, at);
    try fwd.At.input(at, .expert_in, lat);
    const picks = rm.slots orelse r.picks;
    const routed_dt: mlx.mlx_dtype = if (mlx.getShape(x)[0] > 1 and at == null and fwd.linearInputDtype(m.lat_up) == .bfloat16) .bfloat16 else .float32;
    const routed = try o.own(try hy4_stq.routedAs(o.s, self.quant.expert.?, g.swiglu_limit, rb, lat, picks, r.weights, routed_dt));
    const fin = try fwd.moeFinishBack(o, x_bf, m, g, routed, at, back);
    return .{ .y = try o.contiguous(fin.y), .expanded = fin.expanded, .next = fin.next };
}

/// `--expert-cache-bias`'s penalty on layer `l`'s route as the forward's own host array, its bytes the same whatever the
/// cache holds, so a planned forward asks it of its host ring (D102); null where the stream scores none.
fn cachePenalty(self: *Model, o: *Ops, engine: *expert_stream.Engine, l: usize, router: fwd.Router) !?Arr {
    const experts: usize = @intCast(mlx.getShape(router.weight)[0]);
    const host = try self.gpa.alloc(f32, experts);
    defer self.gpa.free(host);
    if (!try engine.cachePenaltyHost(@intCast(l), host)) return null;
    return try o.own(try mlx.newData(host.ptr, &[_]c_int{@intCast(experts)}, .float32));
}

/// A streamed STQ layer's MoE: the picks in groups of distinct experts the layer's cache slots hold (no union
/// workspace), each group's experts filled into slots of the stream's rate-group bank (`Prepared.ragged`, indexed by
/// the slot ids in `remapped`); every [T, k] pick runs the clamped chain, a pick outside the group on the group's first
/// slot at weight 0 (it adds exactly nothing), and the groups' sums add.
fn moeStqStreamed(self: *Model, o: *Ops, l: usize, x: Arr, m: fwd.Moe, g: fwd.MoeGeometry, engine: *expert_stream.Engine, at: ?fwd.At) !Arr {
    // `--expert-cache-bias`: a decode-width pick favours the experts the layer's cache holds; a prefill chunk is exact.
    const pen = if (mlx.getShape(x)[0] <= exl3.kernels.DECODE_ROWS_MAX) try cachePenalty(self, o, engine, l, m.router) else null;
    const r = try fwd.route(o, x, try fwd.cacheBiased(o, m.router, pen), g.top_k, g.scale);
    const experts: usize = @intCast(mlx.getShape(m.router.weight)[0]);
    // The picks steer the groups: a dry recording makes them up and plans the most groups there can be (D20).
    const host = try fwd.hostPicks(self.gpa, o, r.picks, @intCast(experts));
    defer self.gpa.free(host);
    if (at) |a| try a.tap.routes(a.tap.ctx, a.layer, host, r.weights);
    const slots_per_layer = engine.slotsPerLayer();
    const groups = try expertGroups(self.gpa, host, slots_per_layer);
    defer self.gpa.free(groups);
    const n_groups = try plan.bound(groupCount(groups), mostGroups(host.len, experts, slots_per_layer));
    const occurrences = try self.gpa.alloc(u16, host.len);
    defer self.gpa.free(occurrences);
    const slots = try self.gpa.alloc(u32, host.len);
    defer self.gpa.free(slots);
    const mask_v = try self.gpa.alloc(f32, host.len);
    defer self.gpa.free(mask_v);
    const lat = try fwd.expertLatent(o, x, m.lat_down, at);
    try fwd.At.input(at, .expert_in, lat);
    const shape = mlx.getShape(r.weights);
    const dec = self.quant.expert.?;
    var routed: ?Arr = null;
    errdefer if (routed) |acc| {
        _ = mlx.mlx_array_free(acc);
    };
    // Every group runs every pick, those outside it at weight 0: a group's sizes are the picks', the first adds nothing.
    const loop = try plan.frame(@src(), .{});
    defer loop.end();
    var gi: u32 = 0;
    while (gi < n_groups) : (gi += 1) {
        const it = if (plan.planned()) try plan.each(@src(), gi > 0) else plan.Frame.none;
        defer it.end();
        var first: u16 = 0;
        for (host, groups) |v, gv| if (gv == gi) {
            first = @intCast(v);
            break;
        };
        for (host, groups, occurrences, mask_v) |v, gv, *d, *mv| {
            d.* = if (gv == gi) @intCast(v) else first;
            mv.* = if (gv == gi) 1 else 0;
        }
        const distinct = @min(host.len, slots_per_layer);
        var prepared = try prepareRoute(engine, @intCast(l), occurrences, distinct);
        defer prepared.deinit();
        const rb = prepared.ragged orelse {
            std.log.err("hy_v4_flash: layer {d}'s streamed experts are not a rate-group bank", .{l});
            return error.Hy4ExpertLayout;
        };
        for (prepared.remapped, slots) |v, *d| d.* = v;
        var go: Ops = .{ .s = o.s };
        defer go.deinit();
        const pk = try go.own(mlx.mlx_array_new_data(slots.ptr, &[_]c_int{ shape[0], shape[1] }, 2, .uint32));
        const mask = try go.own(mlx.mlx_array_new_data(mask_v.ptr, &[_]c_int{ shape[0], shape[1] }, 2, .float32));
        const w = try go.binary(.mul, r.weights, mask);
        // Read from the route's slots: evaluated before they are given back. Its frame is the bank's alone.
        const part = blk: {
            const kind = try routeKind(&prepared);
            defer kind.end();
            break :blk try go.own(try hy4_stq.routed(go.s, dec, g.swiglu_limit, rb, lat, pk, w));
        };
        // The first group carries a copy of its part, so every group's part dies inside it, as a later one's in its add.
        const next = if (routed) |acc| try go.result(try go.contiguous(try go.binary(.add, acc, part))) else (try copyOf(part, go.s)).?;
        try mlx.check(mlx.mlx_array_eval(next));
        if (routed) |acc| _ = mlx.mlx_array_free(acc);
        routed = next;
    }
    const acc = try o.own(routed.?);
    routed = null;
    const y = try o.contiguous(try fwd.moeFinish(o, x, m, g, acc, at));
    try mlx.check(mlx.mlx_array_eval(y));
    return y;
}

/// The most groups `picks` picks over `experts` experts can make (`expertGroups`): each group but the last holds `slots`
/// distinct experts, each at least one pick.
fn mostGroups(picks: usize, experts: usize, slots: u16) u32 {
    return @intCast(@max(1, (@min(picks, experts) + slots - 1) / slots));
}

/// A streamed route's experts in the stream's slots, filled; in a dry recording, the slots an all-hit route takes, or
/// where a rate group can overflow its pool (`Engine.routeMayOverflow`) the union workspace's as `plan.choose` holds it,
/// with no fill and the cache untouched (`Engine.prepareDry`). `distinct` is the most distinct experts it names.
fn prepareRoute(engine: *expert_stream.Engine, layer: u16, occurrences: []const u16, distinct: usize) !expert_stream.Prepared {
    if (plan.session.mode != .dry) return engine.prepareHost(layer, occurrences);
    const workspace = engine.routeMayOverflow(layer, distinct) and plan.choose(@src(), bool, false);
    return engine.prepareDry(layer, occurrences, workspace);
}

/// The frame of what a route reads, keyed by the bank `prepareRoute` handed out. Open in every layer, the ones that
/// cannot overflow included: the frames around it are keyed without the layer.
fn routeKind(prepared: *const expert_stream.Prepared) plan.Error!plan.Frame {
    if (!plan.planned()) return plan.Frame.none;
    return plan.each(@src(), prepared.workspace);
}

/// Each pick's group: the distinct experts in order of first pick, `slots` to a group, so an expert's picks are one
/// group's and a route's groups are its experts over `slots`, however many picks it has: every group runs every pick.
fn expertGroups(gpa: std.mem.Allocator, picks: []const u32, slots: u16) ![]u32 {
    const out = try gpa.alloc(u32, picks.len);
    errdefer gpa.free(out);
    var seen = std.AutoHashMap(u32, u32).init(gpa);
    defer seen.deinit();
    for (picks, out) |p, *d| {
        const e = try seen.getOrPut(p);
        if (!e.found_existing) e.value_ptr.* = seen.count() - 1;
        d.* = e.value_ptr.* / slots;
    }
    return out;
}

fn groupCount(groups: []const u32) u32 {
    var n: u32 = 0;
    for (groups) |g| n = @max(n, g + 1);
    return n;
}

/// Pools the groups the new token keys `keys` [T, d] close, after the open group's: their pooled keys join
/// `st.pooled`, and the rows of the group still open stay in `st.open`. Pooling is per group, so this equals pooling
/// every token key so far.
fn poolClosed(o: *Ops, st: *LayerState, keys: Arr, w: fwd.Indexer, g: fwd.IndexerGeometry) !void {
    const all = if (st.open) |open| try o.concat(&.{ open, keys }, 0) else keys;
    const n = mlx.getShape(all)[0];
    const closed = @divTrunc(n, g.ratio) * g.ratio;
    if (closed > 0) try appendRows(&st.pooled, &.{try fwd.pool(o, try o.slice(all, 0, 0, closed), w, g)}, o.s);
    // The rows of the group still open go into the layer's open block, in place: the forward keeps nothing of its own.
    if (st.open) |open| _ = mlx.mlx_array_free(open);
    st.open = null;
    if (closed == n) return;
    const cols = mlx.getShape(all)[1];
    try writeFixed(&st.open_buf, try o.slice(all, 0, closed, n), &.{ g.ratio - 1, cols }, o.s);
    st.open = try headRows(st.open_buf.?, @intCast(n - closed), o.s);
}

/// `value` written over the fixed block `buf` from its first element, in place; the block made zeros of `full` (and
/// `value`'s dtype) when it has none or another: it is the state's own, so the forward that writes it leaves nothing of
/// its own behind (docs/bill_redesign.md part 8). No other handle on the block may stand, or the write copies it.
pub fn writeFixed(buf: *?Arr, value: Arr, full: []const c_int, s: mlx.mlx_stream) !void {
    const dt = mlx.mlx_array_dtype(value);
    if (buf.*) |b| if (mlx.mlx_array_dtype(b) != dt or !std.mem.eql(c_int, mlx.getShape(b), full)) {
        _ = mlx.mlx_array_free(b);
        buf.* = null;
    };
    if (buf.* == null) buf.* = (try mlx_pages.ringBuffer(full, dt, s)).array;
    const sh = mlx.getShape(value);
    const start: [4]c_int = @splat(0);
    const next = blk: {
        var up = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(up);
        break :blk try mlx_pages.InPlace.adopt(buf.*.?, .ring).write(&up, value, start[0..sh.len], sh, s);
    };
    _ = mlx.mlx_array_free(buf.*.?);
    buf.* = next.array;
    // The block holds the write from here: one that fails to land is still the block's to release.
    try next.landed(s);
}

/// A copy of `src` in a buffer of its own, or null: what a point keeps, so the state it was taken from is still the
/// sole holder of its blocks and writes them in place.
fn copyOf(src: ?Arr, s: mlx.mlx_stream) !?Arr {
    const a = src orelse return null;
    var c = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(c);
    // The engine's `mlx_copy` is a share of the buffer; its own copy is `copyOwned`.
    try mlx.check(if (mlx.engine.isOurs(a)) mlx.engine.copyOwned(&c, a) else mlx.mlx_copy(&c, a, s));
    return c;
}

/// An empty store whose parts are `rows`' columns and dtypes.
fn storeOf(rows: []const Arr) row_store.Store {
    var parts: [row_store.max_parts]row_store.Part = undefined;
    for (rows, parts[0..rows.len]) |x, *pt| pt.* = .{ .cols = @intCast(mlx.getShape(x)[1]), .dtype = mlx.mlx_array_dtype(x) };
    return row_store.Store.init(parts[0..rows.len], rows_grow_div);
}

/// `dst` made a new handle on `src`, or null; the old one freed.
fn shareInto(dst: *?Arr, src: ?Arr) !void {
    if (dst.*) |old| _ = mlx.mlx_array_free(old);
    dst.* = null;
    const a = src orelse return;
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_array_set(&h, a));
    dst.* = h;
}

/// The first `n` rows of `buf`, a view.
fn headRows(buf: Arr, n: usize, s: mlx.mlx_stream) !Arr {
    const cols = mlx.getShape(buf)[1];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, buf, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ @intCast(n), cols }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return out;
}

/// A sparse layer's store grows by this fraction of its rows (`row_store`): the slack the state bill counts.
pub const rows_grow_div: usize = 16;

/// `rows` (one array a part) appended to `store`, made on the first append with the parts' columns and dtypes.
fn appendRows(store: *?row_store.Store, rows: []const Arr, s: mlx.mlx_stream) !void {
    if (store.* == null) store.* = storeOf(rows);
    try store.*.?.append(rows, s);
}

/// The most arrays a layer's state holds (`stateArrays`).
pub const max_state_arrays = 4 + 2 * row_store.max_parts + 1;

/// Every array layer state `st` holds, in `out`: its ring, its stores' buffers and its open group.
pub fn stateArrays(st: *const LayerState, out: *[max_state_arrays]Arr) []const Arr {
    var n: usize = 0;
    // The open group's block, not its view: the block is what the layer holds.
    inline for (.{ "c", "c_scales", "c_biases", "k_pe", "open_buf" }) |f| if (@field(st, f)) |a| {
        out[n] = a;
        n += 1;
    };
    for ([_]?row_store.Store{ st.rows, st.pooled }) |maybe| if (maybe) |r| for (r.bufs[0..r.n]) |b| if (b.array.ctx != null) {
        out[n] = b.array;
        n += 1;
    };
    return out[0..n];
}

/// Frees layer state `st`: its ring, its stores and its open group.
fn freeLayer(st: *LayerState) void {
    inline for (.{ "c", "c_scales", "c_biases", "k_pe", "open", "open_buf" }) |f| if (@field(st, f)) |a| {
        _ = mlx.mlx_array_free(a);
    };
    if (st.rows) |*r| r.deinit();
    if (st.pooled) |*r| r.deinit();
    st.* = .{};
}

fn freeMixed(m: ?fwd.Mixed) void {
    if (m) |v| inline for (.{ v.post, v.x }) |a| {
        _ = mlx.mlx_array_free(a);
    };
}

fn get(w: *const model_mod.Weights, comptime fmt: []const u8, args: anytype) !Arr {
    var buf: [160]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, args);
    return w.get(name) orelse {
        std.log.err("hy_v4_flash: the pack has no tensor {s}", .{name});
        return error.Hy4TensorMissing;
    };
}

fn mixAt(w: *const model_mod.Weights, l: usize, comptime which: []const u8) !fwd.Mix {
    return .{
        .fn_ = try get(w, "model.layers.{d}." ++ which ++ ".hc_pre.hc_fn", .{l}),
        .base = try get(w, "model.layers.{d}." ++ which ++ ".hc_pre.hc_base", .{l}),
        .scale = try get(w, "model.layers.{d}." ++ which ++ ".hc_pre.hc_scale", .{l}),
    };
}

test "hy4 latent store: at the 8-bit width, rows appended in two chunks read back by range and by gather as one affine round trip" {
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var o: Ops = .{ .s = cpu };
    defer o.deinit();
    var vals: [10 * 128]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.37) * @as(f32, @floatFromInt(1 + i % 7));
    const all = try o.own(mlx.mlx_array_new_data(&vals, &[_]c_int{ 10, 128 }, 2, .float32));
    const pe = try o.slice(all, 1, 0, 64);
    var m: Model = undefined;
    m.kv = .engine_default;
    var st: LayerState = .{};
    defer freeLayer(&st);
    try m.appendLatent(&o, &st, try o.slice(all, 0, 0, 3), try o.slice(pe, 0, 0, 3));
    try m.appendLatent(&o, &st, try o.slice(all, 0, 3, 10), try o.slice(pe, 0, 3, 10));
    try std.testing.expectEqual(@as(usize, 10), st.rows.?.used);
    var whole = try kv_quant.quantizeAffine(cpu, try o.cast(all, .bfloat16), 64, 8);
    defer whole.deinit();
    const want = try o.cast(try o.own(try kv_quant.dequantizeAffine(cpu, whole.q, whole.scales, whole.biases, 64, 8)), .float32);
    var keys = m.sparseKeys(&st);
    try expectSame(&o, try m.latentRange(&o, &keys, 2, 9), try o.slice(want, 0, 2, 9));
    const idx_v = [_]u32{ 9, 0, 4, 4, 7 };
    const idx = try o.own(mlx.mlx_array_new_data(&idx_v, &[_]c_int{5}, 1, .uint32));
    try expectSame(&o, try m.latentAt(&o, &keys, idx), try o.take(want, idx, 0));
    try expectSame(&o, try o.cast(try o.slice(keys.k_pe.?, 0, 0, 10), .float32), try o.cast(try o.cast(pe, .bfloat16), .float32));
}

fn expectSame(o: *Ops, got: Arr, want: Arr) !void {
    const g = try o.contiguous(got);
    const w = try o.contiguous(want);
    try mlx.check(mlx.mlx_array_eval(g));
    try mlx.check(mlx.mlx_array_eval(w));
    const gv = (mlx.mlx_array_data_float32(g) orelse return error.HostRead)[0..mlx.mlx_array_size(g)];
    const wv = (mlx.mlx_array_data_float32(w) orelse return error.HostRead)[0..mlx.mlx_array_size(w)];
    try std.testing.expectEqualSlices(f32, wv, gv);
}

test "hy4 expert groups: a chunk's picks through a cache of fewer slots than its experts group every expert once, so its groups are its experts over the slots" {
    const a = std.testing.allocator;
    const picks = try a.alloc(u32, 4096 * 16);
    defer a.free(picks);
    for (picks, 0..) |*p, i| p.* = @intCast(i % 16);
    for ([_]struct { slots: u16, groups: u32 }{ .{ .slots = 1, .groups = 16 }, .{ .slots = 3, .groups = 6 }, .{ .slots = 16, .groups = 1 } }) |c| {
        const groups = try expertGroups(a, picks, c.slots);
        defer a.free(groups);
        try std.testing.expectEqual(c.groups, groupCount(groups));
        // The picks of an expert are one group's, however far apart they fall.
        for (picks, groups) |p, g| try std.testing.expectEqual(@as(u32, p / c.slots), g);
    }
}
