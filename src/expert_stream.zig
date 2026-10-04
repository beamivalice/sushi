const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const io_mod = @import("expert_io.zig");
const kernels = @import("expert_bf16_kernels.zig");
pub const quant = @import("expert_quant.zig");

pub const Span = struct {
    offset: u64,
    len: u64,
};

pub const TensorLayout = struct {
    data_offset: u64,
    tensor_offset: u64,
    tensor_bytes: u64,
    experts: u16,

    pub fn expertSpan(self: TensorLayout, expert: u16) !Span {
        if (expert >= self.experts) return error.ExpertOutOfRange;
        if (self.experts == 0 or self.tensor_bytes % self.experts != 0) return error.InvalidExpertTensor;
        const len = self.tensor_bytes / self.experts;
        const base = std.math.add(u64, self.data_offset, self.tensor_offset) catch return error.InvalidExpertTensor;
        const delta = std.math.mul(u64, expert, len) catch return error.InvalidExpertTensor;
        return .{ .offset = std.math.add(u64, base, delta) catch return error.InvalidExpertTensor, .len = len };
    }
};

pub const CachePlan = struct {
    slots_per_layer: u16,
    expert_bytes: u64,
    cache_bytes: u64,
    workspace_bytes: u64,
    bounce_bytes: u64,
    layer_bytes: u64,
    prefill_peak_bytes: u64,
};

pub fn fillPeakBytes(expert_bytes: u64, union_size: u16, slots_per_layer: u16, batch_bytes: u64) u64 {
    const workspace = @as(u64, union_size) *| expert_bytes;
    const layer = @as(u64, slots_per_layer) *| expert_bytes;
    return workspace +| layer +| workspace +| @min(workspace, batch_bytes);
}

pub const BOUNCE_BYTES: u64 = 8 * 64 * 1024 * 1024;

pub fn expertBytes(gate_up_rows: u32, hidden: u32, intermediate: u32) !u64 {
    if (gate_up_rows == 0 or hidden == 0 or intermediate == 0) return error.InvalidExpertGeometry;
    const gate_elems = std.math.mul(u64, gate_up_rows, hidden) catch return error.InvalidExpertGeometry;
    const down_elems = std.math.mul(u64, hidden, intermediate) catch return error.InvalidExpertGeometry;
    return std.math.mul(u64, std.math.add(u64, gate_elems, down_elems) catch return error.InvalidExpertGeometry, 2) catch return error.InvalidExpertGeometry;
}

pub const moeLayerCount = quant.moeLayerCount;

fn warmSlotCount(slots_per_layer: u16) u16 {
    return @intCast((@as(u32, slots_per_layer) * 4) / 5);
}

/// Native MXFP4 bytes per expert: two U32-packed gate/up weights, one U32
/// down weight, and their U8 e8m0 scale banks. Bias slots are absent.
pub fn mxfp4ExpertBytes(hidden: u32, intermediate: u32) !u64 {
    if (hidden == 0 or intermediate == 0 or hidden % 32 != 0 or intermediate % 32 != 0)
        return error.InvalidExpertGeometry;
    const hidden_u: u64 = hidden;
    const intermediate_u: u64 = intermediate;
    const gate_weight = std.math.mul(u64, intermediate_u, hidden_u / 8) catch return error.InvalidExpertGeometry;
    const down_weight = std.math.mul(u64, hidden_u, intermediate_u / 8) catch return error.InvalidExpertGeometry;
    const gate_scale = std.math.mul(u64, intermediate_u, hidden_u / 32) catch return error.InvalidExpertGeometry;
    const down_scale = std.math.mul(u64, hidden_u, intermediate_u / 32) catch return error.InvalidExpertGeometry;
    const weight_elements = gate_weight *| 2 +| down_weight;
    const weight_bytes = std.math.mul(u64, weight_elements, 4) catch return error.InvalidExpertGeometry;
    const scale_bytes = gate_scale *| 2 +| down_scale;
    return weight_bytes +| scale_bytes;
}

/// Per-expert bytes of whichever routed-expert layout this checkpoint ships:
/// the fused bf16 pair, or the sum of the nine quantized slices the pack stores.
pub fn expertBytesFor(allocator: std.mem.Allocator, model_dir: []const u8, geometry: Geometry, layout: quant.Layout) !u64 {
    switch (layout) {
        .bf16_fused => return expertBytes(2 * geometry.intermediate, geometry.hidden, geometry.intermediate),
        .bf16_individual, .quantized_split, .mxfp4_split, .mxfp4_individual, .exl3_k4 => {
            var store = try quant.QuantStore.openForLayout(allocator, model_dir, geometry, layout);
            defer store.deinit();
            return store.expertBytes();
        },
    }
}

pub fn exl3ExpertBytes(geometry: Geometry) !u64 {
    const h: u64 = geometry.hidden;
    const i: u64 = geometry.intermediate;
    if (h == 0 or i == 0 or h % 16 != 0 or i % 16 != 0) return error.InvalidExpertGeometry;
    if (quant.expert_exl3.kFromPackedDim(geometry.exl3_n) == null) return error.InvalidExpertGeometry;
    const tile_words: u64 = geometry.exl3_n;
    const gate_up = (h / 16) * (i / 16) * tile_words * 2 + h * 2 + i * 2;
    const down = (i / 16) * (h / 16) * tile_words * 2 + i * 2 + h * 2;
    const both = std.math.mul(u64, gate_up, 2) catch return error.InvalidExpertGeometry;
    return std.math.add(u64, both, down) catch return error.InvalidExpertGeometry;
}

pub fn cachePlanBytes(requested_bytes: u64, layers: u16, experts: u16, expert_bytes: u64) !CachePlan {
    if (layers == 0 or experts == 0 or expert_bytes == 0) return error.InvalidExpertGeometry;
    const per_slot_all_layers = std.math.mul(u64, layers, expert_bytes) catch return error.InvalidExpertGeometry;
    const slots_u64 = @min(requested_bytes / per_slot_all_layers, experts);
    if (slots_u64 == 0) return error.ExpertCacheTooSmall;
    const slots: u16 = @intCast(slots_u64);
    const workspace_bytes = std.math.mul(u64, experts, expert_bytes) catch return error.InvalidExpertGeometry;
    const bounce_bytes: u64 = BOUNCE_BYTES;
    const layer_bytes = std.math.mul(u64, slots, expert_bytes) catch return error.InvalidExpertGeometry;
    return .{
        .slots_per_layer = slots,
        .expert_bytes = expert_bytes,
        .cache_bytes = std.math.mul(u64, std.math.mul(u64, layers, slots) catch return error.InvalidExpertGeometry, expert_bytes) catch return error.InvalidExpertGeometry,
        .workspace_bytes = workspace_bytes,
        .bounce_bytes = bounce_bytes,
        .layer_bytes = layer_bytes,
        .prefill_peak_bytes = fillPeakBytes(expert_bytes, experts, slots, bounce_bytes),
    };
}

pub fn cachePlanBytesForGeometry(requested_bytes: u64, geometry: Geometry, expert_bytes: u64) !CachePlan {
    return cachePlanBytes(requested_bytes, moeLayerCount(geometry), geometry.experts, expert_bytes);
}

pub const MTP_UNSUPPORTED: []const u8 = "MTP under expert streaming needs the head loaded: launch with --mtp";

/// A streamed load keeps the MTP head resident only when `--mtp` asks for it: the head's own
/// experts stay in RAM and its verify rows take the streamed decode path.
pub const MtpUnderStreaming = enum { off, keep, drop_settings, drop_default };

/// Takes the load's RESOLVED MTP choice. On from a launch flag keeps the head; on from a
/// per-model `mtp: true` is dropped with a warning (the setting was written for the resident
/// load of the same pack); the engine default quietly resolves off, since nobody asked for the head.
pub fn mtpUnderStreaming(mtp_on: bool, from_settings: bool, from_default: bool) MtpUnderStreaming {
    if (!mtp_on) return .off;
    if (from_default) return .drop_default;
    return if (from_settings) .drop_settings else .keep;
}

pub fn mtpRefusal(expert_streaming: bool, mtp_requested: bool, head_loaded: bool) ?[]const u8 {
    if (!expert_streaming or !mtp_requested or head_loaded) return null;
    return MTP_UNSUPPORTED;
}

/// PURE: does this load stream its routed experts? A checkpoint with no resident
/// arm always does; a capable one only when a cache or an SSD budget was asked for,
/// so a quantized pack with neither loads resident.
pub fn expertStreamingEngaged(capable: bool, required: bool, explicit_cache_bytes: u64, budget_bytes: u64) bool {
    if (!capable) return false;
    return required or explicit_cache_bytes > 0 or budget_bytes > 0;
}

pub const BudgetLedger = struct {
    budget_bytes: u64,
    trunk_bytes: u64,
    mtp_bytes: u64,
    workspace_bytes: u64,
    selected_bytes: u64,
    bounce_bytes: u64,
    cache_bytes: u64,
    slots_per_layer: u16,
};

pub fn budgetOverriddenByExplicitCache(explicit_cache_bytes: u64, budget_bytes: u64) bool {
    return explicit_cache_bytes > 0 and budget_bytes > 0;
}

/// The budget is the TOTAL resident target (GiB, the `--ssd-budget-gb` unit); the expert
/// cache is what remains after trunk, MTP, the all-experts union workspace, the selected
/// slab and the bounce buffers. The PLE table is a disk gather and is never billed.
pub fn budgetLedger(
    budget_bytes: u64,
    trunk_bytes: u64,
    mtp_bytes: u64,
    layers: u16,
    experts: u16,
    top_k: u16,
    expert_bytes: u64,
    bounce_bytes: u64,
) !BudgetLedger {
    if (layers == 0 or experts == 0 or top_k == 0 or expert_bytes == 0) return error.InvalidExpertGeometry;
    const workspace_bytes = @as(u64, experts) *| expert_bytes;
    const selected_bytes = @as(u64, @min(top_k, experts)) *| expert_bytes;
    const fixed = trunk_bytes +| mtp_bytes +| workspace_bytes +| selected_bytes +| bounce_bytes;
    if (fixed >= budget_bytes) return error.SsdBudgetBelowResident;
    const per_slot = @as(u64, layers) *| expert_bytes;
    const slots_u64 = @min((budget_bytes - fixed) / per_slot, experts);
    if (slots_u64 <= 1) return error.SsdBudgetBelowResident;
    const slots: u16 = @intCast(slots_u64);
    return .{
        .budget_bytes = budget_bytes,
        .trunk_bytes = trunk_bytes,
        .mtp_bytes = mtp_bytes,
        .workspace_bytes = workspace_bytes,
        .selected_bytes = selected_bytes,
        .bounce_bytes = bounce_bytes,
        .cache_bytes = @as(u64, slots) *| per_slot,
        .slots_per_layer = slots,
    };
}

pub const FILL_WORKERS: usize = 4;
pub const COALESCE_MAX: u64 = 64 * 1024 * 1024;

pub const BREAKDOWN_EVERY: u64 = 64;

const PENDING_READERS_MAX: usize = 2 * quant.component_count;

pub const RouteDetail = struct {
    build_ns: u64 = 0,
    x_wait_ns: u64 = 0,
    ids_wait_ns: u64 = 0,
    read_ns: u64 = 0,

    pub fn totalNs(self: RouteDetail) u64 {
        return self.build_ns +| self.x_wait_ns +| self.ids_wait_ns +| self.read_ns;
    }
};

pub const RouteClass = struct {
    layers: u32 = 0,
    build_ns: u64 = 0,
    x_wait_ns: u64 = 0,
    ids_wait_ns: u64 = 0,
    read_ns: u64 = 0,

    fn add(self: *RouteClass, detail: RouteDetail) void {
        self.layers += 1;
        self.build_ns +|= detail.build_ns;
        self.x_wait_ns +|= detail.x_wait_ns;
        self.ids_wait_ns +|= detail.ids_wait_ns;
        self.read_ns +|= detail.read_ns;
    }
};

pub const Breakdown = struct {
    forward: u64 = 0,
    rows: usize = 0,
    wall_ns: u64 = 0,
    route_ns: u64 = 0,
    fill_ns: u64 = 0,
    compute_ns: u64 = 0,
    fill_bytes: u64 = 0,
    hits: u64 = 0,
    union_members: u64 = 0,
    route_linear: RouteClass = .{},
    route_full: RouteClass = .{},
    probe: bool = false,

    pub fn otherNs(self: Breakdown) u64 {
        const accounted = self.route_ns +| self.fill_ns +| self.compute_ns;
        return if (self.wall_ns > accounted) self.wall_ns - accounted else 0;
    }
};

pub const Options = struct {
    io_workers: usize = FILL_WORKERS,
    bounce_size: usize = 64 * 1024 * 1024,
    layout: quant.Layout = .bf16_fused,
};

pub const Binding = struct {
    expert: u16,
    slot: ?u16,
    hit: bool,
};

pub const GroupResolution = struct {
    union_ids: []u16,
    remapped: []u16,
    bindings: []Binding,
    hits: usize,
    misses: usize,
    declined: usize,
    workspace: bool,

    pub fn deinit(self: *GroupResolution, allocator: std.mem.Allocator) void {
        allocator.free(self.union_ids);
        allocator.free(self.remapped);
        allocator.free(self.bindings);
        self.* = undefined;
    }
};

pub const MAX_PICK_TOLERANCE: f32 = 0.6;

pub var pick_tolerance: f32 = 0;

pub fn parsePickTolerance(raw: []const u8) error{InvalidPickTolerance}!f32 {
    const value = std.fmt.parseFloat(f32, raw) catch return error.InvalidPickTolerance;
    if (!(value >= 0) or value > MAX_PICK_TOLERANCE) return error.InvalidPickTolerance;
    return value;
}

pub const PICK_STARVE_LIMIT: u8 = 3;

pub fn substituteMisses(ids: []u16, logits: []const f32, cached: []const bool, tolerance: f32, k: usize, taken: []bool, loading: []bool, starved: []u8) u32 {
    if (!(tolerance > 0) or k == 0 or k > 32) return 0;
    const experts = cached.len;
    const floor_gap: f32 = @log(1.0 - @min(tolerance, MAX_PICK_TOLERANCE));
    @memset(loading[0..experts], false);
    var swapped: u32 = 0;
    var row: usize = 0;
    while ((row + 1) * k <= ids.len) : (row += 1) {
        const row_ids = ids[row * k ..][0..k];
        const row_logits = logits[row * experts ..][0..experts];
        var order: [32]u8 = undefined;
        var misses: usize = 0;
        for (row_ids, 0..) |expert, pos| {
            if (cached[expert] or loading[expert]) continue;
            var at = misses;
            while (at > 0 and row_logits[row_ids[order[at - 1]]] < row_logits[expert]) : (at -= 1) order[at] = order[at - 1];
            order[at] = @intCast(pos);
            misses += 1;
        }
        if (misses == 0) continue;
        @memset(taken[0..experts], false);
        for (row_ids) |expert| taken[expert] = true;
        for (order[0..misses]) |pos| {
            const missed = row_ids[pos];
            if (starved[missed] >= PICK_STARVE_LIMIT) {
                starved[missed] = 0;
                loading[missed] = true;
                continue;
            }
            var best: ?usize = null;
            var best_logit: f32 = -std.math.inf(f32);
            for (0..experts) |candidate| {
                if (taken[candidate] or !(cached[candidate] or loading[candidate])) continue;
                const logit = row_logits[candidate];
                if (!(logit - row_logits[missed] >= floor_gap) or !(logit > best_logit)) continue;
                best = candidate;
                best_logit = logit;
            }
            if (best) |pick| {
                taken[pick] = true;
                row_ids[pos] = @intCast(pick);
                starved[missed] +|= 1;
                swapped += 1;
            } else {
                starved[missed] = 0;
                loading[missed] = true;
            }
        }
    }
    return swapped;
}

pub const GroupCache = struct {
    allocator: std.mem.Allocator,
    expert_to_slot: []i32,
    slot_to_expert: []u16,
    ages: []u64,
    ready: []bool,
    swap_starved: []u8,
    tick: u64 = 0,

    fn empty(allocator: std.mem.Allocator) GroupCache {
        return .{
            .allocator = allocator,
            .expert_to_slot = &.{},
            .slot_to_expert = &.{},
            .ages = &.{},
            .ready = &.{},
            .swap_starved = &.{},
        };
    }

    pub fn init(allocator: std.mem.Allocator, capacity: u16, expert_count: u16) !GroupCache {
        if (capacity == 0 or capacity > expert_count) return error.InvalidCacheCapacity;
        const expert_to_slot = try allocator.alloc(i32, expert_count);
        errdefer allocator.free(expert_to_slot);
        const slot_to_expert = try allocator.alloc(u16, capacity);
        errdefer allocator.free(slot_to_expert);
        const ages = try allocator.alloc(u64, capacity);
        errdefer allocator.free(ages);
        const ready = try allocator.alloc(bool, capacity);
        errdefer allocator.free(ready);
        const swap_starved = try allocator.alloc(u8, expert_count);
        @memset(swap_starved, 0);
        @memset(expert_to_slot, -1);
        @memset(slot_to_expert, std.math.maxInt(u16));
        @memset(ages, 0);
        @memset(ready, false);
        return .{ .allocator = allocator, .expert_to_slot = expert_to_slot, .slot_to_expert = slot_to_expert, .ages = ages, .ready = ready, .swap_starved = swap_starved };
    }

    pub fn deinit(self: *GroupCache) void {
        self.allocator.free(self.expert_to_slot);
        self.allocator.free(self.slot_to_expert);
        self.allocator.free(self.ages);
        self.allocator.free(self.ready);
        self.allocator.free(self.swap_starved);
        self.* = undefined;
    }

    pub fn cachedMask(self: *const GroupCache, out: []bool) void {
        for (out, 0..) |*flag, expert| {
            const raw_slot = if (expert < self.expert_to_slot.len) self.expert_to_slot[expert] else -1;
            flag.* = raw_slot >= 0 and self.ready[@intCast(raw_slot)];
        }
    }

    pub fn markReady(self: *GroupCache, slot: u16) void {
        self.ready[slot] = true;
    }

    fn invalidateUnready(self: *GroupCache) void {
        for (self.ready, 0..) |flag, i| {
            if (!flag) self.evict(@intCast(i));
        }
    }

    fn evict(self: *GroupCache, slot: u16) void {
        const old = self.slot_to_expert[slot];
        if (old != std.math.maxInt(u16)) self.expert_to_slot[old] = -1;
        self.slot_to_expert[slot] = std.math.maxInt(u16);
        self.ready[slot] = false;
        self.ages[slot] = 0;
    }

    fn touch(self: *GroupCache, slot: u16) void {
        self.tick +%= 1;
        if (self.tick == 0) self.tick = 1;
        self.ages[slot] = self.tick;
    }

    fn victim(self: *const GroupCache) u16 {
        var best: u16 = 0;
        for (self.slot_to_expert, 0..) |expert, i| {
            if (expert == std.math.maxInt(u16)) return @intCast(i);
            if (self.ages[i] < self.ages[best]) best = @intCast(i);
        }
        return best;
    }

    fn admit(self: *GroupCache, expert: u16) u16 {
        const slot = self.victim();
        const old = self.slot_to_expert[slot];
        if (old != std.math.maxInt(u16)) self.expert_to_slot[old] = -1;
        self.slot_to_expert[slot] = expert;
        self.expert_to_slot[expert] = slot;
        self.ready[slot] = false;
        self.touch(slot);
        return slot;
    }

    pub fn resolve(self: *GroupCache, allocator: std.mem.Allocator, occurrences: []const u16) !GroupResolution {
        const last = try allocator.alloc(usize, self.expert_to_slot.len);
        defer allocator.free(last);
        @memset(last, std.math.maxInt(usize));
        const counts = try allocator.alloc(u32, self.expert_to_slot.len);
        defer allocator.free(counts);
        @memset(counts, 0);
        for (occurrences, 0..) |expert, i| {
            if (expert >= self.expert_to_slot.len) return error.ExpertOutOfRange;
            last[expert] = i;
            counts[expert] += 1;
        }

        var union_list: std.ArrayList(u16) = .empty;
        defer union_list.deinit(allocator);
        for (occurrences, 0..) |expert, i| {
            if (last[expert] == i) try union_list.append(allocator, expert);
        }
        const union_ids = try union_list.toOwnedSlice(allocator);
        errdefer allocator.free(union_ids);
        const bindings = try allocator.alloc(Binding, union_ids.len);
        errdefer allocator.free(bindings);

        var hits: usize = 0;
        var misses: usize = 0;
        for (union_ids, 0..) |expert, i| {
            const raw_slot = self.expert_to_slot[expert];
            if (raw_slot >= 0 and self.ready[@intCast(raw_slot)]) {
                const slot: u16 = @intCast(raw_slot);
                bindings[i] = .{ .expert = expert, .slot = slot, .hit = true };
                self.touch(slot);
                hits += 1;
            } else {
                if (raw_slot >= 0) self.evict(@intCast(raw_slot));
                bindings[i] = .{ .expert = expert, .slot = null, .hit = false };
                misses += 1;
            }
        }

        const workspace = union_ids.len > self.slot_to_expert.len;
        const admission_cap = self.slot_to_expert.len - hits;
        const admit_count = @min(admission_cap, misses);
        const order = try allocator.alloc(usize, misses);
        defer allocator.free(order);
        var pending: usize = 0;
        for (bindings, 0..) |binding, i| {
            if (binding.hit) continue;
            order[pending] = i;
            pending += 1;
        }
        if (workspace) {
            const Sorter = struct {
                counts: []const u32,
                bindings: []const Binding,
                fn less(ctx: @This(), a: usize, b: usize) bool {
                    const ca = ctx.counts[ctx.bindings[a].expert];
                    const cb = ctx.counts[ctx.bindings[b].expert];
                    if (ca != cb) return ca > cb;
                    return a < b;
                }
            };
            std.sort.pdq(usize, order, Sorter{ .counts = counts, .bindings = bindings }, Sorter.less);
        }
        if (workspace) {
            var taken = admit_count;
            while (taken > 0) {
                taken -= 1;
                const i = order[taken];
                bindings[i].slot = self.admit(bindings[i].expert);
            }
        } else {
            for (order[0..admit_count]) |i| bindings[i].slot = self.admit(bindings[i].expert);
        }
        const remapped = try allocator.alloc(u16, occurrences.len);
        errdefer allocator.free(remapped);
        const union_slots = try allocator.alloc(u16, self.expert_to_slot.len);
        defer allocator.free(union_slots);
        @memset(union_slots, std.math.maxInt(u16));
        for (union_ids, 0..) |expert, i| union_slots[expert] = @intCast(i);
        for (occurrences, 0..) |expert, i| remapped[i] = union_slots[expert];

        return .{
            .union_ids = union_ids,
            .remapped = remapped,
            .bindings = bindings,
            .hits = hits,
            .misses = misses,
            .declined = misses - admit_count,
            .workspace = workspace,
        };
    }
};

pub const Geometry = quant.Geometry;

pub const Component = enum(u1) {
    gate_up,
    down,
};

pub const SourceSpan = quant.SourceSpan;

const SourceFile = struct {
    name: []u8,
    fd: std.c.fd_t,
};

pub const readExact = io_mod.readExact;

fn tensorLayout(allocator: std.mem.Allocator, fd: std.c.fd_t, key: []const u8, shape_expected: [3]u64) !TensorLayout {
    const region = io_mod.tensorRegion(allocator, fd, key) catch |err| return switch (err) {
        error.MissingSafetensorsTensor => error.MissingExpertTensor,
        error.SafetensorsTensorOutOfBounds => error.ExpertTensorOutOfBounds,
        else => error.InvalidExpertTensor,
    };
    if (region.dtype != .bf16 or region.rank != 3 or !std.mem.eql(u64, region.shape[0..3], &shape_expected)) return error.InvalidExpertTensor;
    if (region.tensor_bytes != 2 * shape_expected[0] * shape_expected[1] * shape_expected[2]) return error.InvalidExpertTensor;
    return .{ .data_offset = region.data_offset, .tensor_offset = region.tensor_offset, .tensor_bytes = region.tensor_bytes, .experts = @intCast(shape_expected[0]) };
}

pub const SlabSpec = struct {
    rows: u32,
    cols: u32,
    dtype: mlx.mlx_dtype,
    elem_bytes: u8,
    slot_bytes: u64,
    exl3: bool = false,
    packed_n: u32 = 0,
};

pub const ExpertStore = struct {
    allocator: std.mem.Allocator,
    geometry: Geometry,
    files: []SourceFile,
    spans: []SourceSpan,
    quantized: ?quant.QuantStore = null,

    fn sourceIndex(self: *const ExpertStore, layer: u16, expert: u16, component: Component) usize {
        return (@as(usize, layer) * self.geometry.experts + expert) * 2 + @backingInt(component);
    }

    pub fn span(self: *const ExpertStore, layer: u16, expert: u16, component: Component) SourceSpan {
        return self.spans[self.sourceIndex(layer, expert, component)];
    }

    pub fn layout(self: *const ExpertStore) quant.Layout {
        return if (self.quantized) |*q| q.layout else .bf16_fused;
    }

    pub fn componentCount(self: *const ExpertStore) usize {
        return if (self.quantized == null) 2 else quant.component_count;
    }

    pub fn spanAt(self: *const ExpertStore, layer: u16, expert: u16, ci: usize) SourceSpan {
        if (self.quantized) |*q| return q.span(layer, expert, @fromBackingInt(@intCast(ci)));
        return self.spans[(@as(usize, layer) * self.geometry.experts + expert) * 2 + ci];
    }

    pub fn hasExpertLayer(self: *const ExpertStore, layer: u16) bool {
        if (self.quantized) |*q| return q.hasExpertLayer(layer);
        return layer < self.geometry.layers and layer >= self.geometry.first_moe_layer;
    }

    pub fn componentPresent(self: *const ExpertStore, ci: usize) bool {
        if (self.quantized) |*q| return q.componentPresent(@fromBackingInt(@intCast(ci)));
        return ci < 2;
    }

    pub fn slabSpec(self: *const ExpertStore, ci: usize) SlabSpec {
        if (self.quantized) |*q| {
            const c: quant.Component = @fromBackingInt(@intCast(ci));
            const dtype: mlx.mlx_dtype = switch (q.dtypeOf(c)) {
                .bf16 => .bfloat16,
                .f16 => .float16,
                .u16 => .uint16,
                .u8 => .uint8,
                .u32 => .uint32,
                .other => .bfloat16,
            };
            const elem: u8 = switch (q.dtypeOf(c)) {
                .u8 => 1,
                .u32 => 4,
                .bf16, .f16, .u16 => 2,
                .other => 0,
            };
            return .{ .rows = q.rowsOf(c), .cols = q.colsOf(c), .dtype = dtype, .elem_bytes = elem, .slot_bytes = q.slotBytes(c), .exl3 = q.layout == .exl3_k4, .packed_n = q.packed_n[ci] };
        }
        const rows: u32 = if (ci == 0) 2 * self.geometry.intermediate else self.geometry.hidden;
        const cols: u32 = if (ci == 0) self.geometry.hidden else self.geometry.intermediate;
        return .{ .rows = rows, .cols = cols, .dtype = .bfloat16, .elem_bytes = 2, .slot_bytes = @as(u64, rows) * cols * 2 };
    }

    pub fn slabSpecAt(self: *const ExpertStore, layer: u16, ci: usize) SlabSpec {
        var spec = self.slabSpec(ci);
        if (!spec.exl3) return spec;
        spec.slot_bytes = self.spanAt(layer, 0, ci).len;
        if (spec.packed_n != 0) {
            const tiles = @as(u64, spec.rows) * (spec.cols / spec.packed_n);
            spec.packed_n = @intCast(spec.slot_bytes / (tiles * spec.elem_bytes));
            spec.cols = @intCast(spec.slot_bytes / (spec.rows * spec.elem_bytes));
        }
        return spec;
    }

    pub fn perExpertBytes(self: *const ExpertStore) u64 {
        if (self.quantized) |*q| return q.expertBytes();
        var total: u64 = 0;
        for (0..2) |ci| total += self.slabSpec(ci).slot_bytes;
        return total;
    }

    pub fn fileCount(self: *const ExpertStore) usize {
        if (self.quantized) |*q| return q.files.len;
        return self.files.len;
    }

    pub fn fdAt(self: *const ExpertStore, i: usize) std.c.fd_t {
        if (self.quantized) |*q| return q.files[i].fd;
        return self.files[i].fd;
    }

    pub fn openLayout(allocator: std.mem.Allocator, model_dir: []const u8, geometry: Geometry, chosen: quant.Layout) !ExpertStore {
        if (chosen == .bf16_fused) return open(allocator, model_dir, geometry);
        const q = try quant.QuantStore.openForLayout(allocator, model_dir, geometry, chosen);
        return .{ .allocator = allocator, .geometry = geometry, .files = &.{}, .spans = &.{}, .quantized = q };
    }

    pub fn open(allocator: std.mem.Allocator, model_dir: []const u8, geometry: Geometry) !ExpertStore {
        if (geometry.layers == 0 or geometry.experts == 0 or geometry.hidden == 0 or geometry.intermediate == 0) return error.InvalidExpertGeometry;
        const io = std.Io.Threaded.global_single_threaded.io();
        var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
        defer dir.close(io);
        const index_raw = try dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(32 * 1024 * 1024));
        defer allocator.free(index_raw);
        const index_parsed = std.json.parseFromSlice(std.json.Value, allocator, index_raw, .{}) catch return error.InvalidSafetensorsIndex;
        defer index_parsed.deinit();
        if (index_parsed.value != .object) return error.InvalidSafetensorsIndex;
        const weight_map_value = index_parsed.value.object.get("weight_map") orelse return error.InvalidSafetensorsIndex;
        if (weight_map_value != .object) return error.InvalidSafetensorsIndex;
        const weight_map = weight_map_value.object;

        var files_list: std.ArrayList(SourceFile) = .empty;
        errdefer {
            for (files_list.items) |file| {
                _ = std.c.close(file.fd);
                allocator.free(file.name);
            }
            files_list.deinit(allocator);
        }
        const span_count = std.math.mul(usize, std.math.mul(usize, geometry.layers, geometry.experts) catch return error.InvalidExpertGeometry, 2) catch return error.InvalidExpertGeometry;
        const spans = try allocator.alloc(SourceSpan, span_count);
        errdefer allocator.free(spans);

        const openSource = struct {
            fn call(a: std.mem.Allocator, list: *std.ArrayList(SourceFile), dir_path: []const u8, name: []const u8) !u16 {
                for (list.items, 0..) |file, i| {
                    if (std.mem.eql(u8, file.name, name)) return @intCast(i);
                }
                if (list.items.len >= std.math.maxInt(u16)) return error.TooManyExpertShards;
                const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir_path, name }, 0);
                defer a.free(path);
                const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
                if (fd < 0) return error.MissingExpertShard;
                errdefer _ = std.c.close(fd);
                _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1));
                const owned_name = try a.dupe(u8, name);
                errdefer a.free(owned_name);
                try list.append(a, .{ .name = owned_name, .fd = fd });
                return @intCast(list.items.len - 1);
            }
        }.call;

        var key_buf: [192]u8 = undefined;
        for (0..geometry.layers) |layer_usize| {
            const layer: u16 = @intCast(layer_usize);
            const gate_key = std.fmt.bufPrint(&key_buf, "model.language_model.layers.{d}.mlp.experts.gate_up_proj", .{layer}) catch return error.InvalidExpertGeometry;
            const gate_map = weight_map.get(gate_key) orelse return error.MissingExpertTensor;
            if (gate_map != .string) return error.InvalidSafetensorsIndex;
            const gate_file = try openSource(allocator, &files_list, model_dir, gate_map.string);
            const gate_layout = try tensorLayout(allocator, files_list.items[gate_file].fd, gate_key, .{ geometry.experts, 2 * @as(u64, geometry.intermediate), geometry.hidden });

            const down_key = std.fmt.bufPrint(&key_buf, "model.language_model.layers.{d}.mlp.experts.down_proj", .{layer}) catch return error.InvalidExpertGeometry;
            const down_map = weight_map.get(down_key) orelse return error.MissingExpertTensor;
            if (down_map != .string) return error.InvalidSafetensorsIndex;
            const down_file = try openSource(allocator, &files_list, model_dir, down_map.string);
            const down_layout = try tensorLayout(allocator, files_list.items[down_file].fd, down_key, .{ geometry.experts, geometry.hidden, geometry.intermediate });

            for (0..geometry.experts) |expert_usize| {
                const expert: u16 = @intCast(expert_usize);
                const gate_span = try gate_layout.expertSpan(expert);
                const down_span = try down_layout.expertSpan(expert);
                const base = (@as(usize, layer) * geometry.experts + expert) * 2;
                spans[base] = .{ .file = gate_file, .offset = gate_span.offset, .len = gate_span.len };
                spans[base + 1] = .{ .file = down_file, .offset = down_span.offset, .len = down_span.len };
            }
        }

        return .{ .allocator = allocator, .geometry = geometry, .files = try files_list.toOwnedSlice(allocator), .spans = spans };
    }

    pub fn deinit(self: *ExpertStore) void {
        if (self.quantized) |*q| q.deinit();
        for (self.files) |file| {
            _ = std.c.close(file.fd);
            self.allocator.free(file.name);
        }
        if (self.files.len != 0) self.allocator.free(self.files);
        if (self.spans.len != 0) self.allocator.free(self.spans);
        self.* = undefined;
    }

    pub fn readSpan(self: *const ExpertStore, span_value: SourceSpan, dst: []u8) !void {
        if (span_value.file >= self.files.len or span_value.len != dst.len) return error.InvalidExpertRead;
        try readExact(self.files[span_value.file].fd, dst, span_value.offset);
    }

    pub fn readExpert(self: *const ExpertStore, layer: u16, expert: u16, dst: []u8) !void {
        if (layer >= self.geometry.layers or expert >= self.geometry.experts) return error.ExpertOutOfRange;
        const gate = self.span(layer, expert, .gate_up);
        const down = self.span(layer, expert, .down);
        if (dst.len != gate.len + down.len) return error.InvalidExpertRead;
        try self.readSpan(gate, dst[0..@intCast(gate.len)]);
        try self.readSpan(down, dst[@intCast(gate.len)..]);
    }
};

const TableRegion = struct {
    offset: u64,
    rows: u64,
    dim: u32,
};

fn tensor2dRegion(allocator: std.mem.Allocator, fd: std.c.fd_t, key: []const u8) !TableRegion {
    const region = io_mod.tensorRegion(allocator, fd, key) catch |err| return switch (err) {
        error.MissingSafetensorsTensor => error.MissingNgramTensor,
        error.SafetensorsTensorOutOfBounds => error.NgramTensorOutOfBounds,
        else => error.InvalidNgramTensor,
    };
    if (region.dtype != .bf16 or region.rank != 2 or region.shape[1] > std.math.maxInt(u32)) return error.InvalidNgramTensor;
    if (region.tensor_bytes != 2 * region.shape[0] * region.shape[1]) return error.InvalidNgramTensor;
    return .{ .offset = region.data_offset + region.tensor_offset, .rows = region.shape[0], .dim = @intCast(region.shape[1]) };
}

const TableShard = struct {
    file: u16,
    offset: u64,
};

const NgramKey = struct {
    layer: u16,
    shard: u16,
};

fn ngramKey(key: []const u8) !?NgramKey {
    const prefix = "model.language_model.layers.";
    const middle = ".ple.ple_embedding.ngram_embedding.shard_";
    const suffix = ".weight";
    if (!std.mem.startsWith(u8, key, prefix) or !std.mem.endsWith(u8, key, suffix)) return null;
    const body = key[prefix.len .. key.len - suffix.len];
    const split = std.mem.indexOf(u8, body, middle) orelse return null;
    const layer = std.fmt.parseInt(u16, body[0..split], 10) catch return error.InvalidNgramTensorKey;
    const shard = std.fmt.parseInt(u16, body[split + middle.len ..], 10) catch return error.InvalidNgramTensorKey;
    return .{ .layer = layer, .shard = shard };
}

pub const Bf16NgramStore = struct {
    allocator: std.mem.Allocator,
    files: []SourceFile,
    shards: []TableShard,
    rows_per_shard: u64,
    rows: u64,
    dim: u32,

    pub fn open(allocator: std.mem.Allocator, model_dir: []const u8) !Bf16NgramStore {
        const io = std.Io.Threaded.global_single_threaded.io();
        var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
        defer dir.close(io);
        const index_raw = try dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(32 * 1024 * 1024));
        defer allocator.free(index_raw);
        const index_parsed = std.json.parseFromSlice(std.json.Value, allocator, index_raw, .{}) catch return error.InvalidSafetensorsIndex;
        defer index_parsed.deinit();
        if (index_parsed.value != .object) return error.InvalidSafetensorsIndex;
        const weight_map_value = index_parsed.value.object.get("weight_map") orelse return error.InvalidSafetensorsIndex;
        if (weight_map_value != .object) return error.InvalidSafetensorsIndex;
        const weight_map = weight_map_value.object;

        var table_layer: ?u16 = null;
        var table_shards: usize = 0;
        var max_shard: u16 = 0;
        var weight_it = weight_map.iterator();
        while (weight_it.next()) |entry| {
            const parsed_key = try ngramKey(entry.key_ptr.*) orelse continue;
            if (table_layer) |layer| {
                if (layer != parsed_key.layer) return error.MultipleNgramLayers;
            } else {
                table_layer = parsed_key.layer;
            }
            max_shard = @max(max_shard, parsed_key.shard);
            table_shards += 1;
        }
        const layer = table_layer orelse return error.MissingNgramTensor;
        const shard_count_usize = @as(usize, max_shard) + 1;
        if (table_shards != shard_count_usize or shard_count_usize > std.math.maxInt(u16)) return error.InvalidNgramShardCount;
        const shard_count: u16 = @intCast(shard_count_usize);

        var files_list: std.ArrayList(SourceFile) = .empty;
        errdefer {
            for (files_list.items) |file| {
                _ = std.c.close(file.fd);
                allocator.free(file.name);
            }
            files_list.deinit(allocator);
        }
        const shards = try allocator.alloc(TableShard, shard_count);
        errdefer allocator.free(shards);
        var rows_per_shard: u64 = 0;
        var dim: u32 = 0;
        var key_buf: [192]u8 = undefined;
        for (0..shard_count) |shard_usize| {
            const key = std.fmt.bufPrint(&key_buf, "model.language_model.layers.{d}.ple.ple_embedding.ngram_embedding.shard_{d}.weight", .{ layer, shard_usize }) catch return error.NameTooLong;
            const mapped = weight_map.get(key) orelse return error.MissingNgramTensor;
            if (mapped != .string) return error.InvalidSafetensorsIndex;
            var file_index: ?u16 = null;
            for (files_list.items, 0..) |file, i| {
                if (std.mem.eql(u8, file.name, mapped.string)) {
                    file_index = @intCast(i);
                    break;
                }
            }
            if (file_index == null) {
                const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, mapped.string }, 0);
                defer allocator.free(path);
                const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
                if (fd < 0) return error.MissingNgramShard;
                errdefer _ = std.c.close(fd);
                _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1));
                const name = try allocator.dupe(u8, mapped.string);
                errdefer allocator.free(name);
                try files_list.append(allocator, .{ .name = name, .fd = fd });
                file_index = @intCast(files_list.items.len - 1);
            }
            const region = try tensor2dRegion(allocator, files_list.items[file_index.?].fd, key);
            if (shard_usize == 0) {
                rows_per_shard = region.rows;
                dim = region.dim;
            } else if (region.rows != rows_per_shard or region.dim != dim) {
                return error.NgramShardGeometryMismatch;
            }
            shards[shard_usize] = .{ .file = file_index.?, .offset = region.offset };
        }
        const total_rows = std.math.mul(u64, rows_per_shard, shard_count) catch return error.InvalidNgramTensor;
        return .{ .allocator = allocator, .files = try files_list.toOwnedSlice(allocator), .shards = shards, .rows_per_shard = rows_per_shard, .rows = total_rows, .dim = dim };
    }

    pub fn deinit(self: *Bf16NgramStore) void {
        for (self.files) |file| {
            _ = std.c.close(file.fd);
            self.allocator.free(file.name);
        }
        self.allocator.free(self.files);
        self.allocator.free(self.shards);
        self.* = undefined;
    }

    pub fn gather(self: *const Bf16NgramStore, row_ids: []const i64, out: []f32) !void {
        if (out.len != row_ids.len * self.dim) return error.InvalidNgramOutput;
        const row_bytes: usize = @as(usize, self.dim) * 2;
        const raw = try self.allocator.alloc(u8, row_bytes);
        defer self.allocator.free(raw);
        for (row_ids, 0..) |row_signed, i| {
            try self.readRowBytes(row_signed, raw);
            try self.decodeRowBytes(raw, out[i * self.dim ..][0..self.dim]);
        }
    }

    pub fn readRowBytes(self: *const Bf16NgramStore, row_signed: i64, raw: []u8) !void {
        const row_bytes: usize = @as(usize, self.dim) * 2;
        if (raw.len != row_bytes or row_signed < 0) return error.NgramRowOutOfRange;
        const row: u64 = @intCast(row_signed);
        if (row >= self.rows) return error.NgramRowOutOfRange;
        const shard_index: usize = @intCast(row / self.rows_per_shard);
        const local = row % self.rows_per_shard;
        const shard = self.shards[shard_index];
        const offset = shard.offset + local * row_bytes;
        try readExact(self.files[shard.file].fd, raw, offset);
    }

    pub fn decodeRowBytes(self: *const Bf16NgramStore, raw: []const u8, out: []f32) !void {
        if (raw.len != @as(usize, self.dim) * 2 or out.len != self.dim) return error.InvalidNgramOutput;
        for (0..self.dim) |j| {
            const bits = std.mem.readInt(u16, raw[j * 2 ..][0..2], .little);
            out[j] = @bitCast(@as(u32, bits) << 16);
        }
    }
};

pub var slab_release_timeouts: std.atomic.Value(u64) = .init(0);

const SlabOperand = struct {
    present: bool = false,
    slab: ?*io_mod.PageSlab = null,
    payload: ?*io_mod.ImportPayload = null,
    array: mlx.mlx_array = .{ .ctx = null },
    stride: usize,
    count: u32,

    fn absent() SlabOperand {
        return .{ .stride = 0, .count = 0 };
    }

    fn create(allocator: std.mem.Allocator, count: u32, d0: u32, d1: u32) !SlabOperand {
        return createTyped(allocator, count, d0, d1, .bfloat16, 2);
    }

    fn createTyped(allocator: std.mem.Allocator, count: u32, d0: u32, d1: u32, dtype: mlx.mlx_dtype, elem_bytes: u8) !SlabOperand {
        return createShaped(allocator, count, &.{ @intCast(d0), @intCast(d1) }, dtype, elem_bytes);
    }

    fn createShaped(allocator: std.mem.Allocator, count: u32, inner: []const c_int, dtype: mlx.mlx_dtype, elem_bytes: u8) !SlabOperand {
        if (count == 0 or inner.len == 0 or inner.len > 3 or elem_bytes == 0) return error.InvalidExpertGeometry;
        var stride: usize = elem_bytes;
        var shape: [4]c_int = undefined;
        shape[0] = @intCast(count);
        for (inner, 0..) |d, j| {
            if (d <= 0) return error.InvalidExpertGeometry;
            stride = try std.math.mul(usize, stride, @intCast(d));
            shape[j + 1] = d;
        }
        const slab = try io_mod.PageSlab.create(allocator, stride * count);
        errdefer slab.destroy();
        const payload = try allocator.create(io_mod.ImportPayload);
        errdefer allocator.destroy(payload);
        payload.* = .{};
        const operand = try io_mod.importSlab(slab, shape[0 .. inner.len + 1], dtype, payload, .{});
        if (!operand.aliased) {
            _ = mlx.mlx_array_free(operand.array);
            return error.ExpertSlabImportCopied;
        }
        return .{ .present = true, .slab = slab, .payload = payload, .array = operand.array, .stride = stride, .count = count };
    }

    fn destroy(self: *SlabOperand, allocator: std.mem.Allocator, s: mlx.mlx_stream) void {
        if (!self.present) return;
        const slab = self.slab orelse return;
        const payload = self.payload orelse return;
        _ = mlx.mlx_array_free(self.array);
        var attempt: usize = 0;
        while (payload.released.load(.acquire) == 0 and attempt < 32) : (attempt += 1) {
            _ = mlx.mlx_synchronize(s);
            _ = mlx.mlx_clear_cache();
            std.Thread.yield() catch {};
        }
        if (payload.released.load(.acquire) == 0) {
            const timeouts = slab_release_timeouts.fetchAdd(1, .monotonic) + 1;
            log.warn("[expert-stream] slab release timed out: {d} bytes stay mapped for MLX and are never freed (timeouts so far {d})\n", .{ slab.bytes.len, timeouts });
            self.* = undefined;
            return;
        }
        slab.destroy();
        allocator.destroy(payload);
        self.* = undefined;
    }

    fn slotBytes(self: *const SlabOperand, slot: usize) []u8 {
        if (!self.present) return &.{};
        const slab = self.slab orelse return &.{};
        return slab.bytes[slot * self.stride ..][0..self.stride];
    }

    fn viewAs(self: *const SlabOperand, template: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, self.array, &.{-1}, 1, s));
        var prefix = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(prefix);
        const elements: c_int = @intCast(self.count * self.stride / mlx.mlx_array_itemsize(self.array));
        try mlx.check(mlx.mlx_slice(&prefix, flat, &.{0}, 1, &.{elements}, 1, &.{1}, 1, s));
        var shape: [4]c_int = undefined;
        const dims = mlx.getShape(template);
        @memcpy(shape[0..dims.len], dims);
        shape[0] = @intCast(self.count);
        var view = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_reshape(&view, prefix, &shape, dims.len, s));
        return view;
    }

    fn borrow(self: *const SlabOperand) !mlx.mlx_array {
        if (!self.present) return error.MissingExpertSlab;
        var handle = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(handle);
        try mlx.check(mlx.mlx_array_set(&handle, self.array));
        return handle;
    }
};

pub const Clock = struct {
    io: std.Io,
    start: std.Io.Timestamp,
    mark_ns: u64 = 0,

    pub fn init() Clock {
        const io = std.Io.Threaded.global_single_threaded.io();
        return .{ .io = io, .start = std.Io.Timestamp.now(io, .boot) };
    }

    pub fn lap(self: *Clock) u64 {
        const cumulative: u64 = @intCast(self.start.untilNow(self.io, .boot).nanoseconds);
        const delta = cumulative - self.mark_ns;
        self.mark_ns = cumulative;
        return delta;
    }
};

const HeldLease = struct {
    slab: *io_mod.PageSlab,
    lease: io_mod.Lease,
};

const CacheSnapshot = struct {
    expert_to_slot: []i32,
    slot_to_expert: []u16,
    ages: []u64,
    ready: []bool,
    tick: u64,

    fn restore(self: *const CacheSnapshot, cache: *GroupCache) void {
        @memcpy(cache.expert_to_slot, self.expert_to_slot);
        @memcpy(cache.slot_to_expert, self.slot_to_expert);
        @memcpy(cache.ages, self.ages);
        @memcpy(cache.ready, self.ready);
        cache.tick = self.tick;
    }

    fn deinit(self: *CacheSnapshot, allocator: std.mem.Allocator) void {
        allocator.free(self.expert_to_slot);
        allocator.free(self.slot_to_expert);
        allocator.free(self.ages);
        allocator.free(self.ready);
        self.* = undefined;
    }
};

fn abortFill(operand: *SlabOperand) void {
    if (!operand.present) return;
    const slab = operand.slab orelse return;
    _ = slab.publish() catch {};
    slab.retire() catch {};
}

const LayerStats = struct {
    groups: u64 = 0,
    union_members: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    fill_bytes: u64 = 0,
    host_sync_ns: u64 = 0,
    fill_ns: u64 = 0,
    compute_ns: u64 = 0,
};

const LayerState = struct {
    cache: GroupCache,
    active: bool = false,
    slabs: []SlabOperand = &.{},
    stats: LayerStats = .{},
    spec_host: []i32 = &.{},
    spec_slots: mlx.mlx_array = .{ .ctx = null },
    spec_raw: mlx.mlx_array = .{ .ctx = null },

    /// Brings the host map and its GPU copy up to the cache. The pair only ever
    /// changes together: a failure leaves no GPU map and a host map that forces
    /// the next call to rebuild.
    fn refreshSpec(self: *LayerState, allocator: std.mem.Allocator) !void {
        const experts = self.cache.expert_to_slot.len;
        if (self.spec_host.len != experts) {
            if (self.spec_host.len != 0) allocator.free(self.spec_host);
            self.spec_host = &.{};
            self.spec_host = try allocator.alloc(i32, experts);
            @memset(self.spec_host, -2);
        }
        if (!refreshSpecMap(self.cache.expert_to_slot, self.cache.ready, self.spec_host) and self.spec_slots.ctx != null) return;
        errdefer {
            if (self.spec_slots.ctx != null) _ = mlx.mlx_array_free(self.spec_slots);
            self.spec_slots = .{ .ctx = null };
            if (self.spec_raw.ctx != null) _ = mlx.mlx_array_free(self.spec_raw);
            self.spec_raw = .{ .ctx = null };
            @memset(self.spec_host, -2);
        }
        const gpu = try allocator.alloc(i32, experts);
        defer allocator.free(gpu);
        for (gpu, self.spec_host) |*slot, host| slot.* = @max(host, 0);
        if (@import("builtin").is_test and spec_alloc_fail_for_test) {
            spec_alloc_fail_for_test = false;
            return error.ExpertSpecMapAlloc;
        }
        const fresh = mlx.mlx_array_new_data(gpu.ptr, &[_]c_int{@intCast(experts)}, 1, .int32);
        if (fresh.ctx == null) return error.ExpertSpecMapAlloc;
        errdefer _ = mlx.mlx_array_free(fresh);
        const raw = mlx.mlx_array_new_data(self.spec_host.ptr, &[_]c_int{@intCast(experts)}, 1, .int32);
        if (raw.ctx == null) return error.ExpertSpecMapAlloc;
        if (self.spec_slots.ctx != null) _ = mlx.mlx_array_free(self.spec_slots);
        self.spec_slots = fresh;
        if (self.spec_raw.ctx != null) _ = mlx.mlx_array_free(self.spec_raw);
        self.spec_raw = raw;
    }
};

var spec_alloc_fail_for_test = false;

/// Writes the slot each expert reads from as the cache stands (-1 = not a ready
/// hit) into `out`; returns whether any entry changed.
pub fn refreshSpecMap(expert_to_slot: []const i32, ready: []const bool, out: []i32) bool {
    var changed = false;
    for (out, expert_to_slot) |*cached, raw| {
        const now: i32 = if (raw >= 0 and ready[@intCast(raw)]) raw else -1;
        if (cached.* != now) {
            cached.* = now;
            changed = true;
        }
    }
    return changed;
}

/// A speculative expert result is exact only when the host resolution read
/// every routed id from the slot the GPU gathered for it.
pub fn specMatches(map: []const i32, ids: []const u16, remapped: []const u16, workspace: bool) bool {
    if (workspace or ids.len != remapped.len) return false;
    for (ids, remapped) |expert, slot| {
        if (expert >= map.len) return false;
        const predicted = map[expert];
        if (predicted < 0 or predicted != slot) return false;
    }
    return true;
}

fn createSlabSet(allocator: std.mem.Allocator, store: *const ExpertStore, count: u32, s: mlx.mlx_stream) ![]SlabOperand {
    return createSlabSetAt(allocator, store, null, count, s);
}

fn createSlabSetAt(allocator: std.mem.Allocator, store: *const ExpertStore, layer: ?u16, count: u32, s: mlx.mlx_stream) ![]SlabOperand {
    const n = store.componentCount();
    const slabs = try allocator.alloc(SlabOperand, n);
    errdefer allocator.free(slabs);
    var made: usize = 0;
    errdefer for (slabs[0..made]) |*operand| operand.destroy(allocator, s);
    while (made < n) : (made += 1) {
        const spec = if (layer) |li| store.slabSpecAt(li, made) else store.slabSpec(made);
        slabs[made] = if (!store.componentPresent(made) or spec.slot_bytes == 0)
            SlabOperand.absent()
        else if (spec.exl3 and spec.packed_n != 0)
            try SlabOperand.createShaped(allocator, count, &.{ @intCast(spec.rows), @intCast(spec.cols / spec.packed_n), @intCast(spec.packed_n) }, spec.dtype, spec.elem_bytes)
        else if (spec.exl3)
            try SlabOperand.createShaped(allocator, count, &.{@intCast(spec.cols)}, spec.dtype, spec.elem_bytes)
        else
            try SlabOperand.createTyped(allocator, count, spec.rows, spec.cols, spec.dtype, spec.elem_bytes);
    }
    return slabs;
}

fn presentSlabCount(slabs: []const SlabOperand) u64 {
    var count: u64 = 0;
    for (slabs) |operand| {
        if (operand.present) count += 1;
    }
    return count;
}

pub const Prepared = struct {
    allocator: std.mem.Allocator,
    gate: mlx.mlx_array = .{ .ctx = null },
    up: mlx.mlx_array = .{ .ctx = null },
    down: mlx.mlx_array = .{ .ctx = null },
    remapped: []u16,
    raw_gate_up: mlx.mlx_array = .{ .ctx = null },
    raw_down: mlx.mlx_array = .{ .ctx = null },
    quant_raw: [quant.component_count]mlx.mlx_array = @splat(.{ .ctx = null }),
    quantized: bool = false,
    owns_quant_views: bool = false,
    workspace: bool = false,
    held: [quant.component_count]?HeldLease = @splat(null),
    defer_to: ?*Engine = null,

    pub fn quantOperand(self: *const Prepared, c: quant.Component) mlx.mlx_array {
        return self.quant_raw[@backingInt(c)];
    }

    pub fn deinit(self: *Prepared) void {
        if (self.gate.ctx != null) _ = mlx.mlx_array_free(self.gate);
        if (self.up.ctx != null) _ = mlx.mlx_array_free(self.up);
        if (self.down.ctx != null) _ = mlx.mlx_array_free(self.down);
        if (self.raw_gate_up.ctx != null) _ = mlx.mlx_array_free(self.raw_gate_up);
        if (self.raw_down.ctx != null) _ = mlx.mlx_array_free(self.raw_down);
        if (self.owns_quant_views) for (self.quant_raw) |arr| {
            if (arr.ctx != null) _ = mlx.mlx_array_free(arr);
        };
        for (self.held) |entry| {
            const held = entry orelse continue;
            if (self.defer_to) |engine| {
                if (engine.deferReader(held)) continue;
            }
            held.slab.release(held.lease) catch {};
            held.slab.retire() catch {};
        }
        self.allocator.free(self.remapped);
        self.* = undefined;
    }
};

const FusedViews = struct {
    gate: mlx.mlx_array,
    up: mlx.mlx_array,
    down: mlx.mlx_array,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    geometry: Geometry,
    plan: CachePlan,
    store: ExpertStore,
    layers: []LayerState,
    s: mlx.mlx_stream,
    forward_count: u64 = 0,
    io_pool: ?*io_mod.FillPool = null,
    fds: []std.c.fd_t = &.{},
    union_slabs: []SlabOperand = &.{},
    slab_imports: u64 = 0,
    fallback_imports: u64 = 0,
    fill_experts_total: u64 = 0,
    fill_bytes_total: u64 = 0,
    fill_ns_total: u64 = 0,
    forward_clock: Clock,
    last: Breakdown = .{},
    route_linear: RouteClass = .{},
    route_full: RouteClass = .{},
    pending: [PENDING_READERS_MAX]?HeldLease = @splat(null),

    pub fn pendingReaders(self: *const Engine) usize {
        var n: usize = 0;
        for (self.pending) |entry| {
            if (entry != null) n += 1;
        }
        return n;
    }

    pub fn drainPendingReaders(self: *Engine) void {
        for (&self.pending) |*entry| {
            const held = entry.* orelse continue;
            held.slab.release(held.lease) catch {};
            held.slab.retire() catch {};
            entry.* = null;
        }
    }

    fn deferReader(self: *Engine, held: HeldLease) bool {
        for (&self.pending) |*entry| {
            if (entry.* != null) continue;
            entry.* = held;
            return true;
        }
        return false;
    }

    pub fn initWithOptions(allocator: std.mem.Allocator, model_dir: []const u8, geometry: Geometry, requested_bytes: u64, s: mlx.mlx_stream, opts: Options) !Engine {
        var store = try ExpertStore.openLayout(allocator, model_dir, geometry, opts.layout);
        errdefer store.deinit();
        const plan = try cachePlanBytesForGeometry(requested_bytes, geometry, store.perExpertBytes());
        var engine = Engine{
            .allocator = allocator,
            .geometry = geometry,
            .plan = plan,
            .store = store,
            .layers = &.{},
            .s = s,
            .forward_clock = Clock.init(),
        };
        errdefer engine.releaseSlabs();
        const fds = try allocator.alloc(std.c.fd_t, engine.store.fileCount());
        for (fds, 0..) |*fd, i| fd.* = engine.store.fdAt(i);
        engine.fds = fds;
        engine.io_pool = try io_mod.FillPool.create(allocator, .{
            .workers = opts.io_workers,
            .bounce_cap = opts.bounce_size,
            .coalesce_max = COALESCE_MAX,
        });
        const before_fallback = io_mod.fallback_imports.load(.monotonic);
        const layers = try allocator.alloc(LayerState, geometry.layers);
        for (layers) |*layer| layer.* = .{ .cache = GroupCache.empty(allocator) };
        engine.layers = layers;
        var initialized: usize = 0;
        errdefer {
            // The layers' slabs go before the layers: the outer `releaseSlabs` finds none left.
            engine.releaseSlabs();
            for (layers[0..initialized]) |*layer| if (layer.active) layer.cache.deinit();
            engine.layers = &.{};
            allocator.free(layers);
        }
        for (layers, 0..) |*layer, layer_index| {
            initialized += 1;
            if (!engine.store.hasExpertLayer(@intCast(layer_index))) continue;
            layer.* = .{ .cache = try GroupCache.init(allocator, plan.slots_per_layer, geometry.experts), .active = true };
            layer.slabs = try createSlabSetAt(allocator, &engine.store, @intCast(layer_index), plan.slots_per_layer, s);
            engine.slab_imports += presentSlabCount(layer.slabs);
        }
        engine.union_slabs = try createSlabSet(allocator, &engine.store, geometry.experts, s);
        engine.slab_imports += presentSlabCount(engine.union_slabs);
        engine.fallback_imports = io_mod.fallback_imports.load(.monotonic) - before_fallback;
        log.info("[expert-stream] cache {d:.3} GB, {d} slots/layer, workspace {d:.3} GB, bounce {d:.3} GB, fallback_imports={d}\n", .{
            @as(f64, @floatFromInt(plan.cache_bytes)) / 1e9,
            plan.slots_per_layer,
            @as(f64, @floatFromInt(plan.workspace_bytes)) / 1e9,
            @as(f64, @floatFromInt(plan.bounce_bytes)) / 1e9,
            engine.fallback_imports,
        });
        return engine;
    }

    fn releaseSlabs(self: *Engine) void {
        for (self.layers) |*layer| {
            if (layer.spec_slots.ctx != null) _ = mlx.mlx_array_free(layer.spec_slots);
            layer.spec_slots = .{ .ctx = null };
            if (layer.spec_raw.ctx != null) _ = mlx.mlx_array_free(layer.spec_raw);
            layer.spec_raw = .{ .ctx = null };
            if (layer.spec_host.len != 0) self.allocator.free(layer.spec_host);
            layer.spec_host = &.{};
            for (layer.slabs) |*operand| operand.destroy(self.allocator, self.s);
            if (layer.slabs.len != 0) self.allocator.free(layer.slabs);
            layer.slabs = &.{};
        }
        for (self.union_slabs) |*operand| operand.destroy(self.allocator, self.s);
        if (self.union_slabs.len != 0) self.allocator.free(self.union_slabs);
        self.union_slabs = &.{};
        if (self.io_pool) |pool| pool.destroy();
        self.io_pool = null;
        if (self.fds.len != 0) self.allocator.free(self.fds);
        self.fds = &.{};
    }

    pub fn deinit(self: *Engine) void {
        self.drainPendingReaders();
        self.releaseSlabs();
        for (self.layers) |*layer| if (layer.active) layer.cache.deinit();
        self.allocator.free(self.layers);
        self.store.deinit();
        self.* = undefined;
    }

    pub fn cacheSlotBytes(self: *const Engine, layer: u16, slot: u16, component: Component) []u8 {
        return self.cacheSlotBytesAt(layer, slot, @backingInt(component));
    }

    pub fn cacheSlotBytesAt(self: *const Engine, layer: u16, slot: u16, ci: usize) []u8 {
        if (layer >= self.layers.len or !self.layers[layer].active or ci >= self.layers[layer].slabs.len)
            return &.{};
        return self.layers[layer].slabs[ci].slotBytes(slot);
    }

    pub fn unionSlotBytes(self: *const Engine, position: u16, component: Component) []u8 {
        return self.unionSlotBytesAt(position, @backingInt(component));
    }

    pub fn unionSlotBytesAt(self: *const Engine, position: u16, ci: usize) []u8 {
        return self.union_slabs[ci].slotBytes(position);
    }

    pub fn slotReady(self: *const Engine, layer: u16, slot: u16) bool {
        return layer < self.layers.len and self.layers[layer].active and self.layers[layer].cache.ready[slot];
    }

    pub fn warmSlotsPerLayer(self: *const Engine) u16 {
        return warmSlotCount(self.plan.slots_per_layer);
    }

    pub fn noteRoute(self: *Engine, layer: u16, full_attn: bool, detail: RouteDetail) void {
        if (layer < self.layers.len) self.layers[layer].stats.host_sync_ns +|= detail.totalNs();
        if (full_attn) self.route_full.add(detail) else self.route_linear.add(detail);
    }

    pub fn routeProbeActive(self: *const Engine) bool {
        const next = self.forward_count + 1;
        return next <= 3 or next % BREAKDOWN_EVERY == BREAKDOWN_EVERY / 2;
    }

    pub fn noteExpertCompute(self: *Engine, layer: u16, nanoseconds: u64) void {
        if (layer < self.layers.len) self.layers[layer].stats.compute_ns += nanoseconds;
    }

    pub fn pickState(self: *Engine, layer: u16, cached_out: []bool) ?[]u8 {
        if (layer >= self.layers.len) return null;
        const cache = &self.layers[layer].cache;
        cache.cachedMask(cached_out);
        return cache.swap_starved;
    }

    pub fn layerHits(self: *const Engine, layer: u16) u64 {
        return if (layer < self.layers.len) self.layers[layer].stats.hits else 0;
    }

    pub fn layerUnionMembers(self: *const Engine, layer: u16) u64 {
        return if (layer < self.layers.len) self.layers[layer].stats.union_members else 0;
    }

    pub fn beginForward(self: *Engine) void {
        for (self.layers) |*layer| layer.stats = .{};
        self.route_linear = .{};
        self.route_full = .{};
        self.forward_clock = Clock.init();
    }

    pub fn breakdown(self: *const Engine) Breakdown {
        return self.last;
    }

    pub fn finishForward(self: *Engine, rows: usize) void {
        self.forward_count += 1;
        const wall_ns = self.forward_clock.lap();
        var acc = Breakdown{
            .forward = self.forward_count,
            .rows = rows,
            .wall_ns = wall_ns,
            .route_linear = self.route_linear,
            .route_full = self.route_full,
            .probe = self.route_full.x_wait_ns + self.route_linear.x_wait_ns > 0,
        };
        for (self.layers) |layer| {
            acc.fill_bytes += layer.stats.fill_bytes;
            acc.route_ns += layer.stats.host_sync_ns;
            acc.fill_ns += layer.stats.fill_ns;
            acc.compute_ns += layer.stats.compute_ns;
            acc.hits += layer.stats.hits;
            acc.union_members += layer.stats.union_members;
        }
        self.last = acc;
        const scheduled = self.forward_count <= 2 or self.forward_count % BREAKDOWN_EVERY == 0;
        if (!scheduled and !acc.probe) return;
        const fill_seconds = @as(f64, @floatFromInt(acc.fill_ns)) / 1e9;
        log.info("[expert-stream] forward={d} rows={d} wall_ms={d:.1} route_ms={d:.1} fill_ms={d:.1} compute_ms={d:.1} other_ms={d:.1} fill_gb={d:.3} fill_gbps={d:.2} fill_bytes_per_row={d} hits={d}/{d} slab_leaks={d}\n", .{
            acc.forward,
            acc.rows,
            @as(f64, @floatFromInt(acc.wall_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.fill_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.compute_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.otherNs())) / 1e6,
            @as(f64, @floatFromInt(acc.fill_bytes)) / 1e9,
            if (fill_seconds > 0) @as(f64, @floatFromInt(acc.fill_bytes)) / 1e9 / fill_seconds else 0,
            acc.fill_bytes / @max(rows, 1),
            acc.hits,
            acc.union_members,
            slab_release_timeouts.load(.monotonic),
        });
        log.info("[expert-stream] route forward={d} rows={d} probe={} linear n={d} build_ms={d:.2} xwait_ms={d:.2} idswait_ms={d:.2} read_ms={d:.2} full n={d} build_ms={d:.2} xwait_ms={d:.2} idswait_ms={d:.2} read_ms={d:.2}\n", .{
            acc.forward,
            acc.rows,
            acc.probe,
            acc.route_linear.layers,
            @as(f64, @floatFromInt(acc.route_linear.build_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_linear.x_wait_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_linear.ids_wait_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_linear.read_ns)) / 1e6,
            acc.route_full.layers,
            @as(f64, @floatFromInt(acc.route_full.build_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_full.x_wait_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_full.ids_wait_ns)) / 1e6,
            @as(f64, @floatFromInt(acc.route_full.read_ns)) / 1e6,
        });
    }

    fn splitViews(self: *Engine, gate_up: mlx.mlx_array, down_raw: mlx.mlx_array, slots: c_int) !FusedViews {
        const inter: c_int = @intCast(self.geometry.intermediate);
        const hidden: c_int = @intCast(self.geometry.hidden);
        const strides = [_]c_int{ 1, 1, 1 };
        const axes = [_]c_int{ 0, 2, 1 };
        var gate_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_raw);
        try mlx.check(mlx.mlx_slice(&gate_raw, gate_up, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ slots, inter, hidden }, 3, &strides, 3, self.s));
        var up_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_raw);
        try mlx.check(mlx.mlx_slice(&up_raw, gate_up, &[_]c_int{ 0, inter, 0 }, 3, &[_]c_int{ slots, 2 * inter, hidden }, 3, &strides, 3, self.s));
        var gate = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_transpose_axes(&gate, gate_raw, &axes, 3, self.s));
        var up = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(up);
        try mlx.check(mlx.mlx_transpose_axes(&up, up_raw, &axes, 3, self.s));
        var down = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(down);
        try mlx.check(mlx.mlx_transpose_axes(&down, down_raw, &axes, 3, self.s));
        return .{ .gate = gate, .up = up, .down = down };
    }

    pub const SpecRoute = struct {
        /// int32 [experts]: the slot per expert, 0 where the host map says -1.
        slots: mlx.mlx_array,
        /// int32 [experts]: the host map itself, -1 where an expert is not ready.
        raw: mlx.mlx_array,
        /// Per-expert swap starvation counts (`substituteMisses` state).
        starved: []const u8,
        host: []const i32,
        operands: [quant.component_count]mlx.mlx_array,
    };

    /// The layer's cache map and slab operands, so a decode forward can queue
    /// the expert compute before the host reads the router ids. Null when the
    /// store is not quantized.
    pub fn specRoute(self: *Engine, layer_index: u16) !?SpecRoute {
        if (self.store.quantized == null or self.store.layout() == .bf16_individual or layer_index >= self.layers.len) return null;
        const layer = &self.layers[layer_index];
        if (!layer.active or layer.slabs.len != self.store.componentCount()) return null;
        try layer.refreshSpec(self.allocator);
        var operands: [quant.component_count]mlx.mlx_array = @splat(.{ .ctx = null });
        for (layer.slabs, 0..) |operand, ci| operands[ci] = operand.array;
        return .{ .slots = layer.spec_slots, .raw = layer.spec_raw, .starved = layer.cache.swap_starved, .host = layer.spec_host, .operands = operands };
    }

    pub fn prepareHost(self: *Engine, layer_index: u16, occurrences: []const u16) !Prepared {
        if (layer_index >= self.layers.len) return error.ExpertLayerOutOfRange;
        if (!self.layers[layer_index].active) return error.ExpertLayerAbsent;
        self.drainPendingReaders();
        return self.prepareSlab(layer_index, occurrences);
    }

    fn snapshotCache(self: *Engine, cache: *const GroupCache) !CacheSnapshot {
        return .{
            .expert_to_slot = try self.allocator.dupe(i32, cache.expert_to_slot),
            .slot_to_expert = try self.allocator.dupe(u16, cache.slot_to_expert),
            .ages = try self.allocator.dupe(u64, cache.ages),
            .ready = try self.allocator.dupe(bool, cache.ready),
            .tick = cache.tick,
        };
    }

    fn runFill(self: *Engine, layer_index: u16, spans: []const io_mod.FillSpan) !void {
        if (spans.len == 0) return;
        const pool = self.io_pool orelse return error.MissingFillPool;
        var plan = try io_mod.FillPlan.init(self.allocator, spans, COALESCE_MAX, pool.opts.workers, false);
        defer plan.deinit();
        var bytes: u64 = 0;
        for (spans) |span_value| bytes += span_value.len;
        var clock = Clock.init();
        try pool.submit(self.fds, &plan);
        const outcome = pool.join();
        const elapsed = clock.lap();
        self.fill_ns_total += elapsed;
        self.layers[layer_index].stats.fill_ns += elapsed;
        try outcome;
        self.fill_bytes_total += bytes;
    }

    pub fn cancelFills(self: *Engine) void {
        if (self.io_pool) |pool| pool.requestCancel();
    }

    pub fn warmCache(self: *Engine) !void {
        const warm_count = self.warmSlotsPerLayer();
        if (warm_count == 0) return;

        const occurrences = try self.allocator.alloc(u16, warm_count);
        defer self.allocator.free(occurrences);
        for (occurrences, 0..) |*expert, i| expert.* = @intCast(i);

        for (self.layers, 0..) |*layer, layer_index| {
            if (!layer.active) continue;
            {
                const saved_stats = layer.stats;
                defer layer.stats = saved_stats;
                var prepared = try self.prepareHost(@intCast(layer_index), occurrences);
                prepared.deinit();
            }
        }
    }

    fn prepareSlab(self: *Engine, layer_index: u16, occurrences: []const u16) !Prepared {
        const layer = &self.layers[layer_index];
        if (!layer.active) return error.ExpertLayerAbsent;
        const n = self.store.componentCount();
        if (layer.slabs.len != n) return error.MissingExpertSlab;
        var snapshot = try self.snapshotCache(&layer.cache);
        defer snapshot.deinit(self.allocator);
        var committed = false;
        var wrote_cache = false;
        errdefer if (!committed) {
            if (wrote_cache) layer.cache.invalidateUnready() else snapshot.restore(&layer.cache);
        };

        var resolution = try layer.cache.resolve(self.allocator, occurrences);
        defer resolution.deinit(self.allocator);
        const use_union = resolution.workspace;
        if (use_union and self.union_slabs.len != n) return error.MissingExpertSlab;
        const read_set = if (use_union) self.union_slabs else layer.slabs;

        var read_begun: usize = 0;
        errdefer if (!committed) for (read_set[0..read_begun]) |*operand| abortFill(operand);
        while (read_begun < n) : (read_begun += 1) {
            if (!read_set[read_begun].present) continue;
            const slab = read_set[read_begun].slab orelse return error.MissingExpertSlab;
            _ = try slab.tryBeginFill();
        }
        var cache_begun: usize = 0;
        errdefer if (!committed) for (layer.slabs[0..cache_begun]) |*operand| abortFill(operand);
        if (use_union) {
            while (cache_begun < n) : (cache_begun += 1) {
                if (!layer.slabs[cache_begun].present) continue;
                const slab = layer.slabs[cache_begun].slab orelse return error.MissingExpertSlab;
                _ = try slab.tryBeginFill();
            }
        }

        if (use_union) for (read_set, layer.slabs) |*operand, cached| {
            operand.stride = cached.stride;
        };

        var spans: std.ArrayList(io_mod.FillSpan) = .empty;
        defer spans.deinit(self.allocator);
        var misses: u64 = 0;
        for (resolution.bindings, 0..) |binding, position| {
            if (binding.hit and use_union) {
                const slot = binding.slot orelse return error.ExpertBindingMissing;
                for (0..n) |ci| {
                    if (!read_set[ci].present) continue;
                    @memcpy(read_set[ci].slotBytes(position), layer.slabs[ci].slotBytes(slot));
                }
                continue;
            }
            if (binding.hit) continue;
            misses += 1;
            for (0..n) |ci| {
                if (!read_set[ci].present) continue;
                const source = self.store.spanAt(layer_index, binding.expert, ci);
                const dst = if (use_union)
                    read_set[ci].slotBytes(position)
                else
                    layer.slabs[ci].slotBytes(binding.slot orelse return error.ExpertBindingMissing);
                try spans.append(self.allocator, .{ .file = source.file, .offset = source.offset, .len = source.len, .dst = dst.ptr });
            }
        }
        if (!use_union and spans.items.len > 0) wrote_cache = true;
        try self.runFill(layer_index, spans.items);
        self.fill_experts_total += misses;

        if (use_union) {
            for (resolution.bindings, 0..) |binding, position| {
                if (binding.hit) continue;
                const slot = binding.slot orelse continue;
                wrote_cache = true;
                for (0..n) |ci| {
                    if (!layer.slabs[ci].present) continue;
                    @memcpy(layer.slabs[ci].slotBytes(slot), read_set[ci].slotBytes(position));
                }
                layer.cache.markReady(slot);
            }
            for (layer.slabs) |*operand| {
                if (!operand.present) continue;
                const slab = operand.slab orelse return error.MissingExpertSlab;
                _ = try slab.publish();
                try slab.retire();
            }
        } else {
            for (resolution.bindings) |binding| {
                if (binding.hit) continue;
                layer.cache.markReady(binding.slot.?);
            }
        }

        var held: [quant.component_count]?HeldLease = @splat(null);
        var leased: usize = 0;
        errdefer for (held[0..leased]) |entry| {
            const taken = entry orelse continue;
            taken.slab.release(taken.lease) catch {};
            taken.slab.retire() catch {};
        };
        while (leased < n) : (leased += 1) {
            if (!read_set[leased].present) continue;
            const slab = read_set[leased].slab orelse return error.MissingExpertSlab;
            _ = try slab.publish();
            held[leased] = .{ .slab = slab, .lease = try slab.lease() };
        }

        const remapped = try self.allocator.alloc(u16, occurrences.len);
        errdefer self.allocator.free(remapped);
        if (use_union) {
            @memcpy(remapped, resolution.remapped);
        } else {
            for (occurrences, 0..) |expert, i| {
                const raw = layer.cache.expert_to_slot[expert];
                if (raw < 0) return error.ExpertBindingMissing;
                remapped[i] = @intCast(raw);
            }
        }

        const individual_bf16 = self.store.layout() == .bf16_individual;
        const quantized = self.store.quantized != null and !individual_bf16;
        var raws: [quant.component_count]mlx.mlx_array = @splat(.{ .ctx = null });
        var views: FusedViews = .{ .gate = .{ .ctx = null }, .up = .{ .ctx = null }, .down = .{ .ctx = null } };
        if (quantized) {
            errdefer if (use_union) for (raws) |arr| {
                if (arr.ctx != null) _ = mlx.mlx_array_free(arr);
            };
            for (0..n) |ci| {
                raws[ci] = if (use_union and read_set[ci].present)
                    try read_set[ci].viewAs(layer.slabs[ci].array, self.s)
                else
                    read_set[ci].array;
            }
        } else if (individual_bf16) {
            const axes = [_]c_int{ 0, 2, 1 };
            views.gate = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(views.gate);
            try mlx.check(mlx.mlx_transpose_axes(&views.gate, read_set[@backingInt(quant.Component.gate_w)].array, &axes, 3, self.s));
            views.up = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(views.up);
            try mlx.check(mlx.mlx_transpose_axes(&views.up, read_set[@backingInt(quant.Component.up_w)].array, &axes, 3, self.s));
            views.down = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(views.down);
            try mlx.check(mlx.mlx_transpose_axes(&views.down, read_set[@backingInt(quant.Component.down_w)].array, &axes, 3, self.s));
        } else {
            raws[0] = try read_set[0].borrow();
            errdefer _ = mlx.mlx_array_free(raws[0]);
            raws[1] = try read_set[1].borrow();
            errdefer _ = mlx.mlx_array_free(raws[1]);
            views = try self.splitViews(raws[0], raws[1], @intCast(read_set[0].count));
        }

        layer.stats.groups += 1;
        layer.stats.union_members += resolution.union_ids.len;
        layer.stats.hits += resolution.hits;
        layer.stats.misses += resolution.misses;
        for (spans.items) |span_value| layer.stats.fill_bytes += span_value.len;
        committed = true;
        return .{
            .allocator = self.allocator,
            .gate = views.gate,
            .up = views.up,
            .down = views.down,
            .remapped = remapped,
            .raw_gate_up = if (quantized) .{ .ctx = null } else raws[0],
            .raw_down = if (quantized) .{ .ctx = null } else raws[1],
            .quant_raw = if (quantized) raws else @splat(.{ .ctx = null }),
            .quantized = quantized,
            .owns_quant_views = quantized and use_union,
            .workspace = resolution.workspace,
            .held = held,
            .defer_to = self,
        };
    }
};

test "expert stream span adapter derives fused first middle and last slices" {
    const t = std.testing;
    const layout = TensorLayout{ .data_offset = 4096, .tensor_offset = 8192, .tensor_bytes = 300, .experts = 3 };
    try t.expectEqual(Span{ .offset = 12288, .len = 100 }, try layout.expertSpan(0));
    try t.expectEqual(Span{ .offset = 12388, .len = 100 }, try layout.expertSpan(1));
    try t.expectEqual(Span{ .offset = 12488, .len = 100 }, try layout.expertSpan(2));
    try t.expectError(error.ExpertOutOfRange, layout.expertSpan(3));
}

fn markResolvedReady(cache: *GroupCache, resolution: *const GroupResolution) void {
    for (resolution.bindings) |binding| {
        if (binding.hit) continue;
        if (binding.slot) |slot| cache.markReady(slot);
    }
}

test "expert stream group exact LRU matches replay counts" {
    const t = std.testing;
    var cache = try GroupCache.init(t.allocator, 2, 4);
    defer cache.deinit();

    var r1 = try cache.resolve(t.allocator, &.{ 0, 1 });
    defer r1.deinit(t.allocator);
    markResolvedReady(&cache, &r1);
    try t.expectEqual(@as(usize, 0), r1.hits);
    try t.expectEqual(@as(usize, 2), r1.misses);
    try t.expectEqual(@as(usize, 0), r1.declined);

    var r2 = try cache.resolve(t.allocator, &.{ 0, 2 });
    defer r2.deinit(t.allocator);
    markResolvedReady(&cache, &r2);
    try t.expectEqual(@as(usize, 1), r2.hits);
    try t.expectEqual(@as(usize, 1), r2.misses);
    try t.expectEqual(@as(usize, 0), r2.declined);

    var r3 = try cache.resolve(t.allocator, &.{ 1, 0, 2 });
    defer r3.deinit(t.allocator);
    try t.expectEqual(@as(usize, 2), r3.hits);
    try t.expectEqual(@as(usize, 1), r3.misses);
    try t.expectEqual(@as(usize, 1), r3.declined);
    try t.expect(r3.workspace);
    try t.expectEqualSlices(u16, &.{ 1, 0, 2 }, r3.union_ids);
    try t.expectEqualSlices(u16, &.{ 0, 1, 2 }, r3.remapped);
}

test "expert stream union orders experts by last occurrence and remaps rows" {
    const t = std.testing;
    var cache = try GroupCache.init(t.allocator, 4, 8);
    defer cache.deinit();
    var result = try cache.resolve(t.allocator, &.{ 1, 2, 1, 3, 2 });
    defer result.deinit(t.allocator);
    markResolvedReady(&cache, &result);
    try t.expectEqualSlices(u16, &.{ 1, 3, 2 }, result.union_ids);
    try t.expectEqualSlices(u16, &.{ 0, 2, 0, 1, 2 }, result.remapped);
    try t.expect(!result.workspace);

    var reordered = try cache.resolve(t.allocator, &.{ 1, 2, 3 });
    defer reordered.deinit(t.allocator);
    try t.expectEqualSlices(u16, &.{ 1, 2, 3 }, reordered.union_ids);
    try t.expectEqualSlices(u16, &.{ 0, 1, 2 }, reordered.remapped);
    try t.expectEqual(@as(usize, 3), reordered.hits);
}

test "expert stream cache plan uses decimal gigabytes and reserves full workspace" {
    const t = std.testing;
    const p = try cachePlanBytes(60_000_000_000, 48, 512, try expertBytes(1280, 2560, 640));
    try t.expectEqual(@as(u16, 127), p.slots_per_layer);
    try t.expectEqual(@as(u64, 9_830_400), p.expert_bytes);
    try t.expectEqual(@as(u64, 48 * 127 * 9_830_400), p.cache_bytes);
    try t.expectEqual(@as(u64, 512 * 9_830_400), p.workspace_bytes);
    try t.expectEqual(@as(u64, 8 * 64 * 1024 * 1024), p.bounce_bytes);
}

test "expert stream MXFP4 byte plan and dense prefix use absolute layers" {
    const t = std.testing;
    const geometry = Geometry{
        .layers = 48,
        .experts = 256,
        .hidden = 4096,
        .intermediate = 2048,
        .first_moe_layer = 1,
    };
    try t.expectEqual(@as(u16, 47), moeLayerCount(geometry));
    try t.expectEqual(@as(u64, 13_369_344), try mxfp4ExpertBytes(4096, 2048));
    const plan = try cachePlanBytesForGeometry(60_000_000_000, geometry, try mxfp4ExpertBytes(4096, 2048));
    try t.expectEqual(@as(u16, 95), plan.slots_per_layer);
    try t.expectEqual(@as(u64, 47 * 95 * 13_369_344), plan.cache_bytes);
    try t.expectError(error.InvalidExpertGeometry, mxfp4ExpertBytes(4096, 2050));
    try t.expectError(error.ExpertCacheTooSmall, cachePlanBytesForGeometry(47 * 13_369_344 - 1, geometry, 13_369_344));
}

test "exl3 expert bytes bill the pack's own halfwords per tile" {
    const t = std.testing;
    const h: u64 = 2560;
    const i: u64 = 640;
    for ([_]u32{ 16, 24, 32, 36, 40, 48, 64, 80, 104, 128 }) |n| {
        const b = try exl3ExpertBytes(.{ .layers = 48, .experts = 512, .hidden = h, .intermediate = i, .exl3_n = n });
        const tile: u64 = n;
        const gate_up = (h / 16) * (i / 16) * tile * 2 + h * 2 + i * 2;
        const down = (i / 16) * (h / 16) * tile * 2 + i * 2 + h * 2;
        try t.expectEqual(gate_up * 2 + down, b);
    }
    try t.expectError(error.InvalidExpertGeometry, exl3ExpertBytes(.{ .layers = 48, .experts = 512, .hidden = h, .intermediate = i, .exl3_n = 41 }));
}

test "exl3 expert bytes at the MiMo K2.5 geometry size the resident admission" {
    const t = std.testing;
    const geometry: Geometry = .{
        .layers = 48,
        .experts = 256,
        .hidden = 4096,
        .intermediate = 2048,
        .first_moe_layer = 1,
        .exl3_n = 40,
    };
    const per_expert = try exl3ExpertBytes(geometry);
    try t.expectEqual(@as(u64, 7_901_184), per_expert);
    const routed_layers: u64 = geometry.layers - geometry.first_moe_layer;
    try t.expectEqual(@as(u64, 95_067_045_888), per_expert * geometry.experts * routed_layers);
}

test "expert stream refuses MTP by name, at load and at request parse" {
    const t = std.testing;
    try t.expect(mtpRefusal(false, true, false) == null);
    try t.expect(mtpRefusal(true, false, false) == null);
    try t.expect(mtpRefusal(false, false, false) == null);
    try t.expect(mtpRefusal(true, true, true) == null);
    const why = mtpRefusal(true, true, false) orelse return error.TestExpectedRefusal;
    try t.expect(std.mem.indexOf(u8, why, "expert streaming") != null);
}

test "a capable checkpoint streams only when it must or when a budget was asked for" {
    const t = std.testing;
    try t.expect(!expertStreamingEngaged(false, false, 0, 0));
    try t.expect(!expertStreamingEngaged(false, true, 0, 60 << 30));
    // The dense HF checkpoint cannot load resident: it streams with no budget.
    try t.expect(expertStreamingEngaged(true, true, 0, 0));
    // A quantized pack with neither flag loads resident.
    try t.expect(!expertStreamingEngaged(true, false, 0, 0));
    try t.expect(expertStreamingEngaged(true, false, 0, 50 << 30));
    try t.expect(expertStreamingEngaged(true, false, 60_000_000_000, 0));
}

test "expert stream ssd budget ledger derives the cache and refuses by name" {
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const expert_bytes: u64 = 9_830_400;
    const bounce: u64 = 8 * 64 * 1024 * 1024;
    const trunk: u64 = 9_900_000_000;

    const led = try budgetLedger(100 * GiB, trunk, 0, 48, 512, 10, expert_bytes, bounce);
    try t.expectEqual(@as(u64, 512 * expert_bytes), led.workspace_bytes);
    try t.expectEqual(@as(u64, 10 * expert_bytes), led.selected_bytes);
    try t.expectEqual(@as(u64, 0), led.mtp_bytes);
    try t.expectEqual(trunk, led.trunk_bytes);
    try t.expectEqual(bounce, led.bounce_bytes);
    try t.expectEqual(@as(u16, 194), led.slots_per_layer);
    try t.expectEqual(@as(u64, 194) * 48 * expert_bytes, led.cache_bytes);

    const with_mtp = try budgetLedger(100 * GiB, trunk, 5_200_000_000, 48, 512, 10, expert_bytes, bounce);
    try t.expectEqual(@as(u16, 183), with_mtp.slots_per_layer);

    try t.expectError(error.SsdBudgetBelowResident, budgetLedger(14 * GiB, trunk, 0, 48, 512, 10, expert_bytes, bounce));
    const fixed = trunk + 512 * expert_bytes + 10 * expert_bytes + bounce;
    const per_slot = 48 * expert_bytes;
    try t.expectError(error.SsdBudgetBelowResident, budgetLedger(fixed + 2 * per_slot - 1, trunk, 0, 48, 512, 10, expert_bytes, bounce));
    const two = try budgetLedger(fixed + 2 * per_slot, trunk, 0, 48, 512, 10, expert_bytes, bounce);
    try t.expectEqual(@as(u16, 2), two.slots_per_layer);

    try t.expect(!budgetOverriddenByExplicitCache(0, 100 * GiB));
    try t.expect(!budgetOverriddenByExplicitCache(60_000_000_000, 0));
    try t.expect(budgetOverriddenByExplicitCache(60_000_000_000, 100 * GiB));
}

test "expert stream prefill peak bill covers a 501 expert union and k127 cache layer" {
    const t = std.testing;
    const p = try cachePlanBytes(60_000_000_000, 48, 512, try expertBytes(1280, 2560, 640));
    const peak = fillPeakBytes(p.expert_bytes, 501, p.slots_per_layer, p.bounce_bytes);
    try t.expectEqual(@as(u64, 127 * 9_830_400), p.layer_bytes);
    try t.expect(p.prefill_peak_bytes >= peak);
}

test "expert store maps fused safetensors spans and reads identical bytes" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"model.language_model.layers.0.mlp.experts.gate_up_proj\":{\"dtype\":\"BF16\",\"shape\":[3,2,2],\"data_offsets\":[0,24]},\"model.language_model.layers.0.mlp.experts.down_proj\":{\"dtype\":\"BF16\",\"shape\":[3,2,1],\"data_offsets\":[24,36]}}";
    const file_bytes = try t.allocator.alloc(u8, 8 + header.len + 36);
    defer t.allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.len, .little);
    @memcpy(file_bytes[8 .. 8 + header.len], header);
    for (file_bytes[8 + header.len ..], 0..) |*b, i| b.* = @intCast(i);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = file_bytes });
    try tmp.dir.writeFile(io, .{
        .sub_path = "model.safetensors.index.json",
        .data = "{\"weight_map\":{\"model.language_model.layers.0.mlp.experts.gate_up_proj\":\"model-00001-of-00001.safetensors\",\"model.language_model.layers.0.mlp.experts.down_proj\":\"model-00001-of-00001.safetensors\"}}",
    });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    var store = try ExpertStore.open(t.allocator, path_buf[0..path_len], .{ .layers = 1, .experts = 3, .hidden = 2, .intermediate = 1 });
    defer store.deinit();
    const data_offset: u64 = 8 + header.len;
    try t.expectEqual(SourceSpan{ .file = 0, .offset = data_offset, .len = 8 }, store.span(0, 0, .gate_up));
    try t.expectEqual(SourceSpan{ .file = 0, .offset = data_offset + 16, .len = 8 }, store.span(0, 2, .gate_up));
    try t.expectEqual(SourceSpan{ .file = 0, .offset = data_offset + 24 + 8, .len = 4 }, store.span(0, 2, .down));
    var got: [12]u8 = undefined;
    try store.readExpert(0, 1, &got);
    try t.expectEqualSlices(u8, file_bytes[8 + header.len + 8 .. 8 + header.len + 16], got[0..8]);
    try t.expectEqualSlices(u8, file_bytes[8 + header.len + 28 .. 8 + header.len + 32], got[8..12]);
}

test "bf16 ngram store gathers rows across safetensors shards" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const key0 = "model.language_model.layers.3.ple.ple_embedding.ngram_embedding.shard_0.weight";
    const key1 = "model.language_model.layers.3.ple.ple_embedding.ngram_embedding.shard_1.weight";
    const header = try std.fmt.allocPrint(t.allocator, "{{\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,8]}},\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[8,16]}}}}", .{ key0, key1 });
    defer t.allocator.free(header);
    const file_bytes = try t.allocator.alloc(u8, 8 + header.len + 16);
    defer t.allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.len, .little);
    @memcpy(file_bytes[8 .. 8 + header.len], header);
    const values = [_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100 };
    @memcpy(file_bytes[8 + header.len ..], std.mem.sliceAsBytes(&values));
    try tmp.dir.writeFile(io, .{ .sub_path = "table.safetensors", .data = file_bytes });
    const index = try std.fmt.allocPrint(t.allocator, "{{\"weight_map\":{{\"{s}\":\"table.safetensors\",\"{s}\":\"table.safetensors\"}}}}", .{ key0, key1 });
    defer t.allocator.free(index);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    var store = try Bf16NgramStore.open(t.allocator, path_buf[0..path_len]);
    defer store.deinit();
    try t.expectEqual(@as(u64, 4), store.rows);
    try t.expectEqual(@as(u32, 2), store.dim);
    var out: [4]f32 = undefined;
    try store.gather(&.{ 0, 3 }, &out);
    try t.expectEqualSlices(f32, &.{ 1.0, 2.0, 7.0, 8.0 }, &out);
}

test "expert stream slab operand preserves direct gather output" {
    const t = std.testing;
    const sandboxed = std.c.getenv("CODEX_SANDBOX") != null;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const gate_key = "model.language_model.layers.0.mlp.experts.gate_up_proj";
    const down_key = "model.language_model.layers.0.mlp.experts.down_proj";
    const header = try std.fmt.allocPrint(t.allocator, "{{\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[2,2,2],\"data_offsets\":[0,16]}},\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[2,2,1],\"data_offsets\":[16,24]}}}}", .{ gate_key, down_key });
    defer t.allocator.free(header);
    const file_bytes = try t.allocator.alloc(u8, 8 + header.len + 24);
    defer t.allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.len, .little);
    @memcpy(file_bytes[8 .. 8 + header.len], header);
    const weights = [_]u16{ 0x3f80, 0x4000, 0x4040, 0x4080, 0x4100, 0x4110, 0x4120, 0x4130, 0x40a0, 0x40c0, 0x4140, 0x4150 };
    @memcpy(file_bytes[8 + header.len ..], std.mem.sliceAsBytes(&weights));
    try tmp.dir.writeFile(io, .{ .sub_path = "experts.safetensors", .data = file_bytes });
    const index = try std.fmt.allocPrint(t.allocator, "{{\"weight_map\":{{\"{s}\":\"experts.safetensors\",\"{s}\":\"experts.safetensors\"}}}}", .{ gate_key, down_key });
    defer t.allocator.free(index);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    const s = if (sandboxed) mlx.mlx_default_cpu_stream_new() else mlx.gpuStream();
    defer if (sandboxed) {
        _ = mlx.mlx_stream_free(s);
    };
    var engine = try Engine.initWithOptions(t.allocator, path_buf[0..path_len], .{ .layers = 1, .experts = 2, .hidden = 2, .intermediate = 1 }, 12, s, .{ .bounce_size = 1 << 20 });
    defer engine.deinit();
    {
        var prepared = try engine.prepareHost(0, &.{0});
        defer prepared.deinit();
        const x_data = [_]u16{ 0x3f80, 0x3f80 };
        const x = mlx.mlx_array_new_data(&x_data, &[_]c_int{ 1, 1, 1, 1, 2 }, 5, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const ids_data = [_]u32{0};
        const ids = mlx.mlx_array_new_data(&ids_data, &[_]c_int{ 1, 1, 1 }, 3, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const no_idx = mlx.mlx_array{ .ctx = null };
        var streamed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(streamed);
        try mlx.check(mlx.mlx_gather_mm(&streamed, x, prepared.gate, no_idx, ids, false, s));
        const direct_raw = mlx.mlx_array_new_data(std.mem.sliceAsBytes(weights[0..2]).ptr, &[_]c_int{ 1, 1, 2 }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(direct_raw);
        var direct_w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(direct_w);
        try mlx.check(mlx.mlx_transpose_axes(&direct_w, direct_raw, &[_]c_int{ 0, 2, 1 }, 3, s));
        var direct = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(direct);
        try mlx.check(mlx.mlx_gather_mm(&direct, x, direct_w, no_idx, ids, false, s));
        try mlx.check(mlx.mlx_array_eval(streamed));
        try mlx.check(mlx.mlx_array_eval(direct));
        const direct_bytes = mlx.mlx_array_size(direct) * @sizeOf(u16);
        const streamed_bytes = mlx.mlx_array_size(streamed) * @sizeOf(u16);
        try t.expectEqual(direct_bytes, streamed_bytes);
        try t.expectEqualSlices(u8, @as([*]const u8, @ptrCast(mlx.mlx_array_data_bfloat16(direct).?))[0..direct_bytes], @as([*]const u8, @ptrCast(mlx.mlx_array_data_bfloat16(streamed).?))[0..streamed_bytes]);
    }
    var workspace = try engine.prepareHost(0, &.{ 0, 1 });
    defer workspace.deinit();
    try t.expect(workspace.workspace);
    try t.expectEqualSlices(u16, &.{ 0, 1 }, workspace.remapped);
    var workspace_gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(workspace_gate);
    try mlx.check(mlx.mlx_contiguous(&workspace_gate, workspace.gate, false, s));
    try mlx.check(mlx.mlx_array_eval(workspace_gate));
    try t.expectEqualSlices(u16, &.{ 0x3f80, 0x4000, 0x4100, 0x4110 }, mlx.mlx_array_data_bfloat16(workspace_gate).?[0..4]);
}

test "real qwen expert store spans and source bytes are exact" {
    const t = std.testing;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try @import("test_models.zig").packPath(&path_buf, "Qwen/Qwen3.8-Flash-Next");
    var dir = std.Io.Dir.openDirAbsolute(t.io, path, .{}) catch return error.SkipZigTest;
    dir.close(t.io);
    var store = try ExpertStore.open(t.allocator, path, .{ .layers = 48, .experts = 512, .hidden = 2560, .intermediate = 640 });
    defer store.deinit();
    const gate_first = store.span(2, 0, .gate_up);
    const gate_last = store.span(2, 511, .gate_up);
    const down_first = store.span(2, 0, .down);
    const down_last = store.span(2, 511, .down);
    try t.expectEqual(@as(u64, 6_553_600), gate_first.len);
    try t.expectEqual(@as(u64, 3_276_800), down_first.len);
    try t.expectEqual(@as(u64, 511 * 6_553_600), gate_last.offset - gate_first.offset);
    try t.expectEqual(@as(u64, 511 * 3_276_800), down_last.offset - down_first.offset);
    const combined = try t.allocator.alloc(u8, 9_830_400);
    defer t.allocator.free(combined);
    const direct = try t.allocator.alloc(u8, 9_830_400);
    defer t.allocator.free(direct);
    try store.readExpert(2, 17, combined);
    const gate = store.span(2, 17, .gate_up);
    const down = store.span(2, 17, .down);
    try readExact(store.files[gate.file].fd, direct[0..@intCast(gate.len)], gate.offset);
    try readExact(store.files[down.file].fd, direct[@intCast(gate.len)..], down.offset);
    try t.expectEqualSlices(u8, direct, combined);
}

pub fn writeTinyExpertCheckpoint(allocator: std.mem.Allocator, dir: std.Io.Dir, experts: u16, hidden: u32, inter: u32) ![]u8 {
    const io = std.testing.io;
    const gate_key = "model.language_model.layers.0.mlp.experts.gate_up_proj";
    const down_key = "model.language_model.layers.0.mlp.experts.down_proj";
    const gate_bytes: usize = @as(usize, experts) * 2 * inter * hidden * 2;
    const down_bytes: usize = @as(usize, experts) * hidden * inter * 2;
    const header = try std.fmt.allocPrint(
        allocator,
        "{{\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d},{d}],\"data_offsets\":[0,{d}]}},\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d},{d}],\"data_offsets\":[{d},{d}]}}}}",
        .{ gate_key, experts, 2 * inter, hidden, gate_bytes, down_key, experts, hidden, inter, gate_bytes, gate_bytes + down_bytes },
    );
    defer allocator.free(header);
    const file_bytes = try allocator.alloc(u8, 8 + header.len + gate_bytes + down_bytes);
    errdefer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.len, .little);
    @memcpy(file_bytes[8..][0..header.len], header);
    const payload = file_bytes[8 + header.len ..];
    var i: usize = 0;
    while (i + 1 < payload.len) : (i += 2) {
        const mixed = (@as(u64, i / 2) +% 1) *% 0x9E3779B97F4A7C15;
        const sign: u16 = @intCast((mixed >> 40) & 1);
        const bits: u16 = (sign << 15) | (0x3e00 +% @as(u16, @intCast((mixed >> 53) % 0x0200)));
        std.mem.writeInt(u16, payload[i..][0..2], bits, .little);
    }
    try dir.writeFile(io, .{ .sub_path = "experts.safetensors", .data = file_bytes });
    const index = try std.fmt.allocPrint(
        allocator,
        "{{\"weight_map\":{{\"{s}\":\"experts.safetensors\",\"{s}\":\"experts.safetensors\"}}}}",
        .{ gate_key, down_key },
    );
    defer allocator.free(index);
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    return file_bytes;
}

fn writeTinyExpertCheckpointLayers(allocator: std.mem.Allocator, dir: std.Io.Dir, layers: u16, experts: u16, hidden: u32, inter: u32) ![]u8 {
    const io = std.testing.io;
    if (layers == 0 or experts == 0 or hidden == 0 or inter == 0) return error.InvalidTestGeometry;
    const gate_bytes = @as(usize, experts) * 2 * @as(usize, inter) * @as(usize, hidden) * 2;
    const down_bytes = @as(usize, experts) * @as(usize, hidden) * @as(usize, inter) * 2;
    const per_layer = gate_bytes + down_bytes;

    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    try header.append(allocator, '{');
    for (0..layers) |layer_usize| {
        if (layer_usize != 0) try header.append(allocator, ',');
        const layer = @as(u16, @intCast(layer_usize));
        const layer_offset = layer_usize * per_layer;
        const piece = try std.fmt.allocPrint(
            allocator,
            "\"model.language_model.layers.{d}.mlp.experts.gate_up_proj\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d},{d}],\"data_offsets\":[{d},{d}]}},\"model.language_model.layers.{d}.mlp.experts.down_proj\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d},{d}],\"data_offsets\":[{d},{d}]}}",
            .{
                layer,
                experts,
                2 * inter,
                hidden,
                layer_offset,
                layer_offset + gate_bytes,
                layer,
                experts,
                hidden,
                inter,
                layer_offset + gate_bytes,
                layer_offset + per_layer,
            },
        );
        defer allocator.free(piece);
        try header.appendSlice(allocator, piece);
    }
    try header.append(allocator, '}');

    const payload_bytes = @as(usize, layers) * per_layer;
    const file_bytes = try allocator.alloc(u8, 8 + header.items.len + payload_bytes);
    errdefer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.items.len, .little);
    @memcpy(file_bytes[8..][0..header.items.len], header.items);
    for (file_bytes[8 + header.items.len ..], 0..) |*b, i| b.* = @intCast(i % 251);
    try dir.writeFile(io, .{ .sub_path = "experts.safetensors", .data = file_bytes });

    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(allocator);
    try index.appendSlice(allocator, "{\"weight_map\":{");
    for (0..layers) |layer_usize| {
        if (layer_usize != 0) try index.append(allocator, ',');
        const layer = @as(u16, @intCast(layer_usize));
        const piece = try std.fmt.allocPrint(
            allocator,
            "\"model.language_model.layers.{d}.mlp.experts.gate_up_proj\":\"experts.safetensors\",\"model.language_model.layers.{d}.mlp.experts.down_proj\":\"experts.safetensors\"",
            .{ layer, layer },
        );
        defer allocator.free(piece);
        try index.appendSlice(allocator, piece);
    }
    try index.appendSlice(allocator, "}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    return file_bytes;
}

fn tinyStream() mlx.mlx_stream {
    return if (std.c.getenv("CODEX_SANDBOX") != null) mlx.mlx_default_cpu_stream_new() else mlx.gpuStream();
}

test "expert stream zero copy hit refills nothing and keeps the slab operand" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const raw = try writeTinyExpertCheckpoint(t.allocator, tmp.dir, 4, 64, 64);
    defer t.allocator.free(raw);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const path = path_buf[0..path_len];
    const geom = Geometry{ .layers = 1, .experts = 4, .hidden = 64, .intermediate = 64 };
    const s = tinyStream();
    var zero = try Engine.initWithOptions(t.allocator, path, geom, 2 * 24576, s, .{ .bounce_size = 1 << 20 });
    defer zero.deinit();
    try t.expectEqual(@as(u64, 0), zero.fallback_imports);
    try t.expectEqual(@as(u64, 2 * @as(u64, geom.layers) + 2), zero.slab_imports);

    var first = try zero.prepareHost(0, &.{1});
    const slot = first.remapped[0];
    try t.expect(slot < 2);
    const slab_ptr = zero.cacheSlotBytes(0, slot, .gate_up).ptr;
    first.deinit();
    try t.expectEqual(@as(u64, 1), zero.fill_experts_total);
    try t.expect(zero.slotReady(0, slot));

    var second = try zero.prepareHost(0, &.{1});
    defer second.deinit();
    try t.expectEqual(@as(u64, 1), zero.fill_experts_total);
    try t.expectEqual(slot, second.remapped[0]);
    try t.expectEqual(slab_ptr, zero.cacheSlotBytes(0, slot, .gate_up).ptr);

    const gate_span = zero.store.span(0, 1, .gate_up);
    try t.expectEqualSlices(u8, raw[@intCast(gate_span.offset)..][0..@intCast(gate_span.len)], zero.cacheSlotBytes(0, slot, .gate_up));
    const down_span = zero.store.span(0, 1, .down);
    try t.expectEqualSlices(u8, raw[@intCast(down_span.offset)..][0..@intCast(down_span.len)], zero.cacheSlotBytes(0, slot, .down));
    try t.expectEqual(@as(u64, 0), slab_release_timeouts.load(.monotonic));
}

const TinyEngine = struct {
    tmp: std.testing.TmpDir,
    raw: []u8,
    engine: Engine,

    fn open(slots: u16, experts: u16) !TinyEngine {
        const t = std.testing;
        var tmp = t.tmpDir(.{});
        errdefer tmp.cleanup();
        const raw = try writeTinyExpertCheckpoint(t.allocator, tmp.dir, experts, 64, 64);
        errdefer t.allocator.free(raw);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try tmp.dir.realPath(t.io, &path_buf);
        const geom = Geometry{ .layers = 1, .experts = experts, .hidden = 64, .intermediate = 64 };
        const engine = try Engine.initWithOptions(t.allocator, path_buf[0..path_len], geom, @as(u64, slots) * 24576, tinyStream(), .{ .io_workers = 2, .bounce_size = 1 << 20 });
        return .{ .tmp = tmp, .raw = raw, .engine = engine };
    }

    fn close(self: *TinyEngine) void {
        self.engine.deinit();
        std.testing.allocator.free(self.raw);
        self.tmp.cleanup();
    }

    fn anyReady(self: *const TinyEngine) bool {
        for (self.engine.layers[0].cache.ready) |flag| {
            if (flag) return true;
        }
        return false;
    }
};

const TinyLayersEngine = struct {
    tmp: std.testing.TmpDir,
    raw: []u8,
    engine: Engine,

    fn open(slots: u16, layers: u16, first_moe_layer: u16, experts: u16) !TinyLayersEngine {
        const t = std.testing;
        var tmp = t.tmpDir(.{});
        errdefer tmp.cleanup();
        const raw = try writeTinyExpertCheckpointLayers(t.allocator, tmp.dir, layers, experts, 64, 64);
        errdefer t.allocator.free(raw);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try tmp.dir.realPath(t.io, &path_buf);
        const geom = Geometry{
            .layers = layers,
            .experts = experts,
            .hidden = 64,
            .intermediate = 64,
            .first_moe_layer = first_moe_layer,
        };
        const requested = @as(u64, slots) * @as(u64, moeLayerCount(geom)) * 24576;
        const engine = try Engine.initWithOptions(t.allocator, path_buf[0..path_len], geom, requested, tinyStream(), .{ .io_workers = 2, .bounce_size = 1 << 20 });
        return .{ .tmp = tmp, .raw = raw, .engine = engine };
    }

    fn close(self: *TinyLayersEngine) void {
        self.engine.deinit();
        std.testing.allocator.free(self.raw);
        self.tmp.cleanup();
    }
};

test "expert stream warm cache floors 80 percent, including zero and one slot" {
    const t = std.testing;
    try t.expectEqual(@as(u16, 0), warmSlotCount(0));
    try t.expectEqual(@as(u16, 0), warmSlotCount(1));
    try t.expectEqual(@as(u16, 1), warmSlotCount(2));
    try t.expectEqual(@as(u16, 3), warmSlotCount(4));
    try t.expectEqual(@as(u16, 4), warmSlotCount(5));

    var fixture = try TinyEngine.open(1, 4);
    defer fixture.close();
    try t.expectEqual(@as(u16, 0), fixture.engine.warmSlotsPerLayer());
    try fixture.engine.warmCache();
    try t.expectEqual(@as(u64, 0), fixture.engine.fill_experts_total);
    for (fixture.engine.layers[0].cache.ready) |ready| try t.expect(!ready);
}

test "expert stream warm cache fills lowest experts in every active layer and is a hit thereafter" {
    const t = std.testing;
    var fixture = try TinyLayersEngine.open(5, 3, 1, 8);
    defer fixture.close();
    const engine = &fixture.engine;
    try t.expectEqual(@as(u16, 5), engine.plan.slots_per_layer);
    const warm_count = warmSlotCount(engine.plan.slots_per_layer);
    try t.expectEqual(warm_count, engine.warmSlotsPerLayer());
    try t.expectEqual(@as(u16, 4), warm_count);

    engine.layers[1].stats = .{ .groups = 71, .hits = 72, .misses = 73, .fill_bytes = 74, .host_sync_ns = 75, .fill_ns = 76, .compute_ns = 77 };
    try engine.warmCache();
    try t.expectEqual(@as(u64, 8), engine.fill_experts_total);
    try t.expectEqual(@as(usize, 0), engine.layers[0].slabs.len);
    try t.expect(!engine.layers[0].active);
    try t.expectEqual(@as(u64, 0), engine.layers[0].cache.tick);
    try t.expectEqual(@as(u64, 71), engine.layers[1].stats.groups);
    try t.expectEqual(@as(u64, 72), engine.layers[1].stats.hits);
    try t.expectEqual(@as(u64, 73), engine.layers[1].stats.misses);
    try t.expectEqual(@as(u64, 74), engine.layers[1].stats.fill_bytes);
    try t.expectEqual(@as(u64, 75), engine.layers[1].stats.host_sync_ns);
    try t.expectEqual(@as(u64, 76), engine.layers[1].stats.fill_ns);
    try t.expectEqual(@as(u64, 77), engine.layers[1].stats.compute_ns);

    for (1..3) |layer_usize| {
        const layer: u16 = @intCast(layer_usize);
        try t.expectEqual(@as(u64, warm_count), engine.layers[layer].cache.tick);
        for (0..warm_count) |expert_usize| {
            const expert: u16 = @intCast(expert_usize);
            const raw_slot = engine.layers[layer].cache.expert_to_slot[expert];
            try t.expect(raw_slot >= 0);
            const slot: u16 = @intCast(raw_slot);
            try t.expect(engine.slotReady(layer, slot));
            for ([_]Component{ .gate_up, .down }) |component| {
                const source = engine.store.span(layer, expert, component);
                const want = fixture.raw[@intCast(source.offset)..][0..@intCast(source.len)];
                try t.expectEqualSlices(u8, want, engine.cacheSlotBytes(layer, slot, component));
            }
        }
        try t.expectEqual(std.math.maxInt(u16), engine.layers[layer].cache.slot_to_expert[4]);
        try t.expect(!engine.layers[layer].cache.ready[4]);
    }

    const fills = engine.fill_experts_total;
    const fill_bytes = engine.fill_bytes_total;
    var hit = try engine.prepareHost(1, &.{ 0, 1, 2, 3 });
    hit.deinit();
    try engine.warmCache();
    try t.expectEqual(fills, engine.fill_experts_total);
    try t.expectEqual(fill_bytes, engine.fill_bytes_total);
}

test "expert stream warm cache leaves an active layer reusable after a fill error" {
    const t = std.testing;
    var fixture = try TinyLayersEngine.open(5, 3, 1, 8);
    defer fixture.close();
    const engine = &fixture.engine;
    const bad_index = (1 * @as(usize, 8) + 0) * 2 + @backingInt(Component.down);
    const good = engine.store.spans[bad_index];
    engine.store.spans[bad_index] = .{ .file = good.file, .offset = 1 << 40, .len = good.len };
    try t.expectError(error.FillSpanPastEof, engine.warmCache());
    engine.store.spans[bad_index] = good;

    try t.expectEqual(@as(u64, 0), engine.fill_experts_total);
    for (engine.layers[1].cache.ready) |ready| try t.expect(!ready);
    for (engine.layers[1].cache.expert_to_slot) |slot| try t.expectEqual(@as(i32, -1), slot);

    try engine.warmCache();
    try t.expectEqual(@as(u64, 8), engine.fill_experts_total);
}

test "expert stream a failed component fill leaves no ready slot" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    const down_index = (0 * @as(usize, 4) + 1) * 2 + 1;
    const good = engine.store.spans[down_index];
    engine.store.spans[down_index] = .{ .file = good.file, .offset = 1 << 40, .len = good.len };
    try t.expectError(error.FillSpanPastEof, engine.prepareHost(0, &.{1}));
    try t.expect(!fixture.anyReady());
    try t.expectEqual(@as(i32, -1), engine.layers[0].cache.expert_to_slot[1]);

    engine.store.spans[down_index] = good;
    var recovered = try engine.prepareHost(0, &.{1});
    defer recovered.deinit();
    const slot = recovered.remapped[0];
    try t.expect(engine.slotReady(0, slot));
    try t.expectEqualSlices(u8, fixture.raw[@intCast(good.offset)..][0..@intCast(good.len)], engine.cacheSlotBytes(0, slot, .down));
}

test "expert stream a failed fill never serves the expert whose slot it took" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    var warm = try engine.prepareHost(0, &.{ 0, 1 });
    warm.deinit();
    try t.expectEqual(@as(u64, 2), engine.fill_experts_total);
    const victim_slot: u16 = @intCast(engine.layers[0].cache.expert_to_slot[0]);
    try t.expect(engine.slotReady(0, victim_slot));

    const down_index = (0 * @as(usize, 4) + 2) * 2 + 1;
    const good = engine.store.spans[down_index];
    engine.store.spans[down_index] = .{ .file = good.file, .offset = 1 << 40, .len = good.len };
    try t.expectError(error.FillSpanPastEof, engine.prepareHost(0, &.{2}));
    engine.store.spans[down_index] = good;

    try t.expect(!engine.slotReady(0, victim_slot));
    try t.expectEqual(@as(i32, -1), engine.layers[0].cache.expert_to_slot[0]);

    var again = try engine.prepareHost(0, &.{0});
    defer again.deinit();
    try t.expectEqual(@as(u64, 3), engine.fill_experts_total);
    const gate_span = engine.store.span(0, 0, .gate_up);
    try t.expectEqualSlices(u8, fixture.raw[@intCast(gate_span.offset)..][0..@intCast(gate_span.len)], engine.cacheSlotBytes(0, again.remapped[0], .gate_up));
}

test "expert stream never serves a hit from a slab that is still filling" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    var warm = try engine.prepareHost(0, &.{1});
    const slot = warm.remapped[0];
    warm.deinit();
    try t.expectEqual(@as(u64, 1), engine.fill_experts_total);

    engine.layers[0].cache.ready[slot] = false;
    var refilled = try engine.prepareHost(0, &.{1});
    try t.expectEqual(@as(u64, 2), engine.fill_experts_total);
    refilled.deinit();
    engine.drainPendingReaders();

    _ = try engine.layers[0].slabs[0].slab.?.tryBeginFill();
    try t.expectError(error.SlabFilling, engine.prepareHost(0, &.{2}));
    abortFill(&engine.layers[0].slabs[0]);
}

test "expert stream a failure after the first lease leaves every slab reusable" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    const real = engine.layers[0].slabs[1].slab.?.allocator;
    engine.layers[0].slabs[1].slab.?.allocator = failing.allocator();
    try t.expectError(error.OutOfMemory, engine.prepareHost(0, &.{1}));
    engine.layers[0].slabs[1].slab.?.allocator = real;

    var after = try engine.prepareHost(0, &.{2});
    defer after.deinit();
    const slot = after.remapped[0];
    const gate_span = engine.store.span(0, 2, .gate_up);
    try t.expectEqualSlices(u8, fixture.raw[@intCast(gate_span.offset)..][0..@intCast(gate_span.len)], engine.cacheSlotBytes(0, slot, .gate_up));
}

test "expert stream union workspace refuses reuse while the previous reader holds it" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    var held = try engine.prepareHost(0, &.{ 0, 1, 2 });
    defer held.deinit();
    try t.expect(held.workspace);
    try t.expectError(error.SlabLeased, engine.prepareHost(0, &.{ 0, 1, 3 }));
}

test "expert stream a cancelled fill drains before the slab is reused" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    const canceller = try std.Thread.spawn(.{}, struct {
        fn call(e: *Engine) void {
            e.cancelFills();
        }
    }.call, .{engine});
    const outcome = engine.prepareHost(0, &.{ 0, 1, 2 });
    canceller.join();
    if (outcome) |ok| {
        var prepared = ok;
        defer prepared.deinit();
    } else |err| {
        try t.expect(err == error.FillCancelled);
        try t.expect(!fixture.anyReady());
    }

    var after = try engine.prepareHost(0, &.{3});
    defer after.deinit();
    const slot = after.remapped[0];
    const gate_span = engine.store.span(0, 3, .gate_up);
    try t.expectEqualSlices(u8, fixture.raw[@intCast(gate_span.offset)..][0..@intCast(gate_span.len)], engine.cacheSlotBytes(0, slot, .gate_up));
}

test "expert stream prefill union copies hits and fills only the misses" {
    const t = std.testing;
    var fixture = try TinyEngine.open(4, 8);
    defer fixture.close();
    const engine = &fixture.engine;

    var warm = try engine.prepareHost(0, &.{ 0, 1, 2 });
    try t.expect(!warm.workspace);
    warm.deinit();
    try t.expectEqual(@as(u64, 3), engine.fill_experts_total);

    var chunk = try engine.prepareHost(0, &.{ 0, 1, 2, 3, 4, 4 });
    defer chunk.deinit();
    try t.expect(chunk.workspace);
    try t.expectEqual(@as(u64, 5), engine.fill_experts_total);
    try t.expectEqualSlices(u16, &.{ 0, 1, 2, 3, 4, 4 }, chunk.remapped);

    for ([_]u16{ 0, 1, 2 }) |expert| {
        const slot: u16 = @intCast(engine.layers[0].cache.expert_to_slot[expert]);
        try t.expectEqualSlices(u8, engine.cacheSlotBytes(0, slot, .gate_up), engine.unionSlotBytes(expert, .gate_up));
        try t.expectEqualSlices(u8, engine.cacheSlotBytes(0, slot, .down), engine.unionSlotBytes(expert, .down));
    }
    for ([_]u16{ 3, 4 }) |expert| {
        const source = engine.store.span(0, expert, .gate_up);
        try t.expectEqualSlices(u8, fixture.raw[@intCast(source.offset)..][0..@intCast(source.len)], engine.unionSlotBytes(expert, .gate_up));
    }

    try t.expect(engine.layers[0].cache.expert_to_slot[4] >= 0);
    try t.expectEqual(@as(i32, -1), engine.layers[0].cache.expert_to_slot[3]);
    const admitted: u16 = @intCast(engine.layers[0].cache.expert_to_slot[4]);
    try t.expect(engine.slotReady(0, admitted));
    try t.expectEqualSlices(u8, engine.unionSlotBytes(4, .gate_up), engine.cacheSlotBytes(0, admitted, .gate_up));
}

fn bf16Of(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

test "expert stream bf16 kernels read the imported slab operand" {
    const t = std.testing;
    if (std.c.getenv("CODEX_SANDBOX") != null) return error.SkipZigTest;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;
    const s = engine.s;

    var prepared = try engine.prepareHost(0, &.{ 1, 2, 2, 1 });
    defer prepared.deinit();
    try t.expect(!prepared.workspace);

    var x_bits: [128]u16 = undefined;
    for (&x_bits, 0..) |*v, i| v.* = 0x3f00 +% @as(u16, @intCast(i % 97));
    const x2 = mlx.mlx_array_new_data(&x_bits, &[_]c_int{ 2, 64 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x2);
    const x5 = mlx.mlx_array_new_data(&x_bits, &[_]c_int{ 1, 2, 1, 1, 64 }, 5, .bfloat16);
    defer _ = mlx.mlx_array_free(x5);
    var ids: [4]i32 = undefined;
    for (prepared.remapped, 0..) |slot, i| ids[i] = slot;
    const slots2 = mlx.mlx_array_new_data(&ids, &[_]c_int{ 2, 2 }, 2, .int32);
    defer _ = mlx.mlx_array_free(slots2);
    const ids3 = mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, 2, 2 }, 3, .int32);
    defer _ = mlx.mlx_array_free(ids3);
    const weights: [4]f32 = .{ 0.75, 0.25, 0.4, 0.6 };
    const w2 = mlx.mlx_array_new_data(&weights, &[_]c_int{ 2, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(w2);

    const h = try kernels.gateUpSwiglu(s, x2, prepared.raw_gate_up, slots2);
    defer _ = mlx.mlx_array_free(h);
    const y = try kernels.downReduce(s, h, prepared.raw_down, slots2, w2);
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));

    const no_idx = mlx.mlx_array{ .ctx = null };
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_gather_mm(&gate, x5, prepared.gate, no_idx, ids3, false, s));
    var up = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up);
    try mlx.check(mlx.mlx_gather_mm(&up, x5, prepared.up, no_idx, ids3, false, s));
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_sigmoid(&act, gate, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, act, gate, s));
    var hidden_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden_ref);
    try mlx.check(mlx.mlx_multiply(&hidden_ref, silu, up, s));
    var down_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(down_ref);
    try mlx.check(mlx.mlx_gather_mm(&down_ref, hidden_ref, prepared.down, no_idx, ids3, false, s));
    var squeezed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(squeezed);
    try mlx.check(mlx.mlx_squeeze(&squeezed, down_ref, s));
    const wexp = mlx.mlx_array_new_data(&weights, &[_]c_int{ 2, 2, 1 }, 3, .float32);
    defer _ = mlx.mlx_array_free(wexp);
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, squeezed, wexp, s));
    var reference = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reference);
    try mlx.check(mlx.mlx_sum_axis(&reference, weighted, -2, false, s));
    try mlx.check(mlx.mlx_array_eval(reference));

    const got = mlx.mlx_array_data_bfloat16(y) orelse return error.KernelOutputUnreadable;
    var ref_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_f32);
    try mlx.check(mlx.mlx_astype(&ref_f32, reference, .float32, s));
    try mlx.check(mlx.mlx_array_eval(ref_f32));
    const want = mlx.mlx_array_data_float32(ref_f32) orelse return error.KernelOutputUnreadable;
    var num: f64 = 0;
    var den: f64 = 0;
    for (0..128) |i| {
        const diff = @as(f64, bf16Of(got[i])) - @as(f64, want[i]);
        num += diff * diff;
        den += @as(f64, want[i]) * @as(f64, want[i]);
    }
    try t.expect(den > 0);
    const rel = @sqrt(num / den);
    try t.expect(rel < 0.01);
}

test "expert stream route detail buckets by attention class and probes off the print cadence" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    engine.beginForward();
    try t.expect(engine.routeProbeActive());
    engine.noteRoute(0, false, .{ .build_ns = 1, .x_wait_ns = 2, .ids_wait_ns = 3, .read_ns = 4 });
    engine.noteRoute(0, true, .{ .build_ns = 10, .x_wait_ns = 20, .ids_wait_ns = 30, .read_ns = 40 });
    engine.finishForward(1);
    const first = engine.breakdown();
    try t.expectEqual(@as(u32, 1), first.route_linear.layers);
    try t.expectEqual(@as(u64, 3), first.route_linear.ids_wait_ns);
    try t.expectEqual(@as(u64, 2), first.route_linear.x_wait_ns);
    try t.expectEqual(@as(u32, 1), first.route_full.layers);
    try t.expectEqual(@as(u64, 30), first.route_full.ids_wait_ns);
    try t.expectEqual(@as(u64, 110), first.route_ns);
    try t.expect(first.probe);

    engine.beginForward();
    engine.noteRoute(0, false, .{ .ids_wait_ns = 7 });
    engine.finishForward(1);
    const second = engine.breakdown();
    try t.expectEqual(@as(u64, 7), second.route_linear.ids_wait_ns);
    try t.expectEqual(@as(u32, 0), second.route_full.layers);

    engine.forward_count = BREAKDOWN_EVERY - 1;
    try t.expect(!engine.routeProbeActive());
    engine.forward_count = BREAKDOWN_EVERY + BREAKDOWN_EVERY / 2 - 1;
    try t.expect(engine.routeProbeActive());
}

test "expert stream a streamed reader holds its slab lease until the next route readback" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;
    const slab = engine.layers[0].slabs[0].slab.?;

    var first = try engine.prepareHost(0, &.{1});
    first.deinit();
    try t.expectEqual(@as(usize, 2), engine.pendingReaders());
    try t.expectEqual(@as(usize, 1), slab.readerCount());
    try t.expectError(error.SlabLeased, slab.tryBeginFill());

    var second = try engine.prepareHost(0, &.{2});
    try t.expectEqual(@as(usize, 0), engine.pendingReaders());
    try t.expectEqual(@as(usize, 1), slab.readerCount());
    second.deinit();
    try t.expectEqual(@as(usize, 2), engine.pendingReaders());
    engine.drainPendingReaders();
    try t.expectEqual(@as(usize, 0), engine.pendingReaders());
    try t.expectEqual(@as(usize, 0), slab.readerCount());
}

pub const TinyQuantPack = struct {
    allocator: std.mem.Allocator,
    file: []u8,
    data: [quant.component_count][]align(4) u8,
    rows: [quant.component_count]u32,
    cols: [quant.component_count]u32,

    pub fn dtypeOf(ci: usize) mlx.mlx_dtype {
        return if (quant.partOf(@fromBackingInt(@intCast(ci))) == .weight) .uint32 else .bfloat16;
    }

    pub fn residentBank(self: *const TinyQuantPack, experts: u16, ci: usize) mlx.mlx_array {
        const shape = [_]c_int{ @intCast(experts), @intCast(self.rows[ci]), @intCast(self.cols[ci]) };
        return mlx.mlx_array_new_data(self.data[ci].ptr, &shape, 3, dtypeOf(ci));
    }

    pub fn deinit(self: *TinyQuantPack) void {
        for (self.data) |slice| self.allocator.free(slice);
        self.allocator.free(self.file);
        self.* = undefined;
    }
};

pub fn writeTinyQuantCheckpoint(
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    seed: u64,
    experts: u16,
    hidden: u32,
    inter: u32,
    group_size: u32,
    bits: [3]u32,
) !TinyQuantPack {
    const io = std.testing.io;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var pack = TinyQuantPack{
        .allocator = allocator,
        .file = &.{},
        .data = @splat(&.{}),
        .rows = @splat(0),
        .cols = @splat(0),
    };
    errdefer {
        for (pack.data) |slice| if (slice.len != 0) allocator.free(slice);
    }
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(allocator);
    try header.append(allocator, '{');
    try index.appendSlice(allocator, "{\"weight_map\":{");
    var offset: u64 = 0;
    var key_buf: [192]u8 = undefined;
    for (0..quant.component_count) |ci| {
        const c: quant.Component = @fromBackingInt(@intCast(ci));
        const projection = quant.projectionOf(c);
        const in_dim: u32 = if (projection == .down) inter else hidden;
        const is_weight = quant.partOf(c) == .weight;
        pack.rows[ci] = if (projection == .down) hidden else inter;
        pack.cols[ci] = if (is_weight) in_dim * bits[@backingInt(projection)] / 32 else in_dim / group_size;
        const elem: u64 = if (is_weight) 4 else 2;
        const span_bytes: usize = @intCast(@as(u64, experts) * pack.rows[ci] * pack.cols[ci] * elem);
        const buffer = try allocator.alignedAlloc(u8, .@"4", span_bytes);
        pack.data[ci] = buffer;
        if (is_weight) {
            const words: []u32 = @alignCast(std.mem.bytesAsSlice(u32, buffer));
            for (words) |*word| word.* = rnd.int(u32);
        } else {
            const halves: []u16 = @alignCast(std.mem.bytesAsSlice(u16, buffer));
            const wanted = quant.partOf(c) == .scales;
            for (halves) |*half| {
                const value: f32 = if (wanted) 0.002 + rnd.float(f32) * 0.01 else -0.05 + rnd.float(f32) * 0.1;
                half.* = @truncate(@as(u32, @bitCast(value)) >> 16);
            }
        }
        const key = try quant.tensorKey(&key_buf, 0, c);
        const sep: []const u8 = if (ci == 0) "" else ",";
        const entry = try std.fmt.allocPrint(allocator, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d},{d}],\"data_offsets\":[{d},{d}]}}", .{
            sep, key, if (is_weight) "U32" else "BF16", experts, pack.rows[ci], pack.cols[ci], offset, offset + span_bytes,
        });
        defer allocator.free(entry);
        try header.appendSlice(allocator, entry);
        const mapping = try std.fmt.allocPrint(allocator, "{s}\"{s}\":\"experts.safetensors\"", .{ sep, key });
        defer allocator.free(mapping);
        try index.appendSlice(allocator, mapping);
        offset += span_bytes;
    }
    try header.append(allocator, '}');
    try index.appendSlice(allocator, "}}");
    const file_bytes = try allocator.alloc(u8, 8 + header.items.len + @as(usize, @intCast(offset)));
    errdefer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.items.len, .little);
    @memcpy(file_bytes[8..][0..header.items.len], header.items);
    var at: usize = 8 + header.items.len;
    for (pack.data) |slice| {
        @memcpy(file_bytes[at..][0..slice.len], slice);
        at += slice.len;
    }
    try dir.writeFile(io, .{ .sub_path = "experts.safetensors", .data = file_bytes });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    pack.file = file_bytes;
    return pack;
}

pub fn writeTinyMxfp4Checkpoint(
    allocator: std.mem.Allocator,
    dir: std.Io.Dir,
    experts: u16,
    hidden: u32,
    inter: u32,
) ![]u8 {
    const io = std.testing.io;
    const Plan = struct {
        offset: u64,
        len: u64,
        scale: bool,
    };
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(allocator);
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(allocator);
    try header.append(allocator, '{');
    try index.appendSlice(allocator, "{\"weight_map\":{");
    var plans: [6]Plan = undefined;
    var plan_count: usize = 0;
    var offset: u64 = 0;
    var key_buf: [192]u8 = undefined;
    for ([_]quant.Projection{ .gate, .up, .down }) |projection| {
        for ([_]quant.Part{ .weight, .scales }) |part| {
            const key = try quant.mxfp4TensorKey(&key_buf, 1, projection, part);
            const in_dim: u64 = if (projection == .down) inter else hidden;
            const rows: u64 = if (projection == .down) hidden else inter;
            const cols: u64 = if (part == .weight) in_dim / 8 else in_dim / 32;
            const elem: u64 = if (part == .weight) 4 else 1;
            const len = @as(u64, experts) * rows * cols * elem;
            const sep: []const u8 = if (plan_count == 0) "" else ",";
            const entry = try std.fmt.allocPrint(allocator, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d},{d}],\"data_offsets\":[{d},{d}]}}", .{
                sep,
                key,
                if (part == .weight) "U32" else "U8",
                experts,
                rows,
                cols,
                offset,
                offset + len,
            });
            defer allocator.free(entry);
            try header.appendSlice(allocator, entry);
            const mapping = try std.fmt.allocPrint(allocator, "{s}\"{s}\":\"mxfp4.safetensors\"", .{ sep, key });
            defer allocator.free(mapping);
            try index.appendSlice(allocator, mapping);
            plans[plan_count] = .{ .offset = offset, .len = len, .scale = part == .scales };
            plan_count += 1;
            offset += len;
        }
    }
    try header.append(allocator, '}');
    try index.appendSlice(allocator, "}}");

    const file_bytes = try allocator.alloc(u8, 8 + header.items.len + @as(usize, @intCast(offset)));
    errdefer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.items.len, .little);
    @memcpy(file_bytes[8..][0..header.items.len], header.items);
    const payload = file_bytes[8 + header.items.len ..];
    for (plans[0..plan_count], 0..) |plan, component| {
        const out = payload[@intCast(plan.offset)..][0..@intCast(plan.len)];
        if (plan.scale) {
            const codes = [_]u8{ 0, 125, 126, 127, 128, 129, 254, 255 };
            for (out, 0..) |*byte, i| byte.* = codes[(i + component) % codes.len];
        } else {
            for (out, 0..) |*byte, i| byte.* = @truncate(i * 17 + component * 29);
        }
    }
    try dir.writeFile(io, .{ .sub_path = "mxfp4.safetensors", .data = file_bytes });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    return file_bytes;
}

test "expert stream MXFP4 header spans and fills preserve native bytes" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const raw = try writeTinyMxfp4Checkpoint(t.allocator, tmp.dir, 2, 32, 32);
    defer t.allocator.free(raw);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const geometry = Geometry{
        .layers = 2,
        .experts = 2,
        .hidden = 32,
        .intermediate = 32,
        .first_moe_layer = 1,
    };
    var store = try ExpertStore.openLayout(t.allocator, path_buf[0..path_len], geometry, .mxfp4_split);
    defer store.deinit();

    try t.expectEqual(quant.Layout.mxfp4_split, store.layout());
    try t.expect(!store.hasExpertLayer(0));
    try t.expect(store.hasExpertLayer(1));
    try t.expectEqual(@as(usize, quant.component_count), store.componentCount());
    for (0..quant.component_count) |ci| {
        const c: quant.Component = @fromBackingInt(@intCast(ci));
        try t.expectEqual(@as(u64, 0), store.spanAt(0, 0, ci).len);
        if (quant.partOf(c) == .biases) {
            try t.expect(!store.componentPresent(ci));
            try t.expectEqual(@as(u64, 0), store.slabSpec(ci).slot_bytes);
            try t.expectEqual(@as(u8, 0), store.slabSpec(ci).elem_bytes);
        } else if (quant.partOf(c) == .scales) {
            try t.expectEqual(mlx.mlx_dtype.uint8, store.slabSpec(ci).dtype);
            try t.expectEqual(@as(u8, 1), store.slabSpec(ci).elem_bytes);
        }
    }

    var destination = try t.allocator.alloc(u8, @intCast(store.perExpertBytes()));
    defer t.allocator.free(destination);
    var fills: std.ArrayList(io_mod.FillSpan) = .empty;
    defer fills.deinit(t.allocator);
    var at: usize = 0;
    for (0..quant.component_count) |ci| {
        if (!store.componentPresent(ci)) continue;
        const source = store.spanAt(1, 1, ci);
        const len: usize = @intCast(source.len);
        try fills.append(t.allocator, .{ .file = source.file, .offset = source.offset, .len = source.len, .dst = destination[at..][0..len].ptr });
        at += len;
    }
    try t.expectEqual(destination.len, at);

    const fds = try t.allocator.alloc(std.c.fd_t, store.fileCount());
    defer t.allocator.free(fds);
    for (fds, 0..) |*fd, i| fd.* = store.fdAt(i);
    var fill_plan = try io_mod.FillPlan.init(t.allocator, fills.items, 64 * 1024 * 1024, 1, false);
    defer fill_plan.deinit();
    const pool = try io_mod.FillPool.create(t.allocator, .{ .workers = 1 });
    defer pool.destroy();
    try pool.run(fds, &fill_plan);

    at = 0;
    for (0..quant.component_count) |ci| {
        if (!store.componentPresent(ci)) continue;
        const source = store.spanAt(1, 1, ci);
        const len: usize = @intCast(source.len);
        try t.expectEqualSlices(u8, raw[@intCast(source.offset)..][0..len], destination[at..][0..len]);
        at += len;
    }
}

test "expert stream original MiMo source warms exact MXFP4 bytes" {
    const t = std.testing;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const raw = try quant.writeTinyMxfp4IndividualCheckpoint(t.allocator, tmp.dir, 2, 32, 32);
    defer t.allocator.free(raw);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(t.io, &path_buf);
    const geometry = Geometry{ .layers = 2, .experts = 2, .hidden = 32, .intermediate = 32, .first_moe_layer = 1 };
    const per_expert = try mxfp4ExpertBytes(32, 32);
    var engine = try Engine.initWithOptions(t.allocator, path_buf[0..n], geometry, 2 * per_expert, tinyStream(), .{
        .io_workers = 1,
        .bounce_size = 1 << 20,
        .layout = .mxfp4_individual,
    });
    defer engine.deinit();
    try engine.warmCache();
    try t.expectEqual(@as(u64, 1), engine.fill_experts_total);
    try t.expectEqual(@as(u64, 12), engine.slab_imports);
    try t.expectEqual(@as(usize, 0), engine.layers[0].slabs.len);
    var prepared = try engine.prepareHost(1, &.{0});
    defer prepared.deinit();
    try t.expectEqual(@as(u64, 1), engine.fill_experts_total);
    for (0..quant.component_count) |ci| {
        if (!engine.store.componentPresent(ci)) continue;
        const source = engine.store.spanAt(1, 0, ci);
        try t.expectEqualSlices(u8, raw[@intCast(source.offset)..][0..@intCast(source.len)], engine.cacheSlotBytesAt(1, prepared.remapped[0], ci));
    }
}

test "expert stream MXFP4 keeps nine component ids but imports and leases only six" {
    const t = std.testing;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const raw = try writeTinyMxfp4Checkpoint(t.allocator, tmp.dir, 2, 32, 32);
    defer t.allocator.free(raw);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const geometry = Geometry{
        .layers = 2,
        .experts = 2,
        .hidden = 32,
        .intermediate = 32,
        .first_moe_layer = 1,
    };
    const per_expert = try expertBytesFor(t.allocator, path_buf[0..path_len], geometry, .mxfp4_split);
    var engine = try Engine.initWithOptions(t.allocator, path_buf[0..path_len], geometry, 2 * per_expert, tinyStream(), .{
        .io_workers = 1,
        .bounce_size = 1 << 20,
        .layout = .mxfp4_split,
    });
    defer engine.deinit();

    try t.expectEqual(@as(u64, 12), engine.slab_imports);
    try t.expectEqual(@as(usize, 0), engine.layers[0].slabs.len);
    try t.expectEqual(@as(usize, quant.component_count), engine.layers[1].slabs.len);
    try t.expectError(error.ExpertLayerAbsent, engine.prepareHost(0, &.{1}));

    try engine.warmCache();
    try t.expectEqual(@as(u64, 1), engine.fill_experts_total);
    try t.expect(engine.slotReady(1, 0));
    try t.expect(!engine.slotReady(1, 1));
    var warm_hit = try engine.prepareHost(1, &.{0});
    warm_hit.deinit();
    try t.expectEqual(@as(u64, 1), engine.fill_experts_total);
    for (0..quant.component_count) |ci| {
        if (!engine.store.componentPresent(ci)) continue;
        const source = engine.store.spanAt(1, 0, ci);
        try t.expectEqualSlices(u8, raw[@intCast(source.offset)..][0..@intCast(source.len)], engine.cacheSlotBytesAt(1, 0, ci));
    }

    var prepared = try engine.prepareHost(1, &.{1});
    const slot = prepared.remapped[0];
    try t.expect(prepared.quantized);
    for (0..quant.component_count) |ci| {
        const c: quant.Component = @fromBackingInt(@intCast(ci));
        const operand = prepared.quantOperand(c);
        if (quant.partOf(c) == .biases) {
            try t.expect(operand.ctx == null);
            try t.expect(!engine.layers[1].slabs[ci].present);
        } else {
            try t.expect(operand.ctx != null);
            try t.expect(engine.layers[1].slabs[ci].present);
        }
    }
    prepared.deinit();
    try t.expectEqual(@as(usize, 6), engine.pendingReaders());
    engine.drainPendingReaders();
    for (0..quant.component_count) |ci| {
        if (!engine.store.componentPresent(ci)) continue;
        const source = engine.store.spanAt(1, 1, ci);
        try t.expectEqualSlices(
            u8,
            raw[@intCast(source.offset)..][0..@intCast(source.len)],
            engine.cacheSlotBytesAt(1, slot, ci),
        );
    }
}

test "expert stream quantized slabs alias the nine pack tensors and remap ids" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var pack = try writeTinyQuantCheckpoint(t.allocator, tmp.dir, 0x51ab, 4, 64, 64, 32, .{ 4, 8, 4 });
    defer pack.deinit();
    const raw = pack.file;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const path = path_buf[0..path_len];
    const geom = Geometry{ .layers = 1, .experts = 4, .hidden = 64, .intermediate = 64 };
    const s = tinyStream();

    const per_expert = try expertBytesFor(t.allocator, path, geom, .quantized_split);
    try t.expectEqual(@as(u64, 2 * (64 * 8 * 4 + 2 * 64 * 2 * 2) + (64 * 16 * 4 + 2 * 64 * 2 * 2)), per_expert);

    var engine = try Engine.initWithOptions(t.allocator, path, geom, 2 * per_expert, s, .{
        .bounce_size = 1 << 20,
        .layout = .quantized_split,
    });
    defer engine.deinit();
    try t.expectEqual(@as(u64, 0), engine.fallback_imports);
    try t.expectEqual(@as(u64, quant.component_count * (@as(usize, geom.layers) + 1)), engine.slab_imports);

    var prepared = try engine.prepareHost(0, &.{ 0, 2 });
    try t.expect(prepared.quantized);
    try t.expect(prepared.gate.ctx == null);
    try t.expectEqual(@as(u64, 2), engine.fill_experts_total);
    for (0..quant.component_count) |ci| {
        const operand = prepared.quantOperand(@fromBackingInt(@intCast(ci)));
        try t.expect(operand.ctx != null);
        const shape = mlx.getShape(operand);
        try t.expectEqual(@as(usize, 3), shape.len);
        try t.expectEqual(@as(c_int, 2), shape[0]);
        const spec = engine.store.slabSpec(ci);
        try t.expectEqual(@as(c_int, @intCast(spec.rows)), shape[1]);
        try t.expectEqual(@as(c_int, @intCast(spec.cols)), shape[2]);
        try t.expectEqual(spec.dtype, mlx.mlx_array_dtype(operand));
    }
    for ([_]u16{ 0, 2 }, 0..) |expert, i| {
        const slot = prepared.remapped[i];
        for (0..quant.component_count) |ci| {
            const source = engine.store.spanAt(0, expert, ci);
            try t.expectEqualSlices(
                u8,
                raw[@intCast(source.offset)..][0..@intCast(source.len)],
                engine.cacheSlotBytesAt(0, slot, ci),
            );
        }
    }

    var handles: [quant.component_count]?*anyopaque = undefined;
    for (&handles, 0..) |*handle, ci| handle.* = prepared.quantOperand(@fromBackingInt(@intCast(ci))).ctx;
    prepared.deinit();
    try t.expectEqual(quant.component_count, engine.pendingReaders());
    engine.beginForward();
    var again = try engine.prepareHost(0, &.{ 0, 2 });
    defer again.deinit();
    try t.expectEqual(@as(u64, 2), engine.fill_experts_total);
    for (handles, 0..) |handle, ci| try t.expectEqual(handle, again.quantOperand(@fromBackingInt(@intCast(ci))).ctx);
}

test "expert stream: under streaming an MTP on by flag refuses, a settings mtp is dropped, else off" {
    const t = std.testing;
    try t.expectEqual(MtpUnderStreaming.keep, mtpUnderStreaming(true, false, false));
    try t.expectEqual(MtpUnderStreaming.drop_settings, mtpUnderStreaming(true, true, false));
    try t.expectEqual(MtpUnderStreaming.off, mtpUnderStreaming(false, true, false));
    try t.expectEqual(MtpUnderStreaming.off, mtpUnderStreaming(false, false, false));
    // The engine default never refuses a streamed load: it resolves off.
    try t.expectEqual(MtpUnderStreaming.drop_default, mtpUnderStreaming(true, false, true));
}

test "exl3 Sushi CPU expert bytes at K1 and K8 include all three scale pairs" {
    for ([_]u32{ 16, 128 }) |n| {
        const got = try exl3ExpertBytes(.{ .layers = 48, .experts = 256, .hidden = 4096, .intermediate = 2048, .exl3_n = n });
        const packed_bytes: u64 = if (n == 16) 3 * 1048576 else 3 * 8388608;
        try std.testing.expectEqual(packed_bytes + 3 * (4096 + 2048) * 2, got);
    }
}

test "a uniform EXL3 layout reaches its shards: only grouped or pruned packs refuse streaming" {
    const t = std.testing;
    const opened = ExpertStore.openLayout(t.allocator, "/missing-sushi-coder-pack", .{ .layers = 2, .experts = 4, .hidden = 128, .intermediate = 128 }, .exl3_k4);
    if (opened) |store| {
        var s = store;
        s.deinit();
        return error.TestUnexpectedResult;
    } else |err| try t.expect(err != error.Exl3RaggedStreamingUnsupported and err != error.Exl3RateGroupsStreamingUnsupported);
}

fn writeMimoExl3StreamFixture(dir: std.Io.Dir) !void {
    const a = std.testing.allocator;
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(a);
    try header.append(a, '{');
    try index.appendSlice(a, "{\"weight_map\":{");
    var offset: usize = 0;
    for (1..3) |layer| {
        for (0..9) |ci| {
            const c: quant.Component = @fromBackingInt(@intCast(ci));
            const weight = ci % 3 == 0;
            const n: usize = if (layer == 1) 36 else 64;
            const bytes: usize = 4 * (if (weight) 8 * 8 * n * 2 else 128 * 2);
            const key = try std.fmt.allocPrint(a, "model.layers.{d}.mlp.switch_mlp.{s}_proj.{s}", .{ layer, @tagName(quant.projectionOf(c)), if (weight) "trellis" else if (ci % 3 == 1) "suh" else "svh" });
            defer a.free(key);
            const shape = if (weight) try std.fmt.allocPrint(a, "4,8,8,{d}", .{n}) else try a.dupe(u8, "4,128");
            defer a.free(shape);
            const sep = if (offset == 0) "" else ",";
            const item = try std.fmt.allocPrint(a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{s}],\"data_offsets\":[{d},{d}]}}", .{ sep, key, if (weight) "U16" else "F16", shape, offset, offset + bytes });
            defer a.free(item);
            try header.appendSlice(a, item);
            const entry = try std.fmt.allocPrint(a, "{s}\"{s}\":\"experts.safetensors\"", .{ sep, key });
            defer a.free(entry);
            try index.appendSlice(a, entry);
            offset += bytes;
        }
    }
    try header.append(a, '}');
    try index.appendSlice(a, "}}");
    const file = try a.alloc(u8, 8 + header.items.len + offset);
    defer a.free(file);
    std.mem.writeInt(u64, file[0..8], header.items.len, .little);
    @memcpy(file[8..][0..header.items.len], header.items);
    for (file[8 + header.items.len ..], 0..) |*byte, i| byte.* = @truncate(i *% 13 +% i / 251);
    try dir.writeFile(std.testing.io, .{ .sub_path = "experts.safetensors", .data = file });
    try dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}

fn checkMimoExl3Streaming(runtime: bool) !void {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try writeMimoExl3StreamFixture(tmp.dir);
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const len = try tmp.dir.realPath(t.io, &buf);
    const path = buf[0..len];
    const geometry: Geometry = .{ .layers = 3, .experts = 4, .hidden = 128, .intermediate = 128, .first_moe_layer = 1, .exl3_n = 64 };
    try t.expectEqual(quant.Layout.exl3_k4, try quant.streamingLayoutOfDir(t.allocator, t.io, "mimo_v2", path, 3, 1));
    const per = try expertBytesFor(t.allocator, path, geometry, .exl3_k4);
    try t.expectEqual(try exl3ExpertBytes(geometry), per);
    const fixed = (1 << 20) + 4 * per + 2 * per + BOUNCE_BYTES;
    const ledger = try budgetLedger(fixed + 4 * per, 1 << 20, 0, moeLayerCount(geometry), 4, 2, per, BOUNCE_BYTES);
    try t.expectEqual(@as(u16, 2), ledger.slots_per_layer);
    try t.expectEqual(fixed + ledger.cache_bytes, ledger.budget_bytes);
    var store = try ExpertStore.openLayout(t.allocator, path, geometry, .exl3_k4);
    defer store.deinit();
    try t.expect(!store.hasExpertLayer(0));
    const source_file = try tmp.dir.readFileAlloc(t.io, "experts.safetensors", t.allocator, .limited(1 << 20));
    defer t.allocator.free(source_file);
    for (1..3) |layer| {
        var cache = try GroupCache.init(t.allocator, 2, 4);
        defer cache.deinit();
        const routes = [_]u16{ 3, 0, 3, 2, 1 };
        var resolved = try cache.resolve(t.allocator, &routes);
        defer resolved.deinit(t.allocator);
        for (routes, resolved.remapped) |expert, slot| try t.expectEqual(expert, resolved.union_ids[slot]);
        for (0..9) |ci| {
            const spec = store.slabSpecAt(@intCast(layer), ci);
            if (ci % 3 == 0) try t.expectEqual(@as(u32, if (layer == 1) 36 else 64), spec.packed_n);
            for (0..4) |expert| {
                const span_value = store.spanAt(@intCast(layer), @intCast(expert), ci);
                try t.expectEqual(spec.slot_bytes, span_value.len);
                const bytes = try t.allocator.alloc(u8, @intCast(span_value.len));
                defer t.allocator.free(bytes);
                try readExact(store.fdAt(span_value.file), bytes, span_value.offset);
                try t.expectEqualSlices(u8, source_file[@intCast(span_value.offset)..][0..bytes.len], bytes);
            }
        }
    }
    if (!runtime) return;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var engine = try Engine.initWithOptions(t.allocator, path, geometry, ledger.cache_bytes, s, .{ .layout = .exl3_k4, .bounce_size = 1 << 20 });
    defer engine.deinit();
    try t.expect(!engine.layers[0].active);
    try t.expectEqual(@as(usize, 0), engine.layers[0].slabs.len);
    try t.expectEqual(@as(u64, 0), engine.fallback_imports);
    try t.expectError(error.ExpertLayerAbsent, engine.prepareHost(0, &.{0}));
    try engine.warmCache();
    for ([_]u16{ 1, 2 }) |layer| {
        const before = engine.fill_bytes_total;
        var warm = try engine.prepareHost(layer, &.{0});
        warm.deinit();
        try t.expectEqual(before, engine.fill_bytes_total);
    }
    for ([_]u16{ 1, 2, 1 }) |layer| {
        for ([_][]const u16{ &.{ 3, 0, 3 }, &.{ 0, 2, 3, 1, 2 } }) |routes| {
            var prepared = try engine.prepareHost(layer, routes);
            defer prepared.deinit();
            try t.expectEqual(routes.len > 3, prepared.workspace);
            for (0..9) |ci| {
                const arr = prepared.quantOperand(@fromBackingInt(@intCast(ci)));
                const shape = mlx.getShape(arr);
                if (ci % 3 == 0) try t.expectEqual(@as(c_int, if (layer == 1) 36 else 64), shape[3]);
                try mlx.check(mlx.mlx_array_eval(arr));
                const data: [*]const u8 = if (ci % 3 == 0) @ptrCast(mlx.mlx_array_data_uint16(arr).?) else @ptrCast(mlx.mlx_array_data_float16(arr).?);
                for (routes, prepared.remapped) |expert, slot| {
                    const source = engine.store.spanAt(layer, expert, ci);
                    const bytes = try t.allocator.alloc(u8, @intCast(source.len));
                    defer t.allocator.free(bytes);
                    try readExact(engine.store.fdAt(source.file), bytes, source.offset);
                    const actual = if (prepared.workspace) engine.unionSlotBytesAt(slot, ci) else engine.cacheSlotBytesAt(layer, slot, ci);
                    try t.expectEqualSlices(u8, bytes, actual);
                    try t.expectEqualSlices(u8, bytes, data[@as(usize, slot) * bytes.len ..][0..bytes.len]);
                }
            }
        }
    }
    var allocated: u64 = 0;
    for (engine.layers) |layer| for (layer.slabs) |slab| {
        allocated += slab.slab.?.bytes.len;
    };
    for (engine.union_slabs) |slab| allocated += slab.slab.?.bytes.len;
    try t.expect(allocated <= ledger.cache_bytes + ledger.workspace_bytes + ledger.selected_bytes + ledger.bounce_bytes - 4 * (1 << 20));
}

test "MiMo EXL3 streaming CPU geometry ledger and slab ids cross the dense prefix" {
    try checkMimoExl3Streaming(false);
}

test "MiMo EXL3 streaming imported slabs warm and reuse different layer rates" {
    try checkMimoExl3Streaming(true);
}

test "expert pick tolerance accepts 0 through 0.6 and names a bad value" {
    const t = std.testing;
    try t.expectEqual(@as(f32, 0.0), try parsePickTolerance("0"));
    try t.expectEqual(@as(f32, 0.3), try parsePickTolerance("0.3"));
    try t.expectEqual(@as(f32, 0.6), try parsePickTolerance("0.6"));
    try t.expectError(error.InvalidPickTolerance, parsePickTolerance("0.61"));
    try t.expectError(error.InvalidPickTolerance, parsePickTolerance("1"));
    try t.expectError(error.InvalidPickTolerance, parsePickTolerance("1.5"));
    try t.expectError(error.InvalidPickTolerance, parsePickTolerance("-0.1"));
    try t.expectError(error.InvalidPickTolerance, parsePickTolerance("nan"));
    try t.expectError(error.InvalidPickTolerance, parsePickTolerance("nope"));
    try t.expectError(error.InvalidPickTolerance, parsePickTolerance(""));
}

fn PickScratch(comptime experts: usize) type {
    return struct {
        taken: [experts]bool = undefined,
        loading: [experts]bool = undefined,
        starved: [experts]u8 = @splat(0),

        fn run(self: *@This(), ids: []u16, logits: []const f32, cached: []const bool, tolerance: f32, k: usize) u32 {
            return substituteMisses(ids, logits, cached, tolerance, k, &self.taken, &self.loading, &self.starved);
        }
    };
}

test "expert substitution with zero tolerance changes nothing" {
    const t = std.testing;
    var sc = PickScratch(6){};
    const logits = [_]f32{ 3.0, 2.9, 2.8, 2.0, 1.9, -30.0 };
    var ids = [_]u16{ 0, 1, 2 };
    const cached = [_]bool{ true, true, false, true, true, true };
    try t.expectEqual(@as(u32, 0), sc.run(&ids, &logits, &cached, 0.0, 3));
    try t.expectEqualSlices(u16, &.{ 0, 1, 2 }, &ids);
}

test "expert substitution swaps a missed expert for the best cached one within tolerance" {
    const t = std.testing;
    var sc = PickScratch(6){};
    const logits = [_]f32{ 3.0, 2.9, 2.8, 2.7, 2.0, -30.0 };
    var ids = [_]u16{ 0, 1, 2 };
    const cached = [_]bool{ true, true, false, true, true, true };
    try t.expectEqual(@as(u32, 1), sc.run(&ids, &logits, &cached, 0.3, 3));
    try t.expectEqualSlices(u16, &.{ 0, 1, 3 }, &ids);
}

test "expert substitution keeps a miss whose cached alternatives fall below the tolerance" {
    const t = std.testing;
    var sc = PickScratch(6){};
    const logits = [_]f32{ 3.0, 2.9, 2.8, 0.5, 0.4, -30.0 };
    var ids = [_]u16{ 0, 1, 2 };
    const cached = [_]bool{ true, true, false, true, true, true };
    try t.expectEqual(@as(u32, 0), sc.run(&ids, &logits, &cached, 0.3, 3));
    try t.expectEqualSlices(u16, &.{ 0, 1, 2 }, &ids);
}

test "expert substitution never picks an uncached or already selected expert" {
    const t = std.testing;
    var sc = PickScratch(6){};
    const logits = [_]f32{ 3.0, 2.9, 2.8, 2.79, 2.78, -30.0 };
    var ids = [_]u16{ 0, 1, 2 };
    const cached = [_]bool{ true, true, false, false, true, true };
    try t.expectEqual(@as(u32, 1), sc.run(&ids, &logits, &cached, 0.5, 3));
    try t.expectEqualSlices(u16, &.{ 0, 1, 4 }, &ids);
}

test "expert substitution scores each row alone and gives two misses different replacements" {
    const t = std.testing;
    var sc = PickScratch(6){};
    const logits = [_]f32{
        3.0, 2.9, 2.8, 2.7, 2.6, -30.0,
        2.6, 2.7, 2.8, 2.9, 3.0, -30.0,
    };
    var ids = [_]u16{ 0, 1, 2, 4, 3, 2 };
    const cached = [_]bool{ true, false, false, true, true, true };
    try t.expectEqual(@as(u32, 3), sc.run(&ids, &logits, &cached, 0.5, 3));
    try t.expectEqualSlices(u16, &.{ 0, 3, 4, 4, 3, 0 }, &ids);
}

test "expert substitution lets the more probable miss take the better substitute" {
    const t = std.testing;
    var sc = PickScratch(6){};
    const logits = [_]f32{ 3.0, 2.5, 2.9, 2.8, 2.7, -30.0 };
    var ids = [_]u16{ 0, 1, 2 };
    const cached = [_]bool{ true, false, false, true, true, true };
    try t.expectEqual(@as(u32, 2), sc.run(&ids, &logits, &cached, 0.5, 3));
    try t.expectEqualSlices(u16, &.{ 0, 4, 3 }, &ids);
}

test "expert substitution survives non-finite logits and a trailing partial row" {
    const t = std.testing;
    var sc = PickScratch(6){};
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    const logits = [_]f32{ nan, nan, nan, nan, nan, nan, -inf, -inf, -inf, -inf, -inf, -inf };
    var ids = [_]u16{ 0, 1, 2, 3, 4 };
    const cached = [_]bool{ true, false, false, true, true, true };
    try t.expectEqual(@as(u32, 0), sc.run(&ids, &logits, &cached, 0.5, 3));
    try t.expectEqualSlices(u16, &.{ 0, 1, 2, 3, 4 }, &ids);
}

test "expert substitution reuses an expert another row is fetching anyway" {
    const t = std.testing;
    var sc = PickScratch(6){};
    const logits = [_]f32{
        3.0, 2.9, 2.0, -30.0, -30.0, -30.0,
        3.0, 2.9, 2.8, 2.9,   -30.0, -30.0,
    };
    var ids = [_]u16{ 0, 1, 2, 0, 1, 3 };
    const cached = [_]bool{ true, true, false, false, true, true };
    try t.expectEqual(@as(u32, 1), sc.run(&ids, &logits, &cached, 0.3, 3));
    try t.expectEqualSlices(u16, &.{ 0, 1, 2, 0, 1, 2 }, &ids);
}

test "expert substitution stops swapping an expert out after the starvation limit" {
    const t = std.testing;
    var sc = PickScratch(6){};
    const logits = [_]f32{ 3.0, 2.9, 2.8, 2.7, 2.0, -30.0 };
    const cached = [_]bool{ true, true, false, true, true, true };
    for (0..PICK_STARVE_LIMIT) |_| {
        var ids = [_]u16{ 0, 1, 2 };
        try t.expectEqual(@as(u32, 1), sc.run(&ids, &logits, &cached, 0.3, 3));
        try t.expectEqualSlices(u16, &.{ 0, 1, 3 }, &ids);
    }
    var kept = [_]u16{ 0, 1, 2 };
    try t.expectEqual(@as(u32, 0), sc.run(&kept, &logits, &cached, 0.3, 3));
    try t.expectEqualSlices(u16, &.{ 0, 1, 2 }, &kept);
    var again = [_]u16{ 0, 1, 2 };
    try t.expectEqual(@as(u32, 1), sc.run(&again, &logits, &cached, 0.3, 3));
}

test "group cache reports only ready residents as cached" {
    const t = std.testing;
    var cache = try GroupCache.init(t.allocator, 3, 8);
    defer cache.deinit();
    var first = try cache.resolve(t.allocator, &.{ 0, 1, 2 });
    defer first.deinit(t.allocator);
    var mask: [8]bool = undefined;
    cache.cachedMask(&mask);
    try t.expectEqualSlices(bool, &@as([8]bool, @splat(false)), &mask);
    cache.markReady(first.bindings[0].slot.?);
    cache.markReady(first.bindings[1].slot.?);
    cache.cachedMask(&mask);
    try t.expect(mask[0] and mask[1] and !mask[2] and !mask[3]);
}

test "spec map follows the ready cache slots and reports changes" {
    const t = std.testing;
    const expert_to_slot = [_]i32{ 2, -1, 0, 1 };
    const ready = [_]bool{ true, false, true };
    var out = [_]i32{ -2, -2, -2, -2 };
    try t.expect(refreshSpecMap(&expert_to_slot, &ready, &out));
    try t.expectEqualSlices(i32, &.{ 2, -1, 0, -1 }, &out);
    try t.expect(!refreshSpecMap(&expert_to_slot, &ready, &out));
}

test "spec result is kept only when every id resolved to its gathered slot" {
    const t = std.testing;
    const map = [_]i32{ 3, -1, 0, 7 };
    try t.expect(specMatches(&map, &.{ 0, 2, 3 }, &.{ 3, 0, 7 }, false));
    try t.expect(!specMatches(&map, &.{ 0, 2, 3 }, &.{ 3, 0, 7 }, true));
    try t.expect(!specMatches(&map, &.{ 0, 1 }, &.{ 3, 5 }, false));
    try t.expect(!specMatches(&map, &.{ 0, 2 }, &.{ 3, 1 }, false));
    try t.expect(!specMatches(&map, &.{9}, &.{0}, false));
}

test "a failed spec map refresh leaves no GPU map until a rebuild matches the cache" {
    const t = std.testing;
    var layer = LayerState{ .cache = try GroupCache.init(t.allocator, 2, 4) };
    defer {
        if (layer.spec_slots.ctx != null) _ = mlx.mlx_array_free(layer.spec_slots);
        t.allocator.free(layer.spec_host);
        layer.cache.deinit();
    }
    var first = try layer.cache.resolve(t.allocator, &.{ 0, 1 });
    defer first.deinit(t.allocator);
    markResolvedReady(&layer.cache, &first);
    try layer.refreshSpec(t.allocator);
    var second = try layer.cache.resolve(t.allocator, &.{ 0, 2 });
    defer second.deinit(t.allocator);
    markResolvedReady(&layer.cache, &second);
    spec_alloc_fail_for_test = true;
    try t.expectError(error.ExpertSpecMapAlloc, layer.refreshSpec(t.allocator));
    try t.expect(layer.spec_slots.ctx == null);
    try layer.refreshSpec(t.allocator);
    const gpu = mlx.mlx_array_data_int32(layer.spec_slots).?;
    for (layer.spec_host, 0..) |host, e| try t.expectEqual(@max(host, 0), gpu[e]);
    try t.expectEqual(layer.cache.expert_to_slot[2], layer.spec_host[2]);
}

test "GLM BF16 stream imports split expert slabs and preserves bytes across cache and union" {
    const t = std.testing;
    const fixture = @import("glm_stream_fixture.zig");
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.write(t.allocator, tmp.dir, .none);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const geometry = Geometry{ .layers = 4, .experts = 4, .hidden = 32, .intermediate = 16, .first_moe_layer = 3 };
    const s = tinyStream();
    var engine = try Engine.initWithOptions(t.allocator, path_buf[0..path_len], geometry, 2 * 3072, s, .{ .layout = .bf16_individual, .bounce_size = 1 << 20 });
    defer engine.deinit();
    try t.expectEqual(@as(u64, 6), engine.slab_imports);
    try t.expectEqual(@as(u64, 0), engine.fallback_imports);
    try t.expectEqual(@as(u64, 3072), engine.store.perExpertBytes());
    try t.expectError(error.ExpertLayerAbsent, engine.prepareHost(0, &.{0}));
    try t.expect((try engine.specRoute(3)) == null);
    const groups = [_][]const u16{ &.{ 3, 1 }, &.{ 1, 3, 1 }, &.{ 0, 1, 2, 3 }, &.{ 2, 0 } };
    for (groups, 0..) |ids, group| {
        var prepared = try engine.prepareHost(3, ids);
        defer prepared.deinit();
        try t.expect(!prepared.quantized);
        try t.expectEqual(group == 2, prepared.workspace);
        for ([_]mlx.mlx_array{ prepared.gate, prepared.up, prepared.down }, 0..) |array, pi| {
            try t.expect(array.ctx != null);
            try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(array));
            const shape = mlx.getShape(array);
            try t.expectEqual(@as(usize, 3), shape.len);
            try t.expectEqual(@as(c_int, if (pi == 2) 16 else 32), shape[1]);
            try t.expectEqual(@as(c_int, if (pi == 2) 32 else 16), shape[2]);
            const input_dim: c_int = if (pi == 2) 16 else 32;
            const input_data: [32]u16 = @splat(0x3f80);
            const input = mlx.mlx_array_new_data(&input_data, &[_]c_int{ 1, 1, input_dim }, 3, .bfloat16);
            defer _ = mlx.mlx_array_free(input);
            const first_slot = [_]u32{prepared.remapped[0]};
            const selected = mlx.mlx_array_new_data(&first_slot, &[_]c_int{1}, 1, .uint32);
            defer _ = mlx.mlx_array_free(selected);
            var result = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(result);
            try mlx.check(mlx.mlx_gather_mm(&result, input, array, .{ .ctx = null }, selected, false, s));
            try mlx.check(mlx.mlx_array_eval(result));
            const result_data = mlx.mlx_array_data_bfloat16(result).?;
            // Summing 16 or 32 identical powers-of-two-scaled BF16 values is exact.
            const expected = fixture.value(ids[0], pi) + @as(u16, if (pi == 2) 4 else 5) * 128;
            for (result_data[0..mlx.mlx_array_size(result)]) |v| try t.expectEqual(expected, v);
            const ci = pi * 3;
            for (ids, prepared.remapped) |expert, slot| {
                const bytes = if (prepared.workspace) engine.union_slabs[ci].slotBytes(slot) else engine.cacheSlotBytesAt(3, slot, ci);
                for (0..512) |i| try t.expectEqual(fixture.value(expert, pi), std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little));
            }
        }
        if (group == 1) try t.expectEqual(@as(u64, 2), engine.fill_experts_total);
    }
}
