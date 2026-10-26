const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const io_mod = @import("expert_io.zig");
const kernels = @import("expert_bf16_kernels.zig");
pub const quant = @import("expert_quant.zig");
const sushi_exl3 = @import("sushi_exl3");
const expert_usage = @import("expert_usage.zig");
const Ragged = sushi_exl3.Ragged;

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
    /// Every routed expert held (`StoreShape.all_bytes`): no union workspace, no misses.
    resident: bool = false,
};

pub fn fillPeakBytes(expert_bytes: u64, union_size: u16, slots_per_layer: u16, batch_bytes: u64) u64 {
    const workspace = @as(u64, union_size) *| expert_bytes;
    const layer = @as(u64, slots_per_layer) *| expert_bytes;
    return workspace +| layer +| workspace +| @min(workspace, batch_bytes);
}

/// The fill pools' bounce buffers: the serving pool's `FILL_WORKERS` and the prefetch pool's, each at `bounce_size`.
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
    return (try expertBillFor(allocator, model_dir, geometry, layout)).expert_bytes;
}

pub const ExpertBill = struct { expert_bytes: u64, shape: quant.StoreShape = .{} };

/// `expertBytesFor` with the rest of what the store bills, read from the store the engine will open.
pub fn expertBillFor(allocator: std.mem.Allocator, model_dir: []const u8, geometry: Geometry, layout: quant.Layout) !ExpertBill {
    switch (layout) {
        .bf16_fused => return .{ .expert_bytes = try expertBytes(2 * geometry.intermediate, geometry.hidden, geometry.intermediate) },
        .bf16_individual, .fp8_individual, .quantized_split, .mxfp4_split, .mxfp4_individual, .mxfp4_w123, .exl3_k4 => {
            var store = try quant.QuantStore.openForLayout(allocator, model_dir, geometry, layout);
            defer store.deinit();
            return .{ .expert_bytes = store.expertBytes(), .shape = store.shape() };
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

pub fn cachePlanBytes(requested_bytes: u64, layers: u16, experts: u16, expert_bytes: u64, shape: quant.StoreShape) !CachePlan {
    if (layers == 0 or experts == 0 or expert_bytes == 0) return error.InvalidExpertGeometry;
    if (shape.all_bytes > 0 and requested_bytes >= shape.all_bytes) return .{
        .slots_per_layer = experts,
        .expert_bytes = expert_bytes,
        .cache_bytes = shape.all_bytes,
        .workspace_bytes = 0,
        .bounce_bytes = 0,
        .layer_bytes = shape.all_bytes / layers,
        .prefill_peak_bytes = 0,
        .resident = true,
    };
    const per_slot_all_layers = std.math.mul(u64, layers, expert_bytes) catch return error.InvalidExpertGeometry;
    const slots_u64 = @min(requested_bytes / per_slot_all_layers, experts);
    if (slots_u64 == 0) return error.ExpertCacheTooSmall;
    const slots: u16 = @intCast(slots_u64);
    const workspace_bytes = workspaceBytes(experts, expert_bytes, shape) catch return error.InvalidExpertGeometry;
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
    return cachePlanBytes(requested_bytes, moeLayerCount(geometry), geometry.kept(), expert_bytes, geometry.shape);
}

/// Every expert at the mean, once per union workspace the store keeps, plus its grouped layers' union meta tables.
fn workspaceBytes(experts: u16, expert_bytes: u64, shape: quant.StoreShape) !u64 {
    const one = try std.math.mul(u64, experts, expert_bytes);
    return std.math.add(u64, try std.math.mul(u64, one, shape.union_sets), shape.union_meta_bytes);
}

pub const MTP_UNSUPPORTED: []const u8 = "MTP under expert streaming needs --mtp on a Sushi EXL3 Qwen3.8-Flash-Next pack or DeepSeek-V4.1-Flash";

/// A streamed load of a supported pack keeps the MTP head, with its own experts, resident and
/// verifies on the streamed decode path.
pub const MtpUnderStreaming = enum { off, keep, refuse, drop_settings, drop_default };

/// Takes the load's RESOLVED MTP choice and whether the pack can keep its head. A supported pack
/// keeps it whatever the source; elsewhere a launch flag is refused, a per-model `mtp: true` is
/// dropped with a warning, and the engine default quietly resolves off.
pub fn mtpUnderStreaming(mtp_on: bool, from_settings: bool, from_default: bool, head_supported: bool) MtpUnderStreaming {
    if (!mtp_on) return .off;
    if (head_supported) return .keep;
    if (from_default) return .drop_default;
    return if (from_settings) .drop_settings else .refuse;
}

/// The streamed layouts whose kept head is proven live: a Sushi EXL3 Qwen pack, whose resident tensors carry the
/// canonical `language_model.mtp.` keys the loader keeps, and DeepSeek-V4.1-Flash's DSpark stages. Every other model refuses.
pub fn streamedMtpHeadSupported(is_qwen4: bool, is_dsv41: bool, layout: quant.Layout) bool {
    return (is_qwen4 and layout == .exl3_k4) or (is_dsv41 and (layout == .mxfp4_w123 or layout == .exl3_k4));
}

pub fn mtpRefusal(expert_streaming: bool, mtp_requested: bool, head_loaded: bool) ?[]const u8 {
    if (!expert_streaming or !mtp_requested or head_loaded) return null;
    return MTP_UNSUPPORTED;
}

/// PURE: does this load stream its routed experts? A checkpoint with no resident
/// arm always does; a capable one only when an SSD budget was asked for,
/// so a quantized pack without one loads resident.
pub fn expertStreamingEngaged(capable: bool, required: bool, budget_bytes: u64) bool {
    if (!capable) return false;
    return required or budget_bytes > 0;
}

pub const BudgetLedger = struct {
    budget_bytes: u64,
    trunk_bytes: u64,
    mtp_bytes: u64,
    vision_bytes: u64,
    workspace_bytes: u64,
    selected_bytes: u64,
    bounce_bytes: u64,
    pool_bytes: u64 = 0,
    cache_bytes: u64,
    slots_per_layer: u16,
    resident: bool = false,
};

/// The budget is the TOTAL resident target (GiB, the `--ssd-budget-gb` unit); the expert
/// cache is what remains after trunk, MTP, the vision tower with its encode reserve, the
/// all-experts union workspace, the selected slab and the bounce buffers. The PLE table is a
/// disk gather and is never billed. A budget that holds every routed expert (`StoreShape.all_bytes`) beside the trunk,
/// MTP and tower serves them resident: nothing is ever read on a route, so no workspace, selected slab or bounce.
pub fn budgetLedger(
    budget_bytes: u64,
    trunk_bytes: u64,
    mtp_bytes: u64,
    vision_bytes: u64,
    layers: u16,
    experts: u16,
    top_k: u16,
    expert_bytes: u64,
    bounce_bytes: u64,
    shape: quant.StoreShape,
) !BudgetLedger {
    if (layers == 0 or experts == 0 or top_k == 0 or expert_bytes == 0) return error.InvalidExpertGeometry;
    const beside = trunk_bytes +| mtp_bytes +| vision_bytes +| shape.pool_bytes;
    if (shape.all_bytes > 0 and budget_bytes -| beside >= shape.all_bytes) return .{
        .budget_bytes = budget_bytes,
        .trunk_bytes = trunk_bytes,
        .mtp_bytes = mtp_bytes,
        .vision_bytes = vision_bytes,
        .workspace_bytes = 0,
        .selected_bytes = 0,
        .bounce_bytes = 0,
        .pool_bytes = shape.pool_bytes,
        .cache_bytes = shape.all_bytes,
        .slots_per_layer = experts,
        .resident = true,
    };
    const workspace_bytes = workspaceBytes(experts, expert_bytes, shape) catch return error.InvalidExpertGeometry;
    const selected_bytes = @as(u64, @min(top_k, experts)) *| expert_bytes;
    const fixed = trunk_bytes +| mtp_bytes +| vision_bytes +| workspace_bytes +| selected_bytes +| bounce_bytes +| shape.pool_bytes;
    if (fixed >= budget_bytes) return error.SsdBudgetBelowResident;
    const per_slot = @as(u64, layers) *| expert_bytes;
    const slots_u64 = @min((budget_bytes - fixed) / per_slot, experts);
    // A layer's cache needs two slots, and a grouped layer one expert of every rate group.
    if (slots_u64 < @max(2, std.math.divCeil(u64, shape.layer_floor_bytes, expert_bytes) catch unreachable)) return error.SsdBudgetBelowResident;
    const slots: u16 = @intCast(slots_u64);
    return .{
        .budget_bytes = budget_bytes,
        .trunk_bytes = trunk_bytes,
        .mtp_bytes = mtp_bytes,
        .vision_bytes = vision_bytes,
        .workspace_bytes = workspace_bytes,
        .selected_bytes = selected_bytes,
        .bounce_bytes = bounce_bytes,
        .pool_bytes = shape.pool_bytes,
        .cache_bytes = @as(u64, slots) *| per_slot,
        .slots_per_layer = slots,
    };
}

/// An optional pack sidecar that shapes how the expert cache splits its bytes between layers (see `docs/pack-format.md`).
pub const cache_profile_file = "expert_cache_profile.json";

/// Per routed layer, LRU stack-distance counts of calibration routes in decode order: `counts[i][d]` is the hits layer
/// i's cache gains from its (d+1)-th slot; the last entry counts first touches.
pub const CacheProfile = struct {
    counts: [][]u64,

    pub fn deinit(self: *CacheProfile, allocator: std.mem.Allocator) void {
        for (self.counts) |row| allocator.free(row);
        allocator.free(self.counts);
    }
};

/// Null unless `raw` is a version-1 profile with one row of `experts + 1` counts for each of `layers` routed layers.
pub fn parseCacheProfile(allocator: std.mem.Allocator, raw: []const u8, layers: usize, experts: usize) ?CacheProfile {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const obj = parsed.value.object;
    const version = obj.get("version") orelse return null;
    if (version != .integer or version.integer != 1) return null;
    const rows = obj.get("layers") orelse return null;
    if (rows != .array or rows.array.items.len != layers) return null;
    for (rows.array.items) |row| {
        if (row != .array or row.array.items.len != experts + 1) return null;
        for (row.array.items) |v| if (v != .integer or v.integer < 0) return null;
    }
    const counts = allocator.alloc([]u64, layers) catch return null;
    for (counts, rows.array.items, 0..) |*dst, row, i| {
        dst.* = allocator.alloc(u64, experts + 1) catch {
            for (counts[0..i]) |done| allocator.free(done);
            allocator.free(counts);
            return null;
        };
        for (row.array.items, dst.*) |v, *c| c.* = @intCast(v.integer);
    }
    return .{ .counts = counts };
}

pub fn readCacheProfile(allocator: std.mem.Allocator, model_dir: []const u8, layers: usize, experts: usize) ?CacheProfile {
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return null;
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, cache_profile_file, allocator, .limited(16 * 1024 * 1024)) catch return null;
    defer allocator.free(raw);
    return parseCacheProfile(allocator, raw, layers, experts);
}

/// The experts a rate-grouped layer preloads: each pool's most used experts, as many as it has slots, least used first so
/// the most used end up most recent in the pool's LRU.
fn rankedWarm(allocator: std.mem.Allocator, out: *std.ArrayList(u16), gl: *const GroupedLayer, counts: []const u64, drop: ?*const expert_usage.Drop, layer: u16) !void {
    for (0..gl.groups.count) |g| {
        const lo: u16 = @intCast(gl.groups.base[g]);
        const order = try expert_usage.rank(allocator, counts[lo .. lo + gl.groups.experts[g]]);
        defer allocator.free(order);
        var from = order.len;
        var kept: usize = 0;
        while (from > 0 and kept < gl.slots[g]) {
            from -= 1;
            if (drop) |d| if (d.isDropped(layer, lo + order[from])) continue;
            kept += 1;
        }
        for (order[from..]) |o| {
            if (drop) |d| if (d.isDropped(layer, lo + o)) continue;
            try out.append(allocator, lo + o);
        }
    }
}

fn readWarmUsage(allocator: std.mem.Allocator, model_dir: []const u8, layers: usize, experts: usize) ?expert_usage.Usage {
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return null;
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, expert_usage.file_name, allocator, .limited(16 * 1024 * 1024)) catch return null;
    defer allocator.free(raw);
    return expert_usage.parse(allocator, raw, layers, experts) catch null;
}

/// Splits `total` bytes between layers by decode hits per byte: layer i starts at `floor[i]`, then the layer whose next
/// expert (`mean[i]` bytes, capped at `full[i]`) gains the most hits per byte takes it, until nothing fits. False when
/// the floors alone exceed `total`.
pub fn profileAllot(out: []u64, counts: []const []const u64, mean: []const u64, full: []const u64, floor: []const u64, total: u64) bool {
    var spent: u64 = 0;
    for (out, floor, full) |*o, f, cap| {
        o.* = @min(f, cap);
        spent += o.*;
    }
    if (spent > total) return false;
    while (true) {
        var best: ?usize = null;
        var best_gain: f64 = -1;
        for (out, counts, mean, full, 0..) |o, row, m, cap, i| {
            if (o >= cap or m == 0) continue;
            const step = @min(m, cap - o);
            if (spent + step > total) continue;
            const slot: usize = @intCast(o / m);
            const hits = row[@min(slot, row.len - 2)];
            const gain = @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(step));
            if (gain > best_gain) {
                best = i;
                best_gain = gain;
            }
        }
        const i = best orelse return true;
        const step = @min(mean[i], full[i] - out[i]);
        out[i] += step;
        spent += step;
    }
}

pub fn cacheSplitLine(buf: []u8, profiled: bool, min_slots: u32, max_slots: u32) []const u8 {
    return (if (profiled)
        std.fmt.bufPrint(buf, "[expert-stream] slots per layer: profile ({d}..{d})\n", .{ min_slots, max_slots })
    else
        std.fmt.bufPrint(buf, "[expert-stream] slots per layer: uniform ({d})\n", .{max_slots})) catch buf[0..0];
}

/// Each layer's cache bytes under the pack's profile, indexed by layer, or null for the uniform split: no valid profile,
/// or a routed layer that is not rate-grouped. The layers share the uniform total, so the ledger's bill holds.
fn profiledAllot(allocator: std.mem.Allocator, model_dir: []const u8, store: *const ExpertStore, geometry: Geometry, plan: CachePlan) !?[]u64 {
    const q = if (store.quantized) |*q| q else return null;
    var routed: usize = 0;
    for (0..geometry.layers) |li| {
        if (!store.hasExpertLayer(@intCast(li))) continue;
        if (q.groupsOf(@intCast(li)) == null) return null;
        routed += 1;
    }
    var profile = readCacheProfile(allocator, model_dir, routed, geometry.experts) orelse return null;
    defer profile.deinit(allocator);
    const uniform = @as(u64, plan.slots_per_layer) * plan.expert_bytes;
    const per = try allocator.alloc(u64, 4 * routed);
    defer allocator.free(per);
    const mean = per[0..routed];
    const full = per[routed .. 2 * routed];
    const floor = per[2 * routed .. 3 * routed];
    const split = per[3 * routed ..];
    var r: usize = 0;
    for (0..geometry.layers) |li| {
        const g = if (store.hasExpertLayer(@intCast(li))) q.groupsOf(@intCast(li)).? else continue;
        full[r] = 0;
        for (0..g.count) |gi| full[r] += @as(u64, g.keptOf(gi)) * g.expertBytes(gi, geometry.hidden, geometry.intermediate);
        mean[r] = full[r] / geometry.kept();
        // A quarter of the uniform share keeps a cold layer from thrashing on a prompt unlike the calibration routes.
        floor[r] = @max(g.floorBytes(geometry.hidden, geometry.intermediate), uniform / 4);
        r += 1;
    }
    if (!profileAllot(split, profile.counts, mean, full, floor, uniform * routed)) return null;
    const out = try allocator.alloc(u64, geometry.layers);
    @memset(out, 0);
    r = 0;
    for (0..geometry.layers) |li| {
        if (!store.hasExpertLayer(@intCast(li))) continue;
        out[li] = split[r];
        r += 1;
    }
    return out;
}

pub const FILL_WORKERS: usize = 4;
pub const COALESCE_MAX: u64 = 64 * 1024 * 1024;

pub const BREAKDOWN_EVERY: u64 = 64;

/// Leases one prepared route holds: a split route leases its layer's cache and the union workspace.
const HELD_MAX: usize = 2 * quant.component_count;
const PENDING_READERS_MAX: usize = 2 * HELD_MAX;

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
    copy_ns: u64 = 0,
    unions: u64 = 0,
    route_linear: RouteClass = .{},
    route_full: RouteClass = .{},
    probe: bool = false,

    pub fn otherNs(self: Breakdown) u64 {
        const accounted = self.route_ns +| self.fill_ns +| self.compute_ns +| self.copy_ns;
        return if (self.wall_ns > accounted) self.wall_ns - accounted else 0;
    }
};

pub const Options = struct {
    io_workers: usize = FILL_WORKERS,
    bounce_size: usize = 64 * 1024 * 1024,
    layout: quant.Layout = .bf16_fused,
    /// A drop the engine frees with itself once `initWithOptions` returns; `geometry.drop` must point at it.
    drop_owned: ?*expert_usage.Drop = null,
    /// Rate-grouped trellis tiles land in the pools in `layout.tileMajor` order, which the rate-group kernels read.
    tile_major: bool = false,
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

/// The line for a lossy pick asked of a resident load: every expert is pooled, so no route misses and none is swapped.
pub fn pickInertLine(resident: bool, tolerance: f32) ?[]const u8 {
    if (!resident or tolerance <= 0) return null;
    return "[expert-swap] --expert-pick-tolerance has no effect: every routed expert is resident, so nothing misses";
}

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
                .u8, .f8e4m3 => .uint8,
                .u32 => .uint32,
                .f32 => .float32,
                .other => .bfloat16,
            };
            const elem: u8 = switch (q.dtypeOf(c)) {
                .u8, .f8e4m3 => 1,
                .u32, .f32 => 4,
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
        if (geometry.drop_n != 0) return error.ExpertDropUnsupported;
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

    /// One flat axis of `elems` elements: a rate-grouped layer's trellis storage, pools back to back.
    fn createFlat(allocator: std.mem.Allocator, elems: usize, dtype: mlx.mlx_dtype, elem_bytes: u8) !SlabOperand {
        if (elems == 0 or elems > std.math.maxInt(c_int) or elem_bytes == 0) return error.InvalidExpertGeometry;
        const stride = elems * elem_bytes;
        const slab = try io_mod.PageSlab.create(allocator, stride);
        errdefer slab.destroy();
        const payload = try allocator.create(io_mod.ImportPayload);
        errdefer allocator.destroy(payload);
        payload.* = .{};
        const operand = try io_mod.importSlab(slab, &.{@intCast(elems)}, dtype, payload, .{});
        if (!operand.aliased) {
            _ = mlx.mlx_array_free(operand.array);
            return error.ExpertSlabImportCopied;
        }
        return .{ .present = true, .slab = slab, .payload = payload, .array = operand.array, .stride = stride, .count = 1 };
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
        const own = mlx.getShape(self.array);
        const dims = mlx.getShape(template);
        if (own.len == dims.len and std.mem.eql(c_int, own[1..], dims[1..])) {
            // Same slot shape: slice the slot axis; one flat axis can pass MLX's 32-bit dimensions.
            var start: [4]c_int = @splat(0);
            var stop: [4]c_int = undefined;
            @memcpy(stop[0..own.len], own);
            stop[0] = @intCast(self.count);
            const step: [4]c_int = @splat(1);
            var view = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(view);
            try mlx.check(mlx.mlx_slice(&view, self.array, &start, own.len, &stop, own.len, &step, own.len, s));
            return view;
        }
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, self.array, &.{-1}, 1, s));
        var prefix = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(prefix);
        const elements: c_int = @intCast(self.count * self.stride / mlx.mlx_array_itemsize(self.array));
        try mlx.check(mlx.mlx_slice(&prefix, flat, &.{0}, 1, &.{elements}, 1, &.{1}, 1, s));
        var shape: [4]c_int = undefined;
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
    /// Host copies between a grouped layer's cache and the union workspace, and the routes that took that path.
    copy_ns: u64 = 0,
    unions: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    fill_bytes: u64 = 0,
    host_sync_ns: u64 = 0,
    fill_ns: u64 = 0,
    compute_ns: u64 = 0,
};

/// A rate-grouped layer under streaming: one slot pool per rate group, each with its own cache over the group's
/// experts. The layer's nine slabs hold the pools back to back (trellis flat per projection, scale rows per slot),
/// and a `Ragged` bank over them serves the layer's routed experts by GLOBAL slot id: pool g's slots follow pool
/// g-1's, so the bank's meta table is a function of the pool structure alone. The union workspace has the same
/// structure with every expert of a group as a slot.
const GroupedLayer = struct {
    groups: quant.LayerGroups,
    hidden: u32,
    inter: u32,
    slots: [quant.rate_group_limit]u32 = @splat(0),
    slot_base: [quant.rate_group_limit]u32 = @splat(0),
    /// Where group g's union slots start in the layer's union bank: the kept experts of the groups before it.
    union_base: [quant.rate_group_limit]u32 = @splat(0),
    cache_slots: u32 = 0,
    cache_off: [quant.rate_group_limit][3]u64 = @splat(@splat(0)),
    union_off: [quant.rate_group_limit][3]u64 = @splat(@splat(0)),
    caches: [quant.rate_group_limit]GroupCache = undefined,
    caches_made: usize = 0,
    cache_ragged: Ragged = .{},
    union_ragged: Ragged = .{},
    tile_major: bool = false,

    fn init(allocator: std.mem.Allocator, groups: quant.LayerGroups, hidden: u32, inter: u32, allot_bytes: u64, tile_major: bool) !GroupedLayer {
        var self: GroupedLayer = .{ .groups = groups, .hidden = hidden, .inter = inter, .tile_major = tile_major };
        errdefer self.deinitCaches();
        try self.sizePools(allot_bytes);
        var union_at: [3]u64 = @splat(0);
        var cache_at: [3]u64 = @splat(0);
        var union_slots: u32 = 0;
        for (0..groups.count) |g| {
            self.slot_base[g] = self.cache_slots;
            self.cache_slots += self.slots[g];
            self.union_base[g] = union_slots;
            union_slots += groups.keptOf(g);
            for (0..3) |pi| {
                self.cache_off[g][pi] = cache_at[pi];
                self.union_off[g][pi] = union_at[pi];
                cache_at[pi] += self.slots[g] * self.trellisBytes(g, pi);
                union_at[pi] += @as(u64, groups.keptOf(g)) * self.trellisBytes(g, pi);
            }
        }
        try self.initCaches(allocator);
        return self;
    }

    fn expertCount(self: *const GroupedLayer) usize {
        const last = self.groups.count - 1;
        return @as(usize, self.groups.base[last]) + self.groups.experts[last];
    }

    fn expertBytes(self: *const GroupedLayer, g: usize) u64 {
        return self.groups.expertBytes(g, self.hidden, self.inter);
    }

    /// Splits `allot_bytes` (the ledger's bill for the layer) so every pool caches the same fraction of its experts and
    /// the pools never hold more than was billed. A group that cannot afford one expert of its share is pinned at one
    /// slot, paid first; `ExpertCacheTooSmall` when one slot per group does not fit.
    fn sizePools(self: *GroupedLayer, allot_bytes: u64) !void {
        const count: usize = self.groups.count;
        var pinned: [quant.rate_group_limit]bool = @splat(false);
        while (true) {
            var spare = allot_bytes;
            var free_bytes: u64 = 0;
            for (0..count) |g| {
                if (!pinned[g]) {
                    free_bytes += @as(u64, self.groups.keptOf(g)) * self.expertBytes(g);
                    continue;
                }
                self.slots[g] = 1;
                spare = std.math.sub(u64, spare, self.expertBytes(g)) catch return error.ExpertCacheTooSmall;
            }
            var changed = false;
            for (0..count) |g| {
                if (pinned[g]) continue;
                const fit = spare * self.groups.keptOf(g) / free_bytes;
                if (fit == 0) {
                    pinned[g] = true;
                    changed = true;
                } else self.slots[g] = @intCast(@min(fit, self.groups.keptOf(g)));
            }
            if (!changed) return;
        }
    }

    fn deinitCaches(self: *GroupedLayer) void {
        for (self.caches[0..self.caches_made]) |*cache| cache.deinit();
        self.caches_made = 0;
    }

    fn deinitBanks(self: *GroupedLayer) void {
        self.cache_ragged.deinit();
        self.union_ragged.deinit();
    }

    fn tiles(self: *const GroupedLayer) u64 {
        return @as(u64, self.hidden / 16) * (self.inter / 16);
    }

    fn trellisBytes(self: *const GroupedLayer, g: usize, pi: usize) u64 {
        return self.tiles() * self.groups.n[g][pi] * 2;
    }

    /// Elements of projection pi's flat trellis storage holding `per_group[g]` slots of every group.
    fn trellisElems(self: *const GroupedLayer, pi: usize, per_group: []const u32) usize {
        var total: u64 = 0;
        for (0..self.groups.count) |g| total += @as(u64, per_group[g]) * self.trellisBytes(g, pi) / 2;
        return @intCast(total);
    }

    /// Bytes of one slot's piece of component `ci` in group g.
    fn pieceBytes(self: *const GroupedLayer, g: usize, ci: usize) u64 {
        const pi = ci / 3;
        return switch (ci % 3) {
            0 => self.trellisBytes(g, pi),
            1 => @as(u64, if (pi == 2) self.inter else self.hidden) * 2,
            else => @as(u64, if (pi == 2) self.hidden else self.inter) * 2,
        };
    }

    fn region(self: *const GroupedLayer, operand: *const SlabOperand, g: usize, ci: usize, row: u64, in_cache: bool) []u8 {
        const bytes = operand.slab.?.bytes;
        const len = self.pieceBytes(g, ci);
        const at = if (ci % 3 == 0)
            (if (in_cache) self.cache_off[g][ci / 3] else self.union_off[g][ci / 3]) + row * len
        else
            ((if (in_cache) self.slot_base[g] else self.union_base[g]) + row) * len;
        return bytes[@intCast(at)..][0..@intCast(len)];
    }

    /// The fill of component `ci` of one expert of group `g` from `source` into `dst`; a trellis lands tile-major
    /// when the layer is.
    fn fillSpan(self: *const GroupedLayer, source: SourceSpan, dst: []u8, g: usize, ci: usize) io_mod.FillSpan {
        var span: io_mod.FillSpan = .{ .file = source.file, .offset = source.offset, .len = source.len, .dst = dst.ptr };
        if (self.tile_major and ci % 3 == 0) {
            const pi = ci / 3;
            const it: u32 = (if (pi == 2) self.inter else self.hidden) / 16;
            const ot: u32 = (if (pi == 2) self.hidden else self.inter) / 16;
            span.tile = .{ .it = it, .ot = ot, .n = self.groups.n[g][pi] };
        }
        return span;
    }

    fn cacheRegion(self: *const GroupedLayer, operand: *const SlabOperand, g: usize, ci: usize, slot: u64) []u8 {
        return self.region(operand, g, ci, slot, true);
    }

    fn unionRegion(self: *const GroupedLayer, operand: *const SlabOperand, g: usize, ci: usize, position: u64) []u8 {
        return self.region(operand, g, ci, position, false);
    }

    fn pools(self: *const GroupedLayer, in_union: bool, out: *[quant.rate_group_limit]rate_groups_pool) []const rate_groups_pool {
        for (0..self.groups.count) |g| out[g] = .{ .slots = if (in_union) self.groups.keptOf(g) else self.slots[g], .rates = self.groups.n[g] };
        return out[0..self.groups.count];
    }

    fn buildBanks(self: *GroupedLayer, s: mlx.mlx_stream, cache: []const SlabOperand, union_set: []const SlabOperand) !void {
        try self.buildCacheBank(s, cache);
        errdefer self.cache_ragged.deinit();
        var pool_buf: [quant.rate_group_limit]rate_groups_pool = undefined;
        self.union_ragged = try Ragged.initPools(s, self.hidden, self.inter, self.pools(true, &pool_buf), slabTriple(union_set, 0), slabTriple(union_set, 1), slabTriple(union_set, 2));
        self.union_ragged.tile_major = self.tile_major;
    }

    fn buildCacheBank(self: *GroupedLayer, s: mlx.mlx_stream, cache: []const SlabOperand) !void {
        var pool_buf: [quant.rate_group_limit]rate_groups_pool = undefined;
        self.cache_ragged = try Ragged.initPools(s, self.hidden, self.inter, self.pools(false, &pool_buf), slabTriple(cache, 0), slabTriple(cache, 1), slabTriple(cache, 2));
        self.cache_ragged.tile_major = self.tile_major;
    }

    fn initCaches(self: *GroupedLayer, allocator: std.mem.Allocator) !void {
        errdefer self.deinitCaches();
        for (0..self.groups.count) |g| {
            self.caches[g] = try GroupCache.init(allocator, @intCast(self.slots[g]), self.groups.experts[g]);
            self.caches_made += 1;
        }
    }
};

const rate_groups_pool = sushi_exl3.rate_groups.Pool;

/// The arrays of part `part` (0 trellis, 1 suh, 2 svh) of the three projections.
fn slabTriple(set: []const SlabOperand, part: usize) [3]mlx.mlx_array {
    return .{ set[part].array, set[3 + part].array, set[6 + part].array };
}

const LayerState = struct {
    cache: GroupCache,
    /// Set for a rate-grouped layer: its pools replace `cache`.
    grouped: ?*GroupedLayer = null,
    active: bool = false,
    /// Released for a prefill's working set (`Engine.borrowLayer`): no pool, every route reads into the union workspace.
    borrowed: bool = false,
    /// The experts the pool held when it was released (global ids), read back by `Engine.restoreLayer`, and the bytes the
    /// release gave back.
    saved: []u16 = &.{},
    lent: u64 = 0,
    /// A resident pack's int32 `[experts]` map from expert id to its global slot in the cache bank; null while released.
    resident_map: mlx.mlx_array = .{ .ctx = null },
    /// A rate-grouped layer's `substituteMisses` starvation counts, by global expert id.
    starved: []u8 = &.{},
    slabs: []SlabOperand = &.{},
    stats: LayerStats = .{},
    spec_host: []i32 = &.{},
    spec_slots: mlx.mlx_array = .{ .ctx = null },
    spec_raw: mlx.mlx_array = .{ .ctx = null },

    /// Brings the host map and its GPU copy up to the cache. The pair only ever
    /// changes together: a failure leaves no GPU map and a host map that forces
    /// the next call to rebuild.
    fn refreshSpec(self: *LayerState, allocator: std.mem.Allocator) !void {
        const experts = if (self.grouped) |gl| gl.expertCount() else self.cache.expert_to_slot.len;
        if (self.spec_host.len != experts) {
            if (self.spec_host.len != 0) allocator.free(self.spec_host);
            self.spec_host = &.{};
            self.spec_host = try allocator.alloc(i32, experts);
            @memset(self.spec_host, -2);
        }
        const changed = if (self.grouped) |gl| refreshGroupedSpecMap(gl, self.spec_host) else refreshSpecMap(self.cache.expert_to_slot, self.cache.ready, self.spec_host);
        if (!changed and self.spec_slots.ctx != null) return;
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

/// `refreshSpecMap` for a rate-grouped layer: the GLOBAL slot of each expert (its pool's base plus its slot).
fn refreshGroupedSpecMap(gl: *const GroupedLayer, out: []i32) bool {
    var changed = false;
    for (0..gl.groups.count) |g| {
        const cache = &gl.caches[g];
        const base = gl.groups.base[g];
        for (cache.expert_to_slot, 0..) |raw, local| {
            const now: i32 = if (raw >= 0 and cache.ready[@intCast(raw)]) @as(i32, @intCast(gl.slot_base[g])) + raw else -1;
            if (out[base + local] != now) {
                out[base + local] = now;
                changed = true;
            }
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

/// The nine slabs of a rate-grouped layer's cache (`slots` per group) or, with `slots` null, of the union workspace
/// (`elems` trellis elements per projection): trellis flat per projection, `rows` scale rows. Scale widths and
/// dtypes come from `layer`.
fn createGroupedSlabs(allocator: std.mem.Allocator, layer: *const GroupedLayer, slots: ?[]const u32, rows: u32, s: mlx.mlx_stream, elems: ?[3]usize) ![]SlabOperand {
    const n = quant.component_count;
    const slabs = try allocator.alloc(SlabOperand, n);
    errdefer allocator.free(slabs);
    var made: usize = 0;
    errdefer for (slabs[0..made]) |*operand| operand.destroy(allocator, s);
    while (made < n) : (made += 1) {
        const pi = made / 3;
        slabs[made] = if (made % 3 == 0)
            try SlabOperand.createFlat(allocator, if (slots) |per_group| layer.trellisElems(pi, per_group) else elems.?[pi], .uint16, 2)
        else
            try SlabOperand.createShaped(allocator, rows, &.{@intCast(layer.pieceBytes(0, made) / 2)}, .float16, 2);
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
    /// A rate-grouped layer's bank over the slabs, indexed by the global slot ids in `remapped`.
    ragged: ?*const Ragged = null,
    held: [HELD_MAX]?HeldLease = @splat(null),
    defer_to: ?*Engine = null,
    /// A split route's own bank (`ragged` points at it), freed with the route.
    split: ?*Ragged = null,

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
        if (self.split) |bank| {
            bank.deinit();
            self.allocator.destroy(bank);
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

/// Reads experts a later layer is expected to pick into that layer's pool, on a thread of its own, so a miss the router
/// has not made yet is a hit when the layer is queued. A job owns its layer's cache until `Engine.prefetchWait` returns.
pub const Prefetcher = struct {
    pub const rows_max = 16;
    pub const cand_max = 12;
    const jobs_max = 32;

    const Job = struct {
        layer: u16,
        rows: u8,
        n: u8,
        m: u8,
        cand: [rows_max][cand_max]u16,
    };

    pool: *io_mod.FillPool,
    thread: std.Thread,
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    jobs: [jobs_max]Job = undefined,
    head: usize = 0,
    len: usize = 0,
    busy: i32 = -1,
    quit: bool = false,
    /// Jobs queued or running; zero lets the host skip the lock.
    live: std.atomic.Value(u32) = .init(0),
    jobs_queued: std.atomic.Value(u64) = .init(0),
    jobs_run: std.atomic.Value(u64) = .init(0),
    experts_read: std.atomic.Value(u64) = .init(0),
    host_wait_ns: std.atomic.Value(u64) = .init(0),
    job_ns: std.atomic.Value(u64) = .init(0),
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    geometry: Geometry,
    drop_owned: ?*expert_usage.Drop = null,
    /// The pack's `expert_usage.json` counts, `[routed layer][expert id]`: ranks what `warmCache` preloads.
    warm_usage: ?expert_usage.Usage = null,
    plan: CachePlan,
    store: ExpertStore,
    layers: []LayerState,
    s: mlx.mlx_stream,
    forward_count: u64 = 0,
    io_pool: ?*io_mod.FillPool = null,
    prefetcher: ?*Prefetcher = null,
    fds: []std.c.fd_t = &.{},
    union_slabs: []SlabOperand = &.{},
    /// The union workspace of the rate-grouped layers: sized for the largest of them. A resident pack makes it only
    /// while a layer is released (`borrowLayer`), sized for the layers that may be.
    grouped_union: []SlabOperand = &.{},
    union_elems: [3]usize = @splat(0),
    slab_imports: u64 = 0,
    fallback_imports: u64 = 0,
    fill_experts_total: u64 = 0,
    fill_bytes_total: u64 = 0,
    fill_ns_total: u64 = 0,
    copy_ns_total: u64 = 0,
    copy_bytes_total: u64 = 0,
    forward_clock: Clock,
    last: Breakdown = .{},
    route_linear: RouteClass = .{},
    route_full: RouteClass = .{},
    pending: [PENDING_READERS_MAX]?HeldLease = @splat(null),
    /// The first layer whose pool a prefill may release (`borrowLayer`), and where the bytes still lendable are
    /// published (`ModelConfig.expert_borrow_bytes`, read by admission on other threads).
    borrow_from: u16 = std.math.maxInt(u16),
    borrow_publish: ?*u64 = null,
    /// Routed ids the lossy pick replaced on rate-grouped layers, of the ids it saw (`swapGroupedMisses`).
    swapped_ids: u64 = 0,
    swap_seen_ids: u64 = 0,
    swap_reported_at: u64 = 0,

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
        var sized = geometry;
        if (store.quantized) |*q| sized.shape = q.shape();
        const plan = try cachePlanBytesForGeometry(requested_bytes, sized, store.perExpertBytes());
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
        const allot = if (plan.resident) null else try profiledAllot(allocator, model_dir, &engine.store, geometry, plan);
        defer if (allot) |bytes| allocator.free(bytes);
        var any_uniform = false;
        for (layers, 0..) |*layer, layer_index| {
            initialized += 1;
            if (!engine.store.hasExpertLayer(@intCast(layer_index))) continue;
            if (engine.store.quantized) |*q| if (q.groupsOf(@intCast(layer_index))) |groups| {
                const grouped = try allocator.create(GroupedLayer);
                const layer_allot = if (plan.resident)
                    groups.allBytes(geometry.hidden, geometry.intermediate)
                else if (allot) |bytes| bytes[layer_index] else @as(u64, plan.slots_per_layer) * plan.expert_bytes;
                grouped.* = GroupedLayer.init(allocator, groups.*, geometry.hidden, geometry.intermediate, layer_allot, opts.tile_major) catch |err| {
                    allocator.destroy(grouped);
                    return err;
                };
                layer.* = .{ .cache = GroupCache.empty(allocator), .grouped = grouped, .active = true };
                var per_group: [quant.rate_group_limit]u32 = undefined;
                @memcpy(per_group[0..groups.count], grouped.slots[0..groups.count]);
                layer.slabs = try createGroupedSlabs(allocator, grouped, per_group[0..groups.count], grouped.cache_slots, s, null);
                engine.slab_imports += presentSlabCount(layer.slabs);
                continue;
            };
            any_uniform = true;
            layer.* = .{ .cache = try GroupCache.init(allocator, plan.slots_per_layer, geometry.experts), .active = true };
            layer.slabs = try createSlabSetAt(allocator, &engine.store, @intCast(layer_index), plan.slots_per_layer, s);
            engine.slab_imports += presentSlabCount(layer.slabs);
        }
        // The grouped layers share one union workspace, each projection sized for the layer that needs most of it.
        var widest: ?*const GroupedLayer = null;
        var union_elems: [3]usize = @splat(0);
        for (layers) |*layer| if (layer.grouped) |g| {
            widest = g;
            var every: [quant.rate_group_limit]u32 = undefined;
            for (every[0..g.groups.count], 0..) |*dst, gi| dst.* = g.groups.keptOf(gi);
            for (0..3) |pi| union_elems[pi] = @max(union_elems[pi], g.trellisElems(pi, every[0..g.groups.count]));
        };
        if (any_uniform or widest == null) {
            engine.union_slabs = try createSlabSet(allocator, &engine.store, geometry.kept(), s);
            engine.slab_imports += presentSlabCount(engine.union_slabs);
        }
        engine.union_elems = union_elems;
        if (widest != null and plan.resident) {
            for (layers) |*layer| if (layer.grouped) |g| try g.buildCacheBank(s, layer.slabs);
        } else if (widest) |shape| {
            engine.grouped_union = try createGroupedSlabs(allocator, shape, null, geometry.kept(), s, union_elems);
            engine.slab_imports += presentSlabCount(engine.grouped_union);
            for (layers) |*layer| if (layer.grouped) |g| try g.buildBanks(s, layer.slabs, engine.grouped_union);
        }
        engine.fallback_imports = io_mod.fallback_imports.load(.monotonic) - before_fallback;
        var min_slots: u32 = plan.slots_per_layer;
        var max_slots: u32 = plan.slots_per_layer;
        if (allot != null) {
            min_slots = std.math.maxInt(u32);
            max_slots = 0;
            for (layers) |*layer| if (layer.grouped) |g| {
                min_slots = @min(min_slots, g.cache_slots);
                max_slots = @max(max_slots, g.cache_slots);
            };
        }
        if (plan.resident) try engine.loadResident();
        if (pickInertLine(plan.resident, pick_tolerance)) |line| log.warn("{s}\n", .{line});
        var split_buf: [96]u8 = undefined;
        log.info("{s}", .{cacheSplitLine(&split_buf, allot != null, min_slots, max_slots)});
        if (geometry.drop) |d| {
            engine.drop_owned = opts.drop_owned;
            log.info("[expert-drop] n={d} per layer, {d} experts, {d:.2} GB not loaded, list from {s}\n", .{
                d.n, @as(u64, d.n) * moeLayerCount(geometry), @as(f64, @floatFromInt(engine.droppedBytes())) / 1e9, d.source,
            });
        } else engine.drop_owned = opts.drop_owned;
        if (!plan.resident) engine.warm_usage = readWarmUsage(allocator, model_dir, moeLayerCount(geometry), geometry.experts);
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
        // A bank's arrays alias the slabs, so they go first.
        for (self.layers) |*layer| if (layer.grouped) |grouped| {
            grouped.deinitBanks();
            grouped.deinitCaches();
            self.allocator.destroy(grouped);
            layer.grouped = null;
        };
        for (self.layers) |*layer| {
            if (layer.spec_slots.ctx != null) _ = mlx.mlx_array_free(layer.spec_slots);
            layer.spec_slots = .{ .ctx = null };
            if (layer.spec_raw.ctx != null) _ = mlx.mlx_array_free(layer.spec_raw);
            layer.spec_raw = .{ .ctx = null };
            if (layer.resident_map.ctx != null) _ = mlx.mlx_array_free(layer.resident_map);
            layer.resident_map = .{ .ctx = null };
            if (layer.starved.len != 0) self.allocator.free(layer.starved);
            layer.starved = &.{};
            if (layer.spec_host.len != 0) self.allocator.free(layer.spec_host);
            layer.spec_host = &.{};
            for (layer.slabs) |*operand| operand.destroy(self.allocator, self.s);
            if (layer.slabs.len != 0) self.allocator.free(layer.slabs);
            layer.slabs = &.{};
            if (layer.saved.len != 0) self.allocator.free(layer.saved);
            layer.saved = &.{};
        }
        for (self.union_slabs) |*operand| operand.destroy(self.allocator, self.s);
        if (self.union_slabs.len != 0) self.allocator.free(self.union_slabs);
        self.union_slabs = &.{};
        for (self.grouped_union) |*operand| operand.destroy(self.allocator, self.s);
        if (self.grouped_union.len != 0) self.allocator.free(self.grouped_union);
        self.grouped_union = &.{};
        if (self.io_pool) |pool| pool.destroy();
        self.io_pool = null;
        if (self.fds.len != 0) self.allocator.free(self.fds);
        self.fds = &.{};
    }

    pub fn deinit(self: *Engine) void {
        self.stopPrefetch();
        self.drainPendingReaders();
        self.releaseSlabs();
        for (self.layers) |*layer| if (layer.active) layer.cache.deinit();
        self.allocator.free(self.layers);
        self.store.deinit();
        if (self.drop_owned) |d| {
            d.deinit();
            self.allocator.destroy(d);
        }
        if (self.warm_usage) |*u| u.deinit();
        self.* = undefined;
    }

    /// Bytes `--drop-expert` leaves unloaded: each dropped expert at its own rate group's size.
    pub fn droppedBytes(self: *const Engine) u64 {
        const d = self.geometry.drop orelse return 0;
        var total: u64 = 0;
        for (self.layers) |layer| {
            if (!layer.active) continue;
            if (layer.grouped) |gl| {
                for (0..gl.groups.count) |g| total += @as(u64, gl.groups.dropped[g]) * gl.expertBytes(g);
            } else total += @as(u64, d.n) * self.plan.expert_bytes;
        }
        return total;
    }

    pub fn cacheSlotBytes(self: *const Engine, layer: u16, slot: u16, component: Component) []u8 {
        return self.cacheSlotBytesAt(layer, slot, @backingInt(component));
    }

    pub fn cacheSlotBytesAt(self: *const Engine, layer: u16, slot: u16, ci: usize) []u8 {
        if (layer >= self.layers.len or !self.layers[layer].active or self.layers[layer].grouped != null or ci >= self.layers[layer].slabs.len)
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
        return layer < self.layers.len and self.layers[layer].active and self.layers[layer].grouped == null and self.layers[layer].cache.ready[slot];
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
        if (layer >= self.layers.len or self.layers[layer].grouped != null) return null;
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
            acc.copy_ns += layer.stats.copy_ns;
            acc.unions += layer.stats.unions;
        }
        self.last = acc;
        const scheduled = self.forward_count <= 2 or self.forward_count % BREAKDOWN_EVERY == 0;
        // At debug level every prefill chunk gets its line: a chunk is one forward, so the schedule would skip them.
        if (!scheduled and !acc.probe and !(rows > 64 and log.isDebug())) return;
        const fill_seconds = @as(f64, @floatFromInt(acc.fill_ns)) / 1e9;
        log.info("[expert-stream] forward={d} rows={d} wall_ms={d:.1} route_ms={d:.1} fill_ms={d:.1} compute_ms={d:.1} other_ms={d:.1} fill_gb={d:.3} fill_gbps={d:.2} fill_bytes_per_row={d} hits={d}/{d} slab_leaks={d} copy_ms={d:.1} unions={d}\n", .{
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
            @as(f64, @floatFromInt(acc.copy_ns)) / 1e6,
            acc.unions,
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
        if (!layer.active or layer.grouped != null or layer.slabs.len != self.store.componentCount()) return null;
        try layer.refreshSpec(self.allocator);
        var operands: [quant.component_count]mlx.mlx_array = @splat(.{ .ctx = null });
        for (layer.slabs, 0..) |operand, ci| operands[ci] = operand.array;
        return .{ .slots = layer.spec_slots, .raw = layer.spec_raw, .starved = layer.cache.swap_starved, .host = layer.spec_host, .operands = operands };
    }

    pub const GroupedSpec = struct {
        /// int32 [experts]: the global slot per expert, 0 where the host map says -1.
        slots: mlx.mlx_array,
        /// The host map the check reads: -1 where an expert is not a ready hit of its pool.
        host: []const i32,
        bank: *const Ragged,
    };

    /// A rate-grouped layer's cache map on the GPU and the bank it indexes, so a decode forward can queue the routed
    /// chain before the host reads the router ids (`specMatches` checks it). Null for a layer that is released,
    /// resident (its own GPU map) or not rate-grouped.
    pub fn specGrouped(self: *Engine, layer_index: u16) !?GroupedSpec {
        if (layer_index >= self.layers.len) return null;
        self.prefetchWait(layer_index);
        const layer = &self.layers[layer_index];
        const gl = layer.grouped orelse return null;
        if (!layer.active or layer.borrowed or layer.resident_map.ctx != null or layer.slabs.len != quant.component_count) return null;
        try layer.refreshSpec(self.allocator);
        return .{ .slots = layer.spec_slots, .host = layer.spec_host, .bank = &gl.cache_ragged };
    }

    pub fn prepareHost(self: *Engine, layer_index: u16, occurrences: []const u16) !Prepared {
        if (layer_index >= self.layers.len) return error.ExpertLayerOutOfRange;
        if (!self.layers[layer_index].active) return error.ExpertLayerAbsent;
        self.prefetchWait(layer_index);
        self.drainPendingReaders();
        return self.prepareSlab(layer_index, occurrences, false);
    }

    /// `prepareHost` for a route only the per-group prefill GEMMs read (`rate_groups.moePrefill`): a rate-grouped layer
    /// whose group overflows its pool then reads its cached experts in place instead of copying them into the union.
    pub fn prepareHostWide(self: *Engine, layer_index: u16, occurrences: []const u16) !Prepared {
        if (layer_index >= self.layers.len) return error.ExpertLayerOutOfRange;
        if (!self.layers[layer_index].active) return error.ExpertLayerAbsent;
        self.prefetchWait(layer_index);
        self.drainPendingReaders();
        return self.prepareSlab(layer_index, occurrences, true);
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


    /// Starts the prefetch thread; a no-op where one runs already.
    pub fn startPrefetch(self: *Engine) !void {
        if (self.prefetcher != null) return;
        const pf = try self.allocator.create(Prefetcher);
        errdefer self.allocator.destroy(pf);
        const serving = (self.io_pool orelse return error.MissingFillPool).opts;
        const pool = try io_mod.FillPool.create(self.allocator, .{ .workers = @min(serving.workers, FILL_WORKERS), .bounce_cap = serving.bounce_cap, .coalesce_max = COALESCE_MAX });
        errdefer pool.destroy();
        pf.* = .{ .pool = pool, .thread = undefined };
        pf.thread = try std.Thread.spawn(.{ .stack_size = 256 * 1024 }, prefetchMain, .{ self, pf });
        self.prefetcher = pf;
    }

    fn stopPrefetch(self: *Engine) void {
        const pf = self.prefetcher orelse return;
        const io = std.Io.Threaded.global_single_threaded.io();
        pf.mu.lockUncancelable(io);
        pf.quit = true;
        pf.cv.broadcast(io);
        pf.mu.unlock(io);
        pf.thread.join();
        pf.pool.destroy();
        self.allocator.destroy(pf);
        self.prefetcher = null;
    }

    /// Whether `layer` can be prefetched: a rate-grouped pool that is neither released nor resident.
    pub fn canPrefetch(self: *const Engine, layer: u16) bool {
        if (self.prefetcher == null or layer >= self.layers.len) return false;
        const state = &self.layers[layer];
        return state.grouped != null and state.active and !state.borrowed and state.resident_map.ctx == null and state.slabs.len == quant.component_count;
    }

    /// Queues the experts `layer` is expected to pick: for each of `rows` rows, the first `m` of its `n` ranked candidates
    /// (`cand`, `n` per row) that the pool does not hold. Best effort and never blocking; a full queue drops the job.
    pub fn prefetch(self: *Engine, layer: u16, cand: []const u16, rows: usize, n: usize, m: usize) void {
        const pf = self.prefetcher orelse return;
        if (!self.canPrefetch(layer) or rows == 0 or rows > Prefetcher.rows_max or n == 0 or n > Prefetcher.cand_max or cand.len != rows * n) return;
        const io = std.Io.Threaded.global_single_threaded.io();
        pf.mu.lockUncancelable(io);
        defer pf.mu.unlock(io);
        if (pf.len == Prefetcher.jobs_max) return;
        const job = &pf.jobs[(pf.head + pf.len) % Prefetcher.jobs_max];
        job.layer = layer;
        job.rows = @intCast(rows);
        job.n = @intCast(n);
        job.m = @intCast(@min(m, n));
        for (0..rows) |r| @memcpy(job.cand[r][0..n], cand[r * n ..][0..n]);
        pf.len += 1;
        _ = pf.live.fetchAdd(1, .release);
        _ = pf.jobs_queued.fetchAdd(1, .monotonic);
        pf.cv.broadcast(io);
    }

    /// Blocks until no prefetch job reads or writes `layer`'s cache.
    pub fn prefetchWait(self: *Engine, layer: u16) void {
        const pf = self.prefetcher orelse return;
        if (pf.live.load(.acquire) == 0) return;
        var clock = Clock.init();
        const io = std.Io.Threaded.global_single_threaded.io();
        pf.mu.lockUncancelable(io);
        while (prefetchTouches(pf, layer)) pf.cv.wait(io, &pf.mu) catch {};
        pf.mu.unlock(io);
        _ = pf.host_wait_ns.fetchAdd(clock.lap(), .monotonic);
    }

    fn prefetchTouches(pf: *const Prefetcher, layer: u16) bool {
        if (pf.busy == layer) return true;
        for (0..pf.len) |i| {
            if (pf.jobs[(pf.head + i) % Prefetcher.jobs_max].layer == layer) return true;
        }
        return false;
    }

    /// Blocks until every prefetch job is done.
    pub fn prefetchQuiesce(self: *Engine) void {
        const pf = self.prefetcher orelse return;
        if (pf.live.load(.acquire) == 0) return;
        const io = std.Io.Threaded.global_single_threaded.io();
        pf.mu.lockUncancelable(io);
        while (pf.len != 0 or pf.busy >= 0) pf.cv.wait(io, &pf.mu) catch {};
        pf.mu.unlock(io);
    }

    fn prefetchMain(self: *Engine, pf: *Prefetcher) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        while (true) {
            pf.mu.lockUncancelable(io);
            while (pf.len == 0 and !pf.quit) pf.cv.wait(io, &pf.mu) catch {};
            if (pf.len == 0) {
                pf.mu.unlock(io);
                return;
            }
            const job = pf.jobs[pf.head];
            pf.head = (pf.head + 1) % Prefetcher.jobs_max;
            pf.len -= 1;
            pf.busy = job.layer;
            pf.mu.unlock(io);
            var clock = Clock.init();
            self.prefetchFill(pf, &job) catch {};
            _ = pf.job_ns.fetchAdd(clock.lap(), .monotonic);
            pf.mu.lockUncancelable(io);
            pf.busy = -1;
            _ = pf.live.fetchSub(1, .release);
            pf.cv.broadcast(io);
            pf.mu.unlock(io);
        }
    }

    /// Reads the job's uncached experts into their pools. Where anything fails the pool is left as it was.
    fn prefetchFill(self: *Engine, pf: *Prefetcher, job: *const Prefetcher.Job) !void {
        const layer = &self.layers[job.layer];
        const gl = layer.grouped orelse return;
        const n = quant.component_count;
        const count: usize = gl.groups.count;
        var want: [Prefetcher.rows_max * Prefetcher.cand_max]u16 = undefined;
        var wanted: usize = 0;
        for (0..job.rows) |r| {
            var taken: usize = 0;
            for (job.cand[r][0..job.n]) |expert| {
                if (expert >= gl.expertCount() or self.isDropped(job.layer, expert)) continue;
                const g = gl.groups.groupOf(expert);
                const raw = gl.caches[g].expert_to_slot[expert - gl.groups.base[g]];
                if (raw >= 0 and gl.caches[g].ready[@intCast(raw)]) continue;
                taken += 1;
                if (std.mem.indexOfScalar(u16, want[0..wanted], expert) == null) {
                    want[wanted] = expert;
                    wanted += 1;
                }
                if (taken == job.m) break;
            }
        }
        if (wanted == 0) return;

        var snapshots: [quant.rate_group_limit]CacheSnapshot = undefined;
        var snapshotted: usize = 0;
        defer for (snapshots[0..snapshotted]) |*snapshot| snapshot.deinit(self.allocator);
        while (snapshotted < count) : (snapshotted += 1) snapshots[snapshotted] = try self.snapshotCache(&gl.caches[snapshotted]);
        var committed = false;
        errdefer if (!committed) for (0..count) |g| snapshots[g].restore(&gl.caches[g]);

        var resolutions: [quant.rate_group_limit]GroupResolution = undefined;
        var resolved: usize = 0;
        defer for (resolutions[0..resolved]) |*resolution| resolution.deinit(self.allocator);
        var local: std.ArrayList(u16) = .empty;
        defer local.deinit(self.allocator);
        while (resolved < count) {
            local.clearRetainingCapacity();
            for (want[0..wanted]) |expert| {
                if (gl.groups.groupOf(expert) == resolved) try local.append(self.allocator, expert - gl.groups.base[resolved]);
            }
            resolutions[resolved] = try gl.caches[resolved].resolve(self.allocator, local.items);
            resolved += 1;
            if (resolutions[resolved - 1].workspace) return error.PrefetchTooWide;
        }

        var begun: usize = 0;
        errdefer if (!committed) for (layer.slabs[0..begun]) |*operand| abortFill(operand);
        while (begun < n) : (begun += 1) {
            const slab = layer.slabs[begun].slab orelse return error.MissingExpertSlab;
            _ = try slab.tryBeginFill();
        }

        var spans: std.ArrayList(io_mod.FillSpan) = .empty;
        defer spans.deinit(self.allocator);
        for (resolutions[0..count], 0..) |resolution, g| {
            for (resolution.bindings) |binding| {
                if (binding.hit) continue;
                for (0..n) |ci| {
                    const source = self.store.spanAt(job.layer, gl.groups.base[g] + binding.expert, ci);
                    const dst = gl.cacheRegion(&layer.slabs[ci], g, ci, binding.slot orelse return error.ExpertBindingMissing);
                    try spans.append(self.allocator, gl.fillSpan(source, dst, g, ci));
                }
            }
        }
        {
            var plan = try io_mod.FillPlan.init(self.allocator, spans.items, COALESCE_MAX, pf.pool.opts.workers, false);
            defer plan.deinit();
            try pf.pool.run(self.fds, &plan);
        }
        for (resolutions[0..count], 0..) |resolution, g| {
            for (resolution.bindings) |binding| {
                if (!binding.hit) gl.caches[g].markReady(binding.slot.?);
            }
        }
        committed = true;
        for (layer.slabs) |*operand| abortFill(operand);
        _ = pf.jobs_run.fetchAdd(1, .monotonic);
        _ = pf.experts_read.fetchAdd(wanted, .monotonic);
    }

    pub fn cancelFills(self: *Engine) void {
        if (self.io_pool) |pool| pool.requestCancel();
    }

    pub fn warmCache(self: *Engine) !void {
        self.prefetchQuiesce();
        const warm_count = self.warmSlotsPerLayer();
        if (warm_count == 0) return;

        const occurrences = try self.allocator.alloc(u16, warm_count);
        defer self.allocator.free(occurrences);
        var ranked: std.ArrayList(u16) = .empty;
        defer ranked.deinit(self.allocator);
        var routed: usize = 0;

        for (self.layers, 0..) |*layer, layer_index| {
            if (!layer.active) continue;
            defer routed += 1;
            ranked.clearRetainingCapacity();
            if (layer.grouped) |gl| if (self.warm_usage) |u| {
                try rankedWarm(self.allocator, &ranked, gl, u.counts[routed], self.geometry.drop, @intCast(layer_index));
            };
            const picks: []const u16 = if (ranked.items.len != 0) ranked.items else blk: {
                var next: u16 = 0;
                for (occurrences) |*expert| {
                    while (self.isDropped(@intCast(layer_index), next)) next += 1;
                    expert.* = next;
                    next += 1;
                }
                break :blk occurrences;
            };
            const saved_stats = layer.stats;
            defer layer.stats = saved_stats;
            var prepared = try self.prepareHost(@intCast(layer_index), picks);
            prepared.deinit();
        }
    }

    /// MLX bytes of layer `li`'s pool (its slabs and its bank's meta table), 0 when it is released or not rate-grouped.
    pub fn poolBytes(self: *const Engine, li: u16) u64 {
        const layer = &self.layers[li];
        const gl = layer.grouped orelse return 0;
        if (layer.borrowed) return 0;
        var bytes: u64 = 0;
        for (layer.slabs) |operand| {
            if (operand.present) bytes += operand.stride * operand.count;
        }
        const meta = gl.cache_ragged.meta;
        if (meta.ctx != null) bytes += mlx.mlx_array_size(meta) * mlx.mlx_array_itemsize(meta);
        return bytes;
    }

    pub fn isBorrowed(self: *const Engine, li: u16) bool {
        return self.layers[li].borrowed;
    }

    /// The pool bytes a prefill can still release: every resident pool from `borrow_from` on, less the union workspace
    /// a resident pack must make before its first release.
    pub fn borrowableBytes(self: *const Engine) u64 {
        var bytes: u64 = 0;
        var li: usize = self.borrow_from;
        while (li < self.layers.len) : (li += 1) bytes += self.poolBytes(@intCast(li));
        return bytes -| self.unionPending();
    }

    /// The bytes the union workspace takes: its nine slabs and the meta table of each layer that may be released.
    pub fn unionBytes(self: *const Engine) u64 {
        var bytes: u64 = 0;
        var meta: u64 = 0;
        for (self.layers, 0..) |*layer, li| {
            const gl = layer.grouped orelse continue;
            if (bytes == 0) {
                for (self.union_elems) |elems| bytes += elems * 2;
                for (0..quant.component_count) |ci| {
                    if (ci % 3 != 0) bytes += @as(u64, self.geometry.kept()) * gl.pieceBytes(0, ci);
                }
            }
            if (li >= self.borrow_from) meta += @as(u64, self.geometry.kept()) * sushi_exl3.rate_groups.META_WORDS * 4;
        }
        return bytes + meta;
    }

    /// What a resident pack's first release must first pay for the union workspace; 0 once it exists.
    pub fn unionPending(self: *const Engine) u64 {
        return if (self.plan.resident and self.grouped_union.len == 0) self.unionBytes() else 0;
    }

    /// A resident pack makes the union workspace for its released layers, sized for the widest of all of them.
    fn ensureUnion(self: *Engine) !void {
        if (self.grouped_union.len != 0) return;
        var widest: ?*const GroupedLayer = null;
        for (self.layers) |*layer| if (layer.grouped) |g| {
            widest = g;
        };
        const shape = widest orelse return error.ExpertLayerNotGrouped;
        self.grouped_union = try createGroupedSlabs(self.allocator, shape, null, self.geometry.kept(), self.s, self.union_elems);
        errdefer self.dropUnion();
        var pool_buf: [quant.rate_group_limit]rate_groups_pool = undefined;
        for (self.layers[self.borrow_from..]) |*layer| if (layer.grouped) |g| {
            g.union_ragged = try Ragged.initPools(self.s, g.hidden, g.inter, g.pools(true, &pool_buf), slabTriple(self.grouped_union, 0), slabTriple(self.grouped_union, 1), slabTriple(self.grouped_union, 2));
            g.union_ragged.tile_major = g.tile_major;
        };
    }

    fn dropUnion(self: *Engine) void {
        for (self.layers) |*layer| if (layer.grouped) |g| g.union_ragged.deinit();
        for (self.grouped_union) |*operand| operand.destroy(self.allocator, self.s);
        if (self.grouped_union.len != 0) self.allocator.free(self.grouped_union);
        self.grouped_union = &.{};
    }

    /// Reads every routed expert of a resident pack into its pool and maps each expert to its slot.
    fn loadResident(self: *Engine) !void {
        const io = std.Io.Threaded.global_single_threaded.io();
        const t0 = std.Io.Timestamp.now(io, .awake);
        const all = try self.allocator.alloc(u16, self.geometry.kept());
        defer self.allocator.free(all);
        const before = self.fill_bytes_total;
        for (self.layers, 0..) |*layer, li| {
            if (layer.grouped == null) continue;
            const stats = layer.stats;
            defer layer.stats = stats;
            var next: usize = 0;
            for (0..self.geometry.experts) |e| {
                if (self.isDropped(@intCast(li), @intCast(e))) continue;
                all[next] = @intCast(e);
                next += 1;
            }
            var prepared = try self.prepareHost(@intCast(li), all);
            prepared.deinit();
            try self.buildResidentMap(@intCast(li));
        }
        const ms: u64 = @intCast(@divTrunc(t0.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms));
        log.info("[expert-stream] resident: read {d:.3} GB of routed experts in {d} ms; routes read no SSD\n", .{ @as(f64, @floatFromInt(self.fill_bytes_total - before)) / 1e9, ms });
    }

    fn buildResidentMap(self: *Engine, li: u16) !void {
        const layer = &self.layers[li];
        const gl = layer.grouped orelse return error.ExpertLayerNotGrouped;
        const host = try self.allocator.alloc(i32, self.geometry.experts);
        defer self.allocator.free(host);
        for (host, 0..) |*slot, e| {
            // A dropped expert has no slot; the router never picks it.
            if (self.isDropped(li, @intCast(e))) {
                slot.* = -1;
                continue;
            }
            const g = gl.groups.groupOf(@intCast(e));
            const local = gl.caches[g].expert_to_slot[e - gl.groups.base[g]];
            if (local < 0 or !gl.caches[g].ready[@intCast(local)]) return error.ResidentExpertMissing;
            slot.* = @as(i32, @intCast(gl.slot_base[g])) + local;
        }
        if (layer.resident_map.ctx != null) _ = mlx.mlx_array_free(layer.resident_map);
        layer.resident_map = mlx.mlx_array_new_data(host.ptr, &[_]c_int{@intCast(host.len)}, 1, .int32);
    }

    /// The lossy pick on a rate-grouped layer: a missed expert may take any expert already in one of the layer's pools,
    /// read through that expert's own group, with the missed expert's weight (`substituteMisses`). Returns the swaps.
    pub fn swapGroupedMisses(self: *Engine, li: u16, ids: []u16, scores: []const f32, tolerance: f32, k: usize) !u32 {
        self.prefetchWait(li);
        const layer = &self.layers[li];
        const gl = layer.grouped orelse return 0;
        if (layer.borrowed) return 0;
        const experts: usize = self.geometry.experts;
        if (k == 0 or scores.len != ids.len / k * experts) return 0;
        if (layer.starved.len != experts) {
            if (layer.starved.len != 0) self.allocator.free(layer.starved);
            layer.starved = &.{};
            layer.starved = try self.allocator.alloc(u8, experts);
            @memset(layer.starved, 0);
        }
        const flags = try self.allocator.alloc(bool, experts * 3);
        defer self.allocator.free(flags);
        for (flags[0..experts], 0..) |*cached, e| {
            const g = gl.groups.groupOf(@intCast(e));
            const slot = gl.caches[g].expert_to_slot[e - gl.groups.base[g]];
            cached.* = slot >= 0 and gl.caches[g].ready[@intCast(slot)];
        }
        const swapped = substituteMisses(ids, scores, flags[0..experts], tolerance, k, flags[experts .. 2 * experts], flags[2 * experts ..], layer.starved);
        if (self.swap_seen_ids == 0) log.info("[expert-swap] engaged on rate-grouped layers: tolerance={d:.2} (lossy: a missed expert is replaced by a pooled one within that relative score)\n", .{tolerance});
        const first = swapped > 0 and self.swapped_ids == 0;
        self.swapped_ids += swapped;
        self.swap_seen_ids += ids.len;
        if (first or self.swap_seen_ids -% self.swap_reported_at >= 100_000) {
            self.swap_reported_at = self.swap_seen_ids;
            log.info("[expert-swap] swapped {d} of {d} routed ids so far\n", .{ self.swapped_ids, self.swap_seen_ids });
        }
        return swapped;
    }

    pub const ResidentRoute = struct { map: mlx.mlx_array, bank: *const Ragged };

    /// A resident layer's route needs no host read: the GPU maps the router's ids through `map` into `bank`. Null for a
    /// streamed or released layer.
    pub fn residentRoute(self: *const Engine, li: u16) ?ResidentRoute {
        const layer = &self.layers[li];
        if (layer.borrowed or layer.resident_map.ctx == null) return null;
        return .{ .map = layer.resident_map, .bank = &layer.grouped.?.cache_ragged };
    }

    pub fn publishBorrowable(self: *const Engine) void {
        if (self.borrow_publish) |dst| @atomicStore(u64, dst, self.borrowableBytes(), .monotonic);
    }

    /// Releases rate-grouped layer `li`'s pool: its experts are remembered, its bank dropped and its slabs unmapped
    /// once MLX lets go of them. Until `restoreLayer` its routes read through the union workspace. Returns the bytes
    /// MLX gave back (a slab whose release timed out stays mapped and is not counted).
    pub fn borrowLayer(self: *Engine, li: u16) !u64 {
        self.prefetchWait(li);
        const layer = &self.layers[li];
        const gl = layer.grouped orelse return error.ExpertLayerNotGrouped;
        if (layer.borrowed) return 0;
        self.drainPendingReaders();
        _ = mlx.mlx_synchronize(self.s);
        var saved: std.ArrayList(u16) = .empty;
        errdefer saved.deinit(self.allocator);
        for (gl.caches[0..gl.caches_made], 0..) |*cache, g| {
            for (cache.slot_to_expert, cache.ready) |expert, ready| {
                if (ready and expert != std.math.maxInt(u16)) try saved.append(self.allocator, @intCast(gl.groups.base[g] + expert));
            }
        }
        layer.saved = try saved.toOwnedSlice(self.allocator);
        const meta = gl.cache_ragged.meta;
        var freed: u64 = if (meta.ctx != null) mlx.mlx_array_size(meta) * mlx.mlx_array_itemsize(meta) else 0;
        gl.cache_ragged.deinit();
        gl.deinitCaches();
        for (layer.slabs) |*operand| {
            if (!operand.present) continue;
            const bytes = operand.stride * operand.count;
            const timeouts = slab_release_timeouts.load(.monotonic);
            operand.destroy(self.allocator, self.s);
            if (slab_release_timeouts.load(.monotonic) == timeouts) freed += bytes;
        }
        if (layer.slabs.len != 0) self.allocator.free(layer.slabs);
        layer.slabs = &.{};
        if (layer.resident_map.ctx != null) _ = mlx.mlx_array_free(layer.resident_map);
        layer.resident_map = .{ .ctx = null };
        layer.borrowed = true;
        layer.lent = freed;
        // Released first, so the workspace comes out of the pool it replaces.
        try self.ensureUnion();
        self.publishBorrowable();
        return freed;
    }

    pub fn lentBytes(self: *const Engine, li: u16) u64 {
        return if (self.layers[li].borrowed) self.layers[li].lent else 0;
    }

    /// Gives a released layer its pool back and reads into it the experts it held when released.
    pub fn restoreLayer(self: *Engine, li: u16) !void {
        self.prefetchWait(li);
        const layer = &self.layers[li];
        const gl = layer.grouped orelse return error.ExpertLayerNotGrouped;
        if (!layer.borrowed) return;
        try self.rebuildPool(layer, gl);
        layer.borrowed = false;
        self.publishBorrowable();
        const saved = layer.saved;
        layer.saved = &.{};
        defer self.allocator.free(saved);
        if (saved.len != 0) {
            const stats = layer.stats;
            defer layer.stats = stats;
            var prepared = try self.prepareHost(li, saved);
            prepared.deinit();
        }
        if (!self.plan.resident) return;
        try self.buildResidentMap(li);
        for (self.layers) |*other| if (other.borrowed) return;
        self.drainPendingReaders();
        _ = mlx.mlx_synchronize(self.s);
        self.dropUnion();
        self.publishBorrowable();
    }

    fn rebuildPool(self: *Engine, layer: *LayerState, gl: *GroupedLayer) !void {
        layer.slabs = try createGroupedSlabs(self.allocator, gl, gl.slots[0..gl.groups.count], gl.cache_slots, self.s, null);
        errdefer {
            for (layer.slabs) |*operand| operand.destroy(self.allocator, self.s);
            self.allocator.free(layer.slabs);
            layer.slabs = &.{};
        }
        try gl.buildCacheBank(self.s, layer.slabs);
        errdefer gl.cache_ragged.deinit();
        try gl.initCaches(self.allocator);
    }

    /// A released layer's route: each distinct pick is read into the union workspace at its group position and the
    /// union bank serves the route.
    fn prepareBorrowed(self: *Engine, layer_index: u16, gl: *GroupedLayer, occurrences: []const u16) !Prepared {
        const layer = &self.layers[layer_index];
        const n = quant.component_count;
        if (self.grouped_union.len != n) return error.MissingExpertSlab;
        const position = try self.allocator.alloc(i32, self.geometry.experts);
        defer self.allocator.free(position);
        @memset(position, -1);
        var begun: usize = 0;
        var committed = false;
        errdefer if (!committed) for (self.grouped_union[0..begun]) |*operand| abortFill(operand);
        while (begun < n) : (begun += 1) {
            const slab = self.grouped_union[begun].slab orelse return error.MissingExpertSlab;
            _ = try slab.tryBeginFill();
        }
        var spans: std.ArrayList(io_mod.FillSpan) = .empty;
        defer spans.deinit(self.allocator);
        var next: [quant.rate_group_limit]u32 = @splat(0);
        for (occurrences) |expert| {
            if (position[expert] >= 0) continue;
            const g = gl.groups.groupOf(expert);
            position[expert] = @intCast(next[g]);
            for (0..n) |ci| {
                const source = self.store.spanAt(layer_index, expert, ci);
                const dst = gl.unionRegion(&self.grouped_union[ci], g, ci, next[g]);
                try spans.append(self.allocator, gl.fillSpan(source, dst, g, ci));
            }
            next[g] += 1;
        }
        try self.runFill(layer_index, spans.items);
        var distinct: u64 = 0;
        for (next[0..gl.groups.count]) |count| distinct += count;
        self.fill_experts_total += distinct;

        var held: [HELD_MAX]?HeldLease = @splat(null);
        var leased: usize = 0;
        errdefer for (held[0..leased]) |entry| {
            const taken = entry orelse continue;
            taken.slab.release(taken.lease) catch {};
            taken.slab.retire() catch {};
        };
        while (leased < n) : (leased += 1) {
            const slab = self.grouped_union[leased].slab.?;
            _ = try slab.publish();
            held[leased] = .{ .slab = slab, .lease = try slab.lease() };
        }
        const remapped = try self.allocator.alloc(u16, occurrences.len);
        for (occurrences, remapped) |expert, *slot| {
            const g = gl.groups.groupOf(expert);
            slot.* = @intCast(gl.union_base[g] + @as(u32, @intCast(position[expert])));
        }
        layer.stats.unions += 1;
        layer.stats.union_members += distinct;
        layer.stats.misses += distinct;
        layer.stats.groups += 1;
        for (spans.items) |span_value| layer.stats.fill_bytes += span_value.len;
        committed = true;
        return .{
            .allocator = self.allocator,
            .remapped = remapped,
            .quantized = true,
            .workspace = true,
            .ragged = &gl.union_ragged,
            .held = held,
            .defer_to = self,
        };
    }

    /// `prepareSlab` for a rate-grouped layer: each group resolves against its own pool, and one group overflowing
    /// its pool sends the whole layer through the union workspace. Ids in `remapped` are global slot ids of the
    /// layer's `Ragged` bank: pool g's slots, or its union positions, follow those of group g-1.
    fn prepareGrouped(self: *Engine, layer_index: u16, gl: *GroupedLayer, occurrences: []const u16, wide: bool) !Prepared {
        const layer = &self.layers[layer_index];
        const n = quant.component_count;
        const count: usize = gl.groups.count;
        if (layer.slabs.len != n) return error.MissingExpertSlab;

        var snapshots: [quant.rate_group_limit]CacheSnapshot = undefined;
        var snapshotted: usize = 0;
        defer for (snapshots[0..snapshotted]) |*snapshot| snapshot.deinit(self.allocator);
        while (snapshotted < count) : (snapshotted += 1) snapshots[snapshotted] = try self.snapshotCache(&gl.caches[snapshotted]);
        var committed = false;
        var wrote_cache = false;
        errdefer if (!committed) for (0..count) |g| {
            if (wrote_cache) gl.caches[g].invalidateUnready() else snapshots[g].restore(&gl.caches[g]);
        };

        var resolutions: [quant.rate_group_limit]GroupResolution = undefined;
        var resolved: usize = 0;
        defer for (resolutions[0..resolved]) |*resolution| resolution.deinit(self.allocator);
        var use_union = false;
        {
            var local: std.ArrayList(u16) = .empty;
            defer local.deinit(self.allocator);
            while (resolved < count) {
                local.clearRetainingCapacity();
                for (occurrences) |expert| {
                    if (gl.groups.groupOf(expert) == resolved) try local.append(self.allocator, expert - gl.groups.base[resolved]);
                }
                resolutions[resolved] = try gl.caches[resolved].resolve(self.allocator, local.items);
                use_union = use_union or resolutions[resolved].workspace;
                resolved += 1;
            }
        }
        if (use_union and self.grouped_union.len != n) return error.MissingExpertSlab;
        if (use_union and wide) {
            wrote_cache = true;
            const split = try self.prepareSplit(layer_index, gl, occurrences, resolutions[0..count]);
            committed = true;
            return split;
        }
        const read_set = if (use_union) self.grouped_union else layer.slabs;

        var read_begun: usize = 0;
        errdefer if (!committed) for (read_set[0..read_begun]) |*operand| abortFill(operand);
        while (read_begun < n) : (read_begun += 1) {
            const slab = read_set[read_begun].slab orelse return error.MissingExpertSlab;
            _ = try slab.tryBeginFill();
        }
        var cache_begun: usize = 0;
        errdefer if (!committed) for (layer.slabs[0..cache_begun]) |*operand| abortFill(operand);
        if (use_union) {
            while (cache_begun < n) : (cache_begun += 1) {
                const slab = layer.slabs[cache_begun].slab orelse return error.MissingExpertSlab;
                _ = try slab.tryBeginFill();
            }
        }

        var spans: std.ArrayList(io_mod.FillSpan) = .empty;
        defer spans.deinit(self.allocator);
        var misses: u64 = 0;
        var copy_clock = Clock.init();
        if (use_union) layer.stats.unions += 1;
        for (resolutions[0..count], 0..) |resolution, g| {
            for (resolution.bindings, 0..) |binding, position| {
                if (binding.hit and use_union) {
                    const slot = binding.slot orelse return error.ExpertBindingMissing;
                    for (0..n) |ci| {
                        const dst = gl.unionRegion(&read_set[ci], g, ci, position);
                        @memcpy(dst, gl.cacheRegion(&layer.slabs[ci], g, ci, slot));
                        self.copy_bytes_total += dst.len;
                    }
                    continue;
                }
                if (binding.hit) continue;
                misses += 1;
                for (0..n) |ci| {
                    const source = self.store.spanAt(layer_index, gl.groups.base[g] + binding.expert, ci);
                    const dst = if (use_union)
                        gl.unionRegion(&read_set[ci], g, ci, position)
                    else
                        gl.cacheRegion(&layer.slabs[ci], g, ci, binding.slot orelse return error.ExpertBindingMissing);
                    try spans.append(self.allocator, gl.fillSpan(source, dst, g, ci));
                }
            }
        }
        if (!use_union and spans.items.len > 0) wrote_cache = true;
        const copied = copy_clock.lap();
        layer.stats.copy_ns += copied;
        self.copy_ns_total += copied;
        try self.runFill(layer_index, spans.items);
        self.fill_experts_total += misses;

        if (use_union) {
            _ = copy_clock.lap();
            defer {
                const back = copy_clock.lap();
                layer.stats.copy_ns += back;
                self.copy_ns_total += back;
            }
            for (resolutions[0..count], 0..) |resolution, g| {
                for (resolution.bindings, 0..) |binding, position| {
                    if (binding.hit) continue;
                    const slot = binding.slot orelse continue;
                    wrote_cache = true;
                    for (0..n) |ci| {
                        const dst = gl.cacheRegion(&layer.slabs[ci], g, ci, slot);
                        @memcpy(dst, gl.unionRegion(&read_set[ci], g, ci, position));
                        self.copy_bytes_total += dst.len;
                    }
                    gl.caches[g].markReady(slot);
                }
            }
            for (layer.slabs) |*operand| {
                const slab = operand.slab orelse return error.MissingExpertSlab;
                _ = try slab.publish();
                try slab.retire();
            }
        } else for (resolutions[0..count], 0..) |resolution, g| {
            for (resolution.bindings) |binding| {
                if (!binding.hit) gl.caches[g].markReady(binding.slot.?);
            }
        }

        var held: [HELD_MAX]?HeldLease = @splat(null);
        var leased: usize = 0;
        errdefer for (held[0..leased]) |entry| {
            const taken = entry orelse continue;
            taken.slab.release(taken.lease) catch {};
            taken.slab.retire() catch {};
        };
        while (leased < n) : (leased += 1) {
            const slab = read_set[leased].slab orelse return error.MissingExpertSlab;
            _ = try slab.publish();
            held[leased] = .{ .slab = slab, .lease = try slab.lease() };
        }

        const remapped = try self.allocator.alloc(u16, occurrences.len);
        errdefer self.allocator.free(remapped);
        var next_local: [quant.rate_group_limit]usize = @splat(0);
        for (occurrences, 0..) |expert, i| {
            const g = gl.groups.groupOf(expert);
            const local = expert - gl.groups.base[g];
            if (use_union) {
                const resolution = resolutions[g];
                remapped[i] = @intCast(gl.union_base[g] + resolution.remapped[next_local[g]]);
                next_local[g] += 1;
            } else {
                const raw = gl.caches[g].expert_to_slot[local];
                if (raw < 0) return error.ExpertBindingMissing;
                remapped[i] = @intCast(gl.slot_base[g] + @as(u32, @intCast(raw)));
            }
        }

        for (resolutions[0..count]) |resolution| {
            layer.stats.union_members += resolution.union_ids.len;
            layer.stats.hits += resolution.hits;
            layer.stats.misses += resolution.misses;
        }
        layer.stats.groups += 1;
        for (spans.items) |span_value| layer.stats.fill_bytes += span_value.len;
        committed = true;
        return .{
            .allocator = self.allocator,
            .remapped = remapped,
            .quantized = true,
            .workspace = use_union,
            .ragged = if (use_union) &gl.union_ragged else &gl.cache_ragged,
            .held = held,
            .defer_to = self,
        };
    }

    /// The union path of a wide route: each group keeps its hits and admitted misses in its pool, its declined misses
    /// are read into the union workspace, and `Ragged.initSplit` serves both where they lie. The caller restores the
    /// caches on an error.
    fn prepareSplit(self: *Engine, layer_index: u16, gl: *GroupedLayer, occurrences: []const u16, resolutions: []const GroupResolution) !Prepared {
        const layer = &self.layers[layer_index];
        const n = quant.component_count;
        var overflow: [quant.rate_group_limit]u32 = @splat(0);
        var cache_base: [quant.rate_group_limit]u32 = @splat(0);
        var spill_base: [quant.rate_group_limit]u32 = @splat(0);
        var at: u32 = 0;
        var experts: usize = 0;
        for (resolutions, 0..) |resolution, g| {
            overflow[g] = @intCast(resolution.declined);
            cache_base[g] = at;
            spill_base[g] = at + gl.slots[g];
            at += gl.slots[g] + overflow[g];
            experts += gl.groups.experts[g];
        }
        const sets = [2][]SlabOperand{ layer.slabs, self.grouped_union };
        var begun: [2]usize = @splat(0);
        errdefer for (sets, begun) |set, done| for (set[0..done]) |*operand| abortFill(operand);
        for (0..2) |k| while (begun[k] < n) : (begun[k] += 1) {
            const slab = sets[k][begun[k]].slab orelse return error.MissingExpertSlab;
            _ = try slab.tryBeginFill();
        };

        const to_slot = try self.allocator.alloc(u16, experts);
        defer self.allocator.free(to_slot);
        var spans: std.ArrayList(io_mod.FillSpan) = .empty;
        defer spans.deinit(self.allocator);
        var misses: u64 = 0;
        for (resolutions, 0..) |resolution, g| {
            var spilled: u32 = 0;
            for (resolution.bindings) |binding| {
                const global = gl.groups.base[g] + binding.expert;
                to_slot[global] = @intCast(if (binding.slot) |slot| cache_base[g] + slot else spill_base[g] + spilled);
                if (binding.hit) continue;
                misses += 1;
                for (0..n) |ci| {
                    const source = self.store.spanAt(layer_index, global, ci);
                    const dst = if (binding.slot) |slot| gl.cacheRegion(&layer.slabs[ci], g, ci, slot) else gl.unionRegion(&self.grouped_union[ci], g, ci, spilled);
                    try spans.append(self.allocator, gl.fillSpan(source, dst, g, ci));
                }
                if (binding.slot == null) spilled += 1;
            }
        }
        try self.runFill(layer_index, spans.items);
        self.fill_experts_total += misses;
        for (resolutions, 0..) |resolution, g| {
            for (resolution.bindings) |binding| {
                if (!binding.hit) if (binding.slot) |slot| gl.caches[g].markReady(slot);
            }
        }

        var held: [HELD_MAX]?HeldLease = @splat(null);
        var leased: usize = 0;
        errdefer for (held[0..leased]) |entry| {
            const taken = entry orelse continue;
            taken.slab.release(taken.lease) catch {};
            taken.slab.retire() catch {};
        };
        for (sets) |set| for (set) |operand| {
            const slab = operand.slab orelse return error.MissingExpertSlab;
            _ = try slab.publish();
            held[leased] = .{ .slab = slab, .lease = try slab.lease() };
            leased += 1;
        };

        const bank = try self.allocator.create(Ragged);
        errdefer self.allocator.destroy(bank);
        bank.* = try Ragged.initSplit(self.s, &gl.cache_ragged, &gl.union_ragged, overflow[0..resolutions.len]);
        errdefer bank.deinit();
        const remapped = try self.allocator.alloc(u16, occurrences.len);
        for (occurrences, remapped) |expert, *slot| slot.* = to_slot[expert];

        layer.stats.unions += 1;
        for (resolutions) |resolution| {
            layer.stats.union_members += resolution.union_ids.len;
            layer.stats.hits += resolution.hits;
            layer.stats.misses += resolution.misses;
        }
        layer.stats.groups += 1;
        for (spans.items) |span_value| layer.stats.fill_bytes += span_value.len;
        return .{
            .allocator = self.allocator,
            .remapped = remapped,
            .quantized = true,
            .workspace = true,
            .ragged = bank,
            .split = bank,
            .held = held,
            .defer_to = self,
        };
    }

    /// Whether `--drop-expert` left this layer's expert unloaded: it has no slot and no bytes are ever read for it.
    pub fn isDropped(self: *const Engine, layer: u16, expert: u16) bool {
        const d = self.geometry.drop orelse return false;
        return d.isDropped(layer, expert);
    }

    fn prepareSlab(self: *Engine, layer_index: u16, occurrences: []const u16, wide: bool) !Prepared {
        const layer = &self.layers[layer_index];
        if (!layer.active) return error.ExpertLayerAbsent;
        for (occurrences) |expert| if (self.isDropped(layer_index, expert)) return error.ExpertDropped;
        if (layer.grouped) |grouped| return if (layer.borrowed)
            self.prepareBorrowed(layer_index, grouped, occurrences)
        else
            self.prepareGrouped(layer_index, grouped, occurrences, wide);
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

        var held: [HELD_MAX]?HeldLease = @splat(null);
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

const expectError = @import("test_expect.zig").expectError;

test "a lossy pick on a resident load is named, never silent" {
    try std.testing.expect(pickInertLine(true, 0.2) != null);
    try std.testing.expect(pickInertLine(true, 0) == null);
    try std.testing.expect(pickInertLine(false, 0.2) == null);
}

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
    const p = try cachePlanBytes(60_000_000_000, 48, 512, try expertBytes(1280, 2560, 640), .{});
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

test "expert stream rate-grouped pools stay within the layer's bill and pin a group that cannot afford its share at one slot" {
    const t = std.testing;
    var groups: quant.LayerGroups = .{ .count = 3 };
    groups.experts[0..3].* = .{ 1, 40, 200 };
    groups.n[0] = .{ 48, 48, 48 };
    groups.n[1] = .{ 32, 32, 40 };
    groups.n[2] = .{ 24, 24, 24 };
    var layer: GroupedLayer = .{ .groups = groups, .hidden = 2560, .inter = 640 };
    var all: u64 = 0;
    var minimum: u64 = 0;
    for (0..3) |g| {
        all += @as(u64, groups.experts[g]) * layer.expertBytes(g);
        minimum += layer.expertBytes(g);
    }
    const average = (all + 240) / 241;

    for ([_]u64{ 16, 40, 120, 240 }) |slots| {
        try layer.sizePools(slots * average);
        var held: u64 = 0;
        for (0..3) |g| {
            try t.expect(layer.slots[g] >= 1 and layer.slots[g] <= groups.experts[g]);
            held += @as(u64, layer.slots[g]) * layer.expertBytes(g);
        }
        try t.expect(held <= slots * average);
    }
    try layer.sizePools(241 * average);
    try t.expectEqualSlices(u32, &.{ 1, 40, 200 }, layer.slots[0..3]);

    try layer.sizePools(minimum);
    try t.expectEqualSlices(u32, &.{ 1, 1, 1 }, layer.slots[0..3]);
    try t.expectError(error.ExpertCacheTooSmall, layer.sizePools(minimum - 1));
}

test "expert stream a drop counts its experts out of the plan: workspace, slot cap and the ledger" {
    const t = std.testing;
    const per = try expertBytes(1280, 2560, 640);
    const full = try cachePlanBytes(1 << 40, 48, 512, per, .{});
    const cut = try cachePlanBytesForGeometry(1 << 40, .{ .layers = 48, .experts = 512, .hidden = 2560, .intermediate = 640, .drop_n = 16 }, per);
    try t.expectEqual(@as(u16, 512), full.slots_per_layer);
    try t.expectEqual(@as(u16, 496), cut.slots_per_layer);
    try t.expectEqual(full.workspace_bytes - 16 * per, cut.workspace_bytes);
    const GiB: u64 = 1 << 30;
    const a = try budgetLedger(100 * GiB, 9_900_000_000, 0, 0, 48, 512, 10, per, BOUNCE_BYTES, .{});
    const b = try budgetLedger(100 * GiB, 9_900_000_000, 0, 0, 48, 496, 10, per, BOUNCE_BYTES, .{});
    try t.expectEqual(a.workspace_bytes - 16 * per, b.workspace_bytes);
    try t.expect(b.slots_per_layer >= a.slots_per_layer);
}

test "expert stream rate-grouped pools and the union hold the kept experts of each group only" {
    const t = std.testing;
    var groups: quant.LayerGroups = .{ .count = 3 };
    groups.experts[0..3].* = .{ 3, 30, 100 };
    groups.dropped[0..3].* = .{ 0, 10, 6 };
    groups.n[0] = .{ 48, 48, 48 };
    groups.n[1] = .{ 32, 32, 40 };
    groups.n[2] = .{ 24, 24, 24 };
    groups.base[0..3].* = .{ 0, 3, 33 };
    var layer = try GroupedLayer.init(t.allocator, groups, 2560, 640, 1 << 40, false);
    defer layer.deinitCaches();
    // A bill that holds everything caps every pool at its kept experts, never its stored ones.
    try t.expectEqualSlices(u32, &.{ 3, 20, 94 }, layer.slots[0..3]);
    try t.expectEqualSlices(u32, &.{ 0, 3, 23 }, layer.union_base[0..3]);
    var pools: [quant.rate_group_limit]rate_groups_pool = undefined;
    const union_pools = layer.pools(true, &pools);
    try t.expectEqual(@as(u32, 3), union_pools[0].slots);
    try t.expectEqual(@as(u32, 20), union_pools[1].slots);
    try t.expectEqual(@as(u32, 94), union_pools[2].slots);
    var undropped = groups;
    undropped.dropped = @splat(0);
    var whole: GroupedLayer = .{ .groups = undropped, .hidden = 2560, .inter = 640 };
    var kept_elems: u64 = 0;
    var whole_elems: u64 = 0;
    for (0..3) |g| {
        kept_elems += @as(u64, groups.keptOf(g)) * layer.trellisBytes(g, 0);
        whole_elems += @as(u64, groups.experts[g]) * whole.trellisBytes(g, 0);
    }
    try t.expect(kept_elems < whole_elems);
    try t.expectEqual(@as(u64, kept_elems), layer.union_off[2][0] + @as(u64, groups.keptOf(2)) * layer.trellisBytes(2, 0));
    // A tight bill splits by kept bytes and still pins a group at one slot.
    var min_bill: u64 = 0;
    for (0..3) |g| min_bill += layer.expertBytes(g);
    try layer.sizePools(min_bill);
    try t.expectEqualSlices(u32, &.{ 1, 1, 1 }, layer.slots[0..3]);
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

test "only a Qwen EXL3 pack or a DeepSeek-V4.1 checkpoint or STQ pack keeps its MTP head under streaming" {
    const t = std.testing;
    try t.expect(streamedMtpHeadSupported(true, false, .exl3_k4));
    try t.expect(!streamedMtpHeadSupported(true, false, .bf16_fused));
    try t.expect(!streamedMtpHeadSupported(false, false, .exl3_k4));
    try t.expect(!streamedMtpHeadSupported(false, false, .mxfp4_individual));
    try t.expect(streamedMtpHeadSupported(false, true, .mxfp4_w123));
    try t.expect(streamedMtpHeadSupported(false, true, .exl3_k4));
    try t.expect(!streamedMtpHeadSupported(false, true, .mxfp4_individual));
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
    try t.expect(!expertStreamingEngaged(false, false, 0));
    try t.expect(!expertStreamingEngaged(false, true, 60 << 30));
    // The dense HF checkpoint cannot load resident: it streams with no budget.
    try t.expect(expertStreamingEngaged(true, true, 0));
    // A quantized pack with no budget loads resident.
    try t.expect(!expertStreamingEngaged(true, false, 0));
    try t.expect(expertStreamingEngaged(true, false, 50 << 30));
}

test "expert stream ssd budget ledger derives the cache and refuses by name" {
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const expert_bytes: u64 = 9_830_400;
    const bounce: u64 = 8 * 64 * 1024 * 1024;
    const trunk: u64 = 9_900_000_000;

    const led = try budgetLedger(100 * GiB, trunk, 0, 0, 48, 512, 10, expert_bytes, bounce, .{});
    try t.expectEqual(@as(u64, 512 * expert_bytes), led.workspace_bytes);
    try t.expectEqual(@as(u64, 10 * expert_bytes), led.selected_bytes);
    try t.expectEqual(@as(u64, 0), led.mtp_bytes);
    try t.expectEqual(trunk, led.trunk_bytes);
    try t.expectEqual(bounce, led.bounce_bytes);
    try t.expectEqual(@as(u16, 194), led.slots_per_layer);
    try t.expectEqual(@as(u64, 194) * 48 * expert_bytes, led.cache_bytes);

    const with_mtp = try budgetLedger(100 * GiB, trunk, 5_200_000_000, 0, 48, 512, 10, expert_bytes, bounce, .{});
    try t.expectEqual(@as(u16, 183), with_mtp.slots_per_layer);

    // The vision tower and its encode reserve come out of the budget before the expert slots.
    const with_vision = try budgetLedger(100 * GiB, trunk, 0, 5_200_000_000, 48, 512, 10, expert_bytes, bounce, .{});
    try t.expectEqual(@as(u64, 5_200_000_000), with_vision.vision_bytes);
    try t.expectEqual(@as(u16, 183), with_vision.slots_per_layer);

    try t.expectError(error.SsdBudgetBelowResident, budgetLedger(14 * GiB, trunk, 0, 0, 48, 512, 10, expert_bytes, bounce, .{}));
    const fixed = trunk + 512 * expert_bytes + 10 * expert_bytes + bounce;
    const per_slot = 48 * expert_bytes;
    try t.expectError(error.SsdBudgetBelowResident, budgetLedger(fixed + 2 * per_slot - 1, trunk, 0, 0, 48, 512, 10, expert_bytes, bounce, .{}));
    const two = try budgetLedger(fixed + 2 * per_slot, trunk, 0, 0, 48, 512, 10, expert_bytes, bounce, .{});
    try t.expectEqual(@as(u16, 2), two.slots_per_layer);
    try t.expectError(error.SsdBudgetBelowResident, budgetLedger(fixed + 2 * per_slot, trunk, 0, 1, 48, 512, 10, expert_bytes, bounce, .{}));
}

test "expert stream: a budget that holds every routed expert beside the trunk serves them resident, with no union, selected slab or bounce" {
    const t = std.testing;
    const expert_bytes: u64 = 9_000_000;
    const trunk: u64 = 5_000_000_000;
    const mtp: u64 = 4_000_000_000;
    const shape: quant.StoreShape = .{ .layer_floor_bytes = 30_000_000, .union_meta_bytes = 40 * 384 * 32, .all_bytes = 119_000_000_000 };
    const led = try budgetLedger(trunk + mtp + shape.all_bytes, trunk, mtp, 0, 40, 384, 6, expert_bytes, BOUNCE_BYTES, shape);
    try t.expect(led.resident);
    try t.expectEqual(shape.all_bytes, led.cache_bytes);
    try t.expectEqual(@as(u16, 384), led.slots_per_layer);
    try t.expectEqual(@as(u64, 0), led.workspace_bytes + led.selected_bytes + led.bounce_bytes);
    // A byte short streams: the cache is cut from what the union, selected slab and bounce leave.
    const short = try budgetLedger(trunk + mtp + shape.all_bytes - 1, trunk, mtp, 0, 40, 384, 6, expert_bytes, BOUNCE_BYTES, shape);
    try t.expect(!short.resident);
    try t.expect(short.workspace_bytes > 0 and short.cache_bytes < shape.all_bytes);
    // A store with any uniform layer names no resident size and always streams.
    try t.expect(!(try budgetLedger(500 * (1 << 30), trunk, mtp, 0, 40, 384, 6, expert_bytes, BOUNCE_BYTES, .{ .layer_floor_bytes = 30_000_000 })).resident);

    const plan = try cachePlanBytes(led.cache_bytes, 40, 384, expert_bytes, shape);
    try t.expect(plan.resident);
    try t.expectEqual(shape.all_bytes, plan.cache_bytes);
    try t.expectEqual(@as(u64, 0), plan.workspace_bytes + plan.bounce_bytes);
    try t.expect(!(try cachePlanBytes(short.cache_bytes, 40, 384, expert_bytes, shape)).resident);
}

test "expert stream ssd budget ledger refuses a grouped store below one expert of every rate group" {
    const t = std.testing;
    const expert_bytes: u64 = 1_380_000;
    const bounce: u64 = BOUNCE_BYTES;
    const trunk: u64 = 5_250_000_000;
    // 13 rate groups, each expert wider than the mean: the layer's floor is 17.96 MB, 14 billed slots.
    const shape: quant.StoreShape = .{ .layer_floor_bytes = 17_960_000, .union_sets = 1 };
    const fixed = trunk + 512 * expert_bytes + 10 * expert_bytes + bounce;
    const per_slot = 48 * expert_bytes;
    try t.expectError(error.SsdBudgetBelowResident, budgetLedger(fixed + 13 * per_slot, trunk, 0, 0, 48, 512, 10, expert_bytes, bounce, shape));
    const at_floor = try budgetLedger(fixed + 14 * per_slot, trunk, 0, 0, 48, 512, 10, expert_bytes, bounce, shape);
    try t.expectEqual(@as(u16, 14), at_floor.slots_per_layer);
    // A uniform store has no floor beyond its two slots.
    const uniform = try budgetLedger(fixed + 13 * per_slot, trunk, 0, 0, 48, 512, 10, expert_bytes, bounce, .{});
    try t.expectEqual(@as(u16, 13), uniform.slots_per_layer);
}

test "expert stream ssd budget ledger bills the MLX pool beside the fixed terms and the cache gives it up" {
    const t = std.testing;
    const GiB: u64 = 1 << 30;
    const expert_bytes: u64 = 9_830_400;
    const trunk: u64 = 9_900_000_000;
    const none = try budgetLedger(100 * GiB, trunk, 0, 0, 48, 512, 10, expert_bytes, BOUNCE_BYTES, .{});
    const pooled = try budgetLedger(100 * GiB, trunk, 0, 0, 48, 512, 10, expert_bytes, BOUNCE_BYTES, .{ .pool_bytes = 6 * GiB });
    try t.expectEqual(@as(u64, 0), none.pool_bytes);
    try t.expectEqual(6 * GiB, pooled.pool_bytes);
    try t.expect(pooled.slots_per_layer < none.slots_per_layer);
    // The planned total never passes the budget, and one more slot would.
    const fixed = trunk + pooled.workspace_bytes + pooled.selected_bytes + BOUNCE_BYTES + pooled.pool_bytes;
    try t.expect(fixed + pooled.cache_bytes <= 100 * GiB);
    try t.expect(fixed + pooled.cache_bytes + 48 * expert_bytes > 100 * GiB);
    // A budget the pool alone tips under two slots a layer is refused.
    const floor = fixed + 2 * 48 * expert_bytes;
    try t.expectError(error.SsdBudgetBelowResident, budgetLedger(floor - 1, trunk, 0, 0, 48, 512, 10, expert_bytes, BOUNCE_BYTES, .{ .pool_bytes = 6 * GiB }));
}

test "expert stream: a resident store still bills the pool, and a budget short by it streams" {
    const t = std.testing;
    const trunk: u64 = 5_000_000_000;
    const shape: quant.StoreShape = .{ .all_bytes = 100_000_000_000, .pool_bytes = 2_000_000_000 };
    const exact = try budgetLedger(trunk + shape.all_bytes + shape.pool_bytes, trunk, 0, 0, 40, 384, 6, 9_000_000, BOUNCE_BYTES, shape);
    try t.expect(exact.resident);
    try t.expectEqual(shape.pool_bytes, exact.pool_bytes);
    const short = try budgetLedger(trunk + shape.all_bytes + shape.pool_bytes - 1, trunk, 0, 0, 40, 384, 6, 9_000_000, BOUNCE_BYTES, shape);
    try t.expect(!short.resident);
}

test "expert stream ssd budget ledger bills each union workspace set" {
    const t = std.testing;
    const expert_bytes: u64 = 9_830_400;
    const one = try budgetLedger(100 << 30, 0, 0, 0, 48, 512, 10, expert_bytes, BOUNCE_BYTES, .{});
    const two = try budgetLedger(100 << 30, 0, 0, 0, 48, 512, 10, expert_bytes, BOUNCE_BYTES, .{ .union_sets = 2, .union_meta_bytes = 1000 });
    try t.expectEqual(2 * one.workspace_bytes + 1000, two.workspace_bytes);
}

test "expert stream prefill peak bill covers a 501 expert union and k127 cache layer" {
    const t = std.testing;
    const p = try cachePlanBytes(60_000_000_000, 48, 512, try expertBytes(1280, 2560, 640), .{});
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
    const s = mlx.gpuStream();
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
    const s = mlx.gpuStream();
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
        const engine = try Engine.initWithOptions(t.allocator, path_buf[0..path_len], geom, @as(u64, slots) * 24576, mlx.gpuStream(), .{ .io_workers = 2, .bounce_size = 1 << 20 });
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
        const engine = try Engine.initWithOptions(t.allocator, path_buf[0..path_len], geom, requested, mlx.gpuStream(), .{ .io_workers = 2, .bounce_size = 1 << 20 });
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
    try expectError(error.FillSpanPastEof, engine.prepareHost(0, &.{1}));
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
    try expectError(error.FillSpanPastEof, engine.prepareHost(0, &.{2}));
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
    try expectError(error.SlabFilling, engine.prepareHost(0, &.{2}));
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
    try expectError(error.OutOfMemory, engine.prepareHost(0, &.{1}));
    engine.layers[0].slabs[1].slab.?.allocator = real;

    var after = try engine.prepareHost(0, &.{2});
    defer after.deinit();
    const slot = after.remapped[0];
    const gate_span = engine.store.span(0, 2, .gate_up);
    try t.expectEqualSlices(u8, fixture.raw[@intCast(gate_span.offset)..][0..@intCast(gate_span.len)], engine.cacheSlotBytes(0, slot, .gate_up));
}

test "expert stream a union view past 2^31 elements keeps its slot axes" {
    const s = mlx.gpuStream();
    var slab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slab);
    try mlx.check(mlx.mlx_zeros(&slab, &[_]c_int{ 3, 32768, 32768 }, 3, .uint8, s));
    var layer = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(layer);
    try mlx.check(mlx.mlx_zeros(&layer, &[_]c_int{ 5, 32768, 32768 }, 3, .uint8, s));
    const operand = SlabOperand{ .present = true, .array = slab, .stride = 32768 * 32768, .count = 2 };
    const view = try operand.viewAs(layer, s);
    defer _ = mlx.mlx_array_free(view);
    try std.testing.expectEqualSlices(c_int, &.{ 2, 32768, 32768 }, mlx.getShape(view));
}

test "expert stream union workspace refuses reuse while the previous reader holds it" {
    const t = std.testing;
    var fixture = try TinyEngine.open(2, 4);
    defer fixture.close();
    const engine = &fixture.engine;

    var held = try engine.prepareHost(0, &.{ 0, 1, 2 });
    defer held.deinit();
    try t.expect(held.workspace);
    try expectError(error.SlabLeased, engine.prepareHost(0, &.{ 0, 1, 3 }));
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
    var engine = try Engine.initWithOptions(t.allocator, path_buf[0..n], geometry, 2 * per_expert, mlx.gpuStream(), .{
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
    var engine = try Engine.initWithOptions(t.allocator, path_buf[0..path_len], geometry, 2 * per_expert, mlx.gpuStream(), .{
        .io_workers = 1,
        .bounce_size = 1 << 20,
        .layout = .mxfp4_split,
    });
    defer engine.deinit();

    try t.expectEqual(@as(u64, 12), engine.slab_imports);
    try t.expectEqual(@as(usize, 0), engine.layers[0].slabs.len);
    try t.expectEqual(@as(usize, quant.component_count), engine.layers[1].slabs.len);
    try expectError(error.ExpertLayerAbsent, engine.prepareHost(0, &.{1}));

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
    const s = mlx.gpuStream();

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
    try t.expectEqual(MtpUnderStreaming.refuse, mtpUnderStreaming(true, false, false, false));
    try t.expectEqual(MtpUnderStreaming.drop_settings, mtpUnderStreaming(true, true, false, false));
    try t.expectEqual(MtpUnderStreaming.off, mtpUnderStreaming(false, true, false, false));
    try t.expectEqual(MtpUnderStreaming.off, mtpUnderStreaming(false, false, false, true));
    // The engine default never refuses a streamed load: it resolves off.
    try t.expectEqual(MtpUnderStreaming.drop_default, mtpUnderStreaming(true, false, true, false));
    // A supported pack keeps its head from every source, the engine default included.
    for ([_][2]bool{ .{ false, false }, .{ true, false }, .{ false, true } }) |src|
        try t.expectEqual(MtpUnderStreaming.keep, mtpUnderStreaming(true, src[0], src[1], true));
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
    const ledger = try budgetLedger(fixed + 4 * per, 1 << 20, 0, 0, moeLayerCount(geometry), 4, 2, per, BOUNCE_BYTES, .{});
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
    try expectError(error.ExpertLayerAbsent, engine.prepareHost(0, &.{0}));
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

test "the warm set of a rate-grouped layer is each pool's most used experts, least used first" {
    const t = std.testing;
    var groups: quant.LayerGroups = .{ .count = 2 };
    groups.experts[0..2].* = .{ 3, 5 };
    groups.base[1] = 3;
    var gl: GroupedLayer = .{ .groups = groups, .hidden = 0, .inter = 0 };
    gl.slots[0..2].* = .{ 2, 3 };
    const counts = [_]u64{ 5, 9, 1, 0, 7, 7, 2, 8 };
    var out: std.ArrayList(u16) = .empty;
    defer out.deinit(t.allocator);
    try rankedWarm(t.allocator, &out, &gl, &counts, null, 0);
    try t.expectEqualSlices(u16, &.{ 0, 1, 4, 5, 7 }, out.items);
}

test "a rate-grouped spec map holds the global slot of each ready expert of its pool and nothing else" {
    const t = std.testing;
    var groups: quant.LayerGroups = .{ .count = 2 };
    groups.experts[0..2].* = .{ 3, 5 };
    groups.base[1] = 3;
    var gl: GroupedLayer = .{ .groups = groups, .hidden = 0, .inter = 0 };
    gl.slots[0..2].* = .{ 2, 3 };
    gl.slot_base[1] = 2;
    gl.caches[0] = try GroupCache.init(t.allocator, 2, 3);
    gl.caches[1] = try GroupCache.init(t.allocator, 3, 5);
    gl.caches_made = 2;
    defer gl.deinitCaches();
    try t.expectEqual(@as(usize, 8), gl.expertCount());
    var first = try gl.caches[0].resolve(t.allocator, &.{1});
    defer first.deinit(t.allocator);
    var second = try gl.caches[1].resolve(t.allocator, &.{ 0, 3 });
    defer second.deinit(t.allocator);
    var out: [8]i32 = @splat(-2);
    // Admitted but not yet read from the SSD: no GPU read may use the slot.
    try t.expect(refreshGroupedSpecMap(&gl, &out));
    try t.expectEqualSlices(i32, &@as([8]i32, @splat(-1)), &out);
    markResolvedReady(&gl.caches[0], &first);
    markResolvedReady(&gl.caches[1], &second);
    try t.expect(refreshGroupedSpecMap(&gl, &out));
    try t.expectEqualSlices(i32, &.{ -1, 0, -1, 2, -1, -1, 3, -1 }, &out);
    try t.expect(!refreshGroupedSpecMap(&gl, &out));
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
    const s = mlx.gpuStream();
    var engine = try Engine.initWithOptions(t.allocator, path_buf[0..path_len], geometry, 2 * 3072, s, .{ .layout = .bf16_individual, .bounce_size = 1 << 20 });
    defer engine.deinit();
    try t.expectEqual(@as(u64, 6), engine.slab_imports);
    try t.expectEqual(@as(u64, 0), engine.fallback_imports);
    try t.expectEqual(@as(u64, 3072), engine.store.perExpertBytes());
    try expectError(error.ExpertLayerAbsent, engine.prepareHost(0, &.{0}));
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

test "expert cache profile: the split spends the uniform bytes, gives the most reused layer more, and caps a layer at its experts" {
    const t = std.testing;
    // Layer 0 keeps reusing deep in its stack, layer 1 only its two hottest, layer 2 in between.
    const deep = [_]u64{ 50, 40, 30, 30, 30, 30, 30, 30, 5 };
    const flat = [_]u64{ 50, 40, 1, 1, 1, 1, 1, 1, 5 };
    const mid = [_]u64{ 50, 40, 20, 10, 5, 2, 1, 1, 5 };
    const counts = [_][]const u64{ &deep, &flat, &mid };
    const mean = [_]u64{ 10, 10, 10 };
    const floor = [_]u64{ 10, 10, 10 };
    var out: [3]u64 = undefined;
    try t.expect(profileAllot(&out, &counts, &mean, &.{ 80, 80, 80 }, &floor, 140));
    try t.expectEqualSlices(u64, &.{ 80, 20, 40 }, &out);

    // Bytes past every layer's experts stay unspent.
    try t.expect(profileAllot(&out, &counts, &mean, &.{ 80, 30, 80 }, &floor, 200));
    try t.expectEqualSlices(u64, &.{ 80, 30, 80 }, &out);

    try t.expect(!profileAllot(&out, &counts, &mean, &.{ 80, 80, 80 }, &floor, 25));
}

test "expert cache profile: a missing file, a wrong layer or expert count, or bad JSON reads as no profile" {
    const t = std.testing;
    const good = "{\"version\":1,\"experts\":2,\"tokens\":9,\"source\":\"x\",\"layers\":[[3,2,1],[4,0,2]]}";
    var p = parseCacheProfile(t.allocator, good, 2, 2) orelse return error.TestExpectedProfile;
    defer p.deinit(t.allocator);
    try t.expectEqualSlices(u64, &.{ 4, 0, 2 }, p.counts[1]);
    try t.expect(parseCacheProfile(t.allocator, good, 3, 2) == null);
    try t.expect(parseCacheProfile(t.allocator, good, 2, 3) == null);
    try t.expect(parseCacheProfile(t.allocator, "{\"version\":2,\"layers\":[[3,2,1],[4,0,2]]}", 2, 2) == null);
    try t.expect(parseCacheProfile(t.allocator, "{\"version\":1,\"layers\":[[3,2,1],[4,-1,2]]}", 2, 2) == null);
    try t.expect(parseCacheProfile(t.allocator, "{\"version\":1,\"layers\":[[3,2,1],[4,0", 2, 2) == null);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(t.io, &path_buf)];
    try t.expect(readCacheProfile(t.allocator, path, 2, 2) == null);
    try tmp.dir.writeFile(t.io, .{ .sub_path = cache_profile_file, .data = good });
    var read = readCacheProfile(t.allocator, path, 2, 2) orelse return error.TestExpectedProfile;
    read.deinit(t.allocator);
}

test "expert cache profile: the load line names the split" {
    const t = std.testing;
    var buf: [128]u8 = undefined;
    try t.expectEqualStrings("[expert-stream] slots per layer: profile (212..374)\n", cacheSplitLine(&buf, true, 212, 374));
    try t.expectEqualStrings("[expert-stream] slots per layer: uniform (234)\n", cacheSplitLine(&buf, false, 234, 234));
}
