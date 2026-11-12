//! Tensors of our own on our own Metal queue: the handles `mlx.zig` routes here in place of MLX arrays.
//!
//! Eager: an op encodes its dispatch into the open pass at once and returns a tensor over the output buffer. A view
//! (reshape, slice, transpose) is arithmetic on the host and encodes nothing. `submit` hands the pass to the GPU,
//! `sync` waits for everything submitted; a host read of a tensor's bytes syncs first.
//!
//! One thread drives the engine, the inference thread, as it does MLX.

const std = @import("std");
const builtin = @import("builtin");
const metal = @import("../metal.zig");
const metal_mlx = @import("../metal_mlx.zig");
const io_util = @import("../io_util.zig");
const mlx = @import("../mlx.zig");
const registry = @import("pipeline_registry.zig");
const fill_marks = @import("../fill_marks.zig");

pub const max_dims = 8;
pub const Dtype = mlx.mlx_dtype;

pub const Error = metal.Error || error{ OutOfMemory, MlxError, NotEvaluated, NotRowContiguous, TensorShape, TensorDtype, TensorUnsupported, TensorNull, MlxAtRest, PoolOverBudget, ReadsFillingSlab };

pub fn itemSize(d: Dtype) usize {
    return switch (d) {
        .bool_, .uint8, .int8 => 1,
        .uint16, .int16, .float16, .bfloat16 => 2,
        .uint32, .int32, .float32 => 4,
        .uint64, .int64, .float64, .complex64 => 8,
    };
}

/// The bytes a tensor's data lives in. A view shares its base's store.
pub const Store = struct {
    buffer: metal.Buffer,
    /// Where the data starts in `buffer`, and the host address of that byte.
    at: usize,
    ptr: [*]u8,
    len: usize,
    refs: u32 = 1,
    /// The last pass that binds it. The host touches the bytes only once that pass has run.
    seq: u64 = 0,
    origin: Origin,

    const Origin = union(enum) {
        /// A buffer of the pool, by size class.
        pool: u16,
        /// The memory of an MLX array, which this handle keeps alive.
        mlx: mlx.mlx_array,
        /// Memory its maker mapped and the engine wrapped (`wrapMapped`): `release(payload)` gives it back.
        mapped: struct { payload: ?*anyopaque, release: MappedRelease },
    };
};

/// What the engine refused last for a fill in flight: the bytes a dispatch of the pass binds, the range the fill is
/// writing, and how many dispatches the dropped pass held. For the refusal's message.
pub var refused_fill: struct { read: usize = 0, read_len: usize = 0, fill: usize = 0, fill_len: usize = 0, dispatches: u64 = 0 } = .{};

/// What the pool refused last for its budget: the buffer wanted, what Metal held for the pool, what the tensors in use
/// asked for, and the limit on the first of those. For the refusal's message.
pub var refused_budget: struct { wanted: u64 = 0, held: u64 = 0, asked: u64 = 0, limit: u64 = 0 } = .{};
/// Buffers the pool refused since the process began: a caller that handles a failure reads whether memory was its cause.
pub var pool_refusals: u64 = 0;

const Range = struct { ptr: usize, len: usize };

/// Gives back a mapping the engine wrapped. Called once, on the inference thread.
pub const MappedRelease = *const fn (?*anyopaque) callconv(.c) void;

pub const Tensor = struct {
    store: *Store,
    /// Bytes from the store's start to element 0.
    offset: usize,
    dtype: Dtype,
    ndim: u8,
    shape: [max_dims]c_int,
    /// In elements, as MLX reports strides.
    strides: [max_dims]usize,
    /// An MLX array over the same bytes, made when an MLX op first takes this tensor (`lend`).
    twin: mlx.mlx_array = .{},
    /// A slice update not yet written: `ops.settle` writes it before the bytes are first used.
    pending: ?*Pending = null,
    /// False once released: the box waits for its next tensor, and a second release of the same handle is refused.
    live: bool = true,

    pub fn dims(t: *const Tensor) []const c_int {
        return t.shape[0..t.ndim];
    }

    pub fn size(t: *const Tensor) usize {
        var n: usize = 1;
        for (t.dims()) |d| n *= @intCast(d);
        return n;
    }

    pub fn nbytes(t: *const Tensor) usize {
        return t.size() * itemSize(t.dtype);
    }

    /// The bytes from the first element `t` reads to the end of the last, whatever its strides skip between them.
    pub fn reach(t: *const Tensor) usize {
        var last: usize = 0;
        for (t.dims(), t.strides[0..t.ndim]) |n, stride| {
            if (n == 0) return 0;
            last += (@as(usize, @intCast(n)) - 1) * stride;
        }
        return (last + 1) * itemSize(t.dtype);
    }

    /// Whether an axis reads one element many times, as a broadcast does.
    pub fn repeats(t: *const Tensor) bool {
        for (t.dims(), t.strides[0..t.ndim]) |n, stride| if (n > 1 and stride == 0) return true;
        return false;
    }

    pub fn rowContiguous(t: *const Tensor) bool {
        var want: usize = 1;
        var axis: usize = t.ndim;
        while (axis > 0) {
            axis -= 1;
            if (t.shape[axis] != 1 and t.strides[axis] != want) return false;
            want *= @intCast(t.shape[axis]);
        }
        return true;
    }
};

/// `update` to be written over `start..stop` of the tensor that holds this.
pub const Pending = struct {
    update: *Tensor,
    start: [max_dims]c_int,
    stop: [max_dims]c_int,
};

/// Pool buffers are whole pages: every page count up to `steps`, then `steps` sizes to an octave, so a buffer is
/// under 1/16 larger than what was asked.
const page_bytes = 16 * 1024;
const steps = 16;
const classes = steps * 22;

const Stores = std.AutoHashMapUnmanaged(*Store, void);

/// A submitted pass: its command buffer, the fence it reaches when it has run, and its number.
const Flying = struct { cmd: metal.Committed, fence: metal.Fence, seq: u64, phase: u8 = 0, kernels: Kernels = .{} };

/// What the host is building, set by the model: a pause is charged to the phase of the pass that ends it.
pub var phase: u8 = 0;
pub const phases = 6;
const Finished = struct { cmd: metal.Committed, phase: u8, kernels: Kernels };

/// The kernels a pass dispatched, by name; filled only while the kernel profile is on.
const Kernels = struct {
    n: u32 = 0,
    named: u8 = 0,
    names: [8][]const u8 = undefined,

    fn add(k: *Kernels, name: []const u8) void {
        if (k.named < k.names.len) {
            k.names[k.named] = name;
            k.named += 1;
        }
        k.n += 1;
    }
};

const State = struct {
    dev: *const metal.Device,
    pass: ?metal.Pass = null,
    /// Submitted and not yet waited on, oldest first.
    flying: std.ArrayList(Flying) = .empty,
    /// The pass being encoded; a submit moves it on.
    seq: u64 = 1,
    /// Every pass up to this one has run.
    done: u64 = 0,
    free: [classes]std.ArrayList(*Store) = @splat(.empty),
    boxes: std.ArrayList(*Tensor) = .empty,
    /// MLX-backed stores dropped while a pass that binds them had not run: released at the next sync.
    retired: std.ArrayList(*Store) = .empty,
    /// The stores the dispatches since the last barrier read and wrote, and those of the dispatch being encoded.
    /// MLX's rule (`CommandEncoder`): a barrier goes before a dispatch that reads what one of them wrote or writes
    /// what one of them read; the rest overlap.
    prev_in: Stores = .empty,
    prev_out: Stores = .empty,
    next_in: Stores = .empty,
    next_out: Stores = .empty,
    needs_barrier: bool = false,
    /// Every store the open pass reads or writes, and those it writes. No buffer of ours is hazard-tracked, so a
    /// pass waits for the fence of each earlier pass, not yet run, that wrote a store it binds (MLX's rule).
    pass_binds: Stores = .empty,
    pass_writes: Stores = .empty,
    /// The byte ranges the open pass binds of mappings the engine wraps: a slab's views share its one store, and a
    /// host fill of one part of it must not refuse a pass that reads another (`refuseFillingReads`).
    pass_mapped: std.ArrayList(Range) = .empty,
    writers: std.AutoHashMapUnmanaged(*Store, Flying) = .empty,
    /// The most `pool_bytes` may be (`budget`); null before the first budget.
    pool_limit: ?u64 = null,
    /// Dispatches in the open pass, and how many an early hand-over waits for (`pass`).
    pass_dispatches: u32 = 0,
    kick_after: u32 = kick_first,
    /// A pass was handed over early since the host last waited: the ones after it follow without the GPU being idle.
    burst: bool = false,
    /// When the GPU ended the last command buffer whose times were read; 0 after one that had none.
    gpu_last_end: f64 = 0,
    /// Command buffers that have run and whose GPU times are still to read, oldest first: Metal sets them a moment
    /// after the event a wait returns on.
    untimed: std.ArrayList(Finished) = .empty,
    pass_kernels: Kernels = .{},
    stats: Stats = .{},
};

pub const Stats = struct {
    dispatches: u64 = 0,
    barriers: u64 = 0,
    /// Time the inference thread spent waiting for the GPU, and committing passes.
    wait_ns: u64 = 0,
    commit_ns: u64 = 0,
    submits: u64 = 0,
    waits: u64 = 0,
    /// Bytes of pool buffers Metal has allocated for the engine, in use or free; those free; and what the tensors
    /// in use asked for.
    pool_bytes: u64 = 0,
    pool_free: u64 = 0,
    pool_asked: u64 = 0,
    /// Free buffers released to keep the pool inside its budget.
    pool_evictions: u64 = 0,
    /// Passes handed over early: a burst's first because the GPU had none of ours to run, the rest behind it.
    kicks: u64 = 0,
    /// GPU time of our command buffers, and the time between one ending and the next beginning when that is under
    /// `gap_limit_s`: the GPU had none of ours to run. Buffers whose times Metal had not set yet are counted apart.
    gpu_busy_ns: u64 = 0,
    gpu_gap_ns: u64 = 0,
    gpu_untimed: u64 = 0,
    /// The pauses by length: under 20 us, under 100 us, under 1 ms, longer; how many and their time.
    gpu_gap_count: [4]u64 = @splat(0),
    gpu_gap_sum: [4]u64 = @splat(0),
    /// The pauses by the phase of the pass that ended them.
    gpu_gap_phase: [phases]u64 = @splat(0),
    /// Bytes of the mappings the engine wraps as buffers of its own (`wrapMapped`).
    mapped_bytes: u64 = 0,
    /// MLX arrays read by an op of ours.
    imports: u64 = 0,
    /// Tensors handed to MLX as arrays over their own bytes.
    lends: u64 = 0,
    /// The lends of a tensor over a wrapped mapping: MLX makes a second buffer over pages a buffer of ours wires.
    mapped_lends: u64 = 0,
    /// Slice updates written into the base's bytes, and those that had to copy the base first.
    updates_in_place: u64 = 0,
    updates_copied: u64 = 0,
};

const gpa = std.heap.c_allocator;
var state: ?State = null;

fn st() Error!*State {
    if (state == null) state = .{ .dev = try metal.device() };
    return &state.?;
}

pub fn stats() Stats {
    return if (state) |s| s.stats else .{};
}

/// `Stats.pool_bytes` and `pool_free` for a thread that is not the inference thread (`stats` is a plain read of
/// state it writes): stored where they change, a relaxed store each, so a launch takes no lock for them.
var pool_total = std.atomic.Value(u64).init(0);
var pool_unused = std.atomic.Value(u64).init(0);

pub fn poolTotal() u64 {
    return pool_total.load(.monotonic);
}

pub fn poolFree() u64 {
    return pool_unused.load(.monotonic);
}

var mapped_total: std.atomic.Value(u64) = .init(0);

/// `Stats.mapped_bytes`, readable off the inference thread.
pub fn mappedTotal() u64 {
    return mapped_total.load(.monotonic);
}

/// A pass on our queue when it keeps mappings resident (`metal.Device.touch`); nothing otherwise.
pub fn touch() void {
    const s = &(state orelse return);
    // With nothing wrapped there is no residency to keep, only a release since the last pass to tell the driver of.
    if (s.stats.mapped_bytes == 0) return metal.commitResident();
    (metal.device() catch return).touch();
}

/// Which high-water of the pool's bytes in use: the one the process's peak reading resets, or a forward phase's.
pub const Peak = enum { process, phase };
var pool_peaks: [2]std.atomic.Value(u64) = .{ .init(0), .init(0) };

pub fn poolPeak(which: Peak) u64 {
    return pool_peaks[@intFromEnum(which)].load(.monotonic);
}

/// The high-water starts again from what is in use now.
pub fn resetPoolPeak(which: Peak) void {
    pool_peaks[@intFromEnum(which)].store(poolTotal() - poolFree(), .monotonic);
}

fn publishPool(s: *const State) void {
    pool_total.store(s.stats.pool_bytes, .monotonic);
    pool_unused.store(s.stats.pool_free, .monotonic);
    const in_use = s.stats.pool_bytes - s.stats.pool_free;
    for (&pool_peaks) |*peak| if (in_use > peak.load(.monotonic)) peak.store(in_use, .monotonic);
}

// ── Kernel profile ──

/// Tests set this; otherwise `SUSHI_ENGINE_KERNEL_PROF` decides, read once.
pub var kernel_prof_hook: ?bool = null;
var kernel_prof_env: ?bool = null;

/// With the profile on, every op's pass is handed over and waited for before the next op's begins, so the
/// command buffer's GPU times are the op's. Timing changes; values do not.
pub fn kernelProfOn() bool {
    return kernel_prof_hook orelse io_util.diagEnvOnCached(&kernel_prof_env, "SUSHI_ENGINE_KERNEL_PROF");
}

pub const KernelStat = struct { name: []const u8, dispatches: u64, total_ns: u64, max_ns: u64 };

/// Pipeline handle to the name it was made by.
var pipeline_names: std.AutoHashMapUnmanaged(usize, []const u8) = .empty;
var profile: std.StringHashMapUnmanaged(KernelStat) = .empty;
var profile_view: std.ArrayList(KernelStat) = .empty;

fn pipelineName(raw: metal.id) []const u8 {
    return pipeline_names.get(@intFromPtr(raw)) orelse "(unnamed)";
}

/// One entry per distinct name; a pass of several dispatches is one entry named by all of them, joined by " + ".
fn noteKernels(k: Kernels, ns: u64) void {
    var buf: [1024]u8 = undefined;
    var len: usize = 0;
    for (k.names[0..k.named], 0..) |name, i| {
        const sep = if (i > 0) " + " else "";
        if (len + sep.len + name.len > buf.len) break;
        @memcpy(buf[len..][0..sep.len], sep);
        len += sep.len;
        @memcpy(buf[len..][0..name.len], name);
        len += name.len;
    }
    const key = buf[0..len];
    if (profile.getPtr(key)) |e| {
        e.dispatches += k.n;
        e.total_ns += ns;
        e.max_ns = @max(e.max_ns, ns);
        return;
    }
    const owned = gpa.dupe(u8, key) catch return;
    profile.put(gpa, owned, .{ .name = owned, .dispatches = k.n, .total_ns = ns, .max_ns = ns }) catch gpa.free(owned);
}

/// Reads the times of every finished command buffer, waiting a moment for the ones Metal has not timed yet.
fn flushSpans(s: *State) void {
    const nap: std.c.timespec = .{ .sec = 0, .nsec = 50 * std.time.ns_per_us };
    var tries: u32 = 0;
    while (true) {
        readGpuSpans(s, 0);
        if (s.untimed.items.len == 0 or tries >= 2000) return;
        tries += 1;
        _ = std.c.nanosleep(&nap, null);
    }
}

/// The kernels dispatched since the last reset, longest total GPU time first. Valid until the next call; the
/// caller has synced.
pub fn kernelProfile() []const KernelStat {
    if (state) |*s| flushSpans(s);
    profile_view.clearRetainingCapacity();
    var it = profile.valueIterator();
    while (it.next()) |e| profile_view.append(gpa, e.*) catch break;
    std.mem.sort(KernelStat, profile_view.items, {}, struct {
        fn longer(_: void, a: KernelStat, b: KernelStat) bool {
            return a.total_ns > b.total_ns;
        }
    }.longer);
    return profile_view.items;
}

pub fn resetKernelProfile() void {
    if (state) |*s| flushSpans(s);
    var it = profile.keyIterator();
    while (it.next()) |k| gpa.free(k.*);
    profile.clearRetainingCapacity();
}

/// Debug lines for the profile, `per` forwards to the totals; nothing while the switch is off.
pub fn logKernelProfile(per: f64) void {
    if (!kernelProfOn()) return;
    const top = 40;
    var other: KernelStat = .{ .name = "", .dispatches = 0, .total_ns = 0, .max_ns = 0 };
    var more: usize = 0;
    for (kernelProfile(), 0..) |k, i| {
        if (i >= top) {
            other.dispatches += k.dispatches;
            other.total_ns += k.total_ns;
            more += 1;
            continue;
        }
        logKernelLine(k.name, k, per);
    }
    if (more > 0) {
        var buf: [48]u8 = undefined;
        logKernelLine(std.fmt.bufPrint(&buf, "(other, {d} kernels)", .{more}) catch "(other)", other, per);
    }
}

fn logKernelLine(name: []const u8, k: KernelStat, per: f64) void {
    const log = @import("../log.zig");
    const n: f64 = @floatFromInt(k.dispatches);
    const ns: f64 = @floatFromInt(k.total_ns);
    log.debug("[engine-prof] {s}: {d:.1} dispatches, {d:.3} ms total, {d:.2} us each\n", .{ name, n / per, ns / 1e6 / per, ns / 1e3 / @max(n, 1) });
}

// ── Handles ──

var stream_tag: u8 align(16) = 0;

/// The stream that names the engine: an op given it, or given a tensor of ours, runs here.
pub fn stream() mlx.mlx_stream {
    return .{ .ctx = @ptrFromInt(@intFromPtr(&stream_tag) | 1) };
}

pub fn isStream(s: mlx.mlx_stream) bool {
    return @intFromPtr(s.ctx) == @intFromPtr(&stream_tag) | 1;
}

/// MLX's handles are heap pointers, even; ours carry the low bit.
pub fn isOurs(a: mlx.mlx_array) bool {
    return @intFromPtr(a.ctx) & 1 != 0;
}

/// Where the memory `a`, a tensor of ours, lies in begins: its store's first byte, whatever view `a` is of it.
pub fn storePointer(a: mlx.mlx_array) [*]const u8 {
    return of(a).store.ptr;
}

/// Whether `a`, a tensor of ours, lies over a mapping the engine wrapped (`wrapMapped`).
pub fn isMapped(a: mlx.mlx_array) bool {
    return of(a).store.origin == .mapped;
}

pub fn handle(t: *Tensor) mlx.mlx_array {
    return .{ .ctx = @ptrFromInt(@intFromPtr(t) | 1) };
}

pub fn of(a: mlx.mlx_array) *Tensor {
    std.debug.assert(isOurs(a));
    return @ptrFromInt(@intFromPtr(a.ctx) & ~@as(usize, 1));
}

// ── Stores and tensors ──

fn classOf(len: usize) usize {
    const pages = (len + page_bytes - 1) / page_bytes;
    if (pages <= steps) return pages - 1;
    var shift: usize = std.math.log2_int(usize, pages) - std.math.log2_int(usize, steps);
    var top = (pages + (@as(usize, 1) << @intCast(shift)) - 1) >> @intCast(shift);
    if (top == 2 * steps) {
        shift += 1;
        top = steps;
    }
    return steps - 1 + shift * steps + (top - steps);
}

fn classBytes(class: usize) usize {
    if (class < steps) return (class + 1) * page_bytes;
    const k = class - (steps - 1);
    return ((steps + k % steps) << @intCast(k / steps)) * page_bytes;
}

/// TEST-ONLY, `mlx.fault` for the engine's allocations: `arm(k)` makes the pool refuse its k-th buffer from here with
/// `PoolOverBudget`, so a test walks every error path of a forward without patching it. Compiled only under test.
pub const fault = struct {
    var remaining: u64 = 0; // 0 = disarmed
    var asked: u64 = 0;

    pub fn arm(k: u64) void {
        remaining = k;
    }
    pub fn disarm() void {
        remaining = 0;
    }
    /// Buffers the pool was asked for since the process began: the difference over a request is its sweep's length.
    pub fn count() u64 {
        return asked;
    }
    fn hit() bool {
        asked += 1;
        if (remaining == 0) return false;
        remaining -= 1;
        return remaining == 0;
    }
};

/// TEST-ONLY: with `byte` set, the pool writes it over every buffer it hands out (a reused one once the passes that
/// bound it have run), and `free` writes it over the free ones, so a kernel that reads bytes nothing wrote shows as a
/// changed result. Compiled only under test.
pub const poison = struct {
    pub var byte: ?u8 = null;
    /// Reused buffers handed out as they were: the open pass's predecessors still bound them.
    pub var skipped: u64 = 0;

    /// Writes `b` over every free buffer, every pass having run. The bytes written.
    pub fn free(b: u8) usize {
        const s = &(state orelse return 0);
        syncState(s) catch return 0;
        takeBack();
        var n: usize = 0;
        for (&s.free, 0..) |*list, class| for (list.items) |store| {
            @memset(store.ptr[0..classBytes(class)], b);
            n += classBytes(class);
        };
        return n;
    }

    fn handOut(s: *State, store: *Store, bytes: usize) Error!void {
        const b = byte orelse return;
        if (store.seq > s.done and store.seq < s.seq) try waitFor(s, store.seq);
        if (store.seq <= s.done) @memset(store.ptr[0..bytes], b) else skipped += 1;
    }
};

/// A store of at least `len` bytes from the pool, never one the open pass binds. `for_host` asks for one no
/// submitted pass still binds either, since the host is about to write it. A new buffer stays inside the budget:
/// free buffers go to make room, and with none left the op is refused.
fn poolStore(len: usize, for_host: bool) Error!*Store {
    const s = try st();
    takeBack();
    if (comptime builtin.is_test) if (fault.hit()) {
        pool_refusals += 1;
        return error.PoolOverBudget;
    };
    const class = classOf(len);
    if (class >= classes) return error.TensorShape;
    const bytes = classBytes(class);
    const list = &s.free[class];
    var at = list.items.len;
    while (at > 0) {
        at -= 1;
        const store = list.items[at];
        if (for_host and store.seq > s.done) continue;
        // A buffer the open pass already binds would make its next dispatch wait for the ones before it.
        if (store.seq == s.seq and s.pass != null) continue;
        _ = list.swapRemove(at);
        if (comptime builtin.is_test) try poison.handOut(s, store, bytes);
        store.refs = 1;
        store.len = len;
        s.stats.pool_free -= bytes;
        s.stats.pool_asked += len;
        publishPool(s);
        return store;
    }
    if (s.pool_limit) |limit| {
        // A buffer larger than the cap fits under no trim: an empty pool would otherwise let it through.
        if (bytes > limit or !try trimTo(s, limit - bytes)) {
            refused_budget = .{ .wanted = bytes, .held = s.stats.pool_bytes, .asked = s.stats.pool_asked, .limit = limit };
            pool_refusals += 1;
            return error.PoolOverBudget;
        }
    }
    const buffer = try s.dev.alloc(bytes);
    errdefer buffer.release();
    const store = gpa.create(Store) catch return error.OutOfMemory;
    store.* = .{ .buffer = buffer, .at = 0, .ptr = buffer.bytes().ptr, .len = len, .origin = .{ .pool = @intCast(class) } };
    if (comptime builtin.is_test) if (poison.byte) |b| @memset(store.ptr[0..bytes], b);
    s.stats.pool_bytes += bytes;
    s.stats.pool_asked += len;
    publishPool(s);
    return store;
}

/// Releases free buffers until the pool holds at most `bytes`, waiting for the passes that still bind some when
/// the idle ones are not enough. False when the buffers in use and those the open pass binds hold more.
fn trimTo(s: *State, bytes: u64) Error!bool {
    if (s.stats.pool_bytes <= bytes) return true;
    if (evict(s, bytes)) return true;
    try waitFor(s, s.seq - 1);
    return evict(s, bytes);
}

/// The largest first; a buffer a pass has yet to run over stays.
fn evict(s: *State, bytes: u64) bool {
    var class: usize = classes;
    while (s.stats.pool_bytes > bytes and class > 0) {
        class -= 1;
        const list = &s.free[class];
        var at = list.items.len;
        while (at > 0 and s.stats.pool_bytes > bytes) {
            at -= 1;
            const store = list.items[at];
            if (store.seq > s.done) continue;
            _ = list.swapRemove(at);
            store.buffer.release();
            gpa.destroy(store);
            s.stats.pool_bytes -= classBytes(class);
            s.stats.pool_free -= classBytes(class);
            s.stats.pool_evictions += 1;
        }
    }
    metal.commitResident();
    publishPool(s);
    return s.stats.pool_bytes <= bytes;
}

/// Releases every free buffer no pass still binds; the budget stays as it was.
pub fn trim() void {
    const s = &(state orelse return);
    takeBack();
    const limit = s.pool_limit;
    _ = evict(s, 0);
    s.pool_limit = limit;
}

/// From here Metal holds at most `bytes` for the pool beyond what the tensors in use ask for, free buffers
/// included: a forward's cap, installed as it begins. Free buffers past it go now.
pub fn budget(bytes: u64) Error!void {
    const s = &(state orelse return);
    takeBack();
    s.pool_limit = s.stats.pool_asked +| bytes;
    _ = try trimTo(s, s.pool_limit.?);
}

/// The cap rises by the buffers that tensors of these `lens` bytes take from the pool. For what is made between
/// forwards and outlives them (a restore point's copies): a forward's cap covers a forward's scratch, and the next
/// forward's cap starts from what is in use then.
pub fn room(lens: []const usize) void {
    const s = &(state orelse return);
    const limit = s.pool_limit orelse return;
    var bytes: u64 = 0;
    for (lens) |len| {
        if (len > 0) bytes += classBytes(classOf(len));
    }
    s.pool_limit = limit +| bytes;
}

/// The cap as `uncapped` found it, and the pool buffers in use then.
pub const Uncapped = struct { limit: ?u64 = null, in_use: u64 = 0 };

/// For what is built between forwards and outlives them when its sizes are not known ahead (state read back from
/// disk): no cap from here until `capped`.
pub fn uncapped() Uncapped {
    const s = &(state orelse return .{});
    takeBack();
    const was: Uncapped = .{ .limit = s.pool_limit, .in_use = s.stats.pool_bytes - s.stats.pool_free };
    s.pool_limit = null;
    return was;
}

/// The cap `uncapped` lifted, raised by the buffers that came into use since; free buffers past it go.
pub fn capped(was: Uncapped) void {
    const s = &(state orelse return);
    const limit = was.limit orelse return;
    takeBack();
    s.pool_limit = limit +| ((s.stats.pool_bytes - s.stats.pool_free) -| was.in_use);
    _ = evict(s, s.pool_limit.?);
}

fn dropStore(store: *Store) void {
    store.refs -= 1;
    if (store.refs > 0) return;
    const s = &state.?;
    switch (store.origin) {
        .pool => |class| {
            s.stats.pool_asked -= store.len;
            if (s.free[class].append(gpa, store)) |_| {
                s.stats.pool_free += classBytes(class);
            } else |_| {
                // No room to keep it: it goes once the passes that bind it have run.
                if (store.seq > s.done) waitFor(s, store.seq) catch {};
                store.buffer.release();
                s.stats.pool_bytes -= classBytes(class);
                gpa.destroy(store);
            }
            publishPool(s);
        },
        // The buffer aliases pages someone else keeps: the last pass that binds it runs before they may go.
        .mlx, .mapped => if (store.seq > s.done) {
            s.retired.append(gpa, store) catch {
                syncState(s) catch {};
                freeAlias(s, store);
            };
        } else freeAlias(s, store),
    }
}

/// The end of a store over memory the pool does not own, once no pass binds it.
fn freeAlias(s: *State, store: *Store) void {
    switch (store.origin) {
        .pool => unreachable,
        .mlx => |kept| _ = mlx.mlx_array_free(kept),
        .mapped => |m| {
            store.buffer.release();
            s.stats.mapped_bytes -= store.len;
            mapped_total.store(s.stats.mapped_bytes, .monotonic);
            m.release(m.payload);
        },
    }
    gpa.destroy(store);
}

// ── Tensors lent to MLX ──

/// Stores whose MLX array has gone. MLX may drop an array from a Metal completion thread, so the list has a lock
/// and the inference thread takes the stores back.
var returned_lock: std.atomic.Value(bool) = .init(false);
var returned: std.ArrayList(*Store) = .empty;

fn lockReturned() void {
    while (returned_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
}

fn twinGone(payload: ?*anyopaque) callconv(.c) void {
    const store: *Store = @ptrCast(@alignCast(payload.?));
    lockReturned();
    defer returned_lock.store(false, .release);
    returned.append(gpa, store) catch @panic("engine: out of memory returning a lent store");
}

fn takeBack() void {
    if (returned.items.len == 0) return;
    var mine: std.ArrayList(*Store) = .empty;
    {
        lockReturned();
        defer returned_lock.store(false, .release);
        std.mem.swap(std.ArrayList(*Store), &mine, &returned);
    }
    defer mine.deinit(gpa);
    for (mine.items) |store| dropStore(store);
}

/// `t` as an MLX array over the same bytes, for an op on MLX's queue: ours is waited on first. The array is `t`'s
/// and goes with it; MLX holds the bytes until its last reference to the array goes. `t` is row-contiguous at
/// offset 0. A wrapped mapping keeps no such array (`lendsOnce`): the one returned is the caller's to free once MLX
/// has taken it.
pub fn lend(t: *Tensor) Error!mlx.mlx_array {
    if (t.twin.ctx != null) return t.twin;
    if (mlx.engine.mlx_at_rest.on.load(.monotonic)) return error.MlxAtRest;
    if (t.offset != 0 or !t.rowContiguous() or t.pending != null) return error.NotRowContiguous;
    const s = try st();
    if (t.store.seq > s.done) try waitFor(s, t.store.seq);
    if (t.store.origin == .mlx and sameLayout(t, t.store.origin.mlx)) {
        t.twin = mlx.raw.mlx_array_new();
        try mlx.check(mlx.raw.mlx_array_set(&t.twin, t.store.origin.mlx));
        return t.twin;
    }
    t.store.refs += 1;
    const arr = mlx.raw.mlx_array_new_data_managed_payload(t.store.ptr, &t.shape, t.ndim, t.dtype, t.store, twinGone);
    if (arr.ctx == null) {
        t.store.refs -= 1;
        return error.MlxError;
    }
    s.stats.lends += 1;
    if (t.store.origin == .mapped) {
        s.stats.mapped_lends += 1;
        return arr;
    }
    t.twin = arr;
    return arr;
}

/// Whether a lend of `t` is made for one call and freed after it. MLX bills and wires every buffer it wraps for as
/// long as it holds it: a twin kept with a weight would stand, billed and wired twice, until the unload.
pub fn lendsOnce(t: *const Tensor) bool {
    return t.store.origin == .mapped and t.twin.ctx == null;
}

/// Whether `t` reads the whole of the MLX array `a` as `a` itself does.
fn sameLayout(t: *const Tensor, a: mlx.mlx_array) bool {
    if (t.dtype != mlx.mlx_array_dtype(a) or !std.mem.eql(c_int, t.dims(), mlx.getShape(a))) return false;
    const strides = mlx.mlx_array_strides(a);
    for (0..t.ndim) |axis| if (t.strides[axis] != strides[axis]) return false;
    return true;
}

fn box() Error!*Tensor {
    const s = try st();
    const t = s.boxes.pop() orelse (gpa.create(Tensor) catch return error.OutOfMemory);
    t.twin = .{};
    t.pending = null;
    t.live = true;
    return t;
}

/// The box of a tensor that is gone, kept for the next one.
fn unbox(t: *Tensor) void {
    t.live = false;
    state.?.boxes.append(gpa, t) catch gpa.destroy(t);
}

/// Releases of a tensor already released, since the process began: each was a defect in a caller's error path.
pub var double_releases: u64 = 0;

fn setShape(t: *Tensor, shape: []const c_int) Error!void {
    if (shape.len > max_dims) return error.TensorShape;
    t.ndim = @intCast(shape.len);
    var stride: usize = 1;
    var axis = shape.len;
    while (axis > 0) {
        axis -= 1;
        if (shape[axis] < 0) return error.TensorShape;
        t.shape[axis] = shape[axis];
        t.strides[axis] = stride;
        stride *= @intCast(shape[axis]);
    }
}

/// A row-contiguous tensor over a pool buffer whose bytes are whatever the buffer last held.
pub fn fresh(dtype: Dtype, shape: []const c_int) Error!*Tensor {
    return freshFor(dtype, shape, false);
}

fn freshFor(dtype: Dtype, shape: []const c_int, for_host: bool) Error!*Tensor {
    const t = try box();
    errdefer unbox(t);
    try setShape(t, shape);
    t.dtype = dtype;
    t.offset = 0;
    t.store = try poolStore(@max(t.nbytes(), 1), for_host);
    return t;
}

/// `fresh` over a buffer no pass has yet to run over: the host may write it at once.
pub fn freshHost(dtype: Dtype, shape: []const c_int) Error!*Tensor {
    return freshFor(dtype, shape, true);
}

/// A tensor holding a copy of `bytes`.
pub fn fromHost(bytes: []const u8, dtype: Dtype, shape: []const c_int) Error!*Tensor {
    const t = try freshFor(dtype, shape, true);
    errdefer release(t);
    if (t.nbytes() != bytes.len) return error.TensorShape;
    @memcpy(t.store.ptr[0..bytes.len], bytes);
    return t;
}

/// A tensor over `ptr`, page-aligned memory its maker mapped and filled: no copy, and the pages are wired once, under
/// a buffer of ours. `release(payload)` runs once the tensor is released and the last pass that binds it has run.
pub fn wrapMapped(ptr: [*]u8, dtype: Dtype, shape: []const c_int, payload: ?*anyopaque, release_fn: MappedRelease) Error!*Tensor {
    const s = try st();
    const t = try box();
    errdefer unbox(t);
    try setShape(t, shape);
    t.dtype = dtype;
    t.offset = 0;
    const len = @max(t.nbytes(), 1);
    const store = gpa.create(Store) catch return error.OutOfMemory;
    errdefer gpa.destroy(store);
    const buffer = try (try metal.device()).wrap(ptr, len);
    store.* = .{ .buffer = buffer, .at = 0, .ptr = ptr, .len = len, .origin = .{ .mapped = .{ .payload = payload, .release = release_fn } } };
    t.store = store;
    s.stats.mapped_bytes += len;
    mapped_total.store(s.stats.mapped_bytes, .monotonic);
    return t;
}

/// Another tensor on `base`'s bytes, to be given its own shape, strides and offset.
pub fn share(base: *const Tensor) Error!*Tensor {
    const t = try box();
    t.* = base.*;
    t.twin = .{};
    t.pending = null;
    t.store.refs += 1;
    return t;
}

pub fn release(t: *Tensor) void {
    // A box holds one tensor at a time: released again before it is reused, it would sit in the free list twice and
    // give two owners one tensor, and its store would go back to the pool while in use.
    if (!t.live) {
        double_releases += 1;
        return mlx.latchError("[engine] release: a tensor released twice");
    }
    if (t.twin.ctx != null) _ = mlx.mlx_array_free(t.twin);
    if (t.pending) |p| freePending(p);
    dropStore(t.store);
    unbox(t);
}

// ── MLX memory the engine reads ──

/// A tensor over the memory of an MLX array, evaluated here if it was not; no copy. A pass of ours binds MLX's own
/// Metal buffer, so the memory is wired once. A strided view keeps its strides. The store keeps the array, and
/// with it the buffer, until every pass that binds it has run.
pub fn import(a: mlx.mlx_array) Error!*Tensor {
    // At rest MLX is asked nothing: an array of its own that reaches an op of ours is a model on MLX's work outside
    // its turn, refused by name.
    if (mlx.engine.mlx_at_rest.on.load(.monotonic)) return error.MlxAtRest;
    // An array of no elements has no memory to share: its shape and dtype are all of it.
    const dims = mlx.getShape(a);
    for (dims) |d| if (d == 0) return fresh(mlx.mlx_array_dtype(a), dims);
    const s = try st();
    var available = false;
    try mlx.check(mlx._mlx_array_is_available(&available, a));
    if (!available) try mlx.check(mlx.rawArrayEval(a));
    const ptr: [*]u8 = @constCast(mlx.mlx_array_data_uint8(a) orelse return error.NotEvaluated);
    var raw: ?*anyopaque = null;
    var at: usize = 0;
    try mlx.check(mlx._mlx_array_metal_buffer(&raw, &at, a));
    const dtype = mlx.mlx_array_dtype(a);
    const t = try box();
    errdefer unbox(t);
    try setShape(t, dims);
    const strides = mlx.mlx_array_strides(a);
    for (0..t.ndim) |axis| t.strides[axis] = strides[axis];
    t.dtype = dtype;
    t.offset = 0;
    var kept = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&kept, a));
    errdefer _ = mlx.mlx_array_free(kept);
    const store = gpa.create(Store) catch return error.OutOfMemory;
    store.* = .{ .buffer = .{ .raw = raw orelse return error.NotEvaluated }, .at = at, .ptr = ptr, .len = metal_mlx.span(a) * itemSize(dtype), .origin = .{ .mlx = kept } };
    t.store = store;
    s.stats.imports += 1;
    return t;
}

/// `a` as a tensor the caller releases: ours with one more reference, or an import of an MLX array.
pub fn take(a: mlx.mlx_array) Error!*Tensor {
    // An empty handle is what a maker that failed gave its caller: the op that takes it says so.
    if (a.ctx == null) return error.TensorNull;
    if (isOurs(a)) return share(try ready(a));
    return import(a);
}

/// The tensor behind `a` with its bytes final: a slice update still pending is written first.
pub fn ready(a: mlx.mlx_array) Error!*Tensor {
    const t = of(a);
    if (t.pending != null) try @import("ops.zig").settle(t);
    return t;
}

/// `t` now owns `from`'s bytes and layout; `from` is gone.
pub fn adopt(t: *Tensor, from: *Tensor) void {
    dropStore(t.store);
    t.store = from.store;
    t.offset = from.offset;
    t.strides = from.strides;
    unbox(from);
}

/// Whether no one else reads `store`'s bytes, so that a write into them changes nothing another holder sees. For
/// bytes MLX made this is MLX's own rule for writing in place: the kept handle is the array's and the buffer's only
/// holder, a command buffer of MLX's still in flight counted.
pub fn sole(store: *const Store) Error!bool {
    if (store.refs != 1) return false;
    switch (store.origin) {
        .pool => return true,
        .mlx => |kept| {
            var only = false;
            try mlx.check(mlx._mlx_array_is_donatable(&only, kept));
            return only;
        },
        // A wrapped mapping is its maker's (a weight): never written in place.
        .mapped => return false,
    }
}

pub fn countUpdate(in_place: bool) void {
    const s = &state.?.stats;
    if (in_place) s.updates_in_place += 1 else s.updates_copied += 1;
}

pub fn newPending(update: *Tensor, start: []const c_int, stop: []const c_int) Error!*Pending {
    const p = gpa.create(Pending) catch return error.OutOfMemory;
    p.* = .{ .update = update, .start = @splat(0), .stop = @splat(0) };
    @memcpy(p.start[0..start.len], start);
    @memcpy(p.stop[0..stop.len], stop);
    return p;
}

pub fn freePending(p: *Pending) void {
    release(p.update);
    gpa.destroy(p);
}

// ── Passes ──

/// The first early hand-over of a burst comes after this many dispatches; each one after takes twice as many.
const kick_first = 4;

/// The open pass, begun now if none is. Work encoded while the GPU has none of ours to run is handed over early,
/// in steps that double, so the GPU starts on a draft or a layer while the host still builds the rest of it. The
/// steps after the first do not wait for the GPU to run dry: a pass handed over behind one that has not started
/// starts as that one ends, where one handed to an idle GPU waits Metal's start latency again. The steps start over
/// at the host's next wait. Never between a dispatch's binds and the dispatch.
pub fn pass() Error!metal.Pass {
    const s = try st();
    if (s.pass) |open| {
        if (kernelProfOn() and s.pass_dispatches > 0) {
            s.pass = null;
            try commitPass(s, open);
            try waitFor(s, s.seq - 1);
            flushSpans(s);
        } else if (s.pass_dispatches >= s.kick_after and s.next_in.count() == 0 and s.next_out.count() == 0 and (s.burst or gpuIdle(s))) {
            s.pass = null;
            try commitPass(s, open);
            s.kick_after *|= 2;
            s.burst = true;
            s.stats.kicks += 1;
        }
    }
    if (s.pass == null) s.pass = s.dev.beginConcurrent();
    return s.pass.?;
}

/// Whether every pass handed to the GPU has run.
fn gpuIdle(s: *State) bool {
    return s.flying.items.len == 0 or s.flying.items[s.flying.items.len - 1].cmd.ended();
}

/// Binds the bytes of `t`, which the dispatch being encoded reads, at `index` of the open pass.
pub fn bind(p: metal.Pass, t: *const Tensor, index: usize) void {
    const s = &state.?;
    t.store.seq = s.seq;
    s.next_in.put(gpa, t.store, {}) catch {
        s.needs_barrier = true;
    };
    if (s.prev_out.contains(t.store)) s.needs_barrier = true;
    s.pass_binds.put(gpa, t.store, {}) catch @panic("engine: out of memory tracking a pass");
    if (t.store.origin == .mapped) {
        const range: Range = .{ .ptr = @intFromPtr(t.store.ptr) + t.offset, .len = t.reach() };
        const n = s.pass_mapped.items.len;
        if (n == 0 or !std.meta.eql(s.pass_mapped.items[n - 1], range))
            s.pass_mapped.append(gpa, range) catch @panic("engine: out of memory tracking a pass");
    }
    p.buffer(t.store.buffer, t.store.at + t.offset, index);
}

/// `bind` for bytes the dispatch being encoded writes.
pub fn bindOut(p: metal.Pass, t: *const Tensor, index: usize) void {
    const s = &state.?;
    bind(p, t, index);
    s.next_out.put(gpa, t.store, {}) catch {
        s.needs_barrier = true;
    };
    if (s.prev_in.contains(t.store)) s.needs_barrier = true;
    s.pass_writes.put(gpa, t.store, {}) catch @panic("engine: out of memory tracking a pass");
}

fn beforeDispatch() void {
    const s = &state.?;
    s.stats.dispatches += 1;
    s.pass_dispatches += 1;
    if (kernelProfOn()) s.pass_kernels.add(pipelineName(metal.bound_pipeline));
    if (s.needs_barrier) {
        s.pass.?.barrier();
        s.stats.barriers += 1;
        s.needs_barrier = false;
        std.mem.swap(Stores, &s.prev_in, &s.next_in);
        std.mem.swap(Stores, &s.prev_out, &s.next_out);
    } else {
        var it = s.next_in.keyIterator();
        while (it.next()) |k| s.prev_in.put(gpa, k.*, {}) catch {};
        it = s.next_out.keyIterator();
        while (it.next()) |k| s.prev_out.put(gpa, k.*, {}) catch {};
    }
    s.next_in.clearRetainingCapacity();
    s.next_out.clearRetainingCapacity();
}

/// Encodes the dispatch whose buffers were just bound, behind a barrier when it meets an earlier one's bytes.
pub fn dispatch(p: metal.Pass, grid: metal.Size, group: metal.Size) void {
    beforeDispatch();
    p.dispatch(grid, group);
}

/// `dispatch` with `grid` in threadgroups.
pub fn dispatchGroups(p: metal.Pass, grid: metal.Size, group: metal.Size) void {
    beforeDispatch();
    p.dispatchGroups(grid, group);
}

/// Pipelines made since the process began, the time they took, and those made after `sealed`: a model sets it once
/// its load has run every forward a request can, so a later pipeline is a stall inside a request.
pub var pipelines: u64 = 0;
pub var pipeline_ns: u64 = 0;
pub var pipelines_late: u64 = 0;
/// Pipelines a launch took from those a manifest made ahead: no compile, so neither counted above nor late.
pub var pipelines_preloaded: u64 = 0;
pub var sealed: bool = false;

/// The pipeline `pipeline`, named `name`, was made since `t0`.
pub fn madePipeline(name: []const u8, t0: u64, pipeline: metal.Pipeline) void {
    const ns = nowNs() - t0;
    nameOf(name, pipeline);
    pipelines += 1;
    pipeline_ns += ns;
    const log = @import("../log.zig");
    if (!sealed) return log.debug("[engine] compiled {s}\n", .{name});
    pipelines_late += 1;
    log.warn("[engine] compiled inside a request ({d} ms): {s}\n", .{ ns / std.time.ns_per_ms, name });
}

fn nameOf(name: []const u8, pipeline: metal.Pipeline) void {
    if (gpa.dupe(u8, name)) |owned| pipeline_names.put(gpa, @intFromPtr(pipeline.raw), owned) catch gpa.free(owned) else |_| {}
}

/// `pipeline`, which a manifest made ahead, is first used as `name`.
pub fn preloadedPipeline(name: []const u8, pipeline: metal.Pipeline) void {
    nameOf(name, pipeline);
    pipelines_preloaded += 1;
}

/// The pipeline of the kernel source `text`, entry point `function`: the one a manifest made ahead, else compiled now.
pub fn pipelineFromSource(function: [:0]const u8, text: [:0]const u8) Error!metal.Pipeline {
    if (registry.takeSource(function, text)) |p| {
        preloadedPipeline(function, p);
        return p;
    }
    const t0 = nowNs();
    const pipeline = try (try metal.device()).compile(text, function);
    madePipeline(function, t0, pipeline);
    registry.record(.{ .kind = .source, .function = function, .text = text });
    return pipeline;
}

pub fn nowNs() u64 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t);
    return @as(u64, @intCast(t.sec)) * std.time.ns_per_s + @as(u64, @intCast(t.nsec));
}

/// Ends the open pass's encoding rules and hands it to the GPU: it waits for the passes that wrote what it binds
/// and marks what it writes.
fn commitPass(s: *State, open: metal.Pass) Error!void {
    if (fill_marks.live.load(.acquire) != 0) try refuseFillingReads(s, open);
    clearHazards(s);
    const fence = try s.dev.fence();
    errdefer fence.release();
    // Each pass also waits for the one before it, so passes finish in the order of their numbers and the event's
    // value says that every pass up to it has run.
    var waited: [16]metal.Fence = undefined;
    var n_waited: usize = 0;
    if (s.flying.items.len > 0) {
        waited[0] = s.flying.items[s.flying.items.len - 1].fence;
        open.waitFence(waited[0]);
        n_waited = 1;
    }
    var binds = s.pass_binds.keyIterator();
    while (binds.next()) |store| {
        const writer = s.writers.get(store.*) orelse continue;
        if (writer.seq <= s.done) continue;
        var seen = false;
        for (waited[0..n_waited]) |w| seen = seen or w.raw == writer.fence.raw;
        if (seen) continue;
        open.waitFence(writer.fence);
        if (n_waited < waited.len) {
            waited[n_waited] = writer.fence;
            n_waited += 1;
        }
    }
    const t0 = nowNs();
    const flying: Flying = .{ .cmd = undefined, .fence = fence, .seq = s.seq };
    const kernels = s.pass_kernels;
    s.pass_kernels = .{};
    var writes = s.pass_writes.keyIterator();
    while (writes.next()) |store| s.writers.put(gpa, store.*, flying) catch return error.OutOfMemory;
    open.updateFence(fence);
    clearPassBinds(s);
    s.flying.append(gpa, .{ .cmd = open.commit(), .fence = fence, .seq = s.seq, .phase = phase, .kernels = kernels }) catch return error.OutOfMemory;
    s.stats.commit_ns += nowNs() - t0;
    s.seq += 1;
    s.pass_dispatches = 0;
    s.stats.submits += 1;
}

/// A pass that binds bytes a host fill is still writing is dropped unrun: the GPU would read them half written.
fn refuseFillingReads(s: *State, open: metal.Pass) Error!void {
    const marks = fill_marks.snapshot();
    var binds = s.pass_binds.keyIterator();
    const read: Range, const mark: [2]usize = blk: {
        for (s.pass_mapped.items) |range| if (marks.met(range.ptr, range.len)) |m| break :blk .{ range, m };
        // An import is of one MLX array, so its store is the range it reads; a pool buffer is never a host fill's.
        while (binds.next()) |store| if (store.*.origin == .mlx) {
            const range: Range = .{ .ptr = @intFromPtr(store.*.ptr), .len = store.*.len };
            if (marks.met(range.ptr, range.len)) |m| break :blk .{ range, m };
        };
        return;
    };
    refused_fill = .{ .read = read.ptr, .read_len = read.len, .fill = mark[0], .fill_len = mark[1] - mark[0], .dispatches = s.pass_dispatches };
    open.discard();
    clearHazards(s);
    // The pass never runs, so nothing it bound waits for it: left at this pass, a freed buffer could not leave the pool
    // until a later pass ran, and a request whose first buffer needs that room would be refused for good.
    binds = s.pass_binds.keyIterator();
    while (binds.next()) |store| if (store.*.seq == s.seq) {
        store.*.seq = s.seq -| 1;
    };
    clearPassBinds(s);
    s.pass_kernels = .{};
    s.pass_dispatches = 0;
    return error.ReadsFillingSlab;
}

fn clearPassBinds(s: *State) void {
    s.pass_binds.clearRetainingCapacity();
    s.pass_writes.clearRetainingCapacity();
    s.pass_mapped.clearRetainingCapacity();
}

fn clearHazards(s: *State) void {
    s.prev_in.clearRetainingCapacity();
    s.prev_out.clearRetainingCapacity();
    s.next_in.clearRetainingCapacity();
    s.next_out.clearRetainingCapacity();
    s.needs_barrier = false;
}

/// Hands the open pass to the GPU without waiting for it.
pub fn submit() Error!void {
    const s = try st();
    const open = s.pass orelse return;
    s.pass = null;
    try commitPass(s, open);
}

/// Submits the open pass and waits until the GPU has run everything submitted.
pub fn sync() Error!void {
    return syncState(try st());
}

/// A store over an MLX array that waits for a pass of ours is freed now, its passes waited on. MLX's allocator goes to
/// rest after this, where the free of its array would be a call into MLX outside its turn.
pub fn settleMlxAliases() void {
    const s = &(state orelse return);
    for (s.retired.items) |store| {
        if (store.origin == .mlx) break;
    } else return;
    syncState(s) catch {};
}

/// Ends the open pass and waits for everything submitted; nothing when the engine never ran. A thread that ends
/// with an encoder still open trips a Metal assertion as its autorelease pool drains.
pub fn close() void {
    const s = &(state orelse return);
    syncState(s) catch {};
    metal.commitResident();
    readGpuSpans(s, 0);
}

/// No pass is open and none is running.
pub fn idle() bool {
    const s = &(state orelse return true);
    return s.pass == null and s.flying.items.len == 0;
}

/// A longer pause is the host between forwards (sampling, a request's end), not a pass the GPU waited for.
const gap_limit_s: f64 = 0.05;

/// Reads the times of the finished command buffers in order, stopping at one Metal has not timed yet unless more
/// than `backlog` wait behind it.
fn readGpuSpans(s: *State, backlog: usize) void {
    var n: usize = 0;
    for (s.untimed.items) |done| {
        const span = done.cmd.gpuSpan();
        if ((span[0] <= 0 or span[1] < span[0]) and s.untimed.items.len - n <= backlog) break;
        noteGpuSpan(s, span, done.phase);
        if (done.kernels.n > 0 and span[0] > 0 and span[1] >= span[0]) noteKernels(done.kernels, @intFromFloat((span[1] - span[0]) * 1e9));
        done.cmd.release();
        n += 1;
    }
    s.untimed.replaceRangeAssumeCapacity(0, n, &.{});
}

fn noteGpuSpan(s: *State, span: [2]f64, before: u8) void {
    if (span[0] <= 0 or span[1] < span[0]) {
        s.stats.gpu_untimed += 1;
        s.gpu_last_end = 0;
        return;
    }
    const from = @max(span[0], s.gpu_last_end);
    if (span[1] > from) s.stats.gpu_busy_ns += @intFromFloat((span[1] - from) * 1e9);
    if (s.gpu_last_end > 0 and span[0] > s.gpu_last_end and span[0] - s.gpu_last_end < gap_limit_s) {
        const gap: u64 = @intFromFloat((span[0] - s.gpu_last_end) * 1e9);
        s.stats.gpu_gap_ns += gap;
        const bin: usize = if (gap < 20_000) 0 else if (gap < 100_000) 1 else if (gap < 1_000_000) 2 else 3;
        s.stats.gpu_gap_count[bin] += 1;
        s.stats.gpu_gap_sum[bin] += gap;
        s.stats.gpu_gap_phase[@min(before, phases - 1)] += gap;
    }
    s.gpu_last_end = @max(s.gpu_last_end, span[1]);
}

fn syncState(s: *State) Error!void {
    return waitFor(s, s.seq);
}

/// Waits until pass `upto` and every pass before it have run; the open pass is submitted first when it is one of
/// them. Later passes keep running.
fn waitFor(s: *State, upto_in: u64) Error!void {
    if (s.pass) |open| {
        if (upto_in >= s.seq) {
            s.pass = null;
            try commitPass(s, open);
        }
    }
    const upto = @min(upto_in, s.seq - 1);
    if (upto <= s.done) return;
    s.stats.waits += 1;
    s.kick_after = kick_first;
    s.burst = false;
    var failed: ?Error = null;
    const t0 = nowNs();
    // Oldest first: the passes behind one that failed may never run, and it is found before they are waited for.
    for (s.flying.items) |f| {
        if (f.seq > upto) break;
        f.cmd.waitDone();
        if (f.cmd.failed()) {
            failed = error.MetalCommand;
            break;
        }
    }
    // A failed pass ends the rest for the caller, not for the GPU: the passes behind it run on, and each still binds
    // its buffers until it ends.
    if (failed != null) for (s.flying.items) |f| f.cmd.waitDone();
    s.stats.wait_ns += nowNs() - t0;
    s.done = if (failed == null) upto else s.seq - 1;
    // Passes end in the order of their numbers: the ones behind that have ended too are done.
    if (failed == null) for (s.flying.items) |f| {
        if (f.seq <= s.done) continue;
        if (!f.cmd.ended()) break;
        s.done = f.seq;
    };
    var kept: usize = 0;
    for (s.flying.items) |f| {
        if (f.seq > s.done) {
            s.flying.items[kept] = f;
            kept += 1;
            continue;
        }
        if (failed == null and f.cmd.failed()) failed = error.MetalCommand;
        s.untimed.append(gpa, .{ .cmd = f.cmd, .phase = f.phase, .kernels = f.kernels }) catch {
            s.stats.gpu_untimed += 1;
            s.gpu_last_end = 0;
            f.cmd.release();
        };
        f.fence.release();
    }
    s.flying.shrinkRetainingCapacity(kept);
    readGpuSpans(s, 64);
    if (kept == 0) s.writers.clearRetainingCapacity();
    kept = 0;
    for (s.retired.items) |store| {
        if (store.seq > s.done) {
            s.retired.items[kept] = store;
            kept += 1;
        } else freeAlias(s, store);
    }
    s.retired.shrinkRetainingCapacity(kept);
    metal.commitResident();
    takeBack();
    if (failed) |e| return e;
}

/// Whether every pass that binds `t`'s bytes has run.
pub fn settled(t: *const Tensor) bool {
    return t.store.seq <= state.?.done;
}

/// `fresh` for a tensor the host fills: its buffer is one no submitted pass still binds.
pub fn freshForHost(dtype: Dtype, shape: []const c_int) Error!*Tensor {
    return freshFor(dtype, shape, true);
}

/// `t`'s bytes on the host once every pass that binds them has run. Row-contiguous tensors only.
pub fn hostBytes(t: *const Tensor) Error![]u8 {
    if (!t.rowContiguous()) return error.NotRowContiguous;
    const s = try st();
    if (t.store.seq > s.done) try waitFor(s, t.store.seq);
    return (t.store.ptr + t.offset)[0..t.nbytes()];
}

test "a fresh tensor comes from the pool and goes back to it" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    const a = try fresh(.float32, &.{ 3, 5 });
    try t.expectEqual(@as(usize, 15), a.size());
    try t.expectEqual(@as(usize, 60), a.nbytes());
    try t.expectEqualSlices(usize, &.{ 5, 1 }, a.strides[0..2]);
    try t.expect(a.rowContiguous());
    const first = a.store;
    const pooled = stats().pool_bytes;
    release(a);
    const b = try fresh(.uint8, &.{100});
    defer release(b);
    try t.expectEqual(first, b.store);
    try t.expectEqual(pooled, stats().pool_bytes);
}

test "a second release of a tensor is refused by name: its box and its store are given back once" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    const a = try fresh(.uint8, &.{100});
    release(a);
    const free_bytes = poolFree();
    const refused = double_releases;
    release(a);
    try t.expectEqual(refused + 1, double_releases);
    try t.expect(mlx.takeErrorIf("released twice"));
    try t.expectEqual(free_bytes, poolFree());
    const b = try fresh(.uint8, &.{100});
    defer release(b);
    const c = try fresh(.uint8, &.{100});
    defer release(c);
    try t.expect(b != c and b.store != c.store);
}

test "the pool's high-waters keep what was in use at its fullest, each from its own reset" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    const mb = 1 << 20;
    try sync();
    resetPoolPeak(.process);
    resetPoolPeak(.phase);
    const in_use = poolTotal() - poolFree();
    try t.expectEqual(in_use, poolPeak(.process));
    const a = try fresh(.uint8, &.{9 * mb + 5});
    const full = poolTotal() - poolFree();
    try t.expect(full >= in_use + 9 * mb);
    release(a);
    try t.expectEqual(full, poolPeak(.process));
    resetPoolPeak(.phase);
    try t.expectEqual(in_use, poolPeak(.phase));
    const b = try fresh(.uint8, &.{3 * mb + 5});
    const part = poolTotal() - poolFree();
    release(b);
    try t.expectEqual(part, poolPeak(.phase));
    try t.expectEqual(full, poolPeak(.process));
}

test "the pool's byte mirrors follow every allocation, return and release, and an emptied pool reads zero" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    const mb = 1 << 20;
    try sync();
    trim();
    try t.expectEqual(stats().pool_bytes, poolTotal());
    try t.expectEqual(stats().pool_free, poolFree());
    const in_use = poolTotal() - poolFree();
    const a = try fresh(.uint8, &.{3 * mb + 5});
    const b = try fresh(.uint8, &.{9 * mb + 5});
    try t.expectEqual(stats().pool_bytes, poolTotal());
    try t.expect(poolTotal() - poolFree() >= in_use + 12 * mb);
    release(a);
    release(b);
    try sync();
    try t.expectEqual(stats().pool_free, poolFree());
    try t.expect(poolFree() >= 12 * mb);
    trim();
    try t.expectEqual(@as(u64, 0), poolFree());
    try t.expectEqual(stats().pool_bytes, poolTotal());
    try t.expectEqual(in_use, poolTotal());
}

test "a pool buffer is whole pages and under a sixteenth larger than what was asked" {
    const t = std.testing;
    var len: usize = 1;
    var last: usize = 0;
    while (len < 1 << 34) : (len += len / 7 + 4097) {
        const bytes = classBytes(classOf(len));
        try t.expect(bytes >= len and bytes >= last);
        try t.expectEqual(@as(usize, 0), bytes % page_bytes);
        try t.expect(bytes - len < page_bytes or (bytes - len) * 16 < len);
        // The class of its own size: a returned buffer is found again.
        try t.expectEqual(classOf(len), classOf(bytes));
        last = bytes;
    }
    try t.expectEqual(@as(usize, page_bytes), classBytes(classOf(1)));
    try t.expectEqual(@as(usize, 17 * page_bytes), classBytes(classOf(16 * page_bytes + 1)));
}

test "under a budget the pool gives its free buffers up for a new one, and refuses what does not fit" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    const s = try st();
    const was = s.pool_limit;
    defer s.pool_limit = was;
    const mb = 1 << 20;
    // Nothing free to start with, so the free lists hold only what this test returns.
    try sync();
    try budget(0);
    s.pool_limit = null;
    const kept = try fresh(.uint8, &.{3 * mb + 5});
    defer release(kept);
    const a = try fresh(.uint8, &.{5 * mb + 5});
    const b = try fresh(.uint8, &.{7 * mb + 5});
    release(a);
    release(b);
    try sync();
    // The budget counts what the tensors in use ask for; the free buffers past it go, the largest first.
    try budget(6 * mb);
    const held = stats();
    try t.expect(held.pool_bytes <= held.pool_asked + 6 * mb);
    try t.expectEqual(@as(usize, 0), s.free[classOf(7 * mb + 5)].items.len);
    try t.expectEqual(@as(usize, 1), s.free[classOf(5 * mb + 5)].items.len);
    // A new size inside the budget takes the room of the free buffer.
    const c = try fresh(.uint8, &.{4 * mb + 5});
    try t.expectEqual(@as(usize, 0), s.free[classOf(5 * mb + 5)].items.len);
    try t.expect(stats().pool_bytes <= held.pool_asked + 6 * mb);
    // One past it is refused, and the pool is as it was.
    const before = stats().pool_bytes;
    try t.expectError(error.PoolOverBudget, fresh(.uint8, &.{9 * mb}));
    try t.expectEqual(before, stats().pool_bytes);
    release(c);
}

test "what is built with the cap lifted stays, and the cap comes back raised by it" {
    const t = std.testing;
    const s = try st();
    const was = s.pool_limit;
    defer s.pool_limit = was;
    const mb = 1 << 20;
    try sync();
    try budget(0);
    try t.expectError(error.PoolOverBudget, fresh(.uint8, &.{3 * mb}));
    const lifted = uncapped();
    const kept = try fresh(.uint8, &.{3 * mb});
    defer release(kept);
    const scratch = try fresh(.uint8, &.{5 * mb});
    release(scratch);
    try sync();
    capped(lifted);
    // The cap is the old one and what stayed: the scratch's buffer went, and nothing more fits.
    try t.expectEqual(lifted.limit.? + classBytes(classOf(3 * mb)), s.pool_limit.?);
    try t.expectEqual(@as(usize, 0), s.free[classOf(5 * mb)].items.len);
    try t.expectError(error.PoolOverBudget, fresh(.uint8, &.{3 * mb}));
}

test "GPU time is the union of the command buffers' spans, and a short pause between two is idle time" {
    const t = std.testing;
    var s: State = .{ .dev = undefined };
    noteGpuSpan(&s, .{ 1.000, 1.002 }, 0);
    noteGpuSpan(&s, .{ 1.003, 1.004 }, 2);
    // A buffer that began before the last one ended adds only what runs past it.
    noteGpuSpan(&s, .{ 1.0035, 1.005 }, 0);
    try t.expectEqual(@as(u64, 4_000_000), (s.stats.gpu_busy_ns + 500) / 1000 * 1000);
    try t.expectEqual(@as(u64, 1_000_000), (s.stats.gpu_gap_ns + 500) / 1000 * 1000);
    try t.expectEqual(s.stats.gpu_gap_ns, s.stats.gpu_gap_phase[2]);
    // The host between forwards is not the GPU waiting for a pass.
    noteGpuSpan(&s, .{ 2.000, 2.001 }, 0);
    try t.expectEqual(@as(u64, 1_000_000), (s.stats.gpu_gap_ns + 500) / 1000 * 1000);
    // A buffer Metal has not timed yet says nothing, and neither does the pause after it.
    noteGpuSpan(&s, .{ 0, 0 }, 0);
    noteGpuSpan(&s, .{ 2.002, 2.003 }, 0);
    try t.expectEqual(@as(u64, 1), s.stats.gpu_untimed);
    try t.expectEqual(@as(u64, 1_000_000), (s.stats.gpu_gap_ns + 500) / 1000 * 1000);
    try t.expectEqual(@as(u64, 6_000_000), (s.stats.gpu_busy_ns + 500) / 1000 * 1000);
}

test "a tensor handle carries the tag MLX's handles never have" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    const a = try fresh(.int32, &.{4});
    defer release(a);
    const h = handle(a);
    try t.expect(isOurs(h));
    try t.expectEqual(a, of(h));
    const real = mlx.mlx_array_new_float(1);
    defer _ = mlx.mlx_array_free(real);
    try t.expect(!isOurs(real));
    try t.expect(!isOurs(mlx.mlx_array_new()));
    try t.expect(isStream(stream()));
}

test "host bytes copied in are the tensor's bytes, and a view shares them" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    const data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const a = try fromHost(std.mem.sliceAsBytes(&data), .float32, &.{ 2, 3 });
    defer release(a);
    const v = try share(a);
    defer release(v);
    try t.expectEqual(@as(u32, 2), a.store.refs);
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&data), try hostBytes(v));
    try t.expectError(error.TensorShape, fromHost(std.mem.sliceAsBytes(&data), .float32, &.{ 2, 2 }));
}

fn mlxFloats(vals: []const f32) mlx.mlx_array {
    return mlx.raw.mlx_array_new_data(vals.ptr, &[_]c_int{@intCast(vals.len)}, 1, .float32);
}

test "an MLX array imports fresh each time, and its store can still be written in place" {
    const t = std.testing;
    try mlx.requireGpu();
    const a = mlxFloats(&.{ 1, 2, 3, 4 });
    const before = stats();
    const x = try import(a);
    defer release(x);
    const y = try import(a);
    try t.expect(x.store != y.store);
    try t.expectEqual(before.imports + 2, stats().imports);
    // Only x's store holds the array now: MLX's own rule lets a write go into it.
    release(y);
    _ = mlx.raw.mlx_array_free(a);
    try t.expect(try sole(x.store));
}

