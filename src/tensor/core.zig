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
const fill_marks = @import("../fill_marks.zig");
const plan = @import("plan/plan_api.zig");

pub const max_dims = 8;
pub const Dtype = mlx.mlx_dtype;

pub const Error = metal.Error || error{ OutOfMemory, MlxError, NotEvaluated, NotRowContiguous, TensorShape, TensorDtype, TensorUnsupported, TensorNull, MlxAtRest, PoolOverBudget, ReadsFillingSlab, EnvUnparsed, DryReadback, DryHostWrite, DryStore, PlanMiss, PlanNeedsLease, PlanEscape, PoolClosed, WideFull };

/// The restriction behind the last `TensorUnsupported` or `TensorDtype` an op returned through `refusing`; the route
/// that reports the refusal names it once and clears it.
pub var refused_why: ?[]const u8 = null;

/// `e` for a call the port refuses, with the restriction it enforces: a string that outlives the call.
pub fn refusing(why: []const u8, e: Error) Error {
    refused_why = why;
    return e;
}

/// A shape built a piece at a time: past `max_dims` axes a piece is `TensorShape`, never a write past the buffer.
pub const Dims = struct {
    buf: [max_dims]c_int = undefined,
    len: usize = 0,

    pub fn of(dims: []const c_int) Error!Dims {
        var d: Dims = .{};
        try d.appendSlice(dims);
        return d;
    }

    pub fn append(self: *Dims, d: c_int) Error!void {
        if (self.len == max_dims) return error.TensorShape;
        self.buf[self.len] = d;
        self.len += 1;
    }

    pub fn appendSlice(self: *Dims, ds: []const c_int) Error!void {
        if (ds.len > max_dims - self.len) return error.TensorShape;
        @memcpy(self.buf[self.len..][0..ds.len], ds);
        self.len += ds.len;
    }

    pub fn slice(self: *const Dims) []const c_int {
        return self.buf[0..self.len];
    }
};

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
    /// The recording that asked for it (`Recording.gen`, 0 for none) and its ask there.
    plan_gen: u32 = 0,
    plan_ref: plan.Ref = .{ .instance = 0, .ordinal = 0 },
    /// The recording one of whose dispatches wrote it: a host read of it then reads that execution's GPU work (D10).
    wrote_gen: u32 = 0,
    /// A dry recording's dispatch wrote it, dropped: its bytes are not the ones its tensors say, so no real dispatch or
    /// host read takes it again until it is freed (the counter page, which every launch leaves zero, aside).
    dry_written: bool = false,
    /// An arena store: the pass and the highest level of its last dispatches (`markOf`), which the next tenant of its
    /// bytes goes above; its index in `Arena.live`; and whether its execution ended with it alive (`escaped`): its
    /// bytes are another execution's from then, and a bind or a host read of it is refused.
    level_seq: u64 = 0,
    level_hi: u32 = 0,
    lev_seq: u64 = 0,
    lev_read: u32 = 0,
    lev_write: u32 = 0,
    lev_whole: u32 = 0,
    live_at: u32 = 0,
    escaped: bool = false,

    const Origin = union(enum) {
        /// An ask of a dry recording: no buffer. A host ask keeps a host scratch of its bytes, which its filler writes
        /// (D8); it lives only while recording and is no arena or pool memory.
        dry: ?[]u8,
        /// A buffer of the pool, by size class.
        pool: u16,
        /// The memory of an MLX array, which this handle keeps alive.
        mlx: mlx.mlx_array,
        /// Memory its maker mapped and the engine wrapped (`wrapMapped`): `release(payload)` gives it back. `owned`: a
        /// state owner's buffer (`wrapOwned`), written in place while it is sole; else its maker's (a weight), never.
        mapped: struct { payload: ?*anyopaque, release: MappedRelease, owned: bool = false },
        /// A buffer the engine holds for the process (`counterPage`): never released, never written in place.
        held,
        /// A place in the arena (`beginExecution`), the plan binder's for one execution: its slot goes back to the
        /// binder when its last reference goes. A host-filled ask, from the host ring, holds no slot.
        arena: plan.binder.Slot,
        /// A place of a wide execution in the arena's scratch (`beginWide`), `len` bytes from `at`, given back to the
        /// region's gaps when its last reference goes; a host-filled one, in the host ring's replica, at the next eval
        /// (`evalPoint`).
        wide: struct { at: u64, len: u64, host: bool },
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

const pool_class = @import("pool_class.zig");
const page_bytes = pool_class.page_bytes;
const classes = pool_class.classes;

const Stores = std.AutoHashMapUnmanaged(*Store, void);

/// A submitted pass: its command buffer, the fence it reaches when it has run, and its number.
const Flying = struct { cmd: metal.Committed, fence: metal.Fence, seq: u64, phase: u8 = 0, kernels: Kernels = .{} };

/// What the host is building, set by the model: a pause is charged to the phase of the pass that ends it.
pub var phase: u8 = 0;
pub const phases = 6;
const Finished = struct { cmd: metal.Committed, phase: u8, kernels: Kernels };

/// The kernels a pass dispatched, by name, and the bytes they bind to read and to write (each store once a dispatch);
/// filled only while the kernel profile is on.
const Kernels = struct {
    n: u32 = 0,
    named: u8 = 0,
    names: [8][]const u8 = undefined,
    read: u64 = 0,
    write: u64 = 0,
    /// The return addresses above the op's first dispatch: the code that made it.
    site: [site_frames]usize = undefined,
    site_n: u8 = 0,

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
    /// The stores the dispatch being encoded reads and writes, which give its level (`dispatchLevel`).
    next_in: Stores = .empty,
    next_out: Stores = .empty,
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
    /// Whether an ask outside a wide phase may reach the pool (`closePool`), and how many wide phases are open.
    pool_gate: PoolGate = .open,
    wide: u32 = 0,
    /// Dispatches in the open pass, and how many an early hand-over waits for (`pass`).
    pass_dispatches: u32 = 0,
    kick_after: u32 = kick_first,
    /// Pool bytes freed while the open pass still binds them (`pass_freed_cap`).
    pass_freed: u64 = 0,
    /// A pass was handed over early since the host last waited: the ones after it follow without the GPU being idle.
    burst: bool = false,
    /// When the GPU ended the last command buffer whose times were read; 0 after one that had none.
    gpu_last_end: f64 = 0,
    /// `Stats.gpu_busy_ns` and `gpu_gap_ns` at the last `takeGpuTime`.
    gpu_taken: [2]u64 = .{ 0, 0 },
    /// Command buffers that have run and whose GPU times are still to read, oldest first: Metal sets them a moment
    /// after the event a wait returns on.
    untimed: std.ArrayList(Finished) = .empty,
    pass_kernels: Kernels = .{},
    stats: Stats = .{},
    /// The open pass's dispatches, held until its hand-over encodes them by level (`metal.Recorder`). A dispatch's level
    /// is one past the levels in this pass of the writes it reads and, for a write, of the reads and writes before it
    /// (a disjoint write, of the reads and the whole writes): independent work runs beside, whatever order it was made in.
    rec: metal.Recorder = .{},
    /// The highest level of a dispatch in the open pass: with its number, the engine's mark (`markNow`).
    pass_level_hi: u32 = 0,
    /// The counter page (`counterPage`), made at its first ask.
    counters: ?*Tensor = null,
};

pub const Stats = struct {
    dispatches: u64 = 0,
    barriers: u64 = 0,
    /// Dispatches encoded a second time by the double switch (`kernelDoubled`), not counted in `dispatches`.
    doubled: u64 = 0,
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
    /// Passes handed over because they had freed `pass_freed_cap` bytes they still bind, or, in an execution on the
    /// arena, because they held `execution_pass_bounds.dispatches`.
    freed_kicks: u64 = 0,
    bound_kicks: u64 = 0,
    /// The most dispatches one pass held, and the most passes committed and unfinished at once.
    longest_pass: u64 = 0,
    most_flying: u64 = 0,
    /// Passes a caller handed over by `submit`, outside the kicks and the waits.
    handovers: u64 = 0,
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
    /// Tensors asked of the pool since the engine began, and their bytes (`freshFor`): a recording holds each one.
    asks: u64 = 0,
    ask_bytes: u64 = 0,
    /// Asks made while the pool was closed and no wide phase open, and their bytes: refused, or served by a census.
    closed_asks: u64 = 0,
    closed_ask_bytes: u64 = 0,
    /// Executions on the arena, the places they asked for, and those placed over bytes a dispatch of the open pass
    /// still binds: each takes that dispatch's level as its floor (`markNow`).
    executions: u64 = 0,
    arena_asks: u64 = 0,
    arena_floors: u64 = 0,
    /// Wide executions (D97), the places they asked for, and their scratch's highest top.
    wide_executions: u64 = 0,
    wide_asks: u64 = 0,
    wide_high: u64 = 0,
    /// The last wide execution's own highest top.
    wide_last_high: u64 = 0,
};

const gpa = std.heap.c_allocator;
var state: ?State = null;

fn st() Error!*State {
    if (state == null) {
        double_env = doubleSwitch(std.c.getenv("SUSHI_ENGINE_KERNEL_DOUBLE")) catch |e| {
            std.log.err("SUSHI_ENGINE_KERNEL_DOUBLE takes text from a kernel's pipeline name (letters, digits, _; not a number), or 0 or off", .{});
            return e;
        };
        state = .{ .dev = try metal.device() };
    }
    return &state.?;
}

pub fn stats() Stats {
    return if (state) |s| s.stats else .{};
}

/// GPU busy and idle-between-passes time, in ns, since the last take: what the forwards between two calls took.
fn takeGpuTime(s: *State) [2]u64 {
    const now = [2]u64{ s.stats.gpu_busy_ns, s.stats.gpu_gap_ns };
    defer s.gpu_taken = now;
    return .{ now[0] - s.gpu_taken[0], now[1] - s.gpu_taken[1] };
}

pub fn gpuTimeTaken() [2]u64 {
    return if (state) |*s| takeGpuTime(s) else .{ 0, 0 };
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

/// Like `touch`, waiting for Metal to finish wiring the committed residency sets before returning.
pub fn touchSync() void {
    const s = &(state orelse return);
    if (s.stats.mapped_bytes == 0) return metal.commitResident();
    (metal.device() catch return).touchSync();
}

/// Which high-water of the pool's bytes in use: the one the process's peak reading resets, or a forward phase's.
pub const Peak = enum { process, phase };
var pool_peaks: [2]std.atomic.Value(u64) = .{ .init(0), .init(0) };
var held_peaks: [2]std.atomic.Value(u64) = .{ .init(0), .init(0) };

pub fn poolPeak(which: Peak) u64 {
    return pool_peaks[@backingInt(which)].load(.monotonic);
}

/// The high-water of what the pool holds, its free buffers included: what Metal counts against the device.
pub fn poolHeldPeak(which: Peak) u64 {
    return held_peaks[@backingInt(which)].load(.monotonic);
}

/// Both high-waters start again from now: in use, and held.
pub fn resetPoolPeak(which: Peak) void {
    pool_peaks[@backingInt(which)].store(poolTotal() - poolFree(), .monotonic);
    if (which == .process) {
        peak_set.clearRetainingCapacity();
        peak_set_in_use = poolTotal() - poolFree();
    }
    held_peaks[@backingInt(which)].store(poolTotal(), .monotonic);
}

fn publishPool(s: *const State) void {
    pool_total.store(s.stats.pool_bytes, .monotonic);
    pool_unused.store(s.stats.pool_free, .monotonic);
    const in_use = s.stats.pool_bytes - s.stats.pool_free;
    for (&pool_peaks) |*peak| if (in_use > peak.load(.monotonic)) peak.store(in_use, .monotonic);
    for (&held_peaks) |*peak| if (s.stats.pool_bytes > peak.load(.monotonic)) peak.store(s.stats.pool_bytes, .monotonic);
}

// ── Kernel profile ──

/// Tests set this; otherwise `SUSHI_ENGINE_KERNEL_PROF` decides, read once.
pub var kernel_prof_hook: ?bool = null;
var kernel_prof_env: ?bool = null;

/// With the profile on, every op's pass is handed over on its own when the next op's begins, without a wait, so
/// the GPU stays fed and the time from the end of the buffer before to the end of the op's is the op's. Timing
/// changes; values do not.
pub fn kernelProfOn() bool {
    return kernel_prof_hook orelse io_util.diagEnvOnCached(&kernel_prof_env, "SUSHI_ENGINE_KERNEL_PROF");
}

pub const KernelStat = struct {
    name: []const u8,
    dispatches: u64,
    total_ns: u64,
    max_ns: u64,
    read_bytes: u64 = 0,
    write_bytes: u64 = 0,
    site: [site_frames]usize = undefined,
    site_n: u8 = 0,
};

const site_frames = 40;

/// A profile row is the names of an op's kernels and the code that made it: one kernel at two sites is two rows, and
/// one site's ops of every size are one.
const key_sep = 0;

/// Pipeline handle to the name it was made by.
var pipeline_names: std.AutoHashMapUnmanaged(usize, []const u8) = .empty;
var profile: std.StringHashMapUnmanaged(KernelStat) = .empty;
var profile_view: std.ArrayList(KernelStat) = .empty;

fn pipelineName(raw: metal.id) []const u8 {
    return pipeline_names.get(@intFromPtr(raw)) orelse "(unnamed)";
}

/// One entry per distinct name and site; a pass of several dispatches is one entry named by all of them, joined by " + ".
fn noteKernels(k: Kernels, ns: u64) void {
    var buf: [1024 + 1 + 8]u8 = undefined;
    var len: usize = 0;
    for (k.names[0..k.named], 0..) |name, i| {
        const sep = if (i > 0) " + " else "";
        if (len + sep.len + name.len > 1024) break;
        @memcpy(buf[len..][0..sep.len], sep);
        len += sep.len;
        @memcpy(buf[len..][0..name.len], name);
        len += name.len;
    }
    const name_len = len;
    buf[len] = key_sep;
    len += 1;
    const site_hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(k.site[0..k.site_n]));
    @memcpy(buf[len..][0..8], std.mem.asBytes(&site_hash));
    len += 8;
    const key = buf[0..len];
    if (profile.getPtr(key)) |e| {
        e.dispatches += k.n;
        e.total_ns += ns;
        e.max_ns = @max(e.max_ns, ns);
        e.read_bytes += k.read;
        e.write_bytes += k.write;
        return;
    }
    const owned = gpa.dupe(u8, key) catch return;
    profile.put(gpa, owned, .{ .name = owned[0..name_len], .dispatches = k.n, .total_ns = ns, .max_ns = ns, .read_bytes = k.read, .write_bytes = k.write, .site = k.site, .site_n = k.site_n }) catch gpa.free(owned);
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

/// Kernels a profile lists by name before it sums the rest as one line: every kernel where the switch reads `all`.
fn profileTop(raw: ?[*:0]const u8) usize {
    const v = raw orelse return 40;
    return if (std.mem.eql(u8, std.mem.span(v), "all")) std.math.maxInt(usize) else 40;
}

test "a kernel profile names its top 40 kernels, and every kernel where the switch reads all" {
    try std.testing.expectEqual(@as(usize, 40), profileTop(null));
    try std.testing.expectEqual(@as(usize, 40), profileTop("1"));
    try std.testing.expectEqual(@as(usize, std.math.maxInt(usize)), profileTop("all"));
}

/// The executable's own address of a return address (the load's slide taken off), which `atos -i -o <binary>` reads
/// back to its function, inlined callers and lines.
fn fileAddress(addr: usize) usize {
    return if (comptime builtin.os.tag.isDarwin()) addr -% std.c._dyld_get_image_vmaddr_slide(0) else addr;
}

/// Debug lines for the profile, `per` forwards to the totals; nothing while the switch is off.
pub fn logKernelProfile(per: f64) void {
    if (!kernelProfOn()) return;
    const top = profileTop(std.c.getenv("SUSHI_ENGINE_KERNEL_PROF"));
    var other: KernelStat = .{ .name = "", .dispatches = 0, .total_ns = 0, .max_ns = 0 };
    var more: usize = 0;
    for (kernelProfile(), 0..) |k, i| {
        if (i >= top) {
            other.dispatches += k.dispatches;
            other.total_ns += k.total_ns;
            other.read_bytes += k.read_bytes;
            other.write_bytes += k.write_bytes;
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
    const r: f64 = @floatFromInt(k.read_bytes);
    const w: f64 = @floatFromInt(k.write_bytes);
    var site: [site_frames * 19]u8 = undefined;
    var sw = std.Io.Writer.fixed(&site);
    for (k.site[0..k.site_n]) |addr| sw.print(" 0x{x}", .{fileAddress(addr)}) catch break;
    log.debug("[engine-prof] {s}: {d:.1} dispatches, {d:.3} ms total, {d:.2} us each, reads {d:.0} B, writes {d:.0} B a forward; at{s}\n", .{ name, n / per, ns / 1e6 / per, ns / 1e3 / @max(n, 1), r / per, w / per, sw.buffered() });
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

/// The store under `a` as a key, one a buffer: `storePointer` is its bytes, which every dry store shares (`dry_ptr`).
pub fn storeKey(a: mlx.mlx_array) usize {
    return @intFromPtr(of(a).store);
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

const classOf = pool_class.classOf;
/// What a tensor of `len` bytes takes from the pool: its size class.
pub const poolBytes = pool_class.poolBytes;
const classBytes = pool_class.classBytes;

/// TEST-ONLY, `mlx.fault` for the engine's allocations: `arm(k)` makes the pool refuse its k-th buffer from here with
/// `PoolOverBudget`, so a test walks every error path of a forward without patching it. Compiled only under test.
pub const fault = struct {
    var remaining: u64 = 0; // 0 = disarmed
    var asked: u64 = 0;

    pub fn arm(k: u64) void {
        remaining = k;
        sites.armed();
    }
    pub fn disarm() void {
        remaining = 0;
        sites.disarmed();
    }
    /// Refuses the buffer of the n-th ask at `p.site`, whatever order the asks come in.
    pub fn armAt(p: mlx.FaultSites.SitePoint) void {
        remaining = 0;
        sites.armAt(p);
    }
    pub fn armed() bool {
        return remaining != 0 or sites.target != null;
    }
    /// Buffers the pool was asked for since the process began: the difference over a request is its sweep's length.
    pub fn count() u64 {
        return asked;
    }
    /// Recording for a sweep by site (`mlx.sweepPoints`).
    pub var sites: mlx.FaultSites = .{};
    fn hit() bool {
        asked += 1;
        switch (sites.take()) {
            .noted, .pass => return false,
            .fail => return true,
            .count => {},
        }
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
// ── Pool peak trace ──

/// Tests set this; otherwise `SUSHI_ENGINE_POOL_PEAK_TRACE` decides, read once.
pub var peak_trace_hook: ?bool = null;
var peak_trace_env: ?bool = null;
fn peakTraceOn() bool {
    return peak_trace_hook orelse io_util.diagEnvOnCached(&peak_trace_env, "SUSHI_ENGINE_POOL_PEAK_TRACE");
}

/// A pool ask in use: its bytes and its caller's return addresses, innermost first, past the pool's own frames.
pub const PeakAsk = struct { len: usize, at: [8]usize };
var peak_live: std.AutoHashMapUnmanaged(usize, PeakAsk) = .empty;
var peak_set: std.ArrayList(PeakAsk) = .empty;
var peak_set_in_use: u64 = 0;

/// The asks in use when the pool was fullest since the process peak's last reset.
pub fn peakTrace() []const PeakAsk {
    return peak_set.items;
}

noinline fn tracePoolAsk(store: *const Store, in_use: u64) void {
    var ask: PeakAsk = .{ .len = store.len, .at = @splat(0) };
    var addrs: [ask.at.len + 2]usize = undefined;
    const all = std.debug.captureCurrentStackTrace(.{}, &addrs).return_addresses;
    // This frame and `poolStore`'s.
    if (all.len > 2) for (all[2..], 0..) |r, i| {
        ask.at[i] = r;
    };
    peak_live.put(gpa, @intFromPtr(store), ask) catch return;
    if (in_use <= peak_set_in_use) return;
    peak_set_in_use = in_use;
    peak_set.clearRetainingCapacity();
    var it = peak_live.valueIterator();
    while (it.next()) |v| peak_set.append(gpa, v.*) catch return;
}

/// Debug lines for the trace: the asks in use at the pool's fullest, by call site, largest first; nothing while the
/// switch is off.
pub fn logPoolPeakTrace() void {
    if (!peakTraceOn()) return;
    const log = @import("../log.zig");
    var sites: std.AutoArrayHashMapUnmanaged([8]usize, struct { bytes: u64, n: u64 }) = .empty;
    defer sites.deinit(gpa);
    var total: u64 = 0;
    for (peak_set.items) |ask| {
        const g = sites.getOrPut(gpa, ask.at) catch return;
        if (!g.found_existing) g.value_ptr.* = .{ .bytes = 0, .n = 0 };
        g.value_ptr.bytes += ask.len;
        g.value_ptr.n += 1;
        total += ask.len;
    }
    const Order = struct {
        v: []const @TypeOf(sites.values()[0]),
        pub fn lessThan(o: @This(), a: usize, b: usize) bool {
            return o.v[a].bytes > o.v[b].bytes;
        }
    };
    sites.sort(Order{ .v = sites.values() });
    log.debug("[engine-peak] {d} MB asked by {d} tensors at the pool's fullest ({d} MB in use), {d} call sites\n", .{ total >> 20, peak_set.items.len, peak_set_in_use >> 20, sites.count() });
    for (sites.keys(), sites.values()) |at, v| {
        log.debug("[engine-peak] {d} KB in {d} tensors from:\n", .{ v.bytes >> 10, v.n });
        var addrs = at;
        const n = std.mem.indexOfScalar(usize, &addrs, 0) orelse addrs.len;
        std.debug.dumpStackTrace(&.{ .return_addresses = addrs[0..n], .skipped = .unknown });
    }
}

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
        if (peakTraceOn()) tracePoolAsk(store, s.stats.pool_bytes - s.stats.pool_free);
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
    if (peakTraceOn()) tracePoolAsk(store, s.stats.pool_bytes - s.stats.pool_free);
    publishPool(s);
    return store;
}

/// Releases free buffers until the pool holds at most `bytes`, waiting for the passes that still bind some when
/// the idle ones are not enough, the open one too where no dispatch is between its binds. False when the buffers in
/// use hold more, or those a dispatch being bound still pins.
fn trimTo(s: *State, bytes: u64) Error!bool {
    if (s.stats.pool_bytes <= bytes) return true;
    if (evict(s, bytes)) return true;
    // Only the passes that free enough: the ones after them keep the GPU busy while the host goes on.
    try waitFor(s, try pendingFrees(s, bytes) orelse s.seq - 1);
    if (evict(s, bytes)) return true;
    if (s.pass == null or s.next_in.count() != 0 or s.next_out.count() != 0) return false;
    try waitFor(s, s.seq);
    return evict(s, bytes);
}

/// The committed pass whose end lets `evict` bring the pool to `bytes`, from the free buffers passes still bind.
fn pendingFrees(s: *State, bytes: u64) Error!?u64 {
    var pending: std.ArrayList([2]u64) = .empty;
    defer pending.deinit(gpa);
    for (s.free, 0..) |list, class| for (list.items) |store| {
        if (store.seq > s.done and store.seq < s.seq) pending.append(gpa, .{ store.seq, classBytes(class) }) catch return error.OutOfMemory;
    };
    return passFreeing(pending.items, s.stats.pool_bytes, bytes);
}

/// Of free buffers `pending` (the last pass that binds each, its bytes), the oldest pass after whose end releasing
/// them brings `held` to `bytes`; null when all of them do not.
fn passFreeing(pending: [][2]u64, held: u64, bytes: u64) ?u64 {
    std.mem.sort([2]u64, pending, {}, struct {
        fn lt(_: void, a: [2]u64, b: [2]u64) bool {
            return a[0] < b[0];
        }
    }.lt);
    var left = held;
    for (pending) |p| {
        left -|= p[1];
        if (left <= bytes) return p[0];
    }
    return null;
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

/// Past a load's seal its narrow phases run on the arena it planned (docs/bill_redesign.md, part 9): an ask of the
/// pool outside a wide phase is one the plan missed. `.refuse` fails it by name; `.census` counts it and serves it, a
/// diagnostic that lists what a plan still leaves on the pool.
pub const PoolGate = enum { open, census, refuse };

pub fn closePool(gate: PoolGate) Error!void {
    (try st()).pool_gate = gate;
}

/// The gate in force; `.open` before the engine begins, when there is no pool to gate.
pub fn poolGate() PoolGate {
    return if (state) |s| s.pool_gate else .open;
}

/// `closePool` from a model's turn (`route.turnToModel`): a closed gate begins the engine, so no ask made after passes
/// it; an engine with no device has no pool to gate.
pub fn gatePool(gate: PoolGate) void {
    if (state == null and gate == .open) return;
    const s = st() catch return;
    s.pool_gate = gate;
}

/// A phase that asks the pool past the seal by design (a prefill, an encode, a capture or restore until each is
/// planned): its asks pass the gate until `end`. Before the engine begins there is no gate to pass.
pub const Wide = struct {
    counted: bool,

    pub fn end(w: Wide) void {
        if (w.counted) state.?.wide -= 1;
    }
};

pub fn widePhase() Wide {
    const s = &(state orelse return .{ .counted = false });
    s.wide += 1;
    return .{ .counted = true };
}

fn gateAsk(len: usize) Error!void {
    const s = &(state orelse return);
    if (s.pool_gate == .open or s.wide > 0) return;
    s.stats.closed_asks += 1;
    s.stats.closed_ask_bytes += len;
    if (s.pool_gate == .refuse) return error.PoolClosed;
}

/// The pool's cap in force: the in-use bytes when the budget went in plus the budget; null with none.
pub fn poolLimit() ?u64 {
    return if (state) |s| s.pool_limit else null;
}

/// Releases every free buffer no pass still binds; the budget stays as it was.
pub fn trim() void {
    const s = &(state orelse return);
    takeBack();
    const limit = s.pool_limit;
    _ = evict(s, 0);
    s.pool_limit = limit;
}

/// From here Metal holds at most `bytes` for the pool beyond the buffers of the tensors in use, free buffers and the
/// size rounding of what is made from here included: a forward's cap, installed as it begins. Free buffers past it
/// go now. The base is what those tensors hold, not what they ask: a long request's state rounds up by hundreds of MB,
/// which no forward's bill names.
pub fn budget(bytes: u64) Error!void {
    const s = &(state orelse return);
    takeBack();
    s.pool_limit = (s.stats.pool_bytes - s.stats.pool_free) +| bytes;
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
    if (recordingOn()) noteDeath(store);
    switch (store.origin) {
        .dry => |scratch| {
            if (scratch) |b| freeDryScratch(b);
            gpa.destroy(store);
        },
        .pool => |class| {
            if (peakTraceOn()) _ = peak_live.remove(@intFromPtr(store));
            s.stats.pool_asked -= store.len;
            if (s.free[class].append(gpa, store)) |_| {
                s.stats.pool_free += classBytes(class);
                if (store.seq == s.seq and s.pass != null) s.pass_freed += classBytes(class);
                if (free_cap) |cap| if (s.stats.pool_free > cap) {
                    _ = evict(s, s.stats.pool_bytes - (s.stats.pool_free - cap));
                };
            } else |_| {
                // No room to keep it: it goes once the passes that bind it have run.
                if (store.seq > s.done) waitFor(s, store.seq) catch {};
                store.buffer.release();
                s.stats.pool_bytes -= classBytes(class);
                gpa.destroy(store);
            }
            publishPool(s);
        },
        .held => {},
        .arena => |slot| dropArenaStore(store, slot),
        .wide => |w| dropWideStore(store, w.at, w.len, w.host),
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
        .pool, .held, .dry, .arena, .wide => unreachable,
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
    // A lent store dies on MLX's thread, at a time no recording can order.
    if (plan.session.mode != .off) return refusing("a tensor lent to MLX inside a recording", error.TensorUnsupported);
    if (t.store.escaped) return refusing("an arena store its execution left alive, lent to MLX", error.PlanEscape);
    if (t.store.origin == .arena or t.store.origin == .wide) return refusing("an arena store lent to MLX: its bytes are an execution's", error.TensorUnsupported);
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

/// `fresh` for a launch's output (`kernel.launchWith`), as against its scratch: the tensor `into` named when one is
/// armed, else a new ask. An elementwise op asks `fresh`: it is as often an input's copy or a cast inside another op.
/// The ask an `into` takes is state-owned: the packer places none, so its levels are the owner store's.
pub fn freshOut(dtype: Dtype, shape: []const c_int) Error!*Tensor {
    const dst = into_next orelse return freshFor(dtype, shape, false);
    into_next = null;
    if (dst.dtype != dtype or !std.mem.eql(c_int, dst.dims(), shape) or !dst.rowContiguous() or dst.pending != null or dst.offset != 0)
        return refusing("an op's output unlike the tensor `into` named for it", error.TensorShape);
    const t = try share(dst);
    errdefer release(t);
    if (recordingOn()) _ = plan.session.recorder.?.ask(@max(t.nbytes(), 1), false, true) catch return error.OutOfMemory;
    if (plan.session.mode == .replay) _ = plan.session.binder.?.ask(@max(t.nbytes(), 1), .external) catch |e| return planError(e);
    into_taken = dst.store;
    return t;
}

var into_next: ?*Tensor = null;
var into_taken: ?*const Store = null;

/// The tensor the next op's output is (`plan.into`): a state owner's, which the op writes and the forward holds nothing
/// of after. `taken` with the op's result checks the op wrote there; `end` disarms it, on every path.
pub fn into(dst: *Tensor) Error!Into {
    if (into_next != null) return refusing("an `into` while another is armed", error.TensorUnsupported);
    into_next = dst;
    into_taken = null;
    return .{};
}

/// The store an armed `into` names: a launch that reads it is refused (`kernel.launchWith`).
pub fn intoArmed() ?*const Store {
    return if (into_next) |t| t.store else null;
}

pub const Into = struct {
    /// `result` must be the named tensor's bytes, its output asked through `freshOut`; an op whose output still comes
    /// from `fresh` wrote somewhere else, refused by name.
    pub fn taken(_: Into, result: *const Tensor) Error!void {
        if (into_next != null) return refusing("an `into` that no op's output took", error.TensorUnsupported);
        if (into_taken != result.store) return refusing("an op's result that is not the tensor `into` named", error.TensorUnsupported);
    }

    pub fn end(_: Into) void {
        into_next = null;
        into_taken = null;
    }
};

fn freshFor(dtype: Dtype, shape: []const c_int, for_host: bool) Error!*Tensor {
    const t = try box();
    errdefer unbox(t);
    try setShape(t, shape);
    t.dtype = dtype;
    t.offset = 0;
    const len = @max(t.nbytes(), 1);
    if (plan.session.mode == .replay) {
        t.store = try arenaStore(len, for_host);
    } else if (plan.session.mode == .wide) {
        t.store = try wideStore(len, for_host);
    } else if (dryOn()) {
        t.store = try dryStore(len, for_host);
    } else {
        try gateAsk(len);
        t.store = try poolStore(len, for_host);
        const s = &state.?;
        s.stats.asks += 1;
        s.stats.ask_bytes += len;
    }
    t.store.plan_gen = 0;
    t.store.wrote_gen = 0;
    t.store.dry_written = false;
    if (recordingOn()) {
        errdefer dropStore(t.store);
        t.store.plan_ref = plan.session.recorder.?.ask(len, for_host, false) catch return error.OutOfMemory;
        t.store.plan_gen = recording.gen;
    }
    return t;
}

// ── Recording (docs/bill_redesign.md#recorder) ──

/// The engine's side of a recording: its generation, which tells its asks from an earlier recording's, and the levels
/// of the stores its dispatches bound over the whole execution (D9): `dispatchLevel`'s rule on maps no pass cut clears,
/// a store keyed by its ask, or by its address when it is from before the recording, until it dies.
const Recording = struct {
    gen: u32 = 0,
    lv_write: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    lv_read: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    lv_whole: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    /// A level map could not grow: the recording's levels are short of some dispatches.
    failed: bool = false,
    /// Host reads of bytes the execution's dispatches wrote, and those of them a classified site made (`hostResult`).
    readbacks: u32 = 0,
    results: u32 = 0,
    fill: Fill = .zeros,
    /// The made-up bytes of a dry recording's classified readbacks, freed at its end.
    made_up: std.ArrayList([]u8) = .empty,

    fn deinit(r: *Recording) void {
        for (r.made_up.items) |b| gpa.free(b);
        r.made_up.deinit(gpa);
        r.lv_write.deinit(gpa);
        r.lv_read.deinit(gpa);
        r.lv_whole.deinit(gpa);
    }
};
var recording: Recording = .{};
var recording_gens: u32 = 0;

fn recordingOn() bool {
    return plan.session.mode == .real or plan.session.mode == .dry;
}

/// Dry mode (`plan.Mode.dry`): every host decision runs and no GPU work does.
pub fn dryOn() bool {
    return plan.session.mode == .dry;
}

/// A dry recording's pipeline: none is looked up or made; a group may hold Metal's most threads.
pub const dry_pipeline: metal.Pipeline = .{ .raw = null, .max_threads = 1024 };
/// A dry pass's dispatches go nowhere: each is dropped as it is made.
var dry_rec: metal.Recorder = .{};
/// The address of a dry ask with no bytes, which nothing may read.
const dry_ptr: [*]u8 = @ptrFromInt(page_bytes);

/// A dry host ask this large takes whole pages, which the system hands out zeroed and backs only as they are touched:
/// a state reserved for a long context is written by no host in a dry recording, and zeroing it each probe was most of
/// a plan build's time.
const dry_paged_min: usize = 64 << 10;

fn dryScratch(len: usize) Error![]u8 {
    if (len >= dry_paged_min) return std.heap.page_allocator.alloc(u8, len) catch error.OutOfMemory;
    const b = gpa.alloc(u8, len) catch return error.OutOfMemory;
    @memset(b, 0);
    return b;
}

fn freeDryScratch(b: []u8) void {
    if (b.len >= dry_paged_min) std.heap.page_allocator.free(b) else gpa.free(b);
}

/// Whether `t`'s bytes are a dry ask's, zeros until a host writes them (`dryStore`).
pub fn dryFresh(t: *const Tensor) bool {
    return t.store.origin == .dry;
}

fn dryStore(len: usize, for_host: bool) Error!*Store {
    const store = gpa.create(Store) catch return error.OutOfMemory;
    errdefer gpa.destroy(store);
    // Zeros until its filler writes it, so a read before the fill gives the same bytes every run.
    const scratch: ?[]u8 = if (for_host) try dryScratch(len) else null;
    store.* = .{ .buffer = .{ .raw = null }, .at = 0, .ptr = if (scratch) |b| b.ptr else dry_ptr, .len = len, .origin = .{ .dry = scratch } };
    return store;
}

/// The return addresses of the last dry host read refused (`DryReadback`): the site a classification is owed.
pub var refused_readback: [8]usize = @splat(0);

/// `t`'s bytes for a host read (`route.data`, `route.item`). In a recording, a read of bytes the execution's own
/// dispatches wrote is a readback (D10): a real one reads them, a dry one has none and is refused by name; a read of
/// bytes the host filled, or of state from before the recording, is a plain read.
pub fn readBytes(t: *const Tensor) Error![]u8 {
    if (!recordingOn()) return hostRead(t);
    if (t.store.wrote_gen == recording.gen) {
        recording.readbacks += 1;
        if (!dryOn()) return hostRead(t);
        refused_readback = @splat(0);
        _ = std.debug.captureCurrentStackTrace(.{}, &refused_readback);
        return refusing("a host read of bytes the dry execution's GPU work writes", error.DryReadback);
    }
    if (!dryOn()) return hostRead(t);
    if (!t.rowContiguous()) return error.NotRowContiguous;
    if (t.store.origin == .dry and t.store.origin.dry == null) return refusing("a host read of a dry tensor no host filled", error.DryReadback);
    return (t.store.ptr + t.offset)[0..t.nbytes()];
}

/// Readbacks the running recording has made (D10), and those a classified site made.
pub fn recordingReadbacks() struct { all: u32, results: u32 } {
    return .{ .all = recording.readbacks, .results = recording.results };
}

/// A readback that steers no shape (`plan.hostResult`): `t`'s bytes, and in a dry recording made-up ones under its
/// fill, zeros or the words 1, 2, 3, ... (each taken mod `below` when the site's values index something), so a dry
/// recording under each fill holds the classification: two traces that differ prove the bytes steer a shape.
pub fn hostResult(t: *const Tensor, below: ?u32) Error![]u8 {
    if (!dryOn() or t.store.wrote_gen != recording.gen) {
        if (recordingOn() and t.store.wrote_gen == recording.gen) recording.results += 1;
        return readBytes(t);
    }
    if (!t.rowContiguous()) return error.NotRowContiguous;
    recording.readbacks += 1;
    recording.results += 1;
    const b = gpa.alloc(u8, t.nbytes()) catch return error.OutOfMemory;
    recording.made_up.append(gpa, b) catch {
        gpa.free(b);
        return error.OutOfMemory;
    };
    @memset(b, 0);
    if (recording.fill == .ramp) for (std.mem.bytesAsSlice(u32, b[0 .. b.len / 4 * 4]), 0..) |*w, i| {
        const v: u32 = @intCast((i + 1) & 0xffff_ffff);
        w.* = if (below) |m| v % m else v;
    };
    return b;
}

fn recKey(store: *const Store) u64 {
    return if (store.plan_gen == recording.gen) (@as(u64, 1) << 63) | (@as(u64, store.plan_ref.instance) << 32) | store.plan_ref.ordinal else @intFromPtr(store);
}

/// The store's last reference went: its ask died, or a store from before the recording leaves the maps, so that its
/// address cannot lend a later store its levels.
fn noteDeath(store: *const Store) void {
    if (store.plan_gen == recording.gen) return plan.session.recorder.?.free(store.plan_ref);
    const key: u64 = @intFromPtr(store);
    _ = recording.lv_write.remove(key);
    _ = recording.lv_read.remove(key);
    _ = recording.lv_whole.remove(key);
}

/// The dispatch whose stores were just bound, at its level over the execution, given to the recorder for each ask it
/// binds.
fn noteDispatch(s: *State) void {
    noteLevels(s) catch {
        recording.failed = true;
    };
}

fn noteLevels(s: *State) !void {
    const r = &recording;
    var lev: u32 = 0;
    var it = s.next_in.keyIterator();
    while (it.next()) |k| lev = @max(lev, r.lv_write.get(recKey(k.*)) orelse 0);
    it = s.next_out.keyIterator();
    while (it.next()) |k| {
        const key = recKey(k.*);
        const after = if (s.next_in.contains(k.*)) r.lv_write.get(key) orelse 0 else r.lv_whole.get(key) orelse 0;
        lev = @max(lev, @max(after, r.lv_read.get(key) orelse 0));
    }
    lev += 1;
    it = s.next_in.keyIterator();
    while (it.next()) |k| {
        touchAsk(k.*, lev);
        if (s.next_out.contains(k.*)) continue;
        const g = try r.lv_read.getOrPut(gpa, recKey(k.*));
        g.value_ptr.* = if (g.found_existing) @max(g.value_ptr.*, lev) else lev;
    }
    it = s.next_out.keyIterator();
    while (it.next()) |k| {
        k.*.wrote_gen = r.gen;
        const key = recKey(k.*);
        const g = try r.lv_write.getOrPut(gpa, key);
        if (s.next_in.contains(k.*)) {
            g.value_ptr.* = lev;
            try r.lv_whole.put(gpa, key, lev);
            _ = r.lv_read.remove(key);
        } else {
            g.value_ptr.* = if (g.found_existing) @max(g.value_ptr.*, lev) else lev;
            touchAsk(k.*, lev);
        }
    }
}

fn touchAsk(store: *const Store, lev: u32) void {
    if (store.plan_gen == recording.gen) plan.session.recorder.?.touch(store.plan_ref, lev);
}

/// The made-up bytes a dry recording's classified readbacks read (`hostResult`): zeros, or a ramp of distinct words.
pub const Fill = enum { zeros, ramp };

/// How a recording runs: real, on the pool and the GPU (`plan.Mode.real`), or dry, no GPU work, its choices the
/// driver's and its classified readbacks under `fill` (`plan.Mode.dry`).
pub const How = union(enum) {
    real,
    dry: struct { driver: *plan.Driver, fill: Fill = .zeros },
};

/// From here to `endRecording` every ask, death and dispatch is `recorder`'s.
pub fn beginRecording(how: How, recorder: *plan.Recorder) Error!void {
    if (plan.session.mode != .off) return refusing("a recording inside an execution", error.TensorUnsupported);
    if (into_next != null) return refusing("a recording begun with an `into` armed", error.TensorUnsupported);
    const s = try st();
    recording_gens += 1;
    recording = .{ .gen = recording_gens };
    switch (how) {
        .real => plan.session = .{ .mode = .real, .recorder = recorder },
        .dry => |d| {
            // A dry recording starts with nothing of a real pass open or in flight.
            try syncState(s);
            recording.fill = d.fill;
            plan.session = .{ .mode = .dry, .recorder = recorder, .driver = d.driver };
        },
    }
}

/// The recording's end; `OutOfMemory` when a level map could not grow, so its levels are short of some dispatches.
pub fn endRecording() Error!void {
    if (dryOn()) {
        clearHazards(&state.?);
        dry_rec.clear();
    }
    plan.session = .{};
    into_next = null;
    into_taken = null;
    defer {
        recording.deinit();
        recording = .{};
    }
    if (recording.failed) return error.OutOfMemory;
}

// ── The arena (docs/bill_redesign.md#arena) ──

/// The host ring's replicas: execution `k` fills replica `k mod ring_replicas`, so its host writes wait only for
/// execution `k - ring_replicas` to have run.
pub const ring_replicas = 2;

/// The engine's one arena, `[host ring x replicas | scratch]`, made at a load's end and shared by every planned model:
/// executions never overlap. An execution's asks are places in it that its binder gives.
const Arena = struct {
    buffer: metal.Buffer,
    base: [*]u8,
    layout: plan.binder.Layout,
    binder: plan.Binder,
    /// Per replica of the host ring, the last pass of the execution that filled it last.
    ring_seq: [ring_replicas]u64 = @splat(0),
    /// This execution's host asks have waited for their replica.
    ring_ready: bool = false,
    /// The arena stores alive now, each at its `live_at`: one alive when its execution ends escaped.
    live: std.ArrayList(*Store) = .empty,
    /// Escaped stores not yet dropped: the arena may not go under them.
    escaped: u32 = 0,
    spare: std.ArrayList(*Store) = .empty,
    /// The planned loads that hold it (`arenaFor`): it goes with the last one's `arenaRelease`.
    holders: u32 = 0,
};
var arena: ?Arena = null;
var arena_total = std.atomic.Value(u64).init(0);

/// The arena's bytes, readable off the inference thread: memory a load holds from its end, as it holds its weights.
pub fn arenaTotal() u64 {
    return arena_total.load(.monotonic);
}

/// The arena's layout; null before the first planned load.
pub fn arenaLayout() ?plan.binder.Layout {
    return if (arena) |ar| ar.layout else null;
}

/// Makes the arena `layout` asks for, at a planned load's end, which holds it until its `arenaRelease`; or keeps the one
/// held where it holds each part at least as large: a second model takes the larger of each. Remade only between
/// executions, the new buffer made before the old goes, so a refusal leaves the arena held as it was.
pub fn arenaFor(layout: plan.binder.Layout) Error!void {
    const s = try st();
    if (plan.session.mode != .off) return refusing("an arena made inside a recording or an execution", error.TensorUnsupported);
    if (layout.replicas != ring_replicas) return refusing("an arena of another number of ring replicas than the engine's", error.TensorUnsupported);
    var want = layout;
    var holders: u32 = 1;
    if (arena) |*ar| {
        if (ar.layout.alignment != layout.alignment or ar.layout.keep != layout.keep)
            return refusing("an arena of another alignment or lease than the one held", error.TensorUnsupported);
        if (layout.g <= ar.layout.g and layout.h <= ar.layout.h) {
            ar.holders += 1;
            return;
        }
        try arenaDroppable(ar);
        want.g = @max(want.g, ar.layout.g);
        want.h = @max(want.h, ar.layout.h);
        holders += ar.holders;
    }
    const buffer = try s.dev.allocArena(want.arenaBytes());
    if (arena != null) arenaDrop(s);
    arena = .{ .buffer = buffer, .base = buffer.bytes().ptr, .layout = want, .binder = plan.Binder.init(gpa, want), .holders = holders };
    arena_total.store(buffer.allocatedSize(), .monotonic);
}

/// A planned load lets go of the arena: it goes with the last holder, when none of its stores is alive.
pub fn arenaRelease() Error!void {
    const s = &(state orelse return);
    const ar = &(arena orelse return);
    if (ar.holders > 1) {
        ar.holders -= 1;
        return;
    }
    try arenaDroppable(ar);
    arenaDrop(s);
}

fn arenaDroppable(ar: *const Arena) Error!void {
    if (plan.session.mode == .replay) return refusing("the arena dropped inside an execution", error.TensorUnsupported);
    if (ar.live.items.len > 0 or ar.escaped > 0) return refusing("the arena dropped while a store of it is alive", error.TensorUnsupported);
}

fn arenaDrop(s: *State) void {
    const ar = &arena.?;
    // The passes that bind it run before it goes; a pass that failed still ran.
    syncState(s) catch {};
    ar.buffer.release();
    metal.commitResident();
    for (ar.spare.items) |store| gpa.destroy(store);
    ar.spare.deinit(gpa);
    ar.live.deinit(gpa);
    ar.binder.deinit();
    arena = null;
    arena_total.store(0, .monotonic);
}

/// The engine's mark: the open pass and the highest level of its dispatches so far; with no pass open, the next pass
/// at level 0, which orders nothing (`plan_binder.zig`, "Order").
fn markNow(s: *const State) u64 {
    return (s.seq << 32) | (if (s.pass != null) s.pass_level_hi else 0);
}

fn sessionMark() u64 {
    return markNow(&state.?);
}

/// The mark of an arena store's last dispatches: the next tenant of its bytes goes above them.
fn markOf(store: *const Store) u64 {
    return (store.level_seq << 32) | store.level_hi;
}

fn noteArenaLevel(store: *Store, seq: u64, lev: u32) void {
    if (store.level_seq == seq) {
        store.level_hi = @max(store.level_hi, lev);
    } else {
        store.level_seq = seq;
        store.level_hi = lev;
    }
}

fn planError(e: plan.binder.Error) Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.PlanMiss => refusing("an ask the plan did not foresee (the binder's miss names it)", error.PlanMiss),
        error.PlanNeedsLease => error.PlanNeedsLease,
    };
}

/// An execution of `root` on the arena begins: until `endExecution` (or `abortExecution`) every ask is a place the
/// binder gives from `tables`, and one the plan did not foresee is refused by name (`executionMiss`), never allocated.
pub fn beginExecution(tables: *const plan.pack.Tables, root: plan.FrameId) Error!void {
    if (plan.session.mode != .off) return refusing("an execution begun inside a recording or another execution", error.TensorUnsupported);
    if (into_next != null) return refusing("an execution begun with an `into` armed", error.TensorUnsupported);
    const s = try st();
    const ar = &(arena orelse return refusing("an execution with no arena made", error.TensorUnsupported));
    if (tables.alignment != ar.layout.alignment) return refusing("tables packed at another alignment than the arena's", error.TensorUnsupported);
    ar.binder.begin(tables, root, false, markNow(s)) catch |e| return planError(e);
    ar.ring_ready = false;
    plan.session = .{ .mode = .replay, .binder = &ar.binder, .mark = sessionMark };
    s.stats.executions += 1;
}

/// The execution ended as it should: its root closes. A miss it met, or a store of it still alive, is `PlanMiss`.
pub fn endExecution() Error!void {
    const ar = &arena.?;
    // A frame its code left open: the root cannot close over it.
    if (ar.binder.stack.items.len != 1) {
        ar.binder.refuse(.{ .kind = .structure, .id = ar.binder.innermost() }) catch {};
        abortExecution();
        return refusing("an execution ended with a frame still open", error.PlanMiss);
    }
    // A host-filled ask holds no slot the binder judges: alive past its execution, its replica is refilled under it.
    if (ar.live.items.len > 0) ar.binder.refuse(.{ .kind = .lifetime, .id = ar.binder.innermost() }) catch {};
    ar.binder.end() catch |e| {
        leaveExecution(ar);
        return planError(e);
    };
    leaveExecution(ar);
}

/// The execution failed: its frames go, and its stores still alive are escaped.
pub fn abortExecution() void {
    const ar = &arena.?;
    ar.binder.abort();
    leaveExecution(ar);
}

/// The first miss of the last execution: what was asked against what was recorded, where.
pub fn executionMiss() ?plan.binder.Miss {
    return if (arena) |ar| ar.binder.miss else null;
}

fn leaveExecution(ar: *Arena) void {
    const s = &state.?;
    if (comptime builtin.is_test) arena_check.ended(s, ar);
    ar.ring_seq[ar.binder.replica()] = if (s.pass != null) s.seq else s.seq - 1;
    for (ar.live.items) |store| store.escaped = true;
    ar.escaped += @intCast(ar.live.items.len);
    ar.live.clearRetainingCapacity();
    plan.session = .{};
    into_next = null;
    into_taken = null;
}

// ── The wide region (D97, docs/bill_redesign.md): a prefill chunk's or an encode's execution ──

/// A wide execution's placement in the arena: its scratch from the scratch's start by first fit, its host asks one after
/// another in the host ring's replica. A place reused within the open pass goes above every dispatch that bound its
/// bytes before, as an arena place does; the execution's first places go above everything the pass encoded before it.
const Region = struct {
    fit: plan.region.Fit = .{},
    host_top: u64 = 0,
    begin: u64 = 0,
    replica: u32 = 0,
    ring_ready: bool = false,
    /// The passes `beginWidePadded` pads to; the one followed, its next scratch ask, the choices met, and a choice no
    /// pass took there, which refuses the next ask.
    pad: ?[]const PadPass = null,
    pass: usize = 0,
    cursor: usize = 0,
    met: usize = 0,
    missed: bool = false,
};

/// One pass of a dry recording, as a padded wide execution follows it: each scratch ask's place in order
/// (`plan.region.padOf`), and the choices it took (`plan.region.choicesAgree` says the passes may be mixed).
pub const PadPass = struct { sizes: []const u64, choices: []const plan.TraceChoice = &.{} };
var region: Region = .{};

pub const WideRefusal = struct { asked: u64, room: u64, host: bool };

/// The first ask the open or last wide execution was refused (`WideFull`): what it wanted of the region a load holds,
/// and the room there. An op behind MLX's C API reports any failure as its own, so the caller reads it here.
pub var refused_wide: ?WideRefusal = null;

/// A wide execution begins (D97): until `endWide` (or `abortWide`) every ask is a place in the arena's region, and one
/// past it is refused by name (`WideFull`), never asked of the pool or of Metal.
pub fn beginWide() Error!void {
    if (plan.session.mode != .off) return refusing("a wide execution begun inside a recording or another execution", error.TensorUnsupported);
    if (into_next != null) return refusing("a wide execution begun with an `into` armed", error.TensorUnsupported);
    const s = try st();
    const ar = &(arena orelse return refusing("a wide execution with no arena made", error.TensorUnsupported));
    region.fit.gaps.clearRetainingCapacity();
    region.fit.top = 0;
    region.fit.top_mark = 0;
    region.fit.high = 0;
    region.host_top = 0;
    region.begin = markNow(s);
    // The host ring's replicas turn with every execution, planned or wide: each waits only for its replica's last.
    region.replica = ar.binder.replica();
    ar.binder.executions += 1;
    region.ring_ready = false;
    refused_wide = null;
    plan.session = .{ .mode = .wide, .mark = sessionMark };
    s.stats.wide_executions += 1;
}

/// `beginWide` with each scratch ask placed at the size its pass recorded for it, in order: first fit then places them
/// as the recording did, so the execution tops out where its recording does whatever its own sizes. At each choice it
/// follows a pass that took the same one there. An ask past its recorded size or past the sequence, or one after a
/// choice no pass took there, is refused by name (`PlanMiss`). `passes` outlive the execution.
pub fn beginWidePadded(passes: []const PadPass) Error!void {
    try beginWide();
    region.pad = passes;
    region.pass = 0;
    region.cursor = 0;
    region.met = 0;
    region.missed = false;
    plan.session.on_choice = wideChoice;
}

fn wideChoice(site: plan.Site, value: u32) void {
    const passes = region.pad orelse return;
    const j = region.met;
    region.met += 1;
    const cur = passes[region.pass].choices;
    if (j >= cur.len or cur[j].site != site or cur[j].at != region.cursor) {
        region.missed = true;
        return;
    }
    if (cur[j].value == value) return;
    for (passes, 0..) |ps, q| if (j < ps.choices.len and ps.choices[j].site == site and ps.choices[j].value == value) {
        region.pass = q;
        region.cursor = ps.choices[j].at;
        return;
    };
    region.missed = true;
}

/// The wide execution ended as it should: a store of it still alive is refused by name (`PlanEscape`), and escapes.
pub fn endWide() Error!void {
    const ar = &arena.?;
    const alive = ar.live.items.len;
    leaveWide(ar);
    if (alive > 0) return refusing("a wide execution ended with a store of it still alive", error.PlanEscape);
}

/// The wide execution failed: its stores still alive escape.
pub fn abortWide() void {
    leaveWide(&arena.?);
}

fn leaveWide(ar: *Arena) void {
    const s = &state.?;
    const high = region.fit.high;
    s.stats.wide_high = @max(s.stats.wide_high, high);
    s.stats.wide_last_high = high;
    ar.ring_seq[region.replica] = if (s.pass != null) s.seq else s.seq - 1;
    for (ar.live.items) |store| store.escaped = true;
    ar.escaped += @intCast(ar.live.items.len);
    ar.live.clearRetainingCapacity();
    plan.session = .{};
    into_next = null;
    into_taken = null;
    region.pad = null;
}

fn wideStore(len: usize, for_host: bool) Error!*Store {
    const s = &state.?;
    const ar = &arena.?;
    var place = std.mem.alignForward(u64, @max(len, 1), ar.layout.alignment);
    var at: u64 = undefined;
    var after: u64 = region.begin;
    if (!for_host) if (region.pad) |passes| {
        const pad = passes[region.pass].sizes;
        if (region.missed) return refusing("a wide execution's scratch ask after a choice no pass of its recording took there", error.PlanMiss);
        if (region.cursor == pad.len) return refusing("a wide execution's scratch ask past the sequence its recording made", error.PlanMiss);
        if (place > pad[region.cursor]) return refusing("a wide execution's scratch ask past the size its recording made", error.PlanMiss);
        place = pad[region.cursor];
        region.cursor += 1;
    };
    if (for_host) {
        if (!region.ring_ready) {
            const seq = ar.ring_seq[region.replica];
            if (seq > s.done) try waitFor(s, seq);
            region.ring_ready = true;
        }
        if (region.host_top + place > ar.layout.h) {
            if (refused_wide == null) refused_wide = .{ .asked = region.host_top + place, .room = ar.layout.h, .host = true };
            return refusing("a wide execution's host asks past the region the load holds", error.WideFull);
        }
        at = region.replica * ar.layout.h + region.host_top;
        region.host_top += place;
    } else {
        const got = region.fit.take(place);
        if (got.at + place > ar.layout.g) {
            region.fit.give(gpa, got.at, got.at + place, got.mark) catch {};
            if (refused_wide == null) refused_wide = .{ .asked = got.at + place, .room = ar.layout.g, .host = false };
            return refusing("a wide execution's scratch past the region the load holds", error.WideFull);
        }
        at = ar.layout.h * ar.layout.replicas + got.at;
        after = @max(after, got.mark);
    }
    errdefer if (!for_host) region.fit.give(gpa, at - ar.layout.h * ar.layout.replicas, at - ar.layout.h * ar.layout.replicas + place, after) catch {};
    try ar.live.ensureUnusedCapacity(gpa, 1);
    const store = ar.spare.pop() orelse (gpa.create(Store) catch return error.OutOfMemory);
    store.* = .{ .buffer = ar.buffer, .at = at, .ptr = ar.base + at, .len = len, .origin = .{ .wide = .{ .at = at, .len = place, .host = for_host } } };
    const floor: u32 = @truncate(after);
    if (after >> 32 == s.seq and floor > 0) {
        store.lev_seq = s.seq;
        store.lev_read = floor;
        store.lev_write = floor;
        store.lev_whole = floor;
        s.stats.arena_floors += 1;
    }
    store.live_at = @intCast(ar.live.items.len);
    ar.live.appendAssumeCapacity(store);
    s.stats.wide_asks += 1;
    return store;
}

fn dropWideStore(store: *Store, at: u64, len: u64, host: bool) void {
    const ar = &arena.?;
    if (store.escaped) {
        ar.escaped -= 1;
    } else {
        if (!host) region.fit.give(gpa, at - ar.layout.h * ar.layout.replicas, at - ar.layout.h * ar.layout.replicas + len, markOf(store)) catch {};
        const last = ar.live.pop().?;
        if (last != store) {
            ar.live.items[store.live_at] = last;
            last.live_at = store.live_at;
        }
    }
    ar.spare.append(gpa, store) catch gpa.destroy(store);
}

/// A place in the arena for the next ask of the execution: scratch at its recorded offset, or a host-filled ask from
/// this execution's replica of the host ring. Its first dispatch in the open pass goes above every dispatch that held
/// its bytes before (`Got.after`): the level maps start it at that level.
fn arenaStore(len: usize, for_host: bool) Error!*Store {
    const s = &state.?;
    const ar = &arena.?;
    const got = ar.binder.ask(len, if (for_host) .host else .scratch) catch |e| return planError(e);
    if (for_host and !ar.ring_ready) {
        const seq = ar.ring_seq[ar.binder.replica()];
        if (seq > s.done) try waitFor(s, seq);
        ar.ring_ready = true;
    }
    // A place that fails from here is no store's: the binder gets it back, dead, and no record of it stays alive.
    errdefer ar.binder.release(got.slot, 0);
    try ar.live.ensureUnusedCapacity(gpa, 1);
    const store = ar.spare.pop() orelse (gpa.create(Store) catch return error.OutOfMemory);
    errdefer ar.spare.append(gpa, store) catch gpa.destroy(store);
    store.* = .{ .buffer = ar.buffer, .at = got.offset, .ptr = ar.base + got.offset, .len = len, .origin = .{ .arena = got.slot } };
    const floor: u32 = @truncate(got.after);
    if (got.after >> 32 == s.seq and floor > 0) {
        store.lev_seq = s.seq;
        store.lev_read = floor;
        store.lev_write = floor;
        store.lev_whole = floor;
        s.stats.arena_floors += 1;
    }
    if (comptime builtin.is_test) if (got.slot.item != plan.trace_mod.none) try arena_check.handOut(s, ar, store, got.room, got.after);
    store.live_at = @intCast(ar.live.items.len);
    ar.live.appendAssumeCapacity(store);
    s.stats.arena_asks += 1;
    return store;
}

/// TEST-ONLY, the arena's `poison`: with `byte` set, a place is handed out over `byte` once every pass before it has run,
/// and its guard (`plan_pack.Options.guard`) holds a canary. A canary found changed when its bytes are next handed out,
/// or when its execution ends, counts in `hits`: a kernel wrote past its output. Compiled only under test.
pub const arena_check = struct {
    pub var byte: ?u8 = null;
    pub var hits: u64 = 0;
    /// Places handed out unpoisoned: a tenant before them is still in the open pass, which an op may be encoding.
    pub var skipped: u64 = 0;
    const canary: u8 = 0xA5;
    /// A guard's bytes in the arena, and its store while that is alive.
    const Guard = struct { lo: u64, hi: u64, store: ?*const Store };
    var guards: std.ArrayList(Guard) = .empty;

    fn handOut(s: *State, ar: *Arena, store: *const Store, place_bytes: u64, after: u64) Error!void {
        const b = byte orelse return;
        const lo = store.at;
        const hi = lo + place_bytes;
        // An op may ask for scratch with its pass open: only passes already handed over are waited for, never that one,
        // so a place whose last tenant ran in the open pass goes unpoisoned. A mark at level 0 names no dispatch at all.
        const ran = after >> 32 < s.seq or @as(u32, @truncate(after)) == 0;
        if (ran and s.flying.items.len > 0) try waitFor(s, s.seq - 1);
        var i: usize = 0;
        while (i < guards.items.len) {
            const g = guards.items[i];
            if (g.store == null and g.lo < hi and lo < g.hi) {
                if (ran) check(ar, g);
                _ = guards.swapRemove(i);
            } else i += 1;
        }
        if (!ran) {
            skipped += 1;
            return;
        }
        @memset(ar.base[lo..][0..store.len], b);
        @memset(ar.base[lo + store.len .. hi], canary);
        guards.append(gpa, .{ .lo = lo + store.len, .hi = hi, .store = store }) catch return error.OutOfMemory;
    }

    fn check(ar: *const Arena, g: Guard) void {
        for (ar.base[g.lo..g.hi]) |c| if (c != canary) {
            hits += 1;
            return;
        };
    }

    fn died(store: *const Store) void {
        for (guards.items) |*g| if (g.store == store) {
            g.store = null;
        };
    }

    fn ended(s: *State, ar: *const Arena) void {
        if (byte == null) return;
        syncState(s) catch {};
        for (guards.items) |g| check(ar, g);
        guards.clearRetainingCapacity();
    }
};

fn dropArenaStore(store: *Store, slot: plan.binder.Slot) void {
    const ar = &arena.?;
    if (comptime builtin.is_test) arena_check.died(store);
    if (store.escaped) {
        ar.escaped -= 1;
    } else {
        if (plan.session.mode == .replay) ar.binder.release(slot, markOf(store));
        const last = ar.live.pop().?;
        if (last != store) {
            ar.live.items[store.live_at] = last;
            last.live_at = store.live_at;
        }
    }
    ar.spare.append(gpa, store) catch gpa.destroy(store);
}

/// `fresh` over a buffer no pass has yet to run over: the host may write it at once.
pub fn freshHost(dtype: Dtype, shape: []const c_int) Error!*Tensor {
    return freshFor(dtype, shape, true);
}

/// MLX's `array(value, dtype)` for a float dtype: the value through float, rounded to the dtype on the host.
pub fn scalar(value: f64, dtype: Dtype) Error!*Tensor {
    const f: f32 = @floatCast(value);
    return switch (dtype) {
        .float32 => fromHost(std.mem.asBytes(&f), dtype, &.{}),
        .float16 => fromHost(std.mem.asBytes(&@as(f16, @floatCast(f))), dtype, &.{}),
        .bfloat16 => blk: {
            const bits: u32 = @bitCast(f);
            const rounded: u16 = if (std.math.isNan(f)) 0x7FC0 else @intCast((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
            break :blk fromHost(std.mem.asBytes(&rounded), dtype, &.{});
        },
        else => refusing("a host scalar of a dtype that is not float", error.TensorDtype),
    };
}

/// Words of the counter page (`counterPage`): the first `counter_words` are counters, the rest a counted launch's
/// scratch.
pub const counter_page_words = 8192;
pub const counter_words = 256;

/// A handle on the engine's counter page, u32 [counter_page_words], zero when Metal makes it and held for the process.
/// Only a counted launch binds it (`kernel.launchCounted`), and each leaves the counters zero when it ends.
pub fn counterPage() Error!*Tensor {
    const s = try st();
    if (s.counters == null) {
        const t = try box();
        errdefer unbox(t);
        try setShape(t, &.{counter_page_words});
        t.dtype = .uint32;
        t.offset = 0;
        const buffer = try s.dev.alloc(counter_page_words * 4);
        errdefer buffer.release();
        const store = gpa.create(Store) catch return error.OutOfMemory;
        store.* = .{ .buffer = buffer, .at = 0, .ptr = buffer.bytes().ptr, .len = counter_page_words * 4, .origin = .held };
        t.store = store;
        s.counters = t;
    }
    return share(s.counters.?);
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

/// `wrapMapped` for a state owner's buffer (a request's `Media`): written in place by an op while it is sole, as the
/// pool's are.
pub fn wrapOwned(ptr: [*]u8, dtype: Dtype, shape: []const c_int, payload: ?*anyopaque, release_fn: MappedRelease) Error!*Tensor {
    const t = try wrapMapped(ptr, dtype, shape, payload, release_fn);
    t.store.origin.mapped.owned = true;
    return t;
}

/// Moves the pool buffer under `t` into the mapping at `ptr` (`len` bytes, at least the store's), which the engine
/// wraps as `wrapMapped` would: every tensor on the store reads the mapping from here on, and the pool takes its buffer
/// back. For a weight a load built with an op, which would otherwise sit in the pool under every forward's cap.
pub fn moveToMapped(t: *Tensor, ptr: [*]u8, len: usize, payload: ?*anyopaque, release_fn: MappedRelease) Error!void {
    const s = try st();
    const store = t.store;
    if (store.origin != .pool) return refusing("a move into a mapping of a tensor not in the pool", error.TensorUnsupported);
    if (t.twin.ctx != null or t.pending != null) return refusing("a move into a mapping of a tensor lent to MLX or with a write pending", error.TensorUnsupported);
    if (len < store.len) return refusing("a move into a mapping smaller than the tensor's store", error.TensorUnsupported);
    try syncState(s);
    @memcpy(ptr[0..store.len], store.ptr[0..store.len]);
    const buffer = try (try metal.device()).wrap(ptr, len);
    const old = gpa.create(Store) catch {
        buffer.release();
        return error.OutOfMemory;
    };
    old.* = store.*;
    old.refs = 1;
    store.* = .{ .buffer = buffer, .at = 0, .ptr = ptr, .len = len, .refs = store.refs, .origin = .{ .mapped = .{ .payload = payload, .release = release_fn } } };
    s.stats.mapped_bytes += len;
    mapped_total.store(s.stats.mapped_bytes, .monotonic);
    dropStore(old);
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
    if (dryOn()) return refusing("an MLX array met by a dry recording: MLX's work is never dry", error.TensorUnsupported);
    if (plan.session.mode == .replay) return refusing("an MLX array met by an execution on the arena: MLX's memory is no part of the plan", error.TensorUnsupported);
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
    // Its box waits for the next tensor, and its store may already be another's.
    if (!t.live) return refusing("a tensor used after its release", error.TensorUnsupported);
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
        .pool, .dry, .arena, .wide => return true,
        .mlx => |kept| {
            var only = false;
            try mlx.check(mlx._mlx_array_is_donatable(&only, kept));
            return only;
        },
        // A wrapped mapping is its maker's (a weight), never written in place, but a state owner's own.
        .mapped => |mp| return mp.owned,
        .held => return false,
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

/// Bytes the open pass may free yet still bind before it is handed over: the pool cannot hand them out again until
/// then, so a pass grown by the kicks' doubling held every buffer it freed. MLX commits a command buffer by bytes too.
pub const pass_freed_cap: u64 = 64 * 1024 * 1024;

/// An execution's passes on the arena (docs/bill_redesign.md#arena): at most `dispatches` in one, since the pool's
/// `pass_freed_cap` no longer cuts them and a pass the kicks' doubling grew would hold every reuse of the arena under a
/// floor; and at most `passes` committed and unfinished, so Metal's memory for command buffers is bounded. Tests set them.
pub var execution_pass_bounds: struct { dispatches: u32 = 512, passes: u32 = 16 } = .{};

/// The most free bytes the pool keeps, MLX's cache limit beside it (`gpu_mem`): past it a freed buffer's size class is
/// released at once, as MLX's cache releases past its own; null keeps every free buffer.
pub var free_cap: ?u64 = null;

/// The open pass, begun now if none is. Work encoded while the GPU has none of ours to run is handed over early,
/// in steps that double, so the GPU starts on a draft or a layer while the host still builds the rest of it. The
/// steps after the first do not wait for the GPU to run dry: a pass handed over behind one that has not started
/// starts as that one ends, where one handed to an idle GPU waits Metal's start latency again. The steps start over
/// at the host's next wait. Never between a dispatch's binds and the dispatch.
pub fn pass() Error!metal.Pass {
    const s = try st();
    if (dryOn()) return .{ .pool = null, .cmd = null, .encoder = null, .rec = &dry_rec };
    if (s.pass) |open| {
        if (kernelProfOn() and s.pass_dispatches > 0) {
            s.pass = null;
            try commitPass(s, open);
        } else if (s.pass_dispatches >= s.kick_after and s.next_in.count() == 0 and s.next_out.count() == 0 and (s.burst or gpuIdle(s))) {
            s.pass = null;
            try commitPass(s, open);
            s.kick_after *|= 2;
            s.burst = true;
            s.stats.kicks += 1;
        } else if (s.pass_freed >= pass_freed_cap and s.next_in.count() == 0 and s.next_out.count() == 0) {
            // What it freed goes back to the pool once handed over: a later pass may take it.
            s.pass = null;
            try commitPass(s, open);
            s.stats.freed_kicks += 1;
        } else if (plan.session.mode == .replay and s.pass_dispatches >= execution_pass_bounds.dispatches and s.next_in.count() == 0 and s.next_out.count() == 0) {
            s.pass = null;
            try commitPass(s, open);
            s.stats.bound_kicks += 1;
        }
    }
    if (s.pass == null) {
        s.pass = s.dev.beginConcurrent(&s.rec);
    }
    return s.pass.?;
}

/// Whether every pass handed to the GPU has run.
fn gpuIdle(s: *State) bool {
    return s.flying.items.len == 0 or s.flying.items[s.flying.items.len - 1].cmd.ended();
}

/// Binds the bytes of `t`, which the dispatch being encoded reads, at `index` of the open pass.
pub fn bind(p: metal.Pass, t: *const Tensor, index: usize) void {
    const s = &state.?;
    const g = s.next_in.getOrPut(gpa, t.store) catch @panic("engine: out of memory tracking a pass");
    if (dryOn()) return;
    if (!g.found_existing and kernelProfOn()) s.pass_kernels.read += t.reach();
    bindBuffer(s, p, t, index);
}

fn bindBuffer(s: *State, p: metal.Pass, t: *const Tensor, index: usize) void {
    if (t.store.origin == .dry or t.store.dry_written) @panic("engine: a real dispatch binds a store a dry recording made or wrote: reset the state it ran on");
    if (t.store.escaped) @panic("engine: a dispatch binds an arena store its execution left alive: its bytes are another execution's");
    t.store.seq = s.seq;
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
    s.next_in.put(gpa, t.store, {}) catch @panic("engine: out of memory tracking a pass");
    if (dryOn()) return s.next_out.put(gpa, t.store, {}) catch @panic("engine: out of memory tracking a pass");
    bindBuffer(s, p, t, index);
    bindOutTail(s, t);
}

/// `bindOut` for a write that misses every element the dispatches before it write to `t`: it waits for their reads,
/// not their writes, and is not a read itself, so writes to disjoint rows of one plane run together. A reader of `t`
/// after waits for them all.
pub fn bindOutDisjoint(p: metal.Pass, t: *const Tensor, index: usize) void {
    const s = &state.?;
    if (dryOn()) return s.next_out.put(gpa, t.store, {}) catch @panic("engine: out of memory tracking a pass");
    bindBuffer(s, p, t, index);
    bindOutTail(s, t);
}

fn bindOutTail(s: *State, t: *const Tensor) void {
    const g = s.next_out.getOrPut(gpa, t.store) catch @panic("engine: out of memory tracking a pass");
    if (!g.found_existing and kernelProfOn()) s.pass_kernels.write += t.reach();
    s.pass_writes.put(gpa, t.store, {}) catch @panic("engine: out of memory tracking a pass");
}

fn beforeDispatch() void {
    const s = &state.?;
    if (dryOn()) {
        noteDispatch(s);
        var it = s.next_out.keyIterator();
        while (it.next()) |k| if (k.*.origin != .dry and (s.counters == null or k.* != s.counters.?.store)) {
            k.*.dry_written = true;
        };
        s.next_in.clearRetainingCapacity();
        s.next_out.clearRetainingCapacity();
        dry_rec.clear();
        return;
    }
    s.stats.dispatches += 1;
    s.pass_dispatches += 1;
    if (kernelProfOn()) {
        if (s.pass_kernels.n == 0) {
            const trace = std.debug.captureCurrentStackTrace(.{}, &s.pass_kernels.site);
            s.pass_kernels.site_n = @intCast(trace.return_addresses.len);
        }
        s.pass_kernels.add(pipelineName(metal.bound_pipeline));
    }
    if (recordingOn()) noteDispatch(s);
    s.rec.next_level = 2 * dispatchLevel(s);
    s.next_in.clearRetainingCapacity();
    s.next_out.clearRetainingCapacity();
}

fn levOf(store: *const Store, s: *const State) struct { read: u32, write: u32, whole: u32 } {
    if (store.lev_seq != s.seq) return .{ .read = 0, .write = 0, .whole = 0 };
    return .{ .read = store.lev_read, .write = store.lev_write, .whole = store.lev_whole };
}

fn levSet(store: *Store, s: *const State, lev: u32) void {
    if (store.lev_seq != s.seq) {
        store.lev_seq = s.seq;
        store.lev_read = 0;
        store.lev_write = 0;
        store.lev_whole = 0;
    }
    store.lev_read = @max(store.lev_read, lev);
}

fn levSetWrite(store: *Store, s: *const State, lev: u32, whole: bool) void {
    if (store.lev_seq != s.seq) {
        store.lev_seq = s.seq;
        store.lev_read = 0;
        store.lev_write = 0;
        store.lev_whole = 0;
    }
    store.lev_write = if (whole) lev else @max(store.lev_write, lev);
    if (whole) {
        store.lev_whole = lev;
        store.lev_read = 0;
    }
}

/// The level in the open pass of the dispatch whose stores were just bound (`State.rec`), noted for its stores.
fn dispatchLevel(s: *State) u32 {
    var lev: u32 = 0;
    var it = s.next_in.keyIterator();
    while (it.next()) |k| lev = @max(lev, levOf(k.*, s).write);
    it = s.next_out.keyIterator();
    while (it.next()) |k| {
        const l = levOf(k.*, s);
        const after = if (s.next_in.contains(k.*)) l.write else l.whole;
        lev = @max(lev, @max(after, l.read));
    }
    lev += 1;
    s.pass_level_hi = @max(s.pass_level_hi, lev);
    it = s.next_in.keyIterator();
    while (it.next()) |k| {
        if (k.*.origin == .arena or k.*.origin == .wide) noteArenaLevel(k.*, s.seq, lev);
        if (s.next_out.contains(k.*)) continue;
        levSet(k.*, s, lev);
    }
    it = s.next_out.keyIterator();
    while (it.next()) |k| {
        if (k.*.origin == .arena or k.*.origin == .wide) noteArenaLevel(k.*, s.seq, lev);
        levSetWrite(k.*, s, lev, s.next_in.contains(k.*));
    }
    return lev;
}

/// Records the dispatch whose buffers were just bound, at its level in the pass.
pub fn dispatch(p: metal.Pass, grid: metal.Size, group: metal.Size) void {
    beforeDispatch();
    if (dryOn()) return;
    p.dispatch(grid, group);
    if (kernelDoubled()) {
        doubledLevel();
        p.dispatch(grid, group);
    }
}

/// A doubled dispatch runs after its first run and before the next level.
fn doubledLevel() void {
    const s = &state.?;
    s.rec.next_level += 1;
}

/// `dispatch` with `grid` in threadgroups.
pub fn dispatchGroups(p: metal.Pass, grid: metal.Size, group: metal.Size) void {
    beforeDispatch();
    if (dryOn()) return;
    p.dispatchGroups(grid, group);
    if (kernelDoubled()) {
        doubledLevel();
        p.dispatchGroups(grid, group);
    }
}

/// Tests set this; otherwise `SUSHI_ENGINE_KERNEL_DOUBLE` decides, read when the engine starts.
pub var kernel_double_hook: ?[]const u8 = null;
var double_env: ?[]const u8 = null;

/// The double switch's text: unset, empty, `0` or `off` is off; otherwise a piece of a pipeline name. A bare number
/// is refused, since it would match every name holding that digit rather than switch the doubling on or off.
fn doubleSwitch(raw: ?[*:0]const u8) error{EnvUnparsed}!?[]const u8 {
    const v = std.mem.span(raw orelse return null);
    if (v.len == 0 or std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return null;
    var digits = true;
    for (v) |c| {
        if (!std.ascii.isAlphanumeric(c) and c != '_') return error.EnvUnparsed;
        digits = digits and std.ascii.isDigit(c);
    }
    return if (digits) error.EnvUnparsed else v;
}

/// Whether the dispatch just encoded runs again behind a barrier: its pipeline name holds the switch's text. A
/// forward then pays that kernel twice, so the forward meter's difference is what the kernel costs inside the forward,
/// overlap and all. Its outputs are written twice from the same inputs; a kernel that reads what it writes is off.
fn kernelDoubled() bool {
    const sub = kernel_double_hook orelse double_env orelse return false;
    if (std.mem.indexOf(u8, pipelineName(metal.bound_pipeline), sub) == null) return false;
    state.?.stats.doubled += 1;
    return true;
}

/// Pipelines made since the process began, the time they took, and those made after `sealed`: a model sets it once
/// its load has run every forward a request can, so a later pipeline is a stall inside a request.
pub var pipelines: u64 = 0;
pub var pipeline_ns: u64 = 0;
pub var pipelines_late: u64 = 0;
/// Pipelines a launch took from those a manifest made ahead: no compile, so neither counted above nor late.
pub var pipelines_preloaded: u64 = 0;
pub var sealed: bool = false;
/// Where the inference thread is, for the late line: a caller's own words for the step that may make a pipeline.
pub var late_where: []const u8 = "outside a forward";

/// The pipeline `pipeline`, named `name`, was made since `t0`.
pub fn madePipeline(name: []const u8, t0: u64, pipeline: metal.Pipeline) void {
    const ns = nowNs() - t0;
    nameOf(name, pipeline);
    pipelines += 1;
    pipeline_ns += ns;
    const log = @import("../log.zig");
    if (!sealed) return log.debug("[engine] compiled {s}\n", .{name});
    pipelines_late += 1;
    if (comptime builtin.is_test) late_names.add(name);
    log.warn("[engine] compiled inside a request ({d} ms, {s}): {s}\n", .{ ns / std.time.ns_per_ms, late_where, name });
}

/// The first late pipelines' names in a test process, for a test that finds one to say which.
pub const late_names = struct {
    var buf: [16384]u8 = undefined;
    var len: usize = 0;

    fn add(name: []const u8) void {
        const line = name[0..@min(name.len, 120)];
        if (len + line.len + 1 > buf.len) return;
        @memcpy(buf[len..][0..line.len], line);
        buf[len + line.len] = '\n';
        len += line.len + 1;
    }

    pub fn text() []const u8 {
        return buf[0..len];
    }

    pub fn clear() void {
        len = 0;
    }
};

fn nameOf(name: []const u8, pipeline: metal.Pipeline) void {
    if (gpa.dupe(u8, name)) |owned| pipeline_names.put(gpa, @intFromPtr(pipeline.raw), owned) catch gpa.free(owned) else |_| {}
}

/// `pipeline`, which a manifest made ahead, is first used as `name`.
pub fn preloadedPipeline(name: []const u8, pipeline: metal.Pipeline) void {
    nameOf(name, pipeline);
    pipelines_preloaded += 1;
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
    // At most `passes` committed and unfinished once this one is: the open pass is not one of them, so no wait commits it.
    if (plan.session.mode == .replay and s.flying.items.len >= execution_pass_bounds.passes)
        try waitFor(s, s.flying.items[s.flying.items.len - execution_pass_bounds.passes].seq);
    s.stats.barriers += open.replay();
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
    s.stats.longest_pass = @max(s.stats.longest_pass, s.pass_dispatches);
    s.seq += 1;
    s.pass_dispatches = 0;
    s.pass_level_hi = 0;
    s.pass_freed = 0;
    s.stats.submits += 1;
    s.stats.most_flying = @max(s.stats.most_flying, s.flying.items.len);
}

/// The most dispatches one pass held and the most passes in flight at once since the last take; both start again.
pub fn takePassHighs() struct { longest: u64, flying: u64 } {
    const s = &(state orelse return .{ .longest = 0, .flying = 0 });
    defer {
        s.stats.longest_pass = 0;
        s.stats.most_flying = 0;
    }
    return .{ .longest = s.stats.longest_pass, .flying = s.stats.most_flying };
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
    s.pass_level_hi = 0;
    s.pass_freed = 0;
    return error.ReadsFillingSlab;
}

fn clearPassBinds(s: *State) void {
    s.pass_binds.clearRetainingCapacity();
    s.pass_writes.clearRetainingCapacity();
    s.pass_mapped.clearRetainingCapacity();
}

fn clearHazards(s: *State) void {
    s.next_in.clearRetainingCapacity();
    s.next_out.clearRetainingCapacity();
}

/// Hands the open pass to the GPU without waiting for it.
pub fn submit() Error!void {
    const s = try st();
    const open = s.pass orelse return;
    s.pass = null;
    try commitPass(s, open);
    s.stats.handovers += 1;
}

/// Dispatches encoded into the open pass and not yet handed to the GPU.
pub fn openDispatches() u32 {
    const s = &(state orelse return 0);
    return if (s.pass != null) s.pass_dispatches else 0;
}

/// Submits the open pass and waits until the GPU has run everything submitted.
pub fn sync() Error!void {
    return syncState(try st());
}

/// The forward evaluated an array: a recording marks the wait, and a wide execution waits for every pass submitted
/// and gives back its host ring's bytes past the highest host ask still alive (`plan_region`).
pub fn evalPoint() Error!void {
    if (recordingOn()) plan.session.recorder.?.wait() catch return error.OutOfMemory;
    if (plan.session.mode != .wide) return;
    try syncState(try st());
    const ar = &arena.?;
    const base = region.replica * ar.layout.h;
    var top: u64 = 0;
    for (ar.live.items) |store| switch (store.origin) {
        .wide => |w| if (w.host) {
            top = @max(top, w.at - base + w.len);
        },
        else => {},
    };
    region.host_top = top;
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
        // An op's time starts where the buffer before it ended: a buffer begun early waits there for its fence.
        const from = @max(span[0], s.gpu_last_end);
        noteGpuSpan(s, span, done.phase);
        if (done.kernels.n > 0 and span[0] > 0 and span[1] >= span[0]) noteKernels(done.kernels, if (span[1] > from) @intFromFloat((span[1] - from) * 1e9) else 0);
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
    // A pass that failed may have stopped a counted launch part way: its counters start again from zero.
    if (failed != null) if (s.counters) |c| @memset(c.store.ptr[0 .. counter_words * 4], 0);
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

/// `t`'s bytes for the host to read once the pass that last wrote them has run: a queued pass that only reads them
/// may still be running. Row-contiguous tensors only. A host write takes `hostBytes`.
pub fn hostRead(t: *const Tensor) Error![]u8 {
    if (!t.rowContiguous()) return error.NotRowContiguous;
    // In a dry recording this is a wait (`mlx_array_eval`) and reads nothing.
    if (!dryOn() and (t.store.origin == .dry or t.store.dry_written)) return refusing("a host read of a store a dry recording made or wrote: reset the state it ran on", error.DryStore);
    if (t.store.escaped) return refusing("a host read of an arena store its execution left alive", error.PlanEscape);
    const s = try st();
    if (t.store.seq > s.done) {
        if (s.pass != null and s.pass_writes.contains(t.store)) {
            try waitFor(s, s.seq);
        } else if (s.writers.get(t.store)) |w| {
            if (w.seq > s.done) try waitFor(s, w.seq);
        }
    }
    return (t.store.ptr + t.offset)[0..t.nbytes()];
}

/// `t`'s bytes on the host once every pass that binds them has run, for the host to write. Row-contiguous tensors only.
pub fn hostBytes(t: *const Tensor) Error![]u8 {
    if (!t.rowContiguous()) return error.NotRowContiguous;
    if (dryOn()) {
        if (t.store.origin != .dry or t.store.origin.dry == null) return refusing("a host write in a dry recording into a tensor that is no host ask", error.DryHostWrite);
        return (t.store.ptr + t.offset)[0..t.nbytes()];
    }
    // The host writes only what it filled: arena scratch is reused under the GPU's own order, which no host write follows.
    if (t.store.origin == .wide and !t.store.origin.wide.host) return refusing("a host write into a wide execution's scratch", error.TensorUnsupported);
    if (t.store.origin == .arena and t.store.origin.arena.item != plan.trace_mod.none) return refusing("a host write into the arena's scratch", error.TensorUnsupported);
    if (t.store.escaped) return refusing("a host write into an arena store its execution left alive", error.PlanEscape);
    const s = try st();
    if (t.store.seq > s.done) try waitFor(s, t.store.seq);
    return (t.store.ptr + t.offset)[0..t.nbytes()];
}

test "a long open pass holds no more than it uses plus the bytes it may free before it is handed over" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    const ops = @import("ops.zig");
    try sync();
    trim();
    // 8 MB a step, each step's input freed after it: in use stays at two steps while the pass grows.
    const rows = 2048;
    var x = try fresh(.float32, &.{ rows, 1024 });
    @memset(try hostBytes(x), 0);
    const held0 = stats().pool_bytes;
    var step: usize = 0;
    while (step < 400) : (step += 1) {
        const y = try ops.unary(.negative, x);
        release(x);
        x = y;
    }
    defer release(x);
    const in_use = stats().pool_bytes - stats().pool_free;
    try t.expect(stats().pool_bytes - held0 <= in_use + pass_freed_cap + 2 * 8 * 1024 * 1024);
}

test "a pass encodes its dispatches by dependency level: two chains made one after the other run side by side, one barrier between their two levels" {
    _ = metal.device() catch return error.SkipZigTest;
    const ops = @import("ops.zig");
    try sync();
    var x = [2]*Tensor{ try fromHost(std.mem.sliceAsBytes(&[_]f32{ 1, 2, 3, 4 }), .float32, &.{4}), try fromHost(std.mem.sliceAsBytes(&[_]f32{ 5, 6, 7, 8 }), .float32, &.{4}) };
    const b0 = stats().barriers;
    // Made A1 A2 B1 B2 in one pass (the first kick comes after 4); encoded A1 B1 | A2 B2, where made order gave 2.
    for (0..2) |c| for (0..2) |_| {
        const y = try ops.binary(.add, x[c], x[c]);
        release(x[c]);
        x[c] = y;
    };
    try sync();
    try std.testing.expectEqual(@as(u64, 1), stats().barriers - b0);
    var got: [2][4]f32 = undefined;
    for (x, &got) |t, *g| {
        g.* = std.mem.bytesAsSlice(f32, try hostRead(t))[0..4].*;
        release(t);
    }
    try std.testing.expectEqualSlices(f32, &.{ 4, 8, 12, 16, 20, 24, 28, 32 }, &@as([8]f32, @bitCast(got)));
}

test "the double switch is off at 0, off or empty, names a kernel by its text, and refuses a number or other characters" {
    for ([_]?[*:0]const u8{ null, "", "0", "off" }) |v| try std.testing.expectEqual(@as(?[]const u8, null), try doubleSwitch(v));
    try std.testing.expectEqualStrings("exl3_k4_gemm", (try doubleSwitch("exl3_k4_gemm")).?);
    try std.testing.expectEqualStrings("k4", (try doubleSwitch("k4")).?);
    for ([_][*:0]const u8{ "1", "00", "gemm,rms", "rms ", "gemm*" }) |v| try std.testing.expectError(error.EnvUnparsed, doubleSwitch(v));
}

test "a trim waits for the oldest pass whose end frees enough, not for every pass in flight" {
    const t = std.testing;
    // Free buffers as (the last pass that binds each, its bytes), in no order.
    var pending = [_][2]u64{ .{ 7, 200 }, .{ 3, 50 }, .{ 5, 100 }, .{ 3, 25 } };
    try t.expectEqual(@as(?u64, 3), passFreeing(&pending, 1000, 925));
    try t.expectEqual(@as(?u64, 5), passFreeing(&pending, 1000, 850));
    try t.expectEqual(@as(?u64, 7), passFreeing(&pending, 1000, 625));
    try t.expectEqual(@as(?u64, null), passFreeing(&pending, 1000, 600));
    try t.expectEqual(@as(?u64, null), passFreeing(pending[0..0], 1000, 900));
}

test "the pool keeps no more free bytes than its free cap, as MLX's cache keeps no more than its limit" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    try sync();
    trim();
    const was = free_cap;
    defer free_cap = was;
    free_cap = 32 * 1024 * 1024;
    // Buffers of eleven size classes, each freed once no pass binds it: none is asked for again.
    var mb: usize = 1;
    while (mb <= 1024) : (mb *= 2) {
        const a = try fresh(.uint8, &.{@intCast(mb * 1024 * 1024 / 4 * 3)});
        release(a);
        try t.expect(stats().pool_free <= free_cap.?);
    }
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

test "a tensor used after its release is refused by name, before its box or store is reused" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    const a = try fresh(.uint8, &.{100});
    const h = handle(a);
    release(a);
    try t.expectError(error.TensorUnsupported, take(h));
    try t.expectError(error.TensorUnsupported, ready(h));
    refused_why = null;
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

test "the pool's held high-water counts the free buffers it keeps, which the in-use one does not" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    const mb = 1 << 20;
    try sync();
    resetPoolPeak(.process);
    try t.expectEqual(poolTotal(), poolHeldPeak(.process));
    const a = try fresh(.uint8, &.{9 * mb + 5});
    release(a);
    // `a` is free but kept: what Metal holds stays at its fullest after the in-use bytes fell back.
    const held = poolTotal();
    const b = try fresh(.uint8, &.{17 * mb + 5});
    release(b);
    try t.expect(poolHeldPeak(.process) >= held + 17 * mb);
    try t.expect(poolHeldPeak(.process) > poolPeak(.process));
    resetPoolPeak(.process);
    try t.expectEqual(poolTotal(), poolHeldPeak(.process));
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

test "the pool peak trace keeps the tensors in use at the pool's fullest, each with its caller, from the last reset" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    const mb = 1 << 20;
    peak_trace_hook = true;
    defer peak_trace_hook = null;
    try sync();
    resetPoolPeak(.process);
    const a = try fresh(.uint8, &.{5 * mb});
    const b = try fresh(.uint8, &.{7 * mb});
    release(b);
    const c = try fresh(.uint8, &.{mb});
    release(c);
    release(a);
    // The fullest was a and b together; c came after b went.
    var sum: u64 = 0;
    for (peakTrace()) |ask| {
        try t.expect(ask.at[0] != 0);
        sum += ask.len;
    }
    try t.expectEqual(@as(usize, 2), peakTrace().len);
    try t.expectEqual(@as(u64, 12 * mb), sum);
    resetPoolPeak(.process);
    try t.expectEqual(@as(usize, 0), peakTrace().len);
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
    // The budget counts from the buffers of the tensors in use; the free buffers past it go, the largest first.
    const in_use = stats().pool_bytes - stats().pool_free;
    try budget(6 * mb);
    const held = stats();
    try t.expect(held.pool_bytes <= in_use + 6 * mb);
    try t.expectEqual(@as(usize, 0), s.free[classOf(7 * mb + 5)].items.len);
    try t.expectEqual(@as(usize, 1), s.free[classOf(5 * mb + 5)].items.len);
    // A new size inside the budget takes the room of the free buffer.
    const c = try fresh(.uint8, &.{4 * mb + 5});
    try t.expectEqual(@as(usize, 0), s.free[classOf(5 * mb + 5)].items.len);
    try t.expect(stats().pool_bytes <= in_use + 6 * mb);
    // One past it is refused, and the pool is as it was.
    const before = stats().pool_bytes;
    try t.expectError(error.PoolOverBudget, fresh(.uint8, &.{9 * mb}));
    try t.expectEqual(before, stats().pool_bytes);
    release(c);
}

test "a budget bounds what is made after it, never the size rounding of the tensors in use before it" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    const s = try st();
    const was = s.pool_limit;
    defer s.pool_limit = was;
    const mb = 1 << 20;
    try sync();
    trim();
    // A tensor one byte past a size class holds its whole next class: 4 MB more than it asks at 64 MB.
    const held = try fresh(.uint8, &.{64 * mb + 1});
    defer release(held);
    try t.expect(classBytes(classOf(64 * mb + 1)) - (64 * mb + 1) > 3 * mb);
    const want = classBytes(classOf(8 * mb));
    try budget(want);
    const made = try fresh(.uint8, &.{8 * mb});
    release(made);
}

test "under a budget the pool hands the open pass over for the free buffers it binds, rather than refuse" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    const ops = @import("ops.zig");
    const s = try st();
    const was = s.pool_limit;
    defer s.pool_limit = was;
    try sync();
    trim();
    const mb = 1 << 20;
    const x = try fresh(.float32, &.{ 2048, 1024 });
    @memset(try hostBytes(x), 0);
    // The open pass binds x, so its buffer, freed, cannot go until that pass is handed over.
    const y = try ops.unary(.negative, x);
    defer release(y);
    release(x);
    try t.expect(s.pass != null);
    const want = classBytes(classOf(7 * mb));
    s.pool_limit = s.stats.pool_bytes - classBytes(classOf(8 * mb)) + want;
    const z = try fresh(.uint8, &.{7 * mb});
    release(z);
    try t.expect(s.stats.pool_bytes <= s.pool_limit.?);
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

test "GPU time taken is what ran since the last take" {
    const t = std.testing;
    var s: State = .{ .dev = undefined };
    noteGpuSpan(&s, .{ 1.000, 1.002 }, 0);
    noteGpuSpan(&s, .{ 1.003, 1.004 }, 0);
    const first = takeGpuTime(&s);
    try t.expectEqual(@as(u64, 3_000_000), (first[0] + 500) / 1000 * 1000);
    try t.expectEqual(@as(u64, 1_000_000), (first[1] + 500) / 1000 * 1000);
    try t.expectEqual([2]u64{ 0, 0 }, takeGpuTime(&s));
    noteGpuSpan(&s, .{ 1.006, 1.007 }, 0);
    const second = takeGpuTime(&s);
    try t.expectEqual(@as(u64, 1_000_000), (second[0] + 500) / 1000 * 1000);
    try t.expectEqual(@as(u64, 2_000_000), (second[1] + 500) / 1000 * 1000);
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

/// A test recording: its recorder and, dry, its driver.
const TestRecording = struct {
    recorder: plan.Recorder,
    driver: plan.Driver,

    fn begin(self: *TestRecording, dry: ?Fill) !void {
        self.recorder = try plan.Recorder.begin(std.testing.allocator, .{ .site = 1 });
        self.driver = .{ .gpa = std.testing.allocator };
        try beginRecording(if (dry) |f| .{ .dry = .{ .driver = &self.driver, .fill = f } } else .real, &self.recorder);
    }

    fn deinit(self: *TestRecording) void {
        self.driver.deinit();
        self.recorder.deinit();
    }
};

/// A forward in small: an op's output read back, then a scratch sized by the bytes read when `steer`.
fn readThenAsk(steer: bool) !void {
    const ops = @import("ops.zig");
    const ones = [_]f32{ 1, 1, 1, 1 };
    const a = try fromHost(std.mem.sliceAsBytes(&ones), .float32, &.{4});
    defer release(a);
    const sum = try ops.binary(.add, a, a);
    defer release(sum);
    const ids = try ops.astype(sum, .uint32);
    defer release(ids);
    const got = std.mem.bytesAsSlice(u32, try hostResult(ids, 8));
    const scratch = try fresh(.float32, &.{@intCast(if (steer) 1 + got[0] else 4)});
    release(scratch);
}

test "a dry recording refuses a host read of its own GPU work by name, and answers a classified one under its fill" {
    const t = std.testing;
    try mlx.requireGpu();
    const ops = @import("ops.zig");
    const ones = [_]f32{ 1, 1, 1 };
    for ([_]Fill{ .zeros, .ramp }) |fill| {
        var r: TestRecording = undefined;
        try r.begin(fill);
        defer r.deinit();
        defer endRecording() catch {};
        const a = try fromHost(std.mem.sliceAsBytes(&ones), .float32, &.{3});
        defer release(a);
        // Bytes the host filled read as they were filled.
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&ones), try readBytes(a));
        const b = try ops.binary(.add, a, a);
        defer release(b);
        try t.expectError(error.DryReadback, readBytes(b));
        const got = std.mem.bytesAsSlice(u32, try hostResult(b, 2));
        try t.expectEqualSlices(u32, if (fill == .zeros) &.{ 0, 0, 0 } else &.{ 1, 0, 1 }, @alignCast(got));
        try t.expectEqual(@as(u32, 2), recordingReadbacks().all);
        try t.expectEqual(@as(u32, 1), recordingReadbacks().results);
        try t.expectError(error.DryHostWrite, hostBytes(b));
    }
}

test "two fills of a dry recording give one trace where a classified readback steers nothing, and part where it steers a size" {
    const t = std.testing;
    try mlx.requireGpu();
    for ([_]bool{ false, true }) |steer| {
        var under: [2]TestRecording = undefined;
        for (&under, [_]Fill{ .zeros, .ramp }) |*r, fill| {
            try r.begin(fill);
            defer endRecording() catch {};
            try readThenAsk(steer);
        }
        defer for (&under) |*r| r.deinit();
        const d = plan.trace_mod.firstDifference(under[0].recorder.instances.items, under[1].recorder.instances.items);
        const want: ?plan.trace_mod.Difference = if (steer) .{ .instance = 0, .what = .asks, .at = 3 } else null;
        try t.expectEqual(want, d);
    }
}

test "a store a dry recording made, or wrote over, is refused by name where the host reads it after" {
    const t = std.testing;
    try mlx.requireGpu();
    const ops = @import("ops.zig");
    const ones = [_]f32{ 1, 2, 3, 4 };
    const nine = [_]f32{9};
    const one = try fromHost(std.mem.sliceAsBytes(&nine), .float32, &.{1});
    defer release(one);
    // A write pending from before the recording, on a store it alone holds: settled dry, it writes over that store.
    const owed = blk: {
        const a = try fromHost(std.mem.sliceAsBytes(&ones), .float32, &.{4});
        defer release(a);
        break :blk try ops.sliceUpdate(a, one, &.{0}, &.{1});
    };
    defer release(owed);
    var r: TestRecording = undefined;
    try r.begin(.zeros);
    defer r.deinit();
    const made = blk: {
        defer endRecording() catch {};
        try ops.settle(owed);
        break :blk try ops.binary(.add, owed, owed);
    };
    defer release(made);
    try t.expectError(error.DryStore, hostRead(made));
    try t.expectError(error.DryStore, hostRead(owed));
    // A store it only read stays the host's to read.
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(&nine), try hostRead(one));
}

/// `run` recorded dry as an execution of root site 1, merged and packed by lifetimes.
fn planOfTest(run: anytype) !plan.pack.Tables {
    return planOfTestWith(run, .{ .level_rule = false });
}

fn planOfTestWith(run: anytype, options: plan.pack.Options) !plan.pack.Tables {
    var r: TestRecording = undefined;
    try r.begin(.zeros);
    defer r.deinit();
    {
        defer endRecording() catch {};
        try run.go();
    }
    var tr = try r.recorder.end();
    defer tr.deinit();
    var shapes = try plan.shape.merge(std.testing.allocator, &.{tr}, .{});
    defer shapes.deinit();
    return plan.pack.pack(std.testing.allocator, &shapes, options);
}

fn arenaOfTest(tables: *const plan.pack.Tables) !void {
    try arenaFor(.{ .g = tables.g, .h = tables.h, .alignment = tables.alignment });
}

/// A forward in small, read out into `out`: `2a^2 + a` of the ramp `a`, with a scratch that dies before a later one is
/// made, so its plan puts both on one place; `keep` leaves the result alive past the forward.
const SmallForward = struct {
    n: c_int,
    out: []f32,
    kept: ?*?*Tensor = null,

    fn go(self: SmallForward) !void {
        const ops = @import("ops.zig");
        var ramp: [64]f32 = undefined;
        for (&ramp, 0..) |*v, i| v.* = @floatFromInt(i);
        const n: usize = @intCast(self.n);
        const a = try fromHost(std.mem.sliceAsBytes(ramp[0..n]), .float32, &.{self.n});
        defer release(a);
        const b = try ops.binary(.add, a, a);
        const c = blk: {
            defer release(b);
            break :blk try ops.binary(.multiply, b, a);
        };
        const d = blk: {
            defer release(c);
            break :blk try ops.binary(.add, c, a);
        };
        const got = std.mem.bytesAsSlice(f32, try hostResult(d, null));
        for (self.out[0..n], got) |*o, v| o.* = v;
        if (self.kept) |k| k.* = d else release(d);
    }
};

test "arena: an execution gives the bytes of the pool and asks the pool for nothing" {
    const t = std.testing;
    try mlx.requireGpu();
    var out: [2][8]f32 = undefined;
    var tables = try planOfTest(SmallForward{ .n = 8, .out = &out[0] });
    defer tables.deinit();
    try arenaOfTest(&tables);
    defer arenaRelease() catch {};
    try (SmallForward{ .n = 8, .out = &out[0] }).go();
    try sync();
    const before = stats();
    try beginExecution(&tables, .{ .site = 1 });
    try (SmallForward{ .n = 8, .out = &out[1] }).go();
    try endExecution();
    try t.expectEqualSlices(f32, &out[0], &out[1]);
    try t.expectEqual(before.asks, stats().asks);
    try t.expectEqual(before.pool_bytes, stats().pool_bytes);
    try t.expect(stats().arena_asks > before.arena_asks);
    // What the engine counts for the arena is what Metal holds for it: its layout, rounded to the device's pages.
    try t.expectEqual(arena.?.buffer.allocatedSize(), arenaTotal());
    const laid = arena.?.layout.arenaBytes();
    try t.expect(arenaTotal() >= laid and arenaTotal() < laid + 16384);
}

/// Two products of one input where the second lands on the bytes of a scratch the first's reader still reads in the
/// open pass: the pool runs the second beside the first, the arena after that reader.
const Reuse = struct {
    out: []f32,

    fn go(self: Reuse) !void {
        const ops = @import("ops.zig");
        const vals = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        const x = try fromHost(std.mem.sliceAsBytes(&vals), .float32, &.{8});
        defer release(x);
        const a = try ops.binary(.add, x, x);
        const c = blk: {
            defer release(a);
            break :blk try ops.binary(.multiply, a, a);
        };
        defer release(c);
        const b = try ops.binary(.multiply, x, x);
        defer release(b);
        const d = try ops.binary(.add, c, b);
        defer release(d);
        const got = std.mem.bytesAsSlice(f32, try hostResult(d, null));
        for (self.out, got) |*o, v| o.* = v;
    }
};

test "arena: a place whose last tenant is still read in the open pass is written only after that read" {
    const t = std.testing;
    try mlx.requireGpu();
    var out: [2][8]f32 = undefined;
    var tables = try planOfTest(Reuse{ .out = &out[0] });
    defer tables.deinit();
    const root = tables.get(.{ .site = 1 }).?;
    // The fourth ask (x*x) takes the place of the second (x+x).
    try t.expectEqual(root.asks[1].place.offset, root.asks[3].place.offset);
    try arenaOfTest(&tables);
    defer arenaRelease() catch {};
    var barriers: [2]u64 = undefined;
    for (&barriers, 0..) |*b, arm| {
        try sync();
        const before = stats();
        if (arm == 1) try beginExecution(&tables, .{ .site = 1 });
        try (Reuse{ .out = &out[arm] }).go();
        if (arm == 1) try endExecution();
        b.* = stats().barriers - before.barriers;
        if (arm == 1) try t.expectEqual(before.arena_floors + 1, stats().arena_floors);
    }
    try t.expectEqualSlices(f32, &out[0], &out[1]);
    // On the pool x*x runs beside x+x; on its bytes it waits for (x+x)^2, one level more.
    try t.expectEqual(barriers[0] + 1, barriers[1]);
}

/// A host-filled ask and the GPU work on it, nothing read back: the execution's last pass may still be open at its end.
const HostWrite = struct {
    at: *usize,

    fn go(self: HostWrite) !void {
        const ops = @import("ops.zig");
        const vals = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        const x = try fromHost(std.mem.sliceAsBytes(&vals), .float32, &.{8});
        defer release(x);
        self.at.* = x.store.at;
        const y = try ops.binary(.add, x, x);
        release(y);
    }
};

test "arena: a host-filled ask writes its replica of the ring once the execution two before has run, and waits for nothing else" {
    const t = std.testing;
    try mlx.requireGpu();
    var at: [3]usize = undefined;
    var tables = try planOfTest(HostWrite{ .at = &at[0] });
    defer tables.deinit();
    try arenaOfTest(&tables);
    defer arenaRelease() catch {};
    try sync();
    var waits: [3]u64 = undefined;
    for (&at, &waits) |*a, *w| {
        const before = stats().waits;
        try beginExecution(&tables, .{ .site = 1 });
        try (HostWrite{ .at = a }).go();
        try endExecution();
        w.* = stats().waits - before;
    }
    try t.expectEqual(at[0], at[2]);
    try t.expect(at[0] != at[1]);
    // The second fills the other replica; the third the first's, whose GPU work it waits for once.
    try t.expectEqualSlices(u64, &.{ 0, 0, 1 }, &waits);
}

test "arena: an ask past what the plan recorded is refused by name, and nothing is allocated for it" {
    const t = std.testing;
    try mlx.requireGpu();
    var out: [16]f32 = undefined;
    var tables = try planOfTest(SmallForward{ .n = 8, .out = &out });
    defer tables.deinit();
    try arenaOfTest(&tables);
    defer arenaRelease() catch {};
    try sync();
    const before = stats();
    try beginExecution(&tables, .{ .site = 1 });
    try t.expectError(error.PlanMiss, (SmallForward{ .n = 16, .out = &out }).go());
    abortExecution();
    const miss = executionMiss().?;
    try t.expectEqual(plan.binder.MissKind.oversize, miss.kind);
    try t.expectEqual(@as(u64, 16 * 4), miss.asked);
    try t.expectEqual(@as(u64, 8 * 4), miss.cap);
    try t.expectEqual(before.asks, stats().asks);
    try t.expectEqual(before.pool_bytes, stats().pool_bytes);
}

test "arena: a store alive when its execution ends is a miss, and a read of it after is refused by name" {
    const t = std.testing;
    try mlx.requireGpu();
    var out: [8]f32 = undefined;
    var tables = try planOfTest(SmallForward{ .n = 8, .out = &out });
    defer tables.deinit();
    try arenaOfTest(&tables);
    defer arenaRelease() catch {};
    var kept: ?*Tensor = null;
    try beginExecution(&tables, .{ .site = 1 });
    try (SmallForward{ .n = 8, .out = &out, .kept = &kept }).go();
    try t.expectError(error.PlanMiss, endExecution());
    try t.expectEqual(plan.binder.MissKind.lifetime, executionMiss().?.kind);
    try t.expectError(error.PlanEscape, hostRead(kept.?));
    try t.expectError(error.PlanEscape, lend(kept.?));
    // A write owed from it to a tensor outside the execution is refused, never bound.
    const ops = @import("ops.zig");
    const zeros: [8]f32 = @splat(0);
    const owner = try fromHost(std.mem.sliceAsBytes(&zeros), .float32, &.{8});
    const owed = try ops.sliceUpdate(owner, kept.?, &.{0}, &.{8});
    release(owner);
    try t.expectError(error.PlanEscape, ops.settle(owed));
    release(owed);
    // The arena does not go while the escaped store is alive.
    try t.expectError(error.TensorUnsupported, arenaRelease());
    release(kept.?);
    // A host-filled ask holds no place the binder judges: the engine finds it alive at the end.
    var at: usize = 0;
    var host_tables = try planOfTest(HostWrite{ .at = &at });
    defer host_tables.deinit();
    try arenaFor(.{ .g = host_tables.g, .h = host_tables.h, .alignment = host_tables.alignment });
    defer arenaRelease() catch {};
    try beginExecution(&host_tables, .{ .site = 1 });
    const vals = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const host = try fromHost(std.mem.sliceAsBytes(&vals), .float32, &.{8});
    release(try ops.binary(.add, host, host));
    try t.expectError(error.PlanMiss, endExecution());
    try t.expectEqual(plan.binder.MissKind.lifetime, executionMiss().?.kind);
    try t.expectError(error.PlanEscape, hostRead(host));
    release(host);
}

/// A chain of `n` one-dispatch ops on a vector of eight, read out into `out`.
const Chain = struct {
    n: usize,
    out: []f32,

    fn go(self: Chain) !void {
        const ops = @import("ops.zig");
        const vals = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        var x = try fromHost(std.mem.sliceAsBytes(&vals), .float32, &.{8});
        defer release(x);
        for (0..self.n) |_| {
            const y = try ops.binary(.maximum, x, x);
            release(x);
            x = y;
        }
        const got = std.mem.bytesAsSlice(f32, try hostResult(x, null));
        for (self.out, got) |*o, v| o.* = v;
    }
};

test "arena: an execution hands a pass over at its bound on dispatches and waits while its bound on passes is in flight" {
    const t = std.testing;
    try mlx.requireGpu();
    var out: [2][8]f32 = undefined;
    var tables = try planOfTest(Chain{ .n = 64, .out = &out[0] });
    defer tables.deinit();
    try arenaOfTest(&tables);
    defer arenaRelease() catch {};
    const was = execution_pass_bounds;
    execution_pass_bounds = .{ .dispatches = 8, .passes = 2 };
    defer execution_pass_bounds = was;
    var highs: [2]@TypeOf(takePassHighs()) = undefined;
    for (&highs, 0..) |*h, arm| {
        try sync();
        _ = takePassHighs();
        if (arm == 1) try beginExecution(&tables, .{ .site = 1 });
        try (Chain{ .n = 64, .out = &out[arm] }).go();
        if (arm == 1) try endExecution();
        h.* = takePassHighs();
    }
    try t.expectEqualSlices(f32, &out[0], &out[1]);
    // On the pool the kicks' doubling grows a pass past the bound; on the arena none passes it.
    try t.expect(highs[0].longest > 8);
    try t.expect(highs[1].longest <= 8);
    try t.expect(highs[1].flying <= 2);
}

/// A scratch written one byte past its end after its dispatch ran, as a kernel that overruns its output would.
const Overrun = struct {
    fn go(_: Overrun) !void {
        const ops = @import("ops.zig");
        const vals = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        const x = try fromHost(std.mem.sliceAsBytes(&vals), .float32, &.{8});
        defer release(x);
        const y = try ops.binary(.add, x, x);
        defer release(y);
        if (dryOn()) return;
        try sync();
        y.store.ptr[y.store.len] = 0;
    }
};

test "arena: with poison on, a write past a place is found by its canary, and a forward that writes only its own bytes gives the bytes of the pool" {
    const t = std.testing;
    try mlx.requireGpu();
    arena_check.byte = 0xFF;
    defer arena_check.byte = null;
    var over = try planOfTestWith(Overrun{}, .{ .level_rule = false, .guard = 256 });
    defer over.deinit();
    var out: [2][8]f32 = undefined;
    var small = try planOfTestWith(SmallForward{ .n = 8, .out = &out[0] }, .{ .level_rule = false, .guard = 256 });
    defer small.deinit();
    try arenaFor(.{ .g = @max(over.g, small.g), .h = @max(over.h, small.h), .alignment = over.alignment });
    defer arenaRelease() catch {};
    const hits = arena_check.hits;
    try beginExecution(&over, .{ .site = 1 });
    try (Overrun{}).go();
    try endExecution();
    try t.expectEqual(hits + 1, arena_check.hits);
    try (SmallForward{ .n = 8, .out = &out[0] }).go();
    try beginExecution(&small, .{ .site = 1 });
    try (SmallForward{ .n = 8, .out = &out[1] }).go();
    try endExecution();
    try t.expectEqual(hits + 1, arena_check.hits);
    try t.expectEqualSlices(f32, &out[0], &out[1]);
}

/// A sum over 8192 floats: `reduce.all` asks for its partial sums with its pass already open.
const SumAll = struct {
    out: *f32,

    fn go(self: SumAll) !void {
        var vals: [8192]f32 = undefined;
        for (&vals, 0..) |*v, i| v.* = @floatFromInt(i % 7);
        const x = try fromHost(std.mem.sliceAsBytes(&vals), .float32, &.{8192});
        defer release(x);
        const out = try fresh(.float32, &.{1});
        defer release(out);
        try @import("reduce.zig").all(.sum, x, out);
        self.out.* = std.mem.bytesAsSlice(f32, try hostResult(out, null))[0];
    }
};

test "arena: with poison on, an op that asks for scratch with its pass open keeps every dispatch it encodes" {
    const t = std.testing;
    try mlx.requireGpu();
    var got: [2]f32 = undefined;
    var tables = try planOfTestWith(SumAll{ .out = &got[0] }, .{ .level_rule = false, .guard = 256 });
    defer tables.deinit();
    try arenaOfTest(&tables);
    defer arenaRelease() catch {};
    try (SumAll{ .out = &got[0] }).go();
    arena_check.byte = 0xFF;
    defer arena_check.byte = null;
    try beginExecution(&tables, .{ .site = 1 });
    try (SumAll{ .out = &got[1] }).go();
    try endExecution();
    try t.expectEqual(got[0], got[1]);
}

test "arena: an execution ended with a frame still open is refused by name, and the binder is idle after" {
    const t = std.testing;
    try mlx.requireGpu();
    var out: [8]f32 = undefined;
    var tables = try planOfTest(SmallForward{ .n = 8, .out = &out });
    defer tables.deinit();
    try arenaOfTest(&tables);
    defer arenaRelease() catch {};
    try t.expectError(error.TensorUnsupported, arenaFor(.{ .g = tables.g, .h = tables.h, .alignment = tables.alignment, .replicas = 3 }));
    try beginExecution(&tables, .{ .site = 1 });
    // A frame no `end` closes: its code took the error path past its defer, or never had one.
    plan.session.binder.?.enter(.{ .site = 9 }, .exact) catch {};
    try t.expectError(error.PlanMiss, endExecution());
    try t.expect(!arena.?.binder.active());
    try beginExecution(&tables, .{ .site = 1 });
    try (SmallForward{ .n = 8, .out = &out }).go();
    try endExecution();
}

test "arena: its bytes count in the engine's live bytes from its making, as its weights do" {
    const t = std.testing;
    try mlx.requireGpu();
    const gpu_mem = @import("../gpu_mem.zig");
    var out: [8]f32 = undefined;
    var tables = try planOfTest(SmallForward{ .n = 8, .out = &out });
    defer tables.deinit();
    try sync();
    const before = gpu_mem.activeBytes();
    const peak_before = gpu_mem.peakBytes();
    try arenaOfTest(&tables);
    defer arenaRelease() catch {};
    try t.expect(arenaTotal() > 0);
    try t.expectEqual(before + arenaTotal(), gpu_mem.activeBytes());
    try t.expect(gpu_mem.peakBytes() >= peak_before + arenaTotal());
}

test "a closed pool refuses an ask by name and counts it, a census counts and serves it, and a wide phase opens it for its own asks" {
    const t = std.testing;
    _ = metal.device() catch return error.SkipZigTest;
    defer closePool(.open) catch {};
    const before = stats();
    try closePool(.refuse);
    try t.expectError(error.PoolClosed, fresh(.uint8, &.{64}));
    try t.expectEqual(before.closed_asks + 1, stats().closed_asks);
    try t.expectEqual(before.asks, stats().asks);
    {
        const wide = widePhase();
        defer wide.end();
        const a = try fresh(.uint8, &.{64});
        release(a);
    }
    try t.expectEqual(before.closed_asks + 1, stats().closed_asks);
    try t.expectError(error.PoolClosed, fresh(.uint8, &.{64}));
    try closePool(.census);
    const b = try fresh(.uint8, &.{128});
    release(b);
    try t.expectEqual(before.closed_asks + 3, stats().closed_asks);
    try t.expectEqual(before.closed_ask_bytes + 64 + 64 + 128, stats().closed_ask_bytes);
    try closePool(.open);
    const c = try fresh(.uint8, &.{64});
    release(c);
    try t.expectEqual(before.closed_asks + 3, stats().closed_asks);
}

/// The region a forward asks by `plan.region.need` over its dry recording: what a wide execution of it places.
fn regionOfTest(run: anytype) !plan.region.Need {
    var r: TestRecording = undefined;
    try r.begin(.zeros);
    defer r.deinit();
    {
        defer endRecording() catch {};
        try run.go();
    }
    var tr = try r.recorder.end();
    defer tr.deinit();
    return plan.region.need(std.testing.allocator, &tr, 256);
}

test "wide: an execution gives the bytes of the pool, asks the pool for nothing, and its scratch tops out where its recording's first fit says" {
    const t = std.testing;
    try mlx.requireGpu();
    var out: [2][64]f32 = undefined;
    const need = try regionOfTest(SmallForward{ .n = 64, .out = &out[0] });
    try arenaFor(.{ .g = 1 << 20, .h = 64 << 10 });
    defer arenaRelease() catch {};
    try (SmallForward{ .n = 64, .out = &out[0] }).go();
    try sync();
    const before = stats();
    try beginWide();
    try (SmallForward{ .n = 64, .out = &out[1] }).go();
    try endWide();
    try t.expectEqualSlices(f32, &out[0], &out[1]);
    try t.expectEqual(before.asks, stats().asks);
    try t.expectEqual(before.pool_bytes, stats().pool_bytes);
    try t.expect(stats().wide_asks > before.wide_asks);
    try t.expectEqual(need.scratch, stats().wide_high);
}

test "wide: an execution padded to its recording asks each place at the recorded size, so it tops out where the recording's first fit does though its own asks are smaller; an ask past its size or past the sequence is refused by name" {
    const t = std.testing;
    try mlx.requireGpu();
    try arenaFor(.{ .g = 1 << 20, .h = 64 << 10 });
    defer arenaRelease() catch {};
    const u: c_int = 64 << 10;
    const U: u64 = 64 << 10;
    // First fit is not monotone: a smaller first ask leaves a gap the third no longer fits, so it goes on top.
    for ([_]?[]const u64{ null, &.{ 2 * U, U, 2 * U } }, [_]u64{ 4 * U, 3 * U }) |pad, high| {
        if (pad) |sizes| try beginWidePadded(&.{.{ .sizes = sizes }}) else try beginWide();
        const a = try fresh(.uint8, &.{u});
        const b = try fresh(.uint8, &.{u});
        release(a);
        const c = try fresh(.uint8, &.{2 * u});
        release(b);
        release(c);
        try endWide();
        try t.expectEqual(high, stats().wide_last_high);
    }
    try beginWidePadded(&.{.{ .sizes = &.{U} }});
    try t.expectError(error.PlanMiss, fresh(.uint8, &.{2 * u}));
    abortWide();
    try beginWidePadded(&.{.{ .sizes = &.{U} }});
    const a = try fresh(.uint8, &.{u});
    try t.expectError(error.PlanMiss, fresh(.uint8, &.{u}));
    release(a);
    try endWide();
}

test "wide: a smaller execution padded to a larger one's recording runs as it would, refused nothing, and tops out at the recording's first fit" {
    const t = std.testing;
    try mlx.requireGpu();
    var r: TestRecording = undefined;
    try r.begin(.zeros);
    var big: [64]f32 = undefined;
    {
        defer endRecording() catch {};
        try (SmallForward{ .n = 64, .out = &big }).go();
    }
    var tr = try r.recorder.end();
    defer tr.deinit();
    r.deinit();
    const need = try plan.region.need(t.allocator, &tr, 256);
    const pad = try plan.region.padOf(t.allocator, &tr, 256);
    defer t.allocator.free(pad);
    try arenaFor(.{ .g = 1 << 20, .h = 64 << 10 });
    defer arenaRelease() catch {};
    var out: [2][32]f32 = undefined;
    try (SmallForward{ .n = 32, .out = &out[0] }).go();
    try sync();
    try beginWidePadded(&.{.{ .sizes = pad }});
    try (SmallForward{ .n = 32, .out = &out[1] }).go();
    try endWide();
    try t.expectEqualSlices(f32, &out[0], &out[1]);
    try t.expectEqual(need.scratch, stats().wide_last_high);
}

/// A forward that takes `picks.len` choices at one site: two pages at once, or one and then another; `leak` keeps the
/// first choice's pages past it.
const ChoiceForward = struct {
    picks: []const bool,
    leak: bool = false,

    fn go(self: ChoiceForward) !void {
        const u: c_int = 64 << 10;
        const x = try fresh(.uint8, &.{u});
        var kept: ?*Tensor = null;
        for (self.picks, 0..) |real, i| {
            if (plan.choose(@src(), bool, real)) {
                const a = try fresh(.uint8, &.{2 * u});
                if (self.leak and i == 0) kept = a else release(a);
            } else {
                const a = try fresh(.uint8, &.{u});
                const b = try fresh(.uint8, &.{u});
                release(a);
                release(b);
            }
        }
        if (kept) |k| release(k);
        const y = try fresh(.uint8, &.{u});
        release(y);
        release(x);
    }
};

/// Every pass of `run`'s dry recording, one trace each.
fn passesOfTest(run: anytype) ![]plan.trace_mod.Trace {
    var traces: std.ArrayList(plan.trace_mod.Trace) = .empty;
    errdefer {
        for (traces.items) |*tr| tr.deinit();
        traces.deinit(std.testing.allocator);
    }
    var driver: plan.Driver = .{ .gpa = std.testing.allocator };
    defer driver.deinit();
    while (true) {
        var rec = try plan.Recorder.begin(std.testing.allocator, .{ .site = 1 });
        defer rec.deinit();
        try beginRecording(.{ .dry = .{ .driver = &driver, .fill = .zeros } }, &rec);
        {
            defer endRecording() catch {};
            try run.go();
        }
        try traces.append(std.testing.allocator, try rec.end());
        if (!try driver.next()) break;
    }
    return traces.toOwnedSlice(std.testing.allocator);
}

test "wide: an execution padded to every pass of its recording follows its own choices from pass to pass, mixed as no pass took them, and tops out within the passes' first fit; passes whose choices leave the scratch apart are told" {
    const t = std.testing;
    try mlx.requireGpu();
    const traces = try passesOfTest(ChoiceForward{ .picks = &.{ true, true } });
    defer {
        for (traces) |*tr| tr.deinit();
        t.allocator.free(traces);
    }
    try t.expectEqual(@as(usize, 2), traces.len);
    try t.expect(try plan.region.choicesAgree(t.allocator, traces, 256));
    var passes: [2]PadPass = undefined;
    var most: u64 = 0;
    for (traces, &passes) |*tr, *ps| {
        ps.* = .{ .sizes = try plan.region.padOf(t.allocator, tr, 256), .choices = tr.choices };
        most = @max(most, (try plan.region.need(t.allocator, tr, 256)).scratch);
    }
    defer for (passes) |ps| t.allocator.free(ps.sizes);
    try arenaFor(.{ .g = 1 << 20, .h = 64 << 10 });
    defer arenaRelease() catch {};
    for ([_][2]bool{ .{ true, false }, .{ false, true }, .{ false, false } }) |picks| {
        try beginWidePadded(&passes);
        try (ChoiceForward{ .picks = &picks }).go();
        try endWide();
        try t.expect(stats().wide_last_high <= most);
    }
    const apart = try passesOfTest(ChoiceForward{ .picks = &.{ true, true }, .leak = true });
    defer {
        for (apart) |*tr| tr.deinit();
        t.allocator.free(apart);
    }
    try t.expect(!try plan.region.choicesAgree(t.allocator, apart, 256));
}

/// A choice made in a frame whose two arms ask the same bytes; `hand_up` keeps the arm's ask past the frame's end.
const FramedChoiceForward = struct {
    hand_up: bool,

    fn go(self: FramedChoiceForward) !void {
        const u: c_int = 64 << 10;
        var out: ?*Tensor = null;
        {
            const f = try plan.frame(@src(), .{});
            defer f.end();
            _ = plan.choose(@src(), bool, true);
            const a = try fresh(.uint8, &.{u});
            if (self.hand_up) out = a else release(a);
        }
        if (out) |o| release(o);
        const y = try fresh(.uint8, &.{u});
        release(y);
    }
};

test "wide: an ask made after a choice that outlives the frame the choice was made in parts the passes, though first fit agrees after either arm" {
    const t = std.testing;
    try mlx.requireGpu();
    for ([_]bool{ false, true }) |hand_up| {
        const traces = try passesOfTest(FramedChoiceForward{ .hand_up = hand_up });
        defer {
            for (traces) |*tr| tr.deinit();
            t.allocator.free(traces);
        }
        try t.expectEqual(@as(usize, 2), traces.len);
        try t.expectEqual(!hand_up, try plan.region.choicesAgree(t.allocator, traces, 256));
    }
}

test "wide: a gap reused while its last tenant is still read in the open pass is written only after that read" {
    const t = std.testing;
    try mlx.requireGpu();
    var out: [2][8]f32 = undefined;
    try arenaFor(.{ .g = 1 << 20, .h = 64 << 10 });
    defer arenaRelease() catch {};
    var barriers: [2]u64 = undefined;
    for (&barriers, 0..) |*b, arm| {
        try sync();
        const before = stats();
        if (arm == 1) try beginWide();
        try (Reuse{ .out = &out[arm] }).go();
        if (arm == 1) try endWide();
        b.* = stats().barriers - before.barriers;
        if (arm == 1) try t.expect(stats().arena_floors > before.arena_floors);
    }
    try t.expectEqualSlices(f32, &out[0], &out[1]);
    // On the pool x*x runs beside x+x; on its bytes it waits for (x+x)^2, one level more.
    try t.expectEqual(barriers[0] + 1, barriers[1]);
}

test "wide: an ask past the region is refused by name and takes nothing; a store alive at the end is refused by name" {
    const t = std.testing;
    try mlx.requireGpu();
    try arenaFor(.{ .g = 8 << 10, .h = 4 << 10 });
    defer arenaRelease() catch {};
    const before = stats();
    try beginWide();
    const a = try fresh(.uint8, &.{6 << 10});
    try t.expectError(error.WideFull, fresh(.uint8, &.{4 << 10}));
    try t.expectEqual(@as(u64, 8 << 10), refused_wide.?.room);
    try t.expect(!refused_wide.?.host);
    try t.expectError(error.WideFull, freshHost(.uint8, &.{8 << 10}));
    // The first refusal is the one kept.
    try t.expect(!refused_wide.?.host);
    try t.expectError(error.PlanEscape, endWide());
    release(a);
    try t.expectEqual(before.asks, stats().asks);
    try t.expectEqual(before.pool_bytes, stats().pool_bytes);
}

const HostAfterWait = struct {
    at: *[2]usize,

    fn go(self: HostAfterWait) !void {
        const ops = @import("ops.zig");
        const vals = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        const x = try fromHost(std.mem.sliceAsBytes(&vals), .float32, &.{8});
        self.at[0] = x.store.at;
        const y = try ops.binary(.add, x, x);
        release(x);
        try evalPoint();
        const z = try fromHost(std.mem.sliceAsBytes(&vals), .float32, &.{8});
        self.at[1] = z.store.at;
        const w = try ops.binary(.add, y, z);
        release(y);
        release(z);
        try evalPoint();
        release(w);
    }
};

test "wide: a host-filled ask after an eval takes the host bytes of the asks dead before it, as its recording's region says" {
    const t = std.testing;
    try mlx.requireGpu();
    var at: [2]usize = undefined;
    const need = try regionOfTest(HostAfterWait{ .at = &at });
    try t.expectEqual(@as(u64, 256), need.host);
    try arenaFor(.{ .g = 64 << 10, .h = 16 << 10 });
    defer arenaRelease() catch {};
    try beginWide();
    try (HostAfterWait{ .at = &at }).go();
    try endWide();
    try t.expectEqual(at[0], at[1]);
}

test "wide: a host-filled ask takes the host ring's replica of its execution, its scratch the region past the ring" {
    const t = std.testing;
    try mlx.requireGpu();
    try arenaFor(.{ .g = 64 << 10, .h = 16 << 10 });
    defer arenaRelease() catch {};
    var at: [2]usize = undefined;
    for (&at) |*x| {
        try beginWide();
        try (HostWrite{ .at = x }).go();
        try endWide();
    }
    // Two executions fill the two replicas in turn.
    try t.expect(at[0] < 2 * (16 << 10) and at[1] < 2 * (16 << 10));
    try t.expect(at[0] / (16 << 10) != at[1] / (16 << 10));
    try beginWide();
    const s = try fresh(.float32, &.{8});
    try t.expect(s.store.at >= 2 * (16 << 10));
    release(s);
    try endWide();
}

test "engine: a dispatch's level is its stores' last use, and a later pass starts them over" {
    const t = std.testing;
    try mlx.requireGpu();
    const s = try st();
    const a = try fresh(.float32, &.{4});
    defer release(a);
    const b = try fresh(.float32, &.{4});
    defer release(b);
    var p = try pass();
    bind(p, a, 0);
    bindOut(p, b, 0);
    beforeDispatch();
    try t.expectEqual(@as(u32, 2), s.rec.next_level);
    bind(p, b, 0);
    beforeDispatch();
    try t.expectEqual(@as(u32, 4), s.rec.next_level);
    bind(p, b, 0);
    bind(p, b, 1);
    beforeDispatch();
    try t.expectEqual(@as(u32, 4), s.rec.next_level);
    bind(p, b, 0);
    bindOut(p, b, 1);
    beforeDispatch();
    try t.expectEqual(@as(u32, 6), s.rec.next_level);
    bind(p, b, 0);
    beforeDispatch();
    try t.expectEqual(@as(u32, 8), s.rec.next_level);
    try submit();
    p = try pass();
    bind(p, b, 0);
    beforeDispatch();
    try t.expectEqual(@as(u32, 2), s.rec.next_level);
    try submit();
}
