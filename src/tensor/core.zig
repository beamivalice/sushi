//! Tensors of our own on our own Metal queue: the handles `mlx.zig` routes here in place of MLX arrays.
//!
//! Eager: an op encodes its dispatch into the open pass at once and returns a tensor over the output buffer. A view
//! (reshape, slice, transpose) is arithmetic on the host and encodes nothing. `submit` hands the pass to the GPU,
//! `sync` waits for everything submitted; a host read of a tensor's bytes syncs first.
//!
//! One thread drives the engine, the inference thread, as it does MLX.

const std = @import("std");
const metal = @import("../metal.zig");
const metal_mlx = @import("../metal_mlx.zig");
const mlx = @import("../mlx.zig");

pub const max_dims = 8;
pub const Dtype = mlx.mlx_dtype;

pub const Error = metal.Error || error{ OutOfMemory, MlxError, NotEvaluated, NotRowContiguous, TensorShape, TensorDtype, TensorUnsupported, PoolOverBudget };

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
    };
};

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
const Flying = struct { cmd: metal.Committed, fence: metal.Fence, seq: u64, phase: u8 = 0 };

/// What the host is building, set by the model: a pause is charged to the phase of the pass that ends it.
pub var phase: u8 = 0;
pub const phases = 6;
const Finished = struct { cmd: metal.Committed, phase: u8 };

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
    /// MLX arrays read by an op of ours.
    imports: u64 = 0,
    /// Tensors handed to MLX as arrays over their own bytes.
    lends: u64 = 0,
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

/// A store of at least `len` bytes from the pool, never one the open pass binds. `for_host` asks for one no
/// submitted pass still binds either, since the host is about to write it. A new buffer stays inside the budget:
/// free buffers go to make room, and with none left the op is refused.
fn poolStore(len: usize, for_host: bool) Error!*Store {
    const s = try st();
    takeBack();
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
        store.refs = 1;
        store.len = len;
        s.stats.pool_free -= bytes;
        s.stats.pool_asked += len;
        return store;
    }
    if (s.pool_limit) |limit| {
        if (!try trimTo(s, limit -| bytes)) return error.PoolOverBudget;
    }
    const buffer = try s.dev.alloc(bytes);
    errdefer buffer.release();
    const store = gpa.create(Store) catch return error.OutOfMemory;
    store.* = .{ .buffer = buffer, .at = 0, .ptr = buffer.bytes().ptr, .len = len, .origin = .{ .pool = @intCast(class) } };
    s.stats.pool_bytes += bytes;
    s.stats.pool_asked += len;
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
                store.buffer.release();
                s.stats.pool_bytes -= classBytes(class);
                gpa.destroy(store);
            }
        },
        // The buffer aliases the array's pages: the last pass that binds it runs before the array may go.
        .mlx => if (store.seq > s.done) {
            s.retired.append(gpa, store) catch {
                syncState(s) catch {};
                freeImport(store);
            };
        } else freeImport(store),
    }
}

fn freeImport(store: *Store) void {
    _ = mlx.mlx_array_free(store.origin.mlx);
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
/// offset 0.
pub fn lend(t: *Tensor) Error!mlx.mlx_array {
    if (t.twin.ctx != null) return t.twin;
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
    t.twin = arr;
    s.stats.lends += 1;
    return arr;
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
    return t;
}

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
    errdefer state.?.boxes.append(gpa, t) catch gpa.destroy(t);
    try setShape(t, shape);
    t.dtype = dtype;
    t.offset = 0;
    t.store = try poolStore(@max(t.nbytes(), 1), for_host);
    return t;
}

/// A tensor holding a copy of `bytes`.
pub fn fromHost(bytes: []const u8, dtype: Dtype, shape: []const c_int) Error!*Tensor {
    const t = try freshFor(dtype, shape, true);
    errdefer release(t);
    if (t.nbytes() != bytes.len) return error.TensorShape;
    @memcpy(t.store.ptr[0..bytes.len], bytes);
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
    if (t.twin.ctx != null) _ = mlx.mlx_array_free(t.twin);
    if (t.pending) |p| freePending(p);
    dropStore(t.store);
    state.?.boxes.append(gpa, t) catch gpa.destroy(t);
}

// ── MLX memory the engine reads ──

/// A tensor over the memory of an MLX array, evaluated here if it was not; no copy. A pass of ours binds MLX's own
/// Metal buffer, so the memory is wired once. A strided view keeps its strides. The store keeps the array, and
/// with it the buffer, until every pass that binds it has run.
pub fn import(a: mlx.mlx_array) Error!*Tensor {
    const s = try st();
    var available = false;
    try mlx.check(mlx._mlx_array_is_available(&available, a));
    if (!available) try mlx.check(mlx.rawArrayEval(a));
    var raw: ?*anyopaque = null;
    var at: usize = 0;
    try mlx.check(mlx._mlx_array_metal_buffer(&raw, &at, a));
    const ptr: [*]u8 = @constCast(mlx.mlx_array_data_uint8(a) orelse return error.NotEvaluated);
    const dtype = mlx.mlx_array_dtype(a);
    const t = try box();
    errdefer s.boxes.append(gpa, t) catch gpa.destroy(t);
    try setShape(t, mlx.getShape(a));
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
    state.?.boxes.append(gpa, from) catch gpa.destroy(from);
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
        if (s.pass_dispatches >= s.kick_after and s.next_in.count() == 0 and s.next_out.count() == 0 and (s.burst or gpuIdle(s))) {
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
pub var sealed: bool = false;

/// A pipeline named `name` was made since `t0`.
pub fn madePipeline(name: []const u8, t0: u64) void {
    const ns = nowNs() - t0;
    pipelines += 1;
    pipeline_ns += ns;
    const log = @import("../log.zig");
    if (!sealed) return log.debug("[engine] compiled {s}\n", .{name});
    pipelines_late += 1;
    log.warn("[engine] compiled inside a request ({d} ms): {s}\n", .{ ns / std.time.ns_per_ms, name });
}

pub fn nowNs() u64 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t);
    return @as(u64, @intCast(t.sec)) * std.time.ns_per_s + @as(u64, @intCast(t.nsec));
}

/// Ends the open pass's encoding rules and hands it to the GPU: it waits for the passes that wrote what it binds
/// and marks what it writes.
fn commitPass(s: *State, open: metal.Pass) Error!void {
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
    var writes = s.pass_writes.keyIterator();
    while (writes.next()) |store| s.writers.put(gpa, store.*, flying) catch return error.OutOfMemory;
    open.updateFence(fence);
    s.pass_binds.clearRetainingCapacity();
    s.pass_writes.clearRetainingCapacity();
    s.flying.append(gpa, .{ .cmd = open.commit(), .fence = fence, .seq = s.seq, .phase = phase }) catch return error.OutOfMemory;
    s.stats.commit_ns += nowNs() - t0;
    s.seq += 1;
    s.pass_dispatches = 0;
    s.stats.submits += 1;
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

/// Ends the open pass and waits for everything submitted; nothing when the engine never ran. A thread that ends
/// with an encoder still open trips a Metal assertion as its autorelease pool drains.
pub fn close() void {
    const s = &(state orelse return);
    syncState(s) catch {};
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
        s.untimed.append(gpa, .{ .cmd = f.cmd, .phase = f.phase }) catch {
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
        } else freeImport(store);
    }
    s.retired.shrinkRetainingCapacity(kept);
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
