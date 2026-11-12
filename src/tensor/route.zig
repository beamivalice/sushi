//! The mlx-c entry points as the engine answers them: `mlx.zig` sends a call here when its stream is the engine's or
//! an array it takes is a tensor of ours. Names and signatures are mlx-c's; an op the engine lacks is refused by
//! name, never passed on.

const std = @import("std");
const mlx = @import("../mlx.zig");
const metal = @import("../metal.zig");
const core = @import("core.zig");
const ops = @import("ops.zig");
pub const kernel = @import("kernel.zig");
const stock = @import("stock.zig");
const sorting = @import("sort.zig");
const scan = @import("scan.zig");
const random = @import("random.zig");
const quantized = @import("quantized.zig");
const attention = @import("attention.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const V = mlx.mlx_vector_array;
const Tensor = core.Tensor;
const gpa = std.heap.c_allocator;

/// Read once: the engine exists in this process. Off, no call is ever routed here and no kernel is recorded.
pub var enabled: bool = false;

/// On while a forward of ours runs: the makers that take no stream and no array (a host array, a vector, a kernel
/// config) make engine objects.
pub var active: bool = false;

/// Host arrays and vectors made between `enter` and `Scope.end` are the engine's when `s` is its stream, and make no
/// call into MLX. On any other stream it changes nothing.
pub fn enter(s: S) Scope {
    const own = core.isStream(s);
    const was = active;
    if (own) active = true;
    return .{ .own = own, .was = was };
}

pub const Scope = struct {
    own: bool,
    was: bool,

    pub fn end(self: Scope) void {
        if (self.own) active = self.was;
    }
};

var holds: u32 = 0;

/// MLX's allocator as the first hold found it, its pool emptied. Nothing of MLX's is made or freed under a hold, so
/// the readings stay true to the hold's end and the seam answers with them, on any thread, without a call.
pub const MlxAtRest = struct {
    on: std.atomic.Value(bool) = .init(false),
    active: std.atomic.Value(usize) = .init(0),
    cache: std.atomic.Value(usize) = .init(0),
    peak: std.atomic.Value(usize) = .init(0),
    external: std.atomic.Value(usize) = .init(0),
};
pub var mlx_at_rest: MlxAtRest = .{};

/// A load that is the engine's alone holds the engine's scope open from its first tensor to its unload: every host
/// array and vector made for it is the engine's, and an op on an MLX stream is refused by name.
pub fn hold() void {
    enabled = true;
    if (holds == 0) mlx.restAllocator(&mlx_at_rest);
    holds += 1;
    active = true;
}

pub fn unhold() void {
    holds -= 1;
    if (holds != 0) return;
    active = false;
    mlx_at_rest.on.store(false, .release);
}

pub fn holding() bool {
    return holds > 0;
}

/// Whether a load's mapped weights are wrapped by the engine instead of MLX (`model.wrapRegion`). Set for the span of
/// a load that runs every forward on the engine: a weight of the engine's lent to MLX is wired a second time.
pub var own_weights: bool = false;

/// The mapping at `ptr` as a tensor of the engine's (`core.wrapMapped`); a null handle, the reason latched, when it
/// cannot be wrapped.
pub fn wrapMapped(ptr: [*]u8, dims: []const c_int, dtype: mlx.mlx_dtype, payload: ?*anyopaque, release: core.MappedRelease) A {
    const t = core.wrapMapped(ptr, dtype, dims, payload, release) catch |e| {
        _ = fail("wrap_mapped", e);
        return .{};
    };
    return core.handle(t);
}

/// Ops that went to MLX while the engine is on: what a request still builds there.
pub var mlx_ops: u64 = 0;

/// Calls into mlx-c since the process started, whichever side asked (`mlx.calls`).
pub fn mlxCalls() u64 {
    return mlx.calls.load(.monotonic);
}

/// Whether kernels and their configs are noted for the engine as they are made. A module keeps both for the life of
/// the process; a test process turns the engine on after other tests made theirs.
pub fn noting() bool {
    return enabled or @import("builtin").is_test;
}

pub const Side = enum {
    mlx,
    engine,
    /// An MLX stream while a forward of ours runs: refused, so no op of that forward runs on MLX unseen.
    leak,
};

/// Which side answers a call with these arguments. A stream decides: the engine's is the engine, MLX's is MLX
/// (which takes a tensor of ours as an array over the same bytes). A call without a stream follows its handles.
pub inline fn side(args: anytype) Side {
    if (!enabled) return .mlx;
    inline for (args) |arg| {
        if (@TypeOf(arg) == S) return if (core.isStream(arg)) .engine else if (active) .leak else .mlx;
    }
    // A map's call follows the map, whatever it is given to hold.
    inline for (args) |arg| {
        switch (@TypeOf(arg)) {
            MA, MS, MI => return if (tagged(arg.ctx)) .engine else .mlx,
            else => {},
        }
    }
    inline for (args) |arg| {
        switch (@TypeOf(arg)) {
            A => if (core.isOurs(arg)) return .engine,
            V => if (isVec(arg)) return .engine,
            else => {},
        }
    }
    return .mlx;
}

/// Calls refused by name since the process began: a served request counts none.
pub var refusals: u64 = 0;

/// A call the engine does not answer, by name.
pub fn refuse(comptime name: []const u8, comptime why: []const u8) c_int {
    refusals += 1;
    mlx.latchError("[engine] " ++ name ++ ": " ++ why);
    return 1;
}

/// `a`, a tensor of ours, as an MLX array over the same bytes; the tensor keeps the array. A view is made
/// row-contiguous first.
pub fn lend(a: A) ?A {
    return lendOf(a) catch |e| {
        _ = fail("lend", e);
        return null;
    };
}

/// Whether the array `lend(a)` gives is the caller's to free after the MLX call that takes it (`core.lendsOnce`).
pub fn lendsOnce(a: A) bool {
    return core.lendsOnce(core.of(a));
}

fn lendOf(a: A) core.Error!A {
    const t = try core.ready(a);
    if (t.twin.ctx == null and (t.offset != 0 or !t.rowContiguous())) core.adopt(t, try ops.contiguousCopy(t));
    return core.lend(t);
}


/// An MLX vector of `items`, each tensor of ours lent.
pub fn lentVector(items: []const A) V {
    var stack: [16]A = undefined;
    const lent = if (items.len <= stack.len) stack[0..items.len] else gpa.alloc(A, items.len) catch return .{};
    defer if (items.len > stack.len) gpa.free(lent);
    var once_stack: [16]bool = undefined;
    const once = if (items.len <= stack.len) once_stack[0..items.len] else gpa.alloc(bool, items.len) catch return .{};
    defer if (items.len > stack.len) gpa.free(once);
    @memset(once, false);
    // The vector takes its own reference to each array: a lend made for it alone is freed once it has.
    defer for (lent, once) |a, mine| {
        if (mine) _ = mlx.raw.mlx_array_free(a);
    };
    for (items, lent, once) |a, *out, *mine| {
        if (!core.isOurs(a)) {
            out.* = a;
            continue;
        }
        mlx.noteWeightLend("a vector for MLX", a);
        const made_once = lendsOnce(a);
        out.* = lend(a) orelse return .{};
        mine.* = made_once;
    }
    return mlx.raw.mlx_vector_array_new_data(lent.ptr, lent.len);
}

fn fail(comptime what: []const u8, e: anyerror) c_int {
    var buf: [512]u8 = undefined;
    const said = metal.lastError();
    const msg = switch (e) {
        error.ThreadGroupTooLarge => std.fmt.bufPrintSentinel(&buf, "[engine] Thread group size ({d}) is greater than the maximum allowed threads per threadgroup ({d}).", .{ kernel.refused_threads, kernel.refused_limit }, 0),
        error.MetalCompile, error.MetalCommand, error.MetalBuffer => std.fmt.bufPrintSentinel(&buf, "[engine] {s}: {s}: {s}", .{ what, @errorName(e), said[0..@min(said.len, 400)] }, 0),
        error.PoolOverBudget => std.fmt.bufPrintSentinel(&buf, "[engine] {s}: PoolOverBudget: a buffer of {d} KB with {d} MB held for the pool, {d} MB of it asked by tensors in use, under a limit of {d} MB", .{ what, core.refused_budget.wanted >> 10, core.refused_budget.held >> 20, core.refused_budget.asked >> 20, core.refused_budget.limit >> 20 }, 0),
        error.ReadsFillingSlab => std.fmt.bufPrintSentinel(&buf, "[engine] {s}: ReadsFillingSlab: a pass of {d} dispatches binds {d} bytes at 0x{x} while a fill writes {d} bytes at 0x{x}", .{ what, core.refused_fill.dispatches, core.refused_fill.read_len, core.refused_fill.read, core.refused_fill.fill_len, core.refused_fill.fill }, 0),
        else => std.fmt.bufPrintSentinel(&buf, "[engine] {s}: {s}", .{ what, @errorName(e) }, 0),
    } catch "[engine] error";
    mlx.latchError(msg);
    return 1;
}

fn put(res: *A, t: *Tensor) c_int {
    if (res.ctx != null) _ = mlx_array_free(res.*);
    res.* = core.handle(t);
    return 0;
}

fn op1(comptime what: []const u8, res: *A, a: A, f: anytype, extra: anytype) c_int {
    const x = core.take(a) catch |e| return fail(what, e);
    defer core.release(x);
    return put(res, @call(.auto, f, .{x} ++ extra) catch |e| return fail(what, e));
}

fn op2(comptime what: []const u8, res: *A, a: A, b: A, f: anytype, extra: anytype) c_int {
    const x = core.take(a) catch |e| return fail(what, e);
    defer core.release(x);
    const y = core.take(b) catch |e| return fail(what, e);
    defer core.release(y);
    return put(res, @call(.auto, f, .{ x, y } ++ extra) catch |e| return fail(what, e));
}

// ── Arrays ──

pub fn mlx_array_free(a: A) c_int {
    if (core.isOurs(a)) {
        core.release(core.of(a));
        return 0;
    }
    return mlx.raw.mlx_array_free(a);
}

pub fn mlx_array_set(dst: *A, src: A) c_int {
    if (!core.isOurs(src)) {
        if (core.isOurs(dst.*)) {
            core.release(core.of(dst.*));
            dst.ctx = null;
        }
        return mlx.raw.mlx_array_set(dst, src);
    }
    const from = core.ready(src) catch |e| return fail("array_set", e);
    return put(dst, core.share(from) catch |e| return fail("array_set", e));
}

pub fn mlx_array_new_data(host: ?*const anyopaque, dims: [*]const c_int, dim: c_int, dtype: mlx.mlx_dtype) A {
    const n: usize = @intCast(dim);
    var count: usize = core.itemSize(dtype);
    for (dims[0..n]) |d| count *= @intCast(d);
    const bytes = @as([*]const u8, @ptrCast(host.?))[0..count];
    const t = core.fromHost(bytes, dtype, dims[0..n]) catch |e| {
        _ = fail("array_new_data", e);
        return .{};
    };
    return core.handle(t);
}

/// An array of ours over a buffer no pass binds, and its bytes for the caller to fill before any op reads it.
pub fn hostFilled(dtype: mlx.mlx_dtype, dims: []const c_int, bytes: *[]u8) ?A {
    const t = core.freshHost(dtype, dims) catch |e| {
        _ = fail("host_array", e);
        return null;
    };
    bytes.* = t.store.ptr[0..t.nbytes()];
    return core.handle(t);
}

fn scalar(comptime T: type, value: T, dtype: mlx.mlx_dtype) A {
    const v = value;
    const t = core.fromHost(std.mem.asBytes(&v), dtype, &.{}) catch |e| {
        _ = fail("array_new_scalar", e);
        return .{};
    };
    return core.handle(t);
}

pub fn mlx_array_new_float(val: f32) A {
    return scalar(f32, val, .float32);
}

pub fn mlx_array_new_int(val: c_int) A {
    return scalar(c_int, val, .int32);
}

pub fn mlx_array_new_bool(val: bool) A {
    return scalar(bool, val, .bool_);
}

pub fn ndim(a: A) usize {
    return core.of(a).ndim;
}

pub fn shape(a: A) [*]const c_int {
    return &core.of(a).shape;
}

pub fn size(a: A) usize {
    return core.of(a).size();
}

pub fn dtypeOf(a: A) mlx.mlx_dtype {
    return core.of(a).dtype;
}

pub fn mlx_array_strides(a: A) [*]const usize {
    return &core.of(a).strides;
}

pub fn mlx_array_itemsize(a: A) usize {
    return core.itemSize(core.of(a).dtype);
}

/// The bytes are there once the pass that last wrote them has run.
pub fn mlx_array_eval(a: A) c_int {
    const t = core.ready(a) catch |e| return fail("array_eval", e);
    _ = core.hostBytes(t) catch |e| switch (e) {
        error.NotRowContiguous => core.sync() catch |e2| return fail("array_eval", e2),
        else => return fail("array_eval", e),
    };
    return 0;
}

pub fn _mlx_array_is_available(res: *bool, a: A) c_int {
    res.* = core.of(a).pending == null and core.settled(core.of(a));
    return 0;
}

pub fn data(comptime T: type, a: A) ?[*]const T {
    const t = core.ready(a) catch |e| {
        _ = fail("array_data", e);
        return null;
    };
    const bytes = core.hostBytes(t) catch |e| {
        _ = fail("array_data", e);
        return null;
    };
    return @ptrCast(@alignCast(bytes.ptr));
}

pub fn item(comptime T: type, res: *T, a: A) c_int {
    const t = core.ready(a) catch |e| return fail("array_item", e);
    if (t.size() != 1) return fail("array_item", error.TensorShape);
    const bytes = core.hostBytes(t) catch |e| return fail("array_item", e);
    res.* = std.mem.bytesToValue(T, bytes[0..@sizeOf(T)]);
    return 0;
}

pub fn mlx_array_data_bool(a: A) ?[*]const bool {
    return data(bool, a);
}
pub fn mlx_array_data_float32(a: A) ?[*]const f32 {
    return data(f32, a);
}
pub fn mlx_array_data_float16(a: A) ?[*]const f16 {
    return data(f16, a);
}
pub fn mlx_array_data_bfloat16(a: A) ?[*]const u16 {
    return data(u16, a);
}
pub fn mlx_array_data_int32(a: A) ?[*]const i32 {
    return data(i32, a);
}
pub fn mlx_array_data_uint16(a: A) ?[*]const u16 {
    return data(u16, a);
}
pub fn mlx_array_data_uint32(a: A) ?[*]const u32 {
    return data(u32, a);
}
pub fn mlx_array_data_uint8(a: A) ?[*]const u8 {
    return data(u8, a);
}

pub const isOurs = core.isOurs;
pub const isMapped = core.isMapped;
pub const storePointer = core.storePointer;
pub const stream = core.stream;
pub const stats = core.stats;
pub const poolTotal = core.poolTotal;
pub const poolFree = core.poolFree;

/// What the pool refused last for its budget (`core.refused_budget`).
pub fn refusedBudget() @TypeOf(core.refused_budget) {
    return core.refused_budget;
}
pub const mappedTotal = core.mappedTotal;
pub const touch = core.touch;

/// Custom-kernel launches that built their instance key since the process began.
pub fn kernelKeyBuilds() u64 {
    return kernel.key_builds;
}
pub const budget = core.budget;
pub const room = core.room;
pub const uncapped = core.uncapped;
pub const capped = core.capped;
pub const close = core.close;
pub const kernelProfile = core.kernelProfile;
pub const resetKernelProfile = core.resetKernelProfile;
pub const logKernelProfile = core.logKernelProfile;

/// Names what the host builds from here (0 other, 1 a forward, 2 a draft, 3 a verify, 4 a round's end): the debug
/// line charges each GPU pause to the phase whose pass ended it.
pub fn phase(now: u8) void {
    core.phase = now;
}
pub const trim = core.trim;

pub const nowNs = core.nowNs;

/// A model's load has run every forward a request can (true), or a load begins (false).
pub fn seal(on: bool) void {
    core.sealed = on;
}

pub fn sealed() bool {
    return core.sealed;
}

/// Pipelines made while sealed: each was a stall inside a request.
pub const matrixUnits = @import("matmul.zig").matrixUnits;

pub const pipeline_cache = @import("pipeline_cache.zig");
pub const stock_kernels = stock;
pub const pipeline_registry = @import("pipeline_registry.zig");
pub const metallibPath = stock.metallibPath;

/// Pipelines launches took from those a manifest made ahead.
pub fn preloadedPipelines() u64 {
    return core.pipelines_preloaded;
}

pub fn latePipelines() u64 {
    return core.pipelines_late;
}

/// Pipelines the engine has made in this process and the time they took.
pub fn pipelinesMade() struct { count: u64, ns: u64 } {
    return .{ .count = core.pipelines, .ns = core.pipeline_ns };
}
pub const isStream = core.isStream;

// ── Vectors ──

/// A vector of array handles, ours or MLX's, each a reference the vector owns.
const Vec = struct { items: std.ArrayList(A) = .empty };

pub fn isVec(v: V) bool {
    return @intFromPtr(v.ctx) & 1 != 0;
}

fn vecOf(v: V) *Vec {
    return @ptrFromInt(@intFromPtr(v.ctx) & ~@as(usize, 1));
}

fn newVec() ?*Vec {
    const v = gpa.create(Vec) catch return null;
    v.* = .{};
    return v;
}

fn vecHandle(v: *Vec) V {
    return .{ .ctx = @ptrFromInt(@intFromPtr(v) | 1) };
}

fn vecAppend(v: *Vec, a: A) bool {
    var held: A = .{};
    if (mlx_array_set(&held, a) != 0) return false;
    v.items.append(gpa, held) catch {
        _ = mlx_array_free(held);
        return false;
    };
    return true;
}

pub fn mlx_vector_array_new() V {
    return vecHandle(newVec() orelse return .{});
}

// ── Maps: names to arrays and names to strings, in the order they were put ──

const MA = mlx.mlx_map_string_to_array;
const MS = mlx.mlx_map_string_to_string;
const MI = mlx.mlx_map_string_to_array_iterator;

/// mlx-c's answer for a name a map does not hold, and for an iterator at its end: no error is latched.
const absent: c_int = 2;

fn tagged(ctx: ?*anyopaque) bool {
    return @intFromPtr(ctx) & 1 != 0;
}

/// Whether a map handle's `ctx` is a map of ours.
pub const isMap = tagged;

fn Map(comptime Value: type) type {
    return struct {
        names: std.ArrayList([:0]u8) = .empty,
        values: std.ArrayList(Value) = .empty,

        const Self = @This();

        fn of(ctx: ?*anyopaque) *Self {
            return @ptrFromInt(@intFromPtr(ctx) & ~@as(usize, 1));
        }

        fn create() ?*anyopaque {
            const m = gpa.create(Self) catch return null;
            m.* = .{};
            return @ptrFromInt(@intFromPtr(m) | 1);
        }

        fn find(self: *const Self, key: [*:0]const u8) ?usize {
            const want = std.mem.span(key);
            for (self.names.items, 0..) |name, i| if (std.mem.eql(u8, name, want)) return i;
            return null;
        }

        /// Takes `value`. A name already held gets it, and its old value is returned for the caller to drop.
        fn put(self: *Self, key: [*:0]const u8, value: Value) error{OutOfMemory}!?Value {
            if (self.find(key)) |i| {
                const old = self.values.items[i];
                self.values.items[i] = value;
                return old;
            }
            const name = try gpa.dupeSentinel(u8, std.mem.span(key), 0);
            errdefer gpa.free(name);
            try self.names.append(gpa, name);
            errdefer _ = self.names.pop();
            try self.values.append(gpa, value);
            return null;
        }

        fn destroy(self: *Self) void {
            for (self.names.items) |name| gpa.free(name);
            self.names.deinit(gpa);
            self.values.deinit(gpa);
            gpa.destroy(self);
        }
    };
}

const ArrayMap = Map(A);
const StringMap = Map([:0]u8);

pub fn mlx_map_string_to_array_new() MA {
    return .{ .ctx = ArrayMap.create() };
}

pub fn mlx_map_string_to_array_free(map: MA) c_int {
    const m = ArrayMap.of(map.ctx);
    for (m.values.items) |a| _ = mlx_array_free(a);
    m.destroy();
    return 0;
}

pub fn mlx_map_string_to_array_insert(map: MA, key: [*:0]const u8, value: A) c_int {
    var held: A = .{};
    if (mlx_array_set(&held, value) != 0) return 1;
    const old = ArrayMap.of(map.ctx).put(key, held) catch {
        _ = mlx_array_free(held);
        return fail("map_string_to_array_insert", error.OutOfMemory);
    };
    if (old) |a| _ = mlx_array_free(a);
    return 0;
}

pub fn mlx_map_string_to_array_get(value: *A, map: MA, key: [*:0]const u8) c_int {
    const m = ArrayMap.of(map.ctx);
    return mlx_array_set(value, m.values.items[m.find(key) orelse return absent]);
}

const MapCursor = struct { map: *ArrayMap, at: usize = 0 };

pub fn mlx_map_string_to_array_iterator_new(map: MA) MI {
    const c = gpa.create(MapCursor) catch return .{};
    c.* = .{ .map = ArrayMap.of(map.ctx) };
    return .{ .ctx = @ptrFromInt(@intFromPtr(c) | 1), .map_ctx = map.ctx };
}

pub fn mlx_map_string_to_array_iterator_free(it: MI) c_int {
    gpa.destroy(@as(*MapCursor, @ptrFromInt(@intFromPtr(it.ctx) & ~@as(usize, 1))));
    return 0;
}

pub fn mlx_map_string_to_array_iterator_next(key: *?[*:0]const u8, value: *A, it: MI) c_int {
    const c: *MapCursor = @ptrFromInt(@intFromPtr(it.ctx) & ~@as(usize, 1));
    if (c.at >= c.map.names.items.len) return absent;
    key.* = c.map.names.items[c.at].ptr;
    const rc = mlx_array_set(value, c.map.values.items[c.at]);
    c.at += 1;
    return rc;
}

pub fn mlx_map_string_to_string_new() MS {
    return .{ .ctx = StringMap.create() };
}

pub fn mlx_map_string_to_string_free(map: MS) c_int {
    const m = StringMap.of(map.ctx);
    for (m.values.items) |v| gpa.free(v);
    m.destroy();
    return 0;
}

pub fn mlx_map_string_to_string_insert(map: MS, key: [*:0]const u8, value: [*:0]const u8) c_int {
    const held = gpa.dupeSentinel(u8, std.mem.span(value), 0) catch return fail("map_string_to_string_insert", error.OutOfMemory);
    const old = StringMap.of(map.ctx).put(key, held) catch {
        gpa.free(held);
        return fail("map_string_to_string_insert", error.OutOfMemory);
    };
    if (old) |v| gpa.free(v);
    return 0;
}

/// The string stays the map's: it lives until the name is put again or the map is freed.
pub fn mlx_map_string_to_string_get(value: *[*:0]const u8, map: MS, key: [*:0]const u8) c_int {
    const m = StringMap.of(map.ctx);
    value.* = m.values.items[m.find(key) orelse return absent].ptr;
    return 0;
}

pub fn mlx_vector_array_new_data(items: [*]const A, n: usize) V {
    const v = newVec() orelse return .{};
    for (items[0..n]) |a| {
        if (!vecAppend(v, a)) {
            _ = mlx_vector_array_free(vecHandle(v));
            return .{};
        }
    }
    return vecHandle(v);
}

pub fn mlx_vector_array_new_value(a: A) V {
    return mlx_vector_array_new_data(@ptrCast(&a), 1);
}

pub fn mlx_vector_array_free(v: V) c_int {
    const vec = vecOf(v);
    for (vec.items.items) |a| _ = mlx_array_free(a);
    vec.items.deinit(gpa);
    gpa.destroy(vec);
    return 0;
}

pub fn mlx_vector_array_size(v: V) usize {
    return vecOf(v).items.items.len;
}

pub fn mlx_vector_array_get(res: *A, v: V, idx: usize) c_int {
    const items = vecOf(v).items.items;
    if (idx >= items.len) return fail("vector_array_get", error.TensorShape);
    return mlx_array_set(res, items[idx]);
}

pub fn mlx_vector_array_append_value(v: V, a: A) c_int {
    return if (vecAppend(vecOf(v), a)) 0 else fail("vector_array_append", error.OutOfMemory);
}

/// The arrays of a vector of either kind as tensors the caller releases.
fn takeAll(v: V, out: []*Tensor) core.Error![]*Tensor {
    var n: usize = 0;
    errdefer for (out[0..n]) |t| core.release(t);
    if (isVec(v)) {
        const items = vecOf(v).items.items;
        if (items.len > out.len) return error.TensorShape;
        for (items) |a| {
            out[n] = try core.take(a);
            n += 1;
        }
    } else {
        const count = mlx.raw.mlx_vector_array_size(v);
        if (count > out.len) return error.TensorShape;
        for (0..count) |i| {
            var a = mlx.raw.mlx_array_new();
            defer _ = mlx.raw.mlx_array_free(a);
            try mlx.check(mlx.raw.mlx_vector_array_get(&a, v, i));
            out[n] = try core.import(a);
            n += 1;
        }
    }
    return out[0..n];
}

// ── Evaluation ──

pub fn mlx_eval(v: V) c_int {
    // An MLX array in the vector is MLX's to evaluate; ours are there once the engine has run.
    if (isVec(v)) {
        for (vecOf(v).items.items) |a| {
            if (core.isOurs(a)) {
                _ = core.ready(a) catch |e| return fail("eval", e);
            } else if (mlx.raw.mlx_array_eval(a) != 0) return 1;
        }
    }
    core.sync() catch |e| return fail("eval", e);
    return 0;
}

/// Hands what is encoded to the GPU. An MLX array in the vector goes to MLX's queue.
pub fn mlx_async_eval(v: V) c_int {
    if (isVec(v)) {
        var theirs: [64]A = undefined;
        var n: usize = 0;
        for (vecOf(v).items.items) |a| {
            if (core.isOurs(a)) {
                _ = core.ready(a) catch |e| return fail("async_eval", e);
            } else {
                if (n == theirs.len) return fail("async_eval", error.TensorShape);
                theirs[n] = a;
                n += 1;
            }
        }
        if (n > 0) {
            const raw_vec = mlx.raw.mlx_vector_array_new_data(&theirs, n);
            defer _ = mlx.raw.mlx_vector_array_free(raw_vec);
            if (mlx.raw.mlx_async_eval(raw_vec) != 0) return 1;
        }
    }
    core.submit() catch |e| return fail("async_eval", e);
    return 0;
}

pub fn mlx_synchronize(s: S) c_int {
    _ = s;
    core.sync() catch |e| return fail("synchronize", e);
    return 0;
}

pub fn mlx_stream_free(s: S) c_int {
    _ = s;
    return 0;
}

// ── Views ──

pub fn mlx_reshape(res: *A, a: A, dims: [*]const c_int, n: usize, _: S) c_int {
    return op1("reshape", res, a, ops.reshape, .{dims[0..n]});
}

pub fn mlx_transpose_axes(res: *A, a: A, axes: [*]const c_int, n: usize, _: S) c_int {
    return op1("transpose", res, a, ops.transpose, .{axes[0..n]});
}

pub fn mlx_transpose(res: *A, a: A, _: S) c_int {
    const x = core.take(a) catch |e| return fail("transpose", e);
    defer core.release(x);
    var axes: [core.max_dims]c_int = undefined;
    for (0..x.ndim) |i| axes[i] = @intCast(x.ndim - 1 - i);
    return put(res, ops.transpose(x, axes[0..x.ndim]) catch |e| return fail("transpose", e));
}

pub fn mlx_slice(res: *A, a: A, start: [*]const c_int, n_start: usize, stop: [*]const c_int, n_stop: usize, step: [*]const c_int, n_step: usize, _: S) c_int {
    return op1("slice", res, a, ops.slice, .{ start[0..n_start], stop[0..n_stop], step[0..n_step] });
}

pub fn mlx_squeeze(res: *A, a: A, _: S) c_int {
    return op1("squeeze", res, a, ops.squeeze, .{@as(?[]const c_int, null)});
}

pub fn mlx_expand_dims(res: *A, a: A, axis: c_int, _: S) c_int {
    return op1("expand_dims", res, a, ops.expandDims, .{axis});
}

pub fn mlx_broadcast_to(res: *A, a: A, dims: [*]const c_int, n: usize, _: S) c_int {
    return op1("broadcast_to", res, a, ops.broadcastTo, .{dims[0..n]});
}

pub fn mlx_astype(res: *A, a: A, dtype: mlx.mlx_dtype, _: S) c_int {
    return op1("astype", res, a, ops.astype, .{dtype});
}

pub fn mlx_view(res: *A, a: A, dtype: mlx.mlx_dtype, _: S) c_int {
    return op1("view", res, a, ops.reinterpret, .{dtype});
}

/// MLX's `copy` primitive shares its input's buffer (`Copy::eval_gpu` is `copy_shared_buffer`): a share, not a pass.
pub fn mlx_copy(res: *A, a: A, _: S) c_int {
    return op1("copy", res, a, core.share, .{});
}

/// A copy of `a` in a buffer of its own, where `mlx_copy` gives a share of `a`'s.
pub fn copyOwned(res: *A, a: A) c_int {
    return op1("copy_owned", res, a, ops.contiguousCopy, .{});
}

pub fn mlx_contiguous(res: *A, a: A, _: bool, _: S) c_int {
    return op1("contiguous", res, a, ops.contiguous, .{});
}

// ── Elementwise, reductions, indexing ──

fn bin(comptime op: ops.Binary) fn (*A, A, A, S) c_int {
    return struct {
        fn f(res: *A, a: A, b: A, _: S) c_int {
            const x = core.take(a) catch |e| return fail(@tagName(op), e);
            defer core.release(x);
            const y = core.take(b) catch |e| return fail(@tagName(op), e);
            defer core.release(y);
            return put(res, ops.binary(op, x, y) catch |e| return fail(@tagName(op), e));
        }
    }.f;
}

fn un(comptime op: ops.Unary) fn (*A, A, S) c_int {
    return struct {
        fn f(res: *A, a: A, _: S) c_int {
            const x = core.take(a) catch |e| return fail(@tagName(op), e);
            defer core.release(x);
            return put(res, ops.unary(op, x) catch |e| return fail(@tagName(op), e));
        }
    }.f;
}

pub const mlx_add = bin(.add);
pub const mlx_subtract = bin(.subtract);
pub const mlx_multiply = bin(.multiply);
pub const mlx_divide = bin(.divide);
pub const mlx_maximum = bin(.maximum);
pub const mlx_minimum = bin(.minimum);
pub const mlx_less = bin(.less);
pub const mlx_less_equal = bin(.less_equal);
pub const mlx_greater = bin(.greater);
pub const mlx_greater_equal = bin(.greater_equal);
pub const mlx_equal = bin(.equal);
pub const mlx_logical_and = bin(.logical_and);
pub const mlx_logaddexp = bin(.logaddexp);
pub const mlx_exp = un(.exp);
pub const mlx_log = un(.log);
pub const mlx_abs = un(.abs);
pub const mlx_sigmoid = un(.sigmoid);
pub const mlx_sqrt = un(.sqrt);
pub const mlx_rsqrt = un(.rsqrt);
pub const mlx_square = un(.square);
pub const mlx_floor = un(.floor);
pub const mlx_negative = un(.negative);

/// MLX's integer floor divide of unsigned operands is their quotient (the remainder step cannot fire); of floats it is
/// the floor of the quotient.
pub fn mlx_floor_divide(res: *A, a: A, b: A, _: S) c_int {
    return op2("floor_divide", res, a, b, floorDivideOf, .{});
}

fn floorDivideOf(x: *Tensor, y: *Tensor) core.Error!*Tensor {
    switch (x.dtype) {
        .uint8, .uint16, .uint32, .uint64 => return ops.binary(.divide, x, y),
        .float32, .float16, .bfloat16 => {
            const q = try ops.binary(.divide, x, y);
            defer core.release(q);
            return ops.unary(.floor, q);
        },
        else => return error.TensorDtype,
    }
}

pub fn mlx_where(res: *A, cond: A, a: A, b: A, _: S) c_int {
    const c = core.take(cond) catch |e| return fail("where", e);
    defer core.release(c);
    return op2("where", res, a, b, whereOf, .{c});
}

fn whereOf(x: *Tensor, y: *Tensor, c: *Tensor) core.Error!*Tensor {
    return ops.where(c, x, y);
}

pub fn mlx_sum_axis(res: *A, a: A, axis: c_int, keepdims: bool, _: S) c_int {
    return op1("sum", res, a, sumOf, .{ axis, keepdims });
}

fn sumOf(x: *Tensor, axis: c_int, keepdims: bool) core.Error!*Tensor {
    return ops.reduce(.sum, x, axis, keepdims);
}

pub fn mlx_max_axis(res: *A, a: A, axis: c_int, keepdims: bool, _: S) c_int {
    return op1("max", res, a, maxOf, .{ axis, keepdims });
}

fn maxOf(x: *Tensor, axis: c_int, keepdims: bool) core.Error!*Tensor {
    return ops.reduce(.max, x, axis, keepdims);
}

pub fn mlx_argmax_axis(res: *A, a: A, axis: c_int, keepdims: bool, _: S) c_int {
    return op1("argmax", res, a, ops.argmax, .{ axis, keepdims });
}

pub fn mlx_mean_axis(res: *A, a: A, axis: c_int, keepdims: bool, _: S) c_int {
    return op1("mean", res, a, meanOf, .{ axis, keepdims });
}

/// MLX's mean: the sum times `1 / n`, the reciprocal taken in double and stored in the array's dtype.
fn meanOf(x: *Tensor, axis: c_int, keepdims: bool) core.Error!*Tensor {
    if (x.dtype != .float32) return error.TensorUnsupported;
    const at: usize = @intCast(if (axis < 0) axis + @as(c_int, x.ndim) else axis);
    if (at >= x.ndim) return error.TensorShape;
    const sum = try ops.reduce(.sum, x, axis, keepdims);
    defer core.release(sum);
    const inverse: f32 = @floatCast(1.0 / @as(f64, @floatFromInt(x.shape[at])));
    const scale = try core.fromHost(std.mem.asBytes(&inverse), .float32, &.{});
    defer core.release(scale);
    return ops.binary(.multiply, sum, scale);
}

pub fn mlx_softmax_axis(res: *A, a: A, axis: c_int, precise: bool, _: S) c_int {
    return op1("softmax", res, a, softmaxOf, .{ axis, precise });
}

/// MLX's softmax: its fused kernel over the last axis, and `exp(x - max) / sum` over any other.
fn softmaxOf(x: *Tensor, axis: c_int, precise: bool) core.Error!*Tensor {
    if (x.size() == 0) return core.share(x);
    if (x.dtype != .float32 and x.dtype != .float16 and x.dtype != .bfloat16) return error.TensorDtype;
    if (axis == -1 or axis == @as(c_int, x.ndim) - 1) return stock.softmaxLast(x, precise);
    const in = if (precise) try ops.astype(x, .float32) else try core.share(x);
    defer core.release(in);
    const top = try ops.reduce(.max, in, axis, true);
    defer core.release(top);
    const shifted = try ops.binary(.subtract, in, top);
    defer core.release(shifted);
    const ex = try ops.unary(.exp, shifted);
    defer core.release(ex);
    const total = try ops.reduce(.sum, ex, axis, true);
    defer core.release(total);
    const share = try ops.binary(.divide, ex, total);
    defer core.release(share);
    return ops.astype(share, x.dtype);
}

pub fn mlx_matmul(res: *A, a: A, b: A, _: S) c_int {
    return op2("matmul", res, a, b, @import("matmul.zig").matmul, .{});
}

pub fn mlx_fast_rms_norm(res: *A, x: A, weight: A, eps: f32, _: S) c_int {
    return op2("rms_norm", res, x, weight, stock.rmsNorm, .{eps});
}

pub fn mlx_erf(res: *A, a: A, _: S) c_int {
    return op1("erf", res, a, stock.erf, .{});
}

pub fn mlx_quantized_matmul(res: *A, x: A, w: A, scales: A, biases: A, transpose_w: bool, group_size: mlx.mlx_optional_int, bits: mlx.mlx_optional_int, mode: [*:0]const u8, _: S) c_int {
    if (!std.mem.eql(u8, std.mem.span(mode), "affine") or !transpose_w or biases.ctx == null) return fail("quantized_matmul", error.TensorUnsupported);
    const group: usize = if (group_size.has_value) @intCast(group_size.value) else 64;
    const width: usize = if (bits.has_value) @intCast(bits.value) else 4;
    const tx = core.take(x) catch |e| return fail("quantized_matmul", e);
    defer core.release(tx);
    const tw = core.take(w) catch |e| return fail("quantized_matmul", e);
    defer core.release(tw);
    const ts = core.take(scales) catch |e| return fail("quantized_matmul", e);
    defer core.release(ts);
    const tb = core.take(biases) catch |e| return fail("quantized_matmul", e);
    defer core.release(tb);
    return put(res, quantized.quantizedMatmul(tx, tw, ts, tb, group, width) catch |e| return fail("quantized_matmul", e));
}

pub fn mlx_fast_scaled_dot_product_attention(res: *A, queries: A, keys: A, values: A, scale: f32, mask_mode: [*:0]const u8, mask_arr: A, sinks: A, _: bool, _: S) c_int {
    if (mask_mode[0] != 0 or mask_arr.ctx != null or sinks.ctx != null) return fail("scaled_dot_product_attention", error.TensorUnsupported);
    const q = core.take(queries) catch |e| return fail("scaled_dot_product_attention", e);
    defer core.release(q);
    const k = core.take(keys) catch |e| return fail("scaled_dot_product_attention", e);
    defer core.release(k);
    const v = core.take(values) catch |e| return fail("scaled_dot_product_attention", e);
    defer core.release(v);
    return put(res, attention.attention(q, k, v, scale) catch |e| return fail("scaled_dot_product_attention", e));
}

pub fn mlx_dequantize(res: *A, w: A, scales: A, biases: A, group_size: mlx.mlx_optional_int, bits: mlx.mlx_optional_int, mode: [*:0]const u8, global_scale: A, dtype: mlx.mlx_optional_dtype, _: S) c_int {
    const what = std.meta.stringToEnum(stock.QuantMode, std.mem.span(mode)) orelse return fail("dequantize", error.TensorUnsupported);
    const tw = core.take(w) catch |e| return fail("dequantize", e);
    defer core.release(tw);
    const ts = core.take(scales) catch |e| return fail("dequantize", e);
    defer core.release(ts);
    const tb: ?*Tensor = if (biases.ctx != null) core.take(biases) catch |e| return fail("dequantize", e) else null;
    defer if (tb) |t| core.release(t);
    const tg: ?*Tensor = if (global_scale.ctx != null) core.take(global_scale) catch |e| return fail("dequantize", e) else null;
    defer if (tg) |t| core.release(t);
    const group: usize = if (group_size.has_value) @intCast(group_size.value) else switch (what) {
        .affine => 64,
        .nvfp4 => 16,
        .mxfp4, .mxfp8 => 32,
    };
    const width: usize = if (bits.has_value) @intCast(bits.value) else if (what == .mxfp8) 8 else 4;
    const out_type = if (dtype.has_value) dtype.value else if (what == .affine) ts.dtype else .float32;
    return put(res, stock.dequantize(tw, ts, tb, group, width, what, tg, out_type) catch |e| return fail("dequantize", e));
}

/// Affine only: the float modes' quantizers are not the engine's.
pub fn mlx_quantize(res: *V, w: A, group_size: mlx.mlx_optional_int, bits: mlx.mlx_optional_int, mode: [*:0]const u8, global_scale: A, _: S) c_int {
    const what = std.meta.stringToEnum(stock.QuantMode, std.mem.span(mode)) orelse return fail("quantize", error.TensorUnsupported);
    if (what != .affine or global_scale.ctx != null) return fail("quantize", error.TensorUnsupported);
    const tw = core.take(w) catch |e| return fail("quantize", e);
    defer core.release(tw);
    const group: usize = if (group_size.has_value) @intCast(group_size.value) else 64;
    const width: usize = if (bits.has_value) @intCast(bits.value) else 4;
    const parts = stock.quantize(tw, group, width) catch |e| return fail("quantize", e);
    const v = newVec() orelse {
        for (parts) |t| core.release(t);
        return fail("quantize", error.OutOfMemory);
    };
    for (parts) |t| v.items.append(gpa, core.handle(t)) catch {
        core.release(t);
    };
    if (res.ctx != null) _ = if (isVec(res.*)) mlx_vector_array_free(res.*) else mlx.raw.mlx_vector_array_free(res.*);
    res.* = vecHandle(v);
    return 0;
}

pub fn mlx_take_along_axis(res: *A, a: A, indices: A, axis: c_int, _: S) c_int {
    return op2("take_along_axis", res, a, indices, ops.takeAlong, .{axis});
}

pub fn mlx_sort_axis(res: *A, a: A, axis: c_int, _: S) c_int {
    return op1("sort", res, a, sortOf, .{axis});
}

fn sortOf(x: *Tensor, axis: c_int) core.Error!*Tensor {
    return sorting.sort(x, axis, false);
}

pub fn mlx_argsort_axis(res: *A, a: A, axis: c_int, _: S) c_int {
    return op1("argsort", res, a, argsortOf, .{axis});
}

fn argsortOf(x: *Tensor, axis: c_int) core.Error!*Tensor {
    return sorting.sort(x, axis, true);
}

pub fn mlx_cumsum(res: *A, a: A, axis: c_int, reverse: bool, inclusive: bool, _: S) c_int {
    return op1("cumsum", res, a, scan.cumsum, .{ axis, reverse, inclusive });
}

pub fn mlx_put_along_axis(res: *A, a: A, indices: A, values: A, axis: c_int, _: S) c_int {
    const t = core.take(a) catch |e| return fail("put_along_axis", e);
    defer core.release(t);
    const idx = core.take(indices) catch |e| return fail("put_along_axis", e);
    defer core.release(idx);
    const v = core.take(values) catch |e| return fail("put_along_axis", e);
    defer core.release(v);
    return put(res, ops.putAlong(t, idx, v, axis) catch |e| return fail("put_along_axis", e));
}

pub fn mlx_pad(res: *A, a: A, axes: [*]const c_int, n_axes: usize, low: [*]const c_int, n_low: usize, high: [*]const c_int, n_high: usize, pad_value: A, mode: [*:0]const u8, _: S) c_int {
    if (!std.mem.eql(u8, std.mem.span(mode), "constant")) return fail("pad", error.TensorUnsupported);
    return op2("pad", res, a, pad_value, padOf, .{ axes[0..n_axes], low[0..n_low], high[0..n_high] });
}

fn padOf(x: *Tensor, value: *Tensor, axes: []const c_int, low: []const c_int, high: []const c_int) core.Error!*Tensor {
    return ops.pad(x, axes, low, high, value);
}

pub fn mlx_repeat_axis(res: *A, a: A, repeats: c_int, axis: c_int, _: S) c_int {
    return op1("repeat", res, a, ops.repeat, .{ repeats, axis });
}

pub fn mlx_stack_axis(res: *A, arrays: V, axis: c_int, _: S) c_int {
    var buf: [64]*Tensor = undefined;
    const parts = takeAll(arrays, &buf) catch |e| return fail("stack", e);
    defer for (parts) |t| core.release(t);
    return put(res, ops.stack(parts, axis) catch |e| return fail("stack", e));
}

pub fn mlx_slice_update(res: *A, src: A, update: A, start: [*]const c_int, n_start: usize, stop: [*]const c_int, n_stop: usize, step: [*]const c_int, n_step: usize, _: S) c_int {
    for (step[0..n_step]) |s| {
        if (s != 1) return fail("slice_update", error.TensorUnsupported);
    }
    return op2("slice_update", res, src, update, ops.sliceUpdate, .{ start[0..n_start], stop[0..n_stop] });
}

pub fn mlx_concatenate_axis(res: *A, arrays: V, axis: c_int, _: S) c_int {
    var buf: [64]*Tensor = undefined;
    const parts = takeAll(arrays, &buf) catch |e| return fail("concatenate", e);
    defer for (parts) |t| core.release(t);
    return put(res, ops.concatenate(parts, axis) catch |e| return fail("concatenate", e));
}

pub fn mlx_take_axis(res: *A, a: A, indices: A, axis: c_int, _: S) c_int {
    return op2("take", res, a, indices, ops.take, .{axis});
}

pub fn mlx_arange(res: *A, start: f64, stop: f64, step: f64, dtype: mlx.mlx_dtype, _: S) c_int {
    return put(res, ops.arange(start, stop, step, dtype) catch |e| return fail("arange", e));
}

pub fn mlx_zeros(res: *A, dims: [*]const c_int, n: usize, dtype: mlx.mlx_dtype, _: S) c_int {
    return put(res, ops.zeros(dims[0..n], dtype) catch |e| return fail("zeros", e));
}

// ── Random numbers ──

const no_key = "a draw needs a key: the engine keeps no key sequence";

pub fn mlx_random_bits(res: *A, dims: [*]const c_int, n: usize, width: c_int, key: A, _: S) c_int {
    if (key.ctx == null) return refuse("mlx_random_bits", no_key);
    const k = core.take(key) catch |e| return fail("random_bits", e);
    defer core.release(k);
    return put(res, random.bits(dims[0..n], width, k) catch |e| return fail("random_bits", e));
}

pub fn mlx_random_uniform(res: *A, low: A, high: A, dims: [*]const c_int, n: usize, dtype: mlx.mlx_dtype, key: A, _: S) c_int {
    if (key.ctx == null) return refuse("mlx_random_uniform", no_key);
    const k = core.take(key) catch |e| return fail("random_uniform", e);
    defer core.release(k);
    const lo = core.take(low) catch |e| return fail("random_uniform", e);
    defer core.release(lo);
    const hi = core.take(high) catch |e| return fail("random_uniform", e);
    defer core.release(hi);
    return put(res, random.uniform(lo, hi, dims[0..n], dtype, k) catch |e| return fail("random_uniform", e));
}

pub fn mlx_random_categorical(res: *A, logits: A, axis: c_int, key: A, _: S) c_int {
    if (key.ctx == null) return refuse("mlx_random_categorical", no_key);
    const k = core.take(key) catch |e| return fail("random_categorical", e);
    defer core.release(k);
    const x = core.take(logits) catch |e| return fail("random_categorical", e);
    defer core.release(x);
    return put(res, random.categorical(x, axis, k) catch |e| return fail("random_categorical", e));
}

/// A draw from `logits` under an engine key (`random.drawKey`): no MLX array, call or wait is involved.
pub fn categoricalAt(res: *A, logits: A, key: [2]u32) c_int {
    const x = core.take(logits) catch |e| return fail("categorical_at", e);
    defer core.release(x);
    return put(res, random.categoricalAt(x, -1, key) catch |e| return fail("categorical_at", e));
}

// ── Custom kernels ──

/// A kernel, a config or a list of names made while the engine is on is an object of the engine's, its handle the
/// pointer with the low bit as a tensor's is: making and filling one calls nothing in MLX. MLX's own object (the twin,
/// `mlx.zig`) is made when one first launches on MLX's side.
pub fn isKernel(k: mlx.mlx_fast_metal_kernel) bool {
    return @intFromPtr(k.ctx) & 1 != 0;
}

pub fn isConfig(c: mlx.mlx_fast_metal_kernel_config) bool {
    return @intFromPtr(c.ctx) & 1 != 0;
}

pub fn isStrings(v: mlx.mlx_vector_string) bool {
    return @intFromPtr(v.ctx) & 1 != 0;
}

pub fn kernelOf(k: mlx.mlx_fast_metal_kernel) *kernel.Def {
    return @ptrFromInt(@intFromPtr(k.ctx) & ~@as(usize, 1));
}

pub fn configOf(c: mlx.mlx_fast_metal_kernel_config) *kernel.Config {
    return @ptrFromInt(@intFromPtr(c.ctx) & ~@as(usize, 1));
}

const Strings = struct { items: [][:0]u8 };

fn stringsOf(v: mlx.mlx_vector_string) *Strings {
    return @ptrFromInt(@intFromPtr(v.ctx) & ~@as(usize, 1));
}

/// A list of `n` names of the engine's own, copied; a null handle when memory is short.
pub fn stringsNew(texts: ?[*]const [*:0]const u8, n: usize) mlx.mlx_vector_string {
    const v = gpa.create(Strings) catch return .{};
    v.items = gpa.alloc([:0]u8, n) catch {
        gpa.destroy(v);
        return .{};
    };
    for (0..n) |i| v.items[i] = gpa.dupeSentinel(u8, std.mem.span(texts.?[i]), 0) catch {
        for (v.items[0..i]) |held| gpa.free(held);
        gpa.free(v.items);
        gpa.destroy(v);
        return .{};
    };
    return .{ .ctx = @ptrFromInt(@intFromPtr(v) | 1) };
}

pub fn stringsFree(v: mlx.mlx_vector_string) void {
    const held = stringsOf(v);
    for (held.items) |text| gpa.free(text);
    gpa.free(held.items);
    gpa.destroy(held);
}

pub fn stringsItems(v: mlx.mlx_vector_string) []const [:0]u8 {
    return stringsOf(v).items;
}

/// A kernel of the engine's own, as its maker gave it; a null handle with the reason latched when it cannot be held.
pub fn kernelNative(name: [*:0]const u8, input_names: mlx.mlx_vector_string, output_names: mlx.mlx_vector_string, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx.mlx_fast_metal_kernel {
    var ins: [kernel.max_inputs][]const u8 = undefined;
    var outs: [kernel.max_outputs][]const u8 = undefined;
    const n_in = names(input_names, &ins) orelse return noKernel();
    const n_out = names(output_names, &outs) orelse return noKernel();
    const def = kernel.Def.create(std.mem.span(name), ins[0..n_in], outs[0..n_out], std.mem.span(source), std.mem.span(header), ensure_row_contiguous, atomic_outputs) catch return noKernel();
    return .{ .ctx = @ptrFromInt(@intFromPtr(def) | 1) };
}

fn noKernel() mlx.mlx_fast_metal_kernel {
    _ = refuse("mlx_fast_metal_kernel_new", "more inputs or outputs than a kernel of the engine holds, or no memory");
    return .{};
}

/// A config of the engine's own, empty; a null handle when memory is short.
pub fn configNew() mlx.mlx_fast_metal_kernel_config {
    const c = spare_configs.pop() orelse (gpa.create(kernel.Config) catch return .{});
    c.* = .{};
    return .{ .ctx = @ptrFromInt(@intFromPtr(c) | 1) };
}

pub fn configFree(c: mlx.mlx_fast_metal_kernel_config) void {
    const held = configOf(c);
    spare_configs.append(gpa, held) catch gpa.destroy(held);
}

pub fn configOutput(c: mlx.mlx_fast_metal_kernel_config, dims: [*]const c_int, n: usize, dtype: mlx.mlx_dtype) c_int {
    configOf(c).addOutput(dims[0..n], dtype) catch return refuse("mlx_fast_metal_kernel_config_add_output_arg", "more outputs or dimensions than a config of the engine holds");
    return 0;
}

pub fn configTemplate(c: mlx.mlx_fast_metal_kernel_config, name: [*:0]const u8, value: kernel.TemplateValue) c_int {
    configOf(c).addTemplate(std.mem.span(name), value) catch return refuse("mlx_fast_metal_kernel_config_add_template_arg", "more template arguments, or a longer name, than a config of the engine holds");
    return 0;
}

/// The kernels and configs MLX made (the engine was off, or a test turned it on later), by their MLX object's address.
var defs: std.AutoHashMapUnmanaged(usize, *kernel.Def) = .empty;

/// Records a kernel as its maker gave it, under the MLX kernel object's address.
pub fn kernelNew(k: mlx.mlx_fast_metal_kernel, name: [*:0]const u8, input_names: mlx.mlx_vector_string, output_names: mlx.mlx_vector_string, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) void {
    var ins: [kernel.max_inputs][]const u8 = undefined;
    var outs: [kernel.max_outputs][]const u8 = undefined;
    const n_in = names(input_names, &ins) orelse return;
    const n_out = names(output_names, &outs) orelse return;
    const def = kernel.Def.create(std.mem.span(name), ins[0..n_in], outs[0..n_out], std.mem.span(source), std.mem.span(header), ensure_row_contiguous, atomic_outputs) catch return;
    defs.put(gpa, @intFromPtr(k.ctx), def) catch {};
}

fn names(v: mlx.mlx_vector_string, out: [][]const u8) ?usize {
    if (isStrings(v)) {
        const items = stringsItems(v);
        if (items.len > out.len) return null;
        for (items, 0..) |text, i| out[i] = text;
        return items.len;
    }
    const n = mlx.mlx_vector_string_size(v);
    if (n > out.len) return null;
    for (0..n) |i| {
        var text: [*:0]u8 = undefined;
        if (mlx.mlx_vector_string_get(&text, v, i) != 0) return null;
        out[i] = std.mem.span(text);
    }
    return n;
}

pub fn kernelFree(k: mlx.mlx_fast_metal_kernel) void {
    _ = defs.remove(@intFromPtr(k.ctx));
}

var spare_configs: std.ArrayList(*kernel.Config) = .empty;

/// What each MLX config holds, mirrored as it is filled: a module keeps the configs it builds, so a config made on
/// the MLX side launches on the engine through its mirror.
var mirrors: std.AutoHashMapUnmanaged(usize, *kernel.Config) = .empty;

fn mirror(c: mlx.mlx_fast_metal_kernel_config) ?*kernel.Config {
    const slot = mirrors.getOrPut(gpa, @intFromPtr(c.ctx)) catch return null;
    if (!slot.found_existing) {
        slot.value_ptr.* = spare_configs.pop() orelse (gpa.create(kernel.Config) catch {
            _ = mirrors.remove(@intFromPtr(c.ctx));
            return null;
        });
        slot.value_ptr.*.* = .{};
    }
    return slot.value_ptr.*;
}

pub fn noteOutput(c: mlx.mlx_fast_metal_kernel_config, dims: [*]const c_int, n: usize, dtype: mlx.mlx_dtype) void {
    if (mirror(c)) |m| m.addOutput(dims[0..n], dtype) catch {
        m.partial = true;
    };
}

pub fn noteGrid(c: mlx.mlx_fast_metal_kernel_config, x: c_int, y: c_int, z: c_int) void {
    if (mirror(c)) |m| m.grid = .{ x, y, z };
}

pub fn noteGroup(c: mlx.mlx_fast_metal_kernel_config, x: c_int, y: c_int, z: c_int) void {
    if (mirror(c)) |m| m.group = .{ x, y, z };
}

pub fn noteTemplate(c: mlx.mlx_fast_metal_kernel_config, name: [*:0]const u8, value: kernel.TemplateValue) void {
    if (mirror(c)) |m| m.addTemplate(std.mem.span(name), value) catch {
        m.partial = true;
    };
}

pub fn forgetConfig(c: mlx.mlx_fast_metal_kernel_config) void {
    if (mirrors.fetchRemove(@intFromPtr(c.ctx))) |kv| spare_configs.append(gpa, kv.value) catch gpa.destroy(kv.value);
}

pub fn mlx_fast_metal_kernel_apply(outputs: *V, k: mlx.mlx_fast_metal_kernel, inputs: V, c: mlx.mlx_fast_metal_kernel_config, _: S) c_int {
    const def = if (isKernel(k)) kernelOf(k) else defs.get(@intFromPtr(k.ctx)) orelse return fail("kernel_apply: the kernel was made before the engine was on", error.TensorUnsupported);
    const cfg = if (isConfig(c)) configOf(c) else mirrors.get(@intFromPtr(c.ctx)) orelse return fail("kernel_apply: the config was made before the engine was on", error.TensorUnsupported);
    if (cfg.partial) return fail("kernel_apply: the config holds more outputs or template arguments than the engine mirrors", error.TensorUnsupported);
    var in_buf: [32]*Tensor = undefined;
    const ins = takeAll(inputs, &in_buf) catch |e| return fail("kernel_apply", e);
    defer for (ins) |t| core.release(t);
    var out_buf: [kernel.max_outputs]*Tensor = undefined;
    const outs = out_buf[0..cfg.n_out];
    kernel.launch(def, ins, cfg, outs) catch |e| return fail("kernel_apply", e);
    const v = newVec() orelse {
        for (outs) |t| core.release(t);
        return fail("kernel_apply", error.OutOfMemory);
    };
    for (outs) |t| v.items.append(gpa, core.handle(t)) catch {
        core.release(t);
    };
    if (outputs.ctx != null) _ = if (isVec(outputs.*)) mlx_vector_array_free(outputs.*) else mlx.raw.mlx_vector_array_free(outputs.*);
    outputs.* = vecHandle(v);
    return 0;
}

// ── Tests: each op against the MLX op it replaces ──

const t_std = std.testing;

/// Host data as an MLX array.
fn hostArray(comptime T: type, values: []const T, dims: []const c_int, dtype: mlx.mlx_dtype) A {
    return mlx.raw.mlx_array_new_data(values.ptr, dims.ptr, @intCast(dims.len), dtype);
}

test "work built while the GPU is idle is handed over early, in steps that double, and gives the same value" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    try core.sync();
    const before = core.stats();
    const one = [_]f32{1};
    const host = mlx.mlx_array_new_data(&one, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(host);
    var sum: A = .{};
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, host, host, core.stream()));
    for (0..200) |_| {
        var next: A = .{};
        try mlx.check(mlx.mlx_add(&next, sum, host, core.stream()));
        _ = mlx.mlx_array_free(sum);
        sum = next;
    }
    // Nothing waited, yet the GPU has been given passes; far fewer than the dispatches, since each step doubles.
    const kicks = core.stats().kicks - before.kicks;
    try t_std.expect(kicks >= 1 and kicks <= 8);
    try t_std.expectEqual(before.waits, core.stats().waits);
    var value: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&value, sum));
    try t_std.expectEqual(@as(f32, 202), value);
}

test "once a burst's first pass is handed over, the passes after it follow while the GPU still runs it" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    try core.sync();
    const one = [_]f32{1};
    const host = mlx.mlx_array_new_data(&one, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(host);
    var wide: A = .{};
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_zeros(&wide, &[_]c_int{1 << 25}, 1, .float32, core.stream()));
    try core.sync();
    const before = core.stats();
    // Four long dispatches go first, so the GPU is busy while the short ones behind them are built.
    for (0..4) |_| {
        var next: A = .{};
        try mlx.check(mlx.mlx_add(&next, wide, host, core.stream()));
        _ = mlx.mlx_array_free(wide);
        wide = next;
    }
    var sum: A = .{};
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, host, host, core.stream()));
    for (0..200) |_| {
        var next: A = .{};
        try mlx.check(mlx.mlx_add(&next, sum, host, core.stream()));
        _ = mlx.mlx_array_free(sum);
        sum = next;
    }
    try t_std.expect(core.stats().kicks - before.kicks >= 4);
    try t_std.expectEqual(before.waits, core.stats().waits);
    var value: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&value, sum));
    try t_std.expectEqual(@as(f32, 202), value);
}

test "close ends the pass an op of ours left open" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const one = [_]f32{1};
    const host = mlx.mlx_array_new_data(&one, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(host);
    var sum: A = .{};
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, host, host, core.stream()));
    try t_std.expect(!core.idle());
    core.close();
    try t_std.expect(core.idle());
    var first: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&first, sum));
    try t_std.expectEqual(@as(f32, 2), first);
}

/// Runs `build` on MLX and on the engine over the same MLX inputs and holds the two results to one dtype, one shape
/// and the same bytes.
fn expectSame(comptime build: anytype, inputs: []const A) !void {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);

    var results: [2]A = .{ .{}, .{} };
    defer for (results) |r| {
        if (r.ctx != null) _ = mlx.mlx_array_free(r);
    };
    for ([_]S{ gpu, core.stream() }, 0..) |s, which| {
        const built = try build(s, inputs);
        defer _ = mlx.mlx_array_free(built);
        try mlx.check(mlx.mlx_contiguous(&results[which], built, false, s));
        try mlx.check(mlx.mlx_array_eval(results[which]));
    }
    try t_std.expect(!core.isOurs(results[0]));
    try t_std.expect(core.isOurs(results[1]));
    try t_std.expectEqual(mlx.mlx_array_dtype(results[0]), mlx.mlx_array_dtype(results[1]));
    try t_std.expectEqualSlices(c_int, mlx.getShape(results[0]), mlx.getShape(results[1]));
    const n = mlx.mlx_array_size(results[0]) * mlx.mlx_array_itemsize(results[0]);
    try t_std.expectEqualSlices(u8, mlx.mlx_array_data_uint8(results[0]).?[0..n], mlx.mlx_array_data_uint8(results[1]).?[0..n]);
}

fn ramp(comptime T: type, comptime n: usize, scale: f32, shift: f32) [n]T {
    var out: [n]T = undefined;
    var state: u32 = 0x9e3779b9;
    for (&out) |*v| {
        state = state *% 1664525 +% 1013904223;
        const unit = @as(f32, @floatFromInt(state >> 8)) / 16777216.0;
        const x = unit * scale + shift;
        v.* = switch (@typeInfo(T)) {
            .float => @floatCast(x),
            else => @intFromFloat(x),
        };
    }
    return out;
}

test "engine views give MLX's shapes and bytes: reshape, transpose, slice, squeeze, expand, broadcast, casts" {
    const data_f = ramp(f32, 120, 8, -4);
    const x = hostArray(f32, &data_f, &.{ 2, 3, 4, 5 }, .float32);
    defer _ = mlx.mlx_array_free(x);
    const Build = struct {
        fn reshape(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_reshape(&o, in[0], &[_]c_int{ 6, -1, 5 }, 3, s));
            return o;
        }
        fn transposeThenReshape(s: S, in: []const A) !A {
            var turned: A = .{};
            defer _ = mlx.mlx_array_free(turned);
            try mlx.check(mlx.mlx_transpose_axes(&turned, in[0], &[_]c_int{ 3, 1, 0, 2 }, 4, s));
            var o: A = .{};
            try mlx.check(mlx.mlx_reshape(&o, turned, &[_]c_int{ 15, 8 }, 2, s));
            return o;
        }
        fn slice(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_slice(&o, in[0], &[_]c_int{ 1, 0, 1, -4 }, 4, &[_]c_int{ 2, 3, 4, 5 }, 4, &[_]c_int{ 1, 2, 1, 2 }, 4, s));
            return o;
        }
        fn squeezeExpand(s: S, in: []const A) !A {
            var cut: A = .{};
            defer _ = mlx.mlx_array_free(cut);
            try mlx.check(mlx.mlx_slice(&cut, in[0], &[_]c_int{ 0, 1, 0, 0 }, 4, &[_]c_int{ 2, 2, 4, 5 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
            var flat: A = .{};
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_squeeze(&flat, cut, s));
            var o: A = .{};
            try mlx.check(mlx.mlx_expand_dims(&o, flat, 1, s));
            return o;
        }
        fn broadcast(s: S, in: []const A) !A {
            var row: A = .{};
            defer _ = mlx.mlx_array_free(row);
            try mlx.check(mlx.mlx_slice(&row, in[0], &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, 1, 4, 5 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
            var o: A = .{};
            try mlx.check(mlx.mlx_broadcast_to(&o, row, &[_]c_int{ 3, 2, 4, 5 }, 4, s));
            return o;
        }
        fn castToBf16(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_astype(&o, in[0], .bfloat16, s));
            return o;
        }
        fn castToInt(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_astype(&o, in[0], .int32, s));
            return o;
        }
        fn viewBytes(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_view(&o, in[0], .uint16, s));
            return o;
        }
    };
    inline for (.{ Build.reshape, Build.transposeThenReshape, Build.slice, Build.squeezeExpand, Build.broadcast, Build.castToBf16, Build.castToInt, Build.viewBytes }) |build| {
        try expectSame(build, &.{x});
    }
}

test "engine arithmetic, comparisons, where and reductions give MLX's bytes" {
    const data_a = ramp(f32, 60, 6, -3);
    const data_b = ramp(f32, 5, 2, 0.5);
    const a = hostArray(f32, &data_a, &.{ 3, 4, 5 }, .float32);
    defer _ = mlx.mlx_array_free(a);
    const b = hostArray(f32, &data_b, &.{5}, .float32);
    defer _ = mlx.mlx_array_free(b);
    const Build = struct {
        fn of(comptime f: anytype) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    var o: A = .{};
                    try mlx.check(f(&o, in[0], in[1], s));
                    return o;
                }
            }.build;
        }
        fn of1(comptime f: anytype) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    var o: A = .{};
                    try mlx.check(f(&o, in[0], s));
                    return o;
                }
            }.build;
        }
        fn pick(s: S, in: []const A) !A {
            var cond: A = .{};
            defer _ = mlx.mlx_array_free(cond);
            try mlx.check(mlx.mlx_less(&cond, in[0], in[1], s));
            var o: A = .{};
            try mlx.check(mlx.mlx_where(&o, cond, in[0], in[1], s));
            return o;
        }
        fn sumMid(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_sum_axis(&o, in[0], 1, true, s));
            return o;
        }
        fn maxLast(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_max_axis(&o, in[0], -1, false, s));
            return o;
        }
    };
    inline for (.{ mlx.mlx_add, mlx.mlx_subtract, mlx.mlx_multiply, mlx.mlx_divide, mlx.mlx_maximum, mlx.mlx_minimum, mlx.mlx_less, mlx.mlx_greater_equal, mlx.mlx_equal }) |f| {
        try expectSame(Build.of(f), &.{ a, b });
    }
    inline for (.{ mlx.mlx_exp, mlx.mlx_sigmoid, mlx.mlx_square, mlx.mlx_floor }) |f| {
        try expectSame(Build.of1(f), &.{a});
    }
    inline for (.{ mlx.mlx_sqrt, mlx.mlx_rsqrt }) |f| {
        try expectSame(Build.of1(f), &.{b});
    }
    try expectSame(Build.pick, &.{ a, b });
    try expectSame(Build.sumMid, &.{a});
    try expectSame(Build.maxLast, &.{a});
}

test "a round's verdict on the engine is MLX's: each row's argmax over a vocabulary with masked ids" {
    try mlx.requireGpu();
    const Build = struct {
        fn build(s: S, in: []const A) anyerror!A {
            const ninf = mlx.mlx_array_new_float(-std.math.inf(f32));
            defer _ = mlx.mlx_array_free(ninf);
            var masked: A = .{};
            defer _ = mlx.mlx_array_free(masked);
            try mlx.check(mlx.mlx_where(&masked, in[1], ninf, in[0], s));
            var o: A = .{};
            try mlx.check(mlx.mlx_argmax_axis(&o, masked, 2, false, s));
            return o;
        }
    };
    const vocab = 70000;
    const values = ramp(f32, 5 * vocab, 6, -3);
    // The largest value of every row is masked, so the verdict is not the plain argmax.
    var mask: [vocab]bool = @splat(false);
    for (0..5) |r| mask[std.mem.indexOfMax(f32, values[r * vocab ..][0..vocab])] = true;
    const logits = hostArray(f32, &values, &.{ 1, 5, vocab }, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const flags = hostArray(bool, &mask, &.{vocab}, .bool_);
    defer _ = mlx.mlx_array_free(flags);
    try expectSame(Build.build, &.{ logits, flags });
}

test "engine sum, max, mean and argmax give MLX's bytes under each of MLX's reduction plans" {
    try mlx.requireGpu();
    const Build = struct {
        fn over(comptime f: anytype, comptime axis: c_int, comptime keep: bool) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    var o: A = .{};
                    try mlx.check(f(&o, in[0], axis, keep, s));
                    return o;
                }
            }.build;
        }
        /// The reduction over the first axis of the transposed input: memory does not run along any axis order.
        fn turned(comptime f: anytype) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    var t: A = .{};
                    defer _ = mlx.mlx_array_free(t);
                    try mlx.check(mlx.mlx_transpose(&t, in[0], s));
                    var o: A = .{};
                    try mlx.check(f(&o, t, 0, false, s));
                    return o;
                }
            }.build;
        }
    };
    const values = ramp(f32, 5 * 70000, 6, -3);
    // { shape, axis }: a short row, a row of many outputs, a long row, a short column, a long thin column, a
    // column in two passes, a column in one, a whole vector short and long, an inner axis of three.
    const Case = struct { shape: []const c_int, axis: c_int };
    const cases = [_]Case{
        .{ .shape = &.{ 10, 40 }, .axis = -1 },
        .{ .shape = &.{ 64, 200 }, .axis = -1 },
        .{ .shape = &.{ 4, 4096 }, .axis = -1 },
        .{ .shape = &.{ 5, 20, 7 }, .axis = 1 },
        .{ .shape = &.{ 2000, 8 }, .axis = 0 },
        .{ .shape = &.{ 3, 700, 9 }, .axis = 1 },
        .{ .shape = &.{ 100, 64 }, .axis = 0 },
        .{ .shape = &.{300}, .axis = 0 },
        .{ .shape = &.{5000}, .axis = 0 },
        .{ .shape = &.{ 6, 3, 50 }, .axis = 1 },
    };
    inline for (cases) |case| {
        var n: usize = 1;
        for (case.shape) |d| n *= @intCast(d);
        const x32 = hostArray(f32, values[0..n], case.shape, .float32);
        defer _ = mlx.mlx_array_free(x32);
        const x16 = try castByMlx(x32, .bfloat16);
        defer _ = mlx.mlx_array_free(x16);
        inline for (.{ mlx.mlx_sum_axis, mlx.mlx_max_axis, mlx.mlx_argmax_axis }) |f| {
            try expectSame(Build.over(f, case.axis, true), &.{x32});
            try expectSame(Build.over(f, case.axis, false), &.{x16});
            if (case.shape.len == 2) try expectSame(Build.turned(f), &.{x32});
        }
        try expectSame(Build.over(mlx.mlx_mean_axis, case.axis, true), &.{x32});
    }
    // A row per draft over a vocabulary.
    const vocab = hostArray(f32, &values, &.{ 5, 70000 }, .float32);
    defer _ = mlx.mlx_array_free(vocab);
    try expectSame(Build.over(mlx.mlx_argmax_axis, -1, false), &.{vocab});
    try expectSame(Build.over(mlx.mlx_max_axis, -1, false), &.{vocab});
}

test "engine concatenate, take, slice update, arange and zeros give MLX's bytes" {
    const data_a = ramp(u16, 48, 60000, 0);
    const data_b = ramp(u16, 16, 60000, 0);
    const picks = [_]i32{ 2, 0, 5, 5, 1, 3 };
    const a = hostArray(u16, &data_a, &.{ 6, 8 }, .bfloat16);
    defer _ = mlx.mlx_array_free(a);
    const b = hostArray(u16, &data_b, &.{ 2, 8 }, .bfloat16);
    defer _ = mlx.mlx_array_free(b);
    const idx = hostArray(i32, &picks, &.{ 2, 3 }, .int32);
    defer _ = mlx.mlx_array_free(idx);
    const Build = struct {
        fn concat(s: S, in: []const A) !A {
            const v = mlx.mlx_vector_array_new_data(in.ptr, 2);
            defer _ = mlx.mlx_vector_array_free(v);
            var o: A = .{};
            try mlx.check(mlx.mlx_concatenate_axis(&o, v, 0, s));
            return o;
        }
        fn takeRows(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_take_axis(&o, in[0], in[2], 0, s));
            return o;
        }
        fn takeColumns(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_take_axis(&o, in[0], in[2], 1, s));
            return o;
        }
        fn writeRows(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_slice_update(&o, in[0], in[1], &[_]c_int{ 3, 0 }, 2, &[_]c_int{ 5, 8 }, 2, &[_]c_int{ 1, 1 }, 2, s));
            return o;
        }
        fn counting(s: S, _: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_arange(&o, 0, 4, 1, .int32, s));
            return o;
        }
        fn nothing(s: S, _: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_zeros(&o, &[_]c_int{ 3, 7 }, 2, .float16, s));
            return o;
        }
    };
    inline for (.{ Build.concat, Build.takeRows, Build.takeColumns, Build.writeRows, Build.counting, Build.nothing }) |build| {
        try expectSame(build, &.{ a, b, idx });
    }
}

test "a slice update leaves its base as it was" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const s = core.stream();
    const base_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const row = [_]f32{ 9, 9, 9 };
    const base = hostArray(f32, &base_data, &.{ 2, 3 }, .float32);
    defer _ = mlx.mlx_array_free(base);
    const update = hostArray(f32, &row, &.{ 1, 3 }, .float32);
    defer _ = mlx.mlx_array_free(update);
    var ours: A = .{};
    defer _ = mlx.mlx_array_free(ours);
    try mlx.check(mlx.mlx_astype(&ours, base, .float32, s));
    var written: A = .{};
    defer _ = mlx.mlx_array_free(written);
    try mlx.check(mlx.mlx_slice_update(&written, ours, update, &[_]c_int{ 1, 0 }, 2, &[_]c_int{ 2, 3 }, 2, &[_]c_int{ 1, 1 }, 2, s));
    try mlx.check(mlx.mlx_array_eval(written));
    try t_std.expectEqualSlices(f32, &.{ 1, 2, 3, 9, 9, 9 }, mlx.mlx_array_data_float32(written).?[0..6]);
    try t_std.expectEqualSlices(f32, &base_data, mlx.mlx_array_data_float32(ours).?[0..6]);
}

test "a handle of the engine's is freed by the engine after the engine is switched off" {
    try mlx.requireGpu();
    const was = enabled;
    const was_active = active;
    enabled = true;
    active = true;
    const values = [_]f32{ 1, 2, 3 };
    const a = mlx.mlx_array_new_data(&values, &[_]c_int{3}, 1, .float32);
    active = was_active;
    enabled = false;
    defer enabled = was;
    try std.testing.expect(isOurs(a));
    // A cache may hold the handle past the switch; handing it to MLX's free would read memory MLX never made.
    try std.testing.expectEqual(@as(c_int, 0), mlx.mlx_array_free(a));
}

test "a custom kernel launched on the engine writes what it writes under MLX" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const source =
        \\uint i = thread_position_in_grid.x;
        \\out[i] = T(float(x[i]) * float(SCALE) + float(w[i % uint(w_shape[0])]) + bias);
        \\tail[i] = x_ndim + int(w_strides[0]);
    ;
    const names_in = [_][*:0]const u8{ "x", "w", "bias" };
    const names_out = [_][*:0]const u8{ "out", "tail" };
    const iv = mlx.mlx_vector_string_new_data(&names_in, names_in.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&names_out, names_out.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_engine_probe", iv, ov, source, "", true, false);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);

    const xs = ramp(f32, 64, 10, -5);
    const ws = ramp(f32, 16, 3, 0);
    const few = ramp(f32, 4, 3, 0);
    const x = hostArray(f32, &xs, &.{ 8, 8 }, .float32);
    defer _ = mlx.mlx_array_free(x);
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    // A wide weight, a weight under 8 elements (the constant address space) and a transposed input (a copy first).
    for ([_]bool{ false, true }) |short| for ([_]bool{ false, true }) |turned| {
        const w = if (short) hostArray(f32, &few, &.{4}, .float32) else hostArray(f32, &ws, &.{16}, .float32);
        defer _ = mlx.mlx_array_free(w);
        var bytes: [2][2][64 * 4]u8 = undefined;
        for ([_]S{ gpu, core.stream() }, 0..) |s, which| {
            active = which == 1;
            defer active = false;
            var input: A = .{};
            defer _ = mlx.mlx_array_free(input);
            if (turned) try mlx.check(mlx.mlx_transpose(&input, x, s)) else try mlx.check(mlx.mlx_array_set(&input, x));
            const bias = mlx.mlx_array_new_float(0.25);
            defer _ = mlx.mlx_array_free(bias);
            const inputs = mlx.mlx_vector_array_new_data(&[_]A{ input, w, bias }, 3);
            defer _ = mlx.mlx_vector_array_free(inputs);
            const cfg = mlx.mlx_fast_metal_kernel_config_new();
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{64}, 1, .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{64}, 1, .int32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SCALE", 3));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 64, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
            var outs = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(outs);
            try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, k, inputs, cfg, s));
            for (0..2) |o| {
                var y: A = .{};
                defer _ = mlx.mlx_array_free(y);
                try mlx.check(mlx.mlx_vector_array_get(&y, outs, o));
                try mlx.check(mlx.mlx_array_eval(y));
                const n = mlx.mlx_array_size(y) * mlx.mlx_array_itemsize(y);
                @memcpy(bytes[which][o][0..n], mlx.mlx_array_data_uint8(y).?[0..n]);
                if (which == 1) try t_std.expectEqualSlices(u8, bytes[0][o][0..n], bytes[1][o][0..n]);
            }
        }
    };
}

test "a kernel, its name lists and its config made while the engine is on call nothing in mlx-c, and launch on both sides" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const source =
        \\uint i = thread_position_in_grid.x;
        \\out[i] = x[i] * float(SCALE) + float(threads_per_threadgroup.x > 0);
    ;
    const names_in = [_][*:0]const u8{"x"};
    const names_out = [_][*:0]const u8{"out"};
    const before = mlx.calls.load(.monotonic);
    const iv = mlx.mlx_vector_string_new_data(&names_in, 1);
    const ov = mlx.mlx_vector_string_new_data(&names_out, 1);
    try t_std.expectEqual(@as(usize, 1), mlx.mlx_vector_string_size(iv));
    const k = mlx.mlx_fast_metal_kernel_new("sushi_native_probe", iv, ov, source, "", true, false);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    _ = mlx.mlx_vector_string_free(iv);
    _ = mlx.mlx_vector_string_free(ov);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{16}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SCALE", 3));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 16, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 16, 1, 1));
    {
        const passing = mlx.mlx_fast_metal_kernel_config_new();
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(passing, &[_]c_int{4}, 1, .int32));
        _ = mlx.mlx_fast_metal_kernel_config_free(passing);
    }
    try t_std.expectEqual(before, mlx.calls.load(.monotonic));

    const xs = ramp(f32, 16, 4, -3);
    const x = hostArray(f32, &xs, &.{16}, .float32);
    defer _ = mlx.mlx_array_free(x);
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    var want: [16 * 4]u8 = undefined;
    // The engine first, then MLX's side twice: the second after the config changed under its MLX twin.
    for ([_]S{ core.stream(), gpu, gpu }, 0..) |s, round| {
        active = round == 0;
        defer active = false;
        if (round == 2) try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 8, 1, 1));
        const inputs = mlx.mlx_vector_array_new_data(&[_]A{x}, 1);
        defer _ = mlx.mlx_vector_array_free(inputs);
        var outs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, k, inputs, cfg, s));
        var y: A = .{};
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
        try mlx.check(mlx.mlx_array_eval(y));
        const got = mlx.mlx_array_data_uint8(y).?[0..want.len];
        if (round == 0) @memcpy(&want, got) else try t_std.expectEqualSlices(u8, &want, got);
    }
    const first: f32 = @bitCast(want[0..4].*);
    try t_std.expectEqual(xs[0] * 3 + 1, first);
}

test "a config made while the engine is on refuses by name an output it cannot hold" {
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (0..kernel.max_outputs) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{4}, 1, .float32));
    try t_std.expect(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{4}, 1, .float32) != 0);
}

test "a mapped region the engine wraps is bound by a pass without a copy or a call into MLX, and is given back once, after that pass" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const Region = struct {
        var given_back: usize = 0;
        fn release(payload: ?*anyopaque) callconv(.c) void {
            given_back += 1;
            const span: *const [2]usize = @ptrCast(@alignCast(payload.?));
            _ = std.c.munmap(@ptrFromInt(span[0]), span[1]);
        }
    };
    Region.given_back = 0;
    const n = 1000;
    const len = std.mem.alignForward(usize, n * 4, std.heap.pageSize());
    const mapped = std.c.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    try t_std.expect(mapped != std.c.MAP_FAILED);
    const ptr: [*]u8 = @ptrCast(mapped);
    const values: []f32 = @alignCast(std.mem.bytesAsSlice(f32, ptr[0 .. n * 4]));
    for (values, 0..) |*v, i| v.* = @floatFromInt(i);
    var span = [2]usize{ @intFromPtr(ptr), len };
    try core.sync();
    const calls = mlx.calls.load(.monotonic);
    const before = core.stats().mapped_bytes;

    const t = try core.wrapMapped(ptr, .float32, &.{n}, &span, Region.release);
    try t_std.expectEqual(@intFromPtr(ptr), @intFromPtr(t.store.ptr));
    try t_std.expectEqual(before + n * 4, core.stats().mapped_bytes);
    const x = core.handle(t);
    var sum: A = .{};
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, x, x, core.stream()));
    // The pass that binds it has not run: the mapping stays until it has.
    _ = mlx.mlx_array_free(x);
    try t_std.expectEqual(@as(usize, 0), Region.given_back);
    try mlx.check(mlx.mlx_array_eval(sum));
    const got = mlx.mlx_array_data_float32(sum).?[0..n];
    for (got, 0..) |v, i| try t_std.expectEqual(@as(f32, @floatFromInt(2 * i)), v);
    try core.sync();
    try t_std.expectEqual(@as(usize, 1), Region.given_back);
    try t_std.expectEqual(before, core.stats().mapped_bytes);
    try t_std.expectEqual(calls, mlx.calls.load(.monotonic));
    // Its residency set has been told: a set that still lists the buffer would keep the pages wired with no pass to come.
    try t_std.expect(!@import("../metal.zig").residentDirty());
}

test "a pass that binds the landed half of a wrapped slab runs while the other half fills; one that binds the filling half is refused and leaves the pool its buffers" {
    try mlx.requireGpu();
    const fill_marks = @import("../fill_marks.zig");
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const Region = struct {
        fn release(payload: ?*anyopaque) callconv(.c) void {
            const span: *const [2]usize = @ptrCast(@alignCast(payload.?));
            _ = std.c.munmap(@ptrFromInt(span[0]), span[1]);
        }
    };
    const len = 2 * std.heap.pageSize();
    const n: c_int = @intCast(len / 4);
    const mapped = std.c.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    try t_std.expect(mapped != std.c.MAP_FAILED);
    const ptr: [*]u8 = @ptrCast(mapped);
    const values: []f32 = @alignCast(std.mem.bytesAsSlice(f32, ptr[0..len]));
    for (values, 0..) |*v, i| v.* = @floatFromInt(i);
    var span = [2]usize{ @intFromPtr(ptr), len };
    const slab = core.handle(try core.wrapMapped(ptr, .float32, &.{n}, &span, Region.release));
    defer _ = mlx.mlx_array_free(slab);
    const s = core.stream();
    const half = @divExact(n, 2);
    var halves: [2]A = .{ .{}, .{} };
    defer for (halves) |h| {
        _ = mlx.mlx_array_free(h);
    };
    for (&halves, 0..) |*h, i| {
        const from: c_int = @intCast(i * @as(usize, @intCast(half)));
        try mlx.check(mlx.mlx_slice(h, slab, &[_]c_int{from}, 1, &[_]c_int{from + half}, 1, &[_]c_int{1}, 1, s));
    }
    fill_marks.add(@intFromPtr(ptr) + len / 2, len / 2);
    {
        var sum: A = .{};
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, halves[0], halves[0], s));
        try mlx.check(mlx.mlx_array_eval(sum));
        try t_std.expectEqual(@as(f32, 6), mlx.mlx_array_data_float32(sum).?[3]);
    }
    // A later test sees no limit.
    defer core.budget(1 << 62) catch {};
    try core.budget(0);
    const held = core.stats().pool_bytes;
    try core.budget(1 << 62);
    {
        var sum: A = .{};
        defer _ = mlx.mlx_array_free(sum);
        const built = mlx.mlx_add(&sum, halves[1], halves[1], s);
        try t_std.expect(built != 0 or mlx.mlx_array_eval(sum) != 0);
        var said: [256]u8 = undefined;
        try t_std.expect(std.mem.indexOf(u8, mlx.takeError(&said) orelse "", "ReadsFillingSlab") != null);
    }
    // The refused pass never ran, so what it bound waits for no pass: a cap with nothing to spare gets its buffers back.
    try core.budget(0);
    try t_std.expectEqual(held, core.stats().pool_bytes);
    try core.budget(1 << 62);
    fill_marks.remove(@intFromPtr(ptr) + len / 2);
    var sum: A = .{};
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, halves[1], halves[1], s));
    try mlx.check(mlx.mlx_array_eval(sum));
    try t_std.expectEqual(@as(f32, @floatFromInt(2 * (half + 3))), mlx.mlx_array_data_float32(sum).?[3]);
}

test "configs made and freed in turn launch the pipeline of their own template value" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const names_in = [_][*:0]const u8{"x"};
    const names_out = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(&names_in, names_in.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&names_out, names_out.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_config_reuse_probe", iv, ov, "out[thread_position_in_grid.x] = x[thread_position_in_grid.x] + float(K);", "", true, false);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    const xs = ramp(f32, 4, 1, 0);
    const x = hostArray(f32, &xs, &.{4}, .float32);
    defer _ = mlx.mlx_array_free(x);
    const inputs = mlx.mlx_vector_array_new_data(&[_]A{x}, 1);
    defer _ = mlx.mlx_vector_array_free(inputs);
    for ([_]c_int{ 5, 9, 9, 5 }) |value| {
        active = true;
        defer active = false;
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{4}, 1, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", value));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 4, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 4, 1, 1));
        var outs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, k, inputs, cfg, core.stream()));
        var y: A = .{};
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
        try mlx.check(mlx.mlx_array_eval(y));
        const got = @as([*]const f32, @ptrCast(@alignCast(mlx.mlx_array_data_uint8(y).?)))[0..4];
        for (got, xs) |g, want| try t_std.expectEqual(want + @as(f32, @floatFromInt(value)), g);
    }
}

test "an op the engine lacks, and a threadgroup past the pipeline's limit, are refused by name" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const s = core.stream();
    var buf: [512]u8 = undefined;
    _ = mlx.takeError(&buf);
    const a_data = [_]f32{ 1, 2 };
    const b_data = [_]i32{ 1, 2 };
    const a = hostArray(f32, &a_data, &.{2}, .float32);
    defer _ = mlx.mlx_array_free(a);
    const b = hostArray(i32, &b_data, &.{2}, .int32);
    defer _ = mlx.mlx_array_free(b);
    var o: A = .{};
    try t_std.expectError(error.MlxError, mlx.check(mlx.mlx_add(&o, a, b, s)));
    try t_std.expect(std.mem.indexOf(u8, mlx.takeError(&buf).?, "TensorDtype") != null);

    const names_in = [_][*:0]const u8{"x"};
    const names_out = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(&names_in, 1);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&names_out, 1);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_engine_wide", iv, ov, "out[thread_position_in_grid.x] = x[0];", "", true, false);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    active = true;
    defer active = false;
    const inputs = mlx.mlx_vector_array_new_data(&[_]A{a}, 1);
    defer _ = mlx.mlx_vector_array_free(inputs);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{4096}, 1, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 4096, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 4096, 1, 1));
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try t_std.expectError(error.MlxError, mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, k, inputs, cfg, s)));
    try t_std.expect(mlx.takeThreadgroupLimit() != null);
}

test "an MLX op takes a tensor of ours as an array over the same bytes, and the bytes stay until MLX lets go" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    const ours = core.stream();
    // Past one page, so MLX can alias it; a scalar and a strided view go the same road.
    const data_f = ramp(f32, 6000, 8, -4);
    const host = hostArray(f32, &data_f, &.{ 60, 100 }, .float32);
    defer _ = mlx.mlx_array_free(host);
    var doubled: A = .{};
    try mlx.check(mlx.mlx_add(&doubled, host, host, ours));
    try t_std.expect(core.isOurs(doubled));
    const lends = core.stats().lends;
    var turned: A = .{};
    defer _ = mlx.mlx_array_free(turned);
    try mlx.check(mlx.mlx_transpose(&turned, doubled, ours));

    var sum: A = .{};
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, doubled, host, gpu));
    try t_std.expect(!core.isOurs(sum));
    var sum_t: A = .{};
    defer _ = mlx.mlx_array_free(sum_t);
    try mlx.check(mlx.mlx_add(&sum_t, turned, turned, gpu));
    try t_std.expectEqual(lends + 2, core.stats().lends);
    // The array MLX holds is the tensor's own bytes.
    const store = core.of(doubled).store;
    try t_std.expectEqual(@intFromPtr(store.ptr), @intFromPtr(mlx.mlx_array_data_uint8(core.of(doubled).twin).?));
    // A second MLX op reuses the array.
    var again: A = .{};
    defer _ = mlx.mlx_array_free(again);
    try mlx.check(mlx.mlx_multiply(&again, doubled, doubled, gpu));
    try t_std.expectEqual(lends + 2, core.stats().lends);

    // Dropped by us while MLX's graph still reads it: the bytes are not reused before MLX has evaluated.
    _ = mlx.mlx_array_free(doubled);
    const reused = try core.fresh(.float32, &.{ 60, 100 });
    defer core.release(reused);
    try t_std.expect(reused.store != store);
    try mlx.check(mlx.mlx_array_eval(sum));
    try mlx.check(mlx.mlx_array_eval(sum_t));
    try mlx.check(mlx.mlx_array_eval(again));
    const got = mlx.mlx_array_data_float32(sum).?[0..6000];
    const got_t = mlx.mlx_array_data_float32(sum_t).?[0..6000];
    const got_sq = mlx.mlx_array_data_float32(again).?[0..6000];
    for (data_f, 0..) |x, i| {
        try t_std.expectEqual(x + x + x, got[i]);
        try t_std.expectEqual((x + x) * (x + x), got_sq[i]);
        try t_std.expectEqual((x + x) + (x + x), got_t[(i % 100) * 60 + i / 100]);
    }
}

test "a slice update goes into its base's bytes once the base is dropped, as a KV append does" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const s = core.stream();
    const rows = ramp(f32, 12, 8, -4);
    const host = hostArray(f32, &rows, &.{ 4, 3 }, .float32);
    defer _ = mlx.mlx_array_free(host);
    var buf: A = .{};
    defer _ = mlx.mlx_array_free(buf);
    try mlx.check(mlx.mlx_zeros(&buf, &[_]c_int{ 4, 3 }, 2, .float32, s));
    const before = core.stats();
    const first = core.of(buf).store;
    for (0..4) |r| {
        const at: c_int = @intCast(r);
        var row: A = .{};
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, host, &[_]c_int{ at, 0 }, 2, &[_]c_int{ at + 1, 3 }, 2, &[_]c_int{ 1, 1 }, 2, s));
        var up: A = .{};
        try mlx.check(mlx.mlx_slice_update(&up, buf, row, &[_]c_int{ at, 0 }, 2, &[_]c_int{ at + 1, 3 }, 2, &[_]c_int{ 1, 1 }, 2, s));
        _ = mlx.mlx_array_free(buf);
        buf = up;
    }
    try mlx.check(mlx.mlx_array_eval(buf));
    try t_std.expectEqualSlices(f32, &rows, mlx.mlx_array_data_float32(buf).?[0..12]);
    try t_std.expectEqual(first, core.of(buf).store);
    try t_std.expectEqual(before.updates_in_place + 4, core.stats().updates_in_place);
    try t_std.expectEqual(before.updates_copied, core.stats().updates_copied);
}

test "an append to bytes MLX made goes into them once MLX's other holders are gone, and MLX gets its own array back" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    const s = core.stream();
    const rows = ramp(f32, 12, 8, -4);
    const host = hostArray(f32, &rows, &.{ 4, 3 }, .float32);
    defer _ = mlx.mlx_array_free(host);
    for ([_]bool{ false, true }) |held| {
        var buf: A = .{};
        defer _ = mlx.mlx_array_free(buf);
        // MLX's zeros is one element broadcast; a KV buffer is real bytes.
        var spread: A = .{};
        defer _ = mlx.mlx_array_free(spread);
        try mlx.check(mlx.mlx_zeros(&spread, &[_]c_int{ 4, 3 }, 2, .float32, gpu));
        try mlx.check(mlx.mlx_contiguous(&buf, spread, false, gpu));
        try mlx.check(mlx.mlx_array_eval(buf));
        _ = mlx.mlx_array_free(spread);
        spread = .{};
        const made = mlx.mlx_array_data_float32(buf).?;
        // MLX's command buffer holds the bytes a moment past the eval that waited for it.
        var let_go = false;
        var spins: usize = 0;
        while (!let_go and spins < 1 << 24) : (spins += 1) try mlx.check(mlx._mlx_array_is_donatable(&let_go, buf));
        var other: A = .{};
        defer _ = mlx.mlx_array_free(other);
        if (held) try mlx.check(mlx.mlx_array_set(&other, buf));
        const before = core.stats();
        for (0..4) |r| {
            const at: c_int = @intCast(r);
            var row: A = .{};
            defer _ = mlx.mlx_array_free(row);
            try mlx.check(mlx.mlx_slice(&row, host, &[_]c_int{ at, 0 }, 2, &[_]c_int{ at + 1, 3 }, 2, &[_]c_int{ 1, 1 }, 2, s));
            var up: A = .{};
            try mlx.check(mlx.mlx_slice_update(&up, buf, row, &[_]c_int{ at, 0 }, 2, &[_]c_int{ at + 1, 3 }, 2, &[_]c_int{ 1, 1 }, 2, s));
            _ = mlx.mlx_array_free(buf);
            buf = up;
        }
        try mlx.check(mlx.mlx_array_eval(buf));
        const after = core.stats();
        try t_std.expectEqualSlices(f32, &rows, mlx.mlx_array_data_float32(buf).?[0..12]);
        if (held) {
            // The first append may not touch what `other` still reads; the copy is ours alone from then on.
            try t_std.expectEqual(before.updates_copied + 1, after.updates_copied);
            try t_std.expectEqual(before.updates_in_place + 3, after.updates_in_place);
            for (mlx.mlx_array_data_float32(other).?[0..12]) |v| try t_std.expectEqual(@as(f32, 0), v);
            continue;
        }
        try t_std.expectEqual(before.updates_copied, after.updates_copied);
        try t_std.expectEqual(before.updates_in_place + 4, after.updates_in_place);
        try t_std.expectEqual(made, mlx.mlx_array_data_float32(buf).?);
        // Back on MLX's queue the state is MLX's own array again: no second array over memory MLX holds.
        var sum: A = .{};
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, buf, buf, gpu));
        try mlx.check(mlx.mlx_array_eval(sum));
        try t_std.expectEqual(after.lends, core.stats().lends);
        for (rows, mlx.mlx_array_data_float32(sum).?[0..12]) |want, got| try t_std.expectEqual(want + want, got);
    }
}

test "a slice update of a broadcast base writes one row, whoever holds the bytes" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    const half = [_]f32{ 0.5, 1, 1.5 };
    const new = [_]f32{ 7, 8, 9 };
    const want = [_]f32{ 1, 2, 3, 7, 8, 9, 1, 2, 3, 1, 2, 3 };
    const row = hostArray(f32, &new, &.{ 1, 3 }, .float32);
    defer _ = mlx.mlx_array_free(row);
    const s = core.stream();
    // The one row is the engine's own, then MLX's: either way the update may not land on every row.
    for ([_]S{ s, gpu }) |maker| {
        const host = hostArray(f32, &half, &.{ 1, 3 }, .float32);
        defer _ = mlx.mlx_array_free(host);
        var base: A = .{};
        try mlx.check(mlx.mlx_add(&base, host, host, maker));
        var spread: A = .{};
        try mlx.check(mlx.mlx_broadcast_to(&spread, base, &[_]c_int{ 4, 3 }, 2, maker));
        try mlx.check(mlx.mlx_array_eval(spread));
        _ = mlx.mlx_array_free(base);
        var up: A = .{};
        defer _ = mlx.mlx_array_free(up);
        try mlx.check(mlx.mlx_slice_update(&up, spread, row, &[_]c_int{ 1, 0 }, 2, &[_]c_int{ 2, 3 }, 2, &[_]c_int{ 1, 1 }, 2, s));
        _ = mlx.mlx_array_free(spread);
        var flat: A = .{};
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_contiguous(&flat, up, false, s));
        try mlx.check(mlx.mlx_array_eval(flat));
        try t_std.expectEqualSlices(f32, &want, mlx.mlx_array_data_float32(flat).?[0..12]);
    }
}

test "an MLX stream inside a forward of the engine is refused by name" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    var buf: [512]u8 = undefined;
    _ = mlx.takeError(&buf);
    const data_f = [_]f32{ 1, 2 };
    const a = hostArray(f32, &data_f, &.{2}, .float32);
    defer _ = mlx.mlx_array_free(a);
    var o: A = .{};
    {
        active = true;
        defer active = false;
        try t_std.expectError(error.MlxError, mlx.check(mlx.mlx_add(&o, a, a, gpu)));
        const said = mlx.takeError(&buf).?;
        try t_std.expect(std.mem.indexOf(u8, said, "mlx_add") != null);
        try t_std.expect(std.mem.indexOf(u8, said, "MLX stream inside a forward of the engine") != null);
        // An op the engine has no kernel for names itself.
        try t_std.expectError(error.MlxError, mlx.check(mlx.mlx_argpartition_axis(&o, a, 1, -1, core.stream())));
        try t_std.expect(std.mem.indexOf(u8, mlx.takeError(&buf).?, "mlx_argpartition_axis: the engine has no such op") != null);
    }
    try mlx.check(mlx.mlx_add(&o, a, a, gpu));
    _ = mlx.mlx_array_free(o);
}

/// `a` in `dtype`, made by MLX.
fn castByMlx(a: A, dtype: mlx.mlx_dtype) !A {
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    var o: A = .{};
    try mlx.check(mlx.mlx_astype(&o, a, dtype, gpu));
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

test "engine RMS norm gives MLX's bytes: bf16 and f32, a row under and over the looped limit" {
    try mlx.requireGpu();
    const Build = struct {
        fn norm(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_fast_rms_norm(&o, in[0], in[1], 1e-6, s));
            return o;
        }
    };
    const xs = ramp(f32, 3 * 5120, 6, -3);
    const ws = ramp(f32, 5120, 2, 0.1);
    for ([_][2]c_int{ .{ 12, 1280 }, .{ 3, 5120 }, .{ 240, 64 } }) |dims| {
        const x32 = hostArray(f32, xs[0..@intCast(dims[0] * dims[1])], &dims, .float32);
        defer _ = mlx.mlx_array_free(x32);
        const w32 = hostArray(f32, ws[0..@intCast(dims[1])], &.{dims[1]}, .float32);
        defer _ = mlx.mlx_array_free(w32);
        try expectSame(Build.norm, &.{ x32, w32 });
        const x16 = try castByMlx(x32, .bfloat16);
        defer _ = mlx.mlx_array_free(x16);
        const w16 = try castByMlx(w32, .bfloat16);
        defer _ = mlx.mlx_array_free(w16);
        try expectSame(Build.norm, &.{ x16, w16 });
    }
}

test "engine dequantize gives MLX's bytes: affine 4-bit bf16 and mxfp8 to f32" {
    try mlx.requireGpu();
    const Build = struct {
        fn affine(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_dequantize(&o, in[0], in[1], in[2], mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, .{ .value = .bfloat16, .has_value = true }, s));
            return o;
        }
        fn mxfp8(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_dequantize(&o, in[0], in[1], .{}, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{}, .{ .value = .float32, .has_value = true }, s));
            return o;
        }
    };
    const codes = ramp(u32, 3 * 80, 4.0e9, 0);
    const w = hostArray(u32, &codes, &.{ 3, 80 }, .uint32);
    defer _ = mlx.mlx_array_free(w);
    const scale_f = ramp(f32, 3 * 10, 0.2, 0.01);
    const bias_f = ramp(f32, 3 * 10, 2, -1);
    const scales32 = hostArray(f32, &scale_f, &.{ 3, 10 }, .float32);
    defer _ = mlx.mlx_array_free(scales32);
    const biases32 = hostArray(f32, &bias_f, &.{ 3, 10 }, .float32);
    defer _ = mlx.mlx_array_free(biases32);
    const scales = try castByMlx(scales32, .bfloat16);
    defer _ = mlx.mlx_array_free(scales);
    const biases = try castByMlx(biases32, .bfloat16);
    defer _ = mlx.mlx_array_free(biases);
    try expectSame(Build.affine, &.{ w, scales, biases });

    const exps = ramp(u8, 3 * 10, 6, 124);
    const group_scales = hostArray(u8, &exps, &.{ 3, 10 }, .uint8);
    defer _ = mlx.mlx_array_free(group_scales);
    try expectSame(Build.mxfp8, &.{ w, group_scales });
}

test "a hold that finds MLX as its allocator was last read asks it nothing; after a change it reads once more" {
    try mlx.requireGpu();
    const was_enabled = enabled;
    enabled = true;
    defer @import("../test_support.zig").restoreEngine(was_enabled);
    mlx.noteAllocator();
    var calls = mlxCalls();
    hold();
    unhold();
    try std.testing.expectEqual(calls, mlxCalls());
    // A reading changes nothing in MLX.
    var held_bytes: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&held_bytes));
    _ = mlx.deviceArchitecture();
    calls = mlxCalls();
    hold();
    unhold();
    try std.testing.expectEqual(calls, mlxCalls());
    // An array does.
    {
        var ones = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones);
        try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{ 1 << 10, 4 }, 2, .float32, mlx.gpuStream()));
        try mlx.check(mlx.mlx_array_eval(ones));
    }
    calls = mlxCalls();
    hold();
    unhold();
    try std.testing.expectEqual(calls + 5, mlxCalls());
}

test "under a hold MLX's allocator is not asked: its readings are the hold's first, and a clear, a reset and a limit are no call and change nothing" {
    try mlx.requireGpu();
    const was_enabled = enabled;
    enabled = true;
    defer @import("../test_support.zig").restoreEngine(was_enabled);
    var orig: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&orig, 1 << 30));
    defer {
        var unused: usize = 0;
        _ = mlx.mlx_set_cache_limit(&unused, orig);
    }
    // Something for MLX's pool to hold: the hold empties it as it begins.
    {
        var ones = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ones);
        try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{ 1 << 20, 4 }, 2, .float32, mlx.gpuStream()));
        try mlx.check(mlx.mlx_array_eval(ones));
    }
    hold();
    var is_held = true;
    defer if (is_held) unhold();
    const calls = mlxCalls();
    var now: [4]usize = @splat(1);
    try mlx.check(mlx.mlx_get_active_memory(&now[0]));
    try mlx.check(mlx.mlx_get_cache_memory(&now[1]));
    try mlx.check(mlx.mlx_get_peak_memory(&now[2]));
    try mlx.check(mlx.mlx_get_external_memory(&now[3]));
    try std.testing.expectEqual(@as(usize, 0), now[1]);
    try std.testing.expect(now[2] >= now[0] and now[0] >= now[3]);
    try mlx.check(mlx.mlx_clear_cache());
    try mlx.check(mlx.mlx_reset_peak_memory());
    var peak: usize = 0;
    try mlx.check(mlx.mlx_get_peak_memory(&peak));
    try std.testing.expectEqual(now[0], peak);
    try std.testing.expectEqual(now[0] - now[3], mlx.transientPeak());
    var prev: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&prev, 5));
    try std.testing.expectEqual(@as(usize, 5), prev);
    try std.testing.expectEqual(@as(usize, 7), mlx.setTransientLimit(7));
    const tick = mlx.DecodeBuffers.begin();
    tick.end();
    try std.testing.expectEqual(calls, mlxCalls());
    unhold();
    is_held = false;
    // The limit set under the hold never reached MLX.
    var after: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&after, orig));
    try std.testing.expectEqual(@as(usize, 1 << 30), after);
}

test "a map of the engine's holds arrays and strings by name, in the order they were put, and answers mlx-c's way for a name it lacks" {
    try mlx.requireGpu();
    const t = std.testing;
    const one = [_]f32{ 1, 2 };
    const a = mlx_array_new_data(&one, &[_]c_int{2}, 1, .float32);
    defer _ = mlx_array_free(a);
    const b = mlx_array_new_data(&one, &[_]c_int{ 1, 2 }, 2, .float32);
    defer _ = mlx_array_free(b);

    const map = mlx_map_string_to_array_new();
    defer _ = mlx_map_string_to_array_free(map);
    try t.expectEqual(@as(c_int, 0), mlx_map_string_to_array_insert(map, "second", b));
    try t.expectEqual(@as(c_int, 0), mlx_map_string_to_array_insert(map, "first", a));
    // A name put again takes the new array.
    try t.expectEqual(@as(c_int, 0), mlx_map_string_to_array_insert(map, "second", a));
    var got: A = .{};
    defer _ = mlx_array_free(got);
    try t.expectEqual(@as(c_int, 0), mlx_map_string_to_array_get(&got, map, "second"));
    try t.expectEqual(core.of(a).store, core.of(got).store);
    try t.expectEqual(@as(usize, 1), core.of(got).ndim);
    try t.expectEqual(@as(c_int, 2), mlx_map_string_to_array_get(&got, map, "third"));

    const it = mlx_map_string_to_array_iterator_new(map);
    defer _ = mlx_map_string_to_array_iterator_free(it);
    var key: ?[*:0]const u8 = null;
    for ([_][]const u8{ "second", "first" }) |name| {
        try t.expectEqual(@as(c_int, 0), mlx_map_string_to_array_iterator_next(&key, &got, it));
        try t.expectEqualStrings(name, std.mem.span(key.?));
    }
    try t.expectEqual(@as(c_int, 2), mlx_map_string_to_array_iterator_next(&key, &got, it));

    const meta = mlx_map_string_to_string_new();
    defer _ = mlx_map_string_to_string_free(meta);
    try t.expectEqual(@as(c_int, 0), mlx_map_string_to_string_insert(meta, "layers", "4"));
    try t.expectEqual(@as(c_int, 0), mlx_map_string_to_string_insert(meta, "layers", "40"));
    var value: [*:0]const u8 = undefined;
    try t.expectEqual(@as(c_int, 0), mlx_map_string_to_string_get(&value, meta, "layers"));
    try t.expectEqualStrings("40", std.mem.span(value));
    try t.expectEqual(@as(c_int, 2), mlx_map_string_to_string_get(&value, meta, "init"));

    // MLX's file writer never sees a map of ours: the call is refused by name.
    const was_enabled = enabled;
    enabled = true;
    defer enabled = was_enabled;
    var buf: [256]u8 = undefined;
    _ = mlx.takeError(&buf);
    try t.expectEqual(@as(c_int, 1), mlx.mlx_save_safetensors("/dev/null", map, meta));
    try t.expect(std.mem.indexOf(u8, mlx.takeError(&buf).?, "[engine] mlx_save_safetensors") != null);
}

test "engine quantize gives MLX's codes, scales and biases: affine 8-bit groups of 128 over bf16 and 4-bit groups of 64 over f32" {
    try mlx.requireGpu();
    const Build = struct {
        fn part(comptime group: c_int, comptime bits: c_int, comptime at: usize) fn (S, []const A) anyerror!A {
            return struct {
                fn f(s: S, in: []const A) !A {
                    var v: V = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(v);
                    try mlx.check(mlx.mlx_quantize(&v, in[0], mlx.mlx_optional_int.some(group), mlx.mlx_optional_int.some(bits), "affine", .{}, s));
                    var o: A = .{};
                    try mlx.check(mlx.mlx_vector_array_get(&o, v, at));
                    return o;
                }
            }.f;
        }
    };
    // Groups that are all positive, all negative, mixed, and all zero: every arm of the scale and bias choice.
    var values: [3 * 512]f32 = undefined;
    for (&values, 0..) |*v, i| {
        const g = i / 128;
        const x = @as(f32, @floatFromInt((i * 37) % 251)) / 251.0;
        v.* = switch (g % 4) {
            0 => x + 0.25,
            1 => -x - 0.5,
            2 => x * 3 - 1.5,
            else => 0,
        };
    }
    const w32 = hostArray(f32, &values, &.{ 3, 512 }, .float32);
    defer _ = mlx.mlx_array_free(w32);
    const wbf = try castByMlx(w32, .bfloat16);
    defer _ = mlx.mlx_array_free(wbf);
    inline for (0..3) |at| {
        try expectSame(Build.part(128, 8, at), &.{wbf});
        try expectSame(Build.part(64, 4, at), &.{w32});
    }
}

test "engine vector-matrix products give MLX's bytes: a row against a transposed weight, a short row against a wide matrix, a matrix against a column" {
    try mlx.requireGpu();
    const Build = struct {
        fn againstTransposed(s: S, in: []const A) !A {
            var wt: A = .{};
            defer _ = mlx.mlx_array_free(wt);
            try mlx.check(mlx.mlx_transpose(&wt, in[1], s));
            var o: A = .{};
            try mlx.check(mlx.mlx_matmul(&o, in[0], wt, s));
            return o;
        }
        fn plain(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_matmul(&o, in[0], in[1], s));
            return o;
        }
    };
    const xs = ramp(f32, 5120, 2, -1);
    const ws = ramp(f32, 32 * 5120, 0.2, -0.1);
    const x32 = hostArray(f32, &xs, &.{ 1, 5120 }, .float32);
    defer _ = mlx.mlx_array_free(x32);
    const w32 = hostArray(f32, &ws, &.{ 32, 5120 }, .float32);
    defer _ = mlx.mlx_array_free(w32);
    try expectSame(Build.againstTransposed, &.{ x32, w32 });
    const x16 = try castByMlx(x32, .bfloat16);
    defer _ = mlx.mlx_array_free(x16);
    const w16 = try castByMlx(w32, .bfloat16);
    defer _ = mlx.mlx_array_free(w16);
    try expectSame(Build.againstTransposed, &.{ x16, w16 });

    const short = hostArray(f32, xs[0..4], &.{ 1, 1, 4 }, .float32);
    defer _ = mlx.mlx_array_free(short);
    const wide = hostArray(f32, ws[0 .. 4 * 5120], &.{ 1, 4, 5120 }, .float32);
    defer _ = mlx.mlx_array_free(wide);
    try expectSame(Build.plain, &.{ short, wide });

    const tall = hostArray(f32, ws[0 .. 48 * 300], &.{ 48, 300 }, .float32);
    defer _ = mlx.mlx_array_free(tall);
    const column = hostArray(f32, xs[0..300], &.{300}, .float32);
    defer _ = mlx.mlx_array_free(column);
    try expectSame(Build.plain, &.{ tall, column });
}

test "engine matrix products give MLX's bytes: rows against a weight, two matrices, a long inner axis, batches, dot products" {
    try mlx.requireGpu();
    const Build = struct {
        fn plain(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_matmul(&o, in[0], in[1], s));
            return o;
        }
        fn againstTransposed(s: S, in: []const A) !A {
            var wt: A = .{};
            defer _ = mlx.mlx_array_free(wt);
            try mlx.check(mlx.mlx_transpose(&wt, in[1], s));
            var o: A = .{};
            try mlx.check(mlx.mlx_matmul(&o, in[0], wt, s));
            return o;
        }
        /// `in[0]` with its first two axes swapped, so its batch is not one run of memory.
        fn swappedBatch(s: S, in: []const A) !A {
            var turned: A = .{};
            defer _ = mlx.mlx_array_free(turned);
            try mlx.check(mlx.mlx_transpose_axes(&turned, in[0], &[_]c_int{ 1, 0, 2, 3 }, 4, s));
            var o: A = .{};
            try mlx.check(mlx.mlx_matmul(&o, turned, in[1], s));
            return o;
        }
    };
    const xs = ramp(f32, 24 * 1024, 2, -1);
    const ws = ramp(f32, 1024 * 200, 1, -0.5);
    const Case = struct { a: []const c_int, b: []const c_int, half: bool, build: fn (S, []const A) anyerror!A };
    const cases = [_]Case{
        // Rows against a transposed weight: one, two and three streams of the weight, then too many rows for that.
        .{ .a = &.{ 3, 64 }, .b = &.{ 128, 64 }, .half = true, .build = Build.againstTransposed },
        .{ .a = &.{ 8, 64 }, .b = &.{ 128, 64 }, .half = true, .build = Build.againstTransposed },
        .{ .a = &.{ 12, 64 }, .b = &.{ 48, 64 }, .half = true, .build = Build.againstTransposed },
        .{ .a = &.{ 16, 64 }, .b = &.{ 128, 64 }, .half = true, .build = Build.againstTransposed },
        .{ .a = &.{ 5, 64 }, .b = &.{ 128, 64 }, .half = false, .build = Build.againstTransposed },
        // Two matrices as they lie.
        .{ .a = &.{ 4, 96 }, .b = &.{ 96, 200 }, .half = false, .build = Build.plain },
        .{ .a = &.{ 70, 96 }, .b = &.{ 96, 200 }, .half = true, .build = Build.plain },
        // An inner axis long against the result: the product is summed from parts.
        .{ .a = &.{ 4, 1024 }, .b = &.{ 64, 1024 }, .half = false, .build = Build.againstTransposed },
        .{ .a = &.{ 20, 1024 }, .b = &.{ 1024, 64 }, .half = true, .build = Build.plain },
        // Batches: one run of memory, a matrix shared by every batch, a batch folded into the rows, a broadcast
        // batch axis, a batch that is not one run of memory, and one row per batch.
        .{ .a = &.{ 2, 4, 5, 32 }, .b = &.{ 2, 4, 32, 7 }, .half = true, .build = Build.plain },
        .{ .a = &.{ 2, 5, 32 }, .b = &.{ 32, 7 }, .half = true, .build = Build.plain },
        .{ .a = &.{ 6, 5, 32 }, .b = &.{ 1, 32, 7 }, .half = true, .build = Build.plain },
        .{ .a = &.{ 2, 3, 5, 32 }, .b = &.{ 3, 32, 7 }, .half = true, .build = Build.plain },
        .{ .a = &.{ 3, 2, 5, 32 }, .b = &.{ 2, 3, 32, 7 }, .half = true, .build = Build.swappedBatch },
        .{ .a = &.{ 2, 3, 1, 32 }, .b = &.{ 2, 3, 32, 7 }, .half = true, .build = Build.plain },
        .{ .a = &.{ 2, 3, 1, 32 }, .b = &.{ 3, 32, 7 }, .half = false, .build = Build.plain },
        // One row against one column, and two vectors.
        .{ .a = &.{ 1, 288 }, .b = &.{ 288, 1 }, .half = false, .build = Build.plain },
        .{ .a = &.{ 1, 20000 }, .b = &.{ 20000, 1 }, .half = true, .build = Build.plain },
        .{ .a = &.{300}, .b = &.{300}, .half = true, .build = Build.plain },
    };
    inline for (cases) |case| {
        var na: usize = 1;
        for (case.a) |d| na *= @intCast(d);
        var nb: usize = 1;
        for (case.b) |d| nb *= @intCast(d);
        const a32 = hostArray(f32, xs[0..na], case.a, .float32);
        defer _ = mlx.mlx_array_free(a32);
        const b32 = hostArray(f32, ws[0..nb], case.b, .float32);
        defer _ = mlx.mlx_array_free(b32);
        if (case.half) {
            const a16 = try castByMlx(a32, .bfloat16);
            defer _ = mlx.mlx_array_free(a16);
            const b16 = try castByMlx(b32, .bfloat16);
            defer _ = mlx.mlx_array_free(b16);
            try expectSame(case.build, &.{ a16, b16 });
        } else try expectSame(case.build, &.{ a32, b32 });
    }
}

test "engine softmax, mean, negative, argmax, repeat, stack and take along an axis give MLX's bytes" {
    try mlx.requireGpu();
    const Build = struct {
        fn softLast(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_softmax_axis(&o, in[0], -1, true, s));
            return o;
        }
        fn softMid(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_softmax_axis(&o, in[0], 1, true, s));
            return o;
        }
        fn mean(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_mean_axis(&o, in[0], 1, true, s));
            return o;
        }
        fn negative(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_negative(&o, in[0], s));
            return o;
        }
        fn argmax(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_argmax_axis(&o, in[0], -1, false, s));
            return o;
        }
        fn repeat(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_repeat_axis(&o, in[0], 3, -1, s));
            return o;
        }
        fn stack(s: S, in: []const A) !A {
            const v = mlx.mlx_vector_array_new_data(&[_]A{ in[0], in[0] }, 2);
            defer _ = mlx.mlx_vector_array_free(v);
            var o: A = .{};
            try mlx.check(mlx.mlx_stack_axis(&o, v, 1, s));
            return o;
        }
        fn along(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_take_along_axis(&o, in[0], in[1], 1, s));
            return o;
        }
    };
    const xs = ramp(f32, 4 * 6 * 40, 6, -3);
    const x = hostArray(f32, &xs, &.{ 4, 6, 40 }, .float32);
    defer _ = mlx.mlx_array_free(x);
    const x16 = try castByMlx(x, .bfloat16);
    defer _ = mlx.mlx_array_free(x16);
    inline for (.{ Build.softLast, Build.softMid, Build.negative, Build.argmax, Build.repeat, Build.stack }) |f| {
        try expectSame(f, &.{x});
        try expectSame(f, &.{x16});
    }
    try expectSame(Build.mean, &.{x});
    const picks = ramp(i32, 4 * 3 * 40, 6, 0);
    const idx = hostArray(i32, &picks, &.{ 4, 3, 40 }, .int32);
    defer _ = mlx.mlx_array_free(idx);
    try expectSame(Build.along, &.{ x, idx });
}

test "the engine reads a strided MLX view through its strides" {
    try mlx.requireGpu();
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    const data_f = ramp(f32, 6 * 50, 8, -4);
    const base = hostArray(f32, &data_f, &.{ 6, 50 }, .float32);
    defer _ = mlx.mlx_array_free(base);
    var turned: A = .{};
    defer _ = mlx.mlx_array_free(turned);
    try mlx.check(mlx.mlx_transpose(&turned, base, gpu));
    var view: A = .{};
    defer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_slice(&view, turned, &[_]c_int{ 3, 1 }, 2, &[_]c_int{ 47, 6 }, 2, &[_]c_int{ 2, 1 }, 2, gpu));
    try mlx.check(mlx.mlx_array_eval(view));
    const Build = struct {
        fn doubled(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_add(&o, in[0], in[0], s));
            return o;
        }
        fn rowSums(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_max_axis(&o, in[0], 0, false, s));
            return o;
        }
    };
    try expectSame(Build.doubled, &.{view});
    try expectSame(Build.rowSums, &.{view});
}

/// An add, a vector-matrix product and a sum over an axis on the engine, queued together and read once: their bytes.
fn addMatmulSum() ![492]u8 {
    const gpu_data = ramp(f32, 60, 6, -3);
    const w_data = ramp(f32, 48 * 300, 0.2, -0.1);
    const x_data = ramp(f32, 300, 2, -1);
    const a = hostArray(f32, &gpu_data, &.{ 3, 4, 5 }, .float32);
    defer _ = mlx.mlx_array_free(a);
    const w = hostArray(f32, &w_data, &.{ 48, 300 }, .float32);
    defer _ = mlx.mlx_array_free(w);
    const x = hostArray(f32, &x_data, &.{300}, .float32);
    defer _ = mlx.mlx_array_free(x);
    const s = core.stream();
    var results: [3]A = .{ .{}, .{}, .{} };
    defer for (results) |r| {
        if (r.ctx != null) _ = mlx.mlx_array_free(r);
    };
    try mlx.check(mlx.mlx_add(&results[0], a, a, s));
    try mlx.check(mlx.mlx_matmul(&results[1], w, x, s));
    try mlx.check(mlx.mlx_sum_axis(&results[2], a, 1, true, s));
    var out: [492]u8 = undefined;
    var at: usize = 0;
    for (results) |r| {
        try mlx.check(mlx.mlx_array_eval(r));
        const n = mlx.mlx_array_size(r) * mlx.mlx_array_itemsize(r);
        @memcpy(out[at..][0..n], mlx.mlx_array_data_uint8(r).?[0..n]);
        at += n;
    }
    try t_std.expectEqual(out.len, at);
    return out;
}

test "with the kernel profile on, each dispatch is its own command buffer and the profile names it" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    core.kernel_prof_hook = true;
    defer core.kernel_prof_hook = null;
    try core.sync();
    core.resetKernelProfile();
    const before = core.stats();
    _ = try addMatmulSum();
    const after = core.stats();
    const dispatches = after.dispatches - before.dispatches;
    try t_std.expectEqual(@as(u64, 3), dispatches);
    try t_std.expectEqual(dispatches, after.submits - before.submits);
    var counted: u64 = 0;
    var saw_gemv = false;
    var saw_reduce = false;
    for (core.kernelProfile()) |k| {
        counted += k.dispatches;
        try t_std.expect(k.name.len > 0);
        try t_std.expect(k.total_ns > 0 and k.max_ns > 0 and k.max_ns <= k.total_ns);
        saw_gemv = saw_gemv or std.mem.indexOf(u8, k.name, "gemv") != null;
        saw_reduce = saw_reduce or std.mem.indexOf(u8, k.name, "reduce") != null;
    }
    try t_std.expectEqual(dispatches, counted);
    try t_std.expect(saw_gemv and saw_reduce);
    core.resetKernelProfile();
    try t_std.expectEqual(@as(usize, 0), core.kernelProfile().len);
}

test "the kernel profile leaves values as they were" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const plain = try addMatmulSum();
    core.kernel_prof_hook = true;
    defer core.kernel_prof_hook = null;
    const profiled = try addMatmulSum();
    try t_std.expectEqualSlices(u8, &plain, &profiled);
}

test "with the kernel profile off, the profile is empty and the three ops share one command buffer" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    core.kernel_prof_hook = false;
    defer core.kernel_prof_hook = null;
    try core.sync();
    core.resetKernelProfile();
    const before = core.stats();
    _ = try addMatmulSum();
    const after = core.stats();
    try t_std.expectEqual(@as(u64, 3), after.dispatches - before.dispatches);
    try t_std.expectEqual(@as(u64, 1), after.submits - before.submits);
    try t_std.expectEqual(@as(usize, 0), core.kernelProfile().len);
}

// ── The sampled verdict's ops ──

fn build1(comptime f: anytype) fn (S, []const A) anyerror!A {
    return struct {
        fn build(s: S, in: []const A) anyerror!A {
            var o: A = .{};
            try mlx.check(f(&o, in[0], s));
            return o;
        }
    }.build;
}

fn buildAxis(comptime f: anytype, comptime axis: c_int) fn (S, []const A) anyerror!A {
    return struct {
        fn build(s: S, in: []const A) anyerror!A {
            var o: A = .{};
            try mlx.check(f(&o, in[0], axis, s));
            return o;
        }
    }.build;
}

/// `ramp` rounded down to quarters, so a row holds many equal values.
fn tied(comptime T: type, comptime n: usize) [n]T {
    var out = ramp(T, n, 6, -3);
    for (&out) |*v| v.* = @floatCast(@floor(@as(f32, @floatCast(v.*)) * 4) / 4);
    return out;
}

test "engine log gives MLX's bytes: f32 and bf16, over a row with zeros" {
    try mlx.requireGpu();
    var values = ramp(f32, 130, 1, 0);
    values[3] = 0;
    values[77] = 0;
    values[100] = 1;
    const x = hostArray(f32, &values, &.{ 2, 65 }, .float32);
    defer _ = mlx.mlx_array_free(x);
    const x16 = try castByMlx(x, .bfloat16);
    defer _ = mlx.mlx_array_free(x16);
    try expectSame(build1(mlx.mlx_log), &.{x});
    try expectSame(build1(mlx.mlx_log), &.{x16});
}

test "engine sort and argsort give MLX's bytes: ties, one block and many, a block of rows, a strided view" {
    try mlx.requireGpu();
    const Build = struct {
        fn sliced(s: S, in: []const A) !A {
            var order: A = .{};
            defer _ = mlx.mlx_array_free(order);
            try mlx.check(mlx.mlx_argsort_axis(&order, in[0], -1, s));
            var head: A = .{};
            defer _ = mlx.mlx_array_free(head);
            const extent = mlx.getShape(order);
            try mlx.check(mlx.mlx_slice(&head, order, &.{ 0, 0, 0 }, 3, &.{ extent[0], extent[1], 40 }, 3, &.{ 1, 1, 1 }, 3, s));
            var o: A = .{};
            try mlx.check(mlx.mlx_sort_axis(&o, head, -1, s));
            return o;
        }
        fn negated(s: S, in: []const A) !A {
            var neg: A = .{};
            defer _ = mlx.mlx_array_free(neg);
            try mlx.check(mlx.mlx_negative(&neg, in[0], s));
            var o: A = .{};
            try mlx.check(mlx.mlx_argsort_axis(&o, neg, -1, s));
            return o;
        }
    };
    const vocab = 129280;
    const shapes = [_][]const c_int{ &.{ 1, 1, 40 }, &.{ 1, 3, 200 }, &.{ 1, 2, 2309 }, &.{ 1, 1, vocab }, &.{ 7, 1024 } };
    const row = tied(f32, vocab);
    const rows = tied(f32, 2 * 2309);
    const many = tied(f32, 3 * 200);
    const short = tied(f32, 40);
    const wide = tied(f32, 7 * 1024);
    const sources = [_][]const f32{ &short, &many, &rows, &row, &wide };
    for (shapes, sources) |extent, values| {
        const x = hostArray(f32, values, extent, .float32);
        defer _ = mlx.mlx_array_free(x);
        const x16 = try castByMlx(x, .bfloat16);
        defer _ = mlx.mlx_array_free(x16);
        for ([_]A{ x, x16 }) |input| {
            try expectSame(buildAxis(mlx.mlx_argsort_axis, -1), &.{input});
            try expectSame(buildAxis(mlx.mlx_sort_axis, -1), &.{input});
            try expectSame(Build.negated, &.{input});
        }
    }
    const x = hostArray(f32, &rows, &.{ 1, 2, 2309 }, .float32);
    defer _ = mlx.mlx_array_free(x);
    try expectSame(Build.sliced, &.{x});
}

test "engine cumsum gives MLX's bytes: exclusive and inclusive, short rows, a vocabulary row, a long row, integers along any axis" {
    try mlx.requireGpu();
    const Build = struct {
        fn scan(comptime reverse: bool, comptime inclusive: bool) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    var o: A = .{};
                    try mlx.check(mlx.mlx_cumsum(&o, in[0], -1, reverse, inclusive, s));
                    return o;
                }
            }.build;
        }
        fn along(comptime axis: c_int, comptime reverse: bool, comptime inclusive: bool) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    var o: A = .{};
                    try mlx.check(mlx.mlx_cumsum(&o, in[0], axis, reverse, inclusive, s));
                    return o;
                }
            }.build;
        }
    };
    const vocab = 129280;
    const shapes = [_][]const c_int{ &.{ 1, 4, 40 }, &.{ 3, 6000 }, &.{ 2, 7000 }, &.{vocab}, &.{ 1, 2, vocab } };
    const a = ramp(f32, 4 * 40, 0.02, 0);
    const b = ramp(f32, 3 * 6000, 0.002, 0);
    const c = ramp(f32, 2 * 7000, 0.002, 0);
    const d = ramp(f32, vocab, 0.0002, 0);
    const e = ramp(f32, 2 * vocab, 0.0002, 0);
    const sources = [_][]const f32{ &a, &b, &c, &d, &e };
    for (shapes, sources) |extent, values| {
        const x = hostArray(f32, values, extent, .float32);
        defer _ = mlx.mlx_array_free(x);
        try expectSame(Build.scan(false, false), &.{x});
        try expectSame(Build.scan(false, true), &.{x});
        try expectSame(Build.scan(true, false), &.{x});
    }
    // Integers: the last axis, and a middle one, which MLX scans with another kernel.
    var counts: [3 * 5 * 300]i32 = undefined;
    for (&counts, 0..) |*v, i| v.* = @intCast(i * 7919 % 13);
    const n = hostArray(i32, &counts, &.{ 3, 5, 300 }, .int32);
    defer _ = mlx.mlx_array_free(n);
    try expectSame(Build.scan(false, true), &.{n});
    try expectSame(Build.scan(true, false), &.{n});
    try expectSame(Build.along(1, false, true), &.{n});
    try expectSame(Build.along(0, true, false), &.{n});
}

test "engine put_along_axis and pad give MLX's bytes: keep flags scattered along the last axis, a constant border" {
    try mlx.requireGpu();
    const Build = struct {
        fn scatter(s: S, in: []const A) !A {
            const extent = mlx.getShape(in[2]);
            var zeros: A = .{};
            defer _ = mlx.mlx_array_free(zeros);
            try mlx.check(mlx.mlx_zeros(&zeros, extent.ptr, extent.len, .bool_, s));
            var o: A = .{};
            try mlx.check(mlx.mlx_put_along_axis(&o, zeros, in[0], in[1], -1, s));
            return o;
        }
        fn scatterTrue(s: S, in: []const A) !A {
            const extent = mlx.getShape(in[2]);
            var zeros: A = .{};
            defer _ = mlx.mlx_array_free(zeros);
            try mlx.check(mlx.mlx_zeros(&zeros, extent.ptr, extent.len, .bool_, s));
            const yes = mlx.mlx_array_new_bool(true);
            defer _ = mlx.mlx_array_free(yes);
            var o: A = .{};
            try mlx.check(mlx.mlx_put_along_axis(&o, zeros, in[0], yes, -1, s));
            return o;
        }
        fn border(s: S, in: []const A) !A {
            const ninf = mlx.mlx_array_new_float(-std.math.inf(f32));
            defer _ = mlx.mlx_array_free(ninf);
            var fill: A = .{};
            defer _ = mlx.mlx_array_free(fill);
            try mlx.check(mlx.mlx_astype(&fill, ninf, mlx.mlx_array_dtype(in[0]), s));
            var o: A = .{};
            try mlx.check(mlx.mlx_pad(&o, in[0], &[_]c_int{2}, 1, &[_]c_int{0}, 1, &[_]c_int{17}, 1, fill, "constant", s));
            return o;
        }
    };
    const vocab = 300;
    const picks_u = blk: {
        var out: [4 * 40]u32 = undefined;
        for (0..4) |r| for (0..40) |i| {
            out[r * 40 + i] = @intCast((i * 7 + r * 13) % vocab);
        };
        break :blk out;
    };
    var picks_i: [4 * 40]i32 = undefined;
    for (&picks_i, picks_u) |*o, v| o.* = @intCast(v);
    var flags: [4 * 40]bool = undefined;
    for (&flags, 0..) |*f, i| f.* = i % 3 != 0;
    const base = ramp(f32, 4 * vocab, 4, -2);
    const x = hostArray(f32, &base, &.{ 1, 4, vocab }, .float32);
    defer _ = mlx.mlx_array_free(x);
    const x16 = try castByMlx(x, .bfloat16);
    defer _ = mlx.mlx_array_free(x16);
    const idx_u = hostArray(u32, &picks_u, &.{ 1, 4, 40 }, .uint32);
    defer _ = mlx.mlx_array_free(idx_u);
    const idx_i = hostArray(i32, &picks_i, &.{ 1, 4, 40 }, .int32);
    defer _ = mlx.mlx_array_free(idx_i);
    const keep = hostArray(bool, &flags, &.{ 1, 4, 40 }, .bool_);
    defer _ = mlx.mlx_array_free(keep);
    try expectSame(Build.scatter, &.{ idx_u, keep, x });
    try expectSame(Build.scatter, &.{ idx_i, keep, x });
    try expectSame(Build.scatterTrue, &.{ idx_i, keep, x });
    try expectSame(Build.scatterTrue, &.{ idx_u, keep, x });
    try expectSame(Build.border, &.{x});
    try expectSame(Build.border, &.{x16});
}

test "engine random bits, uniform and categorical give MLX's bytes for a key, and a draw without a key is refused" {
    try mlx.requireGpu();
    const Build = struct {
        fn key(seed: u64) A {
            var k: A = .{};
            _ = mlx.mlx_random_key(&k, seed);
            return k;
        }
        fn bits(comptime width: c_int, comptime extent: []const c_int) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    var o: A = .{};
                    try mlx.check(mlx.mlx_random_bits(&o, extent.ptr, extent.len, width, in[0], s));
                    return o;
                }
            }.build;
        }
        fn uniform(comptime dtype: mlx.mlx_dtype, comptime extent: []const c_int) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    const lo = mlx.mlx_array_new_float(0);
                    defer _ = mlx.mlx_array_free(lo);
                    const hi = mlx.mlx_array_new_float(1);
                    defer _ = mlx.mlx_array_free(hi);
                    var o: A = .{};
                    try mlx.check(mlx.mlx_random_uniform(&o, lo, hi, extent.ptr, extent.len, dtype, in[0], s));
                    return o;
                }
            }.build;
        }
        fn categorical(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_random_categorical(&o, in[1], -1, in[0], s));
            return o;
        }
    };
    const vocab = 129280;
    var row = ramp(f32, vocab, 8, -4);
    for (row[0..1000]) |*v| v.* = -std.math.inf(f32);
    var block = ramp(f32, 4 * 300, 8, -4);
    block[5] = -std.math.inf(f32);
    const logits = hostArray(f32, &row, &.{ 1, vocab }, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const logits16 = try castByMlx(logits, .bfloat16);
    defer _ = mlx.mlx_array_free(logits16);
    const rows = hostArray(f32, &block, &.{ 4, 300 }, .float32);
    defer _ = mlx.mlx_array_free(rows);
    const rows16 = try castByMlx(rows, .bfloat16);
    defer _ = mlx.mlx_array_free(rows16);
    for ([_]u64{ 0, 42, 0x9E3779B97F4A7C15 }) |seed| {
        const k = Build.key(seed);
        defer _ = mlx.mlx_array_free(k);
        try expectSame(Build.bits(4, &.{1}), &.{k});
        try expectSame(Build.bits(4, &.{1001}), &.{k});
        try expectSame(Build.bits(4, &.{ 3, 5 }), &.{k});
        try expectSame(Build.bits(2, &.{7}), &.{k});
        try expectSame(Build.bits(1, &.{9}), &.{k});
        try expectSame(Build.uniform(.float32, &.{1}), &.{k});
        try expectSame(Build.uniform(.float32, &.{ 7, 3 }), &.{k});
        try expectSame(Build.uniform(.bfloat16, &.{ 7, 3 }), &.{k});
        try expectSame(Build.uniform(.float16, &.{5}), &.{k});
        try expectSame(Build.categorical, &.{ k, logits });
        try expectSame(Build.categorical, &.{ k, logits16 });
        try expectSame(Build.categorical, &.{ k, rows });
        try expectSame(Build.categorical, &.{ k, rows16 });
    }
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);
    var drawn: A = .{};
    defer _ = mlx.mlx_array_free(drawn);
    try t_std.expect(mlx.mlx_random_bits(&drawn, &[_]c_int{4}, 1, 4, .{}, core.stream()) != 0);
    try t_std.expect(mlx.mlx_random_categorical(&drawn, logits, -1, .{}, core.stream()) != 0);
}

test "an engine draw key is one function of a root and a position: no two positions of a root share one, and roots do not collide" {
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer seen.deinit(gpa);
    const roots = [_]u64{ 0, 1, 2, 7, 8, 42, 0x9E3779B97F4A7C15, 1 << 63, std.math.maxInt(u64) };
    for (roots) |root| {
        for (0..8192) |position| {
            const key = random.drawKey(root, position);
            try t_std.expectEqual(key, random.drawKey(root, position));
            const fresh = try seen.fetchPut(gpa, @as(u64, key[0]) | @as(u64, key[1]) << 32, {});
            try t_std.expect(fresh == null);
        }
    }
    // Positions far apart and the last representable one keep the one-to-one map.
    var far: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer far.deinit(gpa);
    for ([_]u64{ 0, 1, 1 << 20, 1 << 32, 1 << 40, std.math.maxInt(u64) }) |position| {
        const key = random.drawKey(7, position);
        try t_std.expect((try far.fetchPut(gpa, @as(u64, key[0]) | @as(u64, key[1]) << 32, {})) == null);
    }
}

test "an engine draw from a host key follows the filtered distribution, repeats per key, and touches neither MLX nor a lend" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const vocab = 8;
    const draws = 2000;
    const probs = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    var row: [vocab]f32 = @splat(-std.math.inf(f32));
    for (probs, 0..) |p, i| row[i + 2] = @log(p);
    const logits = try core.fromHost(std.mem.sliceAsBytes(&row), .float32, &.{ 1, vocab });
    defer core.release(logits);
    var only: [vocab]f32 = @splat(-std.math.inf(f32));
    only[5] = 0;
    const single = try core.fromHost(std.mem.sliceAsBytes(&only), .float32, &.{ 1, vocab });
    defer core.release(single);

    const before = core.stats();
    const ops_before = mlx_ops;
    var queued: [draws]*Tensor = undefined;
    for (&queued, 0..) |*out, position| out.* = try random.categoricalAt(logits, -1, random.drawKey(12345, position));
    var counts: [vocab]u32 = @splat(0);
    var first: [16]u32 = undefined;
    for (queued, 0..) |out, position| {
        defer core.release(out);
        const token = std.mem.bytesAsSlice(u32, try core.hostBytes(out))[0];
        counts[token] += 1;
        if (position < first.len) first[position] = token;
    }
    for (0..vocab) |token| {
        const expect: f32 = if (token >= 2 and token < 6) probs[token - 2] else 0;
        const seen = @as(f32, @floatFromInt(counts[token])) / draws;
        try t_std.expect(@abs(seen - expect) < 0.05);
    }
    // The same key draws the same token again, and a filter that leaves one token always draws it.
    for (first, 0..) |token, position| {
        const again = try random.categoricalAt(logits, -1, random.drawKey(12345, position));
        defer core.release(again);
        try t_std.expectEqual(token, std.mem.bytesAsSlice(u32, try core.hostBytes(again))[0]);
        const lone = try random.categoricalAt(single, -1, random.drawKey(position, position));
        defer core.release(lone);
        try t_std.expectEqual(@as(u32, 5), std.mem.bytesAsSlice(u32, try core.hostBytes(lone))[0]);
    }
    const after = core.stats();
    try t_std.expectEqual(ops_before, mlx_ops);
    try t_std.expectEqual(before.lends, after.lends);
    try t_std.expectEqual(before.imports, after.imports);
}

test "the sampler's filters on the engine's stream give MLX's densities: top-k, top-p, both, neither, a suppress mask" {
    try mlx.requireGpu();
    const sampler = @import("../sampler.zig");
    const vocab = 129280;
    const Chain = struct {
        fn of(comptime temperature: f32, comptime top_k: u32, comptime top_p: f32, comptime masked: bool) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    const params: sampler.SamplingParams = .{ .temperature = temperature, .top_k = top_k, .top_p = top_p, .suppress_mask = if (masked) in[1] else null };
                    return sampler.probsAllPositions(in[0], params, s);
                }
            }.build;
        }
    };
    var values = ramp(f32, 3 * vocab, 12, -6);
    // A cluster of equal maxima: the rank rule, not the value, decides which survive.
    for (0..3) |r| for (0..64) |i| {
        values[r * vocab + i * 1999] = 7.5;
    };
    const logits = hostArray(f32, &values, &.{ 1, 3, vocab }, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const one = hostArray(f32, values[0..vocab], &.{ 1, 1, vocab }, .float32);
    defer _ = mlx.mlx_array_free(one);
    var flags: [vocab]bool = @splat(false);
    for (0..64) |i| flags[i * 1999] = i % 5 == 0;
    const mask = hostArray(bool, &flags, &.{vocab}, .bool_);
    defer _ = mlx.mlx_array_free(mask);
    for ([_]A{ logits, one }) |input| {
        try expectSame(Chain.of(0.8, 40, 1.0, false), &.{ input, mask });
        try expectSame(Chain.of(0.8, 0, 0.95, false), &.{ input, mask });
        try expectSame(Chain.of(0.8, 40, 0.95, false), &.{ input, mask });
        try expectSame(Chain.of(0.8, 0, 1.0, false), &.{ input, mask });
        try expectSame(Chain.of(0.8, 40, 0.95, true), &.{ input, mask });
    }
}

test "a call into mlx-c raises the count by one through every kind of path, and a call the engine answers by none" {
    try mlx.requireGpu();
    const was = enabled;
    defer enabled = was;
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    const values = [_]f32{ 1, 2, 3, 4 };
    const a = hostArray(f32, &values, &.{4}, .float32);
    defer _ = mlx.mlx_array_free(a);
    var out: A = .{};
    defer _ = mlx.raw.mlx_array_free(out);

    // A routed op on an MLX stream, with the engine off and on.
    for ([_]bool{ false, true }) |on| {
        enabled = on;
        const before = mlxCalls();
        try mlx.check(mlx.mlx_add(&out, a, a, gpu));
        try t_std.expectEqual(before + 1, mlxCalls());
    }
    enabled = false;

    var before = mlxCalls();
    try mlx.check(mlx.raw.mlx_array_eval(out));
    try t_std.expectEqual(before + 1, mlxCalls());

    // A declaration of mlx-c's own, a query of the array, a kernel config and a vector (which allocates).
    var bytes: usize = 0;
    before = mlxCalls();
    try mlx.check(mlx.mlx_get_active_memory(&bytes));
    try t_std.expectEqual(before + 1, mlxCalls());
    before = mlxCalls();
    _ = mlx.mlx_array_ndim(out);
    try t_std.expectEqual(before + 1, mlxCalls());
    before = mlxCalls();
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    try t_std.expectEqual(before + 1, mlxCalls());
    before = mlxCalls();
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 1, 1, 1));
    try t_std.expectEqual(before + 1, mlxCalls());
    before = mlxCalls();
    _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try t_std.expectEqual(before + 1, mlxCalls());
    before = mlxCalls();
    const vec = mlx.mlx_vector_array_new();
    _ = mlx.mlx_vector_array_free(vec);
    try t_std.expectEqual(before + 2, mlxCalls());

    // An empty handle is a zero `ctx` made here.
    before = mlxCalls();
    try t_std.expectEqual(@as(?*anyopaque, null), mlx.mlx_array_new().ctx);
    try t_std.expectEqual(@as(?*anyopaque, null), mlx.mlx_stream_new().ctx);
    try t_std.expectEqual(before, mlxCalls());

    // The engine's stream: the same op, an eval and the free of a tensor of ours reach MLX not at all.
    enabled = true;
    active = true;
    const ours = mlx.mlx_array_new_data(&values, &[_]c_int{4}, 1, .float32);
    active = false;
    try t_std.expect(isOurs(ours));
    var mine: A = .{};
    // Imports an earlier op retired are freed at the next sync, which is MLX's free then.
    try core.sync();
    before = mlxCalls();
    try mlx.check(mlx.mlx_add(&mine, ours, ours, core.stream()));
    try mlx.check(mlx.mlx_array_eval(mine));
    try t_std.expect(isOurs(mine));
    _ = mlx.mlx_array_free(mine);
    _ = mlx.mlx_array_free(ours);
    try t_std.expectEqual(before, mlxCalls());
}

fn rampBuf(comptime T: type, n: usize, scale: f32, shift: f32) ![]T {
    const out = try std.testing.allocator.alloc(T, n);
    var state: u32 = 0x9e3779b9 ^ @as(u32, @truncate(n));
    for (out) |*v| {
        state = state *% 1664525 +% 1013904223;
        const x = @as(f32, @floatFromInt(state >> 8)) / 16777216.0 * scale + shift;
        v.* = switch (@typeInfo(T)) {
            .float => @floatCast(x),
            else => @intFromFloat(x),
        };
    }
    return out;
}

/// A bf16 array of `n` values, made by MLX from f32 ones.
fn bf16Array(n: usize, dims: []const c_int, scale: f32, shift: f32) !A {
    const values = try rampBuf(f32, n, scale, shift);
    defer std.testing.allocator.free(values);
    const wide = hostArray(f32, values, dims, .float32);
    defer _ = mlx.mlx_array_free(wide);
    return castByMlx(wide, .bfloat16);
}

/// `expectSame`, and the engine's result has MLX's strides as well: a kernel after it picks its plan by them.
fn expectSameLayout(comptime build: anytype, inputs: []const A) !void {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    var strides: [2][8]usize = undefined;
    var rank: [2]usize = undefined;
    for ([_]S{ gpu, core.stream() }, 0..) |s, which| {
        const built = try build(s, inputs);
        defer _ = mlx.mlx_array_free(built);
        try mlx.check(mlx.mlx_array_eval(built));
        rank[which] = mlx.mlx_array_ndim(built);
        @memcpy(strides[which][0..rank[which]], mlx.mlx_array_strides(built)[0..rank[which]]);
    }
    try t_std.expectEqual(rank[0], rank[1]);
    try t_std.expectEqualSlices(usize, strides[0][0..rank[0]], strides[1][0..rank[1]]);
    try expectSame(build, inputs);
}

test "engine erf gives MLX's bytes: f32 under and over the elements MLX takes two to a thread, bf16" {
    try mlx.requireGpu();
    const Build = struct {
        fn erf(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_erf(&o, in[0], s));
            return o;
        }
    };
    for ([_]usize{ 1000, 70000 }) |n| {
        const values = try rampBuf(f32, n, 8, -4);
        defer std.testing.allocator.free(values);
        const x = hostArray(f32, values, &.{@intCast(n)}, .float32);
        defer _ = mlx.mlx_array_free(x);
        try expectSame(Build.erf, &.{x});
        const half = try castByMlx(x, .bfloat16);
        defer _ = mlx.mlx_array_free(half);
        try expectSame(Build.erf, &.{half});
    }
}

test "engine quantized matmul gives MLX's bytes at the tower's shapes: the NAX tiles, aligned and not, 64 rows and 32" {
    try mlx.requireGpu();
    const Build = struct {
        fn product(s: S, in: []const A) !A {
            var o: A = .{};
            try mlx.check(mlx.mlx_quantized_matmul(&o, in[0], in[1], in[2], in[3], true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(8), "affine", s));
            return o;
        }
    };
    // {rows, inner, outputs}: a block's four linears and the aligner's at the release width, an output not whole 64s,
    // and 20 rows against an output over 4096, which MLX still sends to the tiles.
    const cases = [_][3]usize{ .{ 1509, 1024, 3072 }, .{ 1509, 1024, 1024 }, .{ 1536, 1024, 5632 }, .{ 1509, 2816, 1024 }, .{ 169, 9216, 5120 }, .{ 100, 128, 96 }, .{ 20, 128, 8192 } };
    for (cases) |c| {
        const m, const k, const n = c;
        const x = try bf16Array(m * k, &.{ @intCast(m), @intCast(k) }, 4, -2);
        defer _ = mlx.mlx_array_free(x);
        const codes = try rampBuf(u32, n * k / 4, 4.0e9, 0);
        defer std.testing.allocator.free(codes);
        const w = hostArray(u32, codes, &.{ @intCast(n), @intCast(k / 4) }, .uint32);
        defer _ = mlx.mlx_array_free(w);
        const scales = try bf16Array(n * k / 128, &.{ @intCast(n), @intCast(k / 128) }, 0.02, 0.001);
        defer _ = mlx.mlx_array_free(scales);
        const biases = try bf16Array(n * k / 128, &.{ @intCast(n), @intCast(k / 128) }, 2, -1);
        defer _ = mlx.mlx_array_free(biases);
        try expectSameLayout(Build.product, &.{ x, w, scales, biases });
    }
}

test "engine quantized matmul refuses by name what MLX sends to the vector and split-K kernels, and other modes" {
    try mlx.requireGpu();
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const s = core.stream();
    var buf: [512]u8 = undefined;
    _ = mlx.takeError(&buf);
    const x = try bf16Array(8 * 128, &.{ 8, 128 }, 1, 0);
    defer _ = mlx.mlx_array_free(x);
    const codes = try rampBuf(u32, 64 * 32, 4.0e9, 0);
    defer std.testing.allocator.free(codes);
    const w = hostArray(u32, codes, &.{ 64, 32 }, .uint32);
    defer _ = mlx.mlx_array_free(w);
    const scales = try bf16Array(64, &.{ 64, 1 }, 0.02, 0.001);
    defer _ = mlx.mlx_array_free(scales);
    const biases = try bf16Array(64, &.{ 64, 1 }, 1, 0);
    defer _ = mlx.mlx_array_free(biases);
    var o: A = .{};
    try t_std.expectError(error.MlxError, mlx.check(mlx.mlx_quantized_matmul(&o, x, w, scales, biases, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(8), "affine", s)));
    try t_std.expect(std.mem.indexOf(u8, mlx.takeError(&buf).?, "quantized_matmul") != null);
    try t_std.expectError(error.MlxError, mlx.check(mlx.mlx_quantized_matmul(&o, x, w, scales, biases, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(8), "mxfp4", s)));
    try t_std.expect(std.mem.indexOf(u8, mlx.takeError(&buf).?, "quantized_matmul") != null);
}

test "engine attention gives MLX's bytes and strides at the tower's query chunks: aligned and not, a tail of 8 rows or fewer" {
    try mlx.requireGpu();
    const Build = struct {
        const heads = 16;
        const dim = 64;
        /// The tower's own chain: rotated (copied) queries and keys, a viewed value, a chunk of the queries.
        fn chunk(comptime lo: usize, comptime hi: usize) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    const n: c_int = mlx.getShape(in[0])[0];
                    var parts: [3]A = @splat(.{});
                    var made: usize = 0;
                    defer for (parts[0..made]) |p| {
                        _ = mlx.mlx_array_free(p);
                    };
                    inline for (0..3) |j| {
                        var cut: A = .{};
                        defer _ = mlx.mlx_array_free(cut);
                        try mlx.check(mlx.mlx_slice(&cut, in[0], &[_]c_int{ 0, j, 0, 0 }, 4, &[_]c_int{ n, j + 1, heads, dim }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
                        var flat: A = .{};
                        defer _ = mlx.mlx_array_free(flat);
                        try mlx.check(mlx.mlx_reshape(&flat, cut, &[_]c_int{ n, heads, dim }, 3, s));
                        var base = mlx.raw.mlx_array_new();
                        defer _ = mlx.mlx_array_free(base);
                        if (j < 2) {
                            var wide: A = .{};
                            defer _ = mlx.mlx_array_free(wide);
                            try mlx.check(mlx.mlx_astype(&wide, flat, .float32, s));
                            try mlx.check(mlx.mlx_astype(&base, wide, .bfloat16, s));
                        } else {
                            try mlx.check(mlx.mlx_array_set(&base, flat));
                        }
                        var turned: A = .{};
                        defer _ = mlx.mlx_array_free(turned);
                        try mlx.check(mlx.mlx_transpose_axes(&turned, base, &[_]c_int{ 1, 0, 2 }, 3, s));
                        try mlx.check(mlx.mlx_expand_dims(&parts[j], turned, 0, s));
                        made += 1;
                    }
                    var q: A = .{};
                    defer _ = mlx.mlx_array_free(q);
                    try mlx.check(mlx.mlx_slice(&q, parts[0], &[_]c_int{ 0, 0, lo, 0 }, 4, &[_]c_int{ 1, heads, @min(hi, n), dim }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
                    var o: A = .{};
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, q, parts[1], parts[2], 0.125, "", .{}, .{}, false, s));
                    return o;
                }
            }.build;
        }
    };
    // Patch counts: whole chunks and keys of whole 32s; a last chunk of 485 rows; a tail of 8 (two passes on a Max) and of 5.
    inline for (.{ 1536, 1509, 1544, 1541 }) |n| {
        const qkv = try bf16Array(n * 3 * 16 * 64, &.{ n, 3, 16, 64 }, 4, -2);
        defer _ = mlx.mlx_array_free(qkv);
        inline for (0..4) |c| {
            if (c * 512 < n) try expectSameLayout(Build.chunk(c * 512, (c + 1) * 512), &.{qkv});
        }
    }
    // Keys under 1024: the single-pass vector kernel.
    const short = try bf16Array(520 * 3 * 16 * 64, &.{ 520, 3, 16, 64 }, 4, -2);
    defer _ = mlx.mlx_array_free(short);
    try expectSameLayout(Build.chunk(512, 1024), &.{short});
}

test "engine abs, logaddexp and floor divide give MLX's bytes: signed zeros and infinities, the router's softplus, unsigned quotients" {
    try mlx.requireGpu();
    const Build = struct {
        fn binary(comptime f: anytype) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    var o: A = .{};
                    try mlx.check(f(&o, in[0], in[1], s));
                    return o;
                }
            }.build;
        }
        fn softplus(s: S, in: []const A) !A {
            const zero = mlx.mlx_array_new_float(0);
            defer _ = mlx.mlx_array_free(zero);
            var o: A = .{};
            try mlx.check(mlx.mlx_logaddexp(&o, in[0], zero, s));
            return o;
        }
    };
    var values = ramp(f32, 64, 60, -30);
    values[0] = 0;
    values[1] = -0.0;
    values[2] = std.math.inf(f32);
    values[3] = -std.math.inf(f32);
    values[4] = 1e-30;
    values[5] = -1e-9;
    values[6] = 100;
    var others = ramp(f32, 64, 40, -20);
    others[2] = -std.math.inf(f32);
    others[3] = -std.math.inf(f32);
    others[7] = std.math.inf(f32);
    const x = hostArray(f32, &values, &.{ 8, 8 }, .float32);
    defer _ = mlx.mlx_array_free(x);
    const y = hostArray(f32, &others, &.{ 8, 8 }, .float32);
    defer _ = mlx.mlx_array_free(y);
    const x16 = try castByMlx(x, .bfloat16);
    defer _ = mlx.mlx_array_free(x16);
    try expectSame(build1(mlx.mlx_abs), &.{x});
    try expectSame(build1(mlx.mlx_abs), &.{x16});
    try expectSame(Build.binary(mlx.mlx_logaddexp), &.{ x, y });
    try expectSame(Build.softplus, &.{x});
    const wide_values = ramp(f32, 300 * 301, 80, -40);
    const wide = hostArray(f32, &wide_values, &.{ 300, 301 }, .float32);
    defer _ = mlx.mlx_array_free(wide);
    try expectSame(Build.softplus, &.{wide});
    var numer: [48]u32 = undefined;
    for (&numer, 0..) |*v, i| v.* = @intCast(i * 7919 % 1000);
    var denom: [48]u32 = undefined;
    for (&denom, 0..) |*v, i| v.* = @intCast(1 + i % 6);
    const n = hostArray(u32, &numer, &.{ 6, 8 }, .uint32);
    defer _ = mlx.mlx_array_free(n);
    const d = hostArray(u32, &denom, &.{ 6, 8 }, .uint32);
    defer _ = mlx.mlx_array_free(d);
    try expectSame(Build.binary(mlx.mlx_floor_divide), &.{ n, d });
    try expectSame(Build.binary(mlx.mlx_floor_divide), &.{ x, y });
}

test "engine copy shares its input, a view included, and gives MLX's bytes" {
    try mlx.requireGpu();
    const Build = struct {
        fn viewCopy(s: S, in: []const A) !A {
            var t: A = .{};
            defer _ = mlx.mlx_array_free(t);
            try mlx.check(mlx.mlx_transpose_axes(&t, in[0], &[_]c_int{ 1, 0 }, 2, s));
            var o: A = .{};
            try mlx.check(mlx.mlx_copy(&o, t, s));
            return o;
        }
    };
    var values = ramp(f32, 6 * 7, 4, -2);
    const x = hostArray(f32, &values, &.{ 6, 7 }, .float32);
    defer _ = mlx.mlx_array_free(x);
    const x16 = try castByMlx(x, .bfloat16);
    defer _ = mlx.mlx_array_free(x16);
    try expectSame(build1(mlx.mlx_copy), &.{x});
    try expectSame(build1(mlx.mlx_copy), &.{x16});
    try expectSame(Build.viewCopy, &.{x});
}

test "the penalties and the think penalty on the engine's stream give MLX's bytes: one row, verify rows, a bias" {
    try mlx.requireGpu();
    const seen_penalty = @import("../seen_penalty.zig");
    const think_penalty = @import("../think_penalty.zig");
    const vocab = 300;
    const rows = 5;
    const open: u32 = 20;
    const closer: u32 = 21;
    const Chain = struct {
        fn one(comptime pen: seen_penalty.Penalties) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    var o: A = .{};
                    try seen_penalty.apply(&o, in[0], in[1], pen, s);
                    return o;
                }
            }.build;
        }
        fn block(comptime pen: seen_penalty.Penalties) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    var o: A = .{};
                    try seen_penalty.shiftRows(&o, in[0], in[2], in[1], pen, s);
                    return o;
                }
            }.build;
        }
        fn think(comptime span: think_penalty.ThinkPenalty.Phase, comptime biased: bool) fn (S, []const A) anyerror!A {
            return struct {
                fn build(s: S, in: []const A) anyerror!A {
                    const file = [_]@import("../logit_bias.zig").Bias{ .{ .id = 2, .delta = 1, .scope = .reasoning }, .{ .id = 5, .delta = 3, .scope = .answer } };
                    var p = think_penalty.ThinkPenalty{ .lambda = 2.5, .opener_id = open, .closer_id = closer, .phase = span, .file_biases = if (biased) &file else &.{} };
                    try p.prepare(t_std.allocator, vocab, in[3], s);
                    defer p.deinitPrepared();
                    var o: A = .{};
                    try think_penalty.shiftScoped(&o, in[0], in[3], p, in[2], s);
                    return o;
                }
            }.build;
        }
    };
    var values = ramp(f32, rows * vocab, 12, -6);
    const logits = hostArray(f32, &values, &.{ 1, rows, vocab }, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const row = hostArray(f32, values[0..vocab], &.{ 1, vocab }, .float32);
    defer _ = mlx.mlx_array_free(row);
    var counts_host: [vocab]i32 = @splat(0);
    for ([_]usize{ 3, 40, 41, 250 }, 0..) |id, i| counts_host[id] = @intCast(i + 1);
    const counts = hostArray(i32, &counts_host, &.{vocab}, .int32);
    defer _ = mlx.mlx_array_free(counts);
    const ids_host = [rows]i32{ 40, 7, 3, 7, 250 };
    const ids = hostArray(i32, &ids_host, &.{ 1, rows }, .int32);
    defer _ = mlx.mlx_array_free(ids);
    const think_ids_host = [rows]i32{ 7, open, 8, closer, 9 };
    const think_ids = hostArray(i32, &think_ids_host, &.{ 1, rows }, .int32);
    defer _ = mlx.mlx_array_free(think_ids);
    var mask_host: [vocab]bool = @splat(false);
    for ([_]usize{ 2, 5, 100, 299 }) |id| mask_host[id] = true;
    const mask = hostArray(bool, &mask_host, &.{vocab}, .bool_);
    defer _ = mlx.mlx_array_free(mask);

    try expectSame(Chain.one(.{ .repeat = 1.3 }), &.{ row, counts });
    try expectSame(Chain.one(.{ .repeat = 1.3, .presence = 0.5, .frequency = 0.25 }), &.{ row, counts });
    try expectSame(Chain.one(.{ .presence = 0.5, .frequency = 0.25 }), &.{ row, counts });
    try expectSame(Chain.block(.{ .repeat = 1.3 }), &.{ logits, counts, ids });
    try expectSame(Chain.block(.{ .repeat = 1.3, .presence = 0.5, .frequency = 0.25 }), &.{ logits, counts, ids });
    try expectSame(Chain.think(.inside, false), &.{ logits, counts, think_ids, mask });
    try expectSame(Chain.think(.before, false), &.{ logits, counts, think_ids, mask });
    try expectSame(Chain.think(.inside, true), &.{ logits, counts, think_ids, mask });
    try expectSame(Chain.think(.before, true), &.{ logits, counts, think_ids, mask });
}

test "logprobs on the engine's stream rank as MLX's do and agree to the last bits: the chosen token, the top ids, a tie, a wide row" {
    try mlx.requireGpu();
    const sampler = @import("../sampler.zig");
    const was = enabled;
    enabled = true;
    defer enabled = was;
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    const vocab = 129280;
    var values = ramp(f32, vocab, 14, -7);
    // Two equal leaders (the lower id ranks first), and a leader far above the rest.
    values[900] = 9.5;
    values[77] = 9.5;
    values[128000] = 11;
    const f32_row = hostArray(f32, &values, &.{ 1, vocab }, .float32);
    defer _ = mlx.mlx_array_free(f32_row);
    const f32_3d = hostArray(f32, &values, &.{ 1, 1, vocab }, .float32);
    defer _ = mlx.mlx_array_free(f32_3d);
    const bf16_row = try castByMlx(f32_row, .bfloat16);
    defer _ = mlx.mlx_array_free(bf16_row);
    for ([_]A{ f32_row, f32_3d, bf16_row }) |logits| for ([_]u32{ 128000, 900, 5, vocab + 3 }) |chosen| {
        const want = try sampler.computeLogprobs(t_std.allocator, logits, chosen, 5, gpu);
        defer t_std.allocator.free(want.top_logprobs);
        const got = try sampler.computeLogprobs(t_std.allocator, logits, chosen, 5, core.stream());
        defer t_std.allocator.free(got.top_logprobs);
        try t_std.expectEqual(want.top_logprobs.len, got.top_logprobs.len);
        try t_std.expectApproxEqAbs(want.token_logprob, got.token_logprob, 1e-4);
        // A bf16 row ties wherever its fifth place is, and MLX's partition breaks that tie arbitrarily: the leaders only.
        const compared = if (logits.ctx == bf16_row.ctx) 3 else want.top_logprobs.len;
        for (want.top_logprobs[0..compared], got.top_logprobs[0..compared]) |w, g| {
            try t_std.expectEqual(w.token_id, g.token_id);
            try t_std.expectApproxEqAbs(w.logprob, g.logprob, 1e-4);
        }
    };
    // Past twenty the list stops at twenty; a row of two ids gives two.
    const big = try sampler.computeLogprobs(t_std.allocator, f32_row, 0, 99, core.stream());
    defer t_std.allocator.free(big.top_logprobs);
    try t_std.expectEqual(@as(usize, 20), big.top_logprobs.len);
    const two = hostArray(f32, &[_]f32{ 1, 3 }, &.{ 1, 2 }, .float32);
    defer _ = mlx.mlx_array_free(two);
    const short = try sampler.computeLogprobs(t_std.allocator, two, 0, 5, core.stream());
    defer t_std.allocator.free(short.top_logprobs);
    try t_std.expectEqual(@as(usize, 2), short.top_logprobs.len);
    try t_std.expectEqual(@as(u32, 1), short.top_logprobs[0].token_id);
    // The engine's stream reads the row once and puts no op on MLX.
    const before = mlx_ops;
    const once = try sampler.computeLogprobs(t_std.allocator, f32_row, 128000, 5, core.stream());
    defer t_std.allocator.free(once.top_logprobs);
    try t_std.expectEqual(before, mlx_ops);
}
