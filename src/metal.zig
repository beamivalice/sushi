//! Metal from Zig through the Objective-C runtime: the GPU MLX runs on, a command queue of our own, pipelines compiled
//! once and held, and buffers over memory someone else maps. Nothing here calls MLX.
//!
//! What a caller owns that MLX did for it: a buffer outlives every command buffer that binds it (wait before the
//! release), a dispatch that reads what an earlier one wrote relies on encode order (the encoder is serial), and a
//! failed command buffer is reported at its `wait`.

const std = @import("std");

pub const id = ?*anyopaque;
const SEL = ?*anyopaque;

extern fn sel_registerName(name: [*:0]const u8) SEL;
extern fn objc_getClass(name: [*:0]const u8) id;
extern fn objc_msgSend() void;
extern fn objc_autoreleasePoolPush() ?*anyopaque;
extern fn objc_autoreleasePoolPop(pool: ?*anyopaque) void;
extern fn MTLCreateSystemDefaultDevice() id;

/// MTLSize, also the grid and the threadgroup of a dispatch.
pub const Size = extern struct { x: usize, y: usize = 1, z: usize = 1 };

/// One selector per name, registered at its first use.
fn sel(comptime name: [:0]const u8) SEL {
    const S = struct {
        // Naming `name` makes this struct, and so its variable, one per selector.
        const of = name;
        var cached: std.atomic.Value(SEL) = .init(null);
    };
    if (S.cached.load(.monotonic)) |s| return s;
    const s = sel_registerName(S.of);
    S.cached.store(s, .monotonic);
    return s;
}

/// `[target name:args...]` with the argument types taken from `args` and the result type `R`.
fn send(comptime R: type, target: id, comptime name: [:0]const u8, args: anytype) R {
    const T = @typeInfo(@TypeOf(args)).@"struct".field_types;
    const F = switch (T.len) {
        0 => *const fn (id, SEL) callconv(.c) R,
        1 => *const fn (id, SEL, T[0]) callconv(.c) R,
        2 => *const fn (id, SEL, T[0], T[1]) callconv(.c) R,
        3 => *const fn (id, SEL, T[0], T[1], T[2]) callconv(.c) R,
        4 => *const fn (id, SEL, T[0], T[1], T[2], T[3]) callconv(.c) R,
        else => @compileError("metal.send: more arguments than a Metal call here takes"),
    };
    const f: F = @ptrCast(&objc_msgSend);
    return @call(.auto, f, .{ target, sel(name) } ++ args);
}

pub const Error = error{ MetalUnavailable, MetalCompile, MetalBuffer, MetalCommand };

threadlocal var error_text: [2048]u8 = undefined;
threadlocal var error_len: usize = 0;

/// What Metal said about the last `Error` on this thread's device calls; empty when it said nothing.
pub fn lastError() []const u8 {
    return error_text[0..error_len];
}

/// Records a human-readable error message for `lastError()` when a cooler refusal occurs outside a Metal API call.
pub fn setError(what: []const u8) void {
    noteError(null, what);
}

fn noteError(ns_error: id, what: []const u8) void {
    error_len = 0;
    const text: ?[*:0]const u8 = if (ns_error) |e| send(?[*:0]const u8, send(id, e, "localizedDescription", .{}), "UTF8String", .{}) else null;
    const said = if (text) |t| std.mem.span(t) else what;
    error_len = @min(said.len, error_text.len);
    @memcpy(error_text[0..error_len], said[0..error_len]);
}

/// A buffer over memory someone else owns takes MLX's options: shared storage, no hazard tracking. A pass of ours
/// reads such a buffer or writes it in the order its dispatches are encoded, and MLX's queue is idle meanwhile.
const alias_options: usize = 1 << 8;

var shared: ?Device = null;

/// The system GPU, which is the one MLX opens, and our queue on it. Opened once and kept for the process.
pub fn device() Error!*const Device {
    if (shared == null) shared = try Device.open();
    return &shared.?;
}

/// Residency as MLX keeps it (mlx/backend/metal/resident.cpp, MIT): buffers are spread over sets of at most `set_cap`
/// bytes, since macOS decides residency a set at a time and the first pass after a long idle wires a lost set again at
/// its size. Each set holds a standing request and is attached to our queue when it is made: a set on no queue loses
/// its residency.
const max_sets = 32;
const no_set: u8 = 255;
/// What a set holds. The pool's buffers come and go with every forward and a wrapped mapping stays for a load: kept
/// apart, a scratch buffer never changes a set that lists gigabytes of weights. The plan's arena stays for a load too.
const Kind = enum { pool, mapped, arena };
const Set = struct {
    raw: id,
    kind: Kind,
    bytes: usize = 0,
    /// Buffers were added to or taken from it since its last commit.
    dirty: bool = false,
};
var sets: [max_sets]Set = undefined;
var n_sets: u8 = 0;
/// MLX's share of the working set for one set (`MLX_RESIDENCY_SET_MAX_PCT`'s default) and its floor.
const set_percent = 5;
const set_floor: usize = 64 << 20;
var set_cap: usize = set_floor;
var any_dirty = false;

/// How many residency sets hold the engine's buffers, and the bytes in them.
pub fn residentSets() struct { count: usize, bytes: usize } {
    var bytes: usize = 0;
    for (sets[0..n_sets]) |set| bytes += set.bytes;
    return .{ .count = n_sets, .bytes = bytes };
}

/// TEST-ONLY: the cap of a set for the buffers added from now on; returns the cap it replaces.
pub fn setResidentCapForTest(bytes: usize) usize {
    defer set_cap = bytes;
    return set_cap;
}

/// Tells the driver of every buffer added to or taken from a set since the last time. A pass does it as it is handed
/// over; a release that no pass follows (an unload, a trim while idle) needs it said, or the set keeps the pages.
pub fn commitResident() void {
    commitSets();
}

/// Whether a set holds a change the driver has not been told of.
pub fn residentDirty() bool {
    return any_dirty;
}

fn commitSets() void {
    if (!any_dirty) return;
    any_dirty = false;
    for (sets[0..n_sets]) |*set| if (set.dirty) {
        set.dirty = false;
        send(void, set.raw, "commit", .{});
    };
}

var touch_scratch: ?Buffer = null;
/// Passes `Device.touch` has handed over.
pub var touches: u64 = 0;

pub fn touchesCount() u64 {
    return @atomicLoad(u64, &touches, .acquire);
}

fn addSet(dev: *const Device, kind: Kind) ?u8 {
    if (n_sets == max_sets) return null;
    var ns_error: id = null;
    const descriptor = send(id, send(id, objc_getClass("MTLResidencySetDescriptor"), "alloc", .{}), "init", .{});
    defer send(void, descriptor, "release", .{});
    const raw = send(id, dev.raw, "newResidencySetWithDescriptor:error:", .{ descriptor, &ns_error }) orelse {
        noteError(ns_error, "Metal refused a residency set");
        return null;
    };
    send(void, raw, "requestResidency", .{});
    send(void, dev.queue, "addResidencySet:", .{raw});
    sets[n_sets] = .{ .raw = raw, .kind = kind };
    n_sets += 1;
    return n_sets - 1;
}

/// The first set of `kind` that `bytes` fit in; else an emptied set, which takes the kind, or a new one (a buffer over
/// the cap has a set of its own); else, at the most sets a queue takes, the emptiest of the kind, or of all when the
/// kind has none. As MLX's `choose_set_locked`, by kind.
fn chooseSet(dev: *const Device, kind: Kind, bytes: usize) u8 {
    var empty: u8 = no_set;
    var emptiest: u8 = no_set;
    var emptiest_any: u8 = 0;
    for (sets[0..n_sets], 0..) |set, i| {
        if (set.bytes < sets[emptiest_any].bytes) emptiest_any = @intCast(i);
        // The first set is the pool's for good: the device makes it before any buffer.
        if (i != 0 and set.bytes == 0 and empty == no_set) empty = @intCast(i);
        if (set.kind != kind) continue;
        if (set.bytes + bytes <= set_cap) return @intCast(i);
        if (emptiest == no_set or set.bytes < sets[emptiest].bytes) emptiest = @intCast(i);
    }
    if (empty != no_set) {
        sets[empty].kind = kind;
        return empty;
    }
    if (addSet(dev, kind)) |made| return made;
    return if (emptiest != no_set) emptiest else emptiest_any;
}

/// A boolean function constant of a compiled kernel, by its index.
pub const Flag = struct { index: usize, value: bool };

/// An int function constant of a compiled kernel, by its index.
pub const IntConstant = struct { index: usize, value: i32 };

/// `MTLPipelineOptionFailOnBinaryArchiveMiss`: fail instead of compiling AIR when a pipeline is absent from the archive.
pub const pipeline_opt_fail_on_archive_miss: usize = 1 << 2;

/// A serialized GPU binary archive (`MTLBinaryArchive`) holding compiled machine code for one device and OS build.
pub const BinaryArchive = struct {
    raw: id,

    pub fn release(a: BinaryArchive) void {
        send(void, a.raw, "release", .{});
    }

    /// Writes the archive's compiled binaries to `path`.
    pub fn serialize(a: BinaryArchive, path: [:0]const u8) Error!void {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        var ns_error: id = null;
        const text = send(id, objc_getClass("NSString"), "stringWithUTF8String:", .{path.ptr});
        const url = send(id, objc_getClass("NSURL"), "fileURLWithPath:", .{text});
        if (send(u8, a.raw, "serializeToURL:error:", .{ url, &ns_error }) == 0) {
            noteError(ns_error, "the binary archive did not serialize");
            return error.MetalCompile;
        }
    }
};

fn functionOf(lib: Library, function: [:0]const u8, flags: []const Flag, ints: []const IntConstant) Error!id {
    var ns_error: id = null;
    const name = send(id, objc_getClass("NSString"), "stringWithUTF8String:", .{function.ptr});
    if (flags.len == 0 and ints.len == 0) {
        return send(id, lib.raw, "newFunctionWithName:", .{name}) orelse {
            noteError(null, function);
            return error.MetalCompile;
        };
    }
    const values = send(id, send(id, objc_getClass("MTLFunctionConstantValues"), "alloc", .{}), "init", .{});
    defer send(void, values, "release", .{});
    const mtl_data_type_int: usize = 29;
    const mtl_data_type_bool: usize = 53;
    for (flags) |f| {
        var v = f.value;
        send(void, values, "setConstantValue:type:atIndex:", .{ @as(*const anyopaque, &v), mtl_data_type_bool, f.index });
    }
    for (ints) |c| {
        var v = c.value;
        send(void, values, "setConstantValue:type:atIndex:", .{ @as(*const anyopaque, &v), mtl_data_type_int, c.index });
    }
    return send(id, lib.raw, "newFunctionWithName:constantValues:error:", .{ name, values, &ns_error }) orelse {
        noteError(ns_error, function);
        return error.MetalCompile;
    };
}

pub const Device = struct {
    raw: id,
    queue: id,

    fn open() Error!Device {
        const raw = MTLCreateSystemDefaultDevice() orelse return error.MetalUnavailable;
        const queue = send(id, raw, "newCommandQueue", .{}) orelse return error.MetalUnavailable;
        const dev: Device = .{ .raw = raw, .queue = queue };
        // Every buffer of ours is in a set: a command buffer then finds its buffers resident instead of making each
        // one so at commit. The first set is made now, so a refusal is the open's.
        set_cap = @max(send(usize, raw, "recommendedMaxWorkingSetSize", .{}) / 100 * set_percent, set_floor);
        _ = addSet(&dev, .pool) orelse return error.MetalUnavailable;
        return dev;
    }

    /// What Metal has allocated for this process on the device: every buffer, heap and resource, MLX's included.
    pub fn currentAllocatedSize(dev: *const Device) usize {
        return send(usize, dev.raw, "currentAllocatedSize", .{});
    }

    /// The compiled library at `path` (a `.metallib`).
    pub fn load(dev: *const Device, path: [:0]const u8) Error!Library {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        var ns_error: id = null;
        const text = send(id, objc_getClass("NSString"), "stringWithUTF8String:", .{path.ptr});
        const url = send(id, objc_getClass("NSURL"), "fileURLWithPath:", .{text});
        const raw = send(id, dev.raw, "newLibraryWithURL:error:", .{ url, &ns_error }) orelse {
            noteError(ns_error, "the library did not load");
            return error.MetalCompile;
        };
        return .{ .raw = raw };
    }

    /// The compute pipeline of `function`, a kernel `lib` holds.
    pub fn pipelineOf(dev: *const Device, lib: Library, function: [:0]const u8) Error!Pipeline {
        return dev.pipelineOfConstants(lib, function, &.{}, &.{});
    }

    /// `pipelineOf` for a function specialized by boolean function constants, as MLX builds its matmul kernels.
    pub fn pipelineOfFlags(dev: *const Device, lib: Library, function: [:0]const u8, flags: []const Flag) Error!Pipeline {
        return dev.pipelineOfConstants(lib, function, flags, &.{});
    }

    /// `pipelineOfFlags` with int function constants as well.
    pub fn pipelineOfConstants(dev: *const Device, lib: Library, function: [:0]const u8, flags: []const Flag, ints: []const IntConstant) Error!Pipeline {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        var ns_error: id = null;
        const entry = try functionOf(lib, function, flags, ints);
        defer send(void, entry, "release", .{});
        const state = send(id, dev.raw, "newComputePipelineStateWithFunction:error:", .{ entry, &ns_error }) orelse {
            noteError(ns_error, "the pipeline did not build");
            return error.MetalCompile;
        };
        return .{ .raw = state, .max_threads = send(usize, state, "maxTotalThreadsPerThreadgroup", .{}) };
    }

    /// Opens an existing `MTLBinaryArchive` from `path`, or creates an empty one when `path` is null.
    pub fn newBinaryArchive(dev: *const Device, path: ?[:0]const u8) Error!BinaryArchive {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        var ns_error: id = null;
        const desc = send(id, send(id, objc_getClass("MTLBinaryArchiveDescriptor"), "alloc", .{}), "init", .{});
        defer send(void, desc, "release", .{});
        if (path) |p| {
            const text = send(id, objc_getClass("NSString"), "stringWithUTF8String:", .{p.ptr});
            const url = send(id, objc_getClass("NSURL"), "fileURLWithPath:", .{text});
            send(void, desc, "setUrl:", .{url});
        }
        const raw = send(id, dev.raw, "newBinaryArchiveWithDescriptor:error:", .{ desc, &ns_error }) orelse {
            noteError(ns_error, "the binary archive did not open");
            return error.MetalCompile;
        };
        return .{ .raw = raw };
    }

    /// Adds a compute pipeline function (with optional specialization constants) from `lib` into `archive`.
    pub fn addComputeToArchive(_: *const Device, archive: BinaryArchive, lib: Library, function: [:0]const u8, flags: []const Flag, ints: []const IntConstant) Error!void {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        var ns_error: id = null;
        const entry = try functionOf(lib, function, flags, ints);
        defer send(void, entry, "release", .{});
        const desc = send(id, send(id, objc_getClass("MTLComputePipelineDescriptor"), "alloc", .{}), "init", .{});
        defer send(void, desc, "release", .{});
        send(void, desc, "setComputeFunction:", .{entry});
        if (send(u8, archive.raw, "addComputePipelineFunctionsWithDescriptor:error:", .{ desc, &ns_error }) == 0) {
            noteError(ns_error, function);
            return error.MetalCompile;
        }
    }

    /// Builds a compute pipeline using `archive` for precompiled GPU binaries; when `fail_on_miss` is true, refuses
    /// any kernel absent from `archive` instead of compiling AIR from `lib`.
    pub fn pipelineFromArchive(
        dev: *const Device,
        archive: BinaryArchive,
        lib: Library,
        function: [:0]const u8,
        flags: []const Flag,
        ints: []const IntConstant,
        fail_on_miss: bool,
    ) Error!Pipeline {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        var ns_error: id = null;
        const entry = try functionOf(lib, function, flags, ints);
        defer send(void, entry, "release", .{});
        const desc = send(id, send(id, objc_getClass("MTLComputePipelineDescriptor"), "alloc", .{}), "init", .{});
        defer send(void, desc, "release", .{});
        send(void, desc, "setComputeFunction:", .{entry});
        const arr = send(id, objc_getClass("NSArray"), "arrayWithObject:", .{archive.raw});
        send(void, desc, "setBinaryArchives:", .{arr});
        const opts: usize = if (fail_on_miss) pipeline_opt_fail_on_archive_miss else 0;
        const state = send(id, dev.raw, "newComputePipelineStateWithDescriptor:options:reflection:error:", .{ desc, opts, @as(id, null), &ns_error }) orelse {
            noteError(ns_error, "the pipeline did not build from archive");
            return error.MetalCompile;
        };
        return .{ .raw = state, .max_threads = send(usize, state, "maxTotalThreadsPerThreadgroup", .{}) };
    }

    /// A zeroed buffer of `len` bytes that Metal allocates.
    pub fn alloc(dev: *const Device, len: usize) Error!Buffer {
        const raw = send(id, dev.raw, "newBufferWithLength:options:", .{ @max(len, 1), alias_options }) orelse {
            noteError(null, "Metal refused a buffer");
            return error.MetalBuffer;
        };
        return .{ .raw = raw, .set = dev.keepResident(raw, .pool) };
    }

    /// A zeroed buffer of `len` bytes for the plan's arena, in residency sets of its own.
    pub fn allocArena(dev: *const Device, len: usize) Error!Buffer {
        const raw = send(id, dev.raw, "newBufferWithLength:options:", .{ @max(len, 1), alias_options }) orelse {
            noteError(null, "Metal refused the arena's buffer");
            return error.MetalBuffer;
        };
        return .{ .raw = raw, .set = dev.keepResident(raw, .arena) };
    }

    /// A buffer over `len` bytes at the page-aligned `ptr`, which the caller keeps mapped until the buffer is released
    /// and every pass that binds it has run. No copy: the pages are wired once, under this buffer.
    pub fn wrap(dev: *const Device, ptr: [*]u8, len: usize) Error!Buffer {
        const raw = send(id, dev.raw, "newBufferWithBytesNoCopy:length:options:deallocator:", .{ ptr, len, alias_options, @as(id, null) }) orelse {
            noteError(null, "Metal refused a buffer over mapped memory");
            return error.MetalBuffer;
        };
        return .{ .raw = raw, .set = dev.keepResident(raw, .mapped) };
    }

    /// A pass that writes four bytes of scratch, so the driver sees our queue in use: a queue left idle loses the
    /// residency of the sets attached to it within half a minute. Not waited on.
    pub fn touch(dev: *const Device) void {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        if (touch_scratch == null) touch_scratch = dev.alloc(4) catch return;
        const Range = extern struct { location: usize, length: usize };
        const cmd = send(id, dev.queue, "commandBuffer", .{});
        const blit = send(id, cmd, "blitCommandEncoder", .{});
        send(void, blit, "fillBuffer:range:value:", .{ touch_scratch.?.raw, Range{ .location = 0, .length = 4 }, @as(u8, 0) });
        send(void, blit, "endEncoding", .{});
        commitSets();
        send(void, cmd, "commit", .{});
        _ = @atomicRmw(u64, &touches, .Add, 1, .release);
    }

    pub fn fence(dev: *const Device) Error!Fence {
        return .{ .raw = send(id, dev.raw, "newFence", .{}) orelse return error.MetalUnavailable };
    }

    fn keepResident(dev: *const Device, buffer: id, kind: Kind) u8 {
        const len = send(usize, buffer, "length", .{});
        const at = chooseSet(dev, kind, len);
        send(void, sets[at].raw, "addAllocation:", .{buffer});
        sets[at].bytes += len;
        sets[at].dirty = true;
        any_dirty = true;
        return at;
    }

    /// A command buffer with one compute encoder whose dispatches run in the order they are encoded.
    pub fn begin(dev: *const Device) Pass {
        const pool = objc_autoreleasePoolPush();
        const cmd = send(id, dev.queue, "commandBuffer", .{});
        return .{ .pool = pool, .cmd = cmd, .encoder = send(id, cmd, "computeCommandEncoder", .{}) };
    }

    /// A command buffer with one compute encoder whose dispatches may overlap, held in `rec` until `Pass.replay`
    /// encodes them by level.
    pub fn beginConcurrent(dev: *const Device, rec: *Recorder) Pass {
        const pool = objc_autoreleasePoolPush();
        const cmd = send(id, dev.queue, "commandBuffer", .{});
        rec.clear();
        return .{ .pool = pool, .cmd = cmd, .encoder = send(id, cmd, "computeCommandEncoderWithDispatchType:", .{@as(usize, 1)}), .rec = rec };
    }
};

pub const Library = struct {
    raw: id,

    pub fn release(l: Library) void {
        send(void, l.raw, "release", .{});
    }

    /// Names of every kernel function compiled into `l`, owned by the caller.
    pub fn functionNames(l: Library, a: std.mem.Allocator) error{OutOfMemory}![][]u8 {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        const arr = send(id, l.raw, "functionNames", .{});
        const count = send(usize, arr, "count", .{});
        const out = try a.alloc([]u8, count);
        var made: usize = 0;
        errdefer {
            for (out[0..made]) |s| a.free(s);
            a.free(out);
        }
        for (0..count) |i| {
            const ns = send(id, arr, "objectAtIndex:", .{i});
            const ptr = send([*:0]const u8, ns, "UTF8String", .{});
            out[i] = try a.dupe(u8, std.mem.span(ptr));
            made += 1;
        }
        return out;
    }
};

/// Orders two passes whose buffers Metal does not track: one updates it, a later one waits for it.
pub const Fence = struct {
    raw: id,

    pub fn release(f: Fence) void {
        send(void, f.raw, "release", .{});
    }
};

pub const Pipeline = struct {
    raw: id,
    /// The most threads a threadgroup of this pipeline may hold.
    max_threads: usize,

    pub fn release(p: Pipeline) void {
        send(void, p.raw, "release", .{});
    }
};

pub const Buffer = struct {
    raw: id,
    /// The residency set it is in; none for a buffer someone else made (an MLX array's).
    set: u8 = no_set,

    pub fn bytes(b: Buffer) []u8 {
        return send([*]u8, b.raw, "contents", .{})[0..send(usize, b.raw, "length", .{})];
    }

    /// The bytes Metal holds for it: its length rounded to the device's pages.
    pub fn allocatedSize(b: Buffer) usize {
        return send(usize, b.raw, "allocatedSize", .{});
    }

    /// The owner's release: the buffer leaves its residency set. The inference thread only.
    pub fn release(b: Buffer) void {
        if (b.set != no_set) {
            const set = &sets[b.set];
            send(void, set.raw, "removeAllocation:", .{b.raw});
            set.bytes -= send(usize, b.raw, "length", .{});
            set.dirty = true;
            any_dirty = true;
        }
        send(void, b.raw, "release", .{});
    }

    /// One more holder of the buffer, who lets go with `unretain`.
    pub fn retain(b: Buffer) Buffer {
        _ = send(id, b.raw, "retain", .{});
        return b;
    }

    pub fn unretain(b: Buffer) void {
        send(void, b.raw, "release", .{});
    }
};

/// A buffer and the byte in it where a tensor's data starts.
pub const Bound = struct { buffer: Buffer, offset: usize };

/// The pipeline the latest `Pass.pipeline` set, whichever pass: the one thread that encodes asks it at a dispatch.
pub var bound_pipeline: id = null;

/// A concurrent pass's dispatches, held instead of encoded: each keeps the pipeline, buffers and bytes bound when it
/// was made and its level in the pass's dependency order (`next_level`, set by the engine before it), so `replay`
/// encodes them level by level with one barrier between levels.
pub const Recorder = struct {
    const slots = 32;
    const Bind = struct { index: u8, raw: id, offset: usize };
    const Bytes = struct { index: u8, at: u32, len: u32 };
    const Cmd = struct { pipe: id, binds: [2]u32, bytes: [2]u32, grid: Size, group: Size, groups: bool, level: u32 };

    pipe: id = null,
    buf: [slots]Bind = undefined,
    bytes_at: [slots]Bytes = undefined,
    buf_set: u32 = 0,
    bytes_set: u32 = 0,
    /// The level `dispatch` gives the next dispatch, set by the engine before it.
    next_level: u32 = 0,
    cmds: std.ArrayList(Cmd) = .empty,
    binds: std.ArrayList(Bind) = .empty,
    byte_recs: std.ArrayList(Bytes) = .empty,
    arena: std.ArrayList(u8) = .empty,
    order: std.ArrayList(u32) = .empty,
    counts: std.ArrayList(u32) = .empty,

    fn setBuffer(r: *Recorder, raw: id, offset: usize, index: usize) void {
        r.buf[index] = .{ .index = @intCast(index), .raw = raw, .offset = offset };
        r.buf_set |= @as(u32, 1) << @intCast(index);
        r.bytes_set &= ~(@as(u32, 1) << @intCast(index));
    }

    fn setBytes(r: *Recorder, data: []const u8, index: usize) void {
        const at: u32 = @intCast(r.arena.items.len);
        r.arena.appendSlice(std.heap.c_allocator, data) catch @panic("engine: out of memory recording a pass");
        r.bytes_at[index] = .{ .index = @intCast(index), .at = at, .len = @intCast(data.len) };
        r.bytes_set |= @as(u32, 1) << @intCast(index);
        r.buf_set &= ~(@as(u32, 1) << @intCast(index));
    }

    fn record(r: *Recorder, grid: Size, group: Size, groups: bool) void {
        const a = std.heap.c_allocator;
        const b0: u32 = @intCast(r.binds.items.len);
        const y0: u32 = @intCast(r.byte_recs.items.len);
        var m = r.buf_set;
        while (m != 0) : (m &= m - 1) r.binds.append(a, r.buf[@ctz(m)]) catch @panic("engine: out of memory recording a pass");
        m = r.bytes_set;
        while (m != 0) : (m &= m - 1) r.byte_recs.append(a, r.bytes_at[@ctz(m)]) catch @panic("engine: out of memory recording a pass");
        r.cmds.append(a, .{ .pipe = r.pipe, .binds = .{ b0, @intCast(r.binds.items.len) }, .bytes = .{ y0, @intCast(r.byte_recs.items.len) }, .grid = grid, .group = group, .groups = groups, .level = r.next_level }) catch
            @panic("engine: out of memory recording a pass");
    }

    /// Drops what is held: the pass's encoder starts with nothing bound.
    pub fn clear(r: *Recorder) void {
        r.pipe = null;
        r.buf_set = 0;
        r.bytes_set = 0;
        r.cmds.clearRetainingCapacity();
        r.binds.clearRetainingCapacity();
        r.byte_recs.clearRetainingCapacity();
        r.arena.clearRetainingCapacity();
    }

    /// Encodes the held dispatches into `p` in the order of their levels (the order made within one), a barrier
    /// between levels; returns the barriers.
    pub fn replay(r: *Recorder, p: Pass) u64 {
        const a = std.heap.c_allocator;
        // A counting sort by level: stable, so a level keeps the order its dispatches were made in.
        var top: u32 = 0;
        for (r.cmds.items) |c| top = @max(top, c.level);
        r.counts.clearRetainingCapacity();
        r.counts.appendNTimes(a, 0, top + 2) catch @panic("engine: out of memory recording a pass");
        for (r.cmds.items) |c| r.counts.items[c.level + 1] += 1;
        for (1..r.counts.items.len) |i| r.counts.items[i] += r.counts.items[i - 1];
        r.order.resize(a, r.cmds.items.len) catch @panic("engine: out of memory recording a pass");
        for (r.cmds.items, 0..) |c, i| {
            r.order.items[r.counts.items[c.level]] = @intCast(i);
            r.counts.items[c.level] += 1;
        }
        var barriers: u64 = 0;
        var pipe: id = null;
        var level: ?u32 = null;
        for (r.order.items) |i| {
            const c = r.cmds.items[i];
            if (level) |l| if (l != c.level) {
                send(void, p.encoder, "memoryBarrierWithScope:", .{@as(usize, 1)});
                barriers += 1;
            };
            level = c.level;
            if (c.pipe != pipe) send(void, p.encoder, "setComputePipelineState:", .{c.pipe});
            pipe = c.pipe;
            for (r.binds.items[c.binds[0]..c.binds[1]]) |b| send(void, p.encoder, "setBuffer:offset:atIndex:", .{ b.raw, b.offset, @as(usize, b.index) });
            for (r.byte_recs.items[c.bytes[0]..c.bytes[1]]) |y| send(void, p.encoder, "setBytes:length:atIndex:", .{ @as(*const anyopaque, r.arena.items[y.at..].ptr), @as(usize, y.len), @as(usize, y.index) });
            if (c.groups)
                send(void, p.encoder, "dispatchThreadgroups:threadsPerThreadgroup:", .{ c.grid, c.group })
            else
                send(void, p.encoder, "dispatchThreads:threadsPerThreadgroup:", .{ c.grid, c.group });
        }
        r.clear();
        return barriers;
    }
};

pub const Pass = struct {
    pool: ?*anyopaque,
    cmd: id,
    encoder: id,
    /// Set on a concurrent pass: its dispatches are held until `replay`.
    rec: ?*Recorder = null,

    pub fn pipeline(p: Pass, state: Pipeline) void {
        bound_pipeline = state.raw;
        if (p.rec) |r| {
            r.pipe = state.raw;
            return;
        }
        send(void, p.encoder, "setComputePipelineState:", .{state.raw});
    }

    pub fn buffer(p: Pass, b: Buffer, offset: usize, index: usize) void {
        if (p.rec) |r| return r.setBuffer(b.raw, offset, index);
        send(void, p.encoder, "setBuffer:offset:atIndex:", .{ b.raw, offset, index });
    }

    /// A small constant copied into the command buffer.
    pub fn constant(p: Pass, value: anytype, index: usize) void {
        const v = value;
        if (p.rec) |r| return r.setBytes(std.mem.asBytes(&v), index);
        send(void, p.encoder, "setBytes:length:atIndex:", .{ @as(*const anyopaque, &v), @as(usize, @sizeOf(@TypeOf(v))), index });
    }

    /// A few constant bytes copied into the command buffer.
    pub fn bytes(p: Pass, data: []const u8, index: usize) void {
        if (p.rec) |r| return r.setBytes(data, index);
        send(void, p.encoder, "setBytes:length:atIndex:", .{ @as(*const anyopaque, data.ptr), data.len, index });
    }

    pub fn dispatch(p: Pass, grid: Size, group: Size) void {
        if (p.rec) |r| return r.record(grid, group, false);
        send(void, p.encoder, "dispatchThreads:threadsPerThreadgroup:", .{ grid, group });
    }

    /// The pass does not start before the pass that updates `fence` has run.
    pub fn waitFence(p: Pass, fence: Fence) void {
        send(void, p.encoder, "waitForFence:", .{fence.raw});
    }

    /// `fence` is reached when this pass has run.
    pub fn updateFence(p: Pass, fence: Fence) void {
        send(void, p.encoder, "updateFence:", .{fence.raw});
    }

    /// Encodes the held dispatches by level (`Recorder.replay`); returns the barriers put between levels.
    pub fn replay(p: Pass) u64 {
        const r = p.rec orelse return 0;
        return r.replay(p);
    }

    /// `grid` counts threadgroups, not threads.
    pub fn dispatchGroups(p: Pass, grid: Size, group: Size) void {
        if (p.rec) |r| return r.record(grid, group, true);
        send(void, p.encoder, "dispatchThreadgroups:threadsPerThreadgroup:", .{ grid, group });
    }

    /// Ends the encoding and drops the command buffer without running it.
    pub fn discard(p: Pass) void {
        if (p.rec) |r| r.clear();
        send(void, p.encoder, "endEncoding", .{});
        objc_autoreleasePoolPop(p.pool);
    }

    /// Hands the command buffer to the GPU. Every buffer it binds lives until the pass has run (`Committed.wait`,
    /// `waitDone`). No shared event is signaled: one costs about 8 us of CPU per command buffer on Metal's threads.
    pub fn commit(p: Pass) Committed {
        send(void, p.encoder, "endEncoding", .{});
        commitSets();
        _ = send(id, p.cmd, "retain", .{});
        send(void, p.cmd, "commit", .{});
        objc_autoreleasePoolPop(p.pool);
        return .{ .cmd = p.cmd };
    }
};

pub const Committed = struct {
    cmd: id,

    /// Whether the command buffer failed; Metal's text is then in `lastError`.
    pub fn failed(c: Committed) bool {
        const errored = 5;
        if (send(usize, c.cmd, "status", .{}) != errored) return false;
        noteError(send(id, c.cmd, "error", .{}), "the command buffer failed");
        return true;
    }

    /// Whether the command buffer has run, or failed.
    pub fn ended(c: Committed) bool {
        const completed = 4;
        return send(usize, c.cmd, "status", .{}) >= completed;
    }

    /// Blocks until the command buffer has run, or failed; the caller still holds it.
    pub fn waitDone(c: Committed) void {
        send(void, c.cmd, "waitUntilCompleted", .{});
    }

    pub fn release(c: Committed) void {
        send(void, c.cmd, "release", .{});
    }

    /// When the GPU began and ended the command buffer, seconds on Metal's clock; zeros until it has completed.
    pub fn gpuSpan(c: Committed) [2]f64 {
        return .{ send(f64, c.cmd, "GPUStartTime", .{}), send(f64, c.cmd, "GPUEndTime", .{}) };
    }

    /// Blocks until the GPU has run the command buffer; its failure is this call's error, with Metal's text in
    /// `lastError`.
    pub fn wait(c: Committed) Error!void {
        defer send(void, c.cmd, "release", .{});
        send(void, c.cmd, "waitUntilCompleted", .{});
        const completed = 4;
        if (send(usize, c.cmd, "status", .{}) == completed) return;
        noteError(send(id, c.cmd, "error", .{}), "the command buffer failed");
        return error.MetalCommand;
    }
};

const test_scale_fn = "sushi_k_sushi_test_scale_f32__f32_f32s__f32";

fn loadTestLibrary(dev: *const Device) !Library {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = @import("metallib_path.zig").forThisProcess(&buf) catch return error.MetalUnavailable;
    return dev.load(path);
}

fn loadTestScale(dev: *const Device) !Pipeline {
    const lib = try loadTestLibrary(dev);
    return dev.pipelineOf(lib, test_scale_fn);
}

test "buffers are spread over residency sets of a capped size, a buffer over the cap takes a set of its own, and a release gives the bytes back" {
    const dev = device() catch return error.SkipZigTest;
    const page = std.heap.pageSize();
    const was = setResidentCapForTest(3 * page);
    defer _ = setResidentCapForTest(was);
    const before = residentSets();
    var held: [5]Buffer = undefined;
    for (held[0..4]) |*b| b.* = try dev.alloc(page);
    held[4] = try dev.alloc(8 * page);
    const filled = residentSets();
    try std.testing.expectEqual(before.bytes + 12 * page, filled.bytes);
    // Whatever the sets held before, none takes more than three of the four pages, and the buffer over the cap is
    // alone in its set.
    var beside_first: usize = 0;
    for (held[1..4]) |b| beside_first += @intFromBool(b.set == held[0].set);
    try std.testing.expect(beside_first < 3);
    try std.testing.expectEqual(8 * page, sets[held[4].set].bytes);
    for (held) |b| b.release();
    try std.testing.expectEqual(before.bytes, residentSets().bytes);
    // An emptied set is used again before another is made.
    const again = try dev.alloc(8 * page);
    defer again.release();
    try std.testing.expectEqual(filled.count, residentSets().count);
}

test "past the most sets a queue takes, a buffer joins the emptiest set, and every release is still counted" {
    const dev = device() catch return error.SkipZigTest;
    const page = std.heap.pageSize();
    const was = setResidentCapForTest(page);
    defer _ = setResidentCapForTest(was);
    const before = residentSets();
    var held: [max_sets + 8]Buffer = undefined;
    for (&held) |*b| b.* = try dev.alloc(page);
    try std.testing.expectEqual(@as(usize, max_sets), residentSets().count);
    try std.testing.expectEqual(before.bytes + held.len * page, residentSets().bytes);
    for (held) |b| b.release();
    try std.testing.expectEqual(before.bytes, residentSets().bytes);
    commitResident();
    try std.testing.expect(!residentDirty());
}

test "a wrapped mapping never shares a residency set with a buffer of the pool" {
    const dev = device() catch return error.SkipZigTest;
    const page = std.heap.pageSize();
    const mapped = std.c.mmap(null, page, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    try std.testing.expect(mapped != std.c.MAP_FAILED);
    defer _ = std.c.munmap(@alignCast(mapped), page);
    const scratch = try dev.alloc(page);
    defer scratch.release();
    const weight = try dev.wrap(@ptrCast(mapped), page);
    defer weight.release();
    const later = try dev.alloc(page);
    defer later.release();
    try std.testing.expect(weight.set != scratch.set);
    try std.testing.expect(weight.set != later.set);
}

test "a buffer over mapped memory is the memory itself, and a kernel reads it" {
    const dev = device() catch return error.SkipZigTest;
    const page = std.heap.pageSize();
    const mapped = std.c.mmap(null, page, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    try std.testing.expect(mapped != std.c.MAP_FAILED);
    defer _ = std.c.munmap(@alignCast(mapped), page);
    const ptr: [*]u8 = @ptrCast(mapped);
    const values: []f32 = @alignCast(std.mem.bytesAsSlice(f32, ptr[0 .. 16 * 4]));
    for (values, 0..) |*v, k| v.* = @floatFromInt(k);
    const dst = try dev.alloc(16 * 4);
    defer dst.release();
    const before = residentSets().bytes;
    const src = try dev.wrap(ptr, 16 * 4);
    try std.testing.expectEqual(@intFromPtr(ptr), @intFromPtr(src.bytes().ptr));
    try std.testing.expectEqual(before + 16 * 4, residentSets().bytes);
    const state = try loadTestScale(dev);
    defer state.release();
    const pass = dev.begin();
    pass.pipeline(state);
    pass.buffer(src, 0, 0);
    pass.constant(@as(f32, 2), 1);
    pass.buffer(dst, 0, 2);
    pass.dispatch(.{ .x = 16 }, .{ .x = 16 });
    try pass.commit().wait();
    const got: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, dst.bytes()[0 .. 16 * 4]));
    for (got, 0..) |v, k| try std.testing.expectEqual(@as(f32, @floatFromInt(2 * k)), v);
    src.release();
    try std.testing.expectEqual(before, residentSets().bytes);
}

test "a precompiled kernel dispatched from Zig reads one buffer and writes another" {
    const t = std.testing;
    const dev = device() catch return error.SkipZigTest;
    const state = try loadTestScale(dev);
    defer state.release();
    try t.expect(state.max_threads >= 64);

    const n = 1000;
    const x = try dev.alloc(n * 4);
    defer x.release();
    const out = try dev.alloc(n * 4);
    defer out.release();
    const xs: []f32 = @alignCast(std.mem.bytesAsSlice(f32, x.bytes()[0 .. n * 4]));
    for (xs, 0..) |*v, i| v.* = @floatFromInt(i);

    const pass = dev.begin();
    pass.pipeline(state);
    pass.buffer(x, 0, 0);
    pass.constant(@as(f32, 1.5), 1);
    pass.buffer(out, 0, 2);
    pass.dispatch(.{ .x = n }, .{ .x = 64 });
    try pass.commit().wait();

    const got: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, out.bytes()[0 .. n * 4]));
    for (got, 0..) |v, i| try t.expectEqual(@as(f32, @floatFromInt(i)) * 1.5, v);
}

test "a dispatch reads what the dispatch before it wrote" {
    const t = std.testing;
    const dev = device() catch return error.SkipZigTest;
    const state = try loadTestScale(dev);
    defer state.release();
    const n = 4096;
    const a = try dev.alloc(n * 4);
    defer a.release();
    const b = try dev.alloc(n * 4);
    defer b.release();
    const as: []f32 = @alignCast(std.mem.bytesAsSlice(f32, a.bytes()[0 .. n * 4]));
    for (as, 0..) |*v, i| v.* = @floatFromInt(i % 97);

    // a -> b, b -> a, a -> b in one command buffer: 2 * 3 * 5 times the start.
    const pass = dev.begin();
    pass.pipeline(state);
    for ([_]f32{ 2, 3, 5 }, 0..) |scale, step| {
        pass.buffer(if (step % 2 == 0) a else b, 0, 0);
        pass.constant(scale, 1);
        pass.buffer(if (step % 2 == 0) b else a, 0, 2);
        pass.dispatch(.{ .x = n }, .{ .x = 256 });
    }
    try pass.commit().wait();
    const got: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, b.bytes()[0 .. n * 4]));
    for (got, 0..) |v, i| try t.expectEqual(@as(f32, @floatFromInt(i % 97)) * 30, v);
}

test "a missing metallib function is an error that names the function" {
    const t = std.testing;
    const dev = device() catch return error.SkipZigTest;
    const lib = try loadTestLibrary(dev);
    try t.expectError(error.MetalCompile, dev.pipelineOf(lib, "no_such_function"));
    try t.expect(std.mem.indexOf(u8, lastError(), "no_such_function") != null);
}

test "a binary archive serializes compiled GPU pipelines and serves them with fail_on_miss" {
    const t = std.testing;
    const dev = device() catch return error.SkipZigTest;
    const lib = try loadTestLibrary(dev);
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_len = try tmp.dir.realPath(t.io, &dir_buf);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const archive_path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/archive.bin", .{dir_buf[0..dir_len]}, 0);

    const flags = [_]Flag{ .{ .index = 1, .value = true }, .{ .index = 2, .value = false }, .{ .index = 3, .value = false } };
    const writer = try dev.newBinaryArchive(null);
    defer writer.release();
    try dev.addComputeToArchive(writer, lib, "rms_loopedbfloat16", &.{}, &.{});
    try dev.addComputeToArchive(writer, lib, "rope_float32", &flags, &.{});
    try writer.serialize(archive_path);

    const reader = try dev.newBinaryArchive(archive_path);
    defer reader.release();
    const p1 = try dev.pipelineFromArchive(reader, lib, "rms_loopedbfloat16", &.{}, &.{}, true);
    defer p1.release();
    try t.expect(p1.max_threads > 0);
    const p2 = try dev.pipelineFromArchive(reader, lib, "rope_float32", &flags, &.{}, true);
    defer p2.release();
    try t.expect(p2.max_threads > 0);
    // A function not added to the archive fails when fail_on_miss is enforced.
    try t.expectError(error.MetalCompile, dev.pipelineFromArchive(reader, lib, "rms_loopedfloat16", &.{}, &.{}, true));
}
